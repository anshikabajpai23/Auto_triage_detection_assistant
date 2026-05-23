"""
grpo_trainer.py
---------------
Stage 2 (alternative to DPO): Group Relative Policy Optimization (GRPO).

Why GRPO over DPO:
  - No preference pairs needed — uses a rule-based verifiable reward function
  - Generates num_generations outputs per prompt and ranks them within the group
  - Directly optimizes severity accuracy + team routing, not proxy preference signal
  - Works best when reward is deterministic and checkable — which this task is

What this does:
  - Loads SFT checkpoint as starting policy
  - For each prompt, generates num_generations=8 completions
  - Scores each with: severity correctness + team correctness + parse penalty
  - Trains using group-relative advantage (GRPO loss)
  - Logs reward mean/std and KL to W&B
  - Saves best checkpoint to checkpoints/grpo/

Run on BigRed200:
    sbatch scripts/slurm/run_grpo.sh

Run locally:
    python -m src.models.grpo_trainer --config configs/grpo.yaml
"""

import argparse
import os
import re

import pandas as pd
import torch
import yaml
from datasets import Dataset, DatasetDict
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT          = os.path.join(os.path.dirname(__file__), "../..")
PROCESSED_DIR = os.path.join(ROOT, "data/processed")

# ── prompt template (must match SFT exactly) ──────────────────────────────────
PROMPT_TEMPLATE = """\
### Incident report:
{title}
{body}
### Triage (always output severity P0-P4 and one team only):
"""

# ── output parser ─────────────────────────────────────────────────────────────
_OUTPUT_RE = re.compile(
    r"severity\s*:\s*(P[0-4])\s*\|\s*team\s*:\s*(\w+)",
    re.IGNORECASE,
)
SEVERITY_ORDER = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4}


def parse_output(text: str):
    match = _OUTPUT_RE.search(text)
    if not match:
        return None, None
    return match.group(1).upper(), match.group(2).lower()


# ── reward function ───────────────────────────────────────────────────────────
# GRPOTrainer calls: reward_fn(completions, **kwargs)
# Extra dataset columns (true_severity, true_team) are passed as kwargs automatically

def reward_fn(completions: list[str], true_severity: list[str], true_team: list[str], **kwargs) -> list[float]:
    """
    Rule-based reward for each generated completion.

    Scoring:
      +1.0  correct severity
      +0.2  severity off by 1 level
      -0.5  severity off by 2+ levels
      -1.0  extra penalty: true=P0/P1 but predicted P3/P4 (critical miss)
      +0.5  correct team
      -0.5  parse failure (malformed output)
    """
    rewards = []
    # TRL 0.15 may pass completions as dicts — unwrap to plain string
    # TRL 0.15 may pass completions as dicts — unwrap to plain string
    completions = [
        (c[-1])["content"] if isinstance(c, list) and isinstance(c[-1], dict) else str(c)
        for c in completions
    ]
    for completion, true_sev, true_tm in zip(completions, true_severity, true_team):
        pred_sev, pred_team = parse_output(completion)

        if pred_sev is None:
            rewards.append(-0.5)
            continue

        score = 0.0

        # ── severity ──────────────────────────────────────────────────────────
        if pred_sev == true_sev:
            score += 1.0
        else:
            dist = abs(SEVERITY_ORDER[pred_sev] - SEVERITY_ORDER[true_sev])
            if dist == 1:
                score += 0.2
            else:
                score -= 0.5
            # extra penalty for critical misses
            if true_sev in ("P0", "P1") and pred_sev in ("P3", "P4"):
                score -= 1.0

        # ── team ──────────────────────────────────────────────────────────────
        if pred_team == true_tm:
            score += 0.5

        rewards.append(score)

    return rewards


# ── config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── data ──────────────────────────────────────────────────────────────────────

def load_splits(processed_dir: str) -> DatasetDict:
    splits = {}
    for split in ("train", "val"):
        path = os.path.join(processed_dir, f"{split}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run src/data/load_datasets.py first."
            )
        df = pd.read_parquet(path)
        splits[split] = Dataset.from_pandas(df, preserve_index=False)
    print(f"Train: {len(splits['train']):,} rows")
    print(f"Val:   {len(splits['val']):,} rows")
    return DatasetDict(splits)


def build_grpo_dataset(dataset: DatasetDict, tokenizer, max_seq_length: int = 1024) -> DatasetDict:
    """
    Convert parquet splits into GRPOTrainer format.
    Required columns: prompt, true_severity, true_team
    Extra columns (true_severity, true_team) are passed as kwargs to reward_fn.
    """
    TEMPLATE_OVERHEAD  = 40
    COMPLETION_OVERHEAD = 12

    def process(batch):
        prompts, severities, teams = [], [], []
        for title, body, priority, team in zip(
            batch["title"], batch["body"], batch["priority"], batch["team"]
        ):
            title = title.strip() if title else ""
            body  = body.strip() if isinstance(body, str) and body else ""

            # truncate body so triage line always fits
            title_tokens   = len(tokenizer.encode(title, add_special_tokens=False))
            max_body_tokens = max_seq_length - title_tokens - TEMPLATE_OVERHEAD - COMPLETION_OVERHEAD
            if body:
                body_ids = tokenizer.encode(body, add_special_tokens=False)
                if len(body_ids) > max_body_tokens:
                    body = tokenizer.decode(body_ids[:max_body_tokens], skip_special_tokens=True)

            prompt = PROMPT_TEMPLATE.format(
                title=title,
                body=("\n" + body) if body else "",
            )
            prompts.append(prompt)
            severities.append(priority)
            teams.append(team)

        return {"prompt": prompts, "true_severity": severities, "true_team": teams}

    return DatasetDict({
        split: dataset[split].map(
            process,
            batched=True,
            remove_columns=dataset[split].column_names,
        )
        for split in dataset
    })


# ── model ─────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict):
    model_name     = cfg["model_name"]
    sft_checkpoint = os.path.join(ROOT, cfg.get("sft_checkpoint", "checkpoints/sft_weighted_oversampling/checkpoint-18504"))
    hf_token       = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint, token=hf_token, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # required for generation in GRPO

    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type       = cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = cfg.get("bnb_4bit_use_double_quant", True),
    )

    print(f"Loading base model: {model_name} in 4-bit NF4 …")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # BigRed200 sets CUDA_VISIBLE_DEVICES per process —
    # each process sees only its assigned GPU as cuda:0
    n_visible = torch.cuda.device_count()
    device_id  = local_rank if n_visible > 1 else 0
    torch.cuda.set_device(device_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        device_map          = {"": device_id},
        token               = hf_token,
        dtype               = torch.bfloat16,
    )
    base_model.config.use_cache      = False
    base_model.config.pretraining_tp = 1

    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing = cfg.get("gradient_checkpointing", True),
    )

    print(f"Loading SFT LoRA adapters from: {sft_checkpoint}")
    model = PeftModel.from_pretrained(base_model, sft_checkpoint, is_trainable=True)
    model.print_trainable_parameters()

    return model, tokenizer


# ── training ──────────────────────────────────────────────────────────────────

def build_grpo_config(cfg: dict) -> GRPOConfig:
    gck_kwargs = cfg.get("gradient_checkpointing_kwargs", {"use_reentrant": False})

    return GRPOConfig(
        output_dir                    = cfg.get("output_dir", "checkpoints/grpo"),
        # GRPO-specific
        num_generations               = cfg.get("num_generations", 8),       # outputs per prompt
        max_prompt_length             = cfg.get("max_prompt_length", 900),   # fits in 1024 with completion
        max_completion_length         = cfg.get("max_completion_length", 20),# "severity:P1 | team:backend\n" ~= 10 tokens
        beta                          = cfg.get("beta", 0.04),               # KL penalty — lower than DPO
        # Training
        num_train_epochs              = cfg.get("num_train_epochs", 2),
        per_device_train_batch_size   = cfg.get("per_device_train_batch_size", 2),   # 2 prompts × 8 generations = 16 rollouts/step
        per_device_eval_batch_size    = cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps   = cfg.get("gradient_accumulation_steps", 8),   # effective batch = 16 prompts
        learning_rate                 = cfg.get("learning_rate", 5e-6),
        lr_scheduler_type             = cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio                  = cfg.get("warmup_ratio", 0.05),
        weight_decay                  = cfg.get("weight_decay", 0.01),
        max_grad_norm                 = cfg.get("max_grad_norm", 1.0),
        optim                         = cfg.get("optim", "paged_adamw_32bit"),
        bf16                          = cfg.get("bf16", True),
        tf32                          = cfg.get("tf32", True),
        gradient_checkpointing        = cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs = gck_kwargs,
        logging_steps                 = cfg.get("logging_steps", 10),
        eval_strategy                 = cfg.get("eval_strategy", "epoch"),
        save_strategy                 = cfg.get("save_strategy", "epoch"),
        save_total_limit              = cfg.get("save_total_limit", 2),
        load_best_model_at_end        = cfg.get("load_best_model_at_end", True),
        report_to                     = cfg.get("report_to", "wandb"),
        run_name                      = cfg.get("run_name", "grpo-llama-3.2-1b"),
        dataloader_num_workers        = cfg.get("dataloader_num_workers", 4),
        dataloader_pin_memory         = cfg.get("dataloader_pin_memory", True),
        remove_unused_columns         = False,
    )


def train(cfg: dict):
    # ── load data ─────────────────────────────────────────────────────────────
    raw_dataset = load_splits(PROCESSED_DIR)

    # ── load model + tokeniser ────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg)

    # ── build GRPO dataset ────────────────────────────────────────────────────
    max_seq_length = cfg.get("max_seq_length", 1024)
    print("\nBuilding GRPO dataset (formatting prompts + truncating bodies) …")
    grpo_dataset = build_grpo_dataset(raw_dataset, tokenizer, max_seq_length)
    print(f"  Train: {len(grpo_dataset['train']):,} prompts")
    print(f"  Val:   {len(grpo_dataset['val']):,} prompts")

    # ── print sample to verify format ─────────────────────────────────────────
    sample = grpo_dataset["train"][0]
    print(f"\n{'='*60}\nSAMPLE PROMPT:\n{sample['prompt']}")
    print(f"true_severity={sample['true_severity']} | true_team={sample['true_team']}\n{'='*60}\n")

    # ── verify reward function ────────────────────────────────────────────────
    test_rewards = reward_fn(
        completions   = [f"severity:{sample['true_severity']} | team:{sample['true_team']}",
                         "severity:P3 | team:frontend",
                         "garbage output"],
        true_severity = [sample['true_severity']] * 3,
        true_team     = [sample['true_team']] * 3,
    )
    print(f"Reward sanity check (correct / wrong / parse_fail): {test_rewards}")

    # ── GRPO config ───────────────────────────────────────────────────────────
    grpo_config = build_grpo_config(cfg)

    output_dir = cfg.get("output_dir", "checkpoints/grpo")
    existing   = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")] if os.path.exists(output_dir) else []
    print(f"Output dir: {output_dir} | Existing checkpoints: {existing if existing else 'none — fresh start'}")

    latest_checkpoint = None
    if existing:
        latest_checkpoint = os.path.join(
            output_dir,
            sorted(existing, key=lambda x: int(x.split("-")[-1]))[-1]
        )
        print(f"Resuming from: {latest_checkpoint}")

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model            = model,
        reward_funcs     = reward_fn,
        args             = grpo_config,
        train_dataset    = grpo_dataset["train"],
        eval_dataset     = grpo_dataset["val"],
        processing_class = tokenizer,
    )

    print(f"\nStarting GRPO training (num_generations={cfg.get('num_generations', 8)}, beta={cfg.get('beta', 0.04)}) …")
    print("W&B will log: reward_mean, reward_std, kl, policy_loss, completion_length")
    trainer.train(resume_from_checkpoint=latest_checkpoint)

    # ── save ──────────────────────────────────────────────────────────────────
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nGRPO model saved to {output_dir}")
    print("Next: python -m src.eval.metrics --checkpoint checkpoints/grpo --split val")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo.yaml",
                        help="Path to YAML config file.")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
