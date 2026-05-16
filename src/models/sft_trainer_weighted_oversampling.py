"""
sft_trainer_weighted_oversampling.py
-------------------------------------
Same as sft_trainer_weighted_loss.py but adds mild oversampling of minority
severity classes before training to reduce P3-dominance.

Oversampling strategy (mild):
  - Compute majority class count (P3 ~58%)
  - Target each minority class to reach at most 30% of majority count
  - Resample with replacement — never reduces any class, only adds copies
  - P3/P2 stay untouched; P0/P1/P4 get a modest boost

Run on BigRed200:
    sbatch scripts/slurm/run_sft_oversampling.sh

Run locally:
    python -m src.models.sft_trainer_weighted_oversampling --config configs/sft_llama_oversampling.yaml
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from trl import DataCollatorForCompletionOnlyLM
from torch.nn import CrossEntropyLoss

# ── class weights ─────────────────────────────────────────────────────────────
SEVERITY_TOKEN_WEIGHTS = {
    "P0": 2.0,
    "P1": 1.5,
    "P2": 1.2,
    "P3": 1.0,
    "P4": 1.5,
}

# ── oversampling config ────────────────────────────────────────────────────────
# Each minority class is oversampled until it reaches this fraction of the
# majority class size. 0.30 = mild (minority classes reach ~30% of P3 count).
OVERSAMPLE_TARGET_RATIO = 0.30


class WeightedSFTTrainer(SFTTrainer):
    """SFTTrainer with per-severity class weighting on the completion tokens."""

    def __init__(self, *args, severity_token_ids: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.severity_token_ids = severity_token_ids or {}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        weights = torch.ones(logits.shape[0], logits.shape[1], device=logits.device)

        for token_id, weight in self.severity_token_ids.items():
            mask = (labels == token_id)
            weights[mask] = weight

        shift_logits  = logits[..., :-1, :].contiguous()
        shift_labels  = labels[..., 1:].contiguous()
        shift_weights = weights[..., :-1].contiguous()

        loss_fct = CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        loss = loss * shift_weights.view(-1)
        non_masked = shift_labels.view(-1) != -100
        loss = loss[non_masked].mean()

        return (loss, outputs) if return_outputs else loss


# ── paths ─────────────────────────────────────────────────────────────────────
ROOT          = os.path.join(os.path.dirname(__file__), "../..")
PROCESSED_DIR = os.path.join(ROOT, "data/processed")
FALLBACK_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

PROMPT_TEMPLATE = """\
### Incident report:
{title}
{body}
### Triage (always output severity P0-P4 and one team only):
"""
COMPLETION_TEMPLATE = "severity:{priority} | team:{team}\n"


# ── config loading ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── oversampling ──────────────────────────────────────────────────────────────

def oversample_minority_classes(df: pd.DataFrame, target_ratio: float = OVERSAMPLE_TARGET_RATIO, seed: int = 42) -> pd.DataFrame:
    """
    Mildly oversample minority severity classes.

    Each class is upsampled (with replacement) until it reaches
    target_ratio * majority_class_count. Classes already above the target
    are left untouched. P3 (majority) is never touched.

    Args:
        df:           Training DataFrame with a 'priority' column.
        target_ratio: Minority target as a fraction of majority count (default 0.30).
        seed:         Random seed for reproducibility.

    Returns:
        Shuffled DataFrame with oversampled minority rows appended.
    """
    counts       = df["priority"].value_counts()
    majority_n   = counts.max()
    target_n     = int(majority_n * target_ratio)

    print("\nOversampling minority classes:")
    print(f"  Majority class size : {majority_n:,}")
    print(f"  Target per minority : {target_n:,} ({target_ratio:.0%} of majority)")

    rng    = np.random.default_rng(seed)
    frames = [df]

    for priority, count in counts.items():
        if count >= target_n:
            print(f"  {priority}: {count:,} → no oversampling needed")
            continue
        n_extra  = target_n - count
        minority = df[df["priority"] == priority]
        extra    = minority.sample(n=n_extra, replace=True, random_state=seed)
        frames.append(extra)
        print(f"  {priority}: {count:,} → {count + n_extra:,} (+{n_extra:,} synthetic copies)")

    result = pd.concat(frames, ignore_index=True)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"\n  Total train rows after oversampling: {len(result):,} (was {len(df):,})\n")
    return result


# ── dataset ───────────────────────────────────────────────────────────────────

def load_splits(processed_dir: str, oversample: bool = True) -> DatasetDict:
    """Load train/val parquets. Optionally oversample minority classes in train."""
    splits = {}
    for split in ("train", "val"):
        path = os.path.join(processed_dir, f"{split}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run src/data/load_datasets.py first."
            )
        df = pd.read_parquet(path)

        if split == "train" and oversample:
            df = oversample_minority_classes(df)

        splits[split] = Dataset.from_pandas(df, preserve_index=False)

    print(f"Train: {len(splits['train']):,} rows")
    print(f"Val:   {len(splits['val']):,} rows\n")
    return DatasetDict(splits)


def make_formatting_func(tokenizer, max_seq_length: int = 1024):
    def formatting_func(batch: dict) -> list[str]:
        texts = []
        for title, body, priority, team in zip(
            batch["title"], batch["body"], batch["priority"], batch["team"]
        ):
            body = body.strip() if body else ""
            TEMPLATE_OVERHEAD  = 40
            COMPLETION_OVERHEAD = 12
            title_tokens = len(tokenizer.encode(title.strip(), add_special_tokens=False))
            max_body_tokens = max_seq_length - title_tokens - TEMPLATE_OVERHEAD - COMPLETION_OVERHEAD

            if body:
                body_token_ids = tokenizer.encode(body, add_special_tokens=False)
                if len(body_token_ids) > max_body_tokens:
                    body_token_ids = body_token_ids[:max_body_tokens]
                    body = tokenizer.decode(body_token_ids, skip_special_tokens=True)

            prompt     = PROMPT_TEMPLATE.format(title=title.strip(), body=("\n" + body) if body else "")
            completion = COMPLETION_TEMPLATE.format(priority=priority, team=team)
            texts.append(prompt + completion)
        return texts

    return formatting_func


# ── model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict):
    model_name = cfg["model_name"]
    hf_token   = os.environ.get("HF_TOKEN")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=True)
        print(f"Loaded tokeniser: {model_name}")
    except Exception as e:
        print(f"Could not load tokeniser for {model_name}: {e}")
        print(f"Falling back to {FALLBACK_MODEL}")
        model_name = FALLBACK_MODEL
        tokenizer  = AutoTokenizer.from_pretrained(FALLBACK_MODEL, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type       = cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = cfg.get("bnb_4bit_use_double_quant", True),
    )

    print(f"Loading model: {model_name} in 4-bit NF4 …")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        device_map          = "auto",
        token               = hf_token,
        torch_dtype         = torch.bfloat16,
    )
    model.config.use_cache       = False
    model.config.pretraining_tp  = 1

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing = cfg.get("gradient_checkpointing", True),
    )

    lora_config = LoraConfig(
        r              = cfg.get("lora_r", 16),
        lora_alpha     = cfg.get("lora_alpha", 32),
        lora_dropout   = cfg.get("lora_dropout", 0.05),
        target_modules = cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
        bias           = cfg.get("lora_bias", "none"),
        task_type      = TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ── training ──────────────────────────────────────────────────────────────────

def build_training_args(cfg: dict) -> TrainingArguments:
    gck_kwargs = cfg.get("gradient_checkpointing_kwargs", {"use_reentrant": False})

    return TrainingArguments(
        output_dir                    = cfg.get("output_dir", "checkpoints/sft_weighted_oversample"),
        num_train_epochs              = cfg.get("num_train_epochs", 3),
        per_device_train_batch_size   = cfg.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size    = cfg.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps   = cfg.get("gradient_accumulation_steps", 4),
        learning_rate                 = cfg.get("learning_rate", 2e-4),
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
        save_total_limit              = cfg.get("save_total_limit", 3),
        load_best_model_at_end        = cfg.get("load_best_model_at_end", True),
        metric_for_best_model         = cfg.get("metric_for_best_model", "eval_loss"),
        report_to                     = cfg.get("report_to", "wandb"),
        run_name                      = cfg.get("run_name", "sft-llama-oversampling"),
        dataloader_num_workers        = cfg.get("dataloader_num_workers", 4),
        dataloader_pin_memory         = cfg.get("dataloader_pin_memory", True),
        remove_unused_columns         = True,
    )


def save_training_samples(dataset, output_dir: str, n: int = 100):
    import csv
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "sample_training_inputs.csv")
    rows = dataset["train"].select(range(min(n, len(dataset["train"]))))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input_prompt", "expected_output", "full_string"])
        writer.writeheader()
        for row in rows:
            body       = row["body"].strip() if row["body"] else ""
            prompt     = PROMPT_TEMPLATE.format(title=row["title"].strip(), body=("\n" + body) if body else "")
            completion = COMPLETION_TEMPLATE.format(priority=row["priority"], team=row["team"])
            writer.writerow({"input_prompt": prompt, "expected_output": completion, "full_string": prompt + completion})
    print(f"Saved {n} training samples to {path}")


def train(cfg: dict):
    dataset = load_splits(PROCESSED_DIR, oversample=True)

    output_dir = cfg.get("output_dir", "checkpoints/sft_weighted_oversample")
    existing   = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")] if os.path.exists(output_dir) else []
    print(f"Output dir: {output_dir} | Existing checkpoints: {existing if existing else 'none — fresh start'}")
    save_training_samples(dataset, output_dir)

    model, tokenizer = load_model_and_tokenizer(cfg)
    fmt = make_formatting_func(tokenizer, max_seq_length=cfg.get("max_seq_length", 1024))
    samples = fmt({k: dataset["train"][k][:3] for k in ["title", "body", "priority", "team"]})
    for i, s in enumerate(samples[:3]):
        print(f"\n{'='*60}\nSAMPLE {i}:\n{s}\n{'='*60}")

    training_args = build_training_args(cfg)
    collator = DataCollatorForCompletionOnlyLM(
        response_template="### Triage (always output severity P0-P4 and one team only):\n",
        tokenizer=tokenizer,
    )

    fmt_fn     = make_formatting_func(tokenizer, max_seq_length=cfg.get("max_seq_length", 1024))
    sample_str = fmt_fn({k: [dataset["train"][k][0]] for k in ["title", "body", "priority", "team"]})[0]
    enc        = tokenizer(sample_str, return_tensors="pt")
    batch      = collator([{"input_ids": enc["input_ids"][0].tolist(), "attention_mask": enc["attention_mask"][0].tolist()}])
    n_completion_tokens = (batch["labels"] != -100).sum().item()
    print(f"Completion tokens visible to loss: {n_completion_tokens}")
    if n_completion_tokens == 0:
        raise ValueError("DataCollatorForCompletionOnlyLM found 0 completion tokens. "
                         "The response_template string doesn't match what the tokenizer produces in context. "
                         "Training will produce zero loss and learn nothing.")

    severity_token_ids = {}
    for label, weight in SEVERITY_TOKEN_WEIGHTS.items():
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) >= 2:
            digit_token_id = ids[-1]
            severity_token_ids[digit_token_id] = weight
            print(f"  {label} → digit token_id={digit_token_id} ('{tokenizer.decode([digit_token_id])}'), weight={weight}")
            decoded_digit  = tokenizer.decode([ids[-1]])
            expected_digit = label[-1]
            if decoded_digit.strip() != expected_digit:
                print(f"  WARNING: {label} digit should be '{expected_digit}' but token {ids[-1]} decodes as '{decoded_digit.strip()}' — mismatch!")

    trainer = WeightedSFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = dataset["train"],
        eval_dataset       = dataset["val"],
        formatting_func    = make_formatting_func(tokenizer, max_seq_length=cfg.get("max_seq_length", 1024)),
        max_seq_length     = cfg.get("max_seq_length", 1024),
        packing            = cfg.get("packing", False),
        args               = training_args,
        data_collator      = collator,
        severity_token_ids = severity_token_ids,
    )

    print("\nStarting SFT training with weighted loss + mild oversampling …")
    # trainer.train()
    # Auto-resume from latest checkpoint if one exists in output_dir
    latest_checkpoint = None
    if os.path.exists(output_dir):
        checkpoints = sorted(
            [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1])
        )
        if checkpoints:
            latest_checkpoint = os.path.join(output_dir, checkpoints[-1])
            print(f"Resuming from checkpoint: {latest_checkpoint}")
        else:
            print("No existing checkpoints — fresh start")

    print("\nStarting SFT training with weighted loss + mild oversampling …")
    trainer.train(resume_from_checkpoint=latest_checkpoint)

    output_dir = cfg.get("output_dir", "checkpoints/sft_weighted_oversample")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nLoRA adapters saved to {output_dir}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_llama.yaml", help="Path to YAML config file.")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
