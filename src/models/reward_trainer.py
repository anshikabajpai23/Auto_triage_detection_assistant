"""
reward_trainer.py
------------------
Stage 2: Reward Model training with Bradley-Terry pairwise loss.

What this does:
  - Loads LLaMA 3.2 1B in 4-bit NF4 (QLoRA)
  - Replaces LM head with a scalar linear output (hidden_size → 1)
  - Trains on (prompt, chosen, rejected) preference pairs
  - Uses TRL RewardTrainer with Bradley-Terry loss: log σ(r_chosen - r_rejected)
  - Logs pairwise accuracy to W&B
  - Saves best checkpoint to checkpoints/reward_model/

Run on BigRed200:
    sbatch scripts/slurm/run_reward.sh

Run locally:
    python -m src.models.reward_trainer --config configs/reward_model.yaml
"""

import argparse
import json
import os

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import RewardTrainer, RewardConfig

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "../..")


# ── config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── data ──────────────────────────────────────────────────────────────────────

def load_preference_pairs(path: str) -> Dataset:
    """Load JSONL preference pairs into a HuggingFace Dataset."""
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line.strip()))
    return Dataset.from_list(pairs)


def tokenize_pairs(dataset: Dataset, tokenizer, max_length: int = 512) -> Dataset:
    """
    Tokenize chosen and rejected into input_ids/attention_mask.
    RewardTrainer expects: input_ids_chosen, attention_mask_chosen,
                           input_ids_rejected, attention_mask_rejected
    """
    def tokenize(batch):
        chosen_full   = [p + c for p, c in zip(batch["prompt"], batch["chosen"])]
        rejected_full = [p + r for p, r in zip(batch["prompt"], batch["rejected"])]

        tok_chosen   = tokenizer(chosen_full,   max_length=max_length,
                                 truncation=True, padding="max_length")
        tok_rejected = tokenizer(rejected_full, max_length=max_length,
                                 truncation=True, padding="max_length")
        return {
            "input_ids_chosen"      : tok_chosen["input_ids"],
            "attention_mask_chosen" : tok_chosen["attention_mask"],
            "input_ids_rejected"    : tok_rejected["input_ids"],
            "attention_mask_rejected": tok_rejected["attention_mask"],
        }

    return dataset.map(tokenize, batched=True, remove_columns=["prompt", "chosen", "rejected"])


# ── model ─────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict):
    model_name = cfg["model_name"]
    hf_token   = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type       = cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = cfg.get("bnb_4bit_use_double_quant", True),
    )

    print(f"Loading reward model base: {model_name} in 4-bit NF4 …")
    # AutoModelForSequenceClassification with num_labels=1 adds a scalar head
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels          = 1,
        quantization_config = bnb_config,
        device_map          = "auto",
        token               = hf_token,
        torch_dtype         = torch.bfloat16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache    = False

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
        task_type      = TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ── training ──────────────────────────────────────────────────────────────────

def build_training_args(cfg: dict) -> RewardConfig:
    gck_kwargs = cfg.get("gradient_checkpointing_kwargs", {"use_reentrant": False})

    return RewardConfig(
        output_dir                    = cfg.get("output_dir", "checkpoints/reward_model"),
        num_train_epochs              = cfg.get("num_train_epochs", 2),
        per_device_train_batch_size   = cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size    = cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps   = cfg.get("gradient_accumulation_steps", 4),
        learning_rate                 = cfg.get("learning_rate", 1e-5),
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
        run_name                      = cfg.get("run_name", "reward-model-llama-3.2-1b"),
        dataloader_num_workers        = cfg.get("dataloader_num_workers", 4),
        dataloader_pin_memory         = cfg.get("dataloader_pin_memory", True),
        max_length                    = cfg.get("max_length", 512),
        remove_unused_columns         = False,
    )


def train(cfg: dict):
    # ── load data ─────────────────────────────────────────────────────────────
    train_path = os.path.join(ROOT, cfg.get("train_data", "data/preference_pairs/train.jsonl"))
    val_path   = os.path.join(ROOT, cfg.get("val_data",   "data/preference_pairs/val.jsonl"))

    print(f"Loading preference pairs from:\n  train: {train_path}\n  val:   {val_path}")
    train_ds = load_preference_pairs(train_path)
    val_ds   = load_preference_pairs(val_path)
    print(f"Train pairs: {len(train_ds):,}  |  Val pairs: {len(val_ds):,}\n")

    # ── load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg)

    # ── tokenize ──────────────────────────────────────────────────────────────
    max_length = cfg.get("max_length", 512)
    print(f"Tokenizing pairs (max_length={max_length}) …")
    train_ds = tokenize_pairs(train_ds, tokenizer, max_length)
    val_ds   = tokenize_pairs(val_ds,   tokenizer, max_length)

    # ── training args ─────────────────────────────────────────────────────────
    training_args = build_training_args(cfg)

    output_dir = cfg.get("output_dir", "checkpoints/reward_model")
    existing   = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")] if os.path.exists(output_dir) else []
    print(f"Output dir: {output_dir} | Existing checkpoints: {existing if existing else 'none — fresh start'}")

    # ── auto-resume from latest checkpoint ────────────────────────────────────
    latest_checkpoint = None
    if existing:
        latest_checkpoint = os.path.join(
            output_dir,
            sorted(existing, key=lambda x: int(x.split("-")[-1]))[-1]
        )
        print(f"Resuming from: {latest_checkpoint}")

    # ── trainer ───────────────────────────────────────────────────────────────
    trainer = RewardTrainer(
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        tokenizer     = tokenizer,
    )

    print("\nStarting Reward Model training …")
    trainer.train(resume_from_checkpoint=latest_checkpoint)

    # ── save ──────────────────────────────────────────────────────────────────
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nReward model saved to {output_dir}")

    # ── pairwise accuracy on val ───────────────────────────────────────────────
    print("\nComputing pairwise accuracy on val set …")
    model.eval()
    correct = 0
    with torch.inference_mode():
        for i in range(0, len(val_ds), 8):
            batch = val_ds[i:i+8]
            chosen_ids   = torch.tensor(batch["input_ids_chosen"]).to(model.device)
            chosen_mask  = torch.tensor(batch["attention_mask_chosen"]).to(model.device)
            rejected_ids  = torch.tensor(batch["input_ids_rejected"]).to(model.device)
            rejected_mask = torch.tensor(batch["attention_mask_rejected"]).to(model.device)

            r_chosen   = model(input_ids=chosen_ids,   attention_mask=chosen_mask).logits.squeeze(-1)
            r_rejected = model(input_ids=rejected_ids, attention_mask=rejected_mask).logits.squeeze(-1)
            correct   += (r_chosen > r_rejected).sum().item()

    pairwise_acc = correct / len(val_ds)
    print(f"Pairwise accuracy: {pairwise_acc:.1%}  (target > 70%)")
    if pairwise_acc < 0.70:
        print("WARNING: Pairwise accuracy below 70% — consider more training or harder negatives before PPO.")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reward_model.yaml",
                        help="Path to YAML config file.")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
