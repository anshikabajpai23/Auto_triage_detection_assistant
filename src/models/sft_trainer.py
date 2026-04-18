"""
sft_trainer.py
--------------
Stage 1: Supervised Fine-Tuning (SFT) with QLoRA on bug-report triage data.

What this does:
  - Loads LLaMA 3.2 1B in 4-bit NF4 (QLoRA) via BitsAndBytes
  - Attaches LoRA adapters to attention projections only
  - Fine-tunes on (incident description → severity | team) pairs
  - Logs all metrics to Weights & Biases
  - Saves the best LoRA adapter checkpoint to checkpoints/sft/

Run on BigRed200:
    sbatch scripts/slurm/run_sft.sh

Run locally (uses TinyLlama fallback if HF_TOKEN not set):
    python -m src.models.sft_trainer --config configs/sft_llama.yaml

Requirements:
    - HF_TOKEN env var set (LLaMA 3.2 is gated)
    - WANDB_API_KEY env var set  (or run `wandb login` first)
    - bitsandbytes installed (Linux/CUDA only — not available on macOS)
"""

import argparse
import os
import sys

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

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT          = os.path.join(os.path.dirname(__file__), "../..")
PROCESSED_DIR = os.path.join(ROOT, "data/processed")
FALLBACK_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ── output template (must match format_prompts.py exactly) ────────────────────
PROMPT_TEMPLATE = """\
### Incident report:
{title}
{body}
### Triage:
"""
COMPLETION_TEMPLATE = "severity:{priority} | team:{team}"


# ── config loading ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── dataset ───────────────────────────────────────────────────────────────────

def load_splits(processed_dir: str) -> DatasetDict:
    """Load train/val parquets and convert to HuggingFace DatasetDict."""
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
    print(f"Val:   {len(splits['val']):,} rows\n")
    return DatasetDict(splits)


def make_formatting_func(tokenizer):
    """
    Returns a function that formats a batch of rows into full training strings.
    SFTTrainer calls this on each batch — no pre-tokenisation needed.
    """
    def formatting_func(batch: dict) -> list[str]:
        texts = []
        for title, body, priority, team in zip(
            batch["title"], batch["body"], batch["priority"], batch["team"]
        ):
            body = body.strip() if body else ""
            prompt = PROMPT_TEMPLATE.format(
                title=title.strip(),
                body=("\n" + body) if body else "",
            )
            completion = COMPLETION_TEMPLATE.format(priority=priority, team=team)
            texts.append(prompt + completion)
        return texts

    return formatting_func


# ── model loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict):
    model_name = cfg["model_name"]
    hf_token   = os.environ.get("HF_TOKEN")

    # ── tokeniser ─────────────────────────────────────────────────────────────
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, token=hf_token, use_fast=True
        )
        print(f"Loaded tokeniser: {model_name}")
    except Exception as e:
        print(f"Could not load tokeniser for {model_name}: {e}")
        print(f"Falling back to {FALLBACK_MODEL}")
        model_name = FALLBACK_MODEL
        tokenizer  = AutoTokenizer.from_pretrained(FALLBACK_MODEL, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # important for causal LM training

    # ── BitsAndBytes QLoRA config ──────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type       = cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = cfg.get("bnb_4bit_use_double_quant", True),
    )

    # ── base model ────────────────────────────────────────────────────────────
    print(f"Loading model: {model_name} in 4-bit NF4 …")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        device_map          = "auto",
        token               = hf_token,
        torch_dtype         = torch.bfloat16,
    )
    model.config.use_cache = False          # disable KV cache during training
    model.config.pretraining_tp = 1        # tensor parallelism = 1 for single GPU

    # ── prepare for k-bit training (adds gradient hooks) ─────────────────────
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing = cfg.get("gradient_checkpointing", True),
    )

    # ── LoRA adapters ─────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r               = cfg.get("lora_r", 16),
        lora_alpha      = cfg.get("lora_alpha", 32),
        lora_dropout    = cfg.get("lora_dropout", 0.05),
        target_modules  = cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
        bias            = cfg.get("lora_bias", "none"),
        task_type       = TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ── training ──────────────────────────────────────────────────────────────────

def build_training_args(cfg: dict) -> TrainingArguments:
    gck_kwargs = cfg.get("gradient_checkpointing_kwargs", {"use_reentrant": False})

    return TrainingArguments(
        output_dir                  = cfg.get("output_dir", "checkpoints/sft"),
        num_train_epochs            = cfg.get("num_train_epochs", 3),
        per_device_train_batch_size = cfg.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size  = cfg.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 4),
        learning_rate               = cfg.get("learning_rate", 2e-4),
        lr_scheduler_type           = cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio                = cfg.get("warmup_ratio", 0.05),
        weight_decay                = cfg.get("weight_decay", 0.01),
        max_grad_norm               = cfg.get("max_grad_norm", 1.0),
        optim                       = cfg.get("optim", "paged_adamw_32bit"),
        bf16                        = cfg.get("bf16", True),
        tf32                        = cfg.get("tf32", True),
        gradient_checkpointing      = cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs = gck_kwargs,
        logging_steps               = cfg.get("logging_steps", 10),
        eval_strategy               = cfg.get("eval_strategy", "epoch"),
        save_strategy               = cfg.get("save_strategy", "epoch"),
        save_total_limit            = cfg.get("save_total_limit", 3),
        load_best_model_at_end      = cfg.get("load_best_model_at_end", True),
        metric_for_best_model       = cfg.get("metric_for_best_model", "eval_loss"),
        report_to                   = cfg.get("report_to", "wandb"),
        run_name                    = cfg.get("run_name", "sft-llama-3.2-1b"),
        dataloader_num_workers      = cfg.get("dataloader_num_workers", 4),
        dataloader_pin_memory       = cfg.get("dataloader_pin_memory", True),
        remove_unused_columns       = False,
    )


def train(cfg: dict):
    # Load data
    dataset = load_splits(PROCESSED_DIR)

    # Load model + tokeniser
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Build training args
    training_args = build_training_args(cfg)

    # SFTTrainer
    trainer = SFTTrainer(
        model             = model,
        tokenizer         = tokenizer,
        train_dataset     = dataset["train"],
        eval_dataset      = dataset["val"],
        formatting_func   = make_formatting_func(tokenizer),
        max_seq_length    = cfg.get("max_seq_length", 512),
        packing           = cfg.get("packing", False),
        args              = training_args,
    )

    print("\nStarting SFT training …")
    trainer.train()

    # Save final LoRA adapters
    output_dir = cfg.get("output_dir", "checkpoints/sft")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nLoRA adapters saved to {output_dir}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/sft_llama.yaml",
        help="Path to YAML config file.",
    )
    args   = parser.parse_args()
    cfg    = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
