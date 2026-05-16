"""
dpo_trainer.py
---------------
Stage 2: Direct Preference Optimization (DPO) — replaces RM + PPO.

What this does:
  - Loads the SFT checkpoint (oversampling) as the starting policy
  - Keeps a frozen copy as the reference model (no separate RM needed)
  - Trains with DPO loss: -log σ(β * (log π/π_ref on chosen - log π/π_ref on rejected))
  - Uses (prompt, chosen, rejected) pairs from build_preference_pairs.py directly
  - Logs reward_margin, KL, and eval loss to W&B
  - Saves best checkpoint to checkpoints/dpo/

Run on BigRed200:
    sbatch scripts/slurm/run_dpo.sh

Run locally:
    python -m src.models.dpo_trainer --config configs/dpo.yaml
"""

import argparse
import json
import os

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "../..")


# ── config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── data ──────────────────────────────────────────────────────────────────────

def load_preference_pairs(path: str) -> Dataset:
    """Load JSONL preference pairs. DPOTrainer expects: prompt, chosen, rejected."""
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line.strip()))
    ds = Dataset.from_list(pairs)
    print(f"  Loaded {len(ds):,} pairs from {path}")
    return ds


# ── model ─────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict):
    model_name     = cfg["model_name"]
    sft_checkpoint = os.path.join(ROOT, cfg.get("sft_checkpoint", "checkpoints/sft_weighted_oversample"))
    hf_token       = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint, token=hf_token, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # DPO needs left padding for generation

    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type       = cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype    = torch.bfloat16,
        bnb_4bit_use_double_quant = cfg.get("bnb_4bit_use_double_quant", True),
    )

    print(f"Loading base model: {model_name} in 4-bit NF4 …")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config = bnb_config,
        device_map          = "auto",
        token               = hf_token,
        torch_dtype         = torch.bfloat16,
    )
    base_model.config.use_cache      = False
    base_model.config.pretraining_tp = 1

    print(f"Loading SFT LoRA adapters from: {sft_checkpoint}")
    base_model = prepare_model_for_kbit_training(
        base_model,
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

    # Load SFT adapters as starting policy — DPOTrainer uses base (no adapter) as ref
    model = PeftModel.from_pretrained(base_model, sft_checkpoint, is_trainable=True)
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ── training ──────────────────────────────────────────────────────────────────

def build_dpo_config(cfg: dict) -> DPOConfig:
    gck_kwargs = cfg.get("gradient_checkpointing_kwargs", {"use_reentrant": False})

    return DPOConfig(
        output_dir                    = cfg.get("output_dir", "checkpoints/dpo"),
        beta                          = cfg.get("beta", 0.1),
        max_length                    = cfg.get("max_length", 512),
        max_prompt_length             = cfg.get("max_prompt_length", 256),
        num_train_epochs              = cfg.get("num_train_epochs", 2),
        per_device_train_batch_size   = cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size    = cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps   = cfg.get("gradient_accumulation_steps", 4),
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
        run_name                      = cfg.get("run_name", "dpo-llama-3.2-1b"),
        dataloader_num_workers        = cfg.get("dataloader_num_workers", 4),
        dataloader_pin_memory         = cfg.get("dataloader_pin_memory", True),
        remove_unused_columns         = False,
    )


def train(cfg: dict):
    # ── load data ─────────────────────────────────────────────────────────────
    train_path = os.path.join(ROOT, cfg.get("train_data", "data/preference_pairs/train.jsonl"))
    val_path   = os.path.join(ROOT, cfg.get("val_data",   "data/preference_pairs/val.jsonl"))

    print(f"Loading preference pairs …")
    train_ds = load_preference_pairs(train_path)
    val_ds   = load_preference_pairs(val_path)

    # ── load model ────────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(cfg)

    # ── DPO config ────────────────────────────────────────────────────────────
    dpo_config = build_dpo_config(cfg)

    output_dir = cfg.get("output_dir", "checkpoints/dpo")
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
    # ref_model=None → DPOTrainer uses the base model (without LoRA) as reference
    # This is memory-efficient: one model in memory, adapters toggled on/off
    trainer = DPOTrainer(
        model         = model,
        ref_model     = None,
        args          = dpo_config,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        tokenizer     = tokenizer,
    )

    print(f"\nStarting DPO training (beta={cfg.get('beta', 0.1)}) …")
    print("W&B will log: rewards/chosen, rewards/rejected, rewards/margins, logps, KL")
    trainer.train(resume_from_checkpoint=latest_checkpoint)

    # ── save ──────────────────────────────────────────────────────────────────
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nDPO model saved to {output_dir}")
    print("Next: run inference with checkpoints/dpo/ and compare to SFT baseline on val set")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo.yaml",
                        help="Path to YAML config file.")
    args = parser.parse_args()
    cfg  = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
