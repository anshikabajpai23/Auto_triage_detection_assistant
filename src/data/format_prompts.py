"""
format_prompts.py
-----------------
Wraps each bug-report row in the instruction template and tokenises it
for LLaMA 3.2 1B training.

Output format (one string per sample):
    ### Incident report:
    {title}
    {body}
    ### Triage:
    severity:{P0|P1|P2|P3|P4} | team:{team}

Note: over-escalation is detected at inference time by comparing the model's
predicted severity against the severity the engineer filed. No escalation
label is needed during training.

The full string (prompt + completion) is tokenised to 512 tokens, then
saved as a HuggingFace DatasetDict to data/processed/tokenised/.

HuggingFace authentication:
    LLaMA 3.2 is a gated model. You must accept the licence at
    https://huggingface.co/meta-llama/Llama-3.2-1B and set:
        export HF_TOKEN=<your_token>
    before running this script.

    Fallback: if HF_TOKEN is not set, the script attempts to use
    TinyLlama/TinyLlama-1.1B-Chat-v1.0 (ungated, same tokeniser family)
    for local testing.

Run:
    python -m src.data.format_prompts
    python -m src.data.format_prompts --model meta-llama/Llama-3.2-1B
"""

import argparse
import os

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# ── config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "meta-llama/Llama-3.2-1B"
FALLBACK_MODEL  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LENGTH      = 512
DATA_DIR        = os.path.join(os.path.dirname(__file__), "../../data")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
TOKENISED_DIR   = os.path.join(PROCESSED_DIR, "tokenised")


# ── template ──────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
### Incident report:
{title}
{body}
### Triage:
"""

COMPLETION_TEMPLATE = "severity:{priority} | team:{team}"


def build_prompt(row: dict) -> str:
    body = row["body"].strip()
    return PROMPT_TEMPLATE.format(
        title=row["title"].strip(),
        body=("\n" + body) if body else "",
    )


def build_full_text(row: dict) -> str:
    """Prompt + completion concatenated — this is what gets tokenised for SFT."""
    prompt     = build_prompt(row)
    completion = COMPLETION_TEMPLATE.format(
        priority = row["priority"],
        team     = row["team"],
    )
    return prompt + completion


# ── tokenisation ──────────────────────────────────────────────────────────────

def load_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Load the tokeniser for the given model.
    Adds a pad token if the model doesn't have one (LLaMA has no pad token by default).
    """
    hf_token = os.environ.get("HF_TOKEN")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            use_fast=True,
        )
        print(f"Loaded tokeniser: {model_name}")
    except Exception as e:
        if model_name != FALLBACK_MODEL:
            print(f"Could not load {model_name}: {e}")
            print(f"Falling back to {FALLBACK_MODEL} for local testing.")
            tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL, use_fast=True)
        else:
            raise

    # LLaMA and TinyLlama have no pad token — use eos as pad
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def tokenise_batch(texts: list[str], tokenizer: AutoTokenizer) -> dict:
    """Tokenise a batch and return input_ids + attention_mask."""
    return tokenizer(
        texts,
        max_length      = MAX_LENGTH,
        truncation      = True,
        padding         = "max_length",
        return_tensors  = None,   # return plain lists for HF Dataset
    )


# ── main ──────────────────────────────────────────────────────────────────────

def format_split(df: pd.DataFrame, tokenizer: AutoTokenizer, split_name: str) -> Dataset:
    """Convert one DataFrame split into a tokenised HF Dataset."""
    rows = df.to_dict(orient="records")

    # Build text strings
    full_texts = [build_full_text(r) for r in rows]
    prompts    = [build_prompt(r)     for r in rows]

    # Tokenise in batches of 1000 for memory efficiency
    batch_size  = 1000
    input_ids, attention_masks, labels = [], [], []

    for i in range(0, len(full_texts), batch_size):
        batch = full_texts[i : i + batch_size]
        enc   = tokenise_batch(batch, tokenizer)
        input_ids.extend(enc["input_ids"])
        attention_masks.extend(enc["attention_mask"])

        # For causal LM: labels = input_ids (teacher forcing)
        # Tokens corresponding to the prompt portion are masked with -100
        # so the loss is only computed over the completion.
        prompt_batch = prompts[i : i + batch_size]
        prompt_enc   = tokenise_batch(prompt_batch, tokenizer)
        for full_ids, prompt_ids in zip(enc["input_ids"], prompt_enc["input_ids"]):
            prompt_len = sum(1 for t in prompt_ids if t != tokenizer.pad_token_id)
            lbl = full_ids[:]
            lbl[:prompt_len] = [-100] * prompt_len   # mask prompt tokens
            labels.append(lbl)

        if (i // batch_size) % 10 == 0:
            print(f"  [{split_name}] tokenised {min(i+batch_size, len(full_texts)):,} / {len(full_texts):,}")

    # Build HF Dataset
    dataset = Dataset.from_dict({
        "input_ids":      input_ids,
        "attention_mask": attention_masks,
        "labels":         labels,
        # Keep metadata for debugging / eval
        "priority":  [r["priority"] for r in rows],
        "team":      [r["team"]     for r in rows],
        "source":    [r["source"]   for r in rows],
        "id":        [r["id"]       for r in rows],
        "full_text": full_texts,
    })

    return dataset


def main(model_name: str = DEFAULT_MODEL):
    os.makedirs(TOKENISED_DIR, exist_ok=True)

    tokenizer = load_tokenizer(model_name)

    splits = {}
    for split in ("train", "val", "test"):
        path = os.path.join(PROCESSED_DIR, f"{split}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run src/data/load_datasets.py first."
            )
        df = pd.read_parquet(path)
        print(f"\nFormatting {split}: {len(df):,} rows …")
        splits[split] = format_split(df, tokenizer, split)

    dataset_dict = DatasetDict({
        "train": splits["train"],
        "val":   splits["val"],
        "test":  splits["test"],
    })

    dataset_dict.save_to_disk(TOKENISED_DIR)
    print(f"\nSaved tokenised DatasetDict to {TOKENISED_DIR}")
    print(dataset_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HuggingFace model ID to load the tokeniser from.",
    )
    args = parser.parse_args()
    main(args.model)
