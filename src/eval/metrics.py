"""
metrics.py
----------
Evaluation metrics for the incident triage model.

Parses model output strings of the form:
    severity:P1 | team:platform

Computes:
  - Severity macro-F1   (across P0–P4, all classes weighted equally)
  - Severity accuracy
  - Team routing accuracy
  - Per-class severity F1
  - Confusion matrix data
  - Parse failure rate  (malformed outputs — counted as wrong, not crashed)

Usage — standalone eval on a saved checkpoint:
    python -m src.eval.metrics \
        --checkpoint checkpoints/sft \
        --split val

Usage — imported in trainer for per-epoch eval:
    from src.eval.metrics import evaluate_dataframe
    results = evaluate_dataframe(df, predictions)
"""

import argparse
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ── constants ─────────────────────────────────────────────────────────────────
PRIORITY_LABELS = ["P0", "P1", "P2", "P3", "P4"]
TEAM_LABELS     = ["platform", "database", "frontend", "backend",
                   "infra", "security", "mobile"]

# ── output parser ─────────────────────────────────────────────────────────────

# Matches: "severity:P1 | team:platform"  (whitespace-tolerant)
_OUTPUT_RE = re.compile(
    r"severity\s*:\s*(P[0-4])"
    r"\s*\|\s*"
    r"team\s*:\s*(\w+)",
    re.IGNORECASE,
)


def parse_output(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse a model output string into (severity, team).

    Returns (None, None) for malformed outputs — caller treats these as wrong
    predictions rather than raising an exception.

    Examples:
        "severity:P1 | team:platform"  → ("P1", "platform")
        "severity:P0|team:infra"       → ("P0", "infra")
        "some garbage text"            → (None, None)
    """
    if not isinstance(text, str):
        return None, None

    # Try to find the pattern anywhere in the output
    # (model sometimes generates extra text before/after)
    match = _OUTPUT_RE.search(text)
    if not match:
        return None, None

    severity = match.group(1).upper()
    team     = match.group(2).lower()

    # Validate values
    if severity not in PRIORITY_LABELS:
        severity = None
    if team not in TEAM_LABELS:
        team = None

    return severity, team


def parse_outputs_batch(texts: list[str]) -> tuple[list[Optional[str]], list[Optional[str]]]:
    """Parse a list of output strings. Returns (severities, teams)."""
    severities, teams = [], []
    for t in texts:
        s, tm = parse_output(t)
        severities.append(s)
        teams.append(tm)
    return severities, teams


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    true_severities: list[str],
    pred_severities: list[Optional[str]],
    true_teams: list[str],
    pred_teams: list[Optional[str]],
    verbose: bool = True,
) -> dict:
    """
    Compute all evaluation metrics.

    Malformed predictions (None) are treated as wrong — they never match the
    true label, so they naturally reduce F1/accuracy.

    Returns a dict with keys:
        severity_macro_f1, severity_accuracy,
        team_accuracy,
        parse_failure_rate,
        per_class_f1  (dict P0..P4 → f1),
        confusion_matrix (list of lists)
    """
    n = len(true_severities)
    assert n == len(pred_severities) == len(true_teams) == len(pred_teams), \
        "All input lists must be the same length."

    # ── parse failure rate ────────────────────────────────────────────────────
    sev_failures  = sum(1 for s in pred_severities if s is None)
    team_failures = sum(1 for t in pred_teams if t is None)
    parse_failure_rate = sev_failures / n

    # ── replace None with a sentinel that never matches ───────────────────────
    SENTINEL_SEV  = "__PARSE_FAIL__"
    SENTINEL_TEAM = "__PARSE_FAIL__"
    pred_sev_clean  = [s if s is not None else SENTINEL_SEV  for s in pred_severities]
    pred_team_clean = [t if t is not None else SENTINEL_TEAM for t in pred_teams]

    # ── severity metrics ──────────────────────────────────────────────────────
    severity_macro_f1 = f1_score(
        true_severities, pred_sev_clean,
        labels=PRIORITY_LABELS, average="macro", zero_division=0
    )
    severity_accuracy = accuracy_score(true_severities, pred_sev_clean)

    per_class_f1 = f1_score(
        true_severities, pred_sev_clean,
        labels=PRIORITY_LABELS, average=None, zero_division=0
    )
    per_class_f1_dict = dict(zip(PRIORITY_LABELS, per_class_f1.tolist()))

    cm = confusion_matrix(true_severities, pred_sev_clean, labels=PRIORITY_LABELS)

    # ── team metrics ──────────────────────────────────────────────────────────
    team_accuracy = accuracy_score(true_teams, pred_team_clean)

    results = {
        "severity_macro_f1"  : round(severity_macro_f1, 4),
        "severity_accuracy"  : round(severity_accuracy, 4),
        "team_accuracy"      : round(team_accuracy, 4),
        "parse_failure_rate" : round(parse_failure_rate, 4),
        "per_class_f1"       : {k: round(v, 4) for k, v in per_class_f1_dict.items()},
        "confusion_matrix"   : cm.tolist(),
        "n_samples"          : n,
    }

    if verbose:
        _print_results(results, true_severities, pred_sev_clean)

    return results


def _print_results(results: dict, true_sev: list, pred_sev: list):
    """Pretty-print evaluation results to stdout."""
    print("=" * 55)
    print("Evaluation Results")
    print("=" * 55)
    print(f"  Samples            : {results['n_samples']:,}")
    print(f"  Parse failure rate : {results['parse_failure_rate']:.1%}")
    print()
    print(f"  Severity macro-F1  : {results['severity_macro_f1']:.4f}  (target > 0.72)")
    print(f"  Severity accuracy  : {results['severity_accuracy']:.4f}")
    print()
    print(f"  Team accuracy      : {results['team_accuracy']:.4f}  (target > 0.78)")
    print()
    print("  Per-class severity F1:")
    for label, f1 in results["per_class_f1"].items():
        bar = "█" * int(f1 * 20)
        print(f"    {label}  {f1:.4f}  {bar}")
    print()
    print("  Severity confusion matrix (rows=true, cols=pred):")
    header = "       " + "  ".join(f"{l:>4}" for l in PRIORITY_LABELS)
    print(header)
    for label, row in zip(PRIORITY_LABELS, results["confusion_matrix"]):
        print(f"  {label}  " + "  ".join(f"{v:>4}" for v in row))
    print()
    print("  Critical off-diagonal cells:")
    cm = results["confusion_matrix"]
    labels = PRIORITY_LABELS
    for true_idx, true_l in enumerate(labels):
        for pred_idx, pred_l in enumerate(labels):
            if true_idx == pred_idx:
                continue
            v = cm[true_idx][pred_idx]
            if v == 0:
                continue
            severity = "🔴 CRITICAL" if (true_l in ("P0","P1") and pred_l in ("P3","P4")) \
                       else "🟡 notable" if true_l in ("P0","P1") \
                       else ""
            if severity:
                print(f"    true={true_l} → pred={pred_l}: {v:,} {severity}")
    print("=" * 55)


# ── DataFrame-level evaluator ─────────────────────────────────────────────────

def evaluate_dataframe(df: pd.DataFrame, predictions: list[str], verbose: bool = True) -> dict:
    """
    Evaluate predictions against a ground-truth DataFrame.

    Args:
        df:          DataFrame with columns 'priority' and 'team'
        predictions: list of raw model output strings, same length as df
        verbose:     print results to stdout

    Returns:
        metrics dict (same structure as compute_metrics)
    """
    pred_sev, pred_team = parse_outputs_batch(predictions)

    return compute_metrics(
        true_severities = df["priority"].tolist(),
        pred_severities = pred_sev,
        true_teams      = df["team"].tolist(),
        pred_teams      = pred_team,
        verbose         = verbose,
    )


# ── standalone inference + eval ───────────────────────────────────────────────

def run_eval_on_checkpoint(checkpoint_dir: str, split: str = "val"):
    """
    Load a saved checkpoint, run inference on the given split, and print metrics.
    Requires: bitsandbytes (Linux/CUDA only).
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from tqdm import tqdm

    ROOT          = os.path.join(os.path.dirname(__file__), "../..")
    PROCESSED_DIR = os.path.join(ROOT, "data/processed")

    # ── load data ─────────────────────────────────────────────────────────────
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, f"{split}.parquet"))
    print(f"Loaded {split} split: {len(df):,} rows")

    # ── load model ────────────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    hf_token  = os.environ.get("HF_TOKEN")
    base_name = "meta-llama/Llama-3.2-1B"

    print(f"Loading base model: {base_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        quantization_config = bnb_config,
        device_map          = "auto",
        token               = hf_token,
        torch_dtype         = torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading LoRA adapters from: {checkpoint_dir}")
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model.eval()

    # ── inference ─────────────────────────────────────────────────────────────
    PROMPT_TEMPLATE = "### Incident report:\n{title}\n{body}\n### Triage:\n"
    predictions = []

    with torch.inference_mode():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
            body   = row["body"].strip() if row["body"] else ""
            prompt = PROMPT_TEMPLATE.format(
                title=row["title"].strip(),
                body=("\n" + body) if body else "",
            )
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=480
            ).to(model.device)

            output = model.generate(
                **inputs,
                max_new_tokens  = 20,
                do_sample       = False,
                pad_token_id    = tokenizer.eos_token_id,
            )
            # Decode only the newly generated tokens
            generated = output[0][inputs["input_ids"].shape[1]:]
            predictions.append(tokenizer.decode(generated, skip_special_tokens=True))

        # ── save all predictions ──────────────────────────────────────────────────
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    pred_sev, pred_team = parse_outputs_batch(predictions)
    results_df = df[["title", "body", "priority", "team"]].copy()
    results_df["input_prompt"]    = [
        PROMPT_TEMPLATE.format(
            title=row["title"].strip(),
            body=("\n" + row["body"].strip()) if row["body"] else "",
        )
        for _, row in df.iterrows()
    ]
    results_df["raw_output"]      = predictions
    results_df["pred_severity"]   = pred_sev
    results_df["pred_team"]       = pred_team
    results_df["sev_correct"]     = results_df["priority"] == results_df["pred_severity"]
    results_df["team_correct"]    = results_df["team"]     == results_df["pred_team"]
    results_df.to_csv(os.path.join(ROOT, f"results/eval_{split}_predictions.csv"), index=False)
    print(f"Saved all predictions to results/eval_{split}_predictions.csv")

    # ── metrics ───────────────────────────────────────────────────────────────
    results = evaluate_dataframe(df, predictions, verbose=True)
    return results


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate a triage checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to LoRA checkpoint dir.")
    parser.add_argument("--split", default="val", choices=["val", "test"],
                        help="Dataset split to evaluate on.")
    args = parser.parse_args()
    run_eval_on_checkpoint(args.checkpoint, args.split)


if __name__ == "__main__":
    main()
