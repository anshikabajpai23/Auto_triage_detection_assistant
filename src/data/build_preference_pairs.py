"""
build_preference_pairs.py
--------------------------
Generate (prompt, chosen, rejected) preference pairs from SFT predictions
for reward model training.

Strategy:
  - chosen  : always the ground-truth completion
  - rejected: model's wrong prediction (if wrong) OR adversarial adjacent
              severity (if model got it right)
  - Hard negatives (20%): adjacent severity confusions — most informative

Output format (JSONL):
  {"prompt": "### Incident report:...", "chosen": "severity:P1 | team:platform",
   "rejected": "severity:P2 | team:platform"}

Usage:
    python -m src.data.build_preference_pairs \
        --predictions results/eval_val_predictions.csv \
        --output data/preference_pairs/
"""

import argparse
import json
import os
import random

import pandas as pd

# ── constants ─────────────────────────────────────────────────────────────────
PRIORITY_LABELS = ["P0", "P1", "P2", "P3", "P4"]

PROMPT_TEMPLATE = """\
### Incident report:
{title}
{body}
### Triage (always output severity P0-P4 and one team only):
"""

# Adjacent severity for constructing hard negatives
ADJACENT = {
    "P0": ["P1", "P2"],
    "P1": ["P0", "P2"],
    "P2": ["P1", "P3"],
    "P3": ["P2", "P4"],
    "P4": ["P3", "P2"],
}


def build_prompt(title: str, body: str) -> str:
    body = body.strip() if isinstance(body, str) and body else ""
    return PROMPT_TEMPLATE.format(
        title=title.strip(),
        body=("\n" + body) if body else "",
    )


def build_completion(severity: str, team: str) -> str:
    return f"severity:{severity} | team:{team}"


def make_pairs(df: pd.DataFrame, hard_neg_ratio: float = 0.20, seed: int = 42) -> list[dict]:
    """
    Build preference pairs from a predictions DataFrame.

    Args:
        df:              Predictions CSV with columns: title, body, priority,
                         team, pred_severity, pred_team, sev_correct, team_correct
        hard_neg_ratio:  Fraction of correct predictions to use as hard negatives
        seed:            Random seed

    Returns:
        List of dicts with keys: prompt, chosen, rejected
    """
    rng   = random.Random(seed)
    pairs = []

    for _, row in df.iterrows():
        true_sev  = row["priority"]
        true_team = row["team"]
        pred_sev  = row["pred_severity"] if pd.notna(row["pred_severity"]) else None
        pred_team = row["pred_team"]     if pd.notna(row["pred_team"])     else None

        prompt = build_prompt(row["title"], row["body"])
        chosen = build_completion(true_sev, true_team)

        sev_correct  = row["sev_correct"]
        team_correct = row["team_correct"]

        # ── Case 1: model got something wrong → use its prediction as rejected ──
        if not sev_correct or not team_correct:
            if pred_sev is not None and pred_team is not None:
                rejected = build_completion(pred_sev, pred_team)
                if rejected != chosen:
                    pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
                    continue

        # ── Case 2: parse failure → use a plausible wrong answer as rejected ──
        if pred_sev is None or pred_team is None:
            wrong_sev = rng.choice(ADJACENT[true_sev])
            rejected  = build_completion(wrong_sev, true_team)
            pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
            continue

        # ── Case 3: model got it fully right → construct adversarial hard negative ──
        if rng.random() < hard_neg_ratio:
            wrong_sev = rng.choice(ADJACENT[true_sev])
            rejected  = build_completion(wrong_sev, true_team)
            pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    return pairs


def print_stats(pairs: list[dict], label: str):
    print(f"\n{label}: {len(pairs):,} pairs")
    chosen_sevs   = [p["chosen"].split("|")[0].split(":")[1].strip() for p in pairs]
    rejected_sevs = [p["rejected"].split("|")[0].split(":")[1].strip() for p in pairs]
    from collections import Counter
    print("  Chosen severity dist :", dict(sorted(Counter(chosen_sevs).items())))
    print("  Rejected severity dist:", dict(sorted(Counter(rejected_sevs).items())))


def save_jsonl(pairs: list[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"  Saved {len(pairs):,} pairs → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True,
                        help="Path to predictions CSV (from src/eval/metrics.py)")
    parser.add_argument("--output", default="data/preference_pairs/",
                        help="Output directory for JSONL files")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of pairs to hold out for RM validation")
    parser.add_argument("--hard_neg_ratio", type=float, default=0.20,
                        help="Fraction of correct predictions to use as hard negatives")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading predictions from: {args.predictions}")
    df = pd.read_csv(args.predictions)
    print(f"  {len(df):,} rows loaded")
    print(f"  Severity correct: {df['sev_correct'].sum():,} / {len(df):,} ({df['sev_correct'].mean():.1%})")
    print(f"  Team correct    : {df['team_correct'].sum():,} / {len(df):,} ({df['team_correct'].mean():.1%})")

    # Build all pairs
    all_pairs = make_pairs(df, hard_neg_ratio=args.hard_neg_ratio, seed=args.seed)
    print_stats(all_pairs, "All pairs")

    # Train/val split
    rng = random.Random(args.seed)
    rng.shuffle(all_pairs)
    n_val   = int(len(all_pairs) * args.val_ratio)
    val     = all_pairs[:n_val]
    train   = all_pairs[n_val:]

    print(f"\nSplit: {len(train):,} train / {len(val):,} val")

    # Save
    save_jsonl(train, os.path.join(args.output, "train.jsonl"))
    save_jsonl(val,   os.path.join(args.output, "val.jsonl"))
    print("\nDone.")


if __name__ == "__main__":
    main()
