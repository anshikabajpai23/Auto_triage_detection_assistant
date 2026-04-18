"""
load_datasets.py
----------------
Loads and combines bug report datasets into a single unified DataFrame,
then produces stratified train/val/test splits.

Datasets used:
  1. Eclipse Bugzilla  — AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset (88K)
     Fields: Project, Bug ID, Severity Label, Resolution Status, Short Description
     Limitation: all resolutions are FIXED; no timestamps.

  2. Apache Bug Reports — Partha117/apache_bug_reports (22K)
     Fields: id, bug_id, summary, description, report_time, commit_timestamp,
             status, project_name
     Strength: has wontfix/resolved status and timestamps — used for over-escalation.

  3. GitBugs CSV (optional, manual download)
     If present at data/raw/gitbugs.csv, it is merged in.
     Fields expected: id, title, body, priority, resolution, created_at, closed_at, project

Unified schema after loading (all column names normalised):
    id, source, title, body, raw_severity, raw_priority, resolution,
    project, created_at, closed_at, resolution_time_days

Run:
    python -m src.data.load_datasets
"""

import os
import pandas as pd
from datasets import load_dataset

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR      = os.path.join(os.path.dirname(__file__), "../../data")
RAW_DIR       = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
GITBUGS_CSV   = os.path.join(RAW_DIR, "gitbugs.csv")


# ── loaders ───────────────────────────────────────────────────────────────────

def load_eclipse() -> pd.DataFrame:
    """
    Load Eclipse Bugzilla from HuggingFace.

    Notes:
    - Only 'Short Description' is available (no long body).
    - All Resolution Status values are 'FIXED' in this HF snapshot.
    - No timestamps → resolution_time_days will be NaN.
    """
    print("Loading Eclipse Bugzilla …")
    ds = load_dataset("AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset", split="train")
    df = ds.to_pandas()

    out = pd.DataFrame({
        "id":                 df["Bug ID"].astype(str),
        "source":             "eclipse",
        "title":              df["Short Description"].fillna(""),
        "body":               "",          # not available in this dataset
        "raw_severity":       df["Severity Label"].str.lower().fillna(""),
        "raw_priority":       "",          # derived later from raw_severity
        "resolution":         df["Resolution Status"].str.lower().fillna("fixed"),
        "project":            df["Project"].fillna(""),
        "created_at":         pd.NaT,
        "closed_at":          pd.NaT,
        "resolution_time_days": float("nan"),
    })

    print(f"  Eclipse: {len(out):,} rows  |  severity dist:\n{out['raw_severity'].value_counts().to_string()}\n")
    return out


# def load_apache() -> pd.DataFrame:
#     """
#     Load Apache Bug Reports from HuggingFace.

#     Notes:
#     - Has 'resolved wontfix' / 'closed wontfix' status (~182 records).
#     - Uses report_time (created) and commit_timestamp (closed) for resolution time.
#     - No explicit severity/priority field; severity is derived in label_engineering.
#     """
#     print("Loading Apache Bug Reports …")
#     ds = load_dataset("Partha117/apache_bug_reports", split="train")
#     df = ds.to_pandas()

#     # Parse timestamps
#     created_at = pd.to_datetime(df["report_time"], errors="coerce", utc=True)
#     # commit_timestamp is a Unix epoch string
#     closed_at  = pd.to_datetime(
#         pd.to_numeric(df["commit_timestamp"], errors="coerce"),
#         unit="s", utc=True
#     )
#     resolution_time_days = (closed_at - created_at).dt.days

#     out = pd.DataFrame({
#         "id":                   df["bug_id"].astype(str),
#         "source":               "apache",
#         "title":                df["summary"].fillna(""),
#         "body":                 df["description"].fillna(""),
#         "raw_severity":         "",          # no severity in this dataset; assigned heuristically
#         "raw_priority":         "",
    #     "resolution":           df["status"].str.lower().fillna(""),
    #     "project":              df["project_name"].fillna(""),
    #     "created_at":           created_at,
    #     "closed_at":            closed_at,
    #     "resolution_time_days": resolution_time_days,
    # })

    # print(f"  Apache: {len(out):,} rows  |  status dist:\n{out['resolution'].value_counts().to_string()}\n")
    # return out


def load_gitbugs() -> pd.DataFrame | None:
    """
    Load GitBugs CSV if it exists at data/raw/gitbugs.csv.

    Download instructions:
      1. Find the GitBugs dataset repo (search arXiv: 'GitBugs priority prediction dataset').
      2. Download the CSV files and merge on title+body columns.
      3. Save to data/raw/gitbugs.csv with columns:
         id, title, body, priority, resolution, created_at, closed_at, project

    Expected priority values: P1, P2, P3, P4, P5 (will be normalised to P0-P4).
    Expected resolution values: fixed, wontfix, invalid, worksforme, duplicate.
    """
    if not os.path.exists(GITBUGS_CSV):
        print("GitBugs CSV not found — skipping. See docstring for download instructions.")
        return None

    print("Loading GitBugs CSV …")
    df = pd.read_csv(GITBUGS_CSV, low_memory=False)

    required = {"id", "title", "body", "priority", "resolution", "created_at", "closed_at", "project"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"GitBugs CSV missing expected columns: {missing}")

    created_at = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    closed_at  = pd.to_datetime(df["closed_at"],  errors="coerce", utc=True)

    out = pd.DataFrame({
        "id":                   df["id"].astype(str),
        "source":               "gitbugs",
        "title":                df["title"].fillna(""),
        "body":                 df["body"].fillna(""),
        "raw_severity":         "",
        "raw_priority":         df["priority"].str.upper().fillna(""),
        "resolution":           df["resolution"].str.lower().fillna(""),
        "project":              df["project"].fillna(""),
        "created_at":           created_at,
        "closed_at":            closed_at,
        "resolution_time_days": (closed_at - created_at).dt.days,
    })

    print(f"  GitBugs: {len(out):,} rows  |  priority dist:\n{out['raw_priority'].value_counts().to_string()}\n")
    return out


# ── combine ───────────────────────────────────────────────────────────────────

def load_all() -> pd.DataFrame:
    """Load and concatenate all available datasets into one unified DataFrame."""
    # parts = [load_eclipse(), load_apache()]
    parts = [load_eclipse()]

    gitbugs = load_gitbugs()
    if gitbugs is not None:
        parts.append(gitbugs)

    df = pd.concat(parts, ignore_index=True)
    print(f"Combined dataset: {len(df):,} rows from {df['source'].value_counts().to_dict()}\n")
    return df


# ── splitting ─────────────────────────────────────────────────────────────────

def stratified_split(
    df: pd.DataFrame,
    train_frac: float = 0.80,
    val_frac:   float = 0.10,
    seed:       int   = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    80/10/10 stratified split on priority.

    Stratification ensures all priority classes (P0–P4) are represented
    proportionally in every split.
    """
    from sklearn.model_selection import train_test_split

    df = df.copy()

    # First cut: train vs (val+test)
    train, tmp = train_test_split(
        df, test_size=(1 - train_frac), stratify=df["priority"], random_state=seed
    )
    # Second cut: val vs test (equal halves of the remaining 20 %)
    val_size = val_frac / (1 - train_frac)
    val, test = train_test_split(
        tmp, test_size=(1 - val_size), stratify=tmp["priority"], random_state=seed
    )

    train = train.reset_index(drop=True)
    val   = val.reset_index(drop=True)
    test  = test.reset_index(drop=True)

    def report(name, split):
        print(f"  {name:5s}: {len(split):6,} rows  |  priority dist: {split['priority'].value_counts().sort_index().to_dict()}")

    print("Train / val / test split:")
    report("train", train)
    report("val",   val)
    report("test",  test)

    return train, val, test


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # Inline imports here to keep module-level imports light
    from label_engineering import run_label_engineering

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = load_all()
    df = run_label_engineering(df)

    before = len(df)
    df = df[df["team"] != "unknown"].reset_index(drop=True)
    print(f"Dropped {before - len(df):,} rows with team='unknown'. Remaining: {len(df):,}\n")

    train, val, test = stratified_split(df)

    

    train.to_parquet(os.path.join(PROCESSED_DIR, "train.parquet"), index=False)
    val.to_parquet(os.path.join(PROCESSED_DIR, "val.parquet"),     index=False)
    test.to_parquet(os.path.join(PROCESSED_DIR, "test.parquet"),   index=False)

    print(f"\nSaved splits to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
