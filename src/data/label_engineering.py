"""
label_engineering.py
--------------------
All label derivation and text cleaning for the unified bug-report DataFrame.

Runs three sequential passes:
  1. Priority normalisation   → column 'priority'   (P0–P4)
  2. Team label engineering   → column 'team'        (7 buckets)
  3. Text cleaning            → columns 'title', 'body' cleaned in-place

Note: over-escalation detection is handled at inference time by comparing
the model's predicted priority against the filed priority — no label
engineering is needed during training.
"""

import re
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# 1. PRIORITY NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

# Eclipse / Bugzilla severity labels → P0-P4
SEVERITY_TO_PRIORITY = {
    "blocker":     "P0",
    "critical":    "P1",
    "major":       "P2",
    "normal":      "P3",
    "minor":       "P4",
    "trivial":     "P4",
    # "enhancement": "P4",
}

#TODO: WOULD NOT WORK FOR GITHUB
# GitBugs / Jira style P1-P5 → P0-P4  (shift down by 1, cap at P4)
# GITBUGS_PRIORITY_MAP = {
#     "P1": "P0",
#     "P2": "P1",
#     "P3": "P2",
#     "P4": "P3",
#     "P5": "P4",
#     "BLOCKER":  "P0",
#     "CRITICAL": "P1",
#     "MAJOR":    "P2",
#     "MINOR":    "P3",
#     "TRIVIAL":  "P4",
# }

GITBUGS_PRIORITY_MAP = {
    "HIGH": "P1",
    "NORMAL": "P3",
    "LOW": "P4",
    "URGENT": "P0",
    "P1": "P0",
    "P2": "P1",
    "P3": "P2",
    "P4": "P3",
    "P5": "P4",
    "BLOCKER": "P0",
    "CRITICAL": "P1",
    "MAJOR": "P2",
    "MINOR": "P3",
    "TRIVIAL": "P4",
}

def _normalise_priority_row(row: pd.Series) -> str:
    source = row["source"]

    if source == "eclipse":
        return SEVERITY_TO_PRIORITY.get(row["raw_severity"], "P3")

    if source == "gitbugs":
        return GITBUGS_PRIORITY_MAP.get(row["raw_priority"].upper(), "P3")

    # if source == "apache":
    #     # No explicit priority — derive from description text
    #     text = (row["title"] + " " + row["body"]).lower()
    #     if CRITICAL_KEYWORDS.search(text):
    #         return "P1"
    #     # Default based on resolution: wontfix bugs are usually lower severity
    #     if "wontfix" in row["resolution"]:
    #         return "P3"
    #     return "P2"

    return "P3"


def normalise_priority(df: pd.DataFrame) -> pd.DataFrame:
    df["priority"] = df.apply(_normalise_priority_row, axis=1)

    # Sanity check
    null_count = df["priority"].isna().sum()
    assert null_count == 0, f"Null priorities after normalisation: {null_count}"

    print("Priority distribution:")
    print(df["priority"].value_counts().sort_index().to_string())
    print()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. TEAM LABEL ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

# Maps project / component keywords to team buckets.
# Checked in order — first match wins.
# Manually validated against a 200-sample spot-check (see notebooks/02_label_analysis.ipynb).
TEAM_RULES: list[tuple[re.Pattern, str]] = [
    # ── security ───────────────────────────────────────────────────────────
    (re.compile(r"\b(security|auth|ssl|tls|crypto|certificate|permission|"
                r"oauth|saml|jwt|encryption)\b", re.I), "security"),

    # ── database ───────────────────────────────────────────────────────────
    (re.compile(r"\b(database|db|sql|mysql|postgres|sqlite|oracle|mongo|"
                r"jdbc|orm|hibernate|derby|h2)\b", re.I), "database"),

    # ── mobile ─────────────────────────────────────────────────────────────
    (re.compile(r"\b(android|ios|iphone|ipad|mobile|tablet|swift|kotlin)\b",
                re.I), "mobile"),

    # ── frontend / UI ──────────────────────────────────────────────────────
    (re.compile(r"\b(ui|gui|frontend|interface|widget|dialog|window|"
                r"css|html|browser|editor|view|swt|birt|rendering|"
                r"firefox|thunderbird|display|layout)\b", re.I), "frontend"),

    # ── infra / devops ─────────────────────────────────────────────────────
    (re.compile(r"\b(infra|infrastructure|deploy|docker|kubernetes|k8s|"
                r"ci|cd|pipeline|devops|monitoring|logging|tomcat|server|"
                r"maven|gradle|ant|build|jenkins|slurm)\b", re.I), "infra"),

    # ── platform / core runtime ────────────────────────────────────────────
    (re.compile(r"\b(platform|core|runtime|jvm|jdk|classpath|"
                r"aspectj|eclipse[_ ]platform|equinox)\b", re.I), "platform"),

    # ── backend / general Java / API ───────────────────────────────────────
    (re.compile(r"\b(api|backend|service|rest|endpoint|handler|"
                r"controller|jdt|cdt|compiler|parser|refactor)\b", re.I), "backend"),
]


def _assign_team(project: str, title: str, body: str) -> str:
    """Try project field first, then fall back to text search in title+body."""
    text = (project + " " + title + " " + body[:200]).strip()
    for pattern, bucket in TEAM_RULES:
        if pattern.search(text):
            return bucket
    return "unknown"


def engineer_team_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["team"] = df.apply(
        lambda r: _assign_team(r["project"], r["title"], r["body"]), axis=1
    )

    total    = len(df)
    unknown  = (df["team"] == "unknown").sum()
    print(f"Team distribution ({unknown:,} / {total:,} = {100*unknown/total:.1f}% unknown):")
    print(df["team"].value_counts().to_string())
    print()
    
    # import openpyxl
    # examples_per_team = (
    #     df.groupby("team", group_keys=False)
    #       .head(10)
    #       .reset_index(drop=True)
    # )

    # print(examples_per_team[["team", "project", "title", "body"]].to_string(index=False))
    # examples_per_team.to_excel("team_examples.xlsx", index=False)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════════════

# Regex patterns for cleaning
_HTML_TAG      = re.compile(r"<[^>]+>")
_XML_ENTITY    = re.compile(r"&[a-z]+;|&#\d+;")
_STACKTRACE    = re.compile(
    r"((\s*at\s+[\w\.$]+\([\w\.]+:\d+\)[\r\n]+){10})"  # keeps first 10 'at' lines
    r"(\s*at\s+[\w\.$]+\([\w\.]+:\d+\)[\r\n]+)*",
    re.MULTILINE,
)
_MULTI_SPACE   = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Strip HTML, truncate stack traces, normalise whitespace."""
    if not isinstance(text, str):
        return ""
    text = _HTML_TAG.sub(" ", text)
    text = _XML_ENTITY.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _STACKTRACE.sub(r"\1", text)   # keep first 10 'at' lines, drop the rest
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def truncate_to_words(text: str, max_words: int = 400) -> str:
    """Rough word-count truncation (faster than tokenising at this stage)."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def clean_texts(df: pd.DataFrame, max_words: int = 400) -> pd.DataFrame:
    df["title"] = df["title"].apply(clean_text)
    df["body"]  = df["body"].apply(
        lambda t: truncate_to_words(clean_text(t), max_words)
    )
    print(f"Text cleaning done. Max title words: {df['title'].apply(lambda t: len(t.split())).max()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_label_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all three label engineering passes in order.
    Returns DataFrame with new columns:
        priority, team, cleaned title & body.
    """
    print("=" * 60)
    print("Label engineering")
    print("=" * 60)

    df = normalise_priority(df)
    df = engineer_team_labels(df)
    df = clean_texts(df)

    # Final column selection for downstream use
    keep = [
        "id", "source", "title", "body",
        "priority", "team",
        "raw_severity", "resolution", "resolution_time_days",
    ]
    df = df[[c for c in keep if c in df.columns]]

    print(f"\nLabel engineering complete. Final shape: {df.shape}")
    return df
