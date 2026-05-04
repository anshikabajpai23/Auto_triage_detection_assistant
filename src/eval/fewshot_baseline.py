import os, sys, torch, pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from eval.metrics import evaluate_dataframe

PROCESSED_DIR = os.path.join(ROOT, "data/processed")
RESULTS_DIR   = os.path.join(ROOT, "results")

FEW_SHOT_EXAMPLES = [
    {
        "title"   : "Assign a name to the error class _LEGACY_ERROR_TEMP_2024",
        "body"    : "Choose a proper name for the error class _LEGACY_ERROR_TEMP_2024 defined in core/src/main/resources/error/error-classes.json. The name should be short but complete.",
        "priority": "P3",
        "team"    : "platform",
    },
    {
        "title"   : "Use spark.sql.timestampType for data source inference",
        "body"    : "With the configuration spark.sql.timestampType, TIMESTAMP in Spark is a user-specified alias associated with one of the TIMESTAMP_LTZ and TIMESTAMP_NTZ variations. This is quite complicated to handle in data source inference.",
        "priority": "P2",
        "team"    : "database",
    },
    {
        "title"   : "Intermittent test failure: execution-timing async script ordering",
        "body"    : "Filed by: smolnar. Parsed log: treeherder. expected property 0 to be external script #2 but got external script #1. Intermittent failure on mozilla-beta.",
        "priority": "P4",
        "team"    : "frontend",
    },
    {
        "title"   : "Over-aggressive warning for ClientWaitSync must return TIMEOUT_EXPIRED",
        "body"    : "The warning is accurate but likely common for apps to query one or a few times before returning to the event loop. Rather than asking them to refactor or workaround, we should adjust the warning threshold.",
        "priority": "P0",
        "team"    : "backend",
    },
    {
        "title"   : "Assertion failure: exn.isObject() at WasmInstance.cpp:4067",
        "body"    : "The attached testcase crashes on mozilla-central revision 20250124. Build with debug, run with --fuzzing-safe --cpu-count=2 --ion-offthread-compile=off. Backtrace: MarkPendingExceptionAsTrap.",
        "priority": "P0",
        "team"    : "infra",
    },
    {
        "title"   : "cqlsh should prefer newer TLS version by default",
        "body"    : "Some new JDK releases started to disable TLSv1.0 and TLSv1.1. The current default in cqlsh should be updated to prefer TLSv1.2 or higher to maintain compatibility.",
        "priority": "P3",
        "team"    : "security",
    },
    {
        "title"   : "Startup crash running android sw-wr in debug build",
        "body"    : "webrender::device::gl::Device::new closure gl.rs:1417. ErrorReactingGl::get_integer_v crash on startup when running android sw-wr in debug build.",
        "priority": "P1",
        "team"    : "mobile",
    },
]

PROMPT_TEMPLATE = """\
### Incident report:
{title}
{body}
### Triage (always output severity P0-P4 and one team only):
"""

COMPLETION_TEMPLATE = "severity:{priority} | team:{team}\n"


def build_few_shot_prefix() -> str:
    prefix = ""
    for ex in FEW_SHOT_EXAMPLES:
        body   = ex["body"].strip()
        prompt = PROMPT_TEMPLATE.format(title=ex["title"], body="\n" + body)
        completion = COMPLETION_TEMPLATE.format(priority=ex["priority"], team=ex["team"])
        prefix += prompt + completion + "\n\n"
    return prefix


FEW_SHOT_PREFIX = build_few_shot_prefix()


# ── load model ────────────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    quantization_config=bnb_config,
    device_map="auto",
)
model.eval()

newline_token = tokenizer.encode("\n", add_special_tokens=False)[0]

# ── load val set ──────────────────────────────────────────────────────────────
df = pd.read_parquet(os.path.join(PROCESSED_DIR, "val.parquet")).head(1000)
print(f"Evaluating {len(df):,} samples with few-shot base model...")

# ── inference ─────────────────────────────────────────────────────────────────
predictions = []

with torch.inference_mode():
    for _, row in tqdm(df.iterrows(), total=len(df)):
        body   = row["body"].strip() if row["body"] else ""
        prompt = FEW_SHOT_PREFIX + PROMPT_TEMPLATE.format(
            title=row["title"].strip(),
            body=("\n" + body) if body else "",
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens = 15,
            do_sample      = False,
            pad_token_id   = tokenizer.eos_token_id,
            eos_token_id   = [tokenizer.eos_token_id, newline_token],
        )
        generated = output[0][inputs["input_ids"].shape[1]:]
        predictions.append(tokenizer.decode(generated, skip_special_tokens=True))

# ── metrics ───────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
results = evaluate_dataframe(df, predictions, verbose=True)

out_df = df[["title", "body", "priority", "team"]].copy()
out_df["raw_output"] = predictions
out_df.to_csv(os.path.join(RESULTS_DIR, "eval_fewshot_base_predictions.csv"), index=False)
print(f"Saved predictions to {RESULTS_DIR}/eval_fewshot_base_predictions.csv")