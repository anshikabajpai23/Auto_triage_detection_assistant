# Incident Severity Auto-Triage — CLAUDE.md

## Project Overview

Fine-tune LLaMA 3.2 1B on real-world bug reports to produce a **2-head classifier** in a single forward pass:
- **Severity** — P0–P4
- **Team routing** — platform / database / frontend / backend / infra / security / mobile

Over-escalation is detected **at inference time** by comparing the model's predicted severity against the severity the engineer originally filed — no training label needed, no escalation head in the model.

Training pipeline mirrors production LLM alignment: SFT → Reward Model → PPO.

## Architecture

```
Raw bug reports (Eclipse Bugzilla + GitBugs)
        │
        ▼
┌─────────────────────────────────┐
│        Data Pipeline            │
│  - Priority normalization       │
│  - Team label engineering       │
│  - Drop team="unknown" rows     │
│  - Text cleaning + tokenization │
│  - Stratified train/val/test    │
└─────────────┬───────────────────┘
              │ labeled prompt-completion pairs
              ▼
┌─────────────────────────────────┐
│   Stage 1: SFT (QLoRA)          │
│  Base: LLaMA 3.2 1B             │
│  Adapters: r=16, alpha=32       │  ← teaches format + baseline accuracy
│  Output: severity|team          │
│  Est: 1.5–2.5 GPU hrs           │
└─────────────┬───────────────────┘
              │ SFT checkpoint
              ▼
┌─────────────────────────────────┐
│   Stage 2: Reward Model         │
│  Base: LLaMA 3.2 1B             │
│  Head: scalar linear output     │  ← Bradley-Terry pairwise loss
│  Data: preference pairs         │
│  Est: 0.5–1 GPU hr              │
└─────────────┬───────────────────┘
              │ RM checkpoint
              ▼
┌─────────────────────────────────┐
│   Stage 3: PPO                  │
│  Policy: SFT checkpoint         │
│  Reference: SFT frozen (KL)     │  ← TRL PPOTrainer
│  RM: frozen scorer              │
│  Est: 3–6 GPU hrs per run       │
└─────────────┬───────────────────┘
              │ aligned model
              ▼
┌─────────────────────────────────┐
│   Streamlit UI                  │
│  - Live inference               │
│  - Confidence visualization     │
│  - Over-escalation flag:        │
│    predicted sev vs filed sev   │  ← inference-time comparison only
│  - Feedback loop → CSV → RM     │
└─────────────────────────────────┘
```

### Model Output Format
```
severity:P1 | team:platform
```

### Over-Escalation Detection (inference time only)
No training label required. No escalation head in the model. At inference:
- Engineer provides the ticket text **and** the severity they originally filed
- Model predicts the severity the text actually warrants
- If `filed_severity < predicted_severity` (e.g. filed P0, model says P2) → **over-escalation flagged**
- If `filed_severity > predicted_severity` (e.g. filed P3, model says P1) → **under-escalation flagged**

---

## Repository Structure

```
incident-triage/
├── CLAUDE.md
├── README.md
├── FIXES.md                        # log of all critical bugs fixed during training
├── requirements.txt
├── requirements-dev.txt            # macOS local dev (no bitsandbytes)
├── data/
│   ├── raw/                        # downloaded dataset files (gitignored)
│   │   └── gitbugs.csv             # manually assembled from github.com/av9ash/gitbugs
│   ├── processed/                  # cleaned, labeled parquet splits (gitignored)
│   └── preference_pairs/           # DPO/RM training data
├── configs/
│   ├── sft_llama.yaml              # base SFT config
│   ├── sft_llama_oversampling.yaml # SFT config with oversampling output_dir
│   ├── dpo.yaml                    # DPO training config
│   ├── reward_model.yaml
│   └── ppo.yaml
├── src/
│   ├── data/
│   │   ├── load_datasets.py        # load Eclipse + GitBugs, concat, stratified split
│   │   ├── label_engineering.py    # priority normalization, team labels, text cleaning
│   │   ├── format_prompts.py       # wrap samples in instruction template, tokenize
│   │   └── build_preference_pairs.py  # generate (prompt, chosen, rejected) from predictions
│   ├── models/
│   │   ├── sft_trainer.py                      # original QLoRA SFT training loop
│   │   ├── sft_trainer_weighted_loss.py         # SFT + per-class token weighting
│   │   ├── sft_trainer_weighted_oversampling.py # SFT + weighting + mild oversampling ← best SFT
│   │   ├── dpo_trainer.py                       # DPO alignment (replaces RM+PPO)
│   │   ├── reward_trainer.py       # reward model architecture + TRL RewardTrainer
│   │   └── ppo_trainer.py          # TRL PPOTrainer setup + rollout loop
│   └── eval/
│       ├── metrics.py              # severity macro-F1, team accuracy, confusion matrix
│       └── confusion_matrix.py     # severity confusion matrix + off-diagonal analysis
├── scripts/
│   ├── slurm/
│   │   ├── run_sft_train.sh
│   │   ├── run_sft_train_weighted.sh
│   │   ├── run_sft_train_oversampling.sh
│   │   ├── run_dpo.sh
│   │   ├── inference.sh
│   │   ├── inference_weighted_parse.sh
│   │   ├── run_reward.sh
│   │   └── run_ppo.sh
│   └── merge_lora.py               # merge LoRA adapters into base model for inference
├── app/
│   └── streamlit_app.py
├── feedback/                       # user correction logs (gitignored)
└── notebooks/
    ├── 01_eda.ipynb                # dataset exploration + label distribution
    ├── 02_label_analysis.ipynb     # priority normalization validation, team coverage
    └── 03_eval_analysis.ipynb      # final metrics, confusion matrices, SFT vs DPO
```

---

## Datasets

| Dataset | Source | Notes |
|---|---|---|
| Eclipse Bugzilla | `AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset` (HuggingFace) | 88K rows; severity labels; all FIXED resolution; no timestamps |
| GitBugs | Manual download from `github.com/av9ash/gitbugs` | 9 projects (Cassandra, Firefox, Hadoop, etc.); explicit Priority + Resolution fields; merge ZIPs into `data/raw/gitbugs.csv` |
| Apache Bug Reports | **NOT USED** — commented out | Removed; Eclipse + GitBugs sufficient |

**GitBugs download:** Clone or download ZIP from `github.com/av9ash/gitbugs`, combine per-project CSVs, save to `data/raw/gitbugs.csv` with columns: `id, title, body, priority, resolution, created_at, closed_at, project`.

---

## Full Stack

| Layer | Tool | Version |
|---|---|---|
| Base model | LLaMA 3.2 1B (`meta-llama/Llama-3.2-1B`) | Latest HF |
| Fine-tuning | HuggingFace Transformers | >=4.40 |
| PEFT / QLoRA | PEFT + BitsAndBytes | >=0.10 |
| RLHF / PPO | TRL (Transformer RL) | >=0.8 |
| Datasets | HuggingFace datasets | >=2.18 |
| Data wrangling | pandas + scikit-learn | Latest |
| UI | Streamlit | >=1.32 |
| Eval | scikit-learn + matplotlib | Latest |
| Infra | IU BigRed200 A100 40GB | SLURM |
| Experiment tracking | Weights & Biases (wandb) | Latest |

**BigRed200 install:**
```bash
module load python/3.11 cuda/12.1
python3.11 -m venv venv && source venv/bin/activate
pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Local dev (macOS, no GPU):**
```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt
```

Key A100 training flags to always set:
- `gradient_checkpointing=True` — saves ~40% VRAM
- `use_reentrant=False` — required with gradient checkpointing + PEFT
- `bf16=True` — bfloat16 (not fp16) on A100 for stability
- `dataloader_num_workers=4` — prevents CPU bottleneck on BigRed200
- `save_total_limit=3` — disk fills fast; keep only last 3 checkpoints

---

## Label Engineering Details

### Priority Normalization

**Eclipse severity → priority:**
```
blocker → P0 | critical → P1 | major → P2 | normal → P3 | minor/trivial → P4
```

**GitBugs `GITBUGS_PRIORITY_MAP`:**
```python
{
    "HIGH": "P1", "NORMAL": "P3", "LOW": "P4", "URGENT": "P0",
    "P1": "P0", "P2": "P1", "P3": "P2", "P4": "P3", "P5": "P4",
    "BLOCKER": "P0", "CRITICAL": "P1", "MAJOR": "P2", "MINOR": "P3", "TRIVIAL": "P4",
}
```

### Team Buckets
Keyword-matched from `component`/`product` fields to 7 buckets:
`platform`, `database`, `frontend`, `backend`, `infra`, `security`, `mobile`

Rows where team resolves to `unknown` are **dropped before training** — they add noise without a clear label. Spot-check the unknown bucket before each training run to see if additional keywords should be added.

---

## Class Imbalance Handling

### Per-Class Token Weighting
Applied in `sft_trainer_weighted_loss.py` and `sft_trainer_weighted_oversampling.py`.
Higher weight = model penalized more for getting that severity digit wrong.

```python
SEVERITY_TOKEN_WEIGHTS = {
    "P0": 2.0,   # rarest + most critical
    "P1": 1.5,   # rare + high severity
    "P2": 1.2,   # moderately rare
    "P3": 1.0,   # majority class — baseline
    "P4": 1.5,   # rare but often confused with P3
}
```

Weights applied only to the severity digit token (e.g. `'0'` in `severity:P0`) at completion positions. Prompt tokens are masked to -100 and excluded from loss entirely.

### Mild Oversampling
Applied in `sft_trainer_weighted_oversampling.py` before training.
Each minority class is upsampled (with replacement) to reach 30% of the majority class (P3) count.

```
P3: ~72K  (majority, untouched)
P0:  ~2.5K → ~21.6K  (+19K synthetic copies)
P1:  ~9K   → ~21.6K  (+12K synthetic copies)
P4:  ~6K   → ~21.6K  (+15K synthetic copies)
P2: no oversampling needed (already above target)
Total: ~124K → ~140K rows after oversampling
```

Target ratio is configurable via `OVERSAMPLE_TARGET_RATIO = 0.30` at the top of the file.

---

## Task Breakdown

Tasks are grouped by phase. Work through them in order — each phase depends on the previous.
`[x]` = done, `[ ]` = pending.

---

### Phase 0: Environment & Repository Setup ✅

- [x] **0.1** Create the full directory tree as defined in Repository Structure above
- [x] **0.2** Create `requirements.txt` and `requirements-dev.txt`
- [x] **0.3** Set up Python 3.11 virtual environment (`python3.11 -m venv venv`) and install dependencies
- [x] **0.4** Create a W&B project named `incident-triage`; run `wandb login`
- [x] **0.5** Confirm BigRed200 access: ssh in, load CUDA module, activate venv
- [x] **0.6** Create `.gitignore` — excludes `data/raw/`, `data/processed/`, checkpoints, `*.pt`, `*.safetensors`, `venv/`, `feedback/`, `.env`

---

### Phase 1: Data Pipeline ✅

#### 1A — Loading
- [x] **1.1** Write `src/data/load_datasets.py`
  - Load Eclipse Bugzilla via `load_dataset("AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset")`
  - Load GitBugs from `data/raw/gitbugs.csv` (optional, skipped if file not found)
  - Apache dataset is intentionally excluded
  - Concatenate into unified pandas DataFrame; unified columns: `id, source, title, body, raw_severity, raw_priority, resolution, project, created_at, closed_at, resolution_time_days`
  - Drop rows where `team == "unknown"` before splitting
  - Print shape and split distributions to confirm load

#### 1B — Label Engineering
- [x] **1.2** Write priority normalization in `src/data/label_engineering.py`
  - Eclipse: severity string → P0–P4 via `SEVERITY_TO_PRIORITY` map
  - GitBugs: priority string → P0–P4 via `GITBUGS_PRIORITY_MAP` (see above)
  - No null priorities allowed after normalization
- [x] **1.3** Write team label engineering
  - Regex keyword-match on `title + body + project` to 7 team buckets
  - Rows with `team="unknown"` dropped in `load_datasets.py:main()` before split

#### 1C — Text Cleaning
- [x] **1.4** Write text cleaning in `src/data/label_engineering.py`
  - Strip HTML/XML tags with regex
  - Truncate stack traces: keep only first 10 lines of any traceback
  - Normalize whitespace (tabs, multiple spaces, Windows newlines)
  - Truncate combined `title + body` to 400 tokens

#### 1D — Splitting & Formatting
- [x] **1.5** Stratified 80/10/10 split on `priority` in `load_datasets.py`
  - Saves `train.parquet`, `val.parquet`, `test.parquet` to `data/processed/`
- [x] **1.6** Write `src/data/format_prompts.py`
  - Instruction template (no escalation field):
    ```
    ### Incident report:
    {title}
    {body}
    ### Triage:
    severity:{priority} | team:{team}
    ```
  - Tokenizes with LLaMA 3.2 tokenizer (falls back to TinyLlama without HF_TOKEN)
  - Saves HuggingFace `DatasetDict` to `data/processed/`
- [x] **1.7** Write `notebooks/01_eda.ipynb`
  - Source distribution, priority distribution, team distribution
  - Text length histograms, resolution time histogram (GitBugs)
  - Sample tickets per priority level
- [x] **1.8** Write `notebooks/02_label_analysis.ipynb`
  - Priority normalization validation (raw → P0–P4 mapping spot-check)
  - Team label coverage (% unknown before drop)
  - Example tickets per team bucket

---

### Phase 2: SFT Training ✅

#### 2A — Model & Config
- [x] **2.1** Write `configs/sft_llama.yaml`
  ```yaml
  model_name: meta-llama/Llama-3.2-1B
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_use_double_quant: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  lora_target_modules: [q_proj, k_proj, v_proj, o_proj]
  lora_bias: none
  num_train_epochs: 3
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4   # effective batch = 32
  learning_rate: 2e-4
  lr_scheduler_type: cosine
  warmup_ratio: 0.05
  weight_decay: 0.01
  max_grad_norm: 1.0
  optim: paged_adamw_32bit
  max_seq_length: 512
  bf16: true
  tf32: true
  gradient_checkpointing: true
  gradient_checkpointing_kwargs:
    use_reentrant: false
  output_dir: checkpoints/sft
  save_total_limit: 3
  report_to: wandb
  run_name: sft-llama-3.2-1b
  dataloader_num_workers: 4
  dataloader_pin_memory: true
  ```
- [x] **2.2** Write `src/models/sft_trainer.py`
  - BitsAndBytes 4-bit NF4 config, bfloat16 compute dtype, double quant
  - `prepare_model_for_kbit_training()` before `get_peft_model()`
  - `model.config.use_cache = False`, `model.config.pretraining_tp = 1`
  - TRL `SFTTrainer` with `formatting_func` (no pre-tokenization needed)
  - Falls back to TinyLlama if LLaMA 3.2 inaccessible (local testing)
  - Saves LoRA adapters + tokenizer to `checkpoints/sft/`

#### 2B — SLURM
- [x] **2.3** Write `scripts/slurm/run_sft.sh`
  - `#SBATCH --gres=gpu:a100:1 --mem=80G --time=04:00:00`
  - Validates `HF_TOKEN` and `WANDB_API_KEY` before starting
  - Runs: `python -m src.models.sft_trainer --config configs/sft_llama.yaml`

#### 2C — Validation
- [x] **2.4** Write `src/eval/metrics.py`
  - `parse_output(text)` → `(severity, team)` — regex `severity:(P[0-4])\s*\|\s*team:(\w+)`
  - Returns `(None, None)` for malformed output — never crashes
  - `compute_metrics()`: severity macro-F1, severity accuracy, team accuracy, per-class F1, confusion matrix, parse failure rate
  - `run_eval_on_checkpoint()`: full PeftModel inference loop for standalone eval
- [x] **2.5** Run SFT evaluation on the validation set after training on BigRed200
  - Logged per-head metrics to W&B
  - **Best checkpoint: `checkpoints/sft_weighted_oversample`**

  | Run | Parse Fail | Macro-F1 | Sev Acc | Team Acc | P0 F1 | P4 F1 | Notes |
  |---|---|---|---|---|---|---|---|
  | Zero-shot baseline | 100% | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 | |
  | Few-shot baseline | 17.1% | 0.5139 | 0.6458 | 0.5124 | 0.38 | 0.17 | |
  | SFT v1 | 18.7% | 0.4468 | 0.5829 | 0.6493 | 0.24 | 0.16 | |
  | Weighted loss | 17.8% | 0.4719 | 0.6157 | 0.6969 | 0.26 | 0.13 | |
  | **Weighted + oversampling** | **12.0%** | **0.5708** | **0.6812** | **0.8178 ✅** | **0.41** | **0.41** | best SFT checkpoint |
  | Manual threshold (P0=2.0, P3=0.7, P4=1.8) | **0.0%** | 0.2615 | 0.2611 | 0.9380 | 0.37 | 0.38 | parse fail eliminated but macro-F1 collapsed — P3 F1=0, scales too aggressive |
  | **Bayes log-prior correction** | **0.0%** | **0.6656 ↑** | **0.7668** | **0.9380 ✅** | **0.4486** | **0.8422 ↑** | best overall — no extra training, inference-only fix |

  Per-class F1 for log-prior run (val set, checkpoint-18504):
  | Class | F1 | Notes |
  |---|---|---|
  | P0 | 0.4486 | ↑ from 0.41 — rare class, still hardest |
  | P1 | 0.5315 | moderate improvement |
  | P2 | 0.6641 | solid |
  | P3 | 0.8418 | majority class, expected high |
  | P4 | **0.8422** | ↑ from 0.41 — huge gain from prior correction |

  - Team accuracy target **met** (0.9380 > 0.78) ✅ — log-prior correction also improved team accuracy (no more parse failures)
  - Severity macro-F1: **0.6656** (target 0.72, gap = 0.054) — DPO to close this
  - Manual threshold: catastrophic — P3 F1 collapsed to 0 (too aggressive, vacuumed predictions into P4)
  - Bayes log-prior correction (Menon et al. 2013): `logit_i - log(prior_i)` at inference time using **train set** priors (no data leakage)
  - Priors: P0=0.0385, P1=0.0746, P2=0.1785, P3=0.5141, P4=0.1944 (computed from 123,836 train rows)
  - Next: run DPO (`sbatch scripts/slurm/run_dpo.sh`) — expect 0.05–0.10 macro-F1 lift from preference alignment

---

### Phase 3: Reward Model

#### 3A — Preference Data
- [ ] **3.1** Write `src/data/build_preference_pairs.py` (new file)
  - Positive examples: SFT model outputs that exactly match ground-truth `severity|team`
  - Negative examples: wrong severity or wrong team routing
  - Hard negatives: adjacent severity confusion (P1 vs P2, P2 vs P3) — most informative
  - Ratio: 1 positive : 1 negative, 20% of negatives are hard negatives
  - Save as `data/preference_pairs/train.jsonl` and `val.jsonl`
  - Each row: `{"prompt": "...", "chosen": "severity:P1 | team:platform", "rejected": "severity:P0 | team:platform"}`

#### 3B — Model & Training
- [ ] **3.2** Write `configs/reward_model.yaml`
  - Same base model as SFT; lr=1e-5, batch size=4, 1–2 epochs
- [ ] **3.3** Write `src/models/reward_trainer.py`
  - Load LLaMA 3.2 1B; replace LM head with `nn.Linear(hidden_size, 1)`
  - Use TRL `RewardTrainer` with Bradley-Terry pairwise loss
  - Reward signal weights (severity + team only — no escalation):
    - Correct severity: +1.0
    - Correct team: +0.5
    - Wrong severity by 1 level: -0.5
    - Wrong severity by 2+ levels: -1.5
    - Wrong team: -1.0
    - Malformed output: -0.5
  - Log reward mean/std and pairwise accuracy to W&B
  - Save to `checkpoints/reward_model/`
- [ ] **3.4** Write `scripts/slurm/run_reward.sh` (same SLURM pattern as SFT)

#### 3C — Validation
- [ ] **3.5** Validate reward model on held-out preference pairs
  - Report pairwise accuracy: % of pairs where RM score(chosen) > score(rejected)
  - Target > 70% pairwise accuracy before proceeding to PPO

---

### Phase 4: PPO Training

#### 4A — Setup
- [ ] **4.1** Write `configs/ppo.yaml`
  ```yaml
  kl_penalty: 0.1
  cliprange: 0.2
  vf_coef: 0.1
  ent_coef: 0.01
  target_kl: 0.05
  ppo_epochs: 4
  mini_batch_size: 4
  batch_size: 128
  learning_rate: 1e-5
  total_ppo_steps: 1000
  output_dir: checkpoints/ppo
  ```
- [ ] **4.2** Write `src/models/ppo_trainer.py`
  - Policy model: SFT checkpoint + LoRA adapters (unfrozen), `AutoModelForCausalLMWithValueHead`
  - Reference model: same SFT checkpoint, frozen (for KL penalty)
  - Reward model: Stage 2 checkpoint, frozen
  - Rollout loop: generate 128 outputs per step, score with RM, run PPO update
  - W&B logging every step: reward mean/std, KL divergence, entropy, policy loss, value loss
  - Alert: KL > 0.1 → log warning; KL > 0.2 → stop run

#### 4B — Stability Monitoring
- [ ] **4.3** Add stability safeguards in `src/models/ppo_trainer.py`
  - Monitor per-class severity output distribution every 50 steps (catch reward hacking: model outputs P0 on everything to maximize reward)
  - If any single severity class > 70% of outputs, alert to W&B and reduce entropy bonus
  - Save best checkpoint by severity macro-F1, not just reward mean

#### 4C — SLURM
- [ ] **4.4** Write `scripts/slurm/run_ppo.sh` with `--time=08:00:00` (PPO is slow)

---

### Phase 5: Evaluation

- [ ] **5.1** Write `src/eval/confusion_matrix.py`
  - Generate severity confusion matrix on test set
  - Highlight critical off-diagonal cells:
    - P0 predicted as P3+ (critical miss — high severity ignored)
    - P2/P3 predicted as P0 (over-escalation — noise floods oncall)
    - P4 predicted as P0 (worst false alarm)
  - Save as matplotlib figure to `results/confusion_matrix.png`
- [ ] **5.2** Implement baseline comparisons
  - **Rule-based baseline**: regex severity keywords + component → team keyword map; no ML
  - **SFT-only baseline**: SFT checkpoint results from task 2.5
  - **PPO model**: final checkpoint from Stage 3
  - Side-by-side table of all three across: severity macro-F1, severity accuracy, team accuracy
- [ ] **5.3** Write `notebooks/03_eval_analysis.ipynb`
  - Load all baseline results, display comparison table
  - Show confusion matrices side by side (rule-based vs SFT vs PPO)
  - Plot reward improvement curve over PPO training steps (from W&B)
  - Highlight PPO reward improvement % over SFT baseline (target: >15%)
- [ ] **5.4** Run final evaluation on held-out test set with PPO model
  - Targets: severity macro-F1 > 0.72, team accuracy > 0.78

---

### Phase 6: LoRA Merge & Model Export

- [ ] **6.1** Write `scripts/merge_lora.py`
  - Load base LLaMA 3.2 1B
  - Load LoRA adapters from PPO checkpoint
  - Call `model.merge_and_unload()` to bake adapters into base weights
  - Save merged model to `checkpoints/merged/` in HF format
  - Verify: run a single inference call on a test incident; confirm output parses correctly

---

### Phase 7: Streamlit UI

- [ ] **7.1** Write `app/streamlit_app.py` — model loading
  - `@st.cache_resource` to load merged model once
  - Load with `device_map="auto"`, `torch_dtype=torch.bfloat16`
- [ ] **7.2** Write inference + output parsing
  - `parse_triage_output(text)` — extract `(severity, team)` from the generated string
  - Accept optional `filed_severity` input to enable over-escalation comparison
  - Handle malformed outputs gracefully: show "parse error" card, do not crash
- [ ] **7.3** Build result display
  - Severity badge: color-coded (P0=red, P1=orange, P2=yellow, P3=green, P4=gray)
  - Team name display
  - **Over-escalation banner**: shown when `filed_severity < model_predicted_severity` (red)
  - **Under-escalation banner**: shown when `filed_severity > model_predicted_severity` (yellow)
  - Softmax token probabilities as a horizontal bar chart via `st.bar_chart`
- [ ] **7.4** Build feedback panel
  - Thumbs up / thumbs down buttons
  - If thumbs down: show correction dropdowns for severity and team
  - On submit: append `{timestamp, input, filed_severity, model_output, correction}` to `feedback/feedback_log.csv`
  - This CSV can feed future RM retraining runs
- [ ] **7.5** Build history sidebar
  - Store last 10 triage decisions in `st.session_state`
  - Display as clickable list; clicking reloads that incident into the input box

---

### Phase 8: Documentation

- [ ] **8.1** Write `README.md`
  - Project overview (3 sentences)
  - Architecture diagram (same as above)
  - Setup instructions (venv, Python 3.11, BigRed200 access, HF_TOKEN, WANDB_API_KEY)
  - How to run each stage (data pipeline / SFT / RM / PPO / eval / UI)
  - GitBugs download instructions
  - Link to W&B project for live experiment logs
- [ ] **8.2** Add docstrings to all public functions in `src/`
- [ ] **8.3** Add `configs/README.md` explaining every YAML field and valid ranges

---

## Key Metrics (Targets)

| Metric | Target | Status |
|---|---|---|
| Severity macro-F1 | > 0.72 | **0.6656** (Bayes log-prior, val) — DPO to close gap |
| Team routing accuracy | > 0.78 | **0.9380 ✅ achieved** |
| Alignment improvement over SFT | > 15% macro-F1 gain | pending DPO (target: 0.6656 → 0.72+) |

Over-escalation detection is evaluated at inference time only — no training metric.
It is reported in the UI as a comparison between `filed_severity` and `model_predicted_severity`.

---

## Local Testing Before BigRed200

**Smoke test (macOS, no GPU):**
```bash
# 1. Run data pipeline
python -m src.data.load_datasets

# 2. Dry-run SFT with TinyLlama (edit yaml temporarily):
#    num_train_epochs: 1, max_steps: 5, per_device_train_batch_size: 1, report_to: none
python -m src.models.sft_trainer --config configs/sft_llama.yaml
# Falls back to TinyLlama automatically if HF_TOKEN not set
# Confirms: data loading → formatting → tokenization → forward pass → checkpoint save
```

**Files to rsync to BigRed200:**
```bash
rsync -av --exclude='venv/' --exclude='data/raw/' --exclude='data/processed/' \
  --exclude='checkpoints/' --exclude='wandb/' --exclude='*.pyc' \
  ./ username@bigred200.uits.iu.edu:~/triage/
```
Then re-run `python -m src.data.load_datasets` on BigRed200 to generate parquets there.

**Required before submitting SFT job:**
```bash
export HF_TOKEN=hf_...         # request access at huggingface.co/meta-llama/Llama-3.2-1B
export WANDB_API_KEY=...       # from wandb.ai/settings
mkdir -p logs checkpoints/sft
sbatch scripts/slurm/run_sft.sh
```

---

## PPO Failure Modes & Fixes

| Symptom | Likely Cause | Fix |
|---|---|---|
| KL divergence explodes | LR too high or beta too low | Reduce LR to 5e-6; increase KL beta to 0.2 |
| Policy collapse (all outputs identical) | Entropy too low | Increase entropy bonus from 0.01 to 0.05 |
| RM scores everything near 0 | RM overfit | Retrain RM with 2x more hard negatives |
| Severity macro-F1 plateaus below 0.65 | P0/P1 class imbalance | Increase reward weight for P0/P1 correct predictions |
| Model outputs same severity for everything | Reward hacking | Monitor per-class output distribution; alert if any class > 70% |

---

## Dataset Links

| Resource | ID / URL |
|---|---|
| Eclipse Bugzilla | `AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset` (HuggingFace) |
| GitBugs | `github.com/av9ash/gitbugs` — download ZIP, merge project CSVs into `data/raw/gitbugs.csv` |
| LLaMA 3.2 1B | `meta-llama/Llama-3.2-1B` (HuggingFace, gated — request access first) |
| TRL docs | https://huggingface.co/docs/trl |
| PEFT docs | https://huggingface.co/docs/peft |
