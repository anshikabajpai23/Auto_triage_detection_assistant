# Incident Severity Auto-Triage — CLAUDE.md

## Project Overview

Fine-tune LLaMA 3.2 1B on 238K real-world bug reports to produce a **3-head classifier** in a single forward pass:
- **Severity** — P0–P4
- **Team routing** — platform / database / frontend / backend / infra / security / mobile
- **Over-escalation flag** — true / false (novel contribution; no existing triage dataset has this)

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
│  - Over-escalation labels       │  ← novel, derived from resolution metadata
│  - Text cleaning + tokenization │
│  - Stratified train/val/test    │
└─────────────┬───────────────────┘
              │ 238K labeled prompt-completion pairs
              ▼
┌─────────────────────────────────┐
│   Stage 1: SFT (QLoRA)          │
│  Base: LLaMA 3.2 1B             │
│  Adapters: r=16, alpha=32       │  ← teaches format + baseline accuracy
│  Output: severity|team|esc      │
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
│  - Feedback loop → CSV → RM     │
└─────────────────────────────────┘
```

### Multi-Head Output Format
```
severity:P1 | team:platform | escalation:false
```
All three outputs are one token sequence — the reward model scores the entire string as a unit.

### Over-Escalation Label Rule (3 conditions, ALL must hold)
1. Filed priority is P0 or P1
2. Resolution is `wontfix`, `invalid`, `worksforme`, `duplicate`, or `minor`
   — OR filed severity is blocker/critical AND resolved severity is normal/minor/trivial
3. Resolution time > 7 days

Expected positive rate: 8–12% of all tickets.

---

## Repository Structure

```
incident-triage/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                        # downloaded dataset files (gitignored)
│   ├── processed/                  # cleaned, labeled, tokenized HF datasets
│   └── preference_pairs/           # reward model training data
├── configs/
│   ├── sft_llama.yaml
│   ├── reward_model.yaml
│   └── ppo.yaml
├── src/
│   ├── data/
│   │   ├── load_datasets.py        # load Eclipse + GitBugs, concat
│   │   ├── label_engineering.py    # priority normalization, team labels, escalation labels
│   │   └── format_prompts.py       # wrap samples in instruction template, tokenize
│   ├── models/
│   │   ├── sft_trainer.py          # QLoRA SFT training loop
│   │   ├── reward_trainer.py       # reward model architecture + TRL RewardTrainer
│   │   └── ppo_trainer.py          # TRL PPOTrainer setup + rollout loop
│   └── eval/
│       ├── metrics.py              # macro-F1, escalation precision/recall/F1
│       └── confusion_matrix.py     # severity confusion matrix + off-diagonal analysis
├── scripts/
│   ├── slurm/
│   │   ├── run_sft.sh
│   │   ├── run_reward.sh
│   │   └── run_ppo.sh
│   └── merge_lora.py               # merge LoRA adapters into base model for inference
├── app/
│   └── streamlit_app.py
└── notebooks/
    ├── 01_eda.ipynb                # dataset exploration + label distribution
    ├── 02_label_analysis.ipynb     # escalation label validation, class balance
    └── 03_eval_analysis.ipynb      # final metrics, confusion matrices, SFT vs PPO
```

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

Install command:
```bash
pip install transformers peft trl bitsandbytes datasets \
            accelerate wandb streamlit scikit-learn pandas matplotlib
```

Key A100 training flags to always set:
- `gradient_checkpointing=True` — saves ~40% VRAM
- `use_reentrant=False` — required with gradient checkpointing + PEFT
- `bf16=True` — bfloat16 (not fp16) on A100 for stability
- `dataloader_num_workers=4` — prevents CPU bottleneck on BigRed200
- `save_total_limit=3` — disk fills fast; keep only last 3 checkpoints

---

## Task Breakdown

Tasks are grouped by phase. Work through them in order — each phase depends on the previous.

---

### Phase 0: Environment & Repository Setup

- [ ] **0.1** Create the full directory tree as defined in Repository Structure above
- [ ] **0.2** Create `requirements.txt` with all pinned dependencies
- [ ] **0.3** Set up Python virtual environment and install dependencies
- [ ] **0.4** Create a W&B project named `incident-triage`; run `wandb login`
- [ ] **0.5** Confirm BigRed200 access: ssh in, load CUDA module, activate venv
- [ ] **0.6** Create `.gitignore` — exclude `data/raw/`, `data/processed/`, checkpoints, `*.pt`, `*.safetensors`, `venv/`, `.env`

---

### Phase 1: Data Pipeline

#### 1A — Loading
- [ ] **1.1** Write `src/data/load_datasets.py`
  - Load Eclipse Bugzilla via `load_dataset("SinaAhmadi/eclipse_bugzilla")`
  - Load GitBugs manually from CSV files; document download instructions in README
  - Concatenate into a single unified pandas DataFrame with consistent column names: `id`, `title`, `body`, `priority`, `severity`, `component`, `product`, `resolution`, `created_at`, `closed_at`
  - Print shape and sample rows to confirm load

#### 1B — Label Engineering
- [ ] **1.2** Write priority normalization in `src/data/label_engineering.py`
  - Map Eclipse severity: `blocker→P0`, `critical→P1`, `major→P2`, `normal→P3`, `minor/trivial/enhancement→P4`
  - Map GitBugs priority fields to the same P0–P4 scale
  - Assert no nulls remain in the normalized priority column
- [ ] **1.3** Write team label engineering
  - Keyword-match `component` and `product` fields to 7 team buckets: `platform`, `database`, `frontend`, `backend`, `infra`, `security`, `mobile`
  - Assign `unknown` to anything that doesn't match; review the `unknown` bucket manually before training
  - Spot-check 200 random samples; document accuracy of the mapping in a comment
- [ ] **1.4** Write over-escalation label engineering (the novel part)
  - Implement the 3-condition rule exactly as specified
  - Compute `resolution_time_days = (closed_at - created_at).dt.days`
  - Add `is_over_escalated` boolean column
  - Assert positive rate is between 6% and 15%; raise an error if outside this range
  - Log class distribution counts so it's visible in output

#### 1C — Text Cleaning
- [ ] **1.5** Write text cleaning in `src/data/label_engineering.py` (or a helper)
  - Strip HTML/XML tags with regex
  - Truncate stack traces: keep only the first 10 lines of any traceback
  - Normalize whitespace (tabs, multiple spaces, Windows newlines)
  - Truncate combined `title + body` to 400 tokens (count roughly as `len(text.split())`)

#### 1D — Splitting & Formatting
- [ ] **1.6** Write stratified train/val/test split (80/10/10) in `src/data/load_datasets.py`
  - Stratify on `severity × is_over_escalated` jointly
  - Assert escalation positives appear in all three splits
  - Print split sizes and class distributions
- [ ] **1.7** Write `src/data/format_prompts.py`
  - Wrap each sample in the instruction template:
    ```
    ### Incident report:
    {title}\n{body}
    ### Triage:
    severity:{P} | team:{team} | escalation:{true/false}
    ```
  - Tokenize using `AutoTokenizer` from LLaMA 3.2; pad/truncate to 512 tokens
  - Save as HuggingFace `DatasetDict` to `data/processed/` using `save_to_disk`
- [ ] **1.8** Write `notebooks/01_eda.ipynb`
  - Plot severity distribution, team distribution, escalation rate
  - Plot resolution time histogram
  - Show 5 example escalation-positive tickets to verify label quality
- [ ] **1.9** Write `notebooks/02_label_analysis.ipynb`
  - Show confusion between original priority and normalized P0–P4
  - Validate team label coverage (% with `unknown`)
  - Show escalation positive/negative example pairs side by side

---

### Phase 2: SFT Training

#### 2A — Model & Config
- [ ] **2.1** Write `configs/sft_llama.yaml` with all hyperparameters:
  ```yaml
  model_name: meta-llama/Llama-3.2-1B
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: [q_proj, v_proj, k_proj, o_proj]
  learning_rate: 2e-4
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  num_train_epochs: 3
  lr_scheduler_type: cosine
  warmup_ratio: 0.05
  max_seq_length: 512
  weight_decay: 0.01
  bf16: true
  gradient_checkpointing: true
  save_total_limit: 3
  output_dir: checkpoints/sft
  ```
- [ ] **2.2** Write `src/models/sft_trainer.py`
  - Load BitsAndBytes config (4-bit NF4, bfloat16 compute dtype, double quant)
  - Load LLaMA 3.2 1B with `quantization_config` and `device_map="auto"`
  - Apply LoRA via `get_peft_model` with `LoraConfig`
  - Use TRL `SFTTrainer` (handles packing and prompt formatting automatically)
  - Initialize W&B run; log all hyperparams
  - Save final LoRA adapter weights to `checkpoints/sft/`

#### 2B — SLURM
- [ ] **2.3** Write `scripts/slurm/run_sft.sh`
  ```bash
  #!/bin/bash
  #SBATCH --job-name=triage-sft
  #SBATCH --partition=gpu
  #SBATCH --gres=gpu:a100:1
  #SBATCH --mem=80G
  #SBATCH --time=04:00:00
  #SBATCH --output=logs/sft_%j.out
  module load python/3.11 cuda/12.1
  source venv/bin/activate
  python src/models/sft_trainer.py --config configs/sft_llama.yaml
  ```

#### 2C — Validation
- [ ] **2.4** Write a standalone `src/eval/metrics.py`
  - Parse model output string `severity:P1 | team:platform | escalation:false`
  - Compute per-head predictions vs ground truth
  - Compute: severity macro-F1, team routing accuracy, escalation precision / recall / F1
  - Handle malformed output gracefully (count as wrong, do not crash)
- [ ] **2.5** Run SFT evaluation on the validation set after training
  - Log per-head metrics to W&B
  - Target: severity macro-F1 > 0.72, escalation F1 > 0.60 (SFT baseline, not final target)

---

### Phase 3: Reward Model

#### 3A — Preference Data
- [ ] **3.1** Write preference pair construction (add to `src/data/label_engineering.py` or a new file)
  - Positive examples: SFT model outputs that exactly match ground truth triage
  - Negative examples: over-escalations the SFT model missed + wrong team routing
  - Hard negatives: incidents where severity is P1 vs P2 (adjacent classes, most informative)
  - Ratio: 1 positive : 1 negative, 20% hard negatives of the negative pool
  - Save as `data/preference_pairs/train.jsonl` and `val.jsonl`
  - Each row: `{"prompt": "...", "chosen": "severity:P1|...", "rejected": "severity:P0|..."}`

#### 3B — Model & Training
- [ ] **3.2** Write `configs/reward_model.yaml`
  - Same base model, lr=1e-5, batch size=4, 1–2 epochs
- [ ] **3.3** Write `src/models/reward_trainer.py`
  - Load LLaMA 3.2 1B; replace LM head with `nn.Linear(hidden_size, 1)`
  - Use TRL `RewardTrainer` with Bradley-Terry loss
  - Implement the reward signal weights:
    - Correct severity: +1.0
    - Correct team: +0.5
    - Correct escalation flag: +2.0
    - Missed over-escalation: -2.5
    - False escalation flag: -1.0
    - Malformed output: -0.5
  - Log reward mean/std and pairwise accuracy to W&B
  - Save to `checkpoints/reward_model/`
- [ ] **3.4** Write `scripts/slurm/run_reward.sh` (same pattern as SFT script)

#### 3C — Validation
- [ ] **3.5** Validate reward model on held-out preference pairs
  - Report pairwise accuracy: % of pairs where RM prefers chosen over rejected
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
  - Load policy model: SFT checkpoint with LoRA adapters (unfrozen, use `AutoModelForCausalLMWithValueHead` from TRL)
  - Load reference model: same SFT checkpoint, frozen (for KL penalty)
  - Load reward model: Stage 2 checkpoint, frozen
  - Set up `PPOTrainer` with config
  - Write rollout loop: generate 128 outputs per step, score with RM, run PPO update
  - W&B logging every step: reward mean/std, KL divergence, entropy, policy loss, value loss
  - Alert logic: if KL > 0.1, log a warning; if KL > 0.2, stop the run

#### 4B — Stability Monitoring
- [ ] **4.3** Add stability safeguards in `src/models/ppo_trainer.py`
  - Monitor per-class escalation output rate every 50 steps (catch reward hacking: model outputs `escalation:true` on everything)
  - If escalation rate > 60% of all outputs, alert to W&B and reduce entropy bonus
  - Save best checkpoint by escalation F1, not just reward mean

#### 4C — SLURM
- [ ] **4.4** Write `scripts/slurm/run_ppo.sh` with `--time=08:00:00` (PPO is slow)

---

### Phase 5: Evaluation

- [ ] **5.1** Write `src/eval/confusion_matrix.py`
  - Generate severity confusion matrix on test set
  - Highlight the three critical off-diagonal cells:
    - P0 predicted as P3+ (critical miss)
    - P2 predicted as P0 (over-escalation)
    - P4 predicted as P0 (worst false alarm)
  - Save as a matplotlib figure to `results/confusion_matrix.png`
- [ ] **5.2** Implement baseline comparisons
  - **Rule-based baseline**: regex priority keywords + component → team keyword map; no ML
  - **SFT-only baseline**: run the SFT checkpoint through the same eval (already done in 2.5; just load results)
  - **PPO model**: final checkpoint from Stage 3
  - Produce a side-by-side table of all three across all metrics
- [ ] **5.3** Write `notebooks/03_eval_analysis.ipynb`
  - Load all baseline results, display comparison table
  - Show confusion matrices side by side (rule-based vs SFT vs PPO)
  - Plot escalation F1 curve over PPO training steps (from W&B)
  - Highlight the PPO reward improvement % over SFT baseline (target: >15%)
- [ ] **5.4** Run final evaluation on held-out test set with PPO model
  - Targets: severity macro-F1 > 0.72, team accuracy > 0.78, escalation F1 > 0.75

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
  - `parse_triage_output(text)` — extract severity, team, escalation from the generated string
  - Handle malformed outputs gracefully: show "parse error" card, do not crash
- [ ] **7.3** Build result display
  - Severity badge: color-coded (P0=red, P1=orange, P2=yellow, P3=green, P4=gray)
  - Team name display
  - Escalation warning banner (red banner if `escalation:true`)
  - Softmax probabilities as a horizontal bar chart using `st.bar_chart`
- [ ] **7.4** Build feedback panel
  - Thumbs up / thumbs down buttons
  - If thumbs down: show correction dropdowns for severity, team, escalation
  - On submit: append `{timestamp, input, model_output, correction}` to `feedback/feedback_log.csv`
  - This CSV feeds future RM retraining
- [ ] **7.5** Build history sidebar
  - Store last 10 triage decisions in `st.session_state`
  - Display as clickable list; clicking reloads that incident into the input box

---

### Phase 8: Documentation

- [ ] **8.1** Write `README.md`
  - Project overview (3 sentences)
  - Architecture diagram (text-based, same as in this file)
  - Setup instructions (venv, dependencies, BigRed200 access)
  - How to run each stage (SFT / RM / PPO / eval / UI)
  - Link to W&B project for live experiment logs
  - Resume bullet templates (from design doc Section 13)
- [ ] **8.2** Add docstrings to all public functions in `src/`
- [ ] **8.3** Add a `configs/README.md` explaining every YAML field and valid ranges

---

## Key Metrics (Targets)

| Metric | Target |
|---|---|
| Severity macro-F1 | > 0.72 |
| Team routing accuracy | > 0.78 |
| Escalation precision | > 0.80 |
| Escalation recall | > 0.70 |
| **Escalation F1** | **> 0.75** (headline metric) |
| PPO reward improvement over SFT | > 15% |

---

## PPO Failure Modes & Fixes

| Symptom | Likely Cause | Fix |
|---|---|---|
| KL divergence explodes | LR too high or beta too low | Reduce LR to 5e-6; increase KL beta to 0.2 |
| `escalation:true` on >60% outputs | Reward hacking (+2.0 signal dominated) | Increase false positive penalty to -2.0 |
| Policy collapse (all outputs identical) | Entropy too low | Increase entropy bonus from 0.01 to 0.05 |
| RM scores everything near 0 | RM overfit | Retrain RM with 2x more hard negatives |
| Escalation recall < 0.60 | Focal loss insufficient | Increase gamma from 2.0 to 3.0; increase oversample ratio |

---

## Dataset Links

| Resource | ID / URL |
|---|---|
| Eclipse Bugzilla | `SinaAhmadi/eclipse_bugzilla` (HuggingFace) |
| GitBugs | Search arXiv: "GitBugs dataset priority prediction" |
| LLaMA 3.2 1B | `meta-llama/Llama-3.2-1B` (HuggingFace) |
| TRL docs | https://huggingface.co/docs/trl |
| PEFT docs | https://huggingface.co/docs/peft |
