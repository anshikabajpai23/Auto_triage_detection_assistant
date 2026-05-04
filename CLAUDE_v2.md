# Incident Severity Auto-Triage — CLAUDE_v2.md
# (Updated after SFT training, baseline runs, and SFT improvements)
# Original version: CLAUDE.md — DO NOT MODIFY THAT FILE

## What Changed in This Version

- SFT training completed (3 epochs, LLaMA 3.2 1B + QLoRA)
- Zero-shot and few-shot baselines run and recorded
- SFT eval results recorded
- SFT prompt and output format improved
- `unknown` team added as valid label
- Prediction saving added to eval pipeline
- Phase 2 tasks updated to reflect actual completion state

---

## Project Overview

Fine-tune LLaMA 3.2 1B on real-world bug reports to produce a **2-head classifier** in a single forward pass:
- **Severity** — P0–P4
- **Team routing** — platform / database / frontend / backend / infra / security / mobile / unknown

Over-escalation is detected **at inference time** by comparing the model's predicted severity against the severity the engineer originally filed — no training label needed, no escalation head in the model.

Training pipeline mirrors production LLM alignment: SFT → Reward Model → PPO.

---

## Model Output Format
```
severity:P1 | team:platform
```

## Over-Escalation Detection (inference time only)
- Engineer provides ticket text + severity they originally filed
- Model predicts the severity the text actually warrants
- `filed_severity < predicted_severity` → **over-escalation flagged**
- `filed_severity > predicted_severity` → **under-escalation flagged**

---

## Baseline Results (val set, 1000 samples for few-shot / 15,479 for others)

| Model | Severity macro-F1 | Team accuracy | Parse failure | Notes |
|---|---|---|---|---|
| Base zero-shot | 0.0000 | 0.0000 | 100% | No format knowledge |
| Base few-shot (7 domain examples) | TBD | TBD | TBD | Pending job result |
| **SFT (LLaMA 3.2 1B + QLoRA)** | **0.5060** | **0.6029** | **17.1%** | 3 epochs, 123K rows |
| Target | >0.72 | >0.78 | ~0% | After PPO |

### SFT Per-class Severity F1
| Class | F1 | Notes |
|---|---|---|
| P0 | 0.3532 | Critical — 66 P0s predicted as P3 |
| P1 | 0.4950 | 241 P1s predicted as P3 |
| P2 | 0.6339 | Reasonable |
| P3 | 0.8774 | Strong — majority class |
| P4 | 0.1704 | Weak — most predicted as P3 |

---

## SFT Improvements Made

### Prompt format updated
```
# OLD:
### Triage:

# NEW:
### Triage (always output severity P0-P4 and one team only):
```

### Completion template updated
```python
# OLD:
COMPLETION_TEMPLATE = "severity:{priority} | team:{team}"

# NEW:
COMPLETION_TEMPLATE = "severity:{priority} | team:{team}\n"
```
The trailing `\n` teaches the model to stop after the team name, preventing repeated teams and hallucinated fields like `issue:unreachable | steps to reproduce:`.

### Unknown team added to valid labels
`TEAM_LABELS` in `metrics.py` now includes `"unknown"` — model correctly outputs `team:unknown` for unclassifiable tickets, no longer treated as parse failure.

### Prediction saving added to eval pipeline
`run_eval_on_checkpoint()` now saves `results/eval_<split>_predictions.csv` with:
- `input_prompt`, `raw_output`, `pred_severity`, `pred_team`
- `sev_correct`, `team_correct` booleans per row

### Training sample saving added
`sft_trainer.py` saves 100 sample training inputs to `checkpoints/sft/sample_training_inputs.csv` before training starts — shows exact prompt + expected output format.

---

## SFT Improvements Planned (not yet implemented)

| Issue | Fix | Status |
|---|---|---|
| Repeated teams in raw output | `eos_token_id=[eos, newline_token]` in generate() | Pending |
| max_new_tokens too high | Change from 20 → 15 | Pending |
| Prompt masking unreliable with formatting_func | Use `DataCollatorForCompletionOnlyLM` | Pending |
| Class imbalance (P0/P4 weak) | Focal loss or class-weighted loss | Pending |
| Blank body → defaults to P3 | Add "No additional details provided." when body empty | Pending |
| Domain-specific metrics | Critical miss rate, mean severity error, top-2 accuracy | Pending |

---

## Known Issues / Observations from SFT Eval

1. **P3 bias** — model defaults to P3 when body is blank (P3 is majority class at 58% of train)
2. **P0/P1 critical misses** — 66 P0s predicted as P3, 241 P1s predicted as P3. PPO reward shaping targets this specifically
3. **P4 weakness** — most P4s predicted as P3. Adjacent class confusion, hard negatives in RM training will address this
4. **17.1% parse failure** — model sometimes generates input fragments instead of `severity:Px | team:y`. Prompt change + `\n` in completion should reduce this in next training run
5. **"Filed by" → P4** — model correctly learned that admin/meta language correlates with low severity. Valid signal, not a bug
6. **Blank body → P3** — not a bug, just majority-class default. Focal loss will reduce this bias

---

## Dataset

| Dataset | Source | Rows | Notes |
|---|---|---|---|
| Eclipse Bugzilla | `AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset` | 88K | All FIXED resolution; no timestamps |
| GitBugs | `github.com/av9ash/gitbugs` | ~35K | 9 projects; explicit Priority + Resolution |
| Apache | NOT USED | — | Commented out |

After label engineering and dropping `team=unknown`: **123,836 train / 15,479 val / 15,480 test**

---

## Environment

| Component | Version |
|---|---|
| Python | 3.11.13 |
| PyTorch | 2.3.1+cu121 |
| CUDA | 12.2 (cudatoolkit/12.2) |
| transformers | 4.44.0 |
| bitsandbytes | 0.43.3 |
| peft | 0.12.0 |
| trl | 0.9.6 |
| GPU | NVIDIA A100-SXM4-40GB |

---

## Task Breakdown

### Phase 0: Environment & Repository Setup ✅
- [x] 0.1 Directory tree created
- [x] 0.2 requirements.txt and requirements-dev.txt
- [x] 0.3 Python 3.11 venv + dependencies installed on BigRed200
- [x] 0.4 W&B project created, wandb login done
- [x] 0.5 BigRed200 access confirmed
- [x] 0.6 .gitignore created

### Phase 1: Data Pipeline ✅
- [x] 1.1 load_datasets.py — Eclipse + GitBugs, drop team=unknown
- [x] 1.2 Priority normalization (Eclipse severity → P0–P4, GITBUGS_PRIORITY_MAP)
- [x] 1.3 Team label engineering (7 buckets + unknown)
- [x] 1.4 Text cleaning (HTML strip, stack trace truncation, whitespace)
- [x] 1.5 Stratified 80/10/10 split on priority
- [x] 1.6 format_prompts.py with updated prompt template
- [x] 1.7 notebooks/01_eda.ipynb
- [x] 1.8 notebooks/02_label_analysis.ipynb

### Phase 2: SFT Training ✅

#### 2A — Model & Config
- [x] 2.1 configs/sft_llama.yaml (QLoRA, LoRA r=16/alpha=32, 3 epochs)
- [x] 2.2 src/models/sft_trainer.py (BitsAndBytes + LoRA + SFTTrainer + sample saving)
- [x] 2.3 scripts/slurm/run_sft.sh (A100, 4hr, HF_TOKEN + WANDB checks)

#### 2B — Evaluation
- [x] 2.4 src/eval/metrics.py (parse_output, compute_metrics, prediction saving)
- [x] 2.5 SFT eval on val set — severity macro-F1: 0.5060, team accuracy: 0.6029

#### 2C — Baselines
- [x] 2.6 Zero-shot base eval — 100% parse failure (confirms SFT teaches format)
- [x] 2.7 Few-shot base eval (7 domain examples, 1 per team) — results pending
- [x] 2.8 scripts/slurm/run_eval.sh, scripts/slurm/run_fewshot_base.sh

#### 2D — SFT Improvements (in progress)
- [x] 2.9 Added `\n` to COMPLETION_TEMPLATE — stops repeated team generation
- [x] 2.10 Updated PROMPT_TEMPLATE — explicit instruction for output format
- [x] 2.11 Added `unknown` to TEAM_LABELS in metrics.py
- [ ] 2.12 Change max_new_tokens from 20 → 15 in inference
- [ ] 2.13 Add newline stop token to generate() call
- [ ] 2.14 Use DataCollatorForCompletionOnlyLM for correct prompt masking
- [ ] 2.15 Add domain-specific metrics (critical miss rate, mean severity error)
- [ ] 2.16 Retrain with updated prompt/completion format

### Phase 3: Reward Model
- [ ] 3.1 src/data/build_preference_pairs.py
- [ ] 3.2 configs/reward_model.yaml
- [ ] 3.3 src/models/reward_trainer.py
- [ ] 3.4 scripts/slurm/run_reward.sh
- [ ] 3.5 Validate RM pairwise accuracy > 70%

### Phase 4: PPO Training
- [ ] 4.1 configs/ppo.yaml
- [ ] 4.2 src/models/ppo_trainer.py
- [ ] 4.3 PPO stability safeguards
- [ ] 4.4 scripts/slurm/run_ppo.sh

### Phase 5: Evaluation
- [ ] 5.1 src/eval/confusion_matrix.py
- [ ] 5.2 Full baseline comparison table (rule-based vs few-shot vs SFT vs PPO)
- [ ] 5.3 notebooks/03_eval_analysis.ipynb
- [ ] 5.4 Final test set evaluation

### Phase 6: LoRA Merge & Export
- [ ] 6.1 scripts/merge_lora.py

### Phase 7: Streamlit UI
- [ ] 7.1–7.5 app/streamlit_app.py

### Phase 8: Documentation
- [ ] 8.1–8.3 README.md, docstrings, configs/README.md

---

## Key Metrics (Targets)

| Metric | SFT Current | Target |
|---|---|---|
| Severity macro-F1 | 0.5060 | > 0.72 |
| Team routing accuracy | 0.6029 | > 0.78 |
| Parse failure rate | 17.1% | ~0% |
| **PPO reward improvement over SFT** | — | **> 15%** |

---

## PPO Failure Modes & Fixes

| Symptom | Likely Cause | Fix |
|---|---|---|
| KL divergence explodes | LR too high or beta too low | Reduce LR to 5e-6; increase KL beta to 0.2 |
| Policy collapse | Entropy too low | Increase entropy bonus from 0.01 to 0.05 |
| RM scores everything near 0 | RM overfit | Retrain RM with 2x more hard negatives |
| Severity macro-F1 plateaus below 0.65 | P0/P1 class imbalance | Increase reward weight for P0/P1 correct predictions |
| Model outputs same severity for everything | Reward hacking | Monitor per-class output distribution; alert if any class > 70% |

---

## Dataset Links

| Resource | URL |
|---|---|
| Eclipse Bugzilla | `AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset` (HuggingFace) |
| GitBugs | `github.com/av9ash/gitbugs` |
| LLaMA 3.2 1B | `meta-llama/Llama-3.2-1B` (HuggingFace, gated) |
| TRL docs | https://huggingface.co/docs/trl |
| PEFT docs | https://huggingface.co/docs/peft |
| W&B project | https://wandb.ai/bajpaianshika2310-indiana-university/huggingface |
