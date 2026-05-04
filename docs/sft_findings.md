# SFT Phase Findings & Analysis
# Incident Severity Auto-Triage — Technical Document

---

## 1. Setup Summary

| Component | Value |
|---|---|
| Base model | LLaMA 3.2 1B (`meta-llama/Llama-3.2-1B`) |
| Quantization | 4-bit NF4 (QLoRA) via BitsAndBytes |
| LoRA | r=16, alpha=32, dropout=0.05, target: q/k/v/o_proj |
| Epochs | 3 |
| Effective batch size | 32 (per_device=8 × grad_accum=4) |
| Learning rate | 2e-4 (cosine scheduler, 5% warmup) |
| Train rows | 123,836 |
| Val rows | 15,479 |
| GPU | NVIDIA A100-SXM4-40GB |
| Optimizer | paged_adamw_32bit |

---

## 2. Training Data Format

### Input prompt
```
### Incident report:
{title}
{body}
### Triage (always output severity P0-P4 and one team only):
```

### Expected output (completion)
```
severity:{P0|P1|P2|P3|P4} | team:{platform|database|frontend|backend|infra|security|mobile|unknown}
```

### Full training string seen by model
```
### Incident report:
Startup crash running android sw-wr in debug build

webrender::device::gl::Device::new closure gl.rs:1417
### Triage (always output severity P0-P4 and one team only):
severity:P1 | team:mobile
```

---

## 3. Dataset Distribution

### Priority distribution (train)
| Priority | Count | % |
|---|---|---|
| P0 | ~2,400 | ~2% |
| P1 | ~5,400 | ~4% |
| P2 | ~15,500 | ~13% |
| P3 | ~72,000 | ~58% |
| P4 | ~28,500 | ~23% |

**Key issue: severe class imbalance.** P3 is 58% of training data. This directly causes the model to default to P3 for ambiguous inputs (blank body, short title).

### Team distribution (train)
| Team | Approx % |
|---|---|
| platform | 30% |
| backend | 22% |
| frontend | 18% |
| infra | 12% |
| database | 8% |
| mobile | 6% |
| security | 4% |

---

## 4. Baseline Results

### 4A — Zero-shot base (LLaMA 3.2 1B, no fine-tuning)

**Result: 100% parse failure, 0% accuracy on all metrics.**

The base model has no knowledge of the `severity:Px | team:y` output format. It generates natural language responses, stack traces, or random text — none of which the parser can extract a valid prediction from.

This result proves that SFT is necessary: the format itself must be learned before any classification can happen.

### 4B — Few-shot base (LLaMA 3.2 1B + 7 domain examples in prompt)

7 examples used (one per team), drawn from the training set:

| Example | Priority | Team |
|---|---|---|
| Assign name to error class | P3 | platform |
| spark.sql.timestampType inference | P2 | database |
| Intermittent async script test failure | P4 | frontend |
| ClientWaitSync over-aggressive warning | P0 | backend |
| WasmInstance assertion failure crash | P0 | infra |
| cqlsh TLS version default | P3 | security |
| Android sw-wr startup crash | P1 | mobile |

Results: **pending** (job submitted).

### 4C — SFT (LLaMA 3.2 1B + QLoRA, 3 epochs)

Evaluated on full val set (15,479 rows):

| Metric | Value | Target |
|---|---|---|
| Severity macro-F1 | 0.5060 | > 0.72 |
| Severity accuracy | 0.6427 | — |
| Team accuracy | 0.6029 | > 0.78 |
| Parse failure rate | 17.1% | ~0% |

#### Per-class severity F1
| Class | F1 | Observation |
|---|---|---|
| P0 | 0.3532 | Weak — critical miss risk |
| P1 | 0.4950 | Below average |
| P2 | 0.6339 | Reasonable |
| P3 | 0.8774 | Strong (majority class) |
| P4 | 0.1704 | Very weak |

#### Severity confusion matrix
```
         P0    P1    P2    P3    P4
  P0   187    49   185    66     2
  P1   128   471   236   241     0
  P2   133    85  1714   586     4
  P3    10   141   423  7295    13
  P4     5     2    88   483   282
```

#### Critical misses
- **true=P0 → pred=P3: 66** — highest severity ignored as routine
- **true=P1 → pred=P3: 241** — high severity downgraded to routine
- **true=P0 → pred=P2: 185** — severity significantly underestimated

---

## 5. Summary Comparison

| Model | Severity macro-F1 | Team accuracy | Parse failure |
|---|---|---|---|
| Base zero-shot | 0.0000 | 0.0000 | 100.0% |
| Base few-shot | TBD | TBD | TBD |
| **SFT** | **0.5060** | **0.6029** | **17.1%** |
| Target (after PPO) | >0.72 | >0.78 | ~0% |

**SFT contribution over zero-shot:**
- Parse failure: 100% → 17.1% (format learned)
- Severity macro-F1: 0.00 → 0.51
- Team accuracy: 0.00 → 0.60

---

## 6. Issues Found in SFT Output

### Issue 1: Repeated teams in raw output
```
severity:P3 | team:unknown | team:unknown | team:unknown
```
**Cause:** No stop signal after the team name. Model keeps generating in the same format pattern.

**Fix applied:** Added `\n` to end of `COMPLETION_TEMPLATE`. Model now learns to stop at newline. During inference, add newline token to `eos_token_id`.

### Issue 2: Hallucinated fields after team
```
severity:P2 | team:unknown | issue:unreachable | steps to reproduce: [none]
```
**Cause:** Model learned structured field patterns from bug report bodies (which contain "Steps to reproduce:", "Issue:", etc.) and continues generating them.

**Fix applied:** `\n` in completion template stops generation before these fields appear.

### Issue 3: Parse failures — input fragments repeated as output
Some outputs contain fragments of the input title/body instead of the expected format.

**Cause:** Prompt token masking may be unreliable with `formatting_func` in TRL 0.9.6. Loss may be computed over prompt tokens, confusing the model about where input ends and output begins.

**Fix planned:** Use `DataCollatorForCompletionOnlyLM` which explicitly masks prompt tokens with `-100`, ensuring loss is only computed on the completion.

### Issue 4: `team:unknown-platforms`, `team:frontend-ios`
**Cause:** Old issue from before `unknown` was added to `TEAM_LABELS`. Parser was returning `None` for these and they showed as parse failures. Now that `unknown` is accepted, these are classified separately.

### Issue 5: Blank body → defaults to P3
Tickets with empty body and short title default to P3 regardless of actual severity. Expected behavior given P3 is 58% of training data.

**Fix planned:** Focal loss to reduce majority-class bias. Optionally: fill blank body with "No additional details provided." at inference time.

---

## 7. SFT Improvements Applied

| Change | File | Status |
|---|---|---|
| Added `\n` to COMPLETION_TEMPLATE | sft_trainer.py, format_prompts.py | Done |
| Updated PROMPT_TEMPLATE with explicit instruction | sft_trainer.py, format_prompts.py | Done |
| Added `unknown` to TEAM_LABELS | src/eval/metrics.py | Done |
| Added prediction saving to eval | src/eval/metrics.py | Done |
| Added training sample saving | src/models/sft_trainer.py | Done |
| `remove_unused_columns=True` | src/models/sft_trainer.py | Done |

---

## 8. SFT Improvements Planned

| Change | Rationale | Priority |
|---|---|---|
| max_new_tokens: 20 → 15 | Output is ~10 tokens; 20 allows junk generation | High |
| newline stop token in generate() | Hard stop at `\n`, prevents repeated teams | High |
| DataCollatorForCompletionOnlyLM | Correct prompt masking, reduces parse failures | High |
| Focal loss / class-weighted loss | Addresses P0/P4 weakness from class imbalance | Medium |
| Domain-specific metrics | Critical miss rate, mean severity error, top-2 accuracy | Medium |
| Retrain with updated format | Prompt + completion template changed, needs fresh run | Required |

---

## 9. Next Steps

1. Collect few-shot baseline results and complete the comparison table
2. Implement remaining SFT improvements (max_new_tokens, stop token, DataCollator)
3. Retrain SFT with updated prompt format
4. Re-evaluate — target: reduce parse failure to <5%, improve P0 F1 to >0.5
5. Proceed to Phase 3: Reward Model
   - Build preference pairs from SFT outputs vs ground truth
   - Hard negatives: P0↔P1, P1↔P2 adjacent confusion cases
   - Reward weights: correct severity +1.0, correct team +0.5, wrong by 2+ levels -1.5
6. PPO will close remaining gap to target metrics (>0.72 severity F1, >0.78 team accuracy)
