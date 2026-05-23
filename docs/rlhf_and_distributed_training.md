# RLHF, Policy Optimization & Distributed Training
## From Basics to Interview-Level Depth

---

## PART 1: FOUNDATIONS

### What is RLHF?

RLHF (Reinforcement Learning from Human Feedback) is a training paradigm that aligns a language model's outputs with human preferences or task-specific goals. Instead of only training on "what the correct answer is" (supervised learning), RLHF trains on "which answer is better" — a fundamentally different signal.

**Why do we need it?**
Supervised fine-tuning (SFT) teaches the model to imitate the training data. But imitation has a ceiling:
- The model can only be as good as the data it saw
- It doesn't learn to reason about *why* one output is better than another
- It can't improve beyond the quality of human annotations

RLHF lifts that ceiling by letting the model explore, get scored, and update toward better outputs.

**The classic RLHF pipeline (InstructGPT / ChatGPT style):**
```
Step 1: SFT       — fine-tune on human-written examples
Step 2: RM        — train a reward model on human preference pairs
Step 3: PPO       — optimize the policy against the reward model using RL
```

---

### What is a Policy in LLM Context?

In RL, a **policy** is the function that maps states to actions. In LLM context:
- **State** = the prompt (input tokens)
- **Action** = the next token to generate
- **Policy** = the language model itself (the probability distribution over next tokens)
- **Reward** = a scalar score for the complete generated output

When we say "training the policy," we mean updating the LLM's weights so it generates higher-reward outputs.

---

### What is the KL Divergence Penalty?

KL (Kullback-Leibler) divergence measures how much one probability distribution differs from another.

In RLHF, after SFT we have a reference model (frozen copy of SFT). During RL training, we add a penalty:

```
total_reward = task_reward - β × KL(policy || reference)
```

**Why?**
Without this penalty, the policy will find degenerate solutions — outputs that score high on the reward metric but are nonsensical (reward hacking). The KL penalty keeps the policy close to the SFT model, ensuring outputs remain coherent.

**β (beta) controls the tradeoff:**
- High β → stay close to SFT, conservative improvement
- Low β → more freedom to deviate, higher risk of reward hacking

**In this project:** `beta=0.04` — low penalty because our reward is verifiable (exact string match), so reward hacking is less likely than with a learned reward model.

---

## PART 2: THE THREE STAGES IN THIS PROJECT

### Stage 1: SFT (Supervised Fine-Tuning)

**What it does:**
Trains the model to output `severity:P1 | team:platform` given an incident report. Pure imitation learning — we show it examples and it learns the format and baseline accuracy.

**Key technique: QLoRA**
- Full fine-tuning a 1B model requires ~16GB VRAM for weights alone + optimizer states
- QLoRA (Quantized LoRA) loads the base model in 4-bit NF4 precision (~500MB) and adds small trainable adapter matrices (LoRA)
- Only the LoRA adapters (~3.4M params out of 1.24B) are trained
- 97.25% of the model is frozen in 4-bit

**LoRA (Low-Rank Adaptation):**
Instead of updating the full weight matrix W, LoRA adds:
```
W_new = W_frozen + A × B
```
Where A (d × r) and B (r × k) are small matrices with rank r (r=16 here). This reduces trainable params from d×k to r×(d+k).

**Completion-only loss:**
We use `DataCollatorForCompletionOnlyLM` which masks the prompt tokens — loss is only computed on the triage output (`severity:P1 | team:platform`), not on the incident report text. This prevents the model from wasting capacity memorizing the input format.

**Weighted cross-entropy:**
P0/P1 incidents are rare (class imbalance). We upweight their loss contribution so the model doesn't just learn to always predict P3 (majority class).

---

### Stage 2A: DPO (Direct Preference Optimization)

**What it does:**
Given preference pairs (chosen: correct triage, rejected: wrong triage), DPO directly optimizes the policy to prefer the chosen output over the rejected one — without needing a separate reward model.

**How it differs from PPO:**
PPO requires:
1. A reward model (separate trained network)
2. A value function (critic)
3. Complex rollout + update loop

DPO collapses all of this into a single supervised loss:
```
L_DPO = -log σ(β × (log π(chosen|x)/π_ref(chosen|x) - log π(rejected|x)/π_ref(rejected|x)))
```

The math shows that the optimal policy under RLHF *implicitly* satisfies this equation — DPO trains to that optimum directly.

**Reference model in DPO:**
DPO needs a frozen reference model to compute `π_ref`. With PEFT/LoRA, instead of loading two separate models, TRL toggles the LoRA adapters on/off:
- Adapters ON → policy model (trainable)
- Adapters OFF → reference model (base model = SFT baseline)

**Why DPO underperformed here:**
- Only 7,477 preference pairs (small dataset)
- Pairs built from SFT val mistakes — limited diversity
- +0.007 macro-F1 improvement over SFT (marginal)
- DPO works best when preference pairs cover diverse failure modes

---

### Stage 2B: GRPO (Group Relative Policy Optimization)

**What it does:**
For each prompt, generates `num_generations=8` completions, scores each with a rule-based reward, and trains the policy to increase the probability of higher-reward outputs relative to the group average.

**Key insight — why GRPO over DPO for this task:**
Our reward is *verifiable* — we know exactly if `severity:P1 | team:platform` is correct or not. This is different from open-ended tasks where human preference is subjective. When reward is deterministic and checkable, GRPO's rule-based reward outperforms DPO's learned preference signal.

**GRPO vs PPO:**
PPO requires a separate value function (critic) that estimates future rewards. GRPO replaces this with the *group mean*:

```
advantage_i = (reward_i - mean(group_rewards)) / std(group_rewards)
```

- No value function needed
- No reward model needed (rule-based reward replaces it)
- Simpler to implement and more stable

**The reward function:**
```
+1.0  correct severity
+0.2  severity off by 1 level (P1 vs P2)
-0.5  severity off by 2+ levels
-1.0  extra penalty: true=P0/P1 but predicted P3/P4 (critical miss)
+0.5  correct team
-0.5  parse failure (malformed output)
```

**Why these weights:**
- Critical misses (P0 filed as P3) are the worst production failure — extra penalty
- Team routing is valuable but secondary to severity — lower weight
- Parse failure penalized but not catastrophically — model can recover

**GRPO training dynamics:**
- 8 generations per prompt → group of 8 reward scores
- Completions with above-average reward get positive advantage (reinforced)
- Completions with below-average reward get negative advantage (suppressed)
- KL penalty (β=0.04) prevents drift from SFT reference

---

## PART 3: DISTRIBUTED TRAINING

### Why Do We Need It?

Single GPU timeline for this project:
- 123,836 training prompts
- 8 generations per prompt
- ~1.92s per step at batch_size=8
- **66 hours for 2 epochs on 1 GPU** — exceeds 7-hour SLURM limit

Distributed training splits the work across multiple GPUs.

---

### Option 1: DDP (Data Distributed Parallel) — What We're Using

**How it works:**
- Each GPU gets a full copy of the model
- Each GPU processes a different batch of data independently
- After each backward pass, gradients are averaged across all GPUs (AllReduce)
- All GPUs apply the same gradient update → models stay in sync

```
GPU 0: batch [0-7]   → gradients → AllReduce → update
GPU 1: batch [8-15]  → gradients → AllReduce → update
GPU 2: batch [16-23] → gradients → AllReduce → update
GPU 3: batch [24-31] → gradients → AllReduce → update
```

**Speedup:** ~4x fewer steps per epoch with 4 GPUs (123,836 / 32 = 3,870 steps/epoch vs 15,480)

**Why DDP works here:**
- LLaMA 1B in 4-bit = ~500MB — easily fits 4 copies on 4× A100 40GB
- LoRA has only 3.4M trainable params — tiny gradient communication overhead
- Simple to set up with `torchrun`

**Key config change:**
```python
# Each process maps to its own GPU
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)
device_map = {"": local_rank}
```

Without `torch.cuda.set_device(local_rank)`, processes 1-3 fail with "CUDA device busy or unavailable" because their CUDA context was never initialized before weight loading.

**Launch command:**
```bash
torchrun --nproc_per_node=4 -m src.models.grpo_trainer --config configs/grpo.yaml
```

---

### Option 2: DeepSpeed ZeRO

DeepSpeed is Microsoft's distributed training library. ZeRO (Zero Redundancy Optimizer) eliminates redundancy across GPUs.

**Three ZeRO stages:**

| Stage | What's sharded | Memory saving | Use case |
|---|---|---|---|
| ZeRO-1 | Optimizer states | ~4x | Large optimizer states |
| ZeRO-2 | Optimizer states + gradients | ~8x | Medium models |
| ZeRO-3 | Everything (params + optimizer + gradients) | ~64x | Very large models (70B+) |

**ZeRO-2 for this project:**
- Shards optimizer states and gradients across 4 GPUs
- For 3.4M LoRA trainable params: optimizer states = ~13MB → ~3MB per GPU after sharding
- **In practice: negligible benefit** — 3.4M params is tiny, sharding 13MB gives no real speedup
- Adds scatter/gather communication overhead

**ZeRO-3 issues with PEFT + LoRA:**
ZeRO-3 shards model parameters. When LoRA adapters are applied, the parameter gathering during forward/backward pass conflicts with how PEFT manages adapter weights in some versions of TRL. Can cause shape mismatch errors or incorrect gradient computation.

**DeepSpeed with bitsandbytes 4-bit:**
Works with ZeRO-2 (model params stay whole per GPU). ZeRO-3 is problematic because sharding quantized weights requires dequantize → shard → requantize which bitsandbytes doesn't support.

**Conclusion for this project:**
DeepSpeed ZeRO-2 adds complexity with no meaningful benefit for a 1B model with 3.4M trainable LoRA params. The 66→4 hour speedup comes entirely from 4-GPU data parallelism, achievable with plain DDP.

---

### Option 3: FSDP (Fully Sharded Data Parallel)

PyTorch's native answer to DeepSpeed ZeRO-3. Shards model parameters, gradients, and optimizer states across GPUs.

**Why not used here:**
- FSDP + bitsandbytes 4-bit quantization = **incompatible** in most configurations
- FSDP tries to shard quantized tensors, which bitsandbytes doesn't support
- Would require dropping 4-bit quantization → model no longer fits in training memory
- More complex configuration than DDP or DeepSpeed

**When FSDP makes sense:**
Large unquantized models (7B+ in bf16) where model doesn't fit on a single GPU and you don't want the DeepSpeed dependency.

---

### Comparison Table

| Method | Memory | Speed | Complexity | 4-bit compat | Best for |
|---|---|---|---|---|---|
| DDP | 4x model copies | ~4x on 4 GPUs | Low | ✅ | Small-medium models, LoRA |
| DeepSpeed ZeRO-2 | Less per GPU | ~4x on 4 GPUs | Medium | ✅ | Medium models, large optimizer states |
| DeepSpeed ZeRO-3 | Much less per GPU | ~4x on 4 GPUs | High | ⚠️ issues | Very large models (30B+) |
| FSDP | Much less per GPU | ~4x on 4 GPUs | High | ❌ | Large unquantized models |

---

## PART 4: INTERVIEW QUESTIONS & ANSWERS

### Foundational

**Q: What is the difference between SFT and RLHF?**

SFT trains the model to imitate a fixed dataset via cross-entropy loss — the model learns "what was written." RLHF uses a reward signal to train the model toward "what is better" — it can explore beyond the training data distribution. SFT sets a baseline; RLHF refines toward a quality target.

---

**Q: Why do we need a KL penalty in RLHF?**

Without KL penalty, the policy exploits the reward function — it finds degenerate outputs that score high numerically but aren't useful (reward hacking). For a learned reward model, this means outputs that fool the reward model rather than being genuinely good. The KL penalty anchors the policy to the SFT distribution, ensuring coherence while allowing improvement.

---

**Q: What is LoRA and why is it used?**

LoRA (Low-Rank Adaptation) injects small trainable matrices into frozen model layers. Instead of updating all weights W (d×k), it trains A (d×r) and B (r×k) where r << d,k. This reduces trainable parameters by ~99% while preserving most of the fine-tuning benefit. Used here because full fine-tuning of LLaMA 1B requires >40GB VRAM; LoRA with 4-bit quantization (QLoRA) fits in 8GB.

---

**Q: What is 4-bit NF4 quantization?**

NF4 (Normal Float 4-bit) is a quantization format that represents weights using 4 bits instead of 16 (bf16) or 32 (fp32). NF4 uses a non-uniform quantization grid optimized for normally-distributed weights (which neural network weights typically follow). Combined with double quantization (quantizing the quantization constants themselves), it achieves ~4x memory compression with minimal accuracy loss.

---

### Intermediate

**Q: What is the difference between DPO and PPO?**

PPO requires: (1) a trained reward model, (2) a value function (critic), (3) rollout generation, (4) advantage estimation, (5) clipped policy gradient update. It's sample-inefficient and unstable.

DPO collapses this pipeline into a single supervised loss on preference pairs. It proves mathematically that the optimal RLHF policy satisfies a specific equation, then trains directly to that optimum. No RM, no value function, no rollouts — just a binary cross-entropy style loss on (chosen, rejected) pairs.

**Tradeoff:** DPO is simpler but requires high-quality preference pairs. PPO is more complex but can improve beyond the quality of existing pairs.

---

**Q: What is GRPO and how does it differ from PPO?**

PPO uses a learned value function to estimate the baseline reward (how good is this state on average?). GRPO replaces the value function with the empirical mean of a *group* of generations from the same prompt. For each prompt, generate G completions, score them all, then:

```
advantage_i = (reward_i - mean(rewards)) / std(rewards)
```

This eliminates the need for a separate value network, simplifies training, and is particularly effective when the reward is rule-based and verifiable (as in triage classification).

---

**Q: What is reward hacking and how do you detect it?**

Reward hacking is when the policy learns to maximize the reward metric without actually improving at the intended task. Signs in classification:
- Model outputs the same class for every input (mode collapse)
- Any single severity class exceeds 70% of outputs
- Reward mean increases but evaluation accuracy doesn't

In this project: monitor per-class output distribution every N steps. Alert if any class exceeds 70% of outputs — this indicates the model learned the majority class is the highest-expected-reward output rather than genuinely triaging.

---

**Q: Why is `use_reentrant=False` needed with gradient checkpointing + PEFT?**

Gradient checkpointing saves memory by not storing intermediate activations — recomputing them during backward pass. The `use_reentrant=True` (old default) implementation uses Python's autograd hooks and conflicts with PEFT's LoRA adapter parameter management, causing incorrect gradient computation or errors. `use_reentrant=False` uses PyTorch's newer non-reentrant implementation which is compatible with custom parameter structures like LoRA adapters.

---

**Q: In DPO, how is the reference model handled with PEFT?**

Loading two full models (policy + reference) would double VRAM. With PEFT, there's only one copy of the base model. TRL handles the reference by toggling LoRA adapters:
- Forward pass with adapters ON → policy model output
- Forward pass with adapters OFF → base model output = reference

Since only the adapters are trained and the base model is frozen, the base model with adapters disabled is equivalent to the SFT reference. This halves VRAM usage compared to loading two separate models.

---

### Advanced

**Q: Why does DDP require `device_map={"": local_rank}` instead of `device_map="auto"` for GRPO?**

`device_map="auto"` performs *model parallelism* — it splits a single model across all available GPUs on one process. This is appropriate when a single model doesn't fit on one GPU.

In DDP, each process should own its entire model on one GPU. Using `device_map="auto"` in a torchrun DDP context causes each process to attempt to claim all GPUs, resulting in conflicts. `device_map={"": local_rank}` explicitly maps the entire model to the process's assigned GPU, enabling correct DDP behavior.

Additionally, `torch.cuda.set_device(local_rank)` must be called before model loading to initialize the CUDA context on that GPU, otherwise bitsandbytes/safetensors fail to load weights onto uninitialized devices.

---

**Q: Why is ZeRO-3 incompatible with LoRA + bitsandbytes?**

ZeRO-3 shards model parameters across GPUs — each GPU holds 1/N of the parameters at rest, gathering full parameter tensors only during forward/backward passes. This requires parameters to be gathered, used, then discarded (freed) after each layer.

Two problems arise:
1. **bitsandbytes 4-bit quantization:** The gather/scatter operations require dequantizing and requantizing weights at each layer across the network boundary. bitsandbytes quantization is designed for in-place weight storage, not cross-process movement of quantized tensors.

2. **LoRA parameter management:** PEFT wraps specific layers with LoRA adapters. ZeRO-3's parameter gathering conflicts with PEFT's hook-based adapter injection — the gathered full parameter may not include the LoRA delta, or the gradient flow through the gathered tensor bypasses LoRA's trainable A/B matrices.

---

**Q: How does GRPO handle the credit assignment problem differently from PPO?**

PPO's credit assignment: use a value function V(s) trained to estimate cumulative reward from state s. This value function is itself a neural network that must be trained, introducing instability and requiring careful hyperparameter tuning.

GRPO's credit assignment: generate G completions from the same prompt, measure the spread of rewards within that group. The group mean is an unbiased estimate of the baseline reward for that prompt. No learned approximation — the baseline is computed empirically from actual rollouts.

This works especially well for structured output tasks (like triage) where:
- The output space is small (35 possible severity×team combinations)
- Some completions will be correct and some wrong in any group
- The reward signal is immediately clear (no delayed reward problem)

---

**Q: What are the failure modes of GRPO and how do you monitor them?**

| Failure mode | Symptom in W&B | Cause | Fix |
|---|---|---|---|
| Reward hacking | reward_mean rises, F1 flat | Low KL penalty | Increase beta |
| Mode collapse | All outputs identical | Entropy too low | Increase entropy bonus |
| KL explosion | KL > 0.2 | LR too high | Reduce LR, increase beta |
| Severity bias | One class >70% outputs | Class imbalance | Increase reward for rare classes |
| Parse failure domination | All rewards = -0.5 | Model forgot SFT format | Reduce LR, re-check prompt template |

---

**Q: What is the tradeoff between num_generations in GRPO and training efficiency?**

More generations per prompt:
- **Better:** Lower variance in advantage estimates (more samples = better group mean)
- **Better:** Covers more of the output space (8 completions more likely to include both correct and incorrect)
- **Worse:** Higher VRAM (8x more sequences in memory during generation)
- **Worse:** Slower per step (8x more forward passes for generation)

The constraint is: `global_batch_size` (per_device_batch × num_gpus) must be divisible by `num_generations`. With `num_generations=8` and 4 GPUs, minimum `per_device_batch_size=2` (2×4=8, divisible by 8).

---

## PART 5: PROJECT-SPECIFIC DECISIONS SUMMARY

| Decision | Choice | Why |
|---|---|---|
| Base model | LLaMA 3.2 1B | Small enough for A100 40GB with QLoRA, strong baseline |
| Quantization | 4-bit NF4 + double quant | 4x memory reduction, minimal accuracy loss |
| Adapter | LoRA r=16, α=32 | 3.4M trainable params vs 1.24B total |
| Stage 1 | SFT with weighted loss | Correct format + handle P0/P1 class imbalance |
| Stage 2 | GRPO over DPO/PPO | Verifiable reward, no preference pairs needed, no RM needed |
| Reward | Rule-based | Deterministic, no reward model training, no reward hacking |
| KL beta | 0.04 (low) | Reward is verifiable → less need for conservative KL constraint |
| Distributed | DDP via torchrun | 1B model fits on each GPU, LoRA params too small for ZeRO benefit |
| Generations | 8 per prompt | Good signal diversity, fits A100 VRAM with batch_size=8 |
| Evaluation metric | Severity macro-F1 | Weighted equally across P0-P4, penalizes ignoring rare classes |
