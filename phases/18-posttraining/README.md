# Phase 18: Post-training: SFT, LoRA, reward models and RLHF

Training systems.

Connect supervised demonstrations, frozen-base LoRA, reward modeling, sampled PPO and direct preference optimization. Keep the reference, behavior policy, reward model and dataset identities separate.

Predict the mask, denominator or preference direction before training. Keep independent objective checks and changed-condition evidence.

**Prerequisites:** 17: Self-supervised pretraining: masked and contrastive learning; 12: Reinforcement learning.

**Hardware:** Executed CPU synthetic objectives; large-scale and human-feedback evaluation remain separate.

## Study guide: Which parameters and signals change during adaptation?

Connect supervised demonstrations, frozen-base LoRA, reward modeling, sampled PPO and direct preference optimization. Keep the reference, behavior policy, reward model and dataset identities separate.

### Check your starting point

Should response-only loss remove prompt tokens from the model context?

<details><summary>Compare your reasoning</summary>

No. Prompt inputs remain visible under the causal attention contract; the shifted response mask controls which targets receive loss.

</details>

Review: [Supervised fine-tuning with response-only token loss](01-sft/docs/en.md).

### Build in stages

1. **Supervised fine-tuning with response-only token loss.** Supervised fine-tuning learns from demonstrations. The example uses a tiny next-token table so the alignment is visible: prompt IDs determine answer IDs, followed by an end token. In a pretrained Transformer, the same response-mask bookkeeping surrounds a much richer causal forward pass.

   Lessons: [Supervised fine-tuning with response-only token loss](01-sft/docs/en.md).

2. **LoRA: adapt, save and merge low-rank updates.** LoRA parameterizes a weight change as two smaller matrices. It reduces trainable parameter and optimizer-state counts for selected layers; it does not eliminate activation memory or guarantee a latency improvement. Our changed target is deliberately rank one, which makes the adaptation test exact and interpretable.

   Lessons: [LoRA: adapt, save and merge low-rank updates](02-lora/docs/en.md).

3. **Learn a reward model from pairwise preferences.** The reward model assigns a scalar to response features and fits pairwise comparisons using a logistic preference likelihood. All labels here are synthetic and declared. Training such a model from actual human feedback additionally requires a documented rating protocol, disagreement handling and independent evaluation.

   Lessons: [Learn a reward model from pairwise preferences](03-reward/docs/en.md).

4. **RLHF mechanics: a frozen reward and PPO policy updates.** This is a one-prompt, one-terminal-action preference bandit. It isolates the reward-model → rollout → policy-update loop used in RLHF. Synthetic preferences supply the reward model; no humans or language generations are used. Existing RL lessons cover trajectories and PPO in an environment; real language PPO adds sequence masks, value estimation and rollout infrastructure.

   Lessons: [RLHF mechanics: a frozen reward and PPO policy updates](04-rlhf/docs/en.md).

5. **Direct preference optimization and reference-corrected margins.** DPO compares policy log-probability differences with the same differences under a frozen reference. The lab treats each response as one categorical action to isolate the objective. For language responses, substitute the sum of valid response-token log probabilities, with a consistent tokenizer and mask; arbitrary mean-length normalization changes the objective.

   Lessons: [Direct preference optimization and reference-corrected margins](05-dpo/docs/en.md).

### Try a changed condition

Reward rises, but independently reviewed outputs get worse. What has been established?

<details><summary>Compare an approach</summary>

Only improvement against the learned reward. Investigate reward exploitation, preference coverage and independent task metrics; do not label the result human alignment.

</details>

**Symptom:** A PPO ratio is always one during multiple optimizer epochs.

**Check next:** Check whether old log probabilities are incorrectly recomputed from the changing policy.

### Decide what is ready

Complete stages 4–8 of training-methods. Retain shifted token-mask checks, frozen-base/merge evidence, preference oracles, signed PPO clipping and fixed-reference DPO comparisons.

### Further work

The PPO lab is a terminal synthetic preference bandit, not full language-model RLHF. Human preference collection, sequence PPO/value models, large-model PEFT and held-out human evaluation remain separate work.

## Lesson sequence

### 18.01 Supervised fine-tuning with response-only token loss

[Read the lesson](01-sft/docs/en.md) · [Run the code](01-sft/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Align predictions with their target tokens. Separate the attention mask from the loss mask.

**Evidence:** Keep shifted target/role masks, valid response-token counts, gradient support and the changed target-selection result.

**Checkpoint:** Should prompt inputs disappear when using response-only loss?

### 18.02 LoRA: adapt, save and merge low-rank updates

[Read the lesson](02-lora/docs/en.md) · [Run the code](02-lora/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Separate the frozen base from trainable adapters. Understand initialization and the first gradient.

**Evidence:** Keep base-training evidence, unchanged base arrays, adapter factors/rank/alpha, initial-gradient checks and reloaded/new-input merge parity.

**Checkpoint:** Does LoRA require updating the base weights during adapter training?

### 18.03 Learn a reward model from pairwise preferences

[Read the lesson](03-reward/docs/en.md) · [Run the code](03-reward/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Represent a preference as a comparison. Fit score differences with a stable likelihood.

**Evidence:** Keep synthetic label provenance, stable preference-loss and gradient references, offset/scale tests and the learned ordering.

**Checkpoint:** What do pairwise preferences identify directly?

### 18.04 RLHF mechanics: a frozen reward and PPO policy updates

[Read the lesson](04-rlhf/docs/en.md) · [Run the code](04-rlhf/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Keep three identities separate. Collect actions before optimizing them.

**Evidence:** Keep reward/reference identity, collected old probabilities, signed clipping checks, exact reward/KL history and detached behavior information.

**Checkpoint:** Which policy is refreshed for the next rollout batch?

### 18.05 Direct preference optimization and reference-corrected margins

[Read the lesson](05-dpo/docs/en.md) · [Run the code](05-dpo/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Compare chosen and rejected response log probabilities. Subtract the reference preference.

**Evidence:** Keep reference identity, response log-probability conventions, log(2) initialization, independent margin loss, reversed labels and frozen-reference checks.

**Checkpoint:** Does this DPO loop require a separately trained reward model and fresh policy rollouts?

## Phase project

An audited pretraining and adaptation objective toolkit.

**Demonstrate:** Complete stages 4–8 of training-methods. Retain shifted token-mask checks, frozen-base/merge evidence, preference oracles, signed PPO clipping and fixed-reference DPO comparisons.

Project status: implemented staged practice · [Open source](../../projects/training-methods/README.md). Use stages 4, 5, 6, 7, 8 for this phase.



[Primary documentation](https://docs.jax.dev/en/latest/).
