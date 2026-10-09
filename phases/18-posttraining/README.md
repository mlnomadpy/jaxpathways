# Phase 18: Post-training: SFT, LoRA, reward models and RLHF

Training systems.

Connect supervised demonstrations, frozen-base LoRA, reward modeling, sampled PPO and direct preference optimization. Keep the reference, behavior policy, reward model and dataset identities separate.

Predict the mask, denominator or preference direction before training. Keep independent objective checks and changed-condition evidence.

**Prerequisites:** 17: Self-supervised pretraining: masked and contrastive learning; 12: Reinforcement learning; 19: TPU Setup: Provisioning & Runtime Verification; 20: TPU Workflows: Launching Jobs & Checkpointed Experiments; 21: TPU Systems: Generations, Memory, Precision & XProf.

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

1. **Supervised fine-tuning with response-only token loss.** Write an input/next-target/role table, audit ragged likelihood sums, identify supervised positions in the gradient map, and demonstrate a context distinction the transition model cannot represent.

   Lessons: [Supervised fine-tuning with response-only token loss](01-sft/docs/en.md).

2. **LoRA: adapt, save and merge low-rank updates.** Derive and check both adapter-factor gradients, explain asymmetric initialization, verify nonunit merge scaling, and calculate a rank-one capacity limit before interpreting the adaptation curve.

   Lessons: [LoRA: adapt, save and merge low-rank updates](02-lora/docs/en.md).

3. **Learn a reward model from pairwise preferences.** Derive preference likelihood and its linear gradient, inspect contradictory labels, distinguish offsets from scale, and explain the measured comparison margins before handing off a reward model.

   Lessons: [Learn a reward model from pairwise preferences](03-reward/docs/en.md).

4. **RLHF mechanics: a frozen reward and PPO policy updates.** Identify current, behavior, reference and reward states; work both clipping signs; verify baseline cancellation by enumeration; and compare the trained bandit with its regularized optimum and probability plot.

   Lessons: [RLHF mechanics: a frozen reward and PPO policy updates](04-rlhf/docs/en.md).

5. **Direct preference optimization and reference-corrected margins.** Work through reference correction and its gradient, reproduce a falling chosen-probability counterexample, and explain why positive margins do not by themselves prove improved response behavior.

   Lessons: [Direct preference optimization and reference-corrected margins](05-dpo/docs/en.md).

### Try a changed condition

Reward rises, but independently reviewed outputs get worse. What has been established?

<details><summary>Compare an approach</summary>

Only improvement against the learned reward. Investigate reward exploitation, preference coverage and independent task metrics; do not label the result human alignment.

</details>

**Symptom:** A PPO ratio is always one during multiple optimizer epochs.

**Check next:** Check whether old log probabilities are incorrectly recomputed from the changing policy.

### Decide what is ready

Complete stages 4–8 of training-methods. Retain shifted token-mask checks, frozen-base/merge evidence, preference oracles, signed PPO clipping and fixed-reference DPO comparisons. Add the new lesson evidence: ragged likelihood oracle, rank-capacity counterexample, conflicting preferences, exact bandit optimum and a DPO ratio improvement with lower chosen probability. State which model integrations remain absent.

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

Project status: implemented staged practice · [Open source](../../projects/posttraining-audit/README.md). Use stages 1, 2, 3, 4 for this phase. Copy `projects/training-methods/starter/methods.py` to `projects/training-methods/my_methods.py` and write your code in `projects/training-methods/my_methods.py`. Run `python3 projects/training-methods/tests/check.py --implementation projects/training-methods/my_methods.py --stage 4` from the top-level folder to verify stages 4, 5, 6, 7, 8.

Additional project: [Audit pretraining and post-training objectives](../../projects/training-methods/README.md).

[Primary documentation](https://docs.jax.dev/en/latest/).
