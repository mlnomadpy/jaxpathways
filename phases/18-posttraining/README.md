# Phase 18: Post-training: SFT, LoRA, reward models and RLHF

Training systems.

Connect supervised demonstrations, frozen-base LoRA, reward modeling, sampled PPO and direct preference optimization. Keep the reference, behavior policy, reward model and dataset identities separate.

Predict the mask, denominator or preference direction before training. Keep independent objective checks and changed-condition evidence.

**Prerequisites:** 17: Self-supervised pretraining: masked and contrastive learning.

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

A preference update changes chosen/rejected log probabilities from \((-2,-3)\) to \((-3,-5)\), with the reference fixed. Did the chosen absolute probability increase?

<details><summary>Compare an approach</summary>

No. The log-probability margin improves from 1 to 2 while the chosen probability falls. DPO compares reference-corrected relative likelihoods. Record chosen, rejected and margin values separately, and use independent task/retention evaluation before calling the model better.

</details>

**Symptom:** A PPO ratio is always one during multiple optimizer epochs.

**Check next:** Check whether old log probabilities are incorrectly recomputed from the changing policy.

### Decide what is ready

Complete stages 4–8 of training-methods. Retain shifted token-mask checks, frozen-base/merge evidence, preference oracles, signed PPO clipping and fixed-reference DPO comparisons. Add the new lesson evidence: ragged likelihood oracle, rank-capacity counterexample, conflicting preferences, exact bandit optimum and a DPO ratio improvement with lower chosen probability. State which model integrations remain absent.

### Further work

The PPO lesson uses a terminal synthetic preference bandit. The optional, registered project Stage 5 table-policy exercise adds a separate synthetic sequence PPO exercise with a learned table critic and a declared deterministic continuation target at the collection boundary. The [optional saved-decoder appendix](https://www.tahabouhsine.com/jaxpathways/project-guides/posttraining-transformer-lineage.html) demonstrates bounded sequence-PPO mechanics and replay, but its strict heldout reward quality floor failed; this is not evidence of language-model quality. Keep it separate from registered Stage 5. Neither exercise is full language-model RLHF. Real preference provenance, large-model PEFT, task quality, retention and held-out human evaluation remain separate work. Complete rl-01 through rl-04 before the learned-reward PPO lesson; rl-05 is useful optional preparation for project Stage 5. Response-only SFT and dense LoRA do not require the RL phase.

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

Audit SFT, DPO and sequence PPO.

**Demonstrate:** Complete stages 4–8 of training-methods. Retain shifted token-mask checks, frozen-base/merge evidence, preference oracles, signed PPO clipping and fixed-reference DPO comparisons.

Project status: implemented staged practice · [Open source](../../projects/posttraining-audit/README.md). Use stages 1, 2, 3, 4, 5 for this phase. Begin in projects/posttraining-audit/starter/model.py for the response-mask and preference-pair checks. Stages 1 and 2 compare independent calculations and changed conditions; stage 3 retains evidence, and stage 4 integrates those checks. Stage 5 uses starter/sequence_ppo.py to collect synthetic responses, fit a critic, distinguish EOS from truncation, freeze rollout snapshots across PPO epochs and retain token-level evidence. Each stage lists its own edit file and command. This local mechanics exercise does not qualify a Transformer, human preferences, the full capstone or target hardware.

Additional project: [Audit pretraining and post-training objectives](../../projects/training-methods/README.md).

[Primary documentation](https://docs.jax.dev/en/latest/).
