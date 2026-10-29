# A saved decoder enters sequence PPO

This is an optional saved-decoder bridge alongside the table-based Stage 5 sequence-PPO exercise. It is a separate, unregistered extension, not a replacement for that course stage. It continues the saved, two-layer, 10,880-parameter decoder from the text-harness foundations. It connects its real BPE processor and parameters to a small synthetic sequence-PPO objective. The source archive remains the parent run's inference-only package. This exercise makes a fresh child actor from those parameters and starts new actor and critic Adam state at the post-training objective boundary.

The objective is deliberately narrow. The synthetic prompts ask for a direction, and the only generated actions are the BPE single-token IDs for ` up` (383), ` down` (381), `!` (4), and `[EOS]` (2). These IDs were checked against the pinned parent tokenizer. A policy probability is normalized only across the eligible subset of these four actions. Every reference, old-policy, new-policy, sampled KL, and categorical KL calculation uses the same subset at a given prefix. EOS is ineligible until two non-EOS actions have been generated, so its probability is exactly zero before then. This is restricted-action sequence mechanics, not a full-vocabulary language-model objective.

The fixed prompts, exact token IDs, four synthetic preference pairs, and train/development/heldout assignments live in `fixture.json`. `recipe.json` binds the fixture and pins all three inference-package files plus the four foundation source files that define its loader and model. `check.py` checks those pins and a separate immutable approval before importing a candidate, initializing JAX state, or creating an output directory. The approved command must pass the exact approval file issued for these recipe bytes. A changed but valid JSON recipe or fixture is rejected before candidate code loads.

## What the stage adds

First, the frozen parent decoder scores each prompt prefix. A small reward head is fit on the four fixed chosen-versus-rejected pairs with a Bradley-Terry style pairwise loss. Its feature includes the outer product of the restricted parent-logit context and response action counts. The interaction is necessary because prompt-only additive features would cancel in a chosen-minus-rejected comparison for the same prompt. Once fitted, reward parameters are frozen. The heldout pairwise comparison is reported as a diagnostic; it is not used to select a model or change the recipe.

Next, copy the verified parent parameter tree to the trainable actor. Keep another exact parent copy as the frozen reference. Reset Adam moments and step counters for the new objective. The actor runs the actual causal decoder on the complete prefix before sampling each action. At position \(t\), it must not feed action \(a_t\) into the decoder until after it computes \(\pi(a_t\mid s_t)\). The saved prefix IDs make this ordering inspectable. When two sampled actions have different IDs, the next decoder logits must change with the new prefix.

The critic is a new learned linear head on stop-gradient restricted parent logits at each complete prefix. These are frozen Transformer-conditioned features, not hidden-state vectors and not a full vocabulary representation. For a sampled terminal EOS, the value bootstrap is zero. A response that reaches four actions without EOS is a timeout: score its observed partial response at its last generated action, evaluate the critic on the post-action prefix, and use that value in the last TD residual. Both EOS and timeout stop the reverse \(\lambda\) trace. Later padding has zero reward, advantage, target, policy loss, and critic loss.

The pairwise reward for the observed response is placed only on the last valid action. Each valid action also receives sampled rollout shaping

\[
r_t^{\mathrm{shaped}} = r_t^{\mathrm{terminal}} - \beta\left(\log\pi_{\mathrm{old}}(a_t\mid s_t)-\log\pi_{\mathrm{parent}}(a_t\mid s_t)\right).
\]

The code saves each component. It reports exact categorical KL separately, computed between the restricted distributions at contexts visited by the child rollout. This is a conditional per-token metric averaged over sampled visited contexts. Its per-sequence sum is not the divergence between full generated-sequence distributions.

For each sampled rollout, freeze old selected log probabilities and old critic values through both inner PPO epochs. Compute advantages and value targets once. The actor uses the signed clipped minimum

\[
L_{\mathrm{policy}}=-\frac{1}{N_{\mathrm{valid}}}\sum_t m_t\min\left(\rho_t A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right),
\qquad \rho_t=\exp(\log\pi_{\mathrm{new}}-\log\pi_{\mathrm{old}}).
\]

The critic uses a separately clipped value loss and the same valid-action denominator. Padding is excluded explicitly. The old log probabilities, values, masks, and targets are retained so a reviewer can reconstruct each update without relying on the learner's summary metrics.

## Learner edit contract

The starter keeps parent loading, hash checks, model-prefix scoring, restricted categorical probabilities, reward and critic feature construction, state serialization, and evaluation support as working code. Implement the six marked functions without importing `solution.py`:

| Function | Input and output contract |
| --- | --- |
| `fit_pairwise_reward_head(parent, *, max_updates=None)` | Fit only the four training rows. Return a frozen weight array with shape `[3,4]` plus initial/final pair loss and signed training margins. `max_updates=0` must return the declared zero-initialized head without running an optimizer update; the public semantic check uses this dry-run mode. The three rows correspond to up, down, and punctuation action counts; the four columns are the frozen parent restricted-logit features at the prompt prefix. |
| `rollout(...)` | Given actor, critic, reference, reward head, prompt IDs `[B]`, and a PRNG key, return arrays with action and mask shape `[B,4]`, context features `[B,4,4]`, and one saved token prefix per `[batch,time]` action. `target_role_mask` marks valid generated targets. `input_role_mask` is false for the first generated target and true when a prior response action is in context. EOS is valid; later positions are padding. |
| `generalized_advantage_estimate(records, gamma, lam)` | Consume reward, old value, post-action value, target-role, terminal, and truncation arrays shaped `[B,T]`. Return `deltas`, `advantages`, and frozen `value_targets`, each `[B,T]`. Terminal EOS bootstraps zero. Truncation bootstraps from the post-action value and ends the lambda trace. |
| `ppo_policy_loss(new_logps, old_logps, advantages, mask, clip)` | Consume selected-action log probabilities, frozen rollout log probabilities, advantages, and the `[B,T]` valid-target mask. Return the signed clipped-minimum scalar mean over valid actions only. |
| `clipped_value_loss(new_values, old_values, targets, mask, clip)` | Consume critic predictions and frozen rollout values/targets shaped `[B,T]`. Return one half the maximum of clipped and unclipped squared error, averaged over valid response targets. |
| `train(parent, state, reference_params, reward_params, max_updates)` | Run up to the fixed total update clock. Return complete train state and raw update records. Old log probabilities and values, GAE, and targets remain fixed through both inner epochs. `max_updates` supports the declared 4-update checkpoint boundary. |

The examples are intentionally short, but the central boundary is the same as the longer course sequence: each action is predicted from the prefix before that action is appended. The checker grades numeric and protocol behavior, not matching variable names or source expressions from the reference.

## Run the bounded check

From the repository root, install the project-local pins in an isolated environment. The runner is CPU-only and requires the exact recipe approval path from the accepted source review.

```sh
python3 -m venv .venv-transformer-lineage
source .venv-transformer-lineage/bin/activate
python -m pip install -r projects/posttraining-audit/transformer_lineage/requirements.txt
python projects/posttraining-audit/transformer_lineage/check.py \
  --implementation solution \
  --stage all --approval /exact/path/to/run-authorized-lineage-approval.json \
  --expected-approval-sha256 EXACT_APPROVAL_SHA256 \
  --output /tmp/transformer-lineage-run
```

The starter is expected to fail at its first learner function. A copied learner file can be supplied with `--candidate` and graded without optimizer updates by using `--stage semantic` plus its caller-supplied `--candidate-sha256`. This dry run calls all six learner APIs, including a zero-update reward fit and `train(..., max_updates=0)`, and checks real frozen-decoder EOS and timeout rollouts plus independent numeric oracles. It does not qualify a fitted reward head or a trained child. The numeric checks grade semantics rather than matching reference source spelling. The run command requires the exact approval path and caller-supplied approval SHA issued after final source review. Output paths must be fresh locations under the platform's temporary directory, outside the checkout. The output folder is created only after approval, recipe, fixture, parent package, source, and processor preflight succeeds.

For a candidate that should receive an actual mechanics check, use the separate learner-verification approval and a new output path:

```sh
python projects/posttraining-audit/transformer_lineage/check.py \
  --candidate /path/to/completed-learner.py \
  --candidate-sha256 EXACT_CANDIDATE_SHA256 \
  --stage learner-check \
  --approval /exact/path/to/learner-verification-approval.json \
  --expected-approval-sha256 EXACT_LEARNER_APPROVAL_SHA256 \
  --output /tmp/transformer-lineage-learner-check
```

That distinct approval caps each candidate invocation at one outer update, two inner epochs, and four reward-fit steps. The checker reconstructs all four reward-head Adam steps from the pinned synthetic pairs and frozen parent features, then independently replays both actor and critic Adam epochs from the raw first rollout and checks the resulting parameter and moment trees. It also reconstructs the saved loss inputs, masks, advantages, targets, and restricted probabilities. It reports mechanics only. It does not score dev/heldout quality, participate in the 8+4+4 campaign, or authorize corpus, cloud, or release work. Each candidate file is hashed before import and that exact hash is written into its private evidence.

The evidence schema lists raw prefix and action arrays, target and input masks, eligible action support, old and reference restricted probabilities, selected log probabilities, new log probabilities per epoch, KL and reward components, values, advantages, targets, optimizer state, PRNG key, and update/epoch clocks. The checker reconstructs selected log probabilities, sampled KL, and conditional categorical KL directly from the raw probabilities. The fixed batch size of three cycles through four training prompt instances, so each update changes the next prompt IDs. Evaluation folds a distinct episode index into each fixed seed; child and parent receive the same key for each matched prompt, while repeated prompt IDs use distinct keys. The predeclared execution has one uninterrupted 8-update primary lane and one identical 4-update plus fresh-process restore plus 4-update replay lane, for 16 actor update executions total and one shared fitted reward head. The replay lane is diagnostic only and cannot select checkpoints or affect quality floors. At the split and final clocks, the checker compares actor and critic trees, both Adam states, random key, prompt cursor, next prompt IDs, old/reference log probabilities, and update metrics. Checkpoint restore validates the exact leaf structure, finite values, all parameter and optimizer hashes, source identity, and cursor/clock relationship. The child package records both the parent package identity and its own parameter hash.

## Read results with the right scope

The predeclared heldout floor compares mean learned reward against the frozen parent on matched prompt IDs and fixed sampling seeds. It also caps exact restricted categorical KL per valid action. A failed floor remains in the report; there is no retuning after the heldout result. Pairwise heldout accuracy and reward margins are descriptive diagnostics, not human preference measurements. All prompts and preferences are synthetic, and the heldout strings do not establish semantic generalization.

The stage is a bridge exercise about saved-parameter lineage, categorical sequence rollouts, terminal and timeout returns, PPO clipping, and full-state recovery. It does not establish natural-language RLHF quality, human alignment, retained Alice quality, full-vocabulary KL, completion of C02, a public model release, or qualification of the wider capstone.

## Teaching references

The clipped multi-epoch policy objective follows [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347). The advantage recursion follows [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438). JAX's [`stop_gradient`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.stop_gradient.html) makes rollout values and targets constants within an update. The optimizer uses the functional `init`, `update`, and `apply_updates` pattern documented by [Optax](https://optax.readthedocs.io/en/stable/getting_started.html).
