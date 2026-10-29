# Optional appendix: a saved decoder enters sequence PPO

This appendix connects the course's sequence-PPO mechanics to a small saved Transformer. It is an optional extension to [Audit SFT, DPO and sequence PPO](../../../projects/posttraining-audit/README.md). Registered Stage 5 remains the table-policy exercise in `projects/posttraining-audit/starter/sequence_ppo.py`; this bridge does not change its edit file, commands, or route. The bridge has a separate starter, recipe, checker, and approval gates.

The parent is the two-layer decoder saved by the [text-harness foundations project](https://www.tahabouhsine.com/jaxpathways/project-guides/text-harness-foundations.html). Its pinned configuration has width 16, four attention heads, an MLP width of 32, context length 24, vocabulary size 384, and tied embeddings. The parameter tree contains 10,880 parameters. The bridge checks the saved inference package and its source identity before use. The pinned [package manifest](https://github.com/mlnomadpy/jaxpathways/blob/f858816ead940aeb969e23fbc32df232da51196f/projects/text-harness/foundations/reference/run/inference-package/package.json) records the parent package. Data-source provenance and terms are summarized in the [pinned source notice](https://github.com/mlnomadpy/jaxpathways/blob/f858816ead940aeb969e23fbc32df232da51196f/projects/text-harness/foundations/data/LICENSE.md) and in [PARENT_RIGHTS.md](teaching/PARENT_RIGHTS.md). Those notices describe the Alice source. They do not grant new rights to model parameters or child artifacts.

## Predict the boundaries

The fixed synthetic prompts ask for a direction. Each generated action is one of four pinned single-token IDs: ` up` (383), ` down` (381), `!` (4), or `[EOS]` (2). At each prefix, the policy normalizes probabilities only across currently eligible actions from this set. EOS is ineligible until two non-EOS actions have been generated. Thus the first possible EOS action is the third generated action. These restricted distributions do not cover the full vocabulary.

Before reading the implementation, write down your answer to each question:

1. At the first prediction, which tokens are in the decoder prefix? Is the action being predicted already in that prefix?
2. If the third generated action is EOS, after two non-EOS actions, which positions are valid targets and which later positions are padding?
3. If all four actions are generated without EOS, what does the final action's next-value bootstrap represent?
4. If a valid action has a negative sampled log-ratio, can its sampled shaping term increase its reward?
5. Does a small sum of the reported per-action conditional KL values equal the divergence between the full response distributions?

The decoder scores the current prefix before sampling the next action. Only then is that action appended to the prefix for the following prediction. The target-role mask marks generated actions that receive loss. The input-role mask records whether a prior response action is already in the context. The first action is a valid target even though the context contains no earlier response action, so these masks answer different questions.

## Keep four components distinct

Initialization copies the verified parent parameters into a trainable actor and a frozen reference. The actor and a new critic receive fresh Adam state at the post-training objective boundary. The parent package contains inference parameters and a tokenizer, not the language-model optimizer moments or schedule.

A reward head is fitted once on four fixed synthetic training preference pairs and then frozen. Its features combine restricted parent-logit context with response action counts. The interaction lets a response score depend on the prompt context; a prompt-only additive feature would cancel in a chosen-versus-rejected comparison for the same prompt. The heldout pair diagnostic uses only two pairs and is not human preference evidence.

The critic is a new linear value head over stop-gradient restricted parent logits at each prefix. These are frozen Transformer-conditioned action features. They are not Transformer hidden states and do not represent a full-vocabulary value function. During the new objective, the actor and critic update; the parent reference and fitted reward head remain fixed.

## Calculate EOS and timeout by hand

For valid actions, the temporal-difference residual and reverse GAE recursion are

\[
\delta_t = r_t + \gamma b_t V_{\mathrm{old}}(s_{t+1}) - V_{\mathrm{old}}(s_t),
\qquad
A_t = \delta_t + \gamma\lambda c_t A_{t+1}.
\]

Here, \(b_t\) controls value bootstrapping and \(c_t\) controls continuation of the advantage trace. EOS uses zero bootstrap and stops the trace. Timeout uses the post-action critic value for the last residual and also stops the trace. The terminal reward for the observed response is placed on its last valid action. Padding has zero reward, advantage, target, policy loss, and critic loss.

At an EOS or timeout boundary, \(c_t=0\) blocks \(A_{t+1}\) from entering the current \(A_t\). The boundary advantage \(A_t=\delta_t\) can still flow backward into \(A_{t-1}\) when the preceding transition continues. Do not reset the reverse carry after computing the boundary advantage: that would drop the terminal reward credit from earlier actions.

Use \(\gamma=0.9\) and \(\lambda=0.8\). The following is an analytic exercise with the declared recursion, not a new model run.

**EOS after two non-EOS actions.** Let the three actions be `up`, `!`, and EOS. Set old values to \([0.4, 0.6, 0.7]\), rewards to \([0, 0, 2]\), and post-action values to \([0.6, 0.7, 0.9]\). The final value \(0.9\) is deliberately present so you can check that EOS ignores it. The residuals are \([0.14, 0.03, 1.3]\): the final residual is \(2-0.7=1.3\), with no bootstrap. Work backward to get advantages \([0.83552, 0.966, 1.3]\) and targets \([1.23552, 1.566, 2.0]\), where each target is the old value plus its advantage. The fourth array position is padding.

**Timeout at four actions.** Let all four actions be valid and non-EOS. Set old values to \([0.2, 0.3, 0.4, 0.5]\), rewards to \([0, 0, 0, 1]\), and post-action values to \([0.3, 0.4, 0.5, 0.8]\). The residuals are \([0.07, 0.06, 0.05, 1.22]\), since the final residual is \(1 + 0.9(0.8) - 0.5=1.22\). With the trace stopped at timeout, the advantages are \([0.59448256, 0.728448, 0.9284, 1.22]\), and the targets are \([0.79448256, 1.028448, 1.3284, 1.72]\). If you incorrectly set the timeout bootstrap to zero, the advantages become \([0.325744, 0.3552, 0.41, 0.5]\). Compare the two arrays to see how the final bootstrap affects earlier residuals while the stopped trace prevents advantages from crossing the boundary.

For another hand-calculated example, see the EOS and timeout walkthrough in the existing [Stage 5 project guide](../../../projects/posttraining-audit/README.md). The optional [learned values and GAE lesson](https://www.tahabouhsine.com/jaxpathways/lesson-rl-05.html) develops the same recursion.

## Check the signed PPO and critic losses

For each valid action, the probability ratio and signed clipped policy loss are

\[
\rho_t = \exp\!\left(\log\pi_{\mathrm{new}}(a_t\mid s_t)-\log\pi_{\mathrm{old}}(a_t\mid s_t)\right),
\qquad
L_{\mathrm{policy}} = -\frac{1}{N_{\mathrm{valid}}}
\sum_t m_t\min\!\left(\rho_t A_t,\; \operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right).
\]

Take \(\epsilon=0.2\) and \(\rho_t=1.5\), above the upper clip of 1.2. With advantage \(A_t=2\), the two terms are 3 and 2.4, so the minimum is 2.4 and the loss contribution is \(-2.4\). With advantage \(A_t=-2\), the terms are \(-3\) and \(-2.4\), so the minimum is \(-3\) and the loss contribution is 3. The sign changes which side of clipping is restrictive. At the lower side, with \(\rho_t=0.5\), clipped ratio 0.8, and \(A_t=-2\), the terms are \(-1\) and \(-1.6\); the minimum is \(-1.6\), giving loss contribution 1.6. Do not take the minimum of ratios before multiplying by the signed advantage.

The critic has its own clipped-value objective. For each valid target, compare the squared error from the new value with the squared error from the value change clipped around the old prediction. Take the larger error, halve it, and average across valid targets. For a one-target check, use old value 0.5, new value 0.9, target 1.7, and value clip 0.2. The ordinary squared error is \((0.9-1.7)^2=0.64\). The clipped prediction is 0.7, whose squared error is 1.0, so the loss is 0.5. Add any number of padded positions to the arrays: neither this loss nor the policy mean should change because padding is excluded from the mask and denominator.

## Interpret the KL measure narrowly

Each valid action also receives sampled rollout shaping from the selected-action log-ratio between the old actor and frozen parent:

\[
r_t^{\mathrm{shaped}} = r_t^{\mathrm{terminal}} - \beta\left(\log\pi_{\mathrm{old}}(a_t\mid s_t)-\log\pi_{\mathrm{parent}}(a_t\mid s_t)\right).
\]

The selected-action sample can be negative. The report separately computes exact categorical KL over the same eligible four-action support at contexts visited by the child. Its declared ceiling is \(0.8\) averaged per valid action over those sampled contexts. The per-response sum of conditional values is not the exact divergence between full generated-sequence distributions. The metric says nothing about unvisited contexts or the rest of the vocabulary.

## Implement the six learner contracts

The starter preserves parent loading, hash checks, prefix scoring, probability construction, feature construction, serialization, and evaluation support. Learners implement six functions without importing the reference solution:

| Function | Contract to check |
| --- | --- |
| `fit_pairwise_reward_head` | Fit only the four training rows. Return a weight array shaped `[3, 4]`, initial and final pair losses, and signed margins. A zero-update call returns the zero-initialized head without an optimizer step. |
| `rollout` | Return actions and target/input masks shaped `[B, 4]`, context features shaped `[B, 4, 4]`, and a saved token prefix for each batch/time action. EOS is valid; later positions are padding. |
| `generalized_advantage_estimate` | Consume reward, old value, post-action value, target-role, terminal, and truncation arrays shaped `[B, T]`. Return residuals, advantages, and frozen targets with that shape. EOS bootstraps zero; timeout uses the post-action value and ends the trace. |
| `ppo_policy_loss` | Use selected-action new and frozen old log probabilities, advantages, and a valid-target mask shaped `[B, T]`. Return the signed clipped-minimum mean over valid targets only. |
| `clipped_value_loss` | Use new and old critic values, targets, and a valid-target mask shaped `[B, T]`. Return the separately clipped half-squared-error mean over valid targets only. |
| `train` | Run up to the fixed update clock. Keep old log probabilities, old values, GAE, and targets fixed across both inner epochs. Support `max_updates=4` for the checkpoint boundary. |

Before coding, write one expected shape or scalar for each function. For rollout, mark the first valid target even though its input mask is false, then mark the third-action EOS or fourth-action timeout boundary. For the losses, use the hand calculations above and add a padded position. For training, inspect old log probabilities across both inner epochs; they should remain the snapshot from before the update.

## Understand cold starts and replay

The declared mechanics protocol fits the reward head once, then uses the same frozen head in both lanes. It initializes actor and critic Adam state at the new objective boundary. The primary lane has eight outer updates. The replay lane starts from the same parent actor, frozen reference, and fitted reward state, runs four updates, saves the actor, critic, both Adam states, random key, prompt cursor, clocks, and source identity, then restores in a fresh process for four more updates. The checker compares replay with the uninterrupted lane at the split and final clocks. This tests state continuity for the declared recipe; it does not establish model quality.

The bundled [portable non-run approval](portable-nonrun-approval.json) has SHA-256 `28f625fa4a1e8f705ce6b0e4a4ae56e241ce33d7ec5f3ec091e8428df738838b`. It permits source preflight and zero-update semantic inspection only. The separate [learner mechanics approval](learner-mechanics-approval.json), SHA-256 `72ecd919095e2fcc4bc095afee6f27a74571ba61aa1a56ae7c9ca442399c3c48`, permits one bounded mechanics check per candidate: one outer update, two optimizer epochs, and four reward-fit steps. Neither approval permits heldout quality scoring, a longer campaign, retuning, or release of child weights. A replay match does not grant permission to release weights.

From the extracted project root, preflight before importing a candidate. Install the pinned CPU packages, copy the starter to a separate candidate file, edit it, and compute its SHA-256 before either semantic inspection or mechanics checking. The semantic stage does zero optimizer updates. The learner check uses a fresh output directory under the system temporary directory and does not score heldout quality.

**POSIX shell**

```sh
ROOT=projects/posttraining-audit/transformer_lineage
PORTABLE="$ROOT/portable-nonrun-approval.json"
LEARNER="$ROOT/learner-mechanics-approval.json"

python3 "$ROOT/check.py" --stage preflight \
  --approval "$PORTABLE" \
  --expected-approval-sha256 28f625fa4a1e8f705ce6b0e4a4ae56e241ce33d7ec5f3ec091e8428df738838b

python3 -m venv .venv-transformer-lineage
. .venv-transformer-lineage/bin/activate
python -m pip install -r "$ROOT/requirements.txt"
cp "$ROOT/starter.py" "$ROOT/candidate.py"
# Edit candidate.py, then freeze the exact bytes before running a check.
CANDIDATE="$ROOT/candidate.py"
CANDIDATE_SHA=$(python -c 'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "$CANDIDATE")

python "$ROOT/check.py" --stage semantic \
  --candidate "$CANDIDATE" --candidate-sha256 "$CANDIDATE_SHA" \
  --approval "$PORTABLE" \
  --expected-approval-sha256 28f625fa4a1e8f705ce6b0e4a4ae56e241ce33d7ec5f3ec091e8428df738838b

OUT="/tmp/transformer-lineage-learner-check-$(python -c 'import uuid; print(uuid.uuid4().hex)')"
python "$ROOT/check.py" --stage learner-check \
  --candidate "$CANDIDATE" --candidate-sha256 "$CANDIDATE_SHA" \
  --approval "$LEARNER" \
  --expected-approval-sha256 72ecd919095e2fcc4bc095afee6f27a74571ba61aa1a56ae7c9ca442399c3c48 \
  --output "$OUT"
```

**PowerShell**

```powershell
$root = 'projects/posttraining-audit/transformer_lineage'
$portable = Join-Path $root 'portable-nonrun-approval.json'
$learner = Join-Path $root 'learner-mechanics-approval.json'

python (Join-Path $root 'check.py') --stage preflight `
  --approval $portable `
  --expected-approval-sha256 28f625fa4a1e8f705ce6b0e4a4ae56e241ce33d7ec5f3ec091e8428df738838b

py -3 -m venv .venv-transformer-lineage
.\.venv-transformer-lineage\Scripts\Activate.ps1
python -m pip install -r (Join-Path $root 'requirements.txt')
$candidate = Join-Path $root 'candidate.py'
Copy-Item (Join-Path $root 'starter.py') $candidate
# Edit candidate.py, then freeze the exact bytes before running a check.
$candidateSha = (Get-FileHash -Algorithm SHA256 $candidate).Hash.ToLowerInvariant()

python (Join-Path $root 'check.py') --stage semantic `
  --candidate $candidate --candidate-sha256 $candidateSha `
  --approval $portable `
  --expected-approval-sha256 28f625fa4a1e8f705ce6b0e4a4ae56e241ce33d7ec5f3ec091e8428df738838b

$output = Join-Path $env:TEMP ('transformer-lineage-learner-check-' + [guid]::NewGuid().ToString('N'))
python (Join-Path $root 'check.py') --stage learner-check `
  --candidate $candidate --candidate-sha256 $candidateSha `
  --approval $learner `
  --expected-approval-sha256 72ecd919095e2fcc4bc095afee6f27a74571ba61aa1a56ae7c9ca442399c3c48 `
  --output $output
```

The public recorded numeric packet has its own portable check. From the extracted project root, run `python projects/posttraining-audit/transformer_lineage/teaching/v5/check_numeric_packet.py --shipping-approval projects/posttraining-audit/transformer_lineage/portable-nonrun-approval.json --self-test`. It finds the repository root from its own location, so `--repo-root` is unnecessary. Full packet regeneration is different: the builder requires an explicit `--contract` plus retained private raw v5 evidence, receipts, and the frozen as-run source. Those inputs are not a learner regeneration recipe.

## Report the recorded v5 outcome honestly

### Recorded v5 result

![Mean child-minus-parent fitted-reward change is zero on dev and heldout; the strict positive heldout floor fails. Primary signed policy loss from inner epoch 2 by update, averaged per valid action. Exact restricted KL per valid action is 0.000123197, below its 0.8 ceiling.](teaching/v5/figure.svg)

**Figure 1.** Heldout mean child minus matched-parent fitted reward is 0.000 reward units across eight episodes and 29 valid actions; the predeclared strict positive improvement floor therefore fails. The second panel plots the primary lane’s signed clipped policy loss per valid action from the second inner optimizer epoch (epoch index 1), with update numbers 0 through 7; this is a training diagnostic rather than a quality score. Exact restricted conditional KL over eligible actions at sampled child-visited contexts is 0.000123197 per valid action, below the 0.8 ceiling; sampled selected-action KL is separately summarized in evaluation data. EOS and timeout arrays are synthetic zero-update probes, not model quality measurements.

Independent review accepted the v5 mechanics and replay evidence, while the strict heldout reward quality floor remains **failed**. Each of the development and heldout splits had eight matched prompt cases and 29 valid actions. In both splits, all eight sampled child action sequences matched the parent sequences even though the actor parameters changed. A sampled response match does not prove that the policy parameters were unchanged.

Heldout mean reward was \(-0.009211033582687378\) for both child and parent. The declared floor requires child reward to be strictly greater than parent reward, so equality fails it. The pairwise diagnostic was zero of two on development and two of two on heldout. The exact restricted categorical KL averaged over sampled child-visited contexts was \(0.000123196996\) per valid action. This differs from the selected-action log-ratio used for shaping, which can be negative. Neither the two-pair diagnostic nor the small exact KL overrides the failed strict reward floor. Keep these measures separate: mechanics and replay passed independent review; the heldout reward quality floor did not pass.

Across the separate v1 and v5 records, the cumulative ledger contains \(24\) outer updates and \(240\) reward-fit steps. In the earlier v1 development attempt, eight outer updates executed in memory, but only the update-4 checkpoint was durable. One first-development child rollout was produced before the baseline process errored; there was no matched baseline response, matched-quality comparison, child quality score, or replay result. Keep it separate from the v5 split metrics. Do not retry or retune this recipe based on the heldout outcome.

Treat this as a small synthetic mechanics experiment. It tests the declared token-prefix protocol, masks, objective arithmetic, optimizer boundary, state saving, replay, and limited heldout diagnostics. It does not establish natural-language RLHF quality, human alignment, semantic generalization, production utility, or permission to release a child model.

