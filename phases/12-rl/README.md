# Phase 12: Reinforcement learning

Specializations.

Build a functional environment, collect randomized trajectories, train policy-gradient and clipped-policy agents, fit a tabular value baseline, and evaluate independent seeds.

Keep terminal, timeout and reset behavior explicit. Verify a small enumerated expectation before trusting a sampled policy gradient. Fit and check the value baseline separately, then distinguish bootstrap permission from GAE trace continuation.

**Prerequisites:** 05: Neural network training.

**Hardware:** CPU; target accelerator execution requires separate evidence.

## Study guide: Is the policy improving under a fixed evaluation protocol?

Environment semantics and data collection define the learning problem. Verify transitions, masks and behavior-policy probabilities before trusting a noisy return curve.

### Check your starting point

An episode stops because its time budget expires. Must its bootstrap value be zero?

<details><summary>Compare your reasoning</summary>

Not necessarily. Distinguish truncation from genuine terminal states and state whether the task is finite-horizon or continuing. A timeout alone does not imply that future value is zero.

</details>

Review: [Write a functional environment](01-write-a-functional-environment/docs/en.md).

### Build in stages

1. **Test environment and rollout semantics.** Check transitions independently, preserve terminal rewards and label time/environment axes. Store the behavior policy that collected each action.

   Lessons: [Write a functional environment](01-write-a-functional-environment/docs/en.md) · [Batch environments and scan rollouts](02-batch-environments-and-scan-rollouts/docs/en.md).

2. **Compare updates and evaluations fairly.** Audit the clipped objective for both advantage signs. Freeze the evaluation protocol, compare a simple baseline and distinguish training seeds from repeated episodes.

   Lessons: [Policy gradients and PPO](03-policy-gradients-and-ppo/docs/en.md) · [Evaluate agents across seeds](04-evaluate-agents-across-seeds/docs/en.md).

3. **Learn and audit a value baseline.** Fit a value table from retained trajectories, compare its actual masked error with independent calculations, and derive a multi-step GAE trace. Keep the final pre-reset observation and separate bootstrap and trace masks.

   Lessons: [Learn value functions and generalized advantage estimation](05-learn-values-and-gae/docs/en.md).

### Try a changed condition

With discount 0.9, reward 1 and next-state value 10, compare a true terminal target with a time-limit cutoff in a continuing task.

<details><summary>Compare an approach</summary>

The true terminal target is 1. A cutoff that only ends data collection can retain bootstrap value, giving \(1+0.9\cdot10=10\). Use the final pre-reset observation, not a new episode’s reset observation. Declare the task semantics: a genuinely finite-horizon terminal objective can require a different contract. Lesson rl-05 checks both bootstrap modes and reset boundaries with retained observations. The actor remains a finite-horizon reward-to-go implementation.

</details>

**Symptom:** PPO ratios always equal one after every update.

**Check next:** Check that old log probabilities were stored at collection rather than recomputed from the current policy.

### Decide what is ready

Use policy-evaluation. Keep independent transition checks, behavior-policy identity, changed-horizon cases and paired evaluations across declared independent training seeds. Retain the fitted value table, independently recomputed error and terminal/cutoff GAE checks from stage 5.

### Further work

The actor still uses sampled reward-to-go. Connecting a learned critic and GAE to actor updates, continuous actions and larger benchmark environments requires further implementation and held-out evaluation. The separate tabular critic and GAE lesson establishes their bounded mechanics.

## Lesson sequence

### 12.01 Write a functional environment

[Read the lesson](01-write-a-functional-environment/docs/en.md) · [Run the code](01-write-a-functional-environment/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Implement reset and step as pure functions with explicit state. Distinguish termination, time-limit truncation and an absorbing finished state.

**Evidence:** Keep independent transition enumeration, reward/terminal checks and the absorbing-state diagnosis. Explain the state and observation contract.

**Checkpoint:** What should step emit when called on a finished state?

### 12.02 Batch environments and scan rollouts

[Read the lesson](02-batch-environments-and-scan-rollouts/docs/en.md) · [Run the code](02-batch-environments-and-scan-rollouts/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Build a scan over vmapped transitions with explicit key ownership. Reconstruct episode returns and reward-to-go from padded trajectories.

**Evidence:** Keep reconstructed trajectories, randomized keys, padding masks and independent return calculations. Show how a changed key affects interaction without changing the environment rules.

**Checkpoint:** Which axis should be summed to obtain one return per environment from a time-major reward array?

### 12.03 Policy gradients and PPO

[Read the lesson](03-policy-gradients-and-ppo/docs/en.md) · [Run the code](03-policy-gradients-and-ppo/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Derive and implement the score-function policy gradient for complete episodes. Train with fresh stochastic rollouts and a fixed behavior policy per PPO batch.

**Evidence:** Keep the exact small-state policy-gradient comparison, actual training curves and frozen PPO reference quantities. Diagnose changed clipping signs and distinguish Monte Carlo variation from an incorrect objective.

**Checkpoint:** During multiple PPO epochs on one rollout batch, what must remain fixed?

### 12.04 Evaluate agents across seeds

[Read the lesson](04-evaluate-agents-across-seeds/docs/en.md) · [Run the code](04-evaluate-agents-across-seeds/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Separate training seeds, evaluation streams and individual episodes. Report per-agent means and variability with an explicit aggregation unit.

**Evidence:** Keep multiple trained seeds, fixed held-out streams, exact versus sampled returns and failed trajectories. Explain uncertainty and demonstrate evaluation leaves the policy unchanged.

**Checkpoint:** What does evaluating one policy on many episodes fail to measure?

### 12.05 Learn value functions and generalized advantage estimation

[Read the lesson](05-learn-values-and-gae/docs/en.md) · [Run the code](05-learn-values-and-gae/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Fit sampled discounted return targets at gamma=0.9 over at most eight actions, then estimate GAE at gamma=0.9 and lambda=0.95 with terminal and collector-cutoff semantics explicit.

**Evidence:** Keep the discounted target definition, seed-313/314 value tables and fit histories, independent two-transition TD residuals, terminal-versus-cutoff bootstrap/trace checks and a note on the position-only approximation. Critic-fit loss is not policy-improvement evidence.

**Checkpoint:** At a continuing-task time limit, which observation supplies the bootstrap, and should the trace cross the reset?

## Phase project

An evaluated policy in a functional environment.

**Demonstrate:** Compare a trained policy to baselines across independent seeds, then defend a learned value estimate and a hand-computed GAE trace at terminal and collector-cutoff boundaries.

Project status: implemented staged practice · [Open source](../../projects/policy-evaluation/README.md). Copy `projects/policy-evaluation/starter/model.py` to `projects/policy-evaluation/my_model.py` and implement the stage functions there. From the workspace root, run `python3 projects/policy-evaluation/tests/check.py --implementation projects/policy-evaluation/my_model.py --stage 1`, then advance through `--stage 5`. Stage 5 is cumulative and adds a separate tabular critic and GAE; it does not replace the actor training targets.



[Primary documentation](https://github.com/FLAIROx/Jumanji).
