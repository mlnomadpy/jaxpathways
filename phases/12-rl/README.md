# Phase 12: Reinforcement learning

Specializations.

Build a functional environment, collect randomized trajectories, train policy-gradient and clipped-policy agents, and evaluate independent seeds.

Keep terminal and timeout behavior explicit. Verify a small enumerated expectation before trusting a sampled policy gradient, then separate training from held-out evaluation.

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

### Try a changed condition

You evaluate one trained agent over many episodes and report a tiny uncertainty interval. Does this measure sensitivity to training randomness?

<details><summary>Compare an approach</summary>

No. Those episodes characterize that agent under the evaluation distribution. Train independent agents to assess training variation, retain agent-level means and state the experimental unit.

</details>

**Symptom:** PPO ratios always equal one after every update.

**Check next:** Check that old log probabilities were stored at collection rather than recomputed from the current policy.

### Decide what is ready

Use policy-evaluation. Keep independent transition checks, behavior-policy identity, changed-horizon cases and paired evaluations across declared independent training seeds.

### Further work

Continuous actions, value-function estimation and larger benchmark environments need new implementation and evaluation labs; this tabular task is deliberately bounded.

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

## Phase project

An evaluated policy in a functional environment.

**Demonstrate:** Compare a trained policy to a baseline across several seeds and report variability.

Project status: implemented staged practice · [Open source](../../projects/policy-evaluation/README.md).



[Primary documentation](https://github.com/FLAIROx/Jumanji).
