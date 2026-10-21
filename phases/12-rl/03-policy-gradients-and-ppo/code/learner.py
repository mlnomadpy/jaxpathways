"""Policy gradients and PPO · learner-only exercises (3/3 contracts ready)"""

# Exercise: Run REINFORCE with a different training seed and retain the initial and final exact return. Verify that the final policy improves on the uniform baseline without requiring every intermediate point to improve.

# Inputs:
# - A fresh integer training seed different from the earlier example.
# - The finite-horizon `train` function using REINFORCE and one epoch per rollout.
# - The initial uniform policy and returned exact-return history.

# Outputs:
# - The trained policy and its full exact-return history.
# - A comparison against the initial policy under the same exact evaluator.

# Invariants to check:
# - A changed seed changes the sampled learning path.
# - The final exact return improves for this declared run.
# - Intermediate points need not increase monotonically.

# Hints:
# - Pass the new seed to training; do not reuse an evaluation key.
# - Use exact expected return for this tiny environment, not a noisy training batch mean.

import jax.numpy as jnp

def compare_reinforce_seed(seed, train, exact_return):
    # TODO: run one fresh REINFORCE training run
    theta, history = ...
    initial_policy = jnp.zeros((3,))
    # TODO: compare initial and final values and return the evidence
    raise NotImplementedError




# Inspect a frozen behavior reference (Transfer / diagnosis):
# Collect at zero logits, then increase all logits to \(1\). Verify that stored old log probabilities do not change and that live-action ratios include values on both sides of \(1\).

# Inputs:
# - A rollout collected with zero logits and a fixed key.
# - A candidate policy with logits equal to one.
# - Stored behavior log probabilities from the unchanged rollout.

# Outputs:
# - The ratio range and a check that collection-time log probabilities stay fixed.

# Invariants to check:
# - Ratios compare current action likelihood to the saved behavior likelihood.
# - The saved old log probabilities are not recomputed or mutated.

# Hints:
# - Copy or retain the old array before evaluating the candidate.
# - Use the recorded actions and observations in both calculations.

import jax
import jax.numpy as jnp

def inspect_behavior_ratios(rollout, log_probability, key):
    # TODO: collect the behavior batch and save its old log probabilities
    # TODO: evaluate the changed policy on those same observations/actions
    # TODO: calculate and inspect the action probability ratios
    raise NotImplementedError




# Change the time budget (Transfer / diagnosis):
# For horizon \(2\), compute the always-right expected return before running code. Only one initial position can reach the goal.

# Inputs:
# - The always-right policy and horizon 2.
# - The exact finite-horizon evaluator.

# Outputs:
# - An independent hand calculation and the evaluator result.

# Invariants to check:
# - Only the starting positions that can reach the goal within the budget contribute a success reward.
# - The exact result changes with the time budget.

# Hints:
# - Enumerate the two equally likely start positions.
# - Write each path’s transition rewards before taking their mean.

import jax.numpy as jnp

def check_short_horizon(exact_return):
    # TODO: compute the hand expectation from the two starts
    hand_value = ...
    evaluated = exact_return(jnp.full((3,), 100.), horizon=2)
    # TODO: compare the independent value with the evaluator
    raise NotImplementedError

