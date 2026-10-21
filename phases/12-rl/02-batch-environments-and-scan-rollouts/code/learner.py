"""Batch environments and scan rollouts · learner-only exercises (3/3 contracts ready)"""

# Exercise: Change the batch to \(7\) environments and the horizon to \(5\). Verify array shapes, one ending event per episode and zero rewards after finishing.

# Inputs:
# - A policy parameter vector `theta` and explicit JAX key.
# - Changed batch size 7 and horizon 5.
# - The lesson’s `rollout` collector and its time-major records.

# Outputs:
# - Final state for each environment and the retained rollout record.
# - A small set of assertions about shapes, activity, and ending events.

# Invariants to check:
# - Reward and event arrays have shape `(5, 7)`.
# - Every episode has one ending event.
# - Rows after finishing are padding and carry no reward.

# Hints:
# - The first axis is time and the second is environment.
# - The active flag describes the state before the action.

import jax
import jax.numpy as jnp

def inspect_changed_rollout(theta, key):
    # TODO: collect with seven environments and a five-step horizon
    final_state, records = ...
    # TODO: check time/environment axes and episode endings
    raise NotImplementedError




# Catch a shifted mask (Transfer / diagnosis):
# Construct a two-action success from position \(1\). Compare masking by the pre-transition done flag with masking by the post-transition done flag.

# Inputs:
# - State at position 1, unfinished.
# - Two right actions and the lesson transition.

# Outputs:
# - The accumulated transition reward with the correct pre-action mask.
# - A second total showing the effect of a shifted post-action mask.

# Invariants to check:
# - The successful transition remains active.
# - The two totals differ because only one uses the pre-transition done flag.

# Hints:
# - Capture activity before calling `step`.
# - Do not discard the transition that reaches the goal.

import jax.numpy as jnp

def compare_activity_masks(step, State):
    state = State(jnp.int32(1), jnp.int32(0), jnp.bool_(False))
    # TODO: evaluate both actions using pre-step and post-step masks
    raise NotImplementedError




# Check a deterministic policy limit (Transfer / diagnosis):
# Use very large positive logits and verify that every episode follows a shortest route.

# Inputs:
# - A policy with strongly positive action logits.
# - A batch of 23 environments, a fixed key, and an eight-step horizon.

# Outputs:
# - Collected trajectories and a summary of their action and ending behavior.

# Invariants to check:
# - Repeated calls with the same key replay.
# - The sampled rollouts obey the same environment and horizon contract.

# Hints:
# - Use a fresh key to test reproducibility separately.
# - A deterministic limit is a behavioral check, not a return benchmark.

import jax
import jax.numpy as jnp

def inspect_right_biased_policy(rollout, key):
    theta = jnp.full((3,), 100.)
    # TODO: collect with the requested batch and horizon
    final_state, records = ...
    raise NotImplementedError

