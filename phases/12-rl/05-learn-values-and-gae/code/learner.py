"""Learn value functions and generalized advantage estimation · learner-only exercises (3/3 contracts ready)"""

# Exercise: The task lasts at most eight actions and the value target is discounted return with gamma=0.9, matching GAE gamma. Change the collection key from 313 to 314. Before running, predict which of the fitted values and final MSE should remain identical. Then retain both error histories and explain why changed trajectories should change the fitted table.

# Inputs:
# - A policy, an explicit collection key, and a retained rollout with `observation`, `next_observation`, and `active`.
# - A changed collection key 314 after the recorded seed 313 run.
# - The gamma=0.9 discounted-return and value-fit functions.

# Outputs:
# - The fitted four-state value table and initial/per-update masked-MSE history for discounted targets.
# - A comparison that keeps both seed-specific fits visible.

# Invariants to check:
# - Only active transitions contribute to value loss.
# - The value target discounts rewards with gamma=0.9, the same discount used by GAE.
# - The fit uses sampled rollout targets and reports real optimization error.
# - Different collection keys can produce different trajectories and fitted values.

# Hints:
# - Build targets from the collected reward array.
# - Use gamma=.9 in both the discounted return targets and the GAE call.
# - Keep the retained `next_observation` for the later GAE step.
# - Compare final errors without expecting stochastic targets to produce zero error.

import jax
import jax.numpy as jnp

def fit_changed_rollout(theta, seed, rollout, discounted_returns_to_go, fit_value_function):
    # TODO: collect a new rollout using the supplied seed
    final_state, records = ...
    # TODO: form targets and fit the state-value table
    targets = ...
    value_table, loss_history = ...
    # TODO: return the learned table and retained error history
    raise NotImplementedError




# Find a reset leak (Transfer / diagnosis):
# A collector resets immediately after a time limit. The previous transition stores the reset observation as `next_observation`; its value is large. What failed, and what should the record preserve?

# Inputs:
# - A continuing-task transition ending at a collection cutoff.
# - A retained final pre-reset observation that differs from the next episode’s reset observation.

# Outputs:
# - A target using the final transition observation and a trace that stops at reset.

# Invariants to check:
# - The reset observation cannot replace the saved transition result.
# - The bootstrap mask and trace mask are checked independently.

# Hints:
# - Inspect both `next_observation` on the cutoff row and `observation` on the following row.
# - Change a later-episode reward to see whether the prior trace leaks across reset.

import jax.numpy as jnp

def inspect_cutoff(records, value_table, gae):
    # TODO: calculate advantages for the retained cutoff record
    advantages = ...
    # TODO: perturb a following-episode reward and compare earlier advantages
    raise NotImplementedError




# Choose the finite-horizon contract (Transfer / application):
# The corridor objective ends after exactly eight actions, even if the agent has not reached the goal. Should its horizon row bootstrap from a hypothetical next value? Explain which GAE argument to set.

# Inputs:
# - A task whose objective ends after exactly eight actions.
# - A rollout with a truncation flag at its finite-horizon deadline.

# Outputs:
# - GAE advantages under the declared finite-horizon contract.

# Invariants to check:
# - There is no future objective value beyond the true task deadline.
# - The caller explicitly chooses the no-bootstrap truncation setting.

# Hints:
# - Distinguish the end of a task from the end of a data-collection window.
# - Inspect the helper’s default and the call site.

import jax.numpy as jnp

def finite_horizon_advantages(records, value_table, gae):
    # TODO: call the estimator with the finite-horizon bootstrap contract
    raise NotImplementedError

