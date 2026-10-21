"""Write a functional environment · learner-only exercises (3/3 contracts ready)"""

# Exercise: Begin at position \(0\), choose left for three actions with horizon \(3\), and then attempt one more action. Verify the event flags, total reward and absorbing state.

# Inputs:
# - Initial state is position 0, elapsed count 0, and not done.
# - The attempted action is left (0), and the horizon is 3.
# - Use the lesson’s `State` and `step` definitions.

# Outputs:
# - A trace with one next state, reward, termination flag, and truncation flag per transition.
# - A check for a repeated call after the state has finished.

# Invariants to check:
# - The step at the deadline remains part of the return.
# - A call after finishing emits no new reward or ending event.
# - Position and elapsed count follow the declared environment contract.

# Hints:
# - Keep event flags separate from the persistent `done` state.
# - Record the reward from each transition before checking the next state.

from typing import NamedTuple
import jax.numpy as jnp

class State(NamedTuple):
    position: jnp.ndarray
    elapsed: jnp.ndarray
    done: jnp.ndarray

def step(state, action, horizon=3):
    # TODO: implement the transition contract from the lesson
    raise NotImplementedError

def inspect_deadline():
    state = State(jnp.int32(0), jnp.int32(0), jnp.bool_(False))
    rewards = []
    events = []
    # TODO: record the three requested transitions and one call after done
    raise NotImplementedError




# Separate the two target meanings (Transfer / diagnosis):
# Compute the continuing-task bootstrap target for a timeout and a true terminal transition when \(r=-0.01\), \(\gamma=0.9\) and \(V=2\). Compare with our finite-horizon objective.

# Inputs:
# - A transition reward of -0.01, discount 0.9, and next-state value 2.
# - Two meanings for the ending event: a continuing-task timeout and a true terminal.

# Outputs:
# - Both candidate targets and a comparison with the finite-horizon contract.

# Invariants to check:
# - A true terminal has no future value.
# - The finite-horizon deadline and a continuing-task collection cutoff express different objectives.

# Hints:
# - Write the one-step target before substituting the timeout or terminal mask.
# - Keep the reward term in both cases.

import jax.numpy as jnp

def compare_ending_targets(reward, discount, next_value):
    # TODO: compute the continuing-cutoff and terminal targets
    timeout_target = ...
    terminal_target = ...
    raise NotImplementedError




# Enumerate all one-step moves (Transfer / diagnosis):
# Check both actions at every unfinished position with a Python reference, then compare the jitted function.

# Inputs:
# - Every unfinished corridor position and both binary actions.
# - The lesson’s transition function.

# Outputs:
# - A Python reference result and the compiled transition result for each case.

# Invariants to check:
# - Both action choices are checked at both boundaries.
# - The reference check does not call the JAX transition to compute its expected result.

# Hints:
# - Represent the independent reference using ordinary Python integers.
# - Compare next state, reward, and both ending flags.

import jax

def compare_all_moves(step, State):
    compiled_step = jax.jit(step)
    # TODO: enumerate positions/actions and compare with an independent rule
    raise NotImplementedError

