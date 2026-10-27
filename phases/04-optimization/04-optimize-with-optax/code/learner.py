"""Optimize with Optax · learner-only exercises (3/3 contracts ready)"""

# Exercise: At the original zero parameters, compare the first Optax SGD update with an explicit tree-map gradient descent step.

# Inputs:
# - A parameter pytree, an Optax transformation and its state, and a scalar loss.

# Outputs:
# - The updated parameter pytree and next optimizer state.

# Invariants to check:
# - Update leaves match parameter leaves; Optax updates are applied once and state is carried forward.

# Hints:
# - Use value_and_grad to get the gradient tree.
# - Keep optimizer state as a separate returned value so it can be replayed.

import jax
import jax.numpy as jnp
import optax

def sgd_update(params, optimizer, opt_state, loss_fn):
    """The updated parameter pytree and next optimizer state."""
    # TODO: Initialize the Optax state, compute one gradient tree, request updates, and apply them once to the parameters.
    raise NotImplementedError




# Replay a momentum continuation (Transfer / diagnosis):
# Save the state after the first update; run the next update twice from that state and compare every state leaf.

# Inputs:
# - Save the state after the first update; run the next update twice from that state and compare every state leaf.

# Outputs:
# - Equal next parameters and state leaves from replayed continuations.

# Invariants to check:
# - Both continuations start from the same parameters, gradients, and saved optimizer state.

# Hints:
# - Optimizer transformations return new state; reuse the saved value for the replay.

import jax.numpy as jnp
import optax

def practice_replay_a_momentum_continuation(params, opt_state, grads):
    """Implement the replay a momentum continuation check."""
    # TODO: Advance momentum once, snapshot every optimizer-state leaf, and compare two next updates restored from that snapshot.
    raise NotImplementedError




# Catch double subtraction (Transfer / diagnosis):
# At zero parameters, subtract SGD updates by mistake. Show the loss increases; repair using apply_updates.

# Inputs:
# - At zero parameters, subtract SGD updates by mistake. Show the loss increases; repair using apply_updates.

# Outputs:
# - Correct and double-subtracted parameter trees with a loss comparison.

# Invariants to check:
# - Each update is applied with its documented sign exactly once.

# Hints:
# - SGD updates already contain the minus sign.

import jax.numpy as jnp
import optax

def practice_catch_double_subtraction(params, opt_state, grads):
    """Implement the catch double subtraction check."""
    # TODO: Apply the Optax update with apply_updates, then deliberately subtract the update once more and compare losses.
    raise NotImplementedError

