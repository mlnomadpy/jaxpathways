"""Losses and value_and_grad · learner-only exercises (3/3 contracts ready)"""

# Exercise: Foundation · Starting from \(w=1\), calculate the next weight and loss for \(\alpha=0.1\) and \(\alpha=0.3\). Verify both. Explain the overshoot using the error multiplier rather than concluding that autodiff is wrong.

# Inputs:
# - The lesson model inputs and targets.
# - Starting weight 1 and learning rates 0.1 and 0.3.

# Outputs:
# - Each candidate weight and its post-update loss.

# Invariants to check:
# - Both updates use a gradient evaluated at the same starting weight.
# - Loss is recomputed at each candidate.
# - Overshoot is explained through the objective, not an autodiff failure.

# Hints:
# - Keep the initial gradient separate from the candidate evaluations.
# - Compare each rate independently.

import jax
import jax.numpy as jnp

def compare_updates(loss, initial_weight, rates):
    # TODO: compute the initial gradient and evaluate each candidate
    raise NotImplementedError




# Make the data boundary explicit (Practice):
# Rewrite `loss(weight, inputs, targets)` with data as arguments. Use value_and_grad and verify the gradient at \(w=1\) against the hand-derived result.

# Inputs:
# - A scalar weight and input/target arrays passed as arguments.
# - Starting weight 1.

# Outputs:
# - Loss and weight gradient from `value_and_grad`.

# Invariants to check:
# - Data are explicit function inputs.
# - Differentiation is with respect to the weight.
# - Compare with an independent hand result.

# Hints:
# - Place the parameter as the first argument.
# - Use the same fixed data in both checks.

import jax
import jax.numpy as jnp

def explicit_loss(weight, inputs, targets):
    # TODO: return the scalar batch objective
    raise NotImplementedError

def check_value_and_gradient(inputs, targets):
    # TODO: evaluate value_and_grad at the requested weight
    raise NotImplementedError




# Audit reduction and batch replication (Challenge):
# Compare the mean and sum loss gradients. Repeat this batch twice and predict the gradient under both reductions. Explain which learning-rate adjustment makes a sum-based update match a mean-based update.

# Inputs:
# - Inputs and targets for one batch.
# - A repeated copy of that batch.

# Outputs:
# - Mean-loss and sum-loss gradients before and after repetition.

# Invariants to check:
# - Repeating the batch preserves a mean gradient.
# - A sum gradient scales with repeated example count.
# - State the learning-rate adjustment that matches the update.

# Hints:
# - Keep mean and sum in separate functions.
# - Track the batch duplication factor.

import jax
import jax.numpy as jnp

def compare_reductions_and_repeats(inputs, targets, weight):
    # TODO: compute both reductions before and after repeating data
    raise NotImplementedError

