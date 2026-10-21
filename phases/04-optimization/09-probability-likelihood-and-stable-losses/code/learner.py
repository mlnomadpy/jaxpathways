"""Probability, likelihood, and stable losses · learner-only exercises (3/3 contracts ready)"""

# Exercise: Compute the binary losses for positive labels at probabilities \(0.8\) and \(0.2\), and explain which prediction is better supported.

# Inputs:
# - Binary logits and same-shaped binary labels.

# Outputs:
# - A finite per-example loss and a declared reduction.

# Invariants to check:
# - Use the logits-based stable loss; duplicated examples preserve a mean and scale a sum.

# Hints:
# - Check confident correct and incorrect predictions separately.
# - Keep reduction choice explicit so its gradient scale is inspectable.

import jax.numpy as jnp

def binary_cross_entropy(logits, labels):
    """A finite per-example loss and a declared reduction."""
    # TODO: Compute stable per-example binary losses from logits and make the mean-versus-sum reduction explicit.
    raise NotImplementedError




# Fit a constant probability (Transfer / diagnosis):
# A dataset contains labels \((1,1,1,0)\). For a model with one constant logit, show that \(p=3/4\) is stationary and has less mean loss than \(p=1/2\).

# Inputs:
# - A dataset contains labels \((1,1,1,0)\). For a model with one constant logit, show that \(p=3/4\) is stationary and has less mean loss than \(p=1/2\).

# Outputs:
# - Stationarity at the empirical class fraction and a lower mean loss than the half-probability case.

# Invariants to check:
# - Every label contributes once to the mean binary loss.

# Hints:
# - The mean gradient is \(p-\bar y\). Convert the desired probability to log-odds.

import jax.numpy as jnp

def practice_fit_a_constant_probability(labels):
    """Implement the fit a constant probability check."""
    # TODO: Evaluate the constant-logit objective and derivative at logit(log(3)) and at zero.
    raise NotImplementedError




# Catch a reduction mistake (Transfer / diagnosis):
# Duplicate every observation. Verify that mean loss stays fixed while total loss doubles. What happens to gradients?

# Inputs:
# - Duplicate every observation. Verify that mean loss stays fixed while total loss doubles. What happens to gradients?

# Outputs:
# - Unchanged mean loss, doubled total loss, and both gradient scales.

# Invariants to check:
# - The duplicated dataset contains two copies of each original example.

# Hints:
# - The mean divides by the new observation count; the sum does not.

import jax
import jax.numpy as jnp

def practice_catch_a_reduction_mistake(labels):
    """Implement the catch a reduction mistake check."""
    # TODO: Duplicate every observation and compute both mean and total losses and their gradients.
    raise NotImplementedError

