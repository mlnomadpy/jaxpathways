"""Batch a function with vmap · learner-only exercises (3/3 contracts ready)"""

# Exercise: Foundation · Compute each prediction gradient with respect to the shared weight. Predict the shape and the derivative of the dot product, then verify against the input batch.

# Inputs:
# - A single-example prediction function and shared weight vector.
# - A batch of three rows.

# Outputs:
# - One weight gradient per row and the dot-product gradient.

# Invariants to check:
# - Map the batch axis while sharing parameters.
# - Predict the output shape before execution.

# Hints:
# - Compare vectorized output with independent row calculations.
# - Map only the example input axis.

import jax
import jax.numpy as jnp

def batched_prediction_grad(predict, weight, batch):
    # TODO: map the single-example gradient across rows
    raise NotImplementedError




# Turn sensitivities into training gradients (Practice):
# Use targets \([0, 2, 7]\). Define a scalar squared-error function for one example. Compute all per-example parameter gradients and derive the first row independently.

# Inputs:
# - The lesson batch and targets `[0, 2, 7]`.
# - A scalar squared-error function for one example.

# Outputs:
# - One parameter gradient per example.
# - An independent first-row calculation.

# Invariants to check:
# - Each example loss is scalar.
# - A prediction sensitivity is not substituted for a loss gradient.
# - The mapped result preserves the batch axis.

# Hints:
# - Differentiate the per-example loss with respect to parameters.
# - Keep targets aligned with rows.

import jax
import jax.numpy as jnp

def single_loss(parameters, row, target):
    # TODO: define one scalar example loss
    raise NotImplementedError

def collect_example_gradients(parameters, rows, targets):
    # TODO: map the loss gradient across aligned examples
    raise NotImplementedError




# Prove the mean-gradient identity with an experiment (Challenge):
# Define the mean loss from the mapped single loss. Compare its parameter gradient with the mean of individual_grads. Repeat after changing targets and explain why a sum changes the relationship.

# Inputs:
# - A mapped single-example loss and parameter tree.
# - Original targets and a changed target vector.

# Outputs:
# - The batch-mean gradient and average individual gradient for both target sets.

# Invariants to check:
# - The mean reduction uses the example axis.
# - Changed targets are included in the comparison.
# - A sum is identified as a different objective.

# Hints:
# - Define the batch objective from the mapped single loss.
# - Keep the parameter axis shared in `vmap`.

import jax
import jax.numpy as jnp

def compare_batch_and_individual_gradients(single_loss, parameters, rows, targets):
    # TODO: compare a batch-mean gradient with mapped gradients
    raise NotImplementedError

