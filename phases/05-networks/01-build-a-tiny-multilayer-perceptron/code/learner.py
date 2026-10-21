"""Build a tiny multilayer perceptron · learner-only exercises (3/3 contracts ready)"""

# Exercise: Swap class zero and one. Transform only the trained output layer to preserve the boundary and verify predictions on new coordinates.

# Inputs:
# - An MLP parameter pytree and a permutation of its output classes.

# Outputs:
# - Transformed output-layer parameters and logits on held-out coordinates.

# Invariants to check:
# - The transformed model preserves or intentionally permutes class scores as specified; hidden-unit changes must be paired.

# Hints:
# - Inspect which parameter axis represents classes.
# - Test a new coordinate batch, not only the training points.

import jax.numpy as jnp

def permute_output_classes(params, class_permutation):
    """Transformed output-layer parameters and logits on held-out coordinates."""
    # TODO: Transform the output-layer class axis and verify logits on new input coordinates.
    raise NotImplementedError




# Reorder hidden units without changing predictions (Transfer):
# Permute the hidden units and make the matching change to the output layer. Predict whether the logits change. Test the result on new coordinates, then show why permuting only one side breaks the contract.

# Inputs:
# - Permute the hidden units and make the matching change to the output layer. Predict whether the logits change. Test the result on new coordinates, then show why permuting only one side breaks the contract.

# Outputs:
# - Logit parity on new inputs plus a one-sided-permutation counterexample.

# Invariants to check:
# - Incoming and outgoing parameter axes use the same hidden-unit permutation.

# Hints:
# - Columns of the first kernel and rows of the second kernel describe the same hidden units.

import jax.numpy as jnp

def practice_reorder_hidden_units_without_changing_predictions(hidden_permutation, hidden_weights, output_weights):
    """Implement the reorder hidden units without changing predictions check."""
    # TODO: Permute the hidden-unit axis and apply the matching inverse permutation to outgoing weights and biases.
    raise NotImplementedError




# Diagnose accidental pairwise broadcasting (Intermediate):
# Demonstrate how labels shaped \((B, 1)\) and scores shaped \((B,)\) create a \((B, B)\) loss. Repair the contract before averaging.

# Inputs:
# - Demonstrate how labels shaped \((B, 1)\) and scores shaped \((B,)\) create a \((B, B)\) loss. Repair the contract before averaging.

# Outputs:
# - Rejected mismatched shape and a per-example loss after alignment.

# Invariants to check:
# - One score is paired with one label; the reduction is over examples only.

# Hints:
# - Inspect the unreduced loss shape; a scalar mean can hide the error.

import jax.numpy as jnp

def practice_diagnose_accidental_pairwise_broadcasting(scores, labels):
    """Implement the diagnose accidental pairwise broadcasting check."""
    # TODO: Assert score and label shapes match before subtraction, then show the (B,1) by (B,) broadcast shape and repair it.
    raise NotImplementedError

