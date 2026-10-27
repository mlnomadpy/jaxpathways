"""Regularization and honest validation · learner-only exercises (3/3 contracts ready)"""

# Exercise: For a target slope of \(4\) and penalty \(\lambda=1\), derive the fitted slope and verify its gradient.

# Inputs:
# - Training features, targets, and a nonnegative penalty.

# Outputs:
# - Fitted coefficients and predictions on supplied rows.

# Invariants to check:
# - Only designated coefficients are penalized; preprocessing statistics come from training data alone.

# Hints:
# - Separate the intercept from the penalized slope.
# - Compare validation transforms using the training-only statistics.

import jax.numpy as jnp

def ridge_fit(features, targets, penalty):
    """Fitted coefficients and predictions on supplied rows."""
    # TODO: Fit the stated penalized objective while separating intercept penalty and training-only preprocessing.
    raise NotImplementedError




# Do not accidentally shrink the intercept (Transfer / diagnosis):
# Fit constant targets \((5,5,5)\) using inputs \((-1,0,1)\) and a bias. Compare penalizing only the slope with penalizing both parameters at \(\lambda=1\).

# Inputs:
# - Fit constant targets \((5,5,5)\) using inputs \((-1,0,1)\) and a bias. Compare penalizing only the slope with penalizing both parameters at \(\lambda=1\).

# Outputs:
# - Two fitted parameter sets with a prediction comparison.

# Invariants to check:
# - The intercept penalty choice is explicit and the unpenalized fit uses training rows only.

# Hints:
# - Use a diagonal mask \(\operatorname{diag}(1,0)\).

import jax.numpy as jnp

def practice_do_not_accidentally_shrink_the_intercept(features, targets):
    """Implement the do not accidentally shrink the intercept check."""
    # TODO: Fit the constant target data with slope-only and all-parameter penalties and compare both parameter pairs.
    raise NotImplementedError




# Catch preprocessing leakage (Transfer / diagnosis):
# Training inputs are \((0,2)\) and validation input is \(100\). Compare training-only centering with centering all inputs together. Which can be reused for a truly unseen example?

# Inputs:
# - Training inputs are \((0,2)\) and validation input is \(100\). Compare training-only centering with centering all inputs together. Which can be reused for a truly unseen example?

# Outputs:
# - Both centers and the transformed unseen example under the training-only center.

# Invariants to check:
# - Validation data does not influence preprocessing statistics used for fitting.

# Hints:
# - Fit the mean on training data; do not refit it on validation data.

import jax.numpy as jnp

def practice_catch_preprocessing_leakage(train_features, validation_features):
    """Implement the catch preprocessing leakage check."""
    # TODO: Compute the center from training inputs alone and compare it with a center that includes the held-out value.
    raise NotImplementedError

