"""From NumPy to jax.numpy · learner-only exercises (3/3 contracts ready)"""

# Exercise: Foundation · Standardize the original data using its standard deviation plus epsilon \(10^{-6}\). Predict the transformed rows and check feature means and scales. Explain why the scale is only approximately one with epsilon.

# Inputs:
# - A training matrix with observations in rows and features in columns.
# - A positive epsilon for scale stability.

# Outputs:
# - Fitted feature statistics and standardized training rows.

# Invariants to check:
# - Fit statistics feature-wise using training data only.
# - Reuse the same statistics for later rows.
# - Constant features follow the lesson’s stated scale policy.

# Hints:
# - Keep the feature axis during reductions.
# - Center before scaling.
# - Separate fitting statistics from applying them.

import jax.numpy as jnp

def fit_and_standardize(training, epsilon=1e-6):
    # TODO: compute feature statistics and standardized values
    center = ...
    scale = ...
    transformed = ...
    return transformed, center, scale




# Test the constant-feature policy (Practice):
# Fit a standardizer to \([[1, 5]\), \([3, 5]\), \([5, 5]]\). Check the second training column and transform the new row \([7, 6]\). Explain why its second output is \(1\) rather than zero.

# Inputs:
# - Training data with a constant second feature.
# - A new row whose second feature differs.
# - The standardizer fit/apply functions.

# Outputs:
# - Transformed training data and transformed new row.

# Invariants to check:
# - A constant training feature follows the declared unit-scale policy.
# - Applying the transform does not refit statistics.

# Hints:
# - Inspect the fitted training scale.
# - Apply the same saved statistics to the new row.

import jax.numpy as jnp

def check_constant_feature(training, new_row, fit_standardizer, apply_standardizer):
    # TODO: fit once and transform training and new data
    statistics = ...
    training_result = ...
    new_result = ...
    return statistics, training_result, new_result




# Detect accidental batch-dependent preprocessing (Challenge):
# Transform two new rows together and separately using training statistics. The outputs should agree. Then show that re-fitting on the new batch changes the outputs.

# Inputs:
# - Statistics fitted on the original training matrix.
# - Two new rows transformed jointly and one at a time.

# Outputs:
# - A fixed-statistics comparison and a separate refit comparison.

# Invariants to check:
# - A row’s transform is independent of its request-batch companions.
# - Refitting on new rows is a distinct operation.

# Hints:
# - Reuse identical training statistics for both application paths.
# - Keep the newly fitted result separately labeled.

import jax.numpy as jnp

def audit_batch_independence(new_rows, statistics, apply_standardizer, fit_standardizer):
    # TODO: compare joint and separate application, then compare a refit
    together = ...
    separately = ...
    refitted = ...
    return together, separately, refitted

