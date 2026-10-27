"""Meet your arrays and devices · learner-only exercises (3/3 contracts ready)"""

# Exercise: Compute column sums instead. State the resulting shape before running the code.

# Inputs:
# - A two-row, three-feature JAX array named `x`.
# - The lesson has checked its dtype and device.

# Outputs:
# - Column totals and an explicit output-shape check.

# Invariants to check:
# - Reduce the observation axis and preserve the feature axis.
# - Check shape separately from numeric values.

# Hints:
# - Read the dimensions before choosing a reduction axis.
# - The result retains the feature dimension.

import jax.numpy as jnp

def column_summary(x):
    # TODO: reduce observations and return feature-wise values and shape
    values = ...
    shape = ...
    return values, shape




# Add a new observation (Practice):
# Append \([6,7,8]\). Predict row sums and column means, then check both.

# Inputs:
# - The original two-row table and the new observation `[6, 7, 8]`.

# Outputs:
# - The extended table, row totals, and feature means.

# Invariants to check:
# - Appending adds an observation without changing feature count.
# - Row and feature reductions preserve opposite axes.

# Hints:
# - Concatenate along the observation axis.
# - Write down the table shape before reducing.

import jax.numpy as jnp

def extend_and_summarize(table, new_row):
    # TODO: append one observation and compute both summaries
    extended = ...
    row_totals = ...
    feature_means = ...
    return extended, row_totals, feature_means




# Diagnose an impossible reshape (Challenge):
# Try reshaping six values to \((2,4)\). Explain the failure and repair the intended three-measurement table.

# Inputs:
# - A six-value array and target shape `(2, 4)`.
# - The intended table has three measurements per row.

# Outputs:
# - A captured incompatible reshape and a repaired three-feature layout.

# Invariants to check:
# - Reshape preserves the number of elements.
# - The repaired table retains every value.

# Hints:
# - Count required elements before reshaping.
# - Catch the shape error without discarding data.

import jax.numpy as jnp

def diagnose_reshape(values):
    # TODO: attempt the requested shape and capture its incompatibility
    attempted = ...
    repaired = ...
    return attempted, repaired

