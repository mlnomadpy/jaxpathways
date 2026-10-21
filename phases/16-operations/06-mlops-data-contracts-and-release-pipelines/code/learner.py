"""MLOps: data contracts, CI gates and monitoring · learner-only exercises (2/2 contracts ready)"""

# Exercise: Show that averaging slice means without counts changes the overall error.

# Inputs:
# - The two slice mean errors and their example counts.

# Outputs:
# - The unweighted mean of slice means and the count-weighted overall error for comparison.

# Invariants to check:
# - Unequal slice counts can make the unweighted average differ from the overall per-example mean.

# Hints:
# - Multiply each slice mean by its count to recover its error sum.
# - Divide the combined sum by the combined count.

import numpy as np

def compare_slice_averages(slice_means, slice_counts):
    # TODO: Compute the simple mean of slice means and the count-weighted overall mean.
    raise NotImplementedError




# Change the population mix (transfer):
# Keep the two slice errors, but evaluate nine high-slice examples and one low-slice example.

# Inputs:
# - Keep the two slice errors, but evaluate nine high-slice examples and one low-slice example.

# Outputs:
# - A weighted overall error under the new slice counts.

# Invariants to check:
# - Slice errors remain fixed while only population counts change.

# Hints:
# - Change counts rather than the model errors.
# - Weight slice errors by their example counts.

import math

def practice_change_the_population_mix(high_error, low_error, high_count, low_count):
    """Complete the change the population mix check."""
    # TODO: ('Keep high/low slice errors fixed and recompute weighted aggregate error for nine high-slice and one low-slice example.', 'New count-weighted aggregate error and unchanged slice metrics.', 'Weights come from the new example counts, not an unweighted mean of slice means.')
    raise NotImplementedError

