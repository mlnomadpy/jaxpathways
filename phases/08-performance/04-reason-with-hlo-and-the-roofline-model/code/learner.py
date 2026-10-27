"""Reason with HLO and the roofline model · learner-only exercises (2/2 contracts ready)"""

# Exercise: Derive the work and minimum traffic for shapes \((48,24)\) and \((24,12)\). Compare your hand calculation with the contract function and report the hypothetical limiting budget.

# Inputs:
# - Matrix dimensions, element storage size, and a declared hardware budget.

# Outputs:
# - FLOP count, minimum-byte estimate, arithmetic intensity, and hypothetical roofline ceiling.

# Invariants to check:
# - State the storage and traffic assumptions; a bound is not observed throughput.

# Hints:
# - Count multiply-add work and both input/output bytes independently.
# - Change only itemsize to isolate its effect on the estimate.

import numpy as np

def roofline_contract(m, k, n, itemsize):
    """FLOP count, minimum-byte estimate, arithmetic intensity, and hypothetical roofline ceiling."""
    # TODO: Count matrix FLOPs and minimum bytes from the declared dimensions and element size.
    raise NotImplementedError




# Keep the same shapes but change storage precision (Transfer):
# Change only the assumed element size from four bytes to two bytes. Calculate the resulting traffic and intensity, and explain why this is insufficient to predict an actual speedup.

# Inputs:
# - Change only the assumed element size from four bytes to two bytes. Calculate the resulting traffic and intensity, and explain why this is insufficient to predict an actual speedup.

# Outputs:
# - Changed byte estimate and intensity under the explicit storage assumption.

# Invariants to check:
# - Only the bytes per element change; the operation count and shapes stay fixed.

# Hints:
# - The model assumes both inputs and output use that element size; real accumulation and conversion may differ.

import numpy as np

def practice_keep_the_same_shapes_but_change_storage_precision(m, k, n, itemsize):
    """Implement the keep the same shapes but change storage precision check."""
    # TODO: Recompute minimum traffic and arithmetic intensity with two-byte elements while holding dimensions fixed.
    raise NotImplementedError

