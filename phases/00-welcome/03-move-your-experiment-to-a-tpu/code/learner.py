"""Move your experiment to a TPU · learner-only exercises (2/2 contracts ready)"""

# Exercise: Predict the effect of replacing the bias by minus one quarter. Run a second compiled prediction on the selected device. Compare against the original prediction plus the expected constant shift and an independent NumPy calculation.

# Inputs:
# - The original inputs, weights, prediction function, and selected device.
# - A compiled prediction path and changed bias.

# Outputs:
# - The changed device result, an independent NumPy comparison, and device metadata.

# Invariants to check:
# - The observed result comes from the selected device.
# - Check against both a shift-based prediction and an independent calculation.
# - Unavailable requested hardware cannot silently pass as CPU evidence.

# Hints:
# - Keep the original model inputs fixed.
# - Wait for device work before transferring a result for comparison.

import jax
import jax.numpy as jnp
import numpy as np

def check_changed_bias(x, weight, predict, device):
    # TODO: run the changed compiled prediction on the selected device
    changed = ...
    # TODO: compare with an independent host calculation
    raise NotImplementedError




# Test row order without changing the model (Transfer):
# Reverse the observation rows and predict the result order before execution. Verify the first and last values, the complete vector and device placement. Explain why this is a stronger check than only comparing the vector sum.

# Inputs:
# - Observation rows, model weights, and the selected device.
# - A reversed copy of the input rows.

# Outputs:
# - The reversed prediction vector and a NumPy reference.

# Invariants to check:
# - Output row order follows input row order.
# - Compare the full vector, not only its sum.
# - The result is placed on the selected device.

# Hints:
# - Reverse a host copy so the original stays unchanged.
# - Check endpoints and every value after execution completes.

import jax
import numpy as np

def check_reversed_rows(x_host, weights, device, predict):
    # TODO: place reversed rows, predict, and compare the full result
    raise NotImplementedError

