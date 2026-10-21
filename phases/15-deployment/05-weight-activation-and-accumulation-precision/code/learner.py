"""Choose weight, activation and accumulation precision · learner-only exercises (4/4 contracts ready)"""

# Exercise: Add a zero-valued output channel and verify that its quantization scale is finite and its reconstruction is exactly zero.

# Inputs:
# - The output weight matrix with one added all-zero output channel and the channel-wise quantizer.

# Outputs:
# - The zero channel’s quantization scale and reconstructed values.

# Invariants to check:
# - The scale is finite and every reconstructed value for the all-zero channel is exactly zero.

# Hints:
# - Handle a zero channel maximum explicitly when forming its scale.
# - Inspect the reconstructed channel after quantization, not only its integer codes.

import jax.numpy as jnp

def check_zero_output_channel(weights, quantize_weights):
    # TODO: Add an all-zero output channel, run channel-wise quantization, and inspect its scale and reconstruction.
    raise NotImplementedError




# Compute W8A8 with a wide accumulator (Transfer):
# Use calibrated symmetric scales for \(x\) and per-channel scales for \(w\). Multiply integer codes with INT32 accumulation, rescale, and compare to the float baseline. Do not multiply int8 arrays directly.

# Inputs:
# - Use calibrated symmetric scales for \(x\) and per-channel scales for \(w\). Multiply integer codes with INT32 accumulation, rescale, and compare to the float baseline. Do not multiply int8 arrays directly.

# Outputs:
# - An integer-coded output, float baseline, and error comparison.

# Invariants to check:
# - Integer dot products widen to int32 before summation.

# Hints:
# - Cast both integer-code arrays before multiplication. The real accumulator unit is the activation scale multiplied by each output channel’s weight scale; compare with the unchanged floating-point reference.
# - Apply activation and per-output-channel weight scales after accumulation.

import numpy as np
import jax.numpy as jnp

def practice_compute_w8a8_with_a_wide_accumulator(activation, weights, activation_scale, weight_scales):
    """Complete the compute w8a8 with a wide accumulator check."""
    # TODO: ('Quantize activation and per-channel weight values, multiply integer codes with an int32 accumulator, rescale, and compare with float output.', 'Integer-path output, float reference, and per-output error.', 'Integer products are widened before accumulation; calibration scales are retained per channel.')
    raise NotImplementedError




# Change the code origin without changing the values (Transfer):
# Change the activation zero point from \(-3\) to \(17\), shift the input codes accordingly, and prove that predictions stay unchanged. Then feed codes representing real zero and identify the expected bias-only result.

# Inputs:
# - Change the activation zero point from \(-3\) to \(17\), shift the input codes accordingly, and prove that predictions stay unchanged. Then feed codes representing real zero and identify the expected bias-only result.

# Outputs:
# - Equivalent dequantized activations and output for both code origins.

# Invariants to check:
# - Real activation values remain fixed after code and zero-point shift together.

# Hints:
# - Shift the zero point and codes by the same amount before subtracting; verify representability before casting back to INT8.
# - Use the declared zero point to identify which codes represent real zero.

import numpy as np
import jax.numpy as jnp

def practice_change_the_code_origin_without_changing_the_values(codes, old_zero_point, new_zero_point):
    """Complete the change the code origin without changing the values check."""
    # TODO: ('Recenter activation codes from zero point -3 to 17, adjust codes consistently, and test the real-zero input response.', 'Matching predictions before/after recoding plus the expected bias-only result for real zero.', 'Dequantized real activation values are invariant under the zero-point change.')
    raise NotImplementedError




# Account for actual low-bit storage (Transfer / diagnosis):
# Calculate FP32 storage, ideal packed four-bit storage, and the actual int8-backed four-bit simulation for the fixture. Include per-output-channel FP32 scales; keep biases and runtime workspace explicitly outside this subtotal.

# Inputs:
# - Calculate FP32 storage, ideal packed four-bit storage, and the actual int8-backed four-bit simulation for the fixture. Include per-output-channel FP32 scales; keep biases and runtime workspace explicitly outside this subtotal.

# Outputs:
# - Separate ideal packed and actual simulated storage totals.

# Invariants to check:
# - Per-channel scales count as FP32 metadata and biases/workspace stay outside the subtotal.

# Hints:
# - Use nbytes for actual arrays. Packing two signed four-bit codes into one byte is not implemented by astype(int8).
# - Report the actual backing dtype separately from nominal bit width.

import numpy as np
import jax.numpy as jnp

def practice_account_for_actual_low_bit_storage(parameter_count, output_channels, bits):
    """Complete the account for actual low-bit storage check."""
    # TODO: ('Compute FP32, ideal packed four-bit, and int8-backed four-bit payload bytes including per-channel scales.', 'Three storage totals with biases/workspace explicitly excluded.', 'The simulated format is counted using its actual int8 backing storage, not ideal bit packing.')
    raise NotImplementedError

