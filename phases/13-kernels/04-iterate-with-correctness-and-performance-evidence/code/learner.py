"""Iterate with correctness and performance evidence · learner-only exercises (3/3 contracts ready)"""

# Exercise: Add a (15,255) fixture, compare two tile choices and two buffer counts, and retain the same frozen inputs for every candidate. Report errors and padded work separately.

# Inputs:
# - Two frozen float32 inputs shaped (15,255), generated once for all candidates.
# - Two tile choices, (8,128) and (16,256), and two/three input-buffer counts.

# Outputs:
# - Numerical error against the independent NumPy expression for each candidate.
# - Padded work calculated from tile geometry and reported separately from numerical error.

# Invariants to check:
# - All candidates receive the same input arrays.
# - Logical output shape remains (15,255); error and padded work are distinct measurements.

# Hints:
# - Generate the matrix once, outside the candidate loop.
# - Compute padded work from launch geometry rather than wall-clock time.

import numpy as np

def compare_kernel_candidates(run_candidate, first, second, tiles, input_buffer_counts):
    # TODO: keep the fixture fixed while comparing each tile/buffer combination.
    raise NotImplementedError("Compare numerical error and padded work on identical inputs")



# Reject unsupported dtype mixing (Practice):
# Pass a float32 matrix and a bfloat16 matrix together, then pass integer arrays. Verify neither is silently coerced into an undocumented kernel contract.

# Inputs:
# - Float32 and bfloat16 matrices plus integer matrix inputs.
# - A wrapper with an observable launch callback.

# Outputs:
# - Rejection for mixed floating dtypes and integer inputs before launch.

# Invariants to check:
# - The wrapper does not silently cast unsupported inputs.
# - No launch occurs on either invalid case.

# Hints:
# - Compare declared dtype contracts before conversion or dispatch.
# - Test each invalid pair independently.

def reject_unsupported_dtypes(wrapper, float_matrix, bf16_matrix, integer_matrix):
    # TODO: assert mixed and integer inputs fail before launch.
    raise NotImplementedError("Enforce the documented same-floating-dtype contract")



# Bound the end-to-end opportunity (Challenge):
# A profile attributes \(30\%\) of time to a local operation. Derive the best end-to-end speedup if that operation becomes twice as fast, then the impossible-to-exceed limit if its cost vanished.

# Inputs:
# - Amdahl fraction 0.30 and local speedup 2.0.

# Outputs:
# - End-to-end speedup for the stated improvement and the zero-cost limit.

# Invariants to check:
# - Unaccelerated fraction remains unchanged.
# - The limiting result is finite and bounded by the accelerated fraction.

# Hints:
# - Apply S=1/((1-p)+p/s).
# - Set s to infinity for the zero-cost limit.

def amdahl_bounds(fraction, local_speedup):
    # TODO: return achieved and asymptotic end-to-end speedup.
    raise NotImplementedError("Bound system-level gains from a local optimization")
