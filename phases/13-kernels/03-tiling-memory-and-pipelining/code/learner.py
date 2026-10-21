"""Tiling, memory, and pipelining · learner-only exercises (3/3 contracts ready)"""

# Exercise: Use a (9,129) random fixture and verify synchronous, double-input-buffered and triple-input-buffered pipelines for both float32 and bfloat16 represented-input contracts. Keep output buffering at two slots.

# Inputs:
# - Two frozen (9,129) arrays generated from a local NumPy random generator with seed 23.
# - Float32 and bfloat16 represented-input variants, valid tile, and buffer modes (2 input slots synchronous, 2 buffered, 3 buffered).

# Outputs:
# - Outputs for all three schedules at each dtype.
# - Exact comparison with the reference computed from the represented input values.

# Invariants to check:
# - The same source values are used across schedules.
# - Every schedule uses two output slots; the synchronous case is separate from buffered execution.

# Hints:
# - Cast the reference operands to float32 for arithmetic, then cast the result to the requested output dtype.
# - Keep input and output buffer counts separate.

import numpy as np
import jax.numpy as jnp

def compare_pipeline_modes(run_pipeline, seed, tile, dtypes):
    # TODO: generate the two fixed arrays once and compare all schedule/dtype combinations.
    raise NotImplementedError("Compare synchronous and buffered pipelines on fixed represented inputs")



# Reject an unsupported pipeline tile (Practice):
# Try a (3,5) block and verify that the pipeline wrapper refuses it rather than using the permissive arithmetic interpreter as a hidden fallback.

# Inputs:
# - Two shape-compatible arrays and the pipeline wrapper.
# - A supported tile for the valid control case and unsupported tile (3,5).

# Outputs:
# - A clear ValueError for unsupported geometry before arithmetic interpretation or accelerator launch.

# Invariants to check:
# - The valid control establishes that inputs are otherwise acceptable.
# - The arithmetic interpreter is never used as a hidden fallback for the rejected pipeline configuration.

# Hints:
# - Place supported-tile validation before selecting execution backend.
# - Use a test double to confirm no arithmetic path was invoked.

def reject_unsupported_pipeline_tile(pipeline, first, second, supported_tile):
    # TODO: confirm the supported control works, then require rejection for tile (3,5).
    raise NotImplementedError("Reject unsupported pipeline geometry before dispatch")



# Estimate the price of a third input buffer (Challenge):
# Compute the declared data-buffer footprint for two and three input slots at each tile size, with two output slots in both cases. State at least two omitted memory costs.

# Inputs:
# - Matrix shape (9,129), candidate tile sizes, element size, and two output buffer slots.

# Outputs:
# - Data-buffer byte counts for two and three input slots.
# - Explicit list of excluded memory costs.
# - At least two memory costs explicitly excluded from the data-buffer estimate.

# Invariants to check:
# - Output slots stay fixed at two in both estimates.
# - The third input slot adds one input-tile allocation.
# - The footprint is a declared-buffer model, not total HBM usage.

# Hints:
# - Count padded tile elements times bytes per element and slots.
# - Do not include compiler, allocator, or runtime overhead in this data-only estimate.

import math

def estimate_pipeline_buffers(shape, tile_sizes, bytes_per_element):
    # TODO: compute padded input tile bytes for two/three slots and fixed output slots.
    # Return excluded costs alongside estimates.
    raise NotImplementedError("Separate declared data buffers from total device memory")
