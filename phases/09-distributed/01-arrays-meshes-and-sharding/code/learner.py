"""Arrays, meshes, and sharding · learner-only exercises (3/3 contracts ready)"""

# Exercise: Compare row and column layouts of a \((12,8)\) array, and repair a ten-row mean with padding plus an explicit mask.

# Inputs:
# - Four logical CPU devices; row and column NamedSharding objects.
# - A (12, 8) matrix and a separate ten-observation, four-feature batch.

# Outputs:
# - Per-device local shapes and values for both (12, 8) layouts.
# - A masked ten-row column mean plus an unmasked padded mean for diagnosis.

# Invariants to check:
# - Both placements preserve the same global matrix and each shard matches its indexed host slice.
# - Padded rows contribute to neither the masked numerator nor the valid-row denominator.
# - The masked mean agrees with the original ten-row host mean.

# Hints:
# - Read each PartitionSpec against the axis it partitions.
# - Use shard.index for local-value comparisons; mask padded rows before summing and divide by valid rows.

import numpy as np
import jax
import jax.numpy as jnp

def verify_layouts_and_masked_mean(host, uneven_batch, row_sharding, column_sharding):
    # TODO: compare both placements and their indexed shard values.
    # Then pad the ten-row batch, place a validity mask, and compare masked/unmasked means.
    raise NotImplementedError("Verify both shard layouts and mask-preserved batch statistics")



# Change both dimensions (Challenge):
# Use a \((12,8)\) array. Verify row and column layouts and predict every local shape.

# Inputs:
# - A (12,8) host matrix and four logical CPU devices.
# - Row and column shardings from the lesson.

# Outputs:
# - Exact per-device shapes for both placements.

# Invariants to check:
# - Every shard contains the indexed global slice.
# - The global matrix is unchanged by placement.

# Hints:
# - Divide the partitioned dimension by four.
# - Check shard.index as well as shard.data.shape.

import numpy as np
import jax

def verify_changed_layout(host, row_sharding, column_sharding):
    # TODO: place the matrix under both shardings and check each local slice.
    raise NotImplementedError("Verify both local shapes and indexed values")



# Pad an uneven batch without corrupting its mean (Challenge):
# Ten observations cannot be row-partitioned over four devices. Pad to twelve and use a mask to recover the true column mean. Demonstrate the wrong unmasked mean.

# Inputs:
# - A ten-row host batch with four features and a four-device row sharding.

# Outputs:
# - Masked column mean with shape (4,).
# - Unmasked padded mean as a diagnostic.

# Invariants to check:
# - Padding contributes zero to both the numerator and valid-count denominator.
# - The masked result matches the original ten-row host mean.

# Hints:
# - Pad only the row axis to a device-divisible length.
# - Mask invalid rows before summing and divide by valid rows.

import numpy as np
import jax
import jax.numpy as jnp

def masked_column_mean(batch, row_sharding):
    # TODO: pad rows, place the values and mask, then compute both means.
    # Return device results as host arrays for comparison.
    raise NotImplementedError("Implement masked statistics for the ten valid rows")
