"""Practice with four virtual CPU devices · learner-only exercises (3/3 contracts ready)"""

# Exercise: Predict and verify the per-device shapes for twelve rows, then demonstrate how replication repairs an indivisible five-row placement without deleting data.

# Inputs:
# - Four virtual CPU devices and the row-sharded placement from the lesson.
# - Twelve rows with two features, plus a five-row placement case.

# Outputs:
# - Observed shard shapes and slices for twelve rows.
# - A captured uneven-placement failure and a replicated placement retaining all five rows.

# Invariants to check:
# - Every source row appears in the reference comparison.
# - Replication preserves data but does not prove uneven partition support.
# - Label logical CPU devices accurately.

# Hints:
# - Use shard indices to select matching reference rows.
# - Keep partition failure and replicated repair as separate observations.

import jax
import numpy as np

def audit_row_placements(row_sharding, devices):
    # TODO: place the requested arrays and inspect shard metadata
    raise NotImplementedError




# Move from four rows to twelve (Challenge):
# Create twelve two-feature rows with values \(0\) through \(23\). Predict shard shape and check every slice of a squared output.

# Inputs:
# - Four row-sharded devices and twelve two-feature rows.
# - The squared output for each row.

# Outputs:
# - Every output shard and its matching NumPy slice.

# Invariants to check:
# - Each shard index selects matching global rows.
# - Local shape matches an even partition.

# Hints:
# - Build the full reference before selecting slices.
# - Compare the complete shard payload.

import jax
import numpy as np

def verify_twelve_rows(row_sharding):
    # TODO: square the sharded data and compare every indexed slice
    raise NotImplementedError




# Repair a batch that does not divide evenly (Challenge):
# Place a \((5,2)\) array with the same row sharding. Capture the failure. Repair this placement demonstration by replicating it, and explain why that does not prove the partitioned algorithm supports uneven batches.

# Inputs:
# - The same four-way row sharding and a five-by-two input.

# Outputs:
# - A captured partitioning failure and a replicated placement with all rows.

# Invariants to check:
# - No row is dropped to force divisibility.
# - Replication is not described as uneven partition support.

# Hints:
# - Capture the expected placement error.
# - Inspect the replicated values and state the evidence limit.

import jax
import numpy as np

def diagnose_uneven_rows(row_sharding):
    # TODO: capture the partition attempt, then test replicated placement
    raise NotImplementedError

