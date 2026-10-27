"""Communication-efficient algorithms · learner-only exercises (2/2 contracts ready)"""

# Exercise: Repeat the gradient comparison after reversing both observation rows and labels and changing the initial weights. Verify every all-reduce and reconstructed local replica, then explain why reversing only the labels changes the problem.

# Inputs:
# - Reversed observation rows and labels, changed initial weights, and the distributed gradient routine from the lesson.

# Outputs:
# - Gradients and updated parameters after the changed data ordering.
# - Reconstructed local replicas and all-reduce comparisons.
# - A comparison showing why reversing labels without the corresponding rows changes the gradient problem.

# Invariants to check:
# - Reversing both paired arrays preserves examples; reversing labels alone changes targets.
# - Each local replica matches the globally updated parameter vector.

# Hints:
# - Apply the same row permutation to features and labels.
# - Compare local contributions, reduced gradients, and replica state independently.

import numpy as np

def verify_changed_distributed_step(features, labels, initial_weights, update):
    # TODO: reverse paired observations, change the initial state, and audit replicas.
    raise NotImplementedError("Check pairing, reduction, and replica agreement")



# Account for the complete communication cycle (Transfer):
# Compare a complete ring all-reduce with reduce-scatter followed by all-gather for a gradient of one mebibyte at two, four and eight ranks. Also compute per-rank gradient storage. Explain why lower storage does not guarantee fewer communicated bytes.

# Inputs:
# - A one-mebibyte gradient payload and rank counts 2, 4, and 8.
# - A ring collective communication model.

# Outputs:
# - Per-rank storage and communicated bytes for each complete strategy.

# Invariants to check:
# - The all-reduce comparison includes both reduce-scatter and all-gather phases.
# - Per-rank storage is distinct from network bytes transferred.

# Hints:
# - For a ring, account for (P-1) chunks in each phase.
# - Keep byte units explicit and divide payload by rank count only for shard storage.

def compare_ring_collectives(payload_bytes, rank_counts):
    # TODO: calculate complete all-reduce and reduce-scatter/all-gather costs.
    # Return storage and communication in separately named fields.
    raise NotImplementedError("Count both ring phases and per-rank storage")
