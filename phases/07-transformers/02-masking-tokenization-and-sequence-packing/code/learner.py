"""Masking, tokenization, and sequence packing · learner-only exercises (3/3 contracts ready)"""

# Exercise: Encode abcab with a three-character vocabulary. Create shifted inputs/targets, and a loss-eligibility vector for packed segments \([0,0,0,1,1]\) that excludes a cross-segment next-token target.

# Inputs:
# - Token IDs, shifted targets, segment IDs, and a validity mask.

# Outputs:
# - A causal attention mask and loss-eligibility mask that respect segment boundaries.

# Invariants to check:
# - No future token or cross-segment token is visible; padded queries have finite zero outputs.

# Hints:
# - Separate attention permission from target-loss eligibility.
# - Give all-masked padded queries an explicit fallback before applying the output mask.

import jax.numpy as jnp

def packed_causal_mask(segment_ids, valid):
    """A causal attention mask and loss-eligibility mask that respect segment boundaries."""
    # TODO: Build causal and segment-aware permission masks, then define which shifted targets enter loss.
    raise NotImplementedError




# Repair padding-query NaNs (Transfer / diagnosis):
# Use `valid=[True,True,False,False]`. Reproduce the undefined all-masked rows, then give padded queries a fallback and zero their outputs.

# Inputs:
# - Use `valid=[True,True,False,False]`. Reproduce the undefined all-masked rows, then give padded queries a fallback and zero their outputs.

# Outputs:
# - Finite attention outputs with padded query rows exactly zero.

# Invariants to check:
# - Valid queries preserve their allowed keys; only invalid padded-query rows use the fallback.

# Hints:
# - Insert a diagonal fallback only for invalid queries.

import jax.numpy as jnp

def practice_repair_padding_query_nans(valid, logits):
    """Implement the repair padding-query nans check."""
    # TODO: Create an all-masked padded query, add a numerically safe fallback for that row, and zero its output afterward.
    raise NotImplementedError




# Detect packed-example leakage (Transfer / diagnosis):
# Compare causal-only and segment-aware outputs at query \(2\). Explain which prior values caused the difference.

# Inputs:
# - Compare causal-only and segment-aware outputs at query \(2\). Explain which prior values caused the difference.

# Outputs:
# - Two outputs and a boundary-leakage difference check.

# Invariants to check:
# - A packed query attends only to causal keys from its own segment.

# Hints:
# - Query \(2\) should see only value \(8\) from its own segment.

import jax.numpy as jnp

def practice_detect_packed_example_leakage(segment_ids, values):
    """Implement the detect packed-example leakage check."""
    # TODO: Compare outputs at query two with causal-only and segment-aware masks, then identify the prior-segment values that change the result.
    raise NotImplementedError

