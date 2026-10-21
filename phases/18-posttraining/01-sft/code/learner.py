"""Supervised fine-tuning with response-only token loss · learner-only exercises (3/3 contracts ready)"""

# Exercise: Score only the first response token in each sequence and compare the resulting count and objective.

# Inputs:
# - Causal logits, token IDs, role labels, and a response-only target mask.

# Outputs:
# - The count and mean loss for only the first response token per sequence.

# Invariants to check:
# - Logits at time t score token t+1; prompt targets are excluded after shifting.

# Hints:
# - Shift token and role arrays consistently.
# - Count selected shifted targets before averaging.

import jax.numpy as jnp
import numpy as np

def score_first_response_token(logits, tokens, roles):
    # TODO: construct the shifted target mask and return the selected loss and count.
    raise NotImplementedError



# Reject a batch with no answer targets (Transfer):
# Validate an all-prompt batch before computing the token average.

# Inputs:
# - A batch whose roles mark every token as prompt context.

# Outputs:
# - A validation error or explicit skip result before averaging token loss.

# Invariants to check:
# - An all-prompt batch never creates a zero-denominator supervised update.

# Hints:
# - Build the mask for shifted target positions.
# - Choose and document reject-versus-skip behavior.

import jax.numpy as jnp
import numpy as np

def validate_response_targets(tokens, roles):
    # TODO: validate the shifted response mask and return its selected-token count.
    raise NotImplementedError



# Create a failure this model cannot fix (Challenge):
# Build two contexts ending in token zero but requiring different next tokens. Prove that the transition table gives identical distributions. Name the model capability needed to resolve the conflict.

# Inputs:
# - Two contexts with the same final token but different required next-token labels.

# Outputs:
# - The shared transition-table distributions and an explanation-ready conflict indicator.

# Invariants to check:
# - A first-order token transition model has identical outputs for identical final token IDs.

# Hints:
# - Encode the two examples with the exact same context state.
# - Compare full next-token distributions, not just top predictions.

import jax.numpy as jnp
import numpy as np

def expose_context_aliasing(context_a, context_b, labels, transition_table):
    # TODO: compare next-token distributions for contexts sharing the same final token.
    raise NotImplementedError
