"""Train, checkpoint, and generate · learner-only exercises (3/3 contracts ready)"""

# Exercise: Generate twelve new tokens from prompt bcab. Derive the expected cycle first; verify the string. Explain why success on this periodic example is separate from natural-language performance.

# Inputs:
# - The trained toy language-model parameters, prompt "bcab", and the periodic character vocabulary.

# Outputs:
# - The twelve generated characters, an independently derived expected cycle, and a comparison result.

# Invariants to check:
# - Generation follows the known periodic pattern for this fixture; this result does not establish natural-language performance.

# Hints:
# - Derive the next character from the repeating "abc" training pattern before invoking generation.
# - Keep the prompt and token count aligned with the requested continuation.

import jax.numpy as jnp

def check_periodic_generation(params, prompt, count, vocabulary):
    # TODO: Generate the requested continuation and compare it with the independently derived cycle.
    raise NotImplementedError




# Reject an incompatible checkpoint (Transfer / diagnosis):
# Save a fresh snapshot, edit only the vocabulary metadata, and prove the loader rejects it. Restore the metadata and verify recovery. Then forge `step=101` while retaining the 100-step optimizer state and verify that this mismatch is also rejected.

# Inputs:
# - Save a fresh snapshot, edit only the vocabulary metadata, and prove the loader rejects it. Restore the metadata and verify recovery. Then forge `step=101` while retaining the 100-step optimizer state and verify that this mismatch is also rejected.

# Outputs:
# - Two compatibility rejections and a valid restore after restoring metadata.

# Invariants to check:
# - Vocabulary and optimizer-step metadata must agree with the checkpoint payload.

# Hints:
# - Change one vocabulary mapping rather than numeric leaves.

import jax.numpy as jnp
import json
from pathlib import Path
from tempfile import TemporaryDirectory

def practice_reject_an_incompatible_checkpoint(checkpoint, vocabulary):
    """Implement the reject an incompatible checkpoint check."""
    # TODO: Edit only vocabulary metadata in a disposable checkpoint and separately forge the step; require each restore to reject.
    raise NotImplementedError




# Make prompt failures explicit (Transfer / diagnosis):
# Verify empty prompt rejection and unknown-character rejection. Record the difference between a ValueError for missing context and a KeyError for an unknown token.

# Inputs:
# - Verify empty prompt rejection and unknown-character rejection. Record the difference between a ValueError for missing context and a KeyError for an unknown token.

# Outputs:
# - A ValueError for missing context and a KeyError for an unknown token.

# Invariants to check:
# - Prompt validation runs before sampling and does not conflate missing and out-of-vocabulary input.

# Hints:
# - The vocabulary mapping intentionally accepts only a, b and c.

import jax.numpy as jnp

def practice_make_prompt_failures_explicit(prompt, vocabulary):
    """Implement the make prompt failures explicit check."""
    # TODO: Call generation with an empty prompt and an unknown character, and assert their distinct exception types.
    raise NotImplementedError

