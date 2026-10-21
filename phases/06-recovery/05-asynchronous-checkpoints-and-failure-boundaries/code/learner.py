"""Asynchronous checkpoints and failure boundaries · learner-only exercises (2/2 contracts ready)"""

# Exercise: Verify and publish the already written second checkpoint. Restore it and compare the next functional transition against the live state. Keep the original snapshot unchanged.

# Inputs:
# - The already written second-checkpoint path, its live state, and the original step-one snapshot.

# Outputs:
# - The restored second state, a comparison of its next transition with the live continuation, and a published step-two pointer.

# Invariants to check:
# - The restored step-two state matches the live state; both next transitions agree; the original snapshot remains at step one.

# Hints:
# - Restore the second checkpoint against a target shaped like the live state.
# - Publish only after restored values and the next transition have been checked.

import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

def verify_publish_and_compare(checkpoint_path, live_state, original_snapshot, publish_fn):
    # TODO: Restore step two, compare its next transition with the live continuation, then publish it.
    raise NotImplementedError




# Keep only a genuinely accepted target (Transfer):
# Create a manifest pointing to a nonexistent checkpoint in a disposable variable. Write a preflight that rejects it without replacing the real accepted manifest. Explain why a preflight alone cannot prove numerical correctness.

# Inputs:
# - Create a manifest pointing to a nonexistent checkpoint in a disposable variable. Write a preflight that rejects it without replacing the real accepted manifest. Explain why a preflight alone cannot prove numerical correctness.

# Outputs:
# - Rejected missing target and byte-identical accepted manifest.

# Invariants to check:
# - Preflight failure never publishes or replaces the accepted checkpoint pointer.

# Hints:
# - Check path existence and the declared step before restoration; still compare restored contents afterward.

import json
import tempfile
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

def practice_keep_only_a_genuinely_accepted_target(accepted_manifest, candidate_path):
    """Implement the keep only a genuinely accepted target check."""
    # TODO: Build a disposable manifest for a nonexistent checkpoint and reject it without modifying the accepted manifest.
    raise NotImplementedError

