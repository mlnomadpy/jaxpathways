"""ModelOps: own, approve, roll back and retire a model · learner-only exercises (2/2 contracts ready)"""

# Exercise: Change the evaluation hash while keeping the model unchanged, and prove the old approval no longer applies.

# Inputs:
# - The unchanged model bundle with a changed evaluation hash and the approval bound to the prior hash.

# Outputs:
# - A rejection showing the old approval no longer applies, with the active selection unchanged.

# Invariants to check:
# - Approval is bound to the evaluation hash as well as the model; changing that hash invalidates reuse.

# Hints:
# - Compare the approval’s evaluation hash with the current bundle hash.
# - Verify that rejection leaves the selected artifact ID untouched.

def check_changed_evaluation_hash(bundle, prior_approval, active_selection):
    # TODO: Reject reuse of approval after the evaluation hash changes and preserve the current selection.
    raise NotImplementedError




# Review a changed owner (transfer):
# Change the responsible owner and check whether the old bundle approval can be reused.

# Inputs:
# - Change the responsible owner and check whether the old bundle approval can be reused.

# Outputs:
# - An approval rejection for the owner mismatch and unchanged selection.

# Invariants to check:
# - The old approval does not bind to the newly named owner.

# Hints:
# - The owner is part of the reviewed release identity.
# - Check owner identity in addition to bundle digest and expiry.

import hashlib
import json
import math
import tempfile
from pathlib import Path

def practice_review_a_changed_owner(bundle, approval, new_owner, now):
    """Complete the review a changed owner check."""
    # TODO: ('Change the responsible owner and check the old approval bundle against the new owner metadata.', 'Old approval rejection and an unchanged active selection.', 'Approval identity includes the owner as well as the artifact content hash.')
    raise NotImplementedError

