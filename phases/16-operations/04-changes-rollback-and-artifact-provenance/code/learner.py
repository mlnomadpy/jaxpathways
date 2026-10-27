"""Changes, rollback, and artifact provenance · learner-only exercises (3/3 contracts ready)"""

# Exercise: Publish two validated models with distinct predictions at x=1. Record the consumer prediction while A is selected, after B activation, after a rejected candidate and after rollback. Change B’s slope while keeping the fixture validation threshold satisfied. Explain why publication alone does not change behavior and why the rejection must leave both selection and output unchanged.

# Inputs:
# - A candidate artifact, immutable payload/evidence hashes, validation record, and current release selection.

# Outputs:
# - An activation decision or rejection, followed by a rollback that selects the exact prior ID.

# Invariants to check:
# - Failed validation leaves the active pointer unchanged; approval binds to content and evidence identity.

# Hints:
# - Validate all required evidence before mutating the active selection.
# - Compare IDs and payload hashes after rollback, not only a display label.

from pathlib import Path
import hashlib
import json
import tempfile

def promote_validated_artifact(store, artifact_id, payload, evidence, release_note):
    """An activation decision or rejection, followed by a rollback that selects the exact prior ID."""
    # TODO: Verify content and evidence identity before switching the active ID; preserve that ID whenever validation fails.
    raise NotImplementedError




# Reject an incomplete provenance record (Transfer / diagnosis):
# Construct a payload claiming passing validation but lacking the worker-source hash. Verify that it cannot become active.

# Inputs:
# - Construct a payload claiming passing validation but lacking the worker-source hash. Verify that it cannot become active.

# Outputs:
# - A rejected candidate and the unchanged active artifact ID.

# Invariants to check:
# - Required source identity is present before activation.

# Hints:
# - A boolean flag does not replace required source/config/data evidence.
# - Treat a claimed validation pass and artifact provenance as separate fields.

from pathlib import Path
import hashlib
import json
import tempfile

def practice_reject_an_incomplete_provenance_record(candidate, required_hashes, active_id):
    """Complete the reject an incomplete provenance record check."""
    # TODO: ('Require a worker-source hash before activating a payload even if its validation status says passed.', 'A provenance rejection and unchanged active artifact ID.', 'Passing metrics do not substitute for missing source identity.')
    raise NotImplementedError




# Check rollback without history (Transfer / diagnosis):
# Attempt rollback before any previous artifact exists, then verify the first activation remains selected.

# Inputs:
# - Attempt rollback before any previous artifact exists, then verify the first activation remains selected.

# Outputs:
# - A rollback rejection while preserving first activation.

# Invariants to check:
# - No nonexistent previous version is selected.

# Hints:
# - A failed rollback must not clear the current selection.
# - Require a recorded rollback target.

from pathlib import Path
import hashlib
import json
import tempfile

def practice_check_rollback_without_history(store, active_id):
    """Complete the check rollback without history check."""
    # TODO: ('Attempt rollback before any prior artifact is recorded and verify the first active ID remains selected.', 'A rollback rejection and unchanged first activation.', 'No previous artifact means no rollback target exists.')
    raise NotImplementedError

