"""Track experiments and lineage with MLflow · learner-only exercises (2/2 contracts ready)"""

# Exercise: Add a third, deliberately wrong model artifact and show why the smallest training loss alone is insufficient to select a release.

# Inputs:
# - The fitted candidate parameters, held-out inputs and labels, and the current release candidate selected from the two trained runs.

# Outputs:
# - The deliberately wrong third candidate’s held-out loss and a decision that rejects it despite its low training loss.

# Invariants to check:
# - Release selection uses held-out evidence; a low training loss alone cannot override the candidate’s worse validation result.

# Hints:
# - Evaluate the zero-weight, zero-bias candidate on the existing held-out set.
# - Compare its validation error with the selected candidate and record why it is not released.

import jax.numpy as jnp

def evaluate_third_release_candidate(candidate_params, heldout_inputs, heldout_targets, selected_validation_loss):
    # TODO: Measure the deliberately wrong candidate on held-out rows and decide whether it passes release selection.
    raise NotImplementedError




# Change a data value (transfer):
# Change one training target and compute the data fingerprint again without overwriting the original record.

# Inputs:
# - Change one training target and compute the data fingerprint again without overwriting the original record.

# Outputs:
# - Original and changed fingerprints with the source record preserved.

# Invariants to check:
# - The only dataset change is one target value, and its fingerprint is distinct.

# Hints:
# - Hash the same input bytes plus changed target bytes.

import hashlib
import json
from pathlib import Path
import numpy as np
import mlflow
from mlflow import MlflowClient

def practice_change_a_data_value(features, labels):
    """Implement the change a data value check."""
    # TODO: Copy the training targets, change one value, and recompute the content fingerprint without overwriting the original.
    raise NotImplementedError

