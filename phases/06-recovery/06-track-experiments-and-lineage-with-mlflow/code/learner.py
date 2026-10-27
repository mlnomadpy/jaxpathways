"""Track experiments and lineage with MLflow · learner-only exercises (2/2 contracts ready)"""

# Exercise: Replay the two saved linear candidates in a fresh local MLflow experiment and add a third training-lookup model. The lookup stores each training x/y pair and predicts zero for an unseen x. Compute and log actual training and validation MSE for all three, save and reload every artifact, then show which model a training-only rule selects and which one the held-out rule selects.

# Inputs:
# - The two fitted linear artifact dictionaries from the example, training and untouched validation arrays.
# - A third training-lookup model with zero as its unseen-input default.

# Outputs:
# - Three local MLflow runs, each with actual training and held-out MSE and a model.json artifact.
# - Reloaded-prediction checks and distinct training-only versus validation-based selections.

# Invariants to check:
# - The third candidate has zero actual training MSE and validation MSE 6.
# - All comparisons use the same split and data identity; validation is not used to fit any candidate.
# - The selection uses retrieved run records and every artifact is reloaded.

# Hints:
# - Represent lookup inference explicitly and calculate both metrics from predictions.
# - Keep the exercise experiment in its own TemporaryDirectory and log all three model artifacts.

import numpy as np

def compare_three_saved_candidates(candidates, train_x, train_y, validation_x, validation_y):
    # TODO: log, reload and evaluate three candidates; compare selection rules.
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

