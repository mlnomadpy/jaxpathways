"""Learner scaffold: preserve the entire training stream at a checkpoint."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def train_step(state, batch, config):
    """Keep the original one-batch API: return updated parameters, momentum, key, cursor."""
    raise NotImplementedError("implement the deterministic momentum and PRNG update")


def stream_step(state, targets_by_id, config):
    """Consume the next deterministic batch, including the short epoch tail."""
    raise NotImplementedError("advance the exact sample order, optimizer, key, and cursors")


def save_checkpoint(path, step, config, arrays):
    """Keep the original single-file API for compatibility with stages 1–4."""
    raise NotImplementedError("bind config and every array, then replace atomically")


def load_checkpoint(path, expected_config):
    """Keep the original loader API and reject changed configuration or bytes."""
    raise NotImplementedError("verify before returning restored arrays")


def initial_stream_state(source_ids, config, shape=(2,), initial_weight=None, initial_velocity=None):
    """Create explicit parameters, optimizer, key, and deterministic-order state."""
    raise NotImplementedError("freeze unique IDs, initial key, epoch order, and clocks")


def save_generation(root, state, config, identities, expected_parent=None, failpoint=None):
    """Prepare an immutable generation and atomically publish only a valid child."""
    raise NotImplementedError("write complete state and manifest before parent arbitration")


def restore_generation(root, expected_config, expected_identities):
    """Load only the committed generation matching caller-owned expectations."""
    raise NotImplementedError("check publication, schema, data/source identity, and clocks")
