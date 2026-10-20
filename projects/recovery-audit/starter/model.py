"""Starter: preserve optimizer and RNG state across an interrupted update."""
import hashlib, json, os, tempfile
from pathlib import Path
import numpy as np

def train_step(state, batch, config):
    raise NotImplementedError("update weight, momentum, cursor, and RNG key deterministically")
def save_checkpoint(path, step, config, arrays):
    raise NotImplementedError("write an integrity-bound atomic checkpoint")
def load_checkpoint(path, expected_config):
    raise NotImplementedError("verify checkpoint bytes and configuration before restore")
