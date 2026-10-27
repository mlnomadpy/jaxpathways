"""Compute every metric from caller-owned predictions and IDs."""

from __future__ import annotations

import numpy as np


def classification_metrics(logits: np.ndarray, labels: np.ndarray, classes: int, temperature: float = 1.0):
    raise NotImplementedError("TODO: compute top-k, confusion, calibration, and loss metrics")


def fit_temperature(logits: np.ndarray, labels: np.ndarray):
    raise NotImplementedError("TODO: use the frozen 601-point recipe_dev-only grid")


def color_features(images: np.ndarray):
    raise NotImplementedError("TODO: compute the 4x4x3 fit-only color baseline features")


def fit_color_centroids(images: np.ndarray, labels: np.ndarray, classes: int):
    raise NotImplementedError("TODO: fit one centroid per class on fit images only")


def color_logits(model, images: np.ndarray):
    raise NotImplementedError("TODO: score images against fit-only centroids")


def retrieval_metrics(features, ids, labels, component_ids, *, ks=(1, 5, 10)):
    raise NotImplementedError("TODO: rank full query/gallery inventories with group exclusion")


def corruptions(images: np.ndarray):
    raise NotImplementedError("TODO: apply all declared fixed corruptions")
