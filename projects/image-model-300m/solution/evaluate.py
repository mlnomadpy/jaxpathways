"""Independent-input image classification, calibration, retrieval and baseline metrics."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or temperature <= 0 or not np.isfinite(values).all():
        raise ValueError("logits must be finite rank-2 values and temperature must be positive")
    shifted = values / temperature
    shifted -= np.max(shifted, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    normalizer = probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(normalizer).all() or np.any(normalizer <= 0):
        raise ValueError("softmax normalization failed")
    return probabilities / normalizer


def classification_metrics(
    logits: np.ndarray, labels: np.ndarray, classes: int, temperature: float = 1.0
) -> dict[str, Any]:
    scores = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 2 or scores.shape != (targets.size, classes) or targets.ndim != 1:
        raise ValueError("logits and labels have inconsistent shapes")
    if targets.size == 0 or not np.isfinite(scores).all() or np.any(targets < 0) or np.any(targets >= classes):
        raise ValueError("classification inputs are empty, nonfinite, or have invalid labels")
    probabilities = _softmax(scores, temperature)
    predictions = np.argmax(probabilities, axis=1)
    order = np.argsort(-probabilities, axis=1, kind="stable")
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    support = confusion.sum(axis=1)
    recalls = np.divide(np.diag(confusion), support, out=np.zeros(classes, dtype=np.float64), where=support > 0)
    one_hot = np.eye(classes, dtype=np.float64)[targets]
    nll = float(-np.mean(np.log(np.maximum(probabilities[np.arange(targets.size), targets], 1e-300))))
    brier = float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1)))
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    bin_edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    bins = []
    for index in range(10):
        mask = (confidence >= bin_edges[index]) & (
            confidence < bin_edges[index + 1] if index < 9 else confidence <= bin_edges[index + 1]
        )
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        mean_confidence = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / targets.size * abs(accuracy - mean_confidence)
        bins.append(
            {
                "lower": float(bin_edges[index]),
                "upper": float(bin_edges[index + 1]),
                "count": count,
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
            }
        )
    return {
        "count": int(targets.size),
        "top1_accuracy": float(correct.mean()),
        "top3_accuracy": float(np.mean(np.any(order[:, : min(3, classes)] == targets[:, None], axis=1))),
        "per_class_recall": recalls.tolist(),
        "macro_balanced_accuracy": float(recalls.mean()),
        "confusion_matrix": confusion.tolist(),
        "nll": nll,
        "brier": brier,
        "ece_10_equal_width": float(ece),
        "ece_bins": bins,
        "temperature": float(temperature),
    }


def component_bootstrap(
    logits: np.ndarray,
    labels: np.ndarray,
    component_ids: list[str],
    *,
    classes: int,
    temperature: float = 1.0,
    replicates: int = 500,
    seed: int = 8675309,
) -> dict[str, Any]:
    """Percentile intervals from source-component resampling.

    All images from a sampled component move together. A repeated component
    contributes its rows repeatedly, as required by cluster bootstrap
    resampling with replacement. The function reports undefined replicates
    (for example, a resample missing a class) rather than silently dropping
    them from the denominator.
    """
    scores = np.asarray(logits)
    targets = np.asarray(labels)
    if (
        scores.ndim != 2
        or targets.shape != (scores.shape[0],)
        or len(component_ids) != scores.shape[0]
        or replicates <= 0
        or len(set(component_ids)) == 0
        or any(not isinstance(item, str) or not item for item in component_ids)
    ):
        raise ValueError("bootstrap inputs, components, or replicate count are invalid")
    if not np.isfinite(scores).all():
        raise ValueError("bootstrap logits must be finite")
    groups = sorted(set(component_ids))
    members = {group: np.flatnonzero(np.asarray(component_ids) == group) for group in groups}
    rng = np.random.default_rng(seed)
    metrics = ("top1_accuracy", "top3_accuracy", "macro_balanced_accuracy", "nll", "brier", "ece_10_equal_width")
    draws = {name: [] for name in metrics}
    undefined = 0
    for _ in range(replicates):
        sampled = rng.choice(len(groups), size=len(groups), replace=True)
        indices = np.concatenate([members[groups[index]] for index in sampled])
        if set(targets[indices].tolist()) != set(range(classes)):
            undefined += 1
            continue
        current = classification_metrics(scores[indices], targets[indices], classes, temperature)
        for name in metrics:
            draws[name].append(float(current[name]))
    intervals = {}
    for name, values in draws.items():
        intervals[name] = {
            "lower_2_5_percent": float(np.quantile(values, 0.025)) if values else None,
            "upper_97_5_percent": float(np.quantile(values, 0.975)) if values else None,
            "defined_replicates": len(values),
            "undefined_replicates": undefined,
        }
    return {
        "method": "source-component percentile bootstrap",
        "seed": int(seed),
        "replicates": replicates,
        "metrics": intervals,
    }


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    candidates = np.exp(np.linspace(np.log(0.05), np.log(20.0), 601))
    evaluated = [
        (classification_metrics(logits, labels, np.asarray(logits).shape[1], float(value))["nll"], float(value))
        for value in candidates
    ]
    best_nll = min(value[0] for value in evaluated)
    tied = [value for value in evaluated if abs(value[0] - best_nll) <= 1e-14]
    nll, temperature = min(tied, key=lambda item: (abs(np.log(item[1])), item[1]))
    return {"temperature": temperature, "recipe_dev_nll": nll, "grid_count": 601, "lower": 0.05, "upper": 20.0}


def color_features(images: np.ndarray) -> np.ndarray:
    values = np.asarray(images, dtype=np.float32)
    if values.ndim != 4 or values.shape[1:] != (64, 64, 3) or not np.isfinite(values).all():
        raise ValueError("color baseline expects finite normalized NHWC 64x64 RGB images")
    if values.min() < -1.00001 or values.max() > 1.00001:
        raise ValueError("color baseline inputs must be normalized to [-1,1]")
    cells = values.reshape(values.shape[0], 4, 16, 4, 16, 3).mean(axis=(2, 4))
    return cells.reshape(values.shape[0], 48)


def fit_color_centroids(images: np.ndarray, labels: np.ndarray, classes: int) -> dict[str, np.ndarray]:
    features = color_features(images)
    targets = np.asarray(labels, dtype=np.int64)
    if targets.shape != (features.shape[0],) or np.any(targets < 0) or np.any(targets >= classes):
        raise ValueError("fit labels do not match color features")
    means = []
    for class_index in range(classes):
        selected = features[targets == class_index]
        if selected.shape[0] == 0:
            raise ValueError(f"fit split has no examples for class {class_index}")
        means.append(selected.mean(axis=0))
    return {"centroids": np.stack(means), "feature_dim": np.asarray(48, dtype=np.int32)}


def color_logits(model: dict[str, np.ndarray], images: np.ndarray) -> np.ndarray:
    features = color_features(images)
    centroids = np.asarray(model["centroids"], dtype=np.float64)
    if centroids.ndim != 2 or centroids.shape[1] != 48 or not np.isfinite(centroids).all():
        raise ValueError("color centroid table is malformed")
    distances = np.sum(np.square(features[:, None, :] - centroids[None, :, :]), axis=-1)
    return -distances


def retrieval_metrics(
    features: np.ndarray,
    ids: list[str],
    labels: np.ndarray,
    component_ids: list[str],
    *,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, Any]:
    vectors = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    n = len(ids)
    if vectors.ndim != 2 or vectors.shape[0] != n or targets.shape != (n,) or len(component_ids) != n:
        raise ValueError("retrieval arrays have inconsistent row counts")
    if len(set(ids)) != n or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("retrieval IDs must be nonempty and unique")
    if len(set(component_ids)) == 0 or any(not item for item in component_ids):
        raise ValueError("source component IDs must be nonempty")
    if not np.isfinite(vectors).all():
        raise ValueError("retrieval features must be finite")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("zero retrieval vectors are not ranked")
    normalized = vectors / norms[:, None]
    stable_ids = np.asarray(ids, dtype=str)
    rankings: dict[str, list[str]] = {}
    per_query = {}
    recall = {int(k): [] for k in ks}
    aps = []
    no_positive = []
    for index, query_id in enumerate(ids):
        allowed = np.asarray([j != index and component_ids[j] != component_ids[index] for j in range(n)], dtype=bool)
        candidates = np.flatnonzero(allowed)
        scores = normalized[candidates] @ normalized[index]
        order = np.lexsort((stable_ids[candidates], -scores))
        ranked_indices = candidates[order]
        ranked_ids = [ids[j] for j in ranked_indices]
        rankings[query_id] = ranked_ids
        relevant = np.asarray([targets[j] == targets[index] for j in ranked_indices], dtype=bool)
        total_relevant = int(
            np.sum(
                (targets == targets[index]) & np.asarray([component_ids[j] != component_ids[index] for j in range(n)])
            )
        )
        if total_relevant == 0:
            no_positive.append(query_id)
            per_query[query_id] = {
                "relevant_candidates": 0,
                "average_precision": None,
                "recall": {str(k): None for k in ks},
            }
            continue
        cumulative = np.cumsum(relevant)
        precision_at_rank = cumulative / (np.arange(relevant.size) + 1)
        ap = float(np.sum(precision_at_rank * relevant) / total_relevant)
        aps.append(ap)
        recalls = {}
        for k in ks:
            value = float(np.sum(relevant[:k]) / total_relevant)
            recall[int(k)].append(value)
            recalls[str(k)] = value
        per_query[query_id] = {"relevant_candidates": total_relevant, "average_precision": ap, "recall": recalls}
    return {
        "query_count": n,
        "evaluated_query_count": n - len(no_positive),
        "no_positive_query_ids": no_positive,
        "recall_at_k": {str(k): float(np.mean(values)) if values else None for k, values in recall.items()},
        "mean_average_precision": float(np.mean(aps)) if aps else None,
        "per_query": per_query,
        "rankings": rankings,
    }


def corruptions(images: np.ndarray) -> dict[str, np.ndarray]:
    """Apply declared final-set conditions to normalized 64x64 images."""
    values = np.asarray(images, dtype=np.float32)
    if values.ndim != 4 or values.shape[1:] != (64, 64, 3) or not np.isfinite(values).all():
        raise ValueError("corruptions expect finite normalized 64x64 RGB images")
    result = {
        "original": values.copy(),
        "brightness_plus_0.05": np.clip(values + 0.05, -1.0, 1.0),
        "contrast_0.90": np.clip(
            (values - values.mean(axis=(1, 2, 3), keepdims=True)) * 0.90 + values.mean(axis=(1, 2, 3), keepdims=True),
            -1.0,
            1.0,
        ),
        "center_8x8_gray0": values.copy(),
    }
    result["center_8x8_gray0"][:, 28:36, 28:36, :] = 0.0
    blurred = []
    jpeg = []
    for image in values:
        display = np.clip((image + 1.0) * 127.5, 0.0, 255.0).round().astype(np.uint8)
        pil = Image.fromarray(display)
        blur = np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=np.float32)
        blurred.append(blur / 127.5 - 1.0)
        stream = io.BytesIO()
        pil.save(stream, format="JPEG", quality=60, subsampling=0, optimize=False)
        stream.seek(0)
        with Image.open(stream) as decoded:
            pixels = np.asarray(decoded.convert("RGB"), dtype=np.float32)
        jpeg.append(pixels / 127.5 - 1.0)
    result["gaussian_blur_sigma1"] = np.stack(blurred)
    result["jpeg_quality60"] = np.stack(jpeg)
    return result
