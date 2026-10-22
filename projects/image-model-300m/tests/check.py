"""Independent staged checks for the TensorFlow flower photo lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import traceback
from pathlib import Path
from types import ModuleType

from PIL import Image

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import oracles


def load_candidate(specification: str) -> tuple[ModuleType, Path, str]:
    if specification in {"solution", "starter"}:
        module_path = ROOT / specification / "model.py"
        candidate_identity = specification
    else:
        module_path = Path(specification).expanduser().resolve()
        if module_path.is_dir():
            module_path = module_path / "model.py"
        candidate_identity = str(module_path)
    if not module_path.is_file():
        raise FileNotFoundError(f"candidate model source not found: {module_path}")
    parent = str(module_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    sys.modules.pop("model", None)
    spec = importlib.util.spec_from_file_location("model", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load candidate module {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["model"] = module
    spec.loader.exec_module(module)
    return module, module_path, candidate_identity


def load_objectives(candidate_path: Path, specification: str) -> ModuleType:
    if specification in {"solution", "starter"}:
        module_path = ROOT / specification / "objectives.py"
    else:
        module_path = candidate_path.parent / "objectives.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"candidate objective source not found: {module_path}")
    spec = importlib.util.spec_from_file_location("candidate_objectives", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load objective module {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_augmentation(candidate_path: Path, specification: str) -> ModuleType:
    if specification in {"solution", "starter"}:
        module_path = ROOT / specification / "augment.py"
    else:
        module_path = candidate_path.parent / "augment.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"candidate augmentation source not found: {module_path}"
        )
    spec = importlib.util.spec_from_file_location("candidate_augmentation", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load augmentation module {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_evaluation(candidate_path: Path, specification: str) -> ModuleType:
    if specification in {"solution", "starter"}:
        module_path = ROOT / specification / "evaluate.py"
    else:
        module_path = candidate_path.parent / "evaluate.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"candidate evaluation source not found: {module_path}")
    spec = importlib.util.spec_from_file_location("candidate_evaluation", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load evaluation module {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_training(candidate_path: Path, specification: str) -> ModuleType:
    if specification in {"solution", "starter"}:
        module_path = ROOT / specification / "training.py"
    else:
        module_path = candidate_path.parent / "training.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"candidate training source not found: {module_path}")
    spec = importlib.util.spec_from_file_location("candidate_training", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load training module {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_runner(candidate_path: Path, specification: str) -> ModuleType:
    if specification in {"solution", "starter"}:
        module_path = ROOT / specification / "runner.py"
    else:
        module_path = candidate_path.parent / "runner.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"candidate runner source not found: {module_path}")
    parent = str(module_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location("candidate_image_runner", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load candidate runner module {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def assert_close(actual, expected, *, atol: float, rtol: float, label: str) -> float:
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape:
        raise AssertionError(f"{label}: shape {a.shape} != {b.shape}")
    error = float(np.max(np.abs(a - b))) if a.size else 0.0
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{label}: max absolute error {error:.6g} exceeds tolerance"
        )
    return error


def _model_stage(candidate: ModuleType) -> dict:
    config = dict(candidate.PILOT_CONFIG)
    params = candidate.init_vit(jax.random.PRNGKey(711), config)
    head = candidate.init_linear_head(
        jax.random.PRNGKey(712), int(config["width"]), int(config["classes"])
    )
    x = (
        np.random.default_rng(713)
        .uniform(
            -0.95,
            0.95,
            size=(2, config["image_size"], config["image_size"], config["channels"]),
        )
        .astype(np.float32)
    )
    patch_error = assert_close(
        candidate.patchify(jnp.asarray(x), int(config["patch_size"])),
        oracles.patchify(x, int(config["patch_size"])),
        atol=0.0,
        rtol=0.0,
        label="row-major patch order",
    )
    round_trip = candidate.unpatchify(
        candidate.patchify(jnp.asarray(x), int(config["patch_size"])),
        int(config["image_size"]),
        int(config["patch_size"]),
        int(config["channels"]),
    )
    round_trip_error = assert_close(
        round_trip, x, atol=1e-7, rtol=0.0, label="patch round trip"
    )
    features = candidate.encode_images(params, jnp.asarray(x))
    feature_error = assert_close(
        features,
        oracles.vit_features(params, x),
        atol=4e-5,
        rtol=4e-5,
        label="NumPy pre-LN ViT oracle",
    )
    logits = candidate.classification_logits(params, head, jnp.asarray(x))
    logit_error = assert_close(
        logits,
        oracles.vit_logits(params, head, x),
        atol=4e-5,
        rtol=4e-5,
        label="classification logits oracle",
    )
    if (
        logits.shape != (2, int(config["classes"]))
        or not np.isfinite(np.asarray(logits)).all()
    ):
        raise AssertionError("ViT returned invalid logits")

    labels = jnp.asarray([1, 3], dtype=jnp.int32)

    def head_loss(kernel):
        changed_head = {**head, "kernel": kernel}
        out = candidate.classification_logits(params, changed_head, jnp.asarray(x))
        return jnp.mean(-jax.nn.log_softmax(out)[jnp.arange(2), labels])

    head_grad = np.asarray(jax.grad(head_loss)(head["kernel"]))
    head_kernel = np.asarray(head["kernel"], dtype=np.float64)
    head_fd = oracles.finite_difference(
        lambda value: oracles.cross_entropy(
            oracles.vit_logits(params, {**head, "kernel": value}, x), np.asarray(labels)
        ),
        head_kernel,
        (0, 0),
        step=1e-3,
    )
    head_grad_error = abs(float(head_grad[0, 0]) - head_fd)
    if head_grad_error > 2e-3:
        raise AssertionError(
            f"head gradient finite difference failed: {head_grad_error:.6g}"
        )

    changed_config = {
        **config,
        "image_size": 32,
        "patch_size": 8,
        "width": 16,
        "heads": 4,
        "mlp_dim": 24,
        "layers": 1,
    }
    changed_params = candidate.init_vit(jax.random.PRNGKey(714), changed_config)
    changed_x = (
        np.random.default_rng(715)
        .uniform(-1.0, 1.0, size=(1, 32, 32, 3))
        .astype(np.float32)
    )
    changed_features = candidate.encode_images(changed_params, jnp.asarray(changed_x))
    changed_error = assert_close(
        changed_features,
        oracles.vit_features(changed_params, changed_x),
        atol=4e-5,
        rtol=4e-5,
        label="changed image/model configuration oracle",
    )
    interpolated = candidate.encode_images(
        params, jnp.asarray(changed_x), allow_position_interpolation=True
    )
    if (
        interpolated.shape != (1, int(config["width"]))
        or not np.isfinite(np.asarray(interpolated)).all()
    ):
        raise AssertionError(
            "explicit position interpolation did not support a changed resolution"
        )

    key_a, key_b = jax.random.PRNGKey(716), jax.random.PRNGKey(717)
    stochastic_a = candidate.encode_images(
        params, jnp.asarray(x), dropout_key=key_a, train=True
    )
    stochastic_repeat = candidate.encode_images(
        params, jnp.asarray(x), dropout_key=key_a, train=True
    )
    stochastic_b = candidate.encode_images(
        params, jnp.asarray(x), dropout_key=key_b, train=True
    )
    assert_close(
        stochastic_a,
        stochastic_repeat,
        atol=0.0,
        rtol=0.0,
        label="explicit dropout replay",
    )
    if np.array_equal(np.asarray(stochastic_a), np.asarray(stochastic_b)):
        raise AssertionError(
            "different dropout keys produced identical training features"
        )
    try:
        candidate.encode_images(params, jnp.asarray(x), train=True)
    except ValueError:
        pass
    else:
        raise AssertionError("training dropout accepted a missing PRNG key")

    cnn = candidate.init_cnn(jax.random.PRNGKey(718), classes=5, channels=3)
    cnn_x = x[:1, :8, :8, :]
    cnn_logits = candidate.cnn_logits(cnn, jnp.asarray(cnn_x))
    cnn_error = assert_close(
        cnn_logits,
        oracles.cnn_logits(cnn, cnn_x),
        atol=5e-5,
        rtol=5e-5,
        label="NumPy convolution control oracle",
    )
    cnn_train = candidate.cnn_logits(
        cnn, jnp.asarray(cnn_x), dropout_key=jax.random.PRNGKey(719), train=True
    )
    if not np.isfinite(np.asarray(cnn_train)).all():
        raise AssertionError("CNN stochastic training output is nonfinite")

    # Owned negative control: mutate only the instructor reference runtime's
    # softmax behavior, then prove the oracle detects uniform attention. This
    # never inspects or edits the submitted candidate's source spelling.
    reference_path = ROOT / "solution" / "model.py"
    reference_spec = importlib.util.spec_from_file_location(
        "image_reference_model", reference_path
    )
    if reference_spec is None or reference_spec.loader is None:
        raise ImportError(
            "owned reference model is unavailable for the mutation control"
        )
    reference_model = importlib.util.module_from_spec(reference_spec)
    reference_spec.loader.exec_module(reference_model)
    reference_params = reference_model.init_vit(jax.random.PRNGKey(720))
    reference_head = reference_model.init_linear_head(jax.random.PRNGKey(721))
    original_softmax = jax.nn.softmax
    try:
        jax.nn.softmax = (
            lambda scores, axis=-1: jnp.ones_like(scores) / scores.shape[axis]
        )
        uniform_logits = reference_model.classification_logits(
            reference_params, reference_head, jnp.asarray(x)
        )
        try:
            assert_close(
                uniform_logits,
                oracles.vit_logits(reference_params, reference_head, x),
                atol=4e-5,
                rtol=4e-5,
                label="uniform-attention mutation",
            )
        except AssertionError:
            uniform_control_rejected = True
        else:
            uniform_control_rejected = False
    finally:
        jax.nn.softmax = original_softmax
    if not uniform_control_rejected:
        raise AssertionError(
            "independent oracle did not detect the uniform-attention mutation control"
        )

    return {
        "status": "pass",
        "pilot_vit_parameter_count": int(candidate.parameter_count(params)),
        "pilot_cnn_parameter_count": int(candidate.parameter_count(cnn)),
        "max_abs_errors": {
            "patch_order": patch_error,
            "round_trip": round_trip_error,
            "vit_features": feature_error,
            "classification_logits": logit_error,
            "changed_configuration": changed_error,
            "cnn_logits": cnn_error,
            "head_gradient_fd": head_grad_error,
        },
        "uniform_attention_mutation": "rejected by independent NumPy oracle",
        "dropout": "same key replays exactly; changed key changes output",
    }


def _numpy_multi_positive(
    projected: np.ndarray, source_positive: np.ndarray, temperature: float
) -> tuple[float, np.ndarray, np.ndarray]:
    values = np.asarray(projected, dtype=np.float64)
    count = values.shape[0] // 2
    positive = np.block(
        [[source_positive, source_positive], [source_positive, source_positive]]
    ).astype(bool)
    paired = np.block(
        [
            [np.zeros((count, count), dtype=bool), np.eye(count, dtype=bool)],
            [np.eye(count, dtype=bool), np.zeros((count, count), dtype=bool)],
        ]
    )
    positive |= paired
    positive &= ~np.eye(2 * count, dtype=bool)
    similarities = values @ values.T / temperature
    losses = []
    for row in range(2 * count):
        allowed = ~np.eye(2 * count, dtype=bool)[row]
        positives = positive[row]
        denominator_logits = similarities[row, allowed]
        positive_logits = similarities[row, positives]
        denominator = np.logaddexp.reduce(denominator_logits)
        numerator = np.logaddexp.reduce(positive_logits)
        losses.append(denominator - numerator)
    return float(np.mean(losses)), positive, similarities


def _crop_parameters_oracle(key: jax.Array, size: int) -> tuple[int, int, int, int]:
    area_key, ratio_key, top_key, left_key = jax.random.split(key, 4)
    area = float(jax.random.uniform(area_key, (), minval=0.75, maxval=1.0))
    log_ratio = float(
        jax.random.uniform(
            ratio_key, (), minval=np.log(0.90), maxval=np.log(1.1111111111)
        )
    )
    ratio = np.exp(log_ratio)
    height = int(np.rint(size * np.sqrt(area / ratio)))
    width = int(np.rint(size * np.sqrt(area * ratio)))
    if height < 1 or width < 1 or height > size or width > size:
        height = width = size
    top = int(jax.random.randint(top_key, (), 0, size - height + 1))
    left = int(jax.random.randint(left_key, (), 0, size - width + 1))
    return top, left, height, width


def _numpy_crop_resize(
    image: np.ndarray, bounds: tuple[int, int, int, int]
) -> np.ndarray:
    top, left, height, width = bounds
    size = image.shape[0]
    y = np.clip(
        top + (np.arange(size, dtype=np.float64) + 0.5) * height / size - 0.5,
        top,
        top + height - 1,
    )
    x = np.clip(
        left + (np.arange(size, dtype=np.float64) + 0.5) * width / size - 0.5,
        left,
        left + width - 1,
    )
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    y1, x1 = np.minimum(y0 + 1, top + height - 1), np.minimum(x0 + 1, left + width - 1)
    wy, wx = y - np.floor(y), x - np.floor(x)
    result = np.empty((size, size, image.shape[-1]), dtype=np.float64)
    for row in range(size):
        for col in range(size):
            tl, tr = image[y0[row], x0[col]], image[y0[row], x1[col]]
            bl, br = image[y1[row], x0[col]], image[y1[row], x1[col]]
            result[row, col] = (1 - wy[row]) * ((1 - wx[col]) * tl + wx[col] * tr) + wy[
                row
            ] * ((1 - wx[col]) * bl + wx[col] * br)
    return result


def _augmentation_stage(augmentation: ModuleType) -> dict:
    size = 13
    key = jax.random.PRNGKey(0)
    bounds = _crop_parameters_oracle(key, size)
    if bounds == (0, 0, size, size) or (bounds[0] == 0 and bounds[1] == 0):
        raise AssertionError(
            "fixed augmentation fixture did not select a nontrivial crop"
        )
    ramp = np.empty((size, size, 3), dtype=np.float32)
    yy, xx = np.mgrid[:size, :size]
    ramp[..., 0] = yy
    ramp[..., 1] = xx
    ramp[..., 2] = yy * 100 + xx
    crop = augmentation._bilinear_resize_crop(jnp.asarray(ramp), key)
    crop_expected = _numpy_crop_resize(ramp.astype(np.float64), bounds)
    crop_error = assert_close(
        crop,
        crop_expected,
        atol=2e-4,
        rtol=1e-6,
        label="crop-bounded bilinear ramp oracle",
    )
    # Distinct border values make reading pixels outside the chosen crop observable.
    bordered = ramp.copy()
    top, left, height, width = bounds
    outside_rows = np.arange(size)[:, None] < top
    outside_rows |= np.arange(size)[:, None] >= top + height
    outside_cols = np.arange(size)[None, :] < left
    outside_cols |= np.arange(size)[None, :] >= left + width
    outside_crop = outside_rows | outside_cols
    if not outside_crop.any():
        raise AssertionError("crop boundary fixture has no outside pixels to perturb")
    bordered[outside_crop] = np.asarray([-1000, -2000, -3000], dtype=np.float32)
    border_result = augmentation._bilinear_resize_crop(jnp.asarray(bordered), key)
    border_expected = _numpy_crop_resize(bordered.astype(np.float64), bounds)
    border_error = assert_close(
        border_result,
        border_expected,
        atol=2e-4,
        rtol=3e-6,
        label="crop boundary exclusion",
    )
    if np.max(np.abs(np.asarray(border_result) - np.asarray(crop))) > 1e-5:
        raise AssertionError(
            "crop interpolation sampled pixels outside the selected crop"
        )

    normalized = (
        np.random.default_rng(924)
        .uniform(-1, 1, size=(size, size, 3))
        .astype(np.float32)
    )
    view_key = jax.random.PRNGKey(925)
    view = np.asarray(
        augmentation.contrastive_view(jnp.asarray(normalized), view_key),
        dtype=np.float64,
    )
    crop_key, flip_key, brightness_key, contrast_key, saturation_key = jax.random.split(
        view_key, 5
    )
    expected = _numpy_crop_resize(
        normalized.astype(np.float64), _crop_parameters_oracle(crop_key, size)
    )
    if bool(jax.random.bernoulli(flip_key, 0.5)):
        expected = expected[:, ::-1]
    expected += float(jax.random.uniform(brightness_key, (), minval=-0.15, maxval=0.15))
    factor = float(jax.random.uniform(contrast_key, (), minval=0.8, maxval=1.2))
    mean = np.mean(expected)
    expected = (expected - mean) * factor + mean
    factor = float(jax.random.uniform(saturation_key, (), minval=0.8, maxval=1.2))
    luminance = np.sum(
        expected * np.asarray([0.299, 0.587, 0.114]), axis=-1, keepdims=True
    )
    expected = np.clip(luminance + factor * (expected - luminance), -1, 1)
    view_error = assert_close(
        view, expected, atol=3e-5, rtol=2e-5, label="ordered SimCLR transform oracle"
    )
    images = jnp.asarray(np.stack([normalized, normalized[::-1].copy()]))
    keys = jax.random.split(jax.random.PRNGKey(926), 4).reshape((2, 2, 2))
    paired = np.asarray(augmentation.paired_contrastive_views(images, keys))
    if (
        paired.shape != (2, 2, size, size, 3)
        or not np.isfinite(paired).all()
        or paired.min() < -1
        or paired.max() > 1
    ):
        raise AssertionError(
            "paired SimCLR views violate shape, finiteness, or clipping contract"
        )
    weak_keys = jax.random.split(jax.random.PRNGKey(927), 2)
    weak = np.asarray(augmentation.supervised_views(images, weak_keys))
    expected_weak = np.stack(
        [
            np.asarray(augmentation.horizontal_flip(images[i], weak_keys[i]))
            for i in range(2)
        ]
    )
    weak_error = assert_close(
        weak, expected_weak, atol=0, rtol=0, label="supervised flip batch"
    )
    return {
        "status": "pass",
        "bounds": list(bounds),
        "crop_ramp_error": crop_error,
        "crop_border_error": border_error,
        "ordered_view_error": view_error,
        "supervised_flip_error": weak_error,
    }


def _evaluation_stage(evaluation: ModuleType) -> dict:
    logits = np.asarray(
        [
            [2.0, 0.2, -1.0, 0.0, -0.4],
            [0.1, 0.4, 1.5, -0.2, 0.3],
            [-0.3, 1.2, 0.2, 0.5, -0.1],
            [0.0, 0.1, 0.2, 0.3, 1.4],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 2, 1, 4], dtype=np.int64)
    metrics = evaluation.classification_metrics(logits, labels, 5)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    predictions = np.argmax(probs, axis=1)
    recall = [
        float(np.mean(predictions[labels == c] == c)) if np.any(labels == c) else 0.0
        for c in range(5)
    ]
    expected_nll = float(-np.mean(np.log(probs[np.arange(4), labels])))
    expected_brier = float(np.mean(np.sum((probs - np.eye(5)[labels]) ** 2, axis=1)))
    if (
        metrics["count"] != 4
        or abs(metrics["macro_balanced_accuracy"] - np.mean(recall)) > 1e-12
    ):
        raise AssertionError(
            "classification confusion or per-class denominator differs from independent oracle"
        )
    if (
        abs(metrics["nll"] - expected_nll) > 1e-12
        or abs(metrics["brier"] - expected_brier) > 1e-12
    ):
        raise AssertionError(
            "classification calibration/loss metrics differ from independent oracle"
        )
    temperature = evaluation.fit_temperature(logits, labels)
    if temperature["grid_count"] != 601 or not 0.05 <= temperature["temperature"] <= 20:
        raise AssertionError(
            "temperature selection did not use the declared bounded grid"
        )
    shifted_metrics = evaluation.classification_metrics(
        logits + np.asarray([0, 0, 0, 0, 2.0]), labels, 5
    )
    if shifted_metrics["nll"] == metrics["nll"]:
        raise AssertionError("changed logits did not affect evaluation metrics")
    bootstrap_logits = np.random.default_rng(933).normal(size=(15, 5))
    bootstrap_labels = np.tile(np.arange(5), 3)
    bootstrap_components = [
        f"component-{index:02d}" for index in range(len(bootstrap_labels))
    ]
    bootstrap = evaluation.component_bootstrap(
        bootstrap_logits,
        bootstrap_labels,
        bootstrap_components,
        classes=5,
        replicates=40,
        seed=934,
    )
    bootstrap_repeat = evaluation.component_bootstrap(
        bootstrap_logits,
        bootstrap_labels,
        bootstrap_components,
        classes=5,
        replicates=40,
        seed=934,
    )
    if bootstrap != bootstrap_repeat or bootstrap["replicates"] != 40:
        raise AssertionError(
            "component bootstrap is not deterministic for its declared seed"
        )
    if any(
        value["defined_replicates"] + value["undefined_replicates"] != 40
        for value in bootstrap["metrics"].values()
    ):
        raise AssertionError(
            "component bootstrap dropped or duplicated undefined resamples"
        )

    pixels = (
        np.random.default_rng(931)
        .uniform(-1, 1, size=(8, 64, 64, 3))
        .astype(np.float32)
    )
    feature = evaluation.color_features(pixels)
    expected_feature = np.stack(
        [
            pixels[:, row * 16 : (row + 1) * 16, col * 16 : (col + 1) * 16].mean(
                axis=(1, 2)
            )
            for row in range(4)
            for col in range(4)
        ],
        axis=1,
    ).reshape(8, 48)
    feature_error = assert_close(
        feature,
        expected_feature,
        atol=1e-6,
        rtol=1e-6,
        label="4x4 RGB baseline features",
    )
    fit_labels = np.asarray([0, 0, 1, 1, 2, 2, 3, 4])
    centroids = evaluation.fit_color_centroids(pixels, fit_labels, 5)
    baseline_logits = evaluation.color_logits(centroids, pixels)
    if centroids["centroids"].shape != (5, 48) or baseline_logits.shape != (8, 5):
        raise AssertionError(
            "color baseline did not fit or return the declared feature/class shape"
        )
    if not np.array_equal(np.argmax(np.zeros((2, 5)), axis=1), [0, 0]):
        raise AssertionError("uniform baseline's class-order tie assumption changed")

    ids = ["q-a", "g-b", "g-c", "g-d", "g-e"]
    retrieval_features = np.asarray(
        [[1, 0], [1, 0], [0.9, 0.1], [0, 1], [-1, 0]], dtype=np.float64
    )
    retrieval_labels = np.asarray([0, 0, 0, 1, 1])
    components = ["group-a", "group-b", "group-c", "group-d", "group-e"]
    retrieval = evaluation.retrieval_metrics(
        retrieval_features, ids, retrieval_labels, components
    )
    if retrieval["query_count"] != len(ids) or retrieval[
        "evaluated_query_count"
    ] != len(ids):
        raise AssertionError("retrieval silently omitted a query")
    if retrieval["rankings"]["q-a"][:2] != ["g-b", "g-c"]:
        raise AssertionError("retrieval score order or stable ID tie order is wrong")
    for query_id in ids:
        if len(retrieval["rankings"][query_id]) != len(ids) - 1:
            raise AssertionError("retrieval ranking is incomplete")
    no_positive = evaluation.retrieval_metrics(
        retrieval_features[:3], ids[:3], np.asarray([0, 1, 2]), components[:3]
    )
    if (
        no_positive["evaluated_query_count"] != 0
        or len(no_positive["no_positive_query_ids"]) != 3
    ):
        raise AssertionError(
            "retrieval did not preserve the zero-positive query inventory"
        )
    for bad_ids, bad_vectors in (
        (["x", "x"], np.ones((2, 2))),
        (["x", "y"], np.zeros((2, 2))),
    ):
        try:
            evaluation.retrieval_metrics(
                bad_vectors, bad_ids, np.asarray([0, 0]), ["a", "b"]
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "retrieval accepted duplicate IDs or a zero feature vector"
            )
    try:
        evaluation.retrieval_metrics(
            np.asarray([[np.nan, 0], [0, 1]]),
            ["x", "y"],
            np.asarray([0, 0]),
            ["a", "b"],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("retrieval accepted nonfinite features")

    corrupt_input = (
        np.random.default_rng(932)
        .uniform(-0.8, 0.8, size=(2, 64, 64, 3))
        .astype(np.float32)
    )
    corrupt = evaluation.corruptions(corrupt_input)
    expected_keys = {
        "original",
        "brightness_plus_0.05",
        "contrast_0.90",
        "center_8x8_gray0",
        "gaussian_blur_sigma1",
        "jpeg_quality60",
    }
    if set(corrupt) != expected_keys:
        raise AssertionError("corruption inventory differs from its declared cases")
    for name, array in corrupt.items():
        if (
            array.shape != corrupt_input.shape
            or not np.isfinite(array).all()
            or array.min() < -1.00001
            or array.max() > 1.00001
        ):
            raise AssertionError(
                f"corruption {name} has invalid shape, range, or values"
            )
    if any(
        np.array_equal(corrupt[name], corrupt_input)
        for name in expected_keys - {"original"}
    ):
        raise AssertionError("a declared corruption did not change the input")
    return {
        "status": "pass",
        "classification_count": metrics["count"],
        "temperature": temperature,
        "color_feature_error": feature_error,
        "retrieval_query_count": retrieval["query_count"],
        "ranked_ids_per_query": len(ids) - 1,
        "corruption_count": len(corrupt),
    }


def _resume_stage(training: ModuleType, candidate_path: Path) -> dict:
    fit_ids = [f"synthetic-fit-{index:02d}" for index in range(10)]
    seed = 391
    id_digest = hashlib.sha256(
        json.dumps(fit_ids, separators=(",", ":")).encode()
    ).hexdigest()
    identity = {
        "stage": "synthetic-resume-check",
        "fit_ids_sha256": id_digest,
        "fit_order_seed": seed,
        "batch_size": 4,
        "schedule_total_steps": 4,
        "source": "checker-owned synthetic fixture",
    }

    params = {
        "kernel": jnp.asarray([[0.2, -0.1], [0.4, 0.3]], dtype=jnp.float32),
        "bias": jnp.asarray([0.05, -0.03], dtype=jnp.float32),
    }
    optimizer = training.AdamWState(
        jnp.asarray(2, dtype=jnp.int32),
        {
            "kernel": jnp.full_like(params["kernel"], 0.02),
            "bias": jnp.full_like(params["bias"], -0.01),
        },
        {
            "kernel": jnp.full_like(params["kernel"], 0.003),
            "bias": jnp.full_like(params["bias"], 0.004),
        },
    )

    def independent_stream_indices(
        size: int, stream_seed: int, amount: int
    ) -> list[int]:
        result = []
        epoch = 0
        while len(result) < amount:
            generator = np.random.Generator(
                np.random.PCG64(np.random.SeedSequence([stream_seed, epoch]))
            )
            permutation = generator.permutation(size)
            take = min(amount - len(result), size)
            result.extend(int(value) for value in permutation[:take])
            epoch += 1
        return result

    consumed_indices = independent_stream_indices(len(fit_ids), seed, 8)
    consumed_ids = [fit_ids[index] for index in consumed_indices]
    expected_next_indices = independent_stream_indices(len(fit_ids), seed, 12)[8:12]
    expected_next_ids = [fit_ids[index] for index in expected_next_indices]
    expected_spans = [
        {"epoch": 0, "start": 8, "count": 2},
        {"epoch": 1, "start": 0, "count": 2},
    ]
    epoch, cursor = divmod(len(consumed_ids), len(fit_ids))
    sampler = training.IDStreamState(
        epoch=epoch,
        cursor=cursor,
        permutation=np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([seed, epoch]))
        )
        .permutation(len(fit_ids))
        .astype(np.int32),
        seed=seed,
        example_count=len(fit_ids),
    )
    keys = {
        name: jax.random.PRNGKey(seed)
        for name, seed in (
            ("dropout_key", 71),
            ("augmentation_key", 72),
            ("mask_key", 73),
            ("contrastive_key", 74),
        )
    }
    with tempfile.TemporaryDirectory(prefix="c03-resume-") as temp:
        root = Path(temp)
        checkpoint = root / "checkpoint"
        training.save_checkpoint(
            checkpoint,
            params=params,
            optimizer=optimizer,
            dropout_key=keys["dropout_key"],
            augmentation_key=keys["augmentation_key"],
            mask_key=keys["mask_key"],
            contrastive_key=keys["contrastive_key"],
            sampler=sampler,
            identity=identity,
            fit_ids=fit_ids,
            consumed_ids=consumed_ids,
        )

        def next_update(restored):
            batch, next_sampler, spans = training.next_id_batch(restored["sampler"], 4)
            next_step = int(restored["optimizer"].step) + 1
            sample_key = jax.random.fold_in(restored["keys"]["dropout_key"], next_step)
            stochastic = jax.random.uniform(sample_key, ())
            signal = (
                jnp.asarray(np.mean(batch), dtype=jnp.float32) * 0.01
                + stochastic * 0.02
            )
            gradients = jax.tree_util.tree_map(
                lambda value: jnp.ones_like(value) * signal, restored["params"]
            )
            updated, opt_state, _ = training.adamw_update(
                restored["params"],
                gradients,
                restored["optimizer"],
                learning_rate=0.001,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
                weight_decay=1e-4,
                clip_norm=1.0,
            )
            return batch, next_sampler, spans, sample_key, updated, opt_state

        in_process = training.restore_checkpoint(
            checkpoint, params_template=params, identity=identity, fit_ids=fit_ids
        )
        expected = next_update(in_process)
        if (
            expected[0].tolist() != expected_next_indices
            or expected[2] != expected_spans
        ):
            raise AssertionError(
                "candidate next batch differs from independent PCG64 epoch-boundary oracle"
            )
        if (expected[1].epoch, expected[1].cursor) != (1, 2):
            raise AssertionError(
                "candidate did not leave the expected two-item epoch-one cursor"
            )
        first_epoch_ids = [
            fit_ids[index]
            for index in independent_stream_indices(len(fit_ids), seed, len(fit_ids))
        ]
        if len(set(first_epoch_ids)) != len(fit_ids):
            raise AssertionError(
                "checker fixture failed no-drop, no-duplicate first-epoch coverage"
            )
        helper = root / "fresh_process.py"
        helper.write_text(
            "import json, pathlib, sys\n"
            "import jax, jax.numpy as jnp, numpy as np\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "import training\n"
            "prefix=pathlib.Path(sys.argv[2]); cfg=json.loads(pathlib.Path(sys.argv[3]).read_text())\n"
            "params={'kernel':jnp.asarray([[0.2,-0.1],[0.4,0.3]],dtype=jnp.float32),'bias':jnp.asarray([0.05,-0.03],dtype=jnp.float32)}\n"
            "state=training.restore_checkpoint(prefix,params_template=params,identity=cfg['identity'],fit_ids=cfg['fit_ids'])\n"
            "batch,sampler,spans=training.next_id_batch(state['sampler'],4); step=int(state['optimizer'].step)+1\n"
            "key=jax.random.fold_in(state['keys']['dropout_key'],step); signal=jnp.mean(jnp.asarray(batch,dtype=jnp.float32))*0.01+jax.random.uniform(key,())*0.02\n"
            "grad=jax.tree_util.tree_map(lambda x:jnp.ones_like(x)*signal,state['params'])\n"
            "updated,opt,_=training.adamw_update(state['params'],grad,state['optimizer'],learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8,weight_decay=1e-4,clip_norm=1.0)\n"
            "np.savez(pathlib.Path(sys.argv[4]),kernel=np.asarray(updated['kernel']),bias=np.asarray(updated['bias']),first_kernel=np.asarray(opt.first_moment['kernel']),first_bias=np.asarray(opt.first_moment['bias']),second_kernel=np.asarray(opt.second_moment['kernel']),second_bias=np.asarray(opt.second_moment['bias']),permutation=sampler.permutation,step=np.asarray(opt.step))\n"
            "print(json.dumps({'batch':batch.tolist(),'spans':spans,'epoch':sampler.epoch,'cursor':sampler.cursor,'sample_key':np.asarray(key).tolist(),'other_keys':{n:np.asarray(v).tolist() for n,v in state['keys'].items()},'consumed_ids':state['metadata']['consumed_ids'],'consumed_ids_sha256':state['metadata']['consumed_ids_sha256']}))\n"
        )
        config_path = root / "restore.json"
        config_path.write_text(json.dumps({"identity": identity, "fit_ids": fit_ids}))
        output_npz = root / "fresh-output.npz"
        env = dict(os.environ)
        env["JAX_PLATFORM_NAME"] = "cpu"
        completed = subprocess.run(
            [
                sys.executable,
                str(helper),
                str(candidate_path.parent),
                str(checkpoint),
                str(config_path),
                str(output_npz),
            ],
            cwd=str(root),
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        child_record = json.loads(completed.stdout.strip().splitlines()[-1])
        child_output = np.load(output_npz, allow_pickle=False)
        if (
            child_record["batch"] != expected_next_indices
            or child_record["spans"] != expected_spans
        ):
            raise AssertionError(
                "fresh process selected a different next sample or epoch span"
            )
        if (child_record["epoch"], child_record["cursor"]) != (1, 2):
            raise AssertionError("fresh process restored a different sampler position")
        if child_record["sample_key"] != np.asarray(expected[3]).tolist():
            raise AssertionError(
                "fresh process produced a different keyed stochastic step"
            )
        for name, key in in_process["keys"].items():
            if child_record["other_keys"][name] != np.asarray(key).tolist():
                raise AssertionError("fresh process changed a stored RNG stream")
        if (
            child_record["consumed_ids"] != consumed_ids
            or child_record["consumed_ids_sha256"]
            != in_process["metadata"]["consumed_ids_sha256"]
        ):
            raise AssertionError(
                "fresh process did not retain the complete consumed-ID history"
            )
        for name, expected_value in (
            ("kernel", expected[4]["kernel"]),
            ("bias", expected[4]["bias"]),
            ("first_kernel", expected[5].first_moment["kernel"]),
            ("first_bias", expected[5].first_moment["bias"]),
            ("second_kernel", expected[5].second_moment["kernel"]),
            ("second_bias", expected[5].second_moment["bias"]),
            ("permutation", expected[1].permutation),
            ("step", expected[5].step),
        ):
            np.testing.assert_array_equal(
                child_output[name], np.asarray(expected_value)
            )

        def rewrite_archive(prefix: Path, mutation) -> None:
            with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as current:
                arrays = {key: current[key] for key in current.files}
            mutation(arrays)
            np.savez_compressed(prefix.with_suffix(".npz"), **arrays)
            metadata_path = prefix.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text())
            metadata["npz_sha256"] = hashlib.sha256(
                prefix.with_suffix(".npz").read_bytes()
            ).hexdigest()
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n"
            )

        mutations = {
            "same_shape_nonfinite_first_moment": lambda arrays: arrays.__setitem__(
                "first_00000", arrays["first_00000"] * np.nan
            ),
            "same_shape_nonfinite_second_moment": lambda arrays: arrays.__setitem__(
                "second_00000", arrays["second_00000"] * np.nan
            ),
            "negative_second_moment": lambda arrays: arrays.__setitem__(
                "second_00000", -np.abs(arrays["second_00000"]) - 0.01
            ),
            "wrong_moment_dtype": lambda arrays: arrays.__setitem__(
                "first_00000", arrays["first_00000"].astype(np.int32)
            ),
            "duplicate_sampler_permutation": lambda arrays: arrays.__setitem__(
                "sampler_permutation", np.zeros_like(arrays["sampler_permutation"])
            ),
            "wrong_valid_permutation": lambda arrays: arrays.__setitem__(
                "sampler_permutation",
                np.random.Generator(
                    np.random.PCG64(np.random.SeedSequence([seed + 1, 0]))
                )
                .permutation(len(fit_ids))
                .astype(np.int32),
            ),
            "missing_state_leaf": lambda arrays: arrays.pop("first_00000"),
            "extra_state_leaf": lambda arrays: arrays.__setitem__(
                "untracked_state", np.asarray([1], dtype=np.int32)
            ),
        }
        mutation_rejections = {}
        for name, mutation in mutations.items():
            bad_prefix = root / name
            shutil.copy2(checkpoint.with_suffix(".npz"), bad_prefix.with_suffix(".npz"))
            shutil.copy2(
                checkpoint.with_suffix(".json"), bad_prefix.with_suffix(".json")
            )
            rewrite_archive(bad_prefix, mutation)
            try:
                training.restore_checkpoint(
                    bad_prefix,
                    params_template=params,
                    identity=identity,
                    fit_ids=fit_ids,
                )
            except ValueError:
                mutation_rejections[name] = "rejected after recomputed NPZ digest"
            else:
                raise AssertionError(f"checkpoint accepted semantic mutation {name}")
        bad_leaf_name = root / "renamed-tree-leaf"
        shutil.copy2(checkpoint.with_suffix(".npz"), bad_leaf_name.with_suffix(".npz"))
        shutil.copy2(
            checkpoint.with_suffix(".json"), bad_leaf_name.with_suffix(".json")
        )

        def rename_leaf(arrays):
            arrays["renamed_params_leaf"] = arrays.pop("params_00000")

        rewrite_archive(bad_leaf_name, rename_leaf)
        bad_meta = json.loads(bad_leaf_name.with_suffix(".json").read_text())
        bad_meta["trees"]["params"][0]["key"] = "renamed_params_leaf"
        bad_leaf_name.with_suffix(".json").write_text(
            json.dumps(bad_meta, indent=2, sort_keys=True) + "\n"
        )
        try:
            training.restore_checkpoint(
                bad_leaf_name,
                params_template=params,
                identity=identity,
                fit_ids=fit_ids,
            )
        except ValueError:
            mutation_rejections["renamed_tree_leaf"] = (
                "rejected against canonical tree names"
            )
        else:
            raise AssertionError("checkpoint accepted a renamed parameter leaf")
        bad_seed = root / "wrong-seed"
        shutil.copy2(checkpoint.with_suffix(".npz"), bad_seed.with_suffix(".npz"))
        shutil.copy2(checkpoint.with_suffix(".json"), bad_seed.with_suffix(".json"))
        bad_meta = json.loads(bad_seed.with_suffix(".json").read_text())
        bad_meta["sampler"]["seed"] = seed + 1
        bad_seed.with_suffix(".json").write_text(
            json.dumps(bad_meta, indent=2, sort_keys=True) + "\n"
        )
        try:
            training.restore_checkpoint(
                bad_seed, params_template=params, identity=identity, fit_ids=fit_ids
            )
        except ValueError:
            mutation_rejections["wrong_sampler_seed"] = (
                "rejected against source-bound seed"
            )
        else:
            raise AssertionError("checkpoint accepted a changed sampler seed")

        bad_dtype_metadata = root / "forged-dtype-metadata"
        shutil.copy2(
            checkpoint.with_suffix(".npz"), bad_dtype_metadata.with_suffix(".npz")
        )
        shutil.copy2(
            checkpoint.with_suffix(".json"), bad_dtype_metadata.with_suffix(".json")
        )
        bad_meta = json.loads(bad_dtype_metadata.with_suffix(".json").read_text())
        bad_meta["trees"]["first"][0]["dtype"] = "<i4"
        bad_dtype_metadata.with_suffix(".json").write_text(
            json.dumps(bad_meta, indent=2, sort_keys=True) + "\n"
        )
        try:
            training.restore_checkpoint(
                bad_dtype_metadata,
                params_template=params,
                identity=identity,
                fit_ids=fit_ids,
            )
        except ValueError:
            mutation_rejections["forged_declared_dtype"] = (
                "rejected against NPZ leaf dtype"
            )
        else:
            raise AssertionError("checkpoint accepted forged leaf dtype metadata")

        bad_cursor = root / "negative-cursor"
        shutil.copy2(checkpoint.with_suffix(".npz"), bad_cursor.with_suffix(".npz"))
        shutil.copy2(checkpoint.with_suffix(".json"), bad_cursor.with_suffix(".json"))
        bad_meta = json.loads(bad_cursor.with_suffix(".json").read_text())
        bad_meta["sampler"]["cursor"] = -1
        bad_cursor.with_suffix(".json").write_text(
            json.dumps(bad_meta, indent=2, sort_keys=True) + "\n"
        )
        try:
            training.restore_checkpoint(
                bad_cursor, params_template=params, identity=identity, fit_ids=fit_ids
            )
        except ValueError:
            mutation_rejections["negative_sampler_cursor"] = "rejected"
        else:
            raise AssertionError("checkpoint accepted negative sampler cursor")
        inconsistent_cursor = root / "step-cursor-mismatch"
        shutil.copy2(
            checkpoint.with_suffix(".npz"), inconsistent_cursor.with_suffix(".npz")
        )
        shutil.copy2(
            checkpoint.with_suffix(".json"), inconsistent_cursor.with_suffix(".json")
        )
        bad_meta = json.loads(inconsistent_cursor.with_suffix(".json").read_text())
        bad_meta["sampler"]["cursor"] = 5
        inconsistent_cursor.with_suffix(".json").write_text(
            json.dumps(bad_meta, indent=2, sort_keys=True) + "\n"
        )
        try:
            training.restore_checkpoint(
                inconsistent_cursor,
                params_template=params,
                identity=identity,
                fit_ids=fit_ids,
            )
        except ValueError:
            mutation_rejections["step_cursor_mismatch"] = (
                "rejected against optimizer step and consumed count"
            )
        else:
            raise AssertionError(
                "checkpoint accepted cursor inconsistent with optimizer step"
            )

        try:
            training.restore_checkpoint(
                checkpoint,
                params_template=params,
                identity={**identity, "schedule_total_steps": 5},
                fit_ids=fit_ids,
            )
        except ValueError:
            mutation_rejections["changed_schedule_identity"] = "rejected"
        else:
            raise AssertionError("checkpoint accepted a changed schedule identity")
        original_hashes = {
            suffix: hashlib.sha256(
                checkpoint.with_suffix(suffix).read_bytes()
            ).hexdigest()
            for suffix in (".npz", ".json")
        }
        try:
            training.save_checkpoint(
                checkpoint,
                params=params,
                optimizer=optimizer,
                dropout_key=keys["dropout_key"],
                augmentation_key=keys["augmentation_key"],
                mask_key=keys["mask_key"],
                contrastive_key=keys["contrastive_key"],
                sampler=sampler,
                identity=identity,
                fit_ids=fit_ids,
                consumed_ids=consumed_ids,
            )
        except FileExistsError:
            mutation_rejections["checkpoint_collision"] = (
                "rejected without replacing existing files"
            )
        else:
            raise AssertionError("checkpoint writer replaced an existing version")
        if original_hashes != {
            suffix: hashlib.sha256(
                checkpoint.with_suffix(suffix).read_bytes()
            ).hexdigest()
            for suffix in (".npz", ".json")
        }:
            raise AssertionError(
                "checkpoint collision changed a retained checkpoint file"
            )
    return {
        "status": "pass",
        "fresh_process_next_sample_and_update": "exact match",
        "crossed_epoch_batch_spans": expected[2],
        "semantic_mutations": mutation_rejections,
        "note": "The only applied update is a synthetic checker fixture; no flower image training was performed.",
    }


def _public_resume_cli_probe() -> dict:
    """Exercise run.py's public --resume parser and forwarding with mocked approval."""
    import contextlib
    import io as text_io
    import types

    run_path = ROOT / "run.py"
    spec = importlib.util.spec_from_file_location("image_public_resume_probe", run_path)
    if spec is None or spec.loader is None:
        raise ImportError("could not load the public image runner")
    run_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_module)
    calls = []
    source_inventory = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in (
            "model.py",
            "objectives.py",
            "augment.py",
            "training.py",
            "evaluate.py",
            "runner.py",
        )
    }
    fake_data = {
        "status": "pass",
        "source_manifest_sha256": "synthetic-source-manifest",
    }
    run_module._load_verifier = lambda: types.SimpleNamespace(
        verify_data=lambda *args, **kwargs: calls.append("verified-data") or fake_data
    )
    run_module._check_versions = lambda recipe: {"fixture": "mocked"}
    run_module._candidate_source_inventory = lambda candidate_dir: source_inventory
    run_module._verify_approval = lambda *args, **kwargs: calls.append(
        "verified-approval"
    ) or {"reviewer": "checker fixture"}
    model_path = ROOT / "solution" / "model.py"
    run_module._load_candidate_apis = lambda specification: calls.append(
        "loaded-candidate"
    ) or ({"model": types.SimpleNamespace(__file__=str(model_path))}, model_path)
    captured = {}
    pipeline_stub = types.ModuleType("pipeline")

    def execute(**kwargs):
        calls.append("executed-pipeline")
        captured.update(kwargs)
        kwargs["output_directory"].mkdir(parents=True, exist_ok=True)
        return {"status": "mocked-public-resume"}

    pipeline_stub.execute = execute
    card_stub = types.ModuleType("model_card")
    card_stub.write_model_card = lambda *args, **kwargs: None
    prior_modules = {name: sys.modules.get(name) for name in ("pipeline", "model_card")}
    sys.modules["pipeline"] = pipeline_stub
    sys.modules["model_card"] = card_stub
    prior_argv = sys.argv
    try:
        with tempfile.TemporaryDirectory(prefix="c03-public-resume-cli-") as temporary:
            root = Path(temporary)
            data = root / "synthetic-data"
            data.mkdir()
            recipe = root / "synthetic-recipe.json"
            recipe.write_text(
                json.dumps(
                    {
                        "candidate_api_required_files": list(source_inventory),
                        "preoptimization_gate": {"optimization_permitted": True},
                    }
                )
            )
            archive = root / "synthetic.tar.gz"
            archive.write_bytes(b"synthetic fixture")
            approval = root / "approval.json"
            approval.write_text("{}")
            output = root / "run-output"
            sys.argv = [
                str(run_path),
                "--data",
                str(data),
                "--archive",
                str(archive),
                "--recipe",
                str(recipe),
                "--approval",
                str(approval),
                "--output",
                str(output),
                "--implementation",
                "solution",
                "--resume",
            ]
            with contextlib.redirect_stdout(text_io.StringIO()):
                result = run_module.main()
            if result != 0 or captured.get("resume") is not True:
                raise AssertionError("public run.py CLI did not forward --resume")
            if calls != [
                "verified-data",
                "verified-approval",
                "loaded-candidate",
                "executed-pipeline",
            ]:
                raise AssertionError(
                    f"public resume CLI changed preflight ordering: {calls}"
                )
            report = json.loads((output / "run-report.json").read_text())
            if report.get("status") != "mocked-public-resume":
                raise AssertionError(
                    "public CLI did not retain the mocked resume result"
                )
    finally:
        sys.argv = prior_argv
        for name, prior in prior_modules.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior
    return {
        "status": "pass",
        "public_resume_flag_parsed_and_forwarded": captured.get("resume") is True,
        "preflight_order": calls,
        "synthetic_mocked_approval_only": True,
    }


def _consumer_stage(candidate: ModuleType, candidate_path: Path) -> dict:
    consumer_path = ROOT / "solution" / "consumer.py"
    spec = importlib.util.spec_from_file_location(
        "image_consumer_export", consumer_path
    )
    if spec is None or spec.loader is None:
        raise ImportError("could not load local image consumer exporter")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    config = dict(candidate.PILOT_CONFIG)
    encoder = candidate.init_vit(jax.random.PRNGKey(261), config)
    head = candidate.init_linear_head(
        jax.random.PRNGKey(262), int(config["width"]), int(config["classes"])
    )
    probe = (
        np.random.default_rng(263)
        .uniform(-1.0, 1.0, size=(2, 64, 64, 3))
        .astype(np.float32)
    )
    expected_logits = np.asarray(
        candidate.classification_logits(encoder, head, jnp.asarray(probe))
    )
    if not np.isfinite(expected_logits).all():
        raise AssertionError(
            "selected candidate did not produce finite local consumer canary logits"
        )
    recipe = json.loads((ROOT / "data" / "recipe-review-v10.json").read_text())
    provenance = {
        "dataset": recipe["data"]["upstream"]["name"],
        "archive_sha256": recipe["data"]["upstream"]["sha256"],
        "license_notice": "CC BY 2.0 image-level credits are in the upstream notice; no photo pixels are packaged.",
        "classes": recipe["data"]["class_order"],
        "weight_rights_statement": "Private correctness bundle only; no standalone weight redistribution license is asserted.",
    }
    with tempfile.TemporaryDirectory(prefix="c03-consumer-") as temporary:
        root = Path(temporary)
        bundle = root / "bundle"
        report = exporter.export_local_bundle(
            bundle,
            model_module=candidate,
            model_source=candidate_path,
            processor_source=ROOT / "data" / "processor.py",
            encoder_params=encoder,
            head=head,
            provenance=provenance,
        )
        runtime_spec = importlib.util.spec_from_file_location(
            "local_consumer_runtime", bundle / "consumer_runtime.py"
        )
        if runtime_spec is None or runtime_spec.loader is None:
            raise ImportError("could not load exported local consumer runtime")
        runtime = importlib.util.module_from_spec(runtime_spec)
        sys.modules[runtime_spec.name] = runtime
        runtime_spec.loader.exec_module(runtime)
        consumer = runtime.LocalImageClassifier.from_pretrained(
            bundle, local_files_only=True
        )
        actual_logits = np.asarray(
            consumer.model.classification_logits(
                consumer.params, consumer.head, jnp.asarray(probe)
            )
        )
        np.testing.assert_allclose(actual_logits, expected_logits, rtol=0.0, atol=0.0)

        rgba = np.zeros((37, 51, 4), dtype=np.uint8)
        rgba[..., 0] = np.arange(51, dtype=np.uint8)[None, :] * 5
        rgba[..., 1] = np.arange(37, dtype=np.uint8)[:, None] * 6
        rgba[..., 2] = 91
        rgba[..., 3] = np.linspace(0, 255, 51, dtype=np.uint8)[None, :]
        image_path = root / "synthetic-alpha.png"
        Image.fromarray(rgba, mode="RGBA").save(image_path, format="PNG")
        raw_prediction = consumer.predict_bytes(image_path.read_bytes())
        if len(raw_prediction["probabilities"]) != int(
            config["classes"]
        ) or not np.isclose(
            sum(raw_prediction["probabilities"]), 1.0, rtol=0.0, atol=1e-12
        ):
            raise AssertionError(
                "raw-image local consumer returned malformed probabilities"
            )

        cold_script = root / "cold_consumer.py"
        cold_script.write_text(
            "import json, pathlib, sys\n"
            "sys.path.insert(0,sys.argv[1])\n"
            "from consumer_runtime import LocalImageClassifier\n"
            "model=LocalImageClassifier.from_pretrained(sys.argv[1],local_files_only=True)\n"
            "print(json.dumps(model.predict_bytes(pathlib.Path(sys.argv[2]).read_bytes()),sort_keys=True))\n"
        )
        env = dict(os.environ)
        env.update(
            {
                "JAX_PLATFORM_NAME": "cpu",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        completed = subprocess.run(
            [sys.executable, str(cold_script), str(bundle), str(image_path)],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        cold_prediction = json.loads(completed.stdout.strip().splitlines()[-1])
        np.testing.assert_allclose(
            cold_prediction["logits"], raw_prediction["logits"], rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            cold_prediction["probabilities"],
            raw_prediction["probabilities"],
            rtol=0.0,
            atol=0.0,
        )

        corruption_rejections = {}
        for name, filename, mutate in (
            (
                "altered_model_source",
                "model.py",
                lambda path: path.write_text(path.read_text() + "\n# altered\n"),
            ),
            (
                "altered_loader_source",
                "consumer_runtime.py",
                lambda path: path.write_text(path.read_text() + "\n# altered\n"),
            ),
            (
                "altered_pooling_config",
                "config.json",
                lambda path: path.write_text(
                    path.read_text().replace("CLS token", "mean pool")
                ),
            ),
            (
                "altered_weights",
                "weights.npz",
                lambda path: path.write_bytes(path.read_bytes()[:-8] + b"corrupt!"),
            ),
        ):
            damaged = root / name
            shutil.copytree(bundle, damaged)
            mutate(damaged / filename)
            try:
                runtime.LocalImageClassifier.from_pretrained(
                    damaged, local_files_only=True
                )
            except (ValueError, OSError):
                corruption_rejections[name] = (
                    "rejected by local bundle integrity/contract check"
                )
            else:
                raise AssertionError(f"local consumer accepted corrupted {filename}")
        return {
            "status": "pass",
            "selected_candidate_source": str(candidate_path),
            "candidate_to_export_logits": "exact match",
            "raw_png_canary": "finite probabilities from alpha-composited letterbox input",
            "offline_cold_process": "exact prediction match from unrelated working directory",
            "bundle_contains_source_images": report["contains_source_images"],
            "weight_rights_statement": report["weight_rights_statement"],
            "corruption_rejections": corruption_rejections,
        }


def _runner_stage(runner: ModuleType) -> dict:
    fit_ids = [f"runner-fixture-{index:02d}" for index in range(10)]
    images = np.random.default_rng(741).integers(
        0, 256, size=(len(fit_ids), 64, 64, 3), dtype=np.uint8
    )
    labels = (np.arange(len(fit_ids)) % 5).astype(np.int32)
    initial = {
        "weight": jnp.asarray([0.15, -0.08], dtype=jnp.float32),
        "bias": jnp.asarray(0.03, dtype=jnp.float32),
    }

    def loss_fn(params, batch, targets, dropout_key, _mask_key, _positive):
        feature = jnp.mean(batch, axis=(1, 2, 3))
        stochastic_scale = 0.8 + 0.4 * jax.random.uniform(dropout_key, ())
        prediction = stochastic_scale * (
            feature[:, None] * params["weight"][None, :] + params["bias"]
        )
        target = jnp.stack(
            (
                targets.astype(jnp.float32) / 4.0,
                1.0 - targets.astype(jnp.float32) / 4.0,
            ),
            axis=1,
        )
        return jnp.mean(jnp.square(prediction - target)), {
            "selected_mae_target_values": jnp.asarray(0, dtype=jnp.int32)
        }

    identity = {
        "source": "checker-owned runner fixture",
        "recipe_sha256": "fixture-only",
    }
    with tempfile.TemporaryDirectory(prefix="c03-runner-") as temporary:
        root = Path(temporary)
        partial = runner.train_updates(
            stage_id="fixture-stage",
            params=initial,
            loss_fn=loss_fn,
            images_uint8=images,
            labels=labels,
            fit_ids=fit_ids,
            phash_by_id={},
            mode="none",
            batch_size=4,
            updates=3,
            learning_rate=0.001,
            checkpoint_directory=root / "partial",
            identity=identity,
            stop_after=2,
        )
        if partial["complete"] or int(partial["optimizer"].step) != 2:
            raise AssertionError(
                "runner partial execution did not stop at its declared checkpoint boundary"
            )
        resumed = runner.train_updates(
            stage_id="fixture-stage",
            params=initial,
            loss_fn=loss_fn,
            images_uint8=images,
            labels=labels,
            fit_ids=fit_ids,
            phash_by_id={},
            mode="none",
            batch_size=4,
            updates=3,
            learning_rate=0.001,
            checkpoint_directory=root / "resumed",
            identity=identity,
            resume_prefix=partial["checkpoint_prefix"],
        )
        uninterrupted = runner.train_updates(
            stage_id="fixture-stage",
            params=initial,
            loss_fn=loss_fn,
            images_uint8=images,
            labels=labels,
            fit_ids=fit_ids,
            phash_by_id={},
            mode="none",
            batch_size=4,
            updates=3,
            learning_rate=0.001,
            checkpoint_directory=root / "uninterrupted",
            identity=identity,
        )
        resumed_state = (
            resumed["optimizer"].step,
            resumed["optimizer"].first_moment,
            resumed["optimizer"].second_moment,
        )
        full_state = (
            uninterrupted["optimizer"].step,
            uninterrupted["optimizer"].first_moment,
            uninterrupted["optimizer"].second_moment,
        )
        for resumed_leaf, full_leaf in zip(
            jax.tree_util.tree_leaves((resumed["params"], resumed_state)),
            jax.tree_util.tree_leaves((uninterrupted["params"], full_state)),
        ):
            np.testing.assert_array_equal(
                np.asarray(resumed_leaf), np.asarray(full_leaf)
            )
        if (
            resumed["consumed_ids"] != uninterrupted["consumed_ids"]
            or resumed["trace"] != uninterrupted["trace"]
        ):
            raise AssertionError(
                "runner resume differs from the uninterrupted sample/update history"
            )
        spans = resumed["trace"][-1]["epoch_spans"]
        if spans != [
            {"epoch": 0, "start": 8, "count": 2},
            {"epoch": 1, "start": 0, "count": 2},
        ]:
            raise AssertionError(
                "runner stage did not carry its partial epoch tail into the next batch"
            )

        def control_loss(params, batch, _targets, dropout_key, _mask_key, _positive):
            stochastic = jax.random.uniform(dropout_key, ())
            value = jnp.mean(batch) * params["weight"][0] + params["bias"] * stochastic
            return value, {
                "selected_mae_target_values": jnp.asarray(0, dtype=jnp.int32)
            }

        contrastive = runner.train_updates(
            stage_id="probe-mae-name-is-not-an-objective",
            params=initial,
            loss_fn=control_loss,
            images_uint8=images,
            labels=labels,
            fit_ids=fit_ids,
            phash_by_id={item: "0000000000000000" for item in fit_ids},
            mode="contrastive",
            batch_size=4,
            updates=1,
            learning_rate=0.001,
            checkpoint_directory=root / "contrastive-counter-control",
            identity=identity,
        )
        counter_control = contrastive["trace"][0]
        if (
            counter_control["model_view_count"] != 8
            or counter_control["input_pixel_values"] != 8 * 64 * 64 * 3
        ):
            raise AssertionError(
                "contrastive exposure must count both actual model views"
            )
        if counter_control["selected_mae_target_values"] != 0:
            raise AssertionError(
                "stage names must not manufacture masked-reconstruction exposure"
            )
        public_resume = _public_resume_cli_probe()
        return {
            "status": "pass",
            "synthetic_only": True,
            "updates": int(resumed["optimizer"].step),
            "fresh_resume_matches_uninterrupted": True,
            "epoch_boundary_spans": spans,
            "per_update_checkpoint_count": 3,
            "objective_driven_exposure_control": {
                "stage_name": "probe-mae-name-is-not-an-objective",
                "model_view_count": counter_control["model_view_count"],
                "input_pixel_values": counter_control["input_pixel_values"],
                "selected_mae_target_values": counter_control[
                    "selected_mae_target_values"
                ],
            },
            "public_run_resume_cli": public_resume,
            "note": "This checker fixture performs synthetic updates only; it does not load or optimize flower images.",
        }


def _report_stage(model_card_module) -> dict:
    recipe = json.loads((ROOT / "data" / "recipe-review-v10.json").read_text())
    stages = {stage["id"]: stage for stage in recipe["pilot"]["stages"]}
    report = {
        "schema": "jaxpathways.image-pilot-run.v1",
        "recipe_sha256": "fixture-recipe",
        "source_manifest_sha256": "fixture-manifest",
        "stage_order": list(stages),
        "stage_checkpoints": {
            name: {"checkpoint_npz_sha256": "fixture"} for name in stages
        },
        "stage_execution": {
            name: {
                "objective": stage["objective"],
                "parent": stage["parent"],
                "updates": stage["updates"],
                "batch_size": stage["batch_size"],
                "learning_rate": stage["learning_rate"],
                "measured_input_images": stage["updates"] * stage["batch_size"],
                "measured_model_views": stage["updates"] * stage["batch_size"],
                "measured_pixel_values": stage["updates"]
                * stage["batch_size"]
                * 64
                * 64
                * 3,
                "measured_selected_mae_target_values": 0,
            }
            for name, stage in stages.items()
        },
        "deploy_selection": {"dev_only_selected": "baseline-vit"},
        "final_class_counts": {name: 1 for name in recipe["data"]["class_order"]},
        "final_metrics": {
            "baseline-vit": {
                "raw": {"macro_balanced_accuracy": 0.6, "nll": 0.8},
                "calibrated": {"nll": 0.7},
            }
        },
        "quality_gate": {
            "selected_candidate_only": {"floor_a": False, "floor_b": True}
        },
    }
    failed_card = model_card_module.render_model_card(report, recipe)
    if (
        "quality floors **failed**" not in failed_card
        or "`floor_a`: failed" not in failed_card
    ):
        raise AssertionError(
            "measured model card did not describe a failed quality gate"
        )
    report["quality_gate"]["selected_candidate_only"] = {
        "floor_a": True,
        "floor_b": True,
    }
    passed_card = model_card_module.render_model_card(report, recipe)
    if (
        "quality floors **passed**" not in passed_card
        or "`floor_a`: failed" in passed_card
    ):
        raise AssertionError(
            "measured model card hard-coded a failed reference outcome"
        )
    return {"status": "pass", "dynamic_failed_and_passed_gate_wording": True}


def _objective_stage(candidate: ModuleType, objective: ModuleType) -> dict:
    x = (
        np.random.default_rng(801)
        .uniform(-0.9, 0.9, size=(2, 64, 64, 3))
        .astype(np.float32)
    )
    labels = np.asarray([0, 4], dtype=np.int32)
    logits = np.random.default_rng(802).normal(size=(2, 5)).astype(np.float32)
    ce = objective.cross_entropy(jnp.asarray(logits), jnp.asarray(labels))
    ce_error = abs(float(ce) - oracles.cross_entropy(logits, labels))
    if ce_error > 1e-6:
        raise AssertionError(
            f"stable cross-entropy differs from independent oracle: {ce_error:.6g}"
        )
    visible, hidden, mask = objective.random_patch_mask(
        jax.random.PRNGKey(803), 3, 64, 0.5
    )
    visible_np, hidden_np, mask_np = map(np.asarray, (visible, hidden, mask))
    if (
        visible_np.shape != (3, 32)
        or hidden_np.shape != (3, 32)
        or np.any(mask_np.sum(axis=1) != 32)
    ):
        raise AssertionError(
            "MAE mask does not contain the declared visible/hidden populations"
        )
    for row in range(3):
        if len(np.unique(np.r_[visible_np[row], hidden_np[row]])) != 64:
            raise AssertionError("MAE visible and hidden patches are not a partition")

    encoder = candidate.init_vit(jax.random.PRNGKey(804))
    decoder = objective.init_mae_decoder(
        jax.random.PRNGKey(805), patches=64, patch_values=192
    )
    images = jnp.asarray(x)
    mask_key, dropout_key = jax.random.PRNGKey(806), jax.random.PRNGKey(807)
    output = objective.mae_forward(
        encoder,
        decoder,
        images,
        mask_key,
        dropout_key=dropout_key,
        mask_ratio=0.5,
        train=True,
    )
    if (
        output["prediction"].shape != (2, 64, 192)
        or not np.isfinite(np.asarray(output["prediction"])).all()
    ):
        raise AssertionError("MAE decoder returned invalid patch predictions")
    pred = np.asarray(output["prediction"], dtype=np.float64)
    target = np.asarray(output["targets"], dtype=np.float64)
    hidden_mask = np.asarray(output["mask"], dtype=np.float64)[..., None]
    expected_mae = float(
        np.sum((pred - target) ** 2 * hidden_mask)
        / (np.sum(hidden_mask) * target.shape[-1])
    )
    mae_error = abs(float(output["loss"]) - expected_mae)
    if mae_error > 2e-6:
        raise AssertionError(
            f"MAE hidden-only denominator differs from oracle: {mae_error:.6g}"
        )
    hidden_indices = np.asarray(output["hidden_indices"])
    patches = candidate.patchify(images)
    changed_patches = patches.at[0, int(hidden_indices[0, 0]), :].add(0.75)
    changed_images = candidate.unpatchify(changed_patches, 64, 8, 3)
    changed_output = objective.mae_forward(
        encoder,
        decoder,
        changed_images,
        mask_key,
        dropout_key=dropout_key,
        mask_ratio=0.5,
        train=True,
    )
    visible_delta = float(
        np.max(
            np.abs(
                np.asarray(output["visible_encoder_tokens"])
                - np.asarray(changed_output["visible_encoder_tokens"])
            )
        )
    )
    prediction_delta = float(
        np.max(
            np.abs(
                np.asarray(output["prediction"])
                - np.asarray(changed_output["prediction"])
            )
        )
    )
    target_loss_delta = abs(float(output["loss"]) - float(changed_output["loss"]))
    if visible_delta != 0.0 or prediction_delta != 0.0 or target_loss_delta <= 1e-7:
        raise AssertionError(
            "hidden-pixel perturbation leaked into encoder/prediction or failed to change target loss"
        )
    mask_small = objective.random_patch_mask(jax.random.PRNGKey(808), 2, 64, 0.25)
    if (
        np.asarray(mask_small[0]).shape[1] != 48
        or np.asarray(mask_small[1]).shape[1] != 16
    ):
        raise AssertionError(
            "changed MAE mask ratio did not change visible/hidden counts"
        )

    feature_values = np.random.default_rng(809).normal(size=(8, 7)).astype(np.float32)
    feature_values /= np.maximum(
        np.linalg.norm(feature_values, axis=1, keepdims=True), 1e-12
    )
    source_positive = np.eye(4, dtype=bool)
    source_positive[0, 2] = source_positive[2, 0] = True
    temperature = 0.23
    candidate_loss, terms = objective.multi_positive_nt_xent(
        jnp.asarray(feature_values), jnp.asarray(source_positive), temperature
    )
    expected_loss, expected_positive, expected_similarity = _numpy_multi_positive(
        feature_values, source_positive, temperature
    )
    nce_error = abs(float(candidate_loss) - expected_loss)
    positive_error = assert_close(
        terms["positive_mask"],
        expected_positive,
        atol=0.0,
        rtol=0.0,
        label="known-neighbor positive set",
    )
    similarity_error = assert_close(
        terms["similarities"],
        expected_similarity,
        atol=2e-6,
        rtol=2e-6,
        label="contrastive similarity denominator",
    )
    if nce_error > 2e-6:
        raise AssertionError(
            f"multi-positive InfoNCE differs from independent oracle: {nce_error:.6g}"
        )

    def loss_for_values(values):
        return objective.multi_positive_nt_xent(
            values, jnp.asarray(source_positive), temperature
        )[0]

    grad = np.asarray(jax.grad(loss_for_values)(jnp.asarray(feature_values)))
    fd = oracles.finite_difference(
        lambda values: _numpy_multi_positive(values, source_positive, temperature)[0],
        feature_values.astype(np.float64),
        (0, 1),
        step=1e-3,
    )
    nce_grad_error = abs(float(grad[0, 1]) - fd)
    if nce_grad_error > 2e-3:
        raise AssertionError(
            f"contrastive gradient finite difference failed: {nce_grad_error:.6g}"
        )

    # Exercise gradients through the actual model and objective composition.
    # Static config is deliberately kept outside differentiated array trees.
    vit_params = candidate.init_vit(jax.random.PRNGKey(810))
    vit_static = {"config": dict(vit_params["config"])}
    vit_trainable = {
        name: value for name, value in vit_params.items() if name != "config"
    }
    head = candidate.init_linear_head(jax.random.PRNGKey(811), 32, 5)
    decoder = objective.init_mae_decoder(
        jax.random.PRNGKey(812), patches=64, patch_values=192
    )
    decoder_static = {
        name: decoder[name] for name in ("heads", "epsilon", "dropout_rate")
    }
    decoder_trainable = {
        name: value for name, value in decoder.items() if name not in decoder_static
    }

    def assemble_vit(tree):
        return {**tree, "config": vit_static["config"]}

    def assemble_decoder(tree):
        return {**tree, **decoder_static}

    def supervised_loss(encoder_tree, head_tree):
        outputs = candidate.classification_logits(
            assemble_vit(encoder_tree),
            head_tree,
            images,
            dropout_key=jax.random.PRNGKey(813),
            train=True,
        )
        return objective.cross_entropy(outputs, jnp.asarray([0, 4], dtype=jnp.int32))

    supervised_gradients = jax.grad(supervised_loss, argnums=(0, 1))(
        vit_trainable, head
    )

    def mae_train_loss(encoder_tree, decoder_tree):
        return objective.mae_forward(
            assemble_vit(encoder_tree),
            assemble_decoder(decoder_tree),
            images,
            jax.random.PRNGKey(814),
            dropout_key=jax.random.PRNGKey(815),
            mask_ratio=0.5,
            train=True,
        )["loss"]

    mae_gradients = jax.grad(mae_train_loss, argnums=(0, 1))(
        vit_trainable, decoder_trainable
    )

    def tree_norm(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        if not leaves:
            return 0.0
        arrays = [np.asarray(leaf, dtype=np.float64) for leaf in leaves]
        if not all(np.isfinite(value).all() for value in arrays):
            raise AssertionError("an objective produced nonfinite parameter gradients")
        return float(np.sqrt(sum(np.sum(value * value) for value in arrays)))

    supervised_encoder_grad_norm = tree_norm(supervised_gradients[0])
    supervised_head_grad_norm = tree_norm(supervised_gradients[1])
    mae_encoder_grad_norm = tree_norm(mae_gradients[0])
    mae_decoder_grad_norm = tree_norm(mae_gradients[1])
    if (
        min(
            supervised_encoder_grad_norm,
            supervised_head_grad_norm,
            mae_encoder_grad_norm,
            mae_decoder_grad_norm,
        )
        <= 1e-10
    ):
        raise AssertionError(
            "supervised or MAE objective does not reach all intended trainable leaves"
        )

    frozen_features = jax.lax.stop_gradient(
        candidate.encode_images(assemble_vit(vit_trainable), images)
    )

    def probe_loss(head_tree):
        return objective.cross_entropy(
            frozen_features @ head_tree["kernel"] + head_tree["bias"],
            jnp.asarray([0, 4]),
        )

    probe_grad = jax.grad(probe_loss)(head)
    probe_grad_norm = tree_norm(probe_grad)
    probe_leaf_count = len(jax.tree_util.tree_leaves(probe_grad))
    if probe_grad_norm <= 1e-10 or probe_leaf_count != 2:
        raise AssertionError(
            "frozen probe must differentiate its fresh linear head only"
        )

    paired = jnp.asarray(
        np.random.default_rng(816)
        .uniform(-0.9, 0.9, (4, 2, 64, 64, 3))
        .astype(np.float32)
    )
    projector = objective.init_projector(
        jax.random.PRNGKey(817), input_dim=32, hidden_dim=64, output_dim=32
    )
    source_relation = jnp.eye(4, dtype=bool).at[0, 1].set(True).at[1, 0].set(True)

    def simclr_loss(encoder_tree, projector_tree):
        flat = paired.reshape((-1, 64, 64, 3))
        features = candidate.encode_images(
            assemble_vit(encoder_tree),
            flat,
            dropout_key=jax.random.PRNGKey(818),
            train=True,
        )
        projected = objective.project_features(projector_tree, features)
        return objective.multi_positive_nt_xent(
            projected, source_relation, temperature=0.2
        )[0]

    simclr_gradients = jax.grad(simclr_loss, argnums=(0, 1))(vit_trainable, projector)
    simclr_encoder_grad_norm = tree_norm(simclr_gradients[0])
    simclr_projector_grad_norm = tree_norm(simclr_gradients[1])
    if min(simclr_encoder_grad_norm, simclr_projector_grad_norm) <= 1e-10:
        raise AssertionError(
            "SimCLR objective does not reach encoder and projector leaves"
        )

    partial = {
        "last_block": vit_trainable["blocks"][-1],
        "final_ln_scale": vit_trainable["final_ln_scale"],
        "final_ln_bias": vit_trainable["final_ln_bias"],
    }

    def merge_partial(partial_tree):
        blocks = tuple(vit_trainable["blocks"][:-1]) + (partial_tree["last_block"],)
        return {
            **vit_trainable,
            "blocks": blocks,
            "final_ln_scale": partial_tree["final_ln_scale"],
            "final_ln_bias": partial_tree["final_ln_bias"],
            "config": vit_static["config"],
        }

    def partial_loss(partial_tree, head_tree):
        outputs = candidate.classification_logits(
            merge_partial(partial_tree),
            head_tree,
            images,
            dropout_key=jax.random.PRNGKey(819),
            train=True,
        )
        return objective.cross_entropy(outputs, jnp.asarray([0, 4], dtype=jnp.int32))

    partial_gradients = jax.grad(partial_loss, argnums=(0, 1))(partial, head)
    partial_encoder_grad_norm = tree_norm(partial_gradients[0])
    if partial_encoder_grad_norm <= 1e-10 or set(partial_gradients[0]) != {
        "last_block",
        "final_ln_scale",
        "final_ln_bias",
    }:
        raise AssertionError(
            "partial tuning differentiated leaves outside its declared final-block inventory"
        )
    return {
        "status": "pass",
        "cross_entropy_error": ce_error,
        "mae_hidden_loss_error": mae_error,
        "mae_hidden_perturbation": {
            "visible_encoder_max_delta": visible_delta,
            "prediction_max_delta": prediction_delta,
            "target_loss_delta": target_loss_delta,
        },
        "nt_xent_oracle_error": nce_error,
        "positive_mask_error": positive_error,
        "similarity_error": similarity_error,
        "nt_xent_gradient_fd_error": nce_grad_error,
        "training_gradient_norms_without_updates": {
            "supervised_encoder": supervised_encoder_grad_norm,
            "supervised_head": supervised_head_grad_norm,
            "mae_encoder": mae_encoder_grad_norm,
            "mae_decoder": mae_decoder_grad_norm,
            "probe_head_only": probe_grad_norm,
            "simclr_encoder": simclr_encoder_grad_norm,
            "simclr_projector": simclr_projector_grad_norm,
            "partial_encoder_inventory": partial_encoder_grad_norm,
        },
    }


def _composition_stage() -> dict:
    """Check paired-view ordering against a checker-owned NumPy loss oracle."""
    train_steps_path = ROOT / "solution" / "train_steps.py"
    spec = importlib.util.spec_from_file_location(
        "image_train_steps_composition", train_steps_path
    )
    if spec is None or spec.loader is None:
        raise ImportError("could not load the actual SimCLR composition")
    train_steps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_steps)
    objective = load_objectives(ROOT / "solution" / "model.py", "solution")

    class SentinelEncoder:
        @staticmethod
        def encode_images(_params, images, **_kwargs):
            return jnp.asarray(images[:, 0, 0, :], dtype=jnp.float32)

    # Sources 0 and 2 are a known same-source/hash neighbor; source 1 is a
    # negative. Each source's two views are identical basis vectors, making a
    # wrong interleaved flattening measurably disagree with the view-major mask.
    pair = np.asarray(
        [
            [[[1, 0]], [[1, 0]]],
            [[[0, 1]], [[0, 1]]],
            [[[1, 0]], [[1, 0]]],
        ],
        dtype=np.float32,
    )[:, :, None, :, :]
    pair = pair.reshape((3, 2, 1, 1, 2))
    relation = np.asarray(
        [[True, False, True], [False, True, False], [True, False, True]], dtype=bool
    )
    projector = {
        "first_kernel": jnp.eye(2, dtype=jnp.float32),
        "first_bias": jnp.zeros((2,), dtype=jnp.float32),
        "second_kernel": jnp.eye(2, dtype=jnp.float32),
        "second_bias": jnp.zeros((2,), dtype=jnp.float32),
    }
    loss, _metrics = train_steps.simclr_pretraining_loss(
        {},
        projector,
        {"config": {}},
        jnp.asarray(pair),
        jnp.asarray(relation),
        jax.random.PRNGKey(87),
        model_api=SentinelEncoder,
        objective_api=objective,
    )

    ordered = pair.transpose(1, 0, 2, 3, 4).reshape((6, 2)).astype(np.float64)
    ordered /= np.maximum(np.linalg.norm(ordered, axis=1, keepdims=True), 1e-12)
    positive = np.block([[relation, relation], [relation, relation]])
    positive |= np.block(
        [
            [np.zeros_like(relation), np.eye(3, dtype=bool)],
            [np.eye(3, dtype=bool), np.zeros_like(relation)],
        ]
    )
    np.fill_diagonal(positive, False)
    similarities = ordered @ ordered.T / 0.2
    losses = []
    for index in range(6):
        denominator = similarities[index, np.arange(6) != index]
        numerator = similarities[index, positive[index]]
        losses.append(np.logaddexp.reduce(denominator) - np.logaddexp.reduce(numerator))
    expected = float(np.mean(losses))
    interleaved = pair.reshape((6, 2)).astype(np.float64)
    interleaved /= np.maximum(np.linalg.norm(interleaved, axis=1, keepdims=True), 1e-12)
    wrong_similarity = interleaved @ interleaved.T / 0.2
    wrong_losses = [
        np.logaddexp.reduce(wrong_similarity[i, np.arange(6) != i])
        - np.logaddexp.reduce(wrong_similarity[i, positive[i]])
        for i in range(6)
    ]
    wrong = float(np.mean(wrong_losses))
    if not np.isclose(float(loss), expected, atol=2e-6, rtol=2e-6):
        raise AssertionError(
            "SimCLR composition differs from the independent view-major positive oracle"
        )
    if abs(expected - wrong) < 0.5:
        raise AssertionError(
            "checker fixture does not distinguish interleaved from view-major ordering"
        )
    return {
        "status": "pass",
        "independent_view_major_loss": expected,
        "candidate_composed_loss": float(loss),
        "incorrect_interleaved_loss": wrong,
        "known_neighbor_positive_pairs": 2,
        "positive_relation_includes_known_neighbor": bool(
            positive[0, 2] and positive[2, 0]
        ),
    }


def _independent_accuracy_nll(
    logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0
) -> dict[str, float]:
    scores = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if (
        scores.ndim != 2
        or scores.shape[0] != targets.size
        or not np.isfinite(scores).all()
    ):
        raise AssertionError("retained comparison logits are malformed")
    prediction = np.argmax(scores, axis=1)
    recall = []
    for class_index in range(scores.shape[1]):
        selected = targets == class_index
        if not np.any(selected):
            raise AssertionError("independent metric inventory is missing a class")
        recall.append(float(np.mean(prediction[selected] == targets[selected])))
    if temperature <= 0:
        raise AssertionError("independent metric temperature must be positive")
    scaled = scores / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    nll = float(
        -np.mean(
            np.log(np.maximum(probabilities[np.arange(targets.size), targets], 1e-300))
        )
    )
    return {"macro_balanced_accuracy": float(np.mean(recall)), "nll": nll}


def _make_synthetic_graph_inputs(
    root: Path, recipe_path: Path
) -> tuple[Path, Path, Path]:
    data_root = root / "synthetic-data"
    (data_root / "arrays").mkdir(parents=True)
    (data_root / "splits").mkdir()
    recipe = json.loads(recipe_path.read_text())
    for stage in recipe["pilot"]["stages"]:
        stage["batch_size"] = 4
        stage["updates"] = (
            3
            if stage["id"]
            in {
                "baseline-vit",
                "mae-pretrain",
                "simclr-pretrain",
                "probe-random",
                "partial-selected",
            }
            else 2
        )
    recipe["evaluation"]["component_bootstrap"]["replicates"] = 20
    recipe["data"]["upstream"]["name"] = "checker-owned synthetic RGB graph fixture"
    recipe["data"]["upstream"]["sha256"] = hashlib.sha256(
        b"synthetic graph; not flower data"
    ).hexdigest()
    recipe["checker_fixture"] = {
        "synthetic_only": True,
        "model_config": {
            "image_size": 64,
            "channels": 3,
            "patch_size": 32,
            "width": 8,
            "heads": 2,
            "mlp_dim": 16,
            "layers": 1,
            "classes": 5,
            "dropout_rate": 0.1,
            "layer_norm_epsilon": 1e-5,
        },
    }
    recipe_path = root / "synthetic-recipe.json"
    recipe_path.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n")

    names = recipe["data"]["class_order"]
    rows_by_split = {}
    source_records = []
    component_members = {}
    rng = np.random.default_rng(662901)
    yy, xx = np.mgrid[:64, :64]
    for split in ("fit", "recipe_dev", "final"):
        rows = []
        pixels = []
        labels = []
        for index in range(10):
            class_index = index % len(names)
            class_name = names[class_index]
            image_id = f"{class_name}/graph-{split}-{index:02d}.jpg"
            base = np.zeros((64, 64, 3), dtype=np.float32)
            base[..., class_index % 3] = 48 + class_index * 25
            base[..., (class_index + 1) % 3] = (xx * (class_index + 1) + yy * 2) % 128
            base += rng.normal(0, 7, base.shape)
            image = np.clip(base, 0, 255).round().astype(np.uint8)
            pixels.append(image)
            labels.append(class_index)
            phash = hashlib.sha256(image_id.encode()).hexdigest()[:16]
            raw_sha256 = hashlib.sha256((image_id + ":raw").encode()).hexdigest()
            pixel_sha256 = hashlib.sha256(image.tobytes(order="C")).hexdigest()
            creator = f"fixture creator {index}"
            creator_id = f"creator-{split}-{index}"
            source_url = f"https://example.invalid/{split}/{index}"
            rows.append(
                {
                    "id": image_id,
                    "label": class_name,
                    "class_index": class_index,
                    "creator": creator,
                    "creator_id": creator_id,
                    "raw_sha256": raw_sha256,
                    "decoded_pixel_sha256": pixel_sha256,
                    "source_url": source_url,
                }
            )
            source_records.append(
                {
                    "id": image_id,
                    "label": class_name,
                    "creator": creator,
                    "creator_id": creator_id,
                    "raw_sha256": raw_sha256,
                    "pixel_sha256": pixel_sha256,
                    "source_url": source_url,
                    "phash": phash,
                }
            )
            component_members[f"component-{image_id}"] = {"eligible_ids": [image_id]}
        rows_by_split[split] = rows
        np.save(
            data_root / "arrays" / f"{split}_images.npy",
            np.stack(pixels),
            allow_pickle=False,
        )
        np.save(
            data_root / "arrays" / f"{split}_labels.npy",
            np.asarray(labels, dtype=np.int32),
            allow_pickle=False,
        )
        (data_root / "splits" / f"{split}.json").write_text(
            json.dumps(rows, indent=2, sort_keys=True) + "\n"
        )
    manifest = {
        "grouping": {"component_members": component_members},
        "records": source_records,
    }
    (data_root / "source-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    archive_path = root / "synthetic-final-only.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        final_images = np.load(
            data_root / "arrays" / "final_images.npy", allow_pickle=False
        )
        for row, pixels in zip(rows_by_split["final"], final_images):
            stream = io.BytesIO()
            Image.fromarray(pixels).save(stream, format="JPEG", quality=90)
            encoded = stream.getvalue()
            info = tarfile.TarInfo(f"flower_photos/{row['id']}")
            info.size = len(encoded)
            archive.addfile(info, io.BytesIO(encoded))
    return data_root, recipe_path, archive_path


def _graph_stage(candidate_path: Path, output: Path, active_recipe: Path) -> dict:
    """Run all eleven real training/evaluation/export compositions on synthetic rows only."""
    graph_root = output / "synthetic-complete-graph"
    if graph_root.exists():
        raise FileExistsError("graph stage output must be fresh")
    graph_root.mkdir(parents=True)
    data_root, recipe_path, archive_path = _make_synthetic_graph_inputs(
        graph_root, active_recipe
    )
    candidate_dir = graph_root / "candidate"
    shutil.copytree(
        candidate_path.parent,
        candidate_dir,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    candidate_model = candidate_dir / "model.py"
    candidate_source_sha = hashlib.sha256(candidate_model.read_bytes()).hexdigest()
    graph_output = graph_root / "run"

    driver = graph_root / "driver.py"
    driver.write_text("""import argparse, json, sys
from pathlib import Path
project = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(project))
import run
parser = argparse.ArgumentParser()
parser.add_argument('--candidate', required=True)
parser.add_argument('--recipe', required=True)
parser.add_argument('--data', required=True)
parser.add_argument('--archive', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--interrupt-after-stage')
args = parser.parse_args(sys.argv[2:])
api, model_source = run._load_candidate_apis(args.candidate)
candidate_hashes = run._candidate_source_inventory(model_source.parent)
candidate_paths = {name: model_source.parent / name for name in candidate_hashes}
sys.path.insert(0, str(project / 'solution'))
from pipeline import execute, InterruptedStage
try:
    result = execute(recipe_path=Path(args.recipe), data_directory=Path(args.data), archive_path=Path(args.archive),
        output_directory=Path(args.output), implementation_source=model_source, api_modules=api,
        candidate_source_paths=candidate_paths, candidate_source_sha256=candidate_hashes,
        resume=args.resume, interrupt_after_stage=args.interrupt_after_stage)
except InterruptedStage as error:
    print(json.dumps({'status':'interrupted','message':str(error)}))
    raise SystemExit(42)
Path(args.output, 'run-report.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\\n')
print(json.dumps({'status':'pass','stage_count':len(result['stage_order'])}))
""")
    command_base = [
        sys.executable,
        str(driver),
        str(ROOT),
        "--candidate",
        str(candidate_dir),
        "--recipe",
        str(recipe_path),
        "--data",
        str(data_root),
        "--archive",
        str(archive_path),
        "--output",
        str(graph_output),
    ]
    env = dict(os.environ)
    env["JAX_PLATFORM_NAME"] = "cpu"
    interruption_stages = (
        "baseline-vit",
        "mae-pretrain",
        "simclr-pretrain",
        "probe-random",
        "partial-selected",
    )
    interruption_results = []
    for index, stage_name in enumerate(interruption_stages):
        command = command_base + ["--interrupt-after-stage", stage_name]
        if index:
            command.append("--resume")
        result = subprocess.run(
            command, capture_output=True, text=True, env=env, timeout=900
        )
        if result.returncode != 42 or '"status": "interrupted"' not in result.stdout:
            raise AssertionError(
                f"synthetic complete graph did not interrupt at {stage_name}: {result.returncode}\n{result.stdout}\n{result.stderr}"
            )
        interruption_results.append(stage_name)
    final_process = subprocess.run(
        command_base + ["--resume"],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    if final_process.returncode != 0:
        raise AssertionError(
            f"synthetic complete graph did not resume to completion:\n{final_process.stdout}\n{final_process.stderr}"
        )

    report_path = graph_output / "run-report.json"
    report = json.loads(report_path.read_text())
    simclr_trace = report["stage_traces"]["simclr-pretrain"]
    if not simclr_trace or any(
        event.get("source_positive_relation_checked") is not True
        or not isinstance(event.get("source_positive_pair_count"), int)
        for event in simclr_trace
    ):
        raise AssertionError(
            "synthetic graph did not compare both pHash-derived SimCLR relations"
        )
    simclr_pair_counts = {event["source_positive_pair_count"] for event in simclr_trace}
    if len(simclr_pair_counts) != 1:
        raise AssertionError(
            "SimCLR source-positive pair inventory changed between steps"
        )
    uninterrupted_output = graph_root / "uninterrupted-run"
    uninterrupted_command = command_base.copy()
    uninterrupted_command[uninterrupted_command.index("--output") + 1] = str(
        uninterrupted_output
    )
    uninterrupted_process = subprocess.run(
        uninterrupted_command, capture_output=True, text=True, env=env, timeout=900
    )
    if uninterrupted_process.returncode != 0:
        raise AssertionError(
            f"synthetic uninterrupted graph failed:\n{uninterrupted_process.stdout}\n{uninterrupted_process.stderr}"
        )
    uninterrupted_report = json.loads(
        (uninterrupted_output / "run-report.json").read_text()
    )
    if report["stage_traces"] != uninterrupted_report["stage_traces"]:
        raise AssertionError(
            "fresh-process checkpoint recovery changed a stage's sampled IDs or loss trace"
        )
    for stage_name in interruption_stages:
        boundary = report["stage_traces"][stage_name][2]["epoch_spans"]
        if boundary != [
            {"epoch": 0, "start": 8, "count": 2},
            {"epoch": 1, "start": 0, "count": 2},
        ]:
            raise AssertionError(
                f"{stage_name} resume did not preserve the no-drop partial epoch tail"
            )
    for stage_name in report["stage_order"]:
        resumed_receipt = report["stage_checkpoints"][stage_name]
        uninterrupted_receipt = uninterrupted_report["stage_checkpoints"][stage_name]
        for field in (
            "checkpoint_npz_sha256",
            "checkpoint_metadata_sha256",
            "consumed_id_history_sha256",
        ):
            if resumed_receipt[field] != uninterrupted_receipt[field]:
                raise AssertionError(
                    f"resumed {stage_name} state differs from uninterrupted {field}"
                )
    journal = json.loads((graph_output / "pipeline-journal.json").read_text())
    if len(report["stage_order"]) != 11 or any(
        journal["stages"].get(name, {}).get("status") != "complete"
        for name in report["stage_order"]
    ):
        raise AssertionError(
            "synthetic graph did not execute and reload every declared stage"
        )
    run_spec = importlib.util.spec_from_file_location(
        "image_graph_run_preflight", ROOT / "run.py"
    )
    if run_spec is None or run_spec.loader is None:
        raise ImportError("could not load source inventory preflight")
    run_module = importlib.util.module_from_spec(run_spec)
    run_spec.loader.exec_module(run_module)
    approved_inventory = run_module._candidate_source_inventory(candidate_dir)
    if journal["run_identity"].get("candidate_api_source_sha256") != approved_inventory:
        raise AssertionError(
            "pipeline journal omitted a consumed candidate API/dependency source"
        )
    if not {
        "model.py",
        "objectives.py",
        "augment.py",
        "training.py",
        "evaluate.py",
        "runner.py",
    } <= set(approved_inventory):
        raise AssertionError(
            "pipeline source inventory omitted a required candidate API"
        )

    approval = {
        "schema": "jaxpathways.image-pilot-optimization-approval.v1",
        "status": "approved",
        "reviewer": "Sol",
        "recipe_sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(
            (data_root / "source-manifest.json").read_bytes()
        ).hexdigest(),
        "implementation_source_sha256": hashlib.sha256(
            candidate_model.read_bytes()
        ).hexdigest(),
        "candidate_api_source_sha256": approved_inventory,
        "no_recipe_changes_since_review": True,
    }
    approval_path = graph_root / "synthetic-approval.json"
    approval_path.write_text(json.dumps(approval, sort_keys=True))
    run_module._verify_approval(
        approval_path,
        recipe_path,
        {"source_manifest_sha256": approval["source_manifest_sha256"]},
        candidate_model,
        approved_inventory,
    )
    dependency_controls = {}
    for dependency in ("objectives.py", "augment.py", "training.py"):
        changed_dir = graph_root / f"candidate-changed-{dependency[:-3]}"
        shutil.copytree(candidate_dir, changed_dir)
        changed_file = changed_dir / dependency
        changed_file.write_text(
            changed_file.read_text(encoding="utf-8") + "\n# source-binding mutation\n",
            encoding="utf-8",
        )
        changed_inventory = run_module._candidate_source_inventory(changed_dir)
        if (
            hashlib.sha256((changed_dir / "model.py").read_bytes()).hexdigest()
            != candidate_source_sha
        ):
            raise AssertionError(
                "dependency mutation unexpectedly changed candidate model"
            )
        try:
            run_module._verify_approval(
                approval_path,
                recipe_path,
                {"source_manifest_sha256": approval["source_manifest_sha256"]},
                changed_dir / "model.py",
                changed_inventory,
            )
        except PermissionError as error:
            if "candidate_api_source_sha256" not in str(error):
                raise AssertionError(
                    f"{dependency} mutation failed for an unrelated approval field"
                ) from error
            dependency_controls[dependency] = "rejected by approved source inventory"
        else:
            raise AssertionError(
                f"approval accepted changed external dependency {dependency}"
            )
        if dependency == "training.py":
            changed_training_dir = changed_dir

    changed_training_command = command_base.copy()
    candidate_argument = changed_training_command.index("--candidate") + 1
    changed_training_command[candidate_argument] = str(changed_training_dir)
    changed_training_result = subprocess.run(
        changed_training_command + ["--resume"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    changed_training_error = (
        changed_training_result.stdout + changed_training_result.stderr
    )
    if (
        changed_training_result.returncode == 0
        or "pipeline resume identity differs from the frozen source/data/recipe"
        not in changed_training_error
    ):
        raise AssertionError(
            "pipeline resumed after an external training dependency changed"
        )
    if not journal.get("final_split_loaded_after_deploy_selection"):
        raise AssertionError(
            "final scoring occurred without a durable development-only selection lock"
        )
    events = [event["event"] for event in journal["events"]]
    if events.index("deploy_selection_frozen") > events.index("final_split_loaded"):
        raise AssertionError(
            "pipeline accessed final split before freezing development choices"
        )
    if report["candidate_model_source_sha256"] != candidate_source_sha:
        raise AssertionError(
            "run report did not bind the selected candidate model source"
        )
    for relative, digest in report["raw_artifact_sha256"].items():
        artifact = graph_output / relative
        if (
            not artifact.is_file()
            or hashlib.sha256(artifact.read_bytes()).hexdigest() != digest
        ):
            raise AssertionError(f"raw comparison artifact hash mismatch: {relative}")
    for name, exposure in report["stage_execution"].items():
        objective = exposure["objective"].lower()
        updates, batch = exposure["updates"], exposure["batch_size"]
        expected_views = 2 if "nt-xent" in objective else 1
        expected_pixels = updates * batch * expected_views * 64 * 64 * 3
        if exposure["measured_model_views"] != updates * batch * expected_views:
            raise AssertionError(f"model view exposure is incorrect for {name}")
        if exposure["measured_pixel_values"] != expected_pixels:
            raise AssertionError(
                f"actual encoder pixel exposure is incorrect for {name}"
            )
        if "visible-only encoder" in objective:
            expected_targets = updates * batch * 2 * (32 * 32 * 3)
        else:
            expected_targets = 0
        if exposure["measured_selected_mae_target_values"] != expected_targets:
            raise AssertionError(
                f"objective-derived MAE target exposure is incorrect for {name}"
            )

    raw_root = graph_output / "raw"
    comparisons = np.load(
        raw_root / "all-classifier-comparisons.npz", allow_pickle=False
    )
    index = json.loads((raw_root / "classifier-comparison-index.json").read_text())
    if set(index["arrays"]) != set(report["dev_metrics"]) or set(
        index["arrays"]
    ) != set(report["final_metrics"]):
        raise AssertionError("comparison arrays omitted a candidate or fixed control")
    for name, keys in index["arrays"].items():
        for split, metric_name, labels_key in (
            ("development", "dev_metrics", "dev_labels"),
            ("final", "final_metrics", "final_labels"),
        ):
            independent = _independent_accuracy_nll(
                comparisons[keys[split]], comparisons[labels_key]
            )
            reported = report[metric_name][name]["raw"]
            if not np.isclose(
                independent["macro_balanced_accuracy"],
                reported["macro_balanced_accuracy"],
                atol=1e-12,
            ):
                raise AssertionError(
                    f"{split} balanced-accuracy report does not match raw logits for {name}"
                )
            if not np.isclose(
                independent["nll"], reported["nll"], atol=1e-10, rtol=1e-10
            ):
                raise AssertionError(
                    f"{split} NLL report does not match raw logits for {name}"
                )
            calibrated_report = report[metric_name][name].get("calibrated")
            if calibrated_report is not None:
                temperature = float(
                    report["temperature_selection"][name]["temperature"]
                )
                calibrated = _independent_accuracy_nll(
                    comparisons[keys[split]], comparisons[labels_key], temperature
                )
                if not np.isclose(
                    calibrated["nll"], calibrated_report["nll"], atol=1e-10, rtol=1e-10
                ):
                    raise AssertionError(
                        f"{split} calibrated NLL does not match recipe-development temperature for {name}"
                    )
    ranking_artifact = json.loads(
        (raw_root / "final-retrieval-rankings.json").read_text()
    )
    retrieval = np.load(raw_root / "final-retrieval-features.npz", allow_pickle=False)
    expected_ids = retrieval["ids"].astype(str).tolist()
    component_ids = retrieval["component_ids"].astype(str).tolist()
    for arm, payload in ranking_artifact.items():
        rankings = payload["rankings"]
        if (
            set(rankings) != set(expected_ids)
            or len(rankings) != len(expected_ids)
            or any(
                len(values) != len(expected_ids) - 1
                or len(set(values)) != len(values)
                or query_id in values
                for query_id, values in rankings.items()
            )
        ):
            raise AssertionError(
                f"retrieval rankings for {arm} are incomplete or contain duplicate/self IDs"
            )
        features = np.asarray(retrieval[f"features__{arm}"], dtype=np.float64)
        normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
        for query_index, query_id in enumerate(expected_ids):
            candidates = [
                index
                for index, candidate_id in enumerate(expected_ids)
                if index != query_index
                and component_ids[index] != component_ids[query_index]
            ]
            expected_ranking = sorted(
                candidates,
                key=lambda index: (
                    -float(normalized[index] @ normalized[query_index]),
                    expected_ids[index],
                ),
            )
            if rankings[query_id] != [
                expected_ids[index] for index in expected_ranking
            ]:
                raise AssertionError(
                    f"retained retrieval ranking for {arm}/{query_id} differs from exact cosine oracle"
                )
        labels_by_id = dict(zip(expected_ids, retrieval["labels"].astype(int).tolist()))
        components_by_id = dict(zip(expected_ids, component_ids))
        query_recalls = {1: [], 5: [], 10: []}
        average_precisions = []
        for query_id in expected_ids:
            relevant_total = sum(
                labels_by_id[candidate_id] == labels_by_id[query_id]
                and components_by_id[candidate_id] != components_by_id[query_id]
                for candidate_id in expected_ids
                if candidate_id != query_id
            )
            ranked = rankings[query_id]
            hits = np.asarray(
                [
                    labels_by_id[candidate_id] == labels_by_id[query_id]
                    and components_by_id[candidate_id] != components_by_id[query_id]
                    for candidate_id in ranked
                ],
                dtype=bool,
            )
            if relevant_total:
                cumulative = np.cumsum(hits)
                average_precisions.append(
                    float(
                        np.sum((cumulative / (np.arange(hits.size) + 1)) * hits)
                        / relevant_total
                    )
                )
                for k in query_recalls:
                    query_recalls[k].append(float(np.sum(hits[:k]) / relevant_total))
        for k, values in query_recalls.items():
            if not np.isclose(
                payload["recall_at_k"][str(k)], np.mean(values), atol=1e-12
            ):
                raise AssertionError(
                    f"retained recall@{k} for {arm} differs from independent qrel reconstruction"
                )
        if not np.isclose(
            payload["mean_average_precision"], np.mean(average_precisions), atol=1e-12
        ):
            raise AssertionError(
                f"retained MAP for {arm} differs from independent qrel reconstruction"
            )

    condition_arrays = np.load(
        raw_root / "selected-final-condition-logits.npz", allow_pickle=False
    )
    selected_name = report["deploy_selection"]["dev_only_selected"]
    selected_temperature = float(
        report["temperature_selection"][selected_name]["temperature"]
    )
    for condition in report["selected_final_conditions"]:
        independent = _independent_accuracy_nll(
            condition_arrays[f"logits__{condition}"], comparisons["final_labels"]
        )
        measured = report["selected_final_conditions"][condition]["raw"]
        if not np.isclose(
            independent["macro_balanced_accuracy"],
            measured["macro_balanced_accuracy"],
            atol=1e-12,
        ):
            raise AssertionError(
                f"condition metric for {condition} does not match retained candidate logits"
            )
        if not np.isclose(independent["nll"], measured["nll"], atol=1e-10, rtol=1e-10):
            raise AssertionError(
                f"condition NLL for {condition} does not match retained candidate logits"
            )
        calibrated = _independent_accuracy_nll(
            condition_arrays[f"logits__{condition}"],
            comparisons["final_labels"],
            selected_temperature,
        )
        measured_calibrated = report["selected_final_conditions"][condition][
            "calibrated"
        ]
        if not np.isclose(
            calibrated["nll"], measured_calibrated["nll"], atol=1e-10, rtol=1e-10
        ):
            raise AssertionError(
                f"recipe-development calibration for {condition} is not independently reproducible"
            )

    # The pipeline exported the selected candidate module. Independently load
    # its private local bundle and prove the candidate sentinel survives cold load.
    bundle = graph_output / "local-consumer"
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest["files"]["model.py"] != candidate_source_sha:
        raise AssertionError(
            "local consumer did not export the selected candidate model source"
        )
    cold_script = graph_root / "cold_consumer.py"
    cold_script.write_text(
        "import json,sys\nfrom consumer_runtime import LocalImageClassifier\n"
        "model=LocalImageClassifier.from_pretrained(sys.argv[1],local_files_only=True)\n"
        "print(json.dumps(model.predict_bytes(open(sys.argv[2],'rb').read())))\n"
    )
    cold_dir = graph_root / "unrelated-cwd"
    cold_dir.mkdir()
    canary = report["consumer_package"]["raw_image_canaries"][0]
    raw_member = f"flower_photos/{canary['id']}"
    with tarfile.open(archive_path, "r:gz") as archive:
        stream = archive.extractfile(raw_member)
        if stream is None:
            raise AssertionError("synthetic raw-image canary bytes are missing")
        canary_path = graph_root / "canary.jpg"
        canary_path.write_bytes(stream.read())
    cold_env = {
        key: value
        for key, value in env.items()
        if key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    cold_env["PYTHONPATH"] = str(bundle)
    cold = subprocess.run(
        [sys.executable, str(cold_script), str(bundle), str(canary_path)],
        cwd=cold_dir,
        capture_output=True,
        text=True,
        env=cold_env,
        timeout=120,
    )
    if cold.returncode != 0:
        raise AssertionError(
            f"offline candidate consumer cold load failed: {cold.stdout}\n{cold.stderr}"
        )
    cold_prediction = json.loads(cold.stdout)
    if not np.allclose(cold_prediction["logits"], canary["logits"], atol=0, rtol=0):
        raise AssertionError(
            "cold local package prediction differs from the final run's candidate canary"
        )

    selected_candidate, _, _ = load_candidate(str(candidate_model))
    train_steps_spec = importlib.util.spec_from_file_location(
        "image_graph_train_steps", ROOT / "solution" / "train_steps.py"
    )
    if train_steps_spec is None or train_steps_spec.loader is None:
        raise ImportError(
            "could not load the training composition for candidate-path verification"
        )
    train_steps = importlib.util.module_from_spec(train_steps_spec)
    train_steps_spec.loader.exec_module(train_steps)
    reference_spec = importlib.util.spec_from_file_location(
        "image_graph_reference_model", ROOT / "solution" / "model.py"
    )
    if reference_spec is None or reference_spec.loader is None:
        raise ImportError("could not load the independent model comparison")
    reference_model = importlib.util.module_from_spec(reference_spec)
    reference_spec.loader.exec_module(reference_model)
    selected_runner = load_runner(candidate_model, "external")
    selected_training = load_training(candidate_model, "external")
    selected_augmentation = load_augmentation(candidate_model, "external")
    fit_ids = [
        row["id"] for row in json.loads((data_root / "splits" / "fit.json").read_text())
    ]
    fit_images = np.load(data_root / "arrays" / "fit_images.npy", allow_pickle=False)
    fit_labels = np.load(data_root / "arrays" / "fit_labels.npy", allow_pickle=False)
    order = sorted(range(len(fit_ids)), key=lambda index: fit_ids[index])
    sampled, _, _ = selected_training.next_id_batch(
        selected_training.init_id_stream(
            len(fit_ids), selected_runner.ROOT_SEEDS["fit_order"]
        ),
        4,
    )
    sampled_positions = np.asarray(order)[sampled]
    batch = fit_images[sampled_positions].astype(np.float32) / np.float32(
        127.5
    ) - np.float32(1.0)
    labels = fit_labels[sampled_positions].astype(np.int32)
    view_keys = selected_runner._keys_for_ids(
        selected_runner.stage_key(
            selected_runner.ROOT_SEEDS["augmentation"], "baseline-vit"
        ),
        0,
        sampled,
        1,
    )
    candidate_batch = selected_augmentation.supervised_views(
        jnp.asarray(batch), view_keys
    )
    dropout_key = jax.random.fold_in(
        selected_runner.stage_key(
            selected_runner.ROOT_SEEDS["dropout"], "baseline-vit"
        ),
        0,
    )
    recipe = json.loads(recipe_path.read_text())
    selected_candidate.PILOT_CONFIG = dict(recipe["checker_fixture"]["model_config"])
    initial_params = selected_candidate.init_vit(
        jax.random.PRNGKey(int(recipe["pilot"]["seeds"]["initial_vit"]))
    )
    initial_trainable, initial_static = train_steps.split_vit(initial_params)
    initial_head = selected_candidate.init_linear_head(
        selected_runner.stage_key(1729, "head:baseline-vit"),
        int(initial_params["config"]["width"]),
        int(initial_params["config"]["classes"]),
    )
    actual_candidate_loss = train_steps.vit_supervised_loss(
        initial_trainable,
        initial_head,
        initial_static,
        candidate_batch,
        jnp.asarray(labels),
        dropout_key,
        model_api=selected_candidate,
    )
    measured_first_loss = report["stage_traces"]["baseline-vit"][0]["loss"]
    if not np.isclose(
        float(actual_candidate_loss), measured_first_loss, atol=2e-6, rtol=2e-6
    ):
        raise AssertionError(
            "baseline training trace does not use the selected candidate model API"
        )
    # Keep the bypass negative control checker-owned: perturb an owned reference
    # model through its public classifier API, then prove the training objective
    # distinguishes it from accidentally calling the unperturbed global model.
    owned_sentinel_spec = importlib.util.spec_from_file_location(
        "image_graph_owned_sentinel", ROOT / "solution" / "model.py"
    )
    if owned_sentinel_spec is None or owned_sentinel_spec.loader is None:
        raise ImportError("could not load owned model sentinel fixture")
    owned_sentinel = importlib.util.module_from_spec(owned_sentinel_spec)
    owned_sentinel_spec.loader.exec_module(owned_sentinel)
    owned_sentinel.PILOT_CONFIG = dict(recipe["checker_fixture"]["model_config"])
    original_logits = owned_sentinel.classification_logits

    def sentinel_logits(params, head, images, **kwargs):
        base = original_logits(params, head, images, **kwargs)
        offset = jnp.asarray([0.5, -0.25, 0.125, -0.125, 0.0], base.dtype)
        return base + offset

    owned_sentinel.classification_logits = sentinel_logits
    sentinel_params = owned_sentinel.init_vit(
        jax.random.PRNGKey(int(recipe["pilot"]["seeds"]["initial_vit"]))
    )
    sentinel_trainable, sentinel_static = train_steps.split_vit(sentinel_params)
    sentinel_head = owned_sentinel.init_linear_head(
        selected_runner.stage_key(1729, "head:baseline-vit"),
        int(sentinel_params["config"]["width"]),
        int(sentinel_params["config"]["classes"]),
    )
    sentinel_loss = train_steps.vit_supervised_loss(
        sentinel_trainable,
        sentinel_head,
        sentinel_static,
        candidate_batch,
        jnp.asarray(labels),
        dropout_key,
        model_api=owned_sentinel,
    )
    bypassed_sentinel_loss = train_steps.vit_supervised_loss(
        sentinel_trainable,
        sentinel_head,
        sentinel_static,
        candidate_batch,
        jnp.asarray(labels),
        dropout_key,
        model_api=reference_model,
    )
    owned_bypass_sentinel_delta = abs(
        float(sentinel_loss) - float(bypassed_sentinel_loss)
    )
    if owned_bypass_sentinel_delta <= 1e-6:
        raise AssertionError(
            "owned public-API sentinel did not detect bypass to the reference model"
        )

    bundle_runtime_spec = importlib.util.spec_from_file_location(
        "image_graph_bundle_runtime", bundle / "consumer_runtime.py"
    )
    if bundle_runtime_spec is None or bundle_runtime_spec.loader is None:
        raise ImportError("could not load the exported consumer runtime")
    bundle_runtime = importlib.util.module_from_spec(bundle_runtime_spec)
    bundle_runtime_spec.loader.exec_module(bundle_runtime)
    bundled_model = bundle_runtime.LocalImageClassifier.from_pretrained(
        bundle, local_files_only=True
    )
    final_pixels = np.load(
        data_root / "arrays" / "final_images.npy", allow_pickle=False
    )
    final_normalized = final_pixels.astype(np.float32) / np.float32(127.5) - np.float32(
        1.0
    )
    bundled_logits = np.asarray(
        bundled_model.model.classification_logits(
            bundled_model.params, bundled_model.head, jnp.asarray(final_normalized)
        )
    )
    selected_name = report["deploy_selection"]["dev_only_selected"]
    selected_key = index["arrays"][selected_name]["final"]
    if not np.array_equal(bundled_logits, comparisons[selected_key]):
        raise AssertionError(
            "final evaluation did not use the same selected candidate API as local export"
        )

    # A byte mutation with a recomputed checkpoint-internal checksum must not
    # pass the frozen pipeline journal's stage-state boundary.
    damaged_run = graph_root / "damaged-run"
    shutil.copytree(graph_output, damaged_run)
    first_stage = report["stage_order"][0]
    frozen_stage = journal["stages"][first_stage]
    checkpoint = Path(frozen_stage["checkpoint_prefix"])
    checkpoint_relative = (
        checkpoint
        if not checkpoint.is_absolute()
        else checkpoint.relative_to(graph_output)
    )
    checkpoint_copy = damaged_run / checkpoint_relative
    altered = bytearray(checkpoint_copy.with_suffix(".npz").read_bytes())
    altered[-1] ^= 1
    checkpoint_copy.with_suffix(".npz").write_bytes(altered)
    damaged_cmd = [*command_base[:-1], str(damaged_run), "--resume"]
    damaged_process = subprocess.run(
        damaged_cmd, capture_output=True, text=True, env=env, timeout=120
    )
    if (
        damaged_process.returncode == 0
        or "checkpoint bytes changed after journal freeze" not in damaged_process.stderr
    ):
        raise AssertionError("pipeline accepted a mutated completed stage checkpoint")

    return {
        "status": "pass",
        "synthetic_only": True,
        "stage_count": len(report["stage_order"]),
        "fresh_process_interruption_and_resume": interruption_results,
        "all_stage_checkpoints_reloaded": True,
        "candidate_source_inventory_bound": len(approved_inventory),
        "changed_external_dependency_approval_rejections": dependency_controls,
        "changed_training_dependency_resume_rejected": True,
        "candidate_model_source_sha256": candidate_source_sha,
        "owned_reference_api_bypass_sentinel_loss_delta": owned_bypass_sentinel_delta,
        "candidate_api_used_in_training_evaluation_and_export": True,
        "source_inventory_positive_relation_checked": True,
        "simclr_source_positive_pair_count": next(iter(simclr_pair_counts)),
        "dev_and_final_metrics_reconstructed_from_raw_logits": len(index["arrays"]),
        "retrieval_query_count": len(expected_ids),
        "offline_candidate_bundle_cold_process": "exact canary logits",
        "mutated_checkpoint_rejected": True,
        "output": str(graph_output),
        "note": "This reduced checker-owned graph validates stage composition and resume mechanics only; it contains no flower photos, is not a quality estimate, and performs no authorized-corpus optimization.",
    }


def _load_solution_pipeline_module() -> ModuleType:
    solution_directory = ROOT / "solution"
    if str(solution_directory) not in sys.path:
        sys.path.insert(0, str(solution_directory))
    module_path = solution_directory / "pipeline.py"
    module_spec = importlib.util.spec_from_file_location(
        "image_pipeline_input_loader_under_test", module_path
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError("could not load the production pipeline input loader")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _pipeline_input_loader_stage(data_directory: Path, recipe_path: Path) -> dict:
    """Exercise the exact production split loader without initializing a model."""
    pipeline_module = _load_solution_pipeline_module()
    data_root = data_directory.expanduser().resolve()
    class_order = tuple(json.loads(recipe_path.read_text())["data"]["class_order"])
    manifest_path = data_root / "source-manifest.json"
    manifest_before = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text())
    source_by_id = {row["id"]: row for row in manifest["records"]}
    trusted_source_index = pipeline_module._source_metadata_index(manifest)
    actual_counts = {}
    split_file_hashes = {}
    array_file_hashes = {}
    joined_metadata_hashes = {}
    for split in ("fit", "recipe_dev", "final"):
        split_path = data_root / "splits" / f"{split}.json"
        raw_rows = json.loads(split_path.read_text())
        if any("phash" in row for row in raw_rows):
            raise AssertionError(
                "frozen split fixture unexpectedly duplicates source-inventory pHash"
            )
        loaded = pipeline_module.load_split_inputs(
            data_root,
            split,
            source_metadata_by_id=trusted_source_index,
            class_order=class_order,
        )
        if [row["id"] for row in loaded["rows"]] != [row["id"] for row in raw_rows]:
            raise AssertionError(
                "pipeline input loader changed the frozen split ID order"
            )
        if any(
            row["phash"] != source_by_id[row["id"]]["phash"] for row in loaded["rows"]
        ):
            raise AssertionError("pipeline loader did not join trusted source pHashes")
        for row, label in zip(loaded["rows"], loaded["labels"]):
            source_row = source_by_id[row["id"]]
            if (
                row["label"] != source_row["label"]
                or row["class_index"] != class_order.index(row["label"])
                or row["raw_sha256"] != source_row["raw_sha256"]
                or row["decoded_pixel_sha256"] != source_row["pixel_sha256"]
                or row["creator"] != source_row["creator"]
                or row["creator_id"] != source_row["creator_id"]
                or row["source_url"] != source_row["source_url"]
                or int(label) != row["class_index"]
            ):
                raise AssertionError(
                    f"pipeline loader changed split/source identity for {row['id']}"
                )
        joined = hashlib.sha256()
        for row in loaded["rows"]:
            joined.update(row["id"].encode("utf-8"))
            joined.update(b"\0")
            joined.update(row["phash"].lower().encode("ascii"))
            joined.update(b"\n")
        joined_metadata_hashes[split] = joined.hexdigest()
        if loaded["images"].shape[0] != len(raw_rows) or loaded["labels"].shape != (
            len(raw_rows),
        ):
            raise AssertionError(
                "pipeline input loader changed source row/array counts"
            )
        actual_counts[split] = len(raw_rows)
        split_file_hashes[split] = hashlib.sha256(split_path.read_bytes()).hexdigest()
        for suffix in ("images", "labels"):
            array_path = data_root / "arrays" / f"{split}_{suffix}.npy"
            array_file_hashes[array_path.name] = hashlib.sha256(
                array_path.read_bytes()
            ).hexdigest()
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_before:
        raise AssertionError(
            "pipeline input loading modified the frozen source manifest"
        )

    with tempfile.TemporaryDirectory(prefix="c03-source-identity-join-") as temp:
        fixture = Path(temp)
        (fixture / "splits").mkdir()
        (fixture / "arrays").mkdir()
        fixture_rows = []
        fixture_records = []
        for index, (image_id, label, phash) in enumerate(
            (
                ("daisy/a.jpg", "daisy", "0000000000000000"),
                ("dandelion/b.jpg", "dandelion", "0000000000000001"),
            )
        ):
            raw_sha = hashlib.sha256((image_id + ":raw").encode()).hexdigest()
            pixel_sha = hashlib.sha256((image_id + ":pixels").encode()).hexdigest()
            creator = f"fixture creator {index}"
            creator_id = f"fixture-creator-{index}"
            source_url = f"https://example.invalid/photos/{index}"
            fixture_rows.append(
                {
                    "id": image_id,
                    "label": label,
                    "class_index": class_order.index(label),
                    "creator": creator,
                    "creator_id": creator_id,
                    "raw_sha256": raw_sha,
                    "decoded_pixel_sha256": pixel_sha,
                    "source_url": source_url,
                }
            )
            fixture_records.append(
                {
                    "id": image_id,
                    "label": label,
                    "creator": creator,
                    "creator_id": creator_id,
                    "raw_sha256": raw_sha,
                    "pixel_sha256": pixel_sha,
                    "source_url": source_url,
                    "phash": phash,
                }
            )
        (fixture / "splits" / "fit.json").write_text(json.dumps(fixture_rows))
        np.save(
            fixture / "arrays" / "fit_images.npy",
            np.zeros((2, 64, 64, 3), dtype=np.uint8),
            allow_pickle=False,
        )
        np.save(
            fixture / "arrays" / "fit_labels.npy",
            np.asarray([0, 1], dtype=np.int32),
            allow_pickle=False,
        )

        def reset_fixture(rows=None, records=None, labels=(0, 1)):
            (fixture / "splits" / "fit.json").write_text(
                json.dumps(fixture_rows if rows is None else rows)
            )
            (fixture / "source-manifest.json").write_text(
                json.dumps({"records": fixture_records if records is None else records})
            )
            np.save(
                fixture / "arrays" / "fit_labels.npy",
                np.asarray(labels, dtype=np.int32),
                allow_pickle=False,
            )

        reset_fixture()
        loaded_fixture = pipeline_module.load_split_inputs(
            fixture, "fit", class_order=class_order
        )
        known_pairs = pipeline_module._positive_pairs(loaded_fixture["rows"])
        expected_pair = (fixture_rows[0]["id"], fixture_rows[1]["id"])
        if known_pairs != {expected_pair}:
            raise AssertionError("production loader lost the known pHash-neighbor pair")

        mutations = {"known_phash_neighbor_positive": "accepted and related"}
        identity_cases = {
            "split_source_label_mismatch": (
                lambda rows, records: rows[0].update(label="dandelion", class_index=1),
                "split/source label mismatch",
            ),
            "split_source_raw_mismatch": (
                lambda rows, records: rows[0].update(raw_sha256="b" * 64),
                "split/source raw_sha256 mismatch",
            ),
            "split_source_pixel_mismatch": (
                lambda rows, records: rows[0].update(decoded_pixel_sha256="c" * 64),
                "split/source decoded_pixel_sha256 mismatch",
            ),
            "split_class_index_mismatch": (
                lambda rows, records: rows[0].update(class_index=1),
                "split class_index does not match frozen class order",
            ),
            "missing_split_identity_field": (
                lambda rows, records: rows[0].pop("raw_sha256"),
                "missing identity fields",
            ),
            "missing_source_identity_field": (
                lambda rows, records: records[0].pop("pixel_sha256"),
                "missing identity fields",
            ),
        }
        for name, (mutate, expected_error) in identity_cases.items():
            rows = json.loads(json.dumps(fixture_rows))
            records = json.loads(json.dumps(fixture_records))
            mutate(rows, records)
            reset_fixture(rows, records)
            try:
                pipeline_module.load_split_inputs(
                    fixture, "fit", class_order=class_order
                )
            except ValueError as error:
                if expected_error not in str(error):
                    raise AssertionError(
                        f"{name} failed for the wrong reason: {error}"
                    ) from error
                mutations[name] = "rejected at production identity check"
            else:
                raise AssertionError(f"pipeline input loader accepted {name}")

        reset_fixture(labels=(4, 1))
        try:
            pipeline_module.load_split_inputs(fixture, "fit", class_order=class_order)
        except ValueError as error:
            if "row/array class label mismatch" not in str(error):
                raise AssertionError(
                    f"row/array mismatch failed for the wrong reason: {error}"
                ) from error
            mutations["row_array_label_mismatch"] = "rejected at production identity check"
        else:
            raise AssertionError("pipeline input loader accepted a row/array label mismatch")

        cases = {
            "duplicate_source_id": (
                lambda records: records.append(dict(records[0])),
                "duplicate image ID",
            ),
            "split_id_missing_from_inventory": (
                lambda records: records.pop(0),
                "missing from source inventory",
            ),
            "invalid_source_phash": (
                lambda records: records[0].update(phash="01234x6789abcdef"),
                "invalid 16-digit pHash",
            ),
        }
        for name, (mutate, expected_error) in cases.items():
            records = json.loads(json.dumps(fixture_records))
            mutate(records)
            reset_fixture(records=records)
            try:
                pipeline_module.load_split_inputs(
                    fixture, "fit", class_order=class_order
                )
            except ValueError as error:
                if expected_error not in str(error):
                    raise AssertionError(
                        f"{name} failed for the wrong reason: {error}"
                    ) from error
                mutations[name] = "rejected at production identity check"
            else:
                raise AssertionError(f"pipeline input loader accepted {name}")

        duplicate_rows = [*fixture_rows, dict(fixture_rows[1])]
        reset_fixture(rows=duplicate_rows)
        try:
            pipeline_module.load_split_inputs(fixture, "fit", class_order=class_order)
        except ValueError as error:
            if "duplicate image ID" not in str(error):
                raise
            mutations["duplicate_split_id"] = "rejected at production identity check"
        else:
            raise AssertionError("pipeline input loader accepted duplicate split IDs")

        rows_with_phash = json.loads(json.dumps(fixture_rows))
        rows_with_phash[0]["phash"] = fixture_records[0]["phash"]
        reset_fixture(rows=rows_with_phash)
        try:
            pipeline_module.load_split_inputs(fixture, "fit", class_order=class_order)
        except ValueError as error:
            if "duplicates pHash" not in str(error):
                raise
            mutations["split_row_phash_duplicate"] = "rejected at production identity check"
        else:
            raise AssertionError("pipeline input loader accepted split-row pHash")

    return {
        "status": "pass",
        "actual_v7_split_counts": actual_counts,
        "actual_v7_rows_have_no_phash_field": True,
        "joined_phash_source": "source-manifest.records by exact image ID",
        "joined_id_phash_sha256": joined_metadata_hashes,
        "split_row_sha256_unchanged": split_file_hashes,
        "array_sha256_unchanged": array_file_hashes,
        "source_manifest_sha256_unchanged": manifest_before,
        "cross_field_identities_checked": [
            "label",
            "class_index against frozen class order",
            "raw_sha256",
            "decoded_pixel_sha256 against source pixel_sha256",
            "creator",
            "creator_id",
            "source_url",
            "array label against split class_index",
        ],
        "mutation_controls": mutations,
        "model_initialized": False,
        "optimizer_updates": 0,
    }


def run_stage(
    stage: str,
    candidate: ModuleType,
    specification: str,
    data_directory: Path | None = None,
    archive_path: Path | None = None,
    output_directory: Path | None = None,
    recipe_path: Path | None = None,
) -> dict:
    candidate_path = Path(candidate.__file__).resolve()
    if stage == "model":
        return _model_stage(candidate)
    if stage == "objectives":
        objective = load_objectives(candidate_path, specification)
        return _objective_stage(candidate, objective)
    if stage == "augmentation":
        augmentation = load_augmentation(candidate_path, specification)
        return _augmentation_stage(augmentation)
    if stage == "evaluation":
        evaluation = load_evaluation(candidate_path, specification)
        return _evaluation_stage(evaluation)
    if stage == "data":
        if data_directory is None or archive_path is None or recipe_path is None:
            raise ValueError("the data stage requires --data, --archive, and --recipe")
        verifier_path = ROOT / "data" / "verify.py"
        verify_spec = importlib.util.spec_from_file_location(
            "image_data_verifier", verifier_path
        )
        if verify_spec is None or verify_spec.loader is None:
            raise ImportError("could not load source-bound data verifier")
        verifier = importlib.util.module_from_spec(verify_spec)
        verify_spec.loader.exec_module(verifier)
        report = verifier.verify_data(
            data_directory,
            recipe_path=recipe_path,
            archive_path=archive_path,
        )
        with tempfile.TemporaryDirectory(prefix="c03-data-positive-") as temp:
            unchanged = Path(temp) / "unchanged"
            shutil.copytree(data_directory, unchanged)
            unchanged_report = verifier.verify_data(
                unchanged, recipe_path=recipe_path, archive_path=None
            )
            if (
                unchanged_report["source_manifest_sha256"]
                != report["source_manifest_sha256"]
            ):
                raise AssertionError(
                    "unchanged copied-data control differed under mutation arguments"
                )
        with tempfile.TemporaryDirectory(prefix="c03-data-negative-") as temp:
            forged = Path(temp) / "data"
            shutil.copytree(data_directory, forged)
            labels_path = forged / "arrays" / "fit_labels.npy"
            labels = np.load(labels_path, allow_pickle=False).copy()
            labels[0] = (int(labels[0]) + 1) % 5
            np.save(labels_path, labels, allow_pickle=False)
            manifest_path = forged / "source-manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["prepared_array_files"]["arrays/fit_labels.npy"]["sha256"] = (
                hashlib.sha256(labels_path.read_bytes()).hexdigest()
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            try:
                verifier.verify_data(forged, recipe_path=recipe_path, archive_path=None)
            except ValueError as error:
                if "prepared source manifest differs from recipe" not in str(error):
                    raise AssertionError(
                        "label mutation was rejected before the source-bound manifest check"
                    ) from error
                forged_array_rejected = True
            else:
                forged_array_rejected = False
        if not forged_array_rejected:
            raise AssertionError(
                "source-bound data verifier accepted changed labels and a recomputed manifest digest"
            )

        row_negative_controls = {}
        with tempfile.TemporaryDirectory(prefix="c03-row-negative-") as row_temp:
            for name in ("same_class_row_swap", "forged_creator_metadata"):
                forged = Path(row_temp) / name
                shutil.copytree(data_directory, forged)
                row_path = forged / "splits" / "fit.json"
                rows = json.loads(row_path.read_text())
                if name == "same_class_row_swap":
                    left = next(
                        index
                        for index, row in enumerate(rows)
                        if row["class_index"] == rows[0]["class_index"] and index > 0
                    )
                    rows[0], rows[left] = rows[left], rows[0]
                else:
                    rows[0]["creator"] = "Forged creator label"
                    rows[0]["creator_id"] = "forged-creator-id"
                row_path.write_text(
                    json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n"
                )
                manifest_path = forged / "source-manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["prepared_split_rows"]["splits/fit.json"]["sha256"] = (
                    hashlib.sha256(row_path.read_bytes()).hexdigest()
                )
                manifest["prepared_split_rows"]["splits/fit.json"][
                    "bytes"
                ] = row_path.stat().st_size
                manifest_path.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n"
                )
                try:
                    verifier.verify_data(
                        forged, recipe_path=recipe_path, archive_path=None
                    )
                except ValueError as error:
                    if "prepared source manifest differs from recipe" not in str(error):
                        raise AssertionError(
                            f"{name} did not reach the trusted source-manifest binding"
                        ) from error
                    row_negative_controls[name] = (
                        "rejected by trusted manifest after local row hash recomputation"
                    )
                else:
                    raise AssertionError(f"source-bound verifier accepted {name}")
        pipeline_input_result = _pipeline_input_loader_stage(
            data_directory, recipe_path
        )
        return {
            **report,
            "unchanged_copy_positive_control": "passed with the same recipe and no archive argument as mutations",
            "changed_label_and_manifest_mutation": "rejected by frozen recipe hash",
            "split_row_identity_mutations": row_negative_controls,
            "pipeline_input_loader": pipeline_input_result,
        }
    if stage == "resume":
        training = load_training(candidate_path, specification)
        return _resume_stage(training, candidate_path)
    if stage == "consumer":
        return _consumer_stage(candidate, candidate_path)
    if stage == "runner":
        return _runner_stage(load_runner(candidate_path, specification))
    if stage == "composition":
        return _composition_stage()
    if stage == "graph":
        if output_directory is None:
            raise ValueError("the complete graph stage requires --output")
        if recipe_path is None:
            raise ValueError("the complete graph stage requires --recipe")
        return _graph_stage(
            candidate_path,
            output_directory.expanduser().resolve(),
            recipe_path.expanduser().resolve(),
        )
    if stage == "report":
        report_path = ROOT / "solution" / "model_card.py"
        report_spec = importlib.util.spec_from_file_location(
            "image_model_card", report_path
        )
        if report_spec is None or report_spec.loader is None:
            raise ImportError("could not load measured model card renderer")
        report_module = importlib.util.module_from_spec(report_spec)
        report_spec.loader.exec_module(report_module)
        return _report_stage(report_module)
    raise ValueError(
        f"stage {stage!r} is not implemented yet; it is never silently skipped"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation",
        required=True,
        help="solution, starter, or a candidate directory/model.py",
    )
    parser.add_argument(
        "--stage",
        choices=(
            "model",
            "objectives",
            "augmentation",
            "data",
            "resume",
            "runner",
            "composition",
            "graph",
            "evaluation",
            "consumer",
            "report",
            "all",
        ),
        required=True,
    )
    parser.add_argument("--data", type=Path, help="external prepared dataset directory")
    parser.add_argument(
        "--recipe",
        type=Path,
        default=ROOT / "data" / "recipe-review-v10.json",
        help="versioned source/optimization recipe (v10 by default)",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="pinned raw flower archive required for the data stage",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="external evidence output directory"
    )
    args = parser.parse_args()
    candidate_path = None
    identity = args.implementation
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "checker": "image-model-300m/tests/check.py",
        "implementation": identity,
        "stage": args.stage,
        "status": "failed",
        "stages": {},
    }
    try:
        candidate, candidate_path, identity = load_candidate(args.implementation)
        report["implementation"] = identity
        stages = (
            (
                "model",
                "objectives",
                "augmentation",
                "data",
                "resume",
                "runner",
                "composition",
                "graph",
                "evaluation",
                "consumer",
                "report",
            )
            if args.stage == "all"
            else (args.stage,)
        )
        for stage in stages:
            report["stages"][stage] = run_stage(
                stage,
                candidate,
                args.implementation,
                args.data,
                args.archive,
                output,
                args.recipe.expanduser().resolve(),
            )
        report["status"] = "pass"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    report_path = output / "check-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
