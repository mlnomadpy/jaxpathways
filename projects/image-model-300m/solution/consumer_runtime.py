"""Offline local-bundle runtime copied into each private consumer package."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


class LocalImageClassifier:
    """Load a checksummed local model/processor package without network access."""

    def __init__(self, root: Path, model_module, processor_module, params, head):
        self.root = root
        self.model = model_module
        self.processor = processor_module
        self.params = params
        self.head = head

    @classmethod
    def from_pretrained(cls, directory: str | Path, *, local_files_only: bool = True):
        if local_files_only is not True:
            raise ValueError("the local image consumer never downloads files")
        root = Path(directory).expanduser().resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("schema") != "jaxpathways.local-image-consumer.v1":
            raise ValueError("unsupported local consumer manifest")
        expected = manifest.get("files")
        if not isinstance(expected, dict) or set(expected) != {
            "model.py",
            "processor.py",
            "consumer_runtime.py",
            "config.json",
            "weights.npz",
        }:
            raise ValueError("consumer package file inventory is incomplete")
        for name, digest in expected.items():
            path = root / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"consumer package file checksum mismatch: {name}")
        model_module = _load_module(root / "model.py", "offline_candidate_model")
        processor_module = _load_module(root / "processor.py", "offline_image_processor")
        config = json.loads((root / "config.json").read_text())
        expected_processor = {
            "input_formats": ["JPEG", "PNG"],
            "resize": "EXIF transpose, RGB, aspect-preserving Lanczos letterbox on RGB 127 gray to configured square size",
            "normalization": "float32 uint8/127.5 - 1",
            "icc_profile_applied": False,
        }
        if config.get("schema") != "jaxpathways.local-image-model.v1":
            raise ValueError("unsupported local image model config")
        if config.get("pooling") != "pre-LN ViT CLS token, linear classifier, no test-time augmentation":
            raise ValueError("consumer pooling contract differs from the checked local classifier")
        if config.get("processor") != expected_processor:
            raise ValueError("consumer processor contract differs from the checked preprocessing policy")
        model_config = config["model_config"]
        if int(model_config.get("image_size", 0)) <= 0 or int(model_config.get("channels", 0)) != 3:
            raise ValueError("consumer model image dimensions are invalid")
        classes = config.get("classes")
        if not isinstance(classes, list) or len(classes) != int(model_config["classes"]):
            raise ValueError("consumer class names differ from the model output count")
        params = model_module.init_vit(jax.random.PRNGKey(0), model_config)
        trainable_template = {key: value for key, value in params.items() if key != "config"}
        leaves, tree = jax.tree_util.tree_flatten(trainable_template)
        with np.load(root / "weights.npz", allow_pickle=False) as archive:
            expected_arrays = {f"encoder_{index:05d}" for index in range(len(leaves))} | {"head_kernel", "head_bias"}
            if set(archive.files) != expected_arrays:
                raise ValueError("consumer parameter leaf inventory differs from the declared model")
            restored = []
            for index, template in enumerate(leaves):
                value = archive[f"encoder_{index:05d}"]
                template_array = np.asarray(template)
                if value.shape != template_array.shape or value.dtype != template_array.dtype:
                    raise ValueError(f"consumer encoder leaf {index} shape or dtype differs")
                if not np.isfinite(value).all():
                    raise ValueError(f"consumer encoder leaf {index} is nonfinite")
                restored.append(jnp.asarray(value))
            params = {**jax.tree_util.tree_unflatten(tree, restored), "config": model_config}
            head = {"kernel": jnp.asarray(archive["head_kernel"]), "bias": jnp.asarray(archive["head_bias"])}
        if head["kernel"].shape != (int(model_config["width"]), int(model_config["classes"])):
            raise ValueError("consumer classifier head shape differs from model config")
        if head["bias"].shape != (int(model_config["classes"]),):
            raise ValueError("consumer classifier bias shape differs from model config")
        if not np.isfinite(np.asarray(head["kernel"])).all() or not np.isfinite(np.asarray(head["bias"])).all():
            raise ValueError("consumer classifier head contains nonfinite values")
        return cls(root, model_module, processor_module, params, head)

    def predict_bytes(self, raw: bytes) -> dict:
        pixels, source = self.processor.preprocess_bytes(raw, int(self.params["config"]["image_size"]))
        normalized = self.processor.normalize_uint8(pixels[None, ...])
        logits = self.model.classification_logits(self.params, self.head, jnp.asarray(normalized))
        values = np.asarray(logits, dtype=np.float64)
        if values.shape != (1, int(self.params["config"]["classes"])) or not np.isfinite(values).all():
            raise ValueError("consumer model returned invalid logits")
        shifted = values[0] - np.max(values[0])
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        return {
            "logits": values[0].tolist(),
            "probabilities": probabilities.tolist(),
            "class_index": int(np.argmax(probabilities)),
            "processor": source,
        }


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load local package module {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
