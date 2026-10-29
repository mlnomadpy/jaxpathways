"""Export a private, checksummed local raw-image classifier bundle."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import jax
import numpy as np


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_local_bundle(
    destination: str | Path,
    *,
    model_module,
    model_source: str | Path,
    processor_source: str | Path,
    encoder_params: dict[str, Any],
    head: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Write an offline raw-image bundle without copying source photographs.

    The bundle is a private evaluation artifact. It makes no claim that
    trained weights have a standalone redistribution license.
    """
    target = Path(destination).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"consumer bundle destination already exists: {target}")
    if "config" not in encoder_params or "kernel" not in head or "bias" not in head:
        raise ValueError("consumer export requires encoder config and a linear classification head")
    template = model_module.init_vit(jax.random.PRNGKey(0), encoder_params["config"])
    template_leaves, _ = jax.tree_util.tree_flatten({key: value for key, value in template.items() if key != "config"})
    parameter_leaves, _ = jax.tree_util.tree_flatten(
        {key: value for key, value in encoder_params.items() if key != "config"}
    )
    if len(template_leaves) != len(parameter_leaves) or any(
        np.asarray(actual).shape != np.asarray(expected).shape or np.asarray(actual).dtype != np.asarray(expected).dtype
        for actual, expected in zip(parameter_leaves, template_leaves)
    ):
        raise ValueError("candidate encoder leaves differ from its own declared model template")
    target.mkdir(parents=True)
    try:
        model_path = target / "model.py"
        processor_path = target / "processor.py"
        runtime_path = target / "consumer_runtime.py"
        shutil.copyfile(model_source, model_path)
        shutil.copyfile(processor_source, processor_path)
        shutil.copyfile(Path(__file__).with_name("consumer_runtime.py"), runtime_path)
        trainable = {key: value for key, value in encoder_params.items() if key != "config"}
        leaves, _ = jax.tree_util.tree_flatten(trainable)
        arrays = {f"encoder_{index:05d}": np.asarray(leaf) for index, leaf in enumerate(leaves)}
        arrays["head_kernel"] = np.asarray(head["kernel"])
        arrays["head_bias"] = np.asarray(head["bias"])
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("consumer export refuses nonfinite parameters")
        weights_path = target / "weights.npz"
        with weights_path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        config_path = target / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "schema": "jaxpathways.local-image-model.v1",
                    "model_config": encoder_params["config"],
                    "classes": provenance.get("classes"),
                    "pooling": "pre-LN ViT CLS token, linear classifier, no test-time augmentation",
                    "processor": {
                        "input_formats": ["JPEG", "PNG"],
                        "resize": "EXIF transpose, RGB, aspect-preserving Lanczos letterbox on RGB 127 gray to configured square size",
                        "normalization": "float32 uint8/127.5 - 1",
                        "icc_profile_applied": False,
                    },
                    "provenance": provenance,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        files = {
            path.name: _sha(path) for path in (model_path, processor_path, runtime_path, config_path, weights_path)
        }
        manifest = {"schema": "jaxpathways.local-image-consumer.v1", "files": files}
        manifest_path = target / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return {
            "path": str(target),
            "manifest_sha256": _sha(manifest_path),
            "file_hashes": files,
            "parameter_count": int(sum(value.size for value in arrays.values())),
            "contains_source_images": False,
            "weight_rights_statement": provenance.get("weight_rights_statement"),
        }
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
