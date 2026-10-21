"""Small offline inference API for an exported bounded C02 pilot checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from threading import Event

import jax
import jax.numpy as jnp
import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import model
from config import ModelConfig


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        HERE / "config.py",
        HERE / "model.py",
        HERE / "training.py",
        HERE / "run_pilot.py",
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _restore(spec, archive):
    if spec is None or isinstance(spec, (bool, int, float, str)):
        return spec
    if isinstance(spec, list):
        return [_restore(item, archive) for item in spec]
    if "array" in spec:
        value = np.asarray(archive[spec["array"]])
        if list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"]:
            raise ValueError("consumer parameter array differs from its package schema")
        return jnp.asarray(value)
    if "tuple" in spec:
        return tuple(_restore(item, archive) for item in spec["tuple"])
    return {key: _restore(value, archive) for key, value in spec.items()}


class LocalCausalModel:
    """Loaded tokenizer and weights with a bounded greedy generation method."""

    def __init__(
        self, package_dir: Path, manifest: dict, config: ModelConfig, params, tokenizer
    ):
        self.package_dir = package_dir
        self.manifest = manifest
        self.config = config
        self.params = params
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        cancel_event: Event | None = None,
    ) -> dict[str, object]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if type(max_new_tokens) is not int or not 0 <= max_new_tokens <= 32:
            raise ValueError("max_new_tokens must be an integer from 0 through 32")
        ids = [1, *self.tokenizer.encode(prompt, add_special_tokens=False).ids]
        if len(ids) + max_new_tokens > self.config.context:
            raise ValueError("prompt plus generation budget exceeds model context")
        if cancel_event is None:
            generated = model.generate(
                self.params, ids, self.config, max_new_tokens, eos_token_id=2
            )
        else:
            generated = self._generate_cancellable(ids, max_new_tokens, cancel_event)
        return {
            "text": self.tokenizer.decode(generated, skip_special_tokens=True),
            "token_ids": generated,
            "finish_reason": "eos" if generated and generated[-1] == 2 else "length",
            "prompt_tokens": len(ids),
        }

    def _generate_cancellable(self, ids: list[int], maximum: int, cancel_event: Event):
        next_logits, cache = model.prefill(self.params, ids, self.config)
        generated = []
        for _ in range(maximum):
            if cancel_event.is_set():
                raise RuntimeError("generation cancelled or request deadline reached")
            choices = next_logits.at[:, 0].set(-jnp.inf)
            token = int(jnp.argmax(choices[0]))
            generated.append(token)
            if token == 2:
                break
            if cache["position"] >= self.config.context:
                if len(generated) < maximum:
                    raise ValueError(
                        "generation request exceeds absolute context capacity"
                    )
                break
            next_logits, cache = model.decode_step(
                self.params, token, cache, self.config
            )
        return generated


def load_package(package_dir: str | Path) -> LocalCausalModel:
    """Verify every packaged byte before constructing a model for inference."""
    root = Path(package_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "c02-custom-consumer-v1":
        raise ValueError("unsupported consumer package format")
    required_files = {
        "README.md",
        "RIGHTS.md",
        "config.py",
        "config.json",
        "consumer.py",
        "model.py",
        "requirements.txt",
        "parameters.json",
        "parameters.npz",
        "run_pilot.py",
        "serve_local.py",
        "training.py",
        "tokenizer.json",
    }
    file_hashes = manifest.get("files")
    if not isinstance(file_hashes, dict) or set(file_hashes) != required_files:
        raise ValueError("consumer package manifest does not cover the required files")
    for relative, expected in file_hashes.items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or _sha256(path) != expected:
            raise ValueError(
                f"consumer package file failed integrity check: {relative}"
            )
    if manifest.get("candidate_source_sha256") != _candidate_hash():
        raise ValueError("consumer decoder source differs from the exported candidate")
    config = ModelConfig(**json.loads((root / "config.json").read_text()))
    config.validate()
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("consumer tokenizer vocabulary differs from model config")
    params_spec = json.loads((root / "parameters.json").read_text())
    with np.load(root / "parameters.npz", allow_pickle=False) as archive:
        params = _restore(params_spec, archive)
        if set(archive.files) != set(manifest["parameter_arrays"]):
            raise ValueError("consumer package contains unlisted parameter arrays")
        for name in archive.files:
            if not np.isfinite(archive[name]).all():
                raise ValueError("consumer package contains nonfinite model parameters")
    expected_params = model.initialize(jax.random.PRNGKey(0), config)
    if jax.tree.structure(params) != jax.tree.structure(expected_params):
        raise ValueError(
            "consumer parameter tree differs from the declared architecture"
        )
    for actual, expected in zip(
        jax.tree.leaves(params), jax.tree.leaves(expected_params), strict=True
    ):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(
                "consumer parameter shape/dtype differs from the architecture"
            )
    return LocalCausalModel(root, manifest, config, params, tokenizer)
