"""Offline local Hugging Face Hub mixin loader for the custom JAX package."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import ModelHubMixin, snapshot_download
from safetensors.numpy import load_file, save_file
from tokenizers import Tokenizer

try:
    from . import model
    from .config import EncoderConfig
except ImportError:
    import model
    from config import EncoderConfig


def _path_name(path):
    parts = []
    for key in path:
        if hasattr(key, "key"):
            parts.append(str(key.key))
        elif hasattr(key, "idx"):
            parts.append(str(key.idx))
        else:
            parts.append(str(key))
    return ".".join(parts)


def _parameter_arrays(params):
    paths, leaves = zip(*jax.tree_util.tree_flatten_with_path(params)[0])
    names = [_path_name(path) for path in paths]
    if len(set(names)) != len(names):
        raise ValueError("parameter path collision during package export")
    return {f"tensor_{i:04d}": np.asarray(leaf) for i, leaf in enumerate(leaves)}, names


def _load_candidate_model(directory, expected_sha256):
    source = Path(directory) / "model.py"
    raw = source.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError("package candidate implementation checksum mismatch")
    sys.path.insert(0, str(Path(directory).resolve()))
    module_name = f"jaxpathways_packaged_model_{actual[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _save_directory(encoder, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = encoder.config.as_dict()
    (directory / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    config_hash = hashlib.sha256((directory / "config.json").read_bytes()).hexdigest()
    encoder.tokenizer.save(str(directory / "tokenizer.json"))
    arrays, paths = _parameter_arrays(encoder.params)
    save_file(arrays, str(directory / "model.safetensors"))
    for filename in ("config.py", "package_runtime.py"):
        shutil.copyfile(
            Path(__file__).resolve().parent / filename, directory / filename
        )
    implementation = Path(encoder.implementation_path).resolve()
    shutil.copyfile(implementation, directory / "model.py")
    code_hashes = {
        filename: hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        for filename in ("model.py", "config.py", "package_runtime.py")
    }
    metadata = {
        "schema": 1,
        "library": "jaxpathways-scifact-encoder",
        "pooling": "mean final hidden state over non-special content tokens; L2 normalize; zero for empty content",
        "special_ids_excluded_from_pooling": [
            config[k] for k in ("pad_id", "bos_id", "eos_id", "unk_id", "mask_id")
        ],
        "max_length": config["max_length"],
        "config_sha256": config_hash,
        "text_normalization": "none; UTF-8 text enters byte-level BPE",
        "query_document_prefixes": "none",
        "parameter_paths": paths,
        "parameter_tree_sha256": hashlib.sha256(
            json.dumps(paths, separators=(",", ":")).encode()
        ).hexdigest(),
        "model_file_sha256": hashlib.sha256(
            (directory / "model.safetensors").read_bytes()
        ).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(
            (directory / "tokenizer.json").read_bytes()
        ).hexdigest(),
        "source_files_sha256": code_hashes,
        "candidate_source_path_at_export": str(implementation),
    }
    (directory / "package.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def _load_directory(directory):
    directory = Path(directory)
    if not (directory / "model.safetensors").is_file():
        raise ValueError("package is missing model.safetensors")
    config = EncoderConfig(**json.loads((directory / "config.json").read_text()))
    metadata = json.loads((directory / "package.json").read_text())
    code_hashes = metadata.get("source_files_sha256", {})
    if set(code_hashes) != {"model.py", "config.py", "package_runtime.py"}:
        raise ValueError("package source-code inventory is incomplete")
    for filename, expected_hash in code_hashes.items():
        actual_hash = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"package source checksum mismatch: {filename}")
    packaged_model = _load_candidate_model(directory, code_hashes["model.py"])
    raw_weights = (directory / "model.safetensors").read_bytes()
    raw_tokenizer = (directory / "tokenizer.json").read_bytes()
    if hashlib.sha256(raw_weights).hexdigest() != metadata.get("model_file_sha256"):
        raise ValueError("package model weights checksum mismatch")
    if hashlib.sha256(raw_tokenizer).hexdigest() != metadata.get("tokenizer_sha256"):
        raise ValueError("package tokenizer checksum mismatch")
    if hashlib.sha256(
        (directory / "config.json").read_bytes()
    ).hexdigest() != metadata.get("config_sha256"):
        raise ValueError("package configuration checksum mismatch")
    if (
        metadata.get("pooling")
        != "mean final hidden state over non-special content tokens; L2 normalize; zero for empty content"
    ):
        raise ValueError("package pooling policy is unsupported")
    if metadata.get("max_length") != config.max_length:
        raise ValueError("package maximum length disagrees with model config")
    params = packaged_model.initialize(jax.random.key(0), config)
    entries, paths = jax.tree_util.tree_flatten_with_path(params)[0], metadata.get(
        "parameter_paths"
    )
    current_paths = [_path_name(path) for path, _ in entries]
    if current_paths != paths:
        raise ValueError("package parameter-tree paths do not match its config")
    if (
        metadata.get("parameter_tree_sha256")
        != hashlib.sha256(json.dumps(paths, separators=(",", ":")).encode()).hexdigest()
    ):
        raise ValueError("package parameter-tree identity mismatch")
    archive = load_file(str(directory / "model.safetensors"))
    if len(archive) != len(entries):
        raise ValueError("package tensor count mismatch")
    leaves = []
    for index, (_, template) in enumerate(entries):
        array = archive[f"tensor_{index:04d}"]
        if array.shape != template.shape or array.dtype != np.dtype(template.dtype):
            raise ValueError("package parameter shape or dtype mismatch")
        leaves.append(jnp.asarray(array))
    params = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(params), leaves)
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("package tokenizer/config vocabulary mismatch")
    return params, config, tokenizer, metadata, packaged_model


class JAXSentenceEncoder(
    ModelHubMixin,
    library_name="jaxpathways-scifact-encoder",
    pipeline_tag="sentence-similarity",
    tags=["jax", "sentence-embeddings", "retrieval", "in-development"],
):
    """Custom JAX package API; not a Transformers or Sentence Transformers class."""

    def __init__(
        self,
        params,
        config,
        tokenizer,
        package_metadata=None,
        model_module=None,
        implementation_path=None,
    ):
        super().__init__()
        self.params = params
        self.config = (
            config if isinstance(config, EncoderConfig) else EncoderConfig(**config)
        )
        self.tokenizer = tokenizer
        self.package_metadata = package_metadata or {}
        self.model_module = model_module or model
        self.implementation_path = str(
            implementation_path or Path(__file__).resolve().parent / "model.py"
        )

    def encode(self, texts, *, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            ids = []
            for text in batch:
                encoded = self.tokenizer.encode(text, add_special_tokens=False).ids
                values = [self.config.bos_id, *map(int, encoded), self.config.eos_id]
                if len(values) > self.config.max_length:
                    values = values[: self.config.max_length - 1] + [self.config.eos_id]
                ids.append(np.asarray(values, np.int32))
            width = max(map(len, ids), default=1)
            padded = np.full((len(ids), width), self.config.pad_id, np.int32)
            for row, values in enumerate(ids):
                padded[row, : len(values)] = values
            vectors = self.model_module.pool(
                self.params, jnp.asarray(padded), self.config
            )
            outputs.extend(np.asarray(vectors, np.float32))
        return np.asarray(outputs, dtype=np.float32).reshape(
            len(texts), self.config.width
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        _save_directory(self, save_directory)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: str,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token=None,
        **model_kwargs,
    ):
        location = Path(model_id)
        if not location.is_dir():
            if local_files_only:
                raise FileNotFoundError(
                    "offline package load requires a local package directory"
                )
            location = Path(
                snapshot_download(
                    repo_id=model_id,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=False,
                    token=token,
                )
            )
        params, config, tokenizer, metadata, packaged_model = _load_directory(location)
        return cls(
            params, config, tokenizer, metadata, packaged_model, location / "model.py"
        )

    @classmethod
    def from_local_directory(cls, directory):
        return cls._from_pretrained(str(directory), local_files_only=True)


def save_package(
    params,
    config,
    tokenizer,
    directory,
    *,
    model_module=None,
    implementation_path=None,
):
    """Write package files using the same path used by ModelHubMixin."""
    encoder = JAXSentenceEncoder(
        params,
        config,
        tokenizer,
        model_module=model_module,
        implementation_path=implementation_path,
    )
    encoder.save_pretrained(directory)
    return encoder
