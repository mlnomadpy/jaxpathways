"""Pinned external encoder baseline; never substitutes for random initialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MAX_LENGTH = 256


def load_pinned_baseline(snapshot_dir: Path, *, offline=False):
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.exists() or not any(snapshot_dir.iterdir()):
        if offline:
            raise FileNotFoundError(
                "pinned public encoder baseline is absent in offline mode"
            )
        snapshot_download(
            MODEL_ID,
            revision=REVISION,
            local_dir=snapshot_dir,
            local_files_only=False,
            allow_patterns=[
                "README.md",
                "config.json",
                "config_sentence_transformers.json",
                "modules.json",
                "sentence_bert_config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.txt",
                "1_Pooling/config.json",
                "model.safetensors",
            ],
        )
    if not (snapshot_dir / "model.safetensors").is_file():
        raise FileNotFoundError("pinned baseline snapshot is missing model.safetensors")
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(
        str(snapshot_dir), device="cpu", local_files_only=True
    )
    encoder.max_seq_length = MAX_LENGTH
    frozen = json.loads(
        (Path(__file__).resolve().parent / "data" / "frozen_recipe.json").read_text()
    )
    expected = frozen["baseline"]
    weights_sha256 = hashlib.sha256(
        (snapshot_dir / "model.safetensors").read_bytes()
    ).hexdigest()
    tokenizer_sha256 = hashlib.sha256(
        (snapshot_dir / "tokenizer.json").read_bytes()
    ).hexdigest()
    if weights_sha256 != expected["weights_sha256"]:
        raise ValueError(
            "external baseline weight bytes differ from the frozen revision"
        )
    if tokenizer_sha256 != expected["tokenizer_sha256"]:
        raise ValueError("external baseline tokenizer differs from the frozen revision")
    if encoder.max_seq_length != expected["max_length"]:
        raise ValueError("external baseline maximum length differs from frozen recipe")
    pooling_config = json.loads(
        (snapshot_dir / "1_Pooling" / "config.json").read_text()
    )
    if pooling_config != expected["pooling_config"]:
        raise ValueError("external baseline pooling differs from frozen recipe")
    files = {}
    for path in sorted(snapshot_dir.rglob("*")):
        if path.is_file() and ".cache" not in path.parts:
            files[str(path.relative_to(snapshot_dir))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    metadata = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "reported_license": "Apache-2.0",
        "maximum_length": MAX_LENGTH,
        "dtype": "float32",
        "query_document_prefixes": "none",
        "module_pipeline": json.loads((snapshot_dir / "modules.json").read_text()),
        "pooling_config": json.loads(
            (snapshot_dir / "1_Pooling/config.json").read_text()
        ),
        "weights_sha256": weights_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "snapshot_files_sha256": files,
        "possible_overlap": "upstream model metadata includes S2ORC and MS MARCO among training data; overlap with SciFact is unresolved",
    }
    return encoder, metadata


def encode(encoder, texts, *, batch_size=32):
    result = encoder.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    values = np.asarray(result, np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("pinned baseline produced malformed/nonfinite embeddings")
    return values
