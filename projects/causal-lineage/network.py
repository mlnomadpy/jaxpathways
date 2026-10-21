"""Data contracts and a deliberately small causal Transformer decoder."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

ROOT = Path(__file__).resolve().parent
SPECIAL = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "<sep>": 4}
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\W\s]", re.UNICODE)


class CausalDecoder(nn.Module):
    """One pre-LN causal-attention block followed by a vocabulary projection."""

    vocab_size: int
    max_length: int
    width: int
    heads: int
    mlp_width: int

    @nn.compact
    def __call__(self, token_ids, attention_mask):
        positions = jnp.arange(token_ids.shape[1])
        x = nn.Embed(self.vocab_size, self.width, name="token_embedding")(token_ids)
        position = self.param(
            "position_embedding",
            nn.initializers.normal(0.02),
            (self.max_length, self.width),
        )
        x = x + jnp.asarray(position)[positions][None, :, :]
        normalized = nn.LayerNorm(name="attention_norm")(x)
        attended = nn.SelfAttention(
            num_heads=self.heads,
            qkv_features=self.width,
            out_features=self.width,
            dropout_rate=0.0,
            name="causal_attention",
        )(normalized, mask=attention_mask)
        x = x + attended
        normalized = nn.LayerNorm(name="mlp_norm")(x)
        hidden = nn.Dense(self.mlp_width, name="mlp_in")(normalized)
        hidden = nn.gelu(hidden)
        hidden = nn.Dense(self.width, name="mlp_out")(hidden)
        x = x + hidden
        return nn.Dense(self.vocab_size, name="language_model_head")(x)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical_hash(value) -> str:
    packed = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_data() -> tuple[list[dict], list[dict], dict, str]:
    source = ROOT / "data/source/alice-pg11.txt"
    corpus_path = ROOT / "data/corpus.jsonl"
    instruction_path = ROOT / "data/instructions.jsonl"
    manifest_path = ROOT / "data/manifest.json"
    config_path = ROOT / "config.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_sha = sha256_bytes(source.read_bytes())
    if source_sha != manifest["source_sha256"]:
        raise ValueError("pinned Gutenberg source checksum mismatch")
    for name, path in (
        ("corpus.jsonl", corpus_path),
        ("instructions.jsonl", instruction_path),
    ):
        if sha256_bytes(path.read_bytes()) != manifest["data_files"][name]:
            raise ValueError(f"derived data checksum mismatch: {name}")
    passages = read_jsonl(corpus_path)
    instructions = read_jsonl(instruction_path)
    groups: dict[str, set[str]] = {}
    for passage in passages:
        if passage["source_sha256"] != source_sha:
            raise ValueError("passage does not match pinned Gutenberg bytes")
        groups.setdefault(passage["group"], set()).add(passage["split"])
        expected_split = manifest["split_map"].get(passage["chapter"])
        if expected_split != passage["split"]:
            raise ValueError("passage chapter is assigned to the wrong split")
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("a source chapter crosses train/validation/test")
    if len({row["id"] for row in passages}) != len(passages):
        raise ValueError("duplicate passage ID")
    if any(
        row["split"] != "train"
        or manifest["split_map"].get(row["chapter"]) != "train"
        or row["source_sha256"] != source_sha
        for row in instructions
    ):
        raise ValueError("instruction adaptation contains a held-out or stale record")
    if len({row["id"] for row in instructions}) != len(instructions):
        raise ValueError("duplicate instruction record ID")
    digest = hashlib.sha256()
    for path in (source, corpus_path, instruction_path, manifest_path):
        digest.update(path.name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    digest.update(canonical_hash(config).encode("ascii"))
    return passages, instructions, manifest, digest.hexdigest()


def token_words(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def make_tokenizer(
    passages: list[dict], instructions: list[dict], max_vocabulary: int
) -> dict:
    counts = Counter(
        word
        for passage in passages
        if passage["split"] == "train"
        for word in token_words(passage["text"])
    )
    for record in instructions:
        for field in ("instruction", "context", "response"):
            counts.update(token_words(record[field]))
    remaining = sorted(counts, key=lambda word: (-counts[word], word))[
        : max_vocabulary - len(SPECIAL)
    ]
    vocabulary = {
        **SPECIAL,
        **{word: index + len(SPECIAL) for index, word in enumerate(remaining)},
    }
    return {
        "name": "lowercase-regex-word-v1",
        "vocabulary": vocabulary,
        "vocabulary_size": len(vocabulary),
        "special_tokens": SPECIAL,
        "max_vocabulary": max_vocabulary,
        "fit_splits": ["train"],
    }


def encode_text(text: str, tokenizer: dict) -> list[int]:
    vocabulary = tokenizer["vocabulary"]
    return [vocabulary.get(word, SPECIAL["<unk>"]) for word in token_words(text)]


def make_model(config: dict, tokenizer: dict) -> CausalDecoder:
    if config["layers"] != 1:
        raise ValueError(
            "this teaching decoder implements exactly one Transformer block"
        )
    return CausalDecoder(
        vocab_size=tokenizer["vocabulary_size"],
        max_length=config["max_length"],
        width=config["width"],
        heads=config["attention_heads"],
        mlp_width=config["mlp_width"],
    )


def init_params(model: CausalDecoder, seed: int, max_length: int, mask_fn):
    sample = jnp.full((1, max_length), SPECIAL["<bos>"], dtype=jnp.int32)
    return model.init(jax.random.key(seed), sample, mask_fn(sample, SPECIAL["<pad>"]))[
        "params"
    ]


def parameter_count(params) -> int:
    return sum(int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(params))


def tree_bytes(params) -> bytes:
    return serialization.to_bytes(params)


def load_params(template, payload: bytes):
    return serialization.from_bytes(template, payload)
