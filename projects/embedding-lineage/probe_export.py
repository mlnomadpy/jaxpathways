"""Fresh-process model package loader used by the stage handoff check."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
import numpy as np
import jax.numpy as jnp
from flax import serialization
from network import ROOT, SPECIAL, BidirectionalEncoder, make_tokenizer, token_words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    export = Path(args.export)
    manifest = json.loads((export / "manifest.json").read_text())
    source = manifest.get("source_code_identity", {}).get("files", {})
    current = {
        "probe_export.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "network.py": hashlib.sha256((ROOT / "network.py").read_bytes()).hexdigest(),
    }
    if any(source.get(name) != digest for name, digest in current.items()):
        raise ValueError("export loader source-code identity mismatch")
    for name, digest in manifest["files"].items():
        if hashlib.sha256((export / name).read_bytes()).hexdigest() != digest:
            raise ValueError("export file checksum mismatch")
    config = json.loads((export / "config.json").read_text())
    tokenizer = json.loads((export / "tokenizer.json").read_text())
    vocab = tokenizer["vocabulary"]
    words = token_words(args.text)
    ids = (
        [SPECIAL["<bos>"]]
        + [vocab.get(word, SPECIAL["<unk>"]) for word in words]
        + [SPECIAL["<eos>"]]
    )
    tokens = np.full((1, config["max_length"]), SPECIAL["<pad>"], np.int32)
    ids = ids[: config["max_length"]]
    tokens[0, : len(ids)] = ids
    model = BidirectionalEncoder(
        tokenizer["vocabulary_size"],
        config["max_length"],
        config["width"],
        config["attention_heads"],
        config["mlp_width"],
    )
    template = model.init(jax.random.PRNGKey(config["seed"]), jnp.asarray(tokens))[
        "params"
    ]
    params = serialization.from_bytes(
        template, (export / "params.msgpack").read_bytes()
    )
    vector, _ = model.apply({"params": params}, jnp.asarray(tokens))
    vector = np.asarray(vector, dtype=np.float32)
    print(
        json.dumps(
            {
                "embedding_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
                "shape": list(vector.shape),
                "norm": float(np.linalg.norm(vector)),
            }
        )
    )


if __name__ == "__main__":
    main()
