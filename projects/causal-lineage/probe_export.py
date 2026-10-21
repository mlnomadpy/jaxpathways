"""Reload and sample from an exported causal pilot in a fresh process."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import jax
import jax.numpy as jnp
import numpy as np

from network import (
    ROOT,
    SPECIAL,
    canonical_hash,
    encode_text,
    load_data,
    load_params,
    make_model,
    sha256_bytes,
)


def import_implementation(path: Path):
    spec = importlib.util.spec_from_file_location("causal_lineage_export_impl", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load the recorded sampling implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def decode(ids: list[int], tokenizer: dict) -> str:
    inverse = {index: token for token, index in tokenizer["vocabulary"].items()}
    words = []
    for token_id in ids:
        if token_id == SPECIAL["<eos>"]:
            break
        if token_id in (SPECIAL["<pad>"], SPECIAL["<bos>"], SPECIAL["<sep>"]):
            continue
        words.append(inverse.get(token_id, "<unknown>"))
    text = ""
    for word in words:
        if re_punctuation(word) and text:
            text += word
        else:
            text += (" " if text else "") + word
    return text


def re_punctuation(token: str) -> bool:
    return len(token) == 1 and not token.isalnum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True)
    parser.add_argument("--implementation", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--expected-logits", required=True)
    args = parser.parse_args()

    export = Path(args.export).resolve()
    manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["source_code_identity"]["files"].items():
        path = ROOT / name
        if not path.is_file() or sha256_bytes(path.read_bytes()) != digest:
            raise ValueError(f"export source identity mismatch: {name}")
    candidate_path = Path(args.implementation).resolve()
    recorded_implementation = (ROOT / manifest["implementation_path"]).resolve()
    if candidate_path != recorded_implementation:
        raise ValueError("sampling implementation path differs from export receipt")
    candidate = import_implementation(candidate_path)

    config = json.loads((export / "config.json").read_text(encoding="utf-8"))
    tokenizer = json.loads((export / "tokenizer.json").read_text(encoding="utf-8"))
    passages, instructions, _, data_sha = load_data()
    if data_sha != manifest["data_sha256"]:
        raise ValueError("export data identity mismatch")
    if canonical_hash(config) != manifest["config_sha256"]:
        raise ValueError("export configuration identity mismatch")
    if canonical_hash(tokenizer) != manifest["tokenizer_sha256"]:
        raise ValueError("export tokenizer identity mismatch")
    if any(
        not (export / name).is_file()
        or sha256_bytes((export / name).read_bytes()) != digest
        for name, digest in manifest["files"].items()
    ):
        raise ValueError("export artifact checksum mismatch")
    if (
        sha256_bytes((export / "params.msgpack").read_bytes())
        != manifest["params_sha256"]
    ):
        raise ValueError("export parameter checksum mismatch")

    model = make_model(config, tokenizer)
    sample = jnp.full((1, config["max_length"]), SPECIAL["<bos>"], dtype=jnp.int32)
    template = model.init(
        jax.random.key(config["seed"]),
        sample,
        candidate.causal_attention_mask(sample, SPECIAL["<pad>"]),
    )["params"]
    params = load_params(template, (export / "params.msgpack").read_bytes())

    prompt_ids = [SPECIAL["<bos>"]] + encode_text(args.prompt, tokenizer)
    prompt_ids = prompt_ids[-config["max_length"] :]

    @jax.jit
    def forward(token_ids):
        attention = candidate.causal_attention_mask(token_ids, SPECIAL["<pad>"])
        return model.apply({"params": params}, token_ids, attention)

    padded = np.full((1, config["max_length"]), SPECIAL["<pad>"], np.int32)
    padded[0, : len(prompt_ids)] = prompt_ids
    prompt_logits = np.asarray(forward(jnp.asarray(padded))[0, len(prompt_ids) - 1])
    prompt_logits = np.asarray(prompt_logits, dtype="<f4")
    expected_logits = np.load(args.expected_logits)
    if expected_logits.shape != prompt_logits.shape:
        raise ValueError("fresh-process prompt logits have the wrong shape")
    prompt_logits_max_abs_difference = float(
        np.max(np.abs(prompt_logits - expected_logits))
    )
    if not np.allclose(prompt_logits, expected_logits, rtol=1e-5, atol=1e-6):
        raise ValueError("fresh-process prompt logits differ from the in-process model")
    rng = jax.random.key(args.seed)
    generated = []
    current = list(prompt_ids)
    forbidden = (SPECIAL["<pad>"], SPECIAL["<bos>"], SPECIAL["<sep>"])
    for _ in range(args.max_new_tokens):
        padded = np.full((1, config["max_length"]), SPECIAL["<pad>"], np.int32)
        active = current[-config["max_length"] :]
        padded[0, : len(active)] = active
        logits = forward(jnp.asarray(padded))[0, len(active) - 1]
        logits = logits.at[jnp.asarray(forbidden)].set(-jnp.inf)
        rng, sample_key = jax.random.split(rng)
        sampled = int(
            candidate.sample_next_token(logits, sample_key, temperature=0.8, top_k=12)
        )
        generated.append(sampled)
        current.append(sampled)
        if sampled == SPECIAL["<eos>"]:
            break

    print(
        json.dumps(
            {
                "prompt": args.prompt,
                "prompt_logits_sha256": sha256_bytes(prompt_logits.tobytes()),
                "prompt_logits_max_abs_difference": prompt_logits_max_abs_difference,
                "prompt_logits_close": True,
                "generated_ids": generated,
                "sample_text": decode(generated, tokenizer),
                "model_sha256": manifest["model_sha256"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "data_sha256": manifest["data_sha256"],
                "actual_backend": jax.default_backend(),
                "source_passages_loaded": len(passages),
                "adaptation_records_loaded": len(instructions),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
