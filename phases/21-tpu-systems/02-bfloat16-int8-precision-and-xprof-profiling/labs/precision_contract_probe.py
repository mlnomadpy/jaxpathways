#!/usr/bin/env python3
"""Retain a CPU JAX dtype and compiler-IR probe for the tpu-06 dot fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

HERE = Path(__file__).resolve().parent
LESSON_PATH = HERE.parent / "lesson.json"
DEFAULT_OUTPUT = Path(os.environ.get("TMPDIR", "/tmp")) / "tpu06-precision-probe-output"
SHAPE_X = (64, 256)
SHAPE_W = (256, 64)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def fixture(seed: int):
    key_x, key_w = jax.random.split(jax.random.PRNGKey(seed))
    x32 = jax.random.normal(key_x, SHAPE_X, dtype=jnp.float32)
    w32 = jax.random.normal(key_w, SHAPE_W, dtype=jnp.float32) * 0.25
    x_bf16 = x32.astype(jnp.bfloat16)
    w_bf16 = w32.astype(jnp.bfloat16)

    reference_fp32 = jnp.dot(x32, w32, preferred_element_type=jnp.float32)
    requested_fp32 = jnp.dot(x_bf16, w_bf16, preferred_element_type=jnp.float32)
    requested_bf16 = jnp.dot(x_bf16, w_bf16, preferred_element_type=jnp.bfloat16)
    fp32_operand_dot_cast_to_bf16 = requested_fp32.astype(jnp.bfloat16)

    arrays = {
        "x_fp32": np.asarray(x32),
        "w_fp32": np.asarray(w32),
        "x_bf16": np.asarray(x_bf16),
        "w_bf16": np.asarray(w_bf16),
        "reference_fp32": np.asarray(reference_fp32),
        "requested_fp32": np.asarray(requested_fp32),
        "requested_bf16": np.asarray(requested_bf16),
        "requested_fp32_cast_bf16": np.asarray(fp32_operand_dot_cast_to_bf16),
    }
    return arrays


def summarize(seed: int, arrays: dict[str, np.ndarray]) -> dict:
    expected_dtypes = {
        "x_fp32": "float32",
        "w_fp32": "float32",
        "x_bf16": "bfloat16",
        "w_bf16": "bfloat16",
        "reference_fp32": "float32",
        "requested_fp32": "float32",
        "requested_bf16": "bfloat16",
        "requested_fp32_cast_bf16": "bfloat16",
    }
    observed_dtypes = {name: str(value.dtype) for name, value in arrays.items()}
    if observed_dtypes != expected_dtypes:
        raise AssertionError(
            f"seed {seed}: requested and observed array dtypes differ: {observed_dtypes}"
        )
    non_finite = [
        name
        for name, value in arrays.items()
        if not np.isfinite(value.astype(np.float32)).all()
    ]
    if non_finite:
        raise AssertionError(f"seed {seed}: non-finite values in {non_finite}")

    fp32_ref = arrays["reference_fp32"].astype(np.float64)
    fp32_output = arrays["requested_fp32"].astype(np.float64)
    bf16_output = arrays["requested_bf16"].astype(np.float64)
    fp32_cast = arrays["requested_fp32_cast_bf16"]
    bf16_returned = arrays["requested_bf16"]
    matches = np.equal(fp32_cast, bf16_returned)
    different_positions = np.argwhere(~matches)
    discrepancies = [
        {
            "index": [int(index) for index in position],
            "requested_bf16_value": float(
                bf16_returned[tuple(position)].astype(np.float32)
            ),
            "requested_fp32_then_bf16_value": float(
                fp32_cast[tuple(position)].astype(np.float32)
            ),
        }
        for position in different_positions
    ]
    return {
        "seed": seed,
        "shape": {"x": list(arrays["x_fp32"].shape), "w": list(arrays["w_fp32"].shape)},
        "dtypes": observed_dtypes,
        "finite": {
            name: bool(np.isfinite(value.astype(np.float32)).all())
            for name, value in arrays.items()
        },
        "max_abs_error_vs_fp32_jax_reference": {
            "bf16_operands_requested_fp32_result": float(
                np.max(np.abs(fp32_output - fp32_ref))
            ),
            "bf16_operands_requested_bf16_result": float(
                np.max(np.abs(bf16_output - fp32_ref))
            ),
        },
        "mean_abs_error_vs_fp32_jax_reference": {
            "bf16_operands_requested_fp32_result": float(
                np.mean(np.abs(fp32_output - fp32_ref))
            ),
            "bf16_operands_requested_bf16_result": float(
                np.mean(np.abs(bf16_output - fp32_ref))
            ),
        },
        "requested_bf16_vs_same_operands_requested_fp32_result_cast_to_bf16": {
            "equal_count": int(matches.sum()),
            "total_count": int(matches.size),
            "different_count": int((~matches).sum()),
            "different_positions_and_values": discrepancies,
        },
    }


def retain_ir(seed: int, output_dir: Path) -> list[dict]:
    artifacts = []
    for dtype_name, dtype in (
        ("requested-fp32", jnp.float32),
        ("requested-bf16", jnp.bfloat16),
    ):
        function = jax.jit(lambda x, w: jnp.dot(x, w, preferred_element_type=dtype))
        x = jnp.zeros(SHAPE_X, dtype=jnp.bfloat16)
        w = jnp.zeros(SHAPE_W, dtype=jnp.bfloat16)
        lowered = function.lower(x, w)
        stable_path = output_dir / f"seed-{seed}-{dtype_name}.stablehlo.mlir"
        optimized_path = output_dir / f"seed-{seed}-{dtype_name}.optimized-hlo.txt"
        stable_path.write_text(
            str(lowered.compiler_ir(dialect="stablehlo")), encoding="utf-8"
        )
        optimized_path.write_text(lowered.compile().as_text(), encoding="utf-8")
        artifacts.extend(
            [
                {
                    "path": stable_path.name,
                    "sha256": sha256_file(stable_path),
                    "kind": "StableHLO",
                },
                {
                    "path": optimized_path.name,
                    "sha256": sha256_file(optimized_path),
                    "kind": "optimized HLO",
                },
            ]
        )
    return artifacts


def run(output_dir: Path, seeds: list[int]) -> dict:
    if jax.default_backend() != "cpu":
        raise RuntimeError(
            f"This probe is scoped to a CPU rehearsal; JAX selected {jax.default_backend()!r}. "
            "Set JAX_PLATFORMS=cpu explicitly to run it on CPU."
        )
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide one or more distinct integer seeds.")

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    fixture_hashes = []
    for seed in seeds:
        arrays = fixture(seed)
        arrays_path = output_dir / f"seed-{seed}-arrays.npz"
        np.savez_compressed(
            arrays_path,
            **{
                name: value.view(np.uint16) if str(value.dtype) == "bfloat16" else value
                for name, value in arrays.items()
            },
        )
        fixture_hashes.append(
            {"seed": seed, "path": arrays_path.name, "sha256": sha256_file(arrays_path)}
        )
        summaries.append(summarize(seed, arrays))

    ir_artifacts = []
    for seed in [seed for seed in (0, 42) if seed in seeds]:
        ir_artifacts.extend(retain_ir(seed, output_dir))

    summary_path = output_dir / "results.json"
    summary_path.write_text(
        json.dumps({"fixtures": summaries}, indent=2) + "\n", encoding="utf-8"
    )
    lesson_bytes = LESSON_PATH.read_bytes()
    probe_bytes = Path(__file__).read_bytes()
    receipt = {
        "evidence_kind": "cpu_dtype_and_compiler_ir_probe",
        "backend": jax.default_backend(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "numpy": np.__version__,
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
        },
        "fixture": {
            "prng": "jax.random.PRNGKey(seed), split into x and w keys; normal float32 arrays",
            "shape_x": list(SHAPE_X),
            "shape_w": list(SHAPE_W),
            "weight_scale": 0.25,
            "seeds": seeds,
            "changed_condition_seed": next(
                (seed for seed in seeds if seed not in (0, 42)), None
            ),
            "dot_api": "jnp.dot(..., preferred_element_type=...)",
            "output_interpretation": "requested result dtype and observed returned dtype; accumulator internals are not inferred",
        },
        "source_hashes": {
            "lesson.json": sha256_bytes(lesson_bytes),
            "precision_contract_probe.py": sha256_bytes(probe_bytes),
        },
        "raw_array_files": fixture_hashes,
        "array_storage": "BF16 arrays are retained as their exact uint16 bit patterns in NPZ because NumPy archives do not preserve the ml_dtypes bfloat16 dtype metadata; results.json records their observed bfloat16 dtype.",
        "summary": {"path": summary_path.name, "sha256": sha256_file(summary_path)},
        "compiler_ir_artifacts": ir_artifacts,
        "interpretation_limit": "StableHLO and optimized HLO are compiler evidence; they do not prove physical accumulator width or instruction-level reduction order.",
    }
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("TPU06_PRECISION_OUTPUT", DEFAULT_OUTPUT)),
        help="directory for raw arrays, JSON receipt, StableHLO, and optimized HLO; defaults outside this lesson source",
    )
    parser.add_argument(
        "--seeds",
        default="0,42,7",
        help="comma-separated seeds; defaults include the two audited fixtures and one changed seed",
    )
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    output_dir = args.output_dir.expanduser().resolve()
    receipt = run(output_dir, seeds)
    print((output_dir / "results.json").read_text(encoding="utf-8"))
    print(json.dumps(receipt, indent=2))
    print(f"Retained CPU dtype and compiler IR evidence in {output_dir}")


if __name__ == "__main__":
    main()
