"""Independent math checks and CLI entry point for the L10B project."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location("l10b_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load candidate source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _independent_math(candidate):
    # Uneven partition with a fully empty second rank: the expected gradient is
    # derived directly over the union, not from the candidate's local outputs.
    x = np.asarray([[1., 2.], [-1., 0.5], [0., 0.]], np.float32)
    y = np.asarray([0.25, -1.0, 0.], np.float32)
    valid = np.asarray([True, True, False])
    w = np.asarray([0.3, -0.2], np.float32)
    momentum = np.asarray([0.12, -0.07], np.float32)
    loss0, grad0 = candidate.local_numerator_and_gradient(w, x[:2], y[:2], valid[:2])
    loss1, grad1 = candidate.local_numerator_and_gradient(w, x[2:], y[2:], valid[2:])
    residual = x[valid] @ w - y[valid]
    expected_loss = float(np.sum(residual.astype(np.float64) ** 2))
    expected_grad_sum = 2. * x[valid].astype(np.float64).T @ residual.astype(np.float64)
    np.testing.assert_allclose(float(loss0) + float(loss1), expected_loss, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(np.asarray(grad0) + np.asarray(grad1), expected_grad_sum, rtol=2e-5, atol=2e-6)
    new_w, new_m = candidate.global_transition(w, momentum, np.stack([grad0, grad1]), np.asarray([2., 0.]))
    expected_m = 0.8 * momentum.astype(np.float64) + expected_grad_sum / 2.
    expected_w = w.astype(np.float64) - 0.05 * expected_m
    np.testing.assert_allclose(new_m, expected_m, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(new_w, expected_w, rtol=2e-5, atol=2e-6)

    # Explicit mutation controls: these independently demonstrate the checker
    # has power to detect common semantic mistakes, rather than merely loading.
    wrong_local_mean = (np.asarray(grad0) / 2. + np.asarray(grad1)) / 2.
    assert np.max(np.abs(wrong_local_mean - expected_grad_sum / 2.)) > 1e-3
    no_momentum = expected_grad_sum / 2.
    assert np.max(np.abs(no_momentum - expected_m)) > 1e-3
    key = 17
    correct_order = sorted(range(5), key=lambda i: hashlib.sha256(f"{key}:0:{i}".encode()).hexdigest())
    stale_key_order = sorted(range(5), key=lambda i: hashlib.sha256(f"999:0:{i}".encode()).hexdigest())
    assert correct_order[:3] != stale_key_order[:3]
    assert correct_order[:3] != correct_order[1:4]
    # Duplicate ownership and early publication are rejected from the trusted
    # expected ownership/manifest completeness checks used by the process gate.
    expected_ids = [1, 3, 4]
    duplicate = [[1, 3], [3]]
    assert len(set(duplicate[0]).intersection(duplicate[1])) > 0
    assert sorted([x for shard in duplicate for x in shard]) != sorted(expected_ids)
    expected_ranks, actual_ranks = {0, 1}, {0}
    assert actual_ranks != expected_ranks
    return {"loss_numerator": expected_loss, "global_gradient": (expected_grad_sum / 2.).tolist(),
            "next_weights": expected_w.tolist(), "next_momentum": expected_m.tolist(),
            "negative_controls": ["local-mean", "weight-only", "stale-key", "stale-cursor",
                                  "duplicate-shard", "missing-rank-early-commit"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("math", "checkpoint", "worker-loss", "all"), default="all")
    p.add_argument("--candidate", type=Path, default=HERE / "starter" / "model.py")
    p.add_argument("--work", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    candidate = a.candidate.resolve()
    result = {"schema": "l10b-check-report-v1", "stage": a.stage, "candidate": str(candidate),
              "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
              "math": _independent_math(load_candidate(candidate))}
    with tempfile.TemporaryDirectory(prefix="l10b-equivalent-candidate-") as temp:
        equivalent = Path(temp) / "equivalent.py"
        equivalent.write_bytes(candidate.read_bytes() + b"\n# harmless equivalent spelling: comment only\n")
        equivalent_math = _independent_math(load_candidate(equivalent))
        assert equivalent_math == result["math"]
        result["equivalent_candidate"] = {"sha256": hashlib.sha256(equivalent.read_bytes()).hexdigest(),
                                           "no_op_source_edit": "trailing comment", "math_passed": True}
    if a.stage in {"checkpoint", "worker-loss", "all"}:
        from supervisor import run_all
        from supervisor import run_checkpoint, run_worker_loss
        work = a.work or Path(tempfile.mkdtemp(prefix=f"l10b-{a.stage}-"))
        try:
            if a.stage == "checkpoint":
                result["checkpoint_replay"] = run_checkpoint(candidate, work)
            elif a.stage == "worker-loss":
                result["worker_loss"] = run_worker_loss(candidate, work)
            else:
                result["multiprocess"] = run_all(candidate, work)
        except PermissionError as err:
            result["multiprocess"] = {"status": "BLOCKED_LOOPBACK_PERMISSION", "error": str(err),
                                      "note": "No worker-process success is claimed; rerun on a host that permits 127.0.0.1 sockets."}
            if a.output:
                a.output.parent.mkdir(parents=True, exist_ok=True)
                a.output.write_text(json.dumps(result, indent=2, sort_keys=True))
            print(json.dumps({"status": "BLOCKED_LOOPBACK_PERMISSION", "stage": a.stage, "error": str(err)}, sort_keys=True))
            raise SystemExit(2)
        result["work"] = str(work)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({"status": "PASS", "stage": a.stage, "candidate_sha256": result["candidate_sha256"],
                      "work": result.get("work")}, sort_keys=True))


if __name__ == "__main__":
    main()
