"""Independent staged checks for complete-state local recovery.

The checker derives stream order, gradients and PRNG transitions itself. Its
subprocess tests establish a bounded local-filesystem process protocol only.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHECKER = Path(__file__).resolve()
SOLUTION = ROOT / "solution" / "model.py"
CFG = {"lr": 0.05, "momentum": 0.8, "noise_scale": 0.01, "parameter_shape": [2]}
STREAM_CFG = {
    "lr": 0.03,
    "momentum": 0.7,
    "noise_scale": 0.02,
    "batch_size": 3,
    "order_seed": 271,
    "initial_rng_key": [31, 47],
    "parameter_shape": [2],
    "parameter_shape": [2],
}
IDS = np.array([41, 7, 93, 18, 55], dtype=np.int64)
TARGETS = {
    int(i): np.array([((int(i) % 11) - 4) / 7, ((int(i) % 5) - 2) / 3], np.float32)
    for i in IDS
}


def load(implementation):
    p = Path(implementation)
    if str(implementation) == "solution":
        p = SOLUTION
    elif str(implementation) == "starter":
        p = ROOT / "starter" / "model.py"
    elif p.is_dir():
        p = p / "model.py"
    spec = importlib.util.spec_from_file_location(
        "recovery_candidate_" + uuid.uuid4().hex, p
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__candidate_path__ = p.resolve()
    return module


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initial():
    return {
        "weight": np.array([1.0, -1.0], np.float32),
        "velocity": np.array([0.2, -0.3], np.float32),
        "rng_key": np.array([31, 47], np.uint32),
        "cursor": np.int64(0),
    }


def _order_oracle(source_ids, seed, epoch):
    # Independent spelling and data flow from solution._order_for.
    decorated = []
    for sid in map(int, source_ids):
        identity = (str(seed) + ":" + str(epoch) + ":" + str(sid)).encode("utf-8")
        decorated.append((hashlib.sha256(identity).digest(), sid))
    return np.asarray([sid for _, sid in sorted(decorated)], dtype=np.int64)


def _noise_oracle(key, count):
    # Python-int/math derivation, independent of NumPy and the candidate helper.
    mask = (1 << 64) - 1
    x = int(key[0]) | (int(key[1]) << 32)
    vals = []
    while len(vals) < count:
        us = []
        for _ in range(2):
            x ^= x >> 12
            x ^= (x << 25) & mask
            x ^= x >> 27
            x &= mask
            z = (x * 2685821657736338717) & mask
            us.append((z + 0.5) / float(1 << 64))
        r = math.sqrt(-2.0 * math.log(us[0]))
        a = 2.0 * math.pi * us[1]
        vals.extend((r * math.cos(a), r * math.sin(a)))
    next_key = np.array([x & 0xFFFFFFFF, (x >> 32) & 0xFFFFFFFF], dtype=np.uint32)
    return np.asarray(vals[:count], dtype=np.float32), next_key


def _oracle_update(state, batch, config):
    w = np.asarray(state["weight"], dtype=np.float32)
    old_v = np.asarray(state["velocity"], dtype=np.float32)
    noise, key = _noise_oracle(state["rng_key"], w.size)
    noise = noise.reshape(w.shape) * np.float32(config["noise_scale"])
    grad = (w - np.asarray(batch, dtype=np.float32)) + noise
    v = np.float32(config["momentum"]) * old_v + grad
    return {
        **state,
        "weight": w - np.float32(config["lr"]) * v,
        "velocity": v,
        "rng_key": key,
        "cursor": np.asarray(state["cursor"] + 1, dtype=np.int64),
    }


def _oracle_stream_update(state, targets_by_id, config):
    # Independently derive epoch, IDs, mean gradient, momentum and update clock.
    cur = {
        k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v
        for k, v in state.items()
    }
    epoch = int(cur["epoch"])
    position = int(cur["cursor"])
    if position == len(cur["epoch_order"]):
        epoch += 1
        position = 0
        cur["epoch"] = np.int64(epoch)
        cur["cursor"] = np.int64(0)
        cur["epoch_order"] = _order_oracle(
            cur["source_ids"], config["order_seed"], epoch
        )
    order = _order_oracle(cur["source_ids"], config["order_seed"], epoch)
    stop = min(position + config["batch_size"], len(order))
    chosen = order[position:stop]
    rows = [np.asarray(targets_by_id[int(sid)], dtype=np.float32) for sid in chosen]
    w = np.asarray(cur["weight"], dtype=np.float32)
    old_v = np.asarray(cur["velocity"], dtype=np.float32)
    gradient = np.mean(np.stack([w - row for row in rows]), axis=0, dtype=np.float32)
    noise, key = _noise_oracle(cur["rng_key"], w.size)
    gradient = gradient + np.float32(config["noise_scale"]) * noise
    v = np.float32(config["momentum"]) * old_v + gradient
    cur["weight"] = w - np.float32(config["lr"]) * v
    cur["velocity"] = v
    cur["rng_key"] = key
    cur["cursor"] = np.int64(stop)
    cur["consumed_ids"] = np.concatenate((cur["consumed_ids"], chosen)).astype(np.int64)
    cur["update_ends"] = np.append(cur["update_ends"], len(cur["consumed_ids"])).astype(
        np.int64
    )
    cur["optimizer_step"] = np.int64(int(cur["optimizer_step"]) + 1)
    return cur, chosen


def _compare_state(actual, expected, exact=True):
    assert set(actual) == set(expected), (set(actual), set(expected))
    for key in expected:
        if exact:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        else:
            np.testing.assert_allclose(
                actual[key], expected[key], rtol=1e-6, atol=1e-7, err_msg=key
            )


def _identities(candidate_path, source_ids=IDS, targets=TARGETS):
    target_rows = [
        [int(i), np.asarray(targets[int(i)], dtype="<f4").tolist()] for i in source_ids
    ]
    dataset_hash = hashlib.sha256(
        json.dumps(target_rows, separators=(",", ":")).encode()
    ).hexdigest()
    ordered_hash = hashlib.sha256(
        np.asarray(source_ids, dtype=np.int64).tobytes()
    ).hexdigest()
    return {
        "candidate_sha256": file_sha(candidate_path),
        "dataset_sha256": dataset_hash,
        "ordered_source_ids_sha256": ordered_hash,
        "training_source_sha256": file_sha(SOLUTION),
    }


def _stream_initial(m, candidate_path):
    state = m.initial_stream_state(IDS, STREAM_CFG)
    expected_order = _order_oracle(IDS, STREAM_CFG["order_seed"], 0)
    np.testing.assert_array_equal(state["epoch_order"], expected_order)
    return state, _identities(candidate_path)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _run(cmd, timeout=20):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def stage1(m):
    first = m.train_step(initial(), np.array([0.0, 0.0], np.float32), CFG)
    expected = _oracle_update(initial(), np.array([0.0, 0.0], np.float32), CFG)
    _compare_state(first, expected)
    assert int(first["cursor"]) == 1
    malformed = initial()
    malformed["velocity"] = np.array([0.2], np.float32)
    try:
        m.train_step(malformed, np.array([0.0, 0.0], np.float32), CFG)
    except ValueError:
        shape_update_rejected = True
    else:
        raise AssertionError("legacy train_step accepted a wrong momentum shape")
    # Two independent finite-difference sizes check the mean-squared objective
    # gradient in two directions before it enters the momentum update.
    w = np.array([0.7, -0.4], np.float64)
    target = np.array([-0.2, 0.6], np.float64)
    analytic = w - target
    dirs = (np.array([1.0, 0.0]), np.array([0.3, -0.8]))
    checks = []
    objective = lambda x: 0.5 * np.sum((x - target) ** 2)
    for direction in dirs:
        values = []
        for eps in (1e-3, 2e-4):
            fd = (objective(w + eps * direction) - objective(w - eps * direction)) / (
                2 * eps
            )
            exact = float(np.dot(analytic, direction))
            assert abs(fd - exact) < 2e-7
            values.append({"epsilon": eps, "finite_difference": fd, "analytic": exact})
        checks.append(values)
    return {
        "update_matches_independent_oracle": True,
        "legacy_wrong_momentum_shape_rejected": shape_update_rejected,
        "finite_difference_directions": checks,
    }


def stage2(m, candidate_path):
    # Preserve the original public same-process checkpoint/resume exercise.
    legacy = initial()
    full = m.train_step(
        m.train_step(legacy, np.array([0.0, 0.0]), CFG), np.array([0.3, -0.2]), CFG
    )
    split = m.train_step(legacy, np.array([0.0, 0.0]), CFG)
    with tempfile.TemporaryDirectory(prefix="recovery-audit-stage2-") as td:
        checkpoint = Path(td) / "legacy.npz"
        m.save_checkpoint(checkpoint, 1, CFG, split)
        resumed = m.load_checkpoint(checkpoint, CFG)
        continued = m.train_step(resumed["arrays"], np.array([0.3, -0.2]), CFG)
        for key in full:
            np.testing.assert_array_equal(full[key], continued[key])
    state, identities = _stream_initial(m, candidate_path)
    trace = []
    expected = state
    for _ in range(3):
        expected, expected_ids = _oracle_stream_update(expected, TARGETS, STREAM_CFG)
        state = m.stream_step(state, TARGETS, STREAM_CFG)
        _compare_state(state, expected)
        trace.append(
            {
                "step": int(state["optimizer_step"]),
                "ids": expected_ids.tolist(),
                "update_end": int(state["update_ends"][-1]),
            }
        )
    assert trace == [
        {
            "step": 1,
            "ids": _order_oracle(IDS, STREAM_CFG["order_seed"], 0)[:3].tolist(),
            "update_end": 3,
        },
        {
            "step": 2,
            "ids": _order_oracle(IDS, STREAM_CFG["order_seed"], 0)[3:].tolist(),
            "update_end": 5,
        },
        {
            "step": 3,
            "ids": _order_oracle(IDS, STREAM_CFG["order_seed"], 1)[:3].tolist(),
            "update_end": 8,
        },
    ]
    return {
        "legacy_same_process_replay": True,
        "independent_stream_updates": trace,
        "nondivisible_tail": True,
        "source_identity": identities,
    }


def stage3(m, output=None, candidate_path=None):
    state = initial()
    with tempfile.TemporaryDirectory(prefix="recovery-audit-legacy-") as td:
        p = Path(td) / "c.npz"
        m.save_checkpoint(p, 1, CFG, state)
        assert m.load_checkpoint(p, CFG)["step"] == 1
        try:
            m.load_checkpoint(p, {**CFG, "lr": 0.2})
            raise AssertionError("config mismatch accepted")
        except ValueError:
            pass
        with np.load(p, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        d["weight"] = np.array([9.0, -1.0], np.float32)
        np.savez(p, **d)
        try:
            m.load_checkpoint(p, CFG)
            raise AssertionError("corruption accepted")
        except ValueError:
            pass
    report = {
        "config_mismatch_rejected": True,
        "byte_corruption_rejected": True,
        "checkpoint_integrity": "SHA-256",
    }
    target = (
        Path(output) if output else Path(tempfile.gettempdir())
    ) / "recovery-audit-evidence.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, sort_keys=True))
    assert json.loads(target.read_text()) == report
    return report


def _checker_array_digest(arrays):
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.asarray(arrays[name])
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a)
        h.update(name.encode() + b"\0")
        h.update(a.dtype.str.encode() + b"\0")
        h.update(json.dumps(list(a.shape), separators=(",", ":")).encode() + b"\0")
        h.update(a.tobytes())
    return h.hexdigest()


def _worker_resume(args):
    m = load(args.candidate)
    config = json.loads(Path(args.config_file).read_text())
    identities = json.loads(Path(args.identities_file).read_text())
    restored = m.restore_generation(args.root, config, identities)
    result = m.stream_step(restored["state"], TARGETS, config)
    m.save_generation(
        args.root, result, config, identities, expected_parent=restored["generation"]
    )
    np.savez(args.result, **result)
    print(
        json.dumps(
            {
                "pid": os.getpid(),
                "from_generation": restored["generation"],
                "next_step": int(result["optimizer_step"]),
                "next_ids": result["consumed_ids"][-config["batch_size"] :].tolist(),
            }
        )
    )


def stage5(m, candidate_path, scratch=None):
    with tempfile.TemporaryDirectory(prefix="recovery-audit-fresh-", dir=scratch) as td:
        root = Path(td) / "checkpoint"
        config_path = Path(td) / "config.json"
        identities_path = Path(td) / "identities.json"
        result_path = Path(td) / "resumed.npz"
        config_path.write_text(json.dumps(STREAM_CFG))
        identities = _identities(candidate_path)
        identities_path.write_text(json.dumps(identities))
        state, identities = _stream_initial(m, candidate_path)
        state = m.stream_step(state, TARGETS, STREAM_CFG)
        state = m.stream_step(state, TARGETS, STREAM_CFG)
        assert int(state["cursor"]) == len(IDS), "checkpoint must be at the epoch tail"
        parent = m.save_generation(
            root, state, STREAM_CFG, identities, expected_parent=None
        )
        # Independent uninterrupted trajectory takes the next epoch's first batch.
        expected, next_ids = _oracle_stream_update(state, TARGETS, STREAM_CFG)
        proc = _run(
            [
                sys.executable,
                str(CHECKER),
                "--worker-resume",
                "--candidate",
                str(candidate_path),
                "--root",
                str(root),
                "--config-file",
                str(config_path),
                "--identities-file",
                str(identities_path),
                "--result",
                str(result_path),
            ]
        )
        assert proc.returncode == 0, proc.stderr + proc.stdout
        with np.load(result_path, allow_pickle=False) as z:
            resumed = {k: z[k] for k in z.files}
        _compare_state(resumed, expected)
        after = m.restore_generation(root, STREAM_CFG, identities)
        assert after["parent_generation"] == parent["generation"]
        report = {
            "pid": int(json.loads(proc.stdout.strip())["pid"]),
            "exit_code": proc.returncode,
            "tail_cursor": int(state["cursor"]),
            "fresh_process_next_ids": next_ids.tolist(),
            "fresh_process_matches_uninterrupted": True,
            "published_generation": after["generation"],
            "worker_output_sha256": file_sha(result_path),
        }
    report["schema_and_identity_controls"] = _schema_identity_controls(
        m, candidate_path
    )
    return report


def _schema_identity_controls(m, candidate_path):
    state, identities = _stream_initial(m, candidate_path)
    report = {"valid_schema_roundtrip": False}
    with tempfile.TemporaryDirectory(prefix="recovery-audit-schema-") as td:
        root = Path(td) / "store"
        saved = m.save_generation(
            root, state, STREAM_CFG, identities, expected_parent=None
        )
        restored = m.restore_generation(root, STREAM_CFG, identities)
        _compare_state(restored["state"], state)
        report["valid_schema_roundtrip"] = True
        for label, mutate in (
            (
                "velocity_shape_writer_rejected",
                lambda a: a.__setitem__("velocity", np.array([0.2], np.float32)),
            ),
            (
                "scalar_clock_rank_rejected",
                lambda a: a.__setitem__("epoch", np.array([0], np.int64)),
            ),
            (
                "history_rank_rejected",
                lambda a: a.__setitem__("update_ends", np.empty((0, 1), np.int64)),
            ),
        ):
            malformed = {k: np.array(v, copy=True) for k, v in state.items()}
            mutate(malformed)
            try:
                m.save_generation(
                    Path(td) / ("invalid-" + label),
                    malformed,
                    STREAM_CFG,
                    identities,
                    expected_parent=None,
                )
            except ValueError:
                report[label] = True
            else:
                raise AssertionError(label + " accepted malformed state")
        for label, mutate in (
            (
                "scalar_clock_update_rank_rejected",
                lambda a: a.__setitem__("cursor", np.array([0], np.int64)),
            ),
            (
                "history_update_rank_rejected",
                lambda a: a.__setitem__("consumed_ids", np.empty((0, 1), np.int64)),
            ),
        ):
            malformed = {k: np.array(v, copy=True) for k, v in state.items()}
            mutate(malformed)
            try:
                m.stream_step(malformed, TARGETS, STREAM_CFG)
            except ValueError:
                report[label] = True
            else:
                raise AssertionError(label + " accepted malformed state")
        bad_update = {k: np.array(v, copy=True) for k, v in state.items()}
        bad_update["velocity"] = np.array([0.2], np.float32)
        try:
            m.stream_step(bad_update, TARGETS, STREAM_CFG)
        except ValueError:
            report["velocity_shape_update_rejected"] = True
        else:
            raise AssertionError("update accepted mismatched optimizer shape")

        # Rechecksummed storage cannot redefine the caller's expected shape.
        pointer = json.loads((root / "CURRENT.json").read_text())
        directory = root / "generations" / pointer["generation"]
        manifest = json.loads((directory / "manifest.json").read_text())
        with np.load(directory / "state.npz", allow_pickle=False) as z:
            forged = {k: z[k] for k in z.files}
        forged["velocity"] = np.array([0.2], np.float32)
        with open(directory / "state.npz", "wb") as f:
            np.savez(f, **forged)
        manifest["state_sha256"] = file_sha(directory / "state.npz")
        manifest["array_sha256"] = _checker_array_digest(forged)
        manifest["array_schema"] = {
            k: {"dtype": v.dtype.str, "shape": list(v.shape)}
            for k, v in sorted(forged.items())
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        )
        pointer["manifest_sha256"] = file_sha(directory / "manifest.json")
        _write_json(root / "CURRENT.json", pointer)
        try:
            m.restore_generation(root, STREAM_CFG, identities)
        except ValueError as e:
            report["rechecksummed_velocity_shape_restore_rejected"] = (
                "optimizer shape" in str(e)
            )
            assert report["rechecksummed_velocity_shape_restore_rejected"], str(e)
        else:
            raise AssertionError("rechecksummed malformed optimizer shape restored")

        # Config/source binding and locally rechecksummed cursor/order forgeries.
        valid_root = Path(td) / "identity-store"
        m.save_generation(
            valid_root, state, STREAM_CFG, identities, expected_parent=None
        )
        for changed_config, changed_identities in (
            ({**STREAM_CFG, "order_seed": STREAM_CFG["order_seed"] + 1}, identities),
            (STREAM_CFG, {**identities, "ordered_source_ids_sha256": "0" * 64}),
        ):
            try:
                m.restore_generation(valid_root, changed_config, changed_identities)
            except ValueError:
                pass
            else:
                raise AssertionError(
                    "caller-expected config/source identity mismatch accepted"
                )
        ident_pointer = json.loads((valid_root / "CURRENT.json").read_text())
        ident_dir = valid_root / "generations" / ident_pointer["generation"]
        ident_manifest = json.loads((ident_dir / "manifest.json").read_text())
        with np.load(ident_dir / "state.npz", allow_pickle=False) as z:
            bad_order = {k: z[k] for k in z.files}
        bad_order["epoch_order"] = bad_order["epoch_order"][::-1].copy()
        with open(ident_dir / "state.npz", "wb") as f:
            np.savez(f, **bad_order)
        ident_manifest["state_sha256"] = file_sha(ident_dir / "state.npz")
        ident_manifest["array_sha256"] = _checker_array_digest(bad_order)
        ident_manifest["array_schema"] = {
            k: {"dtype": v.dtype.str, "shape": list(v.shape)}
            for k, v in sorted(bad_order.items())
        }
        (ident_dir / "manifest.json").write_text(
            json.dumps(ident_manifest, sort_keys=True, separators=(",", ":"))
        )
        ident_pointer["manifest_sha256"] = file_sha(ident_dir / "manifest.json")
        _write_json(valid_root / "CURRENT.json", ident_pointer)
        try:
            m.restore_generation(valid_root, STREAM_CFG, identities)
        except ValueError as e:
            report["rechecksummed_wrong_order_rejected"] = "epoch order" in str(e)
            assert report["rechecksummed_wrong_order_rejected"], str(e)
        else:
            raise AssertionError("rechecksummed wrong sample order restored")
    return report


def _worker_fault(args):
    m = load(args.candidate)
    config = json.loads(Path(args.config_file).read_text())
    identities = json.loads(Path(args.identities_file).read_text())
    with np.load(args.state_file, allow_pickle=False) as z:
        state = {k: z[k] for k in z.files}
    try:
        m.save_generation(
            args.root,
            state,
            config,
            identities,
            expected_parent=args.expected_parent or None,
            failpoint=args.failpoint,
        )
        status = "returned"
    except (
        Exception
    ) as e:  # the process exits after recording the injected interruption
        status = type(e).__name__ + ":" + str(e)
    Path(args.result).write_text(
        json.dumps(
            {"pid": os.getpid(), "status": status, "failpoint": args.failpoint},
            sort_keys=True,
        )
    )
    os._exit(73)


def stage6(m, candidate_path, scratch=None):
    report = {
        "failpoints": {},
        "publication_boundary": "CURRENT.json atomic replace after immutable generation preparation",
        "scope": "ordinary process termination on one POSIX local filesystem; no power-loss claim",
        "lock_backend": "fcntl.flock",
        "tested_platform": sys.platform,
        "platform_compatible_stages": ["1", "2", "3"],
    }
    with tempfile.TemporaryDirectory(
        prefix="recovery-audit-faults-", dir=scratch
    ) as td:
        base = Path(td)
        root = base / "store"
        state, identities = _stream_initial(m, candidate_path)
        first = m.save_generation(
            root, state, STREAM_CFG, identities, expected_parent=None
        )
        config_file = base / "config.json"
        identities_file = base / "identities.json"
        state_file = base / "state.npz"
        config_file.write_text(json.dumps(STREAM_CFG))
        identities_file.write_text(json.dumps(identities))
        # Advance a valid alternate generation so publication changes are observable.
        changed = m.stream_step(state, TARGETS, STREAM_CFG)
        np.savez(state_file, **changed)
        for failpoint in ("during_temp_write", "after_payload", "after_manifest"):
            prior_temp_dirs = {p.name for p in (root / "generations").glob(".tmp-g-*")}
            result = base / (failpoint + ".json")
            cmd = [
                sys.executable,
                str(CHECKER),
                "--worker-fault",
                "--candidate",
                str(candidate_path),
                "--root",
                str(root),
                "--config-file",
                str(config_file),
                "--identities-file",
                str(identities_file),
                "--state-file",
                str(state_file),
                "--result",
                str(result),
                "--failpoint",
                failpoint,
                "--expected-parent",
                first["generation"],
            ]
            start = time.monotonic()
            proc = _run(cmd, timeout=15)
            assert proc.returncode == 73, {
                "failpoint": failpoint,
                "code": proc.returncode,
                "out": proc.stdout,
                "err": proc.stderr,
            }
            assert result.exists(), "worker did not record failpoint exit"
            worker_record = json.loads(result.read_text())
            expected_status = "RuntimeError:injected crash at " + failpoint
            assert worker_record == {
                "pid": worker_record["pid"],
                "status": expected_status,
                "failpoint": failpoint,
            }, worker_record
            temp_dirs = [
                p
                for p in (root / "generations").glob(".tmp-g-*")
                if p.name not in prior_temp_dirs
            ]
            assert len(temp_dirs) == 1, {
                "failpoint": failpoint,
                "temporary_generations": [p.name for p in temp_dirs],
            }
            temp_dir = temp_dirs[0]
            payload = temp_dir / "state.npz"
            partial = temp_dir / "state.npz.tmp"
            manifest_path = temp_dir / "manifest.json"
            if failpoint == "during_temp_write":
                assert partial.is_file() and 0 < partial.stat().st_size
                assert not payload.exists() and not manifest_path.exists()
                full_buffer = __import__("io").BytesIO()
                np.savez(full_buffer, **changed)
                assert partial.stat().st_size < len(full_buffer.getvalue())
                try:
                    with np.load(partial, allow_pickle=False):
                        raise AssertionError("partial payload was loadable")
                except (ValueError, EOFError, OSError, zipfile.BadZipFile):
                    pass
                artifact = {
                    "kind": "partial_npz_write",
                    "path": str(partial.relative_to(root)),
                    "bytes": partial.stat().st_size,
                    "loadable": False,
                }
            elif failpoint == "after_payload":
                assert (
                    payload.is_file()
                    and not partial.exists()
                    and not manifest_path.exists()
                )
                with np.load(payload, allow_pickle=False) as z:
                    _compare_state({k: z[k] for k in z.files}, changed)
                artifact = {
                    "kind": "complete_payload_without_manifest",
                    "path": str(payload.relative_to(root)),
                    "sha256": file_sha(payload),
                    "matches_expected_state": True,
                }
            else:
                assert (
                    payload.is_file()
                    and manifest_path.is_file()
                    and not partial.exists()
                )
                with np.load(payload, allow_pickle=False) as z:
                    _compare_state({k: z[k] for k in z.files}, changed)
                manifest = json.loads(manifest_path.read_text())
                assert manifest["generation"] == temp_dir.name[len(".tmp-") :]
                assert manifest["parent_generation"] == first["generation"]
                assert (
                    manifest["config"] == STREAM_CFG
                    and manifest["identities"] == identities
                )
                assert manifest["state_sha256"] == file_sha(payload)
                assert manifest["array_sha256"] == _checker_array_digest(changed)
                artifact = {
                    "kind": "complete_unpublished_generation",
                    "path": str(temp_dir.relative_to(root)),
                    "payload_sha256": file_sha(payload),
                    "manifest_sha256": file_sha(manifest_path),
                    "manifest_bound": True,
                }
            restored = m.restore_generation(root, STREAM_CFG, identities)
            assert (
                restored["generation"] == first["generation"]
            ), "incomplete generation became visible"
            report["failpoints"][failpoint] = {
                "pid": worker_record["pid"],
                "worker_status": worker_record["status"],
                "exit_code": proc.returncode,
                "elapsed_seconds": time.monotonic() - start,
                "reached_exact_failpoint": True,
                "interrupted_artifact": artifact,
                "current_unchanged": True,
                "prior_generation_restored": True,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        # Publication failpoint occurs after atomic pointer replacement. It must
        # expose the complete new generation, never a partial one.
        result = base / "after_publish.json"
        proc = _run(
            [
                sys.executable,
                str(CHECKER),
                "--worker-fault",
                "--candidate",
                str(candidate_path),
                "--root",
                str(root),
                "--config-file",
                str(config_file),
                "--identities-file",
                str(identities_file),
                "--state-file",
                str(state_file),
                "--result",
                str(result),
                "--failpoint",
                "after_publish",
                "--expected-parent",
                first["generation"],
            ],
            timeout=15,
        )
        assert proc.returncode == 73 and result.exists()
        worker_record = json.loads(result.read_text())
        assert (
            worker_record["status"] == "RuntimeError:injected crash at after_publish"
        ), worker_record
        assert worker_record["failpoint"] == "after_publish"
        restored = m.restore_generation(root, STREAM_CFG, identities)
        assert restored["generation"] != first["generation"]
        _compare_state(restored["state"], changed)
        report["failpoints"]["after_publish"] = {
            "pid": worker_record["pid"],
            "worker_status": worker_record["status"],
            "exit_code": proc.returncode,
            "reached_exact_failpoint": True,
            "complete_generation_visible": True,
            "restored_state_matches": True,
            "published_generation": restored["generation"],
        }
        # A forged pointer to an uncommitted/nonexistent generation is rejected.
        (root / "CURRENT.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "generation": "g-incomplete",
                    "manifest_sha256": "0" * 64,
                }
            )
        )
        try:
            m.restore_generation(root, STREAM_CFG, identities)
            raise AssertionError("premature pointer to incomplete generation accepted")
        except (ValueError, FileNotFoundError):
            report["premature_publication_rejected"] = True
    return report


def _writer_worker(
    candidate_path,
    root,
    config_file,
    identities_file,
    ready_dir,
    writer_id,
    result_file,
):
    m = load(candidate_path)
    config = json.loads(Path(config_file).read_text())
    identities = json.loads(Path(identities_file).read_text())
    loaded = m.restore_generation(root, config, identities)
    state = loaded["state"]
    # Both writers derive different valid proposals from the same parent.
    proposal = m.stream_step(state, TARGETS, config)
    (Path(ready_dir) / ("ready-" + writer_id)).write_text("ready")
    deadline = time.monotonic() + 10
    while len(list(Path(ready_dir).glob("ready-*"))) < 2:
        if time.monotonic() > deadline:
            raise TimeoutError("concurrent writer rendezvous timed out")
        time.sleep(0.01)
    try:
        saved = m.save_generation(
            root, proposal, config, identities, expected_parent=loaded["generation"]
        )
        result = {"pid": os.getpid(), "status": "published", **saved}
    except RuntimeError as e:
        result = {"pid": os.getpid(), "status": "conflict", "error": str(e)}
    Path(result_file).write_text(json.dumps(result, sort_keys=True))


def stage7(m, candidate_path, scratch=None):
    with tempfile.TemporaryDirectory(
        prefix="recovery-audit-writers-", dir=scratch
    ) as td:
        base = Path(td)
        root = base / "store"
        state, identities = _stream_initial(m, candidate_path)
        parent = m.save_generation(
            root, state, STREAM_CFG, identities, expected_parent=None
        )
        cf, inf = base / "config.json", base / "identities.json"
        cf.write_text(json.dumps(STREAM_CFG))
        inf.write_text(json.dumps(identities))
        ready = base / "ready"
        ready.mkdir()
        outputs = [base / "writer-0.json", base / "writer-1.json"]
        procs = []
        for i in range(2):
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(CHECKER),
                    "--worker-writer",
                    "--candidate",
                    str(candidate_path),
                    "--root",
                    str(root),
                    "--config-file",
                    str(cf),
                    "--identities-file",
                    str(inf),
                    "--ready-dir",
                    str(ready),
                    "--writer-id",
                    str(i),
                    "--result",
                    str(outputs[i]),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            procs.append(proc)
        results = []
        for proc, outpath in zip(procs, outputs):
            stdout, stderr = proc.communicate(timeout=20)
            assert proc.returncode == 0, {
                "code": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
            }
            results.append(json.loads(outpath.read_text()))
        statuses = [x["status"] for x in results]
        assert sorted(statuses) == ["conflict", "published"], results
        current = m.restore_generation(root, STREAM_CFG, identities)
        assert current["parent_generation"] == parent["generation"]
        return {
            "pids": [x["pid"] for x in results],
            "statuses": statuses,
            "winner_generation": current["generation"],
            "expected_parent": parent["generation"],
            "loser_conflict": True,
            "no_silent_lost_update": True,
        }


def _semantic_negative_controls(candidate_path):
    # Mutation fixtures always come from the owned reference, never by editing
    # the learner's file or depending on its private source spelling.
    original = SOLUTION.read_text()
    mutations = {
        "ignored_momentum": (
            'next_velocity = np.float32(config["momentum"]) * velocity + grad',
            "next_velocity = grad",
        ),
        "repeated_rng": (
            'noise, next_key = _random_vector(out["rng_key"], weight.size)',
            'noise, next_key = np.zeros(weight.size, dtype=np.float32), np.asarray(out["rng_key"], dtype=np.uint32)',
        ),
    }
    wrong_batch_tree = ast.parse(original)
    order_functions = [
        node
        for node in wrong_batch_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_order_for"
    ]
    assert len(order_functions) == 1, "expected one top-level _order_for function"
    sort_calls = []
    for node in ast.walk(order_functions[0]):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "sorted" or not node.args:
            continue
        if not isinstance(node.args[0], ast.Name) or node.args[0].id != "ids":
            continue
        key = next((kw.value for kw in node.keywords if kw.arg == "key"), None)
        if not isinstance(key, ast.Lambda):
            continue
        key_calls = [n for n in ast.walk(key.body) if isinstance(n, ast.Call)]
        has_sha256 = any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "sha256"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "hashlib"
            for call in key_calls
        )
        has_hexdigest = any(
            isinstance(call.func, ast.Attribute) and call.func.attr == "hexdigest"
            for call in key_calls
        )
        if has_sha256 and has_hexdigest:
            sort_calls.append(node)
    assert (
        len(sort_calls) == 1
    ), "expected one hash-keyed sorted(ids) call in _order_for"
    wrong_batch_before = ast.get_source_segment(original, sort_calls[0])
    assert wrong_batch_before and original.count(wrong_batch_before) == 1
    wrong_batch_after = "sorted(ids)"
    expected_order = _order_oracle(IDS, STREAM_CFG["order_seed"], 0)
    assert not np.array_equal(
        expected_order, np.sort(IDS)
    ), "wrong-next-batch fixture must differ from plain sorted IDs"
    result = {}
    with tempfile.TemporaryDirectory(prefix="recovery-audit-mutants-") as td:
        reference = load(SOLUTION)
        positive_state = reference.initial_stream_state(IDS, STREAM_CFG)
        positive_expected, _ = _oracle_stream_update(
            positive_state, TARGETS, STREAM_CFG
        )
        _compare_state(
            reference.stream_step(positive_state, TARGETS, STREAM_CFG),
            positive_expected,
        )
        result["unchanged_reference_positive_control"] = (
            "accepted by independent transition comparison"
        )

        def require_rejected(name, candidate):
            test_state = candidate.initial_stream_state(IDS, STREAM_CFG)
            expected, _ = _oracle_stream_update(test_state, TARGETS, STREAM_CFG)
            actual = candidate.stream_step(test_state, TARGETS, STREAM_CFG)
            for field in expected:
                try:
                    np.testing.assert_array_equal(
                        actual[field], expected[field], err_msg=field
                    )
                except AssertionError:
                    return "independent mismatch in " + field
            raise AssertionError(name + " mutation was not detected")

        # Checker self-test: an owned mutation that is a no-op must be treated
        # as an undetected defect, proving the guard does not catch itself.
        noop_path = Path(td) / "noop-equivalent.py"
        noop_path.write_text(original)
        noop = load(noop_path)
        try:
            require_rejected("no_op_mutation", noop)
        except AssertionError as expected_failure:
            assert "was not detected" in str(expected_failure)
            result["no_op_mutation_must_fail_sensitivity_self_test"] = (
                "failed as required"
            )
        else:
            raise AssertionError(
                "checker sensitivity self-test accepted a no-op mutant"
            )

        for name, (before, after) in mutations.items():
            assert before in original, "owned mutation anchor missing: " + name
            path = Path(td) / (name + ".py")
            # The momentum equation appears in both compatibility and stream
            # APIs, so this owned mutant deliberately changes both copies.
            path.write_text(original.replace(before, after))
            mutant = load(path)
            result[name] = "rejected: " + require_rejected(name, mutant)
        mutated_source = original.replace(wrong_batch_before, wrong_batch_after, 1)
        assert mutated_source != original
        wrong_batch_path = Path(td) / "wrong_next_batch.py"
        wrong_batch_path.write_text(mutated_source)
        wrong_batch_mutant = load(wrong_batch_path)
        wrong_batch_state = wrong_batch_mutant.initial_stream_state(IDS, STREAM_CFG)
        wrong_batch_expected, _ = _oracle_stream_update(
            wrong_batch_state, TARGETS, STREAM_CFG
        )
        wrong_batch_actual = wrong_batch_mutant.stream_step(
            wrong_batch_state, TARGETS, STREAM_CFG
        )
        mismatches = [
            field
            for field in wrong_batch_expected
            if not np.array_equal(
                wrong_batch_actual[field], wrong_batch_expected[field]
            )
        ]
        specific_mismatches = [
            field for field in ("epoch_order", "consumed_ids") if field in mismatches
        ]
        assert (
            specific_mismatches
        ), "wrong_next_batch was not rejected by independent epoch/order IDs: " + repr(
            mismatches
        )
        result["wrong_next_batch"] = (
            "rejected: independent mismatch in " + specific_mismatches[0]
        )
    # Weight-only checkpoint control: deleting required state fails even if
    # the payload and local array digest are regenerated by the writer.
    with tempfile.TemporaryDirectory(prefix="recovery-audit-weight-only-") as td:
        m = load(SOLUTION)
        state = m.initial_stream_state(IDS, STREAM_CFG)
        identities = _identities(candidate_path)
        root = Path(td) / "store"
        m.save_generation(root, state, STREAM_CFG, identities, expected_parent=None)
        pointer = json.loads((root / "CURRENT.json").read_text())
        directory = root / "generations" / pointer["generation"]
        with np.load(directory / "state.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k in {"weight"}}
        with open(directory / "state.npz", "wb") as f:
            np.savez(f, **arrays)
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["state_sha256"] = file_sha(directory / "state.npz")
        manifest["array_sha256"] = _checker_array_digest(arrays)
        manifest["array_schema"] = {
            k: {"dtype": v.dtype.str, "shape": list(v.shape)} for k, v in arrays.items()
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        )
        pointer["manifest_sha256"] = file_sha(directory / "manifest.json")
        _write_json(root / "CURRENT.json", pointer)
        try:
            m.restore_generation(root, STREAM_CFG, identities)
            raise AssertionError("weight-only checkpoint accepted")
        except ValueError:
            result["weight_only_checkpoint"] = "rejected after local rechecksumming"
    return result


def run(m, n, output=None, scratch=None):
    candidate_path = m.__candidate_path__
    stages = {}
    selected = ["1", "2", "3", "5", "6", "7"] if n in ("4", "all", "default") else [n]
    for key in selected:
        if key == "1":
            stages[key] = stage1(m)
        elif key == "2":
            stages[key] = stage2(m, candidate_path)
        elif key == "3":
            stages[key] = stage3(m, output, candidate_path)
        elif key == "5":
            stages[key] = stage5(m, candidate_path, scratch)
        elif key == "6":
            stages[key] = stage6(m, candidate_path, scratch)
        elif key == "7":
            stages[key] = stage7(m, candidate_path, scratch)
        else:
            raise ValueError("unknown stage " + key)
    if n in ("4", "all", "default"):
        stages["negative_controls"] = _semantic_negative_controls(candidate_path)
        evidence = {
            "schema_version": 2,
            "project": "recovery-audit",
            "requested_stage": n,
            "candidate_path": str(candidate_path),
            "candidate_sha256": file_sha(candidate_path),
            "checker_sha256": file_sha(CHECKER),
            "reference_sha256": file_sha(SOLUTION),
            "supporting_sources_sha256": {
                str(path.relative_to(ROOT.parent.parent)): file_sha(path)
                for path in (
                    ROOT / "README.md",
                    ROOT / "starter" / "model.py",
                    ROOT / "project.json",
                )
            },
            "python": sys.version,
            "numpy": np.__version__,
            "stages": stages,
            "limitations": [
                "local single-filesystem process protocol only",
                "no power-loss, network-filesystem, multihost, or accelerator qualification",
            ],
        }
        if output:
            output_path = Path(output)
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "protocol-evidence.json").write_text(
                json.dumps(evidence, indent=2, sort_keys=True)
            )
    print(
        f"PASS recovery-audit stage {n}: complete-state stream, fresh-process restore, crash boundaries, and writer arbitration"
    )


def _hidden_worker(args):
    if args.worker_resume:
        return _worker_resume(args)
    if args.worker_fault:
        return _worker_fault(args)
    if args.worker_writer:
        return _writer_worker(
            args.candidate,
            args.root,
            args.config_file,
            args.identities_file,
            args.ready_dir,
            args.writer_id,
            args.result,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--implementation",
        default="starter",
        help="solution/starter directory or external model.py file",
    )
    p.add_argument("--candidate", help=argparse.SUPPRESS)
    p.add_argument("--evidence-dir")
    p.add_argument(
        "--scratch-root", help="optional parent for portable task-owned temporary files"
    )
    p.add_argument(
        "--stage",
        choices=("1", "2", "3", "4", "5", "6", "7", "all", "default"),
        default="all",
    )
    p.add_argument("--worker-resume", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--worker-fault", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--worker-writer", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--root")
    p.add_argument("--config-file")
    p.add_argument("--identities-file")
    p.add_argument("--result")
    p.add_argument("--state-file")
    p.add_argument("--failpoint")
    p.add_argument("--expected-parent", default="")
    p.add_argument("--ready-dir")
    p.add_argument("--writer-id")
    a = p.parse_args()
    if a.worker_resume or a.worker_fault or a.worker_writer:
        a.candidate = a.candidate or a.implementation
        _hidden_worker(a)
    else:
        m = load(a.implementation)
        run(m, a.stage, a.evidence_dir, a.scratch_root)
