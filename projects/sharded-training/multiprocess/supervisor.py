"""Bounded local supervisor for actual multi-controller worker-loss checks."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
import struct
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKER = HERE / "worker.py"
CHECKER = HERE / "check.py"
PYTHON = os.environ.get("JAXPATHWAYS_PYTHON", sys.executable)
WORLD = 2
TOTAL_STEPS = 4
X = np.asarray([[1., 0.], [0., 1.], [1., 1.], [2., -1.], [-1., 2.]], np.float32)
Y = np.asarray([0.5, -0.25, 0.75, 1.5, -1.0], np.float32)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_inventory(root: Path):
    return {
        path.relative_to(root).as_posix(): sha(path)
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def data_sha256():
    x = struct.pack("=10f", 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0, -1.0, 2.0)
    y = struct.pack("=5f", 0.5, -0.25, 0.75, 1.5, -1.0)
    return hashlib.sha256(x + y).hexdigest()


def experiment_identity(candidate: Path, root: Path = HERE, world=WORLD, total_steps=TOTAL_STEPS):
    candidate = candidate.resolve()
    experiment = {
        "schema": "l10b-multiprocess-v1",
        "world": int(world),
        "per_rank": 2,
        "width": 2,
        "seed": 17,
        "learning_rate": 0.05,
        "momentum_rate": 0.8,
        "steps": int(total_steps),
        "data_hash": data_sha256(),
        "sampler": "sha256-key-epoch-id-sort-v1",
    }
    return {
        "schema": "l10b-approved-experiment-identity-v2",
        "candidate": str(candidate),
        "candidate_sha256": sha(candidate),
        "source_hashes": source_inventory(root),
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "jax": importlib.metadata.version("jax"),
            "jaxlib": importlib.metadata.version("jaxlib"),
            "numpy": importlib.metadata.version("numpy"),
        },
        "experiment": experiment,
        "data_sha256": experiment["data_hash"],
    }


def _has_checkpoint_evidence(work: Path):
    checkpoint_root = work / "checkpoints"
    if checkpoint_root.exists() and any(path.is_file() for path in checkpoint_root.rglob("*")):
        return True
    return any(work.glob("rank-info-*.json")) or any(work.glob("transition-*.json")) or any(
        (work / marker).exists()
        for marker in ("final.json", "boundary-ready.json", "incomplete-ready.json")
    )


def _validate_identity_request(identity, candidate: Path, world=WORLD, total_steps=TOTAL_STEPS):
    expected = experiment_identity(candidate, world=world, total_steps=total_steps)
    for key in ("schema", "candidate", "candidate_sha256", "source_hashes", "runtime", "experiment", "data_sha256"):
        if identity.get(key) != expected[key]:
            raise ValueError("approved experiment/config/data identity mismatch")


def ensure_identity(work: Path, candidate: Path, scenario: str, world=WORLD, total_steps=TOTAL_STEPS):
    path = work / "approved-experiment-identity.json"
    if not path.exists():
        if scenario in {"resume", "incomplete"} or _has_checkpoint_evidence(work):
            raise ValueError("approved experiment identity is missing for existing checkpoint state")
        work.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(experiment_identity(candidate, world=world, total_steps=total_steps), sort_keys=True, indent=2))
    else:
        identity = json.loads(path.read_text())
        _validate_identity_request(identity, candidate, world=world, total_steps=total_steps)
    return path


def oracle():
    """Independent NumPy derivation of sampler, weighted gradients and momentum."""
    w = np.zeros(2, np.float64)
    m = np.asarray([0.12, -0.07], np.float64)
    key, epoch, cursor = 17, 0, 0
    rows = []
    for step in range(1, TOTAL_STEPS + 1):
        if cursor >= len(X):
            key = (key + 0x9E3779B9) % (1 << 32)
            epoch += 1
            cursor = 0
        # Independent ordering derivation uses hex digests and Python's key sort.
        order = [i for _, i in sorted((hashlib.sha256(f"{key}:{epoch}:{i}".encode()).hexdigest(), i)
                                      for i in range(len(X)))]
        ids = order[cursor:cursor + 3]
        cursor += len(ids)
        residual = X[ids].astype(np.float64) @ w - Y[ids].astype(np.float64)
        loss = float(np.mean(residual ** 2))
        grad_sum = 2.0 * X[ids].astype(np.float64).T @ residual
        grad = grad_sum / len(ids)
        m = 0.8 * m + grad
        w = w - 0.05 * m
        ownership = [ids[:2], ids[2:]]
        rows.append({"step": step, "ids": np.asarray(ids, np.int32), "ownership": ownership, "loss": loss, "weights": w.copy(), "momentum": m.copy(),
                     "key": key, "epoch": epoch, "cursor": cursor})
    return rows


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Cluster:
    def __init__(self, work: Path, candidate: Path, scenario: str, total_steps=TOTAL_STEPS, fault="none"):
        self.work, self.candidate, self.scenario, self.total_steps = work, candidate, scenario, total_steps
        self.fault = fault
        self.attempt_id = uuid.uuid4().hex
        self.work.mkdir(parents=True, exist_ok=True)
        identity_path = ensure_identity(self.work, candidate, scenario, total_steps=total_steps)
        port = _free_port()
        self.address = f"127.0.0.1:{port}"
        self.procs = []
        for rank in range(WORLD):
            log = (self.work / f"rank-{rank}.log").open("wb")
            env = dict(os.environ)
            env.update({"JAX_NUM_CPU_DEVICES": "1", "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
                        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
            cmd = [PYTHON, str(WORKER), "--rank", str(rank), "--world", str(WORLD), "--coordinator", self.address,
                   "--work", str(self.work), "--identity", str(identity_path),
                   "--candidate", str(candidate), "--scenario", scenario,
                   "--attempt-id", self.attempt_id, "--total-steps", str(total_steps), "--fault", fault]
            proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
            self.procs.append((proc, log))

    def wait_file(self, name: str, timeout=35):
        until = time.monotonic() + timeout
        marker = self.work / name
        while time.monotonic() < until:
            if marker.exists():
                return json.loads(marker.read_text())
            if any(p.poll() is not None for p, _ in self.procs):
                raise RuntimeError(f"worker exited before {name}; logs={self.logs()}")
            time.sleep(0.04)
        raise TimeoutError(f"bounded wait expired for {name}; logs={self.logs()}")

    def kill_rank(self, rank: int):
        proc = self.procs[rank][0]
        pid = proc.pid
        if proc.poll() is None:
            proc.send_signal(signal.SIGKILL)
        try:
            code = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = proc.wait(timeout=5)
        return {"rank": rank, "pid": pid, "signal": "SIGKILL", "returncode": code}

    def stop_survivors(self, timeout=10):
        (self.work / "stop.json").write_text("{}")
        results = []
        for rank, (proc, log) in enumerate(self.procs):
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    code = proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill(); code = proc.wait(timeout=3)
            log.close()
            results.append({"rank": rank, "pid": proc.pid, "returncode": code})
        return results

    def wait_all(self, timeout=45):
        until = time.monotonic() + timeout
        results = []
        for rank, (proc, log) in enumerate(self.procs):
            left = max(0.1, until - time.monotonic())
            try:
                code = proc.wait(timeout=left)
            except subprocess.TimeoutExpired:
                self.stop_survivors()
                raise TimeoutError(f"cluster timeout; logs={self.logs()}")
            log.close()
            results.append({"rank": rank, "pid": proc.pid, "returncode": code})
        if any(x["returncode"] != 0 for x in results):
            raise RuntimeError(f"worker failure {results}; logs={self.logs()}")
        return results

    def logs(self):
        out = {}
        for rank in range(WORLD):
            p = self.work / f"rank-{rank}.log"
            out[str(rank)] = p.read_text(errors="replace")[-5000:] if p.exists() else "missing"
        return out


def run_full(work: Path, candidate: Path, scenario="baseline", fault="none"):
    cluster = Cluster(work, candidate, scenario, fault=fault)
    statuses = cluster.wait_all()
    infos = [json.loads((work / f"rank-info-{cluster.attempt_id}-{rank}.json").read_text()) for rank in range(WORLD)]
    return {"attempt_id": cluster.attempt_id, "processes": statuses, "rank_info": infos}


def kill_at_boundary(work: Path, candidate: Path, incomplete=False):
    cluster = Cluster(work, candidate, "boundary")
    ready = cluster.wait_file("boundary-ready.json")
    survivor_before = {"rank": 0, "pid": cluster.procs[0][0].pid, "alive": cluster.procs[0][0].poll() is None}
    dead = cluster.kill_rank(1)
    stopped = cluster.stop_survivors()
    evidence = {"ready": ready, "survivor_before_stop": survivor_before, "termination": dead,
                "stop_results": stopped, "all_processes": [{"rank": r, "pid": p.pid, "returncode": p.returncode}
                                                             for r, (p, _) in enumerate(cluster.procs)],
                "attempt_id": cluster.attempt_id,
                "rank_info": [json.loads((work / f"rank-info-{cluster.attempt_id}-{r}.json").read_text()) for r in range(WORLD)],
                "logs": cluster.logs()}
    if not incomplete:
        return evidence

    failed = Cluster(work, candidate, "incomplete")
    staged = failed.wait_file("incomplete-ready.json")
    survivor = {"rank": 0, "pid": failed.procs[0][0].pid, "alive": failed.procs[0][0].poll() is None}
    killed = failed.kill_rank(1)
    stopped_failed = failed.stop_survivors()
    cur = json.loads((work / "checkpoints" / "CURRENT.json").read_text())
    gen2 = work / "checkpoints" / "attempts" / failed.attempt_id / "gen-0002"
    evidence.update({"incomplete_staged": staged, "incomplete_survivor": survivor, "incomplete_termination": killed,
                     "current_after_failure": cur["generation"], "current_step_after_failure": cur["step"],
                     "incomplete_attempt_id": failed.attempt_id,
                     "rank_info": [json.loads((work / f"rank-info-{failed.attempt_id}-{r}.json").read_text()) for r in range(WORLD)],
                     "rank0_tmp_exists": (gen2 / "rank-0.json.tmp").exists(),
                     "rank1_final_exists": (gen2 / "rank-1.json").exists(),
                     "manifest2_exists": (gen2 / "manifest.json").exists(), "stop_results": stopped_failed,
                     "all_processes": [{"rank": r, "pid": p.pid, "returncode": p.returncode}
                                       for r, (p, _) in enumerate(failed.procs)], "logs": failed.logs()})
    return evidence


def verify_final(work: Path, expected, source_hash):
    result = json.loads((work / "final.json").read_text())
    st = result["state"]
    exp = expected[-1]
    np.testing.assert_allclose(st["weights"], exp["weights"], rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(st["momentum"], exp["momentum"], rtol=2e-5, atol=2e-6)
    for name in ("step", "key", "epoch", "cursor"):
        assert st[name] == exp[name], (name, st[name], exp[name])
    current = json.loads((work / "checkpoints" / "CURRENT.json").read_text())
    assert current["rank_ids"] == [0, 1] and current["world"] == WORLD
    for rank in range(WORLD):
        assert str(rank) in current["rank_hashes"]
        assert (work / "checkpoints" / current["generation"] / f"rank-{rank}.json").is_file()
    for row in expected:
        trans = json.loads((work / f"transition-{row['step']}.json").read_text())
        assert trans["ids"] == row["ids"].tolist()
        assert trans["rank_ids"]["0"] == list(row["ownership"][0])
        assert trans["rank_ids"]["1"] == list(row["ownership"][1])
        assert trans["counts"] == [len(x) for x in row["ownership"]]
        np.testing.assert_allclose(trans["global_loss"], row["loss"], rtol=2e-5, atol=2e-6)
        assert sum(trans["counts"]) == len(row["ids"])
        assert set(trans["rank_ids"]["0"]).isdisjoint(trans["rank_ids"]["1"])
        assert sorted(trans["ids"]) == sorted(set(trans["ids"]))
        after = trans["state_after"]
        np.testing.assert_allclose(after["weights"], row["weights"], rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(after["momentum"], row["momentum"], rtol=2e-5, atol=2e-6)
        for name in ("key", "epoch", "cursor", "step"):
            assert after[name] == row[name], (name, after[name], row[name])
    return {"final_step": st["step"], "weights": st["weights"], "momentum": st["momentum"],
            "current_generation": current["generation"], "world": current["world"], "candidate_sha256": source_hash}


def ownership_problem(row, observed):
    ids = observed["ids"]
    rank_ids = observed["rank_ids"]
    flattened = rank_ids["0"] + rank_ids["1"]
    if set(rank_ids["0"]).intersection(rank_ids["1"]):
        return "duplicate sample ownership"
    if sorted(flattened) != sorted(ids) or len(flattened) != len(ids):
        return "rank ownership does not exactly partition global IDs"
    if rank_ids["0"] != list(row["ownership"][0]) or rank_ids["1"] != list(row["ownership"][1]):
        return "wrong deterministic rank-to-ID assignment"
    return None


def run_all(candidate: Path, work: Path):
    candidate = candidate.resolve()
    expected = oracle()
    source_hash = sha(candidate)
    identity = experiment_identity(candidate)
    receipt = {"schema": "l10b-local-multiprocess-result-v2", "python": PYTHON,
               "runtime": identity["runtime"], "approved_identity": identity,
               "world": WORLD, "devices_per_process": 1, "total_steps": TOTAL_STEPS,
               "candidate": str(candidate), "candidate_sha256": source_hash,
               "source_hashes": identity["source_hashes"],
               "oracle": [{"ids": x["ids"].tolist(), "counts": [len(y) for y in x["ownership"]], "loss": x["loss"],
                           "weights": x["weights"].tolist(), "momentum": x["momentum"].tolist()} for x in expected],
               "events": []}
    receipt["source_mutation_negative"] = run_source_mutation_negative(candidate, work)
    receipt["preflight_negative_controls"] = {
        "missing_sidecar": run_missing_identity_negative(candidate, work),
        "changed_config": run_config_change_negative(candidate, work),
    }
    base = work / "baseline"
    receipt["events"].append({"kind": "uninterrupted_cluster", "processes": run_full(base, candidate),
                              "verified": verify_final(base, expected, source_hash)})
    committed = work / "committed-loss"
    death = kill_at_boundary(committed, candidate)
    receipt["events"].append({"kind": "observed_kill_after_commit", **death})
    recovered = work / "committed-loss"
    receipt["events"].append({"kind": "fresh_cluster_restore_after_commit", "processes": run_full(recovered, candidate, "resume"),
                               "verified": verify_final(recovered, expected, source_hash)})
    incomplete = work / "incomplete-loss"
    # First all-rank committed generation, then a real SIGKILL while rank 0's
    # next generation is staged but rank 1 has not published its shard.
    bc = Cluster(incomplete, candidate, "boundary")
    ready = bc.wait_file("boundary-ready.json")
    dead = bc.kill_rank(1)
    bc_stops = bc.stop_survivors()
    failed = Cluster(incomplete, candidate, "incomplete")
    staged = failed.wait_file("incomplete-ready.json")
    alive = failed.procs[0][0].poll() is None
    killed = failed.kill_rank(1)
    failed_stops = failed.stop_survivors()
    gen2 = incomplete / "checkpoints" / "attempts" / failed.attempt_id / "gen-0002"
    current = json.loads((incomplete / "checkpoints" / "CURRENT.json").read_text())
    partial = {"kind": "observed_kill_during_incomplete_write", "committed_boundary": ready,
               "committed_rank1_kill": dead, "staged_boundary": staged, "rank0_survivor_alive": alive,
               "rank1_kill": killed, "current_after_failure": current["generation"],
               "current_step_after_failure": current["step"], "incomplete_attempt_id": failed.attempt_id,
               "incomplete_rank_info": [json.loads((incomplete / f"rank-info-{failed.attempt_id}-{r}.json").read_text())
                                         for r in range(WORLD)],
               "committed_stop_results": bc_stops, "incomplete_stop_results": failed_stops,
               "committed_processes": [{"rank": r, "pid": p.pid, "returncode": p.returncode}
                                       for r, (p, _) in enumerate(bc.procs)],
               "committed_attempt_id": bc.attempt_id,
               "committed_rank_info": [json.loads((incomplete / f"rank-info-{bc.attempt_id}-{r}.json").read_text())
                                       for r in range(WORLD)],
               "incomplete_processes": [{"rank": r, "pid": p.pid, "returncode": p.returncode}
                                        for r, (p, _) in enumerate(failed.procs)],
               "rank0_temp_exists": (gen2 / "rank-0.json.tmp").exists(),
               "rank1_final_exists": (gen2 / "rank-1.json").exists(),
               "manifest2_exists": (gen2 / "manifest.json").exists(), "rank0_log": failed.logs()["0"], "rank1_log": failed.logs()["1"]}
    assert partial["current_step_after_failure"] == 1
    assert partial["rank0_temp_exists"] and not partial["rank1_final_exists"] and not partial["manifest2_exists"]
    receipt["events"].append(partial)
    receipt["events"].append({"kind": "fresh_cluster_restore_after_incomplete_write",
                              "processes": run_full(incomplete, candidate, "resume"),
                              "verified": verify_final(incomplete, expected, source_hash)})
    # Owned semantic mutations exercise the independent acceptance boundaries.
    duplicate = work / "negative-duplicate-shard"
    duplicate_processes = run_full(duplicate, candidate, fault="duplicate-shard")
    duplicate_row = json.loads((duplicate / "transition-1.json").read_text())
    duplicate_reason = ownership_problem(expected[0], duplicate_row)
    assert duplicate_reason == "duplicate sample ownership"
    receipt["events"].append({"kind": "negative_duplicate_shard_rejected", "processes": duplicate_processes,
                              "observed_ids": duplicate_row["rank_ids"], "reason": duplicate_reason})

    local_mean = work / "negative-local-mean"
    run_full(local_mean, candidate, fault="local-mean")
    local_state = json.loads((local_mean / "final.json").read_text())["state"]
    local_gap = float(np.max(np.abs(np.asarray(local_state["weights"]) - expected[-1]["weights"])))
    assert local_gap > 1e-3
    receipt["events"].append({"kind": "negative_local_mean_rejected", "final_weight_error": local_gap})

    missing = work / "negative-early-commit"
    missing_cluster = Cluster(missing, candidate, "baseline", fault="omit-rank")
    try:
        missing_cluster.wait_all(timeout=25)
        raise AssertionError("early-commit mutation unexpectedly completed")
    except RuntimeError as err:
        logs = missing_cluster.logs()
        message = "refusing incomplete all-rank commit; missing ranks ['1']"
        assert all(message in logs[str(rank)] for rank in range(WORLD)), logs
        assert not (missing / "checkpoints" / "CURRENT.json").exists()
        receipt["events"].append({"kind": "negative_early_commit_rejected", "expected_error": message,
                                  "attempt_id": missing_cluster.attempt_id,
                                  "rank_info": [json.loads((missing / f"rank-info-{missing_cluster.attempt_id}-{r}.json").read_text()) for r in range(WORLD)],
                                  "processes": [{"rank": rank, "pid": proc.pid, "returncode": proc.returncode}
                                                for rank, (proc, _) in enumerate(missing_cluster.procs)], "logs": logs})

    rehashed = work / "negative-rechecksummed-shape"
    prep = Cluster(rehashed, candidate, "boundary")
    prep.wait_file("boundary-ready.json")
    prep.kill_rank(1)
    prep.stop_survivors()
    current_path = rehashed / "checkpoints" / "CURRENT.json"
    manifest = json.loads(current_path.read_text())
    generation_dir = rehashed / "checkpoints" / manifest["generation"]
    for rank in range(WORLD):
        part = generation_dir / f"rank-{rank}.json"
        payload = json.loads(part.read_text())
        payload["weights"] = [0.0]  # deliberately incompatible shape
        part.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        manifest["rank_hashes"][str(rank)] = sha(part)
    (generation_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    current_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    bad_restore = Cluster(rehashed, candidate, "resume")
    try:
        bad_restore.wait_all(timeout=25)
        raise AssertionError("rechecksummed incompatible checkpoint schema unexpectedly restored")
    except RuntimeError:
        logs = bad_restore.logs()
        error = "checkpoint parameter/optimizer schema mismatch"
        assert all(error in logs[str(rank)] for rank in range(WORLD)), logs
        assert json.loads(current_path.read_text())["step"] == 1
        receipt["events"].append({"kind": "negative_rechecksummed_shape_rejected", "expected_error": error,
                                  "rank_info": [json.loads((rehashed / f"rank-info-{bad_restore.attempt_id}-{r}.json").read_text())
                                                for r in range(WORLD)],
                                  "processes": [{"rank": r, "pid": p.pid, "returncode": p.returncode}
                                                for r, (p, _) in enumerate(bad_restore.procs)], "logs": logs})

    stale = work / "negative-stale-parent"
    prep = Cluster(stale, candidate, "boundary")
    prep.wait_file("boundary-ready.json")
    prep.kill_rank(1)
    prep.stop_survivors()
    current_path = stale / "checkpoints" / "CURRENT.json"
    manifest = json.loads(current_path.read_text())
    manifest["parent_step"] = 41
    (stale / "checkpoints" / manifest["generation"] / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    current_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    bad_restore = Cluster(stale, candidate, "resume")
    try:
        bad_restore.wait_all(timeout=25)
        raise AssertionError("stale parent generation unexpectedly restored")
    except RuntimeError:
        logs = bad_restore.logs()
        error = "checkpoint parent step is stale"
        assert all(error in logs[str(rank)] for rank in range(WORLD)), logs
        receipt["events"].append({"kind": "negative_stale_parent_rejected", "expected_error": error,
                                  "processes": [{"rank": r, "pid": p.pid, "returncode": p.returncode}
                                                for r, (p, _) in enumerate(bad_restore.procs)], "logs": logs})

    receipt["events"].append(run_restore_manifest_negative(
        candidate, work, "negative-changed-world", "checkpoint world size identity mismatch"))
    receipt["events"].append(run_restore_manifest_negative(
        candidate, work, "negative-changed-rank-ownership", "checkpoint is missing or reorders a rank shard"))

    for fault in ("reset-momentum", "stale-key", "stale-cursor"):
        bad = work / f"negative-{fault}"
        preparer = Cluster(bad, candidate, "boundary")
        preparer.wait_file("boundary-ready.json")
        killed = preparer.kill_rank(1)
        preparer.stop_survivors()
        run_full(bad, candidate, "resume", fault=fault)
        bad_state = json.loads((bad / "final.json").read_text())["state"]
        if fault == "reset-momentum":
            gap = float(np.max(np.abs(np.asarray(bad_state["weights"]) - expected[-1]["weights"])))
            assert gap > 1e-3
        else:
            first = json.loads((bad / "transition-2.json").read_text())
            assert first["ids"] != expected[1]["ids"].tolist()
            gap = float(np.max(np.abs(np.asarray(bad_state["weights"]) - expected[-1]["weights"])))
            assert gap > 1e-3
        receipt["events"].append({"kind": f"negative_{fault}_rejected", "source_termination": killed,
                                  "final_weight_error": gap, "next_ids": first["ids"] if fault != "reset-momentum" else None})
    work.mkdir(parents=True, exist_ok=True)
    return receipt


def run_checkpoint(candidate: Path, work: Path):
    """Run uninterrupted and fresh-process checkpoint replay checks."""
    expected = oracle()
    source_hash = sha(candidate)
    base = work / "uninterrupted"
    base_processes = run_full(base, candidate)
    base_verified = verify_final(base, expected, source_hash)
    replay = work / "committed-replay"
    cluster = Cluster(replay, candidate, "boundary")
    ready = cluster.wait_file("boundary-ready.json")
    killed = cluster.kill_rank(1)
    stopped = cluster.stop_survivors()
    restored_processes = run_full(replay, candidate, "resume")
    restored_verified = verify_final(replay, expected, source_hash)
    next_state = json.loads((replay / "transition-2.json").read_text())
    assert next_state["ids"] == expected[1]["ids"].tolist()
    assert next_state["state_before"]["step"] == 1
    np.testing.assert_allclose(next_state["state_after"]["weights"], expected[1]["weights"], rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(next_state["state_after"]["momentum"], expected[1]["momentum"], rtol=2e-5, atol=2e-6)
    for name in ("key", "epoch", "cursor", "step"):
        assert next_state["state_after"][name] == expected[1][name]
    mutation = run_source_mutation_negative(candidate, work)
    changed_world = run_restore_manifest_negative(
        candidate, work, "negative-changed-world", "checkpoint world size identity mismatch")
    changed_ownership = run_restore_manifest_negative(
        candidate, work, "negative-changed-rank-ownership", "checkpoint is missing or reorders a rank shard")
    missing_identity = run_missing_identity_negative(candidate, work)
    changed_config = run_config_change_negative(candidate, work)
    identity = experiment_identity(candidate)
    return {"schema": "l10b-checkpoint-replay-stage-v1",
            "uninterrupted": {"processes": base_processes, "verified": base_verified},
            "fresh_restore": {"boundary": ready, "termination": killed, "stop_results": stopped,
                              "processes": restored_processes, "verified": restored_verified,
                              "next_ids": next_state["ids"], "next_state": next_state["state_after"]},
            "source_mutation_negative": mutation, "changed_world_negative": changed_world,
            "changed_rank_ownership_negative": changed_ownership,
            "missing_identity_negative": missing_identity, "changed_config_negative": changed_config,
            "source_hashes": identity["source_hashes"],
            "runtime": identity["runtime"]}


def run_worker_loss(candidate: Path, work: Path):
    """Run committed-boundary and incomplete-write worker-death programs."""
    expected = oracle()
    source_hash = sha(candidate)
    committed = work / "committed-worker-loss"
    death = kill_at_boundary(committed, candidate)
    resumed = run_full(committed, candidate, "resume")
    resumed_verified = verify_final(committed, expected, source_hash)

    incomplete = work / "incomplete-worker-loss"
    first = Cluster(incomplete, candidate, "boundary")
    ready = first.wait_file("boundary-ready.json")
    first_kill = first.kill_rank(1)
    first_stops = first.stop_survivors()
    partial = Cluster(incomplete, candidate, "incomplete")
    staged = partial.wait_file("incomplete-ready.json")
    survivor = {"rank": 0, "pid": partial.procs[0][0].pid,
                "alive": partial.procs[0][0].poll() is None}
    partial_kill = partial.kill_rank(1)
    partial_stops = partial.stop_survivors()
    current = json.loads((incomplete / "checkpoints" / "CURRENT.json").read_text())
    gen2 = incomplete / "checkpoints" / "attempts" / partial.attempt_id / "gen-0002"
    assert current["step"] == 1
    assert (gen2 / "rank-0.json.tmp").exists()
    assert not (gen2 / "rank-1.json").exists() and not (gen2 / "manifest.json").exists()
    restored = run_full(incomplete, candidate, "resume")
    restored_verified = verify_final(incomplete, expected, source_hash)
    identity = experiment_identity(candidate)
    return {"schema": "l10b-worker-loss-stage-v1",
            "source_hashes": identity["source_hashes"], "runtime": identity["runtime"],
            "committed_boundary": {**death, "restart_processes": resumed, "verified": resumed_verified},
            "incomplete_generation": {"boundary": ready, "first_kill": first_kill,
                                       "first_stop_results": first_stops, "staged": staged,
                                       "survivor": survivor, "kill": partial_kill,
                                       "stop_results": partial_stops,
                                       "current_generation": current["generation"], "current_step": current["step"],
                                       "rank0_temp_exists": True, "rank1_final_exists": False,
                                       "manifest_exists": False, "restart_processes": restored,
                                       "verified": restored_verified}}


def run_source_mutation_negative(candidate: Path, work: Path):
    """Reject changed worker bytes against an older caller-frozen identity."""
    copied = work / "source-mutation" / "multiprocess"
    copied.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(HERE, copied, ignore=shutil.ignore_patterns("__pycache__"))
    try:
        copied_candidate = copied / candidate.resolve().relative_to(HERE)
    except ValueError:
        copied_candidate = copied / "approved_external_candidate.py"
        shutil.copyfile(candidate, copied_candidate)
    identity = experiment_identity(copied_candidate, copied)
    identity_path = copied.parent / "approved-experiment-identity.json"
    identity_path.write_text(json.dumps(identity, sort_keys=True, indent=2))
    worker = copied / "worker.py"
    source = worker.read_text()
    old = '"learning_rate": 0.05'
    assert source.count(old) >= 2
    prefix, suffix = source.rsplit(old, 1)
    worker.write_text(prefix + '"learning_rate": 0.1' + suffix)
    env = dict(os.environ)
    env.update({"JAX_NUM_CPU_DEVICES": "1", "XLA_FLAGS": "--xla_force_host_platform_device_count=1"})
    cmd = [PYTHON, str(worker), "--rank", "0", "--world", "2", "--coordinator", "127.0.0.1:1",
           "--work", str(copied.parent), "--identity", str(identity_path), "--candidate", str(copied_candidate),
           "--scenario", "resume", "--attempt-id", "source-mutation", "--total-steps", str(TOTAL_STEPS)]
    run = subprocess.run(cmd, cwd=copied, env=env, capture_output=True, text=True, timeout=15)
    output = run.stdout + run.stderr
    expected = "approved executable source inventory mismatch"
    assert run.returncode != 0 and expected in output, output
    assert not list(copied.parent.glob("rank-info-*.json"))
    assert not (copied.parent / "checkpoints").exists()
    return {"kind": "changed_worker_source_rejected_before_jax_or_restore", "expected_error": expected,
            "returncode": run.returncode, "stdout_stderr": output[-2000:],
            "approved_worker_sha256": identity["source_hashes"]["worker.py"],
            "changed_worker_sha256": sha(worker)}


def run_restore_manifest_negative(candidate: Path, work: Path, mutation: str, expected_error: str):
    root = work / mutation
    prep = Cluster(root, candidate, "boundary")
    prep.wait_file("boundary-ready.json")
    prep.kill_rank(1)
    prep.stop_survivors()
    current_path = root / "checkpoints" / "CURRENT.json"
    manifest = json.loads(current_path.read_text())
    if mutation == "negative-changed-world":
        manifest["world"] = WORLD + 1
    elif mutation == "negative-changed-rank-ownership":
        manifest["rank_ids"] = [manifest["rank_ids"][0], manifest["rank_ids"][0]]
    else:
        raise ValueError(f"unknown restore identity mutation: {mutation}")
    generation_manifest = root / "checkpoints" / manifest["generation"] / "manifest.json"
    generation_manifest.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    current_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    bad = Cluster(root, candidate, "resume")
    try:
        bad.wait_all(timeout=25)
        raise AssertionError(f"{mutation} unexpectedly restored")
    except RuntimeError:
        logs = bad.logs()
        assert all(expected_error in logs[str(rank)] for rank in range(WORLD)), logs
        assert json.loads(current_path.read_text())["step"] == 1
        return {"kind": mutation + "_rejected", "expected_error": expected_error,
                "attempt_id": bad.attempt_id,
                "processes": [{"rank": rank, "pid": proc.pid, "returncode": proc.returncode}
                              for rank, (proc, _) in enumerate(bad.procs)], "logs": logs}


def _file_snapshot(root: Path):
    return {
        path.relative_to(root).as_posix(): sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _committed_fixture(candidate: Path, work: Path):
    cluster = Cluster(work, candidate, "boundary")
    boundary = cluster.wait_file("boundary-ready.json")
    killed = cluster.kill_rank(1)
    stopped = cluster.stop_survivors()
    current_path = work / "checkpoints" / "CURRENT.json"
    assert json.loads(current_path.read_text())["step"] == 1
    return {"boundary": boundary, "kill": killed, "stop_results": stopped,
            "current_path": current_path, "current_bytes": current_path.read_bytes()}


def run_missing_identity_negative(candidate: Path, work: Path):
    root = work / "negative-missing-approved-identity"
    committed = _committed_fixture(candidate, root)
    identity_path = root / "approved-experiment-identity.json"
    identity_path.unlink()
    before = _file_snapshot(root)
    expected_error = "approved experiment identity is missing for existing checkpoint state"
    try:
        ensure_identity(root, candidate, "resume")
        raise AssertionError("missing identity sidecar was silently recreated")
    except ValueError as err:
        assert expected_error in str(err), str(err)
    after = _file_snapshot(root)
    assert not identity_path.exists()
    assert before == after
    assert committed["current_path"].read_bytes() == committed["current_bytes"]
    return {"kind": "missing_identity_sidecar_rejected_before_startup", "expected_error": expected_error,
            "sidecar_created": identity_path.exists(), "filesystem_unchanged": before == after,
            "current_sha256": hashlib.sha256(committed["current_bytes"]).hexdigest(),
            "rank_info_files": sorted(p.name for p in root.glob("rank-info-*.json")),
            "transition_files": sorted(p.name for p in root.glob("transition-*.json"))}


def run_config_change_negative(candidate: Path, work: Path):
    root = work / "negative-changed-config-preflight"
    committed = _committed_fixture(candidate, root)
    identity_path = root / "approved-experiment-identity.json"
    before = _file_snapshot(root)
    attempt_id = "changed-config-preflight"
    env = dict(os.environ)
    env.update({"JAX_NUM_CPU_DEVICES": "1", "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    command = [PYTHON, str(WORKER), "--rank", "0", "--world", str(WORLD), "--coordinator", "127.0.0.1:1",
               "--work", str(root), "--identity", str(identity_path), "--candidate", str(candidate.resolve()),
               "--scenario", "resume", "--attempt-id", attempt_id, "--total-steps", str(TOTAL_STEPS + 1)]
    run = subprocess.run(command, cwd=HERE, env=env, capture_output=True, text=True, timeout=12)
    output = run.stdout + run.stderr
    expected_error = "approved experiment/config/data identity mismatch"
    assert run.returncode != 0 and expected_error in output, output
    after = _file_snapshot(root)
    assert before == after
    assert committed["current_path"].read_bytes() == committed["current_bytes"]
    assert not list(root.glob(f"rank-info-{attempt_id}-*.json"))
    assert not (root / "transition-2.json").exists()
    return {"kind": "changed_total_steps_rejected_before_jax_or_restore", "expected_error": expected_error,
            "requested_total_steps": TOTAL_STEPS + 1, "returncode": run.returncode,
            "stdout_stderr": output[-2000:], "filesystem_unchanged": before == after,
            "current_sha256": hashlib.sha256(committed["current_bytes"]).hexdigest(),
            "rank_info_for_attempt": [], "transition_2_exists": False}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", type=Path, default=HERE / "solution" / "model.py")
    p.add_argument("--work", type=Path, default=Path(tempfile.mkdtemp(prefix="l10b-multiprocess-")))
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = run_all(a.candidate, a.work)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({"status": "PASS", "work": str(a.work), "candidate_sha256": result["candidate_sha256"],
                      "events": [e["kind"] for e in result["events"]]}, sort_keys=True))


if __name__ == "__main__":
    main()
