"""One actual JAX controller in the L10B local CPU experiment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import importlib.metadata
import os
import struct
import sys
import time
from pathlib import Path
import platform


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world", type=int, required=True)
    p.add_argument("--coordinator", required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--identity", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--scenario", choices=("baseline", "boundary", "incomplete", "resume"), required=True)
    p.add_argument("--attempt-id", required=True)
    p.add_argument("--fault", choices=("none", "duplicate-shard", "local-mean", "reset-momentum", "stale-key", "stale-cursor", "omit-rank"), default="none")
    p.add_argument("--total-steps", type=int, default=4)
    a = p.parse_args()

    # Check the caller-frozen experiment contract before importing JAX,
    # initializing rendezvous, loading learner code, or reading a checkpoint.
    identity = json.loads(a.identity.read_text())
    expected_x = struct.pack("=10f", 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0, -1.0, 2.0)
    expected_y = struct.pack("=5f", 0.5, -0.25, 0.75, 1.5, -1.0)
    expected_data_hash = hashlib.sha256(expected_x + expected_y).hexdigest()
    expected_experiment = {
        "schema": "l10b-multiprocess-v1",
        "world": a.world,
        "per_rank": 2,
        "width": 2,
        "seed": 17,
        "learning_rate": 0.05,
        "momentum_rate": 0.8,
        "steps": a.total_steps,
        "data_hash": expected_data_hash,
        "sampler": "sha256-key-epoch-id-sort-v1",
    }
    if (
        identity.get("schema") != "l10b-approved-experiment-identity-v2"
        or identity.get("experiment") != expected_experiment
        or identity.get("data_sha256") != expected_data_hash
    ):
        raise ValueError("approved experiment/config/data identity mismatch")
    expected_sources = identity["source_hashes"]
    root = Path(__file__).resolve().parent
    actual_sources = {
        path.relative_to(root).as_posix(): digest(path)
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    if actual_sources != expected_sources:
        raise ValueError("approved executable source inventory mismatch")
    runtime = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "jax": importlib.metadata.version("jax"),
        "jaxlib": importlib.metadata.version("jaxlib"),
        "numpy": importlib.metadata.version("numpy"),
    }
    if runtime != identity["runtime"]:
        raise ValueError("approved runtime identity mismatch")
    if str(a.candidate.resolve()) != identity["candidate"]:
        raise ValueError("approved candidate path identity mismatch")
    if digest(a.candidate.resolve()) != identity["candidate_sha256"]:
        raise ValueError("approved candidate source identity mismatch")

    import jax

    # Distributed JAX must initialize before querying devices or making arrays.
    jax.distributed.initialize(
        coordinator_address=a.coordinator,
        num_processes=a.world,
        process_id=a.rank,
        local_device_ids=[0],
        initialization_timeout=20,
        heartbeat_timeout_seconds=8,
        shutdown_timeout_seconds=3,
    )
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import multihost_utils
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    spec = importlib.util.spec_from_file_location("candidate_model", a.candidate)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import candidate")
    model = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = model
    spec.loader.exec_module(model)

    if jax.process_count() != a.world or jax.local_device_count() != 1:
        raise RuntimeError(f"unexpected topology: processes={jax.process_count()}, local_devices={jax.local_device_count()}")
    import jaxlib
    atomic_json(a.work / f"rank-info-{a.attempt_id}-{a.rank}.json", {
        "pid": os.getpid(), "process_index": jax.process_index(), "process_count": jax.process_count(),
        "local_device_count": jax.local_device_count(), "global_device_count": jax.device_count(),
        "local_devices": [str(d) for d in jax.local_devices()], "global_devices": [str(d) for d in jax.devices()],
        "backend": jax.default_backend(), "jax": jax.__version__, "jaxlib": jaxlib.__version__,
        "attempt_id": a.attempt_id,
    })
    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    row_sharding = NamedSharding(mesh, P("data", None))
    vector_sharding = NamedSharding(mesh, P("data",))
    scalar_sharding = NamedSharding(mesh, P("data",))
    world = a.world
    per_rank = 2
    width = 2
    # This fixture is deterministic and intentionally uneven: global batches
    # have 3 then 2 examples, with rank 1 empty in the second batch.
    all_x = np.asarray([[1., 0.], [0., 1.], [1., 1.], [2., -1.], [-1., 2.]], np.float32)
    all_y = np.asarray([0.5, -0.25, 0.75, 1.5, -1.0], np.float32)
    data_hash = hashlib.sha256(all_x.tobytes() + all_y.tobytes()).hexdigest()
    config = {
        "schema": "l10b-multiprocess-v1", "world": world, "per_rank": per_rank,
        "width": width, "seed": 17, "learning_rate": 0.05, "momentum_rate": 0.8,
        "steps": a.total_steps, "data_hash": data_hash,
        "candidate_sha256": digest(a.candidate), "sampler": "sha256-key-epoch-id-sort-v1",
        "source_hashes": identity["source_hashes"], "runtime": runtime,
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    ckpt = a.work / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)

    def place(local, sharding, global_shape):
        return jax.make_array_from_process_local_data(sharding, local, global_shape=global_shape)

    def order_for(key: int, epoch: int):
        # Learner candidate sampler: stable hash-sort; the checker independently
        # derives the same order with a separate implementation.
        rows = [(hashlib.sha256(f"{key}:{epoch}:{i}".encode()).digest(), i) for i in range(len(all_x))]
        return np.asarray([i for _, i in sorted(rows)], np.int32)

    def initial():
        return {"weights": np.zeros(width, np.float32), "momentum": np.asarray([0.12, -0.07], np.float32),
                "key": 17, "epoch": 0, "cursor": 0, "step": 0}

    def load_current():
        manifest_path = ckpt / "CURRENT.json"
        if not manifest_path.exists():
            raise FileNotFoundError("fresh-cluster restore requires committed CURRENT.json")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != "l10b-generation-v1" or manifest["config_hash"] != config_hash:
            raise ValueError("checkpoint experiment/runtime source identity mismatch")
        if manifest.get("world") != world:
            raise ValueError("checkpoint world size identity mismatch")
        expected_ranks = list(range(world))
        if manifest.get("rank_ids") != expected_ranks or sorted(manifest.get("rank_hashes", {})) != [str(r) for r in expected_ranks]:
            raise ValueError("checkpoint is missing or reorders a rank shard")
        generation_path = Path(manifest["generation"])
        if generation_path.is_absolute() or ".." in generation_path.parts or len(generation_path.parts) != 3 or generation_path.parts[0] != "attempts":
            raise ValueError("checkpoint generation path is invalid")
        generation_manifest_path = ckpt / generation_path / "manifest.json"
        if not generation_manifest_path.is_file() or json.loads(generation_manifest_path.read_text()) != manifest:
            raise ValueError("CURRENT pointer does not match its committed generation manifest")
        if int(manifest.get("parent_step", -1)) != int(manifest["step"]) - 1:
            raise ValueError("checkpoint parent step is stale")
        parent_generation = manifest.get("parent_generation")
        if int(manifest["step"]) == 1:
            if parent_generation is not None:
                raise ValueError("initial generation has an unexpected parent")
        else:
            parent_path = Path(parent_generation or "")
            if parent_path.is_absolute() or ".." in parent_path.parts or len(parent_path.parts) != 3 or parent_path.parts[0] != "attempts":
                raise ValueError("checkpoint parent generation is invalid")
            parent_manifest_path = ckpt / parent_path / "manifest.json"
            if not parent_manifest_path.is_file():
                raise ValueError("checkpoint parent generation is missing")
            parent_manifest = json.loads(parent_manifest_path.read_text())
            if parent_manifest.get("step") != int(manifest["step"]) - 1 or parent_manifest.get("config_hash") != config_hash or parent_manifest.get("world") != world:
                raise ValueError("checkpoint parent generation is stale or incompatible")
        all_states = []
        for rank in expected_ranks:
            rank_file = ckpt / generation_path / f"rank-{rank}.json"
            if not rank_file.is_file() or digest(rank_file) != manifest["rank_hashes"][str(rank)]:
                raise ValueError(f"checkpoint rank {rank} is missing or hash-mismatched")
            rank_state = json.loads(rank_file.read_text())
            if rank_state.get("rank") != rank or rank_state.get("config_hash") != config_hash or rank_state.get("attempt_id") != manifest.get("attempt_id"):
                raise ValueError(f"checkpoint rank {rank} ownership identity mismatch")
            all_states.append(rank_state)
        common = ("weights", "momentum", "key", "epoch", "cursor", "step")
        if any(any(rank_state.get(k) != all_states[0].get(k) for k in common) for rank_state in all_states[1:]):
            raise ValueError("replicated rank state diverges")
        part = ckpt / manifest["generation"] / f"rank-{a.rank}.json"
        state = json.loads(part.read_text())
        if state["config_hash"] != config_hash or state["rank"] != a.rank or state.get("attempt_id") != manifest.get("attempt_id"):
            raise ValueError("rank checkpoint ownership mismatch")
        weights = np.asarray(state["weights"], np.float32)
        momentum = np.asarray(state["momentum"], np.float32)
        if weights.shape != (width,) or momentum.shape != (width,) or not np.isfinite(weights).all() or not np.isfinite(momentum).all():
            raise ValueError("checkpoint parameter/optimizer schema mismatch")
        if int(state["step"]) != int(manifest["step"]) or not 0 <= int(state["cursor"]) <= len(all_x):
            raise ValueError("checkpoint step/cursor schema mismatch")
        restored = {"weights": weights, "momentum": momentum,
                    "key": int(state["key"]), "epoch": int(state["epoch"]), "cursor": int(state["cursor"]), "step": int(state["step"])}
        if a.fault == "reset-momentum": restored["momentum"] = np.zeros_like(restored["momentum"])
        if a.fault == "stale-key": restored["key"] = 999
        if a.fault == "stale-cursor": restored["cursor"] = 0
        return restored

    def save_generation(state, incomplete=False):
        generation = f"attempts/{a.attempt_id}/gen-{state['step']:04d}"
        gen_dir = ckpt / generation
        gen_dir.mkdir(parents=True, exist_ok=True)
        payload = {"config_hash": config_hash, "rank": a.rank, "weights": state["weights"].tolist(),
                   "momentum": state["momentum"].tolist(), "key": state["key"], "epoch": state["epoch"],
                   "cursor": state["cursor"], "step": state["step"], "attempt_id": a.attempt_id}
        if incomplete:
            if a.rank == 0:
                (gen_dir / "rank-0.json.tmp").write_text(json.dumps(payload, sort_keys=True))
                atomic_json(a.work / "incomplete-ready.json", {"rank": 0, "pid": os.getpid(), "step": state["step"]})
                while not (a.work / "stop.json").exists():
                    time.sleep(0.03)
                return
            while not (a.work / "stop.json").exists():
                time.sleep(0.03)
            return

        part = gen_dir / f"rank-{a.rank}.json"
        if not (a.fault == "omit-rank" and a.rank == 1):
            atomic_json(part, payload)
        # All ranks must publish their exact shard before rank zero can commit.
        multihost_utils.sync_global_devices(f"l10b-generation-{state['step']}-rank-files")
        missing = [str(r) for r in range(world) if not (gen_dir / f"rank-{r}.json").is_file()]
        if missing:
            raise RuntimeError(f"refusing incomplete all-rank commit; missing ranks {missing}")
        if a.rank == 0:
            hashes = {str(r): digest(gen_dir / f"rank-{r}.json") for r in range(world)}
            parent_path = None
            current_path = ckpt / "CURRENT.json"
            if current_path.exists():
                parent_path = json.loads(current_path.read_text())["generation"]
            manifest = {"schema": "l10b-generation-v1", "generation": generation, "attempt_id": a.attempt_id,
                        "parent_generation": parent_path, "parent_step": state["step"] - 1,
                        "world": world, "config_hash": config_hash, "rank_hashes": hashes,
                        "rank_ids": list(range(world)), "step": state["step"]}
            atomic_json(gen_dir / "manifest.json", manifest)
            atomic_json(ckpt / "CURRENT.json", manifest)
        multihost_utils.sync_global_devices(f"l10b-generation-{state['step']}-commit-visible")

    def transition(state):
        order = order_for(state["key"], state["epoch"])
        if state["cursor"] >= len(order):
            state = dict(state, key=state["key"] + 0x9E3779B9, epoch=state["epoch"] + 1, cursor=0)
            order = order_for(state["key"], state["epoch"])
        batch = order[state["cursor"]:state["cursor"] + 3]
        lo, hi = a.rank * per_rank, (a.rank + 1) * per_rank
        owned = batch[lo:hi]
        if a.fault == "duplicate-shard" and a.rank == 1 and len(batch):
            owned = batch[:1]
        local_x = np.zeros((per_rank, width), np.float32)
        local_y = np.zeros((per_rank,), np.float32)
        local_valid = np.zeros((per_rank,), np.float32)
        local_ids = np.full((per_rank,), -1, np.int32)
        if len(owned):
            n = len(owned)
            local_x[:n], local_y[:n], local_valid[:n], local_ids[:n] = all_x[owned], all_y[owned], 1., owned
        gx = place(local_x, row_sharding, (world * per_rank, width))
        gy = place(local_y, vector_sharding, (world * per_rank,))
        gm = place(local_valid, vector_sharding, (world * per_rank,))
        gids = place(local_ids, vector_sharding, (world * per_rank,))
        # Local data used below is taken from the actual addressable global shards.
        x, y, valid, ids = (np.asarray(v.addressable_shards[0].data) for v in (gx, gy, gm, gids))
        local_sum, local_grad_sum = model.local_numerator_and_gradient(jnp.asarray(state["weights"]),
                                                                      jnp.asarray(x), jnp.asarray(y), jnp.asarray(valid > 0))
        local_count = jnp.sum(jnp.asarray(valid))
        grad_for_collective = local_grad_sum
        if a.fault == "local-mean":
            grad_for_collective = local_grad_sum / jnp.maximum(local_count, 1.)
        gathered_gradients = multihost_utils.process_allgather(grad_for_collective, tiled=False)
        gathered_counts = multihost_utils.process_allgather(local_count, tiled=False)
        gathered_losses = multihost_utils.process_allgather(local_sum, tiled=False)
        w, m = model.global_transition(jnp.asarray(state["weights"]), jnp.asarray(state["momentum"]),
                                       gathered_gradients, gathered_counts)
        w, m = np.asarray(w), np.asarray(m)
        global_ids = np.asarray(batch, np.int32)
        gathered_ids = np.asarray(multihost_utils.process_allgather(jnp.asarray(ids), tiled=False))
        if a.rank == 0:
            atomic_json(a.work / f"transition-{state['step'] + 1}.json", {"ids": global_ids.tolist(),
                "rank_ids": {str(r): [int(i) for i in gathered_ids[r].tolist() if int(i) >= 0] for r in range(world)},
                "counts": [sum(int(i) >= 0 for i in gathered_ids[r]) for r in range(world)],
                "local_ids": ids.tolist(), "local_count": int(local_count),
                "gathered_counts": np.asarray(gathered_counts).reshape(-1).tolist(),
                "global_loss": float(np.asarray(gathered_losses).sum() / np.asarray(gathered_counts).sum()),
                "state_before": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in state.items()},
                "state_after": {"weights": w.tolist(), "momentum": m.tolist(),
                                "key": state["key"], "epoch": state["epoch"],
                                "cursor": state["cursor"] + len(batch), "step": state["step"] + 1}})
        return {"weights": w, "momentum": m, "key": state["key"], "epoch": state["epoch"],
                "cursor": state["cursor"] + len(batch), "step": state["step"] + 1}

    if a.scenario in {"resume", "incomplete"}:
        state = load_current()
    else:
        state = initial()
    start_step = state["step"]
    if a.scenario == "boundary":
        state = transition(state)
        save_generation(state)
        if a.rank == 0:
            atomic_json(a.work / "boundary-ready.json", {"pid": os.getpid(), "step": state["step"]})
        while not (a.work / "stop.json").exists():
            time.sleep(0.03)
        return 0
    if a.scenario == "incomplete":
        state = transition(state)
        save_generation(state, incomplete=True)
        return 0
    while state["step"] < a.total_steps:
        state = transition(state)
        save_generation(state)
    if a.rank == 0:
        atomic_json(a.work / "final.json", {"state": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in state.items()},
                                               "pids": [os.getpid()], "start_step": start_step})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
