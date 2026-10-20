"""Qualify multi-process, sixteen-device v5e collectives and checkpoint replay.

Run the same command on every TPU VM worker. This bounded synthetic workload
is an integration target, not a model-scale or production-training benchmark.
"""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import urllib.parse

GLOBAL_ROWS = 64
FEATURES = 8
TOTAL_DEVICES = 16
CHECKPOINT_STEP = 4
FINAL_STEP = 8
INITIALIZATION_TIMEOUT_SECONDS = 180
HEARTBEAT_TIMEOUT_SECONDS = 120
SHUTDOWN_TIMEOUT_SECONDS = 60


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="unique GCS run prefix, for example gs://bucket/private/run-id",
    )
    parser.add_argument("--expected-processes", type=int, default=4)
    parser.add_argument("--expected-local-devices", type=int, default=4)
    args = parser.parse_args()
    if args.expected_processes < 2 or args.expected_local_devices < 1:
        parser.error("multi-host qualification requires at least two processes")
    if args.expected_processes * args.expected_local_devices != TOTAL_DEVICES:
        parser.error("expected processes × local devices must equal exactly 16")
    if GLOBAL_ROWS % args.expected_processes:
        parser.error("64 fixture rows must divide evenly across expected processes")
    parsed = urllib.parse.urlparse(args.checkpoint)
    if (
        parsed.scheme != "gs"
        or not parsed.netloc
        or not parsed.path.strip("/")
        or parsed.query
        or parsed.fragment
    ):
        parser.error("--checkpoint must be a unique gs://bucket/run-prefix URI")
    return args


def require_tpu_environment():
    if os.environ.get("JAX_PLATFORMS") != "tpu":
        raise RuntimeError(
            "Set JAX_PLATFORMS=tpu before running; CPU fallback is disabled"
        )


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def numpy_fixture():
    import numpy as np

    rng = np.random.default_rng(173)
    x = rng.normal(size=(GLOBAL_ROWS, FEATURES)).astype(np.float32)
    truth = np.linspace(-0.3, 0.4, FEATURES, dtype=np.float32)
    noise = np.random.default_rng(991).normal(0, 0.01, GLOBAL_ROWS).astype(np.float32)
    y = x @ truth + noise
    return x, y


def numpy_transition(weights, momentum, x, y):
    import numpy as np

    residual = x @ weights - y
    gradient = np.float32(2.0 / GLOBAL_ROWS) * (x.T @ residual)
    next_momentum = np.float32(0.8) * momentum + gradient
    next_weights = weights - np.float32(0.04) * next_momentum
    loss = np.mean(residual * residual, dtype=np.float32)
    return (
        next_weights.astype(np.float32),
        next_momentum.astype(np.float32),
        np.float32(loss),
    )


def assert_array_equal(actual, expected, name, np):
    np.testing.assert_allclose(
        np.asarray(actual), expected, rtol=3e-5, atol=3e-6, err_msg=name
    )


def run(args, jax):
    import gcsfs
    import jax.numpy as jnp
    import numpy as np
    import orbax.checkpoint as ocp
    from jax.experimental import multihost_utils
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    storage_client = f"{gcsfs.GCSFileSystem.__module__}.{gcsfs.GCSFileSystem.__name__}"
    if storage_client != "gcsfs.core.GCSFileSystem":
        raise RuntimeError("flat-bucket test requires the standard GCSFS client")

    process_index = jax.process_index()
    process_count = jax.process_count()
    local_device_count = jax.local_device_count()
    global_device_count = jax.device_count()
    topology = np.asarray(
        multihost_utils.process_allgather(
            np.asarray(
                [
                    process_index,
                    process_count,
                    local_device_count,
                    global_device_count,
                    args.expected_processes,
                    args.expected_local_devices,
                ],
                dtype=np.int32,
            )
        )
    ).reshape((-1, 6))
    expected_indices = np.arange(args.expected_processes, dtype=np.int32)
    if (
        topology.shape != (args.expected_processes, 6)
        or not np.array_equal(topology[:, 0], expected_indices)
        or not np.all(topology[:, 1] == args.expected_processes)
        or not np.all(topology[:, 2] == args.expected_local_devices)
        or not np.all(topology[:, 3] == TOTAL_DEVICES)
        or not np.all(topology[:, 4:] == topology[0, 4:])
        or jax.default_backend() != "tpu"
    ):
        raise RuntimeError(
            "observed TPU topology differs from explicit expectations: "
            f"{topology.tolist()} backend={jax.default_backend()}"
        )

    checkpoint_identity = hash_bytes(args.checkpoint.encode("utf-8"))
    gathered_checkpoint_ids = np.asarray(
        multihost_utils.process_allgather(
            np.frombuffer(bytes.fromhex(checkpoint_identity), dtype=np.uint8).copy()
        )
    ).reshape((args.expected_processes, 32))
    if not np.all(gathered_checkpoint_ids == gathered_checkpoint_ids[0]):
        raise RuntimeError("workers were given different shared checkpoint prefixes")
    barrier_prefix = "multihost-" + checkpoint_identity[:12]

    devices = jax.devices()
    local_devices = jax.local_devices()
    if len(devices) != TOTAL_DEVICES or any(d.platform != "tpu" for d in devices):
        raise RuntimeError("all sixteen global devices must report platform=tpu")
    if any(d.device_kind != "TPU v5 lite" for d in devices):
        raise RuntimeError("this qualification target requires v5e devices")
    mesh = Mesh(np.asarray(devices), ("data",))
    row_sharding = NamedSharding(mesh, P("data", None))
    label_sharding = NamedSharding(mesh, P("data"))
    replicated_sharding = NamedSharding(mesh, P())

    source_path = Path(__file__).resolve()
    lesson_path = (
        Path(__file__).resolve().parents[2]
        / "phases/09-distributed/04-resilient-distributed-training/code/main.py"
    )
    source_hash = hash_bytes(source_path.read_bytes())
    lesson_hash = hash_bytes(lesson_path.read_bytes())
    source_identity = np.frombuffer(
        bytes.fromhex(source_hash + lesson_hash), dtype=np.uint8
    ).copy()
    gathered_source_identity = np.asarray(
        multihost_utils.process_allgather(source_identity)
    ).reshape((args.expected_processes, 64))
    if not np.all(gathered_source_identity == gathered_source_identity[0]):
        raise RuntimeError("worker source or base lesson differs across processes")
    source_identity_hash = hash_bytes(gathered_source_identity[0].tobytes())
    multihost_utils.sync_global_devices(barrier_prefix + "-source-equality")

    worker_output = args.output.resolve() / f"process-{process_index:03d}"
    worker_output.mkdir(parents=True, exist_ok=False)

    x_full, y_full = numpy_fixture()
    fixture_hash = hash_bytes(x_full.tobytes() + y_full.tobytes())
    local_rows = GLOBAL_ROWS // process_count
    row_start = process_index * local_rows
    row_stop = row_start + local_rows
    x_local = x_full[row_start:row_stop].copy()
    y_local = y_full[row_start:row_stop].copy()
    local_data_hash = hash_bytes(x_local.tobytes() + y_local.tobytes())
    local_identity = np.asarray([row_start, row_stop], dtype=np.int32)
    gathered_ranges = np.asarray(
        multihost_utils.process_allgather(local_identity)
    ).reshape((args.expected_processes, 2))
    expected_ranges = np.asarray(
        [
            [rank * local_rows, (rank + 1) * local_rows]
            for rank in range(args.expected_processes)
        ],
        dtype=np.int32,
    )
    if not np.array_equal(gathered_ranges, expected_ranges):
        raise RuntimeError(
            f"data ownership ranges overlap or have gaps: {gathered_ranges}"
        )
    gathered_fixture_hashes = np.asarray(
        multihost_utils.process_allgather(
            np.frombuffer(bytes.fromhex(fixture_hash), dtype=np.uint8).copy()
        )
    ).reshape((args.expected_processes, 32))
    if not np.all(gathered_fixture_hashes == gathered_fixture_hashes[0]):
        raise RuntimeError("deterministic global fixture differs across processes")
    gathered_local_hashes = np.asarray(
        multihost_utils.process_allgather(
            np.frombuffer(bytes.fromhex(local_data_hash), dtype=np.uint8).copy()
        )
    ).reshape((args.expected_processes, 32))
    local_hashes_by_process = [row.tobytes().hex() for row in gathered_local_hashes]
    if len(set(local_hashes_by_process)) != args.expected_processes:
        raise RuntimeError("processes do not own distinct local data slices")
    multihost_utils.sync_global_devices(barrier_prefix + "-process-local-data-ready")

    # Each process contributes only its own disjoint rows to the global arrays.
    x = jax.make_array_from_process_local_data(
        row_sharding, x_local, global_shape=(GLOBAL_ROWS, FEATURES)
    )
    y = jax.make_array_from_process_local_data(
        label_sharding, y_local, global_shape=(GLOBAL_ROWS,)
    )
    if x.sharding != row_sharding or y.sharding != label_sharding:
        raise RuntimeError("process-local input was not assembled with requested mesh")
    rows_per_device = GLOBAL_ROWS // TOTAL_DEVICES
    if len(x.addressable_shards) != args.expected_local_devices:
        raise RuntimeError("input does not expose the expected local device shards")
    if any(
        shard.data.shape != (rows_per_device, FEATURES)
        for shard in x.addressable_shards
    ):
        raise RuntimeError("local feature shards have unexpected shapes")
    if any(shard.data.shape != (rows_per_device,) for shard in y.addressable_shards):
        raise RuntimeError("local target shards have unexpected shapes")

    def local_update(a, b, weights, momentum, step):
        residual = jnp.matmul(a, weights, precision=jax.lax.Precision.HIGHEST) - b
        local_gradient = jnp.float32(2.0 / GLOBAL_ROWS) * jnp.matmul(
            a.T, residual, precision=jax.lax.Precision.HIGHEST
        )
        gradient = jax.lax.psum(local_gradient, "data")
        loss = jax.lax.psum(
            jnp.sum(residual * residual, dtype=jnp.float32) / GLOBAL_ROWS,
            "data",
        )
        next_momentum = jnp.float32(0.8) * momentum + gradient
        next_weights = weights - jnp.float32(0.04) * next_momentum
        return next_weights, next_momentum, step + jnp.int32(1), loss

    update = jax.jit(
        jax.shard_map(
            local_update,
            mesh=mesh,
            in_specs=(P("data", None), P("data"), P(), P(), P()),
            out_specs=(P(), P(), P(), P()),
        )
    )

    weights = jax.device_put(np.zeros(FEATURES, dtype=np.float32), replicated_sharding)
    momentum = jax.device_put(np.zeros(FEATURES, dtype=np.float32), replicated_sharding)
    step = jax.device_put(np.asarray(0, dtype=np.int32), replicated_sharding)
    state = {"weights": weights, "momentum": momentum, "step": step}
    reference_weights = np.zeros(FEATURES, dtype=np.float32)
    reference_momentum = np.zeros(FEATURES, dtype=np.float32)
    reference_losses = []
    for _ in range(CHECKPOINT_STEP):
        next_weights, next_momentum, next_step, loss = update(
            x, y, state["weights"], state["momentum"], state["step"]
        )
        jax.block_until_ready((next_weights, next_momentum, next_step, loss))
        reference_weights, reference_momentum, reference_loss = numpy_transition(
            reference_weights, reference_momentum, x_full, y_full
        )
        reference_losses.append(float(reference_loss))
        assert_array_equal(next_weights, reference_weights, "step-4 weights", np)
        assert_array_equal(next_momentum, reference_momentum, "step-4 momentum", np)
        np.testing.assert_allclose(
            float(np.asarray(loss)), reference_loss, rtol=3e-5, atol=3e-6
        )
        state = {"weights": next_weights, "momentum": next_momentum, "step": next_step}
    if int(np.asarray(state["step"])) != CHECKPOINT_STEP:
        raise RuntimeError("training did not reach the expected checkpoint step")

    checkpoint_path = args.checkpoint.rstrip("/") + f"/step-{CHECKPOINT_STEP}"
    multihost_utils.sync_global_devices(barrier_prefix + "-before-save")
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(checkpoint_path, state)
        checkpointer.wait_until_finished()
        multihost_utils.sync_global_devices(barrier_prefix + "-after-save")

        abstract_state = {
            "weights": jax.ShapeDtypeStruct(
                (FEATURES,), jnp.float32, sharding=replicated_sharding
            ),
            "momentum": jax.ShapeDtypeStruct(
                (FEATURES,), jnp.float32, sharding=replicated_sharding
            ),
            "step": jax.ShapeDtypeStruct((), jnp.int32, sharding=replicated_sharding),
        }
        multihost_utils.sync_global_devices(barrier_prefix + "-before-restore")
        restored = checkpointer.restore(checkpoint_path, target=abstract_state)
        multihost_utils.sync_global_devices(barrier_prefix + "-after-restore")

    if int(np.asarray(restored["step"])) != CHECKPOINT_STEP:
        raise RuntimeError("restored checkpoint step counter is not four")
    for name in ("weights", "momentum"):
        np.testing.assert_array_equal(
            np.asarray(restored[name]), np.asarray(state[name])
        )
    assert_array_equal(restored["weights"], reference_weights, "restored weights", np)
    assert_array_equal(
        restored["momentum"], reference_momentum, "restored momentum", np
    )

    state = restored
    for _ in range(CHECKPOINT_STEP, FINAL_STEP):
        next_weights, next_momentum, next_step, loss = update(
            x, y, state["weights"], state["momentum"], state["step"]
        )
        jax.block_until_ready((next_weights, next_momentum, next_step, loss))
        reference_weights, reference_momentum, reference_loss = numpy_transition(
            reference_weights, reference_momentum, x_full, y_full
        )
        reference_losses.append(float(reference_loss))
        assert_array_equal(next_weights, reference_weights, "resumed weights", np)
        assert_array_equal(next_momentum, reference_momentum, "resumed momentum", np)
        np.testing.assert_allclose(
            float(np.asarray(loss)), reference_loss, rtol=3e-5, atol=3e-6
        )
        state = {"weights": next_weights, "momentum": next_momentum, "step": next_step}
    if int(np.asarray(state["step"])) != FINAL_STEP:
        raise RuntimeError("resumed training did not reach step eight")

    multihost_utils.sync_global_devices(barrier_prefix + "-complete")
    device_metadata = [
        {
            "processIndex": int(device.process_index),
            "id": int(device.id),
            "platform": device.platform,
            "kind": device.device_kind,
        }
        for device in devices
    ]
    receipt = {
        "schemaVersion": 1,
        "executedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "executionMode": "real-tpu-multi-host-sixteen-device",
        "backend": jax.default_backend(),
        "processIndex": process_index,
        "processCount": process_count,
        "localDeviceCount": local_device_count,
        "globalDeviceCount": global_device_count,
        "expectedProcessCount": args.expected_processes,
        "expectedLocalDeviceCount": args.expected_local_devices,
        "workerDevices": [
            {
                "id": int(device.id),
                "platform": device.platform,
                "kind": device.device_kind,
            }
            for device in local_devices
        ],
        "globalDevices": device_metadata,
        "pythonVersion": platform.python_version(),
        "jaxVersion": jax.__version__,
        "numpyVersion": np.__version__,
        "orbaxCheckpointVersion": importlib.metadata.version("orbax-checkpoint"),
        "sourceSha256": source_hash,
        "baseLessonSha256": lesson_hash,
        "sourceAndLessonEqualAcrossProcesses": True,
        "sourceIdentityAcrossProcessesSha256": source_identity_hash,
        "fixtureSha256": fixture_hash,
        "localDataSha256": local_data_hash,
        "localDataSha256ByProcess": local_hashes_by_process,
        "ownedRows": [row_start, row_stop],
        "allOwnedRanges": gathered_ranges.tolist(),
        "allRangesAreDisjointAndComplete": True,
        "arrayConstruction": "jax.make_array_from_process_local_data",
        "localRowsPerProcess": local_rows,
        "rowsPerDevice": rows_per_device,
        "dotPrecision": "jax.lax.Precision.HIGHEST",
        "optimizer": {
            "learningRate": 0.04,
            "momentum": 0.8,
            "checkpointStep": CHECKPOINT_STEP,
            "finalStep": FINAL_STEP,
            "checkpointUriScheme": "gs",
            "checkpointIdentitySha256": checkpoint_identity,
            "checkpointUriIncluded": False,
            "storageClient": storage_client,
            "bucketNamespace": "flat",
            "orbaxCheckpointSavedAndWaited": True,
            "abstractShardedRestoreSucceeded": True,
            "restoredStateMatchesSavedStep": True,
            "finalWeightsMatchIndependentNumpy": True,
            "finalMomentumMatchesIndependentNumpy": True,
            "referenceLosses": reference_losses,
        },
        "rendezvous": {
            "jaxDistributedInitializeCompleted": True,
            "initializationTimeoutSeconds": INITIALIZATION_TIMEOUT_SECONDS,
            "heartbeatTimeoutSeconds": HEARTBEAT_TIMEOUT_SECONDS,
            "barriers": [
                "source-equality",
                "process-local-data-ready",
                "before-save",
                "after-save",
                "before-restore",
                "after-restore",
                "complete",
            ],
        },
        "limits": f"One {process_count}-process sixteen-device v5e execution and a tiny synthetic linear workload; no production-quality, model-scale, or generalized performance claim.",
    }
    (worker_output / "execution-multihost.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "processIndex": process_index,
                "receipt": str(worker_output / "execution-multihost.json"),
                "executionMode": receipt["executionMode"],
                "checkpointIdentitySha256": checkpoint_identity,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main():
    args = arguments()
    require_tpu_environment()
    # This test uses a standard flat-namespace bucket and object-only IAM.
    # Select the documented standard client before Orbax imports GCSFS;
    # hierarchical/zonal bucket-layout discovery is outside this test's scope.
    os.environ["GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT"] = "false"
    import jax

    # TPU cluster detection is supported when initialize() arguments are omitted.
    # Keep rendezvous and liveness failures bounded for this qualification run.
    jax.distributed.initialize(
        initialization_timeout=INITIALIZATION_TIMEOUT_SECONDS,
        heartbeat_timeout_seconds=HEARTBEAT_TIMEOUT_SECONDS,
        shutdown_timeout_seconds=SHUTDOWN_TIMEOUT_SECONDS,
    )
    run(args, jax)


if __name__ == "__main__":
    main()
