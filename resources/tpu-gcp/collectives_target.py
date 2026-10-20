"""Qualify a bounded four-device single-host TPU collective workload.

This target is intentionally separate from the CPU-configured lesson source.
It refuses to import JAX unless ``JAX_PLATFORMS=tpu`` is already selected.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

ROOT = Path(__file__).resolve().parents[2]
LESSON = (
    ROOT / "phases/09-distributed/03-communication-efficient-algorithms/code/main.py"
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if not 3 <= args.repeats <= 50:
        parser.error("--repeats must be between 3 and 50")
    return args


def require_target_environment():
    if os.environ.get("JAX_PLATFORMS") != "tpu":
        raise RuntimeError(
            "Set JAX_PLATFORMS=tpu before running; this target never selects or falls back to CPU"
        )

    import jax

    if jax.default_backend() != "tpu":
        raise RuntimeError("the selected JAX backend is not TPU")
    if jax.process_count() != 1:
        raise RuntimeError(
            "this target qualifies one host only; process_count must equal 1"
        )
    if jax.local_device_count() != 4 or jax.device_count() != 4:
        raise RuntimeError(
            "this target requires exactly four local and global TPU devices; "
            f"observed local={jax.local_device_count()} global={jax.device_count()}"
        )
    devices = jax.devices()
    if any(device.platform != "tpu" for device in devices):
        raise RuntimeError("all four devices must report the TPU platform")
    return jax


def run(args, jax):
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    devices = jax.devices()
    mesh = Mesh(np.asarray(devices), ("data",))
    row_sharding = NamedSharding(mesh, P("data", None))
    label_sharding = NamedSharding(mesh, P("data"))
    replicated_sharding = NamedSharding(mesh, P())

    # Match the lesson's deterministic 16-row, 8-feature fixture and objective.
    x_host = np.arange(16 * 8, dtype=np.float32).reshape(16, 8) / 128 - 0.5
    y_host = x_host @ np.linspace(-0.4, 0.3, 8, dtype=np.float32)
    weights_host = np.linspace(0.1, -0.2, 8, dtype=np.float32)
    x = jax.device_put(x_host, row_sharding)
    y = jax.device_put(y_host, label_sharding)
    weights = jax.device_put(weights_host, replicated_sharding)

    def local_contribution(a, b, current_weights):
        # Isolate communication from the TPU's lower-precision default dot policy.
        residual = (
            jnp.matmul(a, current_weights, precision=jax.lax.Precision.HIGHEST) - b
        )
        return 2 * jnp.matmul(a.T, residual, precision=jax.lax.Precision.HIGHEST) / 16

    def reduce_all(a, b, current_weights):
        return jax.lax.psum(local_contribution(a, b, current_weights), "data")

    def reduce_scatter(a, b, current_weights):
        return jax.lax.psum_scatter(
            local_contribution(a, b, current_weights), "data", tiled=True
        )

    all_gradient = jax.jit(
        jax.shard_map(
            reduce_all,
            mesh=mesh,
            in_specs=(P("data", None), P("data"), P()),
            out_specs=P(),
        )
    )
    sharded_gradient = jax.jit(
        jax.shard_map(
            reduce_scatter,
            mesh=mesh,
            in_specs=(P("data", None), P("data"), P()),
            out_specs=P("data"),
        )
    )
    gather = jax.jit(
        jax.shard_map(
            lambda gradient: jax.lax.all_gather(gradient, "data", tiled=True),
            mesh=mesh,
            in_specs=P("data"),
            out_specs=P(),
            check_vma=False,
        )
    )

    # Confirm the arrays were actually partitioned over the selected mesh.
    if x.sharding != row_sharding or y.sharding != label_sharding:
        raise RuntimeError(
            "input arrays do not have the expected row-sharded placement"
        )
    if weights.sharding != replicated_sharding:
        raise RuntimeError("weights are not replicated over the four-device mesh")
    if [shard.data.shape for shard in x.addressable_shards] != [(4, 8)] * 4:
        raise RuntimeError("row-sharded input does not contain four 4x8 local shards")
    if [shard.data.shape for shard in y.addressable_shards] != [(4,)] * 4:
        raise RuntimeError("labels do not contain four length-four local shards")

    expected = 2 * x_host.T @ (x_host @ weights_host - y_host) / 16
    actual_all = all_gradient(x, y, weights).block_until_ready()
    actual_sharded = sharded_gradient(x, y, weights).block_until_ready()
    actual_gathered = gather(actual_sharded).block_until_ready()
    np.testing.assert_allclose(np.asarray(actual_all), expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(
        np.asarray(actual_gathered), expected, atol=2e-6, rtol=2e-6
    )
    if [shard.data.shape for shard in actual_all.addressable_shards] != [(8,)] * 4:
        raise RuntimeError("all-reduce result is not replicated on all four devices")
    if [shard.data.shape for shard in actual_sharded.addressable_shards] != [(2,)] * 4:
        raise RuntimeError(
            "reduce-scatter result does not contain four length-two shards"
        )

    lowered = {
        "psum": str(all_gradient.lower(x, y, weights).compiler_ir(dialect="stablehlo")),
        "psum_scatter": str(
            sharded_gradient.lower(x, y, weights).compiler_ir(dialect="stablehlo")
        ),
        "all_gather": str(
            gather.lower(actual_sharded).compiler_ir(dialect="stablehlo")
        ),
    }
    expected_ops = {
        "psum": "all_reduce",
        "psum_scatter": "reduce_scatter",
        "all_gather": "all_gather",
    }
    for name, operation in expected_ops.items():
        if operation not in lowered[name]:
            raise RuntimeError(f"lowered {name} program lacks StableHLO {operation}")

    compiled = {
        "psum": all_gradient.lower(x, y, weights).compile(),
        "psum_scatter": sharded_gradient.lower(x, y, weights).compile(),
        "all_gather": gather.lower(actual_sharded).compile(),
    }
    inputs = {
        "psum": (x, y, weights),
        "psum_scatter": (x, y, weights),
        "all_gather": (actual_sharded,),
    }
    timing = {}
    for name, executable in compiled.items():
        executable(*inputs[name]).block_until_ready()
        samples_ms = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            executable(*inputs[name]).block_until_ready()
            samples_ms.append((time.perf_counter() - started) * 1000)
        timing[name] = {
            "samples_ms": samples_ms,
            "median_ms": statistics.median(samples_ms),
        }

    if args.trace:
        with jax.profiler.trace(str(output / "trace"), create_perfetto_trace=True):
            for index, (name, executable) in enumerate(compiled.items()):
                with jax.profiler.StepTraceAnnotation("collective", step_num=index):
                    with jax.profiler.TraceAnnotation(name):
                        executable(*inputs[name]).block_until_ready()

    lesson_bytes = LESSON.read_bytes()
    source_bytes = Path(__file__).read_bytes()
    for name, text in lowered.items():
        (output / f"{name}.stablehlo.txt").write_text(text)

    receipt = {
        "schemaVersion": 1,
        "executedAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "executionMode": "real-tpu-single-host-four-device",
        "backend": jax.default_backend(),
        "processCount": jax.process_count(),
        "globalDeviceCount": jax.device_count(),
        "localDeviceCount": jax.local_device_count(),
        "devices": [
            {"platform": d.platform, "kind": d.device_kind, "id": d.id} for d in devices
        ],
        "pythonVersion": platform.python_version(),
        "jaxVersion": jax.__version__,
        "numpyVersion": np.__version__,
        "sourceSha256": hashlib.sha256(source_bytes).hexdigest(),
        "baseLesson": str(LESSON.relative_to(ROOT)),
        "baseLessonSha256": hashlib.sha256(lesson_bytes).hexdigest(),
        "fixtureSha256": hashlib.sha256(
            x_host.tobytes() + y_host.tobytes() + weights_host.tobytes()
        ).hexdigest(),
        "collectives": {
            name: {
                "stablehloOperation": expected_ops[name],
                "stablehloFile": f"{name}.stablehlo.txt",
            }
            for name in expected_ops
        },
        "placement": {
            "meshAxis": "data",
            "meshShape": [len(devices)],
            "inputRowsPerDevice": 4,
            "allReduceResult": "replicated length-8 vector",
            "reduceScatterResult": "length-2 vector per device",
            "allGatherResult": "replicated length-8 vector",
        },
        "numericalChecks": {
            "dotPrecision": "HIGHEST",
            "independentReference": "NumPy float32 fixture; full-batch gradient formula from distributed-03",
            "absoluteTolerance": 2e-6,
            "relativeTolerance": 2e-6,
            "allReduceMatchesReference": True,
            "reduceScatterThenAllGatherMatchesReference": True,
        },
        "timingScope": "warmed synchronized collective calls; excludes compilation, host-to-device placement, and trace overhead",
        "repeats": args.repeats,
        "timings": timing,
        "trace": "trace/" if args.trace else None,
        "limits": "One four-device host and this deterministic small fixture; no multi-host, scaling, or general performance claim.",
    }
    (output / "execution-collectives.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(receipt, indent=2, allow_nan=False))


def main():
    args = arguments()
    jax = require_target_environment()
    run(args, jax)


if __name__ == "__main__":
    main()
