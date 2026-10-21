"""Measure CPU first-call and steady latency, shape tracing, and retain a JAXPR artifact."""

import json, time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np


def performance_report(output_dir, repeats=5):
    if repeats < 3:
        raise ValueError("use at least three repeated calls")
    traces = {"count": 0}

    def traced(x):
        traces["count"] += 1
        return jnp.tanh(x @ x)

    f = jax.jit(traced)
    x = jnp.eye(16, dtype=jnp.float32)
    t = time.perf_counter()
    first = f(x)
    first.block_until_ready()
    first_seconds = time.perf_counter() - t
    samples = []
    for _ in range(repeats):
        t = time.perf_counter()
        y = f(x)
        y.block_until_ready()
        samples.append(time.perf_counter() - t)
    second = f(jnp.eye(32, dtype=jnp.float32))
    second.block_until_ready()
    shape_trace_count = traces["count"]
    jaxpr = str(jax.make_jaxpr(traced)(x))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "workload.jaxpr.txt").write_text(jaxpr)
    report = {
        "backend": jax.default_backend(),
        "shapes": [[16, 16], [32, 32]],
        "trace_count": shape_trace_count,
        "shape_trace_count": shape_trace_count,
        "first_call_seconds": first_seconds,
        "steady_seconds": samples,
        "output_shape": list(y.shape),
        "output": np.asarray(y).tolist(),
        "changed_shape_output": np.asarray(second).tolist(),
        "memory_measured": False,
        "trace_artifact": "workload.jaxpr.txt",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
