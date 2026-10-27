"""Capture a small, numeric-only CPU timeline excerpt for performance-03."""

from __future__ import annotations

import gzip
import hashlib
import json
import platform
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "retained_cpu_timeline_p2_81.json"


def complete(events, name):
    return sorted(
        (event for event in events if event.get("ph") == "X" and event.get("name") == name),
        key=lambda event: event["ts"],
    )


def main():
    if jax.default_backend() != "cpu":
        raise RuntimeError(f"This retained excerpt is CPU-only; got {jax.default_backend()!r}")

    # Tiny fixed arrays keep CPU work bounded; warm the exact specialization first.
    host = np.arange(128, dtype=np.float32).reshape(16, 8) / 128
    weights = np.arange(64, dtype=np.float32).reshape(8, 8) / 64
    x, w = jnp.asarray(host), jnp.asarray(weights)
    compiled = jax.jit(lambda a, b: jnp.tanh(a @ b))
    expected = np.tanh(host @ weights)
    np.testing.assert_allclose(compiled(x, w).block_until_ready(), expected, rtol=2e-5, atol=2e-5)

    with tempfile.TemporaryDirectory(prefix="p2-81-xplane-") as tmp:
        trace_root = Path(tmp) / "trace"
        with jax.profiler.trace(trace_root, create_perfetto_trace=True):
            with jax.profiler.StepTraceAnnotation("learner_step", step_num=0):
                with jax.profiler.TraceAnnotation("input_wait"):
                    time.sleep(0.002)
                with jax.profiler.TraceAnnotation("model_and_wait"):
                    result = compiled(x, w)
                    result.block_until_ready()
            with jax.profiler.StepTraceAnnotation("learner_step", step_num=1):
                with jax.profiler.TraceAnnotation("input_wait"):
                    time.sleep(0.0005)
                with jax.profiler.TraceAnnotation("model_and_wait"):
                    result = compiled(x, w)
                    result.block_until_ready()
        np.testing.assert_allclose(result, expected, rtol=2e-5, atol=2e-5)
        traces = list(trace_root.rglob("perfetto_trace.json.gz"))
        if not traces:
            raise RuntimeError("JAX did not write a Perfetto trace")
        trace_path = max(traces, key=lambda path: path.stat().st_mtime_ns)
        raw_events = json.loads(gzip.decompress(trace_path.read_bytes()))["traceEvents"]

    named = {name: complete(raw_events, name) for name in ("learner_step", "input_wait", "model_and_wait")}
    if any(len(named[name]) != 2 for name in named):
        raise RuntimeError(f"Expected two completed events of each name, found { {k: len(v) for k, v in named.items()} }")
    rows = []
    for index in range(2):
        outer, wait, model = (named[name][index] for name in ("learner_step", "input_wait", "model_and_wait"))
        assert outer["ts"] <= wait["ts"] <= wait["ts"] + wait["dur"] <= model["ts"] + 1
        assert model["ts"] + model["dur"] <= outer["ts"] + outer["dur"] + 1
        rows.append((outer, wait, model))
    # Keep compact numeric evidence only. Perfetto timestamps/durations use microseconds.
    events = []
    for index, row in enumerate(rows):
        condition = "2 ms injected wait" if index == 0 else "0.5 ms injected wait"
        for event in row:
            events.append({"condition": condition, "name": event["name"], "start_us": event["ts"],
                           "duration_us": event["dur"], "end_us": event["ts"] + event["dur"],
                           "pid": event.get("pid"), "tid": event.get("tid")})
    events.sort(key=lambda event: (event["start_us"], -event["duration_us"]))
    outer, wait, model = rows[0]
    parent_duration = outer["dur"]
    child_union = wait["dur"] + model["dur"]
    changed_wait_us = rows[1][1]["dur"]
    predicted_wait_drop_us = wait["dur"] - changed_wait_us
    payload = {
        "schemaVersion": 1,
        "capture": "retained numeric excerpt from an executed JAX Perfetto CPU trace",
        "source": {
            "script": "code/capture_retained_cpu_timeline_p2_81.py",
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "device": str(jax.devices()[0]),
            "workload": "jit(tanh(a @ b)); a=(16,8), b=(8,8), float32; finite NumPy check passed",
        },
        "units": "Perfetto complete-event timestamps and durations in microseconds",
        "warmup": "exact shape/dtype specialization compiled and synchronized before trace",
            "profiled": "two steps in one trace: 2 ms injected host sleep, then 0.5 ms injected host sleep; both followed by the same synchronized model call",
            "unprofiled_comparison": "prediction checked against the second profiled step in this trace; no independent end-to-end timing is claimed",
        "events": events,
        "input_wait_change": {
            "predicted": "reducing requested sleep from 2 ms to 0.5 ms should reduce the input_wait host span by about 1.5 ms, while model_and_wait should stay in the same rough range",
            "measured_wait_drop_us": predicted_wait_drop_us,
            "measured_model_before_us": model["dur"],
            "measured_model_after_us": rows[1][2]["dur"],
        },
        "accounting": {
            "outer_duration_us": parent_duration,
            "children_union_us": child_union,
            "outer_remainder_us": parent_duration - child_union,
            "naive_parent_plus_children_us": parent_duration + child_union,
            "note": "Children are ordered and disjoint in this step; the parent already contains them.",
        },
        "limits": [
            "Trace contains CPU host annotations and runtime events; these spans do not isolate device kernel time.",
            "Injected sleep is a controlled delay, not a production input pipeline measurement.",
            "Profiler instrumentation can perturb short intervals; use a separate synchronized benchmark for performance claims.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUT)
    print(json.dumps({"backend": payload["environment"]["backend"], "events": events, "input_wait_change": payload["input_wait_change"], "accounting": payload["accounting"]}, indent=2))


if __name__ == "__main__":
    main()
