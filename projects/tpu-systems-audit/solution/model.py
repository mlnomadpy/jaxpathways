"""Reference implementation: Account for parameter-state bytes and derive scaling efficiency from supplied measurements."""

import jax.numpy as jnp


def systems_report(
    parameter_count,
    dtype_bytes,
    master_bytes,
    moment_count,
    measured_examples_per_second,
    worker_count,
    single_worker_examples_per_second,
):
    vals = (
        parameter_count,
        dtype_bytes,
        master_bytes,
        moment_count,
        measured_examples_per_second,
        worker_count,
        single_worker_examples_per_second,
    )
    if any(v <= 0 for v in vals):
        raise ValueError("all sizes and rates must be positive")
    weight = parameter_count * dtype_bytes
    masters = parameter_count * master_bytes
    moments = parameter_count * 4 * moment_count
    total = weight + masters + moments
    ideal = single_worker_examples_per_second * worker_count
    return {
        "weight_bytes": weight,
        "master_bytes": masters,
        "moment_bytes": moments,
        "total_bytes": total,
        "ideal_rate": ideal,
        "measured_rate": measured_examples_per_second,
        "efficiency": measured_examples_per_second / ideal,
    }
