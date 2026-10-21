"""Benchmark asynchronous work correctly · learner-only exercises (3/3 contracts ready)"""

# Exercise: Repeat the benchmark with \(96\) rows while preserving the feature and output dimensions. Keep a separate report rather than overwriting the original workload label.

# Inputs:
# - A JAX function, fixed workload, warmup count, and repeat count.

# Outputs:
# - Raw synchronized timings with compilation separated from warmed execution.

# Invariants to check:
# - Every sample waits for completed output; workload shape/dtype and timing boundary are recorded.

# Hints:
# - Call block_until_ready inside each measured interval.
# - Use an independent NumPy result for the changed equivalent workload.

import jax.numpy as jnp
import time
import statistics

def synchronized_benchmark(function, arguments, warmup, repeats):
    """Raw synchronized timings with compilation separated from warmed execution."""
    # TODO: Warm the selected JAX function and synchronize every timed result while retaining raw samples.
    raise NotImplementedError




# Benchmark a different equivalent workload (Transfer):
# Replace tanh with a row-wise sum of squared predictions, returning one scalar per row. Derive the NumPy reference, warm this new function, collect five synchronized samples and record the changed output shape.

# Inputs:
# - Replace tanh with a row-wise sum of squared predictions, returning one scalar per row. Derive the NumPy reference, warm this new function, collect five synchronized samples and record the changed output shape.

# Outputs:
# - Per-row scalar outputs, NumPy parity, and five completed timing samples.

# Invariants to check:
# - The benchmark synchronizes each output and records the changed output shape.

# Hints:
# - Square before reducing the output-feature axis; the result has shape \((64,)\).

import jax.numpy as jnp
import numpy as np
import time
import statistics

def practice_benchmark_a_different_equivalent_workload(features, weights):
    """Implement the benchmark a different equivalent workload check."""
    # TODO: Implement row-wise squared-sum predictions, derive a NumPy reference, warm the changed function, and collect five synchronized samples.
    raise NotImplementedError




# Repair a timer that includes setup but omits completion (Diagnosis):
# Inspect the broken timer below conceptually: start; create inputs; call `jit(predict)(inputs)`; stop. Write a repaired runner and show its explicit boundary. Explain which stages the broken interval mixes and misses.

# Inputs:
# - Inspect the broken timer below conceptually: start; create inputs; call `jit(predict)(inputs)`; stop. Write a repaired runner and show its explicit boundary. Explain which stages the broken interval mixes and misses.

# Outputs:
# - A runner with separately stated setup and completed-work timing boundaries.

# Invariants to check:
# - The measured interval contains the declared workload and waits for its result.

# Hints:
# - Prepare and wait for inputs, reuse one compiled callable, warm it, then synchronize the returned output inside every measured interval.

import jax.numpy as jnp
import time

def practice_repair_a_timer_that_includes_setup_but_omits_completion(function, arguments):
    """Implement the repair a timer that includes setup but omits completion check."""
    # TODO: Move input creation outside the timed region and block on the output before stopping the clock.
    raise NotImplementedError

