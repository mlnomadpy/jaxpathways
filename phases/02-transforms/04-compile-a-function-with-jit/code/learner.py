"""Compile a function with jit · learner-only exercises (3/3 contracts ready)"""

# Exercise: Foundation · Compile `value_and_grad(loss)`, check the loss and gradient against both the eager transform and the manual matrix formula, and explain how you would synchronize a benchmark of its two outputs.

# Inputs:
# - The lesson loss, weight, input and target arrays.
# - Eager and compiled value-and-gradient computations.

# Outputs:
# - Compiled and eager outputs checked against independent formulas.
# - A synchronization plan for every timed output.

# Invariants to check:
# - Validate numerical results before interpreting speed.
# - Wait for every result leaf before stopping the timer.
# - Do not infer a universal speedup.

# Hints:
# - Keep shape and dtype fixed for equivalence.
# - Synchronize the full returned tree.

import jax
import jax.numpy as jnp

def compare_compiled_value_gradient(loss, weight, inputs, targets):
    # TODO: compile value_and_grad and compare against independent checks
    raise NotImplementedError




# Check that a new value remains runtime data (Practice):
# Keep shapes and dtypes fixed, change \(w\), and compare the compiled loss against an independently computed residual mean. Explain why a different numerical answer does not itself imply a new compilation.

# Inputs:
# - A compiled loss and two weights with fixed shapes and dtypes.
# - The same lesson inputs and targets.

# Outputs:
# - Two compiled outputs and independent residual-mean references.

# Invariants to check:
# - Changing numerical values alone keeps the array signature fixed.
# - Check each result independently.
# - Timing alone is not evidence of retracing.

# Hints:
# - Build the reference from residuals, not by calling the compiled function.
# - Keep shape and dtype unchanged.

import jax.numpy as jnp

def compare_runtime_weights(compiled_loss, weight_a, weight_b, inputs, targets):
    # TODO: evaluate both weights and create independent references
    raise NotImplementedError




# Build a report whose timing claim can be checked (Challenge):
# Benchmark the compiled value-and-gradient function for \(20\) calls after warmup, waiting on every output leaf. Include backend, input shapes, dtype, and median. State what your measurement excludes.

# Inputs:
# - A compiled value-and-gradient function and one warmup call.
# - Twenty measured calls with backend, shapes, and dtype metadata.

# Outputs:
# - A synchronized median latency and workload report.

# Invariants to check:
# - Synchronize all output leaves before stopping each timer.
# - Exclude warmup from the repeated-call distribution.
# - State what the measurement does not include.

# Hints:
# - Use tree traversal to wait for every leaf.
# - Store metadata beside the latency statistic.

import time
import jax

def benchmark_compiled(compiled_step, weight, inputs, targets):
    # TODO: warm up, synchronize every output, and retain repeated durations
    raise NotImplementedError

