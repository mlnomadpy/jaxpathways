"""Tracing, static arguments, and recompilation · learner-only exercises (3/3 contracts ready)"""

# Exercise: Foundation · Extend reduce_values with a static max mode while preserving sum and mean. Reject unknown modes. Verify all three modes on \([1, 2, 3]\) and on \([-4, -1, -2]\).

# Inputs:
# - A reducer supporting sum, mean, and max.
# - Two arrays containing positive and negative values.

# Outputs:
# - Jitted results for all supported modes and an error for an unknown mode.

# Invariants to check:
# - The mode is static configuration.
# - Unknown modes are rejected.
# - Both signs of input are checked.

# Hints:
# - Keep array values dynamic and specialize the mode.
# - Reuse the function for changed array values.

import jax
import jax.numpy as jnp

def reduce_values(values, mode):
    # TODO: implement the declared reductions and reject unknown modes
    raise NotImplementedError

def verify_reductions():
    # TODO: compile with mode static and check both input sets
    raise NotImplementedError




# Keep runtime values dynamic (Practice):
# Implement an elementwise absolute value with a numerical selection, jit it, and test mixed-sign vectors. Explain why making the whole vector static is inappropriate.

# Inputs:
# - Mixed-sign numeric vectors.
# - An elementwise absolute-value function.

# Outputs:
# - Jitted results for each vector.

# Invariants to check:
# - Numerical values remain runtime data.
# - Output shape matches its input.

# Hints:
# - Use elementwise numerical selection.
# - Do not make the entire vector static.

import jax
import jax.numpy as jnp

def elementwise_absolute(values):
    # TODO: select the nonnegative magnitude elementwise
    raise NotImplementedError

def verify_absolute(compiled_absolute, vectors):
    # TODO: test the compiled function on each mixed-sign input
    raise NotImplementedError




# Audit a shape-dependent specialization (Challenge):
# Run a jitted sum-of-squares function for lengths \(3\) and \(5\), each with two different value sets. Compare to a loop calculation. Explain which changes can affect specialization and which are ordinary runtime data.

# Inputs:
# - A jitted sum-of-squares function.
# - Inputs of lengths three and five with two value sets each.

# Outputs:
# - Compiled results and an independent Python-loop reference.
# - A note distinguishing shape from data-value changes.

# Invariants to check:
# - Each element contributes once.
# - Separate changed shape from changed values.
# - Do not use elapsed time alone to claim a trace count.

# Hints:
# - Inspect shape metadata directly.
# - Build the reference with a plain Python loop.

import jax
import jax.numpy as jnp

def sum_squares(values):
    # TODO: define the numerical reduction
    raise NotImplementedError

def audit_shapes(compiled, value_sets):
    # TODO: compare compiled results with an independent loop reference
    raise NotImplementedError

