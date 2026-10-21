"""Branches with lax.cond · learner-only exercises (3/3 contracts ready)"""

# Exercise: Write a compiled function that applies \(2x\) when \(x\) is positive and \(x-2\) otherwise. Check positive, negative, and zero inputs.

# Inputs:
# - A scalar or array of values evaluated as scalar calls.

# Outputs:
# - A JIT-compatible piecewise function: twice x when positive, otherwise x minus two.

# Invariants to check:
# - Both lax.cond branches return identical shape and dtype.
# - The zero boundary follows the non-positive branch.

# Hints:
# - Use x > 0 as the predicate.
# - Pass x as the operand and use one scalar result from each branch.

import jax
import jax.numpy as jnp

@jax.jit
def piecewise_rule(x: jax.Array) -> jax.Array:
    """Return 2*x for positive x and x-2 otherwise."""
    # TODO: express the runtime branch with jax.lax.cond.
    raise NotImplementedError




# Match a changed piecewise rule (Practice):
# Implement \(x^2\) for \(x>1\) and \(2x\) otherwise. Derive outputs at \(-1\),\(1\),\(2\) and derivatives away from \(1\).

# Inputs:
# - A scalar input v.

# Outputs:
# - The value of v squared for v>1, and 2v otherwise.

# Invariants to check:
# - At v=1 the false branch is chosen.
# - Both branches return a scalar of compatible dtype.

# Hints:
# - Set the predicate to v > 1.
# - Check derivatives only away from the branch boundary.

import jax
import jax.numpy as jnp

def changed_rule(v: jax.Array) -> jax.Array:
    """Implement the two-branch scalar rule."""
    # TODO: use lax.cond with the exact boundary specified in the task.
    raise NotImplementedError




# Repair incompatible branch shapes (Challenge):
# Return a scalar in one branch and a length-two vector in the other, reproduce the error, then define a consistent two-value result.

# Inputs:
# - A scalar input v used by two branch functions.

# Outputs:
# - A pair of scalar values with the same shape and dtype regardless of the predicate.

# Invariants to check:
# - lax.cond requires compatible results from both branches.
# - The pair gives the same semantic meaning on both paths.

# Hints:
# - Choose what the two returned components represent first.
# - Construct a length-two array in each branch.

import jax
import jax.numpy as jnp

def consistent_pair(v: jax.Array) -> jax.Array:
    """Return a consistent length-two result from both runtime branches."""
    # TODO: make both cond branches return matching shape and dtype.
    raise NotImplementedError

