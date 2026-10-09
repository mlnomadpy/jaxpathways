"""Implement and defend a bounded derivative/compiler contract."""
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np


def analyze(x,direction,weighting):
    """For f(a,b)=(ab, sin(a)+b*b), return value, jvp, vjp, hvp.

    Return a dictionary with those keys. hvp belongs to 0.5*sum(f(x)**2).
    Reject any point/direction/weighting not shaped (2,).
    """
    # Key APIs to use: `contract`, `shape`, `mapping`, `jnp.array`, `jnp.sin`
    # Step 1: Guard input contract (`x.shape != (2,) or direction.shape != (2,) or weighting.shape != (2,)`) and fail fast if violated.
    # Step 2: Function `mapping(z)` implementing this stage's computation:
    # Step 3: Compute forward-mode Jacobian-vector product (`(value, jvp)`).
    # Step 4: Evaluate primal output and reverse-mode pullback function (`vjp`).
    # Step 5: Evaluate `objective` from the current inputs and state.
    # Step 6: Differentiate the objective to obtain gradients `hvp`.
    raise NotImplementedError('Stage 1: forward, reverse and curvature products')


@jax.custom_jvp
def stable_softplus(x):
    """Stable scalar softplus, with an exact registered JVP and higher derivatives."""
    # Key APIs to use: `jnp.logaddexp`
    # Step 1: Return `jnp.logaddexp(0.0, x)` to the caller.
    raise NotImplementedError('Stage 2: stable primal plus exact custom JVP')


@jax.custom_vjp
def stable_logsumexp(x):
    """Stable vector logsumexp, with exact registered forward/backward rules."""
    # Key APIs to use: `special.logsumexp`
    # Step 1: Return `jax.scipy.special.logsumexp(x)` to the caller.
    raise NotImplementedError('Stage 2: stable primal plus exact custom VJP')


def tiny_jvp(closed,primals,tangents):
    """Return flat tuples (primal_outputs, tangent_outputs) by interpreting a jaxpr.

    Support add, mul, neg, sin and reduce_sum using explicit tangent rules;
    initialize captured/literal constants with zero tangents. Reject effects,
    unsupported/multi-result primitives, nonfloating inputs, explicit reduction
    placement and input/tangent shape or dtype mismatches. Do not delegate the
    derivative calculation to jax.jvp, grad, jacfwd or jacrev.
    """
    # Key APIs to use: `contract`, `NotImplementedError`, `put`, `zip`, `jnp.zeros_like`
    # Step 1: Evaluate `program` from the current inputs and state.
    # Step 2: Guard input contract (`program.effects`) and fail fast if violated.
    # Step 3: Guard input contract (`len(primals) != len(program.invars) or len(tangents) != len(primals)`) and fail fast if violated.
    # Step 4: Evaluate `values` from the current inputs and state.
    # Step 5: Evaluate `directions` from the current inputs and state.
    # Step 6: Function `put(var, value, tangent)` implementing this stage's computation:
    raise NotImplementedError('Stage 3: actual primitive-by-primitive transformation')
