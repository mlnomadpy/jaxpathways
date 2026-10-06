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
    raise NotImplementedError('Stage 1: forward, reverse and curvature products')


@jax.custom_jvp
def stable_softplus(x):
    """Stable scalar softplus, with an exact registered JVP and higher derivatives."""
    raise NotImplementedError('Stage 2: stable primal plus exact custom JVP')


@jax.custom_vjp
def stable_logsumexp(x):
    """Stable vector logsumexp, with exact registered forward/backward rules."""
    raise NotImplementedError('Stage 2: stable primal plus exact custom VJP')


def tiny_jvp(closed,primals,tangents):
    """Return flat tuples (primal_outputs, tangent_outputs) by interpreting a jaxpr.

    Support add, mul, neg, sin and reduce_sum using explicit tangent rules;
    initialize captured/literal constants with zero tangents. Reject effects,
    unsupported/multi-result primitives, nonfloating inputs, explicit reduction
    placement and input/tangent shape or dtype mismatches. Do not delegate the
    derivative calculation to jax.jvp, grad, jacfwd or jacrev.
    """
    raise NotImplementedError('Stage 3: actual primitive-by-primitive transformation')
