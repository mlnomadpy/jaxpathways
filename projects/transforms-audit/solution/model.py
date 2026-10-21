"""Compare scalar evaluation, vmap, derivatives, and JIT shape tracing."""

import jax
import jax.numpy as jnp


def transform_report(xs=None):
    f = lambda z: jnp.sin(z) + z * z
    xs = (
        jnp.array([-1.2, -0.1, 0.7, 1.4], dtype=jnp.float32)
        if xs is None
        else jnp.asarray(xs, dtype=jnp.float32)
    )
    vector = jax.vmap(f)
    traces = {"count": 0}

    def counted(x):
        traces["count"] += 1
        return vector(x)

    compiled = jax.jit(counted)
    mapped = vector(xs)
    loop = jnp.stack([f(z) for z in xs])
    grad = jax.vmap(jax.grad(f))(xs)
    result = compiled(xs)
    result.block_until_ready()
    extended = compiled(jnp.concatenate([xs, jnp.array([2.0], dtype=xs.dtype)]))
    extended.block_until_ready()
    return {
        "mapped": mapped,
        "loop": loop,
        "gradient": grad,
        "compiled": result,
        "extended": extended,
        "trace_count": traces["count"],
        "jaxpr": str(jax.make_jaxpr(vector)(xs)),
    }
