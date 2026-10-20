"""Reference implementation: Compute masked batch moments without letting padding affect the denominator."""
import jax.numpy as jnp

def masked_moments(x, mask):
    x=jnp.asarray(x); mask=jnp.asarray(mask, dtype=bool)
    if x.ndim != 2 or mask.shape != (x.shape[0],): raise ValueError("expected (batch, features) and (batch,) mask")
    count=jnp.sum(mask)
    if int(count) == 0: raise ValueError("mask must select at least one row")
    selected=jnp.where(mask[:,None],x,0)
    mean=jnp.sum(selected,axis=0)/count
    var=jnp.sum(jnp.where(mask[:,None],(x-mean)**2,0),axis=0)/count
    return {"count":int(count),"mean":mean,"variance":var}
