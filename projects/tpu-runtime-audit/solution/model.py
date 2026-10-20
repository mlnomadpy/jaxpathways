"""Reference implementation: Inspect actual CPU device inventory and compare explicit BF16 storage with FP32 accumulation."""
import jax
import jax.numpy as jnp
from jax import lax

def runtime_report():
    devices=jax.devices(); x=jnp.array([[1.25,-2.5],[3.0,0.5]],dtype=jnp.float32); w=jnp.array([[0.25,1.5],[-1.0,2.0]],dtype=jnp.float32)
    wb=w.astype(jnp.bfloat16); out=x@wb.astype(jnp.float32)
    participants=jnp.arange(len(devices),dtype=jnp.float32)+1
    collective=jax.pmap(lambda v:lax.psum(v,"workers"),axis_name="workers")(participants)
    return {"backend":jax.default_backend(),"device_count":len(devices),"devices":[str(d) for d in devices],"stored_dtype":str(wb.dtype),"compute_dtype":str(out.dtype),"output":out,"collective_inputs":participants,"collective_outputs":collective}

def simulated_collective_sum(values, omitted=()):
    """CPU-only arithmetic simulation; it does not create worker processes or qualify TPU collectives."""
    values=list(values); omitted=set(omitted)
    if any(i<0 or i>=len(values) for i in omitted): raise ValueError("participant index out of range")
    return sum(v for i,v in enumerate(values) if i not in omitted)
