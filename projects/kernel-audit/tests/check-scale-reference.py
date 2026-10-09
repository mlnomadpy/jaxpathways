"""Independent public scale-assessment arithmetic on four logical CPU devices.

This verifies partition semantics, not network or accelerator performance.
Run in a fresh process
device configuration precedes backend initialization.
"""
import jax
jax.config.update('jax_platforms','cpu')
jax.config.update('jax_num_cpu_devices',4)
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P

rng=np.random.default_rng(73)
x=rng.normal(size=(8,3)).astype(np.float32)
y=rng.normal(size=8).astype(np.float32)
w=np.array([.2,-.1,.3],dtype=np.float32)
reference=2*x.T@(x@w-y)/len(y)
parts=[(x[:3],y[:3]),(x[3:],y[3:])]
local=[2*a.T@(a@w-b)/len(b) for a,b in parts]
weighted=(3*local[0]+5*local[1])/8
wrong=(local[0]+local[1])/2
np.testing.assert_allclose(weighted,reference,rtol=1e-6,atol=1e-6)
assert np.linalg.norm(wrong-reference)>.01
assert len(jax.devices('cpu'))==4
mesh=Mesh(np.asarray(jax.devices('cpu')),('data',))
x_sharded=jax.device_put(x,NamedSharding(mesh,P('data',None)))
y_sharded=jax.device_put(y,NamedSharding(mesh,P('data')))
w_replicated=jax.device_put(w,NamedSharding(mesh,P()))
def objective(weights,features,labels):return jnp.mean((features@weights-labels)**2)
gradient=jax.jit(jax.grad(objective))(w_replicated,x_sharded,y_sharded)
np.testing.assert_allclose(gradient,reference,rtol=1e-6,atol=1e-6)
np.testing.assert_allclose(w_replicated-.1*gradient,w-.1*reference,rtol=1e-6,atol=1e-6)
print('Independent host/global sharded gradient:',reference)
print('Uneven weighted gradient:',weighted,'wrong unweighted gradient:',wrong)
print('Actual backend/device count:',jax.default_backend(),len(jax.devices()))
print('Conditional twofold-local speedup:',1/(.7+.3/2),'zero-local-cost limit:',1/.7)
print('PASS scale reference: objective, update, uneven aggregation and four logical CPU placements; no accelerator/network performance evidence')
