"""LoRA: adapt, save and merge low-rank updates: worked experiments and reference solutions. CPU checks."""



import jax
import jax.numpy as jnp
import numpy as np

def lora_forward(x, base, a, b, alpha=1.):
    rank = a.shape[1]
    return x @ base + (alpha/rank) * (x @ a @ b)

x=jnp.asarray(np.random.default_rng(4).normal(size=(24,6)),jnp.float32)
source_w=jnp.arange(24,dtype=jnp.float32).reshape(6,4)/40
base=jnp.zeros_like(source_w)
base_step=jax.jit(jax.grad(lambda w:jnp.mean((x@w-x@source_w)**2)))
for _ in range(150):base=base-.15*base_step(base)
assert float(jnp.mean((x@base-x@source_w)**2))<1e-4
frozen=np.asarray(base).copy()
u=jnp.array([[.2],[-.3],[.1],[.2],[-.1],[.4]])
v=jnp.array([[.5,-.2,.3,.1]])
target=x@(base+u@v)
a=jax.random.normal(jax.random.key(4),(6,1))*.1;b=jnp.zeros((1,4));params=(a,b)
loss=lambda ab:jnp.mean((lora_forward(x,base,*ab,alpha=1.)-target)**2)
step=jax.jit(jax.value_and_grad(loss));history=[]
initial_grads=step(params)[1]
np.testing.assert_array_equal(initial_grads[0],np.zeros((6,1)))
assert np.linalg.norm(initial_grads[1])>0
for _ in range(250):
    value,g=step(params);history.append(float(value));params=jax.tree.map(lambda a,b:a-.4*b,params,g)
a,b=params
assert history[-1]<history[0]*.02
np.testing.assert_array_equal(base,frozen)
merged=base+a@b
np.testing.assert_allclose(lora_forward(x,base,a,b),x@merged,rtol=1e-5,atol=1e-6)
print('LoRA initial/final:',history[0],history[-1],'; trainable:',a.size+b.size,'base:',base.size)


# Figure data experiment
visual_data={'kind':'line','xlabel':'completed parameter updates before measurement','ylabel':'training objective','series':[{'label':'recorded CPU training loss','x':list(range(len(history))),'y':history}]}
for panel in visual_data.get('panels',[visual_data]):
    panel['x']=panel['series'][0]['x']


# Experiment: Show why two zero factors cannot start learning
zero=(jnp.zeros_like(a),jnp.zeros_like(b))
zero_grad=jax.grad(loss)(zero)
assert all(np.array_equal(np.asarray(g),np.zeros(g.shape)) for g in zero_grad)
print('Both zero factors produce zero first gradients.')

# Reference solution. Try the exercise before reading this.
import tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as folder:
    path=Path(folder)/'adapter.npz'
    np.savez(path,a=np.asarray(a),b=np.asarray(b),alpha=np.array(1.),rank=np.array(1))
    with np.load(path,allow_pickle=False) as z:
        assert int(z['rank'])==z['a'].shape[1]
        np.testing.assert_allclose(lora_forward(x,base,z['a'],z['b'],float(z['alpha'])),x@merged,atol=1e-6,rtol=1e-5)
print('Adapter round trip and merge checked.')

# Reference practice: Check the merge on new inputs
new_x=jnp.asarray(np.random.default_rng(91).normal(size=(7,6)),jnp.float32)
np.testing.assert_allclose(lora_forward(new_x,base,a,b),new_x@merged,atol=1e-6,rtol=1e-5)
assert np.linalg.norm(np.asarray(new_x@(base+2*a@b)-new_x@merged))>1e-3
print('New-input merge passes; accidental double merge changes outputs.')
print("PASS: posttraining-02")
