"""Build a tiny multilayer perceptron: worked experiments and reference solutions. CPU checks."""

# 1. Define the fixture and parameter tree
import jax
import jax.numpy as jnp
import numpy as np
import optax
x = jnp.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
y = jnp.array([0.,1.,1.,0.])
k1, k2 = jax.random.split(jax.random.key(0))
params = {'w1':jax.random.normal(k1,(2,8))*.4, 'b1':jnp.zeros(8),
          'w2':jax.random.normal(k2,(8,1))*.4, 'b2':jnp.zeros(1)}
assert sum(a.size for a in jax.tree.leaves(params)) == 33

# 2. Write prediction and verify the loss
def logits(p, batch):
    return (jnp.tanh(batch@p['w1']+p['b1'])@p['w2']+p['b2']).squeeze(-1)
def loss(p, batch, labels):
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits(p,batch),labels))
scores = np.asarray(logits(params,x))
expected = np.mean(np.logaddexp(0.,scores)-np.asarray(y)*scores)
np.testing.assert_allclose(loss(params,x,y), expected, rtol=1e-6)
zeros = jax.tree.map(jnp.zeros_like, params)
np.testing.assert_allclose(loss(zeros,x,y),np.log(2.),rtol=1e-6)
assert logits(params,jnp.ones((1,2))).shape == (1,)

# 3. Train and evaluate untouched points
tx = optax.adam(.03)
state = tx.init(params)
@jax.jit
def step(p,s):
    value,grads = jax.value_and_grad(loss)(p,x,y)
    updates,s = tx.update(grads,s,p)
    return optax.apply_updates(p,updates),s,value
initial = float(loss(params,x,y))
for _ in range(200): params,state,_ = step(params,state)
rng = np.random.default_rng(12)
held_x = np.repeat(np.asarray(x),8,axis=0)+rng.normal(0,.12,(32,2))
held_y = (held_x[:,0]*held_x[:,1]<0).astype(np.float32)
held_scores = np.asarray(logits(params,jnp.array(held_x)))
accuracy = np.mean((held_scores>0)==held_y)
final = float(loss(params,x,y))
assert final < .03 and accuracy >= .95
print('Training loss:', initial, '->', final, 'held-out accuracy:', accuracy)

import jax
import jax.numpy as jnp
import numpy as np
import optax
x = jnp.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
y = jnp.array([0.,1.,1.,0.])
k1, k2 = jax.random.split(jax.random.key(0))
params = {'w1':jax.random.normal(k1,(2,8))*.4, 'b1':jnp.zeros(8),
          'w2':jax.random.normal(k2,(8,1))*.4, 'b2':jnp.zeros(1)}
assert sum(a.size for a in jax.tree.leaves(params)) == 33

def logits(p, batch):
    return (jnp.tanh(batch@p['w1']+p['b1'])@p['w2']+p['b2']).squeeze(-1)
def loss(p, batch, labels):
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits(p,batch),labels))
scores = np.asarray(logits(params,x))
expected = np.mean(np.logaddexp(0.,scores)-np.asarray(y)*scores)
np.testing.assert_allclose(loss(params,x,y), expected, rtol=1e-6)
zeros = jax.tree.map(jnp.zeros_like, params)
np.testing.assert_allclose(loss(zeros,x,y),np.log(2.),rtol=1e-6)
assert logits(params,jnp.ones((1,2))).shape == (1,)

tx = optax.adam(.03)
state = tx.init(params)
@jax.jit
def step(p,s):
    value,grads = jax.value_and_grad(loss)(p,x,y)
    updates,s = tx.update(grads,s,p)
    return optax.apply_updates(p,updates),s,value
initial = float(loss(params,x,y))
for _ in range(200): params,state,_ = step(params,state)
rng = np.random.default_rng(12)
held_x = np.repeat(np.asarray(x),8,axis=0)+rng.normal(0,.12,(32,2))
held_y = (held_x[:,0]*held_x[:,1]<0).astype(np.float32)
held_scores = np.asarray(logits(params,jnp.array(held_x)))
accuracy = np.mean((held_scores>0)==held_y)
final = float(loss(params,x,y))
assert final < .03 and accuracy >= .95
print('Training loss:', initial, '->', final, 'held-out accuracy:', accuracy)

# Figure data experiment
axis = jnp.linspace(-1.6, 1.6, 61)
gx, gy = jnp.meshgrid(axis, axis)
grid = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)
prob = jax.nn.sigmoid(logits(params, grid)).reshape(gx.shape)
visual_data = {'kind': 'field', 'values': prob.tolist(), 'extent': [-1.6, 1.6, -1.6, 1.6], 'xlabel': 'feature 0', 'ylabel': 'feature 1', 'unit': 'P(label 1)', 'points': x.tolist(), 'labels': y.tolist()}

# Experiment: Duplication preserves a mean objective
repeated_x=jnp.repeat(x,3,axis=0);repeated_y=jnp.repeat(y,3)
np.testing.assert_allclose(loss(params,repeated_x,repeated_y),loss(params,x,y),rtol=1e-5)
original_g=jax.grad(loss)(params,x,y);repeated_g=jax.grad(loss)(params,repeated_x,repeated_y)
for a,b in zip(jax.tree.leaves(original_g),jax.tree.leaves(repeated_g)):
    np.testing.assert_allclose(a,b,rtol=1e-4,atol=1e-7)

# Experiment: An affine stack stays affine
def affine_stack(batch):return (batch@params['w1']+params['b1'])@params['w2']+params['b2']
midpoint=jnp.array([[.2,-.3]]);delta=jnp.array([[.4,.1]])
np.testing.assert_allclose(affine_stack(midpoint),.5*(affine_stack(midpoint+delta)+affine_stack(midpoint-delta)),atol=1e-6)

# Reference solution. Try the exercise before reading this.
complemented=dict(params,w2=-params['w2'],b2=-params['b2'])
changed_x=jnp.array([[-.8,.7],[.7,.9],[-1.1,-.8]])
np.testing.assert_allclose(logits(complemented,changed_x),-logits(params,changed_x),atol=1e-6)
expected_labels=(np.asarray(changed_x)[:,0]*np.asarray(changed_x)[:,1]>0)
assert np.array_equal(np.asarray(logits(complemented,changed_x)>0),expected_labels)

# Reference practice: Reorder hidden units without changing predictions
order=jnp.array([7,0,6,1,5,2,4,3])
probe=jnp.array([[.2,-.9],[-.4,.6],[.8,.1]])
permuted=dict(params,w1=params['w1'][:,order],b1=params['b1'][order],w2=params['w2'][order,:])
np.testing.assert_allclose(logits(permuted,probe),logits(params,probe),rtol=1e-5,atol=1e-5)
broken=dict(params,w1=params['w1'][:,order],b1=params['b1'][order])
assert not np.allclose(logits(broken,probe),logits(params,probe),atol=1e-5)
print('Consistent hidden permutation preserves logits; one-sided permutation changes them.')

# Reference practice: Diagnose accidental pairwise broadcasting
bad=optax.sigmoid_binary_cross_entropy(logits(params,x),y[:,None])
assert bad.shape==(4,4)
good=optax.sigmoid_binary_cross_entropy(logits(params,x),y)
assert good.shape==(4,)
np.testing.assert_allclose(jnp.mean(good),loss(params,x,y),rtol=1e-6)
print("PASS: networks-01")
