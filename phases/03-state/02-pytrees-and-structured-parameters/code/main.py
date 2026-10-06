"""Pytrees and structured parameters: worked experiments and reference solutions. CPU checks."""

# Prepare the inputs
import jax
import jax.numpy as jnp

# Build the computation
params = {"weight": jnp.array([1., -1.]), "bias": jnp.array(0.)}
x = jnp.array([2., 1.])
def loss(p):
    prediction = jnp.dot(p["weight"], x) + p["bias"]
    return (prediction - 3.) ** 2

# Run and check the result
value, grads = jax.value_and_grad(loss)(params)
updated = jax.tree.map(lambda p, g: p - 0.05 * g, params, grads)
print("Before/after:", float(value), float(loss(updated)))
assert grads.keys() == params.keys()
assert loss(updated) < value

import jax
import jax.numpy as jnp
params = {"weight": jnp.array([1., -1.]), "bias": jnp.array(0.)}
x = jnp.array([2., 1.])
def loss(p):
    prediction = jnp.dot(p["weight"], x) + p["bias"]
    return (prediction - 3.) ** 2
value, grads = jax.value_and_grad(loss)(params)
updated = jax.tree.map(lambda p, g: p - 0.05 * g, params, grads)
print("Before/after:", float(value), float(loss(updated)))
assert grads.keys() == params.keys()
assert loss(updated) < value

# Figure data experiment
visual_data = {'kind': 'bar', 'labels': ['weight[0]', 'weight[1]', 'bias'], 'ylabel': 'parameter value', 'series': [{'label': 'before', 'y': [float(params['weight'][0]), float(params['weight'][1]), float(params['bias'])]}, {'label': 'after', 'y': [float(updated['weight'][0]), float(updated['weight'][1]), float(updated['bias'])]}]}

# Experiment: Verify every gradient leaf
assert jnp.allclose(grads["weight"],jnp.array([-8.,-4.]))
assert jnp.allclose(grads["bias"],-4.)
assert jnp.allclose(updated["weight"],jnp.array([1.4,-0.8]))
assert jnp.allclose(updated["bias"],0.2)
assert jnp.allclose(loss(updated),0.64,atol=1e-6)

# Experiment: Flatten and reconstruct without losing identity
leaves,definition=jax.tree.flatten(params)
rebuilt=jax.tree.unflatten(definition,leaves)
assert jax.tree.structure(rebuilt)==jax.tree.structure(params)
assert jnp.array_equal(rebuilt["weight"],params["weight"])
assert jnp.array_equal(rebuilt["bias"],params["bias"])

# Reference solution. Try the exercise before reading this.
norm_squared = sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads))
manual = jnp.sum(grads["weight"] ** 2) + grads["bias"] ** 2
assert jnp.allclose(norm_squared, manual)
assert jnp.allclose(norm_squared, 96.)

# Reference practice: Extend to a nested model
nested={"layer":params,"scale":jnp.array(1.)}
def nested_loss(p):
    pred=p["scale"]*(jnp.dot(p["layer"]["weight"],x)+p["layer"]["bias"])
    return (pred-3.)**2
g=jax.grad(nested_loss)(nested)
assert jax.tree.structure(g)==jax.tree.structure(nested)
assert jnp.allclose(g["scale"],-4.)

# Reference practice: Repair a mismatched update tree
try:
    jax.tree.map(lambda p,g:p-0.05*g,params,{"weight":grads["weight"]})
except ValueError:
    print("Expected tree mismatch")
else:
    raise AssertionError("Expected a missing-key failure")
fixed=jax.tree.map(lambda p,g:p-0.05*g,params,grads)
assert jnp.allclose(loss(fixed),0.64,atol=1e-6)
print("PASS: state-02")
