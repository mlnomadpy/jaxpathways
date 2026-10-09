"""Pytrees and structured parameters: worked experiments and reference solutions. CPU checks."""

# Prepare the inputs
# Step 1 — Prepare the inputs: These explicit inputs define the case that the later checks will...
# Import jax for this computation.
import jax
import jax.numpy as jnp

# Build the computation
# Step 2 — Build the computation: loss reads both parameter leaves.
# Construct `params` via `{"weight": jnp.array([1., -1.]), "bias": jnp.array(0.)}`
params = {"weight": jnp.array([1., -1.]), "bias": jnp.array(0.)}
# Construct `x` via `jnp.array([2., 1.])`
x = jnp.array([2., 1.])
# Function `loss(p)` implementing this stage's computation:
def loss(p):
    # Run `jnp.dot` to compute `prediction`.
    prediction = jnp.dot(p["weight"], x) + p["bias"]
    # Return `(prediction - 3.0) ** 2` to the caller.
    return (prediction - 3.) ** 2

# Run and check the result
# Step 3 — Run and check the result: Compare the output to the expected result below before making the...
# Differentiate the objective to obtain `(value, grads)` via automatic differentiation.
value, grads = jax.value_and_grad(loss)(params)
# Transform every leaf of the parameter PyTree (`updated`).
updated = jax.tree.map(lambda p, g: p - 0.05 * g, params, grads)
# Print the observed values to compare against the expected result.
print("Before/after:", float(value), float(loss(updated)))
# Assert invariant `grads.keys() == params.keys()` holds
assert grads.keys() == params.keys()
# Assert invariant `loss(updated) < value` holds
assert loss(updated) < value

# Step 1 — Prepare the inputs: These explicit inputs define the case that the later checks will...
# Import jax for this computation.
import jax
import jax.numpy as jnp
# Step 2 — Build the computation: loss reads both parameter leaves.
# Construct `params` via `{"weight": jnp.array([1., -1.]), "bias": jnp.array(0.)}`
params = {"weight": jnp.array([1., -1.]), "bias": jnp.array(0.)}
# Construct `x` via `jnp.array([2., 1.])`
x = jnp.array([2., 1.])
# Function `loss(p)` implementing this stage's computation:
def loss(p):
    # Run `jnp.dot` to compute `prediction`.
    prediction = jnp.dot(p["weight"], x) + p["bias"]
    # Return `(prediction - 3.0) ** 2` to the caller.
    return (prediction - 3.) ** 2
# Step 3 — Run and check the result: Compare the output to the expected result below before making the...
# Differentiate the objective to obtain `(value, grads)` via automatic differentiation.
value, grads = jax.value_and_grad(loss)(params)
# Transform every leaf of the parameter PyTree (`updated`).
updated = jax.tree.map(lambda p, g: p - 0.05 * g, params, grads)
# Print the observed values to compare against the expected result.
print("Before/after:", float(value), float(loss(updated)))
# Assert invariant `grads.keys() == params.keys()` holds
assert grads.keys() == params.keys()
# Assert invariant `loss(updated) < value` holds
assert loss(updated) < value

# Figure data experiment
# Compute figure data for: A gradient tree matches parameter leaves
# Compute `visual_data` from `{'kind': 'bar', 'labels': ['weight[0]', 'weight[1]',...`
visual_data = {'kind': 'bar', 'labels': ['weight[0]', 'weight[1]', 'bias'], 'ylabel': 'parameter value', 'series': [{'label': 'before', 'y': [float(params['weight'][0]), float(params['weight'][1]), float(params['bias'])]}, {'label': 'after', 'y': [float(updated['weight'][0]), float(updated['weight'][1]), float(updated['bias'])]}]}

# Experiment: Verify every gradient leaf
# Experiment — Verify every gradient leaf: This check verifies parameter identity and the actual gradient...
# Assert that `jnp.allclose(grads["weight"],jnp.array([-8.,-4.]))`.
assert jnp.allclose(grads["weight"],jnp.array([-8.,-4.]))
# Assert that `jnp.allclose(grads["bias"],-4.)`.
assert jnp.allclose(grads["bias"],-4.)
# Assert that `jnp.allclose(updated["weight"],jnp.array([1.4,-0.8]))`.
assert jnp.allclose(updated["weight"],jnp.array([1.4,-0.8]))
# Assert that `jnp.allclose(updated["bias"],0.2)`.
assert jnp.allclose(updated["bias"],0.2)
# Assert that `jnp.allclose(loss(updated),0.64,atol=1e-6)`.
assert jnp.allclose(loss(updated),0.64,atol=1e-6)

# Experiment: Flatten and reconstruct without losing identity
# Experiment — Flatten and reconstruct without losing identity: The tree definition describes containers; leaves supply...
leaves,definition=jax.tree.flatten(params)
# Run `jax.tree.unflatten` to compute `rebuilt`.
rebuilt=jax.tree.unflatten(definition,leaves)
# Assert invariant `jax.tree.structure(rebuilt)==jax.tree.structure(params)` holds
assert jax.tree.structure(rebuilt)==jax.tree.structure(params)
# Assert invariant `jnp.array_equal(rebuilt["weight"]` holds
assert jnp.array_equal(rebuilt["weight"],params["weight"])
# Assert invariant `jnp.array_equal(rebuilt["bias"]` holds
assert jnp.array_equal(rebuilt["bias"],params["bias"])

# Reference solution. Try the exercise before reading this.
# Exercise solution: Compute the squared norm of all gradient leaves.
# Transform every leaf of the parameter PyTree (`norm_squared`).
norm_squared = sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads))
# Aggregate array values to compute `manual`.
manual = jnp.sum(grads["weight"] ** 2) + grads["bias"] ** 2
# Assert that `jnp.allclose(norm_squared, manual)`.
assert jnp.allclose(norm_squared, manual)
# Assert that `jnp.allclose(norm_squared, 96.)`.
assert jnp.allclose(norm_squared, 96.)

# Reference practice: Extend to a nested model
# Extend to a nested model (Practice): Nested parameter identity survives differentiation.
# Construct `nested` via `{"layer":params,"scale":jnp.array(1.)}`
nested={"layer":params,"scale":jnp.array(1.)}
# Function `nested_loss(p)` implementing this stage's computation:
def nested_loss(p):
    # Compute `pred` from `p["scale"]*(jnp.dot(p["layer"]["weight"],x)+p["layer...`
    pred=p["scale"]*(jnp.dot(p["layer"]["weight"],x)+p["layer"]["bias"])
    # Return `(pred - 3.0) ** 2` to the caller.
    return (pred-3.)**2
# Differentiate the objective to obtain `g` via automatic differentiation.
g=jax.grad(nested_loss)(nested)
# Assert invariant `jax.tree.structure(g)==jax.tree.structure(nested)` holds
assert jax.tree.structure(g)==jax.tree.structure(nested)
# Assert that `jnp.allclose(g["scale"],-4.)`.
assert jnp.allclose(g["scale"],-4.)

# Reference practice: Repair a mismatched update tree
# Repair a mismatched update tree (Challenge): A missing bias gradient is a tree mismatch.
# Run the boundary check and catch the expected exception:
try:
    jax.tree.map(lambda p,g:p-0.05*g,params,{"weight":grads["weight"]})
except ValueError:
    print("Expected tree mismatch")
else:
    raise AssertionError("Expected a missing-key failure")
# Transform every leaf of the parameter PyTree (`fixed`).
fixed=jax.tree.map(lambda p,g:p-0.05*g,params,grads)
# Assert that `jnp.allclose(loss(fixed),0.64,atol=1e-6)`.
assert jnp.allclose(loss(fixed),0.64,atol=1e-6)
print("PASS: state-02")
