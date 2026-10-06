"""Meet your arrays and devices: worked experiments and reference solutions. CPU checks."""

# Prepare the inputs
import jax
import jax.numpy as jnp

# Build the computation
x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)

# Run and check the result
print("Shape:", x.shape)
print("Dtype:", x.dtype)
print("Devices:", x.devices())
print("Row sums:", x.sum(axis=1))
assert x.shape == (2, 3)
assert jnp.allclose(x.sum(axis=1), jnp.array([3., 12.]))

import jax
import jax.numpy as jnp
x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
print("Shape:", x.shape)
print("Dtype:", x.dtype)
print("Devices:", x.devices())
print("Row sums:", x.sum(axis=1))
assert x.shape == (2, 3)
assert jnp.allclose(x.sum(axis=1), jnp.array([3., 12.]))

# Figure data experiment
visual_data = {'kind': 'heatmap', 'values': x.tolist(), 'rows': ['row 0', 'row 1'], 'columns': ['feature 0', 'feature 1', 'feature 2'], 'unit': 'array value'}

# Experiment: Center each measurement
means=x.mean(axis=0,keepdims=True)
centered=x-means
print("Means:",means)
print("Centered:",centered)
assert means.shape==(1,3)
assert jnp.allclose(centered,jnp.array([[-1.5,-1.5,-1.5],[1.5,1.5,1.5]]))

# Experiment: Keep a host reference
import numpy as np
host=np.asarray(x)
assert host.shape==(2,3)
np.testing.assert_array_equal(host,np.arange(6).reshape(2,3))
print("Host dtype:",host.dtype)

# Reference solution. Try the exercise before reading this.
column_sums = x.sum(axis=0)
assert column_sums.shape == (3,)
assert jnp.allclose(column_sums, jnp.array([3., 5., 7.]))

# Reference practice: Add a new observation
table=jnp.arange(9,dtype=jnp.float32).reshape(3,3)
assert jnp.allclose(table.sum(axis=1),jnp.array([3.,12.,21.]))
assert jnp.allclose(table.mean(axis=0),jnp.array([3.,4.,5.]))

# Reference practice: Diagnose an impossible reshape
try:
    x.reshape(2,4)
except TypeError:
    print("Expected element-count mismatch")
else:
    raise AssertionError("Expected reshape to fail")
assert x.reshape(2,3).size==6
print("PASS: welcome-02")
