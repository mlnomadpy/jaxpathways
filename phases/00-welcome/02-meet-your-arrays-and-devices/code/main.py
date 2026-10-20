"""Meet your arrays and devices: worked experiments and reference solutions. CPU checks."""

# Prepare the inputs
# Step 1 — Prepare the inputs: These explicit inputs define the case that the later checks will...
import jax
import jax.numpy as jnp

# Build the computation
# Step 2 — Build the computation: x has two observations and three features.
x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)

# Run and check the result
# Step 3 — Run and check the result: Compare the output to the expected result below before making the...
print("Shape:", x.shape)
print("Dtype:", x.dtype)
print("Devices:", x.devices())
print("Row sums:", x.sum(axis=1))
# Check tensor shape invariant: `x.shape == (2, 3)`
assert x.shape == (2, 3)
assert jnp.allclose(x.sum(axis=1), jnp.array([3., 12.]))

# Step 1 — Prepare the inputs: These explicit inputs define the case that the later checks will...
import jax
import jax.numpy as jnp
# Step 2 — Build the computation: x has two observations and three features.
x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
# Step 3 — Run and check the result: Compare the output to the expected result below before making the...
print("Shape:", x.shape)
print("Dtype:", x.dtype)
print("Devices:", x.devices())
print("Row sums:", x.sum(axis=1))
# Check tensor shape invariant: `x.shape == (2, 3)`
assert x.shape == (2, 3)
assert jnp.allclose(x.sum(axis=1), jnp.array([3., 12.]))

# Figure data experiment
visual_data = {'kind': 'heatmap', 'values': x.tolist(), 'rows': ['row 0', 'row 1'], 'columns': ['feature 0', 'feature 1', 'feature 2'], 'unit': 'array value'}

# Experiment: Center each measurement
# Experiment — Center each measurement: The calculation reuses one mean per column.
# Reduce along axis=0 to compute `means`.
means=x.mean(axis=0,keepdims=True)
centered=x-means
print("Means:",means)
print("Centered:",centered)
# Check tensor shape invariant: `means.shape==(1,3)`
assert means.shape==(1,3)
assert jnp.allclose(centered,jnp.array([[-1.5,-1.5,-1.5],[1.5,1.5,1.5]]))

# Experiment: Keep a host reference
# Experiment — Keep a host reference: The host reference is an inspection copy.
import numpy as np
# Transfer the device result to the host for an independent NumPy comparison.
host=np.asarray(x)
# Check tensor shape invariant: `host.shape==(2,3)`
assert host.shape==(2,3)
# Create evenly spaced index values in ``.
np.testing.assert_array_equal(host,np.arange(6).reshape(2,3))
print("Host dtype:",host.dtype)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Compute column sums instead.
# Reduce along axis=0 to compute `column_sums`.
column_sums = x.sum(axis=0)
# Check tensor shape invariant: `column_sums.shape == (3,)`
assert column_sums.shape == (3,)
assert jnp.allclose(column_sums, jnp.array([3., 5., 7.]))

# Reference practice: Add a new observation
# Add a new observation (Practice): Changing the observation count changes axis 0, while the...
table=jnp.arange(9,dtype=jnp.float32).reshape(3,3)
assert jnp.allclose(table.sum(axis=1),jnp.array([3.,12.,21.]))
assert jnp.allclose(table.mean(axis=0),jnp.array([3.,4.,5.]))

# Reference practice: Diagnose an impossible reshape
# Diagnose an impossible reshape (Challenge): An incompatible reshape is an element-count error.
try:
    x.reshape(2,4)
except TypeError:
    print("Expected element-count mismatch")
else:
    raise AssertionError("Expected reshape to fail")
assert x.reshape(2,3).size==6
print("PASS: welcome-02")
