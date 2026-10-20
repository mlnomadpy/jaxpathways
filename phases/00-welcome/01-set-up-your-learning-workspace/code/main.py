"""Set up your learning workspace: worked experiments and reference solutions. CPU checks."""

# Step 1: Set up imports and input tensors
import os
# Choose CPU before importing JAX for this first experiment.
os.environ["JAX_PLATFORMS"] = "cpu"

# Step 2: Apply the core JAX transformation
import platform
import jax

# Step 3: Verify shapes and numerical invariants
import numpy as np
x = jax.numpy.arange(4, dtype=jax.numpy.float32)
assert float(x.sum()) == 6.0

# Set up your learning workspace: Your first goal is a small, repeatable result: save a Python...
import os
# Choose CPU before importing JAX for this first experiment.
os.environ["JAX_PLATFORMS"] = "cpu"
# Import required JAX, NumPy, and standard-library modules.
import platform
import jax
import numpy as np
print("Python:", platform.python_version())
print("JAX:", jax.__version__)
print("NumPy:", np.__version__)
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
x = jax.numpy.arange(4, dtype=jax.numpy.float32)
print("Sum:", float(x.sum()))
assert float(x.sum()) == 6.0

# Figure data experiment
# Plot the 1D verification array `x` and its elementwise square `x ** 2`:
sq_vals = x ** 2
visual_data = {'kind': 'bar', 'x': list(range(len(x))), 'labels': [f'x[{i}]' for i in range(len(x))], 'xlabel': 'element index', 'ylabel': 'array value', 'series': [{'label': 'x', 'y': [float(v) for v in x]}, {'label': 'x ** 2', 'y': [float(v) for v in sq_vals]}]}



# Reference solution. Try the exercise before reading this.
# Exercise solution: Edit the starter as shown in Step 6: change the array length from 4 to...
x = jax.numpy.arange(6, dtype=jax.numpy.float32)
print("Changed experiment sum:", float(x.sum()))
assert float(x.sum()) == 15.0


print("PASS: welcome-01")
