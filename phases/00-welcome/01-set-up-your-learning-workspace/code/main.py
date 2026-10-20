"""Set up your learning workspace: worked experiments and reference solutions. CPU checks."""

# Select the CPU backend before JAX loads
import os
# Choose CPU before importing JAX for this first experiment.
os.environ["JAX_PLATFORMS"] = "cpu"

# Print the interpreter and JAX runtime details
import platform
import jax
import numpy as np
print("Python:", platform.python_version())
print("JAX:", jax.__version__)

# Sum four JAX array elements and check the result
print("NumPy:", np.__version__)
print("Backend:", jax.default_backend())
print("Devices:", jax.devices())
x = jax.numpy.arange(4, dtype=jax.numpy.float32)
print("Sum:", float(x.sum()))
assert float(x.sum()) == 6.0

# Set up your learning workspace
import os
# Choose CPU before importing JAX for this first experiment.
os.environ["JAX_PLATFORMS"] = "cpu"
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
# Exercise solution
x = jax.numpy.arange(6, dtype=jax.numpy.float32)
print("Changed experiment sum:", float(x.sum()))
assert float(x.sum()) == 15.0


print("PASS: welcome-01")
