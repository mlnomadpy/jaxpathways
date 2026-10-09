"""Set up your learning workspace: worked experiments and reference solutions. CPU checks."""



# Set up your learning workspace: Your first goal is a small, repeatable result: save a Python...
# Import os for this computation.
import os
# Choose CPU before importing JAX for this first experiment.
os.environ["JAX_PLATFORMS"] = "cpu"
# Import required JAX, NumPy, and standard-library modules.
import platform
import jax
import numpy as np
# Print the observed values to compare against the expected result.
print("Python:", platform.python_version())
# Print diagnostic summary of the computed outputs.
print("JAX:", jax.__version__)
# Print diagnostic summary of the computed outputs.
print("NumPy:", np.__version__)
# Print diagnostic summary of the computed outputs.
print("Backend:", jax.default_backend())
# Print diagnostic summary of the computed outputs.
print("Devices:", jax.devices())
# Run `jax.numpy.arange` to compute `x`.
x = jax.numpy.arange(4, dtype=jax.numpy.float32)
# Print the observed values to compare against the expected result.
print("Sum:", float(x.sum()))
# Verify contract: `float(x.sum()) == 6.0`.
assert float(x.sum()) == 6.0



# Reference solution. Try the exercise before reading this.
# Exercise solution: Edit the starter as shown in Step 6: change the array length from 4 to...
x = jax.numpy.arange(6, dtype=jax.numpy.float32)
# Print the observed values to compare against the expected result.
print("Changed experiment sum:", float(x.sum()))
# Verify contract: `float(x.sum()) == 15.0`.
assert float(x.sum()) == 15.0


print("PASS: welcome-01")
