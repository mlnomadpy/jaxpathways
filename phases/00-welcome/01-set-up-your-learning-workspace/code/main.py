"""Set up your learning workspace: worked experiments and reference solutions. CPU checks."""



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



# Reference solution. Try the exercise before reading this.
x = jax.numpy.arange(6, dtype=jax.numpy.float32)
print("Changed experiment sum:", float(x.sum()))
assert float(x.sum()) == 15.0


print("PASS: welcome-01")
