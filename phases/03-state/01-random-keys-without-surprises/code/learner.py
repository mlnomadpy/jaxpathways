"""Random keys without surprises · learner-only exercises (3/3 contracts ready)"""

# Exercise: Split a fresh seed into four keys, then use vmap to draw two normal values from each. Replay from the seed and compare.

# Inputs:
# - An integer seed and a batch-shaped input array x.

# Outputs:
# - A noisy prediction batch that can be replayed exactly from the same seed.

# Invariants to check:
# - Randomness is derived from an explicit JAX PRNG key.
# - Changing x while holding the seed fixed changes the deterministic prediction by the corresponding 2x+1 input difference.

# Hints:
# - Create a root key from the seed and draw one noise value for every element of x.
# - Do not reuse a consumed subkey for another independent draw.

import jax
import jax.numpy as jnp

def noisy_prediction(seed: int, x: jax.Array) -> jax.Array:
    """Return 2*x+1 plus reproducible elementwise Gaussian noise."""
    # TODO: derive a key from seed, sample noise with x.shape, and form the prediction.
    raise NotImplementedError




# Replay a noisy prediction (Practice):
# Draw one noise value per example for prediction \(2x+1\). Reconstruct the same batch from the seed, then change only \(x\).

# Inputs:
# - A seed and a one-dimensional input array x.

# Outputs:
# - A reproducible batch of noisy predictions.

# Invariants to check:
# - The same seed and x produce identical results.
# - Holding the seed fixed while shifting every x by one shifts each prediction by two.

# Hints:
# - Use one normal draw per input element.
# - Reconstruct the key from the seed on every call to make replay explicit.

import jax
import jax.numpy as jnp

def noisy_prediction(seed: int, x: jax.Array) -> jax.Array:
    """Add replayable, elementwise noise to the prediction 2*x+1."""
    # TODO: sample noise from a key made from seed and preserve x.shape.
    raise NotImplementedError




# Diagnose key reuse in a loop (Challenge):
# Write the broken loop that draws with one key three times, demonstrate repeated rows, then repair it by threading the continuation.

# Inputs:
# - A root PRNG key and a requested number of two-value draws.

# Outputs:
# - A broken batch showing repeated draws and a repaired batch of independent subkey draws.

# Invariants to check:
# - Reusing a key reproduces the same row.
# - Each repaired draw consumes a new key and has shape (2,).

# Hints:
# - Keep the broken case as a diagnostic demonstration.
# - Thread the continuation key through the loop; collect only the sample subkeys.

import jax
import jax.numpy as jnp

def compare_key_use(root: jax.Array, count: int = 3) -> tuple[jax.Array, jax.Array]:
    """Return repeated-key draws and draws made from distinct subkeys."""
    # TODO: build both batches and return arrays with shape (count, 2).
    raise NotImplementedError

