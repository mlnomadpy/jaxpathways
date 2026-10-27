"""Learner implementation contract for L10B's tiny multiprocess update."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def local_numerator_and_gradient(weights, x, y, valid):
    """Return summed squared-error and gradient for this rank's valid rows."""
    raise NotImplementedError("compute a masked sum, not a local mean")


def global_transition(weights, momentum, gathered_gradient_sums, gathered_counts,
                      learning_rate=0.05, momentum_rate=0.8):
    """Apply one count-weighted global momentum update to replicated state."""
    raise NotImplementedError("divide once by the global valid count")
