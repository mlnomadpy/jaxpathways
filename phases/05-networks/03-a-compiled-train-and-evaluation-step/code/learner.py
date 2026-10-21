"""A compiled train and evaluation step · learner-only exercises (3/3 contracts ready)"""

# Exercise: Compute held-out metrics in three equal chunks and reconstruct the overall means. Then confirm no state changed.

# Inputs:
# - A model and held-out arrays split into chunks with a final partial batch.

# Outputs:
# - Count-weighted aggregate loss/accuracy and an unchanged model snapshot.

# Invariants to check:
# - Metrics are weighted by example counts; evaluation does not update model or optimizer state.

# Hints:
# - Accumulate metric sums and counts, then divide once.
# - Use a separate clone when deliberately testing a training call on held-out data.

import jax.numpy as jnp

def evaluate_in_chunks(model, features, labels, chunk_size):
    """Count-weighted aggregate loss/accuracy and an unchanged model snapshot."""
    # TODO: Aggregate held-out metric sums by example count and assert evaluation leaves model state unchanged.
    raise NotImplementedError




# Aggregate unequal batches (Transfer):
# Evaluate the held-out set in batches of seven and forty-one observations. Reconstruct both metrics from sums and counts. Use a separate constructed count example to demonstrate why averaging batch accuracies can be wrong.

# Inputs:
# - Evaluate the held-out set in batches of seven and forty-one observations. Reconstruct both metrics from sums and counts. Use a separate constructed count example to demonstrate why averaging batch accuracies can be wrong.

# Outputs:
# - Count-weighted aggregate metrics and a constructed unequal-count accuracy example.

# Invariants to check:
# - Batch metrics are weighted by the number of examples, not averaged equally.

# Hints:
# - Weight each batch mean by its number of observations.

import jax.numpy as jnp

def practice_aggregate_unequal_batches(model, features, labels, batch_sizes):
    """Implement the aggregate unequal batches check."""
    # TODO: Accumulate metric sums and example counts for batch sizes seven and forty-one, then compare with concatenated evaluation.
    raise NotImplementedError




# Catch an evaluation function that trains (Intermediate):
# On an independent clone, deliberately call train_step with held-out data and show the isolation checks fail. Keep the real model untouched.

# Inputs:
# - On an independent clone, deliberately call train_step with held-out data and show the isolation checks fail. Keep the real model untouched.

# Outputs:
# - A detected clone mutation and an unchanged real model snapshot.

# Invariants to check:
# - The model used for held-out evaluation and the model used for the deliberate failure are distinct.

# Hints:
# - Compare both optimizer.step and model parameters around the bad call.

from flax import nnx
import jax.numpy as jnp

def practice_catch_an_evaluation_function_that_trains(model, optimizer, heldout_features, heldout_labels):
    """Implement the catch an evaluation function that trains check."""
    # TODO: Run the mutating train_step only on a disposable clone and compare its state snapshot before and after evaluation.
    raise NotImplementedError

