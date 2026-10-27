"""A sharded training step · learner-only exercises (4/4 contracts ready)"""

# Exercise: Run a changed target/initial-state update, then diagnose the factor-of-four local-mean bug and repair uneven-batch aggregation.

# Inputs:
# - Feature matrix and labels for a linear model; changed true weights [-1, 3], initial weights [0.2, -0.1], and learning rate 0.05.
# - Four equal-size host shards and one uneven-size partition of the same examples.

# Outputs:
# - One changed-target parameter update and an independent gradient check.
# - Four local mean gradients, their naive sum, the correctly averaged global gradient, and an example-count-weighted uneven-shard result.

# Invariants to check:
# - Summing four local means scales the result by four; averaging equal-size local means recovers the global mean.
# - Unequal local batches must be weighted by example count.
# - The update uses the changed target and starts from the supplied initial weights.

# Hints:
# - Derive labels from the changed true weights, then differentiate the mean loss.
# - Compare local-mean aggregation separately from the changed one-step update.
# - For unequal shards, weight each local mean by its local count.

import numpy as np
import jax
import jax.numpy as jnp

def diagnose_sharded_update(features, labels, true_weights, initial_weights, learning_rate, equal_shards, uneven_shards):
    # TODO: derive the changed target update and compare local/global gradients.
    # Report both the naive local-mean sum and count-weighted uneven aggregation.
    raise NotImplementedError("Separate the changed training step from local-mean aggregation")



# Transfer to a different target model (Challenge):
# Change true weights to \([-1,3]\), start at \([0.2,-0.1]\) and use learning rate \(0.05\). Derive and check the new update without reusing the original numeric answer.

# Inputs:
# - Features and labels for a linear model, changed true weights [-1,3], starting weights [0.2,-0.1], and step size 0.05.

# Outputs:
# - Changed one-step parameter vector.
# - Independent objective and gradient comparison.

# Invariants to check:
# - The check derives labels from the changed true weights.
# - The update starts from the supplied initial weights.

# Hints:
# - Use a mean over examples in the loss.
# - Keep truth weights separate from the fitted parameter vector.

import numpy as np
import jax
import jax.numpy as jnp

def transfer_update(features, initial_weights, true_weights, learning_rate):
    # TODO: build labels, differentiate the mean loss, and check the update independently.
    raise NotImplementedError("Implement the changed target and initial state")



# Diagnose summed local means (Challenge):
# Compute four host local mean gradients. Show why summing them gives four times the global mean, then repair the aggregation. Also explain the unequal-batch generalization.

# Inputs:
# - Four equal-sized feature/label shards and a parameter vector.
# - An additional unequal-sized partitioning for the generalization check.

# Outputs:
# - Local mean gradients, their sum, and the correctly aggregated global gradient.

# Invariants to check:
# - Summing four local means is scaled by the number of shards.
# - Unequal shard sizes require weighting by example count.

# Hints:
# - Compare sums of losses with sums of gradients before averaging.
# - Weight each local mean by its local count for unequal partitions.

import numpy as np

def compare_local_mean_gradients(feature_shards, label_shards, weights):
    # TODO: calculate each local mean gradient and both aggregations.
    # Also return an example-count-weighted aggregation.
    raise NotImplementedError("Diagnose local-mean scaling and size-weight the shards")



# Change the mesh-sized regression fixture (Transfer):
# Change the padded batch to twelve rows and use a 2x2 mesh. Choose seven valid rows. State the valid counts on each data shard, preserve the parameter-column and momentum placement, then compare loss, gradient and one update with the independent unsharded formula. Report local shapes and bytes.

# Inputs:
# - A 2x2 logical CPU mesh, padded eight-row feature and target arrays, a valid-row mask, a (4,4) weight matrix and matching momentum.
# - Two mask cases: five valid examples (counts 4/1) and three valid examples (counts 3/0).

# Outputs:
# - Per-device local shapes and float32 bytes for batch, target, parameter, momentum and gradient shards.
# - Masked loss, gradient, updated weights and updated momentum checked against an independent unsharded calculation.

# Invariants to check:
# - X uses P("data",None); W and momentum use P(None,"model"); prediction and target use P("data","model").
# - The loss divides by valid-example count times output width.
# - An empty local shard contributes zero numerator and zero valid count.

# Hints:
# - Use jax.value_and_grad for the global masked objective and an explicitly sharded jax.jit call.
# - For 4/1 counts, compare equal averaging of local means with the correctly weighted global mean.

import jax
import jax.numpy as jnp
import numpy as np

def parameter_sharded_update(mesh, features, targets, weights, momentum, valid_rows):
    # TODO: make NamedSharding objects for data, model, targets, and parameters.
    # Place inputs, differentiate the masked global mean, and apply one momentum step.
    # Return loss, gradient, new weights, new momentum, and a per-device byte ledger.
    raise NotImplementedError("Complete and verify the 2x2 parameter-sharded update")
