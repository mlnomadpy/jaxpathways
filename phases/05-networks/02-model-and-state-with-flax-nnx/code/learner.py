"""Model and state with Flax NNX · learner-only exercises (3/3 contracts ready)"""

# Exercise: Apply the model to one new row and to five rows. Verify the scalar parameter count stays \(13\) and the batch axis survives.

# Inputs:
# - The lesson’s Flax NNX model and one input row plus a five-row batch.

# Outputs:
# - Predictions whose leading dimensions match the input batch sizes, and the trainable scalar count.

# Invariants to check:
# - Applying the same model to one row and five rows preserves the leading batch axis; the parameter count remains unchanged.

# Hints:
# - Use inputs shaped (1, 2) and (5, 2), then inspect output shapes.
# - Count only nnx.Param leaves, not the non-parameter call counter.

from flax import nnx
import jax
import jax.numpy as jnp

def check_model_batch_contract(model, single_row, five_rows):
    # TODO: Apply the model to both batches and verify output shapes and trainable scalar count.
    raise NotImplementedError




# Change a clone without changing its source (Transfer):
# Increase only the cloned model’s output bias by one. Predict the change in every logit and verify the original model’s parameter leaves remain unchanged.

# Inputs:
# - Increase only the cloned model’s output bias by one. Predict the change in every logit and verify the original model’s parameter leaves remain unchanged.

# Outputs:
# - Logit shift on the clone and an unchanged source snapshot.

# Invariants to check:
# - The clone owns independent variable state rather than aliasing the original module.

# Hints:
# - Snapshot the original parameters before editing clone.out.bias.

from flax import nnx
import jax.numpy as jnp

def practice_change_a_clone_without_changing_its_source(model, cloned_model):
    """Implement the change a clone without changing its source check."""
    # TODO: Clone model state, increment only the clone output bias, and compare all source parameter leaves before and after.
    raise NotImplementedError




# Diagnose a shared module reference (Intermediate):
# Assign `shared = model`, increment its counter, and demonstrate why this is not a snapshot. Then reconstruct an independent module.

# Inputs:
# - Assign `shared = model`, increment its counter, and demonstrate why this is not a snapshot. Then reconstruct an independent module.

# Outputs:
# - The shared-reference mutation and an isolated copy check.

# Invariants to check:
# - Assignment preserves object identity; a snapshot must preserve values without shared mutable state.

# Hints:
# - Assignment changes which name refers to the same object.

from flax import nnx

def practice_diagnose_a_shared_module_reference(model):
    """Implement the diagnose a shared module reference check."""
    # TODO: Assign an alias, mutate its counter, compare the original object, and then construct an independent copy.
    raise NotImplementedError

