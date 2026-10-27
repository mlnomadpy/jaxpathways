"""Masked language modeling: predict hidden tokens · learner-only exercises (3/3 contracts ready)"""

# Exercise: Select the first position instead of the second and evaluate the trained model without updating it.

# Inputs:
# - A trained model apply function and parameters, token IDs, original target IDs, mask token ID, and a selection mask with the first position selected.

# Outputs:
# - Predictions and selected-token loss at the first position, evaluated without changing parameters.

# Invariants to check:
# - Only selected positions contribute to loss; clean target IDs are labels, not model inputs. Gradient from the selected loss may flow through shared attention into visible context representations.

# Hints:
# - Corrupt the selected input position while keeping its clean target separate.
# - Run the trained parameters in evaluation mode and do not apply an optimizer update.

import jax.numpy as jnp

def evaluate_alternate_mask(apply_model, params, tokens, targets, mask_id, selected):
    # TODO: corrupt the selected input, run the trained model, and score the selected clean target.
    raise NotImplementedError



# Reject a missing learning signal (Transfer):
# Pass an empty selection mask through the public corruption boundary and require a clear failure.

# Inputs:
# - A token batch and an all-false Boolean selection mask.

# Outputs:
# - A clear ValueError before any zero-count loss reduction.

# Invariants to check:
# - A batch with no selected targets cannot silently produce NaN or a zero training signal.

# Hints:
# - Validate the mask at the public corruption boundary.
# - Check shape and selected count before entering a transformed loss.

import jax.numpy as jnp
import numpy as np

def require_selected_positions(tokens, selected, mask_id):
    # TODO: validate the mask and return a corrupted token batch.
    raise NotImplementedError



# Derive the output gradient without autodiff (Challenge):
# Implement the probability-minus-one-hot formula for the unequal-mask fixture and compare every element with autodiff. Explain where visible context can still receive gradient.

# Inputs:
# - One logits tensor, integer target IDs, and a nonuniform selected mask.

# Outputs:
# - The analytical softmax-cross-entropy gradient and an autodiff comparison.

# Invariants to check:
# - Unselected token gradients are zero; selected token gradients sum to zero across vocabulary.

# Hints:
# - For each selected row, start from probability minus one-hot target.
# - Include the selected-count normalization and compare with autodiff.

import jax.numpy as jnp
import numpy as np

def compare_masked_logit_gradients(logits, targets, selected):
    # TODO: derive the masked gradient and compare it elementwise with autodiff.
    raise NotImplementedError
