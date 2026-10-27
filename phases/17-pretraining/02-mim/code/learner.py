"""Masked image modeling: reconstruct missing patches · learner-only exercises (3/3 contracts ready)"""

# Exercise: Verify the bottom-right patch against a direct slice and explain why a whole-array reshape can hide a spatial permutation.

# Inputs:
# - A rectangular image batch, patch size, and a selected bottom-right patch index.

# Outputs:
# - Patch tokens and the original-space bottom-right patch verified by direct slicing.

# Invariants to check:
# - Patch ordering preserves row/column identity; reconstruction never receives hidden target pixels as encoder inputs.

# Hints:
# - Track the reshape and transpose axes before flattening.
# - Use direct slicing as an independent oracle.

import jax.numpy as jnp
import numpy as np

def check_bottom_right_patch(images, patch_size):
    # TODO: patchify the image and compare the final patch with a direct spatial slice.
    raise NotImplementedError



# Expose the leakage shortcut (Transfer):
# Compare a decoder that returns the clean target with the visible-only model. Explain why zero reconstruction loss is not sufficient evidence.

# Inputs:
# - A visible-only prediction function and parameters, clean patch targets, hidden-patch mask, and patch mask token.

# Outputs:
# - Hidden-region reconstruction errors for the target-copy shortcut and visible-only model, plus a leakage diagnosis.

# Invariants to check:
# - The shortcut has zero error because it receives clean targets; model evidence must come from predictions computed only from visible inputs.

# Hints:
# - Call the visible-only model using corrupted inputs and its parameters.
# - Measure both predictions on the same hidden patches.

import jax.numpy as jnp

def audit_target_leakage(predict_visible, params, clean_patches, hidden, mask_token):
    # TODO: compare the target-copy shortcut with predictions from the visible-only path.
    raise NotImplementedError



# Audit every patch of a rectangular color image (Challenge):
# Construct two 4-by-6 RGB images with numbered entries. Compare patchify with an independent nested slicing oracle for every patch, then change hidden pixels and prove visible features stay fixed.

# Inputs:
# - A patchify function, visible encoder and parameters, two numbered images of shape 4 by 6 by 3, patch size, and hidden-patch indices.

# Outputs:
# - Every patch checked against nested slicing and visible features shown unchanged after hidden pixels are altered.

# Invariants to check:
# - Patch order matches image coordinates; the encoder receives only visible patches, so changing hidden pixels cannot alter its features.

# Hints:
# - Build the patch oracle with explicit row and column loops.
# - Apply the same hidden mask and encoder to both images, changing hidden pixels only.

import jax.numpy as jnp
import numpy as np

def audit_rectangular_patch_layout(patchify, encode_visible, params, images, patch_size, hidden):
    # TODO: compare all patches with a slicing oracle and test hidden-pixel isolation through the encoder.
    raise NotImplementedError
