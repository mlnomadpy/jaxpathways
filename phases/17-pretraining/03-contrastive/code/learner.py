"""Contrastive learning: views, positives and negatives · learner-only exercises (3/3 contracts ready)"""

# Exercise: Permute only the right-hand embeddings after training and demonstrate the effect of broken pair identity.

# Inputs:
# - Left and right embeddings, a trained projection, and pair IDs.

# Outputs:
# - The original and right-permuted contrastive losses with a pair-identity diagnosis.

# Invariants to check:
# - Only the intended paired entries are positives; a permutation changes identity without changing vector values.

# Hints:
# - Apply the same projection before comparing both arrangements.
# - Keep the right-side permutation explicit.

import jax.numpy as jnp
import numpy as np

def compare_pairing_losses(left, right, projection, permutation):
    # TODO: measure the objective before and after permuting right-side pair identity.
    raise NotImplementedError



# Duplicate a source and inspect the objective (Transfer):
# Duplicate all four pairs while keeping diagonal-only positives. Calculate the change in loss and explain the false-negative effect.

# Inputs:
# - Four paired embeddings and a duplicated copy of each pair.

# Outputs:
# - Original and duplicated diagonal-positive losses and their difference.

# Invariants to check:
# - Duplicating candidates while retaining one diagonal positive changes the negative set.

# Hints:
# - Duplicate both sides in the same order.
# - Do not relabel the extra copies as positives.

import jax.numpy as jnp
import numpy as np

def inspect_duplicate_pair_effect(left, right, projection, temperature):
    # TODO: compare diagonal-positive losses before and after duplicating the batch.
    raise NotImplementedError



# Check the encoder gradient through normalization (Challenge):
# Perturb one encoder weight at the initial, noncollapsed matrix and compare centered differences with autodiff. Repeat with a second finite-difference step.

# Inputs:
# - A noncollapsed encoder weight matrix, fixed paired inputs, and two finite-difference step sizes.

# Outputs:
# - Autodiff and centered-difference estimates for one encoder weight.

# Invariants to check:
# - The tested parameter is away from a collapsed representation and both perturbations are symmetric.

# Hints:
# - Differentiate the full normalized-pair objective.
# - Report errors for each step size; finite differences have truncation and roundoff error.

import jax.numpy as jnp
import numpy as np

def check_encoder_gradient(weights, left, right, steps):
    # TODO: compare one autodiff gradient with centered differences at each requested step.
    raise NotImplementedError
