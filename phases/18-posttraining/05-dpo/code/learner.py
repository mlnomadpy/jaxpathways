"""Direct preference optimization and reference-corrected margins · learner-only exercises (3/3 contracts ready)"""

# Exercise: Swap chosen and rejected IDs after training and verify that the loss rises.

# Inputs:
# - Policy and reference log probabilities plus chosen and rejected comparison IDs.

# Outputs:
# - Pairwise loss before and after swapping the chosen/rejected identities.

# Invariants to check:
# - The reference correction uses the same frozen reference for both response IDs.

# Hints:
# - Swap IDs without changing either probability table.
# - Evaluate the loss at the trained policy.

import jax.numpy as jnp
import numpy as np

def compare_swapped_preference_loss(policy_logps, reference_logps, chosen, rejected):
    # TODO: evaluate the corrected objective under original and swapped pair identities.
    raise NotImplementedError



# Freeze the reference numerically and in autodiff (Transfer):
# Differentiate with respect to reference log probabilities and verify that their gradient is zero.

# Inputs:
# - Policy logits, reference log probabilities, chosen/rejected IDs, and beta.

# Outputs:
# - The reference-input gradient of the DPO objective.

# Invariants to check:
# - Reference values affect the numeric margin but are stopped from receiving optimization gradients.

# Hints:
# - Differentiate with respect to the reference argument only.
# - Check the gradient shape and that every entry is zero.

import jax.numpy as jnp
import numpy as np

def check_reference_gradient(policy_logits, reference_logps, chosen, rejected, beta):
    # TODO: differentiate the pair objective with respect to the reference values.
    raise NotImplementedError



# Derive a one-pair categorical gradient (Challenge):
# At a fresh three-action policy, compute the corrected margin and derive the logit gradient for one chosen/rejected pair. Why does the third action have zero direct logit gradient here?

# Inputs:
# - Three-action policy logits, fixed reference log probabilities, one chosen ID, one rejected ID, and beta.

# Outputs:
# - Corrected margin and analytic logit-gradient vector checked against autodiff.

# Invariants to check:
# - Only chosen and rejected action logits have direct gradient; the third action cancels in the pairwise log-probability difference.

# Hints:
# - Write the log-softmax derivative for each selected action.
# - Check all three entries against autodiff.

import jax.numpy as jnp
import numpy as np

def derive_one_pair_logit_gradient(logits, reference_logps, chosen, rejected, beta):
    # TODO: derive the corrected pair margin and compare the three logit derivatives.
    raise NotImplementedError
