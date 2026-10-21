"""RLHF mechanics: a frozen reward and PPO policy updates · learner-only exercises (3/3 contracts ready)"""

# Exercise: Check the KL at the fixed reference and compare it with the final policy’s exact KL.

# Inputs:
# - Reference-policy log probabilities and final-policy log probabilities for the same categorical action set.

# Outputs:
# - The self-KL and \(D_{\mathrm{KL}}(\pi_{\mathrm{final}}\parallel\pi_{\mathrm{ref}})\) for the exact categorical distributions.

# Invariants to check:
# - The self-KL is zero up to floating-point tolerance; compute \(D_{\mathrm{KL}}(\pi_{\mathrm{final}}\parallel\pi_{\mathrm{ref}})=\sum_a \pi_{\mathrm{final}}(a)\log(\pi_{\mathrm{final}}(a)/\pi_{\mathrm{ref}}(a))\), never the reverse direction.

# Hints:
# - Normalize both log-probability vectors.
# - Weight each log ratio by the corresponding final-policy probability.

import jax.numpy as jnp

def compare_reference_and_policy_kl(reference_logps, final_policy_logps):
    # TODO: compute self-KL and D_KL(final policy || frozen reference).
    raise NotImplementedError



# Verify detached behavior information (Transfer):
# Differentiate the objective with respect to supplied old log probabilities and advantages; both must be fixed within this update.

# Inputs:
# - A callable PPO objective, policy logits, supplied old action log probabilities, sampled actions, advantages, reference log probabilities, and PPO coefficients.

# Outputs:
# - Gradients of the objective with respect to old log probabilities and advantages.

# Invariants to check:
# - Both rollout-time inputs are stop-gradient constants in this update, while policy logits remain differentiable.

# Hints:
# - Differentiate the supplied objective with respect to the two rollout arrays only.
# - Check zero arrays with their original shapes.

import jax
import jax.numpy as jnp

def check_rollout_inputs_are_detached(ppo_objective, logits, old_logps, actions, advantages, reference_logps, clip, beta):
    # TODO: differentiate the objective with respect to old_logps and advantages.
    raise NotImplementedError



# Compare with the finite-action regularized optimum (Challenge):
# Compute the analytic optimum for beta 0.2, then compare its regularized objective with the trained policy and the reference.

# Inputs:
# - A finite action set, frozen reference distribution, reward vector, beta equal to 0.2, and trained policy.

# Outputs:
# - The analytic regularized optimum and matching regularized objective values for the reference and trained policy.

# Invariants to check:
# - Every objective uses D_KL(policy || reference) with the same reward, frozen reference, and beta.

# Hints:
# - Normalize reward-adjusted reference log probabilities with log-sum-exp.
# - Compare objective values, including the policy-to-reference KL, rather than only the highest-probability action.

import jax.numpy as jnp
import numpy as np

def compare_policy_with_regularized_optimum(reward, reference_probs, policy_probs, beta):
    # TODO: construct the finite-action optimum and evaluate all policies with D_KL(policy || reference).
    raise NotImplementedError
