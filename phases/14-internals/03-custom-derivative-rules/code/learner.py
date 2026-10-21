"""Custom derivative rules · learner-only exercises (3/3 contracts ready)"""

# Exercise: Check the custom log-sum-exp gradient on \((-3,0.2,2,5)\) against an independent stable softmax and central differences along two directions. Keep the output cotangent explicit.

# Inputs:
# - Logits (-3, 0.2, 2, 5), explicit scalar output cotangent, and two directions.

# Outputs:
# - Custom-gradient result, stable softmax reference, and centered directional differences.

# Invariants to check:
# - The custom gradient scales by the explicit cotangent.
# - Softmax reference is stable under large logits.

# Hints:
# - Subtract the maximum before exponentiating.
# - Use the same cotangent for autodiff and independent directional checks.

import numpy as np
import jax
import jax.numpy as jnp

def check_custom_logsumexp(logits, cotangent, directions, epsilon):
    # TODO: compare custom VJP with stable softmax and directional differences.
    raise NotImplementedError("Audit the explicit cotangent and two independent directions")



# Check common-shift invariance of curvature (Practice):
# Add a common constant of \(1000\) to every logit. Verify the value shift, unchanged gradient, and Hessian row sums.

# Inputs:
# - A stable log-sum-exp custom derivative, a logit vector, and a common shift of 1000.

# Outputs:
# - Value shift, gradient comparison, and Hessian row sums before/after shift.

# Invariants to check:
# - The value increases by the common shift.
# - Gradient stays invariant and Hessian has near-zero row sums.

# Hints:
# - Use centered logits for numerical stability.
# - Compute Hessian row sums from derivatives, not from a stored expected answer.

import jax
import jax.numpy as jnp

def check_shift_curvature(logsumexp, logits, shift):
    # TODO: compare values, gradients, and Hessian row sums under a common shift.
    raise NotImplementedError("Check shift invariance of first and second derivatives")



# Audit a nonlinear seed rule before registering it (Challenge):
# A proposed JVP returns \(\sigma(x)\dot{x}^2\). Show explicitly why it cannot be the derivative map, then repair it and test a mixed linear combination.

# Inputs:
# - A proposed JVP rule returning sigmoid(x)*dx**2 and a repaired tangent-linear rule.

# Outputs:
# - A mixed-seed counterexample to additivity and a repaired-rule check.

# Invariants to check:
# - A derivative is linear in the tangent at fixed primal input.
# - The repaired rule passes mixed linear-combination checks.

# Hints:
# - Compare the rule at dx1, dx2, and dx1+dx2.
# - Keep the primal input fixed in each additivity test.

import jax.numpy as jnp

def audit_jvp_rule(primal, dx1, dx2):
    # TODO: expose nonlinearity in the proposed seed rule, then repair it.
    raise NotImplementedError("Require additivity and homogeneity in every tangent seed")
