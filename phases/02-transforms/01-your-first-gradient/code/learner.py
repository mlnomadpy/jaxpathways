"""Your first gradient · learner-only exercises (3/3 contracts ready)"""

# Exercise: Foundation · Define \(f(x)=x^3\). Derive \(f\prime(x)\), predict the values at \(-2\), \(0\), and \(3\), then verify every prediction with JAX. Explain why the zero at \(x=0\) is separate from a minimum.

# Inputs:
# - A scalar cubic function defined by the learner.
# - Evaluation points -2, 0, and 3.

# Outputs:
# - The function and its JAX derivatives at each point.
# - An explanation that separates a stationary point from a minimum.

# Invariants to check:
# - `grad` receives a scalar-output function.
# - Every requested point is checked.
# - A zero derivative alone does not classify a minimum.

# Hints:
# - Derive the polynomial rule before calling autodiff.
# - Check function behavior on either side of the stationary point.

import jax
import jax.numpy as jnp

def cubic(x):
    # TODO: define the scalar cubic
    raise NotImplementedError

def check_cubic(points):
    # TODO: evaluate and retain one JAX derivative per point
    raise NotImplementedError




# Differentiate the parameter you intended (Practice):
# For the prediction \(\hat y=wx\), find the derivative with respect to \(w\) and then with respect to \(x\). Evaluate them at \(w=2\) and \(x=3\). Predict both answers before choosing `argnums`.

# Inputs:
# - The scalar prediction `weight * x`.
# - Weight 2 and input 3.
# - Two requested differentiation arguments.

# Outputs:
# - A derivative with respect to each argument.

# Invariants to check:
# - The unselected argument remains an ordinary input.
# - Distinguish prediction from derivative.

# Hints:
# - Name the differentiation argument explicitly.
# - Check each derivative against its own input.

import jax

def scalar_prediction(weight, x):
    # TODO: define the scalar prediction
    raise NotImplementedError

def compare_argument_gradients():
    # TODO: transform the function for each requested argument
    raise NotImplementedError




# Repair an objective without hiding its meaning (Challenge):
# A function returns \([x^2,(x-2)^2]\). Explain why ordinary grad rejects it. Build a scalar objective using their mean, derive its derivative, and check \(x=0\), \(1\), and \(2\). Would a sum have the same derivative magnitude?

# Inputs:
# - A two-element output containing squared costs.
# - Inputs 0, 1, and 2.

# Outputs:
# - A scalar mean objective and its derivatives.
# - A comparison with the sum objective’s derivative scale.

# Invariants to check:
# - `grad` receives one scalar objective.
# - Mean and sum remain distinct objectives.

# Hints:
# - Choose the reduction before differentiating.
# - Compare the factors analytically.

import jax
import jax.numpy as jnp

def mean_objective(x):
    # TODO: construct the two costs and reduce to one scalar
    raise NotImplementedError

def compare_reductions(points):
    # TODO: evaluate derivatives for the mean and sum objectives
    raise NotImplementedError

