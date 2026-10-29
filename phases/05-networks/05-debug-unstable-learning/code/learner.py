"""Debug unstable learning · learner-only exercises (3/3 contracts ready)"""

# Exercise: In the MLP lab, compare the corrupted batch, the +20 hidden-bias saturation, and the rate-100 run. Write the expected first diagnostic for each before executing it. Then repair exactly one condition in each case and verify finite or decreasing held-out loss with the fixed held-out arrays.

# Inputs:
# - A scalar linear model features, targets, initial weight, and candidate rates.

# Outputs:
# - Loss/parameter histories that distinguish stable from unstable updates.

# Invariants to check:
# - The stability limit is recomputed for the changed feature scale; reject nonfinite inputs before updating.

# Hints:
# - Differentiate the mean squared objective at the supplied data.
# - Scale the rate when scaling the objective, and check the assumption behind that relation.

import jax
import jax.numpy as jnp

def compare_learning_rates(features, targets, initial_weight, rates):
    """Loss/parameter histories that distinguish stable from unstable updates."""
    # TODO: Compare finite loss/gradient updates across rates and feature/objective scales before accepting a step.
    raise NotImplementedError




# Rescale the objective and compensate the update (Transfer):
# Multiply the squared-loss objective by five. Compare one plain gradient-descent update at the old rate and at one fifth of that rate. Explain why this compensation does not automatically describe Adam or clipping.

# Inputs:
# - Multiply the squared-loss objective by five. Compare one plain gradient-descent update at the old rate and at one fifth of that rate. Explain why this compensation does not automatically describe Adam or clipping.

# Outputs:
# - Both parameter changes and losses for the two rate/objective pairs.

# Invariants to check:
# - The compensation is evaluated for plain gradient descent at the same initial point.

# Hints:
# - Differentiate the scaled objective; plain gradient descent multiplies its gradient by the learning rate.

import jax
import jax.numpy as jnp

def practice_rescale_the_objective_and_compensate_the_update(features, targets, old_rate, scale_factor):
    """Implement the rescale the objective and compensate the update check."""
    # TODO: Compare one gradient update before and after multiplying the objective by five, using the original and one-fifth rates.
    raise NotImplementedError




# Diagnose a nonfinite input before an update (Intermediate):
# Insert one NaN feature. Show that loss/gradient finite checks fail, then repair the source data and recheck the independent initial gradient.

# Inputs:
# - Insert one NaN feature. Show that loss/gradient finite checks fail, then repair the source data and recheck the independent initial gradient.

# Outputs:
# - A rejected nonfinite batch and a finite independent initial gradient after repair.

# Invariants to check:
# - No optimizer transition runs on a batch with nonfinite features.

# Hints:
# - Changing the learning rate cannot make a NaN observation valid.

import jax
import jax.numpy as jnp

def practice_diagnose_a_nonfinite_input_before_an_update(features, targets, weight):
    """Implement the diagnose a nonfinite input before an update check."""
    # TODO: Insert one NaN feature, check input/loss/gradient finiteness before stepping, then repair the source row.
    raise NotImplementedError

