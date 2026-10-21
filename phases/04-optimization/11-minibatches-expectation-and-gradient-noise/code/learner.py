"""Minibatches, expectation, and gradient noise · learner-only exercises (3/3 contracts ready)"""

# Exercise: Enumerate all \(64\) ordered batches of size \(3\). Check their mean and variance against the independent-sampling formula.

# Inputs:
# - A fixed scalar parameter, finite dataset, and ordered minibatch size.

# Outputs:
# - The exact expected minibatch gradient and its variance.

# Invariants to check:
# - The sampler distribution is explicit; weighting unequal microbatches uses example counts.

# Hints:
# - Enumerate the small batch space before relying on a closed-form expectation.
# - Keep the parameter fixed across microbatch gradient accumulation.

import jax
import jax.numpy as jnp

def batch_gradient_statistics(parameter, dataset, batch_size):
    """The exact expected minibatch gradient and its variance."""
    # TODO: Enumerate the small sampler and compute expected gradient and variance at a fixed parameter.
    raise NotImplementedError




# Accumulate before updating (Transfer / diagnosis):
# At the same fixed \(w\), accumulate gradients from microbatches of sizes \(3\) and \(1\). Verify that one update matches the full-batch update.

# Inputs:
# - At the same fixed \(w\), accumulate gradients from microbatches of sizes \(3\) and \(1\). Verify that one update matches the full-batch update.

# Outputs:
# - One accumulated update and a full-batch comparison.

# Invariants to check:
# - The parameter remains fixed until all microbatch contributions are accumulated.

# Hints:
# - Do not update parameters between microbatches. Weight each mean by its sample count.

import jax
import jax.numpy as jnp

def practice_accumulate_before_updating(parameter, microbatches):
    """Implement the accumulate before updating check."""
    # TODO: Differentiate both microbatches at one fixed parameter, sum gradients by example count, then perform one update.
    raise NotImplementedError




# Detect a biased sampler (Transfer / diagnosis):
# A sampler chooses targets \(1\) and \(4\) with probabilities \(0.9\) and \(0.1\), never the others. Compute its expected gradient. Does it estimate the uniform dataset gradient?

# Inputs:
# - A sampler chooses targets \(1\) and \(4\) with probabilities \(0.9\) and \(0.1\), never the others. Compute its expected gradient. Does it estimate the uniform dataset gradient?

# Outputs:
# - The biased expected gradient and the uniform reference gradient.

# Invariants to check:
# - The expectation uses the stated sampling probabilities, including zero probability for omitted targets.

# Hints:
# - Use the sampling probabilities as weights. Missing examples cannot be recovered by naive averaging.

import jax
import jax.numpy as jnp

def practice_detect_a_biased_sampler(targets, probabilities):
    """Implement the detect a biased sampler check."""
    # TODO: Enumerate the sampler-weighted gradient and compare it to the uniform four-target dataset gradient.
    raise NotImplementedError

