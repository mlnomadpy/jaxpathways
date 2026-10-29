# Phase 11: Probabilistic modeling

Specializations.

Specify probability models, derive exact Bayesian regression, implement and diagnose HMC, and compare variational approximations with predictive evidence.

Start with likelihoods, stable losses and explicit random keys. Check a known posterior before trusting a sampler, then separate inference error from model mismatch.

**Prerequisites:** 04: Math & optimization.

**Hardware:** CPU; transparent JAX samplers with an optional NumPyro ecosystem bridge.

## Study guide: Which uncertainty does your model actually represent?

A point estimate answers which parameters fit, but it leaves open how uncertain they are and how uncertain a future observation should be. A small solvable Bayesian regression lets you calculate those quantities separately and compare an approximate method with a known target.

Read distribution plots by their units: density height is not the probability of one value, and latent-function uncertainty does not include all observation noise. Follow the mean, covariance and predictive intervals through sampling or variational inference. Use residuals to question the assumed likelihood as well as the inference algorithm.

### Check your starting point

Should an interval for a new noisy observation equal an interval for the latent mean prediction?

<details><summary>Compare your reasoning</summary>

No. The observation includes additional likelihood noise. Under the independent Gaussian model, add its variance to the latent predictive variance before taking a square root.

</details>

Review: [Build a Bayesian regression](02-build-a-bayesian-regression/docs/en.md).

### Build in stages

1. **Build an exact reference.** Keep distribution units and normalization, derive a posterior in a small solvable case and compare latent versus observation uncertainty.

   Lessons: [Distributions and sampling](01-distributions-and-sampling/docs/en.md) · [Build a Bayesian regression](02-build-a-bayesian-regression/docs/en.md).

2. **Challenge approximate inference.** Compare sampler moments with the exact target, inspect multiple-chain diagnostics and measure the covariance lost by a diagonal approximation. Use predictive residuals to challenge the likelihood itself.

   Lessons: [Hamiltonian Monte Carlo and sampler diagnostics](03-hamiltonian-monte-carlo-and-sampler-diagnostics/docs/en.md) · [Variational inference and model checking](04-variational-inference-and-model-checking/docs/en.md).

### Try a changed condition

A correlated Gaussian target has unit marginal variances and correlation 0.8. A diagonal Gaussian variational fit gets the mean right but reports variance 0.36. Must optimization have failed?

<details><summary>Compare an approach</summary>

Not necessarily. In the reverse-KL Gaussian case the optimal diagonal variances equal inverse precision diagonals, which here are \(1-0.8^2=0.36\). The family cannot represent correlation. Compare mean, covariance and predictive intervals with the exact target; a small optimizer gradient does not certify uncertainty calibration.

</details>

**Symptom:** Intervals are narrow and diagnostics look reassuring despite poor predictions.

**Check next:** Check whether you plotted latent or observation uncertainty, and whether the likelihood or mean function misses structure.

### Decide what is ready

Use bayesian-regression. Retain analytic posterior checks, sampler/approximation diagnostics and a predictive model-mismatch case, with uncertainty assumptions stated.

### Further work

General hierarchical models and robust modern diagnostic workflows need further examples. The instructional sampler is not a replacement for validated inference libraries.

## Lesson sequence

### 11.01 Distributions and sampling

[Read the lesson](01-distributions-and-sampling/docs/en.md) · [Run the code](01-distributions-and-sampling/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Distinguish samples, density, variance and standard error Check normal moments against independent analytic expectations

**Evidence:** Seeded arrays, independent density values, moment discrepancies and underflow diagnosis. Keep the environment, observed outputs and your explanation of the figure.

**Checkpoint:** Which statement explains why a large Monte Carlo sample can have a precise mean while individual readings remain variable?

### 11.02 Build a Bayesian regression

[Read the lesson](02-build-a-bayesian-regression/docs/en.md) · [Run the code](02-build-a-bayesian-regression/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** State the prior, likelihood and observation-noise assumptions Derive posterior precision and check a closed-form special case

**Evidence:** Precision derivation, exact fractional reference, host solve and predictive variance plot. Keep the environment, observed outputs and your explanation of the figure.

**Checkpoint:** Which variance describes a new noisy reading at a given input?

### 11.03 Hamiltonian Monte Carlo and sampler diagnostics

[Read the lesson](03-hamiltonian-monte-carlo-and-sampler-diagnostics/docs/en.md) · [Run the code](03-hamiltonian-monte-carlo-and-sampler-diagnostics/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Implement leapfrog with the correct half-step boundaries Check reversibility and an independent analytic gradient

**Evidence:** Reversibility check, four traces, two-dimensional moments, key replay and classical split R-hat with limitations. Keep the environment, observed outputs and your explanation of the figure.

**Checkpoint:** Four chains have classical split R-hat close to one. What can we conclude?

### 11.04 Variational inference and model checking

[Read the lesson](04-variational-inference-and-model-checking/docs/en.md) · [Run the code](04-variational-inference-and-model-checking/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Derive the diagonal Gaussian variational optimum for a correlated target Separate optimization error from variational-family error

**Evidence:** Analytic and optimized KL, variance comparison, Monte Carlo estimate and explained residual stress fixture. Keep the environment, observed outputs and your explanation of the figure.

**Checkpoint:** The variational mean is correct and the optimization gradient is tiny. Why can posterior intervals still be too narrow?

## Phase project

A Bayesian model with diagnostic evidence.

**Demonstrate:** Report sampler diagnostics and distinguish uncertainty from predictive error.

Project status: implemented staged practice · [Open source](../../projects/bayesian-regression/README.md). Copy `projects/bayesian-regression/starter/model.py` to `projects/bayesian-regression/my_model.py` and write your code in `projects/bayesian-regression/my_model.py`. Run `python3 projects/bayesian-regression/tests/check.py --implementation projects/bayesian-regression/my_model.py --stage 1` from the top-level folder to verify all 3 stages.



[Primary documentation](https://num.pyro.ai/en/stable/).
