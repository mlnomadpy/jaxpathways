# Audit a Bayesian regression model

Build an inference audit whose numerical answer can be checked independently. Start with a conjugate Gaussian regression, use its exact posterior as a reference for Hamiltonian Monte Carlo, and distinguish uncertainty about the latent line from uncertainty about a new noisy observation.

This is a CPU teaching project with synthetic data, a known observation-noise scale and a zero-mean isotropic normal prior. It is not a general-purpose sampler or evidence of real-world calibration. The fixed HMC warmup discard does not adapt step size or a mass matrix. Classical split R-hat is implemented and labeled explicitly; it is not the modern rank-normalized diagnostic.

## Prepare your workspace

Prerequisites: the four probability lessons, JAX transformations, explicit random keys and basic linear algebra. Use the course's pinned requirements-cpu.txt. From the repository root, or the top folder of the extracted project ZIP:

~~~sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
cp projects/bayesian-regression/starter/model.py projects/bayesian-regression/my_model.py
~~~

In Windows PowerShell, activate with .venv\Scripts\Activate.ps1 and copy with Copy-Item. Use the python command selected by your activated environment if python3 is not available. The package execution receipt identifies the actually tested host; these commands do not establish clean installation on every OS.

Keep a report named bayesian_audit.md. Before each run, write your prediction. Afterward record environment, command, outputs, plot and changed-condition result. Never replace an observed failed check with an expected value.

## Stage 1: define and solve the model

Implement fit and log_joint. Your prior has independent weights with standard deviation prior_scale; the likelihood has independent observations with known standard deviation noise_scale. The design matrix includes any intercept column explicitly.

Return the exact posterior mean and covariance. Derive the precision system first; compare with a host NumPy solve using float64. Keep the log-density normalization terms, including the terms that depend on scale. Compare its JAX gradient with the independently derived gradient.

~~~sh
python3 projects/bayesian-regression/tests/check.py --stage 1 --implementation projects/bayesian-regression/my_model.py
~~~

Expected: PASS stage 1 for non-centered designs, several dimensions and scales, rank-deficient features regularized by the prior, derivatives, empty observations, and malformed input checks.

**Keep:** model assumptions, symbolic precision update, an independent reference table and a diagnosis of an accidental target-column shape. Explain why empty data recover the prior and why duplicating a data file is not new evidence.

## Stage 2: verify approximate sampling

Implement sample and diagnose. Each chain owns a split key; each transition owns separate momentum and acceptance keys. Return retained positions, acceptance flags and signed energy error. Rejected proposals must retain the old position.

Use leapfrog with half momentum steps at the boundaries, then a log-space Metropolis test. Implement classical split R-hat for each coordinate and reject zero within-chain variation as an invalid diagnostic input.

~~~sh
python3 projects/bayesian-regression/tests/check.py --stage 2 --implementation projects/bayesian-regression/my_model.py
~~~

Expected: both stage 1 and stage 2 pass. Stage 2 checks two independent seeds against a known Gaussian mean and covariance, exact replay in this environment, changed keys, deliberately stuck chains and unstable integration.

**Keep:** four labeled traces, per-chain acceptance, signed energy-error summaries, exact and empirical moments, and classical split R-hat. State that near-one R-hat does not prove convergence. Explain which modern rank-normalized, effective-sample-size and divergence diagnostics you would add before using a production sampler. The public checks do not implement those additional diagnostics.

## Stage 3: report prediction uncertainty honestly

Implement predict. Return three separately named arrays: predictive mean, latent variance and observation variance. Propagate the full covariance, including off-diagonal terms. Add noise variance only for a future observation.

~~~sh
python3 projects/bayesian-regression/tests/check.py --stage 3 --implementation projects/bayesian-regression/my_model.py
~~~

Expected: all three stages pass. The final stage compares independent quadratic forms with the implementation, simulates posterior weights and new noise, checks extrapolation, and demonstrates failure under a shifted-outcome fixture.

**Keep:** a figure of predictive standard deviation against input, with latent and observation quantities clearly distinguished. Identify axes, units, values near the data center and under extrapolation. Include coverage under the model and under the deliberate shift; the shift is a diagnostic stress test, not an empirical real-data benchmark.

## Reference and review

After your attempt:

~~~sh
python3 projects/bayesian-regression/tests/check.py --stage all --implementation solution
~~~

Reference solutions are public. Passing their tests, completing a quiz and submitting your own independently reasoned report are different kinds of evidence. Attempt the probability synthesis assessment to connect model assumptions, sampler behavior, variational approximation and posterior predictive checks.

A real-data extension should freeze data provenance, license, cleaning rules and split identifiers before fitting. Estimate unknown noise under a stated model, use an inference implementation with modern diagnostics, and report group-specific predictive checks with sample counts. Those extensions are not silently claimed by this bounded project.
