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

## Optional Stage 4: compare NumPyro NUTS with the exact answer

This extension uses a separate project-local environment. It pins NumPyro 0.22.0 with the current course JAX/NumPy pins. It does not change `requirements-cpu.txt` or install into the shared CPU environment. From the repository root, create an isolated environment and install only the optional requirements:

~~~sh
python3 -m venv .venv-numpyro
source .venv-numpyro/bin/activate
python -m pip install -r projects/bayesian-regression/requirements-numpyro.txt
~~~

In Windows PowerShell, activate with `.venv-numpyro\Scripts\Activate.ps1`. Run the learner starter first; it is expected to stop at its TODO. Copy it to a working file and implement `regression_model` plus `run_nuts`, which are the optional checker’s required candidate API:

~~~sh
cp projects/bayesian-regression/starter/numpyro_workflow.py projects/bayesian-regression/my_numpyro_workflow.py
python3 projects/bayesian-regression/tests/check_numpyro.py --implementation projects/bayesian-regression/my_numpyro_workflow.py
~~~

For PowerShell, replace `cp` with `Copy-Item`. The optional checker runs four sequential CPU chains from the candidate `run_nuts`, validates sampled means and covariance against its own independent host NumPy precision solve using estimated Monte Carlo standard errors, then repeats with a different seed, data size and three-parameter design. It independently recomputes rank-normalized split R-hat, bulk and tail effective sample sizes, and NUTS divergences from the raw chains, and constructs paired posterior predictive checks and figure evidence itself. The starter also includes `modern_diagnostics`, `exact_posterior_numpy`, `audit_gaussian_case`, `posterior_predictive_discrepancy`, and `run_reference_workflow` study TODOs; these are ungraded optional exercises, and the acceptance checker does not call them. Do not interpret completing the required sampler API as completing those exercises or as expert approval. A small R-hat or zero divergences is evidence about the run, not proof that every posterior mode was explored.

To run the completed reference implementation and keep raw chains and a fresh evidence receipt outside the repository, use:

~~~sh
python3 projects/bayesian-regression/tests/check_numpyro.py --implementation projects/bayesian-regression/solution/numpyro_workflow.py --evidence-dir /tmp/bayesian-regression-numpyro-evidence
~~~

The committed `reference/numpyro/` assets were generated by the checker from retained raw chains. It includes `numpyro-check-evidence.json`, figure-data JSON, the raw NPZ and a receipt with source, configuration, environment and input-data SHA-256 hashes. In the saved run, the diagnostic plot's horizontal axis names the intercept and slope; its left vertical axis is rank-normalized split R-hat (unitless), with a 1.01 guide, and its right vertical axis is estimated bulk/tail ESS in effective draws. The observed R-hat values were 1.0002 and 1.0003, bulk ESS was about 1,227 and 1,455, tail ESS about 948 and 1,049, and NumPyro recorded zero divergences. ESS estimates information after autocorrelation adjustment; it is not the count of independent saved rows. The R-hat combines split-chain rank-normalized and folded rank-normalized estimates. Bulk ESS applies the same rank transform; tail ESS is the smaller ESS of lower- and upper-five-percent indicators.

The PPC plot's horizontal axis is the quadratic coefficient in residuals after subtracting each posterior draw's model mean; its vertical axis counts paired posterior predictive replications per histogram bin. Each observed and replicated discrepancy uses the same posterior draw. For the linear mean, the observed statistic was 0.870 while the replicated 90% interval was [-0.0513, 0.0510]. Adding the centered quadratic predictor is the remedy: the observed median was -0.0005, its central 90% interval [-0.0490, 0.0508] overlapped the replicated interval [-0.0523, 0.0513], and the paired tail probability was 0.508. This is a known 41-point synthetic stress case, not a real-data coverage estimate. Figure histogram counts, bin edges, and both run summaries are saved as JSON; the retained NPZ contains raw chains, divergences, and both paired discrepancy arrays.

Posterior mean MCSE uses nonoverlapping batch means of raw parameter series; covariance MCSE uses batch means of centered product series. Both the independent checker and the reference helper report those estimates and compare each NumPy posterior moment to its sampler estimate in estimated standard errors. The checker also tests diagnostic behavior on shifted chains, unequal scales, strong autocorrelation, constant inputs and nonfinite inputs. The diagnostic definitions follow the [NumPyro diagnostics API](https://num.pyro.ai/en/stable/diagnostics.html); the rank-normalized and folded R-hat construction follows Vehtari et al., [Rank-normalization, folding, and localization](https://arxiv.org/abs/1903.08008). These diagnostics describe chain behavior for the tested targets. They do not prove convergence, rule out missed modes, validate the model assumptions or establish behavior on a GPU/TPU.

The reference workflow script can also be run directly with `python3 projects/bayesian-regression/solution/numpyro_workflow.py`. Its default output goes to a directory under the system temporary path. Pass `--output` explicitly only when you want to retain that helper's report and figures; the checked reference receipt under `reference/numpyro/` is refreshed with the checker command above.

Keep a copy of the command, environment versions, changed-case diagnostics and both discrepancy intervals in `bayesian_audit.md`. Public checks do not establish expert approval, multimodal sampling reliability, accelerator behavior or real-data calibration. The optional workflow is supplementary to Stages 1–3 and remains in-development.

## Review handoff: explain a changed condition

Shift the nonconstant design column by one and compare the full posterior covariance with the original design. Why can similar means hide different prediction uncertainty?

**Keep for review:** The independent precision/solve calculation, covariance including off-diagonal terms and predictive intervals at two inputs. Diagnose a diagonal approximation separately from sampler convergence.

Write your prediction before opening the reference implementation. Distinguish what the existing checker verifies from your proposed extension; record any extension command, actual output and limitation separately. A reviewer should be able to locate the first failed contract from your saved evidence.
