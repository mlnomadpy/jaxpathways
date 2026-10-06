# Probability phase authoring evidence

## Delivered canonical sources

Four connected lesson sources now exist under phases/11-probability:

| Lesson         | Minutes | Teaching artifact                                                                                                 |
| -------------- | ------: | ----------------------------------------------------------------------------------------------------------------- |
| probability-01 |      90 | Explicit-key normal simulation, independent moment checks, stable mixture log-density                             |
| probability-02 |     110 | Conjugate Gaussian regression, exact fractional oracle, latent versus observation variance                        |
| probability-03 |     140 | Transparent fixed-trajectory HMC, leapfrog reversibility, four-chain diagnostics and unstable integration failure |
| probability-04 |     125 | Exact Gaussian KL optimization, independently derived diagonal optimum, Monte Carlo KL and predictive mismatch    |

Each has three cumulative build steps whose concatenation equals the complete example, at least four mechanism sections, two prediction experiments, two distinct transfer tasks, a separate main exercise, diagnosis, a formative question, primary references and an executed figure. Mathematical prose uses KaTeX delimiters.

The image figures were generated with the repository figure worker and visually inspected. Their source-bound JSON, SVG and PNG outputs are inside the corresponding lesson folders. Plot explanations identify axes, scale, numerical values, interpretation and limits. The first figure uses an explicitly labeled analytic standard-error formula; it does not claim measured empirical coverage.

## Execution performed by the authoring agent

Environment: Python 3.14.3, JAX 0.9.2, NumPy 2.4.4, one CPU device. No GPU/TPU, NumPyro, NUTS or external dataset was executed.

For each lesson a fresh Python process ran, in order: complete example, main reference solution, both experiment blocks, both practice solutions and figure code. All four exited successfully. The figure worker separately ran all four complete examples and plotted their actual visual_data.

Selected actual outputs:

- probability-01: sample mean 2.0067408, variance 9.0352821; stable equal-mixture log density -1000.37994.
- probability-02: mean [0.923077, 1.7777778], diagonal covariance [0.30769232, 0.44444445]; latent variance at input two 2.0854702.
- probability-03: pooled mean [0.98358583, -1.0185838], covariance [[0.98042357, 0.7727629], [0.7727629, 0.9861013]]; classical split R-hat [1.0002148, 1.000253]. Unstable step size 1.2 gave acceptance 0 and maximum positive energy error approximately 5.02e10.
- probability-04: diagonal variances [0.36000016, 0.36000016], exact KL 0.51082563, Monte Carlo KL 0.51191533; modeled self-coverage 0.9494; deterministic curved-fixture coverage 2/9.
- All inline and display lesson math rendered through installed KaTeX with throwOnError enabled.
- Bayesian regression reference project passed all three cumulative stages, with two-seed moment and covariance checks, replay, invalid-input tests, independent host references and shifted-outcome diagnosis.
- Unimplemented project starter intentionally failed stage 1 with its clear NotImplementedError.

Exact project verification command:

```sh
python3 projects/bayesian-regression/tests/check.py --implementation solution --stage all
```

Project outputs included seed 19 mean [0.6829027, -0.4082704] and classical split R-hat [1.0002191, 1.0000464], and seed 43 mean [0.7016259, -0.4002317] with R-hat [1.000242, 0.9997227], for target mean [0.7, -0.4]. Classical R-hat estimates can fall slightly below one.

The root agent must still register sources, regenerate companions and run the whole-corpus script/notebook smoke and application checks. These local checks do not substitute for that integration verification.

## Exact integration metadata

Preserve existing phase paths and sequential prerequisites. Change the four probability lesson statuses to authored. Suggested manifest wording:

| ID             | Objective                                                                                             | Exercise                                                                                 | Evidence                                                                                                          | Check                                                                          |
| -------------- | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| probability-01 | Check distributions, moments and stable log-density arithmetic with explicit random keys.             | Change normal location and scale; compare moments with an analytic standard-error bound. | Seeded arrays, independent density values, moment discrepancies and underflow diagnosis.                          | Distinguish observation variance from the standard error of an average.        |
| probability-02 | Derive a conjugate Bayesian regression posterior and distinguish latent from observation uncertainty. | Strengthen the prior and compare posterior weight norm and marginal variances.           | Precision derivation, exact fractional reference, host solve and predictive variance plot.                        | Explain why a future observation interval includes observation-noise variance. |
| probability-03 | Implement and diagnose fixed-trajectory HMC against a known correlated posterior.                     | Increase step size and explain acceptance collapse through energy errors.                | Reversibility check, four traces, two-dimensional moments, key replay and classical split R-hat with limitations. | Explain what agreement among chain variances can and cannot establish.         |
| probability-04 | Separate variational family error, optimization error and predictive model mismatch.                  | Remove target correlation and verify when the diagonal family becomes exact.             | Analytic and optimized KL, variance comparison, Monte Carlo estimate and explained residual stress fixture.       | Explain why a correct variational mean does not guarantee correct uncertainty. |

Phase metadata:

```json
{
  "projectId": "bayesian-regression",
  "description": "Specify a probabilistic model, derive an exact posterior, audit approximate inference and inspect what its predictions miss.",
  "learningAdvice": "Use the exact Gaussian answer as an oracle before trusting a sampler or variational approximation. Keep inference diagnostics separate from predictive checks and distinguish known-noise synthetic evidence from external-data validation."
}
```

Append projects/bayesian-regression/project.json to curriculum/projects.json and bayesian-regression to course.projectIds. The project has an authored manifest, starter, solution, tests and README. Point the probability pathway capstone to this project and set its status authored, with a scope limited to the implemented Gaussian regression/inference audit.

Assessment registry entry:

```json
{
  "id": "probability",
  "projectId": "bayesian-regression",
  "title": "Probability synthesis",
  "status": "review-draft",
  "scope": "project-synthesis",
  "source": "assessments/probability.md",
  "url": "assessments/probability.html",
  "pathwayId": "probability",
  "phaseId": "probability"
}
```

Reviewer notes: assessments/probability-reviewer.md. Ensure packaging carries these notes alongside the assessment.

## Primary documentation checked

- [JAX random keys](https://docs.jax.dev/en/latest/random-numbers.html)
- [JAX logsumexp](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.special.logsumexp.html)
- [Stan posterior analysis](https://mc-stan.org/docs/reference-manual/analysis.html)

These informed API and diagnostic language. Installed JAX execution supplies the concrete API checks. NumPyro references are an explicitly unexecuted ecosystem bridge.

## Remaining boundaries

Human editorial review, novice learner validation, modern rank-normalized diagnostics/ESS in a production library, unknown observation-noise inference and a licensed real-data evaluation remain distinct extensions. Nothing here claims those were run. The bounded phase and project are authored and locally executed; whole-course registration and generated companion evidence belong to root integration.
