# Deployment completion authoring evidence

## Delivered sources

- phases/15-deployment/02-adapt-a-pretrained-model-and-choose-a-post-training-objective/lesson.json
- phases/15-deployment/04-inference-capacity-batching-and-autoscaling/lesson.json
- projects/deployment-audit: manifest, README, starter, reference and independent three-stage checks
- assessments/ship.md and assessments/ship-reviewer.md

Shared curriculum, route and assessment registries were not edited by this authoring agent.

Both lessons contain three cumulative build steps, six explanatory sections, two prediction experiments, two distinct transfer practices, a separate main exercise, diagnosis, formative checkpoint, KaTeX math, primary references and executed figures. The cumulative code equals the complete example exactly.

## Actual execution

Environment: Python 3.14.3, JAX 0.9.2, NumPy 2.4.4, one local CPU device.

Each lesson was executed in a fresh Python process including complete example, main solution, both experiments, both practice solutions and figure code. Both exited successfully. The figure worker then executed both complete examples separately and saved source-bound JSON, SVG and PNG files. Both figures were visually inspected. Installed KaTeX rendered every inline and display expression with throwOnError enabled.

The project reference passed:

```sh
python3 projects/deployment-audit/tests/check.py --implementation solution --stage all
```

The unimplemented starter intentionally failed stage 1 with a clear NotImplementedError.

### Adaptation results

The source model is actually optimized, saved as a NumPy checkpoint, hashed and reloaded before adaptation. It is a three-parameter logistic synthetic fixture, explicitly not a downloaded foundation model.

| Model                 | Held-out target cross-entropy | Held-out probability squared error | Source cross-entropy |
| --------------------- | ----------------------------: | ---------------------------------: | -------------------: |
| Pretrained source     |                     0.9780183 |                          0.1710871 |            0.5451401 |
| Supervised adaptation |                     0.5554870 |                         0.01209394 |            0.7904531 |
| Teacher adaptation    |                     0.5240296 |                        0.000229821 |            1.0046310 |

All losses are evaluated on the stated common criterion; training losses from different objectives are not directly ranked. The teacher equals the known target generator, a declared favorable assumption. Flipping its probabilities raises held-out target loss to 1.3151594. Source retention gets worse after both adaptations; the figure explains this tradeoff.

Source array/target SHA-256 from the executed fixture: f5fb50c38195b46766bee2edaa81a7ad28386523baa850c5294eb71b47f1f84c.

### Capacity results and measurement boundaries

Deployment-04 measured actual inference for batch shapes 1, 8 and 32, with 40 completed warm calls each. Its timer excludes input placement, JSON, networking and queueing. The saved figure receipt carries the particular timing samples; timings changed across runs, so prose does not freeze them into a claim.

The independent queue example with arrivals [0, 1, 4] milliseconds and service duration 2 milliseconds gave response times [2, 3, 2]. Overloaded arrivals spaced at 0.6 service durations gave a final response of 24.6 service durations across 60 requests. This is a declared simulation, not a load test. The replica estimate and batch-collection arithmetic are likewise labeled hypothetical.

### Project checks

- Stable objective gradient compared with independent float64 host algebra.
- Actual pretraining, checkpoint save/reload, unchanged saved base and changed-task adaptation.
- INT4/INT8 weight rounding bounds and zero-column behavior.
- Six real serialized JAX export artifacts: three precision policies at batch one/eight.
- Independent FP32, dequantized weight-only and integer-accumulation arithmetic comparisons on changed inputs.
- Calibration/source/adaptation/base provenance and artifact corruption rejection.
- Validated JSON request parity, malformed/nonfinite request rejection and shifted activation clipping.
- Actual complete in-process request timings include JSON decode, validation, transfer, deserialized computation, completion and JSON encode.
- Independent queue hand timelines and burst behavior.

One final reference run measured request p50/p95 milliseconds of approximately 0.1104/0.1251 at batch one and 0.1092/0.1121 at batch eight. These are local sample summaries, not performance guarantees. The public test uses 12 timing observations to keep checks quick; learner reporting asks for at least 30.

No HTTP/network server, real autoscaler, production concurrency, external pretrained model, preference optimization, RL, edge device or specialized low-bit accelerator execution was claimed or run.

## Integration metadata for root

Keep the existing lesson paths and canonical sequence. Set deployment-02 and deployment-04 status to authored.

| ID            | Minutes | Objective                                                                                                                          | Exercise                                                                                                        | Evidence                                                                                                              | Check                                                                                   |
| ------------- | ------: | ---------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| deployment-02 |     125 | Adapt a genuinely pretrained checkpoint with supervised and teacher objectives and compare held-out quality plus source retention. | Verify zero-step checkpoint parity, then show that adaptation changes new parameters without mutating the base. | Source/checkpoint hashes, independent loss and gradient, disjoint evaluation results and teacher-failure diagnosis.   | Explain why different training targets require a common held-out evaluation criterion.  |
| deployment-04 |     115 | Measure completed batched inference and derive a capacity hypothesis with explicit arrival and service units.                      | Derive and simulate a simultaneous four-request burst with two-millisecond serial service.                      | Warm timing samples, latency/throughput units, independent queue timeline and labeled simulation/replica assumptions. | Distinguish batch execution latency from amortized example cost and compute throughput. |

Add projects/deployment-audit/project.json to curriculum/projects.json and deployment-audit to course.projectIds. Set the deployment phase projectId to deployment-audit. Suggested phase wording:

```json
{
  "description": "Adapt a trained model, verify framework and export contracts, compare precision policies and measure local inference before planning server or edge capacity.",
  "learningAdvice": "Preserve checkpoint, data and calibration provenance. Compare the same held-out inputs before and after adaptation or conversion, and keep local request timings separate from queue simulation and target-device evidence."
}
```

The ship capstone can be marked authored and linked to deployment-audit when its title/outcome clearly states the delivered scope: a verified inference artifact, measured local request boundary and capacity hypothesis. Do not label the current project as an executed production or network deployment.

Assessment registry entry:

```json
{
  "id": "ship",
  "projectId": "deployment-audit",
  "title": "Shipping synthesis",
  "status": "review-draft",
  "scope": "project-synthesis",
  "source": "assessments/ship.md",
  "url": "assessments/ship.html",
  "pathwayId": "ship",
  "phaseId": "deployment"
}
```

Include ship-reviewer.md with the assessment bundle.

## Primary sources checked

- [JAX export and serialization](https://docs.jax.dev/en/latest/export/export.html)
- [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html)
- [Original distillation paper](https://arxiv.org/abs/1503.02531)

Current installed execution verifies the APIs actually used. The external-model adapter procedure is explicitly an unexecuted extension.

## Integration work still required

Root must register the sources and project, regenerate course companions, run whole-corpus scripts/notebooks and application checks, and update coverage documentation. Local authoring checks do not replace generated-companion validation, human editorial review or target deployment evidence.
