# MLflow, containerization and operations expansion

Added 2026-10-05 after the 83-lesson completion baseline. This is a new extension: the canonical course now has 88 lessons, 17 projects, 10 pathways and seven careers.

## Placement and learner outcomes

| Lesson | Phase | Practical outcome |
| --- | --- | --- |
| recovery-06 | Data and recovery | Real local MLflow tracking of two JAX runs; metric history, data identity, selection and downloaded artifact checks |
| deployment-07 | Deployment | Fresh-process artifact/request contract plus a separate real Docker qualification lab |
| operations-06 | Operations | Data schema/group leakage checks, count-weighted slice gates, CI delivery sequence and drift interpretation |
| operations-07 | Operations | Controlled text-application evaluation, critical-case failures, real MLflow traces, prompt/retrieval/tool versioning and serving measurement boundaries |
| operations-08 | Operations | Ownership, approval bound to immutable release identity, expiry, unchanged state on rejection and revalidated rollback/retirement responsibilities |

Each canonical lesson includes original explanatory sections, a worked experiment, changed-condition practice, failure diagnosis, a checkpoint and a figure connected to actual local results. Math uses the shared KaTeX path. MLflow and Docker claims are supported by actual executions rather than mocked API calls or a Dockerfile alone.

The new engineering-release project contains learner policy stubs, reference implementations, independent cumulative checks, an actual MLflow lab, a minimal inference service, pinned-base Dockerfile and restricted build context, container qualification script, CI shell example, a readable illustrated guide and compact receipts. Stage 4 connects learner policies to the trained model artifact; the Docker command remains a separate explicit dependency.

## Integration

Recovery, deployment and operations phase cards link to the additional project. All ten pathways identify relevant engineering lesson extensions; seven career guides add corresponding responsibilities and review questions. Browser and downloadable career plans preserve those extension IDs. The operations route adds the deployment bridge in prerequisite order. All four modality guides link tracking and release lessons at their training/export/operations stages, with a direct engineering-project guide link.

Operations and shipping assessments include changed-condition engineering extensions and reviewer guidance. Existing lesson IDs, capstone projects and local progress keys are preserved.

## Actual tool evidence

The installed environment provided Python 3.14.3, JAX 0.9.2 and MLflow 3.16.1; requirements-cpu.txt now pins the MLflow version. The local MLflow lab used SQLite with an explicit local artifact directory, trained two candidates, retrieved thirty metrics per run, registered two pyfunc versions, resolved/reloaded the selected immutable version, registered/reloaded one prompt version and recorded a three-span trace. The poorer candidate's validation MSE was approximately 0.11994; the selected candidate's was approximately 5.05e-10 on the fixed synthetic four-row set.

The Docker qualification executed on Docker client/engine 29.3.1 with a Linux/arm64 image. It built from the observed pinned Python base digest, mounted the actually trained artifact read-only, ran as UID 10001 with dropped capabilities and a read-only root filesystem, and used an isolated container network with no published host port. CLI predictions matched independent arithmetic. Internal HTTP readiness and predictions passed, malformed input returned 400, and a wrong model digest failed. The checker removed its own temporary serving container. The local teaching image remains available as jaxpathways-engineering:course.

See projects/engineering-release/outputs/mlflow-report.json and container-report.json. Their source hashes identify the lab/service/Dockerfile that produced the evidence. The recorded model checksum agrees between MLflow and Docker. The project figure comes from the actual tracking receipt; all five lesson plots and the project plot were visually inspected.

## Final validation

- All 88 CPU lessons passed both script and notebook execution: 176 executions.
- All 17 registered project reference suites passed; recorded source hashes match the final project files.
- Production build and `npm run check` passed: 56 tests, 43 rendered pages and 2,046 local references, plus offline chapter, EPUB, workspace and project-bundle checks.
- GitHub Pages checks passed under `/jaxpathways/`, including seven DOM checks. The normal root preview build was restored afterward.
- Content and visualization audits passed. The course has 83 executed plots and five conceptual diagrams across 88 lessons.
- The separate real Docker qualification passed with the trained MLflow artifact, as detailed above.

Machine-readable evidence is recorded in `curriculum/validation.json`, `curriculum/project-validation.json`, `docs/content-audit.json`, `docs/visual-learning-audit.json` and the engineering project's output receipts.

## Boundaries

The LLMOps experiment replays controlled answers and records real traces; it does not call or assess a live foundation model. Exact-answer/citation checks are deliberately narrow regression contracts. The existing text harness separately runs a tiny actual Transformer.

The local ModelOps reviewer field is not authenticated identity, and the single-writer file pointer is not a distributed deployment controller. The CPU policy fixture's image digest is symbolic and explicitly labeled; actual Docker release evidence uses the image identity in its own receipt. Registry alias changes are not deployment actions.

This work does not establish cloud rollout, Kubernetes execution, production concurrency or load, GPU/edge qualification, comprehensive security, or human reviewer approval. Those boundaries are explained in the teaching and assessment text. Browser access was previously denied; no live browser inspection was retried. Existing DOM/static checks cover the added links and rendering contract.

Primary sources checked: [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/quickstart/), [model registry](https://mlflow.org/docs/latest/ml/model-registry/workflow/), [prompt registry](https://mlflow.org/docs/latest/genai/prompt-registry/index.html), [tracing](https://mlflow.org/docs/latest/genai/tracing/quickstart/), [Docker build practices](https://docs.docker.com/build/building/best-practices/), [MLOps delivery](https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning), and [Kubernetes probes](https://kubernetes.io/docs/concepts/workloads/pods/probes/).
