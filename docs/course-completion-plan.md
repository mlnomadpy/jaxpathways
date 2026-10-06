# Full course completion work ledger

> Current planning: [full curriculum audit — 2026-10-06](full-course-audit-2026-10-06.md) and [course expansion plan](course-roadmap-2026-10-06.md). This document preserves an earlier delivery snapshot.

> Subsequent extension: five engineering lessons and one project bring the current course to 88 lessons and 17 projects. See [engineering-operations-expansion.md](engineering-operations-expansion.md). The completed baseline and its validation counts below describe the preceding delivery.

Goal: finish the courses and missing parts, using multiple agents. This ledger preserves the full scope; passing a smaller CPU slice does not complete the goal.

## Baseline verified at goal start

The canonical manifest contains 83 lesson entries in 17 phases: 50 authored and 33 planned. There are 10 pathways, seven careers, two implemented projects, three assessment review drafts and four modality guides whose connected harnesses are not implemented. Existing uncommitted work belongs to the ongoing course upgrade and must be preserved.

## Completion requirements and evidence

1. **All 83 canonical lesson entries:** substantial teaching, staged implementation, defined mathematical notation, independent checks, different transfer practice, actionable failure diagnosis, primary references and an explained figure. Evidence: canonical source, current figure data, generated companions and fresh script/notebook execution. Hardware-specific instruction must include a real executable target path, not only a CPU analogy.
2. **Every phase integration outcome:** learner instructions, starter, reference solution, staged checks and evidence rubric, with connections to the actual lessons. Audit existing phase briefs; do not count a named artifact as an implemented project.
3. **All 10 pathway capstones and assessments:** verify route-specific capabilities, changed conditions, a failure diagnosis and explicit review criteria. A generic prerequisite project cannot stand in for a missing specialization capstone. Public reference tests and independent learner assessment remain distinct.
4. **Four modality harnesses:** image, text, audio and cross-modal systems must connect data, training, recovery, evaluation, precision, export/deployment and inference measurements. Provide CPU teaching runs plus supported target-runtime extensions. Synthetic contract fixtures and documented real-data workflows need separate honest evidence.
5. **Deployment and engineering coverage:** supported Keras/TensorFlow/JAX and other framework boundaries, weight/activation/accumulation/cache precision, calibration, edge deployment, portability/parity and target-runtime measurements. Unsupported combinations must be identified rather than silently claimed.
6. **Learner experience:** course/path/career descriptions and availability reflect real artifacts; all links, downloads, CLI/tutor workspace, math and figure explanations remain coherent. Preserve local progress and existing URLs.
7. **Verification:** full production build/check, all CPU companions, all project references, numerical assessment references, source-bound figure/output fidelity, offline bundles and GitHub Pages base. Inspect figures visually when possible. Prior denied browser access must not be bypassed; unperformed UI/hardware/expert review remains explicitly unverified.

## First authoring wave

- Science agent: science-01..04, scientific inverse-problem project, science assessment.
- Probability agent: probability-01..04, Bayesian regression project, probability assessment.
- RL agent: rl-01..04, policy project and RL assessment.
- Root: completion ledger, registration and validation integration; asynchronous recovery and remaining performance material first, then remaining systems/internals/deployment work.

All remaining canonical work stays in scope: TPU bridge; asynchronous checkpoints; XProf/HLO/rematerialization; distributed communication/recovery; science; probability; RL; Pallas/TPU kernels; advanced derivatives/compiler internals; post-training/capacity; operations. Modality harnesses and unimplemented capstones remain in scope after these lessons.

## Final status — authoring and integration complete

All seven deliverable groups above are implemented and integrated. Final verification passed after the review fixes and after the project guides were added. The remaining qualification limits concern external hardware, real-data generalization and human review; no such result is presented as measured.

| Deliverable | Final evidence |
| --- | --- |
| Canonical lessons | 83/83 authored across 17 phases; 83 scripts plus 83 notebook executions pass with current source-bound receipts |
| Teaching visuals | 78 executed plots and 5 conceptual diagrams; named axes, observations, interpretation and limitations are included |
| Phase integration | 17/17 phases reference implemented work; foundation and systems projects expose phase-specific stage mappings |
| Pathways | 10/10 capstones implemented, prerequisites resolved, unique audiences/first artifacts/study advice/evidence lists |
| Careers | 7 guides with distinct work scenarios, prerequisites, milestones, portfolio links and review questions; downloadable and CLI plans agree |
| Projects | 16/16 registered references pass; final-file hashes match every project receipt |
| Modalities | Image, text, audio and cross-modal harnesses cover data through measured inference, with genuine exports, full-state recovery and precision checks |
| Assessment | 11 synthesis review drafts, including the math extension; numerical math reference passes |
| Web/distribution | Production build, Astro, lint, formatting, 56 tests, 42 pages and 1,947 local references pass; project/workspace/book/EPUB fidelity passes |
| GitHub Pages | Base-path build and local-reference checks pass; 7 DOM interaction checks pass; normal preview build restored |

The four defects identified in internal review were fixed and independently rechecked: blank-image gradients, checkpoint configuration binding, fresh-process distributed resume, and saved dtype validation before JAX narrowing. Their cases are retained in checks. See integration-review.md.

Sixteen static project-guide pages now render the detailed teaching, KaTeX equations, real figures and section navigation directly on the website. Project and catalog pages link to them; learning no longer requires downloading a ZIP to discover the explanation. There are no remaining planned canonical lesson entries or unimplemented phase project briefs.

## Reproduce and inspect

- `npm run build` generates all course and site artifacts.
- `npm run smoke:cpu` executes every lesson script and notebook; rebuild afterward to publish fresh outputs.
- `npm run test:project` runs all registered reference suites and records file hashes in curriculum/project-validation.json.
- `npm run test:math` checks the independent numerical assessment reference.
- `npm run check` checks types, style, tests, curriculum, distribution and static site links.
- `npm run check:deployment` verifies the GitHub Pages base and DOM interactions, then restores the local build.
- `npm run audit:content` and `npm run audit:visuals` regenerate structural inventories.

Evidence is preserved in curriculum/validation.json, curriculum/project-validation.json, the project output reports and docs/*-evidence.md. Whole-scope mapping is in phase-integration-audit.md. The former full-course-audit.md is explicitly labeled historical.

## Qualification limits retained

CPU reference execution does not establish TPU/GPU/edge execution, multi-host reliability, real-device latency, natural-data model quality, independent human review or learner competency. Target-specific commands and supported conversion boundaries are included, but unexecuted target work remains labeled. The synthesis assessments remain review drafts. No paid infrastructure or deployment was performed.

Browser access had previously been denied, so live rendered UI inspection was not attempted again. Static and happy-dom checks passed; actual project figures were visually inspected separately. Human learner walkthroughs and external editorial review remain valuable follow-up work, not completed validation.
