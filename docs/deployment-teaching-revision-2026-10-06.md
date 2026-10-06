# Deployment teaching revision — October 6, 2026

This pass reviews the teaching structure of the 97 authored lessons and repairs a connected eight-lesson phase: deployment and interoperability. Six of its eight lessons previously moved directly from explanation to a complete program. The revision gives every deployment lesson a staged build, more reasoning about failure cases, and practice that changes the conditions of the worked example. It does not mean every course has received another substantive rewrite.

## What learners now do

| Lesson | Gap addressed | New work and check |
| --- | --- | --- |
| Framework bridges | Tensor dimensions alone cannot establish semantic equivalence. | Swap feature order, repair the matching kernel rows, then locate an omitted ReLU using independent two-layer output values. |
| Model adaptation | Comparing losses does not explain how to accept a candidate. | Apply a declared source-retention budget and target-improvement gate; keep the base if neither candidate qualifies. The reused source inputs remain explicitly a retention proxy. |
| PyTorch → Flax | Aggregate error and one output sensitivity can hide a mismatch. | Work through elementwise absolute/relative budgets, inspect intermediate values, and compare input gradients under a nonuniform output cotangent. |
| Serialized export | A same-process round trip leaves the loading boundary underexplained. | Load only serialized bytes in a fresh Python interpreter and prove malformed requests are rejected before inference runs. |
| Numerical precision | Low-bit labels can obscure actual representation cost and output error. | Calculate packed versus int8-backed weight storage, include scale overhead, compare shared/per-channel grids, and reason about amplification and clipping. |
| Containerization | Matching a digest does not validate model contents or behavior. | Reject a correctly hashed invalid artifact, then verify a legitimate changed artifact against independently expected predictions. Distinguish process checks from actual image execution. |
| Serving capacity | Percentiles and average service rates can conceal deadline failures. | Explain an interpolated p95 using a twenty-sample counterexample; count missed deadlines under a simultaneous arrival burst. |
| Edge deployment | Preprocessing and model latency were insufficiently connected to the request contract. | Reject malformed raw sensor inputs before normalization and calculate whether faster inference meets an end-to-end deadline. Preserve a true first-call measurement before warm requests. |

The phase now contains 28 build steps. Four additional conceptual figures explain tolerance budgets, export boundaries, deployment evidence, and serial request latency. Their reading guides identify what each row means and distinguish illustrative numbers from executed measurements. Existing measured figures remain tied to the executable source.

## Connected learner outcome

The phase guide now asks for one reproducible release dossier: candidate selection, feature and parameter maps, layerwise parity, fresh-interpreter export, precision and calibration policy, raw-request rejection cases, artifact identity, request samples, and queue/deadline analysis. Each conclusion needs the inputs and command that produced it. Container and device evidence stays unmeasured until the learner actually runs those environments.

## Verification

- All eight revised generated Python companions passed their assertions in isolated CPU processes during targeted verification.
- The full CPU run passed all 97 scripts and 97 notebooks: 194 executions. Every generated lesson now has execution and figure records matching its current content hash.
- Regenerated all eight deployment figures and inspected them alongside the four new conceptual figures. Strict KaTeX rendering passed for 95 mathematical fields across the phase.
- `npm run build` and `npm run check` passed: 85 tests, 166 static pages, 7,739 local references, 97 offline chapters and pre-rendered lessons, EPUB integrity, and 161 canonical sitemap URLs.
- Content and visual audits passed. Their structural counts do not establish learner comprehension or independent technical review.
- Canonical lesson JSON and the phase manifest are the authoring sources; Markdown, Python companions, notebooks, browser data and books are regenerated from them.

The local version is **0.3.0**. Its source snapshot identifies exactly these eight lessons and the deployment phase as changed; other lesson revision labels remain intact. This authoring pass has not been published. Figure inspection and DOM checks do not substitute for a fresh visual browser review.

## Remaining gaps

These lessons still use deliberately small fixtures. They do not establish natural-data model quality, production load behavior, packed low-bit kernel speed, Docker execution in this revision, or target-device/TPU qualification. Optional real-framework and container labs retain their separate execution requirements.

Further teaching-depth work should connect performance diagnosis, operational monitoring and durable recovery to complete project runs. In particular, several performance and operations lessons still have only one transfer exercise, while recovery tooling needs more staged construction. These are priorities for substantive follow-up, not grounds to label otherwise runnable lessons complete or learner-validated.
