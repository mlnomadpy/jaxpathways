# Course, pathway and career upgrade

> Subsequent extension: five engineering lessons and one project bring the current course to 88 lessons and 17 projects. See [engineering-operations-expansion.md](engineering-operations-expansion.md). The completed baseline and its validation counts below describe the preceding delivery.

Updated 2026-10-05. This report supersedes the earlier 50-lesson upgrade snapshot. The original full-course audit remains a labeled historical baseline.

## What learners now receive

All 83 canonical lessons across 17 phases are authored. Each has readable teaching material, runnable script and notebook companions, worked evidence, practice and figure explanation. The visualization inventory contains 78 executed plots and five conceptual diagrams; conceptual drawings are identified separately from observed outputs. The content audit checks structure and source fidelity, not independent editorial approval.

All 10 pathways describe their audience, prerequisites, first artifact, study advice, focus phases and capstone evidence. Each has an implemented capstone and a synthesis review draft. Eleven assessments include the additional math extension. Routes use shared canonical lessons and preserve existing URLs and learner progress. The TPU route now includes distributed and deployment bridge lessons required by its actual text-system capstone.

All seven career guides describe a concrete work problem, background, skills, staged milestones, portfolio evidence, review questions and limits of the course's evidence. Career details link to the portfolio project and synthesis assessment. Downloadable Markdown and the tutor CLI retain those descriptions and source paths.

## Connected project work

Sixteen registered projects supply learner scaffolds, references, cumulative checks and evidence criteria. Every phase points to an implemented integration project. The first four phases share foundation-toolkit stages 1–4; recovery, performance and distributed share sharded-training with explicit stage/prerequisite notes. Optimization retains regression-audit as the early milestone and adds optimizer-audit after the deeper twelve-lesson sequence.

Four modality guides now link to actual harness stages:

- Image: raw/local PNG input, CNN training, full recovery, evaluation and corruptions, calibrated precision, export, measured requests and rollback.
- Text: UTF-8 byte data, real causal Transformer training, complete Adam/dropout/sampler recovery, held-out token metrics, FP32/BF16/INT8 cache storage, six exported endpoints and genuine profiler traces.
- Audio: waveform/local WAV contracts, STFT features, training and full recovery, fixed evaluation, precision, exported inference and measured waveform-to-output requests.
- Cross-modal: paired image/text provenance, a multi-positive contrastive objective, both encoders trained, full recovery, retrieval failures, calibrated arithmetic, twelve exported branches and a measured release gate.

These are deliberately small synthetic CPU teaching runs with local-file ingestion contracts. They do not claim external-corpus quality, general language understanding, speech recognition, native integer acceleration or physical edge/TPU qualification. Larger MaxText work remains a separately pinned advanced extension.

## Engineering and learning boundaries

Deployment material distinguishes framework conversion from parameter mapping and requires preprocessing/layout/dtype contracts and parity checks. Precision coverage separates weights, activations, accumulation, optimizer memory and KV caches. Target runtimes and unsupported combinations remain explicit.

Plots are accompanied by explanations of the experiment, axes, observed patterns, numerical checks and limits. Project figures preserve failures: shifted retrieval errors, generation mistakes, unstable optimizer updates and a four-device CPU run that is slower than its compiled single-device baseline. Learners must explain those outcomes rather than treating a falling loss as sufficient evidence.

Source content feeds the Astro site, Markdown, notebooks, scripts, project ZIPs, offline book, EPUB, API and tutor workspace. Project bundles preserve actual exported binaries and profiler artifacts. Sixteen static project guides expose their explanations, mathematics, figures and section navigation directly on the website; every project links to its guide. KaTeX renders inline and display mathematics in lessons and project/assessment teaching content.

## Verification

The final full lesson smoke passes 83 scripts and 83 notebook executions. The math numerical reference also passes. The production pipeline checks Astro diagnostics, lint, formatting, 56 automated tests, all 42 rendered pages, local references, project/assessment prerequisites and distribution fidelity. GitHub Pages base checks additionally exercise seven DOM interaction flows and restore the normal preview build.

All sixteen project suites pass with final-file hashes verified. The GitHub Pages build and all seven DOM interaction checks pass. Results are recorded in the completion ledger and machine-readable receipts. See [course-completion-plan.md](course-completion-plan.md), [phase-integration-audit.md](phase-integration-audit.md), [content-audit.md](content-audit.md) and [visual-learning-audit.md](visual-learning-audit.md).

Live browser visual inspection remains unperformed because browser access was previously denied. Static/DOM checks do not replace rendered UI review. Actual project figures were inspected. Physical TPU, multi-host and edge performance, broad real-data generalization and independent human editorial/learner review remain separate qualification work.
