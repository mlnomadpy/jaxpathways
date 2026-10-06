# JAX Pathways

Learn JAX through a shared course, then follow the work you want to do.

## Choose your work

| Core domain | What you become capable of doing | Career routes |
| --- | --- | --- |
| Training & model development | Build, evaluate, checkpoint, and reproduce neural-network training | ML/training engineer; RL engineer |
| Inference & deployment | Adapt, export, verify, and plan serving capacity for trained models | Inference/deployment engineer |
| Operations & accelerator systems | Observe workloads, recover failures, diagnose bottlenecks, and scale computations | Operations engineer; accelerator/performance engineer |
| Research & scientific computing | Differentiate simulations, fit parameters, model uncertainty, and validate algorithms | Scientific ML engineer; research engineer |

## One course, several routes

**Setup → arrays and functions → transformations → state and loops → optimization.**

Then build **models → data and recovery → Transformers → performance**, or branch into scientific computing, probabilistic modeling, RL, distributed systems, kernels, internals, inference, or operations. Each route maps its prerequisites, implemented final project and synthesis assessment review draft. Nineteen staged projects connect every phase, including four complete CPU modality harnesses.

Read the [curriculum](CURRICULUM.md): **19 phases, 97 lesson entries, 10 pathways, four domains and seven career routes**. All 97 entries now have authored teaching, runnable CPU reference companions, exercises and explained figures. The [whole-content audit](docs/content-audit.md) and source-bound execution receipts distinguish authored material, numerical checks and review status.

Two connected phases now cover [masked and contrastive pretraining](phases/17-pretraining/README.md) and [SFT, LoRA, reward models, RLHF mechanics and DPO](phases/18-posttraining/README.md). The [weight-conversion project](projects/weight-conversion/README.md) executes actual PyTorch-to-Flax mapping with layerwise errors, gradient checks and fresh-process artifact inference. See [scope and validation](docs/pretraining-posttraining-expansion.md).

The [2026-10-06 full curriculum audit](docs/full-course-audit-2026-10-06.md) distinguishes current coverage from the remaining gaps. The [next-course plan](docs/course-roadmap-2026-10-06.md) scopes 61 candidate lesson units across 16 packages, beginning with a 14-unit connected text lifecycle. These are planned extensions, not additional authored lessons.

The [teaching-depth review](docs/teaching-depth-audit-2026-10-06.md) distinguishes lesson availability from instructional depth and records the rewrite of the eight pretraining/post-training lessons, with the remaining integration gaps.

The [model lifecycle capstone catalog](docs/model-lifecycle-capstones.md) specifies ten planned projects: 300M-scale text embeddings and causal language models, image/audio understanding, image/audio generation, grounded multimodal generation, CLIP, SigLIP, and retrieval systems. Each connects data recipes, model-specific training phases, benchmarks, monitoring and Hugging Face release. The [tooling plan](docs/model-lifecycle-tooling.md) adds Weights & Biases alongside MLflow and focused evaluation/data/serving tools. These specifications are available; training and release implementation remain planned.

Each phase now includes a study guide with a starting diagnostic, linked build milestones, a changed-condition task and specific project evidence. Read the [phase quality audit](docs/phase-quality-audit.md) for the improvements and remaining content priorities.

Pathways explain the intended learner, first artifact, study sequence and evidence to collect. Engineering extensions now connect MLflow tracking, Docker services, MLOps data/release gates, LLMOps prompt/evaluation/tracing and ModelOps ownership/approval; see the [engineering release project](projects/engineering-release/README.md). Career plans add realistic work scenarios, milestones and role-specific review questions. Available project work includes regression, image classification, scientific inverse problems, Bayesian inference, policy learning, derivative/compiler audits, deployment, operations and Pallas kernels. The project catalog and modality guides identify each connected harness and its remaining extensions; use their current availability rather than inferring completion from a title.

CPU execution and synthetic-data checks do not establish GPU/TPU, multi-host, mobile-device or real-data quality. Guarded target execution paths, framework/precision boundaries and unsupported combinations are documented. Synthesis assessments remain public review drafts; human editorial review, learner walkthroughs and target-device qualification are separate work. The [completion ledger](docs/course-completion-plan.md) records the finished authoring/integration checks and remaining qualification limits.

The course includes five neural-network training lessons (MLP, Flax NNX, compiled steps, CNNs and instability), six data/recovery lessons using Grain, Orbax and MLflow, four Transformer lessons (attention through a tiny training/checkpoint/generation system), and five performance lessons. The synthetic datasets and small models are teaching fixtures, not evidence of production capability or generalization.

Mathematical explanations use locally bundled KaTeX for inline notation and equations. The [learner writing guide](docs/learner-writing-and-math.md) describes the intuition-first approach and how math carries into notebooks and offline books.

## Practice without an accelerator

[Create four logical CPU devices](phases/00-welcome/03-virtual-cpu-devices/docs/en.md), then practise [meshes and sharding](phases/09-distributed/01-arrays-meshes-and-sharding/docs/en.md) and [a sharded training step](phases/09-distributed/02-a-sharded-training-step/docs/en.md). The scripts configure CPU devices before backend initialization. For notebooks, restart the kernel and Run all. Four logical devices share one physical host; they do not emulate TPU speed, memory or network behavior.

The smoke runner checks the actual backend and device count in both fresh script and notebook processes. Receipts record four CPU devices for the logical-device lessons and single-device CPU execution elsewhere. Tested package versions are pinned in requirements-cpu.txt; no Colab account or cloud spending is required for these exercises.

## Develop the website

Use Node.js 22.12+ (Node 24 in CI), npm and Python 3. Python's standard library builds the course downloads; JAX is needed only for execution checks.

```sh
npm ci
npm run dev
```

Open http://127.0.0.1:4173/index.html. Astro serves the site and reloads component/style changes. After editing curriculum content, run `npm run content:build` to refresh generated data and downloads.

```sh
npm run build            # Generate course artifacts, then static Astro HTML in dist/
npm run preview          # Preview the production build
npm run check            # Astro, lint, formatting, content, behavior and output checks
npm run check:deployment # Build/test /jaxpathways/, then restore the local root build
```

Home, Course, Learning paths, Careers, My learning, Resources, the group planner, project workspaces, nineteen illustrated project guides, assessments, and the printable reader are Astro routes. Existing `.html` URLs and browser storage keys are preserved. Learning records remain local to the same browser origin; changing hostname or port does not transfer them automatically. Export a backup from My learning when moving between origins.

See [the architecture and authoring guide](docs/astro-migration.md) for content locations, reusable components, browser modules, styles and GitHub Pages setup.

## Where notebooks fit

The earlier training, Grain/Orbax recovery, and XProf notebooks belong in phases 05, 06, and 08. They become lesson companions and phase integration labs, with the course providing prerequisites, explanation, exercises, and assessments around them.

Their source files and the previously reported 56-lesson Astro collection have not been imported. No TPU validation has been performed in this repository. Learner evidence is self-reported; the site does not issue credentials.

## Author and verify

```sh
npm run build
npm run check
```

`curriculum/course.json` owns phase order and lesson membership. `learning-paths/*.json` owns the routes. Each authored lesson has its own `lesson.json` under `phases/`. `npm run build` combines them into the website manifest, readable sources, companions, and public course resources. Checks verify prerequisites, role coverage, artifact consistency, and the CLI contracts.

Run `npm run audit:content` to regenerate the lesson-by-lesson availability and teaching-component audit. Read [the editorial findings](docs/content-quality.md); passing structure checks is not editorial approval.

See [the authoring contract](content/README.md) and [lesson template](LESSON_TEMPLATE.md).

Original layout and curriculum implementation, with design inspiration from [AI Engineering from Scratch](https://aiengineeringfromscratch.com/). Primary technical references are linked by phase.

## Run the foundation exercises

Use a virtual environment and install `requirements-cpu.txt`, then run `npm run smoke:cpu`. The runner executes each authored script and the code cells of its notebook in fresh CPU processes, with a 60-second limit per execution. It writes `curriculum/validation.json`; the next content build exports `public/validation.json` with the tested Python, JAX, and NumPy versions. This verifies worked examples and reference solutions; it does not validate TPU execution or grade learner work.

Each authored lesson has a phase-based bundle with `docs/en.md`, `code/main.py`, `notebooks/exercise.ipynb`, `quiz.json`, and `outputs/evidence.md`. The full-page browser reader supports a phase menu, an on-page outline, direct lesson links, previous/next navigation, downloadable companions, and separate checkpoint/exercise progress. The older `lessons/` READMEs remain generated compatibility copies. Export/import in **My learning notebook** includes lesson progress and evidence. Progress is stored locally and is self-reported.

Edit the lesson’s `lesson.json`, then run `npm run build`, `npm run check`, and `npm run smoke:cpu`. `public/curriculum.json`, `public/api/v1/`, phase/lesson Markdown, scripts, notebooks, quizzes, and evidence templates are generated; do not edit them directly.

## Repository layout

```text
curriculum/course.json            phase order, prerequisites, lesson membership
learning-paths/<pathway>.json      ordered routes through shared phases
phases/<phase>/<lesson>/
  lesson.json                     authored lesson content
  docs/en.md                      generated narrative
  code/main.py                    runnable example and reference solution
  notebooks/exercise.ipynb         notebook companion
  quiz.json                       formative checkpoint
  outputs/evidence.md              learner evidence template
src/data/                        editable site copy and presentation selections
src/pages/                       Astro routes
src/components/                  reusable semantic UI
src/layouts/                     shared document shell
src/lib/                         pure domain logic, content access and state contracts
src/scripts/                     browser controllers and page entry points
src/styles/                      design tokens and feature styles
public/                          static assets and generated data/downloads
dist/                            generated static deployment output
curriculum/openapi.json           authored API resource contract
scripts/course.py                 terminal course tools
```

## Use the course in a terminal or agent

```bash
python3 scripts/course.py list --available
python3 scripts/course.py list --pathway ship
python3 scripts/course.py show first-gradient
python3 scripts/course.py run first-gradient
python3 scripts/course.py quiz first-gradient
python3 scripts/course.py plan inference-engineer > LEARNING.md
```

Run commands from the repository root. `run` executes a known authored lesson on CPU with a 60-second limit; it rejects planned lessons. Plans preserve prerequisites and explicitly label pending content.

The website exposes static JSON catalogs for lessons, pathways, and roles. See [the resources page source](src/pages/developer.astro), [OpenAPI](curriculum/openapi.json), and [the reference adaptation notes](docs/reference-adaptation.md). These resources support discovery and reading; they do not offer hosted code execution, learner accounts, or a progress-writing API.

## Finish the foundation route

The [regression audit project](projects/regression-audit/README.md) connects arrays, gradients, compiled loops, and optimization through four stages. Public checks cover shape contracts, independently checked gradients, parameter recovery, replay, and held-out results. Run `npm run test:project` to validate all nineteen registered project reference suites. Passing those synthetic checks is bounded evidence, not a reviewed career assessment.

## Read offline or learn with a tutor

`npm run build` creates the printable HTML reader in `dist/downloads/` through Astro. Python generates EPUB, project ZIPs and tutor bundles in `public/downloads/`, which Astro copies into `dist/downloads/`. Find them on the Resources page. Books contain all 97 authored lessons. Project guides are readable on the website and included with their workspaces; assessments remain review drafts.

The landing page leads with agent tutoring. Download `public/downloads/jax-course-workspace.zip`, extract it, open its `jaxpathways` folder in your agent and run the local `npx` command below. This portable workspace includes the current canonical lessons, projects and skills; it is not a Git checkout or website build environment. Ask “Use start-learning to help me learn JAX from this workspace” to begin.

To add the optional tutors from your local checkout with Node.js/npm:

```sh
npx skills add . --skill start-learning learn-jax jax-course-guide check-jax
```

For a GitHub revision containing those skills, the equivalent command is:

```sh
npx skills add mlnomadpy/jaxpathways --skill start-learning learn-jax jax-course-guide check-jax
```

The remote command needs repository access, including configured credentials for a private repository, and does not include unpublished local changes. Its remote installation has not been verified here. Use the local checkout command for the current working source. Installation is optional and has not been performed globally.

In Codex, choose `start-learning` or type `$start-learning`; in Claude Code use `/start-learning`. The tutor helps choose a route and keeps a learner-owned `LEARNING.md`; use `learn-jax` to resume, `jax-course-guide` to find a topic and `check-jax` for formative practice. Read [installation and host instructions](skills/README.md). Skills require the course checkout and preserve the distinction between quizzes, reference execution and your own evidence.

The [course-authoring skill](skills/create-jax-course/SKILL.md) guides contributors through canonical sources, teaching steps, staged projects and execution validation.

## Content depth

The four array lessons and five transformation lessons now include worked reasoning, prediction-led experiments, foundation/practice/challenge exercises, diagnosis, and specific evidence requirements. Authored counts describe availability, not editorial quality. All 97 lessons now have teaching sources; the remaining priorities are realistic data, connected model lifecycles and independent learner transfer. See the [current curriculum audit](docs/full-course-audit-2026-10-06.md) and [content quality standards](docs/content-quality.md).

## Continue into neural networks

The [classifier project](projects/mlp-classifier/README.md) connects the neural-network lessons with shape contracts, independent objective/gradient checks, untouched evaluation data, weighted metric aggregation and replay. Run `python3 projects/mlp-classifier/tests/check.py --implementation solution --stage 3` to verify the reference. The starter deliberately fails until you implement it. Public synthetic tests are formative practice, not a reviewed pathway exam.


## Connect lessons into a model system

Start from [the four modality guides](curriculum/modality-tracks.json), also available from Projects and Learning paths. Each guide gives eight stages, modality-specific failure checks, evidence requirements and links to real preparation lessons. Downloadable Markdown travels with the tutor workspace.

```sh
python3 scripts/course.py guide
python3 scripts/course.py guide image
```

The ten pathways distinguish focus-phase coverage from shared preparation. Career routes describe a work scenario, portfolio evidence and review questions. The [math synthesis draft](assessments/math.md) assesses geometry, regularization, minibatch noise and optimizer state; its [numerical reference](assessments/math-reference.py) can be run separately on CPU. See [the upgrade record](docs/pathways-course-upgrade.md) for implemented changes and remaining gaps.

## Project organization guide

The [tools and project workflow guide](content/guides/project-workflow.md) teaches source/config/data separation, reproducible experiment records, CLI design, and when to add tracking or infrastructure tools. The [practice workspace](resources/project-workspace/README.md) runs with Python’s standard library. `npm run build` publishes the guide and downloadable ZIP at `project-workflow.html`.
