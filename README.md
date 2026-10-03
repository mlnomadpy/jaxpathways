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

Then build **models → data and recovery → Transformers → performance**, or branch into scientific computing, probabilistic modeling, RL, distributed systems, kernels, internals, inference, or operations. Each route includes its prerequisites and ends in a project with a capability assessment.

Read the complete [curriculum](CURRICULUM.md). It has **17 phases, 72 canonical lesson briefs, 10 pathways, four core domains, and seven career routes**. These are course-design artifacts, not 72 published lessons. One sample lesson is authored; the remaining lessons, phase projects, and pathway assessments are planned.

## Use the site

```sh
npm run dev
```

Open http://localhost:4173.

- **Home:** course introduction, core domains, recommended start, and ordered curriculum.
- **Learning paths:** role responsibilities, skills, portfolio expectations, and routes through shared phases.
- **Learning notebook:** saved route, self-reported evidence, backup import/export, and draft club plans.

The homepage has no portfolio forms, club planner, duplicate learning map, or extra motivational sections. Those tools live on their own page.

Python 3 serves the site. Node runs the checks. No package dependencies or build step are needed. Use a server rather than opening the HTML directly because the course loads from JSON.

## Where notebooks fit

The earlier training, Grain/Orbax recovery, and XProf notebooks belong in phases 05, 06, and 08. They become lesson companions and phase integration labs, with the course providing prerequisites, explanation, exercises, and assessments around them.

Their source files and the previously reported 56-lesson Astro collection have not been imported. No TPU validation has been performed in this repository. Learner evidence is self-reported; the site does not issue credentials.

## Author and verify

```sh
npm run check
npm run curriculum:docs
```

`site/curriculum.json` is the single source of truth. The checks verify unique canonical lessons, acyclic prerequisites, route ordering, and domain/role coverage. The document generator produces the GitHub curriculum and phase READMEs.

See [the authoring contract](content/README.md) and [lesson template](LESSON_TEMPLATE.md).

Original layout and curriculum implementation, with design inspiration from [AI Engineering from Scratch](https://aiengineeringfromscratch.com/). Primary technical references are linked by phase.
