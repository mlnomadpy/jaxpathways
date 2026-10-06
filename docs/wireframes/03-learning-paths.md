# 03 — Learning paths

Route: `pathways.html`. Goal: choose recognizable work, without searching roles or completing a quiz.

## Desktop

```text
What would you like to build?
Start with shared foundations. Choose a direction when you are ready.
[Start with JAX — open foundations path]

TRAIN MODELS
[Regression audit]                    [Classifier]
Fit, check, and reproduce a model.     Build and defend a training loop.
Project available · lessons are drafts
[Open path]                           [Open path]

DEPLOY MODELS
[Inference and deployment]
Inspect the available prerequisites and planned serving work.
[Open path]    Partly available

SCIENTIFIC WORK
[Scientific computing] [Probability] [Reinforcement learning]
Actual availability on each card; project claims drawn from manifests.

SYSTEMS WORK
[Distributed computing] [Performance / TPU] [Operations] [JAX internals]

Explore careers connected to these skills (always visible)
  Seven role links grouped under the same four work domains
```

Map the ten existing path manifests to these groups; foundations is featured first, not duplicated as a second route. Model training can feature its classifier project while retaining recovery and Transformer milestones in the full path. Regression belongs to foundations. Do not invent an additional independent classifier path.

Cards show outcome, prerequisite summary, runtime requirements, and concise availability. Limit each to one link. Detailed libraries, role responsibilities, and milestone lists live in path detail.

## Career routes

```text
Training: [ML & training engineer] [RL engineer]
Deployment: [Inference & deployment engineer]
Systems: [Operations engineer] [Accelerator & performance engineer]
Science: [Scientific ML engineer] [Research engineer]
```

Role links open a detail view using the path-detail template, with daily work and ordered milestones. Roles and paths remain distinct data objects, even though the UI groups them together. A role may traverse several paths. Preserve existing role deep links.

## Mobile

```text
What would you like to build?
[Start with JAX]

Training models
[Regression audit / Foundations]
[Classifier / Model training]
Deployment
[Inference and deployment · Partly available]
Science
(cards stacked)
Systems
(cards stacked)
Related careers (always visible)
```

Use section anchor links only if the page becomes long; no category carousel or mandatory filter bar. Each card remains readable without hover.

## States and checks

All categories stay visible; no empty filter state. Planned cards open honest roadmap detail and show available prerequisites. Coming from a role deep link opens that role directly with Back to all paths. Success: a learner can identify a build outcome without knowing a job title or JAX library name.

Career discovery: the main navigation links directly to `careers.html`, where all seven role cards appear immediately. Learning paths retains a visible career section and legacy role URLs. Shared route details and saved plans are unchanged.
