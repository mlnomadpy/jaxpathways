---
name: create-jax-course
description: Author or improve JAX courses, pathways, lessons, and staged projects in this repository, using the source-driven course patterns inspected in AI Engineering from Scratch. Use for curriculum building and course architecture changes.
---

# Create a JAX course

Start from learner work: what they will build, diagnose, or explain, and what evidence demonstrates it. Preserve the existing career routes and shared prerequisites when extending the course. Read [the architecture reference](references/architecture.md) when deciding source layout, generation, or reference parity; read [pedagogy and UX decisions](references/pedagogy-and-ux.md) when drafting teaching steps, figures or learner flows, and read `content/README.md` and `LESSON_TEMPLATE.md` before authoring.

## Inspect before changing

Work from the course checkout. Read `curriculum/course.json`, the relevant `learning-paths/*.json`, the target phase README, and nearby authored `lesson.json` files. Inspect the actual generator and validators rather than guessing schemas. Identify prerequisite lessons, CPU/accelerator requirements, available source notebooks, and current content status. An earlier progress report is not proof that files or execution results exist.

When the user refers to the inspiration repository, inspect the relevant source files and record the inspected revision. Use its lesson bundles, build contract, route manifests, staged projects, and reader behavior as decision evidence. Preserve attribution; adapt the pattern with original JAX teaching rather than copying text or source wholesale.

## Author a coherent slice

Choose a bounded lesson or connected phase that leads to a concrete artifact. Avoid expanding every pathway with shallow placeholder prose. A useful lesson asks one question and supplies:

- A concrete problem and an explanation of the mechanism.
- A small runnable example with expected outputs and numerical tolerances.
- A modification the learner implements, a reference solution, and a failure to diagnose.
- A formative checkpoint whose explanation teaches the misconception.
- An evidence requirement and primary technical references.

Use plain NumPy/math → JAX transformation → ecosystem library progression where it helps understanding. Do not add empty template sections. Figures must clarify a mechanism: label analytic illustrations separately from actual JAX execution. For gradients, shapes, state, randomness, compilation, and performance, test meaningful invariants and independent comparisons. Avoid assertions that merely repeat the implementation.

Verify version-sensitive JAX/Flax/Optax APIs against primary documentation and the installed environment. State hardware and precision assumptions. CPU execution does not establish TPU correctness or performance. Never fabricate accelerator results, benchmark timings, learner success, expert approval, or certification readiness.

Before calling a lesson ready for editorial review, apply `docs/content-quality.md`: require a worked mechanism, independent checks, transfer to changed conditions, actionable diagnosis, and inspectable evidence. A title, runnable toy, and quiz establish availability, not teaching depth. Compare a learner's likely misconceptions with what the explanation and practice actually address. Use the expanded array/transformation lessons as local examples, while improving their limitations rather than treating them as an unquestionable standard.

## Keep sources and outputs distinct

Edit canonical manifests and `phases/<phase>/<lesson>/lesson.json`; do not hand-edit generated lesson Markdown, scripts, notebooks, catalogs, or books. Route manifests reference shared phase IDs, with prerequisites ordered before their dependents. Keep authored and planned status accurate. Do not turn missing notebook files into claimed imports.

For an integration project, keep starter, reference implementation, stage checks, stage commands, and evidence rubric together under `projects/`. Check multiple cases and failure modes to detect hard-coded answers. A quiz pass, public synthetic test pass, self-reported evidence, and reviewed competency are different states. Reflect those differences in the UI.

Change reader components only when the teaching requires them. Keep the homepage focused on starting, choosing work, and navigating the shared course. Put portfolio forms, tools, and organizer features on their own pages.

## Build, verify, and report

Run `npm run build` and `npm run check`. After numerical edits run `npm run smoke:cpu`; after foundation project edits run `npm run test:project`. Run the skill validator after changing this skill. Inspect the generated output and the relevant browser flow, including narrow-screen usability when layout changes. Report failures rather than relabeling failed material as validated.

Finish with the concrete change, available learner action, tests actually run, and material gaps. Publication, cloud runtime spending, and messaging need their own task authorization; this authoring skill grants none of them. Keep prior work and learner progress intact.
