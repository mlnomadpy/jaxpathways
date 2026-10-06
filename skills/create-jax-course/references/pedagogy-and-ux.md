# Pedagogy and reader decisions

Read `docs/reference-course-review.md` for the inspected source revision, corpus scope and planned course/UX changes. The full corpus was structurally indexed; selective close reading does not establish technical verification of every lesson.

## Build a teaching sequence

Use concrete problem → visual mechanism → staged implementation → practical application → named artifact → transfer/diagnosis. Adapt this sequence to the question; do not add empty sections. A longer explanation and one finished code block do not constitute a staged build.

Before drafting, name the learner's starting knowledge, final artifact and one misconception. Introduce unfamiliar vocabulary at first use. For each build increment explain its purpose, which file to create/replace/append, expected intermediate result and how to verify it. Separate starter code from instructor solutions. Include a changed condition and a plausible failure, with reasoning rather than only the correct answer.

For beginner setup, show how to open the terminal, locate the workspace and select its interpreter. Every command needs shell, working folder, environment, expected output and a common repair. Do not require Git/Node/TPU merely because maintainers use them. A tiny downloadable workspace can remove a cloning prerequisite. Package pins and OS instructions must reflect real verification; one platform passing does not validate the others.

## Make figures instructional

Choose the learner question first. Label the inputs and intermediate values; ask for a prediction; connect the output to an independent runnable check. Use static shapes/architecture diagrams, process/dependency diagrams, or interactive parameter-sensitive mechanisms as appropriate. A decorative animation or unrelated widget adds no teaching value.

Use the same variables and shapes in prose, code and figures. Support keyboard control, text equivalents, reduced motion, narrow layouts and print/export fallback. Validate figure IDs at build time. Label analytic simulations separately from actual JAX execution and measured hardware results.

## Connect course and product

A shared foundation leads into work routes with ordered shared lesson IDs, bridges and projects. Current manifests reference phases; lesson-level routes are a proposed schema migration, not an available API. Update build, validators, CLI and reader together if implementing it.

Landing sections should help learners start, choose work/role, inspect a real outcome or navigate the course. Keep careers. Move detailed progress editing, organizers, API docs and coverage to their own pages. The reader should prioritize the artifact and first learning action; avoid duplicate quizzes, progress controls and route grids. Preserve selected route in navigation.

Track quiz correctness, attempted work, self-reported evidence, tested code and reviewed competency separately. Public fixture tests do not prove professional readiness. Validate one coherent learner journey before generating more shallow lessons or claiming parity with the reference.
