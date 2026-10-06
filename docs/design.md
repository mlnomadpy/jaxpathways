# Design direction

Inspired by the technical field-guide feel of https://aiengineeringfromscratch.com/, with original layout, writing, and diagrams.

Palette: midnight #0c101b, panel #131a2b, ink #eef1fa, muted #a4aec6, periwinkle #8c9cff, mint #9ed9c5.
Type: bold monospace display and navigation; Georgia for reading text. No remote font dependency.
Layout: left-aligned introduction beside a transformation diagram; a shared foundation followed by branching goal-based routes; expandable curriculum and a full-page lesson reader with phase navigation and a section outline.
Principle: keep the reference's technical character while replacing intimidating curriculum totals with an approachable first experiment. Explain prerequisites and outcomes before library names.

Review: dark surfaces and blue accents are deliberate reference choices. The central visual is JAX's composable transformations, rather than a generic neural-network illustration. No borrowed testimonials, popularity counts, or completed-content claims.

This is a working editorial prototype. The supplied ChatGPT share was unavailable during initial authoring; its content has not yet been incorporated.

## Curriculum redesign

The landing page now has three jobs: explain the course, help learners choose the work they want to do, and show the ordered phase curriculum. It contains no evidence form, cohort planner, duplicate learning map, or career-card catalog.

Four core domains organize the work: training, inference, operations, and research. Career routes live on `pathways.html`; device-local learner tools live on `notebook.html`.

The course uses 17 numbered phases with short concept names. Each phase owns its lessons, prerequisites, hardware needs, project, and checkpoint. Pathways select those phases rather than repeating concepts under different notebook titles.

The supplied TPU notebook outline has been reconciled into phases 05 (training), 06 (recovery), and 08 (performance). Its actual notebook source and the previously described Astro collection remain unavailable.
