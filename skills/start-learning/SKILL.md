---
name: start-learning
description: Onboard or resume a JAX Pathways learner, choose an available starting lesson and create a persistent study plan grounded in the local course and the learner’s evidence.
---

# Start learning JAX

Find the JAX Pathways checkout before selecting lessons: it contains `curriculum/course.json`, `learning-paths/`, and `phases/`. Search the current workspace and a checkout path already provided by the learner; ask for its location if absent. Installing a skill does not install the course. Resolve every course file and project link against this checkout, never against the installed skill directory. Read canonical sources first; generated `public/api/v1/catalog.json` and `public/validation.json` are useful availability/execution views when present.

## Resume before onboarding

Read `LEARNING.md` in the learner’s chosen workspace when it exists. For “continue” or “resume”, preserve the recorded goal, route, attempts and review queue; resolve the next lesson ID against the current curriculum. Hand off to `learn-jax` when available, or follow the checked-out `skills/learn-jax/SKILL.md`. If the recorded lesson was renamed or is unavailable, show the discrepancy and recommend an available prerequisite; do not invent a substitute completion.

For a new goal alongside existing progress, offer an additional route or a placement update without overwriting history. Archive/reset a plan only when the learner requests that action. A correct quiz, reading completion, a reference-script pass, learner-written executable work and independent review are separate evidence states.

## Choose the work and starting point

Use information already supplied. Ask only the missing decisions that affect the plan: what they want to build, what Python/array/JAX work they can already demonstrate, and their available runtime or study pace. Keep the interview short; an undecided learner can start with shared foundations. Route role-specific goals through the roles and pathway IDs in `curriculum/course.json`; do not fabricate a job guarantee or a new pathway ID.

Read the chosen `learning-paths/<pathway-id>.json` and prerequisite phase/lesson chains. Respect a learner’s explicitly chosen topic while explaining any missing prerequisite. Starting-point self reports are suggestions, not graded placement. If practical placement would help, ask for a small prediction or inspect an existing artifact; do not reveal an assessment’s answer while asking its question. Mark an untested claim as self-reported.

Recommend the earliest useful **authored** lesson whose prerequisites the learner can follow. Show unavailable topics as roadmap entries and explain where the route currently stops. Do not convert planned briefs into lessons, claim the full course is complete, or skip a missing prerequisite silently. Estimate pace using authored lesson minutes only; make unknown roadmap effort explicit.

## Begin with CPU evidence

For a new learner, use `welcome-01` (workspace setup), then `welcome-02` (arrays/devices). Confirm the interpreter, package environment and actual backend before discussing accelerator execution. Explain each terminal action using the setup lesson rather than assuming the learner knows virtual environments. Read `requirements-cpu.txt` for this checkout’s tested versions; setup execution/installations follow the learner’s request and available host permissions.

Offer `welcome-cpu` to practise placement with four logical CPU devices. Its code configures `jax_platforms=cpu` and `jax_num_cpu_devices=4` before backend initialization. A notebook requires kernel restart and Run all with this block first. Logical devices share the host; they exercise meshes/shards and global-array arithmetic, not TPU memory, network, speed or multi-host failures. `distributed-01` and `distributed-02` extend that practice after their required concepts.

Report a lesson’s runtime receipt only when it matches the current generated content revision. A locally authored lesson can be studied without claiming a TPU receipt, expert approval or reviewed competency. Do not provision cloud resources merely because the selected route mentions TPU.

## Keep a learner-owned plan

Create or carefully update `LEARNING.md` in the chosen workspace. Use canonical IDs and inspectable artifact paths. A useful plan includes:

- **Goal:** the learner’s own build goal; optional role ID, pathway ID and pace.
- **Course:** resolved checkout path/revision and runtime choice.
- **Starting point:** next lesson ID, prerequisite reasoning and self-reported or demonstrated placement evidence.
- **Route:** ordered phase IDs with available lessons, planned gaps and intended integration projects. Preserve the full route even when only its foundation is available.
- **Progress:** date, lesson ID, reading/checkpoint result, learner artifact, command/runtime observation and remaining question. Label reference execution separately.
- **Review queue:** misconceptions or exercises to revisit, without exposing unrevealed quiz answers.
- **Next action:** one specific lesson task the learner can perform now.

Browser-local progress does not automatically synchronize with this file. If an exported browser backup is supplied, preserve it and reconcile the learner’s requested items explicitly; do not silently overwrite either history.

Finish with their next available lesson, why it fits their goal, and the host-correct way to use `learn-jax`. In Codex use `$learn-jax` or skill selection; in Claude Code use `/learn-jax`; when the host is unknown say “Use learn-jax to continue my plan.” Offer `jax-course-guide` for a specific topic and `check-jax` for formative practice. Installation, publishing, account changes, remote messages and credentials are outside this onboarding workflow.
