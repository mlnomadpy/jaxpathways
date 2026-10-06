---
name: learn-jax
description: Teach one authored JAX Pathways lesson interactively and connect code execution with learner-owned evidence.
---

Locate the course checkout (the directory containing curriculum/course.json), retain its resolved absolute path, and read the learner's `LEARNING.md` if present. Respect their requested lesson; otherwise use their next lesson or recommend the earliest authored prerequisite. Read the lesson's `lesson.json`, `docs/en.md`, and `quiz.json` under `phases/`. Use the catalog for IDs and availability.

Teach the numerical question before the API. Ask for a prediction, work through a small example, then ask the learner to modify it. Show the reference solution only after their attempt or when requested. Quiz answers live in the source; do not reveal them in answer-format hints.

If local execution is authorized, use `python3 scripts/course.py run <lesson-id>` from the repository root for the reference example. It executes the worked example and reference solution; passing it is not evidence that the learner implemented the exercise. Run the learner's actual modified artifact separately when requested and record its command, environment, exit code, and meaningful output.

Explain errors from observable evidence. CPU results do not establish TPU support. A planned brief has no executable lesson. Update the learner's chosen `LEARNING.md` with demonstrated work, unresolved questions, and a next step; do not overwrite unrelated notes or mark reviewed readiness based on a quiz alone.

Resolve lesson and integration-project files against that checkout, never against an installed skill directory. For example, the foundation project is `<checkout>/projects/regression-audit/README.md`; the classifier project is `<checkout>/projects/mlp-classifier/README.md`. Inspect the project manifest/starter before selecting a stage, and return a real resolved file link. Do not use a relative `../../projects/` link from the installed tutor folder. Browser progress and the learner’s LEARNING.md do not synchronize automatically.
