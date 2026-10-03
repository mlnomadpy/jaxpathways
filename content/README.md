# Curriculum authoring

The course is a connected body of lessons, not a notebook directory. Its source of truth is `site/curriculum.json`.

- **Core domains** organize the work: training, inference, operations, and research.
- **Career routes** explain the responsibilities and capabilities that those skills support.
- **Pathways** select ordered phases from the shared course and define a final project.
- **Phases** own canonical lessons, prerequisites, hardware requirements, a project, and a checkpoint.
- **Lessons** state an objective, exercise, evidence requirement, and understanding check.
- **Notebooks** become runnable lesson companions or phase integration labs.

There is one canonical home for each lesson. A pathway references phase IDs; it never copies lesson text. Phase prerequisites must form an acyclic graph, and every route must include prerequisites before the phase that needs them. `npm run check` validates these contracts.

## A consistent lesson

Use `LESSON_TEMPLATE.md`. Teach one question through intuition, an experiment, a change, diagnosis, verification, and evidence. Bring several lessons together in the phase project. Test the whole capability in the pathway capstone rather than counting quizzes.

An authored lesson needs a narrative, runnable code, expected results, exercises, solutions, numerical verification, and primary references. A notebook also needs pinned/tested dependencies, runtime metadata, a clean-run result, and hardware validation before publication. Never turn a brief into a published lesson just because it has a title and a quiz.

## Reusing the existing notebook topics

The training notebook belongs to phase 05; recovery to phase 06; profiling to phase 08. Their files are still unavailable here. Split them into concept-sized companion labs when imported, preserving one integration exercise for each phase. Do not repeat setup and foundational teaching inside every notebook.

`npm run curriculum:docs` renders `CURRICULUM.md` and phase READMEs for reading on GitHub. These are generated from the same manifest as the website.
