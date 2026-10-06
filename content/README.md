# Curriculum authoring

The course is a connected body of lessons, not a notebook directory. Phase membership lives in `curriculum/course.json`, routes in `learning-paths/`, and authored content in the lesson’s `lesson.json` under `phases/`. The website manifest is generated.

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

`npm run curriculum:docs` renders lesson Markdown, Python scripts, notebooks, `CURRICULUM.md`, and phase READMEs for reading on GitHub. These are generated from the same manifest as the website.

Use `npm run smoke:cpu` after changing numerical material. It executes both the scripts and notebook code cells in fresh CPU processes and records the tested environment in `public/validation.json`. Browser checkpoints are formative checks; exercise completion is self-reported. Neither awards a reviewed competency.

## A lesson bundle

An authored lesson lives in `phases/<phase>/<lesson>/`. Edit its `lesson.json`; `npm run build` generates `docs/en.md`, `code/main.py`, `notebooks/exercise.ipynb`, `quiz.json`, and `outputs/evidence.md`, plus website downloads and API representations. Planned briefs have manifest entries without pretending to supply executable code.

Use problem → concept → small implementation → JAX/library implementation → modification → diagnosis → evidence when that progression helps teach the subject. Do not add empty sections just to satisfy a template. Interactive figures illustrate numerical behavior and state whether they execute JAX or show an analytic model.

## Depth before coverage

An executable toy example is a starting point, not sufficient teaching. Before extending the authored count, apply the editorial questions in [the content quality review](../docs/content-quality.md). Explain the mechanism with worked reasoning, check independent known results, ask for a prediction before an experiment, and require transfer to changed inputs or conditions. Introduce terminology at the point it is needed. Avoid substituting longer prose, more headings, or more quizzes for understanding.

Expanded lesson content can include `objectives`, concept `sections` (title/body/optional formula), worked `experiments` (prediction/code/output/explanation), `practice` (difficulty/prompt/hint/solution/explanation), `takeaways`, and primary `references`. These are optional schema tools, not a section-count rubric. All teaching and runnable solutions must survive generation into the reader, Markdown, notebook, script, and offline book. CPU receipts identify the content revision that was actually run.

## Friendly explanations and equations

Use the [learner writing and math guide](../docs/learner-writing-and-math.md). Author inline math with `\( ... \)` and standalone equations in a section’s `math` field; JSON requires escaped backslashes. KaTeX renders the browser and printable reader, while EPUB receives native MathML. A section’s `formula` remains a plain-text sketch. Begin with a concrete question, define symbols, work through one small result and invite a changed-condition attempt.

## Figures and recorded executions

Every authored lesson has a canonical `content.visual` with a prediction prompt, reading guidance and a connection to the lesson. Use `kind: executed` and a small `code` experiment producing `visual_data` for a real plot; use `kind: conceptual` and labeled nodes for a workflow diagram. Never substitute illustrative values for measured results.

`npm run figures:build` runs each complete example and its figure experiment in a fresh CPU process, then creates SVG, PNG and data/provenance JSON under the lesson’s `outputs/`. `npm run content:build` generates companions. `npm run smoke:cpu` executes scripts and notebook cells, retaining real streams and Matplotlib outputs in `outputs/execution.json`. Run `npm run build` afterward to attach those current results to the reader, notebooks and books. Content hashes and per-cell hashes prevent stale output reuse. `npm run audit:visuals` verifies coverage and updates the whole-course visual audit.

Notebook plotting helpers are self-contained and their source is initially collapsed by compatible readers; plotted outputs remain visible. Matplotlib is pinned in the CPU environment. Browser figures are saved reference results, not a browser Python runtime. Figure runs and full exercise runs have separate timestamps, so timing samples may differ.

## Worked reasoning and mechanism diagrams

The `guided-reasoning` section connects the introduction to a concrete case. Its `check` object contains `prompt` and `answer`; the reader and Markdown hide the answer behind “Compare your reasoning,” and the notebook and EPUB retain the explanation. Use this for reasoning, not an extra completion form.

Optional `content.diagram` supplies `title`, `prediction`, `reading`, `alt`, `scope`, `height` and a structured `panel`. The figure renderer creates `outputs/mechanism.svg` and `outputs/mechanism.png` alongside the existing evidence plot. Place this conceptual or analytic explanation beside the guided section. Never label a source-authored diagram as measured execution.

Mechanism panels support dependency graphs, axis/state tables, clipping geometry, patch layouts, receptive fields, covariance geometry and a conceptual host/device timeline. Graph edges identify endpoints and can specify ports, curves and labels. Explain arrows and verify dimensions against canonical code. Keep numeric JSON configuration canonical: use integers for integral positions and dimensions.

The build copies mechanism assets to the website and EPUB and embeds PNG attachments in downloaded notebooks. Source and renderer hashes govern freshness. `tests/guided-reasoning.test.mjs` checks source/export correspondence, hidden-answer markup, attachment integrity and stale-figure suppression. Independent hand calculations live in `assessments/guided-reasoning-reference.py`.
