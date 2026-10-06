# Source-driven course architecture

Reference revision inspected: `3be078b37ffd8f0c04953c0678e48f5c6d0c7775` of [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch/tree/3be078b37ffd8f0c04953c0678e48f5c6d0c7775).

## What the reference actually does

- `public/build.js` reads repository course sources and generates browser catalogs, metadata, and discovery artifacts. Its curriculum GitHub workflow runs that build and tests.
- `src/pages/lesson.astro` supplies the reader shell. Modules in `src/scripts/reader.js` and `src/lib/lesson-template.js` render the selected canonical lesson JSON, with navigation and exercise tools. Astro builds `dist/lesson.html`.
- `public/content-source.js` selects local repository content during previews and branch-aware raw GitHub URLs during deployment.
- Lessons group narrative, runnable code, quiz, tests/artifacts, and other companions under a phase folder. `learning-paths/*.json` selects lessons for particular work.

Generated content and an HTML frontend coexist. Changing to a framework is not necessary to reproduce this architecture; evaluate frameworks against concrete routing, authoring, accessibility, and maintenance needs.

## This repository's adaptation

| Responsibility | Canonical source | Generated/consumer files |
| --- | --- | --- |
| Phase membership and role/domain definitions | `curriculum/course.json` | `public/curriculum.json`, course Markdown, catalogs |
| Ordered learning routes | `learning-paths/*.json` | Browser route views and API resources |
| Authored lesson | `phases/<phase>/<lesson>/lesson.json` | `docs/en.md`, `code/main.py`, notebook, quiz, evidence template |
| Reader | `src/pages/lesson.astro`, `src/scripts/reader.js`, `src/styles/features/reader.css` | Consumes generated course content |
| Project | `projects/<id>/project.json`, starter/solution/checks | Project reader and downloadable bundle |
| Offline distribution | `scripts/build-distribution.py` | HTML book, EPUB, ZIP bundles |
| Validation | Validators, CLI contracts, CPU runner, project checks | Local results and CPU receipt |

The current JSON-first lesson source guarantees consistent browser, notebook, CLI, and book content. If Markdown authoring is introduced, explicitly designate one canonical narrative source, update generators/validators, and migrate a small lesson slice first. Do not create two independently maintained versions of the same lesson. Inspect consumers before changing schemas or URLs.

## Parity boundaries

Static JSON/OpenAPI resources provide discovery and reading, not remote execution, write APIs, accounts, or secured exams. Translation needs actual reviewed translated content. Offline exports need structurally valid artifacts. TPU claims need execution on the named TPU environment. Career readiness requires reviewed evidence beyond public quiz answers. A richer reference feature is inspiration, not proof of implementation here.
