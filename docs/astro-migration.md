# Astro migration

## Required outcome

All web pages, including assessment pages and the printable reader, build through Astro to static HTML. Existing `.html` links, query parameters, download paths, and local storage keys remain compatible. GitHub Pages serves `dist/` under a configurable project base path.

## Source boundaries

- `curriculum/`, `learning-paths/`, `phases/`, `projects/`, and `assessments/`: canonical educational content and evidence contracts.
- `src/data/`: editable site copy, navigation, presentation selections and resource metadata.
- `src/layouts/` and `src/components/`: shared page shells and reusable semantic markup.
- `src/styles/`: design tokens, shared foundations and feature styles; no historical override-file chain.
- `src/lib/`: pure domain functions, data access and storage contracts with explicit imports/exports.
- `src/scripts/`: browser feature controllers and page entry points; no implicit cross-script globals.
- `public/`: static assets and generated JSON/downloads; no duplicate authored website HTML or browser source scripts.
- `dist/`: generated deployment output only.

The Python distribution pipeline remains responsible for portable course/EPUB/exercise downloads. Astro owns website HTML. Editing a lesson or site-copy record must not require editing markup.

## Verification gates

1. Clean reproducible install and build with a lockfile and supported Node version.
2. Astro diagnostics, JavaScript lint, formatting, curriculum validation and existing behavioral tests.
3. Static output checks for every route, asset, link and course resource, under both root and GitHub Pages base paths.
4. DOM behavior tests for catalog filtering, lesson navigation/checkpoints, saved-state resume, career/path selection, project progress, notebook evidence and backup, organizer scheduling, navigation and copy controls.
5. No legacy pages/scripts/styles are served from public; no required feature relies on shared window globals.
6. Document authoring, module responsibilities, commands, hosting configuration and remaining verification limits.

No remote publication is part of this migration. The deployment workflow is prepared locally for later activation through the repository's GitHub Pages settings.

## Authoring without markup edits

| Change | Source |
| --- | --- |
| Lesson order, roles, domains, availability | `curriculum/course.json` |
| Lesson explanation, runnable code, quiz | The lesson's `phases/.../lesson.json` |
| Learning route | `learning-paths/<id>.json` |
| Project stages and rubric | `projects/<id>/project.json` |
| Assessment content and metadata | `assessments/*.md`, `curriculum/assessments.json` |
| Landing headlines, command, preview selections | `src/data/home.json` |
| Navigation and page metadata | `src/data/navigation.json`, `src/data/pages.json` |
| Resource cards and download links | `src/data/resources.json` |
| Path groupings and conceptual figure text | `src/data/path-groups.json`, `src/data/concept-figures.json` |
| Colors and font families | `src/styles/tokens.css` |
| API specification and execution receipts | `curriculum/openapi.json`, `curriculum/validation.json` |

Run `npm run content:build` after editing educational sources. `npm run dev` and `npm run build` run this step automatically. Generated public files are ignored by Git; do not author them directly. The course CLI and portable ZIP now read `public/curriculum.json`; the archive preserves that same directory layout.

## Components, styles and browser behavior

`SiteLayout` owns document metadata and the shared header. Navigation data is defined once. Homepage sections, phase cards, path cards, career cards, resource cards, assessment actions and printable lesson chapters are components. The landing page, catalog, path and career lists, assessments, and printable reader contain useful content in their generated HTML.

`src/styles/tokens.css` is the shared identity source. `base.css` supplies shared elements and layout; feature files cover reader, catalog, paths, projects, workspace, resources and assessments. Homepage component styles live under `src/styles/home/` and are imported by those components. Historical style/design/identity/refinements override files are removed. Exact-selector declarations superseded later in the same cascade context were eliminated without moving the remaining rules.

Each page imports a small entry module. Pure functions for careers, scheduling, evidence, syntax coloring, search and learner-record validation live in `src/lib/`. `course-state.js` is an explicit module store, not a window global. The query-based lesson reader and selected path/role details remain browser-driven to preserve existing links and local progress. Their view functions and interaction controllers are separate modules; no React runtime or hydration framework is required. Changing a quiz result still never marks an exercise complete.

Astro handles bundling, hashed JS/CSS filenames and deployment asset prefixes. Static links use `siteUrl()`; browser-relative links remain relative to the preserved top-level `.html` pages. There is no client-side page-transition router, so controllers initialize once per document.

## GitHub Pages

The configuration defaults to `/` locally. Set `BASE_PATH=/jaxpathways` and `SITE_URL=https://mlnomadpy.github.io` for this repository's Pages build. `build.format: 'file'` preserves the `.html` output paths. `npm run check:deployment` verifies the prefixed build and restores a root build afterward.

`.github/workflows/deploy-pages.yml` uses the official Astro build/upload action and the Pages deployment action. It builds and checks before deployment, on a main-branch push or manual dispatch. The repository must use **Settings → Pages → Source: GitHub Actions**. Adding the workflow locally does not publish anything; no repository settings, push or deployment have been performed during this migration. Instructions follow [Astro's GitHub Pages guide](https://docs.astro.build/en/guides/deploy/github/).

The separate course-check workflow installs the lockfile with `npm ci`, checks root and subpath builds, and retains the CPU reference tests. A lockfile and Node engine constraint make installations repeatable.

## Verification scope

`npm run check` runs Astro diagnostics, ESLint, Prettier, canonical content validation, the existing domain tests, DOM behavior tests, portable-download fidelity checks, and static site checks. The DOM tests use isolated synthetic browser storage. They inspect every authored lesson and exercise navigation, quizzes, career saves, project stage ticks, evidence forms, schedules, resume, menu and copy behavior. They do not read personal learning records.

`check-site.mjs` inspects all generated routes, links, accessibility references, static catalog counts and CSS asset paths. `check:deployment` repeats output and DOM checks at the GitHub Pages base path. Actual pixel rendering remains unverified because the browser tool denied local-preview access earlier in the conversation; DOM tests do not substitute for screenshot inspection.

## Migration verification receipt

- Clean `npm ci --offline` from the lockfile succeeded with zero reported vulnerabilities.
- Deleted only known generated `public/` directories/files and `dist/`; `npm run build` rebuilt all 12 routes and course artifacts successfully.
- `npm run check` passed: zero Astro errors/warnings, lint and format checks, 37 tests, canonical curriculum validation, portable bundles and EPUB checks, and 392 local references across the static output.
- `npm run check:deployment` passed for `/jaxpathways/`, including seven DOM behavior checks, and restored the root build.
- A freshly extracted course-workspace ZIP successfully ran `scripts/course.py list --available` and `scripts/course.py show welcome-01`.
- A selector-coverage comparison found and corrected missing course lesson-list and phase-project styles during the split. Pixel rendering still requires a browser review when local-preview access is available.
- The module-architecture test verifies acyclic imports and prohibits domain libraries from importing browser controllers. The Astro printable reader also preserves core explanations, code, solutions and evidence for all 38 authored chapters.
- All changes remain local. GitHub Pages has not been published.
