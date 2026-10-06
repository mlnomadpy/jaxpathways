# App architecture and reuse audit

Audited on October 6, 2026. Scope: Astro pages and shared components, browser entry points and controllers, domain libraries, CSS organization, storage contracts, content rendering, and existing verification tools. Existing workspace changes were preserved. This is a source and automated behavior audit; a rendered-browser accessibility or visual audit was not completed because the local preview server could not start in this environment.

The eight upgrade areas below have since been implemented. See [the implementation report](app-architecture-upgrades-2026-10-06.md) for current boundaries, verification and remaining incremental limits.

## Assessment

The app has a sound static Astro foundation and an acyclic JavaScript dependency graph. Page entry points are small, domain calculations have dedicated libraries, and build-time content is shared through `course-data.js`. Reuse is weakest in interactive controllers and the CSS cascade. The app does not need a framework rewrite: clearer feature boundaries and shared browser actions offer more immediate value.

## Findings and changes

| Priority | Finding | Result of this audit |
| --- | --- | --- |
| High | `workspace.js` handled unrelated club-planning, evidence-editing and backup responsibilities. The organizer awaited project data it never used. | Extracted `club-planner.js`; organizer initializes its planner before the notebook-only project request. Existing `setupPlatform` callers remain compatible. |
| High | Course loading accepted reading records with only version and object checks, leaving malformed values available to readers. Invalid or missing saved progress could leave old in-memory state after another load. | Reused `validateReadingState` and reset invalid reading/progress records to fresh defaults. Storage keys now reference the canonical constants. |
| Medium | Four copy implementations duplicated clipboard handling. Some used repeated event listeners or assumed source/status elements always existed. | Added `bindClipboard` in `browser-actions.js`; home, resource, lesson-code and project-command interfaces retain their own selectors and feedback text. Rebinding replaces the previous handler. Missing elements are harmless. |
| Medium | Download generation and select-option construction were embedded in feature controllers. | Added shared `downloadText`, `populateSelect` and `setStatus` actions. Notebook and career downloads share object-URL cleanup, including when initiating the download throws. |
| Medium | Project rendering maintained its own HTML escaping implementation. | Reused `escapeHtml` from `lib/html.js`. |
| Medium | Four guide pages duplicated section navigation markup. | Added typed `GuideContents.astro`, reused for assessments, project guides, project plans and the workflow guide. The existing navigation class, section anchors and HTML labels remain compatible. Labels must come from the trusted Markdown renderer, not user input. |
| Medium | Repeated `initExperience` calls could create new observers and choice adapters for the same document. A global click listener inferred notebook behavior from button text. | Initialization now uses a document-scoped `WeakSet`. Removed the text-dependent listener; notebook edit actions already open their own panel. |

## Upgrade recommendations from the original audit

1. **Split notebook evidence editing from backup restore.** `workspace.js` remains approximately 512 lines. Its nested handlers share mutable `state`, `pendingExtras`, `editingId` and `removed`. Extract backup UI behind explicit `getBackup` and `applyBackup` callbacks, then evidence editing behind a notebook-state contract. Keep restore validation, preview/cancel, storage rollback and retry behavior covered before moving them.
2. **Separate route views from route interactions.** `paths.js` combines career and pathway rendering, local storage, history and selection logic. Large HTML strings make view structure difficult to review. Extract pure view functions first, with explicit course/role/route inputs; then split career and pathway controllers. Remove its import of `catalog.js` by giving navigation a smaller boundary. Preserve focus restoration, browser history and starting-point calculations.
3. **Extract the project view.** `project.js` contains a large one-line HTML template alongside environment switching, storage and copy actions. Move markup to a pure project view module, leaving controller behavior in `scripts`. Avoid breaking shell-command conversion or stage evidence associations.
4. **Reduce global CSS ownership.** `base.css` has roughly 2,200 lines, while reader and pathway styles each exceed 1,200. There are repeated global heading, header and media-query rules. Inventory selectors and cascade order before removal; move feature rules into the existing feature stylesheets and keep only reset, typography and shared primitives in base. Consolidate rules only when doing so preserves source order. Use actual desktop/mobile screenshots before changing the cascade substantially.
5. **Strengthen type checking incrementally.** `tsconfig.json` allows JavaScript but does not enable `checkJs`. Astro diagnostics therefore do not establish full type safety for browser controllers. Add JSDoc contracts and `@ts-check` first to pure domain libraries, then type controller inputs and fetched manifests. Avoid hiding errors with broad casts or enabling strict checks across all legacy controllers in one change.
6. **Make network policy explicit.** Curriculum and project requests do not have the bounded timeout already used by the GitHub badge. The reader fetches validation receipts on every lesson render. Consider a page-local cached receipt request with retry after rejection, and a shared JSON fetch utility with consistent status and timeout behavior. Optional offline data should retain useful UI rather than fail the whole page.
7. **Use explicit choice enhancement contracts.** `experience.js` still enhances all selects, derives field names from label text, overrides individual select value accessors, and watches inserted DOM. Prefer opt-in enhancement attributes and a documented refresh operation. Include dynamic options, disabled states, form resets, keyboard access and no-JavaScript behavior in regression coverage.
8. **Separate build-only libraries visibly.** `assessment-content.js` and `project-guide.js` use Node filesystem/path APIs, while neighboring `lib` modules run in the browser. The current graph does not show accidental browser imports, but a `lib/server` directory or import-boundary test would make the distinction easier to maintain.

## Existing strengths to preserve

- Domain calculations and record validation are independent modules with substantive tests.
- Reading position, checkpoints, exercise evidence and project-stage reports have distinct storage contracts.
- Backup merging validates inputs and preserves existing records; persistence includes rollback behavior.
- Shared HTML escaping and Markdown/math rendering prevent interpreting learner notes as HTML. Repository Markdown rejects raw HTML.
- Static catalogs and guide links remain useful without client-side rendering.
- Deployment-aware links and static route checks protect root and GitHub Pages deployments.

## Verification

The refactor passes Astro diagnostics, ESLint, Prettier, a static build, and the expanded test suite. New regressions cover malformed/unavailable JSON storage, literal clipboard text, repeated clipboard binding, failure feedback, missing copy sources, safe select labels, and invalid reading/progress restoration. Existing integration tests continue to exercise notebook persistence, backup behavior, lesson rendering, pathways, careers and scheduling.

Follow-up refactors should keep the acyclic import test, browser migration tests, root/base static link checks and content checks as gates. Happy DOM tests verify behavior and DOM contracts, but do not prove rendered layout, responsive overflow, contrast or screen-reader usability.
