# JAX Pathways UI and UX audit

Audit date: October 3, 2026. Scope: the current local site at localhost:4173.

The site has a usable curriculum browser and substantial lesson material, but its learning journey is fragmented. A learner can select work, browse roles, choose a pathway, filter phases, download exercises, tick progress and save an artifact. Those actions do not yet form one continuous course experience. The biggest improvement is connecting them, not adding more homepage sections.

## What was inspected

- Browser inspection of the homepage, pathways/careers, beginner lesson, advanced Transformer lesson, learning notebook, classifier project and tools page.
- Six screens measured at 390, 768 and 1280px widths: homepage, pathways, notebook, advanced lesson, classifier project and tools.
- Interaction checks for domain exploration, inference career details, course route selection/reload and lesson section deep links.
- Source review of the shared lesson renderer, navigation, career controls, portfolio, backup/import, cohort planner, project renderer and both stylesheet layers.
- Generated curriculum inventory: 38 available lessons, 35 unwritten entries, 10 pathways, 7 roles, 2 implemented staged projects. Twenty-eight available lessons have staged build blocks. Only two lessons currently use the renderer's dedicated visual figures: gradient and attention mask.

This is a product and interface audit, not a new technical review of all 38 lessons, an exhaustive external-link test, a screen-reader certification, or a performance benchmark. Persistence failure findings below come from source inspection; storage failure was not forced in the user's browser. No learner evidence or completion records were changed. No application fixes were made during this audit.

## Preserve these strengths

The available/unwritten distinction, CPU versus real accelerator limitations, career routes and shared foundations should remain. Numbered builds, predictions before experiments, worked outputs, hints and independent numerical checks are useful. The existing white/blue palette, quiet reader and serif teaching text are a reasonable foundation. Local evidence and export/import are useful capabilities. These should be improved rather than discarded in another wholesale visual restart.

## Fix first: defects and misleading behavior

| Priority | Finding and evidence | Required change / acceptance check |
|---|---|---|
| High | **Explore domain fails.** Clicking the first domain's button produces `ReferenceError: id is not defined` in `choices.js:11`; details never populate. The same handler serves all domains. | Correct the handler and exercise all four domain buttons. Details, career links and pathway links must work with mouse and keyboard. |
| High | **The beginner download label is invisible.** Its computed foreground and background are both `rgb(51, 86, 198)`. A broad reader link rule overrides the primary-button text color. | Fix button/link specificity. Check default, hover and focus contrast on all lesson artifact controls. Screenshot: `ui-audit-reader.png`. |
| High | **Lesson section links lose their destination.** Opening `lesson.html?lesson=welcome-01#section-4` ends at a URL without a hash and scroll position zero. `showLesson` clears the hash and scrolls to the title. | Preserve initial anchors after rendering, and test direct opening, reload, back/forward and copying a section link. |
| Medium | **Course selection is not durable.** Selecting `models` leaves the URL unchanged; reloading returns to `all`. Search and environment changes likewise are not serialized by their handlers. | Put view state in the URL; keep saved learner route separate from a temporary catalog filter. Back/forward should restore the view. |
| Medium | **The tools page breaks phone width.** At a 390px viewport it has a 630px document width. The long command block is not contained. | Constrain and horizontally scroll code panels. Verify document width at 390px and 200% zoom. Screenshot: `ui-audit-mobile-tools.png`. |
| Medium | **Backup is incomplete across the product.** Notebook export includes notebook evidence and lesson progress, but excludes separate project stage keys and the saved career-plan key. | Export/import all learner-owned state through one versioned contract. State precisely what a backup contains; preserve existing merge behavior. |
| Medium | **Save success can be overstated.** Evidence submission clears the form and reports saved even when `save()` returns false. Import also reports success without checking every persistence result. | On failure retain inputs, say work is only in memory and offer export. Test failure handling in an isolated test environment. |
| Medium | **Backup import is not a keyboard control.** The visible trigger is a label for a hidden file input; the label has no keyboard activation semantics. | Use a focusable button connected to the file chooser, with a status announcement and clear invalid-file recovery. Verify with keyboard and assistive technology. |
| Medium | **Source links lose pathway specificity.** Every pathway card's GitHub link is derived from its first phase, so all nine specialization cards point to `00-welcome`. | Link each pathway to its own manifest/guide. Label foundation source separately. |
| Medium | **Classifier prerequisite routing uses an unknown pathway.** Its generated link has `path=training`, while the canonical training pathway is `models`. The lesson renderer falls back to the shared course. | Validate project pathway IDs against the catalog and preserve the selected route when entering prerequisites. |
| Medium | **Career status copy is stale.** Details say later lessons and projects remain planned, despite current authored training/recovery/Transformer lessons and two staged projects. Milestones do not show their individual availability. | Generate status from actual lesson/project coverage per milestone. Avoid implying that all specialization work is available or all is missing. |

## What is missing in the learning journey

| Area | Current gap | Improvement |
|---|---|---|
| Returning learner | Homepage always leads with the same beginner pitch. There is no last-opened lesson or saved section position. | A small **Continue learning** panel: selected career/pathway, current lesson, next action and a link to the full map. |
| Progress semantics | Stored lesson state contains only quiz passed and exercise evidence self-reported. The notebook's next lesson is the first without the evidence tick, not where reading stopped. | Track visited/current section, attempted practice and attached evidence separately. Reviewed evidence must require an actual review workflow. |
| Pathway orientation | A route is mainly a list of phase chips plus a final brief. Available foundations inflate route lesson counts even when the specialization itself is unwritten. | A route overview with ordered milestones, prerequisite bridges, runtime needs, available specialization lessons, implemented projects and explicit future gaps. |
| Career decision | Roles have skills and milestones, but choosing a role mostly leads back into another curriculum selector. | A career detail page that ends in one suggested first lesson and one visible project goal. Keep career routes prominent. |
| Practice handoff | The notebook download appears before guidance on where/how to run it. Later lessons assume a repository checkout while the first lesson uses a standalone ZIP. | A persistent practice drawer with **Local Python / Notebook** modes, exact setup, working folder, filenames, command and expected result. Include the transition from beginner ZIP to full checkout. |
| Lesson prerequisite | The reader prints prerequisite titles as text; it does not link a learner directly to missing preparation. | Link prerequisites, offer a quick refresher and distinguish required prior work from optional accelerator steps. |
| Assessment | The UI presents one multiple-choice checkpoint per lesson, with generic wrong-answer feedback. It cannot establish system-building capability. | Preserve formative quizzes; add targeted misconception feedback and project rubrics. Add pathway synthesis only when real assessment material exists. |
| Evidence | Portfolio entries are pathway-level title/link/note records. They have no lesson/project/stage association, editing, tags, validation outputs or history. | Attach evidence directly from the lesson/project, prefill context, allow editing and undo removal, and show the checks the artifact supports. |
| Review | Entries are labeled “awaiting review,” but no submission queue/reviewer workflow exists. | Say “self-reported; not reviewed” until review is implemented. Later add explicit review requests, feedback and revisions. |
| Projects | Project pages offer commands and checkboxes but no expected stage output, copy command buttons or explanation of common failed checks. | Reuse lesson practice components; show starter structure, stage contract, pass/fail examples and how to keep a report. |
| Search | Search checks phase/title/objective metadata, not lesson explanations, APIs, diagnostics or resources. | Search the actual course text with result excerpts and jump links; offer topic/API aliases. Keep availability filtering. |
| Cohorts | The planner divides whole phases arithmetically into 4/8 weeks, defaults to TPU and assigns unavailable work under a broad warning. | Move organizer tools into a separate area. Budget lesson time, show blocked units, select available work and include session goals, discussion and evidence review. |
| Discovery | Offline reader, tutor tools and APIs exist on `developer.html` but are absent from main learner navigation. | Provide a modest Resources page/footer link. Keep author/API tools secondary to learning. |

## Redesign the main surfaces

### 1. Homepage: a clear entrance, with careers retained

Keep a short explanation, two entrance actions (start foundations / choose work), four domain previews, a visible career-route link and two real projects. Move the full phase catalog to a Course page. The current homepage is 7,639px tall at 390px; a separate catalog would reduce the amount learners must traverse to orient themselves.

Use the small JAX transformation preview as the site's distinctive element. Do not add generic counters, decorative diagrams or repeated feature cards. For returning learners, a compact continuation panel belongs above exploration. Clearly label domains whose specialist lessons are still planned; “learn to serve” currently exceeds what the available inference material delivers.

### 2. Careers and pathway overview: one connected decision

Preserve Training, Inference, Operations/Performance and Research/Scientific Computing. Let the learner choose a domain, inspect roles inside it, and see one route overview. Replace the repeated homepage career buttons, role grid, separate route grid and catalog selector handoffs with a consistent detail surface and stable URL.

Suggested route layout:

```text
Inference & deployment engineer
What you will build | Who this suits | Available now / planned
Foundations → Trained artifact → Recovery → Measurement → Serving
Each milestone: lessons, required preparation, evidence, availability
Suggested first lesson                 View portfolio goal
```

Availability must describe the specialization, not only total foundation lessons. Avoid a readiness percentage or certificate generated from quiz ticks.

### 3. Lesson reader: organize by learning activity

Keep a full readable lesson, but group navigation into **Understand / Build / Experiment / Practice / Evidence**. The current Transformer lesson has 28 equal-weight outline entries and is 14,327px tall on desktop, 17,803px at 390px. It needs stronger grouping, current-section highlighting and saved position. Long lessons could become multiple sessions where the pedagogical boundary is real.

Keep code and its expected output together, use syntax highlighting and meaningful filename/language labels, and mark which snippets append to the same file. Offer a cumulative build view. Put environment setup in one consistent place rather than repeating several hardware notices at the top. Make conceptual diagrams part of explanations: shape flow, pytree/state ownership, PRNG splitting, scan carry, optimizer updates, recovery boundaries and sharding layout. A figure must explain a mechanism or let the learner test a prediction. Only two dedicated figures currently exist across 38 lessons.

On mobile, offer one compact **Navigate lesson** control for phase and section navigation. Keep the current lesson title and practice action easy to reach without fixing a large header over content. Do not force all substantive teaching into hidden tabs; disclosure should organize detail, not hide the course.

### 4. My learning: a useful workspace

Replace the repeated “notebook” introduction with the selected route, current lesson, next task and evidence needing attention. The artifact workspace should connect to specific lessons/projects, support edit/filter/undo and retain output links. Export/import belongs in a clearly labeled backup area, with all state included.

Keep local-first storage initially. Login, cloud sync and certificates are optional later features; none is needed to fix navigation and practice. Explain storage honestly and avoid displaying review states without a reviewer.

### 5. Projects: a workbench

Show the goal, required lessons, extracted folder layout and selected OS setup before stages. Each stage should have an implementation task, copyable command, known passing output, failure diagnosis, evidence slot and rubric. Project checkboxes should appear in My learning and backups. A downloaded project should have a complete standalone setup path, not leave “repository root” interpretation to a beginner.

## Accessibility and visual consistency

Native selects, forms, skip links, reduced-motion rules and visible focus styles already provide a useful baseline. The confirmed 1:1 download contrast and inaccessible import trigger need immediate attention. Follow with keyboard traversal, focus return, 200% zoom, contrast across states and an actual screen-reader pass.

The whole lesson article is `aria-live="polite"`; replace that broad announcement surface with targeted status regions to avoid potentially announcing large lesson replacements. Do not treat the native accessibility tree's rendering of pressed buttons as proof of a semantic defect without assistive-technology testing.

Navigation labels vary: “Course,” “Curriculum,” “Learning paths,” “Learning & career routes.” Use one vocabulary across pages, with current-page indication and consistent breadcrumbs. Small career/phase controls have 30–36px minimum heights; check touch spacing and enlarge crowded targets where necessary. Most styles are layered overrides of a legacy dark stylesheet; the invisible label is evidence that this is now a maintenance problem. Consolidate component rules and tokens after the behavior fixes. A framework migration is not a prerequisite for these improvements.

The five learner screens had no document-level horizontal overflow at the three measured widths. Long reader code scrolls inside its panels; that is different from the tools page widening the entire document. These checks do not certify every interactive expanded state or every lesson.

## Implementation sequence and proof

1. **Repair the existing flow:** domain handler, button contrast, anchors, route URL state, tools overflow, keyboard import, accurate status and project pathway links. Verify each reproduced defect and add meaningful interaction checks.
2. **Connect learner state:** one backup contract; saved lesson/section; direct artifact attachment; honest save failures; edit/undo. Verify a full export/import round trip including career plan and project stages.
3. **Redesign navigation and route pages:** separate Course from homepage; keep careers; show milestone availability and a suggested start. Verify a new learner can choose inference, see that serving is planned and still start available preparation without choosing the same route again.
4. **Redesign the reader and workbench:** grouped outline, current position, practice modes, expected outputs and conceptual visuals. Verify a learner can build, stop midway, resume, diagnose a failed check and attach evidence.
5. **Improve discovery and organizer tools:** content search, resources hub and time-aware cohort planning. Do not expand certifications or remote execution before the underlying course and review workflows exist.

The next design pass should be evaluated by a complete learner journey: **choose work → understand the route → start a lesson → run an experiment → diagnose a change → save evidence → resume later**. More cards or a different color palette would not establish that journey.
