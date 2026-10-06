# Course visual identity and component audit
October 4, 2026. Desktop is the priority, following the user's clarification.

## Finding

The branching mark, function diagram and violet reading margins give the homepage and reader a recognizable direction. The rest of the app largely inherits the previous card/form system. Color consistency is ahead of layout, interaction and component consistency. The next design pass should establish reusable course components and improve desktop composition before adding decoration.

This audit inspected homepage, course, career routes, learner workspace, organizer, resources, project and assessment screens, plus representative setup, gradient, scan and attention lessons. Desktop checks used 1280px; supplemental phone checks used 390px. It covers the shared UI component families, not an individual editorial review of all 38 lessons. Agents inspected screenshots and interaction states independently. No learner artifacts, completion flags or browser security settings were changed.

## Desktop priorities

| Priority | Finding and evidence | Recommended redesign | Completion criterion |
| --- | --- | --- | --- |
| High | Artifact association fields misalign. Optional captions become separate grid tracks; Project stage spans two columns while Related project occupies one. See identity-audit-tools-notebook-desktop.png. | Normalize captions, optional annotations and control tracks. Scope the outer form's last-label rule to direct children. Use a balanced two-by-two association grid. | All four caption/control pairs align at 1280 and 1440px, with no orphaned column or accidental row span. |
| High | The identity is applied through seven stylesheets containing 109 distinct literal color values. Many are legacy or unused; this is a source inventory, not 109 visible colors. Specificity already overrode syntax colors. | Consolidate foundation tokens, layout primitives and explicit components. Remove superseded rules after comparing each screen/state. | One authoritative definition for each component/state; ordinary changes require no new specificity patch. |
| High | Careers still use a generic card grid followed by a large white expanded card. Inference details measured about 2,084px tall at desktop; the unfiltered career page is 4,434px tall before expansion. | Give a selected career a focused route view: goal and portfolio at the top, an ordered prerequisite/milestone spine, availability at each stop and one clear next action. Keep the role picker separate from the selected route. | The route is understandable without reading a long undifferentiated card or scrolling past unrelated pathways. |
| High | Course catalog repeats two introductions. At desktop the search begins around y513 and the first phase around y728. | One compact title/description row, persistent filter toolbar, then phase rows. Put background explanation into optional help. | The first available phase and its useful context are visible within the initial desktop viewport. |
| Medium | Assessment article measures approximately 910px overall, with about 846px available for 19px serif text. Project paragraphs span about 980px; organizer descriptions span 992px. | Limit sustained prose to roughly 65–75 characters. Retain wide layouts for command panels, results and schedule comparisons. | Text width is controlled independently of the page/grid width. No long paragraph inherits the full utility-workspace width. |
| Medium | Organizer repeats the same session instructions in each week; assigned time, capacity and remaining work are buried in prose. | Put shared session guidance once above the schedule. Create a compact summary with distinctly labeled assigned time, capacity and unscheduled work. | Weekly panels emphasize actual lesson differences and overflow, rather than repeating boilerplate. |
| Medium | Resource command Copy sits below the block; project Copy is in its header; lesson Copy has another header treatment. | One code/command component with language, filename or context, Copy and feedback in stable positions. Keep expected results outside executable source. | Same component and interaction vocabulary across reader, project, resources and assessments. |
| Medium | Top navigation order and naming vary. Careers says Learning & career routes on its own page but Career routes elsewhere. Project, organizer and assessment pages have no parent navigation selection. Resources draws a second active underline. | Shared navigation with consistent order/names, one active indicator and explicit breadcrumbs for nested tasks. | Users retain their location while moving into a project, assessment or organizer. |
| Medium | Controls retain multiple font systems: Trebuchet/Avenir, system UI, monospace and Georgia appear in interface contexts. Notebook labels are 12px and input text 13px beside large headings. | Use interface type for controls and labels, reading type for prose, mono for code/data. Define a control type/height scale. | Optional captions, input values and help remain legible and visually related; no accidental monospace form labels. |
| Medium | Status colors do not have a documented semantic contract. Green rules frame route availability even where applied material is unwritten. Expected command results can appear more authoritative than learner evidence. | Define distinct visual states for available draft, unwritten, CPU receipt, self-report and independent review. Always pair color with literal labels. | A green line or token never implies reviewed competency or validated accelerator execution. |
| Medium | Diagram families remain separate implementations. Gradient captions are serif and low emphasis; scan captions use UI type; attention panels have independent color/type rules. | Define a teaching-figure grammar: title, inputs, transformation, outputs, controls, interpretation and execution scope. Preserve each mechanism's appropriate representation. | Figures feel related without turning different concepts into the same decorative chart. |
| Medium | The signature diagrams live in browser code; offline HTML/EPUB and notebook/source companions do not carry the same seven conceptual figures. | Store diagrams and captions alongside canonical lesson content and publish them in supported formats. Give offline material a restrained matching title/navigation treatment. | A learner can recognize and inspect the same mechanism in browser, repository and offline course material. |

## Component coverage

| Component family | What works | What still needs attention |
| --- | --- | --- |
| Brand mark and favicon | Branching shape relates to transformations and pathways. | Document minimum size, clear space and single-color use; do not introduce unrelated icons per page. |
| Homepage diagram | A specific JAX mechanism gives the page character; three selections update the preview. | Keep emphasis on the active transformation; do not proliferate decorative graphs. |
| Headings and page introductions | Strong indigo hierarchy on the homepage/reader. | Secondary pages repeat motivational headings and leave large unused desktop areas. |
| Primary and secondary actions | Main actions are recognizable and text contrast is strong. | Contextual actions, links and secondary buttons need a documented hierarchy. |
| Global navigation and breadcrumbs | Main destinations remain present. | Normalize labels, order, parent selection and underline treatment. |
| Domain entries | Homepage rows are more purposeful than identical cards. | Expanded content and availability disclosures need a consistent route-specific layout. |
| Career cards and selected role | Daily work and responsibilities are concrete. | Selected route needs a focused composition rather than another giant card. |
| Catalog filters | Search, pathway and environment are clear functional controls. | Toolbar is too far down; zero-results state should offer immediate clear/reset actions. |
| Phase accordions and lesson rows | Real phase order and draft counts support scanning. | Availability badges, hardware guidance and nested project descriptions compete for emphasis. |
| Lesson sidebars and contents | 205px phase navigation, 175px contents and central reading paper balance well at 1280px. | Align active-state grammar with other navigation; keep long contents from becoming a second competing lesson. |
| Lesson prose and build steps | Central reader text width is comfortable; build sequence is meaningful. | Preserve the distinction between explanation, implementation, observation and exercise through consistent spacing. |
| Python and command blocks | Shared syntax rendering added during this audit. | Standardize header/context and Copy placement; full parser diagnostics are outside this display-only renderer. |
| Equations and expected outputs | They remain distinct from executable source. | Give equations/output explicit components rather than incidental preformatted styling. |
| Gradient, scan and attention figures | Desktop interactions/scan columns are legible; keyboard controls responded. | Caption typography and panel treatments differ. |
| Disclosure controls | Native disclosure behavior works. | Several independent plus/minus and border treatments still exist. |
| Checkpoints and feedback | Formative feedback is separated from evidence. | Align question, answer choice, feedback and next-action hierarchy with the course visual grammar. |
| Project stages | Stage commands, expected checks and evidence links are inspectable. | Constrain prose; make expected vs observed output and stage progress visually explicit. |
| Artifact form | Useful associations are present. | Desktop grid alignment is the most immediate UI defect. |
| Artifact list, empty and review states | Empty guidance and review-packet affordance are present. | Establish reusable artifact rows and a review preparation summary; do not imply actual reviewer approval. |
| Backup controls | Import/export vocabulary describes the action. | Treat destructive/error/unsaved states as a consistent component family; this audit did not inject storage failures into the live browser. |
| Organizer controls and schedule | Real lesson time and overflow are shown. | Reduce duplicated guidance and distinguish key summary values. |
| Resource/download rows | Resource structure is relatively restrained. | Commands should share the course workbench component; Resources lacks the skip link used elsewhere. |
| Assessment actions, contents and rubric | Tasks and reviewer criteria have meaningful structure. | Long prose width and system-font overrides drift from reader typography. |
| Footers and offline material | Attribution and course-development status remain visible. | Unify footer content/layout and carry identity into generated reading material. |

## Code coloring fixed during the audit

The user reported uncolored code. Lesson blocks had a limited local highlighter, while the homepage preview and commands bypassed it. A legacy terminal rule also colored every span green.

Added site/code.js, a shared display-only Python/command renderer. Keywords, functions, strings, numbers, comments, flags and variables use distinct colors on a deep-violet code surface. Language metadata keeps text fixtures and expected output from being treated as Python. Source text remains intact for copying; embedded HTML is escaped. Commands recolor when the selected OS changes.

Three targeted tests verify Python source/HTML safety, PowerShell path preservation and literal text outputs. All 26 tests plus build/distribution checks pass. Desktop browser checks confirmed separate token colors in the homepage preview and colored project command flags. No numerical lesson implementations or validation receipts changed.

## Evidence and limits

Desktop metrics: identity-audit-desktop-metrics.json. Source-style inventory: identity-audit-style-inventory.json. Rendered contrast evidence: identity-audit-color-pairs.json and identity-audit-contrast.json. A sampled check of 594 normal-state text/background pairs across five pages flagged no text contrast failures. That is not a formal accessibility conformance audit: alpha compositing, placeholder text, SVG graphics, all hover/error states, screen readers and zoom still need dedicated checks.

Screenshots are stored under docs/identity-audit-*.png, including career expansion, catalog, artifact grid, organizer, projects, resources, setup, gradient, scan, attention, assessment and code. Earlier screenshots show the pre-fix code treatment; identity-audit-code-desktop.png records the corrected token colors.

Supplemental phone findings: the catalog's first phase is around y1179; the unfiltered career page is 8759px long, and selected inference content adds a 3215px panel. Some header links measure 38px high. Gradient SVG labels shrink to approximately 7.5px. Organizer controls are narrow/right-aligned. These remain secondary to the desktop priorities above.

This audit does not claim that all 38 lessons were individually reviewed, that every transient state was browser-tested, or that memorability has been measured with learners.

## Suggested next implementation order

1. Desktop component foundation: fix artifact grid, unify code/command panels, constrain prose widths and normalize navigation.
2. Desktop route composition: compact catalog introduction and replace expanded career cards with a focused route view.
3. Teaching and distribution consistency: shared diagram grammar, canonical figure assets, assessment/portfolio components and offline identity.

Keep the branching mark and function diagram. Avoid adding more decorative motifs until these layout and component contracts are stable.
