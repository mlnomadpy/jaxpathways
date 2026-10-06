# Learning access upgrade — October 4, 2026

First implementation of the [wireframe pack](wireframes/README.md). Local preview; publication pending.

## What changed

- Shared navigation uses Course, Learning paths, My learning, and Resources. Phones get a single-row header with an accessible Menu disclosure and Escape dismissal.
- Home leads with Learn JAX / Build working systems, direct start, and visible destinations. Returning learners get the saved lesson first.
- Course navigation uses an expandable, connected phase sequence. Start/resume is direct. Search and route/environment dropdowns are removed from the visible UI; existing query links still work. Planned material is explicitly separated.
- Learning paths are grouped by work, with visible availability, a foundations entry, and a detail view. Use this path saves the selection and opens an available lesson; choosing a starting step never marks earlier practice complete. Career details remain optional.
- Lesson navigation uses direct current-phase links, a course-step disclosure, and a persistent activity bar. Existing section anchors remain stable. Introductory runtime/prerequisite notices and repeated setup explanations are expandable.
- My learning prioritizes continuation. Notes and evidence are optional; associations are prefilled when entered from a lesson or project. Artifact groups use visible choices. Edit, removal, undo, and exports remain available.
- Backup restore validates and previews the merge before writing. Cancel does not alter stored records. Invalid files leave the existing workspace untouched.
- Project setup is expandable; OS selection is a visible choice group, and stages have direct links. Assessments have task links with the full section index in a disclosure. Club planning uses visible budget choices and an explicit Build schedule action.

## Implementation notes

`experience.css` contains the new access/layout layer. `experience.js` provides mobile navigation, choice groups, contextual disclosures, and project/assessment links. Optional tools retain hidden select elements as internal adapters to the existing state handlers; users interact with native labeled radio groups. No saved-record schema migration is required.

Assessment generation includes the shared assets so a rebuild preserves the upgraded navigation. Asset versions were refreshed for the final preview. Existing authored lesson content, checker contracts, runtime receipts, and completion meanings are retained.

## Verification

- `npm run build` succeeded.
- `npm run check` passed: 29 tests plus curriculum/distribution validation.
- New restore tests verify preview/cancel without persistence, confirmed merging, and malformed-file rejection.
- Browser checks covered homepage, course, path list/detail, My learning, projects, assessments, resources, and club planning. Eight page families had no document-wide overflow at 320 px and no visible select controls.
- Verified path activation and resume, section deep links, project command changes for PowerShell, eight-week schedule generation, evidence associations after reload, removal/undo, and keyboard Menu dismissal.
- Desktop and 390 px layouts were inspected. No formal Lighthouse or assistive-technology audit is claimed.

Previews: [course desktop](upgrade-course-desktop.jpg), [course phone](upgrade-course-mobile.jpg), [paths desktop](upgrade-paths-desktop.jpg), [reader phone](upgrade-reader-mobile.jpg).

## Remaining design work

Further editorial mapping can refine individual lesson activities and add meaningful prediction interactions. The current upgrade uses existing authored sections and checks; it does not generate new quizzes, run Python in the browser, or claim reviewed competence. More detailed project/assessment presentation, tutor setup variants, and the offline-reader layout can be refined in subsequent iterations.
