# JAX Pathways — screen and wireframe pack

Design proposal · October 4, 2026 · Markdown wireframes, not implemented UI.

## Product direction

Recognize the next useful action, click it, and learn. No dropdowns, search fields, or required profile questionnaires in the primary experience. Keep the existing transformation-atlas identity: lavender surroundings, white reading paper, deep-purple actions, and meaningful branching diagrams.

The beginner can start without choosing a career. The returning learner resumes their actual position. The experienced learner can browse every available lesson directly. Reading, practice, and independently reviewed capability remain distinct.

## How to read these files

Boxes show layout rather than exact dimensions. `[Label]` is an action, `> Label` is a collapsed disclosure, `●` is the current step, `✓` is a learner-reported practice state, and `○` is a future step. These symbols always have accompanying text in implementation. Wireframe titles, counts, and results are examples unless explicitly identified as existing content.

Every screen spec includes desktop and mobile layouts, actions, and important states. Shared accessibility and state behavior live in the system file. Existing URLs are preserved where possible; new details are proposed views, not claims that routes already exist.

## Screen index

| Screen | Spec | Existing location / proposed view |
|---|---|---|
| Shared navigation, components, identity | [System](00-system.md) | All pages |
| Home: new and returning learner | [Home](01-home.md) | `index.html` |
| Course journey and phase expansion | [Course](02-course.md) | `course.html` |
| Careers: visible role browser | [Careers](16-careers.md) | `careers.html` |
| Project-led path browser and roles | [Learning paths](03-learning-paths.md) | `pathways.html` |
| Path detail and starting point | [Path detail](04-path-detail.md) | Proposed detail view in `pathways.html` |
| Lesson reader and lesson navigation | [Lesson](05-lesson.md) | `lesson.html` |
| Setup, prediction, run, change, check | [Lesson activities](06-lesson-activities.md) | Views inside the lesson reader |
| Project overview and stage workspace | [Project](07-project.md) | `project.html` |
| Assessment overview, task, review | [Assessment](08-assessment.md) | `assessments/foundations.html`, `assessments/models.html` |
| Resume, milestones, saved work | [My learning](09-my-learning.md) | `notebook.html` |
| Save and inspect experimental evidence | [Saved work](10-saved-work.md) | Optional notebook panel / contextual view |
| Export, restore, local data status | [Backups](11-backups.md) | `notebook.html#backups` |
| Downloads, offline reader, tutors, tools | [Resources](12-resources.md) | `developer.html`, `downloads/jax-foundations.html` |
| Club planner and facilitator schedule | [Learning club](13-learning-club.md) | `organizer.html` |
| Loading, unavailable, error, completion | [Shared states](14-shared-states.md) | All pages |
| End-to-end flows and delivery order | [Flows](15-flows-and-delivery.md) | Cross-screen specification |

## Scope and constraints

This pack covers all current page families, plus proposed detail and activity states. Individual lessons, projects, and assessments reuse these templates; it does not invent a separate page layout for every lesson. Raw JSON APIs, source manifests, EPUB files, and notebook downloads are artifacts, not additional website screens.

Use canonical course and path manifests for titles, availability, prerequisites, ordering, and durations. Currently there are two authored integration projects: regression audit and classifier. Other destinations must expose actual partial/planned status; illustrative project ideas must not become active downloads or fabricated outcomes.

No account, cloud sync, browser Python runtime, automatic local-output verification, credential, or reviewer inbox is assumed. Existing browser-local records must survive the eventual redesign. The proposed lesson activities need editorial mapping; they must not divide every lesson into five equal steps mechanically.

## Review decisions

Recommended direction: project-led path cards; one expandable course journey; full-document lessons with activity anchors; compact mobile navigation; optional evidence tools. Build the simpler navigation and continuation first, then add authored interactive checks. See [delivery order](15-flows-and-delivery.md).
