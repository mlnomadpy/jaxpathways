# 15 — Learning flows and delivery plan

## A. New learner

```text
Home → Start learning → Setup lesson → Download workspace
     → visible OS choice → Run → Compare → next activity / next lesson
```

No path questionnaire, registration, search, evidence form, or hardware configuration before starting. Setup is learning content, not onboarding overhead.

## B. Returning learner

```text
Home or My learning → Resume → actual lesson section
                     → next activity → next lesson
```

Last-read location and reported practice remain separate. A returning user with only a selected path starts at its first available lesson. A corrupt position falls back to the lesson title with a clear notice.

## C. Project-motivated learner

```text
Learning paths → recognizable outcome → Path detail
               → inspect project / required lessons
               → Use this path & start → lesson journey → project stages
```

A partial/planned project route exposes its gaps before saving or starting. Available prerequisites remain useful independently.

## D. Experienced learner

```text
Course → expand phase → open any available lesson
or Path detail → choose starting phase → open available lesson
```

No placement quiz required. Skipping changes a chosen starting point, not prior practice records. Course/path context survives Back navigation.

## E. Optional evidence and assessment

```text
Lesson / project / assessment → Save work
                             → prefilled context → optional evidence → Save
My learning → Saved work → inspect / edit / remove with Undo
Project → Assessment → authored tasks → local review summary
```

No external submission or automated grading implied. Quick practice records and actual evidence artifacts are distinct. Typing only appears after choosing to keep personal notes or links.

## F. Move devices

```text
My learning → Data & backup → Download backup
New browser → My learning → Restore → inspect merge → restore → Resume
```

Backups preserve supported old schemas and current record families. Linked files remain external.

## Existing feature migration

| Current control / feature | Proposed access | Preservation requirement |
|---|---|---|
| Course route dropdown | Path cards / path-specific course links | Existing path query links still open correctly |
| Course search input | Phase journey and explicit topic titles | Keep authored topic/API labels visible; browser Find remains possible |
| Hardware dropdown | Runtime label on lesson/project | Logical CPU vs real accelerator distinction stays clear |
| Role search/domain select | Grouped seven-role list | Every role and its existing detail remains reachable |
| Reader phase select | Course map / phase disclosures | Existing lesson/section URLs and browser history preserved |
| Active route select | Change path → explicit Use this path | No accidental change when merely browsing |
| Evidence association selects | Context-prefilled Save work | Preserve existing associations and general artifacts |
| Evidence text fields | Optional evidence disclosure | Old records remain readable/editable; unsaved text retained |
| Artifact filters | Context groups + Show older work | All records remain accessible, no silent truncation |
| Project OS select | Visible radio tiles | Exact commands and validation caveats retained |
| Club selects | Visible path and budget tiles | Scheduler behavior and exports retained |
| Backup import | Accessible Restore button + preview | Accepted older formats and merge rules retained |

Search removal trades fast keyword lookup for visible browsing. Do not compensate with dozens of new controls. Make API names and descriptive lesson titles scannable, keep the full course readable, and reassess discovery through learner observation after implementation.

## Delivery order

1. Unify navigation vocabulary, mobile menu, card/row styles, and availability labels.
2. Implement Home continuation, My learning empty/resume states, and course phase journey.
3. Replace path/role filters with visible groups and path detail, preserving deep links.
4. Refine reader navigation and contextual setup; map existing sections into authored activities.
5. Refine project stages and assessment navigation; retain real task depth and runtime limits.
6. Move optional evidence and backups behind explicit actions; preserve and migrate local records.
7. Refine resources and club planner. Add authored interactive predictions where pedagogically useful.

## Validation for eventual implementation

Check new, returning, selected-path-only, partially authored route, invalid ID, unavailable storage, and imported legacy-data states. Verify keyboard operation, 320/390 px phones, tablet, desktop, 200% zoom, reduced motion, print layout, and contained code overflow. Check browser Back, section deep links, stage links, download paths, and no accidental completion from scrolling.

Observe learners starting a first lesson, locating a known topic, resuming, and opening a project. Look for unnecessary choices and lost context; do not claim conversion or learning gains without evidence. Completion criteria: all current page families represented, primary access requires no dropdown/text entry, actual practice remains substantial, and existing content/data stays reachable.
