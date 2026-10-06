# 13 — Learning club planner

Route: `organizer.html`; secondary utility reached from My learning. Goal: generate a practical schedule through visible choices.

## Planner — desktop

```text
[← My learning]
Plan a learning club.
Choose what to study and how much lesson time you have.

1 WHAT WILL YOU STUDY?
[Start with JAX] [Model training] [Systems] [Other paths]
Use exact paths; Other paths expands the grouped path list.

2 HOW MANY WEEKS?
(●) 4 weeks             ( ) 8 weeks

3 LESSON TIME EACH WEEK
( ) 2 hours  (●) 3 hours  ( ) 4 hours  ( ) 6 hours
Extra setup, discussion, projects, and debugging need additional time.

[Build schedule]
```

These tiles are labeled radio groups. Default to the saved active path if available, otherwise foundations. Show explicit defaults without requiring re-entry. Systems is a group expansion, not an invented manifest; selecting it reveals actual paths before a schedule can be built. No club name, email, date, or freeform field is required.

## Schedule

```text
Start with JAX · 8 weeks · 3 lesson hours / week
[Adjust plan]                          [Download facilitator plan]

Week 1   available lesson titles / estimates
         Predict → run → change → discuss
Week 2   next available lessons
...

WORK OUTSIDE THIS SCHEDULE
Available lessons that did not fit
Planned topics, separately labeled
[Use longer schedule]                 [Increase lesson budget]
```

Follow the current scheduler: assign available lessons in prerequisite order; do not skip an earlier lesson that cannot fit and fill the week with later work. Distinguish overflow, unfinished curriculum, and extra project/discussion time. Estimated time is guidance, not a mastery promise. A schedule does not reserve hardware or issue credentials.

## Mobile

Choice groups wrap into two-column tiles or a vertical list. Build schedule follows choices. Schedule becomes stacked week cards; no wide calendar grid. Adjust plan returns to choices while preserving the generated schedule until a replacement is requested.

## States and checks

No lessons fit: explain which first lesson exceeds the budget and offer a larger budget/longer plan as appropriate; do not claim increasing weeks alone solves a weekly lesson-size constraint. Partial path: schedule available work and expose planned gaps. Download failure: retry and keep the visible schedule. Existing four/eight-week and budget choices are retained; arbitrary date scheduling is outside scope.
