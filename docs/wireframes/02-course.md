# 02 — Course journey

Route: `course.html`; preserve existing path deep links. Goal: see order, availability, and the next lesson without filters.

## Desktop

```text
Your route through JAX.
[Resume current lesson]                [Explore learning paths]

SHARED FOUNDATION
● Setup & first steps       3 available drafts          [Collapse]
  [Set up your workspace]                 Laptop CPU
  [Meet your arrays and devices]          Laptop CPU
  [Practice with four logical devices]    Logical CPU devices
○ Arrays & pure functions                              [Expand]
○ Transformations                                      [Expand]
○ State & loops                                        [Expand]
○ Optimization                                         [Expand]

BUILD FROM THE FOUNDATION
[Model training] [Recovery] [Transformers] [Performance]
[Distributed computing] [Internals]
Each link jumps to its phase section below.

More available phase sections, in canonical order

> Planned curriculum — view upcoming topics
  Scientific computing / Probability / RL / Kernels / Deployment / Operations
```

Expand the current or first phase initially. Phase headings include availability and reported practice counts, not a single ambiguous completion percentage. Available lessons are ordinary links. Planned entries are text with roadmap detail, not disabled buttons that resemble working lessons.

Path-specific view shows `Your path: Start with JAX [All course phases] [Change path]`. Ordered milestones are derived from that path. Other topics remain reachable through All course phases; browsing never overwrites the saved path.

## Expanded phase

```text
Transformations                     5 available lesson drafts
Understand how one function gains new capabilities.
✓ First gradient                    Practice reported   [Open]
● Losses and value_and_grad          Last read           [Resume]
○ Batch a function with vmap         Laptop CPU          [Open]
○ Compile with jit                  Laptop CPU          [Open]
○ Tracing and recompilation         Laptop CPU          [Open]
[Project connection: regression audit]
```

The first-gradient example title/ID mapping comes from content; do not infer status from ordering. No search field, route select, or hardware filter. Runtime is visible where it matters.

## Mobile

```text
Course                                [Menu]
[Resume lesson]
Setup → Arrays → Transforms
(vertical journey continues below)

[● Transformations                −]
  First gradient               [Open]
  Losses                       [Resume]
  vmap                         [Open]
[○ State & loops                  +]

> Planned curriculum
```

Use a vertical sequence, not a horizontally scrollable mandatory map. Do not put a second sidebar above content. Expanding a phase leaves keyboard focus on its heading. Deep links expand the matching phase and target the lesson row.

## States and checks

Loading uses neutral row skeletons without fabricated counts. Failure offers Retry and printable course reader. Empty path shows No available lessons yet, a foundations link, and its actual roadmap. Completed practice leaves lessons openable and offers the next available milestone.

Acceptance: any available lesson is reachable by expanding its phase then opening it; browser Back restores path and expanded phase context.
