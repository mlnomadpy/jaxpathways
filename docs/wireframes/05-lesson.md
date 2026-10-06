# 05 — Lesson reader

Route: `lesson.html?lesson=<id>`; retain path and section deep links. Goal: read, practice, and continue with little navigation overhead.

## Desktop

```text
SHARED HEADER (compact)
Course / Transformations / Your first gradient

LESSONS                  WHITE READING PAPER                 ACTIVITY LINKS
Transformations          Your first gradient                 Understand
[← Course map]           ~duration from content · CPU        Predict
                         Learn how output changes.           Run
Current phase lessons    > Before you start                  Change
● First gradient         [Get lesson workspace]              Check
○ Loss and gradients
○ vmap                   UNDERSTAND                          Current step
○ jit                    Diagram + short explanation         emphasized
○ Tracing                Worked example

[Other phases]           PREDICT
(expand course list)     Click an answer; explanatory feedback

                         RUN / CHANGE / CHECK (full document)
                         [Previous lesson] [Next lesson]
```

No phase select. Other phases opens an inline course list or dedicated navigation panel with phase disclosures. Available current-phase lessons remain directly visible on desktop. Planned lessons are separated into roadmap context.

Activity links are anchors in the same complete article, not a wizard that hides all later content. Preserve existing section IDs or provide aliases. Reading can be continuous; the activity bar moves between authored anchors. Do not automatically label a section completed after scrolling past it.

Before you start contains prerequisite links and any essential environment requirement. One recommended workspace action comes first; notebook, script, reference, and troubleshooting alternatives live in the relevant Run section. Setup lessons use the activity variant described in [activities](06-lesson-activities.md).

## Mobile

```text
LOGO                                    [Menu]
[← Course]                     [Lesson list]
Your first gradient
Laptop CPU · duration
Understand · activity 1 of authored total
> Before you start

Diagram / explanation / example

> Jump to an activity

[Previous activity]                 [Next activity]
```

Do not put a large lessons panel or table of contents above the title. Lesson list expands below its trigger with current-phase lessons and Other phases. Jump to an activity sits below the introduction and is also available from the activity bar. Keep bottom controls clear of code and feedback.

## Completion and progress

Save last lesson and nearest section anchor on this browser. Formative answers, exercise reports, and reading position remain separate records. Next activity is navigation only. At lesson end show [Continue to next lesson], [Revisit a tricky step], and optional [Save work]. A learner-reported practice toggle states I tried the exercise, never Passed lesson.

If all sections were visited but no practice reported, My learning says Last read / practice not reported. If the next prerequisite-dependent lesson is planned, show the next available alternative and explain the gap.

## States and checks

A direct section link opens the full article at that anchor. Back restores the previous section without losing answers. Missing lesson yields a useful recovery screen. Essential code and diagrams render even if progress saving fails. Keyboard and print reading work without sticky navigation. Runtime notices remain visible near Run; logical devices never imply real TPU performance.
