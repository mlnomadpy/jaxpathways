# 10 — Saved work and optional evidence

Views within `notebook.html`; opened contextually from lesson, project, or assessment. Goal: keep useful evidence with minimum bookkeeping.

## Contextual Save work panel

```text
Save work                                      [Close]
From: Regression audit / Stage 2
Default title: Stage 2 — gradient check

[Save this practice record]
This records your report, not your external output.

> Add an evidence link or explanation
  Evidence link (optional)     [text field]
  What did you verify?         [optional multiline note]
  > Rename this record        [optional title field]

[Save]  [Cancel]
Saved on this browser
```

Context is prefilled from the originating lesson/task/stage; never ask learners to select route, lesson, project, and stage again. Save this practice record creates a practice record only. It must not create a fake artifact or count as evidence. Adding a link or note creates an artifact using the existing evidence record contract; implementation must allow optional blank notes or retain the current validation transparently. Do not create blank artifacts to satisfy storage constraints.

Typing appears only when the learner chooses to add their own content. No upload service is proposed. Pasting a link does not fetch or verify the remote artifact. Cancel preserves the current draft until the panel is intentionally discarded; navigating away from a dirty draft must offer Keep editing or Discard draft.

## Saved work list — desktop

```text
[← My learning]
Your work
Jump to: [Lessons] [Projects] [Other experiments]

LESSONS
First gradient experiment
Context / date / note excerpt             [Open record]
PROJECTS
Regression audit / Stage 2                [Open record]
OTHER EXPERIMENTS
User-created records                     [Open record]
```

Group by context instead of dropdown filters. Existing general artifacts remain accessible. Add other experiment is a secondary action; offers visible association cards or None, optional title, and the same evidence panel. For a long history, show recent records and an explicit Show older work button with stable ordering.

## Record detail

```text
Stage 2 — gradient check
Lesson / project / stage links
Saved link (destination shown)
Explanation / observed output note
[Edit] [Download record] [Remove]

After remove: Record removed. [Undo]
```

No Reviewed badge without real reviewer provenance. Existing review-related records remain readable with their actual meaning. Editing keeps association unless Change context is explicitly opened; use grouped lesson/project lists, not selects.

## Mobile and failures

Open as an in-flow dedicated view with back link so the keyboard does not squeeze a tiny modal. Desktop may use a side panel with focus managed correctly. Form labels stay visible. Save failure retains text and offers Retry / Download draft. Invalid optional URL shows inline correction without erasing notes. Removal is recoverable with Undo; bulk purge is not part of this design.
