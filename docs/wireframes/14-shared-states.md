# 14 — Shared screen states

Use these inside the existing shell. Each supplies a useful recovery action without requiring a text field.

## Loading

```text
Course
Loading available lessons…
[neutral skeleton rows]
```

Do not show fake progress percentages, learner states, or counts. Keep header and navigation working. Announce completion once; avoid continuous loading announcements.

## Content load failure

```text
We could not load the lessons.
[Try again]                    [Open printable course reader]
```

Preserve local progress and path context. A failed fetch is not an empty course. If the user is offline, say so only when supported by actual connection state; the page is not promised to be a cached offline app.

## Unknown / unavailable destination

```text
This lesson is unavailable.
The link may be outdated or the topic may be planned.
[Browse available lessons]     [Explore learning paths]
```

For a known planned item, use its actual status rather than vague error copy. For missing IDs, do not guess a similarly named lesson. A route with some authored lessons remains browsable.

## Local save failure

```text
This change was not saved on this browser.
Your current work is still here.
[Try saving again]             [Download current work]
```

Keep input, attempts, and navigation usable in memory where possible. Persistent unsaved indication sits near affected controls. Download is offered only if an export can actually include the retained content. Successful retry replaces the message with Saved on this browser.

## Practice reported / next action

```text
Practice reported for this exercise.
[Continue to next lesson]      [Revisit the experiment]
> Save a note or link
```

No mastery claim, credential, or forced artifact form. If the next lesson is planned, show the next available relevant activity and clearly identify the curriculum gap.

## No saved work

```text
No work saved yet.
You can learn without keeping notes here.
[Resume learning]
```

No forced Add artifact CTA. Empty progress, saved work, and selected-path states have distinct copy.

## Copy / download feedback

Copy succeeds: Copied with a short live announcement. Copy fails: Could not copy; select the command text. Download failure retains the page and offers retry/source fallback. Do not report a downloaded file as executed or verified.

## Mobile / keyboard

Stack recovery actions with primary first. Error messages wrap; no toast-only explanation for important failures. Navigation errors place focus at the new page heading; inline errors remain near their control. Disclosure closure returns focus to its trigger. Restore/cancel operations preserve prior context.
