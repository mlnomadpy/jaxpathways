# 00 — Shared system

## Identity

Use the existing [visual identity](../visual-identity.md). The branching mark and diagrams are the recognizable signature. Retain `#eeebfa` lavender, `#ffffff` reading paper, `#30216b` indigo, `#6043c4` violet, and `#252139` ink. Orange is for diagram annotations with adequate contrast, not small instructional text.

Display and interface: Avenir Next / Trebuchet MS stack. Long lesson prose: Georgia. Code: system monospace. Relax excessive headline tracking. Target body text 17–19 px, supporting text at least 14 px, comfortable reading line length around 60–75 characters. Confirm contrast in the final rendered UI.

## Desktop shell

```text
[branch mark] JAX Pathways     Course  Learning paths  My learning  Resources
───────────────────────────────────────────────────────────────────────────
Home / Course / Transformations                    contextual breadcrumb

PAGE TITLE
One short explanation.

CONTENT                                     OPTIONAL CONTEXT

───────────────────────────────────────────────────────────────────────────
Independent learning project · Curriculum in development
[JAX docs ↗]  [Source ↗]  [Accessibility/help]  [Data & backup]
```

Logo goes home. Active page gets text emphasis plus an underline and `aria-current`. Lesson, project, and assessment screens belong to Course. Club planner belongs to My learning. External links identify their destination.

## Mobile shell

```text
[mark] JAX Pathways                        [Menu]
────────────────────────────────────────────────
PAGE CONTENT

[Previous]                        [Next activity]
```

Menu opens an inline vertical panel below the header containing the four navigation links and Home. It is a navigation disclosure, not a select dropdown. Label changes to Close menu; expose expanded state; Escape closes and returns focus. No hamburger-only label and no second permanent navigation row.

Use one-column content at phone widths. Avoid a fixed bottom bar on ordinary browsing screens; use the activity bar only in learning screens. Reserve space for it and respect device safe areas.

## Component rules

| Component | Behavior |
|---|---|
| Primary action | One visually dominant action per local decision; verb names the destination |
| Path card | Title, concrete outcome, availability, one Open path link; no nested clickable card |
| Lesson row | Title, runtime label, available/draft state, learner practice state |
| Phase disclosure | Whole heading is a labeled button; expand in place; several may stay open |
| Choice tiles | Small, visible option set; selected label/check and native radio semantics |
| Category links | Jump to sections; no exclusive filter when all options can fit on a page |
| Activity step | Anchor in a complete readable document; does not hide essential content |
| Status banner | Explains the outcome and offers a next action; persists if recovery is needed |
| Code panel | Language label, Copy, wrap toggle; scrolling stays within panel |

A disclosure reveals content. Choice tiles choose a value. Navigation links change views. Keep those semantics distinct. Never replace a long select with dozens of equally weighted pills; group choices by meaningful work or prerequisite.

## Interaction contract

No dropdowns or text-entry fields for route, phase, operating system, duration, environment, role, or progress navigation. Optional evidence notes and links are permitted only after Save work is explicitly opened. File-picker buttons are permitted for backup restore. Coding remains an external learning activity; no embedded editor is required.

Selections update URL state where meaningful so Back, reload, and direct links work. Viewing a route does not save it; Use this path does. Browsing a lesson saves reading position but does not report exercise completion. Never silently infer mastery from time, scrolling, or a download.

## Accessibility and motion

All interactive targets have at least 44 × 44 px touch area. Maintain visible focus, logical heading order, skip link, native buttons/links, and text equivalents for diagrams. Accordions expose expanded state. Choice groups use fieldset/legend. Announce feedback without moving focus unexpectedly. Keyboard activation must match pointer activation.

At 200% zoom and 320 px width, content reflows without document-wide horizontal scrolling. Long code and tables have contained overflow. Respect reduced motion; connecting diagrams remain meaningful when static. Completion and selection never rely on color alone.

## Data and capability language

Show Available draft, Partly available, and Planned beside route decisions. Runtime labels distinguish laptop CPU, logical CPU devices, and real accelerator requirements. Local saving reports Saved on this browser; it does not imply cross-device sync. Practice reports are self-reported. Downloaded references are not learner submissions. Assessments are review drafts, not certificates.
