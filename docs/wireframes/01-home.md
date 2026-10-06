# 01 — Home

Route: `index.html`. Primary goal: understand the course and begin with an AI tutor. Browser lessons and career exploration are visible alternatives. Implementation uses the [landing identity design](../landing-identity-redesign.md). The subsequent Astro migration preserves this wireframe; implementation now lives in `src/pages/index.astro` and `src/components/home/`.

## Desktop

```text
JAX Pathways         Course / Learning paths / Careers / My learning / Resources
[Resume saved browser lesson, when applicable]

Learn JAX.                        Pure function → grad ─┐
From arrays                                     → vmap ┼→ training step
 to models.                                     → jit ─┘
Short course promise              Links to the three real lessons
[Start with a tutor] [Explore the course]
Python / CPU prerequisite

Your agent. Your JAX tutor.
1 Download, extract and open the workspace
2 npx skills add . --skill ...                         [Copy command]
3 Use start-learning ...                              [Copy prompt]
> Browser chatbot / setup help

Where do you want to take it?                          [All career routes]
Training             Inference             Research              Systems
Outcome              Outcome               Outcome               Outcome
Available / planned lesson counts (includes foundations)
[Learning path] [Career skills] for each direction

A look inside the course                              [Full curriculum]
Six selected lesson rows: actual phase number / title / explanation / duration
Course inventory: available lesson drafts / projects / planned lessons

Put the pieces together
[Regression concept drawing]                 [Classifier network drawing]
Project outcome + link                       Project outcome + link

[Offline book]       EPUB / Printable reader / First lesson online
Source / Feedback / JAX docs / Design credit
```

## Mobile

The header becomes an accessible Menu button. Without JavaScript, navigation remains visible. The hero stacks text above a simplified transformation diagram with readable labels. Tutor commands wrap, with copy buttons below the text on narrow phones. Work directions and projects become one column; lesson durations are omitted visually. The book and links stack below 380px. There are no dropdowns or text fields.

## Tutor entry

The primary ZIP contains course files and tutors together. Extract it, open `jaxpathways` in the agent, and run the local command in that directory. Node.js/npm is required. The Skills CLI handles agent selection. Copy buttons report success or manual-copy fallback; they never install anything themselves.

The remote GitHub skills are not published yet, so the advertised command uses the downloaded folder. Plain browser chatbots use an attached Markdown lesson rather than an npx installation. This explanation and the full setup link sit in an optional disclosure.

## Returning learner and source integrity

Preserve the browser Resume panel above the hero and all stored reading positions. Tutor plans use `LEARNING.md`; browser progress remains separate.

Astro components generate lesson previews, path availability and counts from the canonical curriculum, with selections in `src/data/home.json`. Lesson numbers identify actual positions, not a complete prerequisite sequence. Planned specialist material is counted explicitly. Project illustrations explain tasks and do not show invented metrics.

## Verification

Build, existing automated checks, local link destinations, source-derived previews, and copy success/failure should pass. Local browser access was denied by the browser tool, so actual desktop/mobile rendering and assistive technology checks remain pending. Do not report them as visually verified.
