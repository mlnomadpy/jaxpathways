# 04 — Path / career detail

Proposed detail view inside `pathways.html`; preserve existing route and role identifiers in URL state. Goal: understand the work and start at a useful point.

## Desktop

```text
[← All learning paths]
Start with JAX                         PROJECT PREVIEW
Build a fitted regression model.       input → predict → loss → update
Available lesson drafts / CPU practice  [Inspect regression audit]
Python basics and a little NumPy

[Use this path & start]  [Browse lessons]

YOUR JOURNEY
1 Setup                    available drafts    [Open phase]
2 Arrays                   available drafts    [Open phase]
3 Transformations          available drafts    [Open phase]
4 State                    available drafts    [Open phase]
5 Optimization             available drafts    [Open phase]
6 Regression audit         project available   [Open project]
7 Synthesis assessment     review draft        [View assessment]

> Already familiar with the basics? Choose a starting phase
  [Setup] [Arrays] [Transformations] [State] [Optimization]

> Related work and careers
> Libraries and references
```

Use this path saves the route only after successful persistence and opens the earliest available required lesson lacking reported practice. If no practice is reported, start at the first lesson. Existing reading position takes precedence for Resume, which is a separate labeled action. Browsing any phase remains allowed without claiming readiness.

Choosing a starting phase is an explicit learner preference, not a proficiency test. It does not mark skipped lessons complete. Show prerequisite links before an advanced lesson, but do not force a questionnaire.

## Partial / planned route

```text
Inference and deployment
Partly available — serving project is planned.

AVAILABLE NOW
Shared foundations / relevant authored lessons [Start available lessons]
PLANNED
Adaptation → export → serving measurements → final project
[Browse foundations]   > View roadmap details
```

No Begin capstone button for an unwritten project. End available milestones with a useful adjacent lesson, not a dead-end completion celebration.

## Career detail variant

Lead with a concrete responsibility, then the role title. Show What you will practice, ordered available/planned milestones, portfolio expectations, and related path links. Do not promise employment, credentials, or a complete authored portfolio project when unavailable.

## Mobile

```text
[← All paths]
Start with JAX
Build a fitted regression model.
Available drafts · laptop CPU
[Use this path & start]
Journey (vertical, availability beside each step)
[Inspect project]
> Choose a starting phase
> Careers and references
```

## States and checks

Existing selected path: Selected on this browser + Resume, rather than asking to save again. Storage failure: Your selection could not be saved + Continue without saving; no false selected state. Invalid ID: show unavailable view and all-paths link. Back returns to the original card/group, not the top of a rebuilt list.
