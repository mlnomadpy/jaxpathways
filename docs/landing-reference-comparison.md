# Landing page comparison

2026-10-05. Reference: [AI Engineering from Scratch](https://aiengineeringfromscratch.com/index.html). The reference was inspected in the browser at its normal desktop viewport, including the hero, domain map and curriculum. JAX Pathways was inspected through its current HTML and CSS; local browser inspection was previously blocked by the URL policy, so rendered layout and mobile findings are not asserted.

## Conclusion

The current JAX Pathways landing page presents installation more strongly than the course. The tutor-first revision removed repetitive marketing copy, but also removed useful learning structure and examples. The solution is to show more of the actual course while making installation compact.

## Observed comparison

| Area | Reference | Current JAX Pathways | Consequence |
| --- | --- | --- | --- |
| Hero | Large course title, declared scope, short terminal band | Tutor headline and a three-step installation panel | Ours explains setup before demonstrating the value of the course |
| Type and layout | Pixel display headings, serif explanations, mono controls, long ruled rows | Shared sans headings/prose, white launch card, small inventory | Our hierarchy has less variety and its most substantial visual is setup |
| Learning structure | Connected domain diagram and visible curriculum rows | One paragraph and links to other pages | Visitors cannot scan our actual learning journey on Home |
| Career discovery | Work domains and career entry near the learning map | Careers now in navigation and a secondary Home link | Access is repaired, but Home still gives few examples of the work |
| Resources | Book editions presented as course artifacts | EPUB and printable-reader links | Our offline edition is easy to miss |

Reference observations describe presentation, not independent verification of its scale, affiliations or content quality. Its newsletter/trivia and numerous hero actions add distractions; our two-action hierarchy and lack of required input fields remain useful.

## Changes in priority order

### 1. Make the hero represent the course

Suggested heading: **Learn JAX, from arrays to models.**

Suggested supporting sentence: **Work through CPU exercises and checked projects with an AI tutor that helps you understand each step.**

Two main actions: **Start with a tutor** and **Explore the course**. Keep the command visible in a compact terminal band. A required “Get the course workspace” link must remain beside the local command, because `npx skills add .` needs the extracted course folder. The full three-step walkthrough belongs in setup help or an optional disclosure. Do not imply that installing skills launches a chatbot or downloads Python automatically.

### 2. Restore a meaningful learning map

Show the real progression: arrays → grad/vmap/jit → state and loops → optimization → models and specialization. Use the established JAX branching identity. Link the nodes to actual lessons or routes. One readable map is enough; no decorative animation is needed.

### 3. Show outcomes and career directions

Use four concise choices matching the existing domains: train models, inference/deployment, scientific/research work and systems/performance. Each needs one concrete outcome, an availability label and a destination. Keep Careers in main navigation and include explicit role links. Do not present all planned specialist work as available.

### 4. Let visitors inspect real lessons

Show a small set of actual lessons rather than only counts. Examples from the current curriculum: Meet your arrays and devices; From NumPy to jax.numpy; Your first gradient; Batch a function with vmap; Compile a function with jit; A compiled train and evaluation step. Pair them with their prerequisite order, not a search field or dropdown.

Keep the honest inventory small: 38 available lesson drafts, two integration projects and 35 planned topics. The complete catalog belongs on Course.

### 5. Give projects and offline reading visible artifacts

Show the regression experiment and classifier as two concrete pieces of work with a brief visual or code/output example. Use actual project data if a result is plotted. Present the existing EPUB/printable reader as one course edition with a small cover and direct actions, without inventing extra books or certification outcomes.

## Proposed page order

```text
Course promise + JAX learning map
[Start with tutor] [Explore course]
Compact npx / start-learning band + required course-workspace link

Choose your work
Models / Inference / Research / Systems → routes and careers

Inspect the curriculum
Small ordered set of real lessons + honest availability

Build the regression experiment or classifier
Concrete project previews

Read offline / Source / Maintainer / Feedback
```

Each section answers a different decision. This is a proposed revision, not implemented UI. Preserve the tutor goal, visible career entry, no required forms, local progress and canonical curriculum data. Reassess desktop/mobile rendering before claiming that a visual redesign is complete.
