# Landing page: tutor-first audit

2026-10-05. Scope: Home and the route from arrival to the first tutor lesson. This is a source and content audit; browser inspection was blocked by the browser tool's URL policy. No screenshot, Lighthouse or rendered mobile claims are made.

## Findings

1. **Primary goal missing from the hero.** “Start learning” opens the browser reader. The course's agent tutor is treated as an optional resource well below the fold, without its installation command. Visitors cannot immediately understand that their own agent can teach the course.
2. **Too many competing directions.** The transformation demo, work domains, two projects, study cards, roadmap and about section all compete with setup. A learner has to interpret the whole product before finding a next action.
3. **Repeated explanations instead of instructions.** “Small experiments. Ideas that stick”, “Start with what’s ready” and “Built around understanding JAX” repeat the promise. Their sections add scroll distance without resolving the install/start sequence.
4. **Installation does not equal a lesson.** The Skills CLI installs tutor instructions. The tutor also needs course files and an invocation. Home needs to show all three: get the course, add the skills, ask to start.
5. **Compatibility needs a concrete boundary.** Skill-enabled coding agents can read the local workspace. A normal browser chatbot cannot automatically access it or install an npx package. Provide a lesson-text fallback; do not promise universal automatic integration.
6. **Remote command currently points to unpublished work.** `git ls-remote origin` reports main at `ac8f09edeaa83391dc3772b3ca7ab761ab00451d`; the matching `origin/main` tree has no `skills/` directory. Do not promote `npx skills add mlnomadpy/jaxpathways` as working for the current tutors. The local command can use a downloadable course workspace containing the current sources.

## Revised hierarchy

Retain the existing lavender, purple, white and readable sans-serif identity. Make the real tutor setup the hero's visual: one compact numbered sequence, a readable command and copy feedback. Remove the transformation preview and marketing sections from Home; lessons retain their actual concept diagrams.

```text
Learn JAX with your AI tutor.    START THE COURSE
What the tutor helps you do     1 Download and open course workspace
Compatible coding agents       2 Copy / run npx skills add . ...
[Browse lessons instead]       3 Copy / say “Use start-learning ...”

WHAT YOU'LL LEARN
Foundation → training → specialization
Availability in one small inventory

BROWSER / EPUB / TUTOR HELP / CHATBOT LESSON TEXT
SOURCE / MAINTAINER / FEEDBACK
```

Mobile stacks the start sequence immediately after the short promise. Commands wrap within their panel; copy buttons remain at least 44px. Copy failure leaves selectable text and says what to do. Existing browser Resume remains available without hiding tutor setup.

## Delivery and verification

The downloadable workspace contains canonical curriculum, lesson sources, projects and tutor skills, plus the CPU environment file and learner CLI. It excludes Git history, local learner records and environment folders. `START-HERE.md` repeats the install/start instructions. The Skills CLI supports local paths and named skills: [official documentation](https://github.com/vercel-labs/skills).

Public GitHub installation remains a release task; no publishing is part of this change. Installation is user initiated and scoped to the downloaded project. Ordinary chatbot use is a manual lesson-text upload, without filesystem execution or automatic progress synchronization.

Verified after implementation:

- Home copy reduced from 639 to 273 words; seven section headings reduced to three.
- Local links and fragments resolve, IDs are unique, and inventory matches the generated curriculum. Home has no input fields or dropdowns.
- Copy actions preserve the exact command/prompt and handle a blocked clipboard with selectable-text instructions.
- Downloaded workspace extracts cleanly; its learner CLI lists 38 authored lessons and reads `welcome-01` without needing the repository checkout.
- Official Skills CLI discovery succeeds on the extracted folder. A project-scoped installation in that temporary folder successfully installs all four learner skills for Codex. No user's agent configuration was changed.
- `npm run build` and `npm run check` pass, including 29 existing tests and the new downloadable-workspace integrity checks.
- Live desktop/mobile rendering remains unverified because of the browser tool's URL-policy block.
