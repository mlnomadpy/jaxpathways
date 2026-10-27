# A calmer course reader

The default page should teach the idea. Supporting material stays available through named, keyboard-accessible disclosures. This is a local preview for review, not a published release.

## Visual direction

Keep the existing identity: white `#ffffff`, violet paper `#f3eff9`, ink `#252139`, muted ink `#5f5873`, violet `#493193`, and green `#28604f`. Georgia carries explanations; Avenir Next carries headings and controls; monospace stays inside code. Keep the reading column left aligned. Disclosure rows use quiet rules and native expansion markers rather than another stack of panels.

## Reading hierarchy

| Visible while reading | Open when needed |
| --- | --- |
| Lesson purpose, core explanation, worked math | Prerequisites, runtime qualification and detailed review status |
| Essential code and its explanation | API reminders, launcher commands and complete assembled code |
| Figure, short measurement/simulation label, and how to read it | Figure data, environment versions, provenance and full logs |
| Exercise prompt and expected conceptual result | Starter scaffolds, hints, reference answers and evidence checklists |
| Assumptions and limitations that change the interpretation | Bibliography and repeated workflow instructions |

```text
Chapter title                         Available draft
One sentence: what you will understand
> Preparation and runtime

The idea
Explanation -> worked example -> figure -> what it means
> Figure data and execution details

Your exercise
The task
> Start this exercise
> Show a hint
> Reveal the reference solution

Carry forward
The key ideas
> Save your work
> References and further reading
```

Never put all lesson prose behind accordions. Never hide the difference between a measured result and a calculated scenario. Setup commands remain visible in the beginner setup lesson, where running them is the lesson itself. Disclosures work without JavaScript. Printing expands supporting material and preserves the existing separate solutions section; it restores the reader's previous state afterward.

Apply the shared figure, execution and command treatment to online lessons too. Native EPUB reading keeps its full narrative, since disclosure support varies between reading apps.
