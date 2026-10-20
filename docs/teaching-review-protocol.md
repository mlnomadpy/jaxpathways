# Human teaching review

Execution checks establish that a reference runs. Human review asks whether the claims are sound and a learner can apply them. Neither code execution nor an internal agent review counts as an independent human review.

## Novice pilot

Use the current beginner ZIP on a fresh account. Record the course revision, OS, Python, terminal and any existing Python knowledge. Ask the participant to think aloud. Do not coach through a failed instruction before recording where they hesitated.

1. Download, extract and run `first_experiment.py` without entering the repository.
2. Predict the sum after changing the range from four to six values. Make the change and explain why the assertion must change too.
3. Complete arrays-02. Before seeing its answer, explain which axis represents an observation and predict one changed shape.
4. Explain the axes, units, comparison and limitation of its figure. Give a different input and ask which visual feature would change.
5. Reopen the lesson, recover the saved place, find the next lesson and export a learning backup.

Record assistance, failed commands, time spent recovering, and the participant's actual explanation. Quiz completion alone is insufficient. Create focused issues for observed failures, then repeat the failed task with changed inputs after repair. Obtain consent before publishing quotations or identifying information.

## Expert review

Start with probability, RL, TPU, precision and deployment, plus their career capability maps. Review equations and assumptions, data leakage, numerical references, gradient behavior, target limitations, figure explanations and whether the project evidence supports each advertised skill. Ask for a counterexample or changed-condition test, not an endorsement based on the outline.

For each concern, retain the exact source revision, location, reproducer or calculation, proposed correction and disposition. A reviewer must be independent of the implementation being reviewed. An AI agent can supply an internal review but cannot stand in for an expert or novice participant.

## Evidence registry

`curriculum/teaching-reviews.json` starts empty. A completed entry contains `lessonId`, `contentHash` (canonical content SHA-256 as reported by the content audit), `kind` (`expert` or `novice`), `reviewer`, `date`, `evidence` (a retained public report path), `outcome` (`accepted` or `changes-requested`) and `changedCondition` (the task and observed explanation). Store consented public summaries only; retain private raw participant records outside this repository.

The audit counts accepted, source-matching expert reviews separately from novice pilots. A later content edit makes the old review historical until the reviewer checks the revision. Do not insert an accepted record for a planned session. The initial empty registry reports zero independently reviewed lessons.

The [open review issue](https://github.com/mlnomadpy/jaxpathways/issues/57) remains open until actual participants and expert reports satisfy its acceptance criteria.
