# Review the staged embedding workflow

Edition 0.13.0 is an author review candidate. It adds a separately scoped
SciFact CPU pilot to the existing embedding-capstone plan and improves the
lesson stage controls. The full approximately-300M capstone remains Planned;
the bounded pilot remains In Development because its retrieval quality gates
failed. The other nine full capstones and thirteen developing course phases
retain their open requirements.

Use the [embedding plan and pilot](http://127.0.0.1:4195/project-plans/embedding-model-300m.html#cpu-pilot),
[book](http://127.0.0.1:4195/book.html) and
[Projects](http://127.0.0.1:4195/projects.html#alignment-retrieval) in the frozen
local preview. The edition-specific verification record distinguishes actual
checks from pending hosted, target and human review.

## Suggested review route

| Stop | Inspect | Try to explain or diagnose |
| --- | --- | --- |
| Data and supervision | Project README, frozen recipe, data notices and preparation code | Why are indirect duplicate components quarantined before splitting? Which corpus text was visible to the tokenizer, which documents were actually consumed, and which final groups were excluded? Weak title/abstract pairs and human evidence labels have different provenance. |
| Encoder and objectives | Learner scaffold, model, objective code and independent checker | Change a future token and a padding token. Why should a bidirectional state respond to the former but remain isolated from the latter? Distinguish selected-token MLM denominators, contrastive candidates and multiple relevant documents. |
| Training figures | Four objective panels, raw trace and teaching manifest | Read each panel in its own units and phase-local update scale. Explain the single-candidate tail batch's zero contrastive loss without interpreting it as good retrieval. The historical plot remains retained as an original artifact. |
| Retrieval and failed gates | Seven-system table, full rankings, shared qrels and metric reconstruction | Why do only 188 of the 300 inventoried queries enter metric denominators? Can a low objective value establish good rankings? Compare the phase-3 pilot with BM25 and the pinned pretrained baseline without inferring a compute-matched comparison. |
| State and consumer | Checkpoints, resume checks, package implementation and measured card | Which parameters, moments, PRNG keys, sample order and cursor must agree on the next update? Which altered bytes or identities must be rejected? The public evidence ZIP does not contain a loadable trained model; the workflow generates a local package. |
| Access and navigation | Pilot page, download, status directory and a lesson | Find the source ZIP and evidence without opening a dropdown. Attempt the starter before the reference. Follow previous/next activities with a keyboard, move into the next lesson and return to the saved reading position. Check the mobile layout and full-size figures. |

## Interpret the evidence narrowly

The full current CPU checker executes the actual staged pilot and independently
reconstructs attention, losses, gradients, recovery and retrieval results.
Separate extraction checks establish that the source ZIP contains its required
files and runs from an unrelated working directory. These do not establish
learner success, cross-platform equivalence or target performance.

The retained final8 run used a small two-layer encoder, eight masked-token
updates, two matched eight-update continuation branches and one 71-update pass
over human-evidence fit queries. Its phase-3 Recall@10 is about 0.0213 and
nDCG@10 about 0.0093, below the precommitted 0.30 and 0.10 floors. The fixed
evaluation population was repeatedly inspected for engineering validation;
it is not a blind benchmark or an MTEB result.

Original final8 receipts, figures and the later attention-checker amendment
remain historical records. New plotting and execution records identify their
own source bytes. Do not treat an old passing receipt as execution of later
code. The numeric public packet intentionally omits trained weights,
checkpoint arrays, tokenizer vocabulary and the pretrained baseline snapshot.

Native EPUB readers, actual assistive-technology use, independent novice and
specialist review, broader benchmark quality, full-scale training, target
qualification, service operations and public trained-model release remain
open. The book compatibility matrix records its separate tested scopes.

Record findings in [the blank review notes](review-notes-0.13.md). Include the
source or artifact identity, prediction, observed result, explanation gap and
requested correction. A successful reference run alone cannot demonstrate
that the learner can diagnose a changed case.
