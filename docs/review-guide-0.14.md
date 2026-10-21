# Author review: edition 0.14.0

This edition carries forward the 104-lesson review series and adds a causal
CPU lifecycle, stronger embedding exercise grading and a complete local recovery
protocol. It is a review candidate. All ten full model/retrieval capstones remain
Planned; thirteen phases remain In Development. Automated checks and internal
numerical review do not establish learner understanding or production readiness.

## Suggested review order

1. Read the [frozen book preview](http://127.0.0.1:4196/book.html) and attempt a learner
   scaffold before opening its worked reference. The lesson bodies are unchanged
   from 0.13; the earlier review guides describe that material's deeper changes.
2. Read the [causal pilot guide](../projects/causal-model-300m/README.md). Predict
   what lower training loss tells you before inspecting the QA bars. Check that
   the figure captions make the axes, phase-specific denominators, matched
   comparisons and failed floors understandable. Follow the learner contract
   through target ownership, generation, recovery and export.
3. Attempt the [recovery project](../projects/recovery-audit/README.md). Stages
   1–3 retain compatible single-file exercises. The integrated stages additionally
   test fresh-process restoration, partial writes and competing writers on a
   local POSIX filesystem. State what these tests cannot tell you about power
   loss, shared storage or multiple hosts.
4. Inspect the [embedding pilot](../projects/embedding-model-300m/README.md).
   Its new numerical conventions explain the tied MLM head and gradient path.
   The amended checker rejects constant output and disconnected gradients while
   accepting a harmless equivalent implementation. Historical training receipts
   remain separate from the fresh current-checker execution.
5. Check Projects, Content status, pathways and careers. The two bounded pilots
   should be discoverable without suggesting that their full 300M specifications
   are complete. Use the [itemized material audit](learning-material-audit-2026-10-21.md)
   to find contribution work throughout the rest of the course.

## How to interpret the measurements

The causal pilot has 33,088 model parameters. It uses frozen WikiText-2/SQuAD
records, train-only tokenization, matched continuation branches and three SFT
branches with explicit parent/reset rules. All six branches scored zero exact
match and token F1 below 0.002 on the custom 128-question final protocol. The
precommitted floors were 0.05 and 0.10. Lower likelihood loss did not make this
recipe useful for answering questions.

Pythia-14M is a contextual pretrained comparison, with a different tokenizer,
training history and possible evaluation overlap. The byte-rate plot normalizes
the output denominator; it does not equalize those histories. The native harness
run uses eight records per task, sufficient for an execution smoke test only.
Neither pilot establishes broad benchmark quality, target training or a trained
model release.

The causal source archive contains the full workflow and source-bound public
measurements. Public answer projections preserve generated outputs and scores
but omit corpus passages, prompts and gold answers. Reproduction commands fetch
the pinned original inputs. Checkpoints, fitted tokenizer payloads and pretrained
weights remain outside this download.

## Feedback to record

Use [the blank review notes](review-notes-0.14.md). Record the section/task,
your prediction, command and environment, observed result, confusing step and
requested correction. Distinguish a teaching gap from an incorrect numerical
result, a platform limit or a task-quality failure. A failed recipe can be useful
teaching material if its evidence and rejection decision are clear.

Image/audio lifecycle repairs continue separately and are excluded from this
edition's source integration. Full approximately-300M training, physical targets,
independent novice/specialist and career calibration, native EPUB readers and
actual assistive-technology sessions remain open. Current local and hosted checks
are recorded separately in the edition verification packet; do not infer a
hosted, human or hardware pass from this guide.
