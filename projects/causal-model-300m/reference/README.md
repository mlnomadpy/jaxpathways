# C02 review evidence

This packet records a bounded CPU experiment with 33,088 model parameters.
All six branches failed their precommitted question-answering quality floors.
The full approximately-300M capstone remains Planned.

`evidence-index.json` binds the current source files and each public evidence
file by SHA-256. The retained source handoff and verification logs distinguish
the accepted training run from later checker, adapter and baseline-guard repairs.
No training weights or fitted tokenizer payload are distributed here.

The figures show observed minibatch training losses, held-out generated-answer
scores and normalized-byte likelihoods. Their captions in the project guide
explain axes, denominators, failed floors and comparison limits. The original
figure index binds the measured source reports used to render them. Later
baseline guard reports retain separate evaluator identities.

The public QA projections preserve every generated prediction, output token ID,
finish reason, question ID and observed score. They omit prompts, passage text,
gold answers and prompt token IDs. Use the pinned preparation/evaluation commands
to reconstruct the complete inputs from their original sources; their full raw
report hashes remain in `retention.json`. Keep the dataset attribution and terms
in `data/RIGHTS.md`. Native task definitions retain their accompanying EleutherAI
license and notice.

The native harness report is an eight-record-per-task execution smoke test.
Local service checks cover the documented cooperative cancellation and bounded
admission behavior. Neither establishes broad benchmark performance, forced
interruption of native JAX operations, production service readiness, target-scale
training, human acceptance or a trained-model release.
