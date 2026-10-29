# Author review: 0.18.1

[Open the complete book](http://127.0.0.1:4210/downloads/jax-foundations.html), [browse the chapters](http://127.0.0.1:4210/book.html), or [download the EPUB](http://127.0.0.1:4210/downloads/jax-foundations.epub). Record your feedback in [review notes](review-notes-0.18.1.md).

This pass extends the 12-lesson revision in 0.18.0 to the other **92 lessons**, and rewrites the learning progression in all **22 chapter introductions**. The combined book now has an editorial pass across all 104 authored lessons. The source code, exercises, formulas, plot computations and maturity labels are preserved. This describes editorial coverage, not independent human acceptance or a completed production-model curriculum.

## What to look for

Read a lesson in this order: the problem, the worked example, the figure, then the code and practice task. At the figure, pause and explain what one point, cell, bar or arrow means. Predict a changed input before opening an answer. If a section only repeats a previous one or a jump still needs an intermediate step, mark that exact point.

The revision emphasizes five questions: why does this concept matter, what happens in a small example, how does the code express it, what does the plotted result show, and what decision follows? Timing samples are kept separate from illustrative calculations and simulated queues. A new CPU capture can change a time without changing the lesson's mathematical conclusion.

## Chapter routes

| Chapter | Revised teaching focus | Suggested review task |
| --- | --- | --- |
| [Setup](http://127.0.0.1:4210/phase-guides/welcome.html) | Saved programs, interpreters, axis meaning and optional device branches | Explain why adding one to each entry increases row and column sums by different amounts. |
| [Arrays](http://127.0.0.1:4210/phase-guides/arrays.html) | Preprocessing, broadcasting, explicit inputs and immutable updates | Construct the pairwise residual matrix caused by a column-shaped target. |
| [Transformations](http://127.0.0.1:4210/phase-guides/transforms.html) | Local change, batch axes, value/gradient pairs, JIT and tracing | Explain the sign of a derivative at negative input and why zero slope is not a constant function. |
| [State](http://127.0.0.1:4210/phase-guides/state.html) | Keys, tree leaves, recurrence state and conditional contracts | Follow which key and carry belong to the next step, rather than identifying only the final value. |
| [Math and optimization](http://127.0.0.1:4210/phase-guides/optimization.html) | Geometry, derivatives, curvature, likelihood, validation and optimizer state | Derive a plotted point and distinguish gradient variance from gradient magnitude. |
| [Networks](http://127.0.0.1:4210/phase-guides/networks.html) | NNX ownership, train/eval state, CNN coordinates and fault localization | Read a confusion row using its true-class denominator; then explain the state preserved by evaluation. |
| [Recovery](http://127.0.0.1:4210/phase-guides/recovery.html) | Record identity, next-read position, complete state and publication | Describe exactly which batch follows a restored checkpoint and why its iterator must share the update boundary. |
| [Transformers](http://127.0.0.1:4210/phase-guides/transformers.html) | Attention rows, masks, next targets, residuals and decoding | Explain why uniform attention is a special case and test packed-document isolation. |
| [Performance](http://127.0.0.1:4210/phase-guides/performance.html) | Measurement boundaries, captures, nested events and memory tradeoffs | Read both axes before comparing panels; explain why a nested child interval is not extra elapsed time. |
| [Distribution](http://127.0.0.1:4210/phase-guides/distributed.html) | Ownership, sums/counts, communication and coordinated restore | Explain how unequal or empty partitions still represent one global objective. |
| [Scientific computing](http://127.0.0.1:4210/phase-guides/science.html) | Integration, trajectory axes, inverse problems and discretization | Explain why forty updates give forty-one saved states when the initial state is included. |
| [Probability](http://127.0.0.1:4210/phase-guides/probability.html) | Density units, posterior/predictive uncertainty and approximate inference | Distinguish uncertainty in the mean from the spread of a future observation. |
| [Reinforcement learning](http://127.0.0.1:4210/phase-guides/rl.html) | Reward, return, value, advantage and episode boundaries | Work the terminal and collector-cutoff targets, then keep the two GAE discount fixtures distinct. |
| [Kernels](http://127.0.0.1:4210/phase-guides/kernels.html) | Tiles, index maps, partial blocks and buffer lifetime | Draw the valid elements of the final tile and identify when a pipeline slot can be reused. |
| [Internals](http://127.0.0.1:4210/phase-guides/internals.html) | Typed programs, directional products, custom rules and compilation stages | Explain a JVP/VJP adjoint comparison and why one derivative direction is insufficient to validate a custom rule. |
| [Deployment](http://127.0.0.1:4210/phase-guides/deployment.html) | Model conversion, precision, exports, request boundaries and queues | Trace one raw request across preprocessing and conversion; separate measured service work from simulated waiting. |
| [Operations](http://127.0.0.1:4210/phase-guides/operations.html) | Worker events, recovery, selection, capacity and lifecycle decisions | Explain which event proves progress, then calculate worker demand, reserve and a hypothetical bill in matching units. |
| [Pretraining](http://127.0.0.1:4210/phase-guides/pretraining.html) | Retained 0.18.0 masking, pairing and transfer explanations | Explain why the learned reconstruction can lose to a visible-mean baseline. |
| [Post-training](http://127.0.0.1:4210/phase-guides/posttraining.html) | Reward likelihood and PPO now connect to the revised SFT/LoRA/DPO route | Separate preference probability, reward score and policy probability; explain the preserved failed quality gate. |
| [TPU setup](http://127.0.0.1:4210/phase-guides/tpu.html) | Per-device capacity, requested target and synchronized arithmetic | Explain why a correct one-device CPU sum does not establish four-device TPU readiness. |
| [TPU workflows](http://127.0.0.1:4210/phase-guides/tpu-workflows.html) | Launch, warmup, state continuation and multi-host ownership | Separate topology discovery from a checked global update and its recovery evidence. |
| [TPU systems](http://127.0.0.1:4210/phase-guides/tpu-systems.html) | Memory ledgers, theoretical ceilings, numerical error and trace interpretation | Explain which conclusion comes from bytes, which from error, and which needs a target trace. |

## Evidence and feedback

The [editorial coverage record](evidence/course-review-0.18.1/editorial-coverage.json) identifies each changed lesson and the preserved executable fields. The [verification packet](evidence/course-review-0.18.1/verification.json) records the final source and export identities. Older review packets and failed model-quality outcomes remain historical evidence.

Feedback can name a lesson ID, quote the confusing sentence, describe your interpretation and give a changed-input prediction. [Open a GitHub issue](https://github.com/mlnomadpy/jaxpathways/issues/new) if you would like to contribute an explanation or a real-world example.

The candidate is not merged or deployed. Expert and novice acceptance, native/assistive reader review, new physical-target measurements and the ten Planned full model capstones remain separate work. Earlier failed quality floors remain failed.
