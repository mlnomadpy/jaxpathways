# Content repair candidate: 0.18.0

All 18 audited issues have an implemented repair in this candidate. This is an implementation and local verification record, not independent learner acceptance, a production-model qualification or a request to close the issues before review. Original audit findings remain preserved.

| Issue | Priority | Area | Implemented repair | Evidence |
| --- | --- | --- | --- | --- |
| [#74](https://github.com/mlnomadpy/jaxpathways/issues/74) | P1 | Equations and claims | Replaced mismatched equations, defined symbols, added substitutions and bounded replay/timing claims. | Numeric contract tests; current lesson receipts. |
| [#75](https://github.com/mlnomadpy/jaxpathways/issues/75) | P1 | Attention exercise | Supplied nonuniform inputs; verified paired and value-only permutations and the uniform exception. | Independent NumPy controls; script and notebook. |
| [#76](https://github.com/mlnomadpy/jaxpathways/issues/76) | P1 | Value targets and GAE | Aligned discount, corrected the terminal/reset answer and explained separate masks. | Two-transition arithmetic; current RL receipt. |
| [#77](https://github.com/mlnomadpy/jaxpathways/issues/77) | P1 | TPU readiness | Separated arithmetic, backend, topology and source readiness; pinned named launch/cleanup scope. | 10 TPU contract checks; CPU fails closed for target readiness. |
| [#78](https://github.com/mlnomadpy/jaxpathways/issues/78) | P1 | Precision bytes | Separated BF16 storage, INT8 codes/scales and expanded FP32 operands. | Executed shape/dtype/byte assertions. |
| [#79](https://github.com/mlnomadpy/jaxpathways/issues/79) | P2 | Exercise evidence | Added independent gradients, three logged MLflow candidates and fresh-process tamper rejection. | Fresh scripts and notebooks; artifact reload assertions. |
| [#80](https://github.com/mlnomadpy/jaxpathways/issues/80) | P2 | Neural-network diagnosis | Used an actual MLP for one-factor faults; checked BatchNorm/Dropout train/eval state. | Changed-input and held-out controls. |
| [#81](https://github.com/mlnomadpy/jaxpathways/issues/81) | P2 | Profiling interpretation | Bounded the workload and retained actual CPU events with a wait intervention. | CPU trace excerpt, duration table and double-counting control. |
| [#82](https://github.com/mlnomadpy/jaxpathways/issues/82) | P2 | Sharding and memory | Added 2x2 parameter/momentum ownership and per-device HBM accounting. | Independent unsharded oracle; uneven and empty shards. |
| [#83](https://github.com/mlnomadpy/jaxpathways/issues/83) | P2 | Framework parity | Added an optional saved PyTorch-to-NNX decoder bridge. | Intermediate values, nonuniform VJP, reload and deliberate semantic faults. |
| [#84](https://github.com/mlnomadpy/jaxpathways/issues/84) | P2 | Pretraining transfer | Saved/reloaded MLM, MIM and contrastive encoders with held-out probes and baselines. | Three executable handoffs; unfavorable baseline results retained. |
| [#85](https://github.com/mlnomadpy/jaxpathways/issues/85) | P2 | Posttraining handoffs | Added pinned saved-decoder SFT/DPO and ambiguous preference analysis. | Fresh executions; held-out groups; prior PPO quality failure preserved. |
| [#86](https://github.com/mlnomadpy/jaxpathways/issues/86) | P2 | LoRA identity | Trained bounded adapter factors and enforced a base-bound manifest on reload. | Fresh-process round trip; wrong base/rank/alpha/shape controls. |
| [#87](https://github.com/mlnomadpy/jaxpathways/issues/87) | P2 | Operational evidence | Derived gates from reloaded predictions and deterministic retrieval traces. | Distinct candidates, case identities and missing/duplicate controls. |
| [#88](https://github.com/mlnomadpy/jaxpathways/issues/88) | P2 | Rollback and retirement | Demonstrated changed consumer behavior and identity-bound approval, rollback and tombstones. | Artifact/cache tampering, exact expiry and active/fallback guards. |
| [#89](https://github.com/mlnomadpy/jaxpathways/issues/89) | P2 | Scaffolding | Put short learner transitions before complete operations/recovery references. | Companion scripts and notebooks execute in order. |
| [#90](https://github.com/mlnomadpy/jaxpathways/issues/90) | P2 | Mechanism figures | Added tracing, state identity, GAE, pipeline and shard ownership diagrams; separated incompatible axes. | All 104 visual source bindings; table renderer tests; sampled visual review. |
| [#91](https://github.com/mlnomadpy/jaxpathways/issues/91) | P2 | Prerequisite routes | Made CPU setup primary, cloud optional and PPO/DPO branches explicit across readers. | Graph tests, HTML/EPUB links and deployment checks. |

Start with [the review guide](review-guide-0.18.0.md), then use [blank review notes](review-notes-0.18.0.md). Exact checks, source identities and artifact hashes are in [verification.json](evidence/course-review-0.18.0/verification.json). Human teaching review, native/assistive reading, physical TPU measurements and the ten Planned full model capstones remain separate requirements. No merge or deployment was performed.
