# Book repair record, 20 October 2026

Edition: **0.7.0**. This follows the [book audit](book-audit-2026-10-20.md) at revision `225f724`. The historical audit and inventory are preserved. [Current counts](course-status.md) are generated from canonical registries.

Three Luna agents implemented separate project, TPU and book slices. The parent reviewed and integrated the work, including additional corrections found during full execution. Source structure, public fixtures, observed CPU execution, physical hardware qualification and independent learner review remain different forms of evidence.

## Findings and changes

| Audit finding | Implemented repair | Boundary |
| --- | --- | --- |
| F01: cloned projects | Nine distinct four-stage projects now check environment identity, masked moments, transform equivalence, full-state recovery, compilation/timing, response/preference losses, runtime evidence, worker receipts and memory budgets. Default learner commands fail untouched starters; reference commands explicitly select the answer key. | These bounded projects do not qualify the entire phase or professional competence. |
| F02: generic TPU launch claims | CPU-only reference classification, reviewed target gate, separate discovery and multi-host collective-update example, independent global-gradient check, worker receipts and retrieval commands. The reader no longer suggests that setting JAX_PLATFORMS adapts every script. | Multi-host target execution still needs a physical TPU receipt. |
| F03: calculated receipt | Analytic visual kind, explicit no-device-result caption and reproducible calculation provenance for hypothetical chip sums. | A CPU calculation is not a TPU measurement. |
| F04: checkpoint contracts | Missing, incompatible, corrupted and already-completed resumes have explicit behavior. Tampered-momentum practice teaches integrity rejection. | Local publication is not a multi-host durable checkpoint guarantee. |
| F05: precision claims | Explicit stored/operand/result types, dequantized FP32 reference, zero-scale handling, accurate error descriptions and scale-byte limits. | No native low-bit target speedup is claimed. |
| F06: lost traces | The actual worked profile is retained at an inspectable path; the CPU utility and lesson checks verify trace files and synchronization boundaries. | Named target traces and hardware bottleneck interpretation remain qualification work. |
| F07: overloaded routes | CPU classifier route omits mandatory TPU, language and RL phases; eight CPU-capable phases no longer require TPU setup. Supervised adaptation no longer inherits the whole RL phase; learned-reward PPO explicitly requires RL preparation. Optional engineering preparation is resolved and displayed. | Whole-phase routes remain coarser than personalized lesson-level routes. |
| F08: offline figures | Standalone HTML embeds primary/mechanism/appendix images and math fonts; EPUB packages figures. Filesystem access lives in server helpers, outside browser bundles. | Native browser and saved-file reading still need manual visual review. |
| F09: CI timeout/publication | Separate site, lesson, four project shards and logger jobs; larger bounded budgets, cached dependencies and uploaded receipts. Pages depends on the same-commit verification workflow. | Local checks do not prove that a future hosted run will pass. No remote run was triggered here. |
| F10: disconnected book | Phase openings, milestones, project handoffs, route/career links and project-structure/experiment-workshop appendices are included. | Project workspaces remain separate runnable bundles. |
| F11: edition/EPUB parity | Current release date/version, chapter titles, richer image alternatives, MathML in appendices and correct canonical online links. | Native EPUB reader, assistive technology and print pagination need manual review. |
| F12: answer leakage/noisy code | Web reference answers, assembled code and writing scaffolds are disclosed deliberately. EPUB solutions move to a separate appendix. Removed 6,626 mechanical syntax-restatement comments across 97 lessons while preserving explanatory comments. | This is an editorial improvement, not measured learner comprehension. |
| F13: disconnected model phases | Added the executable embedding-lineage pilot: 70,765-parameter bidirectional encoder, train-only tokenizer, MLM, two-view contrastive continuation, curated query-positive adaptation, chapter-held-out retrieval, source/checkpoint lineage and fresh-process export. | The measured pilot is weak; planned 300M and other target-scale model releases are not relabeled complete. |
| F14: career evidence | Every advertised skill has lesson, project-stage, artifact, review criterion and required/optional scope. Training defaults to the CPU model route. NumPyro/Diffrax extensions are identified honestly. | Independent portfolio review and employment readiness are not established. |
| F15: memory/roofline | Separate GiB and FLOP/byte panels; explicit weight/master/moment/gradient byte ledger and named exclusions; theoretical limits distinguished from actual bottlenecks. | Real memory use includes additional activation/workspace/communication costs. |
| F16: stale status | Generated registry counts, revised current guidance, historical snapshot labels and release hashing of all registered project teaching code. Mutable run/evidence outputs are kept outside source. | Historical audits intentionally retain the counts they inspected. |

## Actual model pilot

The [embedding project](../projects/embedding-lineage/README.md) uses pinned Project Gutenberg text with its source header and license retained. Chapter groups are disjoint across train/validation/test. Curated query judgments are educational labels, not a benchmark. Five test queries support only a tiny within-novel pilot.

The [recorded result](../projects/embedding-lineage/outputs/pilot-results.json) retains a failed quality hypothesis: dense recall@1 **0.0**, MRR about **0.021**, versus BM25 recall@1 **0.4**, MRR about **0.498**. Falling training losses do not establish useful representations. The [separate-stage curves](../projects/embedding-lineage/outputs/loss-curves.svg) use each objective's own units. The reference checks preserve the weak result rather than turning a passing export check into a quality claim.

The pilot checks independent loss calculations, finite gradients, step counts, source/data/tokenizer identities, actual predecessor checkpoints, changed-loader/config/weight rejection and two fresh-process reloads. Checkpoints are objective handoffs with deliberate optimizer resets. They are not exact mid-stage recovery snapshots; that separate contract is practiced in the recovery projects.

## Verification

- All 103 lesson scripts and notebook companions executed successfully after the final repairs. The first full run exposed stale TPU exercises; 99 passed source-bound records were retained after checking code-cell hashes, and the four affected/final TPU lessons were rerun.
- All primary figures were regenerated, including explicit analytic captions. One transient concurrent file-read failure was rerun successfully. Current visual hashes match their canonical content.
- All 30 registered reference project suites passed. All nine repaired project starters failed untouched, and their retained-evidence checks wrote outside source. The new embedding starter also failed as intended.
- Actual CSV, TensorBoard, offline W&B and local SQLite MLflow histories and model artifacts passed readback checks. No tracker uploads were performed.
- Independent math, TPU launcher and precision utility checks passed. Six complete TPU lesson scripts and the dedicated TPU numerical checks passed on CPU.
- Final `npm run check` passed: Astro/types/lint/format, curriculum validation, 102 automated tests, EPUB/HTML/workspace fidelity, 195 generated pages, 9,781 local references, 22 phase guides and 190 canonical sitemap URLs.
- `npm run check:deployment` passed for `/jaxpathways`, including the existing browser-DOM migration checks, and restored the normal root preview build afterward. These DOM checks are not a native visual browser review.

## Remaining evidence work

Physical TPU/multi-host execution, native low-bit runtime measurements, edge-device qualification, real-traffic service operation, independent expert review and novice learner pilots remain unverified. The ten target-scale lifecycle specifications remain explicitly planned. The new small model pilot supplies a real executable handoff pattern, not a trained 300M release or an MTEB result.

Follow the audit's phase plan to extend real datasets, architectures and target-runtime evidence after these corrected bounded examples. Manual browser, EPUB, screen-reader and print checks remain necessary before making claims about those reading environments.
