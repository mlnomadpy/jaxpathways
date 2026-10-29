# Author review guide: 0.18.0

This candidate repairs content issues #74 through #91 on top of PR #71. Start with the [book overview](http://127.0.0.1:4210/book.html), [complete HTML book](http://127.0.0.1:4210/downloads/jax-foundations.html) or [EPUB](http://127.0.0.1:4210/downloads/jax-foundations.epub). Record feedback in [blank review notes](review-notes-0.18.0.md). The source, artifact hashes and completed checks are recorded in the [verification packet](evidence/course-review-0.18.0/verification.json).

The [18-issue repair matrix](content-repair-status-0.18.0.md) maps each finding to its implementation and evidence.

## Review the teaching, not just the output

A final editorial pass rewrites 12 advanced lessons: neural-network diagnosis, the sharded training step, precision/profiling, all three pretraining lessons, SFT, LoRA, DPO and the final three operations lessons. Read each as a learner: what problem are we solving, what should I predict, what did the run show, and what decision follows?

The MLP chapter now follows the first failing boundary through input corruption, saturation and an oversized update. Its log-scale plot makes the safe and unstable curves readable together. The sharding chapter explains why local counts `[4,4,1,1]` do not mean ten distinct rows. MLM and MIM explain why an improving training objective can still fail a changed-mask or baseline comparison. The operations chapters walk actual case IDs, weighted slice errors and consumer-visible release changes. The saved-model chapters distinguish objective mechanics from the next model handoff and explain what their small changes mean.

A useful review note names the first point where you cannot explain the result in your own words. Reproducing the printed output is not enough to accept the teaching. The 18 original issue repairs cover a wider set of lessons; this extra editorial pass does not claim a new sentence-by-sentence review of all 104.

## Review the correctness fixes first

- [Attention](http://127.0.0.1:4210/lesson-transformers-01.html): the zero-query worked example remains uniform. The exercise now supplies its own nonuniform inputs. Predict why paired permutation preserves the result and why value-only permutation need not change the zero-query example.
- [Values and GAE](http://127.0.0.1:4210/lesson-rl-05.html): value targets and GAE use a declared matching discount. Work the two-transition residual example, then read the reverse-scan diagram and explain its distinct bootstrap and trace masks. The separate diagram fixture declares its own discount.
- [Precision and profiling](http://127.0.0.1:4210/lesson-tpu-06.html): compare BF16 weight storage, INT8 codes/scales, expanded FP32 operands and output bytes. Explain which numbers are stored bytes versus runtime workspace, and why a returned dtype does not reveal internal partial-sum precision.
- [TPU readiness](http://127.0.0.1:4210/lesson-tpu-04.html): CPU arithmetic rehearsal is a valid result with TPU readiness false. The actual gated TPU preflight must fail on CPU or an unexpected device count. Provisioning instructions are not a new authorization to run paid hardware during this review.

The equation repairs also cover array reductions, purity, immutable updates, JIT/tracing, conditionals, NNX state, binary cross entropy, checkpoint state/publication, timing and approval identities. Read an equation together with its symbols, worked substitution and actual code.

## Follow the stronger practical routes

1. **Beginner route:** [array setup](http://127.0.0.1:4210/lesson-welcome-02.html) can continue directly to [arrays](http://127.0.0.1:4210/phase-guides/arrays.html) on CPU. Virtual devices and cloud setup are an optional later branch. Check that this is clear before a beginner sees Mesh or PartitionSpec.
2. **Debugging:** [neural-network diagnosis](http://127.0.0.1:4210/lesson-networks-05.html) introduces one failure at a time in data, hidden activations and update scale. Explain which observation localizes each cause and whether its repair also helps held-out loss. The NNX evaluation lesson now checks BatchNorm/Dropout state separately from parameters.
3. **Sharding and memory:** [2x2 update](http://127.0.0.1:4210/lesson-distributed-02.html) has parameter and momentum ownership, an independent unsharded oracle, unequal counts and an empty data shard. Read the local shape/byte table and [per-device HBM ledger](http://127.0.0.1:4210/lesson-tpu-03.html). Aggregate capacity alone is insufficient.
4. **Framework conversion:** [PyTorch to NNX](http://127.0.0.1:4210/lesson-deployment-08.html) links an optional small saved-decoder lab. It tests QKV/head layout, masks, positions, tied embeddings, nonuniform input gradients and reload. The two deliberate semantic faults must fail at the first divergent attention boundary.
5. **Representation transfer:** [MLM](http://127.0.0.1:4210/lesson-pretraining-01.html), [MIM](http://127.0.0.1:4210/lesson-pretraining-02.html) and [contrastive](http://127.0.0.1:4210/lesson-pretraining-03.html) now save and reload small encoders, then report changed masks, held-out groups or frozen probes against baselines. Some results deliberately fail the useful-baseline comparison. Check that the explanations teach from those failures instead of claiming model quality.
6. **Saved-model adaptation:** [SFT](http://127.0.0.1:4210/lesson-posttraining-01.html), [LoRA](http://127.0.0.1:4210/lesson-posttraining-02.html), [reward fitting](http://127.0.0.1:4210/lesson-posttraining-03.html) and [DPO](http://127.0.0.1:4210/lesson-posttraining-05.html) add guided transfer and identity checks. SFT/DPO require the full workspace and its pinned tokenizer/package. The LoRA reload must reject a same-shape wrong base. Ambiguous preference votes need their own interpretation. The earlier saved-decoder PPO quality gate remains failed.
7. **Engineering evidence:** [MLflow selection](http://127.0.0.1:4210/lesson-recovery-06.html) logs and reloads three candidates, including a memorizer that wins training error but loses held-out selection. Operations now connect predictions, local retrieval traces, behavioral rollback and identity-bound retirement. Ask what a consumer actually returns after each transition.
8. **Trace interpretation:** [actual event example](http://127.0.0.1:4210/lesson-performance-03.html) retains CPU event names, starts and durations. Explain why adding a parent and its children double-counts elapsed time, then predict the effect of changing the injected host wait. No device-kernel or collective interval is invented.

## Run the learning material

Use the [full course workspace](http://127.0.0.1:4210/downloads/jax-course-workspace.zip) for saved-model handoffs. Extract it, work from its `jaxpathways` root and install `requirements-cpu.txt` in a fresh environment. This includes the pinned tokenizer needed for SFT/DPO. The archive contains the already-public frozen foundations package and code; it does not include a new production model release. Individual reference scripts include solutions; learner-only downloads contain authored TODOs and are expected to stop until completed.

The optional framework lab is at `phases/15-deployment/08-pytorch-to-flax-numerical-parity/labs/transformer_bridge.py` inside that workspace. The lesson lists exact optional framework pins. The retained CPU trace capture helper and numeric excerpt are also included with the lesson source. No paid API is needed for the operations adapter.

## Read the book as a learner

Check math, code, tables and figures in the complete HTML, EPUB and the supplied print copies. Use the prerequisite links in the book: PPO requires the RL preparation, while DPO branches from reward learning. Verify that dense code sections begin with the small learner transition and make the full reference easy to recognize.

At phone width, zoom the diagrams and check captions and units. In print, check the ownership table, neural-network plots, reverse-GAE graph, pipeline diagram and actual-event table. Structural checks and selected raster inspection do not establish that every page is easy to learn from.

## Feedback and remaining limits

Use the blank notes or open a GitHub issue with the lesson ID, the explanation or output that confused you, your expected result and the command/runtime you used. Prefer a changed-input prediction over a general approval of the chapter.

There are still 104 lessons, 22 phases and ten Planned full model capstones. Maturity levels and independent human-review status are preserved. The AI review and CPU checks cover bounded teaching mechanics. Expert/novice acceptance, native EPUB and assistive-technology reading, larger licensed-data evaluations and new physical-target measurements remain separate work. This candidate is for review; it is not merged or deployed.
