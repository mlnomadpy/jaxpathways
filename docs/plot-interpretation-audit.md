# Plot interpretation audit

This review covers all 50 authored lessons: 45 recorded numerical figures, five workflow diagrams, the two existing interactive illustrations, and seven existing conceptual cards. The 33 planned lessons have no authored plots to interpret; their requirements remain in `visual-learning-audit.md`.

The review compared the canonical lesson computations, saved numerical figure data, rendered image contact sheets, legends, scales, and surrounding explanations. Every authored figure now has a walkthrough of its axes or layout, concrete visible observations, an explanation of the mechanism, and a boundary on the conclusion. This is a source and artifact review, not independent expert certification or learner testing.

## Problems corrected

| Finding | Why it could mislead | Correction |
|---|---|---|
| Optimizer comparison described “progress” and “oscillation” without interpreting either | A learner could mistake Adam’s deep early dip for convergence, or assume its final ranking held throughout training | Explain the logarithmic scale, common initial loss, SGD’s two coordinate contraction factors, momentum’s rebounds, Adam’s loss at updates 11 and 19, and the ranking at update 50. State that this figure has no clipping or schedule. |
| Gradient-error plot displayed a large negative range on a symmetric-log axis | Maximum absolute error cannot be negative; empty negative ticks distracted from the observed rounding behavior | Use a logarithmic vertical axis, since every recorded error is positive. Explain the small-step plateau and why this quadratic does not exhibit the usual large-step truncation pattern. |
| Gradient-descent explanation treated a symmetric-log scale as a plain log scale | Zero loss appears on this plot, which is impossible on a pure log axis | Explain the linear region around zero and the logarithmic region for large values. Derive the three error multipliers. |
| Curvature caption overlooked the initial point | The point at update zero could be mistaken for a completed update | Identify the shared initial state and explain why the high-curvature coordinate reaches zero at update one. |
| Equal grouped bars were described as overlapping | The reader could search for hidden curves or misunderstand the chart layout | Describe adjacent equal-height bars in iterator recovery, serialization, and sharded gradients. Explain meaningful zero-height bars. |
| Attention explanation did not identify tied high-weight keys | A learner could infer that each query selected exactly one key | Explain why the third key ties with a different key in each query row; verify row sums and link weights to value mixing. |
| Transformer correction heatmap had nearly identical rows without explanation | It could look like a bug, ignored tokens, or a universal property of transformers | Explain constant-offset input rows, layer normalization, identical projected values, and the resulting repeated correction. Distinguish corrections from final outputs. |
| Probability fields and confusion matrices lacked enough reading guidance | Color intensity could be confused with accuracy, calibrated confidence, or identical units across panels | Identify feature axes, observed-label markers, probability scales, actual/predicted confusion axes, and the scope of the synthetic held-out results. |
| Timing plots used independent panel scales and a percentile line without sufficient interpretation | Learners could compare physical panel heights or read p95 as a second system or a maximum | Explain units, scientific notation, measurement boundaries, warm versus cold calls, sampling variation, and interpolation of the empirical percentile. Avoid fixed timing claims that become stale on rerun. |
| Quantization bars did not fully explain what changed between policies | Numerical simulation could be mistaken for native low-bit speed or memory evidence | Explain the reference, absolute-error units, rounding and accumulation policies, weight-only reconstruction, and task-specific limits of the precision ranking. |
| Interactive captions gave output values without explaining their meaning | Changing controls did not teach how to interpret the resulting changes | Add a slope explanation that changes sign with the selected input, and worked attention averages that change with the selected mask and last value. Typeset the live mathematical explanation with KaTeX. |
| Long explanations would render as a single HTML paragraph; notebook prose preceded the plotted output | The connection between image and interpretation was difficult to follow | Render paragraph breaks in the lesson and printable reader. Put notebook walkthroughs immediately after the figure cell. Include the same walkthrough in Markdown and EPUB. |

## Lesson-by-lesson review

The explanations are authored in each lesson’s `content.visual.reading` and `content.visual.connection`, not in generated companions. The inventory below identifies the main interpretation checked in each lesson.

| Lesson | Interpretation reviewed |
|---|---|
| [welcome-01](../phases/00-welcome/01-set-up-your-learning-workspace/lesson.json) | Workflow arrows versus measured time; installation versus verified sum. |
| [welcome-02](../phases/00-welcome/02-meet-your-arrays-and-devices/lesson.json) | Observation and feature axes; concrete row and column reductions. |
| [welcome-cpu](../phases/00-welcome/03-virtual-cpu-devices/lesson.json) | Categorical device ownership versus array contents or device performance. |
| [arrays-01](../phases/01-arrays/01-from-numpy-to-jax-numpy/lesson.json) | Separate column means, preserved spacing, centering versus scaling. |
| [arrays-02](../phases/01-arrays/02-shapes-broadcasting-and-dtypes/lesson.json) | Added bias versus final outputs; repeated columns versus repeated predictions. |
| [arrays-03](../phases/01-arrays/03-pure-functions-and-explicit-inputs/lesson.json) | Constant vertical gap, slope, and intercept under an explicit bias change. |
| [arrays-04](../phases/01-arrays/04-immutable-updates-and-indexing/lesson.json) | Independent immutable branches versus sequential updates. |
| [first-gradient](../phases/02-transforms/01-your-first-gradient/lesson.json) | Selected point, tangent slope, local approximation error, and negative extrapolation. |
| [transforms-02](../phases/02-transforms/02-losses-and-value-and-grad/lesson.json) | Loss value versus derivative sign; no meaning assigned to incidental crossings. |
| [transforms-03](../phases/02-transforms/03-batch-a-function-with-vmap/lesson.json) | One dot product per row; equal results from different inputs. |
| [transforms-04](../phases/02-transforms/04-compile-a-function-with-jit/lesson.json) | Equal loss bars verify numerical agreement, not runtime. |
| [transforms-05](../phases/02-transforms/05-tracing-static-arguments-and-recompilation/lesson.json) | Static branch selection, specialization reuse, and dynamic values. |
| [state-01](../phases/03-state/01-random-keys-without-surprises/lesson.json) | Overlaid replay draws; coordinate index versus time or probability density. |
| [state-02](../phases/03-state/02-pytrees-and-structured-parameters/lesson.json) | Signed parameters, gradient-derived changes, and the resulting prediction error. |
| [state-03](../phases/03-state/03-compiled-loops-with-lax-scan/lesson.json) | Post-update history versus initial carry; halving distance to the fixed point. |
| [state-04](../phases/03-state/04-branches-with-lax-cond/lesson.json) | Branch behavior, continuity, and the nondifferentiable corner. |
| [optimization-05](../phases/04-optimization/05-vectors-norms-and-projections/lesson.json) | Tip-to-tail vectors, residual displacement, orthogonality, and lengths. |
| [optimization-01](../phases/04-optimization/01-linear-algebra-and-loss-intuition/lesson.json) | Targets versus predictions; signed residuals versus squared loss. |
| [optimization-06](../phases/04-optimization/06-least-squares-rank-and-conditioning/lesson.json) | Unavoidable signed residuals and least-squares orthogonality. |
| [optimization-07](../phases/04-optimization/07-chain-rule-jacobians-and-directional-derivatives/lesson.json) | Output rows and input columns; local sensitivity and a directional product. |
| [optimization-02](../phases/04-optimization/02-gradient-checking-and-numerical-accuracy/lesson.json) | Logarithmic axes, rounding plateau, positive errors, and the quadratic exception. |
| [optimization-08](../phases/04-optimization/08-curvature-hessians-and-learning-rates/lesson.json) | Initial point, coordinate values versus loss, and curvature-dependent contraction. |
| [optimization-03](../phases/04-optimization/03-write-gradient-descent-yourself/lesson.json) | Symmetric-log scale including zero; stable, exact, and divergent steps. |
| [optimization-09](../phases/04-optimization/09-probability-likelihood-and-stable-losses/lesson.json) | Logits versus probabilities; observed labels determine the loss of confidence. |
| [optimization-10](../phases/04-optimization/10-regularization-and-validation/lesson.json) | Validation minimum versus curve crossing; prediction error excludes the penalty. |
| [optimization-11](../phases/04-optimization/11-minibatches-expectation-and-gradient-noise/lesson.json) | Exact variance of independent averages versus mean gradient or training loss. |
| [optimization-04](../phases/04-optimization/04-optimize-with-optax/lesson.json) | Genuinely overlapping trained and target lines; slope and intercept. |
| [optimization-12](../phases/04-optimization/12-adam-clipping-and-learning-rate-schedules/lesson.json) | Endpoint and transient optimizer rankings, deterministic rebounds, and log ratios. |
| [networks-01](../phases/05-networks/01-build-a-tiny-multilayer-perceptron/lesson.json) | XOR markers, nonlinear probability regions, and unlabeled grid predictions. |
| [networks-02](../phases/05-networks/02-model-and-state-with-flax-nnx/lesson.json) | Parallel graph/state ingredients, trainable scalars, and independent clone counters. |
| [networks-03](../phases/05-networks/03-a-compiled-train-and-evaluation-step/lesson.json) | Held-out labels versus background predictions and the underlying slanted boundary. |
| [networks-04](../phases/05-networks/04-train-a-cnn-on-a-small-dataset/lesson.json) | Pixel-value units versus confusion counts; diagonal and off-diagonal meaning. |
| [networks-05](../phases/05-networks/05-debug-unstable-learning/lesson.json) | Pre-update indexing, orders of magnitude, and instability with correct gradients. |
| [recovery-01](../phases/06-recovery/01-design-an-input-pipeline/lesson.json) | IDs versus numerical values; ordering, separators, and the partial final batch. |
| [recovery-02](../phases/06-recovery/02-load-batch-and-prefetch-with-grain/lesson.json) | Adjacent equal-height bars and exact next-batch identity after restoration. |
| [recovery-03](../phases/06-recovery/03-save-model-optimizer-and-random-state/lesson.json) | One completed snapshot of coupled state; next-transition verification. |
| [recovery-04](../phases/06-recovery/04-recover-data-position-and-resume/lesson.json) | Dataset identity and position must match the recovered training boundary. |
| [transformers-01](../phases/07-transformers/01-attention-from-small-pieces/lesson.json) | Query rows, key columns, normalized weights, tied matches, and value mixing. |
| [transformers-02](../phases/07-transformers/02-masking-tokenization-and-sequence-packing/lesson.json) | Future zeros, uniform allowed scores, and a concrete prefix average. |
| [transformers-03](../phases/07-transformers/03-a-transformer-block-and-mixed-precision/lesson.json) | Signed residual corrections and the cause of repeated rows for this input. |
| [transformers-04](../phases/07-transformers/04-train-checkpoint-and-generate/lesson.json) | Next-token columns, context positions, greedy choices, and probability limits. |
| [performance-01](../phases/08-performance/01-benchmark-asynchronous-work-correctly/lesson.json) | Independent time scales, first versus warm calls, synchronization, and variability. |
| [performance-02](../phases/08-performance/02-diagnose-recompilation-and-host-synchronization/lesson.json) | Trace-event counts versus execution or latency; zero-height bars are observed reuse. |
| [distributed-01](../phases/09-distributed/01-arrays-meshes-and-sharding/lesson.json) | Same global shape, different local shapes and ownership; no measured speedup. |
| [distributed-02](../phases/09-distributed/02-a-sharded-training-step/lesson.json) | Negative gradients, matching adjacent bars, and the resulting update direction. |
| [internals-01](../phases/14-internals/01-read-your-first-jaxpr/lesson.json) | Rows as computation stages, concrete intermediate values, and a final scalar reduction. |
| [deployment-01](../phases/15-deployment/01-keras-and-pytorch-bridges-to-explicit-jax/lesson.json) | Transposed weight coordinates and the corresponding multiplication contract. |
| [deployment-03](../phases/15-deployment/03-export-and-serve-a-trained-computation/lesson.json) | Output parity under a declared signature; negative scores are not probabilities. |
| [deployment-05](../phases/15-deployment/05-weight-activation-and-accumulation-precision/lesson.json) | Absolute output error, precision policy differences, and simulation boundaries. |
| [deployment-06](../phases/15-deployment/06-edge-ai-conversion-and-device-validation/lesson.json) | Warm request latency, empirical p95, full request boundaries, and CPU proxy scope. |

## Verification and boundaries

The numerical regression checks in `tests/plot-interpretation.test.mjs` verify the optimizer landmarks and changing rankings, gradient-error scale, zero-loss and curvature behavior, least-squares residuals, attention normalization and ties, repeated transformer corrections, token choices, bar parity, and the percentile calculated from the plotted samples. Interactive tests vary the input, mask, and last value and check the corresponding explanations and math rendering.

The delivery checks in `tests/lesson-visuals.test.mjs` verify paragraph preservation and notebook placement after the figure cell. EPUB checks verify each paragraph and its mathematical expressions. Every source change invalidates its previous execution hash; regeneration and CPU execution are required before the current evidence appears in the reader.

The rendered figure artifacts were inspected, including the corrected gradient-error scale. Live browser layout and narrow-screen review remain unverified in this session because browser access to the local preview was denied. DOM tests do not substitute for that review. Learner walkthroughs are still needed to establish whether the explanations resolve real learner confusion. No accelerator, mobile-device, or production-performance validation is claimed.

Completed verification for this revision:

- `npm run figures:build`: all 50 figure artifacts regenerated from the revised canonical sources.
- `npm run smoke:cpu`: all 50 scripts and all 50 notebook companions executed successfully on CPU.
- `npm run build`: site and downloadable course artifacts regenerated.
- `npm run check`: Astro diagnostics, lint, formatting, curriculum validation, all 51 tests, EPUB/bundle fidelity, and 13-page / 729-reference site checks passed.
- `npm run audit:visuals` and `npm run audit:content`: inventories regenerated; 50 authored and 33 planned entries remain correctly distinguished.
