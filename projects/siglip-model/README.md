# Train and compare a SigLIP model

Status: **planned capstone**. No new SigLIP training run, benchmark score, workspace or release is available yet. First complete the [cross-modal harness](../cross-modal-harness/README.md) and the objective preparation in [training methods](../training-methods/README.md).

## The question and comparison

Can we learn the same image–text task by asking whether each pair matches, rather than making captions compete through a softmax? Build SigLIP-style pairwise sigmoid training and compare it with the [CLIP project](../clip-model/README.md). Both use two encoders and normalized embeddings; the losses impose different supervision and scaling. Do not describe SigLIP as CLIP with a renamed activation, or assume it wins on every dataset.

Use the same source splits, towers, tokenizer, processors, embedding dimension and evaluation pool for the objective comparison. Compare equal data exposure and separately report measured compute. Tune each objective on development data with the same search budget; forcing identical learning rates is not necessarily fair. Count both towers when using an approximately 300M total budget. Tiny CPU oracle, real-data pilot and target-scale training are separate gates.

## Stage 1: define all pairs and their labels

Retain image/caption IDs, duplicate groups, source license and preprocessing identity. Construct a signed label matrix: \(z_{ij}=+1\) for a positive pair and \(z_{ij}=-1\) for a negative pair. Begin with a square batch of \(n\) unique pairs and diagonal positives. Then handle multiple captions and semantic duplicates with an explicit positive/ignore policy. Missing annotations are not proof of irrelevance.

**Change and diagnose:** introduce a second correct caption. If it remains labeled negative, the objective pushes a valid match apart. Inspect labels before diagnosing an optimizer problem. Cross-device candidate exchange must preserve IDs and masks, including padded or repeated examples.

## Stage 2: implement stable pairwise sigmoid loss

For unit-normalized embeddings \(u_i,v_j\), learned log-scale \(a\), and learned bias \(b\), use \(s_{ij}=\exp(a)u_i^\top v_j+b\). The baseline reduction is:

\[
L_{\mathrm{SigLIP}}=\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{n}\operatorname{softplus}(-z_{ij}s_{ij}).
\]

Use stable softplus or log-sigmoid. The reduction sums candidate losses for each anchor, then averages anchors. A mean over all \(n^2\) pairs divides this loss and its gradients by another \(n\); record the convention instead of silently changing the learning problem. Chunking must preserve the sum and the original anchor denominator.

At zero logits with two unique pairs, all four terms equal \(\log 2\), so this loss is \(2\log 2\). With larger batches, the number of negatives per anchor grows. For a constant-logit baseline with one positive and \(n-1\) negatives per anchor, setting the sigmoid to the empirical positive fraction gives \(b=-\log(n-1)\) when \(n>1\) and the similarity term is zero. This is an analytic initialization exercise, not a universal recipe for the complete model.

Compare the scalar and gradients against an independent host calculation for mixed labels, extreme logits, multiple batch sizes and nonzero bias. Probe zero-norm embeddings and malformed masks. A correct loss must be finite where the mathematical expression is finite.

## Stage 3: audit negatives and distributed computation

Implement dense all-pair computation first, then chunk candidate blocks and prove scalar/gradient agreement. If you sample negatives, name the sampling distribution and whether importance weighting estimates the full objective or intentionally defines another one. Log selected positive/negative counts and ignored pairs. Independent pairwise terms remove the softmax denominator; they do not remove the need to obtain the chosen negative candidates.

Test local anchors/global candidates with correct gradient flow into both towers, then compare against a single-device global-batch reference. Mask padding from both numerator and denominator. A stop-gradient gather or local-only negatives changes the algorithm unless deliberately specified and evaluated. Multi-device equivalence remains unverified until run on the named topology.

## Stage 4: pretrain, adapt and track a fair experiment

Phase A uses broad image–text pairs. Phase B continues on the chosen domain with broad-domain retention checks. Phase C fine-tunes on curated positives and audited hard negatives; compare frozen versus trainable towers or supported adapters. Keep parent checkpoint, objective/mask version, scale, bias, optimizer/schedule, random state and data cursor across transitions. Test exact resume inside a phase and declare any reset between phases.

Use a bounded comparison table: CLIP versus SigLIP, negative-pool sizes, bias initialization and duplicate-label policy. Log numerator/denominator separately, positive and negative loss contributions, learned scale/bias, gradient norms per tower, source exposure and actual throughput/memory. Preserve local records, then compare the same runs/artifacts in MLflow or W&B. Core correctness exercises require neither a hosted account nor a paid run.

## Stage 5: evaluate scores without overinterpreting them

Use the same fixed image-to-text and text-to-image pools, judgments and zero-shot classification protocol as CLIP. Report Recall@K with the positive-set convention, per-domain counts and query/group-aware uncertainty. Show failures and missing tasks. Use candidate real-data COCO/Flickr30k protocols only after pinning exact splits and checking training overlap.

A sigmoid score is not automatically a calibrated probability in deployment: the training negative prior and deployed candidate distribution can differ. Calibrate any match/abstention threshold on separate representative data and evaluate false positives under the declared prior. A shared scalar bias cannot change within-query rankings by itself. Measure ranking and threshold behavior separately.

## Figures learners must produce and explain

- **Loss versus logit:** horizontal axis is the signed match logit; separate positive and negative softplus curves show why the two labels want opposite movements. Label this as an analytic illustration and connect it to a finite-difference check.
- **Dense versus chunked gradient differences:** plot absolute error against chunk size with dtype and tolerance. Small errors qualify the implementation, not real-image accuracy.
- **Positive/negative score histograms:** share axes and report counts; overlap identifies ambiguous decisions. A sigmoid transformation bounds scores without proving calibration.
- **Quality versus pairs seen and measured cost:** compare CLIP/SigLIP retrieval on the same frozen pool. Include repeated seeds and resource observations; different raw loss magnitudes requires separate verification to establish the better model.

Store raw values, masks, configuration and checkpoint identity. No plots or measurements are claimed by this specification.

## Stage 6: precision, cold loading and retrieval handoff

Keep float32 reference logits/loss reductions where needed and explicitly qualify supported lower-precision tower computations. Compare weight-only and weight-plus-activation quantization on a separate calibration split. Track both retrieval degradation and threshold changes. Package processors, projections, weights, scale, bias and training semantics together; validate cross-framework embeddings/logits with layerwise error checks and downstream rankings.

Publish a model card, immutable Hub revision, complete consumer configuration and fixed cold-load canaries after the actual experiments. A SigLIP-style loss implementation is verified separately from compatibility with an official SigLIP checkpoint architecture. Verify pooling, positional embeddings, tokenizer and all tensor mappings before claiming that compatibility. Containerize the qualified runtime and carry model/processor/index identity into the [retrieval project](../retrieval-system/README.md).

## Completion evidence and teaching still to implement

Required outputs are a data/label manifest, independent loss/gradient oracle, dense/chunked parity, distributed equivalence if claimed, resumed training, controlled CLIP comparison, interpreted figures, precision checks, cold-load package and service qualification. Create starter, reference and adversarial stage checks before marking the project authored. SigLIP 2 and additional objectives are future extensions and must be named separately.

## Primary references

- [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343): objective, learned bias and experimental reference.
- [Official big_vision SigLIP trainer](https://github.com/google-research/big_vision/blob/main/big_vision/trainers/proj/image_text/siglip.py): inspect and pin the source revision for loss reduction and state behavior when implementing.
