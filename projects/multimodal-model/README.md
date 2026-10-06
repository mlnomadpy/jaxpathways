# Capstone: combine trained components into a grounded multimodal model

**Status: planned specification.** Apply the [shared lifecycle](../../docs/model-lifecycle-capstones.md) and [tooling](../../docs/model-lifecycle-tooling.md). This is a grounded **generation** project. The [embedding capstone's cross-modal branch](../embedding-model-300m/MODALITIES.md) separately covers image–text/audio–text retrieval.

Outcome: connect a qualified image or audio encoder to the learner's causal decoder, align their representations, train grounded responses and release a complete media-to-text system. Begin with image-to-text; add audio only after its encoder and data contracts are qualified. Video and arbitrary combinations remain later work.

## Components and data

Reuse the [causal model](../causal-model-300m/README.md), [image model](../image-model-300m/README.md) and, for the audio branch, [audio model](../audio-model-300m/README.md). Record checkpoint versions, processors, normalization and feature extraction layers. A projection/resampler/cross-attention connector is a new trainable component with an explicit tensor/token contract.

The decoder may be around 300M, but publish total, frozen and trainable counts across all towers and the connector. Freeze one component set for the first experiment. Training a small connector is not training the entire combined model from scratch.

Curate image/audio plus caption/question/answer pairs with source/entity/recording identity, answer provenance and permitted use. Split before deriving captions, crops, questions or teacher labels. Inspect whether questions can be answered without the media. A pilot instruction mixture can compare 60% caption/description, 30% grounded question answering and 10% unanswerable/counterfactual cases by examples; also report response-token and media-token exposure. These proportions are hypotheses, not a selected dataset recipe.

## Phases

| Phase | Objective / work | Checkpoint and checks |
| --- | --- | --- |
| 0: component qualification | Reproduce standalone decoder and encoder canaries; specify selected features and media-token layout | Immutable component manifest, shape/value checks and modality processing tests |
| 1: connector alignment | Initially freeze encoder/decoder and train the connector to support conditioned next-token prediction on paired media/text | Connector checkpoint; frozen-leaf checks, correct/absent/shuffled-media comparisons |
| 2: grounded instruction tuning | Train response-only generation with a fixed media/chat template; compare connector-only, LoRA and controlled unfreezing | Joint/adapter checkpoint, shifted response masks, media-token attention boundaries and language-retention results |
| 3: optional preference adaptation | Use grounded preference pairs with auditable criteria; retain image/audio context and response masks | Separate preference checkpoint with external grounding/length/retention checks; preference fit alone is insufficient |
| 4: full evaluation and release | Qualify the combined processor/encoder/connector/decoder and supported precision/service paths | Complete serialized pipeline, task/inference report and immutable Hub download |

An optional contrastive alignment auxiliary can be compared against the next-token connector objective. If used, define pair identities, temperature and loss weighting; it does not replace the generation objective. The final task is not masked reconstruction. For each stage record parent components, optimizer/schedule decisions and actual training state, including media augmentation and data-position state.

## Evaluation and monitoring

Freeze an attainable grounded-task panel and the final holdout before selecting recipes. Choose suitable captioning/visual-question-answering/OCR or audio-question tasks, with exact versions, answer extraction and decoding settings. [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) is a candidate adapter framework; confirm its selected model/tasks rather than claiming universal custom-JAX support.

Report task scores, per-domain/length/resolution/duration slices, unanswerable cases, hallucinated details and retained text-only ability. Compare the decoder alone, correct media, absent media and swapped media. If swapping the media barely affects a visually grounded task, investigate text shortcuts or an ignored connector. Preserve benchmark overlap and teacher-label disclosures. Retrieval metrics alone cannot qualify grounded generation.

Monitor separate tower/connector gradient and update norms, frozen-leaf integrity, media/text token ratios, truncation, response-mask coverage, objective components, data exposure, quality/retention and per-stage runtime/memory. With multiple modalities, retain per-modality metrics so one dominant data source cannot hide a failing branch.

Use W&B multimodal tables or equivalent local/MLflow artifacts for a fixed permitted case panel. Required figures show media-to-token shapes, connector feature statistics, correct/swapped/no-media answers, stage-by-stage grounding/retention, precision errors by component and preprocessing/encoder/prefill/decode timing. Explain observed outputs rather than treating attention images as a complete explanation.

## Complete system release

Package or pin the encoder, connector, causal decoder, tokenizer, image/audio processor, media placeholders, chat template, context/media limits, generation config and supported dtype policies. Test native versus converted intermediate features and final logits/generation; quantization can affect each component differently. Advertise a runtime only after verifying the whole architecture and preprocessing path.

Follow the [Hub publication procedure](../embedding-model-300m/RELEASE.md) with media canaries and a fresh complete-pipeline download. Service tests cover malformed/oversized media, missing modality, bounded media/token budgets, concurrent sessions, cancellation and resource exhaustion. Rollback restores all compatible component revisions and templates together.

The final reviewer should be able to trace a raw media request through the released system, explain the grounding evidence, identify its unsupported cases and reproduce a repaired failure. Separate component receipts make this the visible integration of earlier learner projects rather than an opaque imported demo.
