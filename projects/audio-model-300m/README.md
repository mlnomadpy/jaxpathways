# Capstone: train and release a 300M audio understanding model

**Status: planned specification.** Apply the [common lifecycle gates](../../docs/model-lifecycle-capstones.md) and [tooling](../../docs/model-lifecycle-tooling.md). Extend the [audio harness](../audio-harness/README.md) with AUDIO/DATA/EVAL/NUM/BRIDGE/SERVE preparation. The first advanced outcome is speech recognition; sound-event understanding is a separately evaluated branch. [Audio generation](../audio-generation-model/README.md) is a different project.

Outcome: a complete waveform-to-transcript or waveform-to-event system. A speech encoder candidate with 24 Transformer blocks, width 1,024, 16 heads and MLP width 4,096 is near a 300M backbone budget; count its actual acoustic frontend, positions and task head too. Choose convolutional/temporal details, frame rate and context in pilots. This is an architecture proposal, not a trained or latency-qualified model.

## Data and representation recipe

Choose language, speech domain, offline/streaming requirement and transcript convention. Keep source/recording/speaker identities, sample rates/channels, durations, acoustic conditions, label provenance and permitted use. Group-disjoint splits precede chunking; overlapping segments or repeated speakers can invalidate the intended independence. Freeze transcript normalization and evaluation normalization separately and document both.

A speech-pretraining pilot compares 70% diverse speech and 30% target-domain speech by valid audio seconds; fine-tuning can compare 50/50 supervised broad/target seconds. Report utterance counts and repeated exposure too. For event detection, replace these sources with sound-event data and audit missing/co-occurring labels; do not assume unlabeled events are negatives. The mixture values are hypotheses, not acquired data or optimal recipes.

Freeze waveform decoding, resampling, channels, window/hop, log-mel or learned frontend and valid-length propagation. For CTC, compare character/subword transcript tokenization, define blank and repeated-label behavior, and ensure a valid alignment exists after downsampling. Acoustic frames and text tokens are different sequences.

## Phases

| Phase | Work and objective | Evidence |
| --- | --- | --- |
| 0: signal/reference baseline | Reuse tone/STFT correctness, then train a small supervised speech or event baseline on real data | Resampling/feature oracles, frozen labels and first held-out errors |
| 1: acoustic pretraining | Choose masked latent prediction/reconstruction or a supervised acoustic baseline; if using teacher/quantizer targets, specify their update and gradient rules | Full encoder/frontend checkpoint; mask/time-length checks and downstream probes |
| 2: domain continuation | Adapt on target acoustics with a general-retention mixture and explicit noise/augmentation choices | Continued checkpoint and domain/retention comparison at matched audio exposure |
| 3: task fine-tuning | For ASR train CTC, with sequence-to-sequence decoding as an explicit later alternative; for events train the appropriate single/multi-label objective | Task-head checkpoint; independent tiny alignment or label-loss reference and subgroup evaluation |
| 4: streaming/precision extension | Compare an offline baseline with a causally valid streaming/chunked model if required; qualify supported quantization | State/boundary tests, accuracy-latency tradeoff and whole-waveform inference parity |
| 5: release | Package frontend, model, tokenizer/labels, decoder and runtime | Cold-download canaries, service failure tests and documented task limits |

For CTC, enumerate paths on a tiny logits table and independently verify blank removal and repeat collapse. Reject impossible target lengths rather than hiding them in a mean. A sequence-to-sequence decoder has different teacher-forcing, exposure and hallucination behavior and needs its own checks. A streaming model cannot use future acoustic context from an offline frontend silently.

Every checkpoint includes data position, augmentation RNG, optimizer/schedule and any teacher/quantizer state. Source masks, silence/padding and valid frames must be count-weighted correctly. A phase change records whether the optimizer resets; exact resume is demonstrated within each phase.

## Benchmarks and diagnostics

ASR uses pinned in-domain and out-of-domain speech test sets with WER/CER, normalization and decoding settings fixed. Include noise/accent/duration/speaker slices with counts, insertions/deletions/substitutions, empty references and silent inputs. Compare greedy decoding and an external language-model decoder separately; do not attribute the latter's gain entirely to acoustic weights.

Use appropriate [SUPERB/S3PRL](https://github.com/s3prl/s3prl) protocols for additional speech-representation claims. Speaker verification requires its own trials and EER convention. Sound-event tasks use declared mAP/F1/class metrics and label-completeness assumptions. A good CTC score does not establish speaker identity or general sound semantics.

Monitor audio seconds and valid frames, duration/padding, clipping/silence, sample-rate rejection, source/language shares, component losses, gradients and precision failures. During CTC training add blank dominance, alignment feasibility and dev WER/CER. Measure real-time factor, startup and end-to-end latency under the actual streaming/offline boundary.

Explain waveforms/spectrograms, blank/token probability strips, edit alignments, subgroup error counts and latency-quality curves. Store a bounded set of permitted audio examples and transcripts in W&B/MLflow/local artifacts; aggregate logging must not upload an entire recording corpus.

## Release and learner review

Ship the complete processor/frontend, model/head, transcript tokenizer or class map, blank/decoding rules and any external decoder identity. Test raw waveform parity after framework conversion and precision changes, including resampling and silent/malformed/overlong inputs. Follow the [Hub publication workflow](../embedding-model-300m/RELEASE.md); a weights-only upload cannot reconstruct audio preprocessing or decoding.

Qualify request duration bounds, concurrent streams, cancellation and recurrent/chunk state isolation where applicable. Roll back all processor/model/decoder components together. The portfolio includes an acoustic recipe decision, objective baseline, exact recovery, task failure analysis, measured runtime and clean consumer evidence. Target-scale training remains future work.
