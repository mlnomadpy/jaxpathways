# Capstone: train and release an audio-generation pipeline

**Status: planned specification.** Inherits the [lifecycle contract](../../docs/model-lifecycle-capstones.md) and [tool choices](../../docs/model-lifecycle-tooling.md). Use AUDIO and GEN plus causal sequence/state preparation. Audio understanding and audio generation have separate objectives, artifacts and evaluation.

Outcome: begin with a bounded sound-generation task and progress to class/text-conditioned audio. Speech synthesis is an explicit specialization with transcript/alignment and speaker-data requirements. A roughly 300M token generator or denoiser is a scale hypothesis after pilots; include the codec, conditioning model and waveform decoder in total deployment counts and memory.

## Data and input decisions

Choose speech, music or general sound as the initial population. Curate recordings and conditions with source/artist/speaker/group identities, duration/sample-rate/channel metadata, captions/transcripts, rights and attribution. Split groups before windows; avoid near-duplicate clips across evaluation. For a domain adaptation pilot, compare broad/target mixtures such as 80/20 and 50/50 by valid seconds, recording clip counts and repeated exposure. Exact data and full-scale budgets remain to be selected.

Choose a representation: discrete codec tokens plus an autoregressive model is the first sequence-based route; continuous latents with a denoiser is a separate GEN extension. Qualify a chosen pretrained codec independently or explicitly train one. Freeze codec version, sample rate, bandwidth, frame rate, codebook layout and waveform decoder; its reconstruction quality places a limit on the complete system.

## Phases and objectives

| Phase | Learner work | Evidence |
| --- | --- | --- |
| 0: codec/representation | Encode/decode a known waveform and measure reconstruction, duration, channels and loudness | Golden round-trip waveforms, codec identity and reconstruction limitations |
| 1: generative pretraining | Train a causal acoustic-token model with a declared codebook ordering/delay pattern and valid-token loss; alternatively implement a separately specified denoising model | Full checkpoint; next-token/causality oracle, codec-frame masks and real generated samples |
| 2: conditioning | Add class/text conditioning and test correct, absent and swapped conditions | Conditioned checkpoint, conditioning identity and behavioral ablation |
| 3: domain/task adaptation | Compare full tuning/adapters under a fixed recipe; for TTS add text/phoneme normalization and alignment/duration contracts | Adapted checkpoint, retained coverage and task-specific quality; no unmeasured speaker-generalization claim |
| 4: inference/release | Qualify sampling, stream/chunk boundaries, precision and the complete waveform decoder | Measured quality/runtime tradeoff, loaded-pipeline parity and immutable Hub consumer test |

For multiple codec codebooks, specify exactly which tokens are visible when predicting each target. A faulty delay/interleave scheme can leak the answer while loss improves. Independently test a small schedule and require generation to follow the same factorization as training. In the denoising variant, noise targets/schedules and sampler conventions replace acoustic-token cross-entropy; they are not interchangeable.

Checkpoints retain codec/condition identities, data cursor, optimizer, RNG and sampling-independent training state. Use exact resume within phases; changed codec/token vocabularies require an explicit migration/new training contract. An imported codec is disclosed as a pretrained component, even if the generator starts randomly.

## Evaluation and monitoring

Separate codec reconstruction, generator likelihood/denoising quality and complete audio quality. Use a pinned perceptual/distribution metric such as an appropriate FAD implementation with fixed embedding model, sample rate, duration and reference population; a metric change may reflect preprocessing rather than better sound. Add diversity, condition adherence and a documented listening protocol with held-out prompts and retained failures. The [AudioCraft reference](https://github.com/facebookresearch/audiocraft) supplies codec/generation/evaluation patterns to inspect, not results for our model.

For speech synthesis, measure intelligibility with a fixed recognizer plus reviewed examples, pronunciation/number handling and the claimed speaker behavior; recognizer WER alone is not naturalness. Human MOS/preference results require actual raters and a stated protocol. For music/sound, use appropriate conditioning and perceptual checks rather than ASR scores. Never substitute a short attractive clip for the full evaluation population.

Monitor audio seconds/codec tokens, source exposure, per-codebook loss and utilization, silence/clipping, repeated loops, stop/duration behavior, condition dropout, gradients, memory and throughput. Record sample seeds/temperature/guidance, codebook settings and decoder revision with every listening example. Measure first-audio latency, real-time factor, streaming glitches and complete-request latency separately.

Required explained artifacts: waveform/spectrogram codec comparisons, codebook occupancy, fixed-condition samples across stages, failed/looped generations and quality-versus-latency curves. W&B audio tables or equivalent local/MLflow artifacts should contain only the permitted curated examples.

## Publish a complete generator

Release generator weights, codec/decoder or immutable component references, condition tokenizer/processor, generation schedule and defaults, input/output bounds and the tested consumer pipeline. Follow the [Hub workflow](../embedding-model-300m/RELEASE.md), substituting waveform canaries and audio quality gates. A Transformers/Diffusers/AudioCraft interface is advertised only after a compatible implementation is verified.

Service tests cover duration limits, chunk continuity, cancellation, concurrent state isolation, output sample-rate correctness and resource bounds. Quantize/evaluate the complete pipeline where supported, identifying unquantized stages. Rollback restores codec, generator, conditioner and decoder together. The learner's final report distinguishes component quality, task quality and operational readiness.
