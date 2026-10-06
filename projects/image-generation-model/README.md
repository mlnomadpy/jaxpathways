# Capstone: train and release an image-generation pipeline

**Status: planned specification.** Follow the [shared lifecycle](../../docs/model-lifecycle-capstones.md) and [tooling](../../docs/model-lifecycle-tooling.md). Reuse image data/recovery/export engineering, then GEN-1..3 for the generative mechanisms. No image-generation system is implemented by this plan.

Outcome: a reproducible bounded image generator, beginning with small class-conditioned images and progressing to a justified domain/text-conditioned experiment. A roughly 300M denoiser is an optional target after pilot measurements. Count the denoiser, VAE/codec and conditioning encoders separately and report the total pipeline; do not advertise a 300M system when auxiliary models make it larger.

## Recipe and model choice

Choose resolution, output domain and condition type before sourcing data. Retain image/caption/class provenance, duplicate/entity groups, source splits and permitted use. Audit caption-image agreement, aspect-ratio processing and selection biases. Begin with a simple broad/target mixture comparison (for example 80/20 versus 50/50 sampled images) at a common measured compute budget, retaining actual pixel/latent-token exposure. Exact sources and budgets are selected in the pilot.

Choose pixel-space diffusion first for an independent small reference, then compare a latent-space pipeline when scale warrants it. If using a pretrained autoencoder/text encoder, disclose it and retain its identity/license; importing those components is not from-scratch training of the whole pipeline. An optional trained autoencoder has its own reconstruction/latent-scale qualification. Freeze normalization, latent scale, condition representation and resolution handling before the denoiser experiment.

## Training phases

| Phase | Objective / work | Handoff and independent check |
| --- | --- | --- |
| 0: data/representation | Verify raw-to-pixel/latent preprocessing and decoder reconstruction | Golden round trips, latent statistics and processor/autoencoder identities |
| 1: denoising baseline | Train a noise-prediction DDPM baseline under a specified schedule and timestep sampler | Denoiser/optimizer/EMA/RNG checkpoint; forward-noise moments and one-step reference |
| 2: conditioning | Add class conditioning, then a separately scoped text-conditioning extension; declare condition dropout | Conditional checkpoint; correct/absent/swapped-condition tests and held-out outputs |
| 3: domain adaptation | Compare full fine-tuning and adapters under an explicit data mixture; preserve diversity/general retention | Adapted checkpoint with measured quality/coverage tradeoff and frozen-leaf checks |
| 4: sampling and release | Compare samplers/steps/guidance with fixed seeds and budgets; qualify reduced precision and full exported pipeline | Versioned scheduler/config, native/loaded outputs, latency/memory and quality report |

A baseline uses \(x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon\) and
\(L=\mathbb{E}_{x_0,t,\epsilon}\|\epsilon-\epsilon_\theta(x_t,t,c)\|^2\).
Define the noise schedule, conditioning \(c\), timestep distribution and reduction. An alternative velocity target or flow-matching model changes the target and sampler contract; teach it as an explicit later comparison rather than interchanging formulas. The [DDPM paper](https://arxiv.org/abs/2006.11239) supplies the mechanism reference.

Resume includes optimizer, EMA if used, sampler/data cursor, augmentation/noise/timestep RNG and schedule state. Promotion requires validation and sampling evidence, not just declining denoising loss. Keep the selected model/EMA identity consistent throughout evaluation and release.

## Evaluation and monitoring

Use a fixed held-out image population, prompt/class inventory, seed list, sample count, resolution and feature-extractor/preprocessing revision. Report FID/KID or other selected distribution metrics with their finite-sample/protocol limitations, diversity/coverage, conditioning adherence and a documented human review where performed. [torch-fidelity](https://github.com/toshas/torch-fidelity) is a candidate metric implementation; a favorable scalar is not proof of prompt fidelity or absence of memorization. Add nearest-training-example inspection and held-out domain failures. Do not tune on final prompts or select only attractive seeds.

Log loss by timestep/SNR range, condition dropout, latent/activation scales, gradient norms, nonfinite steps, EMA policy, images/pixels processed, source exposure and memory/throughput. Periodic sampling uses a fixed inexpensive development panel; distinguish sampler time from training-step time. W&B media tables or local/MLflow artifacts retain images alongside exact seed, prompt, checkpoint and scheduler.

Required figures: noise-level progression, timestep-binned error, actual fixed-seed sample evolution, condition swaps, failures/nearest-neighbor examples and quality-versus-sampling-cost curves. Explain stochastic variation and sample-selection rules.

## Package the whole pipeline

Release denoiser plus required autoencoder/conditioning components or immutable references, tokenizer/processor, latent scale, scheduler, guidance convention, supported shapes, dtype and tested loader. [Diffusers](https://github.com/huggingface/diffusers) is an optional reference/consumer path only after an implemented mapping is qualified. Compare intermediate latents and final outputs for numerical conversion; compare distributions/task quality as well for changed precision/samplers.

Follow the [Hub workflow](../embedding-model-300m/RELEASE.md) with fixed-seed image canaries and a clean pipeline download. Service qualification covers bounded resolution/steps, concurrency, deadlines/cancellation and actual peak memory. Rollback restores the compatible pipeline configuration and all components. The final dossier reports measured capability and limits rather than claiming that a denoiser weight file is a production generator.
