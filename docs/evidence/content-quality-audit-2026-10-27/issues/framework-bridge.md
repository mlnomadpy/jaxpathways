<!-- jaxpathways-content-audit:2026-10-27:framework-bridge -->

Audited October 27, 2026, edition 0.17.1, PR #71 head `15bf7aa4fbd4460501cc4d6d7802324673a89f6f`. Priority P2. Finding type: curriculum-extension.

## Teaching problem

There is already an optional real Keras backend lab and an actual PyTorch-to-NNX MLP/LayerNorm conversion with epsilon and gradient checks. These are useful foundations. The promised conversion skill still lacks the Transformer-specific failure modes learners face with saved language/embedding models: fused QKV layout, head axes, positions, masks, processor identity and tied embeddings.

## Acceptance criteria

- [ ] Add a small saved Transformer conversion using the existing chapter architecture, with an explicit source-to-target tensor/name/layout map.
- [ ] Compare asymmetric-input intermediate activations in execution order and report the first divergent layer, absolute/relative error and tolerance rationale.
- [ ] Include a nonuniform input-gradient/VJP check, mask/position parity and save/reload into a fresh process.
- [ ] Inject one QKV/head-layout error and one semantic processor/mask error that cannot be caught by shape checks alone.
- [ ] Pin versions and identify the bounded architecture. Preserve optional-framework installation and do not imply support for every checkpoint family.

## Audited source

- [deployment-01](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/15-deployment/01-keras-and-pytorch-bridges-to-explicit-jax/lesson.json)
- [deployment-08](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/15-deployment/08-pytorch-to-flax-numerical-parity/lesson.json)
- [transformers-03](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/07-transformers/03-a-transformer-block-and-mixed-precision/lesson.json)
- [transformers-04](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/07-transformers/04-train-checkpoint-and-generate/lesson.json)

This issue is a source-level content review. Existing CPU receipts and honest toy/replay/target limitations remain valid within their stated scope. This audit did not rerun all training programs, commission human reviews or launch paid hardware. Keep content maturity, physical-device qualification and model-quality evidence separate.
