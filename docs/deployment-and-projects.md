# Projects and deployment learning update

The Projects catalog (`projects.html`) is generated from the project registry, with a visible main-navigation entry, stage lists, prerequisite course links and ZIP downloads. The two implemented projects remain regression-audit and mlp-classifier. Phase project briefs without implemented workspaces are listed separately. The project workspace and homepage link back to the catalog. See [the wireframe](wireframes/projects.md).

Four canonical lesson drafts extend the shipping pathway:

- deployment-01: Keras/JAX/TensorFlow/PyTorch contracts, parameter layout, preprocessing and state boundaries; optional actual Keras backend comparison.
- deployment-03: JAX serialization round trip, request signatures and rejected inputs.
- deployment-05: weight/activation/accumulator policies, FP32/FP16/BF16 comparisons, INT8/INT4 weight quantization, clipping and W8A8 arithmetic; guidance on FP64 and FP8 limitations.
- deployment-06: edge runtime choices, TensorFlow-to-LiteRT conversion companion, end-to-end request boundaries and target-device evidence requirements.

Existing lesson IDs and URLs are retained. Available lessons are ordered interoperability → export → precision → edge, before the still-planned adaptation and capacity lessons. No deployment capstone is falsely marked implemented. Core code requires only the pinned CPU environment, including flatbuffers for export serialization. Framework companions live beside canonical lesson sources and are included in the downloadable course workspace.

## Verification scope

Core generated scripts and notebooks are checked by `npm run smoke:cpu`; its source-bound receipt is `curriculum/validation.json`. Optional backend and conversion observations are recorded separately in `docs/deployment-lab-validation.json`. Browser rendering was unavailable due to the existing local-URL access restriction; static HTML/link checks, isolated DOM interaction tests and responsive CSS review do not substitute for visual browser verification.

There is no physical edge-device, GPU or TPU execution evidence. INT4 core exercises simulate numerical codes in int8 storage; they do not claim packed storage or INT4 acceleration. Public exercises and checks are learning aids, not independently reviewed competency evidence.
