# JAX Pathways

One foundation. Many pathways.

A friendly, project-led guide to JAX and its ecosystem, with a technical visual direction inspired by [AI Engineering from Scratch](https://aiengineeringfromscratch.com/). The interface, diagrams, and curriculum outline are original.

## Start locally

```sh
npm run dev
```

Open http://localhost:4173. Only Python 3 and Node (for syntax checks) are needed; the site has no package dependencies or build step.

## Choose what you want to build

- **Start with JAX:** arrays, pure functions, differentiation, vectorization, compilation, keys, and state.
- **Train neural networks:** Flax NNX, Optax, evaluation, checkpoints, and attention.
- **Simulate and discover:** differentiable solvers and scientific inverse problems.
- **Model uncertainty:** Bayesian models and inference.
- **Learn by interacting:** functional environments and batched reinforcement learning.
- **Make it fast. Make it scale:** measurement, profiling, sharding, and kernels.
- **Understand JAX itself:** jaxpr, autodiff, custom derivatives, and compilation.
- **Build a TPU training system:** the focused fundamentals → training loop → Grain/Orbax recovery → Transformer → XProf → integration route.
- **Ship and adapt JAX workloads:** framework bridges, serving, and capacity planning.

## What works today

Responsive homepage, searchable/expandable curriculum, sample lesson reader, a selectable TPU pathway map, saved route selection, evidence notebook, JSON backup export/import, and downloadable 4/8-week learning club plans. Learner data stays in the current browser. Imported evidence is rendered as text and links are restricted to HTTP(S).

## Content status

This repository contains **9 proposed pathways, 55 proposed topics, and 1 authored sample lesson**. These are not the 56 authored competencies described in earlier project notes. The source for that Astro collection was unavailable: `mlnomadpy/ai-tpu-pathways` had only an initial README and license when inspected. No full lessons, exams, or notebooks from that project have been imported.

The TPU outline follows the supplied brief. The three earlier training/recovery/profiling notebooks are not present here. No TPU execution or advanced lesson validation has been performed. Evidence is self-reported; the site does not award certificates or claim verified readiness.

## Authoring

Edit `site/curriculum.json` for the route map. See `content/README.md` for the lesson contract and `docs/design.md` for design choices. Read → experiment → checkpoint → evidence → synthesis exam should remain distinct as authored content is added.

Primary references: [JAX](https://docs.jax.dev/), [Flax](https://flax.readthedocs.io/), [Diffrax](https://docs.kidger.site/diffrax/), [NumPyro](https://num.pyro.ai/en/stable/).

## Checks

```sh
npm run check
node --check site/platform.js
```

Use a served preview rather than opening `index.html` directly: the curriculum is loaded from JSON.
