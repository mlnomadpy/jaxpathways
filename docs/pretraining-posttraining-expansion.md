# Pretraining, post-training and architecture parity — 2026-10-05

The course now has 19 phases, 97 authored lessons and 19 staged projects. Phase numbers preserve existing IDs and folders; the models and TPU-oriented learning routes place the new phases after Transformers, with the existing RL phase before post-training. Canonical prerequisites remain explicit rather than relying on numeric phase order.

## Coverage and what actually executes

| Topic | Lesson | Executable scope |
| --- | --- | --- |
| Masked language modeling | pretraining-01 | Trained bidirectional single-head attention on synthetic repeated-token sequences; corruption/loss separation, masked gradients and changed-position evaluation |
| Masked image modeling | pretraining-02 | Visible-only patch features and a trained linear decoder; independent spatial layout, hidden MSE and held-amplitude checks |
| Contrastive learning | pretraining-03 | Actual shared embedding training under symmetric cross-view loss; stable oracle, collapse baseline, pairing invariance and duplicate false-negative diagnosis |
| Supervised fine-tuning | posttraining-01 | Response-only shifted-target objective on a trained categorical next-token table; mask-gradient and denominator checks |
| LoRA | posttraining-02 | Actual base regression training, frozen base, learned rank-one update, first-gradient analysis, saved adapter and new-input merge parity |
| Reward modeling | posttraining-03 | Learned pairwise reward from declared synthetic preferences; independent stable likelihood and gradient check, offset/scale analysis |
| RL and PPO | existing rl-01 through rl-04 | Functional environment, rollouts, actual policy updates and seed-based evaluation; retained as prerequisites |
| RLHF mechanics | posttraining-04 | Learned synthetic reward, actual sampled terminal actions, fixed old probabilities during PPO epochs, fixed reference, exact reward/KL evaluation |
| DPO | posttraining-05 | Actual updates on reference-corrected synthetic preferences, independent softplus oracle, equal-policy baseline and frozen-reference checks |
| PyTorch → Flax | deployment-08 | Real PyTorch state_dict loaded into Flax NNX; layer outputs and input gradients, malformed mapping rejection, injected normalization error and saved target artifact |

Every new lesson has original teaching sections, explicit math, a runnable reference, changed-condition practice, a formative checkpoint and an explained executed figure. All nine figure artifacts were visually inspected. Their data are recorded CPU results rather than simulated learning curves.

## Two staged projects

[training-methods](../projects/training-methods/README.md) provides eight cumulative learner stages, a starter, independent checks and training experiments that call the learner's functions. The checker includes changed shapes, ranks, temperatures, label order and advantage signs. It does not just replay one expected number.

[weight-conversion](../projects/weight-conversion/README.md) provides three stages: complete weight mapping and real-framework parity, numerical error localization, then converted-artifact inference in a fresh process. It uses asymmetric dense dimensions, LayerNorm and exact GELU, several seeds/input scales, explicit elementwise absolute-plus-relative tolerances and input-gradient comparisons. A deliberate epsilon mismatch first fails at normalization. CPU forward maximum absolute errors in the lesson fixture were approximately 2.38e-7 to 3.58e-7, while the wrong-epsilon normalization error was about 0.0465.

The tested environment contains JAX 0.9.2, Flax 0.12.6 and PyTorch 2.11.0. PyTorch is pinned in requirements-cpu.txt because this lab exercises the actual framework rather than an array transpose labeled as conversion.

## Integration

New phase guides and README sources preserve readiness questions, ordered milestones and project evidence. Models/TPU routes and relevant career skills/review questions include the new methods. Modality guides link objective preparation and numerical conversion while explicitly distinguishing those labs from retraining every complete modality harness. The models synthesis draft adds transfer and parity tasks. Generic additional-project links now display the actual project title rather than labeling every extension as an engineering-release project.

## Limits

These are bounded objective and architecture labs. They do not reproduce full BERT/MAE, natural-corpus pretraining, full sequence-language PPO with a learned value model, QLoRA, human-feedback collection, large-model alignment or arbitrary architecture conversion. The reward labels are synthetic. The conversion mapping is specifically dense–LayerNorm–GELU–dense; attention, convolution, tied embeddings and vocabulary remapping require their own executed mappings. Real-data transfer, independent learner review and accelerator/edge qualification remain separate.

Browser access was previously denied and was not retried. Rendering and links are checked through build/static DOM tooling; live mobile usability is not claimed.

## Final verification

- All 97 lessons passed fresh script and notebook execution: 194 CPU executions.
- All 19 registered project reference suites passed, including the new eight-stage objective project and three-stage actual-framework conversion project. Independently recomputed source hashes match every project receipt.
- Production build and `npm run check` passed: 56 tests, 64 rendered pages, 2,785 local references, Astro/lint/format/curriculum checks, 97 offline chapters, EPUB and workspace/project-bundle fidelity, and 11 synthesis review drafts.
- All 19 phase guides preserve lesson coverage, project evidence and explicit remaining-work boundaries.
- GitHub Pages subpath checks and seven DOM tests passed; the normal root preview build was restored afterward.
- Content and visual audits passed: 97 authored entries, 92 executed plot figures and five conceptual diagrams.

Evidence is recorded in `curriculum/validation.json`, `curriculum/project-validation.json`, the nine new lessons' `outputs/execution.json` and `outputs/visual.json`, and the regenerated content/visual audit reports.

## Primary references

The lessons link the original [BERT](https://arxiv.org/abs/1810.04805), [MAE](https://arxiv.org/abs/2111.06377), [SimCLR](https://arxiv.org/abs/2002.05709), [CLIP](https://arxiv.org/abs/2103.00020), [LoRA](https://arxiv.org/abs/2106.09685), [InstructGPT](https://arxiv.org/abs/2203.02155), [PPO](https://arxiv.org/abs/1707.06347) and [DPO](https://arxiv.org/abs/2305.18290) papers, plus official PyTorch/Flax API documentation. The small data, worked values, prose and checks are original course examples; they are not paper reproductions.
