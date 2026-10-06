# RL authoring and execution evidence

## Scope completed

Four canonical lessons under `phases/12-rl` now teach functional environments, batched/scanned rollouts, actual policy-gradient/clipped-policy training, and multi-seed held-out evaluation. Each contains staged code, worked mathematical mechanisms, prediction experiments, an original exercise, distinct transfer tasks, diagnosis, checkpoint, primary references and an executed figure with source-bound data.

`projects/policy-evaluation` supplies a starter API, complete reference, four cumulative stages, independent checks, learner instructions, a manifest and source-hashed validation receipt. `assessments/rl.md` and `assessments/rl-reviewer.md` are a changed-condition synthesis and public reviewer guide. They remain review drafts, not credentialed exams.

## Executed checks

Authoring ran every lesson’s complete program, both experiments, main solution, both practice solutions and figure code in fresh CPU processes. All four passed. Figure workers then ran the complete example and figure experiment, producing SVG/PNG/data receipts under each lesson’s outputs. The author visually inspected the multi-seed evaluation plot.

Project command:

```sh
python3 projects/policy-evaluation/tests/check.py --implementation solution --stage 4 --report projects/policy-evaluation/validation.json
```

All four stages passed:

1. Exhaustive one-step transitions across horizons 1, 3 and 8, deadline priority, absorbing state and random-reset support.
2. Independent Python reconstruction of every trajectory for changed batch sizes/horizons, one ending event per episode, padding masks, replay and changed keys, hand-computed reward-to-go.
3. Independent host recursive expectation versus JAX dynamic programming, central finite differences at two policies, mixed-sign PPO arithmetic, frozen behavior/target gradients and a 32,768-episode sampled policy gradient versus the exact derivative.
4. Two actual algorithms across three training seeds, complete replay, held-out evaluation, policy isolation and malformed-input rejection.

Recorded environment: Python 3.14.3, JAX 0.9.2, NumPy 2.4.4, one CPU device. Exact environment and source hashes are in `projects/policy-evaluation/validation.json` and figure receipts.

| Method | Training seed | Final exact return | Held-out mean |
| --- | ---: | ---: | ---: |
| REINFORCE | 0 | 0.9521624 | 0.9537841 |
| REINFORCE | 7 | 0.9508238 | 0.9536914 |
| REINFORCE | 23 | 0.9493669 | 0.9513769 |
| Clipped updates | 0 | 0.9757839 | 0.9759717 |
| Clipped updates | 7 | 0.9749347 | 0.9747216 |
| Clipped updates | 23 | 0.9756215 | 0.9753711 |

All start from uniform-policy exact expected return 0.466875. The always-right ceiling for this finite corridor/start distribution is 0.985. REINFORCE takes one gradient step per batch, while the clipped method takes three; this is explicitly not a compute-matched benchmark.

The evaluation lesson executes five trained agents: held-out means approximately 0.975972, 0.974722, 0.975371, 0.969448, 0.974551. The shared uniform-policy evaluation mean is 0.466289. Its plot explains which differences arise from policies and which arise from evaluation sampling, and does not invent confidence intervals.

## Independent assessment arithmetic

Host recursive enumeration, using Python math and no JAX transition, produced expected return 0.4503909552227607 for logits (-0.4, 0.6, 1.2) and horizon 4. Central differences at perturbation 0.0001 produced derivatives approximately (0.0940175298, 0.1640302963, 0.0847317862). JAX float32 gave return 0.45039093 and derivatives (0.09401754, 0.16403031, 0.08473177).

The horizon-six uniform-policy expected return is 0.367578125. The always-left and always-right values are -0.06 and 0.985. These changed-condition values are retained in the reviewer guide; no fabricated performance threshold is applied to the learner’s changed training protocol.

## Registration for the integrator

- Set existing `rl-01` through `rl-04` entries to authored at their existing paths, with manifest exercise/evidence/check fields reflecting their actual content.
- Phase `rl`: `projectId: policy-evaluation`; describe executable CPU interaction, policy learning and multi-seed evaluation.
- Add `projects/policy-evaluation/project.json` to the project registry.
- Register assessment `{id: rl, projectId: policy-evaluation, title: Reinforcement-learning synthesis, status: review-draft, scope: project-synthesis, source: assessments/rl.md, url: assessments/rl.html}`.
- The RL pathway can point its authored capstone at this project and assessment, while bounding its outcome to the small CPU environment.
- Update the RL route’s outdated planned-only description and first artifact.

Shared generators and manifests were intentionally left to the root integrator. The complete script/notebook smoke run, exports and whole-site checks must run after registration. Individual authoring execution and figure receipts do not substitute for that final integration gate.

## Limits

One tiny deterministic-transition environment with randomized starts and sampled actions; stationary tabular policies; undiscounted finite-episode objective; no critic, generalized advantage estimator, continuous actions, production agent stack or hardware performance claim. PPO clipping is a surrogate objective, not a guaranteed trust region. No external expert review or novice walkthrough has occurred. Episode sampling and multiple seeds do not establish general RL competence.

Primary sources inspected: current JAX scan and random-split documentation, plus the original PPO paper (arXiv:1707.06347). Implementation APIs were verified in the installed environment. Lesson references also link JAX randomness and Gymnasium’s termination/truncation discussion.
