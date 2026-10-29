<!-- jaxpathways-content-audit:2026-10-27:prerequisite-handoffs -->

Audited October 27, 2026, edition 0.17.1, PR #71 head `15bf7aa4fbd4460501cc4d6d7802324673a89f6f`. Priority P2. Finding type: curriculum-extension.

## Teaching problem

The Setup route introduces Mesh, PartitionSpec, jit and addressable shards before those concepts are taught. It has hardware cues marking advanced setup as optional, but needs an explicit beginner continuation path and a small placement explanation.

The post-training PPO prose tells learners to complete RL-01 through RL-04, with RL-05 useful for the value/GAE bridge. Those cross-phase dependencies are not represented by the posttraining-04 lesson prerequisite, which names only posttraining-03. Do not force all RL prerequisites on SFT/LoRA, which do not need them.

## Acceptance criteria

- [ ] Represent the PPO lesson cross-phase dependencies in canonical metadata and show them in lesson/chapter entry guidance.
- [ ] Provide a CPU-first beginner branch through setup, with virtual-device/TPU setup explicitly deferred until its conceptual prerequisites.
- [ ] Add a brief readiness task for array shapes, pure transforms, rollout masks and policy objectives at the relevant handoffs.
- [ ] Verify that pathway order and book navigation preserve these dependencies without creating cycles or blocking unrelated lessons.

## Audited source

- [welcome-cpu](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/00-welcome/03-virtual-cpu-devices/lesson.json)
- [welcome-03](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/00-welcome/03-move-your-experiment-to-a-tpu/lesson.json)
- [posttraining-04](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/18-posttraining/04-rlhf/lesson.json)
- [rl-01](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/12-rl/01-write-a-functional-environment/lesson.json)
- [rl-02](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/12-rl/02-batch-environments-and-scan-rollouts/lesson.json)
- [rl-03](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/12-rl/03-policy-gradients-and-ppo/lesson.json)
- [rl-04](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/12-rl/04-evaluate-agents-across-seeds/lesson.json)
- [rl-05](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/12-rl/05-learn-values-and-gae/lesson.json)
- [curriculum/course.json](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/curriculum/course.json)

This issue is a source-level content review. Existing CPU receipts and honest toy/replay/target limitations remain valid within their stated scope. This audit did not rerun all training programs, commission human reviews or launch paid hardware. Keep content maturity, physical-device qualification and model-quality evidence separate.
