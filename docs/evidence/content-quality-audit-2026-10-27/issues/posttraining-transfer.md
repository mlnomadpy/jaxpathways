<!-- jaxpathways-content-audit:2026-10-27:posttraining-transfer -->

Audited October 27, 2026, edition 0.17.1, PR #71 head `15bf7aa4fbd4460501cc4d6d7802324673a89f6f`. Priority P2. Finding type: curriculum-extension.

## Teaching problem

The current mechanics are useful and their limitations are disclosed: SFT trains a token transition table that cannot use earlier context, reward ranking uses three training comparisons, and DPO operates on a three-action table. A saved-decoder PPO appendix now exists and reports a failed quality result; it must not be erased or represented as a pass.

The missing teaching step is a required, bounded route that carries a saved sequence model from the existing causal/foundations pilot through learner edits and held-out checks. An optional appendix link alone does not tell learners how to transfer the objective and evidence contracts.

## Acceptance criteria

- [ ] Give a small saved-decoder transfer checklist with exact prerequisite, model/tokenizer identity, input/output artifact and one learner-controlled change at each stage.
- [ ] Score unequal-length responses using summed token log probabilities, shifted response masks and padding rules; compare a hand calculation and an independent reference.
- [ ] Add held-out prompts, preference disagreement slices, reward-scale sensitivity and retention evaluation, with baselines defined before updates.
- [ ] Connect the existing PPO appendix with a changed-condition analysis task that preserves its failed quality result and explicitly separates objective optimization from response usefulness.
- [ ] Keep full production training/release work in #29; this issue can be completed with a bounded teaching fixture and retained evidence, without launching a new model-quality campaign.

## Audited source

- [transformers-04](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/07-transformers/04-train-checkpoint-and-generate/lesson.json)
- [deployment-02](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/15-deployment/02-adapt-a-pretrained-model-and-choose-a-post-training-objective/lesson.json)
- [posttraining-01](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/18-posttraining/01-sft/lesson.json)
- [posttraining-03](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/18-posttraining/03-reward/lesson.json)
- [posttraining-04](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/18-posttraining/04-rlhf/lesson.json)
- [posttraining-05](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/18-posttraining/05-dpo/lesson.json)
- [projects/causal-model-300m/README.md](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/projects/causal-model-300m/README.md)
- [projects/text-harness/README.md](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/projects/text-harness/README.md)
- [projects/posttraining-audit/README.md](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/projects/posttraining-audit/README.md)

This issue is a source-level content review. Existing CPU receipts and honest toy/replay/target limitations remain valid within their stated scope. This audit did not rerun all training programs, commission human reviews or launch paid hardware. Keep content maturity, physical-device qualification and model-quality evidence separate.
