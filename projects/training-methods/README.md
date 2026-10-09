# Audit pretraining and post-training objectives

Build a small objective toolkit, then use it in eight real CPU training experiments. The pretraining phase uses stages 1–3; post-training continues with stages 4–8. The examples deliberately use synthetic sequences, pixels, features and preferences so that the mechanisms have independent oracles. They do not reproduce BERT, MAE, a full language PPO trainer or human alignment.

## Prepare your workspace

Use the top folder of the extracted course or project workspace as your working directory. Activate a Python environment and install `requirements-cpu.txt`. The pinned environment includes JAX, Flax, NumPy and, for actual framework comparison, PyTorch. No GPU is required.

```sh
# Run run command in terminal using the course Python environment
python3 -m pip install -r requirements-cpu.txt
cp projects/training-methods/starter/methods.py projects/training-methods/my_methods.py
```

On Windows PowerShell, use `Copy-Item projects/training-methods/starter/methods.py projects/training-methods/my_methods.py` and the active environment’s Python executable. If an import fails, compare the active interpreter with the one used to install requirements before changing source code.

Read the connected lessons first:

- [pretraining-01](../../phases/17-pretraining/01-mlm/docs/en.md)
- [pretraining-02](../../phases/17-pretraining/02-mim/docs/en.md)
- [pretraining-03](../../phases/17-pretraining/03-contrastive/docs/en.md)
- [posttraining-01](../../phases/18-posttraining/01-sft/docs/en.md)
- [posttraining-02](../../phases/18-posttraining/02-lora/docs/en.md)
- [posttraining-03](../../phases/18-posttraining/03-reward/docs/en.md)
- [posttraining-04](../../phases/18-posttraining/04-rlhf/docs/en.md)
- [posttraining-05](../../phases/18-posttraining/05-dpo/docs/en.md)

## Build your implementation

The starter retains supporting models and input validation. Replace its named `NotImplementedError` functions. Work on your own file; the solution is a reference to inspect after attempting each stage. Every stage command reruns earlier stages, so a later change cannot silently discard a previously verified capability.

### Stage 1: Masked language modeling

Implement `masked_ce, corrupt_tokens, mlm_logits` in `my_methods.py`.

**Demonstrate:** Masked-target reduction, input corruption, uniform baseline and changed mask.

```sh
# Run run command in terminal using the course Python environment
python3 projects/training-methods/tests/check.py --implementation projects/training-methods/my_methods.py --stage 1
```

A successful run prints `PASS stage 1` after the actual assertions. On failure, inspect the named shape, reduction, mapping or numerical comparison before changing a tolerance. Keep the command, observed output and your explanation of a changed case.

### Stage 2: Masked image reconstruction

Implement `patchify, masked_mse` in `my_methods.py`.

**Demonstrate:** Rectangular multichannel patch order, visible-only features and hidden-patch reduction.

```sh
# Run run command in terminal using the course Python environment
python3 projects/training-methods/tests/check.py --implementation projects/training-methods/my_methods.py --stage 2
```

A successful run prints `PASS stage 2` after the actual assertions. On failure, inspect the named shape, reduction, mapping or numerical comparison before changing a tolerance. Keep the command, observed output and your explanation of a changed case.

### Stage 3: Paired contrastive learning

Implement `paired_contrastive` in `my_methods.py`.

**Demonstrate:** Independent symmetric loss, collapsed baseline and broken pair mapping.

```sh
# Run run command in terminal using the course Python environment
python3 projects/training-methods/tests/check.py --implementation projects/training-methods/my_methods.py --stage 3
```

A successful run prints `PASS stage 3` after the actual assertions. On failure, inspect the named shape, reduction, mapping or numerical comparison before changing a tolerance. Keep the command, observed output and your explanation of a changed case.

### Stage 4: Response-only SFT

Implement `response_logps` in `my_methods.py`.

**Demonstrate:** Shifted target masks, independent sequence sums and actual table training.

```sh
# Run run command in terminal using the course Python environment
python3 projects/training-methods/tests/check.py --implementation projects/training-methods/my_methods.py --stage 4
```

A successful run prints `PASS stage 4` after the actual assertions. On failure, inspect the named shape, reduction, mapping or numerical comparison before changing a tolerance. Keep the command, observed output and your explanation of a changed case.

### Stage 5: Frozen-base LoRA

Implement `lora_forward` in `my_methods.py`.

**Demonstrate:** Two ranks and nonunit alpha, frozen base, trained adapter and merged outputs.

```sh
# Run run command in terminal using the course Python environment
python3 projects/training-methods/tests/check.py --implementation projects/training-methods/my_methods.py --stage 5
```

A successful run prints `PASS stage 5` after the actual assertions. On failure, inspect the named shape, reduction, mapping or numerical comparison before changing a tolerance. Keep the command, observed output and your explanation of a changed case.

### Stage 6: Preference reward model

Implement `preference_loss` in `my_methods.py`.

**Demonstrate:** Stable pairwise likelihood, independent finite-difference gradient and learned ordering.

```sh
# Run run command in terminal using the course Python environment
python3 projects/training-methods/tests/check.py --implementation projects/training-methods/my_methods.py --stage 6
```

A successful run prints `PASS stage 6` after the actual assertions. On failure, inspect the named shape, reduction, mapping or numerical comparison before changing a tolerance. Keep the command, observed output and your explanation of a changed case.

### Stage 7: Learned-reward PPO

Implement `ppo_loss` in `my_methods.py`.

**Demonstrate:** Signed clipped ratios, exact KL, actual sampled rollouts and frozen reference.

```sh
# Run run command in terminal using the course Python environment
python3 projects/training-methods/tests/check.py --implementation projects/training-methods/my_methods.py --stage 7
```

A successful run prints `PASS stage 7` after the actual assertions. On failure, inspect the named shape, reduction, mapping or numerical comparison before changing a tolerance. Keep the command, observed output and your explanation of a changed case.

### Stage 8: Direct preference optimization

Implement `dpo_loss` in `my_methods.py`.

**Demonstrate:** Reference correction, independent stable loss and trained positive margins.

```sh
# Run run command in terminal using the course Python environment
python3 projects/training-methods/tests/check.py --implementation projects/training-methods/my_methods.py --stage 8
```

A successful run prints `PASS stage 8` after the actual assertions. On failure, inspect the named shape, reduction, mapping or numerical comparison before changing a tolerance. Keep the command, observed output and your explanation of a changed case.

## Understand the contracts

`masked_ce(logits, targets, selected)` takes batch/time/vocabulary logits and averages only selected target positions. Validate nonempty Boolean masks outside transformed steps. `corrupt_tokens` must preserve unselected inputs and keep the clean target separate. `mlm_logits` implements the supplied bidirectional attention boundary; it must not receive clean hidden targets.

`patchify` uses NHWC inputs and returns row-major patches. The checker uses rectangular multichannel images, so an accidentally correct square grayscale reshape cannot pass. `masked_mse` averages pixels within each patch and then hidden patches; visible reconstruction errors are deliberately excluded.

`paired_contrastive` normalizes representations and averages row and column log-softmax losses. The positive is the matching row ID across views. This is a symmetric cross-view objective, not the full SimCLR negative set. Zero/collapsed representations and duplicate semantic pairs require deliberate interpretation.

`response_logps` sums next-token log probabilities only where the shifted response mask selects a target. It returns one sum per sequence; the caller owns the training denominator. Prompt context remains present even when prompt targets are not supervised.

`lora_forward` uses input-by-output base weights and the update \(\alpha AB/r\). The base is not an optimization variable. Check the initial zero-output adapter, two ranks, nonunit scaling and merged equivalence before interpreting parameter savings.

`preference_loss` fits chosen-minus-rejected score differences with stable softplus. A common score offset cancels, while scaling changes confidence and the downstream reward tradeoff. All preference labels in these labs are synthetic.

`ppo_loss` retains the minimum of signed unclipped/clipped advantage products and an exact categorical KL to a fixed reference. Old probabilities and advantages are detached. The lab samples real terminal actions, then reuses that batch for several updates. It is a bandit objective audit, not sequence PPO with a learned value head.

`dpo_loss` subtracts the reference log-probability margin before applying beta and softplus. At current-equals-reference it must equal \(\log2\), including a nonuniform reference. Language DPO would use consistent response-token sums and tokenizer identities.

## Read the experiments

The `labs/` files execute using your module's functions during the checker. Each records a training history and named invariants. The matching course lessons contain the plotted reference run, axis definitions and interpretation. Do not treat the different losses as directly comparable: masked token NLL, squared pixel error, contrastive NLL, pairwise preference loss and learned reward have different meanings and units.

Before opening a reference solution, predict a changed condition: a new MLM mask position, a rectangular patch grid, a broken pair permutation, a shifted response mask, a different adapter rank, reversed preferences or a changed PPO advantage sign. Retain an independent calculation for the result. The checker uses additional inputs rather than accepting only the lesson's example constants.

## Transfer to a complete harness

After the objective checks, connect the appropriate mechanism to the [text harness](../text-harness/README.md), [image harness](../image-harness/README.md) or [cross-modal harness](../cross-modal-harness/README.md). Record the base model, tokenizer/preprocessing, dataset split, objective/reduction, masking or augmentation seed and optimizer state. Use [engineering-release](../engineering-release/README.md) to track that experiment and gate an artifact. **TODO (Cross-Harness Objective Integration):** Wire these objectives into `text-harness`, `image-harness`, and `cross-modal-harness` to compare end-to-end training runs.

For actual human preference work, define the task/rubric, obtain appropriate data permissions, retain disagreements, split by prompt/source, and evaluate with held-out judgments. A synthetic ordering and rising reward requires separate verification to establish human alignment.

## Review your evidence

Keep your source file, environment versions, exact commands, actual results, one failing case with its repair, and the independent reasoning for each stage. The public reference suite checks bounded correctness; it is not a hidden exam or independent review of your capability.

To inspect the provided implementation after your attempt:

```sh
# Run run command in terminal using the course Python environment
python3 projects/training-methods/tests/check.py --implementation solution --stage all
```
