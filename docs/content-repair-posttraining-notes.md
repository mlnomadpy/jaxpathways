# Posttraining content repair notes

This repair addresses the saved-sequence handoff and adapter identity findings from the October 27, 2026 content audit.

## Saved-decoder route across SFT, reward, DPO and PPO analysis

`phases/18-posttraining/01-sft/lesson.json` now adds a full-workspace SFT transfer practice from the retained foundations inference package. It checks the parent parameter and tokenizer digests, verifies the current `model.py`, `config.py`, and `train.py` source hashes against the package identity, initializes a new post-training update, and applies one response-only update to fixed synthetic pairs. It reports same-case train and held-out sequence NLL before and after, and writes a temporary child inference bundle with parent, tokenizer, config, and child-weight hashes. Its target mask includes response tokens and EOS while excluding prompt and padding targets.

`phases/18-posttraining/03-reward/lesson.json` adds a fixed synthetic held-out set with unanimous and fractional-vote disagreement cases. Its zero-weight preference probabilities and Brier score are captured in a build step between training-pair setup and the fit loop, so the baseline is recorded before reward updates. After fitting, the same examples are scored again. Unanimous accuracy and disagreement probability/Brier remain separate, and no human reward or language-quality claim is made.

`phases/18-posttraining/05-dpo/lesson.json` continues the route with sequence log-probability sums, a one-step preference update, held-out preference/disagreement and retention measurements. A 50/50 preference is marked ambiguous and has no hard-accuracy value; its probability, vote fraction, and expected annotator agreement are reported. The lesson also runs a no-update replay of existing PPO evaluation evidence at reward scales 0.5, 1, and 2. The saved PPO quality gate remains failed: child-minus-parent reward is zero at all three scales, while 8/8 sampled action sequences match. Response match is reported separately and is not called useful behavior. No PPO update, retuning, or quality campaign was run.

The DPO handoff links the published [foundations guide](https://www.tahabouhsine.com/jaxpathways/project-guides/text-harness-foundations.html) and [posttraining/PPO guide](https://www.tahabouhsine.com/jaxpathways/project-guides/posttraining-audit.html). The DPO and SFT lesson text lists their full-workspace dependency; the regular standalone `main.py` SFT/DPO mechanics examples do not load the full pilot package.

### Required files and pinned identities for the full-workspace transfer

The saved-model practices require the full repository and these files:

- `projects/text-harness/foundations/model.py` SHA-256 `94a88ae7fd6a251b5b190b4631496e04bb71a31c1bfe418511fdd37ffef034c9`
- `projects/text-harness/foundations/config.py` SHA-256 `5c05801a7ec1a15b3978c87c77ac4e64d1b056c60487cbcff3cdf72475c8bb64`
- `projects/text-harness/foundations/train.py` SHA-256 `ee9f3bc73505ce8e00fe9680589d97805593e5c8c035cba4709b5f334eb27a2b`
- `projects/text-harness/foundations/reference/run/inference-package/package.json` SHA-256 `02b92a9f56325e5a38be2c40bf263d2ca91db6122ba84dd4df266339457bf278`
- `projects/text-harness/foundations/reference/run/inference-package/params.npz` SHA-256 `271cb6457de5da3f727c343c64b7bd350c00e503ea7ac847eecf51fbd5558802`
- `projects/text-harness/foundations/reference/run/inference-package/tokenizer.json` SHA-256 `c92b534443cc6a1eacd5e78406b2c42e59553df704a07102a03c639d3694f586`
- `projects/text-harness/foundations/requirements-foundations.txt`: JAX/JAXlib 0.9.2, NumPy 2.4.4, Tokenizers 0.22.2.

This is the package input list, not a redistributable license grant. The retained model/source rights limits remain those documented by the foundation project.

## LoRA artifact identity

`phases/18-posttraining/02-lora/lesson.json` keeps its existing fitted-base, gradient-asymmetry, nonunit-scaling, and merge evidence. The primary adapter save exercise now points directly to the complete fresh-process identity practice and states that the older unbound NPZ example is not a reusable artifact. A bounded synthetic rank-two, alpha-three teaching fit trains the exact factors that are then serialized against the same frozen base. The practice records target path, base and factor shapes/dtypes, rank, alpha, and the `alpha/r` rule. A fresh interpreter reloads those trained factors and compares the unmerged and merged computations on new inputs at rank 2 and alpha 3, giving scale 1.5. This closes the adapter save/load mechanics contract; it does not claim a language-model adapter improves task quality.

Negative controls supply a different base with the same shape and dtype, then alter rank, alpha, target path, and factor-shape metadata. The loader rejects each case before prediction. This demonstrates why matching dimensions do not establish model identity.

## Verification

Using `/private/tmp/jaxpathways-0170-clean-learner-env/bin/python`:

- Generated whole-lesson script for `posttraining-01` passed. The saved-decoder one-step SFT train NLL changed from `5.8471966` to `5.8463287`; the two prompt-disjoint held-out NLLs were `5.8963127, 5.8930640` before and `5.8961415, 5.8930955` after. Tokenized training and held-out prompt sets were asserted disjoint. This is retained as a bounded numeric observation, not a quality claim.
- Generated whole-lesson script for `posttraining-03` passed. The prefit held-out Brier score was `0.140625`; post-fit it was `0.0222862`. The two disagreement Brier scores were `0.03125` and `0.0294416` respectively. All labels and features are synthetic.
- Generated whole-lesson script for `posttraining-05` passed. The DPO 50/50 disagreement had `hard_accuracy: None`, predicted chosen probability about `0.002515`, and expected annotator agreement `0.5`; it was not labeled incorrect. The hand-summed unequal-length response terms were `[-5.877766, -5.884814, -5.825323]`, summing to `-17.5879026825`; the JAX result was `-17.5879020691`. The existing PPO held-out reward difference remained zero at all three scales and its strict quality gate remained failed.
- Generated whole-lesson script for `posttraining-02` passed. Existing LoRA base-freeze evidence remained intact. The bounded rank-two fit on the frozen base fell from MSE `0.12098397` to `0.00001915084`; the exact trained factor arrays passed fresh-process merge parity. The loader rejected the wrong same-shape base plus rank, alpha, target, and shape metadata mutations. This is synthetic adapter mechanics evidence only.
- The primary LoRA learner-only script generated successfully and passed `py_compile`; executing it stops at its authored `NotImplementedError` TODO, as expected for an unfilled learner scaffold.
- All four authored lesson JSON files parsed and `git diff --check` passed.

Temporary generated scripts live under `/tmp`. The local lesson changes are `phases/18-posttraining/01-sft/lesson.json`, `02-lora/lesson.json`, `03-reward/lesson.json`, and `05-dpo/lesson.json`, plus this note. Central curriculum metadata remains with the parent repair.
