# Independent review: pretraining and posttraining repairs

Read-only review of the seven lesson sources changed for P2 #84/#85/#86. I reviewed the generated lesson scripts as well as the authored transfer blocks. I did not change any lesson, run a production checkpoint, or use a cloud service.

## Confirmed issues

1. **SFT's held-out prompt set overlaps its training prompt set.** In the saved-decoder practice, training includes `("Q: ", "yes")`, while `heldout` includes `("Q: ", "unknown")` (`phases/18-posttraining/01-sft/lesson.json`, practice “Run one response-masked SFT update…”). Thus one of two held-out examples reuses a training prompt. The script only checks that both held-out NLLs are finite; its reported before/after values do not reveal the overlap. This is valid as a held-out response example, but not as a disjoint held-out-prompt evaluation. The prompt asks for a fixed prompt/response split and the lesson discusses held-out prompts, so either use disjoint prompt groups or label the measurement precisely as held-response NLL.

2. **The LoRA fresh-process round trip serializes new random factors rather than the factors trained by the lesson.** The transfer solution sets `a2` and `b2` from fresh JAX random draws (`posttraining-02/lesson.json`, practice solution “Enforce base identity…”), then validates those factors against their own merged form. It never passes the fitted `a` and `b` from the earlier LoRA experiment into the bundle. The loader's digest and metadata negative controls work, but the exercise prompt says to extend the fitted-base save exercise and the repair notes describe this as the trained matrix fixture. As written it demonstrates the format and algebra for a random adapter, not save/reload of the trained adapter.

3. **The primary LoRA learner starter still describes the old, narrower task.** The main exercise now requires a base hash, complete manifest, separate-process reload and rejection controls (`lesson.json` exercise/solution near lines 85–86), but the lesson-level `learnerStarter` around line 326 still declares only a base matrix, factors, rank/scaling and held-out inputs, and provides `roundtrip_lora_adapter(base, a, b, alpha, heldout_x, path)`. That scaffold contains no manifest, base digest, subprocess, wrong-base test or metadata rejection, so the learner-only path does not inherit the revised exercise contract. The transfer-practice starter has the right richer contract, but it does not repair the primary starter.

4. **DPO labels a 50/50 disagreement as simply incorrect.** The held-out evaluator stores `correct = float(good_lp > bad_lp)` for every pair, including the split-vote row with `vote=0.5`. The generated CPU output records that disagreement as `correct: 0.0` before and after the update. A tie has no single correct direction; this field converts the disagreement into a false binary judgment. The reward-model lesson correctly uses Brier error against fractional votes. For DPO, report the margin/probability against the vote fraction, or omit `correct` for the disagreement slice. The existing output does keep the slice separate, so this is a metric-labeling problem rather than a failure of the mask or DPO update.

## Verified behavior and limits

- **MLM:** the saved embedding/head arrays reload and their parameter digest matches. The changed-mask held-out loss is `3.57109` versus uniform `log(3)=1.09861`; the frozen probe is `0.6667`. The lesson and notes present these as weak/failed transfer signals, not a quality pass. Synthetic groups and three examples remain a mechanics fixture.
- **MIM:** the changed-mask held-out losses (`0.03627`, `0.03820`) are both worse than the visible-mean baselines (`0.00879`, `0.02441`). The separate mask-aware encoder/decoder is saved and reloaded; the frozen probe is evaluated on the held-out image indices. The text explicitly says the simpler baseline wins and does not identify this with the gated image pilot.
- **Contrastive:** held-out groups have two same-class positives each. The multi-positive numerator sums their probability mass. The relation-matrix metric path and separate relevant-ID ranking loop agree exactly on Recall@1 and mAP (both `1.0` on this deliberately easy six-row synthetic fixture). The text labels this evaluation-mechanics evidence, not retrieval quality.
- **SFT/DPO token accounting:** the verified tokenizer marks `[PAD]=0`, `[BOS]=1`, `[EOS]=2`; source, config, parameter and tokenizer hashes match the retained package. Both transfer paths shift logits to `[:-1]` and targets to `[1:]`, begin response targets at the absolute prompt length, include EOS, reject PAD targets, and sum DPO sequence log probabilities. The DPO hand sum matches the host NumPy log-softmax aggregation for an unequal-length response pair.
- **Reference and baselines:** DPO clones the loaded reference tree before update and uses it in every corrected margin. Its held-out preference and retention baselines are recorded before the single update. Reward prefit probabilities are exactly `0.5`; the same four synthetic pairs are scored after fitting, with disagreement Brier reported against fractional votes.
- **LoRA guards:** the child process checks base-byte hash before applying factors; it rejects a shape/dtype-compatible but value-different base, and altered rank, alpha, target path and factor-shape metadata. Rank two / alpha three exercises scale `1.5`; merge parity passes on new inputs. Earlier base-freeze and gradient asymmetry tests remain intact. The issue above is that the serialized factors are newly random, not trained.
- **PPO failure:** the existing evaluation artifact remains unchanged and has `qualityGate.heldoutRewardFloorPassed == false`. The DPO transfer replays the eight recorded episodes at reward scales `0.5`, `1`, and `2`; response IDs still match 8/8 and the held-out reward difference remains zero. No new PPO training or quality claim was made.
- **Portability:** the SFT/DPO practices explicitly require the full repository checkout, the saved package files, and `requirements-foundations.txt` (`jax/jaxlib 0.9.2`, `numpy 2.4.4`, `tokenizers 0.22.2`). I checked the declared package/source digests and ran from the repository root. This is intentionally a full-workspace dependency, not a standalone lesson-script-only exercise.

## Exact local checks

Pretraining generated-script checks:

```sh
PYTHON=/private/tmp/jaxpathways-0170-clean-learner-env/bin/python node resources/content-labs/pretraining-transfer-check.mjs
```

Result: exit 0; all three scripts printed their `PASS: pretraining-*` sentinels. The fixture values above came from that run.

Posttraining generated-script checks used `lessonScript` from `scripts/lesson-artifacts.mjs` to render all four changed lessons (`posttraining-01`, `-02`, `-03`, `-05`) to `/private/tmp/review-posttraining-*.py`, then ran:

```sh
for lesson in 01 02 03 05; do
  /private/tmp/jaxpathways-0170-clean-learner-env/bin/python "/private/tmp/review-posttraining-${lesson}.py" || exit $?
done
```

Result: exit 0; all four scripts printed their `PASS:` sentinels. DPO output retained a failed strict PPO quality gate at each replayed reward scale. All seven JSON files parsed before execution; the affected lesson sources pass `git diff --check`.

## Resolution verification (2026-10-27)

The four findings above were checked against the latest lesson files. All four are resolved in the source reviewed here:

1. **SFT prompt split is disjoint.** Training prompts are `Q:` and `A:`; held-out prompts are `R:` and `B:`. The solution tokenizes each full prompt with the pinned tokenizer, prepends BOS, constructs explicit prompt-group sets, and asserts the sets are disjoint before scoring. The script completed and reported held-out response NLLs for the two disjoint prompts (`[5.8963127, 5.8930640]` before and `[5.8961415, 5.8930955]` after).
2. **LoRA saves trained rank-two factors.** The bounded synthetic target is fit for 300 steps with base weights snapshotted and checked unchanged. Factors `a2` and `b2` from that fit are saved; the script reloads them and asserts byte-value parity with the in-memory trained factors before launching the fresh-process validation. Fit MSE decreased from `0.12098397` to `0.00001915`. The existing child-process checks still pass for alpha/rank `1.5`, merge parity, wrong same-shape base, and altered rank, alpha, target, and factor-shape metadata.
3. **Primary LoRA starter matches the revised exercise.** The lesson-level `learnerStarter` now declares expected base bytes/hash, actually trained factors, target path and factor metadata, changed inputs, manifest-bound output, fresh-process inference, and the requested rejection cases. Its starter signature and TODO describe the child-process and negative-control work. The practice starter also retains the full bundle contract.
4. **DPO disagreement is no longer forced into hard accuracy.** For `vote_fraction == 0.5`, the evaluator emits `hard_accuracy: None` and `ambiguous: True`; it reports predicted probability and expected annotator agreement separately. The generated run printed agreement `0.5` for the tie both before and after the update.

Exact generated-lesson rerun:

```sh
node --input-type=module - <<'JS'
import { readFileSync, writeFileSync } from 'node:fs';
import { lessonScript } from './scripts/lesson-artifacts.mjs';
for (const [id, p] of [
  ['posttraining-01', 'phases/18-posttraining/01-sft/lesson.json'],
  ['posttraining-02', 'phases/18-posttraining/02-lora/lesson.json'],
  ['posttraining-05', 'phases/18-posttraining/05-dpo/lesson.json'],
]) writeFileSync(`/private/tmp/review-latest-${id}.py`, lessonScript(JSON.parse(readFileSync(p, 'utf8'))));
JS
for lesson in 01 02 05; do
  /private/tmp/jaxpathways-0170-clean-learner-env/bin/python "/private/tmp/review-latest-posttraining-${lesson}.py" || exit $?
done
```

Result: exit 0; all three scripts printed their `PASS:` sentinels. The DPO script's existing replay again reports a zero child-minus-parent reward difference at scales 0.5, 1 and 2, with `heldoutRewardFloorPassed` still false and 8/8 response IDs matching. No PPO or cloud run was made.

Current reviewed lesson SHA-256 values:

- `phases/18-posttraining/01-sft/lesson.json`: `14def83d8297a8b34f21681c046d58e4de3f55d496b39a71782a0f46f3fd2409`
- `phases/18-posttraining/02-lora/lesson.json`: `a7b1f24fb1098332d0934c525a81b5293932c7f29aae81acd54d8dc1f214978c`
- `phases/18-posttraining/05-dpo/lesson.json`: `5f08da96cb00389df233380ae47079f8f39a4180725e2308bdef3df4bfe0b2da`
