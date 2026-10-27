# Author review guide: 0.16.0

This candidate adds a synthetic sequence PPO stage to the post-training audit. It does not replace the existing single-action PPO lesson or complete full language-model RLHF. The verified source and evidence are pinned at commit `a0715b45b0caa03729e9d17680f258bbc36e3ef1` on `codex/sequence-ppo-review`, based on draft PR #68. Nothing in this candidate has been merged or deployed.

## Start with the teaching

Open the [project guide](http://127.0.0.1:4200/project-guides/posttraining-audit.html). Predict which bootstrap should survive EOS versus a collection timeout, then read the two-panel figure and its explanation. Check the hand calculation and the counterfactual that sets only the timeout bootstrap to zero. The plot is an independent illustrative calculation, with no KL shaping, rather than a training-performance comparison.

The diagram, inline and display equations, source fixture and plotting script should help you explain each term. Note any undefined term or point where you need an extra example in [your feedback notes](review-notes-0.16.0.md).

## Check the learner journey

Open the [stage workspace](http://127.0.0.1:4200/project.html?id=posttraining-audit). Stages 1–4 use `starter/model.py`; stage 5 names `starter/sequence_ppo.py` and its own checker. Confirm that the edit path, starter excerpt and command agree.

Download the [project ZIP](http://127.0.0.1:4200/downloads/posttraining-audit.zip), extract it, and follow its project-local requirements file from the extracted top folder. The untouched stage 5 starter is expected to fail at a named TODO. It supplies the helper scaffold; the learner implements six documented functions. The public checker accepts an external learner candidate without requiring the reference solution's source spelling.

Keep instructor reference execution separate from your learner implementation:

```sh
python projects/posttraining-audit/tests/check_sequence_ppo.py --implementation solution --stage all --evidence-dir runs/reference
python projects/posttraining-audit/tests/check_sequence_ppo.py --implementation starter --stage 5 --evidence-dir runs/my-attempt
```

Ask whether the function contracts, hints and failure messages are enough to make progress without copying the solution. Automated source and candidate tests establish technical checks; they do not establish that a new learner can follow the teaching.

## Interpret the recorded run

The reference uses 16 outer updates with three inner epochs. Old rollout log probabilities and values stay fixed within an outer update; the current policy and critic change. The fitted reward and initial reference policy stay frozen across the run. Compare the raw trace with the [independent review](evidence/sequence-ppo-2026-10-22/independent-review.json).

The held-out evaluation reuses the same two task features. Its reward improvement and conditional categorical KL are properties of this fixed synthetic fixture. They do not measure unseen semantic generalization, human preference quality, alignment or a Transformer. The timeout continuation is explicitly scripted, and its fitted critic value is imperfect.

## Review the book and scope labels

Open the [book overview](http://127.0.0.1:4200/book.html), then the phase 18 guide in the [HTML reader](http://127.0.0.1:4200/downloads/jax-foundations.html). This edition links the sequence extension and its remaining work. All 104 lesson bodies remain unchanged; phase 18 has a new revision. The project is an available draft with teaching review pending. The full book remains in development, and all ten full capstones remain Planned.

Local qualification passed 177 Node tests, 11 Python identity tests, 52 desktop/mobile Chromium cases, 53 formatter inputs, root/Pages-path link and schema checks, and EPUBCheck 5.4.0 with zero messages. The exact 14-member ZIP also passed in a fresh isolated CPU environment; both figure formats reproduced byte-for-byte there. See the [qualification manifest](evidence/course-review-0.16.0/verification.json) and [raw evidence](evidence/sequence-ppo-2026-10-22/README.md).

The source-bound 0.15.1 PR #68 now has all 15 hosted checks green. Hosted 0.16 verification is a separate pending gate. Native Apple Books, a second EPUB reader, actual assistive-technology speech, full-book print and independent expert/learner teaching reviews remain open. Saved Transformer lineage, real preference provenance, retention and target hardware require further work.
