# Pretraining transfer content repair notes

P2#84 extends the three existing pretraining lessons with bounded, executable synthetic handoffs. Their original mechanism exercises remain in place. The new blocks write a small `encoder.npz` plus an identity manifest to a fresh temporary directory and print its path; no lesson artifact is represented as a project checkpoint or model release.

The route contracts preserve the existing project evidence boundaries. For text, the saved-embedding pilot references `projects/embedding-model-300m/model.py`, `config.py`, `data/frozen_recipe.json`, and `reference/evidence/final8/pilot-report.json`; its final8 retrieval run failed its predeclared floors and the public packet omits loadable weights and tokenizer files. For images, the route references `projects/image-model-300m/TEACHING.md` and `data/processor.py`; the flower pilot's 0.3185 held-out balanced accuracy missed its 0.40 floor and its runner remains gated. No failed or unperformed project gate was changed. Human review status and curriculum/release maturity metadata were not changed here.

## Verification

The lesson script checker builds the exact generated `lessonScript` for each modified JSON, then executes the whole script in a fresh Python process. This includes build steps, the main example, visuals, all experiments, reference solutions, and practice solutions. The dedicated CPU environment was used to keep the run bounded and offline:

```sh
PYTHON=/private/tmp/jaxpathways-0170-clean-learner-env/bin/python node resources/content-labs/pretraining-transfer-check.mjs
```

Outcome: exit 0; each generated lesson printed its `PASS: pretraining-01`, `PASS: pretraining-02`, or `PASS: pretraining-03` sentinel. The changed-condition outputs were:

- MLM: two-position held-out masked loss 3.57109 versus uniform baseline log(3) = 1.09861; frozen synthetic held-out probe accuracy 0.6667. Both are reported as failure signals for changed-condition transfer on this fixture.
- MIM: held-out alternating-mask reconstruction losses 0.03627 and 0.03820; visible-mean baselines 0.00879 and 0.02441; frozen synthetic probe accuracy 0.75. The simpler visible-mean baseline is better on both masks, a negative result for this reconstruction recipe.
- Contrastive: held-out multi-positive loss 0.01623; Recall@1 and mAP both 1.0 in both the relation-matrix metric path and the separate source-ID ranking oracle. This six-row synthetic result only checks the evaluation implementation.

The earlier direct fresh-namespace pass also executed each main example, exercise solution, and new transfer experiment independently using the same environment:

```sh
JAX_PLATFORMS=cpu /private/tmp/jaxpathways-0170-clean-learner-env/bin/python /tmp/check_pretraining_transfer.py
```

Outcome: all nine main/solution/transfer combinations passed. `/tmp/check_pretraining_transfer.py` was a temporary harness; the checked-in generated-script command above is the reproducible verification entry point.
