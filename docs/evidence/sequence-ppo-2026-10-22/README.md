# Synthetic sequence PPO qualification

The [independent review](independent-review.json) covers exact source/recipe identities, scalar and numerical-gradient oracles, fixed rollout snapshots, learner candidates, mutation rejections, the figure and the extracted project ZIP. The raw CPU traces are retained in `root-clean-final3.json` and `extracted-clean.json` beside their execution logs. They contain synthetic task data only.

The first four project stages retain their original Python files. Stage 5 adds a complete six-TODO starter, a separate checker and four pinned direct dependencies. The checker tests learner candidates numerically; source-spelling mutation self-tests apply only to the instructor reference. A completed starter and a differently spelled valid implementation pass. Wrong clipping, EOS/timeout handling and refreshed rollout probabilities fail.

`clean-runtime.json` and `clean-pip-freeze.txt` record the fresh environment. `zip-content-check.json` binds all 14 archived members. `clean-figure-reproduction.json` records byte-for-byte SVG/PNG regeneration from the extracted plotting source and fixture. The earlier project registry is retained in `project-validation-before.json`; only the posttraining-audit entry was refreshed.

This is a small, fixed two-task table-policy exercise. Its held-out instances reuse the same task features. The continuation target is scripted, and categorical KL is exact at sampled contexts rather than for full sequence distributions. Saved Transformer lineage, real preferences, quality/retention, hardware qualification and human teaching review remain open.
