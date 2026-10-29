<!-- jaxpathways-content-audit:2026-10-27:rl-discount -->

Audited October 27, 2026, edition 0.17.1, PR #71 head `15bf7aa4fbd4460501cc4d6d7802324673a89f6f`. Priority P1. Finding type: confirmed-correction.

## Teaching problem

The code fits its value table to `returns_to_go`, an undiscounted reverse cumulative sum, and then feeds this table to GAE with `gamma=0.9`. The lesson does not introduce this as an intentional mismatched value approximation or explain its effect on TD residuals.

A deterministic two-transition example exposes the difference. Rewards `[1,1]`, terminal V=0, and undiscounted values `[2,1,0]` give TD residuals `[-0.1,0]` at gamma=.9. The matched discounted values `[1.9,1,0]` give `[0,0]`. Lower regression error on undiscounted targets does not make this a matched discounted critic.

This does not mean GAE may never use an approximate critic, or that gamma must always equal one. The position-only value table's missing remaining-time state is already disclosed and should remain disclosed. The [GAE paper](https://arxiv.org/html/1506.02438v6) distinguishes discounted value definitions and discount/variance choices.

## Acceptance criteria

- [ ] State the objective, discount, horizon and value target before fitting the table.
- [ ] Either fit discounted targets consistent with gamma, use gamma=1 for the stated undiscounted task, or explicitly teach the mismatch as a controlled approximation with its consequences.
- [ ] Add the two-transition hand calculation and terminal versus cutoff cases with independent expected values.
- [ ] Keep critic-fit improvement distinct from policy improvement. If adding an actor-critic handoff, compare on held-out rollout seeds and an explicit budget.

## Audited source

- [rl-05](https://github.com/mlnomadpy/jaxpathways/blob/15bf7aa4fbd4460501cc4d6d7802324673a89f6f/phases/12-rl/05-learn-values-and-gae/lesson.json)

This issue is a source-level content review. Existing CPU receipts and honest toy/replay/target limitations remain valid within their stated scope. This audit did not rerun all training programs, commission human reviews or launch paid hardware. Keep content maturity, physical-device qualification and model-quality evidence separate.
