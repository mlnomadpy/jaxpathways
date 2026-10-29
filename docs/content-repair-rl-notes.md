# RL-05 reverse GAE diagram notes

The lesson now has a deterministic `content.diagram` mechanism figure alongside the existing executed critic-loss plot. It uses the already declared hand fixture (`gamma=0.5`, `lambda=0.5`) so its numeric labels stay distinct from the critic fit, which uses discounted targets and GAE `gamma=0.9`.

The figure shows the transition from position 0 to position 1, the collector cutoff at final pre-reset position 3, the separate reset observation at position 0, and a true-terminal alternative. Its sole reverse arrow runs from `A1` to `A0` with `gamma * lambda * trace_mask = 0.25`. The cutoff node labels both continuing-task and finite-deadline bootstrap choices. The reset node has no incoming reverse edge across the cutoff.

The text equivalent gives the numeric table: continuing cutoff has `delta1=A1=3` and `A0=1.75`; the finite-horizon deadline has `delta1=A1=1` and `A0=1.25`; a true terminal has zero bootstrap and residual 1. The changed-input prompt asks what changes when a continuing collector cutoff becomes the actual task deadline. The existing position-only value table limitation and critic-loss caveat remain in the lesson.

## Verification

- Executed the complete RL-05 lesson code, reference solution, and all experiment snippets with `/private/tmp/jaxpathways-0170-clean-learner-env/bin/python` on CPU.
- Asserted the computed continuing-cutoff GAE `[1.75, 3, 0]`, finite-horizon GAE `[1.25, 1, 0]`, and terminal advantage `1` against the labels in the diagram.
- Asserted graph structure: reverse edge from `t=1` to `t=0`; no edge from the reset observation; separate cutoff, reset and terminal nodes with the expected masks.
- Rendered and visually inspected the mechanism SVG/PNG in `/private/tmp/rl05-reverse-gae-figure-check/rl-05/`. These are temporary previews; regenerate the repository artifacts after the lesson content is finalized.
