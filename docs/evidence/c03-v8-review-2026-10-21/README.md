# C03 v8 review: optimization withheld

Sol independently matched 27 frozen source files and the accepted prepared-data
identities. The nine-stage synthetic checker passed; strict real-data preflight
stopped with zero optimizer updates. These are bounded observations, not
acceptance of the complete flower lifecycle.

Independent composition probes and source review found four blocking repairs:

- Interleaved image views were flattened against positives expanded in a
  different order. A simple identical-view case gives loss 5.01339 instead of
  0.01339. The passing stage checks did not exercise this composition.
- Exposure counters inferred masked reconstruction from stage-name substrings,
  including supervised probes, and undercounted actual two-view image work.
- The pipeline evaluated in-memory parameters rather than actually reloading
  saved checkpoints and proving phase recovery.
- The actual stage graph hardwired the reference model rather than demonstrating
  learner-candidate binding. Deployment selection and retained raw comparison
  outputs also need correction before final evaluation acceptance.

Luna is implementing the exact R3–R6 contract in the repair ledger. The next
review requires a complete reduced synthetic stage graph with actual objective
compositions, candidate sentinels, checkpoint/phase resume and independent raw
metric reconstruction, then a new immutable source/recipe binding. The accepted
flower split and declared real-data budgets stay unchanged. No flower-image
optimization or full-capstone maturity promotion is authorized by this packet.

The retained records preserve the source-bound passing test scope and the
counterexamples together. `sha256.json` identifies those exact JSON records.
