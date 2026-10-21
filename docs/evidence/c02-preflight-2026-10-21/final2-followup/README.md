# C02 independent evaluation and local-service follow-up

These records extend the earlier training-state inspection. They do not amend
the frozen 0.13 review candidate or qualify the full causal capstone.

## Accepted observations

- Independent scoring reconstructed all 768 saved predictions, 128 per branch.
  Every question/answer, prompt, prompt token sequence, decoded output, finish
  condition and checkpoint identity matched the frozen final2 records. Every
  branch failed both predeclared QA floors. This audit did not independently
  regenerate the outputs or reconstruct held-out NLL logits.
- The complete frozen exporter CLI rejected seven individually altered receipt
  bindings before creating an output directory: recipe, tokenizer, evaluator,
  scorer, checkpoint bytes, checkpoint identity and candidate identity. The
  unchanged failed-quality checkpoint remained exportable for local learning.
  The positive package manifest matched
  `8a44debff5a58a7640f40e0404f955d38190df45b3c92ab962c8f74d82044220`.
- The actual v2 portable package cold-started from an unrelated directory,
  generated two tokens through its localhost endpoint, rejected a Boolean token
  budget and served health while another client's header was incomplete.
  Its owned process stopped, and checked source hashes remained unchanged.
- A separate parser-only test of frozen v3 source enforced a controlled 150 ms
  admission budget on incomplete headers and bodies, returning 408 in about
  151 and 156 ms. Two admitted partial clients produced a correctly framed 503
  for a third client; health recovered after release. Its owned listener stopped.

## Preserved repair history and limits

The v2 source expired an incomplete body correctly, but returned 413 for an
incomplete POST header because it validated body length before the expired
lease. `transport-v2-finding.json` preserves that failed independent assertion.
The separately retained v3 snapshot moves the lease check first and passes the
same header/body canaries plus overload recovery. Do not combine a v2 package
identity with a v3 parser result.

The parser test uses no model inference and a deliberately reduced timeout.
The portable package smoke uses real pilot inference with its default timeout.
Neither proves hard cancellation of an active native JAX operation: generation
cancellation is cooperative between bounded steps. Complete pipeline rollback,
measured latency distributions, native benchmark runs, learner stages and the
remaining full project gates need separate evidence.

`sha256.json` identifies the retained checkers, source snapshots and result files.
Checker paths name the external local data, checkpoints and source snapshots
used during review. Those private runtime directories and trained arrays are
not included here. The original final2 metrics, generations and accepted
training checkpoints remain unchanged; their digests are in the records.
