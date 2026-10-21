# C02 learner contract

Implement `model.py` as the selected decoder candidate. The staged checker loads
that file directly, so the model, loss, cache and generation functions you write
must be the functions called by training, evaluation and the exported consumer.
Keep the public signatures and return raw numerator/count pairs where stated.

`parameter_count` must match the initialized parameter tree, counting tied
embeddings once. `initialize` and `initialize_lora` take explicit JAX keys.
`logits` consumes token IDs plus optional validity, segment and position arrays;
future tokens, padding and other documents cannot affect an active query.
`prefill` returns next-token logits and one K/V cache per layer;
`decode_step` appends one token without mutating a prior cache. Both must agree
with full-forward logits at changed prompt lengths and reject capacity overflow.

Use the two loss interfaces for their distinct input contracts. The aligned
pilot blocks already contain `input_ids`, `target_ids`, `target_mask`,
`segment_ids` and `position_ids`; `aligned_token_loss` must use them directly
without shifting again. `shifted_causal_targets` and `token_loss` are the
ordinary next-token convenience API: they shift once and attach the validity
mask to the target side, excluding padding and cross-document transitions.
Return a summed FP32 loss numerator and the number of active target tokens so
uneven microbatches can be combined by total token count.

For continuation, preserve each prepared aligned block's complete ownership:
valid targets, EOS, document segments and reset positions. The frozen recipe
sets the mixture to 30,720 domain and 7,680 general targets and the matched
general control to 38,400 targets. Do not re-sample, shift or drop a prepared
target. For supervised fine-tuning, `training.sft_batch` builds prompt inputs
and answer-plus-EOS targets from each frozen row. Its target mask must exclude
every prompt token and include every selected answer token and EOS. Check a
short prompt/answer by hand before running the full phase.

LoRA uses the frozen rank-4, alpha-8 rule, so each Q/V delta is `(alpha/rank) A B`
in every decoder block. Initialize B to zero, train adapter leaves only, and
keep base leaves byte-identical. `merge_lora` returns a new parameter tree;
merged and unmerged logits, cache steps and greedy tokens must agree.

The checkpoint APIs in `training.py` are the lifecycle reference: each branch
records its parent and objective, complete Adam state, schedule clock, PRNG,
sampler permutation, epoch and cursor. Stage changes reset only the state frozen
by the recipe. `--stage resume` launches fresh processes to compare the next
update for all six saved branches; use it to test that your candidate is actually
used by the training loss and replay rather than relying on a source hash.

`generate` is greedy cached decoding. It must honor a zero-token budget, return
no more than the requested count, mask PAD, stop immediately on EOS, and match
the independent full-forward oracle for multiple prompts and context boundaries.
`export_consumer.py --implementation <model.py>` packages the chosen candidate;
the cold package recheck validates its copied decoder identity and generated
tokens. The task QA evaluation remains a separate test of the six frozen
checkpoints, and the observed pilot failed its precommitted QA floors.

Recommended order:

1. Implement `parameter_count`, `initialize`, `logits`, shifted/aligned losses,
   then run `--stage model` while changing small fixtures.
2. Implement LoRA and cache APIs; rerun `--stage model` after each behavior change.
3. Prepare and verify the immutable data packet, then run `--stage data`.
4. Run `--stage resume` against all six branches; it must execute candidate code
   in fresh processes and match every saved final update.
5. Run the frozen reloaded evaluation, export the selected candidate, run the
   native harness and consumer checks, then finish with `--stage all` over the
   same data, run, evaluation and package directories.

The reference `projects/causal-model-300m/model.py` is useful for comparison,
but the checker grades the supplied implementation numerically. A copied
starter should fail at its first unfinished API with an actionable `TODO`.
