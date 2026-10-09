# Training-system synthesis: causal model, recovery and runtime evidence

**Scope:** causal Transformer training, full-state checkpoint recovery, mixed precision, serialized inference, and profiler trace analysis using the [text-harness project](../project.html?id=text-harness). **TODO (Cloud TPU VM Qualification):** Execute the multi-device training and trace collection stages on a Cloud TPU VM slice.

Create a report and preserve data/tokenizer identity, reference implementation, checkpoints, trace files, serialized endpoints and actual outputs.

## Task 1: explain the objective and information boundary

Use the text-harness byte tokenizer and a short document with a multibyte UTF-8 character. Show raw bytes, token IDs, BOS/EOS/PAD, shifted inputs/targets and the valid-target mask.

Derive one attention row independently from embeddings, layer normalization, query/key matrices and the head-dimension scale. Demonstrate that a changed future token cannot affect earlier logits. Check the token-normalized loss with independent NumPy arithmetic and compare one parameter derivative with a central difference.

Freeze a held-out split by document identity, source group and content before tuning. Explain why perplexities from incompatible tokenizers or different masking policies are not directly comparable.

**Keep:** token diagram, independent attention/loss calculations, causal failure drill and explicit denominator.

## Task 2: recover the experiment, not just its weights

Train the actual Transformer. Interrupt before an epoch boundary and resume in a fresh process. Compare at least four subsequent updates spanning the boundary: document IDs, dropout keys, losses, Adam moments, parameters, order and cursor.

Deliberately restore with changed data and changed optimizer configuration, and require rejection. Explain why restarting Adam from zero or reseeding dropout is a different experiment even when the next loss happens to look plausible.

Evaluate in uneven batches and verify equality with whole-set sums/counts. Check that evaluation leaves training state unchanged. Train a changed initialization and report the result without hiding a failed seed.

**Keep:** full checkpoint manifest, uninterrupted/restored traces, state comparison and held-out token evidence.

## Task 3: verify the cache and its precision

Implement real prefill and incremental decoding. Append a fixed sequence one token at a time and compare cached results with full causal recomputation after each append. Inspect that old valid cache entries remain unchanged. Reject a full-context append instead of wrapping the position index.

Compare actual FP32, BF16 and INT8 K/V arrays. For INT8, independently reconstruct one token/head vector from integer codes and its scale. Measure bytes including scale metadata and compare logits with the FP32 model.

Then compare dense computation policies separately: rounded BF16 operands and dynamic W8A8 dense arithmetic with INT32 accumulation. State which embedding, normalization, attention, reduction and output operations remain FP32. Do not label the complete model “fully integer” when only selected boundaries are quantized.

**Keep:** per-step parity, dtype/scale evidence, cache-memory accounting, precision error and an overflow failure.

## Task 4: inspect actual exported execution

Serialize and reload separate prefill/decode endpoints in another Python process. Compare fixed token inputs under each declared policy, reject a corrupted artifact and run bounded generation. Report token IDs and decoded text together; a changed sampled or greedy token can follow a small numeric logit change.

Measure synchronized prefill and single-token cached decode with a declared prompt length. Keep first calls and at least thirty warm samples. Explain what repeated one-step timings exclude compared with a complete multi-request service.

Record a real local profiler trace containing training, prefill and cached-decode annotations. Locate those regions in the trace, connect one observed interval to the code boundary, and distinguish host overhead from completed computation. Keep the trace and actual observations instead of substituting a diagram of expected behavior.

**Keep:** byte hashes, runtime parity, timings with units, real trace and explained generation successes/failures.

## Task 5: qualify a target runtime separately

Write a concrete TPU qualification protocol referencing welcome-03 and distributed-04. In an available supported runtime, execute the explicit platform check and retain device/platform/version evidence. For multi-controller work, every controller must initialize before device discovery and use the intended shared checkpoint contract.

Keep the same tokenizer, dataset identities and global valid-token objective when moving the model. Specify global versus per-device batch, parameter/state placement, collective normalization, committed checkpoint boundary and failure/restart verification. Compare a small result with the independent CPU reference before collecting performance claims.

If no physical target is available, mark this task **not demonstrated on target** and submit the exact executable commands and expected evidence. A CPU training-system pass does not become a TPU qualification by changing its title.

## Review

For each task mark accept, revise or not demonstrated. An accepted CPU artifact needs actual source-bound outputs and independent checks. Physical target acceptance additionally needs the named device/process evidence, not screenshots of a CPU result inside a cloud notebook.

After your attempt, read [reviewer notes](tpu-reviewer.md).
