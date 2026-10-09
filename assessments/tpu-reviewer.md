# Training-system synthesis reviewer notes

Use these reference notes to verify byte-tokenization offsets, causal masking, checkpoint recovery, and KV-cache precision.

## Tokens and attention

The tokenizer offsets each UTF-8 byte by three. BOS, EOS and PAD are separate IDs. A multibyte character occupies multiple byte tokens. The loss includes EOS targets and excludes PAD; its denominator is the count of valid predicted tokens, not batch size or padded sequence length.

A scaled attention row uses \(qk^\top/\sqrt{d_h}\), with future positions and padded keys excluded before softmax. The independent reference must use the same layer-normalization epsilon and matrix conventions. Changing future tokens must leave earlier causal outputs unchanged with dropout disabled.

## Recovery

Parameters alone do not preserve Adam continuation, dropout draws or shuffled document order. Require both moments, completed step, random key, order, cursor, epoch, optimizer configuration, tokenizer, corpus identity and checksum. Compare the next several transitions across an epoch boundary in a fresh process. Equal final accuracy is insufficient.

A whole-set loss equals the sum of batch loss sums divided by the total valid-target count. An unweighted average of uneven batch means does not satisfy that invariant.

## Cache and arithmetic

The text reference has one block, two heads, twelve positions and head width twelve. K and V contain \(2\cdot2\cdot12\cdot12=576\) scalar entries. FP32 K/V take \(2304\) bytes, BF16 take \(1152\), and INT8 take \(576\). The unified interface retains an additional \(192\) bytes of FP32 scale arrays, giving \(2496,1344,768\) bytes respectively. These are cache arrays, not peak device memory.

INT8 cache scales are per token/head, independently for K and V. Comparing cached INT8 decoding with an unquantized full reference mixes cache-mechanics error with approximation error. First compare the same policy, then measure the difference from FP32 separately.

Dense W8A8 uses scaled integer dot products with INT32 accumulation, while attention/normalization/output handling remain FP32. Stored cache dtype does not imply every operation uses that dtype or that the target has an accelerated kernel.

## Measured behavior

The fixed CPU reference reports held-out token NLL about 1.071 over 276 valid targets and a training-frequency baseline near 2.352. Several fixed held-out reversal demonstrations succeed, while diagnostic abc| produces cbc instead of cba. Retain both kinds of evidence; a fluent-looking example requires separate verification to establish corpus quality.

Profiler annotations delimit declared operations. Their duration may include Python validation, dispatch and waits. A benchmark that repeats one cached step from the same prefix is verified separately from sustained autoregressive or multi-user server throughput.

## Target qualification

Physical TPU claims need actual TPU device and execution records. Multi-controller claims additionally require all-process participation, global data/count semantics, intended sharding and a committed restart demonstration. The course's logical CPU device exercise and the text harness's single-CPU trace remain useful references but are not replacements for that evidence.

When the target was unavailable, accept the CPU work on its own merits and retain target qualification as not demonstrated. Do not widen tolerances or relabel the platform merely to make the report appear complete.
