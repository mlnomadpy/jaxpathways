# Corrected C04 inventory and pilot encoder

Root independently checked the corrected immutable handoff after Luna's A1–A3
repairs. All 53 source and external-evidence file identities matched. Previously
verified raw audio/PCM/transcript fields remain unchanged. An independent
metadata-column parser, transcript normalizer, token encoder and adjacency-graph
traversal reconstructed all 5,472 target hashes, 79 source components and two
cross-source components. The source book table contains wrapped continuation
lines; the audit parses numeric book-ID rows and records the skipped nonnumeric
lines. No raw transcript characters were removed by the declared normalizer in
this particular inventory.

A separate NumPy forward oracle uses explicit-index temporal convolution,
affine normalization, GELU, sinusoidal positions, bidirectional attention and
both heads. With nonzero affine biases, mixed valid lengths and padded widths
11, 19, 20 and 21, the frozen pilot matches within about 1.4e-6 absolute error.
Singleton versus mixed-batch valid outputs also agree. An in-memory mutation
restoring shape-dependent SAME padding is rejected with a valid-logit difference
of about 1.30. The implementation files themselves were not mutated.

Another independent frontend oracle constructs the periodic Hann window and
HTK-mel-spaced, Hz-area-normalized triangular filters without receiving candidate
constants. Uncentered framing, FFT, normalized power and natural-log features
match within about 1.3e-6 on synthetic signals at frame boundaries and with ragged
or zero valid lengths. Doubling sine amplitude changes active log-power cells by
ln(4), and a rectangular-window mutation is rejected. This is a 16 kHz feature
calculation check, not a raw codec or anti-alias resampler qualification.

These observations support the corrected grouping/target identities and pilot
inference forward behavior. They do not establish dropout/gradient behavior,
complete stage recovery, trained ASR quality or target-scale execution.
The six-role allocation and schedules in the
handoff remain proposals. Exact role/recipe and full runtime approval are still
required before corpus optimization; the model capstone remains Planned.

Retained checkers name the external source/data paths actually inspected. The
model/config snapshots preserve the checked pilot equations, not a loadable
trained model. `sha256.json` identifies the checkers, snapshots and JSON records.
The earlier rejected inventory and padding counterexample remain in the parent
packet as historical evidence.
