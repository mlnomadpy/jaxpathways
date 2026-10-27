# Independent C04 CTC review

This source-only review checks the frozen audio implementation identified in
`result.json`. It enumerates every path over three output symbols rather than
reusing the candidate dynamic program. Thirty target/time cases include empty
and repeated labels, possible and impossible alignments, and right-padded frames.
For possible alignments, loss values and independent NumPy finite differences
agree with the submitted JAX implementation. Padded-frame gradients are zero.
The largest loss error is below 8e-7 and selected gradient error below 9e-8.

The edge probes are separate findings, not passes. A zero valid input length
with an empty target returns ln(3), while the probability of the empty path is
one and its negative log likelihood is zero. A label matrix with zero columns
raises an incidental shape error, including for positive input lengths. The
implementation needs explicit supported-domain checks or correct empty-path
handling before those inputs can be accepted. The corpus workflow currently
uses nonempty audio and no corpus optimization is approved.

Run `independent-check.py` in the audio environment to reproduce. Its absolute
project path identifies the original working source; verify the recorded source
hash first. This is bounded numerical evidence, not ASR quality, target training
or full lifecycle qualification. No trained arrays are included.
