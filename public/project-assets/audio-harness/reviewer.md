# Audio harness reviewer notes

The reference contract yields \((1024-128)/64+1=15\) frames, \(128/2+1=65\) real-frequency bins, \(62.5\) Hz spacing and \(0.128\) seconds per input. Frame centers, not left edges, label the spectrogram time positions. A symmetric Hann window suppresses edge discontinuities but spreads a sinusoid's energy into neighboring bins; the transform does not promise one nonzero frequency bin for every tone.

For normalized features \(Z\), logits \(ZW+b\), probabilities \(P\) and one-hot labels \(Y\), gradients are \(Z^\top(P-Y)/N\) and the row mean of \(P-Y\). Fitted normalization is part of the artifact. Evaluating a held-out set must not refit it or change optimizer/random state.

The observed matrix [[23,0,0],[0,23,0],[1,0,22]] means one true high tone was classified as background. Overall accuracy is \(68/69\); high-tone recall is \(22/23\). The higher score on another shifted sample is not a controlled demonstration that noise helps.

A complete resume includes optimizer memory, all randomness, ordering/cursor, normalization and data/preprocessing identity. Equal final accuracy is much weaker than equality of the subsequent recording IDs, crop offsets, feature arrays, losses and state. The public test compares five updates across an epoch boundary in a new Python process.

Symmetric activation scale \(a\) and per-class weight scale \(s_j\) reconstruct a quantized dot as \(a s_j\sum_i q_{x,i}q_{w,ij}+b_j\). Bias is FP32 after rescaling. Clipping invalidates a rounding-only bound for values outside the calibrated range. FFT/log processing stays FP32 in every provided policy. An actual integer dot does not establish an accelerated integer execution path on another device.

The runtime boundary begins with already decoded PCM. Input accumulation takes 128 milliseconds from an empty stream for this fixed-window task; overlapping streaming windows can change throughput and response semantics and need a separately specified state contract. File decoding, capture, network, queueing, peak memory, energy and thermal effects are not measured by the reference timer.

Accept only claims supported by preserved artifacts, independent checks and observed results. Licensing/provenance fields are a record of the user's evidence, not automatic proof that a recording is permitted.
