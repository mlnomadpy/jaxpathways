# Audio harness synthesis

Create a report that follows one recording through the whole system. Public reference outputs are available for comparison; they are not evidence of your independent implementation.

1. **Signal contract:** derive frame count, frequency-bin spacing and clip duration from the declared sample rate/window/hop. Compare a silence, an impulse and a non-bin-centered tone with independent NumPy calculations. Explain spectral leakage without calling it a label error.
2. **Learning:** derive a classifier gradient, compare two initializations and fit only training normalization. Compare the energy baseline, per-class recall and overall accuracy. Investigate the held-out error with a gain sweep on a copied diagnostic example while leaving the final test result unchanged.
3. **Recovery:** interrupt before an epoch boundary and restore into a fresh process. Preserve next recording IDs, crop offsets, gains, features, losses, momentum and parameters. Change one optimizer field and one recording label and demonstrate rejection.
4. **Precision:** independently reconstruct W8A8 logits, report rounding and clipping separately, and compare class decisions on fixed held-out inputs. Explain why the FFT remains floating point and why compressed weights do not establish whole-artifact memory or speed improvements.
5. **Release:** reload actual serialized files in a fresh process. Reject a corrupted release, wrong sample rate, wrong channel count and old preprocessing version. Reload a retained known-good release and verify the same golden waveform outputs.
6. **Operations:** report a declared timed boundary and actual samples for batch one and four. Account separately for window accumulation and compute latency. Propose a device experiment with specific runtime/operator/precision checks, while marking it unexecuted.
7. **Real-data transfer:** supply a manifest for recordings you may use, including rights, provenance, hashes, groups, splits and segment offsets. If no permitted dataset is available, submit a concrete ingestion/evaluation plan and label external-data quality unmeasured. Never claim synthetic tone accuracy as speech recognition.

Every figure must name axes, units, scale, plotted data and a limitation. Review each task as accept, revise or not demonstrated; do not infer professional readiness from a reference test pass.
