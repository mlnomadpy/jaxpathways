# Audio harness authoring and integration evidence

## Delivered connected project

projects/audio-harness contains a four-stage learner contract, starter, reference, independent checks, a connected run.py, an explanatory guide, bundled synthesis/reviewer tasks, and actual generated plots, checkpoint, exports and report.

The task is synthetic short background/low-tone/high-tone classification. It is explicitly not speech recognition, natural-audio validation or target-edge execution.

Implemented lifecycle:

1. Recording identities/groups and separate train/held/shifted seeds; actual PCM16 WAV ingestion with required rights/provenance/checksums and cross-split group leakage rejection.
2. Raw 8 kHz mono PCM contract; fixed 1024-sample inference windows; 128-sample symmetric Hann STFT, 64-sample hop, 15 frames, 65 real-frequency bins; log-power features and training-only normalization.
3. A genuinely trained three-class classifier with stable objective, independent gradient reference, momentum SGD and explicit crop/gain/noise augmentation.
4. Complete checkpoints including momentum, normalization, RNG, order, cursor, epoch, step, configuration and recording-data identity. Fresh-process tests compare five updates across an epoch boundary, including actual IDs/crops/gains/features/losses/state.
5. Fixed held-out evaluation, count-weighted uneven batches, confusion matrix, non-mutating evaluation, energy baseline, changed initialization and separately generated frequency/noise-shifted recordings.
6. Training-only activation calibration, per-class weight scales, explicit weight/feature/accumulator/output policies and independently checked INT32 arithmetic.
7. Six genuine JAX exports containing waveform preprocessing, normalization and classification: FP32/W8A32/W8A8 at batch one/four. Fresh-process reload/parity, corrupted-release rejection and known-good backup reload.
8. Validated inference and actual completed CPU request timings. Capture, window accumulation, file decoding, network, peak memory and target-device performance remain explicitly unmeasured.

## Commands executed

```sh
python3 projects/audio-harness/tests/check.py --implementation solution --stage all
python3 projects/audio-harness/run.py --implementation solution
```

The all-stage reference passed. It invokes separate Python processes for checkpoint resume and exported inference; those are real subprocess tests, not merely calls to load in the existing process.

The runner independently executes the full pipeline, interrupts at step 40, resumes through step 80 and compares subsequent traces/state. It writes outputs/report.json, two SVG/PNG figures, a short generated WAV error example, a complete checkpoint and six serialized artifacts.

The figures were visually inspected. The guide explains axes, units, frame-center support, spectrum scale, actual confusion counts, noise in the training curve and limitations. Report source_sha256/runner_sha256 bind the saved run to the reference and runner files. Timing values are actual local CPU observations and vary across runs.

Environment: Python 3.14.3, JAX 0.9.2, NumPy 2.4.4, one CPU device. The project uses the existing course CPU dependencies; no external recordings, GPU/TPU, network endpoint or target edge hardware was used.

## Observed results

- Fixed held-out recordings: 68/69 correct; mean cross-entropy about 0.0825687.
- Confusion matrix, true rows / predicted columns: [[23,0,0],[0,23,0],[1,0,22]].
- Energy-only baseline: 46/69 correct.
- Changed model initialization: 68/69 correct.
- Separate frequency/noise-shifted fixture: 72/72 correct. This is not a paired causal experiment and does not establish that noise improves accuracy.
- W8A32 and W8A8: both 68/69 correct on the fixed held-out set.
- Maximum logit differences versus FP32: about 0.0314822 for W8A32 and 0.0468423 for W8A8.
- W8A8 clips one normalized held-out feature value beyond the training calibration range.
- Plotted error held-recording-77-0026 is a true high tone predicted as background. Its spectrum peaks at 1500 Hz; RMS amplitude is about 0.1438 versus a training high-tone median of 0.3837. The guide treats amplitude sensitivity as a hypothesis to test, not a proven sole cause.

Generated artifacts live under projects/audio-harness/outputs. Keep report.json and images with the project ZIP so their README links resolve.

## Exact integration mapping

Add projects/audio-harness/project.json to curriculum/projects.json and audio-harness to course.projectIds. The project pathwayId is ship, status authored. It is a modality integration project; it should not overwrite deployment-audit as the general ship capstone.

The project includes ASSESSMENT.md and reviewer.md in its own bundle. No shared assessment registry was changed. If exposing this synthesis as a standalone assessment page, use a distinct audio identifier rather than replacing ship.

For curriculum/modality-tracks.json, audio track:

```json
{
  "projectIds": ["audio-harness"],
  "status": "authored",
  "prerequisites": "Shared foundations, neural-network training, explicit recovery and deployment contracts. The audio project supplies a connected waveform/STFT tutorial and lifecycle exercise.",
  "available": "A complete CPU waveform-to-inference harness is runnable: trained spectral classifier, full fresh-process recovery, fixed evaluation, calibrated precision and complete exported artifacts. A PCM WAV ingestion path is available; natural-recording quality and target-edge performance remain unmeasured.",
  "firstLessonId": "arrays-02"
}
```

The current root validator constrains modality track status to guided-plan; root must migrate that status contract together with display/types before setting authored, or use guided-plan temporarily with truthful runnable availability. Do not leave the visible statement that the audio-specific harness is planned.

Stage-to-project mapping:

| Modality lifecycle stage | Concrete project evidence                                                     |
| ------------------------ | ----------------------------------------------------------------------------- |
| data                     | Stage 1 waveform contracts, WAV manifest, split/group checks                  |
| model                    | Stages 1–2 independent STFT and classifier gradient; energy baseline          |
| train                    | Stage 2 momentum updates with crop/gain/noise ownership                       |
| recover                  | Stage 2 fresh-process checkpoint replay across an epoch boundary              |
| evaluate                 | Stage 2 fixed confusion matrix, uneven-batch equality, changed seed/data      |
| precision                | Stage 3 train-only scales and independent integer arithmetic                  |
| export                   | Stage 3 six full raw-waveform exports and fresh-process reload                |
| operate                  | Stage 4 completed CPU requests, malformed-input drills and separate edge plan |

Suggested audio plot text: “Inspect the actual held-out error as waveform and spectrogram with a shared seconds axis. Color is log(1 + window-normalized power), not a probability. Read the confusion matrix by true-class rows and predicted-class columns; explain the high-tone/background error before interpreting aggregate accuracy.”

## Boundaries and follow-up evidence

The classifier averages spectra over time and therefore discards temporal order. Its small fixed-window task does not implement speech recognition, streaming overlap state or event-boundary localization. The real-data adapter was tested with a generated PCM WAV; a permitted external collection still requires actual rights and measured held-out quality. FFT and log features stay floating-point in all policies. Integer dot-product correctness does not prove native acceleration on a target device.

Human editorial/learner review, a real-data experiment, sustained device tests and production service behavior remain separate evidence requirements. Root must register the project and regenerate/distribute assets and run the complete registered-project suite.
