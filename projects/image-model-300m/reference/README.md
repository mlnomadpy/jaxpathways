# C03 v14 numeric evidence packet

This packet documents the completed v14 CPU pilot, including its failed classification-quality gate. It is a source-linked numeric teaching record, not a model release. It contains no photographs, prepared pixel arrays, embeddings, checkpoints, fitted processors, or model weights.

Start with [the teaching guide](../TEACHING.md). The exact v14 recipe and approval bytes are copied here as `evidence/v14/approved-v14-recipe.json` and `approved-v14-approval.json`. Their hashes are checked by `tools/check_v14_packet.py`, together with all 28 recipe source paths and all 11 Python candidate files. The source identity record names the frozen v7 source-manifest/archive/split/array hashes and the separate v11 failed-startup, v13 source-freeze, and v14 run/audit receipts.

`public-numeric-evidence.json` carries ordered dev/final IDs, integer labels, raw logits for all 12 comparison arms, selected final uncalibrated T=1 probabilities, selected corruption-condition logits, and full retrieval rank ordinals over the shared final ID/class/component table. `public-training-traces.json` strips the 384 updates to phase-local losses, sampled IDs, and exposure counters. `omitted-private-artifacts.json` preserves hashes and omission reasons for the private run files. These projections do not make the inspected final set available for tuning.

Run the no-download checker and deterministic SVG builder from the project root:

```sh
python3 tools/check_v14_packet.py
python3 tools/build_v14_figures.py
python3 tools/test_v14_packet_controls.py
```

The SVG/PNG figures and `figure-data.json` are derived from the published arrays. `packet-manifest.json` lists SHA-256 values for the public packet, teaching files, and tools. After authoring changes, regenerate it with `python3 tools/make_packet_manifest.py` and rerun the checker. The public checker and the learner's model/architecture solution are separate; it verifies this evidence packet and does not grade an implementation.
