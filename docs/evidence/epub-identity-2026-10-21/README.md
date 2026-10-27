# EPUB publication identity and release preparation

The new explicit edition key identifies a publication across its minor revisions. A pinned UTC modification time distinguishes revisions, while a separate deterministic fingerprint covers rendered payload content and order. Legacy releases retain their previous identity fallback. This is a packaging policy change and is not evidence that the earlier shared identifier caused an Apple Books import problem.

The release preparation command validates the new fields before writing an immutable snapshot. Independent checks passed 11 focused Python tests, distribution checks, and three actual release-command tests. Invalid metadata leaves absent or existing snapshots untouched. Native import of the new edition remains to be tested after its final build.
