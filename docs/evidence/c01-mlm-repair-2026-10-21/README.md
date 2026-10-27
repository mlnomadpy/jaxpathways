# C01 masked-token grader repair

Independent review accepts the reference and a harmless equivalent, and rejects
constant-zero MLM logits, disconnected encoder, disconnected head and disconnected
MLP down-projection gradients. The first repair had an overly permissive absolute
tolerance; the accepted checker uses independent two-step finite differences and
requires a meaningful sensitivity relative to its comparison tolerance.

Root ran the full current reference on the frozen SciFact data and exact offline
MiniLM snapshot. It passed model checks, four fresh-process resume stages, actual
training and independent retrieval/package checks. The same constant-zero candidate
previously accepted by the complete checker now exits at raw-logit comparison,
before any resume/training output. The two verification JSON records and amendment
are retained under `projects/embedding-model-300m/reference/evidence/` and indexed
for the source archive. Historical final8 and the earlier attention amendment remain
unchanged. Retrieval quality still fails its frozen floors; no full capstone,
300M training, physical runtime, human or public trained-model release is qualified.
