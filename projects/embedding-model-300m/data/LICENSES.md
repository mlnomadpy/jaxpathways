# SciFact source and data terms

The C01 retrieval pilot downloads the official SciFact `data.tar.gz` release
from the [AllenAI SciFact repository](https://github.com/allenai/scifact) and
pins its archive bytes, S3 version ID, upstream source commit, source-member
hashes, and normalized output hashes in `frozen_recipe.json` and the generated
external `frozen-manifest.json`.

The upstream [`LICENSE.md`](https://raw.githubusercontent.com/allenai/scifact/master/LICENSE.md)
states separate terms: SciFact claims and evidence annotations are CC BY 4.0;
the scientific abstracts/corpus come from S2ORC under ODC-By 1.0; the upstream
code is Apache-2.0. Cite the SciFact paper and the original upstream project
when reusing the annotations. The corpus retains its S2ORC attribution and
database-rights terms. A model repository's broad metadata is not used to
relicense either source.

The repository retains no downloaded corpus. `data/prepare.py` requires the
exact hash-pinned archive and writes normalized corpus and query files to the
caller-selected external directory. It preserves source IDs, SUPPORT or
CONTRADICT labels, evidence document IDs and rationale sentence IDs. Retrieval
positives come only from `evidence` keys; `cited_doc_ids` are used to build
leakage groups and are never silently converted into relevance labels.

The official test inventory has no public relevance gold. This project uses
official dev as its frozen final evaluation and describes it as a custom
protocol. Its 112 claims with no documented evidence are counted but omitted
from judged retrieval metrics; no negative judgments are invented. Full
corpus candidates remain available, with transductive corpus exposure
disclosed. Final-connected document groups and all official dev query text are
excluded from tokenizer and training input.
