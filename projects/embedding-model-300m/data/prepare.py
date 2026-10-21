"""Freeze SciFact inputs into a reproducible, license-aware local dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

ARCHIVE_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
ARCHIVE_BYTES = 3_115_079
SEED = "scifact-c01-seed-20261021"
FILES = {
    "train": "data/claims_train.jsonl",
    "development": "data/claims_dev.jsonl",
    "unjudged_test": "data/claims_test.jsonl",
    "corpus": "data/corpus.jsonl",
}
MEMBER_SHA256 = {
    "data/claims_train.jsonl": "f4c8fa82d8bd0653a9cc8d61a6ea48c25eacea64e90af5dbf390ebb1b74372f0",
    "data/claims_dev.jsonl": "86f0435d08fdb65d1aa41d1472684f57e6e71930626497bdf4d7a9ec1a632217",
    "data/claims_test.jsonl": "558930d75215c73f84a28fe538307d6d397c9de1ec7239514cf45f80d75d2ca3",
    "data/corpus.jsonl": "b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized(text: str) -> str:
    return " ".join(re.sub(r"\W+", " ", text.casefold()).split())


def _rows(archive: tarfile.TarFile, member: str) -> tuple[list[dict], str]:
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"missing required archive member: {member}")
    raw = stream.read()
    actual = sha256(raw)
    if actual != MEMBER_SHA256[member]:
        raise ValueError(f"official source member hash changed: {member}")
    return [json.loads(line) for line in raw.decode("utf-8").splitlines()], actual


def _evidence_docs(claim: dict) -> set[int]:
    evidence = claim.get("evidence", {})
    result = set()
    for doc_id, annotations in evidence.items():
        if not isinstance(annotations, list) or not annotations:
            raise ValueError(
                f"claim {claim.get('id')} has invalid evidence annotations"
            )
        for annotation in annotations:
            if annotation.get("label") not in {"SUPPORT", "CONTRADICT"}:
                raise ValueError(f"claim {claim.get('id')} has unknown evidence label")
            if not annotation.get("sentences"):
                raise ValueError(
                    f"claim {claim.get('id')} has empty rationale sentences"
                )
        result.add(int(doc_id))
    return result


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def _groups(claims: list[dict], corpus: list[dict]) -> tuple[list[list[int]], dict]:
    corpus_by_id = {int(row["doc_id"]): row for row in corpus}
    if len(corpus_by_id) != len(corpus):
        raise ValueError("duplicate corpus document IDs")
    doc_text_groups: dict[str, list[int]] = defaultdict(list)
    for row in corpus:
        text = normalized(
            row.get("title", "") + " " + " ".join(row.get("abstract") or [])
        )
        doc_text_groups[text].append(int(row["doc_id"]))
    duplicate_doc_groups = [
        sorted(ids) for ids in doc_text_groups.values() if len(ids) > 1
    ]
    doc_group = {doc_id: f"doc:{doc_id}" for doc_id in corpus_by_id}
    for group_index, ids in enumerate(duplicate_doc_groups):
        for doc_id in ids:
            doc_group[doc_id] = f"exactdoc:{group_index}"

    uf = _UnionFind(len(claims))
    owner: dict[str, int] = {}
    linked = []
    for index, claim in enumerate(claims):
        evidence_ids = _evidence_docs(claim)
        cited_ids = {int(value) for value in claim.get("cited_doc_ids", [])}
        linked_ids = evidence_ids | cited_ids
        unknown = linked_ids - corpus_by_id.keys()
        if unknown:
            raise ValueError(
                f"claim {claim['id']} refers to absent documents: {sorted(unknown)[:3]}"
            )
        linked.append(linked_ids)
        keys = [doc_group[doc_id] for doc_id in linked_ids]
        keys.append("claim:" + normalized(claim["claim"]))
        for key in keys:
            if key in owner:
                uf.union(index, owner[key])
            else:
                owner[key] = index
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(claims)):
        components[uf.find(index)].append(index)
    ordered = [
        sorted(indices, key=lambda i: int(claims[i]["id"]))
        for indices in components.values()
    ]
    ordered.sort(key=lambda indices: min(int(claims[i]["id"]) for i in indices))
    info = {
        "exact_duplicate_document_groups": duplicate_doc_groups,
        "components": ordered,
        "linked_doc_ids_by_claim": linked,
        "component_doc_ids": [
            sorted(set().union(*(linked[index] for index in component)))
            for component in ordered
        ],
    }
    return ordered, info


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    ).encode()
    path.write_bytes(payload)
    return sha256(payload)


def prepare(archive_path: Path, output: Path) -> dict:
    raw = archive_path.read_bytes()
    if len(raw) != ARCHIVE_BYTES or sha256(raw) != ARCHIVE_SHA256:
        raise ValueError(
            "archive bytes do not match the frozen official SciFact release"
        )
    with tarfile.open(archive_path, "r:gz") as archive:
        loaded = {key: _rows(archive, member) for key, member in FILES.items()}
        rows = {key: value[0] for key, value in loaded.items()}
        member_hashes = {FILES[key]: value[1] for key, value in loaded.items()}

    corpus = rows["corpus"]
    train = rows["train"]
    final = rows["development"]
    unjudged_test = rows["unjudged_test"]
    for name, values in rows.items():
        key = "doc_id" if name == "corpus" else "id"
        ids = [int(row[key]) for row in values]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate {key} in original {name} inventory")
    corpus_ids = {int(row["doc_id"]) for row in corpus}

    combined_claims = train + final
    components, group_info = _groups(combined_claims, corpus)
    train_count = len(train)
    final_component_ids = {
        component_index
        for component_index, component in enumerate(components)
        if any(index >= train_count for index in component)
    }
    quarantined_indices = sorted(
        index
        for component_index in final_component_ids
        for index in components[component_index]
        if index < train_count
    )
    quarantined_set = set(quarantined_indices)
    direct_final_doc_ids = set().union(
        *(
            _evidence_docs(claim)
            | {int(value) for value in claim.get("cited_doc_ids", [])}
            for claim in final
        )
    )
    final_component_doc_ids = set().union(
        *(set(group_info["component_doc_ids"][index]) for index in final_component_ids)
    )
    if (
        not direct_final_doc_ids <= corpus_ids
        or not final_component_doc_ids <= corpus_ids
    ):
        raise ValueError("final development claim points outside the candidate corpus")
    final_doc_ids = final_component_doc_ids
    remaining = [index for index in range(train_count) if index not in quarantined_set]
    eligible_components = [
        [index for index in component if index < train_count]
        for component_index, component in enumerate(components)
        if component_index not in final_component_ids
    ]
    eligible_components = [component for component in eligible_components if component]
    remaining_claims = train

    ordered_components = []
    for component in eligible_components:
        ids = sorted(int(train[i]["id"]) for i in component)
        digest = hashlib.sha256(
            (SEED + ":" + ",".join(map(str, ids))).encode()
        ).hexdigest()
        ordered_components.append((digest, component))
    ordered_components.sort(key=lambda item: item[0])
    target = round(0.1 * len(remaining))
    development_indices = set()
    for _, component in ordered_components:
        if len(development_indices) >= target:
            break
        development_indices.update(component)
    fit_indices = set(remaining) - development_indices
    fit_claims = [train[i] for i in sorted(fit_indices)]
    development_claims = [train[i] for i in sorted(development_indices)]

    output.mkdir(parents=True, exist_ok=True)
    corpus_hash = _write_jsonl(output / "corpus.jsonl", corpus)
    original_train_hash = _write_jsonl(output / "claims_train_original.jsonl", train)
    fit_hash = _write_jsonl(output / "claims_fit.jsonl", fit_claims)
    recipe_dev_hash = _write_jsonl(
        output / "claims_recipe_dev.jsonl", development_claims
    )
    final_hash = _write_jsonl(output / "claims_final.jsonl", final)
    test_hash = _write_jsonl(
        output / "claims_official_test_unjudged.jsonl", unjudged_test
    )

    linked_by_train = [
        _evidence_docs(claim) | {int(value) for value in claim.get("cited_doc_ids", [])}
        for claim in train
    ]
    pretraining_doc_ids = corpus_ids - final_doc_ids
    id_sets = {
        "corpus_ids": sorted(corpus_ids),
        "fit_query_ids": sorted(int(row["id"]) for row in fit_claims),
        "recipe_development_query_ids": sorted(
            int(row["id"]) for row in development_claims
        ),
        "final_query_ids": sorted(int(row["id"]) for row in final),
        "official_test_unjudged_query_ids": sorted(
            int(row["id"]) for row in unjudged_test
        ),
        "final_heldout_doc_ids": sorted(final_doc_ids),
        "fit_train_doc_ids": sorted(
            set().union(*(linked_by_train[i] for i in fit_indices))
        ),
        "recipe_dev_train_doc_ids": sorted(
            set().union(*(linked_by_train[i] for i in development_indices))
        ),
        "pretraining_doc_ids": sorted(pretraining_doc_ids),
    }
    id_hashes = {
        key: sha256(json.dumps(value, separators=(",", ":")).encode())
        for key, value in id_sets.items()
    }
    id_path = output / "inventories.json"
    id_path.write_text(
        json.dumps({"ids": id_sets, "sha256": id_hashes}, indent=2, sort_keys=True)
        + "\n"
    )

    def label_counts(claims):
        labels = {"SUPPORT": 0, "CONTRADICT": 0}
        links = 0
        judged = 0
        for claim in claims:
            evidence = claim.get("evidence", {})
            judged += bool(evidence)
            links += len(evidence)
            for entries in evidence.values():
                for entry in entries:
                    labels[entry["label"]] += 1
        return {
            "claims": len(claims),
            "judged_queries": judged,
            "positive_evidence_doc_links": links,
            "human_label_instances": labels,
        }

    report = {
        "protocol": "scifact-c01-custom-grouped-fit-dev-final-v1",
        "source": {
            "archive_url": "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
            "archive_sha256": ARCHIVE_SHA256,
            "archive_bytes": len(raw),
            "upstream_repo": "https://github.com/allenai/scifact",
            "upstream_commit": "68b98a56d93e0f9da0d2aab4e6c3294699a0f72e",
            "archive_version_id": "8LiW3OUBLBvhehDzrRdGD1ibIaXL.6PQ",
            "archive_member_sha256": member_hashes,
            "license_file_sha256": "56ee69bc639cbab8f0b2e4b67091ec5e9390aa99a1d65f0a35e4274686c9a806",
            "schema_file_sha256": "f8f4cc2f7bf32347dd47d419be731a5d0eeb60ebb75423619e02919c7f337df7",
            "licenses": {
                "claims_and_evidence": "CC BY 4.0",
                "abstract_corpus": "S2ORC ODC-By 1.0",
                "upstream_code": "Apache-2.0",
            },
        },
        "selection": {
            "normalization": "Unicode casefold; replace non-word runs with spaces; collapse whitespace",
            "final_protocol": "all official dev claims; only claims with evidence document IDs count as judged retrieval queries",
            "positive_definition": "document IDs under claims[*].evidence; SUPPORT and CONTRADICT both count as evidence retrieval positives",
            "cited_doc_ids_policy": "used only for group quarantine and training-group construction; never treated as positive judgments",
            "heldout_docs_removed_from_train": len(final_doc_ids),
            "training_group_rule": "union all official train and final claims sharing exact-normalized query text or cited/evidence document IDs, with exact-normalized duplicate corpus texts grouped first; quarantine every train member of a component containing a final claim",
            "train_recipe_dev_seed": SEED,
            "train_recipe_dev_rule": "sort component SHA-256 hashes; add whole components until >= round(10% of remaining train claims)",
            "corpus_policy": "all 5,183 official corpus docs remain candidates; candidate abstracts include transductive exposure, but final query text and heldout cited/evidence document groups are excluded from tokenizer, MLM and weak-pair training",
            "weak_labels": "weak span/title-abstract links are never represented as human evidence annotations",
            "no_positive_queries": "excluded from retrieval metric denominator and counted explicitly; never assigned an invented negative",
        },
        "counts": {
            "corpus_documents": len(corpus),
            "official_train": label_counts(train),
            "quarantined_train_claims_touching_final_components": len(
                quarantined_indices
            ),
            "fit": label_counts(fit_claims),
            "recipe_development": label_counts(development_claims),
            "official_dev_final": label_counts(final),
            "official_dev_without_evidence_excluded": sum(
                not row.get("evidence") for row in final
            ),
            "official_test_unjudged": {
                "claims": len(unjudged_test),
                "has_public_gold": False,
            },
            "heldout_final_cited_or_evidence_doc_ids": len(final_doc_ids),
            "exact_duplicate_document_groups": len(
                group_info["exact_duplicate_document_groups"]
            ),
            "fit_components": sum(
                1 for _, comp in ordered_components if set(comp) <= fit_indices
            ),
            "recipe_dev_components": sum(
                1 for _, comp in ordered_components if set(comp) <= development_indices
            ),
            "transitive_or_normalized_query_train_quarantine": len(quarantined_indices)
            - sum(
                bool(
                    (
                        set(map(int, claim.get("cited_doc_ids", [])))
                        | _evidence_docs(claim)
                    )
                    & direct_final_doc_ids
                )
                for claim in train
            ),
            "final_direct_cited_or_evidence_doc_ids": len(direct_final_doc_ids),
        },
        "outputs": {
            "corpus.jsonl_sha256": corpus_hash,
            "claims_train_original.jsonl_sha256": original_train_hash,
            "claims_fit.jsonl_sha256": fit_hash,
            "claims_recipe_dev.jsonl_sha256": recipe_dev_hash,
            "claims_final.jsonl_sha256": final_hash,
            "claims_official_test_unjudged.jsonl_sha256": test_hash,
            "inventories.json_sha256": sha256(id_path.read_bytes()),
        },
        "inventories_sha256": id_hashes,
    }
    report_path = output / "frozen-manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def download_archive(destination: Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
        headers={"User-Agent": "jaxpathways-sciFact-c01/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
    if len(raw) != ARCHIVE_BYTES or sha256(raw) != ARCHIVE_SHA256:
        raise ValueError(
            "downloaded latest SciFact archive is not the frozen byte release"
        )
    destination.write_bytes(raw)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, help="already downloaded official frozen data.tar.gz"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="fetch the official archive at the URL frozen by its SHA-256",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="external directory for normalized frozen data",
    )
    args = parser.parse_args()
    archive = args.archive
    if args.download:
        if archive is None:
            archive = (
                Path(tempfile.gettempdir()) / "jaxpathways-scifact-c01-data.tar.gz"
            )
        archive = download_archive(archive)
    elif archive is None:
        default_archive = (
            Path(tempfile.gettempdir()) / "jaxpathways-scifact-c01-data.tar.gz"
        )
        archive = (
            default_archive
            if default_archive.is_file()
            else download_archive(default_archive)
        )
    print(json.dumps(prepare(archive, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
