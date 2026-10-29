"""Dataset identity, train-only byte-level BPE and task-specific pair builders."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_frozen_data(directory: Path):
    directory = Path(directory)
    frozen_recipe = json.loads(
        (Path(__file__).resolve().parent / "frozen_recipe.json").read_text()
    )
    manifest = json.loads((directory / "frozen-manifest.json").read_text())
    recipe = json.loads((directory / "inventories.json").read_text())
    actual = {}
    for name, expected_key in (
        ("corpus.jsonl", "corpus.jsonl_sha256"),
        ("claims_train_original.jsonl", "claims_train_original.jsonl_sha256"),
        ("claims_fit.jsonl", "claims_fit.jsonl_sha256"),
        ("claims_recipe_dev.jsonl", "claims_recipe_dev.jsonl_sha256"),
        ("claims_final.jsonl", "claims_final.jsonl_sha256"),
        (
            "claims_official_test_unjudged.jsonl",
            "claims_official_test_unjudged.jsonl_sha256",
        ),
    ):
        value = digest((directory / name).read_bytes())
        if value != manifest["outputs"][expected_key]:
            raise ValueError(f"frozen dataset file hash mismatch: {name}")
        actual[name] = value
    inventories_hash = digest((directory / "inventories.json").read_bytes())
    if inventories_hash != manifest["outputs"]["inventories.json_sha256"]:
        raise ValueError("frozen inventory file hash mismatch")
    for key, ids in recipe["ids"].items():
        computed = digest(json.dumps(ids, separators=(",", ":")).encode())
        if (
            computed != recipe["sha256"][key]
            or computed != manifest["inventories_sha256"][key]
        ):
            raise ValueError(f"frozen ID inventory hash mismatch: {key}")
    if (
        manifest["source"]["archive_sha256"]
        != frozen_recipe["source"]["archive_sha256"]
    ):
        raise ValueError("prepared data archive does not match project frozen recipe")
    if (
        manifest["source"]["archive_member_sha256"]
        != frozen_recipe["source"]["archive_file_sha256"]
    ):
        raise ValueError(
            "prepared archive source-member hashes do not match project frozen recipe"
        )
    if manifest["inventories_sha256"] != frozen_recipe["inventory_sha256"]:
        raise ValueError(
            "prepared source membership hashes do not match project frozen recipe"
        )
    for filename, expected_hash in frozen_recipe["prepared_output_sha256"].items():
        if manifest["outputs"].get(f"{filename}_sha256") != expected_hash:
            raise ValueError(
                f"prepared output is not bound to project frozen recipe: {filename}"
            )
    corpus = read_jsonl(directory / "corpus.jsonl")
    original_train = read_jsonl(directory / "claims_train_original.jsonl")
    fit = read_jsonl(directory / "claims_fit.jsonl")
    recipe_dev = read_jsonl(directory / "claims_recipe_dev.jsonl")
    final = read_jsonl(directory / "claims_final.jsonl")
    test = read_jsonl(directory / "claims_official_test_unjudged.jsonl")
    if len(corpus) != manifest["counts"]["corpus_documents"]:
        raise ValueError("frozen corpus count mismatch")
    if [int(row["doc_id"]) for row in corpus] != recipe["ids"]["corpus_ids"]:
        raise ValueError("corpus document inventory does not match rows")
    for rows, key in (
        (fit, "fit_query_ids"),
        (recipe_dev, "recipe_development_query_ids"),
        (final, "final_query_ids"),
        (test, "official_test_unjudged_query_ids"),
    ):
        if [int(row["id"]) for row in rows] != recipe["ids"][key]:
            raise ValueError(f"query inventory does not match rows: {key}")
    return {
        "manifest": manifest,
        "frozen_recipe": frozen_recipe,
        "inventories": recipe,
        "corpus": corpus,
        "original_train": original_train,
        "fit": fit,
        "recipe_dev": recipe_dev,
        "final": final,
        "unjudged_test": test,
        "file_hashes": actual,
    }


def document_text(document):
    title = document.get("title", "") or ""
    abstract = " ".join(document.get("abstract") or [])
    return (title + " " + abstract).strip()


def train_byte_bpe(records, output_path: Path, vocab_size=512):
    """Fit a byte-fallback BPE over only the frozen pretraining document IDs."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    texts = [document_text(row) for row in records]
    if not texts or any(not text for text in texts):
        raise ValueError("tokenizer training needs non-empty permitted document text")
    special = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"]
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]", byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.no_padding()
    tokenizer.no_truncation()
    ids = {token: tokenizer.token_to_id(token) for token in special}
    if ids != {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "[MASK]": 4}:
        raise ValueError(f"special-token IDs changed: {ids}")
    raw = tokenizer.to_str().encode()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(raw)
    return tokenizer, {
        "kind": "byte-level-bpe-byte-fallback",
        "vocabulary_size": tokenizer.get_vocab_size(),
        "special_ids": ids,
        "normalization": "none",
        "training_doc_ids": [int(row["doc_id"]) for row in records],
        "training_text_sha256": digest("\n".join(texts).encode()),
        "processor_sha256": digest(raw),
    }


def encode_text(tokenizer, text, max_length):
    encoded = tokenizer.encode(text, add_special_tokens=False).ids
    if any(int(token) == 1 for token in encoded):
        raise ValueError("byte-fallback processor emitted [UNK]")
    values = [2, *map(int, encoded), 3]
    truncated = max(0, len(values) - max_length)
    if truncated:
        values = values[: max_length - 1] + [3]
    return np.asarray(values, dtype=np.int32), truncated


def encode_documents(documents, tokenizer, max_length):
    rows, total_truncated = [], 0
    for document in documents:
        ids, truncated = encode_text(tokenizer, document_text(document), max_length)
        rows.append(
            {"id": int(document["doc_id"]), "text": document_text(document), "ids": ids}
        )
        total_truncated += truncated
    return rows, total_truncated


def encode_queries(claims, tokenizer, max_length):
    rows, total_truncated = [], 0
    for claim in claims:
        ids, truncated = encode_text(tokenizer, claim["claim"], max_length)
        positive_ids = sorted(int(doc_id) for doc_id in claim.get("evidence", {}))
        rows.append(
            {
                "id": int(claim["id"]),
                "text": claim["claim"],
                "ids": ids,
                "positive_doc_ids": positive_ids,
                "evidence": claim.get("evidence", {}),
            }
        )
        total_truncated += truncated
    return rows, total_truncated


def select_pilot_documents(documents, inventory_ids, limit=128):
    allowed = set(map(int, inventory_ids))
    by_id = {int(row["doc_id"]): row for row in documents}
    if not allowed <= by_id.keys():
        raise ValueError("pretraining inventory references absent corpus document")
    selected = sorted(
        allowed,
        key=lambda doc_id: hashlib.sha256(
            f"scifact-c01-pilot-doc-20261021:{doc_id}".encode()
        ).hexdigest(),
    )[:limit]
    return [by_id[doc_id] for doc_id in selected]


def make_weak_pairs(documents):
    """Create explicitly weak same-document title/body and sentence-span pairs."""
    pairs = []
    for document in documents:
        doc_id = int(document["doc_id"])
        title = document.get("title", "") or ""
        sentences = [
            part.strip() for part in (document.get("abstract") or []) if part.strip()
        ]
        if title and sentences:
            pairs.append(
                {
                    "left": title,
                    "right": " ".join(sentences),
                    "group_id": doc_id,
                    "source": "weak_title_abstract",
                }
            )
        if len(sentences) >= 2:
            pairs.append(
                {
                    "left": sentences[0],
                    "right": sentences[-1],
                    "group_id": doc_id,
                    "source": "weak_same_abstract_spans",
                }
            )
    return pairs


def make_human_pairs(claims, corpus):
    """Keep original evidence IDs, labels and rationale sentence IDs intact."""
    docs = {int(row["doc_id"]): row for row in corpus}
    queries = []
    for claim in claims:
        positives = sorted(int(doc_id) for doc_id in claim.get("evidence", {}))
        if not positives:
            continue
        missing = set(positives) - docs.keys()
        if missing:
            raise ValueError(
                f"evidence docs absent from full candidate corpus: {sorted(missing)}"
            )
        queries.append(
            {
                "query_id": int(claim["id"]),
                "query": claim["claim"],
                "positive_doc_ids": positives,
                "evidence": claim["evidence"],
            }
        )
    return queries
