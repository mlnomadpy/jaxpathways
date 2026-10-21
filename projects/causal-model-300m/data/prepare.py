"""Download, verify, group and freeze the C02 public text sources.

This preparer writes source and prepared data beneath the requested output path.
It intentionally keeps downloaded corpus bytes outside the course repository.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

WIKI_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SQUAD_REVISION = "eee5fdbf62f8613a7812b03419e6b29617b74fd1"
SOURCES = {
    "wiki_train": {
        "url": f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{WIKI_REVISION}/wikitext-2-raw-v1/train-00000-of-00001.parquet",
        "path": "wikitext-2-raw-v1/train-00000-of-00001.parquet",
        "sha256": "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7",
        "bytes": 6357543,
    },
    "wiki_validation": {
        "url": f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{WIKI_REVISION}/wikitext-2-raw-v1/validation-00000-of-00001.parquet",
        "path": "wikitext-2-raw-v1/validation-00000-of-00001.parquet",
        "sha256": "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c",
        "bytes": 657209,
    },
    "wiki_test": {
        "url": f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{WIKI_REVISION}/wikitext-2-raw-v1/test-00000-of-00001.parquet",
        "path": "wikitext-2-raw-v1/test-00000-of-00001.parquet",
        "sha256": "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91",
        "bytes": 732610,
    },
    "squad_train": {
        "url": f"https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/{SQUAD_REVISION}/dataset/train-v1.1.json",
        "path": "squad-1.1/train-v1.1.json",
        "sha256": "3527663986b8295af4f7fcdff1ba1ff3f72d07d61a20f487cb238a6ef92fd955",
        "bytes": 30288272,
    },
    "squad_dev": {
        "url": f"https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/{SQUAD_REVISION}/dataset/dev-v1.1.json",
        "path": "squad-1.1/dev-v1.1.json",
        "sha256": "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9",
        "bytes": 4854279,
    },
}
WIKI_LICENSE = "CC BY-SA 3.0 and GFDL, as listed by the dataset authors."
SQUAD_LICENSE = "CC BY-SA 4.0, as stated by the SQuAD authors."
SEED = "c02-wikitext-squad-freeze-20261021"
VOCAB_SIZE = 512
CONTEXT = 512
MAX_NEW_TOKENS = 32
MAX_FINAL = 128
MIN_FINAL = 64
MIN_FINAL_ARTICLES = 16
BLOCK_TARGETS = 512
SFT_UPDATES = 32
BATCH_SIZE = 4
QA_TEMPLATE = "Question: {question}\nContext: {context}\nAnswer:"
PYTHIA_REVISION = "d31f659f077f9ea500c4d7faa999893d7f28cbdf"
PYTHIA_FILES_SHA256 = {
    "README.md": "faea6edf8dbdc9ea6a2ed558e091aa810aeecf2b3804cd1235b48f526044cae8",
    "config.json": "f97f966a66c444890ed461fff2a51eefb15d74303df05b948124719f199b0b17",
    "generation_config.json": "494cbba887cc299273f0c54d7612596918e74c7881a27cbd95e4e7e2464ea090",
    "model.safetensors": "116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1",
    "special_tokens_map.json": "10b8c8852c1e1f70b54d9aff61728408c28971c0e97a6c5a7b2debbd1d3e9c0c",
    "tokenizer.json": "870f4e2baa6b683221fa52004d5d6f40ab8c9d31961617304b78c910c2c3caf2",
    "tokenizer_config.json": "eee017c5bd133137f45907bd0a6e781e2ccd1a533734b7ed2a2f2f4446659809",
}
LM_EVAL_VERSION = "0.4.13"
LM_EVAL_COMMIT = "ddd67220430a2470529f25fd5c05a576ca1057a0"


def qa_prompt(question: str, context: str) -> str:
    """Render the one frozen question template used by filtering and training."""
    return QA_TEMPLATE.format(question=question, context=context)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_json(value) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def model_text(text: str) -> str:
    """Collapse source whitespace while preserving Unicode, case and punctuation."""
    return " ".join(text.split())


def _word_ngram_hashes(text: str, width: int = 13) -> set[bytes]:
    words = re.findall(r"[a-z0-9]+", text.casefold())
    return {
        hashlib.sha256("\x1f".join(words[start : start + width]).encode()).digest()[:8]
        for start in range(max(0, len(words) - width + 1))
    }


def _load_or_download(output: Path, download: bool) -> dict[str, Path]:
    raw_root = output / "source"
    result = {}
    for key, info in SOURCES.items():
        path = raw_root / info["path"]
        if not path.is_file():
            if not download:
                raise FileNotFoundError(
                    f"missing source; rerun with --download: {path}"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(
                info["url"], headers={"User-Agent": "jaxpathways-c02-preparer/1"}
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                path.write_bytes(response.read())
        content = path.read_bytes()
        if len(content) != info["bytes"] or sha256(content) != info["sha256"]:
            raise ValueError(
                f"pinned source mismatch for {key}: {len(content)} bytes, SHA256 {sha256(content)}"
            )
        result[key] = path
    return result


def _wiki_articles(path: Path, split: str) -> list[dict]:
    table = pq.read_table(path, columns=["text"])
    lines = [value.as_py() or "" for value in table.column("text")]
    # WikiText marks article headings as ` = Name = ` and subsections as
    # ` = = Name = = `. Keep the source line and neighbor indices as parser proof.
    article_heading = re.compile(r"^\s*=\s+([^=].*?)\s+=\s*$")
    subsection_heading = re.compile(r"^=\s*=.*=\s*=$")
    articles = []
    current = None
    for row_index, line in enumerate(lines):
        match = article_heading.match(line)
        if match and not subsection_heading.match(line.strip()):
            if current:
                articles.append(current)
            current = {
                "source": "wikitext",
                "split": split,
                "title": match.group(1).strip(),
                "record_id": f"wiki:{split}:{row_index}",
                "source_row_start": row_index,
                "lines": [line.rstrip("\n")],
            }
        elif current is not None:
            current["lines"].append(line.rstrip("\n"))
    if current:
        articles.append(current)
    for article in articles:
        article["text"] = "\n".join(article["lines"])
        article["model_text"] = model_text(article["text"])
        article["normalized_title"] = normalized(article["title"])
        article["normalized_text"] = normalized(article["text"])
        article["paragraphs"] = [
            " ".join(line.split())
            for line in article["lines"]
            if len(normalized(line)) >= 40
        ]
        article["normalized_paragraphs"] = sorted(
            {normalized(line) for line in article["paragraphs"]}
        )
        article["source_row_end_exclusive"] = article["source_row_start"] + len(
            article["lines"]
        )
        article["parser_context"] = {
            "heading_line": article["lines"][0],
            "preceding_line": (
                lines[article["source_row_start"] - 1]
                if article["source_row_start"]
                else None
            ),
            "following_line": (
                lines[article["source_row_start"] + 1]
                if article["source_row_start"] + 1 < len(lines)
                else None
            ),
            "subsection_lines_retained": sum(
                bool(subsection_heading.match(line.strip()))
                for line in article["lines"]
            ),
        }
        article.pop("lines")
    return articles


def _squad_articles(path: Path, split: str) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != "1.1":
        raise ValueError(
            f"expected SQuAD 1.1, found version {payload.get('version')!r}"
        )
    articles = []
    seen_ids = set()
    for article_index, article in enumerate(payload["data"]):
        title = article["title"].strip()
        paragraphs = []
        for para_index, paragraph in enumerate(article["paragraphs"]):
            context = paragraph["context"]
            questions = []
            for qa in paragraph["qas"]:
                qid = str(qa["id"])
                if qid in seen_ids:
                    raise ValueError(f"duplicate SQuAD question id {qid}")
                seen_ids.add(qid)
                answers = [answer["text"] for answer in qa["answers"]]
                if not answers or any(not answer for answer in answers):
                    raise ValueError(f"SQuAD 1.1 question has no answer: {qid}")
                questions.append(
                    {"id": qid, "question": qa["question"], "answers": answers}
                )
            paragraphs.append(
                {
                    "paragraph_id": f"{split}:{article_index}:{para_index}",
                    "context": context,
                    "normalized_context": normalized(context),
                    "questions": questions,
                }
            )
        articles.append(
            {
                "source": "squad",
                "split": split,
                "title": title,
                "normalized_title": normalized(title),
                "record_id": f"squad:{split}:{article_index}",
                "paragraphs": paragraphs,
            }
        )
    return articles


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def _components(articles: list[dict]) -> list[list[int]]:
    uf = UnionFind(len(articles))
    owners: dict[tuple[str, str], int] = {}
    for index, article in enumerate(articles):
        keys = (
            [("title", article["normalized_title"])]
            if article["normalized_title"]
            else []
        )
        if article["source"] == "wikitext":
            keys.extend(
                ("context", paragraph) for paragraph in article["normalized_paragraphs"]
            )
        else:
            for paragraph in article["paragraphs"]:
                if paragraph["normalized_context"]:
                    keys.append(("context", paragraph["normalized_context"]))
                keys.extend(
                    ("question", normalized(item["question"]))
                    for item in paragraph["questions"]
                )
        for key in keys:
            if key in owners:
                uf.union(index, owners[key])
            else:
                owners[key] = index
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(articles)):
        groups[uf.find(index)].append(index)
    return sorted(
        (sorted(group) for group in groups.values()), key=lambda group: min(group)
    )


def _train_bpe(texts: list[str], output: Path) -> dict:
    processor = Tokenizer(models.BPE(unk_token="[UNK]", byte_fallback=True))
    processor.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
    processor.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    processor.train_from_iterator(texts, trainer=trainer)
    ids = {
        name: processor.token_to_id(token)
        for name, token in zip(
            ("pad", "bos", "eos", "unk"), ("[PAD]", "[BOS]", "[EOS]", "[UNK]")
        )
    }
    if ids != {"pad": 0, "bos": 1, "eos": 2, "unk": 3}:
        raise ValueError(f"special token ids changed: {ids}")
    processor_json = processor.to_str().encode()
    path = output / "tokenizer.json"
    path.write_bytes(processor_json)
    probe = "München Ω 🦉 unseen🧪"
    encoded = processor.encode(probe, add_special_tokens=False)
    if processor.decode(encoded.ids) != probe or 3 in encoded.ids:
        raise ValueError("byte-fallback Unicode/unseen-character probe failed")
    byte_count = len(probe.encode("utf-8"))
    token_count = len(encoded.ids)
    return {
        "kind": "bytelevel-bpe",
        "version": "tokenizers==0.22.2",
        "vocabulary_size": processor.get_vocab_size(),
        "special_ids": ids,
        "normalization": "input whitespace collapse in model_text(); Tokenizers normalizer none",
        "prefix_space": False,
        "byte_fallback": True,
        "unknown_policy": "byte fallback; [UNK] id is reserved and must not appear",
        "bos_eos_policy": "builders prepend [BOS] and append [EOS] per independent record; no processor post-processor",
        "processor_sha256": sha256(processor_json),
        "probe": {
            "text": probe,
            "utf8_bytes": byte_count,
            "bpe_tokens": token_count,
            "round_trip": True,
        },
    }


def _build_target_blocks(
    records: list[dict], processor: Tokenizer, output: Path, prefix: str
) -> dict:
    """Build fixed target blocks with explicit segment-local BOS/position resets."""
    blocks = []
    inputs: list[int] = []
    targets: list[int] = []
    segment_ids: list[int] = []
    position_ids: list[int] = []
    segments = []
    seen_targets = 0

    def flush() -> None:
        nonlocal inputs, targets, segment_ids, position_ids, segments
        if not targets:
            return
        if len(targets) != BLOCK_TARGETS:
            raise ValueError("attempted to emit a partial target block")
        arrays = {
            "input_ids": np.asarray(inputs, dtype=np.int32),
            "target_ids": np.asarray(targets, dtype=np.int32),
            "segment_ids": np.asarray(segment_ids, dtype=np.int32),
            "position_ids": np.asarray(position_ids, dtype=np.int32),
            "target_mask": np.ones((BLOCK_TARGETS,), dtype=np.bool_),
        }
        block_id = f"{prefix}:{len(blocks):06d}"
        blocks.append(
            {
                "block_id": block_id,
                "valid_targets": BLOCK_TARGETS,
                "segments": segments,
                "arrays": arrays,
            }
        )
        inputs, targets, segment_ids, position_ids, segments = [], [], [], [], []

    for record in sorted(records, key=lambda row: row["id"]):
        text_ids = processor.encode(
            model_text(record["text"]), add_special_tokens=False
        ).ids + [2]
        seen_targets += len(text_ids)
        offset = 0
        while offset < len(text_ids):
            room = BLOCK_TARGETS - len(targets)
            take = min(room, len(text_ids) - offset)
            piece = text_ids[offset : offset + take]
            local_segment = len(segments)
            # Zero is reserved for padding by the decoder attention mask.
            segment_id = local_segment + 1
            inputs.extend([1] + piece[:-1])
            targets.extend(piece)
            segment_ids.extend([segment_id] * take)
            position_ids.extend(range(take))
            segments.append(
                {
                    "record_id": record["id"],
                    "target_start": offset,
                    "target_end_exclusive": offset + take,
                    "target_count": take,
                    "reset_input_to_bos": True,
                    "position_start": 0,
                }
            )
            offset += take
            if len(targets) == BLOCK_TARGETS:
                flush()
    tail = len(targets)
    # No incomplete tail is fed to a fixed-size pilot schedule. Its exact token
    # count and IDs remain disclosed by the source inventory.
    output_index = output / f"{prefix}-block-index.jsonl"
    index_bytes = b"".join(
        (
            json.dumps(
                {k: v for k, v in block.items() if k != "arrays"},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        for block in blocks
    )
    output_index.write_bytes(index_bytes)
    npz_path = output / f"{prefix}-blocks.npz"
    np.savez_compressed(
        npz_path,
        input_ids=np.stack([block["arrays"]["input_ids"] for block in blocks]),
        target_ids=np.stack([block["arrays"]["target_ids"] for block in blocks]),
        segment_ids=np.stack([block["arrays"]["segment_ids"] for block in blocks]),
        position_ids=np.stack([block["arrays"]["position_ids"] for block in blocks]),
        target_mask=np.stack([block["arrays"]["target_mask"] for block in blocks]),
    )
    expected_targets = sum(
        len(processor.encode(model_text(row["text"]), add_special_tokens=False).ids) + 1
        for row in records
    )
    if (
        expected_targets != seen_targets
        or len(blocks) * BLOCK_TARGETS + tail != seen_targets
    ):
        raise ValueError("packed target accounting lost or duplicated source tokens")
    return {
        "block_count": len(blocks),
        "block_target_width": BLOCK_TARGETS,
        "valid_targets_per_block": BLOCK_TARGETS,
        "total_source_targets_including_eos": seen_targets,
        "complete_block_targets": len(blocks) * BLOCK_TARGETS,
        "incomplete_tail_targets_not_used_in_fixed-block_pilot": tail,
        "coverage_invariant": "complete target-token stream prefix appears once in block arrays; incomplete terminal tail is counted but not trained",
        "npz": {
            "path": npz_path.name,
            "bytes": npz_path.stat().st_size,
            "sha256": sha256(npz_path.read_bytes()),
        },
        "index": {
            "path": output_index.name,
            "bytes": output_index.stat().st_size,
            "sha256": sha256(index_bytes),
        },
        "blocks": [
            {k: v for k, v in block.items() if k != "arrays"} for block in blocks
        ],
    }


def prepare(output: Path, download: bool) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    source_paths = _load_or_download(output, download)
    wiki = []
    for key, split in (
        ("wiki_train", "train"),
        ("wiki_validation", "validation"),
        ("wiki_test", "test"),
    ):
        wiki.extend(_wiki_articles(source_paths[key], split))
    squad_train = _squad_articles(source_paths["squad_train"], "train")
    squad_dev = _squad_articles(source_paths["squad_dev"], "dev")
    articles = wiki + squad_train + squad_dev
    components = _components(articles)
    component_for = {}
    component_id = {}
    component_roles = {}
    wiki_eval_nodes = {
        i
        for i, item in enumerate(articles)
        if item["source"] == "wikitext" and item["split"] in {"validation", "test"}
    }
    squad_dev_nodes = {
        i
        for i, item in enumerate(articles)
        if item["source"] == "squad" and item["split"] == "dev"
    }
    train_component_keys = set()
    for group in components:
        key = sha256("\n".join(articles[i]["record_id"] for i in group).encode())
        for i in group:
            component_for[i] = group
            component_id[i] = key
        if any(i in wiki_eval_nodes or i in squad_dev_nodes for i in group):
            role = "quarantine-final-evaluation-overlap"
        elif any(
            articles[i]["source"] == "squad"
            and articles[i]["split"] == "train"
            and int(sha256((SEED + articles[i]["record_id"]).encode())[:8], 16) % 10
            == 0
            for i in group
        ):
            role = "recipe-development"
        else:
            role = "fit"
            train_component_keys.add(key)
        component_roles[key] = role
    eligible_wiki = [
        item
        for i, item in enumerate(articles)
        if item["source"] == "wikitext"
        and item["split"] == "train"
        and component_roles[component_id[i]] == "fit"
    ]
    if not eligible_wiki:
        raise ValueError(
            "no eligible WikiText train articles remain after component quarantine"
        )
    # Fit byte-level BPE on eligible base text only. The reversible byte baseline
    # provides a tokenizer-independent length/coverage reference.
    tokenizer_info = _train_bpe([item["model_text"] for item in eligible_wiki], output)
    processor = Tokenizer.from_file(str(output / "tokenizer.json"))
    training_bytes = sum(
        len(item["model_text"].encode("utf-8")) for item in eligible_wiki
    )
    training_bpe_tokens = sum(
        len(processor.encode(item["model_text"], add_special_tokens=False).ids)
        for item in eligible_wiki
    )
    tokenizer_info["training_only_length_comparison"] = {
        "record_count": len(eligible_wiki),
        "normalized_utf8_bytes": training_bytes,
        "reversible_byte_baseline_tokens": training_bytes,
        "byte_baseline_coverage": "all UTF-8 bytes have one token ID; exact round trip",
        "bpe_tokens": training_bpe_tokens,
        "bpe_tokens_per_utf8_byte": training_bpe_tokens / training_bytes,
        "bpe_unknown_token_count": 0,
        "bpe_training_round_trip": "all eligible training articles decode to their normalized model_text",
    }
    for item in eligible_wiki:
        encoded = processor.encode(item["model_text"], add_special_tokens=False)
        if 3 in encoded.ids or processor.decode(encoded.ids) != item["model_text"]:
            raise ValueError(f"BPE training round trip failed for {item['record_id']}")
    final_candidates = []
    for article in squad_dev:
        for paragraph in article["paragraphs"]:
            context = model_text(paragraph["context"])
            for question in paragraph["questions"]:
                question_text = model_text(question["question"])
                prompt = qa_prompt(question_text, context)
                question_tokens = len(
                    processor.encode(prompt, add_special_tokens=False).ids
                )
                # Eligibility is input-only and leaves BOS plus the full frozen
                # generation budget inside the same context window.
                if question_tokens <= CONTEXT - 1 - MAX_NEW_TOKENS:
                    final_candidates.append(
                        {
                            "article_title": article["title"],
                            "article_id": article["record_id"],
                            "paragraph_id": paragraph["paragraph_id"],
                            "question_id": question["id"],
                            "question": question_text,
                            "context": context,
                            "answers": question["answers"],
                            "prompt_tokens": question_tokens,
                        }
                    )
    by_final_title: dict[str, list[dict]] = defaultdict(list)
    for row in final_candidates:
        by_final_title[row["article_title"]].append(row)
    for rows in by_final_title.values():
        rows.sort(
            key=lambda row: (
                sha256((SEED + row["question_id"]).encode()),
                row["question_id"],
            )
        )
    ordered_titles = sorted(
        by_final_title, key=lambda title: (sha256((SEED + title).encode()), title)
    )
    # Round-robin over article groups prevents a popular article from consuming
    # the entire small custom evaluation set.
    final = []
    for round_index in range(
        max((len(rows) for rows in by_final_title.values()), default=0)
    ):
        for title in ordered_titles:
            rows = by_final_title[title]
            if round_index < len(rows) and len(final) < MAX_FINAL:
                final.append(rows[round_index])
    titles = {row["article_title"] for row in final}
    if len(final) < MIN_FINAL or len(titles) < MIN_FINAL_ARTICLES:
        raise ValueError(
            f"custom final subset below frozen minimum: {len(final)} questions across {len(titles)} articles"
        )
    # Record all components, including quarantined/long rows, so membership and
    # non-exposure can be independently checked from stable IDs.
    memberships = []
    for index, article in enumerate(articles):
        item = {
            "record_id": article["record_id"],
            "source": article["source"],
            "split": article["split"],
            "title": article["title"],
            "normalized_title": article["normalized_title"],
            "component_sha256": component_id[index],
            "role": component_roles[component_id[index]],
        }
        if article["source"] == "wikitext":
            item["raw_text_sha256"] = sha256(article["text"].encode())
            item["parser_context"] = article["parser_context"]
            item["title_line"] = article["parser_context"]["heading_line"]
        else:
            item["question_ids"] = [
                q["id"] for p in article["paragraphs"] for q in p["questions"]
            ]
            item["paragraph_ids"] = [p["paragraph_id"] for p in article["paragraphs"]]
            item["question_count"] = len(item["question_ids"])
            item["context_sha256s"] = sorted(
                {sha256(p["context"].encode()) for p in article["paragraphs"]}
            )
        memberships.append(item)
    # The external output contains the used model text and QA examples. The raw
    # source inputs remain in source/ and are hash checked on every preparation.
    train_rows = [
        {
            "id": row["record_id"],
            "source": row["source"],
            "title": row["title"],
            "text": row["model_text"],
        }
        for row in eligible_wiki
    ]
    cont_rows = []
    fit_qa_candidates = []
    recipe_dev_rows = []
    for index, article in enumerate(articles):
        role = component_roles[component_id[index]]
        if article["source"] != "squad" or article["split"] != "train":
            continue
        if role == "recipe-development":
            for paragraph in article["paragraphs"]:
                recipe_dev_rows.extend(
                    {
                        "article_title": article["title"],
                        "paragraph_id": paragraph["paragraph_id"],
                        **question,
                        "question": model_text(question["question"]),
                        "context": model_text(paragraph["context"]),
                    }
                    for question in paragraph["questions"]
                )
        elif role == "fit":
            for paragraph in article["paragraphs"]:
                cont_rows.append(
                    {
                        "id": f"{article['record_id']}:{paragraph['paragraph_id']}",
                        "source_paragraph_id": paragraph["paragraph_id"],
                        "article_title": article["title"],
                        "text": model_text(paragraph["context"]),
                    }
                )
                fit_qa_candidates.extend(
                    {
                        "article_title": article["title"],
                        "paragraph_id": paragraph["paragraph_id"],
                        "id": question["id"],
                        "question": model_text(question["question"]),
                        "context": model_text(paragraph["context"]),
                        "answers": question["answers"],
                    }
                    for question in paragraph["questions"]
                )
    fit_qa_rows = []
    sft_length_excluded = []
    for row in fit_qa_candidates:
        prompt = qa_prompt(row["question"], row["context"])
        prompt_ids = processor.encode(prompt, add_special_tokens=False).ids
        target_answer = model_text(row["answers"][0])
        answer_ids = processor.encode(target_answer, add_special_tokens=False).ids
        total_positions = 1 + len(prompt_ids) + len(answer_ids) + 1
        if total_positions > CONTEXT:
            sft_length_excluded.append(
                {
                    "id": row["id"],
                    "article_title": row["article_title"],
                    "prompt_tokens": len(prompt_ids),
                    "first_answer_tokens": len(answer_ids),
                    "required_positions_with_bos_eos": total_positions,
                    "reason": "full prompt plus first original answer and EOS exceeds context; no answer or context cropping",
                }
            )
            continue
        fit_qa_rows.append(
            {
                **row,
                "target_answer": target_answer,
                "prompt_tokens": len(prompt_ids),
                "response_tokens_with_eos": len(answer_ids) + 1,
                "required_positions_with_bos_eos": total_positions,
            }
        )
    outputs = {
        "base-train.jsonl": train_rows,
        "continuation-domain.jsonl": cont_rows,
        "sft-fit.jsonl": fit_qa_rows,
        "sft-fit-excluded.jsonl": sft_length_excluded,
        "qa-recipe-dev.jsonl": recipe_dev_rows,
        "qa-final.jsonl": final,
    }
    block_inventories = {
        "base": _build_target_blocks(train_rows, processor, output, "base-train"),
        "continuation_domain": _build_target_blocks(
            cont_rows, processor, output, "continuation-domain"
        ),
        "continuation_general": _build_target_blocks(
            train_rows, processor, output, "continuation-general"
        ),
    }
    block_orders = {
        name: sorted(
            inventory["blocks"],
            key=lambda block: (
                sha256((SEED + name + block["block_id"]).encode()),
                block["block_id"],
            ),
        )
        for name, inventory in block_inventories.items()
    }
    if (
        len(block_orders["base"]) < 128
        or len(block_orders["continuation_domain"]) < 60
        or len(block_orders["continuation_general"]) < 75
    ):
        raise ValueError(
            "prepared training sources do not contain enough complete target blocks for the frozen schedule"
        )
    sft_order = sorted(
        fit_qa_rows,
        key=lambda row: (sha256((SEED + "sft:" + row["id"]).encode()), row["id"]),
    )[: SFT_UPDATES * BATCH_SIZE]
    if len(sft_order) != SFT_UPDATES * BATCH_SIZE:
        raise ValueError("not enough full-context SFT examples for the frozen schedule")
    pilot_batches = []
    base_selected = block_orders["base"][: 32 * BATCH_SIZE]
    for update in range(32):
        blocks = base_selected[update * BATCH_SIZE : (update + 1) * BATCH_SIZE]
        pilot_batches.append(
            {
                "branch": "base",
                "update": update,
                "block_ids": [row["block_id"] for row in blocks],
                "valid_target_count": sum(row["valid_targets"] for row in blocks),
            }
        )
    domain_selected = block_orders["continuation_domain"][:60]
    general_selected = block_orders["continuation_general"][:75]
    for update in range(15):
        domain_blocks = domain_selected[update * 4 : (update + 1) * 4]
        general_block = general_selected[update * 5 : update * 5 + 1]
        pilot_batches.append(
            {
                "branch": "continuation_mixture",
                "update": update,
                "domain_block_ids": [row["block_id"] for row in domain_blocks],
                "general_block_ids": [row["block_id"] for row in general_block],
                "domain_valid_target_count": 4 * BLOCK_TARGETS,
                "general_valid_target_count": BLOCK_TARGETS,
                "valid_target_count": 5 * BLOCK_TARGETS,
                "gradient_accumulation_microbatches": [4, 1],
            }
        )
        control_blocks = general_selected[update * 5 : (update + 1) * 5]
        pilot_batches.append(
            {
                "branch": "continuation_general_control",
                "update": update,
                "block_ids": [row["block_id"] for row in control_blocks],
                "valid_target_count": 5 * BLOCK_TARGETS,
                "gradient_accumulation_microbatches": [4, 1],
            }
        )
    sft_step_rows = []
    for update in range(SFT_UPDATES):
        rows = sft_order[update * BATCH_SIZE : (update + 1) * BATCH_SIZE]
        step_info = {
            "update": update,
            "question_ids": [row["id"] for row in rows],
            "response_valid_target_count": sum(
                row["response_tokens_with_eos"] for row in rows
            ),
        }
        sft_step_rows.append(step_info)
        for branch in (
            "sft_full_from_domain",
            "sft_lora_from_domain",
            "sft_full_from_general",
        ):
            pilot_batches.append({"branch": branch, **step_info})
    outputs["pilot-batches.jsonl"] = pilot_batches
    pilot_batch_bytes = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in pilot_batches
    )
    (output / "pilot-batches.jsonl").write_bytes(pilot_batch_bytes)
    block_selection = {
        "base_block_ids": [row["block_id"] for row in base_selected],
        "continuation_domain_block_ids": [row["block_id"] for row in domain_selected],
        "continuation_general_block_ids": [row["block_id"] for row in general_selected],
        "sft_question_ids": [row["id"] for row in sft_order],
        "base_valid_targets": len(base_selected) * BLOCK_TARGETS,
        "continuation_mixture_domain_valid_targets": len(domain_selected)
        * BLOCK_TARGETS,
        "continuation_mixture_general_valid_targets": 15 * BLOCK_TARGETS,
        "continuation_mixture_total_valid_targets": 75 * BLOCK_TARGETS,
        "continuation_general_control_valid_targets": 75 * BLOCK_TARGETS,
        "sft_response_valid_targets": sum(
            row["response_tokens_with_eos"] for row in sft_order
        ),
        "sft_response_target_ids_sha256": sha256(
            "\n".join(row["id"] for row in sft_order).encode()
        ),
        "sft_branch_batches_sha256": sha256(pilot_batch_bytes),
    }
    fit_squad_articles = [
        article
        for i, article in enumerate(articles)
        if article["source"] == "squad"
        and article["split"] == "train"
        and component_roles[component_id[i]] == "fit"
    ]
    train_title_keys = {item["normalized_title"] for item in eligible_wiki} | {
        item["normalized_title"] for item in fit_squad_articles
    }
    heldout_title_keys = {item["normalized_title"] for item in squad_dev}
    wiki_eval_title_keys = {
        item["normalized_title"]
        for item in articles
        if item["source"] == "wikitext" and item["split"] in {"validation", "test"}
    }
    title_audit = {
        "normalization": "casefold and retain ASCII alphanumeric tokens, joined by one space",
        "eligible_training_title_keys": len(train_title_keys),
        "official_dev_title_keys": len(heldout_title_keys),
        "exact_normalized_train_to_final_title_aliases": len(
            train_title_keys & heldout_title_keys
        ),
        "matched_title_keys": sorted(train_title_keys & heldout_title_keys),
        "wikitext_validation_test_title_keys": len(wiki_eval_title_keys),
        "exact_normalized_train_to_wikitext_retention_title_aliases": len(
            train_title_keys & wiki_eval_title_keys
        ),
        "wikitext_retention_matched_title_keys": sorted(
            train_title_keys & wiki_eval_title_keys
        ),
    }
    training_ngram_hashes: set[bytes] = set()
    for article in eligible_wiki:
        training_ngram_hashes.update(_word_ngram_hashes(article["model_text"]))
    for article in fit_squad_articles:
        for paragraph in article["paragraphs"]:
            training_ngram_hashes.update(
                _word_ngram_hashes(model_text(paragraph["context"]))
            )
            for question in paragraph["questions"]:
                training_ngram_hashes.update(
                    _word_ngram_hashes(model_text(question["question"]))
                )
                for answer in question["answers"]:
                    training_ngram_hashes.update(_word_ngram_hashes(model_text(answer)))
    final_ngram_hashes: set[bytes] = set()
    for row in final:
        final_ngram_hashes.update(_word_ngram_hashes(row["context"]))
        final_ngram_hashes.update(_word_ngram_hashes(row["question"]))
    overlap_audit = {
        "kind": "exact normalized 13-word gram intersection, SHA-256-truncated-to-64-bit identities",
        "normalization": "casefold, split into ASCII alphanumeric tokens; punctuation and Unicode outside ASCII alphanumerics do not participate",
        "training_sources": [
            "eligible WikiText train article text",
            "eligible SQuAD train passage/question/original answer text",
        ],
        "final_sources": ["selected SQuAD dev contexts and questions"],
        "unique_training_grams": len(training_ngram_hashes),
        "unique_final_grams": len(final_ngram_hashes),
        "shared_final_grams": len(training_ngram_hashes & final_ngram_hashes),
        "final_gram_overlap_fraction": (
            len(training_ngram_hashes & final_ngram_hashes) / len(final_ngram_hashes)
            if final_ngram_hashes
            else 0.0
        ),
        "interpretation": "An exact lexical overlap screen only; it cannot establish absence of semantic or paraphrase contamination.",
    }
    output_hashes = {}
    for name, rows in outputs.items():
        data = b"".join(
            (
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            ).encode()
            for row in rows
        )
        (output / name).write_bytes(data)
        output_hashes[name] = {
            "rows": len(rows),
            "sha256": sha256(data),
            "bytes": len(data),
        }
    source_manifest = {
        "dataset_protocol": "C02 WikiText-2 raw article pretraining plus filtered SQuAD 1.1 passage continuation and answer-span response-only SFT",
        "seed": SEED,
        "preparer": {
            "path": "projects/causal-model-300m/data/prepare.py",
            "sha256": sha256(Path(__file__).read_bytes()),
            "python": "3.14.3",
            "pyarrow": version("pyarrow"),
            "tokenizers": version("tokenizers"),
        },
        "sources": {
            "wikitext": {
                "repository": "Salesforce/wikitext",
                "revision": WIKI_REVISION,
                "configuration": "wikitext-2-raw-v1",
                "license": WIKI_LICENSE,
                "license_source": "https://huggingface.co/datasets/Salesforce/wikitext",
                "attribution": "WikiText authors: Stephen Merity, Caiming Xiong, James Bradbury, Richard Socher; derived from Wikipedia articles.",
                "files": {
                    key: {
                        **{k: v for k, v in info.items() if k != "url"},
                        "download_url": info["url"],
                        "repository_path": info["path"],
                    }
                    for key, info in SOURCES.items()
                    if key.startswith("wiki_")
                },
            },
            "squad": {
                "repository": "rajpurkar/SQuAD-explorer",
                "revision": SQUAD_REVISION,
                "version": "1.1",
                "license": SQUAD_LICENSE,
                "license_source": "https://rajpurkar.github.io/SQuAD-explorer/",
                "attribution": "SQuAD authors: Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang; passages from Wikipedia; questions and answer spans from human annotators.",
                "files": {
                    key: {
                        **{k: v for k, v in info.items() if k != "url"},
                        "download_url": info["url"],
                        "repository_path": info["path"],
                    }
                    for key, info in SOURCES.items()
                    if key.startswith("squad_")
                },
            },
        },
        "normalization": "Source files remain byte-identical. Model-stream text collapses each article/passage's whitespace runs to one ASCII space and trims ends; Unicode, case and punctuation are preserved. Cross-source duplicate keys separately use casefolded alphanumeric tokens. Byte denominators count UTF-8 bytes of this normalized model stream, not original raw-source byte spans.",
        "article_parser": {
            "wikitext_top_heading_regex": r"^\s*=\s+([^=].*?)\s+=\s*$",
            "subsection_regex": r"^\s*=\s*=.*=\s*=$",
            "subsections": "retained within owning top-level article",
            "proof": "membership rows retain source title line, adjacent lines, row range and subsection count",
        },
        "source_file_hashes": {
            key: {
                "sha256": sha256(path.read_bytes()),
                "bytes": path.stat().st_size,
                "relative_path": str(path.relative_to(output)),
            }
            for key, path in source_paths.items()
        },
        "prepared_outputs": output_hashes,
        "tokenizer": tokenizer_info,
        "title_alias_audit": title_audit,
        "overlap_audit": overlap_audit,
    }
    manifest_bytes = stable_json(source_manifest)
    (output / "source-manifest.json").write_bytes(manifest_bytes)
    (output / "memberships.json").write_bytes(
        stable_json(
            {
                "articles": memberships,
                "final_question_ids": [row["question_id"] for row in final],
                "final_article_titles": sorted(titles),
            }
        )
    )
    counts = {
        "wikitext_articles_by_split": {
            split: sum(
                item["source"] == "wikitext" and item["split"] == split
                for item in articles
            )
            for split in ("train", "validation", "test")
        },
        "squad_articles_by_split": {
            split: sum(
                item["source"] == "squad" and item["split"] == split
                for item in articles
            )
            for split in ("train", "dev")
        },
        "squad_questions_by_split": {
            split: sum(
                len(p["questions"])
                for item in articles
                if item["source"] == "squad" and item["split"] == split
                for p in item["paragraphs"]
            )
            for split in ("train", "dev")
        },
        "articles_by_group_role": {
            role: sum(
                component_roles[component_id[index]] == role
                for index in range(len(articles))
            )
            for role in (
                "fit",
                "recipe-development",
                "quarantine-final-evaluation-overlap",
            )
        },
        "components_by_group_role": {
            role: sum(value == role for value in component_roles.values())
            for role in (
                "fit",
                "recipe-development",
                "quarantine-final-evaluation-overlap",
            )
        },
        "quarantined_training_articles_by_source": {
            source: sum(
                item["source"] == source
                and item["split"] == "train"
                and component_roles[component_id[index]]
                == "quarantine-final-evaluation-overlap"
                for index, item in enumerate(articles)
            )
            for source in ("wikitext", "squad")
        },
        "eligible_base_articles": len(train_rows),
        "eligible_continuation_passages": len(cont_rows),
        "eligible_sft_fit_questions": len(fit_qa_rows),
        "sft_fit_questions_excluded_for_full_prompt_answer_length": len(
            sft_length_excluded
        ),
        "recipe_dev_questions": len(outputs["qa-recipe-dev.jsonl"]),
        "final_questions": len(final),
        "final_articles": len(titles),
        "final_input_eligible_questions": len(final_candidates),
        "final_questions_excluded_for_input_context_length": sum(
            len(p["questions"]) for article in squad_dev for p in article["paragraphs"]
        )
        - len(final_candidates),
        "final_input_eligible_questions_not_selected_by_cap": len(final_candidates)
        - len(final),
        "component_count": len(components),
    }
    membership_hash = sha256((output / "memberships.json").read_bytes())
    frozen = {
        "recipe_id": "c02-cpu-pilot-wikitext2-squad11-20261021-v1",
        "status": "frozen-data-protocol-pending-independent-review",
        "source_manifest_sha256": sha256(manifest_bytes),
        "membership_manifest_sha256": membership_hash,
        "tokenizer_sha256": tokenizer_info["processor_sha256"],
        "seed": SEED,
        "source_roles": {
            "base": "eligible WikiText-2 raw train articles",
            "continuation_domain": "eligible SQuAD 1.1 train passage contexts",
            "continuation_control": "same budget from eligible WikiText base-train text",
            "sft_fit": "eligible SQuAD 1.1 train questions and original answer spans",
            "sft_recipe_dev": "held-out SQuAD official-train article components, input/answer only in recipe validation",
            "final": "deterministic filtered subset of official SQuAD 1.1 dev, plus WikiText official validation/test for retention",
        },
        "cross_source_grouping": "normalized article title, normalized exact paragraph/context, and normalized exact question; any connected component touching WikiText validation/test or SQuAD dev is quarantined from all training; training-only SQuAD recipe-development components excluded from all training; SQuAD passage continuation may share only with eligible fit-question groups",
        "final_selection": {
            "algorithm": "evaluate every official SQuAD dev question separately using only the final rendered input prompt; do not inspect answers or answer-span position; admit only if prompt BPE tokens + BOS + max_new_tokens fit the context; sort eligible question IDs within each article by SHA256(seed + question_id), sort articles by SHA256(seed + title), select round-robin up to the cap",
            "generation": {
                "template": QA_TEMPLATE,
                "temperature": 0.0,
                "sampling": False,
                "max_new_tokens": MAX_NEW_TOKENS,
                "stop_token_id": 2,
                "stop_behavior": "stop on first EOS or at max_new_tokens",
                "input_boundary": "BOS is prepended to the prompt; prompt tokens + BOS + max_new_tokens must be <= context",
            },
            "minimum_questions": MIN_FINAL,
            "minimum_articles": MIN_FINAL_ARTICLES,
            "total_official_dev_questions": sum(
                len(p["questions"])
                for article in squad_dev
                for p in article["paragraphs"]
            ),
            "input_eligible_count": len(final_candidates),
            "input_length_excluded_count": sum(
                len(p["questions"])
                for article in squad_dev
                for p in article["paragraphs"]
            )
            - len(final_candidates),
            "not_selected_by_cap_count": len(final_candidates) - len(final),
            "selected_count": len(final),
            "selected_articles": len(titles),
        },
        "tokenizer": {
            "kind": "byte-level BPE",
            "package": "tokenizers==0.22.2",
            "fit_data": "eligible base-train WikiText articles only after model_text whitespace collapse",
            "vocab_size_target": VOCAB_SIZE,
            "context_tokens": CONTEXT,
            "special_ids": tokenizer_info["special_ids"],
            "normalization": "model_text whitespace collapse; Tokenizers normalizer none",
            "byte_baseline": "UTF-8 bytes of same model_text + 4; reversible; special IDs 0-3",
            "processor_sha256": tokenizer_info["processor_sha256"],
        },
        "pilot_schedule": {
            "architecture": {
                "layers": 2,
                "width": 32,
                "heads": 4,
                "mlp_width": 64,
                "vocab_size": tokenizer_info["vocabulary_size"],
                "context": CONTEXT,
                "dropout": 0.1,
                "position": "RoPE",
                "precision": "FP32 parameters, activations, logits, loss and optimizer state",
                "context_selection_reason": "context 256 admitted only seven official-dev articles under the full-passage prompt rule; context 512 admits the required 128-question/16-article minimum while reserving BOS and all 32 output tokens",
            },
            "batch_size": 4,
            "updates": {
                "base": 32,
                "continuation_mixture": 15,
                "continuation_general_control": 15,
                "sft_full_from_domain": 32,
                "sft_lora_from_domain": 32,
                "sft_full_from_general": 32,
            },
            "continuation_target_mix": {
                "domain": 0.8,
                "general": 0.2,
                "unit": "consumed valid aligned target_ids where target_mask is true",
                "allocation": "15 updates each consume four domain blocks and one general block. Every block has exactly 512 true target_mask entries, including EOS targets, so 30,720 domain and 7,680 general targets yield exactly 80/20. Input IDs, target IDs, segment IDs and position IDs are aligned arrays; do not shift target_ids a second time.",
                "domain_targets": 30720,
                "general_targets": 7680,
                "mixture_targets": 38400,
                "general_control_targets": 38400,
            },
            "continuation_budget_adjustment": "The initial 16-update idea would consume 40,960 targets per branch. The frozen exact 80/20 mixture uses 15 updates x 5 x 512 = 38,400 valid targets, matched by the 15-update general-only control; this measured block allocation is frozen before optimization.",
            "optimizer": {
                "name": "AdamW",
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
                "weight_decay": 0.01,
                "global_gradient_norm_clip": 1.0,
                "learning_rates": {
                    "base": 0.0003,
                    "continuation_mixture": 0.0001,
                    "continuation_general_control": 0.0001,
                    "sft_full_from_domain": 0.0001,
                    "sft_lora_from_domain": 0.0005,
                    "sft_full_from_general": 0.0001,
                },
                "schedule": "linear warmup over first max(1, floor(0.1 * stage_updates)) updates, then cosine decay to 10% of the stage learning rate by final update",
            },
            "optimizer_reset": "fresh Adam moments, optimizer step zero, schedule clock zero at every named phase/branch; model parent parameters copied exactly from the named predecessor; no cross-branch optimizer state sharing",
            "parents": {
                "base": "seeded random initialization",
                "continuation_mixture": "base final parameters",
                "continuation_general_control": "same base final parameters",
                "sft_full_from_domain": "continuation_mixture final parameters",
                "sft_lora_from_domain": "continuation_mixture final parameters",
                "sft_full_from_general": "continuation_general_control final parameters",
            },
            "sft_protocol": "Use same seeded, shuffled eligible SQuAD fit question IDs and same response-only target masks for all three branches; select first published gold answer deterministically for fitting and preserve every answer for final scoring; fit input+answer within context without gold-span cropping.",
            "partial_batch": "token-weighted loss numerator/count; no dropped valid targets; base uses four full blocks per update; continuation uses microbatches [4,1] with four domain plus one matched general block for the mixture and five general blocks for its control; final incomplete source tails are measured and excluded",
            "block_contract": {
                "input_ids": "512 IDs; each segment starts with BOS followed by the preceding target tokens",
                "target_ids": "512 aligned next-token IDs, one target per input position; source EOS is included exactly once",
                "target_mask": "all 512 positions true for emitted blocks",
                "segment_ids": "causal attention may only connect positions with the same segment ID",
                "position_ids": "reset to zero at each independently chunked record segment",
                "record_boundary": "each passage/article starts a new BOS segment; chunks after 512 targets restart at BOS; no target transition crosses a record or chunk boundary",
                "tail": "incomplete final stream suffix is explicitly counted and excluded from the fixed-block schedule",
                "alignment_warning": "Do not apply a generic shifted-causal-target helper to these already aligned target_ids.",
            },
            "randomness": {
                "root_seed": 20261021,
                "policy": "explicit JAX PRNG splits for parameter init, dropout, batch order and sampling; dropout 0.1 in every training phase; each branch receives deterministic independent branch key derived from root seed and branch ID",
            },
            "branch_choice": "No final-score-based model selection; report all branches and evaluate predeclared domain and retention gates separately.",
        },
        "lora": {
            "targets": ["q_proj", "v_proj"],
            "target_scope": "every decoder layer",
            "rank": 4,
            "alpha": 8.0,
            "scale": 2.0,
            "orientation": "A[in_features,rank] @ B[rank,out_features] added to the frozen projection kernel",
            "initialization": "A iid normal with stddev 0.02; B exactly zero, so initial effective weights equal the parent",
            "adapter_dropout": 0.0,
            "bias_training": False,
            "optimizer_learning_rate": 0.0005,
        },
        "pretrained_comparator": {
            "repository": "EleutherAI/pythia-14m",
            "revision": PYTHIA_REVISION,
            "files_sha256": PYTHIA_FILES_SHA256,
            "consumer": "Transformers GPTNeoXForCausalLM with safetensors in the separately version-pinned baseline environment listed below",
            "python": "3.14.3",
            "package_versions": {
                "torch": "2.11.0",
                "transformers": "5.5.4",
                "safetensors": "0.5.3",
                "tokenizers": "0.22.2",
            },
            "license": "Apache-2.0 according to the author model card",
            "intended_use": "research on model behavior/functionality/limitations",
            "warning": "The author card says the suite is not intended for deployment or human-facing interactions. This is a contextual 14M baseline, not an instruction-tuned assistant or contamination-free comparator; possible Pile overlap with WikiText/SQuAD is disclosed.",
            "prompt_and_decoding": {
                "template": QA_TEMPLATE,
                "temperature": 0.0,
                "sampling": False,
                "max_new_tokens": MAX_NEW_TOKENS,
                "stop_token_id": 0,
                "stop_behavior": "stop on first EOS or at max_new_tokens",
            },
        },
        "native_lm_evaluation_harness": {
            "repository": "EleutherAI/lm-evaluation-harness",
            "release": LM_EVAL_VERSION,
            "commit": LM_EVAL_COMMIT,
            "task_inventory": ["hellaswag", "arc_easy"],
            "task_definition_source": "task YAML and source code at the pinned harness commit; actual resolved task versions, few-shot settings and sample IDs are recorded in the run receipt before scoring",
        },
        "precommitted_quality_floors": {
            "custom_qa_exact_match_minimum": 0.05,
            "custom_qa_token_f1_minimum": 0.10,
            "maximum_general_retention_regression_nats_per_valid_token_vs_matched_general_control": 0.10,
            "mechanics_and_recovery": "all independent numerical, cache, masking, fresh-process resume and cold-load gates must pass",
        },
        "observed_membership_counts": counts,
        "title_alias_audit": title_audit,
        "overlap_audit": overlap_audit,
        "denominator": "NLL per valid tokenizer target; byte rate per UTF-8 byte after whitespace collapse and trim per selected source unit; raw source bytes and normalized stream bytes reported separately",
        "limitations": [
            "filtered custom SQuAD 1.1 protocol, not an official leaderboard score",
            "heuristic overlap quarantine is not proof of zero semantic contamination",
            "no pretrained baseline result or training evidence is claimed by this data freeze",
            "full-scale 300M training, target qualification and human evaluation remain pending",
        ],
        "prepared_outputs": output_hashes,
        "target_block_inventories": {
            name: {key: value for key, value in inventory.items() if key != "blocks"}
            for name, inventory in block_inventories.items()
        },
        "pilot_block_selection": block_selection,
    }
    recipe_path = output / "frozen_recipe.json"
    recipe_path.write_bytes(stable_json(frozen))
    (output / "recipe.sha256").write_text(
        sha256(recipe_path.read_bytes()) + "  frozen_recipe.json\n"
    )
    verify_prepared(output, require_project_lock=False)
    return {
        "source_manifest_sha256": sha256(manifest_bytes),
        "membership_manifest_sha256": membership_hash,
        "recipe_sha256": sha256(recipe_path.read_bytes()),
        "counts": counts,
        "output": str(output),
        "status": "prepared; independent review and project lock required before training",
    }


def verify_prepared(output: Path, require_project_lock: bool = True) -> dict:
    """Recheck source, serialized membership and non-exposure claims from bytes."""
    manifest = json.loads((output / "source-manifest.json").read_text())
    membership_bytes = (output / "memberships.json").read_bytes()
    memberships = json.loads(membership_bytes)
    recipe = json.loads((output / "frozen_recipe.json").read_text())
    lock_path = Path(__file__).resolve().parent / "frozen_recipe.json"
    if require_project_lock:
        if not lock_path.is_file():
            raise ValueError(
                "project-owned accepted data recipe is missing; training is blocked pending independent data review"
            )
        if lock_path.read_bytes() != (output / "frozen_recipe.json").read_bytes():
            raise ValueError(
                "prepared recipe differs from the project-owned accepted data recipe"
            )
    if sha256(membership_bytes) != recipe["membership_manifest_sha256"]:
        raise ValueError("membership manifest does not match frozen recipe")
    if (
        sha256((output / "source-manifest.json").read_bytes())
        != recipe["source_manifest_sha256"]
    ):
        raise ValueError("source manifest does not match frozen recipe")
    if sha256(Path(__file__).read_bytes()) != manifest["preparer"]["sha256"]:
        raise ValueError("data preparer source differs from the frozen manifest")
    tokenizer_bytes = (output / "tokenizer.json").read_bytes()
    tokenizer_hash = sha256(tokenizer_bytes)
    expected_tokenizer_hash = recipe["tokenizer"]["processor_sha256"]
    if (
        tokenizer_hash != expected_tokenizer_hash
        or tokenizer_hash != recipe["tokenizer_sha256"]
        or tokenizer_hash != manifest["tokenizer"]["processor_sha256"]
    ):
        raise ValueError(
            "prepared tokenizer bytes differ from the frozen processor identity"
        )
    for key, info in manifest["source_file_hashes"].items():
        path = output / info["relative_path"]
        data = path.read_bytes()
        if len(data) != info["bytes"] or sha256(data) != info["sha256"]:
            raise ValueError(f"prepared source bytes changed: {key}")
    for name, info in recipe["prepared_outputs"].items():
        data = (output / name).read_bytes()
        if len(data) != info["bytes"] or sha256(data) != info["sha256"]:
            raise ValueError(f"prepared output bytes changed: {name}")
    for inventory in recipe["target_block_inventories"].values():
        for artifact_name in ("npz", "index"):
            info = inventory[artifact_name]
            data = (output / info["path"]).read_bytes()
            if len(data) != info["bytes"] or sha256(data) != info["sha256"]:
                raise ValueError(f"target block artifact bytes changed: {info['path']}")
    by_id = {row["record_id"]: row for row in memberships["articles"]}
    base = [
        json.loads(line)
        for line in (output / "base-train.jsonl").read_text().splitlines()
    ]
    cont = [
        json.loads(line)
        for line in (output / "continuation-domain.jsonl").read_text().splitlines()
    ]
    fit = [
        json.loads(line) for line in (output / "sft-fit.jsonl").read_text().splitlines()
    ]
    recipe_dev = [
        json.loads(line)
        for line in (output / "qa-recipe-dev.jsonl").read_text().splitlines()
    ]
    final = [
        json.loads(line)
        for line in (output / "qa-final.jsonl").read_text().splitlines()
    ]
    if any(
        by_id[row["id"]]["source"] != "wikitext"
        or by_id[row["id"]]["split"] != "train"
        or by_id[row["id"]]["role"] != "fit"
        for row in base
    ):
        raise ValueError("base train contains a non-fit WikiText article")
    train_title_roles = {
        row["title"]: row["role"]
        for row in memberships["articles"]
        if row["source"] == "squad" and row["split"] == "train"
    }
    if any(train_title_roles[row["article_title"]] != "fit" for row in cont + fit):
        raise ValueError("continuation or SFT fit contains a non-fit SQuAD article")
    if any(
        train_title_roles[row["article_title"]] != "recipe-development"
        for row in recipe_dev
    ):
        raise ValueError("recipe-dev QA contains a fit or held-out article")
    final_ids = {row["question_id"] for row in final}
    if final_ids != set(memberships["final_question_ids"]):
        raise ValueError("final question IDs differ from recorded membership")
    if (
        len({row["article_title"] for row in final}) < MIN_FINAL_ARTICLES
        or len(final) < MIN_FINAL
    ):
        raise ValueError("custom final QA inventory is below its frozen minimum")
    if any(
        not row["answers"] or not row["context"] or not row["question"] for row in final
    ):
        raise ValueError(
            "final QA record is missing its prompt or original answer alternatives"
        )
    if (
        recipe["title_alias_audit"]["exact_normalized_train_to_final_title_aliases"]
        != 0
    ):
        raise ValueError(
            "an exact normalized training title aliases a final SQuAD article"
        )
    if len({row["id"] for row in base}) != len(base):
        raise ValueError("base article IDs are duplicated")
    if len({row["id"] for row in cont}) != len(cont):
        raise ValueError("continuation passage IDs are duplicated")
    if len({row["id"] for row in fit}) != len(fit):
        raise ValueError("SFT question IDs are duplicated")
    return {
        "status": "pass",
        "base_articles": len(base),
        "continuation_passages": len(cont),
        "sft_fit_questions": len(fit),
        "recipe_dev_questions": len(recipe_dev),
        "final_questions": len(final),
        "final_articles": len({row["article_title"] for row in final}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify prepared artifacts against the project-owned accepted recipe",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.verify:
        result = verify_prepared(args.output.resolve())
    else:
        result = prepare(args.output.resolve(), args.download)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
