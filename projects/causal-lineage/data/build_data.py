"""Rebuild the frozen chapter-split language-model and continuation records."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/source/alice-pg11.txt"
SPLITS = {
    "I": "train",
    "II": "train",
    "III": "train",
    "IV": "train",
    "V": "validation",
    "VI": "test",
    "VII": "test",
    "VIII": "test",
}
START = "*** START OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***"
END = "*** END OF THE PROJECT GUTENBERG EBOOK"


def eligible_paragraphs(text: str) -> list[str]:
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n", text)
    ]
    return [
        paragraph
        for paragraph in paragraphs
        if 110 <= len(paragraph) <= 950
        and not paragraph.startswith("[")
        and not paragraph.isupper()
    ]


def main() -> None:
    source_bytes = SOURCE.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    source = source_bytes.decode("utf-8-sig")
    if START not in source or END not in source:
        raise ValueError("pinned Project Gutenberg header markers were not found")
    body = source.split(START, 1)[1].split(END, 1)[0]
    parts = re.split(r"\n\s*CHAPTER\s+([IVXLCDM]+)\.\s*\n", body)

    passages: list[dict] = []
    instructions: list[dict] = []
    for index in range(1, len(parts), 2):
        chapter = parts[index]
        if chapter not in SPLITS:
            continue
        chapter_text = parts[index + 1]
        title = next(
            (line.strip() for line in chapter_text.splitlines() if line.strip()), ""
        )
        eligible = eligible_paragraphs(chapter_text)
        count = min(32, len(eligible))
        selected_positions = sorted(
            {round(i * (len(eligible) - 1) / max(count - 1, 1)) for i in range(count)}
        )
        for passage_index, position in enumerate(selected_positions):
            passages.append(
                {
                    "id": f"pg11-chapter-{chapter}-passage-{passage_index:03d}",
                    "group": f"pg11-chapter-{chapter}",
                    "chapter": chapter,
                    "chapter_title": title,
                    "split": SPLITS[chapter],
                    "source_id": "gutenberg-11",
                    "source_sha256": source_sha,
                    "text": eligible[position],
                }
            )

        # These are short educational SFT targets, not preference or human
        # alignment data. Context and continuation are adjacent source
        # paragraphs from train chapters only; held-out chapters never enter
        # adaptation.
        if SPLITS[chapter] == "train":
            for pair_index, (context, response) in enumerate(
                zip(eligible, eligible[1:])
            ):
                if pair_index >= 8:
                    break
                instructions.append(
                    {
                        "id": f"continue-chapter-{chapter}-pair-{pair_index:02d}",
                        "group": f"pg11-chapter-{chapter}",
                        "chapter": chapter,
                        "split": "train",
                        "source_sha256": source_sha,
                        "instruction": "Continue this story with the next short passage.",
                        "context": context,
                        "response": response,
                    }
                )

    corpus_path = ROOT / "data/corpus.jsonl"
    instruction_path = ROOT / "data/instructions.jsonl"
    corpus_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in passages),
        encoding="utf-8",
    )
    instruction_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in instructions),
        encoding="utf-8",
    )
    manifest = {
        "source_id": "gutenberg-11",
        "title": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
        "edition": "Project Gutenberg eBook 11, Millennium Fulcrum Edition 3.0",
        "source_url": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
        "metadata_url": "https://www.gutenberg.org/ebooks/11",
        "retrieved_at": "2026-10-20",
        "rights": "Project Gutenberg lists this text as public domain in the United States. The included source file retains its terms. Check local copyright law before reuse.",
        "source_sha256": source_sha,
        "split_unit": "chapter",
        "split_map": SPLITS,
        "passage_selection": "At most 32 evenly spaced eligible paragraphs per chapter, matching embedding-lineage.",
        "instruction_selection": "First eight adjacent eligible paragraph pairs in each train chapter; adaptation only; no validation/test examples.",
        "data_files": {
            "corpus.jsonl": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
            "instructions.jsonl": hashlib.sha256(
                instruction_path.read_bytes()
            ).hexdigest(),
        },
        "limitations": [
            "One public-domain English novel; held-out chapters share author, characters, setting and style with train chapters.",
            "The continuation records are derived from narrative adjacency and are not human preference labels or evidence of general instruction following.",
            "The pilot is small CPU training and makes no 300M-scale or production claim.",
        ],
    }
    (ROOT / "data/manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passages": len(passages),
                "passages_by_split": {
                    split: sum(row["split"] == split for row in passages)
                    for split in ("train", "validation", "test")
                },
                "instruction_pairs": len(instructions),
                "source_sha256": source_sha,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
