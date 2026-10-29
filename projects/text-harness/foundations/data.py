"""Chapter-grouped Alice data, train-only byte/BPE processors and packed windows."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "data/source/alice-pg11.txt"
SOURCE_MANIFEST = HERE / "data/manifest.json"
SPLIT_MAP = {
    "I": "train",
    "II": "train",
    "III": "train",
    "IV": "train",
    "V": "validation",
    "VI": "test",
    "VII": "test",
    "VIII": "test",
}
PAD, BOS, EOS, UNK = 0, 1, 2, 3


@dataclass(frozen=True)
class Record:
    record_id: str
    chapter: str
    split: str
    text: str
    normalized_bytes: int


class ByteTokenizer:
    """Reversible UTF-8 byte baseline; special IDs occupy 0 through 3."""

    vocab_size = 260
    special_ids = {"pad": PAD, "bos": BOS, "eos": EOS, "unk": UNK}

    def encode(self, text):
        return [byte + 4 for byte in text.encode("utf-8")]

    def decode(self, ids):
        values = []
        for token in ids:
            token = int(token)
            if token in (PAD, BOS, EOS):
                continue
            if token < 4 or token >= self.vocab_size:
                raise ValueError(f"invalid byte token id {token}")
            values.append(token - 4)
        return bytes(values).decode("utf-8", errors="strict")

    def to_json(self):
        return json.dumps(
            {
                "kind": "utf8-byte-v1",
                "special_ids": self.special_ids,
                "byte_offset": 4,
                "vocabulary_size": self.vocab_size,
                "normalization": "none",
                "unknown_policy": "all UTF-8 bytes represented",
            },
            sort_keys=True,
        )


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def source_identity():
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    actual = _sha256(SOURCE.read_bytes())
    if actual != manifest["source_sha256"]:
        raise ValueError(
            "Project Gutenberg source bytes do not match the retained manifest"
        )
    return manifest


def load_records(source=SOURCE):
    """Select and exact-deduplicate eligible paragraphs before any tokenization."""
    manifest = source_identity()
    if Path(source).resolve() != SOURCE.resolve():
        raw = Path(source).read_bytes()
        if _sha256(raw) != manifest["source_sha256"]:
            raise ValueError(
                "alternate source bytes do not match the immutable source manifest"
            )
        text = raw.decode("utf-8")
    else:
        text = SOURCE.read_text(encoding="utf-8")
    all_starts = list(re.finditer(r"(?m)^CHAPTER ([IVXLCDM]+)\.\s*$", text))
    starts = [match for match in all_starts if match.group(1) in SPLIT_MAP]
    if len(starts) != 8 or [match.group(1) for match in starts] != list(SPLIT_MAP):
        raise ValueError("expected the eight declared chapter boundaries")
    candidates = []
    for match in starts:
        next_chapter = next(
            (
                candidate
                for candidate in all_starts
                if candidate.start() > match.start()
            ),
            None,
        )
        end = (
            next_chapter.start()
            if next_chapter
            else text.find("*** END OF THE PROJECT GUTENBERG EBOOK", match.end())
        )
        chapter_text = text[match.end() : end]
        for paragraph_index, paragraph in enumerate(re.split(r"\n\s*\n", chapter_text)):
            cleaned = " ".join(paragraph.split())
            if len(cleaned.encode("utf-8")) < 45 or len(cleaned.split()) < 8:
                continue
            if cleaned.startswith("[Illustration") or cleaned.isupper():
                continue
            chapter = match.group(1)
            candidates.append(
                Record(
                    f"{chapter}:{paragraph_index:04d}",
                    chapter,
                    SPLIT_MAP[chapter],
                    cleaned,
                    len(cleaned.encode("utf-8")),
                )
            )
    seen_text = {}
    unique = []
    for record in candidates:
        digest = _sha256(record.text.encode("utf-8"))
        previous = seen_text.get(digest)
        if previous and previous[0] != record.text:
            raise ValueError("SHA-256 collision while deduplicating source paragraphs")
        if previous:
            # A duplicate in a different chapter would invalidate the group split.
            if previous[1] != record.chapter:
                raise ValueError(
                    f"duplicate paragraph crosses chapter groups: {record.record_id}"
                )
            continue
        seen_text[digest] = (record.text, record.chapter)
        unique.append(record)
    if not unique or not any(row.split == "train" for row in unique):
        raise ValueError("no eligible source paragraphs")
    return unique, manifest


def train_bpe(records, vocab_size=384, save_path=None):
    """Train a byte-fallback BPE processor only from training chapter records."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    train_records = [row for row in records if row.split == "train"]
    if not train_records or len(train_records) == len(records):
        raise ValueError(
            "train tokenizer requires explicit train and held-out chapter groups"
        )
    special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
    processor = Tokenizer(models.BPE(unk_token="[UNK]", byte_fallback=True))
    processor.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=True
    )
    processor.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    processor.train_from_iterator((row.text for row in train_records), trainer=trainer)
    processor.no_padding()
    processor.no_truncation()
    ids = {
        name: processor.token_to_id(token)
        for name, token in zip(("pad", "bos", "eos", "unk"), special_tokens)
    }
    if ids != {"pad": PAD, "bos": BOS, "eos": EOS, "unk": UNK}:
        raise ValueError(f"BPE special-token IDs changed: {ids}")
    processor_json = processor.to_str()
    processor_hash = _sha256(processor_json.encode())
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(processor_json)
    info = {
        "kind": "bytelevel-bpe",
        "vocabulary_size": processor.get_vocab_size(),
        "special_ids": ids,
        "normalization": "none",
        "unknown_policy": "byte fallback plus [UNK] guard",
        "byte_fallback": True,
        "processor_sha256": processor_hash,
        "trained_record_ids": [row.record_id for row in train_records],
        "trained_chapters": ["I", "II", "III", "IV"],
        "decoding": "ByteLevel decoder; special tokens are not included in text records",
    }
    return processor, info


def encode_text(processor, text):
    if isinstance(processor, ByteTokenizer):
        return processor.encode(text)
    encoded = processor.encode(text, add_special_tokens=False)
    if UNK in encoded.ids:
        raise ValueError(
            "BPE processor emitted [UNK]; byte fallback contract was violated"
        )
    return encoded.ids


def decode_text(processor, token_ids):
    if isinstance(processor, ByteTokenizer):
        return processor.decode(token_ids)
    return processor.decode(
        [int(token) for token in token_ids], skip_special_tokens=True
    )


def pack_records(records, processor, context):
    """Chunk documents without crossing record boundaries; expose every selected token."""
    if context < 4:
        raise ValueError("context must leave room for BOS, text, and EOS")
    rows, segments, text_targets, document_index = [], [], [], []
    segment_id = 0
    text_tokens = normalized_bytes = truncations = chunks = 0
    membership = []
    for record in records:
        ids = encode_text(processor, record.text)
        if not ids:
            continue
        text_tokens += len(ids)
        normalized_bytes += record.normalized_bytes
        record_index = len(membership)
        membership.append(record.record_id)
        chunk_width = context - 1
        # Each chunk is BOS + at most context-1 text IDs + EOS, so no text token is dropped.
        for start in range(0, len(ids), chunk_width):
            chunk = ids[start : start + chunk_width]
            seq = [BOS, *chunk, EOS]
            row = np.full((context + 1,), PAD, dtype=np.int32)
            seg = np.zeros((context + 1,), dtype=np.int32)
            text_marker = np.zeros((context + 1,), dtype=np.bool_)
            row[: len(seq)] = seq
            segment_id += 1
            seg[: len(seq)] = segment_id
            text_marker[1 : 1 + len(chunk)] = True
            rows.append(row)
            segments.append(seg)
            text_targets.append(text_marker[1:])
            document_index.append(record_index)
            chunks += 1
    if not rows:
        raise ValueError("no nonempty packed examples")
    tokens = np.stack(rows)
    segment_ids = np.stack(segments)
    content_mask = np.stack(text_targets)
    document_index = np.asarray(document_index, dtype=np.int32)
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    loss_mask = (targets != PAD) & (segment_ids[:, :-1] == segment_ids[:, 1:])
    stats = {
        "documents": len(membership),
        "record_ids": membership,
        "chunks": chunks,
        "normalized_paragraph_bytes": normalized_bytes,
        "text_tokens": text_tokens,
        "tokens_exposed": text_tokens,
        "tokens_truncated": truncations,
        "loss_targets": int(loss_mask.sum()),
        "context": context,
    }
    return {
        "tokens": tokens,
        "segments": segment_ids,
        "inputs": inputs,
        "targets": targets,
        "loss_mask": loss_mask,
        "text_mask": content_mask,
        "document_index": document_index,
        "stats": stats,
    }


def byte_unigram_bpb(train_records, held_records, alpha=0.1):
    """Fit train-only byte counts and score normalized held-out paragraph bytes."""
    counts = np.full((256,), alpha, dtype=np.float64)
    for row in train_records:
        for value in row.text.encode("utf-8"):
            counts[value] += 1
    probs = counts / counts.sum()
    nll = 0.0
    byte_count = 0
    for row in held_records:
        values = np.frombuffer(row.text.encode("utf-8"), dtype=np.uint8)
        nll += float(-np.log(probs[values]).sum())
        byte_count += len(values)
    return {
        "nll_nats": nll,
        "bytes": byte_count,
        "bits_per_normalized_paragraph_byte": nll / np.log(2) / max(byte_count, 1),
        "counts_source": "training paragraphs only",
        "smoothing_alpha": alpha,
    }
