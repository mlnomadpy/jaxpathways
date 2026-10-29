"""SQuAD v1.1 official normalization, exact-match and token-F1 semantics.

The reference formulas follow the SQuAD v1.1 evaluator mirrored in
Hugging Face Evaluate's metrics/squad/compute_score.py. This small local file is
the immutable implementation used for C02 reports; its bytes are hashed into
the evaluation receipt. SQuAD v1.1 has no no-answer convention: token F1 is zero
when normalized token overlap is empty, even if both normalized strings are
empty, while exact match can be one.
"""

from __future__ import annotations

from collections import Counter
import re
import string


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    predicted_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    common = Counter(predicted_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def score(prediction: str, references: list[str]) -> dict[str, float]:
    if not references:
        raise ValueError("SQuAD scoring needs at least one human reference")
    return {
        "exact_match": max(
            exact_match(prediction, reference) for reference in references
        ),
        "token_f1": max(token_f1(prediction, reference) for reference in references),
    }
