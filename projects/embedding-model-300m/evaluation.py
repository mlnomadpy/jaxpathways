"""Exact full-corpus retrieval evaluation and a deterministic BM25 baseline."""

from __future__ import annotations

import math
import re
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np

try:
    from . import objectives
    from .data.dataset import encode_documents, encode_queries
except ImportError:
    import objectives
    from data.dataset import encode_documents, encode_queries


def exact_rank(query_vectors, document_vectors, query_ids, document_ids):
    qv = np.asarray(query_vectors, np.float64)
    dv = np.asarray(document_vectors, np.float64)
    qids = np.asarray(query_ids, np.int64)
    dids = np.asarray(document_ids, np.int64)
    if qv.ndim != 2 or dv.ndim != 2 or qv.shape[1] != dv.shape[1]:
        raise ValueError("query and document embedding shapes do not agree")
    if qv.shape[0] != len(qids) or dv.shape[0] != len(dids):
        raise ValueError("embedding rows and stable IDs do not agree")
    if len(np.unique(qids)) != len(qids) or len(np.unique(dids)) != len(dids):
        raise ValueError("query and document IDs must be unique")
    if not np.isfinite(qv).all() or not np.isfinite(dv).all():
        raise ValueError("retrieval embeddings must be finite")
    scores = qv @ dv.T
    rankings = {}
    for row, query_id in enumerate(qids):
        order = np.lexsort((dids, -scores[row]))
        rankings[int(query_id)] = [int(dids[index]) for index in order]
    return rankings, scores


def metrics_from_rankings(rankings, claims):
    claim_ids = [int(row["id"]) for row in claims]
    if len(set(claim_ids)) != len(claim_ids):
        raise ValueError("query inventory contains duplicate IDs")
    relevance = {
        int(row["id"]): [int(doc_id) for doc_id in row.get("evidence", {})]
        for row in claims
    }
    result = objectives.retrieval_metrics(rankings, relevance)
    result["judgment_source"] = "SciFact claims[*].evidence document IDs only"
    result["unjudged_queries_excluded"] = result["queries_unjudged"]
    return result


def encode_model(model, params, config, documents, claims, tokenizer, batch_size=32):
    doc_rows, doc_truncated = encode_documents(documents, tokenizer, config.max_length)
    query_rows, query_truncated = encode_queries(claims, tokenizer, config.max_length)

    def packed(rows):
        values = np.full((len(rows), config.max_length), config.pad_id, np.int32)
        for index, row in enumerate(rows):
            values[index, : len(row["ids"])] = row["ids"]
        return values

    doc_input = packed(doc_rows)
    query_input = packed(query_rows)

    @jax.jit
    def apply(ids):
        return model.pool(params, ids, config)

    doc_vectors, doc_ids = [], []
    for start in range(0, len(doc_rows), batch_size):
        rows = doc_rows[start : start + batch_size]
        doc_vectors.extend(
            np.asarray(
                apply(jnp.asarray(doc_input[start : start + batch_size])), np.float32
            )
        )
        doc_ids.extend(row["id"] for row in rows)
    query_vectors, query_ids = [], []
    for start in range(0, len(query_rows), batch_size):
        rows = query_rows[start : start + batch_size]
        query_vectors.extend(
            np.asarray(
                apply(jnp.asarray(query_input[start : start + batch_size])), np.float32
            )
        )
        query_ids.extend(row["id"] for row in rows)
    return {
        "query_vectors": np.asarray(query_vectors, np.float32),
        "document_vectors": np.asarray(doc_vectors, np.float32),
        "query_ids": np.asarray(query_ids, np.int64),
        "document_ids": np.asarray(doc_ids, np.int64),
        "query_truncated_tokens": query_truncated,
        "document_truncated_tokens": doc_truncated,
        "max_length": config.max_length,
    }


def _terms(text):
    return re.findall(r"\w+", text.casefold())


def bm25_rank(documents, claims, *, k1=1.5, b=0.75):
    """Small, transparent exact BM25 implementation for the 5,183-doc corpus."""
    tokenized_docs = [
        _terms((row.get("title", "") or "") + " " + " ".join(row.get("abstract") or []))
        for row in documents
    ]
    doc_ids = np.asarray([int(row["doc_id"]) for row in documents], np.int64)
    frequencies = [Counter(tokens) for tokens in tokenized_docs]
    lengths = np.asarray([len(tokens) for tokens in tokenized_docs], np.float64)
    average_length = float(lengths.mean())
    document_frequency = Counter(
        term for counts in frequencies for term in counts.keys()
    )
    total_docs = len(documents)
    rankings = {}
    for claim in claims:
        terms = _terms(claim["claim"])
        scores = np.zeros(total_docs, np.float64)
        for term in terms:
            df = document_frequency.get(term, 0)
            inverse = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
            tf = np.asarray([counts.get(term, 0) for counts in frequencies], np.float64)
            denominator = tf + k1 * (1 - b + b * lengths / max(average_length, 1e-12))
            scores += inverse * tf * (k1 + 1) / np.maximum(denominator, 1e-12)
        order = np.lexsort((doc_ids, -scores))
        rankings[int(claim["id"])] = [int(doc_ids[index]) for index in order]
    return rankings


def bm25_scores(documents, claims, *, k1=1.5, b=0.75):
    """Return raw BM25 score matrix and stable inventories for independent audit."""
    tokenized_docs = [
        _terms((row.get("title", "") or "") + " " + " ".join(row.get("abstract") or []))
        for row in documents
    ]
    doc_ids = np.asarray([int(row["doc_id"]) for row in documents], np.int64)
    frequencies = [Counter(tokens) for tokens in tokenized_docs]
    lengths = np.asarray([len(tokens) for tokens in tokenized_docs], np.float64)
    average_length = float(lengths.mean())
    document_frequency = Counter(
        term for counts in frequencies for term in counts.keys()
    )
    total_docs = len(documents)
    scores = np.zeros((len(claims), total_docs), np.float64)
    for row_index, claim in enumerate(claims):
        for term in _terms(claim["claim"]):
            df = document_frequency.get(term, 0)
            inverse = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
            for doc_index, counts in enumerate(frequencies):
                tf = counts.get(term, 0)
                denominator = tf + k1 * (
                    1 - b + b * lengths[doc_index] / max(average_length, 1e-12)
                )
                scores[row_index, doc_index] += (
                    inverse * tf * (k1 + 1) / max(denominator, 1e-12)
                )
    query_ids = np.asarray([int(row["id"]) for row in claims], np.int64)
    return query_ids, doc_ids, scores
