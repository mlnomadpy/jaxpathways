"""Masked-token and multi-positive supervised/weak contrastive objectives."""

import math

import jax
import jax.numpy as jnp


def multi_positive_contrastive(
    query_vectors,
    document_vectors,
    positive_mask,
    temperature=0.05,
    candidate_mask=None,
):
    """Average log-softmax loss over each query's complete positive set."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    queries = jnp.asarray(query_vectors, jnp.float32)
    documents = jnp.asarray(document_vectors, jnp.float32)
    positives = jnp.asarray(positive_mask, jnp.bool_)
    if (
        queries.ndim != 2
        or documents.ndim != 2
        or queries.shape[-1] != documents.shape[-1]
    ):
        raise ValueError(
            "query and document vectors must be rank two with equal dimensions"
        )
    if positives.shape != (queries.shape[0], documents.shape[0]):
        raise ValueError("positive_mask shape must be [query_count, document_count]")
    valid_candidates = (
        jnp.ones((documents.shape[0],), dtype=jnp.bool_)
        if candidate_mask is None
        else jnp.asarray(candidate_mask, jnp.bool_)
    )
    if valid_candidates.shape != (documents.shape[0],):
        raise ValueError("candidate_mask shape must be [document_count]")
    positives = positives & valid_candidates[None, :]
    if not isinstance(positives, jax.core.Tracer):
        if not bool(jnp.all(jnp.any(positives, axis=1))):
            raise ValueError("every query needs at least one known positive document")
    logits = queries @ documents.T / temperature
    logits = jnp.where(valid_candidates[None, :], logits, -jnp.inf)
    all_lse = jax.scipy.special.logsumexp(logits, axis=1)
    positive_lse = jax.scipy.special.logsumexp(
        jnp.where(positives, logits, -jnp.inf), axis=1
    )
    return jnp.mean(all_lse - positive_lse)


def retrieval_metrics(rankings, relevance, cutoffs=(1, 5, 10)):
    """Compute macro Recall, MRR and nDCG from per-query ranked doc IDs."""
    ranking_ids = list(map(int, rankings))
    relevance_ids = list(map(int, relevance))
    if len(set(ranking_ids)) != len(ranking_ids):
        raise ValueError("ranking query IDs must be unique")
    if len(set(relevance_ids)) != len(relevance_ids):
        raise ValueError("relevance query IDs must be unique")
    if set(ranking_ids) != set(relevance_ids):
        raise ValueError("rankings must cover every expected query exactly once")
    rows = []
    for query_id, ranked_ids in rankings.items():
        ranked_ids = list(map(int, ranked_ids))
        if len(set(ranked_ids)) != len(ranked_ids):
            raise ValueError(f"query {query_id} ranking contains duplicate documents")
        relevant = set(map(int, relevance.get(int(query_id), ())))
        if not relevant <= set(ranked_ids):
            raise ValueError(
                f"query {query_id} ranking omitted one or more judged positives"
            )
        if not relevant:
            rows.append(
                {
                    "query_id": int(query_id),
                    "judged": False,
                    "relevant_count": 0,
                    "first_relevant_rank": None,
                }
            )
            continue
        ranks = [
            index + 1
            for index, doc_id in enumerate(ranked_ids)
            if int(doc_id) in relevant
        ]
        row = {
            "query_id": int(query_id),
            "judged": True,
            "relevant_count": len(relevant),
            "first_relevant_rank": min(ranks) if ranks else None,
        }
        for k in cutoffs:
            hits = sum(rank <= k for rank in ranks)
            row[f"recall@{k}"] = hits / len(relevant)
            row[f"ndcg@{k}"] = sum(
                1.0 / math.log2(rank + 1.0) for rank in ranks if rank <= k
            ) / sum(1.0 / math.log2(i + 2.0) for i in range(min(len(relevant), k)))
        row["mrr"] = 0.0 if not ranks else 1.0 / min(ranks)
        rows.append(row)
    judged = [row for row in rows if row["judged"]]
    result = {
        "queries_total": len(rows),
        "queries_judged": len(judged),
        "queries_unjudged": len(rows) - len(judged),
        "per_query": rows,
    }
    for key in (
        "recall@1",
        "recall@5",
        "recall@10",
        "ndcg@1",
        "ndcg@5",
        "ndcg@10",
        "mrr",
    ):
        result[key] = (
            0.0 if not judged else float(sum(row[key] for row in judged) / len(judged))
        )
    return result
