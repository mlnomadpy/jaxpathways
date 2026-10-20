"""Shared Flax architecture and data/checkpoint contracts for the staged pilot."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax import serialization

ROOT = Path(__file__).resolve().parent
SPECIAL = {'<pad>': 0, '<mask>': 1, '<unk>': 2, '<bos>': 3, '<eos>': 4}
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\W\s]", re.UNICODE)


class BidirectionalEncoder(nn.Module):
    vocab_size: int
    max_length: int = 48
    width: int = 24
    heads: int = 2
    mlp_width: int = 48

    @nn.compact
    def __call__(self, tokens):
        valid = tokens != SPECIAL['<pad>']
        positions = jnp.arange(tokens.shape[1])
        x = nn.Embed(self.vocab_size, self.width, name='word_embedding')(tokens)
        position = self.param('position_embedding', nn.initializers.normal(0.02), (self.max_length, self.width))
        x = x + position[positions][None, :, :]
        attention_mask = nn.make_attention_mask(valid, valid, dtype=bool)
        attended = nn.SelfAttention(
            num_heads=self.heads,
            qkv_features=self.width,
            out_features=self.width,
            dropout_rate=0.0,
            name='self_attention',
        )(x, mask=attention_mask)
        x = nn.LayerNorm(name='attention_norm')(x + attended)
        hidden = nn.Dense(self.mlp_width, name='mlp_in')(x)
        hidden = nn.gelu(hidden)
        hidden = nn.Dense(self.width, name='mlp_out')(hidden)
        hidden = nn.LayerNorm(name='output_norm')(x + hidden)
        pool_mask = valid & (tokens != SPECIAL['<bos>']) & (tokens != SPECIAL['<eos>']) & (tokens != SPECIAL['<mask>'])
        weights = pool_mask[..., None].astype(hidden.dtype)
        pooled = jnp.sum(hidden * weights, axis=1) / jnp.maximum(jnp.sum(weights, axis=1), 1.0)
        pooled = nn.Dense(self.width, name='projection')(pooled)
        pooled = pooled / jnp.maximum(jnp.linalg.norm(pooled, axis=-1, keepdims=True), 1e-8)
        logits = nn.Dense(self.vocab_size, name='mlm_head')(hidden)
        return pooled, logits


def token_words(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def load_data() -> tuple[list[dict], list[dict], list[dict], str]:
    source = ROOT / 'data/source/alice-pg11.txt'
    corpus_path = ROOT / 'data/corpus.jsonl'
    queries_path = ROOT / 'data/queries.jsonl'
    qrels_path = ROOT / 'data/qrels.jsonl'
    manifest_path = ROOT / 'data/manifest.json'
    files = (source, corpus_path, queries_path, qrels_path, manifest_path, ROOT / 'config.json')
    corpus_hash = hashlib.sha256()
    for path in files:
        corpus_hash.update(path.name.encode())
        corpus_hash.update(path.read_bytes())
    manifest = json.loads(manifest_path.read_text())
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash != manifest['source_sha256']:
        raise ValueError('source text checksum mismatch')
    docs = read_jsonl(corpus_path)
    queries = read_jsonl(queries_path)
    qrels = read_jsonl(qrels_path)
    if any(row['source_sha256'] != source_hash for row in docs):
        raise ValueError('corpus passages do not match pinned source bytes')
    groups: dict[str, set[str]] = {}
    for doc in docs:
        groups.setdefault(doc['group'], set()).add(doc['split'])
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError('source group crosses data splits')
    if len({doc['id'] for doc in docs}) != len(docs):
        raise ValueError('duplicate document ID')
    doc_ids = {doc['id'] for doc in docs}
    query_ids = {query['id'] for query in queries}
    if any(row['doc_id'] not in doc_ids or row['query_id'] not in query_ids for row in qrels):
        raise ValueError('qrels refer to an unknown query or passage')
    for qrel in qrels:
        query = next(item for item in queries if item['id'] == qrel['query_id'])
        doc = next(item for item in docs if item['id'] == qrel['doc_id'])
        if query['split'] != doc['split']:
            raise ValueError('query and judged passage belong to different splits')
    return docs, queries, qrels, corpus_hash.hexdigest()


def make_tokenizer(docs: list[dict], max_vocabulary: int) -> dict:
    counts = Counter(word for doc in docs if doc['split'] == 'train' for word in token_words(doc['text']))
    remaining = sorted(counts, key=lambda word: (-counts[word], word))[:max_vocabulary - len(SPECIAL)]
    vocab = {**SPECIAL, **{word: index + len(SPECIAL) for index, word in enumerate(remaining)}}
    return {'name': 'lowercase-regex-word-v1', 'vocabulary': vocab, 'vocabulary_size': len(vocab),
            'special_tokens': SPECIAL, 'max_vocabulary': max_vocabulary, 'fit_split': 'train'}


def tokenize(text: str, tokenizer: dict, max_length: int) -> tuple[np.ndarray, float]:
    words = token_words(text)
    vocab = tokenizer['vocabulary']
    ids = [SPECIAL['<bos>']] + [vocab.get(word, SPECIAL['<unk>']) for word in words] + [SPECIAL['<eos>']]
    clipped = ids[:max_length]
    result = np.full((max_length,), SPECIAL['<pad>'], dtype=np.int32)
    result[:len(clipped)] = clipped
    oov = sum(word not in vocab for word in words) / max(len(words), 1)
    return result, oov


def batch_tokens(texts: list[str], tokenizer: dict, max_length: int) -> tuple[np.ndarray, float, int]:
    arrays, unknowns = zip(*(tokenize(text, tokenizer, max_length) for text in texts))
    truncations = sum(len(token_words(text)) + 2 > max_length for text in texts)
    return np.stack(arrays), float(np.mean(unknowns)), truncations


def make_model(config: dict, tokenizer: dict):
    return BidirectionalEncoder(
        vocab_size=tokenizer['vocabulary_size'], max_length=config['max_length'],
        width=config['width'], heads=config['attention_heads'], mlp_width=config['mlp_width'])


def init_params(model, seed: int, max_length: int):
    import jax
    sample = jnp.full((1, max_length), SPECIAL['<bos>'], dtype=jnp.int32)
    return model.init(jax.random.PRNGKey(seed), sample)['params']


def tree_parameter_count(params) -> int:
    return sum(int(np.prod(leaf.shape)) for leaf in jax_tree_leaves(params))


def jax_tree_leaves(tree):
    import jax
    return jax.tree.leaves(tree)


def tree_bytes(params) -> bytes:
    return serialization.to_bytes(params)


def load_params(template, payload: bytes):
    return serialization.from_bytes(template, payload)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
