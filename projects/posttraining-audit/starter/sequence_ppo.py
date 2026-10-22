"""Tiny token-conditioned sequence PPO for a synthetic CPU teaching stage."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

PROJECT = Path(
    os.environ.get("POSTTRAINING_AUDIT_PROJECT", Path(__file__).resolve().parents[1])
)
RECIPE_PATH = PROJECT / "reference/sequence_ppo/recipe.json"
RECIPE_BYTES = RECIPE_PATH.read_bytes()
RECIPE = json.loads(RECIPE_BYTES)
RECIPE_SHA256 = hashlib.sha256(RECIPE_BYTES).hexdigest()
TOKENS = RECIPE["tokens"]
PAD = TOKENS["pad"]
BOS = TOKENS["bos"]
EOS = TOKENS["eos"]
VOCAB_SIZE = TOKENS["vocabularySize"]
TASK_FOR_PROMPT = jnp.asarray([item["task"] for item in RECIPE["promptInstances"]])
MAX_TOKENS = int(RECIPE["rollout"]["maxResponseTokens"])


def _masked_logits(logits, position):
    """Keep only response tokens and disallow EOS until a second token."""
    valid = jnp.asarray([False, False, False, False, True, True, True, True])
    valid = valid.at[EOS].set(position >= 1)
    return jnp.where(valid, logits, -1.0e9)


def _selected_log_prob(params, tasks, previous, positions, actions):
    logits = params[tasks, previous, positions, :]
    logits = jax.vmap(_masked_logits)(logits, positions)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, actions[:, None], axis=-1)[:, 0]


def _selected_log_probs_for_records(params, records):
    time, batch = records["actions"].shape
    tasks = jnp.broadcast_to(records["task_ids"][None, :], (time, batch)).reshape(-1)
    positions = jnp.broadcast_to(jnp.arange(time)[:, None], (time, batch)).reshape(-1)
    logp = _selected_log_prob(
        params,
        tasks,
        records["previous_tokens"].reshape(-1),
        positions,
        records["actions"].reshape(-1),
    ).reshape(time, batch)
    return jnp.where(records["response_mask"], logp, 0.0)


def fit_reward_model():
    """TODO: fit the fixed pairwise fixture. Return frozen weights [task, vocabulary] and fit metrics with a SHA-256 digest."""
    raise NotImplementedError(
        "fit the pairwise synthetic reward model and return (weights, fit_metrics)"
    )


def _array_sha256(array):
    value = np.asarray(array, dtype=np.float32)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def initialize(seed=0):
    """Return tiny policy, learned-critic and frozen reference parameters."""
    del seed  # The initial tables are exact zeros; stochasticity begins in rollout keys.
    # The final policy position defines the fixed one-token continuation used
    # only to supervise timeout bootstrap values; collected rollouts stop earlier.
    policy = jnp.zeros((2, VOCAB_SIZE, MAX_TOKENS + 1, VOCAB_SIZE), dtype=jnp.float32)
    critic = jnp.zeros((2, VOCAB_SIZE, MAX_TOKENS + 1), dtype=jnp.float32)
    reference = jax.lax.stop_gradient(policy)
    return policy, critic, reference


def continuation_value_targets(reward_weights):
    """Frozen reward for the recipe's deterministic one-token timeout continuation."""
    tasks = jnp.repeat(jnp.arange(2, dtype=jnp.int32), 3)
    previous = jnp.tile(
        jnp.asarray([TOKENS["responseA"], TOKENS["responseB"], TOKENS["responseC"]]), 2
    )
    continuation = jnp.where(tasks == 0, TOKENS["responseA"], TOKENS["responseC"])
    targets = reward_weights[tasks, continuation]
    return (
        jax.lax.stop_gradient(tasks),
        jax.lax.stop_gradient(previous),
        jax.lax.stop_gradient(targets),
    )


def rollout(policy, critic, reference, reward_weights, prompt_ids, key, *, beta=0.03):
    """TODO: sample the fixed synthetic prompt IDs. Return a record mapping with time-major [time, batch] arrays, frozen behavior/reference log probabilities, role/padding masks, rewards, old values and post-action values."""
    raise NotImplementedError(
        "implement fixed-key ragged rollout collection and retained record fields"
    )


def sequence_reward(reward_weights, task_ids, actions, response_mask):
    """Score observed token counts with the frozen, pair-fitted reward model."""
    token_scores = reward_weights[task_ids[:, None], actions.T]
    return jnp.sum(jnp.where(response_mask.T, token_scores, 0.0), axis=-1)


def generalized_advantage_estimate(records, gamma=0.9, lam=0.8):
    """TODO: compute frozen reverse GAE. All time-major record arrays have shape [time, batch]; return advantages of that same shape with zeros at padding."""
    raise NotImplementedError(
        "implement EOS-zero-bootstrap and timeout-bootstrap GAE with stopped lambda traces"
    )


def ppo_policy_loss(new_logps, old_logps, advantages, response_mask, clip=0.2):
    """TODO: compute the signed clipped PPO policy loss. Inputs are same-shaped [time, batch] log probabilities, advantages and valid target-token mask; return one scalar mean over valid tokens."""
    raise NotImplementedError(
        "implement the signed minimum clipped surrogate with frozen old log probabilities"
    )


def clipped_value_loss(new_values, old_values, targets, response_mask, clip=0.2):
    """TODO: compute the clipped critic loss. Inputs are same-shaped [time, batch] values, frozen old values, frozen value targets and valid target-token mask; return one scalar mean over valid tokens."""
    raise NotImplementedError(
        "implement clipped value regression using frozen rollout targets"
    )


def _current_values(critic, records):
    time, batch = records["actions"].shape
    task = jnp.broadcast_to(records["task_ids"][None, :], (time, batch))
    positions = jnp.broadcast_to(jnp.arange(time)[:, None], (time, batch))
    return critic[task, records["previous_tokens"], positions]


def _tree_sha256(tree):
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        digest.update(np.asarray(leaf, dtype=np.float32).tobytes(order="C"))
    return digest.hexdigest()


def verify_frozen_components(
    reference, reward_weights, reference_sha256, reward_sha256
):
    """Reject changed reference or fitted reward arrays at an update boundary."""
    if _tree_sha256(reference) != reference_sha256:
        raise ValueError("frozen reference hash changed")
    if _array_sha256(reward_weights) != reward_sha256:
        raise ValueError("frozen reward-model hash changed")


def _json_array(value):
    array = np.asarray(value)
    return array.tolist()


def _records_json(records, advantages):
    result = {}
    for name, value in records.items():
        if name.endswith("sha256"):
            result[name] = value
        elif hasattr(value, "shape"):
            result[name] = _json_array(value)
    result["advantages"] = _json_array(advantages)
    result["value_targets"] = _json_array(records["old_values"] + advantages)
    return result


def _initial_policy():
    return initialize()[0]


def train(*, seed=None, updates=None, batch_size=None, epochs=None, learning_rate=None):
    """TODO: run the recipe-defined updates. Return policy, critic, frozen reference/reward arrays, update evidence and fixed heldout evaluation with failures retained."""
    raise NotImplementedError(
        "implement fresh-rollout sequence PPO updates and retained heldout evidence"
    )


def _categorical_kl(policy, reference, task_ids, previous_tokens, positions):
    logits = policy[task_ids, previous_tokens, positions, :]
    ref_logits = reference[task_ids, previous_tokens, positions, :]
    logits = jax.vmap(_masked_logits)(logits, positions)
    ref_logits = jax.vmap(_masked_logits)(ref_logits, positions)
    logp = jax.nn.log_softmax(logits, axis=-1)
    ref_logp = jax.nn.log_softmax(ref_logits, axis=-1)
    probs = jnp.exp(logp)
    return jnp.sum(probs * (logp - ref_logp), axis=-1)


def evaluate(policy, reference, reward_weights, *, prompt_ids, seed, beta=0.03):
    """Fixed-prompt sampled reward and exact categorical KL, with explicit units."""
    records = rollout(
        policy,
        jnp.zeros((2, VOCAB_SIZE, MAX_TOKENS + 1), dtype=jnp.float32),
        reference,
        reward_weights,
        jnp.asarray(prompt_ids, dtype=jnp.int32),
        jax.random.key(seed),
        beta=beta,
    )
    time, batch = records["actions"].shape
    tasks = jnp.broadcast_to(records["task_ids"][None, :], (time, batch))
    positions = jnp.broadcast_to(jnp.arange(time)[:, None], (time, batch))
    exact = _categorical_kl(
        policy,
        reference,
        tasks.reshape(-1),
        records["previous_tokens"].reshape(-1),
        positions.reshape(-1),
    ).reshape(time, batch)
    mask = records["response_mask"]
    token_count = int(jnp.sum(mask))
    sequence_count = int(batch)
    token_kl = jnp.where(mask, exact, 0.0)
    per_sequence_kl = jnp.sum(token_kl, axis=0)
    scores = np.asarray(records["sequence_scores"], dtype=float)
    reasons = []
    if (
        token_count < 1
        or not np.isfinite(scores).all()
        or not np.isfinite(np.asarray(token_kl)).all()
    ):
        reasons.append("nonfinite_or_empty_heldout_evaluation")
    return {
        "prompt_ids": [int(value) for value in prompt_ids],
        "sequence_reward_scores": scores.tolist(),
        "sequence_reward_mean": float(np.mean(scores)),
        "sequence_denominator": sequence_count,
        "exact_kl_per_valid_token": float(jnp.sum(token_kl) / max(token_count, 1)),
        "sum_visited_context_exact_categorical_kl_per_sequence": float(
            jnp.sum(per_sequence_kl) / max(sequence_count, 1)
        ),
        "valid_response_tokens": token_count,
        "token_denominator": token_count,
        "visited_context_categorical_kl_sums_by_sequence": np.asarray(
            per_sequence_kl
        ).tolist(),
        "sampled_rollout_kl_per_valid_token": float(
            jnp.sum(jnp.where(mask, records["sampled_kl"], 0.0)) / max(token_count, 1)
        ),
        "failure_reasons": reasons,
    }


def _tree_to_json(params):
    return _json_array(params)


def save_state(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        policy=np.asarray(result["policy"]),
        critic=np.asarray(result["critic"]),
        reference=np.asarray(result["reference"]),
        reward_weights=np.asarray(result["reward_weights"]),
        recipe_sha256=np.asarray(result["recipe_sha256"]),
    )
    return path


def load_state(path):
    with np.load(path, allow_pickle=False) as state:
        if str(state["recipe_sha256"].item()) != RECIPE_SHA256:
            raise ValueError(
                "saved state recipe hash does not match the checked-in fixture"
            )
        return {
            name: jnp.asarray(state[name])
            for name in ("policy", "critic", "reference", "reward_weights")
        }
