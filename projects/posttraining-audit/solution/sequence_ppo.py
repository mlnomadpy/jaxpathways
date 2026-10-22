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
    """Fit a tiny pairwise reward model, then return its frozen token weights."""
    pairs = RECIPE["preferencePairs"]
    tasks = jnp.asarray(
        [RECIPE["promptInstances"][p["promptId"]]["task"] for p in pairs]
    )

    def counts(sequences):
        out = np.zeros((len(sequences), VOCAB_SIZE), dtype=np.float32)
        for row, sequence in enumerate(sequences):
            for token in sequence:
                if token not in (PAD, BOS, EOS):
                    out[row, token] += 1.0
        return jnp.asarray(out)

    chosen = counts([pair["chosen"] for pair in pairs])
    rejected = counts([pair["rejected"] for pair in pairs])
    weights = jnp.zeros((2, VOCAB_SIZE), dtype=jnp.float32)
    optimizer = optax.adam(RECIPE["rewardFit"]["learningRate"])
    opt_state = optimizer.init(weights)

    def loss_fn(candidate):
        chosen_score = jnp.sum(candidate[tasks] * chosen, axis=-1)
        rejected_score = jnp.sum(candidate[tasks] * rejected, axis=-1)
        margin = chosen_score - rejected_score
        penalty = 0.01 * jnp.mean(jnp.square(candidate))
        return jnp.mean(jax.nn.softplus(-margin)) + penalty

    history = []
    for _ in range(RECIPE["rewardFit"]["updates"]):
        loss, grads = jax.value_and_grad(loss_fn)(weights)
        updates, opt_state = optimizer.update(grads, opt_state, weights)
        weights = optax.apply_updates(weights, updates)
        history.append(float(loss))
    weights = jax.lax.stop_gradient(weights)
    chosen_scores = np.asarray(jnp.sum(weights[tasks] * chosen, axis=-1))
    rejected_scores = np.asarray(jnp.sum(weights[tasks] * rejected, axis=-1))
    return weights, {
        "updates": len(history),
        "initial_pairwise_loss": history[0],
        "final_pairwise_loss": float(loss_fn(weights)),
        "chosen_scores": chosen_scores.tolist(),
        "rejected_scores": rejected_scores.tolist(),
        "margins": (chosen_scores - rejected_scores).tolist(),
        "weights_sha256": _array_sha256(weights),
    }


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
    """Sample a ragged response batch and retain frozen behavior/evidence arrays."""
    prompt_ids = jnp.asarray(prompt_ids, dtype=jnp.int32)
    if prompt_ids.ndim != 1 or prompt_ids.shape[0] < 1:
        raise ValueError("prompt_ids must be a nonempty vector")
    if bool(jnp.any(prompt_ids < 0)) or bool(
        jnp.any(prompt_ids >= len(TASK_FOR_PROMPT))
    ):
        raise ValueError("unknown synthetic prompt id")
    task_ids = TASK_FOR_PROMPT[prompt_ids]
    batch = prompt_ids.shape[0]
    active = jnp.ones((batch,), dtype=bool)
    previous = jnp.full((batch,), BOS, dtype=jnp.int32)
    keys = jax.random.split(key, MAX_TOKENS)
    fields = {
        name: []
        for name in (
            "previous_tokens",
            "actions",
            "response_mask",
            "target_role_mask",
            "input_role_mask",
            "padding_mask",
            "terminated",
            "truncated",
            "old_logps",
            "reference_logps",
            "sampled_kl",
            "old_values",
            "next_values",
        )
    }
    for position in range(MAX_TOKENS):
        logits = policy[task_ids, previous, position, :]
        logits = jax.vmap(_masked_logits)(logits, jnp.full((batch,), position))
        sampled = jax.random.categorical(keys[position], logits, axis=-1).astype(
            jnp.int32
        )
        actions = jnp.where(active, sampled, PAD)
        old_lp = jax.nn.log_softmax(logits, axis=-1)
        old_lp = jnp.take_along_axis(
            old_lp, jnp.clip(actions, 0, VOCAB_SIZE - 1)[:, None], axis=-1
        )[:, 0]
        ref_logits = reference[task_ids, previous, position, :]
        ref_logits = jax.vmap(_masked_logits)(ref_logits, jnp.full((batch,), position))
        ref_lp = jax.nn.log_softmax(ref_logits, axis=-1)
        ref_lp = jnp.take_along_axis(
            ref_lp, jnp.clip(actions, 0, VOCAB_SIZE - 1)[:, None], axis=-1
        )[:, 0]
        old_lp = jnp.where(active, old_lp, 0.0)
        ref_lp = jnp.where(active, ref_lp, 0.0)
        old_value = critic[task_ids, previous, position]
        old_value = jnp.where(active, old_value, 0.0)
        next_value = critic[task_ids, actions, position + 1]
        next_value = jnp.where(active, next_value, 0.0)
        terminal = active & (actions == EOS)
        truncation = active & ~terminal & (position == MAX_TOKENS - 1)
        for name, value in (
            ("previous_tokens", previous),
            ("actions", actions),
            ("response_mask", active),
            ("target_role_mask", active),
            ("input_role_mask", active & (position > 0)),
            ("padding_mask", ~active),
            ("terminated", terminal),
            ("truncated", truncation),
            ("old_logps", old_lp),
            ("reference_logps", ref_lp),
            ("sampled_kl", old_lp - ref_lp),
            ("old_values", old_value),
            ("next_values", next_value),
        ):
            fields[name].append(value)
        active = active & ~terminal
        previous = jnp.where(active, actions, PAD)
    records = {name: jnp.stack(values, axis=0) for name, values in fields.items()}
    records["prompt_ids"] = prompt_ids
    records["task_ids"] = task_ids
    records["lengths"] = jnp.sum(records["response_mask"], axis=0).astype(jnp.int32)
    records["sequence_scores"] = sequence_reward(
        reward_weights, task_ids, records["actions"], records["response_mask"]
    )
    final_index = jnp.maximum(records["lengths"] - 1, 0)
    positions = jnp.arange(MAX_TOKENS)[:, None]
    terminal_reward = jnp.where(
        positions == final_index[None, :], records["sequence_scores"][None, :], 0.0
    )
    records["reward_model_terminal"] = jnp.where(
        records["response_mask"], terminal_reward, 0.0
    )
    records["rewards"] = jnp.where(
        records["response_mask"],
        records["reward_model_terminal"] - beta * records["sampled_kl"],
        0.0,
    )
    records["behavior_policy_sha256"] = _tree_sha256(policy)
    records["reference_sha256"] = _tree_sha256(reference)
    records["reward_model_sha256"] = _array_sha256(reward_weights)
    return records


def sequence_reward(reward_weights, task_ids, actions, response_mask):
    """Score observed token counts with the frozen, pair-fitted reward model."""
    token_scores = reward_weights[task_ids[:, None], actions.T]
    return jnp.sum(jnp.where(response_mask.T, token_scores, 0.0), axis=-1)


def generalized_advantage_estimate(records, gamma=0.9, lam=0.8):
    """Reverse GAE: EOS has zero bootstrap; timeout bootstraps but stops trace."""
    reward = records["rewards"]
    active = jnp.asarray(records["response_mask"], dtype=bool)
    terminated = jnp.asarray(records["terminated"], dtype=bool)
    truncated = jnp.asarray(records["truncated"], dtype=bool)
    old_values = jax.lax.stop_gradient(records["old_values"])
    next_values = jax.lax.stop_gradient(records["next_values"])
    deltas = jnp.where(
        active,
        reward + gamma * (active & ~terminated) * next_values - old_values,
        0.0,
    )
    trace_mask = active & ~(terminated | truncated)

    # Reverse scan takes a mask through its input so padded rows zero the carry.
    def reverse_step(carry, values):
        delta, trace, is_active = values
        advantage = delta + gamma * lam * trace * carry
        advantage = jnp.where(is_active, advantage, 0.0)
        return advantage, advantage

    _, reversed_advantages = jax.lax.scan(
        reverse_step,
        jnp.zeros((reward.shape[1],), dtype=reward.dtype),
        (deltas[::-1], trace_mask[::-1], active[::-1]),
    )
    return jax.lax.stop_gradient(reversed_advantages[::-1])


def ppo_policy_loss(new_logps, old_logps, advantages, response_mask, clip=0.2):
    """Signed PPO clipped surrogate reduced over valid response tokens."""
    mask = jnp.asarray(response_mask, dtype=bool)
    denom = jnp.sum(mask)
    safe_denom = jnp.maximum(denom, 1)
    old_logps = jax.lax.stop_gradient(old_logps)
    advantages = jax.lax.stop_gradient(advantages)
    ratio = jnp.exp(new_logps - old_logps)
    clipped_ratio = jnp.clip(ratio, 1.0 - clip, 1.0 + clip)
    surrogate = jnp.minimum(ratio * advantages, clipped_ratio * advantages)
    return -jnp.sum(jnp.where(mask, surrogate, 0.0)) / safe_denom


def clipped_value_loss(new_values, old_values, targets, response_mask, clip=0.2):
    """PPO clipped critic loss with a frozen old value and valid-token mean."""
    mask = jnp.asarray(response_mask, dtype=bool)
    denom = jnp.maximum(jnp.sum(mask), 1)
    old_values = jax.lax.stop_gradient(old_values)
    targets = jax.lax.stop_gradient(targets)
    clipped = old_values + jnp.clip(new_values - old_values, -clip, clip)
    plain_error = jnp.square(new_values - targets)
    clipped_error = jnp.square(clipped - targets)
    loss = 0.5 * jnp.maximum(plain_error, clipped_error)
    return jnp.sum(jnp.where(mask, loss, 0.0)) / denom


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
    """Fit reward pairs, then train actor and critic on fresh frozen rollouts."""
    cfg = RECIPE["rollout"]
    seed = cfg["seed"] if seed is None else int(seed)
    updates = cfg["updates"] if updates is None else int(updates)
    batch_size = cfg["batchSize"] if batch_size is None else int(batch_size)
    epochs = cfg["innerEpochs"] if epochs is None else int(epochs)
    learning_rate = (
        cfg["learningRate"] if learning_rate is None else float(learning_rate)
    )
    if updates < 1 or batch_size < 2 or epochs < 2 or learning_rate <= 0:
        raise ValueError(
            "positive updates, batch size, at least two epochs and learning rate required"
        )
    prompt_base = np.asarray(cfg["trainingPromptIds"], dtype=np.int32)
    if batch_size > len(prompt_base):
        prompt_ids = np.resize(prompt_base, batch_size)
    else:
        prompt_ids = prompt_base[:batch_size]
    policy, critic, reference = initialize(seed)
    reward_weights, reward_fit = fit_reward_model()
    reference = jax.lax.stop_gradient(reference)
    reward_weights = jax.lax.stop_gradient(reward_weights)
    reference_hash = _tree_sha256(reference)
    reward_hash = _array_sha256(reward_weights)
    policy_optimizer = optax.adam(learning_rate)
    critic_optimizer = optax.adam(learning_rate)
    policy_opt_state = policy_optimizer.init(policy)
    critic_opt_state = critic_optimizer.init(critic)
    continuation_tasks, continuation_previous, continuation_targets = (
        continuation_value_targets(reward_weights)
    )
    key = jax.random.key(seed)
    update_trace = []
    for update_index in range(updates):
        verify_frozen_components(reference, reward_weights, reference_hash, reward_hash)
        key, rollout_key = jax.random.split(key)
        records = rollout(
            policy,
            critic,
            reference,
            reward_weights,
            jnp.asarray(prompt_ids),
            rollout_key,
            beta=cfg["klCoefficient"],
        )
        advantages = generalized_advantage_estimate(
            records, cfg["gamma"], cfg["gaeLambda"]
        )
        targets = jax.lax.stop_gradient(records["old_values"] + advantages)
        old_logps = jax.lax.stop_gradient(records["old_logps"])
        old_values = jax.lax.stop_gradient(records["old_values"])
        old_logp_hash_before = _array_sha256(old_logps)
        old_value_hash_before = _array_sha256(old_values)
        epoch_logps = []
        policy_losses = []
        critic_losses = []
        continuation_values_by_epoch = []
        for _epoch in range(epochs):

            def policy_loss_fn(candidate):
                new_logps = _selected_log_probs_for_records(candidate, records)
                return ppo_policy_loss(
                    new_logps,
                    old_logps,
                    advantages,
                    records["response_mask"],
                    cfg["policyClip"],
                )

            p_loss, p_grads = jax.value_and_grad(policy_loss_fn)(policy)
            p_updates, policy_opt_state = policy_optimizer.update(
                p_grads, policy_opt_state, policy
            )
            policy = optax.apply_updates(policy, p_updates)

            def value_loss_fn(candidate):
                new_values = _current_values(candidate, records)
                rollout_loss = clipped_value_loss(
                    new_values,
                    old_values,
                    targets,
                    records["response_mask"],
                    cfg["valueClip"],
                )
                continuation_values = candidate[
                    continuation_tasks, continuation_previous, MAX_TOKENS
                ]
                continuation_loss = 0.5 * jnp.mean(
                    jnp.square(continuation_values - continuation_targets)
                )
                return rollout_loss + continuation_loss

            c_loss, c_grads = jax.value_and_grad(value_loss_fn)(critic)
            c_updates, critic_opt_state = critic_optimizer.update(
                c_grads, critic_opt_state, critic
            )
            critic = optax.apply_updates(critic, c_updates)
            policy_losses.append(float(p_loss))
            critic_losses.append(float(c_loss))
            epoch_logps.append(
                _json_array(_selected_log_probs_for_records(policy, records))
            )
            continuation_values_by_epoch.append(
                _json_array(
                    critic[continuation_tasks, continuation_previous, MAX_TOKENS]
                )
            )
        verify_frozen_components(reference, reward_weights, reference_hash, reward_hash)
        old_logp_hash_after = _array_sha256(old_logps)
        old_value_hash_after = _array_sha256(old_values)
        update_trace.append(
            {
                "update": update_index + 1,
                "frozen_reference_sha256": reference_hash,
                "frozen_reward_sha256": reward_hash,
                "behavior_policy_sha256": records["behavior_policy_sha256"],
                "old_logps_sha256_before_epochs": old_logp_hash_before,
                "old_logps_sha256_after_epochs": old_logp_hash_after,
                "old_values_sha256_before_epochs": old_value_hash_before,
                "old_values_sha256_after_epochs": old_value_hash_after,
                "policy_losses_by_epoch": policy_losses,
                "critic_losses_by_epoch": critic_losses,
                "inner_epoch_selected_logps": epoch_logps,
                "continuation_value_targets": _json_array(continuation_targets),
                "continuation_values_by_epoch": continuation_values_by_epoch,
                "rollout": _records_json(records, advantages),
            }
        )
    result = {
        "policy": policy,
        "critic": critic,
        "reference": reference,
        "reward_weights": reward_weights,
        "reward_fit": reward_fit,
        "updates": update_trace,
        "frozen_reference_sha256": reference_hash,
        "frozen_reward_sha256": reward_hash,
        "continuation_value_targets": _json_array(continuation_targets),
        "continuation_values": _json_array(
            critic[continuation_tasks, continuation_previous, MAX_TOKENS]
        ),
        "seed": seed,
        "recipe_sha256": RECIPE_SHA256,
    }
    result["evaluation"] = evaluate(
        policy,
        reference,
        reward_weights,
        prompt_ids=cfg["heldoutPromptIds"],
        seed=seed + 100_000,
        beta=cfg["klCoefficient"],
    )
    result["reference_evaluation"] = evaluate(
        reference,
        reference,
        reward_weights,
        prompt_ids=cfg["heldoutPromptIds"],
        seed=seed + 100_000,
        beta=0.0,
    )
    reward_ok = (
        result["evaluation"]["sequence_reward_mean"]
        > result["reference_evaluation"]["sequence_reward_mean"]
    )
    kl_ok = (
        result["evaluation"]["exact_kl_per_valid_token"]
        <= cfg["maximumHeldoutExactKlPerToken"]
    )
    result["evaluation"]["reward_pass"] = bool(reward_ok)
    result["evaluation"]["kl_pass"] = bool(kl_ok)
    if not reward_ok:
        result["evaluation"]["failure_reasons"].append(
            "heldout_reward_not_above_reference"
        )
    if not kl_ok:
        result["evaluation"]["failure_reasons"].append(
            "heldout_exact_kl_above_predeclared_limit"
        )
    result["evaluation"]["status"] = (
        "passed"
        if reward_ok and kl_ok and not result["evaluation"]["failure_reasons"]
        else "failed"
    )
    return result


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
