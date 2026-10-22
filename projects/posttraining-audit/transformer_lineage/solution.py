"""One tiny sequence PPO objective on a checksum-pinned saved decoder."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
import optax

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("TRANSFORMER_LINEAGE_PROJECT_ROOT", HERE))
RECIPE = json.loads((PROJECT_ROOT / "recipe.json").read_text())
FIXTURE = json.loads((PROJECT_ROOT / RECIPE["fixture"]["path"]).read_text())
ACTION_IDS = jnp.asarray(RECIPE["actions"]["restrictedVocabularyIds"], jnp.int32)
ACTION_INDEX = {int(token): index for index, token in enumerate(np.asarray(ACTION_IDS))}
EOS = int(RECIPE["actions"]["singleTokenSpellings"]["eos"]["id"])
PAD = int(RECIPE["actions"]["paddingTokenId"])
MAX_ACTIONS = int(RECIPE["actions"]["maximumResponseActions"])
MIN_NON_EOS = int(RECIPE["actions"]["minimumNonEosActions"])


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tree_sha256(tree) -> str:
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        array = np.asarray(leaf)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_verified_parent(repo_root: Path):
    """Use the parent's checked package loader after check.py's byte preflight."""
    from tokenizers import Tokenizer

    parent_dir = repo_root / RECIPE["parent"]["packagePath"]
    foundation_dir = repo_root / "projects/text-harness/foundations"
    for filename, expected in RECIPE["parent"]["foundationSourceSha256"].items():
        actual = _sha256((foundation_dir / filename).read_bytes())
        if actual != expected:
            raise ValueError(
                f"parent foundation source changed before import: {filename}"
            )
    package_name = (
        f"_lineage_parent_{RECIPE['parent']['foundationSourceSha256']['train.py'][:16]}"
    )
    package = types.ModuleType(package_name)
    package.__path__ = [str(foundation_dir)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    module_name = f"{package_name}.train"
    spec = importlib.util.spec_from_file_location(
        module_name, foundation_dir / "train.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("could not load the pinned parent training module")
    parent_train = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = parent_train
    spec.loader.exec_module(parent_train)
    expected_modules = {
        "train": (parent_train, foundation_dir / "train.py"),
        "model": (parent_train.model, foundation_dir / "model.py"),
        "config": (sys.modules[f"{package_name}.config"], foundation_dir / "config.py"),
    }
    for label, (module, expected_path) in expected_modules.items():
        if Path(module.__file__).resolve() != expected_path.resolve():
            raise ImportError(
                f"pinned parent {label} module resolved from an unexpected path"
            )
    if parent_train.code_identity() != RECIPE["parent"]["foundationSourceSha256"]:
        raise ImportError(
            "loaded parent module identity differs from the pinned foundation sources"
        )

    package_manifest = json.loads((parent_dir / "package.json").read_text())
    parent_params, config, loaded_manifest = parent_train.load_model_package(
        parent_dir, package_manifest["identity"]
    )
    tokenizer = Tokenizer.from_file(str(parent_dir / "tokenizer.json"))
    return {
        "params": parent_params,
        "config": config,
        "manifest": loaded_manifest,
        "tokenizer": tokenizer,
        "model_module": parent_train.model,
        "parent_train_module": parent_train,
        "params_sha256": _tree_sha256(parent_params),
        "package_sha256": _sha256((parent_dir / "package.json").read_bytes()),
        "processor_sha256": _sha256((parent_dir / "tokenizer.json").read_bytes()),
    }


def verify_prompt_and_action_ids(parent):
    tokenizer = parent["tokenizer"]
    for item in FIXTURE["prompts"]:
        actual = tokenizer.encode(item["text"], add_special_tokens=False).ids
        if actual != item["tokenIds"]:
            raise ValueError(
                f"prompt tokenization changed for fixture prompt {item['id']}"
            )
    for name, spec in RECIPE["actions"]["singleTokenSpellings"].items():
        if name == "eos":
            continue
        encoded = tokenizer.encode(spec["text"], add_special_tokens=False).ids
        if encoded != [int(spec["id"])]:
            raise ValueError(f"action {name} is not the pinned single token")
    if parent["tokenizer"].token_to_id("[EOS]") != EOS:
        raise ValueError("parent EOS ID mismatch")


def initialize(parent, seed=None):
    """Copy the verified 10,880-parameter actor and reset objective optimizers."""
    if seed is None:
        seed = int(RECIPE["training"]["seed"])
    actor = jax.tree.map(lambda value: jnp.array(value), parent["params"])
    reference = jax.tree.map(jax.lax.stop_gradient, parent["params"])
    critic = {
        "weight": jnp.zeros((len(ACTION_IDS),), dtype=jnp.float32),
        "bias": jnp.asarray(0.1, dtype=jnp.float32),
    }
    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(RECIPE["training"]["gradientClipGlobalNorm"]),
        optax.adam(RECIPE["training"]["actorLearningRate"]),
    )
    critic_optimizer = optax.chain(
        optax.clip_by_global_norm(RECIPE["training"]["gradientClipGlobalNorm"]),
        optax.adam(RECIPE["training"]["criticLearningRate"]),
    )
    state = {
        "actor_params": actor,
        "critic_params": critic,
        "actor_opt_state": actor_optimizer.init(actor),
        "critic_opt_state": critic_optimizer.init(critic),
        "rng_key": np.asarray(jax.random.PRNGKey(seed), dtype=np.uint32).tolist(),
        "prompt_cursor": 0,
        "update_clock": 0,
        "inner_epoch_clock": 0,
        "parent_params_sha256": parent["params_sha256"],
        "parent_package_sha256": parent["package_sha256"],
        "processor_sha256": parent["processor_sha256"],
        "source_identity": {},
    }
    if _tree_sha256(state["actor_params"]) != parent["params_sha256"]:
        raise AssertionError("actor initialization differs from the frozen parent")
    return state, reference, actor_optimizer, critic_optimizer


def _prompt_item(prompt_id):
    return FIXTURE["prompts"][int(prompt_id)]


def _full_prefix(prompt_id, prior_actions=()):
    return [1, *map(int, _prompt_item(prompt_id)["tokenIds"]), *map(int, prior_actions)]


def _masked_model_logits(model_module, params, prefix, config):
    tokens = jnp.asarray(prefix, dtype=jnp.int32)[None, :]
    segments = jnp.ones_like(tokens, dtype=jnp.int32)
    return model_module.logits(
        params,
        tokens,
        config,
        valid_mask=jnp.ones_like(tokens, dtype=bool),
        segment_ids=segments,
    )[0, -1]


def _restricted_log_probs(full_logits, eos_eligible):
    selected = jnp.take(full_logits.astype(jnp.float32), ACTION_IDS, axis=-1)
    eos_index = ACTION_INDEX[EOS]
    selected = selected.at[eos_index].set(
        jnp.where(eos_eligible, selected[eos_index], jnp.asarray(-1.0e9, jnp.float32))
    )
    log_probs = jax.nn.log_softmax(selected, axis=-1)
    if not eos_eligible:
        log_probs = log_probs.at[eos_index].set(-jnp.inf)
    return log_probs


def _context_features(parent, prompt_id, prior_actions=(), *, eos_eligible=False):
    logits = _masked_model_logits(
        parent["model_module"],
        parent["params"],
        _full_prefix(prompt_id, prior_actions),
        parent["config"],
    )
    log_probs = _restricted_log_probs(logits, eos_eligible)
    return jax.lax.stop_gradient(jnp.where(jnp.isfinite(log_probs), log_probs, 0.0))


def _reward_features(parent, prompt_id, actions):
    context = _context_features(parent, prompt_id, (), eos_eligible=False)
    counts = np.zeros((3,), dtype=np.float32)
    token_to_count = {383: 0, 381: 1, 4: 2}
    for token in actions:
        if int(token) == EOS:
            break
        counts[token_to_count[int(token)]] += 1.0
    interaction = jnp.asarray(counts)[:, None] * context[None, :]
    return interaction.reshape(-1)


def _pair_features(parent):
    rows = []
    for pair in FIXTURE["preferencePairs"]:
        rows.append(
            (
                _reward_features(
                    parent, pair["promptId"], pair["chosenActionSequence"]
                ),
                _reward_features(
                    parent, pair["promptId"], pair["rejectedActionSequence"]
                ),
                int(pair["preferenceLabel"]),
            )
        )
    return rows


def fit_pairwise_reward_head(parent, *, max_updates=None):
    """Fit an interaction head from only the four fixed training comparisons."""
    pairs = _pair_features(parent)
    chosen = jnp.stack([row[0] for row in pairs])
    rejected = jnp.stack([row[1] for row in pairs])
    labels = jnp.asarray([row[2] for row in pairs], dtype=jnp.float32)
    initial = jnp.zeros((len(ACTION_IDS) * 3,), dtype=jnp.float32)
    optimizer = optax.adam(RECIPE["training"]["rewardLearningRate"])
    opt_state = optimizer.init(initial)

    def loss_fn(weight):
        chosen_score = chosen @ weight
        rejected_score = rejected @ weight
        signed_margin = (2.0 * labels - 1.0) * (chosen_score - rejected_score)
        return jnp.mean(jax.nn.softplus(-signed_margin)) + 1e-3 * jnp.mean(weight**2)

    first_loss = float(loss_fn(initial))
    weight = initial
    fit_trace = [
        np.asarray(weight, dtype=np.float32).reshape((3, len(ACTION_IDS))).tolist()
    ]
    fit_updates = int(RECIPE["training"]["rewardFitUpdates"])
    if max_updates is not None:
        fit_updates = min(fit_updates, max(0, int(max_updates)))
    for _ in range(fit_updates):
        loss, grads = jax.value_and_grad(loss_fn)(weight)
        updates, opt_state = optimizer.update(grads, opt_state, weight)
        weight = optax.apply_updates(weight, updates)
        fit_trace.append(
            np.asarray(weight, dtype=np.float32).reshape((3, len(ACTION_IDS))).tolist()
        )
    weight = jax.lax.stop_gradient(weight)
    return {"weight": weight.reshape((3, len(ACTION_IDS)))}, {
        "pair_count": len(pairs),
        "fit_updates": fit_updates,
        "fit_trace": fit_trace,
        "initial_pairwise_loss": first_loss,
        "final_pairwise_loss": float(loss_fn(weight)),
        "training_pair_margins": [
            float((row[0] - row[1]) @ weight) * (2 * row[2] - 1) for row in pairs
        ],
        "reward_head_sha256": _tree_sha256(weight),
    }


def _reward_score(reward_params, parent, prompt_id, actions):
    features = _reward_features(parent, prompt_id, actions)
    return jnp.dot(features, reward_params["weight"].reshape(-1))


def _critic_value(critic_params, features):
    return jnp.dot(critic_params["weight"], features) + critic_params["bias"]


def rollout(
    parent,
    actor_params,
    critic_params,
    reference_params,
    reward_params,
    prompt_ids,
    key,
    *,
    forced_action_sequences=None,
):
    """Collect full-prefix autoregressive actions with a shared restricted support."""
    tcfg, action_count = RECIPE["training"], len(ACTION_IDS)
    prompt_ids = [int(value) for value in prompt_ids]
    if forced_action_sequences is not None and len(forced_action_sequences) != len(
        prompt_ids
    ):
        raise ValueError("forced action sequence count must match prompt batch")
    batch = len(prompt_ids)
    max_steps = MAX_ACTIONS
    token_ids = np.zeros((batch, max_steps), dtype=np.int32)
    target_role_mask = np.zeros((batch, max_steps), dtype=bool)
    input_role_mask = np.zeros((batch, max_steps), dtype=bool)
    padding_mask = np.ones((batch, max_steps), dtype=bool)
    terminated = np.zeros((batch, max_steps), dtype=bool)
    truncated = np.zeros((batch, max_steps), dtype=bool)
    old_logps = np.zeros((batch, max_steps), dtype=np.float32)
    parent_logps = np.zeros((batch, max_steps), dtype=np.float32)
    old_restricted_probabilities = np.zeros(
        (batch, max_steps, action_count), dtype=np.float32
    )
    parent_restricted_probabilities = np.zeros(
        (batch, max_steps, action_count), dtype=np.float32
    )
    eligible_action_mask = np.zeros((batch, max_steps, action_count), dtype=bool)
    sampled_kl = np.zeros((batch, max_steps), dtype=np.float32)
    exact_conditional_kl = np.zeros((batch, max_steps), dtype=np.float32)
    old_values = np.zeros((batch, max_steps), dtype=np.float32)
    post_action_values = np.zeros((batch, max_steps), dtype=np.float32)
    current_features = np.zeros((batch, max_steps, action_count), dtype=np.float32)
    prefixes = [[[] for _ in range(max_steps)] for _ in prompt_ids]
    next_key, *sample_keys = jax.random.split(key, batch * max_steps + 1)
    cursor = 0
    sequences = []
    lengths = np.zeros((batch,), dtype=np.int32)
    for row, prompt_id in enumerate(prompt_ids):
        prior = []
        actions = []
        active = True
        for position in range(max_steps):
            if not active:
                continue
            prefix = _full_prefix(prompt_id, prior)
            prefixes[row][position] = prefix
            eligible = len(prior) >= MIN_NON_EOS
            logits = _masked_model_logits(
                parent["model_module"], actor_params, prefix, parent["config"]
            )
            old_dist = _restricted_log_probs(logits, eligible)
            ref_logits = _masked_model_logits(
                parent["model_module"], reference_params, prefix, parent["config"]
            )
            ref_dist = _restricted_log_probs(ref_logits, eligible)
            feature = _context_features(parent, prompt_id, prior, eos_eligible=eligible)
            old_value = _critic_value(critic_params, feature)
            if forced_action_sequences is None:
                action_index = int(
                    jax.random.categorical(sample_keys[cursor], old_dist)
                )
            else:
                forced = forced_action_sequences[row]
                if position >= len(forced):
                    raise ValueError("forced sequence ended before EOS or timeout")
                forced_action = int(forced[position])
                if forced_action not in ACTION_INDEX:
                    raise ValueError(
                        "forced action is outside the restricted vocabulary"
                    )
                action_index = ACTION_INDEX[forced_action]
                if not bool(jnp.isfinite(old_dist[action_index])):
                    raise ValueError("forced action violates the eligible support")
            cursor += 1
            action = int(ACTION_IDS[action_index])
            chosen_old_logp = float(old_dist[action_index])
            chosen_parent_logp = float(ref_dist[action_index])
            actor_probabilities = jnp.where(
                jnp.isfinite(old_dist), jnp.exp(old_dist), 0.0
            )
            parent_probabilities = jnp.where(
                jnp.isfinite(ref_dist), jnp.exp(ref_dist), 0.0
            )
            exact_kl = jnp.sum(
                jnp.where(
                    actor_probabilities > 0,
                    actor_probabilities * (old_dist - ref_dist),
                    0.0,
                )
            )
            token_ids[row, position] = action
            target_role_mask[row, position] = True
            input_role_mask[row, position] = position > 0
            padding_mask[row, position] = False
            old_logps[row, position] = chosen_old_logp
            parent_logps[row, position] = chosen_parent_logp
            old_restricted_probabilities[row, position] = np.asarray(
                actor_probabilities
            )
            parent_restricted_probabilities[row, position] = np.asarray(
                parent_probabilities
            )
            eligible_action_mask[row, position] = np.isfinite(np.asarray(old_dist))
            sampled_kl[row, position] = chosen_old_logp - chosen_parent_logp
            exact_conditional_kl[row, position] = float(exact_kl)
            old_values[row, position] = float(old_value)
            current_features[row, position] = np.asarray(feature)
            actions.append(action)
            prior.append(action)
            lengths[row] += 1
            is_terminal = action == EOS
            is_truncated = position == max_steps - 1 and not is_terminal
            terminated[row, position] = is_terminal
            truncated[row, position] = is_truncated
            if is_terminal:
                post_action_values[row, position] = 0.0
                active = False
            else:
                post_prefix_feature = _context_features(
                    parent,
                    prompt_id,
                    prior,
                    eos_eligible=len(prior) >= MIN_NON_EOS,
                )
                post_action_values[row, position] = float(
                    _critic_value(critic_params, post_prefix_feature)
                )
                active = not is_truncated
        if forced_action_sequences is not None and len(
            forced_action_sequences[row]
        ) != len(actions):
            raise ValueError(
                "forced actions contain entries after EOS or do not reach timeout"
            )
        sequences.append(actions)
    sequence_scores = np.asarray(
        [
            _reward_score(reward_params, parent, pid, actions)
            for pid, actions in zip(prompt_ids, sequences)
        ],
        dtype=np.float32,
    )
    terminal_reward = np.zeros((batch, max_steps), dtype=np.float32)
    for row, length in enumerate(lengths):
        if length:
            terminal_reward[row, length - 1] = sequence_scores[row]
    valid = target_role_mask
    beta = float(tcfg["klCoefficient"])
    rewards = np.where(valid, terminal_reward - beta * sampled_kl, 0.0).astype(
        np.float32
    )
    return {
        "prompt_ids": prompt_ids,
        "actions": token_ids,
        "prefix_token_ids": prefixes,
        "target_role_mask": target_role_mask,
        "input_role_mask": input_role_mask,
        "padding_mask": padding_mask,
        "terminated": terminated,
        "truncated": truncated,
        "old_logps": old_logps,
        "parent_logps": parent_logps,
        "old_restricted_probabilities": old_restricted_probabilities,
        "parent_restricted_probabilities": parent_restricted_probabilities,
        "eligible_action_mask": eligible_action_mask,
        "sampled_kl": sampled_kl,
        "exact_restricted_conditional_kl": exact_conditional_kl,
        "old_values": old_values,
        "post_action_values": post_action_values,
        "parent_context_features": current_features,
        "sequence_scores": sequence_scores,
        "terminal_reward": terminal_reward,
        "rewards": rewards,
        "shaped_rewards": rewards,
        "lengths": lengths,
        "behavior_policy_sha256": _tree_sha256(actor_params),
        "reference_sha256": _tree_sha256(reference_params),
        "reward_head_sha256": _tree_sha256(reward_params),
        "next_rng_key": np.asarray(next_key, dtype=np.uint32).tolist(),
        "sampling_key": np.asarray(key, dtype=np.uint32).tolist(),
        "sampling_mode": (
            "sampled" if forced_action_sequences is None else "forced-contract-probe"
        ),
    }


def generalized_advantage_estimate(records, gamma=0.9, lam=0.8):
    """Return response-action GAE, using timeout bootstrap and stopping both traces."""
    mask = jnp.asarray(records["target_role_mask"], dtype=jnp.float32)
    terminal = jnp.asarray(records["terminated"], dtype=bool)
    stop_trace = terminal | jnp.asarray(records["truncated"], dtype=bool)
    reward = jnp.asarray(records["rewards"], dtype=jnp.float32)
    old_value = jax.lax.stop_gradient(jnp.asarray(records["old_values"], jnp.float32))
    post_value = jax.lax.stop_gradient(
        jnp.asarray(records["post_action_values"], jnp.float32)
    )
    deltas = mask * (reward + gamma * (~terminal) * post_value - old_value)
    advantages = jnp.zeros_like(deltas)
    carry = jnp.zeros((deltas.shape[0],), dtype=jnp.float32)
    for position in range(deltas.shape[1] - 1, -1, -1):
        carry = deltas[:, position] + gamma * lam * (~stop_trace[:, position]) * carry
        carry = jnp.where(mask[:, position] > 0, carry, 0.0)
        advantages = advantages.at[:, position].set(carry)
    targets = jax.lax.stop_gradient(old_value + advantages)
    return {
        "deltas": deltas,
        "advantages": advantages,
        "value_targets": targets,
    }


def ppo_policy_loss(new_logps, old_logps, advantages, mask, clip=0.2):
    """Signed clipped PPO objective averaged over valid response actions."""
    mask = jnp.asarray(mask, dtype=jnp.float32)
    old_logps = jax.lax.stop_gradient(jnp.asarray(old_logps, jnp.float32))
    advantages = jax.lax.stop_gradient(jnp.asarray(advantages, jnp.float32))
    ratios = jnp.exp(jnp.asarray(new_logps, jnp.float32) - old_logps)
    unclipped = ratios * advantages
    clipped = jnp.clip(ratios, 1.0 - clip, 1.0 + clip) * advantages
    denominator = jnp.maximum(jnp.sum(mask), 1.0)
    return -jnp.sum(mask * jnp.minimum(unclipped, clipped)) / denominator


def clipped_value_loss(new_values, old_values, targets, mask, clip=0.2):
    """Clipped squared value loss with frozen old values and valid-action mean."""
    mask = jnp.asarray(mask, dtype=jnp.float32)
    old_values = jax.lax.stop_gradient(jnp.asarray(old_values, jnp.float32))
    targets = jax.lax.stop_gradient(jnp.asarray(targets, jnp.float32))
    clipped_values = old_values + jnp.clip(new_values - old_values, -clip, clip)
    raw_error = jnp.square(new_values - targets)
    clipped_error = jnp.square(clipped_values - targets)
    denominator = jnp.maximum(jnp.sum(mask), 1.0)
    return 0.5 * jnp.sum(mask * jnp.maximum(raw_error, clipped_error)) / denominator


def _new_logps_for_records(parent, actor_params, records):
    batch, steps = np.asarray(records["actions"]).shape
    rows = []
    for row in range(batch):
        values = []
        for pos in range(steps):
            if not records["target_role_mask"][row, pos]:
                values.append(jnp.asarray(0.0, jnp.float32))
                continue
            prefix = records["prefix_token_ids"][row][pos]
            eligible = pos >= MIN_NON_EOS
            logits = _masked_model_logits(
                parent["model_module"], actor_params, prefix, parent["config"]
            )
            dist = _restricted_log_probs(logits, eligible)
            index = ACTION_INDEX[int(records["actions"][row, pos])]
            values.append(dist[index])
        rows.append(jnp.stack(values))
    return jnp.stack(rows)


def _critic_values(critic_params, feature_array):
    return jnp.asarray(feature_array) @ critic_params["weight"] + critic_params["bias"]


def _serializable_rollout(records):
    result = {}
    for key, value in records.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, (jax.Array,)):
            result[key] = np.asarray(value).tolist()
        else:
            result[key] = value
    return result


def train(parent, state, reference_params, reward_params, *, max_updates=None):
    """Run the fixed PPO loop or a prefix of it for checkpoint-continuity checks."""
    tcfg = RECIPE["training"]
    total_updates = int(tcfg["updates"])
    stop_at = (
        total_updates if max_updates is None else min(int(max_updates), total_updates)
    )
    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(tcfg["gradientClipGlobalNorm"]),
        optax.adam(tcfg["actorLearningRate"]),
    )
    critic_optimizer = optax.chain(
        optax.clip_by_global_norm(tcfg["gradientClipGlobalNorm"]),
        optax.adam(tcfg["criticLearningRate"]),
    )
    history = []
    prompts = FIXTURE["trainingPromptIds"]
    while int(state["update_clock"]) < stop_at:
        cursor = int(state["prompt_cursor"])
        batch_ids = [
            prompts[(cursor + i) % len(prompts)] for i in range(int(tcfg["batchSize"]))
        ]
        state["prompt_cursor"] = (cursor + len(batch_ids)) % len(prompts)
        next_cursor = int(state["prompt_cursor"])
        next_prompt_ids = [
            prompts[(next_cursor + i) % len(prompts)]
            for i in range(int(tcfg["batchSize"]))
        ]
        key = jnp.asarray(state["rng_key"], dtype=jnp.uint32)
        next_key, rollout_key = jax.random.split(key)
        state["rng_key"] = np.asarray(next_key, dtype=np.uint32).tolist()
        records = rollout(
            parent,
            state["actor_params"],
            state["critic_params"],
            reference_params,
            reward_params,
            batch_ids,
            rollout_key,
        )
        gae = generalized_advantage_estimate(
            records, float(tcfg["gamma"]), float(tcfg["gaeLambda"])
        )
        advantages = gae["advantages"]
        targets = gae["value_targets"]
        mask = jnp.asarray(records["target_role_mask"], dtype=jnp.float32)
        old_logps = jnp.asarray(records["old_logps"])
        old_values = jnp.asarray(records["old_values"])
        features = jnp.asarray(records["parent_context_features"])
        epoch_metrics = []
        epoch_new_logps = []
        epoch_pre_update_logps = []
        epoch_pre_update_values = []
        for _ in range(int(tcfg["innerEpochs"])):
            epoch_pre_update_logps.append(
                np.asarray(
                    _new_logps_for_records(parent, state["actor_params"], records)
                ).tolist()
            )
            epoch_pre_update_values.append(
                np.asarray(_critic_values(state["critic_params"], features)).tolist()
            )

            def actor_loss_fn(params):
                new_logps = _new_logps_for_records(parent, params, records)
                return ppo_policy_loss(
                    new_logps,
                    old_logps,
                    advantages,
                    mask,
                    float(tcfg["policyClip"]),
                )

            policy_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(
                state["actor_params"]
            )
            actor_updates, state["actor_opt_state"] = actor_optimizer.update(
                actor_grads, state["actor_opt_state"], state["actor_params"]
            )
            state["actor_params"] = optax.apply_updates(
                state["actor_params"], actor_updates
            )

            def critic_loss_fn(params):
                values = _critic_values(params, features)
                return clipped_value_loss(
                    values,
                    old_values,
                    targets,
                    mask,
                    float(tcfg["valueClip"]),
                )

            critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(
                state["critic_params"]
            )
            critic_updates, state["critic_opt_state"] = critic_optimizer.update(
                critic_grads, state["critic_opt_state"], state["critic_params"]
            )
            state["critic_params"] = optax.apply_updates(
                state["critic_params"], critic_updates
            )
            state["inner_epoch_clock"] = int(state["inner_epoch_clock"]) + 1
            new_logps = _new_logps_for_records(parent, state["actor_params"], records)
            epoch_new_logps.append(np.asarray(new_logps).tolist())
            epoch_metrics.append(
                {
                    "policy_loss": float(policy_loss),
                    "critic_loss": float(critic_loss),
                    "valid_action_denominator": int(np.asarray(mask).sum()),
                }
            )
        update_record = _serializable_rollout(records)
        update_record.update(
            {
                "update": int(state["update_clock"]),
                "next_prompt_ids": next_prompt_ids,
                "advantages": np.asarray(advantages).tolist(),
                "value_targets": np.asarray(targets).tolist(),
                "deltas": np.asarray(gae["deltas"]).tolist(),
                "new_logps_per_epoch": epoch_new_logps,
                "pre_update_logps_per_epoch": epoch_pre_update_logps,
                "pre_update_values_per_epoch": epoch_pre_update_values,
                "epoch_metrics": epoch_metrics,
                "policy_loss_by_epoch": [row["policy_loss"] for row in epoch_metrics],
                "critic_loss_by_epoch": [row["critic_loss"] for row in epoch_metrics],
                "exact_restricted_conditional_kl_by_context": update_record[
                    "exact_restricted_conditional_kl"
                ],
            }
        )
        history.append(update_record)
        state["update_clock"] = int(state["update_clock"]) + 1
    return state, history


def evaluate(parent, actor_params, reference_params, critic_params, reward_params):
    """Evaluate matched prompt IDs and keys; report restricted visited-context KL."""
    results = {}
    for split in ("dev", "heldout"):
        prompt_ids = FIXTURE["matchedEvaluationPromptIds"][split]
        seeds = RECIPE["evaluationFloors"]["matchedSamplingSeeds"]
        child_records, parent_records, kl_values = [], [], []
        episode_index = 0
        for seed in seeds:
            for prompt_id in prompt_ids:
                key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), episode_index)
                episode_index += 1
                child = rollout(
                    parent,
                    actor_params,
                    critic_params,
                    reference_params,
                    reward_params,
                    [prompt_id],
                    key,
                )
                baseline = rollout(
                    parent,
                    reference_params,
                    critic_params,
                    reference_params,
                    reward_params,
                    [prompt_id],
                    key,
                )
                child_records.append(child)
                parent_records.append(baseline)
                mask = child["target_role_mask"][0]
                sequence_kl = 0.0
                for position, prefix in enumerate(child["prefix_token_ids"][0]):
                    if not mask[position]:
                        continue
                    eligible = position >= MIN_NON_EOS
                    actor_logits = _masked_model_logits(
                        parent["model_module"], actor_params, prefix, parent["config"]
                    )
                    ref_logits = _masked_model_logits(
                        parent["model_module"],
                        reference_params,
                        prefix,
                        parent["config"],
                    )
                    actor_lp = _restricted_log_probs(actor_logits, eligible)
                    ref_lp = _restricted_log_probs(ref_logits, eligible)
                    actor_p = jnp.where(jnp.isfinite(actor_lp), jnp.exp(actor_lp), 0.0)
                    ref_p = jnp.where(jnp.isfinite(ref_lp), jnp.exp(ref_lp), 0.0)
                    conditional_kl = jnp.sum(
                        jnp.where(actor_p > 0, actor_p * (actor_lp - ref_lp), 0.0)
                    )
                    sequence_kl += float(conditional_kl)
                kl_values.append(sequence_kl)
        child_scores = [float(row["sequence_scores"][0]) for row in child_records]
        parent_scores = [float(row["sequence_scores"][0]) for row in parent_records]
        eval_pairs = [
            row for row in FIXTURE["evaluationPreferencePairs"] if row["split"] == split
        ]
        pair_margins, pair_correct, pair_rows = [], [], []
        for row in eval_pairs:
            chosen = float(
                _reward_score(
                    reward_params, parent, row["promptId"], row["chosenActionSequence"]
                )
            )
            rejected = float(
                _reward_score(
                    reward_params,
                    parent,
                    row["promptId"],
                    row["rejectedActionSequence"],
                )
            )
            margin = (2 * int(row["preferenceLabel"]) - 1) * (chosen - rejected)
            pair_margins.append(margin)
            pair_correct.append(margin > 0)
            pair_rows.append(
                {
                    "split": split,
                    "prompt_id": row["promptId"],
                    "chosen_action_sequence": row["chosenActionSequence"],
                    "rejected_action_sequence": row["rejectedActionSequence"],
                    "preference_label": row["preferenceLabel"],
                    "chosen_reward_score": chosen,
                    "rejected_reward_score": rejected,
                    "signed_margin": margin,
                }
            )
        sample_count = max(sum(int(row["lengths"][0]) for row in child_records), 1)
        sampled_kl_sum = sum(
            float(np.asarray(row["sampled_kl"])[0].sum()) for row in child_records
        )
        results[split] = {
            "matched_prompt_ids": prompt_ids,
            "matched_sampling_seeds": seeds,
            "unique_matched_episode_count": episode_index,
            "mean_child_reward_per_sequence": float(np.mean(child_scores)),
            "mean_parent_reward_per_sequence": float(np.mean(parent_scores)),
            "sequence_reward_denominator": len(child_scores),
            "mean_exact_restricted_conditional_kl_per_valid_action_at_child_visited_contexts": float(
                np.sum(kl_values) / sample_count
            ),
            "mean_sampled_kl_per_valid_action": float(sampled_kl_sum / sample_count),
            "valid_action_denominator": sample_count,
            "exact_restricted_conditional_kl_per_sequence": kl_values,
            "heldout_pairwise_margins": pair_margins,
            "heldout_pairwise_accuracy": float(np.mean(pair_correct)),
            "pairwise_diagnostic_count": len(pair_correct),
            "pairwise_mean_loss_denominator": len(pair_correct),
            "evaluation_pair_rows": pair_rows,
            "child_rollouts": [_serializable_rollout(row) for row in child_records],
            "matched_parent_rollouts": [
                _serializable_rollout(row) for row in parent_records
            ],
        }
    floors = RECIPE["evaluationFloors"]
    heldout = results["heldout"]
    failures = []
    if floors["heldoutMeanRewardMustExceedParent"] and not (
        heldout["mean_child_reward_per_sequence"]
        > heldout["mean_parent_reward_per_sequence"]
    ):
        failures.append("heldout mean child reward did not exceed matched parent")
    if (
        heldout[
            "mean_exact_restricted_conditional_kl_per_valid_action_at_child_visited_contexts"
        ]
        > floors["maximumExactRestrictedCategoricalKlPerValidAction"]
    ):
        failures.append(
            "heldout restricted conditional KL exceeded its predeclared ceiling"
        )
    results["failure_reasons"] = failures
    results["quality_gate_passed"] = not failures
    return results


def save_state(folder: Path, state):
    """Write full child train state as JSON metadata plus pickle-free NPZ leaves."""
    folder.mkdir(parents=True, exist_ok=False)
    arrays, trees = {}, {}
    tree_names = [
        "actor_params",
        "critic_params",
        "actor_opt_state",
        "critic_opt_state",
    ]
    if "reward_params" in state:
        tree_names.append("reward_params")
    for name in tree_names:
        leaves, treedef = jax.tree.flatten(state[name])
        trees[name] = str(treedef)
        for index, leaf in enumerate(leaves):
            arrays[f"{name}_{index:04d}"] = np.asarray(leaf)
    np.savez(folder / "state.npz", **arrays)
    metadata = {
        "schema": 1,
        "tree_definitions": trees,
        "leaf_manifest": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
        "rng_key": state["rng_key"],
        "prompt_cursor": int(state["prompt_cursor"]),
        "update_clock": int(state["update_clock"]),
        "inner_epoch_clock": int(state["inner_epoch_clock"]),
        "parent_params_sha256": state["parent_params_sha256"],
        "parent_package_sha256": state["parent_package_sha256"],
        "processor_sha256": state["processor_sha256"],
        "source_identity": state.get("source_identity", {}),
        "actor_params_sha256": _tree_sha256(state["actor_params"]),
        "critic_params_sha256": _tree_sha256(state["critic_params"]),
        "actor_opt_state_sha256": _tree_sha256(state["actor_opt_state"]),
        "critic_opt_state_sha256": _tree_sha256(state["critic_opt_state"]),
        "reward_params_sha256": (
            _tree_sha256(state["reward_params"]) if "reward_params" in state else None
        ),
        "state_npz_sha256": _sha256((folder / "state.npz").read_bytes()),
    }
    (folder / "state.json").write_text(json.dumps(metadata, sort_keys=True, indent=2))
    return metadata


def restore_state(folder: Path, template_state):
    """Restore every actor, critic, optimizer, key, cursor, and clock leaf."""
    folder = Path(folder)
    metadata = json.loads((folder / "state.json").read_text())
    raw = (folder / "state.npz").read_bytes()
    if metadata.get("schema") != 1 or _sha256(raw) != metadata.get("state_npz_sha256"):
        raise ValueError("child state checksum or schema mismatch")
    if metadata.get("parent_params_sha256") != template_state["parent_params_sha256"]:
        raise ValueError("checkpoint parent parameter identity mismatch")
    if (
        metadata.get("parent_package_sha256") != template_state["parent_package_sha256"]
        or metadata.get("processor_sha256") != template_state["processor_sha256"]
    ):
        raise ValueError("checkpoint parent package or processor identity mismatch")
    if not isinstance(metadata.get("source_identity"), dict):
        raise ValueError("checkpoint source identity is missing")
    for field in (
        "recipe_sha256",
        "fixture_sha256",
        "approval_sha256",
        "candidate_source_sha256",
        "reward_head_sha256",
    ):
        value = metadata["source_identity"].get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise ValueError(f"checkpoint source identity field is invalid: {field}")
    if (
        metadata["source_identity"].get("recipe_sha256")
        != template_state.get("source_identity", {}).get("recipe_sha256")
        or metadata["source_identity"].get("fixture_sha256")
        != template_state.get("source_identity", {}).get("fixture_sha256")
        or metadata["source_identity"].get("approval_sha256")
        != template_state.get("source_identity", {}).get("approval_sha256")
        or metadata["source_identity"].get("candidate_source_sha256")
        != template_state.get("source_identity", {}).get("candidate_source_sha256")
    ):
        raise ValueError("checkpoint source identity differs from the active run")
    update_clock = metadata.get("update_clock")
    inner_epoch_clock = metadata.get("inner_epoch_clock")
    prompt_cursor = metadata.get("prompt_cursor")
    if not isinstance(update_clock, int) or not 0 <= update_clock <= int(
        RECIPE["training"]["updates"]
    ):
        raise ValueError("checkpoint update clock is outside the declared recipe")
    if not isinstance(
        inner_epoch_clock, int
    ) or inner_epoch_clock != update_clock * int(RECIPE["training"]["innerEpochs"]):
        raise ValueError("checkpoint inner-epoch clock does not match the update clock")
    if not isinstance(prompt_cursor, int) or not 0 <= prompt_cursor < len(
        FIXTURE["trainingPromptIds"]
    ):
        raise ValueError(
            "checkpoint prompt cursor is outside the fixed training prompt cycle"
        )
    expected_cursor = (update_clock * int(RECIPE["training"]["batchSize"])) % len(
        FIXTURE["trainingPromptIds"]
    )
    if prompt_cursor != expected_cursor:
        raise ValueError(
            "checkpoint prompt cursor is inconsistent with the fixed update sequence"
        )
    restored = dict(template_state)
    with np.load(folder / "state.npz", allow_pickle=False) as saved:
        tree_names = [
            "actor_params",
            "critic_params",
            "actor_opt_state",
            "critic_opt_state",
        ]
        if "reward_params" in template_state:
            tree_names.append("reward_params")
        expected_leaf_keys = set()
        for name in tree_names:
            leaves, _ = jax.tree.flatten(template_state[name])
            expected_leaf_keys.update(
                f"{name}_{index:04d}" for index in range(len(leaves))
            )
        if (
            set(saved.files) != expected_leaf_keys
            or set(metadata.get("leaf_manifest", {})) != expected_leaf_keys
        ):
            raise ValueError(
                "checkpoint leaf key set differs from the declared state structure"
            )
        expected_tree_defs = {
            name: str(jax.tree.flatten(template_state[name])[1]) for name in tree_names
        }
        if metadata.get("tree_definitions") != expected_tree_defs:
            raise ValueError(
                "checkpoint pytree structure differs from the active template"
            )
        for name in tree_names:
            leaves, treedef = jax.tree.flatten(template_state[name])
            loaded = []
            for index, leaf in enumerate(leaves):
                key = f"{name}_{index:04d}"
                array = saved[key]
                spec = metadata["leaf_manifest"][key]
                if array.shape != np.asarray(leaf).shape or str(array.dtype) != str(
                    np.asarray(leaf).dtype
                ):
                    raise ValueError(f"state leaf {key} has changed shape or dtype")
                if spec != {"shape": list(array.shape), "dtype": str(array.dtype)}:
                    raise ValueError(f"state leaf {key} manifest mismatch")
                if not np.all(np.isfinite(array)):
                    raise ValueError(f"state leaf {key} contains a non-finite value")
                loaded.append(jnp.asarray(array))
            restored[name] = jax.tree.unflatten(treedef, loaded)
    for name in (
        "rng_key",
        "prompt_cursor",
        "update_clock",
        "inner_epoch_clock",
        "parent_params_sha256",
        "parent_package_sha256",
        "processor_sha256",
        "source_identity",
    ):
        restored[name] = metadata[name]
    if _tree_sha256(restored["actor_params"]) != metadata["actor_params_sha256"]:
        raise ValueError("restored child actor hash mismatch")
    for name, digest_field in (
        ("critic_params", "critic_params_sha256"),
        ("actor_opt_state", "actor_opt_state_sha256"),
        ("critic_opt_state", "critic_opt_state_sha256"),
    ):
        if _tree_sha256(restored[name]) != metadata.get(digest_field):
            raise ValueError(f"restored {name} hash mismatch")
    if "reward_params" in restored:
        reward_sha = _tree_sha256(restored["reward_params"])
        if reward_sha != metadata.get("reward_params_sha256") or reward_sha != metadata[
            "source_identity"
        ].get("reward_head_sha256"):
            raise ValueError(
                "restored reward head differs from its frozen source identity"
            )
    return restored
