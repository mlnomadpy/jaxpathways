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
    """TODO: Fit the fixed synthetic pairwise reward head."""
    raise NotImplementedError(
        "Implement fit_pairwise_reward_head for the optional saved-decoder bridge; see README.md"
    )


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
    """TODO: Collect restricted-action decoder rollouts."""
    raise NotImplementedError(
        "Implement rollout for the optional saved-decoder bridge; see README.md"
    )


def generalized_advantage_estimate(records, gamma=0.9, lam=0.8):
    """TODO: Compute masked terminal and timeout GAE."""
    raise NotImplementedError(
        "Implement generalized_advantage_estimate for the optional saved-decoder bridge; see README.md"
    )


def ppo_policy_loss(new_logps, old_logps, advantages, mask, clip=0.2):
    """TODO: Compute the signed clipped PPO policy objective."""
    raise NotImplementedError(
        "Implement ppo_policy_loss for the optional saved-decoder bridge; see README.md"
    )


def clipped_value_loss(new_values, old_values, targets, mask, clip=0.2):
    """TODO: Compute the separately clipped critic objective."""
    raise NotImplementedError(
        "Implement clipped_value_loss for the optional saved-decoder bridge; see README.md"
    )


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
    """TODO: Run the bounded sequence-PPO update loop."""
    raise NotImplementedError(
        "Implement train for the optional saved-decoder bridge; see README.md"
    )


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
