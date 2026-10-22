"""Independent checks for the bounded synthetic sequence PPO stage."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]


def load(implementation="solution", source_path=None):
    source = (
        Path(source_path)
        if source_path
        else PROJECT / implementation / "sequence_ppo.py"
    )
    os.environ["POSTTRAINING_AUDIT_PROJECT"] = str(PROJECT)
    spec = importlib.util.spec_from_file_location(
        f"sequence_ppo_{implementation}_{source.stem}", source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_mutant(module, replacements):
    source_path = Path(module.__file__)
    source = source_path.read_text()
    for before, after in replacements:
        assert before in source, f"mutation anchor not found: {before[:70]}"
        source = source.replace(before, after, 1)
    mutant = types.ModuleType("sequence_ppo_mutant")
    mutant.__file__ = str(source_path)
    exec(compile(source, str(source_path), "exec"), mutant.__dict__)
    return mutant


def scalar_gae(rewards, values, next_values, active, terminal, truncated, gamma, lam):
    """Small Python oracle, intentionally separate from the JAX reverse scan."""
    time, batch = rewards.shape
    advantages = np.zeros_like(rewards, dtype=np.float64)
    deltas = np.zeros_like(rewards, dtype=np.float64)
    for t in range(time):
        for b in range(batch):
            if not active[t, b]:
                continue
            bootstrap = 0.0 if terminal[t, b] else next_values[t, b]
            deltas[t, b] = rewards[t, b] + gamma * bootstrap - values[t, b]
    for b in range(batch):
        carry = 0.0
        for t in range(time - 1, -1, -1):
            if not active[t, b]:
                carry = 0.0
                continue
            carry = (
                deltas[t, b]
                + gamma * lam * (not (terminal[t, b] or truncated[t, b])) * carry
            )
            advantages[t, b] = carry
    return advantages, deltas


def _finite_difference(fn, point, eps=1e-3):
    point = np.asarray(point, dtype=np.float64)
    grad = np.zeros_like(point)
    for index in np.ndindex(point.shape):
        plus = point.copy()
        minus = point.copy()
        plus[index] += eps
        minus[index] -= eps
        grad[index] = (float(fn(jnp.asarray(plus))) - float(fn(jnp.asarray(minus)))) / (
            2 * eps
        )
    return grad


def check_oracles(module, run_reference_mutations=False):
    gamma, lam = 0.9, 0.8
    # EOS: the last value is ignored because EOS has zero bootstrap.
    rewards = np.asarray([[0.0], [2.0]], dtype=np.float32)
    values = np.asarray([[0.4], [0.6]], dtype=np.float32)
    next_values = np.asarray([[0.6], [0.7]], dtype=np.float32)
    active = np.ones((2, 1), dtype=bool)
    terminal = np.asarray([[False], [True]])
    truncated = np.zeros_like(active)
    records = {
        "rewards": rewards,
        "old_values": values,
        "next_values": next_values,
        "response_mask": active,
        "terminated": terminal,
        "truncated": truncated,
    }
    expected, expected_delta = scalar_gae(
        rewards, values, next_values, active, terminal, truncated, gamma, lam
    )
    np.testing.assert_allclose(expected_delta[:, 0], [0.14, 1.4], atol=1e-7)
    np.testing.assert_allclose(expected[:, 0], [1.148, 1.4], atol=1e-6)
    np.testing.assert_allclose(
        module.generalized_advantage_estimate(records, gamma, lam), expected, atol=1e-6
    )

    # Timeout: bootstrap critic value at the last action, while stopping the lambda trace.
    rewards = np.asarray([[0.0], [0.0], [1.0]], dtype=np.float32)
    values = np.asarray([[0.2], [0.3], [0.5]], dtype=np.float32)
    next_values = np.asarray([[0.3], [0.5], [0.8]], dtype=np.float32)
    active = np.ones((3, 1), dtype=bool)
    terminal = np.zeros_like(active)
    truncated = np.asarray([[False], [False], [True]])
    records.update(
        rewards=rewards,
        old_values=values,
        next_values=next_values,
        response_mask=active,
        terminated=terminal,
        truncated=truncated,
    )
    expected, expected_delta = scalar_gae(
        rewards, values, next_values, active, terminal, truncated, gamma, lam
    )
    np.testing.assert_allclose(expected_delta[:, 0], [0.07, 0.15, 1.22], atol=1e-7)
    np.testing.assert_allclose(expected[:, 0], [0.810448, 1.0284, 1.22], atol=1e-6)
    np.testing.assert_allclose(
        module.generalized_advantage_estimate(records, gamma, lam), expected, atol=1e-6
    )

    # Independent signed PPO calculation. Positive and negative advantages clip on opposite sides.
    ratios = np.asarray([1.4, 0.6])
    advantages = np.asarray([2.0, -2.0])
    clipped = np.clip(ratios, 0.8, 1.2)
    terms = np.minimum(ratios * advantages, clipped * advantages)
    np.testing.assert_allclose(terms, [2.4, -1.6], atol=1e-12)
    expected_loss = -float(np.mean(terms))
    np.testing.assert_allclose(expected_loss, -0.4, atol=1e-12)
    actual = module.ppo_policy_loss(
        jnp.log(jnp.asarray(ratios)),
        jnp.zeros(2),
        jnp.asarray(advantages),
        jnp.ones(2, dtype=bool),
        0.2,
    )
    np.testing.assert_allclose(actual, expected_loss, atol=1e-6)

    # Clipped value loss, independently enumerated token by token.
    new = np.asarray([[1.0, -1.0, 900.0]])
    old = np.asarray([[0.0, 0.0, 0.0]])
    target = np.asarray([[2.0, -2.0, -500.0]])
    mask = np.asarray([[True, True, False]])
    clipped_values = old + np.clip(new - old, -0.2, 0.2)
    expected_value = 0.5 * np.maximum(
        (new - target) ** 2, (clipped_values - target) ** 2
    )
    expected_value = np.sum(np.where(mask, expected_value, 0.0)) / np.sum(mask)
    actual_value = module.clipped_value_loss(new, old, target, mask, 0.2)
    np.testing.assert_allclose(actual_value, expected_value, atol=1e-6)

    # Finite differences check both differentiable objectives, including mask reduction.
    lp = np.asarray([0.1, -0.2, 7.0], dtype=np.float32)
    old_lp = np.asarray([0.0, 0.0, -3.0], dtype=np.float32)
    adv = np.asarray([1.7, -0.8, 100.0], dtype=np.float32)
    mask = np.asarray([True, True, False])
    policy_fn = lambda x: module.ppo_policy_loss(x, old_lp, adv, mask, 0.2)
    autodiff = np.asarray(jax.grad(policy_fn)(jnp.asarray(lp)))
    numeric = _finite_difference(policy_fn, lp)
    np.testing.assert_allclose(autodiff, numeric, atol=2e-3, rtol=2e-3)
    value_fn = lambda x: module.clipped_value_loss(
        x, jnp.zeros(3), jnp.asarray([1.0, -1.0, 50.0]), mask, 0.2
    )
    autodiff_value = np.asarray(jax.grad(value_fn)(jnp.asarray([0.4, -0.3, 8.0])))
    numeric_value = _finite_difference(value_fn, [0.4, -0.3, 8.0])
    np.testing.assert_allclose(autodiff_value, numeric_value, atol=2e-3, rtol=2e-3)

    if run_reference_mutations:
        # Counterfactual mutations must disagree with the independent contracts.
        refreshed_old = np.log(ratios)  # The forbidden mutation makes every ratio one.
        refreshed_terms = np.minimum(
            np.exp(np.log(ratios) - refreshed_old) * advantages,
            np.clip(np.exp(np.log(ratios) - refreshed_old), 0.8, 1.2) * advantages,
        )
        assert not np.allclose(-refreshed_terms.mean(), expected_loss)
        terminal_bootstrap_mutant = scalar_gae(
            np.asarray([[0.0], [2.0]]),
            np.asarray([[0.4], [0.6]]),
            np.asarray([[0.6], [0.7]]),
            np.ones((2, 1), bool),
            np.zeros((2, 1), bool),
            np.zeros((2, 1), bool),
            gamma,
            lam,
        )[0]
        assert not np.allclose(terminal_bootstrap_mutant[:, 0], [1.148, 1.4])
        truncation_no_bootstrap_mutant = scalar_gae(
            np.asarray([[0.0], [0.0], [1.0]]),
            np.asarray([[0.2], [0.3], [0.5]]),
            np.zeros((3, 1)),
            np.ones((3, 1), bool),
            np.zeros((3, 1), bool),
            np.asarray([[False], [False], [True]]),
            gamma,
            lam,
        )[0]
        assert not np.allclose(
            truncation_no_bootstrap_mutant[:, 0], [0.810448, 1.0284, 1.22]
        )
        mutant = load_mutant(
            module,
            [
                (
                    "jnp.minimum(ratio * advantages, clipped_ratio * advantages)",
                    "jnp.maximum(ratio * advantages, clipped_ratio * advantages)",
                )
            ],
        )
        wrong_signed = mutant.ppo_policy_loss(
            jnp.log(jnp.asarray(ratios)),
            jnp.zeros(2),
            jnp.asarray(advantages),
            jnp.ones(2, dtype=bool),
            0.2,
        )
        assert not np.isclose(float(wrong_signed), expected_loss)

        mutant = load_mutant(
            module,
            [
                (
                    "jnp.where(mask, surrogate, 0.0)",
                    "jnp.where(jnp.ones_like(mask), surrogate, 0.0)",
                )
            ],
        )
        padded_loss = mutant.ppo_policy_loss(
            jnp.log(jnp.asarray([1.4, 0.6, 100.0])),
            jnp.zeros(3),
            jnp.asarray([2.0, -2.0, 9.0]),
            jnp.asarray([True, True, False]),
            0.2,
        )
        assert not np.isclose(float(padded_loss), expected_loss)

        terminal_case = {
            "rewards": np.asarray([[0.0], [2.0]], np.float32),
            "old_values": np.asarray([[0.4], [0.6]], np.float32),
            "next_values": np.asarray([[0.6], [0.7]], np.float32),
            "response_mask": np.ones((2, 1), bool),
            "terminated": np.asarray([[False], [True]]),
            "truncated": np.zeros((2, 1), bool),
        }
        mutant = load_mutant(
            module,
            [
                (
                    "gamma * (active & ~terminated) * next_values",
                    "gamma * active * next_values",
                )
            ],
        )
        terminal_bootstrap = np.asarray(
            mutant.generalized_advantage_estimate(terminal_case, gamma, lam)
        )
        assert not np.allclose(terminal_bootstrap[:, 0], [1.148, 1.4])

        timeout_case = {
            "rewards": np.asarray([[0.0], [0.0], [1.0]], np.float32),
            "old_values": np.asarray([[0.2], [0.3], [0.5]], np.float32),
            "next_values": np.asarray([[0.3], [0.5], [0.8]], np.float32),
            "response_mask": np.ones((3, 1), bool),
            "terminated": np.zeros((3, 1), bool),
            "truncated": np.asarray([[False], [False], [True]]),
        }
        mutant = load_mutant(
            module,
            [
                (
                    "gamma * (active & ~terminated) * next_values",
                    "gamma * (active & ~(terminated | truncated)) * next_values",
                )
            ],
        )
        no_timeout_bootstrap = np.asarray(
            mutant.generalized_advantage_estimate(timeout_case, gamma, lam)
        )
        assert not np.allclose(no_timeout_bootstrap[:, 0], [0.810448, 1.0284, 1.22])

        mutant = load_mutant(
            module,
            [
                (
                    "for _epoch in range(epochs):",
                    "for _epoch in range(epochs):\n            old_logps = _selected_log_probs_for_records(policy, records)",
                )
            ],
        )
        refreshed = mutant.train(updates=1, batch_size=8, epochs=2)
        assert (
            refreshed["updates"][0]["old_logps_sha256_before_epochs"]
            != refreshed["updates"][0]["old_logps_sha256_after_epochs"]
        )


def check_rollout_and_training(module):
    cfg = module.RECIPE["rollout"]
    reward_weights, fit = module.fit_reward_model()
    assert fit["final_pairwise_loss"] < fit["initial_pairwise_loss"]
    assert min(fit["margins"]) > 0
    policy, critic, reference = module.initialize()
    records = module.rollout(
        policy,
        critic,
        reference,
        reward_weights,
        cfg["trainingPromptIds"],
        jax.random.key(cfg["seed"]),
    )
    response = np.asarray(records["response_mask"])
    target = np.asarray(records["target_role_mask"])
    input_role = np.asarray(records["input_role_mask"])
    padding = np.asarray(records["padding_mask"])
    actions = np.asarray(records["actions"])
    np.testing.assert_array_equal(response, target)
    np.testing.assert_array_equal(padding, ~response)
    assert not np.any(
        input_role[0]
    )  # BOS/prompt context predicts the first response target.
    assert np.all(np.asarray(records["lengths"]) >= cfg["minimumResponseTokens"])
    assert np.all(
        np.sum(np.asarray(records["terminated"]), axis=0)
        + np.sum(np.asarray(records["truncated"]), axis=0)
        == 1
    )
    assert np.all(actions[~response] == module.PAD)
    previous = np.asarray(records["previous_tokens"])
    for t in range(1, actions.shape[0]):
        live = response[t]
        np.testing.assert_array_equal(previous[t, live], actions[t - 1, live])
        np.testing.assert_array_equal(
            input_role[t], response[t - 1] & ~np.asarray(records["terminated"])[t - 1]
        )
    # Using input roles as the target loss mask is an off-by-one mutation: it drops action 0.
    mask_for_loss = np.asarray([[True, True, True]])
    input_role_mutant = np.asarray([[False, True, True]])
    logps = jnp.log(jnp.asarray([1.4, 0.6, 1.1]))
    old_logps = jnp.zeros(3)
    adv = jnp.asarray([2.0, -2.0, 1.0])
    correct_loss = module.ppo_policy_loss(logps, old_logps, adv, mask_for_loss, 0.2)
    shifted_loss = module.ppo_policy_loss(logps, old_logps, adv, input_role_mutant, 0.2)
    assert not np.isclose(float(correct_loss), float(shifted_loss))
    assert np.all(np.asarray(records["rewards"])[~response] == 0)
    assert np.all(np.asarray(records["sampled_kl"])[~response] == 0)
    advantages = np.asarray(
        module.generalized_advantage_estimate(records, cfg["gamma"], cfg["gaeLambda"])
    )
    assert np.all(advantages[~response] == 0)
    assert np.all(np.asarray(records["old_values"])[~response] == 0)
    for b in range(actions.shape[1]):
        length = int(np.asarray(records["lengths"])[b])
        eos_positions = np.flatnonzero(actions[:length, b] == module.EOS)
        assert len(eos_positions) <= 1
        assert len(eos_positions) == 0 or eos_positions[0] >= 1
        assert len(eos_positions) == 0 or eos_positions[0] == length - 1
        assert np.all(response[:length, b]) and not np.any(response[length:, b])
        assert bool(np.asarray(records["terminated"])[length - 1, b]) == bool(
            len(eos_positions)
        )
        assert bool(np.asarray(records["truncated"])[length - 1, b]) == (
            not bool(len(eos_positions))
        )
        assert np.all(
            np.asarray(records["reward_model_terminal"])[
                np.arange(actions.shape[0]) != length - 1, b
            ]
            == 0
        )

    # Finite-difference one actual policy-table and critic-table coordinate.
    advantages_jax = module.generalized_advantage_estimate(
        records, cfg["gamma"], cfg["gaeLambda"]
    )
    actor_fn = lambda candidate: module.ppo_policy_loss(
        module._selected_log_probs_for_records(candidate, records),
        records["old_logps"],
        advantages_jax,
        records["response_mask"],
        cfg["policyClip"],
    )
    actor_grad = jax.grad(actor_fn)(policy)
    actor_coord = np.unravel_index(
        np.argmax(np.abs(np.asarray(actor_grad))), actor_grad.shape
    )
    actor_eps = 1e-3
    actor_plus = policy.at[actor_coord].add(actor_eps)
    actor_minus = policy.at[actor_coord].add(-actor_eps)
    actor_numeric = (float(actor_fn(actor_plus)) - float(actor_fn(actor_minus))) / (
        2 * actor_eps
    )
    np.testing.assert_allclose(
        float(actor_grad[actor_coord]), actor_numeric, atol=3e-3, rtol=3e-3
    )

    cont_tasks, cont_previous, cont_targets = module.continuation_value_targets(
        reward_weights
    )
    critic_fn = lambda candidate: module.clipped_value_loss(
        module._current_values(candidate, records),
        records["old_values"],
        records["old_values"] + advantages_jax,
        records["response_mask"],
        cfg["valueClip"],
    ) + 0.5 * jnp.mean(
        jnp.square(
            candidate[cont_tasks, cont_previous, module.MAX_TOKENS] - cont_targets
        )
    )
    critic_grad = jax.grad(critic_fn)(critic)
    critic_coord = np.unravel_index(
        np.argmax(np.abs(np.asarray(critic_grad))), critic_grad.shape
    )
    critic_plus = critic.at[critic_coord].add(actor_eps)
    critic_minus = critic.at[critic_coord].add(-actor_eps)
    critic_numeric = (
        float(critic_fn(critic_plus)) - float(critic_fn(critic_minus))
    ) / (2 * actor_eps)
    np.testing.assert_allclose(
        float(critic_grad[critic_coord]), critic_numeric, atol=3e-3, rtol=3e-3
    )

    result = module.train()
    assert result["evaluation"]["status"] in ("passed", "failed")
    assert "failure_reasons" in result["evaluation"]
    assert result["evaluation"]["valid_response_tokens"] > 0
    assert (
        result["evaluation"]["token_denominator"]
        == result["evaluation"]["valid_response_tokens"]
    )
    assert result["evaluation"]["sequence_denominator"] == len(cfg["heldoutPromptIds"])
    assert np.isfinite(result["evaluation"]["exact_kl_per_valid_token"])
    assert result["frozen_reference_sha256"] == module._tree_sha256(result["reference"])
    assert result["frozen_reward_sha256"] == module._array_sha256(
        result["reward_weights"]
    )
    assert np.any(np.abs(np.asarray(result["continuation_values"])) > 1e-6)
    assert np.any(np.abs(np.asarray(result["continuation_value_targets"])) > 1e-6)
    trained_records = module.rollout(
        result["policy"],
        result["critic"],
        result["reference"],
        result["reward_weights"],
        cfg["trainingPromptIds"],
        jax.random.key(cfg["seed"] + 71),
    )
    trunc_positions = np.argwhere(np.asarray(trained_records["truncated"]))
    assert len(trunc_positions) > 0, "fixed trained probe must include a timeout"
    t, b = trunc_positions[0]
    task = int(np.asarray(trained_records["task_ids"])[b])
    action = int(np.asarray(trained_records["actions"])[t, b])
    bootstrap = float(np.asarray(trained_records["next_values"])[t, b])
    expected_bootstrap = float(
        np.asarray(result["critic"])[task, action, module.MAX_TOKENS]
    )
    assert abs(bootstrap) > 1e-6
    np.testing.assert_allclose(bootstrap, expected_bootstrap, atol=1e-6)
    probe_advantage = module.generalized_advantage_estimate(
        trained_records, cfg["gamma"], cfg["gaeLambda"]
    )
    assert np.isfinite(float(probe_advantage[t, b]))
    for update in result["updates"]:
        assert update["frozen_reference_sha256"] == result["frozen_reference_sha256"]
        assert update["frozen_reward_sha256"] == result["frozen_reward_sha256"]
        assert (
            update["old_logps_sha256_before_epochs"]
            == update["old_logps_sha256_after_epochs"]
        )
        assert (
            update["old_values_sha256_before_epochs"]
            == update["old_values_sha256_after_epochs"]
        )
        before = np.asarray(update["rollout"]["old_logps"])
        after = np.asarray(update["inner_epoch_selected_logps"][-1])
        mask = np.asarray(update["rollout"]["response_mask"])
        assert np.any(np.abs(after[mask] - before[mask]) > 1e-7)
        assert len(update["inner_epoch_selected_logps"]) == cfg["innerEpochs"]
        for key in (
            "actions",
            "old_logps",
            "reference_logps",
            "response_mask",
            "target_role_mask",
            "input_role_mask",
            "padding_mask",
            "terminated",
            "truncated",
            "rewards",
            "old_values",
            "advantages",
            "value_targets",
            "sampled_kl",
            "reward_model_terminal",
        ):
            assert key in update["rollout"], f"missing retained field {key}"
        assert np.all(np.asarray(update["rollout"]["value_targets"])[~mask] == 0)

    # Inject an empty/nonfinite eval to ensure train() preserves the evaluator's failure.
    original_evaluate = module.evaluate
    module.evaluate = lambda *args, **kwargs: {
        "sequence_reward_mean": float("nan"),
        "exact_kl_per_valid_token": float("nan"),
        "failure_reasons": ["nonfinite_or_empty_heldout_evaluation"],
    }
    try:
        failed = module.train(updates=1, batch_size=8, epochs=2)
        assert (
            "nonfinite_or_empty_heldout_evaluation"
            in failed["evaluation"]["failure_reasons"]
        )
        assert failed["evaluation"]["status"] == "failed"
    finally:
        module.evaluate = original_evaluate

    # The evaluator itself reports both empty masks and nonfinite scores.
    original_rollout = module.rollout

    def broken_rollout(
        policy, critic, reference, reward_weights, prompt_ids, key, *, beta=0.03
    ):
        count = len(prompt_ids)
        return {
            "actions": jnp.zeros((1, count), dtype=jnp.int32),
            "task_ids": jnp.asarray([0] * count, dtype=jnp.int32),
            "previous_tokens": jnp.ones((1, count), dtype=jnp.int32),
            "response_mask": jnp.zeros((1, count), dtype=bool),
            "sequence_scores": jnp.full((count,), jnp.nan),
            "sampled_kl": jnp.zeros((1, count)),
        }

    module.rollout = broken_rollout
    try:
        bad_eval = module.evaluate(
            policy,
            reference,
            reward_weights,
            prompt_ids=cfg["heldoutPromptIds"],
            seed=cfg["seed"],
        )
        assert "nonfinite_or_empty_heldout_evaluation" in bad_eval["failure_reasons"]
    finally:
        module.rollout = original_rollout

    # Mutating either frozen component changes its digest and must be detected.
    changed_ref = np.asarray(result["reference"]).copy()
    changed_ref.flat[0] += 0.01
    assert (
        module._tree_sha256(jnp.asarray(changed_ref))
        != result["frozen_reference_sha256"]
    )
    try:
        module.verify_frozen_components(
            changed_ref,
            result["reward_weights"],
            result["frozen_reference_sha256"],
            result["frozen_reward_sha256"],
        )
        raise AssertionError("changed reference snapshot was accepted")
    except ValueError:
        pass
    changed_reward = np.asarray(result["reward_weights"]).copy()
    changed_reward.flat[0] += 0.01
    assert module._array_sha256(changed_reward) != result["frozen_reward_sha256"]
    try:
        module.verify_frozen_components(
            result["reference"],
            changed_reward,
            result["frozen_reference_sha256"],
            result["frozen_reward_sha256"],
        )
        raise AssertionError("changed reward snapshot was accepted")
    except ValueError:
        pass

    with tempfile.TemporaryDirectory(prefix="sequence-ppo-reload-") as td:
        state_path = Path(td) / "state.npz"
        module.save_state(state_path, result)
        loaded = module.load_state(state_path)
        direct = module.evaluate(
            loaded["policy"],
            loaded["reference"],
            loaded["reward_weights"],
            prompt_ids=cfg["heldoutPromptIds"],
            seed=cfg["seed"] + 100_000,
            beta=cfg["klCoefficient"],
        )
        probe = (
            "import importlib.util,json,sys,hashlib,os; from pathlib import Path; "
            f"os.environ['POSTTRAINING_AUDIT_PROJECT']={str(PROJECT)!r}; "
            f"p={str(module.__file__)!r}; "
            "s=importlib.util.spec_from_file_location('seqfresh',p); m=importlib.util.module_from_spec(s); "
            "s.loader.exec_module(m); x=m.load_state(sys.argv[1]); c=m.RECIPE['rollout']; "
            "r=m.evaluate(x['policy'],x['reference'],x['reward_weights'],prompt_ids=c['heldoutPromptIds'], "
            "seed=c['seed']+100000,beta=c['klCoefficient']); "
            "print(json.dumps({'evaluation':r,'source_sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()},sort_keys=True))"
        )
        fresh = json.loads(
            subprocess.check_output(
                [sys.executable, "-c", probe, str(state_path)], text=True
            )
        )
        assert fresh == {
            "evaluation": direct,
            "source_sha256": hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest(),
        }
    return result


def run(implementation, stage, evidence_dir=None, candidate_path=None):
    if stage == "all":
        if implementation != "solution" and candidate_path is None:
            raise ValueError(
                "--stage all needs the reference implementation or an explicit --candidate path"
            )
        subprocess.run(
            [
                sys.executable,
                str(PROJECT / "tests/check.py"),
                "--implementation",
                "solution",
                "--stage",
                "4",
            ],
            check=True,
        )
    if candidate_path is not None or implementation == "solution":
        reference = load("solution")
        check_oracles(reference, run_reference_mutations=True)
    module = load(implementation, candidate_path)
    check_oracles(module)
    result = check_rollout_and_training(module)
    if evidence_dir:
        evidence_path = Path(evidence_dir) / "sequence-ppo-evidence.json"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "candidate_path": str(Path(module.__file__).resolve()),
            "candidate_sha256": hashlib.sha256(
                Path(module.__file__).read_bytes()
            ).hexdigest(),
            "recipe_sha256": result["recipe_sha256"],
            "seed": result["seed"],
            "reward_fit": result["reward_fit"],
            "frozen_reference_sha256": result["frozen_reference_sha256"],
            "frozen_reward_sha256": result["frozen_reward_sha256"],
            "updates": result["updates"],
            "reference_evaluation": result["reference_evaluation"],
            "evaluation": result["evaluation"],
        }
        evidence_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        reread = json.loads(evidence_path.read_text())
        assert reread == report
    print(
        f"PASS posttraining-audit stage {stage}: sequence PPO masks, GAE, clipping, frozen rollouts, reload and heldout evidence"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation", choices=("starter", "solution"), default="starter"
    )
    parser.add_argument("--stage", choices=("5", "all"), default="5")
    parser.add_argument("--evidence-dir")
    parser.add_argument(
        "--candidate",
        help="Path to a learner implementation of the sequence PPO module",
    )
    args = parser.parse_args()
    run(args.implementation, args.stage, args.evidence_dir, args.candidate)
