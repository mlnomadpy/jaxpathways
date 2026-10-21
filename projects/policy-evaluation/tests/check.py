"""Independent checks for a deliberately small, inspectable RL system."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--implementation", default="starter")
p.add_argument("--stage", choices=["all", "1", "2", "3", "4", "5"], default="all")
p.add_argument("--report", type=Path)
a = p.parse_args()
stage = 5 if a.stage == "all" else int(a.stage)
path = (
    ROOT / a.implementation / "model.py"
    if a.implementation in ("starter", "solution")
    else Path(a.implementation)
)
spec = importlib.util.spec_from_file_location("learner", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def enumerate_return(theta, horizon):
    probs = 1 / (1 + np.exp(-np.asarray(theta, dtype=float)))

    def visit(position, remaining):
        if position == 3 or remaining == 0:
            return 0.0
        total = 0.0
        for action in (0, 1):
            nxt = min(3, max(0, position + (1 if action else -1)))
            prob = probs[position] if action else 1 - probs[position]
            total += prob * ((1.0 if nxt == 3 else -0.01) + visit(nxt, remaining - 1))
        return total

    return (visit(0, horizon) + visit(1, horizon)) / 2


for horizon in (1, 3, 8):
    for position in range(4):
        for elapsed in range(horizon):
            for action in (0, 1):
                s = m.State(
                    jnp.int32(position), jnp.int32(elapsed), jnp.bool_(position == 3)
                )
                nxt, reward, term, trunc = jax.jit(lambda s, a: m.step(s, a, horizon))(
                    s, action
                )
                if position == 3:
                    assert (
                        int(nxt.position) == 3
                        and float(reward) == 0
                        and not term
                        and not trunc
                    )
                else:
                    target = min(3, max(0, position + (1 if action else -1)))
                    assert (
                        int(nxt.position) == target and int(nxt.elapsed) == elapsed + 1
                    )
                    assert bool(term) == (target == 3)
                    assert bool(trunc) == (target != 3 and elapsed + 1 >= horizon)
                    assert bool(nxt.done) == bool(term | trunc)
                    np.testing.assert_allclose(reward, 1.0 if target == 3 else -0.01)
keys = jax.random.split(jax.random.key(9), 1000)
starts = jax.vmap(m.reset)(keys)
assert set(np.asarray(starts.position).tolist()) == {0, 1}
assert 0.4 < float(starts.position.mean()) < 0.6
assert not bool(starts.done.any()) and not bool(starts.elapsed.any())
print(
    "PASS stage 1: enumerated transitions, boundary priority, absorbing state, randomized reset"
)

if stage >= 2:
    for horizon, batch in [(1, 7), (5, 19), (8, 37)]:
        theta = jnp.array([-0.7, 0.3, 1.1])
        final, r = m.rollout(theta, jax.random.key(81), batch, horizon)
        assert r["reward"].shape == (horizon, batch) and bool(final.done.all())
        assert np.all(np.sum(np.asarray(r["terminated"] | r["truncated"]), axis=0) == 1)
        np.testing.assert_array_equal(r["reward"][~r["active"]], 0)
        # Reconstruct every trajectory using Python, independent of the JAX step.
        for b in range(batch):
            position = int(r["observation"][0, b])
            done = False
            for t in range(horizon):
                assert int(r["observation"][t, b]) == position
                assert bool(r["active"][t, b]) == (not done)
                expected = 0.0
                if not done:
                    position = min(
                        3, max(0, position + (1 if r["action"][t, b] else -1))
                    )
                    expected = 1.0 if position == 3 else -0.01
                    done = position == 3 or t == horizon - 1
                np.testing.assert_allclose(r["reward"][t, b], expected)
        replay = m.rollout(theta, jax.random.key(81), batch, horizon)[1]
        for name in r:
            np.testing.assert_array_equal(r[name], replay[name])
        other = m.rollout(theta, jax.random.key(82), batch, horizon)[1]
        assert not np.array_equal(r["action"], other["action"])
    rewards = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 0.0]])
    np.testing.assert_array_equal(m.returns_to_go(rewards), [[9, 6], [8, 4], [5, 0]])
    print(
        "PASS stage 2: trajectory reconstruction, padding mask, event count, changed seed, return targets"
    )

if stage >= 3:
    for theta in [jnp.array([-0.5, 0.2, 1.0]), jnp.array([0.7, -0.4, 0.8])]:
        for horizon in [3, 5, 8]:
            np.testing.assert_allclose(
                m.exact_return(theta, horizon),
                enumerate_return(theta, horizon),
                atol=2e-7,
            )
        independent = []
        eps = 1e-4
        for i in range(3):
            d = np.eye(3)[i] * eps
            independent.append(
                (
                    enumerate_return(np.asarray(theta) + d, 8)
                    - enumerate_return(np.asarray(theta) - d, 8)
                )
                / (2 * eps)
            )
        np.testing.assert_allclose(
            jax.grad(m.exact_return)(theta), independent, rtol=2e-4, atol=2e-5
        )
    # Deliberate mixed-sign advantages and stale behavior probabilities.
    r = dict(
        observation=jnp.array([[0, 1], [1, 2]]),
        action=jnp.array([[1, 0], [0, 1]]),
        reward=jnp.array([[0.4, -0.2], [-0.1, 0.3]]),
        active=jnp.array([[1, 1], [1, 0]], bool),
        old_logp=jnp.log(jnp.array([[0.2, 0.8], [0.7, 0.4]])),
    )
    theta = jnp.array([0.6, -0.5, 0.9])
    newp = 1 / (1 + np.exp(-np.asarray(theta)[np.asarray(r["observation"])]))
    newp = np.where(np.asarray(r["action"]) == 1, newp, 1 - newp)
    adv = np.array([[0.3, 0.1], [-0.1, 0.3]])
    ratios = newp / np.exp(np.asarray(r["old_logp"]))
    expected = (
        -np.sum(
            np.minimum(ratios * adv, np.clip(ratios, 0.8, 1.2) * adv)
            * np.asarray(r["active"])
        )
        / 2
    )
    np.testing.assert_allclose(m.policy_loss(theta, r), expected, rtol=1e-6)
    logp_grad = jax.grad(lambda old: m.policy_loss(theta, dict(r, old_logp=old)))(
        r["old_logp"]
    )
    reward_grad = jax.grad(lambda rew: m.policy_loss(theta, dict(r, reward=rew)))(
        r["reward"]
    )
    np.testing.assert_array_equal(logp_grad, 0)
    np.testing.assert_array_equal(reward_grad, 0)
    # Fresh Monte Carlo policy gradient agrees with differentiating an enumerated expectation.
    theta = jnp.array([-0.2, 0.3, 0.7])
    r = m.rollout(theta, jax.random.key(902), 32768, 8)[1]
    estimate = -jax.grad(m.policy_loss)(theta, r, "reinforce")
    np.testing.assert_allclose(
        estimate, jax.grad(m.exact_return)(theta), atol=0.012, rtol=0.04
    )
    print(
        "PASS stage 3: independent enumeration and finite differences, frozen targets, PPO signs, Monte Carlo gradient"
    )

results = []
if stage >= 4:
    for method, epochs in [("reinforce", 1), ("ppo", 3)]:
        for seed in (0, 7, 23):
            theta, history = m.train(seed=seed, method=method, epochs=epochs)
            replay, rh = m.train(seed=seed, method=method, epochs=epochs)
            np.testing.assert_array_equal(theta, replay)
            np.testing.assert_array_equal(history, rh)
            assert history.shape == (41,) and np.isfinite(history).all()
            assert float(history[-1]) > 0.90 and float(history[-1] - history[0]) > 0.35
            before = np.array(theta, copy=True)
            evaluation = m.evaluate(theta, [1000, 1001, 1002, 1003])
            np.testing.assert_array_equal(theta, before)
            assert abs(evaluation["mean"] - enumerate_return(theta, 8)) < 0.03
            results.append(
                dict(
                    method=method,
                    trainingSeed=seed,
                    exact=float(history[-1]),
                    heldOut=evaluation["mean"],
                )
            )
    for bad in [jnp.zeros(2), jnp.array([0.0, jnp.nan, 0.0])]:
        try:
            m.evaluate(bad, [1, 2])
        except ValueError:
            pass
        else:
            raise AssertionError("invalid policy accepted")
    for seeds in [[1], [1, 1]]:
        try:
            m.evaluate(jnp.zeros(3), seeds)
        except ValueError:
            pass
        else:
            raise AssertionError("evaluation seed contract ignored")
    print(
        "PASS stage 4: six trained agents, repeatability, held-out streams, state isolation, input rejection"
    )
    print(json.dumps(results, indent=2))

if stage >= 5:
    # Independent scalar oracle for multi-step GAE. The second transition is
    # an external time-limit cutoff: its stored next observation is position 3,
    # while the following row is a reset observation at position 0.
    value_table = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    records = dict(
        observation=jnp.array([[0], [1], [0]], dtype=jnp.int32),
        next_observation=jnp.array([[1], [3], [1]], dtype=jnp.int32),
        reward=jnp.array([[1.0], [2.0], [3.0]]),
        active=jnp.array([[1], [1], [1]], dtype=bool),
        terminated=jnp.array([[0], [0], [1]], dtype=bool),
        truncated=jnp.array([[0], [1], [0]], dtype=bool),
    )
    # t=2: 3 + 0 - V(0) = 2.5; terminal bootstrap is suppressed.
    # t=1: 2 + .5*V(final position 3) - V(1) = 3; cutoff stops the trace.
    # t=0: 1 + .5*V(1) - V(0) + .5*.5*3 = 1.75.
    got = m.generalized_advantage_estimate(
        records, jnp.asarray(value_table), 0.5, 0.5, True
    )
    np.testing.assert_allclose(got[:, 0], [1.75, 3.0, 2.5], atol=1e-6)
    compiled_gae = jax.jit(m.generalized_advantage_estimate)(
        records, jnp.asarray(value_table), 0.5, 0.5, jnp.bool_(True)
    )
    np.testing.assert_allclose(compiled_gae, got, atol=1e-6)
    # Finite-horizon callers opt out of timeout bootstrap, while still stopping
    # the recursion at that same boundary.
    no_cutoff_bootstrap = m.generalized_advantage_estimate(
        records, jnp.asarray(value_table), 0.5, 0.5, False
    )
    np.testing.assert_allclose(no_cutoff_bootstrap[:, 0], [1.25, 1.0, 2.5], atol=1e-6)
    np.testing.assert_allclose(
        m.generalized_advantage_estimate(records, jnp.asarray(value_table), 0.5, 0.5),
        no_cutoff_bootstrap,
        atol=1e-6,
    )
    later_episode_changes = dict(records, reward=records["reward"].at[2, 0].set(300.0))
    changed_later_adv = m.generalized_advantage_estimate(
        later_episode_changes, jnp.asarray(value_table), 0.5, 0.5, True
    )
    # A changed reward in the post-reset episode cannot leak through the cutoff.
    np.testing.assert_allclose(changed_later_adv[:2], got[:2], atol=1e-6)

    # A pure three-transition terminal episode checks reverse accumulation
    # independently of any boundary/reset behavior.
    terminal_records = dict(
        observation=jnp.array([[0], [1], [2]], dtype=jnp.int32),
        next_observation=jnp.array([[1], [2], [3]], dtype=jnp.int32),
        reward=jnp.array([[1.0], [2.0], [3.0]]),
        active=jnp.ones((3, 1), dtype=bool),
        terminated=jnp.array([[0], [0], [1]], dtype=bool),
        truncated=jnp.zeros((3, 1), dtype=bool),
    )
    terminal_adv = m.generalized_advantage_estimate(
        terminal_records, jnp.asarray(value_table), 0.5, 0.5, True
    )
    np.testing.assert_allclose(terminal_adv[:, 0], [1.5625, 2.25, 1.0], atol=1e-6)

    # Padding is inert, including after a true terminal. Empty scans are valid
    # and return empty arrays; fitting an all-padding batch is rejected.
    padded = dict(
        observation=jnp.array([[0, 1], [1, 2], [2, 3]], dtype=jnp.int32),
        next_observation=jnp.array([[1, 2], [2, 3], [3, 3]], dtype=jnp.int32),
        reward=jnp.array([[1.0, 0.2], [0.5, 0.3], [0.0, 0.4]]),
        active=jnp.array([[1, 1], [1, 1], [0, 1]], dtype=bool),
        terminated=jnp.array([[0, 0], [1, 0], [0, 1]], dtype=bool),
        truncated=jnp.zeros((3, 2), dtype=bool),
    )
    padded_adv = m.generalized_advantage_estimate(padded, jnp.asarray(value_table))
    np.testing.assert_array_equal(padded_adv[2, 0], 0.0)
    padded_targets = jnp.zeros((3, 2))
    masked_loss = m.value_loss(jnp.zeros(4), padded, padded_targets)
    changed_padding = padded_targets.at[2, 0].set(1e6)
    np.testing.assert_allclose(
        m.value_loss(jnp.zeros(4), padded, changed_padding), masked_loss
    )
    nonfinite_records = dict(padded, active=padded["active"].at[1, 1].set(False))
    nonfinite_padding = padded_targets.at[2, 0].set(jnp.nan).at[1, 1].set(jnp.inf)
    baseline_table = jnp.array([0.2, -0.4, 0.7, 1.1])
    baseline_grad = jax.grad(lambda table: m.value_loss(table, padded, padded_targets))(
        baseline_table
    )
    nonfinite_loss = m.value_loss(baseline_table, nonfinite_records, nonfinite_padding)
    nonfinite_grad = jax.grad(
        lambda table: m.value_loss(table, nonfinite_records, nonfinite_padding)
    )(baseline_table)
    finite_baseline_loss = m.value_loss(
        baseline_table, nonfinite_records, padded_targets
    )
    finite_baseline_grad = jax.grad(
        lambda table: m.value_loss(table, nonfinite_records, padded_targets)
    )(baseline_table)
    np.testing.assert_allclose(nonfinite_loss, finite_baseline_loss)
    np.testing.assert_allclose(nonfinite_grad, finite_baseline_grad, equal_nan=False)
    target_grad = jax.grad(lambda target: m.value_loss(baseline_table, padded, target))(
        padded_targets
    )
    np.testing.assert_array_equal(target_grad, np.zeros((3, 2)))
    empty = {name: value[:0] for name, value in padded.items()}
    np.testing.assert_array_equal(
        m.generalized_advantage_estimate(empty, jnp.asarray(value_table)).shape, (0, 2)
    )
    np.testing.assert_allclose(
        m.value_loss(jnp.zeros(4), empty, jnp.zeros((0, 2))), 0.0
    )
    try:
        m.fit_value_function(empty, jnp.zeros((0, 2)))
    except ValueError:
        pass
    else:
        raise AssertionError("all-padding critic batch accepted")

    # Fit an actual tabular value function to returns from retained stochastic
    # trajectories. Verify optimization reduces prediction error, then change
    # the collection key and verify both the data and fitted values respond.
    theta = jnp.array([-0.7, 0.3, 1.1])
    rollout_records = m.rollout(theta, jax.random.key(313), 512, 8)[1]
    targets = m.returns_to_go(rollout_records["reward"])
    fitted, history = m.fit_value_function(
        rollout_records, targets, updates=250, learning_rate=0.1
    )
    assert history.shape == (251,) and np.isfinite(np.asarray(history)).all()
    # Independent NumPy oracle: every observed state value at the least-squares
    # optimum is that state's active target sample mean. Also recompute the
    # returned table's masked MSE rather than trusting its history.
    obs_np = np.asarray(rollout_records["observation"])
    active_np = np.asarray(rollout_records["active"], dtype=bool)
    targets_np = np.asarray(targets)
    fitted_np = np.asarray(fitted)
    counts = np.bincount(obs_np[active_np], minlength=4)
    sums = np.bincount(obs_np[active_np], weights=targets_np[active_np], minlength=4)
    sample_means = sums / np.maximum(counts, 1)
    assert np.all(counts[:3] > 0)
    np.testing.assert_allclose(
        fitted_np[counts > 0], sample_means[counts > 0], atol=5e-4, rtol=5e-4
    )
    # Position 3 is terminal and never appears as an active action observation.
    assert counts[3] == 0 and fitted_np[3] == 0.0
    independent_mse = np.square(
        fitted_np[obs_np[active_np]] - targets_np[active_np]
    ).mean()
    np.testing.assert_allclose(
        float(history[-1]), independent_mse, atol=2e-7, rtol=2e-6
    )
    # Verify the objective gradient itself against a NumPy one-hot oracle.
    probe = jnp.array([0.2, -0.3, 0.8, 0.4])
    pred_probe = np.asarray(probe)[obs_np]
    residual = np.where(active_np, pred_probe - targets_np, 0.0)
    grad_oracle = (
        2
        * np.bincount(obs_np[active_np], weights=residual[active_np], minlength=4)
        / active_np.sum()
    )
    grad_got = np.asarray(
        jax.grad(lambda table: m.value_loss(table, rollout_records, targets))(probe)
    )
    np.testing.assert_allclose(grad_got, grad_oracle, atol=2e-6, rtol=1e-5)
    # Targets are a fixed sampled regression target, not a path for gradients.
    np.testing.assert_array_equal(
        jax.grad(lambda target: m.value_loss(fitted, rollout_records, target))(targets),
        np.zeros_like(np.asarray(targets)),
    )
    # The four-state table cannot predict a stochastic return exactly; require
    # a large reduction while respecting the irreducible outcome variance.
    assert float(history[-1]) < float(history[0]) * 0.7
    changed_records = m.rollout(theta, jax.random.key(314), 512, 8)[1]
    assert not np.array_equal(rollout_records["action"], changed_records["action"])
    changed_targets = m.returns_to_go(changed_records["reward"])
    changed_fitted, changed_history = m.fit_value_function(
        changed_records, changed_targets, updates=250, learning_rate=0.1
    )
    assert np.isfinite(np.asarray(changed_history)).all()
    assert not np.allclose(fitted, changed_fitted, atol=1e-4)
    print(
        "PASS stage 5: hand-computed multi-step GAE, pre-reset timeout bootstrap, finite-horizon opt-out, padding/empty cases, independent NumPy critic oracle"
    )
    print(
        "value fit seed 313: initial_mse=%.6f final_mse=%.6f table=%s"
        % (float(history[0]), float(history[-1]), np.asarray(fitted).round(4).tolist())
    )
    print(
        "value fit seed 314: initial_mse=%.6f final_mse=%.6f table=%s"
        % (
            float(changed_history[0]),
            float(changed_history[-1]),
            np.asarray(changed_fitted).round(4).tolist(),
        )
    )
if a.report:
    report = dict(
        schemaVersion=1,
        status="passed",
        stage=stage,
        implementationHash=hashlib.sha256(path.read_bytes()).hexdigest(),
        checkerHash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        environment=dict(
            python=platform.python_version(),
            jax=jax.__version__,
            numpy=np.__version__,
            backend=jax.default_backend(),
        ),
        results=results,
    )
    a.report.write_text(json.dumps(report, indent=2) + "\n")
