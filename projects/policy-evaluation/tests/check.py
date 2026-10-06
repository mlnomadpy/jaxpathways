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
p.add_argument('--implementation', default='starter')
p.add_argument('--stage', type=int, choices=[1, 2, 3, 4], default=4)
p.add_argument('--report', type=Path)
a = p.parse_args()
path = ROOT / a.implementation / 'model.py' if a.implementation in ('starter', 'solution') else Path(a.implementation)
spec = importlib.util.spec_from_file_location('learner', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def enumerate_return(theta, horizon):
    probs = 1 / (1 + np.exp(-np.asarray(theta, dtype=float)))
    def visit(position, remaining):
        if position == 3 or remaining == 0:
            return 0.
        total = 0.
        for action in (0, 1):
            nxt = min(3, max(0, position + (1 if action else -1)))
            prob = probs[position] if action else 1-probs[position]
            total += prob * ((1. if nxt == 3 else -.01) + visit(nxt, remaining-1))
        return total
    return (visit(0, horizon) + visit(1, horizon))/2


for horizon in (1, 3, 8):
    for position in range(4):
        for elapsed in range(horizon):
            for action in (0, 1):
                s = m.State(jnp.int32(position), jnp.int32(elapsed), jnp.bool_(position == 3))
                nxt, reward, term, trunc = jax.jit(lambda s, a: m.step(s, a, horizon))(s, action)
                if position == 3:
                    assert int(nxt.position) == 3 and float(reward) == 0 and not term and not trunc
                else:
                    target = min(3, max(0, position + (1 if action else -1)))
                    assert int(nxt.position) == target and int(nxt.elapsed) == elapsed+1
                    assert bool(term) == (target == 3)
                    assert bool(trunc) == (target != 3 and elapsed+1 >= horizon)
                    assert bool(nxt.done) == bool(term | trunc)
                    np.testing.assert_allclose(reward, 1. if target == 3 else -.01)
keys = jax.random.split(jax.random.key(9), 1000)
starts = jax.vmap(m.reset)(keys)
assert set(np.asarray(starts.position).tolist()) == {0, 1}
assert .4 < float(starts.position.mean()) < .6
assert not bool(starts.done.any()) and not bool(starts.elapsed.any())
print('PASS stage 1: enumerated transitions, boundary priority, absorbing state, randomized reset')

if a.stage >= 2:
    for horizon, batch in [(1, 7), (5, 19), (8, 37)]:
        theta = jnp.array([-.7, .3, 1.1])
        final, r = m.rollout(theta, jax.random.key(81), batch, horizon)
        assert r['reward'].shape == (horizon, batch) and bool(final.done.all())
        assert np.all(np.sum(np.asarray(r['terminated'] | r['truncated']), axis=0) == 1)
        np.testing.assert_array_equal(r['reward'][~r['active']], 0)
        # Reconstruct every trajectory using Python, independent of the JAX step.
        for b in range(batch):
            position = int(r['observation'][0, b]); done = False
            for t in range(horizon):
                assert int(r['observation'][t,b]) == position
                assert bool(r['active'][t,b]) == (not done)
                expected = 0.
                if not done:
                    position = min(3, max(0, position+(1 if r['action'][t,b] else -1)))
                    expected = 1. if position == 3 else -.01
                    done = position == 3 or t == horizon-1
                np.testing.assert_allclose(r['reward'][t,b], expected)
        replay = m.rollout(theta, jax.random.key(81), batch, horizon)[1]
        for name in r: np.testing.assert_array_equal(r[name], replay[name])
        other = m.rollout(theta, jax.random.key(82), batch, horizon)[1]
        assert not np.array_equal(r['action'], other['action'])
    rewards = jnp.array([[1., 2.], [3., 4.], [5., 0.]])
    np.testing.assert_array_equal(m.returns_to_go(rewards), [[9,6], [8,4], [5,0]])
    print('PASS stage 2: trajectory reconstruction, padding mask, event count, changed seed, return targets')

if a.stage >= 3:
    for theta in [jnp.array([-.5,.2,1.]), jnp.array([.7,-.4,.8])]:
        for horizon in [3, 5, 8]:
            np.testing.assert_allclose(m.exact_return(theta, horizon), enumerate_return(theta, horizon), atol=2e-7)
        independent = []
        eps = 1e-4
        for i in range(3):
            d = np.eye(3)[i]*eps
            independent.append((enumerate_return(np.asarray(theta)+d,8)-enumerate_return(np.asarray(theta)-d,8))/(2*eps))
        np.testing.assert_allclose(jax.grad(m.exact_return)(theta), independent, rtol=2e-4, atol=2e-5)
    # Deliberate mixed-sign advantages and stale behavior probabilities.
    r = dict(observation=jnp.array([[0,1],[1,2]]), action=jnp.array([[1,0],[0,1]]),
             reward=jnp.array([[.4,-.2],[-.1,.3]]), active=jnp.array([[1,1],[1,0]],bool),
             old_logp=jnp.log(jnp.array([[.2,.8],[.7,.4]])))
    theta = jnp.array([.6,-.5,.9])
    newp = 1/(1+np.exp(-np.asarray(theta)[np.asarray(r['observation'])]))
    newp = np.where(np.asarray(r['action']) == 1,newp,1-newp)
    adv = np.array([[.3,.1],[-.1,.3]])
    ratios = newp / np.exp(np.asarray(r['old_logp']))
    expected = -np.sum(np.minimum(ratios*adv,np.clip(ratios,.8,1.2)*adv)*np.asarray(r['active']))/2
    np.testing.assert_allclose(m.policy_loss(theta,r), expected,rtol=1e-6)
    logp_grad = jax.grad(lambda old: m.policy_loss(theta, dict(r,old_logp=old)))(r['old_logp'])
    reward_grad = jax.grad(lambda rew: m.policy_loss(theta, dict(r,reward=rew)))(r['reward'])
    np.testing.assert_array_equal(logp_grad,0); np.testing.assert_array_equal(reward_grad,0)
    # Fresh Monte Carlo policy gradient agrees with differentiating an enumerated expectation.
    theta = jnp.array([-.2,.3,.7])
    r = m.rollout(theta,jax.random.key(902),32768,8)[1]
    estimate = -jax.grad(m.policy_loss)(theta,r,'reinforce')
    np.testing.assert_allclose(estimate,jax.grad(m.exact_return)(theta),atol=.012,rtol=.04)
    print('PASS stage 3: independent enumeration and finite differences, frozen targets, PPO signs, Monte Carlo gradient')

results = []
if a.stage >= 4:
    for method, epochs in [('reinforce',1),('ppo',3)]:
        for seed in (0, 7, 23):
            theta, history = m.train(seed=seed,method=method,epochs=epochs)
            replay, rh = m.train(seed=seed,method=method,epochs=epochs)
            np.testing.assert_array_equal(theta,replay); np.testing.assert_array_equal(history,rh)
            assert history.shape == (41,) and np.isfinite(history).all()
            assert float(history[-1]) > .90 and float(history[-1]-history[0]) > .35
            before = np.array(theta,copy=True)
            evaluation = m.evaluate(theta,[1000,1001,1002,1003])
            np.testing.assert_array_equal(theta,before)
            assert abs(evaluation['mean']-enumerate_return(theta,8)) < .03
            results.append(dict(method=method,trainingSeed=seed,exact=float(history[-1]),heldOut=evaluation['mean']))
    for bad in [jnp.zeros(2),jnp.array([0.,jnp.nan,0.])]:
        try: m.evaluate(bad,[1,2])
        except ValueError: pass
        else: raise AssertionError('invalid policy accepted')
    for seeds in [[1],[1,1]]:
        try: m.evaluate(jnp.zeros(3),seeds)
        except ValueError: pass
        else: raise AssertionError('evaluation seed contract ignored')
    print('PASS stage 4: six trained agents, repeatability, held-out streams, state isolation, input rejection')
    print(json.dumps(results,indent=2))
if a.report:
    report = dict(schemaVersion=1,status='passed',stage=a.stage,
                  implementationHash=hashlib.sha256(path.read_bytes()).hexdigest(),
                  checkerHash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  environment=dict(python=platform.python_version(),jax=jax.__version__,numpy=np.__version__,backend=jax.default_backend()),results=results)
    a.report.write_text(json.dumps(report,indent=2)+'\n')
