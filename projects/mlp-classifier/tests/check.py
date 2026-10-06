"""Public checks for three cumulative stages; they are not expert review."""
import argparse
import importlib.util
from pathlib import Path
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--stage', type=int, choices=[1,2,3], default=3)
parser.add_argument('--implementation', default='starter', help='starter, solution, or a Python implementation path')
args = parser.parse_args()
implementation = ROOT/args.implementation/'model.py' if args.implementation in ['starter','solution'] else Path(args.implementation)
if not implementation.is_file() or implementation.suffix != '.py':
    parser.error('implementation must name an existing Python file')
spec = importlib.util.spec_from_file_location('learner_model', implementation)
learner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(learner)


def snapshot(m):
    return [np.array(a, copy=True) for a in jax.tree.leaves(nnx.state(m))]


def independent_scores(m, x):
    hidden = np.tanh(np.asarray(x) @ np.asarray(m.hidden.kernel[...]) + np.asarray(m.hidden.bias[...]))
    return (hidden @ np.asarray(m.out.kernel[...]) + np.asarray(m.out.bias[...])).squeeze(-1)


def independent_loss(scores, labels):
    return np.mean(np.logaddexp(0., scores) - np.asarray(labels)*scores)


x = jnp.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
y = jnp.array([0.,1.,1.,0.])
for width in [3,8]:
    model = learner.make_model(seed=3, width=width)
    assert sum(a.size for a in jax.tree.leaves(nnx.state(model,nnx.Param))) == 4*width+1
    for batch in [x,jnp.array([[.3,-.8]]),jnp.array([[.1,.2],[-.7,.4],[.6,.9]])]:
        actual = model(batch)
        assert actual.shape == (len(batch),)
        np.testing.assert_allclose(actual, independent_scores(model,batch),rtol=1e-5,atol=1e-6)
    replay = learner.make_model(seed=3,width=width)
    for a,b in zip(snapshot(model),snapshot(replay)):np.testing.assert_array_equal(a,b)
print('PASS stage 1: shape, wiring, parameter count, seeded initialization')

if args.stage >= 2:
    for seed in [0,4]:
        model = learner.make_model(seed,8)
        np.testing.assert_allclose(learner.objective(model,x,y),independent_loss(independent_scores(model,x),y),rtol=1e-5)
        gradients = nnx.grad(lambda m:learner.objective(m,x,y))(model)
        actual_grad = float(gradients['out']['bias'][...][0])
        original = np.array(model.out.bias[...],copy=True)
        eps = .002
        model.out.bias[...] = jnp.array(original+eps)
        plus = independent_loss(independent_scores(model,x),y)
        model.out.bias[...] = jnp.array(original-eps)
        minus = independent_loss(independent_scores(model,x),y)
        model.out.bias[...] = jnp.array(original)
        np.testing.assert_allclose(actual_grad,(plus-minus)/(2*eps),rtol=.02,atol=2e-4)
        caught = False
        try:learner.objective(model,x,y[:,None])
        except ValueError:caught = True
        assert caught,'The loss must reject pairwise-broadcast labels'
    print('PASS stage 2: independent loss, two-seed finite differences, label contract')

if args.stage >= 3:
    rng = np.random.default_rng(101)
    held_x = np.repeat(np.asarray(x),10,axis=0)+rng.normal(0,.12,(40,2))
    held_y = (held_x[:,0]*held_x[:,1]<0).astype(np.float32)
    held_x,held_y = jnp.array(held_x),jnp.array(held_y)
    model,history = learner.train(0,x,y)
    assert history.shape==(200,) and np.all(np.isfinite(history))
    initial = learner.make_model(0,8)
    np.testing.assert_allclose(history[0],independent_loss(independent_scores(initial,x),y),rtol=1e-5)
    assert history[-1]<history[0]*.1
    frozen = snapshot(model)
    metrics = learner.evaluate(model,held_x,held_y)
    assert metrics['count']==40 and metrics['accuracy']>=.95
    scores = independent_scores(model,held_x)
    np.testing.assert_allclose(metrics['loss'],independent_loss(scores,held_y),rtol=1e-5,atol=1e-6)
    np.testing.assert_allclose(metrics['accuracy'],np.mean((scores>0)==np.asarray(held_y)),atol=1e-6)
    for a,b in zip(frozen,snapshot(model)):np.testing.assert_array_equal(a,b)
    parts = [learner.evaluate(model,held_x[:7],held_y[:7]),learner.evaluate(model,held_x[7:],held_y[7:])]
    for field in ['loss','accuracy']:
        weighted = sum(p[field]*p['count'] for p in parts)/40
        np.testing.assert_allclose(weighted,metrics[field],rtol=1e-5,atol=1e-6)
    replay,replay_history = learner.train(0,x,y)
    for a,b in zip(snapshot(model),snapshot(replay)):np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(history,replay_history,rtol=1e-6,atol=1e-6)
    changed,changed_history = learner.train(1,x,y,steps=160)
    assert changed_history.shape==(160,)
    assert learner.evaluate(changed,held_x,held_y)['accuracy']>=.9
    print('PASS stage 3: held-out metrics, unequal-batch aggregation, evaluation isolation, full replay, changed seed')
    print('Measured synthetic held-out metrics:',metrics)
