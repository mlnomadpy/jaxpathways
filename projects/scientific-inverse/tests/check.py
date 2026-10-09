"""Public cumulative CPU checks. Passing is not independent scientific review."""
import argparse
import importlib.util
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--stage',choices=['1','2','3','all'],default='all')
parser.add_argument('--implementation',default='starter')
args=parser.parse_args()
stage=3 if args.stage=='all' else int(args.stage)
path=ROOT/args.implementation/'model.py' if args.implementation in ('starter','solution') else Path(args.implementation)
if not path.is_file():parser.error('implementation must name an existing Python file')
spec=importlib.util.spec_from_file_location('learner_science',path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def reject(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('Expected an explicit ValueError for malformed input')

def oracle(k,initials,n,h):
    z=-k*h
    r=1+z+z*z/2+z**3/6+z**4/24
    return np.asarray(initials)[:,None]*r**np.arange(n+1)

for k,starts,n,h in [(0.3,[.4,1.2,2.8],20,.1),(1.1,[1.7],40,.05)]:
    initials=jnp.array(starts)
    actual=m.simulate(k,initials,n,h)
    assert actual.shape==(len(starts),n+1)
    np.testing.assert_allclose(actual,oracle(k,starts,n,h),rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(actual,np.asarray(starts)[:,None]*np.exp(-k*np.arange(n+1)*h),rtol=4e-7)
    perm=np.arange(len(starts))[::-1]
    np.testing.assert_allclose(m.simulate(k,initials[perm],n,h),np.asarray(actual)[perm],rtol=1e-12)
reject(lambda:m.simulate(.7,jnp.ones((2,1))))
reject(lambda:m.simulate(.7,jnp.array([])))
reject(lambda:m.simulate(.7,jnp.ones(2),0,.05))
reject(lambda:m.simulate(.7,jnp.ones(2),5,-.05))
print('PASS stage 1: two rates, grids, independent polynomial and exponential oracles, axes, invalid inputs')

if stage>=2:
    starts=jnp.array([.8,1.7,2.2])
    n,h=40,.05
    observed=jnp.asarray(np.asarray(starts)[:,None]*np.exp(-.65*np.arange(n+1)*h))
    for candidate in (.4,1.0):
        theta=np.log(candidate)
        reference=np.mean((oracle(candidate,starts,n,h)-np.asarray(observed))**2)
        np.testing.assert_allclose(m.loss(theta,starts,observed,h),reference,rtol=1e-11)
        eps=1e-5
        host_loss=lambda p:np.mean((oracle(np.exp(p),starts,n,h)-np.asarray(observed))**2)
        finite=(host_loss(theta+eps)-host_loss(theta-eps))/(2*eps)
        np.testing.assert_allclose(jax.grad(m.loss)(theta,starts,observed,h),finite,rtol=1e-7,atol=1e-10)
    reject(lambda:m.loss(np.log(.7),starts,observed.T,h))
    reject(lambda:m.loss(np.log(.7),starts,observed[:,:,None],h))
    reject(lambda:m.loss(np.log(.7),starts,observed[:,:1],h))
    print('PASS stage 2: independent objective, log-rate finite differences at two probes, malformed-target rejection')

if stage>=3:
    train=jnp.array([.7,1.5,2.3])
    held=jnp.array([.4,1.1,3.2])
    times=np.arange(41)*.05
    for truth,start in [(.35,.9),(1.05,1.8)]:
        labels=jnp.asarray(np.asarray(train)[:,None]*np.exp(-truth*times))
        test=jnp.asarray(np.asarray(held)[:,None]*np.exp(-truth*times))
        labels_before=np.array(labels,copy=True)
        rate,history=m.fit(train,labels,initial_rate=start,updates=300)
        assert np.isfinite(rate) and abs(rate-truth)<2e-5
        assert history.shape==(301,) and np.isfinite(history).all()
        np.testing.assert_allclose(history[0],np.mean((oracle(start,train,40,.05)-labels_before)**2),rtol=1e-11)
        assert history[-1]<history[0]*1e-7
        replay,replay_history=m.fit(train,labels,initial_rate=start,updates=300)
        np.testing.assert_allclose(rate,replay,rtol=1e-12)
        np.testing.assert_allclose(history,replay_history,rtol=1e-12,atol=1e-15)
        metrics=m.evaluate(rate,held,test)
        residual=oracle(rate,held,40,.05)-np.asarray(test)
        np.testing.assert_allclose(metrics['mse'],np.mean(residual**2),rtol=1e-6,atol=1e-20)
        np.testing.assert_allclose(metrics['rmse'],np.sqrt(np.mean(residual**2)),rtol=1e-6,atol=1e-12)
        np.testing.assert_allclose(metrics['max_abs_error'],np.max(np.abs(residual)),rtol=1e-6,atol=1e-12)
        assert metrics['trajectory_count']==3 and metrics['observation_count']==123
        assert metrics['rmse']<1e-5
        np.testing.assert_array_equal(labels,labels_before)
        refined=m.simulate(rate,held,80,.025)[:,::2]
        assert np.sqrt(np.mean((np.asarray(refined)-np.asarray(test))**2))<1e-5
        print('Recovered rate, held-out RMSE:',truth,rate,metrics['rmse'])
    noise=np.random.default_rng(19).normal(0,.005,(3,41))
    noisy=jnp.asarray(np.asarray(train)[:,None]*np.exp(-.7*times)+noise)
    noisy_rate,_=m.fit(train,noisy,updates=300)
    grid=np.linspace(.65,.75,1001)
    errors=np.mean((np.asarray(train)[None,:,None]*np.exp(-grid[:,None,None]*times)-np.asarray(noisy)[None])**2,axis=(1,2))
    assert abs(noisy_rate-grid[np.argmin(errors)])<.0002
    reject(lambda:m.fit(train,noisy,initial_rate=0))
    reject(lambda:m.evaluate(-.7,held,test))
    print('PASS stage 3: changed-parameter recovery, frozen held-out metrics, replay, refinement, noisy independent grid-search agreement')
