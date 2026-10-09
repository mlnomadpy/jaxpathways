"""Cumulative independent contracts; passing public cases is not reviewer approval."""
import argparse
from itertools import product
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import jax
import jax.numpy as jnp

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--implementation',default='solution')
p.add_argument('--stage',default='all')
args=p.parse_args()
path=ROOT/'solution/optimization.py' if args.implementation=='solution' else Path(args.implementation).resolve()
spec=importlib.util.spec_from_file_location('learner',path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
last=5 if args.stage=='all' else int(args.stage)
assert 1<=last<=5

def rejects(fn):
    try:fn()
    except (ValueError,TypeError):return
    raise AssertionError('invalid contract accepted')

def analytic(w,x,y,penalty=0.):
    w,x,y=[np.asarray(a,np.float64) for a in [w,x,y]]
    r=x@w-y
    return .5*np.mean(r*r)+.5*penalty*(w@w),x.T@r/len(x)+penalty*w,x.T@x/len(x)+penalty*np.eye(len(w))

# Changed row/feature sizes and nonzero regularization make scalar/shape shortcuts fail.
for seed,n,d in [(7,5,2),(17,11,3),(29,1,4)]:
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(n,d)).astype(np.float32)
    y=rng.normal(size=n).astype(np.float32)
    w=rng.normal(size=d).astype(np.float32)
    before=[a.copy() for a in [w,x,y]]
    for penalty in [0.,.3]:
        expected=analytic(w,x,y,penalty)
        actual=m.geometry(w,x,y,penalty)
        for key,ref in zip(['value','gradient','hessian'],expected):np.testing.assert_allclose(actual[key],ref,atol=3e-6,rtol=3e-6)
        direction=np.linspace(-.3,.7,d)
        eps=1e-4
        fd=(analytic(w+eps*direction,x,y,penalty)[0]-analytic(w-eps*direction,x,y,penalty)[0])/(2*eps)
        np.testing.assert_allclose(np.dot(actual['gradient'],direction),fd,atol=3e-6,rtol=3e-6)
        hvp=(analytic(w+eps*direction,x,y,penalty)[1]-analytic(w-eps*direction,x,y,penalty)[1])/(2*eps)
        np.testing.assert_allclose(actual['hessian']@direction,hvp,atol=3e-6,rtol=3e-6)
    for a,b in zip([w,x,y],before):np.testing.assert_array_equal(a,b)
rejects(lambda:m.geometry(np.zeros(2,np.float32),np.zeros((3,2),np.float32),np.zeros((3,1),np.float32)))
print('PASS stage 1: independent loss/gradient/Hessian, directional differences, changed shapes and purity',flush=True)
if last==1:sys.exit()
x,y,_=m.fixture(3)
hx,hy,_=m.fixture(4)
s=m.fit_scales(x)
z=m.apply_scales(x,s)
hz=m.apply_scales(hx,s)
np.testing.assert_allclose(s,np.sqrt(np.mean(x.astype(np.float64)**2,axis=0)),rtol=2e-6)
np.testing.assert_array_equal(m.fit_scales(np.zeros((3,2),np.float32)),np.ones(2))
np.testing.assert_array_equal(m.fit_scales(np.ones((1,3),np.float32)),np.ones(3))
w=np.array([.4,-.2],np.float32)
np.testing.assert_allclose(z@(w*s),x@w,atol=2e-6,rtol=2e-6)
raw_h=analytic(w,x,y)[2]
scaled_h=analytic(w,z,y)[2]
assert np.linalg.cond(scaled_h)<np.linalg.cond(raw_h)/50
rejects(lambda:m.apply_scales(x,np.array([1,0],np.float32)))
print('PASS stage 2: train-only RMS, prediction-preserving coordinate change and observed conditioning',flush=True)
if last==2:sys.exit()
xsmall=np.array([[-2.,1.],[0.,-1.],[3.,2.]],np.float32)
ysmall=np.array([1.,-2.,.5],np.float32)
ws=np.array([.25,-.5],np.float32)
per=(xsmall@ws-ysmall)[:,None]*xsmall+.2*ws
single_cov=(per-per.mean(0)).astype(np.float64).T@(per-per.mean(0))/3
for batch in [1,2,3]:
    audit=m.noise_audit(ws,xsmall,ysmall,batch,.2)
    draws=np.array(list(product(range(3),repeat=batch)))
    np.testing.assert_array_equal(audit['draws'],draws)
    np.testing.assert_allclose(audit['gradients'],per[draws].mean(1),atol=2e-6)
    np.testing.assert_allclose(audit['mean'],analytic(ws,xsmall,ysmall,.2)[1],atol=2e-6)
    np.testing.assert_allclose(audit['covariance'],single_cov/batch,atol=3e-6)
# Exact counts distinguish a dataset mean from an unweighted mean of uneven batch means.
weighted=(2*per[:2].mean(0)+per[2])/3
naive=(per[:2].mean(0)+per[2])/2
np.testing.assert_allclose(weighted,per.mean(0),atol=2e-6)
assert np.linalg.norm(naive-per.mean(0))>.1
rejects(lambda:m.noise_audit(ws,xsmall,ysmall,20))
print('PASS stage 3: full ordered enumeration, unbiased mean, covariance / batch and uneven-batch diagnosis',flush=True)
if last==3:sys.exit()

def host_run(w,x,y,hx,hy,ids,lrs,method,penalty=0.,clip=None):
    w=w.astype(np.float64).copy()
    first=np.zeros_like(w)
    second=np.zeros_like(w)
    trajectory=[]
    held=[]
    training=[]
    for k,(chosen,rate) in enumerate(zip(ids,lrs),start=1):
        g=analytic(w,x[chosen],y[chosen],penalty)[1]
        if clip is not None:g=g*min(1.,clip/max(np.linalg.norm(g),1e-12))
        if method=='gd':direction=g
        elif method=='momentum':
            first=.85*first+g
            direction=first
        else:
            first=.9*first+.1*g
            second=.99*second+.01*g*g
            direction=(first/(1-.9**k))/(np.sqrt(second/(1-.99**k))+1e-8)
        w=w-float(rate)*direction
        trajectory.append(w.copy())
        held.append(np.mean((hx@w-hy)**2))
        training.append(np.mean((x@w-y)**2))
    return np.array(trajectory),np.array(held),np.array(training)

# All methods consume exactly the same sampled observations; all metrics are post-update.
plan=m.batch_plan(17,len(z),60,12)
np.testing.assert_array_equal(plan,m.batch_plan(17,len(z),60,12))
assert not np.array_equal(plan,m.batch_plan(18,len(z),60,12))
lrs=m.rates(.03,60,decay_at=30,factor=.2)
assert np.isclose(lrs[29],.03) and np.isclose(lrs[30],.006)
for method,clip in [('gd',None),('momentum',None),('adam',None),('adam',.2)]:
    initial=np.array([.05,-.1],np.float32)
    final,trace=m.run(initial,z,y,hz,hy,plan,lrs,method,.02,clip)
    expected,held,training=host_run(initial,z,y,hz,hy,plan,lrs,method,.02,clip)
    np.testing.assert_allclose(trace['weights'],expected,atol=2e-5,rtol=2e-5)
    np.testing.assert_allclose(trace['held_mse'],held,atol=2e-5,rtol=2e-5)
    np.testing.assert_allclose(trace['train_mse'],training,atol=2e-5,rtol=2e-5)
    np.testing.assert_allclose(trace['batch_objective_before'][0],analytic(initial,z[plan[0]],y[plan[0]],.02)[0],atol=2e-6)
    np.testing.assert_array_equal(final['weights'],trace['weights'][-1])
    assert int(final['step'])==60
    if clip is not None:assert max(trace['clipped_norm'])<=clip+1e-6
state={'weights':jnp.zeros(2),'first':jnp.zeros(2),'second':jnp.zeros(2),'step':jnp.int32(0)}
_,diagnostic=m.optimizer_step(state,jnp.array([100.,.01]),.1,'adam',.2)
assert float(diagnostic['update_norm'])>.1*.2*2  # clipping BEFORE Adam does not bound its normalized update.
# A spectral counterexample isolates the largest-curvature eigendirection.
x=np.array([[1.,0.],[0.,4.]],np.float32)
y=np.zeros(2,np.float32)
initial=np.array([0.,1.],np.float32)
plan=np.tile(np.arange(2,dtype=np.int32),(25,1))
largest=8.
for ratio in [.8,2.2]:
    _,trace=m.run(initial,x,y,x,y,plan,m.rates(ratio/largest,25))
    exact=(1-ratio)**np.arange(1,26)
    np.testing.assert_allclose(trace['weights'][:,1],exact,atol=2e-4,rtol=2e-5)
    assert np.isfinite(trace['train_mse']).all()
    assert (trace['train_mse'][-1]>8.) == (ratio>2)
rejects(lambda:m.run(initial,x,y,x,y,plan.astype(np.float32),m.rates(.1,25)))
print('PASS stage 4: NumPy optimizer trajectories, schedule boundary, clipping order and finite divergence',flush=True)
if last==4:sys.exit()
# Exact duplicate features make coefficients nonunique although predictions can be fixed.
x=np.array([[-2.,-2.],[-1.,-1.],[1.,1.],[2.,2.]],np.float32)
y=(2*x[:,0]).astype(np.float32)
for penalty in [0.,.01,1.]:
    fitted=m.ridge_solution(x,y,penalty)
    expected=np.full(2,5/(5+penalty))
    np.testing.assert_allclose(fitted,expected,atol=2e-8)
    gradient=analytic(fitted,x,y,penalty)[1]
    np.testing.assert_allclose(gradient,0,atol=2e-8)
np.testing.assert_array_equal(x@np.array([2.,0.]),x@np.array([-3.,5.]))
# Changed random design: check a stationarity equation independently of augmented least squares.
x,y,_=m.fixture(77,n=17,feature_ratio=1.5,correlation=.98)
for penalty in [.03,.5]:
    fitted=m.ridge_solution(x,y,penalty)
    h=x.astype(np.float64).T@x/len(x)+penalty*np.eye(2)
    right=x.astype(np.float64).T@y/len(x)
    np.testing.assert_allclose(h@fitted,right,atol=2e-10)
rejects(lambda:m.ridge_solution(x,y,-.1))
print('PASS stage 5: analytic rank-deficient ridge, prediction ambiguity and changed-design stationarity',flush=True)
print(json.dumps({'jax':jax.__version__,'numpy':np.__version__,'backend':jax.default_backend(),'scope':'synthetic CPU numerical optimization verification'},indent=2))
