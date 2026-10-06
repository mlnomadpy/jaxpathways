"""Independent public CPU checks; passing does not establish real-data calibration."""
import argparse
import importlib.util
from pathlib import Path
import math
import jax
import jax.numpy as jnp
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument("--stage",choices=["1","2","3","all"],default="all")
parser.add_argument("--implementation",default="starter")
args=parser.parse_args()
stage=3 if args.stage=="all" else int(args.stage)
path=ROOT/args.implementation/"model.py" if args.implementation in ("starter","solution") else Path(args.implementation)
if not path.is_file():parser.error("implementation must name an existing Python file")
spec=importlib.util.spec_from_file_location("learner",path)
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)

def rejects(call):
    try:call()
    except ValueError:return
    raise AssertionError("Expected a clear ValueError for malformed input")

# Several feature counts, non-centered inputs, scales and rank-deficient data.
for seed,d,n,sigma,tau in [(5,2,7,.7,1.3),(21,3,9,1.2,.8),(41,2,4,.4,2.)]:
    rng=np.random.default_rng(seed)
    X=rng.normal(size=(n,d)).astype(np.float32);X[:,0]=1.
    if seed==41:X[:,1]=1.
    y=rng.normal(size=n).astype(np.float32)
    mean,cov=model.fit(X,y,sigma,tau)
    hostX=X.astype(np.float64);hosty=y.astype(np.float64)
    precision=np.eye(d)/tau**2+hostX.T@hostX/sigma**2
    oracle_mean=np.linalg.solve(precision,hostX.T@hosty/sigma**2)
    oracle_cov=np.linalg.solve(precision,np.eye(d))
    np.testing.assert_allclose(mean,oracle_mean,rtol=2e-5,atol=2e-6)
    np.testing.assert_allclose(cov,oracle_cov,rtol=2e-5,atol=2e-6)
    w=jnp.asarray(rng.normal(size=d),dtype=jnp.float32)
    expected=(-.5*np.sum((hosty-hostX@np.asarray(w))**2)/sigma**2-n*math.log(sigma)
              -.5*np.sum(np.asarray(w,dtype=np.float64)**2)/tau**2-d*math.log(tau)
              -.5*(n+d)*math.log(2*math.pi))
    np.testing.assert_allclose(model.log_joint(w,jnp.array(X),jnp.array(y),sigma,tau),expected,rtol=1e-5)
    gradient=jax.grad(model.log_joint)(w,jnp.array(X),jnp.array(y),sigma,tau)
    oracle_gradient=hostX.T@(hosty-hostX@np.asarray(w))/sigma**2-np.asarray(w)/tau**2
    np.testing.assert_allclose(gradient,oracle_gradient,rtol=2e-5,atol=3e-5)
empty_m,empty_c=model.fit(np.empty((0,3)),np.empty(0),.7,1.5)
np.testing.assert_allclose(empty_m,0);np.testing.assert_allclose(empty_c,np.eye(3)*2.25)
rejects(lambda:model.fit(np.ones((3,2)),np.ones((3,1))))
rejects(lambda:model.fit(np.ones((3,2)),np.ones(3),0.))
rejects(lambda:model.fit(np.array([[np.nan,1.]]),np.ones(1)))
print("PASS stage 1: posterior oracle, non-centered/rank-deficient data, log density, derivatives, input contracts")

if stage>=2:
    target_m=jnp.array([.7,-.4])
    target_c=jnp.array([[.8,.3],[.3,.5]])
    starts=jnp.array([[-3.,-3.],[-3.,3.],[3.,-3.],[3.,3.]])
    for seed in (19,43):
        record=model.sample(jax.random.key(seed),target_m,target_c,starts)
        chains=np.asarray(record["chains"])
        assert chains.shape==(4,1800,2)
        assert np.asarray(record["accepted"]).shape==(4,1800)
        assert np.asarray(record["energy_error"]).shape==(4,1800)
        assert np.isfinite(record["energy_error"]).all()
        np.testing.assert_allclose(chains.mean(axis=(0,1)),target_m,atol=.10)
        np.testing.assert_allclose(np.cov(chains.reshape(-1,2).T),target_c,atol=.12)
        report=model.diagnose(chains)
        half=chains.shape[1]//2
        split=np.concatenate((chains[:,:half],chains[:,-half:]),axis=0)
        W=np.var(split,axis=1,ddof=1).mean(0)
        B=half*np.var(split.mean(1),axis=0,ddof=1)
        expected=np.sqrt(((half-1)*W/half+B/half)/W)
        np.testing.assert_allclose(report["classical_split_rhat"],expected,rtol=1e-6)
        assert np.max(expected)<1.05
        print("seed",seed,"mean",report["mean"],"classical R-hat",expected)
    short_args=dict(warmup=10,draws=40)
    a=model.sample(jax.random.key(8),target_m,target_c,starts,**short_args)
    b=model.sample(jax.random.key(8),target_m,target_c,starts,**short_args)
    c=model.sample(jax.random.key(9),target_m,target_c,starts,**short_args)
    np.testing.assert_array_equal(a["chains"],b["chains"])
    assert not np.array_equal(a["chains"],c["chains"])
    fake=np.random.default_rng(12).normal(0,.1,(4,100,2))+np.arange(4)[:,None,None]*3
    assert np.min(model.diagnose(fake)["classical_split_rhat"])>5
    rejects(lambda:model.diagnose(np.ones((4,100,2))))
    unstable=model.sample(jax.random.key(19),target_m,target_c,starts,step_size=2.5,warmup=0,draws=40)
    assert np.mean(unstable["accepted"])<.1
    assert np.max(unstable["energy_error"])>100
    # Rejecting does not create a new state; all rejected first states retain starts.
    first_accepted=np.asarray(unstable["accepted"])[:,0]
    np.testing.assert_allclose(np.asarray(unstable["chains"])[~first_accepted,0],np.asarray(starts)[~first_accepted])
    print("PASS stage 2: two-seed moments, replay, changed keys, R-hat oracle, stuck and unstable chains")

if stage>=3:
    X=jnp.array([[1.,-1.],[1.,0.],[1.,1.]])
    y=jnp.array([-1.,1.,3.])
    mean,cov=model.fit(X,y)
    query=jnp.array([[1.,0.],[1.,2.],[1.,-3.]])
    report=model.predict(query,mean,cov,.7)
    reference=np.array([row@np.asarray(cov)@row for row in np.asarray(query)])
    np.testing.assert_allclose(report["mean"],np.asarray(query)@np.asarray(mean),rtol=1e-6)
    np.testing.assert_allclose(report["latent_variance"],reference,rtol=1e-6)
    np.testing.assert_allclose(report["observation_variance"],reference+.49,rtol=1e-6)
    assert report["latent_variance"][2]>report["latent_variance"][0]
    rng=np.random.default_rng(62)
    draws=rng.multivariate_normal(np.asarray(mean),np.asarray(cov),size=30000)
    replicated=draws@np.asarray(query).T+rng.normal(0,.7,(30000,3))
    np.testing.assert_allclose(replicated.var(0),report["observation_variance"],rtol=.04)
    rejects(lambda:model.predict(query[:,0],mean,cov))
    rejects(lambda:model.predict(query,mean,cov,-1.))
    rejects(lambda:model.predict(query,mean,jnp.array([[1.,2.],[2.,1.]])))
    # Compare the same nominal intervals under modeled and deliberately shifted outcomes.
    center=np.asarray(report["mean"])
    sd=np.sqrt(np.asarray(report["observation_variance"]))
    covered=np.abs(replicated-center)<=1.96*sd
    shifted=np.abs(replicated+5*sd-center)<=1.96*sd
    assert .93<covered.mean()<.97 and shifted.mean()<.02
    print("PASS stage 3: independent predictive variance, noise simulation, extrapolation, shifted-outcome check")
print("All requested public stages passed. Human review and real-data validation remain separate.")
