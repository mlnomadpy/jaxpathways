"""Cumulative public checks for exact derivatives and a bounded jaxpr interpreter."""
import argparse
import importlib.util
from pathlib import Path
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['1','2','3','all'],default='all');p.add_argument('--implementation',default='starter');args=p.parse_args()
stage=3 if args.stage=='all' else int(args.stage)
path=ROOT/args.implementation/'model.py' if args.implementation in ('starter','solution') else Path(args.implementation)
if not path.is_file():p.error('implementation must name an existing Python file')
spec=importlib.util.spec_from_file_location('learner_derivatives',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def reject(fn,error=ValueError):
    try:fn()
    except error:return
    raise AssertionError('Expected explicit '+error.__name__)

for point,direction,weight in [([.4,-.7],[1.2,-.3],[.5,2.]),([-.2,.8],[-.5,1.3],[1.1,-.4]),([.9,.1],[.2,.6],[-.8,.3])]:
    a,b=point;x=jnp.array(point);v=jnp.array(direction);u=jnp.array(weight)
    jac=np.array([[b,a],[np.cos(a),2*b]])
    value=np.array([a*b,np.sin(a)+b*b])
    hess=jac.T@jac+value[0]*np.array([[0,1],[1,0]])+value[1]*np.array([[-np.sin(a),0],[0,2]])
    actual=m.analyze(x,v,u)
    for name,reference in [('value',value),('jvp',jac@v),('vjp',jac.T@u),('hvp',hess@v)]:
        np.testing.assert_allclose(actual[name],reference,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(np.dot(u,actual['jvp']),np.dot(actual['vjp'],v),rtol=1e-11)
reject(lambda:m.analyze(jnp.ones(3),jnp.ones(2),jnp.ones(2)))
print('PASS stage 1: three independent Jacobian/Hessian references, adjoint identity and shape rejection')

if stage>=2:
    points=jnp.array([-1000.,-4.,0.,.4,3.,1000.])
    primal=jax.vmap(m.stable_softplus)(points)
    np.testing.assert_allclose(primal,np.logaddexp(0,np.asarray(points)),rtol=1e-12,atol=1e-12)
    slopes=jax.vmap(jax.grad(m.stable_softplus))(points)
    # NumPy stable sigmoid formula avoids overflow in the oracle too.
    sigmoid=np.exp(-np.logaddexp(0,-np.asarray(points)))
    np.testing.assert_allclose(slopes,sigmoid,rtol=1e-12,atol=1e-12)
    curvatures=jax.vmap(jax.grad(jax.grad(m.stable_softplus)))(points)
    np.testing.assert_allclose(curvatures,sigmoid*(1-sigmoid),rtol=1e-11,atol=1e-12)
    for values in ([-.8,.4,1.1],[-3.,.2,2.,5.]):
        x=jnp.array(values);host=np.asarray(x)
        weights=np.exp(host-host.max());weights/=weights.sum()
        np.testing.assert_allclose(m.stable_logsumexp(x),host.max()+np.log(np.exp(host-host.max()).sum()),rtol=1e-12)
        np.testing.assert_allclose(jax.grad(m.stable_logsumexp)(x),weights,rtol=1e-12)
        hess=jax.jacrev(jax.grad(m.stable_logsumexp))(x)
        np.testing.assert_allclose(hess,np.diag(weights)-np.outer(weights,weights),rtol=1e-11,atol=1e-12)
        pull=jax.vjp(m.stable_logsumexp,x)[1]
        np.testing.assert_allclose(pull(1.7)[0],2*pull(1.)[0]-pull(.3)[0],rtol=1e-12)
        np.testing.assert_allclose(jax.grad(m.stable_logsumexp)(x+1000),weights,rtol=1e-11)
        reject(lambda:jax.jvp(m.stable_logsumexp,(x,),(jnp.ones_like(x),)),TypeError)
    print('PASS stage 2: stable exact custom primals, gradients, curvature, cotangent linearity, common shifts and custom-VJP boundary')

if stage>=3:
    rng=np.random.default_rng(71)
    for shape in [(3,),(5,),(2,3)]:
        x=jnp.asarray(rng.normal(size=shape));v=jnp.asarray(rng.normal(size=shape));offset=jnp.asarray(rng.normal(size=shape))
        def function(z):return (jnp.sum(z*jnp.sin(z)+offset),-z)
        closed=jax.make_jaxpr(function)(x)
        primal,tangent=m.tiny_jvp(closed,(x,),(v,))
        host=np.asarray(x);seed=np.asarray(v)
        expected=np.sum((np.sin(host)+host*np.cos(host))*seed)
        np.testing.assert_allclose(primal[0],np.sum(host*np.sin(host)+np.asarray(offset)),rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(primal[1],-host,rtol=1e-12)
        np.testing.assert_allclose(tangent[0],expected,rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(tangent[1],-seed,rtol=1e-12)
        transformed=lambda z,d:m.tiny_jvp(closed,(z,),(d,))
        compiled=jax.jit(transformed).lower(x,v).compile()
        cp,ct=compiled(x+.1,v)
        host=host+.1
        np.testing.assert_allclose(ct[0],np.sum((np.sin(host)+host*np.cos(host))*seed),rtol=1e-11,atol=1e-12)
        reject(lambda:m.tiny_jvp(closed,(x,),(jnp.ones((1,)),)))
        reject(lambda:compiled(jnp.ones((7,)),jnp.ones((7,))),TypeError)
    x=jnp.array([.2,-.5,1.]);v=jnp.array([.7,.1,-.4]);c=jnp.array([1.,2.,3.]);dc=jnp.array([.2,-.1,.4])
    explicit=jax.make_jaxpr(lambda z,bias:jnp.sum(z*z+(-2.)*z+bias))(x,c)
    _,dt=m.tiny_jvp(explicit,(x,c),(v,dc))
    np.testing.assert_allclose(dt[0],np.dot(2*np.asarray(x)-2,np.asarray(v))+np.sum(dc),rtol=1e-11)
    unsupported=jax.make_jaxpr(lambda z:jnp.exp(z))(x)
    reject(lambda:m.tiny_jvp(unsupported,(x,),(v,)),NotImplementedError)
    reject(lambda:m.tiny_jvp(jax.make_jaxpr(lambda z:z+z)(x),(x,),(v.astype(jnp.float32),)))
    def effectful(z):
        jax.debug.print('diagnostic value {}',z)
        return z+z
    effects=jax.make_jaxpr(effectful)(x)
    reject(lambda:m.tiny_jvp(effects,(x,),(v,)),NotImplementedError)
    integer=jnp.array([1,2,3]);int_trace=jax.make_jaxpr(lambda z:z+z)(integer)
    reject(lambda:m.tiny_jvp(int_trace,(integer,),(integer,)))
    # The exercise requires an interpreter, not delegating its derivative to JAX.
    import inspect
    source=inspect.getsource(m.tiny_jvp)
    assert not any(token in source for token in ['jax.jvp(', 'jax.grad(', 'jax.jacfwd(', 'jax.jacrev(']),'Write explicit primitive tangent rules'
    print('PASS stage 3: random shapes, independent expression oracles, constants, multiple inputs/outputs, compiled changed values and explicit unsupported/signature rejection')
