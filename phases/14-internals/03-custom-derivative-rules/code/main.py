"""Custom derivative rules: worked experiments and reference solutions. CPU checks."""

# Write stable primal functions and exact derivative contracts
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

@jax.custom_jvp
def stable_softplus(x):
    return jnp.logaddexp(0.0,x)

@stable_softplus.defjvp
def softplus_jvp(primals,tangents):
    x,=primals
    dx,=tangents
    return stable_softplus(x),jax.nn.sigmoid(x)*dx

@jax.custom_vjp
def stable_logsumexp(x):
    return jax.scipy.special.logsumexp(x)

def logsumexp_fwd(x):
    value=stable_logsumexp(x)
    weights=jax.nn.softmax(x)
    return value,weights

def logsumexp_bwd(weights,cotangent):
    return (cotangent*weights,)

stable_logsumexp.defvjp(logsumexp_fwd,logsumexp_bwd)

# Check first and second derivatives independently
points=jnp.array([-4.,0.,3.])
values=jax.vmap(stable_softplus)(points)
first=jax.vmap(jax.grad(stable_softplus))(points)
second=jax.vmap(jax.grad(jax.grad(stable_softplus)))(points)
reference_first=1/(1+np.exp(-np.asarray(points)))
reference_second=reference_first*(1-reference_first)
np.testing.assert_allclose(values,np.logaddexp(0,np.asarray(points)),rtol=1e-12)
np.testing.assert_allclose(first,reference_first,rtol=1e-12)
np.testing.assert_allclose(second,reference_second,rtol=1e-12)
x=jnp.array([-0.8,0.4,1.1])
weights=np.exp(np.asarray(x)-np.max(np.asarray(x)));weights/=weights.sum()
hessian_reference=np.diag(weights)-np.outer(weights,weights)
np.testing.assert_allclose(jax.grad(stable_logsumexp)(x),weights,rtol=1e-12)
np.testing.assert_allclose(jax.jacrev(jax.grad(stable_logsumexp))(x),hessian_reference,rtol=1e-12,atol=1e-12)
print("Softplus first/second:",np.asarray(first),np.asarray(second))
print("Logsumexp gradient:",weights)

# Exercise extremes and seed linearity
extremes=jnp.array([-1000.,0.,1000.])
extreme_values=jax.vmap(stable_softplus)(extremes)
extreme_gradients=jax.vmap(jax.grad(stable_softplus))(extremes)
assert np.isfinite(extreme_values).all() and np.isfinite(extreme_gradients).all()
np.testing.assert_allclose(extreme_values,[0.,np.log(2.),1000.],atol=1e-12)
with np.errstate(over='ignore'):
    naive=np.log1p(np.exp(np.asarray(extremes)))
assert not np.isfinite(naive[-1])
_,pullback=jax.vjp(stable_logsumexp,x)
np.testing.assert_allclose(pullback(2.0-0.3)[0],2*pullback(1.0)[0]-pullback(0.3)[0],rtol=1e-12)
print("Stable extreme values:",np.asarray(extreme_values),"naive positive extreme:",naive[-1])

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

@jax.custom_jvp
def stable_softplus(x):
    return jnp.logaddexp(0.0,x)

@stable_softplus.defjvp
def softplus_jvp(primals,tangents):
    x,=primals
    dx,=tangents
    return stable_softplus(x),jax.nn.sigmoid(x)*dx

@jax.custom_vjp
def stable_logsumexp(x):
    return jax.scipy.special.logsumexp(x)

def logsumexp_fwd(x):
    value=stable_logsumexp(x)
    weights=jax.nn.softmax(x)
    return value,weights

def logsumexp_bwd(weights,cotangent):
    return (cotangent*weights,)

stable_logsumexp.defvjp(logsumexp_fwd,logsumexp_bwd)

points=jnp.array([-4.,0.,3.])
values=jax.vmap(stable_softplus)(points)
first=jax.vmap(jax.grad(stable_softplus))(points)
second=jax.vmap(jax.grad(jax.grad(stable_softplus)))(points)
reference_first=1/(1+np.exp(-np.asarray(points)))
reference_second=reference_first*(1-reference_first)
np.testing.assert_allclose(values,np.logaddexp(0,np.asarray(points)),rtol=1e-12)
np.testing.assert_allclose(first,reference_first,rtol=1e-12)
np.testing.assert_allclose(second,reference_second,rtol=1e-12)
x=jnp.array([-0.8,0.4,1.1])
weights=np.exp(np.asarray(x)-np.max(np.asarray(x)));weights/=weights.sum()
hessian_reference=np.diag(weights)-np.outer(weights,weights)
np.testing.assert_allclose(jax.grad(stable_logsumexp)(x),weights,rtol=1e-12)
np.testing.assert_allclose(jax.jacrev(jax.grad(stable_logsumexp))(x),hessian_reference,rtol=1e-12,atol=1e-12)
print("Softplus first/second:",np.asarray(first),np.asarray(second))
print("Logsumexp gradient:",weights)

extremes=jnp.array([-1000.,0.,1000.])
extreme_values=jax.vmap(stable_softplus)(extremes)
extreme_gradients=jax.vmap(jax.grad(stable_softplus))(extremes)
assert np.isfinite(extreme_values).all() and np.isfinite(extreme_gradients).all()
np.testing.assert_allclose(extreme_values,[0.,np.log(2.),1000.],atol=1e-12)
with np.errstate(over='ignore'):
    naive=np.log1p(np.exp(np.asarray(extremes)))
assert not np.isfinite(naive[-1])
_,pullback=jax.vjp(stable_logsumexp,x)
np.testing.assert_allclose(pullback(2.0-0.3)[0],2*pullback(1.0)[0]-pullback(0.3)[0],rtol=1e-12)
print("Stable extreme values:",np.asarray(extreme_values),"naive positive extreme:",naive[-1])

# Figure data experiment
grid=jnp.linspace(-8.,8.,65)
slopes=np.asarray(jax.vmap(jax.grad(stable_softplus))(grid))
curvatures=np.asarray(jax.vmap(jax.grad(jax.grad(stable_softplus)))(grid))
visual_data={'panels':[{'kind':'line','x':np.asarray(grid).tolist(),'xlabel':'input x','ylabel':'first derivative','series':[{'label':'custom softplus slope','y':slopes.tolist()}]},{'kind':'line','x':np.asarray(grid).tolist(),'xlabel':'input x','ylabel':'second derivative','series':[{'label':'custom softplus curvature','y':curvatures.tolist()}]}]}

# Experiment: Expose a wrong but linear rule
@jax.custom_jvp
def wrong_softplus(z):
    return jnp.logaddexp(0.0,z)
@wrong_softplus.defjvp
def wrong_rule(primals,tangents):
    z,=primals;dz,=tangents
    return wrong_softplus(z),0.5*jax.nn.sigmoid(z)*dz
probe=0.4;eps=1e-5
finite=(np.logaddexp(0,probe+eps)-np.logaddexp(0,probe-eps))/(2*eps)
wrong=float(jax.grad(wrong_softplus)(probe))
right=float(jax.grad(stable_softplus)(probe))
assert abs(wrong-finite)>0.2
np.testing.assert_allclose(right,finite,rtol=1e-9)
print("Wrong/correct/finite-difference derivative:",wrong,right,finite)

# Experiment: Exercise the custom VJP boundary
rejected=False
try:
    jax.jvp(stable_logsumexp,(x,),(jnp.ones_like(x),))
except TypeError as error:
    rejected=True
    print("Expected custom_vjp forward-mode boundary:",str(error).splitlines()[0])
assert rejected
ordinary=lambda z:jax.scipy.special.logsumexp(z)
ordinary_direction=jax.jvp(ordinary,(x,),(jnp.ones_like(x),))[1]
np.testing.assert_allclose(ordinary_direction,1.0,atol=1e-12)
print("Ordinary JAX common-shift JVP:",float(ordinary_direction))

# Reference solution. Try the exercise before reading this.
changed=jnp.array([-3.,0.2,2.,5.])
host=np.asarray(changed);p=np.exp(host-host.max());p/=p.sum()
actual=jax.grad(stable_logsumexp)(changed)
np.testing.assert_allclose(actual,p,rtol=1e-12)
for direction in (np.array([1.,-.5,.2,0.]),np.array([0.,1.,-1.,.3])):
    eps=1e-5
    reference=lambda z:np.max(z)+np.log(np.exp(z-np.max(z)).sum())
    fd=(reference(host+eps*direction)-reference(host-eps*direction))/(2*eps)
    np.testing.assert_allclose(np.dot(np.asarray(actual),direction),fd,rtol=1e-8,atol=1e-10)
np.testing.assert_allclose(jax.vjp(stable_logsumexp,changed)[1](2.5)[0],2.5*p,rtol=1e-12)
print("Changed-size gradient and cotangent scaling verified")

# Reference practice: Check common-shift invariance of curvature
shift=1000.0
np.testing.assert_allclose(stable_logsumexp(x+shift)-stable_logsumexp(x),shift,rtol=1e-12)
np.testing.assert_allclose(jax.grad(stable_logsumexp)(x+shift),jax.grad(stable_logsumexp)(x),rtol=1e-12)
hess=jax.jacrev(jax.grad(stable_logsumexp))(x+shift)
np.testing.assert_allclose(jnp.sum(hess,axis=1),0.,atol=1e-12)
print("Stable shift invariance and zero common-direction curvature verified")

# Reference practice: Audit a nonlinear seed rule before registering it
coefficient=float(jax.nn.sigmoid(0.4))
bad=lambda tangent:coefficient*tangent*tangent
assert not np.isclose(bad(2.0),2*bad(1.0))
good=lambda tangent:jax.jvp(stable_softplus,(jnp.array(0.4),),(jnp.array(tangent),))[1]
np.testing.assert_allclose(good(2*.7-.3),2*good(.7)-good(.3),rtol=1e-12)
print("Nonlinear seed rule rejected by its contract; repaired JVP is linear")
print("PASS: internals-03")
