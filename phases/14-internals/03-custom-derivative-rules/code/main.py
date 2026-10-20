"""Custom derivative rules: worked experiments and reference solutions. CPU checks."""

# Write stable primal functions and exact derivative contracts
# Step 1 — Write stable primal functions and exact derivative contracts: The JVP rule returns primal and tangent; the VJP forward rule...
import jax
# Update state in place with the new values.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

# Register custom derivative rule on `stable_softplus(x)`:
@jax.custom_jvp
def stable_softplus(x):
    return jnp.logaddexp(0.0,x)

@stable_softplus.defjvp
def softplus_jvp(primals,tangents):
    x,=primals
    dx,=tangents
    return stable_softplus(x),jax.nn.sigmoid(x)*dx

# Register custom derivative rule on `stable_logsumexp(x)`:
@jax.custom_vjp
def stable_logsumexp(x):
    return jax.scipy.special.logsumexp(x)

def logsumexp_fwd(x):
    # Evaluate numerically stable log-space cross-entropy/likelihood (`value`).
    value=stable_logsumexp(x)
    # Apply nonlinear activation or probability normalization to compute `weights`.
    weights=jax.nn.softmax(x)
    return value,weights

def logsumexp_bwd(weights,cotangent):
    return (cotangent*weights,)

# Execute `stable_logsumexp.defvjp(logsumexp_fwd,logsumexp_bwd)`.
stable_logsumexp.defvjp(logsumexp_fwd,logsumexp_bwd)

# Check first and second derivatives independently
# Step 2 — Check first and second derivatives independently: An exact first derivative can still hide mistakes in a...
points=jnp.array([-4.,0.,3.])
# Vectorize across the batch dimension with `jax.vmap` (`values`).
values=jax.vmap(stable_softplus)(points)
# Differentiate the objective to obtain `first` via automatic differentiation.
first=jax.vmap(jax.grad(stable_softplus))(points)
# Differentiate the objective to obtain `second` via automatic differentiation.
second=jax.vmap(jax.grad(jax.grad(stable_softplus)))(points)
# Convert `reference_first` to a host NumPy array for inspection or verification.
reference_first=1/(1+np.exp(-np.asarray(points)))
reference_second=reference_first*(1-reference_first)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(values,np.logaddexp(0,np.asarray(points)),rtol=1e-12)
np.testing.assert_allclose(first,reference_first,rtol=1e-12)
np.testing.assert_allclose(second,reference_second,rtol=1e-12)
x=jnp.array([-0.8,0.4,1.1])
# Convert `weights` to a host NumPy array for inspection or verification.
# Accumulate the next contribution into `weights`.
weights=np.exp(np.asarray(x)-np.max(np.asarray(x)))
weights/=weights.sum()
hessian_reference=np.diag(weights)-np.outer(weights,weights)
# Differentiate the objective to obtain gradients ``.
np.testing.assert_allclose(jax.grad(stable_logsumexp)(x),weights,rtol=1e-12)
# Differentiate the objective to obtain gradients ``.
np.testing.assert_allclose(jax.jacrev(jax.grad(stable_logsumexp))(x),hessian_reference,rtol=1e-12,atol=1e-12)
print("Softplus first/second:",np.asarray(first),np.asarray(second))
print("Logsumexp gradient:",weights)

# Exercise extremes and seed linearity
# Step 3 — Exercise extremes and seed linearity: Numerical stability must hold in both the primal and derivative path.
extremes=jnp.array([-1000.,0.,1000.])
# Vectorize across the batch dimension with `jax.vmap` (`extreme_values`).
extreme_values=jax.vmap(stable_softplus)(extremes)
# Differentiate the objective to obtain `extreme_gradients` via automatic differentiation.
extreme_gradients=jax.vmap(jax.grad(stable_softplus))(extremes)
# Confirm that all computed values remain finite (no NaN or Inf).
assert np.isfinite(extreme_values).all() and np.isfinite(extreme_gradients).all()
np.testing.assert_allclose(extreme_values,[0.,np.log(2.),1000.],atol=1e-12)
# Enter `np.errstate(over='ignore')` context block:
with np.errstate(over='ignore'):
    # Convert `naive` to a host NumPy array for inspection or verification.
    naive=np.log1p(np.exp(np.asarray(extremes)))
# Confirm that all computed values remain finite (no NaN or Inf).
assert not np.isfinite(naive[-1])
# Compute exact directional derivative / Jacobian / Hessian (`(_, pullback)`).
_,pullback=jax.vjp(stable_logsumexp,x)
np.testing.assert_allclose(pullback(2.0-0.3)[0],2*pullback(1.0)[0]-pullback(0.3)[0],rtol=1e-12)
print("Stable extreme values:",np.asarray(extreme_values),"naive positive extreme:",naive[-1])

# Step 1 — Write stable primal functions and exact derivative contracts: The JVP rule returns primal and tangent; the VJP forward rule...
import jax
# Update state in place with the new values.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

# Register custom derivative rule on `stable_softplus(x)`:
@jax.custom_jvp
def stable_softplus(x):
    return jnp.logaddexp(0.0,x)

@stable_softplus.defjvp
def softplus_jvp(primals,tangents):
    x,=primals
    dx,=tangents
    return stable_softplus(x),jax.nn.sigmoid(x)*dx

# Register custom derivative rule on `stable_logsumexp(x)`:
@jax.custom_vjp
def stable_logsumexp(x):
    return jax.scipy.special.logsumexp(x)

def logsumexp_fwd(x):
    # Evaluate numerically stable log-space cross-entropy/likelihood (`value`).
    value=stable_logsumexp(x)
    # Apply nonlinear activation or probability normalization to compute `weights`.
    weights=jax.nn.softmax(x)
    return value,weights

def logsumexp_bwd(weights,cotangent):
    return (cotangent*weights,)

# Execute `stable_logsumexp.defvjp(logsumexp_fwd,logsumexp_bwd)`.
stable_logsumexp.defvjp(logsumexp_fwd,logsumexp_bwd)

# Step 2 — Check first and second derivatives independently: An exact first derivative can still hide mistakes in a...
points=jnp.array([-4.,0.,3.])
# Vectorize across the batch dimension with `jax.vmap` (`values`).
values=jax.vmap(stable_softplus)(points)
# Differentiate the objective to obtain `first` via automatic differentiation.
first=jax.vmap(jax.grad(stable_softplus))(points)
# Differentiate the objective to obtain `second` via automatic differentiation.
second=jax.vmap(jax.grad(jax.grad(stable_softplus)))(points)
# Convert `reference_first` to a host NumPy array for inspection or verification.
reference_first=1/(1+np.exp(-np.asarray(points)))
reference_second=reference_first*(1-reference_first)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(values,np.logaddexp(0,np.asarray(points)),rtol=1e-12)
np.testing.assert_allclose(first,reference_first,rtol=1e-12)
np.testing.assert_allclose(second,reference_second,rtol=1e-12)
x=jnp.array([-0.8,0.4,1.1])
# Convert `weights` to a host NumPy array for inspection or verification.
# Accumulate the next contribution into `weights`.
weights=np.exp(np.asarray(x)-np.max(np.asarray(x)))
weights/=weights.sum()
hessian_reference=np.diag(weights)-np.outer(weights,weights)
# Differentiate the objective to obtain gradients ``.
np.testing.assert_allclose(jax.grad(stable_logsumexp)(x),weights,rtol=1e-12)
# Differentiate the objective to obtain gradients ``.
np.testing.assert_allclose(jax.jacrev(jax.grad(stable_logsumexp))(x),hessian_reference,rtol=1e-12,atol=1e-12)
print("Softplus first/second:",np.asarray(first),np.asarray(second))
print("Logsumexp gradient:",weights)

# Step 3 — Exercise extremes and seed linearity: Numerical stability must hold in both the primal and derivative path.
extremes=jnp.array([-1000.,0.,1000.])
# Vectorize across the batch dimension with `jax.vmap` (`extreme_values`).
extreme_values=jax.vmap(stable_softplus)(extremes)
# Differentiate the objective to obtain `extreme_gradients` via automatic differentiation.
extreme_gradients=jax.vmap(jax.grad(stable_softplus))(extremes)
# Confirm that all computed values remain finite (no NaN or Inf).
assert np.isfinite(extreme_values).all() and np.isfinite(extreme_gradients).all()
np.testing.assert_allclose(extreme_values,[0.,np.log(2.),1000.],atol=1e-12)
# Enter `np.errstate(over='ignore')` context block:
with np.errstate(over='ignore'):
    # Convert `naive` to a host NumPy array for inspection or verification.
    naive=np.log1p(np.exp(np.asarray(extremes)))
# Confirm that all computed values remain finite (no NaN or Inf).
assert not np.isfinite(naive[-1])
# Compute exact directional derivative / Jacobian / Hessian (`(_, pullback)`).
_,pullback=jax.vjp(stable_logsumexp,x)
np.testing.assert_allclose(pullback(2.0-0.3)[0],2*pullback(1.0)[0]-pullback(0.3)[0],rtol=1e-12)
print("Stable extreme values:",np.asarray(extreme_values),"naive positive extreme:",naive[-1])

# Figure data experiment
# Compute figure data for: A stable custom rule preserves both slope and curvature
# Generate a uniform grid of points in `grid`.
grid=jnp.linspace(-8.,8.,65)
# Convert `slopes` to a host NumPy array for inspection or verification.
slopes=np.asarray(jax.vmap(jax.grad(stable_softplus))(grid))
# Convert `curvatures` to a host NumPy array for inspection or verification.
curvatures=np.asarray(jax.vmap(jax.grad(jax.grad(stable_softplus)))(grid))
# Convert `visual_data` to a host NumPy array for inspection or verification.
visual_data={'panels':[{'kind':'line','x':np.asarray(grid).tolist(),'xlabel':'input x','ylabel':'first derivative','series':[{'label':'custom softplus slope','y':slopes.tolist()}]},{'kind':'line','x':np.asarray(grid).tolist(),'xlabel':'input x','ylabel':'second derivative','series':[{'label':'custom softplus curvature','y':curvatures.tolist()}]}]}

# Experiment: Expose a wrong but linear rule
# Experiment — Expose a wrong but linear rule: The wrong tangent expression is linear, so transposition can...
# Register custom derivative rule on `wrong_softplus(z)`:
@jax.custom_jvp
def wrong_softplus(z):
    return jnp.logaddexp(0.0,z)
@wrong_softplus.defjvp
def wrong_rule(primals,tangents):
    z,=primals
    dz,=tangents
    return wrong_softplus(z),0.5*jax.nn.sigmoid(z)*dz
probe=0.4
eps=1e-5
finite=(np.logaddexp(0,probe+eps)-np.logaddexp(0,probe-eps))/(2*eps)
# Differentiate the objective to obtain `wrong` via automatic differentiation.
wrong=float(jax.grad(wrong_softplus)(probe))
# Differentiate the objective to obtain `right` via automatic differentiation.
right=float(jax.grad(stable_softplus)(probe))
assert abs(wrong-finite)>0.2
np.testing.assert_allclose(right,finite,rtol=1e-9)
print("Wrong/correct/finite-difference derivative:",wrong,right,finite)

# Experiment: Exercise the custom VJP boundary
# Experiment — Exercise the custom VJP boundary: Adding the same constant to all logits shifts log-sum-exp by...
rejected=False
# Run the boundary check and catch the expected exception:
try:
    jax.jvp(stable_logsumexp,(x,),(jnp.ones_like(x),))
except TypeError as error:
    rejected=True
    print("Expected custom_vjp forward-mode boundary:",str(error).splitlines()[0])
assert rejected
# Evaluate numerically stable log-space cross-entropy/likelihood (`ordinary`).
ordinary=lambda z:jax.scipy.special.logsumexp(z)
# Compute exact directional derivative / Jacobian / Hessian (`ordinary_direction`).
ordinary_direction=jax.jvp(ordinary,(x,),(jnp.ones_like(x),))[1]
np.testing.assert_allclose(ordinary_direction,1.0,atol=1e-12)
print("Ordinary JAX common-shift JVP:",float(ordinary_direction))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Check the custom log-sum-exp gradient on (-3,0.2,2,5) against an...
changed=jnp.array([-3.,0.2,2.,5.])
# Convert `host` to a host NumPy array for inspection or verification.
# Accumulate the next contribution into `p`.
host=np.asarray(changed)
p=np.exp(host-host.max())
p/=p.sum()
# Differentiate the objective to obtain `actual` via automatic differentiation.
actual=jax.grad(stable_logsumexp)(changed)
np.testing.assert_allclose(actual,p,rtol=1e-12)
for direction in (np.array([1.,-.5,.2,0.]),np.array([0.,1.,-1.,.3])):
    eps=1e-5
    # Aggregate array values to compute `reference`.
    reference=lambda z:np.max(z)+np.log(np.exp(z-np.max(z)).sum())
    fd=(reference(host+eps*direction)-reference(host-eps*direction))/(2*eps)
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_allclose(np.dot(np.asarray(actual),direction),fd,rtol=1e-8,atol=1e-10)
np.testing.assert_allclose(jax.vjp(stable_logsumexp,changed)[1](2.5)[0],2.5*p,rtol=1e-12)
print("Changed-size gradient and cotangent scaling verified")

# Reference practice: Check common-shift invariance of curvature
# Check common-shift invariance of curvature (Practice): The common shift leaves probabilities unchanged.
shift=1000.0
np.testing.assert_allclose(stable_logsumexp(x+shift)-stable_logsumexp(x),shift,rtol=1e-12)
# Differentiate the objective to obtain gradients ``.
np.testing.assert_allclose(jax.grad(stable_logsumexp)(x+shift),jax.grad(stable_logsumexp)(x),rtol=1e-12)
# Differentiate the objective to obtain `hess` via automatic differentiation.
hess=jax.jacrev(jax.grad(stable_logsumexp))(x+shift)
np.testing.assert_allclose(jnp.sum(hess,axis=1),0.,atol=1e-12)
print("Stable shift invariance and zero common-direction curvature verified")

# Reference practice: Audit a nonlinear seed rule before registering it
# Audit a nonlinear seed rule before registering it (Challenge): Linearity follows from the definition of a derivative at a...
coefficient=float(jax.nn.sigmoid(0.4))
bad=lambda tangent:coefficient*tangent*tangent
assert not np.isclose(bad(2.0),2*bad(1.0))
# Compute exact directional derivative / Jacobian / Hessian (`good`).
good=lambda tangent:jax.jvp(stable_softplus,(jnp.array(0.4),),(jnp.array(tangent),))[1]
np.testing.assert_allclose(good(2*.7-.3),2*good(.7)-good(.3),rtol=1e-12)
print("Nonlinear seed rule rejected by its contract; repaired JVP is linear")
print("PASS: internals-03")
