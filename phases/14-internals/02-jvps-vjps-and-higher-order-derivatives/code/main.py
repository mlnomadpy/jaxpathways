"""JVPs, VJPs, and higher-order derivatives: worked experiments and reference solutions. CPU checks."""

# Define a small vector map and derive its Jacobian
# Step 1 — Define a small vector map and derive its Jacobian: The map has two inputs and two outputs.
# Import jax for this computation.
import jax
# Update state in place with the new values.
jax.config.update("jax_enable_x64", True)
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np

# Function `mapping(x)` implementing this stage's computation:
def mapping(x):
    # Evaluate `(a, b)` from the current inputs and state.
    a,b=x
    # Return `jnp.array([a * b, jnp.sin(a) + b * b])` to the caller.
    return jnp.array([a*b,jnp.sin(a)+b*b])
# Initialize array `x` with explicit values and shape.
x=jnp.array([0.4,-0.7])
# Initialize array `v` with explicit values and shape.
v=jnp.array([1.2,-0.3])
# Initialize array `u` with explicit values and shape.
u=jnp.array([0.5,2.0])
# Convert `(a, b)` to a host NumPy array for inspection or verification.
a,b=np.asarray(x)
# Initialize array `jacobian` with explicit values and shape.
jacobian=np.array([[b,a],[np.cos(a),2*b]])
# Initialize array `expected_value` with explicit values and shape.
expected_value=np.array([a*b,np.sin(a)+b*b])

# Push a direction forward and pull a measurement backward
# Step 2 — Push a direction forward and pull a measurement backward: The VJP returns one cotangent per input argument, hence the...
# Compute exact directional derivative / Jacobian / Hessian (`(value, jvp)`).
value,jvp=jax.jvp(mapping,(x,),(v,))
# Compute exact directional derivative / Jacobian / Hessian (`(value_again, pullback)`).
value_again,pullback=jax.vjp(mapping,x)
# Run `pullback` to compute `(vjp,)`.
vjp,=pullback(u)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(value,expected_value,rtol=1e-12)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(jvp,jacobian@np.asarray(v),rtol=1e-12)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(vjp,jacobian.T@np.asarray(u),rtol=1e-12)
# Evaluate `jnp.vdot(u, jvp)` and convert the result into Python scalar/collection `left`.
# Evaluate `jnp.vdot(vjp, v)` and convert the result into Python scalar/collection `right`.
left=float(jnp.vdot(u,jvp));right=float(jnp.vdot(vjp,v))
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(left,right,rtol=1e-12)
# Print the observed values to compare against the expected result.
print("Jacobian:",jacobian)
# Print diagnostic summary of the computed outputs.
print("JVP:",np.asarray(jvp),"VJP:",np.asarray(vjp),"adjoint pair:",left,right)

# Compute curvature without materializing a Hessian
# Step 3 — Compute curvature without materializing a Hessian: For squared residuals, the Hessian includes a...
def objective(z):
    # Return `0.5 * jnp.vdot(mapping(z), mapping(z))` to the caller.
    return 0.5*jnp.vdot(mapping(z),mapping(z))
# Differentiate the objective to obtain `gradient` via automatic differentiation.
gradient=jax.grad(objective)(x)
# Differentiate the objective to obtain `hvp` via automatic differentiation.
hvp=jax.jvp(jax.grad(objective),(x,),(v,))[1]
# Evaluate `y` from the current inputs and state.
y=expected_value
# Initialize array `hessian_0` with explicit values and shape.
hessian_0=np.array([[0.,1.],[1.,0.]])
# Initialize array `hessian_1` with explicit values and shape.
hessian_1=np.array([[-np.sin(a),0.],[0.,2.]])
# Perform matrix contraction / projection to compute `hessian`.
hessian=jacobian.T@jacobian+y[0]*hessian_0+y[1]*hessian_1
# Perform matrix contraction / projection to compute ``.
np.testing.assert_allclose(gradient,jacobian.T@y,rtol=1e-12)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(hvp,hessian@np.asarray(v),rtol=1e-12)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(hessian,hessian.T,atol=1e-12)
# Print diagnostic summary of the computed outputs.
print("Hessian-vector product:",np.asarray(hvp))

# Step 1 — Define a small vector map and derive its Jacobian: The map has two inputs and two outputs.
# Import jax for this computation.
import jax
# Update state in place with the new values.
jax.config.update("jax_enable_x64", True)
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np

# Function `mapping(x)` implementing this stage's computation:
def mapping(x):
    # Evaluate `(a, b)` from the current inputs and state.
    a,b=x
    # Return `jnp.array([a * b, jnp.sin(a) + b * b])` to the caller.
    return jnp.array([a*b,jnp.sin(a)+b*b])
# Initialize array `x` with explicit values and shape.
x=jnp.array([0.4,-0.7])
# Initialize array `v` with explicit values and shape.
v=jnp.array([1.2,-0.3])
# Initialize array `u` with explicit values and shape.
u=jnp.array([0.5,2.0])
# Convert `(a, b)` to a host NumPy array for inspection or verification.
a,b=np.asarray(x)
# Initialize array `jacobian` with explicit values and shape.
jacobian=np.array([[b,a],[np.cos(a),2*b]])
# Initialize array `expected_value` with explicit values and shape.
expected_value=np.array([a*b,np.sin(a)+b*b])

# Step 2 — Push a direction forward and pull a measurement backward: The VJP returns one cotangent per input argument, hence the...
# Compute exact directional derivative / Jacobian / Hessian (`(value, jvp)`).
value,jvp=jax.jvp(mapping,(x,),(v,))
# Compute exact directional derivative / Jacobian / Hessian (`(value_again, pullback)`).
value_again,pullback=jax.vjp(mapping,x)
# Run `pullback` to compute `(vjp,)`.
vjp,=pullback(u)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(value,expected_value,rtol=1e-12)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(jvp,jacobian@np.asarray(v),rtol=1e-12)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(vjp,jacobian.T@np.asarray(u),rtol=1e-12)
# Evaluate `jnp.vdot(u, jvp)` and convert the result into Python scalar/collection `left`.
# Evaluate `jnp.vdot(vjp, v)` and convert the result into Python scalar/collection `right`.
left=float(jnp.vdot(u,jvp));right=float(jnp.vdot(vjp,v))
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(left,right,rtol=1e-12)
# Print the observed values to compare against the expected result.
print("Jacobian:",jacobian)
# Print diagnostic summary of the computed outputs.
print("JVP:",np.asarray(jvp),"VJP:",np.asarray(vjp),"adjoint pair:",left,right)

# Step 3 — Compute curvature without materializing a Hessian: For squared residuals, the Hessian includes a...
def objective(z):
    # Return `0.5 * jnp.vdot(mapping(z), mapping(z))` to the caller.
    return 0.5*jnp.vdot(mapping(z),mapping(z))
# Differentiate the objective to obtain `gradient` via automatic differentiation.
gradient=jax.grad(objective)(x)
# Differentiate the objective to obtain `hvp` via automatic differentiation.
hvp=jax.jvp(jax.grad(objective),(x,),(v,))[1]
# Evaluate `y` from the current inputs and state.
y=expected_value
# Initialize array `hessian_0` with explicit values and shape.
hessian_0=np.array([[0.,1.],[1.,0.]])
# Initialize array `hessian_1` with explicit values and shape.
hessian_1=np.array([[-np.sin(a),0.],[0.,2.]])
# Perform matrix contraction / projection to compute `hessian`.
hessian=jacobian.T@jacobian+y[0]*hessian_0+y[1]*hessian_1
# Perform matrix contraction / projection to compute ``.
np.testing.assert_allclose(gradient,jacobian.T@y,rtol=1e-12)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(hvp,hessian@np.asarray(v),rtol=1e-12)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(hessian,hessian.T,atol=1e-12)
# Print diagnostic summary of the computed outputs.
print("Hessian-vector product:",np.asarray(hvp))

# Figure data experiment
# Compute figure data for: Forward and reverse products live in different spaces
# Convert `visual_data` to a host NumPy array for inspection or verification.
visual_data={'panels':[{'kind':'bar','labels':['output 0','output 1'],'ylabel':'directional output change','series':[{'label':'JAX JVP','y':np.asarray(jvp).tolist()},{'label':'analytic J v','y':(jacobian@np.asarray(v)).tolist()}]},{'kind':'bar','labels':['input 0','input 1'],'ylabel':'input cotangent','series':[{'label':'JAX VJP','y':np.asarray(vjp).tolist()},{'label':'analytic transpose J u','y':(jacobian.T@np.asarray(u)).tolist()}]}]}

# Experiment: Reconstruct columns and rows independently
# Experiment — Reconstruct columns and rows independently: The transpose in the forward construction is necessary: vmap...
# Initialize array `basis` with explicit values and shape.
basis=jnp.eye(2)
# Compute exact directional derivative / Jacobian / Hessian (`columns`).
columns=jax.vmap(lambda direction:jax.jvp(mapping,(x,),(direction,))[1])(basis).T
# Vectorize across the batch dimension with `jax.vmap` (`rows`).
rows=jax.vmap(lambda weighting:pullback(weighting)[0])(basis)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(columns,jacobian,rtol=1e-12)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(rows,jacobian,rtol=1e-12)
# Print the observed values to compare against the expected result.
print("Forward columns and reverse rows agree")

# Experiment: Separate exact curvature from Gauss–Newton
# Experiment — Separate exact curvature from Gauss–Newton: Residuals are nonzero, and both output functions have second...
# Perform matrix / vector contraction (`@`) to compute `gauss_newton`.
gauss_newton=jacobian.T@jacobian
# Perform matrix / vector contraction (`@`) to compute `approximation`.
approximation=gauss_newton@np.asarray(v)
# Verify contract: `np.linalg.norm(np.asarray(hvp) - approximation) > 0.1`.
assert np.linalg.norm(np.asarray(hvp)-approximation)>0.1
# Evaluate `eps` from the current inputs and state.
eps=1e-4
# Differentiate the objective to obtain `fd` via automatic differentiation.
fd=(np.asarray(jax.grad(objective)(x+eps*v))-np.asarray(jax.grad(objective)(x-eps*v)))/(2*eps)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(hvp,fd,rtol=1e-7,atol=1e-8)
# Print the observed values to compare against the expected result.
print("Exact HVP:",np.asarray(hvp),"Gauss-Newton product:",approximation)

# Reference solution. Try the exercise before reading this.
# Exercise solution: Repeat the Jacobian, JVP, VJP and adjoint checks at x=(-0.2,0.8) with...
# Initialize array `changed_x` with explicit values and shape.
changed_x=jnp.array([-0.2,0.8]);changed_v=jnp.array([-0.5,1.3]);changed_u=jnp.array([1.1,-0.4])
# Convert `(a2, b2)` to a host NumPy array for inspection or verification.
a2,b2=np.asarray(changed_x)
# Initialize array `j2` with explicit values and shape.
j2=np.array([[b2,a2],[np.cos(a2),2*b2]])
# Compute exact directional derivative / Jacobian / Hessian (`fj`).
fj=jax.jvp(mapping,(changed_x,),(changed_v,))[1]
# Compute exact directional derivative / Jacobian / Hessian (`rj`).
rj=jax.vjp(mapping,changed_x)[1](changed_u)[0]
# Perform matrix contraction / projection to compute ``.
np.testing.assert_allclose(fj,j2@changed_v,rtol=1e-12)
# Perform matrix contraction / projection to compute ``.
np.testing.assert_allclose(rj,j2.T@changed_u,rtol=1e-12)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(jnp.vdot(changed_u,fj),jnp.vdot(rj,changed_v),rtol=1e-12)
# Print the observed values to compare against the expected result.
print("Changed-point adjoint verified")

# Reference practice: Test linearity at a fixed point
# Test linearity at a fixed point (Practice): Linearity in the tangent is a local property.
# Initialize array `w` with explicit values and shape.
w=jnp.array([-0.6,0.9])
# Compute exact directional derivative / Jacobian / Hessian (`product`).
product=lambda direction:jax.jvp(mapping,(x,),(direction,))[1]
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(product(2*v-w),2*product(v)-product(w),rtol=1e-12)
# Compute exact directional derivative / Jacobian / Hessian (`shifted`).
shifted=jax.jvp(mapping,(x+w,),(v,))[1]
# Verify contract: `np.linalg.norm(np.asarray(shifted - product(v))) > 0.1`.
assert np.linalg.norm(np.asarray(shifted-product(v)))>0.1
# Print the observed values to compare against the expected result.
print("Direction linearity holds; the Jacobian changes with the base point")

# Reference practice: Check curvature symmetry without a full Hessian
# Check curvature symmetry without a full Hessian (Challenge): For this smooth scalar objective, mixed partials agree.
# Initialize array `w` with explicit values and shape.
w=jnp.array([0.2,0.9])
# Differentiate the objective to obtain `hw` via automatic differentiation.
hw=jax.jvp(jax.grad(objective),(x,),(w,))[1]
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(jnp.vdot(w,hvp),jnp.vdot(v,hw),rtol=1e-12)
# Evaluate `eps` from the current inputs and state.
eps=1e-4
# Differentiate the objective to obtain `fdw` via automatic differentiation.
fdw=(jax.grad(objective)(x+eps*w)-jax.grad(objective)(x-eps*w))/(2*eps)
# Verify that computed values match the expected reference within numerical tolerance.
np.testing.assert_allclose(hw,fdw,rtol=1e-7,atol=1e-8)
# Print the observed values to compare against the expected result.
print("Bilinear Hessian symmetry and finite-difference product verified")
print("PASS: internals-02")
