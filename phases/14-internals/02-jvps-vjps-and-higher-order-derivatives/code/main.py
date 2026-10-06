"""JVPs, VJPs, and higher-order derivatives: worked experiments and reference solutions. CPU checks."""

# Define a small vector map and derive its Jacobian
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

def mapping(x):
    a,b=x
    return jnp.array([a*b,jnp.sin(a)+b*b])
x=jnp.array([0.4,-0.7])
v=jnp.array([1.2,-0.3])
u=jnp.array([0.5,2.0])
a,b=np.asarray(x)
jacobian=np.array([[b,a],[np.cos(a),2*b]])
expected_value=np.array([a*b,np.sin(a)+b*b])

# Push a direction forward and pull a measurement backward
value,jvp=jax.jvp(mapping,(x,),(v,))
value_again,pullback=jax.vjp(mapping,x)
vjp,=pullback(u)
np.testing.assert_allclose(value,expected_value,rtol=1e-12)
np.testing.assert_allclose(jvp,jacobian@np.asarray(v),rtol=1e-12)
np.testing.assert_allclose(vjp,jacobian.T@np.asarray(u),rtol=1e-12)
left=float(jnp.vdot(u,jvp));right=float(jnp.vdot(vjp,v))
np.testing.assert_allclose(left,right,rtol=1e-12)
print("Jacobian:",jacobian)
print("JVP:",np.asarray(jvp),"VJP:",np.asarray(vjp),"adjoint pair:",left,right)

# Compute curvature without materializing a Hessian
def objective(z):
    return 0.5*jnp.vdot(mapping(z),mapping(z))
gradient=jax.grad(objective)(x)
hvp=jax.jvp(jax.grad(objective),(x,),(v,))[1]
y=expected_value
hessian_0=np.array([[0.,1.],[1.,0.]])
hessian_1=np.array([[-np.sin(a),0.],[0.,2.]])
hessian=jacobian.T@jacobian+y[0]*hessian_0+y[1]*hessian_1
np.testing.assert_allclose(gradient,jacobian.T@y,rtol=1e-12)
np.testing.assert_allclose(hvp,hessian@np.asarray(v),rtol=1e-12)
np.testing.assert_allclose(hessian,hessian.T,atol=1e-12)
print("Hessian-vector product:",np.asarray(hvp))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

def mapping(x):
    a,b=x
    return jnp.array([a*b,jnp.sin(a)+b*b])
x=jnp.array([0.4,-0.7])
v=jnp.array([1.2,-0.3])
u=jnp.array([0.5,2.0])
a,b=np.asarray(x)
jacobian=np.array([[b,a],[np.cos(a),2*b]])
expected_value=np.array([a*b,np.sin(a)+b*b])

value,jvp=jax.jvp(mapping,(x,),(v,))
value_again,pullback=jax.vjp(mapping,x)
vjp,=pullback(u)
np.testing.assert_allclose(value,expected_value,rtol=1e-12)
np.testing.assert_allclose(jvp,jacobian@np.asarray(v),rtol=1e-12)
np.testing.assert_allclose(vjp,jacobian.T@np.asarray(u),rtol=1e-12)
left=float(jnp.vdot(u,jvp));right=float(jnp.vdot(vjp,v))
np.testing.assert_allclose(left,right,rtol=1e-12)
print("Jacobian:",jacobian)
print("JVP:",np.asarray(jvp),"VJP:",np.asarray(vjp),"adjoint pair:",left,right)

def objective(z):
    return 0.5*jnp.vdot(mapping(z),mapping(z))
gradient=jax.grad(objective)(x)
hvp=jax.jvp(jax.grad(objective),(x,),(v,))[1]
y=expected_value
hessian_0=np.array([[0.,1.],[1.,0.]])
hessian_1=np.array([[-np.sin(a),0.],[0.,2.]])
hessian=jacobian.T@jacobian+y[0]*hessian_0+y[1]*hessian_1
np.testing.assert_allclose(gradient,jacobian.T@y,rtol=1e-12)
np.testing.assert_allclose(hvp,hessian@np.asarray(v),rtol=1e-12)
np.testing.assert_allclose(hessian,hessian.T,atol=1e-12)
print("Hessian-vector product:",np.asarray(hvp))

# Figure data experiment
visual_data={'panels':[{'kind':'bar','labels':['output 0','output 1'],'ylabel':'directional output change','series':[{'label':'JAX JVP','y':np.asarray(jvp).tolist()},{'label':'analytic J v','y':(jacobian@np.asarray(v)).tolist()}]},{'kind':'bar','labels':['input 0','input 1'],'ylabel':'input cotangent','series':[{'label':'JAX VJP','y':np.asarray(vjp).tolist()},{'label':'analytic transpose J u','y':(jacobian.T@np.asarray(u)).tolist()}]}]}

# Experiment: Reconstruct columns and rows independently
basis=jnp.eye(2)
columns=jax.vmap(lambda direction:jax.jvp(mapping,(x,),(direction,))[1])(basis).T
rows=jax.vmap(lambda weighting:pullback(weighting)[0])(basis)
np.testing.assert_allclose(columns,jacobian,rtol=1e-12)
np.testing.assert_allclose(rows,jacobian,rtol=1e-12)
print("Forward columns and reverse rows agree")

# Experiment: Separate exact curvature from Gauss–Newton
gauss_newton=jacobian.T@jacobian
approximation=gauss_newton@np.asarray(v)
assert np.linalg.norm(np.asarray(hvp)-approximation)>0.1
eps=1e-4
fd=(np.asarray(jax.grad(objective)(x+eps*v))-np.asarray(jax.grad(objective)(x-eps*v)))/(2*eps)
np.testing.assert_allclose(hvp,fd,rtol=1e-7,atol=1e-8)
print("Exact HVP:",np.asarray(hvp),"Gauss-Newton product:",approximation)

# Reference solution. Try the exercise before reading this.
changed_x=jnp.array([-0.2,0.8]);changed_v=jnp.array([-0.5,1.3]);changed_u=jnp.array([1.1,-0.4])
a2,b2=np.asarray(changed_x)
j2=np.array([[b2,a2],[np.cos(a2),2*b2]])
fj=jax.jvp(mapping,(changed_x,),(changed_v,))[1]
rj=jax.vjp(mapping,changed_x)[1](changed_u)[0]
np.testing.assert_allclose(fj,j2@changed_v,rtol=1e-12)
np.testing.assert_allclose(rj,j2.T@changed_u,rtol=1e-12)
np.testing.assert_allclose(jnp.vdot(changed_u,fj),jnp.vdot(rj,changed_v),rtol=1e-12)
print("Changed-point adjoint verified")

# Reference practice: Test linearity at a fixed point
w=jnp.array([-0.6,0.9])
product=lambda direction:jax.jvp(mapping,(x,),(direction,))[1]
np.testing.assert_allclose(product(2*v-w),2*product(v)-product(w),rtol=1e-12)
shifted=jax.jvp(mapping,(x+w,),(v,))[1]
assert np.linalg.norm(np.asarray(shifted-product(v)))>0.1
print("Direction linearity holds; the Jacobian changes with the base point")

# Reference practice: Check curvature symmetry without a full Hessian
w=jnp.array([0.2,0.9])
hw=jax.jvp(jax.grad(objective),(x,),(w,))[1]
np.testing.assert_allclose(jnp.vdot(w,hvp),jnp.vdot(v,hw),rtol=1e-12)
eps=1e-4
fdw=(jax.grad(objective)(x+eps*w)-jax.grad(objective)(x-eps*w))/(2*eps)
np.testing.assert_allclose(hw,fdw,rtol=1e-7,atol=1e-8)
print("Bilinear Hessian symmetry and finite-difference product verified")
print("PASS: internals-02")
