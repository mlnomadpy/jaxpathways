"""CPU reference: products, exact custom derivatives, restricted jaxpr transformation."""
# Import jax for this computation.
import jax
# Update state in place with the new values.
jax.config.update("jax_enable_x64", True)
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np

# Register custom derivative rule on `stable_softplus(x)`:
@jax.custom_jvp
# Function `stable_softplus(x)` implementing this stage's computation:
def stable_softplus(x):
    # Return `jnp.logaddexp(0.0, x)` to the caller.
    return jnp.logaddexp(0.0,x)

@stable_softplus.defjvp
# Function `softplus_jvp(primals, tangents)` implementing this stage's computation:
def softplus_jvp(primals,tangents):
    # Compute `x,` as `primals`.
    x,=primals
    # Compute `dx,` as `tangents`.
    dx,=tangents
    # Return `(stable_softplus(x), jax.nn.sigmoid(x) * dx)` to the caller.
    return stable_softplus(x),jax.nn.sigmoid(x)*dx

# Register custom derivative rule on `stable_logsumexp(x)`:
@jax.custom_vjp
# Function `stable_logsumexp(x)` implementing this stage's computation:
def stable_logsumexp(x):
    # Return `jax.scipy.special.logsumexp(x)` to the caller.
    return jax.scipy.special.logsumexp(x)

# Function `logsumexp_fwd(x)` implementing this stage's computation:
def logsumexp_fwd(x):
    # Evaluate numerically stable log-space cross-entropy/likelihood (`value`).
    value=stable_logsumexp(x)
    # Apply nonlinear activation or probability normalization to compute `weights`.
    weights=jax.nn.softmax(x)
    # Return `(value, weights)` to the caller.
    return value,weights

# Function `logsumexp_bwd(weights, cotangent)` implementing this stage's computation:
def logsumexp_bwd(weights,cotangent):
    # Return `(cotangent * weights,)` to the caller.
    return (cotangent*weights,)

# Execute `stable_logsumexp.defvjp(logsumexp_fwd,logsumexp_bwd)`.
stable_logsumexp.defvjp(logsumexp_fwd,logsumexp_bwd)

# Define `analyze(x, direction, weighting)` to evaluate the objective and its automatic derivatives:
def analyze(x,direction,weighting):
    # Guard input contract (`x.shape != (2,) or direction.shape != (2,) or weighting.shape != (2,)`) and fail fast if violated.
    if x.shape!=(2,) or direction.shape!=(2,) or weighting.shape!=(2,):
        raise ValueError("point, direction and weighting must each have shape (2,)")
    # Function `mapping(z)` implementing this stage's computation:
    def mapping(z):
        # Compute `a,b` as `z`.
        a,b=z
        # Return `jnp.array([a * b, jnp.sin(a) + b * b])` to the caller.
        return jnp.array([a*b,jnp.sin(a)+b*b])
    # Compute exact directional derivative / Jacobian / Hessian (`(value, jvp)`).
    value,jvp=jax.jvp(mapping,(x,),(direction,))
    # Compute exact directional derivative / Jacobian / Hessian (`vjp`).
    vjp=jax.vjp(mapping,x)[1](weighting)[0]
    # Compute `objective` as `lambda z:0.5*jnp.vdot(mapping(z),mapping(z))`.
    objective=lambda z:0.5*jnp.vdot(mapping(z),mapping(z))
    # Differentiate the objective to obtain gradients `hvp`.
    hvp=jax.jvp(jax.grad(objective),(x,),(direction,))[1]
    # Return `{'value': value, 'jvp': jvp, 'vjp': vjp, 'hvp': hvp}` to the caller.
    return {"value":value,"jvp":jvp,"vjp":vjp,"hvp":hvp}


# Import required JAX, NumPy, and standard-library modules.
from jax.extend import core

# Function `tiny_jvp(closed, primals, tangents)` implementing this stage's computation:
def tiny_jvp(closed,primals,tangents):
    """Interpret a pure flat jaxpr with floating inputs and explicit tangent rules."""
    # Compute `program` as `closed.jaxpr`.
    program=closed.jaxpr
    # Guard input contract (`program.effects`) and fail fast if violated.
    if program.effects:
        raise NotImplementedError("effects are outside this interpreter")
    # Guard input contract (`len(primals) != len(program.invars) or len(tangents) != len(primals)`) and fail fast if violated.
    if len(primals)!=len(program.invars) or len(tangents)!=len(primals):
        raise ValueError("one primal and tangent per input variable required")
    # Construct dictionary `values` with the structured fields for this stage.
    values={}
    directions={}
    # Function `put(var, value, tangent)` implementing this stage's computation:
    def put(var,value,tangent):
        # Compute `values[var]` as `value`.
        values[var]=value
        directions[var]=tangent
    # Loop over `(var, constant)` in `zip(program.constvars, closed.consts)`:
    for var,constant in zip(program.constvars,closed.consts):
        # Allocate initialized array `` with the specified shape and dtype.
        put(var,constant,jnp.zeros_like(constant))
    # Loop over `(var, value, tangent)` in `zip(program.invars, primals, tangents)`:
    for var,value,tangent in zip(program.invars,primals,tangents):
        # Guard input contract (`value.shape != var.aval.shape or value.dtype != var.aval.dtype`) and fail fast if violated.
        if value.shape!=var.aval.shape or value.dtype!=var.aval.dtype:
            raise ValueError("primal shape/dtype differs from traced signature")
        # Guard input contract (`tangent.shape != value.shape or tangent.dtype != value.dtype`) and fail fast if violated.
        if tangent.shape!=value.shape or tangent.dtype!=value.dtype:
            raise ValueError("tangent shape/dtype must match its primal")
        # Guard input contract (`not jnp.issubdtype(value.dtype, jnp.floating)`) and fail fast if violated.
        if not jnp.issubdtype(value.dtype,jnp.floating):
            raise ValueError("this teaching interpreter requires real floating inputs")
        # Run `put` to perform the next check or state transition.
        put(var,value,tangent)
    # Function `read(atom)` implementing this stage's computation:
    def read(atom):
        # Branch on condition `isinstance(atom, core.Literal)`:
        if isinstance(atom,core.Literal):
            return atom.val,jnp.zeros_like(atom.val)
        # Return `(values[atom], directions[atom])` to the caller.
        return values[atom],directions[atom]
    # Loop over `equation` in `program.eqns`:
    for equation in program.eqns:
        # Compute `name` as `equation.primitive.name`.
        name=equation.primitive.name
        # Guard input contract (`name not in {'add', 'mul', 'neg', 'sin', 'reduce_sum'}`) and fail fast if violated.
        if name not in {"add","mul","neg","sin","reduce_sum"}:
            raise NotImplementedError("unsupported primitive: "+name)
        # Guard input contract (`len(equation.outvars) != 1`) and fail fast if violated.
        if len(equation.outvars)!=1:
            raise NotImplementedError("multiple-result primitives are outside this interpreter")
        # Initialize list `operands` for the stage values.
        operands=[read(atom) for atom in equation.invars]
        # Compute `a,da` as `operands[0]`.
        a,da=operands[0]
        # Branch on condition `name == 'add'`:
        if name=="add":
            b,db=operands[1]
            result,tangent=a+b,da+db
        elif name=="mul":
            b,db=operands[1]
            result,tangent=a*b,da*b+a*db
        elif name=="neg":result,tangent=-a,-da
        elif name=="sin":result,tangent=jnp.sin(a),jnp.cos(a)*da
        else:
            if equation.params.get("out_sharding") is not None:
                raise NotImplementedError("explicit reduction placement is outside this interpreter")
            axes=equation.params["axes"]
            result,tangent=jnp.sum(a,axis=axes),jnp.sum(da,axis=axes)
        # Run `put` to perform the next check or state transition.
        put(equation.outvars[0],result,tangent)
    # Initialize list `output` for the stage values.
    output=[read(var) for var in program.outvars]
    # Return `(tuple((v for v, _ in output)), tuple((t for _, t in output)))` to the caller.
    return tuple(v for v,_ in output),tuple(t for _,t in output)
