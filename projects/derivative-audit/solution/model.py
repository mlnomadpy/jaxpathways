"""CPU reference: products, exact custom derivatives, restricted jaxpr transformation."""
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

def analyze(x,direction,weighting):
    if x.shape!=(2,) or direction.shape!=(2,) or weighting.shape!=(2,):
        raise ValueError("point, direction and weighting must each have shape (2,)")
    def mapping(z):
        a,b=z
        return jnp.array([a*b,jnp.sin(a)+b*b])
    value,jvp=jax.jvp(mapping,(x,),(direction,))
    vjp=jax.vjp(mapping,x)[1](weighting)[0]
    objective=lambda z:0.5*jnp.vdot(mapping(z),mapping(z))
    hvp=jax.jvp(jax.grad(objective),(x,),(direction,))[1]
    return {"value":value,"jvp":jvp,"vjp":vjp,"hvp":hvp}


from jax.extend import core

def tiny_jvp(closed,primals,tangents):
    """Interpret a pure flat jaxpr with floating inputs and explicit tangent rules."""
    program=closed.jaxpr
    if program.effects:
        raise NotImplementedError("effects are outside this interpreter")
    if len(primals)!=len(program.invars) or len(tangents)!=len(primals):
        raise ValueError("one primal and tangent per input variable required")
    values={};directions={}
    def put(var,value,tangent):
        values[var]=value;directions[var]=tangent
    for var,constant in zip(program.constvars,closed.consts):
        put(var,constant,jnp.zeros_like(constant))
    for var,value,tangent in zip(program.invars,primals,tangents):
        if value.shape!=var.aval.shape or value.dtype!=var.aval.dtype:
            raise ValueError("primal shape/dtype differs from traced signature")
        if tangent.shape!=value.shape or tangent.dtype!=value.dtype:
            raise ValueError("tangent shape/dtype must match its primal")
        if not jnp.issubdtype(value.dtype,jnp.floating):
            raise ValueError("this teaching interpreter requires real floating inputs")
        put(var,value,tangent)
    def read(atom):
        if isinstance(atom,core.Literal):
            return atom.val,jnp.zeros_like(atom.val)
        return values[atom],directions[atom]
    for equation in program.eqns:
        name=equation.primitive.name
        if name not in {"add","mul","neg","sin","reduce_sum"}:
            raise NotImplementedError("unsupported primitive: "+name)
        if len(equation.outvars)!=1:
            raise NotImplementedError("multiple-result primitives are outside this interpreter")
        operands=[read(atom) for atom in equation.invars]
        a,da=operands[0]
        if name=="add":
            b,db=operands[1];result,tangent=a+b,da+db
        elif name=="mul":
            b,db=operands[1];result,tangent=a*b,da*b+a*db
        elif name=="neg":result,tangent=-a,-da
        elif name=="sin":result,tangent=jnp.sin(a),jnp.cos(a)*da
        else:
            if equation.params.get("out_sharding") is not None:
                raise NotImplementedError("explicit reduction placement is outside this interpreter")
            axes=equation.params["axes"]
            result,tangent=jnp.sum(a,axis=axes),jnp.sum(da,axis=axes)
        put(equation.outvars[0],result,tangent)
    output=[read(var) for var in program.outvars]
    return tuple(v for v,_ in output),tuple(t for _,t in output)
