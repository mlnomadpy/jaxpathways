"""Lowering, compilation, and a tiny transformation: worked experiments and reference solutions. CPU checks."""

# Define a deliberately small derivative interpreter
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

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

# Trace a program and verify the transformed result
offset=jnp.array([0.1,0.2,0.3])
def program(x):
    return jnp.sum(jnp.sin(x)*x+offset)
x=jnp.array([-0.8,0.4,1.1]);direction=jnp.array([0.3,-0.2,0.7])
closed=jax.make_jaxpr(program)(x)
primal_outputs,tangent_outputs=tiny_jvp(closed,(x,),(direction,))
reference_value=np.sum(np.sin(np.asarray(x))*np.asarray(x)+np.asarray(offset))
reference_tangent=np.dot(np.sin(np.asarray(x))+np.asarray(x)*np.cos(np.asarray(x)),np.asarray(direction))
np.testing.assert_allclose(primal_outputs[0],reference_value,rtol=1e-12)
np.testing.assert_allclose(tangent_outputs[0],reference_tangent,rtol=1e-12)
np.testing.assert_allclose(tangent_outputs[0],jax.jvp(program,(x,),(direction,))[1],rtol=1e-12)
print("Supported primitives:",[e.primitive.name for e in closed.jaxpr.eqns])
print("Interpreted primal and tangent:",float(primal_outputs[0]),float(tangent_outputs[0]))

# Lower and compile the transformation itself
transformed=lambda value,tangent:tiny_jvp(closed,(value,),(tangent,))
lowered=jax.jit(transformed).lower(x,direction)
stablehlo=str(lowered.compiler_ir(dialect="stablehlo"))
assert len(stablehlo)>0
compiled=lowered.compile()
compiled_primal,compiled_tangent=compiled(x,direction)
np.testing.assert_allclose(compiled_primal[0],reference_value,rtol=1e-12)
np.testing.assert_allclose(compiled_tangent[0],reference_tangent,rtol=1e-12)
shape_rejected=False
try:
    compiled(jnp.ones(4),jnp.ones(4))
except TypeError:
    shape_rejected=True
assert shape_rejected
print("Lowered IR characters:",len(stablehlo))
print("Compiled transformed outputs match; changed shape explicitly rejected")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

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

offset=jnp.array([0.1,0.2,0.3])
def program(x):
    return jnp.sum(jnp.sin(x)*x+offset)
x=jnp.array([-0.8,0.4,1.1]);direction=jnp.array([0.3,-0.2,0.7])
closed=jax.make_jaxpr(program)(x)
primal_outputs,tangent_outputs=tiny_jvp(closed,(x,),(direction,))
reference_value=np.sum(np.sin(np.asarray(x))*np.asarray(x)+np.asarray(offset))
reference_tangent=np.dot(np.sin(np.asarray(x))+np.asarray(x)*np.cos(np.asarray(x)),np.asarray(direction))
np.testing.assert_allclose(primal_outputs[0],reference_value,rtol=1e-12)
np.testing.assert_allclose(tangent_outputs[0],reference_tangent,rtol=1e-12)
np.testing.assert_allclose(tangent_outputs[0],jax.jvp(program,(x,),(direction,))[1],rtol=1e-12)
print("Supported primitives:",[e.primitive.name for e in closed.jaxpr.eqns])
print("Interpreted primal and tangent:",float(primal_outputs[0]),float(tangent_outputs[0]))

transformed=lambda value,tangent:tiny_jvp(closed,(value,),(tangent,))
lowered=jax.jit(transformed).lower(x,direction)
stablehlo=str(lowered.compiler_ir(dialect="stablehlo"))
assert len(stablehlo)>0
compiled=lowered.compile()
compiled_primal,compiled_tangent=compiled(x,direction)
np.testing.assert_allclose(compiled_primal[0],reference_value,rtol=1e-12)
np.testing.assert_allclose(compiled_tangent[0],reference_tangent,rtol=1e-12)
shape_rejected=False
try:
    compiled(jnp.ones(4),jnp.ones(4))
except TypeError:
    shape_rejected=True
assert shape_rejected
print("Lowered IR characters:",len(stablehlo))
print("Compiled transformed outputs match; changed shape explicitly rejected")

# Figure data experiment
alphas=np.linspace(-1,1,9)
interpreted=[];analytic=[]
for alpha in alphas:
    point=x+alpha*direction
    interpreted.append(float(tiny_jvp(closed,(point,),(direction,))[1][0]))
    host=np.asarray(point)
    analytic.append(float(np.dot(np.sin(host)+host*np.cos(host),np.asarray(direction))))
visual_data={'kind':'line','x':alphas.tolist(),'xlabel':'base-point displacement alpha','ylabel':'directional derivative','series':[{'label':'tiny interpreted tangent','y':interpreted},{'label':'whole-expression analytic reference','y':analytic}]}

# Experiment: Exercise an unsupported primitive
unsupported=jax.make_jaxpr(lambda z:jnp.sum(jnp.exp(z)))(x)
rejected=False
try:
    tiny_jvp(unsupported,(x,),(direction,))
except NotImplementedError as error:
    rejected=True
    assert "exp" in str(error)
    print("Expected unsupported primitive:",error)
assert rejected

# Experiment: Change values while preserving the compiled signature
changed=x+jnp.array([0.1,-0.4,0.2])
cp,ct=compiled(changed,direction)
expected=np.sum(np.sin(np.asarray(changed))*np.asarray(changed)+np.asarray(offset))
expected_d=np.dot(np.sin(np.asarray(changed))+np.asarray(changed)*np.cos(np.asarray(changed)),np.asarray(direction))
np.testing.assert_allclose(cp[0],expected,rtol=1e-12)
np.testing.assert_allclose(ct[0],expected_d,rtol=1e-12)
print("Changed-value compiled tangent:",float(ct[0]))

# Reference solution. Try the exercise before reading this.
def polynomial(z):
    return jnp.sum(z*z+(-2.0)*z)
poly_closed=jax.make_jaxpr(polynomial)(x)
poly_transform=jax.jit(lambda point,seed:tiny_jvp(poly_closed,(point,),(seed,)))
for point in (x,x+0.6):
    pv,pt=poly_transform(point,direction)
    np.testing.assert_allclose(pv[0],np.sum(np.asarray(point)**2-2*np.asarray(point)),rtol=1e-12)
    np.testing.assert_allclose(pt[0],np.dot(2*np.asarray(point)-2,np.asarray(direction)),rtol=1e-12)
print("Changed polynomial transformed and compiled")

# Reference practice: Make a constant into a differentiable input
def explicit(z,c):
    return jnp.sum(jnp.sin(z)*z+c)
explicit_closed=jax.make_jaxpr(explicit)(x,offset)
constant_direction=jnp.array([.2,.1,-.4])
ev,et=tiny_jvp(explicit_closed,(x,offset),(direction,constant_direction))
np.testing.assert_allclose(et[0],reference_tangent+np.sum(np.asarray(constant_direction)),rtol=1e-12)
print("Explicit offset tangent contribution:",float(jnp.sum(constant_direction)))

# Reference practice: Verify multiple outputs and negation
multi=lambda z:(jnp.sum(z*z),-z)
multi_closed=jax.make_jaxpr(multi)(x)
mp,mt=tiny_jvp(multi_closed,(x,),(direction,))
np.testing.assert_allclose(mp[0],np.sum(np.asarray(x)**2),rtol=1e-12)
np.testing.assert_allclose(mp[1],-np.asarray(x),rtol=1e-12)
np.testing.assert_allclose(mt[0],2*np.dot(np.asarray(x),np.asarray(direction)),rtol=1e-12)
np.testing.assert_allclose(mt[1],-np.asarray(direction),rtol=1e-12)
rejected=False
try:tiny_jvp(multi_closed,(x,),(jnp.ones(2),))
except ValueError:rejected=True
assert rejected
print("Multiple output contract and malformed tangent rejection verified")
print("PASS: internals-04")
