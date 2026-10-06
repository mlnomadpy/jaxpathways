# Lowering, compilation, and a tiny transformation

Phase 14: Autodiff & JAX internals · about 130 minutes · CPU

## What you will be able to do

- Interpret literals, captured constants and variables in a supported flat jaxpr.
- Implement actual forward derivative rules without calling jax.jvp inside the interpreter.
- Reject unsupported operations and mismatched signatures explicitly.
- Compare interpreted, JAX, analytic and compiled transformed results.

## The problem

Reading a jaxpr tells you what operations exist, but implementing one small transformation makes its rules concrete. Can we propagate a value and a directional derivative through a traced program, compile that transformed computation, and clearly reject programs we do not support?

## The idea

We will write a tiny forward-mode interpreter for a pure, flat floating-point primitive subset. Each variable carries a pair $(a,\dot a)$. The interpreter applies primal and tangent rules to each equation in order. This is a real derivative transformation with explicit limitations, not a general replacement for JAX’s interpreter or compiler.

## Treat a jaxpr as a typed data-flow program

A closed jaxpr contains input variables, captured constants, equations and output variables. Each equation reads already-defined values and writes new values. Its variable names are presentation details; the object identities and data dependencies carry meaning. We use the public extension namespace for the Literal type, and inspect the installed JAX representation rather than matching printed variable names. The executed environment is JAX 0.9.2, where the closed trace exposes .jaxpr and .consts separately. Newer documentation may describe a unified representation; inspect the installed version and rerun these contracts when upgrading.

## Attach a tangent to every value

For addition, add both primals and tangents. For multiplication, the tangent is $\dot a b+a\dot b$, not the product of the tangents. For sine, multiply the incoming tangent by the cosine of the primal. Reduction sums both values over the same axes. Literals and captured constants have zero tangent with respect to the explicit input. That last qualifier matters: making a formerly captured parameter an input changes the derivative contract.

$$
(a,\dot a)\cdot(b,\dot b)=(ab,\dot a b+a\dot b),\qquad \sin(a,\dot a)=(\sin a,\cos a\,\dot a)
$$

## A small supported subset is an enforceable boundary

The interpreter supports add, multiply, negate, sine and sum-reduction. It rejects effects, unsupported primitives, multiple-result primitives, explicit reduction placement, integer inputs and signature mismatches. It does not interpret nested jit calls, conditionals, scan, random state, complex differentiation or custom derivatives. Silent fallback to executing an unknown primitive would produce a value without a justified derivative, so the explicit exception is essential.

## Verify a complete transformed program

Our example is $F(x)=\sum_i(x_i\sin x_i+c_i)$, with captured offset $c$. Its analytic directional derivative is $\sum_i(\sin x_i+x_i\cos x_i)v_i$. The interpreter implements that derivative through local primitive rules, while the reference derives it from the whole expression. Comparing those approaches checks both the local rules and how they are composed.

## Trace, lower and compile are separate stages

make_jaxpr exposes a JAX-level program. Lowering converts a traced computation toward compiler input, and compile produces an executable specialized to the signature and target. We lower the transformed interpreter, not just the original function. The interpreter’s Python loop runs during tracing; its arithmetic becomes the program that executes later. StableHLO text is useful for inspection, but its length or equation count is not a runtime measurement.

## Specialization is part of the artifact contract

The ahead-of-time executable accepts the same array shapes and dtypes used for lowering. A changed shape requires retracing and recompilation; the compiled callable does not automatically create a new specialization. We deliberately test rejection rather than assuming shape polymorphism. Numeric input values may change within the same signature, and we test those changed values separately. Compiled artifact portability across devices or JAX versions is not established here.

## Define a deliberately small derivative interpreter

Create main.py for the first block, then append subsequent blocks in order using your course CPU environment.

```python
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
```

Each environment entry stores a primal and its tangent. Constants and literals receive zero tangents. Every supported primitive has a written derivative rule; unknown operations fail explicitly.

## Trace a program and verify the transformed result

Create main.py for the first block, then append subsequent blocks in order using your course CPU environment.

```python
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
```

The traced closed constant contributes to the primal but not the input derivative. The interpreter performs genuine forward derivative propagation rather than delegating to jax.jvp.

## Lower and compile the transformation itself

Create main.py for the first block, then append subsequent blocks in order using your course CPU environment.

```python
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
```

The Python interpreter runs while tracing the transformed function; its JAX arithmetic becomes a new compiled program. The ahead-of-time executable is specialized to the recorded input signature.

## Run the example

```python
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
```

Expected: The program contains sin, mul, add and reduce_sum. Interpreted and compiled primal/tangent values match analytic and JAX references; a changed compiled shape is rejected.

## A composed interpreter tracks the analytic directional derivative

**Predict:** As the base point moves along a line, should its directional derivative remain constant?

![A composed interpreter tracks the analytic directional derivative](../outputs/figure.svg)

**Recorded CPU computation**

### Read the figure

The horizontal axis is a scalar displacement $\alpha$ of the base point $x+\alpha v$. The vertical axis is the directional derivative of the scalar program along the fixed direction $v$. Each point comes from a new evaluation of the same supported trace at a shape-compatible input.

The interpreter and analytic curves overlap. Their derivative is about $-0.09853$ at $\alpha=-1$, rises to a sampled peak of about $0.43913$ at the original point $\alpha=0$, then falls to about $0.04102$ at $\alpha=1$. The sign change and nonmonotone shape reflect a changing Jacobian, not disagreement between the two methods.

### Connect it to the computation

The interpreter obtains each value by composing local tangent rules; the reference directly evaluates $\sum_i(\sin z_i+z_i\cos z_i)v_i$ at $z=x+\alpha v$. The code also checks the middle point against JAX’s JVP and checks the lowered, compiled transformed program.

Agreement applies to this primitive subset and tested inputs. It does not validate an unsupported operation, nested control flow, compiler performance or device portability. Those require new rules and new evidence rather than extrapolation from this plot.

```python
alphas=np.linspace(-1,1,9)
interpreted=[];analytic=[]
for alpha in alphas:
    point=x+alpha*direction
    interpreted.append(float(tiny_jvp(closed,(point,),(direction,))[1][0]))
    host=np.asarray(point)
    analytic.append(float(np.dot(np.sin(host)+host*np.cos(host),np.asarray(direction))))
visual_data={'kind':'line','x':alphas.tolist(),'xlabel':'base-point displacement alpha','ylabel':'directional derivative','series':[{'label':'tiny interpreted tangent','y':interpreted},{'label':'whole-expression analytic reference','y':analytic}]}
```

## Recorded reference execution

CPU run: 2026-10-06T01:26:06.438471+00:00. JAX 0.9.2.

```text
Supported primitives: ['sin', 'mul', 'add', 'reduce_sum']
Interpreted primal and tangent: 2.3099803057106576 0.4391291800455618
Lowered IR characters: 1421
Compiled transformed outputs match; changed shape explicitly rejected
Supported primitives: ['sin', 'mul', 'add', 'reduce_sum']
Interpreted primal and tangent: 2.3099803057106576 0.4391291800455618
Lowered IR characters: 1421
Compiled transformed outputs match; changed shape explicitly rejected
Expected unsupported primitive: unsupported primitive: exp
Changed-value compiled tangent: 0.5640324983393596
Changed polynomial transformed and compiled
Explicit offset tangent contribution: -0.09999999999999998
Multiple output contract and malformed tangent rejection verified
PASS: internals-04

```

## Exercise an unsupported primitive

**Predict before running:** If a new function uses exp, should the interpreter guess a derivative or stop?

```python
unsupported=jax.make_jaxpr(lambda z:jnp.sum(jnp.exp(z)))(x)
rejected=False
try:
    tiny_jvp(unsupported,(x,),(direction,))
except NotImplementedError as error:
    rejected=True
    assert "exp" in str(error)
    print("Expected unsupported primitive:",error)
assert rejected
```

**Expected:** An explicit unsupported primitive: exp error is raised.

A numerical result without a supported tangent rule would falsely advertise derivative coverage. Add and verify a rule before accepting the new program.

## Change values while preserving the compiled signature

**Predict before running:** Can an executable specialized to three float64 values accept different values of that same shape?

```python
changed=x+jnp.array([0.1,-0.4,0.2])
cp,ct=compiled(changed,direction)
expected=np.sum(np.sin(np.asarray(changed))*np.asarray(changed)+np.asarray(offset))
expected_d=np.dot(np.sin(np.asarray(changed))+np.asarray(changed)*np.cos(np.asarray(changed)),np.asarray(direction))
np.testing.assert_allclose(cp[0],expected,rtol=1e-12)
np.testing.assert_allclose(ct[0],expected_d,rtol=1e-12)
print("Changed-value compiled tangent:",float(ct[0]))
```

**Expected:** Both compiled outputs agree on a new point without changing the signature.

Specialization constrains abstract shape/dtype contracts. It does not mean every input number is frozen into the program; captured constants are a separate matter.

## Make it yours

Trace a new polynomial using only supported primitives, verify its interpreted derivative at two points, and compile that transformed computation independently.

<details><summary>Reference solution</summary>

```python
def polynomial(z):
    return jnp.sum(z*z+(-2.0)*z)
poly_closed=jax.make_jaxpr(polynomial)(x)
poly_transform=jax.jit(lambda point,seed:tiny_jvp(poly_closed,(point,),(seed,)))
for point in (x,x+0.6):
    pv,pt=poly_transform(point,direction)
    np.testing.assert_allclose(pv[0],np.sum(np.asarray(point)**2-2*np.asarray(point)),rtol=1e-12)
    np.testing.assert_allclose(pt[0],np.dot(2*np.asarray(point)-2,np.asarray(direction)),rtol=1e-12)
print("Changed polynomial transformed and compiled")
```

</details>

## Make a constant into a differentiable input

**Practice**

Replace the captured offset by an explicit argument, seed that argument with a nonzero direction, and verify its contribution to the total tangent.

<details><summary>Hint</summary>

The new program has two input variables and needs two tangent arguments.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
def explicit(z,c):
    return jnp.sum(jnp.sin(z)*z+c)
explicit_closed=jax.make_jaxpr(explicit)(x,offset)
constant_direction=jnp.array([.2,.1,-.4])
ev,et=tiny_jvp(explicit_closed,(x,offset),(direction,constant_direction))
np.testing.assert_allclose(et[0],reference_tangent+np.sum(np.asarray(constant_direction)),rtol=1e-12)
print("Explicit offset tangent contribution:",float(jnp.sum(constant_direction)))
```

Zero tangent belonged to a captured constant relative to the old input signature. An explicit input is allowed to vary, so its seed must propagate.

</details>

## Verify multiple outputs and negation

**Challenge**

Trace a supported function returning both a scalar sum and a negated vector. Check the tuple of primal/tangent outputs and reject a mismatched tangent shape.

<details><summary>Hint</summary>

Multiple program outputs are supported even though each supported primitive returns one result.

</details>

<details><summary>Reference solution and reasoning</summary>

```python
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
```

Output structure and primitive result structure are distinct. The interpreter returns a flat tuple matching the jaxpr outputs, not an arbitrary original Python pytree.

</details>

## Check your understanding

Why does the tiny interpreter raise an error for an unsupported primitive instead of calling that primitive directly?

1. Every unsupported primitive is mathematically nondifferentiable.
2. Executing a primal without a verified tangent rule would silently break the derivative contract.
3. Compiled JAX cannot run exponentials.

<details><summary>Answer and explanation</summary>

Executing a primal without a verified tangent rule would silently break the derivative contract.

The missing piece is this interpreter’s derivative rule and semantics, not necessarily JAX support or mathematical differentiability. Explicit rejection keeps the supported contract honest.

</details>

## Diagnose the result

If primals match but tangents fail, audit multiplication’s two terms and zero tangents for constants. If an unknown primitive appears, stop and inspect its semantics instead of bypassing the error. If compilation rejects a changed shape, rebuild the trace and executable for that signature. Do not infer speed or optimization quality from IR text length.

## Carry forward

- Treat a jaxpr as a typed data-flow program
- Attach a tangent to every value
- A small supported subset is an enforceable boundary
- Verify a complete transformed program
- Trace, lower and compile are separate stages
- Specialization is part of the artifact contract

## Keep your evidence

Annotated primitive rules, constant/literal handling, independent expression oracles, changed compiled values and explicit unsupported/signature errors. Keep the environment, observed outputs and your explanation of the figure.

Keep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.

## Primary references

- [JAX automatic differentiation cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html)
- [Custom JVP and VJP rules](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)
- [Ahead-of-time lowering and compilation](https://docs.jax.dev/en/latest/aot.html)
- [The jaxpr language](https://docs.jax.dev/en/latest/601/jaxpr.html)

