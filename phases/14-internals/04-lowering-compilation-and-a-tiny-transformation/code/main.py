"""Lowering, compilation, and a tiny transformation: worked experiments and reference solutions. CPU checks."""

# Define a deliberately small derivative interpreter
# Step 1 — Define a deliberately small derivative interpreter: Each environment entry stores a primal and its tangent.
# Import jax for this computation.
import jax
# Update state in place with the new values.
jax.config.update("jax_enable_x64", True)
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np

from jax.extend import core

# Function `tiny_jvp(closed, primals, tangents)` implementing this stage's computation:
def tiny_jvp(closed,primals,tangents):
    """Interpret a pure flat jaxpr with floating inputs and explicit tangent rules."""
    # Compute `program` from `closed.jaxpr`
    program=closed.jaxpr
    # Guard input contract (`program.effects`) and fail fast if violated.
    if program.effects:
        raise NotImplementedError("effects are outside this interpreter")
    # Guard input contract (`len(primals) != len(program.invars) or len(tangents) != len(primals)`) and fail fast if violated.
    if len(primals)!=len(program.invars) or len(tangents)!=len(primals):
        raise ValueError("one primal and tangent per input variable required")
    # Compute `values` from `{}`
    values={}
    directions={}
    # Function `put(var, value, tangent)` implementing this stage's computation:
    def put(var,value,tangent):
        # Compute `values[var]` from `value`
        values[var]=value
        directions[var]=tangent
    # Iterate over `(var, constant)` to step through the computation:
    for var,constant in zip(program.constvars,closed.consts):
        # Allocate initialized array `` with the specified shape and dtype.
        put(var,constant,jnp.zeros_like(constant))
    # Iterate over `(var, value, tangent)` to step through the computation:
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
        # Compute `name` from `equation.primitive.name`
        name=equation.primitive.name
        # Guard input contract (`name not in {'add', 'mul', 'neg', 'sin', 'reduce_sum'}`) and fail fast if violated.
        if name not in {"add","mul","neg","sin","reduce_sum"}:
            raise NotImplementedError("unsupported primitive: "+name)
        # Guard input contract (`len(equation.outvars) != 1`) and fail fast if violated.
        if len(equation.outvars)!=1:
            raise NotImplementedError("multiple-result primitives are outside this interpreter")
        # Compute `operands` from `[read(atom) for atom in equation.invars]`
        operands=[read(atom) for atom in equation.invars]
        # Compute `a,da` from `operands[0]`
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
    # Compute `output` from `[read(var) for var in program.outvars]`
    output=[read(var) for var in program.outvars]
    # Return `(tuple((v for v, _ in output)), tuple((t for _, t in output)))` to the caller.
    return tuple(v for v,_ in output),tuple(t for _,t in output)

# Trace a program and verify the transformed result
# Step 2 — Trace a program and verify the transformed result: The traced closed constant contributes to the primal but not the...
# Construct `offset` via `jnp.array([0.1,0.2,0.3])`
offset=jnp.array([0.1,0.2,0.3])
# Function `program(x)` implementing this stage's computation:
def program(x):
    # Return `jnp.sum(jnp.sin(x) * x + offset)` to the caller.
    return jnp.sum(jnp.sin(x)*x+offset)
# Construct `x` via `jnp.array([-0.8,0.4,1.1])`
x=jnp.array([-0.8,0.4,1.1])
direction=jnp.array([0.3,-0.2,0.7])
# Trace or lower the function to inspect its compiler representation (`closed`).
closed=jax.make_jaxpr(program)(x)
# Run `tiny_jvp` to compute `(primal_outputs, tangent_outputs)`.
primal_outputs,tangent_outputs=tiny_jvp(closed,(x,),(direction,))
# Aggregate array values to compute `reference_value`.
reference_value=np.sum(np.sin(np.asarray(x))*np.asarray(x)+np.asarray(offset))
# Convert `reference_tangent` to a host NumPy array for inspection or verification.
reference_tangent=np.dot(np.sin(np.asarray(x))+np.asarray(x)*np.cos(np.asarray(x)),np.asarray(direction))
# Compute `np.testing.assert_allclose(primal_outputs[0],reference_value,rtol` as `1e-12)`.
np.testing.assert_allclose(primal_outputs[0],reference_value,rtol=1e-12)
# Compute `np.testing.assert_allclose(tangent_outputs[0],reference_tangent,rtol` as `1e-12)`.
np.testing.assert_allclose(tangent_outputs[0],reference_tangent,rtol=1e-12)
# Compute `np.testing.assert_allclose(tangent_outputs[0],jax.jvp(program,(x,),(direction,))[1],rtol` as `1e-12)`.
np.testing.assert_allclose(tangent_outputs[0],jax.jvp(program,(x,),(direction,))[1],rtol=1e-12)
# Print diagnostic summary of the computed outputs.
print("Supported primitives:",[e.primitive.name for e in closed.jaxpr.eqns])
# Print diagnostic summary of the computed outputs.
print("Interpreted primal and tangent:",float(primal_outputs[0]),float(tangent_outputs[0]))

# Lower and compile the transformation itself
# Step 3 — Lower and compile the transformation itself: The Python interpreter runs while tracing the transformed...
transformed=lambda value,tangent:tiny_jvp(closed,(value,),(tangent,))
# Wrap with `jax.jit` (`lowered`) so XLA traces and compiles the function.
lowered=jax.jit(transformed).lower(x,direction)
# Trace or lower the function to inspect its compiler representation (`stablehlo`).
stablehlo=str(lowered.compiler_ir(dialect="stablehlo"))
# Assert invariant `len(stablehlo)>0` holds
assert len(stablehlo)>0
# Trace or lower the function to inspect its compiler representation (`compiled`).
compiled=lowered.compile()
# Run `compiled` to compute `(compiled_primal, compiled_tangent)`.
compiled_primal,compiled_tangent=compiled(x,direction)
# Compute `np.testing.assert_allclose(compiled_primal[0],reference_value,rtol` as `1e-12)`.
np.testing.assert_allclose(compiled_primal[0],reference_value,rtol=1e-12)
# Compute `np.testing.assert_allclose(compiled_tangent[0],reference_tangent,rtol` as `1e-12)`.
np.testing.assert_allclose(compiled_tangent[0],reference_tangent,rtol=1e-12)
# Compute `shape_rejected` as `False`.
shape_rejected=False
# Run the boundary check and catch the expected exception:
try:
    compiled(jnp.ones(4),jnp.ones(4))
except TypeError:
    shape_rejected=True
# Assert invariant `shape_rejected` holds
assert shape_rejected
# Print diagnostic summary of the computed outputs.
print("Lowered IR characters:",len(stablehlo))
# Print diagnostic summary of the computed outputs.
print("Compiled transformed outputs match; changed shape explicitly rejected")

# Step 1 — Define a deliberately small derivative interpreter: Each environment entry stores a primal and its tangent.
# Import jax for this computation.
import jax
# Update state in place with the new values.
jax.config.update("jax_enable_x64", True)
# Import jax.numpy for this computation.
import jax.numpy as jnp
import numpy as np

from jax.extend import core

# Function `tiny_jvp(closed, primals, tangents)` implementing this stage's computation:
def tiny_jvp(closed,primals,tangents):
    """Interpret a pure flat jaxpr with floating inputs and explicit tangent rules."""
    # Compute `program` from `closed.jaxpr`
    program=closed.jaxpr
    # Guard input contract (`program.effects`) and fail fast if violated.
    if program.effects:
        raise NotImplementedError("effects are outside this interpreter")
    # Guard input contract (`len(primals) != len(program.invars) or len(tangents) != len(primals)`) and fail fast if violated.
    if len(primals)!=len(program.invars) or len(tangents)!=len(primals):
        raise ValueError("one primal and tangent per input variable required")
    # Compute `values` from `{}`
    values={}
    directions={}
    # Function `put(var, value, tangent)` implementing this stage's computation:
    def put(var,value,tangent):
        # Compute `values[var]` from `value`
        values[var]=value
        directions[var]=tangent
    # Iterate over `(var, constant)` to step through the computation:
    for var,constant in zip(program.constvars,closed.consts):
        # Allocate initialized array `` with the specified shape and dtype.
        put(var,constant,jnp.zeros_like(constant))
    # Iterate over `(var, value, tangent)` to step through the computation:
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
        # Compute `name` from `equation.primitive.name`
        name=equation.primitive.name
        # Guard input contract (`name not in {'add', 'mul', 'neg', 'sin', 'reduce_sum'}`) and fail fast if violated.
        if name not in {"add","mul","neg","sin","reduce_sum"}:
            raise NotImplementedError("unsupported primitive: "+name)
        # Guard input contract (`len(equation.outvars) != 1`) and fail fast if violated.
        if len(equation.outvars)!=1:
            raise NotImplementedError("multiple-result primitives are outside this interpreter")
        # Compute `operands` from `[read(atom) for atom in equation.invars]`
        operands=[read(atom) for atom in equation.invars]
        # Compute `a,da` from `operands[0]`
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
    # Compute `output` from `[read(var) for var in program.outvars]`
    output=[read(var) for var in program.outvars]
    # Return `(tuple((v for v, _ in output)), tuple((t for _, t in output)))` to the caller.
    return tuple(v for v,_ in output),tuple(t for _,t in output)

# Step 2 — Trace a program and verify the transformed result: The traced closed constant contributes to the primal but not the...
# Construct `offset` via `jnp.array([0.1,0.2,0.3])`
offset=jnp.array([0.1,0.2,0.3])
# Function `program(x)` implementing this stage's computation:
def program(x):
    # Return `jnp.sum(jnp.sin(x) * x + offset)` to the caller.
    return jnp.sum(jnp.sin(x)*x+offset)
# Construct `x` via `jnp.array([-0.8,0.4,1.1])`
x=jnp.array([-0.8,0.4,1.1])
direction=jnp.array([0.3,-0.2,0.7])
# Trace or lower the function to inspect its compiler representation (`closed`).
closed=jax.make_jaxpr(program)(x)
# Run `tiny_jvp` to compute `(primal_outputs, tangent_outputs)`.
primal_outputs,tangent_outputs=tiny_jvp(closed,(x,),(direction,))
# Aggregate array values to compute `reference_value`.
reference_value=np.sum(np.sin(np.asarray(x))*np.asarray(x)+np.asarray(offset))
# Convert `reference_tangent` to a host NumPy array for inspection or verification.
reference_tangent=np.dot(np.sin(np.asarray(x))+np.asarray(x)*np.cos(np.asarray(x)),np.asarray(direction))
# Compute `np.testing.assert_allclose(primal_outputs[0],reference_value,rtol` as `1e-12)`.
np.testing.assert_allclose(primal_outputs[0],reference_value,rtol=1e-12)
# Compute `np.testing.assert_allclose(tangent_outputs[0],reference_tangent,rtol` as `1e-12)`.
np.testing.assert_allclose(tangent_outputs[0],reference_tangent,rtol=1e-12)
# Compute `np.testing.assert_allclose(tangent_outputs[0],jax.jvp(program,(x,),(direction,))[1],rtol` as `1e-12)`.
np.testing.assert_allclose(tangent_outputs[0],jax.jvp(program,(x,),(direction,))[1],rtol=1e-12)
# Print diagnostic summary of the computed outputs.
print("Supported primitives:",[e.primitive.name for e in closed.jaxpr.eqns])
# Print diagnostic summary of the computed outputs.
print("Interpreted primal and tangent:",float(primal_outputs[0]),float(tangent_outputs[0]))

# Step 3 — Lower and compile the transformation itself: The Python interpreter runs while tracing the transformed...
transformed=lambda value,tangent:tiny_jvp(closed,(value,),(tangent,))
# Wrap with `jax.jit` (`lowered`) so XLA traces and compiles the function.
lowered=jax.jit(transformed).lower(x,direction)
# Trace or lower the function to inspect its compiler representation (`stablehlo`).
stablehlo=str(lowered.compiler_ir(dialect="stablehlo"))
# Assert invariant `len(stablehlo)>0` holds
assert len(stablehlo)>0
# Trace or lower the function to inspect its compiler representation (`compiled`).
compiled=lowered.compile()
# Run `compiled` to compute `(compiled_primal, compiled_tangent)`.
compiled_primal,compiled_tangent=compiled(x,direction)
# Compute `np.testing.assert_allclose(compiled_primal[0],reference_value,rtol` as `1e-12)`.
np.testing.assert_allclose(compiled_primal[0],reference_value,rtol=1e-12)
# Compute `np.testing.assert_allclose(compiled_tangent[0],reference_tangent,rtol` as `1e-12)`.
np.testing.assert_allclose(compiled_tangent[0],reference_tangent,rtol=1e-12)
# Compute `shape_rejected` as `False`.
shape_rejected=False
# Run the boundary check and catch the expected exception:
try:
    compiled(jnp.ones(4),jnp.ones(4))
except TypeError:
    shape_rejected=True
# Assert invariant `shape_rejected` holds
assert shape_rejected
# Print diagnostic summary of the computed outputs.
print("Lowered IR characters:",len(stablehlo))
# Print diagnostic summary of the computed outputs.
print("Compiled transformed outputs match; changed shape explicitly rejected")

# Figure data experiment
# Compute figure data for: A composed interpreter tracks the analytic directional derivative
# Generate a uniform grid of points in `alphas`.
alphas=np.linspace(-1,1,9)
# Compute `interpreted` from `[]`
interpreted=[]
analytic=[]
# Loop over `alpha` in `alphas`:
for alpha in alphas:
    # Compute `point` from `x+alpha*direction`
    point=x+alpha*direction
    # Append the current step result to `interpreted`.
    interpreted.append(float(tiny_jvp(closed,(point,),(direction,))[1][0]))
    # Convert `host` to a host NumPy array for inspection or verification.
    host=np.asarray(point)
    # Convert `` to a host NumPy array for inspection or verification.
    analytic.append(float(np.dot(np.sin(host)+host*np.cos(host),np.asarray(direction))))
# Compute `visual_data` from `{'kind':'line','x':alphas.tolist(),'xlabel':'base-po...`
visual_data={'kind':'line','x':alphas.tolist(),'xlabel':'base-point displacement alpha','ylabel':'directional derivative','series':[{'label':'tiny interpreted tangent','y':interpreted},{'label':'whole-expression analytic reference','y':analytic}]}

# Experiment: Exercise an unsupported primitive
# Experiment — Exercise an unsupported primitive: A numerical result without a supported tangent rule would...
# Trace or lower the function to inspect its compiler representation (`unsupported`).
unsupported=jax.make_jaxpr(lambda z:jnp.sum(jnp.exp(z)))(x)
# Compute `rejected` as `False`.
rejected=False
# Run the boundary check and catch the expected exception:
try:
    tiny_jvp(unsupported,(x,),(direction,))
except NotImplementedError as error:
    rejected=True
    assert "exp" in str(error)
    print("Expected unsupported primitive:",error)
# Assert invariant `rejected` holds
assert rejected

# Experiment: Change values while preserving the compiled signature
# Experiment — Change values while preserving the compiled signature: Specialization constrains abstract shape/dtype contracts.
# Construct `changed` via `x+jnp.array([0.1,-0.4,0.2])`
changed=x+jnp.array([0.1,-0.4,0.2])
# Run `compiled` to compute `(cp, ct)`.
cp,ct=compiled(changed,direction)
# Aggregate array values to compute `expected`.
expected=np.sum(np.sin(np.asarray(changed))*np.asarray(changed)+np.asarray(offset))
# Convert `expected_d` to a host NumPy array for inspection or verification.
expected_d=np.dot(np.sin(np.asarray(changed))+np.asarray(changed)*np.cos(np.asarray(changed)),np.asarray(direction))
# Compute `np.testing.assert_allclose(cp[0],expected,rtol` as `1e-12)`.
np.testing.assert_allclose(cp[0],expected,rtol=1e-12)
# Compute `np.testing.assert_allclose(ct[0],expected_d,rtol` as `1e-12)`.
np.testing.assert_allclose(ct[0],expected_d,rtol=1e-12)
# Print the observed values to compare against the expected result.
print("Changed-value compiled tangent:",float(ct[0]))

# Reference solution. Try the exercise before reading this.
# Exercise solution: Trace a new polynomial using only supported primitives, verify its...
def polynomial(z):
    # Return `jnp.sum(z * z + -2.0 * z)` to the caller.
    return jnp.sum(z*z+(-2.0)*z)
# Trace or lower the function to inspect its compiler representation (`poly_closed`).
poly_closed=jax.make_jaxpr(polynomial)(x)
# Wrap with `jax.jit` (`poly_transform`) so XLA traces and compiles the function.
poly_transform=jax.jit(lambda point,seed:tiny_jvp(poly_closed,(point,),(seed,)))
# Iterate over `point` to step through the computation:
for point in (x,x+0.6):
    # Run `poly_transform` to compute `(pv, pt)`.
    pv,pt=poly_transform(point,direction)
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_allclose(pv[0],np.sum(np.asarray(point)**2-2*np.asarray(point)),rtol=1e-12)
    # Convert `` to a host NumPy array for inspection or verification.
    np.testing.assert_allclose(pt[0],np.dot(2*np.asarray(point)-2,np.asarray(direction)),rtol=1e-12)
# Print the observed values to compare against the expected result.
print("Changed polynomial transformed and compiled")

# Reference practice: Make a constant into a differentiable input
# Make a constant into a differentiable input (Practice): Zero tangent belonged to a captured constant relative to the...
def explicit(z,c):
    # Return `jnp.sum(jnp.sin(z) * z + c)` to the caller.
    return jnp.sum(jnp.sin(z)*z+c)
# Trace or lower the function to inspect its compiler representation (`explicit_closed`).
explicit_closed=jax.make_jaxpr(explicit)(x,offset)
# Construct `constant_direction` via `jnp.array([.2,.1,-.4])`
constant_direction=jnp.array([.2,.1,-.4])
# Run `tiny_jvp` to compute `(ev, et)`.
ev,et=tiny_jvp(explicit_closed,(x,offset),(direction,constant_direction))
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(et[0],reference_tangent+np.sum(np.asarray(constant_direction)),rtol=1e-12)
# Print the observed values to compare against the expected result.
print("Explicit offset tangent contribution:",float(jnp.sum(constant_direction)))

# Reference practice: Verify multiple outputs and negation
# Verify multiple outputs and negation (Challenge): Output structure and primitive result structure are distinct.
# Aggregate array values to compute `multi`.
multi=lambda z:(jnp.sum(z*z),-z)
# Trace or lower the function to inspect its compiler representation (`multi_closed`).
multi_closed=jax.make_jaxpr(multi)(x)
# Run `tiny_jvp` to compute `(mp, mt)`.
mp,mt=tiny_jvp(multi_closed,(x,),(direction,))
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(mp[0],np.sum(np.asarray(x)**2),rtol=1e-12)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(mp[1],-np.asarray(x),rtol=1e-12)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(mt[0],2*np.dot(np.asarray(x),np.asarray(direction)),rtol=1e-12)
# Convert `` to a host NumPy array for inspection or verification.
np.testing.assert_allclose(mt[1],-np.asarray(direction),rtol=1e-12)
# Compute `rejected` as `False`.
rejected=False
# Run the boundary check and catch the expected exception:
try:tiny_jvp(multi_closed,(x,),(jnp.ones(2),))
except ValueError:rejected=True
# Assert invariant `rejected` holds
assert rejected
# Print the observed values to compare against the expected result.
print("Multiple output contract and malformed tangent rejection verified")
print("PASS: internals-04")
