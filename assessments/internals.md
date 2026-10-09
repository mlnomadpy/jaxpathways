# Internals synthesis: extend and defend a derivative transformation

**Scope:** `jaxpr` inspection, forward/reverse-mode autodiff, and custom VJP rules. Complete the internals lessons and [derivative audit project](../project.html?id=derivative-audit), then transfer those methods to the new map and primitive below.

Record Python/JAX/NumPy versions, actual backend/device count, dtype, commands and tolerances. Preserve your implementation and create a separate assessment harness/report. State predictions before running. Do not change public tests to make a result pass.

### Worked verification scaffold

Run the baseline verification suite with `python3 assessments/check_assessments.py`, and use the starter scaffold below to verify your numerical contracts:

```python
# Worked starter scaffold: rectangular JVP, VJP, and adjoint dot-product identity
import jax
import jax.numpy as jnp

def f_rect(x: jnp.ndarray) -> jnp.ndarray:
    a, b = x[0], x[1]
    return jnp.stack([a * b, jnp.sin(a) + b ** 2, a ** 2 - b])

x0 = jnp.array([0.3, -0.6], dtype=jnp.float32)
v = jnp.array([0.7, -0.2], dtype=jnp.float32)
u = jnp.array([0.5, -0.8, 1.2], dtype=jnp.float32)

# Verify <u, J(x0) v> == <J(x0)^T u, v> to machine precision.
_, jvp_out = jax.jvp(f_rect, (x0,), (v,))
_, vjp_fn = jax.vjp(f_rect, x0)
vjp_out = vjp_fn(u)[0]
print("Adjoint gap:", float(jnp.abs(jnp.dot(u, jvp_out) - jnp.dot(vjp_out, v))))
```

## Task 1: a rectangular derivative map

Use

\[
f(a,b)=\left(ab,\ \sin(a)+b^2,\ a^2-b\right).
\]

At \(x=(0.3,-0.6)\), use input direction \(v=(0.7,-0.2)\) and output cotangent \(u=(0.5,-0.8,1.2)\). Derive the full rectangular Jacobian. Verify the JVP, VJP and adjoint scalar against it, then repeat at a point of your choice with changed seeds.

For \(L(x)=\frac12\lVert f(x)\rVert^2\), derive and test a Hessian-vector product. Show the residual-weighted correction to Gauss–Newton. Compare against central differences of the gradient at two perturbations and discuss cancellation. Explain why matching dimensions in the lesson could obscure the input/output distinction that is visible here.

**Keep:** Jacobian derivation, shapes, forward/reverse products, adjoint equality, exact HVP and a justified numerical comparison.

## Task 2: compose a custom rule with a loss

Use the project's exact custom softplus to define

\[
Q(x)=\sum_i s(w_i x_i+c_i),\quad w=(2,-0.5,1.3),\quad c=(-1,0.2,0.7).
\]

Evaluate at \(x=(0.1,-0.4,0.8)\). Derive the gradient and diagonal Hessian analytically, including both chain-rule factors in the curvature. Verify both against JAX and finite differences. Repeat with a large common input magnitude and report which coordinates saturate; do not assert every gradient must become zero.

In a separate disposable diagnostic, replace the custom derivative coefficient by half its true value. Demonstrate that seed linearity still holds while the derivative disagrees with the primal finite difference. Explain why this is a wrong exact-derivative claim even if an optimization curve still decreases.

**Keep:** chain-rule derivation, stable values, slope/curvature checks, saturation interpretation and wrong-rule diagnosis.

## Task 3: add and validate one actual primitive rule

Extend your tiny interpreter to support `exp` with an explicit primal and tangent rule. Keep its rejection behavior for other unsupported operations. Trace

\[
G(z)=\sum_i \left(e^{z_i}+z_i\sin z_i\right).
\]

Use at least two input shapes, including a matrix, and deterministic random values restricted to \([-1,1]\). Derive the directional derivative as a whole-expression oracle. Compare your interpreted result with that oracle and JAX's JVP. Check tangent linearity at a fixed base point and test zero tangent.

Lower and compile the transformed computation. Evaluate changed values of the same shape, explicitly test rejection of a mismatched compiled signature, then build a new specialization for the second shape. Do not delegate your new rule to an autodiff transform.

Demonstrate that a function containing an unimplemented cosine primitive still raises an informative error. Explain the difference between your interpreter's unsupported primitive and a mathematically nondifferentiable function.

**Keep:** new rule, annotated trace, independent numerical comparisons, compiled outputs, and supported/unsupported contract tests.

## Task 4: explain the actual evidence

Create a derivative comparison figure for your extended transformation along a line through the input space. Identify axes, units or dimensionless quantities, direction, base point, actual representative values and whether any curves overlap. Explain why the changing derivative does not violate linearity in the tangent at a fixed point.

Inspect both the original and transformed lowered programs. Describe an observed structural difference that follows from your tangent rule. Avoid asserting speedup, memory reduction or portability from IR alone. If you add a benchmark, state compilation/warmup/synchronization boundaries, use repeated samples and label device and workload.

## Reviewer decision

- **Accept product reasoning:** The rectangular Jacobian, correct seed spaces and independent HVP checks agree. **Revise:** the transpose direction is guessed from equal shapes or Gauss–Newton is called the exact Hessian without its conditions.
- **Accept custom rule:** Stable primal, chain-rule factors, second derivatives and wrong-rule diagnosis are demonstrated. **Revise:** a surrogate is called exact or first-order agreement is treated as proof of higher-order behavior.
- **Accept transformation:** The exponential rule is explicitly implemented and tested on changed shapes/values with unsupported rejection retained. **Revise:** the interpreter delegates to JAX autodiff, silently falls back on unknown primitives or only works on the original constants.
- **Accept explanation:** Figures and compiler observations are tied to actual saved results with bounded claims. **Revise:** generic captions or IR counts replace interpretation and runtime evidence.

Submit code, report, numerical records and generated figure data. Public references are available; independent reasoning and inspectable changed-condition evidence remain necessary.
