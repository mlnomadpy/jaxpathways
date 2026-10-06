# Audit derivatives and compile a tiny transformation

Build a derivative audit that connects mathematics, JAX transformations and a small interpreter you can explain equation by equation. Complete `internals-01` through `internals-04` first. This is a CPU float64 teaching project with public fixtures; passing is not a claim that your interpreter supports all of JAX.

## Prepare your implementation

Use the course environment in `requirements-cpu.txt`. From the checkout or the extracted project's top folder:

```sh
cp projects/derivative-audit/starter/model.py projects/derivative-audit/my_model.py
python3 projects/derivative-audit/tests/check.py --implementation projects/derivative-audit/my_model.py --stage 1
```

PowerShell users can replace `cp` with `Copy-Item`. Keep the reference separate and attempt each stage before reading it.

## Stage 1: make directions and cotangents concrete

Implement `analyze` for the vector function specified in the starter. Return the original value, a JVP seeded in input space, a VJP seeded in output space, and the Hessian-vector product for half the squared output norm. Reject malformed input shapes.

Derive the Jacobian and Hessian independently. The checker changes the point and both seeds three times and compares every product. Explain why the exact Hessian contains a residual-weighted correction beyond the Jacobian-transpose-Jacobian term. Save one finite-difference verification and one direction-linearity check in your report.

## Stage 2: defend exact custom rules

Register an actual `custom_jvp` for stable scalar softplus and an actual `custom_vjp` for stable vector log-sum-exp. Preserve the primal values and implement the true derivatives. Explain what the forward rule saves and why the backward return is a tuple. Verify first derivatives, second derivatives, common-shift structure and linearity in directional seeds.

```sh
python3 projects/derivative-audit/tests/check.py --implementation projects/derivative-audit/my_model.py --stage 2
```

The tests include input magnitudes of one thousand, two logit vector sizes, a common large shift and a direct JVP rejection for the custom VJP function. A surrogate rule must not be presented as an exact derivative. Deliberately register a wrong but linear rule in a separate diagnostic file and show how a numerical comparison catches it.

## Stage 3: implement an actual derivative transformation

Implement `tiny_jvp(closed, primals, tangents)` using explicit local rules for `add`, `mul`, `neg`, `sin` and `reduce_sum`. Interpret captured constants and literals as fixed relative to explicit inputs. Return flat tuples of primal and tangent outputs. Do not delegate to JAX autodiff from inside this interpreter.

Enforce the supported subset: reject effects, unknown primitives, unsupported result/placement contracts, nonfloating inputs and mismatched shape/dtype signatures. Explain why a multi-output program is supported even though the allowed individual primitives each return one result.

```sh
python3 projects/derivative-audit/tests/check.py --implementation projects/derivative-audit/my_model.py --stage all
```

The checker uses deterministic random vectors and matrices, captured constants, two inputs, multiple outputs, analytic expression references and compiled changed-value evaluations. It also requires explicit rejection of an exponential primitive and invalid signatures. This is a boundary of your interpreter, not a claim that JAX cannot differentiate exponentials.

Save the original jaxpr, annotate each equation's primal/tangent rule, and inspect the transformed lowered program. Explain where Python executes during tracing and where the compiled numerical operations execute. Do not turn IR length or equation count into a speed claim.

## Evidence and reference

Submit implementation, audit script, predicted and observed results, precision/tolerances, environment, a deliberate failure/repair, supported-subset documentation, and an interpretation of your derivative comparison figure. Continue to the [internals synthesis assessment](../../assessments/internals.md) for a changed map and a new primitive rule.

```sh
python3 projects/derivative-audit/tests/check.py --implementation solution --stage all
```

The reference implements a narrow flat real-valued subset. It does not support control-flow subprograms, custom derivative primitives, integer tangents, arbitrary pytrees, complex conventions, effects or general compiler optimization. New support requires explicit semantics and independent checks.
