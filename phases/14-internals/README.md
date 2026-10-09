# Phase 14: Autodiff & JAX internals

Specializations.

Trace functions into jaxpr, check forward and reverse sensitivities, write exact custom derivatives and compile a small transformation with explicit supported primitives.

Keep an independent derivative reference and test changed values, higher derivatives and unsupported cases before trusting a transformation.

**Prerequisites:** 04: Math & optimization.

**Hardware:** CPU.

## Study guide: Which mathematical operation does the transformed program represent?

Use internal representations to explain and verify transformations, not to memorize printed variable names. Separate primal values, tangent directions, cotangents and compiled artifacts.

### Check your starting point

A function maps an input vector to an output vector. Does a JVP multiply by the Jacobian or its transpose?

<details><summary>Compare your reasoning</summary>

A JVP maps an input tangent through the Jacobian. A VJP maps an output cotangent through its transpose. Check the spaces and the adjoint inner-product identity.

</details>

Review: [JVPs, VJPs, and higher-order derivatives](02-jvps-vjps-and-higher-order-derivatives/docs/en.md).

### Build in stages

1. **Read and check transformed programs.** Follow one jaxpr equation, compare directional derivatives independently and verify the adjoint identity at changed inputs.

   Lessons: [Read your first jaxpr](01-read-your-first-jaxpr/docs/en.md) · [JVPs, VJPs, and higher-order derivatives](02-jvps-vjps-and-higher-order-derivatives/docs/en.md).

2. **Enforce a custom transformation boundary.** Require tangent linearity and test higher-order promises. For the tiny interpreter, reject unsupported primitives rather than returning a plausible incomplete result.

   Lessons: [Custom derivative rules](03-custom-derivative-rules/docs/en.md) · [Lowering, compilation, and a tiny transformation](04-lowering-compilation-and-a-tiny-transformation/docs/en.md).

### Try a changed condition

A custom JVP returns the correct derivative when the tangent equals one but squares the tangent internally. Is it valid?

<details><summary>Compare an approach</summary>

No. Derivatives are linear maps in the direction at a fixed base point. Check multiple directions and linear combinations; agreement for one seed is insufficient.

</details>

**Symptom:** A derivative test passes at one input but fails after a shift.

**Check next:** Inspect captured constants, detached values and custom rules. Compare both primal and derivative behavior at changed base points.

### Decide what is ready

Use derivative-audit. Keep analytic/directional checks, the adjoint identity, supported higher-order behavior and explicit unsupported-program rejections.

### Further work

The tiny interpreter is intentionally incomplete. Production extension APIs, effects and full compiler transformation coverage require further targeted work.

## Lesson sequence

### 14.01 Read your first jaxpr

[Read the lesson](01-read-your-first-jaxpr/docs/en.md) · [Run the code](01-read-your-first-jaxpr/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Inspect a small transformed function and identify its operations.

**Evidence:** Save your annotated forward and derivative traces, offset-2 hand calculation, captured/explicit input comparison, unrolled/scan comparison and traced-branch error with both repaired numeric tests.

**Checkpoint:** A jaxpr has more equations after grad. What can you conclude from that observation alone?

### 14.02 JVPs, VJPs, and higher-order derivatives

[Read the lesson](02-jvps-vjps-and-higher-order-derivatives/docs/en.md) · [Run the code](02-jvps-vjps-and-higher-order-derivatives/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Compute and interpret JVP and VJP products with correct domains and shapes. Verify the adjoint dot-product identity against a hand-derived Jacobian.

**Evidence:** Hand-derived Jacobian/Hessian, adjoint scalar, changed-point checks, finite-difference HVP and a Gauss–Newton distinction. Keep the environment, observed outputs and your explanation of the figure.

**Checkpoint:** A VJP is seeded with an output weighting u. What does the returned input cotangent represent?

### 14.03 Custom derivative rules

[Read the lesson](03-custom-derivative-rules/docs/en.md) · [Run the code](03-custom-derivative-rules/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Implement actual custom_jvp and custom_vjp rules with correct return and residual contracts. Verify seed linearity, stable extreme values and second derivatives.

**Evidence:** Stable extreme inputs, analytic slopes/Hessians, seed linearity, independent finite differences and a deliberately wrong-rule diagnosis. Keep the environment, observed outputs and your explanation of the figure.

**Checkpoint:** A custom derivative is linear in its tangent and returns finite values. What additional evidence is necessary before calling it exact?

### 14.04 Lowering, compilation, and a tiny transformation

[Read the lesson](04-lowering-compilation-and-a-tiny-transformation/docs/en.md) · [Run the code](04-lowering-compilation-and-a-tiny-transformation/code/main.py)

Status: authored lesson with CPU exercise.

**Intended outcome:** Interpret literals, captured constants and variables in a supported flat jaxpr. Implement actual forward derivative rules without calling jax.jvp inside the interpreter.

**Evidence:** Annotated primitive rules, constant/literal handling, independent expression oracles, changed compiled values and explicit unsupported/signature errors. Keep the environment, observed outputs and your explanation of the figure.

**Checkpoint:** Why does the tiny interpreter raise an error for an unsupported primitive instead of calling that primitive directly?

## Phase project

Audit derivatives and a tiny transformation.

**Demonstrate:** Explain tracing and verify the derivative rule numerically.

Project status: implemented staged practice · [Open source](../../projects/derivative-audit/README.md). Copy `projects/derivative-audit/starter/model.py` to `projects/derivative-audit/my_model.py` and write your code in `projects/derivative-audit/my_model.py`. Run `python3 projects/derivative-audit/tests/check.py --implementation projects/derivative-audit/my_model.py --stage 1` from the top-level folder to verify all 3 stages.



[Primary documentation](https://docs.jax.dev/en/latest/).
