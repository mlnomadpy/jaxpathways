# Internals authoring evidence — 2026-10-05

## Authored scope

Three missing canonical lesson sources are complete under the existing `phases/14-internals` paths. The existing `internals-01` source was not changed. All three new lessons have staged builds, independent numerical references, two executed experiments, a main transfer exercise, two distinct practice tasks, diagnosis, KaTeX and an executed figure with data-specific interpretation.

| Lesson       | Minutes | Concrete result                                                                                                                                                                            |
| ------------ | ------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| internals-02 |     105 | JVP/VJP products, independent Jacobian, adjoint identity, exact Hessian-vector product, Gauss–Newton comparison and symmetry checks                                                        |
| internals-03 |     120 | Actual exact custom_jvp/custom_vjp rules, stable primal/extreme inputs, seed linearity, higher-order derivatives, wrong-rule diagnosis and direct-JVP interface boundary                   |
| internals-04 |     130 | Actual restricted primitive-by-primitive forward derivative interpreter, constants/literals, multiple inputs/outputs, explicit rejection, lower/compile and changed-value numerical checks |

The interpreter does not delegate differentiation to JAX. It supports add, multiply, negate, sine and sum-reduction on a pure flat real-floating program. Its scope excludes effects, nested/control-flow subprograms, integer/complex differentiation, arbitrary pytrees and unsupported primitive/result/placement contracts. Rejections are part of the documented behavior.

## Root integration

For the three existing planned internals entries, preserve IDs/paths/titles/prerequisite chain, set status to `authored`, take the canonical source exercise, and replace placeholder metadata as follows:

| ID           | Objective                                                                                                   | Evidence                                                                                                                                                 | Check                                                                                            |
| ------------ | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| internals-02 | Interpret and independently verify forward, reverse and exact higher-order directional products.            | Hand-derived Jacobian/Hessian, adjoint scalar, changed-point checks, finite-difference HVP and a Gauss–Newton distinction.                               | Explain which space a JVP direction and a VJP cotangent belong to.                               |
| internals-03 | Define stable exact custom JVP/VJP rules and verify first-order, higher-order and transformation contracts. | Stable extreme inputs, analytic slopes/Hessians, seed linearity, independent finite differences and a deliberately wrong-rule diagnosis.                 | Explain why a linear finite custom derivative can still be mathematically wrong.                 |
| internals-04 | Implement a bounded derivative interpreter and compile its transformed numerical program.                   | Annotated primitive rules, constant/literal handling, independent expression oracles, changed compiled values and explicit unsupported/signature errors. | Explain why unsupported primitives must fail rather than silently return an unjustified tangent. |

Recommended phase metadata:

- `projectId`: `derivative-audit`
- `description`: `Connect traced programs to derivative rules and compiler stages. Verify exact directional products, write stable custom derivatives, and implement a small compiled transformation with explicit support boundaries.`
- `learningAdvice`: `Read the jaxpr first, derive a small Jacobian by hand, then audit custom rules and a primitive-level interpreter. Use independent numerical references before interpreting compiler structure.`

Add `projects/derivative-audit/project.json` to `curriculum/projects.json`. It explicitly requires `internals-01` through `internals-04`. Its three cumulative stages accept starter, solution or an implementation path.

Assessment registry entry:

```json
{
  "id": "internals",
  "projectId": "derivative-audit",
  "title": "Derivative and compiler synthesis",
  "status": "review-draft",
  "scope": "project-synthesis",
  "source": "assessments/internals.md",
  "url": "assessments/internals.html"
}
```

Update the internals route capstone to connect the authored project and public review-draft assessment (`projectId: derivative-audit`, `assessmentId: internals`). Suggested route description: `Trace a function into jaxpr, verify forward and reverse sensitivities, write exact custom derivative rules, and compile a small transformation whose limits you can explain.` The existing first annotated-jaxpr artifact remains appropriate.

## Executed verification

Tested environment: JAX 0.9.2, NumPy 2.4.4, CPU float64 explicitly enabled before array creation. No new dependencies or shared generator edits.

1. Executed each new lesson's complete example, experiments, main solution, both practice solutions and visual-data code sequentially in a separate Python process. All passed. Build-step code concatenates exactly to the complete example.
2. Rendered all non-code prose/math fields with the site's `inlineMath` / `renderMath`: 74, 75 and 74 fields initially passed, with the final prose changes also checked before handoff.
3. Regenerated all three final figures with `scripts/render-lesson-figures.py` loaded through importlib and `worker({'id': ..., 'path': ...})` in fresh CPU processes after final source changes. SVG, PNG and source-bound visual JSON are present. All three PNGs were inspected: axes/legends are readable, paired bars agree separately, softplus curvature peaks at zero, and the interpreted/analytic derivative curves overlap and change sign as described.
4. Ran `python3 projects/derivative-audit/tests/check.py --implementation solution --stage all`: all three stages passed. Tests cover three independent Jacobian/Hessian fixtures, stable custom first/second derivatives, two vector sizes, cotangent linearity and common shifts; random vector/matrix interpreter inputs, constant and literal tangents, multiple inputs/outputs, compiled changed values, malformed shapes/dtypes, integer inputs, effects and unsupported exponential rejection.
5. Ran the starter at stage 1: it failed at its intended NotImplementedError. It cannot be counted as an implemented project.
6. Independently executed assessment references for the rectangular three-output/two-input map, composed custom softplus chain rule, and an actual explicit exponential extension to the tiny interpreter on vector/matrix shapes, including lowering/compilation. All numerical comparisons passed. The full learner submission remains a review task; this is a public assessment draft, not independent expert approval.

Representative values:

- Original JVP `[-0.96, 1.52527319]`, VJP `[1.49212199, -2.6]`, adjoint scalar `2.5705463856`.
- Exact HVP `[1.74991568, -3.38303348]`; Gauss–Newton product `[2.07686964, -2.51938247]`. Their difference is the derived residual-weighted term.
- Softplus slope/curvature at zero: `0.5 / 0.25`; stable values at magnitude one thousand remain finite. At probe `0.4`, deliberately halved slope `0.29934383` disagrees with correct/finite-difference slope `0.59868766`.
- Tiny interpreted primal/tangent `2.3099803057 / 0.4391291800`; compiled outputs agree. Changed-value compiled tangent `0.5640324983`.
- The figure's directional derivative at displacements `-1, 0, 1` is about `-0.09853, 0.43913, 0.04102`. Its sampled maximum occurs at the middle point; prose reports the nonmonotone shape rather than generic agreement alone.
- Assessment rectangular-map JVP `[-0.48, 0.90873554, 0.62]`, VJP `[-0.34426919, -0.09]`, HVP `[2.3945446, -2.24269073]`.
- Composed-softplus assessment gradient `[0.62005104, -0.29934383, 1.10589319]`, curvature diagonal `[0.85563879, 0.06006519, 0.2146614]`.

## Version and review boundaries

Official custom derivative, ahead-of-time compilation, autodiff and jaxpr documentation was inspected. Current online jaxpr documentation describes a newer merged representation, while the installed JAX 0.9.2 uses the executed closed `.jaxpr` / `.consts` interface. The lesson explicitly identifies this version boundary and requires rechecking when upgrading. No compatibility with arbitrary future JAX internals is claimed.

Direct JVP of a custom_vjp function is intentionally rejected in the tested environment; reverse-over-reverse Hessian calculation succeeds for the implemented smooth exact rule. This distinction is taught and tested rather than hidden.

No full repository generators/build/check/smoke were run by this agent to avoid concurrent shared mutations. Root must register, generate, run all-script/all-notebook receipts and perform whole-course integration checks. No browser, TPU, GPU, multi-host, independent learner or external expert validation was performed. IR text length was printed only as an inspection artifact and is explicitly not a performance measure.
