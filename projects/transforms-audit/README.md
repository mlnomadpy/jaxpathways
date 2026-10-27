# Compare scalar, vectorized and compiled functions

This project checks whether three ways of applying \(f(x)=\sin(x)+x^2\) agree: a Python loop, `jax.vmap`, and `jax.jit(jax.vmap(...))`. It also checks the derivative \(f'(x)=\cos(x)+2x\). A successful call on one batch shape does not prove that a changed shape reuses the same trace.

## Before you start

Complete `first-gradient` and `transforms-02` through `transforms-05`. Be comfortable with pure functions and the difference between mapping values and tracing a function. Run from the repository root with `requirements-cpu.txt` installed. Edit `projects/transforms-audit/starter/model.py`.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python3 projects/transforms-audit/tests/check.py --implementation starter --stage 1
python3 projects/transforms-audit/tests/check.py --implementation starter --stage 2
python3 projects/transforms-audit/tests/check.py --implementation starter --stage 3 --evidence-dir runs/transforms-audit
python3 projects/transforms-audit/tests/check.py --implementation starter --stage 4
```

On Windows PowerShell, activate `.venv\Scripts\Activate.ps1` and use `python` instead of `python3`.

Stage 1 compares the vectorized and loop values, then checks the analytic derivative at four inputs. At \(x=0.7\), for example, the derivative is \(1.4+\cos(0.7)\), about 2.165. If value checks pass while derivative checks fail, inspect the argument to `jax.grad` and the axis used by `vmap`.

Stage 2 changes the values and adds a fifth element. The new output must equal \(\sin(x)+x^2\); the compiled function should be traced once for the original shape and once for the extended shape. Do not claim that the function is shape-polymorphic from this result.

Stage 3 saves the JAXPR as `runs/transforms-audit/transforms-audit.jaxpr.txt`. Read the `sin`, multiply and addition operations to connect the traced program to the formula. Without `--evidence-dir`, the artifact is temporary. Stage 4 integrates all checks. Passing output begins `PASS transforms-audit stage N`.

The JAXPR and comparisons are local CPU evidence. The example makes no timing or TPU compilation claim. Reference: `python3 projects/transforms-audit/tests/check.py --implementation solution --stage 4`.

## Review handoff: explain a changed condition

Duplicate every example and compare the sum-loss gradient with the mean-loss gradient before applying any optimizer.

**Keep for review:** An independent per-example gradient sum and the two reductions. The sum doubles, while the mean remains unchanged for exact duplicated examples.

Write your prediction before opening the reference implementation. Distinguish what the existing checker verifies from your proposed extension; record any extension command, actual output and limitation separately. A reviewer should be able to locate the first failed contract from your saved evidence.
