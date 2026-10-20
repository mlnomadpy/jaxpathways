# Separate compilation, execution and tracing evidence

First-call latency combines tracing, compilation and execution. Repeated calls can have different costs. This small CPU workload checks the numerical output, synchronizes each measurement, then calls the same jitted function at two matrix shapes. It retains a JAXPR and a JSON report. It does not measure device memory or provide a profiler trace.

## Before you start

Complete `performance-01` through `performance-05`. Know why asynchronous JAX work must be synchronized before timing. Use the CPU environment from `requirements-cpu.txt`, and edit `projects/performance-audit/starter/model.py`.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python3 projects/performance-audit/tests/check.py --implementation starter --stage 1
python3 projects/performance-audit/tests/check.py --implementation starter --stage 2
python3 projects/performance-audit/tests/check.py --implementation starter --stage 3 --evidence-dir runs/performance-audit
python3 projects/performance-audit/tests/check.py --implementation starter --stage 4
```

On Windows PowerShell, activate `.venv\Scripts\Activate.ps1` and use `python` instead of `python3`.

Stage 1 compares the result for the identity matrix against NumPy. Since \(I I=I\), the diagonal entries of \(\tanh(I)\) are about 0.761594 and the off-diagonal entries are zero. The first timing includes the first call; later timings are synchronized steady calls. Do not interpret one machine's microseconds as a general performance result.

Stage 2 invokes the same compiled function at shapes 16 by 16 and 32 by 32. The Python trace counter is read after both calls and before JAXPR capture, so it counts shape-driven function traces only. Expect two. The separate `make_jaxpr` artifact-generation call is excluded from this count.

Stage 3 writes `report.json` and `workload.jaxpr.txt` under `runs/performance-audit/`. The report explicitly says `memory_measured: false`. Stage 4 integrates the checks. Passing output begins `PASS performance-audit stage N`.

This records CPU first-call and repeated-call observations plus a symbolic program. It is not a hardware profile, a memory measurement, or TPU qualification. Reference command: `python3 projects/performance-audit/tests/check.py --implementation solution --stage 4`.
