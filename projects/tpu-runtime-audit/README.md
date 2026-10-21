# Separate CPU runtime evidence from TPU claims

The project observes the backend and devices available to the process, then demonstrates a precise storage and operand conversion. The weight matrix is stored as BF16 and converted back to FP32 before the matrix product. This checks a dtype path, not a native TPU BF16 kernel or TPU accumulation behavior.

## Before you start

Complete the TPU runtime lessons and the local CPU setup in `requirements-cpu.txt`. Run from the repository root and edit `projects/tpu-runtime-audit/starter/model.py`.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python3 projects/tpu-runtime-audit/tests/check.py --implementation starter --stage 1
python3 projects/tpu-runtime-audit/tests/check.py --implementation starter --stage 2
python3 projects/tpu-runtime-audit/tests/check.py --implementation starter --stage 3 --evidence-dir runs/tpu-runtime-audit
python3 projects/tpu-runtime-audit/tests/check.py --implementation starter --stage 4
```

On Windows PowerShell, activate `.venv\Scripts\Activate.ps1` and use `python` instead of `python3`.

Stage 1 checks that JAX reports the CPU backend and that the actual device list has the same length as `jax.device_count()`. It also checks the small matrix result after BF16 weight storage and FP32 conversion. A passing dtype assertion does not mean the dot product ran in BF16.

Stage 2 sums simulated participant values \([1,2,3]\). The full sum is 6; omitting participant 1 gives 4. The function is ordinary CPU arithmetic, not a process group or TPU collective. Invalid participant indexes must be rejected.

Stage 3 retains backend, device names, storage dtype and operand/result dtype in `runs/tpu-runtime-audit/tpu-runtime-audit-evidence.json`, with a scope statement. Stage 4 integrates all checks. Without `--evidence-dir`, stage evidence is temporary. Passing output begins `PASS tpu-runtime-audit stage N`.

Only CPU discovery and local arithmetic are observed. TPU topology, native BF16 execution and accelerator collectives need separate execution evidence. Reference command: `python3 projects/tpu-runtime-audit/tests/check.py --implementation solution --stage 4`.

## Review handoff: explain a changed condition

Review a receipt with four local devices but missing machine type and topology. Which hardware claim remains unknown?

**Keep for review:** Observed backend/device kinds plus requested machine type, process count and local/global device counts. Do not universally equate devices and chips or infer performance from arithmetic.

Write your prediction before opening the reference implementation. Distinguish what the existing checker verifies from your proposed extension; record any extension command, actual output and limitation separately. A reviewer should be able to locate the first failed contract from your saved evidence.
