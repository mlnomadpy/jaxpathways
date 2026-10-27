# Check your environment and a saved arithmetic result

This project asks two practical questions: which JAX backend ran your program, and can you reproduce a small result from the saved input? A device name in the output tells you what JAX discovered. It does not show that a TPU ran the computation.

## Before you start

Complete `welcome-01` and `welcome-02`. You should be able to open a terminal in the repository root and run a saved Python file. The project is CPU-only and uses JAX and NumPy from `requirements-cpu.txt`.

From the repository root, create and activate an environment, then install the course dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`; use `python` in place of `python3` in the checks. Edit `projects/welcome-audit/starter/model.py`. Run each command from the repository root:

```sh
python3 projects/welcome-audit/tests/check.py --implementation starter --stage 1
python3 projects/welcome-audit/tests/check.py --implementation starter --stage 2
python3 projects/welcome-audit/tests/check.py --implementation starter --stage 3 --evidence-dir runs/welcome-audit
python3 projects/welcome-audit/tests/check.py --implementation starter --stage 4
```

Stage 1 calls for the Gram matrix of \(X=[[1,2],[3,4]]\). Work it by hand: \(XX^T=[[5,11],[11,25]]\), so the checksum is 52. It also checks that JAX reports the CPU backend and the listed device count matches the runtime.

Stage 2 changes the input to \(X+I\). Predict the result before running: \([[2,2],[3,5]][[2,3],[2,5]]=[[8,16],[16,34]]\). If your report still contains the original matrix, check where you build the input. If the checksum differs, sum all four entries, including both off-diagonal terms.

Stage 3 writes a JSON receipt with the input, Gram matrix, checksum, backend and device inventory. Without `--evidence-dir`, the receipt stays in a temporary directory. Stage 4 reruns all checks together. Each stage prints `PASS welcome-audit stage N` when it succeeds.

The receipt supports reproducibility of this CPU calculation. It does not establish TPU compatibility or observed accelerator execution. The reference command is `python3 projects/welcome-audit/tests/check.py --implementation solution --stage 4`.

## Review handoff: explain a changed condition

Ask another person to reconstruct the experiment using only your evidence folder. Which environment and execution facts are actually recoverable?

**Keep for review:** Interpreter/package/device records, exact command, input and output. Mark unavailable facts as unknown; a successful screenshot is not a runnable handoff.

Write your prediction before opening the reference implementation. Distinguish what the existing checker verifies from your proposed extension; record any extension command, actual output and limitation separately. A reviewer should be able to locate the first failed contract from your saved evidence.
