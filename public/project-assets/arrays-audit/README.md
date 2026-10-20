# Audit masked batch moments

Padding can make a batch rectangular, but padded rows must not contribute to the mean or variance. A common mistake is to zero invalid rows and then divide by the batch size. That changes the denominator and biases the result.

## Before you start

Complete `arrays-01` through `arrays-04`. You should know that the input has shape \((batch, features)\) and the validity mask has shape \((batch,)\). Run from the repository root with the CPU requirements installed. Edit `projects/arrays-audit/starter/model.py`.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python3 projects/arrays-audit/tests/check.py --implementation starter --stage 1
python3 projects/arrays-audit/tests/check.py --implementation starter --stage 2
python3 projects/arrays-audit/tests/check.py --implementation starter --stage 3 --evidence-dir runs/arrays-audit
python3 projects/arrays-audit/tests/check.py --implementation starter --stage 4
```

On Windows PowerShell, activate `.venv\Scripts\Activate.ps1` and use `python` instead of `python3`.

Stage 1 uses rows \((1,3)\), \((5,7)\), and \((100,200)\), with the last row masked out. The valid-row mean is \((3,5)\). The population variance is \( ((-2)^2 + 2^2)/2 = 4 \) in each feature, so the vector is \((4,4)\), and count is 2. The extreme padded row must have no effect.

Stage 2 includes the extreme row and masks the middle row. Predict count 2, mean \((50.5,101.5)\), and variance \((2450.25,9702.25)\). The same stage rejects an empty mask and a mask whose length differs from the row count. If either passes, inspect the mask shape check and verify that the denominator is the number of valid rows, not the tensor's leading dimension.

Stage 3 writes count, mean and variance as JSON under `runs/arrays-audit/`. Without `--evidence-dir`, the file is temporary. Stage 4 reruns all three contracts. Passing output begins `PASS arrays-audit stage N`.

This verifies a small CPU reduction with synthetic values. It does not qualify a data pipeline or establish behavior on a TPU. The reference command is `python3 projects/arrays-audit/tests/check.py --implementation solution --stage 4`.
