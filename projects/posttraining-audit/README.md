# Check response-only SFT and a preference pair

The supervised objective should count response tokens and exclude prompt tokens. The preference loss should increase when the preferred response becomes less preferred relative to its alternative. This project uses two short synthetic logits sequences so each result can be checked by hand.

## Before you start

Complete the response-masking and preference-optimization lessons. You should know log-softmax, token masks and the chosen-minus-rejected log-probability margin. Install the CPU course dependencies and edit `projects/posttraining-audit/starter/model.py` from the repository root.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python3 projects/posttraining-audit/tests/check.py --implementation starter --stage 1
python3 projects/posttraining-audit/tests/check.py --implementation starter --stage 2
python3 projects/posttraining-audit/tests/check.py --implementation starter --stage 3 --evidence-dir runs/posttraining-audit
python3 projects/posttraining-audit/tests/check.py --implementation starter --stage 4
```

On Windows PowerShell, activate `.venv\Scripts\Activate.ps1` and use `python` instead of `python3`.

Stage 1 masks out the first and third token. For the selected token, the target logit is 3 and the other logit is 0, so the SFT loss is \(\log(1+e^{-3})\), about 0.0486. Chosen and rejected policy log probabilities are \(-0.7\) and \(-1.0\). With an equal-probability frozen reference, its log-probability difference is zero. Their margin is 0.3; with beta 0.1 the DPO loss is \(-\log(\sigma(0.03))\), about 0.678.

DPO uses the reference-corrected margin \(m=(\log\pi_c-\log\pi_r)-(\log\pi_{ref,c}-\log\pi_{ref,r})\). Stage 2 also checks a nonuniform reference: reference log probabilities \(-0.4\) and \(-1.4\) turn the corrected margin into \(-0.7\), even though the policy alone favors the chosen response.

Stage 2 swaps the two policy preference scores. The margin becomes -0.3 and the DPO loss must rise. An empty response mask must raise a clear error. If the SFT loss changes when only masked prompt logits change, inspect the mask before reduction.

Stage 3 saves response-token count, SFT loss, margin and DPO loss to `runs/posttraining-audit/posttraining-audit-evidence.json`. Without `--evidence-dir`, the file is temporary. Stage 4 runs the full check. Passing output begins `PASS posttraining-audit stage N`.

These are objective mechanics on synthetic values. They do not demonstrate a trained language model, preference-data quality, retained behavior or user-facing response quality. Reference command: `python3 projects/posttraining-audit/tests/check.py --implementation solution --stage 4`.
