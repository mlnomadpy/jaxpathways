# Implement the encoder before training

The stage checker imports this file when you pass `--implementation starter`.
An unfinished function intentionally fails instead of reporting a zero-score
pass. Work from the parameter-tree contract in `model.py` and the frozen
configuration; keep the checked-in reference at the project root available for
comparison, but make your checker command point to the starter while working.

Implement the model APIs in this order:

1. `initialize` and `parameter_count`: create bias-free Q/K/V/O and two-matrix
   GELU blocks, affine layer norms, one token matrix, RoPE and the tied MLM
   transform. `TARGET_CONFIG` is abstractly counted by the checker, so avoid
   allocating a full target parameter array.
2. `hidden_states`: attention must be bidirectional and must mask invalid keys
   and queries. Keep examples in distinct batch rows so one document cannot
   attend to another.
3. `pool`: exclude PAD, BOS, EOS, UNK and MASK tokens, average content states,
   normalize nonzero rows, and return zeros when no content remains.
4. `mlm_logits` and `masked_token_loss`: transform the selected hidden states,
   reuse the token table for vocabulary logits, and report the selected-token
   sum and count separately from the mean.

Use the independent controls in `tests/check.py`: a future-token edit should
change an earlier state, while padding extension and changes to a separate
batch row should not. Masked-token sums/counts and multi-positive gradients are
checked against separate NumPy calculations. The data builder also checks
held-out groups and source hashes before it creates any batches.
