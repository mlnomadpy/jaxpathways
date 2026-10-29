# C01 frozen-recipe history

The first pre-training recipe bytes are preserved in
[`recipe-history/c01-original-pretraining-recipe.json`](recipe-history/c01-original-pretraining-recipe.json), SHA-256
`c74bf0e06cc6dc6f8972c6fecae6c86ed03623f60cfa2b91cff9e141e002e5a3`.
The data, membership, model, objectives, budgets, optimizer resets, parent
choice, branch, and precommitted quality floors in the recipe have not changed.

Before the successful complete offline pilot, the pinned MiniLM baseline's
observed `1_Pooling/config.json` was found to include
`word_embedding_dimension: 384`, which the recipe's pooling metadata had
omitted. This metadata-only correction produced the recipe snapshot consumed by
the successful pilot, SHA-256
`08e90b0b93922cdf1b95b6b7d9477d899ddc2da9f6d99a06d5ee991bbea70a69`.
The exact snapshot is retained in the successful run output as
`frozen-recipe-input.json`.

After that run, independent review demonstrated that a prepared claim could be
changed while recomputing its local manifest. The current frozen recipe adds
hashes for all normalized output files, and `verify_frozen_data` compares these
against the project recipe. This is a data-integrity amendment; it does not
change any training or selection decision. The current recipe hash is
`70649244860d731f92d09e805b025e05a731dbef90463d62b0428d0cb7e21134`.
Any further run uses and snapshots that current recipe before optimization.
