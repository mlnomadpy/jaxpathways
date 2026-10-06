# Release a model that another person can actually use

Status: planned release workflow for the [flagship capstone](README.md). The course should teach and execute this workflow once a qualified artifact exists. This planning task has not created a Hub repository, uploaded a checkpoint or changed any public model.

## Choose the consumer contract before export

The native training source is JAX/Flax. Define a supported native inference package with a pinned installation method, configuration schema and model loader. Hub hosting alone does not make arbitrary Flax NNX weights load through `AutoModel` or Sentence Transformers.

For a broadly usable text release, implement and verify a compatible Transformers/PyTorch encoder mapping and an appropriate Sentence Transformers module stack, or provide a clearly documented native loader. A standard Sentence Transformers package typically combines the encoder with explicit pooling and normalization; use its [custom-model documentation](https://sbert.net/docs/sentence_transformer/usage/custom_models.html) for the selected version. Do not claim compatibility merely because weight files have familiar names.

The portable path must reproduce the architecture, RoPE/attention conventions, normalization, tokenization, pooling and output normalization. Reuse BRIDGE's layerwise errors and compare gradients where training portability is promised. If custom code is required, disclose it, pin/review it and explain the loader requirement; prefer a supported standard implementation when it matches the model accurately.

## Assemble a reviewed release directory

Keep this directory separate from training checkpoints, raw data and tracking storage. Include only files needed for the declared release:

| Artifact | What it must describe or contain |
| --- | --- |
| Model card (`README.md`) | Task/use cases, architecture/actual counts, languages/domains, limitations, phase recipes, data attribution, compute, evaluation coverage, contamination disclosure and intended consumer examples |
| License and attribution | Selected model license and notices consistent with the actual sources/dependencies; unsupported rights claims cannot be filled by a template |
| Configuration | Exact encoder, positions, lengths, dtypes, normalization, pooling, embedding dimension and query/document conventions |
| Tokenizer or modality processor | Actual files needed for frozen token IDs or image/audio preprocessing, including special-token/normalization metadata |
| Inference weights | Verified safe serialization, shard index where needed, tied-weight handling and hashes; strip optimizer/MLM-only state from the inference variant deliberately |
| Consumer integration | Tested native loader and/or Transformers/Sentence Transformers configuration, pinned dependencies and minimal executable example |
| Evaluation records | Frozen task manifest, raw permitted results, aggregation/protocol, baselines and device/precision metadata |
| Provenance | Source/config/data/tokenizer/checkpoint hashes, parent phase identities and export/conversion version |
| Canary inputs/outputs | Small redistributable ordinary, Unicode, boundary and modality-specific cases with actual tolerance policy |
| Release manifest | File inventory/hashes, supported variants, quality/system gate receipts, decision and model/index compatibility version |

Use [Safetensors](https://huggingface.co/docs/safetensors/index) for supported portable tensor weights where applicable; it does not replace architecture/configuration metadata. Preserve complete resumable training checkpoints separately and state whether they are released. Quantized variants require their own scales/layout/kernel contract, loader and evaluation; renaming float weights does not create an INT8/INT4 release.

Follow the [Hub model-card format](https://huggingface.co/docs/hub/main/en/model-cards), including appropriate YAML metadata for library/task/language/license/datasets. Write results from observed receipts. Distinguish planned benchmark coverage, measured scores, observed hardware and untested capabilities. No invented model scores, release identifiers or compute figures belong in the card.

## Pre-upload qualification

1. Load the reviewed directory in a clean process/environment using each advertised consumer path. Keep the exact install command and version lock; ensure examples do not import the training workspace implicitly.
2. Compare raw input → tokenizer/processor → intermediate layers → pooled normalized embeddings against the native checkpoint. Cover multiple batches/lengths, changed inputs, Unicode, padding and empty/overlong policies. Verify ordered output and deterministic evaluation behavior under the stated tolerance.
3. Run the declared retrieval/benchmark and precision gates using the **serialized release**, not an in-memory training model. Test task prefixes, similarity convention and advertised output dimensions.
4. Test corruption/missing-shard/config mismatch rejection and ensure the package contains no tokens, credentials, private records, local absolute-path dependencies or unapproved tracking logs.
5. Run actual container/service canaries, load tests and recovery. Bind evidence to model, processor, container and index identities. Only advertise runtime/format combinations that passed.

## Hugging Face publication lesson

Choose the account/organization, repository name, visibility and access policy for the concrete release. Authentication is an interactive/user-managed step or a scoped credential supplied through the supported environment; never place credentials in notebooks, configs or command logs. The course should explain the difference between preparing a package, uploading a private candidate, making a public release and submitting benchmark results.

The following is a **documentation example for the later upload stage** using the [official Hub upload API](https://huggingface.co/docs/huggingface_hub/guides/upload), not a command executed by this plan:

```python
from huggingface_hub import HfApi, snapshot_download

# Replace these only after the package and destination are reviewed.
repo_id = "YOUR_NAMESPACE/YOUR_EMBEDDING_MODEL"
release_dir = "./release-reviewed"
api = HfApi()  # Uses the user's configured authentication; no embedded token.

# Start with a NEW private candidate repository.
# For an existing repository, inspect its identity/visibility before uploading.
api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=False)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=release_dir,
    commit_message="Upload qualified embedding release candidate",
)
revision = api.repo_info(repo_id=repo_id, repo_type="model").sha
downloaded = snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    revision=revision,
    local_dir="./downloaded-release-candidate",
)
print("Candidate revision:", revision)
print("Downloaded package:", downloaded)
```

Use a serialized release process with a single publishing writer; the retrieved revision must match the candidate manifest. Pin API versions when this stage is implemented. Uploading may involve multiple commits/files, so partial upload success is not release success. After uploading, verify the full file inventory/hashes at the immutable revision and rerun the clean consumer/embedding/retrieval canaries from the download. Use a fresh environment/cache for the cold-load drill.

Keep the candidate private until its downloaded bytes and documentation pass the release gates and the owner chooses publication. When publishing, record the immutable commit and a release tag, publish tested consumer examples and retain the previous qualified revision. Repository creation/access and later visibility changes use the [repository-management guide](https://huggingface.co/docs/huggingface_hub/guides/repository); reproducible loading uses the [download guide](https://huggingface.co/docs/huggingface_hub/guides/download). Moving a mutable branch pointer is not a reproducible model reference.

## Index migration and operation

Document which model/processor/pooling/normalization/dimension/precision created stored document embeddings. For a changed embedding space, build a new index, compare exact-search and ANN canaries, qualify query encoding against that index and shift traffic under an observed rollout. Retain an old compatible model/index pair for rollback. Model-card updates and Hub commits do not automatically deploy a service or rebuild an index.

The release exercise ends when a clean consumer can download the pinned revision, reproduce embeddings within the documented budget, reproduce the declared evaluation protocol, and operate or roll back the matching retrieval system. A public URL alone is not the final success criterion.
