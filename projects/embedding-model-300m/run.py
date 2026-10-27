"""Run the frozen SciFact CPU lifecycle pilot into a caller-owned output path."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import EncoderConfig, TrainConfig
import evaluation
from data import dataset
from baseline import MODEL_ID, REVISION, encode as encode_baseline, load_pinned_baseline
from trainer import (
    identity_for,
    initialize_state,
    save_checkpoint,
    train_human_contrastive,
    train_mlm,
    train_weak_contrastive,
)
from plots import loss_figure, retrieval_figure


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _file_hash(path):
    return _sha(Path(path).read_bytes())


def _write_measured_model_card(package_dir, report, archive_sha256):
    """Replace the generic package README with the measured, rights-aware card."""
    final_suffix = "/final_official_dev_custom_protocol"
    methods = (
        ("Phase 3 human-evidence pilot", "phase3_human_pairs"),
        ("Phase 1 MLM", "phase1_mlm"),
        ("Frozen MiniLM-L6-v2 baseline", "external_minilm_l6_v2"),
        ("BM25 lexical baseline", "bm25"),
    )
    rows = []
    for label, name in methods:
        metrics = report["evaluation"].get(name + final_suffix)
        if metrics is not None:
            rows.append(
                "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                    label,
                    metrics["recall@1"],
                    metrics["recall@10"],
                    metrics["mrr"],
                    metrics["ndcg@10"],
                )
            )
    gates = report["quality_gates"]
    baseline = report["baseline"]
    exposure = report["exposure"]
    measured = report["measured_training_exposure"]
    phase1_exposure = measured["phase1_mlm"]
    phase2_exposure = measured["phase2_contrastive_plus_retained_mlm"]
    phase3_exposure = measured["phase3_human_pairs"]
    floors_pass = bool(gates["phase3_final_floors_pass"])
    gate_status = "has passed" if floors_pass else "has not passed"
    gate_word = "passed" if floors_pass else "failed"
    card = f"""# SciFact C01 encoder — measured local model card

## Status and intended use

This is an in-development, CPU-trained two-layer pilot for educational SciFact evidence-retrieval experiments. It is not the proposed 300M model and {gate_status} its precommitted pilot retrieval floors. Regardless of that pilot result, it is not a qualified biomedical search system, general-purpose embedding model, or production candidate. The custom local JAX package API is described in `package.json`; this is not a `transformers.AutoModel` or Sentence Transformers class.

## Measured evaluation

The evaluation used all 5,183 SciFact abstracts as candidates and all 300 official development claims as a fixed custom final inventory. Metrics include the 188 claims with documented human evidence document IDs; the remaining 112 are listed as unjudged and excluded from metric denominators. Relevance comes only from the original evidence annotations. The official test split has no public judgments.

| System | Recall@1 | Recall@10 | MRR | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Precommitted pilot floors were Recall@10 >= {gates['precommitted_floors']['recall@10']:.2f} and nDCG@10 >= {gates['precommitted_floors']['ndcg@10']:.2f}. The phase-3 pilot **{gate_word}** these floors. This is one deterministic seed and a small architecture; it gives no estimate of between-seed uncertainty. The fixed held-out corpus was rerun for engineering verification and is not a blinded benchmark. MTEB, expert review, broad-domain quality, target-scale training, target-runtime evaluation, and service qualification remain unrun.

The frozen external comparator is `{baseline['model_id']}` at revision `{baseline['revision']}`. It reports Apache-2.0, but its model metadata lists S2ORC and MS MARCO and possible SciFact overlap is unresolved; it is contextual, not contamination-free evidence.

## Architecture and input behavior

The executed pilot has {report['config']['layers']} bidirectional pre-LN Transformer layers, hidden width {report['config']['width']}, {report['config']['heads']} attention heads, GELU two-matrix MLPs, RoPE positions, tied MLM embeddings, and float32 weights. The bundled byte-level BPE tokenizer adds BOS/EOS, truncates at {report['config']['max_length']} tokens, mean-pools content tokens only, and L2-normalizes. Empty-content inputs map to zero vectors. Query/document prefixes are not used. These describe the measured pilot, not the target architecture.

## Data provenance and rights

The source is the pinned AllenAI SciFact release, archive SHA-256 `{archive_sha256}`. Its claims and evidence annotations are CC BY 4.0; the scientific abstract corpus originates from S2ORC under ODC-By 1.0; upstream code is Apache-2.0. See the bundled `DATA_PROVENANCE.md` for attribution and source links. The package assigns no separate model-weight license; do not redistribute it until source-rights implications have been reviewed.

The prepared MLM inventory contained {exposure['mlm_prepared_documents']} documents and {exposure['mlm_prepared_nonpadding_token_count']} nonpadding token positions; these are prepared input totals, not consumed totals. Phase 1 actually consumed {phase1_exposure['unique_documents']} distinct documents ({phase1_exposure['nonpadding_input_tokens']} nonpadding input tokens; {phase1_exposure['selected_masked_tokens']} selected masked-token positions). The joint phase-2 branch consumed {phase2_exposure['unique_retained_mlm_documents']} distinct retained-MLM documents ({phase2_exposure['nonpadding_retained_mlm_input_tokens']} nonpadding tokens; {phase2_exposure['selected_retained_mlm_tokens']} selected masked-token positions) alongside {phase2_exposure['unique_weak_pairs']} weak pairs. Phase 3 consumed {phase3_exposure['unique_fit_queries']} distinct human-evidence fit queries, {phase3_exposure['unique_positive_documents']} positive documents, {phase3_exposure['nonpadding_query_input_tokens']} nonpadding query tokens, and {phase3_exposure['nonpadding_candidate_document_input_tokens']} nonpadding candidate-document tokens. Exact consumed document/query ID inventories and per-query metrics are in bundled `MEASURED_EVIDENCE.json`.

Recipe-development query text and final query text did not enter training pairs. Development-component document text entered the eligible tokenizer corpus ({exposure['recipe_development_documents_in_tokenizer']} documents; {exposure['recipe_development_documents_in_mlm_sample']} in the prepared MLM inventory), but the actual phase-1 and retained-MLM consumed subsets contained {phase1_exposure['recipe_development_documents_consumed_count']} and {phase2_exposure['recipe_development_documents_consumed_count']} such documents respectively. The final-connected document group and final query text were excluded from training. This is a disclosed transductive setup, not an unseen-document estimate.

## Reproducibility

The external source-bound receipt binds the exact candidate, recipe, data outputs, tokenizer, environment, checkpoints, report, and run artifacts. This README is generated from measured report values and its SHA-256 is recorded in `package.json`. Pinned MiniLM weights are not bundled; their revision and file hashes are recorded in the separate baseline receipt.
"""
    package_dir = Path(package_dir)
    readme = package_dir / "README.md"
    readme.write_text(card)
    metadata_path = package_dir / "package.json"
    metadata = json.loads(metadata_path.read_text())
    provenance = package_dir / "DATA_PROVENANCE.md"
    provenance.write_text(f"""# Data provenance and rights

This package was trained in part on the official [AllenAI SciFact release](https://github.com/allenai/scifact). The exact source archive SHA-256 is `{archive_sha256}`. The upstream [SciFact license terms](https://raw.githubusercontent.com/allenai/scifact/master/LICENSE.md) separate CC BY 4.0 claims/evidence annotations from the scientific abstract corpus sourced from S2ORC under ODC-By 1.0; the original code is Apache-2.0. See the [SciFact data schema](https://raw.githubusercontent.com/allenai/scifact/master/doc/data.md) and [paper](https://arxiv.org/abs/2004.14974). Evidence document IDs, not cited-document metadata, supply retrieval positives.

The C01 package does not assert a standalone license for trained model weights. Source-data permissions and attribution implications must be reviewed before redistribution. Full source member hashes, pinned upstream revision, split inventories, and normalized output hashes are recorded in the external source-bound receipt and frozen project recipe. The pinned MiniLM baseline is a separate Apache-2.0-reported model with possible S2ORC/MS MARCO overlap; it is not bundled into this package.
""")
    measured_evidence = {
        "protocol": report["protocol"],
        "config": report["config"],
        "quality_gates": report["quality_gates"],
        "measured_training_exposure": measured,
        "evaluation": {
            key: value
            for key, value in report["evaluation"].items()
            if key.endswith(final_suffix)
        },
        "baseline": {
            key: baseline[key]
            for key in (
                "model_id",
                "revision",
                "reported_license",
                "weights_sha256",
                "tokenizer_sha256",
                "possible_overlap",
            )
        },
        "frozen_data_manifest_sha256": report["frozen_data_manifest_sha256"],
        "frozen_recipe_input_sha256": report["frozen_recipe_input_sha256"],
        "archive_sha256": archive_sha256,
    }
    evidence_path = package_dir / "MEASURED_EVIDENCE.json"
    _save_json(evidence_path, measured_evidence)
    metadata["model_card_sha256"] = _file_hash(readme)
    metadata["data_provenance_sha256"] = _file_hash(provenance)
    metadata["measured_evidence_sha256"] = _file_hash(evidence_path)
    _save_json(metadata_path, metadata)
    return metadata["model_card_sha256"]


def load_implementation(selection):
    value = str(selection)
    if value in {"solution", "reference", "model"}:
        path = ROOT / "model.py"
    elif value == "starter":
        path = ROOT / "starter" / "model.py"
    else:
        path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"candidate implementation not found: {path}")
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["model"] = module
    spec.loader.exec_module(module)
    required = (
        "initialize",
        "parameter_count",
        "hidden_states",
        "mlm_logits",
        "pool",
        "masked_token_loss",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise AssertionError(f"candidate is missing required model APIs: {missing}")
    return module, path


def _pad_rows(rows, max_length, pad_id):
    values = np.full((len(rows), max_length), pad_id, np.int32)
    for index, row in enumerate(rows):
        ids = row["ids"] if isinstance(row, dict) else row
        if len(ids) > max_length:
            raise ValueError("encoded sequence exceeds frozen maximum length")
        values[index, : len(ids)] = ids
    return values


def _hash_source_tree(candidate_path):
    files = [
        ROOT / "config.py",
        ROOT / "model.py",
        ROOT / "objectives.py",
        ROOT / "trainer.py",
        ROOT / "evaluation.py",
        ROOT / "baseline.py",
        ROOT / "plots.py",
        ROOT / "package_runtime.py",
        ROOT / "run.py",
        ROOT / "data" / "prepare.py",
        ROOT / "data" / "dataset.py",
        ROOT / "data" / "frozen_recipe.json",
        ROOT / "data" / "RECIPE_AMENDMENTS.md",
        ROOT / "data" / "recipe-history" / "c01-original-pretraining-recipe.json",
        ROOT / "data" / "LICENSES.md",
        ROOT / "requirements-embedding.txt",
        ROOT / "requirements-embedding-lock.txt",
        ROOT / "tests" / "check.py",
        ROOT / "tests" / "resume_worker.py",
        Path(candidate_path),
    ]
    hashes = {}
    for path in sorted(set(files)):
        if not path.is_file():
            continue
        try:
            name = str(path.relative_to(ROOT))
        except ValueError:
            name = f"candidate_external/{path.name}"
        hashes[name] = _file_hash(path)
    return hashes


def _dataset_digest(dataset_obj, selected_doc_ids, tokenizer_info):
    value = {
        "source_archive_sha256": dataset_obj["manifest"]["source"]["archive_sha256"],
        "fit_file_sha256": dataset_obj["manifest"]["outputs"][
            "claims_fit.jsonl_sha256"
        ],
        "recipe_dev_file_sha256": dataset_obj["manifest"]["outputs"][
            "claims_recipe_dev.jsonl_sha256"
        ],
        "final_file_sha256": dataset_obj["manifest"]["outputs"][
            "claims_final.jsonl_sha256"
        ],
        "selected_mlm_doc_ids": list(map(int, selected_doc_ids)),
        "processor_training_doc_ids": tokenizer_info["training_doc_ids"],
    }
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _pair_order(pairs, limit=32):
    keyed = []
    for pair in pairs:
        text = f"scifact-c01-weak-pair-20261021:{pair['group_id']}:{pair['source']}:{pair['left']}\0{pair['right']}"
        keyed.append((_sha(text.encode()), pair))
    return [pair for _, pair in sorted(keyed, key=lambda item: item[0])[:limit]]


def _save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def _rank_population(
    name, population, model, params, config, documents, tokenizer, output_dir
):
    embeddings = evaluation.encode_model(
        model, params, config, documents, population, tokenizer
    )
    rankings, scores = evaluation.exact_rank(
        embeddings["query_vectors"],
        embeddings["document_vectors"],
        embeddings["query_ids"],
        embeddings["document_ids"],
    )
    metrics = evaluation.metrics_from_rankings(rankings, population)
    folder = Path(output_dir) / "raw" / name
    folder.mkdir(parents=True, exist_ok=True)
    qrels = {
        str(int(row["id"])): sorted(map(int, row.get("evidence", {}).keys()))
        for row in population
    }
    np.savez_compressed(
        folder / "retrieval-arrays.npz",
        query_vectors=embeddings["query_vectors"],
        document_vectors=embeddings["document_vectors"],
        query_ids=embeddings["query_ids"],
        document_ids=embeddings["document_ids"],
        scores=scores.astype(np.float32),
        ranked_doc_ids=np.asarray(
            [[rank for rank in rankings[int(qid)]] for qid in embeddings["query_ids"]],
            np.uint32,
        ),
        query_truncated_tokens=np.asarray(
            [embeddings["query_truncated_tokens"]], np.int64
        ),
        document_truncated_tokens=np.asarray(
            [embeddings["document_truncated_tokens"]], np.int64
        ),
    )
    _save_json(folder / "qrels.json", qrels)
    _save_json(folder / "metrics.json", metrics)
    return metrics


def _save_rankings_from_vectors(
    name, population, query_vectors, doc_vectors, query_ids, doc_ids, output_dir
):
    rankings, scores = evaluation.exact_rank(
        query_vectors, doc_vectors, query_ids, doc_ids
    )
    metrics = evaluation.metrics_from_rankings(rankings, population)
    folder = Path(output_dir) / "raw" / name
    folder.mkdir(parents=True, exist_ok=True)
    qrels = {
        str(int(row["id"])): sorted(map(int, row.get("evidence", {}).keys()))
        for row in population
    }
    np.savez_compressed(
        folder / "retrieval-arrays.npz",
        query_vectors=np.asarray(query_vectors, np.float32),
        document_vectors=np.asarray(doc_vectors, np.float32),
        query_ids=np.asarray(query_ids, np.int64),
        document_ids=np.asarray(doc_ids, np.int64),
        scores=scores.astype(np.float32),
        ranked_doc_ids=np.asarray(
            [[rank for rank in rankings[int(qid)]] for qid in query_ids], np.uint32
        ),
    )
    _save_json(folder / "qrels.json", qrels)
    _save_json(folder / "metrics.json", metrics)
    return metrics


def _cold_package_check(
    model_module,
    implementation_path,
    params,
    config,
    tokenizer,
    sentinel_queries,
    sentinel_docs,
    output,
):
    from package_runtime import save_package

    package_dir = Path(output) / "local-package"
    save_package(
        params,
        config,
        tokenizer,
        package_dir,
        model_module=model_module,
        implementation_path=implementation_path,
    )
    from package_runtime import JAXSentenceEncoder

    encoder = JAXSentenceEncoder.from_pretrained(package_dir, local_files_only=True)
    current = encoder.encode(sentinel_queries + sentinel_docs)
    packed = []
    for text in sentinel_queries + sentinel_docs:
        tokens = tokenizer.encode(text, add_special_tokens=False).ids
        tokens = [config.bos_id, *map(int, tokens), config.eos_id]
        if len(tokens) > config.max_length:
            tokens = tokens[: config.max_length - 1] + [config.eos_id]
        packed.append(tokens)
    padded = np.full(
        (len(packed), max(map(len, packed))), config.pad_id, dtype=np.int32
    )
    for index, tokens in enumerate(packed):
        padded[index, : len(tokens)] = tokens
    candidate_vectors = np.asarray(
        model_module.pool(params, jnp.asarray(padded), config), np.float32
    )
    np.testing.assert_allclose(current, candidate_vectors, rtol=1e-5, atol=1e-6)
    if encoder.package_metadata["source_files_sha256"]["model.py"] != _file_hash(
        implementation_path
    ):
        raise AssertionError(
            "exported package did not bind the selected candidate source"
        )
    child_dir = Path(output) / "cold-process"
    child_dir.mkdir(parents=True, exist_ok=True)
    request_path = Path(output) / "cold-input.json"
    request_path.write_text(json.dumps({"texts": sentinel_queries + sentinel_docs}))
    output_path = Path(output) / "cold-vectors.npy"
    script = (
        "import json,sys,numpy as np;"
        "sys.path.insert(0,sys.argv[1]);"
        "from package_runtime import JAXSentenceEncoder;"
        "m=JAXSentenceEncoder.from_pretrained(sys.argv[1],local_files_only=True);"
        "x=json.load(open(sys.argv[2]));np.save(sys.argv[3],m.encode(x['texts']))"
    )
    env = dict(os.environ)
    env.update(
        {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "JAX_PLATFORMS": "cpu"}
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(package_dir),
            str(request_path),
            str(output_path),
        ],
        cwd=child_dir,
        env=env,
        check=True,
        timeout=180,
    )
    reloaded = np.load(output_path)
    np.testing.assert_allclose(current, reloaded, rtol=1e-5, atol=1e-6)
    q_count = len(sentinel_queries)
    current_scores = current[:q_count] @ current[q_count:].T
    reloaded_scores = reloaded[:q_count] @ reloaded[q_count:].T
    np.testing.assert_allclose(current_scores, reloaded_scores, rtol=1e-5, atol=1e-6)
    doc_ids = np.arange(len(sentinel_docs), dtype=np.int64)
    for left, right in zip(current_scores, reloaded_scores):
        assert np.array_equal(
            np.lexsort((doc_ids, -left)), np.lexsort((doc_ids, -right))
        )

    for name, mutation in (
        (
            "bad-weights",
            lambda folder: (folder / "model.safetensors").write_bytes(
                (folder / "model.safetensors").read_bytes()[:-1] + b"x"
            ),
        ),
        ("bad-tokenizer", lambda folder: (folder / "tokenizer.json").write_text("{}")),
        (
            "bad-pooling",
            lambda folder: _mutate_json(
                folder / "package.json", "pooling", "max pooling"
            ),
        ),
        (
            "bad-config",
            lambda folder: _mutate_json(
                folder / "config.json", "width", config.width + 1
            ),
        ),
        (
            "bad-model-source",
            lambda folder: (folder / "model.py").write_text(
                (folder / "model.py").read_text() + "\n# mutation\n"
            ),
        ),
        (
            "bad-loader-source",
            lambda folder: (folder / "package_runtime.py").write_text(
                (folder / "package_runtime.py").read_text() + "\n# mutation\n"
            ),
        ),
    ):
        destination = Path(output) / name
        shutil.copytree(package_dir, destination)
        mutation(destination)
        try:
            encoder.from_pretrained(destination, local_files_only=True)
        except (ValueError, OSError, KeyError, RuntimeError):
            pass
        else:
            raise AssertionError(f"corrupted package accepted: {name}")
    candidate_source = Path(implementation_path).read_text(encoding="utf-8")
    sentinel_path = Path(output) / "candidate-pool-sentinel.py"
    sentinel_path.write_text(
        candidate_source
        + "\n_original_c01_pool = pool\n"
        + "def pool(params, input_ids, config, valid_mask=None, *, dropout_key=None):\n"
        + "    return jnp.flip(_original_c01_pool(params, input_ids, config, valid_mask, dropout_key=dropout_key), axis=-1)\n"
    )
    sentinel_module, sentinel_source_path = load_implementation(sentinel_path)
    sentinel_package = Path(output) / "candidate-pool-sentinel-package"
    save_package(
        params,
        config,
        tokenizer,
        sentinel_package,
        model_module=sentinel_module,
        implementation_path=sentinel_source_path,
    )
    sentinel_encoder = JAXSentenceEncoder.from_pretrained(
        sentinel_package, local_files_only=True
    )
    sentinel_vectors = sentinel_encoder.encode(sentinel_queries + sentinel_docs)
    sentinel_direct = np.asarray(
        sentinel_module.pool(params, jnp.asarray(padded), config), np.float32
    )
    np.testing.assert_allclose(sentinel_vectors, sentinel_direct, rtol=1e-5, atol=1e-6)
    if np.allclose(sentinel_vectors, current, rtol=1e-5, atol=1e-6):
        raise AssertionError("packaged candidate pool sentinel was not executed")
    return {
        "package": str(package_dir),
        "cold_process_equal_vectors": True,
        "cold_process_equal_rankings": True,
        "corruption_controls_rejected": 6,
        "selected_candidate_pool_matches_export": True,
        "candidate_pool_sentinel_changes_package_output": True,
        "package_metadata": encoder.package_metadata,
    }


def _mutate_json(path, key, value):
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data, sort_keys=True))


def run(
    data_dir, output, implementation="solution", stage="all", offline_baseline=False
):
    data_dir, output = Path(data_dir).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_obj = dataset.verify_frozen_data(data_dir)
    recipe_source = ROOT / "data" / "frozen_recipe.json"
    recipe_snapshot = output / "frozen-recipe-input.json"
    shutil.copyfile(recipe_source, recipe_snapshot)
    recipe_snapshot_sha256 = _file_hash(recipe_snapshot)
    model, candidate_path = load_implementation(implementation)
    if stage not in {"all", "phase1", "phase2", "phase3"}:
        raise ValueError("stage must be all, phase1, phase2, or phase3")
    source_hashes = _hash_source_tree(candidate_path)

    corpus = dataset_obj["corpus"]
    inventory = dataset_obj["inventories"]["ids"]
    doc_by_id = {int(row["doc_id"]): row for row in corpus}
    pretrain_ids = inventory["pretraining_doc_ids"]
    recipe_dev_doc_ids = set(map(int, inventory["recipe_dev_train_doc_ids"]))
    pretrain_docs = [doc_by_id[int(doc_id)] for doc_id in pretrain_ids]
    if len(pretrain_docs) != 4888:
        raise ValueError("frozen pretraining document inventory changed")
    tokenizer, processor_info = dataset.train_byte_bpe(
        pretrain_docs, output / "tokenizer.json", vocab_size=512
    )
    _save_json(output / "tokenizer-info.json", processor_info)
    if processor_info["training_doc_ids"] != pretrain_ids:
        raise AssertionError(
            "processor was fitted outside the exact permitted document inventory"
        )
    config = EncoderConfig(
        vocab_size=processor_info["vocabulary_size"],
        max_length=64,
        width=32,
        layers=2,
        heads=4,
        mlp_width=64,
        dropout=0.1,
        tied_mlm_head=True,
    )
    config.validate()
    _save_json(output / "model-config.json", config.as_dict())

    pilot_docs = dataset.select_pilot_documents(corpus, pretrain_ids, limit=128)
    selected_ids = [int(row["doc_id"]) for row in pilot_docs]
    selected_recipe_dev_docs = set(selected_ids) & recipe_dev_doc_ids
    if set(selected_ids) & set(inventory["final_heldout_doc_ids"]):
        raise AssertionError("final-connected heldout docs entered MLM pilot")
    mlm_rows, mlm_truncated = dataset.encode_documents(
        pilot_docs, tokenizer, config.max_length
    )
    mlm_values = _pad_rows(mlm_rows, config.max_length, config.pad_id)
    np.save(output / "mlm-input-ids.npy", mlm_values)

    weak_pairs = _pair_order(dataset.make_weak_pairs(pilot_docs), limit=32)
    if len(weak_pairs) != 32:
        raise ValueError(f"frozen phase 2 needs 32 weak pairs, found {len(weak_pairs)}")
    weak_left = [
        _pad_rows(
            [
                {
                    "ids": dataset.encode_text(
                        tokenizer, pair["left"], config.max_length
                    )[0]
                }
            ],
            config.max_length,
            config.pad_id,
        )[0]
        for pair in weak_pairs
    ]
    weak_right = [
        _pad_rows(
            [
                {
                    "ids": dataset.encode_text(
                        tokenizer, pair["right"], config.max_length
                    )[0]
                }
            ],
            config.max_length,
            config.pad_id,
        )[0]
        for pair in weak_pairs
    ]
    weak_groups = [int(pair["group_id"]) for pair in weak_pairs]
    pair_inventory = [
        {
            "group_id": int(row["group_id"]),
            "source": row["source"],
            "left_sha256": _sha(row["left"].encode()),
            "right_sha256": _sha(row["right"].encode()),
        }
        for row in weak_pairs
    ]
    _save_json(
        output / "weak-pair-inventory.json",
        {
            "pairs": pair_inventory,
            "sha256": _sha(
                json.dumps(
                    pair_inventory, sort_keys=True, separators=(",", ":")
                ).encode()
            ),
        },
    )

    query_source = dataset.make_human_pairs(dataset_obj["fit"], corpus)
    fit_query_rows, fit_truncated = dataset.encode_queries(
        dataset_obj["fit"], tokenizer, config.max_length
    )
    qrows_by_id = {row["id"]: row for row in fit_query_rows}
    human_rows = []
    for record in query_source:
        row = dict(record)
        row["ids"] = np.pad(
            qrows_by_id[row["query_id"]]["ids"],
            (0, config.max_length - len(qrows_by_id[row["query_id"]]["ids"])),
            constant_values=config.pad_id,
        )
        human_rows.append(row)
    encoded_docs, document_truncated = dataset.encode_documents(
        corpus, tokenizer, config.max_length
    )
    encoded_docs = [
        {
            **row,
            "ids": np.pad(
                row["ids"],
                (0, config.max_length - len(row["ids"])),
                constant_values=config.pad_id,
            ),
        }
        for row in encoded_docs
    ]
    data_hash = _dataset_digest(dataset_obj, selected_ids, processor_info)
    processor_hash = processor_info["processor_sha256"]
    base_config = TrainConfig(
        batch_size=4,
        learning_rate=0.0003,
        weight_decay=0.01,
        warmup_steps=1,
        total_steps=8,
        seed=20261021,
    )
    phase_summaries = {}
    phase_states = {}
    checkpoint_hashes = {}
    raw_traces = []

    initial_state = initialize_state(
        model, config, len(mlm_values), seed=20261021, train_config=base_config
    )
    random_params = initial_state["params"]
    phase1_cfg = replace(base_config, total_steps=8, learning_rate=0.0003)
    phase1_identity = identity_for(
        code_hashes=source_hashes,
        dataset_hash=data_hash,
        processor_hash=processor_hash,
        config=config,
        train_config=phase1_cfg,
        objective={
            "name": "masked_language_modeling",
            "mask_rate": 0.15,
            "selected_count_normalized": True,
        },
        parent=None,
        branch="phase1-mlm",
    )
    phase1_state, phase1_stats = train_mlm(
        model, initial_state, mlm_values, config, phase1_cfg, steps=8
    )
    raw_traces.extend(phase1_stats["trace"])
    checkpoint_hashes["phase1_mlm"] = save_checkpoint(
        output / "checkpoints" / "phase1-mlm", phase1_state, phase1_identity
    )
    phase_states["random_initialization"] = random_params
    phase_states["phase1_mlm"] = phase1_state["params"]
    phase_summaries["phase1_mlm"] = {
        "losses": phase1_stats["losses"],
        "traces": phase1_stats["trace"],
        "selected_mlm_documents": selected_ids,
        "truncated_tokens": mlm_truncated,
    }

    if stage in {"all", "phase2", "phase3"}:
        p1_identity_hash = checkpoint_hashes["phase1_mlm"]
        phase2_cfg = replace(base_config, total_steps=8, learning_rate=0.0001)
        for branch, mlm_weight in (
            ("phase2_contrastive_only", 0.0),
            ("phase2_contrastive_plus_retained_mlm", 0.1),
        ):
            state = initialize_state(
                model,
                config,
                len(weak_left),
                seed=20261022,
                train_config=phase2_cfg,
                initial_params=phase1_state["params"],
                initial_global_step=int(phase1_state["global_step"]),
            )
            identity = identity_for(
                code_hashes=source_hashes,
                dataset_hash=data_hash,
                processor_hash=processor_hash,
                config=config,
                train_config=phase2_cfg,
                objective={
                    "name": "same_document_weak_contrastive",
                    "contrastive_weight": 1.0,
                    "retained_mlm_weight": mlm_weight,
                    "pair_ids_sha256": _sha(
                        json.dumps(
                            pair_inventory, sort_keys=True, separators=(",", ":")
                        ).encode()
                    ),
                },
                parent=p1_identity_hash,
                branch=branch,
            )
            state, stats = train_weak_contrastive(
                model,
                state,
                np.asarray(weak_left),
                np.asarray(weak_right),
                weak_groups,
                config,
                phase2_cfg,
                steps=8,
                retained_mlm_weight=mlm_weight,
                mlm_examples=mlm_values,
                tag=branch,
            )
            raw_traces.extend(stats["trace"])
            checkpoint_hashes[branch] = save_checkpoint(
                output / "checkpoints" / branch, state, identity
            )
            phase_states[branch] = state["params"]
            phase_summaries[branch] = {
                "losses": stats["losses"],
                "traces": stats["trace"],
                "pair_ids_sha256": identity["objective"]["pair_ids_sha256"],
                "parent": p1_identity_hash,
                "optimizer_reset": True,
                "exposure_pairs": sum(len(row["indices"]) for row in stats["trace"]),
            }

    if stage in {"all", "phase3"}:
        parent_hash = checkpoint_hashes["phase2_contrastive_plus_retained_mlm"]
        phase3_cfg = replace(base_config, total_steps=71, learning_rate=0.00005)
        state = initialize_state(
            model,
            config,
            len(human_rows),
            seed=20261023,
            train_config=phase3_cfg,
            initial_params=phase_states["phase2_contrastive_plus_retained_mlm"],
            initial_global_step=int(phase1_state["global_step"]) + 8,
        )
        identity = identity_for(
            code_hashes=source_hashes,
            dataset_hash=data_hash,
            processor_hash=processor_hash,
            config=config,
            train_config=phase3_cfg,
            objective={
                "name": "fit_evidence_multi_positive_contrastive",
                "temperature": 0.05,
                "unknown_candidates": "unlabeled in-batch candidates; not claimed irrelevant",
            },
            parent=parent_hash,
            branch="phase3_human_pairs",
        )
        state, stats = train_human_contrastive(
            model,
            state,
            human_rows,
            encoded_docs,
            config,
            phase3_cfg,
            steps=71,
            max_positive_per_query=4,
        )
        raw_traces.extend(stats["trace"])
        checkpoint_hashes["phase3_human_pairs"] = save_checkpoint(
            output / "checkpoints" / "phase3-human-pairs", state, identity
        )
        phase_states["phase3_human_pairs"] = state["params"]
        phase_summaries["phase3_human_pairs"] = {
            "losses": stats["losses"],
            "traces": stats["trace"],
            "parent": parent_hash,
            "optimizer_reset": True,
            "unique_fit_judged_queries": len(human_rows),
            "partial_tail_batch": stats["trace"][-1]["query_ids"],
        }

    population_map = {
        "recipe_development": dataset_obj["recipe_dev"],
        "final_official_dev_custom_protocol": dataset_obj["final"],
    }
    evaluation_metrics = {}
    params_for_eval = phase_states
    for model_name, params in params_for_eval.items():
        for population_name, population in population_map.items():
            name = f"{model_name}/{population_name}"
            evaluation_metrics[name] = _rank_population(
                name, population, model, params, config, corpus, tokenizer, output
            )

    baseline_dir = output / "baseline_snapshot"
    baseline_model, baseline_info = load_pinned_baseline(
        baseline_dir, offline=offline_baseline
    )
    baseline_text_docs = [dataset.document_text(row) for row in corpus]
    baseline_doc_vectors = encode_baseline(baseline_model, baseline_text_docs)
    doc_ids = np.asarray([int(row["doc_id"]) for row in corpus], np.int64)
    baseline_info["snapshot_directory"] = str(baseline_dir)
    for population_name, population in population_map.items():
        qvectors = encode_baseline(baseline_model, [row["claim"] for row in population])
        qids = np.asarray([int(row["id"]) for row in population], np.int64)
        name = f"external_minilm_l6_v2/{population_name}"
        evaluation_metrics[name] = _save_rankings_from_vectors(
            name, population, qvectors, baseline_doc_vectors, qids, doc_ids, output
        )
    _save_json(output / "baseline-receipt.json", baseline_info)

    for population_name, population in population_map.items():
        name = f"bm25/{population_name}"
        qids, bdoc_ids, bm25_matrix = evaluation.bm25_scores(corpus, population)
        rankings = {
            int(qid): [
                int(bdoc_ids[index]) for index in np.lexsort((bdoc_ids, -scores))
            ]
            for qid, scores in zip(qids, bm25_matrix)
        }
        metrics = evaluation.metrics_from_rankings(rankings, population)
        score_folder = output / "raw" / name
        score_folder.mkdir(parents=True, exist_ok=True)
        # BM25 raw rank matrix is exact and the scores/terms are regenerated by the checker.
        np.savez_compressed(
            score_folder / "rankings.npz",
            query_ids=qids,
            ranked_doc_ids=np.asarray(list(rankings.values()), np.uint32),
            document_ids=bdoc_ids,
            scores=bm25_matrix.astype(np.float32),
        )
        _save_json(
            score_folder / "qrels.json",
            {
                str(int(row["id"])): sorted(map(int, row.get("evidence", {}).keys()))
                for row in population
            },
        )
        _save_json(score_folder / "metrics.json", metrics)
        evaluation_metrics[name] = metrics

    final_metric_candidates = {
        model_name: metrics
        for name, metrics in evaluation_metrics.items()
        if name.endswith("/final_official_dev_custom_protocol")
        for model_name in [name.split("/")[0]]
    }
    final_name = (
        "phase3_human_pairs"
        if "phase3_human_pairs" in phase_states
        else sorted(phase_states)[-1]
    )
    final_quality = final_metric_candidates.get(final_name, {})
    pilot_floors = {"ndcg@10": 0.10, "recall@10": 0.30}
    pilot_floor_pass = (
        bool(final_quality)
        and final_quality.get("ndcg@10", 0.0) >= pilot_floors["ndcg@10"]
        and final_quality.get("recall@10", 0.0) >= pilot_floors["recall@10"]
    )
    phase1_quality = final_metric_candidates.get("phase1_mlm", {})
    retention_regression = (
        None
        if not phase1_quality or not final_quality
        else phase1_quality["ndcg@10"] - final_quality["ndcg@10"]
    )
    promotion_gate = pilot_floor_pass and (
        retention_regression is None or retention_regression <= 0.05
    )
    phase1_trace = [row for row in raw_traces if row["stage"] == "phase1_mlm"]
    phase2_weak_trace = [
        row for row in raw_traces if row["stage"] == "phase2_contrastive_only"
    ]
    phase2_joint_trace = [
        row
        for row in raw_traces
        if row["stage"] == "phase2_contrastive_plus_retained_mlm"
    ]
    phase3_trace = [row for row in raw_traces if row["stage"] == "phase3_human_pairs"]
    phase1_document_ids = sorted(
        {int(selected_ids[index]) for row in phase1_trace for index in row["indices"]}
    )
    phase2_mlm_document_ids = sorted(
        {
            int(selected_ids[index])
            for row in phase2_joint_trace
            for index in row["mlm_indices"]
        }
    )
    phase2_pair_indices = sorted(
        {int(index) for row in phase2_weak_trace for index in row["indices"]}
    )
    phase2_weak_document_ids = sorted(
        {int(pair_inventory[index]["group_id"]) for index in phase2_pair_indices}
    )
    phase3_query_ids = sorted(
        {int(query_id) for row in phase3_trace for query_id in row["query_ids"]}
    )
    query_by_id = {int(row["query_id"]): row for row in human_rows}
    phase3_positive_document_ids = sorted(
        {
            int(doc_id)
            for query_id in phase3_query_ids
            for doc_id in query_by_id[query_id]["positive_doc_ids"]
        }
    )
    measured_exposure = {
        "phase1_mlm": {
            "consumed_document_ids": phase1_document_ids,
            "unique_documents": len(phase1_document_ids),
            "recipe_development_documents_consumed": sorted(
                set(phase1_document_ids) & recipe_dev_doc_ids
            ),
            "recipe_development_documents_consumed_count": len(
                set(phase1_document_ids) & recipe_dev_doc_ids
            ),
            "nonpadding_input_tokens": sum(row["input_tokens"] for row in phase1_trace),
            "selected_masked_tokens": sum(
                row["selected_tokens"] for row in phase1_trace
            ),
        },
        "phase2_contrastive_only": {
            "consumed_weak_pair_indices": phase2_pair_indices,
            "weak_pair_source_document_ids": phase2_weak_document_ids,
            "unique_weak_pairs": len(phase2_pair_indices),
            "nonpadding_pair_input_tokens": sum(
                row["pair_input_tokens"] for row in phase2_weak_trace
            ),
        },
        "phase2_contrastive_plus_retained_mlm": {
            "consumed_weak_pair_indices": sorted(
                {int(index) for row in phase2_joint_trace for index in row["indices"]}
            ),
            "unique_weak_pairs": len(
                {int(index) for row in phase2_joint_trace for index in row["indices"]}
            ),
            "weak_pair_source_document_ids": sorted(
                {
                    int(pair_inventory[index]["group_id"])
                    for row in phase2_joint_trace
                    for index in row["indices"]
                }
            ),
            "retained_mlm_consumed_document_ids": phase2_mlm_document_ids,
            "unique_retained_mlm_documents": len(phase2_mlm_document_ids),
            "recipe_development_documents_consumed": sorted(
                set(phase2_mlm_document_ids) & recipe_dev_doc_ids
            ),
            "recipe_development_documents_consumed_count": len(
                set(phase2_mlm_document_ids) & recipe_dev_doc_ids
            ),
            "nonpadding_pair_input_tokens": sum(
                row["pair_input_tokens"] for row in phase2_joint_trace
            ),
            "nonpadding_retained_mlm_input_tokens": sum(
                row["mlm_input_tokens"] for row in phase2_joint_trace
            ),
            "selected_retained_mlm_tokens": sum(
                row["mlm_selected_tokens"] for row in phase2_joint_trace
            ),
        },
        "phase3_human_pairs": {
            "consumed_fit_query_ids": phase3_query_ids,
            "unique_fit_queries": len(phase3_query_ids),
            "human_evidence_positive_document_ids": phase3_positive_document_ids,
            "unique_positive_documents": len(phase3_positive_document_ids),
            "nonpadding_query_input_tokens": sum(
                row["query_input_tokens"] for row in phase3_trace
            ),
            "nonpadding_candidate_document_input_tokens": sum(
                row["candidate_document_input_tokens"] for row in phase3_trace
            ),
        },
    }
    report = {
        "protocol": dataset_obj["manifest"]["protocol"],
        "status": (
            "pilot_executed_in_development"
            if stage == "all"
            else f"executed_through_{stage}"
        ),
        "target_300m_qualified": False,
        "public_release_qualified": False,
        "candidate_path": str(candidate_path),
        "candidate_sha256": _file_hash(candidate_path),
        "source_hashes": source_hashes,
        "frozen_data_manifest_sha256": _file_hash(data_dir / "frozen-manifest.json"),
        "frozen_inventory_sha256": _file_hash(data_dir / "inventories.json"),
        "frozen_recipe_input_sha256": recipe_snapshot_sha256,
        "selected_mlm_document_ids": selected_ids,
        "selected_mlm_ids_sha256": _sha(
            json.dumps(selected_ids, separators=(",", ":")).encode()
        ),
        "processor_info": processor_info,
        "processor_truncated_tokens": {
            "mlm": mlm_truncated,
            "fit_queries": fit_truncated,
            "corpus_docs": document_truncated,
        },
        "pair_inventory_sha256": _sha(
            json.dumps(pair_inventory, sort_keys=True, separators=(",", ":")).encode()
        ),
        "config": config.as_dict(),
        "target_parameter_count": model.parameter_count(
            __import__("config").TARGET_CONFIG
        ),
        "target_with_mlm_head_parameter_count": model.parameter_count(
            __import__("config").TARGET_CONFIG, include_mlm=True
        ),
        "training_stages": phase_summaries,
        "checkpoint_json_sha256": checkpoint_hashes,
        "evaluation": evaluation_metrics,
        "baseline": baseline_info,
        "quality_gates": {
            "precommitted_floors": pilot_floors,
            "phase3_final_floors_pass": pilot_floor_pass,
            "phase1_to_phase3_ndcg10_drop": retention_regression,
            "pilot_quality_gate_pass": promotion_gate,
            "interpretation": "Even a pilot gate pass is not a target-scale, general embedding or release qualification.",
        },
        "measured_training_exposure": measured_exposure,
        "exposure": {
            "mlm_prepared_documents": len(selected_ids),
            "tokenizer_eligible_documents": len(pretrain_ids),
            "recipe_development_component_documents": len(recipe_dev_doc_ids),
            "recipe_development_documents_in_tokenizer": len(
                set(map(int, pretrain_ids)) & recipe_dev_doc_ids
            ),
            "recipe_development_documents_in_mlm_sample": len(selected_recipe_dev_docs),
            "recipe_development_query_text_training_examples": 0,
            "recipe_development_human_pair_training_examples": 0,
            "final_query_text_training_examples": 0,
            "mlm_prepared_nonpadding_token_count": int(np.count_nonzero(mlm_values)),
            "phase2_weak_pairs_per_branch": 32,
            "phase3_fit_judged_queries": len(human_rows),
            "final_candidate_documents": len(corpus),
            "final_query_inventory": len(dataset_obj["final"]),
            "final_judged_queries": sum(
                bool(row.get("evidence")) for row in dataset_obj["final"]
            ),
        },
        "limitations": [
            "small two-layer CPU pilot, not the 300M target",
            "one frozen split and one seed; training variability is unmeasured",
            "SciFact official dev is used as a custom frozen final protocol because public test judgments are absent",
            "all candidate abstracts are available at ranking time; possible pretrained-baseline overlap with S2ORC remains unresolved",
            "weak pairs are source-based, not human relevance labels",
            "no MTEB/general-embedding, target-runtime, service, remote-package or human acceptance claim",
        ],
    }
    _save_json(output / "pilot-report.json", report)
    _save_json(output / "training-trace.json", raw_traces)
    loss_figure(output / "figures" / "training-objectives.svg", raw_traces)
    final_fig_metrics = {
        key.split("/")[0]: value
        for key, value in evaluation_metrics.items()
        if key.endswith("/final_official_dev_custom_protocol")
        and key.split("/")[0]
        in {
            "random_initialization",
            "phase1_mlm",
            "phase2_contrastive_only",
            "phase2_contrastive_plus_retained_mlm",
            "phase3_human_pairs",
            "external_minilm_l6_v2",
            "bm25",
        }
    }
    retrieval_figure(output / "figures" / "final-retrieval.svg", final_fig_metrics)
    figure_data = {
        "training_objectives": {
            "unit_note": "loss units differ by objective; trace stores per-update values and normalized metrics",
            "traces": raw_traces,
        },
        "final_retrieval": final_fig_metrics,
    }
    _save_json(output / "figure-data.json", figure_data)

    sentinel_queries = [row["claim"] for row in dataset_obj["final"][:3]]
    sentinel_doc_records = sorted(
        corpus, key=lambda row: _sha(str(row["doc_id"]).encode())
    )[:20]
    sentinel_docs = [dataset.document_text(row) for row in sentinel_doc_records]
    package_check = _cold_package_check(
        model,
        candidate_path,
        phase_states[final_name],
        config,
        tokenizer,
        sentinel_queries,
        sentinel_docs,
        output,
    )
    report["local_package"] = package_check
    package_check["model_card_sha256"] = _write_measured_model_card(
        output / "local-package",
        report,
        dataset_obj["manifest"]["source"]["archive_sha256"],
    )
    _save_json(output / "pilot-report.json", report)
    environment = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    dataset_hashes = {
        path.name: _file_hash(path)
        for path in sorted(data_dir.iterdir())
        if path.is_file()
    }
    checkpoint_hashes_full = {}
    for path in sorted((output / "checkpoints").glob("*/checkpoint.json")):
        checkpoint_hashes_full[str(path.relative_to(output))] = _file_hash(path)
        checkpoint_hashes_full[str(path.with_name("state.npz").relative_to(output))] = (
            _file_hash(path.with_name("state.npz"))
        )
    receipt = {
        "schema": "jaxpathways-embedding-c01-source-bound-v1",
        "candidate_sha256": _file_hash(candidate_path),
        "source_files_sha256": source_hashes,
        "data_output_sha256": dataset_hashes,
        "data_source_archive_sha256": dataset_obj["manifest"]["source"][
            "archive_sha256"
        ],
        "frozen_recipe_sha256": recipe_snapshot_sha256,
        "data_inventory_sha256": dataset_obj["manifest"]["inventories_sha256"],
        "processor_sha256": processor_info["processor_sha256"],
        "pair_inventory_sha256": report["pair_inventory_sha256"],
        "checkpoint_files_sha256": checkpoint_hashes_full,
        "baseline_file_sha256": baseline_info["snapshot_files_sha256"],
        "environment": environment,
        "report_sha256": _file_hash(output / "pilot-report.json"),
        "training_trace_sha256": _file_hash(output / "training-trace.json"),
        "figure_data_sha256": _file_hash(output / "figure-data.json"),
    }
    receipt["run_artifacts_sha256"] = {
        str(path.relative_to(output)): _file_hash(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path != output / "sourcebound-receipt.json"
    }
    _save_json(output / "sourcebound-receipt.json", receipt)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation",
        default="solution",
        help="solution, starter, or candidate Python file",
    )
    parser.add_argument(
        "--stage", choices=("all", "phase1", "phase2", "phase3"), default="all"
    )
    parser.add_argument(
        "--data", type=Path, required=True, help="output from data/prepare.py"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="external run output; defaults to a fresh temporary directory",
    )
    parser.add_argument(
        "--offline-baseline",
        action="store_true",
        help="require the pinned baseline to already be present in --output/baseline_snapshot",
    )
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix="jaxpathways-embedding-c01-"))
    report = run(
        args.data, output, args.implementation, args.stage, args.offline_baseline
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "status": report["status"],
                "evaluation": report["evaluation"],
                "quality_gates": report["quality_gates"],
                "local_package": report.get("local_package"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
