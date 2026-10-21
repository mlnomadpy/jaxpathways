"""Fixed greedy SQuAD QA evaluation for the pinned Pythia-14M snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(HERE / "data") not in sys.path:
    sys.path.insert(0, str(HERE / "data"))

import prepare
from squad_scoring import score
from pythia_pin import snapshot_identity


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(data_dir: Path, model_dir: Path, output: Path) -> dict:
    data_dir, model_dir, output = map(Path.resolve, (data_dir, model_dir, output))
    pinned_snapshot = snapshot_identity(model_dir)
    prepare.verify_prepared(data_dir, require_project_lock=True)
    rows = [
        json.loads(line)
        for line in (data_dir / "qa-final.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != 128:
        raise ValueError("expected the frozen 128-question final set")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, use_fast=True
    )
    candidate = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, torch_dtype=torch.float32
    ).to("cpu")
    candidate.eval()
    generation = {
        "algorithm": "greedy argmax; no sampling",
        "max_new_tokens": 32,
        "eos_token_id": tokenizer.eos_token_id,
        "prompt": "the frozen C02 Question/Context/Answer template; tokenizer adds its pinned model special tokens",
        "prompt_truncation": "none",
    }
    records = []
    scores = []
    for row in rows:
        prompt = prepare.qa_prompt(row["question"], row["context"])
        encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
        if (
            encoded["input_ids"].shape[1] + generation["max_new_tokens"]
            > candidate.config.max_position_embeddings
        ):
            raise ValueError(f"prompt exceeds Pythia context for {row['question_id']}")
        with torch.inference_mode():
            generated = candidate.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=generation["max_new_tokens"],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        new_ids = generated[0, encoded["input_ids"].shape[1] :].tolist()
        prediction = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        item = score(prediction, row["answers"])
        scores.append(item)
        records.append(
            {
                "question_id": row["question_id"],
                "article_id": row["article_id"],
                "prompt": prompt,
                "prompt_token_ids": encoded["input_ids"][0].tolist(),
                "generated_token_ids": new_ids,
                "prediction": prediction,
                "human_answers": row["answers"],
                "finish_reason": (
                    "eos"
                    if new_ids and new_ids[-1] == tokenizer.eos_token_id
                    else "max_new_tokens"
                ),
                **item,
            }
        )
    report = {
        "status": "external pretrained Pythia-14M fixed-decoding QA baseline; not instruction-tuned",
        "model": "EleutherAI/pythia-14m",
        "revision": pinned_snapshot["revision"],
        "snapshot_files_sha256": pinned_snapshot["files_sha256"],
        "checkpoint_sha256": sha256(model_dir / "model.safetensors"),
        "tokenizer_sha256": sha256(model_dir / "tokenizer.json"),
        "data_recipe_sha256": sha256(data_dir / "frozen_recipe.json"),
        "evaluator_sha256": sha256(Path(__file__)),
        "pin_verifier_sha256": sha256(HERE / "pythia_pin.py"),
        "scorer_sha256": sha256(HERE / "squad_scoring.py"),
        "selection": "all 128 final IDs, each exactly once; no final-score tuning",
        "generation": generation,
        "questions": len(records),
        "exact_match": sum(item["exact_match"] for item in scores) / len(scores),
        "token_f1": sum(item["token_f1"] for item in scores) / len(scores),
        "per_question": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.data, args.model, args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("questions", "exact_match", "token_f1", "evaluator_sha256")
            }
        )
    )


if __name__ == "__main__":
    main()
