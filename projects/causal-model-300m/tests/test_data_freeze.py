"""Focused parser and frozen-source checks for C02 data preparation."""

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "c02_prepare", PROJECT / "data" / "prepare.py"
)
prepare_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_module)


class WikiTextArticleParsingTests(unittest.TestCase):
    def test_subsections_stay_inside_top_level_article_and_neighbors_are_retained(self):
        lines = [
            "",
            " = Sample Article = \n",
            "Introductory article text.\n",
            " = = First subsection = = \n",
            "Subsection paragraph.\n",
            " = Another Article = \n",
            "Another article paragraph.\n",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny.parquet"
            pq.write_table(pa.table({"text": lines}), path)
            articles = prepare_module._wiki_articles(path, "train")
        self.assertEqual(
            [item["title"] for item in articles], ["Sample Article", "Another Article"]
        )
        self.assertIn("First subsection", articles[0]["text"])
        self.assertEqual(articles[0]["parser_context"]["preceding_line"], "")
        self.assertEqual(
            articles[0]["parser_context"]["following_line"],
            "Introductory article text.\n",
        )
        self.assertEqual(articles[0]["parser_context"]["subsection_lines_retained"], 1)
        self.assertEqual(articles[0]["source_row_end_exclusive"], 5)

    def test_article_title_and_exact_context_or_question_duplicates_connect(self):
        articles = [
            {
                "source": "wikitext",
                "split": "train",
                "title": "Shared Title",
                "normalized_title": "shared title",
                "normalized_paragraphs": ["same passage text"],
            },
            {
                "source": "squad",
                "split": "train",
                "title": "Shared Title",
                "normalized_title": "shared title",
                "paragraphs": [
                    {
                        "normalized_context": "same passage text",
                        "questions": [{"question": "A question?"}],
                    }
                ],
            },
            {
                "source": "squad",
                "split": "train",
                "title": "Different",
                "normalized_title": "different",
                "paragraphs": [
                    {
                        "normalized_context": "other passage",
                        "questions": [{"question": "A question!"}],
                    }
                ],
            },
        ]
        groups = prepare_module._components(articles)
        self.assertEqual(groups, [[0, 1, 2]])

    def test_target_blocks_preserve_aligned_targets_and_reset_document_segments(self):
        class CharacterTokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                return type("Encoding", (), {"ids": [ord(char) + 4 for char in text]})()

        first = "ab" * 150
        second = "xyz" * 100
        records = [{"id": "a", "text": first}, {"id": "b", "text": second}]
        with tempfile.TemporaryDirectory() as temporary:
            result = prepare_module._build_target_blocks(
                records, CharacterTokenizer(), Path(temporary), "tiny"
            )
            arrays = np.load(Path(temporary) / "tiny-blocks.npz")
        self.assertEqual(result["block_count"], 1)
        self.assertEqual(result["total_source_targets_including_eos"], 602)
        self.assertEqual(
            result["incomplete_tail_targets_not_used_in_fixed-block_pilot"], 90
        )
        self.assertEqual(result["complete_block_targets"], 512)
        self.assertTrue(arrays["target_mask"].all())
        self.assertEqual(arrays["target_ids"].shape, (1, 512))
        self.assertEqual(arrays["input_ids"][0, 0], 1)
        self.assertEqual(arrays["input_ids"][0, 301], 1)
        self.assertEqual(arrays["segment_ids"][0, 300], 1)
        self.assertEqual(arrays["segment_ids"][0, 301], 2)
        self.assertEqual(arrays["position_ids"][0, 0], 0)
        self.assertEqual(arrays["position_ids"][0, 300], 300)
        self.assertEqual(arrays["position_ids"][0, 301], 0)
        self.assertEqual(arrays["target_ids"][0, 300], 2)
        self.assertEqual(arrays["target_ids"][0, 301], ord("x") + 4)
        expected_target_prefix = [ord(char) + 4 for char in first] + [2]
        expected_target_prefix += [ord(char) + 4 for char in second]
        expected_target_prefix += [2]
        self.assertEqual(arrays["target_ids"][0].tolist(), expected_target_prefix[:512])
        self.assertTrue(
            np.array_equal(
                arrays["input_ids"][0, 302:512], arrays["target_ids"][0, 301:511]
            )
        )

    def test_pinned_revisions_and_rights_are_explicit(self):
        self.assertEqual(
            prepare_module.WIKI_REVISION, "b08601e04326c79dfdd32d625aee71d232d685c3"
        )
        self.assertEqual(
            prepare_module.SQUAD_REVISION, "eee5fdbf62f8613a7812b03419e6b29617b74fd1"
        )
        self.assertEqual(
            prepare_module.PYTHIA_REVISION,
            "d31f659f077f9ea500c4d7faa999893d7f28cbdf",
        )
        self.assertEqual(len(prepare_module.PYTHIA_FILES_SHA256), 7)
        self.assertEqual(
            prepare_module.LM_EVAL_COMMIT,
            "ddd67220430a2470529f25fd5c05a576ca1057a0",
        )
        rights = (PROJECT / "data" / "RIGHTS.md").read_text()
        self.assertIn("CC BY-SA 3.0", rights)
        self.assertIn("GFDL", rights)
        self.assertIn("CC BY-SA 4.0", rights)
        self.assertIn("trained weights", rights)


class PreparedDataVerificationTests(unittest.TestCase):
    def test_verifier_rejects_changed_tokenizer_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            prepared = Path(temporary)
            membership_bytes = b"{}"
            expected_tokenizer = b"trusted tokenizer identity"
            tokenizer_hash = hashlib.sha256(expected_tokenizer).hexdigest()
            manifest = {
                "preparer": {
                    "sha256": hashlib.sha256(
                        Path(prepare_module.__file__).read_bytes()
                    ).hexdigest()
                },
                "source_file_hashes": {},
                "tokenizer": {"processor_sha256": tokenizer_hash},
            }
            manifest_bytes = prepare_module.stable_json(manifest)
            recipe = {
                "membership_manifest_sha256": hashlib.sha256(
                    membership_bytes
                ).hexdigest(),
                "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "tokenizer_sha256": tokenizer_hash,
                "tokenizer": {"processor_sha256": tokenizer_hash},
                "prepared_outputs": {},
                "target_block_inventories": {},
            }
            (prepared / "memberships.json").write_bytes(membership_bytes)
            (prepared / "source-manifest.json").write_bytes(manifest_bytes)
            (prepared / "frozen_recipe.json").write_bytes(
                prepare_module.stable_json(recipe)
            )
            (prepared / "tokenizer.json").write_bytes(b"rewritten tokenizer")
            with self.assertRaisesRegex(ValueError, "tokenizer bytes differ"):
                prepare_module.verify_prepared(prepared, require_project_lock=False)

    def test_verifier_rejects_artifact_recipe_not_bound_to_project_owned_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "project" / "data"
            data_dir.mkdir(parents=True)
            (data_dir / "frozen_recipe.json").write_text('{"accepted":true}\n')
            prepared = root / "prepared"
            prepared.mkdir()
            (prepared / "source-manifest.json").write_text("{}")
            (prepared / "memberships.json").write_text("{}")
            (prepared / "frozen_recipe.json").write_text("{}")
            original_file = prepare_module.__file__
            try:
                prepare_module.__file__ = str(data_dir / "prepare.py")
                with self.assertRaisesRegex(
                    ValueError, "project-owned accepted data recipe"
                ):
                    prepare_module.verify_prepared(prepared)
            finally:
                prepare_module.__file__ = original_file

    def test_external_prepared_directory_is_hash_bound_and_nonleaking(self):
        prepared = os.environ.get("C02_PREPARED_DATA")
        if not prepared:
            self.skipTest("set C02_PREPARED_DATA to run the real frozen-source audit")
        result = prepare_module.verify_prepared(
            Path(prepared), require_project_lock=False
        )
        self.assertEqual(result["status"], "pass")
        self.assertGreaterEqual(result["final_questions"], prepare_module.MIN_FINAL)
        self.assertGreaterEqual(
            result["final_articles"], prepare_module.MIN_FINAL_ARTICLES
        )
        manifest = json.loads((Path(prepared) / "frozen_recipe.json").read_text())
        self.assertEqual(manifest["pilot_schedule"]["architecture"]["context"], 512)
        selection = manifest["final_selection"]
        self.assertEqual(
            selection["generation"]["template"], prepare_module.QA_TEMPLATE
        )
        self.assertIn("\nContext:", selection["generation"]["template"])
        self.assertEqual(
            selection["total_official_dev_questions"],
            selection["input_eligible_count"]
            + selection["input_length_excluded_count"],
        )
        self.assertEqual(
            selection["input_eligible_count"],
            selection["not_selected_by_cap_count"] + selection["selected_count"],
        )
        self.assertLessEqual(selection["generation"]["max_new_tokens"], 32)
        self.assertEqual(
            manifest["pilot_schedule"]["continuation_target_mix"]["domain_targets"],
            30720,
        )
        self.assertEqual(
            manifest["pilot_schedule"]["continuation_target_mix"]["general_targets"],
            7680,
        )
        self.assertEqual(
            manifest["pilot_schedule"]["continuation_target_mix"]["domain"], 0.8
        )
        final_rows = [
            json.loads(line)
            for line in (Path(prepared) / "qa-final.jsonl").read_text().splitlines()
        ]
        processor = prepare_module.Tokenizer.from_file(
            str(Path(prepared) / "tokenizer.json")
        )
        for row in final_rows:
            prompt = f"Question: {row['question']}\nContext: {row['context']}\nAnswer:"
            prompt_tokens = len(processor.encode(prompt, add_special_tokens=False).ids)
            self.assertEqual(prompt_tokens, row["prompt_tokens"])
            self.assertLessEqual(
                prompt_tokens + 1 + selection["generation"]["max_new_tokens"],
                manifest["pilot_schedule"]["architecture"]["context"],
            )
        self.assertEqual(
            manifest["target_block_inventories"]["base"]["complete_block_targets"],
            512 * manifest["target_block_inventories"]["base"]["block_count"],
        )


if __name__ == "__main__":
    unittest.main()
