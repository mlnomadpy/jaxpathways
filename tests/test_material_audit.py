"""Evidence must match the current source before it can qualify material."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from zipfile import ZipFile

spec = importlib.util.spec_from_file_location(
    "audit", Path(__file__).parents[1] / "scripts/audit-learning-material.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class MaterialAuditTests(unittest.TestCase):
    def test_stale_or_incomplete_receipts_cannot_validate_a_lesson(self):
        receipt = dict(
            contentHash="new",
            script="passed",
            notebookCode="passed",
            runtime=dict(deviceCount=1),
        )
        self.assertEqual(
            audit.lesson_check({"idea": "x"}, "new", receipt, "new"),
            "CPU reference checked",
        )
        self.assertIn(
            "pending", audit.lesson_check({"idea": "x"}, "changed", receipt, "changed")
        )
        self.assertIn(
            "pending", audit.lesson_check({"idea": "x"}, "new", receipt, "old")
        )
        receipt["notebookCode"] = "failed"
        self.assertIn(
            "failed", audit.lesson_check({"idea": "x"}, "new", receipt, "new")
        )
        self.assertEqual(
            audit.lesson_check({}, "new", receipt, "new"), "No lesson body"
        )

    def test_human_reviews_require_current_hash_retained_evidence_and_acceptance(self):
        review = dict(
            lessonId="example",
            contentHash="new",
            kind="expert",
            reviewer="Test reviewer",
            date="2026-10-21",
            changedCondition="A changed case",
            evidence="docs/teaching-review-protocol.md",
            outcome="accepted",
        )
        self.assertEqual(len(audit.accepted_reviews([review], "example", "new")), 1)
        self.assertEqual(audit.accepted_reviews([review], "example", "changed"), [])
        for field, value in [
            ("outcome", "changes-requested"),
            ("kind", "agent"),
            ("evidence", "does-not-exist"),
            ("changedCondition", ""),
        ]:
            self.assertEqual(
                audit.accepted_reviews([{**review, field: value}], "example", "new"), []
            )

    def test_project_and_stage_checks_cannot_use_stale_or_failed_receipts(self):
        original = audit.read
        for status in ("stale", "failed"):

            def changed(path):
                result = original(path)
                if path == "curriculum/project-validation.json":
                    first = result["projects"][0]
                    if status == "stale":
                        first["sourceHash"] = "historical"
                    else:
                        first["status"] = "failed"
                return result

            with patch.object(audit, "read", changed):
                result = audit.build()
            item = next(i for i in result["items"] if i["kind"] == "projects")
            self.assertFalse(item["metrics"]["currentReference"])
            self.assertEqual(
                item["level"], "in-development" if status == "stale" else "needs-repair"
            )
            stages = [t for t in result["tasks"] if t.get("projectId") == item["id"]]
            self.assertTrue(stages)
            self.assertTrue(all(t["level"] == item["level"] for t in stages))

    def test_a_missing_starter_marks_the_project_and_its_stages_for_repair(self):
        is_file = Path.is_file
        missing = audit.ROOT / "projects/regression-audit/starter/model.py"
        with patch.object(
            Path, "is_file", lambda file: file != missing and is_file(file)
        ):
            result = audit.build()
        project = next(
            i
            for i in result["items"]
            if i["kind"] == "projects" and i["id"] == "regression-audit"
        )
        self.assertEqual(project["level"], "needs-repair")
        self.assertIn(
            str(missing.relative_to(audit.ROOT)), project["metrics"]["missingFiles"]
        )
        self.assertTrue(
            all(
                t["level"] == "needs-repair"
                for t in result["tasks"]
                if t.get("projectId") == project["id"]
            )
        )

    def test_epub_chapters_and_phase_guides_keep_scoped_maturity(self):
        course = audit.read("curriculum/course.json")
        with ZipFile(audit.ROOT / "public/downloads/jax-foundations.epub") as book:
            chapter_index = 0
            for phase_index, phase in enumerate(course["phases"]):
                phase_html = book.read(f"OEBPS/phase-{phase_index}.xhtml").decode()
                self.assertIn(f'content-status.html#phases-{phase["id"]}', phase_html)
                for lesson in phase["lessons"]:
                    if lesson["status"] != "authored":
                        continue
                    chapter = book.read(f"OEBPS/chapter-{chapter_index}.xhtml").decode()
                    self.assertIn(
                        f'content-status.html#lessons-{lesson["id"]}', chapter
                    )
                    self.assertIn("Expert review pending", chapter)
                    chapter_index += 1
            self.assertIn(
                "content-status.html#book-reader",
                book.read("OEBPS/intro.xhtml").decode(),
            )

    def test_inventory_covers_canonical_lessons_projects_and_tasks_once(self):
        result = audit.build()
        course = audit.read("curriculum/course.json")
        lessons = [l for p in course["phases"] for l in p["lessons"]]
        self.assertEqual(
            {i["id"] for i in result["items"] if i["kind"] == "lessons"},
            {l["id"] for l in lessons},
        )
        self.assertEqual(
            result["summary"]["byKind"]["projects"],
            len(audit.read("curriculum/projects.json")),
        )
        self.assertEqual(
            len({(i["kind"], i["id"]) for i in result["items"]}), len(result["items"])
        )
        self.assertEqual(len({t["id"] for t in result["tasks"]}), len(result["tasks"]))
        self.assertTrue(
            all((audit.ROOT / item["source"]).is_file() for item in result["items"])
        )
        self.assertTrue(
            all(
                item["contribution"] and item["validation"] and item["review"]
                for item in result["items"]
            )
        )
        for lesson in lessons:
            content = audit.read(f'{lesson["path"]}/lesson.json')["content"]
            self.assertEqual(
                sum(t.get("lessonId") == lesson["id"] for t in result["tasks"]),
                2 + len(content.get("practice", [])),
            )


if __name__ == "__main__":
    unittest.main()
