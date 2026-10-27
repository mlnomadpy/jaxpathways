"""Focused identity and rendered-fingerprint regressions for the EPUB builder."""
from copy import deepcopy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from epub_identity import FINGERPRINT_PROPERTY, modified_at, primary_identifier, rendered_fingerprint


def package(release, identifier, fingerprint="sha256:pending"):
    return (
        '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0" '
        'unique-identifier="book-id" prefix="jp: https://www.tahabouhsine.com/jaxpathways/epub-meta#">'
        '<metadata><dc:identifier id="book-id">' + identifier + '</dc:identifier>'
        '<dc:title>JAX Pathways ' + release["version"] + ' · ' + release["title"] + '</dc:title>'
        '<dc:language>en</dc:language><dc:date>' + release["date"] + '</dc:date>'
        '<meta property="dcterms:modified">' + modified_at(release) + '</meta>'
        '<meta property="' + FINGERPRINT_PROPERTY + '">' + fingerprint + '</meta></metadata>'
        '<manifest><item id="intro" href="intro.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/>'
        '<item id="style" href="style.css" media-type="text/css"/></manifest>'
        '<spine><itemref idref="intro"/><itemref idref="chapter"/></spine></package>'
    ).encode()


def fingerprint(release, payload, lessons):
    identifier = primary_identifier(release, lessons)
    opf = package(release, identifier)
    return identifier, rendered_fingerprint(payload, opf)


class EpubIdentityTests(unittest.TestCase):
    def setUp(self):
        self.release = {
            "version": "0.15.0",
            "date": "2026-10-21",
            "title": "Review edition",
            "modifiedAt": "2026-10-21T19:30:00Z",
            "editionKey": "jaxpathways-2026-review-edition",
        }
        self.lessons = [{"id": "lesson-one", "content": {"text": "lesson"}}]
        self.payload = {
            "mimetype": b"application/epub+zip",
            "META-INF/container.xml": b"<container/>",
            "OEBPS/intro.xhtml": b"intro",
            "OEBPS/chapter.xhtml": b"lesson",
            "OEBPS/phase-1.xhtml": b"phase guide",
            "OEBPS/appendix-workflow.xhtml": b"appendix",
            "OEBPS/style.css": b"body { color: #123; }",
            "OEBPS/figures/chart.png": b"image-bytes",
        }

    def test_rebuild_is_stable_and_extracted_members_recompute(self):
        first_id, first_fingerprint = fingerprint(self.release, self.payload, self.lessons)
        second_id, second_fingerprint = fingerprint(self.release, self.payload, self.lessons)
        self.assertEqual((first_id, first_fingerprint), (second_id, second_fingerprint))
        opf = package(self.release, first_id, "sha256:" + first_fingerprint)
        with tempfile.TemporaryDirectory() as temporary:
            book_path = Path(temporary) / "book.epub"
            with ZipFile(book_path, "w") as book:
                book.writestr("mimetype", self.payload["mimetype"], compress_type=ZIP_STORED)
                for name in sorted(set(self.payload) - {"mimetype"}):
                    book.writestr(name, self.payload[name])
                book.writestr("OEBPS/package.opf", opf)
            with ZipFile(book_path) as book:
                extracted = {name: book.read(name) for name in book.namelist() if name != "OEBPS/package.opf"}
                recomputed = rendered_fingerprint(extracted, book.read("OEBPS/package.opf"))
        self.assertEqual(recomputed, first_fingerprint)

    def test_edition_key_changes_publication_id_without_hashing_the_id(self):
        identifier, before = fingerprint(self.release, self.payload, self.lessons)
        changed = deepcopy(self.release)
        changed["editionKey"] = "jaxpathways-2027-new-edition"
        new_identifier, after = fingerprint(changed, self.payload, self.lessons)
        self.assertNotEqual(identifier, new_identifier)
        self.assertEqual(before, after)

    def test_release_metadata_changes_rendered_fingerprint(self):
        _, baseline = fingerprint(self.release, self.payload, self.lessons)
        for field, value in (
            ("version", "0.15.1"),
            ("title", "Review edition with errata"),
            ("date", "2026-10-22"),
            ("modifiedAt", "2026-10-22T00:00:00Z"),
        ):
            with self.subTest(field=field):
                changed = deepcopy(self.release)
                changed[field] = value
                identifier, current = fingerprint(changed, self.payload, self.lessons)
                self.assertEqual(identifier, primary_identifier(self.release, self.lessons))
                self.assertNotEqual(baseline, current)

    def test_each_rendered_resource_class_changes_fingerprint_and_keeps_id(self):
        identifier, baseline = fingerprint(self.release, self.payload, self.lessons)
        for name in (
            "OEBPS/phase-1.xhtml",
            "OEBPS/appendix-workflow.xhtml",
            "OEBPS/style.css",
            "OEBPS/figures/chart.png",
            "OEBPS/chapter.xhtml",
        ):
            with self.subTest(member=name):
                changed = dict(self.payload)
                changed[name] += b" revised"
                next_identifier, next_fingerprint = fingerprint(self.release, changed, self.lessons)
                self.assertEqual(identifier, next_identifier)
                self.assertNotEqual(baseline, next_fingerprint)

    def test_legacy_id_and_modified_date_fallback_remain_compatible(self):
        legacy = {"version": "0.14.1", "date": "2026-10-21"}
        expected = "urn:sha256:" + hashlib.sha256(
            json.dumps(self.lessons, sort_keys=True).encode()
        ).hexdigest()
        self.assertEqual(primary_identifier(legacy, self.lessons), expected)
        self.assertEqual(modified_at(legacy), "2026-10-21T00:00:00Z")
        for invalid in ("2026-10-21", "2026-10-21T19:30:00+00:00", "2026-02-30T00:00:00Z"):
            release = {**legacy, "editionKey": self.release["editionKey"], "modifiedAt": invalid}
            for operation in (lambda: modified_at(release), lambda: primary_identifier(release, self.lessons)):
                with self.subTest(value=invalid), self.assertRaises(ValueError):
                    operation()

    def test_new_releases_require_both_identity_fields(self):
        for version in ("0.15.0", "0.15.1", "1.0.0"):
            with self.subTest(version=version):
                missing_both = {"version": version, "date": "2026-10-21"}
                for operation in (lambda: primary_identifier(missing_both, self.lessons), lambda: modified_at(missing_both)):
                    with self.assertRaisesRegex(ValueError, "requires editionKey and modifiedAt"):
                        operation()

    def test_semver_release_gate_accepts_valid_current_and_future_versions(self):
        for version in ("0.15.0", "1.0.0"):
            with self.subTest(version=version):
                release = {**self.release, "version": version}
                self.assertTrue(primary_identifier(release, self.lessons).startswith("urn:uuid:"))
                self.assertEqual(modified_at(release), self.release["modifiedAt"])

    def test_semver_numeric_components_reject_leading_zeroes(self):
        for version in ("01.15.0", "0.015.0", "0.15.00"):
            with self.subTest(version=version):
                release = {**self.release, "version": version}
                for operation in (lambda: primary_identifier(release, self.lessons), lambda: modified_at(release)):
                    with self.assertRaisesRegex(ValueError, "semantic major.minor.patch"):
                        operation()

    def test_release_version_rejects_prerelease_build_and_malformed_suffixes(self):
        for version in ("0.15.0-rc.1", "0.15.0+meta", "0.15.0-.."):
            with self.subTest(version=version):
                release = {**self.release, "version": version}
                for operation in (lambda: primary_identifier(release, self.lessons), lambda: modified_at(release)):
                    with self.assertRaisesRegex(ValueError, "semantic major.minor.patch"):
                        operation()

    def test_partial_identity_pair_fails_closed_at_any_release_version(self):
        for version in ("0.14.1", "0.15.0"):
            for partial in (
                {"editionKey": self.release["editionKey"]},
                {"modifiedAt": self.release["modifiedAt"]},
            ):
                release = {"version": version, "date": "2026-10-21", **partial}
                for operation in (lambda: primary_identifier(release, self.lessons), lambda: modified_at(release)):
                    with self.subTest(version=version, partial=partial), self.assertRaisesRegex(ValueError, "provided together"):
                        operation()

    def test_invalid_edition_key_fails_closed(self):
        for invalid in ("", "   ", " padded ", None):
            release = {**self.release, "editionKey": invalid}
            for operation in (lambda: primary_identifier(release, self.lessons), lambda: modified_at(release)):
                with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "editionKey"):
                    operation()


if __name__ == "__main__":
    unittest.main()
