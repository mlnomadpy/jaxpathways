"""Stable publication IDs and deterministic fingerprints for EPUB 3 builds."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import re
import uuid
import xml.etree.ElementTree as ET

OPF = "http://www.idpf.org/2007/opf"
DC = "http://purl.org/dc/elements/1.1/"
FINGERPRINT_PROPERTY = "jp:rendered-edition-fingerprint"
FINGERPRINT_DOMAIN = b"jaxpathways-rendered-epub-v1\0"
RELEASE_VERSION = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def _identity_metadata(release: dict) -> tuple[str | None, str]:
    """Enforce paired edition metadata and its release-version cutoff."""
    version = release.get("version")
    match = RELEASE_VERSION.fullmatch(version) if isinstance(version, str) else None
    if match is None:
        raise ValueError("release version must use semantic major.minor.patch form")
    version_core = tuple(int(part) for part in match.groups())
    has_key = "editionKey" in release
    has_modified_at = "modifiedAt" in release
    if has_key != has_modified_at:
        raise ValueError("release editionKey and modifiedAt must be provided together")
    if not has_key:
        if version_core >= (0, 15, 0):
            raise ValueError("release version 0.15.0 and later requires editionKey and modifiedAt")
        return None, release["date"] + "T00:00:00Z"

    key = release["editionKey"]
    if not isinstance(key, str) or not key.strip() or key != key.strip():
        raise ValueError("release editionKey must be a non-empty trimmed string")
    value = release["modifiedAt"]
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
        raise ValueError("release modifiedAt must use UTC YYYY-MM-DDThh:mm:ssZ")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError("release modifiedAt must be a valid UTC date and time") from error
    return key, value


def modified_at(release: dict) -> str:
    """Return a validated pinned timestamp, or the legacy release-date fallback."""
    return _identity_metadata(release)[1]


def primary_identifier(release: dict, lessons: list) -> str:
    """Keep the historical ID for old records; use an explicit immutable edition key otherwise."""
    key, _ = _identity_metadata(release)
    if key is None:
        identity = hashlib.sha256(json.dumps(lessons, sort_keys=True).encode()).hexdigest()
        return f"urn:sha256:{identity}"
    stable_uuid = uuid.uuid5(uuid.NAMESPACE_URL, "https://www.tahabouhsine.com/jaxpathways/edition/" + key)
    return f"urn:uuid:{stable_uuid}"


def canonical_opf(package_bytes: bytes) -> bytes:
    """Canonicalize OPF while fixing fields that would make its fingerprint recursive."""
    package = ET.fromstring(package_bytes)
    metadata = package.find(f"{{{OPF}}}metadata")
    if metadata is None:
        raise ValueError("EPUB package document has no metadata")
    unique_id = package.attrib.get("unique-identifier")
    primary = metadata.find(f"{{{DC}}}identifier[@id='{unique_id}']") if unique_id else None
    if primary is None:
        raise ValueError("EPUB package document has no primary identifier")
    primary.text = "urn:jaxpathways:primary-identifier-excluded-from-rendered-fingerprint"
    fingerprints = [node for node in metadata.findall(f"{{{OPF}}}meta") if node.attrib.get("property") == FINGERPRINT_PROPERTY]
    if len(fingerprints) > 1:
        raise ValueError("EPUB package document has duplicate rendered edition fingerprints")
    for node in fingerprints:
        node.text = "sha256:rendered-edition-fingerprint-excluded-from-itself"
    return ET.tostring(package, encoding="utf-8", xml_declaration=True)


def rendered_fingerprint(payload: dict[str, bytes], package_bytes: bytes) -> str:
    """Hash ordered member names/bytes and the identifier-neutral canonical package document."""
    digest = hashlib.sha256(FINGERPRINT_DOMAIN)
    for name in sorted(payload):
        path = name.encode("utf-8")
        body = payload[name]
        digest.update(len(path).to_bytes(8, "big"))
        digest.update(path)
        digest.update(len(body).to_bytes(8, "big"))
        digest.update(body)
    package = canonical_opf(package_bytes)
    digest.update(len(package).to_bytes(8, "big"))
    digest.update(package)
    return digest.hexdigest()
