"""Publish the TPU guides with exact workers and checked CPU figure provenance."""
import hashlib
import json
from pathlib import Path
import re
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public/downloads"
OUT.mkdir(parents=True, exist_ok=True)
reference_root = ROOT / "resources/tpu-gcp"
# This portable bundle supports the standalone single-host target. The coordinated
# multi-host target stays repository-only because it requires a shared GCS checkpoint
# and a matching multi-worker launch contract.
TPU_WORKSPACE_FILES = (
    "resources/tpu-gcp/check.py",
    "resources/tpu-gcp/check_precision.py",
    "resources/tpu-gcp/launch.py",
    "resources/tpu-gcp/precision_profile.py",
    "resources/tpu-gcp/render_precision.py",
    "resources/tpu-gcp/requirements-qualification.txt",
    "resources/tpu-gcp/collectives_target.py",
    "resources/tpu-gcp/check_collectives_target.py",
    "phases/09-distributed/03-communication-efficient-algorithms/code/main.py",
)
provenance = json.loads((reference_root / "reference/provenance.json").read_text())
for relative, expected in provenance['sha256'].items():
    observed = hashlib.sha256((reference_root / relative).read_bytes()).hexdigest()
    assert observed == expected, f"TPU precision reference is stale: {relative}; rerun lab and plot before publishing."
report = json.loads((reference_root / 'reference/report.json').read_text())
assert report['platform'] == 'cpu', 'The published illustration must retain its recorded CPU scope.'
assert report['source_sha256'] == provenance['sha256']['precision_profile.py']
reference_out = OUT / 'tpu-performance'
reference_out.mkdir(exist_ok=True)
for path in sorted((reference_root / 'reference').glob('*')):
    shutil.copyfile(path, reference_out / path.name)
for name in ['precision_profile.py', 'render_precision.py']:
    shutil.copyfile(reference_root / name, reference_out / name)

guides = {}
for name in ['tpu-gcp', 'tpu-performance']:
    guide = (ROOT / f"content/guides/{name}.md").read_text()
    guide = re.sub(r"\]\((?!https?://|#)([^)]+)\)",
                   r"](https://www.tahabouhsine.com/jaxpathways/\1)", guide)
    (OUT / f"{name}.md").write_text(guide)
    guides[name] = guide
with ZipFile(OUT / "jax-tpu-gcp.zip", "w", ZIP_DEFLATED) as bundle:
    included = set()
    for relative in TPU_WORKSPACE_FILES:
        path = ROOT / relative
        archive_name = "jax-tpu-gcp/" + path.relative_to(ROOT).as_posix()
        assert archive_name not in included, f"duplicate TPU workspace path: {archive_name}"
        bundle.write(path, archive_name)
        included.add(archive_name)
    for path in sorted((reference_root / 'reference').glob('*')):
        archive_name = "jax-tpu-gcp/" + path.relative_to(ROOT).as_posix()
        assert archive_name not in included, f"duplicate TPU workspace path: {archive_name}"
        bundle.write(path, archive_name)
        included.add(archive_name)
    worker = Path("projects/workload-operations/solution/model.py")
    worker_name = "jax-tpu-gcp/" + worker.as_posix()
    assert worker_name not in included, f"duplicate TPU workspace path: {worker_name}"
    bundle.write(ROOT / worker, worker_name)
    included.add(worker_name)
    bundle.writestr("jax-tpu-gcp/GUIDE.md", guides['tpu-gcp'])
    included.add("jax-tpu-gcp/GUIDE.md")
    bundle.writestr("jax-tpu-gcp/PERFORMANCE.md", guides['tpu-performance'])
    included.add("jax-tpu-gcp/PERFORMANCE.md")
print("Built TPU/GCP guides, precision figures and practice bundle; no cloud resources created.")
