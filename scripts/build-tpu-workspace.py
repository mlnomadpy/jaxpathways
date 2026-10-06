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
    for path in sorted(reference_root.glob("*.py")):
        bundle.write(path, "jax-tpu-gcp/" + path.relative_to(ROOT).as_posix())
    for path in sorted((reference_root / 'reference').glob('*')):
        bundle.write(path, "jax-tpu-gcp/" + path.relative_to(ROOT).as_posix())
    worker = Path("projects/workload-operations/solution/model.py")
    bundle.write(ROOT / worker, "jax-tpu-gcp/" + worker.as_posix())
    bundle.writestr("jax-tpu-gcp/GUIDE.md", guides['tpu-gcp'])
    bundle.writestr("jax-tpu-gcp/PERFORMANCE.md", guides['tpu-performance'])
print("Built TPU/GCP guides, precision figures and practice bundle; no cloud resources created.")
