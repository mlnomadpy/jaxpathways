"""Publish the TPU guide with the exact shared numerical worker it documents."""
from pathlib import Path
import re
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public/downloads"
OUT.mkdir(parents=True, exist_ok=True)
guide = (ROOT / "content/guides/tpu-gcp.md").read_text()
guide = re.sub(r"\]\((?!https?://|#)([^)]+)\)",
               r"](https://www.tahabouhsine.com/jaxpathways/\1)", guide)
(OUT / "tpu-gcp.md").write_text(guide)
with ZipFile(OUT / "jax-tpu-gcp.zip", "w", ZIP_DEFLATED) as bundle:
    for path in sorted((ROOT / "resources/tpu-gcp").glob("*.py")):
        bundle.write(path, "jax-tpu-gcp/" + path.relative_to(ROOT).as_posix())
    worker = Path("projects/workload-operations/solution/model.py")
    bundle.write(ROOT / worker, "jax-tpu-gcp/" + worker.as_posix())
    bundle.writestr("jax-tpu-gcp/GUIDE.md", guide)
print("Built TPU/GCP guide and local training launcher bundle; no cloud resources created.")
