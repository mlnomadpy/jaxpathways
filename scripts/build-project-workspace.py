"""Publish the canonical workflow guide and small, dependency-free practice bundle."""
from pathlib import Path
import re
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "resources/project-workspace"
OUT = ROOT / "public/downloads"
OUT.mkdir(parents=True, exist_ok=True)
guide = (ROOT / "content/guides/project-workflow.md").read_text()
# The portable Markdown must resolve course links outside the website too.
guide = re.sub(r"\]\((?!https?://|#)([^)]+)\)",
               r"](https://www.tahabouhsine.com/jaxpathways/\1)", guide)
(OUT / "project-workflow.md").write_text(guide)
with ZipFile(OUT / "jax-project-workspace.zip", "w", ZIP_DEFLATED) as bundle:
    for path in sorted(SOURCE.rglob("*")):
        relative = path.relative_to(SOURCE)
        if path.is_file() and not any(part in {"__pycache__", ".venv", ".git", "runs"} for part in relative.parts):
            bundle.write(path, "jax-project-workspace/" + relative.as_posix())
    bundle.writestr("jax-project-workspace/GUIDE.md", guide)
print("Built project workflow guide and portable CLI practice workspace.")
