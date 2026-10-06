"""Run every registered reference project, with bounded subprocesses and source hashes."""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
records = []
for relative in json.loads((ROOT/'curriculum/projects.json').read_text()):
    manifest_path = ROOT/relative
    manifest = json.loads(manifest_path.read_text())
    command = shlex.split(manifest['referenceCommand'])
    if command[0] not in ('python', 'python3'):
        raise ValueError(f"Unsupported reference runner for {manifest['id']}")
    script = (ROOT/command[1]).resolve()
    if not script.is_relative_to(manifest_path.parent.resolve()) or not script.is_file():
        raise ValueError('Reference checker must be inside its project')
    files = sorted(p for p in manifest_path.parent.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    source_hash = hashlib.sha256()
    for file in files:
        source_hash.update(str(file.relative_to(ROOT)).encode()+b'\0'+file.read_bytes()+b'\0')
    result = subprocess.run([sys.executable, *command[1:]], cwd=ROOT, capture_output=True, text=True, timeout=240)
    print(result.stdout, end='')
    if result.returncode:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"FAIL reference project: {manifest['id']}")
    records.append(dict(projectId=manifest['id'],sourceHash=source_hash.hexdigest(),command=manifest['referenceCommand'],stdout=result.stdout,status='passed'))
receipt = dict(schemaVersion=1,executedAt=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Registered public reference checkers only; not learner assessment or accelerator validation.',projects=records)
(ROOT/'curriculum/project-validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(f'PASS: {len(records)} registered project reference suites')
