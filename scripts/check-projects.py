"""Run every registered reference project, with bounded subprocesses and source hashes."""
import datetime
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--shard', help='Zero-based shard index/count, for example 0/4')
parser.add_argument('--receipt', help='Output receipt path; sharded runs require a separate path')
args = parser.parse_args()
index, count = 0, 1
if args.shard:
    try:
        index, count = map(int, args.shard.split('/'))
    except ValueError:
        parser.error('--shard must be index/count')
    if count < 1 or not 0 <= index < count:
        parser.error('shard must satisfy 0 <= index < count')
    if not args.receipt:
        parser.error('sharded runs require --receipt to preserve the complete registry receipt')
manifests = json.loads((ROOT/'curriculum/projects.json').read_text())
selected = manifests[index::count]
if not selected:
    parser.error('this shard contains no projects')
records = []
for relative in selected:
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
receipt['shard'] = {'index': index, 'count': count, 'totalProjects': len(manifests)}
destination = ROOT / (args.receipt or 'curriculum/project-validation.json')
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(receipt,indent=2)+'\n')
print(f'PASS: {len(records)} registered project reference suites')
