"""Export canonical assessment metadata and source downloads; Astro owns web markup."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'public/assessments'
OUT.mkdir(parents=True, exist_ok=True)
entries = json.loads((ROOT / 'curriculum/assessments.json').read_text())
for entry in entries:
    source = ROOT / entry['source']
    (OUT / source.name).write_bytes(source.read_bytes())
    notes = ROOT / 'assessments' / (entry['id'] + '-reviewer.md')
    if notes.is_file():
        (OUT / notes.name).write_bytes(notes.read_bytes())
(ROOT / 'public/api/v1/assessments.json').write_text(
    json.dumps(dict(schemaVersion=1, assessments=entries), indent=2) + '\n'
)
print(f'Exported {len(entries)} synthesis assessment drafts for Astro and download.')
