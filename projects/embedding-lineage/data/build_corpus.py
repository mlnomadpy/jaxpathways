"""Extract a small, fixed set of attributed passages from the pinned Project Gutenberg source."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/source/alice-pg11.txt'
SPLITS = {'I': 'train', 'II': 'train', 'III': 'train', 'IV': 'train', 'V': 'validation', 'VI': 'test', 'VII': 'test', 'VIII': 'test'}
source = SOURCE.read_text(encoding='utf-8-sig')
source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
body = source.split('*** START OF THE PROJECT GUTENBERG EBOOK ALICE\'S ADVENTURES IN WONDERLAND ***', 1)[1]
body = body.split('*** END OF THE PROJECT GUTENBERG EBOOK', 1)[0]
parts = re.split(r'\n\s*CHAPTER\s+([IVXLCDM]+)\.\s*\n', body)
rows = []
for index in range(1, len(parts), 2):
    chapter = parts[index]
    if chapter not in SPLITS:
        continue
    title = next((line.strip() for line in parts[index + 1].splitlines() if line.strip()), '')
    paragraphs = [re.sub(r'\s+', ' ', item).strip() for item in re.split(r'\n\s*\n', parts[index + 1])]
    selected = [item for item in paragraphs if 110 <= len(item) <= 950 and not item.startswith('[') and not item.isupper()]
    count = min(32, len(selected))
    positions = sorted({round(i * (len(selected) - 1) / max(count - 1, 1)) for i in range(count)})
    for para_index, text in enumerate(selected[position] for position in positions):
        rows.append({
            'id': f'pg11-chapter-{chapter}-passage-{para_index:03d}',
            'group': f'pg11-chapter-{chapter}',
            'chapter': chapter,
            'chapter_title': title,
            'split': SPLITS[chapter],
            'source_id': 'gutenberg-11',
            'source_sha256': source_sha,
            'text': text,
        })
(ROOT / 'data/corpus.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
print(f'Wrote {len(rows)} attributed passages; split counts: ' + json.dumps({k: sum(r['split'] == k for r in rows) for k in set(SPLITS.values())}, sort_keys=True))
