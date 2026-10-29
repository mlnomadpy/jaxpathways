"""Independent correction audit; no candidate inventory imports."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib
import json
import re
import struct
import unicodedata

ROOT=Path(__file__).resolve().parent
HANDOFF=Path('/private/tmp/jaxpathways-c04-luna-20261021-a1a3-corrected-v3')
LOCK=HANDOFF/'corrected-handoff-lock.json'
sha=lambda data:hashlib.sha256(data).hexdigest()
lock=json.loads(LOCK.read_text())
assert sha(LOCK.read_bytes())=='88e0777f92a0d53d9acacb44f8b6d62d787b209bbad9fdc9bbc1a2b566f18c76'
verified=0
for category in ('source_files','external_evidence_files'):
    for filename,identity in lock[category].items():
        p=Path(filename);h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        assert p.stat().st_size==identity['bytes'] and h.hexdigest()==identity['sha256'],filename
        verified+=1
for name in ('model.py','config.py'):
    assert sha((ROOT/'source'/name).read_bytes())==lock['source_files'][str(Path('/Users/tahabsn/Documents/ChatGPT/jaxpathways/projects/audio-model-300m')/name)]['sha256']

inventory=HANDOFF/'inventory'
rows=list(map(json.loads,(inventory/'utterances.jsonl').read_text().splitlines()))
old=list(map(json.loads,Path('/private/tmp/jaxpathways-c04-luna-20261021/inventory/utterances.jsonl').read_text().splitlines()))
old={r['source']+':'+r['id']:r for r in old}
assert len(rows)==len(old)==5472
chapters={};books={};book_continuation_lines=0
for p in (inventory/'documents').glob('*CHAPTERS.TXT'):
    lines=p.read_text().splitlines()
    header=next(s for s in lines if s.startswith(';ID') and 'BOOK ID' in s)
    fields=[s.strip() for s in header.lstrip(';').split('|')]
    index=fields.index('BOOK ID');assert index==5
    for line in lines:
        if not line.strip() or line.lstrip().startswith(';'):continue
        fields=[s.strip() for s in line.split('|')]
        value=fields[index]
        assert chapters.setdefault(fields[0],value)==value
for p in (inventory/'documents').glob('*BOOKS.TXT'):
    for line in p.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith(';'):continue
        fields=[s.strip() for s in line.split('|')]
        if not fields[0].isdigit():
            book_continuation_lines+=1
            continue
        assert books.setdefault(fields[0],fields[1])==fields[1]
assert chapters['134315']=='2688' and books['2688']=='The Clue of the Twisted Candle'

alphabet=" 'ABCDEFGHIJKLMNOPQRSTUVWXYZ"
assert [alphabet.index(c)+1 for c in "A A'Z"]==[3,1,3,2,28]
edges=defaultdict(list);by_id={};removed=Counter()
for row in rows:
    identity=row['source']+':'+row['id'];assert identity not in by_id
    by_id[identity]=row
    previous=old[identity]
    for key in ('archive_path','channels','sample_rate','samples','seconds','raw_member_sha256','raw_member_bytes','decoded_f32le_sha256','transcript','speaker_id','chapter_id'):
        assert row[key]==previous[key],(identity,key)
    book=chapters[row['chapter_id'].split('-')[-1]]
    assert row['book_id']==book and row['book_title']==books.get(book,'')
    text=unicodedata.normalize('NFKC',row['transcript']).upper().replace('’',"'").replace('‘',"'")
    removed.update(c for c in text if c not in alphabet)
    normalized=' '.join(''.join(c if c in alphabet else ' ' for c in text).split())
    assert row['normalized_asr_candidate']==normalized
    assert row['normalized_asr_sha256']==sha(normalized.encode())
    tokens=[alphabet.index(c)+1 for c in normalized]
    assert row['ctc_tokens_sha256']==sha(struct.pack('<'+'i'*len(tokens),*tokens))
    duplicate=' '.join(unicodedata.normalize('NFKC',row['transcript']).casefold().split())
    assert duplicate==row['normalized_duplicate_key']
    for key,value in [('speaker',row['speaker_id']),('chapter',row['chapter_id']),('book',book),('audio',row['decoded_f32le_sha256']),('text',duplicate)]:
        if value:edges[key,value].append(identity)

# Build an adjacency graph and traverse it; no candidate union-find code.
graph={identity:set() for identity in by_id}
for identities in edges.values():
    anchor=identities[0]
    for identity in identities[1:]:graph[anchor].add(identity);graph[identity].add(anchor)
remaining=set(graph);groups={}
while remaining:
    first=min(remaining);pending=[first];seen=set()
    while pending:
        identity=pending.pop()
        if identity in seen:continue
        seen.add(identity);pending.extend(graph[identity]-seen)
    remaining-=seen
    component=sha('\n'.join(sorted(seen)).encode())
    groups[component]=seen
    assert all(by_id[identity]['component_id']==component for identity in seen)
claimed=json.loads((inventory/'components.json').read_text())
assert len(groups)==len(claimed)==79
for c in claimed:
    assert set(c['utterance_ids'])==groups[c['component_id']]
cross=[k for k,v in groups.items() if len({by_id[i]['source'] for i in v})>1]
assert len(cross)==2
result={'status':'pass','scope':'53 immutable source/evidence identities; metadata-column/known-row audit; unchanged earlier verified raw/PCM/transcript fields; independent normalization and all target hashes; adjacency-graph reconstruction. Does not newly decode archives, approve proposed roles/recipe or authorize corpus optimization.', 'lock_sha256':sha(LOCK.read_bytes()),'locked_files_verified':verified,'rows':len(rows),'target_hashes_reconstructed':len(rows),'components':len(groups),'cross_source_components':len(cross),'raw_books_nonnumeric_continuation_lines':book_continuation_lines,'raw_normalization_removed_symbol_counts':dict(removed),'roles_approved':False,'optimization_authorized':False,'checker_sha256':sha(Path(__file__).read_bytes())}
(ROOT/'inventory-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
