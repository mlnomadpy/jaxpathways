#!/usr/bin/env python3
"""Hash all public packet, teaching, and authoring-tool files."""
import hashlib,json,pathlib
PROJECT=pathlib.Path(__file__).resolve().parents[1]
OUT=PROJECT/'reference/evidence/v14/packet-manifest.json'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
files={}
for rel in ('TEACHING.md','reference/README.md'):
 p=PROJECT/rel
 if p.is_file(): files[rel]=digest(p)
for base in ('tools','reference/evidence/v14'):
 for p in sorted((PROJECT/base).rglob('*')):
  if p.is_file() and p!=OUT and '__pycache__' not in p.parts and p.suffix!='.pyc': files[str(p.relative_to(PROJECT))]=digest(p)
OUT.write_text(json.dumps({'schema':'c03-v14-public-packet-manifest-v1','hash_algorithm':'sha256','files':files},sort_keys=True,separators=(',',':'))+'\n')
print(json.dumps({'manifest':str(OUT),'files':len(files),'sha256':digest(OUT)},sort_keys=True))
