#!/usr/bin/env python3
"""Run adversarial, no-training checks against isolated copies of the packet."""
from __future__ import annotations
import hashlib, importlib.util, json, pathlib, shutil, tempfile
PROJECT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('c03_packet_checker',PROJECT/'tools/check_v14_packet.py')
checker=importlib.util.module_from_spec(spec); spec.loader.exec_module(checker)
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def refresh_manifest(project):
 p=project/'reference/evidence/v14/packet-manifest.json'; m=json.loads(p.read_text())
 m['files']={rel:digest(project/rel) for rel in sorted(m['files']) if (project/rel).is_file()}
 p.write_text(json.dumps(m,sort_keys=True,separators=(',',':'))+'\n')
def snapshot(dest):
 shutil.copytree(PROJECT,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
def expect_failure(name,mutate):
 with tempfile.TemporaryDirectory(prefix='c03-v14-negative-') as td:
  project=pathlib.Path(td)/'image-model-300m'; snapshot(project); mutate(project)
  try: checker.verify(project/'reference/evidence/v14',project,True)
  except Exception as exc: return {'control':name,'result':'rejected','reason':str(exc)}
  raise SystemExit(f'negative control unexpectedly passed: {name}')
def flip_source(p):
 f=p/'README.md'; b=bytearray(f.read_bytes()); b[0]^=1; f.write_bytes(b)
def edit_json(p,rel,fn):
 f=p/rel; x=json.loads(f.read_text()); fn(x); f.write_text(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n'); refresh_manifest(p)
def change_logit(p):
 def mutate(x): x['development']['arms']['baseline-vit'][0][0]+=1000.0
 edit_json(p,'reference/evidence/v14/public-numeric-evidence.json',mutate)
def change_label(p):
 def mutate(x): x['final']['labels'][0]=(x['final']['labels'][0]+1)%5
 edit_json(p,'reference/evidence/v14/public-numeric-evidence.json',mutate)
def change_approval(p):
 f=p/'reference/evidence/v14/approved-v14-approval.json'; b=bytearray(f.read_bytes()); b[-2]^=1; f.write_bytes(b); refresh_manifest(p)
def change_manifest(p):
 f=p/'reference/evidence/v14/packet-manifest.json'; m=json.loads(f.read_text()); m['files'].pop(next(iter(m['files']))); f.write_text(json.dumps(m,sort_keys=True,separators=(',',':'))+'\n')
def change_figure(p):
 def mutate(x): x['figures']['quality-classification']['bars']['baseline-vit']+=0.02
 edit_json(p,'reference/evidence/v14/figure-data.json',mutate)
def add_payload(p):
 (p/'reference/evidence/v14/unexpected-photo.jpg').write_bytes(b'not an image; payload-type control')
def add_weight(p):
 (p/'reference/evidence/v14/unexpected-weights.npz').write_bytes(b'not an archive; payload-type control')
def main():
 controls=[('changed-source-byte',flip_source),('changed-public-logit',change_logit),('changed-public-label',change_label),('changed-approval-byte',change_approval),('tampered-manifest',change_manifest),('tampered-figure-input',change_figure),('accidental-photo-payload',add_payload),('accidental-weight-payload',add_weight)]
 print(json.dumps({'status':'PASS','negative_controls':[expect_failure(n,f) for n,f in controls]},sort_keys=True,indent=2))
if __name__=='__main__': main()
