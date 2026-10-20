import argparse,importlib.util,json,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def load(w):
 p=ROOT/w/'model.py';s=importlib.util.spec_from_file_location('m',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def stage1(m):
 r=m.systems_report(100,2,4,2,160.,4,50.);assert (r['weight_bytes'],r['master_bytes'],r['moment_bytes'],r['total_bytes'])==(200,400,800,1400);assert r['ideal_rate']==200 and np.isclose(r['efficiency'],.8)
def stage2(m):
 r=m.systems_report(50,2,4,2,300.,2,50.);assert r['total_bytes']==700 and r['ideal_rate']==100 and r['efficiency']==3.;
 try:m.systems_report(100,2,4,2,0.,4,50.);raise AssertionError('zero throughput accepted')
 except ValueError:pass
def stage3(m,output=None):
 r=m.systems_report(100,2,4,2,160.,4,50.);e={**r,'scope':'arithmetic capacity estimate and supplied-rate calculation; no device memory or TPU scaling measurement'}
 p=(Path(output) if output else Path(tempfile.gettempdir()))/'tpu-systems-audit-evidence.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(e,sort_keys=True));assert json.loads(p.read_text())==e
def run(m,n,output=None):
 f={'1':stage1,'2':stage2,'3':stage3};[(stage3(m,output) if k=='3' else f[k](m)) for k in (('1','2','3') if n in ('4','all') else (n,))];print('PASS tpu-systems-audit stage '+n+': tensor-byte ledger, changed rates, and scoped report')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--implementation',default='starter');p.add_argument('--evidence-dir');p.add_argument('--stage',choices=('1','2','3','4','all'),default='all');a=p.parse_args();
 if a.evidence_dir: Path(a.evidence_dir).mkdir(parents=True,exist_ok=True);run(load(a.implementation),a.stage,a.evidence_dir)
 else:
  with tempfile.TemporaryDirectory(prefix='tpu-systems-audit-') as td: run(load(a.implementation),a.stage,td)
