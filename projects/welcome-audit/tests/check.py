import argparse,importlib.util,json,tempfile
from pathlib import Path
import jax,numpy as np
ROOT=Path(__file__).resolve().parents[1]
def load(which):
 p=ROOT/which/'model.py'; spec=importlib.util.spec_from_file_location('m',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def stage1(m):
 r=m.environment_report(); expected=np.array([[5,11],[11,25]],dtype=np.float32); np.testing.assert_array_equal(r['gram'],expected); assert r['backend']=='cpu' and len(r['devices'])==jax.device_count()
def stage2(m):
 r=m.environment_report(); changed=np.asarray(r['input'])+np.eye(2); np.testing.assert_array_equal(changed@changed.T,np.array([[8,16],[16,34]]))
def stage3(m,output=None):
 p=(Path(output) if output else Path(tempfile.gettempdir()))/'welcome-audit-receipt.json';p.parent.mkdir(parents=True,exist_ok=True)
 r=m.environment_report(p); saved=json.loads(p.read_text()); assert saved['gram']==r['gram'] and saved['devices']==r['devices']
def run(m,n,output=None):
 fns={'1':stage1,'2':stage2,'3':stage3}
 if n in ('4','all'):
  for i in ('1','2','3'): (stage3(m,output) if i=='3' else fns[i](m))
 else:
  i=n; (stage3(m,output) if i=='3' else fns[i](m))
 print('PASS welcome-audit stage '+n+': CPU environment and saved arithmetic identity')
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--implementation',default='starter'); p.add_argument('--evidence-dir');p.add_argument('--stage',choices=('1','2','3','4','all'),default='all'); a=p.parse_args();
 if a.evidence_dir: Path(a.evidence_dir).mkdir(parents=True,exist_ok=True);run(load(a.implementation),a.stage,a.evidence_dir)
 else:
  with tempfile.TemporaryDirectory(prefix='welcome-audit-') as td: run(load(a.implementation),a.stage,td)
