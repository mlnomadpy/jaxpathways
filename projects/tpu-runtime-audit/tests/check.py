import argparse,importlib.util,json,tempfile
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
ROOT=Path(__file__).resolve().parents[1]
def load(w):
 p=ROOT/w/'model.py';s=importlib.util.spec_from_file_location('m',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def stage1(m):
 r=m.runtime_report();assert r['backend']=='cpu' and r['device_count']==jax.device_count() and r['stored_dtype']=='bfloat16' and r['compute_dtype']=='float32';np.testing.assert_allclose(r['output'],[[2.8125,-3.125],[.25,5.5]],rtol=.02,atol=.02)
def stage2(m):
 values=[1.,2.,3.];assert m.simulated_collective_sum(values)==6 and m.simulated_collective_sum(values,omitted=[1])==4
 try:m.simulated_collective_sum(values,omitted=[8]);raise AssertionError('invalid participant accepted')
 except ValueError:pass
def stage3(m,output=None):
 r=m.runtime_report();e={'backend':r['backend'],'devices':r['devices'],'stored_dtype':r['stored_dtype'],'compute_dtype':r['compute_dtype'],'scope':'CPU observation; simulated participant arithmetic; no TPU qualification'}
 p=(Path(output) if output else Path(tempfile.gettempdir()))/'tpu-runtime-audit-evidence.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(e));assert json.loads(p.read_text())==e
def run(m,n,output=None):
 f={'1':stage1,'2':stage2,'3':stage3};[(stage3(m,output) if k=='3' else f[k](m)) for k in (('1','2','3') if n in ('4','all') else (n,))];print('PASS tpu-runtime-audit stage '+n+': CPU runtime/dtype evidence and explicitly simulated collectives')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--implementation',default='starter');p.add_argument('--evidence-dir');p.add_argument('--stage',choices=('1','2','3','4','all'),default='all');a=p.parse_args();
 if a.evidence_dir: Path(a.evidence_dir).mkdir(parents=True,exist_ok=True);run(load(a.implementation),a.stage,a.evidence_dir)
 else:
  with tempfile.TemporaryDirectory(prefix='tpu-runtime-audit-') as td: run(load(a.implementation),a.stage,td)
