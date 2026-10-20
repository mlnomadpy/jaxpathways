import argparse,importlib.util,json,tempfile
from pathlib import Path
import jax.numpy as jnp,numpy as np
ROOT=Path(__file__).resolve().parents[1]
def load(w):
 p=ROOT/w/'model.py';s=importlib.util.spec_from_file_location('m',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def stage1(m):
 with tempfile.TemporaryDirectory() as td:
  r=m.performance_report(td,4); expected=np.tanh(np.eye(16,dtype=np.float32)); np.testing.assert_allclose(r['output'],expected,rtol=1e-6); assert r['backend']=='cpu' and r['output_shape']==[16,16] and all(x>0 for x in r['steady_seconds'])
def stage2(m):
 with tempfile.TemporaryDirectory() as td:
  r=m.performance_report(td,4); assert r['shape_trace_count']==2 and r['trace_count']==2 and r['shapes']==[[16,16],[32,32]]; np.testing.assert_allclose(r['changed_shape_output'],np.tanh(np.eye(32,dtype=np.float32)),rtol=1e-6); assert not r['memory_measured']
def stage3(m,output=None):
 d=Path(output) if output else Path(tempfile.gettempdir())/'performance-audit-evidence'
 r=m.performance_report(d,4); report=json.loads((d/'report.json').read_text()); trace=(d/'workload.jaxpr.txt').read_text(); assert report['shape_trace_count']==2 and report['trace_count']==2 and 'dot_general' in trace and len(trace)>50
 np.testing.assert_allclose(r['output'],np.tanh(np.eye(16,dtype=np.float32)),rtol=1e-6)
def run(m,n,output=None):
 fs={'1':stage1,'2':stage2,'3':stage3};
 for k in (('1','2','3') if n in ('4','all') else (n,)):(stage3(m,output) if k=='3' else fs[k](m))
 print('PASS performance-audit stage '+n+': independent result, changed-shape traces, retained JAXPR/report; memory not measured')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--implementation',default='starter');p.add_argument('--evidence-dir');p.add_argument('--stage',choices=('1','2','3','4','all'),default='all');a=p.parse_args();
 if a.evidence_dir: Path(a.evidence_dir).mkdir(parents=True,exist_ok=True);run(load(a.implementation),a.stage,a.evidence_dir)
 else:
  with tempfile.TemporaryDirectory(prefix='performance-audit-') as td: run(load(a.implementation),a.stage,td)
