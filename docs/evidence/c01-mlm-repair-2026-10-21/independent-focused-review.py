from pathlib import Path
import ast,hashlib,importlib.util,json,sys
import jax,jax.numpy as jnp
P=Path('/Users/tahabsn/Documents/ChatGPT/jaxpathways/projects/embedding-model-300m');O=Path('/private/tmp/c01-sol-checker-repair2-review');O.mkdir(exist_ok=True);sys.path.insert(0,str(P))
spec=importlib.util.spec_from_file_location('review_checker',P/'tests/check.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(P/'tests/check.py')=='a3867129f43df1a1b6d8bf671d449488987e1339f7abfcae0caf35fc88c6d952';assert sha(P/'model.py')=='0d50093ea497730be48007265b3db25b9044fa13b88a4485cfce7e4c45266f90'
source=(P/'model.py').read_text()
def variant(name):
 t=ast.parse(source)
 for node in t.body:
  if isinstance(node,ast.FunctionDef):
   if name=='equivalent' and node.name=='mlm_logits':
    class Rename(ast.NodeTransformer):
     def visit_Name(self,n):
      if n.id in {'hidden','head','transformed'}:n.id='renamed_'+n.id
      return n
    node=Rename().visit(node)
   if name=='zero_head' and node.name=='mlm_logits':
    node.body=ast.parse("return jnp.zeros((*jnp.asarray(input_ids).shape, _config(config).vocab_size), dtype=jnp.float32)").body
   if (name=='stopped_encoder' and node.name=='hidden_states') or (name=='stopped_head' and node.name=='mlm_logits'):
    assert isinstance(node.body[-1],ast.Return);node.body[-1].value=ast.Call(func=ast.Attribute(value=ast.Attribute(value=ast.Name(id='jax',ctx=ast.Load()),attr='lax',ctx=ast.Load()),attr='stop_gradient',ctx=ast.Load()),args=[node.body[-1].value],keywords=[])
   if name=='stopped_down' and node.name=='hidden_states':
    class Down(ast.NodeTransformer):
     def visit_Subscript(self,n):
      self.generic_visit(n)
      if isinstance(n.value,ast.Name) and n.value.id=='block' and isinstance(n.slice,ast.Constant) and n.slice.value=='down':
       return ast.Call(func=ast.Attribute(value=ast.Attribute(value=ast.Name(id='jax',ctx=ast.Load()),attr='lax',ctx=ast.Load()),attr='stop_gradient',ctx=ast.Load()),args=[n],keywords=[])
      return n
    node=Down().visit(node)
 p=O/(name+'.py');p.write_text(ast.unparse(ast.fix_missing_locations(t))+'\n');return p
record={'checker_sha256':sha(P/'tests/check.py'),'model_sha256':sha(P/'model.py'),'cases':{}}
for name in ['reference','equivalent','zero_head','stopped_encoder','stopped_head','stopped_down']:
 p=P/'model.py' if name=='reference' else variant(name)
 try:
  v=c._check_model(c._load_candidate(p));record['cases'][name]={'accepted':True,'candidate_sha256':sha(p),'mlm':v['mlm_numpy_oracle_and_full_gradient_fd']}
 except AssertionError as e:record['cases'][name]={'accepted':False,'candidate_sha256':sha(p),'error':str(e)[:1000]}
 (O/'result.json').write_text(json.dumps(record,indent=2)+'\n');print(name,record['cases'][name]['accepted'],flush=True)
print(json.dumps({n:v['accepted'] for n,v in record['cases'].items()},indent=2))
