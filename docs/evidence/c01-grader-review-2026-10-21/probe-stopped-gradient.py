"""Check whether the frozen C01 grader detects a disconnected encoder."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys
import jax
import jax.numpy as jnp
import numpy as np

ROOT=Path(__file__).resolve().parent
PROJECT=Path('/private/tmp/jaxpathways-reader-review-fix/projects/embedding-model-300m')
sys.path.insert(0,str(PROJECT))
spec=importlib.util.spec_from_file_location('frozen_checker',PROJECT/'tests/check.py')
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
tree=ast.parse((PROJECT/'model.py').read_text())
matches=0
for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name=='hidden_states':
        assert isinstance(node.body[-1],ast.Return)
        node.body[-1].value=ast.Call(func=ast.Attribute(value=ast.Attribute(value=ast.Name(id='jax',ctx=ast.Load()),attr='lax',ctx=ast.Load()),attr='stop_gradient',ctx=ast.Load()),args=[node.body[-1].value],keywords=[])
        matches+=1
assert matches==1
candidate=ROOT/'stopped_encoder_gradient.py'
candidate.write_text(ast.unparse(ast.fix_missing_locations(tree))+'\n')
module,_=checker._load_candidate(candidate)
checks=checker._check_model((module,candidate))
cfg=checker.EncoderConfig(vocab_size=32,max_length=12,width=16,layers=2,heads=4,mlp_width=24,dropout=.1)
params=module.initialize(jax.random.key(17),cfg)
ids=jnp.asarray([[2,4,9,3],[2,6,4,3]],dtype=jnp.int32)
labels=jnp.asarray([[2,7,9,3],[2,6,8,3]],dtype=jnp.int32)
mask=jnp.asarray([[0,1,0,0],[0,0,1,0]],dtype=bool)
def loss(m,p):return m.masked_token_loss(p,ids,labels,mask,cfg)[0]
reference,_=checker._load_candidate('solution')
gradient=jax.grad(lambda p:loss(module,p))(params)
reference_gradient=jax.grad(lambda p:loss(reference,p))(params)
def norm(tree):return float(np.sqrt(sum(float(jnp.sum(a*a)) for a in jax.tree.leaves(tree))))
record={'scope':'Frozen model sub-check only; forward-identical owned stop-gradient mutation. No full workflow verdict or historical training invalidity implied.', 'checker_sha256':hashlib.sha256((PROJECT/'tests/check.py').read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),'candidate_model_check_passed':True,'checks':checks,'candidate_encoder_block_gradient_norm':norm(gradient['blocks']),'reference_encoder_block_gradient_norm':norm(reference_gradient['blocks']),'forward_logit_max_delta':float(jnp.max(jnp.abs(module.mlm_logits(params,ids,cfg)-reference.mlm_logits(params,ids,cfg))))}
assert record['candidate_encoder_block_gradient_norm']==0
assert record['reference_encoder_block_gradient_norm']>0
(ROOT/'stopped-gradient-result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
