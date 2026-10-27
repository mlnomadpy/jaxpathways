"""Owned negative control for the frozen C01 model acceptance boundary."""
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
spec=importlib.util.spec_from_file_location('c01_frozen_checker',PROJECT/'tests/check.py')
checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
source=(PROJECT/'model.py').read_text()
tree=ast.parse(source)
matches=0
for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name=='mlm_logits':
        node.body=ast.parse('return jnp.zeros((*jnp.asarray(input_ids).shape, _config(config).vocab_size), dtype=jnp.float32)').body
        matches+=1
assert matches==1
candidate=ROOT/'zero_mlm_logits.py'
candidate.write_text(ast.unparse(tree)+'\n')
record={'scope':'Frozen C01 model sub-check only; no claim yet about complete stage-all acceptance or historical run invalidity. Own candidate mutation changes only the MLM logits body to ignore all parameters/inputs.', 'checker_sha256':hashlib.sha256((PROJECT/'tests/check.py').read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256(source.encode()).hexdigest(),'candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest()}
for name,selection in [('reference','solution'),('constant_mlm',candidate)]:
    loaded=checker._load_candidate(selection)
    try:
        result=checker._check_model(loaded)
        record[name]={'model_check_passed':True,'checks':result}
    except Exception as error:
        record[name]={'model_check_passed':False,'error':repr(error)}
    print(name,record[name]['model_check_passed'],flush=True)

bad,_=checker._load_candidate(candidate)
cfg=checker.EncoderConfig(vocab_size=32,max_length=12,width=16,layers=2,heads=4,mlp_width=24,dropout=.1)
params=bad.initialize(jax.random.key(17),cfg)
ids=jnp.asarray([[2,4,9,3],[2,6,4,3]],dtype=jnp.int32)
labels=jnp.asarray([[2,7,9,3],[2,6,8,3]],dtype=jnp.int32)
mask=jnp.asarray([[0,1,0,0],[0,0,1,0]],dtype=bool)
gradient=jax.grad(lambda p:bad.masked_token_loss(p,ids,labels,mask,cfg)[0])(params)
record['constant_mlm']['actual_gradient_norm']=float(np.sqrt(sum(float(jnp.sum(x*x)) for x in jax.tree.leaves(gradient))))
hidden=checker._numpy_encoder_oracle(params,np.asarray(ids),cfg)
head=params['mlm'];z=hidden@np.asarray(head['dense'])+np.asarray(head['dense_bias'])
z=.5*z*(1+np.tanh(np.sqrt(2/np.pi)*(z+.044715*z**3)))
z=checker._numpy_layer_norm(z,np.asarray(head['norm_scale']),np.asarray(head['norm_bias']))
expected=z@np.asarray(params['token_embed']).T+np.asarray(head['vocab_bias'])
record['constant_mlm']['max_logit_error_vs_independent_tied_head']=float(np.max(np.abs(np.asarray(bad.mlm_logits(params,ids,cfg))-expected)))
(ROOT/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
