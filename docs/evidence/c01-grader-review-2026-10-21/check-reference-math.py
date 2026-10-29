"""Verify frozen reference MLM equations/selected derivatives independently."""
from pathlib import Path
import copy
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
model,_=checker._load_candidate('solution')
cfg=checker.EncoderConfig(vocab_size=32,max_length=12,width=16,layers=2,heads=4,mlp_width=24,dropout=.1)
params=model.initialize(jax.random.key(17),cfg)
ids=np.array([[2,4,9,3],[2,6,4,3]],dtype=np.int32)
labels=np.array([[2,7,9,3],[2,6,8,3]],dtype=np.int32)
mask=np.array([[0,1,0,0],[0,0,1,0]],dtype=bool)
def numpy_logits(p):
    x=checker._numpy_encoder_oracle(p,ids,cfg)
    h=p['mlm'];x=x@np.asarray(h['dense'])+np.asarray(h['dense_bias'])
    x=.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+.044715*x**3)))
    x=checker._numpy_layer_norm(x,np.asarray(h['norm_scale']),np.asarray(h['norm_bias']))
    return x@np.asarray(p['token_embed']).T+np.asarray(h['vocab_bias'])
expected=numpy_logits(params)
actual=np.asarray(model.mlm_logits(params,jnp.asarray(ids),cfg))
np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
gradient=jax.grad(lambda p:model.masked_token_loss(p,jnp.asarray(ids),jnp.asarray(labels),jnp.asarray(mask),cfg)[0])(params)
checks=[]
for path,index in [(('blocks',0,'q'),(0,0)),(('blocks',1,'v'),(0,0)),(('blocks',0,'up'),(0,0)),(('mlm','dense'),(0,0)),(('token_embed',),(9,0)),(('mlm','vocab_bias'),(7,))]:
    plus=jax.tree.map(lambda a:np.asarray(a,dtype=np.float64).copy(),params)
    minus=copy.deepcopy(plus)
    a,b,g=plus,minus,gradient
    for key in path:a=a[key];b=b[key];g=g[key]
    epsilon=1e-3;a[index]+=epsilon;b[index]-=epsilon
    fd=(checker._numpy_masked_loss(numpy_logits(plus),labels,mask)[0]-checker._numpy_masked_loss(numpy_logits(minus),labels,mask)[0])/(2*epsilon)
    actual_gradient=float(g[index])
    np.testing.assert_allclose(actual_gradient,fd,rtol=3e-3,atol=1e-5)
    checks.append({'path':list(path),'index':list(index),'jax_gradient':actual_gradient,'independent_numpy_finite_difference':fd})
record={'status':'pass','scope':'Frozen reference tied MLM dense/GELU/norm/vocabulary forward and six independent NumPy loss finite differences across both encoder blocks, tied embeddings and head. Small deterministic inference fixture; does not establish every learner implementation, training quality or target scale.','model_sha256':hashlib.sha256((PROJECT/'model.py').read_bytes()).hexdigest(),'max_logit_error':float(np.max(np.abs(actual-expected))),'selected_derivatives':checks,'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(ROOT/'reference-math-result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
