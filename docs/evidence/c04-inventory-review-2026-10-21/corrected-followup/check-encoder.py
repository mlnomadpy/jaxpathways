"""Independent NumPy temporal/attention oracle for the frozen pilot encoder."""
from pathlib import Path
import hashlib
import json
import math
import sys
import jax
import jax.numpy as jnp
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import model
from config import PILOT_MODEL

def norm(x,p):
    mean=x.mean(axis=-1,keepdims=True)
    return (x-mean)/np.sqrt(((x-mean)**2).mean(axis=-1,keepdims=True)+1e-5)*p['scale']+p['bias']

def gelu(x):
    return .5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+.044715*x**3)))

def reference(p,features,lengths):
    """Explicit index convolution; full bidirectional attention in float64."""
    x=np.array(features,dtype=np.float64)
    lengths=np.array(lengths,dtype=np.int64)
    x[np.arange(x.shape[1])[None,:]>=lengths[:,None]]=0
    for layer in p['frontend']:
        batch,width,channels=x.shape
        result=np.zeros((batch,(width+1)//2,layer['w'].shape[-1]))
        # Fixed kernel centers at 0,2,4,...; explicit one-cell zero boundaries.
        for out in range(result.shape[1]):
            for offset in range(3):
                source=2*out+offset-1
                if 0<=source<width: result[:,out]+=x[:,source]@layer['w'][offset]
        x=gelu(norm(result,layer['norm']))
        lengths=(lengths+1)//2
        x[np.arange(x.shape[1])[None,:]>=lengths[:,None]]=0
    x=x@p['input_proj']['w']+p['input_proj']['b']
    pos=np.arange(x.shape[1])[:,None]
    dimension=np.arange(x.shape[2])[None,:]
    angles=pos*10000.**(-2*(dimension//2)/x.shape[2])
    pe=np.empty_like(angles,dtype=np.float64)
    pe[:,0::2]=np.sin(angles[:,0::2]);pe[:,1::2]=np.cos(angles[:,1::2])
    x+=pe[None]
    valid=np.arange(x.shape[1])[None,:]<lengths[:,None]
    x[~valid]=0
    batch,time,width=x.shape
    heads=4
    for block in p['blocks']:
        qkv=norm(x,block['n1'])@block['qkv']['w']+block['qkv']['b']
        q,k,v=[a.reshape(batch,time,heads,width//heads).transpose(0,2,1,3) for a in np.split(qkv,3,axis=-1)]
        logits=(q@k.swapaxes(-1,-2))/math.sqrt(width//heads)
        logits=np.where(valid[:,None,None,:],logits,-1e30)
        weights=np.exp(logits-logits.max(axis=-1,keepdims=True))
        weights/=weights.sum(axis=-1,keepdims=True)
        attended=(weights@v).transpose(0,2,1,3).reshape(batch,time,width)
        x+=attended@block['attn_out']['w']+block['attn_out']['b']
        x[~valid]=0
        ff=gelu(norm(x,block['n2'])@block['ff1']['w']+block['ff1']['b'])
        x+=ff@block['ff2']['w']+block['ff2']['b']
        x[~valid]=0
    encoded=norm(x,p['final_norm']);encoded[~valid]=0
    return {'encoded':encoded,'lengths':lengths,'logits':encoded@p['ctc_head']['w']+p['ctc_head']['b'],'reconstruction':encoded@p['acoustic_decoder']['w']+p['acoustic_decoder']['b']}

params=model.init_model(jax.random.PRNGKey(841))
rng=np.random.default_rng(842)
def nonzero(path,a):
    # Exercise learned nonzero affine offsets, not only zero initialization.
    key=str(getattr(path[-1],'key',''))
    if key in ('b','bias'):return jnp.asarray(rng.normal(.15,.1,a.shape),dtype=a.dtype)
    if key=='scale':return jnp.asarray(rng.normal(1,.2,a.shape),dtype=a.dtype)
    return a
params=jax.tree_util.tree_map_with_path(nonzero,params)
numpy_params=jax.tree.map(lambda a:np.asarray(a,dtype=np.float64),params)
valid_features=rng.normal(size=(3,11,80)).astype('float32')
lengths=np.array([11,7,1],dtype='int32')
record={'scope':'Pilot inference forward only: independent explicit-index NumPy convolution, affine normalization, GELU, sinusoidal positions, full attention and heads, with nonzero learned biases. No dropout/gradient, raw-audio frontend, recovery, corpus training, target architecture or task-quality acceptance.', 'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'source').glob('*.py')},'cases':[]}
for width in (11,19,20,21):
    features=rng.normal(size=(3,width,80)).astype('float32')*13
    features[:,:11]=valid_features
    expected=reference(numpy_params,features,lengths)
    actual=model.forward(params,features,lengths)
    assert np.array_equal(np.asarray(actual['lengths']),expected['lengths'])
    max_error=0.
    for key in ('encoded','logits','reconstruction'):
        np.testing.assert_allclose(np.asarray(actual[key]),expected[key],atol=1e-5,rtol=3e-5)
        max_error=max(max_error,float(np.max(np.abs(np.asarray(actual[key])-expected[key]))))
    record['cases'].append({'padded_width':width,'batch_lengths':lengths.tolist(),'encoder_lengths':expected['lengths'].tolist(),'max_absolute_forward_error':max_error})
    # Same valid row on its own, with no second utterance or extra batch padding.
    single=model.forward(params,valid_features[:1],lengths[:1])
    np.testing.assert_allclose(np.asarray(actual['logits'][0,:3]),np.asarray(single['logits'][0,:3]),atol=1e-5,rtol=3e-5)

original=model._conv_explicit_symmetric
try:
    def shape_dependent(x,w,stride=2):
        return jax.lax.conv_general_dilated(x,w,(stride,),'SAME',dimension_numbers=('NWC','WIO','NWC'))
    model._conv_explicit_symmetric=shape_dependent
    features=np.pad(valid_features,((0,0),(0,9),(0,0)))
    expected=reference(numpy_params,features,lengths)
    broken=np.asarray(model.forward(params,features,lengths)['logits'])
    delta=float(np.max(np.abs(broken[0,:3]-expected['logits'][0,:3])))
    assert delta>.01,delta
    record['negative_control']={'mutation':'Shape-dependent SAME padding replaces fixed alignment in memory only','max_valid_logit_delta':delta,'rejected':True}
finally:
    model._conv_explicit_symmetric=original
record['status']='pass'
record['checker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(ROOT/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
