"""Connected, inspectable CPU image harness; synthetic fixture, no vision benchmark."""
from pathlib import Path
import hashlib
import json
import os
import tempfile
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import export

PREPROCESS=dict(version='image-v1',height=8,width=8,channels=1,source_channels=[1,3],
                resize='nearest floor-index',rgb_weights=[.2126,.7152,.0722],
                normalization='2 * gray_in_0_1 - 1',layout='NHWC',exif='transpose on file decode')
CLASSES=['vertical','horizontal','cross']


def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def digest_arrays(*arrays):
    h=hashlib.sha256()
    for value in arrays:
        a=np.ascontiguousarray(value);h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def make_dataset(seed=11,count=96,split='train',corruption=False):
    if count<3:raise ValueError('at least three images required')
    rng=np.random.default_rng(seed);labels=np.arange(count,dtype=np.int32)%3
    images=np.empty((count,8,8,1),np.uint8)
    for i,label in enumerate(labels):
        canvas=rng.normal(18,5,(8,8)); row,col=rng.integers(1,6,size=2)
        if label in (0,2):canvas[:,col:col+2]+=rng.uniform(170,210)
        if label in (1,2):canvas[row:row+2,:]+=rng.uniform(170,210)
        images[i,:,:,0]=np.clip(canvas,0,255).astype(np.uint8)
    if corruption:
        # Apply a separate random stream after clean generation: pairs remain aligned.
        altered=images.astype(np.float32);altered[:,:,2:6,:]=18
        altered+=np.random.default_rng(seed+5000).normal(0,28,altered.shape)
        images=np.clip(altered,0,255).astype(np.uint8)
    return dict(images=images,labels=labels,ids=[f'{split}-{seed}-{i:04d}' for i in range(count)],
                groups=[f'{split}-synthetic-{seed}-{i:04d}' for i in range(count)],
                provenance={'kind':'synthetic teaching fixture','seed':seed,'split':split,'corruption':corruption,'license':'original generated fixture; no external photographs'})


def dataset_hash(data):
    return digest_json(dict(arrays=digest_arrays(data['images'],data['labels']),ids=data['ids'],groups=data['groups'],provenance=data.get('provenance',{})))


def validate_splits(training,heldout):
    for data in (training,heldout):
        n=len(data['images'])
        if np.asarray(data['labels']).shape!=(n,) or len(data['ids'])!=n or len(data['groups'])!=n:raise ValueError('dataset shape contract')
        if len(set(data['ids']))!=n:raise ValueError('duplicate image IDs')
        if not np.issubdtype(np.asarray(data['labels']).dtype,np.integer) or np.any((data['labels']<0)|(data['labels']>=3)):raise ValueError('class labels required')
    if set(training['ids'])&set(heldout['ids']) or set(training['groups'])&set(heldout['groups']):raise ValueError('split leakage')
    train_pixels={digest_arrays(image) for image in training['images']}
    if train_pixels & {digest_arrays(image) for image in heldout['images']}:raise ValueError('duplicate image content across splits')


def preprocess(images,layout='NHWC',input_range='uint8'):
    a=np.asarray(images)
    if a.ndim!=4 or not len(a):raise ValueError('nonempty rank-four image batch required')
    if layout=='NCHW':a=np.transpose(a,(0,2,3,1))
    elif layout!='NHWC':raise ValueError('unsupported layout')
    if a.shape[-1] not in (1,3) or min(a.shape[1:3])<1:raise ValueError('grayscale/RGB channels and positive size required')
    if input_range=='uint8':
        if a.dtype!=np.uint8:raise ValueError('uint8 input contract requires uint8 dtype')
        a=a.astype(np.float32)/255.
    elif input_range=='unit_float':
        if not np.issubdtype(a.dtype,np.floating) or not np.isfinite(a).all() or np.any((a<0)|(a>1)):raise ValueError('finite unit-range floating input required')
        a=a.astype(np.float32)
    else:raise ValueError('unsupported input range')
    rows=(np.arange(8)*a.shape[1]//8);cols=(np.arange(8)*a.shape[2]//8)
    a=a[:,rows][:,:,cols]
    if a.shape[-1]==3:a=np.sum(a*np.array(PREPROCESS['rgb_weights'],np.float32),axis=-1,keepdims=True)
    return np.ascontiguousarray(2*a-1,dtype=np.float32)


def read_image(path):
    """External PNG/JPEG/etc through Pillow, explicit EXIF transpose and RGB conversion."""
    from PIL import Image,ImageOps
    with Image.open(path) as image:
        if image.width*image.height>16_000_000:raise ValueError('image exceeds teaching decoder pixel limit')
        if getattr(image,'n_frames',1)!=1:raise ValueError('animated images are not supported')
        return np.asarray(ImageOps.exif_transpose(image).convert('RGB'),dtype=np.uint8)[None]


def load_external_manifest(path):
    """Load user-owned labeled images; relative paths stay inside manifest directory.

    JSON: split, source, license, items [{path,id,group,label}]. This loads supplied
    files only and never downloads data. Split related subjects/sources before use.
    """
    path=Path(path);manifest=json.loads(path.read_text());root=path.parent.resolve()
    if not all(manifest.get(k) for k in ('split','source','license','items')):raise ValueError('data provenance and items required')
    images=[];labels=[];ids=[];groups=[];file_hashes=[]
    for item in manifest['items']:
        image_path=(root/item['path']).resolve()
        if not image_path.is_relative_to(root):raise ValueError('image outside manifest directory')
        if item['label'] not in CLASSES:raise ValueError('unknown image label')
        if not item.get('id') or not item.get('group'):raise ValueError('image and source-group IDs required')
        raw=read_image(image_path)
        rows=np.arange(8)*raw.shape[1]//8;cols=np.arange(8)*raw.shape[2]//8
        images.append(raw[0][rows][:,cols]);labels.append(CLASSES.index(item['label']))
        ids.append(item['id']);groups.append(item['group']);file_hashes.append(hashlib.sha256(image_path.read_bytes()).hexdigest())
    if len(set(ids))!=len(ids):raise ValueError('duplicate image IDs')
    return dict(images=np.stack(images),labels=np.asarray(labels,np.int32),ids=ids,groups=groups,
                provenance=dict(kind='user-supplied image files',split=manifest['split'],source=manifest['source'],license=manifest['license'],file_hashes=file_hashes))


def init_params(seed=0):
    a,b=jax.random.split(jax.random.PRNGKey(seed))
    return dict(conv=jax.random.normal(a,(3,3,1,6))*.3,conv_bias=jnp.zeros(6),
                head=jax.random.normal(b,(6,3))*.2,head_bias=jnp.zeros(3))


def forward(params,x,compute='float32',return_hidden=False):
    dtype={'float32':jnp.float32,'float16':jnp.float16,'bfloat16':jnp.bfloat16}[compute]
    conv=jax.lax.conv_general_dilated(jnp.asarray(x,dtype),params['conv'].astype(dtype),(1,1),'VALID',dimension_numbers=('NHWC','HWIO','NHWC'),preferred_element_type=jnp.float32)
    hidden=jax.nn.relu(conv+params['conv_bias'])
    pooled=hidden.mean(axis=(1,2))
    logits=jax.lax.dot_general(pooled.astype(dtype),params['head'].astype(dtype),(((1,),(0,)),((),())),preferred_element_type=jnp.float32)+params['head_bias']
    return (logits,pooled) if return_hidden else logits


def objective(params,x,labels):
    labels=jnp.asarray(labels)
    if labels.ndim!=1 or labels.shape[0]!=x.shape[0] or not jnp.issubdtype(labels.dtype,jnp.integer):raise ValueError('one integer label per image required')
    logits=forward(params,x)
    return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits),labels[:,None],axis=1))


def config(batch_size=12,learning_rate=.08,momentum=.9):
    if not isinstance(batch_size,int) or batch_size<1 or not 0<learning_rate<1 or not 0<=momentum<1:raise ValueError('invalid optimizer/input configuration')
    return dict(batch_size=batch_size,learning_rate=learning_rate,momentum=momentum)


def start(data,seed=0,settings=None):
    settings=config(**(settings or {}));params=init_params(seed)
    key,order_key=jax.random.split(jax.random.PRNGKey(seed+100))
    order=np.asarray(jax.random.permutation(order_key,len(data['images'])))
    return dict(params=params,velocity=jax.tree.map(jnp.zeros_like,params),key=key,order=order,
                position=0,epoch=0,step=0,seed=seed,data_hash=dataset_hash(data),settings=settings)


@jax.jit
def _update(params,velocity,key,x,y,learning_rate,momentum):
    key,noise_key,flip_key=jax.random.split(key,3)
    flips=jax.random.bernoulli(flip_key,.5,(len(x),1,1,1))
    augmented=jnp.where(flips,x[:,:,::-1,:],x)
    augmented=jnp.clip(augmented+.035*jax.random.normal(noise_key,x.shape),-1,1)
    loss,grad=jax.value_and_grad(objective)(params,augmented,y)
    velocity=jax.tree.map(lambda v,g:momentum*v+g,velocity,grad)
    params=jax.tree.map(lambda p,v:p-learning_rate*v,params,velocity)
    return params,velocity,key,loss,augmented


def advance(state,data,updates=1):
    if dataset_hash(data)!=state['data_hash']:raise ValueError('dataset changed since checkpoint')
    if not isinstance(updates,int) or updates<1:raise ValueError('positive update count required')
    s=dict(state);records=[];n=len(data['images']);batch=s['settings']['batch_size']
    if batch>n:raise ValueError('batch exceeds dataset')
    all_x=preprocess(data['images'])
    for _ in range(updates):
        if s['position']+batch>n:
            s['key'],k=jax.random.split(s['key']);s['order']=np.asarray(jax.random.permutation(k,n));s['position']=0;s['epoch']+=1
        indices=s['order'][s['position']:s['position']+batch]
        p,v,k,loss,aug=_update(s['params'],s['velocity'],s['key'],jnp.asarray(all_x[indices]),jnp.asarray(data['labels'][indices]),s['settings']['learning_rate'],s['settings']['momentum'])
        jax.block_until_ready((p,v,k,loss,aug));s.update(params=p,velocity=v,key=k,position=s['position']+batch,step=s['step']+1)
        records.append(dict(step=s['step'],loss=float(loss),ids=[data['ids'][i] for i in indices],augmented_hash=digest_arrays(aug)))
    return s,records


def evaluate(params,data,batch_size=17,compute='float32'):
    if batch_size<1 or not len(data['images']):raise ValueError('positive evaluation batch and nonempty data required')
    x=preprocess(data['images']);y=np.asarray(data['labels'])
    if y.shape!=(len(x),) or not np.issubdtype(y.dtype,np.integer) or np.any((y<0)|(y>=3)):raise ValueError('valid integer vector labels required')
    logits=np.concatenate([np.asarray(forward(params,jnp.asarray(x[i:i+batch_size]),compute)) for i in range(0,len(x),batch_size)])
    return metrics(logits,y,data['ids'])


def metrics(logits,labels,ids):
    logits=np.asarray(logits,dtype=np.float64);labels=np.asarray(labels)
    if logits.ndim!=2 or logits.shape[1]!=3 or len(logits)==0 or not np.isfinite(logits).all() or labels.shape!=(len(logits),) or not np.issubdtype(labels.dtype,np.integer) or np.any((labels<0)|(labels>=3)) or len(ids)!=len(logits):raise ValueError('invalid evaluation arrays')
    shifted=logits-logits.max(1,keepdims=True)
    losses=np.log(np.exp(shifted).sum(1))-shifted[np.arange(len(labels)),labels]
    predicted=np.argmax(logits,1);confusion=np.zeros((3,3),int)
    np.add.at(confusion,(labels,predicted),1)
    return dict(count=len(labels),loss=float(losses.mean()),loss_sum=float(losses.sum()),correct=int((predicted==labels).sum()),
                accuracy=float((predicted==labels).mean()),confusion=confusion.tolist(),error_ids=[ids[i] for i in np.flatnonzero(predicted!=labels)],logits=logits,predictions=predicted)


def save_checkpoint(path,state):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    arrays={**{'p_'+k:np.asarray(v) for k,v in state['params'].items()},**{'v_'+k:np.asarray(v) for k,v in state['velocity'].items()},'key':np.asarray(state['key']),'order':np.asarray(state['order'])}
    meta={k:state[k] for k in ('step','position','epoch','seed','data_hash','settings')};meta.update(version=1,jax_version=jax.__version__,numpy_version=np.__version__,preprocess=PREPROCESS,arrays_hash=digest_arrays(*[arrays[k] for k in sorted(arrays)]))
    meta['metadata_hash']=digest_json(meta)
    fd,name=tempfile.mkstemp(dir=path.parent,suffix='.npz')
    try:
        with os.fdopen(fd,'wb') as out:
            np.savez(out,**arrays,metadata=np.array(json.dumps(meta,sort_keys=True)));out.flush();os.fsync(out.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def restore_checkpoint(path,data,settings=None):
    with np.load(path,allow_pickle=False) as bundle:
        arrays={k:bundle[k].copy() for k in bundle.files if k!='metadata'};meta=json.loads(str(bundle['metadata']))
    metadata_hash=meta.pop('metadata_hash',None)
    if metadata_hash!=digest_json(meta):raise ValueError('checkpoint metadata integrity mismatch')
    if meta['jax_version']!=jax.__version__ or meta['numpy_version']!=np.__version__:raise ValueError('checkpoint runtime version mismatch')
    if meta['version']!=1 or meta['preprocess']!=PREPROCESS or meta['data_hash']!=dataset_hash(data) or meta['settings']!=config(**(settings or {})):raise ValueError('checkpoint data/preprocess/config mismatch')
    if meta['arrays_hash']!=digest_arrays(*[arrays[k] for k in sorted(arrays)]):raise ValueError('checkpoint integrity mismatch')
    if arrays['key'].shape!=(2,) or arrays['key'].dtype!=np.uint32:raise ValueError('invalid random key')
    if sorted(arrays['order'].tolist())!=list(range(len(data['images']))) or not 0<=meta['position']<=len(data['images']) or meta['step']<0 or meta['epoch']<0:raise ValueError('invalid sampler state')
    params={k: jnp.asarray(arrays['p_'+k]) for k in init_params()};velocity={k:jnp.asarray(arrays['v_'+k]) for k in params}
    for k,template in init_params().items():
        if params[k].shape!=template.shape or velocity[k].shape!=template.shape or not np.isfinite(params[k]).all() or not np.isfinite(velocity[k]).all():raise ValueError('invalid parameter or optimizer state')
    return dict(params=params,velocity=velocity,key=jnp.asarray(arrays['key']),order=arrays['order'],**{k:meta[k] for k in ('step','position','epoch','seed','data_hash','settings')})


def patches(x):
    return np.stack([x[:,i:i+6,j:j+6,:] for i in range(3) for j in range(3)],axis=-2).reshape(len(x),6,6,9)


def quantize(values,scale):
    return np.clip(np.rint(np.asarray(values)/scale),-127,127).astype(np.int8)


def calibrate(params,data,percentile=99.):
    if not 0<percentile<=100:raise ValueError('invalid calibration percentile')
    x=preprocess(data['images']);_,hidden=forward(params,jnp.asarray(x),return_hidden=True)
    input_scale=max(float(np.percentile(np.abs(x),percentile))/127,1e-8)
    hidden_scale=max(float(np.percentile(np.abs(hidden),percentile))/127,1e-8)
    conv=np.asarray(params['conv']).reshape(9,6);head=np.asarray(params['head'])
    cs=np.maximum(np.max(np.abs(conv),axis=0)/127,1e-8);hs=np.maximum(np.max(np.abs(head),axis=0)/127,1e-8)
    return dict(conv=quantize(conv,cs),conv_scale=cs,conv_bias=np.asarray(params['conv_bias']),head=quantize(head,hs),head_scale=hs,head_bias=np.asarray(params['head_bias']),input_scale=np.array(input_scale,np.float32),hidden_scale=np.array(hidden_scale,np.float32),calibration_hash=np.array(dataset_hash(data)),percentile=np.array(percentile))


def integer_forward(quantized,x,accumulator='int32'):
    if accumulator!='int32':raise ValueError('only int32 accumulation is supported')
    q=quantized;raw=np.asarray(x);ix=quantize(raw,q['input_scale'])
    acc=patches(ix).astype(np.int32) @ q['conv'].astype(np.int32)
    hidden=np.maximum(acc.astype(np.float32)*(q['input_scale']*q['conv_scale'])+q['conv_bias'],0).mean(axis=(1,2))
    ih=quantize(hidden,q['hidden_scale'])
    logits=(ih.astype(np.int32)@q['head'].astype(np.int32)).astype(np.float32)*(q['hidden_scale']*q['head_scale'])+q['head_bias']
    return logits,dict(input_clipped=int((np.abs(raw)>127*q['input_scale']).sum()),input_count=raw.size,hidden_clipped=int((np.abs(hidden)>127*q['hidden_scale']).sum()),hidden_count=hidden.size)


def export_model(folder,state,quantized):
    folder=Path(folder)
    if folder.exists():raise ValueError('export destination must be new')
    folder.mkdir(parents=True)
    for batch in (1,4):
        artifact=export.export(jax.jit(lambda x:forward(state['params'],x)))(jax.ShapeDtypeStruct((batch,8,8,1),jnp.float32))
        (folder/f'model-{batch}.jax').write_bytes(artifact.serialize())
    np.savez(folder/'integer.npz',**quantized)
    np.savez(folder/'float-params.npz',**{k:np.asarray(v) for k,v in state['params'].items()})
    names=['model-1.jax','model-4.jax','integer.npz','float-params.npz']
    manifest=dict(version=1,preprocess=PREPROCESS,classes=CLASSES,batches=[1,4],training_data_hash=state['data_hash'],training_settings=state['settings'],calibration_data_hash=str(quantized['calibration_hash']),calibration_percentile=float(quantized['percentile']),completed_steps=state['step'],jax_version=jax.__version__,validated_platform='cpu',integer_accumulator='int32',files={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in names})
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    return manifest


def load_model(folder,expected_preprocess=PREPROCESS):
    folder=Path(folder);meta=json.loads((folder/'manifest.json').read_text())
    if meta['version']!=1 or meta['preprocess']!=expected_preprocess or meta['classes']!=CLASSES or meta['batches']!=[1,4] or meta['integer_accumulator']!='int32':raise ValueError('incompatible artifact contract')
    names=['model-1.jax','model-4.jax','integer.npz','float-params.npz']
    if set(meta['files'])!=set(names):raise ValueError('unexpected artifact files')
    for name in names:
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=meta['files'][name]:raise ValueError('artifact integrity mismatch')
    models={b:export.deserialize((folder/f'model-{b}.jax').read_bytes()) for b in (1,4)}
    with np.load(folder/'integer.npz',allow_pickle=False) as f:q={k:f[k].copy() for k in f.files}
    return dict(models=models,quantized=q,manifest=meta)


def infer(runtime,images,layout='NHWC',input_range='uint8',precision='float32'):
    x=preprocess(images,layout,input_range)
    if precision=='int8':return integer_forward(runtime['quantized'],x)[0]
    if precision!='float32':raise ValueError('runtime supports float32 exported graph or int8 reference')
    outputs=[];position=0
    while position<len(x):
        batch=4 if len(x)-position>=4 else 1
        result=runtime['models'][batch].call(jnp.asarray(x[position:position+batch]));result.block_until_ready()
        outputs.append(np.asarray(result));position+=batch
    return np.concatenate(outputs)


def benchmark(runtime,images,repeats=12,precision='float32'):
    if repeats<3:raise ValueError('at least three timing repetitions required')
    start=time.perf_counter();infer(runtime,images,precision=precision);warmup=time.perf_counter()-start
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter();infer(runtime,images,precision=precision);samples.append(time.perf_counter()-start)
    return dict(precision=precision,backend=jax.default_backend(),device=str(jax.devices()[0]),batch=len(images),repeats=repeats,warmup_s=warmup,samples_s=samples,median_ms=1000*float(np.median(samples)),p95_ms=1000*float(np.percentile(samples,95)),images_per_s=len(images)*repeats/sum(samples),scope='host preprocessing + synchronous inference; excludes file decoding')


def select_runtime(candidate_folder,canary_images,expected_logits):
    """Validate a candidate before the caller swaps its active runtime reference."""
    candidate=load_model(candidate_folder)
    actual=infer(candidate,canary_images)
    if np.shape(actual)!=np.shape(expected_logits) or not np.allclose(actual,expected_logits,rtol=2e-5,atol=2e-5):
        raise ValueError('candidate failed declared canary parity')
    return candidate
