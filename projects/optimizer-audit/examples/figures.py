"""Execute the reference experiments and save the exact plotted numerical evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import platform
import numpy as np
import jax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True)
source=ROOT/'solution/optimization.py';spec=importlib.util.spec_from_file_location('optimization',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checked=subprocess.run([sys.executable,str(ROOT/'tests/check.py'),'--implementation','solution','--stage','all'],check=True,capture_output=True,text=True,timeout=60)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(fig,name):
    fig.tight_layout();fig.savefig(OUT/f'{name}.png',dpi=140);fig.savefig(OUT/f'{name}.svg');plt.close(fig)
def serial(value):
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value

# Conditioning is a coordinate change. Same observations, zero initialization and update budget.
x,y,truth=m.fixture(3);hx,hy,_=m.fixture(4);scales=m.fit_scales(x);z=m.apply_scales(x,scales);hz=m.apply_scales(hx,scales)
raw_h=m.geometry(np.zeros(2,np.float32),x,y)['hessian'];scaled_h=m.geometry(np.zeros(2,np.float32),z,y)['hessian']
raw_eig=np.linalg.eigvalsh(raw_h.astype(np.float64));scaled_eig=np.linalg.eigvalsh(scaled_h.astype(np.float64));plan=np.tile(np.arange(len(x),dtype=np.int32),(80,1))
raw_lr=.9/raw_eig[-1];scaled_lr=.9/scaled_eig[-1]
_,raw=m.run(np.zeros(2,np.float32),x,y,hx,hy,plan,m.rates(raw_lr,80))
_,scaled=m.run(np.zeros(2,np.float32),z,y,hz,hy,plan,m.rates(scaled_lr,80))
opt=m.ridge_solution(x,y);xx,yy=np.meshgrid(np.linspace(-.1,1.75,100),np.linspace(-.07,.025,100));grid=np.stack([xx.ravel(),yy.ravel()],axis=1)
loss=(.5*np.mean((x@grid.T-y[:,None])**2,axis=0)).reshape(xx.shape)
fig,axes=plt.subplots(2,2,figsize=(12,8))
ax=axes[0,0];levels=np.geomspace(float(loss.min())*1.01,float(loss.max()),12);ax.contour(xx,yy,loss,levels=levels,colors='#aebdcb',linewidths=.8)
for trace,name in [(raw['weights'],'raw coordinates'),(scaled['weights']/scales,'RMS-scaled coordinates')]:
    path=np.vstack([np.zeros(2),trace]);ax.plot(path[:,0],path[:,1],label=name,linewidth=1.6)
ax.scatter(*opt,marker='*',s=80,color='black',label='least-squares optimum');ax.set(xlabel='first coefficient (original units)',ylabel='second coefficient (original units)',title='A. Same objective, different update paths');ax.legend(fontsize=8)
ax=axes[0,1];positions=np.arange(2);ax.bar(positions-.17,raw_eig,.34,label='raw');ax.bar(positions+.17,scaled_eig,.34,label='RMS-scaled');ax.set_xticks(positions,['smallest','largest']);ax.set(yscale='log',ylabel='Hessian eigenvalue',title='B. Scaling changes curvature');ax.legend()
ax=axes[1,0];ax.semilogy(np.arange(1,81),raw['train_mse'],label=f'raw, rate {raw_lr:.4g}');ax.semilogy(np.arange(1,81),scaled['train_mse'],label=f'scaled, rate {scaled_lr:.3g}');ax.axhline(np.mean((x@opt-y)**2),color='black',ls=':',label='least-squares MSE');ax.set(xlabel='completed update',ylabel='full-training MSE (log scale)',title='C. Equal updates; rate uses each largest eigenvalue');ax.legend(fontsize=8)
x2=np.array([[1.,0.],[0.,4.]],np.float32);y2=np.zeros(2,np.float32);ids=np.tile(np.arange(2,dtype=np.int32),(25,1));stability={}
ax=axes[1,1]
for ratio in [.8,2.2]:
    _,trace=m.run(np.array([0.,1.],np.float32),x2,y2,x2,y2,ids,m.rates(ratio/8,25));stability[str(ratio)]=trace
    ax.semilogy(np.arange(1,26),np.abs(trace['weights'][:,1]),label=f'rate × largest curvature = {ratio}')
ax.set(xlabel='completed update',ylabel='absolute coefficient error (log scale)',title='D. Contracting versus finite but growing error');ax.legend(fontsize=8)
save(fig,'conditioning')

# Exact variance, then noisy updates with shared sample plan and fixed validation set.
sx=np.array([[-2.,1.],[0.,-1.],[3.,2.]],np.float32);sy=np.array([1.,-2.,.5],np.float32);sw=np.array([.25,-.5],np.float32)
audits=[m.noise_audit(sw,sx,sy,b,.2) for b in [1,2,3]]
x,y,_=m.fixture(17);vx,vy,_=m.fixture(19);sc=m.fit_scales(x);x=m.apply_scales(x,sc);vx=m.apply_scales(vx,sc)
ids=m.batch_plan(71,len(x),150,12)
configs=[('SGD','gd',.12,None,None),('momentum','momentum',.04,None,None),('Adam','adam',.07,None,None),('clipped Adam','adam',.07,.2,None),('clipped Adam + decay','adam',.07,.2,70)]
runs={};schedule={}
for label,method,rate,clip,decay in configs:
    lrs=m.rates(rate,150,decay_at=decay,factor=.1);schedule[label]=lrs
    _,runs[label]=m.run(np.zeros(2,np.float32),x,y,vx,vy,ids,lrs,method,clip_norm=clip)
fig,axes=plt.subplots(2,2,figsize=(12,8))
ax=axes[0,0];ax.bar([1,2,3],[np.trace(a['covariance']) for a in audits],color='#286a91');ax.set(xticks=[1,2,3],xlabel='batch size, iid draws with replacement',ylabel='trace of gradient covariance',title='A. Exact enumeration, not estimated error bars')
ax=axes[0,1]
for label,trace in runs.items():ax.semilogy(np.arange(1,151),trace['held_mse'],label=label)
ax.set(xlabel='completed update',ylabel='fixed validation MSE (log scale)',title='B. Shared examples; declared method-specific rates');ax.legend(fontsize=8)
trace=runs['clipped Adam + decay'];ax=axes[1,0];ax.plot(np.arange(1,151),trace['raw_norm'],label='raw gradient');ax.plot(np.arange(1,151),trace['clipped_norm'],label='after clipping');ax.axhline(.2,color='black',ls=':',label='threshold 0.2');ax.set(xlabel='completed update',ylabel='Euclidean gradient norm',title='C. Clipping acts before optimizer state');ax.legend(fontsize=8)
ax=axes[1,1];ax.plot(np.arange(1,151),trace['update_norm'],label='actual Adam parameter displacement');ax.plot(np.arange(1,151),schedule['clipped Adam + decay']*.2,label='rate × gradient threshold',ls='--');ax.axvline(71,color='gray',ls=':',label='first decayed update = 71');ax.set(xlabel='completed update',ylabel='Euclidean parameter update norm',title='D. A gradient bound is not an Adam update bound');ax.legend(fontsize=8)
save(fig,'noise-and-optimizers')

# Strong correlation makes coefficient recovery sensitive. Validation selects lambda; test once.
x,y,truth=m.fixture(31,n=40,correlation=.99999,feature_ratio=1.,noise=.15)
vx,vy,_=m.fixture(32,n=96,correlation=.99999,feature_ratio=1.,noise=.15)
tx,ty,_=m.fixture(33,n=96,correlation=.99999,feature_ratio=1.,noise=.15)
u,sv,vt=np.linalg.svd(x.astype(np.float64),full_matrices=False);perturbed=(y+.01*u[:,-1]).astype(np.float32)
penalties=[0.,1e-6,1e-4,.01,.1,1.];ridge=[]
for penalty in penalties:
    w=m.ridge_solution(x,y,penalty);changed=m.ridge_solution(x,perturbed,penalty)
    ridge.append({'penalty':penalty,'weights':w,'training_mse':np.mean((x@w-y)**2),'validation_mse':np.mean((vx@w-vy)**2),'coefficient_change':np.linalg.norm(changed-w),'prediction_change_rms':np.sqrt(np.mean((x@(changed-w))**2))})
selected=min(range(len(ridge)),key=lambda i:ridge[i]['validation_mse']);test_mse=float(np.mean((tx@ridge[selected]['weights']-ty)**2))
fig,axes=plt.subplots(1,3,figsize=(14,4.3));positions=np.arange(len(penalties));labels=['0','1e-6','1e-4','0.01','0.1','1']
ax=axes[0];ax.plot(positions,[r['training_mse'] for r in ridge],marker='o',label='training');ax.plot(positions,[r['validation_mse'] for r in ridge],marker='o',label='validation');ax.scatter(selected,ridge[selected]['validation_mse'],marker='*',s=110,color='black',label='validation choice');ax.set(ylabel='unpenalized MSE',title='A. Select penalty on validation, then test once');ax.legend(fontsize=8)
ax=axes[1]
for col in range(2):ax.plot(positions,[r['weights'][col] for r in ridge],marker='o',label=f'coefficient {col+1}');ax.axhline(truth[col],ls=':',color=f'C{col}',alpha=.7)
ax.set(ylabel='coefficient (original feature units)',title='B. Stable prediction does not identify each weight');ax.legend(fontsize=8)
ax=axes[2];ax.semilogy(positions,[r['coefficient_change'] for r in ridge],marker='o',label='coefficient-change norm');ax.semilogy(positions,[r['prediction_change_rms'] for r in ridge],marker='o',label='training-prediction change RMS');ax.set(ylabel='response to target perturbation (log scale)',title='C. Same target change, different sensitivities');ax.legend(fontsize=8)
for ax in axes:ax.set_xticks(positions,labels);ax.set_xlabel('ridge penalty (categorical spacing)')
save(fig,'regularization')
receipt={'environment':{'python':platform.python_version(),'jax':jax.__version__,'numpy':np.__version__,'backend':jax.default_backend()},'source_sha256':digest(source),'checker_sha256':digest(ROOT/'tests/check.py'),'figure_source_sha256':digest(__file__),'checks_stdout':checked.stdout,
         'conditioning':{'raw_eigenvalues':raw_eig,'scaled_eigenvalues':scaled_eig,'raw_condition':raw_eig[-1]/raw_eig[0],'scaled_condition':scaled_eig[-1]/scaled_eig[0],'raw_rate':raw_lr,'scaled_rate':scaled_lr,'raw':raw,'scaled':scaled,'stability':stability},
         'noise':{'batch_sizes':[1,2,3],'covariance_traces':[np.trace(a['covariance']) for a in audits],'mean_gradient':audits[0]['mean']},
         'optimizers':runs,'schedules':schedule,'regularization':{'rows':ridge,'selected_penalty':penalties[selected],'test_mse_once':test_mse,'truth':truth,'target_perturbation_norm':float(np.linalg.norm(perturbed-y))},
         'scope':'Original synthetic CPU teaching experiments. Equal updates/examples are not equal runtime; no universal optimizer ranking, external-data or accelerator claims.'}
(OUT/'evidence.json').write_text(json.dumps(serial(receipt),indent=2,allow_nan=False)+'\n')
print(json.dumps({'raw_condition':receipt['conditioning']['raw_condition'],'scaled_condition':receipt['conditioning']['scaled_condition'],'raw_final_train_mse':float(raw['train_mse'][-1]),'scaled_final_train_mse':float(scaled['train_mse'][-1]),'noise_traces':receipt['noise']['covariance_traces'],'optimizer_final_validation':{k:float(v['held_mse'][-1]) for k,v in runs.items()},'ridge':serial(receipt['regularization'])},indent=2))
