"""Run the connected reference and save measured figures plus a provenance receipt."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'solution'))
import model as m
train=m.fixture(1);held=m.fixture(7,per_class=5);shifted=m.fixture(9,per_class=5,shift=1)
state,losses=m.train(train);clean=m.evaluate(state['params'],held);shift=m.evaluate(state['params'],shifted)
labels=held['labels'];similarity=np.array([[np.mean(clean['scores'][np.ix_(labels==i,labels==j)]) for j in range(4)] for i in range(4)])
confusion=np.zeros((4,4),int)
for a,b in zip(shifted['labels'],shift['image_predictions']):confusion[a,b]+=1
names=['vertical thin','vertical thick','horizontal thin','horizontal thick']
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','svg.fonttype':'none'})
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
axes[0,0].plot(np.arange(1,len(losses)+1),losses);axes[0,0].set(xlabel='completed update',ylabel='pre-update batch contrastive loss',title='A. Actual training batches')
axes[0,1].imshow(held['images'][0],cmap='gray',vmin=0,vmax=1);axes[0,1].set(title='B. Held-out vertical thin image',xlabel='pixel column',ylabel='pixel row')
for ax,values,title,cmap,limits in [(axes[1,0],similarity,'C. Clean class-average similarity','coolwarm',(-1,1)),(axes[1,1],confusion,'D. Shifted image-to-text failures','Purples',(0,5))]:
    im=ax.imshow(values,cmap=cmap,vmin=limits[0],vmax=limits[1]);fig.colorbar(im,ax=ax,shrink=.75)
    ax.set_xticks(range(4),names,rotation=30,ha='right');ax.set_yticks(range(4),names)
    ax.set(title=title,xlabel='caption class' if ax==axes[1,0] else 'predicted class',ylabel='image class' if ax==axes[1,0] else 'true class')
    for i in range(4):
        for j in range(4):
            color=im.cmap(im.norm(values[i,j]))
            luminance=.2126*color[0]+.7152*color[1]+.0722*color[2]
            ax.text(j,i,f'{values[i,j]:.2f}' if ax==axes[1,0] else str(values[i,j]),ha='center',va='center',color='white' if luminance<.45 else 'black')
folder=ROOT/'outputs';folder.mkdir(exist_ok=True)
fig.savefig(folder/'retrieval.png',dpi=140);fig.savefig(folder/'retrieval.svg');plt.close(fig)
policies=[]
with tempfile.TemporaryDirectory() as temporary:
    for policy in ['fp32','w8a32','w8a8']:
        path=Path(temporary)/policy;m.export_artifact(path,state['params'],m.calibrate(state['params'],train),policy);artifact=m.load_artifact(path)
        zi=np.concatenate([m.infer(artifact,images=held['images'][i:i+1]) for i in range(len(labels))])
        zt=np.concatenate([m.infer(artifact,captions=held['captions'][i:i+1]) for i in range(len(labels))])
        scores=zi@zt.T;correct=int(np.sum(labels[scores.argmax(1)]==labels))
        policies.append({'policy':policy,'image_to_text_correct':correct,'count':len(labels),'max_score_error':float(np.max(np.abs(scores-clean['scores']))),'artifact_bytes':sum(f.stat().st_size for f in path.iterdir()),'timing':m.measure(artifact,held['images'][:8])})
paired_control={str(seed):{'clean':m.evaluate(state['params'],m.fixture(seed,per_class=5,shift=0))['image_to_text_correct'],'shifted':m.evaluate(state['params'],m.fixture(seed,per_class=5,shift=1))['image_to_text_correct']} for seed in [7,9]}
receipt={'paired_shift_control':paired_control,'environment':{'jax':m.jax.__version__,'backend':m.jax.default_backend(),'numpy':np.__version__},'source_sha256':hashlib.sha256((ROOT/'solution/model.py').read_bytes()).hexdigest(),'figure_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'train_data_hash':m.dataset_hash(train),'held_data_hash':m.dataset_hash(held),'training_loss':losses,'clean_correct':clean['image_to_text_correct'],'shifted_correct':shift['image_to_text_correct'],'count':len(labels),'class_similarity':similarity.tolist(),'shifted_confusion':confusion.tolist(),'precision':policies,'scope':'Synthetic four-concept image/text fixture. Actual CPU training/export/inference; no natural-language or edge-device qualification.'}
(folder/'evidence.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ['clean_correct','shifted_correct','class_similarity','shifted_confusion','precision']},indent=2))
