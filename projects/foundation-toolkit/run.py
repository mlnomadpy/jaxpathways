"""Connect the four reference stages and plot their actual outputs."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'solution'))
import toolkit as m
report=m.environment_report()
initial=m.initial_state(seed=3,batch=3);final,positions=m.simulate(initial,30)
positions=np.asarray(positions)
mask=np.array([True,True,False])
processed=m.normalize_columns(np.asarray(final['position']),mask)
x=np.asarray(processed['values'])[mask,0];y=(2*x+1).astype(np.float32);weights=np.array([-.1,.2],np.float32)
g=np.asarray(m.batch_gradients(weights,x,y))
r=weights[0]*x+weights[1]-y
oracle=np.stack([r*x+.2*weights[0],r+.2*weights[1]],axis=1)
np.testing.assert_allclose(g,oracle,atol=2e-6)
fig,axes=plt.subplots(2,1,figsize=(8,8),layout='constrained')
for i in range(3):
    path=np.concatenate([np.zeros((1,2)),positions[:,i]],axis=0)
    axes[0].plot(path[:,0],path[:,1],marker='.',label=f'particle {i}')
    axes[0].scatter(path[-1,0],path[-1,1],marker='x',s=70)
axes[0].set(xlabel='first position coordinate (arbitrary units)',ylabel='second position coordinate (arbitrary units)',title='Actual seeded trajectories; crosses mark final positions')
axes[0].legend()
labels=['row 0 slope','row 0 bias','row 1 slope','row 1 bias'];ix=np.arange(4)
axes[1].bar(ix-.18,g.ravel(),.36,label='compiled JAX gradient');axes[1].bar(ix+.18,oracle.ravel(),.36,label='independent residual formula')
axes[1].set_xticks(ix,labels,rotation=15);axes[1].set(ylabel='loss derivative',title='Two valid normalized observations; paired bars should agree')
axes[1].legend()
folder=ROOT/'outputs';folder.mkdir(exist_ok=True)
fig.savefig(folder/'foundations.png',dpi=140);fig.savefig(folder/'foundations.svg');plt.close(fig)
record={'environment':report,'source_sha256':hashlib.sha256((ROOT/'solution/toolkit.py').read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'final_positions':np.asarray(final['position']).tolist(),'mask':mask.tolist(),'normalized':np.asarray(processed['values']).tolist(),'gradients':g.tolist(),'reference':oracle.tolist(),'scope':'Actual CPU reference; dimensionless synthetic state evolution, not a physical model or performance benchmark.'}
(folder/'report.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
