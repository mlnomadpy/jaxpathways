"""Save separate objective curves and a CSV trace from a completed run."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='runs/embedding-lineage/reference')
    parser.add_argument('--output',default='projects/embedding-lineage/outputs')
    args=parser.parse_args()
    run=Path(args.run); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    metrics=[json.loads((run/f'checkpoints/stage-{n}/metrics.json').read_text()) for n in (1,2,3)]
    mlm=metrics[0]['train_mlm_history']; contrastive=metrics[1]['train_contrastive_history']; supcon=metrics[2]['train_supervised_contrastive_history']
    rows=[]
    for item in mlm:
        rows.append({'objective':'masked-token NLL','x_unit':'epoch','step':item['epoch'],'loss':item['train_mlm_nll'],'gradient_norm':item['gradient_norm_mean'],'learning_rate':metrics[0]['optimizer_monitoring']['learning_rate']})
    for item in contrastive:
        rows.append({'objective':'symmetric InfoNCE','x_unit':'optimizer update','step':item['step'],'loss':item['loss'],'gradient_norm':item['gradient_norm'],'learning_rate':metrics[1]['optimizer_monitoring']['learning_rate']})
    for item in supcon:
        rows.append({'objective':'supervised contrastive','x_unit':'optimizer update','step':item['step'],'loss':item['loss'],'gradient_norm':item['gradient_norm'],'learning_rate':metrics[2]['optimizer_monitoring']['learning_rate']})
    with (output/'loss-curves.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['objective','x_unit','step','loss','gradient_norm','learning_rate']); writer.writeheader(); writer.writerows(rows)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    series=[(mlm,'epoch','train_mlm_nll','Masked-token pretraining','Masked-token NLL'),
            (contrastive,'step','loss','Contrastive continuation','Symmetric InfoNCE'),
            (supcon,'step','loss','Supervised adaptation','Mean positive log-probability loss')]
    for ax,(items,xkey,ykey,title,ylabel) in zip(axes,series):
        ax.plot([item[xkey] for item in items],[item[ykey] for item in items],marker='o',linewidth=1.7)
        ax.set_title(title); ax.set_xlabel('Epoch' if xkey=='epoch' else 'Optimizer update'); ax.set_ylabel(ylabel); ax.grid(alpha=.25)
    fig.suptitle('Training objectives by stage (separate scales; not retrieval quality)')
    fig.savefig(output/'loss-curves.svg')
    plt.close(fig)

if __name__=='__main__': main()
