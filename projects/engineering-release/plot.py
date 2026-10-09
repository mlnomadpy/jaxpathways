"""Plot actual recorded MLflow validation metrics, keeping the receipt provenance visible."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
r=json.loads((root/'outputs/mlflow-report.json').read_text())
fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained')
ax.bar([str(v) for v in r['rates']],r['validation_mse'],color=['#6240ad','#087c83'])
ax.set(yscale='log',xlabel='learning rate (same 30-update budget)',ylabel='fixed-set validation MSE (log scale)',title='Actual MLflow run comparison')
for i,value in enumerate(r['validation_mse']):ax.annotate(f'{value:.4g}',(i,value),xytext=(0,5),textcoords='offset points',ha='center')
ax.set_ylim(1e-11,1.)
ax.grid(axis='y',alpha=.2)
fig.savefig(root/'outputs/tracked-runs.png',dpi=140)
fig.savefig(root/'outputs/tracked-runs.svg')
plt.close(fig)
