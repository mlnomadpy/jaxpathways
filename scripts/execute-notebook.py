"""Run notebook code in order in a fresh process, retaining real streams and figure outputs."""
import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR','/tmp/jaxpathways-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
notebook=json.loads(Path(sys.argv[1]).read_text())
namespace={'__name__':'__main__'}
cells=[]
active_outputs=[]
def capture_show(*args,**kwargs):
    for number in plt.get_fignums():
        fig=plt.figure(number);stream=io.BytesIO();fig.savefig(stream,format='png',dpi=140)
        active_outputs.append({'output_type':'display_data','metadata':{},'data':{'image/png':base64.b64encode(stream.getvalue()).decode(),'text/plain':['Executed Matplotlib figure']}})
    plt.close('all')
plt.show=capture_show
for cell in notebook['cells']:
    if cell['cell_type']!='code':continue
    source=''.join(cell['source']);out=io.StringIO();err=io.StringIO();active_outputs=[]
    with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
        exec(compile(source,f"{sys.argv[1]}:cell-{len(cells)+1}",'exec'),namespace)
    outputs=[]
    if out.getvalue():outputs.append({'output_type':'stream','name':'stdout','text':out.getvalue().splitlines(keepends=True)})
    if err.getvalue():outputs.append({'output_type':'stream','name':'stderr','text':err.getvalue().splitlines(keepends=True)})
    outputs.extend(active_outputs)
    cells.append({'sourceHash':hashlib.sha256(source.encode()).hexdigest(),'outputs':outputs})
import jax
result={'notebookCells':cells,'runtime':{'backend':jax.default_backend(),'deviceCount':jax.local_device_count()}}
Path(sys.argv[2]).write_text(json.dumps(result)+'\n')
