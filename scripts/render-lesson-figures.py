"""Execute canonical figure experiments in fresh CPU processes and render reproducible assets."""
import argparse
import contextlib
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

def draw_panel(ax, data):
    import numpy as np
    palette = ['#6240ad', '#087c83', '#bc532c', '#405d9d']
    kind = data['kind']
    if kind in ('heatmap', 'field'):
        a = np.asarray(data['values'], dtype=float)
        assert a.ndim == 2 and np.isfinite(a).all()
        kw = {'cmap': 'PuOr' if data.get('diverging') else 'Purples', 'aspect': 'auto'}
        if kind == 'field':
            kw.update(extent=data['extent'], origin='lower', vmin=0, vmax=1)
        elif data.get('diverging'):
            limit = max(float(np.abs(a).max()), 1e-10)
            kw.update(vmin=-limit, vmax=limit)
        if data['unit'] in ('attention weight', 'next-token probability'):
            kw.update(vmin=0, vmax=1)
        device_ids = 'device ID' in data['unit']
        if device_ids:
            from matplotlib.colors import ListedColormap, BoundaryNorm
            ids=np.unique(a).astype(int)
            kw.update(cmap=ListedColormap(palette[:len(ids)]), norm=BoundaryNorm(np.arange(ids.min()-0.5,ids.max()+1.5),len(ids)))
        im = ax.imshow(a, **kw)
        ax.figure.colorbar(im, ax=ax, shrink=0.8, label=data['unit'], **({'ticks':ids} if device_ids else {}))
        if kind == 'heatmap':
            if a.size <= 32:
                mid = (float(a.min()) + float(a.max())) / 2
                for (i, j), value in np.ndenumerate(a):
                    rgb=im.cmap(im.norm(value))[:3]
                    luminance=sum(c*w for c,w in zip(rgb,[0.2126,0.7152,0.0722]))
                    ax.text(j, i, f'{value:.3g}', ha='center', va='center', fontsize=11,
                            color='white' if luminance < 0.5 else '#20202c')
            ax.set_xticks(range(a.shape[1]), data.get('columns', [str(i) for i in range(a.shape[1])]))
            ax.set_yticks(range(a.shape[0]), data.get('rows', [str(i) for i in range(a.shape[0])]))
            if a.shape[1] > 5: ax.tick_params(axis='x', labelsize=11)
        else:
            pts = np.asarray(data['points']); labels = np.asarray(data['labels'])
            for label, marker in [(0, 'o'), (1, '^')]:
                selected = pts[labels == label]
                ax.scatter(selected[:, 0], selected[:, 1], label=f'observed label {label}', marker=marker,
                           s=38, facecolors='white' if label == 0 else '#f4b547', edgecolors='#272238', linewidths=1)
            ax.legend(loc='upper left', fontsize=10)
    elif kind in ('line', 'bar'):
        series = data['series']
        x = np.asarray(data.get('x', list(range(len(data.get('labels', []))))))
        assert len(x) and np.isfinite(x).all()
        for i, s in enumerate(series):
            y = np.asarray(s['y'], dtype=float)
            assert y.shape == x.shape and np.isfinite(y).all(), s['label']
            if data.get('yscale') == 'log': assert (y > 0).all()
            if kind == 'line':
                ax.plot(x, y, color=palette[i % len(palette)], label=s['label'], linewidth=2.3,
                        linestyle=['-', '--', '-.', ':'][i % 4], marker='o' if len(x) <= 25 else None, markersize=3)
            else:
                width = 0.78 / len(series)
                bars = ax.bar(x + (i - (len(series)-1)/2)*width, y, width, label=s['label'], color=palette[i % len(palette)])
                if len(x)*len(series) <= 12: ax.bar_label(bars, fmt='%.3g', fontsize=10, padding=3)
        if kind == 'bar':
            ax.set_xticks(x, data['labels'])
            ax.margins(y=0.17)
            if any(len(t) > 13 for t in data['labels']): ax.tick_params(axis='x', labelrotation=15, labelsize=11)
        if data.get('xscale'): ax.set_xscale(data['xscale'])
        if data.get('yscale'): ax.set_yscale(data['yscale'], **({'linthresh':data.get('linthresh',2)} if data['yscale']=='symlog' else {}))
        for marker in data.get('markers',[]):
            ax.scatter([marker['x']],[marker['y']],color='#bc532c',s=45,zorder=5)
            ax.annotate(marker['label'],(marker['x'],marker['y']),xytext=(-85,12),textcoords='offset points',fontsize=10)
        for boundary in data.get('boundaries', []): ax.axvline(boundary, color='#716e7b', linestyle=':', linewidth=1)
        ax.grid(axis='y', alpha=0.2)
        ax.legend(fontsize=11, loc='best')
    elif kind == 'vectors':
        for i, arrow in enumerate(data['arrows']):
            start = np.asarray(arrow['start']); end = np.asarray(arrow['end'])
            ax.annotate('', xy=end, xytext=start, arrowprops={'arrowstyle':'->','color':palette[i], 'lw':2.8})
            mid=(start+end)/2
            ax.annotate(arrow['label'], mid, xytext=(7,7), textcoords='offset points', color=palette[i])
        ax.set(xlim=(-0.7,4.5),ylim=(-0.7,5),xlabel='first coordinate',ylabel='second coordinate')
        ax.set_aspect('equal'); ax.grid(alpha=0.2)
    elif kind == 'flow':
        from matplotlib.patches import FancyBboxPatch
        nodes=data['nodes']; ax.set(xlim=(0,1),ylim=(0,len(nodes))); ax.axis('off')
        if data.get('layout') == 'merge':
            import textwrap
            for i,label in enumerate(nodes[:3]):
                x=0.02+i*0.33
                ax.add_patch(FancyBboxPatch((x,3.7),0.29,1.0,boxstyle='round,pad=0.015',facecolor='#f0eafa',edgecolor='#6240ad'))
                ax.text(x+0.145,4.2,textwrap.fill(label,20),ha='center',va='center',fontsize=11)
                ax.annotate('',(0.5,2.8),(x+0.145,3.67),arrowprops={'arrowstyle':'->','color':'#6240ad'})
            for y,label in [(2.4,nodes[3]),(0.9,nodes[4])]:
                ax.add_patch(FancyBboxPatch((0.08,y-0.35),0.84,0.7,boxstyle='round,pad=0.025',facecolor='#f0eafa',edgecolor='#6240ad'))
                ax.text(0.5,y,label,ha='center',va='center',fontsize=12)
            ax.annotate('',(0.5,1.3),(0.5,2.0),arrowprops={'arrowstyle':'->','color':'#6240ad'})
        else:
            for i, label in enumerate(nodes):
                y=len(nodes)-i-0.5
                ax.add_patch(FancyBboxPatch((0.08,y-0.3),0.84,0.6,boxstyle='round,pad=0.025',facecolor='#f0eafa',edgecolor='#6240ad'))
                ax.text(0.5,y,label,ha='center',va='center',fontsize=12)
                if i<len(nodes)-1: ax.annotate('',(0.5,y-0.67),(0.5,y-0.32),arrowprops={'arrowstyle':'->','color':'#6240ad'})
    else: raise ValueError(f'Unknown figure kind: {kind}')
    if data.get('xlabel'): ax.set_xlabel(data['xlabel'])
    if data.get('ylabel'): ax.set_ylabel(data['ylabel'])
    if data.get('title'): ax.set_title(data['title'], loc='left', fontsize=12, pad=12)
    for spine in ('top','right'): ax.spines[spine].set_visible(False)


def worker(lesson):
    os.environ['JAX_PLATFORMS']='cpu'
    os.environ.setdefault('MPLCONFIGDIR','/tmp/jaxpathways-matplotlib')
    source=json.loads((ROOT/lesson['path']/'lesson.json').read_text())['content']; visual=source['visual']
    namespace={'__name__':'__main__'}; captured=io.StringIO()
    # Run the same complete example before the supplemental visualization code.
    with contextlib.redirect_stdout(captured):
        if source.get('buildCode'): exec(compile(source['buildCode'],lesson['id']+'-build','exec'),namespace)
        exec(compile(source['code'],lesson['id']+'-example','exec'),namespace)
        if visual['kind']=='executed':
            exec(compile(visual['code'],lesson['id']+'-figure','exec'),namespace)
            data=namespace['visual_data']
        else: data={'kind':'flow','nodes':visual['nodes'],'layout':visual.get('layout','sequence')}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import jax
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.labelcolor':'#30273e','text.color':'#30273e','svg.fonttype':'none','svg.hashsalt':lesson['id']})
    panels=data.get('panels',[data]); count=len(panels)
    # Stack panels vertically to remain readable at phone width and in EPUB.
    fig, axes=plt.subplots(count,1,figsize=(8,4.6*count),squeeze=False,layout='constrained')
    for ax,panel in zip(axes[:,0],panels):draw_panel(ax,panel)
    folder=ROOT/lesson['path']/'outputs';folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/'figure.svg',metadata={'Date':None,'Description':visual['title']})
    fig.savefig(folder/'figure.png',dpi=140)
    plt.close(fig)
    record={'schemaVersion':1,'lessonId':lesson['id'],'contentHash':digest(source),'rendererHash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'kind':visual['kind'],'generatedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'environment':{'backend':jax.default_backend(),'deviceCount':jax.local_device_count(),'jax':jax.__version__,'numpy':np.__version__,'matplotlib':matplotlib.__version__},
            'stdout':captured.getvalue(),'data':data}
    assert record['environment']['backend']=='cpu'
    assert record['environment']['deviceCount']==source.get('runtime',{}).get('deviceCount',1)
    (folder/'visual.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--lesson');parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    course=json.loads((ROOT/'curriculum/course.json').read_text())
    lessons=[l for p in course['phases'] for l in p['lessons'] if l['status']=='authored' and (not args.lesson or l['id']==args.lesson)]
    if args.worker:
        assert len(lessons)==1;worker(lessons[0])
    else:
        failures=[]
        for lesson in lessons:
            result=subprocess.run([sys.executable,__file__,'--worker','--lesson',lesson['id']],cwd=ROOT,capture_output=True,text=True,timeout=120)
            print(('PASS ' if result.returncode==0 else 'FAIL ')+lesson['id'],flush=True)
            if result.returncode:print(result.stdout,result.stderr,flush=True);failures.append(lesson['id'])
        if failures:raise SystemExit('Failed figures: '+', '.join(failures))
