#!/usr/bin/env python3
"""Create deterministic SVG figures from public numeric evidence only (no archive/JAX)."""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
PROJECT=pathlib.Path(__file__).resolve().parents[1]
EVIDENCE=PROJECT/'reference/evidence/v14'

OBJECTIVE={
 'supervised_cross_entropy':('Cross-entropy loss','nats / example',['baseline-cnn','baseline-vit','probe-random','probe-mae','probe-simclr','probe-supervised','finetune-mae','finetune-simclr','partial-selected']),
 'masked_reconstruction':('Masked reconstruction loss','mean squared normalized-pixel target',['mae-pretrain']),
 'simclr_nt_xent':('SimCLR contrastive loss','NT-Xent nats / anchor',['simclr-pretrain']),
}
COLORS=['#2f6b8a','#c45d35','#527a48','#805b9a','#ae8c29','#466c77','#8b4c56','#666666','#1c7b73']
def load(p): return json.loads(pathlib.Path(p).read_text())
def derive_data(numeric,traces):
 series={}
 for group,(title,unit,names) in OBJECTIVE.items():
  series[group]={'title':title,'unit':unit,'series':[{'name':name,'updates':[r['step'] for r in traces['stages'][name]],'loss':[r['loss'] for r in traces['stages'][name]]} for name in names]}
 final=numeric['final']; labels=final['labels']; baseline_logits=final['arms']['baseline-vit']; color=final['arms']['fit-color-centroid']
 def ba(x):
  p=np.asarray(x,dtype=float).argmax(1); y=np.asarray(labels,dtype=int)
  return float(np.mean([np.mean(p[y==c]==c) for c in range(5)]))
 cond={}
 for name,logits in numeric['selected_final_condition_raw_logits'].items(): cond[name]=ba(logits)
 orig=cond['original']; cond_drop={k:orig-v for k,v in cond.items() if k!='original'}
 sel=numeric['selected_final']
 def nll(logits,t):
  a=np.asarray(logits,dtype=float)/t; a-=a.max(axis=1,keepdims=True); e=np.exp(a); p=e/e.sum(axis=1,keepdims=True); y=np.asarray(labels); return float(-np.log(p[np.arange(len(y)),y]).mean())
 temp=float(numeric['baseline_vit_temperature']['temperature'])
 retrieval={}
 inv=final['inventory']; n=len(inv)
 for arm,rankings in numeric['retrieval_rank_ordinals'].items():
  aps=[]
  for qstr,rank in rankings.items():
   q=int(qstr); rel=sum(inv[j]['class_index']==inv[q]['class_index'] and inv[j]['component']!=inv[q]['component'] for j in range(n) if j!=q)
   hit=0; ap=0.0
   for pos,j in enumerate(rank,1):
    if inv[j]['class_index']==inv[q]['class_index']: hit+=1; ap+=hit/pos
   aps.append(ap/rel)
  retrieval[arm]={'queries':len(aps),'mAP':float(np.mean(aps))}
 return {
  'schema':'c03-v14-figure-data-v1','figures':{
   'loss-supervised':{'title':OBJECTIVE['supervised_cross_entropy'][0],'unit':OBJECTIVE['supervised_cross_entropy'][1],'x_axis':'phase-local update (one-based)','y_axis':'cross-entropy loss (nats/example)','series':series['supervised_cross_entropy']},
   'loss-mae':{'title':OBJECTIVE['masked_reconstruction'][0],'unit':OBJECTIVE['masked_reconstruction'][1],'x_axis':'phase-local update (one-based)','y_axis':'masked reconstruction MSE per normalized-pixel target','series':series['masked_reconstruction']},
   'loss-simclr':{'title':OBJECTIVE['simclr_nt_xent'][0],'unit':OBJECTIVE['simclr_nt_xent'][1],'x_axis':'phase-local update (one-based)','y_axis':'NT-Xent loss (nats/anchor)','series':series['simclr_nt_xent']},
  'quality-classification':{'title':'Held-out classification quality - FAIL','unit':'macro balanced accuracy','sample_count':547,'bars':{'baseline-vit':ba(baseline_logits),'fit-color-centroid':ba(color)},'threshold':0.40,'threshold_label':'required minimum 0.40'},
   'quality-brightness':{'title':'Brightness robustness gate - PASS','unit':'absolute balanced-accuracy change from original','sample_count':547,'bars':{k:abs(v) for k,v in cond_drop.items()},'threshold':0.15,'threshold_label':'maximum allowed absolute change 0.15'},
   'quality-calibration':{'title':'Calibration NLL degradation — PASS','unit':'calibrated-minus-raw NLL (nats/example)','sample_count':547,'bars':{'T=1.0201723142 vs T=1':nll(sel['raw_logits'],temp)-nll(sel['raw_logits'],1.0)},'raw_nll_T1':nll(sel['raw_logits'],1.0),'calibrated_nll':nll(sel['raw_logits'],temp),'threshold':0.05,'threshold_label':'maximum increase 0.05'},
   'quality-retrieval':{'title':'Selected retrieval - baseline ViT','unit':'mean average precision','sample_count':547,'query_count':547,'bars':retrieval},
  }}

def esc(s): return str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('"','&quot;')
def svg_chart(chart):
 W,H=900,500; left,right,top,bottom=100,35,80,105; plotw=W-left-right; ploth=H-top-bottom
 title=chart['title']; unit=chart['unit']; items=[]
 if 'series' in chart:
  allv=[float(y) for s in chart['series']['series'] for y in s['loss']]; low=min(allv); high=max(allv); pad=(high-low)*.08 or .1; ymin=max(0,low-pad); ymax=high+pad
  xcount=max(len(s['loss']) for s in chart['series']['series']); xmap=lambda x:left+(x-1)/max(xcount-1,1)*plotw; ymap=lambda y:top+ploth-(y-ymin)/(ymax-ymin)*ploth
  items.append(f'<text x="{W/2}" y="34" text-anchor="middle" class="title">{esc(title)}</text>')
  for tick in range(5):
   val=ymin+(ymax-ymin)*tick/4; yy=ymap(val); items.append(f'<path d="M {left} {yy:.2f} H {W-right}" class="grid"/><text x="{left-12}" y="{yy+4:.2f}" text-anchor="end" class="tick">{val:.3g}</text>')
  for si,s in enumerate(chart['series']['series']):
   pts=' '.join(f'{xmap(x):.2f},{ymap(y):.2f}' for x,y in zip(s['updates'],s['loss']))
   color=COLORS[si%len(COLORS)]; items.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
   items.append(f'<text x="{left+si*112}" y="{H-45}" class="legend" fill="{color}">{esc(s["name"])}</text>')
  items += [f'<text x="{W/2}" y="{H-14}" text-anchor="middle" class="axis">{esc(chart["x_axis"])}</text>',f'<text x="22" y="{H/2}" transform="rotate(-90 22 {H/2})" text-anchor="middle" class="axis">{esc(chart["y_axis"])}</text>']
 else:
  bars=chart['bars']; vals={k:(v['mAP'] if isinstance(v,dict) else float(v)) for k,v in bars.items()}; n=len(vals)
  vals_all=list(vals.values()); threshold=chart.get('threshold')
  if 'quality-classification' in title: ymin=0; ymax=max(.5,max(vals_all)+.08)
  elif 'Calibration NLL' in title: ymin=-.01; ymax=.06
  elif 'brightness' in title.lower(): ymin=min(-.12,min(vals_all)-.05); ymax=max(.03,float(threshold)+.05)
  else: ymin=0; ymax=max(vals_all+[float(threshold or 0),float(chart.get('threshold_delta',0)),.5])+0.1
  ymap=lambda y:top+ploth-(y-ymin)/(ymax-ymin)*ploth
  items.append(f'<text x="{W/2}" y="34" text-anchor="middle" class="title">{esc(title)}</text>')
  for tick in range(5):
   val=ymin+(ymax-ymin)*tick/4; yy=ymap(val); items.append(f'<path d="M {left} {yy:.2f} H {W-right}" class="grid"/><text x="{left-12}" y="{yy+4:.2f}" text-anchor="end" class="tick">{val:.3f}</text>')
  if threshold is not None:
   yy=ymap(float(threshold)); items.append(f'<path d="M {left} {yy:.2f} H {W-right}" stroke="#9d2f2f" stroke-width="2" stroke-dasharray="7 5"/><text x="{W-right-4}" y="{yy-7:.2f}" text-anchor="end" class="threshold">{esc(chart["threshold_label"])}</text>')
  bw=min(70,plotw/(n*1.8)); gap=plotw/n
  for i,(name,val) in enumerate(vals.items()):
   xx=left+gap*i+(gap-bw)/2; yy=ymap(val); y0=ymap(max(ymin,0)); color=COLORS[i%len(COLORS)]; items.append(f'<rect x="{xx:.2f}" y="{min(yy,y0):.2f}" width="{bw:.2f}" height="{max(1,abs(y0-yy)):.2f}" fill="{color}"/><text x="{xx+bw/2:.2f}" y="{min(yy,y0)-7:.2f}" text-anchor="middle" class="tick">{val:.4f}</text><text x="{xx+bw/2:.2f}" y="{H-65}" text-anchor="middle" class="legend">{esc(name)}</text>')
  items += [f'<text x="{W/2}" y="{H-14}" text-anchor="middle" class="axis">{esc(unit)}; n={chart.get("sample_count","—")}</text>',f'<text x="22" y="{H/2}" transform="rotate(-90 22 {H/2})" text-anchor="middle" class="axis">{esc(unit)}</text>']
 return '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="500" viewBox="0 0 900 500"><style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:20px;font-weight:bold}.axis{font-size:13px}.tick{font-size:11px}.legend{font-size:10px}.grid{stroke:#ddd;stroke-width:1}.threshold{font-size:11px;fill:#9d2f2f}</style><rect width="100%" height="100%" fill="white"/>'+''.join(items)+'</svg>\n'

def png_chart(chart):
 W,H=900,500; left,right,top,bottom=100,35,80,105; pw=W-left-right; ph=H-top-bottom
 img=Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(img); font=ImageFont.load_default(); titlefont=ImageFont.load_default(size=16)
 def vertical_axis(label):
  box=font.getbbox(label); layer=Image.new('RGBA',(box[2]-box[0]+6,box[3]-box[1]+6),(255,255,255,0)); ld=ImageDraw.Draw(layer)
  ld.text((3-box[0],3-box[1]),label,fill='#222',font=font); layer=layer.rotate(90,expand=True)
  img.paste(layer,(8,(H-layer.height)//2),layer)
 d.text((W//2,22),chart['title'],fill='#222',font=titlefont,anchor='mt')
 if 'series' in chart:
  rows=chart['series']['series']; allv=[float(y) for s in rows for y in s['loss']]; lo=min(allv); hi=max(allv); pad=(hi-lo)*.08 or .1; ymin=max(0,lo-pad); ymax=hi+pad
  xcount=max(len(s['loss']) for s in rows); xmap=lambda x:left+(x-1)/max(xcount-1,1)*pw; ymap=lambda y:top+ph-(y-ymin)/(ymax-ymin)*ph
  d.text((W//2,H-25),chart['x_axis'],fill='#222',font=font,anchor='mm'); vertical_axis(chart['y_axis'])
  for tick in range(5):
   val=ymin+(ymax-ymin)*tick/4; yy=int(ymap(val)); d.line((left,yy,W-right,yy),fill='#ddd'); d.text((left-8,yy),f'{val:.3g}',fill='#333',font=font,anchor='rm')
  for i,s in enumerate(rows):
   color=COLORS[i%len(COLORS)]; pts=[(int(xmap(x)),int(ymap(y))) for x,y in zip(s['updates'],s['loss'])]; d.line(pts,fill=color,width=2)
   lx=left+(i%5)*150; ly=H-57+(i//5)*13; d.line((lx,ly+4,lx+18,ly+4),fill=color,width=2); d.text((lx+22,ly),s['name'],fill=color,font=font)
 else:
  bars=chart['bars']; vals={k:(v['mAP'] if isinstance(v,dict) else float(v)) for k,v in bars.items()}; vlist=list(vals.values()); title=chart['title']
  threshold=chart.get('threshold')
  if 'classification quality' in title: ymin=0; ymax=max(.5,max(vlist)+.08)
  elif 'Brightness' in title: ymin=0; ymax=max(.2,max(vlist)+.05)
  elif 'Calibration NLL' in title: ymin=-.01; ymax=.06
  else: ymin=0; ymax=max(.5,max(vlist)+.1)
  ymap=lambda y:top+ph-(y-ymin)/(ymax-ymin)*ph
  d.text((W//2,H-24),f'{chart["unit"]}; n={chart.get("sample_count","—")}',fill='#222',font=font,anchor='mm')
  vertical_axis(chart['unit'])
  for tick in range(5):
   val=ymin+(ymax-ymin)*tick/4; yy=int(ymap(val)); d.line((left,yy,W-right,yy),fill='#ddd'); d.text((left-8,yy),f'{val:.3f}',fill='#333',font=font,anchor='rm')
  if threshold is not None:
   yy=int(ymap(float(threshold))); d.line((left,yy,W-right,yy),fill='#9d2f2f',width=2); d.text((W-right-3,yy-12),chart['threshold_label'],fill='#9d2f2f',font=font,anchor='ra')
  gap=pw/max(1,len(vals)); bw=min(70,int(gap*.6))
  for i,(name,val) in enumerate(vals.items()):
   xx=int(left+gap*i+(gap-bw)/2); yy=int(ymap(val)); d.rectangle((xx,yy,xx+bw,int(ymap(0))),fill=COLORS[i%len(COLORS)]); d.text((xx+bw//2,yy-12),f'{val:.4f}',fill='#222',font=font,anchor='mm'); d.text((xx+bw//2,H-58),name,fill='#222',font=font,anchor='mm')
 return img
def write(packet=EVIDENCE):
 packet=pathlib.Path(packet); data=derive_data(load(packet/'public-numeric-evidence.json'),load(packet/'public-training-traces.json'))
 figdir=packet/'figures'; figdir.mkdir(exist_ok=True)
 (packet/'figure-data.json').write_text(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
 for name,chart in data['figures'].items():
  (figdir/(name+'.svg')).write_text(svg_chart(chart))
  png_chart(chart).save(figdir/(name+'.png'),format='PNG',optimize=False)
 return data
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--packet',type=pathlib.Path,default=EVIDENCE); args=p.parse_args(); data=write(args.packet); print(json.dumps({'figures':len(data['figures']),'output':str(args.packet/'figures')},sort_keys=True))
