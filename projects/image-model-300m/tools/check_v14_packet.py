#!/usr/bin/env python3
"""Offline integrity and metric reconstruction for the public C03 v14 packet."""
from __future__ import annotations
import argparse, hashlib, json, math, pathlib, sys, importlib.util, io
import numpy as np

PROJECT=pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PACKET=PROJECT/'reference/evidence/v14'
EXPECTED_RECIPE='df0a1a138f0635a39fcaa1b03f154403ed272560df9181fb5ad37722614d3c4c'
EXPECTED_APPROVAL='f4fb56a70ec453ffd995c36c56c9b80cbd7cd9f739deb59f4db1fdd95c91f82b'
CLASS_NAMES=['daisy','dandelion','roses','sunflowers','tulips']

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p): return json.loads(p.read_text())
def require(x,msg):
 if not x: raise ValueError(msg)
def softmax(logits,t=1.0):
 a=np.asarray(logits,dtype=np.float64)/float(t); a-=a.max(axis=1,keepdims=True); e=np.exp(a); return e/e.sum(axis=1,keepdims=True)
def metrics(logits,labels,t=1.0):
 y=np.asarray(labels,dtype=np.int64); p=softmax(logits,t); pred=p.argmax(1); recall=[]
 for c in range(5):
  m=y==c; recall.append(float((pred[m]==c).mean()) if m.any() else 0.0)
 nll=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-300)).mean())
 confusion=np.zeros((5,5),dtype=np.int64)
 np.add.at(confusion,(y,pred),1)
 onehot=np.eye(5)[y]
 conf=p.max(1); correct=pred==y; edges=np.linspace(0.0,1.0,11); ece=0.0; bins=[]
 for i in range(10):
  mask=(conf>=edges[i]) & (conf<(edges[i+1]) if i<9 else conf<=edges[i+1]); n=int(mask.sum())
  acc=float(correct[mask].mean()) if n else None; mean_conf=float(conf[mask].mean()) if n else None
  if n: ece += n/len(y)*abs(acc-mean_conf)
  bins.append({'lower':float(edges[i]),'upper':float(edges[i+1]),'count':n,'accuracy':acc,'mean_confidence':mean_conf})
 return {'count':len(y),'macro_balanced_accuracy':float(np.mean(recall)),'nll':nll,'per_class_recall':recall,'top1_accuracy':float(correct.mean()),
         'top3_accuracy':float(np.mean(np.any(np.argsort(-p,axis=1,kind='stable')[:,:3]==y[:,None],axis=1))),
         'confusion_matrix':confusion.tolist(),'brier':float(np.mean(np.sum(np.square(p-onehot),axis=1))),
         'ece_10_equal_width':float(ece),'ece_bins':bins,'temperature':float(t)}
def close(a,b,tol=2e-8): return abs(float(a)-float(b))<=tol
def equal_tree(a,b):
 if isinstance(a,(float,np.floating)): return isinstance(b,(int,float)) and close(a,b,5e-8)
 if isinstance(a,dict): return isinstance(b,dict) and set(a)==set(b) and all(equal_tree(a[k],b[k]) for k in a)
 if isinstance(a,list): return isinstance(b,list) and len(a)==len(b) and all(equal_tree(x,y) for x,y in zip(a,b))
 return a==b
def assert_metrics(derived,reported,context):
 require(equal_tree(derived,reported),f'{context} metrics mismatch')

def verify(packet=DEFAULT_PACKET, project=PROJECT, check_manifest=True):
 packet=pathlib.Path(packet); project=pathlib.Path(project)
 identity=load(packet/'source-and-run-identities.json'); numeric=load(packet/'public-numeric-evidence.json'); trace=load(packet/'public-training-traces.json')
 recipe_path=packet/'approved-v14-recipe.json'; approval_path=packet/'approved-v14-approval.json'
 require(sha(recipe_path)==EXPECTED_RECIPE,'copied recipe bytes differ from approved v14')
 require(sha(approval_path)==EXPECTED_APPROVAL,'copied approval bytes differ from approved v14')
 recipe=load(recipe_path); approval=load(approval_path)
 source=identity['as_run_source_inventory']; candidate=identity['as_run_candidate_python_inventory']
 require(len(source)==28,'as-run source inventory must contain 28 paths')
 require(len(candidate)==11,'candidate inventory must contain 11 Python paths')
 require(set(source)==set(recipe['source_sha256']),'source inventory paths differ from recipe')
 require(approval['recipe_sha256']==EXPECTED_RECIPE,'approval does not bind exact recipe')
 require(approval['source_files_sha256']==recipe['source_sha256'],'approval and recipe source maps differ')
 require(approval['candidate_api_source_sha256']=={k.split('/',1)[1]:v['sha256'] for k,v in candidate.items()},'candidate inventory differs from approval')
 for rel,row in source.items():
  p=project/rel
  require(p.is_file() and sha(p)==row['sha256']==recipe['source_sha256'][rel],f'source byte mismatch: {rel}')
 for rel,row in candidate.items():
  p=project/rel
  require(p.is_file() and sha(p)==row['sha256'],f'candidate byte mismatch: {rel}')
 actual_candidates={str(p.relative_to(project)) for p in (project/'solution').rglob('*.py')}
 require(actual_candidates==set(candidate),'solution Python inventory has added or missing paths')
 require(identity['frozen_v7']['source_manifest_sha256']=='8dbba316320889596cecef1f06d58b479125f7c14d30bfba24aed5fe60a9b67c','unexpected frozen v7 manifest')
 require(identity['frozen_v7']['archive_sha256']=='4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c','unexpected archive identity')
 require(identity['private_run_receipts']['run_report']['sha256']=='a92f35c18287375c267b0f945f95d28f72d466ae3bfa555a5a27a4c217ff9eef','unexpected v14 report identity')
 require(identity['private_run_receipts']['pipeline_journal']['sha256']=='965b7e7963ca56fe7df7f42b1354362801df91104303060768a882689ef56dc0','unexpected v14 journal identity')
 require(identity['private_run_receipts']['final_handoff']['sha256']=='0c728bd80eb0d6537fa885d6f560fd08095a7b7a7ee278522d7547e97239d28d','unexpected v14 handoff identity')
 require(identity['journal_event_order']==['pretraining_selection_frozen','deploy_selection_frozen','final_split_loaded'],'final split was loaded before frozen choices')
 # Packet tree is numeric and source only.
 prohibited={'.jpg','.jpeg','.png','.webp','.gif','.bmp','.tif','.tiff','.heic','.avif','.npy','.npz','.ckpt','.pt','.pth','.bin','.safetensors','.msgpack','.pkl','.pickle','.joblib'}
 for p in packet.rglob('*'):
  if p.is_file():
   require(p.suffix.lower() not in prohibited or (p.suffix.lower()=='.png' and p.parent.name=='figures'),f'prohibited payload: {p.name}')
 require(not any(any(token in p.name.lower() for token in ('weight','checkpoint','feature-vector','embedding','processor-payload')) for p in packet.rglob('*') if p.is_file()),'prohibited weight/checkpoint/feature payload name')
 if check_manifest:
  mp=packet/'packet-manifest.json'; man=load(mp)
  for rel,digest in man['files'].items():
   p=project/rel if not rel.startswith('reference/evidence/v14/') else project/rel
   require(p.is_file() and sha(p)==digest,f'public manifest mismatch: {rel}')
  manifest_paths=set(man['files'])
  expected_paths=set()
  for rel in ('TEACHING.md','reference/README.md'):
   if (project/rel).is_file(): expected_paths.add(rel)
  for base in ('tools','reference/evidence/v14'):
   for p in (project/base).rglob('*'):
    if p.is_file() and p.name!='packet-manifest.json' and '__pycache__' not in p.parts and p.suffix!='.pyc': expected_paths.add(str(p.relative_to(project)))
  require(manifest_paths==expected_paths,'public manifest does not enumerate exactly all packet/tool/document files')
 # Public numeric data and ordered identities.
 require(numeric['schema']=='c03-v14-public-numeric-evidence-v1','numeric evidence schema mismatch')
 dev=numeric['development']; final=numeric['final']; arms=dev['arms']; final_arms=final['arms']
 require(numeric['classes']==CLASS_NAMES,'class order mismatch')
 require(len(dev['ordered_ids'])==548 and len(final['ordered_ids'])==547,'split counts mismatch')
 require(len(set(dev['ordered_ids']))==548 and len(set(final['ordered_ids']))==547,'duplicate split IDs')
 require(set(dev['ordered_ids']).isdisjoint(final['ordered_ids']),'development and final IDs overlap')
 require(dev['labels']==[CLASS_NAMES.index(x.split('/',1)[0]) for x in dev['ordered_ids']],'development label order mismatch')
 require(final['labels']==[CLASS_NAMES.index(x.split('/',1)[0]) for x in final['ordered_ids']],'final label order mismatch')
 require(len(final['inventory'])==547,'final inventory length mismatch')
 require([r['id'] for r in final['inventory']]==final['ordered_ids'],'final inventory ID order mismatch')
 require([r['class_index'] for r in final['inventory']]==final['labels'],'inventory labels mismatch')
 require(all(r['label']==CLASS_NAMES[r['class_index']] and r['component'] for r in final['inventory']),'missing public source identity fields')
 # Figure inputs and SVGs must be derived from the published numeric arrays.
 figmod_path=project/'tools/build_v14_figures.py'
 spec=importlib.util.spec_from_file_location('c03_v14_figures',figmod_path)
 figmod=importlib.util.module_from_spec(spec); spec.loader.exec_module(figmod)
 expected_figures=figmod.derive_data(numeric,trace)
 require(load(packet/'figure-data.json')==expected_figures,'figure-data.json is not derived from the public arrays')
 for name,chart in expected_figures['figures'].items():
  svg=packet/'figures'/(name+'.svg')
  require(svg.is_file() and svg.read_text()==figmod.svg_chart(chart),f'figure SVG is not deterministic or does not match figure data: {name}')
  png=packet/'figures'/(name+'.png'); bio=io.BytesIO(); figmod.png_chart(chart).save(bio,format='PNG',optimize=False)
  require(png.is_file() and png.read_bytes()==bio.getvalue(),f'figure PNG is not deterministic or does not match figure data: {name}')
 require(set(arms)==set(final_arms) and len(arms)==12,'twelve comparison arms missing')
 derived={'dev':{},'final':{}}
 for arm in sorted(arms):
  require(len(arms[arm])==548 and len(final_arms[arm])==547,'logit row count mismatch '+arm)
  require(all(len(row)==5 for row in arms[arm]+final_arms[arm]),'logit class dimension mismatch '+arm)
  dm=metrics(arms[arm],dev['labels']); fm=metrics(final_arms[arm],final['labels'])
  derived['dev'][arm]=dm; derived['final'][arm]=fm
  assert_metrics(dm,numeric['recipe_reported_records']['dev_metrics'][arm]['raw'],f'development {arm}')
  assert_metrics(fm,numeric['recipe_reported_records']['final_metrics'][arm]['raw'],f'final {arm}')
 # Frozen deploy choice is recomputed from development BA and candidate order only.
 select=numeric['dev_selection']; order=select['candidate_order']
 dev_ba={a:metrics(arms[a],dev['labels'])['macro_balanced_accuracy'] for a in order}
 winner=max(order,key=lambda a:(dev_ba[a],-order.index(a)))
 require(winner==select['dev_only_selected']=='baseline-vit','dev-only deploy selection mismatch')
 require(select['frozen_before_final_split_load'] is True and select['final_results_cannot_reselect'] is True,'selection lock not frozen before final split')
 require(select['recipe_dev_ids']==dev['ordered_ids'],'temperature/deploy selection ID order does not match development IDs')
 # Recompute the development temperature on the documented 601-point log grid.
 grid=np.exp(np.linspace(np.log(.05),np.log(20.0),601)); scored=[(metrics(arms[winner],dev['labels'],t)['nll'],float(t)) for t in grid]
 best=min(v[0] for v in scored); tied=[v for v in scored if abs(v[0]-best)<=1e-14]; best_nll,best_t=min(tied,key=lambda z:(abs(math.log(z[1])),z[1]))
 temp=numeric['baseline_vit_temperature']
 require(close(temp['temperature'],best_t,1e-12) and close(temp['recipe_dev_nll'],best_nll,1e-10),'development temperature grid reconstruction mismatch')
 require(close(best_t,1.0201723141714116,1e-12),'unexpected frozen temperature')
 selected=numeric['selected_final']; require(selected['ids']==final['ordered_ids'] and selected['labels']==final['labels'],'selected arrays do not follow final IDs/labels')
 require(np.allclose(selected['raw_logits'],final_arms['baseline-vit'],rtol=0,atol=1e-7),'selected raw logits differ from baseline-vit arm')
 expected_correct=softmax(selected['raw_logits'])[np.arange(len(final['labels'])),final['labels']]
 require(len(selected['uncalibrated_T1_correct_label_probability'])==547 and np.allclose(selected['uncalibrated_T1_correct_label_probability'],expected_correct,rtol=0,atol=3e-7),'T=1 correct-label probabilities do not derive from raw logits')
 cond=numeric['selected_final_condition_raw_logits']; require(set(cond)=={'original','brightness_plus_0.05','contrast_0.90','center_8x8_gray0','gaussian_blur_sigma1','jpeg_quality60','resize_without_letterbox'},'condition set mismatch')
 require(np.allclose(cond['original'],selected['raw_logits'],rtol=0,atol=1e-7),'original condition logits mismatch')
 cond_metrics={k:metrics(v,final['labels']) for k,v in cond.items()}
 reported_conditions=numeric['recipe_reported_records']['selected_final_conditions']
 for name,logits in cond.items():
  assert_metrics(cond_metrics[name],reported_conditions[name]['raw'],f'final condition {name} raw')
  assert_metrics(metrics(logits,final['labels'],best_t),reported_conditions[name]['calibrated'],f'final condition {name} calibrated')
 selected_raw=cond_metrics['original']; bright_drop=selected_raw['macro_balanced_accuracy']-cond_metrics['brightness_plus_0.05']['macro_balanced_accuracy']
 raw_nll=selected_raw['nll']; calibrated_nll=metrics(selected['raw_logits'],final['labels'],best_t)['nll']; nll_delta=calibrated_nll-raw_nll
 require(close(selected_raw['macro_balanced_accuracy'],0.3185224830158059),'selected balanced accuracy changed')
 require(close(bright_drop,0.02490262683114214),'brightness drop reconstruction mismatch')
 require(close(raw_nll,1.5089652910,1e-9) and close(calibrated_nll,1.5087931145,1e-9),'raw/calibrated NLL reconstruction mismatch')
 require(bright_drop<=.15 and nll_delta<=.05 and selected_raw['macro_balanced_accuracy']<.40,'quality gate reconstruction mismatch')
 require(derived['final']['fit-color-centroid']['macro_balanced_accuracy']>selected_raw['macro_balanced_accuracy'],'selected arm must trail fixed color centroid')
 gate=numeric['recipe_reported_records']['quality_gate']['selected_candidate_only']
 outcomes={'final_macro_balanced_accuracy':selected_raw['macro_balanced_accuracy']>=.40,
           'beats_fixed_color_centroid':selected_raw['macro_balanced_accuracy']>derived['final']['fit-color-centroid']['macro_balanced_accuracy'],
           'brightness_drop':bright_drop<=.15,'calibration_nll_delta':nll_delta<=.05}
 require(gate==outcomes and not numeric['recipe_reported_records']['quality_gate']['all_selected_floors_pass'],'quality-gate outcomes do not reconstruct')
 # Retrieval AP/mAP from complete ordinal ranks with same-component exclusion.
 inv=final['inventory']; n=len(inv); ordinal=numeric['retrieval_rank_ordinals']; ret={}
 for arm,byq in ordinal.items():
  require(len(byq)==547,'retrieval query count mismatch '+arm); aps=[]
  for qstr,rank in byq.items():
   qi=int(qstr); require(0<=qi<n,'invalid retrieval query ordinal')
   eligible=[j for j in range(n) if j!=qi and inv[j]['component']!=inv[qi]['component']]
   require(len(rank)==len(eligible) and len(set(rank))==len(rank) and set(rank)==set(eligible),'rank list incomplete or includes excluded items')
   relevant_total=sum(inv[j]['class_index']==inv[qi]['class_index'] for j in eligible)
   require(relevant_total>0,'query has no relevant candidates')
   hits=0; ap_sum=0.0
   for pos,j in enumerate(rank,1):
    if inv[j]['class_index']==inv[qi]['class_index']:
     hits+=1; ap_sum+=hits/pos
   ap=ap_sum/relevant_total; aps.append(ap)
  ret[arm]={'queries':len(aps),'mean_average_precision':float(np.mean(aps))}
  aggregate=numeric['recipe_reported_records']['retrieval_aggregates'][arm]
  require(close(ret[arm]['mean_average_precision'],aggregate['mean_average_precision'],1e-12),f'retrieval mAP mismatch {arm}')
  query_records=numeric['recipe_reported_records']['retrieval_by_query'][arm]
  require(len(query_records)==547,'private per-query metric record count mismatch '+arm)
  for qstr,rank in byq.items():
   qi=int(qstr); qid=inv[qi]['id']; relevant_total=sum(inv[j]['class_index']==inv[qi]['class_index'] for j in range(n) if j!=qi and inv[j]['component']!=inv[qi]['component'])
   hit=0; ap_sum=0.0; recalls={}
   for pos,j in enumerate(rank,1):
    if inv[j]['class_index']==inv[qi]['class_index']: hit+=1; ap_sum+=hit/pos
   for k in (1,5,10): recalls[str(k)]=sum(inv[j]['class_index']==inv[qi]['class_index'] for j in rank[:k])/relevant_total
   qr=query_records[qid]
   require(qr['relevant_candidates']==relevant_total and close(qr['average_precision'],ap_sum/relevant_total,1e-12),f'per-query AP mismatch: {arm}/{qid}')
   require(all(close(qr['recall'][str(k)],recalls[str(k)],1e-12) for k in (1,5,10)),f'per-query recall mismatch: {arm}/{qid}')
 require(close(ret['baseline-vit']['mean_average_precision'],0.23465095738800937,1e-12),'selected mAP mismatch')
 # Stage updates and image/view/target exposure totals from stripped per-update rows.
 stages=trace['stages']; require(len(stages)==11,'expected 11 stages')
 count=total_images=total_views=0; stage_summary={}
 for name,rows in stages.items():
  require([r['step'] for r in rows]==list(range(1,len(rows)+1)),f'phase-local update sequence invalid: {name}')
  require(all(r['stage_id']==name and r['input_image_count']==8 and len(r['sampled_ids'])==8 for r in rows),f'batch exposure mismatch: {name}')
  count+=len(rows); images=sum(r['input_image_count'] for r in rows); views=sum(r['model_view_count'] for r in rows); targets=sum(r['selected_mae_target_values'] for r in rows)
  total_images+=images; total_views+=views
  stage_summary[name]={'updates':len(rows),'source_images':images,'model_views':views,'selected_mae_target_values':targets,'loss_min':min(r['loss'] for r in rows),'loss_max':max(r['loss'] for r in rows)}
 require(count==384 and sum(len(v) for v in stages.values())==384,'update total is not 384')
 require(stage_summary['mae-pretrain']['source_images']==128 and stage_summary['simclr-pretrain']['source_images']==128,'pretraining source-image exposure mismatch')
 require(stage_summary['simclr-pretrain']['model_views']==256,'SimCLR view exposure mismatch')
 require(identity['private_run_receipts']['v11_failed_startup']['status'].startswith('failed before initialization'),'v11 failed attempt history missing')
 require(len(load(packet/'omitted-private-artifacts.json')['items'])>=1000,'private omitted artifact digest index is incomplete')
 return {'status':'PASS','source_paths':28,'candidate_python_paths':11,'dev_examples':548,'final_examples':547,'comparison_arms':12,'dev_deploy_choice':winner,'temperature':best_t,
         'final_macro_balanced_accuracy':selected_raw['macro_balanced_accuracy'],'brightness_drop':bright_drop,'raw_nll':raw_nll,'calibrated_nll':calibrated_nll,
         'quality_gate':'FAIL','retrieval':ret,'updates':count,'stage_exposure':stage_summary,'journal_event_order':identity['journal_event_order'],
         'omitted_private_artifacts':len(load(packet/'omitted-private-artifacts.json')['items'])}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--packet',type=pathlib.Path,default=DEFAULT_PACKET); ap.add_argument('--project-root',type=pathlib.Path,default=PROJECT); ap.add_argument('--no-manifest',action='store_true')
 a=ap.parse_args()
 try: result=verify(a.packet,a.project_root,not a.no_manifest)
 except Exception as e: print(json.dumps({'status':'FAIL','error':str(e)},sort_keys=True),file=sys.stderr); return 1
 print(json.dumps(result,sort_keys=True,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
