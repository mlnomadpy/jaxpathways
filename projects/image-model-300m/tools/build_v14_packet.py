#!/usr/bin/env python3
"""Create the rights-safe numeric projection from the retained private C03 v14 run.

This authoring utility needs NumPy and read access to the accepted run and frozen
source manifest. It emits no pixels, feature vectors, weights, or checkpoints.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, shutil
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reference/evidence/v14"
RUN = pathlib.Path('/private/tmp/c03-cpu-pilot-v14-20261021')
HANDOFF = pathlib.Path('/private/tmp/c03-cpu-pilot-v14-final-handoff.json')
DATA = pathlib.Path('/private/tmp/jaxpathways-c03-data-frozen-split-20261021-v7')
RECIPE = pathlib.Path('/private/tmp/c03-sol-cpu-pilot-approved-v14.json')
APPROVAL = pathlib.Path('/private/tmp/c03-sol-cpu-pilot-approval-v14.json')
ARCHIVE = pathlib.Path('/private/tmp/jaxpathways-flower-photos-clean.tgz')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n')
def clean_json(v):
    if isinstance(v,dict): return {str(k):clean_json(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [clean_json(x) for x in v]
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(np.floating,)): return float(v)
    return v

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=pathlib.Path,default=RUN); ap.add_argument('--handoff',type=pathlib.Path,default=HANDOFF)
    ap.add_argument('--data',type=pathlib.Path,default=DATA); ap.add_argument('--archive',type=pathlib.Path,default=ARCHIVE)
    ap.add_argument('--recipe',type=pathlib.Path,default=RECIPE); ap.add_argument('--approval',type=pathlib.Path,default=APPROVAL)
    ap.add_argument('--output',type=pathlib.Path,default=EVIDENCE)
    a=ap.parse_args(); out=a.output
    if out.exists() and any(out.iterdir()): raise SystemExit(f'refusing nonempty output: {out}')
    out.mkdir(parents=True,exist_ok=True)
    report=json.loads((a.run/'run-report.json').read_text())
    hand=json.loads(a.handoff.read_text()); recipe=json.loads(a.recipe.read_text()); approval=json.loads(a.approval.read_text())
    journal=json.loads((a.run/'pipeline-journal.json').read_text())
    sm_path=a.data/'source-manifest.json'; sm=json.loads(sm_path.read_text())
    expected={k:v for k,v in recipe['source_sha256'].items()}
    candidate={k:v for k,v in approval['candidate_api_source_sha256'].items()}
    if len(expected)!=28 or len(candidate)!=11: raise SystemExit('unexpected approved source inventory cardinality')
    if sha(a.recipe)!='df0a1a138f0635a39fcaa1b03f154403ed272560df9181fb5ad37722614d3c4c': raise SystemExit('recipe hash mismatch')
    if sha(a.approval)!='f4fb56a70ec453ffd995c36c56c9b80cbd7cd9f739deb59f4db1fdd95c91f82b': raise SystemExit('approval hash mismatch')
    if sha(sm_path)!='8dbba316320889596cecef1f06d58b479125f7c14d30bfba24aed5fe60a9b67c': raise SystemExit('v7 source manifest hash mismatch')
    if sha(a.archive)!='4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c': raise SystemExit('archive hash mismatch')
    # Exact bytes reviewed by Sol are retained verbatim, under distinct recipe/approval names.
    shutil.copyfile(a.recipe,out/'approved-v14-recipe.json'); shutil.copyfile(a.approval,out/'approved-v14-approval.json')
    source_ok={}
    for rel,digest in expected.items():
        p=ROOT/rel
        source_ok[rel]={'sha256':digest,'present':p.is_file(),'matches':p.is_file() and sha(p)==digest}
    candidate_ok={}
    for rel,digest in candidate.items():
        p=ROOT/'solution'/rel
        candidate_ok['solution/'+rel]={'sha256':digest,'present':p.is_file(),'matches':p.is_file() and sha(p)==digest}
    if not all(x['matches'] for x in source_ok.values()) or not all(x['matches'] for x in candidate_ok.values()): raise SystemExit('as-run sources differ from reviewed hashes')
    dump(out/'source-and-run-identities.json',{
      'schema':'c03-v14-public-source-identity-v1','as_run_source_inventory':source_ok,'as_run_candidate_python_inventory':candidate_ok,
      'external_recipe':{'file':'approved-v14-recipe.json','sha256':sha(out/'approved-v14-recipe.json')},
      'external_approval':{'file':'approved-v14-approval.json','sha256':sha(out/'approved-v14-approval.json')},
      'frozen_v7':{'source_manifest_sha256':sha(sm_path),'archive_sha256':sha(a.archive),
        'prepared_split_rows':sm['prepared_split_rows'],'prepared_array_files':sm['prepared_array_files']},
      'private_run_receipts':{
        'run_report':{'path':str(a.run/'run-report.json'),'sha256':sha(a.run/'run-report.json')},
        'pipeline_journal':{'path':str(a.run/'pipeline-journal.json'),'sha256':sha(a.run/'pipeline-journal.json')},
        'final_handoff':{'path':str(a.handoff),'sha256':sha(a.handoff)},
        'preflight':{'path':'/private/tmp/c03-sol-v14-no-update-preflight.json','sha256':'42b79f19fe3824696fa1513501b952b4bfc8f4b90cdea125ebca06ebcf2148e8'},
        'v11_failed_startup':{'path':'/private/tmp/c03-cpu-pilot-v11-20261021','status':'failed before initialization; zero optimizer updates',
          'failure_receipt_path':'/private/tmp/c03-cpu-pilot-v11-20261021/pretraining-failure-receipt.json','failure_receipt_sha256':sha(pathlib.Path('/private/tmp/c03-cpu-pilot-v11-20261021/pretraining-failure-receipt.json')),
          'pipeline_journal_path':'/private/tmp/c03-cpu-pilot-v11-20261021/pipeline-journal.json','pipeline_journal_sha256':sha(pathlib.Path('/private/tmp/c03-cpu-pilot-v11-20261021/pipeline-journal.json'))},
        'v13_source_bound_handoff':{'path':'/private/tmp/c03-r11-v13-source-bound-handoff.json','sha256':'b01750ffd6b5f5ad4effb1478829dae0ea1d68da3b21b850dafec064de7cf3cf'},
        'v14_independent_coldload':{'path':'/private/tmp/c03-sol-v14-independent-coldload.json','sha256':'6a12b0023090ad9ba11e3af48054fa534782e205727171c397f04626f242956a'},
        'v14_independent_ancestry':{'path':'/private/tmp/c03-sol-v14-independent-ancestry.json','sha256':sha(pathlib.Path('/private/tmp/c03-sol-v14-independent-ancestry.json'))}},
      'journal_event_order':[e['event'] for e in journal['events']],
      'journal_events':journal['events'],
      'private_run':{'output_directory':str(a.run),'file_count':hand['retained_output_manifest']['file_count'],'bytes':hand['retained_output_manifest'].get('total_bytes',None),
        'report_sha256':report['raw_artifact_sha256']['raw/classifier-comparison-index.json'] if False else sha(a.run/'run-report.json'),
        'handoff_manifest_sha256':sha(a.handoff)},
      'derivation':{
        'numeric_projection':'public-numeric-evidence.json is projected from the named private raw NPZ/JSON files and run report; arrays are serialized as JSON lists.',
        'training_traces':'public-training-traces.json strips each trace to phase-local step, loss, sampled source IDs, and exposure counters.',
        'retrieval':'rank ordinals index the ordered final inventory; ranking order is retained, source IDs/classes/components are attached by inventory.',
        'prohibited_payloads':'pixels, archive members, feature vectors, checkpoints, model weights, fitted processors and private consumer bundle are omitted.'}})
    # Build class/component lookup from frozen v7 source manifest. No source bytes are copied.
    records={r['id']:r for r in sm['records']}
    comp={}
    for component,entry in sm['grouping']['component_members'].items():
        for image_id in entry['all_attributed_ids']: comp[image_id]=component
    with np.load(a.run/'raw/all-classifier-comparisons.npz',allow_pickle=False) as z:
        dev_ids=z['dev_ids'].astype(str).tolist(); final_ids=z['final_ids'].astype(str).tolist()
        dev_labels=z['dev_labels'].astype(int).tolist(); final_labels=z['final_labels'].astype(int).tolist()
        arms={name:{'dev_raw_logits':z['dev_logits__'+name].astype(float).tolist(),'final_raw_logits':z['final_logits__'+name].astype(float).tolist()} for name in report['final_metrics']}
    inventory=[]
    for idx,(image_id,label) in enumerate(zip(final_ids,final_labels)):
        rec=records.get(image_id)
        if rec is None or rec['label'] not in ('daisy','dandelion','roses','sunflowers','tulips') or comp.get(image_id) is None: raise SystemExit(f'missing final identity fields: {image_id}')
        inventory.append({'ordinal':idx,'id':image_id,'label':rec['label'],'class_index':label,'component':comp[image_id]})
    with np.load(a.run/'raw/selected-final-predictions.npz',allow_pickle=False) as z:
        selected={'selected_arm':'baseline-vit','ids':z['ids'].astype(str).tolist(),'labels':z['labels'].astype(int).tolist(),
                  'raw_logits':z['logits'].astype(float).tolist(),
                  'uncalibrated_T1_correct_label_probability':z['probabilities'][np.arange(len(z['labels'])),z['labels']].astype(float).tolist()}
    with np.load(a.run/'raw/selected-final-condition-logits.npz',allow_pickle=False) as z:
        condition={k.removeprefix('logits__'):z[k].astype(float).tolist() for k in z.files}
    rankings=json.loads((a.run/'raw/final-retrieval-rankings.json').read_text())
    id_index={x['id']:x['ordinal'] for x in inventory}
    ordinal_rankings={}
    for arm in ('baseline-vit','finetune-mae','finetune-simclr','partial-selected'):
        ords={}
        for q,ranked_ids in rankings[arm]['rankings'].items():
            if q not in id_index or any(x not in id_index for x in ranked_ids): raise SystemExit('retrieval ID absent from final inventory')
            ords[str(id_index[q])]=[id_index[x] for x in ranked_ids]
        ordinal_rankings[arm]=ords
    # Public temperatures and dev-only selection lock. The checker recomputes the selected temperature.
    dev_selection=dict(report['deploy_selection'])
    frozen_lock=json.loads((a.run/'selection-lock.json').read_text())
    dev_selection['recipe_dev_ids']=frozen_lock['recipe_dev_ids']
    dev_selection['selection_frozen_before_final_split_load']=frozen_lock['selection_frozen_before_final_split_load']
    temp=report['temperature_selection']['baseline-vit']
    dev_probe=report['probe_based_pretrain_selection']
    traces={}
    for stage,rows in report['stage_traces'].items():
        traces[stage]=[]
        for row in rows:
            traces[stage].append({k:clean_json(row[k]) for k in ('stage_id','step','loss','sampled_ids','input_image_count','model_view_count','input_pixel_values','selected_mae_target_values','source_positive_pair_count','source_positive_relation_checked','epoch_spans') if k in row})
    numeric={
      'schema':'c03-v14-public-numeric-evidence-v1','classes':['daisy','dandelion','roses','sunflowers','tulips'],
      'development':{'ordered_ids':dev_ids,'labels':dev_labels,'arms':{k:v['dev_raw_logits'] for k,v in arms.items()}},
      'final':{'ordered_ids':final_ids,'labels':final_labels,'inventory':inventory,'arms':{k:v['final_raw_logits'] for k,v in arms.items()}},
      'selected_final':selected,'selected_final_condition_raw_logits':condition,
      'dev_selection':clean_json(dev_selection),'pretraining_selection':clean_json(dev_probe),
      'baseline_vit_temperature':clean_json(temp),
      'recipe_reported_records':{'dev_metrics':clean_json(report['dev_metrics']),'final_metrics':clean_json(report['final_metrics']),
        'quality_gate':clean_json(report['quality_gate']),'selected_final_conditions':clean_json(report['selected_final_conditions']),
        'retrieval_aggregates':{k:{'evaluated_query_count':v['evaluated_query_count'],'mean_average_precision':v['mean_average_precision']} for k,v in report['retrieval'].items()},
        'retrieval_by_query':{arm:{qid:{'relevant_candidates':v['relevant_candidates'],'average_precision':v['average_precision'],'recall':v['recall']} for qid,v in rec['per_query'].items()} for arm,rec in report['retrieval'].items()}},
      'retrieval_rank_ordinals':ordinal_rankings}
    dump(out/'public-numeric-evidence.json',numeric)
    dump(out/'public-training-traces.json',{'schema':'c03-v14-public-training-traces-v1','stages':traces})
    # Private artifact digest inventory is hash-only, with reason. Do not copy the files.
    omitted=[]
    for rel,meta in hand['retained_output_manifest']['files'].items():
        if rel in ('run-report.json','pipeline-journal.json'): continue
        lower=rel.lower()
        if 'checkpoint' in lower or lower.endswith(('.npz','.npy')) or 'consumer' in lower:
            reason='private trained parameters or optimizer/checkpoint state'
        elif lower.endswith(('.jpg','.jpeg','.png','.webp')) or 'processor' in lower:
            reason='private pixels or fitted preprocessing payload'
        elif 'feature' in lower:
            reason='private feature-vector artifact; rankings are projected to ordinals'
        elif 'rank' in lower or 'prediction' in lower or 'comparison' in lower:
            reason='private raw numeric artifact; permitted fields are published as a traced projection'
        elif 'trace' in lower:
            reason='unstripped private training trace; stripped public exposure/loss fields are published separately'
        else:
            reason='private run artifact retained by hash only; see public numeric projection and run receipts'
        omitted.append({'path':rel,'bytes':meta['bytes'],'sha256':meta['sha256'],'reason':reason})
    dump(out/'omitted-private-artifacts.json',{'schema':'c03-v14-private-omission-digest-index-v1','items':omitted})
    print(f'created public evidence in {out}; source files verified={len(source_ok)} candidate files verified={len(candidate_ok)} final IDs={len(final_ids)}')
if __name__=='__main__': main()
