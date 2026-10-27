"""LLMOps: version prompts, evaluate behavior and trace requests: worked experiments and reference solutions. CPU checks."""



# Bounded local LLMOps adapter. The trace records actual local retrieval and generation calls.
# It uses no MLflow service, network, paid API, or canned per-case answer array.
import hashlib, json

def fingerprint(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(raw).hexdigest()[:16]

corpus={
 'doc-hours':'Support is open Monday through Friday, 09:00 to 17:00.',
 'doc-refund':'Refunds are available within 30 days of purchase.',
 'doc-return':'Return items with the original receipt.'
}
# Stable heterogeneous case IDs and frozen expected outcomes.
cases=[
 {'id':'hours-west-17','query':'When is support open?','expected':'Support is open Monday through Friday, 09:00 to 17:00.','evidence':'doc-hours','supported':True},
 {'id':'refund-order-204','query':'What is the refund period?','expected':'Refunds are available within 30 days of purchase.','evidence':'doc-refund','supported':True},
 {'id':'return-ambiguous-9','query':'Can I return an item?','expected':'Return items with the original receipt.','evidence':'doc-return','supported':True},
 {'id':'unsupported-tax-2','query':'What is the tax rate?','expected':'','evidence':None,'supported':False},
 {'id':'case-sensitive-refund-77','query':'How many days for a refund?','expected':'Refunds are available within 30 days of purchase.','evidence':'doc-refund','supported':True},
]

def retrieve(query,index,excluded=()):
    def normalize(word):
        word=word.lower().strip('?.!,')
        return word[:-1] if len(word)>4 and word.endswith('s') else word
    terms={normalize(w) for w in query.split() if len(w)>3}
    ranked=[]
    for doc_id,text in index.items():
        if doc_id in excluded: continue
        words={normalize(w) for w in text.split()}
        overlap=len(terms & words)
        if overlap: ranked.append((overlap,doc_id,text))
    ranked.sort(key=lambda row:(-row[0],row[1]))
    return ranked[0][1:] if ranked else (None,None)

def local_model_adapter(prompt,context):
    # The prompt is the actual input contract consumed at this model boundary.
    if context is None:
        if 'ABSTAIN_WHEN_EMPTY' in prompt: return ''
        return 'The answer is probably 15 percent.'
    if 'FORMAT_REFUND' in prompt and 'refund' in context.lower(): return 'Answer: '+context
    return context

def run_case(case,candidate):
    trace={'case_id':case['id'],'candidate_id':candidate['id'],'config_hash':candidate['config_hash'],
           'prompt_id':candidate['prompt_id'],'index_id':candidate['index_id'],'model_adapter_id':candidate['model_id']}
    # Simulated infrastructure fault is a transport outcome, not a model answer.
    if case['id'] in candidate['timeouts']:
        trace.update(status='timeout',retrieved_id=None,context=None,response=None,diagnosis='infrastructure')
        return {'id':case['id'],'status':'timeout','response':None},json.dumps(trace,sort_keys=True)
    retrieved_id,context=retrieve(case['query'],candidate['index'],candidate['excluded_docs']+tuple(candidate['case_excluded'].get(case['id'],())))
    trace.update(retrieval={'query':case['query'],'retrieved_id':retrieved_id,'context':context})
    response=local_model_adapter(candidate['prompt'],context)
    trace['generation']={'prompt':candidate['prompt'],'model_input_context':context,'response':response}
    if context is None:
        diagnosis='retrieval' if case['supported'] else ('generation' if response else 'ok')
    elif case['supported'] and retrieved_id!=case['evidence']:
        diagnosis='retrieval'
    elif case['supported'] and response!=case['expected']:
        diagnosis='generation'
    elif not case['supported'] and response:
        diagnosis='generation'
    else: diagnosis='ok'
    trace.update(status='complete',diagnosis=diagnosis)
    return {'id':case['id'],'status':'complete','response':response,'retrieved_id':retrieved_id},json.dumps(trace,sort_keys=True)

def collect_by_id(expected_ids,responses):
    expected=set(expected_ids); got={}
    for response in responses:
        cid=response['id']
        if cid not in expected: raise ValueError('unknown case ID: '+cid)
        if cid in got: raise ValueError('duplicate response ID: '+cid)
        got[cid]=response
    missing=expected-got.keys()
    if missing: raise ValueError('missing response IDs: '+','.join(sorted(missing)))
    return got

def score(cases,responses):
    paired=collect_by_id([c['id'] for c in cases],responses)
    outcomes=[]
    for case in cases:
        result=paired[case['id']]
        passed=(result['status']=='complete' and
                (result['response']==case['expected']) and
                (not case['supported'] or result['retrieved_id']==case['evidence']))
        outcomes.append({'id':case['id'],'passed':passed,'status':result['status']})
    return {'count':len(outcomes),'passed':sum(x['passed'] for x in outcomes),
            'fraction':sum(x['passed'] for x in outcomes)/len(outcomes),'cases':outcomes}

def make_candidate(cid,prompt_id,prompt,index_id,index,excluded=(),timeouts=(),case_excluded=None):
    model_id='extractive-local-adapter-v1'
    config_hash=fingerprint({'prompt_id':prompt_id,'prompt':prompt,'index_id':index_id,
                             'index_hash':fingerprint(index),'model_id':model_id,
                             'excluded_docs':list(excluded),'timeouts':list(timeouts),
                             'case_excluded':case_excluded or {}})
    return {'id':cid,'prompt_id':prompt_id,'prompt':prompt,'index_id':index_id,'index':index,
            'model_id':model_id,'config_hash':config_hash,'excluded_docs':tuple(excluded),'timeouts':tuple(timeouts),
            'case_excluded':case_excluded or {}}

baseline=make_candidate('candidate-baseline-01','prompt-v1-careless','ANSWER_EVEN_WITHOUT_EVIDENCE FORMAT_REFUND',
                        'index-v1-complete',corpus,case_excluded={'refund-order-204':('doc-refund',)},
                        timeouts=('return-ambiguous-9',))
candidate=make_candidate('candidate-grounded-02','prompt-v2-grounded','ABSTAIN_WHEN_EMPTY',
                         'index-v2-complete',corpus)
runs=[]
for config in (baseline,candidate):
    responses=[]; traces=[]
    for case in cases:
        response,trace=run_case(case,config); responses.append(response); traces.append(trace)
    # Simulate asynchronous completion: pair by ID, never by list position.
    responses=list(reversed(responses))
    summary=score(cases,responses)
    runs.append((summary,traces,responses))

before,trace_before,responses_before=runs[0]
after,trace_after,responses_after=runs[1]
assert before['count']==5 and before['fraction']==.2
assert after['count']==5 and after['fraction']==1.
assert [x['id'] for x in before['cases']]==[x['id'] for x in after['cases']]
assert [x['id'] for x in cases]==[json.loads(x)['case_id'] for x in trace_after]
assert baseline['model_id']==candidate['model_id']
assert baseline['config_hash']!=candidate['config_hash']
# Pairing handles reordering; duplicates and incomplete response inventories are rejected.
assert len(collect_by_id([c['id'] for c in cases],responses_after))==5
for damaged in (responses_after[:-1],responses_after+[responses_after[0]],responses_after+[{'id':'invented-case','status':'complete','response':'x'}]):
    try: collect_by_id([c['id'] for c in cases],damaged)
    except ValueError: pass
    else: raise AssertionError('bad case-ID inventory accepted')

def diagnostic(trace,case):
    if trace['diagnosis']=='infrastructure': return 'Infrastructure: timeout occurred before retrieval or generation.'
    if trace['diagnosis']=='retrieval': return 'Retrieval: the required evidence did not reach the model boundary.'
    if trace['diagnosis']=='generation': return 'Generation: inspect the recorded prompt, correct context and adapter response.'
    return 'No failure observed for this case.'
by_id={t['case_id']:t for t in (json.loads(x) for x in trace_before)}
for cid in ('refund-order-204','case-sensitive-refund-77','return-ambiguous-9'):
    case=next(c for c in cases if c['id']==cid)
    print(cid+':',diagnostic(by_id[cid],case))
print('serialized local retrieval+generation trace sample:',trace_before[0])
print('baseline:',json.dumps(before,sort_keys=True))
print('candidate:',json.dumps(after,sort_keys=True))
print('same frozen case IDs and denominator:',len(cases),[c['id'] for c in cases])
print('baseline candidate identity:',baseline['id'],baseline['prompt_id'],baseline['index_id'],baseline['model_id'],baseline['config_hash'])
print('candidate identity:',candidate['id'],candidate['prompt_id'],candidate['index_id'],candidate['model_id'],candidate['config_hash'])
print('Traces are actual deterministic local adapter executions, not hosted LLM or general quality evidence.')

# Figure data experiment
visual_data={'kind':'bar','labels':['baseline (5 cases)','grounded candidate (5 cases)'],'xlabel':'candidate (same stable IDs and denominator)','ylabel':'fixture pass fraction','series':[{'label':'observed pass fraction','y':[before['fraction'],after['fraction']]}]}

# Experiment: Hold the model and prompt fixed while removing one retrieval result
retrieval_only_config=make_candidate('candidate-drop-one','prompt-v2-grounded','ABSTAIN_WHEN_EMPTY',
    'index-v2-complete',corpus,case_excluded={'refund-order-204':('doc-refund',)})
assert retrieval_only_config['prompt_id']==candidate['prompt_id']
assert retrieval_only_config['model_id']==candidate['model_id']
assert retrieval_only_config['index_id']==candidate['index_id']
assert retrieval_only_config['config_hash']!=candidate['config_hash']
retrieval_only_responses=[]; retrieval_only_traces=[]
for fixed_case in cases:
    response,trace=run_case(fixed_case,retrieval_only_config)
    retrieval_only_responses.append(response);retrieval_only_traces.append(json.loads(trace))
retrieval_only_score=score(cases,retrieval_only_responses)
assert retrieval_only_score['count']==5 and retrieval_only_score['passed']==4
failed_trace=next(t for t in retrieval_only_traces if t['case_id']=='refund-order-204')
assert failed_trace['diagnosis']=='retrieval' and failed_trace['retrieval']['context'] is None
assert sum(not outcome['passed'] for outcome in retrieval_only_score['cases'])==1
print('Retrieval-only change:',retrieval_only_score['fraction'],failed_trace['diagnosis'],
      'config hash changed:',retrieval_only_config['config_hash']!=candidate['config_hash'])


# Experiment: Reject an incomplete case inventory before reporting a score
expected_ids=[case['id'] for case in cases]
assert len(collect_by_id(expected_ids,list(reversed(responses_after))))==5
incomplete_baseline=[r for r in responses_before if r['id']!='unsupported-tax-2']
naive_returned_only_fraction=before['passed']/len(incomplete_baseline)
assert naive_returned_only_fraction==.25 and before['fraction']==.2
try:collect_by_id(expected_ids,incomplete_baseline)
except ValueError as error:assert 'unsupported-tax-2' in str(error)
else:raise AssertionError('missing unsupported case silently removed from the denominator')
print('Incomplete run rejected before scoring; misleading returned-only ratio would be',naive_returned_only_fraction)


# Reference solution. Try the exercise before reading this.
trace={t['case_id']:t for t in (json.loads(x) for x in trace_before)}['case-sensitive-refund-77']
assert trace['retrieval']['retrieved_id']=='doc-refund'
assert trace['generation']['model_input_context']==cases[1]['expected']
assert trace['generation']['response'].startswith('Answer: ')
assert trace['diagnosis']=='generation'
# candidate uses all five same IDs; compare only prompt separately by retaining baseline index.
prompt_only=make_candidate('candidate-prompt-only','prompt-v2-grounded','ABSTAIN_WHEN_EMPTY',baseline['index_id'],baseline['index'],excluded=baseline['excluded_docs'],timeouts=baseline['timeouts'],case_excluded=baseline['case_excluded'])
rs=[run_case(case,prompt_only)[0] for case in cases]
paired=score(cases,rs)
assert paired['count']==5 and paired['fraction']==.6
print('Prompt-only score with baseline retrieval and infrastructure settings:',paired['passed'],'/',paired['count'])

# Reference practice: Check abstention structure
unsupported=next(case for case in cases if not case['supported'])
replies=[run_case(case,candidate)[0] for case in cases]
for reply in replies:
    if reply['id']==unsupported['id']:
        reply['response']='Tax is 15 percent.'
result=score(cases,replies)
failed=next(item for item in result['cases'] if item['id']==unsupported['id'])
assert result['count']==5 and not failed['passed']
print('malformed abstention retained and failed:',failed)

# Reference practice: Keep failed work in the evaluation inventory
replies=[run_case(case,candidate)[0] for case in cases]
replies[0]['response']=''
result=score(cases,replies)
assert result['count']==5 and result['passed']==4
assert result['cases'][0]['id']=='hours-west-17' and not result['cases'][0]['passed']
print('failed answer remains in denominator:',result)
print("PASS: operations-07")
