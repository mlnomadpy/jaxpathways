"""LLMOps: version prompts, evaluate behavior and trace requests: worked experiments and reference solutions. CPU checks."""

# Prepare the explicit contract
import os
os.environ["MLFLOW_DISABLE_AGENT_HINT"]="1"
os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"]="false"
os.environ["MLFLOW_ENABLE_TELEMETRY"]="false"
import tempfile
from pathlib import Path
import mlflow
from mlflow import MlflowClient

# Run and inspect the controlled experiment
def llm_case(case, answer, allowed_citations):
    """Deterministic replay evaluator, not an LLM judge or a general safety classifier."""
    if set(answer) != {'answer', 'citations', 'abstain'} or not isinstance(answer['answer'], str):
        return False
    if not isinstance(answer['citations'], list) or any(not isinstance(c, str) for c in answer['citations']):
        return False
    if type(answer['abstain']) is not bool or not set(answer['citations']) <= set(allowed_citations):
        return False
    if case['unanswerable']:
        return answer['abstain'] and not answer['citations'] and answer['answer'] == ''
    return not answer['abstain'] and answer['answer'] == case['expected'] and case['document'] in answer['citations']

cases = [dict(id=f'answerable-{i}', unanswerable=False, expected='CPU', document='course-v1') for i in range(9)]
cases += [dict(id='unsupported', unanswerable=True)]
answer = dict(answer='CPU', citations=['course-v1'], abstain=False)
unsafe = [dict(answer) for _ in cases]
guarded = [dict(answer) for _ in cases[:-1]] + [dict(answer='', citations=[], abstain=True)]
results = []
with tempfile.TemporaryDirectory(prefix='llmops-') as folder:
    mlflow.set_tracking_uri('sqlite:///' + str(Path(folder) / 'traces.db'))
    client = MlflowClient(); experiment = client.create_experiment('replay-evaluation', artifact_location=(Path(folder)/'artifacts').as_uri())
    mlflow.set_experiment(experiment_id=experiment)
    for name,replies in [('always-answer',unsafe),('contract-aware',guarded)]:
        with mlflow.start_span(name='evaluate-replay', span_type='CHAIN') as span:
            span.set_inputs({'candidate': name, 'dataset_version': 'public-fixture-v1', 'case_count': len(cases)})
            passed = [llm_case(case, reply, ['course-v1']) for case,reply in zip(cases,replies)]
            summary = {'overall': sum(passed)/len(passed), 'answerable': sum(passed[:-1])/9, 'unanswerable': float(passed[-1])}
            span.set_outputs(summary); results.append(summary)
    traces = mlflow.search_traces(experiment_ids=[experiment], return_type='list')
    assert len(traces) == 2
assert results[0] == {'overall': .9, 'answerable': 1., 'unanswerable': 0.}
assert results[1] == {'overall': 1., 'answerable': 1., 'unanswerable': 1.}
assert not llm_case(cases[0],dict(answer='CPU',citations=['invented'],abstain=False),['course-v1'])
print('Observed replay scores:', results)
print('Two real MLflow traces; no LLM API was called and no general model quality was evaluated.')

import os
os.environ["MLFLOW_DISABLE_AGENT_HINT"]="1"
os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"]="false"
os.environ["MLFLOW_ENABLE_TELEMETRY"]="false"
import tempfile
from pathlib import Path
import mlflow
from mlflow import MlflowClient

def llm_case(case, answer, allowed_citations):
    """Deterministic replay evaluator, not an LLM judge or a general safety classifier."""
    if set(answer) != {'answer', 'citations', 'abstain'} or not isinstance(answer['answer'], str):
        return False
    if not isinstance(answer['citations'], list) or any(not isinstance(c, str) for c in answer['citations']):
        return False
    if type(answer['abstain']) is not bool or not set(answer['citations']) <= set(allowed_citations):
        return False
    if case['unanswerable']:
        return answer['abstain'] and not answer['citations'] and answer['answer'] == ''
    return not answer['abstain'] and answer['answer'] == case['expected'] and case['document'] in answer['citations']

cases = [dict(id=f'answerable-{i}', unanswerable=False, expected='CPU', document='course-v1') for i in range(9)]
cases += [dict(id='unsupported', unanswerable=True)]
answer = dict(answer='CPU', citations=['course-v1'], abstain=False)
unsafe = [dict(answer) for _ in cases]
guarded = [dict(answer) for _ in cases[:-1]] + [dict(answer='', citations=[], abstain=True)]
results = []
with tempfile.TemporaryDirectory(prefix='llmops-') as folder:
    mlflow.set_tracking_uri('sqlite:///' + str(Path(folder) / 'traces.db'))
    client = MlflowClient(); experiment = client.create_experiment('replay-evaluation', artifact_location=(Path(folder)/'artifacts').as_uri())
    mlflow.set_experiment(experiment_id=experiment)
    for name,replies in [('always-answer',unsafe),('contract-aware',guarded)]:
        with mlflow.start_span(name='evaluate-replay', span_type='CHAIN') as span:
            span.set_inputs({'candidate': name, 'dataset_version': 'public-fixture-v1', 'case_count': len(cases)})
            passed = [llm_case(case, reply, ['course-v1']) for case,reply in zip(cases,replies)]
            summary = {'overall': sum(passed)/len(passed), 'answerable': sum(passed[:-1])/9, 'unanswerable': float(passed[-1])}
            span.set_outputs(summary); results.append(summary)
    traces = mlflow.search_traces(experiment_ids=[experiment], return_type='list')
    assert len(traces) == 2
assert results[0] == {'overall': .9, 'answerable': 1., 'unanswerable': 0.}
assert results[1] == {'overall': 1., 'answerable': 1., 'unanswerable': 1.}
assert not llm_case(cases[0],dict(answer='CPU',citations=['invented'],abstain=False),['course-v1'])
print('Observed replay scores:', results)
print('Two real MLflow traces; no LLM API was called and no general model quality was evaluated.')


# Figure data experiment
visual_data={'kind':'bar','labels':['overall (10)','answerable (9)','unsupported (1)'],'xlabel':'fixture group (count)','ylabel':'fixture pass fraction','series':[{'label':name,'y':[r[k] for k in ['overall','answerable','unanswerable']]} for name,r in zip(['always-answer','contract-aware'],results)]}

# Experiment: Separate aggregate and critical-case behavior
release=[r['overall']>=.9 and r['unanswerable']==1. for r in results]
assert release==[False,True]
print('Release decisions:',release)

# Experiment: Reject an incomplete replay before reporting a score
def score_replay(inventory, replies, allowed_citations):
    ids = [case['id'] for case in inventory]
    reply_ids = [reply['id'] for reply in replies]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('empty or duplicate case inventory')
    if len(set(reply_ids)) != len(reply_ids) or set(reply_ids) != set(ids):
        raise ValueError('missing, duplicate or unknown response IDs')
    by_id = {reply['id']: reply['response'] for reply in replies}
    passed = [llm_case(case, by_id[case['id']], allowed_citations) for case in inventory]
    return {'count': len(ids), 'passed': sum(passed), 'fraction': sum(passed) / len(ids)}

complete_replies = [{'id': case['id'], 'response': reply} for case, reply in zip(cases, unsafe)]
replay_summary = score_replay(cases, list(reversed(complete_replies)), ['course-v1'])
assert replay_summary == {'count': 10, 'passed': 9, 'fraction': .9}
for broken in (complete_replies[:-1], complete_replies + [complete_replies[0]],
               complete_replies[:-1] + [dict(id='unknown', response=answer)]):
    try: score_replay(cases, broken, ['course-v1'])
    except ValueError: pass
    else: raise AssertionError('incomplete or ambiguous evaluation accepted')
print('Complete inventory:', replay_summary, '; missing, duplicate and unknown responses rejected.')

# Reference solution. Try the exercise before reading this.
for citation in ['invented','old-index-v0']:
    assert not llm_case(cases[0],dict(answer='CPU',citations=[citation],abstain=False),['course-v1'])
print('Unrecognized citations rejected.')

# Reference practice: Check abstention structure
bad=dict(answer='An unsupported claim',citations=[],abstain=True)
assert not llm_case(cases[-1],bad,['course-v1'])
print('An abstention flag alone does not satisfy the output contract.')

# Reference practice: Keep failed work in the evaluation inventory
timeout_replies = [{'id': case['id'], 'response': dict(reply)} for case, reply in zip(cases, guarded)]
timeout_replies[0]['response'] = dict(answer='', citations=[], abstain=True)
timeout_summary = score_replay(cases, timeout_replies, ['course-v1'])
assert timeout_summary == {'count': 10, 'passed': 9, 'fraction': .9}
print('Failed answerable work remains in denominator:', timeout_summary)
print("PASS: operations-07")
