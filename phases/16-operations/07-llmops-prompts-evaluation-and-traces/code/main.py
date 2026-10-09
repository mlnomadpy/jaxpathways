"""LLMOps: version prompts, evaluate behavior and trace requests: worked experiments and reference solutions. CPU checks."""

# Prepare the explicit contract
# Step 1 — Prepare the explicit contract: For a text generator, record the model revision, tokenizer, prompt...
# Import os for this computation.
import os
# Configure environment variable before initializing the runtime.
os.environ["MLFLOW_DISABLE_AGENT_HINT"]="1"
# Configure environment variable before initializing the runtime.
os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"]="false"
# Configure environment variable before initializing the runtime.
os.environ["MLFLOW_ENABLE_TELEMETRY"]="false"
# Import required JAX, NumPy, and standard-library modules.
import tempfile
from pathlib import Path
import mlflow
from mlflow import MlflowClient

# Run and inspect the controlled experiment
# Step 2 — Run and inspect the controlled experiment: A trace connects one request’s spans: retrieval, model invocation,...
def llm_case(case, answer, allowed_citations):
    """Deterministic replay evaluator, not an LLM judge or a general safety classifier."""
    # Branch on condition `set(answer) != {'answer', 'citations', 'abstain'} or not isinstance(answer['answer'], str)`:
    if set(answer) != {'answer', 'citations', 'abstain'} or not isinstance(answer['answer'], str):
        return False
    # Branch on condition `not isinstance(answer['citations'], list) or any((not isinstance(c, str) for c in answer['citations']))`:
    if not isinstance(answer['citations'], list) or any(not isinstance(c, str) for c in answer['citations']):
        return False
    # Branch on condition `type(answer['abstain']) is not bool or not set(answer['citations']) <= set(allowed_citations)`:
    if type(answer['abstain']) is not bool or not set(answer['citations']) <= set(allowed_citations):
        return False
    # Branch on condition `case['unanswerable']`:
    if case['unanswerable']:
        return answer['abstain'] and not answer['citations'] and answer['answer'] == ''
    # Return `not answer['abstain'] and answer['answer'] == case['expected'] and (case['document'] in answer['citations'])` to the caller.
    return not answer['abstain'] and answer['answer'] == case['expected'] and case['document'] in answer['citations']

# Compute `cases` from `[dict(id=f'answerable-{i}', unanswerable=False, expe...`
cases = [dict(id=f'answerable-{i}', unanswerable=False, expected='CPU', document='course-v1') for i in range(9)]
# Accumulate the next contribution into `cases`.
cases += [dict(id='unsupported', unanswerable=True)]
# Evaluate `answer='CPU', citations=['course-v1'], abstain=False` and convert the result into Python scalar/collection `answer`.
answer = dict(answer='CPU', citations=['course-v1'], abstain=False)
# Compute `unsafe` from `[dict(answer) for _ in cases]`
unsafe = [dict(answer) for _ in cases]
# Compute `guarded` from `[dict(answer) for _ in cases[:-1]] + [dict(answer=''...`
guarded = [dict(answer) for _ in cases[:-1]] + [dict(answer='', citations=[], abstain=True)]
# Compute `results` from `[]`
results = []
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory(prefix='llmops-') as folder:
    # Execute `mlflow.set_tracking_uri('sqlite:///' + str(Path(folder) / 'traces.db'))`.
    mlflow.set_tracking_uri('sqlite:///' + str(Path(folder) / 'traces.db'))
    # Run `MlflowClient` to compute `client`.
    # Read or serialize artifact data on disk (`experiment`).
    client = MlflowClient()
    experiment = client.create_experiment('replay-evaluation', artifact_location=(Path(folder)/'artifacts').as_uri())
    # Run `mlflow.set_experiment` to perform the next check or state transition.
    mlflow.set_experiment(experiment_id=experiment)
    # Loop over `(name, replies)` in `[('always-answer', unsafe), ('contract-aware', guarded)]`:
    for name,replies in [('always-answer',unsafe),('contract-aware',guarded)]:
        # Enter managed runtime/context scope for this block:
        with mlflow.start_span(name='evaluate-replay', span_type='CHAIN') as span:
            # Run `span.set_inputs` to perform the next check or state transition.
            span.set_inputs({'candidate': name, 'dataset_version': 'public-fixture-v1', 'case_count': len(cases)})
            # Compute `passed` from `[llm_case(case, reply, ['course-v1']) for case,reply...`
            passed = [llm_case(case, reply, ['course-v1']) for case,reply in zip(cases,replies)]
            # Compute `summary` from `{'overall': sum(passed)/len(passed), 'answerable': s...`
            summary = {'overall': sum(passed)/len(passed), 'answerable': sum(passed[:-1])/9, 'unanswerable': float(passed[-1])}
            # Run `span.set_outputs` to perform the next check or state transition.
            # Run `span.set_outputs` to perform the next check or state transition.
            span.set_outputs(summary)
            results.append(summary)
    # Run `mlflow.search_traces` to compute `traces`.
    traces = mlflow.search_traces(experiment_ids=[experiment], return_type='list')
    # Assert invariant `len(traces) == 2` holds
    assert len(traces) == 2
# Assert invariant `results[0] == {'overall': .9` holds
assert results[0] == {'overall': .9, 'answerable': 1., 'unanswerable': 0.}
# Assert invariant `results[1] == {'overall': 1.` holds
assert results[1] == {'overall': 1., 'answerable': 1., 'unanswerable': 1.}
# Assert invariant `not llm_case(cases[0]` holds
assert not llm_case(cases[0],dict(answer='CPU',citations=['invented'],abstain=False),['course-v1'])
# Print the observed values to compare against the expected result.
print('Observed replay scores:', results)
# Print diagnostic summary of the computed outputs.
print('Two real MLflow traces; no LLM API was called and no general model quality was evaluated.')

# Step 3: Verify invariants on the completed state
assert results[1] == {'overall': 1., 'answerable': 1., 'unanswerable': 1.}
assert not llm_case(cases[0],dict(answer='CPU',citations=['invented'],abstain=False),['course-v1'])

# Step 1 — Prepare the explicit contract: For a text generator, record the model revision, tokenizer, prompt...
# Import os for this computation.
import os
# Configure environment variable before initializing the runtime.
os.environ["MLFLOW_DISABLE_AGENT_HINT"]="1"
# Configure environment variable before initializing the runtime.
os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"]="false"
# Configure environment variable before initializing the runtime.
os.environ["MLFLOW_ENABLE_TELEMETRY"]="false"
# Import required JAX, NumPy, and standard-library modules.
import tempfile
from pathlib import Path
import mlflow
from mlflow import MlflowClient

# Step 2 — Run and inspect the controlled experiment: A trace connects one request’s spans: retrieval, model invocation,...
def llm_case(case, answer, allowed_citations):
    """Deterministic replay evaluator, not an LLM judge or a general safety classifier."""
    # Branch on condition `set(answer) != {'answer', 'citations', 'abstain'} or not isinstance(answer['answer'], str)`:
    if set(answer) != {'answer', 'citations', 'abstain'} or not isinstance(answer['answer'], str):
        return False
    # Branch on condition `not isinstance(answer['citations'], list) or any((not isinstance(c, str) for c in answer['citations']))`:
    if not isinstance(answer['citations'], list) or any(not isinstance(c, str) for c in answer['citations']):
        return False
    # Branch on condition `type(answer['abstain']) is not bool or not set(answer['citations']) <= set(allowed_citations)`:
    if type(answer['abstain']) is not bool or not set(answer['citations']) <= set(allowed_citations):
        return False
    # Branch on condition `case['unanswerable']`:
    if case['unanswerable']:
        return answer['abstain'] and not answer['citations'] and answer['answer'] == ''
    # Return `not answer['abstain'] and answer['answer'] == case['expected'] and (case['document'] in answer['citations'])` to the caller.
    return not answer['abstain'] and answer['answer'] == case['expected'] and case['document'] in answer['citations']

# Compute `cases` from `[dict(id=f'answerable-{i}', unanswerable=False, expe...`
cases = [dict(id=f'answerable-{i}', unanswerable=False, expected='CPU', document='course-v1') for i in range(9)]
# Accumulate the next contribution into `cases`.
cases += [dict(id='unsupported', unanswerable=True)]
# Evaluate `answer='CPU', citations=['course-v1'], abstain=False` and convert the result into Python scalar/collection `answer`.
answer = dict(answer='CPU', citations=['course-v1'], abstain=False)
# Compute `unsafe` from `[dict(answer) for _ in cases]`
unsafe = [dict(answer) for _ in cases]
# Compute `guarded` from `[dict(answer) for _ in cases[:-1]] + [dict(answer=''...`
guarded = [dict(answer) for _ in cases[:-1]] + [dict(answer='', citations=[], abstain=True)]
# Compute `results` from `[]`
results = []
# Create an isolated temporary directory to run and inspect artifacts safely:
with tempfile.TemporaryDirectory(prefix='llmops-') as folder:
    # Execute `mlflow.set_tracking_uri('sqlite:///' + str(Path(folder) / 'traces.db'))`.
    mlflow.set_tracking_uri('sqlite:///' + str(Path(folder) / 'traces.db'))
    # Run `MlflowClient` to compute `client`.
    # Read or serialize artifact data on disk (`experiment`).
    client = MlflowClient()
    experiment = client.create_experiment('replay-evaluation', artifact_location=(Path(folder)/'artifacts').as_uri())
    # Run `mlflow.set_experiment` to perform the next check or state transition.
    mlflow.set_experiment(experiment_id=experiment)
    # Loop over `(name, replies)` in `[('always-answer', unsafe), ('contract-aware', guarded)]`:
    for name,replies in [('always-answer',unsafe),('contract-aware',guarded)]:
        # Enter managed runtime/context scope for this block:
        with mlflow.start_span(name='evaluate-replay', span_type='CHAIN') as span:
            # Run `span.set_inputs` to perform the next check or state transition.
            span.set_inputs({'candidate': name, 'dataset_version': 'public-fixture-v1', 'case_count': len(cases)})
            # Compute `passed` from `[llm_case(case, reply, ['course-v1']) for case,reply...`
            passed = [llm_case(case, reply, ['course-v1']) for case,reply in zip(cases,replies)]
            # Compute `summary` from `{'overall': sum(passed)/len(passed), 'answerable': s...`
            summary = {'overall': sum(passed)/len(passed), 'answerable': sum(passed[:-1])/9, 'unanswerable': float(passed[-1])}
            # Run `span.set_outputs` to perform the next check or state transition.
            # Run `span.set_outputs` to perform the next check or state transition.
            span.set_outputs(summary)
            results.append(summary)
    # Run `mlflow.search_traces` to compute `traces`.
    traces = mlflow.search_traces(experiment_ids=[experiment], return_type='list')
    # Assert invariant `len(traces) == 2` holds
    assert len(traces) == 2
# Assert invariant `results[0] == {'overall': .9` holds
assert results[0] == {'overall': .9, 'answerable': 1., 'unanswerable': 0.}
# Assert invariant `results[1] == {'overall': 1.` holds
assert results[1] == {'overall': 1., 'answerable': 1., 'unanswerable': 1.}
# Assert invariant `not llm_case(cases[0]` holds
assert not llm_case(cases[0],dict(answer='CPU',citations=['invented'],abstain=False),['course-v1'])
# Print the observed values to compare against the expected result.
print('Observed replay scores:', results)
# Print diagnostic summary of the computed outputs.
print('Two real MLflow traces; no LLM API was called and no general model quality was evaluated.')

# Figure data experiment
# Compute figure data for: The unsupported case changes the release decision
# Compute `visual_data` from `{'kind':'bar','labels':['overall (10)','answerable (...`
visual_data={'kind':'bar','labels':['overall (10)','answerable (9)','unsupported (1)'],'xlabel':'fixture group (count)','ylabel':'fixture pass fraction','series':[{'label':name,'y':[r[k] for k in ['overall','answerable','unanswerable']]} for name,r in zip(['always-answer','contract-aware'],results)]}

# Experiment: Separate aggregate and critical-case behavior
# Experiment — Separate aggregate and critical-case behavior: The extra gate encodes a required behavior that the overall...
release=[r['overall']>=.9 and r['unanswerable']==1. for r in results]
# Assert invariant `release==[False,True]` holds
assert release==[False,True]
# Print the observed values to compare against the expected result.
print('Release decisions:',release)

# Experiment: Reject an incomplete replay before reporting a score
# Experiment — Reject an incomplete replay before reporting a score: The case inventory defines the denominator.
def score_replay(inventory, replies, allowed_citations):
    # Compute `ids` from `[case['id'] for case in inventory]`
    ids = [case['id'] for case in inventory]
    # Compute `reply_ids` from `[reply['id'] for reply in replies]`
    reply_ids = [reply['id'] for reply in replies]
    # Guard input contract (`not ids or len(set(ids)) != len(ids)`) and fail fast if violated.
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('empty or duplicate case inventory')
    # Guard input contract (`len(set(reply_ids)) != len(reply_ids) or set(reply_ids) != set(ids)`) and fail fast if violated.
    if len(set(reply_ids)) != len(reply_ids) or set(reply_ids) != set(ids):
        raise ValueError('missing, duplicate or unknown response IDs')
    # Compute `by_id` from `{reply['id']: reply['response'] for reply in replies}`
    by_id = {reply['id']: reply['response'] for reply in replies}
    # Compute `passed` from `[llm_case(case, by_id[case['id']], allowed_citations...`
    passed = [llm_case(case, by_id[case['id']], allowed_citations) for case in inventory]
    # Return `{'count': len(ids), 'passed': sum(passed), 'fraction': sum(passed) / len(ids)}` to the caller.
    return {'count': len(ids), 'passed': sum(passed), 'fraction': sum(passed) / len(ids)}

# Compute `complete_replies` from `[{'id': case['id'], 'response': reply} for case, rep...`
complete_replies = [{'id': case['id'], 'response': reply} for case, reply in zip(cases, unsafe)]
# Run `score_replay` to compute `replay_summary`.
replay_summary = score_replay(cases, list(reversed(complete_replies)), ['course-v1'])
# Assert invariant `replay_summary == {'count': 10` holds
assert replay_summary == {'count': 10, 'passed': 9, 'fraction': .9}
# Iterate over `broken` to step through the computation:
for broken in (complete_replies[:-1], complete_replies + [complete_replies[0]],
               complete_replies[:-1] + [dict(id='unknown', response=answer)]):
    try:
        score_replay(cases, broken, ['course-v1'])
    except ValueError:
        pass
    else:
        raise AssertionError('incomplete or ambiguous evaluation accepted')
# Print the observed values to compare against the expected result.
print('Complete inventory:', replay_summary, '; missing, duplicate and unknown responses rejected.')

# Reference solution. Try the exercise before reading this.
# Exercise solution: Add a response with an invented citation and require the evaluator to...
# Iterate over `citation` to step through the computation:
for citation in ['invented','old-index-v0']:
    # Assert invariant `not llm_case(cases[0]` holds
    assert not llm_case(cases[0],dict(answer='CPU',citations=[citation],abstain=False),['course-v1'])
# Print the observed values to compare against the expected result.
print('Unrecognized citations rejected.')

# Reference practice: Check abstention structure
# Check abstention structure (transfer): Validate the full response object; a boolean alone cannot...
bad=dict(answer='An unsupported claim',citations=[],abstain=True)
# Assert invariant `not llm_case(cases[-1]` holds
assert not llm_case(cases[-1],bad,['course-v1'])
# Print the observed values to compare against the expected result.
print('An abstention flag alone does not satisfy the output contract.')

# Reference practice: Keep failed work in the evaluation inventory
# Keep failed work in the evaluation inventory (Transfer): The fixture represents an unsuccessful answer as an empty...
timeout_replies = [{'id': case['id'], 'response': dict(reply)} for case, reply in zip(cases, guarded)]
# Evaluate `answer='', citations=[], abstain=True` and convert the result into Python scalar/collection `timeout_replies[0]['response']`.
timeout_replies[0]['response'] = dict(answer='', citations=[], abstain=True)
# Run `score_replay` to compute `timeout_summary`.
timeout_summary = score_replay(cases, timeout_replies, ['course-v1'])
# Assert invariant `timeout_summary == {'count': 10` holds
assert timeout_summary == {'count': 10, 'passed': 9, 'fraction': .9}
# Print the observed values to compare against the expected result.
print('Failed answerable work remains in denominator:', timeout_summary)
print("PASS: operations-07")
