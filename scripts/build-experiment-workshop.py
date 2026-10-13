"""Verify recorded workshop evidence before it enters the published project bundle."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'projects/reproducible-experiments'
SUITE = ROOT / 'reference/suite'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = json.loads((SUITE/'manifest.json').read_text())
plan = json.loads((ROOT/'configs/workshop.json').read_text())
report = json.loads((SUITE/'report/comparison.json').read_text())
provenance = json.loads((ROOT/'reference/provenance.json').read_text())
assert report['complete'] and report['backend'] == 'cpu' and len(report['runs']) == 6
assert manifest['seeds'] == plan['seeds'] and manifest['hypothesis'] == plan['hypothesis'] and manifest['decision_rule'] == plan['decision_rule']
assert sha(SUITE/'data.npz') == manifest['data_sha256']
for name, expected in manifest['source_files'].items():
    current = ROOT / ('solution/workflow.py' if name == 'workflow.py' else name)
    assert sha(current) == expected == sha(SUITE/'source'/name), 'Workshop CPU reference is stale: '+name
for relative, expected in provenance['sha256'].items():
    assert sha(ROOT/relative) == expected, 'Workshop reference provenance changed: '+relative
for run in report['runs']:
    run_id = run['run_id']
    cfg = json.loads((SUITE/'configs'/(run_id+'.json')).read_text())
    expected = {**plan['common'], 'name': cfg['name'], 'seed': cfg['seed'], 'learning_rate': plan['learning_rates'][cfg['name']]}
    assert cfg == expected and sha(SUITE/'configs'/(run_id+'.json')) == manifest['config_files'][run_id]
    job = json.loads((SUITE/'runs'/run_id/'job.json').read_text())
    attempt = job['attempts'][-1]
    folder = SUITE/'runs'/run_id/f"attempt-{attempt['attempt']:02d}"
    summary = json.loads((folder/'summary.json').read_text())
    assert sha(folder/'summary.json') == attempt['summary_sha256']
    assert summary['plan_sha256'] == sha(SUITE/'manifest.json')
    assert json.loads((folder/'environment.json').read_text())['backend'] == 'cpu'
    for name, key in [('checkpoint.npz','checkpoint_sha256'), ('metrics.csv','metrics_sha256'), ('environment.json','environment_sha256')]:
        assert sha(folder/name) == summary[key]
    with (folder/'metrics.csv').open() as stream:
        rows = [{key: int(value) if key == 'step' else float(value) if value else None for key, value in row.items()} for row in csv.DictReader(stream)]
    assert rows == run['history'] and rows[-1]['validation_mse'] == summary['final_validation_mse']
print('Verified workshop: six recorded CPU runs, frozen source/config/data, metrics, artifacts and figure provenance; optional adapters checked separately.')
