"""Run real training with all optional local adapters; independently read back history."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        plan = json.loads((ROOT/'configs/workshop.json').read_text())
        plan['seeds'] = [3, 7]
        plan['common'].update(steps=3, log_every=2, width=3, loggers=['csv', 'tensorboard', 'wandb', 'mlflow'])
        config = root/'plan.json'
        config.write_text(json.dumps(plan))
        suite = root/'suite'
        for args in [('plan', '--config', str(config), '--suite', str(suite)), ('job', '--suite', str(suite), '--run-id', 'baseline-s3', '--timeout', '120')]:
            result = subprocess.run([sys.executable, str(ROOT/'experiment.py'), *args], capture_output=True, text=True, timeout=180)
            assert result.returncode == 0, result.stdout + result.stderr + '\n' + '\n'.join(p.read_text()[-3500:] for p in suite.rglob('stderr.txt'))
        folder = suite/'runs/baseline-s3/attempt-01'
        with (folder/'metrics.csv').open() as handle: rows = list(csv.DictReader(handle))
        wanted = [(int(r['step']), float(r['validation_mse'])) for r in rows]
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = EventAccumulator(str(folder/'tensorboard')).Reload()
        observed = [(e.step, e.value) for e in events.Scalars('validation_mse')]
        assert len(observed) == len(wanted)
        for a, b in zip(observed, wanted): assert a[0] == b[0] and abs(a[1] - b[1]) < 1e-6
        summary = json.loads((folder/'summary.json').read_text())
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri='sqlite:///' + str(folder/'mlflow.db'))
        metrics = client.get_metric_history(summary['loggers']['mlflow']['id'], 'validation_mse')
        assert [(e.step, e.value) for e in metrics] == wanted
        artifacts = {a.path for a in client.list_artifacts(summary['loggers']['mlflow']['id'])}
        assert {'checkpoint.npz', 'config.json', 'metrics.csv', 'environment.json'} <= artifacts
        # Decode the local W&B event stream using the pinned SDK's internal reader.
        # This is a verifier, not an application dependency on private SDK APIs.
        from wandb.sdk.internal.datastore import DataStore
        from wandb.proto import wandb_internal_pb2
        files = list((folder/'wandb').rglob('*.wandb'))
        assert len(files) == 1 and summary['loggers']['wandb']['mode'] == 'offline'
        store = DataStore()
        store.open_for_scan(str(files[0]))
        logged = []
        while True:
            data = store.scan_data()
            if data is None: break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.HasField('history'):
                row = {(item.key or '.'.join(item.nested_key)): json.loads(item.value_json) for item in record.history.item}
                if 'validation_mse' in row: logged.append((row['completed_step'], row['validation_mse']))
        store.close()
        assert logged == wanted, (logged, wanted)
        assert not list(ROOT.glob('mlruns*')), 'tracker wrote outside the run directory'
    print('PASS: actual Flax training history read back from CSV, TensorBoard events, offline W&B events and SQLite MLflow, including model artifacts')


if __name__ == '__main__':
    main()
