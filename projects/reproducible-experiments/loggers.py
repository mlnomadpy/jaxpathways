"""Same post-update observations to local CSV and optional real tracker backends."""
import csv
import json
from pathlib import Path
import time

COLUMNS = ['step', 'train_mse', 'validation_mse', 'gradient_norm', 'update_ms']


class Loggers:
    def __init__(self, directory, config, identity):
        self.directory = Path(directory)
        self.handle = (self.directory/'metrics.csv').open('x', newline='')
        self.csv = csv.DictWriter(self.handle, fieldnames=COLUMNS)
        self.csv.writeheader()
        self.tb = self.wb = self.ml = None
        self.receipts = {'csv': {'path': 'metrics.csv'}}
        try:
            if 'tensorboard' in config['loggers']:
                from tensorboard.summary.writer.event_file_writer import EventFileWriter
                self.tb = EventFileWriter(str(self.directory/'tensorboard'))
                self.receipts['tensorboard'] = {'path': 'tensorboard'}
            if 'wandb' in config['loggers']:
                import os
                for key in ['WANDB_CACHE_DIR', 'WANDB_CONFIG_DIR', 'WANDB_DATA_DIR']:
                    os.environ[key] = str(self.directory/key.lower())
                import wandb
                # Explicit offline mode prevents account requirements and uploads.
                self.wb = wandb.init(project='jax-reproducible-workshop', name=identity['run_id'],
                                     dir=str(self.directory), config={**config, **identity}, mode='offline')
                self.receipts['wandb'] = {'id': self.wb.id, 'mode': 'offline', 'path': str(Path(self.wb.dir).relative_to(self.directory))}
            if 'mlflow' in config['loggers']:
                import os
                os.environ['MLFLOW_DISABLE_AGENT_HINT'] = '1'
                import mlflow
                self.ml = mlflow
                mlflow.set_tracking_uri('sqlite:///' + str((self.directory/'mlflow.db').resolve()))
                experiment_id = mlflow.create_experiment('jax-reproducible-workshop',
                    artifact_location=(self.directory/'mlflow-artifacts').resolve().as_uri())
                mlflow.set_experiment(experiment_id=experiment_id)
                active = mlflow.start_run(run_name=identity['run_id'])
                mlflow.log_params({k: json.dumps(v) if isinstance(v, list) else v for k, v in config.items()})
                mlflow.set_tags(identity)
                self.receipts['mlflow'] = {'id': active.info.run_id, 'database': 'mlflow.db'}
        except Exception:
            self.close(success=False)
            raise

    def log(self, row):
        self.csv.writerow(row)
        self.handle.flush()
        metrics = {key: value for key, value in row.items() if key != 'step' and value is not None}
        if self.tb:
            from tensorboard.compat.proto import event_pb2, summary_pb2
            summary = summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=k, simple_value=v) for k, v in metrics.items()])
            self.tb.add_event(event_pb2.Event(wall_time=time.time(), step=row['step'], summary=summary))
            self.tb.flush()
        if self.wb:
            self.wb.log({'completed_step': row['step'], **metrics}, step=row['step'])
        if self.ml:
            self.ml.log_metrics(metrics, step=row['step'])

    def close(self, success=True):
        if self.tb:
            self.tb.close()
        if self.wb:
            self.wb.finish(exit_code=0 if success else 1)
        if self.ml:
            if success:
                for name in ['config.json', 'environment.json', 'metrics.csv', 'checkpoint.npz']:
                    if (self.directory/name).exists(): self.ml.log_artifact(str(self.directory/name))
            self.ml.end_run(status='FINISHED' if success else 'FAILED')
        if not self.handle.closed: self.handle.close()
