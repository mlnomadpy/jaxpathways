"""Validate complete evidence before comparing runs; preserve incomplete inventories."""
import csv
import json
from pathlib import Path
import statistics

from experiment import digest, load_module, verify_suite, write_json


def history(path):
    with Path(path).open(newline='') as stream:
        return [{key: int(value) if key == 'step' else float(value) if value else None for key, value in row.items()}
                for row in csv.DictReader(stream)]


def render_figures(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors = {'baseline': '#315bb5', 'candidate': '#b34d21'}
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7), layout='constrained')
    for axis, metric, title in zip(axes, ('train_mse', 'validation_mse'), ('Training split: 64 examples', 'Fixed validation split: 32 examples')):
        for name, color in colors.items():
            selected = [r for r in report['runs'] if r['config']['name'] == name]
            x = [r['step'] for r in selected[0]['history']]
            ys = np.array([[r[metric] for r in run['history']] for run in selected])
            axis.fill_between(x, ys.min(0), ys.max(0), color=color, alpha=.14)
            axis.plot(x, ys.mean(0), color=color, label=name+' mean')
        axis.set(title=title, xlabel='Completed updates (zero is initialization)', ylabel='Mean squared error')
        axis.grid(alpha=.2)
        axis.legend()
    fig.suptitle('Recorded CPU reference' if report['backend'] == 'cpu' else 'Recorded '+report['backend']+' run')
    fig.savefig(output/'loss_curves.png', dpi=160)
    plt.close(fig)
    pairs = report['paired']
    fig, axis = plt.subplots(figsize=(6.5, 3.8), layout='constrained')
    axis.axhline(0, color='#536074', linewidth=1)
    axis.scatter([str(s) for s in pairs['seeds']], pairs['differences'], color='#315bb5', s=55)
    axis.axhline(pairs['mean_difference'], color='#b34d21', linestyle='--', label='Paired mean')
    axis.set(xlabel='Initialization seed (same seed within each pair)', ylabel='Candidate minus baseline MSE',
             title='Below zero favors the candidate on this split')
    axis.grid(axis='y', alpha=.2)
    axis.legend()
    fig.savefig(output/'paired.png', dpi=160)
    plt.close(fig)


def analyze(suite, output):
    manifest = verify_suite(suite)
    workflow = load_module(suite/'source/workflow.py')
    output.mkdir(parents=True, exist_ok=False)
    records = []
    inventory = []
    for run_id in manifest['config_files']:
        root = suite/'runs'/run_id
        cfg = json.loads((suite/'configs'/(run_id+'.json')).read_text())
        item = {'run_id': run_id, 'status': 'missing', 'attempts': 0}
        try:
            if not (root/'job.json').exists():
                inventory.append(item)
                continue
            job = json.loads((root/'job.json').read_text())
            item.update(status=job['status'], attempts=len(job['attempts']))
            if job['status'] != 'completed':
                inventory.append(item)
                continue
            attempt = job['attempts'][-1]
            directory = root/f"attempt-{attempt['attempt']:02d}"
            if attempt['summary_sha256'] != digest(directory/'summary.json'): raise ValueError('summary changed')
            summary = json.loads((directory/'summary.json').read_text())
            if summary['status'] != 'completed' or summary['plan_sha256'] != digest(suite/'manifest.json'):
                raise ValueError('plan changed or incomplete worker')
            if summary['data_sha256'] != manifest['data_sha256'] or summary['config_sha256'] != manifest['config_files'][run_id]:
                raise ValueError('run belongs to another dataset or configuration')
            if summary['metric_contract'] != manifest['metric_contract'] or summary['run_id'] != run_id:
                raise ValueError('metric contract or run identity changed')
            for file, key in [('metrics.csv', 'metrics_sha256'), ('checkpoint.npz', 'checkpoint_sha256'), ('environment.json', 'environment_sha256')]:
                if digest(directory/file) != summary[key]: raise ValueError(file+' changed')
            if json.loads((directory/'config.json').read_text()) != cfg: raise ValueError('worker used another configuration')
            rows = workflow.validate_history(history(directory/'metrics.csv'), cfg)
            if rows[-1]['validation_mse'] != summary['final_validation_mse']: raise ValueError('final metric mismatch')
            environment = json.loads((directory/'environment.json').read_text())
            if environment['backend'] != cfg['backend']: raise ValueError('backend mismatch')
            records.append({'run_id': run_id, 'config': cfg, 'environment': environment, 'history': rows,
                            'attempts': item['attempts'], 'evidence': str(directory), 'loggers': summary['loggers']})
        except (ValueError, KeyError, OSError) as error:
            item.update(status='invalid', error=str(error))
        inventory.append(item)
    complete = len(records) == len(manifest['config_files']) and all(r['status'] == 'completed' for r in inventory)
    report = {'schema': 1, 'complete': complete, 'plan': manifest, 'inventory': inventory, 'runs': records,
              'scope': 'full-batch synthetic regression; validation comparison, not a final test result'}
    if complete:
        contracts = []
        for r in records:
            cfg = {k: v for k, v in r['config'].items() if k not in ('name', 'seed', 'learning_rate')}
            env = {k: r['environment'][k] for k in ('python', 'jax', 'jaxlib', 'flax', 'optax', 'numpy', 'logger_versions', 'backend', 'device_kind', 'process_count')}
            contracts.append(json.dumps([cfg, env], sort_keys=True))
        if len(set(contracts)) != 1: raise ValueError('incompatible model, step, metric, environment or device contracts')
        report['backend'] = records[0]['environment']['backend']
        report['paired'] = workflow.paired_summary([{'name': r['config']['name'], 'seed': r['config']['seed'],
                                                    'validation_mse': r['history'][-1]['validation_mse']} for r in records], manifest['seeds'])
        render_figures(report, output)
    write_json(output/'comparison.json', report)
    with (output/'comparison.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['run_id', 'status', 'attempts', 'validation_mse'])
        writer.writeheader()
        for row in inventory:
            matched = next((r for r in records if r['run_id'] == row['run_id']), None)
            writer.writerow({k: row[k] for k in ('run_id', 'status', 'attempts')} | {'validation_mse': matched['history'][-1]['validation_mse'] if matched else ''})
    lines = ['# Research notebook: reproducible JAX/Flax experiments', '', '## Question and declared decision rule', '',
             manifest['hypothesis'], '', manifest['decision_rule'], '', '## Execution inventory', '',
             '| Run | Status | Attempts |', '| --- | --- | ---: |']
    lines += [f"| {r['run_id']} | {r['status']} | {r['attempts']} |" for r in inventory]
    lines += ['', '## Provenance', '', f"Plan SHA-256: `{digest(suite/'manifest.json')}`.",
              f"Data SHA-256: `{manifest['data_sha256']}`.", '',
              'The saved plan, source/, configs/, run environments, metrics and checkpoints identify this comparison.',
              'Full source/config/data checksums detect accidental changes; they are not authenticated attestations.', '',
              'Machine-readable rows and the complete observation histories are in [comparison.json](comparison.json).',
              'Portable inventory: [comparison.csv](comparison.csv).', '']
    if complete:
        paired = report['paired']
        lines += ['## Results', '', '![Training and validation MSE versus completed updates; lines are seed means and bands are seed ranges.](loss_curves.png)', '',
                  'Read each panel separately. Zero is the untrained model; every later observation follows its update. '
                  'The shaded range spans the smallest and largest seed result at each step, not a confidence interval.', '',
                  '![Paired candidate-minus-baseline final validation MSE for each initialization seed.](paired.png)', '',
                  'Each point compares the same seed on the same validation examples. Below zero favors the candidate; '
                  'the dashed line is the paired mean. Seeds vary initialization only, not the fixed dataset.', '',
                  f"Paired mean difference: {paired['mean_difference']:.8g}; sample SD: {paired['sample_sd_difference']:.8g}; pairs: {paired['count']}.",
                  f"Constant predictor validation MSE: {manifest['constant_validation_mse']:.8g}.", '',
                  'These are descriptive validation results on a synthetic fixture, not evidence of a population-wide advantage.', '',
                  '## Evidence links', '']
        for r in records:
            # Relative links preserve usability when the whole suite is moved.
            import os
            link = Path(os.path.relpath(Path(r['evidence']), output)).as_posix()
            lines += [f"- [{r['run_id']}: metrics]({link}/metrics.csv), [configuration]({link}/config.json), [environment]({link}/environment.json), [checkpoint]({link}/checkpoint.npz)."]
        lines += ['', '## Your interpretation', '',
                  'Write your decision against the declared rule. Explain one seed pair and one visible curve feature. '
                  'Separate what you observed from the proposed cause. Record the next changed-condition experiment before running it.', '',
                  'No numerical resume was tested here. Retried jobs start from initialization; use the linked recovery lessons for continuation.',
                  'Logging is outside the compiled step. Update timings include the update and post-update training loss, '
                  'while training wall time additionally includes validation and logging. Neither measures distributed scaling.']
    else:
        lines += ['## Comparison withheld', '', 'At least one planned run is missing, failed or invalid. No paired mean or winning configuration is reported.',
                  'Inspect the inventory and raw attempts. Launch missing jobs individually, then analyze into a new output directory.']
    (output/'RESEARCH_NOTEBOOK.md').write_text('\n'.join(lines)+'\n')
    print('Saved', output/'RESEARCH_NOTEBOOK.md', 'complete='+str(complete))
    return 0 if complete else 1
