import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

const root = process.cwd();

function runPython(id, extra = '', env = {}) {
  const source = `import glob, json\npath = next(p for p in glob.glob('phases/*tpu*/**/lesson.json', recursive=True) if json.load(open(p))['id'] == ${JSON.stringify(id)})\nlesson = json.load(open(path))\nnamespace = {}\nexec(compile(lesson['content']['code'], path, 'exec'), namespace)\n${extra}\n`;
  const result = spawnSync('python3', ['-c', source], {
    cwd: root,
    encoding: 'utf8',
    env: { ...process.env, ...env },
  });
  assert.equal(result.status, 0, `${id} CPU check failed:\n${result.stdout}\n${result.stderr}`);
  return result.stdout;
}

test('tpu-02 validates checkpoint integrity, compatibility, missing and completed resume', () => {
  const extra = `import json, tempfile\nfrom pathlib import Path\nrun = namespace['run_tpu_ready_job']\nwith tempfile.TemporaryDirectory() as td:\n    run_dir = Path(td) / 'run'\n    finished = run(run_dir, steps=2, save_every=2)\n    completed = run(run_dir, steps=2, save_every=2, resume=True)\n    assert completed['state_hash'] == finished['state_hash'] and completed['step_losses'] == []\n    try: run(Path(td) / 'missing', steps=1, resume=True)\n    except FileNotFoundError: pass\n    else: raise AssertionError('missing requested checkpoint was accepted')\n    try: run(run_dir, steps=3, resume=True, lr=0.01)\n    except ValueError as exc: assert 'configuration' in str(exc)\n    else: raise AssertionError('changed run configuration was accepted')\n    checkpoint = run_dir / 'checkpoint-latest.json'\n    saved = json.loads(checkpoint.read_text())\n    saved['w'][0][0] += 1\n    checkpoint.write_text(json.dumps(saved))\n    try: run(run_dir, steps=3, resume=True)\n    except ValueError as exc: assert 'hash' in str(exc)\n    else: raise AssertionError('modified checkpoint state was accepted')`;
  runPython('tpu-02', extra);
});

test('tpu-03 memory ledger matches explicit bytes per parameter', () => {
  const extra = `budget = namespace['estimate_memory_gib'](7.0, 8192)\nassert budget['fp32_train_state_gib'] == 104.308\nassert budget['mixed_bf16_train_state_gib'] == 117.347\nassert budget['kv_bf16_gib'] == 1.0`;
  runPython('tpu-03', extra);
  const file = path.join(root, 'phases/21-tpu-systems/01-tpu-generations-and-hbm-memory-budgeting/lesson.json');
  const lesson = JSON.parse(readFileSync(file, 'utf8'));
  assert.equal(lesson.content.visual.kind, 'analytic');
  assert.match(lesson.content.visual.connection, /not device measurements/i);
});

test('tpu-04 labels the four-device result as calculated expectation', () => {
  const file = path.join(root, 'phases/19-tpu/02-bootstrap-venv-and-verify-tpu-runtime-receipt/lesson.json');
  const lesson = JSON.parse(readFileSync(file, 'utf8'));
  assert.equal(lesson.content.visual.kind, 'analytic');
  assert.match(lesson.content.visual.title, /not an observed runtime receipt/i);
  assert.match(lesson.content.visual.reading, /does not read a TPU execution receipt/i);
});

test('tpu-05 routes the CPU-only lesson away from TPU and includes a collective update program', () => {
  runPython('tpu-05');
  const file = path.join(root, 'phases/20-tpu-workflows/02-run-course-scripts-and-multihost-tpu-slices/lesson.json');
  const lesson = JSON.parse(readFileSync(file, 'utf8'));
  const trainingCommand = lesson.content.sections[2].commands.find((command) => /jax\.lax\.psum/.test(command.code));
  assert.ok(trainingCommand, 'multi-host training command is present');
  assert.match(trainingCommand.code, /jax\.lax\.psum/);
  assert.match(trainingCommand.code, /process_index \+ 1/);
  assert.match(trainingCommand.code, /sync_global_devices/);
  assert.match(trainingCommand.code, /expected_mean_gradient = -float\(process_count \+ 1\)/);
  assert.match(trainingCommand.code, /expected_theta = 0\.1 \* float\(process_count \+ 1\)/);
  assert.match(lesson.content.sections[2].commands.at(-1).code, /--worker=all/);
  assert.match(trainingCommand.expected, /after running on and naming a multi-host slice/i);
});

test('tpu-06 retains an inspectable CPU trace from the measured workload', () => {
  const temp = mkdtempSync(path.join(os.tmpdir(), 'tpu-lesson-trace-'));
  try {
    const trace = path.join(temp, 'trace');
    const output = runPython('tpu-06', '', { TPU_TRACE_DIR: trace });
    assert.match(output, /Retained trace directory:/);
    assert.match(output, /Captured \*\.xplane\.pb trace count: 1/);
    assert.ok(existsSync(path.join(trace, 'plugins', 'profile')) || readdirSync(trace).length > 0);
  } finally {
    rmSync(temp, { recursive: true, force: true });
  }
});

test('analytic figures render to isolated artifacts with non-device provenance', () => {
  const temp = mkdtempSync(path.join(os.tmpdir(), 'tpu-analytic-figures-'));
  try {
    for (const id of ['tpu-03', 'tpu-04']) {
      const result = spawnSync('python3', ['scripts/render-lesson-figures.py', '--worker', '--lesson', id, '--output-dir', temp], { cwd: root, encoding: 'utf8' });
      assert.equal(result.status, 0, `${id} render failed: ${result.stdout} ${result.stderr}`);
      const folder = path.join(temp, id);
      const record = JSON.parse(readFileSync(path.join(folder, 'visual.json'), 'utf8'));
      assert.equal(record.kind, 'analytic');
      assert.equal(record.provenance, 'analytic_calculation_not_device_execution');
      assert.ok(record.data.panels?.length || ['bar','line','table','heatmap'].includes(record.data.kind));
      assert.ok(existsSync(path.join(folder, 'figure.svg')) && existsSync(path.join(folder, 'figure.png')));
      const svg = readFileSync(path.join(folder, 'figure.svg'), 'utf8');
      assert.match(svg, /Analytic calculation/);
      assert.match(svg, /no device result recorded/);
    }
  } finally {
    rmSync(temp, { recursive: true, force: true });
  }
});
