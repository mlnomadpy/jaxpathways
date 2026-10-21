import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
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
  runPython('tpu-03', `specs = {item['gen']: item for item in namespace['TPU_SPECS']}\nassert specs['v6e']['bw_gbs'] == 1638.0\nassert abs(specs['v6e']['ridge_flops_per_byte'] - 560.4) < 0.01\nassert specs['v5e']['ridge_flops_per_byte'] == 240.5\nassert specs['v5p']['ridge_flops_per_byte'] == 166.0`);
  const file = path.join(root, 'phases/21-tpu-systems/01-tpu-generations-and-hbm-memory-budgeting/lesson.json');
  const lesson = JSON.parse(readFileSync(file, 'utf8'));
  assert.equal(lesson.content.visual.kind, 'analytic');
  assert.match(lesson.content.visual.connection, /not device measurements/i);
  const course = JSON.parse(readFileSync('curriculum/course.json', 'utf8'));
  const metadata = course.phases.flatMap(phase => phase.lessons).find(lesson => lesson.id === 'tpu-03');
  assert.match(metadata.check, /18 bytes per parameter/);
});

test('v5e Flex-start instructions keep a bounded queue and explicit owner cleanup', () => {
  const lesson = JSON.parse(readFileSync('phases/19-tpu/01-get-a-tpu-running-and-verify-the-runtime/lesson.json', 'utf8'));
  const commands = lesson.content.sections.flatMap(section => section.commands || []).map(command => command.code).join('\n');
  assert.match(commands, /queued-resources create/);
  assert.match(commands, /--max-run-duration=1h/);
  assert.match(commands, /--valid-until-duration=30m/);
  assert.match(commands, /queued-resources delete/);
  assert.match(commands, /--force --quiet/);
  assert.match(commands, /queued-resources list/);
  assert.doesNotMatch(commands, /resize-requests|instance-templates/);
  const prose = lesson.content.sections.map(section => section.body).join('\n');
  assert.doesNotMatch(prose, /READY.*STOPPED|complete every TPU verification/);
});

test('tpu-04 labels the four-device result as calculated expectation', () => {
  const file = path.join(root, 'phases/19-tpu/02-bootstrap-venv-and-verify-tpu-runtime-receipt/lesson.json');
  const lesson = JSON.parse(readFileSync(file, 'utf8'));
  assert.equal(lesson.content.visual.kind, 'analytic');
  assert.match(lesson.content.visual.title, /not an observed runtime receipt/i);
  assert.match(lesson.content.visual.reading, /does not read a TPU execution receipt/i);
});

test('Phase 19 registry tasks use the lesson exercises and receipt fields', () => {
  const course = JSON.parse(readFileSync('curriculum/course.json', 'utf8'));
  const phase = course.phases.find(item => item.id === 'tpu');
  const tpu01 = phase.lessons.find(item => item.id === 'tpu-01');
  const tpu04 = phase.lessons.find(item => item.id === 'tpu-04');
  const lesson01 = JSON.parse(readFileSync('phases/19-tpu/01-get-a-tpu-running-and-verify-the-runtime/lesson.json', 'utf8'));
  const lesson04 = JSON.parse(readFileSync('phases/19-tpu/02-bootstrap-venv-and-verify-tpu-runtime-receipt/lesson.json', 'utf8'));
  assert.match(tpu01.exercise, /verify_runtime_contract.*v5litepod-4.*v6e-4.*hosts == 1.*at least 64 GiB/i);
  assert.match(lesson01.content.exercise, /v5litepod-4.*v6e-4.*hosts == 1.*at least 64 GiB/i);
  assert.match(lesson01.content.learnerStarter.code, /def inspect_runtime_and_slice\(verify_runtime_contract, expected_backend, min_devices, topologies\)/);
  assert.match(lesson01.content.learnerStarter.code, /filter the supplied catalog by host count and HBM threshold/i);
  const output = runPython('tpu-01', `exec(compile(lesson['content']['solution'], path + ':solution', 'exec'), namespace)
catalog = namespace['topologies']
for name, chips, hbm in [('v5litepod-4', 4, 64), ('v6e-4', 4, 128)]:
    spec = next(row for row in catalog if row['slice'] == name)
    assert spec['hosts'] == 1 and spec['chips_per_host'] == chips
    assert spec['chips_per_host'] * spec['hbm_gib_per_chip'] == hbm
assert [row['slice'] for row in catalog if row['hosts'] == 1 and row['chips_per_host'] * row['hbm_gib_per_chip'] >= 64] == ['v5litepod-4', 'v5litepod-8', 'v6e-4']
assert [row['slice'] for row in catalog if row['hosts'] == 1 and row['chips_per_host'] * row['hbm_gib_per_chip'] >= 128] == ['v5litepod-8', 'v6e-4']`, { JAX_PLATFORMS: 'cpu' });
  assert.match(output, /v5litepod-4 hosts: 1 chips: 4 total HBM GiB: 64/);
  assert.match(output, /v6e-4 hosts: 1 chips: 4 total HBM GiB: 128/);
  assert.match(output, /at least 64 GiB: \[?'?v5litepod-4/);
  assert.match(tpu04.exercise, /closed_form_sum_of_squares\(4\).*closed_form_sum_of_squares\(8\).*10416\.0.*85344\.0/);
  assert.match(tpu04.exercise, /receipt\['verified'\]/);
  assert.match(lesson04.content.code, /def build_runtime_receipt\(/);
  assert.match(lesson04.content.code, /"local_device_count".*"observed_total".*"expected_total"/s);
  assert.match(lesson04.content.solution, /s4 == 10416\.0 and s8 == 85344\.0 and receipt\["verified"\] is True/);
  assert.doesNotMatch(JSON.stringify(tpu04), /verify_runtime_contract|contract_ok|expected_pmap_sum_of_squares|signed JSON/);
  const runtimeMilestone = phase.studyGuide.milestones.find(item => item.lessonIds.includes('tpu-04'));
  assert.match(runtimeMilestone.task, /build_runtime_receipt\(\)/);
  assert.doesNotMatch(runtimeMilestone.task, /verify_runtime_contract|contract_ok|expected_pmap_sum_of_squares|signed JSON/);
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
  const phase = JSON.parse(readFileSync('curriculum/course.json', 'utf8')).phases.find(phase => phase.id === 'tpu-systems');
  const metadata = phase.lessons.find(lesson => lesson.id === 'tpu-06');
  assert.match(metadata.exercise, /capture_warmed_trace\(steps=3\)/);
  assert.match(metadata.objective, /requested result-dtype/);
  assert.doesNotMatch(JSON.stringify(phase), /verify float32 accumulator precision|accumulator drift|verify FP32 accumulation inside|capture_warmed_xprof_trace/);
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

test('tpu-06 precision probe retains raw outputs and independently checks dtype and error summaries', () => {
  const lessonDir = path.join(root, 'phases/21-tpu-systems/02-bfloat16-int8-precision-and-xprof-profiling');
  const lesson = JSON.parse(readFileSync(path.join(lessonDir, 'lesson.json'), 'utf8'));
  const lessonText = JSON.stringify(lesson.content);
  assert.doesNotMatch(lessonText, /\b(?:bf16|fp32)\s+accum(?:ulation)?\b/i);
  assert.doesNotMatch(lessonText, /every partial sum.{0,100}rounded to 7/i);
  assert.match(lesson.content.sections[0].check.answer, /do not establish that every partial sum/i);
  assert.match(lesson.content.visual.reading, /do not expose the precision of every internal partial sum/i);
  const probe = path.join(lessonDir, 'labs/precision_contract_probe.py');
  const temp = mkdtempSync(path.join(os.tmpdir(), 'tpu06-precision-contract-'));
  try {
    const outputDir = path.join(temp, 'evidence');
    const run = spawnSync('python3', [probe, '--output-dir', outputDir], {
      cwd: root,
      encoding: 'utf8',
      env: { ...process.env, JAX_PLATFORMS: 'cpu' },
    });
    assert.equal(run.status, 0, `precision probe failed:\n${run.stdout}\n${run.stderr}`);

    const receipt = JSON.parse(readFileSync(path.join(outputDir, 'receipt.json'), 'utf8'));
    const resultPath = path.join(outputDir, receipt.summary.path);
    const results = JSON.parse(readFileSync(resultPath, 'utf8')).fixtures;
    assert.equal(receipt.backend, 'cpu');
    assert.deepEqual(receipt.fixture.seeds, [0, 42, 7]);
    assert.equal(receipt.fixture.changed_condition_seed, 7);
    assert.deepEqual(results.map(({ seed }) => seed), [0, 42, 7]);
    assert.equal(
      receipt.source_hashes['lesson.json'],
      createHash('sha256').update(readFileSync(path.join(lessonDir, 'lesson.json'))).digest('hex'),
    );
    assert.equal(
      receipt.source_hashes['precision_contract_probe.py'],
      createHash('sha256').update(readFileSync(probe)).digest('hex'),
    );
    for (const file of [
      ...receipt.raw_array_files,
      ...receipt.compiler_ir_artifacts,
      receipt.summary,
    ]) {
      assert.equal(
        createHash('sha256').update(readFileSync(path.join(outputDir, file.path))).digest('hex'),
        file.sha256,
        `${file.path} source-bound evidence hash`,
      );
    }
    assert.equal(receipt.compiler_ir_artifacts.length, 8);
    assert.match(receipt.interpretation_limit, /do not prove physical accumulator width/i);

    const verifier = String.raw`
import json, sys
import numpy as np
import ml_dtypes
from pathlib import Path
summary_path, arrays_dir = Path(sys.argv[1]), Path(sys.argv[2])
fixtures = json.loads(summary_path.read_text())['fixtures']
assert [row['seed'] for row in fixtures] == [0, 42, 7]
for row in fixtures:
    with np.load(arrays_dir / f"seed-{row['seed']}-arrays.npz") as raw:
        arrays = {name: raw[name] for name in raw.files}
    for name in ('x_bf16', 'w_bf16', 'requested_bf16', 'requested_fp32_cast_bf16'):
        arrays[name] = arrays[name].view(ml_dtypes.bfloat16)
    assert arrays['x_fp32'].dtype == np.dtype('float32')
    assert arrays['w_fp32'].dtype == np.dtype('float32')
    assert str(arrays['x_bf16'].dtype) == 'bfloat16'
    assert str(arrays['w_bf16'].dtype) == 'bfloat16'
    assert str(arrays['requested_fp32'].dtype) == 'float32'
    assert str(arrays['requested_bf16'].dtype) == 'bfloat16'
    assert np.array_equal(arrays['x_bf16'], arrays['x_fp32'].astype(arrays['x_bf16'].dtype))
    assert np.array_equal(arrays['w_bf16'], arrays['w_fp32'].astype(arrays['w_bf16'].dtype))
    for value in arrays.values():
        assert np.isfinite(value.astype(np.float32)).all()
    reference = arrays['reference_fp32'].astype(np.float64)
    for key, output_name in [('bf16_operands_requested_fp32_result', 'requested_fp32'),
                             ('bf16_operands_requested_bf16_result', 'requested_bf16')]:
        error = np.abs(arrays[output_name].astype(np.float64) - reference)
        np.testing.assert_allclose(row['max_abs_error_vs_fp32_jax_reference'][key], error.max(), rtol=0, atol=1e-12)
        np.testing.assert_allclose(row['mean_abs_error_vs_fp32_jax_reference'][key], error.mean(), rtol=0, atol=1e-12)
    cast = arrays['requested_fp32'].astype(arrays['requested_bf16'].dtype)
    equal = cast == arrays['requested_bf16']
    comparison = row['requested_bf16_vs_same_operands_requested_fp32_result_cast_to_bf16']
    assert comparison['equal_count'] == int(equal.sum())
    assert comparison['different_count'] == int((~equal).sum())
    assert comparison['total_count'] == equal.size
    positions = np.argwhere(~equal)
    expected = [{'index': [int(v) for v in pos],
                 'requested_bf16_value': float(arrays['requested_bf16'][tuple(pos)].astype(np.float32)),
                 'requested_fp32_then_bf16_value': float(cast[tuple(pos)].astype(np.float32))}
                for pos in positions]
    assert comparison['different_positions_and_values'] == expected
`;
    const verify = (summaryPath) =>
      spawnSync('python3', ['-c', verifier, summaryPath, outputDir], {
        cwd: root,
        encoding: 'utf8',
      });
    const verified = verify(resultPath);
    assert.equal(verified.status, 0, `independent raw-array verification failed:\n${verified.stderr}`);

    const forgedPath = path.join(temp, 'forged-results.json');
    const forgedResults = JSON.parse(readFileSync(resultPath, 'utf8'));
    forgedResults.fixtures[0].max_abs_error_vs_fp32_jax_reference.bf16_operands_requested_bf16_result *= 0.01;
    writeFileSync(forgedPath, JSON.stringify(forgedResults));
    const forged = verify(forgedPath);
    assert.notEqual(forged.status, 0, 'independent checker rejects a fabricated error summary');
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
