import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';

const read = (path) => JSON.parse(readFileSync(path, 'utf8'));
const registry = read('curriculum/hardware-validation.json');

test('current hardware claims retain exact program identities and actual target receipts', () => {
  assert.equal(registry.schemaVersion, 1);
  for (const entry of registry.qualifications) {
    assert.match(entry.sourceCommit, /^[0-9a-f]{40}$/);
    assert(entry.scope);
    assert(readFileSync(entry.environment, 'utf8').includes('libtpu==0.0.37'));
    if (entry.status !== 'current') {
      assert.equal(entry.status, 'historical');
      continue;
    }
    for (const [path, expected] of Object.entries(entry.sources)) {
      assert.equal(createHash('sha256').update(readFileSync(path)).digest('hex'), expected,
        `${entry.id}: ${path} changed; rerun on hardware or mark the qualification historical`);
    }
    for (const path of [entry.evidence].flat()) {
      const receipt = read(path);
      if (entry.kind === 'single-host-training-recovery-precision') {
        assert.equal(receipt.backend, 'tpu');
        assert.equal(receipt.commit, entry.sourceCommit);
        assert.deepEqual(receipt.runs.map((run) => [run.name, run.exitCode]),
          [['training', 0], ['resume', 0], ['precision', 0]]);
        for (const run of receipt.runs) assert.equal(run.sha256, entry.sources[run.source]);
      } else if (entry.kind === 'single-device-pallas') {
        assert.equal(receipt.executionMode, 'real-tpu');
        assert.equal(receipt.targetValidated, true);
        assert.equal(receipt.implementationSha256, entry.sources['projects/kernel-audit/solution/model.py']);
        assert(receipt.devices.every((device) => device.platform === 'tpu'));
        assert.equal(receipt.measurement.actual_backend, 'tpu');
        assert.equal(receipt.max_abs_error_vs_represented_numpy_oracle, 0);
      } else if (entry.kind === 'single-host-four-device-collectives') {
        assert.equal(receipt.backend, 'tpu');
        assert.equal(receipt.executionMode, 'real-tpu-single-host-four-device');
        assert.equal(receipt.processCount, 1);
        assert.equal(receipt.globalDeviceCount, 4);
        assert.equal(receipt.localDeviceCount, 4);
        assert.equal(receipt.devices.length, 4);
        assert(receipt.devices.every((device) => device.platform === 'tpu'));
        assert.equal(receipt.sourceSha256, entry.sources['resources/tpu-gcp/collectives_target.py']);
        assert.equal(receipt.baseLessonSha256, entry.sources[receipt.baseLesson]);
        assert.equal(receipt.numericalChecks.dotPrecision, 'HIGHEST');
        assert.equal(receipt.numericalChecks.allReduceMatchesReference, true);
        assert.equal(receipt.numericalChecks.reduceScatterThenAllGatherMatchesReference, true);
        const root = path.slice(0, path.lastIndexOf('/'));
        for (const [name, collective] of Object.entries(receipt.collectives)) {
          const hlo = readFileSync(`${root}/${collective.stablehloFile}`, 'utf8');
          assert(hlo.includes(`stablehlo.${collective.stablehloOperation}`));
          const samples = receipt.timings[name].samples_ms;
          assert.equal(samples.length, receipt.repeats);
          assert(samples.every((sample) => Number.isFinite(sample) && sample > 0));
        }
        const traceFiles = readdirSync(`${root}/trace`, { recursive: true });
        assert(traceFiles.some((file) => file.endsWith('.xplane.pb')));
        const perfetto = traceFiles.find((file) => file.endsWith('perfetto_trace.json.gz'));
        const trace = JSON.parse(gunzipSync(readFileSync(`${root}/trace/${perfetto}`)));
        assert(trace.traceEvents.some((event) => event.name === 'tpu::System::Execute'));
      } else if (entry.kind === 'multi-host-training-recovery') {
        assert.equal(receipt.executionMode, 'real-tpu-multi-host-sixteen-device');
        assert.equal(receipt.backend, 'tpu');
        assert.equal(receipt.processCount, 4);
        assert.equal(receipt.globalDeviceCount, 16);
        assert.equal(receipt.localDeviceCount, 4);
        assert.equal(receipt.globalDevices.length, 16);
        assert(receipt.globalDevices.every((device) => device.platform === 'tpu' && device.kind === 'TPU v5 lite'));
        assert.equal(receipt.sourceSha256, entry.sources['resources/tpu-gcp/multihost_target.py']);
        assert.equal(receipt.baseLessonSha256, entry.sources['phases/09-distributed/04-resilient-distributed-training/code/main.py']);
        assert.equal(receipt.sourceAndLessonEqualAcrossProcesses, true);
        assert.equal(receipt.allRangesAreDisjointAndComplete, true);
        assert.equal(receipt.arrayConstruction, 'jax.make_array_from_process_local_data');
        assert.deepEqual(receipt.ownedRows, [receipt.processIndex * 16, (receipt.processIndex + 1) * 16]);
        assert.equal(receipt.localDataSha256ByProcess[receipt.processIndex], receipt.localDataSha256);
        assert.equal(receipt.optimizer.checkpointStep, 4);
        assert.equal(receipt.optimizer.finalStep, 8);
        assert.equal(receipt.optimizer.checkpointUriScheme, 'gs');
        assert.equal(receipt.optimizer.checkpointUriIncluded, false);
        assert.equal(receipt.optimizer.storageClient, 'gcsfs.core.GCSFileSystem');
        assert.equal(receipt.optimizer.bucketNamespace, 'flat');
        for (const key of ['orbaxCheckpointSavedAndWaited', 'abstractShardedRestoreSucceeded',
          'restoredStateMatchesSavedStep', 'finalWeightsMatchIndependentNumpy', 'finalMomentumMatchesIndependentNumpy']) {
          assert.equal(receipt.optimizer[key], true, key);
        }
      } else {
        assert.fail(`Unknown qualification kind: ${entry.kind}`);
      }
    }
  }
});

test('multi-host proof retains all ranks, one checkpoint identity and a portable checkpoint', () => {
  const entry = registry.qualifications.find((item) => item.kind === 'multi-host-training-recovery');
  const receipts = entry.evidence.map(read).sort((a, b) => a.processIndex - b.processIndex);
  assert.deepEqual(receipts.map((item) => item.processIndex), [0, 1, 2, 3]);
  const environments = entry.environments.map((path) => readFileSync(path, 'utf8'));
  assert.equal(new Set(environments).size, 1, 'all workers must use the same recorded environment');
  for (const version of ['flax==0.12.6', 'orbax-checkpoint==0.11.34', 'tensorstore==0.1.82', 'gcsfs==2026.10.1']) {
    assert(environments[0].includes(version));
  }
  assert.equal(new Set(receipts.map((item) => item.fixtureSha256)).size, 1);
  assert.equal(new Set(receipts.map((item) => item.optimizer.checkpointIdentitySha256)).size, 1);
  assert.equal(new Set(receipts[0].localDataSha256ByProcess).size, 4);
  for (const receipt of receipts) {
    assert.deepEqual(receipt.allOwnedRanges, [[0, 16], [16, 32], [32, 48], [48, 64]]);
    assert.deepEqual(receipt.optimizer.referenceLosses, receipts[0].optimizer.referenceLosses);
  }
  const root = 'docs/evidence/tpu-qualification-2026-10-20/v5e16';
  const manifest = read(`${root}/checkpoint-files.json`);
  for (const [path, expected] of Object.entries(manifest.publishedFiles)) {
    assert.equal(createHash('sha256').update(readFileSync(`${root}/checkpoint-step-4/${path}`)).digest('hex'), expected);
    if (path !== 'commit_success.txt') assert.equal(expected, manifest.files[path]);
  }
  const restored = read(`${root}/checkpoint-cpu-restore.json`);
  assert.equal(restored.backend, 'cpu');
  assert.equal(restored.state.step, 4);
  assert.equal(restored.weightsMatchNumpy, true);
  assert.equal(restored.momentumMatchesNumpy, true);
  assert.equal(restored.verificationSourceSha256, createHash('sha256').update(readFileSync(`${root}/restore-on-cpu.py`)).digest('hex'));
  const cleanup = read(`${root}/cleanup.json`);
  for (const key of ['ownedQueuedResourceCount', 'ownedTpuVmCount', 'ownedBucketCount', 'activeTemporaryServiceAccountCount']) {
    assert.equal(cleanup[key], 0, key);
  }
  assert.equal(cleanup.syntheticCheckpointDownloaded, true);
  assert.equal(cleanup.otherWorkloadsModified, false);
});

test('precision evidence includes actual integer checks and a retained TPU device trace', () => {
  const root = 'docs/evidence/tpu-qualification-2026-10-20/v5e1/smoke-v1/precision';
  const report = read(`${root}/report.json`);
  assert.equal(report.platform, 'tpu');
  assert.equal(report.local_device_count, 1);
  assert.deepEqual(report.results.map((result) => `${result.case}/${result.mode}`),
    ['balanced/fp32', 'balanced/bf16', 'balanced/int8', 'outlier/fp32', 'outlier/bf16', 'outlier/int8']);
  for (const result of report.results) {
    assert(Number.isFinite(result.max_absolute_error));
    assert(Number.isFinite(result.relative_l2_error));
    assert.equal(result.integer_accumulation_checked, result.mode === 'int8');
  }
  const files = readdirSync(`${root}/trace`, { recursive: true }).map((file) => `${root}/trace/${file}`);
  assert(files.some((file) => file.endsWith('.xplane.pb') && statSync(file).size > 0));
  const trace = JSON.parse(gunzipSync(readFileSync(files.find((file) => file.endsWith('perfetto_trace.json.gz')))));
  assert(trace.traceEvents.some((event) => event.name === 'tpu::System::Execute'));
});
