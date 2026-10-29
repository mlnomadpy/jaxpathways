import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

const root = 'projects/posttraining-audit/transformer_lineage';
const checker = `${root}/teaching/v5/check_numeric_packet.py`;
const approval = `${root}/portable-nonrun-approval.json`;
const approvalSha256 = '28f625fa4a1e8f705ce6b0e4a4ae56e241ce33d7ec5f3ec091e8428df738838b';

test('public numeric-packet self-check rejects tampering and preserves the failed quality verdict', () => {
  const approvalBytes = readFileSync(approval);
  assert.equal(createHash('sha256').update(approvalBytes).digest('hex'), approvalSha256);
  const report = JSON.parse(
    execFileSync('python3', [checker, '--shipping-approval', approval, '--self-test'], {
      encoding: 'utf8',
      maxBuffer: 4 * 1024 * 1024,
    }),
  );
  assert.equal(report.status, 'passed');
  assert.equal(report.qualityStatus, 'QUALITY_GATE_FAILED');
  assert.equal(report.shippingFilesVerified, 15);
  assert.equal(report.asRunFilesHistoricallyBound, 15);
  assert.equal(report.asRunByteIdenticalShippingFiles, 14);
  assert.equal(report.tamperControlsRejected.length, 10);
  assert.deepEqual(report.tamperControlsRejected, [
    'prompt-schedule-mutation',
    'early-eos-action-grammar',
    'payload-byte-tamper',
    'rehash-metric-tamper',
    'balanced-rehashed-episode-mutation',
    'terminal-trace-reset-mutant',
    'forbidden-logit-payload',
    'misleading-quality-pass',
    'source-inventory-tamper',
    'extra-payload',
  ]);
  assert.equal(
    report.evaluation.heldout.meanChildReward,
    report.evaluation.heldout.meanParentReward,
    'equal heldout reward must remain a failed strict-positive quality floor',
  );
  assert.equal(report.evaluation.heldout.validActionDenominator, 29);
  assert.equal(
    report.payloads['figure.svg'].sha256,
    'cadb82d712650be2483285e654e5132f91ab84b4b7d502be184f980adceb964f',
  );
});
