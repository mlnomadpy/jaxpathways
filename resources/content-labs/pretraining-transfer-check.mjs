import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { lessonScript } from '../../scripts/lesson-artifacts.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const python = process.env.PYTHON || 'python3';
const output = mkdtempSync(path.join(os.tmpdir(), 'jaxpathways-pretraining-script-check-'));
const lessons = [
  ['01-mlm', 'pretraining-01', 'Masked language modeling'],
  ['02-mim', 'pretraining-02', 'Masked image modeling'],
  ['03-contrastive', 'pretraining-03', 'Contrastive representation learning'],
];

try {
  for (const [slug, id, title] of lessons) {
    const sourcePath = path.join(root, 'phases/17-pretraining', slug, 'lesson.json');
    const authored = JSON.parse(readFileSync(sourcePath, 'utf8'));
    const source = lessonScript({
      id,
      title,
      path: `phases/17-pretraining/${slug}`,
      minutes: authored.minutes,
      content: authored.content,
    });
    const scriptPath = path.join(output, `${id}.py`);
    writeFileSync(scriptPath, source);
    const result = spawnSync(python, [scriptPath], {
      cwd: root,
      encoding: 'utf8',
      env: { ...process.env, JAX_PLATFORMS: 'cpu' },
    });
    process.stdout.write(result.stdout || '');
    process.stderr.write(result.stderr || '');
    if (result.status !== 0) {
      throw new Error(`${id} generated lesson script exited ${result.status}`);
    }
  }
} finally {
  rmSync(output, { recursive: true, force: true });
}
