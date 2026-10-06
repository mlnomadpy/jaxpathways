import { readFileSync, existsSync, mkdirSync, copyFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
const hash = value => createHash('sha256').update(value).digest('hex');
export function loadLessonEvidence(lesson) {
  const folder = `${lesson.path}/outputs`;
  for (const dir of ['public/figures', 'public/executions']) mkdirSync(dir, { recursive: true });
  const visualPath = `${folder}/visual.json`;
  if (existsSync(visualPath)) {
    const record = JSON.parse(readFileSync(visualPath, 'utf8'));
    if (record.contentHash === lesson.contentHash && record.rendererHash === hash(readFileSync('scripts/render-lesson-figures.py'))) {
      for (const ext of ['svg', 'png']) copyFileSync(`${folder}/figure.${ext}`, `public/figures/${lesson.id}.${ext}`);
      copyFileSync(visualPath, `public/figures/${lesson.id}.json`);
      lesson.visualArtifact = { image: `figures/${lesson.id}.svg`, png: `figures/${lesson.id}.png`, data: `figures/${lesson.id}.json`, generatedAt: record.generatedAt, environment: record.environment, contentHash: record.contentHash, kind: record.kind };
      if (lesson.content.diagram && ['svg','png'].every(ext=>existsSync(`${folder}/mechanism.${ext}`))) {
        for (const ext of ['svg','png']) copyFileSync(`${folder}/mechanism.${ext}`, `public/figures/${lesson.id}-mechanism.${ext}`);
        lesson.visualArtifact.mechanismImage = `figures/${lesson.id}-mechanism.svg`;
        lesson.visualArtifact.mechanismPng = `figures/${lesson.id}-mechanism.png`;
      }
    }
  }
  const executionPath = `${folder}/execution.json`;
  if (existsSync(executionPath)) {
    const record = JSON.parse(readFileSync(executionPath, 'utf8'));
    if (record.contentHash === lesson.contentHash) {
      copyFileSync(executionPath, `public/executions/${lesson.id}.json`);
      lesson.execution = { contentHash: record.contentHash, executedAt: record.executedAt, environment: record.environment, stdout: record.stdout, url: `executions/${lesson.id}.json` };
    }
  }
}
export function figureNotebookCode(lesson) {
  const renderer = readFileSync('scripts/render-lesson-figures.py', 'utf8');
  const helper = renderer.slice(renderer.indexOf('def draw_panel('), renderer.indexOf('\ndef worker('));
  const data = lesson.content.visual.kind === 'conceptual' ? `visual_data = ${JSON.stringify({kind:'flow',nodes:lesson.content.visual.nodes,layout:lesson.content.visual.layout||'sequence'})}\n` : '';
  return `# Render the figure from the data computed above. Change the data experiment and rerun.\nimport matplotlib.pyplot as plt\nplt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13, 'axes.labelcolor': '#30273e', 'text.color': '#30273e'})\n${helper}\n${data}panels = visual_data.get('panels', [visual_data])\nfig, axes = plt.subplots(len(panels), 1, figsize=(8, 4.6 * len(panels)), squeeze=False, layout='constrained')\nfor ax, panel in zip(axes[:, 0], panels):\n    draw_panel(ax, panel)\nplt.show()`;
}
export function restoreNotebookOutputs(lesson, cells) {
  const path = `${lesson.path}/outputs/execution.json`;
  if (!lesson.execution || !existsSync(path)) return cells;
  const record = JSON.parse(readFileSync(path, 'utf8'));
  const codeCells = cells.filter(cell => cell.cell_type === 'code');
  if (codeCells.length !== record.notebookCells.length || !codeCells.every((cell, i) => hash(cell.source.join('')) === record.notebookCells[i].sourceHash)) return cells;
  for (const [i, cell] of codeCells.entries()) {
    cell.execution_count = i + 1;
    cell.outputs = record.notebookCells[i].outputs;
  }
  return cells;
}
