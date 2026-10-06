import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { Window } from 'happy-dom';
import { lessonVisual, lessonExecution } from '../src/lib/lesson-visuals.js';
import { markdownText } from '../scripts/lesson-artifacts.mjs';
import { figureNotebookCode } from '../scripts/figure-artifacts.mjs';
const course = JSON.parse(fs.readFileSync('public/curriculum.json','utf8'));
const lessons = course.phases.flatMap(p=>p.lessons).filter(l=>l.status==='authored');

test('every authored lesson has current figure data, explanation, and executed notebook outputs', () => {
  const rendererHash=createHash('sha256').update(fs.readFileSync('scripts/render-lesson-figures.py')).digest('hex');
  for(const lesson of lessons){
    const v=lesson.content.visual;
    assert(v?.title && v.prediction && v.reading && v.connection,lesson.id);
    const data=JSON.parse(fs.readFileSync(`${lesson.path}/outputs/visual.json`));
    assert.equal(data.contentHash,lesson.contentHash,lesson.id);
    assert.equal(data.rendererHash,rendererHash,lesson.id);
    assert.equal(lesson.visualArtifact.kind,v.kind);
    assert.equal(lesson.execution.contentHash,lesson.contentHash);
    const record=JSON.parse(fs.readFileSync(`${lesson.path}/outputs/execution.json`));
    const notebook=JSON.parse(fs.readFileSync(lesson.artifacts.notebookSource));
    const cells=notebook.cells.filter(c=>c.cell_type==='code');
    assert.equal(cells.length,record.notebookCells.length);
    for(const [i,cell] of cells.entries()){
      assert.equal(cell.execution_count,i+1);
      assert.equal(createHash('sha256').update(cell.source.join('')).digest('hex'),record.notebookCells[i].sourceHash);
      assert.deepEqual(cell.outputs,record.notebookCells[i].outputs);
    }
    const figureIndex=notebook.cells.findIndex(c=>c.cell_type==='code'&&c.source.join('')===figureNotebookCode(lesson));
    assert(figureIndex>=0,lesson.id);
    const walkthrough=notebook.cells[figureIndex+1];
    assert.equal(walkthrough.cell_type,'markdown');
    assert(walkthrough.source.join('').includes(markdownText(v.reading)),lesson.id);
    assert(walkthrough.source.join('').includes(markdownText(v.connection)),lesson.id);
    assert(cells.some(c=>c.outputs.some(o=>o.output_type==='display_data'&&o.data['image/png'])),lesson.id);
    if(v.code)assert(cells.some(c=>c.source.join('')===v.code),lesson.id);
    const markdown=fs.readFileSync(lesson.artifacts.source,'utf8');
    assert(markdown.includes('../outputs/figure.svg'),lesson.id);
    assert(markdown.includes(lesson.execution.stdout),lesson.id);
  }
});

test('figures have captions, explanations, keyboard access and downloadable evidence', async () => {
  const window=new Window();
  try{
    for(const lesson of lessons){
      window.document.body.innerHTML=lessonVisual(lesson)+lessonExecution(lesson);
      assert(window.document.querySelector('img')?.getAttribute('alt'),lesson.id);
      assert(window.document.querySelector('figcaption')?.textContent.includes(lesson.content.visual.kind==='executed'?'CPU computation':'Conceptual'),lesson.id);
      const walkthrough=window.document.querySelector('.figure-walkthrough');
      assert.equal(walkthrough.querySelectorAll('p').length, lesson.content.visual.reading.split('\n\n').length+lesson.content.visual.connection.split('\n\n').length,lesson.id);
      assert(walkthrough.textContent.includes(lesson.content.visual.connection.split('\n\n').at(-1).split(/\\\(|`/)[0]),lesson.id);
      assert.equal(window.document.querySelector('.figure-scroll').getAttribute('tabindex'),'0');
      assert.equal(window.document.querySelector('.lesson-execution pre').textContent,lesson.execution.stdout);
      assert(window.document.querySelectorAll('a[download]').length===2);
    }
  }finally{await window.happyDOM.abort();}
});

test('stale artifacts are suppressed and output is escaped',()=>{
  const lesson=lessons[0];
  assert.equal(lessonVisual({...lesson,contentHash:'changed'}),'');
  assert.equal(lessonExecution({...lesson,contentHash:'changed'}),'');
  const html=lessonExecution({...lesson,execution:{...lesson.execution,stdout:'<script>bad()</script>'}});
  assert(!html.includes('<script>'));
  assert(html.includes('&lt;script&gt;'));
});
