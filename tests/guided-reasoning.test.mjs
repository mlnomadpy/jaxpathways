import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
import { Window } from 'happy-dom';
import { deepConcepts } from '../src/lib/lesson-content.js';
import { lessonMechanism } from '../src/lib/lesson-visuals.js';
import { markdownText } from '../scripts/lesson-artifacts.mjs';

const course=JSON.parse(fs.readFileSync('public/curriculum.json','utf8'));
const lessons=course.phases.flatMap(p=>p.lessons).filter(l=>l.status==='authored');

test('worked reasoning and concealed answers survive every lesson companion', async()=>{
  const window=new Window();
  try {
    for(const lesson of lessons){
      const section=lesson.content.sections.find(s=>s.id==='guided-reasoning');
      assert(section?.body && section.check?.prompt && section.check.answer,lesson.id);
      window.document.body.innerHTML=deepConcepts(lesson.content,lesson);
      const check=window.document.querySelector('.reasoning-check');
      assert(check,lesson.id);
      assert(!check.querySelector('details').open,lesson.id);
      assert(check.querySelector('summary').textContent.includes('Compare your reasoning'));
      const markdown=fs.readFileSync(lesson.artifacts.source,'utf8');
      const notebook=JSON.parse(fs.readFileSync(lesson.artifacts.notebookSource,'utf8'));
      for(const text of [section.body,section.check.prompt,section.check.answer]){
        assert(markdown.includes(markdownText(text)),lesson.id);
        assert(notebook.cells.some(c=>c.cell_type==='markdown'&&c.source.join('').includes(markdownText(text))),lesson.id);
      }
    }
  }finally{await window.happyDOM.abort();}
});

test('mechanism diagrams are current, accessible and embedded for offline notebooks',async()=>{
  const window=new Window();
  try {
    for(const phase of course.phases){
      assert(phase.lessons.some(l=>l.content?.diagram),phase.id);
      for(const lesson of phase.lessons.filter(l=>l.content?.diagram)){
        const d=lesson.content.diagram;
        const record=JSON.parse(fs.readFileSync(lesson.path+'/outputs/visual.json','utf8'));
        assert.deepEqual(record.diagram,d,lesson.id);
        assert.equal(record.contentHash,lesson.contentHash,lesson.id);
        window.document.body.innerHTML=lessonMechanism(lesson);
        const img=window.document.querySelector('img');
        assert(img?.getAttribute('alt'),lesson.id);
        assert(window.document.getElementById(img.getAttribute('aria-describedby')),lesson.id);
        const notebook=JSON.parse(fs.readFileSync(lesson.artifacts.notebookSource,'utf8'));
        const cell=notebook.cells.find(c=>c.attachments?.['mechanism.png']);
        assert(cell?.source.join('').includes('attachment:mechanism.png'),lesson.id);
        assert.deepEqual(Buffer.from(cell.attachments['mechanism.png']['image/png'],'base64'),fs.readFileSync(lesson.path+'/outputs/mechanism.png'));
        assert(fs.readFileSync(lesson.artifacts.source,'utf8').includes('../outputs/mechanism.svg'),lesson.id);
        assert.equal(lessonMechanism({...lesson,contentHash:'stale'}),'');
      }
    }
  }finally{await window.happyDOM.abort();}
});

test('hand-worked numeric examples agree with independent calculations',()=>{
  const output=execFileSync('python3',['assessments/guided-reasoning-reference.py'],{encoding:'utf8'});
  assert(output.includes('independent worked-example calculations'));
});
