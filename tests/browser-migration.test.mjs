import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { Window } from 'happy-dom';
import { loadCourse, courseState } from '../src/lib/course-state.js';
import { initExperience } from '../src/scripts/experience.js';
import { initHomeCopy } from '../src/scripts/home.js';
import { renderHomeContinue } from '../src/scripts/resume.js';
import { initCatalog, restoreCourseView } from '../src/scripts/catalog.js';
import { initReaderEvents, showLesson } from '../src/scripts/reader.js';
import { setupChoices } from '../src/scripts/paths.js';
import { initProject } from '../src/scripts/project.js';
import { setupPlatform } from '../src/scripts/workspace.js';

async function page(name, query = '', records = {}) {
  const window = new Window({ url: `https://example.org/jaxpathways/${name}.html${query}`, settings: { disableJavaScriptEvaluation: true, disableCSSFileLoading: true, disableJavaScriptFileLoading: true } });
  window.document.write(fs.readFileSync(`dist/${name}.html`, 'utf8'));
  window.HTMLElement.prototype.scrollIntoView = function () {};
  for (const key of ['document','location','history','navigator','localStorage','MutationObserver','HTMLSelectElement','HTMLElement','Event','FormData','KeyboardEvent']) Object.defineProperty(globalThis,key,{value:window[key],configurable:true,writable:true});
  globalThis.window=window;
  globalThis.matchMedia=window.matchMedia.bind(window);
  globalThis.requestAnimationFrame=window.requestAnimationFrame.bind(window);
  const requests=[];
  globalThis.fetch=async input=>{
    const url=new URL(String(input),window.location.href);requests.push(url.pathname);
    assert(url.pathname.startsWith('/jaxpathways/'),`escaped deployment base: ${url}`);
    const file='dist/'+url.pathname.slice('/jaxpathways/'.length);
    return {ok:fs.existsSync(file),json:async()=>JSON.parse(fs.readFileSync(file,'utf8'))};
  };
  for(const [key,value] of Object.entries(records))window.localStorage.setItem(key,JSON.stringify(value));
  Object.assign(courseState,{standalone:false,course:null,curriculum:[],activePath:'all',courseQuery:'',courseHardware:'all',completed:false,readingState:{version:1,lessons:{}},readingFrame:0,lessonProgress:{version:1,lessons:{}}});
  return {window,requests,$:selector=>window.document.querySelector(selector),all:selector=>[...window.document.querySelectorAll(selector)],close:()=>window.happyDOM.close()};
}

test('home retains saved reading position, copies exact tutor text and controls keyboard navigation',async()=>{
  const reading={version:1,lastLessonId:'first-gradient',path:'foundations',sectionId:'section-2',updatedAt:'2026-10-04',lessons:{'first-gradient':{sectionId:'section-2',updatedAt:'2026-10-04'}}};
  const p=await page('index','',{'jaxpathways-reading-v1':reading});
  try {
    await loadCourse();initExperience();renderHomeContinue();initHomeCopy();
    assert.equal(p.$('#home-continue').hidden,false);
    assert.match(p.$('#home-continue .primary').href,/lesson=first-gradient&path=foundations#section-2$/);
    const copied=[];Object.defineProperty(navigator,'clipboard',{value:{writeText:async text=>copied.push(text)},configurable:true});
    p.$('[data-home-copy="tutor-install"]').click();await Promise.resolve();
    assert.equal(copied[0],p.$('#tutor-install').textContent);
    assert.match(p.$('#tutor-install-status').textContent,/Copied/);
    navigator.clipboard.writeText=async()=>{throw Error('Clipboard unavailable');};
    p.$('[data-home-copy="tutor-prompt"]').click();await Promise.resolve();
    assert.match(p.$('#tutor-prompt-status').textContent,/manually/);
    const menu=p.$('.navigation-toggle');menu.click();assert.equal(menu.getAttribute('aria-expanded'),'true');
    menu.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}));assert.equal(menu.getAttribute('aria-expanded'),'false');
    const themeToggle=p.$('[data-theme-toggle]');assert.equal(themeToggle.getAttribute('aria-pressed'),'false');
    themeToggle.click();assert.equal(p.window.document.documentElement.dataset.theme,'dark');assert.equal(themeToggle.getAttribute('aria-pressed'),'true');assert.equal(localStorage.getItem('jaxpathways-theme'),'dark');
    themeToggle.click();assert.equal(p.window.document.documentElement.dataset.theme,'light');assert.equal(themeToggle.getAttribute('aria-pressed'),'false');assert.equal(localStorage.getItem('jaxpathways-theme'),'light');
    assert.equal(localStorage.getItem('jaxpathways-reading-v1'),JSON.stringify(reading));
  } finally {await p.close();}
});

test('catalog retains route, roadmap and browser-history filtering under a project base',async()=>{
  const p=await page('course','?path=ship&phase=deployment&roadmap=1');
  try {
    await loadCourse();initCatalog();
    assert.equal(courseState.activePath,'ship');assert.equal(p.$('#phase-deployment').open,true);assert.equal(p.$('#phase-deployment').hidden,false);
    history.replaceState(null,'','course.html?path=foundations');restoreCourseView();
    assert.equal(courseState.activePath,'foundations');assert.equal(p.$('#show-roadmap').checked,false);assert.equal(p.$('#phase-deployment').hidden,true);
    assert(p.all('.lesson-link').length>70);assert.equal(p.requests[0],'/jaxpathways/curriculum.json');
  } finally {await p.close();}
});

test('every authored lesson renders, and checkpoint progress stays separate from exercise evidence',async()=>{
  const p=await page('lesson','?lesson=welcome-01');
  try {
    await loadCourse();initReaderEvents();
    for(const phase of courseState.course.phases)for(const lesson of phase.lessons.filter(l=>l.status==='authored')){
      showLesson(phase,lesson);assert.equal(p.$('#lesson-title').textContent,lesson.title);
      assert(p.$('#lesson-checkpoint'));assert(p.$('.copy-snippet'));assert(p.$('#reader-toc').children.length>0);
      const equations=p.all('#lesson-body annotation[encoding="application/x-tex"]').map(node=>node.textContent);
      for(const section of lesson.content.sections||[])if(section.math)assert(equations.includes(section.math),`reader omitted equation in ${lesson.id}`);
      // Let pending reader tasks settle before replacing the next full chapter.
      await new Promise(setImmediate);
    }
    const phase=courseState.course.phases[0],lesson=phase.lessons[0];showLesson(phase,lesson);
    p.$(`input[name="answer"][value="${lesson.content.answer}"]`).checked=true;
    p.$('#lesson-checkpoint').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}));
    let saved=JSON.parse(localStorage.getItem('jaxpathways-lessons-v1'));
    assert.equal(saved.lessons[lesson.id].checkpointPassed,true);assert.equal(saved.lessons[lesson.id].evidenceSaved,false);
    p.$('#exercise-complete').click();saved=JSON.parse(localStorage.getItem('jaxpathways-lessons-v1'));assert.equal(saved.lessons[lesson.id].evidenceSaved,true);
    p.$('#next').click();assert.notEqual(p.$('#lesson-title').textContent,lesson.title);
    assert(JSON.parse(localStorage.getItem('jaxpathways-reading-v1')).lastLessonId);
  } finally {await p.close();}
});

test('all career routes render and a selected route saves the existing plan schema',async()=>{
  const p=await page('careers','?role=inference-engineer');
  try {
    await loadCourse();setupChoices();assert.equal(p.all('#role-cards .role-card').length,courseState.course.roles.length);assert.equal(p.$('#role-detail').hidden,false);
    p.$('#career-save').click();const saved=JSON.parse(localStorage.getItem('jaxpathways-career-plan-v1'));
    assert.equal(saved.version,1);assert.equal(saved.roleId,'inference-engineer');assert(saved.startPhaseId);
    p.$('#close-career').click();assert.equal(p.$('#role-detail').hidden,true);
  } finally {await p.close();}
});

test('all learning paths open with their actual lesson availability',async()=>{
  const p=await page('pathways','?path=models');
  try {
    await loadCourse();setupChoices();assert.equal(p.all('.path-card').length,9);assert.equal(p.$('#path-detail').hidden,false);
    assert(p.$('#path-detail h1').textContent.includes('neural'));assert(p.$('#use-path'));assert(p.all('.path-journey li').length>0);
  } finally {await p.close();}
});

test('both project workspaces render and stage ticks preserve their storage keys',async()=>{
  for(const id of ['regression-audit','mlp-classifier']){
    const p=await page('project',`?id=${id}`);
    try {
      initExperience();await initProject();assert(p.$('.project-stages'));const first=p.$('[data-stage]');assert(first);first.checked=true;first.dispatchEvent(new Event('change'));
      assert.deepEqual(JSON.parse(localStorage.getItem('jaxpathways-project-'+id)),[first.dataset.stage]);
      assert.match(p.$('#project-status').textContent,/self-reported/);
    } finally {await p.close();}
  }
});

test('notebook saves evidence through the real form and planner uses canonical durations',async()=>{
  let p=await page('notebook');
  try {
    await loadCourse();await setupPlatform(courseState.curriculum);
    p.$('#evidence-title').value='Migration experiment';p.$('#evidence-note').value='Observed a changed result and explained the difference.';
    p.$('#evidence-form').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}));
    const saved=JSON.parse(localStorage.getItem('jaxpathways-notebook-v1'));assert.equal(saved.evidence[0].title,'Migration experiment');
    assert.equal(p.all('#evidence-list article').length,1);
  } finally {await p.close();}
  p=await page('organizer');
  try {
    await loadCourse();await setupPlatform(courseState.curriculum);assert(!p.requests.some(request=>request.endsWith('/api/v1/projects.json')),'organizer must not fetch notebook project data');assert(p.all('#club-plan article').length>0);assert.equal(p.$('#club-download').disabled,false);assert(p.$('#club-summary').textContent.length>20);
  } finally {await p.close();}
});

test('permanent lesson pages retain pathway, section and existing learner evidence when enhanced', async()=>{
  const p=await page('lesson-welcome-01','?path=foundations#section-2',{'jaxpathways-lessons-v1':{version:1,lessons:{'welcome-01':{checkpointPassed:true,evidenceSaved:true}}}});
  try {
    const staticTitle=p.$('#lesson-title');
    assert.equal(staticTitle.textContent,'Set up your learning workspace');
    await loadCourse();courseState.activePath='foundations';initReaderEvents();
    const found=courseState.course.phases[0];showLesson(found,found.lessons[0]);
    assert.equal(p.$('#lesson-title'),staticTitle);
    assert.equal(location.pathname,'/jaxpathways/lesson-welcome-01.html');
    assert.equal(location.search,'?path=foundations');assert.equal(location.hash,'#section-2');
    assert.match(p.$('#exercise-complete').textContent,/unfinished/);
    assert.match(p.$('#checkpoint-result').textContent,/previously passed/);
    assert.equal(JSON.parse(localStorage.getItem('jaxpathways-lessons-v1')).lessons['welcome-01'].evidenceSaved,true);
  } finally {await p.close();}
});
