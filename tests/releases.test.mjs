import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { Window } from 'happy-dom';
import { initReleaseNotice, initRevisionNotices } from '../src/scripts/release-notices.js';
import { releaseKey, revisionKey } from '../src/lib/releases.js';

function browser() {
 const window = new Window();
 globalThis.document = window.document; globalThis.localStorage = window.localStorage;
 window.document.body.innerHTML = '<aside id="release-notice" data-version="0.2.0" hidden><button id="dismiss-release">Dismiss</button></aside><div id="revision"></div>';
 return window;
}
test('release notices distinguish first visits, returning learners and acknowledged releases without changing progress', async () => {
 const w=browser(); try {
 initReleaseNotice(); assert(document.querySelector('#release-notice').hidden);
 assert.equal(JSON.parse(localStorage.getItem(releaseKey)), '0.2.0');
 localStorage.setItem(releaseKey, JSON.stringify('0.1.0'));
 const progress=JSON.stringify({version:1,lessons:{'welcome-01':{checkpointPassed:true,evidenceSaved:true}}});
 localStorage.setItem('jaxpathways-lessons-v1', progress);
 initReleaseNotice(); assert.equal(document.querySelector('#release-notice').hidden,false);
 document.querySelector('#dismiss-release').click(); assert(document.querySelector('#release-notice').hidden);
 initReleaseNotice(); assert(document.querySelector('#release-notice').hidden);
 assert.equal(localStorage.getItem('jaxpathways-lessons-v1'),progress);
 localStorage.removeItem(releaseKey);localStorage.setItem('jaxpathways-reading-v1',JSON.stringify({lastLessonId:'welcome-01'}));
 initReleaseNotice();assert.equal(document.querySelector('#release-notice').hidden,false);
 } finally { await w.happyDOM.abort(); }
});
test('changed revision notices survive reader enhancement and collapsed phases are not marked visited', async () => {
 const w=browser(); try {
 localStorage.setItem(revisionKey,JSON.stringify({'lessons:test':'old'}));
 const markup='<p data-revision-key="lessons:test" data-revision-hash="new"><span data-revision-notice hidden>Changed</span></p>';
 document.querySelector('#revision').innerHTML=markup;
 initRevisionNotices();assert.equal(document.querySelector('[data-revision-notice]').hidden,false);
 document.querySelector('#revision').innerHTML=markup;
 initRevisionNotices();assert.equal(document.querySelector('[data-revision-notice]').hidden,false);
 document.body.insertAdjacentHTML('beforeend','<details class="phase-card"><summary>Phase</summary><p data-revision-key="phases:test" data-revision-hash="phase"><span data-revision-notice hidden></span></p></details>');
 initRevisionNotices();assert.equal(JSON.parse(localStorage.getItem(revisionKey))['phases:test'],undefined);
 const details=document.querySelector('details'); details.open=true; details.dispatchEvent(new w.Event('toggle'));
 assert.equal(JSON.parse(localStorage.getItem(revisionKey))['phases:test'],'phase');
 localStorage.setItem(revisionKey,'broken json');assert.doesNotThrow(()=>initRevisionNotices());
 } finally { await w.happyDOM.abort(); }
});
test('a release snapshot detects changed content, retains unchanged revisions and requires a new version', () => {
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'jax-release-'));
 const write=(name,value)=>{fs.mkdirSync(path.dirname(path.join(dir,name)),{recursive:true});fs.writeFileSync(path.join(dir,name),JSON.stringify(value));};
 const run=(prepare=false)=>spawnSync(process.execPath,[path.resolve('scripts/release.mjs'),...(prepare?['--prepare']:[])],{cwd:dir,encoding:'utf8'});
 try {
 for(const name of ['projects','learning-paths'])fs.mkdirSync(path.join(dir,name));
 const lessons=['one','two'].map(id=>({id,title:id,status:'authored',path:'phases/'+id}));
 write('curriculum/course.json',{phases:lessons.map(lesson=>({id:lesson.id,title:lesson.title,lessons:[lesson]}))});
 for(const lesson of lessons)write(lesson.path+'/lesson.json',{content:{idea:'First explanation'}});
 const notes=version=>[{version,date:'2026-10-06',title:'Test release',changes:['A change']}];
 write('curriculum/releases.json',notes('0.2.0'));write('package.json',{version:'0.2.0'});
 assert.equal(run(true).status,0);assert.equal(run().status,0);
 write('phases/one/lesson.json',{content:{idea:'A clearer explanation'}});
 assert.notEqual(run().status,0);assert.notEqual(run(true).status,0);
 write('curriculum/releases.json',[...notes('0.2.1'),...notes('0.2.0')]);write('package.json',{version:'0.2.1'});
 assert.equal(run(true).status,0);
 const snapshots=JSON.parse(fs.readFileSync(path.join(dir,'curriculum/release-snapshots.json')));
 assert.deepEqual(snapshots[0].changedLessons,['one']);assert.deepEqual(snapshots[0].changedPhases,['one']);
 assert.equal(snapshots[0].lessons.two.version,'0.2.0');assert.equal(snapshots[0].lessons.one.version,'0.2.1');
 assert.equal(snapshots.length,2);assert.equal(run().status,0);
 } finally { fs.rmSync(dir,{recursive:true,force:true}); }
});

test('historical reader URLs keep deployment base, route, search and section on the permanent lesson page', async()=>{
 const { legacyLessonUrl }=await import('../src/lib/urls.js');
 assert.equal(legacyLessonUrl('arrays-02','https://example.org/jaxpathways/lesson.html?lesson=arrays-02&path=foundations&find=shape#section-3'),'https://example.org/jaxpathways/lesson-arrays-02.html?path=foundations&find=shape#section-3');
});
