import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { Window } from 'happy-dom';

const base = (process.env.BASE_PATH || '/').replace(/\/$/, '') + '/';
const origin = 'https://example.org';
const files = fs.readdirSync('dist', {recursive:true}).filter(file=>file.endsWith('.html'));
const expected=['updates',...JSON.parse(fs.readFileSync('curriculum/course.json','utf8')).phases.flatMap(p=>p.lessons).filter(l=>l.status==='authored').map(l=>'lesson-'+l.id),...JSON.parse(fs.readFileSync('curriculum/project-plans.json','utf8')).map(project=>'project-plans/'+project.id),...JSON.parse(fs.readFileSync('curriculum/course.json','utf8')).phases.map(p=>'phase-guides/'+p.id),'index','course','lesson','pathways','careers','notebook','organizer','project','projects','developer','project-workflow',...JSON.parse(fs.readFileSync('curriculum/projects.json','utf8')).map(manifest=>'project-guides/'+JSON.parse(fs.readFileSync(manifest,'utf8')).id),...JSON.parse(fs.readFileSync('curriculum/assessments.json','utf8')).map(assessment=>assessment.url.replace(/\.html$/,'')),'downloads/jax-foundations', ...JSON.parse(fs.readFileSync('curriculum/modality-tracks.json','utf8')).tracks.map(track=>'tracks/'+track.id)];
assert.deepEqual([...files].sort(),expected.map(file=>file+'.html').sort(),'all website routes must be built by Astro');
const course=JSON.parse(fs.readFileSync('public/curriculum.json','utf8'));
const projectRegistry=JSON.parse(fs.readFileSync('public/api/v1/projects.json','utf8')).projects;
const documentCache=new Map();
function documentFor(file){
 if(documentCache.has(file))return documentCache.get(file).document;
 const window=new Window({settings:{disableJavaScriptEvaluation:true,disableCSSFileLoading:true,disableJavaScriptFileLoading:true}});
 window.document.write(fs.readFileSync('dist/'+file,'utf8'));documentCache.set(file,window);return window.document;
}
// Release parsed page trees as we go; caching every chapter exceeds small CI heaps.
async function releaseDocuments(){
 await Promise.all([...documentCache.values()].map(window=>window.happyDOM.close()));
 documentCache.clear();
}
let checked=0;
function checkLink(raw,from){
 if(!raw||/^(?:https?:|mailto:|data:|tel:)/.test(raw))return;
 const url=new URL(raw,origin+base+from);
 assert.equal(url.origin,origin);
 assert(url.pathname.startsWith(base),`link escaped deployment base in ${from}: ${raw}`);
 const target=url.pathname.slice(base.length)||'index.html';
 assert(fs.existsSync('dist/'+target),`missing target in ${from}: ${raw}`);
 if(url.hash&&target.endsWith('.html'))assert(documentFor(target).getElementById(decodeURIComponent(url.hash.slice(1))),`missing anchor in ${from}: ${raw}`);
 if(url.searchParams.has('lesson'))assert(course.phases.some(p=>p.lessons.some(l=>l.id===url.searchParams.get('lesson'))),`unknown lesson ${raw}`);
 if(url.searchParams.has('path'))assert(course.pathways.some(p=>p.id===url.searchParams.get('path')),`unknown path ${raw}`);
 if(target==='project.html'&&url.searchParams.has('id'))assert(projectRegistry.some(project=>project.id===url.searchParams.get('id')),`unknown project ${raw}`);
 if(url.searchParams.has('role'))assert(course.roles.some(r=>r.id===url.searchParams.get('role')),`unknown role ${raw}`);
 checked++;
}
for(const file of files){
 const document=documentFor(file),ids=new Set();
 for(const node of document.querySelectorAll('[id]')){assert(!ids.has(node.id),`duplicate id ${node.id} in ${file}`);ids.add(node.id);}
 assert(document.title.trim(),`missing title: ${file}`);
 if(!file.startsWith('downloads/'))assert(document.querySelector('header nav a[href$="projects.html"]'),'Projects navigation missing: '+file);
 for(const node of document.querySelectorAll('[href],[src],[srcset]'))for(const attr of ['href','src','srcset']){const raw=node.getAttribute(attr);if(raw)checkLink(raw,file);}
 for(const node of document.querySelectorAll('[aria-labelledby],[aria-describedby],[aria-controls]'))for(const attr of ['aria-labelledby','aria-describedby','aria-controls'])for(const id of (node.getAttribute(attr)||'').split(/\s+/).filter(Boolean))assert(ids.has(id),`missing accessibility target ${id} in ${file}`);
 await releaseDocuments();
}
for(const file of fs.readdirSync('dist/_astro').filter(file=>file.endsWith('.css'))){
 const css=fs.readFileSync('dist/_astro/'+file,'utf8');
 for(const match of css.matchAll(/url\(["']?([^\s)'";]+)["']?\)/g))checkLink(match[1],'_astro/'+file);
}
assert.equal(documentFor('course.html').querySelectorAll('.lesson-link').length,course.phases.flatMap(p=>p.lessons).length);
assert.deepEqual([...documentFor('projects.html').querySelectorAll('[data-project-id]')].map(node=>node.dataset.projectId).sort(),projectRegistry.map(project=>project.id).sort(),'Projects must include every canonical project exactly once');
assert.equal(documentFor('careers.html').querySelectorAll('.role-card').length,course.roles.length);
assert.equal(documentFor('index.html').querySelectorAll('.lesson-preview li').length,6);
await releaseDocuments();
assert.equal(documentFor('downloads/jax-foundations.html').querySelectorAll('article').length,course.phases.flatMap(phase=>phase.lessons).filter(lesson=>lesson.status==='authored').length);
const printedMath=[...documentFor('downloads/jax-foundations.html').querySelectorAll('annotation[encoding="application/x-tex"]')].map(node=>node.textContent);
for(const phase of course.phases)for(const lesson of phase.lessons)for(const section of lesson.content?.sections||[])if(section.math)assert(printedMath.includes(section.math),`printable book omitted equation in ${lesson.id}`);
for(const file of fs.readdirSync('public',{recursive:true}))assert(!/\.(html|js|css)$/.test(file),`authored code/HTML escaped Astro build in public: ${file}`);
for(const name of ['style.css','design.css','identity.css','refinements.css','experience.css','app.js'])assert(!fs.existsSync(path.join('dist',name)),`legacy source shipped: ${name}`);
await releaseDocuments();
console.log(`PASS: ${files.length} Astro pages, ${checked} local references, accessibility IDs, static catalogs and clean public boundary at ${base}`);

for(const phase of course.phases){
 const doc=documentFor(`phase-guides/${phase.id}.html`),g=phase.studyGuide;
 assert(doc.querySelector('h1').textContent.includes(g.question),phase.id+': missing guide heading');
 for(const id of ['readiness','milestones','transfer','completion'])assert(doc.getElementById(id),phase.id+': missing guide section');
 for(const lesson of phase.lessons)assert([...doc.querySelectorAll('a')].some(a=>a.getAttribute('href')===base+'lesson.html?lesson='+lesson.id),phase.id+': lesson omitted from guide');
 for(const value of [g.completion,g.furtherWork,g.transfer.approach])assert(doc.body.textContent.includes(value),phase.id+': missing phase evidence or limits');
 assert.equal(doc.querySelectorAll('input,select,textarea').length,0,phase.id+': study guide should not require form entry');
 await releaseDocuments();
}
console.log(`PASS: ${course.phases.length} phase guides preserve lesson coverage, evidence and further-work boundaries.`);
