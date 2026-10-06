import fs from 'node:fs';
import assert from 'node:assert/strict';
import { Window } from 'happy-dom';
const base=(process.env.BASE_PATH||'/').replace(/\/$/,'')+'/';
const site=process.env.SITE_URL||'https://www.tahabouhsine.com';
const course=JSON.parse(fs.readFileSync('public/curriculum.json'));
const read=file=>{const w=new Window({settings:{disableJavaScriptEvaluation:true,disableCSSFileLoading:true,disableJavaScriptFileLoading:true}});w.document.write(fs.readFileSync('dist/'+file,'utf8'));return w;};
const sitemap=fs.readFileSync('dist/sitemap.xml','utf8');
const locations=[...sitemap.matchAll(/<loc>([^<]+)<\/loc>/g)].map(match=>match[1]);
assert.equal(new Set(locations).size,locations.length,'duplicate sitemap entries');
for(const location of locations){
 const url=new URL(location);assert.equal(url.origin,site);assert(url.pathname.startsWith(base));
 const file=url.pathname.slice(base.length)||'index.html';assert(fs.existsSync('dist/'+file),`sitemap target missing: ${file}`);
 const w=read(file);assert.equal(w.document.querySelector('link[rel="canonical"]')?.href,location,`canonical mismatch: ${file}`);
 assert(!w.document.querySelector('meta[name="robots"]')?.content.includes('noindex'),`noindex page in sitemap: ${file}`);await w.happyDOM.abort();
}
let checked=0;
for(const phase of course.phases)for(const lesson of phase.lessons.filter(l=>l.status==='authored')){
 const file=`lesson-${lesson.id}.html`,w=read(file),doc=w.document;
 assert.equal(doc.querySelectorAll('h1').length,1,file);assert.equal(doc.querySelector('h1').textContent,lesson.title);
 assert.equal(doc.querySelector('meta[name="description"]').content,lesson.objective);
 assert.equal(doc.querySelector('[data-static-lesson]')?.dataset.staticLesson,lesson.id);
 assert(doc.querySelector('.reasoning-check'),'lesson explanation missing from initial HTML');
 assert(doc.querySelector('.katex'),'math missing from initial HTML');
 assert(doc.querySelector('[data-revision-key]'),'missing lesson revision');
 assert(doc.querySelector('.lesson-feedback a')?.href.includes('context='),'missing contextual feedback');
 const schema=JSON.parse(doc.querySelector('script[type="application/ld+json"]').textContent);
 assert.equal(schema['@type'],'LearningResource');assert.equal(schema.name,lesson.title);
 const canonical=site+base+file;assert(locations.includes(canonical));assert.equal(doc.querySelector('meta[property="og:url"]').content,canonical);
 await w.happyDOM.abort();checked++;
}
assert(fs.readFileSync('dist/robots.txt','utf8').includes(site+base+'sitemap.xml'));
assert(!locations.some(url=>/\/(lesson|notebook|project|organizer)\.html/.test(url)));
console.log(`PASS: ${checked} pre-rendered lessons; ${locations.length} canonical sitemap URLs; structured data, share metadata, revisions and feedback at ${base}`);
