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
 assert(!w.document.querySelector('meta[name="robots"]')?.content.includes('noindex'),`noindex page in sitemap: ${file}`);await w.happyDOM.close();
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
 const resource=schema['@graph'].find(item=>item['@type']==='LearningResource');
 assert(resource);assert.equal(resource.name,lesson.title);
 assert.equal(resource.url,site+base+file);
 const phaseCourse=schema['@graph'].find(item=>item['@type']==='Course');
 assert.equal(phaseCourse.name,phase.title);assert.equal(resource.isPartOf['@id'],phaseCourse['@id']);
 const canonical=site+base+file;assert(locations.includes(canonical));assert.equal(doc.querySelector('meta[property="og:url"]').content,canonical);
 await w.happyDOM.close();checked++;
}
assert(fs.readFileSync('dist/robots.txt','utf8').includes(site+base+'sitemap.xml'));
assert(!locations.some(url=>/\/(lesson|notebook|project|organizer)\.html/.test(url)));
// Every public destination must describe itself in initial HTML.
const titles=new Set();
for(const location of locations){
 const file=new URL(location).pathname.slice(base.length)||'index.html';
 const w=read(file),doc=w.document;
 assert(doc.title.trim(),`missing title: ${file}`);
 assert(!titles.has(doc.title),`duplicate title: ${file}`);titles.add(doc.title);
 assert(doc.querySelector('meta[name="description"]')?.content.trim(),`missing description: ${file}`);
 assert.equal(doc.querySelectorAll('h1').length,1,`heading count: ${file}`);
 const data=JSON.parse(doc.querySelector('script[type="application/ld+json"]')?.textContent||'null');
 assert.equal(data?.['@context'],'https://schema.org',file);
 const graph=data['@graph'];assert(Array.isArray(graph),`missing schema graph: ${file}`);
 const page=graph.find(item=>['WebPage','CollectionPage'].includes(item['@type']));
 assert.equal(page?.url,location,`WebPage URL: ${file}`);
 const author=graph.find(item=>item['@type']==='Person');assert.equal(author?.name,'Taha Bouhsine');
 assert.equal(page.author['@id'],author['@id']);
 if(file!=='index.html'){
  const breadcrumb=graph.find(item=>item['@type']==='BreadcrumbList');
  const visible=[...doc.querySelectorAll('nav[aria-label="Breadcrumb"] li')];
  assert.equal(breadcrumb?.itemListElement.length,visible.length,`breadcrumb mismatch: ${file}`);
  breadcrumb.itemListElement.forEach((item,index)=>{assert.equal(item.position,index+1);assert.equal(item.name,visible[index].textContent.trim());});
 }
 if(file.startsWith('phase-guides/')){
  const item=graph.find(item=>item['@type']==='Course');assert(item?.provider?.['@id']);
  assert(item.hasPart.length>0);assert.equal(item.url,location);
 }else if(!file.startsWith('lesson-')){
  assert(!graph.some(item=>item['@type']==='Course'),`non-course labelled Course: ${file}`);
 }
 await w.happyDOM.close();
}
const catalog=read('course.html');
const list=JSON.parse(catalog.document.querySelector('script[type="application/ld+json"]').textContent)['@graph'].find(item=>item['@type']==='ItemList');
assert.equal(list.itemListElement.length,course.phases.length);
list.itemListElement.forEach((entry,index)=>{assert.equal(entry.position,index+1);assert.equal(entry.item['@type'],'Course');assert(locations.includes(entry.item.url));});
await catalog.happyDOM.close();
for(const route of course.pathways)assert(locations.includes(site+base+`learning-paths/${route.id}.html`));
for(const role of course.roles)assert(locations.includes(site+base+`career-guides/${role.id}.html`));
const book=read('book.html');
const bookData=JSON.parse(book.document.querySelector('script[type="application/ld+json"]').textContent)['@graph'].find(item=>item['@type']==='Book');
assert(bookData?.bookEdition);assert.equal(bookData.encoding.length,2);assert(bookData.isAccessibleForFree);
assert(locations.includes(bookData.url));
for(const media of bookData.encoding)assert(fs.existsSync('dist/'+new URL(media.contentUrl).pathname.slice(base.length)));
await book.happyDOM.close();
const reader=read('downloads/jax-foundations.html');
assert.equal(reader.document.querySelector('link[rel="canonical"]')?.href,site+base+'downloads/jax-foundations.html');
assert(reader.document.querySelector('meta[name="robots"]')?.content.includes('noindex'));
assert.equal(reader.document.querySelector('meta[property="og:type"]')?.content,'book');
await reader.happyDOM.close();
console.log(`PASS: ${checked} pre-rendered lessons; ${locations.length} canonical sitemap URLs; structured data, share metadata, revisions and feedback at ${base}`);
