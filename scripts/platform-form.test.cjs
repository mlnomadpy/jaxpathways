const {test}=require('node:test');
const assert=require('node:assert/strict');
const {setupPlatform}=require('../src/scripts/workspace.js');
const {courseState}=require('../src/lib/course-state.js');
// Exercise the real submit handler with an isolated DOM/storage fixture; no user records.
async function harness(failStorage){
 const nodes=new Map();
 const element=()=>({value:'',textContent:'',innerHTML:'',hidden:false,children:[],addEventListener(type,listener){this.listeners ||= {}; (this.listeners[type] ||= []).push(listener);},append(...values){this.children.push(...values);},replaceChildren(...values){this.children=values;},setAttribute(){},focus(){},reset(){for(const id of ['#evidence-title','#evidence-url','#evidence-note'])nodes.get(id).value='';}});
 for(const id of ['active-route','continue','evidence-route','evidence-lesson','evidence-project','evidence-stage','evidence-title','evidence-url','evidence-note','evidence-submit','evidence-status','evidence-form','cancel-edit','evidence-filter-route','evidence-filter-type','undo-remove','evidence-count','evidence-results','evidence-list','backup-status','export','import','import-button','route-next','lesson-progress-summary','project-progress-summary','saved-career'])nodes.set('#'+id,element());
 const lesson={id:'welcome-01',title:'A fixture lesson',status:'authored'},phase={id:'welcome',title:'Welcome',lessons:[lesson]};
 const storage=new Map();
 const context={console,Date,Map,Set,JSON,Array,Number,Boolean,String,URLSearchParams,crypto:{randomUUID:()=> 'isolated-test-id'},document:{createElement:element},location:{search:''},localStorage:{getItem:k=>storage.get(k)??null,setItem:(k,v)=>{if(failStorage)throw Error('quota');storage.set(k,v);},removeItem:k=>storage.delete(k),get length(){return storage.size;},key:i=>[...storage.keys()][i]},fetch:async()=>({ok:true,json:async()=>({projects:[]})}),course:{phases:[phase],roles:[]},phaseById:id=>id==='welcome'?phase:null,lessonProgress:{version:1,lessons:{}},completed:false,$:selector=>nodes.get(selector)||null,escapeHtml:s=>s};
 context.document.querySelector=selector=>nodes.get(selector)||null;
 for(const key of ['document','location','localStorage','fetch'])globalThis[key]=context[key];
 Object.assign(courseState,{course:context.course,lessonProgress:context.lessonProgress,completed:false});
 await setupPlatform([{id:'foundations',title:'Foundations',outcome:'Learn the fixture',lessons:[lesson]}]);
 nodes.get('#evidence-title').value='My experiment';nodes.get('#evidence-note').value='Observed an independently verified result.';
 return {nodes,storage,addRestorePreview:()=>{const preview=element();nodes.set('#restore-preview',preview);return preview;},submit:()=>nodes.get('#evidence-form').onsubmit({preventDefault(){},target:nodes.get('#evidence-form')})};
}
test('failed artifact save retains inputs and retry edits the same in-memory record',async()=>{
 const {nodes,storage,submit}=await harness(true);submit();
 assert.equal(nodes.get('#evidence-title').value,'My experiment');assert.match(nodes.get('#evidence-status').textContent,/inputs are retained/);assert.equal(nodes.get('#evidence-submit').textContent,'Retry saving');assert.equal(storage.size,0);
 submit();assert.equal(nodes.get('#evidence-count').textContent,'1 artifact · self-reported; not reviewed');assert.equal(nodes.get('#evidence-note').value,'Observed an independently verified result.');
});
test('successful artifact save persists once, clears inputs and leaves learning milestones untouched',async()=>{
 const {nodes,storage,submit}=await harness(false);submit();
 const saved=JSON.parse(storage.get('jaxpathways-notebook-v1'));assert.equal(saved.evidence.length,1);assert.equal(saved.evidence[0].title,'My experiment');assert.equal(nodes.get('#evidence-title').value,'');assert.match(nodes.get('#evidence-status').textContent,/saved on this device/);
 assert.equal(storage.has('jaxpathways-lessons-v1'),false);assert.equal(storage.has('jaxpathways-first-gradient'),false);
});

// The restore preview must validate first and write only after the merge action.
test('backup preview and cancellation leave stored records untouched',async()=>{
 const {nodes,storage,addRestorePreview}=await harness(false);const preview=addRestorePreview();
 const incoming={version:1,route:'foundations',sampleCompleted:false,evidence:[{id:'restored-work',route:'foundations',title:'Restored experiment',url:'',note:'Kept observed output.',date:'2026-10-04'}]};
 await nodes.get('#import').onchange({target:{files:[{name:'fixture.json',size:200,text:async()=>JSON.stringify(incoming)}],value:'fixture.json'}});
 assert.equal(storage.size,0);assert.equal(preview.hidden,false);assert.match(nodes.get('#backup-status').textContent,/Choose Restore and merge/);
 preview.children.find(node=>node.textContent==='Cancel').onclick();
 assert.equal(preview.hidden,true);assert.equal(storage.size,0);
});
test('confirmed backup merge updates the notebook and rejects malformed follow-up files',async()=>{
 const {nodes,storage,addRestorePreview}=await harness(false);const preview=addRestorePreview();
 const incoming={version:1,route:'foundations',sampleCompleted:false,evidence:[{id:'restored-work',route:'foundations',title:'Restored experiment',url:'',note:'Kept observed output.',date:'2026-10-04'}]};
 await nodes.get('#import').onchange({target:{files:[{name:'fixture.json',size:200,text:async()=>JSON.stringify(incoming)}],value:'fixture.json'}});
 preview.children.find(node=>node.textContent==='Restore and merge').onclick();
 const saved=storage.get('jaxpathways-notebook-v1');assert.equal(JSON.parse(saved).evidence[0].title,'Restored experiment');assert.match(nodes.get('#backup-status').textContent,/merged and saved/);
 await nodes.get('#import').onchange({target:{files:[{name:'bad.json',size:10,text:async()=>'{invalid'}],value:'bad.json'}});
 assert.equal(storage.get('jaxpathways-notebook-v1'),saved);assert.match(nodes.get('#backup-status').textContent,/was not changed/);
});
