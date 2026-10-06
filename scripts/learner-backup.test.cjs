const {test}=require('node:test');
const assert=require('node:assert/strict');
const {normalizeLearnerBackup,mergeLearnerBackup,persistLearnerBackup,validateReadingState}=require('../src/lib/learner-records.js');
const context={routeIds:['foundations','models'],lessonIds:['welcome-01','networks-01'],roleIds:['training-engineer'],phaseIds:['welcome','networks']};
const base=()=>({version:2,notebook:{version:1,route:'foundations',evidence:[]},sampleCompleted:false,lessonProgress:{version:1,lessons:{}},projects:{},careerPlan:null,reading:null});
const reading=(date,id='welcome-01')=>({version:1,lastLessonId:id,path:'foundations',sectionId:'section-4',updatedAt:date,lessons:{[id]:{sectionId:'section-4',updatedAt:date}}});
test('legacy version-1 backup merges checkpoints without erasing unrelated project, career or reading state',()=>{
 const current=base();current.projects={'mlp-classifier':['1']};current.careerPlan={version:1,roleId:'training-engineer',pathwayId:'models',startPhaseId:'networks',knownPhaseIds:['welcome']};current.reading=reading('2026-10-03T01:00:00Z');
 const merged=mergeLearnerBackup(current,{version:1,route:'models',evidence:[],sampleCompleted:true,lessonProgress:{version:1,lessons:{'welcome-01':{checkpointPassed:true,evidenceSaved:false}}}},context);
 assert.deepEqual(merged.projects,current.projects);assert.deepEqual(merged.careerPlan,current.careerPlan);assert.deepEqual(merged.reading,current.reading);assert.equal(merged.lessonProgress.lessons['welcome-01'].evidenceSaved,false);assert.equal(merged.sampleCompleted,true);
});
test('version-2 roundtrip preserves project stages, evidence associations, edited records and latest reading position',()=>{
 const current=base();current.projects={'mlp-classifier':['1']};current.reading=reading('2026-10-03T01:00:00Z');current.notebook.evidence=[{id:'artifact',route:'models',title:'Old title',note:'Old result',url:'',date:'2026-10-03',updatedAt:'2026-10-03T01:00:00Z'}];
 const imported=base();imported.notebook.route='models';imported.projects={'mlp-classifier':['2','1']};imported.reading=reading('2026-10-03T02:00:00Z','networks-01');imported.notebook.evidence=[{...current.notebook.evidence[0],title:'Fixed result',updatedAt:'2026-10-03T02:00:00Z',lessonId:'networks-01',projectId:'mlp-classifier',stageId:'2'}];
 const merged=mergeLearnerBackup(current,JSON.parse(JSON.stringify(imported)),context);
 assert.deepEqual(merged.projects['mlp-classifier'],['1','2']);assert.equal(merged.notebook.evidence.length,1);assert.equal(merged.notebook.evidence[0].title,'Fixed result');assert.equal(merged.notebook.evidence[0].stageId,'2');assert.equal(merged.reading.lastLessonId,'networks-01');assert.ok(merged.reading.lessons['welcome-01']);assert.equal(current.notebook.evidence[0].title,'Old title');
});
test('invalid state is rejected before persistence rather than writing arbitrary keys',()=>{
 for(const mutation of [b=>b.projects={'../../danger':['1']},b=>b.reading=reading('not a date'),b=>b.notebook.evidence=[{id:'x',route:'models',title:'x',note:'x',url:'javascript:alert(1)',date:'2026-10-03'}],b=>b.lessonProgress={version:1,lessons:{'missing':{checkpointPassed:true,evidenceSaved:true}}}]){const backup=base();mutation(backup);assert.throws(()=>normalizeLearnerBackup(backup,context));}
 assert.equal(validateReadingState({...reading('2026-10-03T00:00:00Z'),sectionId:'<script>'},context),false);
});
test('failed multi-key persistence reports failure and restores earlier values where storage permits',()=>{
 const values=new Map([['jaxpathways-notebook-v1','old notebook'],['jaxpathways-lessons-v1','old progress']]);
 const storage={getItem:k=>values.get(k)??null,setItem:(k,v)=>{if(k==='jaxpathways-project-mlp-classifier')throw Error('quota');values.set(k,v);},removeItem:k=>values.delete(k)};
 const incoming=base();incoming.projects={'mlp-classifier':['1']};
 assert.equal(persistLearnerBackup(storage,incoming),false);assert.equal(values.get('jaxpathways-notebook-v1'),'old notebook');assert.equal(values.get('jaxpathways-lessons-v1'),'old progress');assert.equal(values.has('jaxpathways-first-gradient'),false);
});
