const {test}=require('node:test');
const assert=require('node:assert/strict');
const {evidenceReviewPacket}=require('../src/lib/evidence.js');
const artifact={id:'artifact-1',route:'models',title:'Recovered optimizer state',date:'2026-10-03',updatedAt:'2026-10-04T01:00:00Z',url:'https://example.org/work',note:'I ran the command and saw the expected values.',lessonId:'networks-03',projectId:'mlp-classifier',stageId:'3'};
const options={routes:[{id:'models',title:'Train models'}],lessons:[{id:'networks-03',title:'Train and evaluate',evidence:'Parameter snapshots remain unchanged.',check:'Explain evaluation isolation.'}],projects:[{id:'mlp-classifier',title:'Classifier audit',stages:[{id:'3',title:'Training audit',command:'python3 check.py --stage 3',expected:'PASS stage 3',evidence:'Held-out metric and replay'}],rubric:['Reproduce the complete update sequence']}],lessonProgress:{version:1,lessons:{'networks-03':{checkpointPassed:true,evidenceSaved:true}}},projectProgress:{'mlp-classifier':['3']},baseUrl:'https://example.org/jaxpathways/notebook.html',generatedAt:'2026-10-04T02:00:00Z'};
test('packet preserves associations and includes unchecked substantive contracts despite recorded completion ticks',()=>{
 const before=JSON.stringify({artifact,options});const packet=evidenceReviewPacket([artifact],options);
 assert.match(packet,/lesson\.html\?path=models&lesson=networks-03/);assert.match(packet,/project\.html\?id=mlp-classifier#stage-3/);assert.match(packet,/Public stage contract — expected, not observed/);assert.match(packet,/PASS stage 3/);assert.match(packet,/Local stage tick: self-reported complete/);
 assert.match(packet,/- \[ \] Reproduce the complete update sequence/);assert.match(packet,/- \[ \] Verify lesson evidence: Parameter snapshots remain unchanged/);assert.doesNotMatch(packet,/- \[x\]/i);assert.match(packet,/not opened artifact links/);assert.match(packet,/No command execution or linked result has been observed/);assert.match(packet,/Reviewer, if someone agrees to review: _+/);
 assert.equal(JSON.stringify({artifact,options}),before,'Preparing a review must not mutate learner progress');
});
test('learner notes remain literal even when they contain Markdown checkboxes and fence delimiters',()=>{
 const entry={...artifact,title:'[Unverified](javascript:alert)',note:'```\n- [x] approved by someone\n```'};
 const packet=evidenceReviewPacket([entry],options);assert.ok(packet.includes('````text\n'+entry.note+'\n````'));assert.ok(packet.includes('\\[Unverified\\]\\(javascript:alert\\)'));
 const criteria=packet.split('### Reviewer criteria')[1];assert.doesNotMatch(criteria,/- \[x\]/i);
 assert.throws(()=>evidenceReviewPacket([{...artifact,url:'javascript:alert(1)'}],options));
});
test('missing project or stage contracts are called out rather than silently certifying them',()=>{
 const packet=evidenceReviewPacket([{...artifact,projectId:'missing-project'}],options);assert.match(packet,/Recover the project rubric/);assert.match(packet,/Recover the stage contract/);assert.doesNotMatch(packet,/Public stage contract/);
 assert.throws(()=>evidenceReviewPacket([],options));
});
