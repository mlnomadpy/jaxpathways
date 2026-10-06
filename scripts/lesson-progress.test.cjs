const {test}=require('node:test');
const assert=require('node:assert/strict');
const {validateLessonProgress,mergeLessonProgress}=require('../src/lib/learner-records.js');
const ids=['first-gradient','state-01'];
test('checkpoint success never implies exercise evidence',()=>{
  const state=mergeLessonProgress({version:1,lessons:{}},{version:1,lessons:{'first-gradient':{checkpointPassed:true,evidenceSaved:false}}},ids);
  assert.equal(state.lessons['first-gradient'].evidenceSaved,false);
});
test('import preserves existing evidence while adding checkpoints',()=>{
  const current={version:1,lessons:{'first-gradient':{checkpointPassed:false,evidenceSaved:true}}};
  const incoming={version:1,lessons:{'first-gradient':{checkpointPassed:true,evidenceSaved:false},'state-01':{checkpointPassed:false,evidenceSaved:true}}};
  const merged=mergeLessonProgress(current,incoming,ids);
  assert.deepEqual(merged.lessons['first-gradient'],{checkpointPassed:true,evidenceSaved:true});
  assert.equal(merged.lessons['state-01'].checkpointPassed,false);
  assert.equal(current.lessons['first-gradient'].checkpointPassed,false);
});
test('malformed and unknown lesson progress is rejected before merging',()=>{
  for(const value of [{version:2,lessons:{}},{version:1,lessons:[]},{version:1,lessons:{'unknown':{checkpointPassed:true,evidenceSaved:false}}},{version:1,lessons:{'first-gradient':{checkpointPassed:'yes',evidenceSaved:false}}}]){
    assert.ok(!validateLessonProgress(value,ids));
    assert.throws(()=>mergeLessonProgress({version:1,lessons:{}},value,ids));
  }
});
