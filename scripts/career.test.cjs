const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const {careerStartingPhase,careerPlanMarkdown,careerCoverage}=require('../src/lib/careers.js');
const course=JSON.parse(fs.readFileSync(require('node:path').join(__dirname,'../public/curriculum.json')));
const role=course.roles.find(r=>r.id==='operations-engineer');
test('an unknown foundation is recommended before a claimed advanced capability',()=>{
  assert.equal(careerStartingPhase(role,['welcome','transforms','state','optimization'],course),'arrays');
});
test('a learner who knows the foundations starts with the first required applied phase',()=>{
  assert.equal(careerStartingPhase(role,role.startingPointChecks.map(c=>c.phaseId),course),'networks');
});
test('new learners start at setup regardless of their career choice',()=>{
  for(const candidate of course.roles)assert.equal(careerStartingPhase(candidate,[],course),'welcome');
});
test('the exported plan retains the full prerequisite route and verification criteria',()=>{
  const result=careerPlanMarkdown(role,course,'networks');
  assert.ok(result.includes('00: Setup & first steps'));
  assert.ok(result.includes('16: Workload operations'));
  assert.ok(result.includes(role.readiness));
  assert.ok(result.includes(role.milestones.at(-1).criteria[0]));
  assert.ok(result.includes('not been graded'));
});
test('career counts distinguish available foundations from missing specialist material',()=>{
  const scientific=course.pathways.find(r=>r.id==='science');
  const coverage=careerCoverage(scientific.phaseIds,course);
  assert.ok(coverage.available>0);
  assert.equal(coverage.appliedAvailable,course.phases.find(p=>p.id==='science').lessons.filter(l=>l.status==='authored').length);
  assert.equal(coverage.appliedPlanned,course.phases.find(p=>p.id==='science').lessons.filter(l=>l.status!=='authored').length);
  const training=careerCoverage(['networks'],course);
  assert.equal(training.available,5);
  assert.equal(training.planned,0);
  assert.deepEqual(training.projects,['mlp-classifier']);
});
