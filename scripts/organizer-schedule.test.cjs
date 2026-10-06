const {test}=require('node:test');
const assert=require('node:assert/strict');
const {planLessonSchedule}=require('../src/lib/schedule.js');
test('organizer budgets real lesson minutes rather than dividing phase counts',()=>{
 const lessons=[{id:'a',minutes:40},{id:'b',minutes:45},{id:'c',minutes:90},{id:'d',minutes:30}];
 const result=planLessonSchedule(lessons,2,120);
 assert.deepEqual(result.assignments.map(list=>list.map(l=>l.id)),[['a','b'],['c','d']]);assert.deepEqual(result.used,[85,120]);assert.equal(result.total,205);assert.equal(result.capacity,240);assert.deepEqual(result.overflow,[]);
});
test('a lesson that cannot fit blocks later lessons rather than skipping a prerequisite',()=>{
 const lessons=[{id:'a',minutes:40},{id:'b',minutes:200},{id:'c',minutes:20}];
 const result=planLessonSchedule(lessons,2,120);
 assert.deepEqual(result.assignments.map(list=>list.map(l=>l.id)),[['a'],[]]);assert.deepEqual(result.overflow.map(l=>l.id),['b','c']);assert.equal(result.total,260);
 assert.throws(()=>planLessonSchedule(lessons,0,120));assert.throws(()=>planLessonSchedule(lessons,4,-5));
});
test('summary separates unused capacity, blocked available work and unwritten topics',()=>{
 const {scheduleMetrics}=require('../src/lib/schedule.js');
 const plan=planLessonSchedule([{id:'a',minutes:40},{id:'b',minutes:200},{id:'c',minutes:20}],2,120);
 assert.deepEqual(scheduleMetrics(plan,3),{assignedHours:0.7,capacityHours:4,assignedLessons:1,remainingLessons:2,remainingHours:3.7,unwritten:3});
 assert.deepEqual(scheduleMetrics(planLessonSchedule([],4,180)),{assignedHours:0,capacityHours:12,assignedLessons:0,remainingLessons:0,remainingHours:0,unwritten:0});
});
