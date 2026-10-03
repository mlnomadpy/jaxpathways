import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
const course = JSON.parse(readFileSync(new URL('../site/curriculum.json', import.meta.url)));
assert.equal(course.schemaVersion, 2);
const phases = new Map(course.phases.map(p => [p.id,p]));
assert.equal(phases.size, course.phases.length, 'Phase IDs must be unique');
const lessons = new Map();
for (const [i,phase] of course.phases.entries()) {
  assert.equal(phase.number, String(i).padStart(2,'0'));
  for (const field of ['title','part','hardware','project','checkpoint','source']) assert.ok(phase[field]?.length, `${phase.id}: ${field} is required`);
  for (const dep of phase.prerequisitePhaseIds) assert.ok(phases.has(dep), `Unknown prerequisite ${dep}`);
  for (const [j,lesson] of phase.lessons.entries()) {
    assert.ok(!lessons.has(lesson.id), `Duplicate canonical lesson ${lesson.id}`);
    lessons.set(lesson.id,lesson);
    for (const field of ['title','objective','exercise','evidence','check']) assert.ok(lesson[field]?.length, `${lesson.id}: ${field} is required`);
    assert.ok(['sample','planned'].includes(lesson.status));
    assert.deepEqual(lesson.prerequisites,j?[phase.lessons[j-1].id]:[], 'Lesson order must be explicit');
  }
}
const visiting = new Set(), visited = new Set();
function visit(id) {
  assert.ok(!visiting.has(id), `Prerequisite cycle at ${id}`);
  if(visited.has(id))return;
  visiting.add(id);
  for(const dep of phases.get(id).prerequisitePhaseIds)visit(dep);
  visiting.delete(id);visited.add(id);
}
for(const id of phases.keys())visit(id);
const routes = new Set();
for(const route of course.pathways) {
  assert.ok(!routes.has(route.id));routes.add(route.id);
  assert.ok(!('lessons' in route),'Pathways must reference canonical phases, not copy lessons');
  const seen = new Set();
  for(const id of route.phaseIds) {
    assert.ok(phases.has(id),`${route.id}: unknown phase ${id}`);
    assert.ok(!seen.has(id),`${route.id}: repeated phase ${id}`);
    for(const dep of phases.get(id).prerequisitePhaseIds)assert.ok(seen.has(dep),`${route.id}: ${id} needs ${dep} earlier in the route`);
    seen.add(id);
  }
  assert.ok(route.capstone.title && route.capstone.assessment && route.capstone.evidence.length);
}
assert.equal([...lessons.values()].filter(l=>l.status==='sample').length,1);
assert.ok(lessons.has('first-gradient'));
for(const notebook of course.notebookIntegration)assert.ok(phases.has(notebook.phaseId));
console.log(`OK: ${phases.size} phases, ${lessons.size} canonical lesson briefs, ${routes.size} routes; prerequisites acyclic and ordered.`);
console.log('Content status: 1 authored sample; all remaining lessons, projects, and assessments planned.');
const roleIds=new Set();
for(const role of course.roles){
  assert.ok(!roleIds.has(role.id));roleIds.add(role.id);
  assert.ok(role.pathwayIds.includes(role.defaultPathwayId));
  for(const id of role.pathwayIds)assert.ok(routes.has(id),`Unknown route for ${role.id}: ${id}`);
  assert.ok(role.responsibilities.length&&role.skills.length&&role.portfolio&&role.readiness);
}
const coveredRoutes=new Set(),coveredRoles=new Set(),domainIds=new Set();
for(const domain of course.domains){
  assert.ok(!domainIds.has(domain.id));domainIds.add(domain.id);
  assert.ok(domain.pathwayIds.includes(domain.defaultPathwayId));
  for(const id of domain.pathwayIds){assert.ok(routes.has(id));coveredRoutes.add(id);}
  for(const id of domain.roleIds){assert.ok(roleIds.has(id));coveredRoles.add(id);}
  assert.ok(domain.skills.length&&domain.artifact);
}
for(const id of routes)assert.ok(id==='foundations'||coveredRoutes.has(id),`Route missing a core domain: ${id}`);
for(const id of roleIds)assert.ok(coveredRoles.has(id),`Role missing a core domain: ${id}`);
console.log(`OK: ${domainIds.size} core domains, ${roleIds.size} career routes; every specialization and role has a home.`);
