import { markdownText } from './lesson-artifacts.mjs';
import { inlineMath, renderMath } from '../src/lib/math.js';
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
const course = JSON.parse(readFileSync(new URL('../public/curriculum.json', import.meta.url)));
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
    assert.ok(['authored','planned'].includes(lesson.status));
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
const authored=[...lessons.values()].filter(l=>l.status==='authored');
for(const lesson of authored){
  const c=lesson.content;
  function validateMath(value,key=''){
    if(['code','solution','buildCode','formula'].includes(key))return;
    if(typeof value==='string'){if(key==='math')renderMath(value,true);else inlineMath(value);}
    else if(Array.isArray(value))value.forEach(item=>validateMath(item,key));
    else if(value&&typeof value==='object')Object.entries(value).forEach(([k,v])=>validateMath(v,k));
  }
  validateMath(c);
  assert.ok(c&&c.idea&&c.code&&c.output&&c.exercise&&c.solution&&c.explanation);
  assert.ok(Number.isInteger(c.answer)&&c.answer>=0&&c.answer<c.options.length);
  assert.ok(lesson.minutes>0);
  for(const [kind,path] of Object.entries(lesson.artifacts)){assert.ok(!path.includes('..'));const base=(kind==='source'||kind.endsWith('Source'))?'../':'../public/';readFileSync(new URL(base+path,import.meta.url));}
  const notebook=JSON.parse(readFileSync(new URL('../public/'+lesson.artifacts.notebook,import.meta.url)));
  assert.equal(notebook.nbformat,4);
  const code=notebook.cells.filter(c=>c.cell_type==='code').map(c=>c.source.join('')).join('\n');
  assert.ok((!c.buildCode||code.includes(c.buildCode))&&code.includes(c.code)&&code.includes(c.solution),`${lesson.id}: notebook is out of sync`);
  const script=readFileSync(new URL('../public/'+lesson.artifacts.script,import.meta.url),'utf8');assert.ok(script.includes(c.code)&&script.includes(c.solution),`${lesson.id}: script is out of sync`);
  const markdown=readFileSync(new URL('../'+lesson.artifacts.source,import.meta.url),'utf8');
  for(const field of (c.setupBundle?['problem','objectives','sections','diagnosis','references']:['problem','objectives','sections','experiments','practice','diagnosis','references']))assert.ok(c[field]?.length,`${lesson.id}: missing required teaching component ${field}`);
  for(const step of c.buildSteps||[]){
    assert.ok(step.title&&step.instruction&&step.code&&step.explanation,`${lesson.id}: incomplete build step`);
    assert.ok(code.includes(step.code)&&script.includes(step.code)&&markdown.includes(step.code),`${lesson.id}: staged build omitted from companion`);
  }
  for(const section of c.sections||[])assert.ok(section.title&&section.body,`${lesson.id}: concept section incomplete`);
  for(const experiment of c.experiments||[]) {
    assert.ok(experiment.title&&experiment.prediction&&experiment.code&&experiment.output&&experiment.explanation);
    assert.ok(code.includes(experiment.code)&&script.includes(experiment.code),`${lesson.id}: worked experiment missing from executable companions`);
  }
  for(const practice of c.practice||[]) {
    assert.ok(practice.title&&practice.difficulty&&practice.prompt&&practice.hint&&practice.solution&&practice.explanation);
    assert.ok(code.includes(practice.solution)&&script.includes(practice.solution),`${lesson.id}: practice solution missing from executable companions`);
  }
  for(const section of c.sections||[])assert.ok(markdown.includes(markdownText(section.body)),`${lesson.id}: narrative omitted from Markdown`);
  for(const section of c.sections||[])for(const cmd of section.commands||[]){assert.ok(cmd.label&&cmd.shell&&cmd.code&&cmd.expected);assert.ok(markdown.includes(cmd.code)&&markdown.includes(markdownText(cmd.expected)),`${lesson.id}: terminal instruction omitted`);}
  for(const section of c.sections||[])if(section.math){assert.ok(markdown.includes('$$\n'+section.math+'\n$$'),`${lesson.id}: math omitted from Markdown`);assert.ok(notebook.cells.some(cell=>cell.cell_type==='markdown'&&cell.source.join('').includes(section.math)),`${lesson.id}: math omitted from notebook`);}
  for(const objective of c.objectives||[])assert.ok(markdown.includes(markdownText(objective)),`${lesson.id}: objective omitted from Markdown`);

}
assert.ok(lessons.has('first-gradient'));
for(const notebook of course.notebookIntegration)assert.ok(phases.has(notebook.phaseId));
console.log(`OK: ${phases.size} phases, ${lessons.size} canonical lesson briefs, ${routes.size} routes; prerequisites acyclic and ordered.`);
console.log(`Content status: ${authored.length} authored lessons with script/notebook companions; ${lessons.size-authored.length} planned. ${course.projectIds?.length||1} staged integration projects and synthesis assessments are linked across all routes.`);
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
for(const role of course.roles){
  const route=course.pathways.find(r=>r.id===role.defaultPathwayId);
  assert.ok(role.background && role.milestones.length && role.startingPointChecks.length);
  for(const milestone of role.milestones){
    assert.ok(milestone.title&&milestone.deliverable&&milestone.criteria.length);
    for(const id of milestone.phaseIds)assert.ok(route.phaseIds.includes(id),`${role.id}: milestone outside default route: ${id}`);
  }
  assert.deepEqual(role.startingPointChecks.map(c=>c.phaseId),route.phaseIds.slice(0,5),'Starting checks must follow the route foundation');
}
console.log('OK: career milestones and starting-point checks reference their required route phases.');
for(const project of JSON.parse(readFileSync(new URL('../public/api/v1/projects.json',import.meta.url))).projects){
  assert.ok(routes.has(project.pathwayId),`${project.id}: unknown project pathway ${project.pathwayId}`);
  for(const id of project.requiredLessonIds)assert.ok(authored.some(l=>l.id===id),`${project.id}: unavailable project prerequisite ${id}`);
}

// Cross-cutting project plans must point to real preparation, not invented lessons.
const trackIds = new Set(course.modalityTracks.map(track=>track.id));
assert.equal(trackIds.size,course.modalityTracks.length);
for(const track of course.modalityTracks){
  assert.ok(['guided-plan','authored'].includes(track.status));
  if(track.status==='authored'){
    assert(track.harnessProjectId && track.projectIds.includes(track.harnessProjectId));
    const project=JSON.parse(readFileSync(new URL(`../projects/${track.harnessProjectId}/project.json`,import.meta.url)));
    assert.equal(project.status,'authored');
    for(const stage of track.stages){
      assert(stage.projectStageIds?.length,track.id+': missing runnable stage mapping');
      for(const id of stage.projectStageIds)assert(project.stages.some(s=>s.id===id),track.id+': unknown project stage '+id);
    }
  } else assert(!track.harnessProjectId);
  assert.equal(lessons.get(track.firstLessonId)?.status,'authored');
  assert.equal(new Set(track.stages.map(stage=>stage.id)).size,8);
  for(const id of track.pathwayIds)assert(routes.has(id));
  for(const id of track.projectIds)assert(course.projectIds.includes(id));
  for(const stage of track.stages){
    for(const key of ['title','explanation','application','artifact','check'])assert(stage[key]?.length,track.id+': missing '+key);
    for(const id of stage.lessonIds)assert.equal(lessons.get(id)?.status,'authored',track.id+': missing preparation '+id);
  }
}
for(const route of course.pathways){
  for(const key of ['audience','firstArtifact','studyAdvice'])assert(route[key]?.length);
  assert(route.focusPhaseIds.length);
  for(const id of route.focusPhaseIds)assert(route.phaseIds.includes(id));
  for(const id of route.modalityTrackIds)assert(trackIds.has(id));
}
for(const role of course.roles){
  assert(role.workExample && role.scopeNote && role.reviewQuestions.length);
  for(const id of role.modalityTrackIds)assert(trackIds.has(id));
}
for(const lesson of authored)for(const practice of lesson.content.practice || [])assert.notEqual(practice.prompt,lesson.content.exercise,lesson.id+': duplicated practice task');

// Phase integration stages must resolve to implemented project stages.
for(const phase of course.phases){
  assert(phase.projectId,phase.id+': missing phase integration project');
  assert(course.projectIds.includes(phase.projectId),phase.id+': unregistered project');
  const project=JSON.parse(readFileSync(new URL(`../projects/${phase.projectId}/project.json`,import.meta.url)));
  for(const stage of phase.projectStageIds||[])assert(project.stages.some(s=>s.id===stage),phase.id+': unknown integration stage '+stage);
}
for(const route of course.pathways){
  const project=JSON.parse(readFileSync(new URL(`../projects/${route.capstone.projectId}/project.json`,import.meta.url)));
  const routeLessons=new Set(route.phaseIds.flatMap(id=>phases.get(id).lessons.map(l=>l.id)));
  for(const id of project.requiredLessonIds)assert(routeLessons.has(id),route.id+': capstone prerequisite outside route '+id);
}

for(const phase of course.phases)for(const id of phase.additionalProjectIds||[])assert(course.projectIds.includes(id),'Unknown engineering project '+id);
for(const route of course.pathways)for(const id of route.engineeringLessonIds||[])assert.equal(lessons.get(id)?.status,'authored','Unknown engineering lesson '+id);

// A guide must cover the actual ordered phase, not silently omit difficult lessons.
for (const phase of course.phases) {
  const g = phase.studyGuide;
  assert(g && g.question && g.why && g.completion && g.furtherWork, `${phase.id}: incomplete study guide`);
  assert(g.readiness.prompt && g.readiness.answer && g.transfer.prompt && g.transfer.approach);
  assert(g.pitfall.symptom && g.pitfall.check);
  assert(phase.lessons.some(l=>l.id===g.readiness.lessonId), `${phase.id}: readiness repair must link to this phase`);
  assert.deepEqual(g.milestones.flatMap(m=>m.lessonIds), phase.lessons.map(l=>l.id), `${phase.id}: study milestones must cover each lesson once in order`);
  for (const m of g.milestones) assert(m.title && m.task && m.lessonIds.length);
  function checkGuideMath(value) {
    if (typeof value === 'string') inlineMath(value);
    else if (Array.isArray(value)) value.forEach(checkGuideMath);
    else if (value && typeof value === 'object') Object.values(value).forEach(checkGuideMath);
  }
  checkGuideMath(g);
  const source = readFileSync(new URL(`../phases/${phase.number}-${phase.id}/README.md`, import.meta.url), 'utf8');
  for (const text of [g.readiness.prompt, g.readiness.answer, g.transfer.prompt, g.transfer.approach, g.completion, g.furtherWork, ...g.milestones.map(m=>m.task)]) assert(source.includes(text), `${phase.id}: phase guide omitted from Markdown`);
}
