import {loadLessonEvidence,restoreNotebookOutputs} from './figure-artifacts.mjs';
import {lessonMarkdown,lessonScript,lessonCells} from './lesson-artifacts.mjs';
import {createHash} from 'node:crypto';
import {readFileSync,writeFileSync,mkdirSync,readdirSync} from 'node:fs';
const root=new URL('../',import.meta.url);
mkdirSync(new URL('public/',root),{recursive:true});
for(const name of ['validation.json','openapi.json'])writeFileSync(new URL('public/'+name,root),readFileSync(new URL('curriculum/'+name,root)));

const course=JSON.parse(readFileSync(new URL('curriculum/course.json',root)));
const modalityManifest=JSON.parse(readFileSync(new URL('curriculum/modality-tracks.json',root)));
course.modalityTracks=modalityManifest.tracks;
course.pathways=course.pathwayIds.map(id=>{if(!/^[a-z0-9-]+$/.test(id))throw Error('Invalid pathway ID');return JSON.parse(readFileSync(new URL(`learning-paths/${id}.json`,root)));});
for(const phase of course.phases)for(const lesson of phase.lessons)if(lesson.status==='authored'){
  if(!/^phases\/\d{2}-[a-z0-9-]+\/\d{2}-[a-z0-9-]+$/.test(lesson.path))throw Error('Invalid lesson path');
  const authored=JSON.parse(readFileSync(new URL(lesson.path+'/lesson.json',root)));
  if(authored.id!==lesson.id||authored.schemaVersion!==1)throw Error('Invalid lesson source: '+lesson.id);
  lesson.minutes=authored.minutes;lesson.content=authored.content;
  lesson.contentHash=createHash('sha256').update(JSON.stringify(authored.content)).digest('hex');
  loadLessonEvidence(lesson);
}
writeFileSync(new URL('public/curriculum.json',root),JSON.stringify(course,null,2)+'\n');
const phaseName=id=>{const p=course.phases.find(p=>p.id===id);return `${p.number}: ${p.title}`;};
let text=`# JAX Pathways curriculum\n\n${course.title}.\n\nLearn the shared foundation, build complete systems, then specialize. Notebooks are runnable companions to lessons. The course manifest is [curriculum/course.json](curriculum/course.json); authored content lives in each lesson folder; this document is generated with \`npm run curriculum:docs\`.\n\n## Start here\n\nIf you know basic Python, begin at phase 00. If you already use NumPy, use the foundations route to translate that experience into transformed functions. Math is introduced when needed, with optimization providing the common bridge to applications.\n\nA CPU is sufficient for the early lessons. TPU and multi-device exercises declare their hardware explicitly.\n\n## The course\n\n`;
for(const phase of course.phases){
  const dir=`phases/${phase.number}-${phase.id}`;mkdirSync(new URL(dir+'/',root),{recursive:true});
  const prereqs=phase.prerequisitePhaseIds.map(phaseName).join('; ')||'Basic Python; no previous JAX experience';
  let body=`# Phase ${phase.number}: ${phase.title}\n\n${phase.part}.\n\n${phase.description}\n\n${phase.learningAdvice}\n\n**Prerequisites:** ${prereqs}.\n\n**Hardware:** ${phase.hardware}.\n\n## Lesson sequence\n\n`;
  const g=phase.studyGuide;
  const lessonLink=id=>{const l=phase.lessons.find(l=>l.id===id);return `[${l.title}](${l.path.split('/').pop()}/docs/en.md)`;};
  const guide=`## Study guide: ${g.question}\n\n${g.why}\n\n### Check your starting point\n\n${g.readiness.prompt}\n\n<details><summary>Compare your reasoning</summary>\n\n${g.readiness.answer}\n\n</details>\n\nReview: ${lessonLink(g.readiness.lessonId)}.\n\n### Build in stages\n\n${g.milestones.map((m,i)=>`${i+1}. **${m.title}.** ${m.task}\n\n   Lessons: ${m.lessonIds.map(lessonLink).join(' · ')}.\n`).join('\n')}\n### Try a changed condition\n\n${g.transfer.prompt}\n\n<details><summary>Compare an approach</summary>\n\n${g.transfer.approach}\n\n</details>\n\n**Symptom:** ${g.pitfall.symptom}\n\n**Check next:** ${g.pitfall.check}\n\n### Decide what is ready\n\n${g.completion}\n\n### Further work\n\n${g.furtherWork}\n\n`;
  body=body.replace('## Lesson sequence\n\n',guide+'## Lesson sequence\n\n');
  for(const [i,l]of phase.lessons.entries())body+=`### ${phase.number}.${String(i+1).padStart(2,'0')} ${l.title}\n\n${l.status==='authored'?`[Read the lesson](${l.path.split('/').pop()}/docs/en.md) · [Run the code](${l.path.split('/').pop()}/code/main.py)\n\n`:''}Status: ${l.status==='authored'?'authored lesson with CPU exercise':'roadmap: no lesson material'}.\n\n**Intended outcome:** ${l.objective}\n\n${l.status==='authored'?`**Evidence:** ${l.evidence}\n\n**Checkpoint:** ${l.check}`:'Teaching material, runnable exercises and assessments have not been authored.'}\n\n`;
  body+=`## Phase project\n\n${phase.project}.\n\n**Demonstrate:** ${phase.checkpoint}\n\n${phase.projectId?`Project status: implemented staged practice · [Open source](../../projects/${phase.projectId}/README.md).${phase.projectStageIds?.length ? ' Use stages '+phase.projectStageIds.join(', ')+' for this phase.' : ''}${phase.projectNote ? ' '+phase.projectNote : ''}`:'Project status: planned brief.'}\n\n${(phase.additionalProjectIds||[]).map(id=>'Additional project: ['+JSON.parse(readFileSync(new URL('projects/'+id+'/project.json',root))).title+'](../../projects/'+id+'/README.md).').join('\n')}\n\n[Primary documentation](${phase.source}).\n`;
  writeFileSync(new URL(dir+'/README.md',root),body);
  text+=`### [${phase.number}: ${phase.title}](${dir}/README.md)\n\n${phase.part}. Start after: ${prereqs}.\n\n`;
  for(const l of phase.lessons)text+=`- ${l.title}${l.status==='authored'?' (lesson and exercise available)':''}\n`;
  text+=`\n**Build:** ${phase.project}.\n\n**Checkpoint:** ${phase.checkpoint}\n\n`;
}
text+='## Choose the work you want to do\n\n';
for(const d of course.domains)text+=`### ${d.title}\n\n${d.description}\n\n**Capabilities:** ${d.skills.join('; ')}.\n\n**Pathways:** ${d.pathwayIds.map(id=>course.pathways.find(r=>r.id===id).title).join('; ')}.\n\n**Roles:** ${d.roleIds.map(id=>course.roles.find(r=>r.id===id).title).join('; ')}.\n\n**Artifact:** ${d.artifact}\n\n`;
text+='## Career routes\n\n';
for(const r of course.roles)text+=`### ${r.title}\n\n${r.description}\n\n**Work:** ${r.responsibilities.join('; ')}.\n\n**Skills:** ${r.skills.join('; ')}.\n\n**Portfolio:** ${r.portfolio}\n\n**Demonstrate:** ${r.readiness}\n\n**Milestones:**\n${r.milestones.map((m,i)=>`${i+1}. ${m.title}: ${m.deliverable}`).join('\n')}\n\n`;
text+='## Goal-based pathways\n\nEach pathway references the same canonical phases. It adds a final project and synthesis assessment, rather than duplicating content.\n\n';
for(const r of course.pathways)text+=`### ${r.title}\n\n${r.description}\n\n**For:** ${r.audience}\n\n**First artifact:** ${r.firstArtifact}\n\n**Study advice:** ${r.studyAdvice}\n\n**Engineering extensions (follow each lesson prerequisite):** ${(r.engineeringLessonIds||[]).join(', ')}. See [the engineering release project](projects/engineering-release/README.md).\n\n**Follow:** ${r.phaseIds.map(phaseName).join(' → ')}.\n\n**Final project:** ${r.capstone.title}.\n\n**Assessment:** ${r.capstone.assessment}\n\n**Evidence to collect:**\n${r.capstone.evidence.map(item=>'- '+item).join('\n')}\n\n`;
text+='## Modality project guides\n\nThese guides connect the full lifecycle and identify available harnesses separately from planned work.\n\n'+course.modalityTracks.map(track=>`- [${track.title}](public/guides/${track.id}.md): ${track.summary}`).join('\n')+'\n\n';
text+='## Where the three notebook topics belong\n\n';
for(const n of course.notebookIntegration)text+=`- ${n.legacy} → ${phaseName(n.phaseId)}. ${n.status}.\n`;
text+='\nTheir source files have not been imported. Once available, split them into lesson companions and phase integration labs without reproducing the foundation in each notebook.\n\n## Completion means evidence\n\nReading a lesson, solving its exercise, passing a checkpoint, and completing the pathway project are different milestones. No route awards readiness or a certificate without reviewed artifacts and a synthesis assessment. Each pathway identifies its implemented project and synthesis review draft where available. A brief without a linked implementation remains planned; public checks do not supply independent review.\n';
writeFileSync(new URL('CURRICULUM.md',root),text);

// Generate runnable companions and readable source from the canonical lesson content.
for(const phase of course.phases)for(const lesson of phase.lessons){
  if(lesson.status!=='authored')continue;
  for(const folder of ['docs','code','notebooks','outputs'])mkdirSync(new URL(lesson.path+'/'+folder+'/',root),{recursive:true});
  mkdirSync(new URL(`lessons/${lesson.id}/`,root),{recursive:true});
  mkdirSync(new URL('public/exercises/',root),{recursive:true});
  const markdown=lessonMarkdown(lesson,phase);
  writeFileSync(new URL(lesson.artifacts.source,root),markdown.replaceAll('../figures/'+lesson.id+'-mechanism.svg', '../outputs/mechanism.svg').replaceAll(`../figures/${lesson.id}.svg`, '../outputs/figure.svg'));
  writeFileSync(new URL(`lessons/${lesson.id}/README.md`,root),markdown.replaceAll('../figures/'+lesson.id+'-mechanism.svg', '../../'+lesson.path+'/outputs/mechanism.svg').replaceAll(`../figures/${lesson.id}.svg`, `../../${lesson.path}/outputs/figure.svg`)); // compatibility for existing source links
  writeFileSync(new URL('public/'+lesson.artifacts.markdown,root),markdown);
  writeFileSync(new URL('public/'+lesson.artifacts.script,root),lessonScript(lesson));
  const cell=(type,source,i)=>({cell_type:type,id:`cell-${i}`,metadata:type==='code'&&source.startsWith('# Render the figure')?{jupyter:{source_hidden:true}}:{},...(type==='markdown'&&source.includes('attachment:mechanism.png')?{attachments:{'mechanism.png':{'image/png':readFileSync(new URL(lesson.path+'/outputs/mechanism.png',root)).toString('base64')}}}:{}),source:source.split(/(?<=\n)/),...(type==='code'?{execution_count:null,outputs:[]}: {})});
  const cells=lessonCells(lesson);
  const generatedCells=restoreNotebookOutputs(lesson,cells.map(([t,s],i)=>cell(t,s,i)));
  const executed=generatedCells.filter(c=>c.cell_type==='code').every(c=>c.execution_count!==null);
  writeFileSync(new URL('public/'+lesson.artifacts.notebook,root),JSON.stringify({nbformat:4,nbformat_minor:5,metadata:{kernelspec:{display_name:'Python 3',language:'python',name:'python3'},...(executed?{course_execution:{contentHash:lesson.contentHash,executedAt:lesson.execution.executedAt,environment:lesson.execution.environment,scope:'Recorded instructor reference run; rerun all cells to recompute.'}}:{})},cells:generatedCells},null,2)+'\n');
}

  // Companion source files and tool-readable resources share the same generated content.
for(const phase of course.phases)for(const lesson of phase.lessons){
  if(lesson.status!=='authored')continue;
  for(const [local,source] of [['script','scriptSource'],['notebook','notebookSource']])writeFileSync(new URL(lesson.artifacts[source],root),readFileSync(new URL('public/'+lesson.artifacts[local],root)));
  const c=lesson.content;
  writeFileSync(new URL(lesson.artifacts.quizSource,root),JSON.stringify({schemaVersion:1,lessonId:lesson.id,questions:[{stage:'post',question:c.question,options:c.options,correct:c.answer,explanation:c.explanation}]},null,2)+'\n');
  const evidence=`# Evidence: ${lesson.title}\n\nCopy this template into your own portfolio and fill it after running your modified exercise.\n\n## Evidence required\n\n${lesson.evidence}\n\n- Command: \`python3 ${lesson.artifacts.scriptSource}\`\n- Working directory: repository root\n- Python / JAX / NumPy versions: \n- Hardware and device: \n- My prediction before running: \n- My change to the exercise: \n- Exit code and meaningful output: \n- Why the result supports or contradicts my prediction: \n- Checkpoint result: \n- Artifact link or saved filename: \n- Review status: self-reported; not reviewed\n`;
  writeFileSync(new URL(lesson.artifacts.evidenceSource,root),evidence);
  writeFileSync(new URL('public/'+lesson.artifacts.evidence,root),evidence);
  writeFileSync(new URL(lesson.path+'/README.md',root),`# ${lesson.title}\n\n[Read](docs/en.md) · [Code](code/main.py) · [Notebook](notebooks/exercise.ipynb) · [Quiz](quiz.json) · [Evidence template](outputs/evidence.md)\n\nRun from the repository root: \`python3 ${lesson.artifacts.scriptSource}\`\n\nEdit lesson.json to update this lesson, then run \`npm run build\`. Other files in this bundle are generated. CPU examples only; TPU validation pending.\n`);
}
const json=(path,value)=>{mkdirSync(new URL(path.substring(0,path.lastIndexOf('/')+1),root),{recursive:true});writeFileSync(new URL(path,root),JSON.stringify(value,null,2)+'\n');};
const catalog={schemaVersion:1,title:course.title,phases:course.phases.map(({lessons,...phase})=>({...phase,lessons:lessons.map(({content:_content,...lesson})=>({...lesson,url:`lesson.html?lesson=${lesson.id}`}))}))};
json('public/api/v1/catalog.json',catalog);
json('public/api/v1/pathways.json',{schemaVersion:1,pathways:course.pathways});
json('public/api/v1/roles.json',{schemaVersion:1,roles:course.roles,domains:course.domains});
for(const phase of course.phases)for(const lesson of phase.lessons)json(`public/api/v1/lessons/${lesson.id}.json`,{schemaVersion:1,phaseId:phase.id,...lesson});
const llms=`# JAX Pathways\n\nIndependent JAX course. Authored CPU examples and planned lessons are labeled separately.\n\n- [Catalog](api/v1/catalog.json)\n- [Pathways](api/v1/pathways.json)\n- [Roles](api/v1/roles.json)\n- [API contract](openapi.json)\n\n## Available lessons\n\n${course.phases.flatMap(p=>p.lessons).filter(l=>l.status==='authored').map(l=>`- [${l.title}](lesson.html?lesson=${l.id}) · [Source](api/v1/lessons/${l.id}.json) · ${l.path}`).join('\n')}\n\nCheckpoint success is distinct from runnable exercise evidence. Do not claim TPU validation, expert review, certificates, or authored content for planned lessons.\n`;
writeFileSync(new URL('public/llms.txt',root),llms);

const projects=JSON.parse(readFileSync(new URL('curriculum/projects.json',root)));
json('public/api/v1/projects.json',{schemaVersion:1,projects:projects.map(path=>JSON.parse(readFileSync(new URL(path,root))))});
// Publish figure/audio/source assets used by the static project teaching guides.
for(const manifest of projects){
  const project=JSON.parse(readFileSync(new URL(manifest,root)));
  const directory=`projects/${project.id}/`;
  for(const file of readdirSync(new URL(directory,root),{recursive:true})){
    if(!/\.(png|svg|wav|md)$/.test(file))continue;
    const output=`public/project-assets/${project.id}/${file}`;
    mkdirSync(new URL(output.slice(0,output.lastIndexOf('/')+1),root),{recursive:true});
    writeFileSync(new URL(output,root),readFileSync(new URL(directory+file,root)));
  }
}


// Export the same guided plans for offline learners and coding tutors.
mkdirSync(new URL('public/guides/',root),{recursive:true});
for(const track of course.modalityTracks){
  let body=`# ${track.title}\n\n${track.harnessProjectId ? 'Runnable CPU harness: '+track.harnessProjectId+'. Open project.html?id='+track.harnessProjectId+' or projects/'+track.harnessProjectId+'/README.md in the course workspace.' : 'Guided project plan; connected harness implementation remains planned.'}\n\n${track.summary}\n\n${track.available}\n\n## Before you start\n\n${track.prerequisites}\n\n## Your target artifact\n\n${track.task}\n\n## Figures to explain\n\n${track.plot}\n\n`+track.stages.map((stage,i)=>`## ${i+1}. ${stage.title}\n\n${stage.explanation}\n\n${stage.application}\n\n**Keep:** ${stage.artifact}\n\n**Check:** ${stage.check}\n\n**Available preparation lessons:** ${stage.lessonIds.join(', ')}\n${track.harnessProjectId ? '\n**Runnable project:** '+track.harnessProjectId+' · stages '+stage.projectStageIds.join(', ')+'\n' : ''}`).join('\n');
  body+='\n'+modalityManifest.sharedSections.map(section=>`## ${section.title}\n\n${section.paragraphs[0]}\n\n${section.items.map(item=>'- '+item).join('\n')}\n\n${section.paragraphs[1]}\n`).join('\n');
  body+='\n## References\n\n'+modalityManifest.references.map(ref=>`- [${ref.title}](${ref.url})`).join('\n')+'\n';
  writeFileSync(new URL(`public/guides/${track.id}.md`,root),body);
}
