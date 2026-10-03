import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
const course=JSON.parse(readFileSync(new URL('../site/curriculum.json',import.meta.url)));
const root=new URL('../',import.meta.url);
const phaseName=id=>{const p=course.phases.find(p=>p.id===id);return `${p.number}: ${p.title}`;};
let text=`# JAX Pathways curriculum\n\n${course.title}.\n\nLearn the shared foundation, build complete systems, then specialize. Notebooks are runnable companions to lessons. The source of truth is [site/curriculum.json](site/curriculum.json); this document is generated with \`npm run curriculum:docs\`.\n\n## Start here\n\nIf you know basic Python, begin at phase 00. If you already use NumPy, use the foundations route to translate that experience into transformed functions. Math is introduced when needed, with optimization providing the common bridge to applications.\n\nA CPU is sufficient for the early lessons. TPU and multi-device exercises declare their hardware explicitly.\n\n## The course\n\n`;
for(const phase of course.phases){
  const dir=`phases/${phase.number}-${phase.id}`;mkdirSync(new URL(dir+'/',root),{recursive:true});
  const prereqs=phase.prerequisitePhaseIds.map(phaseName).join('; ')||'Basic Python; no previous JAX experience';
  let body=`# Phase ${phase.number}: ${phase.title}\n\n${phase.part}.\n\n**Prerequisites:** ${prereqs}.\n\n**Hardware:** ${phase.hardware}.\n\n## Lesson sequence\n\n`;
  for(const [i,l]of phase.lessons.entries())body+=`### ${phase.number}.${String(i+1).padStart(2,'0')} ${l.title}\n\nStatus: ${l.status==='sample'?'authored sample':'planned brief'}.\n\n**Learn and build:** ${l.objective}\n\n**Evidence:** ${l.evidence}\n\n**Checkpoint:** ${l.check}\n\n`;
  body+=`## Phase project\n\n${phase.project}.\n\n**Demonstrate:** ${phase.checkpoint}\n\nProject status: planned brief.\n\n[Primary documentation](${phase.source}).\n`;
  writeFileSync(new URL(dir+'/README.md',root),body);
  text+=`### [${phase.number}: ${phase.title}](${dir}/README.md)\n\n${phase.part}. Start after: ${prereqs}.\n\n`;
  for(const l of phase.lessons)text+=`- ${l.title}${l.status==='sample'?' (sample available)':''}\n`;
  text+=`\n**Build:** ${phase.project}.\n\n**Checkpoint:** ${phase.checkpoint}\n\n`;
}
text+='## Choose the work you want to do\n\n';
for(const d of course.domains)text+=`### ${d.title}\n\n${d.description}\n\n**Capabilities:** ${d.skills.join('; ')}.\n\n**Pathways:** ${d.pathwayIds.map(id=>course.pathways.find(r=>r.id===id).title).join('; ')}.\n\n**Roles:** ${d.roleIds.map(id=>course.roles.find(r=>r.id===id).title).join('; ')}.\n\n**Artifact:** ${d.artifact}\n\n`;
text+='## Career routes\n\n';
for(const r of course.roles)text+=`### ${r.title}\n\n${r.description}\n\n**Work:** ${r.responsibilities.join('; ')}.\n\n**Skills:** ${r.skills.join('; ')}.\n\n**Portfolio:** ${r.portfolio}\n\n**Demonstrate:** ${r.readiness}\n\n`;
text+='## Goal-based pathways\n\nEach pathway references the same canonical phases. It adds a final project and synthesis assessment, rather than duplicating content.\n\n';
for(const r of course.pathways)text+=`### ${r.title}\n\n${r.description}\n\n**Follow:** ${r.phaseIds.map(phaseName).join(' → ')}.\n\n**Final project:** ${r.capstone.title}.\n\n**Assessment:** ${r.capstone.assessment}\n\n`;
text+='## Where the three notebook topics belong\n\n';
for(const n of course.notebookIntegration)text+=`- ${n.legacy} → ${phaseName(n.phaseId)}. ${n.status}.\n`;
text+='\nTheir source files have not been imported. Once available, split them into lesson companions and phase integration labs without reproducing the foundation in each notebook.\n\n## Completion means evidence\n\nReading a lesson, solving its exercise, passing a checkpoint, and completing the pathway project are different milestones. No route awards readiness or a certificate without reviewed artifacts and a synthesis assessment. All projects and assessments in this revision are planned briefs.\n';
writeFileSync(new URL('CURRICULUM.md',root),text);
