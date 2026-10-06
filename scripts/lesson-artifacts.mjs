import {figureNotebookCode} from './figure-artifacts.mjs';
// Convert only prose fields for Markdown engines; executable code stays byte-for-byte intact.
export function markdownText(text) {
  return String(text).replace(/(`+)([^\n]*?)\1|\\\(([\s\S]*?)\\\)|\\\[([\s\S]*?)\\\]/g,
    (source, codeFence, _code, inline, display) => codeFence ? source : inline !== undefined ? `$${inline}$` : `$$\n${display}\n$$`);
}
function markdownContent(value, key = '') {
  if (['code','solution','buildCode','formula','math'].includes(key)) return value;
  if (typeof value === 'string') return markdownText(value);
  if (Array.isArray(value)) return value.map(item => markdownContent(item, key));
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([k,v])=>[k,markdownContent(v,k)]));
  return value;
}
// All lesson companions include the same teaching sections and runnable experiments.
const fence = code => `\`\`\`python\n${code}\n\`\`\``;
const terminalMarkdown = s => (s.commands||[]).map(c=>`\n\n**${c.label}**\n\n\`\`\`${c.shell}\n${c.code}\n\`\`\`\n\n**Expected:** ${c.expected}`).join('');
const paragraphs = (heading, body) => `## ${heading}\n\n${body}\n\n`;
function mechanismMarkdown(lesson, notebook=false) {
  const d=lesson.content.diagram;
  if(!d)return '';
  const hasImage=lesson.visualArtifact?.mechanismImage;
  const image=notebook?'attachment:mechanism.png':'../figures/'+lesson.id+'-mechanism.svg';
  return '\n\n### '+d.title+'\n\n**Predict:** '+markdownText(d.prediction)+'\n\n'+(hasImage?'!['+d.title+']('+image+')\n\n':'')+'*'+d.scope+'*\n\n'+markdownText(d.reading);
}
function sectionMarkdown(s,lesson,notebook=false) {
  return s.body+terminalMarkdown(s)+(s.math?'\n\n$$\n'+s.math+'\n$$':'')+(s.formula?'\n\n'+String.fromCharCode(96).repeat(3)+'text\n'+s.formula+'\n'+String.fromCharCode(96).repeat(3):'')+(s.id==='guided-reasoning'?mechanismMarkdown(lesson,notebook):'')+(s.check?'\n\n### Pause and reason\n\n'+s.check.prompt+'\n\n<details><summary>Compare your reasoning</summary>\n\n'+s.check.answer+'\n\n</details>':'');
}
export function lessonMarkdown(lesson, phase) {
  const c=markdownContent(lesson.content);
  let out=`# ${lesson.title}\n\nPhase ${phase.number}: ${phase.title} · about ${lesson.minutes} minutes · ${c.runtime?.kind==='virtual-cpu'?`${c.runtime.deviceCount} logical CPU devices`:'CPU'}\n\n`;
  if(c.setupBundle)out+=`${c.setupBundle.title}: public/${c.setupBundle.url}\n\nIn the lesson reader, use the beginner workspace download button. In a local checkout, the bundle is under public/downloads/.\n\n`;
  if(c.objectives)out+=paragraphs('What you will be able to do',c.objectives.map(o=>`- ${o}`).join('\n'));
  if(c.problem)out+=paragraphs('The problem',c.problem);
  out+=paragraphs('The idea',c.idea);
  for(const s of c.sections||[])out+=paragraphs(s.title,sectionMarkdown(s,lesson));
  for(const step of c.buildSteps||[])out+=paragraphs(step.title,step.instruction+'\n\n'+fence(step.code)+'\n\n'+step.explanation);
  if(c.buildCode)out+=paragraphs('Build a numerical estimate',fence(c.buildCode)+'\n\n'+c.buildOutput);
  out+=paragraphs('Run the example',fence(c.code)+'\n\nExpected: '+c.output);
  if(c.visual){ const v=c.visual; out+=paragraphs(v.title,`**Predict:** ${v.prediction}\n\n${lesson.visualArtifact ? `![${v.title}](../figures/${lesson.id}.svg)\n\n` : ''}**${v.kind==='executed'?'Recorded CPU computation':'Conceptual diagram'}**\n\n### Read the figure\n\n${v.reading}\n\n### Connect it to the computation\n\n${v.connection}${v.code?'\n\n'+fence(v.code):''}`); }
  if(lesson.execution)out+=paragraphs('Recorded reference execution',`CPU run: ${lesson.execution.executedAt}. JAX ${lesson.execution.environment.jax}.\n\n`+'```text\n'+lesson.execution.stdout+'\n```');
  for(const e of c.experiments||[])out+=paragraphs(e.title,`**Predict before running:** ${e.prediction}\n\n${fence(e.code)}\n\n**Expected:** ${e.output}\n\n${e.explanation}`);
  out+=paragraphs('Make it yours',`${c.exercise}\n\n<details><summary>Reference solution</summary>\n\n${fence(c.solution)}\n\n</details>`);
  for(const p of c.practice||[])out+=paragraphs(p.title,`**${p.difficulty}**\n\n${p.prompt}\n\n<details><summary>Hint</summary>\n\n${p.hint}\n\n</details>\n\n<details><summary>Reference solution and reasoning</summary>\n\n${fence(p.solution)}\n\n${p.explanation}\n\n</details>`);
  out+=paragraphs('Check your understanding',`${c.question}\n\n${c.options.map((a,i)=>`${i+1}. ${a}`).join('\n')}\n\n<details><summary>Answer and explanation</summary>\n\n${c.options[c.answer]}\n\n${c.explanation}\n\n</details>`);
  if(c.diagnosis)out+=paragraphs('Diagnose the result',c.diagnosis);
  if(c.takeaways)out+=paragraphs('Carry forward',c.takeaways.map(t=>`- ${t}`).join('\n'));
  out+=paragraphs('Keep your evidence',lesson.evidence+'\n\nKeep predictions, modified code, observed results, and reasoning. A checkpoint alone does not demonstrate the exercise.');
  out+=paragraphs('Primary references',(c.references||[{title:'Primary documentation',url:lesson.source||phase.source}]).map(r=>`- [${r.title}](${r.url})`).join('\n'));
  return out;
}
export function lessonScript(lesson){
  const c=lesson.content;
  return `"""${lesson.title}: worked experiments and reference solutions. CPU checks."""\n\n`+(c.buildSteps||[]).map(step=>`# ${step.title}\n${step.code}`).join('\n\n')+'\n\n'+(c.buildCode?c.buildCode+'\n\n':'')+c.code+'\n\n'+(c.visual?.code?'# Figure data experiment\n'+c.visual.code+'\n\n':'')+(c.experiments||[]).map(e=>`# Experiment: ${e.title}\n${e.code}`).join('\n\n')+`\n\n# Reference solution. Try the exercise before reading this.\n${c.solution}\n\n`+(c.practice||[]).map(p=>`# Reference practice: ${p.title}\n${p.solution}`).join('\n\n')+`\nprint("PASS: ${lesson.id}")\n`;
}
export function lessonCells(lesson){
  const c=markdownContent(lesson.content);
  const cells=[['markdown',`# ${lesson.title}\n\n${c.runtime?.kind==='virtual-cpu'?`Start a fresh kernel, then Run all: the first cell configures ${c.runtime.deviceCount} logical CPU devices before backend initialization. They share one physical CPU; this is sharding practice, not TPU performance emulation.`:'CPU companion; no accelerator is required.'} TPU execution not validated.\n\n${c.objectives?c.objectives.map(o=>'- '+o).join('\n')+'\n\n':''}${c.problem||''}\n\n## The idea\n\n${c.idea}`]];
  for(const s of c.sections||[])cells.push(['markdown','## '+s.title+'\n\n'+sectionMarkdown(s,lesson,true)]);
  for(const step of c.buildSteps||[])cells.push(['markdown',`## ${step.title}\n\n${step.instruction}`],['code',step.code],['markdown',step.explanation]);
  if(c.buildCode)cells.push(['markdown','## Numerical estimate\n\n'+c.buildOutput],['code',c.buildCode]);
  cells.push(['markdown','## Run the example'],['code',c.code],['markdown','Expected: '+c.output]);
  if(c.visual){ const v=c.visual; cells.push(['markdown',`## ${v.title}\n\nPredict: ${v.prediction}\n\n${v.kind==='executed'?'CPU computation; rerun to recompute the data.':'Conceptual diagram; the arrows are an instructional map.'}\n\nRun the cells below to produce the figure, then follow the walkthrough beneath it.`]); if(v.code)cells.push(['code',v.code]); cells.push(['code',figureNotebookCode(lesson)],['markdown',`### Read the figure\n\n${v.reading}\n\n### Connect it to the computation\n\n${v.connection}`]); }
  for(const e of c.experiments||[])cells.push(['markdown',`## ${e.title}\n\nPredict: ${e.prediction}`],['code',e.code],['markdown',`Expected: ${e.output}\n\n${e.explanation}`]);
  cells.push(['markdown','## Your exercise\n\n'+c.exercise],['code','# Write your attempt here.'],['markdown','## Reference solution\n\nTry your own implementation first.'],['code',c.solution]);
  for(const p of c.practice||[])cells.push(['markdown',`## ${p.title} · ${p.difficulty}\n\n${p.prompt}\n\nHint: ${p.hint}`],['code','# Write your attempt here.'],['markdown','## Reference reasoning\n\n'+p.explanation],['code',p.solution]);
  cells.push(['markdown',`## Checkpoint\n\n${c.question}\n\n${c.options.map((o,i)=>`${i+1}. ${o}`).join('\n')}\n\n## Diagnose\n\n${c.diagnosis||''}\n\n## Evidence\n\n${lesson.evidence}`]);
  if(c.takeaways)cells.push(['markdown','## Carry forward\n\n'+c.takeaways.map(t=>'- '+t).join('\n')]);
  if(c.references)cells.push(['markdown','## Primary references\n\n'+c.references.map(r=>`- [${r.title}](${r.url})`).join('\n')]);
  return cells;
}
