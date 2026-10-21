import {figureNotebookCode} from './figure-artifacts.mjs';
import {executionScope} from '../src/lib/runtime.js';
import {buildStarterScaffold,extractCodeApis,isLearnerStarterContract} from '../src/lib/code-teaching.js';
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
function exerciseGuideMarkdown(_solutionCode, title = 'How to write this code', learnerStarter) {
  if (!isLearnerStarterContract(learnerStarter)) {
    return `### ${title}\n\n**Learner starter pending:** this lesson has no separate authored input/output contract, checks, hints, and TODO scaffold yet.`;
  }
  const apis = extractCodeApis(learnerStarter.code);
  const scaffold = buildStarterScaffold(learnerStarter);
  let guide = `### ${title} — Step-by-step recipe & starter scaffold\n\n`;
  const field = (heading, items) => items.length ? `**${heading}:**\n${items.map(item => `- ${markdownText(item)}`).join('\n')}\n\n` : '';
  guide += field('Inputs', learnerStarter.inputs);
  guide += field('Outputs', learnerStarter.outputs);
  guide += field('Invariants to check', learnerStarter.invariants);
  guide += field('Hints', learnerStarter.hints);
  if (apis.length) {
    guide += `**Key functions & syntax to use:**\n` + apis.map(a => `- \`${a.token}\` — ${a.summary}`).join('\n') + '\n\n';
  }
  guide += `**Starter code scaffold (fill in the TODOs):**\n\n${fence(scaffold)}`;
  return guide;
}
function referenceNotebookAttemptScaffold(solutionCode) {
  // The instructor notebook retains its historical cell source so saved
  // execution receipts continue to match. Learner-only notebooks use the
  // authored contract directly and never call this reference helper.
  const scaffold = buildStarterScaffold(solutionCode);
  const commented = scaffold.split('\n').map(l => l.trim().startsWith('#') || !l.trim() ? l : `# ${l}`).join('\n');
  return `# Write your attempt here (uncomment and fill in the TODOs below):\n${commented}`;
}
const terminalMarkdown = s => (s.commands||[]).map(c=>`\n\n**${c.label}**\n\n\`\`\`${c.shell}\n${c.code}\n\`\`\`\n\n**Expected:** ${c.expected}`).join('');
const paragraphs = (heading, body) => `## ${heading}\n\n${body}\n\n`;
const feedbackMarkdown = c => c.options.map((option, index) =>
  `**${index + 1}. ${option}${index === c.answer ? ' (correct)' : ''}**\n\n${c.optionFeedback[index]}`).join('\n\n');
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
  const scope=executionScope(lesson);
  out+=scope.description+' '+scope.target+'\n\n';
  if(c.setupBundle)out+=`${c.setupBundle.title}: public/${c.setupBundle.url}\n\nIn the lesson reader, use the beginner workspace download button. In a local checkout, the bundle is under public/downloads/.\n\n`;
  if(c.objectives)out+=paragraphs('What you will be able to do',c.objectives.map(o=>`- ${o}`).join('\n'));
  if(c.problem)out+=paragraphs('The problem',c.problem);
  out+=paragraphs('The idea',c.idea);
  for(const s of c.sections||[])out+=paragraphs(s.title,sectionMarkdown(s,lesson));
  for(const step of c.buildSteps||[])out+=paragraphs(step.title,step.instruction+'\n\n'+fence(step.code)+'\n\n'+step.explanation);
  if(c.buildCode)out+=paragraphs('Build a numerical estimate',fence(c.buildCode)+'\n\n'+c.buildOutput);
  out+=paragraphs('Run the example',fence(c.code)+'\n\nExpected: '+c.output);
  if(c.visual){ const v=c.visual; const scope=v.kind==='executed'?'Recorded CPU computation':v.kind==='analytic'?'Analytic calculation; no device result recorded':'Conceptual diagram'; out+=paragraphs(v.title,`**Predict:** ${v.prediction}\n\n${lesson.visualArtifact ? `![${v.title}](../figures/${lesson.id}.svg)\n\n` : ''}**${scope}**\n\n### Read the figure\n\n${v.reading}\n\n### Connect it to the computation\n\n${v.connection}${v.code?'\n\n'+fence(v.code):''}`); }
  if(lesson.execution)out+=paragraphs('Recorded reference execution',`CPU run: ${lesson.execution.executedAt}. JAX ${lesson.execution.environment.jax}.\n\n`+'```text\n'+lesson.execution.stdout+'\n```');
  for(const e of c.experiments||[])out+=paragraphs(e.title,`**Predict before running:** ${e.prediction}\n\n${fence(e.code)}\n\n**Expected:** ${e.output}\n\n${e.explanation}`);
  out+=paragraphs('Make it yours',`${c.exercise}\n\n${exerciseGuideMarkdown(c.solution, 'How to write this exercise', c.learnerStarter)}\n\n<details><summary>Reference solution</summary>\n\n${fence(c.solution)}\n\n</details>`);
  for(const p of c.practice||[])out+=paragraphs(p.title,`**${p.difficulty}**\n\n${p.prompt}\n\n<details><summary>Hint</summary>\n\n${p.hint}\n\n</details>\n\n${exerciseGuideMarkdown(p.solution, 'How to write: ' + p.title, p.learnerStarter)}\n\n<details><summary>Reference solution and reasoning</summary>\n\n${fence(p.solution)}\n\n${p.explanation}\n\n</details>`);
  out+=paragraphs('Check your understanding',`${c.question}\n\n${c.options.map((a,i)=>`${i+1}. ${a}`).join('\n')}\n\n<details><summary>Answer and explanation</summary>\n\n${feedbackMarkdown(c)}\n\n</details>`);
  if(c.diagnosis)out+=paragraphs('Diagnose the result',c.diagnosis);
  if(c.takeaways)out+=paragraphs('Carry forward',c.takeaways.map(t=>`- ${t}`).join('\n'));
  out+=paragraphs('Keep your evidence',lesson.evidence+'\n\nSave your predictions, modified code, terminal outputs, and reasoning in your engineering portfolio.');
  out+=paragraphs('Primary references',(c.references||[{title:'Primary documentation',url:lesson.source||phase.source}]).map(r=>`- [${r.title}](${r.url})`).join('\n'));
  return out;
}
export function lessonScript(lesson){
  const c=lesson.content;
  return `"""${lesson.title}: worked experiments and reference solutions. CPU checks."""\n\n`+(c.buildSteps||[]).map(step=>`# ${step.title}\n${step.code}`).join('\n\n')+'\n\n'+(c.buildCode?c.buildCode+'\n\n':'')+c.code+'\n\n'+(c.visual?.code?'# Figure data experiment\n'+c.visual.code+'\n\n':'')+(c.experiments||[]).map(e=>`# Experiment: ${e.title}\n${e.code}`).join('\n\n')+`\n\n# Reference solution. Try the exercise before reading this.\n${c.solution}\n\n`+(c.practice||[]).map(p=>`# Reference practice: ${p.title}\n${p.solution}`).join('\n\n')+`\nprint("PASS: ${lesson.id}")\n`;
}
export function lessonCells(lesson){
  const c=markdownContent(lesson.content);
  const scope=executionScope(lesson);
  const cells=[['markdown',`# ${lesson.title}\n\n${c.runtime?.kind==='virtual-cpu'?`Start a fresh kernel, then Run all: the first cell configures ${c.runtime.deviceCount} logical CPU devices before backend initialization so you can practice multi-device sharding locally.\n\n`:''}${scope.description} ${scope.target}\n\n${c.objectives?c.objectives.map(o=>'- '+o).join('\n')+'\n\n':''}${c.problem||''}\n\n## The idea\n\n${c.idea}`]];
  for(const s of c.sections||[])cells.push(['markdown','## '+s.title+'\n\n'+sectionMarkdown(s,lesson,true)]);
  for(const step of c.buildSteps||[])cells.push(['markdown',`## ${step.title}\n\n${step.instruction}`],['code',step.code],['markdown',step.explanation]);
  if(c.buildCode)cells.push(['markdown','## Numerical estimate\n\n'+c.buildOutput],['code',c.buildCode]);
  cells.push(['markdown','## Run the example'],['code',c.code],['markdown','Expected: '+c.output]);
  if(c.visual){ const v=c.visual; const provenance=v.kind==='executed'?'CPU computation; rerun to recompute the data.':v.kind==='analytic'?'Analytic calculation; this is not a recorded device result.':'Conceptual diagram; the arrows are an instructional map.'; cells.push(['markdown',`## ${v.title}\n\nPredict: ${v.prediction}\n\n${provenance}\n\nRun the cells below to produce the figure, then follow the walkthrough beneath it.`]); if(v.code)cells.push(['code',v.code]); cells.push(['code',figureNotebookCode(lesson)],['markdown',`### Read the figure\n\n${v.reading}\n\n### Connect it to the computation\n\n${v.connection}`]); }
  for(const e of c.experiments||[])cells.push(['markdown',`## ${e.title}\n\nPredict: ${e.prediction}`],['code',e.code],['markdown',`Expected: ${e.output}\n\n${e.explanation}`]);
  cells.push(['markdown','## Your exercise\n\n'+c.exercise+'\n\n'+exerciseGuideMarkdown(c.solution,'How to write this exercise',c.learnerStarter)],['code',referenceNotebookAttemptScaffold(c.solution)],['markdown','## Reference solution\n\nTry your own implementation first.'],['code',c.solution]);
  for(const p of c.practice||[])cells.push(['markdown',`## ${p.title} · ${p.difficulty}\n\n${p.prompt}\n\nHint: ${p.hint}\n\n${exerciseGuideMarkdown(p.solution,'How to write: '+p.title,p.learnerStarter)}`],['code',referenceNotebookAttemptScaffold(p.solution)],['markdown','## Reference reasoning\n\n'+p.explanation],['code',p.solution]);
  cells.push(['markdown',`## Checkpoint\n\n${c.question}\n\n${c.options.map((o,i)=>`${i+1}. ${o}`).join('\n')}\n\n## Diagnose\n\n${c.diagnosis||''}\n\n## Evidence\n\n${lesson.evidence}`]);
  cells.push(['markdown',`<details><summary>Compare checkpoint reasoning after your attempt</summary>\n\n${feedbackMarkdown(c)}\n\n</details>`]);
  if(c.takeaways)cells.push(['markdown','## Carry forward\n\n'+c.takeaways.map(t=>'- '+t).join('\n')]);
  if(c.references)cells.push(['markdown','## Primary references\n\n'+c.references.map(r=>`- [${r.title}](${r.url})`).join('\n')]);
  return cells;
}

function contractMarkdown(contract) {
  if (!isLearnerStarterContract(contract)) {
    return '**Learner starter pending:** this lesson does not yet have a separate authored contract and TODO scaffold.';
  }
  const section = (heading, items) => items.length ? `**${heading}:**\n\n${items.map(item => `- ${markdownText(item)}`).join('\n')}\n\n` : '';
  return section('Inputs', contract.inputs) + section('Outputs', contract.outputs) +
    section('Invariants to check', contract.invariants) + section('Hints', contract.hints) +
    `**Your TODO scaffold:**\n\n${fence(contract.code)}`;
}

/**
 * Learner-only notebook content. It intentionally omits worked examples,
 * recorded expected values, reference code, checkpoint answers, and rationales.
 */
export function learnerNotebookCells(lesson) {
  const c = lesson.content;
  const coverage = learnerStarterCoverage(lesson);
  const pending = coverage.total - coverage.ready;
  const cells = [['markdown', `# ${lesson.title} · learner-only workspace\n\nAuthored learner starters: ${coverage.ready}/${coverage.total} tasks. Pending: ${pending}. Any task marked pending needs its own contract before this file is a complete starter. Worked examples, recorded expected values, solutions, checkpoint answers, and rationales are not included.`]];
  cells.push(
    ['markdown', `## Exercise · Make it yours\n\n${c.exercise}\n\n${contractMarkdown(c.learnerStarter)}`],
    ['code', learnerNotebookScaffold(c.learnerStarter)],
  );
  for (const p of c.practice || []) {
    cells.push(
      ['markdown', `## Practice · ${p.title} · ${p.difficulty}\n\n${p.prompt}\n\n${contractMarkdown(p.learnerStarter)}`],
      ['code', learnerNotebookScaffold(p.learnerStarter)],
    );
  }
  return cells;
}

function learnerNotebookScaffold(contract) {
  if (!isLearnerStarterContract(contract)) {
    return '# Learner starter pending: no separately authored contract or TODO scaffold is available for this task.';
  }
  // Keep the authored TODO code executable/editable in the learner file. The
  // commented attempt copy remains only in the instructor reference notebook.
  return buildStarterScaffold(contract);
}

export function learnerStarterCoverage(lesson) {
  const contracts = [lesson.content?.learnerStarter, ...(lesson.content?.practice || []).map((p) => p.learnerStarter)];
  return {
    total: contracts.length,
    ready: contracts.filter(isLearnerStarterContract).length,
  };
}

/** Report complete versus missing learner contracts from loaded canonical lessons. */
export function learnerStarterInventory(lessons) {
  const entries = lessons.filter((lesson) => lesson.status === 'authored').map((lesson) => {
    const tasks = [
      { scope: 'exercise', contract: lesson.content?.learnerStarter },
      ...(lesson.content?.practice || []).map((practice, index) => ({
        scope: `practice:${index + 1}`,
        title: practice.title,
        contract: practice.learnerStarter,
      })),
    ];
    const missing = tasks.filter((task) => !isLearnerStarterContract(task.contract)).map(({ contract: _contract, ...task }) => task);
    const ready = tasks.length - missing.length;
    return { lessonId: lesson.id, ready, total: tasks.length, complete: missing.length === 0, missing };
  });
  return {
    totalLessons: entries.length,
    completeLessons: entries.filter((entry) => entry.complete).length,
    incompleteLessons: entries.filter((entry) => !entry.complete).length,
    readyContracts: entries.reduce((sum, entry) => sum + entry.ready, 0),
    totalContracts: entries.reduce((sum, entry) => sum + entry.total, 0),
    lessons: entries,
  };
}

/** Derive learner-only URL and workspace paths from the canonical lesson location. */
export function learnerArtifactPaths(lesson) {
  return {
    learnerNotebook: `exercises/${lesson.id}-learner.ipynb`,
    learnerScript: `exercises/${lesson.id}-learner.py`,
    learnerNotebookSource: `${lesson.path}/notebooks/learner.ipynb`,
    learnerScriptSource: `${lesson.path}/code/learner.py`,
  };
}

/** Learner URLs are available only after every task contract is authored. */
export function readyLearnerArtifactPaths(lesson) {
  const coverage = learnerStarterCoverage(lesson);
  return coverage.total > 0 && coverage.ready === coverage.total ? learnerArtifactPaths(lesson) : {};
}

function pythonComment(text) {
  return String(text).split('\n').map(line => `# ${line}`).join('\n');
}

/**
 * Learner-only Python source. It contains authored exercise contracts and
 * attempt scaffolds only; reference execution remains in lessonScript().
 */
export function learnerScript(lesson) {
  const c = lesson.content;
  const coverage = learnerStarterCoverage(lesson);
  const pieces = [`"""${String(lesson.title).replaceAll('"""', "'''")} · learner-only exercises (${coverage.ready}/${coverage.total} contracts ready)"""`,
    pythonComment(`Exercise: ${c.exercise}`)];
  const appendContract = (contract) => {
    if (!isLearnerStarterContract(contract)) {
      pieces.push('# Learner starter pending: no separate contract or scaffold is authored yet.');
      return;
    }
    for (const [heading, values] of Object.entries({
      Inputs: contract.inputs, Outputs: contract.outputs,
      'Invariants to check': contract.invariants, Hints: contract.hints,
    })) {
      pieces.push(pythonComment(`${heading}:\n${values.map(value => `- ${value}`).join('\n')}`));
    }
    pieces.push(contract.code);
  };
  appendContract(c.learnerStarter);
  for (const p of c.practice || []) {
    pieces.push('', pythonComment(`${p.title} (${p.difficulty}):\n${p.prompt}`));
    appendContract(p.learnerStarter);
  }
  return pieces.join('\n\n') + '\n';
}
