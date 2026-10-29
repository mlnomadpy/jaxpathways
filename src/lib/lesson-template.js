import { revisionMarkup, feedbackUrl, feedbackAccess } from './releases.js';
import { materialNotice } from './material-status.js';
import { lessonVisual, lessonExecution } from './lesson-visuals.js';
import { inlineMath } from './math.js';
import { escapeHtml } from './html.js';
import { renderRadioGroup } from '../components/ui/radio-group.js';
import { runtimeLabel, executionScope, courseTypeCue } from './runtime.js';
import { practiceDrawer } from './practice-content.js';
import { gradientFigure, attentionMaskFigure } from './diagram-markup.js';
import { courseConceptFigure } from './concept-figures.js';
import {
  deepConcepts,
  prose,
  codePanel,
  workedExperiments,
  furtherPractice,
} from './lesson-content.js';
import {
  isLearnerStarterContract,
  renderBuildStepCodeBreakdown,
  renderExerciseCodeTeacher,
} from './code-teaching.js';

/** Counts authored learner contracts across the exercise and all practice tasks. */
/** @param {import('../types/course').LessonContent | undefined} content */
export function learnerStarterCoverage(content) {
  const contracts = [
    content?.learnerStarter,
    ...(content?.practice || []).map(
      (/** @type {{ learnerStarter?: import('../types/course').LearnerStarterContract }} */ task) =>
        task.learnerStarter,
    ),
  ];
  return { ready: contracts.filter(isLearnerStarterContract).length, total: contracts.length };
}

/** Offer downloads only when every task has an authored learner contract. */
/** @param {import('../types/course').Lesson['artifacts'] | undefined} artifacts @param {import('../types/course').LessonContent | undefined} content @param {'notebook' | 'script'} kind */
export function learnerDownloadLink(artifacts, content, kind) {
  const coverage = learnerStarterCoverage(content);
  const notebook = kind === 'notebook';
  const label = notebook ? 'notebook' : 'Python script';
  const href = notebook ? artifacts?.learnerNotebook : artifacts?.learnerScript;
  if (!coverage.total || coverage.ready !== coverage.total || !href) {
    return `<p class="learner-starter-pending"><strong>Learner-only ${label} pending.</strong> ${coverage.ready} of ${coverage.total} task starters are authored; download appears when every task has its own contract.</p>`;
  }
  const coverageLabel =
    coverage.ready === coverage.total
      ? `${coverage.ready} task starter${coverage.ready === 1 ? '' : 's'}`
      : `${coverage.ready} of ${coverage.total} task starters; ${coverage.total - coverage.ready} pending`;
  return `<a class="button${notebook ? '' : ' learner-script-download'}" href="${escapeHtml(href)}" download>Download learner-only ${label} · ${coverageLabel}</a>`;
}
/**
 * Pure lesson view: escaped canonical content, with browser behavior bound separately.
 * @param {{
 *   phase: import('../types/course').Phase;
 *   lesson: import('../types/course').Lesson;
 *   prerequisiteLinks: string;
 *   allLessons: import('../types/course').Lesson[];
 *   index: number;
 *   activePath: string;
 * }} params
 */
export function renderLesson({ phase, lesson, prerequisiteLinks, allLessons, index, activePath }) {
  const c = lesson.content,
    artifacts = lesson.artifacts,
    authored = lesson.status === 'authored' && Boolean(c && artifacts),
    cue = courseTypeCue(phase),
    scope = executionScope(lesson),
    buildFile = c?.setupBundle ? 'first_experiment.py' : 'main.py',
    stages = [
      'Understand',
      'Build',
      ...(c?.experiments?.length ? ['Experiment'] : []),
      'Practice',
      'Evidence',
    ],
    stageCount = stages.length,
    stageNumber = (/** @type {string} */ name) => stages.indexOf(name) + 1,
    phaseLessonIndex = Math.max(
      0,
      phase.lessons.findIndex((l) => l.id === lesson.id),
    );
  return (
    `<div class="course-cue-row"><span class="course-type-pill" data-course-kind="${escapeHtml(cue.kind)}">${escapeHtml(cue.badge)}</span><span class="course-meta-pill">${escapeHtml(cue.hardwareBadge)}</span><span class="course-meta-pill">Lesson ${phaseLessonIndex + 1} of ${phase.lessons.length} in Phase ${escapeHtml(phase.number)}</span></div>` +
    `<p class="edition">Phase ${phase.number} / ${escapeHtml(phase.title)}</p><h1 id="lesson-title" tabindex="-1">${escapeHtml(lesson.title)}</h1>${revisionMarkup('lessons', lesson.id)}<p class="lesson-status">${authored ? `Lesson · about ${lesson.minutes} minutes · ${runtimeLabel(lesson)}` : 'TODO · lesson in progress'}</p>${authored ? materialNotice('lessons', lesson.id) : ''}<p>${inlineMath(lesson.objective)}</p><p><a href="phase-guides/${encodeURIComponent(phase.id)}.html">Phase study guide: preparation, milestones and project evidence</a></p><details class="before-start"><summary>Before you start</summary><p class="soft"><strong>Before this:</strong> ${prerequisiteLinks}<br><strong>Hardware:</strong> ${inlineMath(lesson.hardware || phase.hardware)}</p>${authored ? `<aside class="runtime-note"><strong>${runtimeLabel(lesson)}</strong><p>${escapeHtml(scope.description)}</p><p>${escapeHtml(scope.target)} <a href="lesson-tpu-05.html">Read the execution matrix</a>.</p></aside>` : ''}</details>` +
    (authored && c && artifacts
      ? `
    <div class="lesson-artifacts">${c.setupBundle ? `<a class="button primary" href="${escapeHtml(c.setupBundle.url)}" download>${escapeHtml(c.setupBundle.title)}</a>` : learnerDownloadLink(artifacts, c, 'notebook')}<details class="lesson-resources"><summary>Other downloads</summary><a href="${escapeHtml(artifacts.notebook)}" download>Worked reference notebook</a>${c.setupBundle ? learnerDownloadLink(artifacts, c, 'notebook') : ''} ${learnerDownloadLink(artifacts, c, 'script')}<a href="${escapeHtml(artifacts.script)}" download>Python worked examples</a><a href="${escapeHtml(artifacts.markdown)}" download>Lesson text</a></details></div>
    ${practiceDrawer(lesson)}
    <div id="lesson-stage-bar" class="lesson-stage-bar" data-active-stage="Understand" aria-label="Lesson stage progress">
      <div class="stage-bar-context">
        <span class="stage-bar-phase">Phase ${escapeHtml(phase.number)}.${String(phaseLessonIndex + 1).padStart(2, '0')}</span>
        <span id="stage-bar-active-label" class="stage-bar-active-label">Stage 1 of ${stageCount} · Understand</span>
      </div>
      <div class="stage-bar-pills">
        <a class="stage-pill" data-stage-pill="Understand" href="#stage-understand" aria-current="step"><span class="stage-step-num">1</span> Understand</a>
        <a class="stage-pill" data-stage-pill="Build" href="#stage-build"><span class="stage-step-num">2</span> Build &amp; Run</a>
        ${c.experiments?.length ? '<a class="stage-pill" data-stage-pill="Experiment" href="#stage-experiment"><span class="stage-step-num">3</span> Experiment</a>' : ''}
        <a class="stage-pill" data-stage-pill="Practice" href="#stage-practice"><span class="stage-step-num">${stageNumber('Practice')}</span> Exercise &amp; Check</a>
        <a class="stage-pill" data-stage-pill="Evidence" href="#stage-evidence"><span class="stage-step-num">${stageNumber('Evidence')}</span> Evidence</a>
      </div>
      <div class="stage-bar-navigation" role="group" aria-label="Previous and next lesson activities">
        <a class="button" id="previous-activity" href="#lesson-title" hidden>Previous activity</a>
        <a class="button primary" id="next-activity" href="#stage-build">Next: Build</a>
      </div>
    </div>
    <div id="stage-understand" class="lesson-stage stage-understand" data-lesson-stage="Understand" data-stage-active="true">
      <div class="stage-transition-banner"><span class="stage-kicker">Stage 1 of ${stageCount} · Understand the Mechanism</span><span class="stage-summary">Core mental model, equations &amp; visual walkthrough</span></div>
      ${c.objectives ? `<h2>What you will be able to do</h2><ul>${c.objectives.map((o) => `<li>${inlineMath(o)}</li>`).join('')}</ul>` : ''}${c.problem ? `<h2>The problem</h2>${prose(c.problem)}` : ''}<h2>The idea</h2>${prose(c.idea)}${c.figure === 'gradient' ? gradientFigure() : c.figure === 'attention-mask' ? attentionMaskFigure() : ''}${typeof courseConceptFigure === 'function' ? courseConceptFigure(lesson.id) : ''}${deepConcepts(c, lesson)}${lessonVisual(lesson)}
    </div>
    <div id="stage-build" class="lesson-stage stage-build" data-lesson-stage="Build">
      <div class="stage-transition-banner"><span class="stage-kicker">Stage 2 of ${stageCount} · Build &amp; Run the Reference</span><span class="stage-summary">Step-by-step code construction &amp; verified execution output</span></div>
      ${c.setupBundle ? '<p class="stage-code-guide"><strong>Your workspace:</strong> Stay inside the extracted <code>jax-start-here</code> folder. The complete program is <code>first_experiment.py</code>. The blocks below explain that existing file; do not append a second copy. Use the OS-specific interpreter command from the setup steps.</p>' : '<p class="stage-code-guide"><strong>Your build:</strong> Create a separate <code>main.py</code> in your practice folder. Append each block in order and run <code>python main.py</code> in the lesson environment. The downloaded script is the complete reference, so keep it separate from your attempt.</p>'}
      ${(c.buildSteps || []).map((step, i) => `<section class="build-step"><h2><span class="build-step-number">${i + 1}</span> ${inlineMath(step.title.replace(/^\d+[.)]\s*/, ''))}</h2>${prose(step.instruction)}${codePanel(step.code, `${buildFile} · ${c.setupBundle ? 'read existing block' : 'append build step'} ${i + 1}`)}${prose(step.explanation)}${renderBuildStepCodeBreakdown(step.code)}</section>`).join('')}${c.buildSteps?.length ? `<details class="cumulative-build"><summary>View the cumulative build in ${buildFile}</summary><p>${c.setupBundle ? 'These blocks explain the file already included in your workspace.' : 'These blocks append in order to your separate practice file.'}</p>${codePanel(c.buildSteps.map((step) => step.code).join('\n\n'), `${buildFile} · cumulative Python build`)}</details>` : ''}${c.buildCode ? `<h2>Build a numerical estimate</h2>${codePanel(c.buildCode, 'Numerical estimate')}${prose(c.buildOutput)}` : ''}
      <h2>Run the example</h2>${c.setupBundle ? '<p>The workspace already contains this file as <code>first_experiment.py</code>. Run it with the command for your OS in step 5 above.</p>' : `<div class="run-command"><code>python3 ${escapeHtml(artifacts.scriptSource)}</code><button id="copy-run">Copy command</button></div><details class="command-context"><summary>About this command</summary><p class="soft">This shortcut uses an activated macOS/Linux course environment in the repository folder. For Windows, use the explicit interpreter command in “Setup help” above. The script includes worked examples and reference solutions.</p></details>`}${c.buildSteps?.length ? `<details class="assembled-program"><summary>Inspect the complete assembled example</summary>${codePanel(c.code, `${buildFile} · complete Python example`)}</details>` : codePanel(c.code, c.setupBundle ? 'first_experiment.py · Python' : 'main.py · worked Python example')}<p><strong>Expected result:</strong> ${inlineMath(c.output)}</p>
      ${lessonExecution(lesson)}
    </div>
    ${
      c.experiments?.length
        ? `<div id="stage-experiment" class="lesson-stage stage-experiment" data-lesson-stage="Experiment">
      <div class="stage-transition-banner"><span class="stage-kicker">Stage ${stageNumber('Experiment')} of ${stageCount} · Controlled Experiments</span><span class="stage-summary">Predict before running &amp; test boundary conditions</span></div>
      ${workedExperiments(c)}
    </div>`
        : ''
    }
    <div id="stage-practice" class="lesson-stage stage-practice" data-lesson-stage="Practice">
      <div class="stage-transition-banner"><span class="stage-kicker">Stage ${stageNumber('Practice')} of ${stageCount} · Hands-On Exercise &amp; Checkpoint</span><span class="stage-summary">Modify the code yourself, solve transfer tasks &amp; test your understanding</span></div>
      <div class="exercise-workbench">
        <div class="exercise-workbench-header">
          <span class="exercise-badge">Hands-On Exercise · Modify &amp; Verify</span>
          <span class="exercise-hint">Run in the declared lesson environment</span>
        </div>
        <h2>Make it yours</h2>
        ${c.setupBundle ? '<p class="stage-code-guide"><strong>Where to make the change:</strong> Edit the existing <code>first_experiment.py</code> inside <code>jax-start-here</code>, then run it again using your OS-specific setup command. Save your prediction and output before opening the reference solution.</p>' : '<p class="stage-code-guide"><strong>Where to put your code:</strong> Modify your own <code>main.py</code>, or use the <em>Make it yours</em> cell in your downloaded notebook. Run your attempt in the declared environment and verify your prediction before opening the reference solution.</p>'}
        ${prose(c.exercise)}${renderExerciseCodeTeacher(c.solution, 'How to write this exercise code', c.learnerStarter)}<details><summary>Reveal the reference solution</summary>${codePanel(c.solution, 'Reference solution')}</details>
      </div>
      ${furtherPractice(c)}<form id="lesson-checkpoint" class="lesson-checkpoint"><h2>Check your understanding</h2>${prose(c.question)}${renderRadioGroup({ name: 'answer', legend: 'Choose one answer', required: true, options: c.options.map((label, i) => ({ value: String(i), label })) }, inlineMath)}<button class="primary" type="submit">Check my answer</button><p id="checkpoint-result" role="status"></p></form>
      ${c.diagnosis ? `<h2>Diagnose the result</h2>${prose(c.diagnosis)}` : ''}
    </div>
    <div id="stage-evidence" class="lesson-stage stage-evidence" data-lesson-stage="Evidence">
      <div class="stage-transition-banner"><span class="stage-kicker">Stage ${stageNumber('Evidence')} of ${stageCount} · Keep Your Evidence</span><span class="stage-summary">Record your verification receipt &amp; carry takeaways forward</span></div>
      ${c.takeaways ? `<h2>Carry forward</h2><ul>${c.takeaways.map((t) => `<li>${inlineMath(t)}</li>`).join('')}</ul>` : ''}<h2>Keep your evidence</h2><p><a href="${escapeHtml(artifacts.evidence)}" download>Download evidence template</a></p>${prose(lesson.evidence)}<p>Save your modified code, its output, and an explanation of your prediction. Checkpoint answers and exercise evidence are tracked separately.</p><div class="actions"><button id="exercise-complete"></button><button id="reset-lesson">Reset this lesson’s progress</button></div><p id="lesson-save-status" class="soft" role="status"></p><p class="soft">Saved in this browser. Export or back up your portfolio anytime in <a href="notebook.html">My learning</a>.</p><p><a class="button" href="notebook.html?lesson=${encodeURIComponent(lesson.id)}${activePath === 'all' ? '' : '&path=' + encodeURIComponent(activePath)}#portfolio">Save work for this lesson</a></p><p id="runtime-status" class="soft">Reference execution: CPU validation only. ${escapeHtml(scope.description)} ${escapeHtml(scope.target)} <a href="lesson-tpu-05.html">Execution matrix and target qualification</a>.</p>
      ${c?.references ? `<h2>Primary references</h2><ul>${c.references.map((r) => `<li><a href="${escapeHtml(r.url)}">${escapeHtml(r.title)} ↗</a></li>`).join('')}</ul>` : `<p><a href="${escapeHtml(lesson.source || phase.source)}">Primary documentation ↗</a></p>`}${phase.projectId && phase.lessons.filter((l) => l.status === 'authored').at(-1)?.id === lesson.id ? `<div class="phase-project-handoff"><p class="edition">Phase ${escapeHtml(phase.number)} Integration Project</p><h2>Bring this phase together: ${inlineMath(phase.project)}</h2><p><strong>Checkpoint to prove:</strong> ${inlineMath(phase.checkpoint)}</p>${phase.projectNote ? prose(phase.projectNote) : ''}<div class="actions"><a class="button primary" href="project.html?id=${encodeURIComponent(phase.projectId)}">Open project workspace${phase.projectStageIds?.length ? ' · stages ' + phase.projectStageIds.map(escapeHtml).join(', ') : ''}</a><a class="button" href="project-guides/${encodeURIComponent(phase.projectId)}.html">Read project guide &amp; figures</a></div></div>` : ''}
    </div>`
      : `<aside class="roadmap-notice"><h2>This lesson has not been written</h2><p>This is a roadmap entry. There is no teaching material, exercise, worked solution, or assessment available for this topic yet.</p><p><a class="button primary" href="lesson-welcome-01.html">Start the available foundation</a> <a href="course.html">Browse the course and roadmap</a></p></aside><p><a href="${escapeHtml(lesson.source || phase.source)}">Primary documentation ↗</a></p>`) +
    `<aside class="lesson-feedback"><strong>Help improve this lesson</strong><p>Something unclear, incorrect, or missing? <a href="${escapeHtml(feedbackUrl(lesson.title, `lesson-${lesson.id}.html`))}">Open a GitHub issue</a> with your feedback. The link includes the lesson and course version.</p>${feedbackAccess ? `<p class="soft">${escapeHtml(feedbackAccess)}</p>` : ''}</aside><div class="lesson-nav">${index > 0 ? `<button id="previous">Previous: ${escapeHtml(allLessons[index - 1].title)}</button>` : '<span>First lesson in this route</span>'}${index >= 0 && index < allLessons.length - 1 ? `<button id="next">Next: ${escapeHtml(allLessons[index + 1].title)}</button>` : '<button id="phase-project">View the phase project</button>'}</div>`
  );
}
