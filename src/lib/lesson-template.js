import { revisionMarkup, feedbackUrl, feedbackAccess } from './releases.js';
import { lessonVisual, lessonExecution } from './lesson-visuals.js';
import { inlineMath } from './math.js';
import { escapeHtml } from './html.js';
import { renderRadioGroup } from '../components/ui/radio-group.js';
import { runtimeLabel, practiceKind, courseTypeCue } from './runtime.js';
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
import { renderBuildStepCodeBreakdown, renderExerciseCodeTeacher } from './code-teaching.js';
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
    phaseLessonIndex = Math.max(
      0,
      phase.lessons.findIndex((l) => l.id === lesson.id),
    );
  return (
    `<div class="course-cue-row"><span class="course-type-pill" data-course-kind="${escapeHtml(cue.kind)}">${escapeHtml(cue.badge)}</span><span class="course-meta-pill">${escapeHtml(cue.hardwareBadge)}</span><span class="course-meta-pill">Lesson ${phaseLessonIndex + 1} of ${phase.lessons.length} in Phase ${escapeHtml(phase.number)}</span></div>` +
    `<p class="edition">Phase ${phase.number} / ${escapeHtml(phase.title)}</p><h1 id="lesson-title" tabindex="-1">${escapeHtml(lesson.title)}</h1>${revisionMarkup('lessons', lesson.id)}<p class="lesson-status">${authored ? `Lesson · about ${lesson.minutes} minutes · ${runtimeLabel(lesson)}` : 'TODO · lesson in progress'}</p><p>${inlineMath(lesson.objective)}</p><p><a href="phase-guides/${encodeURIComponent(phase.id)}.html">Phase study guide: preparation, milestones and project evidence</a></p><details class="before-start"><summary>Before you start</summary><p class="soft"><strong>Before this:</strong> ${prerequisiteLinks}<br><strong>Hardware:</strong> ${inlineMath(lesson.hardware || phase.hardware)}</p>${authored ? `<aside class="runtime-note"><strong>${runtimeLabel(lesson)}</strong><p>${practiceKind(lesson) === 'virtual-cpu' ? 'Configures multiple logical CPU devices to practice device placement, meshes, and SPMD sharding locally. Stage to a Cloud TPU VM (Phases 19–21) to benchmark physical TPU interconnect and MXU throughput.' : 'Runs locally on CPU out of the box, and stages directly to a Cloud TPU VM using the Phase 19–21 workflow.'}</p></aside>` : ''}</details>` +
    (authored && c && artifacts
      ? `
    <div class="lesson-artifacts">${c.setupBundle ? `<a class="button primary" href="${escapeHtml(c.setupBundle.url)}" download>${escapeHtml(c.setupBundle.title)}</a>` : ''}${!c.setupBundle ? `<a class="button" href="${escapeHtml(artifacts.notebook)}" download>Download notebook</a>` : ''}<details class="lesson-resources"><summary>Other downloads</summary>${c.setupBundle ? `<a href="${escapeHtml(artifacts.notebook)}" download>Lesson notebook</a>` : ''}<a href="${escapeHtml(artifacts.script)}" download>Python worked examples</a><a href="${escapeHtml(artifacts.markdown)}" download>Lesson text</a></details></div>
    ${practiceDrawer(lesson)}
    <div id="lesson-stage-bar" class="lesson-stage-bar" data-active-stage="Understand" aria-label="Lesson stage progress">
      <div class="stage-bar-context">
        <span class="stage-bar-phase">Phase ${escapeHtml(phase.number)}.${String(phaseLessonIndex + 1).padStart(2, '0')}</span>
        <span id="stage-bar-active-label" class="stage-bar-active-label">Stage 1 of 5 · Understand</span>
      </div>
      <div class="stage-bar-pills">
        <a class="stage-pill" data-stage-pill="Understand" href="#stage-understand" aria-current="step"><span class="stage-step-num">1</span> Understand</a>
        <a class="stage-pill" data-stage-pill="Build" href="#stage-build"><span class="stage-step-num">2</span> Build &amp; Run</a>
        ${c.experiments?.length ? '<a class="stage-pill" data-stage-pill="Experiment" href="#stage-experiment"><span class="stage-step-num">3</span> Experiment</a>' : ''}
        <a class="stage-pill" data-stage-pill="Practice" href="#stage-practice"><span class="stage-step-num">4</span> Exercise &amp; Check</a>
        <a class="stage-pill" data-stage-pill="Evidence" href="#stage-evidence"><span class="stage-step-num">5</span> Evidence</a>
      </div>
    </div>
    <div id="stage-understand" class="lesson-stage stage-understand" data-lesson-stage="Understand" data-stage-active="true">
      <div class="stage-transition-banner"><span class="stage-kicker">Stage 1 of 5 · Understand the Mechanism</span><span class="stage-summary">Core mental model, equations &amp; visual walkthrough</span></div>
      ${c.objectives ? `<h2>What you will be able to do</h2><ul>${c.objectives.map((o) => `<li>${inlineMath(o)}</li>`).join('')}</ul>` : ''}${c.problem ? `<h2>The problem</h2>${prose(c.problem)}` : ''}<h2>The idea</h2>${prose(c.idea)}${c.figure === 'gradient' ? gradientFigure() : c.figure === 'attention-mask' ? attentionMaskFigure() : ''}${typeof courseConceptFigure === 'function' ? courseConceptFigure(lesson.id) : ''}${deepConcepts(c, lesson)}${lessonVisual(lesson)}
    </div>
    <div id="stage-build" class="lesson-stage stage-build" data-lesson-stage="Build">
      <div class="stage-transition-banner"><span class="stage-kicker">Stage 2 of 5 · Build &amp; Run the Reference</span><span class="stage-summary">Step-by-step code construction &amp; verified execution output</span></div>
      <p class="stage-code-guide"><strong>Where to put &amp; run this code:</strong> Work in <code>${escapeHtml(artifacts.scriptSource)}</code> (or downloaded <code>${escapeHtml(artifacts.script)}</code> / <code>${escapeHtml(artifacts.notebook)}</code>) and execute from the repository root with <code>python3 ${escapeHtml(artifacts.scriptSource)}</code>.</p>
      ${(c.buildSteps || []).map((step, i) => `<section class="build-step"><h2><span class="build-step-number">${i + 1}</span> ${inlineMath(step.title.replace(/^\d+[.)]\s*/, ''))}</h2>${prose(step.instruction)}${codePanel(step.code, `main.py · append build step ${i + 1}`)}${renderBuildStepCodeBreakdown(step.code)}${prose(step.explanation)}</section>`).join('')}${c.buildSteps?.length ? `<details class="cumulative-build"><summary>View the cumulative build in main.py</summary><p>These blocks append in order to the same file.</p>${codePanel(c.buildSteps.map((step) => step.code).join('\n\n'), 'main.py · cumulative Python build')}</details>` : ''}${c.buildCode ? `<h2>Build a numerical estimate</h2>${codePanel(c.buildCode, 'Numerical estimate')}${prose(c.buildOutput)}` : ''}
      <h2>Run the example</h2>${c.setupBundle ? '<p>The workspace already contains this file as <code>first_experiment.py</code>. Run it with the command for your OS in step 5 above.</p>' : `<div class="run-command"><code>python3 ${escapeHtml(artifacts.scriptSource)}</code><button id="copy-run">Copy command</button></div><details class="command-context"><summary>About this command</summary><p class="soft">This shortcut uses an activated macOS/Linux course environment in the repository folder. For Windows, use the explicit interpreter command in “Setup help” above. The script includes worked examples and reference solutions.</p></details>`}${c.buildSteps?.length ? `<details class="assembled-program"><summary>Inspect the complete assembled example</summary>${codePanel(c.code, 'main.py · complete Python example')}</details>` : codePanel(c.code, c.setupBundle ? 'first_experiment.py · Python' : 'main.py · worked Python example')}<p><strong>Expected result:</strong> ${inlineMath(c.output)}</p>
      ${lessonExecution(lesson)}
    </div>
    ${
      c.experiments?.length
        ? `<div id="stage-experiment" class="lesson-stage stage-experiment" data-lesson-stage="Experiment">
      <div class="stage-transition-banner"><span class="stage-kicker">Stage 3 of 5 · Controlled Experiments</span><span class="stage-summary">Predict before running &amp; test boundary conditions</span></div>
      ${workedExperiments(c)}
    </div>`
        : ''
    }
    <div id="stage-practice" class="lesson-stage stage-practice" data-lesson-stage="Practice">
      <div class="stage-transition-banner"><span class="stage-kicker">Stage 4 of 5 · Hands-On Exercise &amp; Checkpoint</span><span class="stage-summary">Modify the code yourself, solve transfer tasks &amp; test your understanding</span></div>
      <div class="exercise-workbench">
        <div class="exercise-workbench-header">
          <span class="exercise-badge">Hands-On Exercise · Modify &amp; Verify</span>
          <span class="exercise-hint">Run in your local Python environment or Cloud TPU VM</span>
        </div>
        <h2>Make it yours</h2>
        <p class="stage-code-guide"><strong>Where to put your code:</strong> Append your solution at the bottom of <code>${escapeHtml(artifacts.scriptSource)}</code> (or inside the <em>Make it yours</em> cell of <code>${escapeHtml(artifacts.notebook)}</code>) and run <code>python3 ${escapeHtml(artifacts.scriptSource)}</code> to verify your prediction before opening the reference solution.</p>
        ${prose(c.exercise)}${renderExerciseCodeTeacher(c.solution, 'How to write this exercise code')}<details><summary>Reveal the reference solution</summary>${codePanel(c.solution, 'Reference solution')}</details>
      </div>
      ${furtherPractice(c)}<form id="lesson-checkpoint" class="lesson-checkpoint"><h2>Check your understanding</h2>${prose(c.question)}${renderRadioGroup({ name: 'answer', legend: 'Choose one answer', required: true, options: c.options.map((label, i) => ({ value: String(i), label })) }, inlineMath)}<button class="primary" type="submit">Check my answer</button><p id="checkpoint-result" role="status"></p></form>
      ${c.diagnosis ? `<h2>Diagnose the result</h2>${prose(c.diagnosis)}` : ''}
    </div>
    <div id="stage-evidence" class="lesson-stage stage-evidence" data-lesson-stage="Evidence">
      <div class="stage-transition-banner"><span class="stage-kicker">Stage 5 of 5 · Keep Your Evidence</span><span class="stage-summary">Record your verification receipt &amp; carry takeaways forward</span></div>
      ${c.takeaways ? `<h2>Carry forward</h2><ul>${c.takeaways.map((t) => `<li>${inlineMath(t)}</li>`).join('')}</ul>` : ''}<h2>Keep your evidence</h2><p><a href="${escapeHtml(artifacts.evidence)}" download>Download evidence template</a></p>${prose(lesson.evidence)}<p>Save your modified code, its output, and an explanation of your prediction. Checkpoint answers and exercise evidence are tracked separately.</p><div class="actions"><button id="exercise-complete"></button><button id="reset-lesson">Reset this lesson’s progress</button></div><p id="lesson-save-status" class="soft" role="status"></p><p class="soft">Saved in this browser. Export or back up your portfolio anytime in <a href="notebook.html">My learning</a>.</p><p><a class="button" href="notebook.html?lesson=${encodeURIComponent(lesson.id)}${activePath === 'all' ? '' : '&path=' + encodeURIComponent(activePath)}#portfolio">Save work for this lesson</a></p><p id="runtime-status" class="soft">Verified on local CPU. Stage to a Cloud TPU VM via <a href="tpu-gcp.html">Cloud TPU setup</a> to record accelerator execution receipts (<strong>TODO:</strong> add automated Cloud TPU CI execution receipts).</p>
      ${c?.references ? `<h2>Primary references</h2><ul>${c.references.map((r) => `<li><a href="${escapeHtml(r.url)}">${escapeHtml(r.title)} ↗</a></li>`).join('')}</ul>` : `<p><a href="${escapeHtml(lesson.source || phase.source)}">Primary documentation ↗</a></p>`}${phase.projectId && phase.lessons.filter((l) => l.status === 'authored').at(-1)?.id === lesson.id ? `<div class="phase-project-handoff"><p class="edition">Phase ${escapeHtml(phase.number)} Integration Project</p><h2>Bring this phase together: ${inlineMath(phase.project)}</h2><p><strong>Checkpoint to prove:</strong> ${inlineMath(phase.checkpoint)}</p>${phase.projectNote ? prose(phase.projectNote) : ''}<div class="actions"><a class="button primary" href="project.html?id=${encodeURIComponent(phase.projectId)}">Open project workspace${phase.projectStageIds?.length ? ' · stages ' + phase.projectStageIds.map(escapeHtml).join(', ') : ''}</a><a class="button" href="project-guides/${encodeURIComponent(phase.projectId)}.html">Read project guide &amp; figures</a></div></div>` : ''}
    </div>`
      : `<aside class="roadmap-notice"><h2>This lesson has not been written</h2><p>This is a roadmap entry. There is no teaching material, exercise, worked solution, or assessment available for this topic yet.</p><p><a class="button primary" href="lesson-welcome-01.html">Start the available foundation</a> <a href="course.html">Browse the course and roadmap</a></p></aside><p><a href="${escapeHtml(lesson.source || phase.source)}">Primary documentation ↗</a></p>`) +
    `<aside class="lesson-feedback"><strong>Help improve this lesson</strong><p>Something unclear, incorrect, or missing? <a href="${escapeHtml(feedbackUrl(lesson.title, `lesson-${lesson.id}.html`))}">Open a GitHub issue</a> with your feedback. The link includes the lesson and course version.</p>${feedbackAccess ? `<p class="soft">${escapeHtml(feedbackAccess)}</p>` : ''}</aside><div class="lesson-nav">${index > 0 ? `<button id="previous">Previous: ${escapeHtml(allLessons[index - 1].title)}</button>` : '<span>First lesson in this route</span>'}${index >= 0 && index < allLessons.length - 1 ? `<button id="next">Next: ${escapeHtml(allLessons[index + 1].title)}</button>` : '<button id="phase-project">View the phase project</button>'}</div>`
  );
}
