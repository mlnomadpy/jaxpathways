import { $ } from '../../lib/dom.js';
import { courseState, phaseById } from '../../lib/course-state.js';
import { validateReadingState } from '../../lib/learner-records.js';
import { escapeHtml } from '../../lib/html.js';
import { lessonLink } from '../../lib/urls.js';

export function renderNotebookSummary({ routes, projects, allLessons, context, state, backup }) {
  const route = routes.find((r) => r.id === state.route);
  $('#route-next').textContent = route.outcome;
  if ($('#active-route-title')) $('#active-route-title').textContent = route.title;
  const available = route.lessons.filter((l) => l.status === 'authored');
  const exercised = available.filter(
    (l) => courseState.lessonProgress.lessons[l.id]?.evidenceSaved,
  );
  const storedReading = backup().reading;
  const reading = validateReadingState(storedReading, context) ? storedReading : null;
  const last = allLessons.find((l) => l.id === reading?.lastLessonId && l.status === 'authored');
  const next = available.find((l) => !courseState.lessonProgress.lessons[l.id]?.evidenceSaved);
  const hasPractice = Object.keys(courseState.lessonProgress.lessons).length > 0;
  const resume = last
    ? `<p class="edition">Resume your reading</p><h3>${escapeHtml(last.title)}</h3><a class="button primary" href="${lessonLink(last.id, reading.path)}${reading.sectionId ? '#' + encodeURIComponent(reading.sectionId) : ''}">Continue from your saved section</a><p class="soft">Reading position is separate from completing an exercise.</p>`
    : next
      ? `<p class="edition">${hasPractice ? 'Your next experiment' : 'Your learning starts here'}</p><h3>${escapeHtml(next.title)}</h3><a class="button primary" href="${lessonLink(next.id, route.id)}">Start this lesson</a>`
      : '<p>All available exercises on this route are complete. Review your saved portfolio artifacts below.</p>';
  $('#lesson-progress-summary').innerHTML =
    resume +
    `<p>${exercised.length} / ${available.length} exercises reported · ${available.filter((l) => courseState.lessonProgress.lessons[l.id]?.checkpointPassed).length} checkpoints passed</p>${last && next ? `<a href="${lessonLink(next.id, route.id)}">Next unmarked exercise on ${escapeHtml(route.title)}: ${escapeHtml(next.title)}</a>` : ''}`;
  const reportedProjects = projects.filter((p) => (backup().projects[p.id] || []).length);
  $('#project-progress-summary').innerHTML = reportedProjects.length
    ? `<h3>Your project practice</h3>${reportedProjects
        .map((p) => {
          const checked = backup().projects[p.id] || [],
            next = p.stages.find((s) => !checked.includes(s.id)) || p.stages[0];
          return `<p><a class="button" href="project.html?id=${encodeURIComponent(p.id)}#stage-${encodeURIComponent(next.id)}">Resume ${escapeHtml(p.title)}</a> · ${checked.length} / ${p.stages.length} stages reported</p>`;
        })
        .join('')}`
    : '<p>Ready to build something? <a href="project.html?id=regression-audit">Explore the regression project</a></p>';
  const plan = backup().careerPlan,
    role = courseState.course.roles.find((r) => r.id === plan?.roleId),
    phase = phaseById(plan?.startPhaseId);
  $('#saved-career').hidden = !role || !phase;
  if (role && phase)
    $('#saved-career').innerHTML =
      `<p class="edition">Saved career plan</p><h3>${escapeHtml(role.title)}</h3><p>Suggested starting phase: ${escapeHtml(phase.title)}. This is based on your self-check.</p><a href="careers.html?role=${encodeURIComponent(role.id)}#roles">Review my career plan</a>`;
}
