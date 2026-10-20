import { $ } from '../lib/dom.js';
import { lessonLink } from '../lib/urls.js';
import { planLessonSchedule, scheduleMetrics } from '../lib/schedule.js';
import { downloadText, setStatus } from './browser-actions.js';

const download = downloadText;
const status = setStatus;

/**
 * Render the facilitator planner when its page controls are present.
 * @param {(import('../types/course').Pathway & { lessons: import('../types/course').Lesson[] })[]} routes
 */
export function setupClubPlanner(routes) {
  const routeSelect = /** @type {HTMLSelectElement | null} */ ($('#club-route'));
  const weeksSelect = /** @type {HTMLSelectElement | null} */ ($('#club-weeks'));
  const budgetSelect = /** @type {HTMLSelectElement | null} */ ($('#club-budget'));
  const summaryContainer = $('#club-summary');
  const planContainer = $('#club-plan');
  const blockedList = $('#club-blocked');
  const downloadButton = /** @type {HTMLButtonElement | null} */ ($('#club-download'));
  if (
    !(routeSelect instanceof HTMLSelectElement) ||
    !(weeksSelect instanceof HTMLSelectElement) ||
    !(budgetSelect instanceof HTMLSelectElement) ||
    !summaryContainer ||
    !planContainer ||
    !blockedList ||
    !(downloadButton instanceof HTMLButtonElement)
  ) {
    return;
  }
  const routeControl = /** @type {HTMLSelectElement} */ (routeSelect);
  const weeksControl = /** @type {HTMLSelectElement} */ (weeksSelect);
  const budgetControl = /** @type {HTMLSelectElement} */ (budgetSelect);
  const summaryTarget = /** @type {HTMLElement} */ (summaryContainer);
  const plan = /** @type {HTMLElement} */ (planContainer);
  const blockedTarget = /** @type {HTMLElement} */ (blockedList);
  const downloadControl = /** @type {HTMLButtonElement} */ (downloadButton);

  let planText = '';
  function render() {
    const route = routes.find((candidate) => candidate.id === routeControl.value);
    if (!route) return;
    const weeks = Number(weeksControl.value);
    const budget = Number(budgetControl.value);
    const available = route.lessons.filter((lesson) => lesson.status === 'authored');
    const unwritten = route.lessons.filter((lesson) => lesson.status !== 'authored');
    const { assignments, used, overflow, total, capacity } = planLessonSchedule(
      available,
      weeks,
      budget,
    );
    const metrics = scheduleMetrics(
      { assignments, used, overflow, total, capacity },
      unwritten.length,
    );
    const summary = document.createElement('dl');
    summary.className = 'club-metrics';
    /** @type {[string, string, string][]} */
    const summaryRows = [
      [
        'Assigned',
        `${metrics.assignedHours} hours`,
        `${metrics.assignedLessons} lessons in this schedule`,
      ],
      ['Capacity', `${metrics.capacityHours} hours`, `${weeks} weeks × ${budget / 60} hours`],
      [
        'Still to schedule',
        `${metrics.remainingLessons} lessons`,
        `${metrics.remainingHours} estimated hours`,
      ],
      [
        'Unwritten',
        `${metrics.unwritten} ${metrics.unwritten === 1 ? 'topic' : 'topics'}`,
        'Outside the runnable plan',
      ],
    ];
    for (const [title, value, detail] of summaryRows) {
      const item = document.createElement('div'),
        label = document.createElement('dt'),
        number = document.createElement('dd'),
        description = document.createElement('p');
      label.textContent = title;
      number.textContent = value;
      description.textContent = detail;
      item.append(label, number, description);
      summary.append(item);
    }
    summaryTarget.replaceChildren(summary);
    plan.replaceChildren();

    planText = `# ${route.title}: ${weeks}-week learning club\n\nCPU-first facilitator draft. Lesson minutes are estimates, not mastery promises. Budget ${budget} minutes/week; projects, setup failures and review may require extra sessions.\n\n`;
    assignments.forEach((lessons, index) => {
      const article = document.createElement('article');
      const heading = document.createElement('h3');
      heading.textContent = `Week ${index + 1} · ${used[index]} / ${budget} minutes`;
      article.append(heading);
      const list = document.createElement('ul');
      if (lessons.length) {
        for (const lesson of lessons) {
          const item = document.createElement('li');
          const link = document.createElement('a');
          link.href = lessonLink(lesson.id, route.id);
          link.textContent = `${lesson.title} · ${lesson.minutes || 60} min`;
          item.append(link);
          list.append(item);
        }
      } else {
        const item = document.createElement('li');
        item.textContent = 'Discussion and evidence review; no new lesson allocated.';
        list.append(item);
      }
      article.append(list);
      const sessionGuidance =
        'Session: predict one result, run the build, compare a changed condition, and review one diagnostic artifact together.';
      plan.append(article);
      planText += `## Week ${index + 1} (${used[index]} / ${budget} min)\n${lessons.map((lesson) => `- ${lesson.title} (${lesson.minutes || 60} min)`).join('\n') || '- Discussion and evidence review'}\n\n${sessionGuidance}\n\n`;
    });
    const blocked = [
      ...overflow.map(
        (lesson) =>
          `${lesson.title}: available, but needs a later or longer session (${lesson.minutes || 60} min)`,
      ),
      ...unwritten.map(
        (lesson) => `${lesson.title}: unwritten; do not assign as a runnable lesson`,
      ),
    ];
    blockedTarget.replaceChildren();
    for (const text of blocked) {
      const item = document.createElement('li');
      item.textContent = text;
      blockedTarget.append(item);
    }
    if (!blocked.length) {
      const item = document.createElement('li');
      item.textContent =
        'All available lessons fit the chosen budget. Allocate additional project time separately.';
      blockedTarget.append(item);
    }
    planText += `## Work outside this schedule\n${blocked.map((text) => '- ' + text).join('\n')}\n\n## Integration project\n${route.capstone.title}\n${route.capstone.projectId ? 'Implemented public-check project; allocate additional build and evidence review time.' : 'Project brief is planned; do not promise a runnable capstone.'}\n`;
    downloadControl.disabled = false;
    downloadControl.onclick = () => download('jaxpathways-club-plan.md', planText, 'text/markdown');
  }
  const build = document.createElement('button');
  build.type = 'button';
  build.className = 'primary';
  build.textContent = 'Build schedule';
  downloadControl.before(build);
  build.onclick = render;
  for (const select of [routeControl, weeksControl, budgetControl])
    select.onchange = () => {
      status('#club-summary', 'Choices changed. Build the schedule to see your new plan.');
      downloadControl.disabled = true;
    };
  render();
}
