import { $ } from '../lib/dom.js';
import { lessonLink } from '../lib/urls.js';
import { planLessonSchedule, scheduleMetrics } from '../lib/schedule.js';
import { downloadText, setStatus } from './browser-actions.js';

const download = downloadText;
const status = setStatus;

export function setupClubPlanner(routes) {
  let planText = '';
  function render() {
    const route = routes.find((r) => r.id === $('#club-route').value),
      weeks = Number($('#club-weeks').value),
      budget = Number($('#club-budget').value);
    const available = route.lessons.filter((l) => l.status === 'authored');
    const unwritten = route.lessons.filter((l) => l.status !== 'authored');
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
    for (const [title, value, detail] of [
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
    ]) {
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
    $('#club-summary').replaceChildren(summary);
    $('#club-plan').replaceChildren();

    planText = `# ${route.title}: ${weeks}-week learning club\n\nCPU-first facilitator draft. Lesson minutes are estimates, not mastery promises. Budget ${budget} minutes/week; projects, setup failures and review may require extra sessions.\n\n`;
    assignments.forEach((lessons, i) => {
      const article = document.createElement('article');
      const h = document.createElement('h3');
      h.textContent = `Week ${i + 1} · ${used[i]} / ${budget} minutes`;
      article.append(h);
      const list = document.createElement('ul');
      if (lessons.length)
        for (const lesson of lessons) {
          const li = document.createElement('li');
          const a = document.createElement('a');
          a.href = lessonLink(lesson.id, route.id);
          a.textContent = `${lesson.title} · ${lesson.minutes || 60} min`;
          li.append(a);
          list.append(li);
        }
      else {
        const li = document.createElement('li');
        li.textContent = 'Discussion and evidence review; no new lesson allocated.';
        list.append(li);
      }
      article.append(list);
      const sessionGuidance =
        'Session: predict one result, run the build, compare a changed condition, and review one diagnostic artifact together.';
      $('#club-plan').append(article);
      planText += `## Week ${i + 1} (${used[i]} / ${budget} min)\n${lessons.map((l) => `- ${l.title} (${l.minutes || 60} min)`).join('\n') || '- Discussion and evidence review'}\n\n${sessionGuidance}\n\n`;
    });
    const blocked = [
      ...overflow.map(
        (l) =>
          `${l.title}: available, but needs a later or longer session (${l.minutes || 60} min)`,
      ),
      ...unwritten.map((l) => `${l.title}: unwritten; do not assign as a runnable lesson`),
    ];
    $('#club-blocked').replaceChildren();
    for (const text of blocked) {
      const li = document.createElement('li');
      li.textContent = text;
      $('#club-blocked').append(li);
    }
    if (!blocked.length) {
      const li = document.createElement('li');
      li.textContent =
        'All available lessons fit the chosen budget. Allocate additional project time separately.';
      $('#club-blocked').append(li);
    }
    planText += `## Work outside this schedule\n${blocked.map((text) => '- ' + text).join('\n')}\n\n## Integration project\n${route.capstone.title}\n${route.capstone.projectId ? 'Implemented public-check project; allocate additional build and evidence review time.' : 'Project brief is planned; do not promise a runnable capstone.'}\n`;
    $('#club-download').disabled = false;
    $('#club-download').onclick = () =>
      download('jaxpathways-club-plan.md', planText, 'text/markdown');
  }
  const build = document.createElement('button');
  build.type = 'button';
  build.className = 'primary';
  build.textContent = 'Build schedule';
  $('#club-download').before(build);
  build.onclick = render;
  for (const selector of ['#club-route', '#club-weeks', '#club-budget'])
    $(selector).onchange = () => {
      status('#club-summary', 'Choices changed. Build the schedule to see your new plan.');
      $('#club-download').disabled = true;
    };
  render();
}
