import {
  learnerStorageKeys,
  validateLearnerNotebook,
  learnerId,
  validateReadingState,
  normalizeLearnerBackup,
  mergeLearnerBackup,
  persistLearnerBackup,
} from '../lib/learner-records.js';
import { courseState, phaseById } from '../lib/course-state.js';
import { $ } from '../lib/dom.js';
import { evidenceReviewPacket } from '../lib/evidence.js';
import { escapeHtml } from '../lib/html.js';
import { planLessonSchedule, scheduleMetrics } from '../lib/schedule.js';

async function setupPlatform(routes) {
  const key = learnerStorageKeys.notebook;
  const allLessons = courseState.course.phases.flatMap((p) => p.lessons);
  const context = {
    routeIds: routes.map((r) => r.id),
    lessonIds: allLessons.map((l) => l.id),
    roleIds: courseState.course.roles.map((r) => r.id),
    phaseIds: courseState.course.phases.map((p) => p.id),
  };
  const read = (storageKey, fallback = null) => {
    try {
      const raw = localStorage.getItem(storageKey);
      return raw === null ? fallback : JSON.parse(raw);
    } catch {
      return fallback;
    }
  };
  let state = { version: 1, route: 'foundations', evidence: [] };
  const saved = read(key);
  if (validateLearnerNotebook(saved, context)) state = saved;
  let projects = [];
  try {
    const response = await fetch('api/v1/projects.json', { cache: 'no-store' });
    if (response.ok) projects = (await response.json()).projects;
  } catch {}
  let pendingExtras = null,
    editingId = null,
    removed = null;
  const status = (id, text) => {
    const element = $(id);
    if (element) element.textContent = text;
  };
  const download = (name, text, type) => {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  function storedProjects() {
    const values = {};
    try {
      for (let i = 0; i < localStorage.length; i++) {
        const storageKey = localStorage.key(i);
        if (storageKey?.startsWith('jaxpathways-project-')) {
          const id = storageKey.slice('jaxpathways-project-'.length);
          const stages = read(storageKey, []);
          if (learnerId(id) && Array.isArray(stages) && stages.every((s) => typeof s === 'string'))
            values[id] = stages;
        }
      }
    } catch {}
    return values;
  }
  function backup() {
    return {
      version: 2,
      notebook: state,
      sampleCompleted: courseState.completed,
      lessonProgress: courseState.lessonProgress,
      projects: pendingExtras?.projects || storedProjects(),
      careerPlan: pendingExtras?.careerPlan || read(learnerStorageKeys.careerPlan),
      reading: pendingExtras?.reading || read(learnerStorageKeys.reading),
    };
  }
  function save() {
    try {
      localStorage.setItem(key, JSON.stringify(state));
      status('#backup-status', '');
      return true;
    } catch {
      status(
        '#backup-status',
        'Device storage could not save these changes. They remain in this page only. Export a backup before closing.',
      );
      return false;
    }
  }
  function populate(select, entries, blank) {
    if (!select) return;
    select.replaceChildren();
    if (blank) {
      const option = document.createElement('option');
      option.value = '';
      option.textContent = blank;
      select.append(option);
    }
    for (const entry of entries) {
      const option = document.createElement('option');
      option.value = entry.id;
      option.textContent = entry.title;
      select.append(option);
    }
  }
  for (const selector of [
    '#active-route',
    '#evidence-route',
    '#club-route',
    '#evidence-filter-route',
  ])
    populate($(selector), routes, selector === '#evidence-filter-route' ? 'All pathways' : null);
  if ($('#club-route')) {
    $('#club-route').value = state.route;
    setupClubPlanner();
  }
  if (!$('#active-route')) return;
  $('#active-route').value = state.route;
  $('#continue').onclick = () => {
    location.href = `course.html?path=${encodeURIComponent(state.route)}`;
  };
  $('#active-route').onchange = (e) => {
    state.route = e.target.value;
    save();
    routeSummary();
  };
  const stageSelect = $('#evidence-stage');
  function updateContexts() {
    const route = routes.find((r) => r.id === $('#evidence-route').value);
    populate(
      $('#evidence-lesson'),
      route.lessons.filter((l) => l.status === 'authored'),
      'No lesson association',
    );
    populate($('#evidence-project'), projects, 'No project association');
    populate(stageSelect, [], 'Whole project / no stage');
  }
  function updateStages() {
    const project = projects.find((p) => p.id === $('#evidence-project').value);
    populate(
      stageSelect,
      (project?.stages || []).map((s) => ({ id: s.id, title: s.title })),
      'Whole project / no stage',
    );
  }
  $('#evidence-route').value = state.route;
  updateContexts();
  $('#evidence-route').onchange = updateContexts;
  $('#evidence-project').onchange = updateStages;
  $('#evidence-lesson').onchange = () => {
    if (!$('#evidence-title').value)
      $('#evidence-title').value =
        allLessons.find((l) => l.id === $('#evidence-lesson').value)?.title || '';
  };
  function clearForm() {
    editingId = null;
    $('#evidence-form').reset();
    $('#evidence-route').value = state.route;
    updateContexts();
    $('#evidence-submit').textContent = 'Save artifact';
    $('#cancel-edit').hidden = true;
  }
  $('#cancel-edit').onclick = () => {
    clearForm();
    status('#evidence-status', 'Edit cancelled. The saved artifact was kept.');
  };
  function association(entry) {
    const pieces = [];
    if (entry.lessonId)
      pieces.push(allLessons.find((l) => l.id === entry.lessonId)?.title || entry.lessonId);
    if (entry.projectId) {
      const p = projects.find((p) => p.id === entry.projectId);
      pieces.push(
        (p?.title || entry.projectId) +
          (entry.stageId
            ? ' / ' +
              (p?.stages.find((s) => s.id === entry.stageId)?.title || 'Stage ' + entry.stageId)
            : ''),
      );
    }
    return pieces.join(' · ');
  }
  function filteredEvidence() {
    const route = $('#evidence-filter-route').value,
      type = $('#evidence-filter-type').value;
    return state.evidence.filter(
      (e) =>
        (!route || e.route === route) &&
        (!type ||
          (type === 'lesson'
            ? e.lessonId
            : type === 'project'
              ? e.projectId
              : !e.lessonId && !e.projectId)),
    );
  }
  function exportReview(entries) {
    try {
      const packet = evidenceReviewPacket(entries, {
        routes,
        lessons: allLessons,
        projects,
        lessonProgress: courseState.lessonProgress,
        projectProgress: backup().projects,
        baseUrl: location.href,
        generatedAt: new Date().toISOString(),
      });
      download('jaxpathways-review-packet.md', packet, 'text/markdown');
      status(
        '#review-status',
        'Markdown review packet prepared for download. No review was performed and nothing was sent externally.',
      );
    } catch {
      status(
        '#review-status',
        'Add an artifact or clear the filters before preparing a packet. Your saved records were not changed.',
      );
    }
  }
  if ($('#review-export')) $('#review-export').onclick = () => exportReview(filteredEvidence());
  function renderEvidence() {
    const entries = filteredEvidence();
    if ($('#review-count'))
      $('#review-count').textContent =
        `${entries.length} artifact${entries.length === 1 ? '' : 's'} will be included using the filters above.`;
    $('#evidence-count').textContent =
      `${state.evidence.length} artifact${state.evidence.length === 1 ? '' : 's'} · self-reported; not reviewed`;
    $('#evidence-results').textContent =
      `${entries.length} shown. Saving an artifact does not tick lesson exercises or project stages.`;
    $('#evidence-list').replaceChildren();
    if (!entries.length) {
      const p = document.createElement('p');
      p.className = 'empty-state';
      p.textContent = state.evidence.length
        ? 'No saved work in this group.'
        : 'No work saved yet. You can keep learning without adding notes here.';
      $('#evidence-list').append(p);
    }
    for (const entry of entries.slice().reverse()) {
      const article = document.createElement('article');
      article.className = 'evidence-entry';
      const meta = document.createElement('p');
      meta.className = 'edition';
      meta.textContent =
        (routes.find((r) => r.id === entry.route)?.title || entry.route) +
        ' · ' +
        entry.date.slice(0, 10);
      const title = document.createElement('h3');
      title.textContent = entry.title;
      const associated = document.createElement('p');
      associated.className = 'artifact-context';
      associated.textContent = association(entry) || 'General experiment';
      const note = document.createElement('p');
      note.className = 'artifact-note';
      note.textContent = entry.note;
      article.append(meta, title, associated, note);
      const actions = document.createElement('div');
      actions.className = 'artifact-actions';
      if (entry.url) {
        const link = document.createElement('a');
        link.href = entry.url;
        link.textContent = 'Open artifact ↗';
        link.rel = 'noopener noreferrer';
        link.target = '_blank';
        actions.append(link);
      }
      if (entry.lessonId) {
        const link = document.createElement('a');
        link.href = `lesson.html?path=${encodeURIComponent(entry.route)}&lesson=${encodeURIComponent(entry.lessonId)}`;
        link.textContent = 'Related lesson';
        actions.append(link);
      }
      const review = document.createElement('button');
      review.textContent = 'Review packet';
      review.setAttribute('aria-label', 'Prepare review packet for ' + entry.title);
      review.onclick = () => exportReview([entry]);
      actions.append(review);
      const edit = document.createElement('button');
      edit.textContent = 'Edit';
      edit.setAttribute('aria-label', 'Edit ' + entry.title);
      edit.onclick = () => {
        if ($('#save-work')) $('#save-work').open = true;
        editingId = entry.id;
        $('#evidence-route').value = entry.route;
        updateContexts();
        $('#evidence-lesson').value = entry.lessonId || '';
        $('#evidence-project').value = entry.projectId || '';
        updateStages();
        stageSelect.value = entry.stageId || '';
        $('#evidence-title').value = entry.title;
        $('#evidence-url').value = entry.url;
        $('#evidence-note').value = entry.note;
        $('#evidence-submit').textContent = 'Save changes';
        $('#cancel-edit').hidden = false;
        $('#evidence-title').focus();
        status('#evidence-status', 'Editing ' + entry.title + '.');
      };
      const remove = document.createElement('button');
      remove.textContent = 'Remove';
      remove.setAttribute('aria-label', 'Remove ' + entry.title);
      remove.onclick = () => {
        removed = { entry, index: state.evidence.findIndex((e) => e.id === entry.id) };
        state.evidence = state.evidence.filter((e) => e.id !== entry.id);
        const persisted = save();
        renderEvidence();
        $('#undo-remove').hidden = false;
        $('#undo-remove').focus();
        status(
          '#evidence-status',
          persisted
            ? 'Artifact removed. Undo is available until your next removal.'
            : 'Artifact removed in this page only; device storage could not save. Undo or export before closing.',
        );
      };
      actions.append(edit, remove);
      article.append(actions);
      $('#evidence-list').append(article);
    }
  }
  $('#evidence-filter-route').onchange = renderEvidence;
  $('#evidence-filter-type').onchange = renderEvidence;
  $('#undo-remove').onclick = () => {
    if (!removed) return;
    state.evidence.splice(Math.min(removed.index, state.evidence.length), 0, removed.entry);
    removed = null;
    const persisted = save();
    renderEvidence();
    $('#undo-remove').hidden = true;
    status(
      '#evidence-status',
      persisted ? 'Artifact restored.' : 'Artifact restored in memory only. Export before closing.',
    );
  };
  function routeSummary() {
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
      ? `<p class="edition">Resume your reading</p><h3>${escapeHtml(last.title)}</h3><a class="button primary" href="lesson.html?lesson=${encodeURIComponent(last.id)}&path=${encodeURIComponent(reading.path)}${reading.sectionId ? '#' + encodeURIComponent(reading.sectionId) : ''}">Continue from your saved section</a><p class="soft">Reading position is separate from completing an exercise.</p>`
      : next
        ? `<p class="edition">${hasPractice ? 'Your next experiment' : 'Your learning starts here'}</p><h3>${escapeHtml(next.title)}</h3><a class="button primary" href="lesson.html?path=${route.id}&lesson=${next.id}">Start this lesson</a>`
        : '<p>All available exercises are self-reported complete. Review your artifacts; remaining course topics may still be unwritten.</p>';
    $('#lesson-progress-summary').innerHTML =
      resume +
      `<p>${exercised.length} / ${available.length} exercises reported · ${available.filter((l) => courseState.lessonProgress.lessons[l.id]?.checkpointPassed).length} checkpoints passed</p>${last && next ? `<a href="lesson.html?path=${route.id}&lesson=${next.id}">Next unmarked exercise on ${escapeHtml(route.title)}: ${escapeHtml(next.title)}</a>` : ''}`;
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
        `<p class="edition">Saved career plan</p><h3>${escapeHtml(role.title)}</h3><p>Suggested starting phase: ${escapeHtml(phase.title)}. This is based on your self-check.</p><a href="pathways.html?role=${encodeURIComponent(role.id)}#careers">Review my career plan</a>`;
  }
  $('#evidence-form').onsubmit = (e) => {
    e.preventDefault();
    const title = $('#evidence-title').value.trim(),
      note = $('#evidence-note').value.trim(),
      url = $('#evidence-url').value.trim();
    if (!title || !note) {
      status('#evidence-status', 'Add a title and describe what you verified.');
      return;
    }
    if (url && !/^https?:\/\//i.test(url)) {
      status('#evidence-status', 'Use an http or https artifact link.');
      return;
    }
    if (!editingId && state.evidence.length >= 500) {
      status(
        '#evidence-status',
        'This notebook has 500 artifacts. Export a backup before removing older entries.',
      );
      return;
    }
    const previous = state.evidence.find((e) => e.id === editingId);
    const now = new Date().toISOString();
    const entry = {
      id: editingId || crypto.randomUUID(),
      route: $('#evidence-route').value,
      title,
      url,
      note,
      date: previous?.date || now.slice(0, 10),
      updatedAt: now,
      lessonId: $('#evidence-lesson').value,
      projectId: $('#evidence-project').value,
      stageId: stageSelect.value,
    };
    if (previous) state.evidence[state.evidence.indexOf(previous)] = entry;
    else state.evidence.push(entry);
    const persisted = save();
    renderEvidence();
    if (persisted) {
      clearForm();
      status('#evidence-status', 'Artifact saved on this device. Self-reported; not reviewed.');
    } else {
      editingId = entry.id;
      $('#evidence-submit').textContent = 'Retry saving';
      $('#cancel-edit').hidden = false;
      status(
        '#evidence-status',
        'Saving to device storage failed. Your inputs are retained and the artifact is in this page only. Retry or export a backup before closing.',
      );
    }
  };
  $('#export').onclick = () => {
    try {
      const outgoing = normalizeLearnerBackup(backup(), context);
      download(
        'jaxpathways-learning-backup-v2.json',
        JSON.stringify(outgoing, null, 2),
        'application/json',
      );
      status(
        '#backup-status',
        'Backup downloaded: artifacts, lesson progress, project stages, career plan, and saved reading position.',
      );
    } catch {
      status(
        '#backup-status',
        'Some stored records have an unsupported format. Do not clear storage; keep your inputs and contact the course maintainer.',
      );
    }
  };
  $('#import-button').onclick = () => $('#import').click();
  $('#import').onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const preview = $('#restore-preview');
    if (preview) {
      preview.hidden = true;
      preview.replaceChildren();
    }
    try {
      if (file.size > 3000000) throw Error('File is too large');
      const incoming = JSON.parse(await file.text());
      // Validate before showing the merge action. No local records change yet.
      const checked = mergeLearnerBackup(backup(), incoming, context);
      const apply = () => {
        try {
          const merged = mergeLearnerBackup(backup(), incoming, context),
            persisted = persistLearnerBackup(localStorage, merged);
          state = merged.notebook;
          courseState.lessonProgress = merged.lessonProgress;
          courseState.completed = merged.sampleCompleted;
          pendingExtras = merged;
          $('#active-route').value = state.route;
          clearForm();
          renderEvidence();
          routeSummary();
          if (preview) {
            preview.hidden = true;
            preview.replaceChildren();
          }
          status(
            '#backup-status',
            persisted
              ? 'Backup merged and saved. Existing artifacts, checkpoints, and project stage reports were preserved.'
              : 'Backup merged into this page only. Device storage failed; export a new backup before closing. Existing storage was restored where possible.',
          );
        } catch {
          status(
            '#backup-status',
            'This backup could not be restored. Your current records are still available. Download them before trying again.',
          );
        }
      };
      if (!preview) {
        apply();
        return;
      }
      preview.hidden = false;
      const heading = document.createElement('h3');
      heading.textContent = 'Restore this backup?';
      heading.tabIndex = -1;
      const description = document.createElement('p');
      description.textContent = `${file.name}. After merging: ${checked.notebook.evidence.length} artifacts and ${Object.keys(checked.lessonProgress.lessons).length} lesson progress records. Existing checkpoints and stage reports are retained.`;
      const restore = document.createElement('button');
      restore.type = 'button';
      restore.className = 'primary';
      restore.textContent = 'Restore and merge';
      restore.onclick = apply;
      const cancel = document.createElement('button');
      cancel.type = 'button';
      cancel.textContent = 'Cancel';
      cancel.onclick = () => {
        preview.hidden = true;
        preview.replaceChildren();
        $('#import-button').focus();
      };
      preview.append(heading, description, restore, cancel);
      heading.focus();
      status('#backup-status', 'Backup checked. Choose Restore and merge to update this browser.');
    } catch {
      status(
        '#backup-status',
        'This file is not a supported JAX Pathways backup, or could not be read. Your current workspace was not changed. Choose the original JSON backup and try again.',
      );
    } finally {
      e.target.value = '';
    }
  };
  const params = new URLSearchParams(location.search);
  const contextRoute = params.get('path') || params.get('route');
  if (context.routeIds.includes(contextRoute)) $('#evidence-route').value = contextRoute;
  updateContexts();
  if (context.lessonIds.includes(params.get('lesson'))) {
    $('#evidence-lesson').value = params.get('lesson');
    $('#evidence-title').value = allLessons.find((l) => l.id === params.get('lesson')).title;
  }
  if (projects.some((p) => p.id === params.get('project'))) {
    $('#evidence-project').value = params.get('project');
    updateStages();
    stageSelect.value = params.get('stage') || '';
    if (!$('#evidence-title').value)
      $('#evidence-title').value = projects.find((p) => p.id === params.get('project')).title;
  }
  renderEvidence();
  routeSummary();
  function setupClubPlanner() {
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
            a.href = `lesson.html?path=${route.id}&lesson=${lesson.id}`;
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
          'Session: predict one result, run the build, compare a changed condition, and discuss one diagnostic artifact. Peer feedback is not reviewed certification.';
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
}

export { setupPlatform };
