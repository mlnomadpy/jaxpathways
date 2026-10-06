import { setSelectValue } from '../browser-actions.js';
import { $ } from '../../lib/dom.js';
import { courseState } from '../../lib/course-state.js';
import { evidenceReviewPacket } from '../../lib/evidence.js';
import {
  downloadText as download,
  setStatus as status,
  populateSelect as populate,
} from '../browser-actions.js';

export function setupEvidence({ routes, projects, allLessons, context, getState, save, backup }) {
  let editingId = null,
    removed = null;
  const stageSelect = $('#evidence-stage');
  function updateWorkContext() {
    const target = $('#work-context');
    if (target)
      target.textContent = [
        '#evidence-route',
        '#evidence-lesson',
        '#evidence-project',
        '#evidence-stage',
      ]
        .map((selector) => {
          const select = $(selector);
          return select.value ? select.selectedOptions[0]?.textContent : '';
        })
        .filter(Boolean)
        .join(' / ');
  }
  for (const selector of [
    '#evidence-route',
    '#evidence-lesson',
    '#evidence-project',
    '#evidence-stage',
  ]) {
    $(selector).addEventListener('change', updateWorkContext);
  }

  function updateContexts() {
    const route = routes.find((r) => r.id === $('#evidence-route').value);
    populate(
      $('#evidence-lesson'),
      route.lessons.filter((l) => l.status === 'authored'),
      'No lesson association',
    );
    populate($('#evidence-project'), projects, 'No project association');
    populate(stageSelect, [], 'Whole project / no stage');
    updateWorkContext();
  }
  function updateStages() {
    const project = projects.find((p) => p.id === $('#evidence-project').value);
    populate(
      stageSelect,
      (project?.stages || []).map((s) => ({ id: s.id, title: s.title })),
      'Whole project / no stage',
    );
    updateWorkContext();
  }
  setSelectValue($('#evidence-route'), getState().route);
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
    setSelectValue($('#evidence-route'), getState().route);
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
    return getState().evidence.filter(
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
      `${getState().evidence.length} artifact${getState().evidence.length === 1 ? '' : 's'} · self-reported; not reviewed`;
    $('#evidence-results').textContent =
      `${entries.length} shown. Saving an artifact does not tick lesson exercises or project stages.`;
    $('#evidence-list').replaceChildren();
    if (!entries.length) {
      const p = document.createElement('p');
      p.className = 'empty-state';
      p.textContent = getState().evidence.length
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
        setSelectValue($('#evidence-route'), entry.route);
        updateContexts();
        setSelectValue($('#evidence-lesson'), entry.lessonId || '');
        setSelectValue($('#evidence-project'), entry.projectId || '');
        updateStages();
        setSelectValue(stageSelect, entry.stageId || '');
        updateWorkContext();
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
        removed = { entry, index: getState().evidence.findIndex((e) => e.id === entry.id) };
        getState().evidence = getState().evidence.filter((e) => e.id !== entry.id);
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
    getState().evidence.splice(
      Math.min(removed.index, getState().evidence.length),
      0,
      removed.entry,
    );
    removed = null;
    const persisted = save();
    renderEvidence();
    $('#undo-remove').hidden = true;
    status(
      '#evidence-status',
      persisted ? 'Artifact restored.' : 'Artifact restored in memory only. Export before closing.',
    );
  };
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
    if (!editingId && getState().evidence.length >= 500) {
      status(
        '#evidence-status',
        'This notebook has 500 artifacts. Export a backup before removing older entries.',
      );
      return;
    }
    const previous = getState().evidence.find((e) => e.id === editingId);
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
    if (previous) getState().evidence[getState().evidence.indexOf(previous)] = entry;
    else getState().evidence.push(entry);
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
  const params = new URLSearchParams(location.search);
  const contextRoute = params.get('path') || params.get('route');
  if (context.routeIds.includes(contextRoute)) setSelectValue($('#evidence-route'), contextRoute);
  updateContexts();
  if (context.lessonIds.includes(params.get('lesson'))) {
    setSelectValue($('#evidence-lesson'), params.get('lesson'));
    $('#evidence-title').value = allLessons.find((l) => l.id === params.get('lesson')).title;
  }
  if (projects.some((p) => p.id === params.get('project'))) {
    setSelectValue($('#evidence-project'), params.get('project'));
    updateStages();
    setSelectValue(stageSelect, params.get('stage') || '');
    if (!$('#evidence-title').value)
      $('#evidence-title').value = projects.find((p) => p.id === params.get('project')).title;
  }
  updateWorkContext();
  renderEvidence();
  return { clearForm, renderEvidence };
}
