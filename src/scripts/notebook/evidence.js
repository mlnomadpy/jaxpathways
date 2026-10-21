// @ts-check

import { setSelectValue } from '../browser-actions.js';
import { $, requireElement } from '../../lib/dom.js';
import { courseState } from '../../lib/course-state.js';
import { evidenceReviewPacket } from '../../lib/evidence.js';
import { lessonLink } from '../../lib/urls.js';
import {
  downloadText as download,
  setStatus as status,
  populateSelect as populate,
} from '../browser-actions.js';

/** @param {import('../../types/learner-records').NotebookEvidenceParams} options */
export function setupEvidence({ routes, projects, allLessons, context, getState, save, backup }) {
  /** @type {string | null} */
  let editingId = null;
  /** @type {{ entry: import('../../types/learner-records').EvidenceEntry; index: number } | null} */
  let removed = null;
  /** @param {string} selector @returns {HTMLSelectElement} */
  const select = (selector) => {
    const element = requireElement(selector);
    if (element.tagName !== 'SELECT') throw new Error(`Expected a select: ${selector}`);
    return /** @type {HTMLSelectElement} */ (element);
  };
  /** @param {string} selector @returns {HTMLInputElement} */
  const input = (selector) => {
    const element = requireElement(selector);
    if (element.tagName !== 'INPUT') throw new Error(`Expected an input: ${selector}`);
    return /** @type {HTMLInputElement} */ (element);
  };
  /** @param {string} selector @returns {HTMLTextAreaElement} */
  const textarea = (selector) => {
    const element = requireElement(selector);
    if (element.tagName !== 'TEXTAREA') throw new Error(`Expected a textarea: ${selector}`);
    return /** @type {HTMLTextAreaElement} */ (element);
  };
  /** @param {string} selector @returns {HTMLButtonElement} */
  const button = (selector) => {
    const element = requireElement(selector);
    if (element.tagName !== 'BUTTON') throw new Error(`Expected a button: ${selector}`);
    return /** @type {HTMLButtonElement} */ (element);
  };
  /** @param {string} selector @returns {HTMLFormElement} */
  const form = (selector) => {
    const element = requireElement(selector);
    if (element.tagName !== 'FORM') throw new Error(`Expected a form: ${selector}`);
    return /** @type {HTMLFormElement} */ (element);
  };
  const stageElement = $('#evidence-stage');
  const stageSelect =
    stageElement?.tagName === 'SELECT' ? /** @type {HTMLSelectElement} */ (stageElement) : null;
  const contextFields = [
    '#evidence-route',
    '#evidence-lesson',
    '#evidence-project',
    ...(stageSelect ? ['#evidence-stage'] : []),
  ];
  function updateWorkContext() {
    const target = $('#work-context');
    if (target)
      target.textContent = contextFields
        .map((selector) => {
          const select = /** @type {HTMLSelectElement} */ ($(selector));
          return select.value
            ? [...select.options].find((option) => option.value === select.value)?.textContent
            : '';
        })
        .filter(Boolean)
        .join(' / ');
  }
  for (const selector of contextFields) {
    select(selector).addEventListener('change', updateWorkContext);
  }

  function updateContexts() {
    const route = routes.find((r) => r.id === select('#evidence-route').value);
    if (!route) return;
    populate(
      select('#evidence-lesson'),
      route.lessons.filter((l) => l.status === 'authored'),
      'No lesson association',
    );
    populate(select('#evidence-project'), projects, 'No project association');
    populate(stageSelect, [], 'Whole project / no stage');
    updateWorkContext();
  }
  function updateStages() {
    const project = projects.find((p) => p.id === select('#evidence-project').value);
    populate(
      stageSelect,
      (project?.stages || []).map((s) => ({ id: s.id, title: s.title })),
      'Whole project / no stage',
    );
    updateWorkContext();
  }
  setSelectValue(select('#evidence-route'), getState().route);
  updateContexts();
  select('#evidence-route').onchange = updateContexts;
  select('#evidence-project').onchange = updateStages;
  select('#evidence-lesson').onchange = () => {
    if (!input('#evidence-title').value)
      input('#evidence-title').value =
        allLessons.find((l) => l.id === select('#evidence-lesson').value)?.title || '';
  };
  function clearForm() {
    editingId = null;
    form('#evidence-form').reset();
    setSelectValue(select('#evidence-route'), getState().route);
    updateContexts();
    button('#evidence-submit').textContent = 'Save artifact';
    button('#cancel-edit').hidden = true;
  }
  button('#cancel-edit').onclick = () => {
    clearForm();
    status('#evidence-status', 'Edit cancelled. The saved artifact was kept.');
  };
  /** @param {import('../../types/learner-records').EvidenceEntry} entry */
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
  /** @returns {import('../../types/learner-records').EvidenceEntry[]} */
  function filteredEvidence() {
    const route = select('#evidence-filter-route').value,
      type = select('#evidence-filter-type').value;
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
  /** @param {import('../../types/learner-records').EvidenceEntry[]} entries */
  function exportReview(entries) {
    try {
      const packet = evidenceReviewPacket(entries, {
        routes,
        lessons: allLessons,
        projects,
        // course-state validates the saved value against version 1 before it is exposed.
        lessonProgress: /** @type {import('../../types/learner-records').LessonProgressState} */ (
          courseState.lessonProgress
        ),
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
  const reviewExport = $('#review-export');
  if (reviewExport) reviewExport.onclick = () => exportReview(filteredEvidence());
  function renderEvidence() {
    const entries = filteredEvidence();
    const reviewCount = $('#review-count');
    if (reviewCount)
      reviewCount.textContent = `${entries.length} artifact${entries.length === 1 ? '' : 's'} will be included using the filters above.`;
    requireElement('#evidence-count').textContent =
      `${getState().evidence.length} artifact${getState().evidence.length === 1 ? '' : 's'} saved in local portfolio`;
    requireElement('#evidence-results').textContent =
      `${entries.length} shown. Saving an artifact does not tick lesson exercises or project stages.`;
    const list = requireElement('#evidence-list');
    list.replaceChildren();
    if (!entries.length) {
      const p = document.createElement('p');
      p.className = 'empty-state';
      p.textContent = getState().evidence.length
        ? 'No saved work in this group.'
        : 'No work saved yet. You can keep learning without adding notes here.';
      list.append(p);
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
        link.href = lessonLink(entry.lessonId, entry.route);
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
        const saveWork = $('#save-work');
        if (saveWork?.tagName === 'DETAILS')
          /** @type {HTMLDetailsElement} */ (saveWork).open = true;
        editingId = entry.id;
        setSelectValue(select('#evidence-route'), entry.route);
        updateContexts();
        setSelectValue(select('#evidence-lesson'), entry.lessonId || '');
        setSelectValue(select('#evidence-project'), entry.projectId || '');
        updateStages();
        setSelectValue(stageSelect, entry.stageId || '');
        updateWorkContext();
        input('#evidence-title').value = entry.title;
        input('#evidence-url').value = entry.url;
        input('#evidence-note').value = entry.note;
        button('#evidence-submit').textContent = 'Save changes';
        button('#cancel-edit').hidden = false;
        input('#evidence-title').focus();
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
        button('#undo-remove').hidden = false;
        button('#undo-remove').focus();
        status(
          '#evidence-status',
          persisted
            ? 'Artifact removed. Undo is available until your next removal.'
            : 'Artifact removed in this page only; device storage could not save. Undo or export before closing.',
        );
      };
      actions.append(edit, remove);
      article.append(actions);
      list.append(article);
    }
  }
  select('#evidence-filter-route').onchange = renderEvidence;
  select('#evidence-filter-type').onchange = renderEvidence;
  button('#undo-remove').onclick = () => {
    if (!removed) return;
    getState().evidence.splice(
      Math.min(removed.index, getState().evidence.length),
      0,
      removed.entry,
    );
    removed = null;
    const persisted = save();
    renderEvidence();
    button('#undo-remove').hidden = true;
    status(
      '#evidence-status',
      persisted ? 'Artifact restored.' : 'Artifact restored in memory only. Export before closing.',
    );
  };
  form('#evidence-form').onsubmit = (e) => {
    e.preventDefault();
    const title = input('#evidence-title').value.trim(),
      note = textarea('#evidence-note').value.trim(),
      url = input('#evidence-url').value.trim();
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
      route: select('#evidence-route').value,
      title,
      url,
      note,
      date: previous?.date || now.slice(0, 10),
      updatedAt: now,
      lessonId: select('#evidence-lesson').value,
      projectId: select('#evidence-project').value,
      stageId: stageSelect?.value || '',
    };
    if (previous) getState().evidence[getState().evidence.indexOf(previous)] = entry;
    else getState().evidence.push(entry);
    const persisted = save();
    renderEvidence();
    if (persisted) {
      clearForm();
      status('#evidence-status', 'Artifact saved on this device in your local portfolio.');
    } else {
      editingId = entry.id;
      button('#evidence-submit').textContent = 'Retry saving';
      button('#cancel-edit').hidden = false;
      status(
        '#evidence-status',
        'Saving to device storage failed. Your inputs are retained and the artifact is in this page only. Retry or export a backup before closing.',
      );
    }
  };
  const params = new URLSearchParams(location.search);
  const contextRoute = params.get('path') || params.get('route');
  if (contextRoute && context.routeIds.includes(contextRoute))
    setSelectValue(select('#evidence-route'), contextRoute);
  updateContexts();
  const lessonId = params.get('lesson');
  if (lessonId && context.lessonIds.includes(lessonId)) {
    setSelectValue(select('#evidence-lesson'), lessonId);
    input('#evidence-title').value = allLessons.find((l) => l.id === lessonId)?.title || '';
  }
  const projectId = params.get('project');
  if (projectId && projects.some((p) => p.id === projectId)) {
    setSelectValue(select('#evidence-project'), projectId);
    updateStages();
    setSelectValue(stageSelect, params.get('stage') || '');
    if (!input('#evidence-title').value)
      input('#evidence-title').value = projects.find((p) => p.id === projectId)?.title || '';
  }
  updateWorkContext();
  renderEvidence();
  return { clearForm, renderEvidence };
}
