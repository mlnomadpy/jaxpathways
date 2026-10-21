import labelsRegistry from '../../public/api/v1/material-labels.json' with { type: 'json' };
import { escapeHtml } from './html.js';
import { siteUrl } from './urls.js';

const entries = new Map(labelsRegistry.map((item) => [`${item.kind}:${item.id}`, item]));
const labels = {
  'available-draft': 'Available draft',
  'in-development': 'In development',
  planned: 'Planned',
  'needs-repair': 'Needs repair',
};

/** @param {string} kind @param {string} id */
export function materialStatus(kind, id) {
  const item = entries.get(`${kind}:${id}`);
  if (!item) throw new Error(`Missing learning-material status: ${kind}:${id}`);
  return item;
}

/** @param {string} kind @param {string} id */
export function materialStatusLink(kind, id) {
  return siteUrl(`content-status.html#${kind}-${encodeURIComponent(id)}`);
}

/** @param {string} level */
export function materialStatusLabel(level) {
  return labels[/** @type {keyof typeof labels} */ (level)] || 'Review pending';
}

/** A noninteractive label is safe inside a card's disclosure control.
 * @param {string} kind @param {string} id
 */
export function materialBadge(kind, id) {
  const item = materialStatus(kind, id);
  return `<span class="material-badge" data-material-key="${escapeHtml(kind + ':' + id)}" data-maturity="${escapeHtml(item.level)}">${materialStatusLabel(item.level)}</span>`;
}

/** @param {string} kind @param {string} id */
export function materialNotice(kind, id) {
  const item = materialStatus(kind, id);
  return `<aside class="material-status" aria-label="Learning material status" data-material-key="${escapeHtml(kind + ':' + id)}"><p><a class="material-badge" data-maturity="${escapeHtml(item.level)}" href="${materialStatusLink(kind, id)}">${materialStatusLabel(item.level)}</a><span>${escapeHtml(item.validation)}</span></p><p>${escapeHtml(item.review)}. <a href="${materialStatusLink(kind, id)}">Scope and ways to contribute</a>.</p></aside>`;
}
