// @ts-check

/** @type {Record<string, string>} */
const entities = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

/** @param {unknown} value */
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (character) => entities[character]);

export { escapeHtml };
