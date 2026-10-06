// @ts-check

import { refreshChoices } from './choices.js';

/** Bind or replace copy behavior, including for dynamically rendered code panels.
 * @param {{ root?: Document | Element, selector: string, source: (button: HTMLButtonElement) => Element | null, feedback: (button: HTMLButtonElement) => HTMLElement | null, success: string | ((button: HTMLButtonElement) => string), failure: string }} options
 */
export function bindClipboard({ root = document, selector, source, feedback, success, failure }) {
  for (const button of root.querySelectorAll('button')) {
    if (!button.matches(selector)) continue;
    button.onclick = async () => {
      const content = source(button);
      const status = feedback(button);
      if (!content || !status) return;
      try {
        await navigator.clipboard.writeText(content.textContent);
        status.dataset.state = 'success';
        status.textContent = typeof success === 'function' ? success(button) : success;
      } catch {
        status.dataset.state = 'error';
        status.textContent = failure;
      }
    };
  }
}

/** @param {string} selector
 * @param {string} text
 */
export function setStatus(selector, text) {
  const element = document.querySelector(selector);
  if (element) element.textContent = text;
}

/** @param {string} name
 * @param {string} text
 * @param {string} [type]
 */
export function downloadText(name, text, type = 'text/plain') {
  const url = URL.createObjectURL(new Blob([text], { type }));
  try {
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = name;
    anchor.click();
  } finally {
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
}

/** @param {HTMLSelectElement | null} select
 * @param {{ id: string, title: string }[]} entries
 * @param {string | null} [blank]
 */
export function populateSelect(select, entries, blank) {
  if (!select) return;
  const options = blank ? [{ id: '', title: blank }, ...entries] : entries;
  select.replaceChildren(
    ...options.map((entry) => {
      const option = document.createElement('option');
      option.value = entry.id;
      option.textContent = entry.title;
      return option;
    }),
  );
  refreshChoices(select);
}

/** Programmatic selection is explicit; no native property interception is needed.
 * @param {HTMLSelectElement | null} select
 * @param {string} value
 */
export function setSelectValue(select, value) {
  if (!select) return;
  select.value = value;
  refreshChoices(select);
}
