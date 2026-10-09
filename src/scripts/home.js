import { bindClipboard } from './browser-actions.js';

export function initHomeCopy() {
  bindClipboard({
    selector: '[data-home-copy]',
    source: (button) => document.getElementById(button.dataset.homeCopy || ''),
    feedback: (button) => document.getElementById(`${button.dataset.homeCopy}-status`),
    success: (button) =>
      button.dataset.homeCopy === 'tutor-install'
        ? 'Copied. Run it in the course folder’s terminal.'
        : 'Copied. Paste it into your agent chat.',
    failure: 'Select the text above and copy it manually.',
  });
}
