import { bindClipboard } from './browser-actions.js';

export function initResourceCopy() {
  bindClipboard({
    selector: '[data-copy]',
    source: (button) => document.getElementById(button.dataset.copy),
    feedback: (button) =>
      button.closest('.resource-code')?.querySelector('.copy-status') ||
      document.getElementById('resource-copy-status'),
    success: 'Command copied. Run it from your course checkout when you are ready.',
    failure: 'Clipboard access is unavailable. Select the command text and copy it manually.',
  });
}
