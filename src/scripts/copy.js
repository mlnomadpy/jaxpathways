import { bindClipboard } from './browser-actions.js';

export function bindCodeCopy() {
  bindClipboard({
    selector: '.copy-snippet',
    source: (button) => button.closest('.code-panel')?.querySelector('code') || null,
    feedback: (button) => button.closest('.code-panel')?.querySelector('.copy-feedback') || null,
    success: 'Code copied.',
    failure: 'Clipboard unavailable. Select the code and copy it manually.',
  });
}
