import { initNavigation } from './navigation.js';
import { enhanceChoices } from './choices.js';
import { initReadingTools } from './reading-tools.js';

const enhancedDocuments = new WeakSet();

export function initExperience() {
  if (enhancedDocuments.has(document)) return;
  enhancedDocuments.add(document);
  initNavigation();
  revealContext();
  enhanceChoices();
  initReadingTools();
}

function revealContext() {
  const params = new URLSearchParams(location.search),
    save = /** @type {HTMLDetailsElement | null} */ (document.getElementById('save-work'));
  if (save && (params.has('lesson') || params.has('project') || location.hash === '#portfolio'))
    save.open = true;
  if (location.hash === '#backups') {
    const panel = /** @type {HTMLDetailsElement | null} */ (
      document.getElementById('backup-panel')
    );
    if (panel) panel.open = true;
  }
}
