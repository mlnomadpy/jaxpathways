import { readStoredJson } from '../lib/storage.js';
import { releaseKey, revisionKey } from '../lib/releases.js';
const changedThisVisit = new Set();

export function initReleaseNotice() {
  const notice = document.querySelector('#release-notice');
  if (!notice) return;
  const version = notice.dataset.version;
  const previous = readStoredJson(releaseKey);
  const acknowledge = () => {
    try {
      localStorage.setItem(releaseKey, JSON.stringify(version));
    } catch {}
  };
  // First-time visitors see the version in the footer; a banner is for returning readers.
  const alreadyLearning = readStoredJson('jaxpathways-reading-v1')?.lastLessonId;
  if (typeof previous !== 'string' && !alreadyLearning) {
    acknowledge();
    return;
  }
  if (previous === version) return;
  notice.hidden = false;
  document.querySelector('#dismiss-release').onclick = () => {
    acknowledge();
    notice.hidden = true;
  };
}

export function initRevisionNotices(root = document) {
  const saved = readStoredJson(revisionKey, {});
  const revisions = saved && typeof saved === 'object' && !Array.isArray(saved) ? saved : {};
  const record = (node) => {
    const key = node.dataset.revisionKey,
      hash = node.dataset.revisionHash;
    if (typeof revisions[key] === 'string' && revisions[key] !== hash) {
      changedThisVisit.add(`${key}:${hash}`);
    }
    if (changedThisVisit.has(`${key}:${hash}`)) {
      const message = node.querySelector('[data-revision-notice]');
      if (message) message.hidden = false;
    }
    revisions[key] = hash;
    try {
      localStorage.setItem(revisionKey, JSON.stringify(revisions));
    } catch {}
  };
  root.querySelectorAll('[data-revision-key]').forEach((node) => {
    if (node.dataset.revisionBound) return;
    node.dataset.revisionBound = 'true';
    const disclosure = node.closest('details.phase-card');
    // Merely loading the catalog must not mark a collapsed phase as reviewed.
    if (disclosure && !disclosure.open) {
      disclosure.addEventListener(
        'toggle',
        () => {
          if (disclosure.open) record(node);
        },
        { once: true },
      );
    } else record(node);
  });
}
