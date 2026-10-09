import { readStoredJson } from '../lib/storage.js';
import { releaseKey, revisionKey } from '../lib/releases.js';
const changedThisVisit = new Set();

export function initReleaseNotice() {
  const notice = /** @type {HTMLElement | null} */ (document.querySelector('#release-notice'));
  if (!notice) return;
  const version = notice.dataset.version;
  const previous = readStoredJson(releaseKey);
  const acknowledge = () => {
    try {
      localStorage.setItem(releaseKey, JSON.stringify(version));
    } catch {}
  };
  // First-time visitors see the version in the footer; a banner is for returning readers.
  const reading = /** @type {{ lastLessonId?: string } | null} */ (
    readStoredJson('jaxpathways-reading-v1')
  );
  const alreadyLearning = reading?.lastLessonId;
  if (typeof previous !== 'string' && !alreadyLearning) {
    acknowledge();
    return;
  }
  if (previous === version) return;
  notice.hidden = false;
  const dismiss = /** @type {HTMLElement | null} */ (document.querySelector('#dismiss-release'));
  if (dismiss)
    dismiss.onclick = () => {
      acknowledge();
      notice.hidden = true;
    };
}

/** @param {ParentNode} [root] */
export function initRevisionNotices(root = document) {
  const saved = readStoredJson(revisionKey, {});
  const revisions = /** @type {Record<string, string>} */ (
    saved && typeof saved === 'object' && !Array.isArray(saved) ? saved : {}
  );
  /** @param {HTMLElement} node */
  const record = (node) => {
    const key = node.dataset.revisionKey,
      hash = node.dataset.revisionHash;
    if (!key || !hash) return;
    if (typeof revisions[key] === 'string' && revisions[key] !== hash) {
      changedThisVisit.add(`${key}:${hash}`);
    }
    if (changedThisVisit.has(`${key}:${hash}`)) {
      const message = /** @type {HTMLElement | null} */ (
        node.querySelector('[data-revision-notice]')
      );
      if (message) message.hidden = false;
    }
    revisions[key] = hash;
    try {
      localStorage.setItem(revisionKey, JSON.stringify(revisions));
    } catch {}
  };
  root.querySelectorAll('[data-revision-key]').forEach((el) => {
    const node = /** @type {HTMLElement} */ (el);
    if (node.dataset.revisionBound) return;
    node.dataset.revisionBound = 'true';
    const disclosure = /** @type {HTMLDetailsElement | null} */ (
      node.closest('details.phase-card')
    );
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
