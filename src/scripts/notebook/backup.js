import { $ } from '../../lib/dom.js';
import {
  normalizeLearnerBackup,
  mergeLearnerBackup,
  persistLearnerBackup,
} from '../../lib/learner-records.js';
import { downloadText as download, setStatus as status } from '../browser-actions.js';

export function setupBackup({ context, getBackup, applyBackup }) {
  $('#export').onclick = () => {
    try {
      const outgoing = normalizeLearnerBackup(getBackup(), context);
      download(
        'jaxpathways-learning-backup-v2.json',
        JSON.stringify(outgoing, null, 2),
        'application/json',
      );
      status(
        '#backup-status',
        'Backup downloaded: artifacts, lesson progress, project stages, career plan, and saved reading position.',
      );
    } catch {
      status(
        '#backup-status',
        'Some stored records have an unsupported format. Do not clear storage; keep your inputs and contact the course maintainer.',
      );
    }
  };
  $('#import-button').onclick = () => $('#import').click();
  $('#import').onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const preview = $('#restore-preview');
    if (preview) {
      preview.hidden = true;
      preview.replaceChildren();
    }
    try {
      if (file.size > 3000000) throw Error('File is too large');
      const incoming = JSON.parse(await file.text());
      // Validate before showing the merge action. No local records change yet.
      const checked = mergeLearnerBackup(getBackup(), incoming, context);
      const apply = () => {
        try {
          const merged = mergeLearnerBackup(getBackup(), incoming, context),
            persisted = persistLearnerBackup(localStorage, merged);
          applyBackup(merged);
          if (preview) {
            preview.hidden = true;
            preview.replaceChildren();
          }
          status(
            '#backup-status',
            persisted
              ? 'Backup merged and saved. Existing artifacts, checkpoints, and project stage reports were preserved.'
              : 'Backup merged into this page only. Device storage failed; export a new backup before closing. Existing storage was restored where possible.',
          );
        } catch {
          status(
            '#backup-status',
            'This backup could not be restored. Your current records are still available. Download them before trying again.',
          );
        }
      };
      if (!preview) {
        apply();
        return;
      }
      preview.hidden = false;
      const heading = document.createElement('h3');
      heading.textContent = 'Restore this backup?';
      heading.tabIndex = -1;
      const description = document.createElement('p');
      description.textContent = `${file.name}. After merging: ${checked.notebook.evidence.length} artifacts and ${Object.keys(checked.lessonProgress.lessons).length} lesson progress records. Existing checkpoints and stage reports are retained.`;
      const restore = document.createElement('button');
      restore.type = 'button';
      restore.className = 'primary';
      restore.textContent = 'Restore and merge';
      restore.onclick = apply;
      const cancel = document.createElement('button');
      cancel.type = 'button';
      cancel.textContent = 'Cancel';
      cancel.onclick = () => {
        preview.hidden = true;
        preview.replaceChildren();
        $('#import-button').focus();
      };
      preview.append(heading, description, restore, cancel);
      heading.focus();
      status('#backup-status', 'Backup checked. Choose Restore and merge to update this browser.');
    } catch {
      status(
        '#backup-status',
        'This file is not a supported JAX Pathways backup, or could not be read. Your current workspace was not changed. Choose the original JSON backup and try again.',
      );
    } finally {
      e.target.value = '';
    }
  };
}
