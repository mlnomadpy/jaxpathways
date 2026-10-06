import { loadCourse } from '../lib/course-state.js';
import { initExperience } from './experience.js';

/** Page-local startup: a failed catalog request leaves a useful recovery message. */
export async function bootCourse(initialize, errorTarget) {
  initExperience();
  try {
    await loadCourse();
    await initialize();
  } catch (error) {
    console.error('Unable to initialize the course page', error);
    const target = document.querySelector(errorTarget);
    if (target) {
      const message = document.createElement('p');
      message.textContent = 'The course could not load. Refresh this page to try again.';
      message.setAttribute('role', 'alert');
      target.replaceChildren(message);
    }
  }
}
