import { bootCourse } from '../boot.js';
import { initHomeCopy } from '../home.js';
import { renderHomeContinue } from '../resume.js';
initHomeCopy();
const legacy = new URL(location.href);
if (
  legacy.hash === '#curriculum' ||
  legacy.searchParams.has('path') ||
  legacy.searchParams.has('phase')
) {
  legacy.pathname = legacy.pathname.replace(/(?:index\.html)?$/, 'course.html');
  location.replace(legacy.href);
} else {
  bootCourse(renderHomeContinue, '#home-continue');
}
