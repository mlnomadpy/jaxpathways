import { bootCourse } from '../boot.js';
import { initCatalog, restoreCourseView } from '../catalog.js';
bootCourse(() => {
  initCatalog();
  window.addEventListener('popstate', restoreCourseView);
}, '#phases');
