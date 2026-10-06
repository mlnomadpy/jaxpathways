import { bootCourse } from '../boot.js';
import { courseState } from '../../lib/course-state.js';
import { setupPlatform } from '../workspace.js';
bootCourse(() => setupPlatform(courseState.curriculum), '#main');
