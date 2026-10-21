import type { Lesson, Pathway } from './course';
import type { ProjectManifest } from './manifests';

export interface LessonProgressEntry {
  checkpointPassed: boolean;
  evidenceSaved: boolean;
}

export interface LessonProgressState {
  version: 1;
  lessons: Record<string, LessonProgressEntry>;
}

export interface EvidenceEntry {
  id: string;
  route: string;
  title: string;
  note: string;
  url: string;
  date: string;
  updatedAt?: string;
  lessonId?: string;
  projectId?: string;
  stageId?: string;
}

export interface LearnerNotebook {
  version: 1;
  route: string;
  evidence: EvidenceEntry[];
}

export interface ReadingPosition {
  sectionId: string;
  updatedAt: string;
}

export interface ReadingState {
  version: 1;
  lastLessonId: string;
  path: string;
  sectionId: string;
  updatedAt: string;
  lessons: Record<string, ReadingPosition>;
}

export interface CareerPlan {
  version: 1;
  roleId: string;
  pathwayId: string;
  startPhaseId: string;
  knownPhaseIds: string[];
}

export interface ValidationContext {
  lessonIds: string[];
  routeIds: string[];
  roleIds: string[];
  phaseIds: string[];
}

export interface LearnerBackup {
  version: 2;
  notebook: LearnerNotebook;
  sampleCompleted: boolean;
  lessonProgress: LessonProgressState;
  projects: Record<string, string[]>;
  careerPlan: CareerPlan | null;
  reading: ReadingState | null;
}

export interface NotebookRoute extends Pathway {
  lessons: Lesson[];
}

export interface NotebookViewContext extends ValidationContext {}

export interface NotebookSummaryParams {
  routes: NotebookRoute[];
  projects: ProjectManifest[];
  allLessons: Lesson[];
  context: NotebookViewContext;
  state: LearnerNotebook;
  backup: () => LearnerBackup;
}

export interface NotebookEvidenceParams {
  routes: NotebookRoute[];
  projects: ProjectManifest[];
  allLessons: Lesson[];
  context: NotebookViewContext;
  getState: () => LearnerNotebook;
  save: () => boolean;
  backup: () => LearnerBackup;
}

export interface NotebookBackupParams {
  context: NotebookViewContext;
  getBackup: () => LearnerBackup;
  applyBackup: (backup: LearnerBackup) => void;
}
