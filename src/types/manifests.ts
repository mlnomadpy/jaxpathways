export interface ProjectStage {
  id: string;
  title: string;
  command: string;
  evidence: string;
  expected: string;
  diagnosis: string;
}

export interface ProjectManifest {
  schemaVersion: number;
  id: string;
  title: string;
  pathwayId: string;
  hardware: string;
  source: string;
  setupNote: string;
  workspaceFile: string;
  requiredLessonIds: string[];
  stages: ProjectStage[];
  rubric: string[];
  referenceCommand?: string;
  assessmentId?: string;
  prepare?: { command: string; label: string }[];
}

export interface ValidationReceipt {
  python: string;
  jax: string;
  numpy: string;
  lessons: {
    lessonId: string;
    contentHash: string;
    runtime?: { deviceCount: number };
  }[];
}
