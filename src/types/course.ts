/** Public content contracts shared by build-time Astro components. */
export interface Command {
  label: string;
  code: string;
  expected: string;
  shell?: string;
}
export interface LessonContent {
  runtime?: { kind: string };
  execution?: { kind: string };
  diagram?: {
    title: string;
    prediction: string;
    reading: string;
    alt: string;
    scope: string;
    height: number;
    panel: Record<string, unknown>;
  };
  visual?: {
    title: string;
    kind: 'executed' | 'conceptual';
    prediction: string;
    reading: string;
    connection: string;
    alt: string;
    code?: string;
    nodes?: string[];
    layout?: 'merge' | 'sequence';
  };
  setupBundle?: { title: string; url: string };
  figure?: string;
  objectives?: string[];
  problem?: string;
  idea: string;
  sections?: {
    id?: string;
    title: string;
    body: string;
    check?: { prompt: string; answer: string };
    commands?: Command[];
    math?: string;
    formula?: string;
  }[];
  buildSteps?: { title: string; instruction: string; code: string; explanation: string }[];
  buildCode?: string;
  buildOutput?: string;
  code: string;
  output: string;
  experiments?: {
    title: string;
    prediction: string;
    code: string;
    output: string;
    explanation: string;
  }[];
  exercise: string;
  solution: string;
  practice?: {
    title: string;
    difficulty: string;
    prompt: string;
    hint: string;
    solution: string;
    explanation: string;
  }[];
  question: string;
  options: string[];
  answer: number;
  explanation: string;
  diagnosis?: string;
  takeaways?: string[];
  references?: { title: string; url: string }[];
}
export interface Lesson {
  prerequisites: string[];
  contentHash?: string;
  hardware?: string;
  source?: string;
  path?: string;
  check?: string;
  runtime?: { kind: string; deviceCount?: number };
  artifacts?: {
    source: string;
    script: string;
    notebook: string;
    markdown: string;
    scriptSource: string;
    notebookSource: string;
    quizSource: string;
    evidenceSource: string;
    evidence: string;
  };
  visualArtifact?: {
    mechanismImage?: string;
    mechanismPng?: string;
    image: string;
    png: string;
    data: string;
    generatedAt: string;
    contentHash: string;
    kind: string;
    environment: { jax: string; deviceCount: number };
  };
  execution?: {
    contentHash: string;
    executedAt: string;
    environment: { python: string; jax: string; deviceCount: number };
    stdout: string;
    url: string;
  };
  id: string;
  title: string;
  status: 'authored' | 'planned';
  objective: string;
  evidence: string;
  minutes?: number;
  content?: LessonContent;
}
export interface Phase {
  studyGuide: {
    question: string;
    why: string;
    readiness: { prompt: string; answer: string; lessonId: string };
    milestones: { title: string; task: string; lessonIds: string[] }[];
    transfer: { prompt: string; approach: string };
    pitfall: { symptom: string; check: string };
    completion: string;
    furtherWork: string;
  };
  coreLessonIds?: string[];
  description: string;
  learningAdvice: string;
  part: string;
  prerequisitePhaseIds: string[];
  hardware: string;
  project: string;
  checkpoint: string;
  source?: string;
  id: string;
  number: string;
  title: string;
  lessons: Lesson[];
  projectId?: string;
  projectStageIds?: string[];
  projectNote?: string;
  additionalProjectIds?: string[];
}
export interface Pathway {
  practicalGuide?: { title: string; url: string; scope: string };
  additionalGuides?: { title: string; url: string; scope: string }[];
  assessmentExtensions?: { id: string; title: string; url: string; status: string }[];
  engineeringLessonIds?: string[];
  prerequisites: string;
  outcome: string;
  libraries?: string;
  source?: string;
  audience: string;
  firstArtifact: string;
  focusPhaseIds: string[];
  modalityTrackIds: string[];
  studyAdvice: string;
  description: string;
  capstone: {
    title: string;
    status?: string;
    assessment: string;
    evidence: string[];
    projectId?: string;
    assessmentDraft?: { id: string; status: string; source: string; url: string };
  };
  id: string;
  title: string;
  phaseIds: string[];
}
export interface Role {
  id: string;
  title: string;
  headline: string;
  description: string;
  defaultPathwayId: string;
  pathwayIds: string[];
  responsibilities: string[];
  skills: string[];
  portfolio: string;
  readiness: string;
  background: string;
  workExample: string;
  modalityTrackIds: string[];
  reviewQuestions: string[];
  scopeNote: string;
  milestones: {
    title: string;
    phaseIds: string[];
    deliverable: string;
    criteria: string[];
  }[];
  startingPointChecks: { phaseId: string; question: string }[];
}
export interface Domain {
  id: string;
  title: string;
  headline: string;
  description: string;
  skills: string[];
  pathwayIds: string[];
  defaultPathwayId: string;
  roleIds: string[];
  artifact: string;
}
export interface ModalityTrack {
  id: string;
  title: string;
  summary: string;
  task: string;
  prerequisites: string;
  firstLessonId: string;
  pathwayIds: string[];
  projectIds: string[];
  harnessProjectId?: string;
  status: string;
  available: string;
  plot: string;
  stages: {
    id: string;
    title: string;
    explanation: string;
    artifact: string;
    check: string;
    lessonIds: string[];
    application: string;
    projectStageIds?: string[];
  }[];
}
export interface Course {
  title: string;
  phases: Phase[];
  pathways: Pathway[];
  projectIds: string[];
  roles: Role[];
  domains: Domain[];
  modalityTracks?: ModalityTrack[];
}

declare global {
  interface ImportMetaEnv {
    BASE_URL: string;
  }
  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }
}
