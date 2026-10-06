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
  audience: string;
  firstArtifact: string;
  focusPhaseIds: string[];
  modalityTrackIds: string[];
  studyAdvice: string;
  description: string;
  capstone: { title: string; projectId?: string };
  id: string;
  title: string;
  phaseIds: string[];
}
export interface Course {
  title: string;
  phases: Phase[];
  pathways: Pathway[];
  projectIds: string[];
  roles: { id: string; title: string; headline: string; responsibilities: string[] }[];
}
