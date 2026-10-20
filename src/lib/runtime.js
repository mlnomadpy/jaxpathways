// @ts-check

/** @param {import('../types/course').Lesson} lesson */
function practiceKind(lesson) {
  return lesson.content?.runtime?.kind === 'virtual-cpu' ||
    lesson.content?.execution?.kind === 'virtual-cpu' ||
    /virtual.*cpu|cpu.*virtual/i.test(lesson.title)
    ? 'virtual-cpu'
    : 'cpu';
}

/** @param {import('../types/course').Lesson} lesson */
function runtimeLabel(lesson) {
  return practiceKind(lesson) === 'virtual-cpu' ? 'Logical CPU devices' : 'CPU practice';
}

/** Classify the reference separately from optional target-specific programs.
 * CPU execution receipts never qualify a target runner.
 * @param {import('../types/course').Lesson} lesson
 */
function executionScope(lesson) {
  const logical = practiceKind(lesson) === 'virtual-cpu';
  const source = lesson.content?.code || '';
  const cpuOnly =
    logical || /JAX_PLATFORMS[\s\S]{0,30}cpu|jax\.devices\(["']cpu["']\)/.test(source);
  const targetRunners = {
    'tpu-04': 'runtime receipt collector',
    'tpu-06': 'precision and profiling runner',
    'kernels-02': 'Pallas TPU kernel',
  };
  const targetRunner = targetRunners[/** @type {keyof typeof targetRunners} */ (lesson.id)];
  return {
    reference: logical ? 'logical-cpu' : cpuOnly ? 'cpu-only' : 'cpu-reference',
    description: logical
      ? 'This reference configures logical CPU devices to practice placement and sharding. Logical devices do not measure TPU performance.'
      : cpuOnly
        ? 'This reference explicitly selects CPU. Uploading the same file or setting JAX_PLATFORMS does not adapt it to TPU.'
        : 'This worked reference is validated on CPU. Accelerator compatibility requires a separate source review and target execution receipt.',
    target: targetRunner
      ? `A separate ${targetRunner} is implemented in the target instructions. Bounded hardware receipts cover v5e1 launch, resume and precision checks; four single-device Pallas cases; and a four-device v5e collective fixture. Each receipt applies only to its recorded source, environment, cases and hardware. Other lessons and TPU configurations require separate qualification.`
      : 'Use the execution matrix to review device selection and data ownership before adapting a separate target program.',
  };
}

/**
 * Derive visual course-type cues, hardware badges, pacing, and curriculum era metadata for a phase.
 * @param {import('../types/course').Phase} phase
 */
function courseTypeCue(phase) {
  const totalMinutes = phase.lessons.reduce((sum, l) => sum + (l.minutes || 25), 0);
  const lessonCount = phase.lessons.length;
  const paceLabel =
    lessonCount <= 2
      ? `Bite-sized · ${lessonCount} lessons (~${totalMinutes}m)`
      : totalMinutes < 90
        ? `${lessonCount} lessons (~${totalMinutes}m)`
        : `${lessonCount} lessons (~${Math.round((totalMinutes / 60) * 10) / 10}h)`;
  if (phase.id === 'welcome') {
    return {
      kind: 'onboarding',
      filterGroup: 'onboarding-tpu',
      badge: 'Onboarding · Local CPU',
      eraId: 'era-onboarding',
      eraTitle: 'Start Here · CPU Setup & Optional TPU Track',
      eraSubtitle:
        'Start with the local CPU environment. Add the optional TPU specialization when you need to qualify accelerator execution.',
      hardwareBadge: 'Laptop CPU',
      paceLabel,
    };
  }
  if (phase.id === 'tpu' || phase.id === 'tpu-workflows' || phase.id === 'tpu-systems') {
    const stepNum = phase.id === 'tpu' ? 1 : phase.id === 'tpu-workflows' ? 2 : 3;
    return {
      kind: 'tpu-core',
      filterGroup: 'onboarding-tpu',
      badge: `Cloud TPU Mini-Course · Step ${stepNum} of 3`,
      eraId: 'era-onboarding',
      eraTitle: 'Start Here · CPU Setup & Optional TPU Track',
      eraSubtitle:
        'Start with the local CPU environment. Add the optional TPU specialization when you need to qualify accelerator execution.',
      hardwareBadge: 'Cloud TPU VM + CPU verifier',
      paceLabel,
    };
  }
  if (phase.part === 'Shared foundations') {
    return {
      kind: 'foundation',
      filterGroup: 'foundation',
      badge: 'Core Foundation · Local CPU',
      eraId: 'era-foundations',
      eraTitle: 'Part I · Shared CPU Foundations',
      eraSubtitle:
        'Master arrays, pure functions, jit/grad/vmap transforms, explicit state, and numerical optimization on CPU.',
      hardwareBadge: 'Laptop CPU',
      paceLabel,
    };
  }
  if (phase.part === 'Training systems') {
    const usesTpu = (phase.prerequisitePhaseIds || []).includes('tpu');
    return {
      kind: usesTpu ? 'tpu-system' : 'training-system',
      filterGroup: 'training',
      badge: usesTpu ? 'Training Systems · CPU + Cloud TPU' : 'Training Systems · Local CPU',
      eraId: 'era-training',
      eraTitle: 'Part II · Complete Training Systems',
      eraSubtitle:
        'Build neural networks, checkpoint recovery, Transformers, pretraining, post-training, and performance profiling.',
      hardwareBadge: usesTpu ? 'CPU lab + Cloud TPU VM' : 'Laptop CPU',
      paceLabel,
    };
  }
  if (['distributed', 'kernels', 'deployment', 'operations'].includes(phase.id)) {
    return {
      kind: 'systems-specialization',
      filterGroup: 'tpu-scale',
      badge: 'Systems Specialization · CPU Reference',
      eraId: 'era-scale',
      eraTitle: 'Part III · Accelerator Scale, Kernels, Deployment & Operations',
      eraSubtitle:
        'Partition across device meshes, author Pallas TPU kernels, ship inference services, and operate production workloads.',
      hardwareBadge: 'CPU reference; target qualification separate',
      paceLabel,
    };
  }
  return {
    kind: 'domain-specialization',
    filterGroup: 'domain',
    badge: 'Domain Specialization · Local CPU',
    eraId: 'era-domains',
    eraTitle: 'Part IV · Research & Domain Specializations',
    eraSubtitle:
      'Apply differentiable programming to scientific simulations, Bayesian inference, reinforcement learning, and JAX internals.',
    hardwareBadge: 'Laptop CPU',
    paceLabel,
  };
}

export { practiceKind, runtimeLabel, courseTypeCue, executionScope };
