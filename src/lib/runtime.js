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
      eraTitle: 'Start Here · Setup & Cloud TPU Track',
      eraSubtitle:
        'Set up your local CPU environment first, then complete the 3-step Cloud TPU mini-course before starting accelerator phases.',
      hardwareBadge: 'Laptop CPU + TPU bridge',
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
      eraTitle: 'Start Here · Setup & Cloud TPU Track',
      eraSubtitle:
        'Set up your local CPU environment first, then complete the 3-step Cloud TPU mini-course before starting accelerator phases.',
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
  const usesTpu = (phase.prerequisitePhaseIds || []).includes('tpu');
  if (usesTpu) {
    return {
      kind: 'tpu-specialization',
      filterGroup: 'tpu-scale',
      badge: 'TPU Scale & Production · CPU + Cloud TPU',
      eraId: 'era-scale',
      eraTitle: 'Part III · Accelerator Scale, Kernels, Deployment & Operations',
      eraSubtitle:
        'Partition across device meshes, author Pallas TPU kernels, ship inference services, and operate production workloads.',
      hardwareBadge: 'Logical CPU + Cloud TPU target',
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

export { practiceKind, runtimeLabel, courseTypeCue };
