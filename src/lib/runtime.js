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

export { practiceKind, runtimeLabel };
