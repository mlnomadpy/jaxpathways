// @ts-check

/**
 * @template {{ minutes?: number }} T
 * @param {T[]} lessons
 * @param {number} weeks
 * @param {number} budget
 */
function planLessonSchedule(lessons, weeks, budget) {
  if (
    !Number.isInteger(weeks) ||
    weeks < 1 ||
    weeks > 52 ||
    !Number.isFinite(budget) ||
    budget <= 0
  )
    throw Error('Invalid schedule budget');
  /** @type {T[][]} */
  const assignments = Array.from({ length: weeks }, () => []);
  /** @type {number[]} */
  const used = Array(weeks).fill(0);
  /** @type {T[]} */
  const overflow = [];
  let week = 0,
    blocked = false,
    total = 0;
  for (const lesson of lessons) {
    const minutes = Number(lesson.minutes) > 0 ? Number(lesson.minutes) : 60;
    total += minutes;
    if (blocked) {
      overflow.push(lesson);
      continue;
    }
    while (week < weeks - 1 && used[week] > 0 && used[week] + minutes > budget) week++;
    if (used[week] + minutes > budget) {
      overflow.push(lesson);
      blocked = true;
      continue;
    }
    assignments[week].push(lesson);
    used[week] += minutes;
  }
  return { assignments, used, overflow, total, capacity: weeks * budget };
}

/** @param {{ used: number[], capacity: number, assignments: unknown[][], overflow: unknown[], total: number }} schedule
 * @param {number} [unwritten]
 */
function scheduleMetrics(schedule, unwritten = 0) {
  const assignedMinutes = schedule.used.reduce((sum, minutes) => sum + minutes, 0);
  /** @param {number} minutes */
  const hours = (minutes) => Math.round((minutes / 60) * 10) / 10;
  return {
    assignedHours: hours(assignedMinutes),
    capacityHours: hours(schedule.capacity),
    assignedLessons: schedule.assignments.reduce((sum, lessons) => sum + lessons.length, 0),
    remainingLessons: schedule.overflow.length,
    remainingHours: hours(schedule.total - assignedMinutes),
    unwritten,
  };
}

export { planLessonSchedule, scheduleMetrics };
