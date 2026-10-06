function searchLesson(lesson, query) {
  if (!query) return { matched: true, excerpt: '' };
  const c = lesson.content || {};
  const chunks = [
    lesson.title,
    lesson.objective,
    c.problem,
    c.idea,
    ...(c.sections || []).flatMap((s) => [s.title, s.body, s.formula]),
    ...(c.experiments || []).flatMap((e) => [e.title, e.prediction, e.explanation, e.code]),
    c.exercise,
    c.diagnosis,
    ...(c.practice || []).flatMap((p) => [p.title, p.prompt, p.explanation]),
    ...(c.references || []).map((r) => r.title),
  ].filter(Boolean);
  const aliases = {
    prng: 'random',
    rng: 'random',
    autodiff: 'gradient',
    batching: 'batch',
    pytree: 'pytrees',
    checkpointing: 'checkpoint',
    tpu: 'tpu',
    jit: 'jit',
    vmap: 'vmap',
    hlo: 'hlo',
  };
  const terms = [query, aliases[query]].filter(Boolean);
  for (const chunk of chunks) {
    const lower = chunk.toLowerCase(),
      term = terms.find((t) => lower.includes(t));
    if (!term) continue;
    const index = lower.indexOf(term),
      start = Math.max(0, index - 45),
      end = Math.min(chunk.length, index + 150);
    return {
      matched: true,
      excerpt:
        (start ? '…' : '') +
        chunk.slice(start, end).replace(/\s+/g, ' ') +
        (end < chunk.length ? '…' : ''),
    };
  }
  return { matched: false, excerpt: '' };
}

export { searchLesson };
