import { escapeHtml } from './html.js';
import { inlineMath, mathProse } from './math.js';
import { siteUrl } from './urls.js';

/** A conceptual mechanism belongs beside its worked explanation. */
export function lessonMechanism(lesson) {
  const diagram = lesson.content?.diagram;
  const artifact = lesson.visualArtifact;
  if (!diagram || !artifact?.mechanismImage || artifact.contentHash !== lesson.contentHash)
    return '';
  const id = `mechanism-${lesson.id}`;
  return `<figure class="lesson-mechanism" aria-labelledby="${escapeHtml(id)}"><h3 id="${escapeHtml(id)}">${escapeHtml(diagram.title)}</h3><p class="visual-prediction"><strong>Predict:</strong> ${inlineMath(diagram.prediction)}</p><div class="figure-scroll" role="region" aria-label="${escapeHtml(diagram.title)}" tabindex="0"><img src="${escapeHtml(siteUrl(artifact.mechanismImage))}" alt="${escapeHtml(diagram.title)}. The explanation below describes the relationships." aria-describedby="${escapeHtml(id)}-reading" loading="lazy" decoding="async" /></div><figcaption>${escapeHtml(diagram.scope)}</figcaption><div id="${escapeHtml(id)}-reading">${mathProse(diagram.reading)}</div><a href="${escapeHtml(siteUrl(artifact.mechanismImage))}">Open full-size diagram</a></figure>`;
}

/** Render only artifacts bound to the current canonical lesson content. */
export function lessonVisual(lesson) {
  const visual = lesson.content?.visual;
  const artifact = lesson.visualArtifact;
  if (!lesson.contentHash || !visual || !artifact || artifact.contentHash !== lesson.contentHash)
    return '';
  const url = (path) => escapeHtml(siteUrl(path));
  const scope =
    visual.kind === 'executed'
      ? 'Plot from a recorded CPU computation'
      : 'Conceptual workflow diagram';
  return `<section class="lesson-visual" aria-labelledby="visual-${escapeHtml(lesson.id)}"><h2 id="visual-${escapeHtml(lesson.id)}">${escapeHtml(visual.title)}</h2><p class="visual-prediction"><strong>Predict before looking:</strong> ${inlineMath(visual.prediction)}</p><figure><div class="figure-scroll" role="region" aria-label="${escapeHtml(visual.title)}" tabindex="0"><img src="${url(artifact.image)}" alt="${escapeHtml(visual.title)}. A detailed explanation follows the figure." loading="lazy" decoding="async" /></div><figcaption>${scope} · JAX ${escapeHtml(artifact.environment.jax)} · ${artifact.environment.deviceCount} CPU device${artifact.environment.deviceCount === 1 ? '' : 's'}</figcaption></figure><div class="figure-walkthrough"><h3>Read the figure</h3>${mathProse(visual.reading)}<h3>Connect it to the computation</h3>${mathProse(visual.connection)}</div><p class="figure-links"><a href="${url(artifact.image)}">Open full-size figure</a><a href="${url(artifact.data)}" download>Download data and provenance</a></p><p class="soft">${visual.kind === 'executed' ? 'The notebook contains the data experiment, plotting code, and recorded figure output. Rerun it to test your changes.' : 'The arrows describe the workflow; the reference execution below separately records the code results.'}</p></section>`;
}

export function lessonExecution(lesson) {
  const run = lesson.execution;
  if (!lesson.contentHash || !run || run.contentHash !== lesson.contentHash) return '';
  return `<section class="lesson-execution"><h2>Recorded reference execution</h2><p>This output was captured from the instructor’s complete script, including its assertions and reference solutions. It is a saved CPU run, not code executing in this page.</p><p class="execution-meta">${escapeHtml(run.executedAt)} · Python ${escapeHtml(run.environment.python)} · JAX ${escapeHtml(run.environment.jax)} · ${run.environment.deviceCount} CPU device${run.environment.deviceCount === 1 ? '' : 's'}</p><pre tabindex="0" aria-label="Captured Python standard output"><code>${escapeHtml(run.stdout)}</code></pre><p><a href="${escapeHtml(siteUrl(run.url))}" download>Download execution record</a> · The notebook includes executed cells and figure outputs. Compare your run with these values; timings depend on the machine.</p></section>`;
}
