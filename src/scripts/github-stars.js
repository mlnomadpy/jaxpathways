import repository from '../data/repository.json' with { type: 'json' };

/** One public request updates every badge; offline/private/rate-limited responses retain the dated snapshot. */
export async function refreshGitHubStars(root = document, fetcher = fetch) {
  const links = root.querySelectorAll('[data-github-link]');
  if (!links.length) return;
  try {
    const response = await fetcher(repository.apiUrl, {
      credentials: 'omit',
      headers: { Accept: 'application/vnd.github+json' },
      signal: AbortSignal.timeout(5000),
    });
    if (!response.ok) return;
    const { stargazers_count: count } = await response.json();
    if (!Number.isSafeInteger(count) || count < 0) return;
    for (const link of /** @type {NodeListOf<HTMLElement>} */ (links)) {
      const stars = link.querySelector('[data-github-stars]');
      if (stars) stars.textContent = count.toLocaleString('en-US');
      link.title = `GitHub repository · ${count.toLocaleString('en-US')} stars, checked just now`;
    }
  } catch {
    // The ordinary repository link and verified count work without the API or JavaScript.
  }
}
