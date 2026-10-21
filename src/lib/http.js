// @ts-check

/** @typedef {{ fetcher?: typeof fetch, timeoutMs?: number, cache?: RequestCache, parse?: (value: unknown) => unknown }} JsonRequestOptions */

/** Fetch JSON with a bounded wait, including while reading its response body.
 * @param {string} path
 * @template [T=unknown]
 * @param {Omit<JsonRequestOptions, 'parse'> & {parse?: (value: unknown) => T}} [options]
 * @returns {Promise<T>}
 */
export async function fetchJson(
  path,
  {
    fetcher = globalThis.fetch,
    timeoutMs = 10000,
    cache = 'default',
    parse = /** @type {(value: unknown) => T} */ ((value) => value),
  } = {},
) {
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error('Invalid request timeout');
  const controller = new AbortController();
  /** @type {ReturnType<typeof setTimeout> | undefined} */
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => {
      controller.abort();
      reject(new Error(`Request timed out: ${path}`));
    }, timeoutMs);
  });
  try {
    const request = (async () => {
      const response = await fetcher(path, { cache, signal: controller.signal });
      if (!response.ok) throw new Error(`Request failed: ${path} (${response.status})`);
      return parse(await response.json());
    })();
    return await Promise.race([request, timeout]);
  } finally {
    clearTimeout(timer);
  }
}

/** Cache successful page resources; failed requests may be retried.
 * Fetchers are separate cache scopes, allowing injected clients in tests.
 * @param {string} path
 * @template [T=unknown]
 * @param {Omit<JsonRequestOptions, 'parse'> & {parse?: (value: unknown) => T}} [options]
 * @returns {(fetcher?: typeof fetch) => Promise<T>}
 */
export function createJsonResource(path, options = {}) {
  /** @type {WeakMap<typeof fetch, Promise<T>>} */
  const requests = new WeakMap();
  return async (fetcher = options.fetcher || globalThis.fetch) => {
    const cached = requests.get(fetcher);
    if (cached) return cached;
    const pending = fetchJson(path, { ...options, fetcher }).catch((error) => {
      requests.delete(fetcher);
      throw error;
    });
    requests.set(fetcher, pending);
    return pending;
  };
}
