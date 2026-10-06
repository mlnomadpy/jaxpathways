// @ts-check

/** Read optional browser records without treating malformed or unavailable storage as data.
 * @param {string} key
 * @param {unknown} [fallback]
 * @param {Pick<Storage, 'getItem'>} [storage]
 * @returns {unknown}
 */
export function readStoredJson(key, fallback = null, storage) {
  try {
    const raw = (storage || globalThis.localStorage).getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}
