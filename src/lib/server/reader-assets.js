import { readFileSync } from 'node:fs';
import path from 'node:path';

const imageCache = new Map();
/** Server-only embedding keeps the shared browser renderer free of filesystem APIs.
 * @param {string} asset
 * @returns {string}
 */
export function embeddedImage(asset) {
  if (!imageCache.has(asset)) {
    const bytes = readFileSync(path.join(process.cwd(), 'public', asset));
    const mime = asset.endsWith('.svg') ? 'image/svg+xml' : 'image/png';
    imageCache.set(asset, `data:${mime};base64,${bytes.toString('base64')}`);
  }
  return imageCache.get(asset);
}
