import registry from '../../../public/api/v1/content-status.json' with { type: 'json' };
import { feedbackUrl } from '../releases.js';
export const materialRegistry = registry;

/** @param {string} kind @param {string} id */
export function contributionUrl(kind, id) {
  const item = registry.items.find((item) => item.kind === kind && item.id === id);
  if (!item) throw new Error(`Missing contribution context: ${kind}:${id}`);
  const url = new URL(feedbackUrl(`Contribution: ${item.title}`, item.url));
  url.searchParams.set(
    'context',
    `${url.searchParams.get('context')}\nMaterial status: ${item.level}\nSuggested contribution: ${item.contribution}`,
  );
  return url.href;
}
