/** @template {Element} [T=HTMLElement] @param {string} selector @returns {T | null} */
const $ = (selector) => /** @type {T | null} */ (document.querySelector(selector));

/** @template {Element} [T=HTMLElement] @param {string} selector @param {ParentNode} [root=document] @returns {T} */
const requireElement = (selector, root = document) => {
  const element = /** @type {T | null} */ (root.querySelector(selector));
  if (!element) throw new Error(`Required page element is missing: ${selector}`);
  return element;
};

export { $, requireElement };
