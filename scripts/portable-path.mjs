/** Repository identifiers and file URLs use forward slashes on every host. */
export const portablePath = (relative) => relative.replaceAll('\\', '/');
