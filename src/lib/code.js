const CourseCode = (() => {
  const escape = (s) =>
    String(s).replace(
      /[&<>"']/g,
      (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c],
    );
  const keywords = new Set(
    'def return import from as if else elif for in while with try except finally raise assert lambda class pass and or not True False None async await yield break continue del is nonlocal global'.split(
      ' ',
    ),
  );
  const builtins = new Set(
    'print range len float int str dict list tuple zip enumerate sum min max abs type ValueError'.split(
      ' ',
    ),
  );
  function highlight(source, language = 'python') {
    if (language === 'text') return escape(source);
    const shell = language === 'shell';
    const pattern =
      /"{3}[\s\S]*?"{3}|'{3}[\s\S]*?'{3}|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|#[^\n]*|\$[A-Za-z_][\w:]*(?:\.[\w]+)*|--?[A-Za-z][\w-]*|\b(?:0x[\da-fA-F]+|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\b|\b[A-Za-z_]\w*\b|[^\w]/g;
    return (String(source).match(pattern) || [])
      .map((token, i, tokens) => {
        let kind = '';
        if (token.startsWith('#')) kind = 'comment';
        else if (/^['"]/.test(token)) kind = 'string';
        else if (/^\d/.test(token)) kind = 'number';
        else if (shell && /^--?\w/.test(token)) kind = 'option';
        else if (shell && token.startsWith('$')) kind = 'variable';
        else if (!shell && keywords.has(token)) kind = 'keyword';
        else if (!shell && builtins.has(token)) kind = 'builtin';
        else if (shell && /^(python\d*|py|pip|npx|npm|source|cp|Copy|Item)$/.test(token))
          kind = 'builtin';
        else if (!shell && /^\w+$/.test(token) && tokens.slice(i + 1).find((t) => t.trim()) === '(')
          kind = 'function';
        return kind
          ? '<span class="syntax-' + kind + '">' + escape(token) + '</span>'
          : escape(token);
      })
      .join('');
  }
  function paint(root = document, defaultLanguage = 'shell') {
    root.querySelectorAll('pre').forEach((pre) => {
      const node = pre.querySelector('code') || pre;
      const language =
        node.dataset.language ||
        node.closest('[data-language]')?.dataset.language ||
        defaultLanguage;
      node.dataset.language = language;
      node.innerHTML = highlight(node.textContent, language);
    });
  }
  return { highlight, paint };
})();

export { CourseCode };
