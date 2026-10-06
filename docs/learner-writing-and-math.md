# Learner writing and mathematical notation

Teach with patient, concrete explanations. Use these general teaching principles rather than reproducing any instructor’s distinctive wording or persona.

1. Open with a question the learner can picture. Explain why it matters before naming the API.
2. Introduce one idea at a time: intuition, a small example, notation, then code.
3. Define symbols where they first appear. Read sums and matrix operations in words.
4. Work through at least one number by hand. Ask for a prediction before running code.
5. Explain mistakes as useful evidence. Give a specific next check instead of “obviously,” “simply,” or “just.”
6. Use “we” for guided reasoning and “you” for the learner’s next action. Encourage attempts without exaggerated praise or promising mastery.
7. Keep limitations and numerical assumptions, but explain their practical meaning. Friendliness must not weaken correctness.
8. End with a changed condition and something the learner can explain independently.

## Author math in the canonical lesson JSON

Use `\\( ... \\)` for inline math (JSON escapes each backslash). Put a standalone LaTeX equation in a section’s `math` field, without delimiters. Use `\\[ ... \\]` or `$$ ... $$` for display math in prose if needed; avoid blank lines inside one expression. Single dollar signs remain ordinary text so prices and shell variables are not accidentally parsed.

```json
{
  "title": "Average the squared errors",
  "body": "For example \\(i\\), the residual \\(r_i\\) is the prediction minus the target. Square each residual, add the results, then divide by the number of examples \\(N\\).",
  "math": "L=\\frac{1}{N}\\sum_{i=1}^{N}r_i^2"
}
```

Use `aligned` for multiple equation lines. Keep `formula` for plain-text shape sketches, state transitions and pseudocode. Never put TeX in Python code blocks or rewrite runnable examples into mathematical notation.

The browser reader and Astro printable book share a KaTeX renderer with HTML plus accessible MathML. CSS and fonts are bundled locally. EPUB uses KaTeX-generated native MathML and marks the affected chapters in its manifest. Markdown/notebooks retain source TeX, with inline expressions converted to single-dollar delimiters for those readers; section equations use double-dollar display delimiters. EPUB math appearance depends on the reader’s MathML support.

Rendering escapes prose HTML and leaves inline code literal. KaTeX uses `trust: false`, bounded macro expansion and fresh macro state for every expression. Malformed TeX fails validation rather than silently shipping a broken equation. These choices follow the [KaTeX API](https://katex.org/docs/api.html), [options](https://katex.org/docs/options.html) and [security guidance](https://katex.org/docs/security.html).

## Scope of this revision

All 42 authored lesson openings received a learner-oriented edit. Eighteen display equations across core mathematical topics replace mathematical ASCII blocks or add a worked explanation; code/data-flow sketches remain text. The most substantial explanation rewrites cover standardization, derivatives, the chain rule, regression loss, and precision choices, with additional notation explanations for gradients, attention and quantization. This is an editorial pass, not a claim that every paragraph or every exercise has undergone independent pedagogical review.

## Inline notation follow-up

The inline pass extends across explanations, predictions, exercises, hints, expected outputs and checkpoint choices. Matrix/vector dimensions, numerical arrays, scalar values and expressions use explicit math delimiters. Executable Python expressions use inline code. The regression introduction now explains matrix membership, the weighted prediction, bias broadcasting and mean squared loss in KaTeX, with `X @ w` labeled as its Python spelling. A regression test checks this paragraph so inline notation cannot silently return to plain text.
