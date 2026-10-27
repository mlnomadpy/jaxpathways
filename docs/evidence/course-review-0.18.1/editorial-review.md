# Teaching review notes

The pass assigned 92 lessons to subject-specific Luna writers, using the complete lesson and its retained execution and figure data. Three separate Luna reviews examined 51 distinct lessons for semantic consistency with the implementations and artifacts. These are AI-assisted editorial reviews, not human expert acceptance or a learner pilot.

The revision preserves the 12 lesson rewrites from 0.18.0 and changes no executable lesson code, formulas, commands, exercises or plot experiments. All 22 chapter introductions now explain the purpose and progression of their material.

Parent review and follow-up copyediting corrected:

- A trajectory-array explanation that put the initial condition in the forty-first column. Column zero stores the initial condition; forty updates give forty-one saved times.
- A reduction introduction that changed its input values before the learner saw the program. It now uses the program's exact array and makes adding one an explicit transfer question.
- Repetition in the first derivative lesson. The next section now uses negative and zero inputs to distinguish slope sign from output sign and stationarity from a constant function.
- Nested KaTeX delimiters, malformed escaped Greek symbols and a missing list delimiter, so mathematical examples render rather than showing control characters or broken notation.
- Mathematical shapes and GAE expressions written as code literals instead of rendered math.
- An operations explanation whose reserve, cost and duty-fraction paragraphs had drifted under the wrong headings. The revised sequence connects the four-worker reserve calculation to its hypothetical hourly bill and keeps duty fraction separate from hardware utilization.
- Timing descriptions tied to obsolete sample values. The revised figure explanations teach their boundaries, units, panels and supported comparisons; numerical hand examples are explicitly illustrative. They do not imply an optimal batch size from one noisy CPU capture.
- The distinction between the six-step checkpoint plot and the separate eight-step launcher example. Each keeps its own save boundary and results.

No maturity level or failed model-quality result was promoted by this review. The review guide asks the author to identify missing intermediate reasoning, unclear figure readings and changed-input predictions that still need work.
