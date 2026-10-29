import { CourseCode } from './code.js';
import { escapeHtml } from './html.js';
import { inlineMath } from './math.js';

/** @type {Array<{ pattern: RegExp; token: string; summary: string }>} */
const API_CATALOG = [
  {
    pattern: /\bjnp\.arange\s*\(/,
    token: 'jnp.arange(n, dtype=...)',
    summary:
      'Creates a 1-D JAX array of evenly spaced values `[0, 1, ..., n-1]` on the target device.',
  },
  {
    pattern: /\bjnp\.linspace\s*\(/,
    token: 'jnp.linspace(start, stop, num)',
    summary: 'Creates `num` evenly spaced float points across the closed interval `[start, stop]`.',
  },
  {
    pattern: /\bjnp\.array\s*\(/,
    token: 'jnp.array(values, dtype=...)',
    summary: 'Constructs a JAX array from Python/NumPy values.',
  },
  {
    pattern: /\bjnp\.zeros\s*\(/,
    token: 'jnp.zeros(shape, dtype=...)',
    summary: 'Creates a JAX array of the requested shape filled with zeros.',
  },
  {
    pattern: /\bjnp\.ones\s*\(/,
    token: 'jnp.ones(shape, dtype=...)',
    summary: 'Creates a JAX array of the requested shape filled with ones.',
  },
  {
    pattern: /\bjnp\.full\s*\(/,
    token: 'jnp.full(shape, fill_value, dtype=...)',
    summary: 'Creates an array of the requested shape with every element set to `fill_value`.',
  },
  {
    pattern: /\bjnp\.eye\s*\(/,
    token: 'jnp.eye(N, M=None, k=0, dtype=...)',
    summary:
      'Creates a square or rectangular identity array: ones lie on diagonal `k`, and other entries are zero.',
  },
  {
    pattern: /(?<!jnp)(?<!np)(?<!numpy)(?<!jax\.numpy)\.reshape\s*\(/,
    token: 'array.reshape(new_shape)',
    summary: 'Reorganizes tensor axes without changing the total element count (`array.size`).',
  },
  {
    pattern: /\bjnp\.reshape\s*\(/,
    token: 'jnp.reshape(a, newshape)',
    summary:
      'Returns a JAX array with the requested shape while preserving element order and count.',
  },
  {
    pattern: /\bjax\.numpy\.reshape\s*\(/,
    token: 'jax.numpy.reshape(a, newshape)',
    summary:
      'Returns a JAX array with the requested shape while preserving element order and count.',
  },
  {
    pattern: /\bnp\.reshape\s*\(/,
    token: 'np.reshape(a, newshape)',
    summary:
      'Returns a NumPy array with the requested shape while preserving element order and count.',
  },
  {
    pattern: /(?<!jnp)(?<!np)(?<!numpy)(?<!jax\.numpy)\.sum\s*\(/,
    token: 'array.sum(axis=...)',
    summary: 'Adds values along the selected axis; that axis is removed unless `keepdims=True`.',
  },
  {
    pattern: /(?<!jnp)(?<!np)(?<!numpy)(?<!jax\.numpy)\.mean\s*\(/,
    token: 'array.mean(axis=..., keepdims=...)',
    summary:
      'Averages values along the selected axis; that axis is removed unless `keepdims=True`.',
  },
  {
    pattern: /\bjnp\.sum\s*\(/,
    token: 'jnp.sum(x, axis=...)',
    summary: 'Adds JAX array values along the selected axis.',
  },
  {
    pattern: /\bjnp\.mean\s*\(/,
    token: 'jnp.mean(x, axis=..., keepdims=...)',
    summary: 'Averages JAX array values along the selected axis.',
  },
  {
    pattern: /\bjnp\.allclose\s*\(/,
    token: 'jnp.allclose(actual, expected, rtol=..., atol=...)',
    summary:
      'Returns a JAX boolean scalar indicating whether corresponding values are within the requested tolerance.',
  },
  {
    pattern: /\bnp\.allclose\s*\(/,
    token: 'np.allclose(actual, expected, rtol=..., atol=...)',
    summary:
      'Returns a Python boolean indicating whether NumPy-convertible values satisfy the tolerance test.',
  },
  {
    pattern: /\bnp\.testing\.assert_allclose\s*\(/,
    token: 'np.testing.assert_allclose(actual, desired, rtol=..., atol=...)',
    summary:
      'Checks a host-side numerical expectation and raises `AssertionError` when values differ beyond tolerance.',
  },
  {
    pattern: /\bjnp\.isfinite\s*\(/,
    token: 'jnp.isfinite(x)',
    summary: 'Returns a JAX boolean array marking elements that are neither `NaN` nor infinite.',
  },
  {
    pattern: /\bnp\.isfinite\s*\(/,
    token: 'np.isfinite(x)',
    summary: 'Returns a NumPy boolean array marking elements that are neither `NaN` nor infinite.',
  },
  {
    pattern: /\bjax\.grad\s*\(/,
    token: 'jax.grad(loss_fn)(params, ...)',
    summary:
      'Transforms a scalar-output function into one returning the gradient PyTree with the same structure as `params`.',
  },
  {
    pattern: /\bjax\.value_and_grad\s*\(/,
    token: 'jax.value_and_grad(loss_fn)(params, ...)',
    summary:
      'Transforms a pure function to return its value and gradient `(value, grads)` from the same evaluation.',
  },
  {
    pattern: /\bnnx\.value_and_grad\s*\(/,
    token: 'nnx.value_and_grad(loss_fn)(model, ...)',
    summary:
      'Differentiates a loss involving Flax NNX objects and handles their filtered state through the NNX transform.',
  },
  {
    pattern: /\bjax\.jvp\s*\(/,
    token: 'jax.jvp(fn, primals, tangents)',
    summary:
      'Computes a function value and Jacobian-vector product for the supplied primal values and tangent directions.',
  },
  {
    pattern: /\bjax\.vjp\s*\(/,
    token: 'jax.vjp(fn, *primals)',
    summary: 'Computes a function value and a pullback for vector-Jacobian products.',
  },
  {
    pattern: /\bjax\.linearize\s*\(/,
    token: 'jax.linearize(fn, *primals)',
    summary: 'Computes a function value and a reusable linearized function for tangent inputs.',
  },
  {
    pattern: /\bjax\.jacfwd\s*\(/,
    token: 'jax.jacfwd(fn)',
    summary:
      'Builds the Jacobian with forward-mode differentiation, often effective when there are fewer inputs than outputs.',
  },
  {
    pattern: /\bjax\.jacrev\s*\(/,
    token: 'jax.jacrev(fn)',
    summary:
      'Builds the Jacobian with reverse-mode differentiation, often effective when there are fewer outputs than inputs.',
  },
  {
    pattern: /\bjax\.hessian\s*\(/,
    token: 'jax.hessian(fn)',
    summary: 'Computes the second-derivative matrix of a scalar-valued function.',
  },
  {
    pattern: /\bjax\.vmap\s*\(/,
    token: 'jax.vmap(fn, in_axes=..., out_axes=...)',
    summary: 'Vectorizes a single-example function across mapped array axes without a Python loop.',
  },
  {
    pattern: /\bjax\.jit\s*\(|@jax\.jit\b/,
    token: 'jax.jit(fn) / @jax.jit',
    summary:
      'Traces a function for its input signatures and compiles it; changed shapes, dtypes, or static arguments can trigger another compilation.',
  },
  {
    pattern: /\bnnx\.jit\s*\(|@nnx\.jit\b/,
    token: 'nnx.jit(fn) / @nnx.jit',
    summary:
      'Compiles an NNX function while tracking and propagating NNX object state through the transformation.',
  },
  {
    pattern: /\bjax\.lax\.scan\s*\(/,
    token: 'jax.lax.scan(step_fn, init_carry, xs, length=...)',
    summary:
      'Compiles a sequential loop whose step returns `(next_carry, y)`, yielding `(final_carry, stacked_ys)`.',
  },
  {
    pattern: /\bjax\.tree\.map\s*\(/,
    token: 'jax.tree.map(fn, tree, ...)',
    summary: 'Applies a function leaf-by-leaf across matching PyTrees.',
  },
  {
    pattern: /\bjax\.tree_util\.tree_map\s*\(/,
    token: 'jax.tree_util.tree_map(fn, tree, ...)',
    summary: 'Applies a function leaf-by-leaf across matching PyTrees.',
  },
  {
    pattern: /\bjax\.random\.PRNGKey\s*\(/,
    token: 'jax.random.PRNGKey(seed)',
    summary: 'Creates a legacy-format explicit PRNG key from a seed.',
  },
  {
    pattern: /\bjax\.random\.key\s*\(/,
    token: 'jax.random.key(seed)',
    summary: 'Creates a typed explicit PRNG key from a seed.',
  },
  {
    pattern: /\bjax\.random\.split\s*\(/,
    token: 'jax.random.split(key, num=...)',
    summary:
      'Derives new keys for independent random operations; using the same key again repeats its random stream.',
  },
  {
    pattern: /\bjax\.random\.normal\s*\(/,
    token: 'jax.random.normal(key, shape, dtype=...)',
    summary: 'Draws normal random values using an explicit PRNG key.',
  },
  {
    pattern: /\bjax\.random\.uniform\s*\(/,
    token: 'jax.random.uniform(key, shape, minval=..., maxval=...)',
    summary: 'Draws uniform random values using an explicit PRNG key.',
  },
  {
    pattern: /\bjax\.random\.choice\s*\(/,
    token: 'jax.random.choice(key, a, shape=...)',
    summary: 'Samples from an array or integer range using an explicit PRNG key.',
  },
  {
    pattern: /\bjax\.random\.bernoulli\s*\(/,
    token: 'jax.random.bernoulli(key, p=..., shape=...)',
    summary: 'Draws Bernoulli boolean samples using an explicit PRNG key.',
  },
  {
    pattern: /\bjax\.random\.randint\s*\(/,
    token: 'jax.random.randint(key, shape, minval, maxval)',
    summary: 'Draws integer samples from a half-open interval using an explicit PRNG key.',
  },
  {
    pattern: /\bnnx\.Module\s*\(/,
    token: 'nnx.Module',
    summary:
      'Defines a Flax NNX module whose variables and child modules are represented as object state.',
  },
  {
    pattern: /\bnnx\.Linear\s*\(/,
    token: 'nnx.Linear(in_features, out_features, ...)',
    summary: 'Creates a learned affine layer in Flax NNX.',
  },
  {
    pattern: /\bnnx\.Rngs\s*\(/,
    token: 'nnx.Rngs(seed)',
    summary: 'Provides named random streams used to initialize and operate on NNX modules.',
  },
  {
    pattern: /\bnnx\.Optimizer\s*\(/,
    token: 'nnx.Optimizer(model, tx, wrt=...)',
    summary: 'Wraps an Optax transformation with NNX model parameter and optimizer state.',
  },
  {
    pattern: /\boptax\.adam\s*\(/,
    token: 'optax.adam(learning_rate, ...)',
    summary:
      'Creates Adam’s gradient transformation; call its `update` and apply updates to the parameters.',
  },
  {
    pattern: /\boptax\.sgd\s*\(/,
    token: 'optax.sgd(learning_rate, ...)',
    summary: 'Creates stochastic gradient descent’s gradient transformation.',
  },
  {
    pattern: /\boptax\.adamw\s*\(/,
    token: 'optax.adamw(learning_rate, ...)',
    summary: 'Creates AdamW’s gradient transformation with decoupled weight decay.',
  },
  {
    pattern: /\boptax\.chain\s*\(/,
    token: 'optax.chain(tx, ...)',
    summary: 'Composes gradient transformations in sequence.',
  },
  {
    pattern: /\boptax\.clip_by_global_norm\s*\(/,
    token: 'optax.clip_by_global_norm(max_norm)',
    summary: 'Scales a gradient PyTree when its global norm exceeds `max_norm`.',
  },
  {
    pattern: /\boptax\.softmax_cross_entropy\s*\(/,
    token: 'optax.softmax_cross_entropy(logits, labels)',
    summary:
      'Computes per-example categorical cross entropy from logits and probability-distribution labels, reducing over the class axis.',
  },
  {
    pattern: /\boptax\.sigmoid_binary_cross_entropy\s*\(/,
    token: 'optax.sigmoid_binary_cross_entropy(logits, labels)',
    summary:
      'Computes elementwise binary cross entropy from logits and labels; output has the same shape and needs an explicit reduction.',
  },
  {
    pattern: /\boptax\.losses\.softmax_cross_entropy\s*\(/,
    token: 'optax.losses.softmax_cross_entropy(logits, labels)',
    summary:
      'Computes per-example categorical cross entropy from logits and probability-distribution labels, reducing over the class axis.',
  },
  {
    pattern: /\boptax\.losses\.sigmoid_binary_cross_entropy\s*\(/,
    token: 'optax.losses.sigmoid_binary_cross_entropy(logits, labels)',
    summary:
      'Computes elementwise binary cross entropy from logits and labels; output has the same shape and needs an explicit reduction.',
  },
  {
    pattern: /\bjax\.pmap\s*\(/,
    token: 'jax.pmap(fn, in_axes=..., out_axes=...)',
    summary:
      'Maps a function across leading array axes and executes replicas in parallel on XLA devices.',
  },
  {
    pattern: /(?<![\w.])jax\.device_put\s*\(/,
    token: 'jax.device_put(x, device=...)',
    summary:
      'Places or transfers an array to the requested device or sharding; dispatch is asynchronous.',
  },
  {
    pattern: /\bMesh\s*\(/,
    token: 'Mesh(devices, axis_names)',
    summary: 'Names device axes in a mesh used to describe array placement.',
  },
  {
    pattern: /\bPartitionSpec\s*\(/,
    token: 'PartitionSpec(...)',
    summary: 'Specifies which logical array axes are partitioned over named mesh axes.',
  },
  {
    pattern: /\bNamedSharding\s*\(/,
    token: 'NamedSharding(mesh, spec)',
    summary: 'Combines a device mesh and partition specification to describe array placement.',
  },
  {
    pattern: /\bshard_map\s*\(/,
    token: 'shard_map(fn, mesh=..., in_specs=..., out_specs=...)',
    summary:
      'Runs a function on per-device shards with an explicit mesh and input/output partition specs.',
  },
  {
    pattern: /\bjnp\.linalg\.lstsq\s*\(/,
    token: 'jnp.linalg.lstsq(A, b, rcond=...)',
    summary:
      'Finds `x` minimizing `||A x - b||²`. For `A` shaped `(M, N)` and `b` shaped `(M,)` or `(M, K)`, `x` has shape `(N,)` or `(N, K)`. In the pinned JAX API, the summed squared residual has shape `(1,)` for vector `b` and `(K,)` for matrix `b`; rank and singular values describe the matrix.',
  },
  {
    pattern: /(?<!jax)\.block_until_ready\s*\(/,
    token: 'output.block_until_ready()',
    summary:
      'Waits for asynchronous device work on this array to finish, which is needed before measuring completed computation time.',
  },
  {
    pattern: /(?<![\w.])jax\.block_until_ready\s*\(/,
    token: 'jax.block_until_ready(tree)',
    summary: 'Waits for asynchronous device work on every array in a PyTree to finish.',
  },
];

/**
 * Replace comments and quoted literals with spaces while preserving offsets and newlines.
 * @param {string} code
 */
function maskPythonCommentsAndStrings(code) {
  const chars = code.split('');
  let quote = '';
  let triple = false;
  let escaped = false;
  let comment = false;

  for (let i = 0; i < chars.length; i += 1) {
    const char = chars[i];
    const next = chars[i + 1];
    if (comment) {
      if (char === '\n') comment = false;
      else chars[i] = ' ';
      continue;
    }
    if (quote) {
      if (char !== '\n') chars[i] = ' ';
      if (escaped) {
        escaped = false;
        continue;
      }
      if (char === '\\') {
        escaped = true;
        continue;
      }
      if (triple && char === quote && next === quote && chars[i + 2] === quote) {
        chars[i + 1] = ' ';
        chars[i + 2] = ' ';
        i += 2;
        quote = '';
        triple = false;
      } else if (!triple && char === quote) {
        quote = '';
      }
      continue;
    }
    if (char === '#') {
      comment = true;
      chars[i] = ' ';
    } else if (char === '"' || char === "'") {
      quote = char;
      triple = next === char && chars[i + 2] === char;
      chars[i] = ' ';
      if (triple) {
        chars[i + 1] = ' ';
        chars[i + 2] = ' ';
        i += 2;
      }
    }
  }
  return chars.join('');
}

/**
 * Find the first call that is not merely the tail of a longer attribute path.
 * @param {string} source
 * @param {RegExp} pattern
 * @returns {number}
 */
function findCallIndex(source, pattern) {
  const callPattern = new RegExp(pattern.source, `${pattern.flags}g`);
  let match = callPattern.exec(source);
  while (match) {
    if (match[0].startsWith('.') || source[match.index - 1] !== '.') return match.index;
    match = callPattern.exec(source);
  }
  return -1;
}

/**
 * Extract the key APIs used in a Python code block.
 * @param {string} code
 */
export function extractCodeApis(code) {
  const source = maskPythonCommentsAndStrings(code);
  const matches = API_CATALOG.map((item) => ({ item, index: findCallIndex(source, item.pattern) }))
    .filter(({ index }) => index >= 0)
    .sort((left, right) => left.index - right.index)
    .map(({ item }) => item);
  return matches.slice(0, 4);
}

/** @typedef {{ inputs: string[]; outputs: string[]; invariants: string[]; hints: string[]; code: string }} LearnerStarterContract */

/** @param {unknown} value @returns {value is LearnerStarterContract} */
export function isLearnerStarterContract(value) {
  if (!value || typeof value !== 'object') return false;
  const candidate = /** @type {Record<string, unknown>} */ (value);
  return (
    ['inputs', 'outputs', 'invariants', 'hints'].every((key) => {
      const items = candidate[key];
      return (
        Array.isArray(items) &&
        items.length > 0 &&
        items.every((item) => typeof item === 'string' && item.trim())
      );
    }) &&
    typeof candidate.code === 'string' &&
    Boolean(candidate.code.trim())
  );
}

/**
 * Return code from an authored learner contract. A legacy reference-code
 * argument intentionally yields no scaffold: a missing contract is a content
 * gap, not a reason to infer an answer-shaped starter.
 * @param {string | LearnerStarterContract | undefined} source
 */
export function buildStarterScaffold(source) {
  if (isLearnerStarterContract(source)) {
    return String(source.code || '').trim();
  }
  return '';
}

/**
 * Render a learner-authored starter contract. This function deliberately does
 * not accept or inspect the reference solution.
 * @param {LearnerStarterContract | undefined} contract
 * @param {string} title
 * @returns {string}
 */
export function renderLearnerStarterContract(contract, title = 'Learner starter contract') {
  if (!isLearnerStarterContract(contract)) {
    return `<aside class="learner-starter-pending"><strong>Learner starter pending.</strong> This lesson does not yet provide a separate input/output contract, checks, hints, and TODO scaffold.</aside>`;
  }
  /** @param {string} heading @param {string[]} values */
  const list = (heading, values) =>
    values?.length
      ? `<h4>${heading}</h4><ul>${values.map((value) => `<li>${inlineMath(value)}</li>`).join('')}</ul>`
      : '';
  const code = buildStarterScaffold(contract);
  return `<section class="learner-starter-contract"><p class="code-guide-kicker">${escapeHtml(title)}</p>
    ${list('Inputs', contract.inputs)}${list('Outputs', contract.outputs)}${list('Invariants to check', contract.invariants)}${list('Hints', contract.hints)}
    <div class="starter-scaffold"><div class="starter-scaffold-header"><span>Your attempt · complete the TODOs</span></div><pre data-language="python"><code>${CourseCode.highlight(code, 'python')}</code></pre></div>
  </section>`;
}

/**
 * Extract learner-facing steps from an authored contract, never from solution comments.
 * @param {LearnerStarterContract | undefined} source
 */
export function extractCodingSteps(source) {
  if (source && typeof source === 'object' && Array.isArray(source.hints)) {
    return source.hints.slice(0, 5);
  }
  return [];
}

/**
 * Render a compact breakdown of the JAX/Python APIs used in a Stage 2 build step.
 * @param {string} code
 */
export function renderBuildStepCodeBreakdown(code) {
  const apis = extractCodeApis(code);
  if (!apis.length) return '';
  return `<details class="code-teaching-guide build-code-breakdown reading-disclosure">
    <summary>Explain the APIs in this step</summary>
    <ul class="code-api-list">${apis
      .map((api) => `<li><code>${escapeHtml(api.token)}</code>: ${inlineMath(api.summary)}</li>`)
      .join('')}</ul>
  </details>`;
}

/**
 * Render a pre-exercise teaching walkthrough + starter code scaffold before the learner writes code.
 * @param {string} _solutionCode
 * @param {string} title
 * @param {LearnerStarterContract | undefined} learnerStarter
 */
export function renderExerciseCodeTeacher(
  _solutionCode,
  title = 'How to write this code',
  learnerStarter,
) {
  const starter = isLearnerStarterContract(learnerStarter) ? learnerStarter : undefined;
  const scaffold = buildStarterScaffold(starter);
  const apis = starter ? extractCodeApis(scaffold) : [];
  const contract = renderLearnerStarterContract(
    starter,
    'Learner-only inputs, outputs, checks and hints',
  );
  return `<details class="code-teaching-guide"><summary>Show a writing guide and starter scaffold</summary>
    <p class="code-guide-kicker">${escapeHtml(title)}</p>
    ${contract}
    ${
      apis.length
        ? `<p class="code-guide-subhead"><strong>1. Key functions &amp; syntax to use:</strong></p>
    <ul class="code-api-list">${apis
      .map((api) => `<li><code>${escapeHtml(api.token)}</code> — ${inlineMath(api.summary)}</li>`)
      .join('')}</ul>`
        : ''
    }
  </details>`;
}
