import { CourseCode } from './code.js';
import { escapeHtml } from './html.js';
import { inlineMath } from './math.js';

/** @type {Array<{ pattern: RegExp; token: string; summary: string }>} */
const API_CATALOG = [
  {
    pattern: /\bjnp\.arange\b/,
    token: 'jnp.arange(n, dtype=...)',
    summary:
      'Creates a 1-D JAX array of evenly spaced values `[0, 1, ..., n-1]` on the target device.',
  },
  {
    pattern: /\bjnp\.linspace\b/,
    token: 'jnp.linspace(start, stop, num)',
    summary: 'Creates `num` evenly spaced float points across the closed interval `[start, stop]`.',
  },
  {
    pattern: /\bjnp\.array\b/,
    token: 'jnp.array(values, dtype=...)',
    summary: 'Constructs an immutable device-backed JAX array from Python/NumPy values.',
  },
  {
    pattern: /\bjnp\.(?:zeros|ones|full|eye)\b/,
    token: 'jnp.zeros / jnp.ones(shape, dtype=...)',
    summary: 'Allocates a tensor of the given `shape` initialized with constants.',
  },
  {
    pattern: /\.reshape\s*\(/,
    token: 'array.reshape(new_shape)',
    summary: 'Reorganizes tensor axes without changing the total element count (`array.size`).',
  },
  {
    pattern: /\.(?:sum|mean)\s*\(|\bjnp\.(?:sum|mean)\b/,
    token: 'x.sum(axis=...) / x.mean(axis=..., keepdims=...)',
    summary:
      'Reduces values along the named `axis` (the axis you name is collapsed unless `keepdims=True`).',
  },
  {
    pattern: /\bjnp\.allclose\b|\bnp\.allclose\b|\bassert_allclose\b/,
    token: 'jnp.allclose(actual, expected, rtol=..., atol=...)',
    summary: 'Checks that two arrays match elementwise within floating-point tolerance.',
  },
  {
    pattern: /\bjnp\.isfinite\b|\bnp\.isfinite\b/,
    token: 'jnp.isfinite(x)',
    summary: 'Returns a boolean mask verifying that no element is `NaN` or `Inf`.',
  },
  {
    pattern: /\bjax\.grad\b/,
    token: 'jax.grad(loss_fn)(params, ...)',
    summary:
      'Transforms a scalar-output function into a function returning the gradient PyTree with the same structure as `params`.',
  },
  {
    pattern: /\b(?:jax|nnx)\.value_and_grad\b/,
    token: 'jax.value_and_grad(loss_fn)(params, ...)',
    summary:
      'Evaluates both the scalar loss and its gradient PyTree `(loss_val, grads)` in a single forward+backward pass.',
  },
  {
    pattern: /\bjax\.(?:jvp|vjp|linearize|jacfwd|jacrev|hessian)\b/,
    token: 'jax.jvp / jax.vjp / jax.jacfwd / jax.hessian',
    summary:
      'Computes JVPs, VJPs, Jacobians or Hessians by differentiating the implemented operations. Results still have floating-point error.',
  },
  {
    pattern: /\bjax\.vmap\b/,
    token: 'jax.vmap(fn, in_axes=..., out_axes=...)',
    summary:
      'Vectorizes a single-example function across a batch axis without writing a Python loop.',
  },
  {
    pattern: /\b(?:jax|nnx)\.jit\b/,
    token: 'jax.jit(fn) / @jax.jit',
    summary:
      'Traces `fn` for its input signatures and compiles an executable. Changed shapes, dtypes or static arguments can trigger another compilation.',
  },
  {
    pattern: /\bjax\.lax\.scan\b/,
    token: 'jax.lax.scan(step_fn, init_carry, xs, length=...)',
    summary:
      'Compiles a sequential loop where `step_fn(carry, x)` returns `(next_carry, y)`, returning `(final_carry, stacked_ys)`.',
  },
  {
    pattern: /\bjax\.tree\.map\b|\bjax\.tree_util\.tree_map\b/,
    token: 'jax.tree.map(lambda p, g: ..., params, grads)',
    summary:
      'Applies a function leaf-by-leaf across matching PyTrees (such as updating every parameter tensor with its gradient).',
  },
  {
    pattern: /\bjax\.random\.(?:PRNGKey|key|split|normal|uniform|choice|bernoulli|randint)\b/,
    token: 'jax.random.PRNGKey(seed) & jax.random.split(key)',
    summary:
      'Uses explicit PRNG keys. Split a key when you need independent random draws; reusing a key repeats its random stream.',
  },
  {
    pattern: /\bnnx\.(?:Module|Linear|Rngs|Optimizer|Param)\b/,
    token: 'nnx.Module / nnx.Linear / nnx.Optimizer',
    summary:
      'Flax NNX stateful module and optimizer containers with explicit RNG streams (`nnx.Rngs`) and traced graph updates.',
  },
  {
    pattern:
      /\boptax\.(?:adam|sgd|adamw|chain|clip_by_global_norm|softmax_cross_entropy|sigmoid_binary_cross_entropy)\b/,
    token: 'optax.adam(lr) / optax.apply_updates(params, updates)',
    summary:
      'Optax gradient transformations and numerically stable loss functions over parameter PyTrees.',
  },
  {
    pattern: /\b(?:Mesh|PartitionSpec|NamedSharding|shard_map|pmap|device_put)\b/,
    token: 'Mesh + PartitionSpec + NamedSharding',
    summary:
      'Maps logical tensor axes onto physical device mesh axes for SPMD data, tensor, or pipeline parallelism.',
  },
  {
    pattern: /\bblock_until_ready\b/,
    token: 'jax.block_until_ready(output)',
    summary:
      'Synchronizes with the accelerator/CPU device so asynchronous dispatch finishes before wall-clock timing.',
  },
];

/**
 * Extract the key APIs used in a Python code block.
 * @param {string} code
 */
export function extractCodeApis(code) {
  const matches = API_CATALOG.filter((item) => item.pattern.test(code));
  if (matches.length > 0) return matches.slice(0, 4);

  // Only explain APIs with an authored description. A guessed call name does
  // not tell a learner what the computation means.
  return [];
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
      ? `<h4>${heading}</h4><ul>${values.map((value) => `<li>${inlineMath(escapeHtml(value))}</li>`).join('')}</ul>`
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
  return `<div class="code-teaching-guide build-code-breakdown">
    <p class="code-guide-kicker">Code building blocks in this step</p>
    <ul class="code-api-list">${apis
      .map((api) => `<li><code>${escapeHtml(api.token)}</code> — ${inlineMath(api.summary)}</li>`)
      .join('')}</ul>
  </div>`;
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
