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
      'Computes exact forward-mode JVP, reverse-mode VJP, full Jacobians, or second-order curvature.',
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
      'Traces `fn` with abstract shapes and compiles a fused XLA executable cached by input shape and dtype.',
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
      'Manages explicit, stateless PRNG keys—always split a key before passing subkeys into independent random draws.',
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

  /** @type {Array<{ token: string; summary: string }>} */
  const fallback = [];
  const fnCalls = [...code.matchAll(/\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)?)\s*\(/g)]
    .map((m) => m[1])
    .filter(
      (name) =>
        ![
          'print',
          'len',
          'range',
          'dict',
          'list',
          'tuple',
          'float',
          'int',
          'str',
          'bool',
          'set',
        ].includes(name),
    );
  const uniqueCalls = [...new Set(fnCalls)].slice(0, 2);
  for (const fn of uniqueCalls) {
    fallback.push({
      token: `${fn}(...)`,
      summary: `Call \`${fn}\` with your updated parameters or inputs from this lesson's workspace.`,
    });
  }
  if (/\bassert\b/.test(code)) {
    fallback.push({
      token: 'assert condition',
      summary:
        'Verify that the observed output shape, status, or numerical value satisfies the contract.',
    });
  }
  return fallback;
}

/**
 * Build a commented starter scaffold with TODOs from a reference solution snippet.
 * @param {string} solutionCode
 */
export function buildStarterScaffold(solutionCode) {
  const lines = solutionCode.split('\n');
  /** @type {string[]} */
  const out = [];
  for (const line of lines) {
    const trimmed = line.trim();
    const indent = (line.match(/^\s*/) || [''])[0];
    if (!trimmed || trimmed.startsWith('#')) {
      out.push(line);
      continue;
    }
    if (/^(?:import|from)\s+/.test(trimmed)) {
      out.push(line);
      continue;
    }
    if (/^(?:with|for|while|try|except|else|finally|def|class)\b.*:\s*$/.test(trimmed)) {
      out.push(line);
      continue;
    }
    const assignMatch = line.match(/^(\s*)([a-zA-Z_][\w, ()[\]'".]*?)\s*=\s*(.+)$/);
    if (assignMatch && !trimmed.startsWith('assert ')) {
      const [, lead, lhs, rhs] = assignMatch;
      const callMatch = rhs.match(/^([a-zA-Z_][\w.]*)\s*\(/);
      const hint = callMatch ? `${callMatch[1]}(...)` : '...';
      out.push(`${lead}${lhs.trim()} = ${hint}  # TODO: compute ${lhs.trim()}`);
      continue;
    }
    if (trimmed.startsWith('assert ')) {
      const expr = trimmed.slice(7).trim();
      const lhs = expr.split(/==|!=|<=|>=|<|>|\band\b|\bor\b|\bin\b/)[0].trim();
      out.push(`${indent}assert ${lhs || '...'}  # TODO: complete assertion check`);
      continue;
    }
    if (trimmed.startsWith('return ')) {
      out.push(`${indent}return ...  # TODO: return computed result`);
      continue;
    }
    out.push(line);
  }
  return out.join('\n');
}

/**
 * Extract step-by-step natural language coding steps from a commented solution block.
 * @param {string} solutionCode
 */
export function extractCodingSteps(solutionCode) {
  const lines = solutionCode.split('\n');
  /** @type {string[]} */
  const steps = [];
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('#')) {
      const text = trimmed.replace(/^#+\s*/, '').trim();
      if (
        text &&
        !text.startsWith('Exercise solution:') &&
        !text.includes('(Practice):') &&
        !text.includes('(Challenge):') &&
        !text.includes('(Transfer')
      ) {
        steps.push(text);
      }
    }
  }
  if (steps.length === 0) {
    steps.push('Set up the changed input or configuration parameters requested in the prompt.');
    steps.push(
      'Run the target function or transformation and assert that the output matches your prediction.',
    );
  }
  return steps.slice(0, 5);
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
 * @param {string} solutionCode
 * @param {string} title
 */
export function renderExerciseCodeTeacher(solutionCode, title = 'How to write this code') {
  const apis = extractCodeApis(solutionCode);
  const steps = extractCodingSteps(solutionCode);
  const scaffold = buildStarterScaffold(solutionCode);
  return `<div class="code-teaching-guide">
    <p class="code-guide-kicker">${escapeHtml(title)} — Step-by-step recipe &amp; starter scaffold</p>
    ${
      apis.length
        ? `<p class="code-guide-subhead"><strong>1. Key functions &amp; syntax to use:</strong></p>
    <ul class="code-api-list">${apis
      .map((api) => `<li><code>${escapeHtml(api.token)}</code> — ${inlineMath(api.summary)}</li>`)
      .join('')}</ul>`
        : ''
    }
    <p class="code-guide-subhead"><strong>${apis.length ? '2.' : '1.'} Step-by-step implementation plan:</strong></p>
    <ol class="code-step-list">${steps.map((s) => `<li>${inlineMath(s)}</li>`).join('')}</ol>
    <div class="starter-scaffold">
      <div class="starter-scaffold-header">
        <span>Starter code scaffold · copy into your file &amp; fill in the TODOs</span>
      </div>
      <pre data-language="python"><code>${CourseCode.highlight(scaffold, 'python')}</code></pre>
    </div>
  </div>`;
}
