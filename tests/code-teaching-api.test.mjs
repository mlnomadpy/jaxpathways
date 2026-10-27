import test from 'node:test';
import assert from 'node:assert/strict';
import { extractCodeApis } from '../src/lib/code-teaching.js';

const tokens = (code) => extractCodeApis(code).map(({ token }) => token);

test('teaches Flax NNX gradients and JAX gradients with their own APIs', () => {
  const apis = extractCodeApis(`
loss, grads = nnx.value_and_grad(loss_fn)(model)
value, grads = jax.value_and_grad(loss_fn)(params)
compiled = jax.jit(step)
stateful = nnx.jit(train_step)
`);

  assert.deepEqual(
    apis.map(({ token }) => token),
    [
      'nnx.value_and_grad(loss_fn)(model, ...)',
      'jax.value_and_grad(loss_fn)(params, ...)',
      'jax.jit(fn) / @jax.jit',
      'nnx.jit(fn) / @nnx.jit',
    ],
  );
  assert.match(apis[0].summary, /Flax NNX/);
  assert.match(apis[1].summary, /pure function/);
  assert.match(apis[2].summary, /input signatures/);
  assert.match(apis[3].summary, /tracking and propagating NNX object state/);
});

test('distinguishes JAX boolean comparisons from NumPy assertions', () => {
  const apis = extractCodeApis(`
assert np.allclose(actual, expected)
np.testing.assert_allclose(actual, expected, atol=1e-6)
ok = jnp.allclose(actual, expected)
`);

  assert.deepEqual(
    apis.map(({ token }) => token),
    [
      'np.allclose(actual, expected, rtol=..., atol=...)',
      'np.testing.assert_allclose(actual, desired, rtol=..., atol=...)',
      'jnp.allclose(actual, expected, rtol=..., atol=...)',
    ],
  );
  assert.match(apis[0].summary, /Python boolean/);
  assert.match(apis[1].summary, /raises `AssertionError`/);
  assert.match(apis[2].summary, /JAX boolean scalar/);
});

test('gives array constructors, device placement, and mapping distinct descriptions', () => {
  const apis = extractCodeApis(`
identity = jnp.eye(3)
offset = jnp.full((2, 2), 7)
x = jax.device_put(data, sharding)
y = jax.pmap(step)(x)
`);

  assert.deepEqual(
    apis.map(({ token }) => token),
    [
      'jnp.eye(N, M=None, k=0, dtype=...)',
      'jnp.full(shape, fill_value, dtype=...)',
      'jax.device_put(x, device=...)',
      'jax.pmap(fn, in_axes=..., out_axes=...)',
    ],
  );
  assert.match(apis[0].summary, /identity array/);
  assert.match(apis[1].summary, /fill_value/);
  assert.match(apis[2].summary, /Places or transfers/);
  assert.match(apis[3].summary, /parallel on XLA devices/);
});

test('keeps Optax loss calls separate and explains their output contracts', () => {
  const apis = extractCodeApis(`
losses = optax.softmax_cross_entropy(logits, one_hot)
element_losses = optax.sigmoid_binary_cross_entropy(logits, labels)
`);

  assert.deepEqual(
    apis.map(({ token }) => token),
    [
      'optax.softmax_cross_entropy(logits, labels)',
      'optax.sigmoid_binary_cross_entropy(logits, labels)',
    ],
  );
  assert.match(apis[0].summary, /class axis/);
  assert.match(apis[1].summary, /same shape/);
});

test('explains least squares solution, residual, and matrix shapes', () => {
  const apis = extractCodeApis('x, residual, rank, singular_values = jnp.linalg.lstsq(A, b)');

  assert.deepEqual(
    apis.map(({ token }) => token),
    ['jnp.linalg.lstsq(A, b, rcond=...)'],
  );
  assert.match(apis[0].summary, /minimizing `\|\|A x - b\|\|²`/);
  assert.match(apis[0].summary, /`\(M, N\)`/);
  assert.match(apis[0].summary, /summed squared residual/);
  assert.match(apis[0].summary, /shape `\(1,\)` for vector `b`/);
  assert.match(apis[0].summary, /`\(K,\)` for matrix `b`/);
  assert.match(apis[0].summary, /rank and singular values/);
});

test('does not confuse namespace reductions with array methods', () => {
  assert.deepEqual(tokens('jnp.mean(values)'), ['jnp.mean(x, axis=..., keepdims=...)']);
  assert.deepEqual(tokens('jnp.sum(values)'), ['jnp.sum(x, axis=...)']);
  assert.deepEqual(tokens('values.mean(axis=0)'), ['array.mean(axis=..., keepdims=...)']);
  assert.deepEqual(tokens('values.sum(axis=0)'), ['array.sum(axis=...)']);
  assert.deepEqual(tokens('np.mean(values)\nnp.sum(values)'), []);
});

test('distinguishes namespace reshape functions from the array method', () => {
  assert.deepEqual(
    tokens(`
host = np.reshape(values, (2, 3))
method = values.reshape((2, 3))
device = jnp.reshape(values, (2, 3))
explicit_namespace = jax.numpy.reshape(values, (2, 3))
`),
    [
      'np.reshape(a, newshape)',
      'array.reshape(new_shape)',
      'jnp.reshape(a, newshape)',
      'jax.numpy.reshape(a, newshape)',
    ],
  );
});

test('labels JAX and array synchronization methods accurately', () => {
  const apis = extractCodeApis(`
ready = jax.block_until_ready(tree)
ready_array = output.block_until_ready()
placed = jax.device_put(data, device=target)
`);

  assert.deepEqual(
    apis.map(({ token }) => token),
    ['jax.block_until_ready(tree)', 'output.block_until_ready()', 'jax.device_put(x, device=...)'],
  );
});

test('ignores call-like text in comments and quoted or triple-quoted strings', () => {
  assert.deepEqual(
    tokens(String.raw`
# nnx.value_and_grad(loss_fn)(model) and jnp.linalg.lstsq(A, b)
message = "np.allclose(a, b)"
doc = '''jax.device_put(x)\noptax.softmax_cross_entropy(z, y)'''
single = 'jnp.eye(3)'
`),
    [],
  );
});

test('returns recognized calls in source order, capped at four, and ignores unknown calls', () => {
  assert.deepEqual(
    tokens(`
first = jnp.full((2,), 3)
second = jnp.eye(2)
third = jnp.linalg.lstsq(A, b)
unrecognized = custom_solver(A, b)
fourth = jax.device_put(first)
fifth = np.allclose(first, second)
`),
    [
      'jnp.full(shape, fill_value, dtype=...)',
      'jnp.eye(N, M=None, k=0, dtype=...)',
      'jnp.linalg.lstsq(A, b, rcond=...)',
      'jax.device_put(x, device=...)',
    ],
  );
});

test('does not infer an API from a bare symbol or an unqualified call', () => {
  assert.deepEqual(tokens('jnp.eye\nallclose(a, b)\nmy_device_put(x)\nfake.jax.device_put(x)'), []);
});
