"""Automated reference verifier for all 11 synthesis assessments."""
import jax
import jax.numpy as jnp
import numpy as np


def check_foundations():
    x = jnp.array([-2.0, 0.0, 3.0], dtype=jnp.float32)
    y = -1.5 * x + 0.25
    params = {"w": jnp.float32(0.5), "b": jnp.float32(-0.5)}
    loss_fn = lambda p: jnp.mean((p["w"] * x + p["b"] - y) ** 2)
    grads = jax.grad(loss_fn)(params)
    res = np.asarray(params["w"] * x + params["b"] - y)
    assert np.allclose(float(grads["w"]), float(np.mean(2.0 * res * np.asarray(x))), atol=1e-5)
    assert np.allclose(float(grads["b"]), float(np.mean(2.0 * res)), atol=1e-5)


def check_math():
    X = jnp.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], dtype=jnp.float32)
    H = (2.0 / len(X)) * (X.T @ X)
    eigvals = jnp.linalg.eigvalsh(H)
    assert float(eigvals[0]) > 0.0 and float(eigvals[1]) > float(eigvals[0])


def check_models():
    logits = jnp.array([-100.0, 0.0, 100.0], dtype=jnp.float32)
    labels = jnp.array([0.0, 1.0, 1.0], dtype=jnp.float32)
    bce = jnp.mean(jnp.logaddexp(0.0, logits) - labels * logits)
    assert jnp.isfinite(bce)


def check_science():
    u0 = jnp.array([3.0, 1.0], dtype=jnp.float32)
    sens_total = jax.grad(lambda k: jnp.sum(u0 + 0.05 * jnp.array([-k * (u0[0] - u0[1]), k * (u0[0] - u0[1])])))(0.4)
    assert abs(float(sens_total)) < 1e-6


def check_rl():
    ratios = jnp.array([0.6, 1.0, 1.4], dtype=jnp.float32)
    pos = jnp.minimum(ratios * 2.0, jnp.clip(ratios, 0.8, 1.2) * 2.0)
    neg = jnp.minimum(ratios * -2.0, jnp.clip(ratios, 0.8, 1.2) * -2.0)
    assert np.allclose(np.asarray(pos), [1.2, 2.0, 2.4], atol=1e-5)
    assert np.allclose(np.asarray(neg), [-1.6, -2.0, -2.8], atol=1e-5)


def check_probability():
    X = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([-1.0, 1.0, 3.0])
    prec = X.T @ X + np.eye(2) / 4.0
    cov = np.linalg.inv(prec)
    mean = cov @ (X.T @ y)
    assert abs(cov[0, 1]) < 1e-7 and mean[1] > 1.5


def check_internals():
    f = lambda x: jnp.stack([x[0] * x[1], jnp.sin(x[0]) + x[1] ** 2, x[0] ** 2 - x[1]])
    x0 = jnp.array([0.3, -0.6], dtype=jnp.float32)
    v = jnp.array([0.7, -0.2], dtype=jnp.float32)
    u = jnp.array([0.5, -0.8, 1.2], dtype=jnp.float32)
    _, jvp_out = jax.jvp(f, (x0,), (v,))
    _, vjp_fn = jax.vjp(f, x0)
    assert abs(float(jnp.dot(u, jvp_out) - jnp.dot(vjp_fn(u)[0], v))) < 1e-6


def check_ship():
    arrivals = [0, 1, 4]
    service = 2
    starts, finishes = [], []
    clock = 0
    for a in arrivals:
        s = max(a, clock)
        f = s + service
        starts.append(s)
        finishes.append(f)
        clock = f
    assert starts == [0, 2, 4] and finishes == [2, 4, 6]


def check_ops():
    true_rate = 200.0 / (2.0 + 6.0)
    naive_rate = 0.5 * (100.0 / 2.0 + 100.0 / 6.0)
    assert true_rate == 25.0 and naive_rate > true_rate


def check_scale():
    X = np.arange(16, dtype=np.float32).reshape(8, 2) / 8.0
    w = np.array([0.5, -0.25], dtype=np.float32)
    y = X @ np.array([1.2, -0.8], dtype=np.float32)
    g1 = (2.0 / 3.0) * X[:3].T @ (X[:3] @ w - y[:3])
    g2 = (2.0 / 5.0) * X[3:].T @ (X[3:] @ w - y[3:])
    g_global = (2.0 / 8.0) * X.T @ (X @ w - y)
    assert np.allclose((3.0 * g1 + 5.0 * g2) / 8.0, g_global, atol=1e-6)


def check_tpu():
    q = jnp.ones((4, 8), dtype=jnp.float32)
    k1 = jnp.arange(32, dtype=jnp.float32).reshape(4, 8) / 32.0
    k2 = k1.at[3].set(99.0)
    mask = jnp.tril(jnp.ones((4, 4), dtype=bool))
    att = lambda k: jax.nn.softmax(jnp.where(mask, (q @ k.T) / jnp.sqrt(8.0), -1e9), axis=-1) @ k
    assert float(jnp.max(jnp.abs(att(k1)[:3] - att(k2)[:3]))) < 1e-6


if __name__ == "__main__":
    for fn in (
        check_foundations,
        check_math,
        check_models,
        check_science,
        check_rl,
        check_probability,
        check_internals,
        check_ship,
        check_ops,
        check_scale,
        check_tpu,
    ):
        fn()
    print("PASS: all 11 synthesis assessment reference checks verified.")
