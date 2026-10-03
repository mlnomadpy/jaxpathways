"""First gradient: run in an environment with JAX installed."""
from jax import grad


def f(x):
    return x ** 2


if __name__ == '__main__':
    derivative = float(grad(f)(3.0))
    assert abs(derivative - 6.0) < 1e-6
    print(f'Derivative of x² at x=3: {derivative}')
