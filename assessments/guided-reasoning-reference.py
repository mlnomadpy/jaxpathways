"""Independent arithmetic checks for the new hand-worked teaching examples.

These check the stated analytic examples, not learner competence or hardware
performance. Run from the repository root with the course Python environment.
"""
import math
import numpy as np


def verify():
    checks = []

    def close(name, actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
        checks.append(name)

    x = np.array([[1., 2., 3.], [4., 5., 6.]])
    close("row and feature reductions", [*x.sum(axis=1), *x.sum(axis=0)], [6, 15, 5, 7, 9])
    mean = np.mean([2., 6.])
    close("fit training mean; apply to held-out", np.array([2., 6., 10.]) - mean, [-2, 2, 6])
    prediction = np.array([1., 3.])
    close("accidental all-pairs broadcast", np.mean((prediction - prediction[:, None]) ** 2), 2)
    close("local tangent approximation error", 3.01**2 - (9 + 6 * .01), .0001)
    close("additive scan outputs", np.cumsum([1, 2, 3]), [1, 3, 6])
    v, u = np.array([3., 4.]), np.array([2., 0.])
    projection = u * (v @ u) / (u @ u)
    close("scale-invariant projection", projection, [3, 0])
    close("projection orthogonality", (v - projection) @ u, 0)
    x, w = np.array([[1., 2.], [3., 0.]]), np.array([2., -1.])
    prediction = x @ w + 1
    close("complete regression prediction", prediction, [1, 7])
    close("complete regression MSE", np.mean((prediction - [2, 5]) ** 2), 2.5)
    jacobian = np.array([[4., 1.], [3., 2.]])
    direction, sensitivity = np.array([1., -1.]), np.array([2., -1.])
    close("JVP", jacobian @ direction, [3, 1])
    close("VJP", jacobian.T @ sensitivity, [5, 0])
    close("adjoint identity", [sensitivity @ (jacobian @ direction), (jacobian.T @ sensitivity) @ direction], [5, 5])
    close("curvature coordinate factors", 1 - .25 * np.array([1, 10]), [.75, -1.5])
    small, large = 0 - .1 * (-6), 0 - 1.1 * (-6)
    close("gradient descent stable and overshooting steps", [small, (small-3)**2, large, (large-3)**2], [.6, 5.76, 6.6, 12.96])
    close("weighted example mean", (2 * 1 + 1 * 4) / 3, 2)
    g = np.array([3., 4.])
    close("global norm clipping", g * min(1, 2.5 / np.linalg.norm(g)), [1.5, 2])
    close("coordinate clipping norm", np.linalg.norm(np.clip(g, -2.5, 2.5)), 2.5 * math.sqrt(2))
    close("attention value mixture", np.array([.25, .75]) @ np.array([[2., 0.], [0., 4.]]), [.5, 3])
    close("roofline ridge and bandwidth bound", [1000 / 100, 100 * 2], [10, 200])
    close("Euler refinement at equal time", [(1 - .5), (1 - .25) ** 2], [.5, .5625])
    close("standard error versus observation spread", 2 / math.sqrt(100), .2)
    covariance = np.array([[1., .8], [.8, 1.]])
    close("variance of a sum", np.ones(2) @ covariance @ np.ones(2), 3.6)
    close("zero-covariance sum variance", np.ones(2) @ np.eye(2) @ np.ones(2), 2)
    close("bias then ReLU", np.maximum(np.array([-2., 1.]) + [1, -3], 0), [0, 0])
    scale, zero_point = .5, 3
    q = round(1 / scale) + zero_point
    close("affine quantization round trip", [q, scale * (q - zero_point)], [5, 1])
    close("DPO ratio improves while chosen probability falls", [.4 / .2, .3 / .1], [2, 3])
    print(f"PASS: {len(checks)} independent worked-example calculations")
    return checks


if __name__ == "__main__":
    verify()
