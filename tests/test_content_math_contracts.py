"""Numeric teaching contracts for the October content corrections, CPU only."""

import ast
import json
import os
from pathlib import Path
import unittest

os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COURSE = json.loads((ROOT / "curriculum/course.json").read_text())
LESSONS = {l["id"]: l for p in COURSE["phases"] for l in p["lessons"]}


def definitions(id):
    content = json.loads((ROOT / LESSONS[id]["path"] / "lesson.json").read_text())[
        "content"
    ]
    tree = ast.parse(content["code"])
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
    ]
    namespace = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            id,
            "exec",
        ),
        namespace,
    )
    return namespace


class TeachingMath(unittest.TestCase):
    def test_matrix_loss_and_gradient_have_matching_denominator(self):
        loss = definitions("transforms-04")["loss"]
        x = np.array([[1.0, 2.0], [-3.0, 0.5], [0.25, 4.0]], dtype=np.float32)
        w = np.array([0.3, -0.2], dtype=np.float32)
        y = np.array([1.0, 0.0, -2.0], dtype=np.float32)
        residual = x @ w - y
        expected = 2 * x.T @ residual / len(y)
        np.testing.assert_allclose(
            jax.grad(loss)(jnp.array(w), jnp.array(x), jnp.array(y)),
            expected,
            rtol=2e-6,
        )
        self.assertFalse(np.allclose(expected, x.T @ residual))
        np.testing.assert_allclose(
            loss(jnp.zeros(2), jnp.array([[1.0, 2.0]]), jnp.ones(1)), 1.0
        )

    def test_nnx_unfiltered_split_has_one_complete_state(self):
        scope = definitions("networks-02")
        nnx = scope["nnx"]
        model = scope["TinyMLP"]()
        result = nnx.split(model)
        self.assertEqual(len(result), 2)
        _, state = result
        self.assertEqual(sum(x.size for x in jax.tree.leaves(state)), 14)
        self.assertEqual(
            sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))), 13
        )

    def test_binary_loss_matches_one_logit_per_example(self):
        scope = definitions("networks-03")
        z = np.array([-4.0, 0.0, 2.0], dtype=np.float32)
        y = np.array([0.0, 1.0, 1.0], dtype=np.float32)

        class FrozenLogits:
            def __call__(self, x):
                return jnp.asarray(z)

        actual = scope["objective"](FrozenLogits(), None, jnp.asarray(y))
        np.testing.assert_allclose(
            actual, np.mean(np.logaddexp(0.0, z) - y * z), rtol=2e-6
        )
        self.assertNotAlmostEqual(
            float(actual),
            float(np.mean(np.logaddexp(0.0, z[None, :]) - y[:, None] * z[None, :])),
        )

    def test_minimal_block_normalization_has_no_affine_parameters(self):
        normalize = definitions("transformers-03")["layer_norm"]
        x = np.array([[1.0, 3.0], [-2.0, 7.0]], dtype=np.float32)
        expected = (x - x.mean(-1, keepdims=True)) / np.sqrt(
            x.var(-1, keepdims=True) + 1e-5
        )
        np.testing.assert_allclose(normalize(jnp.array(x)), expected, rtol=2e-6)
        np.testing.assert_allclose(normalize(jnp.array(x + 5)), expected, rtol=2e-6)
        self.assertFalse(
            np.allclose(normalize(jnp.array(x + np.array([0.0, -10.0]))), expected)
        )


if __name__ == "__main__":
    unittest.main()
