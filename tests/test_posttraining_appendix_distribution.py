"""Focused positive and tamper checks for the saved-decoder EPUB figure."""

import ast
import base64
import re
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/check-distribution.py"
NAMESPACE = "http://www.w3.org/1999/xhtml"
SVG = b'<svg xmlns="http://www.w3.org/2000/svg"><text>Mean reward improvement</text><text>Primary policy loss \xc2\xb7 epoch 2</text></svg>'
ALT = "Final figure alt text."


def load_checker():
    tree = ast.parse(CHECKER.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "check_posttraining_appendix_figure"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"base64": base64}
    exec(compile(module, str(CHECKER), "exec"), namespace)
    return namespace["check_posttraining_appendix_figure"]


CHECK_FIGURE = load_checker()


def load_math_checker():
    tree = ast.parse(CHECKER.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "check_math_annotations"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"re": re}
    exec(compile(module, str(CHECKER), "exec"), namespace)
    return namespace["check_math_annotations"]


CHECK_MATH = load_math_checker()


def fixture(*, images=1, caption_last=True, image_src=None):
    src = image_src or "data:image/svg+xml;base64," + base64.b64encode(SVG).decode()
    image_nodes = "".join(f'<img src="{src}" alt="{ALT}"/>' for _ in range(images))
    caption = "<figcaption><strong>Figure 1.</strong> strict positive improvement floor therefore fails</figcaption>"
    extra = "<p>trailing</p>" if not caption_last else ""
    order = caption if caption_last else caption + extra
    markup = (
        f'<section xmlns="{NAMESPACE}"><figure class="print-visual-intro">'
        "<h3>Recorded v5 result</h3><div>"
        + image_nodes
        + "</div>"
        + order
        + "</figure></section>"
    )
    return ET.fromstring(markup)


class AppendixFigureDistributionTest(unittest.TestCase):
    def test_exact_embedded_figure_passes(self):
        CHECK_FIGURE(fixture(), SVG, ALT)

    def test_missing_duplicate_or_external_image_fails(self):
        for bridge in (
            fixture(images=0),
            fixture(images=2),
            fixture(image_src="/figure.svg"),
        ):
            with self.subTest(bridge=bridge):
                with self.assertRaises(AssertionError):
                    CHECK_FIGURE(bridge, SVG, ALT)

    def test_tampered_svg_alt_or_caption_order_fails(self):
        cases = (
            (fixture(), b"<svg>tampered</svg>", ALT),
            (fixture(), SVG, "wrong alt"),
            (fixture(caption_last=False), SVG, ALT),
        )
        for bridge, svg, alt in cases:
            with self.subTest(svg=svg, alt=alt):
                with self.assertRaises(AssertionError):
                    CHECK_FIGURE(bridge, svg, alt)

    def test_math_comparison_trims_outer_whitespace_but_preserves_equation(self):
        source = r"\[  \delta_t = r_t + V_t  \]"
        annotation = r"\delta_t = r_t + V_t"
        self.assertEqual(CHECK_MATH(source, [f"\n{annotation}\n"], "fixture"), 1)
        with self.assertRaisesRegex(AssertionError, "equation missing"):
            CHECK_MATH(source, [r"\delta_t = r_t - V_t"], "fixture")
        with self.assertRaisesRegex(AssertionError, "equation missing"):
            CHECK_MATH(source, [], "fixture")


if __name__ == "__main__":
    unittest.main()
