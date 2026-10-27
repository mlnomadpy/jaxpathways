"""The executed-figure renderer must accept actual ownership tables."""

import importlib.util
import os
from pathlib import Path
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jaxpathways-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = Path(__file__).resolve().parents[1] / "scripts/render-lesson-figures.py"
spec = importlib.util.spec_from_file_location("figure_renderer", path)
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


class OwnershipTables(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_actual_shape_and_byte_table_is_drawn(self):
        fig, ax = plt.subplots()
        renderer.draw_panel(
            ax,
            {
                "kind": "table",
                "headers": ["Array", "Shape", "Bytes"],
                "rows": [["W", "(4, 2)", 32], ["momentum", "(4, 2)", 32]],
            },
        )
        fig.canvas.draw()
        cells = list(ax.tables[0].get_celld().values())
        labels = [cell.get_text().get_text() for cell in cells]
        self.assertEqual(len(cells), 9)
        self.assertIn("momentum", labels)
        self.assertEqual(labels.count("32"), 2)

    def test_ragged_table_is_rejected(self):
        _, ax = plt.subplots()
        with self.assertRaises(AssertionError):
            renderer.draw_panel(
                ax, {"kind": "table", "headers": ["Array", "Bytes"], "rows": [["W"]]}
            )


if __name__ == "__main__":
    unittest.main()
