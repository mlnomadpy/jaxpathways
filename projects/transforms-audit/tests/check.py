import argparse, importlib.util, json, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(w):
    p = ROOT / w / "model.py"
    s = importlib.util.spec_from_file_location("m", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def stage1(m):
    r = m.transform_report()
    x = np.array([-1.2, -0.1, 0.7, 1.4])
    np.testing.assert_allclose(r["mapped"], np.sin(x) + x * x, rtol=1e-6)
    np.testing.assert_allclose(r["gradient"], np.cos(x) + 2 * x, rtol=1e-5)
    np.testing.assert_allclose(r["compiled"], r["loop"], rtol=1e-6)


def stage2(m):
    x = np.array([-0.3, 0.4, 2.1], np.float32)
    r = m.transform_report(x)
    np.testing.assert_allclose(r["mapped"], np.sin(x) + x * x, rtol=1e-6)
    np.testing.assert_allclose(r["extended"][-1], np.sin(2) + 4, rtol=1e-6)
    assert r["trace_count"] == 2


def stage3(m, output=None):
    r = m.transform_report()
    p = (
        Path(output) if output else Path(tempfile.gettempdir())
    ) / "transforms-audit.jaxpr.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(r["jaxpr"])
    assert "sin" in p.read_text() and "mul" in p.read_text()


def run(m, n, output=None):
    f = {"1": stage1, "2": stage2, "3": stage3}
    [
        (stage3(m, output) if k == "3" else f[k](m))
        for k in (("1", "2", "3") if n in ("4", "all") else (n,))
    ]
    print(
        "PASS transforms-audit stage "
        + n
        + ": equivalent transforms, changed shapes, and retained JAXPR"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--implementation", default="starter")
    p.add_argument("--evidence-dir")
    p.add_argument("--stage", choices=("1", "2", "3", "4", "all"), default="all")
    a = p.parse_args()
    if a.evidence_dir:
        Path(a.evidence_dir).mkdir(parents=True, exist_ok=True)
        run(load(a.implementation), a.stage, a.evidence_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="transforms-audit-") as td:
            run(load(a.implementation), a.stage, td)
