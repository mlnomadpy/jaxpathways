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
    x = np.array([[1.0, 3.0], [5.0, 7.0], [100.0, 200.0]], np.float32)
    r = m.masked_moments(x, [1, 1, 0])
    np.testing.assert_allclose(r["mean"], [3, 5])
    np.testing.assert_allclose(r["variance"], [4, 4])
    assert r["count"] == 2


def stage2(m):
    x = np.array([[1.0, 3.0], [5.0, 7.0], [100.0, 200.0]], np.float32)
    r = m.masked_moments(x, [1, 0, 1])
    np.testing.assert_allclose(r["mean"], [50.5, 101.5])
    np.testing.assert_allclose(r["variance"], [2450.25, 9702.25])
    for bad, mask in [(x, [0, 0, 0]), (x[:2], [1, 0, 1])]:
        try:
            m.masked_moments(bad, mask)
            raise AssertionError("invalid batch accepted")
        except ValueError:
            pass


def stage3(m, output=None):
    r = m.masked_moments([[2.0, 4.0], [8.0, 10.0]], [1, 0])
    payload = {
        "count": r["count"],
        "mean": np.asarray(r["mean"]).tolist(),
        "variance": np.asarray(r["variance"]).tolist(),
    }
    p = (
        Path(output) if output else Path(tempfile.gettempdir())
    ) / "arrays-audit-evidence.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload))
    saved = json.loads(p.read_text())
    assert saved == {"count": 1, "mean": [2.0, 4.0], "variance": [0.0, 0.0]}


def run(m, n, output=None):
    f = {"1": stage1, "2": stage2, "3": stage3}
    [
        (stage3(m, output) if k == "3" else f[k](m))
        for k in (("1", "2", "3") if n in ("4", "all") else (n,))
    ]
    print(
        "PASS arrays-audit stage "
        + n
        + ": masked reductions, changed masks, and saved evidence"
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
        with tempfile.TemporaryDirectory(prefix="arrays-audit-") as td:
            run(load(a.implementation), a.stage, td)
