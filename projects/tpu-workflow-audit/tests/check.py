import argparse, importlib.util, json, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(w):
    p = ROOT / w / "model.py"
    s = importlib.util.spec_from_file_location("m", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def stage1(m):
    r = m.workflow_report(["w0", "w1", "w2"], ["w0", "w2"], [5, 5, 5])
    assert r["missing"] == ["w1"] and not r["publishable"]
    ok = m.workflow_report(["w0", "w1"], ["w0", "w1"], [7, 7])
    assert ok["publishable"] and ok["common_step"] == 7


def stage2(m):
    assert not m.workflow_report(["w0", "w1"], ["w0", "w1"], [7, 6])["publishable"]
    for workers, done, steps in [
        (["w0", "w0"], ["w0", "w0"], [1, 1]),
        (["w0", "w1"], ["w0", "w0"], [1, 1]),
        (["w0"], ["ghost"], [1]),
        (["w0"], ["w0"], [-1]),
        (["w0"], ["w0"], [1.5]),
        (["w0"], ["w0"], [float("nan")]),
        (["w0"], ["w0"], [float("inf")]),
        (["w0"], ["w0"], [True]),
    ]:
        try:
            m.workflow_report(workers, done, steps)
            raise AssertionError("invalid run plan accepted")
        except ValueError:
            pass


def stage3(m, output=None):
    r = m.workflow_report(["a", "b"], ["a", "b"], [9, 9])
    e = {
        **r,
        "scope": "simulation of receipt gating; no TPU VM or multi-host process was run",
    }
    p = (
        Path(output) if output else Path(tempfile.gettempdir())
    ) / "tpu-workflow-audit-evidence.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(e, sort_keys=True))
    assert json.loads(p.read_text()) == e


def run(m, n, output=None):
    f = {"1": stage1, "2": stage2, "3": stage3}
    [
        (stage3(m, output) if k == "3" else f[k](m))
        for k in (("1", "2", "3") if n in ("4", "all") else (n,))
    ]
    print(
        "PASS tpu-workflow-audit stage "
        + n
        + ": simulated receipt gate, duplicate/negative validation, saved scope evidence"
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
        with tempfile.TemporaryDirectory(prefix="tpu-workflow-audit-") as td:
            run(load(a.implementation), a.stage, td)
