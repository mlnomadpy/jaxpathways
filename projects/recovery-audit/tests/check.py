import argparse, importlib.util, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(w):
    p = ROOT / w / "model.py"
    s = importlib.util.spec_from_file_location("m", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


CFG = {"lr": 0.05, "momentum": 0.8, "noise_scale": 0.01}


def initial():
    return {
        "weight": np.array([1.0, -1.0], np.float32),
        "velocity": np.array([0.2, -0.3], np.float32),
        "rng_key": np.array([31, 47], np.uint32),
        "cursor": np.int64(0),
    }


def stage1(m):
    a = m.train_step(initial(), np.array([0.0, 0.0], np.float32), CFG)
    assert int(a["cursor"]) == 1 and not np.array_equal(
        a["weight"], initial()["weight"]
    )


def stage2(m):
    state = initial()
    full = m.train_step(
        m.train_step(state, np.array([0.0, 0.0]), CFG), np.array([0.3, -0.2]), CFG
    )
    split = m.train_step(state, np.array([0.0, 0.0]), CFG)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.npz"
        m.save_checkpoint(p, 1, CFG, split)
        resumed = m.load_checkpoint(p, CFG)
        continued = m.train_step(resumed["arrays"], np.array([0.3, -0.2]), CFG)
        for key in full:
            np.testing.assert_array_equal(full[key], continued[key])


def stage3(m, output=None):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.npz"
        m.save_checkpoint(p, 1, CFG, initial())
        try:
            m.load_checkpoint(p, {**CFG, "lr": 0.2})
            raise AssertionError("config mismatch accepted")
        except ValueError:
            pass
        with np.load(p) as z:
            d = {k: z[k] for k in z.files}
        d["weight"] = np.array([9.0, -1.0], np.float32)
        np.savez(p, **d)
        try:
            m.load_checkpoint(p, CFG)
            raise AssertionError("corruption accepted")
        except ValueError:
            pass
    report = {
        "config_mismatch_rejected": True,
        "corruption_rejected": True,
        "checkpoint_integrity": "sha256",
    }
    target = (
        Path(output) if output else Path(tempfile.gettempdir())
    ) / "recovery-audit-evidence.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(__import__("json").dumps(report))
    assert __import__("json").loads(target.read_text()) == report


def run(m, n, output=None):
    fs = {"1": stage1, "2": stage2, "3": stage3}
    keys = ("1", "2", "3") if n in ("4", "all") else (n,)
    for k in keys:
        (stage3(m, output) if k == "3" else fs[k](m))
    print(
        "PASS recovery-audit stage "
        + n
        + ": optimizer/RNG continuation and checkpoint failure handling"
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
        with tempfile.TemporaryDirectory(prefix="recovery-audit-") as td:
            run(load(a.implementation), a.stage, td)
