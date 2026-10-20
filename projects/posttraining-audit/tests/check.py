import argparse, importlib.util, json, tempfile
from pathlib import Path
import jax.numpy as jnp, numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(w):
    p = ROOT / w / "model.py"
    s = importlib.util.spec_from_file_location("m", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def inputs():
    return (
        np.array([[[3.0, 0.0], [0.0, 3.0], [2.0, 0.0]]], np.float32),
        np.array([[0, 1, 0]]),
        np.array([[False, True, False]]),
    )


def stage1(m):
    x, t, mask = inputs()
    r = m.posttraining_losses(x, t, mask, jnp.array([-0.7]), jnp.array([-1.0]))
    expected = -np.log(np.exp(3) / (np.exp(0) + np.exp(3)))
    np.testing.assert_allclose(float(r["sft_loss"]), expected, rtol=1e-5)
    assert (
        r["response_tokens"] == 1
        and float(r["preference_margin"][0]) > 0
        and float(r["dpo_loss"][0]) < np.log(2)
    )


def stage2(m):
    x, t, mask = inputs()
    a = m.posttraining_losses(x, t, mask, jnp.array([-0.7]), jnp.array([-1.0]))
    b = m.posttraining_losses(x, t, mask, jnp.array([-1.0]), jnp.array([-0.7]))
    assert float(b["preference_margin"][0]) < 0 and float(b["dpo_loss"][0]) > float(
        a["dpo_loss"][0]
    )
    c = m.posttraining_losses(
        x,
        t,
        mask,
        jnp.array([-0.7]),
        jnp.array([-1.0]),
        reference_chosen_logp=-0.4,
        reference_rejected_logp=-1.4,
    )
    np.testing.assert_allclose(np.asarray(c["preference_margin"]), [-0.7], atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(c["dpo_loss"]), [np.logaddexp(0.0, 0.07)], rtol=1e-6
    )
    try:
        m.posttraining_losses(
            x, t, np.zeros_like(mask), jnp.array([-0.7]), jnp.array([-1.0])
        )
        raise AssertionError("empty response mask accepted")
    except ValueError:
        pass


def stage3(m, output=None):
    x, t, mask = inputs()
    r = m.posttraining_losses(x, t, mask, jnp.array([-0.7]), jnp.array([-1.0]))
    report = {
        "response_tokens": r["response_tokens"],
        "sft_loss": float(r["sft_loss"]),
        "preference_margin": np.asarray(r["preference_margin"]).tolist(),
        "dpo_loss": np.asarray(r["dpo_loss"]).tolist(),
    }
    p = (
        Path(output) if output else Path(tempfile.gettempdir())
    ) / "posttraining-audit-evidence.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report))
    assert json.loads(p.read_text()) == report


def run(m, n, output=None):
    f = {"1": stage1, "2": stage2, "3": stage3}
    [
        (stage3(m, output) if k == "3" else f[k](m))
        for k in (("1", "2", "3") if n in ("4", "all") else (n,))
    ]
    print(
        "PASS posttraining-audit stage "
        + n
        + ": masked SFT, changed preference pairs, and saved slice evidence"
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
        with tempfile.TemporaryDirectory(prefix="posttraining-audit-") as td:
            run(load(a.implementation), a.stage, td)
