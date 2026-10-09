"""Public CPU release checks with independent numerical and queue references."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import numpy as np
import jax
import jax.numpy as jnp

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument("--implementation",default="starter")
parser.add_argument("--stage",choices=["1","2","3","all"],default="all")
args=parser.parse_args()
stage=3 if args.stage=="all" else int(args.stage)
path=root/args.implementation/"model.py" if args.implementation in ("starter","solution") else Path(args.implementation)
spec=importlib.util.spec_from_file_location("learner",path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
def rejects(call):
    try:call()
    except ValueError:return
    raise AssertionError("Expected ValueError for invalid input")

rng=np.random.default_rng(51)
X=rng.normal(size=(80,3)).astype(np.float32)
X[:,-1]=1.
teacher=np.array([1.,-.5,.2],np.float32)
targets=1/(1+np.exp(-(X@teacher)))
initial=jnp.zeros(3)
trained,history=m.train(initial,jnp.array(X),jnp.array(targets))
expected_gradient=X.astype(np.float64).T@(.5-targets.astype(np.float64))/len(X)
np.testing.assert_allclose(jax.grad(m.objective)(initial,jnp.array(X),jnp.array(targets)),expected_gradient,rtol=1e-5,atol=1e-6)
np.testing.assert_allclose(history[0],np.log(2),rtol=1e-6)
assert history[-1]<history[0]-.05
assert np.isfinite(history).all() and history.shape==(250,)
zero,empty=m.train(trained,jnp.array(X),jnp.array(targets),steps=0)
np.testing.assert_array_equal(zero,trained)
assert empty.shape==(0,)
# Saved and reloaded base is actually trained; adaptation is a distinct target.
with tempfile.TemporaryDirectory() as checkpoint_dir:
    checkpoint=Path(checkpoint_dir)/"base.npy"
    np.save(checkpoint,np.asarray(trained))
    base_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    restored=jnp.asarray(np.load(checkpoint,allow_pickle=False))
    new_target=1/(1+np.exp(-(X@np.array([-.4,1.2,-.1]))))
    adapted,adapt_history=m.train(restored,jnp.asarray(X),jnp.asarray(new_target))
    assert adapt_history[-1]<adapt_history[0]-.05
    np.testing.assert_array_equal(restored,trained)
rejects(lambda:m.train(initial,jnp.asarray(X),jnp.ones((80,1))))
print("PASS stage 1: independent objective gradient, genuine pretraining, checkpoint reload and changed-task adaptation")

if stage>=2:
    for bits in (4,8):
        W=np.array([[.1,0.,1.7],[-.31,0.,-.8],[.92,0.,.15]],np.float32)
        codes,scales=m.quantize(W,bits)
        assert codes.dtype==np.int8 and scales.shape==(1,3)
        assert np.isfinite(scales).all() and np.all(codes[:,1]==0)
        assert np.max(np.abs(codes))<=2**(bits-1)-1
        np.testing.assert_array_less(np.abs(codes.astype(np.float32)*scales-W),np.broadcast_to(scales/2+1e-6,W.shape))
    provenance={"source_data_sha256":hashlib.sha256(X.tobytes()+targets.tobytes()).hexdigest(),
                "adaptation_data_sha256":hashlib.sha256(X.tobytes()+new_target.tobytes()).hexdigest(),"training_steps":500,
                "objective":"source soft BCE then target soft BCE","base_checkpoint_sha256":base_hash}
    with tempfile.TemporaryDirectory() as folder:
        manifest=m.prepare(folder,adapted,X,provenance)
        loaded,exports=m.load(folder)
        assert loaded["provenance"]==provenance
        assert loaded["calibration_sha256"]==hashlib.sha256(X.tobytes()).hexdigest()
        for policy in ("fp32","w8a32","w8a8"):
            assert policy in loaded["policies"]
        assert loaded["policies"]["w8a8"]["accumulator"]=="int32"
        q,s=m.quantize(np.asarray(adapted)[:,None],8)
        for batch in (1,8):
            probe=rng.normal(0,.4,(batch,3)).astype(np.float32)
            probe[:,-1]=1.
            float_expected=probe@np.asarray(adapted)
            for policy in ("fp32","w8a32","w8a8"):
                actual=np.asarray(exports[f"{policy}:{batch}"].call(jnp.asarray(probe)))
                if policy=="fp32":expected=float_expected
                elif policy=="w8a32":expected=(probe@(q.astype(np.float32)*s)).ravel()
                else:
                    a=loaded["activation_scale"]
                    aq=np.clip(np.rint(probe/a),-127,127).astype(np.int64)
                    expected=(aq@q.astype(np.int64)).astype(np.float32).ravel()*a*float(s[0,0])
                np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
                assert np.max(np.abs(actual-float_expected))<.04
        # Corrupt the file actually loaded and require rejection.
        item=next(iter(loaded["artifacts"].values()))
        damaged=Path(folder)/item["file"]
        saved=damaged.read_bytes()
        damaged.write_bytes(saved+b"bad")
        rejects(lambda:m.load(folder))
        damaged.write_bytes(saved)
        print("PASS stage 2: six exported round trips, independent precision arithmetic, calibration provenance and corruption rejection")
        if stage>=3:
            reports=[]
            for batch in (1,8):
                probe=rng.normal(0,.4,(batch,3)).astype(np.float32)
                probe[:,-1]=1.
                payload=json.dumps({"features":probe.tolist()})
                response=json.loads(m.request(payload,loaded,exports))
                np.testing.assert_allclose(response["scores"],probe@np.asarray(adapted),atol=2e-6)
                report=m.benchmark(payload,loaded,exports,repeats=12)
                assert report["network_included"] is False and report["device"]=="CPU"
                assert len(report["samples_ms"])==12 and min(report["samples_ms"])>0
                np.testing.assert_allclose(report["p50_ms"],np.percentile(report["samples_ms"],50))
                np.testing.assert_allclose(report["examples_per_second"],1000*batch/report["p50_ms"])
                reports.append(report)
            for payload in ('{}','{"features":[[1,2]]}','{"features":[["wrong",0,1]]}','{"features":[[NaN,0,1]]}'):
                rejects(lambda payload=payload:m.request(payload,loaded,exports))
            shifted=json.loads(m.request(json.dumps({"features":[[100.,-100.,1.]]}),loaded,exports,"w8a8"))
            assert shifted["clipped_activations"]==2
            np.testing.assert_allclose(m.simulate([0,1,4],2),[2,3,2])
            np.testing.assert_allclose(m.simulate([0,0,0],2),[2,4,6])
            rejects(lambda:m.simulate([1,0],2))
            print("PASS stage 3: real JSON request parity, timed batch requests, clipping, bad inputs and independent queue timelines")
            print("Actual CPU request p50/p95 milliseconds:",[(r["batch"],r["p50_ms"],r["p95_ms"]) for r in reports])
print("All requested stages passed. No network, autoscaler or edge device was measured.")
