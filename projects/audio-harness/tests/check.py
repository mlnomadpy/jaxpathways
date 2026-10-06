"""Independent cumulative checks for an actual waveform-to-export lifecycle."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil
import wave
import numpy as np
import jax
import jax.numpy as jnp

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument("--implementation",default="starter")
parser.add_argument("--stage",choices=["1","2","3","4","all"],default="all")
parser.add_argument("--resume-worker")
parser.add_argument("--worker-output")
args=parser.parse_args()
path=(ROOT/args.implementation/"model.py" if args.implementation in ("starter","solution") else Path(args.implementation)).resolve()
spec=importlib.util.spec_from_file_location("learner_audio",path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
stage=4 if args.stage=="all" else int(args.stage)

def rejects(call):
    try:call()
    except ValueError:return
    raise AssertionError("Expected explicit ValueError")

def snapshot(state):
    return {k:np.array(v,copy=True) for k,v in state.items() if hasattr(v,"shape")}

def compare_states(a,b):
    assert a["step"]==b["step"] and a["cursor"]==b["cursor"] and a["epoch"]==b["epoch"]
    for key in snapshot(a):np.testing.assert_array_equal(a[key],b[key])

train=m.make_dataset(41,96,"train")
if args.resume_worker:
    config=json.loads((Path(args.resume_worker)/"checkpoint.json").read_text())["config"]
    state=m.load_checkpoint(args.resume_worker,train,config)
    trace=[]
    for _ in range(5):
        state,item=m.step(state,train);trace.append(item)
    m.save_checkpoint(args.worker_output,state)
    (Path(args.worker_output)/"trace.json").write_text(json.dumps(trace))
    raise SystemExit(0)

held=m.make_dataset(77,69,"held")
shifted=m.make_dataset(98,72,"shifted",True)
assert not set(train["ids"])&set(held["ids"])
assert not set(train["groups"])&set(held["groups"])
# Independent NumPy framing/window/FFT reference, including silence and impulse.
t=np.arange(1024)/8000
waveforms=np.stack([np.zeros(1024),.5*np.sin(2*np.pi*500*t),np.eye(1,1024,64)[0]]).astype(np.float32)
window=np.hanning(128)
frames=np.stack([waveforms[:,offset:offset+128] for offset in range(0,897,64)],axis=1)
power=np.abs(np.fft.rfft(frames*window,axis=-1))**2/np.sum(window**2)
expected=np.log1p(power).mean(axis=1)
actual=np.asarray(m.features(waveforms))
assert actual.shape==(3,65)
np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
np.testing.assert_array_equal(actual[0],np.zeros(65))
assert actual[1].argmax()==8
for bad in [lambda:m.features(waveforms,sample_rate=16000),
            lambda:m.features(waveforms,channels=2),
            lambda:m.features(waveforms[:,:1000]),
            lambda:m.features(np.full((1,1024),np.nan)),
            lambda:m.features(waveforms,version="wrong")]:
    rejects(bad)
# Actual PCM WAV ingestion with provenance and group leakage checks.
with tempfile.TemporaryDirectory() as directory:
    folder=Path(directory);wav_path=folder/"recording.wav"
    pcm=np.rint(train["waveforms"][0]*32767).astype("<i2")
    with wave.open(str(wav_path),"wb") as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(8000);wav.writeframes(pcm.tobytes())
    row={"id":"own-recording","file":"recording.wav","label":0,"split":"train","group":"session-a",
         "license":"generated fixture, project teaching use","source":"local synthetic test",
         "sha256":hashlib.sha256(wav_path.read_bytes()).hexdigest()}
    manifest=folder/"clips.json";manifest.write_text(json.dumps([row]))
    loaded=m.load_wav_manifest(manifest,"train")
    np.testing.assert_allclose(loaded["waveforms"][0],pcm.astype(np.float32)/32768,atol=1e-7)
    manifest.write_text(json.dumps([row,dict(row,id="leak",split="test")]))
    rejects(lambda:m.load_wav_manifest(manifest,"train"))
print("PASS stage 1: waveform contracts, independent STFT, silence, frequency bins, actual WAV provenance and split isolation")

if stage>=2:
    state=m.create_state(train,seed=3)
    normalized=np.asarray((m.features(m.fixed_windows(train)[:5])-state["mean"])/state["std"])
    labels=train["labels"][:5]
    logits=normalized@np.asarray(state["W"])+np.asarray(state["b"])
    probabilities=np.exp(logits-logits.max(1,keepdims=True));probabilities/=probabilities.sum(1,keepdims=True)
    delta=(probabilities-np.eye(3)[labels])/len(labels)
    oracle_W=normalized.T@delta;oracle_b=delta.sum(0)
    auto=jax.grad(m.objective,argnums=(0,1))(state["W"],state["b"],jnp.asarray(normalized),jnp.asarray(labels))
    np.testing.assert_allclose(auto[0],oracle_W,atol=2e-6,rtol=2e-5)
    np.testing.assert_allclose(auto[1],oracle_b,atol=2e-6,rtol=2e-5)
    start_loss=m.evaluate(state,held)["loss"]
    for _ in range(7):state,_=m.step(state,train)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint=Path(directory)/"interrupted";fresh=Path(directory)/"fresh-process"
        m.save_checkpoint(checkpoint,state)
        expected=state;trace=[]
        for _ in range(5):
            expected,item=m.step(expected,train);trace.append(item)
        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--implementation",str(path),
                               "--resume-worker",str(checkpoint),"--worker-output",str(fresh)],
                              capture_output=True,text=True,timeout=90)
        assert result.returncode==0,result.stdout+result.stderr
        restored=m.load_checkpoint(fresh,train,state["config"])
        compare_states(expected,restored)
        replay_trace=json.loads((fresh/"trace.json").read_text())
        assert trace==replay_trace,"IDs, crops, gains, feature bytes and losses must replay across epoch boundary"
        changed=m.make_dataset(42,96,"train")
        rejects(lambda:m.load_checkpoint(checkpoint,changed,state["config"]))
        rejects(lambda:m.load_checkpoint(checkpoint,train,dict(state["config"],rate=.1)))
    history=[]
    for _ in range(73):
        state,item=m.step(state,train);history.append(item["loss"])
    report=m.evaluate(state,held)
    assert report["accuracy"]>=.95 and report["loss"]<start_loss*.2
    frozen=snapshot(state)
    uneven=m.evaluate(state,held,batch_size=11);whole=m.evaluate(state,held,batch_size=len(held["ids"]))
    np.testing.assert_allclose(uneven["loss"],whole["loss"],rtol=1e-5,atol=1e-6)
    assert uneven["correct"]==whole["correct"] and uneven["confusion"]==whole["confusion"]
    for k,v in frozen.items():np.testing.assert_array_equal(state[k],v)
    changed_state=m.create_state(train,seed=9)
    for _ in range(80):changed_state,_=m.step(changed_state,train)
    changed_report=m.evaluate(changed_state,held)
    shift_report=m.evaluate(state,shifted)
    assert changed_report["accuracy"]>=.95 and shift_report["accuracy"]>=.85
    # Equal-energy tones defeat an energy-only classifier that assigns every audible clip to low-tone.
    held_windows=m.fixed_windows(held);energy=np.sqrt(np.mean(held_windows**2,axis=1))
    baseline=np.where(energy<.04,0,1)
    baseline_accuracy=float(np.mean(baseline==held["labels"]))
    assert report["accuracy"]>baseline_accuracy+.2
    print("PASS stage 2: gradient oracle, full fresh-process recovery, held-out aggregation, unchanged evaluation state and changed seed/data")
    print("Measured held/shifted/changed-seed/baseline accuracies:",report["accuracy"],shift_report["accuracy"],changed_report["accuracy"],baseline_accuracy)

if stage>=3:
    calibration=m.calibrate(state,train)
    rejects(lambda:m.calibrate(state,held))
    windows=m.fixed_windows(held)
    normalized=np.asarray((m.features(windows)-state["mean"])/state["std"])
    q=np.asarray(calibration["qW"]);s=np.asarray(calibration["weight_scales"]);a=calibration["activation_scale"]
    qa=np.clip(np.rint(normalized/a),-127,127).astype(np.int64)
    integer_oracle=(qa@q.astype(np.int64)).astype(np.float32)*(a*s)+np.asarray(state["b"])
    actual=np.asarray(m.policy_scores(state,jnp.asarray(windows),calibration,"w8a8"))
    np.testing.assert_allclose(actual,integer_oracle,rtol=2e-5,atol=2e-5)
    quant_accuracy=float(np.mean(actual.argmax(1)==held["labels"]))
    assert quant_accuracy>=report["accuracy"]-.03
    weight_rounding=np.abs(q.astype(np.float32)*s-np.asarray(state["W"]))
    assert np.all(weight_rounding<=s/2+1e-6)
    with tempfile.TemporaryDirectory() as directory:
        release=Path(directory)/"release"
        manifest=m.export_release(release,state,calibration)
        loaded,artifacts=m.load_release(release)
        backup=Path(directory)/"known-good-release";shutil.copytree(release,backup)
        assert len(artifacts)==6 and loaded["preprocessing"]["sample_rate"]==8000
        for policy in ("fp32","w8a32","w8a8"):
            for batch in (1,4):
                raw=windows[:batch]
                expected=np.asarray(m.policy_scores(state,jnp.asarray(raw),calibration,policy))
                got=m.infer_release(loaded,artifacts,raw,policy=policy)
                np.testing.assert_allclose(got["logits"],expected,rtol=3e-5,atol=3e-5)
        item=next(iter(manifest["artifacts"].values()))
        artifact=release/item["file"];original=artifact.read_bytes();artifact.write_bytes(original+b"corrupt")
        rejects(lambda:m.load_release(release))
        backup_manifest,backup_artifacts=m.load_release(backup)
        rollback=m.infer_release(backup_manifest,backup_artifacts,windows[:1],policy="fp32")
        np.testing.assert_allclose(rollback["logits"],np.asarray(m.policy_scores(state,jnp.asarray(windows[:1]),calibration,"fp32")),atol=3e-5,rtol=3e-5)
        artifact.write_bytes(original)
        # A new process loads full exported waveform preprocessing and classifier.
        input_file=Path(directory)/"input.npy";output_file=Path(directory)/"output.npy"
        np.save(input_file,windows[:4])
        program="import importlib.util,numpy as np,sys; s=importlib.util.spec_from_file_location('m',sys.argv[1]); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);a,b=m.load_release(sys.argv[2]);np.save(sys.argv[4],m.infer_release(a,b,np.load(sys.argv[3]),policy='w8a8')['logits'])"
        result=subprocess.run([sys.executable,"-c",program,str(path),str(release),str(input_file),str(output_file)],capture_output=True,text=True,timeout=90)
        assert result.returncode==0,result.stdout+result.stderr
        np.testing.assert_allclose(np.load(output_file),actual[:4],rtol=3e-5,atol=3e-5)
        print("PASS stage 3: training-only calibration, integer oracle, six complete exports, fresh-process reload and corruption rejection")
        print("Measured held-out W8A8 accuracy:",quant_accuracy)
        if stage>=4:
            for batch in (1,4):
                timing=m.benchmark(loaded,artifacts,windows[:batch],policy="w8a8",repeats=12)
                assert len(timing["samples_ms"])==12 and min(timing["samples_ms"])>0
                np.testing.assert_allclose(timing["p95_ms"],np.percentile(timing["samples_ms"],95))
                np.testing.assert_allclose(timing["examples_per_second"],1000*batch/timing["p50_ms"])
                print("Actual CPU timing:",batch,timing["p50_ms"],timing["p95_ms"],"milliseconds")
            for bad in [lambda:m.infer_release(loaded,artifacts,windows[:1],sample_rate=16000),
                        lambda:m.infer_release(loaded,artifacts,windows[:1],channels=2),
                        lambda:m.infer_release(loaded,artifacts,windows[:2]),
                        lambda:m.infer_release(loaded,artifacts,windows[:1],version="future"),
                        lambda:m.infer_release(loaded,artifacts,np.full((1,1024),1.1))]:
                rejects(bad)
            # Silence is finite and maps to the taught background class.
            silent=m.infer_release(loaded,artifacts,np.zeros((1,1024),np.float32),policy="fp32")
            assert np.isfinite(silent["logits"]).all() and silent["class_ids"][0]==0
            print("PASS stage 4: measured full preprocessing/inference boundary, malformed request drills and silence behavior")
print("All requested audio stages passed. Synthetic sound classification and CPU evidence only.")
