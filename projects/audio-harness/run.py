"""Run the connected reference lifecycle and save actual plots/artifacts/evidence."""
import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
import wave
from pathlib import Path
import sys
import numpy as np
import jax
import jax.numpy as jnp
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"jaxpathways-audio-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument("--implementation",default="solution")
parser.add_argument("--output",default=str(root/"outputs"))
args=parser.parse_args()
path=root/args.implementation/"model.py" if args.implementation in ("starter","solution") else Path(args.implementation)
spec=importlib.util.spec_from_file_location("audio",path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
train=m.make_dataset(41,96,"train");held=m.make_dataset(77,69,"held");shift=m.make_dataset(98,72,"shifted",True)
state=m.create_state(train,seed=3)
history=[]
for _ in range(40):
    state,trace=m.step(state,train);history.append(trace)
m.save_checkpoint(out/"checkpoint",state)
resumed=m.load_checkpoint(out/"checkpoint",train,state["config"])
for _ in range(40):
    state,expected=m.step(state,train);resumed,actual=m.step(resumed,train)
    assert expected==actual
    history.append(actual)
for key in ("W","b","vW","vb","key","order"):np.testing.assert_array_equal(state[key],resumed[key])
calibration=m.calibrate(resumed,train)
m.export_release(out/"release",resumed,calibration)
manifest,exports=m.load_release(out/"release")
held_windows=m.fixed_windows(held);shift_windows=m.fixed_windows(shift)
normal=m.evaluate(resumed,held);changed=m.evaluate(resumed,shift)
float_scores=np.asarray(m.policy_scores(resumed,jnp.asarray(held_windows),calibration,"fp32"))
precision={}
normalized=np.asarray((m.features(held_windows)-resumed["mean"])/resumed["std"])
for policy in ("fp32","w8a32","w8a8"):
    prediction=np.asarray(m.policy_scores(resumed,jnp.asarray(held_windows),calibration,policy))
    precision[policy]={"accuracy":float(np.mean(prediction.argmax(1)==held["labels"])),
                       "max_logit_error":float(np.max(np.abs(prediction-float_scores))),
                       "clipped_feature_values":int(np.sum(np.abs(normalized)>127*calibration["activation_scale"])) if policy=="w8a8" else 0,
                       "calibration_count":len(train["ids"])}
timings={policy:m.benchmark(manifest,exports,held_windows[:1],policy=policy,repeats=30)
         for policy in ("fp32","w8a32","w8a8")}
energy=np.sqrt(np.mean(held_windows**2,axis=1));baseline=np.where(energy<.04,0,1)
baseline_accuracy=float(np.mean(baseline==held["labels"]))
wrong=np.flatnonzero(float_scores.argmax(1)!=held["labels"])
example=int(wrong[0]) if len(wrong) else 1
signal=held_windows[example]
with wave.open(str(out/"error-example.wav"),"wb") as audio:
    audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(8000)
    audio.writeframes(np.rint(signal*32767).astype("<i2").tobytes())
window=np.hanning(128)
frames=np.stack([signal[start:start+128] for start in range(0,897,64)])
power=np.abs(np.fft.rfft(frames*window,axis=-1))**2/np.sum(window**2)
log_power=np.log1p(power)
time_axis=np.arange(1024)/8000
frame_center=(np.arange(15)*64+63.5)/8000
frequency=np.fft.rfftfreq(128,1/8000)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"svg.fonttype":"none"})
fig,axes=plt.subplots(2,1,figsize=(9,6.5),layout="constrained",sharex=True)
axes[0].plot(time_axis,signal,color="#6240ad",linewidth=1)
axes[0].set(ylabel="PCM amplitude",title=f"Held-out {held['ids'][example]}: true {m.CLASSES[int(held['labels'][example])]}, predicted {m.CLASSES[int(float_scores[example].argmax())]}")
image=axes[1].pcolormesh(frame_center,frequency,log_power.T,shading="nearest",cmap="magma")
axes[1].set(xlabel="seconds within the 1024-sample inference window",ylabel="frequency (Hz)",ylim=(0,2200))
fig.colorbar(image,ax=axes[1],label="log(1 + window-normalized power)")
fig.savefig(out/"waveform-spectrogram.png",dpi=150);fig.savefig(out/"waveform-spectrogram.svg");plt.close(fig)
fig,axes=plt.subplots(3,1,figsize=(8.5,12),layout="constrained")
axes[0].plot(np.arange(80),[t["loss"] for t in history],color="#6240ad")
axes[0].set(xlabel="completed updates before this batch loss",ylabel="mean training cross-entropy (nats)",title="Augmented minibatch objective")
conf=np.asarray(normal["confusion"]);im=axes[1].imshow(conf,cmap="Purples")
axes[1].set(xticks=[0,1,2],yticks=[0,1,2],xticklabels=m.CLASSES,yticklabels=m.CLASSES,xlabel="predicted class",ylabel="true class",title="Fixed held-out recordings (69)")
for (i,j),value in np.ndenumerate(conf):axes[1].text(j,i,str(value),ha="center",va="center",color="white" if value>12 else "#241c31")
axes[1].tick_params(axis="x",rotation=25)
names=["energy baseline","FP32","W8A8","shifted FP32"];values=[baseline_accuracy,normal["accuracy"],precision["w8a8"]["accuracy"],changed["accuracy"]]
bars=axes[2].bar(names,values,color=["#a08cbf","#6240ad","#087c83","#b05c32"])
axes[2].bar_label(bars,fmt="%.3f");axes[2].set(ylabel="fraction correct",ylim=(0,1.13),title="Separate declared evaluations")
axes[2].tick_params(axis="x",rotation=30)
fig.savefig(out/"training-evaluation.png",dpi=150);fig.savefig(out/"training-evaluation.svg");plt.close(fig)
report={"schema_version":1,"source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment":{"python":sys.version,"jax":jax.__version__,"numpy":np.__version__,"device":str(jax.devices()[0])},
        "task":"synthetic short background/low-tone/high-tone classification; not speech recognition",
        "preprocessing":m.PREPROCESS,"train_ids":train["ids"],"held_ids":held["ids"],
        "train_hash":m.dataset_hash(train),"held_hash":m.dataset_hash(held),
        "recovery":{"interrupted_step":40,"resumed_to_step":80,"same_process_trace_and_state_match":True,
                    "fresh_process_proof":"tests/check.py stage 2"},
        "training_loss":[t["loss"] for t in history],"held":normal,"shifted":changed,
        "energy_baseline_accuracy":baseline_accuracy,"precision":precision,"timings":timings,
        "plot_example":{"id":held["ids"][example],"true":int(held["labels"][example]),
                        "predicted":int(float_scores[example].argmax()),
                        "frequency_peak_hz":float(frequency[power.mean(axis=0).argmax()]),
                        "rms":float(np.sqrt(np.mean(signal**2))),
                        "training_high_tone_median_rms":float(np.median(np.sqrt(np.mean(m.fixed_windows(train)[train["labels"]==2]**2,axis=1)))),
                        "log_power":log_power.tolist(),"waveform":signal.tolist()},
        "artifacts":manifest["artifacts"],"artifact_bytes":{key:(out/"release"/item["file"]).stat().st_size for key,item in manifest["artifacts"].items()},
        "memory":{"fp32_weight_bytes":int(np.asarray(state["W"]).nbytes),"int8_weight_and_scale_bytes":int(calibration["qW"].nbytes+calibration["weight_scales"].nbytes),"peak_runtime_memory_measured":False},"limitations":["synthetic recordings","fixed short windows",
        "no real-data quality claim","no native integer acceleration claim","CPU only; no target edge device",
        "capture, buffering, file decoding and networking excluded from timing"]}
(out/"report.json").write_text(json.dumps(report,indent=2))
print(json.dumps({"held":normal,"shifted":changed,"precision":precision,"energy_baseline_accuracy":baseline_accuracy,
                  "example":{k:v for k,v in report["plot_example"].items() if k not in ("waveform","log_power")}},indent=2))
print("Wrote",out)
