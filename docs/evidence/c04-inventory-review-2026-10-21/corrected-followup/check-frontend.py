"""Construct the declared analysis window and mel bank without candidate data."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import audio

# Derive every analysis constant from the recorded physical pilot policy.
rate,window_size,hop,nfft,bands=16000,400,160,512,80
window=np.sin(np.pi*np.arange(window_size)/window_size)**2
low=2595*np.log10(1+20/700)
high=2595*np.log10(1+7600/700)
edges=700*(10**(np.linspace(low,high,bands+2)/2595)-1)
frequencies=np.arange(nfft//2+1)*rate/nfft
bank=np.zeros((nfft//2+1,bands))
for band in range(bands):
    a,b,c=edges[band:band+3]
    for k,hz in enumerate(frequencies):
        if a<=hz<=b:bank[k,band]=(hz-a)/(b-a)
        elif b<hz<=c:bank[k,band]=(c-hz)/(c-b)
    bank[:,band]*=2/(c-a)
np.testing.assert_allclose(np.asarray(audio.WINDOW),window,atol=4e-8,rtol=2e-6)
np.testing.assert_allclose(np.asarray(audio.MEL_MATRIX),bank,atol=1e-8,rtol=2e-6)

def oracle(x,lengths):
    count=1+(x.shape[1]-window_size)//hop
    values=np.zeros((x.shape[0],count,bands))
    valid=np.maximum(0,1+(lengths-window_size)//hop)
    for row,length in enumerate(lengths):
        for frame in range(valid[row]):
            start=frame*hop
            samples=np.asarray(x[row,start:start+window_size],dtype=np.float64)*window
            spectrum=np.fft.rfft(samples,n=nfft)
            power=(spectrum.real**2+spectrum.imag**2)/(window@window)
            values[row,frame]=np.log(np.maximum(power@bank,1e-5))
    return values,valid

rng=np.random.default_rng(121)
record={'scope':'Independent periodic-Hann/HTK-spaced, Hz-area-normalized triangular bank construction and uncentered log-mel forward/reduction on synthetic 16 kHz inputs. No raw codec or polyphase-resampler acceptance, fitting statistics, corpus execution or task quality.', 'audio_source_sha256':hashlib.sha256((ROOT/'source/audio.py').read_bytes()).hexdigest(),'cases':[]}
for width in (400,559,560,1600,1601):
    lengths=np.array([width,max(0,width-179),0],dtype=np.int32)
    x=rng.normal(size=(3,width)).astype('float32')*.05
    # Nonzero values beyond each row's valid length must not affect valid frames.
    x[np.arange(width)[None,:]>=lengths[:,None]]=19
    expected,valid=oracle(x,lengths)
    actual,observed=audio.log_mel(x,lengths)
    np.testing.assert_array_equal(np.asarray(observed),valid)
    np.testing.assert_allclose(np.asarray(actual),expected,atol=3e-5,rtol=3e-6)
    record['cases'].append({'padded_samples':width,'valid_samples':lengths.tolist(),'valid_frames':valid.tolist(),'max_log_mel_absolute_error':float(np.max(np.abs(np.asarray(actual)-expected)))})
time=np.arange(1600)/rate
sine=(.05*np.sin(2*np.pi*440*time)).astype('float32')
a,_=audio.log_mel(sine[None,:],np.array([1600]))
b,_=audio.log_mel((2*sine)[None,:],np.array([1600]))
expected,_=oracle(sine[None,:],np.array([1600]))
active=expected>np.log(1e-5)+2
np.testing.assert_allclose((np.asarray(b)-np.asarray(a))[active],np.log(4),atol=3e-5)
record['amplitude_control']={'amplitude_ratio':2,'expected_log_power_shift':float(np.log(4)),'checked_cells':int(active.sum())}
original=audio.WINDOW
try:
    # A rectangular-window mutation must fail the complete signal pipeline,
    # even if FFT/reference code were otherwise unchanged.
    import jax.numpy as jnp
    audio.WINDOW=jnp.ones((400,))
    broken,_=audio.log_mel(sine[None,:],np.array([1600]))
    delta=float(np.max(np.abs(np.asarray(broken)-expected)))
    assert delta>.1,delta
    record['negative_control']={'mutation':'Rectangular window, in memory only','max_log_mel_absolute_delta':delta,'rejected':True}
finally:audio.WINDOW=original
record['status']='pass'
record['checker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(ROOT/'frontend-result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
