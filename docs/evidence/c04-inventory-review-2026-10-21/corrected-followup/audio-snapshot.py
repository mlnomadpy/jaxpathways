"""Raw waveform decoding and a time-preserving, length-aware log-mel frontend.

The constants are a pilot proposal pending the data/recipe review. WAV/FLAC
decoding always reports the original metadata and never relabels a sample rate.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import jax.numpy as jnp
from scipy.signal import resample_poly
import soundfile as sf

from config import (FMAX_HZ, FMIN_HZ, HOP_LENGTH, LOG_FLOOR, MAX_SAMPLES,
                    N_FFT, N_MELS, SAMPLE_RATE, WIN_LENGTH)

SUPPORTED_RATES = (8_000, 11_025, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000)


def decode_audio(path: str | Path) -> tuple[np.ndarray, dict]:
    """Decode WAV/FLAC, average channels, anti-alias resample to 16 kHz mono."""
    path = Path(path)
    if path.suffix.lower() not in {".wav", ".flac"}:
        raise ValueError("only WAV and FLAC are supported")
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate not in SUPPORTED_RATES:
        raise ValueError(f"unsupported sample rate: {rate}")
    if data.shape[0] == 0 or not np.isfinite(data).all():
        raise ValueError("audio must contain finite samples")
    peak = float(np.max(np.abs(data)))
    if peak > 1.0001:
        raise ValueError("decoded floating PCM exceeds [-1, 1]")
    mono = data.mean(axis=1, dtype=np.float32)
    if rate != SAMPLE_RATE:
        from math import gcd
        divisor = gcd(rate, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // divisor, rate // divisor).astype(np.float32)
    if len(mono) == 0 or len(mono) > MAX_SAMPLES:
        raise ValueError("decoded duration must be positive and at most 8 seconds")
    meta = {"path": str(path), "source_rate": int(rate), "source_channels": int(data.shape[1]),
            "source_frames": int(data.shape[0]), "source_peak": peak,
            "sample_rate": SAMPLE_RATE, "samples": int(len(mono)),
            "seconds": len(mono) / SAMPLE_RATE, "resampler": "scipy.signal.resample_poly"}
    return np.ascontiguousarray(mono, dtype=np.float32), meta


def frame_count(length: int | np.ndarray) -> int | np.ndarray:
    """Unpadded STFT frame count for valid sample lengths."""
    length = jnp.asarray(length, dtype=jnp.int32)
    return jnp.maximum(0, 1 + jnp.floor_divide(length - WIN_LENGTH, HOP_LENGTH))


def _mel_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_matrix() -> np.ndarray:
    """Slaney-style triangular weights defined by equally spaced HTK-mel points."""
    hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)
    points = _mel_hz(np.linspace(hz_to_mel(FMIN_HZ), hz_to_mel(FMAX_HZ), N_MELS + 2))
    bins = np.arange(N_FFT // 2 + 1) * (SAMPLE_RATE / N_FFT)
    weights = np.zeros((N_FFT // 2 + 1, N_MELS), dtype=np.float32)
    for m in range(N_MELS):
        left, center, right = points[m:m + 3]
        weights[:, m] = np.maximum(0.0, np.minimum((bins - left) / (center - left),
                                                    (right - bins) / (right - center)))
    # Normalize each filter to unit area in Hz, then use power spectra.
    weights *= (2.0 / (points[2:] - points[:-2]))[None, :]
    return weights


MEL_MATRIX = jnp.asarray(mel_matrix())
WINDOW = jnp.asarray(np.hanning(WIN_LENGTH + 1)[:-1].astype(np.float32))


def log_mel(waveforms, sample_lengths):
    """Compute uncentered periodic-Hann log-mel features and valid frame counts.

    Inputs are right-padded (batch, samples) float32 at 16 kHz. Padding is
    excluded before framing; output tail frames are zero and lengths are exact.
    """
    x = jnp.asarray(waveforms, dtype=jnp.float32)
    lengths = jnp.asarray(sample_lengths, dtype=jnp.int32)
    if x.ndim != 2 or lengths.shape != (x.shape[0],):
        raise ValueError("expected waveforms (batch,samples) and one valid length per row")
    if x.shape[1] < WIN_LENGTH:
        raise ValueError("padded waveform width must include one analysis window")
    frames = jax_frame(x)
    windowed = frames * WINDOW[None, None, :]
    spectrum = jnp.fft.rfft(windowed, n=N_FFT, axis=-1)
    power = (jnp.real(spectrum) ** 2 + jnp.imag(spectrum) ** 2) / jnp.sum(WINDOW ** 2)
    mel = jnp.einsum("btf,fm->btm", power, MEL_MATRIX)
    result = jnp.log(jnp.maximum(mel, LOG_FLOOR))
    valid = frame_count(lengths)
    mask = jnp.arange(result.shape[1])[None, :] < valid[:, None]
    return jnp.where(mask[:, :, None], result, 0.0), valid


def jax_frame(x):
    # lax.conv_general_dilated_patches has awkward kernel ordering; explicit
    # gather makes the no-center-padding boundary visible and testable.
    starts = jnp.arange(0, x.shape[1] - WIN_LENGTH + 1, HOP_LENGTH)
    offsets = jnp.arange(WIN_LENGTH)
    return x[:, starts[:, None] + offsets[None, :]]


def resample_length(length: int, source_rate: int, target_rate: int = SAMPLE_RATE) -> int:
    """Expected scipy resample_poly output length, using its ceil convention."""
    from math import gcd
    g = gcd(source_rate, target_rate)
    return (length * (target_rate // g) + (source_rate // g) - 1) // (source_rate // g)
