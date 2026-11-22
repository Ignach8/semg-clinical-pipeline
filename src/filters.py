"""SENIAM-compliant clinical surface EMG preprocessing filters."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt


class PreprocessedChannel(NamedTuple):
    """Preprocessed sEMG channel containing filtered signal and RMS envelope."""

    filtered: NDArray[np.float64]
    rms_envelope: NDArray[np.float64]


def _validate_signal(signal: ArrayLike) -> NDArray[np.float64]:
    arr: NDArray[np.float64] = np.asarray(signal, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D signal array, got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError("Signal array must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Signal contains non-finite values (NaN or Inf).")
    return arr


def remove_dc_offset(signal: ArrayLike) -> NDArray[np.float64]:
    """Remove mean DC baseline offset from an sEMG channel."""
    arr: NDArray[np.float64] = _validate_signal(signal)
    # Remove electrode-skin half-cell potential and ADC baseline bias prior to
    # IIR filtering to prevent edge transient ringing.
    return arr - float(np.mean(arr))


def apply_bandpass_filter(
    signal: ArrayLike,
    fs: float = 1000.0,
    lowcut: float = 20.0,
    highcut: float = 450.0,
    order: int = 4,
) -> NDArray[np.float64]:
    """Apply a zero-phase Butterworth bandpass filter per SENIAM guidelines."""
    arr: NDArray[np.float64] = _validate_signal(signal)
    if fs <= 0.0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}.")
    if order < 1:
        raise ValueError(f"Filter order must be >= 1, got {order}.")

    nyquist: float = 0.5 * fs
    if not (0.0 < lowcut < highcut < nyquist):
        raise ValueError(
            f"Cutoff frequencies must satisfy 0 < lowcut ({lowcut}) < "
            f"highcut ({highcut}) < Nyquist ({nyquist})."
        )

    # SENIAM guidelines specify a 20-450 Hz passband for surface EMG:
    # 20 Hz high-pass suppresses electrode motion artifacts and firing-rate
    # instability, while 450 Hz low-pass removes high-frequency instrumentation
    # noise below the 500 Hz Nyquist frequency at fs=1000 Hz.
    # Second-order sections (SOS) with forward-backward filtering ensure
    # numerical stability and zero phase distortion to preserve activation timing.
    sos: NDArray[np.float64] = butter(
        order,
        [lowcut, highcut],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    filtered: NDArray[np.float64] = sosfiltfilt(sos, arr)
    return np.asarray(filtered, dtype=np.float64)


def apply_notch_filter(
    signal: ArrayLike,
    fs: float = 1000.0,
    freq: float = 50.0,
    q: float = 30.0,
) -> NDArray[np.float64]:
    """Apply a zero-phase IIR notch filter to attenuate powerline interference."""
    arr: NDArray[np.float64] = _validate_signal(signal)
    if fs <= 0.0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}.")
    if q <= 0.0:
        raise ValueError(f"Quality factor q must be positive, got {q}.")

    nyquist: float = 0.5 * fs
    if not (0.0 < freq < nyquist):
        raise ValueError(
            f"Notch frequency must satisfy 0 < freq ({freq}) < Nyquist ({nyquist})."
        )

    # 50 Hz targets Chilean/European AC mains interference; a narrow high-Q
    # notch (Q=30, ~1.67 Hz -3 dB bandwidth) limits spectral loss within the
    # overlapping 50-150 Hz physiological sEMG power peak.
    b: NDArray[np.float64]
    a: NDArray[np.float64]
    b, a = iirnotch(w0=freq, Q=q, fs=fs)
    filtered: NDArray[np.float64] = filtfilt(b, a, arr)
    return np.asarray(filtered, dtype=np.float64)


def compute_rms_envelope(
    signal: ArrayLike,
    fs: float = 1000.0,
    window_ms: float = 100.0,
) -> NDArray[np.float64]:
    """Compute the moving root-mean-square (RMS) amplitude envelope."""
    arr: NDArray[np.float64] = _validate_signal(signal)
    if fs <= 0.0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}.")
    if window_ms <= 0.0:
        raise ValueError(f"Window duration must be positive, got {window_ms}.")

    window_samples: int = max(1, int(round(fs * window_ms / 1000.0)))

    # A 100 ms moving RMS window balances smooth amplitude estimation with
    # temporal responsiveness during sustained isometric contractions;
    # reflective padding prevents artificial envelope roll-off at signal edges.
    mean_square: NDArray[np.float64] = uniform_filter1d(
        np.square(arr),
        size=window_samples,
        mode="reflect",
    )
    return np.sqrt(np.maximum(mean_square, 0.0))


def preprocess_channel(
    signal: ArrayLike,
    fs: float = 1000.0,
    *,
    lowcut: float = 20.0,
    highcut: float = 450.0,
    order: int = 4,
    notch_freq: float = 50.0,
    notch_q: float = 30.0,
    window_ms: float = 100.0,
) -> PreprocessedChannel:
    """Run SENIAM preprocessing (DC removal, bandpass, notch) and RMS envelope."""
    dc_removed: NDArray[np.float64] = remove_dc_offset(signal)
    bandpassed: NDArray[np.float64] = apply_bandpass_filter(
        dc_removed,
        fs=fs,
        lowcut=lowcut,
        highcut=highcut,
        order=order,
    )
    filtered: NDArray[np.float64] = apply_notch_filter(
        bandpassed,
        fs=fs,
        freq=notch_freq,
        q=notch_q,
    )
    rms_envelope: NDArray[np.float64] = compute_rms_envelope(
        filtered,
        fs=fs,
        window_ms=window_ms,
    )
    return PreprocessedChannel(filtered=filtered, rms_envelope=rms_envelope)
