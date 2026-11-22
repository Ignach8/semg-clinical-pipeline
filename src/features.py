"""Clinical neuromuscular biomarker extraction from preprocessed sEMG signals."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.integrate import cumulative_trapezoid
from scipy.signal import welch


class ChannelBiomarkers(BaseModel):
    """Neuromuscular biomarkers extracted from a single preprocessed sEMG channel."""

    model_config = ConfigDict(frozen=True)

    rms_uv: float = Field(
        ...,
        ge=0.0,
        description="Root-mean-square activation amplitude in microvolts (uV).",
    )
    mdf_hz: float = Field(
        ...,
        ge=0.0,
        description="Median frequency of the Welch power spectral density (Hz).",
    )
    mnf_hz: float = Field(
        ...,
        ge=0.0,
        description="Mean frequency of the Welch power spectral density (Hz).",
    )
    fatigue_slope_hz_per_s: float = Field(
        ...,
        description=(
            "Linear regression slope of epoch median frequency over time (Hz/s); "
            "negative values indicate progressive neuromuscular fatigue."
        ),
    )


class BilateralAssessment(BaseModel):
    """Structured bilateral sEMG biomarker summary for clinical JSON export."""

    model_config = ConfigDict(frozen=True)

    sampling_rate_hz: float = Field(
        ...,
        gt=0.0,
        description="Acquisition sampling frequency (Hz).",
    )
    duration_s: float = Field(
        ...,
        gt=0.0,
        description="Total duration of the analyzed recording (seconds).",
    )
    left: ChannelBiomarkers = Field(
        ...,
        description="Biomarkers extracted from the left muscle channel.",
    )
    right: ChannelBiomarkers = Field(
        ...,
        description="Biomarkers extracted from the right muscle channel.",
    )
    symmetry_index_pct: float = Field(
        ...,
        ge=0.0,
        le=200.0,
        description=(
            "Robinson bilateral Symmetry Index (%), where 0% denotes perfect "
            "symmetry and values above 15% indicate clinically relevant asymmetry."
        ),
    )
    dominant_side: Literal["left", "right", "symmetric"] = Field(
        ...,
        description="Side exhibiting higher RMS neuromuscular activation amplitude.",
    )


def _validate_signal(signal: ArrayLike) -> NDArray[np.float64]:
    arr: NDArray[np.float64] = np.asarray(signal, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D signal array, got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError("Signal array must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Signal contains non-finite values (NaN or Inf).")
    return arr


def _compute_welch_psd(
    signal: NDArray[np.float64],
    fs: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    # Adaptive Welch segment length (1024 samples for multi-second recordings,
    # 512 samples for 1-second epochs) ensures >=3 overlapping Hann segments
    # to reduce spectral variance while preserving ~1-2 Hz frequency resolution.
    nperseg: int = min(signal.size, 1024) if signal.size >= 2048 else min(signal.size, 512)
    freqs: NDArray[np.float64]
    psd: NDArray[np.float64]
    freqs, psd = welch(signal, fs=fs, window="hann", nperseg=nperseg)
    return np.asarray(freqs, dtype=np.float64), np.asarray(psd, dtype=np.float64)


def compute_rms(signal: ArrayLike) -> float:
    """Compute overall root-mean-square (RMS) sEMG amplitude in microvolts."""
    arr: NDArray[np.float64] = _validate_signal(signal)
    return float(np.sqrt(np.mean(np.square(arr))))


def compute_spectral_metrics(
    signal: ArrayLike,
    fs: float = 1000.0,
) -> tuple[float, float]:
    """Compute Welch PSD Median Frequency (MDF) and Mean Frequency (MNF) in Hz."""
    arr: NDArray[np.float64] = _validate_signal(signal)
    if fs <= 0.0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}.")
    if arr.size < 2:
        raise ValueError("Signal must contain at least 2 samples for spectral estimation.")

    freqs: NDArray[np.float64]
    psd: NDArray[np.float64]
    freqs, psd = _compute_welch_psd(arr, fs=fs)

    # Continuous trapezoidal integration with linear interpolation avoids
    # discrete FFT-bin staircase bias in median frequency estimation.
    cumulative_power: NDArray[np.float64] = np.asarray(
        cumulative_trapezoid(psd, freqs, initial=0.0),
        dtype=np.float64,
    )
    total_power: float = float(cumulative_power[-1])
    if total_power <= 0.0:
        return (0.0, 0.0)

    mdf_hz: float = float(np.interp(0.5 * total_power, cumulative_power, freqs))
    mnf_hz: float = float(np.trapezoid(freqs * psd, freqs) / total_power)
    return (mdf_hz, mnf_hz)


def compute_fatigue_slope(
    signal: ArrayLike,
    fs: float = 1000.0,
    epoch_sec: float = 1.0,
) -> float:
    """Compute linear regression slope of epoch MDF over time in Hz/s."""
    arr: NDArray[np.float64] = _validate_signal(signal)
    if fs <= 0.0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}.")
    if epoch_sec <= 0.0:
        raise ValueError(f"Epoch duration must be positive, got {epoch_sec}.")

    epoch_samples: int = int(round(fs * epoch_sec))
    if epoch_samples < 2:
        raise ValueError("Epoch length must span at least 2 samples.")

    n_epochs: int = arr.size // epoch_samples
    if n_epochs < 2:
        raise ValueError(
            f"Signal duration must span at least 2 full epochs of {epoch_sec} s."
        )

    epoch_times: NDArray[np.float64] = np.empty(n_epochs, dtype=np.float64)
    epoch_mdfs: NDArray[np.float64] = np.empty(n_epochs, dtype=np.float64)
    epoch_rmss: NDArray[np.float64] = np.empty(n_epochs, dtype=np.float64)

    for idx in range(n_epochs):
        start: int = idx * epoch_samples
        segment: NDArray[np.float64] = arr[start : start + epoch_samples]
        epoch_times[idx] = (idx + 0.5) * epoch_sec
        epoch_rmss[idx] = compute_rms(segment)
        mdf_hz, _ = compute_spectral_metrics(segment, fs=fs)
        epoch_mdfs[idx] = mdf_hz

    # Restrict fatigue regression to active contraction epochs (>=25% of peak
    # epoch RMS) so quiescent baseline epochs—whose flat noise floor has an
    # artificially high MDF—do not bias the neuromuscular fatigue slope.
    peak_rms: float = float(np.max(epoch_rmss))
    active_mask: NDArray[np.bool_] = (
        epoch_rmss >= 0.25 * peak_rms
        if peak_rms > 0.0
        else np.ones(n_epochs, dtype=np.bool_)
    )
    selected_times: NDArray[np.float64] = (
        epoch_times[active_mask] if int(np.sum(active_mask)) >= 2 else epoch_times
    )
    selected_mdfs: NDArray[np.float64] = (
        epoch_mdfs[active_mask] if int(np.sum(active_mask)) >= 2 else epoch_mdfs
    )

    t_centered: NDArray[np.float64] = selected_times - float(np.mean(selected_times))
    denom: float = float(np.dot(t_centered, t_centered))
    if denom <= 0.0:
        return 0.0

    mdf_centered: NDArray[np.float64] = selected_mdfs - float(np.mean(selected_mdfs))
    return float(np.dot(t_centered, mdf_centered) / denom)


def compute_bilateral_symmetry(left_rms: float, right_rms: float) -> float:
    """Compute Robinson bilateral Symmetry Index (%) between left and right RMS."""
    if not (np.isfinite(left_rms) and np.isfinite(right_rms)):
        raise ValueError("RMS values must be finite.")
    if left_rms < 0.0 or right_rms < 0.0:
        raise ValueError("RMS values must be non-negative.")

    mean_rms: float = 0.5 * (left_rms + right_rms)
    if mean_rms == 0.0:
        return 0.0

    # Robinson Symmetry Index (SI = |L - R| / (0.5 * (L + R)) * 100%) provides
    # a side-agnostic clinical asymmetry metric without assuming which limb is
    # impaired; 0% indicates perfect symmetry and >15% flags clinical asymmetry.
    return float(abs(left_rms - right_rms) / mean_rms * 100.0)


def extract_bilateral_biomarkers(
    left_signal: ArrayLike,
    right_signal: ArrayLike,
    fs: float = 1000.0,
    *,
    epoch_sec: float = 1.0,
) -> BilateralAssessment:
    """Extract structured bilateral neuromuscular biomarkers as a Pydantic model."""
    left_arr: NDArray[np.float64] = _validate_signal(left_signal)
    right_arr: NDArray[np.float64] = _validate_signal(right_signal)
    if left_arr.size != right_arr.size:
        raise ValueError(
            f"Left ({left_arr.size}) and right ({right_arr.size}) signals must have equal length."
        )
    if fs <= 0.0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}.")

    left_rms: float = compute_rms(left_arr)
    left_mdf, left_mnf = compute_spectral_metrics(left_arr, fs=fs)
    left_slope: float = compute_fatigue_slope(left_arr, fs=fs, epoch_sec=epoch_sec)

    right_rms: float = compute_rms(right_arr)
    right_mdf, right_mnf = compute_spectral_metrics(right_arr, fs=fs)
    right_slope: float = compute_fatigue_slope(right_arr, fs=fs, epoch_sec=epoch_sec)

    symmetry_index: float = compute_bilateral_symmetry(left_rms, right_rms)
    dominant_side: Literal["left", "right", "symmetric"]
    if np.isclose(left_rms, right_rms):
        dominant_side = "symmetric"
    elif left_rms > right_rms:
        dominant_side = "left"
    else:
        dominant_side = "right"

    return BilateralAssessment(
        sampling_rate_hz=float(fs),
        duration_s=float(left_arr.size / fs),
        left=ChannelBiomarkers(
            rms_uv=left_rms,
            mdf_hz=left_mdf,
            mnf_hz=left_mnf,
            fatigue_slope_hz_per_s=left_slope,
        ),
        right=ChannelBiomarkers(
            rms_uv=right_rms,
            mdf_hz=right_mdf,
            mnf_hz=right_mnf,
            fatigue_slope_hz_per_s=right_slope,
        ),
        symmetry_index_pct=symmetry_index,
        dominant_side=dominant_side,
    )
