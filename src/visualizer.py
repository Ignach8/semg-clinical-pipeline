"""Publication-grade matplotlib visualizations for clinical bilateral sEMG analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from scipy.signal import welch

from src.features import (
    compute_bilateral_symmetry,
    compute_rms,
    compute_spectral_metrics,
)

_COLOR_RAW: str = "#94A3B8"
_COLOR_LEFT: str = "#1D4ED8"
_COLOR_RIGHT: str = "#C2410C"
_COLOR_ASYMMETRY: str = "#F59E0B"


def _as_1d_float(values: ArrayLike, name: str) -> NDArray[np.float64]:
    arr: NDArray[np.float64] = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"Expected a non-empty 1D array for '{name}'.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Array '{name}' contains non-finite values.")
    return arr


def _style_axis(ax: Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, color="#94A3B8")
    ax.tick_params(labelsize=9)


def plot_emg_analysis(
    time_s: ArrayLike,
    raw_left: ArrayLike,
    filt_left: ArrayLike,
    env_left: ArrayLike,
    filt_right: ArrayLike,
    env_right: ArrayLike,
    fs: float = 1000.0,
    output_path: str | Path = "results/emg_clinical_plot.png",
) -> Path:
    """Render and save a 3-panel clinical bilateral sEMG analysis figure."""
    if fs <= 0.0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}.")

    t_arr: NDArray[np.float64] = _as_1d_float(time_s, "time_s")
    raw_l: NDArray[np.float64] = _as_1d_float(raw_left, "raw_left")
    flt_l: NDArray[np.float64] = _as_1d_float(filt_left, "filt_left")
    rms_l: NDArray[np.float64] = _as_1d_float(env_left, "env_left")
    flt_r: NDArray[np.float64] = _as_1d_float(filt_right, "filt_right")
    rms_r: NDArray[np.float64] = _as_1d_float(env_right, "env_right")

    expected_len: int = t_arr.size
    for name, arr in (
        ("raw_left", raw_l),
        ("filt_left", flt_l),
        ("env_left", rms_l),
        ("filt_right", flt_r),
        ("env_right", rms_r),
    ):
        if arr.size != expected_len:
            raise ValueError(
                f"Array '{name}' length ({arr.size}) does not match time_s ({expected_len})."
            )

    left_rms_uv: float = compute_rms(flt_l)
    right_rms_uv: float = compute_rms(flt_r)
    si_pct: float = compute_bilateral_symmetry(left_rms_uv, right_rms_uv)
    mdf_left, _ = compute_spectral_metrics(flt_l, fs=fs)
    mdf_right, _ = compute_spectral_metrics(flt_r, fs=fs)

    # Match the Welch PSD segment sizing used in biomarker extraction so the
    # plotted spectral envelope aligns directly with the reported MDF values.
    nperseg: int = min(flt_l.size, 1024) if flt_l.size >= 2048 else min(flt_l.size, 512)
    freqs_l: NDArray[np.float64]
    psd_l: NDArray[np.float64]
    freqs_l, psd_l = welch(flt_l, fs=fs, window="hann", nperseg=nperseg)
    freqs_r: NDArray[np.float64]
    psd_r: NDArray[np.float64]
    freqs_r, psd_r = welch(flt_r, fs=fs, window="hann", nperseg=nperseg)

    fig: Figure = Figure(figsize=(11.0, 9.5), dpi=150, layout="constrained")
    FigureCanvasAgg(fig)
    axes: NDArray[np.object_] = fig.subplots(3, 1)
    ax_raw: Axes = axes[0]
    ax_env: Axes = axes[1]
    ax_psd: Axes = axes[2]

    fig.suptitle(
        "Clinical Bilateral Surface EMG Analysis",
        fontsize=13,
        fontweight="bold",
        color="#0F172A",
    )

    ax_raw.plot(
        t_arr,
        raw_l,
        color=_COLOR_RAW,
        linewidth=0.75,
        alpha=0.75,
        label="Raw Left (DC + Motion <15 Hz + 50 Hz)",
    )
    ax_raw.plot(
        t_arr,
        flt_l,
        color=_COLOR_LEFT,
        linewidth=0.85,
        alpha=0.95,
        label="Filtered Left (SENIAM 20-450 Hz, 50 Hz Notch)",
    )
    ax_raw.set_title(
        "1. Left Channel Conditioning: Raw vs. SENIAM-Filtered sEMG",
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=6,
    )
    ax_raw.set_xlabel("Time (s)", fontsize=9.5)
    ax_raw.set_ylabel("Amplitude (uV)", fontsize=9.5)
    ax_raw.set_xlim(float(t_arr[0]), float(t_arr[-1]))
    raw_min: float = float(min(np.min(raw_l), np.min(flt_l)))
    raw_max: float = float(max(np.max(raw_l), np.max(flt_l)))
    raw_span: float = max(raw_max - raw_min, 1.0)
    ax_raw.set_ylim(raw_min - 0.08 * raw_span, raw_max + 0.34 * raw_span)
    ax_raw.legend(loc="upper right", fontsize=8.5, frameon=True, framealpha=0.92)
    _style_axis(ax_raw)

    ax_env.plot(
        t_arr,
        rms_l,
        color=_COLOR_LEFT,
        linewidth=1.8,
        label=f"Left RMS Envelope (Overall RMS = {left_rms_uv:.1f} uV)",
    )
    ax_env.plot(
        t_arr,
        rms_r,
        color=_COLOR_RIGHT,
        linewidth=1.8,
        label=f"Right RMS Envelope (Overall RMS = {right_rms_uv:.1f} uV)",
    )
    ax_env.fill_between(
        t_arr,
        rms_l,
        rms_r,
        color=_COLOR_ASYMMETRY,
        alpha=0.20,
        label=f"Bilateral Asymmetry (SI = {si_pct:.1f}%)",
    )
    ax_env.set_title(
        "2. Bilateral RMS Envelopes (100 ms Moving Window)",
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=6,
    )
    ax_env.set_xlabel("Time (s)", fontsize=9.5)
    ax_env.set_ylabel("RMS Amplitude (uV)", fontsize=9.5)
    ax_env.set_xlim(float(t_arr[0]), float(t_arr[-1]))
    env_max: float = float(max(np.max(rms_l), np.max(rms_r), 1.0))
    ax_env.set_ylim(0.0, env_max * 1.44)
    ax_env.legend(loc="upper right", fontsize=8.5, frameon=True, framealpha=0.92)
    _style_axis(ax_env)

    ax_psd.plot(
        freqs_l,
        psd_l,
        color=_COLOR_LEFT,
        linewidth=1.5,
        label="Left Channel PSD",
    )
    ax_psd.fill_between(freqs_l, psd_l, color=_COLOR_LEFT, alpha=0.10)
    ax_psd.plot(
        freqs_r,
        psd_r,
        color=_COLOR_RIGHT,
        linewidth=1.5,
        label="Right Channel PSD",
    )
    ax_psd.fill_between(freqs_r, psd_r, color=_COLOR_RIGHT, alpha=0.10)
    ax_psd.axvline(
        mdf_left,
        color=_COLOR_LEFT,
        linestyle="--",
        linewidth=1.3,
        label=f"Left MDF = {mdf_left:.1f} Hz",
    )
    ax_psd.axvline(
        mdf_right,
        color=_COLOR_RIGHT,
        linestyle=":",
        linewidth=1.6,
        label=f"Right MDF = {mdf_right:.1f} Hz",
    )
    ax_psd.set_title(
        "3. Power Spectral Density (Welch) and Median Frequency Comparison",
        fontsize=10.5,
        fontweight="bold",
        loc="left",
        pad=6,
    )
    ax_psd.set_xlabel("Frequency (Hz)", fontsize=9.5)
    ax_psd.set_ylabel("PSD (uV^2/Hz)", fontsize=9.5)
    ax_psd.set_xlim(0.0, min(500.0, 0.5 * fs))
    psd_max: float = float(max(np.max(psd_l), np.max(psd_r), 1e-6))
    ax_psd.set_ylim(0.0, psd_max * 1.25)
    ax_psd.legend(loc="upper right", fontsize=8.5, frameon=True, framealpha=0.92, ncol=2)
    _style_axis(ax_psd)

    destination: Path = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150)
    return destination
