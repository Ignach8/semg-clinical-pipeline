"""Synthetic 2-channel bilateral quadriceps sEMG signal generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def _shape_shwedyk_emg(
    white_noise: NDArray[np.float64],
    fs: float,
    f_low: float,
    f_high: float,
) -> NDArray[np.float64]:
    """Shape unit-variance white noise with the Shwedyk sEMG spectral model."""
    n_samples: int = white_noise.size
    freqs: NDArray[np.float64] = np.fft.rfftfreq(n_samples, d=1.0 / fs)

    # Shwedyk physiological surface EMG PSD model reproduces the characteristic
    # unimodal bandpass spectral shape (50-150 Hz peak) of motor unit action
    # potential trains recorded at the skin surface.
    f_sq: NDArray[np.float64] = np.square(freqs)
    amplitude_response: NDArray[np.float64] = (
        (f_high**2)
        * freqs
        / (np.sqrt(f_sq + f_low**2) * (f_sq + f_high**2))
    )
    amplitude_response[0] = 0.0

    spectrum: NDArray[np.complex128] = np.fft.rfft(white_noise) * amplitude_response
    shaped: NDArray[np.float64] = np.fft.irfft(spectrum, n=n_samples)
    std: float = float(np.std(shaped))
    return shaped / std if std > 0.0 else shaped


def _build_isometric_envelope(
    time_s: NDArray[np.float64],
    duration_s: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build a symmetric isometric contraction envelope and fatigue progression."""
    onset_start: float = 0.10 * duration_s
    plateau_start: float = 0.15 * duration_s
    plateau_end: float = 0.85 * duration_s
    offset_end: float = 0.90 * duration_s

    envelope: NDArray[np.float64] = np.zeros_like(time_s, dtype=np.float64)

    ramp_up_mask: NDArray[np.bool_] = (time_s >= onset_start) & (time_s < plateau_start)
    ramp_up_phase: NDArray[np.float64] = (
        (time_s[ramp_up_mask] - onset_start) / (plateau_start - onset_start)
    )
    envelope[ramp_up_mask] = 0.5 * (1.0 - np.cos(np.pi * ramp_up_phase))

    plateau_mask: NDArray[np.bool_] = (time_s >= plateau_start) & (time_s <= plateau_end)
    envelope[plateau_mask] = 1.0

    ramp_down_mask: NDArray[np.bool_] = (time_s > plateau_end) & (time_s <= offset_end)
    ramp_down_phase: NDArray[np.float64] = (
        (time_s[ramp_down_mask] - plateau_end) / (offset_end - plateau_end)
    )
    envelope[ramp_down_mask] = 0.5 * (1.0 + np.cos(np.pi * ramp_down_phase))

    # Nonlinear fatigue progression concentrates spectral compression in the
    # second half of the isometric hold, reflecting metabolic accumulation and
    # reduced muscle fiber conduction velocity.
    normalized_hold: NDArray[np.float64] = np.clip(
        (time_s - plateau_start) / (plateau_end - plateau_start),
        0.0,
        1.0,
    )
    fatigue_weight: NDArray[np.float64] = np.power(normalized_hold, 1.5)
    return envelope, fatigue_weight


def _synthesize_channel(
    rng: np.random.Generator,
    time_s: NDArray[np.float64],
    fs: float,
    envelope: NDArray[np.float64],
    fatigue_weight: NDArray[np.float64],
    active_rms_uv: float,
    dc_offset_uv: float,
    motion_freqs_hz: tuple[float, float, float],
    motion_amps_uv: tuple[float, float, float],
    powerline_amp_uv: float,
    powerline_phase_rad: float,
    noise_floor_rms_uv: float = 4.0,
) -> NDArray[np.float64]:
    """Synthesize one raw sEMG channel with fatigue and acquisition artifacts."""
    n_samples: int = time_s.size

    # Fresh vs. fatigued quadriceps spectral parameters model conduction
    # velocity slowing (~15-20% downward shift in median frequency over the hold).
    fresh_emg: NDArray[np.float64] = _shape_shwedyk_emg(
        rng.standard_normal(n_samples),
        fs=fs,
        f_low=58.0,
        f_high=175.0,
    )
    fatigued_emg: NDArray[np.float64] = _shape_shwedyk_emg(
        rng.standard_normal(n_samples),
        fs=fs,
        f_low=34.0,
        f_high=105.0,
    )

    # Equal-power cross-fading shifts instantaneous spectral energy smoothly
    # toward lower frequencies without introducing artificial amplitude modulation.
    carrier_emg: NDArray[np.float64] = (
        np.sqrt(1.0 - fatigue_weight) * fresh_emg
        + np.sqrt(fatigue_weight) * fatigued_emg
    )
    physiological_emg: NDArray[np.float64] = active_rms_uv * envelope * carrier_emg

    # Low-frequency motion artifacts (<15 Hz) model skin-electrode interface
    # displacement, cable sway, and muscle-belly bulging during contraction.
    motion_artifact: NDArray[np.float64] = np.zeros(n_samples, dtype=np.float64)
    for freq_hz, amp_uv in zip(motion_freqs_hz, motion_amps_uv, strict=True):
        phase_rad: float = float(rng.uniform(0.0, 2.0 * np.pi))
        motion_artifact += amp_uv * np.sin(2.0 * np.pi * freq_hz * time_s + phase_rad)

    # 50 Hz interference matches the Chilean/European AC power grid standard.
    powerline_interference: NDArray[np.float64] = powerline_amp_uv * np.sin(
        2.0 * np.pi * 50.0 * time_s + powerline_phase_rad
    )

    # White noise floor approximates ADS1292 input-referred noise at 1 kSPS
    # combined with skin-electrode thermal noise.
    instrumentation_noise: NDArray[np.float64] = (
        noise_floor_rms_uv * rng.standard_normal(n_samples)
    )

    return (
        physiological_emg
        + dc_offset_uv
        + motion_artifact
        + powerline_interference
        + instrumentation_noise
    )


def generate_synthetic_emg(
    fs: float = 1000.0,
    duration_s: float = 10.0,
    left_rms_uv: float = 150.0,
    asymmetry_reduction: float = 0.30,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a 2-channel synthetic bilateral quadriceps isometric sEMG dataset."""
    if fs <= 0.0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}.")
    if duration_s <= 0.0:
        raise ValueError(f"Duration must be positive, got {duration_s}.")
    if left_rms_uv <= 0.0:
        raise ValueError(f"Baseline active RMS must be positive, got {left_rms_uv}.")
    if not (0.0 <= asymmetry_reduction < 1.0):
        raise ValueError(
            f"Asymmetry reduction must lie in [0, 1), got {asymmetry_reduction}."
        )

    rng: np.random.Generator = np.random.default_rng(seed)
    n_samples: int = int(round(fs * duration_s))
    time_s: NDArray[np.float64] = np.arange(n_samples, dtype=np.float64) / fs
    timestamp_ms: NDArray[np.float64] = np.round(time_s * 1000.0, 3)

    envelope: NDArray[np.float64]
    fatigue_weight: NDArray[np.float64]
    envelope, fatigue_weight = _build_isometric_envelope(time_s, duration_s)

    # Attenuate right quadriceps amplitude by ~30% relative to the left side
    # to model unilateral post-injury or post-stroke neuromuscular weakness.
    right_rms_uv: float = left_rms_uv * (1.0 - asymmetry_reduction)

    ch1_left_uv: NDArray[np.float64] = _synthesize_channel(
        rng=rng,
        time_s=time_s,
        fs=fs,
        envelope=envelope,
        fatigue_weight=fatigue_weight,
        active_rms_uv=left_rms_uv,
        dc_offset_uv=450.0,
        motion_freqs_hz=(0.7, 3.2, 8.5),
        motion_amps_uv=(75.0, 40.0, 22.0),
        powerline_amp_uv=45.0,
        powerline_phase_rad=0.25,
    )
    ch2_right_uv: NDArray[np.float64] = _synthesize_channel(
        rng=rng,
        time_s=time_s,
        fs=fs,
        envelope=envelope,
        fatigue_weight=fatigue_weight,
        active_rms_uv=right_rms_uv,
        dc_offset_uv=-320.0,
        motion_freqs_hz=(0.5, 4.1, 11.0),
        motion_amps_uv=(65.0, 35.0, 18.0),
        powerline_amp_uv=38.0,
        powerline_phase_rad=1.10,
    )

    return pd.DataFrame(
        {
            "timestamp_ms": timestamp_ms,
            "ch1_left_uv": np.round(ch1_left_uv, 4),
            "ch2_right_uv": np.round(ch2_right_uv, 4),
        }
    )


def generate_sample_csv(
    output_path: str | Path = "data/sample_emg_2ch.csv",
    *,
    fs: float = 1000.0,
    duration_s: float = 10.0,
    seed: int = 42,
) -> Path:
    """Generate and persist the 2-channel synthetic sEMG dataset as CSV."""
    destination: Path = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset: pd.DataFrame = generate_synthetic_emg(
        fs=fs,
        duration_s=duration_s,
        seed=seed,
    )
    dataset.to_csv(destination, index=False)
    return destination


if __name__ == "__main__":
    generate_sample_csv()
