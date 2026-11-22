"""Unit and integration tests for clinical sEMG filtering, biomarkers, and pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from main import run_pipeline
from src.features import (
    BilateralAssessment,
    ChannelBiomarkers,
    compute_bilateral_symmetry,
    compute_rms,
    extract_bilateral_biomarkers,
)
from src.filters import (
    apply_bandpass_filter,
    apply_notch_filter,
    compute_rms_envelope,
    preprocess_channel,
    remove_dc_offset,
)
from src.synthetic_emg import generate_synthetic_emg


def test_notch_filter_attenuates_50hz_powerline() -> None:
    """Verify that the 50 Hz notch filter strongly suppresses a pure 50 Hz tone."""
    fs: float = 1000.0
    t: NDArray[np.float64] = np.arange(5000, dtype=np.float64) / fs
    sine_50hz: NDArray[np.float64] = 100.0 * np.sin(2.0 * np.pi * 50.0 * t)
    sine_100hz: NDArray[np.float64] = 100.0 * np.sin(2.0 * np.pi * 100.0 * t)

    notched_50: NDArray[np.float64] = apply_notch_filter(
        sine_50hz,
        fs=fs,
        freq=50.0,
        q=30.0,
    )
    notched_100: NDArray[np.float64] = apply_notch_filter(
        sine_100hz,
        fs=fs,
        freq=50.0,
        q=30.0,
    )

    steady: slice = slice(1500, 3500)
    attenuation_db: float = float(
        20.0 * np.log10(compute_rms(notched_50[steady]) / compute_rms(sine_50hz[steady]))
    )
    assert attenuation_db < -60.0
    assert compute_rms(notched_100[steady]) == pytest.approx(
        compute_rms(sine_100hz[steady]),
        rel=0.01,
    )


def test_bandpass_filter_removes_dc_and_low_freq_drift_preserves_100hz() -> None:
    """Verify SENIAM 20-450 Hz bandpass rejects DC and <10 Hz drift while passing 100 Hz."""
    fs: float = 1000.0
    t: NDArray[np.float64] = np.arange(5000, dtype=np.float64) / fs

    dc_offset: float = 500.0
    low_freq_drift: NDArray[np.float64] = 120.0 * np.sin(2.0 * np.pi * 3.0 * t)
    target_100hz: NDArray[np.float64] = 100.0 * np.sin(2.0 * np.pi * 100.0 * t)
    composite: NDArray[np.float64] = dc_offset + low_freq_drift + target_100hz

    filtered: NDArray[np.float64] = apply_bandpass_filter(
        composite,
        fs=fs,
        lowcut=20.0,
        highcut=450.0,
        order=4,
    )

    steady: slice = slice(1000, 4000)
    assert abs(float(np.mean(filtered[steady]))) < 0.1
    assert compute_rms(filtered[steady]) == pytest.approx(
        compute_rms(target_100hz[steady]),
        rel=0.02,
    )
    residual_error: float = compute_rms(filtered[steady] - target_100hz[steady])
    assert residual_error < 1.0


def test_bilateral_symmetry_index() -> None:
    """Verify Robinson Symmetry Index equals 0.0% for equal RMS and matches formula otherwise."""
    assert compute_bilateral_symmetry(120.0, 120.0) == 0.0
    assert compute_bilateral_symmetry(0.0, 0.0) == 0.0

    expected_si: float = abs(100.0 - 70.0) / (0.5 * (100.0 + 70.0)) * 100.0
    assert compute_bilateral_symmetry(100.0, 70.0) == pytest.approx(expected_si, rel=1e-9)
    assert compute_bilateral_symmetry(70.0, 100.0) == pytest.approx(expected_si, rel=1e-9)

    with pytest.raises(ValueError):
        compute_bilateral_symmetry(-10.0, 50.0)


def test_extract_bilateral_biomarkers_returns_valid_assessment() -> None:
    """Verify bilateral biomarker extraction returns a validated Pydantic assessment."""
    fs: float = 1000.0
    df = generate_synthetic_emg(fs=fs, duration_s=10.0, seed=42)

    left_proc = preprocess_channel(df["ch1_left_uv"].to_numpy(dtype=np.float64), fs=fs)
    right_proc = preprocess_channel(df["ch2_right_uv"].to_numpy(dtype=np.float64), fs=fs)

    assessment: BilateralAssessment = extract_bilateral_biomarkers(
        left_signal=left_proc.filtered,
        right_signal=right_proc.filtered,
        fs=fs,
    )

    assert isinstance(assessment, BilateralAssessment)
    assert isinstance(assessment.left, ChannelBiomarkers)
    assert isinstance(assessment.right, ChannelBiomarkers)
    assert assessment.left.rms_uv > 0.0
    assert assessment.right.rms_uv > 0.0
    assert assessment.left.mdf_hz > 0.0
    assert assessment.right.mdf_hz > 0.0
    assert assessment.left.mnf_hz > 0.0
    assert assessment.right.mnf_hz > 0.0
    assert assessment.left.rms_uv > assessment.right.rms_uv
    assert assessment.symmetry_index_pct > 15.0
    assert assessment.dominant_side == "left"
    assert assessment.left.fatigue_slope_hz_per_s < -0.1
    assert assessment.right.fatigue_slope_hz_per_s < -0.1

    roundtrip: BilateralAssessment = BilateralAssessment.model_validate_json(
        assessment.model_dump_json()
    )
    assert roundtrip == assessment


def test_pipeline_generates_biomarkers_json_and_plot(tmp_path: Path) -> None:
    """Verify end-to-end pipeline produces valid biomarkers.json and emg_clinical_plot.png."""
    csv_path: Path = tmp_path / "data" / "sample_emg_2ch.csv"
    results_dir: Path = tmp_path / "results"

    assessment: BilateralAssessment = run_pipeline(
        input_csv=csv_path,
        results_dir=results_dir,
        fs=1000.0,
        regenerate=True,
    )

    biomarkers_path: Path = results_dir / "biomarkers.json"
    plot_path: Path = results_dir / "emg_clinical_plot.png"
    summary_path: Path = results_dir / "clinical_summary.md"

    assert csv_path.is_file()
    assert biomarkers_path.is_file()
    assert plot_path.is_file()
    assert summary_path.is_file()

    loaded: BilateralAssessment = BilateralAssessment.model_validate_json(
        biomarkers_path.read_text(encoding="utf-8")
    )
    assert loaded == assessment

    png_bytes: bytes = plot_path.read_bytes()
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png_bytes) > 10_000


def test_dc_removal_and_rms_envelope_amplitude_preservation() -> None:
    """Verify DC offset removal centers the signal and RMS envelope avoids edge droop."""
    fs: float = 1000.0
    shifted: NDArray[np.float64] = np.full(1000, 250.0, dtype=np.float64)
    centered: NDArray[np.float64] = remove_dc_offset(shifted)
    assert np.allclose(centered, 0.0)

    constant_signal: NDArray[np.float64] = np.full(1000, 40.0, dtype=np.float64)
    envelope: NDArray[np.float64] = compute_rms_envelope(
        constant_signal,
        fs=fs,
        window_ms=100.0,
    )
    assert envelope.shape == constant_signal.shape
    assert np.allclose(envelope, 40.0)
