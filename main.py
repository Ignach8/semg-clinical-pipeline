"""End-to-end clinical bilateral surface EMG processing and reporting pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.features import BilateralAssessment, extract_bilateral_biomarkers
from src.filters import PreprocessedChannel, preprocess_channel
from src.synthetic_emg import generate_sample_csv
from src.visualizer import plot_emg_analysis

DEFAULT_DATA_DIR: Path = Path("data")
DEFAULT_RESULTS_DIR: Path = Path("results")
DEFAULT_CSV_PATH: Path = DEFAULT_DATA_DIR / "sample_emg_2ch.csv"
DEFAULT_FS_HZ: float = 1000.0

ASYMMETRY_THRESHOLD_PCT: float = 15.0
FATIGUE_SLOPE_THRESHOLD_HZ_PER_S: float = -0.1

REQUIRED_COLUMNS: tuple[str, str, str] = (
    "timestamp_ms",
    "ch1_left_uv",
    "ch2_right_uv",
)


def _evaluate_clinical_flags(
    assessment: BilateralAssessment,
) -> tuple[str, str, str, str]:
    """Derive clinical asymmetry and neuromuscular fatigue interpretation flags."""
    if assessment.symmetry_index_pct > ASYMMETRY_THRESHOLD_PCT:
        asymmetry_status: str = "FLAGGED: Clinically significant asymmetry"
        weaker_side: str = "right" if assessment.dominant_side == "left" else "left"
        asymmetry_detail: str = (
            f"Symmetry Index is {assessment.symmetry_index_pct:.2f}% "
            f"(exceeds {ASYMMETRY_THRESHOLD_PCT:.1f}% threshold), indicating "
            f"reduced activation on the {weaker_side} side relative to the "
            f"{assessment.dominant_side} side."
        )
    else:
        asymmetry_status = "NORMAL: Within symmetric clinical range"
        asymmetry_detail = (
            f"Symmetry Index is {assessment.symmetry_index_pct:.2f}% "
            f"(within <= {ASYMMETRY_THRESHOLD_PCT:.1f}% clinical threshold)."
        )

    left_fatigued: bool = (
        assessment.left.fatigue_slope_hz_per_s < FATIGUE_SLOPE_THRESHOLD_HZ_PER_S
    )
    right_fatigued: bool = (
        assessment.right.fatigue_slope_hz_per_s < FATIGUE_SLOPE_THRESHOLD_HZ_PER_S
    )

    if left_fatigued and right_fatigued:
        fatigue_status: str = "FLAGGED: Progressive bilateral neuromuscular fatigue"
        fatigue_detail: str = (
            f"Both channels exhibit negative MDF slopes below "
            f"{FATIGUE_SLOPE_THRESHOLD_HZ_PER_S:.1f} Hz/s "
            f"(Left: {assessment.left.fatigue_slope_hz_per_s:.2f} Hz/s, "
            f"Right: {assessment.right.fatigue_slope_hz_per_s:.2f} Hz/s)."
        )
    elif left_fatigued or right_fatigued:
        fatigued_side: str = "left" if left_fatigued else "right"
        fatigue_status = f"FLAGGED: Progressive unilateral fatigue ({fatigued_side})"
        fatigue_detail = (
            f"The {fatigued_side} channel exhibits an MDF slope below "
            f"{FATIGUE_SLOPE_THRESHOLD_HZ_PER_S:.1f} Hz/s "
            f"(Left: {assessment.left.fatigue_slope_hz_per_s:.2f} Hz/s, "
            f"Right: {assessment.right.fatigue_slope_hz_per_s:.2f} Hz/s)."
        )
    else:
        fatigue_status = "NORMAL: No significant neuromuscular fatigue detected"
        fatigue_detail = (
            f"Both channels remain at or above the "
            f"{FATIGUE_SLOPE_THRESHOLD_HZ_PER_S:.1f} Hz/s MDF slope threshold "
            f"(Left: {assessment.left.fatigue_slope_hz_per_s:.2f} Hz/s, "
            f"Right: {assessment.right.fatigue_slope_hz_per_s:.2f} Hz/s)."
        )

    return asymmetry_status, asymmetry_detail, fatigue_status, fatigue_detail


def build_clinical_summary_markdown(
    assessment: BilateralAssessment,
    plot_filename: str = "emg_clinical_plot.png",
) -> str:
    """Build a structured Markdown clinical report from bilateral biomarkers."""
    asymmetry_status, asymmetry_detail, fatigue_status, fatigue_detail = (
        _evaluate_clinical_flags(assessment)
    )
    rms_delta: float = assessment.right.rms_uv - assessment.left.rms_uv
    mdf_delta: float = assessment.right.mdf_hz - assessment.left.mdf_hz
    mnf_delta: float = assessment.right.mnf_hz - assessment.left.mnf_hz
    slope_delta: float = (
        assessment.right.fatigue_slope_hz_per_s
        - assessment.left.fatigue_slope_hz_per_s
    )

    lines: list[str] = [
        "# Clinical Surface EMG Assessment Summary",
        "",
        "## Acquisition Parameters",
        "",
        f"- **Sampling Rate:** {assessment.sampling_rate_hz:.1f} Hz",
        f"- **Recording Duration:** {assessment.duration_s:.2f} s",
        "- **Preprocessing:** DC offset removal, SENIAM 20-450 Hz 4th-order zero-phase Butterworth bandpass, 50 Hz notch (Q=30), 100 ms moving RMS envelope",
        "",
        "## Bilateral Biomarker Comparison",
        "",
        "| Biomarker | Left Channel | Right Channel | Delta (Right - Left) |",
        "| --- | ---: | ---: | ---: |",
        f"| RMS Amplitude (uV) | {assessment.left.rms_uv:.2f} | {assessment.right.rms_uv:.2f} | {rms_delta:+.2f} |",
        f"| Median Frequency, MDF (Hz) | {assessment.left.mdf_hz:.2f} | {assessment.right.mdf_hz:.2f} | {mdf_delta:+.2f} |",
        f"| Mean Frequency, MNF (Hz) | {assessment.left.mnf_hz:.2f} | {assessment.right.mnf_hz:.2f} | {mnf_delta:+.2f} |",
        f"| Fatigue Slope (Hz/s) | {assessment.left.fatigue_slope_hz_per_s:.2f} | {assessment.right.fatigue_slope_hz_per_s:.2f} | {slope_delta:+.2f} |",
        "",
        "## Bilateral Symmetry and Clinical Interpretation",
        "",
        f"- **Bilateral Symmetry Index (SI):** {assessment.symmetry_index_pct:.2f}%",
        f"- **Dominant Side (Higher RMS):** {assessment.dominant_side}",
        f"- **Asymmetry Flag (> {ASYMMETRY_THRESHOLD_PCT:.1f}% threshold):** {asymmetry_status}",
        f"  - {asymmetry_detail}",
        f"- **Fatigue Flag (MDF slope < {FATIGUE_SLOPE_THRESHOLD_HZ_PER_S:.1f} Hz/s):** {fatigue_status}",
        f"  - {fatigue_detail}",
        "",
        "## Visual Analysis",
        "",
        f"![Clinical Bilateral Surface EMG Analysis]({plot_filename})",
        "",
    ]
    return "\n".join(lines)


def format_stdout_summary(
    assessment: BilateralAssessment,
    csv_path: Path,
    json_path: Path,
    plot_path: Path,
    markdown_path: Path,
) -> str:
    """Format an aligned plain-text clinical summary table for stdout."""
    asymmetry_status, _, fatigue_status, _ = _evaluate_clinical_flags(assessment)

    metric_w: int = 24
    val_w: int = 14
    divider: str = (
        f"+-{'-' * metric_w}-+-{'-' * val_w}-+-{'-' * val_w}-+-{'-' * val_w}-+"
    )
    rows: list[tuple[str, str, str, str]] = [
        (
            "RMS Amplitude (uV)",
            f"{assessment.left.rms_uv:.2f}",
            f"{assessment.right.rms_uv:.2f}",
            f"{assessment.right.rms_uv - assessment.left.rms_uv:+.2f}",
        ),
        (
            "Median Frequency (Hz)",
            f"{assessment.left.mdf_hz:.2f}",
            f"{assessment.right.mdf_hz:.2f}",
            f"{assessment.right.mdf_hz - assessment.left.mdf_hz:+.2f}",
        ),
        (
            "Mean Frequency (Hz)",
            f"{assessment.left.mnf_hz:.2f}",
            f"{assessment.right.mnf_hz:.2f}",
            f"{assessment.right.mnf_hz - assessment.left.mnf_hz:+.2f}",
        ),
        (
            "Fatigue Slope (Hz/s)",
            f"{assessment.left.fatigue_slope_hz_per_s:.2f}",
            f"{assessment.right.fatigue_slope_hz_per_s:.2f}",
            f"{assessment.right.fatigue_slope_hz_per_s - assessment.left.fatigue_slope_hz_per_s:+.2f}",
        ),
    ]

    lines: list[str] = [
        "sEMG Clinical Pipeline Summary",
        f"Input: {csv_path} | fs = {assessment.sampling_rate_hz:.1f} Hz | Duration = {assessment.duration_s:.2f} s",
        divider,
        f"| {'Biomarker':<{metric_w}} | {'Left':>{val_w}} | {'Right':>{val_w}} | {'Delta (R - L)':>{val_w}} |",
        divider,
    ]
    for label, left_val, right_val, delta_val in rows:
        lines.append(
            f"| {label:<{metric_w}} | {left_val:>{val_w}} | {right_val:>{val_w}} | {delta_val:>{val_w}} |"
        )
    lines.extend(
        [
            divider,
            f"Bilateral Symmetry Index : {assessment.symmetry_index_pct:.2f}% (Dominant: {assessment.dominant_side})",
            f"Asymmetry Assessment     : {asymmetry_status}",
            f"Fatigue Assessment       : {fatigue_status}",
            f"Outputs                  : {json_path}, {plot_path}, {markdown_path}",
        ]
    )
    return "\n".join(lines)


def run_pipeline(
    input_csv: Path = DEFAULT_CSV_PATH,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    fs: float = DEFAULT_FS_HZ,
    regenerate: bool = False,
) -> BilateralAssessment:
    """Execute the full sEMG preprocessing, biomarker extraction, and reporting pipeline."""
    input_csv.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if regenerate or not input_csv.exists():
        generate_sample_csv(output_path=input_csv, fs=fs)

    df: pd.DataFrame = pd.read_csv(input_csv)
    missing_cols: list[str] = [
        col for col in REQUIRED_COLUMNS if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"Input CSV '{input_csv}' is missing required columns: {missing_cols}."
        )

    time_s: NDArray[np.float64] = (
        df["timestamp_ms"].to_numpy(dtype=np.float64) / 1000.0
    )
    raw_left: NDArray[np.float64] = df["ch1_left_uv"].to_numpy(dtype=np.float64)
    raw_right: NDArray[np.float64] = df["ch2_right_uv"].to_numpy(dtype=np.float64)

    left_processed: PreprocessedChannel = preprocess_channel(raw_left, fs=fs)
    right_processed: PreprocessedChannel = preprocess_channel(raw_right, fs=fs)

    assessment: BilateralAssessment = extract_bilateral_biomarkers(
        left_signal=left_processed.filtered,
        right_signal=right_processed.filtered,
        fs=fs,
    )

    biomarkers_path: Path = results_dir / "biomarkers.json"
    biomarkers_path.write_text(
        assessment.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    plot_path: Path = results_dir / "emg_clinical_plot.png"
    plot_emg_analysis(
        time_s=time_s,
        raw_left=raw_left,
        filt_left=left_processed.filtered,
        env_left=left_processed.rms_envelope,
        filt_right=right_processed.filtered,
        env_right=right_processed.rms_envelope,
        fs=fs,
        output_path=plot_path,
    )

    summary_path: Path = results_dir / "clinical_summary.md"
    summary_markdown: str = build_clinical_summary_markdown(
        assessment=assessment,
        plot_filename=plot_path.name,
    )
    summary_path.write_text(summary_markdown, encoding="utf-8")

    print(
        format_stdout_summary(
            assessment=assessment,
            csv_path=input_csv,
            json_path=biomarkers_path,
            plot_path=plot_path,
            markdown_path=summary_path,
        )
    )
    return assessment


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the sEMG clinical pipeline."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run the end-to-end bilateral sEMG clinical processing pipeline.",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate the synthetic 2-channel sEMG CSV even if it already exists.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Path to the 2-channel sEMG CSV file (default: {DEFAULT_CSV_PATH}).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Output directory for JSON, plot, and Markdown report (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=DEFAULT_FS_HZ,
        help=f"Sampling frequency in Hz (default: {DEFAULT_FS_HZ}).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the sEMG clinical pipeline."""
    args: argparse.Namespace = parse_args(argv)
    run_pipeline(
        input_csv=args.input,
        results_dir=args.results_dir,
        fs=args.fs,
        regenerate=args.regenerate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
