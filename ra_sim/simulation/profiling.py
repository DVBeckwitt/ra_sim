"""Profiling helpers for diffraction baseline measurements."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ra_sim.simulation.diffraction import (
    PROFILE_COUNTER_SIZE,
    PROFILE_SLOT_CALCULATE_PHI_CYCLES,
    PROFILE_SLOT_DETECTOR_PROJECTION_CALLS,
    PROFILE_SLOT_DETECTOR_PROJECTION_CYCLES,
    PROFILE_SLOT_PIXEL_DEPOSITION_CALLS,
    PROFILE_SLOT_PIXEL_DEPOSITION_CYCLES,
    PROFILE_SLOT_SOLVE_Q_CALLS,
    PROFILE_SLOT_SOLVE_Q_CYCLES,
    process_peaks_parallel_safe,
    profile_counter_labels,
    read_cycle_counter,
)


def calibrate_cycle_frequency_hz(*, samples: int = 5, sleep_s: float = 0.02) -> float:
    """Estimate CPU-cycle frequency used by ``read_cycle_counter``."""

    samples = max(1, int(samples))
    sleep_s = max(float(sleep_s), 1e-4)
    rates: list[float] = []

    for _ in range(samples):
        c0 = int(read_cycle_counter())
        t0 = time.perf_counter()
        time.sleep(sleep_s)
        c1 = int(read_cycle_counter())
        t1 = time.perf_counter()
        dt = t1 - t0
        dc = c1 - c0
        if dt > 0.0 and dc > 0:
            rates.append(dc / dt)

    if not rates:
        return 0.0
    return float(np.median(np.asarray(rates, dtype=np.float64)))


def summarize_profile_counters(
    timing_out: np.ndarray,
    *,
    cycle_frequency_hz: float,
) -> dict[str, float]:
    """Convert per-reflection cycle counters into aggregate timing metrics."""

    arr = np.asarray(timing_out, dtype=np.uint64)
    if arr.ndim != 2 or arr.shape[1] != PROFILE_COUNTER_SIZE:
        raise ValueError(
            f"timing_out must have shape (num_peaks, {PROFILE_COUNTER_SIZE}), "
            f"received {arr.shape!r}"
        )

    totals = arr.sum(axis=0, dtype=np.uint64)
    hz = float(cycle_frequency_hz)

    def _cycles_to_seconds(slot: int) -> float:
        if hz <= 0.0:
            return float("nan")
        return float(totals[slot]) / hz

    return {
        "calculate_phi_s": _cycles_to_seconds(PROFILE_SLOT_CALCULATE_PHI_CYCLES),
        "solve_q_s": _cycles_to_seconds(PROFILE_SLOT_SOLVE_Q_CYCLES),
        "solve_q_calls": float(int(totals[PROFILE_SLOT_SOLVE_Q_CALLS])),
        "detector_projection_s": _cycles_to_seconds(PROFILE_SLOT_DETECTOR_PROJECTION_CYCLES),
        "detector_projection_calls": float(int(totals[PROFILE_SLOT_DETECTOR_PROJECTION_CALLS])),
        "pixel_deposition_s": _cycles_to_seconds(PROFILE_SLOT_PIXEL_DEPOSITION_CYCLES),
        "pixel_deposition_calls": float(int(totals[PROFILE_SLOT_PIXEL_DEPOSITION_CALLS])),
    }


def profile_process_peaks(
    *process_args,
    peak_runner=process_peaks_parallel_safe,
    cycle_frequency_hz: float | None = None,
    **process_kwargs,
) -> tuple[tuple[Any, ...], dict[str, float], np.ndarray]:
    """Run ``process_peaks_parallel`` with internal profiling counters enabled."""

    if "timing_out" in process_kwargs:
        raise ValueError("profile_process_peaks manages timing_out internally")

    miller = np.asarray(process_args[0], dtype=np.float64)
    timing_out = np.zeros((miller.shape[0], PROFILE_COUNTER_SIZE), dtype=np.uint64)

    kwargs = dict(process_kwargs)
    kwargs["timing_out"] = timing_out

    start = time.perf_counter()
    result = peak_runner(*process_args, **kwargs)
    process_elapsed = time.perf_counter() - start

    hz = float(cycle_frequency_hz) if cycle_frequency_hz is not None else calibrate_cycle_frequency_hz()
    metrics = summarize_profile_counters(timing_out, cycle_frequency_hz=hz)
    metrics["process_peaks_parallel_s"] = float(process_elapsed)
    metrics["cycle_frequency_hz"] = hz

    return result, metrics, timing_out


def timed_plot_image(
    image: np.ndarray,
    *,
    output_path: str | Path | None = None,
    dpi: int = 200,
    cmap: str = "turbo",
) -> float:
    """Render and optionally save a simulated image, returning elapsed seconds."""

    start = time.perf_counter()
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    im = ax.imshow(np.asarray(image), cmap=cmap, origin="upper")
    ax.set_title("RA-SIM Baseline Reference")
    ax.set_xlabel("Pixel column")
    ax.set_ylabel("Pixel row")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=int(dpi))
    plt.close(fig)
    return float(time.perf_counter() - start)


def write_profile_report(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON profile report and return the resolved path."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: (value if isinstance(value, (str, int, float, bool)) else value)
        for key, value in payload.items()
    }
    serializable["profile_counter_labels"] = list(profile_counter_labels())
    out.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return out


def summarize_peak_statistics(hit_tables: list[Any]) -> dict[str, np.ndarray]:
    """Summarize per-reflection peak centroid/FWHM/intensity from hit tables."""

    num_peaks = len(hit_tables)
    centroid_col = np.full(num_peaks, np.nan, dtype=np.float64)
    centroid_row = np.full(num_peaks, np.nan, dtype=np.float64)
    fwhm_px = np.full(num_peaks, np.nan, dtype=np.float64)
    integrated_intensity = np.full(num_peaks, np.nan, dtype=np.float64)
    num_hits = np.zeros(num_peaks, dtype=np.int64)

    fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))

    for i, hits in enumerate(hit_tables):
        arr = np.asarray(hits, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            continue

        intensity = arr[:, 0]
        col = arr[:, 1]
        row = arr[:, 2]

        valid = np.isfinite(intensity) & np.isfinite(col) & np.isfinite(row) & (intensity > 0.0)
        if not np.any(valid):
            continue

        w = intensity[valid]
        x = col[valid]
        y = row[valid]
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            continue

        x_c = float(np.sum(w * x) / w_sum)
        y_c = float(np.sum(w * y) / w_sum)
        var_x = float(np.sum(w * (x - x_c) ** 2) / w_sum)
        var_y = float(np.sum(w * (y - y_c) ** 2) / w_sum)
        sigma_eq = np.sqrt(max(0.0, 0.5 * (var_x + var_y)))

        centroid_col[i] = x_c
        centroid_row[i] = y_c
        fwhm_px[i] = float(fwhm_factor * sigma_eq)
        integrated_intensity[i] = w_sum
        num_hits[i] = int(np.count_nonzero(valid))

    return {
        "centroid_col": centroid_col,
        "centroid_row": centroid_row,
        "fwhm_px": fwhm_px,
        "integrated_intensity": integrated_intensity,
        "num_hits": num_hits,
    }


def save_simulation_artifact(
    path: str | Path,
    *,
    image: np.ndarray,
    hit_tables: list[Any],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a compact simulation artifact used by validation comparisons."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    peak_stats = summarize_peak_statistics(hit_tables)
    metadata_json = json.dumps(metadata or {}, sort_keys=True)

    np.savez_compressed(
        out,
        image=np.asarray(image, dtype=np.float32),
        centroid_col=np.asarray(peak_stats["centroid_col"], dtype=np.float64),
        centroid_row=np.asarray(peak_stats["centroid_row"], dtype=np.float64),
        fwhm_px=np.asarray(peak_stats["fwhm_px"], dtype=np.float64),
        integrated_intensity=np.asarray(peak_stats["integrated_intensity"], dtype=np.float64),
        num_hits=np.asarray(peak_stats["num_hits"], dtype=np.int64),
        metadata_json=np.asarray(metadata_json),
    )
    return out


def load_simulation_artifact(path: str | Path) -> dict[str, np.ndarray]:
    """Load a simulation artifact saved by :func:`save_simulation_artifact`."""

    artifact_path = Path(path)
    with np.load(artifact_path, allow_pickle=False) as npz:
        return {key: np.array(npz[key]) for key in npz.files}


def compute_validation_metrics(
    baseline_artifact: dict[str, np.ndarray],
    candidate_artifact: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute baseline-vs-candidate validation metrics."""

    baseline_image = np.asarray(baseline_artifact["image"], dtype=np.float64)
    candidate_image = np.asarray(candidate_artifact["image"], dtype=np.float64)
    if baseline_image.shape != candidate_image.shape:
        raise ValueError(
            f"Image shape mismatch: baseline={baseline_image.shape}, "
            f"candidate={candidate_image.shape}"
        )

    pixel_abs = np.abs(candidate_image - baseline_image)
    pixel_residual_max = float(np.max(pixel_abs))
    pixel_residual_mean = float(np.mean(pixel_abs))

    base_col = np.asarray(baseline_artifact["centroid_col"], dtype=np.float64)
    base_row = np.asarray(baseline_artifact["centroid_row"], dtype=np.float64)
    cand_col = np.asarray(candidate_artifact["centroid_col"], dtype=np.float64)
    cand_row = np.asarray(candidate_artifact["centroid_row"], dtype=np.float64)

    if base_col.shape != cand_col.shape or base_row.shape != cand_row.shape:
        raise ValueError("Centroid array shape mismatch between baseline and candidate artifacts")

    centroid_mask = (
        np.isfinite(base_col)
        & np.isfinite(base_row)
        & np.isfinite(cand_col)
        & np.isfinite(cand_row)
    )
    centroid_shift = np.sqrt((cand_col - base_col) ** 2 + (cand_row - base_row) ** 2)
    centroid_vals = centroid_shift[centroid_mask]

    base_fwhm = np.asarray(baseline_artifact["fwhm_px"], dtype=np.float64)
    cand_fwhm = np.asarray(candidate_artifact["fwhm_px"], dtype=np.float64)
    if base_fwhm.shape != cand_fwhm.shape:
        raise ValueError("FWHM array shape mismatch between baseline and candidate artifacts")
    fwhm_mask = np.isfinite(base_fwhm) & np.isfinite(cand_fwhm) & (base_fwhm > 0.0)
    fwhm_shift_pct = 100.0 * (cand_fwhm - base_fwhm) / base_fwhm
    fwhm_vals = fwhm_shift_pct[fwhm_mask]

    base_int = np.asarray(baseline_artifact["integrated_intensity"], dtype=np.float64)
    cand_int = np.asarray(candidate_artifact["integrated_intensity"], dtype=np.float64)
    if base_int.shape != cand_int.shape:
        raise ValueError("Integrated-intensity array shape mismatch between baseline and candidate artifacts")
    intensity_mask = np.isfinite(base_int) & np.isfinite(cand_int) & (base_int > 0.0)
    intensity_shift_pct = 100.0 * (cand_int - base_int) / base_int
    intensity_vals = intensity_shift_pct[intensity_mask]

    def _mean_or_nan(values: np.ndarray) -> float:
        if values.size == 0:
            return float("nan")
        return float(np.mean(values))

    def _max_or_nan(values: np.ndarray) -> float:
        if values.size == 0:
            return float("nan")
        return float(np.max(values))

    def _max_abs_or_nan(values: np.ndarray) -> float:
        if values.size == 0:
            return float("nan")
        return float(np.max(np.abs(values)))

    def _mean_abs_or_nan(values: np.ndarray) -> float:
        if values.size == 0:
            return float("nan")
        return float(np.mean(np.abs(values)))

    return {
        "pixel_residual_max": pixel_residual_max,
        "pixel_residual_mean": pixel_residual_mean,
        "peak_centroid_shift_px_mean": _mean_or_nan(centroid_vals),
        "peak_centroid_shift_px_max": _max_or_nan(centroid_vals),
        "peak_centroid_shift_count": float(int(centroid_vals.size)),
        "fwhm_shift_pct_mean": _mean_or_nan(fwhm_vals),
        "fwhm_shift_pct_mean_abs": _mean_abs_or_nan(fwhm_vals),
        "fwhm_shift_pct_max_abs": _max_abs_or_nan(fwhm_vals),
        "fwhm_shift_count": float(int(fwhm_vals.size)),
        "integrated_intensity_shift_pct_mean": _mean_or_nan(intensity_vals),
        "integrated_intensity_shift_pct_mean_abs": _mean_abs_or_nan(intensity_vals),
        "integrated_intensity_shift_pct_max_abs": _max_abs_or_nan(intensity_vals),
        "integrated_intensity_shift_count": float(int(intensity_vals.size)),
    }
