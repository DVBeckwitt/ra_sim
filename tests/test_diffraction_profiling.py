from __future__ import annotations

import numpy as np

from ra_sim.simulation import diffraction
from ra_sim.simulation.profiling import (
    compute_validation_metrics,
    profile_process_peaks,
    summarize_peak_statistics,
)


def test_process_peaks_parallel_py_func_populates_timing_out(monkeypatch) -> None:
    expected = np.array([10, 20, 1, 30, 2, 40, 3], dtype=np.uint64)

    def fake_calculate_phi(H, K, L, *args, **kwargs):
        profile_counters = args[-1]
        if profile_counters is not None:
            profile_counters[:] = expected
        return (
            np.empty((0, 7), dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            0,
        )

    monkeypatch.setattr(diffraction, "calculate_phi", fake_calculate_phi)

    miller = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    intensities = np.array([1.0, 1.0], dtype=np.float64)
    image_size = 8
    timing_out = np.zeros((miller.shape[0], diffraction.PROFILE_COUNTER_SIZE), dtype=np.uint64)

    diffraction.process_peaks_parallel.py_func(
        miller,
        intensities,
        image_size,
        1.0,
        1.0,
        1.0,
        np.zeros((image_size, image_size), dtype=np.float64),
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 + 0.0j,
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        0.5,
        0.5,
        0.0,
        np.array([1.0], dtype=np.float64),
        0.0,
        0.0,
        np.array([4.0, 4.0], dtype=np.float64),
        6.0,
        0.0,
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        save_flag=0,
        timing_out=timing_out,
    )

    assert np.all(timing_out[0] == 0)
    assert np.array_equal(timing_out[1], expected)


def test_profile_process_peaks_reports_expected_metrics() -> None:
    counters = np.array([100, 200, 2, 300, 4, 400, 5], dtype=np.uint64)

    def fake_runner(*args, timing_out=None, **kwargs):
        assert timing_out is not None
        timing_out[:] = counters
        return (
            np.zeros((2, 2), dtype=np.float64),
            [],
            np.zeros((1, 1, 5), dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            np.zeros((1, 1), dtype=np.int64),
            [],
        )

    miller = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    result, metrics, timing_out = profile_process_peaks(
        miller,
        np.array([1.0], dtype=np.float64),
        peak_runner=fake_runner,
        cycle_frequency_hz=1000.0,
    )

    assert np.array_equal(timing_out[0], counters)
    assert np.isclose(metrics["calculate_phi_s"], 0.1)
    assert np.isclose(metrics["solve_q_s"], 0.2)
    assert metrics["solve_q_calls"] == 2.0
    assert np.isclose(metrics["detector_projection_s"], 0.3)
    assert metrics["detector_projection_calls"] == 4.0
    assert np.isclose(metrics["pixel_deposition_s"], 0.4)
    assert metrics["pixel_deposition_calls"] == 5.0
    assert metrics["process_peaks_parallel_s"] >= 0.0
    assert isinstance(result, tuple)


def test_summarize_peak_statistics_computes_centroid_fwhm_and_intensity() -> None:
    hit_tables = [
        np.array(
            [
                [2.0, 10.0, 20.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 12.0, 20.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 10.0, 22.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        np.empty((0, 7), dtype=np.float64),
    ]

    summary = summarize_peak_statistics(hit_tables)

    assert np.isclose(summary["centroid_col"][0], 10.5)
    assert np.isclose(summary["centroid_row"][0], 20.5)
    assert np.isclose(summary["fwhm_px"][0], 2.0393339803376177)
    assert np.isclose(summary["integrated_intensity"][0], 4.0)
    assert summary["num_hits"][0] == 3
    assert np.isnan(summary["centroid_col"][1])
    assert np.isnan(summary["fwhm_px"][1])
    assert np.isnan(summary["integrated_intensity"][1])
    assert summary["num_hits"][1] == 0


def test_compute_validation_metrics_reports_requested_shifts() -> None:
    baseline = {
        "image": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "centroid_col": np.array([10.0, 20.0], dtype=np.float64),
        "centroid_row": np.array([10.0, 20.0], dtype=np.float64),
        "fwhm_px": np.array([2.0, 4.0], dtype=np.float64),
        "integrated_intensity": np.array([100.0, 50.0], dtype=np.float64),
    }
    candidate = {
        "image": np.array([[2.0, 4.0], [1.0, 5.0]], dtype=np.float32),
        "centroid_col": np.array([11.0, 22.0], dtype=np.float64),
        "centroid_row": np.array([10.0, 18.0], dtype=np.float64),
        "fwhm_px": np.array([2.2, 3.6], dtype=np.float64),
        "integrated_intensity": np.array([110.0, 40.0], dtype=np.float64),
    }

    metrics = compute_validation_metrics(baseline, candidate)

    assert np.isclose(metrics["pixel_residual_max"], 2.0)
    assert np.isclose(metrics["pixel_residual_mean"], 1.5)
    assert np.isclose(metrics["peak_centroid_shift_px_mean"], 1.9142135623730951)
    assert np.isclose(metrics["peak_centroid_shift_px_max"], 2.8284271247461903)
    assert metrics["peak_centroid_shift_count"] == 2.0
    assert np.isclose(metrics["fwhm_shift_pct_mean"], 0.0)
    assert np.isclose(metrics["fwhm_shift_pct_mean_abs"], 10.0)
    assert np.isclose(metrics["fwhm_shift_pct_max_abs"], 10.0)
    assert metrics["fwhm_shift_count"] == 2.0
    assert np.isclose(metrics["integrated_intensity_shift_pct_mean"], -5.0)
    assert np.isclose(metrics["integrated_intensity_shift_pct_mean_abs"], 15.0)
    assert np.isclose(metrics["integrated_intensity_shift_pct_max_abs"], 20.0)
    assert metrics["integrated_intensity_shift_count"] == 2.0
