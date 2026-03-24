from __future__ import annotations

import numpy as np

from ra_sim.simulation import diffraction
from ra_sim.simulation.mosaic_profiles import generate_random_profiles


def _build_process_args(num_samples: int):
    beam_x, beam_y, theta, phi, wavelength = generate_random_profiles(
        num_samples,
        divergence_sigma=0.02,
        bw_sigma=0.004,
        lambda0=1.54,
        bandwidth=0.006,
        rng=np.random.default_rng(21),
    )
    image_size = 8
    image = np.zeros((image_size, image_size), dtype=np.float64)
    center = np.array([image_size / 2.0, image_size / 2.0], dtype=np.float64)
    unit_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    n_detector = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    miller = np.array([[1.0, 0.0, 1.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    args = (
        miller,
        intensities,
        image_size,
        1.0,
        1.0,
        1.54,
        image,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 + 0.0j,
        beam_x,
        beam_y,
        theta,
        phi,
        0.5,
        0.5,
        0.0,
        wavelength,
        0.0,
        0.0,
        center,
        6.0,
        0.0,
        unit_x,
        n_detector,
    )
    return args


def test_process_peaks_parallel_safe_clusters_and_expands_outputs(monkeypatch) -> None:
    args = _build_process_args(128)
    best_indices = np.full(1, -1, dtype=np.int64)
    observed: dict[str, np.ndarray | int] = {}

    def fake_kernel(*kernel_args, **kernel_kwargs):
        clustered_n = int(np.asarray(kernel_args[16]).shape[0])
        observed["clustered_n"] = clustered_n
        observed["sample_weights"] = np.asarray(kernel_kwargs["sample_weights"], dtype=np.float64)
        best_out = kernel_kwargs.get("best_sample_indices_out")
        if isinstance(best_out, np.ndarray):
            best_out[:] = 0
        status = np.arange(clustered_n, dtype=np.int64).reshape(1, clustered_n)
        return (
            np.zeros((8, 8), dtype=np.float64),
            [],
            np.zeros((1, 1, 5), dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            status,
            [],
        )

    monkeypatch.setattr(diffraction, "process_peaks_parallel", fake_kernel)

    result = diffraction.process_peaks_parallel_safe(
        *args,
        save_flag=0,
        record_status=True,
        best_sample_indices_out=best_indices,
    )

    raw_n = int(np.asarray(args[16]).shape[0])
    clustered_n = int(observed["clustered_n"])
    assert clustered_n < raw_n
    np.testing.assert_allclose(np.sum(observed["sample_weights"]), float(raw_n))
    assert result[4].shape == (1, raw_n)
    assert np.all(result[4] >= 0)
    assert np.all(result[4] < clustered_n)
    assert 0 <= int(best_indices[0]) < raw_n


def test_process_peaks_parallel_safe_keeps_raw_samples_for_forced_indices(monkeypatch) -> None:
    args = _build_process_args(128)
    observed: dict[str, object] = {}

    def fake_kernel(*kernel_args, **kernel_kwargs):
        observed["sample_count"] = int(np.asarray(kernel_args[16]).shape[0])
        observed["has_sample_weights"] = "sample_weights" in kernel_kwargs
        return (
            np.zeros((8, 8), dtype=np.float64),
            [],
            np.zeros((1, 1, 5), dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            np.zeros((1, int(np.asarray(kernel_args[16]).shape[0])), dtype=np.int64),
            [],
        )

    monkeypatch.setattr(diffraction, "process_peaks_parallel", fake_kernel)

    diffraction.process_peaks_parallel_safe(
        *args,
        save_flag=0,
        single_sample_indices=np.array([17], dtype=np.int64),
    )

    assert observed["sample_count"] == int(np.asarray(args[16]).shape[0])
    assert observed["has_sample_weights"] is False


def test_process_peaks_parallel_safe_can_prefer_python_runner(monkeypatch) -> None:
    args = _build_process_args(8)
    call_order: list[str] = []

    def fake_compiled(*_args, **_kwargs):
        call_order.append("compiled")
        raise AssertionError("compiled runner should be skipped")

    def fake_python(*_args, **_kwargs):
        call_order.append("python")
        return (
            np.zeros((8, 8), dtype=np.float64),
            [],
            np.zeros((1, 1, 5), dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            np.zeros((1, int(np.asarray(args[16]).shape[0])), dtype=np.int64),
            [],
        )

    fake_compiled.py_func = fake_python
    monkeypatch.setattr(diffraction, "process_peaks_parallel", fake_compiled)

    diffraction.process_peaks_parallel_safe(
        *args,
        save_flag=0,
        prefer_python_runner=True,
    )

    assert call_order == ["python"]
