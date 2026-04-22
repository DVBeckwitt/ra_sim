from __future__ import annotations

import warnings

import numpy as np
import pytest

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


def _build_qr_process_args() -> tuple[object, ...]:
    return (
        {1: {"hk": (1, 0), "L": np.array([0.0]), "I": np.array([1.0]), "deg": 1}},
        8,
        4.0,
        7.0,
        1.54,
        np.zeros((8, 8), dtype=np.float64),
        0.1,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 + 0.0j,
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        0.5,
        0.4,
        0.2,
        np.ones(1, dtype=np.float64),
        0.0,
        0.0,
        np.array([4.0, 4.0], dtype=np.float64),
        0.0,
        0.0,
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        0,
    )


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
        sample_qr_ring_once=False,
    )

    raw_n = int(np.asarray(args[16]).shape[0])
    clustered_n = int(observed["clustered_n"])
    assert clustered_n < raw_n
    np.testing.assert_allclose(np.sum(observed["sample_weights"]), float(raw_n))
    assert result[4].shape == (1, raw_n)
    assert np.all(result[4] >= 0)
    assert np.all(result[4] < clustered_n)
    assert 0 <= int(best_indices[0]) < raw_n


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
        sample_qr_ring_once=False,
    )

    assert call_order == ["python"]
    stats = diffraction.get_last_process_peaks_safe_stats()
    assert stats["used_python_runner"] is True


def test_process_peaks_parallel_safe_skips_last_cache_build_on_runner_path(
    monkeypatch,
) -> None:
    args = _build_process_args(8)
    build_calls = 0

    def fake_kernel(*_args, **_kwargs):
        return (
            np.zeros((8, 8), dtype=np.float64),
            [],
            np.zeros((1, 1, 5), dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            np.zeros((1, int(np.asarray(args[16]).shape[0])), dtype=np.int64),
            [],
        )

    def fail_build_intersection_cache(*_args, **_kwargs):
        nonlocal build_calls
        build_calls += 1
        raise AssertionError("disabled last-intersection cache should not be built")

    monkeypatch.setattr(diffraction, "_retain_last_intersection_cache", lambda: False)
    monkeypatch.setattr(diffraction, "build_intersection_cache", fail_build_intersection_cache)
    monkeypatch.setattr(diffraction, "process_peaks_parallel", fake_kernel)

    diffraction.process_peaks_parallel_safe(
        *args,
        save_flag=0,
        enable_safe_cache=False,
        sample_qr_ring_once=False,
    )

    assert build_calls == 0
    assert diffraction.get_last_intersection_cache() == []


def test_process_peaks_parallel_safe_injects_default_solve_q_trig_kwargs(
    monkeypatch,
) -> None:
    args = _build_process_args(8)
    observed: dict[str, object] = {}

    def fake_kernel(*_args, **kwargs):
        observed["default_solve_q_dtheta"] = kwargs.get("default_solve_q_dtheta")
        observed["default_solve_q_cos"] = kwargs.get("default_solve_q_cos")
        observed["default_solve_q_sin"] = kwargs.get("default_solve_q_sin")
        observed["numba_thread_count"] = kwargs.get("numba_thread_count")
        return (
            np.zeros((8, 8), dtype=np.float64),
            [],
            np.zeros((1, 1, 5), dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            np.zeros((1, int(np.asarray(args[16]).shape[0])), dtype=np.int64),
            [],
        )

    monkeypatch.setattr(diffraction, "process_peaks_parallel", fake_kernel)

    diffraction.process_peaks_parallel_safe(
        *args,
        save_flag=0,
        sample_qr_ring_once=False,
    )

    expected = diffraction.get_default_solve_q_trig_kwargs()
    assert observed["default_solve_q_dtheta"] == expected["default_solve_q_dtheta"]
    np.testing.assert_allclose(
        np.asarray(observed["default_solve_q_cos"], dtype=np.float64),
        expected["default_solve_q_cos"],
    )
    np.testing.assert_allclose(
        np.asarray(observed["default_solve_q_sin"], dtype=np.float64),
        expected["default_solve_q_sin"],
    )
    assert int(observed["numba_thread_count"]) >= 1


def test_get_default_solve_q_trig_kwargs_returns_readonly_views() -> None:
    default_kwargs = diffraction.get_default_solve_q_trig_kwargs()
    cos_view = np.asarray(default_kwargs["default_solve_q_cos"], dtype=np.float64)
    sin_view = np.asarray(default_kwargs["default_solve_q_sin"], dtype=np.float64)

    assert not np.shares_memory(
        diffraction._CANONICAL_DEFAULT_SOLVE_Q_COS,
        diffraction._DEFAULT_SOLVE_Q_COS,
    )
    assert not np.shares_memory(
        diffraction._CANONICAL_DEFAULT_SOLVE_Q_SIN,
        diffraction._DEFAULT_SOLVE_Q_SIN,
    )
    assert not np.shares_memory(cos_view, diffraction._DEFAULT_SOLVE_Q_COS)
    assert not np.shares_memory(sin_view, diffraction._DEFAULT_SOLVE_Q_SIN)
    assert cos_view.flags.writeable is False
    assert sin_view.flags.writeable is False

    with pytest.raises(ValueError):
        cos_view[0] = cos_view[0] + 1.0

    fresh_kwargs = diffraction.get_default_solve_q_trig_kwargs()
    np.testing.assert_allclose(
        np.asarray(fresh_kwargs["default_solve_q_cos"], dtype=np.float64),
        np.asarray(default_kwargs["default_solve_q_cos"], dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.asarray(fresh_kwargs["default_solve_q_sin"], dtype=np.float64),
        np.asarray(default_kwargs["default_solve_q_sin"], dtype=np.float64),
    )


def test_process_peaks_parallel_wrapper_injects_runtime_kwargs(monkeypatch) -> None:
    args = _build_process_args(8)
    observed: dict[str, object] = {}

    def fake_impl(*_args, **kwargs):
        observed["default_solve_q_dtheta"] = kwargs.get("default_solve_q_dtheta")
        observed["default_solve_q_cos"] = kwargs.get("default_solve_q_cos")
        observed["default_solve_q_sin"] = kwargs.get("default_solve_q_sin")
        observed["numba_thread_count"] = kwargs.get("numba_thread_count")
        return (
            np.zeros((8, 8), dtype=np.float64),
            [],
            np.zeros((1, 1, 5), dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            np.zeros((1, int(np.asarray(args[16]).shape[0])), dtype=np.int64),
            [],
        )

    monkeypatch.setattr(diffraction, "_process_peaks_parallel_impl", fake_impl)

    diffraction.process_peaks_parallel(
        *args,
        save_flag=0,
        sample_qr_ring_once=False,
    )

    expected = diffraction.get_default_solve_q_trig_kwargs()
    assert observed["default_solve_q_dtheta"] == expected["default_solve_q_dtheta"]
    np.testing.assert_allclose(
        np.asarray(observed["default_solve_q_cos"], dtype=np.float64),
        expected["default_solve_q_cos"],
    )
    np.testing.assert_allclose(
        np.asarray(observed["default_solve_q_sin"], dtype=np.float64),
        expected["default_solve_q_sin"],
    )
    assert int(observed["numba_thread_count"]) >= 1


def test_process_peaks_parallel_py_func_injects_runtime_kwargs(monkeypatch) -> None:
    args = _build_process_args(8)
    observed: dict[str, object] = {}

    def fake_python_impl(*_args, **kwargs):
        observed["default_solve_q_dtheta"] = kwargs.get("default_solve_q_dtheta")
        observed["default_solve_q_cos"] = kwargs.get("default_solve_q_cos")
        observed["default_solve_q_sin"] = kwargs.get("default_solve_q_sin")
        observed["numba_thread_count"] = kwargs.get("numba_thread_count")
        return (
            np.zeros((8, 8), dtype=np.float64),
            [],
            np.zeros((1, 1, 5), dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            np.zeros((1, int(np.asarray(args[16]).shape[0])), dtype=np.int64),
            [],
        )

    class DummyImpl:
        py_func = staticmethod(fake_python_impl)

    monkeypatch.setattr(diffraction, "_process_peaks_parallel_impl", DummyImpl())

    diffraction.process_peaks_parallel.py_func(
        *args,
        save_flag=0,
        sample_qr_ring_once=False,
    )

    expected = diffraction.get_default_solve_q_trig_kwargs()
    assert observed["default_solve_q_dtheta"] == expected["default_solve_q_dtheta"]
    np.testing.assert_allclose(
        np.asarray(observed["default_solve_q_cos"], dtype=np.float64),
        expected["default_solve_q_cos"],
    )
    np.testing.assert_allclose(
        np.asarray(observed["default_solve_q_sin"], dtype=np.float64),
        expected["default_solve_q_sin"],
    )
    assert int(observed["numba_thread_count"]) >= 1


def test_process_peaks_parallel_safe_strips_runtime_kwargs_for_older_runner(
    monkeypatch,
) -> None:
    args = _build_process_args(8)
    runtime_kwargs = diffraction.get_process_peaks_runtime_kwargs(numba_thread_count=3)
    observed: dict[str, object] = {}

    def fake_old_runner(
        *kernel_args,
        save_flag=0,
        sample_qr_ring_once=True,
    ):
        observed["sample_qr_ring_once"] = sample_qr_ring_once
        return (
            np.zeros((8, 8), dtype=np.float64),
            [],
            np.zeros((1, 1, 5), dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            np.zeros((1, int(np.asarray(kernel_args[16]).shape[0])), dtype=np.int64),
            [],
        )

    monkeypatch.setattr(diffraction, "process_peaks_parallel", fake_old_runner)

    diffraction.process_peaks_parallel_safe(
        *args,
        save_flag=0,
        sample_qr_ring_once=False,
        **runtime_kwargs,
    )

    assert observed["sample_qr_ring_once"] is False


def test_process_peaks_parallel_safe_strips_runtime_kwargs_for_older_python_runner(
    monkeypatch,
) -> None:
    args = _build_process_args(8)
    runtime_kwargs = diffraction.get_process_peaks_runtime_kwargs(numba_thread_count=5)
    call_order: list[str] = []

    def fake_compiled(*_args, **_kwargs):
        call_order.append("compiled")
        raise AssertionError("compiled runner should be skipped")

    def fake_old_python(
        *kernel_args,
        save_flag=0,
        sample_qr_ring_once=True,
    ):
        call_order.append("python")
        assert sample_qr_ring_once is False
        return (
            np.zeros((8, 8), dtype=np.float64),
            [],
            np.zeros((1, 1, 5), dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            np.zeros((1, int(np.asarray(kernel_args[16]).shape[0])), dtype=np.int64),
            [],
        )

    fake_compiled.py_func = fake_old_python
    monkeypatch.setattr(diffraction, "process_peaks_parallel", fake_compiled)

    diffraction.process_peaks_parallel_safe(
        *args,
        save_flag=0,
        prefer_python_runner=True,
        sample_qr_ring_once=False,
        **runtime_kwargs,
    )

    assert call_order == ["python"]


def test_process_peaks_parallel_safe_keeps_raw_beams_for_q_ring_sampling(monkeypatch) -> None:
    args = _build_process_args(128)
    observed: dict[str, object] = {}

    def fake_kernel(*kernel_args, **kernel_kwargs):
        observed["sample_count"] = int(np.asarray(kernel_args[16]).shape[0])
        observed["has_sample_weights"] = "sample_weights" in kernel_kwargs
        observed["sample_qr_ring_once"] = kernel_kwargs.get("sample_qr_ring_once")
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
    )

    assert observed["sample_count"] == int(np.asarray(args[16]).shape[0])
    assert observed["has_sample_weights"] is False
    assert observed["sample_qr_ring_once"] is True


def test_process_peaks_parallel_compiles_without_dynamic_global_cache_warning() -> None:
    args = _build_process_args(1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diffraction.process_peaks_parallel(
            *args,
            save_flag=0,
            sample_qr_ring_once=False,
        )

    assert not any(
        "Cannot cache compiled function" in str(w.message) and "dynamic globals" in str(w.message)
        for w in caught
    )


def test_process_qr_rods_parallel_safe_accepts_enable_safe_cache(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_qr_dict_to_arrays(_qr_dict):
        return (
            np.array([[1.0, 0.0, 1.0]], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            np.array([1], dtype=np.int32),
            None,
        )

    def fake_process_peaks_parallel_safe(*args, **kwargs):
        observed["enable_safe_cache"] = kwargs.get("enable_safe_cache")
        return (
            np.zeros((8, 8), dtype=np.float64),
            [],
            np.zeros((1, 1, 5), dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            np.zeros((1, 1), dtype=np.int64),
            [],
        )

    monkeypatch.setattr(
        "ra_sim.utils.stacking_fault.qr_dict_to_arrays",
        fake_qr_dict_to_arrays,
    )
    monkeypatch.setattr(
        diffraction,
        "process_peaks_parallel_safe",
        fake_process_peaks_parallel_safe,
    )

    result = diffraction.process_qr_rods_parallel_safe(
        *_build_qr_process_args(),
        enable_safe_cache=True,
    )

    assert observed["enable_safe_cache"] is True
    assert np.array_equal(result[-1], np.array([1], dtype=np.int32))


def test_process_qr_rods_parallel_safe_can_prefer_python_runner(monkeypatch) -> None:
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
            np.zeros((1, 1), dtype=np.int64),
            [],
            np.array([1], dtype=np.int32),
        )

    fake_compiled.py_func = fake_python
    monkeypatch.setattr(diffraction, "process_qr_rods_parallel", fake_compiled)

    result = diffraction.process_qr_rods_parallel_safe(
        *_build_qr_process_args(),
        prefer_python_runner=True,
    )

    assert call_order == ["python"]
    assert np.array_equal(result[-1], np.array([1], dtype=np.int32))


def test_process_qr_rods_parallel_safe_forwards_safe_runner_stats(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_qr_dict_to_arrays(_qr_dict):
        return (
            np.array([[1.0, 0.0, 1.0]], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            np.array([1], dtype=np.int32),
            None,
        )

    def fake_process_peaks_parallel_safe(*args, **kwargs):
        del args
        observed["prefer_python_runner"] = kwargs.get("prefer_python_runner")
        safe_stats_out = kwargs.get("_safe_stats_out")
        if isinstance(safe_stats_out, dict):
            safe_stats_out["used_python_runner"] = True
        return (
            np.zeros((8, 8), dtype=np.float64),
            [],
            np.zeros((1, 1, 5), dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            np.zeros((1, 1), dtype=np.int64),
            [],
        )

    monkeypatch.setattr(
        "ra_sim.utils.stacking_fault.qr_dict_to_arrays",
        fake_qr_dict_to_arrays,
    )
    monkeypatch.setattr(
        diffraction,
        "process_peaks_parallel_safe",
        fake_process_peaks_parallel_safe,
    )

    safe_stats: dict[str, object] = {}
    result = diffraction.process_qr_rods_parallel_safe(
        *_build_qr_process_args(),
        prefer_python_runner=True,
        _safe_stats_out=safe_stats,
    )

    assert observed["prefer_python_runner"] is True
    assert safe_stats["used_python_runner"] is True
    assert np.array_equal(result[-1], np.array([1], dtype=np.int32))
