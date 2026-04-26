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
    return (
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


def _fake_result_for_args(args):
    n_samp = int(np.asarray(args[16]).shape[0])
    return (
        np.zeros((8, 8), dtype=np.float64),
        [],
        np.zeros((1, 1, 5), dtype=np.float64),
        np.zeros(1, dtype=np.int64),
        np.zeros((1, n_samp), dtype=np.int64),
        [],
    )


def test_process_peaks_parallel_safe_can_prefer_python_runner(monkeypatch) -> None:
    args = _build_process_args(8)
    call_order: list[str] = []

    def fake_compiled(*_args, **_kwargs):
        call_order.append("compiled")
        raise AssertionError("compiled runner should be skipped")

    def fake_python(*_args, **_kwargs):
        call_order.append("python")
        return _fake_result_for_args(args)

    fake_compiled.py_func = fake_python
    monkeypatch.setattr(diffraction, "process_peaks_parallel", fake_compiled)

    diffraction.process_peaks_parallel_safe(
        *args,
        save_flag=0,
        prefer_python_runner=True,
    )

    assert call_order == ["python"]
    stats = diffraction.get_last_process_peaks_safe_stats()
    assert stats["used_python_runner"] is True


def test_process_peaks_parallel_safe_skips_last_cache_build_when_disabled(monkeypatch) -> None:
    args = _build_process_args(8)
    build_calls = 0

    def fake_kernel(*_args, **_kwargs):
        return _fake_result_for_args(args)

    def fail_build_intersection_cache(*_args, **_kwargs):
        nonlocal build_calls
        build_calls += 1
        raise AssertionError("disabled last-intersection cache should not be built")

    monkeypatch.setattr(diffraction, "_retain_last_intersection_cache", lambda: False)
    monkeypatch.setattr(diffraction, "build_intersection_cache", fail_build_intersection_cache)
    monkeypatch.setattr(diffraction, "build_branch_representative_intersection_cache", fail_build_intersection_cache)
    monkeypatch.setattr(diffraction, "process_peaks_parallel", fake_kernel)

    diffraction.process_peaks_parallel_safe(
        *args,
        save_flag=0,
    )

    assert build_calls == 0
    assert diffraction.get_last_intersection_cache() == []


def test_process_peaks_parallel_safe_injects_default_solve_q_trig_kwargs(monkeypatch) -> None:
    args = _build_process_args(8)
    observed: dict[str, object] = {}

    def fake_kernel(*_args, **kwargs):
        observed["default_solve_q_dtheta"] = kwargs.get("default_solve_q_dtheta")
        observed["default_solve_q_cos"] = kwargs.get("default_solve_q_cos")
        observed["default_solve_q_sin"] = kwargs.get("default_solve_q_sin")
        observed["numba_thread_count"] = kwargs.get("numba_thread_count")
        return _fake_result_for_args(args)

    monkeypatch.setattr(diffraction, "process_peaks_parallel", fake_kernel)

    diffraction.process_peaks_parallel_safe(
        *args,
        save_flag=0,
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


def test_process_peaks_parallel_safe_omitted_events_forwards_default_fifty(monkeypatch):
    args = _build_process_args(8)
    seen = {}

    def fake_kernel(*kernel_args, **kwargs):
        seen["n_samp"] = int(np.asarray(kernel_args[16]).shape[0])
        seen["events_per_beam_phase"] = kwargs.get("events_per_beam_phase")
        return _fake_result_for_args(args)

    monkeypatch.setattr(diffraction, "process_peaks_parallel", fake_kernel)
    diffraction.process_peaks_parallel_safe(*args, save_flag=0)
    assert seen["n_samp"] == int(np.asarray(args[16]).shape[0])
    assert seen["events_per_beam_phase"] == 50


def test_process_peaks_parallel_safe_zero_events_normalizes_to_one(monkeypatch):
    args = _build_process_args(8)
    seen = {}

    def fake_kernel(*_args, **kwargs):
        seen["events_per_beam_phase"] = kwargs.get("events_per_beam_phase")
        return _fake_result_for_args(args)

    monkeypatch.setattr(diffraction, "process_peaks_parallel", fake_kernel)
    diffraction.process_peaks_parallel_safe(*args, save_flag=0, events_per_beam_phase=0)
    assert seen["events_per_beam_phase"] == 1


def test_process_peaks_parallel_safe_negative_events_normalizes_to_one(monkeypatch):
    args = _build_process_args(8)
    seen = {}

    def fake_kernel(*_args, **kwargs):
        seen["events_per_beam_phase"] = kwargs.get("events_per_beam_phase")
        return _fake_result_for_args(args)

    monkeypatch.setattr(diffraction, "process_peaks_parallel", fake_kernel)
    diffraction.process_peaks_parallel_safe(*args, save_flag=0, events_per_beam_phase=-5)
    assert seen["events_per_beam_phase"] == 1


def test_normalize_events_per_beam_phase_backend_clamps_upper_bound():
    assert diffraction.normalize_events_per_beam_phase_backend(999999) == 1000


def test_process_peaks_parallel_safe_does_not_cluster_beam_replacement(monkeypatch):
    args = _build_process_args(128)
    seen = {}

    def fake_kernel(*kernel_args, **kwargs):
        seen["n_samp"] = int(np.asarray(kernel_args[16]).shape[0])
        seen["events_per_beam_phase"] = kwargs.get("events_per_beam_phase")
        return _fake_result_for_args(args)

    monkeypatch.setattr(diffraction, "process_peaks_parallel", fake_kernel)
    diffraction.process_peaks_parallel_safe(*args, save_flag=0, events_per_beam_phase=50)
    assert seen["n_samp"] == int(np.asarray(args[16]).shape[0])
    assert seen["events_per_beam_phase"] == 50


def test_process_peaks_parallel_safe_bypasses_source_template_cache(monkeypatch):
    args = _build_process_args(8)
    called = 0

    def fail_build_phase(_params):
        raise AssertionError("weighted event path must not build source-template cache")

    def fake_kernel(*_args, **_kwargs):
        nonlocal called
        called += 1
        return _fake_result_for_args(args)

    monkeypatch.setattr(diffraction, "_build_phase_space_entry", fail_build_phase)
    monkeypatch.setattr(diffraction, "process_peaks_parallel", fake_kernel)

    diffraction.process_peaks_parallel_safe(
        *args,
        save_flag=0,
        enable_safe_cache=True,
        events_per_beam_phase=50,
    )

    assert called == 1
    assert diffraction.get_last_process_peaks_safe_stats()["used_safe_cache"] is False


def test_process_peaks_parallel_py_func_forwards_normalized_events(monkeypatch):
    args = _build_process_args(8)
    seen = {}

    def fake_weighted(*_args, **kwargs):
        seen["events_per_beam_phase"] = kwargs.get("events_per_beam_phase")
        return (
            np.zeros((8, 8), dtype=np.float64),
            [],
            np.zeros((1, 1, 5), dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            np.zeros((1, int(np.asarray(args[16]).shape[0])), dtype=np.int64),
            [],
            [np.empty((0, diffraction.HIT_ROW_WITH_PROVENANCE_WIDTH), dtype=np.float64)],
        )

    monkeypatch.setattr(diffraction, "_process_peaks_parallel_weighted_events_python", fake_weighted)
    diffraction.process_peaks_parallel.py_func(*args, save_flag=0, events_per_beam_phase=0)
    assert seen["events_per_beam_phase"] == 1


def test_process_peaks_parallel_safe_updates_last_cache_with_sampled_and_representative_rows(monkeypatch):
    args = _build_process_args(8)
    monkeypatch.setattr(diffraction, "_retain_last_intersection_cache", lambda: True)

    def fake_kernel(*_args, **_kwargs):
        diffraction._set_last_process_peaks_representative_hit_tables(
            [
                np.array(
                    [[1.0, 10.0, 11.0, -1.0, 1.0, 0.0, 1.0, np.nan, np.nan, 0.0]],
                    dtype=np.float64,
                )
            ]
        )
        return (
            np.zeros((8, 8), dtype=np.float64),
            [
                np.array(
                    [
                        [0.5, 10.0, 11.0, -1.0, 1.0, 0.0, 1.0, np.nan, np.nan, 0.0],
                        [0.5, 10.0, 11.0, -1.0, 1.0, 0.0, 1.0, np.nan, np.nan, 0.0],
                    ],
                    dtype=np.float64,
                )
            ],
            np.zeros((1, 1, 5), dtype=np.float64),
            np.zeros(1, dtype=np.int64),
            np.zeros((1, int(np.asarray(args[16]).shape[0])), dtype=np.int64),
            [np.empty((0, 3), dtype=np.float64)],
        )

    monkeypatch.setattr(diffraction, "process_peaks_parallel", fake_kernel)
    diffraction.process_peaks_parallel_safe(*args, save_flag=0, events_per_beam_phase=2)

    views = diffraction.get_last_intersection_cache_views()
    assert len(views["sampled_event_rows"]) == 2
    assert len(views["branch_representative_rows"]) == 1
    sampled = np.asarray(views["sampled_event_rows"][0], dtype=np.float64)
    representative = np.asarray(views["branch_representative_rows"][0], dtype=np.float64)
    assert sampled.shape == (1, 17)
    assert representative.shape == (1, 17)
    assert float(sampled[0, 4]) == 0.5
    assert float(representative[0, 4]) == 1.0
    representative_only = diffraction.get_last_intersection_cache()
    assert len(representative_only) == 1
    np.testing.assert_allclose(
        np.asarray(representative_only[0], dtype=np.float64),
        representative,
    )


def test_process_peaks_parallel_safe_keeps_weighted_events_when_sample_qr_ring_once_true(monkeypatch):
    args = _build_process_args(8)
    seen = {}

    def fake_kernel(*_args, **kwargs):
        seen["sample_qr_ring_once"] = kwargs.get("sample_qr_ring_once")
        seen["events_per_beam_phase"] = kwargs.get("events_per_beam_phase")
        return _fake_result_for_args(args)

    monkeypatch.setattr(diffraction, "process_peaks_parallel", fake_kernel)
    diffraction.process_peaks_parallel_safe(
        *args,
        save_flag=0,
        sample_qr_ring_once=True,
        events_per_beam_phase=50,
    )
    assert seen["sample_qr_ring_once"] is True
    assert seen["events_per_beam_phase"] == 50
