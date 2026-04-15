from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading

import numpy as np
import pytest

from ra_sim.simulation import engine
from ra_sim.simulation.engine import simulate, simulate_qr_rods
from ra_sim.simulation.types import (
    BeamSamples,
    DebyeWallerParams,
    DetectorGeometry,
    MosaicParams,
    SimulationRequest,
)


def _build_request(image_size: int = 8) -> SimulationRequest:
    return SimulationRequest(
        miller=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        geometry=DetectorGeometry(
            image_size=image_size,
            av=4.0,
            cv=7.0,
            lambda_angstrom=1.54,
            distance_m=0.1,
            gamma_deg=0.0,
            Gamma_deg=0.0,
            chi_deg=0.0,
            psi_deg=0.0,
            psi_z_deg=0.0,
            zs=0.0,
            zb=0.0,
            center=np.array([image_size / 2.0, image_size / 2.0], dtype=np.float64),
            theta_initial_deg=0.0,
            cor_angle_deg=0.0,
            unit_x=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            n_detector=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        ),
        beam=BeamSamples(
            beam_x_array=np.zeros(1, dtype=np.float64),
            beam_y_array=np.zeros(1, dtype=np.float64),
            theta_array=np.zeros(1, dtype=np.float64),
            phi_array=np.zeros(1, dtype=np.float64),
            wavelength_array=np.ones(1, dtype=np.float64),
        ),
        mosaic=MosaicParams(
            sigma_mosaic_deg=0.5,
            gamma_mosaic_deg=0.4,
            eta=0.2,
        ),
        debye_waller=DebyeWallerParams(x=0.0, y=0.0),
        n2=1.0 + 0.0j,
        image_buffer=np.zeros((image_size, image_size), dtype=np.float64),
    )


def _fake_simulation_result(request: SimulationRequest):
    image_size = int(request.geometry.image_size)
    return engine.SimulationResult(
        image=np.zeros((image_size, image_size), dtype=np.float64),
        hit_tables=[],
        q_data=np.empty(0, dtype=np.float64),
        q_count=np.empty(0, dtype=np.float64),
        all_status=np.empty(0, dtype=np.float64),
        miss_tables=[],
    )


def _set_forward_safe_run_result(used_python_runner: bool | None) -> None:
    engine._set_last_forward_simulation_safe_run_used_python_runner(
        used_python_runner
    )


def test_simulate_respects_typed_request_with_custom_runner() -> None:
    request = _build_request()

    def fake_runner(*args, **kwargs):
        image = np.array(args[6], copy=True)
        image += 2.0
        return (
            image,
            [np.array([[1, 2, 3]], dtype=np.float64)],
            np.array([1.0], dtype=np.float64),
            np.array([2.0], dtype=np.float64),
            np.array([3.0], dtype=np.float64),
            [np.empty((0, 3), dtype=np.float64)],
        )

    result = simulate(request, peak_runner=fake_runner)
    assert np.allclose(result.image, 2.0)
    assert result.degeneracy is None
    assert result.q_data.shape == (1,)


def test_simulate_forwards_extended_kernel_options() -> None:
    request = _build_request()
    request.optics_mode = 7
    request.collect_hit_tables = False
    request.accumulate_image = False
    request.single_sample_indices = np.array([0], dtype=np.int64)
    request.best_sample_indices_out = np.array([-1], dtype=np.int64)
    request.geometry.pixel_size_m = 172e-6
    request.geometry.sample_width_m = 2.5e-3
    request.geometry.sample_length_m = 4.0e-3
    request.beam.n2_sample_array = np.array([1.0 + 0.1j], dtype=np.complex128)
    seen: dict[str, object] = {}

    def fake_runner(*args, **kwargs):
        seen.update(kwargs)
        image = np.array(args[6], copy=True)
        return (
            image,
            [np.array([[1, 2, 3]], dtype=np.float64)],
            np.array([1.0], dtype=np.float64),
            np.array([2.0], dtype=np.float64),
            np.array([3.0], dtype=np.float64),
            [np.empty((0, 3), dtype=np.float64)],
        )

    simulate(request, peak_runner=fake_runner)

    assert seen["optics_mode"] == 7
    assert seen["collect_hit_tables"] is False
    assert seen["accumulate_image"] is False
    assert np.array_equal(seen["single_sample_indices"], request.single_sample_indices)
    assert np.array_equal(seen["best_sample_indices_out"], request.best_sample_indices_out)
    assert seen["pixel_size_m"] == request.geometry.pixel_size_m
    assert seen["sample_width_m"] == request.geometry.sample_width_m
    assert seen["sample_length_m"] == request.geometry.sample_length_m
    assert np.array_equal(seen["n2_sample_array_override"], request.beam.n2_sample_array)


def test_simulate_reruns_to_build_intersection_cache_when_hit_tables_are_skipped() -> None:
    request = _build_request()
    request.collect_hit_tables = False
    calls: list[dict[str, object]] = []

    def fake_runner(*args, **kwargs):
        calls.append(
            {
                "collect_hit_tables": kwargs["collect_hit_tables"],
                "accumulate_image": kwargs["accumulate_image"],
            }
        )
        image = np.array(args[6], copy=True)
        hit_tables = []
        if kwargs["collect_hit_tables"]:
            hit_tables = [
                np.array(
                    [[3.0, 4.0, 5.0, 6.0, 1.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
            ]
        return (
            image,
            hit_tables,
            np.array([1.0], dtype=np.float64),
            np.array([2.0], dtype=np.float64),
            np.array([3.0], dtype=np.float64),
            [np.empty((0, 3), dtype=np.float64)],
        )

    result = simulate(request, peak_runner=fake_runner)

    assert calls == [
        {"collect_hit_tables": False, "accumulate_image": True},
        {"collect_hit_tables": True, "accumulate_image": False},
    ]
    assert len(result.hit_tables) == 0
    assert result.intersection_cache is not None
    assert len(result.intersection_cache) == 1
    table = np.asarray(result.intersection_cache[0], dtype=np.float64)
    assert table.shape == (1, 14)
    assert np.isclose(float(table[0, 0]), np.pi / np.sqrt(3.0))
    assert np.isclose(float(table[0, 1]), 2.0 * np.pi / 7.0)
    assert np.isclose(float(table[0, 2]), 4.0)
    assert np.isclose(float(table[0, 3]), 5.0)


def test_simulate_uses_reserved_cpu_worker_count_for_numba_thread_limit(monkeypatch) -> None:
    request = _build_request()
    seen: list[int] = []

    @contextmanager
    def fake_thread_limit(num_threads):
        seen.append(int(num_threads))
        yield

    monkeypatch.setattr(engine, "default_reserved_cpu_worker_count", lambda: 10)
    monkeypatch.setattr(engine, "temporary_numba_thread_limit", fake_thread_limit)

    def fake_runner(*args, **kwargs):
        image = np.array(args[6], copy=True)
        return (
            image,
            [np.array([[1, 2, 3]], dtype=np.float64)],
            np.array([1.0], dtype=np.float64),
            np.array([2.0], dtype=np.float64),
            np.array([3.0], dtype=np.float64),
            [np.empty((0, 3), dtype=np.float64)],
        )

    simulate(request, peak_runner=fake_runner)

    assert seen == [10]


def test_forward_warmup_does_not_mark_warmed_on_python_fallback(monkeypatch) -> None:
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_WARMED", False)
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_WARMUP_FAILED", False)
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_DISABLED", False)

    def fake_run_simulation_request(request, *, peak_runner=None):
        _set_forward_safe_run_result(True)
        return _fake_simulation_result(request)

    monkeypatch.setattr(engine, "_run_simulation_request", fake_run_simulation_request)

    warmed = engine.warmup_forward_simulation_numba()

    assert warmed is False
    assert engine._FORWARD_SIMULATION_NUMBA_WARMED is False
    assert engine._FORWARD_SIMULATION_NUMBA_WARMUP_FAILED is True
    assert engine._FORWARD_SIMULATION_NUMBA_DISABLED is False


def test_simulate_skips_repeat_forward_warmup_after_failure(monkeypatch) -> None:
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_WARMED", False)
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_WARMUP_FAILED", False)
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_DISABLED", False)
    warmup_calls = 0

    def fake_run_simulation_request(request: SimulationRequest, *, peak_runner=None):
        nonlocal warmup_calls
        if int(request.geometry.image_size) == 2:
            warmup_calls += 1
            raise RuntimeError("warmup failed")
        _set_forward_safe_run_result(False)
        return _fake_simulation_result(request)

    monkeypatch.setattr(engine, "_run_simulation_request", fake_run_simulation_request)
    request = _build_request()

    simulate(request)
    simulate(request)

    assert warmup_calls == 1
    assert engine._FORWARD_SIMULATION_NUMBA_WARMED is True
    assert engine._FORWARD_SIMULATION_NUMBA_WARMUP_FAILED is False


def test_simulate_serializes_first_real_forward_run_after_warmup_failure(monkeypatch) -> None:
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_WARMED", False)
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_WARMUP_FAILED", False)
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_DISABLED", False)
    warmup_calls = 0
    active_real_calls = 0
    max_active_real_calls = 0
    state_lock = threading.Lock()
    start_gate = threading.Barrier(2)

    def fake_run_simulation_request(request: SimulationRequest, *, peak_runner=None):
        nonlocal warmup_calls, active_real_calls, max_active_real_calls
        if int(request.geometry.image_size) == 2:
            warmup_calls += 1
            raise RuntimeError("warmup failed")
        _set_forward_safe_run_result(False)
        with state_lock:
            active_real_calls += 1
            max_active_real_calls = max(max_active_real_calls, active_real_calls)
        try:
            threading.Event().wait(0.05)
            return _fake_simulation_result(request)
        finally:
            with state_lock:
                active_real_calls -= 1

    monkeypatch.setattr(engine, "_run_simulation_request", fake_run_simulation_request)
    request = _build_request()

    def run_once():
        start_gate.wait(timeout=5.0)
        return simulate(request)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_once) for _ in range(2)]
        for future in futures:
            future.result(timeout=5.0)

    assert warmup_calls == 1
    assert max_active_real_calls == 1
    assert engine._FORWARD_SIMULATION_NUMBA_WARMED is True
    assert engine._FORWARD_SIMULATION_NUMBA_WARMUP_FAILED is False


def test_simulate_clears_forward_warmed_state_after_python_fallback(monkeypatch) -> None:
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_WARMED", True)
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_WARMUP_FAILED", False)
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_DISABLED", False)
    def fake_run_simulation_request(request, *, peak_runner=None):
        _set_forward_safe_run_result(True)
        return _fake_simulation_result(request)

    monkeypatch.setattr(engine, "_run_simulation_request", fake_run_simulation_request)

    simulate(_build_request())

    assert engine._FORWARD_SIMULATION_NUMBA_WARMED is False
    assert engine._FORWARD_SIMULATION_NUMBA_WARMUP_FAILED is True
    assert engine._FORWARD_SIMULATION_NUMBA_DISABLED is True


def test_simulate_prefers_python_runner_after_forward_numba_is_disabled(monkeypatch) -> None:
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_WARMED", False)
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_WARMUP_FAILED", True)
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_DISABLED", True)
    seen: dict[str, object] = {}

    @contextmanager
    def fake_thread_limit(_num_threads):
        yield

    def fake_runner(*args, **kwargs):
        seen.update(kwargs)
        image = np.array(args[6], copy=True)
        return (
            image,
            [],
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            [],
        )

    monkeypatch.setattr(engine, "temporary_numba_thread_limit", fake_thread_limit)
    monkeypatch.setattr(engine, "process_peaks_parallel_safe", fake_runner)

    simulate(_build_request(), peak_runner=engine.process_peaks_parallel_safe)

    assert seen["prefer_python_runner"] is True


def test_simulate_disables_forward_numba_after_serialized_real_call_exception(monkeypatch) -> None:
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_WARMED", False)
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_WARMUP_FAILED", False)
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_DISABLED", False)
    seen: dict[str, object] = {"warmup_calls": 0, "real_calls": 0}

    @contextmanager
    def fake_thread_limit(_num_threads):
        yield

    def fake_runner(*args, **kwargs):
        image_size = int(args[2])
        if image_size == 2:
            seen["warmup_calls"] = int(seen["warmup_calls"]) + 1
            raise RuntimeError("warmup failed")
        if not bool(kwargs.get("prefer_python_runner", False)):
            seen["real_calls"] = int(seen["real_calls"]) + 1
            raise RuntimeError("real failed")
        seen["prefer_python_runner"] = kwargs.get("prefer_python_runner")
        image = np.array(args[6], copy=True)
        return (
            image,
            [],
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            [],
        )

    monkeypatch.setattr(engine, "temporary_numba_thread_limit", fake_thread_limit)
    monkeypatch.setattr(engine, "process_peaks_parallel_safe", fake_runner)

    with pytest.raises(RuntimeError, match="real failed"):
        simulate(_build_request(), peak_runner=engine.process_peaks_parallel_safe)

    assert seen["warmup_calls"] == 1
    assert seen["real_calls"] == 1
    assert engine._FORWARD_SIMULATION_NUMBA_WARMED is False
    assert engine._FORWARD_SIMULATION_NUMBA_WARMUP_FAILED is True
    assert engine._FORWARD_SIMULATION_NUMBA_DISABLED is True

    simulate(_build_request(), peak_runner=engine.process_peaks_parallel_safe)

    assert seen["prefer_python_runner"] is True


def test_simulate_waiter_uses_python_runner_after_forward_numba_is_disabled(monkeypatch) -> None:
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_WARMED", False)
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_WARMUP_FAILED", True)
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_DISABLED", False)
    seen: dict[str, object] = {"real_calls": 0, "python_calls": 0}
    state_lock = threading.Lock()
    start_gate = threading.Barrier(2)

    @contextmanager
    def fake_thread_limit(_num_threads):
        yield

    def fake_runner(*args, **kwargs):
        if not bool(kwargs.get("prefer_python_runner", False)):
            with state_lock:
                seen["real_calls"] = int(seen["real_calls"]) + 1
            threading.Event().wait(0.05)
            raise RuntimeError("real failed")
        with state_lock:
            seen["python_calls"] = int(seen["python_calls"]) + 1
            seen["prefer_python_runner"] = kwargs.get("prefer_python_runner")
        image = np.array(args[6], copy=True)
        return (
            image,
            [],
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            [],
        )

    monkeypatch.setattr(engine, "temporary_numba_thread_limit", fake_thread_limit)
    monkeypatch.setattr(engine, "process_peaks_parallel_safe", fake_runner)
    request = _build_request()

    def run_once():
        start_gate.wait(timeout=5.0)
        try:
            simulate(request, peak_runner=engine.process_peaks_parallel_safe)
            return "ok"
        except RuntimeError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_once) for _ in range(2)]
        outcomes = [future.result(timeout=5.0) for future in futures]

    assert outcomes.count("real failed") == 1
    assert outcomes.count("ok") == 1
    assert seen["real_calls"] == 1
    assert seen["python_calls"] == 1
    assert seen["prefer_python_runner"] is True
    assert engine._FORWARD_SIMULATION_NUMBA_WARMED is False
    assert engine._FORWARD_SIMULATION_NUMBA_WARMUP_FAILED is True
    assert engine._FORWARD_SIMULATION_NUMBA_DISABLED is True


def test_simulate_keeps_forward_numba_disabled_after_concurrent_fallback_and_success(
    monkeypatch,
) -> None:
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_WARMED", True)
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_WARMUP_FAILED", False)
    monkeypatch.setattr(engine, "_FORWARD_SIMULATION_NUMBA_DISABLED", False)
    seen: dict[str, int] = {"fallback_calls": 0, "success_calls": 0}
    shared_stats = {"used_python_runner": False}
    shared_stats_lock = threading.Lock()
    start_gate = threading.Barrier(2)
    fallback_recorded = threading.Event()
    call_count = 0

    @contextmanager
    def fake_thread_limit(_num_threads):
        yield

    def fake_runner(*args, **kwargs):
        nonlocal call_count
        with shared_stats_lock:
            call_count += 1
            is_fallback = call_count == 1
        if not is_fallback:
            assert fallback_recorded.wait(timeout=5.0)
        safe_stats_out = kwargs["_safe_stats_out"]
        safe_stats_out["used_python_runner"] = is_fallback
        with shared_stats_lock:
            shared_stats["used_python_runner"] = is_fallback
        key = "fallback_calls" if is_fallback else "success_calls"
        seen[key] += 1
        if is_fallback:
            fallback_recorded.set()
        image = np.array(args[6], copy=True)
        return (
            image,
            [],
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            [],
        )

    monkeypatch.setattr(engine, "temporary_numba_thread_limit", fake_thread_limit)
    monkeypatch.setattr(engine, "process_peaks_parallel_safe", fake_runner)
    request = _build_request()

    def run_once():
        start_gate.wait(timeout=5.0)
        return simulate(request, peak_runner=engine.process_peaks_parallel_safe)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_once) for _ in range(2)]
        for future in futures:
            future.result(timeout=5.0)

    assert call_count == 2
    assert seen["fallback_calls"] == 1
    assert seen["success_calls"] == 1
    assert shared_stats["used_python_runner"] is False
    assert engine._FORWARD_SIMULATION_NUMBA_WARMED is False
    assert engine._FORWARD_SIMULATION_NUMBA_WARMUP_FAILED is True
    assert engine._FORWARD_SIMULATION_NUMBA_DISABLED is True


def test_simulate_qr_rods_respects_typed_request_with_custom_runner() -> None:
    request = _build_request()
    qr_dict = {1: {"hk": (1, 0), "L": np.array([0.0]), "I": np.array([1.0]), "deg": 1}}

    def fake_runner(*args, **kwargs):
        image = np.array(args[5], copy=True)
        image += 5.0
        return (
            image,
            [np.array([[1, 2, 3]], dtype=np.float64)],
            np.array([1.0], dtype=np.float64),
            np.array([2.0], dtype=np.float64),
            np.array([3.0], dtype=np.float64),
            [np.empty((0, 3), dtype=np.float64)],
            np.array([1], dtype=np.int32),
        )

    result = simulate_qr_rods(qr_dict, request, peak_runner=fake_runner)
    assert np.allclose(result.image, 5.0)
    assert np.array_equal(result.degeneracy, np.array([1], dtype=np.int32))


def test_simulate_qr_rods_forwards_extended_kernel_options() -> None:
    request = _build_request()
    request.optics_mode = 5
    request.collect_hit_tables = False
    request.accumulate_image = False
    request.geometry.pixel_size_m = 90e-6
    request.geometry.sample_width_m = 1.0e-3
    request.geometry.sample_length_m = 3.0e-3
    request.beam.n2_sample_array = np.array([1.0 + 0.05j], dtype=np.complex128)
    qr_dict = {1: {"hk": (1, 0), "L": np.array([0.0]), "I": np.array([1.0]), "deg": 1}}
    seen: dict[str, object] = {}

    def fake_runner(*args, **kwargs):
        seen.update(kwargs)
        image = np.array(args[5], copy=True)
        return (
            image,
            [np.array([[1, 2, 3]], dtype=np.float64)],
            np.array([1.0], dtype=np.float64),
            np.array([2.0], dtype=np.float64),
            np.array([3.0], dtype=np.float64),
            [np.empty((0, 3), dtype=np.float64)],
            np.array([1], dtype=np.int32),
        )

    simulate_qr_rods(qr_dict, request, peak_runner=fake_runner)

    assert seen["optics_mode"] == 5
    assert seen["collect_hit_tables"] is False
    assert seen["accumulate_image"] is False
    assert seen["pixel_size_m"] == request.geometry.pixel_size_m
    assert seen["sample_width_m"] == request.geometry.sample_width_m
    assert seen["sample_length_m"] == request.geometry.sample_length_m
    assert np.array_equal(seen["n2_sample_array_override"], request.beam.n2_sample_array)


def test_simulate_qr_rods_reruns_to_build_intersection_cache_when_hit_tables_are_skipped() -> None:
    request = _build_request()
    request.collect_hit_tables = False
    qr_dict = {1: {"hk": (1, 0), "L": np.array([0.0]), "I": np.array([1.0]), "deg": 1}}
    calls: list[dict[str, object]] = []

    def fake_runner(*args, **kwargs):
        calls.append(
            {
                "collect_hit_tables": kwargs["collect_hit_tables"],
                "accumulate_image": kwargs["accumulate_image"],
            }
        )
        image = np.array(args[5], copy=True)
        hit_tables = []
        if kwargs["collect_hit_tables"]:
            hit_tables = [
                np.array(
                    [[2.0, 7.0, 8.0, 9.0, 1.0, 0.0, 0.0]],
                    dtype=np.float64,
                )
            ]
        return (
            image,
            hit_tables,
            np.array([1.0], dtype=np.float64),
            np.array([2.0], dtype=np.float64),
            np.array([3.0], dtype=np.float64),
            [np.empty((0, 3), dtype=np.float64)],
            np.array([1], dtype=np.int32),
        )

    result = simulate_qr_rods(qr_dict, request, peak_runner=fake_runner)

    assert calls == [
        {"collect_hit_tables": False, "accumulate_image": True},
        {"collect_hit_tables": True, "accumulate_image": False},
    ]
    assert len(result.hit_tables) == 0
    assert result.intersection_cache is not None
    assert len(result.intersection_cache) == 1
    table = np.asarray(result.intersection_cache[0], dtype=np.float64)
    assert table.shape == (1, 14)
    assert np.isclose(float(table[0, 2]), 7.0)
    assert np.isclose(float(table[0, 3]), 8.0)
    assert np.array_equal(result.degeneracy, np.array([1], dtype=np.int32))


def test_simulate_qr_rods_uses_reserved_cpu_worker_count_for_numba_thread_limit(
    monkeypatch,
) -> None:
    request = _build_request()
    qr_dict = {1: {"hk": (1, 0), "L": np.array([0.0]), "I": np.array([1.0]), "deg": 1}}
    seen: list[int] = []

    @contextmanager
    def fake_thread_limit(num_threads):
        seen.append(int(num_threads))
        yield

    monkeypatch.setattr(engine, "default_reserved_cpu_worker_count", lambda: 10)
    monkeypatch.setattr(engine, "temporary_numba_thread_limit", fake_thread_limit)

    def fake_runner(*args, **kwargs):
        image = np.array(args[5], copy=True)
        return (
            image,
            [np.array([[1, 2, 3]], dtype=np.float64)],
            np.array([1.0], dtype=np.float64),
            np.array([2.0], dtype=np.float64),
            np.array([3.0], dtype=np.float64),
            [np.empty((0, 3), dtype=np.float64)],
            np.array([1], dtype=np.int32),
        )

    simulate_qr_rods(qr_dict, request, peak_runner=fake_runner)

    assert seen == [10]
