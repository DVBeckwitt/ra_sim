from __future__ import annotations

import contextlib
import importlib
from types import SimpleNamespace

import numpy as np

from tests.test_gui_runtime_update_actions import _prepare_runtime


def _minimal_mosaic_params() -> dict[str, object]:
    return {
        "solve_q_steps": 32,
        "solve_q_rel_tol": 1.0e-6,
        "solve_q_mode": 0,
        "events_per_beam_phase": 50,
        "beam_x_array": np.asarray([0.0], dtype=np.float64),
        "theta_array": np.asarray([0.0], dtype=np.float64),
        "_sampling_signature": (),
    }


def _analysis_job(**overrides: object) -> dict[str, object]:
    job = {
        "job_id": 1,
        "signature": ("analysis", 1),
        "epoch": 0,
        "image": np.ones((4, 4), dtype=np.float64),
        "background_image": None,
        "npt_rad": 8,
        "npt_azim": 6,
        "is_preview": False,
        "cached_bg_res2": None,
        "cached_bg_caked": None,
        "intersection_cache": (),
        "sim_cache_sig": ("sim-cache",),
        "bg_cache_sig": None,
        "sim_caking_sig": ("sim-caking",),
        "bg_caking_sig": None,
        "distance_m": 0.5,
        "center": np.asarray([2.0, 2.0], dtype=np.float64),
        "pixel_size_m": 1.0e-4,
        "wavelength_m": 1.24e-10,
        "gamma_deg": 0.0,
        "Gamma_deg": 0.0,
        "chi_deg": 0.0,
        "psi_deg": 0.0,
        "psi_z_deg": 0.0,
        "theta_initial_deg": 0.2,
        "cor_angle_deg": 0.0,
        "zs": 0.0,
        "zb": 0.0,
        "q_space_geometry": {
            "distance_m": 0.5,
            "center": np.asarray([2.0, 2.0], dtype=np.float64),
            "pixel_size_m": 1.0e-4,
            "wavelength_m": 1.24e-10,
            "gamma_deg": 0.0,
            "Gamma_deg": 0.0,
            "chi_deg": 0.0,
            "psi_deg": 0.0,
            "psi_z_deg": 0.0,
            "theta_initial_deg": 0.2,
            "cor_angle_deg": 0.0,
            "zs": 0.0,
            "zb": 0.0,
        },
        "q_space_requested": False,
        "caked_outputs_requested": True,
    }
    job.update(overrides)
    return job


def test_geometry_source_signature_includes_distance_m() -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    params_a = {"distance_m": 0.5}
    params_b = {"distance_m": 0.6}

    sig_a = runtime_session._geometry_source_snapshot_signature_from_params(
        params_a,
        mosaic_params=_minimal_mosaic_params(),
        optics_mode_component=0,
        sf_prune_bias=0.0,
        sf_prune_stats={},
        ordered_structure_scale=1.0,
        qr_cylinder_replace_requested=False,
        primary_source_signature=None,
        secondary_source_signature=None,
        secondary_a=0.0,
        secondary_c=0.0,
    )
    sig_b = runtime_session._geometry_source_snapshot_signature_from_params(
        params_b,
        mosaic_params=_minimal_mosaic_params(),
        optics_mode_component=0,
        sf_prune_bias=0.0,
        sf_prune_stats={},
        ordered_structure_scale=1.0,
        qr_cylinder_replace_requested=False,
        primary_source_signature=None,
        secondary_source_signature=None,
        secondary_a=0.0,
        secondary_c=0.0,
    )

    assert sig_a != sig_b


def test_prepare_q_space_display_payload_filters_only_qr_aligned_arrays(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    def _convert_stub(_image, **_kwargs):
        return SimpleNamespace(
            qr=np.asarray([-0.2, np.nan, 0.1, 0.2], dtype=np.float64),
            qz=np.asarray([0.0, 1.0], dtype=np.float64),
            intensity=np.arange(8, dtype=np.float64).reshape(2, 4),
            sum_signal=np.arange(8, dtype=np.float64).reshape(2, 4) + 10.0,
            sum_normalization=np.ones((2, 4), dtype=np.float64),
            count=np.ones((4, 2), dtype=np.float64),
        )

    monkeypatch.setattr(runtime_session, "convert_image_to_q_space", _convert_stub)

    payload = runtime_session._prepare_q_space_display_payload(
        np.ones((4, 4), dtype=np.float64),
        npt_rad=4,
        npt_azim=2,
        distance_m=0.5,
        center=np.asarray([2.0, 2.0], dtype=np.float64),
        pixel_size_m=1.0e-4,
        wavelength_m=1.24e-10,
        gamma_deg=0.0,
        Gamma_deg=0.0,
        chi_deg=0.0,
        psi_deg=0.0,
        psi_z_deg=0.0,
        theta_initial_deg=0.0,
        cor_angle_deg=0.0,
        zs=0.0,
        zb=0.0,
    )

    assert isinstance(payload, dict)
    np.testing.assert_allclose(payload["qr"], [0.1, 0.2])
    np.testing.assert_allclose(payload["image"], [[2.0, 3.0], [6.0, 7.0]])
    assert np.asarray(payload["sum_signal"]).shape == (2, 2)
    assert np.asarray(payload["sum_normalization"]).shape == (2, 2)
    assert np.asarray(payload["count"]).shape == (4, 2)


def test_q_space_only_analysis_does_not_call_caking(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")

    monkeypatch.setattr(
        runtime_session,
        "caking",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no caking")),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_q_space_display_payload_with_geometry",
        lambda *_args, **_kwargs: {
            "image": np.ones((2, 2), dtype=np.float64),
            "qr": np.asarray([0.1, 0.2], dtype=np.float64),
            "qz": np.asarray([0.0, 1.0], dtype=np.float64),
            "extent": [0.0, 0.3, -0.5, 1.5],
        },
    )

    result = runtime_session._run_analysis_job(
        _analysis_job(q_space_requested=True, caked_outputs_requested=False)
    )

    assert result["sim_res2"] is None
    assert result["bg_res2"] is None
    assert result["sim_caked"] is None
    assert isinstance(result["sim_q_space"], dict)


def test_caked_analysis_still_calls_caking(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    calls: list[tuple[int, int]] = []
    sim_res2 = object()

    monkeypatch.setattr(runtime_session, "_build_analysis_integrator", lambda _job: object())
    monkeypatch.setattr(
        runtime_session,
        "temporary_numba_thread_limit",
        lambda *_args, **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(runtime_session, "default_reserved_cpu_worker_count", lambda: 1)
    monkeypatch.setattr(
        runtime_session,
        "caking",
        lambda *_args, **kwargs: (
            calls.append((int(kwargs["npt_rad"]), int(kwargs["npt_azim"]))) or sim_res2
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_display_payload",
        lambda *_args, **_kwargs: {
            "image": np.ones((2, 2), dtype=np.float64),
            "radial": np.asarray([1.0, 2.0], dtype=np.float64),
            "azimuth": np.asarray([0.0, 1.0], dtype=np.float64),
            "extent": [0.0, 1.0, 0.0, 1.0],
        },
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_intersection_cache",
        lambda *_args, **_kwargs: (),
    )

    result = runtime_session._run_analysis_job(_analysis_job())

    assert calls == [(8, 6)]
    assert result["sim_res2"] is sim_res2
    assert isinstance(result["sim_caked"], dict)


def test_restore_q_space_payload_uses_stored_simulation_geometry(monkeypatch) -> None:
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    captured: list[dict[str, object]] = []
    stored_payloads: list[dict[str, object]] = []
    qgeom = dict(_analysis_job()["q_space_geometry"])
    qgeom["distance_m"] = 0.5
    qgeom["theta_initial_deg"] = 0.2

    monkeypatch.setattr(
        runtime_session,
        "simulation_runtime_state",
        SimpleNamespace(
            analysis_preview_bins=(8, 6),
            last_analysis_cache_sig=("analysis", 1),
            last_res2_sim=object(),
            last_res2_background=None,
            ai_cache={"ai": None, "detector_shape": (4, 4)},
            stored_intersection_cache=(),
            unscaled_image=np.ones((4, 4), dtype=np.float64),
            stored_simulation_q_space_geometry=qgeom,
            last_caked_image_unscaled=None,
            last_caked_radial_values=None,
            last_caked_azimuth_values=None,
            last_caked_extent=None,
            last_caked_intersection_cache=None,
            last_caked_background_image_unscaled=None,
            last_q_space_image_unscaled=None,
        ),
    )
    monkeypatch.setattr(
        runtime_session,
        "_prepare_caked_display_payload",
        lambda *_args, **_kwargs: {
            "image": np.ones((2, 2), dtype=np.float64),
            "radial": np.asarray([1.0, 2.0], dtype=np.float64),
            "azimuth": np.asarray([0.0, 1.0], dtype=np.float64),
            "extent": [0.0, 1.0, 0.0, 1.0],
            "transform_bundle": None,
        },
    )
    monkeypatch.setattr(runtime_session, "_set_live_caked_transform_bundle", lambda *_a, **_k: None)
    monkeypatch.setattr(runtime_session, "_prepare_caked_intersection_cache", lambda *_a, **_k: ())
    monkeypatch.setattr(runtime_session, "pixel_size_m", 1.0e-4, raising=False)
    monkeypatch.setattr(
        runtime_session,
        "corto_detector_var",
        SimpleNamespace(get=lambda: 0.6),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "center_x_var",
        SimpleNamespace(get=lambda: 3.0),
        raising=False,
    )
    monkeypatch.setattr(
        runtime_session,
        "center_y_var",
        SimpleNamespace(get=lambda: 3.0),
        raising=False,
    )

    def _q_space_payload(_image, **kwargs):
        captured.append(dict(kwargs))
        return {
            "image": np.ones((2, 2), dtype=np.float64),
            "qr": np.asarray([0.1, 0.2], dtype=np.float64),
            "qz": np.asarray([0.0, 1.0], dtype=np.float64),
            "extent": [0.0, 0.3, -0.5, 1.5],
        }

    monkeypatch.setattr(runtime_session, "_prepare_q_space_display_payload", _q_space_payload)
    monkeypatch.setattr(
        runtime_session,
        "_store_q_space_display_payload",
        lambda **kwargs: stored_payloads.append(dict(kwargs)),
    )
    monkeypatch.setattr(runtime_session, "_current_app_shell_view_mode", lambda: "q_space")

    restored = runtime_session._restore_caked_display_payload_from_cached_results(
        background_visible=False,
        q_space_requested=False,
    )

    assert restored is True
    assert captured
    assert captured[0]["distance_m"] == 0.5
    assert captured[0]["theta_initial_deg"] == 0.2
    assert stored_payloads


def test_q_space_only_current_result_does_not_require_caked_res2(monkeypatch) -> None:
    runtime_session, fixture = _prepare_runtime(monkeypatch)
    state = runtime_session.simulation_runtime_state
    monkeypatch.setattr(runtime_session, "_current_app_shell_view_mode", lambda: "q_space")
    requested_jobs: list[dict[str, object]] = []
    monkeypatch.setattr(
        runtime_session,
        "_request_async_analysis_job",
        lambda job: requested_jobs.append(dict(job)) or "submitted",
    )

    sim_caking_sig = (fixture["sim_signature"], fixture["caked_geometry_sig"], 1.0)
    sim_cache_sig = (
        sim_caking_sig,
        int(runtime_session.DEFAULT_Q_SPACE_DISPLAY_QR_BINS),
        int(runtime_session.DEFAULT_Q_SPACE_DISPLAY_QZ_BINS),
    )
    current_analysis_cache_sig = (sim_cache_sig, None)
    analysis_sig = (
        sim_caking_sig,
        None,
        fixture["q_space_payload_geometry_sig"],
        False,
        True,
    )
    state.last_analysis_signature = analysis_sig
    state.last_analysis_cache_sig = current_analysis_cache_sig
    state.analysis_preview_active = False
    state.last_res2_sim = None
    state.last_res2_background = None
    state.analysis_ready_result = None
    state.analysis_epoch = 0
    state.last_q_space_image_unscaled = np.ones((2, 2), dtype=np.float64)
    state.last_q_space_extent = [0.0, 1.0, -1.0, 1.0]
    state.last_q_space_payload_signature = (
        current_analysis_cache_sig,
        fixture["q_space_payload_geometry_sig"],
    )
    state.stored_simulation_q_space_geometry = dict(_analysis_job()["q_space_geometry"])

    runtime_session.do_update()

    assert requested_jobs == []
