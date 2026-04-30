from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
import math
import threading

import numpy as np
import pytest

from ra_sim.fitting import optimization as opt
from ra_sim.simulation import engine as sim_engine
from ra_sim.simulation.types import (
    DetectorGeometry,
    BeamSamples,
    DebyeWallerParams,
    MosaicParams,
    SimulationRequest,
    SimulationResult,
)


def _base_params(image_size: int, *, optics_mode: int = 0) -> dict:
    return {
        "gamma": 0.0,
        "Gamma": 0.0,
        "corto_detector": 0.1,
        "theta_initial": 0.0,
        "cor_angle": 0.0,
        "zs": 0.0,
        "zb": 0.0,
        "chi": 0.0,
        "a": 4.0,
        "c": 7.0,
        "center": [image_size / 2.0, image_size / 2.0],
        "lambda": 1.0,
        "n2": 1.0,
        "psi": 0.0,
        "psi_z": 0.0,
        "debye_x": 0.0,
        "debye_y": 0.0,
        "optics_mode": int(optics_mode),
        "mosaic_params": {
            "beam_x_array": np.zeros(1, dtype=np.float64),
            "beam_y_array": np.zeros(1, dtype=np.float64),
            "theta_array": np.zeros(1, dtype=np.float64),
            "phi_array": np.zeros(1, dtype=np.float64),
            "sigma_mosaic_deg": 0.2,
            "gamma_mosaic_deg": 0.1,
            "eta": 0.05,
            "wavelength_array": np.ones(1, dtype=np.float64),
        },
    }


def _fake_process_peaks(*args, **kwargs):
    image_size = int(args[2])
    image = np.zeros((image_size, image_size), dtype=np.float64)
    hit_tables = [
        np.array(
            [[1.0, 4.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
    ]
    return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []


def _fake_process_peaks_same_hkl_two_hits(*args, **kwargs):
    image_size = int(args[2])
    image = np.zeros((image_size, image_size), dtype=np.float64)
    hit_tables = [
        np.array(
            [
                [1.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.0],
                [0.8, 8.0, 8.0, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    ]
    return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []


def test_trial_sim_caked_image_payload_accepts_axes_only_builder():
    calls = []

    def axes_builder(detector_image, *, local_params=None, axes_only=False):
        del detector_image, local_params
        calls.append(bool(axes_only))
        return {
            "available": True,
            "axes_only": True,
            "image": None,
            "radial_axis": np.asarray([1.0, 2.0], dtype=np.float64),
            "azimuth_axis": np.asarray([-1.0, 1.0], dtype=np.float64),
            "detector_simulation_signature": "axes_only",
            "caked_simulation_signature": "axes_only",
        }

    subset = opt.ReflectionSimulationSubset(
        miller=np.empty((0, 3), dtype=np.float64),
        intensities=np.empty((0,), dtype=np.float64),
        measured_entries=[],
        original_indices=np.empty((0,), dtype=np.int64),
        total_reflection_count=0,
        fixed_source_reflection_count=0,
        fallback_hkl_count=0,
        reduced=False,
    )
    dataset_ctx = opt.GeometryFitDatasetContext(
        dataset_index=0,
        label="bg0",
        theta_initial=0.0,
        subset=subset,
        sim_caked_image_builder=axes_builder,
        sim_caked_image_builder_kind="test_axes_only",
    )

    payload = opt._build_trial_sim_caked_image_payload(
        dataset_ctx,
        sim_buffer=np.zeros((4, 4), dtype=np.float64),
        trial_params={},
        axes_only=True,
    )

    assert calls == [True]
    assert payload["available"] is True
    assert payload["axes_only"] is True
    assert payload["image"].size == 0
    assert payload["radial_axis"].tolist() == [1.0, 2.0]
    assert payload["azimuth_axis"].tolist() == [-1.0, 1.0]


def test_estimate_pixel_size_prefers_positive_sources_in_order():
    params = _base_params(12)
    params["pixel_size"] = 2.5e-4
    params["pixel_size_m"] = 1.5e-4
    params["debye_x"] = 5.0e-5
    assert np.isclose(opt._estimate_pixel_size(params), 2.5e-4)

    params = _base_params(12)
    params["pixel_size_m"] = 1.5e-4
    params["debye_x"] = 5.0e-5
    assert np.isclose(opt._estimate_pixel_size(params), 1.5e-4)

    params = _base_params(12)
    params["debye_x"] = 5.0e-5
    assert np.isclose(opt._estimate_pixel_size(params), 5.0e-5)

    params = _base_params(12)
    params["corto_detector"] = 0.2
    assert np.isclose(opt._estimate_pixel_size(params), 0.2 / 4096.0)


def _fake_process_three_reflections(*args, **kwargs):
    image_size = int(args[2])
    miller_subset = np.asarray(args[0], dtype=np.float64)
    image = np.zeros((image_size, image_size), dtype=np.float64)
    coord_map = {
        (1, 0, 0): (4.0, 4.0),
        (0, 1, 0): (10.0, 10.0),
        (0, 0, 1): (14.0, 14.0),
    }
    hit_tables = []
    for row in miller_subset:
        hkl = tuple(int(round(v)) for v in row)
        col, row_px = coord_map[hkl]
        hit_tables.append(
            np.array(
                [[10.0, col, row_px, 0.0, float(hkl[0]), float(hkl[1]), float(hkl[2])]],
                dtype=np.float64,
            )
        )
    return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []


def _process_wrapper_args(image_size: int):
    params = _base_params(image_size)
    mosaic = params["mosaic_params"]
    return (
        np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        int(image_size),
        float(params["a"]),
        float(params["c"]),
        float(params["lambda"]),
        np.zeros((image_size, image_size), dtype=np.float64),
        float(params["corto_detector"]),
        float(params["gamma"]),
        float(params["Gamma"]),
        float(params["chi"]),
        float(params["psi"]),
        float(params["psi_z"]),
        float(params["zs"]),
        float(params["zb"]),
        params["n2"],
        np.asarray(mosaic["beam_x_array"], dtype=np.float64),
        np.asarray(mosaic["beam_y_array"], dtype=np.float64),
        np.asarray(mosaic["theta_array"], dtype=np.float64),
        np.asarray(mosaic["phi_array"], dtype=np.float64),
        float(mosaic["sigma_mosaic_deg"]),
        float(mosaic["gamma_mosaic_deg"]),
        float(mosaic["eta"]),
        np.asarray(mosaic["wavelength_array"], dtype=np.float64),
        float(params["debye_x"]),
        float(params["debye_y"]),
        np.asarray(params["center"], dtype=np.float64),
        float(params["theta_initial"]),
        float(params["cor_angle"]),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )


def _build_forward_warmup_request() -> SimulationRequest:
    return SimulationRequest(
        miller=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        geometry=DetectorGeometry(
            image_size=4,
            av=1.0,
            cv=1.0,
            lambda_angstrom=1.0,
            distance_m=0.1,
            gamma_deg=0.0,
            Gamma_deg=0.0,
            chi_deg=0.0,
            psi_deg=0.0,
            psi_z_deg=0.0,
            zs=0.0,
            zb=0.0,
            center=np.array([2.0, 2.0], dtype=np.float64),
            theta_initial_deg=0.0,
            cor_angle_deg=0.0,
            unit_x=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            n_detector=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            pixel_size_m=1.0e-4,
            sample_width_m=0.0,
            sample_length_m=0.0,
        ),
        beam=BeamSamples(
            beam_x_array=np.array([0.0], dtype=np.float64),
            beam_y_array=np.array([0.0], dtype=np.float64),
            theta_array=np.array([0.0], dtype=np.float64),
            phi_array=np.array([0.0], dtype=np.float64),
            wavelength_array=np.array([1.0], dtype=np.float64),
        ),
        mosaic=MosaicParams(
            sigma_mosaic_deg=0.2,
            gamma_mosaic_deg=0.1,
            eta=0.05,
            solve_q_steps=1000,
            solve_q_rel_tol=5.0e-4,
            solve_q_mode=0,
        ),
        debye_waller=DebyeWallerParams(x=0.0, y=0.0),
        n2=1.0 + 0.0j,
        image_buffer=np.zeros((4, 4), dtype=np.float64),
        save_flag=0,
        record_status=False,
        thickness=0.0,
        optics_mode=0,
        collect_hit_tables=True,
        accumulate_image=True,
        exit_projection_mode="internal",
    )


def test_build_global_point_matches_uses_global_assignment():
    simulated = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)]
    measured = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]

    matches = opt._build_global_point_matches(simulated, measured)

    assert {(int(sim_idx), int(meas_idx)) for *_pts, sim_idx, meas_idx in matches} == {
        (0, 0),
        (1, 1),
        (2, 2),
    }
    assert np.isclose(
        sum(float(distance) for *_pts, distance, _sim_idx, _meas_idx in matches),
        2.0 * np.sqrt(2.0),
    )


def test_dynamic_point_match_reanchors_measured_anchor_and_reports_motion(
    monkeypatch,
):
    calls: dict[str, object] = {}

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, 12.0, 12.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_reanchor(
        measured_entry,
        simulated_detector_point,
        *,
        local_params=None,
        dataset_ctx=None,
    ):
        calls["reanchor"] = {
            "measured_entry": dict(measured_entry),
            "simulated_detector_point": tuple(simulated_detector_point),
            "dataset_index": (int(dataset_ctx.dataset_index) if dataset_ctx is not None else None),
        }
        return {
            "x": float(simulated_detector_point[0]),
            "y": float(simulated_detector_point[1]),
            "detector_x": float(simulated_detector_point[0]),
            "detector_y": float(simulated_detector_point[1]),
        }

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    subset = opt.ReflectionSimulationSubset(
        miller=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        measured_entries=[
            {
                "label": "peak-0",
                "hkl": (1, 0, 0),
                "overlay_match_index": 0,
                "source_table_index": 0,
                "source_row_index": 0,
                "detector_x": 6.0,
                "detector_y": 6.0,
                "x": 6.0,
                "y": 6.0,
            }
        ],
        original_indices=np.array([0], dtype=np.int64),
        total_reflection_count=1,
        fixed_source_reflection_count=1,
        fallback_hkl_count=0,
        reduced=False,
    )
    dataset_ctx = opt.GeometryFitDatasetContext(
        dataset_index=0,
        label="bg0",
        theta_initial=0.0,
        subset=subset,
        experimental_image=np.zeros((32, 32), dtype=np.float64),
        dynamic_reanchor_enabled=True,
        dynamic_reanchor_callback=fake_reanchor,
    )
    local = _base_params(32)
    local["pixel_size"] = 1.0
    local["corto_detector"] = 100.0

    residual, diagnostics, summary = opt._evaluate_geometry_fit_dataset_dynamic_point_matches(
        local,
        dataset_ctx,
        image_size=32,
        missing_pair_penalty_deg=5.0,
        theta_value=0.0,
        collect_diagnostics=True,
    )

    assert calls["reanchor"]["simulated_detector_point"] == (12.0, 12.0)
    assert calls["reanchor"]["dataset_index"] == 0
    assert residual.shape == (2,)
    assert np.allclose(residual, 0.0, atol=1.0e-12)
    assert diagnostics[0]["measured_reanchor_attempted"] is True
    assert diagnostics[0]["measured_reanchor_status"] == "updated"
    assert diagnostics[0]["measured_reanchor_motion_px"] > 0.0
    assert summary["matched_pair_count"] == 1
    assert summary["measured_anchor_reanchor_enabled"] is True
    assert summary["measured_anchor_reanchor_attempt_count"] == 1
    assert summary["measured_anchor_reanchor_count"] == 1
    assert summary["measured_anchor_reanchor_fail_count"] == 0
    assert summary["measured_anchor_motion_max_px"] > 0.0


def test_dynamic_point_match_reanchor_callback_failure_reports_legacy_status(
    monkeypatch,
):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, 12.0, 12.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    subset = opt.ReflectionSimulationSubset(
        miller=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        measured_entries=[
            {
                "label": "peak-0",
                "hkl": (1, 0, 0),
                "overlay_match_index": 0,
                "source_table_index": 0,
                "source_row_index": 0,
                "detector_x": 6.0,
                "detector_y": 6.0,
                "x": 6.0,
                "y": 6.0,
            }
        ],
        original_indices=np.array([0], dtype=np.int64),
        total_reflection_count=1,
        fixed_source_reflection_count=1,
        fallback_hkl_count=0,
        reduced=False,
    )
    dataset_ctx = opt.GeometryFitDatasetContext(
        dataset_index=0,
        label="bg0",
        theta_initial=0.0,
        subset=subset,
        experimental_image=np.zeros((32, 32), dtype=np.float64),
        dynamic_reanchor_enabled=True,
        dynamic_reanchor_callback=lambda *args, **kwargs: None,
    )
    local = _base_params(32)
    local["pixel_size"] = 1.0
    local["corto_detector"] = 100.0

    residual, diagnostics, summary = opt._evaluate_geometry_fit_dataset_dynamic_point_matches(
        local,
        dataset_ctx,
        image_size=32,
        missing_pair_penalty_deg=5.0,
        theta_value=0.0,
        collect_diagnostics=True,
    )

    assert residual.shape == (2,)
    assert diagnostics[0]["measured_reanchor_attempted"] is True
    assert diagnostics[0]["measured_reanchor_status"] == "callback_failed"
    assert summary["measured_anchor_reanchor_attempt_count"] == 1
    assert summary["measured_anchor_reanchor_count"] == 0
    assert summary["measured_anchor_reanchor_fail_count"] == 1


def test_dynamic_point_match_exact_projector_blocks_analytic_fallback_and_ignores_stale_sim_display(
    monkeypatch,
):
    projector_calls: list[dict[str, object]] = []

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, 12.0, 12.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def projector(cols, rows, *, local_params, anchor_kind, input_frame):
        cols_arr = np.asarray(cols, dtype=np.float64).reshape(-1)
        rows_arr = np.asarray(rows, dtype=np.float64).reshape(-1)
        projector_calls.append(
            {
                "anchor_kind": str(anchor_kind),
                "input_frame": str(input_frame),
                "cols": cols_arr.tolist(),
                "rows": rows_arr.tolist(),
                "local_params": dict(local_params),
            }
        )
        return {
            "two_theta_deg": cols_arr + (200.0 if anchor_kind == "measured" else 100.0),
            "phi_deg": rows_arr - (60.0 if anchor_kind == "measured" else 50.0),
            "fit_space_source": "dataset_fit_space_projector",
            "input_frame": str(input_frame),
            "fit_space_projector_kind": "exact_caked_bundle",
            "cake_bundle_signature": f"sig-{anchor_kind}-{local_params.get('gamma', 0.0)}",
            "fit_space_local_params_signature": f"lp-{local_params.get('gamma', 0.0)}",
            "valid": True,
            "invalid_reason": None,
            "native_frame_conversion_source": f"test-{input_frame}",
            "native_frame_conversion_count": 0 if input_frame == "native_detector" else 1,
            "native_cols": cols_arr,
            "native_rows": rows_arr,
            "caked_projection_source": "fit_space_projector_native_detector",
        }

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(
        opt,
        "_detector_pixels_to_fit_space",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("analytic detector fit-space path must not run")
        ),
    )

    subset = opt.ReflectionSimulationSubset(
        miller=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        measured_entries=[
            {
                "label": "peak-0",
                "hkl": (1, 0, 0),
                "overlay_match_index": 0,
                "source_table_index": 0,
                "source_row_index": 0,
                "fit_source_identity_only": True,
                "native_col": 30.0,
                "native_row": 40.0,
                "sim_col": 999.0,
                "sim_row": 888.0,
            }
        ],
        original_indices=np.array([0], dtype=np.int64),
        total_reflection_count=1,
        fixed_source_reflection_count=1,
        fallback_hkl_count=0,
        reduced=False,
    )
    dataset_ctx = opt.GeometryFitDatasetContext(
        dataset_index=0,
        label="bg0",
        theta_initial=0.0,
        subset=subset,
        experimental_image=np.zeros((32, 32), dtype=np.float64),
        fit_space_projector=projector,
        fit_space_projector_kind="exact_caked_bundle",
    )
    local = _base_params(32)
    local["pixel_size"] = 1.0
    local["corto_detector"] = 100.0
    local["gamma"] = 1.5

    residual, diagnostics, summary = opt._evaluate_geometry_fit_dataset_dynamic_point_matches(
        local,
        dataset_ctx,
        image_size=32,
        missing_pair_penalty_deg=5.0,
        theta_value=0.0,
        collect_diagnostics=True,
    )

    assert residual.shape == (2,)
    assert len(diagnostics) == 1
    assert diagnostics[0]["measured_fit_space_source"] == "dataset_fit_space_projector"
    assert diagnostics[0]["simulated_fit_space_source"] == "dataset_fit_space_projector"
    assert diagnostics[0]["measured_detector_input_frame"] == "native_detector"
    assert diagnostics[0]["simulated_detector_input_frame"] == "fit_detector"
    assert diagnostics[0]["measured_native_frame_conversion_count"] == 0
    assert diagnostics[0]["simulated_native_frame_conversion_count"] == 1
    assert diagnostics[0]["fit_space_projector_kind"] == "exact_caked_bundle"
    assert diagnostics[0]["cake_bundle_signature"] == "sig-simulated-1.5"
    assert summary["exact_fit_space_projector_available"] is True
    assert projector_calls == [
        {
            "anchor_kind": "measured",
            "input_frame": "native_detector",
            "cols": [30.0],
            "rows": [40.0],
            "local_params": dict(local),
        },
        {
            "anchor_kind": "simulated",
            "input_frame": "fit_detector",
            "cols": [12.0],
            "rows": [12.0],
            "local_params": dict(local),
        },
    ]


def test_angular_difference_deg_wraps_phi_edge() -> None:
    assert opt._wrap_phi_deg(-358.0) == pytest.approx(2.0)
    assert opt._wrap_phi_deg(358.0) == pytest.approx(-2.0)
    assert opt._angular_difference_deg(-179.0, 179.0) == pytest.approx(2.0)
    assert opt._angular_difference_deg(179.0, -179.0) == pytest.approx(-2.0)


def test_exact_caked_manual_residual_audit_reports_degree_units(monkeypatch) -> None:
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, 12.0, 12.0, 0.0, 0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def projector(cols, rows, *, local_params, anchor_kind, input_frame):
        del local_params
        cols_arr = np.asarray(cols, dtype=np.float64).reshape(-1)
        rows_arr = np.asarray(rows, dtype=np.float64).reshape(-1)
        count = cols_arr.size
        if anchor_kind == "measured":
            two_theta = np.full(count, 30.0, dtype=np.float64)
            phi = np.full(count, 179.0, dtype=np.float64)
        else:
            two_theta = np.full(count, 33.0, dtype=np.float64)
            phi = np.full(count, -179.0, dtype=np.float64)
        return {
            "two_theta_deg": two_theta,
            "phi_deg": phi,
            "fit_space_source": "dataset_fit_space_projector",
            "input_frame": str(input_frame),
            "fit_space_projector_kind": "exact_caked_bundle",
            "cake_bundle_signature": f"sig-{anchor_kind}",
            "fit_space_local_params_signature": "lp-test",
            "valid": True,
            "invalid_reason": None,
            "native_frame_conversion_source": f"test-{input_frame}",
            "native_frame_conversion_count": 0 if input_frame == "native_detector" else 1,
            "native_cols": cols_arr,
            "native_rows": rows_arr,
            "caked_projection_source": "fit_space_projector_native_detector",
        }

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(
        opt,
        "_detector_pixels_to_fit_space",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("analytic detector fit-space path must not run")
        ),
    )
    subset = opt.ReflectionSimulationSubset(
        miller=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        measured_entries=[
            {
                "label": "c2-row-0",
                "pair_id": "manual-0",
                "hkl": (0, 0, 1),
                "overlay_match_index": 0,
                "source_table_index": 0,
                "source_row_index": 0,
                "fit_source_identity_only": True,
                "native_col": 30.0,
                "native_row": 40.0,
            }
        ],
        original_indices=np.array([0], dtype=np.int64),
        total_reflection_count=1,
        fixed_source_reflection_count=1,
        fallback_hkl_count=0,
        reduced=False,
    )
    dataset_ctx = opt.GeometryFitDatasetContext(
        dataset_index=0,
        label="bg0",
        theta_initial=0.0,
        subset=subset,
        experimental_image=np.zeros((32, 32), dtype=np.float64),
        fit_space_projector=projector,
        fit_space_projector_kind="exact_caked_bundle",
    )
    local = _base_params(32)
    local["pixel_size"] = 1.0
    local["corto_detector"] = 100.0
    local["_hk0_peak_priority_weight"] = 2.0

    residual, diagnostics, summary = opt._evaluate_geometry_fit_dataset_dynamic_point_matches(
        local,
        dataset_ctx,
        image_size=32,
        missing_pair_penalty_deg=5.0,
        theta_value=0.0,
        collect_diagnostics=True,
    )

    expected_raw_norm = np.sqrt(3.0**2 + 2.0**2)
    expected_weighted_norm = np.sqrt(6.0**2 + 4.0**2)
    assert residual.tolist() == pytest.approx([6.0, 4.0])
    diag = diagnostics[0]
    assert diag["manual_pair_id"] == "manual-0"
    assert diag["measured_two_theta_deg"] == pytest.approx(30.0)
    assert diag["measured_phi_deg"] == pytest.approx(179.0)
    assert diag["projected_two_theta_deg"] == pytest.approx(33.0)
    assert diag["projected_phi_deg"] == pytest.approx(-179.0)
    assert 0.0 <= diag["measured_two_theta_deg"] <= 90.0
    assert 0.0 <= diag["projected_two_theta_deg"] <= 90.0
    assert -180.0 <= diag["measured_phi_deg"] <= 180.0
    assert -180.0 <= diag["projected_phi_deg"] <= 180.0
    assert diag["delta_two_theta_deg"] == pytest.approx(3.0)
    assert diag["wrapped_delta_phi_deg"] == pytest.approx(2.0)
    assert diag["raw_angular_norm_deg"] == pytest.approx(expected_raw_norm)
    assert diag["weight"] == pytest.approx(2.0)
    assert diag["weighted_delta_two_theta_deg"] == pytest.approx(6.0)
    assert diag["weighted_delta_phi_deg"] == pytest.approx(4.0)
    assert diag["solver_residual_vector"] == pytest.approx([6.0, 4.0])
    assert diag["metric_name"] == "raw_angular_rms_deg"
    assert diag["metric_unit"] == "deg"
    assert diag["weighted_metric_unit"] == "weighted_deg"
    assert summary["raw_angular_rms_deg"] == pytest.approx(expected_raw_norm)
    assert summary["raw_angular_max_deg"] == pytest.approx(expected_raw_norm)
    assert summary["raw_angular_component_max_abs_deg"] == pytest.approx(3.0)
    assert summary["raw_angular_max_deg"] <= opt.RAW_ANGULAR_VECTOR_NORM_BOUND_DEG
    assert summary["raw_angular_component_max_abs_deg"] <= opt.RAW_ANGULAR_COMPONENT_BOUND_DEG
    assert summary["raw_angular_delta_sanity_ok"] is True
    assert summary["raw_angular_range_sanity_ok"] is True
    assert summary["raw_angular_range_failure_count"] == 0
    assert summary["raw_angular_sanity_ok"] is True
    assert summary["measured_two_theta_min_deg"] == pytest.approx(30.0)
    assert summary["measured_two_theta_max_deg"] == pytest.approx(30.0)
    assert summary["projected_two_theta_min_deg"] == pytest.approx(33.0)
    assert summary["projected_two_theta_max_deg"] == pytest.approx(33.0)
    assert summary["measured_phi_wrapped_min_deg"] == pytest.approx(179.0)
    assert summary["projected_phi_wrapped_min_deg"] == pytest.approx(-179.0)
    assert summary["weighted_angular_rms_weighted_deg"] == pytest.approx(expected_weighted_norm)
    assert summary["weighted_angular_max_weighted_deg"] == pytest.approx(expected_weighted_norm)
    assert summary["optimizer_component_rms_weighted_deg"] == pytest.approx(np.sqrt(26.0))
    assert summary["metric_name"] == "raw_angular_rms_deg"
    assert summary["metric_unit"] == "deg"
    assert summary["weighted_metric_name"] == "weighted_angular_rms_weighted_deg"
    assert summary["weighted_metric_unit"] == "weighted_deg"
    assert summary["optimizer_component_count"] == residual.size
    assert summary["detector_pixel_rms_px"] == pytest.approx(summary["unweighted_peak_rms_px"])
    assert np.isnan(float(summary["caked_pixel_rms_px"]))
    assert summary["manual_caked_residual_row_count"] == 1
    assert summary["dataset_fit_space_projector_row_count"] == 2
    assert summary["invalid_dataset_fit_space_projector_row_count"] == 0
    assert summary["analytic_detector_fit_space_row_count"] == 0
    assert summary["fallback_entry_count"] == 0
    assert summary["subset_fallback_hkl_count"] == 0


def test_exact_caked_manual_residual_audit_rejects_out_of_range_angles(monkeypatch) -> None:
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, 12.0, 12.0, 0.0, 0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def projector(cols, rows, *, local_params, anchor_kind, input_frame):
        del rows, local_params
        cols_arr = np.asarray(cols, dtype=np.float64).reshape(-1)
        if anchor_kind == "measured":
            two_theta = np.full(cols_arr.size, 300.0, dtype=np.float64)
            phi = np.full(cols_arr.size, 10.0, dtype=np.float64)
        else:
            two_theta = np.full(cols_arr.size, 303.0, dtype=np.float64)
            phi = np.full(cols_arr.size, 12.0, dtype=np.float64)
        return {
            "two_theta_deg": two_theta,
            "phi_deg": phi,
            "fit_space_source": "dataset_fit_space_projector",
            "input_frame": str(input_frame),
            "fit_space_projector_kind": "exact_caked_bundle",
            "cake_bundle_signature": f"sig-{anchor_kind}",
            "fit_space_local_params_signature": "lp-test",
            "valid": True,
            "invalid_reason": None,
            "native_frame_conversion_source": f"test-{input_frame}",
            "native_frame_conversion_count": 0 if input_frame == "native_detector" else 1,
            "native_cols": cols_arr,
            "native_rows": np.asarray(cols, dtype=np.float64).reshape(-1),
            "caked_projection_source": "fit_space_projector_native_detector",
        }

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    subset = opt.ReflectionSimulationSubset(
        miller=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        measured_entries=[
            {
                "label": "c2-row-0",
                "pair_id": "manual-0",
                "hkl": (0, 0, 1),
                "overlay_match_index": 0,
                "source_table_index": 0,
                "source_row_index": 0,
                "fit_source_identity_only": True,
                "native_col": 30.0,
                "native_row": 40.0,
            }
        ],
        original_indices=np.array([0], dtype=np.int64),
        total_reflection_count=1,
        fixed_source_reflection_count=1,
        fallback_hkl_count=0,
        reduced=False,
    )
    dataset_ctx = opt.GeometryFitDatasetContext(
        dataset_index=0,
        label="bg0",
        theta_initial=0.0,
        subset=subset,
        experimental_image=np.zeros((32, 32), dtype=np.float64),
        fit_space_projector=projector,
        fit_space_projector_kind="exact_caked_bundle",
    )
    local = _base_params(32)
    local["pixel_size"] = 1.0
    local["corto_detector"] = 100.0

    residual, _diagnostics, summary = opt._evaluate_geometry_fit_dataset_dynamic_point_matches(
        local,
        dataset_ctx,
        image_size=32,
        missing_pair_penalty_deg=5.0,
        theta_value=0.0,
        collect_diagnostics=True,
    )

    assert residual.tolist() == pytest.approx([3.0, 2.0])
    assert summary["raw_angular_delta_sanity_ok"] is True
    assert summary["raw_angular_range_sanity_ok"] is False
    assert summary["raw_angular_range_failure_count"] == 1
    assert summary["raw_angular_sanity_ok"] is False
    assert summary["measured_two_theta_max_deg"] == pytest.approx(300.0)
    assert summary["projected_two_theta_max_deg"] == pytest.approx(303.0)


def test_exact_caked_optimizer_component_rms_includes_line_residuals(monkeypatch) -> None:
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array([[1.0, 12.0, 12.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
            np.array([[1.0, 14.0, 12.0, 0.0, 0.0, 0.0, 2.0]], dtype=np.float64),
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def projector(cols, rows, *, local_params, anchor_kind, input_frame):
        del rows, local_params
        cols_arr = np.asarray(cols, dtype=np.float64).reshape(-1)
        if anchor_kind == "measured":
            two_theta = np.where(cols_arr < 21.0, 30.0, 40.0)
            phi = np.where(cols_arr < 21.0, 0.0, 10.0)
        else:
            two_theta = np.where(cols_arr < 13.0, 31.0, 41.0)
            phi = np.where(cols_arr < 13.0, 0.0, 12.0)
        return {
            "two_theta_deg": two_theta.astype(np.float64),
            "phi_deg": phi.astype(np.float64),
            "fit_space_source": "dataset_fit_space_projector",
            "input_frame": str(input_frame),
            "fit_space_projector_kind": "exact_caked_bundle",
            "cake_bundle_signature": f"sig-{anchor_kind}",
            "fit_space_local_params_signature": "lp-test",
            "valid": True,
            "invalid_reason": None,
            "native_frame_conversion_source": f"test-{input_frame}",
            "native_frame_conversion_count": 0 if input_frame == "native_detector" else 1,
            "native_cols": cols_arr,
            "native_rows": cols_arr,
            "caked_projection_source": "fit_space_projector_native_detector",
        }

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    entries = [
        {
            "label": f"c2-row-{idx}",
            "pair_id": f"manual-{idx}",
            "hkl": (0, 0, idx + 1),
            "overlay_match_index": idx,
            "source_table_index": idx,
            "source_row_index": 0,
            "fit_source_identity_only": True,
            "native_col": 20.0 + 2.0 * idx,
            "native_row": 40.0,
        }
        for idx in range(2)
    ]
    subset = opt.ReflectionSimulationSubset(
        miller=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=np.float64),
        intensities=np.array([1.0, 1.0], dtype=np.float64),
        measured_entries=entries,
        original_indices=np.array([0, 1], dtype=np.int64),
        total_reflection_count=2,
        fixed_source_reflection_count=2,
        fallback_hkl_count=0,
        reduced=False,
    )
    dataset_ctx = opt.GeometryFitDatasetContext(
        dataset_index=0,
        label="bg0",
        theta_initial=0.0,
        subset=subset,
        experimental_image=np.zeros((32, 32), dtype=np.float64),
        fit_space_projector=projector,
        fit_space_projector_kind="exact_caked_bundle",
    )
    local = _base_params(32)
    local["pixel_size"] = 1.0
    local["corto_detector"] = 100.0
    local["_q_group_line_constraints_enabled"] = True

    residual, _diagnostics, summary = opt._evaluate_geometry_fit_dataset_dynamic_point_matches(
        local,
        dataset_ctx,
        image_size=32,
        missing_pair_penalty_deg=5.0,
        theta_value=0.0,
        collect_diagnostics=True,
    )

    assert summary["matched_pair_count"] == 2
    assert residual.size > 2 * summary["matched_pair_count"]
    assert summary["optimizer_component_count"] == residual.size
    assert summary["optimizer_component_rms_weighted_deg"] == pytest.approx(
        float(np.sqrt(np.mean(residual * residual)))
    )


def test_exact_caked_optimizer_component_rms_rejects_nonfinite_vector() -> None:
    summary = opt._angular_degree_residual_audit_summary(
        measured_two_theta_deg=[30.0],
        measured_phi_deg=[179.0],
        projected_two_theta_deg=[33.0],
        projected_phi_deg=[-179.0],
        delta_two_theta_deg=[3.0],
        wrapped_delta_phi_deg=[2.0],
        weighted_delta_two_theta_deg=[6.0],
        weighted_delta_phi_deg=[4.0],
        priority_weights=[2.0],
        solver_residual_vector=[6.0, float("nan"), 4.0],
    )

    assert summary["optimizer_component_count"] == 3
    assert summary["optimizer_component_nonfinite_count"] == 1
    assert math.isnan(summary["optimizer_component_rms_weighted_deg"])
    merged = opt._merged_angular_degree_residual_audit_summary([summary])
    assert merged["optimizer_component_count"] == 3
    assert merged["optimizer_component_nonfinite_count"] == 1
    assert math.isnan(merged["optimizer_component_rms_weighted_deg"])


def test_exact_caked_residual_audit_rejects_missing_or_nonfinite_rows() -> None:
    summary = opt._angular_degree_residual_audit_summary(
        measured_two_theta_deg=[30.0],
        measured_phi_deg=[0.0],
        projected_two_theta_deg=[31.0],
        projected_phi_deg=[1.0],
        delta_two_theta_deg=[1.0, 2.0],
        wrapped_delta_phi_deg=[1.0, float("nan")],
        weighted_delta_two_theta_deg=[1.0, 2.0],
        weighted_delta_phi_deg=[1.0, 2.0],
        priority_weights=[1.0, 1.0],
        solver_residual_vector=[1.0, 1.0, 2.0, 2.0],
    )

    assert summary["raw_angular_row_count"] == 2
    assert summary["raw_angular_delta_failure_count"] == 1
    assert summary["raw_angular_range_row_count"] == 2
    assert summary["raw_angular_range_failure_count"] == 1
    assert summary["raw_angular_delta_sanity_ok"] is False
    assert summary["raw_angular_range_sanity_ok"] is False
    assert summary["raw_angular_sanity_ok"] is False
    assert math.isnan(summary["raw_angular_rms_deg"])
    assert math.isnan(summary["raw_angular_max_deg"])
    assert summary["weighted_angular_recompute_failure_count"] == 1
    assert math.isnan(summary["weighted_angular_rms_weighted_deg"])
    merged = opt._merged_angular_degree_residual_audit_summary([summary])
    assert merged["raw_angular_row_count"] == 2
    assert merged["raw_angular_delta_failure_count"] == 1
    assert merged["raw_angular_range_failure_count"] == 1
    assert merged["raw_angular_sanity_ok"] is False
    assert math.isnan(merged["raw_angular_rms_deg"])
    assert merged["weighted_angular_recompute_failure_count"] == 1
    assert math.isnan(merged["weighted_angular_rms_weighted_deg"])


def test_exact_caked_residual_audit_rejects_extra_angle_rows() -> None:
    summary = opt._angular_degree_residual_audit_summary(
        measured_two_theta_deg=[30.0, 31.0],
        measured_phi_deg=[0.0, 1.0],
        projected_two_theta_deg=[31.0, 32.0],
        projected_phi_deg=[1.0, 2.0],
        delta_two_theta_deg=[1.0],
        wrapped_delta_phi_deg=[1.0],
        weighted_delta_two_theta_deg=[1.0],
        weighted_delta_phi_deg=[1.0],
        priority_weights=[1.0],
        solver_residual_vector=[1.0, 1.0],
    )

    assert summary["raw_angular_row_count"] == 1
    assert summary["raw_angular_range_row_count"] == 1
    assert summary["raw_angular_range_failure_count"] == 1
    assert summary["raw_angular_delta_sanity_ok"] is True
    assert summary["raw_angular_range_sanity_ok"] is False
    assert summary["raw_angular_sanity_ok"] is False
    assert summary["measured_two_theta_max_deg"] == pytest.approx(31.0)
    assert summary["projected_phi_wrapped_max_deg"] == pytest.approx(2.0)


def test_exact_caked_residual_audit_rejects_delta_recompute_mismatch() -> None:
    summary = opt._angular_degree_residual_audit_summary(
        measured_two_theta_deg=[30.0],
        measured_phi_deg=[0.0],
        projected_two_theta_deg=[31.0],
        projected_phi_deg=[1.0],
        delta_two_theta_deg=[180.0],
        wrapped_delta_phi_deg=[0.0],
        weighted_delta_two_theta_deg=[180.0],
        weighted_delta_phi_deg=[0.0],
        priority_weights=[1.0],
        solver_residual_vector=[180.0, 0.0],
    )

    assert summary["raw_angular_delta_recompute_failure_count"] == 1
    assert summary["raw_angular_delta_failure_count"] == 1
    assert summary["raw_angular_range_sanity_ok"] is True
    assert summary["raw_angular_delta_sanity_ok"] is False
    assert summary["raw_angular_sanity_ok"] is False
    assert math.isnan(summary["raw_angular_rms_deg"])


def test_exact_caked_residual_audit_rejects_weighted_recompute_mismatch() -> None:
    summary = opt._angular_degree_residual_audit_summary(
        measured_two_theta_deg=[30.0],
        measured_phi_deg=[179.0],
        projected_two_theta_deg=[33.0],
        projected_phi_deg=[-179.0],
        delta_two_theta_deg=[3.0],
        wrapped_delta_phi_deg=[2.0],
        weighted_delta_two_theta_deg=[7.0],
        weighted_delta_phi_deg=[4.0],
        priority_weights=[2.0],
        solver_residual_vector=[6.0, 4.0],
    )

    assert summary["weighted_angular_recompute_failure_count"] == 1
    assert summary["weighted_angular_failure_count"] == 1
    assert math.isnan(summary["weighted_angular_rms_weighted_deg"])
    assert math.isnan(summary["weighted_angular_max_weighted_deg"])


def test_exact_caked_residual_audit_rejects_solver_point_component_mismatch() -> None:
    summary = opt._angular_degree_residual_audit_summary(
        measured_two_theta_deg=[30.0],
        measured_phi_deg=[179.0],
        projected_two_theta_deg=[33.0],
        projected_phi_deg=[-179.0],
        delta_two_theta_deg=[3.0],
        wrapped_delta_phi_deg=[2.0],
        weighted_delta_two_theta_deg=[6.0],
        weighted_delta_phi_deg=[4.0],
        priority_weights=[2.0],
        solver_residual_vector=[6.0, 5.0],
    )

    assert summary["optimizer_point_component_count"] == 2
    assert summary["optimizer_point_component_failure_count"] == 1
    assert summary["weighted_angular_failure_count"] == 1
    assert math.isnan(summary["weighted_angular_rms_weighted_deg"])


def test_exact_caked_residual_audit_compares_solver_points_to_recomputed_weighted() -> None:
    summary = opt._angular_degree_residual_audit_summary(
        measured_two_theta_deg=[30.0],
        measured_phi_deg=[179.0],
        projected_two_theta_deg=[33.0],
        projected_phi_deg=[-179.0],
        delta_two_theta_deg=[3.0],
        wrapped_delta_phi_deg=[2.0],
        weighted_delta_two_theta_deg=[7.0],
        weighted_delta_phi_deg=[4.0],
        priority_weights=[2.0],
        solver_residual_vector=[7.0, 4.0],
    )

    assert summary["weighted_angular_recompute_failure_count"] == 1
    assert summary["optimizer_point_component_failure_count"] == 1
    assert summary["weighted_angular_failure_count"] == 2
    assert math.isnan(summary["weighted_angular_rms_weighted_deg"])


def test_exact_caked_residual_audit_uses_matched_point_components_not_full_prefix() -> None:
    summary = opt._angular_degree_residual_audit_summary(
        measured_two_theta_deg=[30.0],
        measured_phi_deg=[179.0],
        projected_two_theta_deg=[33.0],
        projected_phi_deg=[-179.0],
        delta_two_theta_deg=[3.0],
        wrapped_delta_phi_deg=[2.0],
        weighted_delta_two_theta_deg=[6.0],
        weighted_delta_phi_deg=[4.0],
        priority_weights=[2.0],
        solver_residual_vector=[50.0, 0.0, 6.0, 4.0],
        optimizer_point_residual_vector=[6.0, 4.0],
    )

    assert summary["optimizer_component_count"] == 4
    assert summary["optimizer_point_component_count"] == 2
    assert summary["optimizer_point_component_failure_count"] == 0
    assert summary["weighted_angular_failure_count"] == 0
    assert summary["weighted_angular_rms_weighted_deg"] == pytest.approx(math.hypot(6.0, 4.0))


def test_exact_caked_merged_diagnostics_reject_weighted_recompute_mismatch() -> None:
    summary = opt._merged_angular_degree_residual_audit_summary(
        [],
        [
            {
                "match_status": "matched",
                "measured_two_theta_deg": 30.0,
                "measured_phi_deg": 179.0,
                "projected_two_theta_deg": 33.0,
                "projected_phi_deg": -179.0,
                "delta_two_theta_deg": 3.0,
                "wrapped_delta_phi_deg": 2.0,
                "weighted_delta_two_theta_deg": 7.0,
                "weighted_delta_phi_deg": 4.0,
                "weight": 2.0,
                "solver_residual_vector": [6.0, 4.0],
            }
        ],
    )

    assert summary["weighted_angular_recompute_failure_count"] == 1
    assert summary["weighted_angular_failure_count"] == 1
    assert math.isnan(summary["weighted_angular_rms_weighted_deg"])


def test_exact_caked_merged_diagnostics_uses_diagnostic_point_components() -> None:
    summary = opt._merged_angular_degree_residual_audit_summary(
        [],
        [
            {
                "match_status": "matched",
                "measured_two_theta_deg": 30.0,
                "measured_phi_deg": 179.0,
                "projected_two_theta_deg": 33.0,
                "projected_phi_deg": -179.0,
                "delta_two_theta_deg": 3.0,
                "wrapped_delta_phi_deg": 2.0,
                "weighted_delta_two_theta_deg": 6.0,
                "weighted_delta_phi_deg": 4.0,
                "weight": 2.0,
                "solver_residual_vector": [6.0, 4.0],
            }
        ],
        solver_residual_vector=[50.0, 0.0, 6.0, 4.0],
    )

    assert summary["optimizer_component_count"] == 4
    assert summary["optimizer_point_component_count"] == 2
    assert summary["optimizer_point_component_failure_count"] == 0
    assert summary["weighted_angular_failure_count"] == 0


def test_exact_caked_merged_diagnostics_preserve_dataset_audit_failures() -> None:
    dataset_summary = opt._angular_degree_residual_audit_summary(
        measured_two_theta_deg=[30.0, 31.0],
        measured_phi_deg=[0.0, 1.0],
        projected_two_theta_deg=[31.0, 32.0],
        projected_phi_deg=[1.0, 2.0],
        delta_two_theta_deg=[1.0],
        wrapped_delta_phi_deg=[1.0],
        weighted_delta_two_theta_deg=[1.0],
        weighted_delta_phi_deg=[1.0],
        priority_weights=[1.0],
        solver_residual_vector=[1.0, 1.0],
    )
    summary = opt._merged_angular_degree_residual_audit_summary(
        [dataset_summary],
        [
            {
                "match_status": "matched",
                "measured_two_theta_deg": 30.0,
                "measured_phi_deg": 0.0,
                "projected_two_theta_deg": 31.0,
                "projected_phi_deg": 1.0,
                "delta_two_theta_deg": 1.0,
                "wrapped_delta_phi_deg": 1.0,
                "weighted_delta_two_theta_deg": 1.0,
                "weighted_delta_phi_deg": 1.0,
                "weight": 1.0,
                "solver_residual_vector": [1.0, 1.0],
            }
        ],
        solver_residual_vector=[1.0, 1.0],
    )

    assert dataset_summary["raw_angular_range_failure_count"] == 1
    assert summary["raw_angular_range_failure_count"] == 1
    assert summary["raw_angular_range_sanity_ok"] is False
    assert summary["raw_angular_sanity_ok"] is False


def test_native_detector_projector_shape_mismatch_returns_invalid_payload() -> None:
    from ra_sim.gui import geometry_fit as gui_geometry_fit

    def projector(cols, rows, *, local_params, anchor_kind, input_frame):
        del cols, rows, local_params, anchor_kind, input_frame
        return {
            "two_theta_deg": np.array([1.0], dtype=np.float64),
            "phi_deg": np.array([2.0, 3.0], dtype=np.float64),
            "valid": True,
        }

    result = gui_geometry_fit.project_geometry_fit_native_detector_points_to_caked_space(
        projector,
        [[10.0, 20.0], [30.0, 40.0]],
        local_params={},
    )

    assert result["valid"] is False
    assert result["invalid_reason"] == "projector_shape_mismatch"
    assert np.asarray(result["caked_points"], dtype=np.float64).shape == (2, 2)
    assert np.all(np.isnan(result["caked_points"]))


def test_caked_point_reprojection_residual_path_uses_native_detector_projector(
    monkeypatch,
) -> None:
    projector_calls = []

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, 12.0, 12.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def projector(cols, rows, *, local_params, anchor_kind, input_frame):
        cols_arr = np.asarray(cols, dtype=np.float64).reshape(-1)
        rows_arr = np.asarray(rows, dtype=np.float64).reshape(-1)
        theta = float(local_params.get("theta_initial", 0.0))
        distance = float(local_params.get("corto_detector", 0.0))
        projector_calls.append(
            {
                "anchor_kind": str(anchor_kind),
                "input_frame": str(input_frame),
                "cols": cols_arr.tolist(),
                "rows": rows_arr.tolist(),
                "theta_initial": theta,
                "corto_detector": distance,
            }
        )
        return {
            "two_theta_deg": cols_arr * 0.01 + theta * 0.5 + distance * 0.001,
            "phi_deg": rows_arr * 0.01 + theta * 0.05 + distance * 0.0002,
            "fit_space_source": "dataset_fit_space_projector",
            "input_frame": str(input_frame),
            "fit_space_projector_kind": "exact_caked_bundle",
            "cake_bundle_signature": f"sig-{anchor_kind}-{theta:.6f}-{distance:.6f}",
            "fit_space_local_params_signature": f"lp-{theta:.6f}-{distance:.6f}",
            "valid": True,
            "invalid_reason": None,
            "native_frame_conversion_source": f"test-{input_frame}",
            "native_frame_conversion_count": 0 if input_frame == "native_detector" else 1,
            "native_cols": cols_arr,
            "native_rows": rows_arr,
        }

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(
        opt,
        "_detector_pixels_to_fit_space",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("analytic detector fit-space path must not run")
        ),
    )

    subset = opt.ReflectionSimulationSubset(
        miller=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        measured_entries=[
            {
                "label": "peak-0",
                "hkl": (1, 0, 0),
                "overlay_match_index": 0,
                "source_table_index": 0,
                "source_row_index": 0,
                "fit_source_identity_only": True,
                "native_col": 30.0,
                "native_row": 40.0,
                "caked_x": -999999.0,
                "caked_y": 999999.0,
                "raw_caked_x": -888888.0,
                "raw_caked_y": 888888.0,
                "background_two_theta_deg": -777777.0,
                "background_phi_deg": 777777.0,
            }
        ],
        original_indices=np.array([0], dtype=np.int64),
        total_reflection_count=1,
        fixed_source_reflection_count=1,
        fallback_hkl_count=0,
        reduced=False,
    )
    dataset_ctx = opt.GeometryFitDatasetContext(
        dataset_index=0,
        label="bg0",
        theta_initial=0.0,
        subset=subset,
        experimental_image=np.zeros((32, 32), dtype=np.float64),
        fit_space_projector=projector,
        fit_space_projector_kind="exact_caked_bundle",
    )
    local = _base_params(32)
    local["pixel_size"] = 1.0
    local["corto_detector"] = 100.0
    local["theta_initial"] = 2.0
    theta_local = dict(local)
    theta_local["theta_initial"] = 2.1
    distance_local = dict(local)
    distance_local["corto_detector"] = 101.0

    def _evaluate(local_params):
        residual, diagnostics, summary = opt._evaluate_geometry_fit_dataset_dynamic_point_matches(
            local_params,
            dataset_ctx,
            image_size=32,
            missing_pair_penalty_deg=5.0,
            theta_value=float(local_params["theta_initial"]),
            collect_diagnostics=True,
        )
        assert np.all(np.isfinite(residual))
        assert summary["exact_fit_space_projector_available"] is True
        assert diagnostics[0]["measured_fit_space_source"] == "dataset_fit_space_projector"
        assert diagnostics[0]["measured_detector_input_frame"] == "native_detector"
        assert diagnostics[0]["fit_space_projector_kind"] == "exact_caked_bundle"
        assert str(diagnostics[0]["measured_cake_bundle_signature"]).startswith("sig-measured-")
        assert diagnostics[0]["measured_native_frame_conversion_count"] == 0
        assert diagnostics[0]["measured_two_theta_deg"] != pytest.approx(-777777.0)
        assert np.isfinite(diagnostics[0]["measured_two_theta_deg"])
        assert np.isfinite(diagnostics[0]["measured_phi_deg"])
        return diagnostics[0]

    base_diag = _evaluate(local)
    theta_diag = _evaluate(theta_local)
    distance_diag = _evaluate(distance_local)

    assert theta_diag["measured_two_theta_deg"] != pytest.approx(
        base_diag["measured_two_theta_deg"]
    )
    assert distance_diag["measured_two_theta_deg"] != pytest.approx(
        base_diag["measured_two_theta_deg"]
    )
    measured_calls = [call for call in projector_calls if call["anchor_kind"] == "measured"]
    assert [call["input_frame"] for call in measured_calls] == [
        "native_detector",
        "native_detector",
        "native_detector",
    ]
    assert measured_calls[0]["cols"] == [30.0]
    assert measured_calls[0]["rows"] == [40.0]


def test_dynamic_point_match_invalid_exact_projector_marks_row_invalid_without_analytic_fallback(
    monkeypatch,
):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, 12.0, 12.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def projector(cols, rows, *, local_params, anchor_kind, input_frame):
        cols_arr = np.asarray(cols, dtype=np.float64).reshape(-1)
        rows_arr = np.asarray(rows, dtype=np.float64).reshape(-1)
        return {
            "two_theta_deg": np.full(cols_arr.shape, np.nan, dtype=np.float64),
            "phi_deg": np.full(rows_arr.shape, np.nan, dtype=np.float64),
            "fit_space_source": "dataset_fit_space_projector",
            "input_frame": str(input_frame),
            "fit_space_projector_kind": "exact_caked_bundle",
            "cake_bundle_signature": f"sig-{anchor_kind}",
            "fit_space_local_params_signature": "lp",
            "valid": False,
            "invalid_reason": f"bad_{anchor_kind}",
            "native_frame_conversion_source": f"test-{input_frame}",
            "native_frame_conversion_count": 0 if input_frame == "native_detector" else 1,
            "native_cols": cols_arr,
            "native_rows": rows_arr,
        }

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(
        opt,
        "_detector_pixels_to_fit_space",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("analytic detector fit-space path must not run")
        ),
    )

    subset = opt.ReflectionSimulationSubset(
        miller=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        measured_entries=[
            {
                "label": "peak-0",
                "hkl": (1, 0, 0),
                "overlay_match_index": 0,
                "source_table_index": 0,
                "source_row_index": 0,
                "fit_source_identity_only": True,
                "native_col": 30.0,
                "native_row": 40.0,
            }
        ],
        original_indices=np.array([0], dtype=np.int64),
        total_reflection_count=1,
        fixed_source_reflection_count=1,
        fallback_hkl_count=0,
        reduced=False,
    )
    dataset_ctx = opt.GeometryFitDatasetContext(
        dataset_index=0,
        label="bg0",
        theta_initial=0.0,
        subset=subset,
        experimental_image=np.zeros((32, 32), dtype=np.float64),
        fit_space_projector=projector,
        fit_space_projector_kind="exact_caked_bundle",
    )
    local = _base_params(32)
    local["pixel_size"] = 1.0
    local["corto_detector"] = 100.0

    residual, diagnostics, summary = opt._evaluate_geometry_fit_dataset_dynamic_point_matches(
        local,
        dataset_ctx,
        image_size=32,
        missing_pair_penalty_deg=5.0,
        theta_value=0.0,
        collect_diagnostics=True,
    )

    assert len(diagnostics) == 1
    assert diagnostics[0]["valid"] is False
    assert diagnostics[0]["measured_fit_space_source"] == "invalid_dataset_fit_space_projector"
    assert diagnostics[0]["measured_invalid_projection_reason"] == "bad_measured"
    assert diagnostics[0]["invalid_projection_reason"] == "bad_measured"
    assert summary["exact_fit_space_projector_available"] is True
    assert int(summary["invalid_dataset_fit_space_projector_row_count"]) >= 1
    assert np.all(np.isnan(residual) | np.isfinite(residual))


def test_detector_pixels_to_fit_space_matches_zero_tilt_geometry() -> None:
    cols = np.array([10.0, 13.0, 10.0], dtype=np.float64)
    rows = np.array([7.0, 10.0, 13.0], dtype=np.float64)
    center = [10.0, 10.0]
    detector_distance = 5.0
    pixel_size = 2.0

    two_theta, phi = opt._detector_pixels_to_fit_space(
        cols,
        rows,
        center=center,
        detector_distance=detector_distance,
        pixel_size=pixel_size,
    )

    x = (cols - float(center[1])) * pixel_size
    z = (float(center[0]) - rows) * pixel_size
    expected_two_theta = np.degrees(np.arctan2(np.hypot(x, z), np.full_like(x, detector_distance)))
    expected_phi = np.degrees(np.arctan2(x, z))
    expected_phi = (expected_phi + 180.0) % 360.0 - 180.0

    assert np.allclose(two_theta, expected_two_theta, atol=1.0e-12)
    assert np.allclose(phi, expected_phi, atol=1.0e-12)
    assert phi[1] == 90.0
    assert phi[-1] == -180.0


def test_pixel_to_angles_matches_zero_tilt_single_pixel_formula() -> None:
    col = 13.0
    row = 10.0
    center = [10.0, 10.0]
    detector_distance = 5.0
    pixel_size = 2.0

    single_two_theta, single_phi = opt._pixel_to_angles(
        col,
        row,
        center,
        detector_distance,
        pixel_size,
    )

    x = (col - float(center[1])) * pixel_size
    z = (float(center[0]) - row) * pixel_size
    expected_two_theta = float(np.degrees(np.arctan2(np.hypot(x, z), detector_distance)))
    expected_phi = float(np.degrees(np.arctan2(x, z)))
    expected_phi = (expected_phi + 180.0) % 360.0 - 180.0

    assert single_two_theta == pytest.approx(expected_two_theta)
    assert single_phi == pytest.approx(expected_phi)
    assert single_phi == 90.0


def test_detector_pixels_to_fit_space_ignores_gamma_and_Gamma() -> None:
    cols = np.array([8.0, 12.0, 15.0], dtype=np.float64)
    rows = np.array([6.0, 10.0, 14.0], dtype=np.float64)
    center = [10.0, 10.0]

    base_two_theta, base_phi = opt._detector_pixels_to_fit_space(
        cols,
        rows,
        center=center,
        detector_distance=5.0,
        pixel_size=2.0,
    )
    tilted_two_theta, tilted_phi = opt._detector_pixels_to_fit_space(
        cols,
        rows,
        center=center,
        detector_distance=5.0,
        pixel_size=2.0,
        gamma_deg=3.5,
        Gamma_deg=-7.25,
    )

    assert np.allclose(tilted_two_theta, base_two_theta, atol=1.0e-12)
    assert np.allclose(tilted_phi, base_phi, atol=1.0e-12)


def test_measured_fit_space_anchor_prefers_background_detector_anchor_over_cached_fit_space() -> (
    None
):
    center = [10.0, 10.0]
    entry = {
        "detector_x": 13.0,
        "detector_y": 7.0,
        "background_detector_x": 14.0,
        "background_detector_y": 6.0,
        "background_two_theta_deg": 91.0,
        "background_phi_deg": -42.0,
    }

    anchor, reason, metadata = opt._measured_fit_space_anchor(
        entry,
        center=center,
        detector_distance=5.0,
        pixel_size=2.0,
    )
    expected = opt._pixel_to_angles(14.0, 6.0, center, 5.0, 2.0)

    assert reason == "background_detector_fit_space_anchor"
    assert anchor == pytest.approx(expected)
    assert metadata["anchor_source"] == "background_detector_fit_space_anchor"

    fallback_entry = {
        "background_detector_x": 14.0,
        "background_detector_y": 6.0,
        "background_two_theta_deg": 91.0,
        "background_phi_deg": -42.0,
    }
    fallback_anchor, fallback_reason, fallback_metadata = opt._measured_fit_space_anchor(
        fallback_entry,
        center=center,
        detector_distance=5.0,
        pixel_size=2.0,
    )
    expected_fallback = opt._pixel_to_angles(14.0, 6.0, center, 5.0, 2.0)

    assert fallback_reason == "background_detector_fit_space_anchor"
    assert fallback_anchor == pytest.approx(expected_fallback)
    assert fallback_metadata["anchor_source"] == "background_detector_fit_space_anchor"


def test_measured_fit_space_anchor_ignores_gamma_and_Gamma() -> None:
    entry = {
        "detector_x": 13.0,
        "detector_y": 7.0,
    }

    base_anchor, base_reason, _base_metadata = opt._measured_fit_space_anchor(
        entry,
        center=[10.0, 10.0],
        detector_distance=5.0,
        pixel_size=2.0,
    )
    tilted_anchor, tilted_reason, tilted_metadata = opt._measured_fit_space_anchor(
        entry,
        center=[10.0, 10.0],
        detector_distance=5.0,
        pixel_size=2.0,
        gamma_deg=8.0,
        Gamma_deg=-4.0,
    )

    assert base_reason == "detector_fit_space_anchor"
    assert tilted_reason == "detector_fit_space_anchor"
    assert tilted_anchor == pytest.approx(base_anchor)
    assert tilted_metadata["anchor_source"] == "detector_fit_space_anchor"


def test_measured_fit_space_anchor_keeps_cached_fit_space_stable_when_wavelength_changes(
    monkeypatch,
) -> None:
    def fake_theoretical_two_theta(entry, *, a_lattice, c_lattice, wavelength):
        del entry, a_lattice, c_lattice
        if wavelength == pytest.approx(1.0):
            return 21.0
        return 33.0

    monkeypatch.setattr(
        opt,
        "_entry_theoretical_two_theta_deg",
        fake_theoretical_two_theta,
    )

    anchor, reason, metadata = opt._measured_fit_space_anchor(
        {
            "background_two_theta_deg": 20.0,
            "background_phi_deg": 5.0,
            "background_reference_a": 4.0,
            "background_reference_c": 7.0,
            "background_reference_lambda": 1.0,
        },
        center=[10.0, 10.0],
        detector_distance=5.0,
        pixel_size=2.0,
        a_lattice=4.0,
        c_lattice=7.0,
        wavelength=1.3,
    )

    assert reason == "cached_fit_space_anchor"
    assert anchor == pytest.approx((20.0, 5.0))
    assert metadata["two_theta_adjustment_deg"] == 0.0
    assert metadata["reference_two_theta_deg"] == pytest.approx(21.0)
    assert metadata["current_theoretical_two_theta_deg"] == pytest.approx(33.0)


def test_measured_fit_space_anchor_prefers_explicit_fit_space_override_over_detector_anchor() -> (
    None
):
    anchor, reason, metadata = opt._measured_fit_space_anchor(
        {
            "detector_x": 13.0,
            "detector_y": 7.0,
            "background_two_theta_deg": 20.0,
            "background_phi_deg": 5.0,
            "fit_space_anchor_override": True,
        },
        center=[10.0, 10.0],
        detector_distance=5.0,
        pixel_size=2.0,
    )

    assert reason == "cached_fit_space_anchor"
    assert anchor == pytest.approx((20.0, 5.0))
    assert metadata["anchor_source"] == "cached_fit_space_anchor"
    assert metadata["cached_two_theta_deg"] == pytest.approx(20.0)
    assert metadata["cached_phi_deg"] == pytest.approx(5.0)


def _exact_projector_dataset_ctx(projector):
    subset = opt.ReflectionSimulationSubset(
        miller=np.empty((0, 3), dtype=np.float64),
        intensities=np.empty((0,), dtype=np.float64),
        measured_entries=[],
        original_indices=np.empty((0,), dtype=np.int64),
        total_reflection_count=0,
        fixed_source_reflection_count=0,
        fallback_hkl_count=0,
        reduced=False,
    )
    return opt.GeometryFitDatasetContext(
        dataset_index=0,
        label="bg0",
        theta_initial=0.0,
        subset=subset,
        fit_space_projector=projector,
        fit_space_projector_kind="exact_caked_bundle",
    )


def test_manual_caked_fit_target_anchor_uses_cached_two_theta_phi_even_with_exact_projector() -> (
    None
):
    def fail_projector(*_args, **_kwargs):
        raise AssertionError("manual caked target must not be reprojected")

    anchor, reason, metadata = opt._measured_fit_space_anchor(
        {
            "native_col": 13.0,
            "native_row": 7.0,
            "background_two_theta_deg": 20.0,
            "background_phi_deg": 5.0,
            "fit_space_anchor_override": True,
        },
        center=[10.0, 10.0],
        detector_distance=5.0,
        pixel_size=2.0,
        dataset_ctx=_exact_projector_dataset_ctx(fail_projector),
        local_params={"center": [10.0, 10.0], "theta_initial": 1.0},
    )

    assert reason == "cached_fit_space_anchor"
    assert anchor == pytest.approx((20.0, 5.0))
    assert metadata["anchor_source"] == "cached_fit_space_anchor"
    assert metadata["fit_space_source"] == "cached_fit_space_anchor"
    assert metadata["two_theta_adjustment_deg"] == 0.0
    assert metadata["valid"] is True


@pytest.mark.parametrize(
    ("center", "distance", "gamma", "Gamma", "local_params"),
    [
        ([10.0, 10.0], 5.0, 0.0, 0.0, {"theta_initial": 1.0}),
        ([11.0, 10.0], 5.0, 0.0, 0.0, {"theta_initial": 1.0}),
        ([10.0, 10.0], 6.0, 0.0, 0.0, {"theta_initial": 1.0}),
        ([10.0, 10.0], 5.0, 2.0, 0.0, {"theta_initial": 1.0}),
        ([10.0, 10.0], 5.0, 0.0, -3.0, {"theta_initial": 1.0}),
        ([10.0, 10.0], 5.0, 0.0, 0.0, {"theta_initial": 1.25}),
    ],
)
def test_manual_caked_fit_target_anchor_is_immutable_under_trial_geometry(
    center,
    distance,
    gamma,
    Gamma,
    local_params,
) -> None:
    projector_calls = 0

    def tracking_projector(*_args, **_kwargs):
        nonlocal projector_calls
        projector_calls += 1
        return {
            "two_theta_deg": np.asarray([999.0], dtype=np.float64),
            "phi_deg": np.asarray([111.0], dtype=np.float64),
            "native_cols": np.asarray([13.0], dtype=np.float64),
            "native_rows": np.asarray([7.0], dtype=np.float64),
            "fit_space_source": "dataset_fit_space_projector",
            "fit_space_projector_kind": "exact_caked_bundle",
            "valid": True,
        }

    anchor, reason, metadata = opt._measured_fit_space_anchor(
        {
            "native_col": 13.0,
            "native_row": 7.0,
            "background_two_theta_deg": 20.0,
            "background_phi_deg": 5.0,
            "fit_space_anchor_override": True,
        },
        center=center,
        detector_distance=float(distance),
        pixel_size=2.0,
        gamma_deg=float(gamma),
        Gamma_deg=float(Gamma),
        dataset_ctx=_exact_projector_dataset_ctx(tracking_projector),
        local_params=local_params,
    )

    assert projector_calls == 0
    assert reason == "cached_fit_space_anchor"
    assert anchor == pytest.approx((20.0, 5.0))
    assert metadata["cached_two_theta_deg"] == pytest.approx(20.0)
    assert metadata["cached_phi_deg"] == pytest.approx(5.0)


@pytest.mark.parametrize(
    "entry_overrides",
    [
        {"background_two_theta_deg": float("nan"), "background_phi_deg": 5.0},
        {"background_two_theta_deg": 20.0, "background_phi_deg": float("nan")},
        {"background_two_theta_deg": None, "background_phi_deg": None},
    ],
)
def test_manual_caked_fit_target_anchor_override_nonfinite_cache_uses_existing_fallback(
    entry_overrides,
) -> None:
    projector_calls = 0

    def projector(cols, rows, *, local_params, anchor_kind, input_frame):
        nonlocal projector_calls
        del local_params, anchor_kind
        projector_calls += 1
        cols_arr = np.asarray(cols, dtype=np.float64).reshape(-1)
        rows_arr = np.asarray(rows, dtype=np.float64).reshape(-1)
        return {
            "two_theta_deg": np.asarray([31.0], dtype=np.float64),
            "phi_deg": np.asarray([-9.0], dtype=np.float64),
            "native_cols": cols_arr,
            "native_rows": rows_arr,
            "fit_space_source": "dataset_fit_space_projector",
            "fit_space_projector_kind": "exact_caked_bundle",
            "input_frame": str(input_frame),
            "fit_space_local_params_signature": "unit-test",
            "cake_bundle_signature": "unit-test",
            "valid": True,
            "invalid_reason": None,
            "native_frame_conversion_source": "",
            "native_frame_conversion_count": 0,
        }

    entry = {
        "native_col": 13.0,
        "native_row": 7.0,
        "background_two_theta_deg": 20.0,
        "background_phi_deg": 5.0,
        "fit_space_anchor_override": True,
    }
    entry.update(entry_overrides)

    anchor, reason, metadata = opt._measured_fit_space_anchor(
        entry,
        center=[10.0, 10.0],
        detector_distance=5.0,
        pixel_size=2.0,
        dataset_ctx=_exact_projector_dataset_ctx(projector),
        local_params={"center": [10.0, 10.0], "theta_initial": 1.0},
    )

    assert projector_calls == 1
    assert reason == "dataset_fit_space_projector"
    assert anchor == pytest.approx((31.0, -9.0))
    assert metadata["anchor_source"] == "dataset_fit_space_projector"
    assert metadata["fit_space_source"] == "dataset_fit_space_projector"


def test_fit_space_provenance_summary_counts_native_detector_anchor_sources() -> None:
    summary = opt._fit_space_provenance_summary(
        {},
        cached_anchor_count=1,
        detector_anchor_count=99,
        anchor_source_counts={
            "cached_fit_space_anchor": 1,
            "native_fit_space_anchor": 2,
            "background_detector_fit_space_anchor": 3,
            "detector_fit_space_anchor": 4,
            "display_fit_space_anchor": 5,
        },
    )

    assert int(summary["fit_space_anchor_count_cached"]) == 1
    assert int(summary["fit_space_anchor_count_detector"]) == 14
    assert summary["fit_space_anchor_source_counts"] == {
        "cached_fit_space_anchor": 1,
        "dataset_fit_space_projector": 0,
        "invalid_dataset_fit_space_projector": 0,
        "native_fit_space_anchor": 2,
        "background_detector_fit_space_anchor": 3,
        "detector_fit_space_anchor": 4,
        "display_fit_space_anchor": 5,
    }


def test_dynamic_point_match_reanchor_does_not_mutate_measured_entries(monkeypatch):
    callback_entries: list[dict[str, object]] = []

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, 12.0, 12.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_reanchor(
        measured_entry,
        simulated_detector_point,
        *,
        local_params=None,
        dataset_ctx=None,
    ):
        del local_params, dataset_ctx
        callback_entries.append(dict(measured_entry))
        return {
            "x": float(simulated_detector_point[0]),
            "y": float(simulated_detector_point[1]),
            "detector_x": float(simulated_detector_point[0]),
            "detector_y": float(simulated_detector_point[1]),
            "measured_reanchor_motion_px": 3.0,
        }

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    original_entry = {
        "label": "peak-0",
        "hkl": (1, 0, 0),
        "overlay_match_index": 0,
        "source_table_index": 0,
        "source_row_index": 0,
        "detector_x": 6.0,
        "detector_y": 6.0,
        "background_detector_x": 6.0,
        "background_detector_y": 6.0,
        "x": 6.0,
        "y": 6.0,
    }
    subset = opt.ReflectionSimulationSubset(
        miller=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        measured_entries=[dict(original_entry)],
        original_indices=np.array([0], dtype=np.int64),
        total_reflection_count=1,
        fixed_source_reflection_count=1,
        fallback_hkl_count=0,
        reduced=False,
    )
    dataset_ctx = opt.GeometryFitDatasetContext(
        dataset_index=0,
        label="bg0",
        theta_initial=0.0,
        subset=subset,
        experimental_image=np.zeros((32, 32), dtype=np.float64),
        dynamic_reanchor_enabled=True,
        dynamic_reanchor_callback=fake_reanchor,
    )
    local = _base_params(32)
    local["pixel_size"] = 1.0
    local["corto_detector"] = 100.0

    for _ in range(2):
        residual, diagnostics, summary = opt._evaluate_geometry_fit_dataset_dynamic_point_matches(
            local,
            dataset_ctx,
            image_size=32,
            missing_pair_penalty_deg=5.0,
            theta_value=0.0,
            collect_diagnostics=True,
        )
        assert residual.shape == (2,)
        assert diagnostics[0]["measured_reanchor_status"] == "updated"
        assert summary["matched_pair_count"] == 1
        assert summary["measured_anchor_motion_mean_px"] == pytest.approx(3.0)
        assert summary["measured_anchor_motion_rms_px"] == pytest.approx(3.0)
        assert summary["measured_anchor_motion_max_px"] == pytest.approx(3.0)

    assert len(callback_entries) == 2
    assert callback_entries[0]["detector_x"] == 6.0
    assert callback_entries[1]["detector_x"] == 6.0
    assert subset.measured_entries[0] == original_entry


def test_resolve_parallel_worker_count_auto_reserves_two_threads(monkeypatch):
    monkeypatch.setattr(opt, "_available_parallel_thread_budget", lambda: 12)

    assert opt._resolve_parallel_worker_count("auto", max_tasks=32) == 10
    assert opt._resolve_parallel_worker_count(None, max_tasks=8) == 8


def test_resolve_parallel_worker_count_auto_keeps_one_worker_minimum(monkeypatch):
    monkeypatch.setattr(opt, "_available_parallel_thread_budget", lambda: 2)

    assert opt._resolve_parallel_worker_count("auto", max_tasks=32) == 1


def test_fit_geometry_parameters_cost_fn_uses_updated_psi_z(monkeypatch):
    target = 1.25
    psi_z_seen = []

    def fake_compute(*args, **kwargs):
        psi_z = float(kwargs["psi_z"])
        psi_z_seen.append(psi_z)
        return np.array([psi_z - target], dtype=np.float64)

    monkeypatch.setattr(opt, "compute_peak_position_error_geometry_local", fake_compute)

    image_size = 8
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=[],
        var_names=["psi_z"],
        experimental_image=None,
    )

    assert result.success
    assert abs(float(result.x[0]) - target) < 1e-8
    assert any(abs(v - target) < 1e-3 for v in psi_z_seen)


def test_fit_geometry_parameters_applies_parameter_priors(monkeypatch):
    target = 1.0

    def fake_compute(*args, **kwargs):
        gamma = float(args[0])
        return np.array([gamma - target], dtype=np.float64)

    monkeypatch.setattr(opt, "compute_peak_position_error_geometry_local", fake_compute)

    image_size = 8
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=[],
        var_names=["gamma"],
        experimental_image=None,
        refinement_config={
            "solver": {"restarts": 0},
            "priors": {"gamma": {"center": 0.0, "sigma": 0.25}},
        },
    )

    assert result.success
    assert 0.0 < float(result.x[0]) < 0.1
    assert isinstance(result.parameter_prior_summary, list)
    assert result.parameter_prior_summary == [{"name": "gamma", "center": 0.0, "sigma": 0.25}]


def test_fit_geometry_parameters_pixel_path_forwards_optics_mode(monkeypatch):
    optics_seen = []

    def fake_process(*args, **kwargs):
        optics_seen.append(kwargs.get("optics_mode"))
        return _fake_process_peaks(*args, **kwargs)

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    image_size = 8
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
    )

    assert result.success
    assert optics_seen
    assert all(mode == 1 for mode in optics_seen)


def test_fit_geometry_parameters_pixel_path_uses_central_geometry_ray(monkeypatch):
    process_calls = []

    def fake_process(*args, **kwargs):
        process_calls.append(
            {
                "kwargs": dict(kwargs),
                "wavelength_array": np.asarray(args[5], dtype=np.float64).copy(),
                "beam_x_array": np.asarray(args[16], dtype=np.float64).copy(),
                "beam_y_array": np.asarray(args[17], dtype=np.float64).copy(),
                "theta_array": np.asarray(args[18], dtype=np.float64).copy(),
                "phi_array": np.asarray(args[19], dtype=np.float64).copy(),
            }
        )
        return _fake_process_peaks(*args, **kwargs)

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 8
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {"restarts": 0},
            "full_beam_polish": {"enabled": False},
        },
    )

    assert result.success
    assert process_calls
    for call in process_calls:
        assert call["beam_x_array"].shape == (1,)
        assert call["beam_y_array"].shape == (1,)
        assert call["theta_array"].shape == (1,)
        assert call["phi_array"].shape == (1,)
        assert call["wavelength_array"].shape == (1,)
        assert np.allclose(call["beam_x_array"], [0.0])
        assert np.allclose(call["beam_y_array"], [0.0])
        assert np.allclose(call["theta_array"], [0.0])
        assert np.allclose(call["phi_array"], [0.0])
        assert np.allclose(call["wavelength_array"], [1.0])
        assert call["kwargs"].get("best_sample_indices_out") is None
    assert isinstance(result.point_match_summary, dict)
    assert bool(result.point_match_summary["central_ray_mode"]) is True
    assert bool(result.point_match_summary["single_ray_enabled"]) is False
    assert int(result.point_match_summary["single_ray_forced_count"]) == 0


def test_fit_geometry_parameters_pixel_path_runs_without_full_ray_polish(
    monkeypatch,
):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        gamma = float(args[8])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, 2.0 + 4.0 * gamma, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    solve_calls = {"count": 0}

    def fake_least_squares(residual_fn, x0, **kwargs):
        solve_calls["count"] += 1
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message=f"solve#{solve_calls['count']}",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 12
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {"restarts": 0, "stagnation_probe": False},
            "full_beam_polish": {"enabled": False},
        },
    )

    assert result.success
    assert solve_calls["count"] >= 1
    assert np.allclose(np.asarray(result.x, dtype=float), np.array([0.5]))
    assert np.isclose(float(result.cost), 0.0)
    assert isinstance(result.geometry_fit_debug_summary, dict)
    assert str(result.geometry_fit_debug_summary["main_solve_seed"]["seed_kind"]) == "u=0"
    assert isinstance(result.point_match_summary, dict)
    assert bool(result.point_match_summary["central_ray_mode"]) is True
    assert bool(result.point_match_summary["single_ray_enabled"]) is False


def test_process_peaks_wrapper_prefers_python_runner_when_numba_disabled(monkeypatch):
    process_calls = []

    def fake_process(*args, **kwargs):
        process_calls.append(
            {
                "kwargs": dict(kwargs),
                "wavelength_array": np.asarray(args[5], dtype=np.float64).copy(),
                "beam_x_array": np.asarray(args[16], dtype=np.float64).copy(),
                "beam_y_array": np.asarray(args[17], dtype=np.float64).copy(),
                "theta_array": np.asarray(args[18], dtype=np.float64).copy(),
                "phi_array": np.asarray(args[19], dtype=np.float64).copy(),
            }
        )
        return _fake_process_peaks(*args, **kwargs)

    monkeypatch.setattr(opt, "process_peaks_parallel", fake_process)
    monkeypatch.setattr(opt, "_USE_NUMBA_PROCESS_PEAKS", False)

    image_size = 8
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["mosaic_params"].update(
        {
            "beam_x_array": np.array([0.3, -0.2, 0.1], dtype=np.float64),
            "beam_y_array": np.array([0.4, -0.1, 0.2], dtype=np.float64),
            "theta_array": np.array([0.02, -0.03, 0.01], dtype=np.float64),
            "phi_array": np.array([0.04, -0.05, 0.02], dtype=np.float64),
            "wavelength_array": np.array([0.8, 1.2, 1.4], dtype=np.float64),
        }
    )
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {"restarts": 0},
            "single_ray": {"enabled": False},
            "use_numba": False,
            "full_beam_polish": {"enabled": False},
        },
    )

    assert result.success
    assert process_calls
    assert all(call["kwargs"].get("prefer_python_runner") is True for call in process_calls)
    for call in process_calls:
        assert np.allclose(call["beam_x_array"], [0.0])
        assert np.allclose(call["beam_y_array"], [0.0])
        assert np.allclose(call["theta_array"], [0.0])
        assert np.allclose(call["phi_array"], [0.0])
        assert np.allclose(call["wavelength_array"], [1.0])


def test_process_peaks_wrapper_serializes_first_numba_warmup(monkeypatch):
    process_calls = []
    call_gate = threading.Barrier(2)
    state_lock = threading.Lock()
    active_calls = 0
    max_active_calls = 0

    def fake_process(*args, **kwargs):
        nonlocal active_calls
        nonlocal max_active_calls
        with state_lock:
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
            process_calls.append(dict(kwargs))
        try:
            threading.Event().wait(0.05)
            return _fake_process_peaks(*args, **kwargs)
        finally:
            with state_lock:
                active_calls -= 1

    monkeypatch.setattr(opt, "process_peaks_parallel", fake_process)
    monkeypatch.setattr(opt, "_USE_NUMBA_PROCESS_PEAKS", True)
    monkeypatch.setattr(opt, "_NUMBA_PROCESS_PEAKS_WARMED", False)

    process_args = _process_wrapper_args(8)

    def run_once():
        call_gate.wait(timeout=5.0)
        return opt._process_peaks_parallel_safe(*process_args, save_flag=0)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_once) for _ in range(2)]
        for future in futures:
            image, *_ = future.result(timeout=5.0)
            assert image.shape == (8, 8)

    assert len(process_calls) == 2
    assert max_active_calls == 1
    assert opt._NUMBA_PROCESS_PEAKS_WARMED is True


def test_simulate_forward_warmup_serializes_first_compile(monkeypatch):
    call_lock = threading.Lock()
    active_warmup_calls = 0
    max_active_warmup_calls = 0
    run_calls = 0
    warmup_started = threading.Event()
    warmup_complete = threading.Event()

    def fake_run_simulation_request(request: SimulationRequest, *, peak_runner=None):
        nonlocal active_warmup_calls, max_active_warmup_calls, run_calls
        if int(request.geometry.image_size) == 2:
            with call_lock:
                active_warmup_calls += 1
                max_active_warmup_calls = max(max_active_warmup_calls, active_warmup_calls)
                run_calls += 1
            warmup_started.set()
            warmup_complete.wait(timeout=5.0)
            with call_lock:
                active_warmup_calls -= 1
        else:
            with call_lock:
                run_calls += 1

        return SimulationResult(
            image=np.zeros(
                (int(request.geometry.image_size), int(request.geometry.image_size)),
                dtype=np.float64,
            ),
            hit_tables=[],
            q_data=np.empty((0, 0, 0), dtype=np.float64),
            q_count=np.empty(0, dtype=np.float64),
            all_status=np.empty(0, dtype=np.float64),
            miss_tables=[],
        )

    monkeypatch.setattr(sim_engine, "_FORWARD_SIMULATION_NUMBA_WARMED", False)
    monkeypatch.setattr(sim_engine, "_FORWARD_SIMULATION_NUMBA_WARMUP_THREAD", None)
    monkeypatch.setattr(
        sim_engine,
        "_run_simulation_request",
        fake_run_simulation_request,
    )

    request = _build_forward_warmup_request()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(sim_engine.simulate, request) for _ in range(2)]
        warmup_started.wait(timeout=5.0)
        warmup_complete.set()
        for future in futures:
            future.result(timeout=5.0)

    assert max_active_warmup_calls == 1
    assert run_calls >= 3
    assert sim_engine._FORWARD_SIMULATION_NUMBA_WARMED is True


def test_process_peaks_wrapper_disables_numba_after_python_fallback(monkeypatch):
    process_calls = []
    safe_stats = {"used_python_runner": True}

    def fake_safe_wrapper(*args, **kwargs):
        process_calls.append(dict(kwargs))
        return _fake_process_peaks(*args, **kwargs)

    monkeypatch.setattr(opt, "process_peaks_parallel", fake_safe_wrapper)
    monkeypatch.setattr(opt, "_DIFFRACTION_PROCESS_PEAKS_SAFE_WRAPPER", fake_safe_wrapper)
    monkeypatch.setattr(
        opt,
        "get_last_process_peaks_safe_stats",
        lambda: dict(safe_stats),
    )
    monkeypatch.setattr(opt, "_USE_NUMBA_PROCESS_PEAKS", True)
    monkeypatch.setattr(opt, "_NUMBA_PROCESS_PEAKS_WARMED", False)

    process_args = _process_wrapper_args(8)

    opt._process_peaks_parallel_safe(*process_args, save_flag=0)

    assert opt._USE_NUMBA_PROCESS_PEAKS is False
    assert opt._NUMBA_PROCESS_PEAKS_WARMED is False
    assert process_calls[0].get("prefer_python_runner") is None

    opt._process_peaks_parallel_safe(*process_args, save_flag=0)

    assert process_calls[1].get("prefer_python_runner") is True


def test_fit_geometry_parameters_pixel_path_restricts_simulation_to_selected_reflections(
    monkeypatch,
):
    process_calls = []

    def fake_process(*args, **kwargs):
        process_calls.append(
            {
                "miller": np.asarray(args[0], dtype=np.float64).copy(),
                "wavelength_array": np.asarray(args[5], dtype=np.float64).copy(),
                "beam_x_array": np.asarray(args[16], dtype=np.float64).copy(),
                "beam_y_array": np.asarray(args[17], dtype=np.float64).copy(),
                "theta_array": np.asarray(args[18], dtype=np.float64).copy(),
                "phi_array": np.asarray(args[19], dtype=np.float64).copy(),
                "kwargs": dict(kwargs),
            }
        )
        miller_arg = np.asarray(args[0], dtype=np.float64)
        image_size = int(args[2])
        hit_tables = []
        for row in miller_arg:
            hit_tables.append(
                np.array(
                    [[1.0, 5.0, 4.0, 0.0, row[0], row[1], row[2]]],
                    dtype=np.float64,
                )
            )
        best_sample_indices_out = kwargs.get("best_sample_indices_out")
        if isinstance(best_sample_indices_out, np.ndarray):
            best_sample_indices_out[:] = 0
        image = np.zeros((image_size, image_size), dtype=np.float64)
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 12
    miller = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 5.0,
            "y": 4.0,
            "source_table_index": 1,
            "source_row_index": 0,
        }
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {"restarts": 0},
            "full_beam_polish": {"enabled": False},
        },
    )

    assert result.success
    assert process_calls
    assert all(call["miller"].shape == (1, 3) for call in process_calls)
    assert all(
        np.allclose(call["miller"], np.array([[2.0, 0.0, 0.0]], dtype=np.float64))
        for call in process_calls
    )
    for call in process_calls:
        assert np.allclose(call["beam_x_array"], [0.0])
        assert np.allclose(call["beam_y_array"], [0.0])
        assert np.allclose(call["theta_array"], [0.0])
        assert np.allclose(call["phi_array"], [0.0])
        assert np.allclose(call["wavelength_array"], [1.0])
        assert call["kwargs"].get("best_sample_indices_out") is None
    assert isinstance(result.point_match_summary, dict)
    assert int(result.point_match_summary["simulated_reflection_count"]) == 1
    assert int(result.point_match_summary["total_reflection_count"]) == 3
    assert bool(result.point_match_summary["subset_reduced"]) is True
    assert bool(result.point_match_summary["central_ray_mode"]) is True
    assert bool(result.point_match_summary["single_ray_enabled"]) is False
    assert int(result.point_match_summary["single_ray_forced_count"]) == 0


def test_prepare_reflection_subset_preserves_distinct_reflections_within_one_q_group() -> None:
    miller = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    measured = [
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0",
            "x": 4.0,
            "y": 4.0,
            "q_group_key": ["q", 1],
            "source_table_index": 0,
            "source_reflection_index": 0,
            "source_reflection_is_full": True,
            "source_row_index": 0,
        },
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 4.2,
            "y": 4.1,
            "q_group_key": ("q", 1),
            "source_table_index": 1,
            "source_reflection_index": 1,
            "source_reflection_is_full": True,
            "source_row_index": 0,
        },
        {
            "hkl": (3, 0, 0),
            "label": "3,0,0",
            "x": 6.0,
            "y": 5.0,
            "q_group_key": ("q", 2),
            "source_table_index": 2,
            "source_reflection_index": 2,
            "source_reflection_is_full": True,
            "source_row_index": 0,
        },
    ]

    subset = opt._prepare_reflection_subset(miller, intensities, measured)

    assert subset.reduced is False
    assert np.allclose(
        subset.miller,
        np.array(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            dtype=np.float64,
        ),
    )
    assert np.array_equal(subset.original_indices, np.array([0, 1, 2], dtype=np.int64))
    assert len(subset.measured_entries) == 3
    assert [entry["q_group_key"] for entry in subset.measured_entries] == [
        ("q", 1),
        ("q", 1),
        ("q", 2),
    ]
    assert [entry["source_table_index"] for entry in subset.measured_entries] == [
        0,
        1,
        2,
    ]


def test_prepare_reflection_subset_rebinds_stale_source_identity_by_hkl() -> None:
    miller = np.array(
        [
            [5.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([5.0, 2.0, 7.0], dtype=np.float64)
    measured = [
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 0,
            "source_row_index": 0,
            "fit_source_identity_only": True,
        }
    ]

    subset = opt._prepare_reflection_subset(miller, intensities, measured)

    assert subset.reduced is True
    assert subset.fixed_source_reflection_count == 0
    assert subset.fallback_hkl_count == 1
    assert np.array_equal(subset.original_indices, np.array([1], dtype=np.int64))
    assert np.allclose(subset.miller, np.array([[2.0, 0.0, 0.0]], dtype=np.float64))
    for key in (
        "source_table_index",
        "source_reflection_index",
        "resolved_table_index",
        "source_row_index",
        "source_peak_index",
    ):
        assert key not in subset.measured_entries[0]


def test_prepare_reflection_subset_prefers_source_reflection_index_over_stale_table_index() -> None:
    miller = np.array(
        [
            [5.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([5.0, 2.0, 7.0], dtype=np.float64)
    measured = [
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 0,
            "source_reflection_index": 1,
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "fit_source_identity_only": True,
        }
    ]

    subset = opt._prepare_reflection_subset(miller, intensities, measured)

    assert subset.reduced is True
    assert subset.fixed_source_reflection_count == 1
    assert subset.fallback_hkl_count == 0
    assert np.array_equal(subset.original_indices, np.array([1], dtype=np.int64))
    assert np.allclose(subset.miller, np.array([[2.0, 0.0, 0.0]], dtype=np.float64))
    assert subset.measured_entries[0]["source_table_index"] == 0
    assert subset.measured_entries[0]["source_reflection_index"] == 1
    assert subset.measured_entries[0]["resolved_table_index"] == 0
    assert subset.measured_entries[0]["source_row_index"] == 0


def test_prepare_reflection_subset_preserves_trusted_identity_and_only_remaps_local_lookup_ids() -> (
    None
):
    miller = np.array(
        [
            [5.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([5.0, 2.0, 7.0], dtype=np.float64)
    measured = [
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 0,
            "source_reflection_index": 1,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "resolved_table_index": 99,
            "resolved_peak_index": 99,
            "fit_source_identity_only": True,
        }
    ]

    subset = opt._prepare_reflection_subset(miller, intensities, measured)

    assert subset.reduced is True
    assert subset.fixed_source_reflection_count == 1
    assert subset.fallback_hkl_count == 0
    entry = subset.measured_entries[0]
    assert entry["source_reflection_index"] == 1
    assert entry["source_reflection_namespace"] == "full_reflection"
    assert entry["source_reflection_is_full"] is True
    assert entry["source_branch_index"] == 1
    assert entry["source_peak_index"] == 1
    assert entry["source_table_index"] == 0
    assert entry["source_row_index"] == 0
    assert entry["resolved_table_index"] == 0
    assert entry["resolved_peak_index"] == 1


def test_prepare_reflection_subset_keeps_duplicate_fixed_source_rows_out_of_hkl_fallback() -> None:
    miller = np.array(
        [
            [2.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([2.0, 7.0], dtype=np.float64)
    measured = [
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0-a",
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 9,
            "source_row_index": 0,
            "fit_source_identity_only": True,
        },
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0-b",
            "x": 5.0,
            "y": 5.0,
            "source_table_index": 9,
            "source_row_index": 1,
            "fit_source_identity_only": True,
        },
    ]

    subset = opt._prepare_reflection_subset(miller, intensities, measured)

    assert subset.fixed_source_reflection_count == 0
    assert subset.fallback_hkl_count == 1
    assert np.array_equal(subset.original_indices, np.array([0], dtype=np.int64))
    for entry in subset.measured_entries:
        for key in (
            "source_table_index",
            "source_reflection_index",
            "resolved_table_index",
            "source_row_index",
            "source_peak_index",
        ):
            assert key not in entry


def test_fit_geometry_parameters_pixel_path_keeps_residual_size_when_pair_status_changes(
    monkeypatch,
):
    captured_residuals = {}

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        gamma = float(args[8])
        sim_col = 4.0 if gamma < 0.5 else 20.0
        sim_row = 4.0 if gamma < 0.5 else 20.0
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, sim_col, sim_row, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x_match = np.asarray(x0, dtype=float).copy()
        x_match[0] = 0.0
        x_missing = np.asarray(x0, dtype=float).copy()
        x_missing[0] = 1.0

        residual_match = np.asarray(residual_fn(x_match), dtype=float)
        residual_missing = np.asarray(residual_fn(x_missing), dtype=float)
        captured_residuals["match"] = residual_match.copy()
        captured_residuals["missing"] = residual_missing.copy()

        assert residual_match.shape == residual_missing.shape == (2,)

        return opt.OptimizeResult(
            x=x_match,
            fun=residual_match,
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x_match, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 24
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"hkl": (1, 0, 0), "label": "1,0,0", "x": 4.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        pixel_tol=2.0,
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "weighted_matching": False,
                "missing_pair_penalty_px": 11.0,
            },
            "single_ray": {"enabled": False},
            "identifiability": {"enabled": False},
            "full_beam_polish": {"enabled": False},
        },
    )

    assert result.success
    assert np.allclose(captured_residuals["match"], [0.0, 0.0])
    assert np.allclose(captured_residuals["missing"], [11.0, 0.0])


def test_fit_geometry_parameters_dynamic_point_path_uses_angular_missing_penalty(
    monkeypatch,
):
    captured_residuals = {}
    phase = {"mode": "match"}

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        if phase["mode"] == "match":
            hit_tables = [
                np.array(
                    [[1.0, 12.0, 12.0, 0.0, 1.0, 0.0, 0.0]],
                    dtype=np.float64,
                )
            ]
        else:
            hit_tables = [np.empty((0, 7), dtype=np.float64)]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x_match = np.asarray(x0, dtype=float).copy()
        x_match[0] = 0.0
        x_missing = np.asarray(x0, dtype=float).copy()
        x_missing[0] = 1.0

        phase["mode"] = "match"
        residual_match = np.asarray(residual_fn(x_match), dtype=float)
        phase["mode"] = "missing"
        residual_missing = np.asarray(residual_fn(x_missing), dtype=float)
        captured_residuals["match"] = residual_match.copy()
        captured_residuals["missing"] = residual_missing.copy()
        phase["mode"] = "match"

        assert residual_match.shape == residual_missing.shape == (2,)

        return opt.OptimizeResult(
            x=x_match,
            fun=residual_match,
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x_match, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 24
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["debye_x"] = 1.0
    params["debye_y"] = 1.0
    measured = [
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0",
            "x": 12.0,
            "y": 12.0,
            "detector_x": 12.0,
            "detector_y": 12.0,
            "source_table_index": 0,
            "source_row_index": 0,
            "fit_source_identity_only": True,
        }
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        pixel_tol=2.0,
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "dynamic_point_geometry_fit": True,
                "weighted_matching": False,
                "use_measurement_uncertainty": False,
                "missing_pair_penalty_deg": 7.0,
            },
            "single_ray": {"enabled": False},
            "identifiability": {"enabled": False},
            "full_beam_polish": {"enabled": False},
        },
    )

    assert result.success
    assert result.final_metric_name == "dynamic_angular_point_match"
    assert np.allclose(captured_residuals["match"], [0.0, 0.0])
    assert np.allclose(captured_residuals["missing"], [7.0, 0.0])
    assert int(result.point_match_summary["missing_pair_count"]) == 0
    assert bool(result.geometry_fit_debug_summary["dynamic_point_geometry_fit"]) is True


def test_fit_geometry_parameters_manual_point_fit_with_cached_sources_defaults_to_central_point_match(
    monkeypatch,
):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [np.array([[1.0, 12.0, 12.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64)]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 24
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["debye_x"] = 1.0
    params["debye_y"] = 1.0
    measured = [
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0",
            "x": 12.0,
            "y": 12.0,
            "detector_x": 12.0,
            "detector_y": 12.0,
            "source_table_index": 0,
            "source_row_index": 0,
            "fit_source_identity_only": True,
        }
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        pixel_tol=2.0,
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "manual_point_fit_mode": True,
                "restarts": 0,
                "weighted_matching": False,
                "use_measurement_uncertainty": False,
            },
            "single_ray": {"enabled": False},
            "identifiability": {"enabled": False},
            "full_beam_polish": {"enabled": False},
        },
    )

    assert result.success
    assert result.final_metric_name == "central_point_match"
    assert bool(result.geometry_fit_debug_summary["dynamic_point_geometry_fit"]) is False


def test_fit_geometry_parameters_dynamic_point_path_records_fit_space_provenance(
    monkeypatch,
):
    captured: dict[str, np.ndarray] = {}

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, 12.0, 12.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            ),
            np.array(
                [[1.0, 14.0, 10.0, 0.0, 0.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        captured["residual"] = np.asarray(residual_fn(x), dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=captured["residual"],
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 24
    miller = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([1.0, 1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["pixel_size_m"] = 1.0e-4
    params["lambda"] = 1.1
    measured = [
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0",
            "x": 12.0,
            "y": 12.0,
            "detector_x": 12.0,
            "detector_y": 12.0,
            "background_two_theta_deg": 20.0,
            "background_phi_deg": 5.0,
            "background_reference_a": 4.0,
            "background_reference_c": 7.0,
            "background_reference_lambda": 1.0,
            "source_table_index": 0,
            "source_row_index": 0,
            "fit_source_identity_only": True,
        },
        {
            "hkl": (0, 1, 0),
            "label": "0,1,0",
            "x": 14.0,
            "y": 10.0,
            "detector_x": 14.0,
            "detector_y": 10.0,
            "source_table_index": 1,
            "source_row_index": 0,
            "fit_source_identity_only": True,
        },
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "dynamic_point_geometry_fit": True,
                "weighted_matching": False,
                "use_measurement_uncertainty": False,
                "missing_pair_penalty_deg": 7.0,
            },
            "single_ray": {"enabled": False},
            "identifiability": {"enabled": False},
            "full_beam_polish": {"enabled": False},
        },
    )

    assert result.success
    assert captured["residual"].shape == (4,)
    assert isinstance(result.point_match_summary, dict)
    assert result.point_match_summary["fit_space_pixel_size_source"] == "pixel_size_m"
    assert np.isclose(
        float(result.point_match_summary["fit_space_pixel_size_value"]),
        1.0e-4,
    )
    assert np.isclose(
        float(result.point_match_summary["fit_space_pixel_size_m_raw"]),
        1.0e-4,
    )
    assert np.isclose(
        float(result.point_match_summary["fit_space_debye_x_raw"]),
        0.0,
    )
    assert np.isclose(
        float(result.point_match_summary["fit_space_debye_y_raw"]),
        0.0,
    )
    assert int(result.point_match_summary["fit_space_anchor_count_cached"]) == 0
    assert int(result.point_match_summary["fit_space_anchor_count_detector"]) == 2
    assert result.point_match_summary["fit_space_anchor_source_counts"] == {
        "cached_fit_space_anchor": 0,
        "dataset_fit_space_projector": 0,
        "invalid_dataset_fit_space_projector": 0,
        "native_fit_space_anchor": 0,
        "background_detector_fit_space_anchor": 0,
        "detector_fit_space_anchor": 2,
        "display_fit_space_anchor": 0,
    }
    assert int(result.point_match_summary["fit_space_two_theta_adjustment_count"]) == 0
    assert float(result.point_match_summary["fit_space_two_theta_adjustment_total_abs_deg"]) == 0.0
    assert np.isnan(
        float(result.point_match_summary["fit_space_two_theta_adjustment_mean_abs_deg"])
    )
    assert np.isnan(float(result.point_match_summary["fit_space_two_theta_adjustment_max_abs_deg"]))
    assert len(result.point_match_summary["per_dataset"]) == 1
    assert int(result.point_match_summary["per_dataset"][0]["fit_space_anchor_count_cached"]) == 0
    assert int(result.point_match_summary["per_dataset"][0]["fit_space_anchor_count_detector"]) == 2


def test_simulate_and_compare_hkl_forwards_optics_mode(monkeypatch):
    optics_seen = []

    def fake_process(*args, **kwargs):
        optics_seen.append(kwargs.get("optics_mode"))
        return _fake_process_peaks(*args, **kwargs)

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    image_size = 8
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=2)
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]

    distances, *_ = opt.simulate_and_compare_hkl(
        miller,
        intensities,
        image_size,
        params,
        measured,
    )

    assert distances.size == 2
    assert optics_seen == [2]


def test_simulate_and_compare_hkl_can_force_python_runner(monkeypatch):
    prefer_python_runner_seen = []

    def fake_process(*args, **kwargs):
        prefer_python_runner_seen.append(kwargs.get("prefer_python_runner"))
        return _fake_process_peaks(*args, **kwargs)

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    image_size = 8
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]

    distances, *_ = opt.simulate_and_compare_hkl(
        miller,
        intensities,
        image_size,
        params,
        measured,
        prefer_python_runner=True,
    )

    assert distances.size == 2
    assert prefer_python_runner_seen == [True]


def test_simulate_and_compare_hkl_restricts_to_measured_hkl_subset(monkeypatch):
    process_millers = []

    def fake_process(*args, **kwargs):
        miller_arg = np.asarray(args[0], dtype=np.float64)
        process_millers.append(miller_arg.copy())
        image_size = int(args[2])
        hit_tables = []
        for row in miller_arg:
            hit_tables.append(
                np.array(
                    [[1.0, 4.0, 4.0, 0.0, row[0], row[1], row[2]]],
                    dtype=np.float64,
                )
            )
        image = np.zeros((image_size, image_size), dtype=np.float64)
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    image_size = 10
    miller = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([1.0, 2.0, 2.5, 3.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "2,0,0", "x": 4.0, "y": 4.0}]

    distances, *_ = opt.simulate_and_compare_hkl(
        miller,
        intensities,
        image_size,
        params,
        measured,
    )

    assert distances.size > 0
    assert process_millers
    assert process_millers[0].shape == (2, 3)
    assert np.allclose(
        process_millers[0],
        np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64),
    )


def test_simulate_and_compare_hkl_keeps_hkl_fallback_when_source_indices_are_stale(
    monkeypatch,
):
    process_millers = []

    def fake_process(*args, **kwargs):
        miller_arg = np.asarray(args[0], dtype=np.float64)
        process_millers.append(miller_arg.copy())
        image_size = int(args[2])
        hit_tables = []
        coord_map = {
            (1, 0, 0): (1.0, 1.0),
            (2, 0, 0): (4.0, 4.0),
            (3, 0, 0): (7.0, 7.0),
        }
        for row in miller_arg:
            hkl = tuple(int(round(v)) for v in row)
            col, row_px = coord_map[hkl]
            hit_tables.append(
                np.array(
                    [[1.0, col, row_px, 0.0, row[0], row[1], row[2]]],
                    dtype=np.float64,
                )
            )
        image = np.zeros((image_size, image_size), dtype=np.float64)
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    image_size = 12
    miller = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 0,
            "source_row_index": 1,
        }
    ]

    distances, sim_coords, meas_coords, sim_millers, meas_millers = opt.simulate_and_compare_hkl(
        miller,
        intensities,
        image_size,
        params,
        measured,
    )

    assert process_millers
    assert process_millers[0].shape == (1, 3)
    assert np.allclose(
        process_millers[0],
        np.array([[2.0, 0.0, 0.0]], dtype=np.float64),
    )
    assert distances.size == 2
    assert sim_coords == [(4.0, 4.0)]
    assert meas_coords == [(4.0, 4.0)]
    assert sim_millers == [(2, 0, 0)]
    assert meas_millers == [(2, 0, 0)]


def test_simulate_and_compare_hkl_falls_back_when_in_range_source_indices_point_to_wrong_hkl(
    monkeypatch,
):
    process_millers = []

    def fake_process(*args, **kwargs):
        miller_arg = np.asarray(args[0], dtype=np.float64)
        process_millers.append(miller_arg.copy())
        image_size = int(args[2])
        hit_tables = []
        coord_map = {
            (1, 0, 0): (1.0, 1.0),
            (2, 0, 0): (4.0, 4.0),
            (3, 0, 0): (7.0, 7.0),
        }
        for row in miller_arg:
            hkl = tuple(int(round(v)) for v in row)
            col, row_px = coord_map[hkl]
            hit_tables.append(
                np.array(
                    [[1.0, col, row_px, 0.0, row[0], row[1], row[2]]],
                    dtype=np.float64,
                )
            )
        image = np.zeros((image_size, image_size), dtype=np.float64)
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    image_size = 12
    miller = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 0,
            "source_row_index": 0,
        }
    ]

    distances, sim_coords, meas_coords, sim_millers, meas_millers = opt.simulate_and_compare_hkl(
        miller,
        intensities,
        image_size,
        params,
        measured,
    )

    assert process_millers
    assert process_millers[0].shape == (1, 3)
    assert np.allclose(
        process_millers[0],
        np.array([[2.0, 0.0, 0.0]], dtype=np.float64),
    )
    assert distances.size == 2
    assert sim_coords == [(4.0, 4.0)]
    assert meas_coords == [(4.0, 4.0)]
    assert sim_millers == [(2, 0, 0)]
    assert meas_millers == [(2, 0, 0)]


def test_simulate_and_compare_hkl_fixed_source_match_uses_detector_anchor(monkeypatch):
    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", _fake_process_peaks)

    image_size = 12
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0",
            "x": 99.0,
            "y": 98.0,
            "background_detector_x": 4.0,
            "background_detector_y": 4.0,
            "source_table_index": 0,
            "source_row_index": 0,
            "fit_source_resolution_kind": "source_row",
        }
    ]

    distances, sim_coords, meas_coords, sim_millers, meas_millers = opt.simulate_and_compare_hkl(
        miller,
        intensities,
        image_size,
        params,
        measured,
    )

    assert distances.size == 2
    assert np.allclose(distances, 0.0)
    assert sim_coords == [(4.0, 4.0)]
    assert meas_coords == [(4.0, 4.0)]
    assert sim_millers == [(1, 0, 0)]
    assert meas_millers == [(1, 0, 0)]


def test_resolve_fixed_source_matches_prefers_source_reflection_index() -> None:
    entry = {
        "hkl": (2, 0, 0),
        "label": "2,0,0",
        "x": 4.0,
        "y": 4.0,
        "source_table_index": 0,
        "source_reflection_index": 1,
        "resolved_table_index": 1,
        "source_row_index": 0,
    }
    hit_tables = [
        np.asarray([[1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64),
        np.asarray([[1.0, 4.0, 4.0, 0.0, 2.0, 0.0, 0.0]], dtype=np.float64),
    ]

    resolved, fallback_entries, resolution_lookup = opt._resolve_fixed_source_matches(
        [entry],
        hit_tables,
    )

    assert len(resolved) == 1
    assert fallback_entries == []
    assert resolved[0][1] == (4.0, 4.0)
    assert resolved[0][2] == (2, 0, 0)
    assert resolution_lookup[id(entry)]["resolution_kind"] == "fixed_source"
    assert resolution_lookup[id(entry)]["resolution_reason"] == "resolved"


def test_resolve_fixed_source_matches_keeps_distinct_branches(monkeypatch) -> None:
    monkeypatch.setattr(
        opt,
        "hit_tables_to_max_positions",
        lambda hit_tables: np.asarray([[1.0, 2.0, 2.0, 1.0, 8.0, 8.0]], dtype=np.float64),
    )
    entries = [
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0-left",
            "x": 2.0,
            "y": 2.0,
            "source_reflection_index": 0,
            "resolved_table_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_row_index": 0,
        },
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0-right",
            "x": 8.0,
            "y": 8.0,
            "source_reflection_index": 0,
            "resolved_table_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "source_row_index": 0,
        },
    ]
    hit_tables = [
        np.asarray(
            [
                [1.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 8.0, 8.0, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    ]

    resolved, fallback_entries, _resolution_lookup = opt._resolve_fixed_source_matches(
        entries,
        hit_tables,
    )

    assert fallback_entries == []
    assert [item[1] for item in resolved] == [(2.0, 2.0), (8.0, 8.0)]


def _provider_local_singleton_entry(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "hkl": (2, 0, 0),
        "label": "2,0,0",
        "x": 4.0,
        "y": 4.0,
        "source_table_index": 99,
        "original_source_table_index": 99,
        "resolved_table_index": 0,
        "source_row_index": 24,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "resolved_peak_index": 1,
        "fit_source_resolution_kind": "provider_fixed_source_local",
        "optimizer_request_source": "provider_pair",
        "optimizer_request_has_fixed_source": True,
        "optimizer_request_fallback_row": False,
        "provider_local_subset_provenance": True,
        "provider_local_subset_assignment": "provider_local_duplicate_hkl_branch",
        "provider_local_subset_branch_provenance": True,
        "provider_local_subset_duplicate_hkl_count": 2,
        "provider_selected_source_identity_canonical": {
            "normalized_hkl": [2, 0, 0],
            "source_table_index": 99,
            "source_peak_index": 1,
        },
    }
    entry.update(overrides)
    return entry


def _locked_qr_fixed_source_entry(**overrides: object) -> dict[str, object]:
    entry = _provider_local_singleton_entry(
        q_group_key=("q_group", "primary", 1, 10),
        branch_group_key=("branch_group", "primary", 1),
        source_table_index=99,
        resolved_table_index=0,
        source_row_index=24,
        source_branch_index=1,
        source_peak_index=1,
        source_reflection_index=910,
        source_reflection_namespace="subset",
        source_reflection_is_full=False,
        branch_id="locked-branch",
        best_sample_index=3,
        mosaic_top_rank_key=("rank", 3),
        selection_reason="manual_pick",
    )
    entry.update(overrides)
    return entry


def _locked_qr_trial_source_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "hkl": (2, 0, 0),
        "q_group_key": ("q_group", "primary", 1, 10),
        "branch_group_key": ("branch_group", "primary", 1),
        "source_table_index": 0,
        "source_row_index": 5,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "source_reflection_index": 42,
        "source_reflection_namespace": "subset",
        "source_reflection_is_full": False,
        "branch_id": "trial-branch",
        "best_sample_index": 7,
        "mosaic_top_rank_key": ("rank", 7),
        "selection_reason": "trial_rebuild",
        "native_col": 12.0,
        "native_row": 13.0,
    }
    row.update(overrides)
    return row


def _locked_qr_source_rows_payload(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "available": bool(rows),
        "rows": [dict(row) for row in rows],
        "row_count": len(rows),
        "source": "unit_test",
        "source_rows_signature": "unit-test",
    }


def _point_only_qr_dataset_ctx(source_rows_builder, projector, image_builder=None):
    subset = opt.ReflectionSimulationSubset(
        miller=np.array([[2.0, 0.0, 0.0]], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        measured_entries=[],
        original_indices=np.array([0], dtype=np.int64),
        total_reflection_count=1,
        fixed_source_reflection_count=1,
        fallback_hkl_count=0,
        reduced=False,
    )
    return opt.GeometryFitDatasetContext(
        dataset_index=0,
        label="bg0",
        theta_initial=0.0,
        subset=subset,
        fit_space_projector=projector,
        fit_space_projector_kind="exact_caked_bundle",
        sim_caked_image_builder=image_builder,
        sim_caked_image_builder_kind="must_not_call",
        qr_fit_trial_source_rows_builder=source_rows_builder,
        qr_fit_trial_source_rows_builder_kind="unit_test_dynamic_rows",
    )


def _point_only_dynamic_qr_row(two_theta: float, phi: float) -> dict[str, object]:
    row = _locked_qr_trial_source_row(
        native_col=12.0,
        native_row=13.0,
        source_table_index=99,
        source_row_index=24,
        source_reflection_index=910,
        branch_id="locked-branch",
        best_sample_index=3,
        mosaic_top_rank_key=("rank", 3),
        selection_reason="manual_pick",
        caked_x=float(two_theta),
        caked_y=float(phi),
        two_theta_deg=float(two_theta),
        phi_deg=float(phi),
        sim_visual_caked_deg=[float(two_theta), float(phi)],
        actual_source="sim_visual_caked_deg",
        source_kind="sim_visual_caked_deg",
        projection_frame="caked_display",
        coordinate_provenance="trial_geometry_projection",
        is_dynamic_trial_row=True,
        physical_branch_slot=1,
    )
    row["fit_qr_branch_key"] = {
        "q_group_key": list(row["q_group_key"]),
        "hkl": list(row["hkl"]),
        "physical_branch_slot": 1,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "source_table_index": int(row["source_table_index"]),
        "source_row_index": int(row["source_row_index"]),
        "source_reflection_index": int(row["source_reflection_index"]),
        "source_reflection_namespace": row["source_reflection_namespace"],
        "source_reflection_is_full": bool(row["source_reflection_is_full"]),
        "branch_id": row["branch_id"],
        "best_sample_index": int(row["best_sample_index"]),
        "mosaic_top_rank_key": list(row["mosaic_top_rank_key"]),
        "selection_reason": row["selection_reason"],
    }
    return row


def _point_only_source_from_params(
    local_params: Mapping[str, object] | None,
) -> tuple[float, float]:
    params = local_params if isinstance(local_params, Mapping) else {}
    center_delta = float(params.get("center_x", 0.0) or 0.0)
    theta_delta = float(params.get("theta_initial", 0.0) or 0.0)
    return 42.0 + 0.25 * center_delta + 2.0 * theta_delta, -179.7


def test_qr_fit_point_only_projection_uses_dynamic_sim_visual_caked_deg_and_skips_image_refinement(
    monkeypatch,
) -> None:
    image_builder_calls = 0
    projector_calls = 0
    refinement_calls = 0

    def source_rows_builder(*, local_params=None):
        two_theta, phi = _point_only_source_from_params(local_params)
        return _locked_qr_source_rows_payload([_point_only_dynamic_qr_row(two_theta, phi)])

    def projector(cols, rows, *, local_params=None, **_kwargs):
        nonlocal projector_calls
        projector_calls += 1
        assert np.asarray(cols, dtype=float).tolist() == [12.0]
        assert np.asarray(rows, dtype=float).tolist() == [13.0]
        two_theta, phi = _point_only_source_from_params(local_params)
        return {
            "two_theta_deg": np.asarray([two_theta], dtype=np.float64),
            "phi_deg": np.asarray([phi], dtype=np.float64),
            "native_cols": np.asarray([12.0], dtype=np.float64),
            "native_rows": np.asarray([13.0], dtype=np.float64),
            "fit_space_source": "dataset_fit_space_projector",
            "fit_space_projector_kind": "exact_caked_bundle",
            "fit_space_local_params_signature": "unit-test",
            "cake_bundle_signature": "unit-test",
            "input_frame": "native_detector",
            "invalid_reason": None,
            "native_frame_conversion_source": "",
            "native_frame_conversion_count": 0,
            "valid": True,
        }

    def image_builder(*_args, **_kwargs):
        nonlocal image_builder_calls
        image_builder_calls += 1
        raise AssertionError("point-only projection must not build caked image")

    def refine_image(*_args, **_kwargs):
        nonlocal refinement_calls
        refinement_calls += 1
        raise AssertionError("point-only projection must not refine caked image")

    monkeypatch.setattr(opt, "_refine_sim_caked_peak_from_image", refine_image)

    dataset_ctx = _point_only_qr_dataset_ctx(source_rows_builder, projector, image_builder)
    locked = _locked_qr_fixed_source_entry()

    prediction = opt._resolve_qr_fit_prediction_from_trial_params(
        locked,
        {"center_x": 4.0, "theta_initial": 0.5},
        {
            "dataset_ctx": dataset_ctx,
            "hit_tables": [np.asarray([[1.0, 99.0, 99.0, 0.0, 2.0, 0.0, 0.0]], dtype=np.float64)],
            "sim_buffer": np.zeros((30, 30), dtype=np.float64),
            "image_size": 30,
            "fit_center": [15.0, 15.0],
            "detector_distance": 0.1,
            "pixel_size": 1.0,
            "gamma_deg": 0.0,
            "Gamma_deg": 0.0,
            "prediction_source_rows_cache": {},
            "_qr_fit_point_only_projection": True,
        },
        locked,
    )

    expected = _point_only_source_from_params({"center_x": 4.0, "theta_initial": 0.5})
    assert image_builder_calls == 0
    assert refinement_calls == 0
    assert projector_calls == 0
    assert prediction["available"] is True
    assert prediction["_qr_fit_point_only_projection"] is True
    assert (
        prediction["hit_table_resolution_reason"] == "locked_hit_table_resolver_skipped_point_only"
    )
    assert prediction["sim_refinement_status"] == "point_only_dynamic_sim_visual_caked_deg"
    assert prediction["sim_refinement_caked_image_source"] == "point_only_dynamic_source_row"
    assert prediction["point_only_projector_skipped"] is True
    assert prediction["sim_nominal_caked_deg"] == pytest.approx(list(expected))
    assert prediction["sim_refined_caked_deg"] == pytest.approx(list(expected))
    assert prediction["dynamic_baseline_anchor_actual_source"] == "sim_visual_caked_deg"
    assert prediction["point_only_projected_minus_dynamic_delta_two_theta_deg"] == pytest.approx(
        0.0
    )
    assert prediction["point_only_projected_minus_dynamic_delta_phi_deg_wrapped"] == pytest.approx(
        0.0
    )
    assert prediction["point_only_projected_dynamic_match"] is True


def test_qr_fit_point_only_projection_rejects_stale_dynamic_source() -> None:
    def source_rows_builder(*, local_params=None):
        del local_params
        row = _point_only_dynamic_qr_row(42.0, -179.7)
        row["actual_source"] = "clicked_visual_candidate"
        row["source_kind"] = "clicked_visual_candidate"
        row["coordinate_provenance"] = "saved_manual_coordinate_materialization"
        row["is_dynamic_trial_row"] = False
        return _locked_qr_source_rows_payload([row])

    def projector(*_args, **_kwargs):
        return {
            "two_theta_deg": np.asarray([42.0], dtype=np.float64),
            "phi_deg": np.asarray([-179.7], dtype=np.float64),
            "native_cols": np.asarray([12.0], dtype=np.float64),
            "native_rows": np.asarray([13.0], dtype=np.float64),
            "fit_space_source": "dataset_fit_space_projector",
            "fit_space_projector_kind": "exact_caked_bundle",
            "fit_space_local_params_signature": "unit-test",
            "cake_bundle_signature": "unit-test",
            "input_frame": "native_detector",
            "invalid_reason": None,
            "native_frame_conversion_source": "",
            "native_frame_conversion_count": 0,
            "valid": True,
        }

    prediction = opt._resolve_qr_fit_prediction_from_trial_params(
        _locked_qr_fixed_source_entry(),
        {"center_x": 0.0},
        {
            "dataset_ctx": _point_only_qr_dataset_ctx(source_rows_builder, projector),
            "hit_tables": (),
            "sim_buffer": np.zeros((30, 30), dtype=np.float64),
            "image_size": 30,
            "fit_center": [15.0, 15.0],
            "detector_distance": 0.1,
            "pixel_size": 1.0,
            "prediction_source_rows_cache": {},
            "_qr_fit_point_only_projection": True,
        },
        _locked_qr_fixed_source_entry(),
    )

    assert prediction["available"] is False
    assert prediction["unavailable_reason"] == "point_only_dynamic_sim_visual_caked_deg_unavailable"


def test_qr_fit_point_only_projection_skips_projector_when_dynamic_source_is_valid() -> None:
    projector_calls = 0

    def source_rows_builder(*, local_params=None):
        del local_params
        return _locked_qr_source_rows_payload([_point_only_dynamic_qr_row(42.0, -179.7)])

    def projector(*_args, **_kwargs):
        nonlocal projector_calls
        projector_calls += 1
        return {
            "two_theta_deg": np.asarray([45.0], dtype=np.float64),
            "phi_deg": np.asarray([-170.0], dtype=np.float64),
            "native_cols": np.asarray([12.0], dtype=np.float64),
            "native_rows": np.asarray([13.0], dtype=np.float64),
            "fit_space_source": "dataset_fit_space_projector",
            "fit_space_projector_kind": "exact_caked_bundle",
            "fit_space_local_params_signature": "unit-test",
            "cake_bundle_signature": "unit-test",
            "input_frame": "native_detector",
            "invalid_reason": None,
            "native_frame_conversion_source": "",
            "native_frame_conversion_count": 0,
            "valid": True,
        }

    prediction = opt._resolve_qr_fit_prediction_from_trial_params(
        _locked_qr_fixed_source_entry(),
        {"center_x": 0.0},
        {
            "dataset_ctx": _point_only_qr_dataset_ctx(source_rows_builder, projector),
            "hit_tables": (),
            "sim_buffer": np.zeros((30, 30), dtype=np.float64),
            "image_size": 30,
            "fit_center": [15.0, 15.0],
            "detector_distance": 0.1,
            "pixel_size": 1.0,
            "prediction_source_rows_cache": {},
            "_qr_fit_point_only_projection": True,
        },
        _locked_qr_fixed_source_entry(),
    )

    assert prediction["available"] is True
    assert projector_calls == 0
    assert prediction["sim_refinement_status"] == "point_only_dynamic_sim_visual_caked_deg"
    assert prediction["sim_nominal_projected_caked_deg"] == pytest.approx([42.0, -179.7])
    assert prediction["sim_nominal_caked_deg"] == pytest.approx([42.0, -179.7])
    assert prediction["sim_refined_caked_deg"] == pytest.approx([42.0, -179.7])
    assert prediction["point_only_projector_skipped"] is True
    assert prediction["point_only_projected_dynamic_match"] is True
    assert prediction["point_only_projected_minus_dynamic_delta_two_theta_deg"] == pytest.approx(
        0.0
    )
    assert prediction["point_only_projected_minus_dynamic_delta_phi_deg_wrapped"] == pytest.approx(
        0.0
    )


def test_fit_geometry_parameters_point_only_projection_skips_locked_refinement_preflight(
    monkeypatch,
) -> None:
    projector_calls = 0
    process_calls = 0

    def fake_process(*args, **_kwargs):
        nonlocal process_calls
        process_calls += 1
        image_size = int(args[2])
        return (
            np.zeros((image_size, image_size), dtype=np.float64),
            [],
            np.empty((0, 0, 0)),
            np.empty(0),
            np.empty(0),
            [],
        )

    def fake_least_squares(residual_fn, x0, **_kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    def source_rows_builder(*, local_params=None):
        two_theta, phi = _point_only_source_from_params(local_params)
        return _locked_qr_source_rows_payload([_point_only_dynamic_qr_row(two_theta, phi)])

    def projector(_cols, _rows, *, local_params=None, **_kwargs):
        nonlocal projector_calls
        projector_calls += 1
        two_theta, phi = _point_only_source_from_params(local_params)
        return {
            "two_theta_deg": np.asarray([two_theta], dtype=np.float64),
            "phi_deg": np.asarray([phi], dtype=np.float64),
            "native_cols": np.asarray([12.0], dtype=np.float64),
            "native_rows": np.asarray([13.0], dtype=np.float64),
            "fit_space_source": "dataset_fit_space_projector",
            "fit_space_projector_kind": "exact_caked_bundle",
            "fit_space_local_params_signature": "unit-test",
            "cake_bundle_signature": "unit-test",
            "input_frame": "native_detector",
            "invalid_reason": None,
            "native_frame_conversion_source": "",
            "native_frame_conversion_count": 0,
            "valid": True,
        }

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    measured = _locked_qr_fixed_source_entry(
        native_col=8.0,
        native_row=9.0,
        background_two_theta_deg=40.0,
        background_phi_deg=179.8,
        fit_space_anchor_override=True,
    )
    image_size = 30
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)
    dataset_specs = [
        {
            "dataset_index": 0,
            "label": "bg0",
            "theta_initial": 0.0,
            "measured_peaks": [measured],
            "experimental_image": experimental_image,
            "fit_space_projector": projector,
            "fit_space_projector_kind": "exact_caked_bundle",
            "qr_fit_trial_source_rows_builder": source_rows_builder,
            "qr_fit_trial_source_rows_builder_kind": "unit_test_dynamic_rows",
            "sim_caked_image_builder": lambda *_args, **_kwargs: pytest.fail(
                "point-only mode must not generate a caked image"
            ),
            "sim_caked_image_builder_kind": "must_not_call",
        }
    ]

    result = opt.fit_geometry_parameters(
        np.array([[2.0, 0.0, 0.0]], dtype=np.float64),
        np.array([1.0], dtype=np.float64),
        image_size,
        _base_params(image_size, optics_mode=1),
        measured_peaks=[measured],
        var_names=["theta_initial"],
        experimental_image=experimental_image,
        dataset_specs=dataset_specs,
        refinement_config={
            "solver": {
                "manual_point_fit_mode": True,
                "dynamic_point_geometry_fit": True,
                "_qr_fit_point_only_projection": True,
                "fixed_manual_prediction_preflight_guard": True,
                "restarts": 0,
                "weighted_matching": False,
                "use_measurement_uncertainty": False,
                "max_nfev": 1,
            },
            "single_ray": {"enabled": False},
            "identifiability": {"enabled": False},
            "full_beam_polish": {"enabled": False},
            "image_refinement": {"enabled": False},
        },
    )

    assert result.success
    assert process_calls == 0
    assert projector_calls == 0
    assert result.geometry_fit_debug_summary["locked_mosaic_refinement_validation"] == (
        "skipped_point_only"
    )
    assert result.point_match_diagnostics[0]["sim_refinement_status"] == (
        "point_only_dynamic_sim_visual_caked_deg"
    )


def test_rung3_objective_uses_cached_caked_targets_and_dynamic_sim_sources(
    monkeypatch,
) -> None:
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        return (
            np.zeros((image_size, image_size), dtype=np.float64),
            [],
            np.empty((0, 0, 0)),
            np.empty(0),
            np.empty(0),
            [],
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    def source_rows_builder(*, local_params=None):
        two_theta, phi = _point_only_source_from_params(local_params)
        return _locked_qr_source_rows_payload([_point_only_dynamic_qr_row(two_theta, phi)])

    def projector(_cols, _rows, *, local_params=None, **_kwargs):
        two_theta, phi = _point_only_source_from_params(local_params)
        return {
            "two_theta_deg": np.asarray([two_theta], dtype=np.float64),
            "phi_deg": np.asarray([phi], dtype=np.float64),
            "native_cols": np.asarray([12.0], dtype=np.float64),
            "native_rows": np.asarray([13.0], dtype=np.float64),
            "fit_space_source": "dataset_fit_space_projector",
            "fit_space_projector_kind": "exact_caked_bundle",
            "fit_space_local_params_signature": "unit-test",
            "cake_bundle_signature": "unit-test",
            "input_frame": "native_detector",
            "invalid_reason": None,
            "native_frame_conversion_source": "",
            "native_frame_conversion_count": 0,
            "valid": True,
        }

    measured = _locked_qr_fixed_source_entry(
        native_col=8.0,
        native_row=9.0,
        background_two_theta_deg=40.0,
        background_phi_deg=179.8,
        caked_x=40.0,
        caked_y=179.8,
        fit_space_anchor_override=True,
        fit_space_anchor_source="manual_caked_click",
    )
    dataset_ctx = _point_only_qr_dataset_ctx(source_rows_builder, projector)
    dataset_ctx.subset.measured_entries = [measured]
    local = _base_params(30, optics_mode=1)
    local.update(
        {
            "center_x": 4.0,
            "center_y": 15.0,
            "center": [4.0, 15.0],
            "theta_initial": 0.5,
            "_qr_fit_point_only_projection": True,
        }
    )

    residual, diagnostics, summary = opt._evaluate_geometry_fit_dataset_dynamic_point_matches(
        local,
        dataset_ctx,
        image_size=30,
        missing_pair_penalty_deg=5.0,
        theta_value=0.5,
        collect_diagnostics=True,
    )

    expected_source = _point_only_source_from_params(local)
    expected_delta_theta = expected_source[0] - 40.0
    expected_delta_phi = opt._wrap_phi_deg(expected_source[1] - 179.8)
    assert residual.tolist() == pytest.approx([expected_delta_theta, expected_delta_phi])
    assert len(diagnostics) == 1
    diag = diagnostics[0]
    assert diag["measured_anchor_source"] == "cached_fit_space_anchor"
    assert diag["measured_fit_space_source"] == "cached_fit_space_anchor"
    assert diag["sim_refined_caked_deg"] == pytest.approx(list(expected_source))
    assert diag["simulated_two_theta_deg"] == pytest.approx(expected_source[0])
    assert diag["simulated_phi_deg"] == pytest.approx(expected_source[1])
    assert diag["manual_qr_fit_target_caked_deg"] == pytest.approx([40.0, 179.8])
    assert diag["manual_qr_fit_source_caked_deg"] == pytest.approx(list(expected_source))
    assert diag["manual_qr_fit_residual_caked_deg"] == pytest.approx(
        [expected_delta_theta, expected_delta_phi]
    )
    assert diag["delta_two_theta_deg"] == pytest.approx(expected_delta_theta)
    assert diag["wrapped_delta_phi_deg"] == pytest.approx(expected_delta_phi)
    assert diag["solver_residual_vector"] == pytest.approx(
        [expected_delta_theta, expected_delta_phi]
    )
    assert diag["metric_unit"] == "deg"
    assert diag["sim_refinement_status"] == "point_only_dynamic_sim_visual_caked_deg"
    assert summary["manual_caked_residual_row_count"] == 1
    assert summary["cached_fit_space_anchor_row_count"] == 1


def test_rung3_objective_changes_dynamic_sim_source_when_trial_params_change(
    monkeypatch,
) -> None:
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        return (
            np.zeros((image_size, image_size), dtype=np.float64),
            [],
            np.empty((0, 0, 0)),
            np.empty(0),
            np.empty(0),
            [],
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    def source_rows_builder(*, local_params=None):
        two_theta, phi = _point_only_source_from_params(local_params)
        return _locked_qr_source_rows_payload([_point_only_dynamic_qr_row(two_theta, phi)])

    def projector(_cols, _rows, *, local_params=None, **_kwargs):
        two_theta, phi = _point_only_source_from_params(local_params)
        return {
            "two_theta_deg": np.asarray([two_theta], dtype=np.float64),
            "phi_deg": np.asarray([phi], dtype=np.float64),
            "native_cols": np.asarray([12.0], dtype=np.float64),
            "native_rows": np.asarray([13.0], dtype=np.float64),
            "fit_space_source": "dataset_fit_space_projector",
            "fit_space_projector_kind": "exact_caked_bundle",
            "fit_space_local_params_signature": "unit-test",
            "cake_bundle_signature": "unit-test",
            "input_frame": "native_detector",
            "invalid_reason": None,
            "native_frame_conversion_source": "",
            "native_frame_conversion_count": 0,
            "valid": True,
        }

    measured = _locked_qr_fixed_source_entry(
        native_col=8.0,
        native_row=9.0,
        background_two_theta_deg=40.0,
        background_phi_deg=179.8,
        fit_space_anchor_override=True,
        fit_space_anchor_source="manual_caked_click",
    )
    dataset_ctx = _point_only_qr_dataset_ctx(source_rows_builder, projector)
    dataset_ctx.subset.measured_entries = [measured]

    def evaluate(center_x: float, theta_initial: float):
        local = _base_params(30, optics_mode=1)
        local.update(
            {
                "center_x": float(center_x),
                "center_y": 15.0,
                "center": [float(center_x), 15.0],
                "theta_initial": float(theta_initial),
                "_qr_fit_point_only_projection": True,
            }
        )
        _residual, diagnostics, _summary = opt._evaluate_geometry_fit_dataset_dynamic_point_matches(
            local,
            dataset_ctx,
            image_size=30,
            missing_pair_penalty_deg=5.0,
            theta_value=float(theta_initial),
            collect_diagnostics=True,
        )
        return diagnostics[0]

    base = evaluate(0.0, 0.0)
    center_changed = evaluate(4.0, 0.0)
    theta_changed = evaluate(0.0, 0.5)

    assert base["measured_two_theta_deg"] == pytest.approx(40.0)
    assert center_changed["measured_two_theta_deg"] == pytest.approx(40.0)
    assert theta_changed["measured_two_theta_deg"] == pytest.approx(40.0)
    assert center_changed["simulated_two_theta_deg"] != pytest.approx(
        base["simulated_two_theta_deg"]
    )
    assert theta_changed["simulated_two_theta_deg"] != pytest.approx(
        base["simulated_two_theta_deg"]
    )


def test_locked_qr_fixed_source_resolves_stale_row_by_q_group_hkl_branch_slot() -> None:
    locked = _locked_qr_fixed_source_entry()
    source_row = _locked_qr_trial_source_row(
        source_table_index=3,
        source_row_index=77,
        source_reflection_index=321,
        branch_id="trial-different-identity",
    )

    point, payload = opt._resolve_locked_qr_trial_detector_point_from_source_rows(
        locked,
        source_rows_payload=_locked_qr_source_rows_payload([source_row]),
    )

    assert point == (12.0, 13.0)
    assert payload["resolution_reason"] == "locked_fit_qr_q_group_hkl_branch_slot_resolved"
    assert payload["trial_source_rows_fit_qr_branch_key_candidate_count"] == 0
    assert payload["trial_source_rows_q_group_hkl_slot_candidate_count"] == 1
    assert payload["resolved_source_row_position"] == 0


def test_locked_qr_fixed_source_uses_single_provider_backed_branch_slot_proof() -> None:
    locked = _locked_qr_fixed_source_entry()
    rows = [
        _locked_qr_trial_source_row(source_table_index=1, source_row_index=10),
        _locked_qr_trial_source_row(source_table_index=2, source_row_index=11),
        _locked_qr_trial_source_row(
            source_table_index=99,
            source_row_index=24,
            row_origin="manual_picker_saved_source_coverage",
            provider_backed_live_source_row=True,
            native_col=21.0,
            native_row=22.0,
        ),
    ]

    point, payload = opt._resolve_locked_qr_trial_detector_point_from_source_rows(
        locked,
        source_rows_payload=_locked_qr_source_rows_payload(rows),
    )

    assert point == (21.0, 22.0)
    assert payload["resolution_reason"] == "locked_fit_qr_q_group_hkl_branch_slot_resolved"
    assert payload["trial_source_rows_q_group_hkl_slot_candidate_count"] == 3
    assert payload["trial_source_rows_q_group_hkl_slot_proven_candidate_count"] == 1
    assert payload["locked_representative_intensity_defaulted"] is True


def test_locked_qr_fixed_source_resolves_zero_qr_00l_with_collapsed_branch() -> None:
    locked = _locked_qr_fixed_source_entry(
        hkl=(0, 0, 3),
        q_group_key=("q_group", "primary", 0, 3),
        branch_group_key=("branch_group", "primary", 0),
        source_branch_index=0,
        source_peak_index=0,
    )
    source_row = _locked_qr_trial_source_row(
        hkl=(0, 0, 3),
        q_group_key=("q_group", "primary", 0, 3),
        branch_group_key=("branch_group", "primary", 1),
        source_branch_index=1,
        source_peak_index=1,
        source_table_index=4,
        source_row_index=10,
    )

    point, payload = opt._resolve_locked_qr_trial_detector_point_from_source_rows(
        locked,
        source_rows_payload=_locked_qr_source_rows_payload([source_row]),
    )

    assert point == (12.0, 13.0)
    assert payload["resolution_reason"] == "locked_fit_qr_q_group_hkl_00l_collapsed_resolved"
    assert payload["locked_qr_zero_qr_00l_branch_unconstrained"] is True
    assert payload["trial_source_rows_q_group_hkl_slot_candidate_count"] == 1


def test_locked_qr_fixed_source_rejects_non00l_without_branch_slot_proof() -> None:
    locked = _locked_qr_fixed_source_entry(source_branch_index=1, source_peak_index=1)
    source_row = _locked_qr_trial_source_row()
    source_row.pop("source_branch_index")
    source_row.pop("source_peak_index")

    point, payload = opt._resolve_locked_qr_trial_detector_point_from_source_rows(
        locked,
        source_rows_payload=_locked_qr_source_rows_payload([source_row]),
    )

    assert point is None
    assert payload["resolution_reason"] == "locked_fit_qr_q_group_hkl_branch_slot_missing"
    assert payload["trial_source_rows_q_group_hkl_slot_candidate_count"] == 0
    assert payload["trial_source_rows_q_group_hkl_slot_missing_branch_count"] == 1


def test_locked_qr_fixed_source_rejects_non00l_ambiguous_branch_slot_proof() -> None:
    locked = _locked_qr_fixed_source_entry(source_branch_index=1, source_peak_index=1)
    rows = [
        _locked_qr_trial_source_row(source_table_index=1, source_row_index=10),
        _locked_qr_trial_source_row(source_table_index=2, source_row_index=11),
    ]

    point, payload = opt._resolve_locked_qr_trial_detector_point_from_source_rows(
        locked,
        source_rows_payload=_locked_qr_source_rows_payload(rows),
    )

    assert point is None
    assert payload["resolution_reason"] == "locked_fit_qr_q_group_hkl_branch_slot_ambiguous"
    assert payload["trial_source_rows_q_group_hkl_slot_candidate_count"] == 2
    assert payload["ambiguous_candidate_count"] == 2


def test_resolve_fixed_source_matches_rescues_provider_local_singleton_row() -> None:
    entry = _provider_local_singleton_entry(q_group_key=("q", 2))
    hit_tables = [np.asarray([[1.0, 4.0, 4.0, -10.0, 2.0, 0.0, 0.0]], dtype=np.float64)]

    resolved, fallback_entries, resolution_lookup = opt._resolve_fixed_source_matches(
        [entry],
        hit_tables,
    )

    assert len(resolved) == 1
    assert fallback_entries == []
    diag = resolution_lookup[id(entry)]
    assert diag["resolution_kind"] == "fixed_source"
    assert diag["resolution_reason"] == ("provider_local_stale_row_index_single_row_table_resolved")
    assert diag["stale_source_row_index"] == 24
    assert diag["source_row_count"] == 1
    assert diag["resolved_sim_hkl"] == (2, 0, 0)
    assert diag["resolved_peak_index"] == 1
    assert diag["resolved_source_row_position"] == 0
    assert diag["provider_local_stale_row_index_single_row_table_resolved"] is True
    assert diag["prediction_source_status"] == "available"
    assert diag["provider_local_subset_provenance"] is True
    assert diag["provider_local_subset_branch_provenance"] is True


def test_resolve_fixed_source_matches_provider_local_stale_row_single_row_table() -> None:
    entry = _provider_local_singleton_entry(
        q_group_key=("q", 2),
        source_row_index=24,
        source_branch_index=1,
        source_peak_index=1,
        resolved_peak_index=1,
    )
    hit_tables = [np.asarray([[1.0, 4.0, 4.0, 10.0, 2.0, 0.0, 0.0]], dtype=np.float64)]

    resolved, fallback_entries, resolution_lookup = opt._resolve_fixed_source_matches(
        [entry],
        hit_tables,
    )

    assert len(resolved) == 1
    assert fallback_entries == []
    diag = resolution_lookup[id(entry)]
    assert diag["resolution_kind"] == "fixed_source"
    assert diag["resolution_reason"] == ("provider_local_stale_row_index_single_row_table_resolved")
    assert diag["provider_local_stale_row_index_single_row_table_resolved"] is True
    assert diag["prediction_source_status"] == "available"
    assert diag["stale_source_row_index"] == 24
    assert diag["source_row_count"] == 1


def test_resolve_fixed_source_matches_provider_local_stale_row_unique_branch() -> None:
    entry = _provider_local_singleton_entry(
        source_row_index=24,
        source_branch_index=1,
        source_peak_index=1,
        resolved_peak_index=1,
    )
    hit_tables = [
        np.asarray(
            [
                [1.0, 3.0, 3.0, -10.0, 2.0, 0.0, 0.0],
                [1.0, 4.0, 4.0, 10.0, 2.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    ]

    resolved, fallback_entries, resolution_lookup = opt._resolve_fixed_source_matches(
        [entry],
        hit_tables,
    )

    assert len(resolved) == 1
    assert resolved[0][1] == (4.0, 4.0)
    assert fallback_entries == []
    diag = resolution_lookup[id(entry)]
    assert diag["resolution_kind"] == "fixed_source"
    assert diag["resolution_reason"] == "provider_local_stale_row_index_branch_resolved"
    assert diag["provider_local_stale_row_index_branch_resolved"] is True
    assert diag["provider_local_branch_match_row_count"] == 1
    assert diag["prediction_source_status"] == "available"


def test_resolve_fixed_source_matches_saved_detector_is_diagnostic_after_stale_row_proof() -> None:
    entry = _provider_local_singleton_entry(
        source_row_index=24,
        source_branch_index=1,
        source_peak_index=1,
        resolved_peak_index=1,
        sim_visual_detector_native_px=(9.0, 8.0),
    )
    hit_tables = [
        np.asarray(
            [
                [1.0, 3.0, 3.0, -10.0, 2.0, 0.0, 0.0],
                [1.0, 4.0, 4.0, 10.0, 2.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    ]

    resolved, fallback_entries, resolution_lookup = opt._resolve_fixed_source_matches(
        [entry],
        hit_tables,
    )

    assert len(resolved) == 1
    assert resolved[0][1] == (4.0, 4.0)
    assert fallback_entries == []
    diag = resolution_lookup[id(entry)]
    assert diag["resolution_reason"] == "provider_local_stale_row_index_branch_resolved"
    assert diag["provider_local_stale_row_index_branch_resolved"] is True
    assert diag["provider_local_saved_sim_detector_diagnostic_only"] is True
    assert diag["prediction_source_status"] == "available"


def test_resolve_fixed_source_matches_provider_local_duplicate_hkl_ambiguous_rejects() -> None:
    entry = _provider_local_singleton_entry(
        source_row_index=24,
        source_branch_index=1,
        source_peak_index=1,
        resolved_peak_index=1,
    )
    hit_tables = [
        np.asarray(
            [
                [1.0, 4.0, 4.0, 10.0, 2.0, 0.0, 0.0],
                [1.0, 5.0, 5.0, 12.0, 2.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    ]

    resolved, fallback_entries, resolution_lookup = opt._resolve_fixed_source_matches(
        [entry],
        hit_tables,
    )

    assert resolved == []
    assert fallback_entries == []
    diag = resolution_lookup[id(entry)]
    assert diag["resolution_kind"] == "fixed_source"
    assert diag["resolution_reason"] in {
        "provider_local_duplicate_hkl_unproven",
        "unavailable_ambiguous",
    }
    assert diag["prediction_source_status"] == "unavailable_ambiguous"
    assert diag["provider_local_branch_match_row_count"] == 2


def test_resolve_fixed_source_matches_provider_local_duplicate_hkl_saved_px_rejects() -> None:
    entry = _provider_local_singleton_entry(
        source_row_index=24,
        source_branch_index=1,
        source_peak_index=1,
        resolved_peak_index=1,
        provider_local_subset_assignment="provider_local_duplicate_hkl_unproven",
        provider_local_subset_branch_provenance=False,
        provider_selected_source_identity_canonical={
            "normalized_hkl": [2, 0, 0],
            "source_table_index": 99,
            "source_row_index": 24,
            "source_peak_index": 0,
            "source_branch_index": 0,
        },
        sim_visual_detector_native_px=(9.0, 8.0),
    )
    hit_tables = [
        np.asarray(
            [
                [1.0, 4.0, 4.0, 10.0, 2.0, 0.0, 0.0],
                [1.0, 5.0, 5.0, 12.0, 2.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    ]

    resolved, fallback_entries, resolution_lookup = opt._resolve_fixed_source_matches(
        [entry],
        hit_tables,
    )

    assert resolved == []
    assert fallback_entries == []
    diag = resolution_lookup[id(entry)]
    assert diag["resolution_kind"] == "fixed_source"
    assert diag["resolution_reason"] == "provider_local_duplicate_hkl_unproven"
    assert diag["prediction_source_status"] == "unavailable_ambiguous"


def test_provider_local_saved_sim_detector_point_rejects_duplicate_hkl_display_shortcut() -> None:
    entry = _provider_local_singleton_entry(
        source_row_index=24,
        source_branch_index=1,
        source_peak_index=1,
        resolved_peak_index=1,
        provider_local_subset_assignment="provider_local_duplicate_hkl_unproven",
        provider_local_subset_branch_provenance=False,
        provider_selected_source_identity_canonical={
            "normalized_hkl": [2, 0, 0],
            "source_table_index": 99,
            "source_row_index": 24,
            "source_peak_index": 1,
            "source_branch_index": 1,
        },
        sim_visual_detector_display_px=(9.0, 8.0),
        sim_visual_detector_canonical_native_px=(19.0, 18.0),
        sim_visual_detector_canonical_native_source=("display_to_native_sim_coords(unit_test)"),
    )

    point, payload, reason = opt._provider_local_saved_sim_detector_point(entry)

    assert point is None
    assert reason == "provider_local_duplicate_hkl_unproven"
    assert payload["prediction_source_status"] == "unavailable_ambiguous"


def test_resolve_fixed_source_matches_saved_detector_rejects_ambiguous_stale_row() -> None:
    entry = _provider_local_singleton_entry(
        source_row_index=24,
        source_branch_index=1,
        source_peak_index=1,
        resolved_peak_index=1,
        provider_selected_source_identity_canonical={
            "normalized_hkl": [2, 0, 0],
            "source_table_index": 99,
            "source_row_index": 24,
            "source_peak_index": 1,
            "source_branch_index": 1,
        },
        sim_visual_detector_display_px=(9.0, 8.0),
        sim_visual_detector_canonical_native_px=(19.0, 18.0),
    )
    hit_tables = [
        np.asarray(
            [
                [1.0, 4.0, 4.0, 10.0, 2.0, 0.0, 0.0],
                [1.0, 5.0, 5.0, 12.0, 2.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    ]

    resolved, fallback_entries, resolution_lookup = opt._resolve_fixed_source_matches(
        [entry],
        hit_tables,
    )

    assert resolved == []
    assert fallback_entries == []
    diag = resolution_lookup[id(entry)]
    assert diag["resolution_reason"] == "provider_local_duplicate_hkl_unproven"
    assert diag["prediction_source_status"] == "unavailable_ambiguous"
    assert diag["provider_local_saved_sim_detector_fallback_rejected_reason"] == (
        "provider_local_duplicate_hkl_unproven"
    )


def test_resolve_fixed_source_matches_saved_detector_waits_for_row_proof() -> None:
    entry = _provider_local_singleton_entry(
        source_row_index=24,
        source_branch_index=1,
        source_peak_index=1,
        resolved_peak_index=1,
        provider_selected_source_identity_canonical={
            "normalized_hkl": [2, 0, 0],
            "source_table_index": 99,
            "source_peak_index": 1,
            "source_branch_index": 1,
        },
        sim_visual_detector_display_px=(9.0, 8.0),
        sim_visual_detector_canonical_native_px=(19.0, 18.0),
        sim_visual_detector_canonical_native_source=("display_to_native_sim_coords(unit_test)"),
    )
    hit_tables = [np.asarray([[1.0, 4.0, 4.0, -10.0, 3.0, 0.0, 0.0]], dtype=np.float64)]

    resolved, fallback_entries, resolution_lookup = opt._resolve_fixed_source_matches(
        [entry],
        hit_tables,
    )

    assert resolved == []
    assert fallback_entries == []
    diag = resolution_lookup[id(entry)]
    assert diag["resolution_kind"] == "fixed_source"
    assert diag["resolution_reason"] == "prediction_branch_source_switched"
    assert diag["resolution_subreason"] == "provider_local_hkl_mismatch"
    assert diag["provider_local_saved_sim_detector_fallback_rejected_reason"] == (
        "provider_local_hkl_mismatch"
    )


def test_geometry_fit_correspondence_saved_detector_waits_for_row_proof() -> None:
    correspondence = _provider_local_singleton_entry(
        source_row_index=24,
        source_branch_index=1,
        source_peak_index=1,
        resolved_peak_index=1,
        provider_selected_source_identity_canonical={
            "normalized_hkl": [2, 0, 0],
            "source_table_index": 99,
            "source_peak_index": 1,
            "source_branch_index": 1,
        },
        sim_visual_detector_display_px=(9.0, 8.0),
        sim_visual_detector_canonical_native_px=(19.0, 18.0),
        sim_visual_detector_canonical_native_source=("display_to_native_sim_coords(unit_test)"),
    )
    hit_tables = [np.asarray([[1.0, 4.0, 4.0, -10.0, 3.0, 0.0, 0.0]], dtype=np.float64)]

    point, payload = opt._geometry_fit_correspondence_simulated_point_payload(
        correspondence,
        hit_tables=hit_tables,
        max_positions=np.zeros((1, 6), dtype=np.float64),
    )

    assert point is None
    assert payload["resolution_reason"] == "prediction_branch_source_switched"
    assert payload["resolution_subreason"] == "provider_local_hkl_mismatch"
    assert payload["provider_local_saved_sim_detector_fallback_rejected_reason"] == (
        "provider_local_hkl_mismatch"
    )


def test_resolve_fixed_source_matches_does_not_upgrade_local_to_full_reflection() -> None:
    entry = _provider_local_singleton_entry(
        source_row_index=24,
        source_branch_index=1,
        source_peak_index=1,
        resolved_peak_index=1,
    )
    hit_tables = [np.asarray([[1.0, 4.0, 4.0, 10.0, 2.0, 0.0, 0.0]], dtype=np.float64)]

    resolved, fallback_entries, resolution_lookup = opt._resolve_fixed_source_matches(
        [entry],
        hit_tables,
    )

    assert len(resolved) == 1
    assert fallback_entries == []
    assert entry.get("source_reflection_namespace") != "full_reflection"
    assert resolution_lookup[id(entry)].get("source_reflection_namespace") != "full_reflection"


def test_resolve_fixed_source_matches_rejects_provider_local_singleton_negatives() -> None:
    base_hit_row = [1.0, 4.0, 4.0, -10.0, 2.0, 0.0, 0.0]
    cases = [
        (
            _provider_local_singleton_entry(),
            [np.empty((0, 7), dtype=np.float64)],
            "zero_row",
            "provider_local_hkl_mismatch",
        ),
        (
            _provider_local_singleton_entry(),
            [
                np.asarray(
                    [
                        base_hit_row,
                        [1.0, 5.0, 5.0, -12.0, 2.0, 0.0, 0.0],
                    ],
                    dtype=np.float64,
                )
            ],
            "multi_row",
            "provider_local_branch_match_zero",
        ),
        (
            _provider_local_singleton_entry(),
            [np.asarray([[1.0, 4.0, 4.0, -10.0, 3.0, 0.0, 0.0]], dtype=np.float64)],
            "hkl_mismatch",
            "provider_local_hkl_mismatch",
        ),
        (
            _provider_local_singleton_entry(
                provider_local_subset_assignment="provider_local_duplicate_hkl_unproven",
                provider_local_subset_branch_provenance=False,
                source_branch_index=None,
                source_peak_index=None,
                resolved_peak_index=None,
            ),
            [np.asarray([[1.0, 4.0, 4.0, 10.0, 2.0, 0.0, 0.0]], dtype=np.float64)],
            "duplicate_hkl_unproven",
            "provider_local_duplicate_hkl_unproven",
        ),
        (
            _provider_local_singleton_entry(
                source_branch_index=0,
                source_peak_index=1,
                resolved_peak_index=None,
            ),
            [np.asarray([base_hit_row], dtype=np.float64)],
            "conflicting_branch_ids_do_not_mask_mismatch",
            "provider_local_branch_identity_conflict",
        ),
        (
            _provider_local_singleton_entry(provider_local_subset_provenance=False),
            [np.asarray([base_hit_row], dtype=np.float64)],
            "no_provider_local_provenance",
            "provider_local_subset_provenance_missing",
        ),
        (
            _provider_local_singleton_entry(
                fit_source_resolution_kind="source_row",
                provider_local_subset_provenance=False,
            ),
            [np.asarray([base_hit_row], dtype=np.float64)],
            "ordinary_stale_non_provider",
            "provider_local_subset_provenance_missing",
        ),
    ]

    for entry, hit_tables, _case_name, expected_reason in cases:
        resolved, fallback_entries, resolution_lookup = opt._resolve_fixed_source_matches(
            [entry],
            hit_tables,
        )

        assert resolved == []
        diag = resolution_lookup[id(entry)]
        if _case_name in {
            "conflicting_branch_ids_do_not_mask_mismatch",
            "ordinary_stale_non_provider",
        }:
            assert fallback_entries == [entry]
            assert diag["resolution_kind"] == "hkl_fallback"
            assert diag["resolution_reason"] == expected_reason
            continue
        assert fallback_entries == []
        assert diag["resolution_kind"] == "fixed_source"
        assert expected_reason in {
            diag.get("resolution_reason"),
            diag.get("resolution_subreason"),
        }
        assert diag["prediction_source_status"] in {
            "unavailable",
            "unavailable_ambiguous",
        }


def test_geometry_fit_correspondence_dynamic_payload_recovers_stale_provider_branch() -> None:
    correspondence = _provider_local_singleton_entry(q_group_key=("q", 2))
    hit_tables = [np.asarray([[1.0, 4.0, 4.0, -10.0, 2.0, 0.0, 0.0]], dtype=np.float64)]
    max_positions = np.zeros((1, 6), dtype=np.float64)

    point, payload = opt._geometry_fit_correspondence_simulated_point_payload(
        correspondence,
        hit_tables=hit_tables,
        max_positions=max_positions,
    )

    assert point == (4.0, 4.0)
    assert payload["resolution_reason"] == (
        "provider_local_stale_row_index_single_row_table_resolved"
    )
    assert payload["provider_local_stale_row_index_single_row_table_resolved"] is True
    assert payload["prediction_source_status"] == "available"
    assert payload["resolved_table_index"] == 0
    assert payload["requested_source_peak_index"] == 1
    assert payload["requested_source_row_index"] == 24
    assert payload["hit_table_row_count"] == 1
    assert payload["valid_row_count"] == 1
    assert payload["hkl_exists_in_table"] is True
    assert payload["branch_filtered_row_count"] == 0
    assert payload["requested_branch_exists"] is False
    assert payload["requested_peak_exists"] is False
    assert payload["legacy_peak_exists"] is False
    assert payload["projection_available"] is True
    assert payload["projection_finite"] is True
    assert payload["off_detector"] is False
    assert "fallback" not in str(payload).lower()


def test_geometry_fit_correspondence_dynamic_payload_marks_peak_from_branch_row() -> None:
    correspondence = _provider_local_singleton_entry()
    hit_tables = [np.asarray([[1.0, 4.0, 4.0, 10.0, 2.0, 0.0, 0.0]], dtype=np.float64)]

    point, payload = opt._geometry_fit_correspondence_simulated_point_payload(
        correspondence,
        hit_tables=hit_tables,
        max_positions=np.zeros((1, 6), dtype=np.float64),
    )

    assert point == (4.0, 4.0)
    assert payload["resolution_reason"] == (
        "provider_local_stale_row_index_single_row_table_resolved"
    )
    assert payload["provider_local_stale_row_index_single_row_table_resolved"] is True
    assert payload["prediction_source_status"] == "available"
    assert payload["requested_branch_exists"] is True
    assert payload["requested_peak_exists"] is True
    assert payload["legacy_peak_exists"] is False
    assert payload["branch_filtered_row_count"] == 1


def test_geometry_fit_correspondence_local_branch_invalid_frozen_recovers_provider_branch() -> None:
    correspondence = _provider_local_singleton_entry(
        frozen_locator_kind="local_branch",
        frozen_table_namespace="current_full_local",
        frozen_table_index=0,
        frozen_branch_index=99,
    )
    hit_tables = [np.asarray([[1.0, 4.0, 4.0, 10.0, 2.0, 0.0, 0.0]], dtype=np.float64)]

    point, payload = opt._geometry_fit_correspondence_simulated_point_payload(
        correspondence,
        hit_tables=hit_tables,
        max_positions=np.zeros((1, 6), dtype=np.float64),
    )

    assert point == (4.0, 4.0)
    assert payload["resolution_reason"] == ("provider_local_branch_recovered_stale_peak_index")
    assert payload["resolution_subreason"] == "provider_local_branch_identity_unique"
    assert payload["resolved_peak_index"] == 1
    assert payload["requested_branch_exists"] is True
    assert payload["requested_peak_exists"] is True
    assert payload["legacy_peak_exists"] is False
    assert payload["frozen_branch_index_recovered_from_provider_identity"] is True
    assert "fallback" not in str(payload).lower()


def test_geometry_fit_correspondence_local_branch_invalid_frozen_rejects_ambiguous_branch() -> None:
    correspondence = _provider_local_singleton_entry(
        frozen_locator_kind="local_branch",
        frozen_table_namespace="current_full_local",
        frozen_table_index=0,
        frozen_branch_index=99,
    )

    point, payload = opt._geometry_fit_correspondence_simulated_point_payload(
        correspondence,
        hit_tables=[
            np.asarray(
                [
                    [1.0, 4.0, 4.0, 10.0, 2.0, 0.0, 0.0],
                    [2.0, 5.0, 5.0, 12.0, 2.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        ],
        max_positions=np.zeros((1, 6), dtype=np.float64),
    )

    assert point is None
    assert payload["resolution_reason"] == "missing_source_peak_not_recoverable"
    assert payload["resolution_subreason"] == "provider_local_branch_match_multiple"
    assert payload["provider_local_branch_match_row_count"] == 2
    assert "fallback" not in str(payload).lower()


def test_geometry_fit_correspondence_local_branch_invalid_frozen_rejects_wrong_branch() -> None:
    correspondence = _provider_local_singleton_entry(
        frozen_locator_kind="local_branch",
        frozen_table_namespace="current_full_local",
        frozen_table_index=0,
        frozen_branch_index=99,
    )

    point, payload = opt._geometry_fit_correspondence_simulated_point_payload(
        correspondence,
        hit_tables=[np.asarray([[1.0, 4.0, 4.0, -10.0, 2.0, 0.0, 0.0]], dtype=np.float64)],
        max_positions=np.zeros((1, 6), dtype=np.float64),
    )

    assert point is None
    assert payload["resolution_reason"] == "missing_source_peak_not_recoverable"
    assert payload["resolution_subreason"] == "provider_local_branch_match_zero"
    assert payload["provider_local_branch_hkl_row_count"] == 1
    assert payload["provider_local_branch_match_row_count"] == 0
    assert "fallback" not in str(payload).lower()


def test_geometry_fit_correspondence_local_branch_invalid_frozen_rejects_unknown_branch() -> None:
    correspondence = _provider_local_singleton_entry(
        frozen_locator_kind="local_branch",
        frozen_table_namespace="current_full_local",
        frozen_table_index=0,
        frozen_branch_index=99,
    )

    point, payload = opt._geometry_fit_correspondence_simulated_point_payload(
        correspondence,
        hit_tables=[
            np.asarray(
                [
                    [1.0, 4.0, 4.0, 10.0, 2.0, 0.0, 0.0],
                    [2.0, 5.0, 5.0, 0.0, 2.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        ],
        max_positions=np.zeros((1, 6), dtype=np.float64),
    )

    assert point is None
    assert payload["resolution_reason"] == "missing_source_peak_not_recoverable"
    assert payload["resolution_subreason"] == ("provider_local_branch_ambiguous_missing_metadata")
    assert payload["provider_local_branch_hkl_row_count"] == 2
    assert payload["provider_local_branch_match_row_count"] == 1
    assert payload["provider_local_branch_missing_metadata_count"] == 1
    assert "fallback" not in str(payload).lower()


def test_geometry_fit_correspondence_local_branch_invalid_frozen_missing_identity_fails() -> None:
    correspondence = _provider_local_singleton_entry(
        frozen_locator_kind="local_branch",
        frozen_table_namespace="current_full_local",
        frozen_table_index=0,
        frozen_branch_index=99,
        source_branch_index=None,
        source_peak_index=None,
        resolved_peak_index=None,
    )

    point, payload = opt._geometry_fit_correspondence_simulated_point_payload(
        correspondence,
        hit_tables=[np.asarray([[1.0, 4.0, 4.0, 10.0, 2.0, 0.0, 0.0]], dtype=np.float64)],
        max_positions=np.zeros((1, 6), dtype=np.float64),
    )

    assert point is None
    assert payload["resolution_reason"] == "missing_source_peak_not_recoverable"
    assert payload["resolution_subreason"] == "missing_provider_local_branch_identity"
    assert payload["requested_peak_exists"] is False
    assert payload["legacy_peak_exists"] is False
    assert "fallback" not in str(payload).lower()


def test_geometry_fit_correspondence_local_branch_invalid_frozen_non_provider_unchanged() -> None:
    correspondence = {
        "hkl": (2, 0, 0),
        "resolved_table_index": 0,
        "frozen_locator_kind": "local_branch",
        "frozen_table_namespace": "current_full_local",
        "frozen_table_index": 0,
        "frozen_branch_index": 99,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "resolved_peak_index": 1,
    }

    point, payload = opt._geometry_fit_correspondence_simulated_point_payload(
        correspondence,
        hit_tables=[np.asarray([[1.0, 4.0, 4.0, 10.0, 2.0, 0.0, 0.0]], dtype=np.float64)],
        max_positions=np.zeros((1, 6), dtype=np.float64),
    )

    assert point is None
    assert payload["resolution_reason"] == "missing_source_peak_index"
    assert "missing_source_peak_not_recoverable" not in str(payload)


@pytest.mark.parametrize(
    ("correspondence", "hit_tables", "expected_subreason"),
    [
        (
            _provider_local_singleton_entry(),
            [
                np.asarray(
                    [
                        [1.0, 4.0, 4.0, -10.0, 2.0, 0.0, 0.0],
                        [1.0, 5.0, 5.0, -12.0, 2.0, 0.0, 0.0],
                    ],
                    dtype=np.float64,
                )
            ],
            "provider_local_branch_match_zero",
        ),
        (
            _provider_local_singleton_entry(),
            [np.asarray([[1.0, 4.0, 4.0, -10.0, 3.0, 0.0, 0.0]], dtype=np.float64)],
            "provider_local_hkl_mismatch",
        ),
    ],
)
def test_geometry_fit_correspondence_dynamic_payload_rejects_unsafe_provider_recovery(
    correspondence: dict[str, object],
    hit_tables: list[np.ndarray],
    expected_subreason: str,
) -> None:
    point, payload = opt._geometry_fit_correspondence_simulated_point_payload(
        correspondence,
        hit_tables=hit_tables,
        max_positions=np.zeros((1, 6), dtype=np.float64),
    )

    assert point is None
    assert expected_subreason in {
        payload.get("resolution_reason"),
        payload.get("resolution_subreason"),
    }
    assert payload["prediction_source_status"] in {
        "unavailable",
        "unavailable_ambiguous",
    }
    assert payload["resolved_table_index"] == 0
    assert payload["provider_local_saved_sim_detector_fallback_rejected_reason"]


def test_geometry_fit_correspondence_dynamic_payload_accepts_single_row_hkl_proof() -> None:
    for correspondence, hit_table in (
        (
            _provider_local_singleton_entry(),
            np.asarray([[1.0, 4.0, 4.0, 0.0, 2.0, 0.0, 0.0]], dtype=np.float64),
        ),
        (
            _provider_local_singleton_entry(
                provider_local_subset_assignment="provider_local_unique_hkl",
                provider_local_subset_branch_provenance=False,
            ),
            np.asarray([[1.0, 4.0, 4.0, -10.0, 2.0, 0.0, 0.0]], dtype=np.float64),
        ),
    ):
        point, payload = opt._geometry_fit_correspondence_simulated_point_payload(
            correspondence,
            hit_tables=[hit_table],
            max_positions=np.zeros((1, 6), dtype=np.float64),
        )

        assert point == (4.0, 4.0)
        assert payload["resolution_reason"] == (
            "provider_local_stale_row_index_single_row_table_resolved"
        )
        assert payload["prediction_source_status"] == "available"


def test_geometry_fit_correspondence_dynamic_payload_does_not_search_hkl_wide() -> None:
    correspondence = _provider_local_singleton_entry()
    hit_tables = [
        np.empty((0, 7), dtype=np.float64),
        np.asarray([[1.0, 8.0, 8.0, -10.0, 2.0, 0.0, 0.0]], dtype=np.float64),
    ]

    point, payload = opt._geometry_fit_correspondence_simulated_point_payload(
        correspondence,
        hit_tables=hit_tables,
        max_positions=np.zeros((2, 6), dtype=np.float64),
    )

    assert point is None
    assert payload["resolution_reason"] == "prediction_branch_source_switched"
    assert payload["resolution_subreason"] == "provider_local_hkl_mismatch"
    assert payload["resolved_table_index"] == 0
    assert payload["hit_table_row_count"] == 0
    assert payload["hkl_exists_in_table"] is False


def test_geometry_fit_correspondence_simulated_point_prefers_branch_identity() -> None:
    correspondence = {
        "source_reflection_index": 0,
        "resolved_table_index": 0,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }
    hit_tables = [
        np.asarray(
            [
                [1.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 8.0, 8.0, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    ]
    max_positions = np.asarray([[1.0, 2.0, 2.0, 1.0, 8.0, 8.0]], dtype=np.float64)

    point, reason = opt._geometry_fit_correspondence_simulated_point(
        correspondence,
        hit_tables=hit_tables,
        max_positions=max_positions,
    )

    assert point == (8.0, 8.0)
    assert reason == "resolved_source_peak"


def test_prepare_reflection_subset_clears_stale_local_source_ids() -> None:
    miller = np.array(
        [
            [2.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([2.0, 7.0], dtype=np.float64)
    measured = [
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 13,
            "source_reflection_index": 13,
            "source_row_index": 0,
            "source_peak_index": 1,
            "fit_source_identity_only": True,
        }
    ]

    subset = opt._prepare_reflection_subset(miller, intensities, measured)

    assert subset.fixed_source_reflection_count == 0
    assert subset.fallback_hkl_count == 1
    assert np.array_equal(subset.original_indices, np.array([0], dtype=np.int64))
    for key in (
        "source_table_index",
        "source_reflection_index",
        "source_row_index",
        "source_peak_index",
        "resolved_table_index",
        "resolved_peak_index",
    ):
        assert key not in subset.measured_entries[0]


def test_filter_simulation_subset_preserves_local_fixed_source_rows() -> None:
    miller = np.array(
        [
            [5.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([5.0, 2.0, 7.0], dtype=np.float64)
    measured = [
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 99,
            "source_peak_index": 1,
            "q_group_key": ("q", 2),
            "branch_group_key": ("branch", 1),
            "fit_source_resolution_kind": "provider_fixed_source_local",
            "optimizer_request_source": "provider_pair",
            "optimizer_request_has_fixed_source": True,
            "optimizer_request_fallback_row": False,
            "provider_selected_source_identity_canonical": {
                "normalized_hkl": [2, 0, 0],
                "source_table_index": 99,
                "source_peak_index": 1,
            },
        }
    ]

    subset = opt._prepare_reflection_subset(miller, intensities, measured)

    assert subset.reduced is True
    assert subset.fixed_source_reflection_count == 1
    assert subset.fallback_hkl_count == 0
    assert np.array_equal(subset.original_indices, np.array([1], dtype=np.int64))
    entry = subset.measured_entries[0]
    assert entry["original_source_table_index"] == 99
    assert entry["source_table_index"] == 99
    assert entry["resolved_table_index"] == 0
    assert entry["source_peak_index"] == 1
    assert entry["resolved_peak_index"] == 1
    assert entry["q_group_key"] == ("q", 2)
    assert entry["branch_group_key"] == ("branch", 1)
    assert "source_reflection_index" not in entry
    hit_tables = [np.asarray([[1.0, 4.0, 4.0, 10.0, 2.0, 0.0, 0.0]], dtype=np.float64)]
    resolved, fallback_entries, resolution_lookup = opt._resolve_fixed_source_matches(
        subset.measured_entries,
        hit_tables,
    )
    assert len(resolved) == 1
    assert fallback_entries == []
    assert resolution_lookup[id(entry)]["resolution_kind"] == "fixed_source"
    assert resolution_lookup[id(entry)]["resolution_reason"] == "resolved"


def test_filter_simulation_subset_assigns_duplicate_hkl_local_fixed_rows() -> None:
    miller = np.array(
        [
            [5.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([5.0, 2.0, 3.0, 7.0], dtype=np.float64)
    measured = [
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 99,
            "source_peak_index": 0,
            "source_branch_index": 0,
            "source_row_index": 24,
            "fit_source_resolution_kind": "provider_fixed_source_local",
            "optimizer_request_source": "provider_pair",
            "optimizer_request_has_fixed_source": True,
            "optimizer_request_fallback_row": False,
        },
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 8.0,
            "y": 8.0,
            "source_table_index": 101,
            "source_peak_index": 1,
            "source_branch_index": 1,
            "source_row_index": 24,
            "fit_source_resolution_kind": "provider_fixed_source_local",
            "optimizer_request_source": "provider_pair",
            "optimizer_request_has_fixed_source": True,
            "optimizer_request_fallback_row": False,
        },
    ]

    subset = opt._prepare_reflection_subset(miller, intensities, measured)

    assert subset.reduced is True
    assert subset.fixed_source_reflection_count == 2
    assert subset.fallback_hkl_count == 0
    assert np.array_equal(subset.original_indices, np.array([1, 2], dtype=np.int64))
    assert subset.measured_entries[0]["original_source_table_index"] == 99
    assert subset.measured_entries[0]["resolved_table_index"] == 0
    assert subset.measured_entries[0]["source_peak_index"] == 0
    assert (
        subset.measured_entries[0]["provider_local_subset_assignment"]
        == "provider_local_duplicate_hkl_branch"
    )
    assert subset.measured_entries[0]["provider_local_subset_branch_provenance"] is True
    assert subset.measured_entries[1]["original_source_table_index"] == 101
    assert subset.measured_entries[1]["resolved_table_index"] == 1
    assert subset.measured_entries[1]["source_peak_index"] == 1
    assert (
        subset.measured_entries[1]["provider_local_subset_assignment"]
        == "provider_local_duplicate_hkl_branch"
    )
    assert subset.measured_entries[1]["provider_local_subset_branch_provenance"] is True
    assert "source_reflection_index" not in subset.measured_entries[0]
    assert "source_reflection_index" not in subset.measured_entries[1]

    hit_tables = [
        np.asarray(
            [
                [1.0, 4.0, 4.0, -10.0, 2.0, 0.0, 0.0],
                [1.0, 5.0, 5.0, 10.0, 2.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                [1.0, 7.0, 7.0, -10.0, 2.0, 0.0, 0.0],
                [1.0, 8.0, 8.0, 10.0, 2.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    ]
    resolved, fallback_entries, resolution_lookup = opt._resolve_fixed_source_matches(
        subset.measured_entries,
        hit_tables,
    )
    assert len(resolved) == 2
    assert fallback_entries == []
    assert all(
        resolution_lookup[id(entry)]["resolution_kind"] == "fixed_source"
        for entry in subset.measured_entries
    )


def test_filter_simulation_subset_marks_duplicate_hkl_local_rows_without_branch_unproven() -> None:
    miller = np.array(
        [
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([2.0, 3.0], dtype=np.float64)
    measured = [
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 99,
            "source_row_index": 24,
            "q_group_key": ("q", 2),
            "fit_source_resolution_kind": "provider_fixed_source_local",
            "optimizer_request_source": "provider_pair",
            "optimizer_request_has_fixed_source": True,
            "optimizer_request_fallback_row": False,
        }
    ]

    subset = opt._prepare_reflection_subset(miller, intensities, measured)

    assert subset.reduced is True
    assert subset.fixed_source_reflection_count == 1
    assert subset.fallback_hkl_count == 0
    entry = subset.measured_entries[0]
    assert entry["provider_local_subset_assignment"] == "provider_local_duplicate_hkl_unproven"
    assert entry["provider_local_subset_branch_provenance"] is False
    assert entry["provider_local_subset_duplicate_hkl_count"] == 2
    resolved, fallback_entries, resolution_lookup = opt._resolve_fixed_source_matches(
        [entry],
        [np.asarray([[1.0, 4.0, 4.0, 10.0, 2.0, 0.0, 0.0]], dtype=np.float64)],
    )
    assert resolved == []
    assert fallback_entries == []
    diag = resolution_lookup[id(entry)]
    assert diag["resolution_kind"] == "fixed_source"
    assert diag["resolution_reason"] == "provider_local_duplicate_hkl_unproven"
    assert diag["prediction_source_status"] == "unavailable_ambiguous"


def test_geometry_fit_correspondence_simulated_point_ignores_stale_source_table_ids() -> None:
    correspondence = {
        "source_table_index": 13,
        "source_reflection_index": 13,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }
    hit_tables = [
        np.asarray(
            [
                [1.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 8.0, 8.0, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    ]
    max_positions = np.asarray([[1.0, 2.0, 2.0, 1.0, 8.0, 8.0]], dtype=np.float64)

    point, reason = opt._geometry_fit_correspondence_simulated_point(
        correspondence,
        hit_tables=hit_tables,
        max_positions=max_positions,
    )

    assert point is None
    assert reason == "missing_source_table_index"


def test_collect_geometry_fit_simulated_candidates_keeps_deadband_fallback_branchless(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        opt,
        "hit_tables_to_max_positions",
        lambda hit_tables: np.asarray([[1.0, 150.0, 250.0, 0.0, 0.0, 0.0]], dtype=np.float64),
    )

    candidates = opt._collect_geometry_fit_simulated_candidates(
        np.asarray([[1.0, 0.0, 3.0]], dtype=np.float64),
        [
            np.asarray(
                [[50.0, 150.0, 250.0, 5.0e-4, 1.0, 0.0, 3.0]],
                dtype=np.float64,
            )
        ],
        original_indices=np.asarray([7], dtype=np.int64),
    )

    candidate = candidates[(1, 0, 3)][0]
    assert candidate["source_reflection_index"] == 7
    assert "source_branch_index" not in candidate
    assert "source_peak_index" not in candidate
    assert "resolved_peak_index" not in candidate


def test_geometry_fit_correspondence_simulated_point_rejects_legacy_branch_alias_for_trusted_identity() -> (
    None
):
    point, reason = opt._geometry_fit_correspondence_simulated_point(
        {
            "source_reflection_index": 7,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "resolved_table_index": 0,
            "source_row_index": 0,
            "source_peak_index": 1,
        },
        hit_tables=[
            np.asarray(
                [
                    [1.0, 2.0, 2.0, -22.0, 1.0, 0.0, 3.0],
                    [1.0, 8.0, 8.0, 22.0, 1.0, 0.0, 3.0],
                ],
                dtype=np.float64,
            )
        ],
        max_positions=np.asarray([[1.0, 2.0, 2.0, 1.0, 8.0, 8.0]], dtype=np.float64),
    )

    assert point is None
    assert reason == "missing_source_peak_index"


def test_resolve_fixed_source_matches_allows_trusted_deadband_source_row_fallback() -> None:
    entry = {
        "hkl": (1, 0, 3),
        "label": "1,0,3",
        "x": 4.0,
        "y": 4.0,
        "source_reflection_index": 7,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "resolved_table_index": 0,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }
    hit_tables = [
        np.asarray(
            [[1.0, 4.0, 4.0, 0.0, 1.0, 0.0, 3.0]],
            dtype=np.float64,
        )
    ]

    resolved, fallback_entries, resolution_lookup = opt._resolve_fixed_source_matches(
        [entry],
        hit_tables,
    )

    assert fallback_entries == []
    assert len(resolved) == 1
    assert resolved[0][1] == (4.0, 4.0)
    assert resolved[0][2] == (1, 0, 3)
    assert resolution_lookup[id(entry)]["resolution_kind"] == "fixed_source"
    assert resolution_lookup[id(entry)]["resolution_reason"] == "resolved"


def test_geometry_fit_correspondence_simulated_point_allows_trusted_deadband_source_row_fallback() -> (
    None
):
    point, reason = opt._geometry_fit_correspondence_simulated_point(
        {
            "hkl": (1, 0, 3),
            "source_reflection_index": 0,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "frozen_locator_kind": "trusted_branch",
            "frozen_table_namespace": "full_reflection",
            "frozen_table_index": 0,
            "frozen_branch_index": 1,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
        },
        hit_tables=[
            np.asarray(
                [[1.0, 4.0, 4.0, 0.0, 1.0, 0.0, 3.0]],
                dtype=np.float64,
            )
        ],
        max_positions=np.asarray([[1.0, 4.0, 4.0, 4.0, 4.0, 4.0]], dtype=np.float64),
    )

    assert point == (4.0, 4.0)
    assert reason == "resolved_source_row_fallback"


def test_geometry_fit_correspondence_simulated_point_prefers_trusted_source_row_for_row_bound_fits() -> (
    None
):
    point, reason = opt._geometry_fit_correspondence_simulated_point(
        {
            "hkl": (1, 0, 3),
            "source_reflection_index": 7,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 1,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "fit_source_resolution_kind": "source_row",
        },
        hit_tables=[
            np.asarray(
                [
                    [1.0, 2.0, 2.0, 0.0, 1.0, 0.0, 3.0],
                    [1.0, 8.0, 8.0, 0.0, 1.0, 0.0, 3.0],
                ],
                dtype=np.float64,
            )
        ],
        max_positions=np.asarray([[1.0, 2.0, 2.0, 1.0, 8.0, 8.0]], dtype=np.float64),
        trusted_full_reflection_local_index_map={7: 0},
    )

    assert point == (8.0, 8.0)
    assert reason == "resolved_source_row"


def test_geometry_fit_correspondence_simulated_point_uses_source_row_provenance() -> None:
    point, reason = opt._geometry_fit_correspondence_simulated_point(
        {
            "resolved_table_index": 0,
            "source_row_index": 9,
            "hkl": (1, 0, 0),
        },
        hit_tables=[
            np.asarray(
                [
                    [np.nan, 2.0, 2.0, -22.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 8.0, 8.0, 22.0, 1.0, 0.0, 0.0, 0.0, 9.0, 0.0],
                ],
                dtype=np.float64,
            )
        ],
        max_positions=np.asarray([[1.0, 2.0, 2.0, 1.0, 8.0, 8.0]], dtype=np.float64),
    )

    assert point == (8.0, 8.0)
    assert reason == "resolved_source_row"


def _install_identity_bridge_solver_stubs(monkeypatch) -> None:
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        miller_local = np.asarray(args[0], dtype=np.float64)
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = []
        for row in miller_local:
            h, k, l = (int(round(float(v))) for v in row[:3])
            if (h, k, l) == (1, 0, 0):
                hit_tables.append(
                    np.asarray(
                        [
                            [1.0, 2.0, 2.0, -10.0, 1.0, 0.0, 0.0],
                            [1.0, 8.0, 8.0, 10.0, 1.0, 0.0, 0.0],
                        ],
                        dtype=np.float64,
                    )
                )
            else:
                hit_tables.append(
                    np.asarray(
                        [[1.0, 50.0, 50.0, 0.0, float(h), float(k), float(l)]],
                        dtype=np.float64,
                    )
                )
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)


def _run_identity_bridge_case(
    monkeypatch,
    *,
    trusted: bool,
    enable_full_beam_polish: bool = True,
    display_point: tuple[float, float] = (8.0, 8.0),
    detector_point: tuple[float, float] | None = None,
) -> dict[str, object]:
    _install_identity_bridge_solver_stubs(monkeypatch)

    image_size = 20
    miller = np.array(
        [[5.0, 0.0, 0.0], [1.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    intensities = np.array([5.0, 10.0, 7.0], dtype=np.float64)
    input_pair = {
        "pair_id": "bg0:pair0",
        "fit_run_id": "fit-123",
        "label": "1,0,0",
        "hkl": (1, 0, 0),
        "x": float(display_point[0]),
        "y": float(display_point[1]),
        "sigma_px": 1.0,
    }
    if detector_point is not None:
        input_pair.update(
            {
                "detector_x": float(detector_point[0]),
                "detector_y": float(detector_point[1]),
                "background_detector_x": float(detector_point[0]),
                "background_detector_y": float(detector_point[1]),
                "native_col": float(detector_point[0]),
                "native_row": float(detector_point[1]),
            }
        )
    if trusted:
        input_pair.update(
            {
                "source_table_index": 99,
                "source_reflection_index": 1,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "source_row_index": 0,
                "source_branch_index": 1,
                "source_peak_index": 1,
            }
        )

    subset = opt._prepare_reflection_subset(miller, intensities, [dict(input_pair)])
    subset_hit_tables = [
        np.asarray(
            [
                [1.0, 2.0, 2.0, -10.0, 1.0, 0.0, 0.0],
                [1.0, 8.0, 8.0, 10.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
    ]
    resolved, fallback_entries, resolution_lookup = opt._resolve_fixed_source_matches(
        subset.measured_entries,
        subset_hit_tables,
    )
    params = _base_params(image_size, optics_mode=1)
    params["mosaic_params"] = {
        "beam_x_array": np.array([0.0, 0.1], dtype=np.float64),
        "beam_y_array": np.zeros(2, dtype=np.float64),
        "theta_array": np.zeros(2, dtype=np.float64),
        "phi_array": np.zeros(2, dtype=np.float64),
        "sigma_mosaic_deg": 0.2,
        "gamma_mosaic_deg": 0.1,
        "eta": 0.05,
        "wavelength_array": np.ones(2, dtype=np.float64),
    }
    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=[dict(input_pair)],
        var_names=["gamma"],
        experimental_image=np.zeros((image_size, image_size), dtype=np.float64),
        refinement_config={
            "solver": {"restarts": 0, "weighted_matching": False},
            "full_beam_polish": {
                "enabled": bool(enable_full_beam_polish),
                "max_nfev": 10,
            },
            "identifiability": {"enabled": False},
        },
    )

    return {
        "input_pair": dict(input_pair),
        "subset_output_pair": dict(subset.measured_entries[0]),
        "resolved_entries": list(resolved),
        "fallback_entries": list(fallback_entries),
        "resolved_diag": dict(resolution_lookup[id(subset.measured_entries[0])]),
        "point_diag": dict(result.point_match_diagnostics[0]),
        "seed_record": (
            dict(result.full_beam_polish_summary["seed_correspondence_records"][0])
            if result.full_beam_polish_summary.get("seed_correspondence_records")
            else None
        ),
        "result": result,
    }


def test_resolve_fixed_source_matches_preserves_trusted_identity_payload(
    monkeypatch,
) -> None:
    case = _run_identity_bridge_case(monkeypatch, trusted=True)
    trusted_pair = dict(case["input_pair"])
    subset_output_pair = dict(case["subset_output_pair"])
    resolved_diag = dict(case["resolved_diag"])

    assert len(case["resolved_entries"]) == 1
    assert case["fallback_entries"] == []
    for record in (subset_output_pair, resolved_diag):
        for field in (
            "pair_id",
            "fit_run_id",
            "hkl",
            "source_reflection_index",
            "source_reflection_namespace",
            "source_reflection_is_full",
            "source_branch_index",
            "source_peak_index",
        ):
            assert record.get(field) == trusted_pair.get(field)
    assert resolved_diag.get("source_reflection_index_namespace") == "full_reflection"
    assert resolved_diag.get("source_table_index_namespace") == "full_hit_table"
    assert resolved_diag.get("source_row_index_namespace") == "full_hit_table"
    assert resolved_diag.get("source_peak_index_namespace") == "branch_index"
    assert resolved_diag.get("source_branch_index_namespace") == "branch_index"


def test_point_match_diagnostics_preserve_trusted_identity_payload(
    monkeypatch,
) -> None:
    case = _run_identity_bridge_case(monkeypatch, trusted=True)
    trusted_pair = dict(case["input_pair"])
    point_diag = dict(case["point_diag"])

    assert bool(case["result"].success) is True
    for field in (
        "pair_id",
        "fit_run_id",
        "hkl",
        "source_reflection_index",
        "source_reflection_namespace",
        "source_reflection_is_full",
        "source_branch_index",
        "source_peak_index",
    ):
        assert point_diag.get(field) == trusted_pair.get(field)
    assert point_diag.get("source_reflection_index_namespace") == "full_reflection"
    assert point_diag.get("source_table_index_namespace") == "full_hit_table"
    assert point_diag.get("source_row_index_namespace") == "full_hit_table"
    assert point_diag.get("source_peak_index_namespace") == "branch_index"
    assert point_diag.get("source_branch_index_namespace") == "branch_index"


def test_full_beam_seed_correspondence_prefers_detector_anchor_over_display_xy(
    monkeypatch,
) -> None:
    case = _run_identity_bridge_case(
        monkeypatch,
        trusted=True,
        display_point=(80.0, 80.0),
        detector_point=(8.0, 8.0),
    )

    point_diag = dict(case["point_diag"])
    seed_record = dict(case["seed_record"])
    start_diag = dict(case["result"].full_beam_polish_summary["start_point_match_diagnostics"][0])

    for record in (point_diag, seed_record, start_diag):
        assert float(record["measured_x"]) == 8.0
        assert float(record["measured_y"]) == 8.0
    assert float(point_diag["distance_px"]) == 0.0
    assert float(start_diag["distance_px"]) == 0.0


def test_fixed_correspondence_path_remaps_trusted_full_reflection_indices_without_polish(
    monkeypatch,
) -> None:
    case = _run_identity_bridge_case(
        monkeypatch,
        trusted=True,
        enable_full_beam_polish=False,
    )
    trusted_pair = dict(case["input_pair"])
    point_diag = dict(case["point_diag"])

    assert bool(case["result"].success) is True
    assert case["result"].final_metric_name == "central_point_match"
    assert point_diag["match_kind"] == "fixed_source"
    assert point_diag["match_status"] == "matched"
    assert point_diag["resolution_kind"] == "fixed_source"
    assert int(point_diag["resolved_table_index"]) == 0
    for field in (
        "pair_id",
        "fit_run_id",
        "hkl",
        "source_reflection_index",
        "source_reflection_namespace",
        "source_reflection_is_full",
        "source_branch_index",
        "source_peak_index",
    ):
        assert point_diag.get(field) == trusted_pair.get(field)


def test_fixed_correspondence_evaluator_remaps_trusted_full_reflection_indices(
    monkeypatch,
) -> None:
    _install_identity_bridge_solver_stubs(monkeypatch)

    image_size = 20
    miller = np.array(
        [[5.0, 0.0, 0.0], [1.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    intensities = np.array([5.0, 10.0, 7.0], dtype=np.float64)
    measured_entry = {
        "pair_id": "bg0:pair0",
        "fit_run_id": "fit-123",
        "label": "1,0,0",
        "hkl": (1, 0, 0),
        "measured_x": 8.0,
        "measured_y": 8.0,
        "x": 8.0,
        "y": 8.0,
        "sigma_px": 1.0,
        "source_table_index": 99,
        "source_reflection_index": 1,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "seed_match_status": "matched",
        "match_status": "matched",
        "resolution_kind": "fixed_source",
    }

    subset = opt._prepare_reflection_subset(miller, intensities, [dict(measured_entry)])
    dataset_ctx = opt.GeometryFitDatasetContext(
        dataset_index=0,
        label="dataset_0",
        theta_initial=0.0,
        subset=subset,
        experimental_image=np.zeros((image_size, image_size), dtype=np.float64),
    )
    params = _base_params(image_size, optics_mode=1)
    params["mosaic_params"] = {
        "beam_x_array": np.array([0.0, 0.1], dtype=np.float64),
        "beam_y_array": np.zeros(2, dtype=np.float64),
        "theta_array": np.zeros(2, dtype=np.float64),
        "phi_array": np.zeros(2, dtype=np.float64),
        "sigma_mosaic_deg": 0.2,
        "gamma_mosaic_deg": 0.1,
        "eta": 0.05,
        "wavelength_array": np.ones(2, dtype=np.float64),
    }

    correspondence = dict(subset.measured_entries[0])
    correspondence.update(
        {
            "measured_x": 8.0,
            "measured_y": 8.0,
            "seed_match_status": "matched",
            "match_status": "matched",
            "resolution_kind": "fixed_source",
        }
    )

    residuals, diagnostics, summary = opt._evaluate_geometry_fit_dataset_fixed_correspondences(
        params,
        dataset_ctx,
        [correspondence],
        image_size=image_size,
        weighted_matching=False,
        solver_f_scale=1.0,
        missing_pair_penalty=50.0,
        theta_value=0.0,
        use_measurement_uncertainty=False,
        anisotropic_uncertainty=False,
        match_radius_px=np.inf,
        collect_diagnostics=True,
    )

    assert residuals.shape == (2,)
    assert int(summary["fixed_source_resolved_count"]) == 1
    assert int(summary["matched_pair_count"]) == 1
    assert int(summary["missing_pair_count"]) == 0
    assert diagnostics[0]["match_kind"] == "fixed_correspondence"
    assert diagnostics[0]["match_status"] == "matched"
    assert diagnostics[0]["resolution_kind"] == "fixed_source"
    assert int(diagnostics[0]["resolved_table_index"]) == 0
    assert diagnostics[0]["correspondence_resolution_reason"] == "resolved_source_peak"
    assert np.isclose(float(diagnostics[0]["distance_px"]), 0.0)


def test_seed_correspondence_records_preserve_trusted_identity_payload(
    monkeypatch,
) -> None:
    case = _run_identity_bridge_case(monkeypatch, trusted=True)
    trusted_pair = dict(case["input_pair"])
    seed_record = dict(case["seed_record"])

    for field in (
        "pair_id",
        "fit_run_id",
        "hkl",
        "source_reflection_index",
        "source_reflection_namespace",
        "source_reflection_is_full",
        "source_branch_index",
        "source_peak_index",
    ):
        assert seed_record.get(field) == trusted_pair.get(field)
    assert seed_record.get("source_reflection_index_namespace") == "full_reflection"
    assert seed_record.get("source_table_index_namespace") == "full_hit_table"
    assert seed_record.get("source_row_index_namespace") == "full_hit_table"
    assert seed_record.get("source_peak_index_namespace") == "branch_index"
    assert seed_record.get("source_branch_index_namespace") == "branch_index"
    assert seed_record["frozen_locator_kind"] == "trusted_branch"
    assert seed_record["frozen_table_namespace"] == "full_reflection"


def test_full_beam_polish_freezes_trusted_row_locator_for_row_bound_fits(
    monkeypatch,
) -> None:
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.asarray(
                [
                    [1.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.0],
                    [1.0, 8.0, 8.0, 0.0, 1.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["mosaic_params"] = {
        "beam_x_array": np.array([0.0, 0.1], dtype=np.float64),
        "beam_y_array": np.zeros(2, dtype=np.float64),
        "theta_array": np.zeros(2, dtype=np.float64),
        "phi_array": np.zeros(2, dtype=np.float64),
        "sigma_mosaic_deg": 0.2,
        "gamma_mosaic_deg": 0.1,
        "eta": 0.05,
        "wavelength_array": np.ones(2, dtype=np.float64),
    }
    measured = [
        {
            "label": "1,0,0",
            "hkl": (1, 0, 0),
            "x": 8.0,
            "y": 8.0,
            "source_reflection_index": 0,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 1,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "fit_source_resolution_kind": "source_row",
        }
    ]

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=np.zeros((image_size, image_size), dtype=np.float64),
        refinement_config={
            "solver": {
                "restarts": 0,
                "dynamic_point_geometry_fit": True,
                "weighted_matching": False,
                "use_measurement_uncertainty": False,
            },
            "full_beam_polish": {"enabled": True, "max_nfev": 10},
            "identifiability": {"enabled": False},
        },
    )

    assert bool(result.success) is True
    assert result.point_match_diagnostics[0]["match_status"] == "matched"
    seed_record = dict(result.full_beam_polish_summary["seed_correspondence_records"][0])
    assert seed_record["frozen_locator_kind"] == "trusted_row"
    assert seed_record["frozen_table_namespace"] == "full_reflection"


def test_full_beam_polish_remaps_trusted_full_reflection_indices_into_subset_local_tables(
    monkeypatch,
) -> None:
    case = _run_identity_bridge_case(monkeypatch, trusted=True)
    trusted_pair = dict(case["input_pair"])
    result = case["result"]
    summary = dict(result.full_beam_polish_summary)
    summary_diag = dict(summary["point_match_diagnostics"][0])
    point_diag = dict(result.point_match_diagnostics[0])

    assert bool(result.success) is True
    assert bool(summary["accepted"]) is True
    assert str(summary["status"]) == "accepted"
    assert result.final_metric_name == "full_beam_fixed_correspondence"
    assert int(summary["matched_pair_count_before"]) == 1
    assert int(summary["matched_pair_count_after"]) == 1
    for diag in (summary_diag, point_diag):
        assert diag["match_kind"] == "full_beam_fixed"
        assert diag["match_status"] == "matched"
        assert diag["resolution_kind"] == "fixed_source"
        assert bool(diag["trusted_full_reflection_remapped"]) is True
        assert int(diag["frozen_table_index"]) == 1
        assert str(diag["frozen_table_namespace"]) == "full_reflection"
        assert int(diag["resolved_table_index"]) == 0
        for field in (
            "pair_id",
            "fit_run_id",
            "hkl",
            "source_reflection_index",
            "source_reflection_namespace",
            "source_reflection_is_full",
            "source_branch_index",
            "source_peak_index",
        ):
            assert diag.get(field) == trusted_pair.get(field)


def test_hkl_fallback_bridge_does_not_retrust_identity(monkeypatch) -> None:
    case = _run_identity_bridge_case(monkeypatch, trusted=False)
    resolved_diag = dict(case["resolved_diag"])
    point_diag = dict(case["point_diag"])
    seed_record = dict(case["seed_record"])

    assert case["resolved_entries"] == []
    assert len(case["fallback_entries"]) == 1
    assert resolved_diag["resolution_kind"] == "hkl_fallback"
    for record in (resolved_diag, point_diag, seed_record):
        assert record.get("pair_id") == "bg0:pair0"
        assert record.get("fit_run_id") == "fit-123"
        assert record.get("source_reflection_namespace") in (None, "")
        assert record.get("source_reflection_is_full") in (None, False)
    assert point_diag["resolution_kind"] == "hkl_fallback"
    assert seed_record["frozen_locator_kind"] == "local_branch"
    assert seed_record["frozen_table_namespace"] == "current_full_local"


def test_measured_source_peak_index_with_source_forwards_legacy_fallback(
    monkeypatch,
) -> None:
    seen: list[bool] = []

    def fake_resolve(entry, *, allow_legacy_peak_fallback=False):
        seen.append(bool(allow_legacy_peak_fallback))
        return None, None, None

    monkeypatch.setattr(opt, "resolve_canonical_branch", fake_resolve)

    peak_idx, peak_source = opt._measured_source_peak_index_with_source(
        {"source_peak_index": 1},
        allow_legacy_peak_fallback=True,
    )

    assert seen == [True]
    assert peak_idx == 1
    assert peak_source == "source_peak_index"


def test_geometry_fit_correspondence_simulated_point_allows_untrusted_local_row_locator() -> None:
    point, reason = opt._geometry_fit_correspondence_simulated_point(
        {
            "resolved_table_index": 0,
            "source_row_index": 1,
            "hkl": (1, 0, 0),
        },
        hit_tables=[
            np.asarray(
                [
                    [1.0, 2.0, 2.0, -22.0, 1.0, 0.0, 0.0],
                    [1.0, 8.0, 8.0, 22.0, 1.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        ],
        max_positions=np.asarray([[1.0, 2.0, 2.0, 1.0, 8.0, 8.0]], dtype=np.float64),
    )

    assert point == (8.0, 8.0)
    assert reason == "resolved_source_row"


def test_measured_detector_anchor_prefers_native_detector_coords() -> None:
    anchor, reason = opt._measured_detector_anchor(
        {
            "native_col": 1083.0,
            "native_row": 1151.0,
            "background_detector_x": 1083.0,
            "background_detector_y": 1151.0,
            "detector_x": 1848.0,
            "detector_y": 1083.0,
            "x": 1848.0,
            "y": 1083.0,
        }
    )

    assert anchor == (1083.0, 1151.0)
    assert reason == "resolved_native_anchor"


def test_measured_detector_anchor_prefers_background_detector_over_sim_native_hint() -> None:
    anchor, reason = opt._measured_detector_anchor(
        {
            "background_detector_x": 1083.0,
            "background_detector_y": 1151.0,
            "sim_native_x": 1501.0,
            "sim_native_y": 1602.0,
            "x": 1848.0,
            "y": 1083.0,
        }
    )

    assert anchor == (1083.0, 1151.0)
    assert reason == "resolved_background_detector_anchor"


def test_measured_detector_anchor_prefers_background_detector_over_detector_coords() -> None:
    anchor, reason = opt._measured_detector_anchor(
        {
            "background_detector_x": 1083.0,
            "background_detector_y": 1151.0,
            "detector_x": 1848.0,
            "detector_y": 1083.0,
            "x": 1848.0,
            "y": 1083.0,
        }
    )

    assert anchor == (1083.0, 1151.0)
    assert reason == "resolved_background_detector_anchor"


def test_measured_detector_anchor_ignores_sim_native_hint_when_display_anchor_exists() -> None:
    anchor, reason = opt._measured_detector_anchor(
        {
            "sim_native_x": 1501.0,
            "sim_native_y": 1602.0,
            "x": 1848.0,
            "y": 1083.0,
        }
    )

    assert anchor == (1848.0, 1083.0)
    assert reason == "resolved_display_anchor"


def test_measured_fit_space_anchor_prefers_background_detector_over_sim_native_hint() -> None:
    anchor, reason, metadata = opt._measured_fit_space_anchor(
        {
            "background_detector_x": 1083.0,
            "background_detector_y": 1151.0,
            "sim_native_x": 1501.0,
            "sim_native_y": 1602.0,
        },
        center=(1024.0, 1024.0),
        detector_distance=250.0,
        pixel_size=0.1,
    )

    assert anchor is not None
    assert reason == "background_detector_fit_space_anchor"
    assert metadata["anchor_source"] == "background_detector_fit_space_anchor"


def test_measured_fit_space_anchor_prefers_background_detector_over_detector_coords() -> None:
    anchor, reason, metadata = opt._measured_fit_space_anchor(
        {
            "background_detector_x": 1083.0,
            "background_detector_y": 1151.0,
            "detector_x": 1848.0,
            "detector_y": 1083.0,
        },
        center=(1024.0, 1024.0),
        detector_distance=250.0,
        pixel_size=0.1,
    )

    assert anchor is not None
    assert reason == "background_detector_fit_space_anchor"
    assert metadata["anchor_source"] == "background_detector_fit_space_anchor"


def test_measured_fit_space_anchor_ignores_sim_native_hint_when_display_anchor_exists() -> None:
    anchor, reason, metadata = opt._measured_fit_space_anchor(
        {
            "sim_native_x": 1501.0,
            "sim_native_y": 1602.0,
            "x": 1848.0,
            "y": 1083.0,
        },
        center=(1024.0, 1024.0),
        detector_distance=250.0,
        pixel_size=0.1,
    )

    assert anchor is not None
    assert reason == "display_fit_space_anchor"
    assert metadata["anchor_source"] == "display_fit_space_anchor"


def test_measured_fit_space_anchor_rejects_ambiguous_background_detector_frame_when_exact_projector_exists(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        opt,
        "_detector_pixels_to_fit_space",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("analytic detector fit-space path must not run")
        ),
    )

    def _unexpected_projector(*_args, **_kwargs):
        raise AssertionError("ambiguous detector frame must not call projector")

    dataset_ctx = opt.GeometryFitDatasetContext(
        dataset_index=0,
        label="bg0",
        theta_initial=0.0,
        subset=opt.ReflectionSimulationSubset(
            miller=np.zeros((0, 3), dtype=np.float64),
            intensities=np.zeros(0, dtype=np.float64),
            measured_entries=[],
            original_indices=np.zeros(0, dtype=np.int64),
            total_reflection_count=0,
            fixed_source_reflection_count=0,
            fallback_hkl_count=0,
            reduced=False,
        ),
        experimental_image=np.zeros((4, 4), dtype=np.float64),
        fit_space_projector=_unexpected_projector,
        fit_space_projector_kind="exact_caked_bundle",
    )

    anchor, reason, metadata = opt._measured_fit_space_anchor(
        {
            "background_detector_x": 1083.0,
            "background_detector_y": 1151.0,
        },
        center=(1024.0, 1024.0),
        detector_distance=250.0,
        pixel_size=0.1,
        dataset_ctx=dataset_ctx,
        local_params={"gamma": 0.0},
    )

    assert anchor is None
    assert reason == "invalid_dataset_fit_space_projector"
    assert metadata["measured_detector_field_name"] == (
        "background_detector_x/background_detector_y"
    )
    assert metadata["measured_detector_input_frame"] is None
    assert metadata["measured_detector_frame_reason"] == ("ambiguous_background_detector_frame")
    assert metadata["fit_space_source"] == "invalid_dataset_fit_space_projector"
    assert metadata["invalid_projection_reason"] == "ambiguous_measured_detector_frame"


def test_resolve_geometry_fit_correspondence_remaps_trusted_full_reflection_row_locator() -> None:
    point, payload = opt._resolve_geometry_fit_correspondence(
        {
            "hkl": (1, 0, 0),
            "source_reflection_index": 7,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "fit_source_resolution_kind": "source_row",
            "frozen_locator_kind": "trusted_row",
            "frozen_table_namespace": "full_reflection",
            "frozen_table_index": 7,
            "frozen_row_index": 1,
            "source_row_index": 1,
        },
        hit_tables=[
            np.asarray(
                [
                    [1.0, 2.0, 2.0, 0.0, 1.0, 0.0, 0.0],
                    [1.0, 8.0, 8.0, 0.0, 1.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        ],
        max_positions=np.asarray([[1.0, 2.0, 2.0, 1.0, 8.0, 8.0]], dtype=np.float64),
        trusted_full_reflection_local_index_map={7: 0},
    )

    assert point == (8.0, 8.0)
    assert payload["resolution_reason"] == "resolved_source_row"
    assert int(payload["resolved_table_index"]) == 0
    assert bool(payload["trusted_full_reflection_remapped"]) is True


def test_resolve_geometry_fit_correspondence_rejects_mismatched_local_table_signature() -> None:
    point, payload = opt._resolve_geometry_fit_correspondence(
        {
            "frozen_locator_kind": "local_row",
            "frozen_table_namespace": "current_full_local",
            "frozen_table_index": 0,
            "frozen_row_index": 1,
            "frozen_table_signature": "seed-signature",
        },
        hit_tables=[
            np.asarray(
                [
                    [1.0, 2.0, 2.0, -22.0, 1.0, 0.0, 0.0],
                    [1.0, 8.0, 8.0, 22.0, 1.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        ],
        max_positions=np.asarray([[1.0, 2.0, 2.0, 1.0, 8.0, 8.0]], dtype=np.float64),
        current_local_table_signature="current-signature",
    )

    assert point is None
    assert payload["resolution_reason"] == "frozen_table_signature_mismatch"


def test_resolve_geometry_fit_correspondence_rejects_trusted_full_reflection_index_missing_from_subset() -> (
    None
):
    point, payload = opt._resolve_geometry_fit_correspondence(
        {
            "hkl": (1, 0, 0),
            "source_reflection_index": 7,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "frozen_locator_kind": "trusted_branch",
            "frozen_table_namespace": "full_reflection",
            "frozen_table_index": 7,
            "frozen_branch_index": 1,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
        },
        hit_tables=[
            np.asarray(
                [
                    [1.0, 2.0, 2.0, -22.0, 1.0, 0.0, 0.0],
                    [1.0, 8.0, 8.0, 22.0, 1.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        ],
        max_positions=np.asarray([[1.0, 2.0, 2.0, 1.0, 8.0, 8.0]], dtype=np.float64),
        trusted_full_reflection_local_index_map={5: 0},
    )

    assert point is None
    assert payload["resolution_reason"] == "trusted_full_reflection_index_not_in_subset"
    assert bool(payload["trusted_full_reflection_remapped"]) is False
    assert int(payload["frozen_table_index"]) == 7
    assert payload["frozen_table_namespace"] == "full_reflection"
    assert "resolved_table_index" not in payload


def test_fit_geometry_parameters_pixel_path_falls_back_from_stale_in_range_source_indices(
    monkeypatch,
):
    process_millers = []

    def fake_process(*args, **kwargs):
        miller_arg = np.asarray(args[0], dtype=np.float64)
        process_millers.append(miller_arg.copy())
        image_size = int(args[2])
        hit_tables = []
        coord_map = {
            (1, 0, 0): (1.0, 1.0),
            (2, 0, 0): (4.0, 4.0),
            (3, 0, 0): (7.0, 7.0),
        }
        best_sample_indices_out = kwargs.get("best_sample_indices_out")
        if isinstance(best_sample_indices_out, np.ndarray):
            best_sample_indices_out[:] = 0
        for row in miller_arg:
            hkl = tuple(int(round(v)) for v in row)
            col, row_px = coord_map[hkl]
            hit_tables.append(
                np.array(
                    [[1.0, col, row_px, 0.0, row[0], row[1], row[2]]],
                    dtype=np.float64,
                )
            )
        image = np.zeros((image_size, image_size), dtype=np.float64)
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 12
    miller = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 0,
            "source_row_index": 0,
        }
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={"solver": {"restarts": 0, "weighted_matching": False}},
    )

    assert result.success
    assert process_millers
    assert np.allclose(process_millers[0], np.array([[2.0, 0.0, 0.0]], dtype=np.float64))
    assert result.fun.size == 2
    assert np.allclose(result.fun, np.zeros(2, dtype=np.float64))
    assert isinstance(result.point_match_summary, dict)
    assert int(result.point_match_summary["fixed_source_resolved_count"]) == 0
    assert int(result.point_match_summary["fixed_source_reflection_count"]) == 0
    assert int(result.point_match_summary["fallback_entry_count"]) == 1
    assert int(result.point_match_summary["matched_pair_count"]) == 1
    assert isinstance(result.point_match_diagnostics, list)
    assert len(result.point_match_diagnostics) == 1
    assert result.point_match_diagnostics[0]["resolution_kind"] == "hkl_fallback"
    assert result.point_match_diagnostics[0]["source_table_index"] is None
    assert result.point_match_diagnostics[0].get("source_reflection_namespace") in (
        None,
        "",
    )
    assert result.point_match_diagnostics[0].get("source_reflection_is_full") in (
        None,
        False,
    )
    assert int(result.point_match_diagnostics[0]["resolved_table_index"]) == 0
    assert result.point_match_diagnostics[0]["match_status"] == "matched"


def test_fit_geometry_parameters_supports_center_component_variables(monkeypatch):
    target_row = 2.5
    target_col = 5.5
    centers_seen = []

    def fake_compute(*args, **kwargs):
        center_row = float(args[10])
        center_col = float(args[11])
        centers_seen.append((center_row, center_col))
        return np.array(
            [center_row - target_row, center_col - target_col],
            dtype=np.float64,
        )

    monkeypatch.setattr(opt, "compute_peak_position_error_geometry_local", fake_compute)

    image_size = 8
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=[],
        var_names=["center_x", "center_y"],
        experimental_image=None,
    )

    assert result.success
    assert abs(float(result.x[0]) - target_row) < 1e-8
    assert abs(float(result.x[1]) - target_col) < 1e-8
    assert any(
        abs(row - target_row) < 1e-3 and abs(col - target_col) < 1e-3 for row, col in centers_seen
    )


def test_build_measured_dict_skips_malformed_entries():
    measured = [
        {"label": "1,0,0", "x": 4.0, "y": 5.0},
        {"label": "bad-label", "x": 1.0, "y": 2.0},
        {"hkl": (2, 1, 0), "x": "6.0", "y": "7.0"},
        (3, 0, 0, 8.0, 9.0),
        {"label": "4,0,0", "x": np.nan, "y": 1.0},
        (1, 2),
    ]

    measured_dict = opt.build_measured_dict(measured)

    assert measured_dict == {
        (1, 0, 0): [(4.0, 5.0)],
        (2, 1, 0): [(6.0, 7.0)],
        (3, 0, 0): [(8.0, 9.0)],
    }


def test_build_measured_dict_prefers_detector_anchor_when_present():
    measured = [
        {
            "hkl": (1, 0, 0),
            "x": 40.0,
            "y": 50.0,
            "background_detector_x": 4.0,
            "background_detector_y": 5.0,
        },
        {
            "hkl": (2, 0, 0),
            "x": 60.0,
            "y": 70.0,
            "detector_x": 6.0,
            "detector_y": 7.0,
        },
    ]

    measured_dict = opt.build_measured_dict(measured)

    assert measured_dict == {
        (1, 0, 0): [(4.0, 5.0)],
        (2, 0, 0): [(6.0, 7.0)],
    }


def test_fit_geometry_parameters_tolerates_bad_measured_labels(monkeypatch):
    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", _fake_process_peaks)

    image_size = 8
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {"label": "not,a,peak", "x": 2.0, "y": 2.0},
        {"label": "1,0,0", "x": 4.0, "y": 4.0},
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
    )

    assert result.success


def test_simulate_and_compare_hkl_preserves_fixed_source_row_assignments(monkeypatch):
    monkeypatch.setattr(
        opt,
        "_process_peaks_parallel_safe",
        _fake_process_peaks_same_hkl_two_hits,
    )

    image_size = 12
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0",
            "x": 2.0,
            "y": 2.0,
            "overlay_match_index": 7,
            "source_table_index": 0,
            "source_row_index": 1,
            "fit_source_resolution_kind": "source_row",
        },
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0",
            "x": 8.0,
            "y": 8.0,
            "overlay_match_index": 3,
            "source_table_index": 0,
            "source_row_index": 0,
            "fit_source_resolution_kind": "source_row",
        },
    ]

    distances, sim_coords, meas_coords, *_ = opt.simulate_and_compare_hkl(
        miller,
        intensities,
        image_size,
        params,
        measured,
    )

    assert sim_coords == [(8.0, 8.0), (2.0, 2.0)]
    assert meas_coords == [(2.0, 2.0), (8.0, 8.0)]
    assert distances.size == 4
    assert np.max(distances) > 0.0


def test_fit_geometry_parameters_pixel_path_keeps_fixed_source_row_assignments(
    monkeypatch,
):
    monkeypatch.setattr(
        opt,
        "_process_peaks_parallel_safe",
        _fake_process_peaks_same_hkl_two_hits,
    )

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 12
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0",
            "x": 2.0,
            "y": 2.0,
            "overlay_match_index": 7,
            "source_table_index": 0,
            "source_row_index": 1,
            "fit_source_resolution_kind": "source_row",
        },
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0",
            "x": 8.0,
            "y": 8.0,
            "overlay_match_index": 3,
            "source_table_index": 0,
            "source_row_index": 0,
            "fit_source_resolution_kind": "source_row",
        },
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "weighted_matching": False,
            }
        },
    )

    assert result.success
    assert result.fun.size == 4
    assert np.max(np.abs(result.fun)) >= 5.0
    assert isinstance(result.point_match_summary, dict)
    assert int(result.point_match_summary["fixed_source_resolved_count"]) == 2
    assert isinstance(result.point_match_diagnostics, list)
    assert len(result.point_match_diagnostics) == 2
    assert all(
        entry["resolution_kind"] == "fixed_source" and entry["match_status"] == "matched"
        for entry in result.point_match_diagnostics
    )
    assert {
        str(entry["fit_source_resolution_kind"]) for entry in result.point_match_diagnostics
    } == {"source_row"}
    assert {int(entry["overlay_match_index"]) for entry in result.point_match_diagnostics} == {3, 7}


def test_fixed_correspondence_summary_reports_real_branch_mismatch(monkeypatch):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [
                    [1.0, 4.0, 4.0, -1.0, 1.0, 0.0, 0.0],
                    [1.0, 8.0, 8.0, 1.0, 1.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    image_size = 12
    local = _base_params(image_size, optics_mode=1)
    subset = opt.ReflectionSimulationSubset(
        miller=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        intensities=np.array([1.0], dtype=np.float64),
        measured_entries=[],
        original_indices=np.array([0], dtype=np.int64),
        total_reflection_count=1,
        fixed_source_reflection_count=1,
        fallback_hkl_count=0,
        reduced=False,
    )
    dataset_ctx = opt.GeometryFitDatasetContext(
        dataset_index=0,
        label="d0",
        theta_initial=0.0,
        subset=subset,
        experimental_image=np.zeros((image_size, image_size), dtype=np.float64),
    )
    base_correspondence = {
        "hkl": (1, 0, 0),
        "measured_x": 4.0,
        "measured_y": 4.0,
        "resolution_kind": "fixed_source",
        "frozen_locator_kind": "local_branch",
        "frozen_table_namespace": "current_full_local",
        "frozen_table_index": 0,
        "frozen_branch_index": 0,
    }

    _residual, diagnostics, summary = opt._evaluate_geometry_fit_dataset_fixed_correspondences(
        local,
        dataset_ctx,
        [dict(base_correspondence, source_branch_index=1)],
        image_size=image_size,
        weighted_matching=False,
        solver_f_scale=1.0,
        missing_pair_penalty=10.0,
        theta_value=0.0,
        collect_diagnostics=True,
    )

    assert summary["matched_pair_count"] == 1
    assert summary["branch_mismatch_count"] == 1
    assert diagnostics[0]["source_branch_index"] == 1
    assert diagnostics[0]["resolved_peak_index"] == 0
    assert diagnostics[0]["branch_mismatch"] is True

    row_correspondence = dict(
        base_correspondence,
        frozen_locator_kind="local_row",
        frozen_row_index=0,
    )
    _residual, diagnostics, summary = opt._evaluate_geometry_fit_dataset_fixed_correspondences(
        local,
        dataset_ctx,
        [dict(row_correspondence, source_branch_index=1)],
        image_size=image_size,
        weighted_matching=False,
        solver_f_scale=1.0,
        missing_pair_penalty=10.0,
        theta_value=0.0,
        collect_diagnostics=True,
    )

    assert summary["matched_pair_count"] == 1
    assert summary["branch_mismatch_count"] == 1
    assert diagnostics[0]["correspondence_resolution_reason"] == "resolved_source_row"
    assert diagnostics[0]["source_branch_index"] == 1
    assert diagnostics[0]["resolved_peak_index"] == 0
    assert diagnostics[0]["branch_mismatch"] is True

    _residual, diagnostics, summary = opt._evaluate_geometry_fit_dataset_fixed_correspondences(
        local,
        dataset_ctx,
        [dict(base_correspondence, source_branch_index=0)],
        image_size=image_size,
        weighted_matching=False,
        solver_f_scale=1.0,
        missing_pair_penalty=10.0,
        theta_value=0.0,
        collect_diagnostics=True,
    )

    assert summary["matched_pair_count"] == 1
    assert summary["branch_mismatch_count"] == 0
    assert diagnostics[0]["branch_mismatch"] is False


def test_fit_geometry_parameters_pixel_path_probes_out_of_flat_start_region(
    monkeypatch,
):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        gamma = float(args[8])
        sim_col = 6.0 if gamma >= 0.25 else 2.0
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, sim_col, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="flat-local-solver",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 12
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 6.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "weighted_matching": False,
                "stagnation_probe": True,
                "stagnation_probe_fraction": 0.5,
            }
        },
    )

    assert result.x.shape == (1,)
    assert float(result.x[0]) >= 0.25
    assert np.allclose(np.asarray(result.fun, dtype=float), [0.0, 0.0])
    assert any(
        str(entry.get("seed_kind", "")) in {"axis", "global"}
        for entry in getattr(result, "restart_history", [])
        if str(entry.get("message", "")) == "prescore"
    )


def test_fit_geometry_parameters_multistart_keeps_trusted_parameters_fixed(
    monkeypatch,
):
    solve_starts = []

    def fake_compute(*args, **kwargs):
        gamma = float(args[0])
        Gamma = float(args[1])
        return np.array([gamma - 0.2, Gamma - 0.8], dtype=np.float64)

    def fake_least_squares(residual_fn, x0, **kwargs):
        x0_arr = np.asarray(x0, dtype=float)
        solve_starts.append(x0_arr.copy())
        return opt.OptimizeResult(
            x=x0_arr,
            fun=np.asarray(residual_fn(x0_arr), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x0_arr, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "compute_peak_position_error_geometry_local", fake_compute)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=[[1.0, 0.0, 0.0, 4.0, 4.0]],
        var_names=["gamma", "Gamma"],
        experimental_image=None,
        refinement_config={
            "bounds": {
                "gamma": {"mode": "absolute", "min": 0.0, "max": 1.0},
                "Gamma": {"mode": "absolute", "min": 0.0, "max": 1.0},
            },
            "priors": {"gamma": {"sigma": 0.05}},
            "seed_search": {
                "prescore_top_k": 6,
                "n_global": 4,
                "n_jitter": 2,
                "jitter_sigma_u": 0.5,
                "min_seed_separation_u": 0.1,
                "trusted_prior_fraction_of_span": 0.15,
            },
        },
    )

    assert result.success
    assert solve_starts
    assert all(np.isclose(float(start[0]), 0.0) for start in solve_starts)
    assert any(abs(float(start[1])) > 1.0e-9 for start in solve_starts)
    param_entries = result.geometry_fit_debug_summary["parameter_entries"]
    assert str(param_entries[0]["seed_group"]) == "trusted"
    assert str(param_entries[1]["seed_group"]) == "uncertain"


def test_fit_geometry_parameters_pixel_path_broad_restart_seed_escapes_far_coupled_minimum(
    monkeypatch,
):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        gamma = float(args[8])
        Gamma = float(args[9])
        sim_col = 6.0 if gamma >= 0.75 and Gamma >= 0.75 else 2.0
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, sim_col, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="flat-local-solver",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 12
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 6.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma", "Gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "bounds": {
                "gamma": {"mode": "absolute", "min": 0.0, "max": 1.0},
                "Gamma": {"mode": "absolute", "min": 0.0, "max": 1.0},
            },
            "solver": {
                "restarts": 4,
                "restart_jitter": 0.15,
                "weighted_matching": False,
                "stagnation_probe": False,
            },
        },
    )

    assert result.x.shape == (2,)
    assert float(result.x[0]) >= 0.75
    assert float(result.x[1]) >= 0.75
    assert np.allclose(np.asarray(result.fun, dtype=float), [0.0, 0.0])
    assert any(
        str(entry.get("seed_kind", "")) == "global"
        and str(entry.get("message", "")) == "prescore"
        and float(entry.get("cost", np.inf)) <= 1.0e-9
        for entry in getattr(result, "restart_history", [])
    )


def _seed_multistart_fixed_pair_measured_entry(
    *,
    hkl=(1, 0, 0),
    pair_id="fixed-provider-pair-0",
    source_table_index=0,
    source_peak_index=0,
    source_branch_index=0,
    x=5.0,
    y=4.0,
) -> dict:
    return {
        "hkl": tuple(hkl),
        "label": f"{int(hkl[0])},{int(hkl[1])},{int(hkl[2])}",
        "x": float(x),
        "y": float(y),
        "background_detector_x": float(x),
        "background_detector_y": float(y),
        "pair_id": str(pair_id),
        "source_table_index": int(source_table_index),
        "resolved_table_index": int(source_table_index),
        "source_row_index": 0,
        "source_branch_index": int(source_branch_index),
        "source_peak_index": int(source_peak_index),
        "resolved_peak_index": int(source_peak_index),
        "fit_source_resolution_kind": "provider_fixed_source_local",
        "optimizer_request_source": "provider_pair",
        "optimizer_request_has_fixed_source": True,
        "optimizer_request_fallback_row": False,
        "provider_selected_source_identity_canonical": {
            "normalized_hkl": [int(hkl[0]), int(hkl[1]), int(hkl[2])],
            "source_table_index": int(source_table_index),
            "source_peak_index": int(source_peak_index),
        },
    }


def _run_seed_multistart_fixed_pair_case(
    monkeypatch,
    fake_process,
    *,
    measured_peaks=None,
    miller=None,
    intensities=None,
):
    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="flat-local-solver",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 12
    if miller is None:
        miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    else:
        miller = np.asarray(miller, dtype=np.float64)
    if intensities is None:
        intensities = np.ones(int(miller.shape[0]), dtype=np.float64)
    else:
        intensities = np.asarray(intensities, dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)
    if measured_peaks is None:
        measured_peaks = [_seed_multistart_fixed_pair_measured_entry()]

    return opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured_peaks,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "bounds": {
                "gamma": {"mode": "absolute", "min": 0.0, "max": 1.0},
            },
            "solver": {
                "manual_point_fit_mode": True,
                "weighted_matching": False,
                "missing_pair_penalty_px": 0.0,
                "missing_pair_penalty_deg": 0.0,
                "q_group_line_constraints": False,
                "use_measurement_uncertainty": False,
            },
            "seed_search": {
                "prescore_top_k": 6,
                "n_global": 4,
                "n_jitter": 0,
                "min_seed_separation_u": 0.1,
            },
            "full_beam_polish": {"enabled": False},
            "identifiability": {"enabled": False},
        },
    )


def _seed_multistart_clean_when_gamma_small(*args, **kwargs):
    image_size = int(args[2])
    gamma = float(args[8])
    image = np.zeros((image_size, image_size), dtype=np.float64)
    if gamma <= 0.25:
        hit_tables = [
            np.array(
                [[1.0, 6.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
    else:
        hit_tables = [np.empty((0, 7), dtype=np.float64)]
    return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []


def _seed_multistart_only_first_fixed_pair_available(*args, **kwargs):
    image_size = int(args[2])
    image = np.zeros((image_size, image_size), dtype=np.float64)
    hit_tables = [
        np.array(
            [[1.0, 6.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
    ]
    return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []


def test_seed_multistart_rejects_seed_that_loses_fixed_pairs(monkeypatch):
    result = _run_seed_multistart_fixed_pair_case(
        monkeypatch,
        _seed_multistart_clean_when_gamma_small,
    )

    trace = result.seed_multistart_trace
    rejected = [
        record
        for record in trace["seed_records"]
        if record.get("rejection_reason") == "fixed_source_or_pair_integrity_lost"
    ]
    assert rejected
    assert all(record["lost_pair_ids"] for record in rejected)
    assert int(trace["seeds_rejected_for_pair_integrity"]) >= 1


def test_seed_multistart_expected_count_does_not_shrink_when_reference_missing(monkeypatch):
    measured_peaks = [
        _seed_multistart_fixed_pair_measured_entry(
            hkl=(1, 0, 0),
            pair_id="fixed-provider-pair-0",
            source_table_index=0,
            x=5.0,
            y=4.0,
        ),
        _seed_multistart_fixed_pair_measured_entry(
            hkl=(2, 0, 0),
            pair_id="fixed-provider-pair-1",
            source_table_index=1,
            x=7.0,
            y=4.0,
        ),
    ]

    result = _run_seed_multistart_fixed_pair_case(
        monkeypatch,
        _seed_multistart_only_first_fixed_pair_available,
        measured_peaks=measured_peaks,
        miller=np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64),
    )

    trace = result.seed_multistart_trace
    assert not result.success
    assert int(trace["fixed_source_pair_count"]) == 2
    assert len(trace["fixed_manual_pair_invariants"]) == 2
    assert str(trace["failure_reason"]) == "seed_multistart_incompatible_with_fixed_manual_pairs"
    assert any(
        "fixed-provider-pair-1" in record.get("missing_pair_ids", [])
        or "fixed-provider-pair-1" in record.get("lost_pair_ids", [])
        for record in trace["seed_records"]
    )


def test_seed_multistart_dirty_generation_is_not_overwritten_by_clean_prescore(monkeypatch):
    calls = {"count": 0}

    def fake_process(*args, **kwargs):
        calls["count"] += 1
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        if calls["count"] == 2:
            hit_tables = [np.empty((0, 7), dtype=np.float64)]
        else:
            hit_tables = [
                np.array(
                    [[1.0, 6.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                    dtype=np.float64,
                )
            ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    result = _run_seed_multistart_fixed_pair_case(monkeypatch, fake_process)

    trace = result.seed_multistart_trace
    dirty_then_clean = None
    for record in trace["seed_records"]:
        stages = {str(stage["stage"]): stage for stage in record.get("stages", [])}
        if (
            stages.get("generation", {}).get("clean") is False
            and stages.get("prescore", {}).get("clean") is True
        ):
            dirty_then_clean = record
            break
    assert dirty_then_clean is not None
    assert bool(dirty_then_clean["clean"]) is False
    assert str(dirty_then_clean["rejection_stage"]) == "generation"
    assert dirty_then_clean["lost_pair_ids"]
    final_stage = [
        stage
        for stage in dirty_then_clean["stages"]
        if str(stage.get("stage")) == "final_selection"
    ][-1]
    assert bool(final_stage["clean"]) is False
    assert final_stage["lost_pair_ids"]


def test_seed_multistart_rejects_seed_when_fixed_pair_frame_changes(monkeypatch):
    original_identity_payload = opt._dynamic_reanchor_identity_payload
    matched_payload_calls = {"count": 0}

    def frame_shifting_identity_payload(entry, pair_index):
        payload = original_identity_payload(entry, pair_index)
        if (
            str(payload.get("pair_id")) == "fixed-provider-pair-0"
            and str(entry.get("match_status", "")).lower() == "matched"
        ):
            matched_payload_calls["count"] += 1
            payload["frame"] = 0 if matched_payload_calls["count"] == 1 else 1
        return payload

    monkeypatch.setattr(
        opt,
        "_dynamic_reanchor_identity_payload",
        frame_shifting_identity_payload,
    )

    result = _run_seed_multistart_fixed_pair_case(
        monkeypatch,
        _seed_multistart_clean_when_gamma_small,
    )

    trace = result.seed_multistart_trace
    rejected = [
        record
        for record in trace["seed_records"]
        if "fixed-provider-pair-0" in record.get("rematched_pair_ids", [])
    ]
    assert rejected
    assert all(
        record.get("rejection_reason") == "fixed_source_or_pair_integrity_lost"
        for record in rejected
    )


def test_seed_multistart_selects_only_seed_with_clean_fixed_pairs(monkeypatch):
    result = _run_seed_multistart_fixed_pair_case(
        monkeypatch,
        _seed_multistart_clean_when_gamma_small,
    )

    trace = result.seed_multistart_trace
    invariant = trace["fixed_manual_pair_invariants"][0]
    assert invariant["hkl"] == [1, 0, 0]
    assert invariant["source_branch_index"] == 0
    assert "branch_group_key" in invariant
    assert "frame" in invariant
    selected_index = int(trace["selected_seed_index"])
    selected_record = next(
        record for record in trace["seed_records"] if int(record["seed_index"]) == selected_index
    )
    dirty_costs = [
        float(record["cost"])
        for record in trace["seed_records"]
        if record.get("rejection_reason") == "fixed_source_or_pair_integrity_lost"
        and np.isfinite(float(record["cost"]))
    ]
    assert dirty_costs
    assert min(dirty_costs) < float(trace["selected_seed_cost"])
    assert bool(selected_record["clean"]) is True
    assert bool(trace["selected_seed_clean"]) is True
    assert int(trace["selected_seed_fixed_source_resolved_count"]) == 1
    assert int(trace["selected_seed_matched_pair_count"]) == 1


def test_seed_multistart_fails_when_all_seeds_dirty(monkeypatch):
    calls = {"count": 0}

    def fake_process(*args, **kwargs):
        calls["count"] += 1
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        if calls["count"] == 1:
            hit_tables = [
                np.array(
                    [[1.0, 6.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                    dtype=np.float64,
                )
            ]
        else:
            hit_tables = [np.empty((0, 7), dtype=np.float64)]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    result = _run_seed_multistart_fixed_pair_case(monkeypatch, fake_process)

    trace = result.seed_multistart_trace
    assert not result.success
    assert str(trace["failure_reason"]) == "seed_multistart_incompatible_with_fixed_manual_pairs"
    assert bool(trace["selected_seed_clean"]) is False
    assert int(trace["seeds_solved"]) == 0
    assert all(
        record.get("rejection_reason") == "fixed_source_or_pair_integrity_lost"
        for record in trace["seed_records"]
    )


def test_seed_multistart_trace_reports_per_seed_pair_counters(monkeypatch):
    result = _run_seed_multistart_fixed_pair_case(
        monkeypatch,
        _seed_multistart_clean_when_gamma_small,
    )

    trace = result.seed_multistart_trace
    selected_index = int(trace["selected_seed_index"])
    selected_record = next(
        record for record in trace["seed_records"] if int(record["seed_index"]) == selected_index
    )
    stages = {str(stage["stage"]) for stage in selected_record["stages"]}
    assert {"generation", "prescore", "solve_start", "solve_end", "final_selection"} <= stages
    for key in (
        "fixed_source_resolved_count",
        "matched_pair_count",
        "fallback_entry_count",
        "missing_pair_count",
        "branch_mismatch_count",
        "provider_to_optimizer_identity_match",
        "provider_to_optimizer_point_match",
    ):
        assert key in selected_record
    assert "lost_pair_ids_by_seed" in trace
    assert "fallback_pair_ids_by_seed" in trace
    assert "rematched_pair_ids_by_seed" in trace
    assert "missing_pair_ids_by_seed" in trace
    assert isinstance(result.point_match_summary["seed_multistart_trace"], dict)


def test_fit_geometry_parameters_records_prescore_and_local_seed_history(monkeypatch):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        gamma = float(args[8])
        Gamma = float(args[9])
        sim_col = 6.0 if gamma >= 0.75 and Gamma >= 0.75 else 2.0
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, sim_col, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="flat-local-solver",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 12
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 6.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma", "Gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "bounds": {
                "gamma": {"mode": "absolute", "min": 0.0, "max": 1.0},
                "Gamma": {"mode": "absolute", "min": 0.0, "max": 1.0},
            },
            "solver": {
                "restarts": 4,
                "restart_jitter": 0.15,
                "weighted_matching": False,
                "stagnation_probe": False,
                "workers": 2,
                "parallel_mode": "restarts",
                "worker_numba_threads": 2,
            },
        },
    )

    assert result.x.shape == (2,)
    assert float(result.x[0]) >= 0.75
    assert float(result.x[1]) >= 0.75
    assert any(
        str(entry.get("message", "")) == "prescore"
        for entry in getattr(result, "restart_history", [])
    )
    assert any(
        str(entry.get("message", "")) == "flat-local-solver"
        for entry in getattr(result, "restart_history", [])
    )


def test_fit_geometry_parameters_reports_unweighted_peak_rms(monkeypatch):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[50.0, 14.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([25.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "weighted_matching": True,
                "f_scale_px": 1.0,
            }
        },
    )

    assert result.success
    raw_distance = float(result.point_match_diagnostics[0]["distance_px"])
    assert np.isclose(float(result.point_match_summary["matched_pair_count"]), 1.0)
    assert np.isclose(float(result.point_match_summary["unweighted_peak_rms_px"]), raw_distance)
    assert np.isclose(float(result.rms_px), raw_distance)
    assert str(result.point_match_summary["peak_weighting_mode"]) == "uniform"
    assert np.isfinite(float(result.weighted_residual_rms_px))
    assert float(result.rms_px) > float(result.weighted_residual_rms_px)


def test_fit_geometry_parameters_seed_status_reports_missing_pair_counts(monkeypatch):
    status_messages = []

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        return image, [], np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([25.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={"solver": {"restarts": 0, "stagnation_probe": False}},
        status_callback=status_messages.append,
    )

    assert result.success
    assert any("running normalized-u multistart solve" in msg for msg in status_messages)
    assert any("identity seed" in msg and "cost=" in msg for msg in status_messages)
    assert str(result.geometry_fit_debug_summary["main_solve_seed"]["seed_kind"]) == "u=0"
    assert int(result.point_match_summary["missing_pair_count"]) == 1


def test_fit_geometry_parameters_records_bound_proximity_summary(monkeypatch):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[50.0, 14.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(kwargs["bounds"][1], dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([25.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "bounds": {"gamma": {"mode": "absolute", "min": 0.0, "max": 1.0}},
            "solver": {"restarts": 0, "stagnation_probe": False},
        },
    )

    assert result.success
    assert result.bound_hits == ["gamma"]
    assert result.boundary_warning == (
        "Possible identifiability issue: parameters finished near bounds (gamma=upper)."
    )
    assert result.bound_proximity_summary == {
        "threshold_fraction": 0.01,
        "near_bound_parameters": [
            {
                "name": "gamma",
                "side": "upper",
                "value": 1.0,
                "bound": 1.0,
                "gap": 0.0,
                "span": 1.0,
                "gap_fraction": 0.0,
            }
        ],
    }


def test_full_beam_polish_rejects_match_count_regression(monkeypatch):
    solve_calls = []

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        gamma = float(args[8])
        beam_x_array = np.asarray(args[16], dtype=np.float64)
        image = np.zeros((image_size, image_size), dtype=np.float64)

        if beam_x_array.size > 1 and gamma >= 0.5:
            hit_tables = [
                np.array(
                    [[10.0, 4.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
                np.empty((0, 7), dtype=np.float64),
            ]
        else:
            hit_tables = [
                np.array(
                    [[10.0, 5.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
                np.array(
                    [[10.0, 9.0, 8.0, 0.0, 0.0, 1.0, 0.0]],
                    dtype=np.float64,
                ),
            ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        call_index = len(solve_calls)
        x = np.array([0.0], dtype=float) if call_index == 0 else np.array([1.0], dtype=float)
        solve_calls.append(x.copy())
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    intensities = np.array([25.0, 20.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["mosaic_params"] = {
        "beam_x_array": np.array([0.0, 0.1], dtype=np.float64),
        "beam_y_array": np.zeros(2, dtype=np.float64),
        "theta_array": np.zeros(2, dtype=np.float64),
        "phi_array": np.zeros(2, dtype=np.float64),
        "sigma_mosaic_deg": 0.2,
        "gamma_mosaic_deg": 0.1,
        "eta": 0.05,
        "wavelength_array": np.ones(2, dtype=np.float64),
    }
    measured = [
        {
            "label": "1,0,0",
            "hkl": (1, 0, 0),
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 0,
            "source_row_index": 0,
            "sigma_px": 1.0,
        },
        {
            "label": "0,1,0",
            "hkl": (0, 1, 0),
            "x": 8.0,
            "y": 8.0,
            "source_table_index": 1,
            "source_row_index": 0,
            "sigma_px": 1.0,
        },
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "loss": "linear",
                "weighted_matching": False,
                "use_measurement_uncertainty": True,
                "stagnation_probe": False,
            },
            "full_beam_polish": {"enabled": True, "max_nfev": 10},
            "identifiability": {"enabled": False},
        },
    )

    assert result.success
    assert len(solve_calls) >= 2
    assert np.allclose(np.asarray(result.x, dtype=float), np.array([0.0]))
    assert isinstance(result.full_beam_polish_summary, dict)
    assert bool(result.full_beam_polish_summary["accepted"]) is False
    assert str(result.full_beam_polish_summary["status"]) == "no_op_optimum"
    assert str(result.full_beam_polish_summary["selection_status"]) == "no_op_optimum"
    assert bool(result.full_beam_polish_summary["fit_quality_passed"]) is True
    assert (
        str(result.full_beam_polish_summary["selected_candidate_name"]) != "full_beam_polish_result"
    )
    assert str(result.full_beam_polish_summary["selected_candidate_name"]) == str(
        result.full_beam_polish_summary["best_valid_raw_detector_candidate_name"]
    )
    assert str(result.full_beam_polish_summary["selected_candidate_source"]) == str(
        result.full_beam_polish_summary["best_valid_raw_detector_candidate_source"]
    )
    assert (
        str(result.full_beam_polish_summary["reason"])
        == "no_valid_candidate_improved_raw_detector_alignment"
    )
    assert np.isfinite(float(result.full_beam_polish_summary["candidate_cost"]))
    assert np.isfinite(float(result.full_beam_polish_summary["start_cost"]))
    assert int(result.full_beam_polish_summary["matched_pair_count_before"]) == 2
    assert int(result.full_beam_polish_summary["candidate_matched_pair_count"]) == 1
    assert int(result.point_match_summary["matched_pair_count"]) == 2


def _run_manual_seven_pair_full_beam_polish_case(
    monkeypatch,
    *,
    drop_after_polish: bool,
    manual_fixed_source: bool = True,
    manual_marker_style: str = "provider_identity",
):
    solve_calls = []

    def fake_process(*args, **kwargs):
        miller_arg = np.asarray(args[0], dtype=np.float64)
        image_size = int(args[2])
        gamma = float(args[8])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = []
        for idx in range(len(miller_arg)):
            if drop_after_polish and gamma >= 0.5 and idx >= 5:
                hit_tables.append(np.empty((0, 7), dtype=np.float64))
                continue
            measured_x = 4.0 + float(idx) * 2.0
            measured_y = 4.0 + float(idx)
            offset = 5.0 if gamma < 0.5 else 0.0
            hit_tables.append(
                np.array(
                    [
                        [
                            10.0,
                            measured_x + offset,
                            measured_y,
                            0.0,
                            float(idx + 1),
                            0.0,
                            0.0,
                        ]
                    ],
                    dtype=np.float64,
                )
            )
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        call_index = len(solve_calls)
        x = np.array([0.0], dtype=float) if call_index == 0 else np.array([1.0], dtype=float)
        solve_calls.append(x.copy())
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 32
    miller = np.array([[float(idx + 1), 0.0, 0.0] for idx in range(7)], dtype=np.float64)
    intensities = np.linspace(25.0, 19.0, 7, dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["mosaic_params"] = {
        "beam_x_array": np.array([0.0, 0.1], dtype=np.float64),
        "beam_y_array": np.zeros(2, dtype=np.float64),
        "theta_array": np.zeros(2, dtype=np.float64),
        "phi_array": np.zeros(2, dtype=np.float64),
        "sigma_mosaic_deg": 0.2,
        "gamma_mosaic_deg": 0.1,
        "eta": 0.05,
        "wavelength_array": np.ones(2, dtype=np.float64),
    }
    measured = []
    for idx in range(7):
        entry = {
            "label": f"{idx + 1},0,0",
            "pair_id": f"pair-{idx}",
            "hkl": (idx + 1, 0, 0),
            "x": 4.0 + float(idx) * 2.0,
            "y": 4.0 + float(idx),
            "source_table_index": idx,
            "source_row_index": 0,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_index": idx,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "frozen_locator_kind": "trusted_row",
            "frozen_table_namespace": "full_reflection",
            "frozen_table_index": idx,
            "frozen_row_index": 0,
            "resolved_table_index": idx,
            "resolved_peak_index": 0,
            "resolution_kind": "fixed_source",
            "q_group_key": ("q", idx),
            "branch_group_key": ("branch", idx),
            "sigma_px": 1.0,
        }
        if manual_fixed_source:
            identity = {
                "pair_id": f"pair-{idx}",
                "hkl": [idx + 1, 0, 0],
                "source_table_index": idx,
                "source_row_index": 0,
                "source_branch_index": 0,
                "source_peak_index": 0,
            }
            if manual_marker_style == "optimizer_request_only":
                entry.update(
                    {
                        "optimizer_request_source": "provider_pair",
                        "optimizer_request_has_fixed_source": True,
                        "optimizer_request_fallback_row": False,
                    }
                )
            elif manual_marker_style == "manual_identity_only":
                entry.update(
                    {
                        "optimizer_request_fallback_row": False,
                        "manual_picker_selected_source_identity_canonical": dict(identity),
                    }
                )
            else:
                entry.update(
                    {
                        "fit_source_resolution_kind": "provider_fixed_source_local",
                        "optimizer_request_source": "provider_pair",
                        "optimizer_request_has_fixed_source": True,
                        "optimizer_request_fallback_row": False,
                        "provider_selected_source_identity_canonical": dict(identity),
                        "manual_picker_selected_source_identity_canonical": dict(identity),
                        "selected_source_identity_canonical": dict(identity),
                    }
                )
        measured.append(entry)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=np.zeros((image_size, image_size), dtype=np.float64),
        refinement_config={
            "solver": {
                "restarts": 0,
                "loss": "linear",
                "weighted_matching": False,
                "use_measurement_uncertainty": True,
                "stagnation_probe": False,
            },
            "full_beam_polish": {"enabled": True, "max_nfev": 10},
            "identifiability": {"enabled": False},
        },
    )
    return result, solve_calls


def test_full_beam_polish_manual_fixed_source_clean_seven_pairs_pass(monkeypatch):
    result, solve_calls = _run_manual_seven_pair_full_beam_polish_case(
        monkeypatch,
        drop_after_polish=False,
    )

    summary = dict(result.full_beam_polish_summary)
    assert result.success
    assert len(solve_calls) >= 2
    assert np.allclose(np.asarray(result.x, dtype=float), np.array([1.0]))
    assert bool(summary["accepted"]) is True
    assert bool(summary["fit_quality_passed"]) is True
    assert bool(summary["manual_fixed_source_mode"]) is True
    assert int(summary["polish_manual_fixed_source_pair_count_before"]) == 7
    assert int(summary["polish_fixed_source_resolved_count_before"]) == 7
    assert int(summary["polish_fixed_source_resolved_count_after"]) == 7
    assert int(summary["polish_matched_pair_count_before"]) == 7
    assert int(summary["polish_matched_pair_count_after"]) == 7
    assert int(summary["polish_missing_pair_count_after"]) == 0
    assert int(summary["polish_fallback_entry_count_after"]) == 0
    assert int(summary["polish_branch_mismatch_count_after"]) == 0
    assert summary["polish_lost_pair_ids"] == []


def test_full_beam_polish_manual_fixed_source_missing_pairs_blocks_even_if_metrics_improve(
    monkeypatch,
):
    result, solve_calls = _run_manual_seven_pair_full_beam_polish_case(
        monkeypatch,
        drop_after_polish=True,
    )

    summary = dict(result.full_beam_polish_summary)
    assert result.success
    assert len(solve_calls) >= 2
    assert np.allclose(np.asarray(result.x, dtype=float), np.array([0.0]))
    assert bool(summary["accepted"]) is False
    assert bool(summary["fit_quality_passed"]) is False
    assert bool(summary["manual_fixed_source_mode"]) is True
    assert summary["selected_candidate_name"] is None
    assert summary["selected_candidate_source"] is None
    assert str(summary["selection_status"]) == "blocked_manual_fixed_pairs"
    assert str(summary["failure_reason"]) == (
        "full_beam_polish_incompatible_with_fixed_manual_pairs"
    )
    assert str(summary["diagnosis_classification"]) == "fixed_source_or_pair_integrity_lost"
    assert bool(summary["polish_started"]) is True
    assert bool(summary["polish_completed"]) is True
    assert int(summary["polish_fixed_source_resolved_count_before"]) == 7
    assert int(summary["polish_fixed_source_resolved_count_after"]) == 7
    assert int(summary["polish_matched_pair_count_before"]) == 7
    assert int(summary["polish_matched_pair_count_after"]) == 5
    assert int(summary["polish_missing_pair_count_after"]) == 2
    assert int(summary["polish_fallback_entry_count_after"]) == 0
    assert int(summary["polish_branch_mismatch_count_after"]) == 0
    assert float(summary["candidate_unweighted_peak_rms_px"]) < float(
        summary["start_unweighted_peak_rms_px"]
    )
    assert summary["polish_lost_pair_ids"] == ["pair-5", "pair-6"]
    assert summary["polish_missing_pair_ids"] == ["pair-5", "pair-6"]
    assert summary["polish_fallback_pair_ids"] == []
    assert summary["polish_rematched_pair_ids"] == []
    details = list(summary["polish_lost_pair_details"])
    assert [detail["pair_id"] for detail in details] == ["pair-5", "pair-6"]
    for detail in details:
        for key in (
            "pair_id",
            "hkl",
            "source_branch_index",
            "q_group_key",
            "branch_group_key",
            "provider_identity",
            "pre_polish_simulated_point",
            "post_polish_simulated_point",
            "missing_reason",
        ):
            assert key in detail
    ledger = list(summary["candidate_ledger"])
    polish_entries = [
        entry for entry in ledger if str(entry.get("candidate_name")) == "full_beam_polish_result"
    ]
    assert polish_entries
    assert all(entry.get("selected") is False for entry in polish_entries)
    assert all(entry.get("selected") is False for entry in ledger)


def test_full_beam_polish_count_only_seven_pairs_does_not_trigger_manual_block(
    monkeypatch,
):
    result, solve_calls = _run_manual_seven_pair_full_beam_polish_case(
        monkeypatch,
        drop_after_polish=True,
        manual_fixed_source=False,
    )

    summary = dict(result.full_beam_polish_summary)
    assert result.success
    assert len(solve_calls) >= 2
    assert bool(summary["manual_fixed_source_mode"]) is False
    assert int(summary["polish_manual_fixed_source_pair_count_before"]) == 0
    assert str(summary.get("failure_reason") or "") != (
        "full_beam_polish_incompatible_with_fixed_manual_pairs"
    )
    assert str(summary.get("diagnosis_classification") or "") != (
        "fixed_source_or_pair_integrity_lost"
    )
    assert bool(summary["fit_quality_passed"]) is True
    assert str(summary["selection_status"]) == "no_op_optimum"


def test_full_beam_polish_manual_request_markers_survive_full_beam_correspondence_copy(
    monkeypatch,
):
    result, _solve_calls = _run_manual_seven_pair_full_beam_polish_case(
        monkeypatch,
        drop_after_polish=True,
        manual_marker_style="optimizer_request_only",
    )

    summary = dict(result.full_beam_polish_summary)
    assert bool(summary["manual_fixed_source_mode"]) is True
    assert int(summary["polish_manual_fixed_source_pair_count_before"]) == 7
    assert bool(summary["fit_quality_passed"]) is False
    assert summary["selected_candidate_name"] is None
    assert str(summary["failure_reason"]) == (
        "full_beam_polish_incompatible_with_fixed_manual_pairs"
    )
    assert summary["polish_lost_pair_ids"] == ["pair-5", "pair-6"]


def test_full_beam_polish_selects_best_valid_raw_candidate_when_fixed_pairs_hold():
    dynamic_point_result = {
        "candidate_name": "dynamic_point_result",
        "x_vector_source": "current_result.x",
        "valid_raw_detector_candidate": True,
        "matched_fixed_pair_count": 7,
        "missing_fixed_pair_count": 0,
        "branch_mismatch_count": 0,
        "rms_px": 88.7671,
        "max_px": 186.0813,
        "outside_radius_count": 2,
        "weighted_objective": 1.0,
    }
    full_beam_polish_result = {
        "candidate_name": "full_beam_polish_result",
        "x_vector_source": "full_beam_polish",
        "valid_raw_detector_candidate": True,
        "matched_fixed_pair_count": 7,
        "missing_fixed_pair_count": 0,
        "branch_mismatch_count": 0,
        "rms_px": 4.8773,
        "max_px": 8.3621,
        "outside_radius_count": 0,
        "weighted_objective": 50.0,
    }
    rejected_start = {
        "candidate_name": "requested_start",
        "x_vector_source": "requested_x0",
        "valid_raw_detector_candidate": False,
        "matched_fixed_pair_count": 6,
        "missing_fixed_pair_count": 1,
        "branch_mismatch_count": 1,
        "rms_px": 4.5,
        "max_px": 8.0,
        "outside_radius_count": 0,
        "weighted_objective": 0.5,
    }

    ledger = [
        dynamic_point_result,
        full_beam_polish_result,
        rejected_start,
    ]

    best_valid_raw_candidate = opt._best_valid_raw_detector_candidate(ledger)

    assert best_valid_raw_candidate is full_beam_polish_result
    assert (
        opt._raw_detector_alignment_is_better(
            full_beam_polish_result,
            dynamic_point_result,
        )
        is True
    )
    assert (
        opt._should_promote_best_valid_full_beam_candidate(
            accepted=False,
            preserve_rejected_start=False,
            no_op_optimum=False,
            best_valid_raw_candidate=best_valid_raw_candidate,
        )
        is True
    )


def test_full_beam_polish_duplicate_fixed_picks_keep_full_beam_candidate_valid(
    monkeypatch,
) -> None:
    solve_calls = []

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        gamma = float(args[8])
        beam_x_array = np.asarray(args[16], dtype=np.float64)
        image = np.zeros((image_size, image_size), dtype=np.float64)

        if beam_x_array.size > 1 and gamma >= 0.5:
            hit_tables = [
                np.array(
                    [[10.0, 4.2, 4.0, 0.0, 1.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
                np.array(
                    [[10.0, 8.2, 8.0, 0.0, 0.0, 1.0, 0.0]],
                    dtype=np.float64,
                ),
                np.empty((0, 7), dtype=np.float64),
            ]
        else:
            hit_tables = [
                np.array(
                    [[10.0, 6.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
                np.array(
                    [[10.0, 10.0, 8.0, 0.0, 0.0, 1.0, 0.0]],
                    dtype=np.float64,
                ),
                np.array(
                    [[10.0, 4.4, 4.0, 0.0, 1.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
            ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        call_index = len(solve_calls)
        x = np.array([0.0], dtype=float) if call_index == 0 else np.array([1.0], dtype=float)
        solve_calls.append(x.copy())
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 24
    miller = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    intensities = np.array([25.0, 20.0, 15.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["mosaic_params"] = {
        "beam_x_array": np.array([0.0, 0.1], dtype=np.float64),
        "beam_y_array": np.zeros(2, dtype=np.float64),
        "theta_array": np.zeros(2, dtype=np.float64),
        "phi_array": np.zeros(2, dtype=np.float64),
        "sigma_mosaic_deg": 0.2,
        "gamma_mosaic_deg": 0.1,
        "eta": 0.05,
        "wavelength_array": np.ones(2, dtype=np.float64),
    }
    measured = [
        {
            "label": "1,0,0",
            "hkl": (1, 0, 0),
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 0,
            "source_row_index": 0,
            "sigma_px": 1.0,
        },
        {
            "label": "0,1,0",
            "hkl": (0, 1, 0),
            "x": 8.0,
            "y": 8.0,
            "source_table_index": 1,
            "source_row_index": 0,
            "sigma_px": 1.0,
        },
        {
            "label": "1,0,0-dup",
            "hkl": (1, 0, 0),
            "x": 4.5,
            "y": 4.0,
            "source_table_index": 0,
            "source_row_index": 0,
            "sigma_px": 1.0,
        },
    ]

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=np.zeros((image_size, image_size), dtype=np.float64),
        refinement_config={
            "solver": {
                "restarts": 0,
                "loss": "linear",
                "weighted_matching": False,
                "use_measurement_uncertainty": True,
                "stagnation_probe": False,
            },
            "full_beam_polish": {"enabled": True, "max_nfev": 10},
            "identifiability": {"enabled": False},
        },
    )

    assert result.success
    assert len(solve_calls) >= 2
    assert np.allclose(np.asarray(result.x, dtype=float), np.array([1.0]))
    summary = dict(result.full_beam_polish_summary)
    assert bool(summary["accepted"]) is True
    assert str(summary["selected_candidate_name"]) == "full_beam_polish_result"
    assert str(summary["best_valid_raw_detector_candidate_name"]) == "full_beam_polish_result"
    assert str(summary["best_valid_raw_detector_candidate_source"]) == "full_beam_polish"
    ledger = list(summary["candidate_ledger"])
    full_entry = next(
        entry for entry in ledger if entry["candidate_name"] == "full_beam_polish_result"
    )
    assert full_entry["valid_raw_detector_candidate"] is True
    assert full_entry["rejection_reason"] is None
    assert int(full_entry["matched_fixed_pair_count"]) == 2
    assert full_entry["selected"] is True
    assert int(summary["candidate_fixed_source_resolved_count"]) == 2
    assert int(summary["candidate_resolved_fixed_matched_pair_count"]) == 2


def test_full_beam_polish_rejects_objective_aligned_metric_regression(monkeypatch):
    solve_calls = []

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        gamma = float(args[8])
        beam_x_array = np.asarray(args[16], dtype=np.float64)
        image = np.zeros((image_size, image_size), dtype=np.float64)

        if beam_x_array.size > 1 and gamma >= 0.5:
            hit_tables = [
                np.array(
                    [[10.0, 4.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
                np.array(
                    [[10.0, 12.0, 8.0, 0.0, 0.0, 1.0, 0.0]],
                    dtype=np.float64,
                ),
            ]
        else:
            hit_tables = [
                np.array(
                    [[10.0, 5.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
                np.array(
                    [[10.0, 9.0, 8.0, 0.0, 0.0, 1.0, 0.0]],
                    dtype=np.float64,
                ),
            ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        call_index = len(solve_calls)
        x = np.array([0.0], dtype=float) if call_index == 0 else np.array([1.0], dtype=float)
        solve_calls.append(x.copy())
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    intensities = np.array([25.0, 20.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["mosaic_params"] = {
        "beam_x_array": np.array([0.0, 0.1], dtype=np.float64),
        "beam_y_array": np.zeros(2, dtype=np.float64),
        "theta_array": np.zeros(2, dtype=np.float64),
        "phi_array": np.zeros(2, dtype=np.float64),
        "sigma_mosaic_deg": 0.2,
        "gamma_mosaic_deg": 0.1,
        "eta": 0.05,
        "wavelength_array": np.ones(2, dtype=np.float64),
    }
    measured = [
        {
            "label": "1,0,0",
            "hkl": (1, 0, 0),
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 0,
            "source_row_index": 0,
            "sigma_px": 1.0,
        },
        {
            "label": "0,1,0",
            "hkl": (0, 1, 0),
            "x": 8.0,
            "y": 8.0,
            "source_table_index": 1,
            "source_row_index": 0,
            "sigma_px": 1.0,
        },
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "loss": "linear",
                "weighted_matching": False,
                "use_measurement_uncertainty": True,
                "stagnation_probe": False,
            },
            "full_beam_polish": {"enabled": True, "max_nfev": 10},
            "identifiability": {"enabled": False},
        },
    )

    assert result.success
    assert len(solve_calls) >= 2
    assert np.allclose(np.asarray(result.x, dtype=float), np.array([0.0]))
    assert isinstance(result.full_beam_polish_summary, dict)
    assert bool(result.full_beam_polish_summary["accepted"]) is False
    assert str(result.full_beam_polish_summary["status"]) == "no_op_optimum"
    assert str(result.full_beam_polish_summary["selection_status"]) == "no_op_optimum"
    assert bool(result.full_beam_polish_summary["fit_quality_passed"]) is True
    assert str(result.full_beam_polish_summary["reason"]) == (
        "no_valid_candidate_improved_raw_detector_alignment"
    )
    assert str(result.full_beam_polish_summary["selected_candidate_name"]) != (
        "full_beam_polish_result"
    )
    assert str(result.full_beam_polish_summary["selected_candidate_name"]) == str(
        result.full_beam_polish_summary["best_valid_raw_detector_candidate_name"]
    )
    assert str(result.full_beam_polish_summary["selected_candidate_source"]) == str(
        result.full_beam_polish_summary["best_valid_raw_detector_candidate_source"]
    )
    ledger = list(result.full_beam_polish_summary["candidate_ledger"])
    selected_entry = next(entry for entry in ledger if entry["selected"] is True)
    assert len(ledger) >= 2
    assert any(
        entry["candidate_name"] == selected_entry["candidate_name"]
        and entry["x_vector_source"] == selected_entry["x_vector_source"]
        and entry["selected"] is True
        and entry["valid_raw_detector_candidate"] is True
        for entry in ledger
    )
    assert any(
        entry["candidate_name"] == "full_beam_polish_result"
        and entry["valid_raw_detector_candidate"] is True
        and float(entry["rms_px"]) > 1.0
        for entry in ledger
    )
    assert float(result.full_beam_polish_summary["candidate_cost"]) > float(
        result.full_beam_polish_summary["start_cost"]
    )
    assert float(result.full_beam_polish_summary["candidate_point_match_cost"]) > float(
        result.full_beam_polish_summary["start_point_match_cost"]
    )
    assert float(result.full_beam_polish_summary["candidate_weighted_rms_px"]) > float(
        result.full_beam_polish_summary["start_weighted_rms_px"]
    )
    start_pm_diagnostics = result.full_beam_polish_summary["start_point_match_diagnostics"]
    candidate_pm_diagnostics = result.full_beam_polish_summary["candidate_point_match_diagnostics"]
    assert len(start_pm_diagnostics) == 2
    assert len(candidate_pm_diagnostics) == 2
    for entry in list(start_pm_diagnostics) + list(candidate_pm_diagnostics):
        assert "weighted_dx_px" in entry
        assert "weighted_dy_px" in entry
        assert "distance_weight" in entry
        assert "sigma_weight" in entry
        assert "priority_weight" in entry
        assert "weight" in entry
    assert (
        int(result.full_beam_polish_summary["start_point_match_summary"]["matched_pair_count"]) == 2
    )
    assert (
        int(result.full_beam_polish_summary["candidate_point_match_summary"]["matched_pair_count"])
        == 2
    )
    assert np.isclose(
        float(
            result.full_beam_polish_summary["start_point_match_summary"]["unweighted_peak_rms_px"]
        ),
        1.0,
    )
    assert np.isclose(
        float(
            result.full_beam_polish_summary["candidate_point_match_summary"][
                "unweighted_peak_rms_px"
            ]
        ),
        np.sqrt(8.0),
    )
    assert result.final_metric_name == "full_beam_fixed_correspondence"
    assert int(result.point_match_summary["matched_pair_count"]) == 2
    assert str(result.point_match_summary["metric_name"]) == "full_beam_fixed_correspondence"
    assert len(result.point_match_diagnostics) == 2
    for diag in result.point_match_diagnostics:
        assert diag["match_kind"] == "full_beam_fixed"
        assert diag["match_status"] == "matched"


def test_full_beam_polish_rejects_objective_aligned_improvement_when_raw_peak_metrics_regress(
    monkeypatch,
):
    solve_calls = []

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        gamma = float(args[8])
        beam_x_array = np.asarray(args[16], dtype=np.float64)
        image = np.zeros((image_size, image_size), dtype=np.float64)

        if beam_x_array.size > 1 and gamma >= 0.75:
            hit_tables = [
                np.array(
                    [[10.0, 4.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
                np.array(
                    [[10.0, 42.0, 8.0, 0.0, 0.0, 1.0, 0.0]],
                    dtype=np.float64,
                ),
            ]
        else:
            hit_tables = [
                np.array(
                    [[10.0, 5.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
                np.array(
                    [[10.0, 38.0, 8.0, 0.0, 0.0, 1.0, 0.0]],
                    dtype=np.float64,
                ),
            ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        call_index = len(solve_calls)
        x = np.array([0.0], dtype=float) if call_index == 0 else np.array([1.0], dtype=float)
        solve_calls.append(x.copy())
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 60
    miller = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    intensities = np.array([25.0, 20.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["mosaic_params"] = {
        "beam_x_array": np.array([0.0, 0.1], dtype=np.float64),
        "beam_y_array": np.zeros(2, dtype=np.float64),
        "theta_array": np.zeros(2, dtype=np.float64),
        "phi_array": np.zeros(2, dtype=np.float64),
        "sigma_mosaic_deg": 0.2,
        "gamma_mosaic_deg": 0.1,
        "eta": 0.05,
        "wavelength_array": np.ones(2, dtype=np.float64),
    }
    measured = [
        {
            "label": "1,0,0",
            "hkl": (1, 0, 0),
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 0,
            "source_row_index": 0,
            "sigma_px": 1.0,
        },
        {
            "label": "0,1,0",
            "hkl": (0, 1, 0),
            "x": 8.0,
            "y": 8.0,
            "source_table_index": 1,
            "source_row_index": 0,
            "source_reflection_index": 1,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "sigma_px": 1000.0,
        },
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "loss": "linear",
                "weighted_matching": False,
                "use_measurement_uncertainty": True,
                "stagnation_probe": False,
            },
            "full_beam_polish": {
                "enabled": True,
                "max_nfev": 10,
                "match_radius_px": 24.0,
            },
            "identifiability": {"enabled": False},
        },
    )

    assert result.success
    assert len(solve_calls) >= 2
    assert np.allclose(np.asarray(result.x, dtype=float), np.array([0.0]))
    assert isinstance(result.full_beam_polish_summary, dict)
    assert bool(result.full_beam_polish_summary["accepted"]) is False
    assert str(result.full_beam_polish_summary["status"]) == "no_op_optimum"
    assert str(result.full_beam_polish_summary["selection_status"]) == "no_op_optimum"
    assert bool(result.full_beam_polish_summary["fit_quality_passed"]) is True
    assert result.final_metric_name == "full_beam_fixed_correspondence"
    assert np.allclose(
        np.asarray(result.full_beam_polish_summary["x"], dtype=float),
        np.asarray(result.x, dtype=float),
    )
    assert np.allclose(
        np.asarray(result.full_beam_polish_summary["fun"], dtype=float),
        np.asarray(result.fun, dtype=float),
    )
    assert int(result.full_beam_polish_summary["matched_pair_count_before"]) == 2
    assert int(result.full_beam_polish_summary["matched_pair_count_after"]) == 2
    assert str(result.full_beam_polish_summary["acceptance_metric_scope"]) == (
        "all_resolved_fixed_correspondences"
    )
    assert str(result.full_beam_polish_summary["start_acceptance_metric_scope"]) == (
        "all_resolved_fixed_correspondences"
    )
    assert str(result.full_beam_polish_summary["candidate_acceptance_metric_scope"]) == (
        "all_resolved_fixed_correspondences"
    )
    assert float(result.full_beam_polish_summary["candidate_point_match_cost"]) < float(
        result.full_beam_polish_summary["start_point_match_cost"]
    )
    assert int(result.full_beam_polish_summary["start_match_radius_exceeded_count"]) == 1
    assert int(result.full_beam_polish_summary["candidate_match_radius_exceeded_count"]) == 1
    assert float(result.full_beam_polish_summary["candidate_weighted_rms_px"]) < float(
        result.full_beam_polish_summary["start_weighted_rms_px"]
    )
    assert str(result.full_beam_polish_summary["reason"]) == (
        "no_valid_candidate_improved_raw_detector_alignment"
    )
    assert str(result.full_beam_polish_summary["selected_candidate_name"]) == (
        "central_point_result"
    )
    assert str(result.full_beam_polish_summary["selected_candidate_source"]) == ("current_result.x")
    assert str(result.full_beam_polish_summary["start_vector_source"]) == "current_result.x"
    assert str(result.full_beam_polish_summary["selected_candidate_name"]) == str(
        result.full_beam_polish_summary["best_valid_raw_detector_candidate_name"]
    )
    assert str(result.full_beam_polish_summary["selected_candidate_source"]) == str(
        result.full_beam_polish_summary["best_valid_raw_detector_candidate_source"]
    )
    ledger = list(result.full_beam_polish_summary["candidate_ledger"])
    selected_entry = next(entry for entry in ledger if entry["selected"] is True)
    assert str(selected_entry["x_vector_source"]) == "current_result.x"
    assert str(selected_entry["x_vector_source"]) == str(
        result.full_beam_polish_summary["selected_candidate_source"]
    )
    assert any(
        entry["candidate_name"] == "full_beam_polish_result"
        and entry["valid_raw_detector_candidate"] is True
        and float(entry["weighted_objective"]) < float(selected_entry["weighted_objective"])
        for entry in ledger
    )
    assert float(result.full_beam_polish_summary["candidate_all_match_rms_px"]) > float(
        result.full_beam_polish_summary["start_all_match_rms_px"]
    )
    assert float(result.full_beam_polish_summary["candidate_all_match_peak_max_px"]) > float(
        result.full_beam_polish_summary["start_all_match_peak_max_px"]
    )
    start_diags = list(result.full_beam_polish_summary["start_point_match_diagnostics"])
    candidate_diags = list(result.full_beam_polish_summary["candidate_point_match_diagnostics"])
    start_identity = sorted(
        (
            tuple(entry["hkl"]),
            entry.get("source_reflection_index"),
            entry.get("source_branch_index"),
            entry.get("source_peak_index"),
            entry.get("resolved_table_index"),
            entry.get("resolved_peak_index"),
        )
        for entry in start_diags
    )
    candidate_identity = sorted(
        (
            tuple(entry["hkl"]),
            entry.get("source_reflection_index"),
            entry.get("source_branch_index"),
            entry.get("source_peak_index"),
            entry.get("resolved_table_index"),
            entry.get("resolved_peak_index"),
        )
        for entry in candidate_diags
    )
    assert start_identity == candidate_identity
    outlier_start = next(entry for entry in start_diags if entry["hkl"] == (0, 1, 0))
    outlier_candidate = next(entry for entry in candidate_diags if entry["hkl"] == (0, 1, 0))
    assert outlier_start["match_status"] == "matched"
    assert outlier_candidate["match_status"] == "matched"
    assert bool(outlier_start["match_radius_exceeded"]) is True
    assert bool(outlier_candidate["match_radius_exceeded"]) is True
    assert outlier_candidate["distance_px"] > outlier_start["distance_px"]


def test_full_beam_polish_no_op_selection_uses_seeded_start_when_winner_is_not_current_result(
    monkeypatch,
):
    solve_calls = []
    original_point_match_evaluator = opt._evaluate_geometry_fit_dataset_point_matches

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        gamma = float(args[8])
        image = np.zeros((image_size, image_size), dtype=np.float64)

        def _hit_tables(first_x: float, second_x: float):
            return [
                np.array(
                    [[10.0, first_x, 4.0, 0.0, 1.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
                np.array(
                    [[10.0, second_x, 8.0, 0.0, 0.0, 1.0, 0.0]],
                    dtype=np.float64,
                ),
            ]

        if abs(gamma - 0.0) < 1.0e-9:
            hit_tables = _hit_tables(4.0, 8.0)
        elif abs(gamma - 1.0) < 1.0e-9:
            hit_tables = _hit_tables(5.0, 10.0)
        elif abs(gamma - 2.0) < 1.0e-9:
            hit_tables = _hit_tables(4.0, 42.0)
        else:
            hit_tables = [
                np.empty((0, 7), dtype=np.float64),
                np.empty((0, 7), dtype=np.float64),
            ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        call_index = len(solve_calls)
        x = np.array([1.0], dtype=float) if call_index == 0 else np.array([2.0], dtype=float)
        solve_calls.append(x.copy())
        fun = (
            np.zeros(1, dtype=float) if call_index == 0 else np.asarray(residual_fn(x), dtype=float)
        )
        return opt.OptimizeResult(
            x=x,
            fun=fun,
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    def wrapped_point_match_evaluator(local, dataset_ctx, **kwargs):
        gamma = float(local.get("gamma", np.nan))
        if abs(gamma - 1.0) < 1.0e-9:
            return (
                np.array([], dtype=float),
                [],
                {
                    "dataset_index": int(dataset_ctx.dataset_index),
                    "dataset_label": str(dataset_ctx.label),
                    "theta_initial_deg": float(kwargs.get("theta_value", 0.0)),
                    "measured_count": 2,
                    "fixed_source_resolved_count": 0,
                    "fallback_entry_count": 0,
                    "matched_pair_count": 0,
                    "missing_pair_count": 2,
                    "unweighted_peak_rms_px": float("nan"),
                    "unweighted_peak_mean_px": float("nan"),
                    "unweighted_peak_median_px": float("nan"),
                    "unweighted_peak_max_px": float("nan"),
                    "outside_radius_count": 0,
                    "point_match_cost": 400.0,
                    "weighted_rms_px": float("nan"),
                },
            )
        residual, diagnostics, summary = original_point_match_evaluator(
            local,
            dataset_ctx,
            **kwargs,
        )
        if abs(gamma - 0.0) < 1.0e-9:
            for entry in diagnostics:
                hkl = tuple(entry.get("hkl", ()))
                if hkl == (1, 0, 0):
                    entry["source_row_index"] = 0
                    entry["source_table_index"] = entry.get("resolved_table_index", 0)
                elif hkl == (0, 1, 0):
                    entry["source_row_index"] = 0
                    entry["source_table_index"] = entry.get("resolved_table_index", 1)
        return residual, diagnostics, summary

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)
    monkeypatch.setattr(
        opt,
        "_evaluate_geometry_fit_dataset_point_matches",
        wrapped_point_match_evaluator,
    )

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    intensities = np.array([25.0, 20.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["mosaic_params"] = {
        "beam_x_array": np.array([0.0, 0.1], dtype=np.float64),
        "beam_y_array": np.zeros(2, dtype=np.float64),
        "theta_array": np.zeros(2, dtype=np.float64),
        "phi_array": np.zeros(2, dtype=np.float64),
        "sigma_mosaic_deg": 0.2,
        "gamma_mosaic_deg": 0.1,
        "eta": 0.05,
        "wavelength_array": np.ones(2, dtype=np.float64),
    }
    measured = [
        {
            "label": "1,0,0",
            "hkl": (1, 0, 0),
            "x": 4.0,
            "y": 4.0,
            "sigma_px": 1.0,
        },
        {
            "label": "0,1,0",
            "hkl": (0, 1, 0),
            "x": 8.0,
            "y": 8.0,
            "sigma_px": 1.0,
        },
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "loss": "linear",
                "weighted_matching": False,
                "use_measurement_uncertainty": True,
                "stagnation_probe": False,
            },
            "full_beam_polish": {"enabled": True, "max_nfev": 10},
            "identifiability": {"enabled": False},
        },
    )

    assert result.success
    assert len(solve_calls) >= 2
    assert np.allclose(np.asarray(result.x, dtype=float), np.array([0.0]))
    assert result.final_metric_name == "full_beam_fixed_correspondence"
    assert str(result.full_beam_polish_summary["status"]) == "no_op_optimum"
    assert str(result.full_beam_polish_summary["selection_status"]) == "no_op_optimum"
    assert bool(result.full_beam_polish_summary["fit_quality_passed"]) is True
    assert bool(result.full_beam_polish_summary["preserved_start_on_reject"]) is False
    assert str(result.full_beam_polish_summary["reason"]) == (
        "no_valid_candidate_improved_raw_detector_alignment"
    )
    assert str(result.full_beam_polish_summary["selected_candidate_name"]) == "main_solve_start"
    assert (
        str(result.full_beam_polish_summary["selected_candidate_source"])
        == "geometry_fit_progress.start_x"
    )
    assert (
        str(result.full_beam_polish_summary["start_vector_source"])
        == "geometry_fit_progress.start_x"
    )
    assert str(result.full_beam_polish_summary["selected_candidate_name"]) == str(
        result.full_beam_polish_summary["best_valid_raw_detector_candidate_name"]
    )
    assert str(result.full_beam_polish_summary["selected_candidate_source"]) == str(
        result.full_beam_polish_summary["best_valid_raw_detector_candidate_source"]
    )
    ledger = list(result.full_beam_polish_summary["candidate_ledger"])
    selected_entry = next(entry for entry in ledger if entry["selected"] is True)
    assert str(selected_entry["candidate_name"]) == "main_solve_start"
    assert str(selected_entry["x_vector_source"]) == "geometry_fit_progress.start_x"
    assert any(
        entry["candidate_name"] == "central_point_result"
        and entry["x_vector_source"] == "current_result.x"
        and entry["valid_raw_detector_candidate"] is False
        for entry in ledger
    )
    assert any(
        entry["candidate_name"] == "full_beam_polish_result"
        and entry["x_vector_source"] == "full_beam_polish"
        and entry["valid_raw_detector_candidate"] is True
        and entry["selected"] is False
        for entry in ledger
    )
    assert np.allclose(np.asarray(result.fun, dtype=float), 0.0)
    assert int(result.point_match_summary["matched_pair_count"]) == 2
    assert int(result.point_match_summary["fixed_source_resolved_count"]) == 2
    assert int(result.point_match_summary.get("missing_fixed_pair_count", 0)) == 0
    assert str(result.point_match_summary["metric_name"]) == "full_beam_fixed_correspondence"


def test_full_beam_polish_rejection_retains_fixed_correspondence_start_when_current_result_loses_all_fixed_pairs(
    monkeypatch,
):
    from ra_sim.gui import geometry_fit as gui_geometry_fit

    solve_calls = []

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        gamma = float(args[8])
        beam_x_array = np.asarray(args[16], dtype=np.float64)
        image = np.zeros((image_size, image_size), dtype=np.float64)

        if beam_x_array.size > 1 and gamma >= 0.5:
            hit_tables = [
                np.empty((0, 7), dtype=np.float64),
                np.empty((0, 7), dtype=np.float64),
            ]
        else:
            hit_tables = [
                np.array(
                    [[10.0, 4.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
                np.array(
                    [[10.0, 8.0, 8.0, 0.0, 0.0, 1.0, 0.0]],
                    dtype=np.float64,
                ),
            ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        call_index = len(solve_calls)
        x = np.array([1.0], dtype=float) if call_index == 0 else np.array([2.0], dtype=float)
        solve_calls.append(x.copy())
        fun = (
            np.zeros(1, dtype=float) if call_index == 0 else np.asarray(residual_fn(x), dtype=float)
        )
        return opt.OptimizeResult(
            x=x,
            fun=fun,
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    intensities = np.array([25.0, 20.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["mosaic_params"] = {
        "beam_x_array": np.array([0.0, 0.1], dtype=np.float64),
        "beam_y_array": np.zeros(2, dtype=np.float64),
        "theta_array": np.zeros(2, dtype=np.float64),
        "phi_array": np.zeros(2, dtype=np.float64),
        "sigma_mosaic_deg": 0.2,
        "gamma_mosaic_deg": 0.1,
        "eta": 0.05,
        "wavelength_array": np.ones(2, dtype=np.float64),
    }
    measured = [
        {
            "label": "1,0,0",
            "hkl": (1, 0, 0),
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 0,
            "source_row_index": 0,
            "sigma_px": 1.0,
        },
        {
            "label": "0,1,0",
            "hkl": (0, 1, 0),
            "x": 8.0,
            "y": 8.0,
            "source_table_index": 1,
            "source_row_index": 0,
            "sigma_px": 1.0,
        },
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "loss": "linear",
                "weighted_matching": False,
                "use_measurement_uncertainty": True,
                "stagnation_probe": False,
            },
            "full_beam_polish": {"enabled": True, "max_nfev": 10},
            "identifiability": {"enabled": False},
        },
    )

    assert len(solve_calls) >= 2
    assert np.allclose(np.asarray(result.x, dtype=float), np.array([0.0]))
    assert isinstance(result.full_beam_polish_summary, dict)
    assert bool(result.full_beam_polish_summary["accepted"]) is False
    assert bool(result.full_beam_polish_summary["preserved_start_on_reject"]) is True
    assert str(result.full_beam_polish_summary["status"]) == "retained_start_safe_fallback"
    assert (
        str(result.full_beam_polish_summary["selection_status"]) == "retained_start_safe_fallback"
    )
    assert bool(result.full_beam_polish_summary["fit_quality_passed"]) is False
    assert str(result.full_beam_polish_summary["start_vector_source"]) != "current_result.x"
    assert len(list(result.full_beam_polish_summary["candidate_start_vector_sources"])) >= 2
    assert "resolved_fixed_pairs_decreased" in str(result.full_beam_polish_summary["reason"])
    assert int(result.full_beam_polish_summary["selected_start_fixed_source_resolved_count"]) == 2
    assert int(result.full_beam_polish_summary["current_fixed_source_resolved_count"]) == 0
    assert int(result.full_beam_polish_summary["candidate_fixed_source_resolved_count"]) == 0
    current_fallback = dict(result.full_beam_polish_summary["current_detector_fallback_summary"])
    rejected_start_fallback = dict(
        result.full_beam_polish_summary["rejected_start_detector_fallback_summary"]
    )
    assert bool(current_fallback["accepted"]) is False
    assert bool(rejected_start_fallback["accepted"]) is True
    assert int(current_fallback["matched_pair_count"]) == 0
    assert int(rejected_start_fallback["matched_pair_count"]) == 2
    assert int(result.full_beam_polish_summary["matched_pair_count_before"]) == 2
    assert int(result.full_beam_polish_summary["candidate_matched_pair_count"]) == 0
    assert result.final_metric_name == "full_beam_fixed_correspondence"
    assert isinstance(result.fun, np.ndarray)
    assert result.fun.ndim == 1
    assert np.allclose(result.fun, 0.0)
    assert int(result.point_match_summary["matched_pair_count"]) == 2
    assert int(result.point_match_summary["fixed_source_resolved_count"]) == 2
    assert (
        int(
            result.point_match_summary.get(
                "matched_fixed_pair_count",
                result.point_match_summary["fixed_source_resolved_count"],
            )
        )
        == 2
    )
    assert int(result.point_match_summary.get("missing_fixed_pair_count", 0)) == 0
    rejection_reasons = gui_geometry_fit.build_geometry_fit_rejection_reason_lines(
        result,
        rms=float(result.rms_px),
    )
    assert "No matched peak pairs were available for the fitted solution." not in rejection_reasons


def test_raw_detector_candidate_selector_prefers_lower_max_error_over_better_median() -> None:
    candidate_a = {
        "candidate_name": "candidate_a",
        "matched_fixed_pair_count": 7,
        "branch_mismatch_count": 0,
        "rms_px": 4.0,
        "median_px": 3.5,
        "max_px": 12.0,
        "outside_radius_count": 0,
        "weighted_objective": 8.0,
    }
    candidate_b = {
        "candidate_name": "candidate_b",
        "matched_fixed_pair_count": 7,
        "branch_mismatch_count": 0,
        "rms_px": 4.0,
        "median_px": 3.9,
        "max_px": 6.0,
        "outside_radius_count": 0,
        "weighted_objective": 8.5,
    }

    assert opt._raw_detector_candidate_is_better(candidate_b, candidate_a) is True
    assert opt._raw_detector_candidate_is_better(candidate_a, candidate_b) is False


def test_full_beam_polish_rejection_preserves_detector_bad_start_when_no_better_reference_exists(
    monkeypatch,
):
    from ra_sim.gui import geometry_fit as gui_geometry_fit

    solve_calls = []

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        gamma = float(args[8])
        beam_x_array = np.asarray(args[16], dtype=np.float64)
        image = np.zeros((image_size, image_size), dtype=np.float64)

        if beam_x_array.size > 1 and gamma >= 0.5:
            hit_tables = [
                np.empty((0, 7), dtype=np.float64),
                np.empty((0, 7), dtype=np.float64),
            ]
        else:
            hit_tables = [
                np.array(
                    [[10.0, 204.0, 204.0, 0.0, 1.0, 0.0, 0.0]],
                    dtype=np.float64,
                ),
                np.array(
                    [[10.0, 308.0, 308.0, 0.0, 0.0, 1.0, 0.0]],
                    dtype=np.float64,
                ),
            ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        call_index = len(solve_calls)
        x = np.array([1.0], dtype=float) if call_index == 0 else np.array([2.0], dtype=float)
        solve_calls.append(x.copy())
        fun = (
            np.zeros(1, dtype=float) if call_index == 0 else np.asarray(residual_fn(x), dtype=float)
        )
        return opt.OptimizeResult(
            x=x,
            fun=fun,
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 400
    miller = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    intensities = np.array([25.0, 20.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["mosaic_params"] = {
        "beam_x_array": np.array([0.0, 0.1], dtype=np.float64),
        "beam_y_array": np.zeros(2, dtype=np.float64),
        "theta_array": np.zeros(2, dtype=np.float64),
        "phi_array": np.zeros(2, dtype=np.float64),
        "sigma_mosaic_deg": 0.2,
        "gamma_mosaic_deg": 0.1,
        "eta": 0.05,
        "wavelength_array": np.ones(2, dtype=np.float64),
    }
    measured = [
        {
            "label": "1,0,0",
            "hkl": (1, 0, 0),
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 0,
            "source_row_index": 0,
            "sigma_px": 1.0,
        },
        {
            "label": "0,1,0",
            "hkl": (0, 1, 0),
            "x": 8.0,
            "y": 8.0,
            "source_table_index": 1,
            "source_row_index": 0,
            "sigma_px": 1.0,
        },
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "loss": "linear",
                "weighted_matching": False,
                "use_measurement_uncertainty": True,
                "stagnation_probe": False,
            },
            "full_beam_polish": {"enabled": True, "max_nfev": 10},
            "identifiability": {"enabled": False},
        },
    )

    assert len(solve_calls) >= 2
    assert np.allclose(np.asarray(result.x, dtype=float), np.array([0.0]))
    assert isinstance(result.full_beam_polish_summary, dict)
    assert bool(result.full_beam_polish_summary["accepted"]) is False
    assert bool(result.full_beam_polish_summary["preserved_start_on_reject"]) is True
    assert str(result.full_beam_polish_summary["status"]) == "retained_start_safe_fallback"
    assert (
        str(result.full_beam_polish_summary["selection_status"]) == "retained_start_safe_fallback"
    )
    assert bool(result.full_beam_polish_summary["fit_quality_passed"]) is False
    assert str(result.full_beam_polish_summary["start_vector_source"]) != "current_result.x"
    assert len(list(result.full_beam_polish_summary["candidate_start_vector_sources"])) >= 2
    assert "resolved_fixed_pairs_decreased" in str(result.full_beam_polish_summary["reason"])
    current_fallback = dict(result.full_beam_polish_summary["current_detector_fallback_summary"])
    rejected_start_fallback = dict(
        result.full_beam_polish_summary["rejected_start_detector_fallback_summary"]
    )
    assert bool(current_fallback["accepted"]) is False
    assert bool(rejected_start_fallback["accepted"]) is True
    assert int(rejected_start_fallback["matched_pair_count"]) == 2
    assert int(rejected_start_fallback["missing_pair_count"]) == 0
    assert float(rejected_start_fallback["unweighted_peak_rms_px"]) > 100.0
    assert float(rejected_start_fallback["unweighted_peak_max_px"]) > 150.0
    assert result.final_metric_name == "full_beam_fixed_correspondence"
    assert (
        int(
            result.point_match_summary.get(
                "matched_fixed_pair_count",
                result.point_match_summary["fixed_source_resolved_count"],
            )
        )
        == 2
    )
    assert int(result.point_match_summary.get("missing_fixed_pair_count", 0)) == 0
    assert float(result.point_match_summary["unweighted_peak_rms_px"]) > 100.0
    rejection_reasons = gui_geometry_fit.build_geometry_fit_rejection_reason_lines(
        result,
        rms=float(result.rms_px),
    )
    assert any("RMS residual" in reason for reason in rejection_reasons)
    assert "No matched peak pairs were available for the fitted solution." not in rejection_reasons


def test_full_beam_polish_rejection_preserves_central_point_match_result(monkeypatch):
    def fake_process(*args, **kwargs):
        miller_arg = np.asarray(args[0], dtype=np.float64)
        image_size = int(args[2])
        hit_tables = []
        coord_map = {
            (1, 0, 0): (1.0, 1.0),
            (2, 0, 0): (4.0, 4.0),
            (3, 0, 0): (7.0, 7.0),
        }
        best_sample_indices_out = kwargs.get("best_sample_indices_out")
        if isinstance(best_sample_indices_out, np.ndarray):
            best_sample_indices_out[:] = 0
        for row in miller_arg:
            hkl = tuple(int(round(v)) for v in row)
            col, row_px = coord_map[hkl]
            hit_tables.append(
                np.array(
                    [[1.0, col, row_px, 0.0, row[0], row[1], row[2]]],
                    dtype=np.float64,
                )
            )
        image = np.zeros((image_size, image_size), dtype=np.float64)
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 12
    miller = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {
            "hkl": (2, 0, 0),
            "label": "2,0,0",
            "x": 4.0,
            "y": 4.0,
            "source_table_index": 0,
            "source_row_index": 0,
        }
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    baseline = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {"restarts": 0, "weighted_matching": False},
            "full_beam_polish": {"enabled": False},
            "identifiability": {"enabled": False},
        },
    )
    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {"restarts": 0, "weighted_matching": False},
            "full_beam_polish": {"enabled": True, "max_nfev": 10},
            "identifiability": {"enabled": False},
        },
    )

    assert baseline.success
    assert result.success
    assert result.final_metric_name == baseline.final_metric_name == "central_point_match"
    assert np.allclose(result.fun, baseline.fun)
    assert (
        result.point_match_summary["matched_pair_count"]
        == baseline.point_match_summary["matched_pair_count"]
    )
    assert np.isclose(
        float(result.point_match_summary["unweighted_peak_rms_px"]),
        float(baseline.point_match_summary["unweighted_peak_rms_px"]),
        equal_nan=True,
    )
    assert len(result.point_match_diagnostics) == len(baseline.point_match_diagnostics)
    for result_entry, baseline_entry in zip(
        result.point_match_diagnostics,
        baseline.point_match_diagnostics,
    ):
        for key in (
            "match_status",
            "match_kind",
            "resolution_reason",
            "resolution_kind",
            "source_table_index",
            "source_row_index",
            "resolved_table_index",
            "resolved_peak_index",
            "source_branch_index",
        ):
            assert result_entry.get(key) == baseline_entry.get(key)
        for key in (
            "measured_x",
            "measured_y",
            "simulated_x",
            "simulated_y",
            "dx_px",
            "dy_px",
            "distance_px",
        ):
            assert np.isclose(
                float(result_entry.get(key, np.nan)),
                float(baseline_entry.get(key, np.nan)),
                equal_nan=True,
            )
    assert int(result.point_match_summary["matched_pair_count"]) == 1
    assert int(result.point_match_summary["matched_pair_count"]) == int(
        baseline.point_match_summary["matched_pair_count"]
    )
    assert isinstance(result.full_beam_polish_summary, dict)
    assert bool(result.full_beam_polish_summary["accepted"]) is False
    assert str(result.full_beam_polish_summary["reason"]) == "no_seed_correspondences"


def test_full_beam_polish_keeps_resolved_fixed_correspondence_outside_match_radius(
    monkeypatch,
):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        miller_local = np.asarray(args[0], dtype=np.float64)
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = []
        for row in miller_local:
            h, k, l = (int(round(float(v))) for v in row[:3])
            if (h, k, l) == (1, 0, 0):
                hit_tables.append(
                    np.asarray(
                        [
                            [1.0, 2.0, 2.0, -10.0, 1.0, 0.0, 0.0],
                            [1.0, 8.75, 8.0, 10.0, 1.0, 0.0, 0.0],
                        ],
                        dtype=np.float64,
                    )
                )
            else:
                hit_tables.append(
                    np.asarray(
                        [[1.0, 50.0, 50.0, 0.0, float(h), float(k), float(l)]],
                        dtype=np.float64,
                    )
                )
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array(
        [[5.0, 0.0, 0.0], [1.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    intensities = np.array([5.0, 10.0, 7.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["mosaic_params"] = {
        "beam_x_array": np.array([0.0, 0.1], dtype=np.float64),
        "beam_y_array": np.zeros(2, dtype=np.float64),
        "theta_array": np.zeros(2, dtype=np.float64),
        "phi_array": np.zeros(2, dtype=np.float64),
        "sigma_mosaic_deg": 0.2,
        "gamma_mosaic_deg": 0.1,
        "eta": 0.05,
        "wavelength_array": np.ones(2, dtype=np.float64),
    }
    measured = [
        {
            "pair_id": "bg0:pair0",
            "fit_run_id": "fit-123",
            "label": "1,0,0",
            "hkl": (1, 0, 0),
            "x": 8.0,
            "y": 8.0,
            "sigma_px": 1.0,
            "source_table_index": 99,
            "source_reflection_index": 1,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
        }
    ]

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=np.zeros((image_size, image_size), dtype=np.float64),
        refinement_config={
            "solver": {
                "restarts": 0,
                "weighted_matching": False,
                "use_measurement_uncertainty": True,
                "stagnation_probe": False,
            },
            "full_beam_polish": {
                "enabled": True,
                "max_nfev": 10,
                "match_radius_px": 0.5,
            },
            "identifiability": {"enabled": False},
        },
    )

    assert bool(result.success) is True
    assert bool(result.full_beam_polish_summary["accepted"]) is True
    assert result.final_metric_name == "full_beam_fixed_correspondence"
    assert int(result.full_beam_polish_summary["matched_pair_count_before"]) == 1
    assert int(result.full_beam_polish_summary["matched_pair_count_after"]) == 1
    start_diag = dict(result.full_beam_polish_summary["start_point_match_diagnostics"][0])
    final_diag = dict(result.point_match_diagnostics[0])
    for entry in (start_diag, final_diag):
        assert entry["match_status"] == "matched"
        assert entry["match_kind"] == "full_beam_fixed"
        assert entry["resolution_reason"] == "outside_match_radius"
        assert bool(entry["match_radius_exceeded"]) is True
        assert np.isclose(float(entry["distance_px"]), 0.75)


def test_fit_geometry_parameters_uses_manual_peak_sigma_by_default(monkeypatch):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[50.0, 14.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([25.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {
            "label": "1,0,0",
            "x": 4.0,
            "y": 4.0,
            "sigma_px": 5.0,
            "placement_error_px": 4.5,
            "source_table_index": 0,
            "source_row_index": 0,
        }
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "weighted_matching": False,
                "f_scale_px": 1.0,
            }
        },
    )

    assert result.success
    assert np.allclose(np.asarray(result.fun, dtype=float), [2.0, 0.0])
    diag = result.point_match_diagnostics[0]
    assert float(diag["distance_px"]) == 10.0
    assert float(diag["measurement_sigma_px"]) == 5.0
    assert np.isclose(float(diag["sigma_weight"]), 0.2)
    assert np.isclose(float(diag["weight"]), 0.2)
    assert np.isclose(float(diag["weighted_dx_px"]), 2.0)
    assert np.isclose(float(diag["placement_error_px"]), 4.5)
    assert str(result.point_match_summary["peak_weighting_mode"]) == "measurement_sigma"
    assert int(result.point_match_summary["custom_sigma_count"]) == 1
    assert np.isclose(float(result.point_match_summary["measurement_sigma_median_px"]), 5.0)
    assert np.isclose(float(result.rms_px), 10.0)
    assert float(result.weighted_residual_rms_px) < float(result.rms_px)


def test_fit_geometry_parameters_can_ignore_manual_peak_sigma_when_disabled(monkeypatch):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[50.0, 14.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([25.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {
            "label": "1,0,0",
            "x": 4.0,
            "y": 4.0,
            "sigma_px": 5.0,
            "placement_error_px": 4.5,
            "source_table_index": 0,
            "source_row_index": 0,
        }
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "weighted_matching": False,
                "use_measurement_uncertainty": False,
                "f_scale_px": 1.0,
            }
        },
    )

    assert result.success
    assert np.allclose(np.asarray(result.fun, dtype=float), [10.0, 0.0])
    diag = result.point_match_diagnostics[0]
    assert float(diag["distance_px"]) == 10.0
    assert float(diag["measurement_sigma_px"]) == 5.0
    assert np.isclose(float(diag["sigma_weight"]), 1.0)
    assert np.isclose(float(diag["weight"]), 1.0)
    assert np.isclose(float(diag["weighted_dx_px"]), 10.0)
    assert np.isclose(float(diag["placement_error_px"]), 4.5)
    assert str(result.point_match_summary["peak_weighting_mode"]) == "uniform"
    assert int(result.point_match_summary["custom_sigma_count"]) == 0
    assert np.isclose(float(result.rms_px), 10.0)
    assert np.isclose(
        float(result.weighted_residual_rms_px),
        np.sqrt((10.0**2 + 0.0**2) / 2.0),
    )


def test_fit_geometry_parameters_joint_backgrounds_share_theta_offset(monkeypatch):
    target_offset = 0.75

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        theta_initial = float(args[27])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, theta_initial, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        best_sample_indices_out = kwargs.get("best_sample_indices_out")
        if isinstance(best_sample_indices_out, np.ndarray):
            best_sample_indices_out[:] = 0
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["theta_offset"] = 0.0

    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)
    dataset_specs = [
        {
            "dataset_index": 0,
            "label": "bg0",
            "theta_initial": 3.0,
            "measured_peaks": [{"label": "1,0,0", "x": 3.0 + target_offset, "y": 4.0}],
            "experimental_image": experimental_image,
        },
        {
            "dataset_index": 1,
            "label": "bg1",
            "theta_initial": 7.0,
            "measured_peaks": [{"label": "1,0,0", "x": 7.0 + target_offset, "y": 4.0}],
            "experimental_image": experimental_image,
        },
    ]

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=dataset_specs[0]["measured_peaks"],
        var_names=["theta_offset"],
        experimental_image=experimental_image,
        dataset_specs=dataset_specs,
        refinement_config={
            "solver": {"restarts": 0, "weighted_matching": False, "max_nfev": 40},
            "single_ray": {"enabled": False},
        },
    )

    assert result.success
    assert result.x.shape == (1,)
    assert abs(float(result.x[0]) - target_offset) < 1e-6
    assert np.allclose(np.asarray(result.fun, dtype=float), 0.0, atol=1e-6)
    assert int(result.point_match_summary["dataset_count"]) == 2
    assert int(result.point_match_summary["matched_pair_count"]) == 2
    assert len(result.point_match_summary["per_dataset"]) == 2
    assert len(result.point_match_diagnostics) == 2
    assert {int(entry["dataset_index"]) for entry in result.point_match_diagnostics} == {0, 1}


def test_fit_geometry_parameters_accepts_numpy_dataset_specs(monkeypatch):
    target_offset = 0.5

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        theta_initial = float(args[27])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, theta_initial, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["theta_offset"] = 0.0

    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)
    dataset_specs = np.array(
        [
            {
                "dataset_index": 0,
                "label": "bg0",
                "theta_initial": 3.0,
                "measured_peaks": [{"label": "1,0,0", "x": 3.0 + target_offset, "y": 4.0}],
                "experimental_image": experimental_image,
            },
            {
                "dataset_index": 1,
                "label": "bg1",
                "theta_initial": 7.0,
                "measured_peaks": [{"label": "1,0,0", "x": 7.0 + target_offset, "y": 4.0}],
                "experimental_image": experimental_image,
            },
        ],
        dtype=object,
    )

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=dataset_specs[0]["measured_peaks"],
        var_names=["theta_offset"],
        experimental_image=experimental_image,
        dataset_specs=dataset_specs,
        refinement_config={
            "solver": {"restarts": 0, "weighted_matching": False, "max_nfev": 40},
            "single_ray": {"enabled": False},
        },
    )

    assert result.success
    assert result.x.shape == (1,)
    assert abs(float(result.x[0]) - target_offset) < 1e-6
    assert int(result.point_match_summary["dataset_count"]) == 2


def test_fit_geometry_parameters_parallelizes_multi_dataset_point_matching(monkeypatch):
    threaded_calls = []

    def fake_threaded_map(fn, items, *, max_workers, numba_threads=None):
        item_list = list(items)
        threaded_calls.append(
            {
                "max_workers": int(max_workers),
                "numba_threads": numba_threads,
                "count": len(item_list),
            }
        )
        return [fn(item) for item in item_list]

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        theta_initial = float(args[27])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, theta_initial, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    monkeypatch.setattr(opt, "_threaded_map", fake_threaded_map)
    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["theta_offset"] = 0.0

    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)
    dataset_specs = [
        {
            "dataset_index": 0,
            "label": "bg0",
            "theta_initial": 3.0,
            "measured_peaks": [{"label": "1,0,0", "x": 3.5, "y": 4.0}],
            "experimental_image": experimental_image,
        },
        {
            "dataset_index": 1,
            "label": "bg1",
            "theta_initial": 7.0,
            "measured_peaks": [{"label": "1,0,0", "x": 7.5, "y": 4.0}],
            "experimental_image": experimental_image,
        },
    ]

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=dataset_specs[0]["measured_peaks"],
        var_names=["theta_offset"],
        experimental_image=experimental_image,
        dataset_specs=dataset_specs,
        refinement_config={
            "solver": {
                "restarts": 0,
                "weighted_matching": False,
                "max_nfev": 40,
                "workers": 2,
                "parallel_mode": "datasets",
                "worker_numba_threads": 3,
            },
            "single_ray": {"enabled": False},
        },
    )

    assert result.success
    assert threaded_calls
    assert any(call["max_workers"] == 2 and call["count"] == 2 for call in threaded_calls)
    assert any(call["numba_threads"] == 3 for call in threaded_calls)
    assert result.parallelization_summary["dataset_workers"] == 2
    assert result.parallelization_summary["restart_workers"] == 1


def test_compute_sensitivity_weights_can_equalize_roi_totals():
    image_size = 31
    params = _base_params(image_size, optics_mode=1)
    params["gamma"] = 0.0
    params["pixel_size"] = 0.01
    miller = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    max_positions = np.array(
        [
            [1.0, 8.0, 15.0, np.nan, np.nan, np.nan],
            [25.0, 22.0, 15.0, np.nan, np.nan, np.nan],
        ],
        dtype=np.float64,
    )
    yy, xx = np.mgrid[0:image_size, 0:image_size]

    def simulator(local_params):
        shift = float(local_params.get("gamma", 0.0))
        image = np.exp(
            -((xx - (8.0 + shift)) ** 2 + (yy - 15.0) ** 2) / (2.0 * 0.8**2)
        ) + 25.0 * np.exp(-((xx - (22.0 + shift)) ** 2 + (yy - 15.0) ** 2) / (2.0 * 0.8**2))
        return image.astype(np.float64), max_positions

    base_sim, _ = simulator(params)
    rois_raw = opt.build_tube_rois(
        miller,
        max_positions,
        params,
        image_size,
        base_width=3.0,
    )
    opt.compute_sensitivity_weights(
        base_sim,
        params,
        ["gamma"],
        rois_raw,
        simulator,
        downsample_factor=1,
        percentile=80.0,
        huber_percentile=100.0,
        per_reflection_quota=25,
        off_tube_fraction=0.0,
        normalize_per_roi=False,
    )
    raw_sums = [float(np.sum(np.asarray(roi.weights, dtype=float))) for roi in rois_raw]
    assert raw_sums[1] > raw_sums[0] * 2.0

    rois_equal = opt.build_tube_rois(
        miller,
        max_positions,
        params,
        image_size,
        base_width=3.0,
    )
    opt.compute_sensitivity_weights(
        base_sim,
        params,
        ["gamma"],
        rois_equal,
        simulator,
        downsample_factor=1,
        percentile=80.0,
        huber_percentile=100.0,
        per_reflection_quota=25,
        off_tube_fraction=0.0,
        normalize_per_roi=True,
    )
    equal_sums = [float(np.sum(np.asarray(roi.weights, dtype=float))) for roi in rois_equal]
    assert np.isclose(equal_sums[0], 1.0)
    assert np.isclose(equal_sums[1], 1.0)


def test_fit_geometry_parameters_can_accept_roi_image_refinement(monkeypatch):
    captured_cfg = {}

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    def fake_stage_two(
        experimental_image,
        miller,
        intensities,
        image_size,
        params,
        var_names,
        simulator,
        measured_dict,
        *,
        cfg,
    ):
        captured_cfg.update(cfg)
        updated_params = dict(params)
        updated_params["gamma"] = 0.25
        updated_params["center"] = list(params.get("center", [image_size / 2.0, image_size / 2.0]))
        updated_params["center_x"] = float(updated_params["center"][0])
        updated_params["center_y"] = float(updated_params["center"][1])
        stage_result = opt.OptimizeResult(
            x=np.array([0.25], dtype=float),
            fun=np.zeros(1, dtype=float),
            success=True,
            status=1,
            message="roi-refine-ok",
            nfev=2,
            active_mask=np.zeros(1, dtype=int),
            optimality=0.0,
        )
        stage_result.initial_cost = 12.0
        stage_result.final_cost = 2.0
        return (
            updated_params,
            stage_result,
            [object(), object(), object()],
            np.zeros((image_size, image_size), dtype=float),
            lambda x: np.zeros(1, dtype=float),
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", _fake_process_three_reflections)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)
    monkeypatch.setattr(opt, "_stage_two_refinement", fake_stage_two)

    image_size = 20
    miller = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    intensities = np.array([5.0, 4.0, 3.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {"label": "1,0,0", "x": 4.0, "y": 4.0},
        {"label": "0,1,0", "x": 10.0, "y": 10.0},
        {"label": "0,0,1", "x": 14.0, "y": 14.0},
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {"restarts": 0, "weighted_matching": False},
            "image_refinement": {"enabled": True, "min_rois": 3},
        },
    )

    assert result.success
    assert np.isclose(float(result.x[0]), 0.25)
    assert isinstance(result.image_refinement_summary, dict)
    assert bool(result.image_refinement_summary["accepted"]) is True
    assert str(result.image_refinement_summary["status"]) == "accepted"
    assert bool(captured_cfg["equal_peak_weights"]) is True
    assert int(result.image_refinement_summary["selected_roi_count"]) == 3
    assert "ROI/image refinement accepted" in str(result.message)
    assert int(result.point_match_summary["matched_pair_count"]) == 3


def test_fit_geometry_parameters_defaults_to_point_only_fit_without_image_stages(
    monkeypatch,
):
    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    def fail_stage_one(*args, **kwargs):
        raise AssertionError("ridge refinement should be disabled by default")

    def fail_stage_two(*args, **kwargs):
        raise AssertionError("image refinement should be disabled by default")

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", _fake_process_peaks)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)
    monkeypatch.setattr(opt, "_stage_one_initialize", fail_stage_one)
    monkeypatch.setattr(opt, "_stage_two_refinement", fail_stage_two)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([10.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={"solver": {"restarts": 0, "weighted_matching": False}},
    )

    assert result.success
    assert isinstance(result.ridge_refinement_summary, dict)
    assert bool(result.ridge_refinement_summary["enabled"]) is False
    assert str(result.ridge_refinement_summary["reason"]) == "disabled_by_config"
    assert isinstance(result.image_refinement_summary, dict)
    assert bool(result.image_refinement_summary["enabled"]) is False
    assert str(result.image_refinement_summary["reason"]) == "disabled_by_config"


def test_fit_geometry_parameters_manual_point_fit_mode_uses_lean_defaults(
    monkeypatch,
):
    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", _fake_process_peaks)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([10.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={"solver": {"manual_point_fit_mode": True}},
    )

    assert result.success
    assert isinstance(result.geometry_fit_debug_summary, dict)
    assert isinstance(result.geometry_fit_debug_summary["solver"], dict)
    solver_debug = result.geometry_fit_debug_summary["solver"]
    assert bool(solver_debug["manual_point_fit_mode"]) is True
    assert np.isclose(float(solver_debug["f_scale_px"]), 1.0)
    assert bool(solver_debug["q_group_line_constraints"]) is True
    assert np.isclose(float(solver_debug["hk0_peak_priority_weight"]), 6.0)
    assert int(solver_debug["restarts"]) >= 1
    assert bool(solver_debug["use_measurement_uncertainty"]) is False
    assert bool(solver_debug["full_beam_polish_enabled"]) is False
    assert isinstance(result.full_beam_polish_summary, dict)
    assert bool(result.full_beam_polish_summary["enabled"]) is False
    assert str(result.full_beam_polish_summary["reason"]) == "disabled_by_config"


def test_fit_geometry_parameters_manual_point_fit_adds_q_group_line_residuals(
    monkeypatch,
):
    captured: dict[str, np.ndarray] = {}

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, 4.0, 5.0, 0.0, 1.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
            np.array(
                [[1.0, 8.0, 9.0, 0.0, -1.0, 1.0, 0.0]],
                dtype=np.float64,
            ),
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        residual = np.asarray(residual_fn(x), dtype=float)
        captured["residual"] = residual.copy()
        return opt.OptimizeResult(
            x=x,
            fun=residual,
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 32
    miller = np.array([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0, 1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    params["debye_x"] = 1.0
    params["debye_y"] = 1.0
    measured = [
        {
            "hkl": (1, 1, 0),
            "x": 4.0,
            "y": 5.0,
            "source_table_index": 0,
            "source_row_index": 0,
            "q_group_key": ("q_group", "primary", 1, 0),
        },
        {
            "hkl": (-1, 1, 0),
            "x": 8.0,
            "y": 9.0,
            "source_table_index": 1,
            "source_row_index": 0,
            "q_group_key": ("q_group", "primary", 1, 0),
        },
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={"solver": {"manual_point_fit_mode": True}},
    )

    assert result.success
    assert captured["residual"].shape == (6,)
    assert np.allclose(captured["residual"], 0.0, atol=1.0e-6)
    assert int(result.point_match_summary["line_group_count"]) == 1
    assert int(result.point_match_summary["resolved_line_group_count"]) == 1
    assert int(result.point_match_summary["missing_line_group_count"]) == 0


def test_fit_geometry_parameters_hk0_peaks_receive_priority_weight(monkeypatch):
    captured: dict[str, np.ndarray] = {}

    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, 5.0, 5.0, 0.0, 0.0, 0.0, 2.0]],
                dtype=np.float64,
            ),
            np.array(
                [[1.0, 9.0, 9.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            ),
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        residual = np.asarray(residual_fn(x), dtype=float)
        captured["residual"] = residual.copy()
        return opt.OptimizeResult(
            x=x,
            fun=residual,
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 24
    miller = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([10.0, 10.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {
            "hkl": (0, 0, 2),
            "x": 4.0,
            "y": 5.0,
            "source_table_index": 0,
            "source_row_index": 0,
        },
        {
            "hkl": (1, 0, 0),
            "x": 8.0,
            "y": 9.0,
            "source_table_index": 1,
            "source_row_index": 0,
        },
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "weighted_matching": False,
                "hk0_peak_priority_weight": 5.0,
            }
        },
    )

    assert result.success
    assert np.allclose(captured["residual"], [5.0, 0.0, 1.0, 0.0])
    diag_by_hkl = {
        tuple(diag["hkl"]): diag
        for diag in result.point_match_diagnostics
        if isinstance(diag.get("hkl"), tuple)
    }
    assert np.isclose(float(diag_by_hkl[(0, 0, 2)]["priority_weight"]), 5.0)
    assert str(diag_by_hkl[(0, 0, 2)]["priority_class"]) == "hk0"
    assert np.isclose(float(diag_by_hkl[(1, 0, 0)]["priority_weight"]), 1.0)
    assert str(diag_by_hkl[(1, 0, 0)]["priority_class"]) == "default"


def test_fit_geometry_parameters_manual_point_fit_guardrail_aborts_bound_hugging_bad_seed(
    monkeypatch,
):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[10.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x_bad = np.asarray(kwargs["bounds"][1], dtype=float)
        for _ in range(40):
            residual_fn(x_bad)
        raise AssertionError("manual fail-fast guardrail did not abort the solve")

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 1024
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 400.0, "y": 400.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma", "Gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "bounds": {
                "gamma": {"mode": "absolute", "min": 0.0, "max": 1.0},
                "Gamma": {"mode": "absolute", "min": 0.0, "max": 1.0},
            },
            "solver": {
                "manual_point_fit_mode": True,
            },
        },
    )

    assert not result.success
    assert "Manual point-fit guardrail stopped the solve" in str(result.message)
    assert "gamma, Gamma" in str(result.message)
    assert str(result.early_stop_reason).startswith("Manual point-fit guardrail stopped the solve")
    assert bool(result.geometry_fit_progress["aborted_early"]) is True
    assert float(result.point_match_summary["unweighted_peak_rms_px"]) > 250.0
    assert sorted(result.bound_hits) == ["Gamma", "gamma"]


def test_fit_geometry_parameters_can_accept_ridge_refinement(monkeypatch):
    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    def fake_stage_one_initialize(
        experimental_image,
        params,
        var_names,
        simulator,
        *,
        downsample_factor,
        max_nfev,
        bounds=None,
        x_scale=None,
    ):
        updated_params = dict(params)
        updated_params["gamma"] = 0.125
        stage_result = opt.OptimizeResult(
            x=np.array([0.125], dtype=float),
            fun=np.zeros(1, dtype=float),
            success=True,
            status=1,
            message="ridge-refine-ok",
            nfev=2,
            active_mask=np.zeros(1, dtype=int),
            optimality=0.0,
        )
        stage_result.initial_cost = 5.0
        stage_result.final_cost = 1.0
        stage_result.cost_reduction = 0.8
        return updated_params, stage_result

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", _fake_process_peaks)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)
    monkeypatch.setattr(opt, "_stage_one_initialize", fake_stage_one_initialize)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([10.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {"restarts": 0, "weighted_matching": False},
            "ridge_refinement": {"enabled": True},
            "image_refinement": {"enabled": False},
        },
    )

    assert result.success
    assert np.isclose(float(result.x[0]), 0.125)
    assert isinstance(result.ridge_refinement_summary, dict)
    assert bool(result.ridge_refinement_summary["accepted"]) is True
    assert str(result.ridge_refinement_summary["status"]) == "accepted"
    assert "Ridge refinement accepted" in str(result.message)


def test_fit_geometry_parameters_supports_anisotropic_measurement_weighting(monkeypatch):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[20.0, 14.0, 18.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([25.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [
        {
            "label": "1,0,0",
            "x": 14.0,
            "y": 10.0,
            "sigma_px": 2.0,
        }
    ]
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "solver": {
                "restarts": 0,
                "weighted_matching": False,
                "use_measurement_uncertainty": True,
                "anisotropic_measurement_uncertainty": True,
                "radial_sigma_scale": 1.0,
                "tangential_sigma_scale": 4.0,
            },
            "ridge_refinement": {"enabled": False},
            "image_refinement": {"enabled": False},
        },
    )

    assert result.success
    assert np.allclose(np.asarray(result.fun, dtype=float), [0.0, 1.0])
    diag = result.point_match_diagnostics[0]
    assert np.isclose(float(diag["sigma_radial_px"]), 2.0)
    assert np.isclose(float(diag["sigma_tangential_px"]), 8.0)
    assert np.isclose(float(diag["weighted_dx_px"]), 0.0)
    assert np.isclose(float(diag["weighted_dy_px"]), 1.0)
    assert np.isclose(float(diag["weighted_tangential_residual_px"]), 1.0)
    assert bool(diag["anisotropic_sigma_used"]) is True
    assert str(result.point_match_summary["peak_weighting_mode"]) == "measurement_covariance"
    assert int(result.point_match_summary["anisotropic_sigma_count"]) == 1
    assert np.isclose(float(result.rms_px), 8.0)
    assert float(result.weighted_residual_rms_px) < float(result.rms_px)


def test_fit_geometry_parameters_reports_solver_and_data_only_identifiability(
    monkeypatch,
):
    def fake_compute(*args, **kwargs):
        gamma = float(args[0])
        return np.array([gamma - 1.0], dtype=np.float64)

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "compute_peak_position_error_geometry_local", fake_compute)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size)
    params["gamma"] = 1.0

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=[[1.0, 0.0, 0.0, 4.0, 4.0]],
        var_names=["gamma", "Gamma"],
        experimental_image=None,
        refinement_config={
            "solver": {"restarts": 0},
            "identifiability": {"enabled": True},
        },
    )

    assert result.success

    solver_summary = result.identifiability_summary
    assert isinstance(solver_summary, dict)
    assert str(solver_summary["status"]) == "ok"
    assert str(solver_summary["diagnostic_scope"]) == "solver_conditioned_active"
    assert bool(solver_summary["includes_priors"]) is True
    assert int(solver_summary["num_parameters"]) == 2
    assert int(solver_summary["rank"]) == 1
    assert bool(solver_summary["underconstrained"]) is True
    solver_entries = solver_summary["parameter_entries"]
    assert [entry["name"] for entry in solver_entries] == ["gamma", "Gamma"]
    assert float(solver_entries[0]["column_norm"]) > 0.0
    assert np.isclose(float(solver_entries[1]["column_norm"]), 0.0)

    data_summary = result.data_only_identifiability_summary
    assert isinstance(data_summary, dict)
    assert str(data_summary["status"]) == "ok"
    assert str(data_summary["diagnostic_scope"]) == "data_only_all_selectable"
    assert bool(data_summary["includes_priors"]) is False
    assert int(data_summary["rank"]) == 1
    assert bool(data_summary["underconstrained"]) is True
    weak_parameters = data_summary["weak_parameters"]
    assert len(weak_parameters) == 1
    assert str(weak_parameters[0]["name"]) == "Gamma"
    assert list(result.next_stage_recommendations) == []
    assert result.next_stage_recommendation is None


def test_fit_geometry_parameters_correlated_inactive_block_is_recommended_together(
    monkeypatch,
):
    def fake_compute(*args, **kwargs):
        gamma = float(args[0])
        Gamma = float(args[1])
        coupled = gamma + Gamma - 1.0
        return np.array([coupled, 2.0 * coupled], dtype=np.float64)

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "compute_peak_position_error_geometry_local", fake_compute)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size)

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=[[1.0, 0.0, 0.0, 4.0, 4.0]],
        var_names=[],
        candidate_param_names=["gamma", "Gamma"],
        experimental_image=None,
        refinement_config={
            "solver": {"restarts": 0},
            "identifiability": {"enabled": True},
        },
    )

    assert bool(result.success) is False
    assert np.isclose(float(result.cost), 2.5)
    assert str(result.message) == "fixed-parameter mode identity"
    assert str(result.identifiability_summary["status"]) == "failed"
    assert str(result.identifiability_summary["reason"]) == "empty_parameter_vector"

    high_pairs = result.data_only_identifiability_summary["high_correlation_pairs"]
    assert len(high_pairs) == 1
    assert {
        str(high_pairs[0]["name_i"]),
        str(high_pairs[0]["name_j"]),
    } == {"gamma", "Gamma"}
    assert float(high_pairs[0]["abs_correlation"]) > 0.99

    recommendations = result.next_stage_recommendations
    assert len(recommendations) == 1
    assert set(recommendations[0]["params"]) == {"gamma", "Gamma"}
    assert "Correlated block" in str(recommendations[0]["reason"])
    assert isinstance(result.next_stage_recommendation, dict)
    assert set(result.next_stage_recommendation["params"]) == {"gamma", "Gamma"}


def test_fit_geometry_parameters_weak_inactive_parameter_is_not_recommended(
    monkeypatch,
):
    def fake_compute(*args, **kwargs):
        gamma = float(args[0])
        return np.array([gamma - 1.0], dtype=np.float64)

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "compute_peak_position_error_geometry_local", fake_compute)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size)
    params["gamma"] = 1.0

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=[[1.0, 0.0, 0.0, 4.0, 4.0]],
        var_names=["gamma"],
        candidate_param_names=["gamma", "Gamma"],
        experimental_image=None,
        refinement_config={
            "solver": {"restarts": 0},
            "identifiability": {"enabled": True},
        },
    )

    assert result.success
    weak_parameters = result.data_only_identifiability_summary["weak_parameters"]
    assert len(weak_parameters) == 1
    assert str(weak_parameters[0]["name"]) == "Gamma"
    assert list(result.next_stage_recommendations) == []
    assert result.next_stage_recommendation is None


def test_fit_geometry_parameters_reports_retired_stage_placeholders(monkeypatch):
    def fake_compute(*args, **kwargs):
        gamma = float(args[0])
        Gamma = float(args[1])
        dist = float(args[2])
        return np.array([gamma - 1.0, Gamma - 2.0, dist - 3.0], dtype=np.float64)

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "compute_peak_position_error_geometry_local", fake_compute)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size)
    params["corto_detector"] = 0.1

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=[],
        var_names=["gamma", "Gamma", "corto_detector"],
        experimental_image=None,
        refinement_config={
            "solver": {
                "restarts": 0,
                "staged_release": {"enabled": True},
                "reparameterize_pairs": {"enabled": True},
            },
            "identifiability": {
                "enabled": True,
                "adaptive_regularization": {"enabled": True},
                "auto_freeze": True,
                "selective_thaw": {"enabled": True},
            },
        },
    )

    assert result.success
    for summary in (
        result.reparameterization_summary,
        result.staged_release_summary,
        result.adaptive_regularization_summary,
        result.auto_freeze_summary,
        result.selective_thaw_summary,
    ):
        assert isinstance(summary, dict)
        assert str(summary["status"]) == "skipped"
        assert bool(summary["accepted"]) is False


def test_fit_geometry_parameters_selects_best_discrete_mode(monkeypatch):
    def fake_process(*args, **kwargs):
        image_size = int(args[2])
        image = np.zeros((image_size, image_size), dtype=np.float64)
        hit_tables = [
            np.array(
                [[1.0, 2.0, 4.0, 0.0, 1.0, 0.0, 0.0]],
                dtype=np.float64,
            )
        ]
        return image, hit_tables, np.empty((0, 0, 0)), np.empty(0), np.empty(0), []

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 12
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    experimental_image = np.zeros((image_size, image_size), dtype=np.float64)
    transformed = opt._transform_points_orientation_local(
        [(2.0, 4.0)],
        (image_size, image_size),
        indexing_mode="xy",
        k=1,
        flip_x=False,
        flip_y=False,
        flip_order="yx",
    )
    measured = [{"label": "1,0,0", "x": transformed[0][0], "y": transformed[0][1]}]

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=[],
        candidate_param_names=["gamma"],
        experimental_image=experimental_image,
        refinement_config={
            "discrete_modes": {
                "enabled": True,
                "rot90": [0, 1, 2, 3],
                "flip_x": [False],
                "flip_y": [False],
            },
            "identifiability": {"enabled": False},
            "full_beam_polish": {"enabled": False},
        },
    )

    assert bool(result.success) is False
    assert np.isclose(float(result.cost), 0.0)
    assert str(result.message) == "fixed-parameter mode rot270"
    assert isinstance(result.chosen_discrete_mode, dict)
    assert int(result.chosen_discrete_mode["k"]) == 3
    assert str(result.discrete_mode_summary["selected_label"]) == "rot270"


def test_fit_geometry_parameters_emits_normalized_multistart_status_updates(
    monkeypatch,
):
    status_messages = []

    def fake_compute(*args, **kwargs):
        gamma = float(args[0])
        Gamma = float(args[1])
        dist = float(args[2])
        return np.array([gamma - 1.0, Gamma - 2.0, dist - 3.0], dtype=np.float64)

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return opt.OptimizeResult(
            x=x,
            fun=np.asarray(residual_fn(x), dtype=float),
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "compute_peak_position_error_geometry_local", fake_compute)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 20
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size)
    params["corto_detector"] = 0.1

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=[],
        var_names=["gamma", "Gamma", "corto_detector"],
        experimental_image=None,
        refinement_config={
            "solver": {"restarts": 0},
            "identifiability": {"enabled": False},
        },
        status_callback=status_messages.append,
    )

    assert result.success
    assert isinstance(result.geometry_fit_debug_summary, dict)
    assert int(result.geometry_fit_debug_summary["dataset_count"]) == 1
    assert list(result.geometry_fit_debug_summary["var_names"]) == [
        "gamma",
        "Gamma",
        "corto_detector",
    ]
    assert isinstance(result.geometry_fit_debug_summary.get("solve_progress"), dict)
    assert int(result.geometry_fit_debug_summary["solve_progress"]["evaluation_count"]) >= 1
    assert any("Geometry fit: setup mode=angle" in msg for msg in status_messages)
    assert any("running normalized-u multistart solve" in msg for msg in status_messages)
    assert any("Geometry fit: mode identity prescore" in msg for msg in status_messages)
    assert any(
        "Geometry fit: multistart summary selected_mode=identity" in msg for msg in status_messages
    )
    assert any("identity seed" in msg and "cost=" in msg for msg in status_messages)
    assert any("complete" in msg and "metric=angle" in msg for msg in status_messages)


def test_fit_geometry_parameters_u_solver_live_update_reuses_point_summary(
    monkeypatch,
):
    calls = {"process": 0}
    live_updates: list[dict[str, object]] = []

    def fake_process(*args, **kwargs):
        calls["process"] += 1
        return _fake_process_peaks(*args, **kwargs)

    def fake_least_squares(residual_fn, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        before = int(calls["process"])
        fun = np.asarray(residual_fn(x), dtype=float)
        assert int(calls["process"]) == before + 1
        return opt.OptimizeResult(
            x=x,
            fun=fun,
            success=True,
            status=1,
            message="ok",
            nfev=1,
            active_mask=np.zeros_like(x, dtype=int),
            optimality=0.0,
        )

    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", fake_process)
    monkeypatch.setattr(opt, "least_squares", fake_least_squares)

    image_size = 12
    miller = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    intensities = np.array([1.0], dtype=np.float64)
    params = _base_params(image_size, optics_mode=1)
    measured = [{"label": "1,0,0", "x": 4.0, "y": 4.0}]

    result = opt.fit_geometry_parameters(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks=measured,
        var_names=["gamma"],
        experimental_image=np.zeros((image_size, image_size), dtype=np.float64),
        refinement_config={
            "solver": {"restarts": 0, "weighted_matching": False},
            "identifiability": {"enabled": False},
            "full_beam_polish": {"enabled": False},
        },
        live_update_callback=lambda payload: live_updates.append(dict(payload)),
    )

    assert result.success
    assert live_updates
    assert any("u_trial" in payload for payload in live_updates)
    assert any(
        int(payload.get("point_match_summary", {}).get("matched_pair_count", 0)) == 1
        for payload in live_updates
    )
    assert any(
        int(payload.get("point_match_summary", {}).get("branch_mismatch_count", -1)) == 0
        for payload in live_updates
    )
