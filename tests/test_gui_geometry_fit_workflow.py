import importlib.util
import json
import hashlib
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import numpy as np
from pathlib import Path, PurePosixPath
import pytest
from types import SimpleNamespace

from ra_sim.fitting import optimization as opt
from ra_sim.gui import (
    geometry_fit,
    geometry_fit_coordinate_diagnostics as coord_diag,
    geometry_overlay,
    manual_geometry,
)
from ra_sim.io.data_loading import load_gui_state_file, save_gui_state_file


class _DummyVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class _DummySlider:
    def __init__(self, from_value, to_value):
        self._values = {
            "from": from_value,
            "to": to_value,
        }

    def cget(self, key):
        return self._values[key]


def _make_prepared_run(
    *,
    joint_background_mode: bool,
    tmp_path=None,
    stamp: str = "20260328_120000",
    log_dir=None,
):
    downloads_dir = tmp_path if tmp_path is not None else "C:/tmp"
    resolved_log_dir = log_dir if log_dir is not None else downloads_dir
    return geometry_fit.GeometryFitPreparedRun(
        fit_params={"theta_initial": 3.0, "theta_offset": 0.0},
        selected_background_indices=[0, 1] if joint_background_mode else [0],
        background_theta_values=[3.0, 4.0] if joint_background_mode else [3.0],
        joint_background_mode=joint_background_mode,
        current_dataset={
            "dataset_index": 0,
            "measured_for_fit": [{"x": 1.0, "y": 2.0}],
            "experimental_image_for_fit": "fit-image",
            "initial_pairs_display": [{"pair": 1}],
            "native_background": np.zeros((4, 5)),
            "orientation_choice": {"label": "rotate"},
            "pair_count": 3,
            "group_count": 2,
        },
        dataset_infos=[
            {"dataset_index": 0, "summary_line": "bg[0]"},
            {"dataset_index": 1, "summary_line": "bg[1]"},
        ]
        if joint_background_mode
        else [{"dataset_index": 0, "summary_line": "bg[0]"}],
        dataset_specs=[
            {"dataset_index": 0, "theta_initial": 3.0},
            {"dataset_index": 1, "theta_initial": 4.0},
        ]
        if joint_background_mode
        else [{"dataset_index": 0, "theta_initial": 3.0}],
        start_cmd_line=(
            "start: vars=gamma,a "
            f"datasets={2 if joint_background_mode else 1} "
            "current_groups=2 current_points=3"
        ),
        start_log_sections=[
            ("Fitting variables (start values):", ["gamma=0.200000", "a=4.100000"]),
            ("Manual geometry datasets:", ["bg[0]"]),
        ],
        max_display_markers=120,
        geometry_runtime_cfg={"bounds": {"gamma": [0.0, 1.0]}},
    ), geometry_fit.GeometryFitRuntimePostprocessConfig(
        current_background_index=0,
        downloads_dir=downloads_dir,
        stamp=stamp,
        log_path=Path(resolved_log_dir) / f"geometry_fit_log_{stamp}.txt",
        solver_inputs=geometry_fit.GeometryFitRuntimeSolverInputs(
            miller="miller-data",
            intensities="intensity-data",
            image_size=512,
        ),
        sim_display_rotate_k=1,
        background_display_rotate_k=3,
        simulate_and_compare_hkl=None,
        aggregate_match_centers=None,
        build_overlay_records=None,
        compute_frame_diagnostics=None,
        log_dir=resolved_log_dir,
    )


def _make_runtime_action_prepare_bindings():
    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=100,
        display_rotate_k=3,
        geometry_manual_pairs_for_index=lambda idx: [{"pair": idx}],
        load_background_by_index=lambda idx: (
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_simulated_peaks_for_params=lambda *args, **kwargs: [],
        geometry_manual_simulated_lookup=lambda _peaks: {},
        geometry_manual_entry_display_coords=lambda entry: None,
        unrotate_display_peaks=lambda entries, shape, *, k: [],
        display_to_native_sim_coords=lambda col, row, shape: (col, row),
        select_fit_orientation=lambda *args, **kwargs: ({}, {}),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: [],
        orient_image_for_fit=lambda image, **kwargs: image,
    )
    return geometry_fit.GeometryFitRuntimePreparationBindings(
        fit_config={},
        theta_initial=9.0,
        apply_geometry_fit_background_selection=lambda **kwargs: True,
        current_geometry_fit_background_indices=lambda **kwargs: [0],
        geometry_fit_uses_shared_theta_offset=lambda *_args, **_kwargs: False,
        apply_background_theta_metadata=lambda **kwargs: True,
        current_background_theta_values=lambda **kwargs: [9.0],
        current_geometry_theta_offset=lambda **kwargs: 0.0,
        ensure_geometry_fit_caked_view=lambda: None,
        manual_dataset_bindings=manual_dataset_bindings,
        build_runtime_config=lambda fit_params: dict(fit_params),
    )


def _make_runtime_action_execution_bindings(
    tmp_path,
    progress_texts,
    cmd_events,
    *,
    log_dir=None,
):
    class _SimulationState:
        def __init__(self):
            self.profile_cache = {}
            self.last_simulation_signature = "sig"

    class _BackgroundState:
        def __init__(self):
            self.current_background_index = 0

    return geometry_fit.GeometryFitRuntimeActionExecutionBindings(
        downloads_dir=tmp_path,
        simulation_runtime_state=_SimulationState(),
        background_runtime_state=_BackgroundState(),
        theta_initial_var=_DummyVar(3.0),
        geometry_theta_offset_var=None,
        current_ui_params=lambda: {},
        var_map={},
        background_theta_for_index=lambda idx, strict_count=False: 7.5,
        refresh_status=lambda: None,
        update_manual_pick_button_label=lambda: None,
        capture_undo_state=lambda: {},
        push_undo_state=lambda state: None,
        replace_dataset_cache=lambda payload: None,
        request_preview_skip_once=lambda: None,
        schedule_update=lambda: None,
        draw_overlay_records=lambda records, marker_limit: None,
        draw_initial_pairs_overlay=lambda pairs, marker_limit: None,
        set_last_overlay_state=lambda state: None,
        set_progress_text=lambda text: progress_texts.append(text),
        cmd_line=lambda text: cmd_events.append(text),
        solver_inputs=geometry_fit.GeometryFitRuntimeSolverInputs(
            miller="miller-data",
            intensities="intensity-data",
            image_size=512,
        ),
        sim_display_rotate_k=1,
        background_display_rotate_k=3,
        simulate_and_compare_hkl=lambda *args, **kwargs: None,
        aggregate_match_centers=lambda *args, **kwargs: ([], [], []),
        build_overlay_records=lambda *args, **kwargs: [],
        compute_frame_diagnostics=lambda records: ({}, None),
        log_dir=log_dir,
    )


def _make_stub_caked_bundle(
    *,
    detector_shape: tuple[int, int],
    radial_axis,
    azimuth_axis,
):
    radial_vec = np.asarray(radial_axis, dtype=np.float64).copy()
    gui_display_axis = np.asarray(azimuth_axis, dtype=np.float64).copy()
    raw_azimuth_vec = np.asarray(
        geometry_fit.gui_phi_to_raw_phi(gui_display_axis),
        dtype=np.float64,
    )
    raw_azimuth_vec = np.sort(raw_azimuth_vec, kind="stable")
    gui_azimuth_vec = np.asarray(
        geometry_fit.raw_phi_to_gui_phi(raw_azimuth_vec),
        dtype=np.float64,
    )
    return geometry_fit.CakeTransformBundle(
        detector_shape=tuple(int(v) for v in detector_shape),
        radial_deg=radial_vec,
        raw_azimuth_deg=np.asarray(raw_azimuth_vec, dtype=np.float64).copy(),
        gui_azimuth_deg=np.asarray(gui_azimuth_vec, dtype=np.float64).copy(),
        lut=object(),
    )


def test_prepare_geometry_fit_run_builds_joint_background_datasets_with_current_first() -> None:
    calls: dict[str, object] = {
        "selection": [],
        "theta": [],
        "datasets": [],
        "runtime_cfg": [],
        "ensure_caked": 0,
    }

    def _build_dataset(background_index, *, theta_base, base_fit_params, orientation_cfg):
        calls["datasets"].append(
            (
                int(background_index),
                float(theta_base),
                float(base_fit_params["theta_initial"]),
                float(base_fit_params["theta_offset"]),
                dict(orientation_cfg),
            )
        )
        return {
            "dataset_index": int(background_index),
            "pair_count": 2,
            "group_count": 1,
            "summary_line": f"bg[{int(background_index)}]",
            "spec": {"dataset_index": int(background_index)},
            "orientation_choice": {"label": "identity"},
            "orientation_diag": {"pairs": 2},
            "measured_for_fit": [],
            "experimental_image_for_fit": f"image-{int(background_index)}",
            "initial_pairs_display": [],
            "native_background": np.zeros((4, 4)),
        }

    result = geometry_fit.prepare_geometry_fit_run(
        params={"theta_initial": 9.0, "theta_offset": 0.1},
        var_names=["gamma", "theta_offset"],
        fit_config={
            "geometry": {
                "orientation": {"mode": "auto"},
                "auto_match": {"max_display_markers": 150},
            }
        },
        osc_files=["bg0.osc", "bg1.osc", "bg2.osc"],
        current_background_index=1,
        theta_initial=9.0,
        preserve_live_theta=False,
        apply_geometry_fit_background_selection=(
            lambda **kwargs: calls["selection"].append(kwargs) or True
        ),
        current_geometry_fit_background_indices=lambda **kwargs: [0, 1, 2],
        geometry_fit_uses_shared_theta_offset=lambda indices: list(indices) == [0, 1, 2],
        apply_background_theta_metadata=(lambda **kwargs: calls["theta"].append(kwargs) or True),
        current_background_theta_values=lambda **kwargs: [1.0, 2.0, 3.0],
        current_geometry_theta_offset=lambda **kwargs: 0.5,
        geometry_manual_pairs_for_index=lambda idx: [{"pair": idx}],
        ensure_geometry_fit_caked_view=lambda: calls.__setitem__(
            "ensure_caked", int(calls["ensure_caked"]) + 1
        ),
        build_dataset=_build_dataset,
        build_runtime_config=lambda fit_params: (
            calls["runtime_cfg"].append(dict(fit_params))
            or {"bounds": {"gamma": [0.0, 1.0]}, "debug_logging": True}
        ),
    )

    assert result.error_text is None
    assert result.prepared_run is not None

    prepared = result.prepared_run
    assert prepared.fit_params["theta_offset"] == 0.5
    assert prepared.fit_params["theta_initial"] == 2.0
    assert prepared.selected_background_indices == [0, 1, 2]
    assert prepared.background_theta_values == [1.0, 2.0, 3.0]
    assert prepared.joint_background_mode is True
    assert prepared.dataset_specs == [
        {"dataset_index": 1},
        {"dataset_index": 0},
        {"dataset_index": 2},
    ]
    assert prepared.start_cmd_line == (
        "start: vars=gamma,theta_offset datasets=3 current_groups=1 current_points=2"
    )
    assert prepared.start_log_sections == [
        (
            "Fitting variables (start values):",
            [
                "gamma=nan",
                "theta_offset=0.500000",
            ],
        ),
        (
            "Manual geometry datasets:",
            ["bg[1]", "bg[0]", "bg[2]"],
        ),
        (
            "Current orientation diagnostics:",
            [
                "pairs=2",
                "chosen=identity",
                "identity_rms_px=nan",
                "best_rms_px=nan",
                "reason=n/a",
            ],
        ),
    ]
    assert prepared.max_display_markers == 150
    assert prepared.geometry_runtime_cfg == (
        geometry_fit.apply_manual_point_geometry_fit_runtime_overrides(
            geometry_fit.apply_joint_geometry_fit_runtime_safety_overrides(
                {"bounds": {"gamma": [0.0, 1.0]}, "debug_logging": True},
                joint_background_mode=True,
            ),
            joint_background_mode=True,
        )
    )
    assert [entry["dataset_index"] for entry in prepared.dataset_infos] == [1, 0, 2]
    assert prepared.current_dataset["dataset_index"] == 1

    assert calls["selection"] == [{"trigger_update": False, "sync_live_theta": True}]
    assert calls["theta"] == [{"trigger_update": False, "sync_live_theta": True}]
    assert calls["ensure_caked"] == 0
    assert calls["datasets"] == [
        (1, 2.0, 2.0, 0.5, {"mode": "auto"}),
        (0, 1.0, 2.0, 0.5, {"mode": "auto"}),
        (2, 3.0, 2.0, 0.5, {"mode": "auto"}),
    ]
    assert calls["runtime_cfg"] == [{"theta_initial": 2.0, "theta_offset": 0.5}]


def test_prepare_geometry_fit_run_rejects_selection_without_active_background() -> None:
    result = geometry_fit.prepare_geometry_fit_run(
        params={"theta_initial": 9.0},
        var_names=["gamma"],
        fit_config={},
        osc_files=["bg0.osc", "bg1.osc"],
        current_background_index=1,
        theta_initial=9.0,
        preserve_live_theta=True,
        apply_geometry_fit_background_selection=lambda **kwargs: True,
        current_geometry_fit_background_indices=lambda **kwargs: [0],
        geometry_fit_uses_shared_theta_offset=lambda *_args, **_kwargs: False,
        apply_background_theta_metadata=lambda **kwargs: True,
        current_background_theta_values=lambda **kwargs: [9.0],
        current_geometry_theta_offset=lambda **kwargs: 0.0,
        geometry_manual_pairs_for_index=lambda idx: [{"pair": idx}],
        ensure_geometry_fit_caked_view=lambda: None,
        build_dataset=lambda *_args, **_kwargs: {},
        build_runtime_config=lambda fit_params: dict(dict(fit_params), debug_logging=True),
    )

    assert result.prepared_run is None
    assert result.error_text is not None
    assert "active background must be part of the fit selection" in result.error_text


def test_prepare_geometry_fit_run_reports_missing_manual_pairs_by_background_name() -> None:
    result = geometry_fit.prepare_geometry_fit_run(
        params={"theta_initial": 4.0},
        var_names=["gamma"],
        fit_config={},
        osc_files=["C:/data/bg0.osc", "C:/data/bg1.osc"],
        current_background_index=1,
        theta_initial=4.0,
        preserve_live_theta=False,
        apply_geometry_fit_background_selection=lambda **kwargs: True,
        current_geometry_fit_background_indices=lambda **kwargs: [0, 1],
        geometry_fit_uses_shared_theta_offset=lambda indices: len(indices) > 1,
        apply_background_theta_metadata=lambda **kwargs: True,
        current_background_theta_values=lambda **kwargs: [3.0, 4.0],
        current_geometry_theta_offset=lambda **kwargs: 0.25,
        geometry_manual_pairs_for_index=lambda idx: [] if idx == 0 else [{"pair": idx}],
        ensure_geometry_fit_caked_view=lambda: None,
        build_dataset=lambda *_args, **_kwargs: {},
        build_runtime_config=lambda fit_params: dict(fit_params),
    )

    assert result.prepared_run is None
    assert result.error_text == (
        "Geometry fit unavailable: save manual Qr/Qz pairs first for bg0.osc."
    )


def test_prepare_geometry_fit_run_can_prepare_multi_background_datasets_without_fit_vars() -> None:
    built_indices: list[int] = []

    result = geometry_fit.prepare_geometry_fit_run(
        params={"theta_initial": 6.0, "theta_offset": 0.0},
        var_names=(),
        fit_config={},
        osc_files=["C:/data/bg0.osc", "C:/data/bg1.osc", "C:/data/bg2.osc"],
        current_background_index=2,
        theta_initial=6.0,
        preserve_live_theta=False,
        apply_geometry_fit_background_selection=lambda **kwargs: True,
        current_geometry_fit_background_indices=lambda **kwargs: [0, 1],
        geometry_fit_uses_shared_theta_offset=lambda *_args, **_kwargs: False,
        apply_background_theta_metadata=lambda **kwargs: True,
        current_background_theta_values=lambda **kwargs: [3.0, 4.0, 5.0],
        current_geometry_theta_offset=lambda **kwargs: 0.0,
        geometry_manual_pairs_for_index=lambda idx: [{"pair": idx}],
        ensure_geometry_fit_caked_view=lambda: None,
        build_dataset=lambda background_index, *, theta_base, base_fit_params, orientation_cfg: (
            built_indices.append(int(background_index))
            or {
                "dataset_index": int(background_index),
                "pair_count": 1,
                "group_count": 1,
                "summary_line": f"bg[{int(background_index)}]",
                "spec": {
                    "dataset_index": int(background_index),
                    "theta_initial": float(theta_base),
                },
                "orientation_choice": {"label": "identity"},
                "orientation_diag": {"pairs": 1},
                "measured_for_fit": [],
                "experimental_image_for_fit": f"image-{int(background_index)}",
                "initial_pairs_display": [],
                "native_background": np.zeros((2, 2)),
            }
        ),
        build_runtime_config=lambda fit_params: dict(fit_params),
        require_selected_var_names=False,
        require_active_background_in_selection=False,
        include_all_selected_backgrounds=True,
    )

    assert result.error_text is None
    assert result.prepared_run is not None
    assert result.prepared_run.selected_background_indices == [0, 1]
    assert result.prepared_run.current_dataset["dataset_index"] == 0
    assert result.prepared_run.dataset_specs == [
        {"dataset_index": 0, "theta_initial": 3.0},
        {"dataset_index": 1, "theta_initial": 4.0},
    ]
    assert result.prepared_run.joint_background_mode is False
    assert built_indices == [0, 1]


def test_prepare_geometry_fit_run_blocks_lattice_refinement_by_default() -> None:
    blocked = geometry_fit.prepare_geometry_fit_run(
        params={"theta_initial": 4.0, "a": 4.2},
        var_names=["a"],
        fit_config={"geometry": {"lattice_refinement": {"enabled": False}}},
        osc_files=["C:/data/bg0.osc"],
        current_background_index=0,
        theta_initial=4.0,
        preserve_live_theta=False,
        apply_geometry_fit_background_selection=lambda **kwargs: True,
        current_geometry_fit_background_indices=lambda **kwargs: [0],
        geometry_fit_uses_shared_theta_offset=lambda *_args, **_kwargs: False,
        apply_background_theta_metadata=lambda **kwargs: True,
        current_background_theta_values=lambda **kwargs: [4.0],
        current_geometry_theta_offset=lambda **kwargs: 0.0,
        geometry_manual_pairs_for_index=lambda idx: [{"pair": idx}],
        ensure_geometry_fit_caked_view=lambda: None,
        build_dataset=lambda *_args, **_kwargs: {},
        build_runtime_config=lambda fit_params: dict(fit_params),
    )

    assert blocked.prepared_run is None
    assert blocked.error_text == (
        "Geometry fit unavailable: lattice parameters (a) are frozen by default. "
        "Enable `fit.geometry.lattice_refinement.enabled` to refine them."
    )

    allowed = geometry_fit.prepare_geometry_fit_run(
        params={"theta_initial": 4.0, "a": 4.2},
        var_names=["a"],
        fit_config={"geometry": {"lattice_refinement": {"enabled": True}}},
        osc_files=["C:/data/bg0.osc"],
        current_background_index=0,
        theta_initial=4.0,
        preserve_live_theta=False,
        apply_geometry_fit_background_selection=lambda **kwargs: True,
        current_geometry_fit_background_indices=lambda **kwargs: [0],
        geometry_fit_uses_shared_theta_offset=lambda *_args, **_kwargs: False,
        apply_background_theta_metadata=lambda **kwargs: True,
        current_background_theta_values=lambda **kwargs: [4.0],
        current_geometry_theta_offset=lambda **kwargs: 0.0,
        geometry_manual_pairs_for_index=lambda idx: [{"pair": idx}],
        ensure_geometry_fit_caked_view=lambda: None,
        build_dataset=lambda background_index, **kwargs: {
            "dataset_index": int(background_index),
            "pair_count": 1,
            "group_count": 1,
            "summary_line": "bg[0]",
            "spec": {"dataset_index": int(background_index)},
            "orientation_choice": {"label": "identity"},
            "orientation_diag": {"pairs": 1},
            "measured_for_fit": [],
            "experimental_image_for_fit": "image-0",
            "initial_pairs_display": [],
            "native_background": np.zeros((2, 2)),
        },
        build_runtime_config=lambda fit_params: dict(
            dict(fit_params),
            debug_logging=True,
        ),
    )

    assert allowed.error_text is None
    assert allowed.prepared_run is not None
    assert any(
        title == "Fitting variables (start values):" and "a=4.200000" in lines
        for title, lines in allowed.prepared_run.start_log_sections
    )


def test_prepare_runtime_geometry_fit_run_builds_prepared_run_from_runtime_bindings() -> None:
    calls = {"ensure_caked": 0}
    native_background = np.arange(20, dtype=np.float64).reshape(4, 5)
    display_background = np.arange(42, dtype=np.float64).reshape(6, 7)
    orientation_choice = {
        "indexing_mode": "yx",
        "k": 1,
        "flip_x": False,
        "flip_y": True,
        "flip_order": "xy",
        "label": "rotate+flip",
    }
    orientation_diag = {"pairs": 1, "reason": "ok"}

    result = geometry_fit.prepare_runtime_geometry_fit_run(
        params={"theta_initial": 9.0, "theta_offset": 0.25},
        var_names=["gamma"],
        preserve_live_theta=False,
        bindings=geometry_fit.GeometryFitRuntimePreparationBindings(
            fit_config={
                "geometry": {
                    "orientation": {"mode": "auto"},
                    "auto_match": {"max_display_markers": 90},
                }
            },
            theta_initial=9.0,
            apply_geometry_fit_background_selection=lambda **kwargs: True,
            current_geometry_fit_background_indices=lambda **kwargs: [0],
            geometry_fit_uses_shared_theta_offset=lambda *_args, **_kwargs: False,
            apply_background_theta_metadata=lambda **kwargs: True,
            current_background_theta_values=lambda **kwargs: [9.0],
            current_geometry_theta_offset=lambda **kwargs: 0.0,
            ensure_geometry_fit_caked_view=lambda: calls.__setitem__(
                "ensure_caked", int(calls["ensure_caked"]) + 1
            ),
            manual_dataset_bindings=geometry_fit.GeometryFitRuntimeManualDatasetBindings(
                osc_files=["C:/tmp/bg0.osc"],
                current_background_index=0,
                image_size=100,
                display_rotate_k=3,
                geometry_manual_pairs_for_index=lambda idx: [
                    {
                        "q_group_key": ("q", 1),
                        "source_table_index": 1,
                        "source_row_index": 2,
                        "hkl": (1, 1, 0),
                        "x": 30.0,
                        "y": 40.0,
                    }
                ],
                load_background_by_index=(lambda idx: (native_background, display_background)),
                apply_background_backend_orientation=lambda image: None,
                geometry_manual_simulated_peaks_for_params=(
                    lambda params, *, prefer_cache: [{"dummy": True}]
                ),
                geometry_manual_simulated_lookup=lambda _peaks: {
                    (1, 2): {
                        "hkl": (1, 1, 0),
                        "q_group_key": ("q", 1),
                        "sim_col": 9.0,
                        "sim_row": 8.0,
                        "sim_col_raw": 9.0,
                        "sim_row_raw": 8.0,
                    }
                },
                geometry_manual_entry_display_coords=lambda entry: (50.0, 60.0),
                unrotate_display_peaks=(lambda entries, shape, *, k: [{"x": 30.0, "y": 40.0}]),
                display_to_native_sim_coords=lambda col, row, shape: (1.0, 2.0),
                select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
                    orientation_choice,
                    orientation_diag,
                ),
                apply_orientation_to_entries=lambda entries, shape, **kwargs: [
                    {"x": 31.0, "y": 41.0}
                ],
                orient_image_for_fit=lambda image, **kwargs: "fit-image",
            ),
            build_runtime_config=lambda fit_params: {
                "bounds": {"gamma": [0.0, 1.0]},
                "seen_theta": float(fit_params["theta_initial"]),
            },
        ),
    )

    assert result.error_text is None
    assert result.prepared_run is not None
    prepared = result.prepared_run
    assert prepared.current_dataset["label"] == "bg0.osc"
    assert "experimental_image_for_fit" not in prepared.current_dataset
    prepared_spec = dict(prepared.dataset_specs[0])
    assert prepared_spec.pop("manual_point_pairs") == prepared.current_dataset["manual_point_pairs"]
    assert [prepared_spec] == [
        {
            "dataset_index": 0,
            "label": "bg0.osc",
            "theta_initial": 9.0,
            "measured_peaks": [
                {
                    "x": 31.0,
                    "y": 41.0,
                    "display_col": 31.0,
                    "display_row": 41.0,
                    "fit_detector_x": 31.0,
                    "fit_detector_y": 41.0,
                    "detector_x": 31.0,
                    "detector_y": 41.0,
                    "detector_input_frame": "fit_detector",
                    "detector_input_frame_reason": "apply_orientation_to_entries",
                    "fit_source_identity_only": True,
                    "overlay_match_index": 0,
                    "pair_id": "bg0:pair0",
                    "q_group_key": ("q", 1),
                    "source_table_index": 1,
                    "source_row_index": 2,
                }
            ],
            "experimental_image": "fit-image",
            "dynamic_reanchor_callback": None,
            "dynamic_reanchor_enabled": False,
            "fit_space_projector": None,
            "fit_space_projector_kind": None,
            "fit_space_projector_unavailable_reason": "exact_caked_view_unavailable",
        }
    ]
    assert prepared.max_display_markers == 90
    assert prepared.geometry_runtime_cfg == (
        geometry_fit.apply_manual_point_geometry_fit_runtime_overrides(
            geometry_fit.apply_joint_geometry_fit_runtime_safety_overrides(
                {"bounds": {"gamma": [0.0, 1.0]}, "seen_theta": 9.0},
                joint_background_mode=False,
            ),
            joint_background_mode=False,
        )
    )
    assert calls["ensure_caked"] == 0


def test_build_geometry_manual_fit_dataset_assembles_orientation_ready_payload() -> None:
    calls: dict[str, object] = {}
    native_background = np.arange(20, dtype=np.float64).reshape(4, 5)
    display_background = np.arange(42, dtype=np.float64).reshape(6, 7)
    measured_native = [{"x": 30.0, "y": 40.0}]
    measured_for_fit = [{"x": 31.0, "y": 41.0}]
    orientation_choice = {
        "indexing_mode": "yx",
        "k": 1,
        "flip_x": False,
        "flip_y": True,
        "flip_order": "xy",
        "label": "rotate+flip",
    }
    orientation_diag = {"pairs": 1, "reason": "ok"}

    def _apply_background_backend_orientation(image):
        calls["backend_image"] = image
        return None

    def _simulated_peaks_for_params(params, *, prefer_cache):
        calls["sim_params"] = (dict(params), prefer_cache)
        return [{"dummy": True}]

    def _unrotate_display_peaks(entries, shape, *, k):
        calls["unrotate"] = (list(entries), tuple(shape), k)
        return measured_native

    def _display_to_native_sim_coords(col, row, shape):
        calls["display_to_native"] = (col, row, tuple(shape))
        return (1.0, 2.0)

    def _select_fit_orientation(sim_pts, meas_pts, shape, *, cfg):
        calls["select_orientation"] = (
            list(sim_pts),
            list(meas_pts),
            tuple(shape),
            dict(cfg),
        )
        return orientation_choice, orientation_diag

    def _apply_orientation_to_entries(entries, shape, **kwargs):
        calls["apply_orientation"] = (list(entries), tuple(shape), dict(kwargs))
        return measured_for_fit

    def _orient_image_for_fit(image, **kwargs):
        calls["orient_image"] = (image, dict(kwargs))
        return "fit-image"

    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=100,
        display_rotate_k=3,
        geometry_manual_pairs_for_index=lambda idx: [
            {
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
                "hkl": (1, 1, 0),
                "x": 30.0,
                "y": 40.0,
            }
        ],
        load_background_by_index=lambda idx: (native_background, display_background),
        apply_background_backend_orientation=_apply_background_backend_orientation,
        geometry_manual_simulated_peaks_for_params=_simulated_peaks_for_params,
        geometry_manual_simulated_lookup=lambda _simulated_peaks: {
            (1, 2): {
                "hkl": (1, 1, 0),
                "q_group_key": ("q", 1),
                "sim_col": 9.0,
                "sim_row": 8.0,
                "sim_col_raw": 9.0,
                "sim_row_raw": 8.0,
            }
        },
        geometry_manual_entry_display_coords=lambda entry: (50.0, 60.0),
        unrotate_display_peaks=_unrotate_display_peaks,
        display_to_native_sim_coords=_display_to_native_sim_coords,
        select_fit_orientation=_select_fit_orientation,
        apply_orientation_to_entries=_apply_orientation_to_entries,
        orient_image_for_fit=_orient_image_for_fit,
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.25},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={"mode": "auto"},
    )

    assert dataset["dataset_index"] == 0
    assert dataset["label"] == "bg0.osc"
    assert dataset["theta_base"] == 1.5
    assert dataset["theta_effective"] == 1.75
    assert dataset["group_count"] == 1
    assert dataset["pair_count"] == 1
    assert dataset["resolved_source_pair_count"] == 1
    assert dataset["initial_pairs_display"] == [
        {
            "overlay_match_index": 0,
            "pair_id": "bg0:pair0",
            "hkl": (1, 1, 0),
            "q_group_key": ("q", 1),
            "bg_display": (50.0, 60.0),
            "source_table_index": 1,
            "source_row_index": 2,
            "selected_source_identity_canonical": {
                "normalized_hkl": [1, 1, 0],
                "source_table_index": 1,
                "source_row_index": 2,
                "q_group_key": ["q", 1],
            },
            "sim_display": (9.0, 8.0),
            "provider_simulated_frame": "display",
            "provider_simulated_point_source": "live_source_row_projection",
            "bg_native": (30.0, 40.0),
            "sim_native": (1.0, 2.0),
        }
    ]
    assert dataset["measured_native"] == measured_native
    assert dataset["measured_for_fit"] == measured_for_fit
    assert "experimental_image_for_fit" not in dataset
    assert dataset["orientation_choice"] == orientation_choice
    assert dataset["orientation_diag"] == orientation_diag
    dataset_spec = dict(dataset["spec"])
    assert dataset_spec.pop("manual_point_pairs") == dataset["manual_point_pairs"]
    assert dataset_spec == {
        "dataset_index": 0,
        "label": "bg0.osc",
        "theta_initial": 1.5,
        "measured_peaks": measured_for_fit,
        "experimental_image": "fit-image",
        "dynamic_reanchor_callback": None,
        "dynamic_reanchor_enabled": False,
        "fit_space_projector": None,
        "fit_space_projector_kind": None,
        "fit_space_projector_unavailable_reason": "exact_caked_view_unavailable",
    }
    assert dataset["measured_for_fit"] == [
        {
            "x": 31.0,
            "y": 41.0,
            "display_col": 31.0,
            "display_row": 41.0,
            "fit_detector_x": 31.0,
            "fit_detector_y": 41.0,
            "detector_x": 31.0,
            "detector_y": 41.0,
            "detector_input_frame": "fit_detector",
            "detector_input_frame_reason": "apply_orientation_to_entries",
            "fit_source_identity_only": True,
            "overlay_match_index": 0,
            "pair_id": "bg0:pair0",
            "q_group_key": ("q", 1),
            "source_table_index": 1,
            "source_row_index": 2,
        }
    ]
    assert "orientation=rotate+flip" in dataset["summary_line"]

    sim_params, prefer_cache = calls["sim_params"]
    assert sim_params["theta_initial"] == 1.75
    assert prefer_cache is True
    assert calls["unrotate"][1:] == ((6, 7), 3)
    assert calls["display_to_native"] == (9.0, 8.0, (100, 100))
    assert calls["backend_image"] is native_background
    assert calls["orient_image"] == (
        native_background,
        {
            "indexing_mode": "yx",
            "k": 1,
            "flip_x": False,
            "flip_y": True,
            "flip_order": "xy",
        },
    )
    assert calls["select_orientation"] == (
        [(1.0, 2.0)],
        [(30.0, 40.0)],
        (4, 5),
        {"mode": "auto"},
    )


def test_build_geometry_manual_fit_dataset_preserves_multiple_entries_per_q_group() -> None:
    calls: dict[str, object] = {}
    orientation_choice = {
        "indexing_mode": "xy",
        "k": 0,
        "flip_x": False,
        "flip_y": False,
        "flip_order": "yx",
        "label": "identity",
    }

    def _entry_display_coords(entry):
        if tuple(entry.get("hkl", ())) == (1, 1, 0):
            return (50.0, 60.0)
        return (100.0, 100.0)

    def _select_fit_orientation(sim_pts, meas_pts, shape, *, cfg):
        calls["select_orientation"] = (
            list(sim_pts),
            list(meas_pts),
            tuple(shape),
            dict(cfg),
        )
        return orientation_choice, {"pairs": len(sim_pts)}

    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=100,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda idx: [
            {
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
                "hkl": (1, 1, 0),
                "x": 30.0,
                "y": 40.0,
            },
            {
                "q_group_key": ("q", 1),
                "source_table_index": 2,
                "source_row_index": 4,
                "hkl": (-1, 1, 0),
                "x": 70.0,
                "y": 80.0,
            },
        ],
        load_background_by_index=lambda idx: (
            np.zeros((4, 5), dtype=np.float64),
            np.zeros((4, 5), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_simulated_peaks_for_params=lambda params, *, prefer_cache: [
            {"dummy": True}
        ],
        geometry_manual_simulated_lookup=lambda _peaks: {
            (1, 2): {
                "hkl": (1, 1, 0),
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
                "sim_col": 50.0,
                "sim_row": 60.0,
                "sim_col_raw": 50.0,
                "sim_row_raw": 60.0,
            },
            (2, 4): {
                "hkl": (-1, 1, 0),
                "q_group_key": ("q", 1),
                "source_table_index": 2,
                "source_row_index": 4,
                "sim_col": 10.0,
                "sim_row": 15.0,
                "sim_col_raw": 10.0,
                "sim_row_raw": 15.0,
            },
        },
        geometry_manual_entry_display_coords=_entry_display_coords,
        unrotate_display_peaks=lambda entries, shape, *, k: [dict(entry) for entry in entries],
        display_to_native_sim_coords=lambda col, row, shape: (float(col) / 10.0, float(row) / 10.0),
        select_fit_orientation=_select_fit_orientation,
        apply_orientation_to_entries=lambda entries, shape, **kwargs: [
            dict(entry) for entry in entries
        ],
        orient_image_for_fit=lambda image, **kwargs: image,
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={"mode": "auto"},
    )

    assert dataset["group_count"] == 1
    assert dataset["pair_count"] == 2
    assert len(dataset["measured_display"]) == 2
    assert len(dataset["initial_pairs_display"]) == 2
    assert [entry["hkl"] for entry in dataset["initial_pairs_display"]] == [
        (1, 1, 0),
        (-1, 1, 0),
    ]
    assert [entry["sim_display"] for entry in dataset["initial_pairs_display"]] == [
        (50.0, 60.0),
        (10.0, 15.0),
    ]
    assert [entry["q_group_key"] for entry in dataset["measured_for_fit"]] == [
        ("q", 1),
        ("q", 1),
    ]
    assert [entry["source_table_index"] for entry in dataset["measured_for_fit"]] == [
        1,
        2,
    ]
    assert [entry["source_row_index"] for entry in dataset["measured_for_fit"]] == [
        2,
        4,
    ]
    assert all(entry["fit_source_identity_only"] is True for entry in dataset["measured_for_fit"])
    assert calls["select_orientation"] == (
        [(5.0, 6.0), (1.0, 1.5)],
        [(30.0, 40.0), (70.0, 80.0)],
        (4, 5),
        {"mode": "auto"},
    )


def test_build_geometry_manual_fit_dataset_preserves_branch_and_full_reflection_provenance() -> (
    None
):
    source_row = {
        "hkl": (1, 1, 0),
        "q_group_key": ("q", 1),
        "source_table_index": 1,
        "source_reflection_index": 7,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_row_index": 2,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "sim_col": 9.0,
        "sim_row": 8.0,
    }

    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=100,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda idx: [
            {
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_reflection_index": 7,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "source_row_index": 2,
                "source_branch_index": 0,
                "source_peak_index": 0,
                "hkl": (1, 1, 0),
                "x": 30.0,
                "y": 40.0,
            }
        ],
        load_background_by_index=lambda idx: (
            np.zeros((4, 5), dtype=np.float64),
            np.zeros((4, 5), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_simulated_peaks_for_params=lambda params, *, prefer_cache: [
            dict(source_row)
        ],
        geometry_manual_simulated_lookup=lambda _simulated_peaks: {(1, 2): dict(source_row)},
        geometry_manual_entry_display_coords=lambda entry: (30.0, 40.0),
        unrotate_display_peaks=lambda entries, shape, *, k: [dict(entry) for entry in entries],
        display_to_native_sim_coords=lambda col, row, shape: (float(col), float(row)),
        select_fit_orientation=lambda *args, **kwargs: (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "xy",
                "label": "identity",
            },
            {"pairs": 1},
        ),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: [
            dict(entry) for entry in entries
        ],
        orient_image_for_fit=lambda image, **kwargs: image,
        geometry_manual_source_rows_for_background=lambda *args, **kwargs: [dict(source_row)],
        geometry_manual_rebuild_source_rows_for_background=lambda *args, **kwargs: [
            dict(source_row)
        ],
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.25},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={},
    )

    measured_entry = dataset["measured_for_fit"][0]
    initial_entry = dataset["initial_pairs_display"][0]

    assert measured_entry["pair_id"] == "bg0:pair0"
    assert measured_entry["source_reflection_index"] == 7
    assert measured_entry["source_reflection_namespace"] == "full_reflection"
    assert measured_entry["source_reflection_is_full"] is True
    assert measured_entry["source_branch_index"] == 0
    assert measured_entry["source_peak_index"] == 0
    assert initial_entry["pair_id"] == "bg0:pair0"
    assert initial_entry["source_reflection_index"] == 7
    assert initial_entry["source_reflection_namespace"] == "full_reflection"
    assert initial_entry["source_reflection_is_full"] is True
    assert initial_entry["source_branch_index"] == 0
    assert dataset["source_rows_for_trace"][0]["source_reflection_index"] == 7
    assert dataset["source_rows_for_trace"][0]["source_peak_index"] == 0
    diag = dataset["source_resolution_diagnostics"][0]
    assert diag["fit_resolution_kind"] in {"source_row", "source_peak"}
    assert diag["raw_saved_entry"]["source_reflection_index"] == 7
    assert diag["normalized_saved_entry"]["source_reflection_index"] == 7


def _refresh_legacy_dense_pair_entry(entry):
    refreshed = dict(entry)
    caked_y = refreshed.get(
        "refined_sim_caked_y",
        refreshed.get("caked_y", refreshed.get("raw_caked_y")),
    )
    try:
        signed_value = float(caked_y)
    except Exception:
        return refreshed
    if not np.isfinite(signed_value) or abs(float(signed_value)) <= 1.0e-3:
        refreshed.pop("source_branch_index", None)
        return refreshed
    branch_idx = 0 if float(signed_value) < 0.0 else 1
    refreshed["source_branch_index"] = int(branch_idx)
    refreshed["source_peak_index"] = int(branch_idx)
    return refreshed


def _make_legacy_dense_manual_dataset_bindings(
    *,
    saved_entries,
    simulated_rows,
    refresh_pairs=True,
):
    def _lookup(_simulated_peaks):
        lookup = {}
        for row in simulated_rows:
            try:
                table_idx = int(row.get("source_table_index"))
                row_idx = int(row.get("source_row_index"))
            except Exception:
                continue
            lookup[(table_idx, row_idx)] = dict(row)
        return lookup

    def _rows(*args, **kwargs):
        return [dict(row) for row in simulated_rows]

    return geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=64,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda idx: [dict(entry) for entry in saved_entries],
        load_background_by_index=lambda idx: (
            np.zeros((4, 5), dtype=np.float64),
            np.zeros((4, 5), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_simulated_peaks_for_params=lambda params, *, prefer_cache: [
            dict(row) for row in simulated_rows
        ],
        geometry_manual_simulated_lookup=_lookup,
        geometry_manual_entry_display_coords=lambda entry: (
            float(entry.get("x", 0.0)),
            float(entry.get("y", 0.0)),
        ),
        unrotate_display_peaks=lambda entries, shape, *, k: [dict(entry) for entry in entries],
        display_to_native_sim_coords=lambda col, row, shape: (float(col), float(row)),
        select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "yx",
                "label": "identity",
            },
            {"pairs": len(sim_pts)},
        ),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: [
            dict(entry) for entry in entries
        ],
        orient_image_for_fit=lambda image, **kwargs: image,
        geometry_manual_source_rows_for_background=_rows,
        geometry_manual_rebuild_source_rows_for_background=_rows,
        geometry_manual_refresh_pair_entry=(
            _refresh_legacy_dense_pair_entry if refresh_pairs else None
        ),
    )


def _load_geometry_preflight_probe_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "debug"
        / "validate_geometry_preflight_rebind.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_geometry_preflight_rebind",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_new4_ladder_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "debug"
        / "run_new4_geometry_fit_ladder.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_new4_geometry_fit_ladder",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _required_new4_local_headless_paths(state_path: Path) -> list[Path]:
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    state = payload.get("state", {}) if isinstance(payload, dict) else {}
    files = state.get("files", {}) if isinstance(state, dict) else {}
    if not isinstance(files, dict):
        return []

    paths: list[Path] = []
    background_files = files.get("background_files", [])
    if isinstance(background_files, list):
        paths.extend(Path(str(path)) for path in background_files if path)
    for key in ("primary_cif_path", "secondary_cif_path"):
        path = files.get(key)
        if path:
            paths.append(Path(str(path)))
    return paths


def _missing_new4_local_headless_paths(state_path: Path) -> list[Path]:
    return [path for path in _required_new4_local_headless_paths(state_path) if not path.exists()]


def _write_probe_state(
    tmp_path,
    *,
    entries,
    peak_records=None,
    q_group_rows=None,
):
    state_path = tmp_path / "probe_state.json"
    save_gui_state_file(
        state_path,
        {
            "files": {
                "background_files": ["C:/tmp/bg0.osc"],
                "current_background_index": 0,
            },
            "variables": {
                "geometry_fit_background_selection_var": "current",
            },
            "geometry": {
                "manual_pairs": [
                    {
                        "background_index": 0,
                        "background_name": "bg0",
                        "background_path": "C:/tmp/bg0.osc",
                        "entries": [dict(entry) for entry in entries],
                    }
                ],
                "peak_records": list(peak_records or []),
                "q_group_rows": list(q_group_rows or []),
            },
        },
    )
    return state_path


def test_build_geometry_manual_fit_dataset_uses_raw_sim_display_for_native_coords() -> None:
    calls: dict[str, object] = {}

    def _display_to_native_sim_coords(col, row, shape):
        calls["display_to_native"] = (col, row, tuple(shape))
        return (11.0, 12.0)

    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=64,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda idx: [
            {
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
                "hkl": (1, 1, 0),
                "x": 30.0,
                "y": 40.0,
            }
        ],
        load_background_by_index=lambda idx: (
            np.zeros((4, 5), dtype=np.float64),
            np.zeros((4, 5), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_simulated_peaks_for_params=lambda params, *, prefer_cache: [
            {"dummy": True}
        ],
        geometry_manual_simulated_lookup=lambda _simulated_peaks: {
            (1, 2): {
                "display_col": 901.0,
                "display_row": -802.0,
                "sim_col": 91.0,
                "sim_row": 82.0,
                "sim_col_raw": 9.0,
                "sim_row_raw": 8.0,
                "caked_x": 91.0,
                "caked_y": 82.0,
                "hkl": (1, 1, 0),
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
            }
        },
        geometry_manual_entry_display_coords=lambda entry: (30.0, 40.0),
        unrotate_display_peaks=lambda entries, shape, *, k: [{"x": 30.0, "y": 40.0}],
        display_to_native_sim_coords=_display_to_native_sim_coords,
        select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "yx",
                "label": "identity",
            },
            {"pairs": 1},
        ),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: list(entries),
        orient_image_for_fit=lambda image, **kwargs: image,
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={},
    )

    assert dataset["initial_pairs_display"][0]["sim_display"] == (91.0, 82.0)
    assert dataset["initial_pairs_display"][0]["sim_native"] == (11.0, 12.0)
    assert calls["display_to_native"] == (9.0, 8.0, (64, 64))


def test_build_geometry_manual_fit_dataset_projects_raw_sim_image_rows_into_background_frame() -> (
    None
):
    callbacks = manual_geometry.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: np.zeros((5, 5), dtype=float),
        last_caked_radial_values=lambda: np.array([], dtype=float),
        last_caked_azimuth_values=lambda: np.array([], dtype=float),
        current_background_display=lambda: np.zeros((5, 5), dtype=float),
        current_background_native=lambda: np.ones((5, 5), dtype=float),
        image_size=lambda: 5,
        display_rotate_k=3,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        detector_pixel_to_scattering_angles=lambda *_args: (_ for _ in ()).throw(
            AssertionError("analytic forward fallback should not be used")
        ),
    )
    expected_sim = manual_geometry._default_rotate_point(1.0, 2.0, (5, 5), 3)

    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=5,
        display_rotate_k=3,
        geometry_manual_pairs_for_index=lambda idx: [
            {
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
                "hkl": (1, 1, 0),
                "x": float(expected_sim[0]) + 0.25,
                "y": float(expected_sim[1]) + 0.25,
            }
        ],
        load_background_by_index=lambda idx: (
            np.zeros((5, 5), dtype=np.float64),
            np.zeros((5, 5), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_source_rows_for_background=lambda *args, **kwargs: [
            {
                "display_col": 400.0,
                "display_row": -500.0,
                "sim_col": 300.0,
                "sim_row": -200.0,
                "sim_col_raw": 1.0,
                "sim_row_raw": 2.0,
                "hkl": (1, 1, 0),
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
            }
        ],
        geometry_manual_simulated_peaks_for_params=lambda params, *, prefer_cache: [],
        geometry_manual_simulated_lookup=lambda _simulated_peaks: {},
        geometry_manual_entry_display_coords=lambda entry: (
            float(entry.get("x", 0.0)),
            float(entry.get("y", 0.0)),
        ),
        geometry_manual_project_peaks_to_current_view=callbacks.project_peaks_to_current_view,
        unrotate_display_peaks=lambda entries, shape, *, k: [dict(entry) for entry in entries],
        display_to_native_sim_coords=lambda col, row, shape: (float(col), float(row)),
        select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "yx",
                "label": "identity",
            },
            {"pairs": len(sim_pts)},
        ),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: list(entries),
        orient_image_for_fit=lambda image, **kwargs: image,
        pick_uses_caked_space=lambda: False,
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={},
    )

    assert dataset["initial_pairs_display"][0]["sim_display"] == (
        float(expected_sim[0]),
        float(expected_sim[1]),
    )
    assert dataset["initial_pairs_display"][0]["sim_native"] == (1.0, 2.0)


def test_build_geometry_manual_fit_dataset_reprojects_refined_detector_display_without_stale_native() -> (
    None
):
    refined_native = (2.0, 1.0)
    refined_display = manual_geometry._default_rotate_point(
        float(refined_native[0]),
        float(refined_native[1]),
        (5, 5),
        3,
    )
    callbacks = manual_geometry.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: np.zeros((5, 5), dtype=float),
        last_caked_radial_values=lambda: np.array([], dtype=float),
        last_caked_azimuth_values=lambda: np.array([], dtype=float),
        current_background_display=lambda: np.zeros((5, 5), dtype=float),
        current_background_native=lambda: np.ones((5, 5), dtype=float),
        image_size=lambda: 5,
        display_rotate_k=3,
        display_to_native_sim_coords=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("refined detector display should use background inverse")
        ),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        detector_pixel_to_scattering_angles=lambda *_args: (_ for _ in ()).throw(
            AssertionError("analytic forward fallback should not be used")
        ),
    )

    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=5,
        display_rotate_k=3,
        geometry_manual_pairs_for_index=lambda idx: [
            {
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
                "hkl": (1, 1, 0),
                "x": float(refined_display[0]) + 0.25,
                "y": float(refined_display[1]) + 0.25,
                "refined_sim_x": float(refined_display[0]),
                "refined_sim_y": float(refined_display[1]),
            }
        ],
        load_background_by_index=lambda idx: (
            np.zeros((5, 5), dtype=np.float64),
            np.zeros((5, 5), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_source_rows_for_background=lambda *args, **kwargs: [
            {
                "display_col": 400.0,
                "display_row": -500.0,
                "sim_col": 300.0,
                "sim_row": -200.0,
                "sim_col_raw": 99.0,
                "sim_row_raw": 88.0,
                "native_col": 1.0,
                "native_row": 2.0,
                "sim_native_x": 1.0,
                "sim_native_y": 2.0,
                "hkl": (1, 1, 0),
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
            }
        ],
        geometry_manual_simulated_peaks_for_params=lambda params, *, prefer_cache: [],
        geometry_manual_simulated_lookup=lambda _simulated_peaks: {},
        geometry_manual_entry_display_coords=lambda entry: (
            float(entry.get("x", 0.0)),
            float(entry.get("y", 0.0)),
        ),
        geometry_manual_project_peaks_to_current_view=callbacks.project_peaks_to_current_view,
        unrotate_display_peaks=lambda entries, shape, *, k: [dict(entry) for entry in entries],
        display_to_native_sim_coords=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fit fallback should not use sim-image inverse here")
        ),
        select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "yx",
                "label": "identity",
            },
            {"pairs": len(sim_pts)},
        ),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: list(entries),
        orient_image_for_fit=lambda image, **kwargs: image,
        pick_uses_caked_space=lambda: False,
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={},
    )

    assert dataset["initial_pairs_display"][0]["sim_display"] == (
        float(refined_display[0]),
        float(refined_display[1]),
    )
    assert dataset["initial_pairs_display"][0]["sim_native"] == (
        float(refined_native[0]),
        float(refined_native[1]),
    )


def test_build_geometry_manual_fit_dataset_does_not_count_stale_source_ids_as_resolved() -> None:
    unrelated_source_row = {
        "q_group_key": ("other", 9),
        "source_table_index": 1,
        "source_row_index": 2,
        "source_peak_index": 0,
        "hkl": (9, 9, 9),
        "sim_col": 11.0,
        "sim_row": 12.0,
    }
    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=64,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda idx: [
            {
                "q_group_key": ("q", 1),
                "source_table_index": 7,
                "source_row_index": 9,
                "hkl": (1, 1, 0),
                "x": 30.0,
                "y": 40.0,
            }
        ],
        load_background_by_index=lambda idx: (
            np.zeros((4, 5), dtype=np.float64),
            np.zeros((4, 5), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_simulated_peaks_for_params=(
            lambda params, *, prefer_cache: [dict(unrelated_source_row)]
        ),
        geometry_manual_simulated_lookup=lambda _simulated_peaks: {
            (1, 2): dict(unrelated_source_row)
        },
        geometry_manual_entry_display_coords=lambda entry: (50.0, 60.0),
        unrotate_display_peaks=lambda entries, shape, *, k: [dict(entry) for entry in entries],
        display_to_native_sim_coords=lambda col, row, shape: (float(col), float(row)),
        select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "yx",
                "label": "identity",
            },
            {"pairs": len(sim_pts)},
        ),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: [
            dict(entry) for entry in entries
        ],
        orient_image_for_fit=lambda image, **kwargs: image,
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={"mode": "auto"},
    )

    assert dataset["resolved_source_pair_count"] == 0
    assert "source_table_index" not in dataset["measured_for_fit"][0]
    assert "source_row_index" not in dataset["measured_for_fit"][0]
    assert "source_peak_index" not in dataset["measured_for_fit"][0]


def test_build_geometry_manual_fit_dataset_copyback_preserves_canonical_identity_after_legacy_dense_rebind() -> (
    None
):
    saved_entries = [
        {
            "pair_id": "bg0:pair0",
            "q_group_key": ("q", 1),
            "source_table_index": 7,
            "source_reflection_index": 7,
            "source_row_index": 0,
            "source_peak_index": 7,
            "hkl": (1, 1, 0),
            "x": 48.0,
            "y": 59.0,
            "caked_y": 12.0,
            "refined_sim_caked_y": 12.0,
            "refined_sim_x": 50.0,
            "refined_sim_y": 60.0,
        }
    ]
    simulated_rows = [
        {
            "q_group_key": ("q", 1),
            "source_table_index": 0,
            "source_reflection_index": 200,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (1, 1, 0),
            "sim_col": 50.0,
            "sim_row": 60.0,
        }
    ]
    manual_dataset_bindings = _make_legacy_dense_manual_dataset_bindings(
        saved_entries=saved_entries,
        simulated_rows=simulated_rows,
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={"mode": "auto"},
    )

    measured_entry = dataset["measured_for_fit"][0]
    initial_entry = dataset["initial_pairs_display"][0]
    diag = dataset["source_resolution_diagnostics"][0]

    assert dataset["resolved_source_pair_count"] == 1
    for payload in (measured_entry, initial_entry):
        assert payload["pair_id"] == "bg0:pair0"
        assert payload["hkl"] == (1, 1, 0)
        assert payload["source_reflection_index"] == 200
        assert payload["source_reflection_namespace"] == "full_reflection"
        assert payload["source_reflection_is_full"] is True
        assert payload["source_branch_index"] == 1
        assert payload["source_peak_index"] == 1
    assert diag["strict_resolved"] is False
    assert diag["fit_resolved"] is True
    assert diag["fit_resolution_kind"] == "legacy_dense_q_group_rebind"
    assert diag["failure_reason"] is None
    assert diag["raw_saved_entry"]["source_reflection_index"] == 7
    assert diag["raw_saved_entry"]["source_peak_index"] == 7
    assert diag["normalized_saved_entry"]["source_peak_index"] == 1
    assert diag["legacy_branch_hint_source"] == "refined_sim_caked_y"
    assert diag["legacy_fit_bound_entry"]["source_reflection_index"] == 200
    assert diag["legacy_fit_bound_entry"]["source_branch_index"] == 1


def test_build_geometry_manual_fit_dataset_rebinds_mirrored_legacy_dense_pairs_by_branch_and_geometry() -> (
    None
):
    saved_entries = [
        {
            "pair_id": "bg0:pair0",
            "q_group_key": ("q", 5),
            "source_table_index": 12,
            "source_reflection_index": 12,
            "source_row_index": 0,
            "source_peak_index": 12,
            "hkl": (-1, 0, 5),
            "x": 110.0,
            "y": 100.0,
            "caked_y": -8.0,
            "refined_sim_caked_y": -8.0,
            "refined_sim_x": 110.0,
            "refined_sim_y": 100.0,
        },
        {
            "pair_id": "bg0:pair1",
            "q_group_key": ("q", 5),
            "source_table_index": 13,
            "source_reflection_index": 13,
            "source_row_index": 0,
            "source_peak_index": 13,
            "hkl": (-1, 0, 5),
            "x": 189.5,
            "y": 97.0,
            "caked_y": 9.0,
            "refined_sim_caked_y": 9.0,
            "refined_sim_x": 189.5,
            "refined_sim_y": 97.0,
        },
    ]
    simulated_rows = [
        {
            "q_group_key": ("q", 5),
            "source_table_index": 0,
            "source_reflection_index": 202,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "hkl": (-1, 0, 5),
            "sim_col": 110.0,
            "sim_row": 100.0,
        },
        {
            "q_group_key": ("q", 5),
            "source_table_index": 1,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "sim_col": 182.0,
            "sim_row": 98.0,
        },
        {
            "q_group_key": ("q", 5),
            "source_table_index": 2,
            "source_reflection_index": 204,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "sim_col": 190.0,
            "sim_row": 96.0,
        },
    ]
    manual_dataset_bindings = _make_legacy_dense_manual_dataset_bindings(
        saved_entries=saved_entries,
        simulated_rows=simulated_rows,
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={"mode": "auto"},
    )

    assert dataset["resolved_source_pair_count"] == 2
    measured_entries = dataset["measured_for_fit"]
    initial_entries = dataset["initial_pairs_display"]
    assert [entry["source_reflection_index"] for entry in measured_entries] == [202, 204]
    assert [entry["source_branch_index"] for entry in measured_entries] == [0, 1]
    assert [entry["source_peak_index"] for entry in measured_entries] == [0, 1]
    assert [entry["source_reflection_index"] for entry in initial_entries] == [202, 204]
    assert [entry["source_branch_index"] for entry in initial_entries] == [0, 1]
    assert all(
        entry["source_reflection_namespace"] == "full_reflection" for entry in measured_entries
    )
    assert all(entry["source_reflection_is_full"] is True for entry in measured_entries)
    diag0, diag1 = dataset["source_resolution_diagnostics"]
    assert diag0["fit_resolution_kind"] == "legacy_dense_q_group_rebind"
    assert diag0["legacy_branch_hint_source"] == "refined_sim_caked_y"
    assert diag1["fit_resolution_kind"] == "legacy_dense_q_group_rebind"
    assert diag1["legacy_branch_hint_source"] == "refined_sim_caked_y"
    assert diag1["legacy_geometry_hint_source"] == "measured_display"
    assert diag1["legacy_candidate_count_initial"] == 3
    assert diag1["legacy_candidate_count_after_branch"] == 2
    assert diag1["legacy_chosen_live_row"]["source_reflection_index"] == 204


def test_build_geometry_manual_fit_dataset_prefers_background_nearest_candidate_over_stale_identity() -> (
    None
):
    saved_entries = [
        {
            "pair_id": "bg0:pair0",
            "q_group_key": ("q", 5),
            "source_table_index": 203,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 203,
            "hkl": (-1, 0, 5),
            "x": 1000.0,
            "y": 1000.0,
            "caked_y": 9.0,
            "refined_sim_caked_y": 9.0,
            "refined_sim_x": 2500.0,
            "refined_sim_y": 2500.0,
        }
    ]
    simulated_rows = [
        {
            "q_group_key": ("q", 5),
            "source_table_index": 1,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "sim_col": 2504.0,
            "sim_row": 2497.0,
        },
        {
            "q_group_key": ("q", 5),
            "source_table_index": 2,
            "source_reflection_index": 204,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "sim_col": 1003.0,
            "sim_row": 998.0,
        },
    ]
    manual_dataset_bindings = _make_legacy_dense_manual_dataset_bindings(
        saved_entries=saved_entries,
        simulated_rows=simulated_rows,
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={"mode": "auto"},
    )

    assert dataset["resolved_source_pair_count"] == 1
    measured_entry = dataset["measured_for_fit"][0]
    diag = dataset["source_resolution_diagnostics"][0]
    assert measured_entry["source_reflection_index"] == 204
    assert measured_entry["source_branch_index"] == 1
    assert diag["fit_resolution_kind"] == "legacy_dense_q_group_rebind"
    assert diag["legacy_geometry_hint_source"] == "measured_display"
    assert diag["selected_candidate_source_identity_fields"]["source_reflection_index"] == 204
    assert diag["selected_live_simulated_current_view_point"] == (1003.0, 998.0)
    assert diag["selected_to_background_distance_px"] == pytest.approx((3.0**2 + 2.0**2) ** 0.5)
    assert diag["saved_simulated_detector_hint"] == (2500.0, 2500.0)


def test_build_geometry_manual_fit_dataset_rebinds_nonlegacy_stale_source_by_background_point() -> (
    None
):
    saved_entries = [
        {
            "pair_id": "bg0:pair0",
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 999,
            "source_reflection_index": 999,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "x": 1000.0,
            "y": 1000.0,
            "refined_sim_x": 2500.0,
            "refined_sim_y": 2500.0,
        }
    ]
    simulated_rows = [
        {
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 1,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "sim_col": 2504.0,
            "sim_row": 2497.0,
        },
        {
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 2,
            "source_reflection_index": 204,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "sim_col": 1003.0,
            "sim_row": 998.0,
        },
    ]
    manual_dataset_bindings = _make_legacy_dense_manual_dataset_bindings(
        saved_entries=saved_entries,
        simulated_rows=simulated_rows,
        refresh_pairs=False,
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={"mode": "auto"},
    )

    assert dataset["resolved_source_pair_count"] == 1
    measured_entry = dataset["measured_for_fit"][0]
    diag = dataset["source_resolution_diagnostics"][0]
    assert measured_entry["source_reflection_index"] == 204
    assert measured_entry["source_branch_index"] == 1
    assert diag["strict_resolved"] is False
    assert diag["fit_resolved"] is True
    assert diag["fit_resolution_kind"] == "q_group_fallback"
    assert diag["selected_candidate_source_identity_fields"]["source_reflection_index"] == 204
    assert diag["selected_live_simulated_current_view_point"] == (1003.0, 998.0)
    assert diag["selected_to_background_distance_px"] == pytest.approx((3.0**2 + 2.0**2) ** 0.5)


def test_build_geometry_manual_fit_dataset_prefers_nearest_candidate_over_existing_cached_identity() -> (
    None
):
    saved_entries = [
        {
            "pair_id": "bg0:pair0",
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 1,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "x": 1000.0,
            "y": 1000.0,
            "refined_sim_x": 2500.0,
            "refined_sim_y": 2500.0,
        }
    ]
    simulated_rows = [
        {
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 1,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "sim_col": 2504.0,
            "sim_row": 2497.0,
        },
        {
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 2,
            "source_reflection_index": 204,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "sim_col": 1003.0,
            "sim_row": 998.0,
        },
    ]
    manual_dataset_bindings = _make_legacy_dense_manual_dataset_bindings(
        saved_entries=saved_entries,
        simulated_rows=simulated_rows,
        refresh_pairs=False,
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={"mode": "auto"},
    )

    assert dataset["resolved_source_pair_count"] == 1
    measured_entry = dataset["measured_for_fit"][0]
    initial_entry = dataset["initial_pairs_display"][0]
    diag = dataset["source_resolution_diagnostics"][0]
    assert measured_entry["source_reflection_index"] == 204
    assert initial_entry["source_reflection_index"] == 204
    assert diag["strict_resolved"] is True
    assert diag["fit_resolved"] is True
    assert diag["fit_resolution_kind"] == "q_group_fallback"
    assert diag["selected_candidate_source_identity_fields"]["source_reflection_index"] == 204
    assert diag["selected_live_simulated_current_view_point"] == (1003.0, 998.0)
    assert diag["selected_to_background_distance_px"] == pytest.approx((3.0**2 + 2.0**2) ** 0.5)


def test_build_geometry_manual_fit_dataset_rebinds_by_branch_group_key_before_distance() -> None:
    saved_entries = [
        {
            "pair_id": "bg0:pair0",
            "branch_group_key": ("branch", "wanted"),
            "source_table_index": 1,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "x": 1000.0,
            "y": 1000.0,
            "refined_sim_x": 1000.0,
            "refined_sim_y": 1000.0,
        }
    ]
    simulated_rows = [
        {
            "branch_group_key": ("branch", "wrong"),
            "source_table_index": 1,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "sim_col": 1000.0,
            "sim_row": 1000.0,
        },
        {
            "branch_group_key": ("branch", "wanted"),
            "source_table_index": 2,
            "source_reflection_index": 204,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "sim_col": 1300.0,
            "sim_row": 1300.0,
        },
    ]
    manual_dataset_bindings = _make_legacy_dense_manual_dataset_bindings(
        saved_entries=saved_entries,
        simulated_rows=simulated_rows,
        refresh_pairs=False,
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={"mode": "auto"},
    )

    assert dataset["resolved_source_pair_count"] == 1
    measured_entry = dataset["measured_for_fit"][0]
    diag = dataset["source_resolution_diagnostics"][0]
    assert measured_entry["source_reflection_index"] == 204
    assert measured_entry["source_branch_index"] == 1
    assert measured_entry["branch_group_key"] == ("branch", "wanted")
    assert diag["fit_resolution_kind"] == "group_fallback"
    assert diag["selected_candidate_source_identity_fields"]["source_reflection_index"] == 204
    assert diag["selected_live_simulated_current_view_point"] == (1300.0, 1300.0)


def _point_provider_optimizer_guard(monkeypatch):
    state = {
        "optimizer_guard_installed": True,
        "optimizer_called": False,
        "optimizer_call_count": 0,
        "optimizer_entrypoints_guarded": [],
    }

    def _guard(name):
        state["optimizer_entrypoints_guarded"].append(name)

        def _fail(*_args, **_kwargs):
            state["optimizer_called"] = True
            state["optimizer_call_count"] += 1
            raise AssertionError("Optimizer must not run in point-provider parity test")

        return _fail

    for name in (
        "execute_runtime_geometry_fit",
        "execute_runtime_geometry_fit_solver_phase",
        "solve_geometry_fit_request",
    ):
        if hasattr(geometry_fit, name):
            monkeypatch.setattr(geometry_fit, name, _guard(name))
    monkeypatch.setattr(
        opt,
        "fit_geometry_parameters",
        _guard("ra_sim.fitting.optimization.fit_geometry_parameters"),
    )
    return state


def _build_point_provider_dataset(saved_entries, simulated_rows):
    bindings = _make_legacy_dense_manual_dataset_bindings(
        saved_entries=saved_entries,
        simulated_rows=simulated_rows,
        refresh_pairs=False,
    )
    return geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=bindings,
        orientation_cfg={"mode": "auto"},
    )


def _build_saved_state_point_provider_report_dataset():
    return geometry_fit.build_geometry_fit_saved_state_point_provider_dataset(
        0,
        [
            {
                "q_group_key": ("q_group", "primary", 0, 3),
                "source_table_index": 1,
                "source_reflection_index": 1,
                "source_row_index": 0,
                "source_branch_index": 0,
                "source_peak_index": 0,
                "source_label": "primary",
                "hkl": (0, 0, 3),
                "x": 10.0,
                "y": 20.0,
                "refined_sim_x": 11.0,
                "refined_sim_y": 21.0,
            }
        ],
    )


def _coordinate_visual_capture():
    return {
        "visual_truth_available": True,
        "plotting_metadata": {"image_shape": [100, 100], "y_axis_inverted": True},
        "pairs": [
            {
                "pair_index": 0,
                "pair_id": "pair-a",
                "hkl": [0, 0, 1],
                "source_branch_index": 0,
                "q_group_key": ["q", "a"],
                "visual_background_point": [10.0, 20.0],
                "visual_background_frame": "display",
                "visual_simulated_point": [14.0, 25.0],
                "visual_simulated_frame": "display",
            },
            {
                "pair_index": 1,
                "pair_id": "pair-b",
                "hkl": [0, 0, 2],
                "source_branch_index": 1,
                "q_group_key": ["q", "b"],
                "visual_background_point": [30.0, 40.0],
                "visual_background_frame": "display",
                "visual_simulated_point": [36.0, 48.0],
                "visual_simulated_frame": "display",
            },
        ],
    }


def _coordinate_surface_rows(
    *,
    background_offset=(0.0, 0.0),
    simulated_offset=(0.0, 0.0),
    frame="display",
    order=(0, 1),
    xy_swap=False,
    y_flip_height=None,
):
    rows = []
    visual_pairs = _coordinate_visual_capture()["pairs"]
    for source_idx, visual in enumerate([visual_pairs[idx] for idx in order]):
        bg = list(visual["visual_background_point"])
        sim = list(visual["visual_simulated_point"])
        if xy_swap:
            bg = [bg[1], bg[0]]
            sim = [sim[1], sim[0]]
        if y_flip_height is not None:
            bg = [bg[0], float(y_flip_height) - bg[1]]
            sim = [sim[0], float(y_flip_height) - sim[1]]
        bg = [bg[0] + float(background_offset[0]), bg[1] + float(background_offset[1])]
        sim = [sim[0] + float(simulated_offset[0]), sim[1] + float(simulated_offset[1])]
        rows.append(
            {
                "pair_index": int(source_idx),
                "pair_id": visual["pair_id"],
                "hkl": visual["hkl"],
                "source_branch_index": visual["source_branch_index"],
                "q_group_key": visual["q_group_key"],
                "background_point": bg,
                "background_frame": frame,
                "simulated_point": sim,
                "simulated_frame": frame,
            }
        )
    return rows


def _all_coordinate_surfaces(**surface_overrides):
    exact_rows = _coordinate_surface_rows()
    surfaces = {
        "provider_pairs": [dict(row) for row in exact_rows],
        "manual_point_pairs": [dict(row) for row in exact_rows],
        "initial_pairs_display": [dict(row) for row in exact_rows],
        "measured_for_fit": [dict(row) for row in exact_rows],
        'spec["measured_peaks"]': [dict(row) for row in exact_rows],
    }
    surfaces.update(surface_overrides)
    return surfaces


def test_visual_backend_parity_detects_exact_match() -> None:
    report = coord_diag.build_coordinate_parity_diagnosis(
        _coordinate_visual_capture(),
        _all_coordinate_surfaces(),
    )

    assert report["ok"] is True
    assert report["classification"] == "visual_backend_parity_ok"


def test_visual_backend_parity_detects_provider_mismatch() -> None:
    report = coord_diag.build_coordinate_parity_diagnosis(
        _coordinate_visual_capture(),
        _all_coordinate_surfaces(
            provider_pairs=_coordinate_surface_rows(background_offset=(2.0, 0.0)),
        ),
    )

    assert report["first_mismatching_surface"] == "provider_pairs"
    assert report["recommended_fix_location"] == "provider_pair_construction"


def test_visual_backend_parity_detects_dataset_mismatch() -> None:
    report = coord_diag.build_coordinate_parity_diagnosis(
        _coordinate_visual_capture(),
        _all_coordinate_surfaces(
            manual_point_pairs=_coordinate_surface_rows(simulated_offset=(0.0, 3.0)),
        ),
    )

    assert report["first_mismatching_surface"] == "manual_point_pairs"
    assert report["recommended_fix_location"] == "build_geometry_manual_fit_dataset"


def test_visual_backend_parity_detects_optimizer_request_mismatch() -> None:
    report = coord_diag.build_coordinate_parity_diagnosis(
        _coordinate_visual_capture(),
        _all_coordinate_surfaces(
            **{
                "optimizer_request.measured_peaks": _coordinate_surface_rows(
                    background_offset=(0.0, -2.0)
                )
            }
        ),
    )

    assert report["first_mismatching_surface"] == "optimizer_request.measured_peaks"
    assert report["recommended_fix_location"] == "GeometryFitSolverRequest_construction"


def test_coordinate_diagnostic_detects_optimizer_request_mismatch() -> None:
    report = coord_diag.build_coordinate_parity_diagnosis(
        _coordinate_visual_capture(),
        _all_coordinate_surfaces(
            **{
                "optimizer_request.measured_peaks": _coordinate_surface_rows(
                    background_offset=(0.0, -2.0)
                )
            }
        ),
    )

    assert report["first_mismatching_surface"] == "optimizer_request.measured_peaks"
    assert report["recommended_fix_location"] == "GeometryFitSolverRequest_construction"
    assert report["optimizer_request_compared"] is True
    assert report["optimizer_request_visual_parity_ok"] is False


def test_visual_backend_parity_detects_optimizer_request_pair_order_mismatch() -> None:
    report = coord_diag.build_coordinate_parity_diagnosis(
        _coordinate_visual_capture(),
        _all_coordinate_surfaces(
            **{"optimizer_request.measured_peaks": _coordinate_surface_rows(order=(1, 0))}
        ),
    )

    request_result = report["surface_results"]["optimizer_request.measured_peaks"]
    assert request_result["ordered_pairs_match"] is False
    assert request_result["unordered_pairs_match"] is True
    assert report["first_mismatching_surface"] == "optimizer_request.measured_peaks"
    assert report["recommended_fix_location"] == "GeometryFitSolverRequest_construction"


def test_coordinate_transform_diagnosis_detects_xy_swap() -> None:
    report = coord_diag.build_coordinate_parity_diagnosis(
        _coordinate_visual_capture(),
        _all_coordinate_surfaces(provider_pairs=_coordinate_surface_rows(xy_swap=True)),
    )

    best = report["best_transform_by_surface"]["provider_pairs"]
    assert best["best_transform_name"] in {"swap_axes", "row_col_swap"}
    assert best["transform_name"] == best["best_transform_name"]
    assert report["classification"] == "axis_swap_detected"


def test_coordinate_transform_diagnosis_detects_y_flip() -> None:
    report = coord_diag.build_coordinate_parity_diagnosis(
        _coordinate_visual_capture(),
        _all_coordinate_surfaces(
            provider_pairs=_coordinate_surface_rows(y_flip_height=100.0),
        ),
    )

    best = report["best_transform_by_surface"]["provider_pairs"]
    assert best["best_transform_name"] in {"flip_y", "top_left_vs_bottom_left_origin"}
    assert best["transform_name"] == best["best_transform_name"]
    assert report["classification"] == "origin_flip_detected"


def test_coordinate_transform_diagnosis_detects_constant_translation() -> None:
    report = coord_diag.build_coordinate_parity_diagnosis(
        _coordinate_visual_capture(),
        _all_coordinate_surfaces(
            provider_pairs=_coordinate_surface_rows(
                background_offset=(5.0, -3.0),
                simulated_offset=(5.0, -3.0),
            ),
        ),
    )

    assert report["classification"] == "constant_translation_detected"
    assert (
        report["best_transform_by_surface"]["provider_pairs"]["best_transform_name"]
        == "constant_translation"
    )


def test_coordinate_transform_schema_contains_best_transform_name() -> None:
    report = coord_diag.build_coordinate_parity_diagnosis(
        _coordinate_visual_capture(),
        _all_coordinate_surfaces(provider_pairs=_coordinate_surface_rows(xy_swap=True)),
    )

    best = report["best_transform_by_surface"]["provider_pairs"]
    assert "best_transform_name" in best
    assert best["best_transform_name"] == best["transform_name"]


def test_visual_backend_parity_detects_pair_order_mismatch() -> None:
    report = coord_diag.build_coordinate_parity_diagnosis(
        _coordinate_visual_capture(),
        _all_coordinate_surfaces(provider_pairs=_coordinate_surface_rows(order=(1, 0))),
    )

    provider_result = report["surface_results"]["provider_pairs"]
    assert provider_result["ordered_pairs_match"] is False
    assert provider_result["unordered_pairs_match"] is True
    assert report["first_mismatching_surface"] == "provider_pairs"
    assert report["classification"] == "pair_order_mismatch"
    assert (
        report["best_transform_by_surface"]["provider_pairs"]["classification"]
        == "not_scored_pair_order_mismatch"
    )
    assert report["best_transform_by_surface"]["provider_pairs"]["best_transform_name"] is None
    assert report["best_transform_by_surface"]["provider_pairs"]["transform_name"] is None


def test_visual_backend_parity_detects_frame_mismatch() -> None:
    report = coord_diag.build_coordinate_parity_diagnosis(
        _coordinate_visual_capture(),
        _all_coordinate_surfaces(provider_pairs=_coordinate_surface_rows(frame="detector_native")),
    )

    provider_result = report["surface_results"]["provider_pairs"]
    assert provider_result["frame_mismatch_count"] > 0
    assert provider_result["passes_visual_parity"] is False
    assert report["classification"] == "frame_mismatch_detected"


def test_visual_backend_parity_fails_missing_required_surface() -> None:
    surfaces = _all_coordinate_surfaces()
    surfaces.pop("measured_for_fit")

    report = coord_diag.build_coordinate_parity_diagnosis(
        _coordinate_visual_capture(),
        surfaces,
    )

    assert report["ok"] is False
    assert report["missing_required_surfaces"] == ["measured_for_fit"]
    assert report["surface_results"]["measured_for_fit"]["available"] is False
    assert report["first_mismatching_surface"] == "measured_for_fit"
    assert report["recommended_fix_location"] == "measured_for_fit_construction"


def test_visual_capture_is_not_backend_reconstruction() -> None:
    visual_records = [
        {
            "pair_index": 0,
            "pair_id": "pair-a",
            "hkl": [0, 0, 1],
            "source_branch_index": 0,
            "q_group_key": ["q", "a"],
            "bg_display": [10.0, 20.0],
            "sim_display": [14.0, 25.0],
        }
    ]
    nonsense_provider_rows = [
        {
            "pair_index": 0,
            "pair_id": "pair-a",
            "background_point": [999.0, 999.0],
            "simulated_point": [1000.0, 1000.0],
        }
    ]

    capture = coord_diag.collect_geometry_visual_pair_positions(visual_records)

    assert nonsense_provider_rows[0]["background_point"] == [999.0, 999.0]
    assert capture["visual_truth_available"] is True
    assert capture["pairs"][0]["visual_background_point"] == [10.0, 20.0]
    assert capture["pairs"][0]["visual_simulated_point"] == [14.0, 25.0]


def test_new4_visual_capture_path_uses_manual_render_wrapper_not_saved_builder(
    monkeypatch,
    tmp_path,
) -> None:
    def _fail_saved_builder(*_args, **_kwargs):
        raise AssertionError("saved-field visual builder must not be used")

    monkeypatch.setattr(
        coord_diag,
        "build_visual_overlay_records_from_saved_entries",
        _fail_saved_builder,
    )

    report = coord_diag.run_new4_visual_backend_coordinate_diagnostic(
        state_path=Path("artifacts/geometry_fit_gui_states/new4.json"),
        provider_report_path=Path(
            "artifacts/geometry_fit_gui_states/new4_point_provider_report.json"
        ),
        background_index=0,
        output_dir=tmp_path / "new4",
    )

    assert report["visual_truth_available"] is True
    assert report["visual_capture_path_confirmed"] is True
    assert "render_current_geometry_manual_pairs" in report["visual_capture_path"]


def test_diagnostic_does_not_call_optimizer(monkeypatch, tmp_path) -> None:
    def _fail(*_args, **_kwargs):
        raise AssertionError("optimizer must not run")

    monkeypatch.setattr(geometry_fit, "solve_geometry_fit_request", _fail)
    monkeypatch.setattr(opt, "least_squares", _fail)

    report = coord_diag.run_new4_visual_backend_coordinate_diagnostic(
        state_path=Path("artifacts/geometry_fit_gui_states/new4.json"),
        provider_report_path=Path(
            "artifacts/geometry_fit_gui_states/new4_point_provider_report.json"
        ),
        background_index=0,
        output_dir=tmp_path / "new4",
    )

    assert report["optimizer_called"] is False
    assert report["least_squares_called"] is False


def _new4_visual_rows_for_optimizer_request() -> list[dict[str, object]]:
    state = coord_diag.load_gui_state_payload(Path("artifacts/geometry_fit_gui_states/new4.json"))
    entries = coord_diag.saved_entries_for_background(state, background_index=0)
    overlay = coord_diag.capture_manual_geometry_overlay_input_from_render_path(
        entries,
        background_index=0,
    )
    visual = coord_diag.collect_geometry_visual_pair_positions(
        overlay["pairs"],
        max_display_markers=int(overlay["max_display_markers"]),
    )
    rows = []
    for pair in visual["pairs"]:
        rows.append(
            {
                "pair_index": pair["pair_index"],
                "pair_id": pair["pair_id"],
                "hkl": pair["hkl"],
                "source_branch_index": pair["source_branch_index"],
                "q_group_key": pair["q_group_key"],
                "x": pair["visual_background_point"][0],
                "y": pair["visual_background_point"][1],
                "simulated_point": pair["visual_simulated_point"],
                "background_frame": pair["visual_background_frame"],
                "simulated_frame": pair["visual_simulated_frame"],
            }
        )
    return rows


def test_coordinate_diagnostic_compares_optimizer_request_when_available(
    monkeypatch,
    tmp_path,
) -> None:
    rows = _new4_visual_rows_for_optimizer_request()

    def _fake_capture(*, state_path, background_index):
        del state_path, background_index
        return {
            "rows": rows,
            "optimizer_entrypoints_called": [],
            "optimizer_request_missing_fields": [],
            "optimizer_request_missing_fields_by_row": [],
            "optimizer_request_capture_error": None,
        }

    monkeypatch.setattr(
        coord_diag,
        "capture_optimizer_request_rows_from_solver_request",
        _fake_capture,
    )

    report = coord_diag.run_new4_visual_backend_coordinate_diagnostic(
        state_path=Path("artifacts/geometry_fit_gui_states/new4.json"),
        provider_report_path=Path(
            "artifacts/geometry_fit_gui_states/new4_point_provider_report.json"
        ),
        background_index=0,
        output_dir=tmp_path / "new4",
        include_optimizer_request=True,
    )

    assert report["optimizer_request_compared"] is True
    assert report["optimizer_request_pair_count"] == 7
    assert report["optimizer_request_visual_parity_ok"] is True
    assert report["optimizer_called"] is False
    assert report["least_squares_called"] is False
    assert report["optimizer_entrypoints_called"] == []
    assert report["state_hash_unchanged"] is True


def test_coordinate_diagnostic_optimizer_request_capture_failure_is_incomplete_not_frame_mismatch(
    monkeypatch,
    tmp_path,
) -> None:
    capture_error = "Failed to capture headless execution setup context."

    def _fake_capture(*, state_path, background_index):
        del state_path, background_index
        return {
            "rows": [],
            "optimizer_entrypoints_called": [],
            "optimizer_request_missing_fields": [],
            "optimizer_request_missing_fields_by_row": [],
            "optimizer_request_capture_error": capture_error,
        }

    monkeypatch.setattr(
        coord_diag,
        "capture_optimizer_request_rows_from_solver_request",
        _fake_capture,
    )

    report = coord_diag.run_new4_visual_backend_coordinate_diagnostic(
        state_path=Path("artifacts/geometry_fit_gui_states/new4.json"),
        provider_report_path=Path(
            "artifacts/geometry_fit_gui_states/new4_point_provider_report.json"
        ),
        background_index=0,
        output_dir=tmp_path / "new4",
        include_optimizer_request=True,
    )

    assert report["ok"] is False
    assert report["classification"] == "diagnostic_incomplete_optimizer_request_unavailable"
    assert report["optimizer_request_compared"] is False
    assert report["optimizer_request_pair_count"] == 0
    assert report["optimizer_request_visual_parity_ok"] is False
    assert report["optimizer_request_unavailable_reason"] == "solver_request_capture_failed"
    assert report["optimizer_request_capture_error"] == capture_error
    assert report["first_mismatching_surface"] is None
    assert report["recommended_fix_location"] == "optimizer_request_capture"
    assert "optimizer_request.measured_peaks" not in report["surfaces_compared"]
    assert "optimizer_request.measured_peaks" not in report["surface_results"]
    assert all(
        result.get("frame_mismatch_count", 0) == 0 for result in report["surface_results"].values()
    )


def test_optimizer_request_capture_missing_ladder_returns_structured_error(
    monkeypatch,
    tmp_path,
) -> None:
    def _missing_module(name):
        assert name == "scripts.debug.run_new4_geometry_fit_ladder"
        raise ModuleNotFoundError("No module named 'scripts'", name="scripts")

    monkeypatch.setattr(coord_diag.importlib, "import_module", _missing_module)

    result = coord_diag.capture_optimizer_request_rows_from_solver_request(
        state_path=tmp_path / "missing_state.json",
        background_index=0,
    )

    assert result["rows"] == []
    assert result["optimizer_entrypoints_called"] == []
    assert result["optimizer_request_missing_fields"] == []
    assert result["optimizer_request_missing_fields_by_row"] == []
    assert (
        "requires scripts/debug/run_new4_geometry_fit_ladder.py"
        in result["optimizer_request_capture_error"]
    )


def test_coordinate_diagnostic_without_optimizer_request_still_ok_for_dataset_surfaces(
    tmp_path,
) -> None:
    report = coord_diag.run_new4_visual_backend_coordinate_diagnostic(
        state_path=Path("artifacts/geometry_fit_gui_states/new4.json"),
        provider_report_path=Path(
            "artifacts/geometry_fit_gui_states/new4_point_provider_report.json"
        ),
        background_index=0,
        output_dir=tmp_path / "new4",
        include_optimizer_request=False,
    )

    assert report["ok"] is True
    assert report["classification"] == "visual_backend_parity_ok"
    assert report["optimizer_request_compared"] is False
    assert report["optimizer_request_unavailable_reason"] == "not_requested"
    assert "optimizer_request.measured_peaks" not in report["surfaces_compared"]


def test_coordinate_diagnostic_optimizer_request_path_does_not_solve(
    monkeypatch,
    tmp_path,
) -> None:
    def _fail(*_args, **_kwargs):
        raise AssertionError("solve path must not run")

    monkeypatch.setattr(opt, "least_squares", _fail)
    monkeypatch.setattr(geometry_fit, "solve_geometry_fit_request", _fail)

    report = coord_diag.run_new4_visual_backend_coordinate_diagnostic(
        state_path=Path("artifacts/geometry_fit_gui_states/new4.json"),
        provider_report_path=Path(
            "artifacts/geometry_fit_gui_states/new4_point_provider_report.json"
        ),
        background_index=0,
        output_dir=tmp_path / "new4",
        include_optimizer_request=True,
    )

    assert report["optimizer_called"] is False
    assert report["least_squares_called"] is False
    assert report["optimizer_entrypoints_called"] == []
    assert report["state_hash_unchanged"] is True


def test_new4_diagnostic_rung_absent_does_not_fail_optimizer_optional(tmp_path) -> None:
    report = coord_diag.run_new4_visual_backend_coordinate_diagnostic(
        state_path=Path("artifacts/geometry_fit_gui_states/new4.json"),
        provider_report_path=Path(
            "artifacts/geometry_fit_gui_states/new4_point_provider_report.json"
        ),
        background_index=0,
        output_dir=tmp_path / "new4",
        rung_report_path=tmp_path / "missing_rung_01_objective_dry_run.json",
    )

    assert report["optimizer_request_compared"] is False
    assert report["optimizer_request_unavailable_reason"] == "not_requested"
    assert "optimizer_request.measured_peaks" not in report["surfaces_compared"]


def test_new4_diagnostic_rung_lacks_coordinates_reports_reason(tmp_path) -> None:
    rung_path = tmp_path / "rung_01_objective_dry_run.json"
    rung_path.write_text(
        json.dumps(
            {
                "optimizer_request_pair_handoff": [
                    {"pair_index": 0, "hkl": [0, 0, 3]},
                ],
            }
        ),
        encoding="utf-8",
    )

    report = coord_diag.run_new4_visual_backend_coordinate_diagnostic(
        state_path=Path("artifacts/geometry_fit_gui_states/new4.json"),
        provider_report_path=Path(
            "artifacts/geometry_fit_gui_states/new4_point_provider_report.json"
        ),
        background_index=0,
        output_dir=tmp_path / "new4",
        rung_report_path=rung_path,
    )

    assert report["optimizer_request_compared"] is False
    assert report["optimizer_request_unavailable_reason"] == "not_requested"


def test_new4_diagnostic_rung_rows_do_not_compare_without_optimizer_flag(
    tmp_path,
) -> None:
    state = coord_diag.load_gui_state_payload(Path("artifacts/geometry_fit_gui_states/new4.json"))
    entries = coord_diag.saved_entries_for_background(state, background_index=0)
    overlay = coord_diag.capture_manual_geometry_overlay_input_from_render_path(
        entries,
        background_index=0,
    )
    visual = coord_diag.collect_geometry_visual_pair_positions(
        overlay["pairs"],
        max_display_markers=int(overlay["max_display_markers"]),
    )
    rows = []
    for pair in visual["pairs"]:
        rows.append(
            {
                "pair_index": pair["pair_index"],
                "pair_id": pair["pair_id"],
                "hkl": pair["hkl"],
                "source_branch_index": pair["source_branch_index"],
                "q_group_key": pair["q_group_key"],
                "x": pair["visual_background_point"][0],
                "y": pair["visual_background_point"][1],
                "simulated_point": pair["visual_simulated_point"],
            }
        )
    rung_path = tmp_path / "rung_01_objective_dry_run.json"
    rung_path.write_text(
        json.dumps({"optimizer_request_measured_peaks": rows}),
        encoding="utf-8",
    )

    report = coord_diag.run_new4_visual_backend_coordinate_diagnostic(
        state_path=Path("artifacts/geometry_fit_gui_states/new4.json"),
        provider_report_path=Path(
            "artifacts/geometry_fit_gui_states/new4_point_provider_report.json"
        ),
        background_index=0,
        output_dir=tmp_path / "new4",
        rung_report_path=rung_path,
    )

    assert report["optimizer_request_compared"] is False
    assert "optimizer_request.measured_peaks" not in report["surfaces_compared"]


def test_new4_visual_backend_coordinate_report_writes_files(tmp_path) -> None:
    script = Path("scripts/debug/diagnose_new4_visual_backend_coordinates.py")
    output_dir = tmp_path / "new4"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--state",
            "artifacts/geometry_fit_gui_states/new4.json",
            "--provider-report",
            "artifacts/geometry_fit_gui_states/new4_point_provider_report.json",
            "--background-index",
            "0",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    json_path = output_dir / "coordinate_transform_diagnosis.json"
    csv_path = output_dir / "coordinate_transform_pairs.csv"
    overlay_path = output_dir / "coordinate_transform_overlay.png"
    vectors_path = output_dir / "coordinate_transform_vectors.png"

    for path in (json_path, csv_path, overlay_path, vectors_path):
        assert path.exists()
        assert path.stat().st_size > 0

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(coord_diag.REQUIRED_SURFACES).issubset(set(report["surfaces_compared"]))
    assert report["optimizer_request_compared"] is False
    assert report["pairs"]
    first_pair = report["pairs"][0]
    assert "visual_background_point" in first_pair
    assert "visual_simulated_point" in first_pair
    assert first_pair["surfaces"]


def test_canonical_geometry_source_identity_normalizes_tuple_and_list_fields() -> None:
    left = {
        "hkl": (1, 0, 3),
        "q_group_key": ("q_group", "primary", 1, 3),
        "source_table_index": np.int64(12),
        "source_reflection_index": np.int64(44),
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": np.bool_(True),
        "source_row_index": np.int64(7),
        "source_branch_index": np.int64(1),
        "source_peak_index": np.int64(1),
        "source_label": "primary",
    }
    right = {
        "hkl": [1, 0, 3],
        "q_group_key": ["q_group", "primary", 1, 3],
        "source_table_index": 12,
        "source_reflection_index": 44,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_row_index": 7,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "source_label": "primary",
    }

    assert manual_geometry.canonical_geometry_source_identity(left) == (
        manual_geometry.canonical_geometry_source_identity(right)
    )


def test_canonical_geometry_source_identity_distinguishes_persisted_source_rows() -> None:
    left = {
        "hkl": (1, 0, 3),
        "q_group_key": ("q_group", "primary", 1, 3),
        "source_table_index": 12,
        "source_reflection_index": 44,
        "source_row_index": 7,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "source_label": "primary",
    }
    right = dict(left)
    right["source_row_index"] = 8

    assert manual_geometry.canonical_geometry_source_identity(left) != (
        manual_geometry.canonical_geometry_source_identity(right)
    )


def test_canonical_geometry_source_identity_omits_null_optional_bool() -> None:
    missing = {"hkl": (1, 0, 3), "source_reflection_index": 44}
    null_value = {
        "hkl": [1, 0, 3],
        "source_reflection_index": 44,
        "source_reflection_is_full": None,
    }
    false_value = {
        "hkl": [1, 0, 3],
        "source_reflection_index": 44,
        "source_reflection_is_full": False,
    }

    assert manual_geometry.canonical_geometry_source_identity(missing) == (
        manual_geometry.canonical_geometry_source_identity(null_value)
    )
    assert manual_geometry.canonical_geometry_source_identity(missing) != (
        manual_geometry.canonical_geometry_source_identity(false_value)
    )


def test_geometry_fit_point_provider_matches_manual_qr_picker_assignments() -> None:
    saved_entries = [
        {
            "pair_id": "bg0:pair0",
            "q_group_key": ("q_group", "primary", 1, 3),
            "source_table_index": 10,
            "source_reflection_index": 10,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_label": "primary",
            "hkl": (1, 0, 3),
            "x": 10.25,
            "y": 20.5,
            "refined_sim_x": 11.75,
            "refined_sim_y": 21.25,
        }
    ]
    simulated_rows = [
        {
            "q_group_key": ("q_group", "primary", 1, 3),
            "source_table_index": 10,
            "source_reflection_index": 10,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_label": "primary",
            "hkl": (1, 0, 3),
            "sim_col": 111.0,
            "sim_row": 222.0,
        }
    ]

    dataset = _build_point_provider_dataset(saved_entries, simulated_rows)
    report = dataset["point_provider_report"]
    pair = report["pairs"][0]

    assert report["manual_picker_pair_count"] == 1
    assert report["point_provider_pair_count"] == 1
    assert report["pair_count_match"] is True
    assert report["ordered_pairs_match"] is True
    assert pair["source_identity_match"] is True
    assert pair["background_point_match"] is True
    assert pair["simulated_point_match"] is True
    assert pair["dataset_points_match_provider_points"] is True
    assert pair["provider_selected_simulated_point"] == [11.75, 21.25]
    assert dataset["initial_pairs_display"][0]["sim_display"] == (11.75, 21.25)


def test_point_provider_saved_refined_sim_point_beats_live_overlay(monkeypatch) -> None:
    def _no_refined_override(_entry, resolved_source_entry, **_kwargs):
        return resolved_source_entry

    monkeypatch.setattr(
        manual_geometry,
        "geometry_manual_apply_refined_simulated_override",
        _no_refined_override,
    )
    saved_entries = [
        {
            "pair_id": "bg0:pair0",
            "q_group_key": ("q_group", "primary", 1, 3),
            "source_table_index": 10,
            "source_reflection_index": 10,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_label": "primary",
            "hkl": (1, 0, 3),
            "x": 10.25,
            "y": 20.5,
            "refined_sim_x": 11.75,
            "refined_sim_y": 21.25,
        }
    ]
    simulated_rows = [
        {
            "q_group_key": ("q_group", "primary", 1, 3),
            "source_table_index": 10,
            "source_reflection_index": 10,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_label": "primary",
            "hkl": (1, 0, 3),
            "sim_col": 111.0,
            "sim_row": 222.0,
        }
    ]

    dataset = _build_point_provider_dataset(saved_entries, simulated_rows)
    report_pair = dataset["point_provider_report"]["pairs"][0]

    assert dataset["initial_pairs_display"][0]["sim_display"] == (11.75, 21.25)
    assert dataset["manual_point_pairs"][0]["simulated_point"] == [11.75, 21.25]
    assert report_pair["provider_selected_simulated_point"] == [11.75, 21.25]
    assert report_pair["dataset_simulated_point"] == [11.75, 21.25]


def test_point_provider_saved_refined_sim_point_overwrites_caked_prefill(
    monkeypatch,
) -> None:
    def _no_refined_override(_entry, resolved_source_entry, **_kwargs):
        return resolved_source_entry

    monkeypatch.setattr(
        manual_geometry,
        "geometry_manual_apply_refined_simulated_override",
        _no_refined_override,
    )
    saved_entries = [
        {
            "pair_id": "bg0:pair0",
            "q_group_key": ("q_group", "primary", 1, 3),
            "source_table_index": 10,
            "source_reflection_index": 10,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_label": "primary",
            "hkl": (1, 0, 3),
            "x": 10.25,
            "y": 20.5,
            "refined_sim_caked_x": 55.5,
            "refined_sim_caked_y": 66.5,
        }
    ]
    simulated_rows = [
        {
            "q_group_key": ("q_group", "primary", 1, 3),
            "source_table_index": 10,
            "source_reflection_index": 10,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_label": "primary",
            "hkl": (1, 0, 3),
            "sim_col": 111.0,
            "sim_row": 222.0,
            "caked_x": 333.0,
            "caked_y": 444.0,
            "two_theta_deg": 333.0,
            "phi_deg": 444.0,
        }
    ]
    bindings = replace(
        _make_legacy_dense_manual_dataset_bindings(
            saved_entries=saved_entries,
            simulated_rows=simulated_rows,
            refresh_pairs=False,
        ),
        pick_uses_caked_space=lambda: True,
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=bindings,
        orientation_cfg={"mode": "auto"},
    )

    assert dataset["initial_pairs_display"][0]["sim_display"] == (55.5, 66.5)
    assert dataset["initial_pairs_display"][0]["sim_caked_display"] == (55.5, 66.5)
    assert dataset["manual_point_pairs"][0]["simulated_point"] == [55.5, 66.5]
    assert dataset["spec"]["manual_point_pairs"][0]["simulated_point"] == [
        55.5,
        66.5,
    ]
    assert dataset["point_provider_report"]["pairs"][0]["provider_selected_simulated_point"] == [
        55.5,
        66.5,
    ]
    assert dataset["point_provider_report"]["pairs"][0]["dataset_simulated_point"] == [
        55.5,
        66.5,
    ]
    assert dataset["point_provider_report"]["pairs"][0]["simulated_point_match"] is True
    assert (
        dataset["point_provider_report"]["pairs"][0]["provider_dataset_fingerprint_match"] is True
    )


def test_point_provider_saved_refined_display_replaces_stale_live_native() -> None:
    captured: dict[str, object] = {}
    saved_entries = [
        {
            "pair_id": "bg0:pair0",
            "q_group_key": ("q_group", "primary", 1, 3),
            "source_table_index": 10,
            "source_reflection_index": 10,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_label": "primary",
            "hkl": (1, 0, 3),
            "x": 10.25,
            "y": 20.5,
            "refined_sim_x": 11.75,
            "refined_sim_y": 21.25,
        }
    ]
    simulated_rows = [
        {
            "q_group_key": ("q_group", "primary", 1, 3),
            "source_table_index": 10,
            "source_reflection_index": 10,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_label": "primary",
            "hkl": (1, 0, 3),
            "sim_col": 111.0,
            "sim_row": 222.0,
            "native_col": 777.0,
            "native_row": 888.0,
        }
    ]

    def _display_to_native_sim_coords(col, row, shape):
        return (float(col) + 100.0, float(row) + 100.0)

    def _select_fit_orientation(sim_pts, meas_pts, shape, *, cfg):
        captured["sim_pts"] = list(sim_pts)
        return (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "yx",
                "label": "identity",
            },
            {"pairs": len(sim_pts)},
        )

    bindings = replace(
        _make_legacy_dense_manual_dataset_bindings(
            saved_entries=saved_entries,
            simulated_rows=simulated_rows,
            refresh_pairs=False,
        ),
        display_to_native_sim_coords=_display_to_native_sim_coords,
        select_fit_orientation=_select_fit_orientation,
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=bindings,
        orientation_cfg={"mode": "auto"},
    )

    assert dataset["initial_pairs_display"][0]["sim_native"] == (111.75, 121.25)
    assert captured["sim_pts"] == [(111.75, 121.25)]


def test_point_provider_preserves_picker_selected_identity_when_resolvable() -> None:
    saved_entries = [
        {
            "pair_id": "bg0:pair0",
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 1,
            "source_reflection_index": 101,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_label": "selected-a",
            "hkl": (-1, 0, 5),
            "x": 100.0,
            "y": 100.0,
            "refined_sim_x": 300.0,
            "refined_sim_y": 300.0,
        }
    ]
    simulated_rows = [
        {
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 1,
            "source_reflection_index": 101,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_label": "selected-a",
            "hkl": (-1, 0, 5),
            "sim_col": 305.0,
            "sim_row": 305.0,
        },
        {
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 2,
            "source_reflection_index": 102,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_label": "tempting-b",
            "hkl": (-1, 0, 5),
            "sim_col": 101.0,
            "sim_row": 99.0,
        },
    ]

    dataset = _build_point_provider_dataset(saved_entries, simulated_rows)
    pair = dataset["point_provider_report"]["pairs"][0]

    assert pair["parity_mode"] == "picker_saved_value_preserved"
    assert pair["rebinding_fallback_used"] is False
    assert pair["provider_selected_source_identity_canonical"]["source_label"] == "selected-a"
    assert pair["provider_selected_simulated_point"] == [300.0, 300.0]


def test_point_provider_fallback_matches_picker_rule_when_source_identity_missing() -> None:
    saved_entries = [
        {
            "pair_id": "bg0:pair0",
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_branch_index": 1,
            "hkl": (-1, 0, 5),
            "x": 1000.0,
            "y": 1000.0,
        }
    ]
    simulated_rows = [
        {
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 10,
            "source_reflection_index": 10,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_label": "wrong-branch",
            "hkl": (-1, 0, 5),
            "sim_col": 1000.0,
            "sim_row": 1000.0,
        },
        {
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 11,
            "source_reflection_index": 11,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "source_label": "expected",
            "hkl": (-1, 0, 5),
            "sim_col": 1003.0,
            "sim_row": 998.0,
        },
        {
            "q_group_key": ("q_group", "primary", 1, 9),
            "source_table_index": 12,
            "source_reflection_index": 12,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "source_label": "wrong-hkl",
            "hkl": (-1, 0, 9),
            "sim_col": 1001.0,
            "sim_row": 999.0,
        },
    ]

    dataset = _build_point_provider_dataset(saved_entries, simulated_rows)
    provider_pair = dataset["provider_pairs"][0]

    assert provider_pair["parity_mode"] == "picker_rule_fallback"
    assert provider_pair["rebinding_fallback_used"] is True
    assert provider_pair["provider_pair_index"] == 0
    assert provider_pair["selected_source_identity_canonical"]["normalized_hkl"] == [
        -1,
        0,
        5,
    ]
    assert provider_pair["selected_source_identity_canonical"]["source_branch_index"] == 1
    assert provider_pair["simulated_point"] == [1003.0, 998.0]


def test_point_provider_stale_locator_is_diagnostic_when_saved_assignment_resolves() -> None:
    saved_entries = [
        {
            "pair_id": "bg0:pair0",
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 999,
            "source_reflection_index": 999,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_label": "saved-picker",
            "hkl": (-1, 0, 5),
            "x": 100.0,
            "y": 100.0,
            "refined_sim_x": 300.0,
            "refined_sim_y": 300.0,
        }
    ]
    simulated_rows = [
        {
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 1,
            "source_reflection_index": 101,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_label": "live-semantic-match",
            "hkl": (-1, 0, 5),
            "sim_col": 101.0,
            "sim_row": 99.0,
        }
    ]

    report = _build_point_provider_dataset(saved_entries, simulated_rows)["point_provider_report"]
    pair = report["pairs"][0]

    assert report["ok"] is True
    assert report["fallback_pair_count"] == 0
    assert pair["rebinding_fallback_used"] is False
    assert pair["fallback_reason"] is None
    assert pair["source_locator_identity_match"] is False
    assert pair["source_semantic_identity_match"] is True
    assert pair["stale_source_identity_diagnostic"] is True
    assert pair["source_identity_match"] is True
    assert pair["provider_selected_source_identity_canonical"]["source_table_index"] == 999
    assert pair["provider_selected_simulated_point"] == [300.0, 300.0]


def test_point_provider_marks_stale_saved_identity_as_fallback() -> None:
    saved_entries = [
        {
            "pair_id": "bg0:pair0",
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 999,
            "source_reflection_index": 999,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_label": "stale",
            "hkl": (-1, 0, 5),
            "x": 100.0,
            "y": 100.0,
        }
    ]
    simulated_rows = [
        {
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 1,
            "source_reflection_index": 101,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_label": "fallback",
            "hkl": (-1, 0, 5),
            "sim_col": 101.0,
            "sim_row": 99.0,
        }
    ]

    report = _build_point_provider_dataset(saved_entries, simulated_rows)["point_provider_report"]
    pair = report["pairs"][0]

    assert report["ok"] is False
    assert report["fallback_pair_count"] == 1
    assert pair["rebinding_fallback_used"] is True
    assert pair["fallback_reason"] == "missing_saved_simulated_point"
    assert pair["source_identity_match"] is False
    assert pair["provider_simulated_point_source"] == "fallback_rebind"


def test_point_provider_report_is_stable_and_machine_readable() -> None:
    saved_entries = [
        {
            "q_group_key": ("q_group", "primary", 0, 3),
            "source_table_index": 1,
            "source_reflection_index": 1,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_label": "primary",
            "hkl": (0, 0, 3),
            "x": 10.0,
            "y": 20.0,
            "refined_sim_x": 11.0,
            "refined_sim_y": 21.0,
        }
    ]
    simulated_rows = [
        {
            "q_group_key": ("q_group", "primary", 0, 3),
            "source_table_index": 1,
            "source_reflection_index": 1,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_label": "primary",
            "hkl": (0, 0, 3),
            "sim_col": 111.0,
            "sim_row": 222.0,
        }
    ]

    dataset = _build_point_provider_dataset(saved_entries, simulated_rows)
    rendered = json.dumps(dataset["point_provider_report"], sort_keys=True)

    assert rendered
    assert "source_rows_for_trace" not in rendered
    assert dataset["point_provider_report"]["pairs"][0]["dataset_pair_fingerprint"]
    assert dataset["point_provider_report"]["provider_dataset_fingerprint_match"] is True
    assert dataset["point_provider_report"]["optimizer_guard_installed"] is False


def test_point_provider_report_reads_actual_handoff_rows_not_manual_trace() -> None:
    saved_entries = [
        {
            "q_group_key": ("q_group", "primary", 0, 3),
            "source_table_index": 1,
            "source_reflection_index": 1,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_label": "primary",
            "hkl": (0, 0, 3),
            "x": 10.0,
            "y": 20.0,
            "refined_sim_x": 11.0,
            "refined_sim_y": 21.0,
        }
    ]
    simulated_rows = [
        {
            "q_group_key": ("q_group", "primary", 0, 3),
            "source_table_index": 1,
            "source_reflection_index": 1,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_label": "primary",
            "hkl": (0, 0, 3),
            "sim_col": 111.0,
            "sim_row": 222.0,
        }
    ]
    dataset = _build_point_provider_dataset(saved_entries, simulated_rows)
    dataset["initial_pairs_display"][0]["sim_display"] = (999.0, 999.0)
    dataset["measured_for_fit"][0]["x"] = 888.0
    dataset["measured_for_fit"][0]["y"] = 777.0
    dataset["spec"]["measured_peaks"] = [dict(dataset["measured_for_fit"][0])]
    dataset["manual_point_pairs"][0]["simulated_point"] = [11.0, 21.0]

    report = geometry_fit.build_geometry_fit_point_provider_report(dataset)
    pair = report["pairs"][0]

    assert report["ok"] is False
    assert report["dataset_provider_mismatch_count"] == 1
    assert pair["manual_point_pairs_match_provider_points"] is True
    assert pair["initial_pairs_match_provider_points"] is False
    assert pair["measured_for_fit_match_provider_points"] is False
    assert pair["spec_measured_peaks_match_provider_points"] is False
    assert pair["provider_dataset_fingerprint_match"] is False
    assert report["provider_dataset_fingerprint_match"] is False
    assert pair["dataset_background_point"] == [888.0, 777.0]
    assert pair["dataset_simulated_point"] == [999.0, 999.0]
    assert pair["provider_selected_simulated_point"] == [11.0, 21.0]


def test_point_provider_report_rejects_measured_and_spec_simulated_point_drift() -> None:
    dataset = _build_saved_state_point_provider_report_dataset()
    dataset["measured_for_fit"][0]["sim_display"] = (999.0, 999.0)
    dataset["spec"]["measured_peaks"][0]["sim_display"] = (999.0, 999.0)

    report = geometry_fit.build_geometry_fit_point_provider_report(dataset)
    pair = report["pairs"][0]

    assert report["ok"] is False
    assert report["dataset_provider_mismatch_count"] > 0
    assert report["measured_for_fit_match_provider_points"] is False
    assert report["spec_measured_peaks_match_provider_points"] is False
    assert pair["measured_for_fit_match_provider_points"] is False
    assert pair["spec_measured_peaks_match_provider_points"] is False


def test_point_provider_report_rejects_missing_measured_and_spec_simulated_points() -> None:
    dataset = _build_saved_state_point_provider_report_dataset()
    for row in (dataset["measured_for_fit"][0], dataset["spec"]["measured_peaks"][0]):
        for key in (
            "sim_display",
            "sim_native",
            "simulated_two_theta_deg",
            "simulated_phi_deg",
        ):
            row.pop(key, None)

    report = geometry_fit.build_geometry_fit_point_provider_report(dataset)
    pair = report["pairs"][0]

    assert report["ok"] is False
    assert report["dataset_provider_mismatch_count"] > 0
    assert report["measured_for_fit_match_provider_points"] is False
    assert report["spec_measured_peaks_match_provider_points"] is False
    assert pair["measured_for_fit_match_provider_points"] is False
    assert pair["spec_measured_peaks_match_provider_points"] is False


@pytest.mark.parametrize(
    ("surface", "source_key"),
    (
        ("manual_point_pairs", "simulated_point_source"),
        ("initial_pairs_display", "provider_simulated_point_source"),
        ("measured_for_fit", "provider_simulated_point_source"),
        ("spec_measured_peaks", "provider_simulated_point_source"),
    ),
)
def test_point_provider_report_rejects_non_picker_owned_surface_sources(
    surface,
    source_key,
) -> None:
    dataset = _build_saved_state_point_provider_report_dataset()
    if surface == "spec_measured_peaks":
        row = dataset["spec"]["measured_peaks"][0]
        expected_pair_key = "spec_measured_peaks_point_sources_picker_owned"
    else:
        row = dataset[surface][0]
        expected_pair_key = {
            "manual_point_pairs": "manual_point_pairs_point_sources_picker_owned",
            "initial_pairs_display": "initial_pairs_point_sources_picker_owned",
            "measured_for_fit": "measured_for_fit_point_sources_picker_owned",
        }[surface]
    row[source_key] = "fallback_rebind"

    report = geometry_fit.build_geometry_fit_point_provider_report(dataset)
    pair = report["pairs"][0]

    assert report["ok"] is False
    assert report["all_point_sources_picker_owned"] is False
    assert report["dataset_provider_mismatch_count"] > 0
    assert pair[expected_pair_key] is False


@pytest.mark.parametrize(
    "surface",
    (
        "manual_point_pairs",
        "initial_pairs_display",
        "measured_for_fit",
        "spec_measured_peaks",
    ),
)
def test_point_provider_report_counts_extra_handoff_surface_rows(surface) -> None:
    dataset = _build_saved_state_point_provider_report_dataset()
    if surface == "spec_measured_peaks":
        rows = dataset["spec"]["measured_peaks"]
    else:
        rows = dataset[surface]
    rows.append(dict(rows[0]))

    report = geometry_fit.build_geometry_fit_point_provider_report(dataset)

    assert report["ok"] is False
    assert report["surface_pair_count_match"] is False
    assert report["surface_pair_count_mismatch_count"] > 0
    assert report["extra_surface_row_count"] > 0
    assert report["dataset_provider_mismatch_count"] > 0


@pytest.mark.parametrize(
    "surface",
    (
        "manual_point_pairs",
        "initial_pairs_display",
        "measured_for_fit",
        "spec_measured_peaks",
    ),
)
def test_point_provider_report_counts_missing_handoff_surface_rows(surface) -> None:
    dataset = _build_saved_state_point_provider_report_dataset()
    if surface == "spec_measured_peaks":
        dataset["spec"]["measured_peaks"] = []
    else:
        dataset[surface] = []

    report = geometry_fit.build_geometry_fit_point_provider_report(dataset)

    assert report["ok"] is False
    assert report["surface_pair_count_match"] is False
    assert report["surface_pair_count_mismatch_count"] > 0
    assert report["missing_surface_row_count"] > 0
    assert report["dataset_provider_mismatch_count"] > 0
    assert f"{surface}_row_count_mismatch" in (report["point_provider_parity_gate"]["reason_codes"])


def test_point_provider_does_not_call_optimizer(monkeypatch) -> None:
    guard_state = _point_provider_optimizer_guard(monkeypatch)
    saved_entries = [
        {
            "q_group_key": ("q_group", "primary", 0, 3),
            "source_table_index": 1,
            "source_reflection_index": 1,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_label": "primary",
            "hkl": (0, 0, 3),
            "x": 10.0,
            "y": 20.0,
            "refined_sim_x": 11.0,
            "refined_sim_y": 21.0,
        }
    ]
    simulated_rows = [
        {
            "q_group_key": ("q_group", "primary", 0, 3),
            "source_table_index": 1,
            "source_reflection_index": 1,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "source_label": "primary",
            "hkl": (0, 0, 3),
            "sim_col": 111.0,
            "sim_row": 222.0,
        }
    ]

    dataset = _build_point_provider_dataset(saved_entries, simulated_rows)
    report = geometry_fit.build_geometry_fit_point_provider_report(
        dataset,
        optimizer_guard_state=guard_state,
    )

    assert report["optimizer_guard_installed"] is True
    assert report["optimizer_called"] is False
    assert report["optimizer_call_count"] == 0
    assert set(report["optimizer_entrypoints_guarded"]) >= {
        "execute_runtime_geometry_fit",
        "execute_runtime_geometry_fit_solver_phase",
        "solve_geometry_fit_request",
        "ra_sim.fitting.optimization.fit_geometry_parameters",
    }


def test_point_provider_parity_for_new4_saved_state_without_running_optimizer(
    monkeypatch,
) -> None:
    guard_state = _point_provider_optimizer_guard(monkeypatch)
    repo_root = Path(__file__).resolve().parents[1]
    probe = _load_geometry_preflight_probe_module()

    def _fail_provider_only_boundary(*_args, **_kwargs):
        raise AssertionError("Provider-only parity must not enter headless preflight")

    monkeypatch.setattr(
        probe,
        "_prepare_validation_context",
        _fail_provider_only_boundary,
    )
    monkeypatch.setattr(
        probe,
        "_capture_preflight",
        _fail_provider_only_boundary,
    )
    monkeypatch.setattr(
        probe.hgf,
        "run_headless_geometry_fit",
        _fail_provider_only_boundary,
    )
    report = probe._run_point_provider_report_only(
        repo_root / "artifacts" / "geometry_fit_gui_states" / "new4.json",
        background_index=0,
        optimizer_guard_state=guard_state,
    )

    assert report["ok"] is True
    assert report["classification"] == "point_provider_parity_ok"
    assert report["point_provider_parity_gate"]["ok"] is True
    assert report["manual_picker_pair_count"] == 7
    assert report["point_provider_pair_count"] == 7
    assert report["manual_point_pair_count"] == 7
    assert report["initial_pairs_display_count"] == 7
    assert report["measured_for_fit_count"] == 7
    assert report["spec_measured_peaks_count"] == 7
    assert report["pair_count_match"] is True
    assert report["surface_pair_count_match"] is True
    assert report["ordered_pairs_match"] is True
    assert report["unordered_pairs_match"] is True
    assert report["missing_pair_count"] == 0
    assert report["branch_mismatch_count"] == 0
    assert report["fallback_pair_count"] == 0
    assert report["background_point_mismatch_count"] == 0
    assert report["simulated_point_mismatch_count"] == 0
    assert report["frame_mismatch_count"] == 0
    assert report["dataset_provider_mismatch_count"] == 0
    assert report["manual_point_pairs_match_provider_points"] is True
    assert report["initial_pairs_match_provider_points"] is True
    assert report["measured_for_fit_match_provider_points"] is True
    assert report["spec_measured_peaks_match_provider_points"] is True
    assert report["provider_dataset_fingerprint_match"] is True
    assert report["all_dataset_surfaces_match_provider_points"] is True
    assert report["all_point_sources_picker_owned"] is True
    assert report["optimizer_guard_installed"] is True
    assert report["optimizer_called"] is False
    assert report["optimizer_call_count"] == 0
    assert report["optimizer_path_entered"] is False
    assert report["optimizer_entrypoints_called"] == []
    for pair in report["pairs"]:
        assert pair["manual_picker_truth_available"] is True
        assert pair["missing_truth_fields"] == []
        assert pair["parity_mode"] == "picker_saved_value_preserved"
        assert pair["rebinding_fallback_used"] is False
        assert pair["manual_truth_mutated_by_refresh"] is False
        assert pair["source_identity_match"] is True
        assert pair["background_frame_match"] is True
        assert pair["simulated_frame_match"] is True
        assert pair["background_point_match"] is True
        assert pair["simulated_point_match"] is True
        assert pair["selected_to_background_distance_match"] is True
        assert pair["dataset_points_match_provider_points"] is True
        assert pair["manual_point_pairs_match_provider_points"] is True
        assert pair["initial_pairs_match_provider_points"] is True
        assert pair["measured_for_fit_match_provider_points"] is True
        assert pair["spec_measured_peaks_match_provider_points"] is True
        assert pair["provider_dataset_fingerprint_match"] is True
        assert pair["provider_point_sources_picker_owned"] is True
        assert pair["dataset_point_sources_picker_owned"] is True
        assert pair["provider_background_point_source"] in {
            "manual_picker_saved",
            "manual_picker_cache",
        }
        assert pair["provider_simulated_point_source"] in {
            "manual_picker_saved",
            "manual_picker_cache",
        }
        assert pair["dataset_background_point_source"] == pair["provider_background_point_source"]
        assert pair["dataset_simulated_point_source"] == pair["provider_simulated_point_source"]
        assert pair["provider_background_point_source"] not in {
            "live_source_row_projection",
            "fallback_rebind",
        }
        assert pair["provider_simulated_point_source"] not in {
            "live_source_row_projection",
            "fallback_rebind",
        }
        assert pair["provider_pair_fingerprint"] == pair["dataset_pair_fingerprint"]


def _minimal_new4_ladder_context(ladder):
    del ladder
    base_prepared_run, _postprocess_config = _make_prepared_run(joint_background_mode=False)
    prepared_run = replace(
        base_prepared_run,
        fit_params={
            "gamma": 0.2,
            "Gamma": 0.1,
            "chi": 0.0,
            "cor_angle": 0.0,
            "theta_initial": 3.0,
            "theta_offset": 0.0,
            "corto_detector": 0.25,
            "zs": 0.001,
            "zb": 0.002,
            "a": 4.1,
            "c": 28.0,
            "psi_z": 0.0,
            "center_x": 32.0,
            "center_y": 33.0,
            "center": [32.0, 33.0],
        },
        current_dataset={"measured_for_fit": []},
        dataset_specs=[],
        geometry_runtime_cfg={"solver": {}, "optimizer": {}, "bounds": {}},
    )
    return {
        "prepared_run": prepared_run,
        "solver_inputs": geometry_fit.GeometryFitRuntimeSolverInputs(
            miller=[],
            intensities=[],
            image_size=64,
        ),
        "saved_var_names": ["theta_initial"],
    }


def _new4_ladder_provider_identity_pair(
    index: int,
    *,
    full_source: bool = True,
    missing_identity: bool = False,
):
    hkl = (int(index), 0, 1)
    identity = {}
    if not missing_identity:
        identity.update(
            {
                "normalized_hkl": list(hkl),
                "source_table_index": int(index),
                "source_row_index": 0,
                "source_peak_index": int(index % 2),
                "source_branch_index": int(index % 2),
                "source_label": "primary",
                "label": f"{hkl[0]},{hkl[1]},{hkl[2]}",
            }
        )
        if full_source:
            identity.update(
                {
                    "source_reflection_index": int(index),
                    "source_reflection_namespace": "full_reflection",
                    "source_reflection_is_full": True,
                }
            )
    provider_pair = {
        "pair_index": int(index),
        "provider_pair_index": int(index),
        "dataset_pair_index": int(index),
        "selected_source_identity_canonical": identity,
        "background_point": [10.0 + index, 20.0 + index],
        "rebinding_fallback_used": False,
        "fallback_reason": None,
    }
    manual_pair = {
        "pair_index": int(index),
        "provider_pair_index": int(index),
        "dataset_pair_index": int(index),
        "selected_source_identity_canonical": identity,
        "solver_measured_point": [10.0 + index, 20.0 + index],
    }
    measured_row = {
        "hkl": hkl,
        "label": f"{hkl[0]},{hkl[1]},{hkl[2]}",
        "x": 10.0 + index,
        "y": 20.0 + index,
        "fit_source_resolution_kind": "q_group_fallback",
    }
    return provider_pair, manual_pair, measured_row


def _new4_ladder_context_with_provider_rows(
    *,
    full_source: bool,
    missing_identity: bool = False,
):
    base_prepared_run, _postprocess_config = _make_prepared_run(joint_background_mode=False)
    provider_pairs = []
    manual_pairs = []
    measured_rows = []
    miller = []
    for index in range(7):
        provider_pair, manual_pair, measured_row = _new4_ladder_provider_identity_pair(
            index,
            full_source=full_source,
            missing_identity=missing_identity,
        )
        provider_pairs.append(provider_pair)
        manual_pairs.append(manual_pair)
        measured_rows.append(measured_row)
        miller.append([index, 0, 1])
    prepared_run = replace(
        base_prepared_run,
        fit_params={
            "center_x": 32.0,
            "center_y": 33.0,
            "theta_initial": 3.0,
        },
        current_dataset={
            "dataset_index": 0,
            "provider_pairs": provider_pairs,
            "manual_point_pairs": manual_pairs,
            "measured_for_fit": measured_rows,
            "pair_count": 7,
        },
        dataset_specs=[
            {
                "dataset_index": 0,
                "theta_initial": 3.0,
                "measured_peaks": list(measured_rows),
            }
        ],
        geometry_runtime_cfg={"solver": {}, "optimizer": {}, "bounds": {}},
    )
    return {
        "prepared_run": prepared_run,
        "solver_inputs": geometry_fit.GeometryFitRuntimeSolverInputs(
            miller=np.asarray(miller, dtype=float),
            intensities=np.ones(7, dtype=float),
            image_size=64,
        ),
        "saved_var_names": ["theta_initial"],
    }


def _green_new4_ladder_provider_payload():
    return {
        "ok": True,
        "classification": "point_provider_parity_ok",
        "manual_picker_pair_count": 7,
        "point_provider_pair_count": 7,
        "point_provider_parity_gate": {"ok": True},
        "dataset_provider_mismatch_count": 0,
        "fallback_pair_count": 0,
        "optimizer_called": False,
        "provider_guard_ok": True,
        "status": "pass",
        "pass": True,
    }


def _install_fast_new4_ladder_stubs(monkeypatch, ladder):
    monkeypatch.setattr(
        ladder,
        "_capture_solver_context",
        lambda *_args, **_kwargs: _minimal_new4_ladder_context(ladder),
    )

    def _dry_run(_context, *, output_path, max_nfev):
        del _context, max_nfev
        payload = _green_rung1_report()
        payload.update({"after_rms_px": 1.0, "after_max_error_px": 2.0})
        ladder._write_json(output_path, payload)
        return payload

    def _sensitivity(_context, *, output_path, max_nfev, **kwargs):
        del _context, max_nfev, kwargs
        payload = _green_rung2_report(active_params=("center_x", "center_y"))
        payload["active_parameters"] = list(payload["active_params"])
        payload["near_zero_parameters"] = []
        payload["unsafe_parameters"] = []
        ladder._write_json(output_path, payload)
        return payload

    def _solver_rung(*, active_names, output_path, rung, rung_name, **_kwargs):
        payload = _green_one_param_report(active_names)
        payload.update({"rung": rung, "rung_name": rung_name})
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "run_objective_dry_run", _dry_run)
    monkeypatch.setattr(ladder, "run_sensitivity_scan", _sensitivity)
    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver_rung)


def _sensitivity_point_summary(
    *,
    fixed: int = 7,
    fallback: int = 0,
    matched: int = 7,
    missing: int = 0,
    branch_mismatch: int = 0,
) -> dict[str, object]:
    return {
        "fixed_source_resolved_count": int(fixed),
        "matched_fixed_pair_count": int(fixed),
        "fallback_entry_count": int(fallback),
        "matched_pair_count": int(matched),
        "missing_pair_count": int(missing),
        "branch_mismatch_count": int(branch_mismatch),
    }


def _sensitivity_eval(
    label: str,
    *,
    residual_norm: float,
    delta_norm: float = 0.0,
    finite: bool = True,
    moved: bool = True,
    step_applied: float = 0.1,
    clipped: bool = False,
    point_summary: dict[str, object] | None = None,
    include_point_summary: bool = True,
) -> dict[str, object]:
    payload = {
        "label": label,
        "moved": bool(moved),
        "raised": False,
        "error_text": "",
        "residual_norm": float(residual_norm),
        "delta_norm": float(delta_norm),
        "finite": bool(finite),
        "step_applied": float(step_applied),
        "clipped": bool(clipped),
    }
    if include_point_summary:
        payload["point_match_summary"] = point_summary or _sensitivity_point_summary()
    return payload


def _sensitivity_probe_record(
    *,
    base_value: float = 10.0,
    requested_step: float = 0.1,
    plus_delta: float = 1.0,
    minus_delta: float = 1.0,
    plus_finite: bool = True,
    minus_finite: bool = True,
    plus_summary: dict[str, object] | None = None,
    minus_summary: dict[str, object] | None = None,
    plus_moved: bool = True,
    minus_moved: bool = True,
    plus_clipped: bool = False,
    minus_clipped: bool = False,
    plus_step_applied: float | None = None,
    minus_step_applied: float | None = None,
) -> dict[str, object]:
    plus_step = (
        float(plus_step_applied)
        if plus_step_applied is not None
        else (float(requested_step) if plus_moved else 0.0)
    )
    minus_step = (
        float(minus_step_applied)
        if minus_step_applied is not None
        else (-float(requested_step) if minus_moved else 0.0)
    )
    return {
        "base_value": float(base_value),
        "requested_step": float(requested_step),
        "evals": [
            _sensitivity_eval(
                "base",
                residual_norm=2.0,
                step_applied=0.0,
                point_summary=_sensitivity_point_summary(),
            ),
            _sensitivity_eval(
                "plus",
                residual_norm=2.0 + float(plus_delta),
                delta_norm=float(plus_delta),
                finite=bool(plus_finite),
                moved=bool(plus_moved),
                step_applied=plus_step,
                clipped=bool(plus_clipped),
                point_summary=plus_summary,
            ),
            _sensitivity_eval(
                "minus",
                residual_norm=2.0 + float(minus_delta),
                delta_norm=float(minus_delta),
                finite=bool(minus_finite),
                moved=bool(minus_moved),
                step_applied=minus_step,
                clipped=bool(minus_clipped),
                point_summary=minus_summary,
            ),
        ],
    }


def _green_rung1_report() -> dict[str, object]:
    return {
        "rung": 1,
        "rung_name": "objective_dry_run",
        "status": "ok",
        "pass": True,
        "provider_pair_count": 7,
        "dataset_pair_count": 7,
        "optimizer_request_pair_count": 7,
        "fixed_source_pair_count": 7,
        "fallback_row_count": 0,
        "fixed_source_resolution_fallback_count": 0,
        "missing_fixed_source_count": 0,
        "provider_to_optimizer_identity_match": True,
        "provider_to_optimizer_point_match": True,
        "fallback_entry_count": 0,
        "matched_pair_count": 7,
        "missing_pair_count": 0,
        "branch_mismatch_count": 0,
        "objective_dry_run_residual_finite": True,
        "least_squares_called": False,
        "optimizer_solve_called": False,
    }


def _green_rung2_report(
    *,
    active_params=("center_x",),
    near_zero_params=(),
    non_finite_params=(),
    unsafe_params=(),
) -> dict[str, object]:
    active = [str(name) for name in active_params]
    entries = [
        {
            "param_name": name,
            "name": name,
            "status": "active",
            "classification": "active",
            "provider_pair_count": 7,
            "dataset_pair_count": 7,
            "optimizer_request_pair_count": 7,
            "fixed_source_pair_count": 7,
            "fallback_row_count": 0,
            "fixed_source_resolution_fallback_count": 0,
            "missing_fixed_source_count": 0,
            "fixed_source_resolved_count": 7,
            "fallback_entry_count": 0,
            "matched_pair_count": 7,
            "missing_pair_count": 0,
            "branch_mismatch_count": 0,
            "provider_to_optimizer_identity_match": True,
            "provider_to_optimizer_point_match": True,
        }
        for name in active
    ]
    return {
        "rung": 2,
        "rung_name": "sensitivity_scan",
        "status": "ok",
        "pass": True,
        "provider_pair_count": 7,
        "dataset_pair_count": 7,
        "optimizer_request_pair_count": 7,
        "fixed_source_pair_count": 7,
        "fallback_row_count": 0,
        "fixed_source_resolution_fallback_count": 0,
        "missing_fixed_source_count": 0,
        "fixed_source_resolved_count": 7,
        "fallback_entry_count": 0,
        "matched_pair_count": 7,
        "missing_pair_count": 0,
        "branch_mismatch_count": 0,
        "provider_to_optimizer_identity_match": True,
        "provider_to_optimizer_point_match": True,
        "residual_probe_called": True,
        "least_squares_called": False,
        "optimizer_solve_called": False,
        "active_params": active,
        "near_zero_params": [str(name) for name in near_zero_params],
        "non_finite_params": [str(name) for name in non_finite_params],
        "unsafe_params": [str(name) for name in unsafe_params],
        "active_param_count": len(active),
        "near_zero_param_count": len(near_zero_params),
        "non_finite_param_count": len(non_finite_params),
        "unsafe_param_count": len(unsafe_params),
        "state_hash_unchanged": True,
        "params": entries,
    }


def _green_one_param_report(active_names, *, fallback_row_count=0) -> dict[str, object]:
    name = str(active_names[0])
    return {
        "rung": 3,
        "rung_name": f"one_param_{name}",
        "status": "ok",
        "pass": True,
        "param_name": name,
        "active_params": [name],
        "var_names": [name],
        "candidate_param_names": [name],
        "before_rms_px": 10.0,
        "after_rms_px": 9.9,
        "before_max_error_px": 20.0,
        "after_max_error_px": 19.5,
        "residuals_finite": True,
        "residual_norm": 12.0,
        "last_residual_norm": 12.0,
        "parameter_deltas": [
            {
                "name": name,
                "start": 1.0,
                "final": 1.1,
                "delta": 0.1,
                "lower": 0.0,
                "upper": 2.0,
                "within_bounds": True,
            }
        ],
        "nfev": 3,
        "elapsed_seconds": 0.1,
        "elapsed_s": 0.1,
        "timeout_seconds": 120.0,
        "timeout_s": 120.0,
        "rejection_reason": "",
        "fixed_source_pair_count": 7,
        "fallback_row_count": int(fallback_row_count),
        "fixed_source_resolution_fallback_count": 0,
        "missing_fixed_source_count": 0,
        "fixed_source_resolved_count": 7,
        "fallback_entry_count": 0,
        "matched_pair_count": 7,
        "missing_pair_count": 0,
        "branch_mismatch_count": 0,
        "provider_to_optimizer_identity_match": True,
        "provider_to_optimizer_point_match": True,
        "least_squares_called": True,
        "optimizer_solve_called": True,
        "point_match_summary": {},
    }


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _green_pair_report(
    active_names,
    *,
    status="ok",
    fallback_row_count=0,
    after_rms_px=9.8,
    after_max_error_px=19.0,
    dirty_timeout_abort=False,
) -> dict[str, object]:
    names = [str(name) for name in active_names]
    return {
        "rung": 4,
        "rung_name": "pair_" + "_".join(names),
        "status": str(status),
        "pass": str(status) in {"ok", "pass"},
        "active_params": names,
        "var_names": names,
        "candidate_param_names": names,
        "effective_var_names_seen_by_solver": names,
        "before_rms_px": 10.0,
        "after_rms_px": float(after_rms_px),
        "before_max_error_px": 20.0,
        "after_max_error_px": float(after_max_error_px),
        "residuals_finite": True,
        "residual_norm": 12.0,
        "last_residual_norm": 12.0,
        "parameter_deltas": [
            {
                "name": name,
                "start": float(index + 1),
                "final": float(index + 1) + 0.1,
                "delta": 0.1,
                "lower": 0.0,
                "upper": 20.0,
                "within_bounds": True,
            }
            for index, name in enumerate(names)
        ],
        "nfev": 3,
        "elapsed_seconds": 0.1,
        "elapsed_s": 0.1,
        "timeout_seconds": 120.0,
        "timeout_s": 120.0,
        "rejection_reason": "",
        "fixed_source_pair_count": 7,
        "fallback_row_count": int(fallback_row_count),
        "fixed_source_resolution_fallback_count": 0,
        "missing_fixed_source_count": 0,
        "fixed_source_resolved_count": 7,
        "fallback_entry_count": 0,
        "matched_pair_count": 7,
        "missing_pair_count": 0,
        "branch_mismatch_count": 0,
        "provider_to_optimizer_identity_match": True,
        "provider_to_optimizer_point_match": True,
        "least_squares_called": True,
        "optimizer_solve_called": True,
        "real_solve_called": True,
        "dirty_timeout_abort": bool(dirty_timeout_abort),
        "point_match_summary": {},
    }


def _write_pair_evidence(
    ladder,
    tmp_path,
    state_path,
    *,
    passed_params=("chi", "cor_angle", "theta_initial", "corto_detector", "zs", "zb", "c"),
    include_diagnosis=True,
    caked_status="pass",
) -> tuple[Path, Path | None, Path]:
    state_hash = _hash_file(state_path)
    one_param_summary = {
        "status": "ok",
        "state_path": str(state_path.resolve()),
        "background_index": 0,
        "state_sha256_before": state_hash,
        "state_sha256_after": state_hash,
        "state_hash_unchanged": True,
        "provider_guard_after_ok": True,
        "dirty_timeout_abort": False,
        "passed_params": [str(name) for name in passed_params],
    }
    one_param_path = tmp_path / "rung_03_one_param_summary.json"
    ladder._write_json(one_param_path, one_param_summary)

    diagnosis_path: Path | None = None
    if include_diagnosis:
        diagnosis = {
            "status": "ok",
            "one_param_filter": "a",
            "state_path": str(state_path.resolve()),
            "background_index": 0,
            "state_sha256_before": state_hash,
            "state_sha256_after": state_hash,
            "state_hash_unchanged": True,
            "dirty_timeout_abort": False,
            "diagnosis_classification": "usable",
        }
        diagnosis_path = tmp_path / "variant_summary.json"
        ladder._write_json(diagnosis_path, diagnosis)

    caked = {
        "status": str(caked_status),
        "background_index": 0,
        "state_hash_before": state_hash,
        "state_hash_after": state_hash,
        "point_count": 7,
        "exact_projector_available": True,
        "theta_projector_signature_changed": True,
        "distance_projector_signature_changed": True,
        "full_background_recake_call_count": 0,
        "provider_guard_before_ok": True,
        "provider_guard_after_ok": True,
        "new4_state_hash_unchanged": True,
    }
    caked_path = tmp_path / "rung_03b_caked_point_reprojection.json"
    ladder._write_json(caked_path, caked)
    return one_param_path, diagnosis_path, caked_path


def _install_green_provider_guard(monkeypatch, ladder, calls=None) -> None:
    def _green_guard(*, output_path, **_kwargs):
        if calls is not None:
            calls.append(str(output_path))
        payload = _green_new4_ladder_provider_payload()
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "run_provider_guard", _green_guard)
    if hasattr(ladder, "_run_provider_guard_report"):

        def _green_guard_report(*, rung_name="provider_guard_after", **_kwargs):
            if calls is not None:
                calls.append(str(rung_name))
            return _green_new4_ladder_provider_payload()

        monkeypatch.setattr(ladder, "_run_provider_guard_report", _green_guard_report)


def _install_new4_timing_clock(monkeypatch, ladder) -> None:
    ticks = {"value": 0.0}
    base = datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc)

    def _fake_perf_counter() -> float:
        ticks["value"] += 1.0
        return ticks["value"]

    def _fake_utc_now() -> datetime:
        return base + timedelta(seconds=ticks["value"])

    monkeypatch.setattr(ladder, "_perf_counter", _fake_perf_counter)
    monkeypatch.setattr(ladder, "_utc_now", _fake_utc_now)


def _write_new4_timing_fixture_reports(ladder, run_dir: Path, rung_ids) -> dict[str, object]:
    collector = ladder._TimingCollector(
        run_dir=run_dir,
        expected_rung_ids=tuple(str(rung_id) for rung_id in rung_ids),
    )
    token = ladder._ACTIVE_TIMING_COLLECTOR.set(collector)
    try:
        for index, rung_id in enumerate(rung_ids):
            rung_id = str(rung_id)
            if rung_id == "3A":
                path = run_dir / "rung_03a_a_diagnosis" / "variant_summary.json"
                rung_name = "a_diagnosis"
                rung = 3
            elif rung_id == "3B":
                path = (
                    run_dir
                    / "rung_03b_caked_point_reprojection"
                    / "rung_03b_caked_point_reprojection.json"
                )
                rung_name = "caked_point_reprojection"
                rung = 3
            else:
                path = run_dir / f"rung_{int(rung_id):02d}_fixture.json"
                rung_name = f"rung_{rung_id}_fixture"
                rung = int(rung_id)
            with ladder._timed_report_window(rung_id, rung_name):
                ladder._write_json(
                    path,
                    {
                        "rung": rung,
                        "rung_name": rung_name,
                        "status": "ok",
                        "pass": True,
                        "stage_timing_s": {"build": float(index + 1) / 10.0},
                    },
                )
        return collector.summary()
    finally:
        ladder._ACTIVE_TIMING_COLLECTOR.reset(token)


def test_new4_ladder_timing_summary_schema(monkeypatch, tmp_path) -> None:
    ladder = _load_new4_ladder_module()
    _install_new4_timing_clock(monkeypatch, ladder)
    run_dir = tmp_path / "timing"
    run_dir.mkdir()

    summary = _write_new4_timing_fixture_reports(ladder, run_dir, ("0", "1", "2", "5"))

    assert summary["timing_collection_mode"] == "current_run"
    assert summary["completed_rung_count"] == 4
    assert summary["missing_expected_rungs"] == []
    assert summary["slowest_rung"]
    assert isinstance(summary["slowest_rung_elapsed_s"], float)
    for item in summary["rung_timings"]:
        assert {"rung_id", "rung_index", "rung_name", "status", "elapsed_s", "report_path"} <= set(item)
        assert np.isfinite(float(item["elapsed_s"]))

    report = json.loads((run_dir / "rung_00_fixture.json").read_text(encoding="utf-8"))
    for key in (
        "started_at_iso",
        "finished_at_iso",
        "elapsed_s",
        "elapsed_seconds",
        "stage_elapsed_s",
        "run_id",
        "run_dir",
        "rung_id",
        "rung_index",
        "rung_name",
        "report_path",
    ):
        assert key in report
    assert np.isfinite(float(report["elapsed_s"]))


def test_new4_ladder_expected_timing_ids_stop_at_rung_5() -> None:
    ladder = _load_new4_ladder_module()
    allowed = {"0", "1", "2", "3", "3A", "3B", "4", "5"}

    for max_rung in ("combined", "selected", "feature", "features"):
        ids = ladder._expected_rung_ids_for_run(max_rung)
        assert set(ids) <= allowed
        assert "6" not in ids
        assert "7" not in ids


def test_new4_ladder_timing_summary_writes_explicit_copy(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    _install_new4_timing_clock(monkeypatch, ladder)
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    explicit_timing_path = tmp_path / "latest_timing_summary.json"
    _install_green_provider_guard(monkeypatch, ladder)
    _install_fast_new4_ladder_stubs(monkeypatch, ladder)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="sensitivity",
        timestamp="timed_run",
        timing_report=explicit_timing_path,
    )

    run_timing_path = tmp_path / "timed_run" / "rung_timing_summary.json"
    run_summary = json.loads(run_timing_path.read_text(encoding="utf-8"))
    explicit_summary = json.loads(explicit_timing_path.read_text(encoding="utf-8"))
    assert result["status"] == "pass"
    assert run_summary == explicit_summary
    assert {item["rung_id"] for item in run_summary["rung_timings"]} == {"0", "1", "2"}


def test_new4_ladder_timing_summary_does_not_change_pass_fail(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    _install_new4_timing_clock(monkeypatch, ladder)
    run_dir = tmp_path / "timing"
    run_dir.mkdir()
    collector = ladder._TimingCollector(run_dir=run_dir, expected_rung_ids=("0", "5"))
    token = ladder._ACTIVE_TIMING_COLLECTOR.set(collector)
    try:
        ok_path = run_dir / "rung_00_provider_guard.json"
        skipped_path = run_dir / "rung_05_block_skipped.json"
        ladder._write_json(
            ok_path,
            {"rung": 0, "rung_name": "provider_guard", "status": "ok", "pass": True, "elapsed_s": 99.0},
        )
        ladder._write_json(
            skipped_path,
            {"rung": 5, "rung_name": "block_skipped", "status": "skipped", "pass": False, "elapsed_s": 33.0},
        )
        summary = collector.summary()
    finally:
        ladder._ACTIVE_TIMING_COLLECTOR.reset(token)

    ok_report = json.loads(ok_path.read_text(encoding="utf-8"))
    skipped_report = json.loads(skipped_path.read_text(encoding="utf-8"))
    assert ok_report["status"] == "ok"
    assert ok_report["pass"] is True
    assert skipped_report["status"] == "skipped"
    assert skipped_report["pass"] is False
    assert [item["status"] for item in summary["rung_timings"]] == ["ok", "skipped"]


def test_new4_ladder_timing_report_prints_all_completed_rungs(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    _install_new4_timing_clock(monkeypatch, ladder)
    run_dir = tmp_path / "timing"
    run_dir.mkdir()
    rung_ids = ("0", "1", "2", "3", "3A", "3B", "4", "5")

    summary = _write_new4_timing_fixture_reports(ladder, run_dir, rung_ids)
    table = ladder._format_timing_table(summary)

    assert table.splitlines()[0] == "Rung | Status | elapsed_s | report_path"
    for rung_id in rung_ids:
        assert f"{rung_id} | ok |" in table

def test_new4_ladder_timing_threshold_optional(monkeypatch, tmp_path) -> None:
    ladder = _load_new4_ladder_module()
    _install_new4_timing_clock(monkeypatch, ladder)
    run_dir = tmp_path / "timing"
    run_dir.mkdir()
    collector = ladder._TimingCollector(run_dir=run_dir, expected_rung_ids=("5",))
    token = ladder._ACTIVE_TIMING_COLLECTOR.set(collector)
    try:
        path = run_dir / "rung_05_block_slow.json"
        ladder._write_json(
            path,
            {
                "rung": 5,
                "rung_name": "block_slow",
                "status": "ok",
                "pass": True,
                "elapsed_s": 5000.0,
            },
        )
        monkeypatch.delenv("RA_SIM_NEW4_LADDER_TIMING_MAX_S", raising=False)
        unset_summary = collector.summary()
        monkeypatch.setenv("RA_SIM_NEW4_LADDER_TIMING_MAX_S", "1")
        exceeded_summary = collector.summary()
    finally:
        ladder._ACTIVE_TIMING_COLLECTOR.reset(token)

    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert report["pass"] is True
    assert unset_summary["timing_threshold_status"] == "not_configured"
    assert unset_summary["timing_threshold_exceeded_rungs"] == []
    assert exceeded_summary["timing_threshold_status"] == "exceeded"
    assert exceeded_summary["timing_threshold_exceeded_rungs"][0]["rung_id"] == "5"


def test_new4_ladder_runs_provider_guard_before_optimizer(monkeypatch, tmp_path) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")

    def _failing_guard(*, output_path, **_kwargs):
        payload = {
            "ok": False,
            "classification": "point_provider_parity_failed",
            "provider_guard_ok": False,
            "provider_guard_failures": ["forced"],
        }
        ladder._write_json(output_path, payload)
        return payload

    def _optimizer_boundary(*_args, **_kwargs):
        raise AssertionError("optimizer path should not run after guard failure")

    monkeypatch.setattr(ladder, "run_provider_guard", _failing_guard)
    monkeypatch.setattr(ladder, "_capture_solver_context", _optimizer_boundary)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="center",
        timestamp="guard_fail",
    )

    assert result["status"] == "aborted"
    assert result["reason"] == "provider_guard_failed"
    assert (tmp_path / "guard_fail" / "rung_00_provider_guard.json").exists()


def test_new4_ladder_parser_accepts_sensitivity_max_rung() -> None:
    ladder = _load_new4_ladder_module()
    parser = ladder.build_arg_parser()

    args = parser.parse_args(
        [
            "--state",
            "new4.json",
            "--background-index",
            "0",
            "--output-root",
            "out",
            "--max-rung",
            "sensitivity",
        ]
    )

    assert args.max_rung == "sensitivity"
    assert args.use_subprocess is False
    assert args.diagnostic_logging is False

    diagnostic_args = parser.parse_args(
        [
            "--state",
            "new4.json",
            "--background-index",
            "0",
            "--output-root",
            "out",
            "--use-subprocess",
            "--diagnostic-logging",
        ]
    )

    assert diagnostic_args.use_subprocess is True
    assert diagnostic_args.diagnostic_logging is True


def test_new4_ladder_sensitivity_max_rung_stops_before_solve(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")

    def _green_guard(*, output_path, **_kwargs):
        payload = _green_new4_ladder_provider_payload()
        ladder._write_json(output_path, payload)
        return payload

    def _dry_run(_context, *, output_path, max_nfev):
        del _context, max_nfev
        payload = _green_rung1_report()
        ladder._write_json(output_path, payload)
        return payload

    def _sensitivity(_context, *, output_path, max_nfev, **kwargs):
        del _context, max_nfev, kwargs
        payload = {
            "rung": 2,
            "rung_name": "sensitivity_scan",
            "status": "ok",
            "pass": True,
            "residual_probe_called": True,
            "least_squares_called": False,
            "optimizer_solve_called": False,
            "active_params": ["center_x"],
        }
        ladder._write_json(output_path, payload)
        return payload

    def _solver_rung(**_kwargs):
        raise AssertionError("solve rung must not run for max_rung=sensitivity")

    monkeypatch.setattr(ladder, "run_provider_guard", _green_guard)
    monkeypatch.setattr(
        ladder,
        "_capture_solver_context",
        lambda *_args, **_kwargs: _minimal_new4_ladder_context(ladder),
    )
    monkeypatch.setattr(ladder, "run_objective_dry_run", _dry_run)
    monkeypatch.setattr(ladder, "run_sensitivity_scan", _sensitivity)
    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver_rung)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="sensitivity",
        timestamp="sensitivity_stop",
    )
    run_dir = tmp_path / "sensitivity_stop"

    assert result["status"] == "pass"
    assert (run_dir / "rung_01_objective_dry_run.json").exists()
    assert (run_dir / "rung_02_sensitivity_scan.json").exists()
    assert not list(run_dir.glob("rung_03_*.json"))
    assert result["residual_probe_called"] is True
    assert result["least_squares_called"] is False
    assert result["optimizer_solve_called"] is False


def test_new4_ladder_aborts_before_sensitivity_when_rung1_not_green(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")

    def _green_guard(*, output_path, **_kwargs):
        payload = _green_new4_ladder_provider_payload()
        ladder._write_json(output_path, payload)
        return payload

    def _fallback_dry_run(_context, *, output_path, max_nfev):
        del _context, max_nfev
        payload = dict(_green_rung1_report())
        payload.update(
            {
                "status": "failed",
                "pass": False,
                "fallback_row_count": 1,
                "fixed_source_resolution_fallback_count": 1,
            }
        )
        ladder._write_json(output_path, payload)
        return payload

    def _sensitivity(*_args, **_kwargs):
        raise AssertionError("sensitivity must not run after rung 1 fallback")

    monkeypatch.setattr(ladder, "run_provider_guard", _green_guard)
    monkeypatch.setattr(
        ladder,
        "_capture_solver_context",
        lambda *_args, **_kwargs: _minimal_new4_ladder_context(ladder),
    )
    monkeypatch.setattr(ladder, "run_objective_dry_run", _fallback_dry_run)
    monkeypatch.setattr(ladder, "run_sensitivity_scan", _sensitivity)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="sensitivity",
        timestamp="rung1_fail",
    )

    assert result["status"] == "aborted"
    assert result["reason"] == "objective_dry_run_failed"
    assert result["residual_probe_called"] is False
    assert result["least_squares_called"] is False
    assert not (tmp_path / "rung1_fail" / "rung_02_sensitivity_scan.json").exists()


def test_new4_ladder_sensitivity_direct_call_aborts_on_bad_rung1(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    context = _new4_ladder_context_with_provider_rows(full_source=True)
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    output_path = tmp_path / "rung_02_sensitivity_scan.json"
    bad_rung1 = dict(_green_rung1_report())
    bad_rung1.update(
        {
            "status": "fail",
            "pass": False,
            "provider_pair_count": "bad",
            "provider_to_optimizer_identity_match": "false",
            "provider_to_optimizer_point_match": "false",
            "fallback_row_count": 1,
            "fixed_source_resolution_fallback_count": 1,
        }
    )

    def _probe_boundary(*_args, **_kwargs):
        raise AssertionError("sensitivity probe must not run after bad rung 1")

    monkeypatch.setattr(ladder, "_run_with_probe_least_squares", _probe_boundary)

    report = ladder.run_sensitivity_scan(
        context,
        output_path=output_path,
        max_nfev=20,
        rung_1_report=bad_rung1,
        state_path=state_path,
    )
    written = json.loads(output_path.read_text(encoding="utf-8"))

    assert report == written
    assert report["status"] == "aborted"
    assert report["pass"] is False
    assert report["reason"] == "rung_1_not_green"
    assert "rung_1_status_not_ok" in report["rung_1_failures"]
    assert "provider_pair_count_-999999_expected_7" in report["rung_1_failures"]
    assert "provider_to_optimizer_identity_match_false_expected_True" in report["rung_1_failures"]
    assert "provider_to_optimizer_point_match_false_expected_True" in report["rung_1_failures"]
    assert report["provider_pair_count"] == 0
    assert report["provider_to_optimizer_identity_match"] is False
    assert report["provider_to_optimizer_point_match"] is False
    assert report["residual_probe_called"] is False
    assert report["least_squares_called"] is False
    assert report["optimizer_solve_called"] is False
    assert report["params"] == []


def test_new4_ladder_sensitivity_rejects_stale_green_rung1(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    context = _new4_ladder_context_with_provider_rows(full_source=True)
    output_path = tmp_path / "rung_02_sensitivity_scan.json"
    stale_rung1 = dict(_green_rung1_report())
    for key in (
        "dataset_pair_count",
        "optimizer_request_pair_count",
        "provider_to_optimizer_identity_match",
        "provider_to_optimizer_point_match",
    ):
        stale_rung1.pop(key)

    def _probe_boundary(*_args, **_kwargs):
        raise AssertionError("sensitivity probe must not run after stale rung 1")

    monkeypatch.setattr(ladder, "_run_with_probe_least_squares", _probe_boundary)

    report = ladder.run_sensitivity_scan(
        context,
        output_path=output_path,
        max_nfev=20,
        rung_1_report=stale_rung1,
    )

    assert report["status"] == "aborted"
    assert report["reason"] == "rung_1_not_green"
    assert "dataset_pair_count_-999999_expected_7" in report["rung_1_failures"]
    assert "optimizer_request_pair_count_-999999_expected_7" in report["rung_1_failures"]
    assert "provider_to_optimizer_identity_match_None_expected_True" in report["rung_1_failures"]
    assert "provider_to_optimizer_point_match_None_expected_True" in report["rung_1_failures"]
    assert report["residual_probe_called"] is False


def test_new4_local_headless_path_helper_reports_missing_required_paths(
    tmp_path,
) -> None:
    existing_background = tmp_path / "existing.osc"
    missing_background = tmp_path / "missing.osc"
    missing_cif = tmp_path / "missing.cif"
    state_path = tmp_path / "new4.json"
    existing_background.write_text("placeholder", encoding="utf-8")
    state_path.write_text(
        json.dumps(
            {
                "state": {
                    "files": {
                        "background_files": [
                            str(existing_background),
                            str(missing_background),
                        ],
                        "primary_cif_path": str(missing_cif),
                        "secondary_cif_path": None,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert _required_new4_local_headless_paths(state_path) == [
        existing_background,
        missing_background,
        missing_cif,
    ]
    assert _missing_new4_local_headless_paths(state_path) == [
        missing_background,
        missing_cif,
    ]


def test_new4_ladder_sensitivity_real_new4_cli_smoke(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = Path("artifacts/geometry_fit_gui_states/new4.json")
    if not state_path.exists():
        pytest.skip("new4 artifact is not present")
    missing_paths = _missing_new4_local_headless_paths(state_path)
    if missing_paths:
        pytest.skip(
            "new4 local headless data paths are unavailable: "
            + ", ".join(str(path) for path in missing_paths)
        )
    output_root = tmp_path / "ladder"
    monkeypatch.setattr(ladder, "_run_stamp", lambda: "cli_sensitivity_smoke")

    exit_code = ladder.main(
        [
            "--state",
            str(state_path),
            "--background-index",
            "0",
            "--output-root",
            str(output_root),
            "--max-rung",
            "sensitivity",
            "--max-nfev",
            "20",
            "--timeout-seconds",
            "120",
        ]
    )
    run_dir = output_root / "cli_sensitivity_smoke"
    rung1_path = run_dir / "rung_01_objective_dry_run.json"
    rung2_path = run_dir / "rung_02_sensitivity_scan.json"

    assert exit_code == 0
    assert rung1_path.exists()
    assert rung2_path.exists()
    assert not [
        path
        for path in run_dir.glob("rung_*.json")
        if path.name.startswith(("rung_03_", "rung_04_", "rung_05_"))
    ]
    rung1 = json.loads(rung1_path.read_text(encoding="utf-8"))
    rung2 = json.loads(rung2_path.read_text(encoding="utf-8"))
    assert rung1["status"] == "ok"
    assert rung1["fixed_source_pair_count"] == 7
    assert rung1["fallback_row_count"] == 0
    assert rung1["least_squares_called"] is False
    assert rung2["residual_probe_called"] is True
    assert rung2["least_squares_called"] is False
    assert rung2["optimizer_solve_called"] is False
    assert rung2["status"] == "ok"
    for param in rung2["params"]:
        for direction in ("base_eval", "plus_eval", "minus_eval"):
            eval_report = param[direction]
            if not bool(eval_report["moved"]):
                assert eval_report["counter_source"] == "not_evaluated"
                continue
            assert eval_report["counter_source"] == "point_match_summary"
            assert eval_report["fixed_source_clean"] is True
            assert eval_report["fixed_source_pair_count"] == 7
            assert eval_report["fallback_entry_count"] == 0
            assert eval_report["matched_pair_count"] == 7
            assert eval_report["missing_pair_count"] == 0
            assert eval_report["branch_mismatch_count"] == 0


def test_new4_ladder_sensitivity_does_not_call_real_least_squares(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    context = _new4_ladder_context_with_provider_rows(full_source=True)
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    called = {"least_squares": False}

    def _raising_least_squares(*_args, **_kwargs):
        called["least_squares"] = True
        raise AssertionError("real least_squares must not be called")

    monkeypatch.setattr(ladder.opt, "least_squares", _raising_least_squares)

    report = ladder.run_sensitivity_scan(
        context,
        output_path=tmp_path / "rung_02_sensitivity_scan.json",
        max_nfev=20,
        rung_1_report=_green_rung1_report(),
        state_path=state_path,
    )

    assert report["residual_probe_called"] is True
    assert report["least_squares_called"] is False
    assert report["optimizer_solve_called"] is False
    assert called["least_squares"] is False


def test_new4_ladder_sensitivity_request_fallback_marks_unsafe_without_probe(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    context = _new4_ladder_context_with_provider_rows(full_source=True)
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")

    monkeypatch.setattr(
        ladder,
        "_strict_no_fallback_failures",
        lambda _summary: ["forced_request_fallback"],
    )

    def _probe_boundary(*_args, **_kwargs):
        raise AssertionError("residual probe must not run after request fallback")

    monkeypatch.setattr(ladder, "_run_with_probe_least_squares", _probe_boundary)

    report = ladder.run_sensitivity_scan(
        context,
        output_path=tmp_path / "rung_02_sensitivity_scan.json",
        max_nfev=20,
        rung_1_report=_green_rung1_report(),
        state_path=state_path,
    )

    assert report["residual_probe_called"] is False
    assert report["least_squares_called"] is False
    assert report["optimizer_solve_called"] is False
    assert report["unsafe_param_count"] == len(report["params"])
    assert all(entry["status"] == "unsafe" for entry in report["params"])


def test_new4_ladder_sensitivity_reports_fixed_source_counter_breaks(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    context = _new4_ladder_context_with_provider_rows(full_source=True)
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    broken_plus = _sensitivity_point_summary(fallback=1)
    records = {
        "center_x": _sensitivity_probe_record(plus_delta=1.0, minus_delta=0.5),
        "center_y": _sensitivity_probe_record(
            plus_delta=1.0,
            minus_delta=0.5,
            plus_summary=broken_plus,
        ),
        "theta_initial": _sensitivity_probe_record(
            plus_delta=0.0,
            minus_delta=0.0,
        ),
    }

    def _fake_probe(request, *, mode):
        assert mode == "sensitivity"
        return SimpleNamespace(), [records[request.var_names[0]]]

    monkeypatch.setattr(ladder, "_run_with_probe_least_squares", _fake_probe)

    report = ladder.run_sensitivity_scan(
        context,
        output_path=tmp_path / "rung_02_sensitivity_scan.json",
        max_nfev=20,
        rung_1_report=_green_rung1_report(),
        state_path=state_path,
    )
    by_name = {entry["param_name"]: entry for entry in report["params"]}

    assert by_name["center_x"]["status"] == "active"
    assert by_name["center_y"]["status"] == "unsafe"
    assert by_name["center_y"]["plus_eval"]["fallback_entry_count"] == 1
    assert by_name["center_y"]["plus_eval"]["fixed_source_clean"] is False
    assert by_name["theta_initial"]["status"] == "near_zero"
    for entry in report["params"]:
        for direction in ("base_eval", "plus_eval", "minus_eval"):
            eval_report = entry[direction]
            for key in (
                "fixed_source_pair_count",
                "fallback_entry_count",
                "matched_pair_count",
                "missing_pair_count",
                "branch_mismatch_count",
            ):
                assert key in eval_report


def test_new4_ladder_sensitivity_missing_probe_summary_marks_param_unsafe(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    context = _new4_ladder_context_with_provider_rows(full_source=True)
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    record = _sensitivity_probe_record(plus_delta=1.0, minus_delta=0.5)
    del record["evals"][1]["point_match_summary"]
    records = {
        "center_x": record,
        "center_y": _sensitivity_probe_record(plus_delta=1.0, minus_delta=0.5),
        "theta_initial": _sensitivity_probe_record(plus_delta=0.0, minus_delta=0.0),
    }

    def _fake_probe(request, *, mode):
        assert mode == "sensitivity"
        return SimpleNamespace(), [records[request.var_names[0]]]

    monkeypatch.setattr(ladder, "_run_with_probe_least_squares", _fake_probe)

    report = ladder.run_sensitivity_scan(
        context,
        output_path=tmp_path / "rung_02_sensitivity_scan.json",
        max_nfev=20,
        rung_1_report=_green_rung1_report(),
        state_path=state_path,
    )
    by_name = {entry["param_name"]: entry for entry in report["params"]}

    assert by_name["center_x"]["status"] == "unsafe"
    assert by_name["center_x"]["plus_eval"]["counter_source"] == "missing"
    assert by_name["center_x"]["plus_eval"]["fixed_source_clean"] is False
    assert "plus_missing_point_match_summary" in by_name["center_x"]["unsafe_reasons"]
    assert by_name["center_y"]["status"] == "active"


def test_new4_ladder_sensitivity_eval_counters_require_point_summary() -> None:
    ladder = _load_new4_ladder_module()

    summary = ladder._sensitivity_eval_summary(
        {
            "label": "plus",
            "moved": True,
            "finite": True,
            "delta_norm": 1.0,
            "step_applied": 0.5,
        },
        request_summary={
            "fixed_source_pair_count": 7,
            "fallback_row_count": 0,
            "matched_pair_count": 7,
            "missing_pair_count": 0,
            "branch_mismatch_count": 0,
        },
    )

    assert summary["counter_source"] == "missing"
    assert summary["fixed_source_pair_count"] == -1
    assert summary["fallback_entry_count"] == -1
    assert summary["matched_pair_count"] == -1
    assert summary["missing_pair_count"] == -1
    assert summary["branch_mismatch_count"] == -1
    assert summary["fixed_source_clean"] is False
    assert summary["fixed_source_failures"] == ["missing_point_match_summary"]

    dirty = ladder._sensitivity_eval_summary(
        {
            "label": "plus",
            "moved": True,
            "finite": True,
            "delta_norm": 1.0,
            "step_applied": 0.5,
            "point_match_summary": {
                "fixed_source_resolved_count": None,
                "fallback_entry_count": 0,
                "matched_pair_count": "seven",
                "missing_pair_count": 0,
                "branch_mismatch_count": float("nan"),
            },
        },
        request_summary={
            "fixed_source_pair_count": 7,
            "fallback_row_count": 0,
            "matched_pair_count": 7,
            "missing_pair_count": 0,
            "branch_mismatch_count": 0,
        },
    )

    assert dirty["counter_source"] == "missing"
    assert dirty["fixed_source_pair_count"] == -1
    assert dirty["matched_pair_count"] == -1
    assert dirty["branch_mismatch_count"] == -1
    assert dirty["fixed_source_clean"] is False
    assert "fixed_source_pair_count_-1_expected_7" in dirty["fixed_source_failures"]
    assert "matched_pair_count_-1_expected_7" in dirty["fixed_source_failures"]
    assert "branch_mismatch_count_-1_expected_0" in dirty["fixed_source_failures"]

    clipped = ladder._sensitivity_eval_summary(
        {
            "label": "minus",
            "moved": False,
            "clipped": True,
            "step_applied": 0.0,
        },
        request_summary={},
    )

    assert clipped["counter_source"] == "not_evaluated"
    assert clipped["fixed_source_clean"] is True
    assert clipped["fixed_source_failures"] == []


def test_new4_ladder_sensitivity_probe_uses_fresh_base_vector() -> None:
    ladder = _load_new4_ladder_module()
    probe = ladder._ProbeLeastSquares(mode="sensitivity")
    seen: list[tuple[float, float]] = []

    def _residual(vector):
        arr = np.asarray(vector, dtype=float)
        seen.append((float(arr[0]), float(arr[1])))
        return np.asarray([arr[0], arr[1]], dtype=float)

    base = np.asarray([10.0, 20.0], dtype=float)
    bounds = (
        np.asarray([0.0, 0.0], dtype=float),
        np.asarray([100.0, 100.0], dtype=float),
    )

    probe(_residual, base, bounds=bounds)
    probe(_residual, base, bounds=bounds)

    first = seen[:3]
    second = seen[3:6]
    assert first[0] == (10.0, 20.0)
    assert first[1][0] > 10.0 and first[1][1] == 20.0
    assert first[2][0] < 10.0 and first[2][1] == 20.0
    assert second[0] == (10.0, 20.0)
    assert second[1][0] > 10.0 and second[1][1] == 20.0
    assert second[2][0] < 10.0 and second[2][1] == 20.0


def test_new4_ladder_sensitivity_probe_records_bound_clipping() -> None:
    ladder = _load_new4_ladder_module()

    def _record(base_value: float, lower: float, upper: float) -> dict[str, object]:
        probe = ladder._ProbeLeastSquares(mode="sensitivity")

        def _residual(vector):
            arr = np.asarray(vector, dtype=float)
            return np.asarray([arr[0]], dtype=float)

        probe(
            _residual,
            np.asarray([base_value], dtype=float),
            bounds=(
                np.asarray([lower], dtype=float),
                np.asarray([upper], dtype=float),
            ),
        )
        return probe.records[0]

    both_valid = _record(10.0, 0.0, 20.0)
    plus_clipped = _record(10.0, 0.0, 10.0)
    minus_clipped = _record(10.0, 10.0, 20.0)
    both_clipped = _record(10.0, 10.0, 10.0)

    assert both_valid["plus_step_applied"] > 0.0
    assert both_valid["minus_step_applied"] < 0.0
    assert both_valid["plus_clipped"] is False
    assert both_valid["minus_clipped"] is False
    assert plus_clipped["plus_step_applied"] == 0.0
    assert plus_clipped["minus_step_applied"] < 0.0
    assert plus_clipped["plus_clipped"] is True
    assert minus_clipped["plus_step_applied"] > 0.0
    assert minus_clipped["minus_step_applied"] == 0.0
    assert minus_clipped["minus_clipped"] is True
    assert both_clipped["plus_step_applied"] == 0.0
    assert both_clipped["minus_step_applied"] == 0.0


def test_new4_ladder_sensitivity_status_uses_applied_step_and_no_move_unsafe(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    context = _new4_ladder_context_with_provider_rows(full_source=True)
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    records = {
        "center_x": _sensitivity_probe_record(
            requested_step=1.0,
            plus_delta=2.0,
            minus_delta=0.0,
            plus_step_applied=0.25,
            minus_moved=False,
            plus_clipped=True,
        ),
        "center_y": _sensitivity_probe_record(
            plus_delta=1.0,
            minus_delta=1.0,
            plus_moved=False,
            minus_moved=False,
            plus_clipped=True,
            minus_clipped=True,
        ),
        "theta_initial": _sensitivity_probe_record(
            plus_delta=0.0,
            minus_delta=0.0,
        ),
    }

    def _fake_probe(request, *, mode):
        assert mode == "sensitivity"
        return SimpleNamespace(), [records[request.var_names[0]]]

    monkeypatch.setattr(ladder, "_run_with_probe_least_squares", _fake_probe)

    report = ladder.run_sensitivity_scan(
        context,
        output_path=tmp_path / "rung_02_sensitivity_scan.json",
        max_nfev=20,
        rung_1_report=_green_rung1_report(),
        state_path=state_path,
    )
    by_name = {entry["param_name"]: entry for entry in report["params"]}

    assert by_name["center_x"]["plus_clipped"] is True
    assert by_name["center_x"]["minus_eval"]["moved"] is False
    assert by_name["center_x"]["minus_eval"]["counter_source"] == "not_evaluated"
    assert by_name["center_x"]["sensitivity_norm"] == pytest.approx(8.0)
    assert by_name["center_y"]["status"] == "unsafe"
    assert "no_valid_movement" in by_name["center_y"]["unsafe_reasons"]


def test_new4_ladder_active_theta_name_selects_one_active_theta() -> None:
    ladder = _load_new4_ladder_module()
    base_context = _minimal_new4_ladder_context(ladder)
    assert ladder._active_theta_name(base_context) == "theta_initial"
    assert [
        name
        for name in ladder._candidate_order(base_context)
        if name in {"theta_initial", "theta_offset"}
    ] == ["theta_initial"]

    offset_context = dict(base_context)
    offset_context["saved_var_names"] = ["theta_offset"]
    assert ladder._active_theta_name(offset_context) == "theta_offset"
    assert [
        name
        for name in ladder._candidate_order(offset_context)
        if name in {"theta_initial", "theta_offset"}
    ] == ["theta_offset"]

    prepared_run = base_context["prepared_run"]
    no_theta_context = dict(base_context)
    no_theta_context["prepared_run"] = replace(
        prepared_run,
        fit_params={
            key: value
            for key, value in prepared_run.fit_params.items()
            if key not in {"theta_initial", "theta_offset"}
        },
    )
    no_theta_context["saved_var_names"] = []
    assert ladder._active_theta_name(no_theta_context) is None
    assert [
        name
        for name in ladder._candidate_order(no_theta_context)
        if name in {"theta_initial", "theta_offset"}
    ] == []


def test_new4_ladder_sensitivity_report_schema_and_summary_counts(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    context = _new4_ladder_context_with_provider_rows(full_source=True)
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    records = {
        "center_x": _sensitivity_probe_record(plus_delta=1.0, minus_delta=0.5),
        "center_y": _sensitivity_probe_record(
            plus_delta=1e-12,
            minus_delta=1e-12,
        ),
        "theta_initial": _sensitivity_probe_record(
            plus_delta=1.0,
            minus_delta=1.0,
            plus_finite=False,
        ),
    }

    def _fake_probe(request, *, mode):
        assert mode == "sensitivity"
        return SimpleNamespace(), [records[request.var_names[0]]]

    monkeypatch.setattr(ladder, "_run_with_probe_least_squares", _fake_probe)

    report = ladder.run_sensitivity_scan(
        context,
        output_path=tmp_path / "rung_02_sensitivity_scan.json",
        max_nfev=20,
        rung_1_report=_green_rung1_report(),
        state_path=state_path,
    )

    for key in (
        "status",
        "rung_1_status",
        "provider_pair_count",
        "fixed_source_pair_count",
        "fallback_entry_count",
        "residual_probe_called",
        "least_squares_called",
        "optimizer_solve_called",
        "active_param_count",
        "near_zero_param_count",
        "non_finite_param_count",
        "unsafe_param_count",
        "active_params",
        "unsafe_params",
        "state_hash_unchanged",
        "params",
    ):
        assert key in report

    assert report["active_param_count"] == 1
    assert report["near_zero_param_count"] == 1
    assert report["non_finite_param_count"] == 1
    assert report["unsafe_param_count"] == 0
    assert (
        report["active_param_count"]
        + report["near_zero_param_count"]
        + report["non_finite_param_count"]
        + report["unsafe_param_count"]
    ) == len(report["params"])
    assert report["state_hash_unchanged"] is True
    assert report["least_squares_called"] is False
    assert report["optimizer_solve_called"] is False


def test_new4_ladder_one_param_rung_uses_candidate_param_names(monkeypatch) -> None:
    ladder = _load_new4_ladder_module()
    request = ladder.build_solver_request(
        _minimal_new4_ladder_context(ladder),
        ["gamma"],
        max_nfev=20,
    )
    captured: dict[str, object] = {}

    def _fake_solver(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks,
        var_names,
        *,
        candidate_param_names=None,
        **_kwargs,
    ):
        del miller, intensities, image_size, params, measured_peaks
        captured["var_names"] = list(var_names)
        captured["candidate_param_names"] = list(candidate_param_names or [])
        return SimpleNamespace()

    geometry_fit.solve_geometry_fit_request(request, solve_fit=_fake_solver)

    assert captured["var_names"] == ["gamma"]
    assert captured["candidate_param_names"] == ["gamma"]
    assert request.candidate_param_names == ["gamma"]


def test_new4_ladder_one_param_parser_accepts_max_rung() -> None:
    ladder = _load_new4_ladder_module()
    parser = ladder.build_arg_parser()

    args = parser.parse_args(
        [
            "--state",
            "new4.json",
            "--background-index",
            "0",
            "--output-root",
            "out",
            "--max-rung",
            "one-param",
            "--one-param-filter",
            "a",
        ]
    )

    assert args.max_rung == "one-param"
    assert args.one_param_filter == "a"


def test_new4_ladder_pair_parser_accepts_evidence_inputs() -> None:
    ladder = _load_new4_ladder_module()
    parser = ladder.build_arg_parser()

    args = parser.parse_args(
        [
            "--state",
            "new4.json",
            "--background-index",
            "0",
            "--output-root",
            "out",
            "--max-rung",
            "pairs",
            "--one-param-summary",
            "one.json",
            "--one-param-diagnosis-summary",
            "diag.json",
            "--caked-point-reprojection-report",
            "caked.json",
        ]
    )
    pair_args = parser.parse_args(
        [
            "--state",
            "new4.json",
            "--background-index",
            "0",
            "--output-root",
            "out",
            "--max-rung",
            "pair",
        ]
    )

    assert args.max_rung == "pairs"
    assert args.one_param_summary == "one.json"
    assert args.one_param_diagnosis_summary == "diag.json"
    assert args.caked_point_reprojection_report == "caked.json"
    assert pair_args.max_rung == "pair"


def _install_pair_rung_common_stubs(
    monkeypatch,
    ladder,
    *,
    active_params,
) -> None:
    _install_green_provider_guard(monkeypatch, ladder)
    monkeypatch.setattr(
        ladder,
        "_capture_solver_context",
        lambda *_args, **_kwargs: _minimal_new4_ladder_context(ladder),
    )
    monkeypatch.setattr(
        ladder,
        "run_objective_dry_run",
        lambda _context, *, output_path, max_nfev: (
            ladder._write_json(output_path, _green_rung1_report()) or _green_rung1_report()
        ),
    )
    monkeypatch.setattr(
        ladder,
        "run_sensitivity_scan",
        lambda _context, *, output_path, max_nfev, **_kwargs: (
            ladder._write_json(
                output_path,
                _green_rung2_report(active_params=tuple(active_params)),
            )
            or _green_rung2_report(active_params=tuple(active_params))
        ),
    )


def test_new4_ladder_pair_rung_runs_allowed_pairs_and_stops_before_blocks(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    one_param_path, diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
    )
    active = ("a", "c", "chi", "cor_angle", "theta_initial", "corto_detector", "zs", "zb")
    _install_pair_rung_common_stubs(monkeypatch, ladder, active_params=active)
    attempted: list[list[str]] = []
    seen_hashes: list[str] = []
    timing_windows: list[str | None] = []

    def _solver(*, active_names, output_path, state_hash_before, **_kwargs):
        attempted.append([str(name) for name in active_names])
        seen_hashes.append(str(state_hash_before))
        window = ladder._ACTIVE_TIMING_WINDOW.get()
        timing_windows.append(window.rung_id if window is not None else None)
        assert _kwargs["use_subprocess"] is False
        assert _kwargs["diagnostic_logging"] is False
        assert _kwargs["dirty_timeout_on_timeout"] is True
        assert _kwargs["context"]["prepared_run"] is not None
        payload = _green_pair_report(active_names)
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="pairs",
        timestamp="pairs",
        one_param_summary=one_param_path,
        one_param_diagnosis_summary=diagnosis_path,
        caked_point_reprojection_report=caked_path,
    )
    run_dir = tmp_path / "pairs"
    pair_report = json.loads((run_dir / "rung_04_pair_a_c.json").read_text())

    assert result["status"] == "ok"
    assert attempted == [
        ["a", "c"],
        ["chi", "cor_angle"],
        ["theta_initial", "cor_angle"],
        ["corto_detector", "theta_initial"],
        ["zs", "zb"],
    ]
    assert len(set(seen_hashes)) == 1
    assert result["skipped_pairs"] == []
    assert result["provider_guard_after_ok"] is True
    assert result["state_hash_unchanged"] is True
    assert result["passed_pairs"][0]["pair"] == ["a", "c"]
    assert result["stable_pairs"] == result["passed_pairs"]
    assert result["best_pair_by_rms"]["pair"] == ["a", "c"]
    assert result["recommended_next_blocks"]
    assert pair_report["candidate_param_names"] == ["a", "c"]
    assert pair_report["var_names"] == ["a", "c"]
    assert pair_report["effective_var_names_seen_by_solver"] == ["a", "c"]
    assert pair_report["base_parameter_values"] == {"a": 4.1, "c": 28.0}
    assert not list(run_dir.glob("rung_05_*.json"))
    assert not list(run_dir.glob("rung_06_*.json"))


def test_new4_ladder_pair_rung_accepts_bounded_solver_status_when_contract_passes(
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    state_hash = _hash_file(state_path)

    finalized = ladder._finalize_pair_report(
        _green_pair_report(["a", "c"], status="failed"),
        pair_name="a_c",
        pair=["a", "c"],
        state_path=state_path,
        state_hash_before=state_hash,
        timeout_seconds=120.0,
        base_parameter_values={"a": 4.1, "c": 28.0},
    )

    assert finalized["status"] == "ok"
    assert finalized["pass"] is True
    assert finalized["failure_reason"] is None
    assert finalized["pair_guard_failures"] == []


def test_new4_ladder_pair_rung_skips_params_not_allowed_by_singletons(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    one_param_path, _diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
        passed_params=("chi", "cor_angle", "c"),
        include_diagnosis=False,
    )
    _install_pair_rung_common_stubs(
        monkeypatch,
        ladder,
        active_params=("a", "c", "chi", "cor_angle", "theta_initial", "corto_detector", "zs", "zb"),
    )
    attempted: list[list[str]] = []

    def _solver(*, active_names, output_path, **_kwargs):
        attempted.append([str(name) for name in active_names])
        payload = _green_pair_report(active_names)
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="pairs",
        timestamp="pair_skip",
        one_param_summary=one_param_path,
        caked_point_reprojection_report=caked_path,
    )

    assert attempted == [["chi", "cor_angle"]]
    assert result["status"] == "ok"
    assert result["allowed_params"] == ["c", "chi", "cor_angle"]
    assert {item["pair_name"] for item in result["skipped_pairs"]} == {
        "a_c",
        "theta_initial_cor_angle",
        "corto_detector_theta_initial",
        "zs_zb",
    }
    assert {item["skip_reason"] for item in result["skipped_pairs"]} == {
        "param_not_allowed_by_singleton_evidence"
    }


def test_new4_ladder_pair_rung_nonusable_a_skips_only_a_pairs(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    one_param_path, diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
        passed_params=(
            "a",
            "c",
            "chi",
            "cor_angle",
            "theta_initial",
            "corto_detector",
            "zs",
            "zb",
        ),
    )
    diagnosis = json.loads(Path(diagnosis_path).read_text(encoding="utf-8"))
    diagnosis["status"] = "failed"
    diagnosis["diagnosis_classification"] = "hang_solver_pathology"
    ladder._write_json(Path(diagnosis_path), diagnosis)
    active = (
        "a",
        "c",
        "chi",
        "cor_angle",
        "theta_initial",
        "corto_detector",
        "zs",
        "zb",
    )
    _install_pair_rung_common_stubs(monkeypatch, ladder, active_params=active)
    attempted: list[list[str]] = []

    def _solver(*, active_names, output_path, **_kwargs):
        attempted.append([str(name) for name in active_names])
        payload = _green_pair_report(active_names)
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="pairs",
        timestamp="nonusable_a_pair_skip",
        one_param_summary=one_param_path,
        one_param_diagnosis_summary=diagnosis_path,
        caked_point_reprojection_report=caked_path,
    )

    assert result["status"] == "ok"
    assert attempted == [
        ["chi", "cor_angle"],
        ["theta_initial", "cor_angle"],
        ["corto_detector", "theta_initial"],
        ["zs", "zb"],
    ]
    assert "a" in result["disallowed_params"]
    assert "a" not in result["allowed_params"]
    assert {item["pair_name"] for item in result["skipped_pairs"]} == {"a_c"}
    assert "one_param_diagnosis:a_diagnosis_not_usable" in result[
        "local_usability_failures"
    ]
    assert result["fatal_evidence_failures"] == []


def test_new4_ladder_pair_rung_rejects_stale_one_param_summary(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    one_param_path, diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
    )
    stale = json.loads(one_param_path.read_text(encoding="utf-8"))
    stale["state_sha256_before"] = "stale"
    ladder._write_json(one_param_path, stale)
    _install_pair_rung_common_stubs(monkeypatch, ladder, active_params=("a", "c"))
    monkeypatch.setattr(
        ladder,
        "_run_solver_rung_with_timeout",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("pair solve must not run")),
    )

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="pairs",
        timestamp="stale_summary",
        one_param_summary=one_param_path,
        one_param_diagnosis_summary=diagnosis_path,
        caked_point_reprojection_report=caked_path,
    )

    assert result["status"] == "failed"
    assert result["failure_reason"] == "pair_evidence_not_current"
    assert "one_param:state_sha256_before_not_current" in result["evidence_failures"]
    assert (tmp_path / "stale_summary" / "rung_04_pair_summary.json").exists()


def test_new4_ladder_pair_rung_requires_caked_guard_for_theta_distance(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    one_param_path, _diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
        passed_params=("theta_initial", "cor_angle"),
        include_diagnosis=False,
        caked_status="fail",
    )
    _install_pair_rung_common_stubs(
        monkeypatch,
        ladder,
        active_params=("theta_initial", "cor_angle"),
    )
    monkeypatch.setattr(
        ladder,
        "_run_solver_rung_with_timeout",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("theta pair must not solve")),
    )

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="pairs",
        timestamp="caked_fail",
        one_param_summary=one_param_path,
        caked_point_reprojection_report=caked_path,
    )
    pair_report = json.loads(
        (tmp_path / "caked_fail" / "rung_04_pair_theta_initial_cor_angle.json").read_text()
    )

    assert result["status"] == "failed"
    assert result["failure_reason"] == "no_pair_solve_passed"
    assert pair_report["status"] == "failed"
    assert pair_report["failure_reason"] == "caked_point_reprojection_guard_failed"
    assert "status_not_pass" in pair_report["pair_guard_failures"]


def test_new4_ladder_pair_rung_rejects_caked_report_without_hashes(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    one_param_path, _diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
        passed_params=("theta_initial", "cor_angle"),
        include_diagnosis=False,
    )
    caked = json.loads(caked_path.read_text())
    caked.pop("state_hash_before")
    caked.pop("state_hash_after")
    ladder._write_json(caked_path, caked)
    _install_pair_rung_common_stubs(
        monkeypatch,
        ladder,
        active_params=("theta_initial", "cor_angle"),
    )
    monkeypatch.setattr(
        ladder,
        "_run_solver_rung_with_timeout",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("theta pair must not solve")),
    )

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="pairs",
        timestamp="caked_missing_hash",
        one_param_summary=one_param_path,
        caked_point_reprojection_report=caked_path,
    )
    pair_report = json.loads(
        (tmp_path / "caked_missing_hash" / "rung_04_pair_theta_initial_cor_angle.json").read_text()
    )

    assert result["status"] == "failed"
    assert pair_report["failure_reason"] == "caked_point_reprojection_guard_failed"
    assert "state_hash_before_missing" in pair_report["pair_guard_failures"]
    assert "state_hash_after_missing" in pair_report["pair_guard_failures"]


def test_new4_ladder_pair_rung_rejects_caked_report_without_background_index(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    one_param_path, _diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
        passed_params=("theta_initial", "cor_angle"),
        include_diagnosis=False,
    )
    caked = json.loads(caked_path.read_text())
    caked.pop("background_index")
    ladder._write_json(caked_path, caked)
    _install_pair_rung_common_stubs(
        monkeypatch,
        ladder,
        active_params=("theta_initial", "cor_angle"),
    )
    monkeypatch.setattr(
        ladder,
        "_run_solver_rung_with_timeout",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("theta pair must not solve")),
    )

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="pairs",
        timestamp="caked_missing_background",
        one_param_summary=one_param_path,
        caked_point_reprojection_report=caked_path,
    )
    pair_report = json.loads(
        (
            tmp_path / "caked_missing_background" / "rung_04_pair_theta_initial_cor_angle.json"
        ).read_text()
    )

    assert result["status"] == "failed"
    assert pair_report["failure_reason"] == "caked_point_reprojection_guard_failed"
    assert "background_index_missing" in pair_report["pair_guard_failures"]


def test_new4_ladder_pair_rung_rejects_caked_report_wrong_background_index(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    one_param_path, _diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
        passed_params=("theta_initial", "cor_angle"),
        include_diagnosis=False,
    )
    caked = json.loads(caked_path.read_text())
    caked["background_index"] = 1
    ladder._write_json(caked_path, caked)
    _install_pair_rung_common_stubs(
        monkeypatch,
        ladder,
        active_params=("theta_initial", "cor_angle"),
    )
    monkeypatch.setattr(
        ladder,
        "_run_solver_rung_with_timeout",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("theta pair must not solve")),
    )

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="pairs",
        timestamp="caked_wrong_background",
        one_param_summary=one_param_path,
        caked_point_reprojection_report=caked_path,
    )
    pair_report = json.loads(
        (
            tmp_path / "caked_wrong_background" / "rung_04_pair_theta_initial_cor_angle.json"
        ).read_text()
    )

    assert result["status"] == "failed"
    assert pair_report["failure_reason"] == "caked_point_reprojection_guard_failed"
    assert "background_index_mismatch" in pair_report["pair_guard_failures"]


def test_new4_ladder_pair_rung_dirty_timeout_aborts_remaining(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    one_param_path, diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
        passed_params=("c", "chi", "cor_angle"),
    )
    _install_pair_rung_common_stubs(
        monkeypatch,
        ladder,
        active_params=("a", "c", "chi", "cor_angle"),
    )
    attempted: list[list[str]] = []

    def _solver(*, active_names, output_path, **_kwargs):
        attempted.append([str(name) for name in active_names])
        payload = {
            "rung": 4,
            "rung_name": "pair_a_c",
            "status": "timeout",
            "active_params": list(active_names),
            "var_names": list(active_names),
            "candidate_param_names": list(active_names),
            "effective_var_names_seen_by_solver": list(active_names),
            "dirty_timeout_abort": True,
            "elapsed_s": 0.01,
            "timeout_s": 0.01,
        }
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="pairs",
        timestamp="dirty_pair_timeout",
        one_param_summary=one_param_path,
        one_param_diagnosis_summary=diagnosis_path,
        caked_point_reprojection_report=caked_path,
    )

    assert attempted == [["a", "c"]]
    assert result["status"] == "failed"
    assert result["timed_out_pairs"] == [{"pair_name": "a_c", "pair": ["a", "c"]}]
    assert {(item["pair_name"], item["skip_reason"]) for item in result["skipped_pairs"]} >= {
        ("chi_cor_angle", "dirty_timeout_abort"),
        ("theta_initial_cor_angle", "param_not_allowed_by_singleton_evidence"),
    }
    assert result["provider_guard_after_ok"] is False


def test_new4_ladder_pair_rung_dirty_counter_timeout_aborts_remaining(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    one_param_path, diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
        passed_params=("c", "chi", "cor_angle"),
    )
    _install_pair_rung_common_stubs(
        monkeypatch,
        ladder,
        active_params=("a", "c", "chi", "cor_angle"),
    )
    attempted: list[list[str]] = []

    def _solver(*, active_names, output_path, **_kwargs):
        attempted.append([str(name) for name in active_names])
        payload = {
            "rung": 4,
            "rung_name": "pair_a_c",
            "status": "timeout",
            "active_params": list(active_names),
            "var_names": list(active_names),
            "candidate_param_names": list(active_names),
            "effective_var_names_seen_by_solver": list(active_names),
            "dirty_timeout_abort": False,
            "fixed_source_counters_dirty_seen": True,
            "fixed_source_counters_clean_at_last_heartbeat": False,
            "fixed_source_counter_failures_at_last_heartbeat": ["fallback_row_count=1"],
            "elapsed_s": 0.01,
            "timeout_s": 0.01,
        }
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="pairs",
        timestamp="dirty_counter_pair_timeout",
        one_param_summary=one_param_path,
        one_param_diagnosis_summary=diagnosis_path,
        caked_point_reprojection_report=caked_path,
    )
    pair_report = json.loads(
        (tmp_path / "dirty_counter_pair_timeout" / "rung_04_pair_a_c.json").read_text()
    )

    assert attempted == [["a", "c"]]
    assert pair_report["failure_reason"] == "fixed_source_or_pair_integrity_lost"
    assert result["status"] == "failed"
    assert {(item["pair_name"], item["skip_reason"]) for item in result["skipped_pairs"]} >= {
        ("chi_cor_angle", "fixed_source_or_pair_integrity_lost")
    }
    assert result["provider_guard_after_ok"] is False


def test_new4_ladder_pair_rung_fallback_rows_fail_integrity(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    one_param_path, diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
        passed_params=("c",),
    )
    _install_pair_rung_common_stubs(monkeypatch, ladder, active_params=("a", "c"))

    def _solver(*, active_names, output_path, **_kwargs):
        payload = _green_pair_report(active_names, fallback_row_count=1)
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="pairs",
        timestamp="pair_fallback",
        one_param_summary=one_param_path,
        one_param_diagnosis_summary=diagnosis_path,
        caked_point_reprojection_report=caked_path,
    )
    pair_report = json.loads((tmp_path / "pair_fallback" / "rung_04_pair_a_c.json").read_text())

    assert result["status"] == "failed"
    assert result["failure_reason"] == "no_pair_solve_passed"
    assert result["any_pair_loss"] is True
    assert pair_report["failure_reason"] == "fixed_source_or_pair_integrity_lost"


@pytest.mark.parametrize(
    ("a_c_passes", "expected_extensions"),
    [
        (False, []),
        (True, [["c", "psi_z"], ["a", "psi_z"]]),
    ],
)
def test_new4_ladder_pair_rung_psi_extensions_require_a_c_pass(
    monkeypatch,
    tmp_path,
    a_c_passes,
    expected_extensions,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    active = (
        "a",
        "c",
        "psi_z",
        "chi",
        "cor_angle",
        "theta_initial",
        "corto_detector",
        "zs",
        "zb",
    )
    one_param_path, diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
        passed_params=active,
    )
    _install_pair_rung_common_stubs(monkeypatch, ladder, active_params=active)
    attempted: list[list[str]] = []

    def _solver(*, active_names, output_path, **_kwargs):
        names = [str(name) for name in active_names]
        attempted.append(names)
        payload = _green_pair_report(
            names,
            fallback_row_count=0 if names != ["a", "c"] or a_c_passes else 1,
        )
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    ladder._run_pair_stage(
        state_path=state_path,
        background_index=0,
        run_dir=tmp_path / f"psi_extensions_{a_c_passes}",
        context=_minimal_new4_ladder_context(ladder),
        sensitivity=_green_rung2_report(active_params=active),
        reports=[],
        state_hash_before=_hash_file(state_path),
        max_nfev=20,
        timeout_seconds=120.0,
        one_param_summary_path=one_param_path,
        one_param_diagnosis_summary_path=diagnosis_path,
        caked_point_reprojection_report_path=caked_path,
        include_psi_extension_pairs=True,
    )

    assert [item for item in attempted if "psi_z" in item] == expected_extensions


def test_new4_ladder_block_parser_accepts_block_aliases_and_pair_summary() -> None:
    ladder = _load_new4_ladder_module()
    parser = ladder.build_arg_parser()

    block_args = parser.parse_args(
        [
            "--state",
            "new4.json",
            "--background-index",
            "0",
            "--output-root",
            "out",
            "--max-rung",
            "block",
            "--pair-summary",
            "pairs.json",
            "--timestamp",
            "pair_run_id",
        ]
    )
    blocks_args = parser.parse_args(
        [
            "--state",
            "new4.json",
            "--background-index",
            "0",
            "--output-root",
            "out",
            "--max-rung",
            "blocks",
        ]
    )

    assert block_args.max_rung == "block"
    assert block_args.pair_summary == "pairs.json"
    assert block_args.timestamp == "pair_run_id"
    assert blocks_args.max_rung == "blocks"


def _write_block_pair_summary(
    ladder,
    tmp_path,
    state_path,
    *,
    passed_pairs,
    status="ok",
    stale_hash=False,
    run_id=None,
    timestamp=None,
    provider_report_hash=None,
    dirty_timeout_abort=False,
) -> Path:
    state_hash = _hash_file(state_path)
    summary = {
        "rung": 4,
        "rung_name": "pair_summary",
        "status": str(status),
        "state_path": str(state_path.resolve()),
        "background_index": 0,
        "state_sha256_before": "stale" if stale_hash else state_hash,
        "state_sha256_after": state_hash,
        "state_hash_unchanged": not stale_hash,
        "provider_guard_after_ok": True,
        "run_id": str(run_id or ""),
        "timestamp": str(timestamp or ""),
        "provider_report_hash": str(provider_report_hash or ""),
        "dirty_timeout_abort": bool(dirty_timeout_abort),
        "passed_pairs": [
            {"pair_name": "_".join(pair), "pair": [str(name) for name in pair]}
            for pair in passed_pairs
        ],
        "stable_pairs": [
            {"pair_name": "_".join(pair), "pair": [str(name) for name in pair]}
            for pair in passed_pairs
        ],
    }
    path = tmp_path / "rung_04_pair_summary.json"
    ladder._write_json(path, summary)
    return path


def _install_block_rung_common_stubs(monkeypatch, ladder, *, active_params) -> None:
    _install_pair_rung_common_stubs(
        monkeypatch,
        ladder,
        active_params=tuple(active_params),
    )


def _green_block_report(
    active_names,
    *,
    status="ok",
    fallback_row_count=0,
    after_rms_px=9.8,
    after_max_error_px=19.0,
    dirty_timeout_abort=False,
    effective_names=None,
) -> dict[str, object]:
    payload = _green_pair_report(
        active_names,
        status=status,
        fallback_row_count=fallback_row_count,
        after_rms_px=after_rms_px,
        after_max_error_px=after_max_error_px,
        dirty_timeout_abort=dirty_timeout_abort,
    )
    names = [str(name) for name in active_names]
    payload.update(
        {
            "rung": 5,
            "rung_name": "block_" + "_".join(names),
            "active_params": names,
            "var_names": names,
            "candidate_param_names": names,
            "effective_var_names_seen_by_solver": (
                [str(name) for name in effective_names] if effective_names is not None else names
            ),
        }
    )
    return payload


def test_new4_ladder_block_rung_runs_dependency_backed_blocks_not_a_c(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    _one_param_path, _diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
    )
    pair_summary = _write_block_pair_summary(
        ladder,
        tmp_path,
        state_path,
        passed_pairs=(
            ("a", "c"),
            ("a", "psi_z"),
            ("chi", "cor_angle"),
            ("theta_initial", "cor_angle"),
            ("corto_detector", "theta_initial"),
            ("zs", "zb"),
        ),
    )
    _install_block_rung_common_stubs(
        monkeypatch,
        ladder,
        active_params=(
            "a",
            "c",
            "psi_z",
            "chi",
            "cor_angle",
            "theta_initial",
            "corto_detector",
            "zs",
            "zb",
        ),
    )
    attempted: list[list[str]] = []
    seen_hashes: list[str] = []

    def _solver(*, active_names, output_path, state_hash_before, **_kwargs):
        attempted.append([str(name) for name in active_names])
        seen_hashes.append(str(state_hash_before))
        payload = _green_block_report(active_names)
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="blocks",
        timestamp="blocks",
        pair_summary=pair_summary,
        caked_point_reprojection_report=caked_path,
    )
    run_dir = tmp_path / "blocks"
    first_report = json.loads(
        (run_dir / "rung_05_block_corto_detector_theta_initial_cor_angle.json").read_text(
            encoding="utf-8"
        )
    )

    assert result["status"] == "ok"
    assert attempted == [
        ["corto_detector", "theta_initial", "cor_angle"],
        ["chi", "cor_angle", "theta_initial"],
        ["corto_detector", "theta_initial", "zs", "zb"],
        ["a", "c", "psi_z"],
    ]
    assert ["a", "c"] not in attempted
    assert not (run_dir / "rung_05_block_a_c.json").exists()
    assert len(set(seen_hashes)) == 1
    assert first_report["candidate_param_names"] == [
        "corto_detector",
        "theta_initial",
        "cor_angle",
    ]
    assert first_report["var_names"] == first_report["candidate_param_names"]
    assert (
        first_report["effective_var_names_seen_by_solver"] == first_report["candidate_param_names"]
    )
    assert first_report["provider_guard_after_ok"] is True
    assert first_report["caked_point_reprojection_guard_ok"] is True
    assert result["state_hash_unchanged"] is True
    assert result["passed_blocks"]
    assert result["recommended_next_full_candidate"] is not None
    assert not list(run_dir.glob("rung_06_*.json"))
    assert not list(run_dir.glob("*full*"))
    assert not list(run_dir.glob("*feature*"))
    assert not list(run_dir.glob("*baseline*"))


def test_new4_ladder_block_default_builds_same_run_evidence(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    _install_block_rung_common_stubs(
        monkeypatch,
        ladder,
        active_params=(
            "a",
            "c",
            "psi_z",
            "chi",
            "cor_angle",
            "theta_initial",
            "corto_detector",
            "zs",
            "zb",
        ),
    )
    calls: list[str] = []

    def _one_param_stage(
        *,
        run_dir,
        state_path,
        background_index,
        state_hash_before,
        reports,
        **_kwargs,
    ):
        calls.append("one_param")
        summary = {
            "status": "ok",
            "state_path": str(Path(state_path).resolve()),
            "background_index": int(background_index),
            "state_sha256_before": state_hash_before,
            "state_sha256_after": state_hash_before,
            "state_hash_unchanged": True,
            "provider_guard_after_ok": True,
            "dirty_timeout_abort": False,
            "passed_params": [
                "a",
                "c",
                "psi_z",
                "chi",
                "cor_angle",
                "theta_initial",
                "corto_detector",
                "zs",
                "zb",
            ],
        }
        ladder._write_json(Path(run_dir) / "rung_03_one_param_summary.json", summary)
        reports.append(summary)
        return summary

    def _caked_guard(*, output_root, run_id, state_path, background_index):
        calls.append("caked")
        state_hash = _hash_file(Path(state_path))
        report = {
            "status": "pass",
            "background_index": int(background_index),
            "state_hash_before": state_hash,
            "state_hash_after": state_hash,
            "point_count": 7,
            "exact_projector_available": True,
            "theta_projector_signature_changed": True,
            "distance_projector_signature_changed": True,
            "full_background_recake_call_count": 0,
            "provider_guard_before_ok": True,
            "provider_guard_after_ok": True,
            "new4_state_hash_unchanged": True,
        }
        report_path = Path(output_root) / str(run_id) / "rung_03b_caked_point_reprojection.json"
        ladder._write_json(report_path, report)
        report["report_path"] = str(report_path)
        return report

    def _pair_stage(
        *,
        run_dir,
        state_path,
        include_psi_extension_pairs,
        caked_point_reprojection_report_path,
        **_kwargs,
    ):
        calls.append("pairs")
        assert include_psi_extension_pairs is True
        assert caked_point_reprojection_report_path is not None
        return json.loads(
            _write_block_pair_summary(
                ladder,
                Path(run_dir),
                Path(state_path),
                passed_pairs=(
                    ("a", "c"),
                    ("c", "psi_z"),
                    ("chi", "cor_angle"),
                    ("theta_initial", "cor_angle"),
                    ("corto_detector", "theta_initial"),
                    ("zs", "zb"),
                ),
            ).read_text(encoding="utf-8")
        )

    attempted: list[list[str]] = []

    def _solver(*, active_names, output_path, **_kwargs):
        attempted.append([str(name) for name in active_names])
        payload = _green_block_report(active_names)
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_one_param_stage", _one_param_stage)
    monkeypatch.setattr(ladder, "_run_caked_point_reprojection_guard", _caked_guard)
    monkeypatch.setattr(ladder, "_run_pair_stage", _pair_stage)
    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="blocks",
        timestamp="default_blocks",
    )

    assert calls == ["one_param", "caked", "pairs"]
    assert result["status"] == "ok"
    assert attempted[-1] == ["a", "c", "psi_z"]
    assert (tmp_path / "default_blocks" / "rung_05_block_summary.json").exists()


def test_new4_ladder_fresh_blocks_nonusable_a_runs_unrelated_blocks(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    active = (
        "a",
        "c",
        "psi_z",
        "chi",
        "cor_angle",
        "theta_initial",
        "corto_detector",
        "zs",
        "zb",
    )
    _install_block_rung_common_stubs(monkeypatch, ladder, active_params=active)
    calls: list[str] = []

    def _one_param_stage(
        *,
        run_dir,
        state_path,
        background_index,
        state_hash_before,
        reports,
        **_kwargs,
    ):
        calls.append("one_param")
        summary = {
            "status": "ok_with_failures",
            "state_path": str(Path(state_path).resolve()),
            "background_index": int(background_index),
            "state_sha256_before": state_hash_before,
            "state_sha256_after": state_hash_before,
            "state_hash_unchanged": True,
            "provider_guard_after_ok": True,
            "dirty_timeout_abort": False,
            "passed_params": [
                "c",
                "psi_z",
                "chi",
                "cor_angle",
                "theta_initial",
                "corto_detector",
                "zs",
                "zb",
            ],
            "timed_out_params": ["a"],
        }
        ladder._write_json(Path(run_dir) / "rung_03_one_param_summary.json", summary)
        reports.append(summary)
        return summary

    def _diagnosis_variants(*, output_root, state_path, background_index, **_kwargs):
        calls.append("diagnosis")
        state_hash = _hash_file(Path(state_path))
        summary = {
            "status": "ok",
            "one_param_filter": "a",
            "state_path": str(Path(state_path).resolve()),
            "background_index": int(background_index),
            "state_sha256_before": state_hash,
            "state_sha256_after": state_hash,
            "state_hash_unchanged": True,
            "dirty_timeout_abort": False,
            "diagnosis_classification": "hang_solver_pathology",
        }
        ladder._write_json(Path(output_root) / "variant_summary.json", summary)
        return summary

    def _caked_guard(*, output_root, run_id, state_path, background_index):
        calls.append("caked")
        state_hash = _hash_file(Path(state_path))
        report = {
            "status": "pass",
            "background_index": int(background_index),
            "state_hash_before": state_hash,
            "state_hash_after": state_hash,
            "point_count": 7,
            "exact_projector_available": True,
            "theta_projector_signature_changed": True,
            "distance_projector_signature_changed": True,
            "full_background_recake_call_count": 0,
            "provider_guard_before_ok": True,
            "provider_guard_after_ok": True,
            "new4_state_hash_unchanged": True,
        }
        report_path = Path(output_root) / str(run_id) / "rung_03b_caked_point_reprojection.json"
        ladder._write_json(report_path, report)
        report["report_path"] = str(report_path)
        return report

    attempted_pairs: list[list[str]] = []
    attempted_blocks: list[list[str]] = []

    def _solver(*, rung, active_names, output_path, **_kwargs):
        names = [str(name) for name in active_names]
        if int(rung) == 4:
            attempted_pairs.append(names)
            payload = _green_pair_report(names)
        else:
            attempted_blocks.append(names)
            payload = _green_block_report(names)
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_one_param_stage", _one_param_stage)
    monkeypatch.setattr(ladder, "run_one_param_diagnosis_variants", _diagnosis_variants)
    monkeypatch.setattr(ladder, "_run_caked_point_reprojection_guard", _caked_guard)
    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="blocks",
        timestamp="fresh_nonusable_a_blocks",
    )

    assert calls == ["one_param", "diagnosis", "caked"]
    assert attempted_pairs == [
        ["chi", "cor_angle"],
        ["theta_initial", "cor_angle"],
        ["corto_detector", "theta_initial"],
        ["zs", "zb"],
    ]
    assert attempted_blocks == [
        ["corto_detector", "theta_initial", "cor_angle"],
        ["chi", "cor_angle", "theta_initial"],
        ["corto_detector", "theta_initial", "zs", "zb"],
    ]
    assert result["status"] == "ok"
    assert result["provider_guard_after_ok"] is True
    assert "a" in result["disallowed_params"]
    assert "one_param_diagnosis:a_diagnosis_not_usable" in result[
        "local_usability_failures"
    ]
    assert result["fatal_evidence_failures"] == []
    assert result["failed_blocks"] == []
    assert result["skipped_blocks"] == [
        {"block_name": "a_c_psi_z", "block": ["a", "c", "psi_z"]}
    ]


def test_new4_ladder_block_missing_dependency_skips_not_fails(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    _one_param_path, _diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
    )
    pair_summary = _write_block_pair_summary(
        ladder,
        tmp_path,
        state_path,
        passed_pairs=(("corto_detector", "theta_initial"),),
    )
    _install_block_rung_common_stubs(
        monkeypatch,
        ladder,
        active_params=("corto_detector", "theta_initial", "cor_angle"),
    )
    monkeypatch.setattr(
        ladder,
        "_run_solver_rung_with_timeout",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("block solve must not run")),
    )

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="blocks",
        timestamp="missing_dep",
        pair_summary=pair_summary,
        caked_point_reprojection_report=caked_path,
    )
    block_report = json.loads(
        (
            tmp_path / "missing_dep" / "rung_05_block_corto_detector_theta_initial_cor_angle.json"
        ).read_text(encoding="utf-8")
    )

    assert block_report["status"] == "skipped"
    assert block_report["skip_reason"] == "missing_pair_evidence"
    assert block_report["candidate_param_names"] == [
        "corto_detector",
        "theta_initial",
        "cor_angle",
    ]
    assert block_report["var_names"] == block_report["candidate_param_names"]
    assert (
        block_report["effective_var_names_seen_by_solver"] == block_report["candidate_param_names"]
    )
    assert block_report["rung_id"] == "5"
    assert np.isfinite(float(block_report["elapsed_s"]))
    assert result["failed_blocks"] == []
    assert result["skipped_blocks"]
    assert result["failure_reason"] == "no_dependency_backed_blocks"


def test_new4_ladder_block_pair_summary_run_id_can_match_timestamp(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    _one_param_path, _diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
    )
    pair_summary = _write_block_pair_summary(
        ladder,
        tmp_path,
        state_path,
        passed_pairs=(("corto_detector", "theta_initial"), ("theta_initial", "cor_angle")),
        run_id="pair_run",
        timestamp="pair_run",
    )
    _install_block_rung_common_stubs(
        monkeypatch,
        ladder,
        active_params=("corto_detector", "theta_initial", "cor_angle"),
    )
    attempted: list[list[str]] = []
    timing_windows: list[str | None] = []

    def _solver(*, active_names, output_path, **_kwargs):
        attempted.append([str(name) for name in active_names])
        window = ladder._ACTIVE_TIMING_WINDOW.get()
        timing_windows.append(window.rung_id if window is not None else None)
        payload = _green_block_report(active_names)
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="blocks",
        timestamp="pair_run",
        pair_summary=pair_summary,
        caked_point_reprojection_report=caked_path,
    )

    assert result["status"] == "ok"
    assert attempted == [["corto_detector", "theta_initial", "cor_angle"]]
    assert result["evidence_failures"] == []


def test_new4_ladder_block_pair_summary_run_id_mismatch_blocks_solve(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    _one_param_path, _diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
    )
    pair_summary = _write_block_pair_summary(
        ladder,
        tmp_path,
        state_path,
        passed_pairs=(("corto_detector", "theta_initial"), ("theta_initial", "cor_angle")),
        run_id="old_pair_run",
        timestamp="old_pair_run",
    )
    _install_block_rung_common_stubs(
        monkeypatch,
        ladder,
        active_params=("corto_detector", "theta_initial", "cor_angle"),
    )
    monkeypatch.setattr(
        ladder,
        "_run_solver_rung_with_timeout",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("mismatch must not solve")),
    )

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="blocks",
        timestamp="new_block_run",
        pair_summary=pair_summary,
        caked_point_reprojection_report=caked_path,
    )

    assert result["status"] == "failed"
    assert result["failure_reason"] == "pair_evidence_not_current"
    assert "pair:run_id_not_current" in result["evidence_failures"]
    assert "pair:timestamp_not_current" in result["evidence_failures"]


def test_new4_ladder_block_pair_summary_provider_hash_mismatch_blocks_solve(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    _one_param_path, _diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
    )
    pair_summary = _write_block_pair_summary(
        ladder,
        tmp_path,
        state_path,
        passed_pairs=(("corto_detector", "theta_initial"), ("theta_initial", "cor_angle")),
        provider_report_hash="stale_provider_hash",
    )
    _install_block_rung_common_stubs(
        monkeypatch,
        ladder,
        active_params=("corto_detector", "theta_initial", "cor_angle"),
    )
    monkeypatch.setattr(
        ladder,
        "_run_solver_rung_with_timeout",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("mismatch must not solve")),
    )

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="blocks",
        timestamp="provider_hash_mismatch",
        pair_summary=pair_summary,
        caked_point_reprojection_report=caked_path,
    )

    assert result["status"] == "failed"
    assert result["failure_reason"] == "pair_evidence_not_current"
    assert "pair:provider_report_hash_not_current" in result["evidence_failures"]
    assert "pair:provider_report_hash_not_current" in result["fatal_evidence_failures"]


def test_new4_ladder_block_malformed_pair_summary_blocks_solve(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    _one_param_path, _diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
    )
    pair_summary = tmp_path / "malformed_pair_summary.json"
    pair_summary.write_text("{", encoding="utf-8")
    _install_block_rung_common_stubs(
        monkeypatch,
        ladder,
        active_params=("corto_detector", "theta_initial", "cor_angle"),
    )
    monkeypatch.setattr(
        ladder,
        "_run_solver_rung_with_timeout",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("malformed must not solve")),
    )

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="blocks",
        timestamp="malformed_pair_summary",
        pair_summary=pair_summary,
        caked_point_reprojection_report=caked_path,
    )

    assert result["status"] == "failed"
    assert result["failure_reason"] == "pair_evidence_not_current"
    assert "pair:missing_pair_summary_unreadable" in result["evidence_failures"]
    assert not [
        path
        for path in (tmp_path / "malformed_pair_summary").glob("rung_05_block_*.json")
        if path.name != "rung_05_block_summary.json"
    ]


def test_new4_ladder_block_dirty_pair_summary_blocks_solve(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    _one_param_path, _diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
    )
    pair_summary = _write_block_pair_summary(
        ladder,
        tmp_path,
        state_path,
        passed_pairs=(("corto_detector", "theta_initial"), ("theta_initial", "cor_angle")),
        dirty_timeout_abort=True,
    )
    _install_block_rung_common_stubs(
        monkeypatch,
        ladder,
        active_params=("corto_detector", "theta_initial", "cor_angle"),
    )
    monkeypatch.setattr(
        ladder,
        "_run_solver_rung_with_timeout",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("dirty evidence must not solve")),
    )

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="blocks",
        timestamp="dirty_pair_summary",
        pair_summary=pair_summary,
        caked_point_reprojection_report=caked_path,
    )

    assert result["status"] == "failed"
    assert result["failure_reason"] == "pair_evidence_not_current"
    assert "pair:dirty_timeout_abort" in result["evidence_failures"]
    assert "pair:dirty_timeout_abort" in result["fatal_evidence_failures"]


def test_new4_ladder_block_rejects_stale_pair_summary_before_solve(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    _one_param_path, _diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
    )
    pair_summary = _write_block_pair_summary(
        ladder,
        tmp_path,
        state_path,
        passed_pairs=(("corto_detector", "theta_initial"), ("theta_initial", "cor_angle")),
        stale_hash=True,
    )
    _install_block_rung_common_stubs(
        monkeypatch,
        ladder,
        active_params=("corto_detector", "theta_initial", "cor_angle"),
    )
    monkeypatch.setattr(
        ladder,
        "_run_solver_rung_with_timeout",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("stale evidence must not solve")),
    )

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="blocks",
        timestamp="stale_pair_summary",
        pair_summary=pair_summary,
        caked_point_reprojection_report=caked_path,
    )

    assert result["status"] == "failed"
    assert result["failure_reason"] == "pair_evidence_not_current"
    assert "pair:state_sha256_before_not_current" in result["evidence_failures"]
    assert not [
        path
        for path in (tmp_path / "stale_pair_summary").glob("rung_05_block_*.json")
        if path.name != "rung_05_block_summary.json"
    ]


def test_new4_ladder_block_stale_caked_report_blocks_theta_before_solve(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    _one_param_path, _diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
        caked_status="fail",
    )
    pair_summary = _write_block_pair_summary(
        ladder,
        tmp_path,
        state_path,
        passed_pairs=(("corto_detector", "theta_initial"), ("theta_initial", "cor_angle")),
    )
    _install_block_rung_common_stubs(
        monkeypatch,
        ladder,
        active_params=("corto_detector", "theta_initial", "cor_angle"),
    )
    monkeypatch.setattr(
        ladder,
        "_run_solver_rung_with_timeout",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("caked failure must not solve")),
    )

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="blocks",
        timestamp="stale_caked_block",
        pair_summary=pair_summary,
        caked_point_reprojection_report=caked_path,
    )
    block_report = json.loads(
        (
            tmp_path
            / "stale_caked_block"
            / "rung_05_block_corto_detector_theta_initial_cor_angle.json"
        ).read_text(encoding="utf-8")
    )

    assert block_report["status"] == "failed"
    assert block_report["failure_reason"] == "caked_point_reprojection_guard_failed"
    assert block_report["caked_point_reprojection_guard_ok"] is False
    assert "status_not_pass" in block_report["block_guard_failures"]
    assert result["failed_blocks"] == [
        {
            "block_name": "corto_detector_theta_initial_cor_angle",
            "block": ["corto_detector", "theta_initial", "cor_angle"],
        }
    ]


def test_new4_ladder_block_effective_var_names_must_match_request(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    _one_param_path, _diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
    )
    pair_summary = _write_block_pair_summary(
        ladder,
        tmp_path,
        state_path,
        passed_pairs=(("a", "c"), ("a", "psi_z")),
    )
    _install_block_rung_common_stubs(
        monkeypatch,
        ladder,
        active_params=("a", "c", "psi_z"),
    )

    def _solver(*, active_names, output_path, **_kwargs):
        payload = _green_block_report(active_names, effective_names=["a", "c"])
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="blocks",
        timestamp="bad_effective_names",
        pair_summary=pair_summary,
        caked_point_reprojection_report=caked_path,
    )
    block_report = json.loads(
        (tmp_path / "bad_effective_names" / "rung_05_block_a_c_psi_z.json").read_text(
            encoding="utf-8"
        )
    )

    assert block_report["status"] == "failed"
    assert block_report["failure_reason"] == "solve_flag_guard_failed"
    assert "effective_var_names_seen_by_solver_not_block" in block_report["block_guard_failures"]
    assert result["failure_reason"] == "no_block_solve_passed"


def test_new4_ladder_block_fallback_rows_fail_integrity(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    _one_param_path, _diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
    )
    pair_summary = _write_block_pair_summary(
        ladder,
        tmp_path,
        state_path,
        passed_pairs=(("a", "c"), ("a", "psi_z")),
    )
    _install_block_rung_common_stubs(
        monkeypatch,
        ladder,
        active_params=("a", "c", "psi_z"),
    )

    def _solver(*, active_names, output_path, **_kwargs):
        payload = _green_block_report(active_names, fallback_row_count=1)
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="blocks",
        timestamp="block_fallback",
        pair_summary=pair_summary,
        caked_point_reprojection_report=caked_path,
    )
    block_report = json.loads(
        (tmp_path / "block_fallback" / "rung_05_block_a_c_psi_z.json").read_text(encoding="utf-8")
    )

    assert block_report["status"] == "failed"
    assert block_report["failure_reason"] == "fixed_source_or_pair_integrity_lost"
    assert result["failure_reason"] == "no_block_solve_passed"


def test_new4_ladder_block_timeout_writes_partial_json(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    _one_param_path, _diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
    )
    pair_summary = _write_block_pair_summary(
        ladder,
        tmp_path,
        state_path,
        passed_pairs=(("a", "c"), ("a", "psi_z")),
    )
    _install_block_rung_common_stubs(
        monkeypatch,
        ladder,
        active_params=("a", "c", "psi_z"),
    )

    def _solver(*, active_names, output_path, **_kwargs):
        payload = {
            "rung": 5,
            "rung_name": "block_a_c_psi_z",
            "status": "timeout",
            "active_params": list(active_names),
            "var_names": list(active_names),
            "candidate_param_names": list(active_names),
            "effective_var_names_seen_by_solver": list(active_names),
            "dirty_timeout_abort": False,
            "last_nfev": 2,
            "last_residual_norm": 12.0,
            "last_rms_px": 9.9,
            "last_max_error_px": 19.9,
            "elapsed_s": 0.01,
            "timeout_s": 0.01,
        }
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="blocks",
        timestamp="block_timeout",
        pair_summary=pair_summary,
        caked_point_reprojection_report=caked_path,
    )
    block_report = json.loads(
        (tmp_path / "block_timeout" / "rung_05_block_a_c_psi_z.json").read_text(encoding="utf-8")
    )

    assert block_report["status"] == "timeout"
    assert block_report["failure_reason"] == "timeout"
    assert block_report["last_nfev"] == 2
    assert result["timed_out_blocks"] == [{"block_name": "a_c_psi_z", "block": ["a", "c", "psi_z"]}]
    assert result["failure_reason"] == "no_block_solve_passed"


def test_new4_ladder_block_dirty_timeout_aborts_remaining(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    _one_param_path, _diagnosis_path, caked_path = _write_pair_evidence(
        ladder,
        tmp_path,
        state_path,
    )
    pair_summary = _write_block_pair_summary(
        ladder,
        tmp_path,
        state_path,
        passed_pairs=(
            ("a", "c"),
            ("a", "psi_z"),
            ("chi", "cor_angle"),
            ("theta_initial", "cor_angle"),
            ("corto_detector", "theta_initial"),
            ("zs", "zb"),
        ),
    )
    _install_block_rung_common_stubs(
        monkeypatch,
        ladder,
        active_params=(
            "a",
            "c",
            "psi_z",
            "chi",
            "cor_angle",
            "theta_initial",
            "corto_detector",
            "zs",
            "zb",
        ),
    )
    attempted: list[list[str]] = []

    def _solver(*, active_names, output_path, **_kwargs):
        attempted.append([str(name) for name in active_names])
        payload = {
            "rung": 5,
            "rung_name": "block_corto_detector_theta_initial_cor_angle",
            "status": "timeout",
            "active_params": list(active_names),
            "var_names": list(active_names),
            "candidate_param_names": list(active_names),
            "effective_var_names_seen_by_solver": list(active_names),
            "dirty_timeout_abort": True,
            "elapsed_s": 0.01,
            "timeout_s": 0.01,
        }
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="blocks",
        timestamp="dirty_block_timeout",
        pair_summary=pair_summary,
        caked_point_reprojection_report=caked_path,
    )

    assert attempted == [["corto_detector", "theta_initial", "cor_angle"]]
    assert result["dirty_timeout_abort"] is True
    assert result["timed_out_blocks"] == [
        {
            "block_name": "corto_detector_theta_initial_cor_angle",
            "block": ["corto_detector", "theta_initial", "cor_angle"],
        }
    ]
    assert {(item["block_name"], tuple(item["block"])) for item in result["skipped_blocks"]} >= {
        ("chi_cor_angle_theta_initial", ("chi", "cor_angle", "theta_initial")),
        ("a_c_psi_z", ("a", "c", "psi_z")),
    }


def _run_one_param_with_sensitivity_payload(
    monkeypatch,
    tmp_path,
    payload: dict[str, object],
    *,
    timestamp: str = "dirty_sensitivity",
) -> tuple[dict[str, object], Path]:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    _install_green_provider_guard(monkeypatch, ladder)
    monkeypatch.setattr(
        ladder,
        "_capture_solver_context",
        lambda *_args, **_kwargs: _minimal_new4_ladder_context(ladder),
    )
    monkeypatch.setattr(
        ladder,
        "run_objective_dry_run",
        lambda _context, *, output_path, max_nfev: (
            ladder._write_json(output_path, _green_rung1_report()) or _green_rung1_report()
        ),
    )

    def _sensitivity(_context, *, output_path, max_nfev, **_kwargs):
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "run_sensitivity_scan", _sensitivity)
    monkeypatch.setattr(
        ladder,
        "_run_solver_rung_with_timeout",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("solve must not run")),
    )

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="one-param",
        timestamp=timestamp,
    )
    return result, tmp_path / timestamp


@pytest.mark.parametrize(
    ("field", "dirty_value"),
    [
        ("fallback_row_count", 1),
        ("fixed_source_resolution_fallback_count", 1),
        ("missing_fixed_source_count", 1),
        ("fixed_source_resolved_count", 6),
        ("provider_to_optimizer_identity_match", False),
        ("provider_to_optimizer_point_match", False),
    ],
)
def test_new4_ladder_one_param_blocks_dirty_rung2_top_level_contract(
    monkeypatch,
    tmp_path,
    field,
    dirty_value,
) -> None:
    payload = _green_rung2_report(active_params=("center_x",))
    payload[field] = dirty_value

    result, run_dir = _run_one_param_with_sensitivity_payload(
        monkeypatch,
        tmp_path,
        payload,
        timestamp=f"dirty_{field}",
    )

    assert result["status"] == "failed"
    assert result["failure_reason"] == "sensitivity_not_green"
    assert field in " ".join(result["rung_2_failures"])
    assert not [
        path
        for path in run_dir.glob("rung_03_one_param_*.json")
        if path.name != "rung_03_one_param_summary.json"
    ]


def test_new4_ladder_one_param_blocks_dirty_rung2_active_eval_contract(
    monkeypatch,
    tmp_path,
) -> None:
    payload = _green_rung2_report(active_params=("center_x",))
    payload["params"][0]["fallback_row_count"] = 1

    result, run_dir = _run_one_param_with_sensitivity_payload(
        monkeypatch,
        tmp_path,
        payload,
        timestamp="dirty_active_eval",
    )

    assert result["status"] == "failed"
    assert result["failure_reason"] == "sensitivity_not_green"
    assert "center_x_fallback_row_count_1_expected_0" in result["rung_2_failures"]
    assert not [
        path
        for path in run_dir.glob("rung_03_one_param_*.json")
        if path.name != "rung_03_one_param_summary.json"
    ]


@pytest.mark.parametrize(
    "field",
    [
        "fixed_source_resolved_count",
        "fallback_row_count",
        "provider_to_optimizer_point_match",
    ],
)
def test_new4_ladder_one_param_blocks_missing_rung2_active_eval_contract(
    monkeypatch,
    tmp_path,
    field,
) -> None:
    payload = _green_rung2_report(active_params=("center_x",))
    payload["params"][0].pop(field)

    result, run_dir = _run_one_param_with_sensitivity_payload(
        monkeypatch,
        tmp_path,
        payload,
        timestamp=f"missing_active_eval_{field}",
    )

    assert result["status"] == "failed"
    assert result["failure_reason"] == "sensitivity_not_green"
    assert field in " ".join(result["rung_2_failures"])
    assert not [
        path
        for path in run_dir.glob("rung_03_one_param_*.json")
        if path.name != "rung_03_one_param_summary.json"
    ]


@pytest.mark.parametrize("scope", ["top_level", "active_eval"])
def test_new4_ladder_one_param_blocks_bool_rung2_counter_values(
    monkeypatch,
    tmp_path,
    scope,
) -> None:
    payload = _green_rung2_report(active_params=("center_x",))
    if scope == "top_level":
        payload["fallback_row_count"] = False
    else:
        payload["params"][0]["fallback_row_count"] = False

    result, run_dir = _run_one_param_with_sensitivity_payload(
        monkeypatch,
        tmp_path,
        payload,
        timestamp=f"bool_counter_{scope}",
    )

    assert result["status"] == "failed"
    assert result["failure_reason"] == "sensitivity_not_green"
    assert "fallback_row_count_-999999_expected_0" in " ".join(result["rung_2_failures"])
    assert not [
        path
        for path in run_dir.glob("rung_03_one_param_*.json")
        if path.name != "rung_03_one_param_summary.json"
    ]


def test_new4_ladder_one_param_does_not_use_stale_sensitivity_by_default(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    stale = tmp_path / "stale_sensitivity.json"
    stale.write_text(
        json.dumps(_green_rung2_report(active_params=("center_y",))),
        encoding="utf-8",
    )
    attempted: list[str] = []
    _install_green_provider_guard(monkeypatch, ladder)
    monkeypatch.setattr(
        ladder,
        "_capture_solver_context",
        lambda *_args, **_kwargs: _minimal_new4_ladder_context(ladder),
    )
    monkeypatch.setattr(
        ladder,
        "run_objective_dry_run",
        lambda _context, *, output_path, max_nfev: (
            ladder._write_json(output_path, _green_rung1_report()) or _green_rung1_report()
        ),
    )

    def _fresh_sensitivity(_context, *, output_path, max_nfev, **_kwargs):
        payload = _green_rung2_report(active_params=("center_x",))
        ladder._write_json(output_path, payload)
        return payload

    def _solver(*, active_names, output_path, **_kwargs):
        attempted.extend(active_names)
        payload = _green_one_param_report(active_names)
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "run_sensitivity_scan", _fresh_sensitivity)
    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="one-param",
        timestamp="fresh_one_param",
    )

    assert attempted == ["center_x"]
    assert result["active_params_from_sensitivity"] == ["center_x"]
    assert "center_y" not in result["attempted_params"]
    assert stale.exists()


def test_new4_ladder_one_param_uses_only_active_sensitivity_params(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    attempted: list[str] = []
    _install_green_provider_guard(monkeypatch, ladder)
    monkeypatch.setattr(
        ladder,
        "_capture_solver_context",
        lambda *_args, **_kwargs: _minimal_new4_ladder_context(ladder),
    )
    monkeypatch.setattr(
        ladder,
        "run_objective_dry_run",
        lambda _context, *, output_path, max_nfev: (
            ladder._write_json(output_path, _green_rung1_report()) or _green_rung1_report()
        ),
    )

    def _sensitivity(_context, *, output_path, max_nfev, **_kwargs):
        payload = _green_rung2_report(
            active_params=("center_x", "center_y", "gamma"),
            near_zero_params=("center_y",),
            non_finite_params=("gamma",),
        )
        ladder._write_json(output_path, payload)
        return payload

    def _solver(*, active_names, output_path, **_kwargs):
        attempted.extend(active_names)
        payload = _green_one_param_report(active_names)
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "run_sensitivity_scan", _sensitivity)
    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="one-param",
        timestamp="active_only",
    )

    assert attempted == ["center_x"]
    assert result["attempted_params"] == ["center_x"]
    assert sorted(result["skipped_params"]) == ["center_y", "gamma"]


def test_new4_ladder_one_param_filter_only_attempts_selected_param(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    attempted: list[str] = []
    _install_green_provider_guard(monkeypatch, ladder)
    monkeypatch.setattr(
        ladder,
        "_capture_solver_context",
        lambda *_args, **_kwargs: _minimal_new4_ladder_context(ladder),
    )
    monkeypatch.setattr(
        ladder,
        "run_objective_dry_run",
        lambda _context, *, output_path, max_nfev: (
            ladder._write_json(output_path, _green_rung1_report()) or _green_rung1_report()
        ),
    )
    monkeypatch.setattr(
        ladder,
        "run_sensitivity_scan",
        lambda _context, *, output_path, max_nfev, **_kwargs: (
            ladder._write_json(
                output_path,
                _green_rung2_report(active_params=("chi", "a", "c")),
            )
            or _green_rung2_report(active_params=("chi", "a", "c"))
        ),
    )

    def _solver(*, active_names, output_path, **_kwargs):
        attempted.extend(active_names)
        payload = _green_one_param_report(active_names)
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="one-param",
        one_param_filter="a",
        timestamp="filter_a",
    )

    assert attempted == ["a"]
    assert result["attempted_params"] == ["a"]
    assert result["filtered_params"] == ["chi", "c"]
    assert result["failed_params"] == []
    run_dir = tmp_path / "filter_a"
    assert (run_dir / "rung_03_one_param_a.json").exists()
    assert not list(run_dir.glob("rung_04_*.json"))
    assert not list(run_dir.glob("rung_05_*.json"))
    assert not list(run_dir.glob("rung_06_*.json"))


def test_new4_ladder_one_param_filter_inactive_fails_before_solve(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    _install_green_provider_guard(monkeypatch, ladder)
    monkeypatch.setattr(
        ladder,
        "_capture_solver_context",
        lambda *_args, **_kwargs: _minimal_new4_ladder_context(ladder),
    )
    monkeypatch.setattr(
        ladder,
        "run_objective_dry_run",
        lambda _context, *, output_path, max_nfev: (
            ladder._write_json(output_path, _green_rung1_report()) or _green_rung1_report()
        ),
    )
    monkeypatch.setattr(
        ladder,
        "run_sensitivity_scan",
        lambda _context, *, output_path, max_nfev, **_kwargs: (
            ladder._write_json(output_path, _green_rung2_report(active_params=("chi", "c")))
            or _green_rung2_report(active_params=("chi", "c"))
        ),
    )
    monkeypatch.setattr(
        ladder,
        "_run_solver_rung_with_timeout",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("solve must not run")),
    )

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="one-param",
        one_param_filter="a",
        timestamp="filter_inactive",
    )

    assert result["status"] == "failed"
    assert result["failure_reason"] == "filtered_param_not_active"
    assert result["attempted_params"] == []
    assert result["filtered_params"] == ["chi", "c"]
    run_dir = tmp_path / "filter_inactive"
    assert not [
        path
        for path in run_dir.glob("rung_03_one_param_*.json")
        if path.name != "rung_03_one_param_summary.json"
    ]


def test_new4_ladder_one_param_sets_candidate_param_names_singleton(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    seen: list[tuple[list[str], list[str]]] = []
    _install_green_provider_guard(monkeypatch, ladder)
    monkeypatch.setattr(
        ladder,
        "_capture_solver_context",
        lambda *_args, **_kwargs: _minimal_new4_ladder_context(ladder),
    )
    monkeypatch.setattr(
        ladder,
        "run_objective_dry_run",
        lambda _context, *, output_path, max_nfev: (
            ladder._write_json(output_path, _green_rung1_report()) or _green_rung1_report()
        ),
    )
    monkeypatch.setattr(
        ladder,
        "run_sensitivity_scan",
        lambda _context, *, output_path, max_nfev, **_kwargs: (
            ladder._write_json(
                output_path,
                _green_rung2_report(active_params=("center_x", "center_y")),
            )
            or _green_rung2_report(active_params=("center_x", "center_y"))
        ),
    )

    def _solver(*, active_names, output_path, **_kwargs):
        payload = _green_one_param_report(active_names)
        seen.append((list(payload["var_names"]), list(payload["candidate_param_names"])))
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="one-param",
        timestamp="singleton",
    )

    assert seen == [(["center_x"], ["center_x"]), (["center_y"], ["center_y"])]


def test_new4_ladder_one_param_no_active_params_fails_before_solve(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    _install_green_provider_guard(monkeypatch, ladder)
    monkeypatch.setattr(
        ladder,
        "_capture_solver_context",
        lambda *_args, **_kwargs: _minimal_new4_ladder_context(ladder),
    )
    monkeypatch.setattr(
        ladder,
        "run_objective_dry_run",
        lambda _context, *, output_path, max_nfev: (
            ladder._write_json(output_path, _green_rung1_report()) or _green_rung1_report()
        ),
    )
    monkeypatch.setattr(
        ladder,
        "run_sensitivity_scan",
        lambda _context, *, output_path, max_nfev, **_kwargs: (
            ladder._write_json(output_path, _green_rung2_report(active_params=()))
            or _green_rung2_report(active_params=())
        ),
    )
    monkeypatch.setattr(
        ladder,
        "_run_solver_rung_with_timeout",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("solve must not run")),
    )

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="one-param",
        timestamp="no_active",
    )

    assert result["status"] == "failed"
    assert result["failure_reason"] == "no_active_params"
    assert result["attempted_params"] == []


def test_new4_ladder_one_param_does_not_run_when_sensitivity_not_green(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    _install_green_provider_guard(monkeypatch, ladder)
    monkeypatch.setattr(
        ladder,
        "_capture_solver_context",
        lambda *_args, **_kwargs: _minimal_new4_ladder_context(ladder),
    )
    monkeypatch.setattr(
        ladder,
        "run_objective_dry_run",
        lambda _context, *, output_path, max_nfev: (
            ladder._write_json(output_path, _green_rung1_report()) or _green_rung1_report()
        ),
    )

    def _bad_sensitivity(_context, *, output_path, max_nfev, **_kwargs):
        payload = _green_rung2_report(active_params=("center_x",))
        payload.update({"status": "fail", "pass": False})
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "run_sensitivity_scan", _bad_sensitivity)
    monkeypatch.setattr(
        ladder,
        "_run_solver_rung_with_timeout",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("solve must not run")),
    )

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="one-param",
        timestamp="sensitivity_fail",
    )
    run_dir = tmp_path / "sensitivity_fail"

    assert result["status"] == "failed"
    assert result["failure_reason"] == "sensitivity_not_green"
    assert not [
        path
        for path in run_dir.glob("rung_03_one_param_*.json")
        if path.name != "rung_03_one_param_summary.json"
    ]


def test_new4_ladder_one_param_fails_on_fallback_rows(monkeypatch, tmp_path) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    _install_green_provider_guard(monkeypatch, ladder)
    monkeypatch.setattr(
        ladder,
        "_capture_solver_context",
        lambda *_args, **_kwargs: _minimal_new4_ladder_context(ladder),
    )
    monkeypatch.setattr(
        ladder,
        "run_objective_dry_run",
        lambda _context, *, output_path, max_nfev: (
            ladder._write_json(output_path, _green_rung1_report()) or _green_rung1_report()
        ),
    )
    monkeypatch.setattr(
        ladder,
        "run_sensitivity_scan",
        lambda _context, *, output_path, max_nfev, **_kwargs: (
            ladder._write_json(output_path, _green_rung2_report(active_params=("center_x",)))
            or _green_rung2_report(active_params=("center_x",))
        ),
    )

    def _solver(*, active_names, output_path, **_kwargs):
        payload = _green_one_param_report(active_names, fallback_row_count=1)
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="one-param",
        timestamp="fallback_rows",
    )
    report = json.loads(
        (tmp_path / "fallback_rows" / "rung_03_one_param_center_x.json").read_text(encoding="utf-8")
    )

    assert report["status"] == "failed"
    assert report["failure_reason"] == "fixed_source_or_pair_integrity_lost"
    assert result["failure_reason"] == "no_one_param_solve_passed"
    assert result["any_pair_loss"] is True
    assert result["all_passing_fixed_source_counters_clean"] is False
    assert not list((tmp_path / "fallback_rows").glob("rung_04_*.json"))


def test_new4_ladder_one_param_clean_top_level_counters_without_point_summary_ok(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    _install_green_provider_guard(monkeypatch, ladder)
    monkeypatch.setattr(
        ladder,
        "_capture_solver_context",
        lambda *_args, **_kwargs: _minimal_new4_ladder_context(ladder),
    )
    monkeypatch.setattr(
        ladder,
        "run_objective_dry_run",
        lambda _context, *, output_path, max_nfev: (
            ladder._write_json(output_path, _green_rung1_report()) or _green_rung1_report()
        ),
    )
    monkeypatch.setattr(
        ladder,
        "run_sensitivity_scan",
        lambda _context, *, output_path, max_nfev, **_kwargs: (
            ladder._write_json(output_path, _green_rung2_report(active_params=("center_x",)))
            or _green_rung2_report(active_params=("center_x",))
        ),
    )

    def _solver(*, active_names, output_path, **_kwargs):
        payload = _green_one_param_report(active_names)
        payload["point_match_summary"] = {}
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="one-param",
        timestamp="clean_no_point_summary",
    )
    report = json.loads(
        (tmp_path / "clean_no_point_summary" / "rung_03_one_param_center_x.json").read_text(
            encoding="utf-8"
        )
    )

    assert result["status"] == "ok"
    assert result["any_pair_loss"] is False
    assert report["fixed_source_counters_clean_at_last_heartbeat"] is True
    assert report["fixed_source_counter_failures_at_last_heartbeat"] == []


def test_new4_ladder_one_param_all_active_fail_summary(monkeypatch, tmp_path) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    _install_green_provider_guard(monkeypatch, ladder)
    monkeypatch.setattr(
        ladder,
        "_capture_solver_context",
        lambda *_args, **_kwargs: _minimal_new4_ladder_context(ladder),
    )
    monkeypatch.setattr(
        ladder,
        "run_objective_dry_run",
        lambda _context, *, output_path, max_nfev: (
            ladder._write_json(output_path, _green_rung1_report()) or _green_rung1_report()
        ),
    )
    monkeypatch.setattr(
        ladder,
        "run_sensitivity_scan",
        lambda _context, *, output_path, max_nfev, **_kwargs: (
            ladder._write_json(
                output_path,
                _green_rung2_report(active_params=("center_x", "center_y")),
            )
            or _green_rung2_report(active_params=("center_x", "center_y"))
        ),
    )

    def _solver(*, active_names, output_path, **_kwargs):
        payload = _green_one_param_report(active_names)
        payload["after_rms_px"] = 11.0
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="one-param",
        timestamp="all_fail",
    )

    assert result["status"] == "failed"
    assert result["failure_reason"] == "no_one_param_solve_passed"
    assert result["failed_params"] == ["center_x", "center_y"]


def test_new4_ladder_one_param_partial_success_exposes_failures(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    _install_green_provider_guard(monkeypatch, ladder)
    monkeypatch.setattr(
        ladder,
        "_capture_solver_context",
        lambda *_args, **_kwargs: _minimal_new4_ladder_context(ladder),
    )
    monkeypatch.setattr(
        ladder,
        "run_objective_dry_run",
        lambda _context, *, output_path, max_nfev: (
            ladder._write_json(output_path, _green_rung1_report()) or _green_rung1_report()
        ),
    )
    monkeypatch.setattr(
        ladder,
        "run_sensitivity_scan",
        lambda _context, *, output_path, max_nfev, **_kwargs: (
            ladder._write_json(
                output_path,
                _green_rung2_report(active_params=("center_x", "center_y", "gamma")),
            )
            or _green_rung2_report(active_params=("center_x", "center_y", "gamma"))
        ),
    )

    def _solver(*, active_names, output_path, **_kwargs):
        name = active_names[0]
        if name == "center_y":
            payload = _green_one_param_report(active_names)
            payload["after_max_error_px"] = 25.0
        elif name == "gamma":
            payload = {
                "rung": 3,
                "rung_name": "one_param_gamma",
                "status": "timeout",
                "active_params": ["gamma"],
                "var_names": ["gamma"],
                "candidate_param_names": ["gamma"],
                "elapsed_s": 0.01,
                "timeout_s": 0.01,
                "dirty_timeout_abort": False,
            }
        else:
            payload = _green_one_param_report(active_names)
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="one-param",
        timestamp="partial",
    )

    assert result["status"] == "ok_with_failures"
    assert result["passed_params"] == ["center_x"]
    assert result["failed_params"] == ["center_y"]
    assert result["timed_out_params"] == ["gamma"]
    assert result["any_timeout"] is True
    assert result["all_passing_fixed_source_counters_clean"] is True


def test_new4_ladder_one_param_clean_timeout_continues_to_next_param(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    attempted: list[str] = []
    _install_green_provider_guard(monkeypatch, ladder)
    monkeypatch.setattr(
        ladder,
        "_capture_solver_context",
        lambda *_args, **_kwargs: _minimal_new4_ladder_context(ladder),
    )
    monkeypatch.setattr(
        ladder,
        "run_objective_dry_run",
        lambda _context, *, output_path, max_nfev: (
            ladder._write_json(output_path, _green_rung1_report()) or _green_rung1_report()
        ),
    )
    monkeypatch.setattr(
        ladder,
        "run_sensitivity_scan",
        lambda _context, *, output_path, max_nfev, **_kwargs: (
            ladder._write_json(
                output_path,
                _green_rung2_report(active_params=("center_x", "center_y")),
            )
            or _green_rung2_report(active_params=("center_x", "center_y"))
        ),
    )

    def _solver(*, active_names, output_path, **_kwargs):
        name = active_names[0]
        attempted.append(name)
        if name == "center_x":
            payload = {
                "rung": 3,
                "rung_name": "one_param_center_x",
                "status": "timeout",
                "active_params": ["center_x"],
                "var_names": ["center_x"],
                "candidate_param_names": ["center_x"],
                "elapsed_s": 0.01,
                "timeout_s": 0.01,
                "dirty_timeout_abort": False,
            }
        else:
            payload = _green_one_param_report(active_names)
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="one-param",
        timestamp="clean_timeout_continue",
    )

    assert attempted == ["center_x", "center_y"]
    assert result["status"] == "ok_with_failures"
    assert result["timed_out_params"] == ["center_x"]
    assert result["passed_params"] == ["center_y"]
    assert result["all_fixed_source_counters_clean"] is False
    assert result["all_passing_fixed_source_counters_clean"] is True


def test_new4_ladder_one_param_each_param_uses_same_base_state(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {"value": 1}}\n', encoding="utf-8")
    seen_hashes: list[str] = []
    _install_green_provider_guard(monkeypatch, ladder)
    monkeypatch.setattr(
        ladder,
        "_capture_solver_context",
        lambda *_args, **_kwargs: _minimal_new4_ladder_context(ladder),
    )
    monkeypatch.setattr(
        ladder,
        "run_objective_dry_run",
        lambda _context, *, output_path, max_nfev: (
            ladder._write_json(output_path, _green_rung1_report()) or _green_rung1_report()
        ),
    )
    monkeypatch.setattr(
        ladder,
        "run_sensitivity_scan",
        lambda _context, *, output_path, max_nfev, **_kwargs: (
            ladder._write_json(
                output_path,
                _green_rung2_report(active_params=("center_x", "center_y")),
            )
            or _green_rung2_report(active_params=("center_x", "center_y"))
        ),
    )

    def _solver(*, active_names, output_path, state_hash_before, **_kwargs):
        seen_hashes.append(state_hash_before)
        payload = _green_one_param_report(active_names)
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="one-param",
        timestamp="same_base",
    )

    assert len(set(seen_hashes)) == 1
    assert result["state_hash_unchanged"] is True


def test_new4_ladder_one_param_dirty_timeout_aborts_remaining(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    attempted: list[str] = []
    _install_green_provider_guard(monkeypatch, ladder)
    monkeypatch.setattr(
        ladder,
        "_capture_solver_context",
        lambda *_args, **_kwargs: _minimal_new4_ladder_context(ladder),
    )
    monkeypatch.setattr(
        ladder,
        "run_objective_dry_run",
        lambda _context, *, output_path, max_nfev: (
            ladder._write_json(output_path, _green_rung1_report()) or _green_rung1_report()
        ),
    )
    monkeypatch.setattr(
        ladder,
        "run_sensitivity_scan",
        lambda _context, *, output_path, max_nfev, **_kwargs: (
            ladder._write_json(
                output_path,
                _green_rung2_report(active_params=("center_x", "center_y", "gamma")),
            )
            or _green_rung2_report(active_params=("center_x", "center_y", "gamma"))
        ),
    )

    def _solver(*, active_names, output_path, **_kwargs):
        attempted.extend(active_names)
        payload = {
            "rung": 3,
            "rung_name": "one_param_center_x",
            "status": "timeout",
            "active_params": list(active_names),
            "var_names": list(active_names),
            "candidate_param_names": list(active_names),
            "elapsed_s": 0.01,
            "timeout_s": 0.01,
            "dirty_timeout_abort": True,
        }
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "_run_solver_rung_with_timeout", _solver)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="one-param",
        timestamp="dirty_timeout",
    )

    assert attempted == ["center_x"]
    assert result["dirty_timeout_abort"] is True
    assert result["status"] == "failed"
    assert result["skipped_params"] == ["center_y", "gamma"]


def test_new4_ladder_one_param_provider_guard_after(monkeypatch, tmp_path) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    guard_calls: list[str] = []
    _install_green_provider_guard(monkeypatch, ladder, calls=guard_calls)
    monkeypatch.setattr(
        ladder,
        "_capture_solver_context",
        lambda *_args, **_kwargs: _minimal_new4_ladder_context(ladder),
    )
    monkeypatch.setattr(
        ladder,
        "run_objective_dry_run",
        lambda _context, *, output_path, max_nfev: (
            ladder._write_json(output_path, _green_rung1_report()) or _green_rung1_report()
        ),
    )
    monkeypatch.setattr(
        ladder,
        "run_sensitivity_scan",
        lambda _context, *, output_path, max_nfev, **_kwargs: (
            ladder._write_json(output_path, _green_rung2_report(active_params=("center_x",)))
            or _green_rung2_report(active_params=("center_x",))
        ),
    )
    monkeypatch.setattr(
        ladder,
        "_run_solver_rung_with_timeout",
        lambda *, active_names, output_path, **_kwargs: (
            ladder._write_json(output_path, _green_one_param_report(active_names))
            or _green_one_param_report(active_names)
        ),
    )

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="one-param",
        timestamp="provider_after_one_param",
    )

    assert result["provider_guard_after_ok"] is True
    assert any("provider_guard_after" in call for call in guard_calls)


def _finalized_timeout_report(ladder, tmp_path, **overrides):
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    state_hash = hashlib.sha256(state_path.read_bytes()).hexdigest()
    payload = {
        "rung": 3,
        "rung_name": "one_param_a",
        "status": "timeout",
        "active_params": ["a"],
        "var_names": ["a"],
        "candidate_param_names": ["a"],
        "elapsed_s": 120.0,
        "timeout_s": 120.0,
        "last_nfev": 2,
        "last_residual_norm": 12.0,
        "last_rms_px": 5.0,
        "last_max_error_px": 8.0,
        "last_parameter_value": 4.2,
        "current_bounds": {"lower": 3.0, "upper": 5.0},
        "last_point_match_summary": _sensitivity_point_summary(),
        "fixed_source_counters_clean_at_last_heartbeat": True,
        "fixed_source_counter_failures_at_last_heartbeat": [],
        "fixed_source_counters_dirty_seen": False,
        "fixed_source_counter_failures_seen": [],
        "child_process_killed_cleanly": True,
        "dirty_timeout_abort": False,
        "state_sha256_before": state_hash,
        "state_sha256_after": state_hash,
    }
    payload.update(overrides)
    return ladder._finalize_one_param_report(
        payload,
        param_name="a",
        state_path=state_path,
        state_hash_before=state_hash,
        timeout_seconds=float(payload.get("timeout_s", 120.0)),
    )


def test_new4_ladder_timeout_with_nfev_progress_classifies_slow(tmp_path) -> None:
    ladder = _load_new4_ladder_module()

    report = _finalized_timeout_report(ladder, tmp_path)

    assert report["diagnosis_classification"] == "slow_needs_separate_strategy"
    assert report["failure_reason"] == "timeout"


def test_new4_ladder_timeout_without_nfev_progress_classifies_hang(tmp_path) -> None:
    ladder = _load_new4_ladder_module()

    report = _finalized_timeout_report(ladder, tmp_path, last_nfev=0)

    assert report["diagnosis_classification"] == "hang_solver_pathology"


def test_new4_ladder_dirty_fixed_source_heartbeat_classifies_integrity_loss(
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()

    report = _finalized_timeout_report(
        ladder,
        tmp_path,
        fixed_source_counters_clean_at_last_heartbeat=False,
        fixed_source_counter_failures_at_last_heartbeat=["missing_pair_count_1_expected_0"],
        fixed_source_counters_dirty_seen=True,
        fixed_source_counter_failures_seen=["missing_pair_count_1_expected_0"],
    )

    assert report["diagnosis_classification"] == "fixed_source_or_pair_integrity_lost"
    assert report["failure_reason"] == "fixed_source_or_pair_integrity_lost"


def test_new4_ladder_dirty_child_kill_aborts_variant_runner(monkeypatch, tmp_path) -> None:
    ladder = _load_new4_ladder_module()
    calls: list[str] = []

    def _run_ladder(*, timestamp, **_kwargs):
        calls.append(str(timestamp))
        return {
            "status": "failed",
            "reports": [
                {
                    "rung": 3,
                    "param_name": "a",
                    "status": "timeout",
                    "dirty_timeout_abort": True,
                    "diagnosis_classification": "hang_solver_pathology",
                }
            ],
        }

    monkeypatch.setattr(ladder, "run_ladder", _run_ladder)

    result = ladder.run_one_param_diagnosis_variants(
        state_path=tmp_path / "new4.json",
        background_index=0,
        output_root=tmp_path / "variants",
        one_param_filter="a",
    )

    assert calls == ["a_nfev5_t120"]
    assert result["failure_reason"] == "dirty_timeout_abort"


def test_new4_ladder_one_param_inactive_filter_aborts_variant_runner(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    calls: list[str] = []

    def _run_ladder(*, timestamp, **_kwargs):
        calls.append(str(timestamp))
        return {
            "status": "failed",
            "failure_reason": "filtered_param_not_active",
            "reports": [],
        }

    monkeypatch.setattr(ladder, "run_ladder", _run_ladder)

    result = ladder.run_one_param_diagnosis_variants(
        state_path=tmp_path / "new4.json",
        background_index=0,
        output_root=tmp_path / "variants",
        one_param_filter="a",
    )

    written = json.loads((tmp_path / "variants" / "variant_summary.json").read_text())
    assert calls == ["a_nfev5_t120"]
    assert result["status"] == "failed"
    assert result["failure_reason"] == "filtered_param_not_active"
    assert written["failure_reason"] == "filtered_param_not_active"


def test_new4_ladder_one_param_no_solve_guard_failure_aborts_variant_runner(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    calls: list[str] = []

    def _run_ladder(*, timestamp, **_kwargs):
        calls.append(str(timestamp))
        return {
            "status": "aborted",
            "reason": "provider_guard_failed",
            "reports": [],
        }

    monkeypatch.setattr(ladder, "run_ladder", _run_ladder)

    result = ladder.run_one_param_diagnosis_variants(
        state_path=tmp_path / "new4.json",
        background_index=0,
        output_root=tmp_path / "variants",
        one_param_filter="a",
    )

    assert calls == ["a_nfev5_t120"]
    assert result["status"] == "failed"
    assert result["failure_reason"] == "provider_guard_failed"


def test_new4_ladder_one_param_nonfinite_result_residual_not_usable(tmp_path) -> None:
    ladder = _load_new4_ladder_module()
    point_summary = {
        **_sensitivity_point_summary(),
        "unweighted_peak_rms_px": 5.0,
        "unweighted_peak_max_px": 8.0,
    }
    request_rows = [
        {
            "optimizer_request_has_fixed_source": True,
            "source_table_index": index,
            "source_peak_index": index,
        }
        for index in range(7)
    ]
    request = SimpleNamespace(
        var_names=["a"],
        candidate_param_names=["a"],
        params={"a": 4.1},
        dataset_specs=[{"measured_peaks": request_rows}],
        measured_peaks=[],
        refinement_config={
            "bounds": {"a": (3.0, 5.0)},
            "optimizer_request_handoff_summary": {
                "provider_pair_count": 7,
                "dataset_pair_count": 7,
                "provider_to_optimizer_identity_match": True,
                "provider_to_optimizer_point_match": True,
            },
        },
    )
    result = SimpleNamespace(
        fun=np.asarray([1.0, np.nan], dtype=float),
        point_match_summary=point_summary,
        rms_px=5.0,
        x=np.asarray([4.2], dtype=float),
        nfev=3,
        success=True,
        message="ok",
    )
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    state_hash = hashlib.sha256(state_path.read_bytes()).hexdigest()

    report = ladder._result_report(
        request=request,
        result=result,
        rung=3,
        rung_name="one_param_a",
        started_at=0.0,
        before_summary={**point_summary, "unweighted_peak_rms_px": 6.0},
        extra={
            "least_squares_called": True,
            "optimizer_solve_called": True,
            "real_solve_called": True,
            "state_sha256_before": state_hash,
            "state_sha256_after": state_hash,
            "state_hash_unchanged": True,
        },
    )
    finalized = ladder._finalize_one_param_report(
        report,
        param_name="a",
        state_path=state_path,
        state_hash_before=state_hash,
        timeout_seconds=120.0,
    )

    assert report["residuals_finite"] is False
    assert not np.isfinite(report["last_residual_norm"])
    assert finalized["status"] == "failed"
    assert finalized["failure_reason"] == "non_finite_residual"
    assert finalized["diagnosis_classification"] != "usable"


def test_new4_ladder_timeout_writes_partial_report(monkeypatch, tmp_path) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")

    def _sleeping_worker(**kwargs):
        ladder._heartbeat_write(
            kwargs["heartbeat_path"],
            {
                "status": "running",
                "last_nfev": 2,
                "nfev": 2,
                "last_residual_norm": 12.0,
                "last_rms_px": 5.0,
                "last_max_error_px": 8.0,
                "last_parameter_value": 4.2,
                "current_bounds": {"lower": 3.0, "upper": 5.0},
                "last_point_match_summary": _sensitivity_point_summary(),
                "fixed_source_counters_clean_at_last_heartbeat": True,
                "fixed_source_counter_failures_at_last_heartbeat": [],
                "heartbeat_count": 1,
                "last_heartbeat_elapsed_s": 0.01,
                "residual_eval_trace": [
                    {
                        "eval_count": 2,
                        "nfev": 2,
                        "elapsed_s": 0.01,
                        "residual_norm": 12.0,
                        "rms_px": 5.0,
                        "max_error_px": 8.0,
                        "parameter_name": "gamma",
                        "parameter_value": 4.2,
                        "bounds": {"lower": 3.0, "upper": 5.0},
                        "point_match_summary": _sensitivity_point_summary(),
                        "fixed_source_counters_clean": True,
                        "fixed_source_counter_failures": [],
                    }
                ],
            },
        )
        time.sleep(0.2)
        return {"status": "ok", "pass": True}

    monkeypatch.setattr(ladder, "_worker_solve_once", _sleeping_worker)
    output_path = tmp_path / "rung_timeout.json"
    report = ladder._run_solver_rung_with_timeout(
        state_path=state_path,
        background_index=0,
        active_names=["gamma"],
        output_path=output_path,
        max_nfev=20,
        timeout_seconds=0.01,
        rung=3,
        rung_name="one_param_gamma",
        use_subprocess=False,
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "timeout"
    assert written["status"] == "timeout"
    assert written["active_params"] == ["gamma"]
    assert written["candidate_param_names"] == ["gamma"]
    assert "elapsed_s" in written
    assert written["timeout_s"] == 0.01
    for key in (
        "before_rms_px",
        "after_rms_px",
        "before_max_error_px",
        "after_max_error_px",
        "parameter_before",
        "parameter_after",
        "parameter_delta",
        "parameter_bounds",
        "residuals_finite",
        "provider_pair_count",
        "dataset_pair_count",
        "optimizer_request_pair_count",
        "fixed_source_pair_count",
        "fallback_row_count",
        "fixed_source_resolution_fallback_count",
        "missing_fixed_source_count",
        "fixed_source_resolved_count",
        "fallback_entry_count",
        "matched_pair_count",
        "missing_pair_count",
        "branch_mismatch_count",
        "provider_to_optimizer_identity_match",
        "provider_to_optimizer_point_match",
        "least_squares_called",
        "optimizer_solve_called",
        "state_sha256_before",
        "state_sha256_after",
        "state_hash_unchanged",
        "failure_reason",
        "last_nfev",
        "last_residual_norm",
        "last_rms_px",
        "last_max_error_px",
        "last_parameter_value",
        "current_bounds",
        "last_point_match_summary",
        "last_heartbeat_elapsed_s",
        "heartbeat_count",
        "fixed_source_counters_clean_at_last_heartbeat",
        "fixed_source_counter_failures_at_last_heartbeat",
        "child_process_killed_cleanly",
        "dirty_timeout_abort",
        "diagnosis_classification",
    ):
        assert key in written
    assert written["failure_reason"] == "timeout"
    assert written["last_nfev"] == 2
    assert written["last_residual_norm"] == 12.0
    assert written["last_point_match_summary"]["matched_pair_count"] == 7
    assert written["heartbeat_count"] == 1


def test_new4_ladder_warm_solver_passes_context_to_worker(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text('{"state": {}}\n', encoding="utf-8")
    context = {"prepared_run": object()}
    seen: dict[str, object] = {}

    def _worker(**kwargs):
        seen.update(kwargs)
        payload = {
            "rung": 4,
            "rung_name": "pair_a_c",
            "status": "ok",
            "pass": True,
            "active_params": ["a", "c"],
            "candidate_param_names": ["a", "c"],
            "phase_timing_s": {"capture_solver_context_s": 0.0},
            "first_residual_elapsed_s": 0.01,
            "solver_context_reused": True,
            "diagnostic_logging": True,
        }
        ladder._write_json(kwargs["output_path"], payload)
        return payload

    monkeypatch.setattr(ladder, "_worker_solve_once", _worker)

    report = ladder._run_solver_rung_with_timeout(
        state_path=state_path,
        background_index=0,
        active_names=["a", "c"],
        output_path=tmp_path / "rung_04_pair_a_c.json",
        max_nfev=20,
        timeout_seconds=1.0,
        rung=4,
        rung_name="pair_a_c",
        use_subprocess=False,
        context=context,
        diagnostic_logging=True,
    )

    assert seen["context"] is context
    assert seen["diagnostic_logging"] is True
    assert report["solver_context_reused"] is True
    assert report["phase_timing_s"]["capture_solver_context_s"] == 0.0
    assert report["first_residual_elapsed_s"] == 0.01


def test_new4_ladder_does_not_mutate_new4_state(monkeypatch, tmp_path) -> None:
    ladder = _load_new4_ladder_module()
    repo_root = Path(__file__).resolve().parents[1]
    state_path = repo_root / "artifacts" / "geometry_fit_gui_states" / "new4.json"
    before_hash = hashlib.sha256(state_path.read_bytes()).hexdigest()

    def _green_guard(*, output_path, **_kwargs):
        payload = _green_new4_ladder_provider_payload()
        ladder._write_json(output_path, payload)
        return payload

    monkeypatch.setattr(ladder, "run_provider_guard", _green_guard)
    _install_fast_new4_ladder_stubs(monkeypatch, ladder)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="center",
        timestamp="immutability",
    )

    after_hash = hashlib.sha256(state_path.read_bytes()).hexdigest()
    assert result["status"] == "pass"
    assert after_hash == before_hash


def test_new4_ladder_keeps_point_provider_report_green(monkeypatch, tmp_path) -> None:
    ladder = _load_new4_ladder_module()
    repo_root = Path(__file__).resolve().parents[1]
    state_path = repo_root / "artifacts" / "geometry_fit_gui_states" / "new4.json"
    _install_fast_new4_ladder_stubs(monkeypatch, ladder)

    result = ladder.run_ladder(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path,
        max_rung="center",
        timestamp="provider_after",
    )
    report = ladder.preflight_probe._run_point_provider_report_only(
        state_path,
        background_index=0,
    )

    assert result["status"] == "pass"
    assert report["ok"] is True
    assert report["classification"] == "point_provider_parity_ok"
    assert report["point_provider_parity_gate"]["ok"] is True
    assert report["manual_picker_pair_count"] == 7
    assert report["point_provider_pair_count"] == 7


def test_new4_ladder_objective_dry_run_rejects_fallback_rows(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    context = _new4_ladder_context_with_provider_rows(
        full_source=False,
        missing_identity=True,
    )

    def _should_not_probe(*_args, **_kwargs):
        raise AssertionError("objective probe should not run for fallback request")

    monkeypatch.setattr(ladder, "_run_with_probe_least_squares", _should_not_probe)
    report = ladder.run_objective_dry_run(
        context,
        output_path=tmp_path / "rung_01_objective_dry_run.json",
        max_nfev=20,
    )

    assert report["status"] == "failed"
    assert report["failure_reason"] == "optimizer_request_used_fallback_rows"
    assert report["optimizer_request_pair_count"] == 7
    assert report["fallback_row_count"] == 7
    assert report["fixed_source_resolution_fallback_count"] == 7
    assert report["missing_fixed_source_count"] == 7
    assert report["objective_eval_called"] is False
    assert report["least_squares_called"] is False
    assert report["optimizer_solve_called"] is False
    assert report["objective_dry_run_residual_finite"] is False


@pytest.mark.parametrize("full_source", [True, False])
def test_new4_ladder_objective_dry_run_preserves_provider_fixed_rows(
    monkeypatch,
    tmp_path,
    full_source,
) -> None:
    ladder = _load_new4_ladder_module()
    context = _new4_ladder_context_with_provider_rows(full_source=full_source)
    captured: dict[str, object] = {}

    def _fake_probe(request, *, mode):
        captured["mode"] = mode
        captured["request"] = request
        point_match_summary = {
            "measured_count": 7,
            "matched_pair_count": 7,
            "missing_pair_count": 0,
            "branch_mismatch_count": 0,
            "fixed_source_resolved_count": 7,
            "fixed_source_reflection_count": 7,
            "matched_fixed_pair_count": 7,
            "missing_fixed_pair_count": 0,
            "fallback_entry_count": 0,
            "fallback_hkl_count": 0,
            "subset_fallback_hkl_count": 0,
            "unweighted_peak_rms_px": 0.5,
            "unweighted_peak_max_px": 0.75,
        }
        return (
            SimpleNamespace(
                point_match_summary=point_match_summary,
                rms_px=0.5,
                success=True,
                message="dry-run",
                nfev=1,
                x=np.asarray([32.0], dtype=float),
                geometry_fit_stage_timings={},
            ),
            [{"finite": True, "residual_norm": 1.0}],
        )

    monkeypatch.setattr(ladder, "_run_with_probe_least_squares", _fake_probe)
    report = ladder.run_objective_dry_run(
        context,
        output_path=tmp_path / "rung_01_objective_dry_run.json",
        max_nfev=20,
    )

    request = captured["request"]
    request_rows = list(request.dataset_specs[0]["measured_peaks"])
    provider_pairs = context["prepared_run"].current_dataset["provider_pairs"]
    assert report["status"] == "ok"
    assert report["pass"] is True
    assert report["optimizer_request_pair_count"] == 7
    assert report["fixed_source_pair_count"] == 7
    assert report["fallback_row_count"] == 0
    assert report["fixed_source_resolution_fallback_count"] == 0
    assert report["provider_to_optimizer_identity_match"] is True
    assert report["provider_to_optimizer_point_match"] is True
    assert report["objective_eval_called"] is True
    assert report["least_squares_called"] is False
    assert report["optimizer_solve_called"] is False
    assert report["objective_dry_run_residual_finite"] is True
    for index, row in enumerate(request_rows):
        identity = provider_pairs[index]["selected_source_identity_canonical"]
        assert row["hkl"] == tuple(identity["normalized_hkl"])
        if full_source:
            assert row["source_reflection_index"] == identity["source_reflection_index"]
            assert row["source_reflection_namespace"] == "full_reflection"
            assert row["fit_source_resolution_kind"] == "provider_fixed_source"
        else:
            assert "source_reflection_index" not in row
            assert "source_reflection_namespace" not in row
            assert row["fit_source_resolution_kind"] == "provider_fixed_source_local"
            assert row["source_table_index"] == identity["source_table_index"]
        assert row["source_peak_index"] == identity["source_peak_index"]
        assert row["x"] == pytest.approx(10.0 + index)
        assert row["y"] == pytest.approx(20.0 + index)


def test_optimizer_request_preserves_local_provider_source_identity() -> None:
    ladder = _load_new4_ladder_module()
    context = _new4_ladder_context_with_provider_rows(full_source=False)

    request = ladder.build_solver_request(context, ["center_x"], max_nfev=20)
    summary = dict(request.refinement_config["optimizer_request_handoff_summary"])
    request_rows = list(request.dataset_specs[0]["measured_peaks"])

    assert summary["optimizer_request_pair_count"] == 7
    assert summary["fixed_source_pair_count"] == 7
    assert summary["fallback_row_count"] == 0
    assert summary["fixed_source_resolution_fallback_count"] == 0
    assert summary["missing_fixed_source_count"] == 0
    assert summary["provider_to_optimizer_identity_match"] is True
    assert summary["provider_to_optimizer_point_match"] is True
    for index, row in enumerate(request_rows):
        assert row["fit_source_resolution_kind"] == "provider_fixed_source_local"
        assert row["optimizer_request_has_fixed_source"] is True
        assert row["optimizer_request_fallback_row"] is False
        assert "source_reflection_index" not in row
        assert row["source_table_index"] == index
        assert row["source_peak_index"] == index % 2


def test_optimizer_request_rejects_inconsistent_local_provider_source() -> None:
    ladder = _load_new4_ladder_module()
    context = _new4_ladder_context_with_provider_rows(full_source=False)
    identity = context["prepared_run"].current_dataset["provider_pairs"][0][
        "selected_source_identity_canonical"
    ]
    identity["source_table_index"] = 1
    identity["source_table_index_namespace"] = "full_reflection"

    request = ladder.build_solver_request(context, ["center_x"], max_nfev=20)
    summary = dict(request.refinement_config["optimizer_request_handoff_summary"])
    first_row = request.dataset_specs[0]["measured_peaks"][0]

    assert summary["fixed_source_pair_count"] == 6
    assert summary["fallback_row_count"] == 1
    assert summary["missing_fixed_source_count"] == 1
    assert first_row["optimizer_request_fallback_row"] is True
    assert "source_table_index_hkl_mismatch" in first_row["optimizer_request_fallback_reason"]


def test_new4_ladder_does_not_start_solve_if_objective_uses_fallback(
    monkeypatch,
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    context = _new4_ladder_context_with_provider_rows(
        full_source=False,
        missing_identity=True,
    )
    called = {"least_squares": False}

    def _raising_least_squares(*_args, **_kwargs):
        called["least_squares"] = True
        raise AssertionError("least_squares should not be called")

    monkeypatch.setattr(ladder.opt, "least_squares", _raising_least_squares)
    report = ladder.run_objective_dry_run(
        context,
        output_path=tmp_path / "rung_01_objective_dry_run.json",
        max_nfev=20,
    )

    assert report["status"] == "failed"
    assert report["failure_reason"] == "optimizer_request_used_fallback_rows"
    assert called["least_squares"] is False
    assert report["objective_eval_called"] is False
    assert report["least_squares_called"] is False


def test_new4_rung1_direct_objective_dry_run_green_or_fail_before_solve(
    tmp_path,
) -> None:
    ladder = _load_new4_ladder_module()
    repo_root = Path(__file__).resolve().parents[1]
    state_path = repo_root / "artifacts" / "geometry_fit_gui_states" / "new4.json"

    context = ladder._capture_solver_context(state_path, 0)
    report = ladder.run_objective_dry_run(
        context,
        output_path=tmp_path / "rung_01_objective_dry_run.json",
        max_nfev=20,
    )

    assert report["status"] == "ok"
    assert report["pass"] is True
    assert report["provider_pair_count"] == 7
    assert report["dataset_pair_count"] == 7
    assert report["optimizer_request_pair_count"] == 7
    assert report["fixed_source_pair_count"] == 7
    assert report["fallback_row_count"] == 0
    assert report["fixed_source_resolution_fallback_count"] == 0
    assert report["missing_fixed_source_count"] == 0
    assert report["provider_to_optimizer_identity_match"] is True
    assert report["provider_to_optimizer_point_match"] is True
    assert report["objective_eval_called"] is True
    assert report["least_squares_called"] is False
    assert report["optimizer_solve_called"] is False
    assert report["objective_dry_run_residual_finite"] is True
    assert report["matched_pair_count"] == 7
    assert report["missing_pair_count"] == 0
    assert report["branch_mismatch_count"] == 0


def test_build_geometry_manual_fit_dataset_skips_cached_candidate_with_missing_hkl() -> None:
    saved_entries = [
        {
            "pair_id": "bg0:pair0",
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 1,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "x": 1000.0,
            "y": 1000.0,
        }
    ]
    simulated_rows = [
        {
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 1,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "sim_col": 2504.0,
            "sim_row": 2497.0,
        },
        {
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 2,
            "source_reflection_index": 204,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "sim_col": 1003.0,
            "sim_row": 998.0,
        },
    ]
    manual_dataset_bindings = _make_legacy_dense_manual_dataset_bindings(
        saved_entries=saved_entries,
        simulated_rows=simulated_rows,
        refresh_pairs=False,
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={"mode": "auto"},
    )

    assert dataset["resolved_source_pair_count"] == 1
    measured_entry = dataset["measured_for_fit"][0]
    diag = dataset["source_resolution_diagnostics"][0]
    assert measured_entry["source_reflection_index"] == 204
    assert diag["strict_resolved"] is False
    assert diag["fit_resolved"] is True
    assert diag["fit_resolution_kind"] == "q_group_fallback"
    assert diag["selected_candidate_source_identity_fields"]["source_reflection_index"] == 204


def test_build_geometry_manual_fit_dataset_rejects_nonlegacy_candidates_without_current_view_point() -> (
    None
):
    saved_entries = [
        {
            "pair_id": "bg0:pair0",
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "x": 1000.0,
            "y": 1000.0,
        }
    ]
    simulated_rows = [
        {
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 1,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "native_col": 2504.0,
            "native_row": 2497.0,
        },
        {
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 2,
            "source_reflection_index": 204,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "native_col": 1003.0,
            "native_row": 998.0,
        },
    ]
    manual_dataset_bindings = _make_legacy_dense_manual_dataset_bindings(
        saved_entries=saved_entries,
        simulated_rows=simulated_rows,
        refresh_pairs=False,
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={"mode": "auto"},
    )

    assert dataset["resolved_source_pair_count"] == 0
    assert "source_reflection_index" not in dataset["measured_for_fit"][0]
    diag = dataset["source_resolution_diagnostics"][0]
    assert diag["strict_resolved"] is False
    assert diag["fit_resolved"] is False
    assert diag["fit_resolution_kind"] is None


def test_build_geometry_manual_fit_dataset_fails_closed_on_legacy_dense_geometry_tie() -> None:
    saved_entries = [
        {
            "pair_id": "bg0:pair0",
            "q_group_key": ("q", 2),
            "source_table_index": 9,
            "source_reflection_index": 9,
            "source_row_index": 0,
            "source_peak_index": 9,
            "hkl": (2, 0, 1),
            "x": 100.0,
            "y": 80.0,
            "caked_y": 4.0,
            "refined_sim_caked_y": 4.0,
            "refined_sim_x": 100.0,
            "refined_sim_y": 80.0,
        }
    ]
    simulated_rows = [
        {
            "q_group_key": ("q", 2),
            "source_table_index": 0,
            "source_reflection_index": 300,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (2, 0, 1),
            "sim_col": 100.0,
            "sim_row": 80.0,
        },
        {
            "q_group_key": ("q", 2),
            "source_table_index": 1,
            "source_reflection_index": 301,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (2, 0, 1),
            "sim_col": 100.0000005,
            "sim_row": 80.0,
        },
    ]
    manual_dataset_bindings = _make_legacy_dense_manual_dataset_bindings(
        saved_entries=saved_entries,
        simulated_rows=simulated_rows,
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={"mode": "auto"},
    )

    assert dataset["resolved_source_pair_count"] == 0
    assert "source_reflection_index" not in dataset["measured_for_fit"][0]
    assert "source_reflection_index" not in dataset["initial_pairs_display"][0]
    diag = dataset["source_resolution_diagnostics"][0]
    assert diag["fit_resolved"] is False
    assert diag["fit_resolution_kind"] is None
    assert diag["failure_reason"] == "legacy_rebind_ambiguous_geometry_tie"
    assert diag["legacy_geometry_hint_source"] == "measured_display"
    assert diag["legacy_candidate_count_after_branch"] == 2
    assert diag["legacy_best_score"] == pytest.approx(0.0)
    assert diag["legacy_second_best_score"] == pytest.approx(5.0e-7)


def test_headless_geometry_fit_legacy_dense_rebind_matches_shared_preflight(
    monkeypatch,
    tmp_path,
) -> None:
    from ra_sim import headless_geometry_fit

    saved_entries = [
        {
            "pair_id": "bg0:pair0",
            "q_group_key": ("q", 5),
            "source_table_index": 12,
            "source_reflection_index": 12,
            "source_row_index": 0,
            "source_peak_index": 12,
            "hkl": (-1, 0, 5),
            "x": 110.0,
            "y": 100.0,
            "caked_y": -8.0,
            "refined_sim_caked_y": -8.0,
            "refined_sim_x": 110.0,
            "refined_sim_y": 100.0,
        },
        {
            "pair_id": "bg0:pair1",
            "q_group_key": ("q", 5),
            "source_table_index": 13,
            "source_reflection_index": 13,
            "source_row_index": 0,
            "source_peak_index": 13,
            "hkl": (-1, 0, 5),
            "x": 189.5,
            "y": 97.0,
            "caked_y": 9.0,
            "refined_sim_caked_y": 9.0,
            "refined_sim_x": 189.5,
            "refined_sim_y": 97.0,
        },
    ]
    simulated_rows = [
        {
            "q_group_key": ("q", 5),
            "source_table_index": 0,
            "source_reflection_index": 202,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "hkl": (-1, 0, 5),
            "sim_col": 110.0,
            "sim_row": 100.0,
        },
        {
            "q_group_key": ("q", 5),
            "source_table_index": 1,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "sim_col": 182.0,
            "sim_row": 98.0,
        },
        {
            "q_group_key": ("q", 5),
            "source_table_index": 2,
            "source_reflection_index": 204,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "sim_col": 190.0,
            "sim_row": 96.0,
        },
    ]
    workflow_dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=_make_legacy_dense_manual_dataset_bindings(
            saved_entries=saved_entries,
            simulated_rows=simulated_rows,
        ),
        orientation_cfg={"mode": "auto"},
    )

    defaults = headless_geometry_fit._RuntimeDefaults(
        primary_cif_path="C:/tmp/primary.cif",
        secondary_cif_path=None,
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=64,
        pixel_size_m=1.0,
        lambda_angstrom=1.54,
        psi_deg=0.0,
        defaults={
            "theta_initial": 1.5,
            "cor_angle": 0.0,
            "gamma": 0.0,
            "Gamma": 0.0,
            "chi": 0.0,
            "psi_z": 0.0,
            "zs": 0.0,
            "zb": 0.0,
            "sample_width_m": 0.0,
            "sample_length_m": 0.0,
            "sample_depth_m": 0.0,
            "debye_x": 0.0,
            "debye_y": 0.0,
            "corto_detector": 0.075,
            "a": 4.0,
            "b": 4.0,
            "c": 10.0,
            "a2": None,
            "c2": None,
            "center_x": 32.0,
            "center_y": 32.0,
            "sigma_mosaic_deg": 1.0,
            "gamma_mosaic_deg": 1.0,
            "eta": 0.0,
            "bandwidth_percent": 0.0,
            "solve_q_steps": 64,
            "solve_q_rel_tol": 1.0e-4,
            "solve_q_mode": 1,
            "optics_mode": 0,
            "p0": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "w0": 1.0,
            "w1": 0.0,
            "w2": 0.0,
            "finite_stack": False,
            "stack_layers": 1,
            "phase_delta_expression": "0.0",
            "phi_l_divisor": 1.0,
            "weight1": 1.0,
            "weight2": 1.0,
        },
        fit_config={},
        intensity_threshold=0.0,
        include_rods_flag=False,
        two_theta_range=(0.0, 90.0),
        mx=4,
        background_flags={
            "backend_rotation_k": 0,
            "backend_flip_x": False,
            "backend_flip_y": False,
        },
    )

    captured = {}

    monkeypatch.setattr(
        headless_geometry_fit, "_build_runtime_defaults", lambda saved_state: defaults
    )
    monkeypatch.setattr(
        headless_geometry_fit,
        "_restore_manual_pairs",
        lambda osc_files, saved_rows: {0: [dict(entry) for entry in saved_entries]},
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_manual_geometry,
        "geometry_manual_pairs_for_index",
        lambda index, *, pairs_by_background: [
            dict(entry) for entry in pairs_by_background.get(int(index), ())
        ],
    )
    monkeypatch.setattr(
        headless_geometry_fit,
        "_load_structure_model",
        lambda defaults, saved_state, var_store, simulation_runtime_state: (
            SimpleNamespace(
                miller=np.asarray([[0, 0, 1]], dtype=np.int64),
                intensities=np.asarray([1.0], dtype=np.float64),
            ),
            None,
            "C:/tmp/primary.cif",
            complex(1.0, 0.0),
        ),
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_background,
        "load_background_image_by_index",
        lambda index, **kwargs: {
            "background_images": [np.zeros((4, 5), dtype=np.float64)],
            "background_images_native": [np.zeros((4, 5), dtype=np.float64)],
            "background_images_display": [np.zeros((4, 5), dtype=np.float64)],
            "background_image": np.zeros((4, 5), dtype=np.float64),
            "background_display": np.zeros((4, 5), dtype=np.float64),
        },
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_background_theta,
        "current_geometry_fit_background_indices",
        lambda **kwargs: [0],
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_background_theta,
        "geometry_fit_uses_shared_theta_offset",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_background_theta,
        "current_geometry_theta_offset",
        lambda **kwargs: 0.0,
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_background_theta,
        "current_background_theta_values",
        lambda **kwargs: [1.5],
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_background_theta,
        "background_theta_for_index",
        lambda index, **kwargs: 1.5,
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_background_theta,
        "apply_background_theta_metadata",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_background_theta,
        "apply_geometry_fit_background_selection",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_geometry_fit,
        "build_runtime_geometry_fit_value_callbacks",
        lambda bindings: SimpleNamespace(
            current_var_names=lambda: ["gamma"],
            current_params=lambda: {
                "a": 4.0,
                "c": 10.0,
                "lambda": 1.54,
                "theta_initial": 1.5,
                "theta_offset": 0.0,
                "corto_detector": 0.075,
                "gamma": 0.0,
                "Gamma": 0.0,
                "chi": 0.0,
                "cor_angle": 0.0,
                "zb": 0.0,
                "zs": 0.0,
                "center": [32.0, 32.0],
            },
            current_ui_params=lambda: {},
            var_map={},
        ),
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_geometry_q_group_manager,
        "make_runtime_geometry_fit_simulation_callbacks",
        lambda **kwargs: SimpleNamespace(
            simulate_hit_tables=lambda *args, **kwargs: [],
            last_simulation_diagnostics=lambda: {},
        ),
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_manual_geometry,
        "make_runtime_geometry_manual_projection_callbacks",
        lambda **kwargs: SimpleNamespace(
            simulated_peaks_for_params=lambda params, *, prefer_cache: [],
            simulated_lookup=lambda simulated_peaks: {},
            entry_display_coords=lambda entry: (
                float(entry.get("x", 0.0)),
                float(entry.get("y", 0.0)),
            ),
            refresh_entry_geometry=_refresh_legacy_dense_pair_entry,
        ),
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_manual_geometry,
        "geometry_manual_live_peak_candidates_from_records",
        lambda records, **kwargs: [dict(row) for row in simulated_rows],
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_geometry_q_group_manager,
        "_resolve_live_peak_record_fallback_provenance",
        lambda *args, **kwargs: {
            "active_signature_matches": True,
            "active_revision_matches": False,
            "expected_table_count": len(simulated_rows),
            "source_snapshot_row_count": len(simulated_rows),
            "source_snapshot_background_index": 0,
            "source_row_hkl_lookup": {
                (int(row["source_table_index"]), int(row["source_row_index"])): tuple(
                    int(v) for v in row["hkl"]
                )
                for row in simulated_rows
            },
        },
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_geometry_fit,
        "rebuild_geometry_fit_source_rows",
        lambda **kwargs: geometry_fit.GeometryFitSourceRowRebuildResult(
            background_index=0,
            requested_signature=("sig",),
            requested_signature_summary="sig",
            projected_rows=[dict(row) for row in simulated_rows],
            stored_rows=[dict(row) for row in simulated_rows],
            rebuild_source="peak_records",
            rebuild_attempts=["live_preview"],
            diagnostics={"status": "rebuild_ok"},
            source_reflection_indices=[
                int(row["source_reflection_index"]) for row in simulated_rows
            ],
            metadata={},
        ),
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_geometry_overlay,
        "select_fit_orientation",
        lambda sim_pts, meas_pts, shape, *, cfg: (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "yx",
                "label": "identity",
            },
            {"pairs": len(sim_pts)},
        ),
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_geometry_overlay,
        "apply_orientation_to_entries",
        lambda entries, shape, **kwargs: [dict(entry) for entry in entries],
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_geometry_overlay,
        "orient_image_for_fit",
        lambda image, **kwargs: image,
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_geometry_overlay,
        "unrotate_display_peaks",
        lambda entries, rotated_shape, *, k=None, default_display_rotate_k=0: [
            dict(entry) for entry in entries
        ],
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_geometry_overlay,
        "display_to_native_sim_coords",
        lambda col, row, image_shape, sim_display_rotate_k=0: (float(col), float(row)),
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_geometry_overlay,
        "native_sim_to_display_coords",
        lambda col, row, image_shape, sim_display_rotate_k=0: (float(col), float(row)),
    )
    monkeypatch.setattr(headless_geometry_fit, "get_dir", lambda name: str(tmp_path))
    monkeypatch.setattr(
        headless_geometry_fit.gui_geometry_fit,
        "build_runtime_geometry_fit_execution_setup",
        lambda **kwargs: SimpleNamespace(),
    )

    def _fake_execute_runtime_geometry_fit(*, prepared_run, **kwargs):
        captured["dataset"] = prepared_run.current_dataset
        captured["geometry_runtime_cfg"] = prepared_run.geometry_runtime_cfg
        return geometry_fit.GeometryFitRuntimeExecutionResult(
            log_path=tmp_path / "headless_geometry_fit.log",
            apply_result=geometry_fit.GeometryFitRuntimeApplyResult(
                accepted=True,
                rejection_reason=None,
                rms=0.0,
                fitted_params=None,
                postprocess=None,
            ),
        )

    monkeypatch.setattr(
        headless_geometry_fit.gui_geometry_fit,
        "execute_runtime_geometry_fit",
        _fake_execute_runtime_geometry_fit,
    )

    saved_state = {
        "files": {
            "background_files": ["C:/tmp/bg0.osc"],
            "current_background_index": 0,
        },
        "geometry": {
            "manual_pairs": [
                {"background_index": 0, "entries": [dict(entry) for entry in saved_entries]}
            ],
            "peak_records": [{"dummy": True}],
        },
    }

    result = headless_geometry_fit.run_headless_geometry_fit(
        saved_state,
        state_path=tmp_path / "saved_state.json",
        downloads_dir=tmp_path,
        stamp="headless_test",
    )

    assert result.accepted is True
    assert result.rms_px == 0.0
    headless_dataset = captured["dataset"]
    assert (
        headless_dataset["resolved_source_pair_count"]
        == workflow_dataset["resolved_source_pair_count"]
    )
    assert [entry["source_reflection_index"] for entry in headless_dataset["measured_for_fit"]] == [
        entry["source_reflection_index"] for entry in workflow_dataset["measured_for_fit"]
    ]
    assert [entry["source_branch_index"] for entry in headless_dataset["measured_for_fit"]] == [
        entry["source_branch_index"] for entry in workflow_dataset["measured_for_fit"]
    ]
    assert [
        entry["fit_source_resolution_kind"] for entry in headless_dataset["measured_for_fit"]
    ] == [entry["fit_source_resolution_kind"] for entry in workflow_dataset["measured_for_fit"]]


def test_headless_geometry_fit_canonical_pairs_match_shared_preflight(
    monkeypatch,
    tmp_path,
) -> None:
    from ra_sim import headless_geometry_fit

    saved_entries = [
        {
            "pair_id": "bg0:pair0",
            "q_group_key": ("q", 5),
            "source_table_index": 0,
            "source_reflection_index": 202,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "hkl": (-1, 0, 5),
            "x": 110.0,
            "y": 100.0,
            "caked_y": -8.0,
            "refined_sim_caked_y": -8.0,
            "refined_sim_x": 110.0,
            "refined_sim_y": 100.0,
        },
        {
            "pair_id": "bg0:pair1",
            "q_group_key": ("q", 5),
            "source_table_index": 1,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "x": 189.5,
            "y": 97.0,
            "caked_y": 9.0,
            "refined_sim_caked_y": 9.0,
            "refined_sim_x": 189.5,
            "refined_sim_y": 97.0,
        },
    ]
    simulated_rows = [
        {
            "q_group_key": ("q", 5),
            "source_table_index": 0,
            "source_reflection_index": 202,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "hkl": (-1, 0, 5),
            "sim_col": 110.0,
            "sim_row": 100.0,
        },
        {
            "q_group_key": ("q", 5),
            "source_table_index": 1,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "sim_col": 190.0,
            "sim_row": 96.0,
        },
        {
            "q_group_key": ("q", 5),
            "source_table_index": 2,
            "source_reflection_index": 204,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "sim_col": 182.0,
            "sim_row": 98.0,
        },
    ]
    workflow_dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=_make_legacy_dense_manual_dataset_bindings(
            saved_entries=saved_entries,
            simulated_rows=simulated_rows,
            refresh_pairs=False,
        ),
        orientation_cfg={"mode": "auto"},
    )

    defaults = headless_geometry_fit._RuntimeDefaults(
        primary_cif_path="C:/tmp/primary.cif",
        secondary_cif_path=None,
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=64,
        pixel_size_m=1.0,
        lambda_angstrom=1.54,
        psi_deg=0.0,
        defaults={
            "theta_initial": 1.5,
            "cor_angle": 0.0,
            "gamma": 0.0,
            "Gamma": 0.0,
            "chi": 0.0,
            "psi_z": 0.0,
            "zs": 0.0,
            "zb": 0.0,
            "sample_width_m": 0.0,
            "sample_length_m": 0.0,
            "sample_depth_m": 0.0,
            "debye_x": 0.0,
            "debye_y": 0.0,
            "corto_detector": 0.075,
            "a": 4.0,
            "b": 4.0,
            "c": 10.0,
            "a2": None,
            "c2": None,
            "center_x": 32.0,
            "center_y": 32.0,
            "sigma_mosaic_deg": 1.0,
            "gamma_mosaic_deg": 1.0,
            "eta": 0.0,
            "bandwidth_percent": 0.0,
            "solve_q_steps": 64,
            "solve_q_rel_tol": 1.0e-4,
            "solve_q_mode": 1,
            "optics_mode": 0,
            "p0": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "w0": 1.0,
            "w1": 0.0,
            "w2": 0.0,
            "finite_stack": False,
            "stack_layers": 1,
            "phase_delta_expression": "0.0",
            "phi_l_divisor": 1.0,
            "weight1": 1.0,
            "weight2": 1.0,
        },
        fit_config={},
        intensity_threshold=0.0,
        include_rods_flag=False,
        two_theta_range=(0.0, 90.0),
        mx=4,
        background_flags={
            "backend_rotation_k": 0,
            "backend_flip_x": False,
            "backend_flip_y": False,
        },
    )

    captured = {}

    monkeypatch.setattr(
        headless_geometry_fit, "_build_runtime_defaults", lambda saved_state: defaults
    )
    monkeypatch.setattr(
        headless_geometry_fit,
        "_restore_manual_pairs",
        lambda osc_files, saved_rows: {0: [dict(entry) for entry in saved_entries]},
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_manual_geometry,
        "geometry_manual_pairs_for_index",
        lambda index, *, pairs_by_background: [
            dict(entry) for entry in pairs_by_background.get(int(index), ())
        ],
    )
    monkeypatch.setattr(
        headless_geometry_fit,
        "_load_structure_model",
        lambda defaults, saved_state, var_store, simulation_runtime_state: (
            SimpleNamespace(
                miller=np.asarray([[0, 0, 1]], dtype=np.int64),
                intensities=np.asarray([1.0], dtype=np.float64),
            ),
            None,
            "C:/tmp/primary.cif",
            complex(1.0, 0.0),
        ),
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_background,
        "load_background_image_by_index",
        lambda index, **kwargs: {
            "background_images": [np.zeros((4, 5), dtype=np.float64)],
            "background_images_native": [np.zeros((4, 5), dtype=np.float64)],
            "background_images_display": [np.zeros((4, 5), dtype=np.float64)],
            "background_image": np.zeros((4, 5), dtype=np.float64),
            "background_display": np.zeros((4, 5), dtype=np.float64),
        },
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_background_theta,
        "current_geometry_fit_background_indices",
        lambda **kwargs: [0],
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_background_theta,
        "geometry_fit_uses_shared_theta_offset",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_background_theta,
        "current_geometry_theta_offset",
        lambda **kwargs: 0.0,
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_background_theta,
        "current_background_theta_values",
        lambda **kwargs: [1.5],
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_background_theta,
        "background_theta_for_index",
        lambda index, **kwargs: 1.5,
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_background_theta,
        "apply_background_theta_metadata",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_background_theta,
        "apply_geometry_fit_background_selection",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_geometry_fit,
        "build_runtime_geometry_fit_value_callbacks",
        lambda bindings: SimpleNamespace(
            current_var_names=lambda: ["gamma"],
            current_params=lambda: {
                "a": 4.0,
                "c": 10.0,
                "lambda": 1.54,
                "theta_initial": 1.5,
                "theta_offset": 0.0,
                "corto_detector": 0.075,
                "gamma": 0.0,
                "Gamma": 0.0,
                "chi": 0.0,
                "cor_angle": 0.0,
                "zb": 0.0,
                "zs": 0.0,
                "center": [32.0, 32.0],
            },
            current_ui_params=lambda: {},
            var_map={},
        ),
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_geometry_q_group_manager,
        "make_runtime_geometry_fit_simulation_callbacks",
        lambda **kwargs: SimpleNamespace(
            simulate_hit_tables=lambda *args, **kwargs: [],
            last_simulation_diagnostics=lambda: {},
        ),
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_manual_geometry,
        "make_runtime_geometry_manual_projection_callbacks",
        lambda **kwargs: SimpleNamespace(
            simulated_peaks_for_params=lambda params, *, prefer_cache: [],
            simulated_lookup=lambda simulated_peaks: {},
            entry_display_coords=lambda entry: (
                float(entry.get("x", 0.0)),
                float(entry.get("y", 0.0)),
            ),
            refresh_entry_geometry=lambda entry: dict(entry),
        ),
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_geometry_fit,
        "rebuild_geometry_fit_source_rows",
        lambda **kwargs: geometry_fit.GeometryFitSourceRowRebuildResult(
            background_index=0,
            requested_signature=("sig",),
            requested_signature_summary="sig",
            projected_rows=[dict(row) for row in simulated_rows],
            stored_rows=[dict(row) for row in simulated_rows],
            rebuild_source="source_snapshot_rebuild",
            rebuild_attempts=["source_snapshot"],
            diagnostics={"status": "rebuild_ok"},
            source_reflection_indices=[
                int(row["source_reflection_index"]) for row in simulated_rows
            ],
            metadata={},
        ),
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_geometry_overlay,
        "select_fit_orientation",
        lambda sim_pts, meas_pts, shape, *, cfg: (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "yx",
                "label": "identity",
            },
            {"pairs": len(sim_pts)},
        ),
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_geometry_overlay,
        "apply_orientation_to_entries",
        lambda entries, shape, **kwargs: [dict(entry) for entry in entries],
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_geometry_overlay,
        "orient_image_for_fit",
        lambda image, **kwargs: image,
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_geometry_overlay,
        "unrotate_display_peaks",
        lambda entries, rotated_shape, *, k=None, default_display_rotate_k=0: [
            dict(entry) for entry in entries
        ],
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_geometry_overlay,
        "display_to_native_sim_coords",
        lambda col, row, image_shape, sim_display_rotate_k=0: (float(col), float(row)),
    )
    monkeypatch.setattr(
        headless_geometry_fit.gui_geometry_overlay,
        "native_sim_to_display_coords",
        lambda col, row, image_shape, sim_display_rotate_k=0: (float(col), float(row)),
    )
    monkeypatch.setattr(headless_geometry_fit, "get_dir", lambda name: str(tmp_path))
    monkeypatch.setattr(
        headless_geometry_fit.gui_geometry_fit,
        "build_runtime_geometry_fit_execution_setup",
        lambda **kwargs: SimpleNamespace(),
    )

    def _fake_execute_runtime_geometry_fit(*, prepared_run, **kwargs):
        captured["dataset"] = prepared_run.current_dataset
        captured["geometry_runtime_cfg"] = prepared_run.geometry_runtime_cfg
        return geometry_fit.GeometryFitRuntimeExecutionResult(
            log_path=tmp_path / "headless_geometry_fit.log",
            apply_result=geometry_fit.GeometryFitRuntimeApplyResult(
                accepted=True,
                rejection_reason=None,
                rms=0.0,
                fitted_params=None,
                postprocess=None,
            ),
        )

    monkeypatch.setattr(
        headless_geometry_fit.gui_geometry_fit,
        "execute_runtime_geometry_fit",
        _fake_execute_runtime_geometry_fit,
    )

    saved_state = {
        "files": {
            "background_files": ["C:/tmp/bg0.osc"],
            "current_background_index": 0,
        },
        "geometry": {
            "manual_pairs": [
                {"background_index": 0, "entries": [dict(entry) for entry in saved_entries]}
            ],
            "peak_records": [],
        },
    }

    result = headless_geometry_fit.run_headless_geometry_fit(
        saved_state,
        state_path=tmp_path / "saved_state.json",
        downloads_dir=tmp_path,
        stamp="headless_test",
    )

    assert result.accepted is True
    assert result.rms_px == 0.0
    headless_dataset = captured["dataset"]
    assert (
        headless_dataset["resolved_source_pair_count"]
        == workflow_dataset["resolved_source_pair_count"]
    )
    for field in (
        "source_reflection_index",
        "source_reflection_namespace",
        "source_reflection_is_full",
        "source_branch_index",
        "source_peak_index",
        "fit_source_resolution_kind",
    ):
        assert [entry.get(field) for entry in headless_dataset["measured_for_fit"]] == [
            entry.get(field) for entry in workflow_dataset["measured_for_fit"]
        ]
    headless_runtime_cfg = captured["geometry_runtime_cfg"]
    assert bool(headless_runtime_cfg["solver"]["dynamic_point_geometry_fit"]) is True
    assert "manual_point_fit_mode" not in headless_runtime_cfg["solver"]


def test_select_live_candidate_for_saved_entry_rejects_same_branch_pixel_tie() -> None:
    probe = _load_geometry_preflight_probe_module()

    result = probe._select_live_candidate_for_saved_entry(
        saved_entry={
            "hkl": (-1, 0, 5),
            "source_branch_index": 1,
            "source_peak_index": 1,
            "x": 10.0,
            "y": 10.0,
        },
        grouped_candidates={
            ("q_group", "primary", 1, 5): [
                {
                    "hkl": (-1, 0, 5),
                    "source_reflection_index": 203,
                    "source_reflection_namespace": "full_reflection",
                    "source_reflection_is_full": True,
                    "source_branch_index": 1,
                    "source_peak_index": 1,
                    "sim_col_raw": 0.0,
                    "sim_row_raw": 10.0,
                    "sim_col": 0.0,
                    "sim_row": 10.0,
                },
                {
                    "hkl": (-1, 0, 5),
                    "source_reflection_index": 204,
                    "source_reflection_namespace": "full_reflection",
                    "source_reflection_is_full": True,
                    "source_branch_index": 1,
                    "source_peak_index": 1,
                    "sim_col_raw": 20.0,
                    "sim_row_raw": 10.0,
                    "sim_col": 20.0,
                    "sim_row": 10.0,
                },
            ]
        },
        saved_background_current_view_point=(10.0, 10.0),
        saved_background_current_view_frame="current_view_display",
    )

    assert result["ok"] is False
    assert result["selection_status"] == "ambiguous_live_row_selection"
    assert result["selection_coordinate_frame"] == "current_view_px"
    assert len(result["tied_candidate_inventory"]) == 2


def test_select_live_candidate_for_saved_entry_prefers_current_view_near_candidate() -> None:
    probe = _load_geometry_preflight_probe_module()

    result = probe._select_live_candidate_for_saved_entry(
        saved_entry={
            "hkl": (-1, 0, 5),
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_branch_index": 1,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_peak_index": 1,
            "x": 1000.0,
            "y": 1000.0,
            "refined_sim_native_x": 2500.0,
            "refined_sim_native_y": 2500.0,
        },
        grouped_candidates={
            ("q_group", "primary", 1, 5): [
                {
                    "hkl": (-1, 0, 5),
                    "source_reflection_index": 203,
                    "source_reflection_namespace": "full_reflection",
                    "source_reflection_is_full": True,
                    "source_branch_index": 1,
                    "source_peak_index": 1,
                    "native_col": 2504.0,
                    "native_row": 2497.0,
                    "display_col": 2504.0,
                    "display_row": 2497.0,
                    "sim_col": 1003.0,
                    "sim_row": 998.0,
                },
                {
                    "hkl": (-1, 0, 5),
                    "source_reflection_index": 204,
                    "source_reflection_namespace": "full_reflection",
                    "source_reflection_is_full": True,
                    "source_branch_index": 1,
                    "source_peak_index": 1,
                    "native_col": 1003.0,
                    "native_row": 998.0,
                    "display_col": 1003.0,
                    "display_row": 998.0,
                    "sim_col": 2504.0,
                    "sim_row": 2497.0,
                },
            ]
        },
        saved_background_current_view_point=(1000.0, 1000.0),
        saved_background_current_view_frame="current_view_display",
    )

    assert result["ok"] is True
    assert result["selected_candidate"]["source_reflection_index"] == 203
    assert result["selection_coordinate_frame"] == "current_view_px"
    assert result["background_distance_reference_frame"] == "current_view_display"
    assert result["background_distance_px"] == pytest.approx((3.0**2 + 2.0**2) ** 0.5)
    assert result["best_distance_px"] == result["background_distance_px"]
    inventory_by_source = {
        entry["source_reflection_index"]: entry
        for entry in result["matching_branch_candidate_inventory"]
    }
    assert inventory_by_source[203]["coordinate_frame"] == "current_view_px"
    assert inventory_by_source[204]["coordinate_frame"] == "current_view_px"
    assert inventory_by_source[204][
        "distance_to_saved_background_current_view_point_px"
    ] == pytest.approx((1504.0**2 + 1497.0**2) ** 0.5)
    assert inventory_by_source[203][
        "distance_to_saved_background_current_view_point_px"
    ] == pytest.approx((3.0**2 + 2.0**2) ** 0.5)
    assert inventory_by_source[203]["distance_to_saved_sim_detector_hint_px"] == pytest.approx(
        (4.0**2 + 3.0**2) ** 0.5
    )
    assert inventory_by_source[204]["distance_to_saved_sim_detector_hint_px"] == pytest.approx(
        (1497.0**2 + 1502.0**2) ** 0.5
    )


def test_select_live_candidate_for_saved_entry_prefers_live_simulated_current_view_point_over_display_aliases() -> (
    None
):
    probe = _load_geometry_preflight_probe_module()

    result = probe._select_live_candidate_for_saved_entry(
        saved_entry={
            "hkl": (-1, 0, 5),
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_branch_index": 1,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_peak_index": 1,
            "refined_sim_native_x": 2500.0,
            "refined_sim_native_y": 2500.0,
            "caked_x": 29.861040445064752,
            "caked_y": -59.079850372490654,
        },
        grouped_candidates={
            ("q_group", "primary", 1, 5): [
                {
                    "hkl": (-1, 0, 5),
                    "source_reflection_index": 203,
                    "source_reflection_namespace": "full_reflection",
                    "source_reflection_is_full": True,
                    "source_branch_index": 1,
                    "source_peak_index": 1,
                    "native_col": 2504.0,
                    "native_row": 2497.0,
                    "display_col": 29.861040445064752,
                    "display_row": -59.079850372490654,
                    "sim_col": 300.0,
                    "sim_row": -200.0,
                },
                {
                    "hkl": (-1, 0, 5),
                    "source_reflection_index": 204,
                    "source_reflection_namespace": "full_reflection",
                    "source_reflection_is_full": True,
                    "source_branch_index": 1,
                    "source_peak_index": 1,
                    "native_col": 1003.0,
                    "native_row": 998.0,
                    "display_col": 300.0,
                    "display_row": -200.0,
                    "sim_col": 29.861040445064752,
                    "sim_row": -59.079850372490654,
                },
            ]
        },
        saved_background_current_view_point=(29.861040445064752, -59.079850372490654),
        saved_background_current_view_frame="current_view_display",
    )

    assert result["ok"] is True
    assert result["selected_candidate"]["source_reflection_index"] == 204
    assert result["selection_coordinate_frame"] == "current_view_px"
    assert result["background_distance_reference_frame"] == "current_view_display"
    assert result["background_distance_px"] == pytest.approx(0.0)
    inventory_by_source = {
        entry["source_reflection_index"]: entry
        for entry in result["matching_branch_candidate_inventory"]
    }
    assert inventory_by_source[204][
        "distance_to_saved_background_current_view_point_px"
    ] == pytest.approx(0.0)
    assert inventory_by_source[203][
        "distance_to_saved_background_current_view_point_px"
    ] == pytest.approx(
        ((300.0 - 29.861040445064752) ** 2 + (-200.0 + 59.079850372490654) ** 2) ** 0.5
    )
    assert inventory_by_source[204]["distance_to_saved_sim_detector_hint_px"] == pytest.approx(
        (1497.0**2 + 1502.0**2) ** 0.5
    )


def test_select_live_candidate_for_saved_entry_ignores_background_shaped_display_only_candidate() -> (
    None
):
    probe = _load_geometry_preflight_probe_module()

    result = probe._select_live_candidate_for_saved_entry(
        saved_entry={
            "hkl": (-1, 0, 5),
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_branch_index": 1,
            "source_peak_index": 1,
            "x": 1000.0,
            "y": 1000.0,
        },
        grouped_candidates={
            ("q_group", "primary", 1, 5): [
                {
                    "hkl": (-1, 0, 5),
                    "source_reflection_index": 203,
                    "source_reflection_namespace": "full_reflection",
                    "source_reflection_is_full": True,
                    "source_branch_index": 1,
                    "source_peak_index": 1,
                    "x": 1000.0,
                    "y": 1000.0,
                    "display_col": 1000.0,
                    "display_row": 1000.0,
                },
                {
                    "hkl": (-1, 0, 5),
                    "source_reflection_index": 204,
                    "source_reflection_namespace": "full_reflection",
                    "source_reflection_is_full": True,
                    "source_branch_index": 1,
                    "source_peak_index": 1,
                    "sim_col": 1003.0,
                    "sim_row": 998.0,
                },
            ]
        },
        saved_background_current_view_point=(1000.0, 1000.0),
        saved_background_current_view_frame="current_view_display",
    )

    assert result["ok"] is True
    assert result["selected_candidate"]["source_reflection_index"] == 204
    inventory_by_source = {
        entry["source_reflection_index"]: entry
        for entry in result["matching_branch_candidate_inventory"]
    }
    assert inventory_by_source[203]["selected_live_simulated_current_view_frame"] is None
    assert np.isfinite(
        inventory_by_source[204]["distance_to_saved_background_current_view_point_px"]
    )
    assert not np.isfinite(
        inventory_by_source[203]["distance_to_saved_background_current_view_point_px"]
    )


def test_select_live_candidate_for_saved_entry_falls_back_to_detector_native_without_saved_current_view() -> (
    None
):
    probe = _load_geometry_preflight_probe_module()

    result = probe._select_live_candidate_for_saved_entry(
        saved_entry={
            "hkl": (-1, 0, 5),
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_branch_index": 1,
            "source_peak_index": 1,
            "detector_x": 1000.0,
            "detector_y": 1000.0,
            "refined_sim_native_x": 1005.0,
            "refined_sim_native_y": 1010.0,
        },
        grouped_candidates={
            ("q_group", "primary", 1, 5): [
                {
                    "hkl": (-1, 0, 5),
                    "source_reflection_index": 203,
                    "source_reflection_namespace": "full_reflection",
                    "source_reflection_is_full": True,
                    "source_branch_index": 1,
                    "source_peak_index": 1,
                    "native_col": 2510.0,
                    "native_row": 2525.0,
                },
                {
                    "hkl": (-1, 0, 5),
                    "source_reflection_index": 204,
                    "source_reflection_namespace": "full_reflection",
                    "source_reflection_is_full": True,
                    "source_branch_index": 1,
                    "source_peak_index": 1,
                    "native_col": 1008.0,
                    "native_row": 1012.0,
                },
            ]
        },
        saved_background_current_view_point=None,
        saved_background_current_view_frame=None,
    )

    assert result["ok"] is True
    assert result["selected_candidate"]["source_reflection_index"] == 204
    assert result["selection_coordinate_frame"] == "detector_native_px"
    assert result["background_distance_reference_frame"] == "native_detector"
    assert result["best_distance_px"] == pytest.approx((3.0**2 + 2.0**2) ** 0.5)
    inventory_by_source = {
        entry["source_reflection_index"]: entry
        for entry in result["matching_branch_candidate_inventory"]
    }
    assert inventory_by_source[203]["coordinate_frame"] == "detector_native_px"
    assert inventory_by_source[204]["coordinate_frame"] == "detector_native_px"
    assert inventory_by_source[203]["distance_to_saved_sim_detector_hint_px"] == pytest.approx(
        (1505.0**2 + 1515.0**2) ** 0.5
    )
    assert inventory_by_source[204]["distance_to_saved_sim_detector_hint_px"] == pytest.approx(
        (3.0**2 + 2.0**2) ** 0.5
    )
    assert "distance_to_saved_background_current_view_point_px" not in inventory_by_source[204]


def test_run_fresh_slot_validation_prefers_saved_xy_over_raw_xy_and_display_for_current_view_target() -> (
    None
):
    probe = _load_geometry_preflight_probe_module()
    captured: dict[str, object] = {}
    saved_entry = {
        "pair_id": "bg0:pair0",
        "label": "-1,0,5",
        "hkl": (-1, 0, 5),
        "q_group_key": ("q_group", "primary", 1, 5),
        "x": 1822.0,
        "y": 1375.0,
        "raw_x": 291.0,
        "raw_y": 173.0,
        "display_col": 99.0,
        "display_row": 88.0,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }

    def _fake_select_live_candidate_for_saved_entry(**kwargs):
        captured.update(kwargs)
        return {
            "ok": False,
            "failure_stage": "grouped_candidate_regeneration",
            "selection_status": "missing_live_candidate",
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        probe,
        "_select_live_candidate_for_saved_entry",
        _fake_select_live_candidate_for_saved_entry,
    )
    try:
        result = probe._run_fresh_slot_validation(
            context={
                "saved_entries": [dict(saved_entry)],
                "raw_saved_entries": [dict(saved_entry)],
            },
            background_index=0,
            slot_index=0,
            runtime={
                "projection_callbacks": SimpleNamespace(
                    pick_uses_caked_space=lambda: False,
                ),
                "group_cache": {},
                "current_source_rows": [],
                "grouped_candidates": {},
                "grouped_candidate_source": "pick_cache",
            },
        )
    finally:
        monkeypatch.undo()

    assert captured["saved_background_current_view_point"] == pytest.approx((1822.0, 1375.0))
    assert captured["saved_background_current_view_frame"] == "current_view_display"
    assert result["saved_entry"]["saved_background_current_view_point"] == pytest.approx(
        [1822.0, 1375.0]
    )
    assert result["saved_entry"]["saved_background_current_view_frame"] == ("current_view_display")


def test_run_fresh_slot_validation_uses_display_point_as_last_noncaked_current_view_fallback() -> (
    None
):
    probe = _load_geometry_preflight_probe_module()
    captured: dict[str, object] = {}
    saved_entry = {
        "pair_id": "bg0:pair0",
        "label": "-1,0,5",
        "hkl": (-1, 0, 5),
        "q_group_key": ("q_group", "primary", 1, 5),
        "display_col": 91.0,
        "display_row": 82.0,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }

    def _fake_select_live_candidate_for_saved_entry(**kwargs):
        captured.update(kwargs)
        return {
            "ok": False,
            "failure_stage": "grouped_candidate_regeneration",
            "selection_status": "missing_live_candidate",
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        probe,
        "_select_live_candidate_for_saved_entry",
        _fake_select_live_candidate_for_saved_entry,
    )
    try:
        result = probe._run_fresh_slot_validation(
            context={
                "saved_entries": [dict(saved_entry)],
                "raw_saved_entries": [dict(saved_entry)],
            },
            background_index=0,
            slot_index=0,
            runtime={
                "projection_callbacks": SimpleNamespace(
                    pick_uses_caked_space=lambda: False,
                ),
                "group_cache": {},
                "current_source_rows": [],
                "grouped_candidates": {},
                "grouped_candidate_source": "pick_cache",
            },
        )
    finally:
        monkeypatch.undo()

    assert captured["saved_background_current_view_point"] == pytest.approx((91.0, 82.0))
    assert captured["saved_background_current_view_frame"] == "current_view_display"
    assert result["saved_entry"]["saved_background_current_view_point"] == pytest.approx(
        [91.0, 82.0]
    )
    assert result["saved_entry"]["saved_background_current_view_frame"] == ("current_view_display")


def test_run_fresh_slot_validation_passes_caked_saved_current_view_point_to_selection() -> None:
    probe = _load_geometry_preflight_probe_module()
    captured: dict[str, object] = {}
    saved_entry = {
        "pair_id": "bg0:pair0",
        "label": "-1,0,5",
        "hkl": (-1, 0, 5),
        "q_group_key": ("q_group", "primary", 1, 5),
        "x": 1822.0,
        "y": 1375.0,
        "caked_x": 29.861040445064752,
        "caked_y": -59.079850372490654,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }

    def _fake_select_live_candidate_for_saved_entry(**kwargs):
        captured.update(kwargs)
        return {
            "ok": False,
            "failure_stage": "grouped_candidate_regeneration",
            "selection_status": "missing_live_candidate",
        }

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        probe,
        "_select_live_candidate_for_saved_entry",
        _fake_select_live_candidate_for_saved_entry,
    )
    try:
        result = probe._run_fresh_slot_validation(
            context={
                "saved_entries": [dict(saved_entry)],
                "raw_saved_entries": [dict(saved_entry)],
            },
            background_index=0,
            slot_index=0,
            runtime={
                "projection_callbacks": SimpleNamespace(
                    pick_uses_caked_space=lambda: True,
                ),
                "group_cache": {},
                "current_source_rows": [],
                "grouped_candidates": {},
                "grouped_candidate_source": "pick_cache",
            },
        )
    finally:
        monkeypatch.undo()

    assert captured["saved_background_current_view_point"] == pytest.approx(
        (29.861040445064752, -59.079850372490654)
    )
    assert captured["saved_background_current_view_frame"] == "caked_display"
    assert result["saved_entry"]["saved_background_current_view_point"] == pytest.approx(
        [29.861040445064752, -59.079850372490654]
    )
    assert result["saved_entry"]["saved_background_current_view_frame"] == ("caked_display")


def test_compatibility_probe_slot_indices_prefers_first_mirrored_group() -> None:
    probe = _load_geometry_preflight_probe_module()

    result = probe._compatibility_probe_slot_indices(
        [
            {
                "q_group_key": ("q_group", "primary", 0, 1),
                "hkl": (0, 0, 1),
            },
            {
                "q_group_key": ("q_group", "primary", 1, 5),
                "hkl": (-1, 0, 5),
            },
            {
                "q_group_key": ("q_group", "primary", 1, 5),
                "hkl": (-1, 0, 5),
            },
        ]
    )

    assert result == [1, 2]


def test_saved_to_selected_identity_delta_reports_legacy_canonicalization() -> None:
    probe = _load_geometry_preflight_probe_module()

    saved_entry = {
        "pair_id": "bg0:pair1",
        "hkl": (-1, 0, 5),
        "source_reflection_index": None,
        "source_reflection_namespace": None,
        "source_reflection_is_full": False,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }
    selected_entry = {
        "pair_id": "selected-live-row",
        "hkl": (-1, 0, 5),
        "source_reflection_index": 214,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }

    delta, classification = probe._classify_saved_to_selected_identity_delta(
        saved_entry,
        selected_entry,
    )

    assert classification == "legacy_saved_identity_canonicalized"
    assert delta == {
        "source_reflection_index": {
            "saved": None,
            "selected": 214,
        },
        "source_reflection_namespace": {
            "saved": None,
            "selected": "full_reflection",
        },
        "source_reflection_is_full": {
            "saved": False,
            "selected": True,
        },
    }


def test_saved_to_selected_identity_delta_rejects_conflicting_trusted_reflection_id() -> None:
    probe = _load_geometry_preflight_probe_module()

    saved_entry = {
        "pair_id": "bg0:pair1",
        "hkl": (-1, 0, 5),
        "source_reflection_index": 15,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }
    selected_entry = {
        "pair_id": "selected-live-row",
        "hkl": (-1, 0, 5),
        "source_reflection_index": 214,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }

    delta, classification = probe._classify_saved_to_selected_identity_delta(
        saved_entry,
        selected_entry,
    )

    assert classification == "identity_drift"
    assert delta == {
        "source_reflection_index": {
            "saved": 15,
            "selected": 214,
        }
    }


def test_saved_to_selected_identity_delta_accepts_legacy_table_index_alias() -> None:
    probe = _load_geometry_preflight_probe_module()

    saved_entry = {
        "pair_id": "bg0:pair0",
        "hkl": (0, 0, 3),
        "source_reflection_index": 10,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_table_index": 10,
        "source_row_index": 24,
        "source_branch_index": 0,
        "source_peak_index": 0,
    }
    selected_entry = {
        "pair_id": "selected-live-row",
        "hkl": (0, 0, 3),
        "source_reflection_index": 411,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_table_index": 411,
        "source_row_index": 0,
        "source_branch_index": 0,
        "source_peak_index": 0,
    }

    delta, classification = probe._classify_saved_to_selected_identity_delta(
        saved_entry,
        selected_entry,
    )

    assert classification == "legacy_saved_identity_canonicalized"
    assert delta == {
        "source_reflection_index": {
            "saved": 10,
            "selected": 411,
        }
    }


def test_saved_to_selected_identity_delta_rejects_alias_without_row_provenance() -> None:
    probe = _load_geometry_preflight_probe_module()

    saved_entry = {
        "pair_id": "bg0:pair0",
        "hkl": (0, 0, 3),
        "source_reflection_index": 10,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_table_index": 10,
        "source_row_index": 24,
        "source_branch_index": 0,
        "source_peak_index": 0,
    }
    selected_entry = {
        "pair_id": "selected-live-row",
        "hkl": (0, 0, 3),
        "source_reflection_index": 411,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "source_peak_index": 0,
    }

    delta, classification = probe._classify_saved_to_selected_identity_delta(
        saved_entry,
        selected_entry,
    )

    assert classification == "identity_drift"
    assert delta == {
        "source_reflection_index": {
            "saved": 10,
            "selected": 411,
        }
    }


def test_probe_main_aliases_full_to_fresh_all(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    probe = _load_geometry_preflight_probe_module()
    calls: list[str] = []

    monkeypatch.setattr(
        probe.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            state=str(tmp_path / "dummy.json"),
            background_index=0,
            mode="full",
            sentinel_slot_index=1,
            export_fresh_state=None,
        ),
    )
    monkeypatch.setattr(
        probe,
        "_run_fresh_all_contract_validation",
        lambda *args, **kwargs: calls.append("fresh-all") or {"ok": True, "classification": "pass"},
    )
    monkeypatch.setattr(
        probe,
        "_run_fresh_contract_validation",
        lambda *args, **kwargs: pytest.fail("fresh sentinel path should not run"),
    )
    monkeypatch.setattr(
        probe,
        "_run_saved_state_compatibility_validation",
        lambda *args, **kwargs: pytest.fail("compatibility path should not run"),
    )

    exit_code = probe.main()
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert calls == ["fresh-all"]
    assert payload["requested_mode"] == "full"
    assert payload["effective_mode"] == "fresh-all"
    assert "aliases fresh-all" in payload["mode_note"]


def test_probe_main_writes_report_file_and_prints_summary_when_report_path_present(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    probe = _load_geometry_preflight_probe_module()
    report_path = tmp_path / "report.json"

    monkeypatch.setattr(
        probe.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            state=str(tmp_path / "dummy.json"),
            background_index=0,
            mode="full",
            sentinel_slot_index=1,
            export_fresh_state=None,
            report_path=str(report_path),
        ),
    )
    monkeypatch.setattr(
        probe,
        "_run_fresh_all_contract_validation",
        lambda *args, **kwargs: {
            "ok": True,
            "classification": "pass",
            "background_index": 0,
            "processed_manual_entry_count": 7,
            "bound_manual_entry_count": 7,
            "resolved_source_pair_count": 7,
            "targeted_performance_gate": {"ok": True},
            "slot_results": [{"pair_id": "bg0:pair0"}],
        },
    )
    monkeypatch.setattr(
        probe,
        "_run_fresh_contract_validation",
        lambda *args, **kwargs: pytest.fail("fresh sentinel path should not run"),
    )
    monkeypatch.setattr(
        probe,
        "_run_saved_state_compatibility_validation",
        lambda *args, **kwargs: pytest.fail("compatibility path should not run"),
    )

    exit_code = probe.main()
    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert stdout_payload == {
        "background_index": 0,
        "bound_manual_entry_count": 7,
        "classification": "pass",
        "ok": True,
        "processed_manual_entry_count": 7,
        "report_path": str(report_path.resolve()),
        "resolved_source_pair_count": 7,
        "targeted_performance_gate_ok": True,
    }
    assert file_payload["requested_mode"] == "full"
    assert file_payload["effective_mode"] == "fresh-all"
    assert file_payload["slot_results"] == [{"pair_id": "bg0:pair0"}]


def test_build_group_cache_source_rows_callback_accepts_manual_pick_cache_consumer(
    monkeypatch,
) -> None:
    probe = _load_geometry_preflight_probe_module()
    required_pairs = [{"pair_id": "bg0:pair0", "hkl": (1, 0, 0)}]
    captured: dict[str, object] = {}

    def _fake_build_geometry_manual_pick_cache(**kwargs):
        captured["rows"] = kwargs["source_rows_for_background"](
            0,
            {"theta_initial": 3.0},
            consumer="manual_pick_cache",
        )
        return {"grouped_candidates": {}}, None, None

    def _fake_source_rows(
        background_index,
        param_set=None,
        *,
        consumer=None,
        required_pairs=None,
    ):
        captured["background_index"] = background_index
        captured["param_set"] = dict(param_set or {})
        captured["consumer"] = consumer
        captured["required_pairs"] = required_pairs
        return [{"pair_id": "live"}]

    monkeypatch.setattr(
        probe.gui_manual_geometry,
        "build_geometry_manual_pick_cache",
        _fake_build_geometry_manual_pick_cache,
    )

    cache_data = probe._build_group_cache(
        background_index=0,
        params={"theta_initial": 3.0},
        dataset={"simulation_diagnostics": {"requested_signature": ("sig", 0)}},
        manual_dataset_bindings=SimpleNamespace(
            load_background_by_index=lambda _idx: (
                np.zeros((2, 2)),
                np.zeros((2, 2)),
            ),
            current_background_index=0,
            geometry_manual_source_rows_for_background=_fake_source_rows,
            geometry_manual_match_config=lambda: {},
        ),
        projection_callbacks=SimpleNamespace(
            simulated_peaks_for_params=lambda *args, **kwargs: [],
            pick_candidates=lambda rows: {},
            simulated_lookup=lambda peaks: {},
        ),
        required_pairs=required_pairs,
    )

    assert cache_data == {"grouped_candidates": {}}
    assert captured["rows"] == [{"pair_id": "live"}]
    assert captured["background_index"] == 0
    assert captured["param_set"] == {"theta_initial": 3.0}
    assert captured["consumer"] == "manual_pick_cache"
    assert captured["required_pairs"] == required_pairs


def test_current_source_rows_for_background_forwards_saved_entries_as_required_pairs() -> None:
    probe = _load_geometry_preflight_probe_module()
    saved_entries = [{"pair_id": "bg0:pair0", "hkl": (1, 0, 0)}]
    captured: dict[str, object] = {}

    def _fake_source_rows(
        background_index,
        param_set=None,
        *,
        consumer=None,
        required_pairs=None,
    ):
        captured["background_index"] = background_index
        captured["param_set"] = dict(param_set or {})
        captured["consumer"] = consumer
        captured["required_pairs"] = required_pairs
        return [{"pair_id": "live"}]

    rows = probe._current_source_rows_for_background(
        background_index=0,
        context={
            "manual_dataset_bindings": SimpleNamespace(
                geometry_manual_source_rows_for_background=_fake_source_rows
            ),
            "params": {"theta_initial": 3.0},
            "saved_entries": saved_entries,
            "dataset": {},
        },
        consumer="manual_pick_group_probe",
    )

    assert rows == [{"pair_id": "live"}]
    assert captured["background_index"] == 0
    assert captured["param_set"] == {"theta_initial": 3.0}
    assert captured["consumer"] == "manual_pick_group_probe"
    assert captured["required_pairs"] == saved_entries


def test_build_group_cache_source_rows_callback_tolerates_provider_without_required_pairs(
    monkeypatch,
) -> None:
    probe = _load_geometry_preflight_probe_module()
    captured: dict[str, object] = {}

    def _fake_build_geometry_manual_pick_cache(**kwargs):
        captured["rows"] = kwargs["source_rows_for_background"](
            0,
            {"theta_initial": 3.0},
            consumer="manual_pick_cache",
            required_pairs=[{"pair_id": "bg0:pair0"}],
        )
        return {"grouped_candidates": {}}, None, None

    def _fake_source_rows(
        background_index,
        param_set=None,
        *,
        consumer=None,
    ):
        captured["background_index"] = background_index
        captured["param_set"] = dict(param_set or {})
        captured["consumer"] = consumer
        return [{"pair_id": "live"}]

    monkeypatch.setattr(
        probe.gui_manual_geometry,
        "build_geometry_manual_pick_cache",
        _fake_build_geometry_manual_pick_cache,
    )

    cache_data = probe._build_group_cache(
        background_index=0,
        params={"theta_initial": 3.0},
        dataset={"simulation_diagnostics": {"requested_signature": ("sig", 0)}},
        manual_dataset_bindings=SimpleNamespace(
            load_background_by_index=lambda _idx: (
                np.zeros((2, 2)),
                np.zeros((2, 2)),
            ),
            current_background_index=0,
            geometry_manual_source_rows_for_background=_fake_source_rows,
            geometry_manual_match_config=lambda: {},
        ),
        projection_callbacks=SimpleNamespace(
            simulated_peaks_for_params=lambda *args, **kwargs: [],
            pick_candidates=lambda rows: {},
            simulated_lookup=lambda peaks: {},
        ),
        required_pairs=[{"pair_id": "bg0:pair0", "hkl": (1, 0, 0)}],
    )

    assert cache_data == {"grouped_candidates": {}}
    assert captured["rows"] == [{"pair_id": "live"}]
    assert captured["background_index"] == 0
    assert captured["param_set"] == {"theta_initial": 3.0}
    assert captured["consumer"] == "manual_pick_cache"


def test_current_source_rows_for_background_tolerates_provider_without_required_pairs() -> None:
    probe = _load_geometry_preflight_probe_module()
    captured: dict[str, object] = {}

    def _fake_source_rows(
        background_index,
        param_set=None,
        *,
        consumer=None,
    ):
        captured["background_index"] = background_index
        captured["param_set"] = dict(param_set or {})
        captured["consumer"] = consumer
        return [{"pair_id": "live"}]

    rows = probe._current_source_rows_for_background(
        background_index=0,
        context={
            "manual_dataset_bindings": SimpleNamespace(
                geometry_manual_source_rows_for_background=_fake_source_rows
            ),
            "params": {"theta_initial": 3.0},
            "saved_entries": [{"pair_id": "bg0:pair0", "hkl": (1, 0, 0)}],
            "dataset": {},
        },
        consumer="manual_pick_group_probe",
    )

    assert rows == [{"pair_id": "live"}]
    assert captured["background_index"] == 0
    assert captured["param_set"] == {"theta_initial": 3.0}
    assert captured["consumer"] == "manual_pick_group_probe"


def test_saved_state_compatibility_validation_handles_two_entry_pair(
    monkeypatch,
    tmp_path,
) -> None:
    probe = _load_geometry_preflight_probe_module()
    saved_entries = [
        {
            "q_group_key": ("q_group", "primary", 1, 5),
            "hkl": (-1, 0, 5),
        },
        {
            "q_group_key": ("q_group", "primary", 1, 5),
            "hkl": (-1, 0, 5),
        },
    ]
    context = {
        "ok": True,
        "state_path": str(tmp_path / "dummy.json"),
        "background_index": 0,
        "saved_pair_count": 2,
        "captured_preflight_error_text": None,
        "used_isolated_background_dataset": False,
        "dataset_pair_count": 2,
        "dataset_resolved_source_pair_count": 2,
        "harness_validation": {"valid": True},
        "saved_entries": saved_entries,
        "dataset": {},
        "group_cache": {},
        "manual_dataset_bindings": SimpleNamespace(
            geometry_manual_entry_display_coords=lambda entry: (0.0, 0.0)
        ),
    }
    calls: list[int] = []

    monkeypatch.setattr(
        probe,
        "_prepare_validation_context",
        lambda state_path, background_index: context,
    )

    def _fake_validate_pair(
        *,
        background_index,
        slot_index,
        expected_pair_id,
        saved_entries,
        dataset,
        group_cache,
        entry_display_coords,
    ):
        calls.append(int(slot_index))
        identity_entry = {
            "hkl": (-1, 0, 5),
            "source_reflection_index": 200 + int(slot_index),
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": int(slot_index),
            "source_peak_index": int(slot_index),
        }
        return {
            "ok": True,
            "expected_pair_id": expected_pair_id,
            "chosen_resolved_live_row": dict(identity_entry),
            "emitted_preflight_normalized_pair": {
                **identity_entry,
                "pair_id": expected_pair_id,
            },
            "fit_resolution_kind": "source_row",
            "overlay_resolution_kind": "source_row",
        }

    monkeypatch.setattr(probe, "_validate_pair", _fake_validate_pair)

    result = probe._run_saved_state_compatibility_validation(tmp_path / "dummy.json", 0)

    assert result["ok"] is True
    assert result["classification"] == "pass"
    assert result["checked_slot_indices"] == [0, 1]
    assert calls == [0, 1, 0, 1]


def test_fresh_all_contract_validation_exports_slot_order_and_runs_compatibility(
    monkeypatch,
    tmp_path,
) -> None:
    probe = _load_geometry_preflight_probe_module()
    saved_entries = [
        {
            "pair_id": f"bg0:pair{slot_index}",
            "q_group_key": ("q_group", "primary", 1, slot_index + 1),
            "hkl": (-1, 0, slot_index + 1),
            "source_reflection_index": 10 + slot_index,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": slot_index % 2,
            "source_peak_index": slot_index % 2,
        }
        for slot_index in range(9)
    ]
    state_path = tmp_path / "geometry_probe_state.json"
    context = {
        "ok": True,
        "state_path": str(state_path),
        "background_index": 0,
        "saved_pair_count": len(saved_entries),
        "captured_preflight_error_text": None,
        "used_isolated_background_dataset": False,
        "runtime_prepare_ok": True,
        "dataset_pair_count": len(saved_entries),
        "dataset_resolved_source_pair_count": len(saved_entries),
        "harness_validation": {"valid": True},
        "saved_entries": saved_entries,
        "saved_state": {
            "files": {
                "background_files": ["C:/tmp/bg0.osc"],
                "current_background_index": 0,
            },
            "variables": {
                "geometry_fit_background_selection_var": "current",
            },
            "geometry": {
                "manual_pairs": [
                    {
                        "background_index": 0,
                        "background_name": "bg0",
                        "background_path": "C:/tmp/bg0.osc",
                        "entries": [dict(entry) for entry in saved_entries],
                    }
                ],
                "peak_records": ["stale"],
                "q_group_rows": ["stale"],
            },
        },
        "dataset": {},
        "group_cache": {},
        "manual_dataset_bindings": SimpleNamespace(
            geometry_manual_entry_display_coords=lambda entry: (0.0, 0.0)
        ),
    }

    monkeypatch.setattr(
        probe,
        "_prepare_validation_context",
        lambda state_path, background_index: context,
    )
    monkeypatch.setattr(
        probe,
        "_prepare_fresh_slot_runtime",
        lambda **kwargs: {"prepared": True},
    )

    def _fake_run_fresh_slot_validation(*, context, background_index, slot_index, runtime):
        selected_candidate = {
            "pair_id": f"bg0:pair{int(slot_index)}",
            "q_group_key": ("q_group", "primary", 1, int(slot_index) + 1),
            "hkl": (-1, 0, int(slot_index) + 1),
            "source_reflection_index": 200 + int(slot_index),
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": int(slot_index) % 2,
            "source_peak_index": int(slot_index) % 2,
        }
        emitted_pair = {
            **selected_candidate,
        }
        return {
            "ok": True,
            "rebind_ok": True,
            "classification": "pass",
            "slot_index": int(slot_index),
            "saved_entry": {
                "pair_id": f"bg0:pair{int(slot_index)}",
                "hkl": (-1, 0, int(slot_index) + 1),
                "source_branch_index": int(slot_index) % 2,
                "source_peak_index": int(slot_index) % 2,
            },
            "candidate_selection": {
                "ok": True,
                "selection_status": "selected",
                "background_distance_px": float(slot_index) + 0.25,
                "best_distance_px": float(slot_index) + 0.25,
                "selected_candidate": dict(selected_candidate),
            },
            "saved_to_selected_identity_delta": (
                {
                    "source_reflection_index": {
                        "saved": 10 + int(slot_index),
                        "selected": 200 + int(slot_index),
                    }
                }
                if int(slot_index) == 0
                else {}
            ),
            "saved_to_selected_identity_delta_classification": (
                "legacy_saved_identity_canonicalized"
                if int(slot_index) == 0
                else "saved_identity_already_canonical"
            ),
            "emitted_pair": emitted_pair,
        }

    monkeypatch.setattr(
        probe,
        "_run_fresh_slot_validation",
        _fake_run_fresh_slot_validation,
    )

    def _fake_compatibility(state_path, background_index):
        exported_state = load_gui_state_file(state_path)["state"]
        manual_pairs = exported_state["geometry"]["manual_pairs"]
        assert len(manual_pairs) == 1
        assert [entry["pair_id"] for entry in manual_pairs[0]["entries"]] == [
            f"bg0:pair{slot_index}" for slot_index in range(9)
        ]
        assert exported_state["geometry"]["peak_records"] == []
        assert exported_state["geometry"]["q_group_rows"] == []
        return {
            "ok": True,
            "classification": "pass",
            "checked_slot_indices": list(range(9)),
        }

    monkeypatch.setattr(
        probe,
        "_run_saved_state_compatibility_validation",
        _fake_compatibility,
    )

    export_path = tmp_path / "fresh_all_export.json"
    result = probe._run_fresh_all_contract_validation(
        state_path,
        background_index=0,
        export_fresh_state_path=export_path,
    )

    assert result["ok"] is True
    assert result["classification"] == "pass"
    assert len(result["slot_results"]) == 9
    assert result["processed_manual_entry_count"] == 9
    assert result["bound_manual_entry_count"] == 9
    assert result["missing_manual_entry_count"] == 0
    assert result["branch_mismatch_count"] == 0
    assert result["runtime_prepare_ok"] is True
    assert result["isolated_rebind_ok"] is True
    assert result["fresh_export_ok"] is True
    assert result["compatibility_ok"] is True
    assert result["background_distance_gate_ok"] is True
    assert result["background_distances_all_finite"] is True
    assert result["background_distance_px"] == [float(slot_index) + 0.25 for slot_index in range(9)]
    assert result["max_background_distance_px"] == pytest.approx(8.25)
    assert result["background_distance_gate_threshold_px"] == pytest.approx(100.0)
    assert result["deprecated_aliases_present"] is True
    assert result["detector_distance_gate_ok"] is True
    assert result["candidate_distance_gate_ok"] is True
    assert result["candidate_distances_all_finite"] is True
    assert result["candidate_distance_px"] == result["background_distance_px"]
    assert result["max_candidate_distance_px"] == result["max_background_distance_px"]
    assert (
        result["candidate_distance_gate_threshold_px"]
        == result["background_distance_gate_threshold_px"]
    )
    assert result["slot_results"][0]["saved_to_selected_identity_delta_classification"] == (
        "legacy_saved_identity_canonicalized"
    )
    assert result["slot_results"][1]["saved_to_selected_identity_delta_classification"] == (
        "saved_identity_already_canonical"
    )
    assert result["exported_fresh_state_path"] == str(export_path.resolve())
    assert result["fresh_state_export_written"] is True
    assert result["exported_state_compatibility"]["ok"] is True


def test_fresh_contract_validation_validates_temp_state_before_export(
    monkeypatch,
    tmp_path,
) -> None:
    probe = _load_geometry_preflight_probe_module()
    saved_entry = {
        "pair_id": "bg0:pair0",
        "q_group_key": ("q_group", "primary", 1, 1),
        "hkl": (-1, 0, 1),
        "source_reflection_index": 10,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 0,
        "source_peak_index": 0,
    }
    state_path = tmp_path / "geometry_probe_state.json"
    context = {
        "ok": True,
        "state_path": str(state_path),
        "background_index": 0,
        "saved_pair_count": 1,
        "captured_preflight_error_text": None,
        "used_isolated_background_dataset": False,
        "runtime_prepare_ok": True,
        "dataset_pair_count": 1,
        "dataset_resolved_source_pair_count": 1,
        "harness_validation": {"valid": True},
        "saved_entries": [saved_entry],
        "saved_state": {
            "files": {
                "background_files": ["C:/tmp/bg0.osc"],
                "current_background_index": 0,
            },
            "variables": {
                "geometry_fit_background_selection_var": "current",
            },
            "geometry": {
                "manual_pairs": [
                    {
                        "background_index": 0,
                        "background_name": "bg0",
                        "background_path": "C:/tmp/bg0.osc",
                        "entries": [dict(saved_entry)],
                    }
                ],
                "peak_records": ["stale"],
                "q_group_rows": ["stale"],
            },
        },
        "dataset": {},
        "group_cache": {},
        "manual_dataset_bindings": SimpleNamespace(
            geometry_manual_entry_display_coords=lambda entry: (0.0, 0.0)
        ),
    }

    monkeypatch.setattr(
        probe,
        "_prepare_validation_context",
        lambda state_path, background_index: context,
    )
    monkeypatch.setattr(
        probe,
        "_prepare_fresh_slot_runtime",
        lambda **kwargs: {"prepared": True},
    )
    monkeypatch.setattr(
        probe,
        "_run_fresh_slot_validation",
        lambda **kwargs: {
            "ok": True,
            "rebind_ok": True,
            "classification": "pass",
            "slot_index": int(kwargs["slot_index"]),
            "saved_entry": dict(saved_entry),
            "candidate_selection": {
                "ok": True,
                "selection_status": "selected",
                "background_distance_px": 1.25,
                "best_distance_px": 1.25,
                "selected_candidate": {
                    "pair_id": "bg0:pair0",
                    "q_group_key": ("q_group", "primary", 1, 1),
                    "hkl": (-1, 0, 1),
                    "source_branch_index": 0,
                    "source_peak_index": 0,
                },
            },
            "saved_to_selected_identity_delta": {},
            "saved_to_selected_identity_delta_classification": "saved_identity_already_canonical",
            "emitted_pair": {
                "pair_id": "bg0:pair0",
                "q_group_key": ("q_group", "primary", 1, 1),
                "hkl": (-1, 0, 1),
                "source_branch_index": 0,
                "source_peak_index": 0,
            },
        },
    )

    compatibility_paths: list[Path] = []

    def _fake_compatibility(checked_state_path, background_index):
        checked_path = Path(checked_state_path).resolve()
        compatibility_paths.append(checked_path)
        exported_state = load_gui_state_file(checked_path)["state"]
        manual_pairs = exported_state["geometry"]["manual_pairs"]
        assert len(manual_pairs) == 1
        assert [entry["pair_id"] for entry in manual_pairs[0]["entries"]] == ["bg0:pair0"]
        assert exported_state["geometry"]["peak_records"] == []
        assert exported_state["geometry"]["q_group_rows"] == []
        return {
            "ok": True,
            "classification": "pass",
            "checked_slot_indices": [0],
        }

    monkeypatch.setattr(
        probe,
        "_run_saved_state_compatibility_validation",
        _fake_compatibility,
    )

    export_path = tmp_path / "fresh_export.json"
    result = probe._run_fresh_contract_validation(
        state_path,
        background_index=0,
        sentinel_slot_index=0,
        export_fresh_state_path=export_path,
    )

    assert len(compatibility_paths) == 1
    assert compatibility_paths[0] != state_path.resolve()
    assert compatibility_paths[0] != export_path.resolve()
    assert result["ok"] is True
    assert result["classification"] == "pass"
    assert result["compatibility_ok"] is True
    assert result["fresh_export_ok"] is True
    assert result["fresh_state_export_written"] is True
    assert result["exported_fresh_state_path"] == str(export_path.resolve())
    assert result["exported_state_compatibility"]["ok"] is True
    assert export_path.exists() is True


def test_fresh_contract_validation_does_not_export_when_temp_state_fails_compatibility(
    monkeypatch,
    tmp_path,
) -> None:
    probe = _load_geometry_preflight_probe_module()
    saved_entry = {
        "pair_id": "bg0:pair0",
        "q_group_key": ("q_group", "primary", 1, 1),
        "hkl": (-1, 0, 1),
        "source_branch_index": 0,
        "source_peak_index": 0,
    }
    state_path = tmp_path / "geometry_probe_state.json"
    context = {
        "ok": True,
        "state_path": str(state_path),
        "background_index": 0,
        "saved_pair_count": 1,
        "captured_preflight_error_text": None,
        "used_isolated_background_dataset": False,
        "runtime_prepare_ok": True,
        "dataset_pair_count": 1,
        "dataset_resolved_source_pair_count": 1,
        "harness_validation": {"valid": True},
        "saved_entries": [saved_entry],
        "saved_state": {
            "files": {
                "background_files": ["C:/tmp/bg0.osc"],
                "current_background_index": 0,
            },
            "variables": {
                "geometry_fit_background_selection_var": "current",
            },
            "geometry": {
                "manual_pairs": [
                    {
                        "background_index": 0,
                        "background_name": "bg0",
                        "background_path": "C:/tmp/bg0.osc",
                        "entries": [dict(saved_entry)],
                    }
                ],
                "peak_records": ["stale"],
                "q_group_rows": ["stale"],
            },
        },
        "dataset": {},
        "group_cache": {},
        "manual_dataset_bindings": SimpleNamespace(
            geometry_manual_entry_display_coords=lambda entry: (0.0, 0.0)
        ),
    }

    monkeypatch.setattr(
        probe,
        "_prepare_validation_context",
        lambda state_path, background_index: context,
    )
    monkeypatch.setattr(
        probe,
        "_prepare_fresh_slot_runtime",
        lambda **kwargs: {"prepared": True},
    )
    monkeypatch.setattr(
        probe,
        "_run_fresh_slot_validation",
        lambda **kwargs: {
            "ok": True,
            "rebind_ok": True,
            "classification": "pass",
            "slot_index": int(kwargs["slot_index"]),
            "saved_entry": dict(saved_entry),
            "candidate_selection": {
                "ok": True,
                "selection_status": "selected",
                "background_distance_px": 1.25,
                "best_distance_px": 1.25,
                "selected_candidate": {
                    "pair_id": "bg0:pair0",
                    "q_group_key": ("q_group", "primary", 1, 1),
                    "hkl": (-1, 0, 1),
                    "source_branch_index": 0,
                    "source_peak_index": 0,
                },
            },
            "saved_to_selected_identity_delta": {},
            "saved_to_selected_identity_delta_classification": "saved_identity_already_canonical",
            "emitted_pair": {
                "pair_id": "bg0:pair0",
                "q_group_key": ("q_group", "primary", 1, 1),
                "hkl": (-1, 0, 1),
                "source_branch_index": 0,
                "source_peak_index": 0,
            },
        },
    )

    compatibility_paths: list[Path] = []

    def _fake_compatibility(checked_state_path, background_index):
        checked_path = Path(checked_state_path).resolve()
        compatibility_paths.append(checked_path)
        exported_state = load_gui_state_file(checked_path)["state"]
        assert exported_state["geometry"]["peak_records"] == []
        assert exported_state["geometry"]["q_group_rows"] == []
        return {
            "ok": False,
            "classification": "seam_failure",
            "failed_pair": {"failure_stage": "compatibility"},
        }

    monkeypatch.setattr(
        probe,
        "_run_saved_state_compatibility_validation",
        _fake_compatibility,
    )

    export_path = tmp_path / "should_not_write_fresh.json"
    result = probe._run_fresh_contract_validation(
        state_path,
        background_index=0,
        sentinel_slot_index=0,
        export_fresh_state_path=export_path,
    )

    assert len(compatibility_paths) == 1
    assert compatibility_paths[0] != state_path.resolve()
    assert compatibility_paths[0] != export_path.resolve()
    assert result["ok"] is False
    assert result["classification"] == "fresh_contract_state_compatibility_fail"
    assert result["compatibility_ok"] is False
    assert result["fresh_export_ok"] is False
    assert result["fresh_state_export_written"] is False
    assert "exported_fresh_state_path" not in result
    assert result["failed_pair"]["failure_stage"] == "compatibility"
    assert export_path.exists() is False


def test_fresh_contract_validation_reports_export_write_failure_without_crashing(
    monkeypatch,
    tmp_path,
) -> None:
    probe = _load_geometry_preflight_probe_module()
    saved_entry = {
        "pair_id": "bg0:pair0",
        "q_group_key": ("q_group", "primary", 1, 1),
        "hkl": (-1, 0, 1),
        "source_branch_index": 0,
        "source_peak_index": 0,
    }
    state_path = tmp_path / "geometry_probe_state.json"
    context = {
        "ok": True,
        "state_path": str(state_path),
        "background_index": 0,
        "saved_pair_count": 1,
        "captured_preflight_error_text": None,
        "used_isolated_background_dataset": False,
        "runtime_prepare_ok": True,
        "dataset_pair_count": 1,
        "dataset_resolved_source_pair_count": 1,
        "harness_validation": {"valid": True},
        "saved_entries": [saved_entry],
        "saved_state": {
            "files": {
                "background_files": ["C:/tmp/bg0.osc"],
                "current_background_index": 0,
            },
            "variables": {
                "geometry_fit_background_selection_var": "current",
            },
            "geometry": {
                "manual_pairs": [
                    {
                        "background_index": 0,
                        "background_name": "bg0",
                        "background_path": "C:/tmp/bg0.osc",
                        "entries": [dict(saved_entry)],
                    }
                ],
                "peak_records": [],
                "q_group_rows": [],
            },
        },
        "dataset": {},
        "group_cache": {},
        "manual_dataset_bindings": SimpleNamespace(
            geometry_manual_entry_display_coords=lambda entry: (0.0, 0.0)
        ),
    }

    monkeypatch.setattr(
        probe,
        "_prepare_validation_context",
        lambda state_path, background_index: context,
    )
    monkeypatch.setattr(
        probe,
        "_prepare_fresh_slot_runtime",
        lambda **kwargs: {"prepared": True},
    )
    monkeypatch.setattr(
        probe,
        "_run_fresh_slot_validation",
        lambda **kwargs: {
            "ok": True,
            "rebind_ok": True,
            "classification": "pass",
            "slot_index": int(kwargs["slot_index"]),
            "saved_entry": dict(saved_entry),
            "candidate_selection": {
                "ok": True,
                "selection_status": "selected",
                "background_distance_px": 1.25,
                "best_distance_px": 1.25,
                "selected_candidate": {
                    "pair_id": "bg0:pair0",
                    "q_group_key": ("q_group", "primary", 1, 1),
                    "hkl": (-1, 0, 1),
                    "source_branch_index": 0,
                    "source_peak_index": 0,
                },
            },
            "saved_to_selected_identity_delta": {},
            "saved_to_selected_identity_delta_classification": "saved_identity_already_canonical",
            "emitted_pair": dict(saved_entry),
        },
    )
    monkeypatch.setattr(
        probe,
        "_run_saved_state_compatibility_validation",
        lambda *args, **kwargs: {
            "ok": True,
            "classification": "pass",
            "checked_slot_indices": [0],
        },
    )

    export_path = tmp_path / "fresh_export.json"
    previous_export_state = {"sentinel": "keep-me"}
    save_gui_state_file(export_path, previous_export_state)
    original_save_gui_state_file = probe.save_gui_state_file

    def _save_with_export_failure(path, state):
        resolved_path = Path(path).resolve()
        if (
            resolved_path.parent == export_path.resolve().parent
            and resolved_path.name.startswith(f"{export_path.name}.")
            and resolved_path.suffix == ".tmp"
        ):
            resolved_path.write_text("partial", encoding="utf-8")
            raise OSError("disk full")
        return original_save_gui_state_file(path, state)

    monkeypatch.setattr(probe, "save_gui_state_file", _save_with_export_failure)

    result = probe._run_fresh_contract_validation(
        state_path,
        background_index=0,
        sentinel_slot_index=0,
        export_fresh_state_path=export_path,
    )

    assert result["ok"] is False
    assert result["classification"] == "fresh_contract_export_fail"
    assert result["compatibility_ok"] is True
    assert result["fresh_export_ok"] is False
    assert result["fresh_state_export_written"] is False
    assert "exported_fresh_state_path" not in result
    assert result["failed_pair"]["failure_stage"] == "fresh_contract_export"
    assert result["failed_pair"]["error_type"] == "OSError"
    assert result["failed_pair"]["error_text"] == "disk full"
    assert result["failed_pair"]["partial_target_removed"] is True
    assert export_path.exists() is True
    assert load_gui_state_file(export_path)["state"] == previous_export_state


def test_fresh_contract_validation_reports_temp_state_save_failure_without_crashing(
    monkeypatch,
    tmp_path,
) -> None:
    probe = _load_geometry_preflight_probe_module()
    saved_entry = {
        "pair_id": "bg0:pair0",
        "q_group_key": ("q_group", "primary", 1, 1),
        "hkl": (-1, 0, 1),
        "source_branch_index": 0,
        "source_peak_index": 0,
    }
    state_path = tmp_path / "geometry_probe_state.json"
    context = {
        "ok": True,
        "state_path": str(state_path),
        "background_index": 0,
        "saved_pair_count": 1,
        "captured_preflight_error_text": None,
        "used_isolated_background_dataset": False,
        "runtime_prepare_ok": True,
        "dataset_pair_count": 1,
        "dataset_resolved_source_pair_count": 1,
        "harness_validation": {"valid": True},
        "saved_entries": [saved_entry],
        "saved_state": {
            "files": {
                "background_files": ["C:/tmp/bg0.osc"],
                "current_background_index": 0,
            },
            "variables": {
                "geometry_fit_background_selection_var": "current",
            },
            "geometry": {
                "manual_pairs": [
                    {
                        "background_index": 0,
                        "background_name": "bg0",
                        "background_path": "C:/tmp/bg0.osc",
                        "entries": [dict(saved_entry)],
                    }
                ],
                "peak_records": [],
                "q_group_rows": [],
            },
        },
        "dataset": {},
        "group_cache": {},
        "manual_dataset_bindings": SimpleNamespace(
            geometry_manual_entry_display_coords=lambda entry: (0.0, 0.0)
        ),
    }

    monkeypatch.setattr(
        probe,
        "_prepare_validation_context",
        lambda state_path, background_index: context,
    )
    monkeypatch.setattr(
        probe,
        "_prepare_fresh_slot_runtime",
        lambda **kwargs: {"prepared": True},
    )
    monkeypatch.setattr(
        probe,
        "_run_fresh_slot_validation",
        lambda **kwargs: {
            "ok": True,
            "rebind_ok": True,
            "classification": "pass",
            "slot_index": int(kwargs["slot_index"]),
            "saved_entry": dict(saved_entry),
            "candidate_selection": {
                "ok": True,
                "selection_status": "selected",
                "background_distance_px": 1.25,
                "best_distance_px": 1.25,
                "selected_candidate": dict(saved_entry),
            },
            "saved_to_selected_identity_delta": {},
            "saved_to_selected_identity_delta_classification": "saved_identity_already_canonical",
            "emitted_pair": dict(saved_entry),
        },
    )
    monkeypatch.setattr(
        probe,
        "_run_saved_state_compatibility_validation",
        lambda *args, **kwargs: pytest.fail("compatibility should not run after temp-save failure"),
    )

    original_save_gui_state_file = probe.save_gui_state_file

    def _save_with_temp_failure(path, state):
        resolved_path = Path(path).resolve()
        if resolved_path.name.startswith("fresh_state.json.") and resolved_path.suffix == ".tmp":
            raise OSError("temp disk full")
        return original_save_gui_state_file(path, state)

    monkeypatch.setattr(probe, "save_gui_state_file", _save_with_temp_failure)

    export_path = tmp_path / "should_not_write_fresh.json"
    result = probe._run_fresh_contract_validation(
        state_path,
        background_index=0,
        sentinel_slot_index=0,
        export_fresh_state_path=export_path,
    )

    assert result["ok"] is False
    assert result["classification"] == "fresh_contract_temp_save_fail"
    assert result["compatibility_ok"] is False
    assert result["fresh_export_ok"] is False
    assert result["fresh_state_export_written"] is False
    assert result["exported_state_compatibility"]["classification"] == "temp_state_save_failed"
    assert result["state_compatibility"]["classification"] == "temp_state_save_failed"
    assert result["failed_pair"]["failure_stage"] == "fresh_contract_temp_state_save"
    assert result["failed_pair"]["error_type"] == "OSError"
    assert result["failed_pair"]["error_text"] == "temp disk full"
    assert export_path.exists() is False


def test_fresh_contract_validation_slot_failure_keeps_explicit_status_fields(
    monkeypatch,
    tmp_path,
) -> None:
    probe = _load_geometry_preflight_probe_module()
    state_path = tmp_path / "geometry_probe_state.json"
    context = {
        "ok": True,
        "state_path": str(state_path),
        "background_index": 0,
        "saved_pair_count": 1,
        "captured_preflight_error_text": None,
        "used_isolated_background_dataset": False,
        "runtime_prepare_ok": True,
        "dataset_pair_count": 1,
        "dataset_resolved_source_pair_count": 1,
        "harness_validation": {"valid": True},
        "saved_entries": [{"pair_id": "bg0:pair0"}],
        "saved_state": {"geometry": {"manual_pairs": []}},
        "dataset": {},
        "group_cache": {},
        "manual_dataset_bindings": SimpleNamespace(
            geometry_manual_entry_display_coords=lambda entry: (0.0, 0.0)
        ),
    }

    monkeypatch.setattr(
        probe,
        "_prepare_validation_context",
        lambda state_path, background_index: context,
    )
    monkeypatch.setattr(
        probe,
        "_prepare_fresh_slot_runtime",
        lambda **kwargs: {"prepared": True},
    )
    monkeypatch.setattr(
        probe,
        "_run_fresh_slot_validation",
        lambda **kwargs: {
            "ok": False,
            "classification": "slot_fail",
            "failed_pair": {"failure_stage": "slot_validation"},
        },
    )

    export_path = tmp_path / "should_not_write_fresh.json"
    result = probe._run_fresh_contract_validation(
        state_path,
        background_index=0,
        sentinel_slot_index=0,
        export_fresh_state_path=export_path,
    )

    assert result["ok"] is False
    assert result["classification"] == "slot_fail"
    assert result["compatibility_ok"] is False
    assert result["fresh_export_ok"] is False
    assert result["fresh_state_export_written"] is False
    assert result["state_compatibility"]["classification"] == "skipped_due_to_slot_failure"
    assert result["exported_state_compatibility"]["classification"] == "skipped_due_to_slot_failure"
    assert result["failed_pair"]["failure_stage"] == "slot_validation"
    assert export_path.exists() is False


def test_fresh_all_contract_validation_reports_export_write_failure_without_crashing(
    monkeypatch,
    tmp_path,
) -> None:
    probe = _load_geometry_preflight_probe_module()
    saved_entries = [
        {
            "pair_id": f"bg0:pair{slot_index}",
            "q_group_key": ("q_group", "primary", 1, slot_index + 1),
            "hkl": (-1, 0, slot_index + 1),
            "source_branch_index": slot_index % 2,
            "source_peak_index": slot_index % 2,
            "placement_error_px": 10.0,
        }
        for slot_index in range(2)
    ]
    state_path = tmp_path / "geometry_probe_state.json"
    context = {
        "ok": True,
        "state_path": str(state_path),
        "background_index": 0,
        "saved_pair_count": len(saved_entries),
        "captured_preflight_error_text": None,
        "used_isolated_background_dataset": False,
        "runtime_prepare_ok": True,
        "dataset_pair_count": len(saved_entries),
        "dataset_resolved_source_pair_count": len(saved_entries),
        "harness_validation": {"valid": True},
        "saved_entries": saved_entries,
        "saved_state": {
            "files": {
                "background_files": ["C:/tmp/bg0.osc"],
                "current_background_index": 0,
            },
            "variables": {
                "geometry_fit_background_selection_var": "current",
            },
            "geometry": {
                "manual_pairs": [
                    {
                        "background_index": 0,
                        "background_name": "bg0",
                        "background_path": "C:/tmp/bg0.osc",
                        "entries": [dict(entry) for entry in saved_entries],
                    }
                ],
                "peak_records": [],
                "q_group_rows": [],
            },
        },
        "dataset": {},
        "group_cache": {},
        "manual_dataset_bindings": SimpleNamespace(
            geometry_manual_entry_display_coords=lambda entry: (0.0, 0.0)
        ),
    }

    monkeypatch.setattr(
        probe,
        "_prepare_validation_context",
        lambda state_path, background_index: context,
    )
    monkeypatch.setattr(
        probe,
        "_prepare_fresh_slot_runtime",
        lambda **kwargs: {"prepared": True},
    )
    monkeypatch.setattr(
        probe,
        "_run_fresh_slot_validation",
        lambda **kwargs: {
            "ok": True,
            "rebind_ok": True,
            "classification": "pass",
            "slot_index": int(kwargs["slot_index"]),
            "saved_entry": dict(saved_entries[int(kwargs["slot_index"])]),
            "candidate_selection": {
                "ok": True,
                "selection_status": "selected",
                "background_distance_px": 4.0,
                "best_distance_px": 4.0,
                "selected_candidate": {
                    "hkl": (-1, 0, int(kwargs["slot_index"]) + 1),
                    "source_branch_index": int(kwargs["slot_index"]) % 2,
                    "source_peak_index": int(kwargs["slot_index"]) % 2,
                },
            },
            "saved_to_selected_identity_delta": {},
            "saved_to_selected_identity_delta_classification": "saved_identity_already_canonical",
            "emitted_pair": {"pair_id": f"bg0:pair{int(kwargs['slot_index'])}"},
        },
    )
    monkeypatch.setattr(
        probe,
        "_run_saved_state_compatibility_validation",
        lambda *args, **kwargs: {"ok": True, "classification": "pass"},
    )

    export_path = tmp_path / "fresh_all_export.json"
    previous_export_state = {"sentinel": "keep-me-too"}
    save_gui_state_file(export_path, previous_export_state)
    original_save_gui_state_file = probe.save_gui_state_file

    def _save_with_export_failure(path, state):
        resolved_path = Path(path).resolve()
        if (
            resolved_path.parent == export_path.resolve().parent
            and resolved_path.name.startswith(f"{export_path.name}.")
            and resolved_path.suffix == ".tmp"
        ):
            resolved_path.write_text("partial", encoding="utf-8")
            raise OSError("disk full")
        return original_save_gui_state_file(path, state)

    monkeypatch.setattr(probe, "save_gui_state_file", _save_with_export_failure)

    result = probe._run_fresh_all_contract_validation(
        state_path,
        background_index=0,
        export_fresh_state_path=export_path,
    )

    assert result["ok"] is False
    assert result["classification"] == "fresh_all_contract_export_fail"
    assert result["compatibility_ok"] is True
    assert result["fresh_export_ok"] is False
    assert result["fresh_state_export_written"] is False
    assert "exported_fresh_state_path" not in result
    assert result["failed_pair"]["failure_stage"] == "fresh_all_contract_export"
    assert result["failed_pair"]["error_type"] == "OSError"
    assert result["failed_pair"]["error_text"] == "disk full"
    assert result["failed_pair"]["partial_target_removed"] is True
    assert export_path.exists() is True
    assert load_gui_state_file(export_path)["state"] == previous_export_state


def test_fresh_all_contract_validation_reports_temp_state_save_failure_without_crashing(
    monkeypatch,
    tmp_path,
) -> None:
    probe = _load_geometry_preflight_probe_module()
    saved_entries = [
        {
            "pair_id": f"bg0:pair{slot_index}",
            "q_group_key": ("q_group", "primary", 1, slot_index + 1),
            "hkl": (-1, 0, slot_index + 1),
            "source_branch_index": slot_index % 2,
            "source_peak_index": slot_index % 2,
            "placement_error_px": 10.0,
        }
        for slot_index in range(2)
    ]
    state_path = tmp_path / "geometry_probe_state.json"
    context = {
        "ok": True,
        "state_path": str(state_path),
        "background_index": 0,
        "saved_pair_count": len(saved_entries),
        "captured_preflight_error_text": None,
        "used_isolated_background_dataset": False,
        "runtime_prepare_ok": True,
        "dataset_pair_count": len(saved_entries),
        "dataset_resolved_source_pair_count": len(saved_entries),
        "harness_validation": {"valid": True},
        "saved_entries": saved_entries,
        "saved_state": {"geometry": {"manual_pairs": []}},
        "dataset": {},
        "group_cache": {},
        "manual_dataset_bindings": SimpleNamespace(
            geometry_manual_entry_display_coords=lambda entry: (0.0, 0.0)
        ),
    }

    monkeypatch.setattr(
        probe,
        "_prepare_validation_context",
        lambda state_path, background_index: context,
    )
    monkeypatch.setattr(
        probe,
        "_prepare_fresh_slot_runtime",
        lambda **kwargs: {"prepared": True},
    )
    monkeypatch.setattr(
        probe,
        "_run_fresh_slot_validation",
        lambda **kwargs: {
            "ok": True,
            "rebind_ok": True,
            "classification": "pass",
            "slot_index": int(kwargs["slot_index"]),
            "saved_entry": dict(saved_entries[int(kwargs["slot_index"])]),
            "candidate_selection": {
                "ok": True,
                "selection_status": "selected",
                "background_distance_px": 4.0,
                "best_distance_px": 4.0,
                "selected_candidate": {
                    "hkl": (-1, 0, int(kwargs["slot_index"]) + 1),
                    "source_branch_index": int(kwargs["slot_index"]) % 2,
                    "source_peak_index": int(kwargs["slot_index"]) % 2,
                },
            },
            "saved_to_selected_identity_delta": {},
            "saved_to_selected_identity_delta_classification": "saved_identity_already_canonical",
            "emitted_pair": {"pair_id": f"bg0:pair{int(kwargs['slot_index'])}"},
        },
    )
    monkeypatch.setattr(
        probe,
        "_run_saved_state_compatibility_validation",
        lambda *args, **kwargs: pytest.fail("compatibility should not run after temp-save failure"),
    )

    original_save_gui_state_file = probe.save_gui_state_file

    def _save_with_temp_failure(path, state):
        resolved_path = Path(path).resolve()
        if (
            resolved_path.name.startswith("fresh_all_state.json.")
            and resolved_path.suffix == ".tmp"
        ):
            raise OSError("temp disk full")
        return original_save_gui_state_file(path, state)

    monkeypatch.setattr(probe, "save_gui_state_file", _save_with_temp_failure)

    export_path = tmp_path / "should_not_write_fresh_all.json"
    result = probe._run_fresh_all_contract_validation(
        state_path,
        background_index=0,
        export_fresh_state_path=export_path,
    )

    assert result["ok"] is False
    assert result["classification"] == "fresh_all_contract_temp_save_fail"
    assert result["compatibility_ok"] is False
    assert result["fresh_export_ok"] is False
    assert result["fresh_state_export_written"] is False
    assert result["exported_state_compatibility"]["classification"] == "temp_state_save_failed"
    assert result["state_compatibility"]["classification"] == "temp_state_save_failed"
    assert result["failed_pair"]["failure_stage"] == "fresh_all_contract_temp_state_save"
    assert result["failed_pair"]["error_type"] == "OSError"
    assert result["failed_pair"]["error_text"] == "temp disk full"
    assert export_path.exists() is False


def test_fresh_all_contract_validation_classifies_runtime_prepare_isolated_rebind_only(
    monkeypatch,
    tmp_path,
) -> None:
    probe = _load_geometry_preflight_probe_module()
    saved_entries = [
        {
            "pair_id": f"bg0:pair{slot_index}",
            "q_group_key": ("q_group", "primary", 1, slot_index + 1),
            "hkl": (-1, 0, slot_index + 1),
            "source_branch_index": slot_index % 2,
            "source_peak_index": slot_index % 2,
            "placement_error_px": 10.0,
        }
        for slot_index in range(2)
    ]
    state_path = tmp_path / "geometry_probe_state.json"
    context = {
        "ok": True,
        "state_path": str(state_path),
        "background_index": 0,
        "saved_pair_count": len(saved_entries),
        "captured_preflight_error_text": "Geometry fit unavailable: saved manual pairs no longer resolve.",
        "used_isolated_background_dataset": True,
        "runtime_prepare_ok": False,
        "dataset_pair_count": len(saved_entries),
        "dataset_resolved_source_pair_count": len(saved_entries),
        "harness_validation": {"valid": True},
        "saved_entries": saved_entries,
        "saved_state": {"geometry": {"manual_pairs": []}},
        "dataset": {},
        "group_cache": {},
        "manual_dataset_bindings": SimpleNamespace(
            geometry_manual_entry_display_coords=lambda entry: (0.0, 0.0)
        ),
    }

    monkeypatch.setattr(
        probe,
        "_prepare_validation_context",
        lambda state_path, background_index: context,
    )
    monkeypatch.setattr(
        probe,
        "_prepare_fresh_slot_runtime",
        lambda **kwargs: {"prepared": True},
    )
    monkeypatch.setattr(
        probe,
        "_run_fresh_slot_validation",
        lambda **kwargs: {
            "ok": True,
            "rebind_ok": True,
            "classification": "pass",
            "slot_index": int(kwargs["slot_index"]),
            "saved_entry": dict(saved_entries[int(kwargs["slot_index"])]),
            "candidate_selection": {
                "ok": True,
                "selection_status": "selected",
                "background_distance_px": 4.0,
                "best_distance_px": 4.0,
                "selected_candidate": {
                    "hkl": (-1, 0, int(kwargs["slot_index"]) + 1),
                    "source_branch_index": int(kwargs["slot_index"]) % 2,
                    "source_peak_index": int(kwargs["slot_index"]) % 2,
                },
            },
            "saved_to_selected_identity_delta": {},
            "saved_to_selected_identity_delta_classification": "saved_identity_already_canonical",
            "emitted_pair": {"pair_id": f"bg0:pair{int(kwargs['slot_index'])}"},
        },
    )
    monkeypatch.setattr(
        probe,
        "_run_saved_state_compatibility_validation",
        lambda *args, **kwargs: {"ok": True, "classification": "pass"},
    )

    export_path = tmp_path / "should_not_write_fresh_all.json"
    result = probe._run_fresh_all_contract_validation(
        state_path,
        background_index=0,
        export_fresh_state_path=export_path,
    )

    assert result["ok"] is False
    assert result["classification"] == "runtime_prepare_failed_but_isolated_rebind_ok"
    assert result["runtime_prepare_ok"] is False
    assert result["isolated_rebind_ok"] is True
    assert result["fresh_export_ok"] is False
    assert result["fresh_state_export_written"] is False
    assert "exported_fresh_state_path" not in result
    assert export_path.exists() is False
    assert result["compatibility_ok"] is True
    assert result["background_distance_gate_ok"] is True
    assert result["detector_distance_gate_ok"] is True
    assert result["max_background_distance_px"] == pytest.approx(4.0)
    assert result["max_candidate_distance_px"] == pytest.approx(4.0)


def test_fresh_all_contract_validation_fails_detector_distance_gate_when_candidates_are_far(
    monkeypatch,
    tmp_path,
) -> None:
    probe = _load_geometry_preflight_probe_module()
    saved_entries = [
        {
            "pair_id": f"bg0:pair{slot_index}",
            "q_group_key": ("q_group", "primary", 1, slot_index + 1),
            "hkl": (-1, 0, slot_index + 1),
            "source_branch_index": 0,
            "source_peak_index": 0,
            "placement_error_px": 8.0,
        }
        for slot_index in range(2)
    ]
    state_path = tmp_path / "geometry_probe_state.json"
    context = {
        "ok": True,
        "state_path": str(state_path),
        "background_index": 0,
        "saved_pair_count": len(saved_entries),
        "captured_preflight_error_text": None,
        "used_isolated_background_dataset": False,
        "runtime_prepare_ok": True,
        "dataset_pair_count": len(saved_entries),
        "dataset_resolved_source_pair_count": len(saved_entries),
        "harness_validation": {"valid": True},
        "saved_entries": saved_entries,
        "saved_state": {"geometry": {"manual_pairs": []}},
        "dataset": {},
        "group_cache": {},
        "manual_dataset_bindings": SimpleNamespace(
            geometry_manual_entry_display_coords=lambda entry: (0.0, 0.0)
        ),
    }

    monkeypatch.setattr(
        probe,
        "_prepare_validation_context",
        lambda state_path, background_index: context,
    )
    monkeypatch.setattr(
        probe,
        "_prepare_fresh_slot_runtime",
        lambda **kwargs: {"prepared": True},
    )
    monkeypatch.setattr(
        probe,
        "_run_fresh_slot_validation",
        lambda **kwargs: {
            "ok": True,
            "rebind_ok": True,
            "classification": "pass",
            "slot_index": int(kwargs["slot_index"]),
            "saved_entry": dict(saved_entries[int(kwargs["slot_index"])]),
            "candidate_selection": {
                "ok": True,
                "selection_status": "selected",
                "background_distance_px": 12.0 if int(kwargs["slot_index"]) == 0 else 150.0,
                "best_distance_px": 12.0 if int(kwargs["slot_index"]) == 0 else 150.0,
                "selected_candidate": {
                    "hkl": (-1, 0, int(kwargs["slot_index"]) + 1),
                    "source_branch_index": 0,
                    "source_peak_index": 0,
                },
            },
            "saved_to_selected_identity_delta": {},
            "saved_to_selected_identity_delta_classification": "saved_identity_already_canonical",
            "emitted_pair": {"pair_id": f"bg0:pair{int(kwargs['slot_index'])}"},
        },
    )
    monkeypatch.setattr(
        probe,
        "_run_saved_state_compatibility_validation",
        lambda *args, **kwargs: {"ok": True, "classification": "pass"},
    )

    result = probe._run_fresh_all_contract_validation(
        state_path,
        background_index=0,
    )

    assert result["ok"] is False
    assert result["runtime_prepare_ok"] is True
    assert result["isolated_rebind_ok"] is True
    assert result["compatibility_ok"] is True
    assert result["fresh_export_ok"] is True
    assert result["background_distance_gate_ok"] is False
    assert result["background_distance_px"] == [12.0, 150.0]
    assert result["max_background_distance_px"] == pytest.approx(150.0)
    assert result["background_distance_gate_threshold_px"] == pytest.approx(100.0)
    assert result["detector_distance_gate_ok"] is False
    assert result["candidate_distance_gate_ok"] is False
    assert result["candidate_distance_px"] == result["background_distance_px"]
    assert result["max_candidate_distance_px"] == result["max_background_distance_px"]
    assert (
        result["candidate_distance_gate_threshold_px"]
        == result["background_distance_gate_threshold_px"]
    )


def test_downstream_identity_validation_rejects_non_canonical_input(
    monkeypatch,
    tmp_path,
) -> None:
    probe = _load_geometry_preflight_probe_module()
    state_path = _write_probe_state(
        tmp_path,
        entries=[
            {
                "pair_id": "bg0:pair0",
                "hkl": (1, 0, 0),
                "source_reflection_index": 100,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "source_branch_index": 0,
                "source_peak_index": 0,
            }
        ],
        peak_records=["stale"],
    )

    monkeypatch.setattr(
        probe,
        "_capture_execution_setup",
        lambda **kwargs: pytest.fail("input contract should fail before preflight"),
    )

    result = probe._run_downstream_identity_validation(
        state_path,
        background_index=0,
    )

    assert result["ok"] is False
    assert result["classification"] == "invalid_downstream_identity_input"
    assert result["failed_stage"] == "input_contract"
    assert [stage["stage"] for stage in result["stage_results"]] == ["input_contract"]


def test_downstream_identity_validation_stops_at_subset_drift(
    monkeypatch,
    tmp_path,
) -> None:
    probe = _load_geometry_preflight_probe_module()
    saved_entries = [
        {
            "pair_id": "bg0:pair0",
            "hkl": (1, 0, 0),
            "source_reflection_index": 100,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": 0,
            "source_peak_index": 0,
        },
        {
            "pair_id": "bg0:pair1",
            "hkl": (1, 0, 0),
            "source_reflection_index": 101,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": 1,
            "source_peak_index": 1,
        },
    ]
    state_path = _write_probe_state(tmp_path, entries=saved_entries)
    prepared_run, postprocess_config = _make_prepared_run(
        joint_background_mode=False,
        tmp_path=tmp_path,
    )
    prepared_run = replace(
        prepared_run,
        current_dataset={
            **prepared_run.current_dataset,
            "dataset_index": 0,
            "measured_for_fit": [dict(entry) for entry in saved_entries],
        },
        dataset_specs=[{"dataset_index": 0, "theta_initial": 3.0}],
    )
    setup = geometry_fit.GeometryFitRuntimeExecutionSetup(
        ui_bindings="unused",
        postprocess_config=postprocess_config,
    )

    monkeypatch.setattr(
        probe,
        "_capture_execution_setup",
        lambda **kwargs: {
            "prepare_kwargs": {"var_names": ["gamma"]},
            "prepare_result": SimpleNamespace(prepared_run=prepared_run, error_text=None),
            "execute_kwargs": {"setup": setup, "var_names": ["gamma"]},
        },
    )
    monkeypatch.setattr(
        probe.gui_geometry_fit,
        "build_geometry_fit_solver_request",
        lambda **kwargs: SimpleNamespace(
            measured_peaks=[dict(entry) for entry in saved_entries],
            miller="miller",
            intensities="intensity",
            params={},
            dataset_specs=[{"dataset_index": 0}],
        ),
    )
    drifted_subset_entries = [dict(entry) for entry in saved_entries]
    drifted_subset_entries[1]["source_peak_index"] = 0
    monkeypatch.setattr(
        probe.opt,
        "_build_geometry_fit_dataset_contexts",
        lambda *args, **kwargs: [
            SimpleNamespace(
                subset=SimpleNamespace(
                    measured_entries=[dict(entry) for entry in drifted_subset_entries]
                )
            )
        ],
    )
    solve_calls: list[str] = []

    def _fail_if_solver_runs(*args, **kwargs):
        solve_calls.append("solve")
        pytest.fail("solver should not run after subset-stage drift")

    monkeypatch.setattr(
        probe.gui_geometry_fit,
        "solve_geometry_fit_request",
        _fail_if_solver_runs,
    )

    result = probe._run_downstream_identity_validation(
        state_path,
        background_index=0,
    )

    assert result["ok"] is False
    assert result["classification"] == "seam_failure"
    assert result["failed_stage"] == "subset_measured_entries"
    assert solve_calls == []
    assert [stage["stage"] for stage in result["stage_results"]] == [
        "input_contract",
        "preflight_normalized_pairs",
        "solver_request_measured_peaks",
        "subset_measured_entries",
    ]


def test_downstream_identity_validation_preserves_canonical_identity(
    monkeypatch,
    tmp_path,
) -> None:
    probe = _load_geometry_preflight_probe_module()
    saved_entries = [
        {
            "pair_id": "bg0:pair0",
            "hkl": (1, 0, 0),
            "source_reflection_index": 100,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": 0,
            "source_peak_index": 0,
        },
        {
            "pair_id": "bg0:pair1",
            "hkl": (1, 0, 0),
            "source_reflection_index": 101,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": 1,
            "source_peak_index": 1,
        },
    ]
    state_path = _write_probe_state(tmp_path, entries=saved_entries)
    prepared_run, postprocess_config = _make_prepared_run(
        joint_background_mode=False,
        tmp_path=tmp_path,
    )
    prepared_run = replace(
        prepared_run,
        current_dataset={
            **prepared_run.current_dataset,
            "dataset_index": 0,
            "measured_for_fit": [dict(entry) for entry in saved_entries],
        },
        dataset_specs=[{"dataset_index": 0, "theta_initial": 3.0}],
    )
    setup = geometry_fit.GeometryFitRuntimeExecutionSetup(
        ui_bindings="unused",
        postprocess_config=postprocess_config,
    )

    monkeypatch.setattr(
        probe,
        "_capture_execution_setup",
        lambda **kwargs: {
            "prepare_kwargs": {"var_names": ["gamma"]},
            "prepare_result": SimpleNamespace(prepared_run=prepared_run, error_text=None),
            "execute_kwargs": {"setup": setup, "var_names": ["gamma"]},
        },
    )
    monkeypatch.setattr(
        probe.gui_geometry_fit,
        "build_geometry_fit_solver_request",
        lambda **kwargs: SimpleNamespace(
            measured_peaks=[dict(entry) for entry in saved_entries],
            miller="miller",
            intensities="intensity",
            params={},
            dataset_specs=[{"dataset_index": 0}],
        ),
    )
    monkeypatch.setattr(
        probe.opt,
        "_build_geometry_fit_dataset_contexts",
        lambda *args, **kwargs: [
            SimpleNamespace(
                subset=SimpleNamespace(measured_entries=[dict(entry) for entry in saved_entries])
            )
        ],
    )
    seed_records = [
        {
            **dict(entry),
            "dataset_index": 0,
            "overlay_match_index": slot_index,
            "match_input_index": slot_index,
            "match_status": "matched",
            "frozen_locator_kind": "trusted_branch",
            "frozen_table_namespace": "full_reflection",
        }
        for slot_index, entry in enumerate(saved_entries)
    ]
    point_match_diagnostics = list(
        reversed(
            [
                {
                    **dict(entry),
                    "dataset_index": 0,
                    "overlay_match_index": slot_index,
                    "match_input_index": slot_index,
                    "match_status": "matched",
                    "match_kind": "full_beam_fixed",
                    "resolution_kind": "fixed_source",
                }
                for slot_index, entry in enumerate(saved_entries)
            ]
        )
    )
    monkeypatch.setattr(
        probe.gui_geometry_fit,
        "solve_geometry_fit_request",
        lambda *args, **kwargs: SimpleNamespace(
            full_beam_polish_summary={
                "seed_correspondence_records": seed_records,
                "point_match_diagnostics": [],
            },
            point_match_diagnostics=point_match_diagnostics,
            final_metric_name="full_beam_fixed_correspondence",
        ),
    )

    result = probe._run_downstream_identity_validation(
        state_path,
        background_index=0,
    )

    assert result["ok"] is True
    assert result["classification"] == "pass"
    assert result["final_metric_name"] == "full_beam_fixed_correspondence"
    assert [stage["stage"] for stage in result["stage_results"]] == list(
        probe.DOWNSTREAM_IDENTITY_STAGE_ORDER
    )
    assert result["stage_results"][-2]["ok"] is True
    assert result["stage_results"][-1]["ok"] is True


def test_downstream_identity_validation_uses_optimizer_captured_full_beam_diagnostics(
    monkeypatch,
    tmp_path,
) -> None:
    probe = _load_geometry_preflight_probe_module()
    saved_entries = [
        {
            "pair_id": "bg0:pair0",
            "hkl": (1, 0, 0),
            "source_reflection_index": 100,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": 0,
            "source_peak_index": 0,
        }
    ]
    state_path = _write_probe_state(tmp_path, entries=saved_entries)
    prepared_run, postprocess_config = _make_prepared_run(
        joint_background_mode=False,
        tmp_path=tmp_path,
    )
    prepared_run = replace(
        prepared_run,
        current_dataset={
            **prepared_run.current_dataset,
            "dataset_index": 0,
            "measured_for_fit": [dict(entry) for entry in saved_entries],
        },
        dataset_specs=[{"dataset_index": 0, "theta_initial": 3.0}],
    )
    setup = geometry_fit.GeometryFitRuntimeExecutionSetup(
        ui_bindings="unused",
        postprocess_config=postprocess_config,
    )

    monkeypatch.setattr(
        probe,
        "_capture_execution_setup",
        lambda **kwargs: {
            "prepare_kwargs": {"var_names": ["gamma"]},
            "prepare_result": SimpleNamespace(prepared_run=prepared_run, error_text=None),
            "execute_kwargs": {"setup": setup, "var_names": ["gamma"]},
        },
    )
    monkeypatch.setattr(
        probe.gui_geometry_fit,
        "build_geometry_fit_solver_request",
        lambda **kwargs: SimpleNamespace(
            measured_peaks=[dict(entry) for entry in saved_entries],
            miller="miller",
            intensities="intensity",
            params={},
            dataset_specs=[{"dataset_index": 0}],
        ),
    )
    monkeypatch.setattr(
        probe.opt,
        "_build_geometry_fit_dataset_contexts",
        lambda *args, **kwargs: [
            SimpleNamespace(
                subset=SimpleNamespace(measured_entries=[dict(entry) for entry in saved_entries])
            )
        ],
    )
    seed_records = [
        {
            **dict(entry),
            "dataset_index": 0,
            "overlay_match_index": 0,
            "match_input_index": 0,
            "match_status": "matched",
            "frozen_locator_kind": "trusted_branch",
            "frozen_table_namespace": "full_reflection",
        }
        for entry in saved_entries
    ]
    selected_full_beam_diag = [
        {
            **dict(saved_entries[0]),
            "dataset_index": 0,
            "overlay_match_index": 0,
            "match_input_index": 0,
            "match_kind": "fixed_correspondence",
            "match_status": "matched",
            "resolution_kind": "fixed_source",
            "resolution_reason": "resolved",
            "match_radius_exceeded": False,
            "resolved_table_index": 3,
            "resolved_peak_index": 1,
            "distance_px": 2.0,
            "weighted_dx_px": 1.25,
            "weighted_dy_px": -0.5,
            "distance_weight": 0.5,
            "sigma_weight": 2.0,
            "priority_weight": 1.0,
            "weight": 1.0,
        }
    ]
    candidate_full_beam_diag = [
        {
            **dict(saved_entries[0]),
            "dataset_index": 0,
            "overlay_match_index": 0,
            "match_input_index": 0,
            "match_kind": "fixed_correspondence",
            "match_status": "matched",
            "resolution_kind": "fixed_source",
            "resolution_reason": "resolved",
            "match_radius_exceeded": False,
            "resolved_table_index": 3,
            "resolved_peak_index": 1,
            "distance_px": 3.0,
            "weighted_dx_px": 0.25,
            "weighted_dy_px": 0.5,
            "distance_weight": 0.25,
            "sigma_weight": 4.0,
            "priority_weight": 1.0,
            "weight": 1.0,
        }
    ]
    monkeypatch.setattr(
        probe.gui_geometry_fit,
        "solve_geometry_fit_request",
        lambda *args, **kwargs: SimpleNamespace(
            full_beam_polish_summary={
                "accepted": False,
                "reason": "weighted_rms_regressed",
                "seed_correspondence_records": seed_records,
                "point_match_diagnostics": selected_full_beam_diag,
                "start_point_match_diagnostics": selected_full_beam_diag,
                "candidate_point_match_diagnostics": candidate_full_beam_diag,
                "start_point_match_summary": {"matched_pair_count": 1},
                "candidate_point_match_summary": {"matched_pair_count": 1},
            },
            point_match_diagnostics=[
                {
                    **dict(saved_entries[0]),
                    "dataset_index": 0,
                    "overlay_match_index": 0,
                    "match_input_index": 0,
                    "match_kind": "fixed_correspondence",
                    "match_status": "matched",
                    "resolution_kind": "fixed_source",
                    "resolved_table_index": 99,
                    "resolved_peak_index": 99,
                    "distance_px": 99.0,
                    "weighted_dx_px": 999.0,
                    "weighted_dy_px": 999.0,
                }
            ],
            final_metric_name="central_point_match",
        ),
    )

    result = probe._run_downstream_identity_validation(
        state_path,
        background_index=0,
    )

    assert result["ok"] is False
    assert result["failed_stage"] == "full_beam_fixed_correspondence"
    assert result["failed_pair"]["failure_reason"] == "unexpected_final_metric_name"
    comparison = result["full_beam_point_match_comparison"]
    assert comparison["comparison_classification"] == "objective_acceptance_mismatch"
    assert comparison["start_identity_key_count"] == 1
    assert comparison["candidate_identity_key_count"] == 1
    assert len(comparison["paired_entries"]) == 1
    paired_entry = comparison["paired_entries"][0]
    assert paired_entry["pair_id"] == "bg0:pair0"
    assert paired_entry["hkl"] == (1, 0, 0)
    assert paired_entry["source_reflection_index"] == 100
    assert paired_entry["source_branch_index"] == 0
    assert paired_entry["source_peak_index"] == 0
    assert paired_entry["start_weighted_dx_px"] == 1.25
    assert paired_entry["candidate_weighted_dx_px"] == 0.25
    assert paired_entry["start_resolved_table_index"] == 3
    assert paired_entry["candidate_resolved_table_index"] == 3
    assert paired_entry["start_resolution_reason"] == "resolved"
    assert paired_entry["candidate_resolution_reason"] == "resolved"
    assert paired_entry["start_match_radius_exceeded"] is False
    assert paired_entry["candidate_match_radius_exceeded"] is False
    assert paired_entry["delta_px"] == 1.0
    assert paired_entry["coverage_drift"] is False
    assert paired_entry["resolved_correspondence_drift"] is False
    assert paired_entry["start_weighted_dx_px"] != 999.0
    assert paired_entry["candidate_weighted_dx_px"] != 999.0
    assert comparison["ranked_by_delta_px"][0]["pair_id"] == "bg0:pair0"
    assert comparison["ranked_by_candidate_distance_px"][0]["pair_id"] == "bg0:pair0"
    assert comparison["ranked_by_delta_sq_px"][0]["pair_id"] == "bg0:pair0"


def test_downstream_identity_validation_promotes_coverage_mismatch_classification(
    monkeypatch,
    tmp_path,
) -> None:
    probe = _load_geometry_preflight_probe_module()
    saved_entries = [
        {
            "pair_id": "bg0:pair0",
            "hkl": (1, 0, 0),
            "source_reflection_index": 100,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": 0,
            "source_peak_index": 0,
        }
    ]
    state_path = _write_probe_state(tmp_path, entries=saved_entries)
    prepared_run, postprocess_config = _make_prepared_run(
        joint_background_mode=False,
        tmp_path=tmp_path,
    )
    prepared_run = replace(
        prepared_run,
        current_dataset={
            **prepared_run.current_dataset,
            "dataset_index": 0,
            "measured_for_fit": [dict(entry) for entry in saved_entries],
        },
        dataset_specs=[{"dataset_index": 0, "theta_initial": 3.0}],
    )
    setup = geometry_fit.GeometryFitRuntimeExecutionSetup(
        ui_bindings="unused",
        postprocess_config=postprocess_config,
    )

    monkeypatch.setattr(
        probe,
        "_capture_execution_setup",
        lambda **kwargs: {
            "prepare_kwargs": {"var_names": ["gamma"]},
            "prepare_result": SimpleNamespace(prepared_run=prepared_run, error_text=None),
            "execute_kwargs": {"setup": setup, "var_names": ["gamma"]},
        },
    )
    monkeypatch.setattr(
        probe.gui_geometry_fit,
        "build_geometry_fit_solver_request",
        lambda **kwargs: SimpleNamespace(
            measured_peaks=[dict(entry) for entry in saved_entries],
            miller="miller",
            intensities="intensity",
            params={},
            dataset_specs=[{"dataset_index": 0}],
        ),
    )
    monkeypatch.setattr(
        probe.opt,
        "_build_geometry_fit_dataset_contexts",
        lambda *args, **kwargs: [
            SimpleNamespace(
                subset=SimpleNamespace(measured_entries=[dict(entry) for entry in saved_entries])
            )
        ],
    )
    seed_records = [
        {
            **dict(entry),
            "dataset_index": 0,
            "overlay_match_index": 0,
            "match_input_index": 0,
            "match_status": "matched",
            "frozen_locator_kind": "trusted_branch",
            "frozen_table_namespace": "full_reflection",
        }
        for entry in saved_entries
    ]
    missing_full_beam_diag = [
        {
            **dict(saved_entries[0]),
            "dataset_index": 0,
            "overlay_match_index": 0,
            "match_input_index": 0,
            "match_kind": "full_beam_fixed",
            "match_status": "missing_pair",
            "resolution_kind": "fixed_source",
            "resolved_table_index": 3,
            "resolved_peak_index": 1,
            "distance_weight": 1.0,
            "sigma_weight": 2.0,
            "priority_weight": 1.0,
            "weight": 2.0,
            "weighted_dx_px": 0.0,
            "weighted_dy_px": 0.0,
        }
    ]
    monkeypatch.setattr(
        probe.gui_geometry_fit,
        "solve_geometry_fit_request",
        lambda *args, **kwargs: SimpleNamespace(
            full_beam_polish_summary={
                "accepted": False,
                "reason": "weighted_rms_regressed",
                "seed_correspondence_records": seed_records,
                "point_match_diagnostics": missing_full_beam_diag,
                "start_point_match_diagnostics": missing_full_beam_diag,
                "candidate_point_match_diagnostics": missing_full_beam_diag,
                "start_point_match_summary": {"matched_pair_count": 0},
                "candidate_point_match_summary": {"matched_pair_count": 0},
            },
            point_match_diagnostics=[
                {
                    **dict(saved_entries[0]),
                    "dataset_index": 0,
                    "overlay_match_index": 0,
                    "match_input_index": 0,
                    "match_kind": "fixed_correspondence",
                    "match_status": "matched",
                    "resolution_kind": "fixed_source",
                }
            ],
            final_metric_name="central_point_match",
        ),
    )

    result = probe._run_downstream_identity_validation(
        state_path,
        background_index=0,
    )

    assert result["ok"] is False
    assert result["classification"] == "identity_key_coverage_mismatch"
    assert result["failed_stage"] == "full_beam_identity_coverage"
    assert (
        result["stage_results"][-1]["coverage_mismatch_classification"]
        == "identity_key_coverage_mismatch"
    )
    assert result["stage_results"][-1]["failure_reason"] == "unresolved_full_beam_correspondence"


def test_fresh_one_pair_gui_state_save_reload_preflight_resolves_without_generic_fallback(
    tmp_path,
) -> None:
    saved_entry = {
        "q_group_key": ("q_group", "primary", 1, 5),
        "source_table_index": 1,
        "source_reflection_index": 203,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "hkl": (-1, 0, 5),
        "x": 189.5,
        "y": 97.0,
        "caked_x": 30.0,
        "caked_y": 9.0,
        "refined_sim_caked_x": 30.0,
        "refined_sim_caked_y": 9.0,
        "refined_sim_x": 189.5,
        "refined_sim_y": 97.0,
    }
    simulated_rows = [
        {
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 0,
            "source_reflection_index": 202,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "hkl": (-1, 0, 5),
            "sim_col": 110.0,
            "sim_row": 100.0,
        },
        {
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 1,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "sim_col": 189.5,
            "sim_row": 97.0,
        },
        {
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 2,
            "source_reflection_index": 204,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "hkl": (-1, 0, 5),
            "sim_col": 182.0,
            "sim_row": 98.0,
        },
    ]
    state_path = tmp_path / "fresh_one_pair_state.json"
    save_gui_state_file(
        state_path,
        {
            "files": {
                "background_files": ["C:/tmp/bg0.osc", "C:/tmp/bg1.osc"],
                "current_background_index": 0,
            },
            "variables": {
                "geometry_fit_background_selection_var": "current",
            },
            "geometry": {
                "manual_pairs": [
                    {
                        "background_index": 0,
                        "entries": [dict(saved_entry)],
                    }
                ],
                "peak_records": [],
                "q_group_rows": [],
            },
        },
    )

    loaded_state = load_gui_state_file(state_path)["state"]
    loaded_entry = loaded_state["geometry"]["manual_pairs"][0]["entries"][0]
    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=_make_legacy_dense_manual_dataset_bindings(
            saved_entries=[loaded_entry],
            simulated_rows=simulated_rows,
            refresh_pairs=False,
        ),
        orientation_cfg={"mode": "auto"},
    )

    assert dataset["resolved_source_pair_count"] == 1
    measured_pair = dataset["measured_for_fit"][0]
    resolution_diag = dataset["source_resolution_diagnostics"][0]
    assert measured_pair["source_reflection_index"] == 203
    assert measured_pair["source_reflection_namespace"] == "full_reflection"
    assert measured_pair["source_reflection_is_full"] is True
    assert measured_pair["source_branch_index"] == 1
    assert measured_pair["source_peak_index"] == 1
    assert resolution_diag["fit_resolution_kind"] in {"source_row", "source_peak"}
    assert resolution_diag["overlay_resolution_kind"] in {"source_row", "source_peak"}


def test_build_geometry_manual_fit_dataset_preserves_caked_display_coords() -> None:
    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=64,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda idx: [
            {
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
                "hkl": (1, 1, 0),
                "x": 30.0,
                "y": 40.0,
                "caked_x": 150.0,
                "caked_y": 160.0,
            }
        ],
        load_background_by_index=lambda idx: (
            np.zeros((4, 5), dtype=np.float64),
            np.zeros((4, 5), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_simulated_peaks_for_params=(
            lambda params, *, prefer_cache: [{"dummy": True}]
        ),
        geometry_manual_simulated_lookup=lambda _simulated_peaks: {
            (1, 2): {
                "display_col": 901.0,
                "display_row": -802.0,
                "sim_col": 91.0,
                "sim_row": 82.0,
                "sim_col_raw": 9.0,
                "sim_row_raw": 8.0,
                "caked_x": 91.0,
                "caked_y": 82.0,
                "hkl": (1, 1, 0),
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
            }
        },
        geometry_manual_entry_display_coords=lambda entry: (30.0, 40.0),
        unrotate_display_peaks=lambda entries, shape, *, k: [{"x": 30.0, "y": 40.0}],
        display_to_native_sim_coords=lambda col, row, shape: (11.0, 12.0),
        select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "yx",
                "label": "identity",
            },
            {"pairs": 1},
        ),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: list(entries),
        orient_image_for_fit=lambda image, **kwargs: image,
        pick_uses_caked_space=lambda: True,
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={},
    )

    assert dataset["initial_pairs_display"][0]["sim_caked_display"] == (91.0, 82.0)
    assert dataset["initial_pairs_display"][0]["sim_display"] == (91.0, 82.0)
    assert dataset["initial_pairs_display"][0]["bg_display"] == (30.0, 40.0)
    assert dataset["initial_pairs_display"][0]["bg_caked_display"] == (150.0, 160.0)


def test_build_geometry_manual_fit_dataset_projects_raw_detector_rows_into_caked_angles(
    monkeypatch,
) -> None:
    bundle = _make_stub_caked_bundle(
        detector_shape=(8, 8),
        radial_axis=np.linspace(10.0, 17.0, 8),
        azimuth_axis=np.linspace(13.0, 20.0, 8),
    )
    monkeypatch.setattr(
        manual_geometry,
        "_detector_pixel_to_caked_bin",
        lambda live_bundle, col, row: (
            (11.5, 13.5)
            if live_bundle is bundle and (float(col), float(row)) == (3.0, 4.0)
            else (None, None)
        ),
    )
    callbacks = manual_geometry.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
        last_caked_radial_values=lambda: np.linspace(10.0, 17.0, 8),
        last_caked_azimuth_values=lambda: np.linspace(13.0, 20.0, 8),
        current_background_display=lambda: np.zeros((8, 8), dtype=float),
        current_background_native=lambda: np.ones((8, 8), dtype=float),
        ai=lambda: object(),
        caked_transform_bundle=lambda: bundle,
        image_size=lambda: 8,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        detector_pixel_to_scattering_angles=lambda *_args: (_ for _ in ()).throw(
            AssertionError("analytic forward fallback should not be used")
        ),
    )

    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=8,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda idx: [
            {
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
                "hkl": (1, 1, 0),
                "caked_x": 12.0,
                "caked_y": 14.0,
            }
        ],
        load_background_by_index=lambda idx: (
            np.zeros((8, 8), dtype=np.float64),
            np.zeros((8, 8), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_source_rows_for_background=lambda *args, **kwargs: [
            {
                "display_col": 400.0,
                "display_row": -500.0,
                "sim_col": 300.0,
                "sim_row": -200.0,
                "sim_col_raw": 3.0,
                "sim_row_raw": 4.0,
                "caked_x": 401.0,
                "caked_y": -499.0,
                "raw_caked_x": 402.0,
                "raw_caked_y": -498.0,
                "two_theta_deg": 403.0,
                "phi_deg": -497.0,
                "hkl": (1, 1, 0),
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
            }
        ],
        geometry_manual_simulated_peaks_for_params=lambda params, *, prefer_cache: [],
        geometry_manual_simulated_lookup=lambda _simulated_peaks: {},
        geometry_manual_entry_display_coords=lambda entry: (
            float(entry.get("caked_x", 0.0)),
            float(entry.get("caked_y", 0.0)),
        ),
        geometry_manual_project_peaks_to_current_view=callbacks.project_peaks_to_current_view,
        unrotate_display_peaks=lambda entries, shape, *, k: [dict(entry) for entry in entries],
        display_to_native_sim_coords=lambda col, row, shape: (float(col), float(row)),
        select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "yx",
                "label": "identity",
            },
            {"pairs": len(sim_pts)},
        ),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: list(entries),
        orient_image_for_fit=lambda image, **kwargs: image,
        pick_uses_caked_space=lambda: True,
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={},
    )

    initial_entry = dataset["initial_pairs_display"][0]
    assert initial_entry["sim_display"] == (11.5, 13.5)
    assert initial_entry["sim_caked_display"] == (11.5, 13.5)
    assert initial_entry["simulated_two_theta_deg"] == 11.5
    assert initial_entry["simulated_phi_deg"] == 13.5
    assert initial_entry["sim_native"] == (3.0, 4.0)


def test_build_geometry_manual_fit_dataset_exact_projector_uses_manual_selection_sequence(
    monkeypatch,
) -> None:
    bundle = _make_stub_caked_bundle(
        detector_shape=(8, 8),
        radial_axis=np.linspace(10.0, 17.0, 8),
        azimuth_axis=np.linspace(-4.0, 3.0, 8),
    )
    call_order: list[tuple[str, float, float]] = []

    monkeypatch.setattr(
        geometry_fit,
        "_fit_detector_coords_to_native_detector_coords",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("native_detector input must not convert through fit frame")
        ),
    )
    monkeypatch.setattr(
        geometry_fit,
        "_geometry_fit_resolve_dynamic_reanchor_caked_bundle",
        lambda **kwargs: bundle,
    )
    monkeypatch.setattr(
        manual_geometry,
        "_detector_pixel_to_caked_bin",
        lambda live_bundle, col, row: (
            call_order.append(("detector_pixel_to_caked_bin", float(col), float(row)))
            or (
                (22.5, -35.5)
                if live_bundle is bundle and (float(col), float(row)) == (4.0, 6.0)
                else (21.5, -33.5)
                if live_bundle is bundle and (float(col), float(row)) == (3.0, 4.0)
                else (None, None)
            )
        ),
    )

    callbacks = manual_geometry.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
        last_caked_radial_values=lambda: np.linspace(10.0, 17.0, 8),
        last_caked_azimuth_values=lambda: np.linspace(-4.0, 3.0, 8),
        current_background_display=lambda: np.zeros((8, 8), dtype=float),
        current_background_native=lambda: np.ones((8, 8), dtype=float),
        ai=lambda: object(),
        caked_transform_bundle=lambda: bundle,
        image_size=lambda: 8,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        detector_pixel_to_scattering_angles=lambda *_args: (_ for _ in ()).throw(
            AssertionError("analytic forward fallback should not be used")
        ),
    )

    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=8,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda idx: [
            {
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
                "hkl": (1, 1, 0),
                "caked_x": 12.0,
                "caked_y": 14.0,
            }
        ],
        load_background_by_index=lambda idx: (
            np.zeros((8, 8), dtype=np.float64),
            np.zeros((8, 8), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_source_rows_for_background=lambda *args, **kwargs: [
            {
                "sim_col_raw": 3.0,
                "sim_row_raw": 4.0,
                "hkl": (1, 1, 0),
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
            }
        ],
        geometry_manual_simulated_peaks_for_params=lambda params, *, prefer_cache: [],
        geometry_manual_simulated_lookup=lambda _simulated_peaks: {},
        geometry_manual_entry_display_coords=lambda entry: (
            float(entry.get("caked_x", 0.0)),
            float(entry.get("caked_y", 0.0)),
        ),
        geometry_manual_project_peaks_to_current_view=callbacks.project_peaks_to_current_view,
        unrotate_display_peaks=lambda entries, shape, *, k: [dict(entry) for entry in entries],
        display_to_native_sim_coords=lambda col, row, shape: (float(col), float(row)),
        native_detector_coords_to_bundle_detector_coords=lambda col, row: (
            call_order.append(
                ("native_detector_coords_to_bundle_detector_coords", float(col), float(row))
            )
            or (float(col) + 1.0, float(row) + 2.0)
        ),
        select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "yx",
                "label": "identity",
            },
            {"pairs": len(sim_pts)},
        ),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: list(entries),
        orient_image_for_fit=lambda image, **kwargs: image,
        pick_uses_caked_space=lambda: True,
        geometry_manual_caked_view_for_index=lambda idx: {
            "background_image": np.zeros((8, 8), dtype=float),
            "radial_axis": np.linspace(10.0, 17.0, 8),
            "azimuth_axis": np.linspace(-4.0, 3.0, 8),
            "transform_bundle": bundle,
        },
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={},
    )

    call_order.clear()
    projector = dataset["spec"]["fit_space_projector"]
    assert callable(projector)
    result = projector(
        np.asarray([3.0], dtype=np.float64),
        np.asarray([4.0], dtype=np.float64),
        local_params={"gamma": 0.0},
        anchor_kind="measured",
        input_frame="native_detector",
    )

    assert result["valid"] is True
    assert result["fit_space_source"] == "dataset_fit_space_projector"
    assert result["input_frame"] == "native_detector"
    assert result["fit_space_projector_kind"] == "exact_caked_bundle"
    assert result["native_frame_conversion_count"] == 0
    assert result["two_theta_deg"].tolist() == [22.5]
    assert result["phi_deg"].tolist() == [-35.5]
    assert call_order == [
        ("native_detector_coords_to_bundle_detector_coords", 3.0, 4.0),
        ("detector_pixel_to_caked_bin", 4.0, 6.0),
    ]


def test_build_geometry_manual_fit_dataset_exact_projector_converts_fit_detector_once(
    monkeypatch,
) -> None:
    bundle = _make_stub_caked_bundle(
        detector_shape=(8, 8),
        radial_axis=np.linspace(10.0, 17.0, 8),
        azimuth_axis=np.linspace(-4.0, 3.0, 8),
    )
    conversion_calls: list[tuple[float, float]] = []
    detector_calls: list[tuple[float, float]] = []

    monkeypatch.setattr(
        geometry_fit,
        "_fit_detector_coords_to_native_detector_coords",
        lambda col, row: (
            conversion_calls.append((float(col), float(row)))
            or (float(col) + 10.0, float(row) + 20.0)
        ),
    )
    monkeypatch.setattr(
        geometry_fit,
        "_geometry_fit_resolve_dynamic_reanchor_caked_bundle",
        lambda **kwargs: bundle,
    )
    monkeypatch.setattr(
        manual_geometry,
        "_detector_pixel_to_caked_bin",
        lambda live_bundle, col, row: (
            detector_calls.append((float(col), float(row)))
            or (
                (24.5, -34.5)
                if live_bundle is bundle and (float(col), float(row)) == (13.0, 24.0)
                else (21.5, -33.5)
                if live_bundle is bundle and (float(col), float(row)) == (3.0, 4.0)
                else (None, None)
            )
        ),
    )

    callbacks = manual_geometry.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
        last_caked_radial_values=lambda: np.linspace(10.0, 17.0, 8),
        last_caked_azimuth_values=lambda: np.linspace(-4.0, 3.0, 8),
        current_background_display=lambda: np.zeros((8, 8), dtype=float),
        current_background_native=lambda: np.ones((8, 8), dtype=float),
        ai=lambda: object(),
        caked_transform_bundle=lambda: bundle,
        image_size=lambda: 8,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        detector_pixel_to_scattering_angles=lambda *_args: (_ for _ in ()).throw(
            AssertionError("analytic forward fallback should not be used")
        ),
    )

    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=8,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda idx: [
            {
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
                "hkl": (1, 1, 0),
                "caked_x": 12.0,
                "caked_y": 14.0,
            }
        ],
        load_background_by_index=lambda idx: (
            np.zeros((8, 8), dtype=np.float64),
            np.zeros((8, 8), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_source_rows_for_background=lambda *args, **kwargs: [
            {
                "sim_col_raw": 3.0,
                "sim_row_raw": 4.0,
                "hkl": (1, 1, 0),
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
            }
        ],
        geometry_manual_simulated_peaks_for_params=lambda params, *, prefer_cache: [],
        geometry_manual_simulated_lookup=lambda _simulated_peaks: {},
        geometry_manual_entry_display_coords=lambda entry: (
            float(entry.get("caked_x", 0.0)),
            float(entry.get("caked_y", 0.0)),
        ),
        geometry_manual_project_peaks_to_current_view=callbacks.project_peaks_to_current_view,
        backend_detector_coords_to_native_detector_coords=lambda col, row, native_shape=None: (
            float(col),
            float(row),
        ),
        unrotate_display_peaks=lambda entries, shape, *, k: [dict(entry) for entry in entries],
        display_to_native_sim_coords=lambda col, row, shape: (float(col), float(row)),
        select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "yx",
                "label": "identity",
            },
            {"pairs": len(sim_pts)},
        ),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: list(entries),
        orient_image_for_fit=lambda image, **kwargs: image,
        pick_uses_caked_space=lambda: True,
        geometry_manual_caked_view_for_index=lambda idx: {
            "background_image": np.zeros((8, 8), dtype=float),
            "radial_axis": np.linspace(10.0, 17.0, 8),
            "azimuth_axis": np.linspace(-4.0, 3.0, 8),
            "transform_bundle": bundle,
        },
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={},
    )

    conversion_calls.clear()
    detector_calls.clear()
    projector = dataset["spec"]["fit_space_projector"]
    result = projector(
        np.asarray([3.0], dtype=np.float64),
        np.asarray([4.0], dtype=np.float64),
        local_params={"gamma": 0.0},
        anchor_kind="simulated",
        input_frame="fit_detector",
    )

    assert result["valid"] is True
    assert result["fit_space_source"] == "dataset_fit_space_projector"
    assert result["input_frame"] == "fit_detector"
    assert result["native_frame_conversion_count"] == 1
    assert conversion_calls == [(3.0, 4.0)]
    assert detector_calls == [(13.0, 24.0)]
    assert result["two_theta_deg"].tolist() == [24.5]
    assert result["phi_deg"].tolist() == [-34.5]


def test_build_geometry_manual_fit_dataset_exact_projector_uses_current_local_params(
    monkeypatch,
) -> None:
    bundle_a = _make_stub_caked_bundle(
        detector_shape=(8, 8),
        radial_axis=np.linspace(10.0, 17.0, 8),
        azimuth_axis=np.linspace(-4.0, 3.0, 8),
    )
    bundle_b = _make_stub_caked_bundle(
        detector_shape=(8, 8),
        radial_axis=np.linspace(11.0, 18.0, 8),
        azimuth_axis=np.linspace(-3.0, 4.0, 8),
    )

    monkeypatch.setattr(
        geometry_fit,
        "_geometry_fit_resolve_dynamic_reanchor_caked_bundle",
        lambda **kwargs: (
            bundle_b
            if float(dict(kwargs.get("params") or {}).get("gamma", 0.0)) > 0.5
            else bundle_a
        ),
    )
    monkeypatch.setattr(
        manual_geometry,
        "_detector_pixel_to_caked_bin",
        lambda live_bundle, col, row: (
            (22.0, -30.0)
            if live_bundle is bundle_a
            else (24.0, -32.0)
            if live_bundle is bundle_b
            else (None, None)
        ),
    )

    callbacks = manual_geometry.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
        last_caked_radial_values=lambda: np.linspace(10.0, 17.0, 8),
        last_caked_azimuth_values=lambda: np.linspace(-4.0, 3.0, 8),
        current_background_display=lambda: np.zeros((8, 8), dtype=float),
        current_background_native=lambda: np.ones((8, 8), dtype=float),
        ai=lambda: object(),
        caked_transform_bundle=lambda: bundle_a,
        image_size=lambda: 8,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(
            AssertionError("detector angular maps should not be used")
        ),
        detector_pixel_to_scattering_angles=lambda *_args: (_ for _ in ()).throw(
            AssertionError("analytic forward fallback should not be used")
        ),
    )

    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=8,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda idx: [
            {
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
                "hkl": (1, 1, 0),
                "caked_x": 12.0,
                "caked_y": 14.0,
            }
        ],
        load_background_by_index=lambda idx: (
            np.zeros((8, 8), dtype=np.float64),
            np.zeros((8, 8), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_source_rows_for_background=lambda *args, **kwargs: [
            {
                "sim_col_raw": 3.0,
                "sim_row_raw": 4.0,
                "hkl": (1, 1, 0),
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
            }
        ],
        geometry_manual_simulated_peaks_for_params=lambda params, *, prefer_cache: [],
        geometry_manual_simulated_lookup=lambda _simulated_peaks: {},
        geometry_manual_entry_display_coords=lambda entry: (
            float(entry.get("caked_x", 0.0)),
            float(entry.get("caked_y", 0.0)),
        ),
        geometry_manual_project_peaks_to_current_view=callbacks.project_peaks_to_current_view,
        unrotate_display_peaks=lambda entries, shape, *, k: [dict(entry) for entry in entries],
        display_to_native_sim_coords=lambda col, row, shape: (float(col), float(row)),
        select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "yx",
                "label": "identity",
            },
            {"pairs": len(sim_pts)},
        ),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: list(entries),
        orient_image_for_fit=lambda image, **kwargs: image,
        pick_uses_caked_space=lambda: True,
        geometry_manual_caked_view_for_index=lambda idx: {
            "background_image": np.zeros((8, 8), dtype=float),
            "radial_axis": np.linspace(10.0, 17.0, 8),
            "azimuth_axis": np.linspace(-4.0, 3.0, 8),
            "transform_bundle": bundle_a,
        },
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={},
    )

    projector = dataset["spec"]["fit_space_projector"]
    result_a = projector(
        np.asarray([3.0], dtype=np.float64),
        np.asarray([4.0], dtype=np.float64),
        local_params={"gamma": 0.0},
        anchor_kind="simulated",
        input_frame="native_detector",
    )
    result_b = projector(
        np.asarray([3.0], dtype=np.float64),
        np.asarray([4.0], dtype=np.float64),
        local_params={"gamma": 1.0},
        anchor_kind="simulated",
        input_frame="native_detector",
    )

    assert result_a["valid"] is True
    assert result_b["valid"] is True
    assert result_a["two_theta_deg"].tolist() == [22.0]
    assert result_b["two_theta_deg"].tolist() == [24.0]
    assert result_a["cake_bundle_signature"] != result_b["cake_bundle_signature"]


def test_copy_geometry_fit_dataset_spec_for_state_strips_fit_space_projector() -> None:
    copied = geometry_fit._copy_geometry_fit_dataset_spec_for_state(
        {
            "dataset_index": 0,
            "fit_space_projector": lambda *_args, **_kwargs: None,
            "fit_space_projector_kind": "exact_caked_bundle",
            "cake_bundle_signature": "sig-1",
        }
    )

    assert "fit_space_projector" not in copied
    assert copied["fit_space_projector_kind"] == "exact_caked_bundle"
    assert copied["cake_bundle_signature"] == "sig-1"


def test_build_geometry_manual_fit_dataset_refreshes_manual_pairs_from_saved_caked_angles() -> None:
    def _refresh_entry(entry):
        return manual_geometry.refresh_geometry_manual_pair_entry(
            entry,
            background_display_shape=(4, 5),
            background_display_to_native_detector_coords=lambda col, row: (
                float(col),
                float(row),
            ),
            caked_angles_to_background_display_coords=lambda two_theta, phi: (
                float(two_theta) - 10.0,
                float(phi) - 20.0,
            ),
            native_detector_coords_to_caked_display_coords=lambda col, row: (
                float(col) + 10.0,
                float(row) + 20.0,
            ),
            rotate_point_for_display=lambda col, row, _shape, _k: (
                float(col),
                float(row),
            ),
            display_rotate_k=0,
        )

    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=64,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda idx: [
            {
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
                "hkl": (1, 1, 0),
                "x": 30.0,
                "y": 40.0,
                "detector_x": 30.0,
                "detector_y": 40.0,
                "background_two_theta_deg": 23.0,
                "background_phi_deg": -36.0,
                "caked_x": 150.0,
                "caked_y": 160.0,
                "raw_caked_x": 151.0,
                "raw_caked_y": 161.0,
            }
        ],
        load_background_by_index=lambda idx: (
            np.zeros((4, 5), dtype=np.float64),
            np.zeros((4, 5), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_simulated_peaks_for_params=(
            lambda params, *, prefer_cache: [{"dummy": True}]
        ),
        geometry_manual_simulated_lookup=lambda _simulated_peaks: {
            (1, 2): {
                "sim_col": 91.0,
                "sim_row": 82.0,
                "sim_col_raw": 9.0,
                "sim_row_raw": 8.0,
            }
        },
        geometry_manual_entry_display_coords=lambda entry: (
            float(entry["x"]),
            float(entry["y"]),
        ),
        geometry_manual_refresh_pair_entry=_refresh_entry,
        unrotate_display_peaks=lambda entries, shape, *, k: list(entries),
        display_to_native_sim_coords=lambda col, row, shape: (11.0, 12.0),
        select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "yx",
                "label": "identity",
            },
            {"pairs": 1},
        ),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: list(entries),
        orient_image_for_fit=lambda image, **kwargs: image,
        pick_uses_caked_space=lambda: False,
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={},
    )

    assert dataset["initial_pairs_display"][0]["bg_display"] == (13.0, -56.0)
    assert dataset["initial_pairs_display"][0]["background_two_theta_deg"] == 23.0
    assert dataset["initial_pairs_display"][0]["background_phi_deg"] == -36.0
    assert dataset["measured_for_fit"][0]["background_two_theta_deg"] == 23.0
    assert dataset["measured_for_fit"][0]["background_phi_deg"] == -36.0
    assert dataset["measured_for_fit"][0]["background_detector_x"] == 13.0
    assert dataset["measured_for_fit"][0]["background_detector_y"] == -56.0
    assert dataset["spec"]["measured_peaks"][0]["background_two_theta_deg"] == 23.0
    assert dataset["spec"]["measured_peaks"][0]["background_phi_deg"] == -36.0


def test_build_geometry_manual_fit_dataset_uses_saved_refined_caked_coords_without_live_source() -> (
    None
):
    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=64,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda idx: [
            {
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
                "hkl": (1, 1, 0),
                "x": 30.0,
                "y": 40.0,
                "caked_x": 150.0,
                "caked_y": 160.0,
                "refined_sim_x": 9.0,
                "refined_sim_y": 8.0,
                "refined_sim_caked_x": 91.0,
                "refined_sim_caked_y": 82.0,
            }
        ],
        load_background_by_index=lambda idx: (
            np.zeros((4, 5), dtype=np.float64),
            np.zeros((4, 5), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_simulated_peaks_for_params=(
            lambda params, *, prefer_cache: [
                {
                    "q_group_key": ("other", 9),
                    "source_table_index": 10,
                    "source_row_index": 11,
                    "source_peak_index": 12,
                    "hkl": (9, 9, 9),
                    "sim_col": 1.0,
                    "sim_row": 2.0,
                }
            ]
        ),
        geometry_manual_simulated_lookup=lambda _simulated_peaks: {},
        geometry_manual_entry_display_coords=lambda entry: (150.0, 160.0),
        unrotate_display_peaks=lambda entries, shape, *, k: [{"x": 30.0, "y": 40.0}],
        display_to_native_sim_coords=lambda col, row, shape: (11.0, 12.0),
        select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "yx",
                "label": "identity",
            },
            {"pairs": 1},
        ),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: list(entries),
        orient_image_for_fit=lambda image, **kwargs: image,
        pick_uses_caked_space=lambda: True,
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={},
    )

    assert dataset["initial_pairs_display"][0]["sim_display"] == (91.0, 82.0)
    assert dataset["initial_pairs_display"][0]["sim_caked_display"] == (91.0, 82.0)
    assert dataset["initial_pairs_display"][0]["sim_native"] == (11.0, 12.0)


def test_build_geometry_manual_fit_dataset_prefers_current_view_overlay_fallback_in_caked_view() -> (
    None
):
    simulated_rows = [
        {
            "q_group_key": ("q", 1),
            "source_table_index": 0,
            "source_row_index": 0,
            "source_peak_index": 0,
            "hkl": (1, 1, 0),
            "display_col": 13.1,
            "display_row": 2.1,
            "caked_x": 13.1,
            "caked_y": 2.1,
            "sim_col": 500.0,
            "sim_row": 600.0,
            "sim_col_raw": 5.0,
            "sim_row_raw": 6.0,
        },
        {
            "q_group_key": ("q", 1),
            "source_table_index": 1,
            "source_row_index": 0,
            "source_peak_index": 0,
            "hkl": (1, 1, 0),
            "display_col": 30.0,
            "display_row": 40.0,
            "caked_x": 30.0,
            "caked_y": 40.0,
            "sim_col": 300.2,
            "sim_row": 400.2,
            "sim_col_raw": 7.0,
            "sim_row_raw": 8.0,
        },
    ]
    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=64,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda idx: [
            {
                "q_group_key": ("q", 1),
                "hkl": (1, 1, 0),
                "x": 300.0,
                "y": 400.0,
                "caked_x": 13.0,
                "caked_y": 2.0,
            }
        ],
        load_background_by_index=lambda idx: (
            np.zeros((4, 5), dtype=np.float64),
            np.zeros((4, 5), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_source_rows_for_background=(
            lambda idx, params, *, consumer, required_pairs=None: [
                dict(row) for row in simulated_rows
            ]
        ),
        geometry_manual_simulated_peaks_for_params=(  # pragma: no branch
            lambda params, *, prefer_cache: [dict(row) for row in simulated_rows]
        ),
        geometry_manual_simulated_lookup=lambda _simulated_peaks: {},
        geometry_manual_entry_display_coords=lambda entry: (
            float(entry.get("caked_x", entry.get("x", 0.0))),
            float(entry.get("caked_y", entry.get("y", 0.0))),
        ),
        unrotate_display_peaks=lambda entries, shape, *, k: [dict(entry) for entry in entries],
        display_to_native_sim_coords=lambda col, row, shape: (float(col), float(row)),
        select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "yx",
                "label": "identity",
            },
            {"pairs": len(sim_pts)},
        ),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: list(entries),
        orient_image_for_fit=lambda image, **kwargs: image,
        pick_uses_caked_space=lambda: True,
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={},
    )

    initial_entry = dataset["initial_pairs_display"][0]
    diag = dataset["source_resolution_diagnostics"][0]

    assert initial_entry["sim_display"] == (13.1, 2.1)
    assert initial_entry["sim_caked_display"] == (13.1, 2.1)
    assert diag["overlay_resolution_kind"] == "q_group_fallback"
    assert diag["overlay_source_row_key"] == (0, 0)
    assert diag["overlay_distance_px"] == pytest.approx(np.hypot(0.1, 0.1))


def test_build_geometry_manual_fit_dataset_prefers_detector_display_over_stale_sim_aliases() -> (
    None
):
    simulated_rows = [
        {
            "q_group_key": ("q", 1),
            "source_table_index": 0,
            "source_row_index": 0,
            "source_peak_index": 0,
            "hkl": (1, 1, 0),
            "display_col": 300.0,
            "display_row": -200.0,
            "sim_col": 500.0,
            "sim_row": 600.0,
            "sim_col_raw": 5.0,
            "sim_row_raw": 6.0,
        }
    ]
    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=64,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda idx: [
            {
                "q_group_key": ("q", 1),
                "hkl": (1, 1, 0),
                "x": 13.0,
                "y": 2.0,
                "refined_sim_x": 13.1,
                "refined_sim_y": 2.1,
            }
        ],
        load_background_by_index=lambda idx: (
            np.zeros((4, 5), dtype=np.float64),
            np.zeros((4, 5), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_source_rows_for_background=(
            lambda idx, params, *, consumer, required_pairs=None: [
                dict(row) for row in simulated_rows
            ]
        ),
        geometry_manual_simulated_peaks_for_params=(
            lambda params, *, prefer_cache: [dict(row) for row in simulated_rows]
        ),
        geometry_manual_simulated_lookup=lambda _simulated_peaks: {},
        geometry_manual_entry_display_coords=lambda entry: (
            float(entry.get("x", 0.0)),
            float(entry.get("y", 0.0)),
        ),
        unrotate_display_peaks=lambda entries, shape, *, k: [dict(entry) for entry in entries],
        display_to_native_sim_coords=lambda col, row, shape: (float(col), float(row)),
        select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "yx",
                "label": "identity",
            },
            {"pairs": len(sim_pts)},
        ),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: list(entries),
        orient_image_for_fit=lambda image, **kwargs: image,
        pick_uses_caked_space=lambda: False,
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={},
    )

    initial_entry = dataset["initial_pairs_display"][0]

    assert initial_entry["sim_display"] == (13.1, 2.1)


def test_build_geometry_manual_fit_dataset_rebuilds_missing_snapshot_and_enables_dynamic_reanchor(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}
    snapshot_diag = {"status": "snapshot_empty"}
    valid_source_row = {
        "q_group_key": ("q", 1),
        "source_table_index": 1,
        "source_row_index": 2,
        "source_peak_index": 0,
        "hkl": (1, 1, 0),
        "sim_col": 18.0,
        "sim_row": 19.0,
    }

    def _rebuild_source_rows(
        background_idx,
        params,
        *,
        consumer,
        prior_diagnostics,
        required_pairs,
    ):
        calls["rebuild"] = {
            "background_idx": int(background_idx),
            "params": dict(params),
            "consumer": consumer,
            "prior_diagnostics": dict(prior_diagnostics),
            "required_pairs": [dict(entry) for entry in (required_pairs or ())],
        }
        snapshot_diag.clear()
        snapshot_diag.update(
            {
                "status": "snapshot_hit",
                "cache_source": "source_snapshot_rebuild",
                "rebuild_source": "test_rebuild",
                "consumer": consumer,
            }
        )
        return [dict(valid_source_row)]

    def _build_background_context(image, cfg):
        calls["background_context"] = {
            "shape": tuple(np.asarray(image).shape),
            "cfg": dict(cfg),
        }
        return {"image_shape": tuple(np.asarray(image).shape)}

    monkeypatch.setattr(
        geometry_fit,
        "build_background_peak_context",
        _build_background_context,
    )

    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=64,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda idx: [
            {
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
                "source_peak_index": 0,
                "hkl": (1, 1, 0),
                "x": 30.0,
                "y": 40.0,
            }
        ],
        load_background_by_index=lambda idx: (
            np.zeros((6, 7), dtype=np.float64),
            np.zeros((6, 7), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_source_rows_for_background=(
            lambda idx, params, *, consumer, required_pairs=None: []
        ),
        geometry_manual_rebuild_source_rows_for_background=_rebuild_source_rows,
        geometry_manual_last_source_snapshot_diagnostics=lambda: dict(snapshot_diag),
        geometry_manual_simulated_peaks_for_params=(
            lambda params, *, prefer_cache: [dict(valid_source_row)]
        ),
        geometry_manual_simulated_lookup=lambda peaks: {(1, 2): dict(valid_source_row)},
        geometry_manual_entry_display_coords=lambda entry: (30.0, 40.0),
        unrotate_display_peaks=lambda entries, shape, *, k: [{"x": 30.0, "y": 40.0}],
        display_to_native_sim_coords=lambda col, row, shape: (float(col), float(row)),
        select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "xy",
                "label": "identity",
            },
            {"pairs": 1},
        ),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: list(entries),
        orient_image_for_fit=lambda image, **kwargs: np.asarray(image, dtype=np.float64),
        geometry_manual_match_config=lambda: {"search_radius": 4.0},
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={},
    )

    assert calls["rebuild"]["background_idx"] == 0
    assert calls["rebuild"]["consumer"] == "geometry_fit_dataset"
    assert calls["rebuild"]["prior_diagnostics"]["status"] == "snapshot_empty"
    assert calls["rebuild"]["required_pairs"] == [
        {
            "q_group_key": ("q", 1),
            "source_table_index": 1,
            "source_row_index": 2,
            "source_peak_index": 0,
            "hkl": (1, 1, 0),
            "x": 30.0,
            "y": 40.0,
        }
    ]
    assert calls["background_context"] == {
        "shape": (6, 7),
        "cfg": {"search_radius": 4.0},
    }
    assert dataset["resolved_source_pair_count"] == 1
    assert dataset["spec"]["dynamic_reanchor_enabled"] is True
    assert callable(dataset["spec"]["dynamic_reanchor_callback"])
    assert dataset["measured_for_fit"][0]["fit_detector_x"] == 30.0
    assert dataset["measured_for_fit"][0]["fit_detector_y"] == 40.0
    assert dataset["measured_for_fit"][0]["detector_input_frame"] == "fit_detector"
    assert dataset["cache_metadata"]["cache_source"] == "source_snapshot_rebuild"
    assert "rebuild_source:test_rebuild" in dataset["cache_metadata"]["cache_provenance"]


def test_geometry_fit_dynamic_reanchor_uses_background_detector_cache_as_raw_seed(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    def _fake_refine(
        candidate,
        raw_col,
        raw_row,
        *,
        display_background,
        cache_data,
        use_caked_space,
        radial_axis=None,
        azimuth_axis=None,
        match_simulated_peaks_to_peak_context,
    ):
        calls["candidate"] = dict(candidate)
        calls["raw"] = (float(raw_col), float(raw_row))
        calls["shape"] = tuple(np.asarray(display_background).shape)
        calls["cache_data"] = dict(cache_data)
        calls["use_caked_space"] = bool(use_caked_space)
        calls["radial_axis"] = radial_axis
        calls["azimuth_axis"] = azimuth_axis
        calls["matcher"] = match_simulated_peaks_to_peak_context
        return 44.0, 45.0

    monkeypatch.setattr(
        geometry_fit.gui_manual_geometry,
        "geometry_manual_refine_preview_point",
        _fake_refine,
    )
    monkeypatch.setattr(
        geometry_fit,
        "build_background_peak_context",
        lambda image, cfg: {
            "img_valid": True,
            "shape": tuple(np.asarray(image).shape),
            "cfg": dict(cfg),
        },
    )

    valid_source_row = {
        "q_group_key": ("q", 1),
        "source_table_index": 1,
        "source_row_index": 2,
        "source_peak_index": 0,
        "hkl": (1, 1, 0),
        "sim_col": 18.0,
        "sim_row": 19.0,
    }
    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=64,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda idx: [
            {
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
                "source_peak_index": 0,
                "hkl": (1, 1, 0),
                "x": 30.0,
                "y": 40.0,
            }
        ],
        load_background_by_index=lambda idx: (
            np.zeros((6, 7), dtype=np.float64),
            np.zeros((6, 7), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_source_rows_for_background=(
            lambda idx, params, *, consumer, required_pairs=None: [dict(valid_source_row)]
        ),
        geometry_manual_simulated_peaks_for_params=(
            lambda params, *, prefer_cache: [dict(valid_source_row)]
        ),
        geometry_manual_simulated_lookup=lambda peaks: {(1, 2): dict(valid_source_row)},
        geometry_manual_entry_display_coords=lambda entry: (30.0, 40.0),
        unrotate_display_peaks=lambda entries, shape, *, k: [{"x": 30.0, "y": 40.0}],
        display_to_native_sim_coords=lambda col, row, shape: (float(col), float(row)),
        select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "xy",
                "label": "identity",
            },
            {"pairs": 1},
        ),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: list(entries),
        orient_image_for_fit=lambda image, **kwargs: np.asarray(image, dtype=np.float64),
        geometry_manual_match_config=lambda: {"search_radius": 4.0},
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={},
    )
    callback = dataset["spec"]["dynamic_reanchor_callback"]
    result = callback(
        dict(dataset["measured_for_fit"][0]),
        (50.0, 51.0),
        local_params={"gamma": 0.0},
        dataset_ctx=SimpleNamespace(dataset_index=0),
    )

    assert result["detector_x"] == 44.0
    assert result["detector_y"] == 45.0
    assert calls["raw"] == (30.0, 40.0)
    assert calls["candidate"]["sim_col"] == 50.0
    assert calls["candidate"]["sim_row"] == 51.0
    assert calls["candidate"]["sim_col_local"] == 50.0
    assert calls["candidate"]["sim_row_local"] == 51.0
    assert calls["candidate"]["sim_col_global"] == 50.0
    assert calls["candidate"]["sim_row_global"] == 51.0
    assert calls["shape"] == (6, 7)
    assert calls["use_caked_space"] is False
    assert calls["radial_axis"] is None
    assert calls["azimuth_axis"] is None
    assert callable(calls["matcher"])


def test_geometry_fit_dynamic_reanchor_uses_exact_caked_bundle_without_analytic_fallback(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}
    radial_axis = np.linspace(20.0, 26.0, 7, dtype=np.float64)
    azimuth_axis = np.linspace(-179.5, 179.5, 6, dtype=np.float64)
    roundtripped_azimuth_axis = np.asarray(
        geometry_fit.raw_phi_to_gui_phi(
            np.sort(
                np.asarray(
                    geometry_fit.gui_phi_to_raw_phi(azimuth_axis),
                    dtype=np.float64,
                ),
                kind="stable",
            )
        ),
        dtype=np.float64,
    )
    roundtripped_azimuth_axis = roundtripped_azimuth_axis[
        np.argsort(roundtripped_azimuth_axis, kind="stable")
    ]
    assert not np.array_equal(azimuth_axis, roundtripped_azimuth_axis)

    def _fake_refine(
        candidate,
        raw_col,
        raw_row,
        *,
        display_background,
        cache_data,
        use_caked_space,
        radial_axis=None,
        azimuth_axis=None,
        match_simulated_peaks_to_peak_context,
    ):
        calls["candidate"] = dict(candidate)
        calls["raw"] = (float(raw_col), float(raw_row))
        calls["shape"] = tuple(np.asarray(display_background).shape)
        calls["cache_data"] = dict(cache_data)
        calls["use_caked_space"] = bool(use_caked_space)
        calls["radial_axis"] = np.asarray(radial_axis, dtype=np.float64)
        calls["azimuth_axis"] = np.asarray(azimuth_axis, dtype=np.float64)
        calls["matcher"] = match_simulated_peaks_to_peak_context
        return 22.5, -35.5

    monkeypatch.setattr(
        geometry_fit.gui_manual_geometry,
        "geometry_manual_refine_preview_point",
        _fake_refine,
    )
    monkeypatch.setattr(
        geometry_fit,
        "build_background_peak_context",
        lambda image, cfg: {
            "img_valid": True,
            "shape": tuple(np.asarray(image).shape),
            "cfg": dict(cfg),
        },
    )
    monkeypatch.setattr(
        geometry_fit,
        "_detector_pixels_to_fit_space",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("analytic caked fallback should not run")
        ),
    )
    bundle = _make_stub_caked_bundle(
        detector_shape=(6, 7),
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
    )
    projector_calls: list[tuple[object, float, float]] = []
    monkeypatch.setattr(
        geometry_fit.gui_manual_geometry,
        "_detector_pixel_to_caked_bin",
        lambda transform_bundle, col, row: (
            projector_calls.append((transform_bundle, float(col), float(row)))
            or {(50.0, 51.0): (24.5, -34.5)}.get(
                (float(col), float(row)),
                (None, None),
            )
        ),
    )

    valid_source_row = {
        "q_group_key": ("q", 1),
        "source_table_index": 1,
        "source_row_index": 2,
        "source_peak_index": 0,
        "hkl": (1, 1, 0),
        "sim_col": 18.0,
        "sim_row": 19.0,
    }
    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=64,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda idx: [
            {
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
                "source_peak_index": 0,
                "hkl": (1, 1, 0),
                "x": 30.0,
                "y": 40.0,
                "caked_x": 150.0,
                "caked_y": 160.0,
                "background_two_theta_deg": 23.0,
                "background_phi_deg": -36.0,
            }
        ],
        load_background_by_index=lambda idx: (
            np.zeros((6, 7), dtype=np.float64),
            np.zeros((6, 7), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_source_rows_for_background=(
            lambda idx, params, *, consumer, required_pairs=None: [dict(valid_source_row)]
        ),
        geometry_manual_simulated_peaks_for_params=(
            lambda params, *, prefer_cache: [dict(valid_source_row)]
        ),
        geometry_manual_simulated_lookup=lambda peaks: {(1, 2): dict(valid_source_row)},
        geometry_manual_entry_display_coords=lambda entry: (30.0, 40.0),
        unrotate_display_peaks=lambda entries, shape, *, k: [{"x": 30.0, "y": 40.0}],
        display_to_native_sim_coords=lambda col, row, shape: (float(col), float(row)),
        select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "xy",
                "label": "identity",
            },
            {"pairs": 1},
        ),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: list(entries),
        orient_image_for_fit=lambda image, **kwargs: np.asarray(image, dtype=np.float64),
        geometry_manual_match_config=lambda: {"search_radius": 4.0},
        geometry_manual_caked_view_for_index=lambda idx: {
            "background": np.zeros((6, 7), dtype=np.float64),
            "radial_axis": radial_axis,
            "azimuth_axis": azimuth_axis,
            "raw_azimuth_axis": np.asarray(bundle.raw_azimuth_deg, dtype=np.float64),
            "transform_bundle": bundle,
        },
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={},
    )
    callback = dataset["spec"]["dynamic_reanchor_callback"]
    local_params = {
        "center": [32.0, 32.0],
        "corto_detector": 100.0,
        "pixel_size": 1.0,
        "gamma": 0.0,
        "Gamma": 0.0,
    }
    result = callback(
        dict(dataset["measured_for_fit"][0]),
        (50.0, 51.0),
        local_params=local_params,
        dataset_ctx=SimpleNamespace(dataset_index=0),
    )
    assert calls["raw"] == (150.0, 160.0)
    assert calls["shape"] == (6, 7)
    assert calls["use_caked_space"] is True
    assert np.allclose(calls["radial_axis"], radial_axis)
    assert np.allclose(calls["azimuth_axis"], azimuth_axis)
    assert calls["candidate"]["sim_col"] == pytest.approx(24.5)
    assert calls["candidate"]["sim_row"] == pytest.approx(-34.5)
    assert np.isfinite(float(calls["candidate"]["sim_col_local"]))
    assert np.isfinite(float(calls["candidate"]["sim_row_local"]))
    assert projector_calls == [(bundle, 50.0, 51.0)]
    assert callable(calls["matcher"])
    assert result["background_two_theta_deg"] == pytest.approx(22.5)
    assert result["background_phi_deg"] == pytest.approx(-35.5)
    assert result["fit_space_anchor_override"] is True
    assert result["measured_reanchor_motion_px"] == pytest.approx(0.0)


def test_geometry_fit_dynamic_reanchor_projects_detector_click_into_caked_seed_when_missing_cached_angles(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}
    radial_axis = np.linspace(20.0, 26.0, 7, dtype=np.float64)
    azimuth_axis = np.linspace(-40.0, -35.0, 6, dtype=np.float64)

    def _fake_refine(
        candidate,
        raw_col,
        raw_row,
        *,
        display_background,
        cache_data,
        use_caked_space,
        radial_axis=None,
        azimuth_axis=None,
        match_simulated_peaks_to_peak_context,
    ):
        calls["candidate"] = dict(candidate)
        calls["raw"] = (float(raw_col), float(raw_row))
        calls["shape"] = tuple(np.asarray(display_background).shape)
        calls["cache_data"] = dict(cache_data)
        calls["use_caked_space"] = bool(use_caked_space)
        calls["radial_axis"] = np.asarray(radial_axis, dtype=np.float64)
        calls["azimuth_axis"] = np.asarray(azimuth_axis, dtype=np.float64)
        calls["matcher"] = match_simulated_peaks_to_peak_context
        return 22.5, -35.5

    monkeypatch.setattr(
        geometry_fit.gui_manual_geometry,
        "geometry_manual_refine_preview_point",
        _fake_refine,
    )
    monkeypatch.setattr(
        geometry_fit,
        "build_background_peak_context",
        lambda image, cfg: {
            "img_valid": True,
            "shape": tuple(np.asarray(image).shape),
            "cfg": dict(cfg),
        },
    )
    bundle = _make_stub_caked_bundle(
        detector_shape=(6, 7),
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
    )
    projector_calls: list[tuple[object, float, float]] = []
    monkeypatch.setattr(
        geometry_fit.gui_manual_geometry,
        "_detector_pixel_to_caked_bin",
        lambda transform_bundle, col, row: (
            projector_calls.append((transform_bundle, float(col), float(row)))
            or {
                (50.0, 51.0): (24.5, -34.5),
                (30.0, 40.0): (22.25, -36.75),
            }.get(
                (float(col), float(row)),
                (None, None),
            )
        ),
    )

    valid_source_row = {
        "q_group_key": ("q", 1),
        "source_table_index": 1,
        "source_row_index": 2,
        "source_peak_index": 0,
        "hkl": (1, 1, 0),
        "sim_col": 18.0,
        "sim_row": 19.0,
    }
    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=64,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda idx: [
            {
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
                "source_peak_index": 0,
                "hkl": (1, 1, 0),
                "x": 30.0,
                "y": 40.0,
            }
        ],
        load_background_by_index=lambda idx: (
            np.zeros((6, 7), dtype=np.float64),
            np.zeros((6, 7), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_source_rows_for_background=(
            lambda idx, params, *, consumer, required_pairs=None: [dict(valid_source_row)]
        ),
        geometry_manual_simulated_peaks_for_params=(
            lambda params, *, prefer_cache: [dict(valid_source_row)]
        ),
        geometry_manual_simulated_lookup=lambda peaks: {(1, 2): dict(valid_source_row)},
        geometry_manual_entry_display_coords=lambda entry: (30.0, 40.0),
        unrotate_display_peaks=lambda entries, shape, *, k: [{"x": 30.0, "y": 40.0}],
        display_to_native_sim_coords=lambda col, row, shape: (float(col), float(row)),
        select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "xy",
                "label": "identity",
            },
            {"pairs": 1},
        ),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: list(entries),
        orient_image_for_fit=lambda image, **kwargs: np.asarray(image, dtype=np.float64),
        geometry_manual_match_config=lambda: {"search_radius": 4.0},
        geometry_manual_caked_view_for_index=lambda idx: {
            "background": np.zeros((6, 7), dtype=np.float64),
            "radial_axis": radial_axis,
            "azimuth_axis": azimuth_axis,
            "transform_bundle": bundle,
        },
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={},
    )
    callback = dataset["spec"]["dynamic_reanchor_callback"]
    local_params = {
        "center": [32.0, 32.0],
        "corto_detector": 100.0,
        "pixel_size": 1.0,
        "gamma": 0.0,
        "Gamma": 0.0,
    }
    result = callback(
        dict(dataset["measured_for_fit"][0]),
        (50.0, 51.0),
        local_params=local_params,
        dataset_ctx=SimpleNamespace(dataset_index=0),
    )
    assert calls["shape"] == (6, 7)
    assert calls["use_caked_space"] is True
    assert np.allclose(calls["radial_axis"], radial_axis)
    assert np.allclose(calls["azimuth_axis"], azimuth_axis)
    assert calls["candidate"]["sim_col"] == pytest.approx(24.5)
    assert calls["candidate"]["sim_row"] == pytest.approx(-34.5)
    assert calls["raw"][0] == pytest.approx(22.25)
    assert calls["raw"][1] == pytest.approx(-36.75)
    assert projector_calls == [
        (bundle, 50.0, 51.0),
        (bundle, 30.0, 40.0),
    ]
    assert callable(calls["matcher"])
    assert result["background_two_theta_deg"] == pytest.approx(22.5)
    assert result["background_phi_deg"] == pytest.approx(-35.5)
    assert result["fit_space_anchor_override"] is True


@pytest.mark.parametrize(
    "bundle_factory",
    [
        lambda radial_axis, azimuth_axis: None,
        (
            lambda radial_axis, azimuth_axis: geometry_fit.CakeTransformBundle(
                detector_shape=(9, 9),
                radial_deg=np.asarray(radial_axis, dtype=np.float64).copy(),
                raw_azimuth_deg=np.asarray(
                    geometry_fit.gui_phi_to_raw_phi(azimuth_axis),
                    dtype=np.float64,
                ).copy(),
                gui_azimuth_deg=np.asarray(azimuth_axis, dtype=np.float64).copy(),
                lut=object(),
            )
        ),
    ],
    ids=["missing_bundle", "invalid_bundle"],
)
def test_geometry_fit_dynamic_reanchor_missing_or_invalid_bundle_falls_back_to_detector_space(
    monkeypatch,
    bundle_factory,
) -> None:
    calls: dict[str, object] = {}
    radial_axis = np.linspace(20.0, 26.0, 7, dtype=np.float64)
    azimuth_axis = np.linspace(-40.0, -35.0, 6, dtype=np.float64)
    raw_azimuth_axis = np.asarray(
        geometry_fit.gui_phi_to_raw_phi(azimuth_axis),
        dtype=np.float64,
    )
    rebuild_calls: list[tuple[tuple[int, int], int, int]] = []

    def _fake_refine(
        candidate,
        raw_col,
        raw_row,
        *,
        display_background,
        cache_data,
        use_caked_space,
        radial_axis=None,
        azimuth_axis=None,
        match_simulated_peaks_to_peak_context,
    ):
        calls["candidate"] = dict(candidate)
        calls["raw"] = (float(raw_col), float(raw_row))
        calls["shape"] = tuple(np.asarray(display_background).shape)
        calls["cache_data"] = dict(cache_data)
        calls["use_caked_space"] = bool(use_caked_space)
        calls["radial_axis"] = radial_axis
        calls["azimuth_axis"] = azimuth_axis
        calls["matcher"] = match_simulated_peaks_to_peak_context
        return 44.0, 45.0

    monkeypatch.setattr(
        geometry_fit.gui_manual_geometry,
        "geometry_manual_refine_preview_point",
        _fake_refine,
    )
    monkeypatch.setattr(
        geometry_fit,
        "build_background_peak_context",
        lambda image, cfg: {
            "img_valid": True,
            "shape": tuple(np.asarray(image).shape),
            "cfg": dict(cfg),
        },
    )
    monkeypatch.setattr(
        geometry_fit,
        "resolve_cake_transform_bundle",
        lambda ai, detector_shape, radial_deg, *, raw_azimuth_deg=None, **_kwargs: (
            (
                rebuild_calls.append(
                    (
                        tuple(int(v) for v in tuple(detector_shape)[:2]),
                        int(np.asarray(radial_deg, dtype=np.float64).size),
                        int(np.asarray(raw_azimuth_deg, dtype=np.float64).size),
                    )
                )
                or None
            )
            if ai is not None
            else None
        ),
    )
    monkeypatch.setattr(
        geometry_fit.gui_manual_geometry,
        "_detector_pixel_to_caked_bin",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("caked projection should not run without a valid bundle")
        ),
    )

    valid_source_row = {
        "q_group_key": ("q", 1),
        "source_table_index": 1,
        "source_row_index": 2,
        "source_peak_index": 0,
        "hkl": (1, 1, 0),
        "sim_col": 18.0,
        "sim_row": 19.0,
    }
    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=64,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda idx: [
            {
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
                "source_peak_index": 0,
                "hkl": (1, 1, 0),
                "x": 30.0,
                "y": 40.0,
            }
        ],
        load_background_by_index=lambda idx: (
            np.zeros((6, 7), dtype=np.float64),
            np.zeros((6, 7), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_source_rows_for_background=(
            lambda idx, params, *, consumer, required_pairs=None: [dict(valid_source_row)]
        ),
        geometry_manual_simulated_peaks_for_params=(
            lambda params, *, prefer_cache: [dict(valid_source_row)]
        ),
        geometry_manual_simulated_lookup=lambda peaks: {(1, 2): dict(valid_source_row)},
        geometry_manual_entry_display_coords=lambda entry: (30.0, 40.0),
        unrotate_display_peaks=lambda entries, shape, *, k: [{"x": 30.0, "y": 40.0}],
        display_to_native_sim_coords=lambda col, row, shape: (float(col), float(row)),
        select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "xy",
                "label": "identity",
            },
            {"pairs": 1},
        ),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: list(entries),
        orient_image_for_fit=lambda image, **kwargs: np.asarray(image, dtype=np.float64),
        geometry_manual_match_config=lambda: {"search_radius": 4.0},
        geometry_manual_caked_view_for_index=lambda idx: {
            "background": np.ones((6, 7), dtype=np.float64),
            "radial_axis": radial_axis,
            "azimuth_axis": azimuth_axis,
            "raw_azimuth_axis": raw_azimuth_axis,
            "transform_bundle": bundle_factory(radial_axis, azimuth_axis),
        },
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={},
    )
    callback = dataset["spec"]["dynamic_reanchor_callback"]
    result = callback(
        dict(dataset["measured_for_fit"][0]),
        (50.0, 51.0),
        local_params={
            "center": [32.0, 32.0],
            "corto_detector": 100.0,
            "pixel_size": 1.0,
        },
        dataset_ctx=SimpleNamespace(dataset_index=0),
    )

    assert result["detector_x"] == 44.0
    assert result["detector_y"] == 45.0
    assert "fit_space_anchor_override" not in result
    assert calls["shape"] == (6, 7)
    assert calls["raw"] == (30.0, 40.0)
    assert calls["candidate"]["sim_col"] == 50.0
    assert calls["candidate"]["sim_row"] == 51.0
    assert calls["use_caked_space"] is False
    assert calls["radial_axis"] is None
    assert calls["azimuth_axis"] is None
    assert callable(calls["matcher"])
    assert rebuild_calls == [((6, 7), 7, 6)]


def test_geometry_fit_dynamic_reanchor_projects_lut_in_native_detector_coords(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}
    radial_axis = np.linspace(20.0, 26.0, 7, dtype=np.float64)
    azimuth_axis = np.linspace(-40.0, -35.0, 6, dtype=np.float64)
    orientation_choice = {
        "indexing_mode": "xy",
        "k": 1,
        "flip_x": False,
        "flip_y": False,
        "flip_order": "yx",
        "label": "rot90-ccw",
    }
    orientation_transform = {
        key: orientation_choice[key]
        for key in ("indexing_mode", "k", "flip_x", "flip_y", "flip_order")
    }
    backend_shape = (6, 7)
    sim_backend_point = (1.0, 2.0)
    measured_backend_point = (3.0, 4.0)
    sim_fit_point = geometry_overlay.transform_points_orientation(
        [sim_backend_point],
        backend_shape,
        **orientation_transform,
    )[0]
    measured_fit_point = geometry_overlay.transform_points_orientation(
        [measured_backend_point],
        backend_shape,
        **orientation_transform,
    )[0]

    def _fake_refine(
        candidate,
        raw_col,
        raw_row,
        *,
        display_background,
        cache_data,
        use_caked_space,
        radial_axis=None,
        azimuth_axis=None,
        match_simulated_peaks_to_peak_context,
    ):
        calls["candidate"] = dict(candidate)
        calls["raw"] = (float(raw_col), float(raw_row))
        calls["shape"] = tuple(np.asarray(display_background).shape)
        calls["cache_data"] = dict(cache_data)
        calls["use_caked_space"] = bool(use_caked_space)
        calls["radial_axis"] = np.asarray(radial_axis, dtype=np.float64)
        calls["azimuth_axis"] = np.asarray(azimuth_axis, dtype=np.float64)
        calls["matcher"] = match_simulated_peaks_to_peak_context
        return 22.5, -35.5

    monkeypatch.setattr(
        geometry_fit.gui_manual_geometry,
        "geometry_manual_refine_preview_point",
        _fake_refine,
    )
    monkeypatch.setattr(
        geometry_fit,
        "build_background_peak_context",
        lambda image, cfg: {
            "img_valid": True,
            "shape": tuple(np.asarray(image).shape),
            "cfg": dict(cfg),
        },
    )
    bundle = _make_stub_caked_bundle(
        detector_shape=(6, 7),
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
    )
    projector_calls: list[tuple[object, float, float]] = []
    monkeypatch.setattr(
        geometry_fit.gui_manual_geometry,
        "_detector_pixel_to_caked_bin",
        lambda transform_bundle, col, row: (
            projector_calls.append((transform_bundle, float(col), float(row)))
            or {
                (50.0, 51.0): (24.5, -34.5),
                (30.0, 40.0): (22.25, -36.75),
            }.get(
                (float(col), float(row)),
                (None, None),
            )
        ),
    )

    backend_inverse_calls: list[tuple[float, float, tuple[int, int] | None]] = []
    backend_to_native = {
        sim_backend_point: (50.0, 51.0),
        measured_backend_point: (30.0, 40.0),
    }
    valid_source_row = {
        "q_group_key": ("q", 1),
        "source_table_index": 1,
        "source_row_index": 2,
        "source_peak_index": 0,
        "hkl": (1, 1, 0),
        "sim_col": 18.0,
        "sim_row": 19.0,
    }
    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=64,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda idx: [
            {
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
                "source_peak_index": 0,
                "hkl": (1, 1, 0),
                "x": 30.0,
                "y": 40.0,
            }
        ],
        load_background_by_index=lambda idx: (
            np.zeros((6, 7), dtype=np.float64),
            np.zeros((6, 7), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: np.asarray(
            image,
            dtype=np.float64,
        ),
        backend_detector_coords_to_native_detector_coords=lambda col, row, native_shape=None: (
            backend_inverse_calls.append(
                (
                    float(col),
                    float(row),
                    (
                        tuple(int(v) for v in tuple(native_shape)[:2])
                        if native_shape is not None
                        else None
                    ),
                )
            )
            or backend_to_native.get((float(col), float(row)), (None, None))
        ),
        geometry_manual_source_rows_for_background=(
            lambda idx, params, *, consumer, required_pairs=None: [dict(valid_source_row)]
        ),
        geometry_manual_simulated_peaks_for_params=(
            lambda params, *, prefer_cache: [dict(valid_source_row)]
        ),
        geometry_manual_simulated_lookup=lambda peaks: {(1, 2): dict(valid_source_row)},
        geometry_manual_entry_display_coords=lambda entry: (30.0, 40.0),
        unrotate_display_peaks=lambda entries, shape, *, k: [{"x": 30.0, "y": 40.0}],
        display_to_native_sim_coords=lambda col, row, shape: (float(col), float(row)),
        select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
            dict(orientation_choice),
            {"pairs": 1},
        ),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: list(entries),
        orient_image_for_fit=lambda image, **kwargs: np.asarray(image, dtype=np.float64),
        geometry_manual_match_config=lambda: {"search_radius": 4.0},
        geometry_manual_caked_view_for_index=lambda idx: {
            "background": np.zeros((6, 7), dtype=np.float64),
            "radial_axis": radial_axis,
            "azimuth_axis": azimuth_axis,
            "transform_bundle": bundle,
        },
    )

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.0},
        manual_dataset_bindings=manual_dataset_bindings,
        orientation_cfg={},
    )
    callback = dataset["spec"]["dynamic_reanchor_callback"]
    local_params = {
        "center": [32.0, 32.0],
        "corto_detector": 100.0,
        "pixel_size": 1.0,
        "gamma": 0.0,
        "Gamma": 0.0,
    }
    measured_entry = dict(dataset["measured_for_fit"][0])
    measured_entry.pop("background_detector_x", None)
    measured_entry.pop("background_detector_y", None)
    measured_entry["detector_x"] = float(measured_fit_point[0])
    measured_entry["detector_y"] = float(measured_fit_point[1])
    measured_entry["fit_detector_x"] = float(measured_fit_point[0])
    measured_entry["fit_detector_y"] = float(measured_fit_point[1])

    result = callback(
        measured_entry,
        sim_fit_point,
        local_params=local_params,
        dataset_ctx=SimpleNamespace(dataset_index=0),
    )

    assert backend_inverse_calls == [
        (1.0, 2.0, (6, 7)),
        (float(measured_backend_point[0]), float(measured_backend_point[1]), (6, 7)),
    ]
    assert projector_calls == [
        (bundle, 50.0, 51.0),
        (bundle, 30.0, 40.0),
    ]
    assert calls["raw"] == (22.25, -36.75)
    assert calls["shape"] == (6, 7)
    assert calls["use_caked_space"] is True
    assert np.allclose(calls["radial_axis"], radial_axis)
    assert np.allclose(calls["azimuth_axis"], azimuth_axis)
    assert calls["candidate"]["sim_col"] == pytest.approx(24.5)
    assert calls["candidate"]["sim_row"] == pytest.approx(-34.5)
    assert callable(calls["matcher"])
    assert result["background_two_theta_deg"] == pytest.approx(22.5)
    assert result["background_phi_deg"] == pytest.approx(-35.5)
    assert result["fit_space_anchor_override"] is True


def test_rebuild_geometry_fit_source_rows_preserves_runtime_failure_status() -> None:
    runtime_diag = {
        "stage": "simulate_hit_tables",
        "status": "empty_hit_tables",
        "miller_shape": [96, 3],
        "intensity_shape": [96],
        "image_size": 2048,
        "param_summary": {
            "a": 4.143,
            "c": 28.64,
            "lambda": 1.54,
            "theta_initial": 5.0,
        },
        "mosaic_array_sizes": {
            "beam_x_array": 489,
            "beam_y_array": 489,
            "theta_array": 489,
            "phi_array": 489,
            "wavelength_array": 489,
        },
        "hit_table_count": 0,
        "nonempty_hit_table_count": 0,
        "finite_hit_row_total": 0,
        "hit_row_counts": [],
    }

    result = geometry_fit.rebuild_geometry_fit_source_rows(
        background_index=1,
        background_label="bg1.osc",
        params_local={"a": 4.143, "c": 28.64},
        consumer="geometry_fit_dataset",
        prior_diagnostics={"status": "snapshot_empty"},
        requested_signature=("sig", 1),
        requested_signature_summary="sig-summary",
        can_use_live_runtime_cache=False,
        build_live_rows=None,
        get_memory_intersection_cache=None,
        load_logged_intersection_cache=None,
        logged_cache_matches_params=None,
        build_source_rows_from_hit_tables=lambda _tables: ([], [], []),
        simulate_hit_tables=lambda _params: [],
        last_runtime_simulation_diagnostics=lambda: dict(runtime_diag),
        project_rows=None,
        live_cache_inventory={"source_snapshot_count": 0},
    )

    assert result.stored_rows == []
    assert result.diagnostics["status"] == "empty_hit_tables"
    assert result.diagnostics["runtime_simulation"]["status"] == "empty_hit_tables"
    assert result.diagnostics["miller_shape"] == [96, 3]
    assert result.diagnostics["intensity_shape"] == [96]
    assert result.diagnostics["image_size"] == 2048


def test_rebuild_geometry_fit_source_rows_rejects_invalid_live_cache_for_required_pair_and_records_dual_path_diff() -> (
    None
):
    live_rows = [
        {
            "hkl": (1, 0, 0),
            "q_group_key": ("q", 1),
            "source_table_index": 5,
            "source_row_index": 0,
            "source_peak_index": 13,
            "sim_col": 1.0,
            "sim_row": 2.0,
        }
    ]
    rebuilt_rows = [
        {
            "hkl": (1, 0, 0),
            "q_group_key": ("q", 1),
            "source_table_index": 0,
            "source_reflection_index": 7,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "sim_col": 10.0,
            "sim_row": 20.0,
        }
    ]

    result = geometry_fit.rebuild_geometry_fit_source_rows(
        background_index=0,
        background_label="bg0.osc",
        params_local={"a": 4.143, "c": 28.64},
        consumer="geometry_fit_dataset",
        prior_diagnostics={"status": "snapshot_hit"},
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        can_use_live_runtime_cache=True,
        build_live_rows=lambda: list(live_rows),
        get_memory_intersection_cache=lambda: [],
        load_logged_intersection_cache=lambda: ([], None),
        logged_cache_matches_params=lambda _meta, _params: False,
        build_source_rows_from_hit_tables=lambda _tables: (
            list(rebuilt_rows),
            None,
            None,
            None,
        ),
        simulate_hit_tables=lambda _params: [object()],
        last_runtime_simulation_diagnostics=lambda: {"status": "success"},
        project_rows=None,
        required_pairs=[
            {
                "pair_id": "bg0:pair0",
                "overlay_match_index": 0,
                "hkl": (1, 0, 0),
                "q_group_key": ("q", 1),
                "source_table_index": 5,
                "source_row_index": 0,
                "source_peak_index": 13,
            }
        ],
        live_cache_inventory={"source_snapshot_count": 1},
    )

    assert result.rebuild_source == "fresh_simulation"
    assert isinstance(result.metadata, dict)
    validation = result.metadata["live_runtime_cache_validation"]
    assert validation["valid"] is False
    assert validation["pair_failures"][0]["pair_id"] == "bg0:pair0"
    assert "branch_candidates" in validation["pair_failures"][0]
    rebuilt_validation = result.metadata["live_runtime_cache_rebuild_validation"]
    assert rebuilt_validation["valid"] is True
    assert rebuilt_validation["resolved_pairs"][0]["pair_id"] == "bg0:pair0"
    diff_entries = result.metadata["live_runtime_cache_dual_path_diff"]
    assert diff_entries == [
        {
            "pair_id": "bg0:pair0",
            "before_status": "failed",
            "after_status": "resolved",
            "before_canonical_identity": None,
            "after_canonical_identity": [7, 1],
            "before_simulated_point": None,
            "after_simulated_point": [10.0, 20.0],
            "before_reason": "missing_canonical_candidate",
            "after_reason": None,
        }
    ]


def test_rebuild_geometry_fit_source_rows_emits_stage_callback_for_accepted_live_runtime_cache() -> (
    None
):
    live_rows = [
        {
            "hkl": (1, 0, 0),
            "q_group_key": ("q", 1),
            "source_reflection_index": 7,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "sim_col": 10.0,
            "sim_row": 20.0,
        }
    ]
    stage_events: list[tuple[str, dict[str, object]]] = []

    result = geometry_fit.rebuild_geometry_fit_source_rows(
        background_index=0,
        background_label="bg0.osc",
        params_local={"a": 4.143, "c": 28.64},
        consumer="geometry_fit_dataset",
        prior_diagnostics={"status": "snapshot_empty"},
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        can_use_live_runtime_cache=True,
        build_live_rows=lambda: list(live_rows),
        get_memory_intersection_cache=lambda: pytest.fail(
            "accepted live runtime cache should not touch memory cache"
        ),
        load_logged_intersection_cache=lambda: pytest.fail(
            "accepted live runtime cache should not touch logged cache"
        ),
        logged_cache_matches_params=lambda _meta, _params: pytest.fail(
            "accepted live runtime cache should not check logged cache params"
        ),
        build_source_rows_from_hit_tables=lambda _tables: pytest.fail(
            "accepted live runtime cache should not rebuild source rows"
        ),
        simulate_hit_tables=lambda _params: pytest.fail(
            "accepted live runtime cache should not run fresh simulation"
        ),
        last_runtime_simulation_diagnostics=lambda: {"status": "unused"},
        project_rows=None,
        required_pairs=[
            {
                "pair_id": "bg0:pair0",
                "overlay_match_index": 0,
                "hkl": (1, 0, 0),
                "q_group_key": ("q", 1),
                "source_reflection_index": 7,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "source_row_index": 0,
                "source_branch_index": 1,
                "source_peak_index": 1,
            }
        ],
        live_cache_inventory={"source_snapshot_count": 1},
        stage_callback=lambda stage, payload: stage_events.append((str(stage), dict(payload))),
    )

    assert result.rebuild_source == "live_runtime_cache"
    assert [stage for stage, _payload in stage_events] == [
        "source_cache_target_collection_start",
        "source_cache_target_collection_ready",
        "source_cache_targeted_projected_cache_start",
        "source_cache_targeted_projected_cache_miss",
        "source_cache_live_runtime_cache_validation_start",
        "source_cache_live_runtime_cache_validation_ready",
        "source_cache_live_runtime_cache_accepted",
        "source_cache_project_rows_start",
        "source_cache_project_rows_ready",
    ]
    accepted_payload = stage_events[6][1]
    assert accepted_payload["row_count"] == 1
    assert accepted_payload["required_pair_count"] == 1
    assert accepted_payload["validated_pair_count"] == 1


def test_validate_geometry_fit_live_source_rows_rejection_counts_follow_hkl_then_branch() -> None:
    validation = geometry_fit.validate_geometry_fit_live_source_rows(
        [
            {
                "hkl": (1, 0, 0),
                "q_group_key": ("q", 1),
                "source_reflection_index": 7,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "source_row_index": 0,
                "source_branch_index": 1,
                "source_peak_index": 0,
            },
            {
                "hkl": (2, 0, 0),
                "q_group_key": ("q", 2),
                "source_reflection_index": 8,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "source_row_index": 1,
                "source_branch_index": 1,
                "source_peak_index": 1,
            },
        ],
        required_pairs=[
            {
                "pair_id": "hkl-miss",
                "overlay_match_index": 0,
                "hkl": (9, 9, 9),
                "q_group_key": ("q", 1),
                "source_reflection_index": 70,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "source_branch_index": 1,
            },
            {
                "pair_id": "branch-miss",
                "overlay_match_index": 1,
                "hkl": (2, 0, 0),
                "q_group_key": ("q", 2),
                "source_reflection_index": 80,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "source_branch_index": 0,
            },
        ],
    )

    assert validation["valid"] is False
    assert validation["missing_required_pair_count"] == 2
    assert validation["hkl_missing_candidate_count"] == 1
    assert validation["branch_mismatch_count"] == 1
    pair_failures = {str(entry["pair_id"]): dict(entry) for entry in validation["pair_failures"]}
    assert pair_failures["hkl-miss"]["candidate_count_total"] == 1
    assert pair_failures["hkl-miss"]["candidate_count_after_hkl_filter"] == 0
    assert pair_failures["hkl-miss"]["candidate_count_after_branch_filter"] == 0
    assert pair_failures["branch-miss"]["candidate_count_total"] == 1
    assert pair_failures["branch-miss"]["candidate_count_after_hkl_filter"] == 1
    assert pair_failures["branch-miss"]["candidate_count_after_branch_filter"] == 0


def test_rebuild_geometry_fit_source_rows_stage_callback_failure_does_not_abort() -> None:
    live_rows = [
        {
            "hkl": (1, 0, 0),
            "q_group_key": ("q", 1),
            "source_reflection_index": 7,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "sim_col": 10.0,
            "sim_row": 20.0,
        }
    ]

    result = geometry_fit.rebuild_geometry_fit_source_rows(
        background_index=0,
        background_label="bg0.osc",
        params_local={"a": 4.143, "c": 28.64},
        consumer="geometry_fit_dataset",
        prior_diagnostics={"status": "snapshot_empty"},
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        can_use_live_runtime_cache=True,
        build_live_rows=lambda: list(live_rows),
        get_memory_intersection_cache=lambda: pytest.fail(
            "stage callback failure should not force memory cache fallback"
        ),
        load_logged_intersection_cache=lambda: pytest.fail(
            "stage callback failure should not force logged cache fallback"
        ),
        logged_cache_matches_params=lambda _meta, _params: pytest.fail(
            "stage callback failure should not touch logged cache params"
        ),
        build_source_rows_from_hit_tables=lambda _tables: pytest.fail(
            "stage callback failure should not rebuild source rows"
        ),
        simulate_hit_tables=lambda _params: pytest.fail(
            "stage callback failure should not run fresh simulation"
        ),
        last_runtime_simulation_diagnostics=lambda: {"status": "unused"},
        project_rows=None,
        required_pairs=[
            {
                "pair_id": "bg0:pair0",
                "overlay_match_index": 0,
                "hkl": (1, 0, 0),
                "q_group_key": ("q", 1),
                "source_reflection_index": 7,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "source_row_index": 0,
                "source_branch_index": 1,
                "source_peak_index": 1,
            }
        ],
        live_cache_inventory={"source_snapshot_count": 1},
        stage_callback=lambda _stage, _payload: (_ for _ in ()).throw(
            RuntimeError("callback boom")
        ),
    )

    assert result.rebuild_source == "live_runtime_cache"
    assert result.diagnostics["stage_callback_failure_count"] == 9
    assert (
        result.diagnostics["stage_callback_last_failed_stage"] == "source_cache_project_rows_ready"
    )


def _targeted_required_pair(
    *,
    pair_id: str,
    hkl: tuple[int, int, int],
    branch_index: int | None,
    q_group_key: object | None,
    x: float = 10.0,
    y: float = 20.0,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    pair = {
        "pair_id": pair_id,
        "overlay_match_index": 0,
        "hkl": hkl,
        "x": float(x),
        "y": float(y),
    }
    if branch_index is not None:
        pair["source_branch_index"] = int(branch_index)
    if q_group_key is not None:
        pair["q_group_key"] = q_group_key
    if isinstance(extra, dict):
        pair.update(extra)
    return pair


def _targeted_source_row(
    *,
    hkl: tuple[int, int, int],
    branch_index: int | None,
    q_group_key: object | None,
    sim_col: float,
    sim_row: float,
    source_peak_index: int = 0,
) -> dict[str, object]:
    row = {
        "hkl": hkl,
        "sim_col": float(sim_col),
        "sim_row": float(sim_row),
        "source_row_index": int(source_peak_index),
        "source_peak_index": int(source_peak_index),
    }
    if branch_index is not None:
        row["source_branch_index"] = int(branch_index)
    if q_group_key is not None:
        row["q_group_key"] = q_group_key
    return row


def test_targeted_preflight_collects_required_hkl_branch_keys() -> None:
    targets = geometry_fit.collect_geometry_fit_required_manual_fit_targets(
        [
            _targeted_required_pair(
                pair_id="bg0:pair0",
                hkl=(1, 0, 0),
                branch_index=1,
                q_group_key=("q", 1),
            ),
            _targeted_required_pair(
                pair_id="bg0:pair1",
                hkl=(2, 0, 0),
                branch_index=None,
                q_group_key=("q", 2),
            ),
            _targeted_required_pair(
                pair_id="bg0:pair2",
                hkl=(3, 0, 0),
                branch_index=None,
                q_group_key=None,
            ),
        ],
        background_index=0,
    )

    required_keys = geometry_fit._geometry_fit_required_branch_group_keys(targets)

    assert [target["branch_constraint_status"] for target in targets] == [
        "constrained",
        "recovered_from_q_group",
        "unconstrained_missing_branch",
    ]
    assert targets[2]["branch_unconstrained"] is True
    assert required_keys == [
        ((1, 0, 0), 1, ("q", 1)),
        ((2, 0, 0), None, ("q", 2)),
        ((3, 0, 0), None, None),
    ]
    rich_targets = geometry_fit.collect_geometry_fit_required_manual_fit_targets(
        [
            _targeted_required_pair(
                pair_id="bg0:pair3",
                hkl=(4, 0, 0),
                branch_index=1,
                q_group_key=("preferred", 1),
                extra={
                    "source_q_group_key": ("source", 1),
                    "branch_group_key": ("branch", 1),
                },
            )
        ],
        background_index=0,
    )

    assert rich_targets[0]["q_group_key"] == ("preferred", 1)
    assert geometry_fit._geometry_fit_required_branch_group_keys(rich_targets) == [
        ((4, 0, 0), 1, ("preferred", 1))
    ]


def _assert_targeted_preflight_filters_required_groups_only() -> None:
    required_pairs = [
        _targeted_required_pair(
            pair_id="bg0:pair0",
            hkl=(1, 0, 0),
            branch_index=1,
            q_group_key=("q", 1),
        ),
        _targeted_required_pair(
            pair_id="bg0:pair1",
            hkl=(2, 0, 0),
            branch_index=0,
            q_group_key=("q", 2),
        ),
    ]
    targets = geometry_fit.collect_geometry_fit_required_manual_fit_targets(
        required_pairs,
        background_index=0,
    )
    required_keys = geometry_fit._geometry_fit_required_branch_group_keys(targets)
    filtered_rows, counts, matched_keys = (
        geometry_fit._geometry_fit_filter_entries_for_required_branch_groups(
            [
                _targeted_source_row(
                    hkl=(1, 0, 0),
                    branch_index=1,
                    q_group_key=("q", 1),
                    sim_col=10.0,
                    sim_row=20.0,
                ),
                _targeted_source_row(
                    hkl=(1, 0, 0),
                    branch_index=0,
                    q_group_key=("q", 1),
                    sim_col=11.0,
                    sim_row=21.0,
                    source_peak_index=1,
                ),
                _targeted_source_row(
                    hkl=(2, 0, 0),
                    branch_index=0,
                    q_group_key=("q", 2),
                    sim_col=30.0,
                    sim_row=40.0,
                    source_peak_index=2,
                ),
                _targeted_source_row(
                    hkl=(9, 9, 9),
                    branch_index=0,
                    q_group_key=("q", 9),
                    sim_col=50.0,
                    sim_row=60.0,
                    source_peak_index=3,
                ),
            ],
            required_keys,
        )
    )

    assert counts == {
        "total_count": 4,
        "after_hkl_filter_count": 3,
        "after_branch_filter_count": 2,
        "unrelated_count": 2,
    }
    assert [(row["hkl"], row.get("source_branch_index")) for row in filtered_rows] == [
        ((1, 0, 0), 1),
        ((2, 0, 0), 0),
    ]
    assert matched_keys == [
        ((1, 0, 0), 1, ("q", 1)),
        ((2, 0, 0), 0, ("q", 2)),
    ]


def test_targeted_preflight_does_not_score_unrelated_hkls_or_branches() -> None:
    _assert_targeted_preflight_filters_required_groups_only()


def test_point_provider_uses_targeted_branch_groups_only() -> None:
    _assert_targeted_preflight_filters_required_groups_only()


def test_targeted_preflight_does_not_score_unrelated_same_branch_groups() -> None:
    targets = geometry_fit.collect_geometry_fit_required_manual_fit_targets(
        [
            _targeted_required_pair(
                pair_id="bg0:pair0",
                hkl=(1, 0, 0),
                branch_index=1,
                q_group_key=("q", 1),
            )
        ],
        background_index=0,
    )
    required_keys = geometry_fit._geometry_fit_required_branch_group_keys(targets)
    filtered_rows, counts, matched_keys = (
        geometry_fit._geometry_fit_filter_entries_for_required_branch_groups(
            [
                _targeted_source_row(
                    hkl=(1, 0, 0),
                    branch_index=1,
                    q_group_key=("q", 1),
                    sim_col=10.0,
                    sim_row=20.0,
                ),
                _targeted_source_row(
                    hkl=(1, 0, 0),
                    branch_index=1,
                    q_group_key=("q", 2),
                    sim_col=11.0,
                    sim_row=21.0,
                    source_peak_index=1,
                ),
            ],
            required_keys,
        )
    )

    assert counts == {
        "total_count": 2,
        "after_hkl_filter_count": 2,
        "after_branch_filter_count": 1,
        "unrelated_count": 1,
    }
    assert len(filtered_rows) == 1
    assert filtered_rows[0]["q_group_key"] == ("q", 1)
    assert matched_keys == [((1, 0, 0), 1, ("q", 1))]


def test_manual_pick_change_only_changes_targeted_subset_not_full_cache() -> None:
    first_targets = geometry_fit.collect_geometry_fit_required_manual_fit_targets(
        [
            _targeted_required_pair(
                pair_id="bg0:pair0",
                hkl=(1, 0, 0),
                branch_index=1,
                q_group_key=("q", 1),
                x=10.0,
                y=20.0,
            ),
        ],
        background_index=0,
    )
    moved_targets = geometry_fit.collect_geometry_fit_required_manual_fit_targets(
        [
            _targeted_required_pair(
                pair_id="bg0:pair0",
                hkl=(1, 0, 0),
                branch_index=1,
                q_group_key=("q", 1),
                x=25.0,
                y=35.0,
            ),
        ],
        background_index=0,
    )
    expanded_targets = geometry_fit.collect_geometry_fit_required_manual_fit_targets(
        [
            _targeted_required_pair(
                pair_id="bg0:pair0",
                hkl=(1, 0, 0),
                branch_index=1,
                q_group_key=("q", 1),
                x=10.0,
                y=20.0,
            ),
            _targeted_required_pair(
                pair_id="bg0:pair1",
                hkl=(2, 0, 0),
                branch_index=0,
                q_group_key=("q", 2),
                x=30.0,
                y=40.0,
            ),
        ],
        background_index=0,
    )

    first_keys = geometry_fit._geometry_fit_required_branch_group_keys(first_targets)
    moved_keys = geometry_fit._geometry_fit_required_branch_group_keys(moved_targets)
    expanded_keys = geometry_fit._geometry_fit_required_branch_group_keys(expanded_targets)
    first_key_digest = geometry_fit._geometry_fit_required_branch_group_keys_digest(
        first_keys,
        background_index=0,
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        preflight_mode="manual_geometry_targeted",
    )
    moved_key_digest = geometry_fit._geometry_fit_required_branch_group_keys_digest(
        moved_keys,
        background_index=0,
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        preflight_mode="manual_geometry_targeted",
    )
    expanded_key_digest = geometry_fit._geometry_fit_required_branch_group_keys_digest(
        expanded_keys,
        background_index=0,
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        preflight_mode="manual_geometry_targeted",
    )
    caked_key_digest = geometry_fit._geometry_fit_required_branch_group_keys_digest(
        first_keys,
        background_index=0,
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        preflight_mode="manual_geometry_targeted",
        consumer="geometry_fit_dataset",
        projection_view_mode="caked",
    )
    detector_key_digest = geometry_fit._geometry_fit_required_branch_group_keys_digest(
        first_keys,
        background_index=0,
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        preflight_mode="manual_geometry_targeted",
        consumer="geometry_fit_dataset",
        projection_view_mode="detector",
    )
    other_consumer_key_digest = geometry_fit._geometry_fit_required_branch_group_keys_digest(
        first_keys,
        background_index=0,
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        preflight_mode="manual_geometry_targeted",
        consumer="manual_overlay_refresh",
        projection_view_mode="detector",
    )
    caked_sig_one_digest = geometry_fit._geometry_fit_required_branch_group_keys_digest(
        first_keys,
        background_index=0,
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        preflight_mode="manual_geometry_targeted",
        consumer="geometry_fit_dataset",
        projection_view_mode="caked",
        projection_view_signature={
            "mode": "caked",
            "radial_axis": [1.0, 2.0],
            "azimuth_axis": [-10.0, 10.0],
        },
    )
    caked_sig_two_digest = geometry_fit._geometry_fit_required_branch_group_keys_digest(
        first_keys,
        background_index=0,
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        preflight_mode="manual_geometry_targeted",
        consumer="geometry_fit_dataset",
        projection_view_mode="caked",
        projection_view_signature={
            "mode": "caked",
            "radial_axis": [1.0, 2.0, 3.0],
            "azimuth_axis": [-10.0, 10.0],
        },
    )
    detector_sig_one_digest = geometry_fit._geometry_fit_required_branch_group_keys_digest(
        first_keys,
        background_index=0,
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        preflight_mode="manual_geometry_targeted",
        consumer="geometry_fit_dataset",
        projection_view_mode="detector",
        projection_view_signature={
            "mode": "detector",
            "background_index": 0,
            "current_background_index": 0,
            "detector_shape": [64, 64],
            "analysis_bins": [5, 7],
            "available": True,
        },
    )
    detector_sig_two_digest = geometry_fit._geometry_fit_required_branch_group_keys_digest(
        first_keys,
        background_index=0,
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        preflight_mode="manual_geometry_targeted",
        consumer="geometry_fit_dataset",
        projection_view_mode="detector",
        projection_view_signature={
            "mode": "detector",
            "background_index": 0,
            "current_background_index": 1,
            "detector_shape": [64, 64],
            "analysis_bins": [5, 7],
            "available": True,
        },
    )
    detector_sig_three_digest = geometry_fit._geometry_fit_required_branch_group_keys_digest(
        first_keys,
        background_index=0,
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        preflight_mode="manual_geometry_targeted",
        consumer="geometry_fit_dataset",
        projection_view_mode="detector",
        projection_view_signature={
            "mode": "detector",
            "background_index": 0,
            "current_background_index": 1,
            "detector_shape": [64, 64],
            "analysis_bins": [11, 13],
            "available": True,
        },
    )

    assert first_keys == moved_keys
    assert first_key_digest == moved_key_digest
    assert geometry_fit._geometry_fit_manual_target_scoring_digest(
        first_targets
    ) != geometry_fit._geometry_fit_manual_target_scoring_digest(moved_targets)
    assert expanded_keys != first_keys
    assert expanded_key_digest != first_key_digest
    assert caked_key_digest != detector_key_digest
    assert other_consumer_key_digest != detector_key_digest
    assert caked_sig_one_digest != caked_sig_two_digest
    assert detector_sig_one_digest == detector_sig_two_digest
    assert detector_sig_one_digest == detector_sig_three_digest


def test_targeted_preflight_projects_only_required_candidate_rows() -> None:
    required_pairs = [
        _targeted_required_pair(
            pair_id="bg0:pair0",
            hkl=(1, 0, 0),
            branch_index=1,
            q_group_key=("q", 1),
        )
    ]
    projected_row_batches: list[list[dict[str, object]]] = []

    result = geometry_fit.rebuild_geometry_fit_source_rows(
        background_index=0,
        background_label="bg0.osc",
        params_local={"a": 4.143, "c": 28.64},
        consumer="geometry_fit_dataset",
        prior_diagnostics={"status": "snapshot_empty"},
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        can_use_live_runtime_cache=False,
        build_live_rows=None,
        get_memory_intersection_cache=lambda: [],
        load_logged_intersection_cache_metadata=lambda: None,
        load_logged_intersection_cache=lambda: pytest.fail(
            "metadata miss should reject logged cache before heavy load"
        ),
        logged_cache_matches_params=lambda _meta, _params: {
            "matches": False,
            "mismatch_reason": "empty_cache",
            "heavy_hit_table_load_attempted": False,
        },
        build_source_rows_from_hit_tables=lambda _tables, **_kwargs: (
            [
                _targeted_source_row(
                    hkl=(1, 0, 0),
                    branch_index=1,
                    q_group_key=("q", 1),
                    sim_col=10.0,
                    sim_row=20.0,
                ),
                _targeted_source_row(
                    hkl=(1, 0, 0),
                    branch_index=1,
                    q_group_key=("q", 1),
                    sim_col=11.0,
                    sim_row=21.0,
                    source_peak_index=1,
                ),
                _targeted_source_row(
                    hkl=(1, 0, 0),
                    branch_index=0,
                    q_group_key=("q", 1),
                    sim_col=12.0,
                    sim_row=22.0,
                    source_peak_index=2,
                ),
                _targeted_source_row(
                    hkl=(9, 9, 9),
                    branch_index=1,
                    q_group_key=("q", 9),
                    sim_col=90.0,
                    sim_row=91.0,
                    source_peak_index=3,
                ),
            ],
            None,
            None,
            None,
        ),
        simulate_hit_tables=lambda _params, **_kwargs: [object()],
        last_runtime_simulation_diagnostics=lambda: {"status": "success"},
        project_rows=lambda rows: (
            projected_row_batches.append(
                [dict(entry) for entry in rows or () if isinstance(entry, dict)]
            )
            or list(rows or ())
        ),
        required_pairs=required_pairs,
        required_manual_fit_targets=geometry_fit.collect_geometry_fit_required_manual_fit_targets(
            required_pairs,
            background_index=0,
        ),
        preflight_mode="manual_geometry_targeted",
        live_cache_inventory={"source_snapshot_count": 0},
    )

    assert [(row["hkl"], row.get("source_branch_index")) for row in result.stored_rows] == [
        ((1, 0, 0), 1),
        ((1, 0, 0), 1),
    ]
    assert len(projected_row_batches) == 1
    assert len(projected_row_batches[0]) == 2
    assert result.diagnostics["total_source_rows_available"] == 4
    assert result.diagnostics["candidate_rows_after_hkl_filter"] == 3
    assert result.diagnostics["candidate_rows_after_branch_filter"] == 2
    assert result.diagnostics["candidate_rows_scored_for_background_distance"] == 2
    assert result.diagnostics["source_rows_projected_for_rebinding"] == 2
    assert result.diagnostics["unrelated_projected_row_count_for_rebinding"] == 0
    assert result.diagnostics["full_source_rows_projected_for_rebinding"] is False


def test_targeted_fresh_simulation_receives_required_branch_group_filter() -> None:
    required_pairs = [
        _targeted_required_pair(
            pair_id="bg0:pair0",
            hkl=(1, 0, 0),
            branch_index=1,
            q_group_key=("q", 1),
        )
    ]
    captured_kwargs: dict[str, object] = {}

    result = geometry_fit.rebuild_geometry_fit_source_rows(
        background_index=0,
        background_label="bg0.osc",
        params_local={"a": 4.143, "c": 28.64},
        consumer="geometry_fit_dataset",
        prior_diagnostics={"status": "snapshot_empty"},
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        can_use_live_runtime_cache=False,
        build_live_rows=None,
        get_memory_intersection_cache=lambda: [],
        load_logged_intersection_cache_metadata=lambda: None,
        load_logged_intersection_cache=lambda: pytest.fail(
            "fresh targeted simulation should not touch heavy logged cache"
        ),
        logged_cache_matches_params=lambda _meta, _params: {
            "matches": False,
            "mismatch_reason": "empty_cache",
            "heavy_hit_table_load_attempted": False,
        },
        build_source_rows_from_hit_tables=lambda _tables, **kwargs: (
            [
                _targeted_source_row(
                    hkl=(1, 0, 0),
                    branch_index=1,
                    q_group_key=("q", 1),
                    sim_col=10.0,
                    sim_row=20.0,
                ),
            ],
            None,
            None,
            None,
        ),
        simulate_hit_tables=lambda _params, **kwargs: captured_kwargs.update(kwargs) or [object()],
        last_runtime_simulation_diagnostics=lambda: {"status": "success"},
        project_rows=lambda rows: list(rows or ()),
        required_pairs=required_pairs,
        required_manual_fit_targets=geometry_fit.collect_geometry_fit_required_manual_fit_targets(
            required_pairs,
            background_index=0,
        ),
        preflight_mode="manual_geometry_targeted",
        live_cache_inventory={"source_snapshot_count": 0},
    )

    assert captured_kwargs["preflight_mode"] == "manual_geometry_targeted"
    assert captured_kwargs["required_branch_group_keys"] == [((1, 0, 0), 1, ("q", 1))]
    assert result.diagnostics["targeted_simulation_supported"] is True
    assert result.diagnostics["targeted_simulation_used"] is True
    assert result.diagnostics["targeted_performance_gate"]["ok"] is True


def test_full_validation_mode_keeps_manual_geometry_targeted_preflight() -> None:
    required_pairs = [
        _targeted_required_pair(
            pair_id="bg0:pair0",
            hkl=(1, 0, 0),
            branch_index=1,
            q_group_key=("q", 1),
        )
    ]
    captured_kwargs: dict[str, object] = {}

    result = geometry_fit.rebuild_geometry_fit_source_rows(
        background_index=0,
        background_label="bg0.osc",
        params_local={"a": 4.143, "c": 28.64},
        consumer="geometry_fit_dataset",
        prior_diagnostics={"status": "snapshot_empty"},
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        can_use_live_runtime_cache=False,
        build_live_rows=None,
        get_memory_intersection_cache=lambda: [],
        load_logged_intersection_cache_metadata=lambda: None,
        load_logged_intersection_cache=lambda: pytest.fail(
            "fresh targeted simulation should not touch heavy logged cache"
        ),
        logged_cache_matches_params=lambda _meta, _params: {
            "matches": False,
            "mismatch_reason": "empty_cache",
            "heavy_hit_table_load_attempted": False,
        },
        build_source_rows_from_hit_tables=lambda _tables, **_kwargs: (
            [
                _targeted_source_row(
                    hkl=(1, 0, 0),
                    branch_index=1,
                    q_group_key=("q", 1),
                    sim_col=10.0,
                    sim_row=20.0,
                ),
            ],
            None,
            None,
            None,
        ),
        simulate_hit_tables=lambda _params, **kwargs: captured_kwargs.update(kwargs) or [object()],
        last_runtime_simulation_diagnostics=lambda: {"status": "success"},
        project_rows=lambda rows: list(rows or ()),
        required_pairs=required_pairs,
        required_manual_fit_targets=geometry_fit.collect_geometry_fit_required_manual_fit_targets(
            required_pairs,
            background_index=0,
        ),
        preflight_mode="full",
        live_cache_inventory={"source_snapshot_count": 0},
    )

    assert captured_kwargs["preflight_mode"] == "manual_geometry_targeted"
    assert result.diagnostics["preflight_mode"] == "manual_geometry_targeted"
    assert result.diagnostics["targeted_preflight_enabled"] is True
    assert result.diagnostics["targeted_performance_gate"]["ok"] is True


def test_targeted_fallback_filters_before_expansion_when_simulator_filter_not_supported() -> None:
    required_pairs = [
        _targeted_required_pair(
            pair_id="bg0:pair0",
            hkl=(1, 0, 0),
            branch_index=1,
            q_group_key=("q", 1),
        )
    ]
    build_kwargs: dict[str, object] = {}
    stage_events: list[tuple[str, dict[str, object]]] = []
    simulate_calls: list[str] = []

    def _simulate_without_targeting(_params):
        simulate_calls.append("full")
        return [object()]

    result = geometry_fit.rebuild_geometry_fit_source_rows(
        background_index=0,
        background_label="bg0.osc",
        params_local={"a": 4.143, "c": 28.64},
        consumer="geometry_fit_dataset",
        prior_diagnostics={"status": "snapshot_empty"},
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        can_use_live_runtime_cache=False,
        build_live_rows=None,
        get_memory_intersection_cache=lambda: [],
        load_logged_intersection_cache_metadata=lambda: None,
        load_logged_intersection_cache=lambda: pytest.fail(
            "fallback path should still reject heavy logged cache on metadata miss"
        ),
        logged_cache_matches_params=lambda _meta, _params: {
            "matches": False,
            "mismatch_reason": "empty_cache",
            "heavy_hit_table_load_attempted": False,
        },
        build_source_rows_from_hit_tables=lambda _tables, **kwargs: (
            build_kwargs.update(kwargs)
            or (
                [
                    _targeted_source_row(
                        hkl=(1, 0, 0),
                        branch_index=1,
                        q_group_key=("q", 1),
                        sim_col=10.0,
                        sim_row=20.0,
                    ),
                    _targeted_source_row(
                        hkl=(9, 9, 9),
                        branch_index=0,
                        q_group_key=("q", 9),
                        sim_col=90.0,
                        sim_row=91.0,
                        source_peak_index=9,
                    ),
                ],
                None,
                None,
                None,
            )
        ),
        simulate_hit_tables=_simulate_without_targeting,
        last_runtime_simulation_diagnostics=lambda: {"status": "success"},
        project_rows=lambda rows: list(rows or ()),
        required_pairs=required_pairs,
        required_manual_fit_targets=geometry_fit.collect_geometry_fit_required_manual_fit_targets(
            required_pairs,
            background_index=0,
        ),
        preflight_mode="manual_geometry_targeted",
        live_cache_inventory={"source_snapshot_count": 0},
        stage_callback=lambda stage, payload: stage_events.append((str(stage), dict(payload))),
    )

    assert build_kwargs["required_branch_group_keys"] == [((1, 0, 0), 1, ("q", 1))]
    assert [(row["hkl"], row.get("source_branch_index")) for row in result.stored_rows] == [
        ((1, 0, 0), 1),
    ]
    assert result.diagnostics["targeted_simulation_fallback_reason"] == (
        "simulator_filter_not_supported"
    )
    assert result.diagnostics["full_fresh_simulation_fallback_used"] is True
    assert result.diagnostics["unrelated_scored_row_count_for_rebinding"] == 1
    assert result.diagnostics["targeted_performance_gate"]["ok"] is False
    assert simulate_calls == ["full"]
    assert [stage for stage, _payload in stage_events] == [
        "source_cache_target_collection_start",
        "source_cache_target_collection_ready",
        "source_cache_targeted_projected_cache_start",
        "source_cache_targeted_projected_cache_miss",
        "source_cache_memory_intersection_cache_start",
        "source_cache_memory_intersection_cache_miss",
        "source_cache_logged_intersection_cache_start",
        "source_cache_logged_intersection_cache_miss",
        "source_cache_targeted_fresh_simulation_start",
        "source_cache_targeted_fresh_simulation_unsupported",
        "source_cache_full_simulation_fallback_start",
        "source_cache_full_simulation_fallback_ready",
        "source_cache_targeted_fresh_simulation_ready",
        "source_cache_targeted_source_rows_start",
        "source_cache_targeted_source_rows_ready",
        "source_cache_project_rows_start",
        "source_cache_project_rows_ready",
    ]


def test_targeted_fresh_simulation_requires_required_branch_group_filter_support() -> None:
    required_pairs = [
        _targeted_required_pair(
            pair_id="bg0:pair0",
            hkl=(1, 0, 0),
            branch_index=1,
            q_group_key=("q", 1),
        )
    ]
    simulate_preflight_modes: list[object] = []

    def _simulate_preflight_only(_params, *, preflight_mode=None):
        simulate_preflight_modes.append(preflight_mode)
        return [object()]

    result = geometry_fit.rebuild_geometry_fit_source_rows(
        background_index=0,
        background_label="bg0.osc",
        params_local={"a": 4.143, "c": 28.64},
        consumer="geometry_fit_dataset",
        prior_diagnostics={"status": "snapshot_empty"},
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        can_use_live_runtime_cache=False,
        build_live_rows=None,
        get_memory_intersection_cache=lambda: [],
        load_logged_intersection_cache_metadata=lambda: None,
        load_logged_intersection_cache=lambda: pytest.fail(
            "unsupported targeted simulation should not touch heavy logged cache"
        ),
        logged_cache_matches_params=lambda _meta, _params: {
            "matches": False,
            "mismatch_reason": "empty_cache",
            "heavy_hit_table_load_attempted": False,
        },
        build_source_rows_from_hit_tables=lambda _tables, **_kwargs: (
            [
                _targeted_source_row(
                    hkl=(1, 0, 0),
                    branch_index=1,
                    q_group_key=("q", 1),
                    sim_col=10.0,
                    sim_row=20.0,
                )
            ],
            None,
            None,
            None,
        ),
        simulate_hit_tables=_simulate_preflight_only,
        last_runtime_simulation_diagnostics=lambda: {"status": "success"},
        project_rows=lambda rows: list(rows or ()),
        required_pairs=required_pairs,
        required_manual_fit_targets=geometry_fit.collect_geometry_fit_required_manual_fit_targets(
            required_pairs,
            background_index=0,
        ),
        preflight_mode="manual_geometry_targeted",
        live_cache_inventory={"source_snapshot_count": 0},
    )

    assert simulate_preflight_modes == [None]
    assert result.diagnostics["targeted_simulation_supported"] is False
    assert result.diagnostics["targeted_simulation_used"] is False
    assert result.diagnostics["targeted_simulation_fallback_reason"] == (
        "simulator_filter_not_supported"
    )
    assert result.diagnostics["full_fresh_simulation_fallback_used"] is True
    assert result.diagnostics["targeted_performance_gate"]["ok"] is False


def test_targeted_source_build_requires_required_branch_group_filter_support() -> None:
    required_pairs = [
        _targeted_required_pair(
            pair_id="bg0:pair0",
            hkl=(1, 0, 0),
            branch_index=1,
            q_group_key=("q", 1),
        )
    ]
    build_preflight_modes: list[object] = []

    def _build_preflight_only(_tables, *, preflight_mode=None):
        build_preflight_modes.append(preflight_mode)
        return (
            [
                _targeted_source_row(
                    hkl=(1, 0, 0),
                    branch_index=1,
                    q_group_key=("q", 1),
                    sim_col=10.0,
                    sim_row=20.0,
                )
            ],
            None,
            None,
            None,
        )

    result = geometry_fit.rebuild_geometry_fit_source_rows(
        background_index=0,
        background_label="bg0.osc",
        params_local={"a": 4.143, "c": 28.64},
        consumer="geometry_fit_dataset",
        prior_diagnostics={"status": "snapshot_empty"},
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        can_use_live_runtime_cache=False,
        build_live_rows=None,
        get_memory_intersection_cache=lambda: [object()],
        load_logged_intersection_cache=lambda: pytest.fail(
            "accepted memory cache should not touch logged cache"
        ),
        logged_cache_matches_params=lambda _meta, _params: pytest.fail(
            "accepted memory cache should not check logged params"
        ),
        build_source_rows_from_hit_tables=_build_preflight_only,
        simulate_hit_tables=lambda _params: pytest.fail(
            "accepted memory cache should not run simulation"
        ),
        last_runtime_simulation_diagnostics=lambda: {"status": "unused"},
        project_rows=lambda rows: list(rows or ()),
        required_pairs=required_pairs,
        required_manual_fit_targets=geometry_fit.collect_geometry_fit_required_manual_fit_targets(
            required_pairs,
            background_index=0,
        ),
        preflight_mode="manual_geometry_targeted",
        live_cache_inventory={"source_snapshot_count": 0},
    )

    assert build_preflight_modes == ["manual_geometry_targeted"]
    assert result.diagnostics["targeted_cache_hit"] is True
    assert result.diagnostics["full_source_rows_built_for_rebinding"] is True
    assert result.diagnostics["targeted_performance_gate"]["ok"] is False


def test_targeted_simulation_used_follows_runtime_diagnostics() -> None:
    required_pairs = [
        _targeted_required_pair(
            pair_id="bg0:pair0",
            hkl=(1, 0, 0),
            branch_index=1,
            q_group_key=("q", 1),
        )
    ]
    captured_kwargs: dict[str, object] = {}

    result = geometry_fit.rebuild_geometry_fit_source_rows(
        background_index=0,
        background_label="bg0.osc",
        params_local={"a": 4.143, "c": 28.64},
        consumer="geometry_fit_dataset",
        prior_diagnostics={"status": "snapshot_empty"},
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        can_use_live_runtime_cache=False,
        build_live_rows=None,
        get_memory_intersection_cache=lambda: [],
        load_logged_intersection_cache_metadata=lambda: None,
        load_logged_intersection_cache=lambda: pytest.fail(
            "targeted fresh simulation should skip heavy logged cache on metadata miss"
        ),
        logged_cache_matches_params=lambda _meta, _params: {
            "matches": False,
            "mismatch_reason": "empty_cache",
            "heavy_hit_table_load_attempted": False,
        },
        build_source_rows_from_hit_tables=lambda _tables, **_kwargs: (
            [
                _targeted_source_row(
                    hkl=(1, 0, 0),
                    branch_index=1,
                    q_group_key=("q", 1),
                    sim_col=10.0,
                    sim_row=20.0,
                ),
            ],
            None,
            None,
            None,
        ),
        simulate_hit_tables=lambda _params, **kwargs: captured_kwargs.update(kwargs) or [object()],
        last_runtime_simulation_diagnostics=lambda: {
            "status": "success",
            "targeted_simulation_supported": True,
            "targeted_simulation_used": False,
            "targeted_simulation_fallback_reason": "targeted_hkl_filter_empty",
        },
        project_rows=lambda rows: list(rows or ()),
        required_pairs=required_pairs,
        required_manual_fit_targets=geometry_fit.collect_geometry_fit_required_manual_fit_targets(
            required_pairs,
            background_index=0,
        ),
        preflight_mode="manual_geometry_targeted",
        live_cache_inventory={"source_snapshot_count": 0},
    )

    assert captured_kwargs["required_branch_group_keys"] == [((1, 0, 0), 1, ("q", 1))]
    assert result.diagnostics["targeted_simulation_supported"] is True
    assert result.diagnostics["targeted_simulation_used"] is False
    assert result.diagnostics["targeted_simulation_fallback_reason"] == (
        "targeted_hkl_filter_empty"
    )
    assert result.diagnostics["targeted_performance_gate"]["ok"] is False


def test_stored_rows_only_payload_is_never_projected_cache_hit() -> None:
    required_pairs = [
        _targeted_required_pair(
            pair_id="bg0:pair0",
            hkl=(1, 0, 0),
            branch_index=1,
            q_group_key=("q", 1),
        )
    ]
    build_calls: list[int] = []
    simulation_calls: list[int] = []

    result = geometry_fit.rebuild_geometry_fit_source_rows(
        background_index=0,
        background_label="bg0.osc",
        params_local={"a": 4.143, "c": 28.64},
        consumer="geometry_fit_dataset",
        prior_diagnostics={"status": "snapshot_empty"},
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        can_use_live_runtime_cache=False,
        build_live_rows=None,
        projection_view_mode="detector",
        projection_view_signature={"mode": "detector", "detector_shape": [64, 64]},
        get_memory_intersection_cache=lambda: [],
        load_logged_intersection_cache_metadata=lambda: None,
        load_logged_intersection_cache=lambda: pytest.fail(
            "stored-only targeted cache miss should not touch heavy logged cache"
        ),
        logged_cache_matches_params=lambda _meta, _params: {
            "matches": False,
            "mismatch_reason": "empty_cache",
            "heavy_hit_table_load_attempted": False,
        },
        build_source_rows_from_hit_tables=lambda _tables, **_kwargs: (
            build_calls.append(1)
            or [
                _targeted_source_row(
                    hkl=(1, 0, 0),
                    branch_index=1,
                    q_group_key=("q", 1),
                    sim_col=10.0,
                    sim_row=20.0,
                ),
            ],
            None,
            None,
            None,
        ),
        simulate_hit_tables=lambda _params, **_kwargs: simulation_calls.append(1) or [object()],
        last_runtime_simulation_diagnostics=lambda: {"status": "success"},
        project_rows=lambda rows: list(rows or ()),
        required_pairs=required_pairs,
        required_manual_fit_targets=geometry_fit.collect_geometry_fit_required_manual_fit_targets(
            required_pairs,
            background_index=0,
        ),
        preflight_mode="manual_geometry_targeted",
        live_cache_inventory={"source_snapshot_count": 0},
        get_targeted_projected_cache=lambda _digest: {
            "requested_signature": geometry_fit._geometry_fit_cache_jsonable(("sig", 0)),
            "projection_view_signature": {
                "mode": "detector",
                "detector_shape": [64, 64],
            },
            "consumer": "geometry_fit_dataset",
            "stored_rows": [
                _targeted_source_row(
                    hkl=(1, 0, 0),
                    branch_index=1,
                    q_group_key=("q", 1),
                    sim_col=999.0,
                    sim_row=999.0,
                )
            ],
            "projected_rows": [],
            "cache_source": "broken_targeted_cache",
        },
        store_targeted_projected_cache=lambda *_args, **_kwargs: None,
    )

    assert simulation_calls == [1]
    assert build_calls == [1]
    assert result.diagnostics["targeted_cache_hit"] is False
    assert result.diagnostics["cache_source"] != "targeted_projected_cache"
    assert result.projected_rows[0]["sim_col"] == 10.0


def test_detector_targeted_cache_hit_ignores_current_background_signature_drift() -> None:
    required_pairs = [
        _targeted_required_pair(
            pair_id="bg0:pair0",
            hkl=(1, 0, 0),
            branch_index=1,
            q_group_key=("q", 1),
        )
    ]
    cached_rows = [
        _targeted_source_row(
            hkl=(1, 0, 0),
            branch_index=1,
            q_group_key=("q", 1),
            sim_col=33.0,
            sim_row=44.0,
        ),
    ]

    result = geometry_fit.rebuild_geometry_fit_source_rows(
        background_index=0,
        background_label="bg0.osc",
        params_local={"a": 4.143, "c": 28.64},
        consumer="geometry_fit_dataset",
        prior_diagnostics={"status": "snapshot_empty"},
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        can_use_live_runtime_cache=False,
        build_live_rows=None,
        projection_view_mode="detector",
        projection_view_signature={
            "mode": "detector",
            "background_index": 0,
            "current_background_index": 1,
            "detector_shape": [64, 64],
            "analysis_bins": [5, 7],
            "available": True,
        },
        get_memory_intersection_cache=lambda: [],
        load_logged_intersection_cache_metadata=lambda: None,
        load_logged_intersection_cache=lambda: pytest.fail(
            "detector targeted cache hit should not touch heavy logged cache"
        ),
        logged_cache_matches_params=lambda _meta, _params: {
            "matches": False,
            "mismatch_reason": "empty_cache",
            "heavy_hit_table_load_attempted": False,
        },
        build_source_rows_from_hit_tables=lambda *_args, **_kwargs: pytest.fail(
            "detector targeted cache hit should skip fresh row rebuild"
        ),
        simulate_hit_tables=lambda *_args, **_kwargs: pytest.fail(
            "detector targeted cache hit should skip simulation"
        ),
        last_runtime_simulation_diagnostics=lambda: {"status": "success"},
        project_rows=lambda _rows: pytest.fail(
            "detector targeted cache hit should use cached projected rows"
        ),
        required_pairs=required_pairs,
        required_manual_fit_targets=geometry_fit.collect_geometry_fit_required_manual_fit_targets(
            required_pairs,
            background_index=0,
        ),
        preflight_mode="manual_geometry_targeted",
        live_cache_inventory={"source_snapshot_count": 0},
        get_targeted_projected_cache=lambda _digest: {
            "requested_signature": geometry_fit._geometry_fit_cache_jsonable(("sig", 0)),
            "projection_view_signature": {
                "mode": "detector",
                "background_index": 0,
                "current_background_index": 0,
                "detector_shape": [64, 64],
                "analysis_bins": [5, 7],
                "available": True,
            },
            "projected_rows": cached_rows,
            "consumer": "geometry_fit_dataset",
            "cache_source": "targeted_projected_cache",
        },
        store_targeted_projected_cache=lambda *_args, **_kwargs: None,
    )

    assert result.diagnostics["targeted_cache_hit"] is True
    assert result.diagnostics["cache_source"] == "targeted_projected_cache"
    assert result.projected_rows == [dict(cached_rows[0], background_index=0)]


def test_detector_targeted_cache_hit_ignores_analysis_bins_drift() -> None:
    required_pairs = [
        _targeted_required_pair(
            pair_id="bg0:pair0",
            hkl=(1, 0, 0),
            branch_index=1,
            q_group_key=("q", 1),
        )
    ]
    cached_rows = [
        _targeted_source_row(
            hkl=(1, 0, 0),
            branch_index=1,
            q_group_key=("q", 1),
            sim_col=55.0,
            sim_row=66.0,
        ),
    ]

    result = geometry_fit.rebuild_geometry_fit_source_rows(
        background_index=0,
        background_label="bg0.osc",
        params_local={"a": 4.143, "c": 28.64},
        consumer="geometry_fit_dataset",
        prior_diagnostics={"status": "snapshot_empty"},
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        can_use_live_runtime_cache=False,
        build_live_rows=None,
        projection_view_mode="detector",
        projection_view_signature={
            "mode": "detector",
            "background_index": 0,
            "current_background_index": 1,
            "detector_shape": [64, 64],
            "analysis_bins": [11, 13],
            "available": True,
        },
        get_memory_intersection_cache=lambda: [],
        load_logged_intersection_cache_metadata=lambda: None,
        load_logged_intersection_cache=lambda: pytest.fail(
            "analysis-bin targeted cache miss should not touch heavy logged cache"
        ),
        logged_cache_matches_params=lambda _meta, _params: {
            "matches": False,
            "mismatch_reason": "empty_cache",
            "heavy_hit_table_load_attempted": False,
        },
        build_source_rows_from_hit_tables=lambda *_args, **_kwargs: pytest.fail(
            "analysis-bin detector cache hit should skip fresh row rebuild"
        ),
        simulate_hit_tables=lambda *_args, **_kwargs: pytest.fail(
            "analysis-bin detector cache hit should skip simulation"
        ),
        last_runtime_simulation_diagnostics=lambda: {"status": "success"},
        project_rows=lambda _rows: pytest.fail(
            "analysis-bin detector cache hit should use cached projected rows"
        ),
        required_pairs=required_pairs,
        required_manual_fit_targets=geometry_fit.collect_geometry_fit_required_manual_fit_targets(
            required_pairs,
            background_index=0,
        ),
        preflight_mode="manual_geometry_targeted",
        live_cache_inventory={"source_snapshot_count": 0},
        get_targeted_projected_cache=lambda _digest: {
            "requested_signature": geometry_fit._geometry_fit_cache_jsonable(("sig", 0)),
            "projection_view_signature": {
                "mode": "detector",
                "background_index": 0,
                "current_background_index": 0,
                "detector_shape": [64, 64],
                "analysis_bins": [5, 7],
                "available": True,
            },
            "projected_rows": cached_rows,
            "consumer": "geometry_fit_dataset",
            "cache_source": "targeted_projected_cache",
        },
        store_targeted_projected_cache=lambda *_args, **_kwargs: None,
    )

    assert result.diagnostics["targeted_cache_hit"] is True
    assert result.diagnostics["cache_source"] == "targeted_projected_cache"
    assert result.projected_rows == [dict(cached_rows[0], background_index=0)]


def test_noncurrent_caked_rebuild_uses_caked_payload_failure_reason() -> None:
    result = geometry_fit.rebuild_geometry_fit_source_rows(
        background_index=1,
        background_label="bg1.osc",
        params_local={"a": 4.143, "c": 28.64},
        consumer="geometry_fit_dataset",
        prior_diagnostics={"status": "snapshot_empty"},
        requested_signature=("sig", 1),
        requested_signature_summary="sig-summary-1",
        can_use_live_runtime_cache=False,
        build_live_rows=None,
        projection_view_mode="caked",
        projection_view_signature={
            "mode": "caked",
            "available": True,
            "background_index": 1,
            "current_background_index": 0,
            "detector_shape": [64, 64],
            "radial_axis": [1.0, 2.0],
            "azimuth_axis": [-10.0, 10.0],
        },
        projection_payload=None,
        get_memory_intersection_cache=lambda: [],
        load_logged_intersection_cache_metadata=lambda: None,
        load_logged_intersection_cache=lambda: pytest.fail(
            "missing caked payload should reject before heavy logged cache load"
        ),
        logged_cache_matches_params=lambda _meta, _params: {
            "matches": False,
            "mismatch_reason": "empty_cache",
            "heavy_hit_table_load_attempted": False,
        },
        build_source_rows_from_hit_tables=lambda _tables, **_kwargs: (
            [
                _targeted_source_row(
                    hkl=(1, 0, 0),
                    branch_index=1,
                    q_group_key=("q", 1),
                    sim_col=10.0,
                    sim_row=20.0,
                ),
            ],
            None,
            None,
            None,
        ),
        simulate_hit_tables=lambda _params, **_kwargs: [object()],
        last_runtime_simulation_diagnostics=lambda: {"status": "success"},
        project_rows=lambda _rows: pytest.fail(
            "strict caked projection should not use the generic projector"
        ),
        project_rows_for_background_view=lambda _rows: pytest.fail(
            "missing caked payload should reject before background projection"
        ),
    )

    assert result.stored_rows
    assert result.projected_rows == []
    assert result.diagnostics["projection_failure_reason"] == ("missing_background_caked_payload")


def test_full_fresh_simulation_fallback_does_not_pass_targeted_performance_gate() -> None:
    required_pairs = [
        _targeted_required_pair(
            pair_id="bg0:pair0",
            hkl=(1, 0, 0),
            branch_index=1,
            q_group_key=("q", 1),
        )
    ]

    result = geometry_fit.rebuild_geometry_fit_source_rows(
        background_index=0,
        background_label="bg0.osc",
        params_local={"a": 4.143, "c": 28.64},
        consumer="geometry_fit_dataset",
        prior_diagnostics={"status": "snapshot_empty"},
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        can_use_live_runtime_cache=False,
        build_live_rows=None,
        get_memory_intersection_cache=lambda: [],
        load_logged_intersection_cache_metadata=lambda: None,
        load_logged_intersection_cache=lambda: pytest.fail(
            "fallback path should not attempt heavy logged cache load"
        ),
        logged_cache_matches_params=lambda _meta, _params: {
            "matches": False,
            "mismatch_reason": "empty_cache",
            "heavy_hit_table_load_attempted": False,
        },
        build_source_rows_from_hit_tables=lambda _tables, **_kwargs: (
            [
                _targeted_source_row(
                    hkl=(1, 0, 0),
                    branch_index=1,
                    q_group_key=("q", 1),
                    sim_col=10.0,
                    sim_row=20.0,
                ),
            ],
            None,
            None,
            None,
        ),
        simulate_hit_tables=lambda _params: [object()],
        last_runtime_simulation_diagnostics=lambda: {"status": "success"},
        project_rows=lambda rows: list(rows or ()),
        required_pairs=required_pairs,
        required_manual_fit_targets=geometry_fit.collect_geometry_fit_required_manual_fit_targets(
            required_pairs,
            background_index=0,
        ),
        preflight_mode="manual_geometry_targeted",
        live_cache_inventory={"source_snapshot_count": 0},
    )

    assert result.stored_rows
    assert result.diagnostics["targeted_simulation_fallback_reason"] == (
        "simulator_filter_not_supported"
    )
    assert result.diagnostics["full_fresh_simulation_fallback_used"] is True
    assert result.diagnostics["targeted_performance_gate"]["ok"] is False


def test_simulator_filter_unsupported_never_passes_targeted_performance_gate() -> None:
    gate = geometry_fit._geometry_fit_targeted_performance_gate_payload(
        {
            "preflight_mode": "manual_geometry_targeted",
            "targeted_preflight_enabled": True,
            "targeted_simulation_supported": False,
            "targeted_simulation_used": False,
            "targeted_cache_hit": True,
            "full_fresh_simulation_fallback_used": False,
            "targeted_simulation_fallback_reason": "simulator_filter_not_supported",
            "required_hkl_branch_group_count": 1,
            "source_rows_projected_for_rebinding": 1,
            "candidate_rows_scored_for_background_distance": 1,
            "unrelated_projected_row_count_for_rebinding": 0,
            "unrelated_scored_row_count_for_rebinding": 0,
            "full_source_rows_built_for_rebinding": False,
            "full_source_rows_projected_for_rebinding": False,
        }
    )

    assert gate["ok"] is False


def test_logged_cache_params_mismatch_rejects_before_heavy_hit_table_load() -> None:
    required_pairs = [
        _targeted_required_pair(
            pair_id="bg0:pair0",
            hkl=(1, 0, 0),
            branch_index=1,
            q_group_key=("q", 1),
        )
    ]
    stage_events: list[tuple[str, dict[str, object]]] = []

    result = geometry_fit.rebuild_geometry_fit_source_rows(
        background_index=0,
        background_label="bg0.osc",
        params_local={"a": 4.143, "c": 28.64},
        consumer="geometry_fit_dataset",
        prior_diagnostics={"status": "snapshot_empty"},
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        can_use_live_runtime_cache=False,
        build_live_rows=None,
        get_memory_intersection_cache=lambda: [],
        load_logged_intersection_cache_metadata=lambda: {
            "signature_digest": "logged-digest",
        },
        load_logged_intersection_cache=lambda: pytest.fail(
            "params mismatch should reject before loading heavy hit tables"
        ),
        logged_cache_matches_params=lambda _meta, _params: {
            "matches": False,
            "expected_signature_digest": "expected-digest",
            "actual_signature_digest": "logged-digest",
            "mismatch_reason": "params_mismatch",
            "heavy_hit_table_load_attempted": False,
        },
        build_source_rows_from_hit_tables=lambda _tables, **_kwargs: (
            [
                _targeted_source_row(
                    hkl=(1, 0, 0),
                    branch_index=1,
                    q_group_key=("q", 1),
                    sim_col=10.0,
                    sim_row=20.0,
                )
            ],
            None,
            None,
            None,
        ),
        simulate_hit_tables=lambda _params, **_kwargs: [object()],
        last_runtime_simulation_diagnostics=lambda: {"status": "success"},
        project_rows=lambda rows: list(rows or ()),
        required_pairs=required_pairs,
        required_manual_fit_targets=geometry_fit.collect_geometry_fit_required_manual_fit_targets(
            required_pairs,
            background_index=0,
        ),
        preflight_mode="manual_geometry_targeted",
        live_cache_inventory={"source_snapshot_count": 0},
        stage_callback=lambda stage, payload: stage_events.append((str(stage), dict(payload))),
    )

    miss_payload = next(
        payload
        for stage, payload in stage_events
        if stage == "source_cache_logged_intersection_cache_miss"
    )

    assert result.stored_rows
    assert miss_payload["status"] == "params_mismatch"
    assert miss_payload["expected_signature_digest"] == "expected-digest"
    assert miss_payload["actual_signature_digest"] == "logged-digest"
    assert miss_payload["mismatch_reason"] == "params_mismatch"
    assert miss_payload["heavy_hit_table_load_attempted"] is False


def test_peak_record_fallback_with_restored_provenance_matches_rebuild_for_active_pairs() -> None:
    peak_records = [
        {
            "display_col": 10.0,
            "display_row": 20.0,
            "native_col": 10.0,
            "native_row": 20.0,
            "sim_col_raw": 10.0,
            "sim_row_raw": 20.0,
            "hkl": (1, 0, 0),
            "q_group_key": ("q", 1),
            "source_table_index": 0,
            "source_row_index": 0,
            "source_peak_index": 13,
            "phi": 15.0,
        }
    ]
    rebuilt_rows = [
        {
            "hkl": (1, 0, 0),
            "q_group_key": ("q", 1),
            "source_table_index": 0,
            "source_reflection_index": 7,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "sim_col": 10.0,
            "sim_row": 20.0,
        }
    ]
    required_pairs = [
        {
            "pair_id": "bg0:pair0",
            "overlay_match_index": 0,
            "hkl": (1, 0, 0),
            "q_group_key": ("q", 1),
            "source_table_index": 0,
            "source_reflection_index": 7,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 1,
            "source_peak_index": 1,
        }
    ]

    broken_rows = manual_geometry.geometry_manual_live_peak_candidates_from_records(
        peak_records,
        source_reflection_indices_local=[7],
        active_signature_matches=False,
    )
    fixed_rows = manual_geometry.geometry_manual_live_peak_candidates_from_records(
        peak_records,
        source_reflection_indices_local=[7],
        source_row_hkl_lookup={(0, 0): (1, 0, 0)},
        active_signature_matches=True,
    )
    broken_validation = geometry_fit.validate_geometry_fit_live_source_rows(
        broken_rows,
        required_pairs=required_pairs,
    )
    fixed_validation = geometry_fit.validate_geometry_fit_live_source_rows(
        fixed_rows,
        required_pairs=required_pairs,
    )
    rebuild_validation = geometry_fit.validate_geometry_fit_live_source_rows(
        rebuilt_rows,
        required_pairs=required_pairs,
    )

    def _digest(value) -> str:
        payload = json.dumps(value, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    assert _digest(broken_validation["resolved_pairs"]) != _digest(
        rebuild_validation["resolved_pairs"]
    )
    assert _digest(fixed_validation["resolved_pairs"]) == _digest(
        rebuild_validation["resolved_pairs"]
    )
    assert len(fixed_validation["resolved_pairs"]) == len(rebuild_validation["resolved_pairs"]) == 1
    assert (
        geometry_fit._geometry_fit_required_pair_dual_path_diff(
            fixed_validation,
            rebuild_validation,
        )
        == []
    )
    assert geometry_fit._geometry_fit_required_pair_dual_path_diff(
        broken_validation,
        rebuild_validation,
    ) == [
        {
            "pair_id": "bg0:pair0",
            "before_status": "failed",
            "after_status": "resolved",
            "before_canonical_identity": None,
            "after_canonical_identity": [7, 1],
            "before_simulated_point": None,
            "after_simulated_point": [10.0, 20.0],
            "before_reason": "missing_trusted_reflection_row",
            "after_reason": None,
        }
    ]

    accepted = geometry_fit.rebuild_geometry_fit_source_rows(
        background_index=0,
        background_label="bg0.osc",
        params_local={"a": 4.143, "c": 28.64},
        consumer="geometry_fit_dataset",
        prior_diagnostics={"status": "snapshot_hit"},
        requested_signature=("sig", 0),
        requested_signature_summary="sig-summary",
        can_use_live_runtime_cache=True,
        build_live_rows=lambda: {
            "rows": list(fixed_rows),
            "cache_metadata": {
                "cache_source": "peak_records",
                "active_signature_matches": True,
                "source_snapshot_row_count": 1,
            },
        },
        get_memory_intersection_cache=lambda: [],
        load_logged_intersection_cache=lambda: ([], None),
        logged_cache_matches_params=lambda _meta, _params: False,
        build_source_rows_from_hit_tables=lambda _tables: (
            list(rebuilt_rows),
            None,
            None,
            None,
        ),
        simulate_hit_tables=lambda _params: [object()],
        last_runtime_simulation_diagnostics=lambda: {"status": "success"},
        project_rows=None,
        required_pairs=required_pairs,
        live_cache_inventory={"source_snapshot_count": 1},
    )

    assert accepted.rebuild_source == "live_runtime_cache"
    assert accepted.metadata["live_runtime_cache_metadata"]["cache_source"] == ("peak_records")
    assert accepted.metadata["live_runtime_cache_metadata"]["active_signature_matches"] is True


def test_build_geometry_manual_fit_dataset_includes_runtime_exception_details_in_failure() -> None:
    snapshot_diag = {
        "status": "process_peaks_parallel_exception",
        "exception_type": "RuntimeError",
        "exception_message": "boom",
    }

    def _rebuild_source_rows(
        background_idx,
        params,
        *,
        consumer,
        prior_diagnostics,
        required_pairs,
    ):
        return []

    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["C:/tmp/bg0.osc"],
        current_background_index=0,
        image_size=64,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda idx: [
            {
                "q_group_key": ("q", 1),
                "source_table_index": 1,
                "source_row_index": 2,
                "source_peak_index": 0,
                "hkl": (1, 1, 0),
                "x": 30.0,
                "y": 40.0,
            }
        ],
        load_background_by_index=lambda idx: (
            np.zeros((6, 7), dtype=np.float64),
            np.zeros((6, 7), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_source_rows_for_background=(
            lambda idx, params, *, consumer, required_pairs=None: []
        ),
        geometry_manual_rebuild_source_rows_for_background=_rebuild_source_rows,
        geometry_manual_last_source_snapshot_diagnostics=lambda: dict(snapshot_diag),
        geometry_manual_simulated_peaks_for_params=(lambda params, *, prefer_cache: []),
        geometry_manual_simulated_lookup=lambda peaks: {},
        geometry_manual_entry_display_coords=lambda entry: (30.0, 40.0),
        unrotate_display_peaks=lambda entries, shape, *, k: [{"x": 30.0, "y": 40.0}],
        display_to_native_sim_coords=lambda col, row, shape: (float(col), float(row)),
        select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
            {
                "indexing_mode": "xy",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "xy",
                "label": "identity",
            },
            {"pairs": 1},
        ),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: list(entries),
        orient_image_for_fit=lambda image, **kwargs: np.asarray(image, dtype=np.float64),
        geometry_manual_match_config=lambda: {"search_radius": 4.0},
    )

    with pytest.raises(RuntimeError, match=r"Runtime=RuntimeError: boom"):
        geometry_fit.build_geometry_manual_fit_dataset(
            0,
            theta_base=1.5,
            base_fit_params={"theta_offset": 0.0},
            manual_dataset_bindings=manual_dataset_bindings,
            orientation_cfg={},
        )


def test_build_runtime_geometry_fit_value_callbacks_reads_live_runtime_values() -> None:
    shared_theta = {"value": True}
    background_state = {"index": 1}
    theta_offset_state = {"value": 0.125}
    optics_mode = {"value": "exact"}

    callbacks = geometry_fit.build_runtime_geometry_fit_value_callbacks(
        geometry_fit.GeometryFitRuntimeValueBindings(
            fit_zb_var=_DummyVar(False),
            fit_zs_var=_DummyVar(False),
            fit_theta_var=_DummyVar(True),
            fit_psi_z_var=_DummyVar(False),
            fit_chi_var=_DummyVar(False),
            fit_cor_var=_DummyVar(False),
            fit_gamma_var=_DummyVar(True),
            fit_Gamma_var=_DummyVar(False),
            fit_dist_var=_DummyVar(False),
            fit_a_var=_DummyVar(True),
            fit_c_var=_DummyVar(False),
            fit_center_x_var=_DummyVar(True),
            fit_center_y_var=_DummyVar(False),
            zb_var=_DummyVar(0.1),
            zs_var=_DummyVar(0.2),
            theta_initial_var=_DummyVar(9.5),
            psi_z_var=_DummyVar(0.7),
            chi_var=_DummyVar(0.5),
            cor_angle_var=_DummyVar(0.6),
            sample_width_var=_DummyVar(0.01),
            sample_length_var=_DummyVar(0.02),
            sample_depth_var=_DummyVar(0.003),
            gamma_var=_DummyVar(0.8),
            Gamma_var=_DummyVar(0.9),
            corto_detector_var=_DummyVar(1.0),
            a_var=_DummyVar(4.2),
            c_var=_DummyVar(32.1),
            center_x_var=_DummyVar(100.0),
            center_y_var=_DummyVar(200.0),
            debye_x_var=_DummyVar(3.0),
            debye_y_var=_DummyVar(4.0),
            geometry_theta_offset_var=_DummyVar("0.125"),
            current_background_index=lambda: background_state["index"],
            geometry_fit_uses_shared_theta_offset=lambda: shared_theta["value"],
            current_geometry_theta_offset=lambda strict=False: theta_offset_state["value"],
            background_theta_for_index=(
                lambda index, strict_count=False: {0: 2.75, 1: 4.25}[int(index)]
            ),
            build_mosaic_params=lambda: {"sigma_mosaic_deg": 0.2},
            current_optics_mode_flag=lambda: optics_mode["value"],
            lambda_value=1.54,
            psi=12.0,
            n2=1.7,
            pixel_size_value=1.2e-4,
        )
    )

    assert callbacks.current_var_names() == ["theta_offset", "gamma", "a", "center_x"]

    params = callbacks.current_params()
    assert params["theta_initial"] == 4.25
    assert params["theta_offset"] == 0.125
    assert params["mosaic_params"] == {"sigma_mosaic_deg": 0.2}
    assert params["center"] == [100.0, 200.0]
    assert params["optics_mode"] == "exact"
    assert np.array_equal(params["uv1"], np.array([1.0, 0.0, 0.0]))
    assert np.array_equal(params["uv2"], np.array([0.0, 1.0, 0.0]))

    ui_params = callbacks.current_ui_params()
    assert ui_params == {
        "zb": 0.1,
        "zs": 0.2,
        "theta_initial": 9.5,
        "psi_z": 0.7,
        "chi": 0.5,
        "cor_angle": 0.6,
        "gamma": 0.8,
        "Gamma": 0.9,
        "corto_detector": 1.0,
        "a": 4.2,
        "c": 32.1,
        "center_x": 100.0,
        "center_y": 200.0,
        "center": [100.0, 200.0],
        "theta_offset": 0.125,
    }
    assert callbacks.var_map["theta_initial"].get() == 9.5
    assert callbacks.var_map["center_x"].get() == 100.0

    shared_theta["value"] = False
    theta_offset_state["value"] = 0.333333
    callbacks.var_map["theta_initial"].set(8.75)

    assert callbacks.current_var_names() == ["theta_initial", "gamma", "a", "center_x"]
    params = callbacks.current_params()
    assert params["theta_initial"] == 8.75
    assert params["theta_offset"] == 0.0
    assert params["center_x"] == 100.0


def test_make_runtime_geometry_fit_action_prepare_bindings_factory_builds_prepare_bundle() -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["bg0.osc", "bg1.osc"],
        current_background_index=1,
        image_size=256,
        display_rotate_k=3,
        geometry_manual_pairs_for_index=lambda idx: [{"pair": idx}],
        load_background_by_index=lambda idx: (
            np.zeros((2, 2), dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
        ),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_simulated_peaks_for_params=lambda *args, **kwargs: [],
        geometry_manual_simulated_lookup=lambda _peaks: {},
        geometry_manual_entry_display_coords=lambda entry: None,
        unrotate_display_peaks=lambda entries, shape, *, k: [],
        display_to_native_sim_coords=lambda col, row, shape: (col, row),
        select_fit_orientation=lambda *args, **kwargs: ({}, {}),
        apply_orientation_to_entries=lambda entries, shape, **kwargs: [],
        orient_image_for_fit=lambda image, **kwargs: image,
    )

    factory = geometry_fit.make_runtime_geometry_fit_action_prepare_bindings_factory(
        fit_config={"geometry": {"mode": "test"}},
        theta_initial=9.5,
        apply_geometry_fit_background_selection=lambda **kwargs: True,
        current_geometry_fit_background_indices=lambda **kwargs: [0, 1],
        geometry_fit_uses_shared_theta_offset=lambda *_args, **_kwargs: False,
        apply_background_theta_metadata=lambda **kwargs: True,
        current_background_theta_values=lambda **kwargs: [9.5, 10.5],
        current_geometry_theta_offset=lambda **kwargs: 0.25,
        ensure_geometry_fit_caked_view=lambda: None,
        manual_dataset_bindings=manual_dataset_bindings,
        build_runtime_config_factory=lambda var_names, fit_params: (
            calls.append((list(var_names), dict(fit_params)))
            or {"vars": list(var_names), "fit_params": dict(fit_params)}
        ),
    )

    bindings = factory(["gamma", "a"])

    assert isinstance(
        bindings,
        geometry_fit.GeometryFitRuntimePreparationBindings,
    )
    assert bindings.fit_config == {"geometry": {"mode": "test"}}
    assert bindings.manual_dataset_bindings is manual_dataset_bindings
    assert list(bindings.manual_dataset_bindings.osc_files) == ["bg0.osc", "bg1.osc"]
    assert bindings.manual_dataset_bindings.current_background_index == 1
    assert bindings.theta_initial == 9.5
    assert bindings.manual_dataset_bindings.image_size == 256
    assert bindings.manual_dataset_bindings.display_rotate_k == 3
    assert bindings.build_runtime_config({"gamma": 0.2, "a": 4.1}) == {
        "vars": ["gamma", "a"],
        "fit_params": {"gamma": 0.2, "a": 4.1},
    }
    assert calls == [(["gamma", "a"], {"gamma": 0.2, "a": 4.1})]


def test_make_runtime_geometry_fit_manual_dataset_bindings_factory_reads_live_values() -> None:
    runtime_state = {"background_index": 1}

    factory = geometry_fit.make_runtime_geometry_fit_manual_dataset_bindings_factory(
        osc_files_factory=lambda: ["bg0.osc", "bg1.osc"],
        current_background_index_factory=lambda: runtime_state["background_index"],
        image_size=256,
        display_rotate_k=3,
        geometry_manual_pairs_for_index="manual-pairs",
        load_background_by_index="load-background",
        apply_background_backend_orientation="apply-orientation",
        geometry_manual_simulated_peaks_for_params="manual-sim-peaks",
        geometry_manual_simulated_lookup="manual-sim-lookup",
        geometry_manual_entry_display_coords="manual-entry-coords",
        unrotate_display_peaks="unrotate",
        display_to_native_sim_coords="display-to-native",
        select_fit_orientation="select-orientation",
        apply_orientation_to_entries="apply-entries",
        orient_image_for_fit="orient-image",
    )

    first = factory()
    runtime_state["background_index"] = 4
    second = factory()

    assert isinstance(first, geometry_fit.GeometryFitRuntimeManualDatasetBindings)
    assert list(first.osc_files) == ["bg0.osc", "bg1.osc"]
    assert first.current_background_index == 1
    assert first.image_size == 256
    assert first.display_rotate_k == 3
    assert first.geometry_manual_pairs_for_index == "manual-pairs"
    assert second.current_background_index == 4
    assert second.geometry_manual_simulated_lookup == "manual-sim-lookup"


def test_build_runtime_geometry_fit_config_factory_reads_runtime_constraint_values() -> None:
    calls: list[tuple[str, object]] = []
    factory = geometry_fit.build_runtime_geometry_fit_config_factory(
        base_config={"solver": {"loss": "soft_l1"}},
        current_constraint_state=lambda names: (
            calls.append(("constraints", list(names or [])))
            or {"gamma": {"window": 0.25, "pull": 0.5}}
        ),
        current_parameter_domains=lambda names: (
            calls.append(("domains", list(names or []))) or {"gamma": (0.0, 1.0)}
        ),
    )

    runtime_cfg = factory(["gamma"], {"gamma": 0.6, "a": 4.1})

    assert runtime_cfg["allow_unsafe_runtime"] is False
    assert runtime_cfg["use_numba"] is False
    assert runtime_cfg["bounds"] == {"gamma": [0.0, 1.0]}
    assert runtime_cfg["priors"] == {}
    assert runtime_cfg["candidate_param_names"] == ["gamma"]
    assert runtime_cfg["solver"] == {
        "loss": "soft_l1",
        "workers": "auto",
        "parallel_mode": "auto",
        "worker_numba_threads": 0,
    }
    assert runtime_cfg["optimizer"] == runtime_cfg["solver"]
    assert calls == [
        ("domains", ["gamma"]),
    ]


def test_geometry_fit_constraint_name_helpers_handle_shared_theta_offset() -> None:
    assert geometry_fit.geometry_fit_constraint_source_name("theta_offset") == "theta_initial"
    assert geometry_fit.geometry_fit_constraint_source_name("gamma") == "gamma"
    assert (
        geometry_fit.geometry_fit_constraint_parameter_name(
            "theta_initial",
            use_shared_theta_offset=True,
        )
        == "theta_offset"
    )
    assert (
        geometry_fit.geometry_fit_constraint_parameter_name(
            "theta_initial",
            use_shared_theta_offset=False,
        )
        == "theta_initial"
    )


def test_read_runtime_geometry_fit_constraint_state_normalizes_live_control_values() -> None:
    controls = {
        "theta_initial": {
            "window_var": _DummyVar("nan"),
            "pull_var": _DummyVar("0.2"),
        },
        "gamma": {
            "window_var": _DummyVar("-2.5"),
            "pull_var": _DummyVar("3.0"),
        },
        "a": {
            "window_var": _DummyVar("0.4"),
            "pull_var": _DummyVar("bad"),
        },
    }

    state = geometry_fit.read_runtime_geometry_fit_constraint_state(
        controls=controls,
        names=["theta_offset", "gamma", "a"],
        use_shared_theta_offset=False,
    )

    assert state == {
        "gamma": {"window": 0.0, "pull": 1.0},
        "a": {"window": 0.4, "pull": 0.0},
    }


def test_read_runtime_geometry_fit_parameter_domains_uses_bounds_and_slider_ranges() -> None:
    parameter_specs = {
        "theta_initial": {
            "value_slider": _DummySlider(-3.0, 4.0),
            "value_var": _DummyVar(9.5),
            "step": 0.1,
        },
        "gamma": {
            "value_slider": _DummySlider(2.0, -1.0),
            "value_var": _DummyVar(0.8),
            "step": 0.05,
        },
        "center_x": {
            "value_slider": _DummySlider(0.0, 255.0),
            "value_var": _DummyVar(100.0),
            "step": 1.0,
        },
    }

    domains = geometry_fit.read_runtime_geometry_fit_parameter_domains(
        parameter_specs=parameter_specs,
        image_size=256,
        fit_config={
            "geometry": {
                "bounds": {
                    "theta_offset": {"min": 0.25, "max": -0.5},
                }
            }
        },
        names=["theta_initial", "gamma", "center_x"],
        use_shared_theta_offset=True,
    )

    assert domains == {
        "theta_initial": (-4.0, 4.0),
        "gamma": (-1.0, 2.0),
        "center_x": (0.0, 255.0),
    }


def test_runtime_geometry_fit_constraint_defaults_use_shared_helpers() -> None:
    parameter_specs = {
        "theta_initial": {
            "value_slider": _DummySlider(-3.0, 4.0),
            "value_var": _DummyVar(9.5),
            "step": 0.1,
        },
        "gamma": {
            "value_slider": _DummySlider(0.0, 2.0),
            "value_var": _DummyVar(0.8),
            "step": 0.05,
        },
    }

    theta_window = geometry_fit.default_runtime_geometry_fit_constraint_window(
        name="theta_initial",
        parameter_specs=parameter_specs,
        fit_config={
            "geometry": {
                "bounds": {
                    "theta_offset": {
                        "mode": "relative",
                        "min": -0.3,
                        "max": 0.1,
                    }
                },
                "priors": {
                    "theta_offset": {
                        "sigma": 0.15,
                    }
                },
            }
        },
        parameter_domains={"theta_offset": (-0.5, 0.5)},
        current_theta_offset=0.2,
        use_shared_theta_offset=True,
    )
    gamma_window = geometry_fit.default_runtime_geometry_fit_constraint_window(
        name="gamma",
        parameter_specs=parameter_specs,
        fit_config={},
        parameter_domains={"gamma": (0.0, 2.0)},
        current_theta_offset=0.0,
        use_shared_theta_offset=False,
    )
    theta_pull = geometry_fit.default_runtime_geometry_fit_constraint_pull(
        name="theta_initial",
        fit_config={
            "geometry": {
                "priors": {
                    "theta_offset": {
                        "sigma": 0.15,
                    }
                }
            }
        },
        window=theta_window,
        use_shared_theta_offset=True,
    )

    assert theta_window == 0.3
    assert gamma_window == 0.5
    assert np.isclose(theta_pull, (1.0 - 0.5) / 0.95)
    assert (
        geometry_fit.default_runtime_geometry_fit_constraint_pull(
            name="gamma",
            fit_config={"geometry": {"priors": {"gamma": {"sigma": "bad"}}}},
            window=gamma_window,
            use_shared_theta_offset=False,
        )
        == 0.0
    )


def test_build_runtime_geometry_fit_action_bindings_composes_helper_bundles(
    monkeypatch,
) -> None:
    calls: list[tuple[str, object]] = []
    manual_dataset_bindings = geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["bg0.osc"],
        current_background_index=0,
        image_size=256,
        display_rotate_k=3,
        geometry_manual_pairs_for_index="manual-pairs",
        load_background_by_index="load-background",
        apply_background_backend_orientation="apply-orientation",
        geometry_manual_simulated_peaks_for_params="manual-sim-peaks",
        geometry_manual_simulated_lookup="manual-sim-lookup",
        geometry_manual_entry_display_coords="manual-entry-coords",
        unrotate_display_peaks="unrotate",
        display_to_native_sim_coords="display-to-native",
        select_fit_orientation="select-orientation",
        apply_orientation_to_entries="apply-entries",
        orient_image_for_fit="orient-image",
    )

    monkeypatch.setattr(
        geometry_fit,
        "make_runtime_geometry_fit_action_prepare_bindings_factory",
        lambda **kwargs: calls.append(("prepare", kwargs)) or "prepare-factory",
    )
    monkeypatch.setattr(
        geometry_fit,
        "build_runtime_geometry_fit_action_execution_bindings",
        lambda **kwargs: calls.append(("execution", kwargs)) or "execution-bindings",
    )

    value_callbacks = SimpleNamespace(current_var_names=lambda: [], current_params=lambda: {})
    solver_inputs = geometry_fit.GeometryFitRuntimeSolverInputs(
        miller="miller-data",
        intensities="intensity-data",
        image_size=512,
    )

    bindings = geometry_fit.build_runtime_geometry_fit_action_bindings(
        value_callbacks=value_callbacks,
        fit_config={"geometry": {}},
        theta_initial=9.5,
        apply_geometry_fit_background_selection="apply-selection",
        current_geometry_fit_background_indices="current-bg-indices",
        geometry_fit_uses_shared_theta_offset="uses-shared-theta",
        apply_background_theta_metadata="apply-theta-meta",
        current_background_theta_values="current-theta-values",
        current_geometry_theta_offset="current-theta-offset",
        ensure_geometry_fit_caked_view="ensure-caked",
        manual_dataset_bindings=manual_dataset_bindings,
        build_runtime_config_factory="build-runtime-config",
        downloads_dir="downloads-dir",
        simulation_runtime_state="sim-state",
        background_runtime_state="bg-state",
        theta_initial_var="theta-var",
        geometry_theta_offset_var="theta-offset-var",
        current_ui_params="current-ui-params",
        var_map={"gamma": "gamma-var"},
        background_theta_for_index="bg-theta-for-index",
        refresh_status="refresh-status",
        update_manual_pick_button_label="update-manual-pick",
        capture_undo_state="capture-undo",
        push_undo_state="push-undo",
        replace_dataset_cache="replace-dataset-cache",
        request_preview_skip_once="request-skip",
        schedule_update="schedule-update",
        draw_overlay_records="draw-overlay",
        draw_initial_pairs_overlay="draw-initial",
        set_last_overlay_state="set-overlay-state",
        set_progress_text="set-progress",
        cmd_line="cmd-line",
        solver_inputs=solver_inputs,
        sim_display_rotate_k=1,
        background_display_rotate_k=3,
        simulate_and_compare_hkl="simulate-and-compare",
        aggregate_match_centers="aggregate-centers",
        build_overlay_records="build-overlay-records",
        compute_frame_diagnostics="frame-diagnostics",
        solve_fit="solve-fit",
        stamp_factory="stamp-factory",
        flush_ui="flush-ui",
    )

    assert isinstance(bindings, geometry_fit.GeometryFitRuntimeActionBindings)
    assert bindings.value_callbacks is value_callbacks
    assert bindings.prepare_bindings_factory == "prepare-factory"
    assert bindings.execution_bindings == "execution-bindings"
    assert bindings.solve_fit == "solve-fit"
    assert bindings.stamp_factory == "stamp-factory"
    assert bindings.flush_ui == "flush-ui"
    assert calls == [
        (
            "prepare",
            {
                "fit_config": {"geometry": {}},
                "theta_initial": 9.5,
                "apply_geometry_fit_background_selection": "apply-selection",
                "current_geometry_fit_background_indices": "current-bg-indices",
                "geometry_fit_uses_shared_theta_offset": "uses-shared-theta",
                "apply_background_theta_metadata": "apply-theta-meta",
                "current_background_theta_values": "current-theta-values",
                "current_geometry_theta_offset": "current-theta-offset",
                "ensure_geometry_fit_caked_view": "ensure-caked",
                "manual_dataset_bindings": manual_dataset_bindings,
                "build_runtime_config_factory": "build-runtime-config",
            },
        ),
        (
            "execution",
            {
                "downloads_dir": "downloads-dir",
                "simulation_runtime_state": "sim-state",
                "background_runtime_state": "bg-state",
                "theta_initial_var": "theta-var",
                "geometry_theta_offset_var": "theta-offset-var",
                "current_ui_params": "current-ui-params",
                "var_map": {"gamma": "gamma-var"},
                "background_theta_for_index": "bg-theta-for-index",
                "refresh_status": "refresh-status",
                "update_manual_pick_button_label": "update-manual-pick",
                "capture_undo_state": "capture-undo",
                "push_undo_state": "push-undo",
                "replace_dataset_cache": "replace-dataset-cache",
                "request_preview_skip_once": "request-skip",
                "schedule_update": "schedule-update",
                "draw_overlay_records": "draw-overlay",
                "draw_initial_pairs_overlay": "draw-initial",
                "set_last_overlay_state": "set-overlay-state",
                "set_progress_text": "set-progress",
                "cmd_line": "cmd-line",
                "solver_inputs": solver_inputs,
                "sim_display_rotate_k": 1,
                "background_display_rotate_k": 3,
                "simulate_and_compare_hkl": "simulate-and-compare",
                "aggregate_match_centers": "aggregate-centers",
                "build_overlay_records": "build-overlay-records",
                "compute_frame_diagnostics": "frame-diagnostics",
                "live_update_callback": None,
            },
        ),
    ]


def test_make_runtime_geometry_fit_action_bindings_factory_reads_live_values(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []
    runtime_state = {"background_index": 1, "theta_initial": 9.5}
    solver_inputs = {
        "value": geometry_fit.GeometryFitRuntimeSolverInputs(
            miller="miller-1",
            intensities="intensity-1",
            image_size=512,
        )
    }
    callback_counter = {"count": 0}

    monkeypatch.setattr(
        geometry_fit,
        "build_runtime_geometry_fit_action_bindings",
        lambda **kwargs: calls.append(kwargs) or f"bindings-{len(calls)}",
    )

    def _value_callbacks():
        callback_counter["count"] += 1
        return f"value-callbacks-{callback_counter['count']}"

    manual_dataset_calls: list[object] = []
    manual_dataset = {"value": "dataset-bindings-1"}

    factory = geometry_fit.make_runtime_geometry_fit_action_bindings_factory(
        value_callbacks_factory=_value_callbacks,
        fit_config={"geometry": {"mode": "test"}},
        theta_initial_factory=lambda: runtime_state["theta_initial"],
        apply_geometry_fit_background_selection="apply-selection",
        current_geometry_fit_background_indices="current-bg-indices",
        geometry_fit_uses_shared_theta_offset="uses-shared-theta",
        apply_background_theta_metadata="apply-theta-meta",
        current_background_theta_values="current-theta-values",
        current_geometry_theta_offset="current-theta-offset",
        ensure_geometry_fit_caked_view="ensure-caked",
        manual_dataset_bindings_factory=lambda: (
            manual_dataset_calls.append(runtime_state["background_index"])
            or manual_dataset["value"]
        ),
        build_runtime_config_factory="build-runtime-config",
        downloads_dir="downloads-dir",
        simulation_runtime_state="sim-state",
        background_runtime_state="bg-state",
        theta_initial_var="theta-var",
        geometry_theta_offset_var="theta-offset-var",
        current_ui_params="current-ui-params",
        var_map={"gamma": "gamma-var"},
        background_theta_for_index="bg-theta-for-index",
        refresh_status="refresh-status",
        update_manual_pick_button_label="update-manual-pick",
        capture_undo_state="capture-undo",
        push_undo_state="push-undo",
        replace_dataset_cache="replace-dataset-cache",
        request_preview_skip_once="request-skip",
        schedule_update="schedule-update",
        draw_overlay_records="draw-overlay",
        draw_initial_pairs_overlay="draw-initial",
        set_last_overlay_state="set-overlay-state",
        set_progress_text="set-progress",
        cmd_line="cmd-line",
        solver_inputs_factory=lambda: solver_inputs["value"],
        sim_display_rotate_k=1,
        background_display_rotate_k=3,
        simulate_and_compare_hkl="simulate-and-compare",
        aggregate_match_centers="aggregate-centers",
        build_overlay_records="build-overlay-records",
        compute_frame_diagnostics="frame-diagnostics",
        solve_fit="solve-fit",
        stamp_factory="stamp-factory",
        flush_ui="flush-ui",
    )

    assert factory() == "bindings-1"

    runtime_state["background_index"] = 4
    runtime_state["theta_initial"] = 12.25
    manual_dataset["value"] = "dataset-bindings-2"
    solver_inputs["value"] = geometry_fit.GeometryFitRuntimeSolverInputs(
        miller="miller-2",
        intensities="intensity-2",
        image_size=1024,
    )

    assert factory() == "bindings-2"
    assert calls[0]["value_callbacks"] == "value-callbacks-1"
    assert calls[1]["value_callbacks"] == "value-callbacks-2"
    assert calls[0]["manual_dataset_bindings"] == "dataset-bindings-1"
    assert calls[1]["manual_dataset_bindings"] == "dataset-bindings-2"
    assert calls[0]["theta_initial"] == 9.5
    assert calls[1]["theta_initial"] == 12.25
    assert calls[0]["solver_inputs"].miller == "miller-1"
    assert calls[1]["solver_inputs"].miller == "miller-2"
    assert manual_dataset_calls == [1, 4]


def test_make_runtime_geometry_fit_action_callback_runs_before_action() -> None:
    events: list[object] = []

    callback = geometry_fit.make_runtime_geometry_fit_action_callback(
        bindings_factory=lambda: events.append("bindings") or "live-bindings",
        before_run=lambda: events.append("before"),
        run_action=lambda **kwargs: events.append(("run", kwargs)) or "result",
        after_run=lambda result: events.append(("after", result)),
    )

    assert callback() is None
    assert events == [
        "before",
        "bindings",
        ("run", {"bindings": "live-bindings"}),
        ("after", "result"),
    ]


def test_build_geometry_fit_action_notice_returns_rejection_warning() -> None:
    notice = geometry_fit.build_geometry_fit_action_notice(
        geometry_fit.GeometryFitRuntimeActionResult(
            params={},
            var_names=["gamma"],
            preserve_live_theta=True,
            execution_result=geometry_fit.GeometryFitRuntimeExecutionResult(
                log_path=Path("C:/tmp/geometry_fit_log.txt"),
                apply_result=geometry_fit.GeometryFitRuntimeApplyResult(
                    accepted=False,
                    rejection_reason="RMS residual 624.37 px exceeds the acceptance limit of 100.00 px.",
                    rms=624.374847,
                    fitted_params=None,
                    postprocess=None,
                ),
            ),
        )
    )

    assert notice == geometry_fit.GeometryFitActionNotice(
        level="warning",
        title="Geometry Fit Rejected",
        message=(
            "Geometry fit finished but the solution was rejected.\n"
            "RMS residual 624.37 px exceeds the acceptance limit of 100.00 px.\n"
            "The live geometry state was left unchanged.\n"
            "Fit log: C:\\tmp\\geometry_fit_log.txt"
        ),
    )


def test_build_geometry_fit_action_notice_formats_windows_log_path_on_posix_notice_host() -> None:
    notice = geometry_fit.build_geometry_fit_action_notice(
        geometry_fit.GeometryFitRuntimeActionResult(
            params={},
            var_names=["gamma"],
            preserve_live_theta=True,
            execution_result=geometry_fit.GeometryFitRuntimeExecutionResult(
                log_path=PurePosixPath("C:/tmp/geometry_fit_log.txt"),
                apply_result=geometry_fit.GeometryFitRuntimeApplyResult(
                    accepted=False,
                    rejection_reason="RMS residual 624.37 px exceeds the acceptance limit of 100.00 px.",
                    rms=624.374847,
                    fitted_params=None,
                    postprocess=None,
                ),
            ),
        )
    )

    assert notice == geometry_fit.GeometryFitActionNotice(
        level="warning",
        title="Geometry Fit Rejected",
        message=(
            "Geometry fit finished but the solution was rejected.\n"
            "RMS residual 624.37 px exceeds the acceptance limit of 100.00 px.\n"
            "The live geometry state was left unchanged.\n"
            "Fit log: C:\\tmp\\geometry_fit_log.txt"
        ),
    )


def test_build_geometry_fit_action_notice_returns_error_notice() -> None:
    notice = geometry_fit.build_geometry_fit_action_notice(
        geometry_fit.GeometryFitRuntimeActionResult(
            params={},
            var_names=["gamma"],
            preserve_live_theta=True,
            error_text="Geometry fit failed: boom",
        )
    )

    assert notice == geometry_fit.GeometryFitActionNotice(
        level="error",
        title="Geometry Fit Failed",
        message="Geometry fit failed: boom",
    )


def test_build_geometry_fit_action_notice_returns_error_notice_with_preflight_log() -> None:
    notice = geometry_fit.build_geometry_fit_action_notice(
        geometry_fit.GeometryFitRuntimeActionResult(
            params={},
            var_names=["gamma"],
            preserve_live_theta=True,
            prepare_result=geometry_fit.GeometryFitPreparationResult(
                error_text="Geometry fit unavailable",
                log_path=PurePosixPath("C:/tmp/geometry_fit_log.txt"),
            ),
            error_text="Geometry fit unavailable",
        )
    )

    assert notice == geometry_fit.GeometryFitActionNotice(
        level="error",
        title="Geometry Fit Failed",
        message="Geometry fit unavailable\nFit log: C:\\tmp\\geometry_fit_log.txt",
    )


def test_apply_geometry_fit_result_values_updates_named_vars_and_offset_text() -> None:
    gamma_var = _DummyVar(0.0)
    a_var = _DummyVar(0.0)
    theta_offset_var = _DummyVar("")

    geometry_fit.apply_geometry_fit_result_values(
        ["gamma", "theta_offset", "a"],
        [1.25, 0.33333333, 4.5],
        var_map={
            "gamma": gamma_var,
            "a": a_var,
        },
        geometry_theta_offset_var=theta_offset_var,
    )

    assert gamma_var.get() == 1.25
    assert a_var.get() == 4.5
    assert theta_offset_var.get() == "0.333333"


def test_write_geometry_fit_run_start_log_emits_prepared_prelude() -> None:
    events: list[tuple[str, object]] = []
    prepared_run = geometry_fit.GeometryFitPreparedRun(
        fit_params={"theta_initial": 2.5},
        selected_background_indices=[1],
        background_theta_values=[2.0],
        joint_background_mode=False,
        current_dataset={"dataset_index": 1},
        dataset_infos=[{"dataset_index": 1}],
        dataset_specs=[{"dataset_index": 1}],
        start_cmd_line="start: vars=gamma datasets=1 current_groups=1 current_points=2",
        start_log_sections=[
            ("Section A:", ["one", "two"]),
            ("Section B:", ["three"]),
        ],
        max_display_markers=120,
        geometry_runtime_cfg={"bounds": {}},
    )

    geometry_fit.write_geometry_fit_run_start_log(
        stamp="20260328_120000",
        prepared_run=prepared_run,
        cmd_line=lambda text: events.append(("cmd", text)),
        log_line=lambda text="": events.append(("line", text)),
        log_section=lambda title, lines: events.append(("section", (title, list(lines)))),
    )

    assert events == [
        (
            "cmd",
            "start: vars=gamma datasets=1 current_groups=1 current_points=2",
        ),
        ("line", "Geometry fit started: 20260328_120000"),
        ("line", ""),
        ("section", ("Section A:", ["one", "two"])),
        ("section", ("Section B:", ["three"])),
    ]


def test_build_geometry_fit_start_log_sections_include_request_and_runtime_context() -> None:
    sections = geometry_fit.build_geometry_fit_start_log_sections(
        params={"gamma": 0.2, "a": 4.1},
        var_names=["gamma", "a"],
        dataset_infos=[{"summary_line": "bg[0] bg0"}],
        current_dataset={
            "dataset_index": 0,
            "label": "bg0",
            "group_count": 2,
            "pair_count": 3,
            "resolved_source_pair_count": 2,
            "theta_base": 3.0,
            "theta_effective": 3.1,
            "orientation_choice": {"label": "rotate"},
            "orientation_diag": {
                "pairs": 3,
                "identity_rms_px": 1.2,
                "best_rms_px": 0.8,
                "reason": "improved",
            },
        },
        selected_background_indices=[0, 2],
        joint_background_mode=True,
        geometry_runtime_cfg={
            "debug_logging": True,
            "use_numba": False,
            "allow_unsafe_runtime": False,
            "optimizer": {
                "loss": "linear",
                "f_scale_px": 1.0,
                "manual_point_fit_mode": True,
                "weighted_matching": False,
                "q_group_line_constraints": True,
            },
            "solver": {
                "parallel_mode": "auto",
                "workers": "auto",
                "worker_numba_threads": 0,
            },
            "discrete_modes": {"enabled": False},
            "identifiability": {"enabled": True},
        },
    )

    assert sections == [
        (
            "Fitting variables (start values):",
            [
                "gamma=0.200000",
                "a=4.100000",
            ],
        ),
        (
            "Manual geometry datasets:",
            ["bg[0] bg0"],
        ),
        (
            "Current orientation diagnostics:",
            [
                "pairs=3",
                "chosen=rotate",
                "identity_rms_px=1.2000",
                "best_rms_px=0.8000",
                "reason=improved",
            ],
        ),
    ]


def test_build_geometry_fit_start_log_sections_omit_debug_context_by_default() -> None:
    sections = geometry_fit.build_geometry_fit_start_log_sections(
        params={"gamma": 0.2, "a": 4.1},
        var_names=["gamma", "a"],
        dataset_infos=[{"summary_line": "bg[0] bg0"}],
        current_dataset={
            "dataset_index": 0,
            "label": "bg0",
            "group_count": 2,
            "pair_count": 3,
            "orientation_choice": {"label": "rotate"},
            "orientation_diag": {
                "pairs": 3,
                "identity_rms_px": 1.2,
                "best_rms_px": 0.8,
                "reason": "improved",
            },
        },
        selected_background_indices=[0, 2],
        joint_background_mode=True,
        geometry_runtime_cfg={},
    )

    assert [title for title, _lines in sections] == [
        "Fitting variables (start values):",
        "Manual geometry datasets:",
        "Current orientation diagnostics:",
    ]


def test_build_geometry_fit_dataset_cache_metadata_summarizes_source_tables() -> None:
    metadata = geometry_fit.build_geometry_fit_dataset_cache_metadata(
        background_index=2,
        current_background_index=1,
        simulated_peaks=[
            {
                "source_table_index": 0,
                "source_row_index": 2,
                "source_peak_index": 5,
                "hkl": (1, 1, 0),
                "q_group_key": ("q", 1),
                "qr": 1.25,
                "qz": -0.5,
                "sim_col": 10.0,
                "sim_row": 20.0,
            },
            {
                "source_table_index": 0,
                "source_row_index": 4,
                "source_peak_index": 7,
                "hkl": (1, 1, 0),
                "q_group_key": ("q", 1),
                "qr": 1.25,
                "qz": -0.5,
                "sim_col": 11.0,
                "sim_row": 21.0,
            },
            {
                "source_table_index": 1,
                "source_row_index": 1,
                "source_peak_index": 2,
                "hkl": (0, 0, 3),
                "q_group_key": ("q", 2),
                "qr": 0.0,
                "qz": 1.5,
                "sim_col": np.nan,
                "sim_row": 30.0,
            },
        ],
        source_resolution_diagnostics=[
            {
                "saved_source_table_index": 0,
                "fit_resolved": True,
                "fit_source_row_key": (0, 4),
                "fit_source_peak_key": (0, 7),
                "fit_resolution_kind": "legacy_dense_hkl_rebind",
            },
            {
                "saved_source_table_index": 1,
                "fit_resolved": False,
                "fit_resolution_kind": None,
            },
        ],
        pair_count=2,
        resolved_source_pair_count=1,
    )

    assert metadata["cache_action"] == "rebuilt"
    assert metadata["reused"] is False
    assert metadata["rebuilt"] is True
    assert metadata["background_index"] == 2
    assert metadata["current_background_index"] == 1
    assert metadata["pair_count"] == 2
    assert metadata["resolved_source_pair_count"] == 1
    assert metadata["simulated_peak_count"] == 3
    assert metadata["table_count"] == 2
    assert metadata["table_summaries"] == [
        {
            "source_table_index": 0,
            "nominal_hkl": [1, 1, 0],
            "q_group_key": ["q", 1],
            "qr": 1.25,
            "qz": -0.5,
            "row_count_before_grouping": 2,
            "row_count_after_grouping": 1,
            "dropped_nonfinite_row_count": 0,
            "nominal_hkl_recovery_count": 1,
            "merged_group_count": 1,
            "representative_row_indices_kept": [4],
        },
        {
            "source_table_index": 1,
            "nominal_hkl": [0, 0, 3],
            "q_group_key": ["q", 2],
            "qr": 0.0,
            "qz": 1.5,
            "row_count_before_grouping": 1,
            "row_count_after_grouping": 0,
            "dropped_nonfinite_row_count": 1,
            "nominal_hkl_recovery_count": 0,
            "merged_group_count": 0,
            "representative_row_indices_kept": [],
        },
    ]


def test_build_geometry_fit_dataset_cache_log_lines_include_per_table_metadata() -> None:
    lines = geometry_fit.build_geometry_fit_dataset_cache_log_lines(
        [
            {
                "dataset_index": 0,
                "label": "bg0.osc",
                "cache_metadata": {
                    "cache_action": "rebuilt",
                    "reused": False,
                    "rebuilt": True,
                    "stale_reason": "geometry-fit dataset prep rebuilds from fresh simulation rows (prefer_cache=False).",
                    "cache_source": "geometry_manual_simulated_peaks_for_params(prefer_cache=False)",
                    "cache_provenance": [
                        "geometry_manual_simulated_peaks_for_params(prefer_cache=False)",
                        "geometry_manual_simulated_lookup",
                        "build_geometry_manual_fit_dataset",
                    ],
                    "pair_count": 3,
                    "resolved_source_pair_count": 2,
                    "simulated_peak_count": 9,
                    "table_count": 1,
                    "table_summaries": [
                        {
                            "source_table_index": 4,
                            "nominal_hkl": [1, 1, 0],
                            "q_group_key": ["q", 1],
                            "qr": 1.25,
                            "qz": -0.5,
                            "row_count_before_grouping": 6,
                            "row_count_after_grouping": 2,
                            "dropped_nonfinite_row_count": 1,
                            "nominal_hkl_recovery_count": 1,
                            "merged_group_count": 3,
                            "representative_row_indices_kept": [3, 7],
                        }
                    ],
                },
            }
        ]
    )

    assert lines[0] == (
        "bg0.osc: cache_action=rebuilt reused=False rebuilt=True "
        "stale_reason=geometry-fit dataset prep rebuilds from fresh simulation rows "
        "(prefer_cache=False). "
        "source=geometry_manual_simulated_peaks_for_params(prefer_cache=False)"
    )
    assert "geometry_manual_simulated_lookup" in lines[1]
    assert lines[2] == ("bg0.osc: pair_count=3 resolved_source_pairs=2 simulated_peaks=9 tables=1")
    assert lines[3] == (
        "bg0.osc: table[4] nominal_hkl=[1, 1, 0] q_group_key=[q, 1] "
        "qr=1.250000 qz=-0.500000 rows_before=6 rows_after=2 "
        "dropped_nonfinite=1 nominal_hkl_recovery=1 merged_groups=3 "
        "representative_rows=[3, 7]"
    )


def test_build_geometry_fit_start_log_sections_include_dataset_cache_diagnostics() -> None:
    sections = geometry_fit.build_geometry_fit_start_log_sections(
        params={"gamma": 0.2},
        var_names=["gamma"],
        dataset_infos=[
            {
                "dataset_index": 0,
                "label": "bg0.osc",
                "summary_line": "bg[0] bg0.osc",
                "cache_metadata": {
                    "cache_action": "rebuilt",
                    "reused": False,
                    "rebuilt": True,
                    "stale_reason": "geometry-fit dataset prep rebuilds from fresh simulation rows (prefer_cache=False).",
                    "cache_source": "geometry_manual_simulated_peaks_for_params(prefer_cache=False)",
                    "cache_provenance": ["build_geometry_manual_fit_dataset"],
                    "pair_count": 2,
                    "resolved_source_pair_count": 1,
                    "simulated_peak_count": 5,
                    "table_count": 1,
                    "table_summaries": [
                        {
                            "source_table_index": 2,
                            "nominal_hkl": [1, 0, 2],
                            "q_group_key": ["q", 2],
                            "qr": 1.0,
                            "qz": 2.0,
                            "row_count_before_grouping": 3,
                            "row_count_after_grouping": 1,
                            "dropped_nonfinite_row_count": 0,
                            "nominal_hkl_recovery_count": 0,
                            "merged_group_count": 2,
                            "representative_row_indices_kept": [4],
                        }
                    ],
                },
            }
        ],
        current_dataset={
            "dataset_index": 0,
            "label": "bg0.osc",
            "group_count": 1,
            "pair_count": 2,
            "orientation_choice": {"label": "identity"},
            "orientation_diag": {
                "pairs": 2,
                "identity_rms_px": 1.0,
                "best_rms_px": 1.0,
                "reason": "ok",
            },
        },
        selected_background_indices=[0],
        joint_background_mode=False,
        geometry_runtime_cfg={"debug_logging": True},
    )

    assert any(
        title == "Geometry-fit dataset cache diagnostics:"
        and "bg0.osc: cache_action=rebuilt" in lines[0]
        for title, lines in sections
    )


def test_build_geometry_fit_source_resolution_log_lines_include_unresolved_pair_details() -> None:
    lines = geometry_fit.build_geometry_fit_source_resolution_log_lines(
        [
            {
                "label": "bg0.osc",
                "pair_count": 3,
                "resolved_source_pair_count": 1,
                "simulated_peak_count": 70,
                "simulated_lookup_count": 61,
                "source_resolution_diagnostics": [
                    {"pair_index": 0, "strict_resolved": True},
                    {
                        "pair_index": 1,
                        "strict_resolved": False,
                        "saved_source_row_key": (2, 5),
                        "saved_source_peak_key": None,
                        "saved_hkl": (1, 1, 0),
                        "saved_q_group_key": ("q", 1),
                        "saved_display_point": (50.0, 60.0),
                        "row_candidate_status": "missing",
                        "peak_candidate_status": "no_saved_peak",
                        "overlay_resolution_kind": "q_group_fallback",
                        "overlay_source_row_key": (0, 7),
                        "overlay_source_peak_key": None,
                        "overlay_hkl": (1, 1, 0),
                        "overlay_distance_px": 3.25,
                        "failure_reason": (
                            "saved source row is missing from the current simulated rows; "
                            "only a q-group fallback candidate was available"
                        ),
                    },
                ],
            }
        ]
    )

    assert lines[0] == (
        "bg0.osc: resolved_source_pairs=1/3 simulated_peaks=70 simulated_source_rows=61"
    )
    assert "saved_row=(2, 5)" in lines[1]
    assert "saved_hkl=[1, 1, 0]" in lines[1]
    assert "strict_row=missing" in lines[2]
    assert "fallback=q_group_fallback" in lines[2]
    assert "fallback_row=(0, 7)" in lines[2]
    assert "fallback_distance_px=3.250" in lines[2]
    assert "reason=saved source row is missing from the current simulated rows" in lines[3]


def test_build_geometry_fit_source_resolution_log_lines_note_fit_only_rebinds() -> None:
    lines = geometry_fit.build_geometry_fit_source_resolution_log_lines(
        [
            {
                "label": "bg0.osc",
                "pair_count": 1,
                "resolved_source_pair_count": 1,
                "simulated_peak_count": 1,
                "simulated_lookup_count": 0,
                "source_resolution_diagnostics": [
                    {
                        "pair_index": 0,
                        "strict_resolved": False,
                        "fit_resolved": True,
                        "fit_resolution_kind": "legacy_dense_q_group_rebind",
                    }
                ],
            }
        ]
    )

    assert lines[1] == (
        "bg0.osc: all saved manual pairs resolved for fit (including legacy source-id remaps)."
    )


def test_build_geometry_fit_source_resolution_log_lines_include_branch_details() -> None:
    lines = geometry_fit.build_geometry_fit_source_resolution_log_lines(
        [
            {
                "label": "bg0.osc",
                "pair_count": 1,
                "resolved_source_pair_count": 1,
                "simulated_peak_count": 2,
                "simulated_lookup_count": 2,
                "source_resolution_diagnostics": [
                    {
                        "pair_index": 0,
                        "strict_resolved": True,
                        "fit_resolved": True,
                        "fit_resolution_kind": "strict_source_match",
                        "saved_source_branch_index": 0,
                        "saved_source_branch_source": "caked_y",
                        "strict_source_branch_index": 0,
                        "strict_source_branch_source": "source_branch_index",
                        "fit_source_branch_index": 0,
                        "fit_source_branch_source": "source_branch_index",
                        "overlay_source_branch_index": 0,
                        "overlay_source_branch_source": "source_branch_index",
                    }
                ],
            }
        ]
    )

    assert lines[1] == "bg0.osc: all saved manual pairs strictly resolved."
    assert any(
        "saved_branch=0(caked_y)" in line
        and "fit_branch=0(source_branch_index)" in line
        and "fallback_branch=0(source_branch_index)" in line
        for line in lines[2:]
    )


def test_build_geometry_fit_simulation_diagnostic_log_lines_include_runtime_failure_details() -> (
    None
):
    lines = geometry_fit.build_geometry_fit_simulation_diagnostic_log_lines(
        [
            {
                "label": "bg0.osc",
                "simulated_peak_count": 0,
                "simulation_diagnostics": {
                    "source": "fresh",
                    "status": "simulation_exception",
                    "projected_peak_count": 0,
                    "raw_peak_count": 0,
                    "miller_shape": [96, 3],
                    "intensity_shape": [96],
                    "image_size": 2048,
                    "missing_param_keys": [],
                    "missing_mosaic_keys": ["beam_x_array"],
                    "param_summary": {
                        "a": 4.1,
                        "c": 28.6,
                        "lambda": 1.5406,
                        "theta_initial": 5.0,
                        "center": [1024.0, 1024.0],
                        "n2": 1.0,
                    },
                    "mosaic_array_sizes": {
                        "beam_x_array": 0,
                        "beam_y_array": 1,
                        "theta_array": 1,
                        "phi_array": 1,
                        "wavelength_array": 1,
                        "wavelength_i_array": 0,
                        "sample_weights": 0,
                    },
                    "runtime_simulation": {
                        "stage": "simulate_preview_style_peaks",
                        "status": "process_peaks_parallel_exception",
                        "hit_table_count": 0,
                        "nonempty_hit_table_count": 0,
                        "finite_hit_row_total": 0,
                        "hit_row_counts": [0, 0, 0],
                        "exception_type": "RuntimeError",
                        "exception_message": "boom",
                    },
                    "exception_type": "RuntimeError",
                    "exception_message": "boom",
                },
            }
        ]
    )

    assert "source=fresh status=simulation_exception" in lines[0]
    assert "miller_shape=[96, 3]" in lines[1]
    assert "missing_mosaic=[beam_x_array]" in lines[2]
    assert "params a=4.100000 c=28.600000" in lines[3]
    assert "runtime stage=simulate_preview_style_peaks" in lines[5]
    assert "runtime hit_row_counts=[0, 0, 0]" in lines[6]
    assert "exception=RuntimeError: boom" in lines[7]


def test_build_geometry_fit_preflight_log_sections_include_fresh_simulation_diagnostics() -> None:
    sections = geometry_fit.build_geometry_fit_preflight_log_sections(
        error_text="Geometry fit unavailable",
        params={"gamma": 0.2},
        var_names=["gamma"],
        dataset_infos=[
            {
                "dataset_index": 0,
                "label": "bg0.osc",
                "summary_line": "bg[0] bg0.osc",
                "pair_count": 2,
                "group_count": 1,
                "resolved_source_pair_count": 0,
                "simulation_diagnostics": {
                    "source": "fresh",
                    "status": "empty_simulated_peaks",
                    "projected_peak_count": 0,
                    "raw_peak_count": 0,
                    "miller_shape": [2, 3],
                    "intensity_shape": [2],
                    "image_size": 512,
                },
                "source_resolution_diagnostics": [],
            }
        ],
        current_dataset={
            "dataset_index": 0,
            "label": "bg0.osc",
            "pair_count": 2,
            "group_count": 1,
            "resolved_source_pair_count": 0,
            "orientation_choice": {"label": "identity"},
            "orientation_diag": {"pairs": 2, "reason": "ok"},
        },
        selected_background_indices=[0],
        joint_background_mode=False,
        geometry_runtime_cfg={"debug_logging": True},
    )

    assert any(title == "Fresh simulation diagnostics:" for title, _lines in sections)


def test_build_geometry_fit_preflight_log_sections_omit_debug_sections_by_default() -> None:
    sections = geometry_fit.build_geometry_fit_preflight_log_sections(
        error_text="Geometry fit unavailable",
        params={"gamma": 0.2},
        var_names=["gamma"],
        dataset_infos=[
            {
                "dataset_index": 0,
                "label": "bg0.osc",
                "summary_line": "bg[0] bg0.osc",
                "pair_count": 2,
                "group_count": 1,
                "resolved_source_pair_count": 0,
                "simulation_diagnostics": {
                    "source": "fresh",
                    "status": "empty_simulated_peaks",
                },
                "source_resolution_diagnostics": [{"pair_index": 0, "strict_resolved": False}],
            }
        ],
        current_dataset={
            "dataset_index": 0,
            "label": "bg0.osc",
            "pair_count": 2,
            "group_count": 1,
            "orientation_choice": {"label": "identity"},
            "orientation_diag": {"pairs": 2, "reason": "ok"},
        },
        selected_background_indices=[0],
        joint_background_mode=False,
        geometry_runtime_cfg={},
    )

    assert [title for title, _lines in sections] == [
        "Failure:",
        "Fitting variables (start values):",
        "Manual geometry datasets:",
        "Current orientation diagnostics:",
    ]


def test_build_geometry_fit_calibration_lines_report_pixel_size_provenance() -> None:
    lines = geometry_fit.build_geometry_fit_calibration_lines(
        {
            "pixel_size_m": 1.0e-4,
            "debye_x": 0.0,
            "debye_y": 0.0,
        }
    )

    assert lines[0] == "pixel_size_source=pixel_size_m value=0.000100"
    assert lines[1] == ("pixel_size=nan pixel_size_m=0.000100 debye_x=0.000000 debye_y=0.000000")
    assert lines[2] == "warning=debye_x <= 0; using pixel_size_m instead"


def test_build_geometry_fit_point_match_failure_reason_lines_summarize_unresolved_pairs() -> None:
    lines = geometry_fit.build_geometry_fit_point_match_failure_reason_lines(
        [
            {
                "dataset_index": 0,
                "dataset_label": "bg0.osc",
                "overlay_match_index": 7,
                "match_status": "missing_pair",
                "hkl": (1, 1, 0),
                "q_group_key": ("q", 1),
                "measured_x": 50.0,
                "measured_y": 60.0,
                "simulated_x": np.nan,
                "simulated_y": np.nan,
                "distance_px": np.nan,
                "measured_two_theta_deg": 11.25,
                "measured_phi_deg": 6.5,
                "delta_two_theta_deg": np.nan,
                "delta_phi_deg": np.nan,
                "resolution_reason": "match_radius_exceeded",
                "measured_resolution_reason": "measured_anchor_missing",
                "detector_resolution_reason": "detector_source_missing",
                "correspondence_resolution_reason": "source_row_out_of_range",
                "fit_source_resolution_kind": "legacy_dense_q_group_rebind",
            }
        ]
    )

    assert lines[0] == "match_status: missing_pair=1"
    assert lines[1] == "resolution_reason: match_radius_exceeded=1"
    assert lines[2] == "measured_resolution_reason: measured_anchor_missing=1"
    assert lines[3] == "detector_resolution_reason: detector_source_missing=1"
    assert lines[4] == "correspondence_resolution_reason: source_row_out_of_range=1"
    assert lines[5] == "fit_source_resolution_kind: legacy_dense_q_group_rebind=1"
    assert lines[6] == "dataset[0] bg0.osc: unresolved_pairs=1"
    assert "overlay_index=7" in lines[7]
    assert "measured_detector=[50.000, 60.000]" in lines[7]
    assert "measured_angles=[11.250, 6.500]" in lines[7]
    assert "simulated=<none>" in lines[8]
    assert "resolution_reason=source_row_out_of_range" in lines[8]


def test_build_geometry_fit_point_match_failure_reason_lines_include_branch_details() -> None:
    lines = geometry_fit.build_geometry_fit_point_match_failure_reason_lines(
        [
            {
                "dataset_index": 0,
                "dataset_label": "bg0.osc",
                "overlay_match_index": 7,
                "match_status": "missing_pair",
                "hkl": (1, 1, 0),
                "measured_x": 50.0,
                "measured_y": 60.0,
                "simulated_x": np.nan,
                "simulated_y": np.nan,
                "distance_px": np.nan,
                "measured_two_theta_deg": 11.25,
                "measured_phi_deg": 6.5,
                "delta_two_theta_deg": np.nan,
                "delta_phi_deg": np.nan,
                "resolution_reason": "match_radius_exceeded",
                "fit_source_resolution_kind": "legacy_dense_q_group_rebind",
                "source_branch_index": 1,
                "source_branch_resolution_source": "source_branch_index",
                "resolved_peak_index": 1,
            }
        ]
    )

    assert any(
        "source_branch=1(source_branch_index)" in line
        and "resolved_branch=1(resolved_peak_index)" in line
        and "fit_kind=legacy_dense_q_group_rebind" in line
        for line in lines
    )


def test_build_geometry_fit_solver_request_uses_prepared_run_payloads(
    monkeypatch,
) -> None:
    prepared_run, postprocess_config = _make_prepared_run(joint_background_mode=True)
    monkeypatch.setattr(
        geometry_fit,
        "apply_geometry_fit_runtime_safety_overrides",
        lambda cfg, **kwargs: (dict(cfg), None),
    )

    request = geometry_fit.build_geometry_fit_solver_request(
        prepared_run=prepared_run,
        var_names=["gamma", "a"],
        solver_inputs=postprocess_config.solver_inputs,
    )

    assert request.miller == "miller-data"
    assert request.intensities == "intensity-data"
    assert request.image_size == 512
    assert request.params == {"theta_initial": 3.0, "theta_offset": 0.0}
    assert request.measured_peaks == [{"x": 1.0, "y": 2.0}]
    assert request.var_names == ["gamma", "a"]
    assert not hasattr(request, "experimental_image")
    assert request.dataset_specs == [
        {"dataset_index": 0, "theta_initial": 3.0},
        {"dataset_index": 1, "theta_initial": 4.0},
    ]
    assert request.refinement_config == {"bounds": {"gamma": [0.0, 1.0]}}
    assert request.runtime_safety_note is None


def test_build_geometry_fit_solver_request_preserves_preflight_pair_identity_fields(
    monkeypatch,
) -> None:
    prepared_run, postprocess_config = _make_prepared_run(joint_background_mode=False)
    preflight_pair = {
        "pair_id": "bg0:pair0",
        "hkl": (1, 1, 0),
        "x": 30.0,
        "y": 40.0,
        "source_reflection_index": 7,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_branch_index": 1,
        "source_peak_index": 1,
    }
    prepared_run = replace(
        prepared_run,
        current_dataset={
            **prepared_run.current_dataset,
            "measured_for_fit": [dict(preflight_pair)],
        },
    )
    monkeypatch.setattr(
        geometry_fit,
        "apply_geometry_fit_runtime_safety_overrides",
        lambda cfg, **kwargs: (dict(cfg), None),
    )

    request = geometry_fit.build_geometry_fit_solver_request(
        prepared_run=prepared_run,
        var_names=["gamma"],
        solver_inputs=postprocess_config.solver_inputs,
    )

    assert len(request.measured_peaks) == 1
    solver_pair = dict(request.measured_peaks[0])
    for field in (
        "pair_id",
        "hkl",
        "source_reflection_index",
        "source_reflection_namespace",
        "source_reflection_is_full",
        "source_branch_index",
        "source_peak_index",
    ):
        assert solver_pair[field] == preflight_pair[field]


def test_solver_request_to_subset_mapping_and_seed_correspondence_preserves_trusted_identity(
    monkeypatch,
) -> None:
    def _base_params(image_size: int) -> dict[str, object]:
        return {
            "gamma": 0.0,
            "Gamma": 0.0,
            "corto_detector": 0.1,
            "theta_initial": 0.0,
            "theta_offset": 0.0,
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
            "optics_mode": 1,
            "mosaic_params": {
                "beam_x_array": np.array([0.0, 0.1], dtype=np.float64),
                "beam_y_array": np.zeros(2, dtype=np.float64),
                "theta_array": np.zeros(2, dtype=np.float64),
                "phi_array": np.zeros(2, dtype=np.float64),
                "sigma_mosaic_deg": 0.2,
                "gamma_mosaic_deg": 0.1,
                "eta": 0.05,
                "wavelength_array": np.ones(2, dtype=np.float64),
            },
        }

    def _fake_process(*args, **kwargs):
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

    def _fake_least_squares(residual_fn, x0, **kwargs):
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

    monkeypatch.setattr(
        geometry_fit,
        "apply_geometry_fit_runtime_safety_overrides",
        lambda cfg, **kwargs: (dict(cfg), None),
    )
    monkeypatch.setattr(opt, "_process_peaks_parallel_safe", _fake_process)
    monkeypatch.setattr(opt, "least_squares", _fake_least_squares)

    image_size = 20
    miller = np.array(
        [[5.0, 0.0, 0.0], [1.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    intensities = np.array([5.0, 10.0, 7.0], dtype=np.float64)
    trusted_pair = {
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
    prepared_run, postprocess_config = _make_prepared_run(joint_background_mode=False)
    prepared_run = replace(
        prepared_run,
        fit_params=_base_params(image_size),
        current_dataset={
            **prepared_run.current_dataset,
            "measured_for_fit": [dict(trusted_pair)],
            "experimental_image_for_fit": np.zeros(
                (image_size, image_size),
                dtype=np.float64,
            ),
        },
        dataset_specs=[
            {
                "dataset_index": 0,
                "label": "bg[0]",
                "theta_initial": 0.0,
                "experimental_image": np.zeros((image_size, image_size), dtype=np.float64),
            }
        ],
    )
    postprocess_config = replace(
        postprocess_config,
        solver_inputs=geometry_fit.GeometryFitRuntimeSolverInputs(
            miller=miller,
            intensities=intensities,
            image_size=image_size,
        ),
    )

    request = geometry_fit.build_geometry_fit_solver_request(
        prepared_run=prepared_run,
        var_names=["gamma"],
        solver_inputs=postprocess_config.solver_inputs,
    )
    solver_pair = dict(request.measured_peaks[0])
    dataset_contexts = opt._build_geometry_fit_dataset_contexts(
        request.miller,
        request.intensities,
        request.params,
        request.measured_peaks,
        None,
        request.dataset_specs,
    )
    subset_entry = dict(dataset_contexts[0].subset.measured_entries[0])

    result = geometry_fit.solve_geometry_fit_request(
        request,
        solve_fit=opt.fit_geometry_parameters,
    )

    seed_record = dict(result.full_beam_polish_summary["seed_correspondence_records"][0])
    full_beam_record = dict(result.point_match_diagnostics[0])
    fixed_groups = opt._build_geometry_fit_fixed_correspondence_groups(
        result.point_match_diagnostics
    )
    fixed_record = dict(fixed_groups[0][0])
    for record in (subset_entry, seed_record, full_beam_record, fixed_record):
        for field in (
            "pair_id",
            "hkl",
            "source_reflection_index",
            "source_reflection_namespace",
            "source_reflection_is_full",
            "source_branch_index",
            "source_peak_index",
        ):
            assert record.get(field) == solver_pair[field]
    assert seed_record["frozen_locator_kind"] == "trusted_branch"
    assert seed_record["frozen_table_namespace"] == "full_reflection"
    assert full_beam_record["resolution_kind"] == "fixed_source"
    assert full_beam_record["match_status"] == "matched"


def test_apply_geometry_fit_runtime_safety_overrides_for_windows_python_313() -> None:
    runtime_cfg, note = geometry_fit.apply_geometry_fit_runtime_safety_overrides(
        {
            "bounds": {"gamma": [0.0, 1.0]},
            "solver": {
                "workers": "auto",
                "parallel_mode": "auto",
                "worker_numba_threads": 0,
            },
            "use_numba": True,
        },
        platform_name="nt",
        version_info=(3, 13, 0),
        env={},
    )

    assert runtime_cfg == {
        "bounds": {"gamma": [0.0, 1.0]},
        "solver": {
            "workers": "auto",
            "parallel_mode": "auto",
            "worker_numba_threads": 0,
        },
        "use_numba": True,
    }
    assert note == (
        "Windows/Python 3.13 runtime guard enabled: "
        "unsafe runtime disabled, safe-wrapper Numba allowed."
    )


def test_apply_geometry_fit_runtime_safety_overrides_updates_optimizer_alias() -> None:
    runtime_cfg, note = geometry_fit.apply_geometry_fit_runtime_safety_overrides(
        {
            "optimizer": {
                "workers": "auto",
                "parallel_mode": "auto",
                "worker_numba_threads": 0,
            },
            "solver": {
                "workers": "auto",
                "parallel_mode": "auto",
                "worker_numba_threads": 0,
            },
            "use_numba": True,
        },
        platform_name="nt",
        version_info=(3, 13, 0),
        env={},
    )

    assert runtime_cfg["use_numba"] is True
    assert runtime_cfg["optimizer"] == {
        "workers": "auto",
        "parallel_mode": "auto",
        "worker_numba_threads": 0,
    }
    assert runtime_cfg["solver"] == runtime_cfg["optimizer"]
    assert note == (
        "Windows/Python 3.13 runtime guard enabled: "
        "unsafe runtime disabled, safe-wrapper Numba allowed."
    )


def test_apply_geometry_fit_runtime_safety_overrides_honors_opt_out_env() -> None:
    runtime_cfg, note = geometry_fit.apply_geometry_fit_runtime_safety_overrides(
        {
            "solver": {
                "workers": "auto",
                "parallel_mode": "auto",
            },
            "use_numba": True,
        },
        platform_name="nt",
        version_info=(3, 13, 0),
        env={"RA_SIM_ALLOW_UNSAFE_GEOMETRY_FIT_RUNTIME": "1"},
    )

    assert runtime_cfg == {
        "solver": {
            "workers": "auto",
            "parallel_mode": "auto",
        },
        "use_numba": True,
    }
    assert note is None


def test_apply_geometry_fit_runtime_safety_overrides_honors_config_opt_out() -> None:
    runtime_cfg, note = geometry_fit.apply_geometry_fit_runtime_safety_overrides(
        {
            "solver": {
                "workers": "auto",
                "parallel_mode": "auto",
            },
            "use_numba": True,
            "allow_unsafe_runtime": True,
        },
        platform_name="nt",
        version_info=(3, 13, 0),
        env={},
    )

    assert runtime_cfg == {
        "solver": {
            "workers": "auto",
            "parallel_mode": "auto",
        },
        "use_numba": True,
        "allow_unsafe_runtime": True,
    }
    assert note is None


def test_solve_geometry_fit_request_forwards_status_callback_when_supported() -> None:
    prepared_run, postprocess_config = _make_prepared_run(joint_background_mode=False)
    request = geometry_fit.build_geometry_fit_solver_request(
        prepared_run=prepared_run,
        var_names=["gamma", "a"],
        solver_inputs=postprocess_config.solver_inputs,
    )
    events: list[object] = []

    def _solve_fit(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks,
        var_names,
        *,
        pixel_tol,
        experimental_image,
        dataset_specs,
        refinement_config,
        status_callback,
    ):
        status_callback("solver-stage")
        events.append(
            {
                "miller": miller,
                "intensities": intensities,
                "image_size": image_size,
                "params": dict(params),
                "measured_peaks": list(measured_peaks),
                "var_names": list(var_names),
                "pixel_tol": pixel_tol,
                "experimental_image": experimental_image,
                "dataset_specs": dataset_specs,
                "refinement_config": dict(refinement_config),
            }
        )
        return "solver-result"

    result = geometry_fit.solve_geometry_fit_request(
        request,
        solve_fit=_solve_fit,
        status_callback=lambda text: events.append(("status", text)),
    )

    assert result == "solver-result"
    assert events == [
        ("status", "solver-stage"),
        {
            "miller": "miller-data",
            "intensities": "intensity-data",
            "image_size": 512,
            "params": {"theta_initial": 3.0, "theta_offset": 0.0},
            "measured_peaks": [{"x": 1.0, "y": 2.0}],
            "var_names": ["gamma", "a"],
            "pixel_tol": float("inf"),
            "experimental_image": None,
            "dataset_specs": [
                {
                    "dataset_index": 0,
                    "theta_initial": 3.0,
                    "measured_peaks": [{"x": 1.0, "y": 2.0}],
                }
            ],
            "refinement_config": {
                "bounds": {"gamma": [0.0, 1.0]},
                "use_numba": False,
                "solver": {
                    "parallel_mode": "auto",
                    "workers": "auto",
                    "worker_numba_threads": 0,
                },
            },
        },
    ]


def test_apply_joint_geometry_fit_runtime_safety_overrides_only_changes_joint_runs() -> None:
    base_cfg = {
        "solver": {
            "workers": "auto",
            "parallel_mode": "auto",
            "worker_numba_threads": 0,
            "restarts": 4,
            "stagnation_probe": True,
            "stagnation_probe_pairwise": True,
            "stagnation_probe_random_directions": 2,
        },
        "identifiability": {"enabled": True},
        "use_numba": True,
        "bounds": {"gamma": [0.0, 1.0]},
    }

    unchanged = geometry_fit.apply_joint_geometry_fit_runtime_safety_overrides(
        base_cfg,
        joint_background_mode=False,
    )
    changed = geometry_fit.apply_joint_geometry_fit_runtime_safety_overrides(
        base_cfg,
        joint_background_mode=True,
    )

    assert unchanged == base_cfg
    assert changed == {
        "solver": {
            "workers": "auto",
            "parallel_mode": "auto",
            "worker_numba_threads": 0,
            "restarts": 1,
            "stagnation_probe": False,
            "stagnation_probe_pairwise": False,
            "stagnation_probe_random_directions": 0,
        },
        "identifiability": {"enabled": False},
        "use_numba": True,
        "bounds": {"gamma": [0.0, 1.0]},
    }
    assert base_cfg["solver"]["workers"] == "auto"
    assert base_cfg["identifiability"]["enabled"] is True


def test_apply_joint_geometry_fit_runtime_safety_overrides_honors_unsafe_runtime_opt_in() -> None:
    base_cfg = {
        "solver": {
            "workers": "auto",
            "parallel_mode": "auto",
            "worker_numba_threads": 0,
            "restarts": 4,
        },
        "identifiability": {"enabled": True},
        "use_numba": True,
        "allow_unsafe_runtime": True,
        "bounds": {"gamma": [0.0, 1.0]},
    }

    changed = geometry_fit.apply_joint_geometry_fit_runtime_safety_overrides(
        base_cfg,
        joint_background_mode=True,
    )

    assert changed == base_cfg


def test_apply_manual_point_geometry_fit_runtime_overrides_forces_single_model_path() -> None:
    base_cfg = {
        "solver": {
            "workers": "auto",
            "parallel_mode": "auto",
            "worker_numba_threads": 0,
            "dynamic_point_geometry_fit": True,
            "loss": "soft_l1",
            "f_scale_px": 6.0,
            "weighted_matching": True,
            "use_measurement_uncertainty": True,
            "anisotropic_measurement_uncertainty": True,
            "restarts": 3,
            "restart_jitter": 0.25,
            "stagnation_probe": True,
            "stagnation_probe_pairwise": True,
            "stagnation_probe_random_directions": 2,
            "staged_release": {"enabled": True},
            "reparameterize_pairs": {"enabled": True},
            "missing_pair_penalty_deg": 7.5,
        },
        "seed_search": {
            "prescore_top_k": 8,
            "n_global": 24,
            "n_jitter": 12,
            "jitter_sigma_u": 0.5,
        },
        "identifiability": {
            "enabled": True,
            "auto_freeze": True,
            "selective_thaw": {"enabled": True},
            "adaptive_regularization": {"enabled": True},
        },
        "discrete_modes": {
            "enabled": True,
            "rot90": [0, 1, 2, 3],
            "flip_x": [False, True],
            "flip_y": [False, True],
        },
        "full_beam_polish": {"enabled": True},
        "ridge_refinement": {"enabled": True},
        "image_refinement": {"enabled": True},
    }

    changed = geometry_fit.apply_manual_point_geometry_fit_runtime_overrides(
        base_cfg,
        joint_background_mode=True,
    )

    assert changed["solver"]["manual_point_fit_mode"] is True
    assert changed["solver"]["missing_pair_penalty_px"] == 20.0
    assert changed["solver"]["missing_pair_penalty_deg"] == 7.5
    assert changed["solver"]["q_group_line_constraints"] is True
    assert changed["solver"]["q_group_line_angle_weight"] == 0.6
    assert changed["solver"]["q_group_line_offset_weight"] == 1.0
    assert changed["solver"]["q_group_line_missing_penalty_scale"] == 0.35
    assert changed["solver"]["hk0_peak_priority_weight"] == 6.0
    assert changed["solver"]["workers"] == "auto"
    assert changed["solver"]["parallel_mode"] == "auto"
    assert changed["solver"]["worker_numba_threads"] == 0
    assert "dynamic_point_geometry_fit" not in changed["solver"]
    assert changed["solver"]["loss"] == "linear"
    assert changed["solver"]["f_scale_px"] == 1.0
    assert changed["solver"]["weighted_matching"] is False
    assert changed["solver"]["use_measurement_uncertainty"] is False
    assert changed["solver"]["anisotropic_measurement_uncertainty"] is False
    assert "restarts" not in changed["solver"]
    assert "restart_jitter" not in changed["solver"]
    assert "stagnation_probe" not in changed["solver"]
    assert "stagnation_probe_pairwise" not in changed["solver"]
    assert "stagnation_probe_random_directions" not in changed["solver"]
    assert "staged_release" not in changed["solver"]
    assert "reparameterize_pairs" not in changed["solver"]
    assert changed["optimizer"] == changed["solver"]
    assert changed["seed_search"] == {
        "prescore_top_k": 1,
        "n_global": 0,
        "n_jitter": 0,
        "jitter_sigma_u": 0.5,
    }
    assert changed["use_numba"] is False
    assert changed["allow_unsafe_runtime"] is False
    assert changed["sampling"] == {"fit_sample_count": 8}
    assert changed["discrete_modes"] == {
        "enabled": False,
        "rot90": [0, 1, 2, 3],
        "flip_x": [False, True],
        "flip_y": [False, True],
    }
    assert changed["identifiability"]["enabled"] is True
    assert "auto_freeze" not in changed["identifiability"]
    assert "selective_thaw" not in changed["identifiability"]
    assert "adaptive_regularization" not in changed["identifiability"]
    assert "full_beam_polish" not in changed
    assert "ridge_refinement" not in changed
    assert "image_refinement" not in changed


def test_apply_manual_point_geometry_fit_runtime_overrides_preserves_unsafe_parallel_settings() -> (
    None
):
    base_cfg = {
        "solver": {
            "workers": "auto",
            "parallel_mode": "auto",
            "worker_numba_threads": 0,
            "loss": "soft_l1",
        },
        "use_numba": True,
        "allow_unsafe_runtime": True,
    }

    changed = geometry_fit.apply_manual_point_geometry_fit_runtime_overrides(
        base_cfg,
        joint_background_mode=False,
    )

    assert changed["solver"]["manual_point_fit_mode"] is True
    assert changed["solver"]["workers"] == "auto"
    assert changed["solver"]["parallel_mode"] == "auto"
    assert changed["solver"]["worker_numba_threads"] == 0
    assert changed["solver"]["loss"] == "linear"
    assert changed["use_numba"] is True
    assert changed["allow_unsafe_runtime"] is True
    assert changed["sampling"] == {"fit_sample_count": 8}


def test_apply_manual_point_geometry_fit_runtime_overrides_preserves_safe_wrapper_numba() -> None:
    base_cfg = {
        "solver": {
            "workers": "auto",
            "parallel_mode": "auto",
            "worker_numba_threads": 0,
        },
        "use_numba": True,
        "allow_unsafe_runtime": False,
    }

    changed = geometry_fit.apply_manual_point_geometry_fit_runtime_overrides(
        base_cfg,
        joint_background_mode=False,
    )

    assert changed["use_numba"] is True
    assert changed["allow_unsafe_runtime"] is False
    assert changed["sampling"] == {"fit_sample_count": 8}


def test_apply_dynamic_point_geometry_fit_runtime_overrides_preserves_richer_solver_path() -> None:
    base_cfg = {
        "solver": {
            "manual_point_fit_mode": True,
            "dynamic_point_geometry_fit": False,
            "workers": "auto",
            "parallel_mode": "auto",
            "worker_numba_threads": 0,
            "loss": "soft_l1",
            "f_scale_px": 8.0,
            "weighted_matching": True,
            "use_measurement_uncertainty": True,
            "anisotropic_measurement_uncertainty": True,
            "restarts": 3,
        },
        "full_beam_polish": {"enabled": True, "max_nfev": 40},
        "use_numba": True,
        "allow_unsafe_runtime": True,
    }

    changed = geometry_fit.apply_dynamic_point_geometry_fit_runtime_overrides(
        base_cfg,
        joint_background_mode=False,
    )

    assert changed["solver"]["dynamic_point_geometry_fit"] is True
    assert "manual_point_fit_mode" not in changed["solver"]
    assert changed["solver"]["loss"] == "soft_l1"
    assert changed["solver"]["f_scale_px"] == 8.0
    assert changed["solver"]["weighted_matching"] is True
    assert changed["solver"]["use_measurement_uncertainty"] is True
    assert changed["solver"]["anisotropic_measurement_uncertainty"] is True
    assert changed["solver"]["restarts"] == 3
    assert changed["full_beam_polish"] == {"enabled": True, "max_nfev": 40}
    assert changed["use_numba"] is True
    assert changed["allow_unsafe_runtime"] is True


def test_prepare_geometry_fit_run_rejects_dataset_without_orientation_anchor_pairs() -> None:
    result = geometry_fit.prepare_geometry_fit_run(
        params={"theta_initial": 4.0},
        var_names=["gamma"],
        fit_config={},
        osc_files=["C:/data/bg0.osc"],
        current_background_index=0,
        theta_initial=4.0,
        preserve_live_theta=False,
        apply_geometry_fit_background_selection=lambda **kwargs: True,
        current_geometry_fit_background_indices=lambda **kwargs: [0],
        geometry_fit_uses_shared_theta_offset=lambda *_args, **_kwargs: False,
        apply_background_theta_metadata=lambda **kwargs: True,
        current_background_theta_values=lambda **kwargs: [4.0],
        current_geometry_theta_offset=lambda **kwargs: 0.0,
        geometry_manual_pairs_for_index=lambda idx: [{"pair": idx}],
        ensure_geometry_fit_caked_view=lambda: None,
        build_dataset=lambda background_index, **kwargs: {
            "dataset_index": int(background_index),
            "label": "bg0.osc",
            "pair_count": 2,
            "group_count": 1,
            "summary_line": "bg[0]",
            "spec": {"dataset_index": int(background_index)},
            "orientation_choice": {"label": "identity"},
            "orientation_diag": {"pairs": 0, "reason": "insufficient_pairs"},
            "resolved_source_pair_count": 1,
            "measured_for_fit": [],
            "experimental_image_for_fit": "image-0",
            "initial_pairs_display": [],
            "native_background": np.zeros((2, 2)),
        },
        build_runtime_config=lambda fit_params: dict(fit_params),
    )

    assert result.prepared_run is None
    assert result.error_text == (
        "Geometry fit unavailable: orientation preflight produced no usable "
        "simulated/measured anchor pairs for bg0.osc. Refresh the picks before "
        "fitting."
    )


def test_prepare_geometry_fit_run_rejects_fallback_only_manual_pairs() -> None:
    result = geometry_fit.prepare_geometry_fit_run(
        params={"theta_initial": 4.0},
        var_names=["gamma"],
        fit_config={},
        osc_files=["C:/data/bg0.osc", "C:/data/bg1.osc"],
        current_background_index=0,
        theta_initial=4.0,
        preserve_live_theta=False,
        apply_geometry_fit_background_selection=lambda **kwargs: True,
        current_geometry_fit_background_indices=lambda **kwargs: [0, 1],
        geometry_fit_uses_shared_theta_offset=lambda indices: len(indices) > 1,
        apply_background_theta_metadata=lambda **kwargs: True,
        current_background_theta_values=lambda **kwargs: [4.0, 5.0],
        current_geometry_theta_offset=lambda **kwargs: 0.0,
        geometry_manual_pairs_for_index=lambda idx: [{"pair": idx}],
        ensure_geometry_fit_caked_view=lambda: None,
        build_dataset=lambda background_index, **kwargs: {
            "dataset_index": int(background_index),
            "label": f"bg{int(background_index)}.osc",
            "pair_count": 2,
            "group_count": 1,
            "summary_line": f"bg[{int(background_index)}]",
            "spec": {"dataset_index": int(background_index)},
            "orientation_choice": {"label": "identity"},
            "orientation_diag": {"pairs": 2, "reason": "ok"},
            "resolved_source_pair_count": 0,
            "measured_for_fit": [],
            "experimental_image_for_fit": f"image-{int(background_index)}",
            "initial_pairs_display": [],
            "native_background": np.zeros((2, 2)),
        },
        build_runtime_config=lambda fit_params: dict(fit_params),
    )

    assert result.prepared_run is None
    assert result.error_text == (
        "Geometry fit unavailable: saved manual pairs no longer resolve to "
        "current simulated source rows on any selected background. Refresh "
        "the picks before fitting."
    )


def test_prepare_geometry_fit_run_rejects_partially_unresolved_manual_pairs() -> None:
    result = geometry_fit.prepare_geometry_fit_run(
        params={"theta_initial": 4.0},
        var_names=["gamma"],
        fit_config={},
        osc_files=["C:/data/bg0.osc", "C:/data/bg1.osc"],
        current_background_index=0,
        theta_initial=4.0,
        preserve_live_theta=False,
        apply_geometry_fit_background_selection=lambda **kwargs: True,
        current_geometry_fit_background_indices=lambda **kwargs: [0, 1],
        geometry_fit_uses_shared_theta_offset=lambda indices: len(indices) > 1,
        apply_background_theta_metadata=lambda **kwargs: True,
        current_background_theta_values=lambda **kwargs: [4.0, 5.0],
        current_geometry_theta_offset=lambda **kwargs: 0.0,
        geometry_manual_pairs_for_index=lambda idx: [{"pair": idx}],
        ensure_geometry_fit_caked_view=lambda: None,
        build_dataset=lambda background_index, **kwargs: {
            "dataset_index": int(background_index),
            "label": f"bg{int(background_index)}.osc",
            "pair_count": 2,
            "group_count": 1,
            "summary_line": f"bg[{int(background_index)}]",
            "spec": {"dataset_index": int(background_index)},
            "orientation_choice": {"label": "identity"},
            "orientation_diag": {"pairs": 2, "reason": "ok"},
            "resolved_source_pair_count": 1 if int(background_index) == 0 else 2,
            "measured_for_fit": [],
            "experimental_image_for_fit": f"image-{int(background_index)}",
            "initial_pairs_display": [],
            "native_background": np.zeros((2, 2)),
        },
        build_runtime_config=lambda fit_params: dict(fit_params),
    )

    assert result.prepared_run is None
    assert result.error_text == (
        "Geometry fit unavailable: some saved manual pairs no longer resolve "
        "to current simulated source rows: bg0.osc (1/2). Refresh the picks "
        "before fitting."
    )


def test_geometry_fit_dataset_cache_helpers_copy_and_validate_dataset_bundle() -> None:
    prepared_run, _postprocess_config = _make_prepared_run(joint_background_mode=True)
    prepared_run = replace(
        prepared_run,
        dataset_infos=[
            {
                "dataset_index": 0,
                "label": "bg0.osc",
                "summary_line": "bg[0]",
                "cache_metadata": {
                    "cache_action": "rebuilt",
                    "reused": False,
                    "rebuilt": True,
                    "stale_reason": None,
                    "cache_source": "geometry_manual_simulated_peaks_for_params(prefer_cache=False)",
                    "cache_provenance": ["build_geometry_manual_fit_dataset"],
                    "table_count": 1,
                    "table_summaries": [
                        {
                            "source_table_index": 0,
                            "nominal_hkl": [1, 1, 0],
                            "q_group_key": ["q", 1],
                            "qr": 1.25,
                            "qz": -0.5,
                            "row_count_before_grouping": 2,
                            "row_count_after_grouping": 1,
                            "dropped_nonfinite_row_count": 0,
                            "nominal_hkl_recovery_count": 1,
                            "merged_group_count": 1,
                            "representative_row_indices_kept": [4],
                        }
                    ],
                },
            },
            {"dataset_index": 1, "label": "bg1.osc", "summary_line": "bg[1]"},
        ],
    )

    payload = geometry_fit.build_geometry_fit_dataset_cache_payload(
        prepared_run,
        current_background_index=1,
    )

    assert payload["selected_background_indices"] == [0, 1]
    assert payload["current_background_index"] == 1
    assert payload["joint_background_mode"] is True
    assert payload["background_theta_values"] == [3.0, 4.0]
    assert payload["dataset_specs"] == [
        {"dataset_index": 0, "theta_initial": 3.0},
        {"dataset_index": 1, "theta_initial": 4.0},
    ]
    assert payload["cache_metadata"] == {
        "cache_action": "rebuilt",
        "reused": False,
        "rebuilt": True,
        "stale_reason": None,
        "cache_source": "build_geometry_fit_dataset_cache_payload",
        "cache_provenance": [
            "build_geometry_manual_fit_dataset",
            "build_geometry_fit_dataset_cache_payload",
        ],
        "dataset_count": 2,
        "dataset_cache_metadata": [
            {
                "cache_action": "rebuilt",
                "reused": False,
                "rebuilt": True,
                "stale_reason": None,
                "cache_source": "geometry_manual_simulated_peaks_for_params(prefer_cache=False)",
                "cache_provenance": ["build_geometry_manual_fit_dataset"],
                "table_count": 1,
                "table_summaries": [
                    {
                        "source_table_index": 0,
                        "nominal_hkl": [1, 1, 0],
                        "q_group_key": ["q", 1],
                        "qr": 1.25,
                        "qz": -0.5,
                        "row_count_before_grouping": 2,
                        "row_count_after_grouping": 1,
                        "dropped_nonfinite_row_count": 0,
                        "nominal_hkl_recovery_count": 1,
                        "merged_group_count": 1,
                        "representative_row_indices_kept": [4],
                    }
                ],
                "dataset_index": 0,
                "label": "bg0.osc",
            }
        ],
    }
    assert (
        geometry_fit.geometry_fit_dataset_cache_stale_reason(
            payload,
            selected_background_indices=[0, 1],
            current_background_index=1,
            joint_background_mode=True,
            background_theta_values=[3.0, 4.0],
        )
        is None
    )
    assert (
        geometry_fit.geometry_fit_dataset_cache_stale_reason(
            payload,
            selected_background_indices=[0],
            current_background_index=1,
            joint_background_mode=True,
            background_theta_values=[3.0, 4.0],
        )
        == "Geometry-fit background selection changed. Rerun geometry fit."
    )
    assert (
        geometry_fit.geometry_fit_dataset_cache_stale_reason(
            payload,
            selected_background_indices=[0, 1],
            current_background_index=0,
            joint_background_mode=True,
            background_theta_values=[3.0, 4.0],
        )
        == "Active background changed since geometry fit. Rerun geometry fit."
    )


def test_build_geometry_fit_export_records_pairs_simulated_and_measured_rows() -> None:
    rows = geometry_fit.build_geometry_fit_export_records(
        agg_millers=[(1, 1, 0), (2, 0, 1)],
        agg_sim_coords=[(10.2, 20.8), (30.1, 40.9)],
        agg_meas_coords=[(11.0, 21.0), (29.0, 39.0)],
        pixel_offsets=[
            ((1, 1, 0), 0.8, 0.2, 0.8246),
            ((2, 0, 1), -1.1, -1.9, 2.1954),
        ],
    )

    assert rows == [
        {
            "source": "sim",
            "hkl": (1, 1, 0),
            "x": 10,
            "y": 20,
            "dist_px": 0.8246,
        },
        {
            "source": "sim",
            "hkl": (2, 0, 1),
            "x": 30,
            "y": 40,
            "dist_px": 2.1954,
        },
        {
            "source": "meas",
            "hkl": (1, 1, 0),
            "x": 11,
            "y": 21,
            "dist_px": 0.8246,
        },
        {
            "source": "meas",
            "hkl": (2, 0, 1),
            "x": 29,
            "y": 39,
            "dist_px": 2.1954,
        },
    ]


def test_build_geometry_fit_export_records_uses_authoritative_point_match_diagnostics() -> None:
    rows = geometry_fit.build_geometry_fit_export_records(
        [
            {
                "dataset_index": 1,
                "overlay_match_index": 7,
                "match_status": "matched",
                "hkl": [1, 1, 0],
                "measured_x": 11.5,
                "measured_y": 19.0,
                "simulated_x": 10.0,
                "simulated_y": 20.0,
                "dx_px": -1.5,
                "dy_px": 1.0,
                "distance_px": np.hypot(-1.5, 1.0),
                "source_table_index": 2,
                "source_row_index": 5,
                "source_peak_index": 9,
                "resolution_kind": "fixed_source_row",
                "correspondence_resolution_reason": "matched source row",
            }
        ]
    )

    assert rows == [
        {
            "dataset_index": 1,
            "overlay_match_index": 7,
            "match_status": "matched",
            "hkl": (1, 1, 0),
            "measured_x": 11.5,
            "measured_y": 19.0,
            "simulated_x": 10.0,
            "simulated_y": 20.0,
            "dx_px": -1.5,
            "dy_px": 1.0,
            "distance_px": np.hypot(-1.5, 1.0),
            "source_table_index": 2,
            "source_row_index": 5,
            "source_peak_index": 9,
            "resolution_kind": "fixed_source_row",
            "resolution_reason": "matched source row",
        }
    ]


def test_geometry_fit_post_solver_helpers_format_diagnostics_and_summary() -> None:
    class _Result:
        success = True
        status = 3
        message = "done"
        nfev = 17
        cost = 1.25
        robust_cost = 1.0
        weighted_residual_rms_px = 0.5
        solver_loss = "soft_l1"
        solver_f_scale = 2.0
        optimality = 0.005
        active_mask = [0, 1]
        optimizer_method = "trf"
        final_metric_name = "central_point_match"
        chosen_discrete_mode = {"rot90": 1}
        bound_hits = ["Gamma"]
        boundary_warning = (
            "Possible identifiability issue: parameters finished near bounds (Gamma=upper)."
        )
        restart_history = [{"restart": 0, "cost": 2.5, "success": False, "message": "retry"}]
        rms_px = 0.75
        x = [1.5, 2.5]
        point_match_summary = {
            "matched_pair_count": 4,
            "fixed_source_resolved_count": 3,
            "unweighted_peak_rms_px": 0.75,
            "unweighted_peak_max_px": 2.0,
        }
        identifiability_summary = {"underconstrained": False}
        geometry_fit_debug_summary = {
            "point_match_mode": True,
            "dataset_count": 1,
            "var_names": ["gamma", "a"],
            "solver": {
                "loss": "soft_l1",
                "f_scale_px": 2.0,
                "max_nfev": 40,
                "restarts": 2,
                "weighted_matching": True,
                "missing_pair_penalty_px": 12.0,
                "use_measurement_uncertainty": True,
                "anisotropic_measurement_uncertainty": False,
                "full_beam_polish_enabled": True,
                "full_beam_polish_match_radius_px": 24.0,
            },
            "parallelization": {
                "mode": "datasets",
                "configured_workers": 4,
                "dataset_workers": 2,
                "restart_workers": 1,
                "worker_numba_threads": 1,
                "numba_thread_budget": 8,
            },
            "seed_search": {
                "prescore_top_k": 2,
                "n_global": 3,
                "n_jitter": 1,
                "jitter_sigma_u": 0.25,
                "min_seed_separation_u": 0.2,
                "trusted_prior_fraction_of_span": 0.15,
            },
            "discrete_modes": [
                {"label": "identity"},
                {"rot90": 1},
            ],
            "mode_seed_summary": [
                {
                    "mode": {"label": "identity"},
                    "seed_count": 5,
                    "selected_seed_count": 2,
                    "selected_seeds": [
                        {
                            "seed_kind": "zero",
                            "seed_label": "u=0",
                            "cost": 1.75,
                        }
                    ],
                }
            ],
            "selected_mode_label": "rot90=1",
            "solve_counts": {
                "prescored": 6,
                "solved": 2,
            },
            "main_solve_seed": {
                "seed_kind": "zero",
                "seed_label": "u=0",
                "cost": 3.5,
                "weighted_rms_px": 1.25,
                "point_match_summary": {
                    "matched_pair_count": 4,
                    "missing_pair_count": 1,
                    "unweighted_peak_rms_px": 1.5,
                    "unweighted_peak_max_px": 2.0,
                },
            },
            "dataset_entries": [
                {
                    "dataset_index": 0,
                    "label": "bg0",
                    "theta_initial_deg": 3.0,
                    "measured_count": 4,
                    "subset_reflection_count": 6,
                    "total_reflection_count": 8,
                    "fixed_source_reflection_count": 2,
                    "fallback_hkl_count": 1,
                    "subset_reduced": True,
                }
            ],
            "parameter_entries": [
                {
                    "name": "gamma",
                    "group": "tilt",
                    "start": 1.0,
                    "final": 1.5,
                    "delta": 0.5,
                    "lower_bound": -1.0,
                    "upper_bound": 2.0,
                    "scale": 0.25,
                    "prior_enabled": True,
                    "prior_center": 1.2,
                    "prior_sigma": 0.1,
                }
            ],
            "final": {
                "cost": 1.25,
                "robust_cost": 1.0,
                "weighted_rms_px": 0.5,
                "final_full_beam_rms_px": 0.75,
                "metric_name": "central_point_match",
            },
            "solve_progress": {
                "label": "main solve",
                "evaluation_count": 12,
                "status_emit_count": 4,
                "best_cost_seen": 1.0,
                "last_cost_seen": 1.1,
                "best_weighted_rms_px": 0.5,
                "last_weighted_rms_px": 0.55,
                "aborted_early": False,
                "trace": [
                    {
                        "eval": 1,
                        "reason": "milestone",
                        "current_cost": 3.5,
                        "best_cost": 3.5,
                        "weighted_rms_px": 1.25,
                    }
                ],
            },
        }
        reparameterization_summary = {
            "status": "accepted",
            "reason": "accepted",
            "accepted": True,
            "final_cost": 0.9,
            "nfev": 5,
        }
        staged_release_summary = {
            "status": "skipped",
            "reason": "disabled_by_config",
            "accepted": False,
        }
        adaptive_regularization_summary = {
            "status": "accepted",
            "accepted": True,
            "applied_parameters": ["Gamma"],
            "release_accepted": True,
        }
        full_beam_polish_summary = {
            "status": "accepted",
            "reason": "accepted",
            "accepted": True,
            "seed_correspondence_count": 4,
            "matched_pair_count_before": 4,
            "matched_pair_count_after": 4,
            "fixed_source_resolved_count": 3,
            "start_rms_px": 1.5,
            "final_rms_px": 0.75,
            "nfev": 3,
            "max_nfev": 40,
        }
        ridge_refinement_summary = {"status": "skipped", "accepted": False}
        image_refinement_summary = {"status": "skipped", "accepted": False}
        auto_freeze_summary = {"status": "accepted", "accepted": True, "fixed_parameters": ["a"]}
        selective_thaw_summary = {
            "status": "accepted",
            "accepted": True,
            "thawed_parameters": ["a"],
        }

    diagnostics = geometry_fit.build_geometry_fit_optimizer_diagnostics_lines(_Result())
    assert diagnostics[:4] == [
        "success=True",
        "status=3",
        "message=done",
        "nfev=17",
    ]
    assert "optimizer_method=trf" in diagnostics
    assert "weighted_residual_rms_px=0.500000" in diagnostics
    assert "display_rms_px=0.750000" in diagnostics
    assert "final_metric_name=central_point_match" in diagnostics
    assert "solver_discrete_mode=rot90=1" in diagnostics
    assert "bound_hits=[Gamma]" in diagnostics
    assert diagnostics[-1] == "restart[0] cost=2.500000 success=False msg=retry"

    debug_lines = geometry_fit.build_geometry_fit_debug_lines(_Result())
    assert debug_lines[0] == "point_match_mode=True datasets=1 vars=gamma,a"
    assert any("seed_search prescore_top_k=2 n_global=3 n_jitter=1" in line for line in debug_lines)
    assert any(
        "discrete_modes count=2 selected=rot90=1 labels=[identity, rot90=1]" in line
        for line in debug_lines
    )
    assert any(
        "mode_seed[0] label=identity total_seeds=5 selected_seeds=2" in line for line in debug_lines
    )
    assert any("main_seed_point_match matched=4 missing=1" in line for line in debug_lines)
    assert any("main_seed kind=zero label=u=0 cost=3.500000" in line for line in debug_lines)
    assert any("seed_rms_px=1.500000" in line for line in debug_lines)
    assert any("dataset[0] label=bg0" in line for line in debug_lines)
    assert any(
        "param[gamma] group=tilt start=1.000000 final=1.500000 delta=0.500000" in line
        for line in debug_lines
    )
    assert any(
        "full_beam_polish=True full_beam_radius_px=24.000000" in line for line in debug_lines
    )
    assert any("final metric=central_point_match cost=1.250000" in line for line in debug_lines)
    assert any("solve_counts prescored=6 solved=2" in line for line in debug_lines)
    assert any(
        "solve_progress label=main solve evaluations=12" in line and "last_cost=1.100000" in line
        for line in debug_lines
    )
    assert any("solve_progress[0] eval=1 reason=milestone" in line for line in debug_lines)

    stage_lines = geometry_fit.build_geometry_fit_stage_summary_lines(_Result())
    assert any(
        line.startswith("reparameterization:")
        and "status=accepted" in line
        and "final_cost=0.900000" in line
        for line in stage_lines
    )
    assert any(
        line.startswith("adaptive_regularization:") and "applied_parameters=[Gamma]" in line
        for line in stage_lines
    )
    assert any(
        line.startswith("full_beam_polish:")
        and "seed_correspondence_count=4" in line
        and "final_rms_px=0.750000" in line
        and "rms_delta_px=-0.750000" in line
        for line in stage_lines
    )
    assert any(
        line.startswith("auto_freeze:") and "fixed_parameters=[a]" in line for line in stage_lines
    )

    result_lines = geometry_fit.build_geometry_fit_result_lines(
        ["gamma", "a"],
        _Result().x,
        rms=0.75,
    )
    assert result_lines == [
        "gamma = 1.500000",
        "a = 2.500000",
        "RMS residual = 0.750000 px",
    ]

    overlay_lines = geometry_fit.build_geometry_fit_overlay_diagnostic_lines(
        {
            "paired_records": 3,
            "sim_display_med_px": 1.2,
            "bg_display_med_px": 2.3,
            "sim_display_p90_px": 4.5,
            "bg_display_p90_px": 5.6,
        },
        overlay_record_count=7,
    )
    assert overlay_lines == [
        "transform_rule=sim:native_to_overlay_display; bg:inverse_orientation_then_overlay_display",
        "overlay_records=7",
        "paired_records=3",
        "sim_display_med_px=1.200",
        "bg_display_med_px=2.300",
        "sim_display_p90_px=4.500",
        "bg_display_p90_px=5.600",
    ]

    summary_lines = geometry_fit.build_geometry_fit_summary_lines(
        current_dataset={
            "group_count": 4,
            "pair_count": 9,
            "orientation_choice": {"label": "rotate"},
        },
        overlay_record_count=7,
        var_names=["gamma", "a"],
        values=_Result().x,
        rms=0.75,
        save_path="C:/tmp/matched.npy",
        result=_Result(),
    )
    assert summary_lines[:4] == [
        "manual_groups=4",
        "manual_points=9",
        "overlay_records=7",
        "orientation=rotate",
    ]
    assert "solver_discrete_mode=rot90=1" in summary_lines
    assert "final_metric=central_point_match" in summary_lines
    assert "matched_pairs=4" in summary_lines
    assert "matched_peak_rms_px=0.750000" in summary_lines
    assert "underconstrained=False" in summary_lines
    assert summary_lines[-4:] == [
        "gamma = 1.500000",
        "a = 2.500000",
        "RMS residual = 0.750000 px",
        "Matched peaks saved to: C:/tmp/matched.npy",
    ]


def test_geometry_fit_post_solver_helpers_build_runtime_params_offsets_and_status_text() -> None:
    fitted = geometry_fit.build_geometry_fit_fitted_params(
        {"lambda": 1.54, "mosaic_params": {"sigma": 0.1}},
        zb=0.1,
        zs=0.2,
        theta_initial=3.0,
        theta_offset=0.4,
        chi=0.5,
        cor_angle=0.6,
        psi_z=0.7,
        gamma=0.8,
        Gamma=0.9,
        corto_detector=1.0,
        a=4.2,
        c=32.1,
        center_x=100.0,
        center_y=200.0,
    )
    assert fitted["lambda"] == 1.54
    assert fitted["theta_offset"] == 0.4
    assert fitted["center"] == [100.0, 200.0]
    assert fitted["a"] == 4.2

    pixel_offsets = geometry_fit.build_geometry_fit_pixel_offsets(
        [(1, 1, 0), (2, 0, 1)],
        [(10.0, 20.0), (30.5, 40.5)],
        [(8.5, 19.0), (31.0, 41.0)],
    )
    assert pixel_offsets == [
        ((1, 1, 0), 1.5, 1.0, np.hypot(1.5, 1.0)),
        ((2, 0, 1), -0.5, -0.5, np.hypot(-0.5, -0.5)),
    ]

    pixel_lines = geometry_fit.build_geometry_fit_pixel_offset_lines(pixel_offsets)
    assert pixel_lines == [
        "HKL=(1, 1, 0): dx=1.5000, dy=1.0000, |Δ|=1.8028 px",
        "HKL=(2, 0, 1): dx=-0.5000, dy=-0.5000, |Δ|=0.7071 px",
    ]

    filtered_overlay = geometry_fit.filter_geometry_fit_overlay_point_match_diagnostics(
        [
            {"dataset_index": 0, "match": "drop"},
            {"dataset_index": 1, "match": "keep"},
            "ignore",
        ],
        joint_background_mode=True,
        current_background_index=1,
    )
    assert filtered_overlay == [{"dataset_index": 1, "match": "keep"}]

    point_summary_lines = geometry_fit.build_geometry_fit_point_match_summary_lines(
        {"claimed": 4, "qualified": 3}
    )
    assert point_summary_lines == ["claimed=4", "qualified=3"]

    progress_text = geometry_fit.build_geometry_fit_progress_text(
        current_dataset={
            "pair_count": 9,
            "group_count": 4,
            "orientation_choice": {"label": "rotate"},
        },
        dataset_count=3,
        joint_background_mode=True,
        var_names=["gamma", "a"],
        values=[1.5, 2.5],
        rms=0.75,
        pixel_offsets=pixel_offsets,
        export_record_count=4,
        save_path="C:/tmp/matched.npy",
        log_path="C:/tmp/fit_log.txt",
        frame_warning="frame warning",
    )
    assert "Manual geometry fit complete:" in progress_text
    assert "gamma = 1.5000" in progress_text
    assert "Orientation = rotate" in progress_text
    assert "Manual pairs: 9 points across 4 groups | joint backgrounds=3" in progress_text
    assert "Saved 4 peak records → C:/tmp/matched.npy" in progress_text
    assert "frame warning" in progress_text
    assert "HKL=(1, 1, 0): dx=1.5000, dy=1.0000, |Δ|=1.8028 px" in progress_text


def test_build_geometry_fit_point_match_summary_lines_warns_for_fallback_only_matches() -> None:
    lines = geometry_fit.build_geometry_fit_point_match_summary_lines(
        {
            "measured_count": 3,
            "fixed_source_resolved_count": 0,
            "fallback_entry_count": 3,
        }
    )

    assert lines == [
        "WARNING: fit used only HKL-fallback correspondences; no fixed source-row anchors resolved.",
        "fallback_entry_count=3",
        "fixed_source_resolved_count=0",
        "measured_count=3",
    ]


def test_geometry_fit_result_rms_falls_back_to_residual_vector() -> None:
    class _Result:
        rms_px = float("nan")
        fun = np.array([3.0, 4.0], dtype=np.float64)

    assert np.isclose(
        geometry_fit.geometry_fit_result_rms(_Result()),
        5.0 / np.sqrt(2.0),
    )


def test_build_geometry_fit_rejection_reason_lines_flags_underconstraint_and_bounds() -> None:
    result = SimpleNamespace(
        success=True,
        point_match_summary={
            "matched_pair_count": 3,
            "unweighted_peak_max_px": 2.5,
        },
        identifiability_summary={"underconstrained": True},
        bound_proximity_summary={"near_bound_parameters": [{"name": "Gamma", "side": "upper"}]},
    )

    reasons = geometry_fit.build_geometry_fit_rejection_reason_lines(result, rms=0.75)

    assert reasons == [
        "Fit is underconstrained according to the final identifiability diagnostics.",
        "Parameters finished within 1% of a finite bound span from a bound: Gamma(upper).",
    ]


def test_build_geometry_fit_rejection_reason_lines_ignores_bound_proximity_without_underconstraint() -> (
    None
):
    result = SimpleNamespace(
        success=True,
        point_match_summary={
            "matched_pair_count": 7,
            "unweighted_peak_max_px": 23.14,
        },
        identifiability_summary={"underconstrained": False},
        bound_proximity_summary={
            "near_bound_parameters": [{"name": "theta_initial", "side": "lower"}]
        },
    )

    reasons = geometry_fit.build_geometry_fit_rejection_reason_lines(result, rms=9.65)

    assert reasons == []


def test_build_geometry_fit_profile_cache_merges_mosaic_and_current_fit_values() -> None:
    profile_cache = geometry_fit.build_geometry_fit_profile_cache(
        {"existing": 1, "gamma": -1.0},
        {"sigma_mosaic_deg": 0.2},
        theta_initial=3.0,
        theta_offset=0.4,
        cor_angle=0.6,
        chi=0.5,
        zs=0.2,
        zb=0.1,
        gamma=0.8,
        Gamma=0.9,
        corto_detector=1.0,
        a=4.2,
        c=32.1,
        center_x=100.0,
        center_y=200.0,
    )

    assert profile_cache == {
        "existing": 1,
        "gamma": 0.8,
        "sigma_mosaic_deg": 0.2,
        "theta_initial": 3.0,
        "theta_offset": 0.4,
        "cor_angle": 0.6,
        "chi": 0.5,
        "zs": 0.2,
        "zb": 0.1,
        "Gamma": 0.9,
        "corto_detector": 1.0,
        "a": 4.2,
        "c": 32.1,
        "center_x": 100.0,
        "center_y": 200.0,
    }


def test_build_runtime_geometry_fit_result_bindings_uses_current_ui_snapshot() -> None:
    gamma_var = _DummyVar(0.0)
    a_var = _DummyVar(0.0)
    theta_offset_var = _DummyVar("")

    bindings = geometry_fit.build_runtime_geometry_fit_result_bindings(
        fit_params={"lambda": 1.54},
        base_profile_cache={"existing": 1},
        mosaic_params={"sigma_mosaic_deg": 0.2},
        current_ui_params=lambda: {
            "zb": 0.1,
            "zs": 0.2,
            "theta_initial": 3.0,
            "theta_offset": float(theta_offset_var.get() or 0.0),
            "chi": 0.5,
            "cor_angle": 0.6,
            "psi_z": 0.7,
            "gamma": gamma_var.get(),
            "Gamma": 0.9,
            "corto_detector": 1.0,
            "a": a_var.get(),
            "c": 32.1,
            "center_x": 100.0,
            "center_y": 200.0,
        },
        var_map={
            "gamma": gamma_var,
            "a": a_var,
        },
        geometry_theta_offset_var=theta_offset_var,
        log_section=lambda *_args, **_kwargs: None,
        capture_undo_state=lambda: {"undo": True},
        sync_joint_background_theta=lambda: None,
        refresh_status=lambda: None,
        update_manual_pick_button_label=lambda: None,
        replace_profile_cache=lambda _cache: None,
        push_undo_state=lambda _state: None,
        request_preview_skip_once=lambda: None,
        mark_last_simulation_dirty=lambda: None,
        schedule_update=lambda: None,
        postprocess_result=lambda fitted_params, rms: geometry_fit.GeometryFitPostprocessResult(
            fitted_params=fitted_params,
            point_match_summary_lines=[],
            pixel_offsets=[],
            overlay_records=[],
            overlay_state={},
            overlay_diagnostic_lines=[],
            frame_warning=None,
            export_records=[],
            save_path="matched.npy",
            fit_summary_lines=[],
            progress_text=f"rms={rms}",
        ),
        draw_overlay_records=lambda *_args, **_kwargs: None,
        draw_initial_pairs_overlay=lambda *_args, **_kwargs: None,
        set_last_overlay_state=lambda _state: None,
        save_export_records=lambda *_args, **_kwargs: None,
        set_progress_text=lambda _text: None,
        cmd_line=lambda _text: None,
    )

    bindings.apply_result_values(["gamma", "theta_offset", "a"], [1.25, 0.4, 4.5])

    assert gamma_var.get() == 1.25
    assert a_var.get() == 4.5
    assert theta_offset_var.get() == "0.4"
    assert bindings.build_profile_cache() == {
        "existing": 1,
        "sigma_mosaic_deg": 0.2,
        "theta_initial": 3.0,
        "theta_offset": 0.4,
        "cor_angle": 0.6,
        "chi": 0.5,
        "zs": 0.2,
        "zb": 0.1,
        "gamma": 1.25,
        "Gamma": 0.9,
        "corto_detector": 1.0,
        "a": 4.5,
        "c": 32.1,
        "center_x": 100.0,
        "center_y": 200.0,
    }
    assert bindings.build_fitted_params() == {
        "lambda": 1.54,
        "zb": 0.1,
        "zs": 0.2,
        "theta_initial": 3.0,
        "psi_z": 0.7,
        "chi": 0.5,
        "cor_angle": 0.6,
        "gamma": 1.25,
        "Gamma": 0.9,
        "corto_detector": 1.0,
        "a": 4.5,
        "c": 32.1,
        "center_x": 100.0,
        "center_y": 200.0,
        "center": [100.0, 200.0],
        "theta_offset": 0.4,
    }


def test_capture_runtime_geometry_fit_undo_state_builds_overlay_fallback_from_manual_pairs() -> (
    None
):
    calls: list[tuple[str, object]] = []

    snapshot = geometry_fit.capture_runtime_geometry_fit_undo_state(
        current_ui_params=lambda: {"gamma": 1.25},
        current_profile_cache=lambda: {"profile": 1},
        copy_state_value=geometry_fit.copy_geometry_fit_state_value,
        last_overlay_state=lambda: None,
        build_initial_pairs_display=(
            lambda background_index, *, param_set, prefer_cache: (
                calls.append(
                    (
                        "build",
                        {
                            "background_index": background_index,
                            "param_set": dict(param_set),
                            "prefer_cache": prefer_cache,
                        },
                    )
                )
                or ([], [{"pair": "saved"}])
            )
        ),
        current_background_index=lambda: 2,
        current_fit_params=lambda: {"a": 4.2},
        pending_pairs_display=lambda: [{"pair": "pending"}],
    )

    assert calls == [
        (
            "build",
            {
                "background_index": 2,
                "param_set": {"a": 4.2},
                "prefer_cache": True,
            },
        )
    ]
    assert snapshot == {
        "ui_params": {"gamma": 1.25},
        "profile_cache": {"profile": 1},
        "overlay_state": {
            "overlay_records": [],
            "initial_pairs_display": [{"pair": "saved"}, {"pair": "pending"}],
            "max_display_markers": 2,
        },
    }


def test_capture_runtime_geometry_fit_undo_state_accepts_numpy_overlay_payloads() -> None:
    calls: list[object] = []

    snapshot = geometry_fit.capture_runtime_geometry_fit_undo_state(
        current_ui_params=lambda: {"gamma": 1.25},
        current_profile_cache=lambda: {"profile": 1},
        copy_state_value=geometry_fit.copy_geometry_fit_state_value,
        last_overlay_state=lambda: {
            "overlay_records": np.array([{"overlay": True}], dtype=object),
        },
        build_initial_pairs_display=(
            lambda *args, **kwargs: calls.append(("build", args, kwargs)) or ([], [])
        ),
        current_background_index=lambda: 2,
        current_fit_params=lambda: {"a": 4.2},
        pending_pairs_display=lambda: [],
    )

    assert calls == []
    overlay_records = snapshot["overlay_state"]["overlay_records"]
    assert isinstance(overlay_records, np.ndarray)
    assert overlay_records.tolist() == [{"overlay": True}]


def test_redraw_runtime_geometry_fit_overlay_state_prefers_overlay_records() -> None:
    events: list[object] = []

    rendered = geometry_fit.redraw_runtime_geometry_fit_overlay_state(
        {
            "overlay_records": [{"overlay": True}],
            "initial_pairs_display": [{"pair": 1}],
            "max_display_markers": 5,
        },
        draw_overlay_records=lambda records, marker_limit: events.append(
            ("draw_overlay", list(records), marker_limit)
        ),
        draw_initial_pairs_overlay=lambda pairs, marker_limit: events.append(
            ("draw_initial", list(pairs), marker_limit)
        ),
    )

    assert rendered is True
    assert events == [("draw_overlay", [{"overlay": True}], 5)]


def test_redraw_runtime_geometry_fit_overlay_state_falls_back_to_initial_pairs() -> None:
    events: list[object] = []

    rendered = geometry_fit.redraw_runtime_geometry_fit_overlay_state(
        {
            "initial_pairs_display": np.array([{"pair": 1}], dtype=object),
            "max_display_markers": 3,
        },
        draw_overlay_records=lambda records, marker_limit: events.append(
            ("draw_overlay", list(records), marker_limit)
        ),
        draw_initial_pairs_overlay=lambda pairs, marker_limit: events.append(
            ("draw_initial", list(pairs), marker_limit)
        ),
    )

    assert rendered is True
    assert events == [("draw_initial", [{"pair": 1}], 3)]


def test_restore_runtime_geometry_fit_undo_state_applies_state_and_redraws_overlay() -> None:
    gamma_var = _DummyVar(0.0)
    a_var = _DummyVar(0.0)
    theta_offset_var = _DummyVar("")
    stored_profile_cache: dict[str, object] = {}
    stored_overlay_state: dict[str, object] | None = None
    events: list[object] = []

    restored = geometry_fit.restore_runtime_geometry_fit_undo_state(
        {
            "ui_params": {"gamma": 1.5, "a": 4.4, "theta_offset": 0.25},
            "profile_cache": {"profile": 2},
            "overlay_state": {
                "overlay_records": [{"overlay": True}],
                "max_display_markers": 5,
            },
        },
        var_map={"gamma": gamma_var, "a": a_var},
        geometry_theta_offset_var=theta_offset_var,
        replace_profile_cache=lambda profile_cache: stored_profile_cache.update(profile_cache),
        set_last_overlay_state=lambda overlay_state: events.append(
            ("overlay_state", overlay_state)
        ),
        request_preview_skip_once=lambda: events.append("skip_preview_once"),
        mark_last_simulation_dirty=lambda: events.append("mark_dirty"),
        cancel_pending_update=lambda: events.append("cancel_pending"),
        run_update=lambda: events.append("run_update"),
        draw_overlay_records=lambda records, marker_limit: events.append(
            ("draw_overlay", list(records), marker_limit)
        ),
        draw_initial_pairs_overlay=lambda pairs, marker_limit: events.append(
            ("draw_initial", list(pairs), marker_limit)
        ),
        refresh_status=lambda: events.append("refresh"),
        update_manual_pick_button_label=lambda: events.append("button"),
    )

    assert gamma_var.get() == 1.5
    assert a_var.get() == 4.4
    assert theta_offset_var.get() == "0.25"
    assert stored_profile_cache == {"profile": 2}
    assert restored["overlay_state"] == {
        "overlay_records": [{"overlay": True}],
        "max_display_markers": 5,
    }
    assert events == [
        ("overlay_state", {"overlay_records": [{"overlay": True}], "max_display_markers": 5}),
        "skip_preview_once",
        "mark_dirty",
        "cancel_pending",
        "run_update",
        ("draw_overlay", [{"overlay": True}], 5),
        "refresh",
        "button",
    ]


def test_restore_runtime_geometry_fit_undo_state_normalizes_numpy_overlay_payloads() -> None:
    events: list[object] = []

    restored = geometry_fit.restore_runtime_geometry_fit_undo_state(
        {
            "ui_params": {},
            "profile_cache": {"profile": 2},
            "overlay_state": {
                "overlay_records": np.array([{"overlay": True}], dtype=object),
                "initial_pairs_display": np.array([{"pair": 1}], dtype=object),
                "max_display_markers": 5,
            },
        },
        var_map={},
        geometry_theta_offset_var=None,
        replace_profile_cache=lambda _profile_cache: None,
        set_last_overlay_state=lambda _overlay_state: None,
        draw_overlay_records=lambda records, marker_limit: events.append(
            ("draw_overlay", list(records), marker_limit)
        ),
        draw_initial_pairs_overlay=lambda pairs, marker_limit: events.append(
            ("draw_initial", list(pairs), marker_limit)
        ),
    )

    assert restored["overlay_state"]["overlay_records"].tolist() == [{"overlay": True}]
    assert events == [("draw_overlay", [{"overlay": True}], 5)]


def test_build_runtime_geometry_fit_undo_restore_callback_defers_live_bindings(
    monkeypatch,
) -> None:
    calls: list[tuple[dict[str, object], dict[str, object]]] = []
    first_var = _DummyVar(0.0)
    second_var = _DummyVar(1.0)
    live_var_map = {"gamma": first_var}
    live_theta_offset_var = _DummyVar("")

    def fake_restore_runtime_state(state: dict[str, object], **kwargs: object) -> dict[str, object]:
        calls.append((dict(state), dict(kwargs)))
        return {"restored": True}

    monkeypatch.setattr(
        geometry_fit,
        "restore_runtime_geometry_fit_undo_state",
        fake_restore_runtime_state,
    )

    callback = geometry_fit.build_runtime_geometry_fit_undo_restore_callback(
        var_map_factory=lambda: live_var_map,
        geometry_theta_offset_var_factory=lambda: live_theta_offset_var,
        replace_profile_cache=lambda _cache: None,
        set_last_overlay_state=lambda _state: None,
        run_update=lambda: None,
    )

    live_var_map = {"gamma": second_var}
    live_theta_offset_var = _DummyVar("0.25")

    restored = callback({"saved": True})

    assert restored == {"restored": True}
    assert len(calls) == 1
    passed_state, passed_kwargs = calls[0]
    assert passed_state == {"saved": True}
    assert passed_kwargs["var_map"] is live_var_map
    assert passed_kwargs["geometry_theta_offset_var"] is live_theta_offset_var


def test_undo_runtime_geometry_fit_restores_and_commits_history() -> None:
    events: list[object] = []

    handled = geometry_fit.undo_runtime_geometry_fit(
        has_history=lambda: True,
        capture_current_state=lambda: {"current": True},
        read_undo_state=lambda: {"saved": True},
        restore_state=lambda state: events.append(("restore", dict(state))),
        commit_undo=lambda state: events.append(("commit", dict(state))),
        update_button_state=lambda: events.append("update_buttons"),
        set_progress_text=lambda text: events.append(("progress", text)),
    )

    assert handled is True
    assert events == [
        ("restore", {"saved": True}),
        ("commit", {"current": True}),
        "update_buttons",
        ("progress", "Restored the previous geometry-fit state."),
    ]


def test_redo_runtime_geometry_fit_reports_restore_failure() -> None:
    events: list[object] = []

    handled = geometry_fit.redo_runtime_geometry_fit(
        has_history=lambda: True,
        capture_current_state=lambda: {"current": True},
        read_redo_state=lambda: {"saved": True},
        restore_state=lambda _state: (_ for _ in ()).throw(RuntimeError("bad state")),
        commit_redo=lambda state: events.append(("commit", dict(state))),
        update_button_state=lambda: events.append("update_buttons"),
        set_progress_text=lambda text: events.append(("progress", text)),
    )

    assert handled is False
    assert events == [("progress", "Failed to redo geometry fit: bad state")]


def test_build_runtime_geometry_fit_history_callbacks_composes_undo_and_redo() -> None:
    history_state = SimpleNamespace(undo_stack=[{"undo": True}], redo_stack=[{"redo": True}])
    events: list[tuple[str, object]] = []
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_undo_runtime_geometry_fit(**kwargs: object) -> bool:
        calls.append(("undo", dict(kwargs)))
        return True

    def fake_redo_runtime_geometry_fit(**kwargs: object) -> bool:
        calls.append(("redo", dict(kwargs)))
        return False

    old_undo = geometry_fit.undo_runtime_geometry_fit
    old_redo = geometry_fit.redo_runtime_geometry_fit
    geometry_fit.undo_runtime_geometry_fit = fake_undo_runtime_geometry_fit
    geometry_fit.redo_runtime_geometry_fit = fake_redo_runtime_geometry_fit
    try:
        callbacks = geometry_fit.build_runtime_geometry_fit_history_callbacks(
            history_state=history_state,
            capture_current_state=lambda: {"current": True},
            restore_state=lambda state: events.append(("restore", dict(state))),
            copy_state_value=lambda value: {"copied": value},
            history_limit=lambda: 7,
            peek_last_undo_state=(
                lambda state, *, copy_state_value: (
                    events.append(("peek_undo", state, copy_state_value("undo"))) or {"undo": True}
                )
            ),
            peek_last_redo_state=(
                lambda state, *, copy_state_value: (
                    events.append(("peek_redo", state, copy_state_value("redo"))) or {"redo": True}
                )
            ),
            commit_undo=(
                lambda state, current_state, *, copy_state_value, limit: events.append(
                    (
                        "commit_undo",
                        state,
                        dict(current_state),
                        copy_state_value("undo"),
                        limit,
                    )
                )
            ),
            commit_redo=(
                lambda state, current_state, *, copy_state_value, limit: events.append(
                    (
                        "commit_redo",
                        state,
                        dict(current_state),
                        copy_state_value("redo"),
                        limit,
                    )
                )
            ),
            update_button_state=lambda: events.append(("buttons", None)),
            set_progress_text=lambda text: events.append(("progress", text)),
        )

        assert callbacks.undo() is True
        assert callbacks.redo() is False
    finally:
        geometry_fit.undo_runtime_geometry_fit = old_undo
        geometry_fit.redo_runtime_geometry_fit = old_redo

    assert [name for name, _ in calls] == ["undo", "redo"]
    undo_kwargs = calls[0][1]
    redo_kwargs = calls[1][1]
    assert undo_kwargs["has_history"]() is True
    assert redo_kwargs["has_history"]() is True
    assert undo_kwargs["read_undo_state"]() == {"undo": True}
    assert redo_kwargs["read_redo_state"]() == {"redo": True}
    undo_kwargs["commit_undo"]({"current": True})
    redo_kwargs["commit_redo"]({"current": True})
    assert events == [
        ("peek_undo", history_state, {"copied": "undo"}),
        ("peek_redo", history_state, {"copied": "redo"}),
        ("commit_undo", history_state, {"current": True}, {"copied": "undo"}, 7),
        ("commit_redo", history_state, {"current": True}, {"copied": "redo"}, 7),
    ]


def test_make_runtime_geometry_tool_action_callbacks_refreshes_button_state_and_label() -> None:
    events: list[object] = []
    label_calls: list[dict[str, object]] = []

    callbacks = geometry_fit.make_runtime_geometry_tool_action_callbacks(
        geometry_fit_history_state=SimpleNamespace(undo_stack=[{"undo": True}], redo_stack=[]),
        manual_pick_armed=lambda: False,
        set_manual_pick_armed=lambda enabled: events.append(("armed", enabled)),
        current_background_index=lambda: 3,
        current_pick_session=lambda: {"step": 2},
        manual_pick_session_active=lambda: False,
        build_manual_pick_button_label=lambda **kwargs: (
            label_calls.append(dict(kwargs)) or "Pick Label"
        ),
        pairs_for_index=lambda index: [{"index": index}],
        pair_group_count=lambda index: int(index) + 1,
        set_manual_pick_text=lambda text: events.append(("label", text)),
        set_history_button_state=lambda can_undo, can_redo: events.append(
            ("history", can_undo, can_redo)
        ),
    )

    callbacks.update_fit_history_button_state()
    callbacks.update_manual_pick_button_label()

    assert events == [
        ("history", True, False),
        ("label", "Pick Label"),
    ]
    assert len(label_calls) == 1
    assert label_calls[0]["armed"] is False
    assert label_calls[0]["current_background_index"] == 3
    assert label_calls[0]["pick_session"] == {"step": 2}
    assert callable(label_calls[0]["pairs_for_index"])
    assert callable(label_calls[0]["pair_group_count"])


def test_make_runtime_geometry_tool_action_callbacks_sets_manual_pick_mode() -> None:
    events: list[object] = []
    armed = {"value": False}

    class _Widget:
        def __init__(self):
            self.cursor = None

        def configure(self, **kwargs):
            self.cursor = kwargs.get("cursor")

    widget = _Widget()
    show_caked_2d_var = _DummyVar(False)

    callbacks = geometry_fit.make_runtime_geometry_tool_action_callbacks(
        geometry_fit_history_state=SimpleNamespace(undo_stack=[], redo_stack=[]),
        manual_pick_armed=lambda: armed["value"],
        set_manual_pick_armed=lambda enabled: armed.__setitem__("value", bool(enabled)),
        current_background_index=lambda: 0,
        current_pick_session=lambda: None,
        manual_pick_session_active=lambda: False,
        build_manual_pick_button_label=lambda **kwargs: "Manual Pick Armed",
        pairs_for_index=lambda index: [],
        pair_group_count=lambda index: 0,
        set_manual_pick_text=lambda text: events.append(("label", text)),
        show_caked_2d_var=show_caked_2d_var,
        toggle_caked_2d=lambda: events.append("toggle-caked"),
        set_hkl_pick_mode=lambda enabled, message=None: events.append(
            ("hkl-pick", enabled, message)
        ),
        set_geometry_preview_exclude_mode=lambda enabled, message=None: events.append(
            ("preview-exclude", enabled, message)
        ),
        cancel_manual_pick_session=lambda **kwargs: events.append(("cancel", kwargs)),
        canvas_widget=lambda: widget,
        set_progress_text=lambda text: events.append(("progress", text)),
    )

    callbacks.set_manual_pick_mode(True, "armed")
    callbacks.set_manual_pick_mode(False, "disarmed")

    assert armed["value"] is False
    assert show_caked_2d_var.get() is False
    assert widget.cursor == ""
    assert events == [
        ("hkl-pick", False, None),
        ("preview-exclude", False, None),
        ("label", "Manual Pick Armed"),
        ("progress", "armed"),
        ("cancel", {"restore_view": True, "redraw": True}),
        ("label", "Manual Pick Armed"),
        ("progress", "disarmed"),
    ]


def test_make_runtime_geometry_tool_action_callbacks_does_not_force_caked_view_when_arming() -> (
    None
):
    events: list[object] = []
    armed = {"value": False}

    callbacks = geometry_fit.make_runtime_geometry_tool_action_callbacks(
        geometry_fit_history_state=SimpleNamespace(undo_stack=[], redo_stack=[]),
        manual_pick_armed=lambda: armed["value"],
        set_manual_pick_armed=lambda enabled: armed.__setitem__("value", bool(enabled)),
        current_background_index=lambda: 0,
        current_pick_session=lambda: None,
        manual_pick_session_active=lambda: False,
        build_manual_pick_button_label=lambda **kwargs: "Manual Pick Armed",
        pairs_for_index=lambda index: [],
        pair_group_count=lambda index: 0,
        show_caked_2d_var=_DummyVar(False),
        toggle_caked_2d=lambda: events.append("toggle-caked"),
        ensure_geometry_fit_caked_view=lambda: events.append("ensure-caked"),
        set_hkl_pick_mode=lambda enabled, message=None: events.append(
            ("hkl-pick", enabled, message)
        ),
        set_geometry_preview_exclude_mode=lambda enabled, message=None: events.append(
            ("preview-exclude", enabled, message)
        ),
    )

    callbacks.set_manual_pick_mode(True, "armed")

    assert armed["value"] is True
    assert events == [
        ("hkl-pick", False, None),
        ("preview-exclude", False, None),
    ]


def test_make_runtime_geometry_tool_action_callbacks_clears_current_pairs() -> None:
    events: list[object] = []

    callbacks = geometry_fit.make_runtime_geometry_tool_action_callbacks(
        geometry_fit_history_state=SimpleNamespace(undo_stack=[], redo_stack=[]),
        manual_pick_armed=lambda: False,
        set_manual_pick_armed=lambda enabled: events.append(("armed", enabled)),
        current_background_index=lambda: 2,
        current_pick_session=lambda: None,
        manual_pick_session_active=lambda: False,
        build_manual_pick_button_label=lambda **kwargs: "Cleared Label",
        pairs_for_index=lambda index: [{"index": index}],
        pair_group_count=lambda index: 1,
        set_manual_pick_text=lambda text: events.append(("label", text)),
        cancel_manual_pick_session=lambda **kwargs: events.append(("cancel", kwargs)),
        push_manual_undo_state=lambda: events.append("push-undo"),
        clear_pairs_for_current_background=lambda index: events.append(("clear-pairs", index)),
        clear_geometry_pick_artists=lambda: events.append("clear-artists"),
        refresh_status=lambda: events.append("refresh"),
        set_progress_text=lambda text: events.append(("progress", text)),
    )

    callbacks.clear_current_manual_pairs()

    assert events == [
        "push-undo",
        ("cancel", {"restore_view": True, "redraw": False}),
        ("clear-pairs", 2),
        "clear-artists",
        ("label", "Cleared Label"),
        "refresh",
        ("progress", "Cleared saved geometry pairs for the current background image."),
    ]


def test_build_runtime_geometry_fit_execution_setup_wires_stateful_runtime_callbacks(
    tmp_path,
) -> None:
    prepared_run, postprocess_config = _make_prepared_run(
        joint_background_mode=True,
        tmp_path=tmp_path,
        stamp="20260328_123000",
    )

    class _SimulationState:
        def __init__(self):
            self.profile_cache = {"existing": 1}
            self.last_simulation_signature = "sig"

    class _BackgroundState:
        def __init__(self):
            self.current_background_index = 1

    simulation_state = _SimulationState()
    background_state = _BackgroundState()
    theta_initial_var = _DummyVar(0.0)
    progress_events: list[object] = []

    setup = geometry_fit.build_runtime_geometry_fit_execution_setup(
        prepared_run=prepared_run,
        mosaic_params={"sigma_mosaic_deg": 0.2},
        stamp="20260328_123000",
        downloads_dir=tmp_path,
        simulation_runtime_state=simulation_state,
        background_runtime_state=background_state,
        theta_initial_var=theta_initial_var,
        geometry_theta_offset_var=None,
        current_ui_params=lambda: {},
        var_map={},
        background_theta_for_index=lambda idx, strict_count=False: 7.5,
        refresh_status=lambda: progress_events.append("refresh"),
        update_manual_pick_button_label=lambda: progress_events.append("button"),
        capture_undo_state=lambda: {"undo": True},
        push_undo_state=lambda state: progress_events.append(("push", dict(state))),
        replace_dataset_cache=lambda payload: progress_events.append(("cache", payload)),
        request_preview_skip_once=lambda: progress_events.append("skip"),
        schedule_update=lambda: progress_events.append("schedule"),
        draw_overlay_records=lambda records, marker_limit: progress_events.append(
            ("draw_overlay", marker_limit)
        ),
        draw_initial_pairs_overlay=lambda pairs, marker_limit: progress_events.append(
            ("draw_initial", marker_limit)
        ),
        set_last_overlay_state=lambda state: progress_events.append(("overlay", state)),
        set_progress_text=lambda text: progress_events.append(("progress", text)),
        cmd_line=lambda text: progress_events.append(("cmd", text)),
        solver_inputs=postprocess_config.solver_inputs,
        sim_display_rotate_k=postprocess_config.sim_display_rotate_k,
        background_display_rotate_k=postprocess_config.background_display_rotate_k,
        simulate_and_compare_hkl=lambda *args, **kwargs: None,
        aggregate_match_centers=lambda *args, **kwargs: None,
        build_overlay_records=lambda *args, **kwargs: [],
        compute_frame_diagnostics=lambda *args, **kwargs: ({}, None),
    )

    assert setup.postprocess_config.log_path == (tmp_path / "geometry_fit_log_20260328_123000.txt")
    assert setup.postprocess_config.current_background_index == 1
    setup.ui_bindings.sync_joint_background_theta()
    assert theta_initial_var.get() == 4.0
    setup.ui_bindings.replace_profile_cache({"gamma": 1.25})
    assert simulation_state.profile_cache == {"gamma": 1.25}
    setup.ui_bindings.mark_last_simulation_dirty()
    assert simulation_state.last_simulation_signature is None
    save_path = tmp_path / "export.npy"
    setup.ui_bindings.save_export_records(save_path, [{"source": "sim"}])
    assert np.load(save_path, allow_pickle=True).tolist() == [{"source": "sim"}]


def test_build_runtime_geometry_fit_execution_setup_uses_log_dir_when_provided(
    tmp_path,
) -> None:
    downloads_dir = tmp_path / "downloads"
    log_dir = tmp_path / "logs"
    prepared_run, postprocess_config = _make_prepared_run(
        joint_background_mode=False,
        tmp_path=downloads_dir,
        stamp="20260328_123010",
        log_dir=log_dir,
    )

    class _SimulationState:
        profile_cache = {}
        last_simulation_signature = "sig"

    class _BackgroundState:
        current_background_index = 0

    setup = geometry_fit.build_runtime_geometry_fit_execution_setup(
        prepared_run=prepared_run,
        mosaic_params={"sigma_mosaic_deg": 0.2},
        stamp="20260328_123010",
        downloads_dir=downloads_dir,
        log_dir=log_dir,
        simulation_runtime_state=_SimulationState(),
        background_runtime_state=_BackgroundState(),
        theta_initial_var=_DummyVar(0.0),
        geometry_theta_offset_var=None,
        current_ui_params=lambda: {},
        var_map={},
        background_theta_for_index=lambda idx, strict_count=False: 3.0,
        refresh_status=lambda: None,
        update_manual_pick_button_label=lambda: None,
        capture_undo_state=lambda: {},
        push_undo_state=lambda state: None,
        replace_dataset_cache=lambda payload: None,
        request_preview_skip_once=lambda: None,
        schedule_update=lambda: None,
        draw_overlay_records=lambda records, marker_limit: None,
        draw_initial_pairs_overlay=lambda pairs, marker_limit: None,
        set_last_overlay_state=lambda state: None,
        set_progress_text=lambda text: None,
        cmd_line=lambda text: None,
        solver_inputs=postprocess_config.solver_inputs,
        sim_display_rotate_k=postprocess_config.sim_display_rotate_k,
        background_display_rotate_k=postprocess_config.background_display_rotate_k,
        simulate_and_compare_hkl=lambda *args, **kwargs: None,
        aggregate_match_centers=lambda *args, **kwargs: None,
        build_overlay_records=lambda *args, **kwargs: [],
        compute_frame_diagnostics=lambda *args, **kwargs: ({}, None),
    )

    assert setup.postprocess_config.downloads_dir == downloads_dir
    assert setup.postprocess_config.log_dir == log_dir
    assert setup.postprocess_config.log_path == (log_dir / "geometry_fit_log_20260328_123010.txt")
    assert setup.postprocess_config.log_path.parent == log_dir


def test_run_runtime_geometry_fit_action_prepares_builds_setup_and_executes(
    tmp_path,
) -> None:
    prepared_run, _ = _make_prepared_run(
        joint_background_mode=False,
        tmp_path=tmp_path,
        stamp="20260328_130000",
    )
    progress_texts: list[str] = []
    cmd_events: list[str] = []
    calls: dict[str, object] = {}
    execution_bindings = _make_runtime_action_execution_bindings(
        tmp_path,
        progress_texts,
        cmd_events,
    )

    def _prepare_bindings_factory(var_names):
        calls["prepare_factory_var_names"] = list(var_names)
        return _make_runtime_action_prepare_bindings()

    def _prepare_run(*, params, var_names, preserve_live_theta, bindings):
        calls["prepare_run"] = {
            "params": dict(params),
            "var_names": list(var_names),
            "preserve_live_theta": preserve_live_theta,
            "bindings": bindings,
        }
        return geometry_fit.GeometryFitPreparationResult(prepared_run=prepared_run)

    def _build_execution_setup(*, prepared_run, mosaic_params, stamp, bindings):
        calls["build_execution_setup"] = {
            "prepared_run": prepared_run,
            "mosaic_params": dict(mosaic_params),
            "stamp": stamp,
            "bindings": bindings,
        }
        return "setup-token"

    def _execute_run(
        *,
        prepared_run,
        var_names,
        preserve_live_theta,
        solve_fit,
        setup,
        flush_ui=None,
    ):
        calls["execute_run"] = {
            "prepared_run": prepared_run,
            "var_names": list(var_names),
            "preserve_live_theta": preserve_live_theta,
            "solve_fit": solve_fit,
            "setup": setup,
            "flush_ui": flush_ui,
        }
        return geometry_fit.GeometryFitRuntimeExecutionResult(
            log_path=tmp_path / "geometry_fit_log_20260328_130000.txt"
        )

    solve_fit = lambda *_args, **_kwargs: None
    flush_ui = lambda: None
    action = geometry_fit.run_runtime_geometry_fit_action(
        bindings=geometry_fit.GeometryFitRuntimeActionBindings(
            value_callbacks=geometry_fit.GeometryFitRuntimeValueCallbacks(
                current_var_names=lambda: ["gamma", "a"],
                current_params=lambda: {
                    "theta_initial": 3.0,
                    "gamma": 0.2,
                    "a": 4.1,
                    "mosaic_params": {"sigma_mosaic_deg": 0.2},
                },
                current_ui_params=lambda: {},
                var_map={},
            ),
            prepare_bindings_factory=_prepare_bindings_factory,
            execution_bindings=execution_bindings,
            solve_fit=solve_fit,
            stamp_factory=lambda: "20260328_130000",
            flush_ui=flush_ui,
        ),
        prepare_run=_prepare_run,
        build_execution_setup=_build_execution_setup,
        execute_run=_execute_run,
    )

    assert action.error_text is None
    assert action.var_names == ["gamma", "a"]
    assert action.preserve_live_theta is True
    assert action.prepare_result is not None
    assert action.prepare_result.prepared_run is prepared_run
    assert action.execution_result is not None
    assert calls["prepare_factory_var_names"] == ["gamma", "a"]
    assert calls["prepare_run"]["params"] == {
        "theta_initial": 3.0,
        "gamma": 0.2,
        "a": 4.1,
        "mosaic_params": {"sigma_mosaic_deg": 0.2},
    }
    assert calls["prepare_run"]["var_names"] == ["gamma", "a"]
    assert calls["prepare_run"]["preserve_live_theta"] is True
    assert calls["build_execution_setup"] == {
        "prepared_run": prepared_run,
        "mosaic_params": {"sigma_mosaic_deg": 0.2},
        "stamp": "20260328_130000",
        "bindings": execution_bindings,
    }
    assert calls["execute_run"] == {
        "prepared_run": prepared_run,
        "var_names": ["gamma", "a"],
        "preserve_live_theta": True,
        "solve_fit": solve_fit,
        "setup": "setup-token",
        "flush_ui": flush_ui,
    }
    assert progress_texts == []
    assert cmd_events == []


def test_run_runtime_geometry_fit_action_uses_fit_only_mosaic_samples_for_solver(
    tmp_path,
) -> None:
    prepared_run, _ = _make_prepared_run(
        joint_background_mode=False,
        tmp_path=tmp_path,
        stamp="20260328_130004",
    )
    prepared_run = replace(
        prepared_run,
        geometry_runtime_cfg={
            "bounds": {"gamma": [0.0, 1.0]},
            "sampling": {"fit_sample_count": 8},
        },
    )
    progress_texts: list[str] = []
    cmd_events: list[str] = []
    build_mosaic_calls: list[int] = []
    calls: dict[str, object] = {}
    solve_fit = lambda *_args, **_kwargs: None
    execution_bindings = _make_runtime_action_execution_bindings(
        tmp_path,
        progress_texts,
        cmd_events,
    )

    def _build_mosaic_params(*, sample_count=32):
        build_mosaic_calls.append(int(sample_count))
        return {
            "sigma_mosaic_deg": 0.2,
            "sample_count": int(sample_count),
        }

    def _prepare_run(*, params, var_names, preserve_live_theta, bindings):
        calls["prepare_run"] = {
            "params": dict(params),
            "var_names": list(var_names),
            "preserve_live_theta": preserve_live_theta,
            "bindings": bindings,
        }
        return geometry_fit.GeometryFitPreparationResult(prepared_run=prepared_run)

    def _build_execution_setup(*, prepared_run, mosaic_params, stamp, bindings):
        calls["build_execution_setup"] = {
            "prepared_run": prepared_run,
            "mosaic_params": dict(mosaic_params),
            "stamp": stamp,
            "bindings": bindings,
        }
        return "setup-token"

    def _execute_run(
        *,
        prepared_run,
        var_names,
        preserve_live_theta,
        solve_fit,
        setup,
        flush_ui=None,
    ):
        calls["execute_run"] = {
            "prepared_run": prepared_run,
            "var_names": list(var_names),
            "preserve_live_theta": preserve_live_theta,
            "solve_fit": solve_fit,
            "setup": setup,
            "flush_ui": flush_ui,
            "solver_mosaic_params": dict(prepared_run.fit_params["mosaic_params"]),
        }
        return geometry_fit.GeometryFitRuntimeExecutionResult(
            log_path=tmp_path / "geometry_fit_log_20260328_130004.txt"
        )

    action = geometry_fit.run_runtime_geometry_fit_action(
        bindings=geometry_fit.GeometryFitRuntimeActionBindings(
            value_callbacks=geometry_fit.GeometryFitRuntimeValueCallbacks(
                current_var_names=lambda: ["gamma", "a"],
                current_params=lambda: {
                    "theta_initial": 3.0,
                    "gamma": 0.2,
                    "a": 4.1,
                    "mosaic_params": {
                        "sigma_mosaic_deg": 0.2,
                        "sample_count": 32,
                    },
                },
                current_ui_params=lambda: {},
                var_map={},
                build_mosaic_params=_build_mosaic_params,
            ),
            prepare_bindings_factory=lambda _var_names: _make_runtime_action_prepare_bindings(),
            execution_bindings=execution_bindings,
            solve_fit=solve_fit,
            stamp_factory=lambda: "20260328_130004",
        ),
        prepare_run=_prepare_run,
        build_execution_setup=_build_execution_setup,
        execute_run=_execute_run,
    )

    assert action.error_text is None
    assert build_mosaic_calls == [8]
    assert calls["build_execution_setup"] == {
        "prepared_run": prepared_run,
        "mosaic_params": {
            "sigma_mosaic_deg": 0.2,
            "sample_count": 32,
        },
        "stamp": "20260328_130004",
        "bindings": execution_bindings,
    }
    assert calls["execute_run"] == {
        "prepared_run": prepared_run,
        "var_names": ["gamma", "a"],
        "preserve_live_theta": True,
        "solve_fit": solve_fit,
        "setup": "setup-token",
        "flush_ui": None,
        "solver_mosaic_params": {
            "sigma_mosaic_deg": 0.2,
            "sample_count": 8,
        },
    }
    assert prepared_run.fit_params["mosaic_params"] == {
        "sigma_mosaic_deg": 0.2,
        "sample_count": 8,
    }
    assert progress_texts == []
    assert cmd_events == ["Geometry fit: solver sample count=8"]


def test_run_runtime_geometry_fit_action_reports_prepare_exception(tmp_path) -> None:
    progress_texts: list[str] = []
    cmd_events: list[str] = []

    action = geometry_fit.run_runtime_geometry_fit_action(
        bindings=geometry_fit.GeometryFitRuntimeActionBindings(
            value_callbacks=geometry_fit.GeometryFitRuntimeValueCallbacks(
                current_var_names=lambda: ["gamma"],
                current_params=lambda: {"theta_initial": 3.0, "mosaic_params": {}},
                current_ui_params=lambda: {},
                var_map={},
            ),
            prepare_bindings_factory=lambda _var_names: _make_runtime_action_prepare_bindings(),
            execution_bindings=_make_runtime_action_execution_bindings(
                tmp_path,
                progress_texts,
                cmd_events,
            ),
            solve_fit=lambda *_args, **_kwargs: None,
            stamp_factory=lambda: "20260328_130001",
        ),
        prepare_run=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        build_execution_setup=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("build_execution_setup should not be called")
        ),
        execute_run=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("execute_run should not be called")
        ),
    )

    assert action.execution_result is None
    assert action.prepare_result is not None
    assert action.error_text == "Geometry fit failed: boom"
    assert action.prepare_result.log_path == (tmp_path / "geometry_fit_log_20260328_130001.txt")
    log_text = action.prepare_result.log_path.read_text(encoding="utf-8")
    assert "Geometry fit aborted before solver start: 20260328_130001" in log_text
    assert progress_texts == ["Geometry fit failed: boom"]
    assert cmd_events == ["failed: boom"]


def test_run_runtime_geometry_fit_action_writes_preflight_log_to_log_dir(
    tmp_path,
) -> None:
    downloads_dir = tmp_path / "downloads"
    log_dir = tmp_path / "logs"
    progress_texts: list[str] = []
    cmd_events: list[str] = []

    action = geometry_fit.run_runtime_geometry_fit_action(
        bindings=geometry_fit.GeometryFitRuntimeActionBindings(
            value_callbacks=geometry_fit.GeometryFitRuntimeValueCallbacks(
                current_var_names=lambda: ["gamma"],
                current_params=lambda: {"theta_initial": 3.0, "mosaic_params": {}},
                current_ui_params=lambda: {},
                var_map={},
            ),
            prepare_bindings_factory=lambda _var_names: _make_runtime_action_prepare_bindings(),
            execution_bindings=_make_runtime_action_execution_bindings(
                downloads_dir,
                progress_texts,
                cmd_events,
                log_dir=log_dir,
            ),
            solve_fit=lambda *_args, **_kwargs: None,
            stamp_factory=lambda: "20260328_130001",
        ),
        prepare_run=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        build_execution_setup=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("build_execution_setup should not be called")
        ),
        execute_run=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("execute_run should not be called")
        ),
    )

    expected_log_path = log_dir / "geometry_fit_log_20260328_130001.txt"
    assert action.prepare_result is not None
    assert action.prepare_result.log_path == expected_log_path
    assert expected_log_path.exists()
    assert not (downloads_dir / "geometry_fit_log_20260328_130001.txt").exists()


def test_run_runtime_geometry_fit_action_surfaces_preflight_error_text(tmp_path) -> None:
    progress_texts: list[str] = []
    cmd_events: list[str] = []

    action = geometry_fit.run_runtime_geometry_fit_action(
        bindings=geometry_fit.GeometryFitRuntimeActionBindings(
            value_callbacks=geometry_fit.GeometryFitRuntimeValueCallbacks(
                current_var_names=lambda: ["theta_initial", "gamma"],
                current_params=lambda: {"theta_initial": 3.0, "mosaic_params": {}},
                current_ui_params=lambda: {},
                var_map={},
            ),
            prepare_bindings_factory=lambda _var_names: _make_runtime_action_prepare_bindings(),
            execution_bindings=_make_runtime_action_execution_bindings(
                tmp_path,
                progress_texts,
                cmd_events,
            ),
            solve_fit=lambda *_args, **_kwargs: None,
            stamp_factory=lambda: "20260328_130002",
        ),
        prepare_run=lambda **kwargs: geometry_fit.GeometryFitPreparationResult(
            error_text="Geometry fit unavailable"
        ),
        build_execution_setup=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("build_execution_setup should not be called")
        ),
        execute_run=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("execute_run should not be called")
        ),
    )

    assert action.execution_result is None
    assert action.prepare_result is not None
    assert action.error_text == "Geometry fit unavailable"
    assert action.preserve_live_theta is False
    assert action.prepare_result.log_path == (tmp_path / "geometry_fit_log_20260328_130002.txt")
    log_text = action.prepare_result.log_path.read_text(encoding="utf-8")
    assert "Geometry fit aborted before solver start: 20260328_130002" in log_text
    assert "Geometry fit unavailable" in log_text
    assert "stage=preflight" in log_text
    assert progress_texts == ["Geometry fit unavailable"]
    assert cmd_events == []


def test_run_runtime_geometry_fit_action_persists_debug_preflight_log_without_error_text(
    tmp_path,
) -> None:
    progress_texts: list[str] = []
    cmd_events: list[str] = []

    def _prepare_bindings_factory(_var_names):
        return replace(
            _make_runtime_action_prepare_bindings(),
            fit_config={"geometry": {"debug_logging": True}},
        )

    action = geometry_fit.run_runtime_geometry_fit_action(
        bindings=geometry_fit.GeometryFitRuntimeActionBindings(
            value_callbacks=geometry_fit.GeometryFitRuntimeValueCallbacks(
                current_var_names=lambda: ["gamma"],
                current_params=lambda: {"theta_initial": 3.0, "mosaic_params": {}},
                current_ui_params=lambda: {},
                var_map={},
            ),
            prepare_bindings_factory=_prepare_bindings_factory,
            execution_bindings=_make_runtime_action_execution_bindings(
                tmp_path,
                progress_texts,
                cmd_events,
            ),
            solve_fit=lambda *_args, **_kwargs: None,
            stamp_factory=lambda: "20260328_130003",
        ),
        prepare_run=lambda **kwargs: geometry_fit.GeometryFitPreparationResult(),
        build_execution_setup=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("build_execution_setup should not be called")
        ),
        execute_run=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("execute_run should not be called")
        ),
    )

    assert action.execution_result is None
    assert action.prepare_result is not None
    assert action.error_text is None
    assert action.prepare_result.log_path == (tmp_path / "geometry_fit_log_20260328_130003.txt")
    log_text = action.prepare_result.log_path.read_text(encoding="utf-8")
    assert "Geometry fit aborted before solver start: 20260328_130003" in log_text
    assert "Geometry fit aborted before solver start." in log_text
    assert "stage=preflight" in log_text
    assert "Run request:" in log_text
    assert progress_texts == []
    assert cmd_events == []


def test_execute_runtime_geometry_fit_solver_phase_stamps_fit_run_id_before_solve(
    tmp_path,
) -> None:
    prepared_run, postprocess_config = _make_prepared_run(
        joint_background_mode=False,
        tmp_path=tmp_path,
        stamp="20260328_130004",
    )
    pair_entry = {
        "pair_id": "bg0:pair0",
        "overlay_match_index": 0,
        "hkl": (1, 1, 0),
        "source_table_index": 1,
        "source_row_index": 2,
        "source_branch_index": 0,
        "source_peak_index": 0,
        "x": 30.0,
        "y": 40.0,
    }
    prepared_run.current_dataset["measured_for_fit"] = [dict(pair_entry)]
    prepared_run.current_dataset["initial_pairs_display"] = [dict(pair_entry)]
    prepared_run.current_dataset["source_rows_for_trace"] = [dict(pair_entry)]
    prepared_run.dataset_infos[0].update(
        {
            "dataset_index": 0,
            "label": "bg0.osc",
            "measured_for_fit": [dict(pair_entry)],
            "initial_pairs_display": [dict(pair_entry)],
            "source_rows_for_trace": [dict(pair_entry)],
        }
    )

    seen: dict[str, object] = {}

    def _solve_fit(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks,
        var_names,
        **kwargs,
    ):
        seen["measured_peaks"] = [dict(entry) for entry in measured_peaks]
        return SimpleNamespace(success=True)

    execution_result = geometry_fit.execute_runtime_geometry_fit_solver_phase(
        prepared_run=prepared_run,
        var_names=["gamma"],
        solve_fit=_solve_fit,
        solver_inputs=postprocess_config.solver_inputs,
        stamp="20260328_130004",
        log_path=tmp_path / "geometry_fit_log_20260328_130004.txt",
    )

    assert execution_result.error_text is None
    assert seen["measured_peaks"][0]["fit_run_id"] == "20260328_130004"
    assert prepared_run.current_dataset["initial_pairs_display"][0]["fit_run_id"] == (
        "20260328_130004"
    )
    assert prepared_run.dataset_infos[0]["source_rows_for_trace"][0]["fit_run_id"] == (
        "20260328_130004"
    )


def test_finalize_runtime_geometry_fit_execution_writes_trace_for_rejected_run(
    tmp_path,
    monkeypatch,
) -> None:
    prepared_run, postprocess_config = _make_prepared_run(
        joint_background_mode=False,
        tmp_path=tmp_path,
        stamp="20260328_130005",
    )
    pair_entry = {
        "pair_id": "bg0:pair0",
        "overlay_match_index": 0,
        "hkl": (1, 1, 0),
        "q_group_key": ("q", 1),
        "source_table_index": 1,
        "source_reflection_index": 7,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_row_index": 2,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "x": 30.0,
        "y": 40.0,
    }
    measured_entry = {
        **pair_entry,
        "fit_source_identity_only": True,
        "measured_x": 30.0,
        "measured_y": 40.0,
    }
    source_row = {
        "hkl": (1, 1, 0),
        "q_group_key": ("q", 1),
        "source_table_index": 1,
        "source_reflection_index": 7,
        "source_reflection_namespace": "full_reflection",
        "source_reflection_is_full": True,
        "source_row_index": 2,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "sim_col": 10.0,
        "sim_row": 20.0,
    }
    prepared_run.current_dataset.update(
        {
            "label": "bg0.osc",
            "initial_pairs_display": [dict(pair_entry)],
            "measured_for_fit": [dict(measured_entry)],
            "source_rows_for_trace": [dict(source_row)],
            "pair_count": 1,
            "group_count": 1,
        }
    )
    prepared_run.dataset_infos[0].update(
        {
            "dataset_index": 0,
            "label": "bg0.osc",
            "initial_pairs_display": [dict(pair_entry)],
            "measured_for_fit": [dict(measured_entry)],
            "source_rows_for_trace": [dict(source_row)],
            "pair_count": 1,
            "group_count": 1,
            "simulation_diagnostics": {
                "created_from": "live_runtime_cache",
                "requested_signature_summary": "sig-summary",
                "stored_signature_summary": "sig-summary",
                "cache_metadata": {
                    "live_runtime_cache_dual_path_diff": [
                        {
                            "pair_id": "bg0:pair0",
                            "before_status": "failed",
                            "after_status": "resolved",
                        }
                    ]
                },
                "live_runtime_cache_validation": {
                    "pair_failures": [
                        {
                            "pair_id": "bg0:pair0",
                            "reason": "missing_canonical_candidate",
                            "branch_candidates": [],
                        }
                    ]
                },
            },
        }
    )

    result = SimpleNamespace(
        weighted_residual_rms_px=1.5,
        rms_px=1392.6754,
        point_match_summary={
            "matched_pair_count": 1,
            "fixed_source_reflection_count": 1,
            "subset_fallback_hkl_count": 0,
        },
        point_match_diagnostics=[
            {
                "dataset_index": 0,
                "pair_id": "bg0:pair0",
                "fit_run_id": "20260328_130005",
                "overlay_match_index": 0,
                "match_status": "matched",
                "hkl": (1, 1, 0),
                "source_reflection_index": 7,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "source_table_index": 1,
                "source_row_index": 2,
                "source_branch_index": 1,
                "source_peak_index": 1,
                "resolved_table_index": 1,
                "resolved_peak_index": 1,
                "resolution_kind": "fixed_source",
                "measured_x": 30.0,
                "measured_y": 40.0,
                "simulated_x": 10.0,
                "simulated_y": 20.0,
                "distance_px": 22.360679775,
                "weighted_dx_px": -1.0,
                "weighted_dy_px": 2.0,
            }
        ],
        full_beam_polish_summary={
            "fit_quality_passed": False,
            "selection_status": "retained_start_safe_fallback",
            "selected_candidate_name": "retained_start_safe_fallback",
            "selected_candidate_source": "requested_x0",
            "best_valid_raw_detector_candidate_name": "requested_start",
            "best_valid_raw_detector_candidate_source": "requested_x0",
            "constraint_count": 2,
            "active_fit_variable_count": 1,
            "active_fit_variables": ["gamma"],
            "candidate_ledger": [
                {
                    "candidate_name": "requested_start",
                    "x_vector_source": "requested_x0",
                    "matched_pair_count": 1,
                    "missing_pair_count": 0,
                    "branch_mismatch_count": 0,
                    "rms_px": 22.360679775,
                    "median_px": 22.360679775,
                    "max_px": 22.360679775,
                    "outside_radius_count": 0,
                    "weighted_objective": 5.0,
                    "accepted_or_rejected": "accepted",
                    "rejection_reason": None,
                    "valid_raw_detector_candidate": True,
                    "selected": False,
                },
                {
                    "candidate_name": "retained_start_safe_fallback",
                    "x_vector_source": "requested_x0",
                    "matched_pair_count": 1,
                    "missing_pair_count": 0,
                    "branch_mismatch_count": 0,
                    "rms_px": 22.360679775,
                    "median_px": 22.360679775,
                    "max_px": 22.360679775,
                    "outside_radius_count": 0,
                    "weighted_objective": 5.0,
                    "accepted_or_rejected": "accepted",
                    "rejection_reason": None,
                    "valid_raw_detector_candidate": True,
                    "selected": True,
                },
            ],
            "seed_correspondence_records": [
                {
                    "dataset_index": 0,
                    "pair_id": "bg0:pair0",
                    "fit_run_id": "20260328_130005",
                    "overlay_match_index": 0,
                    "hkl": (1, 1, 0),
                    "source_reflection_index": 7,
                    "source_reflection_namespace": "full_reflection",
                    "source_reflection_is_full": True,
                    "source_table_index": 1,
                    "source_row_index": 2,
                    "source_branch_index": 1,
                    "source_peak_index": 1,
                    "resolved_table_index": 1,
                    "resolved_peak_index": 1,
                    "match_status": "matched",
                    "simulated_x": 10.0,
                    "simulated_y": 20.0,
                }
            ],
            "point_match_diagnostics": [],
        },
        final_metric_name="full_beam_fixed_correspondence",
    )

    monkeypatch.setattr(
        geometry_fit,
        "build_runtime_geometry_fit_execution_result_bindings",
        lambda **kwargs: "bindings",
    )
    monkeypatch.setattr(
        geometry_fit,
        "apply_runtime_geometry_fit_result",
        lambda **kwargs: geometry_fit.GeometryFitRuntimeApplyResult(
            accepted=False,
            rejection_reason="detector-space regression",
            rms=1392.6754,
            fitted_params=None,
            postprocess=None,
        ),
    )

    setup = geometry_fit.GeometryFitRuntimeExecutionSetup(
        ui_bindings=geometry_fit.GeometryFitRuntimeUiBindings(
            fit_params=prepared_run.fit_params,
            base_profile_cache={},
            mosaic_params={},
            current_ui_params=lambda: {},
            var_map={},
            geometry_theta_offset_var=None,
            capture_undo_state=lambda: {},
            sync_joint_background_theta=None,
            refresh_status=lambda: None,
            update_manual_pick_button_label=lambda: None,
            replace_profile_cache=lambda cache: None,
            replace_dataset_cache=lambda payload: None,
            push_undo_state=lambda state: None,
            request_preview_skip_once=lambda: None,
            mark_last_simulation_dirty=lambda: None,
            schedule_update=lambda: None,
            draw_overlay_records=lambda records, marker_limit: None,
            draw_initial_pairs_overlay=lambda pairs, marker_limit: None,
            set_last_overlay_state=lambda state: None,
            save_export_records=lambda path, records: None,
            set_progress_text=lambda text: None,
            cmd_line=lambda text: None,
        ),
        postprocess_config=postprocess_config,
    )

    execution_result = geometry_fit.finalize_runtime_geometry_fit_execution(
        execution_result=geometry_fit.GeometryFitRuntimeExecutionResult(
            log_path=Path(postprocess_config.log_path),
            solver_result=result,
        ),
        prepared_run=prepared_run,
        var_names=["gamma"],
        preserve_live_theta=True,
        setup=setup,
    )

    expected_trace_path = tmp_path / "geometry_fit_trace_20260328_130005.jsonl"
    assert execution_result.trace_path == expected_trace_path
    assert expected_trace_path.exists()
    records = [
        json.loads(line)
        for line in expected_trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        record["record_type"] == "run"
        and record["fit_run_id"] == "20260328_130005"
        and record["accepted"] is False
        and record["fit_quality_passed"] is False
        and record["selection_status"] == "retained_start_safe_fallback"
        and record["selected_candidate_name"] == "retained_start_safe_fallback"
        and record["selected_candidate_source"] == "requested_x0"
        and record["best_valid_raw_detector_candidate_name"] == "requested_start"
        and record["best_valid_raw_detector_candidate_source"] == "requested_x0"
        and record["constraint_count"] == 2
        and record["active_fit_variable_count"] == 1
        and record["active_fit_variables"] == ["gamma"]
        and len(record["candidate_ledger"]) == 2
        for record in records
    )
    assert any(
        record["phase"] == "saved_pairs"
        and record["pair_id"] == "bg0:pair0"
        and record["source_reflection_namespace"] == "full_reflection"
        for record in records
    )
    assert any(
        record["phase"] == "seed_correspondence"
        and record["pair_id"] == "bg0:pair0"
        and record["canonical_identity"] == [7, 1]
        for record in records
    )
    assert any(
        record["record_type"] == "dual_path_diff"
        and record["pair_id"] == "bg0:pair0"
        and record["before_status"] == "failed"
        and record["after_status"] == "resolved"
        for record in records
    )
    assert any(
        record["phase"] == "acceptance_residuals"
        and record["pair_id"] == "bg0:pair0"
        and record["optimizer_residual_px"] == pytest.approx(np.hypot(1.0, 2.0))
        and record["detector_residual_px"] == pytest.approx(22.360679775)
        for record in records
    )


def test_postprocess_geometry_fit_result_builds_overlay_export_and_status_payloads(
    tmp_path,
) -> None:
    class _Result:
        point_match_summary = {
            "claimed": 4,
            "qualified": 3,
            "measured_count": 2,
            "fixed_source_resolved_count": 1,
        }
        point_match_diagnostics = [
            {
                "dataset_index": 0,
                "overlay_match_index": 0,
                "match_status": "matched",
                "hkl": (9, 9, 9),
                "measured_x": 101.0,
                "measured_y": 201.0,
                "simulated_x": 100.0,
                "simulated_y": 200.0,
                "dx_px": -1.0,
                "dy_px": -1.0,
                "distance_px": np.hypot(-1.0, -1.0),
                "source_table_index": 0,
                "source_row_index": 1,
                "resolution_kind": "fixed_source_row",
                "resolution_reason": "seed anchor",
                "name": "drop",
            },
            {
                "dataset_index": 1,
                "overlay_match_index": 3,
                "match_status": "matched",
                "hkl": (1, 1, 0),
                "measured_x": 11.5,
                "measured_y": 19.0,
                "simulated_x": 10.0,
                "simulated_y": 20.0,
                "dx_px": -1.5,
                "dy_px": 1.0,
                "distance_px": np.hypot(-1.5, 1.0),
                "source_table_index": 2,
                "source_row_index": 5,
                "resolution_kind": "fixed_source_row",
                "resolution_reason": "matched source row",
                "name": "keep",
            },
        ]

    calls: dict[str, object] = {}

    def _simulate_and_compare_hkl(
        miller,
        intensities,
        image_size,
        fitted_params,
        measured_for_fit,
        *,
        pixel_tol,
    ):
        calls["simulate"] = {
            "miller": miller,
            "intensities": intensities,
            "image_size": image_size,
            "fitted_params": dict(fitted_params),
            "measured_for_fit": measured_for_fit,
            "pixel_tol": pixel_tol,
        }
        return (
            None,
            [(10.0, 20.0)],
            [(11.5, 19.0)],
            [(1, 1, 0)],
            [(1, 1, 0)],
        )

    def _aggregate_match_centers(sim_coords, meas_coords, sim_millers, meas_millers):
        calls["aggregate"] = True
        return sim_coords, meas_coords, sim_millers

    def _build_overlay_records(
        initial_pairs_display,
        overlay_point_match_diagnostics,
        *,
        native_shape,
        orientation_choice,
        sim_display_rotate_k,
        background_display_rotate_k,
    ):
        calls["overlay"] = {
            "initial_pairs_display": list(initial_pairs_display),
            "overlay_point_match_diagnostics": overlay_point_match_diagnostics,
            "native_shape": native_shape,
            "orientation_choice": dict(orientation_choice),
            "sim_display_rotate_k": sim_display_rotate_k,
            "background_display_rotate_k": background_display_rotate_k,
        }
        return [{"overlay": True}]

    def _compute_frame_diagnostics(records):
        calls["frame_diag"] = list(records)
        return (
            {
                "paired_records": 1,
                "sim_display_med_px": 1.2,
                "bg_display_med_px": 2.3,
                "sim_display_p90_px": 3.4,
                "bg_display_p90_px": 4.5,
            },
            "frame warning",
        )

    postprocess = geometry_fit.postprocess_geometry_fit_result(
        fitted_params={"theta_initial": 3.0, "theta_offset": 0.4, "a": 4.2},
        result=_Result(),
        current_dataset={
            "measured_for_fit": [{"x": 1.0, "y": 2.0}],
            "initial_pairs_display": [{"pair": 1}],
            "native_background": np.zeros((4, 5)),
            "orientation_choice": {"label": "rotate"},
            "pair_count": 3,
            "group_count": 2,
        },
        joint_background_mode=True,
        current_background_index=1,
        dataset_count=2,
        var_names=["gamma", "a"],
        values=[1.5, 2.5],
        rms=0.75,
        miller="miller-data",
        intensities="intensity-data",
        image_size=512,
        max_display_markers=120,
        downloads_dir=tmp_path,
        stamp="20260328_120000",
        log_path=tmp_path / "geometry_fit_log.txt",
        sim_display_rotate_k=1,
        background_display_rotate_k=3,
        simulate_and_compare_hkl=_simulate_and_compare_hkl,
        aggregate_match_centers=_aggregate_match_centers,
        build_overlay_records=_build_overlay_records,
        compute_frame_diagnostics=_compute_frame_diagnostics,
    )

    assert "simulate" not in calls
    assert "aggregate" not in calls
    assert calls["overlay"]["overlay_point_match_diagnostics"] == [
        {
            "dataset_index": 1,
            "overlay_match_index": 3,
            "match_status": "matched",
            "hkl": (1, 1, 0),
            "measured_x": 11.5,
            "measured_y": 19.0,
            "simulated_x": 10.0,
            "simulated_y": 20.0,
            "dx_px": -1.5,
            "dy_px": 1.0,
            "distance_px": np.hypot(-1.5, 1.0),
            "source_table_index": 2,
            "source_row_index": 5,
            "resolution_kind": "fixed_source_row",
            "resolution_reason": "matched source row",
            "name": "keep",
        }
    ]
    assert postprocess.point_match_summary_lines == [
        "claimed=4",
        "fixed_source_resolved_count=1",
        "measured_count=2",
        "qualified=3",
    ]
    assert postprocess.pixel_offsets == [
        {
            "dataset_index": 0,
            "overlay_match_index": 0,
            "match_status": "matched",
            "hkl": (9, 9, 9),
            "dx_px": -1.0,
            "dy_px": -1.0,
            "distance_px": np.hypot(-1.0, -1.0),
        },
        {
            "dataset_index": 1,
            "overlay_match_index": 3,
            "match_status": "matched",
            "hkl": (1, 1, 0),
            "dx_px": -1.5,
            "dy_px": 1.0,
            "distance_px": np.hypot(-1.5, 1.0),
        },
    ]
    assert postprocess.overlay_records == [{"overlay": True}]
    assert postprocess.overlay_state == {
        "overlay_records": [{"overlay": True}],
        "initial_pairs_display": [{"pair": 1}],
        "max_display_markers": 120,
    }
    assert postprocess.overlay_diagnostic_lines == [
        "transform_rule=sim:native_to_overlay_display; bg:inverse_orientation_then_overlay_display",
        "overlay_records=1",
        "paired_records=1",
        "sim_display_med_px=1.200",
        "bg_display_med_px=2.300",
        "sim_display_p90_px=3.400",
        "bg_display_p90_px=4.500",
    ]
    assert postprocess.export_records == [
        {
            "dataset_index": 0,
            "overlay_match_index": 0,
            "match_status": "matched",
            "hkl": (9, 9, 9),
            "measured_x": 101.0,
            "measured_y": 201.0,
            "simulated_x": 100.0,
            "simulated_y": 200.0,
            "dx_px": -1.0,
            "dy_px": -1.0,
            "distance_px": np.hypot(-1.0, -1.0),
            "source_table_index": 0,
            "source_row_index": 1,
            "source_peak_index": None,
            "resolution_kind": "fixed_source_row",
            "resolution_reason": "seed anchor",
        },
        {
            "dataset_index": 1,
            "overlay_match_index": 3,
            "match_status": "matched",
            "hkl": (1, 1, 0),
            "measured_x": 11.5,
            "measured_y": 19.0,
            "simulated_x": 10.0,
            "simulated_y": 20.0,
            "dx_px": -1.5,
            "dy_px": 1.0,
            "distance_px": np.hypot(-1.5, 1.0),
            "source_table_index": 2,
            "source_row_index": 5,
            "source_peak_index": None,
            "resolution_kind": "fixed_source_row",
            "resolution_reason": "matched source row",
        },
    ]
    assert postprocess.save_path == tmp_path / "matched_peaks_20260328_120000.npy"
    assert postprocess.fit_summary_lines == [
        "manual_groups=2",
        "manual_points=3",
        "overlay_records=1",
        "orientation=rotate",
        "fixed_source_resolved=1",
        "gamma = 1.500000",
        "a = 2.500000",
        "RMS residual = 0.750000 px",
        f"Matched peaks saved to: {postprocess.save_path}",
    ]
    assert "dataset=1 idx=3 HKL=(1, 1, 0): dx=-1.5000, dy=1.0000, |Δ|=1.8028 px" in (
        postprocess.progress_text
    )
    assert "frame warning" in postprocess.progress_text
    assert "joint backgrounds=2" in postprocess.progress_text
    assert f"Saved 2 peak records → {postprocess.save_path}" in postprocess.progress_text


def test_postprocess_geometry_fit_result_composes_solver_discrete_mode_for_overlay(
    tmp_path,
) -> None:
    native_shape = (9, 7)
    base_orientation = {
        "indexing_mode": "yx",
        "k": 1,
        "flip_x": True,
        "flip_y": False,
        "flip_order": "xy",
        "label": "base",
    }
    solver_mode = {
        "indexing_mode": "xy",
        "k": 3,
        "flip_x": False,
        "flip_y": True,
        "flip_order": "yx",
    }
    expected_orientation = geometry_overlay.compose_orientation_transforms(
        native_shape,
        base_orientation,
        solver_mode,
    )
    calls: dict[str, object] = {}

    class _Result:
        point_match_summary = {"claimed": 1, "qualified": 1}
        point_match_diagnostics = []
        chosen_discrete_mode = solver_mode

    def _build_overlay_records(
        initial_pairs_display,
        overlay_point_match_diagnostics,
        *,
        native_shape,
        orientation_choice,
        sim_display_rotate_k,
        background_display_rotate_k,
    ):
        calls["orientation_choice"] = dict(orientation_choice)
        return []

    postprocess = geometry_fit.postprocess_geometry_fit_result(
        fitted_params={"theta_initial": 3.0},
        result=_Result(),
        current_dataset={
            "measured_for_fit": [{"x": 1.0, "y": 2.0}],
            "initial_pairs_display": [{"pair": 1}],
            "native_background": np.zeros(native_shape),
            "orientation_choice": dict(base_orientation),
            "pair_count": 1,
            "group_count": 1,
        },
        joint_background_mode=False,
        current_background_index=0,
        dataset_count=1,
        var_names=["gamma"],
        values=[1.5],
        rms=0.75,
        miller="miller-data",
        intensities="intensity-data",
        image_size=512,
        max_display_markers=120,
        downloads_dir=tmp_path,
        stamp="20260328_120000",
        log_path=tmp_path / "geometry_fit_log.txt",
        sim_display_rotate_k=1,
        background_display_rotate_k=3,
        simulate_and_compare_hkl=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("simulate_and_compare_hkl should not be called")
        ),
        aggregate_match_centers=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("aggregate_match_centers should not be called")
        ),
        build_overlay_records=_build_overlay_records,
        compute_frame_diagnostics=lambda records: ({}, None),
    )

    assert calls["orientation_choice"] == expected_orientation
    assert "solver_discrete_mode=rot90=3+flip_y" in postprocess.fit_summary_lines


def test_apply_runtime_geometry_fit_result_orchestrates_runtime_side_effects(
    tmp_path,
) -> None:
    events: list[object] = []

    postprocess = geometry_fit.GeometryFitPostprocessResult(
        fitted_params={"theta_initial": 3.0},
        point_match_summary_lines=["claimed=4"],
        pixel_offsets=[((1, 1, 0), 1.5, 1.0, np.hypot(1.5, 1.0))],
        overlay_records=[{"overlay": True}],
        overlay_state={"overlay_records": [{"overlay": True}]},
        overlay_diagnostic_lines=["overlay=ok"],
        frame_warning="warn",
        export_records=[{"source": "sim"}],
        save_path=tmp_path / "matched.npy",
        fit_summary_lines=["summary=ok"],
        progress_text="fit complete",
    )

    class _Result:
        x = np.array([1.5, 2.5], dtype=float)
        rms_px = 0.75
        success = True
        status = 2
        message = "ok"
        nfev = 5
        cost = 1.0
        robust_cost = 1.0
        solver_loss = "soft_l1"
        solver_f_scale = 1.0
        optimality = 0.1
        active_mask = [0]
        restart_history = []

    applied_values: list[tuple[list[object], list[object]]] = []
    stored_profile_cache: dict[str, object] = {}
    stored_overlay_state: dict[str, object] = {}
    stored_progress_text: list[str] = []
    saved_exports: list[tuple[object, object]] = []

    outcome = geometry_fit.apply_runtime_geometry_fit_result(
        result=_Result(),
        var_names=["gamma", "a"],
        current_dataset={
            "initial_pairs_display": [{"pair": 1}],
            "group_count": 2,
            "pair_count": 3,
        },
        dataset_count=4,
        joint_background_mode=True,
        preserve_live_theta=False,
        max_display_markers=120,
        bindings=geometry_fit.GeometryFitRuntimeResultBindings(
            log_section=lambda title, lines: events.append((title, list(lines))),
            capture_undo_state=lambda: {"undo": True},
            apply_result_values=lambda names, values: applied_values.append(
                (list(names), list(values))
            ),
            sync_joint_background_theta=lambda: events.append("sync_theta"),
            refresh_status=lambda: events.append("refresh_status"),
            update_manual_pick_button_label=lambda: events.append("update_button"),
            build_profile_cache=lambda: {"profile": 1},
            replace_profile_cache=lambda cache: stored_profile_cache.update(cache),
            push_undo_state=lambda state: events.append(("push_undo", dict(state))),
            request_preview_skip_once=lambda: events.append("skip_once"),
            mark_last_simulation_dirty=lambda: events.append("mark_dirty"),
            schedule_update=lambda: events.append("schedule_update"),
            build_fitted_params=lambda: {"theta_initial": 3.0, "gamma": 1.5},
            postprocess_result=lambda fitted_params, rms: (
                events.append(("postprocess", dict(fitted_params), rms)) or postprocess
            ),
            draw_overlay_records=lambda records, marker_limit: events.append(
                ("draw_overlay", list(records), marker_limit)
            ),
            draw_initial_pairs_overlay=lambda pairs, marker_limit: events.append(
                ("draw_initial", list(pairs), marker_limit)
            ),
            set_last_overlay_state=lambda state: stored_overlay_state.update(state),
            save_export_records=lambda save_path, export_records: saved_exports.append(
                (save_path, list(export_records))
            ),
            set_progress_text=lambda text: stored_progress_text.append(text),
            cmd_line=lambda text: events.append(("cmd", text)),
        ),
    )

    assert applied_values == [(["gamma", "a"], [1.5, 2.5])]
    assert stored_profile_cache == {"profile": 1}
    assert stored_overlay_state == {"overlay_records": [{"overlay": True}]}
    assert stored_progress_text == ["fit complete"]
    assert saved_exports == [(tmp_path / "matched.npy", [{"source": "sim"}])]
    assert outcome.accepted is True
    assert outcome.rejection_reason is None
    assert outcome.rms == 0.75
    assert outcome.fitted_params == {"theta_initial": 3.0, "gamma": 1.5}
    assert outcome.postprocess is postprocess
    assert events == [
        (
            "Optimizer diagnostics:",
            [
                "success=True",
                "status=2",
                "message=ok",
                "nfev=5",
                "cost=1.000000",
                "robust_cost=1.000000",
                "solver_loss=soft_l1",
                "solver_f_scale=1.000000",
                "optimality=0.100000",
                "active_mask=[0]",
                "display_rms_px=0.750000",
            ],
        ),
        "sync_theta",
        "refresh_status",
        "update_button",
        ("push_undo", {"undo": True}),
        "skip_once",
        "mark_dirty",
        "schedule_update",
        (
            "Optimization result:",
            [
                "gamma = 1.500000",
                "a = 2.500000",
                "RMS residual = 0.750000 px",
            ],
        ),
        ("postprocess", {"theta_initial": 3.0, "gamma": 1.5}, 0.75),
        ("Point-match summary:", ["claimed=4"]),
        ("Overlay frame diagnostics:", ["overlay=ok"]),
        ("draw_overlay", [{"overlay": True}], 120),
        (
            "Pixel offsets (native frame):",
            ["HKL=(1, 1, 0): dx=1.5000, dy=1.0000, |Δ|=1.8028 px"],
        ),
        ("Fit summary:", ["summary=ok"]),
        ("cmd", "done: datasets=4 groups=2 points=3 rms=0.7500px"),
    ]


def test_apply_runtime_geometry_fit_result_logs_point_match_diagnostics_when_debug_enabled(
    tmp_path,
) -> None:
    events: list[object] = []

    postprocess = geometry_fit.GeometryFitPostprocessResult(
        fitted_params={
            "theta_initial": 3.0,
            "gamma": 1.5,
            "pixel_size_m": 1.0e-4,
            "debye_x": 0.0,
            "debye_y": 0.0,
        },
        point_match_summary_lines=["claimed=4"],
        pixel_offsets=[],
        overlay_records=[],
        overlay_state={},
        overlay_diagnostic_lines=["overlay=ok"],
        frame_warning=None,
        export_records=[],
        save_path=tmp_path / "matched.npy",
        fit_summary_lines=["summary=ok"],
        progress_text="fit complete",
    )

    class _Result:
        x = np.array([1.5, 2.5], dtype=float)
        rms_px = 0.75
        success = True
        status = 2
        message = "ok"
        nfev = 5
        cost = 1.0
        robust_cost = 1.0
        solver_loss = "soft_l1"
        solver_f_scale = 1.0
        optimality = 0.1
        active_mask = [0]
        restart_history = []
        point_match_diagnostics = [
            {
                "dataset_index": 0,
                "dataset_label": "bg0.osc",
                "overlay_match_index": 7,
                "match_status": "missing_pair",
                "hkl": (1, 1, 0),
                "q_group_key": ("q", 1),
                "measured_x": 50.0,
                "measured_y": 60.0,
                "simulated_x": np.nan,
                "simulated_y": np.nan,
                "distance_px": np.nan,
                "measured_two_theta_deg": 11.25,
                "measured_phi_deg": 6.5,
                "delta_two_theta_deg": np.nan,
                "delta_phi_deg": np.nan,
                "resolution_reason": "match_radius_exceeded",
                "measured_resolution_reason": "measured_anchor_missing",
                "detector_resolution_reason": "detector_source_missing",
                "correspondence_resolution_reason": "source_row_out_of_range",
            }
        ]

    outcome = geometry_fit.apply_runtime_geometry_fit_result(
        result=_Result(),
        var_names=["gamma", "a"],
        current_dataset={
            "initial_pairs_display": [{"pair": 1}],
            "group_count": 2,
            "pair_count": 3,
        },
        dataset_count=1,
        joint_background_mode=False,
        preserve_live_theta=False,
        max_display_markers=120,
        bindings=geometry_fit.GeometryFitRuntimeResultBindings(
            log_section=lambda title, lines: events.append((title, list(lines))),
            capture_undo_state=lambda: {"undo": True},
            apply_result_values=lambda _names, _values: None,
            sync_joint_background_theta=None,
            refresh_status=lambda: None,
            update_manual_pick_button_label=lambda: None,
            build_profile_cache=lambda: {},
            replace_profile_cache=lambda _cache: None,
            push_undo_state=lambda _state: None,
            request_preview_skip_once=lambda: None,
            mark_last_simulation_dirty=lambda: None,
            schedule_update=lambda: None,
            build_fitted_params=lambda: {
                "theta_initial": 3.0,
                "gamma": 1.5,
                "pixel_size_m": 1.0e-4,
                "debye_x": 0.0,
                "debye_y": 0.0,
            },
            postprocess_result=lambda _fitted_params, _rms: postprocess,
            draw_overlay_records=lambda _records, _marker_limit: None,
            draw_initial_pairs_overlay=lambda _pairs, _marker_limit: None,
            set_last_overlay_state=lambda _state: None,
            save_export_records=lambda _save_path, _export_records: None,
            set_progress_text=lambda _text: None,
            cmd_line=lambda text: events.append(("cmd", text)),
            geometry_runtime_cfg={"debug_logging": True},
        ),
    )

    assert outcome.accepted is True
    assert ("Point-match summary:", ["claimed=4"]) in events
    diagnostic_sections = [lines for title, lines in events if title == "Point-match diagnostics:"]
    assert len(diagnostic_sections) == 1
    assert diagnostic_sections[0][0] == "match_status: missing_pair=1"
    assert "dataset[0] bg0.osc: unresolved_pairs=1" in diagnostic_sections[0]
    assert any(
        "overlay_index=7" in line and "measured_detector=[50.000, 60.000]" in line
        for line in diagnostic_sections[0]
    )
    assert any(
        "resolution_reason=source_row_out_of_range" in line for line in diagnostic_sections[0]
    )


def test_apply_runtime_geometry_fit_result_rejects_absurd_manual_fit_before_update() -> None:
    events: list[object] = []
    applied_values: list[tuple[list[object], list[object]]] = []
    progress_messages: list[str] = []

    class _Result:
        x = np.array([0.00034, 0.000502, -0.018376, 1.985904], dtype=float)
        rms_px = 1195.373582
        success = True
        status = 2
        message = "ftol satisfied"
        nfev = 13
        cost = 469781.260275
        robust_cost = 469781.260275
        solver_loss = "linear"
        solver_f_scale = 8.0
        optimality = 0.007503
        active_mask = [0, 0, 0, 0]
        restart_history = []
        point_match_summary = {
            "matched_pair_count": 3,
            "unweighted_peak_max_px": 2070.4477776558383,
        }

    outcome = geometry_fit.apply_runtime_geometry_fit_result(
        result=_Result(),
        var_names=["zb", "zs", "psi_z", "cor_angle"],
        current_dataset={
            "initial_pairs_display": [{"pair": 1}],
            "group_count": 2,
            "pair_count": 3,
        },
        dataset_count=1,
        joint_background_mode=False,
        preserve_live_theta=False,
        max_display_markers=120,
        bindings=geometry_fit.GeometryFitRuntimeResultBindings(
            log_section=lambda title, lines: events.append((title, list(lines))),
            capture_undo_state=lambda: {"undo": True},
            apply_result_values=lambda names, values: applied_values.append(
                (list(names), list(values))
            ),
            sync_joint_background_theta=None,
            refresh_status=lambda: events.append("refresh_status"),
            update_manual_pick_button_label=lambda: events.append("update_button"),
            build_profile_cache=lambda: {"profile": 1},
            replace_profile_cache=lambda cache: events.append(("replace_profile", cache)),
            push_undo_state=lambda state: events.append(("push_undo", dict(state))),
            request_preview_skip_once=lambda: events.append("skip_once"),
            mark_last_simulation_dirty=lambda: events.append("mark_dirty"),
            schedule_update=lambda: events.append("schedule_update"),
            build_fitted_params=lambda: {"theta_initial": 3.0},
            postprocess_result=lambda fitted_params, rms: events.append(
                ("postprocess", dict(fitted_params), rms)
            ),
            draw_overlay_records=lambda records, marker_limit: events.append(
                ("draw_overlay", list(records), marker_limit)
            ),
            draw_initial_pairs_overlay=lambda pairs, marker_limit: events.append(
                ("draw_initial", list(pairs), marker_limit)
            ),
            set_last_overlay_state=lambda state: events.append(("set_overlay_state", state)),
            save_export_records=lambda save_path, export_records: events.append(
                ("save_export", save_path, list(export_records))
            ),
            set_progress_text=lambda text: progress_messages.append(text),
            cmd_line=lambda text: events.append(("cmd", text)),
        ),
    )

    assert applied_values == []
    assert outcome.accepted is False
    assert outcome.rms == 1195.373582
    assert outcome.fitted_params is None
    assert outcome.postprocess is None
    assert "RMS residual 1195.37 px exceeds the acceptance limit" in str(outcome.rejection_reason)
    assert progress_messages == [
        "Manual geometry fit rejected:\n"
        "RMS residual 1195.37 px exceeds the acceptance limit of 100.00 px.\n"
        "Largest matched-peak offset 2070.45 px exceeds the acceptance limit of 150.00 px.\n"
        "RMS residual = 1195.37 px\n"
        "Manual pairs: 3 points across 2 groups\n"
        "Add more manual points or remove outliers before rerunning the fit."
    ]
    assert events == [
        (
            "Optimizer diagnostics:",
            [
                "success=True",
                "status=2",
                "message=ftol satisfied",
                "nfev=13",
                "cost=469781.260275",
                "robust_cost=469781.260275",
                "solver_loss=linear",
                "solver_f_scale=8.000000",
                "optimality=0.007503",
                "active_mask=[0, 0, 0, 0]",
                "display_rms_px=1195.373582",
            ],
        ),
        (
            "Optimization result:",
            [
                "zb = 0.000340",
                "zs = 0.000502",
                "psi_z = -0.018376",
                "cor_angle = 1.985904",
                "RMS residual = 1195.373582 px",
            ],
        ),
        (
            "Point-match summary:",
            [
                "matched_pair_count=3",
                "unweighted_peak_max_px=2070.4477776558383",
            ],
        ),
        (
            "Fit rejected:",
            [
                "RMS residual 1195.37 px exceeds the acceptance limit of 100.00 px.",
                "Largest matched-peak offset 2070.45 px exceeds the acceptance limit of 150.00 px.",
            ],
        ),
        (
            "cmd",
            "rejected: datasets=1 groups=2 points=3 rms=1195.3736px reason=RMS residual 1195.37 px exceeds the acceptance limit of 100.00 px.",
        ),
    ]


def test_execute_runtime_geometry_fit_runs_solver_logs_and_applies_result(
    tmp_path,
    monkeypatch,
) -> None:
    keep_diag = {
        "dataset_index": 0,
        "overlay_match_index": 0,
        "match_status": "matched",
        "hkl": (1, 1, 0),
        "measured_x": 11.5,
        "measured_y": 19.0,
        "simulated_x": 10.0,
        "simulated_y": 20.0,
        "dx_px": -1.5,
        "dy_px": 1.0,
        "distance_px": np.hypot(-1.5, 1.0),
        "source_table_index": None,
        "source_row_index": None,
        "source_peak_index": None,
        "resolution_kind": "",
        "resolution_reason": None,
    }

    prepared_run, postprocess_config = _make_prepared_run(
        joint_background_mode=False,
        tmp_path=tmp_path,
    )
    prepared_run = replace(
        prepared_run,
        geometry_runtime_cfg={
            "bounds": {"gamma": [0.0, 1.0]},
            "completion_chime": {
                "enabled": True,
                "mode": "alias",
                "alias": "SystemNotification",
            },
        },
    )
    gamma_var = _DummyVar(0.2)
    a_var = _DummyVar(4.1)
    theta_initial_var = _DummyVar(3.0)
    theta_offset_var = _DummyVar("0.0")
    events: list[object] = []
    progress_texts: list[str] = []
    saved_exports: list[tuple[object, object]] = []
    stored_profile_cache: dict[str, object] = {}
    stored_overlay_state: dict[str, object] = {}
    solver_calls: list[dict[str, object]] = []
    completion_chime_calls: list[dict[str, object] | None] = []

    monkeypatch.setattr(
        geometry_fit,
        "play_completion_chime",
        lambda config: completion_chime_calls.append(
            dict(config) if isinstance(config, dict) else None
        ),
    )

    class _Result:
        x = np.array([1.5, 2.5], dtype=float)
        rms_px = 0.75
        success = True
        status = 2
        message = "ok"
        nfev = 5
        cost = 1.0
        robust_cost = 1.0
        solver_loss = "soft_l1"
        solver_f_scale = 1.0
        optimality = 0.1
        active_mask = [0]
        restart_history = []
        point_match_summary = {"claimed": 4, "qualified": 3}
        point_match_diagnostics = [keep_diag]
        geometry_fit_debug_summary = {
            "point_match_mode": True,
            "dataset_count": 1,
            "var_names": ["gamma", "a"],
            "parameter_entries": [],
            "dataset_entries": [],
            "solver": {"loss": "soft_l1"},
        }
        reparameterization_summary = {"status": "skipped", "accepted": False}
        staged_release_summary = {"status": "skipped", "accepted": False}
        adaptive_regularization_summary = {"status": "skipped", "accepted": False}
        ridge_refinement_summary = {"status": "skipped", "accepted": False}
        image_refinement_summary = {"status": "skipped", "accepted": False}
        auto_freeze_summary = {"status": "skipped", "accepted": False}
        selective_thaw_summary = {"status": "skipped", "accepted": False}

    def _solve_fit(
        miller,
        intensities,
        image_size,
        params,
        measured_peaks,
        var_names,
        *,
        pixel_tol,
        experimental_image,
        dataset_specs,
        refinement_config,
        status_callback,
    ):
        status_callback("Geometry fit: running main solve")
        status_callback("Geometry fit: complete (cost=1.000000, rms=0.7500px)")
        solver_calls.append(
            {
                "miller": miller,
                "intensities": intensities,
                "image_size": image_size,
                "params": dict(params),
                "measured_peaks": list(measured_peaks),
                "var_names": list(var_names),
                "pixel_tol": pixel_tol,
                "experimental_image": experimental_image,
                "dataset_specs": dataset_specs,
                "refinement_config": dict(refinement_config),
            }
        )
        return _Result()

    def _simulate_and_compare_hkl(
        miller,
        intensities,
        image_size,
        fitted_params,
        measured_for_fit,
        *,
        pixel_tol,
    ):
        events.append(
            (
                "simulate",
                miller,
                intensities,
                image_size,
                dict(fitted_params),
                list(measured_for_fit),
                pixel_tol,
            )
        )
        return (
            None,
            [(10.0, 20.0)],
            [(11.5, 19.0)],
            [(1, 1, 0)],
            [(1, 1, 0)],
        )

    postprocess_config = geometry_fit.GeometryFitRuntimePostprocessConfig(
        current_background_index=postprocess_config.current_background_index,
        downloads_dir=postprocess_config.downloads_dir,
        stamp=postprocess_config.stamp,
        log_path=postprocess_config.log_path,
        solver_inputs=postprocess_config.solver_inputs,
        sim_display_rotate_k=postprocess_config.sim_display_rotate_k,
        background_display_rotate_k=postprocess_config.background_display_rotate_k,
        simulate_and_compare_hkl=_simulate_and_compare_hkl,
        aggregate_match_centers=(
            lambda sim_coords, meas_coords, sim_millers, meas_millers: (
                sim_coords,
                meas_coords,
                sim_millers,
            )
        ),
        build_overlay_records=(
            lambda initial_pairs_display, overlay_point_match_diagnostics, **_kwargs: [
                {
                    "initial_pairs_display": list(initial_pairs_display),
                    "diagnostics": list(overlay_point_match_diagnostics),
                }
            ]
        ),
        compute_frame_diagnostics=(
            lambda _records: (
                {
                    "paired_records": 1,
                    "sim_display_med_px": 1.2,
                    "bg_display_med_px": 2.3,
                    "sim_display_p90_px": 3.4,
                    "bg_display_p90_px": 4.5,
                },
                "frame warning",
            )
        ),
    )

    execution = geometry_fit.execute_runtime_geometry_fit(
        prepared_run=prepared_run,
        var_names=["gamma", "a"],
        preserve_live_theta=True,
        solve_fit=_solve_fit,
        setup=geometry_fit.GeometryFitRuntimeExecutionSetup(
            ui_bindings=geometry_fit.GeometryFitRuntimeUiBindings(
                fit_params=prepared_run.fit_params,
                base_profile_cache={"existing": 1},
                mosaic_params={"sigma_mosaic_deg": 0.2},
                current_ui_params=lambda: {
                    "zb": 0.1,
                    "zs": 0.2,
                    "theta_initial": theta_initial_var.get(),
                    "theta_offset": float(theta_offset_var.get() or 0.0),
                    "chi": 0.5,
                    "cor_angle": 0.6,
                    "psi_z": 0.7,
                    "gamma": gamma_var.get(),
                    "Gamma": 0.9,
                    "corto_detector": 1.0,
                    "a": a_var.get(),
                    "c": 32.1,
                    "center_x": 100.0,
                    "center_y": 200.0,
                },
                var_map={"gamma": gamma_var, "a": a_var},
                geometry_theta_offset_var=theta_offset_var,
                capture_undo_state=lambda: {"undo": True},
                sync_joint_background_theta=None,
                refresh_status=lambda: events.append("refresh_status"),
                update_manual_pick_button_label=lambda: events.append("update_button"),
                replace_profile_cache=lambda cache: stored_profile_cache.update(cache),
                replace_dataset_cache=lambda payload: events.append(("cache", payload)),
                push_undo_state=lambda state: events.append(("push_undo", dict(state))),
                request_preview_skip_once=lambda: events.append("skip_once"),
                mark_last_simulation_dirty=lambda: events.append("mark_dirty"),
                schedule_update=lambda: events.append("schedule_update"),
                draw_overlay_records=lambda records, marker_limit: events.append(
                    ("draw_overlay", list(records), marker_limit)
                ),
                draw_initial_pairs_overlay=lambda pairs, marker_limit: events.append(
                    ("draw_initial", list(pairs), marker_limit)
                ),
                set_last_overlay_state=lambda state: stored_overlay_state.update(state),
                save_export_records=lambda save_path, export_records: saved_exports.append(
                    (save_path, list(export_records))
                ),
                set_progress_text=lambda text: progress_texts.append(text),
                cmd_line=lambda text: events.append(("cmd", text)),
            ),
            postprocess_config=postprocess_config,
        ),
        flush_ui=lambda: events.append("flush_ui"),
    )

    assert execution.error_text is None
    assert execution.solver_request is not None
    assert execution.solver_result is not None
    assert execution.apply_result is not None
    assert solver_calls == [
        {
            "miller": "miller-data",
            "intensities": "intensity-data",
            "image_size": 512,
            "params": {"theta_initial": 3.0, "theta_offset": 0.0},
            "measured_peaks": [
                {
                    "x": 1.0,
                    "y": 2.0,
                    "fit_run_id": "20260328_120000",
                }
            ],
            "var_names": ["gamma", "a"],
            "pixel_tol": float("inf"),
            "experimental_image": None,
            "dataset_specs": [
                {
                    "dataset_index": 0,
                    "theta_initial": 3.0,
                    "fit_run_id": "20260328_120000",
                    "measured_peaks": [
                        {
                            "x": 1.0,
                            "y": 2.0,
                            "fit_run_id": "20260328_120000",
                        }
                    ],
                }
            ],
            "refinement_config": {
                "bounds": {"gamma": [0.0, 1.0]},
                "completion_chime": {
                    "enabled": True,
                    "mode": "alias",
                    "alias": "SystemNotification",
                },
                "use_numba": False,
                "solver": {
                    "parallel_mode": "auto",
                    "workers": "auto",
                    "worker_numba_threads": 0,
                },
            },
        }
    ]
    assert progress_texts[0] == "Running geometry fit from saved manual Qr/Qz pairs…"
    assert "Geometry fit: running main solve" in progress_texts
    assert "Geometry fit: complete (cost=1.000000, rms=0.7500px)" in progress_texts
    assert "Manual geometry fit complete:" in progress_texts[-1]
    assert stored_profile_cache["existing"] == 1
    assert stored_profile_cache["gamma"] == 1.5
    assert stored_profile_cache["a"] == 2.5
    assert stored_overlay_state["max_display_markers"] == 120
    assert saved_exports == [
        (
            tmp_path / "matched_peaks_20260328_120000.npy",
            [
                {
                    "dataset_index": 0,
                    "overlay_match_index": 0,
                    "match_status": "matched",
                    "hkl": (1, 1, 0),
                    "measured_x": 11.5,
                    "measured_y": 19.0,
                    "simulated_x": 10.0,
                    "simulated_y": 20.0,
                    "dx_px": -1.5,
                    "dy_px": 1.0,
                    "distance_px": np.hypot(-1.5, 1.0),
                    "source_table_index": None,
                    "source_row_index": None,
                    "source_peak_index": None,
                    "resolution_kind": "",
                    "resolution_reason": None,
                },
            ],
        )
    ]
    cache_events = [event for event in events if isinstance(event, tuple) and event[0] == "cache"]
    assert cache_events == [
        (
            "cache",
            {
                "selected_background_indices": [0],
                "current_background_index": 0,
                "joint_background_mode": False,
                "background_theta_values": [3.0],
                "dataset_specs": [
                    {
                        "dataset_index": 0,
                        "theta_initial": 3.0,
                        "fit_run_id": "20260328_120000",
                    }
                ],
                "cache_metadata": {
                    "cache_action": "rebuilt",
                    "reused": False,
                    "rebuilt": True,
                    "stale_reason": None,
                    "cache_source": "build_geometry_fit_dataset_cache_payload",
                    "cache_provenance": [
                        "build_geometry_manual_fit_dataset",
                        "build_geometry_fit_dataset_cache_payload",
                    ],
                    "dataset_count": 1,
                    "dataset_cache_metadata": [],
                },
            },
        )
    ]
    assert ("cmd", "start: vars=gamma,a datasets=1 current_groups=2 current_points=3") in events
    assert ("cmd", "Geometry fit: running main solve") in events
    assert ("cmd", "Geometry fit: complete (cost=1.000000, rms=0.7500px)") in events
    assert ("cmd", "done: datasets=1 groups=2 points=3 rms=0.7500px") in events
    assert completion_chime_calls == [
        {
            "enabled": True,
            "mode": "alias",
            "alias": "SystemNotification",
        }
    ]
    draw_overlay_event = next(
        event for event in events if isinstance(event, tuple) and event[0] == "draw_overlay"
    )
    assert draw_overlay_event[2] == 120
    assert draw_overlay_event[1][0]["initial_pairs_display"][0]["pair"] == 1
    overlay_diag = draw_overlay_event[1][0]["diagnostics"][0]
    for key, value in keep_diag.items():
        assert overlay_diag[key] == value
    assert "flush_ui" in events


def test_execute_runtime_geometry_fit_reports_solver_failure_and_closes_log(
    tmp_path,
) -> None:
    prepared_run, postprocess_config = _make_prepared_run(
        joint_background_mode=False,
        tmp_path=tmp_path,
        stamp="20260328_120001",
    )
    progress_texts: list[str] = []
    events: list[object] = []

    execution = geometry_fit.execute_runtime_geometry_fit(
        prepared_run=prepared_run,
        var_names=["gamma", "a"],
        preserve_live_theta=True,
        solve_fit=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        setup=geometry_fit.GeometryFitRuntimeExecutionSetup(
            ui_bindings=geometry_fit.GeometryFitRuntimeUiBindings(
                fit_params=prepared_run.fit_params,
                base_profile_cache={},
                mosaic_params={},
                current_ui_params=lambda: {},
                var_map={},
                geometry_theta_offset_var=None,
                capture_undo_state=lambda: {},
                sync_joint_background_theta=None,
                refresh_status=lambda: None,
                update_manual_pick_button_label=lambda: None,
                replace_profile_cache=lambda _cache: None,
                replace_dataset_cache=lambda _payload: None,
                push_undo_state=lambda _state: None,
                request_preview_skip_once=lambda: None,
                mark_last_simulation_dirty=lambda: None,
                schedule_update=lambda: None,
                draw_overlay_records=lambda *_args, **_kwargs: None,
                draw_initial_pairs_overlay=lambda *_args, **_kwargs: None,
                set_last_overlay_state=lambda _state: None,
                save_export_records=lambda *_args, **_kwargs: None,
                set_progress_text=lambda text: progress_texts.append(text),
                cmd_line=lambda text: events.append(text),
            ),
            postprocess_config=postprocess_config,
        ),
    )

    assert execution.error_text == "Geometry fit failed: boom"
    assert progress_texts == [
        "Running geometry fit from saved manual Qr/Qz pairs…",
        "Geometry fit failed: boom",
    ]
    assert events == [
        "start: vars=gamma,a datasets=1 current_groups=2 current_points=3",
        "failed: boom",
    ]


def test_validate_geometry_fit_live_source_rows_drops_trusted_deadband_branch() -> None:
    validation = geometry_fit.validate_geometry_fit_live_source_rows(
        [
            {
                "hkl": (1, 0, 3),
                "q_group_key": ("q_group", "primary", 1, 3),
                "source_table_index": 0,
                "source_reflection_index": 7,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "source_row_index": 0,
                "source_branch_index": 0,
                "source_peak_index": 0,
                "simulated_phi_deg": -22.0,
            },
            {
                "hkl": (1, 0, 3),
                "q_group_key": ("q_group", "primary", 1, 3),
                "source_table_index": 0,
                "source_reflection_index": 7,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "source_row_index": 1,
                "source_branch_index": 1,
                "source_peak_index": 1,
                "simulated_phi_deg": 22.0,
            },
        ],
        required_pairs=[
            {
                "pair_id": "bg0:pair0",
                "overlay_match_index": 0,
                "hkl": (1, 0, 3),
                "q_group_key": ("q_group", "primary", 1, 3),
                "source_reflection_index": 7,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "source_row_index": 0,
                "source_peak_index": 0,
                "simulated_phi_deg": 5.0e-4,
            }
        ],
    )

    assert validation["valid"] is False
    assert validation["branch_mismatch_count"] == 0
    assert validation["hkl_missing_candidate_count"] == 0
    assert validation["pair_failures"] == [
        {
            "pair_id": "bg0:pair0",
            "overlay_match_index": 0,
            "reason": "ambiguous_branch_deadband",
            "hkl": (1, 0, 3),
            "source_table_index": None,
            "source_reflection_index": 7,
            "source_row_index": 0,
            "source_peak_index": 0,
            "source_branch_index": None,
            "trusted_identity_required": True,
            "candidate_count_total": 2,
            "candidate_count_after_hkl_filter": 2,
            "candidate_count_after_branch_filter": 0,
            "branch_candidates": [],
        }
    ]


def test_validate_geometry_fit_live_source_rows_excludes_saved_input_failures_from_rejection_counts() -> (
    None
):
    validation = geometry_fit.validate_geometry_fit_live_source_rows(
        [
            {
                "hkl": (1, 0, 0),
                "q_group_key": ("q", 1),
                "source_reflection_index": 7,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "source_row_index": 0,
                "source_branch_index": 1,
                "source_peak_index": 0,
            },
            {
                "hkl": (2, 0, 0),
                "q_group_key": ("q", 2),
                "source_reflection_index": 8,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "source_row_index": 1,
                "source_branch_index": 1,
                "source_peak_index": 1,
            },
        ],
        required_pairs=[
            {
                "pair_id": "missing-branch",
                "overlay_match_index": 0,
                "hkl": (1, 0, 0),
                "q_group_key": ("q", 1),
                "source_reflection_index": 7,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
            },
            {
                "pair_id": "missing-trusted-row",
                "overlay_match_index": 1,
                "hkl": (2, 0, 0),
                "q_group_key": ("q", 2),
                "source_reflection_index": 8,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "source_row_index": 99,
                "source_branch_index": 1,
            },
        ],
    )

    assert validation["valid"] is False
    assert validation["missing_required_pair_count"] == 2
    assert validation["branch_mismatch_count"] == 0
    assert validation["hkl_missing_candidate_count"] == 0
    pair_failures = {str(entry["pair_id"]): dict(entry) for entry in validation["pair_failures"]}
    assert pair_failures["missing-branch"]["reason"] == "missing_branch"
    assert pair_failures["missing-branch"]["candidate_count_total"] == 1
    assert pair_failures["missing-branch"]["candidate_count_after_hkl_filter"] == 1
    assert pair_failures["missing-branch"]["candidate_count_after_branch_filter"] == 0
    assert pair_failures["missing-trusted-row"]["reason"] == "missing_trusted_reflection_row"
    assert pair_failures["missing-trusted-row"]["candidate_count_total"] == 1
    assert pair_failures["missing-trusted-row"]["candidate_count_after_hkl_filter"] == 1
    assert pair_failures["missing-trusted-row"]["candidate_count_after_branch_filter"] == 1


def test_build_geometry_fit_caked_roi_selection_only_keeps_selected_branch() -> None:
    source_rows = [
        {
            "hkl": (1, 0, 0),
            "q_group_key": ("selected", 1),
            "source_table_index": 0,
            "source_reflection_index": 7,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "background_two_theta_deg": 1.0,
            "background_phi_deg": -10.0,
            "detector_x": 80.0,
            "detector_y": 80.0,
        },
        {
            "hkl": (1, 0, 0),
            "q_group_key": ("selected", 1),
            "source_table_index": 0,
            "source_reflection_index": 7,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 1,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "background_two_theta_deg": 2.0,
            "background_phi_deg": -10.0,
            "detector_x": 84.0,
            "detector_y": 80.0,
        },
        {
            "hkl": (1, 0, 0),
            "q_group_key": ("selected", 1),
            "source_table_index": 0,
            "source_reflection_index": 7,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 2,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "background_two_theta_deg": 1.0,
            "background_phi_deg": 10.0,
            "detector_x": 40.0,
            "detector_y": 40.0,
        },
        {
            "hkl": (2, 0, 0),
            "q_group_key": ("other", 2),
            "source_table_index": 1,
            "source_reflection_index": 9,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "background_two_theta_deg": 3.0,
            "background_phi_deg": -20.0,
            "detector_x": 60.0,
            "detector_y": 60.0,
        },
    ]

    projection_map = {
        (1.0, -10.0): (20.0, 20.0),
        (2.0, -10.0): (24.0, 20.0),
        (1.0, 10.0): (40.0, 40.0),
        (3.0, -20.0): (60.0, 60.0),
    }

    def _project(two_theta_deg: float, phi_deg: float):
        return projection_map.get((float(two_theta_deg), float(phi_deg)))

    selection = geometry_fit.build_geometry_fit_caked_roi_selection(
        source_rows,
        required_pairs=[
            {
                "pair_id": "bg0:pair0",
                "overlay_match_index": 0,
                "hkl": (1, 0, 0),
                "q_group_key": ("selected", 1),
                "source_reflection_index": 7,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "source_row_index": 0,
                "source_branch_index": 0,
                "source_peak_index": 0,
            }
        ],
        image_shape=(128, 128),
        fit_config={"caked_roi": {"enabled": True, "half_width_px": 0.0}},
        fit_space_to_detector_point=_project,
    )

    assert selection["valid"] is True
    assert selection["resolved_pair_count"] == 1
    assert selection["selected_branch_count"] == 1
    pixels = set(zip(selection["rows"].tolist(), selection["cols"].tolist()))
    assert (20, 20) in pixels
    assert (20, 22) in pixels
    assert (80, 80) not in pixels
    assert (60, 60) not in pixels


def test_build_geometry_fit_caked_roi_selection_does_not_reuse_stale_detector_coords_when_projection_fails() -> (
    None
):
    source_rows = [
        {
            "hkl": (1, 0, 0),
            "q_group_key": ("selected", 1),
            "source_table_index": 0,
            "source_reflection_index": 7,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "detector_x": 80.0,
            "detector_y": 80.0,
            "background_two_theta_deg": 1.0,
            "background_phi_deg": -10.0,
        },
        {
            "hkl": (1, 0, 0),
            "q_group_key": ("selected", 1),
            "source_table_index": 0,
            "source_reflection_index": 7,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 1,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "detector_x": 84.0,
            "detector_y": 80.0,
            "background_two_theta_deg": 2.0,
            "background_phi_deg": -10.0,
        },
        {
            "hkl": (1, 0, 0),
            "q_group_key": ("selected", 1),
            "source_table_index": 0,
            "source_reflection_index": 7,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 2,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "detector_x": 40.0,
            "detector_y": 40.0,
            "background_two_theta_deg": 1.0,
            "background_phi_deg": 10.0,
        },
    ]

    def _project(two_theta_deg: float, phi_deg: float):
        del two_theta_deg, phi_deg
        return None

    selection = geometry_fit.build_geometry_fit_caked_roi_selection(
        source_rows,
        required_pairs=[
            {
                "pair_id": "bg0:pair0",
                "overlay_match_index": 0,
                "hkl": (1, 0, 0),
                "q_group_key": ("selected", 1),
                "source_reflection_index": 7,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "source_row_index": 0,
                "source_branch_index": 0,
                "source_peak_index": 0,
            }
        ],
        image_shape=(128, 128),
        fit_config={"caked_roi": {"enabled": True, "half_width_px": 0.0}},
        fit_space_to_detector_point=_project,
    )

    assert selection["valid"] is False
    assert selection["fallback_reason"] == "no_native_detector_points"
    assert selection["pixel_count"] == 0
    assert selection["selected_branch_count"] == 0
    assert selection["rows"].size == 0
    assert selection["cols"].size == 0


def test_geometry_fit_caked_roi_angle_point_uses_canonical_angles_only() -> None:
    assert geometry_fit._geometry_fit_caked_roi_angle_point(
        {
            "background_two_theta_deg": 23.0,
            "background_phi_deg": -36.0,
            "caked_x": 150.0,
            "caked_y": 160.0,
        }
    ) == pytest.approx((23.0, -36.0))
    assert (
        geometry_fit._geometry_fit_caked_roi_angle_point(
            {
                "caked_x": 150.0,
                "caked_y": 160.0,
            }
        )
        is None
    )


def _load_new4_caked_reprojection_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "debug"
        / "run_new4_caked_point_reprojection_check.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_new4_caked_point_reprojection_check",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_caked_point_reprojection_context(
    *,
    projector_kind="exact_caked_bundle",
    reported_input_frame=None,
    reported_source="fit_space_projector_native_detector",
    include_projector=True,
    include_bindings=False,
):
    projector_calls = []

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
            "input_frame": str(
                input_frame if reported_input_frame is None else reported_input_frame
            ),
            "fit_space_projector_kind": str(projector_kind),
            "cake_bundle_signature": f"sig-theta-{theta:.8f}-dist-{distance:.8f}",
            "fit_space_local_params_signature": (f"lp-theta-{theta:.8f}-dist-{distance:.8f}"),
            "valid": True,
            "invalid_reason": None,
            "native_frame_conversion_source": "identity_native_detector",
            "native_frame_conversion_count": 0,
            "native_cols": cols_arr,
            "native_rows": rows_arr,
            "caked_projection_source": str(reported_source),
        }

    entries = []
    provider_pairs = []
    for index in range(7):
        point = [100.0 + index, 200.0 + 2.0 * index]
        hkl = [index, 0, index + 1]
        q_group_key = ["q_group", "primary", index, index + 1]
        entries.append(
            {
                "detector_x": point[0],
                "detector_y": point[1],
                "hkl": hkl,
                "q_group_key": q_group_key,
                "source_branch_index": 0,
                "caked_x": -999999.0,
                "caked_y": 999999.0,
                "raw_caked_x": -888888.0,
                "raw_caked_y": 888888.0,
                "background_two_theta_deg": -777777.0,
                "background_phi_deg": 777777.0,
                "refined_sim_caked_x": -666666.0,
                "refined_sim_caked_y": 666666.0,
            }
        )
        provider_pairs.append(
            {
                "pair_index": index,
                "normalized_hkl": hkl,
                "q_group_key": q_group_key,
                "source_branch_index": 0,
            }
        )
    context = {
        "params": {
            "theta_initial": 2.0,
            "corto_detector": 100.0,
            "center": [10.0, 10.0],
            "pixel_size": 0.1,
        },
        "saved_entries": entries,
        "dataset": {
            "spec": {
                "fit_space_projector_kind": str(projector_kind),
            }
        },
        "geometry_runtime_cfg": {
            "bounds": {
                "theta_initial": [-10.0, 10.0],
                "corto_detector": [10.0, 200.0],
            }
        },
    }
    if include_projector:
        context["dataset"]["spec"]["fit_space_projector"] = projector
    if include_bindings:
        context["bindings"] = object()
    provider_report = {
        "ok": True,
        "manual_point_pair_count": 7,
        "pairs": provider_pairs,
    }
    return context, provider_report, projector_calls


def _run_stub_caked_point_reprojection(tmp_path, **context_kwargs):
    module = _load_new4_caked_reprojection_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text(json.dumps({"state": {"sentinel": True}}), encoding="utf-8")
    provider_after_factory = context_kwargs.pop("provider_after_factory", None)
    context, provider_report, projector_calls = _make_caked_point_reprojection_context(
        **context_kwargs
    )
    report = module.run_caked_point_reprojection_probe_from_context(
        state_path=state_path,
        background_index=0,
        context=context,
        provider_before_report=provider_report,
        provider_after_factory=(
            provider_after_factory
            if provider_after_factory is not None
            else lambda: dict(provider_report)
        ),
        run_dir=tmp_path / "new4_run",
    )
    return report, projector_calls, module


def test_caked_point_reprojection_recomputes_when_theta_changes(tmp_path) -> None:
    report, projector_calls, _module = _run_stub_caked_point_reprojection(tmp_path)

    assert report["status"] == "pass"
    assert report["point_only_reprojection_called"] is True
    assert len(projector_calls) == 3
    assert all(call["input_frame"] == "native_detector" for call in projector_calls)
    assert report["theta_projector_signature_changed"] is True
    assert report["any_theta_point_shifted"] is True
    assert report["theta_two_theta_changed"] is True
    assert report["all_theta_reprojected_points_finite"] is True
    assert report["exact_projector_available"] is True
    assert report["all_exact_projector_used"] is True
    assert report["all_projection_projector_kinds_exact"] is True
    assert report["all_projection_signatures_present"] is True
    assert report["all_projection_input_frames_native"] is True
    assert report["all_projection_sources_native"] is True
    assert report["stale_alias_guard_installed"] is True
    assert report["stale_caked_field_read_count"] == 0
    assert report["full_background_recake_called"] is False
    assert report["full_background_recake_call_count"] == 0
    assert all(pair["stale_caked_fields_used"] is False for pair in report["pairs"])


def test_caked_point_reprojection_recomputes_when_corto_detector_changes(tmp_path) -> None:
    report, _projector_calls, _module = _run_stub_caked_point_reprojection(tmp_path)

    assert report["status"] == "pass"
    assert report["distance_projector_signature_changed"] is True
    assert report["any_distance_point_shifted"] is True
    assert report["all_distance_reprojected_points_finite"] is True
    assert report["full_background_recake_called"] is False
    assert report["full_background_recake_call_count"] == 0


def test_caked_point_reprojection_uses_detector_pixel_path(tmp_path) -> None:
    report, projector_calls, _module = _run_stub_caked_point_reprojection(tmp_path)

    first_pair = report["pairs"][0]
    expected_base = [
        100.0 * 0.01 + 2.0 * 0.5 + 100.0 * 0.001,
        200.0 * 0.01 + 2.0 * 0.05 + 100.0 * 0.0002,
    ]
    assert first_pair["base_caked_point"] == pytest.approx(expected_base)
    assert first_pair["base_caked_point"][0] != pytest.approx(-777777.0)
    assert first_pair["stale_caked_fields_present"] is True
    assert first_pair["stale_caked_fields_used"] is False
    assert first_pair["projection_input_frame"] == "native_detector"
    assert first_pair["caked_projection_source"] == "fit_space_projector_native_detector"
    assert first_pair["exact_projector_used"] is True
    assert projector_calls[0]["cols"] == pytest.approx([100.0 + i for i in range(7)])
    assert projector_calls[0]["rows"] == pytest.approx([200.0 + 2.0 * i for i in range(7)])


def test_caked_point_reprojection_does_not_recake_background_image(tmp_path) -> None:
    report, _projector_calls, _module = _run_stub_caked_point_reprojection(tmp_path)

    assert report["status"] == "pass"
    assert report["full_recake_guard_installed"] is True
    assert report["full_background_recake_called"] is False
    assert report["full_background_recake_call_count"] == 0
    assert report["point_only_reprojection_call_count"] == 3


def test_caked_point_reprojection_fails_without_exact_projector(tmp_path) -> None:
    report, _projector_calls, _module = _run_stub_caked_point_reprojection(
        tmp_path,
        projector_kind="approximate_projector",
    )

    assert report["status"] == "fail"
    assert report["exact_projector_available"] is False
    assert report["all_exact_projector_used"] is False
    assert report["all_projection_projector_kinds_exact"] is False
    assert "exact_projector_available" in report["failures"]


def test_caked_point_reprojection_fails_on_wrong_projector_input_frame(tmp_path) -> None:
    report, _projector_calls, _module = _run_stub_caked_point_reprojection(
        tmp_path,
        reported_input_frame="fit_detector",
    )

    assert report["status"] == "fail"
    assert report["all_projection_input_frames_native"] is False
    assert "all_projection_input_frames_native" in report["failures"]
    assert report["projection_metadata"]["base"]["projection_input_frame"] == "fit_detector"


def test_caked_point_reprojection_fails_on_wrong_projection_source(tmp_path) -> None:
    report, _projector_calls, _module = _run_stub_caked_point_reprojection(
        tmp_path,
        reported_source="stale_caked_alias",
    )

    assert report["status"] == "fail"
    assert report["all_projection_sources_native"] is False
    assert "all_projection_sources_native" in report["failures"]
    assert report["projection_metadata"]["base"]["caked_projection_source"] == "stale_caked_alias"


def test_caked_point_reprojection_stale_alias_guard_blocks_reads(tmp_path) -> None:
    module = _load_new4_caked_reprojection_module()
    guard = module.StaleAliasAccessGuard()
    wrapped = guard.wrap_entries([{"detector_x": 1.0, "detector_y": 2.0, "caked_x": -1.0}])[0]

    assert "caked_x" in wrapped
    with pytest.raises(RuntimeError, match="stale caked alias read forbidden"):
        wrapped.get("caked_x")
    assert guard.read_count == 1

    report, _projector_calls, _module = _run_stub_caked_point_reprojection(tmp_path)
    assert report["status"] == "pass"
    assert report["stale_alias_guard_installed"] is True
    assert report["stale_caked_field_read_count"] == 0


def test_caked_point_reprojection_provider_after_recake_attempt_fails(tmp_path) -> None:
    module = _load_new4_caked_reprojection_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text(json.dumps({"state": {"sentinel": True}}), encoding="utf-8")
    context, provider_report, _projector_calls = _make_caked_point_reprojection_context()

    def provider_after_factory():
        module.exact_cake_portable.convert_image_to_angle_space()
        return dict(provider_report)

    report = module.run_caked_point_reprojection_probe_from_context(
        state_path=state_path,
        background_index=0,
        context=context,
        provider_before_report=provider_report,
        provider_after_factory=provider_after_factory,
        run_dir=tmp_path / "new4_run",
    )

    assert report["status"] == "fail"
    assert report["full_background_recake_called"] is True
    assert report["full_background_recake_call_count"] == 1
    assert "full_background_recake_not_called" in report["failures"]


def test_caked_point_reprojection_projector_recovery_recake_attempt_fails(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_new4_caked_reprojection_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text(json.dumps({"state": {"sentinel": True}}), encoding="utf-8")
    context, provider_report, _projector_calls = _make_caked_point_reprojection_context(
        include_projector=False,
        include_bindings=True,
    )

    def build_dataset_with_forbidden_recake(*_args, **_kwargs):
        module.exact_cake_portable.convert_image_to_angle_space()
        return {}

    monkeypatch.setattr(
        module.preflight,
        "_build_single_background_dataset",
        build_dataset_with_forbidden_recake,
    )

    report = module.run_caked_point_reprojection_probe_from_context(
        state_path=state_path,
        background_index=0,
        context=context,
        provider_before_report=provider_report,
        provider_after_factory=lambda: dict(provider_report),
        run_dir=tmp_path / "new4_run",
    )

    assert report["status"] == "fail"
    assert report["full_background_recake_called"] is True
    assert report["full_background_recake_call_count"] == 1
    assert "full_background_recake_not_called" in report["failures"]


def test_new4_caked_point_reprojection_context_recake_attempt_fails(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_new4_caked_reprojection_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text(json.dumps({"state": {"sentinel": True}}), encoding="utf-8")

    monkeypatch.setattr(
        module.preflight,
        "_run_point_provider_report_only",
        lambda *_args, **_kwargs: {"ok": True, "manual_point_pair_count": 7},
    )

    def prepare_context_with_forbidden_recake(*_args, **_kwargs):
        module.exact_cake_portable.convert_image_to_angle_space()
        return {"ok": True}

    monkeypatch.setattr(
        module.preflight,
        "_prepare_validation_context",
        prepare_context_with_forbidden_recake,
    )

    report = module.run_new4_caked_point_reprojection_check(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path / "new4",
        run_id="recake_context",
    )

    assert report["status"] == "fail"
    assert report["full_recake_guard_installed"] is True
    assert report["full_background_recake_called"] is True
    assert report["full_background_recake_call_count"] == 1
    assert "full_background_recake_not_called" in report["failures"]


def test_new4_caked_point_reprojection_provider_before_recake_attempt_fails(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_new4_caked_reprojection_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text(json.dumps({"state": {"sentinel": True}}), encoding="utf-8")

    def provider_before_with_forbidden_recake(*_args, **_kwargs):
        module.exact_cake_portable.convert_image_to_angle_space()
        return {"ok": True, "manual_point_pair_count": 7}

    monkeypatch.setattr(
        module.preflight,
        "_run_point_provider_report_only",
        provider_before_with_forbidden_recake,
    )
    monkeypatch.setattr(
        module.preflight,
        "_prepare_validation_context",
        lambda *_args, **_kwargs: {"ok": True},
    )

    report = module.run_new4_caked_point_reprojection_check(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path / "new4",
        run_id="recake_provider_before",
    )

    assert report["status"] == "fail"
    assert report["full_recake_guard_installed"] is True
    assert report["full_background_recake_called"] is True
    assert report["full_background_recake_call_count"] == 1
    assert "full_background_recake_not_called" in report["failures"]
    assert report["provider_guard_before_ok"] is False


def test_new4_caked_point_reprojection_context_mutation_fails_state_guard(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_new4_caked_reprojection_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text(json.dumps({"state": {"sentinel": True}}), encoding="utf-8")
    context, provider_report, _projector_calls = _make_caked_point_reprojection_context()
    context["ok"] = True

    monkeypatch.setattr(
        module.preflight,
        "_run_point_provider_report_only",
        lambda *_args, **_kwargs: dict(provider_report),
    )

    def prepare_context_with_state_mutation(*_args, **_kwargs):
        state_path.write_text(json.dumps({"state": {"mutated": True}}), encoding="utf-8")
        return dict(context)

    monkeypatch.setattr(
        module.preflight,
        "_prepare_validation_context",
        prepare_context_with_state_mutation,
    )

    report = module.run_new4_caked_point_reprojection_check(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path / "new4",
        run_id="mutated_context",
    )

    assert report["status"] == "fail"
    assert report["new4_state_hash_unchanged"] is False
    assert report["state_hash_before"] != report["state_hash_after"]
    assert "new4_state_hash_unchanged" in report["failures"]


def test_new4_caked_point_reprojection_context_error_reports_state_hash_change(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_new4_caked_reprojection_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text(json.dumps({"state": {"sentinel": True}}), encoding="utf-8")

    monkeypatch.setattr(
        module.preflight,
        "_run_point_provider_report_only",
        lambda *_args, **_kwargs: {"ok": True, "manual_point_pair_count": 7},
    )

    def prepare_context_with_state_mutation(*_args, **_kwargs):
        state_path.write_text(json.dumps({"state": {"mutated": True}}), encoding="utf-8")
        raise FileNotFoundError("missing local New4 image")

    monkeypatch.setattr(
        module.preflight,
        "_prepare_validation_context",
        prepare_context_with_state_mutation,
    )

    report = module.run_new4_caked_point_reprojection_check(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path / "new4",
        run_id="mutated_missing_context",
    )

    assert report["status"] == "fail"
    assert report["classification"] == "validation_context_unavailable"
    assert report["new4_state_hash_unchanged"] is False
    assert report["state_hash_before"] != report["state_hash_after"]
    assert report["full_background_recake_called"] is False
    assert "new4_state_hash_unchanged" in report["failures"]


def test_new4_caked_point_reprojection_context_error_fails_provider_before_guard(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_new4_caked_reprojection_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text(json.dumps({"state": {"sentinel": True}}), encoding="utf-8")

    monkeypatch.setattr(
        module.preflight,
        "_run_point_provider_report_only",
        lambda *_args, **_kwargs: {"ok": False, "manual_point_pair_count": 7},
    )
    monkeypatch.setattr(
        module.preflight,
        "_prepare_validation_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileNotFoundError("missing local New4 image")
        ),
    )

    report = module.run_new4_caked_point_reprojection_check(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path / "new4",
        run_id="missing_context_bad_provider",
    )

    assert report["status"] == "fail"
    assert report["classification"] == "validation_context_unavailable"
    assert report["provider_guard_before_ok"] is False
    assert report["new4_state_hash_unchanged"] is True
    assert report["full_background_recake_called"] is False
    assert report["failures"] == ["provider_guard_before_ok"]


def test_new4_caked_point_reprojection_context_error_skips_without_recake(
    tmp_path,
    monkeypatch,
) -> None:
    module = _load_new4_caked_reprojection_module()
    state_path = tmp_path / "new4.json"
    state_path.write_text(json.dumps({"state": {"sentinel": True}}), encoding="utf-8")

    monkeypatch.setattr(
        module.preflight,
        "_run_point_provider_report_only",
        lambda *_args, **_kwargs: {"ok": True, "manual_point_pair_count": 7},
    )
    monkeypatch.setattr(
        module.preflight,
        "_prepare_validation_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileNotFoundError("missing local New4 image")
        ),
    )

    report = module.run_new4_caked_point_reprojection_check(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path / "new4",
        run_id="missing_context",
    )

    assert report["status"] == "skip"
    assert report["classification"] == "validation_context_unavailable"
    assert report["failures"] == []
    assert report["full_background_recake_called"] is False
    assert report["full_background_recake_call_count"] == 0
    assert report["guard_error"].startswith("FileNotFoundError:")


def test_new4_caked_point_reprojection_report(tmp_path) -> None:
    if os.environ.get("RA_SIM_RUN_NEW4_CAKED_POINT_SMOKE") != "1":
        pytest.skip("real New4 caked point smoke is opt-in")
    state_path = (
        Path(__file__).resolve().parents[1] / "artifacts" / "geometry_fit_gui_states" / "new4.json"
    )
    if not state_path.is_file():
        pytest.skip("new4 saved state unavailable")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    manual_pairs = payload.get("state", {}).get("geometry", {}).get("manual_pairs", [])
    background_paths = [
        Path(str(pair.get("background_path")))
        for pair in manual_pairs
        if isinstance(pair, Mapping) and pair.get("background_path")
    ]
    missing_paths = [path for path in background_paths if not path.is_file()]
    if missing_paths:
        pytest.skip("local New4 image path unavailable")

    module = _load_new4_caked_reprojection_module()
    report = module.run_new4_caked_point_reprojection_check(
        state_path=state_path,
        background_index=0,
        output_root=tmp_path / "new4",
        run_id="smoke",
    )

    assert report["status"] == "pass"
    assert report["point_count"] == 7
    assert report["provider_guard_before_ok"] is True
    assert report["provider_guard_after_ok"] is True
    assert report["new4_state_hash_unchanged"] is True
    assert report["exact_projector_available"] is True
    assert report["all_exact_projector_used"] is True
    assert report["all_projection_input_frames_native"] is True
    assert report["all_projection_sources_native"] is True
    assert report["full_background_recake_called"] is False
    for pair in report["pairs"]:
        assert pair["projection_input_frame"] == "native_detector"
        assert pair["caked_projection_source"] == "fit_space_projector_native_detector"
        assert pair["exact_projector_used"] is True
        assert pair["stale_caked_fields_used"] is False
        assert np.all(np.isfinite(pair["base_caked_point"]))
        assert np.all(np.isfinite(pair["theta_perturbed_caked_point"]))
        assert np.all(np.isfinite(pair["distance_perturbed_caked_point"]))


def test_geometry_fit_canonical_live_source_entry_ignores_source_peak_mirror() -> None:
    is_canonical, reason = geometry_fit._geometry_fit_is_canonical_live_source_entry(
        {
            "hkl": (1, 0, 0),
            "source_reflection_index": 7,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 1,
        }
    )

    assert is_canonical is True
    assert reason is None


def test_build_geometry_fit_caked_roi_selection_falls_back_when_roi_is_too_large() -> None:
    source_rows = [
        {
            "hkl": (1, 0, 0),
            "q_group_key": ("selected", 1),
            "source_table_index": 0,
            "source_reflection_index": 7,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 0,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "background_two_theta_deg": 1.0,
            "background_phi_deg": -10.0,
            "detector_x": 80.0,
            "detector_y": 80.0,
        },
        {
            "hkl": (1, 0, 0),
            "q_group_key": ("selected", 1),
            "source_table_index": 0,
            "source_reflection_index": 7,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_row_index": 1,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "background_two_theta_deg": 2.0,
            "background_phi_deg": -10.0,
            "detector_x": 84.0,
            "detector_y": 80.0,
        },
    ]

    projection_map = {
        (1.0, -10.0): (5.0, 25.0),
        (2.0, -10.0): (45.0, 25.0),
    }

    def _project(two_theta_deg: float, phi_deg: float):
        return projection_map.get((float(two_theta_deg), float(phi_deg)))

    selection = geometry_fit.build_geometry_fit_caked_roi_selection(
        source_rows,
        required_pairs=[
            {
                "pair_id": "bg0:pair0",
                "overlay_match_index": 0,
                "hkl": (1, 0, 0),
                "q_group_key": ("selected", 1),
                "source_reflection_index": 7,
                "source_reflection_namespace": "full_reflection",
                "source_reflection_is_full": True,
                "source_row_index": 0,
                "source_branch_index": 0,
                "source_peak_index": 0,
            }
        ],
        image_shape=(50, 50),
        fit_config={
            "caked_roi": {
                "enabled": True,
                "half_width_px": 8.0,
                "max_detector_fraction": 0.05,
            }
        },
        fit_space_to_detector_point=_project,
    )

    assert selection["valid"] is False
    assert selection["fallback_reason"] == "roi_too_large"
    assert selection["pixel_count"] > 0
    assert selection["fraction"] > 0.05
