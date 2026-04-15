import importlib.util
import json
import hashlib
from dataclasses import replace
import numpy as np
from pathlib import Path
import pytest
from types import SimpleNamespace

from ra_sim.fitting import optimization as opt
from ra_sim.gui import geometry_fit, geometry_overlay, manual_geometry
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
        apply_background_theta_metadata=(
            lambda **kwargs: calls["theta"].append(kwargs) or True
        ),
        current_background_theta_values=lambda **kwargs: [1.0, 2.0, 3.0],
        current_geometry_theta_offset=lambda **kwargs: 0.5,
        geometry_manual_pairs_for_index=lambda idx: [{"pair": idx}],
        ensure_geometry_fit_caked_view=lambda: calls.__setitem__(
            "ensure_caked", int(calls["ensure_caked"]) + 1
        ),
        build_dataset=_build_dataset,
        build_runtime_config=lambda fit_params: calls["runtime_cfg"].append(
            dict(fit_params)
        )
        or {"bounds": {"gamma": [0.0, 1.0]}, "debug_logging": True},
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
                load_background_by_index=(
                    lambda idx: (native_background, display_background)
                ),
                apply_background_backend_orientation=lambda image: None,
                geometry_manual_simulated_peaks_for_params=(
                    lambda params, *, prefer_cache: [{"dummy": True}]
                ),
                geometry_manual_simulated_lookup=lambda _peaks: {
                    (1, 2): {"sim_col": 9.0, "sim_row": 8.0}
                },
                geometry_manual_entry_display_coords=lambda entry: (50.0, 60.0),
                unrotate_display_peaks=(
                    lambda entries, shape, *, k: [{"x": 30.0, "y": 40.0}]
                ),
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
    assert prepared.dataset_specs == [
        {
            "dataset_index": 0,
            "label": "bg0.osc",
            "theta_initial": 9.0,
            "measured_peaks": [
                {
                    "x": 31.0,
                    "y": 41.0,
                    "detector_x": 31.0,
                    "detector_y": 41.0,
                    "background_detector_x": 31.0,
                    "background_detector_y": 41.0,
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
            (1, 2): {"sim_col": 9.0, "sim_row": 8.0}
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
            "sim_display": (9.0, 8.0),
            "bg_native": (30.0, 40.0),
            "sim_native": (1.0, 2.0),
        }
    ]
    assert dataset["measured_native"] == measured_native
    assert dataset["measured_for_fit"] == measured_for_fit
    assert "experimental_image_for_fit" not in dataset
    assert dataset["orientation_choice"] == orientation_choice
    assert dataset["orientation_diag"] == orientation_diag
    assert dataset["spec"] == {
        "dataset_index": 0,
        "label": "bg0.osc",
        "theta_initial": 1.5,
        "measured_peaks": measured_for_fit,
        "experimental_image": "fit-image",
        "dynamic_reanchor_callback": None,
        "dynamic_reanchor_enabled": False,
    }
    assert dataset["measured_for_fit"] == [
        {
            "x": 31.0,
            "y": 41.0,
            "detector_x": 31.0,
            "detector_y": 41.0,
            "background_detector_x": 31.0,
            "background_detector_y": 41.0,
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
        geometry_manual_simulated_peaks_for_params=lambda params, *, prefer_cache: [{"dummy": True}],
        geometry_manual_simulated_lookup=lambda _peaks: {
            (1, 2): {
                "source_table_index": 1,
                "source_row_index": 2,
                "sim_col": 50.0,
                "sim_row": 60.0,
            },
            (2, 4): {
                "source_table_index": 2,
                "source_row_index": 4,
                "sim_col": 10.0,
                "sim_row": 15.0,
            },
        },
        geometry_manual_entry_display_coords=_entry_display_coords,
        unrotate_display_peaks=lambda entries, shape, *, k: [dict(entry) for entry in entries],
        display_to_native_sim_coords=lambda col, row, shape: (float(col) / 10.0, float(row) / 10.0),
        select_fit_orientation=_select_fit_orientation,
        apply_orientation_to_entries=lambda entries, shape, **kwargs: [dict(entry) for entry in entries],
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
    assert all(
        entry["fit_source_identity_only"] is True
        for entry in dataset["measured_for_fit"]
    )
    assert calls["select_orientation"] == (
        [(5.0, 6.0), (1.0, 1.5)],
        [(30.0, 40.0), (70.0, 80.0)],
        (4, 5),
        {"mode": "auto"},
    )


def test_build_geometry_manual_fit_dataset_preserves_branch_and_full_reflection_provenance() -> None:
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
        geometry_manual_simulated_lookup=lambda _simulated_peaks: {
            (1, 2): dict(source_row)
        },
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
        apply_orientation_to_entries=lambda entries, shape, **kwargs: [dict(entry) for entry in entries],
        orient_image_for_fit=lambda image, **kwargs: image,
        geometry_manual_source_rows_for_background=lambda *args, **kwargs: [
            dict(source_row)
        ],
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
        apply_orientation_to_entries=lambda entries, shape, **kwargs: [dict(entry) for entry in entries],
        orient_image_for_fit=lambda image, **kwargs: image,
        geometry_manual_source_rows_for_background=_rows,
        geometry_manual_rebuild_source_rows_for_background=_rows,
        geometry_manual_refresh_pair_entry=(
            _refresh_legacy_dense_pair_entry if refresh_pairs else None
        ),
    )


def _load_new2_probe_module():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "debug"
        / "validate_new2_preflight_rebind.py"
    )
    spec = importlib.util.spec_from_file_location(
        "validate_new2_preflight_rebind",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
        geometry_manual_simulated_peaks_for_params=lambda params, *, prefer_cache: [{"dummy": True}],
        geometry_manual_simulated_lookup=lambda _simulated_peaks: {
            (1, 2): {
                "sim_col": 91.0,
                "sim_row": 82.0,
                "sim_col_raw": 9.0,
                "sim_row_raw": 8.0,
            }
        },
        geometry_manual_entry_display_coords=lambda entry: (150.0, 160.0),
        unrotate_display_peaks=lambda entries, shape, *, k: [{"x": 30.0, "y": 40.0}],
        display_to_native_sim_coords=_display_to_native_sim_coords,
        select_fit_orientation=lambda sim_pts, meas_pts, shape, *, cfg: (
            {"indexing_mode": "xy", "k": 0, "flip_x": False, "flip_y": False, "flip_order": "yx", "label": "identity"},
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
        apply_orientation_to_entries=lambda entries, shape, **kwargs: [dict(entry) for entry in entries],
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


def test_build_geometry_manual_fit_dataset_copyback_preserves_canonical_identity_after_legacy_dense_rebind() -> None:
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


def test_build_geometry_manual_fit_dataset_rebinds_mirrored_legacy_dense_pairs_by_branch_and_geometry() -> None:
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
    assert all(entry["source_reflection_namespace"] == "full_reflection" for entry in measured_entries)
    assert all(entry["source_reflection_is_full"] is True for entry in measured_entries)
    diag0, diag1 = dataset["source_resolution_diagnostics"]
    assert diag0["fit_resolution_kind"] == "legacy_dense_q_group_rebind"
    assert diag0["legacy_branch_hint_source"] == "refined_sim_caked_y"
    assert diag1["fit_resolution_kind"] == "legacy_dense_q_group_rebind"
    assert diag1["legacy_branch_hint_source"] == "refined_sim_caked_y"
    assert diag1["legacy_geometry_hint_source"] == "refined_sim_display"
    assert diag1["legacy_candidate_count_initial"] == 3
    assert diag1["legacy_candidate_count_after_branch"] == 2
    assert diag1["legacy_chosen_live_row"]["source_reflection_index"] == 204


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
    assert diag["legacy_geometry_hint_source"] == "refined_sim_display"
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

    monkeypatch.setattr(headless_geometry_fit, "_build_runtime_defaults", lambda saved_state: defaults)
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
            rebuild_source="peak_records_fallback",
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
    assert headless_dataset["resolved_source_pair_count"] == workflow_dataset["resolved_source_pair_count"]
    assert [
        entry["source_reflection_index"] for entry in headless_dataset["measured_for_fit"]
    ] == [entry["source_reflection_index"] for entry in workflow_dataset["measured_for_fit"]]
    assert [
        entry["source_branch_index"] for entry in headless_dataset["measured_for_fit"]
    ] == [entry["source_branch_index"] for entry in workflow_dataset["measured_for_fit"]]
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

    monkeypatch.setattr(headless_geometry_fit, "_build_runtime_defaults", lambda saved_state: defaults)
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
    assert headless_dataset["resolved_source_pair_count"] == workflow_dataset["resolved_source_pair_count"]
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


def test_select_live_candidate_for_saved_entry_rejects_same_branch_pixel_tie() -> None:
    probe = _load_new2_probe_module()

    result = probe._select_live_candidate_for_saved_entry(
        saved_entry={
            "hkl": (-1, 0, 5),
            "source_branch_index": 1,
            "source_peak_index": 1,
            "refined_sim_x": 10.0,
            "refined_sim_y": 10.0,
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
    )

    assert result["ok"] is False
    assert result["selection_status"] == "ambiguous_live_row_selection"
    assert len(result["tied_candidate_inventory"]) == 2


def test_compatibility_probe_slot_indices_prefers_first_mirrored_group() -> None:
    probe = _load_new2_probe_module()

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
    probe = _load_new2_probe_module()

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

    assert classification == "legacy_saved_identity_canonicalized"
    assert delta == {
        "source_reflection_index": {
            "saved": 15,
            "selected": 214,
        }
    }


def test_probe_main_aliases_full_to_fresh_all(
    monkeypatch,
    capsys,
    tmp_path,
) -> None:
    probe = _load_new2_probe_module()
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
        lambda *args, **kwargs: calls.append("fresh-all")
        or {"ok": True, "classification": "pass"},
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


def test_saved_state_compatibility_validation_handles_two_entry_pair(
    monkeypatch,
    tmp_path,
) -> None:
    probe = _load_new2_probe_module()
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
    probe = _load_new2_probe_module()
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
    state_path = tmp_path / "new2_like.json"
    context = {
        "ok": True,
        "state_path": str(state_path),
        "background_index": 0,
        "saved_pair_count": len(saved_entries),
        "captured_preflight_error_text": None,
        "used_isolated_background_dataset": False,
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
        emitted_pair = {
            "pair_id": f"bg0:pair{int(slot_index)}",
            "q_group_key": ("q_group", "primary", 1, int(slot_index) + 1),
            "hkl": (-1, 0, int(slot_index) + 1),
            "source_reflection_index": 200 + int(slot_index),
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": int(slot_index) % 2,
            "source_peak_index": int(slot_index) % 2,
        }
        return {
            "ok": True,
            "classification": "pass",
            "slot_index": int(slot_index),
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
    assert result["slot_results"][0]["saved_to_selected_identity_delta_classification"] == (
        "legacy_saved_identity_canonicalized"
    )
    assert result["slot_results"][1]["saved_to_selected_identity_delta_classification"] == (
        "saved_identity_already_canonical"
    )
    assert result["exported_fresh_state_path"] == str(export_path.resolve())
    assert result["exported_state_compatibility"]["ok"] is True


def test_downstream_identity_validation_rejects_non_canonical_input(
    monkeypatch,
    tmp_path,
) -> None:
    probe = _load_new2_probe_module()
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
    probe = _load_new2_probe_module()
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
    probe = _load_new2_probe_module()
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
                subset=SimpleNamespace(
                    measured_entries=[dict(entry) for entry in saved_entries]
                )
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
    probe = _load_new2_probe_module()
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
                subset=SimpleNamespace(
                    measured_entries=[dict(entry) for entry in saved_entries]
                )
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
    probe = _load_new2_probe_module()
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
                subset=SimpleNamespace(
                    measured_entries=[dict(entry) for entry in saved_entries]
                )
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
                "sim_col": 91.0,
                "sim_row": 82.0,
                "sim_col_raw": 9.0,
                "sim_row_raw": 8.0,
            }
        },
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

    assert dataset["initial_pairs_display"][0]["sim_caked_display"] == (91.0, 82.0)
    assert dataset["initial_pairs_display"][0]["bg_caked_display"] == (150.0, 160.0)


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


def test_build_geometry_manual_fit_dataset_uses_saved_refined_caked_coords_without_live_source() -> None:
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
        geometry_manual_simulated_lookup=lambda peaks: {
            (1, 2): dict(valid_source_row)
        },
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
    assert dataset["measured_for_fit"][0]["background_detector_x"] == 30.0
    assert dataset["measured_for_fit"][0]["background_detector_y"] == 40.0
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
        lambda image, cfg: {"img_valid": True, "shape": tuple(np.asarray(image).shape), "cfg": dict(cfg)},
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
        geometry_manual_simulated_lookup=lambda peaks: {
            (1, 2): dict(valid_source_row)
        },
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


def test_geometry_fit_dynamic_reanchor_uses_caked_fit_space_seed_and_returns_fit_space_override(
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
    bundle = geometry_fit.CakeTransformBundle(
        detector_shape=(6, 7),
        radial_deg=np.array([20.0], dtype=np.float64),
        raw_azimuth_deg=np.array([0.0], dtype=np.float64),
        gui_azimuth_deg=np.array([0.0], dtype=np.float64),
        lut=object(),
    )
    projector_calls: list[tuple[object, float, float]] = []
    monkeypatch.setattr(
        geometry_fit,
        "detector_pixel_to_caked_bin",
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
            lambda idx, params, *, consumer, required_pairs=None: [
                dict(valid_source_row)
            ]
        ),
        geometry_manual_simulated_peaks_for_params=(
            lambda params, *, prefer_cache: [dict(valid_source_row)]
        ),
        geometry_manual_simulated_lookup=lambda peaks: {
            (1, 2): dict(valid_source_row)
        },
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
    bundle = geometry_fit.CakeTransformBundle(
        detector_shape=(6, 7),
        radial_deg=np.array([20.0], dtype=np.float64),
        raw_azimuth_deg=np.array([0.0], dtype=np.float64),
        gui_azimuth_deg=np.array([0.0], dtype=np.float64),
        lut=object(),
    )
    projector_calls: list[tuple[object, float, float]] = []
    monkeypatch.setattr(
        geometry_fit,
        "detector_pixel_to_caked_bin",
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
            lambda idx, params, *, consumer, required_pairs=None: [
                dict(valid_source_row)
            ]
        ),
        geometry_manual_simulated_peaks_for_params=(
            lambda params, *, prefer_cache: [dict(valid_source_row)]
        ),
        geometry_manual_simulated_lookup=lambda peaks: {
            (1, 2): dict(valid_source_row)
        },
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


def test_rebuild_geometry_fit_source_rows_rejects_invalid_live_cache_for_required_pair_and_records_dual_path_diff() -> None:
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


def test_peak_record_fallback_with_restored_provenance_matches_rebuild_for_active_pairs() -> None:
    peak_records = [
        {
            "display_col": 10.0,
            "display_row": 20.0,
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
    assert geometry_fit._geometry_fit_required_pair_dual_path_diff(
        fixed_validation,
        rebuild_validation,
    ) == []
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
                "cache_source": "peak_records_fallback",
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
    assert accepted.metadata["live_runtime_cache_metadata"]["cache_source"] == (
        "peak_records_fallback"
    )
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
        geometry_manual_simulated_peaks_for_params=(
            lambda params, *, prefer_cache: []
        ),
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
            current_geometry_theta_offset=lambda strict=False: theta_offset_state[
                "value"
            ],
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
            calls.append(("domains", list(names or [])))
            or {"gamma": (0.0, 1.0)}
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
                log_path=Path("C:/tmp/geometry_fit_log.txt"),
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
    assert lines[2] == (
        "bg0.osc: pair_count=3 resolved_source_pairs=2 simulated_peaks=9 tables=1"
    )
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
        "bg0.osc: resolved_source_pairs=1/3 simulated_peaks=70 "
        "simulated_source_rows=61"
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
        "bg0.osc: all saved manual pairs resolved for fit "
        "(including legacy source-id remaps)."
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


def test_build_geometry_fit_simulation_diagnostic_log_lines_include_runtime_failure_details() -> None:
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
                "source_resolution_diagnostics": [
                    {"pair_index": 0, "strict_resolved": False}
                ],
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
    assert lines[1] == (
        "pixel_size=nan pixel_size_m=0.000100 debye_x=0.000000 debye_y=0.000000"
    )
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
            "dataset_specs": [{"dataset_index": 0, "theta_initial": 3.0}],
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


def test_apply_manual_point_geometry_fit_runtime_overrides_preserves_unsafe_parallel_settings() -> None:
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
            "Possible identifiability issue: parameters finished near bounds "
            "(Gamma=upper)."
        )
        restart_history = [
            {"restart": 0, "cost": 2.5, "success": False, "message": "retry"}
        ]
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
    assert any(
        "seed_search prescore_top_k=2 n_global=3 n_jitter=1" in line
        for line in debug_lines
    )
    assert any(
        "discrete_modes count=2 selected=rot90=1 labels=[identity, rot90=1]"
        in line
        for line in debug_lines
    )
    assert any(
        "mode_seed[0] label=identity total_seeds=5 selected_seeds=2" in line
        for line in debug_lines
    )
    assert any("main_seed_point_match matched=4 missing=1" in line for line in debug_lines)
    assert any("main_seed kind=zero label=u=0 cost=3.500000" in line for line in debug_lines)
    assert any("seed_rms_px=1.500000" in line for line in debug_lines)
    assert any("dataset[0] label=bg0" in line for line in debug_lines)
    assert any("param[gamma] group=tilt start=1.000000 final=1.500000 delta=0.500000" in line for line in debug_lines)
    assert any("full_beam_polish=True full_beam_radius_px=24.000000" in line for line in debug_lines)
    assert any("final metric=central_point_match cost=1.250000" in line for line in debug_lines)
    assert any("solve_counts prescored=6 solved=2" in line for line in debug_lines)
    assert any(
        "solve_progress label=main solve evaluations=12" in line
        and "last_cost=1.100000" in line
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
        line.startswith("adaptive_regularization:")
        and "applied_parameters=[Gamma]" in line
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
        line.startswith("auto_freeze:")
        and "fixed_parameters=[a]" in line
        for line in stage_lines
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
        "transform_rule=sim:direct_native_to_display; bg:inverse_orientation_then_display_rotation",
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
        bound_proximity_summary={
            "near_bound_parameters": [{"name": "Gamma", "side": "upper"}]
        },
    )

    reasons = geometry_fit.build_geometry_fit_rejection_reason_lines(result, rms=0.75)

    assert reasons == [
        "Fit is underconstrained according to the final identifiability diagnostics.",
        "Parameters finished within 1% of a finite bound span from a bound: Gamma(upper).",
    ]


def test_build_geometry_fit_rejection_reason_lines_ignores_bound_proximity_without_underconstraint() -> None:
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


def test_capture_runtime_geometry_fit_undo_state_builds_overlay_fallback_from_manual_pairs() -> None:
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
        set_last_overlay_state=lambda overlay_state: events.append(("overlay_state", overlay_state)),
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


def test_make_runtime_geometry_tool_action_callbacks_does_not_force_caked_view_when_arming() -> None:
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

    assert setup.postprocess_config.log_path == (
        tmp_path / "geometry_fit_log_20260328_123000.txt"
    )
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
    assert setup.postprocess_config.log_path == (
        log_dir / "geometry_fit_log_20260328_123010.txt"
    )
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
            prepare_bindings_factory=lambda _var_names: (
                _make_runtime_action_prepare_bindings()
            ),
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
            prepare_bindings_factory=lambda _var_names: (
                _make_runtime_action_prepare_bindings()
            ),
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
    assert action.prepare_result.log_path == (
        tmp_path / "geometry_fit_log_20260328_130001.txt"
    )
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
            prepare_bindings_factory=lambda _var_names: (
                _make_runtime_action_prepare_bindings()
            ),
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
            prepare_bindings_factory=lambda _var_names: (
                _make_runtime_action_prepare_bindings()
            ),
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
    assert action.prepare_result.log_path == (
        tmp_path / "geometry_fit_log_20260328_130002.txt"
    )
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
    assert action.prepare_result.log_path == (
        tmp_path / "geometry_fit_log_20260328_130003.txt"
    )
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
        "transform_rule=sim:direct_native_to_display; bg:inverse_orientation_then_display_rotation",
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
    diagnostic_sections = [
        lines
        for title, lines in events
        if title == "Point-match diagnostics:"
    ]
    assert len(diagnostic_sections) == 1
    assert diagnostic_sections[0][0] == "match_status: missing_pair=1"
    assert "dataset[0] bg0.osc: unresolved_pairs=1" in diagnostic_sections[0]
    assert any(
        "overlay_index=7" in line and "measured_detector=[50.000, 60.000]" in line
        for line in diagnostic_sections[0]
    )
    assert any(
        "resolution_reason=source_row_out_of_range" in line
        for line in diagnostic_sections[0]
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
    assert "RMS residual 1195.37 px exceeds the acceptance limit" in str(
        outcome.rejection_reason
    )
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
        event
        for event in events
        if isinstance(event, tuple) and event[0] == "draw_overlay"
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
            "branch_candidates": [],
        }
    ]


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
            "detector_x": 20.0,
            "detector_y": 20.0,
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
            "detector_x": 24.0,
            "detector_y": 20.0,
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
                "detector_x": 80.0,
                "detector_y": 80.0,
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
            "detector_x": 60.0,
            "detector_y": 60.0,
        },
    ]

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
        fit_config={"caked_roi": {"half_width_px": 0.0}},
    )

    assert selection["valid"] is True
    assert selection["resolved_pair_count"] == 1
    assert selection["selected_branch_count"] == 1
    pixels = set(zip(selection["rows"].tolist(), selection["cols"].tolist()))
    assert (20, 20) in pixels
    assert (20, 22) in pixels
    assert (80, 80) not in pixels
    assert (60, 60) not in pixels


def test_build_geometry_fit_caked_roi_selection_prefers_fit_space_projection() -> None:
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
            "two_theta_deg": 1.0,
            "phi_deg": -10.0,
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
            "two_theta_deg": 2.0,
            "phi_deg": -10.0,
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
            "two_theta_deg": 1.0,
            "phi_deg": 10.0,
        },
    ]

    projection_map = {
        (1.0, -10.0): (20.0, 20.0),
        (2.0, -10.0): (24.0, 20.0),
        (1.0, 10.0): (40.0, 40.0),
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
        fit_config={"caked_roi": {"half_width_px": 0.0}},
        fit_space_to_detector_point=_project,
    )

    assert selection["valid"] is True
    pixels = set(zip(selection["rows"].tolist(), selection["cols"].tolist()))
    assert (20, 20) in pixels
    assert (20, 22) in pixels
    assert (80, 80) not in pixels
    assert (40, 40) not in pixels


def test_geometry_fit_caked_roi_angle_point_uses_canonical_angles_only() -> None:
    assert geometry_fit._geometry_fit_caked_roi_angle_point(
        {
            "background_two_theta_deg": 23.0,
            "background_phi_deg": -36.0,
            "caked_x": 150.0,
            "caked_y": 160.0,
        }
    ) == pytest.approx((23.0, -36.0))
    assert geometry_fit._geometry_fit_caked_roi_angle_point(
        {
            "caked_x": 150.0,
            "caked_y": 160.0,
        }
    ) is None


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
            "detector_x": 5.0,
            "detector_y": 25.0,
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
            "detector_x": 45.0,
            "detector_y": 25.0,
        },
    ]

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
                "half_width_px": 8.0,
                "max_detector_fraction": 0.05,
            }
        },
    )

    assert selection["valid"] is False
    assert selection["fallback_reason"] == "roi_too_large"
    assert selection["pixel_count"] > 0
    assert selection["fraction"] > 0.05
