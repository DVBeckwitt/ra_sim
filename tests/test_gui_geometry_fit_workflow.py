import numpy as np

from ra_sim.gui import geometry_fit


class _DummyVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


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
        or {"bounds": {"gamma": [0.0, 1.0]}},
    )

    assert result.error_text is None
    assert result.prepared_run is not None

    prepared = result.prepared_run
    assert prepared.fit_params["theta_offset"] == 0.5
    assert prepared.fit_params["theta_initial"] == 2.5
    assert prepared.selected_background_indices == [0, 1, 2]
    assert prepared.background_theta_values == [1.0, 2.0, 3.0]
    assert prepared.joint_background_mode is True
    assert prepared.max_display_markers == 150
    assert [entry["dataset_index"] for entry in prepared.dataset_infos] == [1, 0, 2]
    assert prepared.current_dataset["dataset_index"] == 1

    assert calls["selection"] == [{"trigger_update": False, "sync_live_theta": True}]
    assert calls["theta"] == [{"trigger_update": False, "sync_live_theta": True}]
    assert calls["ensure_caked"] == 1
    assert calls["datasets"] == [
        (1, 2.0, 2.5, 0.5, {"mode": "auto"}),
        (0, 1.0, 2.5, 0.5, {"mode": "auto"}),
        (2, 3.0, 2.5, 0.5, {"mode": "auto"}),
    ]
    assert calls["runtime_cfg"] == [{"theta_initial": 2.5, "theta_offset": 0.5}]


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
        build_runtime_config=lambda fit_params: dict(fit_params),
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

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=1.5,
        base_fit_params={"theta_offset": 0.25},
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
        orientation_cfg={"mode": "auto"},
    )

    assert dataset["dataset_index"] == 0
    assert dataset["label"] == "bg0.osc"
    assert dataset["theta_base"] == 1.5
    assert dataset["theta_effective"] == 1.75
    assert dataset["group_count"] == 1
    assert dataset["pair_count"] == 1
    assert dataset["initial_pairs_display"] == [
        {
            "overlay_match_index": 0,
            "hkl": (1, 1, 0),
            "bg_display": (50.0, 60.0),
            "sim_display": (9.0, 8.0),
        }
    ]
    assert dataset["measured_native"] == measured_native
    assert dataset["measured_for_fit"] == measured_for_fit
    assert dataset["experimental_image_for_fit"] == "fit-image"
    assert dataset["orientation_choice"] == orientation_choice
    assert dataset["orientation_diag"] == orientation_diag
    assert dataset["spec"] == {
        "dataset_index": 0,
        "label": "bg0.osc",
        "theta_initial": 1.5,
        "measured_peaks": measured_for_fit,
        "experimental_image": "fit-image",
    }
    assert "orientation=rotate+flip" in dataset["summary_line"]

    sim_params, prefer_cache = calls["sim_params"]
    assert sim_params["theta_initial"] == 1.75
    assert prefer_cache is True
    assert calls["unrotate"][1:] == ((6, 7), 3)
    assert calls["display_to_native"] == (9.0, 8.0, (100, 100))
    assert calls["select_orientation"] == (
        [(1.0, 2.0)],
        [(30.0, 40.0)],
        (4, 5),
        {"mode": "auto"},
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


def test_geometry_fit_post_solver_helpers_format_diagnostics_and_summary() -> None:
    class _Result:
        success = True
        status = 3
        message = "done"
        nfev = 17
        cost = 1.25
        robust_cost = 1.0
        solver_loss = "soft_l1"
        solver_f_scale = 2.0
        optimality = 0.005
        active_mask = [0, 1]
        restart_history = [
            {"restart": 0, "cost": 2.5, "success": False, "message": "retry"}
        ]
        rms_px = 0.75
        x = [1.5, 2.5]

    diagnostics = geometry_fit.build_geometry_fit_optimizer_diagnostics_lines(_Result())
    assert diagnostics[:4] == [
        "success=True",
        "status=3",
        "message=done",
        "nfev=17",
    ]
    assert diagnostics[-1] == "restart[0] cost=2.500000 success=False msg=retry"

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
    )
    assert summary_lines == [
        "manual_groups=4",
        "manual_points=9",
        "overlay_records=7",
        "orientation=rotate",
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
    assert "HKL=(1, 1, 0): |Δ|=1.80px (dx=1.50, dy=1.00)" in progress_text


def test_geometry_fit_result_rms_falls_back_to_residual_vector() -> None:
    class _Result:
        rms_px = float("nan")
        fun = np.array([3.0, 4.0], dtype=np.float64)

    assert np.isclose(
        geometry_fit.geometry_fit_result_rms(_Result()),
        5.0 / np.sqrt(2.0),
    )


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


def test_postprocess_geometry_fit_result_builds_overlay_export_and_status_payloads(
    tmp_path,
) -> None:
    class _Result:
        point_match_summary = {"claimed": 4, "qualified": 3}
        point_match_diagnostics = [
            {"dataset_index": 0, "name": "drop"},
            {"dataset_index": 1, "name": "keep"},
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
        calls["aggregate"] = (
            list(sim_coords),
            list(meas_coords),
            list(sim_millers),
            list(meas_millers),
        )
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
        fitted_params={"theta_initial": 3.0, "a": 4.2},
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

    assert calls["simulate"] == {
        "miller": "miller-data",
        "intensities": "intensity-data",
        "image_size": 512,
        "fitted_params": {"theta_initial": 3.0, "a": 4.2},
        "measured_for_fit": [{"x": 1.0, "y": 2.0}],
        "pixel_tol": float("inf"),
    }
    assert calls["overlay"]["overlay_point_match_diagnostics"] == [
        {"dataset_index": 1, "name": "keep"}
    ]
    assert postprocess.point_match_summary_lines == ["claimed=4", "qualified=3"]
    assert postprocess.pixel_offsets == [((1, 1, 0), -1.5, 1.0, np.hypot(-1.5, 1.0))]
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
            "source": "sim",
            "hkl": (1, 1, 0),
            "x": 10,
            "y": 20,
            "dist_px": np.hypot(-1.5, 1.0),
        },
        {
            "source": "meas",
            "hkl": (1, 1, 0),
            "x": 11,
            "y": 19,
            "dist_px": np.hypot(-1.5, 1.0),
        },
    ]
    assert postprocess.save_path == tmp_path / "matched_peaks_20260328_120000.npy"
    assert postprocess.fit_summary_lines == [
        "manual_groups=2",
        "manual_points=3",
        "overlay_records=1",
        "orientation=rotate",
        "gamma = 1.500000",
        "a = 2.500000",
        "RMS residual = 0.750000 px",
        f"Matched peaks saved to: {postprocess.save_path}",
    ]
    assert "frame warning" in postprocess.progress_text
    assert "joint backgrounds=2" in postprocess.progress_text
    assert f"Saved 2 peak records → {postprocess.save_path}" in postprocess.progress_text


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
        x = [1.5, 2.5]
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
