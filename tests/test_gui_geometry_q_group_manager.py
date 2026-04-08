from datetime import datetime

import numpy as np

from ra_sim.gui import geometry_q_group_manager, state


class _FakeVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def _entry(group_key, *, peak_count, total_intensity, source="primary"):
    return {
        "key": group_key,
        "source_label": source,
        "qr": 1.25,
        "qz": 0.5,
        "gz_index": group_key[3],
        "total_intensity": total_intensity,
        "peak_count": peak_count,
        "hkl_preview": [(1, 0, 0), (1, 1, 0), (2, 0, 0), (2, 1, 0)],
    }


def test_geometry_q_group_manager_geometry_metadata_helpers() -> None:
    rows = geometry_q_group_manager.geometry_reference_hit_rows(
        [
            [10.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0],
            [5.0, np.nan, 2.0, 0.0, 1.0, 0.0, 0.0],
            [7.0, 1.0, 2.0, 0.0, 1.0, 0.0],
        ]
    )
    key, qr_val, qz_val = geometry_q_group_manager.reflection_q_group_metadata(
        (1.0, 0.0, 2.0),
        source_label="SECONDARY",
        a_value=2.5,
        c_value=5.0,
    )

    assert len(rows) == 1
    np.testing.assert_allclose(rows[0], [10.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0])
    assert key == ("q_group", "secondary", 1, 2)
    assert np.isclose(qr_val, (2.0 * np.pi / 2.5) * np.sqrt(4.0 / 3.0))
    assert np.isclose(qz_val, (2.0 * np.pi / 5.0) * 2.0)
    assert (
        geometry_q_group_manager.geometry_q_group_key_from_entry(
            {
                "hkl_raw": (1.0, 0.0, 2.0),
                "source_label": "secondary",
                "av": 2.5,
                "cv": 5.0,
            }
        )
        == key
    )
    missing_key, missing_qr, missing_qz = (
        geometry_q_group_manager.reflection_q_group_metadata(
            (1.0, 0.0, 1.25),
            source_label="primary",
            a_value=3.0,
            c_value=6.0,
        )
    )
    assert missing_key is None
    assert np.isnan(missing_qr)
    assert np.isnan(missing_qz)

    nominal_key, nominal_qr, nominal_qz = (
        geometry_q_group_manager.reflection_q_group_metadata(
            (1.0, 0.0, 1.25),
            source_label="primary",
            a_value=3.0,
            c_value=6.0,
            allow_nominal_hkl_indices=True,
        )
    )
    assert nominal_key == ("q_group", "primary", 1, 1)
    assert np.isclose(nominal_qr, (2.0 * np.pi / 3.0) * np.sqrt(4.0 / 3.0))
    assert np.isclose(nominal_qz, 2.0 * np.pi / 6.0)


def test_geometry_q_group_manager_nominal_hkl_grouping_supports_cache_rows() -> None:
    cache_like_hit_tables = [
        np.asarray(
            [
                [12.0, 10.2, 20.8, 0.0, 1.0, 0.0, 1.29],
            ],
            dtype=float,
        )
    ]

    peaks = geometry_q_group_manager.build_geometry_fit_simulated_peaks(
        cache_like_hit_tables,
        image_shape=(32, 32),
        native_sim_to_display_coords=lambda col, row, _shape: (col, row),
        peak_table_lattice=[(3.0, 5.0, "primary")],
        allow_nominal_hkl_indices=True,
    )
    entries = geometry_q_group_manager.build_geometry_q_group_entries(
        cache_like_hit_tables,
        peak_table_lattice=[(3.0, 5.0, "primary")],
        allow_nominal_hkl_indices=True,
    )

    assert len(peaks) == 1
    assert peaks[0]["hkl"] == (1, 0, 1)
    assert peaks[0]["q_group_key"] == ("q_group", "primary", 1, 1)
    assert peaks[0]["q_group_nominal_hkl"] is True
    assert len(entries) == 1
    assert entries[0]["key"] == ("q_group", "primary", 1, 1)
    assert entries[0]["hkl_preview"] == [(1, 0, 1)]


def test_geometry_q_group_manager_builds_entries_from_hit_tables() -> None:
    entries = geometry_q_group_manager.build_geometry_q_group_entries(
        [
            np.asarray(
                [
                    [10.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0],
                    [5.0, 3.0, 4.0, 0.0, 1.0, 0.0, 0.0],
                    [8.0, 5.0, 6.0, 0.0, 1.0, 0.0, 1.0],
                ],
                dtype=float,
            ),
            np.asarray(
                [
                    [6.0, 7.0, 8.0, 0.0, 1.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
        ],
        peak_table_lattice=[
            (3.0, 5.0, "primary"),
            (4.0, 6.0, "secondary"),
        ],
    )

    assert [entry["key"] for entry in entries] == [
        ("q_group", "primary", 1, 0),
        ("q_group", "primary", 1, 1),
        ("q_group", "secondary", 3, 0),
    ]
    assert entries[0]["total_intensity"] == 15.0
    assert entries[0]["peak_count"] == 2
    assert entries[0]["hkl_preview"] == [(1, 0, 0)]
    assert np.isclose(entries[1]["qz"], 2.0 * np.pi / 5.0)
    assert np.isclose(
        entries[2]["qr"],
        (2.0 * np.pi / 4.0) * np.sqrt((4.0 / 3.0) * 3.0),
    )

    fallback_entries = geometry_q_group_manager.build_geometry_q_group_entries(
        [
            np.asarray(
                [
                    [4.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0],
                ],
                dtype=float,
            )
        ],
        peak_table_lattice=None,
        primary_a=7.0,
        primary_c=9.0,
    )
    assert fallback_entries[0]["source_label"] == "primary"
    assert np.isclose(
        fallback_entries[0]["qr"],
        (2.0 * np.pi / 7.0) * np.sqrt(4.0 / 3.0),
    )


def test_geometry_q_group_manager_builds_simulated_peaks_from_hit_tables() -> None:
    peaks = geometry_q_group_manager.build_geometry_fit_simulated_peaks(
        [
            np.asarray(
                [
                    [10.0, 1.2, 2.8, 0.0, 1.0, 0.0, 0.0],
                    [8.0, 3.0, 4.0, 0.0, 1.0, 0.0, 1.0],
                ],
                dtype=float,
            )
        ],
        image_shape=(20, 30),
        native_sim_to_display_coords=lambda col, row, shape: (
            col + float(shape[1]),
            row + float(shape[0]),
        ),
        peak_table_lattice=[(3.0, 5.0, "primary")],
        primary_a=7.0,
        primary_c=9.0,
        default_source_label=None,
        round_pixel_centers=False,
    )

    assert len(peaks) == 2
    assert peaks[0]["hkl"] == (1, 0, 0)
    assert peaks[0]["sim_col"] == 31.2
    assert peaks[0]["sim_row"] == 22.8
    assert peaks[0]["source_label"] == "primary"
    assert peaks[0]["source_peak_index"] == 0
    assert peaks[1]["q_group_key"] == ("q_group", "primary", 1, 1)
    assert peaks[1]["source_row_index"] == 1

    fallback_peaks = geometry_q_group_manager.build_geometry_fit_simulated_peaks(
        [
            np.asarray(
                [
                    [4.0, 1.2, 2.8, 0.0, 1.0, 1.0, 0.0],
                ],
                dtype=float,
            )
        ],
        image_shape=(20, 30),
        native_sim_to_display_coords=lambda col, row, shape: (col, row),
        peak_table_lattice=None,
        primary_a=4.0,
        primary_c=6.0,
        default_source_label=None,
        round_pixel_centers=True,
    )
    assert fallback_peaks[0]["source_label"] == "table_0"
    assert fallback_peaks[0]["sim_col"] == 1.0
    assert fallback_peaks[0]["sim_row"] == 3.0
    assert fallback_peaks[0]["q_group_key"] == ("q_group", "primary", 3, 0)


def test_geometry_q_group_manager_aggregates_peak_centers_from_max_positions() -> None:
    assert geometry_q_group_manager.geometry_fit_peak_center_from_max_position(
        [9.0, 1.0, 2.0, 4.0, 8.0, 9.0]
    ) == (1.0, 2.0)
    assert geometry_q_group_manager.geometry_fit_peak_center_from_max_position(
        [3.0, 1.0, 2.0, 7.0, 5.0, 6.0]
    ) == (5.0, 6.0)

    peaks = geometry_q_group_manager.aggregate_geometry_fit_peak_centers_from_max_positions(
        [
            [9.0, 1.0, 2.0, 4.0, 8.0, 9.0],
            [3.0, 1.0, 2.0, 7.0, 5.0, 6.0],
            [6.0, np.nan, np.nan, 2.0, 10.0, 12.0],
        ],
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        np.asarray([4.0, -6.0, 7.0], dtype=float),
    )

    assert peaks == [
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0",
            "sim_col": 3.0,
            "sim_row": 4.0,
            "weight": 10.0,
        },
        {
            "hkl": (2, 0, 1),
            "label": "2,0,1",
            "sim_col": 10.0,
            "sim_row": 12.0,
            "weight": 7.0,
        },
    ]


def test_geometry_q_group_manager_simulate_geometry_fit_helpers() -> None:
    captured = {}

    def _build_mosaic(params):
        captured["params"] = dict(params)
        return {
            "beam_x_array": np.asarray([1.0, 2.0], dtype=float),
            "beam_y_array": np.asarray([3.0, 4.0], dtype=float),
            "theta_array": np.asarray([5.0, 6.0], dtype=float),
            "phi_array": np.asarray([7.0, 8.0], dtype=float),
            "sigma_mosaic_deg": 0.1,
            "gamma_mosaic_deg": 0.2,
            "eta": 0.3,
        }

    def _process_peaks_parallel(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return (
            np.zeros((32, 32), dtype=float),
            [
                np.asarray(
                    [
                        [10.0, 1.2, 2.8, 0.0, 1.0, 0.0, 0.0],
                    ],
                    dtype=float,
                )
            ],
        )

    param_set = {
        "a": 3.0,
        "c": 5.0,
        "lambda": 1.54,
        "corto_detector": 100.0,
        "gamma": 1.0,
        "Gamma": 2.0,
        "chi": 3.0,
        "psi": 4.0,
        "psi_z": 5.0,
        "zs": 6.0,
        "zb": 7.0,
        "n2": "n2",
        "debye_x": 0.1,
        "debye_y": 0.2,
        "center": (11.0, 12.0),
        "theta_initial": 8.0,
        "cor_angle": 9.0,
        "optics_mode": 2,
    }
    miller_array = np.asarray([[1.0, 0.0, 0.0]], dtype=float)
    intensity_array = np.asarray([5.0], dtype=float)

    hit_tables = geometry_q_group_manager.simulate_geometry_fit_hit_tables(
        miller_array,
        intensity_array,
        32,
        param_set,
        build_geometry_fit_central_mosaic_params=_build_mosaic,
        process_peaks_parallel=_process_peaks_parallel,
        default_solve_q_steps=123,
        default_solve_q_rel_tol=2.5e-4,
        default_solve_q_mode=1,
    )

    assert len(hit_tables) == 1
    np.testing.assert_allclose(captured["args"][5], [1.54, 1.54])
    np.testing.assert_allclose(captured["args"][16], [1.0, 2.0])
    np.testing.assert_allclose(captured["args"][23], [1.54, 1.54])
    assert captured["args"][15] == "n2"
    assert captured["kwargs"]["optics_mode"] == 2
    assert captured["kwargs"]["solve_q_steps"] == 123
    assert captured["kwargs"]["solve_q_mode"] == 1

    centers = geometry_q_group_manager.simulate_geometry_fit_peak_centers(
        miller_array,
        intensity_array,
        32,
        param_set,
        build_geometry_fit_central_mosaic_params=_build_mosaic,
        process_peaks_parallel=_process_peaks_parallel,
        hit_tables_to_max_positions=lambda _tables: [
            [9.0, 1.0, 2.0, 4.0, 6.0, 7.0]
        ],
        default_solve_q_steps=123,
        default_solve_q_rel_tol=2.5e-4,
        default_solve_q_mode=1,
    )
    assert centers == [
        {
            "hkl": (1, 0, 0),
            "label": "1,0,0",
            "sim_col": 1.0,
            "sim_row": 2.0,
            "weight": 5.0,
        }
    ]

    preview_peaks = geometry_q_group_manager.simulate_geometry_fit_preview_style_peaks(
        miller_array,
        intensity_array,
        32,
        param_set,
        build_geometry_fit_central_mosaic_params=_build_mosaic,
        process_peaks_parallel=_process_peaks_parallel,
        native_sim_to_display_coords=lambda col, row, shape: (
            col + float(shape[1]),
            row + float(shape[0]),
        ),
        peak_table_lattice=[(3.0, 5.0, "primary")],
        primary_a=7.0,
        primary_c=9.0,
        default_source_label=None,
        round_pixel_centers=False,
        default_solve_q_steps=123,
        default_solve_q_rel_tol=2.5e-4,
        default_solve_q_mode=1,
    )

    assert preview_peaks[0]["sim_col"] == 33.2
    assert preview_peaks[0]["sim_row"] == 34.8
    assert preview_peaks[0]["source_label"] == "primary"
    assert preview_peaks[0]["q_group_key"] == ("q_group", "primary", 1, 0)


def test_geometry_q_group_manager_runtime_simulation_callback_bundle_uses_live_values(
    monkeypatch,
) -> None:
    calls = []
    live = {
        "peak_table_lattice": [(3.0, 5.0, "primary")],
        "primary_a": 7.0,
        "primary_c": 9.0,
    }

    monkeypatch.setattr(
        geometry_q_group_manager,
        "simulate_geometry_fit_hit_tables",
        lambda *args, **kwargs: calls.append(("hit_tables", args, kwargs)) or ["hit"],
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "simulate_geometry_fit_peak_centers",
        lambda *args, **kwargs: calls.append(("peak_centers", args, kwargs))
        or [{"hkl": (1, 0, 0)}],
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "simulate_geometry_fit_preview_style_peaks",
        lambda *args, **kwargs: calls.append(("preview_style", args, kwargs))
        or [{"hkl": (1, 0, 1)}],
    )

    bundle = geometry_q_group_manager.make_runtime_geometry_fit_simulation_callbacks(
        build_geometry_fit_central_mosaic_params="build-mosaic",
        process_peaks_parallel="process-parallel",
        hit_tables_to_max_positions="to-maxpos",
        native_sim_to_display_coords="native-to-display",
        peak_table_lattice_factory=lambda: live["peak_table_lattice"],
        primary_a_factory=lambda: live["primary_a"],
        primary_c_factory=lambda: live["primary_c"],
        default_source_label=None,
        round_pixel_centers=False,
        default_solve_q_steps=123,
        default_solve_q_rel_tol=2.5e-4,
        default_solve_q_mode=1,
    )

    assert bundle.simulate_hit_tables("miller", "intensity", 32, {"a": 1.0}) == ["hit"]
    assert bundle.simulate_peak_centers("miller", "intensity", 32, {"a": 1.0}) == [
        {"hkl": (1, 0, 0)}
    ]
    assert bundle.simulate_preview_style_peaks(
        "miller",
        "intensity",
        32,
        {"a": 1.0},
    ) == [{"hkl": (1, 0, 1)}]

    assert calls[0] == (
        "hit_tables",
        ("miller", "intensity", 32, {"a": 1.0}),
        {
            "build_geometry_fit_central_mosaic_params": "build-mosaic",
            "process_peaks_parallel": "process-parallel",
            "default_solve_q_steps": 123,
            "default_solve_q_rel_tol": 2.5e-4,
            "default_solve_q_mode": 1,
        },
    )
    assert calls[1] == (
        "peak_centers",
        ("miller", "intensity", 32, {"a": 1.0}),
        {
            "build_geometry_fit_central_mosaic_params": "build-mosaic",
            "process_peaks_parallel": "process-parallel",
            "hit_tables_to_max_positions": "to-maxpos",
            "default_solve_q_steps": 123,
            "default_solve_q_rel_tol": 2.5e-4,
            "default_solve_q_mode": 1,
        },
    )
    assert calls[2] == (
        "preview_style",
        ("miller", "intensity", 32, {"a": 1.0}),
        {
            "build_geometry_fit_central_mosaic_params": "build-mosaic",
            "process_peaks_parallel": "process-parallel",
            "native_sim_to_display_coords": "native-to-display",
            "peak_table_lattice": [(3.0, 5.0, "primary")],
            "primary_a": 7.0,
            "primary_c": 9.0,
            "default_source_label": None,
            "round_pixel_centers": False,
            "default_solve_q_steps": 123,
            "default_solve_q_rel_tol": 2.5e-4,
            "default_solve_q_mode": 1,
        },
    )

    live["peak_table_lattice"] = None
    live["primary_a"] = 11.0
    live["primary_c"] = 13.0
    bundle.simulate_preview_style_peaks("miller", "intensity", 64, {"c": 2.0})
    assert calls[3] == (
        "preview_style",
        ("miller", "intensity", 64, {"c": 2.0}),
        {
            "build_geometry_fit_central_mosaic_params": "build-mosaic",
            "process_peaks_parallel": "process-parallel",
            "native_sim_to_display_coords": "native-to-display",
            "peak_table_lattice": None,
            "primary_a": 11.0,
            "primary_c": 13.0,
            "default_source_label": None,
            "round_pixel_centers": False,
            "default_solve_q_steps": 123,
            "default_solve_q_rel_tol": 2.5e-4,
            "default_solve_q_mode": 1,
        },
    )


def test_geometry_q_group_manager_runtime_value_callback_bundle_uses_live_values(
    monkeypatch,
) -> None:
    calls = []
    runtime_state = state.SimulationRuntimeState(
        peak_records=[
            {
                "display_col": 1.5,
                "display_row": 2.5,
                "hkl_raw": [1, 0, 0],
                "intensity": 7.0,
                "source_label": "primary",
                "source_table_index": 0,
                "source_row_index": 1,
                "q_group_key": ("q_group", "primary", 1, 0),
            },
            {
                "display_col": np.nan,
                "display_row": 9.0,
            },
        ],
        stored_max_positions_local=["maxpos"],
        stored_sim_image=np.zeros((20, 30), dtype=float),
        stored_peak_table_lattice=[(3.0, 5.0, "primary")],
    )
    preview_state = state.GeometryPreviewState(
        excluded_q_groups={("q_group", "primary", 1, 0)}
    )
    q_group_state = state.GeometryQGroupState(
        cached_entries=[_entry(("q_group", "primary", 1, 0), peak_count=2, total_intensity=10.0)]
    )
    live = {
        "primary_a": 7.0,
        "primary_c": 9.0,
        "var_names": ["gamma"],
        "image_size": 64,
    }

    monkeypatch.setattr(
        geometry_q_group_manager,
        "build_geometry_fit_simulated_peaks",
        lambda *args, **kwargs: calls.append(("build_peaks", args, kwargs))
        or [
            {
                "sim_col": 4.0,
                "sim_row": 5.0,
                "weight": 6.0,
                "hkl": (1, 0, 0),
                "label": "1,0,0",
                "q_group_key": ("q_group", "primary", 1, 0),
                "source_table_index": 0,
                "source_row_index": 1,
            }
        ],
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "filter_geometry_fit_simulated_peaks",
        lambda *args, **kwargs: calls.append(("filter_peaks", args, kwargs))
        or ([{"filtered": True}], 1, 2),
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "collapse_geometry_fit_simulated_peaks",
        lambda *args, **kwargs: calls.append(("collapse_peaks", args, kwargs))
        or ([{"collapsed": True}], 3),
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "build_geometry_q_group_entries",
        lambda *args, **kwargs: calls.append(("build_entries", args, kwargs))
        or [{"entry": True}],
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "build_geometry_q_group_export_rows",
        lambda *args, **kwargs: calls.append(("export_rows", args, kwargs))
        or [{"row": True}],
    )
    monkeypatch.setattr(
        geometry_q_group_manager.gui_controllers,
        "clone_geometry_q_group_entries",
        lambda entries: calls.append(("clone_entries", (entries,), {}))
        or [{"cloned": True}],
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "build_geometry_q_group_window_status_text",
        lambda *args, **kwargs: calls.append(("window_status", args, kwargs))
        or "status-text",
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "build_geometry_preview_exclude_button_label",
        lambda *args, **kwargs: calls.append(("button_label", args, kwargs))
        or "button-label",
    )

    bundle = geometry_q_group_manager.make_runtime_geometry_q_group_value_callbacks(
        simulation_runtime_state=runtime_state,
        preview_state=preview_state,
        q_group_state=q_group_state,
        fit_config={"geometry": {"auto_match": {"min_matches": 4}}},
        current_geometry_fit_var_names_factory=lambda: live["var_names"],
        primary_a_factory=lambda: live["primary_a"],
        primary_c_factory=lambda: live["primary_c"],
        image_size_factory=lambda: live["image_size"],
        native_sim_to_display_coords="native-to-display",
    )

    cached_preview_peaks = bundle.build_live_preview_simulated_peaks_from_cache()
    assert len(cached_preview_peaks) == 1
    assert cached_preview_peaks[0]["sim_col"] == 4.0
    assert cached_preview_peaks[0]["sim_row"] == 5.0
    assert cached_preview_peaks[0]["weight"] == 6.0
    assert cached_preview_peaks[0]["hkl"] == (1, 0, 0)
    assert cached_preview_peaks[0]["label"] == "1,0,0"
    assert cached_preview_peaks[0]["q_group_key"] == ("q_group", "primary", 1, 0)
    assert bundle.filter_simulated_peaks([{"seed": True}]) == ([{"filtered": True}], 1, 2)
    assert bundle.collapse_simulated_peaks(
        [{"seed": True}],
        merge_radius_px=4.5,
    ) == ([{"collapsed": True}], 3)
    assert bundle.build_entries_snapshot() == [{"entry": True}]
    assert bundle.clone_entries([{"a": 1}]) == [{"cloned": True}]
    assert bundle.listed_entries() == [{"cloned": True}]
    assert bundle.listed_keys() == {("q_group", "primary", 1, 0)}
    assert bundle.key_from_jsonable(["q_group", "primary", 1, 0]) == (
        "q_group",
        "primary",
        1,
        0,
    )
    assert bundle.export_rows() == [{"row": True}]
    assert "primary" in bundle.format_line(q_group_state.cached_entries[0])
    assert bundle.current_min_matches() == 4
    assert bundle.excluded_count() == 1
    assert bundle.build_window_status() == "status-text"
    assert bundle.build_preview_exclude_button_label() == "button-label"
    preview_entry = {
        "hkl": (1, 0, 0),
        "source_label": "primary",
        "source_table_index": 0,
        "source_row_index": 1,
    }
    preview_state.excluded_keys = {
        ("peak", "primary", 0, 1, 1, 0, 0)
    }
    assert bundle.live_preview_match_key(preview_entry) == (
        "peak",
        "primary",
        0,
        1,
        1,
        0,
        0,
    )
    assert bundle.live_preview_match_hkl(preview_entry) == (1, 0, 0)
    assert bundle.live_preview_match_is_excluded(preview_entry) is True
    assert bundle.filter_live_preview_matches([preview_entry, {"hkl": (2, 0, 0)}]) == (
        [{"hkl": (2, 0, 0)}],
        1,
    )
    filtered_pairs, preview_stats, excluded_total = (
        bundle.apply_live_preview_match_exclusions(
            [preview_entry, {"hkl": (2, 0, 0), "distance_px": 3.0}],
            {"search_radius_px": 18.0},
        )
    )
    assert filtered_pairs == [{"hkl": (2, 0, 0), "distance_px": 3.0}]
    assert preview_stats["excluded_count"] == 1
    assert excluded_total == 1

    assert (
        "build_peaks",
        (["maxpos"],),
        {
            "image_shape": (20, 30),
            "native_sim_to_display_coords": "native-to-display",
            "peak_table_lattice": [(3.0, 5.0, "primary")],
            "primary_a": 7.0,
            "primary_c": 9.0,
            "default_source_label": "primary",
            "round_pixel_centers": True,
            "allow_nominal_hkl_indices": False,
        },
    ) in calls
    assert (
        "filter_peaks",
        ([{"seed": True}],),
        {
            "listed_keys": {("q_group", "primary", 1, 0)},
            "excluded_q_groups": {("q_group", "primary", 1, 0)},
        },
    ) in calls
    assert (
        "collapse_peaks",
        ([{"seed": True}],),
        {"merge_radius_px": 4.5},
    ) in calls
    assert (
        "build_entries",
        (["maxpos"],),
        {
            "peak_table_lattice": [(3.0, 5.0, "primary")],
            "primary_a": 7.0,
            "primary_c": 9.0,
            "allow_nominal_hkl_indices": False,
        },
    ) in calls
    assert (
        "clone_entries",
        ([{"a": 1}],),
        {},
    ) in calls
    assert (
        "export_rows",
        (),
        {
            "preview_state": preview_state,
            "q_group_state": q_group_state,
            "entries": None,
        },
    ) in calls
    assert (
        "window_status",
        (),
        {
            "preview_state": preview_state,
            "q_group_state": q_group_state,
            "fit_config": {"geometry": {"auto_match": {"min_matches": 4}}},
            "current_geometry_fit_var_names": ["gamma"],
            "entries": None,
        },
    ) in calls
    assert (
        "button_label",
        (),
        {
            "preview_state": preview_state,
            "q_group_state": q_group_state,
            "entries": None,
        },
    ) in calls

    runtime_state.stored_max_positions_local = None
    runtime_state.stored_sim_image = None
    runtime_state.stored_peak_table_lattice = None
    live["primary_a"] = 11.0
    live["primary_c"] = 13.0
    live["image_size"] = 48
    cached_preview_peaks = bundle.build_live_preview_simulated_peaks_from_cache()
    assert len(cached_preview_peaks) == 1
    assert cached_preview_peaks[0]["sim_col"] == 1.5
    assert cached_preview_peaks[0]["sim_row"] == 2.5
    assert cached_preview_peaks[0]["weight"] == 7.0
    assert cached_preview_peaks[0]["hkl"] == (1, 0, 0)
    assert cached_preview_peaks[0]["label"] == "1,0,0"
    assert cached_preview_peaks[0]["q_group_key"] == ("q_group", "primary", 1, 0)
    runtime_state.peak_records[0].pop("q_group_key")
    cached_preview_peaks = bundle.build_live_preview_simulated_peaks_from_cache()
    assert cached_preview_peaks[0]["q_group_key"] == ("q_group", "primary", 1, 0)


def test_geometry_q_group_manager_runtime_value_callback_bundle_uses_nominal_cache_grouping() -> None:
    runtime_state = state.SimulationRuntimeState(
        stored_max_positions_local=[
            np.asarray(
                [[12.0, 10.2, 20.8, 0.0, 1.0, 0.0, 1.29]],
                dtype=float,
            )
        ],
        stored_sim_image=np.zeros((32, 32), dtype=float),
        stored_peak_table_lattice=[(3.0, 5.0, "primary")],
        stored_primary_intersection_cache=[
            np.asarray(
                [[1.1, 1.2, 10.2, 20.8, 12.0, 0.0, 1.0, 0.0, 1.29]],
                dtype=float,
            )
        ],
    )
    preview_state = state.GeometryPreviewState()
    q_group_state = state.GeometryQGroupState()

    bundle = geometry_q_group_manager.make_runtime_geometry_q_group_value_callbacks(
        simulation_runtime_state=runtime_state,
        preview_state=preview_state,
        q_group_state=q_group_state,
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        primary_a_factory=lambda: 3.0,
        primary_c_factory=lambda: 5.0,
        image_size_factory=lambda: 32,
        native_sim_to_display_coords=lambda col, row, _shape: (col, row),
    )

    cached_preview_peaks = bundle.build_live_preview_simulated_peaks_from_cache()
    entries = bundle.build_entries_snapshot()

    assert len(cached_preview_peaks) == 1
    assert cached_preview_peaks[0]["hkl"] == (1, 0, 1)
    assert cached_preview_peaks[0]["q_group_key"] == ("q_group", "primary", 1, 1)
    assert cached_preview_peaks[0]["q_group_nominal_hkl"] is True
    assert len(entries) == 1
    assert entries[0]["key"] == ("q_group", "primary", 1, 1)


def test_geometry_q_group_manager_live_preview_exclusion_helpers() -> None:
    excluded_entry = {
        "hkl": (1, 0, 0),
        "source_label": "primary",
        "source_table_index": 2,
        "source_row_index": 3,
        "distance_px": 9.0,
        "confidence": 0.1,
    }
    indexed_entry = {
        "hkl": (2, 0, 1),
        "source_peak_index": 5,
        "distance_px": 2.0,
        "confidence": 0.8,
    }
    coord_entry = {
        "hkl": (3, 0, 2),
        "sim_x": 11.24,
        "sim_y": 8.66,
        "distance_px": 4.0,
        "confidence": 0.4,
    }
    excluded_key = geometry_q_group_manager.live_geometry_preview_match_key(
        excluded_entry
    )
    preview_state = state.GeometryPreviewState(excluded_keys={excluded_key})
    preview_state.overlay.pairs = [dict(excluded_entry)]

    assert excluded_key == ("peak", "primary", 2, 3, 1, 0, 0)
    assert geometry_q_group_manager.live_geometry_preview_match_key(indexed_entry) == (
        "peak_index",
        5,
        2,
        0,
        1,
    )
    assert geometry_q_group_manager.live_geometry_preview_match_key(coord_entry) == (
        "hkl_coord",
        3,
        0,
        2,
        11.2,
        8.7,
    )
    assert geometry_q_group_manager.live_geometry_preview_match_hkl(coord_entry) == (
        3,
        0,
        2,
    )
    assert (
        geometry_q_group_manager.live_geometry_preview_match_is_excluded(
            preview_state,
            excluded_entry,
        )
        is True
    )

    filtered, excluded_count = (
        geometry_q_group_manager.filter_live_geometry_preview_matches(
            preview_state,
            [excluded_entry, indexed_entry, coord_entry, "bad"],
        )
    )
    assert filtered == [indexed_entry, coord_entry]
    assert excluded_count == 1

    filtered_pairs, stats, excluded_total = (
        geometry_q_group_manager.apply_live_geometry_preview_match_exclusions(
            preview_state,
            [excluded_entry, indexed_entry, coord_entry],
            {"search_radius_px": 18.0},
        )
    )
    assert filtered_pairs == [indexed_entry, coord_entry]
    assert excluded_total == 1
    assert stats["excluded_count"] == 1
    assert stats["matched_count"] == 2
    assert stats["matched_after_exclusions"] == 2
    assert np.isclose(stats["mean_match_distance_px"], 3.0)
    assert np.isclose(stats["p90_match_distance_px"], 3.8)
    assert np.isclose(stats["median_match_confidence"], 0.6)


def test_geometry_q_group_manager_filters_simulated_peaks_by_listed_keys_and_exclusions() -> None:
    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "primary", 1, 1)
    filtered, excluded_count, total_groups = (
        geometry_q_group_manager.filter_geometry_fit_simulated_peaks(
            [
                {"hkl": (1, 0, 0), "source_label": "primary", "av": 3.0, "cv": 5.0},
                {"hkl": (1, 0, 1), "source_label": "primary", "av": 3.0, "cv": 5.0},
                {"hkl": (2, 0, 0), "source_label": "secondary", "av": 4.0, "cv": 6.0},
                {"hkl": "bad"},
            ],
            listed_keys=[key1, key2],
            excluded_q_groups={key2},
        )
    )

    assert [entry["q_group_key"] for entry in filtered] == [key1]
    assert excluded_count == 3
    assert total_groups == 2


def test_geometry_q_group_manager_collapses_degenerate_simulated_peaks() -> None:
    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "secondary", 3, 0)
    collapsed, collapsed_count = (
        geometry_q_group_manager.collapse_geometry_fit_simulated_peaks(
            [
                {
                    "q_group_key": key1,
                    "hkl": (1, 0, 0),
                    "label": "1,0,0",
                    "sim_col": 10.0,
                    "sim_row": 10.0,
                    "weight": 2.0,
                    "source_peak_index": 5,
                },
                {
                    "q_group_key": key1,
                    "hkl": (0, 1, 0),
                    "label": "0,1,0",
                    "sim_col": 11.0,
                    "sim_row": 12.0,
                    "weight": 3.0,
                    "source_peak_index": 1,
                },
                {
                    "q_group_key": key1,
                    "hkl": (1, 0, 1),
                    "label": "1,0,1",
                    "sim_col": 30.0,
                    "sim_row": 30.0,
                    "weight": 0.0,
                    "source_peak_index": 7,
                },
                {
                    "q_group_key": key2,
                    "hkl": (1, 1, 0),
                    "label": "1,1,0",
                    "sim_col": 50.0,
                    "sim_row": 50.0,
                    "weight": 4.0,
                    "source_peak_index": 0,
                },
            ],
            merge_radius_px=3.0,
        )
    )

    assert collapsed_count == 1
    assert [entry["q_group_key"] for entry in collapsed] == [key1, key1, key2]
    assert collapsed[0]["source_peak_index"] == 1
    assert collapsed[0]["weight"] == 5.0
    assert collapsed[0]["degenerate_count"] == 2
    assert collapsed[0]["degenerate_hkls"] == [(1, 0, 0), (0, 1, 0)]
    assert collapsed[1]["weight"] == 1.0
    assert collapsed[1]["degenerate_count"] == 1
    assert collapsed[1]["degenerate_hkls"] == [(1, 0, 1)]
    assert collapsed[2]["weight"] == 4.0


def test_geometry_q_group_manager_formats_lines_and_builds_status_text() -> None:
    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "secondary", 2, 1)
    preview_state = state.GeometryPreviewState(excluded_q_groups={key2})
    q_group_state = state.GeometryQGroupState(
        cached_entries=[
            _entry(key1, peak_count=2, total_intensity=10.0),
            _entry(key2, peak_count=3, total_intensity=20.0, source="secondary"),
        ]
    )

    line = geometry_q_group_manager.format_geometry_q_group_line(
        q_group_state.cached_entries[0]
    )
    status = geometry_q_group_manager.build_geometry_q_group_window_status_text(
        preview_state=preview_state,
        q_group_state=q_group_state,
        fit_config={"geometry": {"auto_match": {"min_matches": 4}}},
        current_geometry_fit_var_names=["gamma"],
    )

    assert "primary" in line
    assert "HKL=" in line
    assert ", ..." in line
    assert (
        geometry_q_group_manager.current_geometry_auto_match_min_matches(
            {},
            ["gamma", "chi"],
        )
        == 6
    )
    assert (
        geometry_q_group_manager.geometry_q_group_excluded_count(
            preview_state,
            q_group_state,
        )
        == 1
    )
    assert (
        geometry_q_group_manager.build_geometry_preview_exclude_button_label(
            preview_state=preview_state,
            q_group_state=q_group_state,
        )
        == "Choose Active Qr/Qz Groups (1 off)"
    )
    assert "Included Qr/Qz groups: 1/2" in status
    assert "Selected peaks: 2/5" in status
    assert "Need >= 4  short 2" in status
    assert "Intensity=10.000/30.000" in status


def test_geometry_q_group_manager_live_preview_config_overlay_and_render_helpers() -> None:
    preview_cfg = geometry_q_group_manager.build_live_geometry_preview_auto_match_config(
        {
            "geometry": {
                "auto_match": {
                    "search_radius_px": 20.0,
                    "max_display_markers": 1,
                    "max_p90_distance_px": 30.0,
                    "max_mean_distance_px": 10.0,
                }
            }
        }
    )
    empty_overlay = geometry_q_group_manager.build_empty_live_geometry_preview_overlay_state(
        signature="empty-sig",
        min_matches=3,
        max_display_markers=5,
        q_group_total=4,
        q_group_excluded=1,
        excluded_q_peaks=2,
    )
    overlay_state = geometry_q_group_manager.build_live_geometry_preview_overlay_state(
        signature="sig",
        matched_pairs=[
            {"sim_x": 1.0, "sim_y": 2.0, "x": 11.0, "y": 12.0},
            {"sim_x": 3.0, "sim_y": 4.0, "x": 13.0, "y": 14.0},
        ],
        match_stats={
            "simulated_count": 4,
            "search_radius_px": 18.0,
            "p90_match_distance_px": 35.0,
            "mean_match_distance_px": 11.0,
        },
        preview_auto_match_cfg=preview_cfg,
        auto_match_attempts=[{"radius": 18.0}],
        min_matches=2,
        q_group_total=3,
        q_group_excluded=1,
        excluded_q_peaks=2,
        collapsed_degenerate_peaks=1,
    )

    assert preview_cfg["relax_on_low_matches"] is False
    assert preview_cfg["context_margin_px"] == 192.0
    assert empty_overlay["pairs"] == []
    assert empty_overlay["q_group_total"] == 4
    assert overlay_state["quality_fail"] is True
    assert overlay_state["max_display_markers"] == 1
    assert overlay_state["collapsed_degenerate_peaks"] == 1
    assert overlay_state["auto_match_attempts"] == [{"radius": 18.0}]

    preview_state = state.GeometryPreviewState()
    preview_state.overlay.pairs = list(overlay_state["pairs"])
    preview_state.overlay.simulated_count = int(overlay_state["simulated_count"])
    preview_state.overlay.min_matches = int(overlay_state["min_matches"])
    preview_state.overlay.best_radius = float(overlay_state["best_radius"])
    preview_state.overlay.mean_dist = float(overlay_state["mean_dist"])
    preview_state.overlay.p90_dist = float(overlay_state["p90_dist"])
    preview_state.overlay.quality_fail = bool(overlay_state["quality_fail"])
    preview_state.overlay.max_display_markers = int(overlay_state["max_display_markers"])
    preview_state.overlay.q_group_total = int(overlay_state["q_group_total"])
    preview_state.overlay.q_group_excluded = int(overlay_state["q_group_excluded"])
    preview_state.overlay.collapsed_degenerate_peaks = int(
        overlay_state["collapsed_degenerate_peaks"]
    )

    draw_calls = []
    status_messages = []
    rendered = geometry_q_group_manager.render_live_geometry_preview_overlay_state(
        preview_state=preview_state,
        draw_live_geometry_preview_overlay=lambda pairs, *, max_display_markers: draw_calls.append(
            (list(pairs), max_display_markers)
        ),
        filter_live_preview_matches=lambda pairs: (list(pairs), 1),
        set_status_text=lambda text: status_messages.append(text),
        update_status=True,
    )
    quiet_render = geometry_q_group_manager.render_live_geometry_preview_overlay_state(
        preview_state=preview_state,
        draw_live_geometry_preview_overlay=lambda pairs, *, max_display_markers: draw_calls.append(
            ("quiet", list(pairs), max_display_markers)
        ),
        filter_live_preview_matches=lambda pairs: ([], 0),
        set_status_text=lambda text: status_messages.append(f"quiet:{text}"),
        update_status=False,
    )

    assert rendered is True
    assert quiet_render is True
    assert draw_calls[0][1] == 1
    assert draw_calls[1][0] == "quiet"
    assert "Live auto-match preview: 2/4 active peaks" in status_messages[0]
    assert "Excluded=1." in status_messages[0]
    assert "Qr/Qz groups on=2/3." in status_messages[0]
    assert "Degenerate collapsed=1." in status_messages[0]
    assert "Geometry fit would stop on the quality gate." in status_messages[0]
    assert "Showing 1/2 overlays." in status_messages[0]
    assert len(status_messages) == 1


def test_geometry_q_group_manager_runtime_live_preview_render_helpers(
    monkeypatch,
) -> None:
    draw_calls = []
    status_messages = []
    preview_state = state.GeometryPreviewState()
    preview_state.overlay.pairs = [{"hkl": (1, 0, 0), "x": 1.0, "y": 2.0}]
    preview_state.overlay.simulated_count = 1
    preview_state.overlay.min_matches = 1
    preview_state.overlay.max_display_markers = 7

    monkeypatch.setattr(
        geometry_q_group_manager.gui_overlays,
        "draw_live_geometry_preview_overlay",
        lambda axis, pairs, **kwargs: draw_calls.append((axis, list(pairs), kwargs)),
    )

    bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=preview_state,
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        axis="axis",
        geometry_preview_artists=[],
        draw_idle=lambda: None,
        normalize_hkl_key=lambda value: tuple(value) if isinstance(value, tuple) else None,
        live_preview_match_is_excluded=lambda _entry: False,
        filter_live_preview_matches=lambda pairs: (list(pairs), 0),
        set_status_text=lambda text: status_messages.append(text),
    )

    assert geometry_q_group_manager.runtime_live_geometry_preview_enabled(bindings) is True
    geometry_q_group_manager.draw_runtime_live_geometry_preview_overlay(
        bindings,
        [{"hkl": (2, 0, 0), "x": 3.0, "y": 4.0}],
        max_display_markers=9,
    )
    assert (
        geometry_q_group_manager.render_runtime_live_geometry_preview_state(
            bindings,
            update_status=True,
        )
        is True
    )

    assert draw_calls[0][0] == "axis"
    assert draw_calls[0][1] == [{"hkl": (2, 0, 0), "x": 3.0, "y": 4.0}]
    assert draw_calls[0][2]["max_display_markers"] == 9
    assert draw_calls[1][1] == preview_state.overlay.pairs
    assert draw_calls[1][2]["max_display_markers"] == 7
    assert "Live auto-match preview: 1/1 active peaks" in status_messages[0]


def test_geometry_q_group_manager_runtime_live_preview_simulated_peak_resolution() -> None:
    preferred_cache_calls: list[str] = []
    preferred_sim_calls: list[tuple[tuple[int, ...], tuple[int, ...], int, dict[str, object]]] = []
    preferred_bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        has_cached_hit_tables=True,
        build_live_preview_simulated_peaks_from_cache=lambda: (
            preferred_cache_calls.append("cache")
            or [{"source": "cache"}]
        ),
        simulate_preview_style_peaks=lambda miller, intensities, image_size, params: (
            preferred_sim_calls.append(
                (tuple(miller.shape), tuple(intensities.shape), int(image_size), dict(params))
            )
            or [{"source": "central"}]
        ),
        miller=[[1.0, 0.0, 0.0]],
        intensities=[2.0],
        image_size=48,
        current_geometry_fit_params_factory=lambda: {"omega": 0.5},
    )

    preferred = geometry_q_group_manager.resolve_runtime_live_geometry_preview_simulated_peaks(
        preferred_bindings,
        update_status=True,
    )

    assert preferred == [{"source": "central"}]
    assert preferred_sim_calls == [((1, 3), (1,), 48, {"omega": 0.5})]
    assert preferred_cache_calls == []

    success_bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        has_cached_hit_tables=True,
        build_live_preview_simulated_peaks_from_cache=lambda: [],
        simulate_preview_style_peaks=lambda miller, intensities, image_size, params: [
            {
                "size": int(image_size),
                "miller_shape": tuple(miller.shape),
                "intensity_shape": tuple(intensities.shape),
                "params": dict(params),
            }
        ],
        miller=[[1.0, 0.0, 0.0]],
        intensities=[2.0],
        image_size=128,
        current_geometry_fit_params_factory=lambda: {"omega": 1.5},
    )

    simulated = geometry_q_group_manager.resolve_runtime_live_geometry_preview_simulated_peaks(
        success_bindings,
        update_status=True,
    )

    assert simulated == [
        {
            "size": 128,
            "miller_shape": (1, 3),
            "intensity_shape": (1,),
            "params": {"omega": 1.5},
        }
    ]

    failure_events = []
    failure_bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        has_cached_hit_tables=True,
        build_live_preview_simulated_peaks_from_cache=lambda: [],
        simulate_preview_style_peaks=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("boom")
        ),
        miller=[[1.0, 0.0, 0.0]],
        intensities=[2.0],
        image_size=64,
        current_geometry_fit_params_factory=lambda: {"omega": 2.0},
        clear_geometry_preview_artists=lambda: failure_events.append("clear"),
        set_status_text=lambda text: failure_events.append(text),
    )

    failed = geometry_q_group_manager.resolve_runtime_live_geometry_preview_simulated_peaks(
        failure_bindings,
        update_status=True,
    )

    assert failed is None
    assert failure_events == [
        "clear",
        "Live auto-match preview unavailable: failed to simulate peaks (boom).",
    ]

    empty_events = []
    empty_bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        has_cached_hit_tables=False,
        build_live_preview_simulated_peaks_from_cache=lambda: [],
        clear_geometry_preview_artists=lambda: empty_events.append("clear"),
        set_status_text=lambda text: empty_events.append(text),
    )

    empty = geometry_q_group_manager.resolve_runtime_live_geometry_preview_simulated_peaks(
        empty_bindings,
        update_status=True,
    )

    assert empty is None
    assert empty_events == [
        "clear",
        "Live auto-match preview unavailable: no simulated peaks are available.",
    ]


def test_geometry_q_group_manager_runtime_live_preview_background_resolution() -> None:
    disabled_events = []
    disabled_bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: False,
        refresh_live_geometry_preview=lambda: None,
        clear_geometry_preview_artists=lambda: disabled_events.append("clear"),
        set_status_text=lambda text: disabled_events.append(text),
    )

    assert (
        geometry_q_group_manager.resolve_runtime_live_geometry_preview_background(
            disabled_bindings,
            update_status=True,
        )
        is None
    )
    assert disabled_events == ["clear"]

    caked_events = []
    caked_bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        caked_view_enabled=lambda: True,
        clear_geometry_preview_artists=lambda: caked_events.append("clear"),
        set_status_text=lambda text: caked_events.append(text),
    )

    assert (
        geometry_q_group_manager.resolve_runtime_live_geometry_preview_background(
            caked_bindings,
            update_status=True,
        )
        is None
    )
    assert caked_events == [
        "clear",
        "Live auto-match preview unavailable in 2D caked view.",
    ]

    hidden_events = []
    hidden_bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        caked_view_enabled=lambda: False,
        background_visible=False,
        current_background_display_factory=lambda: None,
        clear_geometry_preview_artists=lambda: hidden_events.append("clear"),
        set_status_text=lambda text: hidden_events.append(text),
    )

    assert (
        geometry_q_group_manager.resolve_runtime_live_geometry_preview_background(
            hidden_bindings,
            update_status=True,
        )
        is None
    )
    assert hidden_events == [
        "clear",
        "Live auto-match preview unavailable: background image is hidden.",
    ]

    success_bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        caked_view_enabled=lambda: False,
        background_visible=True,
        current_background_display_factory=lambda: "background-image",
    )

    assert (
        geometry_q_group_manager.resolve_runtime_live_geometry_preview_background(
            success_bindings,
            update_status=True,
        )
        == "background-image"
    )


def test_geometry_q_group_manager_runtime_live_preview_seed_state_resolution() -> None:
    preview_state = state.GeometryPreviewState()
    no_group_events = []
    no_group_bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=preview_state,
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        filter_simulated_peaks=lambda peaks: ([], 2, 5),
        collapse_simulated_peaks=lambda peaks, *, merge_radius_px=6.0: (
            list(peaks or []),
            int(merge_radius_px),
        ),
        excluded_q_group_count=lambda: 3,
        clear_geometry_preview_artists=lambda: no_group_events.append("clear"),
        set_status_text=lambda text: no_group_events.append(text),
    )

    no_groups = geometry_q_group_manager.resolve_runtime_live_geometry_preview_seed_state(
        no_group_bindings,
        [{"seed": True}],
        preview_auto_match_cfg={"max_display_markers": 11},
        min_matches=4,
        signature=("sig", 1),
        update_status=True,
    )

    assert no_groups is None
    assert no_group_events == [
        "clear",
        "Live auto-match preview unavailable: no Qr/Qz groups are selected.",
    ]
    assert preview_state.overlay.q_group_total == 5
    assert preview_state.overlay.q_group_excluded == 3
    assert preview_state.overlay.excluded_q_peaks == 2
    assert preview_state.overlay.max_display_markers == 11

    collapse_events = []
    collapse_bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        filter_simulated_peaks=lambda peaks: (list(peaks or []), 1, 4),
        collapse_simulated_peaks=lambda peaks, *, merge_radius_px=6.0: (
            collapse_events.append(("collapse", list(peaks), merge_radius_px)),
            ([], 7),
        )[-1],
        clear_geometry_preview_artists=lambda: collapse_events.append("clear"),
        set_status_text=lambda text: collapse_events.append(text),
    )

    collapsed = geometry_q_group_manager.resolve_runtime_live_geometry_preview_seed_state(
        collapse_bindings,
        [{"seed": 1}],
        preview_auto_match_cfg={"search_radius_px": 30.0},
        min_matches=4,
        signature=("sig", 2),
        update_status=True,
    )

    assert collapsed is None
    assert collapse_events == [
        ("collapse", [{"seed": 1}], 6.0),
        "clear",
        "Live auto-match preview unavailable: no geometry-fit seeds remain after Qr/Qz collapse.",
    ]

    success_events = []
    success_bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        filter_simulated_peaks=lambda peaks: (list(peaks or []), 2, 6),
        collapse_simulated_peaks=lambda peaks, *, merge_radius_px=6.0: (
            success_events.append(("collapse", merge_radius_px)),
            ([{"collapsed": True}], 3),
        )[-1],
        set_status_text=lambda text: success_events.append(text),
    )

    success = geometry_q_group_manager.resolve_runtime_live_geometry_preview_seed_state(
        success_bindings,
        [{"seed": 2}],
        preview_auto_match_cfg={"degenerate_merge_radius_px": 4.5},
        min_matches=5,
        signature=("sig", 3),
        update_status=True,
    )

    assert success == ([{"collapsed": True}], 2, 6, 3)
    assert success_events == [("collapse", 4.5)]


def test_geometry_q_group_manager_runtime_live_preview_match_result_application(
    monkeypatch,
) -> None:
    events = []
    preview_state = state.GeometryPreviewState()

    monkeypatch.setattr(
        geometry_q_group_manager.gui_controllers,
        "replace_geometry_preview_overlay_state",
        lambda preview_state_value, overlay_state: events.append(
            ("replace", preview_state_value, overlay_state)
        ),
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "render_runtime_live_geometry_preview_state",
        lambda bindings, *, update_status=True: (
            events.append(("render", bindings, update_status)),
            True,
        )[-1],
    )

    bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=preview_state,
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: None,
        excluded_q_group_count=lambda: 2,
    )

    applied = (
        geometry_q_group_manager.apply_runtime_live_geometry_preview_match_results(
            bindings,
            signature=("sig", 4),
            matched_pairs=[{"x": 1.0, "y": 2.0, "sim_x": 0.5, "sim_y": 1.5}],
            match_stats={
                "simulated_count": 4,
                "search_radius_px": 18.0,
                "mean_match_distance_px": 6.0,
                "p90_match_distance_px": 9.0,
            },
            preview_auto_match_cfg={"max_display_markers": 6},
            auto_match_attempts=[{"radius": 18.0}],
            min_matches=3,
            q_group_total=5,
            excluded_q_peaks=1,
            collapsed_deg_preview=2,
            update_status=False,
        )
    )

    assert applied is True
    assert events[0][0] == "replace"
    assert events[0][1] is preview_state
    assert events[0][2]["signature"] == ("sig", 4)
    assert events[0][2]["q_group_total"] == 5
    assert events[0][2]["q_group_excluded"] == 2
    assert events[0][2]["excluded_q_peaks"] == 1
    assert events[0][2]["collapsed_degenerate_peaks"] == 2
    assert events[0][2]["auto_match_attempts"] == [{"radius": 18.0}]
    assert events[1] == ("render", bindings, False)


def test_refresh_geometry_q_group_window_uses_cached_entries_and_view_helpers(
    monkeypatch,
) -> None:
    calls = {}
    preview_state = state.GeometryPreviewState()
    q_group_state = state.GeometryQGroupState(
        cached_entries=[_entry(("q_group", "primary", 1, 0), peak_count=2, total_intensity=10.0)]
    )

    def _refresh(**kwargs):
        calls["kwargs"] = kwargs
        return True

    monkeypatch.setattr(
        geometry_q_group_manager.gui_views,
        "refresh_geometry_q_group_window",
        _refresh,
    )

    ok = geometry_q_group_manager.refresh_geometry_q_group_window(
        view_state=state.GeometryQGroupViewState(window=object()),
        preview_state=preview_state,
        q_group_state=q_group_state,
        fit_config={"geometry": {"auto_match": {"min_matches": 1}}},
        current_geometry_fit_var_names=["gamma"],
        on_toggle=lambda key, row_var: (key, row_var),
    )

    assert ok is True
    assert calls["kwargs"]["entries"] == q_group_state.cached_entries
    assert calls["kwargs"]["format_line"] is geometry_q_group_manager.format_geometry_q_group_line
    assert "ready" in calls["kwargs"]["status_text"]


def test_geometry_q_group_manager_toggle_and_bulk_enable_update_preview_state() -> None:
    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "secondary", 2, 1)
    preview_state = state.GeometryPreviewState()
    q_group_state = state.GeometryQGroupState(
        cached_entries=[
            _entry(key1, peak_count=2, total_intensity=10.0),
            _entry(key2, peak_count=3, total_intensity=20.0, source="secondary"),
        ]
    )

    assert (
        geometry_q_group_manager.apply_geometry_q_group_checkbox_change(
            preview_state,
            key1,
            _FakeVar(False),
        )
        == "Excluded"
    )
    assert preview_state.excluded_q_groups == {key1}

    assert (
        geometry_q_group_manager.apply_geometry_q_group_checkbox_change(
            preview_state,
            key1,
            _FakeVar(True),
        )
        == "Included"
    )
    assert preview_state.excluded_q_groups == set()

    action, count = geometry_q_group_manager.set_all_geometry_q_groups_enabled(
        preview_state,
        q_group_state,
        enabled=False,
    )
    assert (action, count) == ("Excluded", 2)
    assert preview_state.excluded_q_groups == {key1, key2}

    action, count = geometry_q_group_manager.set_all_geometry_q_groups_enabled(
        preview_state,
        q_group_state,
        enabled=True,
    )
    assert (action, count) == ("Included", 2)
    assert preview_state.excluded_q_groups == set()


def test_geometry_q_group_manager_save_load_helpers_round_trip() -> None:
    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "secondary", 2, 1)
    key3 = ("q_group", "primary", 3, 2)
    preview_state = state.GeometryPreviewState(excluded_q_groups={key2})
    q_group_state = state.GeometryQGroupState(
        cached_entries=[
            _entry(key1, peak_count=2, total_intensity=10.0),
            _entry(key2, peak_count=3, total_intensity=20.0, source="secondary"),
        ]
    )

    export_rows = geometry_q_group_manager.build_geometry_q_group_export_rows(
        preview_state=preview_state,
        q_group_state=q_group_state,
    )
    payload = geometry_q_group_manager.build_geometry_q_group_save_payload(
        export_rows,
        saved_at="2026-03-26T12:00:00",
    )
    saved_state, error = geometry_q_group_manager.load_geometry_q_group_saved_state(
        payload
    )

    assert error is None
    assert payload["included_count"] == 1
    assert export_rows[0]["included"] is True
    assert export_rows[1]["included"] is False
    assert "Qr=" in export_rows[0]["display_label"]
    assert saved_state == {key1: True, key2: False}

    target_preview_state = state.GeometryPreviewState()
    target_q_group_state = state.GeometryQGroupState(
        cached_entries=[
            _entry(key1, peak_count=2, total_intensity=10.0),
            _entry(key2, peak_count=3, total_intensity=20.0, source="secondary"),
            _entry(key3, peak_count=1, total_intensity=5.0),
        ]
    )

    summary, error = geometry_q_group_manager.apply_loaded_geometry_q_group_saved_state(
        preview_state=target_preview_state,
        q_group_state=target_q_group_state,
        saved_state=saved_state,
    )

    assert error is None
    assert summary == {
        "matched_total": 2,
        "included_total": 1,
        "current_only": 1,
        "saved_only": 0,
    }
    assert target_preview_state.excluded_q_groups == {key2, key3}


def test_geometry_q_group_manager_load_helpers_report_validation_errors() -> None:
    saved_state, error = geometry_q_group_manager.load_geometry_q_group_saved_state("bad")
    assert saved_state is None
    assert error == "Invalid Qr/Qz peak list file: expected a JSON object."

    summary, error = geometry_q_group_manager.apply_loaded_geometry_q_group_saved_state(
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        saved_state={},
    )
    assert summary is None
    assert error == "Loaded Qr/Qz peak list does not contain any valid rows."


def test_geometry_q_group_manager_checkbox_side_effects_update_status() -> None:
    key1 = ("q_group", "primary", 1, 0)
    preview_state = state.GeometryPreviewState()
    events = []

    changed = (
        geometry_q_group_manager.apply_geometry_q_group_checkbox_change_with_side_effects(
            preview_state=preview_state,
            group_key=key1,
            row_var=_FakeVar(False),
            invalidate_geometry_manual_pick_cache=lambda: events.append("invalidate"),
            update_geometry_preview_exclude_button_label=lambda: events.append("label"),
            update_geometry_q_group_window_status=lambda: events.append("status"),
            live_geometry_preview_enabled=lambda: False,
            refresh_live_geometry_preview=lambda: events.append("refresh_live"),
            set_status_text=lambda text: events.append(text),
        )
    )

    assert changed is True
    assert preview_state.excluded_q_groups == {key1}
    assert events == [
        "invalidate",
        "label",
        "status",
        "Excluded one Qr/Qz group for geometry fitting.",
    ]


def test_geometry_q_group_manager_bulk_enable_side_effects_cover_empty_and_live_refresh() -> None:
    preview_state = state.GeometryPreviewState()
    empty_state = state.GeometryQGroupState()
    empty_messages = []

    changed = (
        geometry_q_group_manager.set_all_geometry_q_groups_enabled_with_side_effects(
            preview_state=preview_state,
            q_group_state=empty_state,
            enabled=False,
            invalidate_geometry_manual_pick_cache=lambda: empty_messages.append(
                "invalidate"
            ),
            update_geometry_preview_exclude_button_label=lambda: empty_messages.append(
                "label"
            ),
            refresh_geometry_q_group_window=lambda: empty_messages.append("refresh"),
            live_geometry_preview_enabled=lambda: False,
            refresh_live_geometry_preview=lambda: empty_messages.append("live"),
            set_status_text=lambda text: empty_messages.append(text),
        )
    )

    assert changed is False
    assert empty_messages == [
        'No listed Qr/Qz groups are available. Press "Update Listed Peaks" first.'
    ]

    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "secondary", 2, 1)
    q_group_state = state.GeometryQGroupState(
        cached_entries=[
            _entry(key1, peak_count=2, total_intensity=10.0),
            _entry(key2, peak_count=3, total_intensity=20.0, source="secondary"),
        ]
    )
    events = []

    changed = (
        geometry_q_group_manager.set_all_geometry_q_groups_enabled_with_side_effects(
            preview_state=preview_state,
            q_group_state=q_group_state,
            enabled=False,
            invalidate_geometry_manual_pick_cache=lambda: events.append("invalidate"),
            update_geometry_preview_exclude_button_label=lambda: events.append("label"),
            refresh_geometry_q_group_window=lambda: events.append("refresh"),
            live_geometry_preview_enabled=lambda: True,
            refresh_live_geometry_preview=lambda: events.append("live"),
            set_status_text=lambda text: events.append(text),
        )
    )

    assert changed is True
    assert preview_state.excluded_q_groups == {key1, key2}
    assert events == ["invalidate", "label", "refresh", "live"]


def test_geometry_q_group_manager_request_update_side_effects_marks_refresh() -> None:
    q_group_state = state.GeometryQGroupState()
    events = []

    geometry_q_group_manager.request_geometry_q_group_window_update_with_side_effects(
        q_group_state=q_group_state,
        clear_last_simulation_signature=lambda: events.append("clear_signature"),
        invalidate_geometry_manual_pick_cache=lambda: events.append("invalidate"),
        set_status_text=lambda text: events.append(text),
        schedule_update=lambda: events.append("schedule"),
    )

    assert q_group_state.refresh_requested is True
    assert events == [
        "clear_signature",
        "invalidate",
        "Updating listed Qr/Qz peaks from the current simulation...",
        "schedule",
    ]


def test_geometry_q_group_manager_snapshot_replace_side_effects_trim_exclusions() -> None:
    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "secondary", 2, 1)
    stale_key = ("q_group", "stale", 9, 9)
    preview_state = state.GeometryPreviewState(excluded_q_groups={key1, stale_key})
    q_group_state = state.GeometryQGroupState()
    events = []

    entries = (
        geometry_q_group_manager.replace_geometry_q_group_entries_snapshot_with_side_effects(
            preview_state=preview_state,
            q_group_state=q_group_state,
            entries=[
                _entry(key1, peak_count=2, total_intensity=10.0),
                _entry(key2, peak_count=3, total_intensity=20.0, source="secondary"),
            ],
            invalidate_geometry_manual_pick_cache=lambda: events.append("invalidate"),
            update_geometry_preview_exclude_button_label=lambda: events.append(
                "label"
            ),
        )
    )

    assert [entry["key"] for entry in entries] == [key1, key2]
    assert q_group_state.cached_entries == entries
    assert preview_state.excluded_q_groups == {key1}
    assert events == ["invalidate", "label"]


def test_geometry_q_group_manager_preview_exclusion_toggle_and_clear_helpers() -> None:
    key1 = ("pair", 1)
    q_group_key = ("q_group", "primary", 1, 0)
    preview_state = state.GeometryPreviewState(
        excluded_q_groups={q_group_key},
    )
    preview_state.overlay.pairs = [
        {
            "id": 1,
            "sim_x": 10.0,
            "sim_y": 10.0,
            "x": 20.0,
            "y": 10.0,
        }
    ]
    toggle_events = []

    changed = geometry_q_group_manager.toggle_live_geometry_preview_exclusion_at(
        preview_state=preview_state,
        col=10.5,
        row=10.0,
        live_preview_match_key=lambda entry: ("pair", int(entry["id"])),
        live_preview_match_hkl=lambda _entry: (1, 0, 0),
        render_live_geometry_preview_state=lambda: toggle_events.append("render"),
        max_distance_px=5.0,
        set_status_text=lambda text: toggle_events.append(text),
    )
    changed_again = geometry_q_group_manager.toggle_live_geometry_preview_exclusion_at(
        preview_state=preview_state,
        col=10.5,
        row=10.0,
        live_preview_match_key=lambda entry: ("pair", int(entry["id"])),
        live_preview_match_hkl=lambda _entry: (1, 0, 0),
        render_live_geometry_preview_state=lambda: toggle_events.append("render"),
        max_distance_px=5.0,
        set_status_text=lambda text: toggle_events.append(text),
    )
    missed = geometry_q_group_manager.toggle_live_geometry_preview_exclusion_at(
        preview_state=preview_state,
        col=100.0,
        row=100.0,
        live_preview_match_key=lambda entry: ("pair", int(entry["id"])),
        live_preview_match_hkl=lambda _entry: (1, 0, 0),
        render_live_geometry_preview_state=lambda: toggle_events.append("unexpected"),
        max_distance_px=5.0,
        set_status_text=lambda text: toggle_events.append(text),
    )

    assert changed is True
    assert changed_again is True
    assert missed is False
    assert key1 not in preview_state.excluded_keys
    assert toggle_events == [
        "render",
        "Excluded live preview peak HKL=(1, 0, 0) from geometry fit.",
        "render",
        "Included live preview peak HKL=(1, 0, 0) from geometry fit.",
        "No preview pair within 5px to toggle.",
    ]

    clear_events = []
    preview_state.excluded_keys = {key1}
    geometry_q_group_manager.clear_live_geometry_preview_exclusions_with_side_effects(
        preview_state=preview_state,
        invalidate_geometry_manual_pick_cache=lambda: clear_events.append("invalidate"),
        update_geometry_preview_exclude_button_label=lambda: clear_events.append(
            "label"
        ),
        refresh_geometry_q_group_window=lambda: clear_events.append("refresh"),
        live_geometry_preview_enabled=lambda: False,
        refresh_live_geometry_preview=lambda: clear_events.append("live"),
        set_status_text=lambda text: clear_events.append(text),
    )

    assert preview_state.excluded_keys == set()
    assert preview_state.excluded_q_groups == set()
    assert clear_events == [
        "invalidate",
        "label",
        "refresh",
        "Reset all Qr/Qz geometry-fit selections.",
    ]


def test_geometry_q_group_manager_runtime_preview_exclude_mode_helper_updates_hkl_and_status(
    monkeypatch,
) -> None:
    preview_state = state.GeometryPreviewState()
    events = []

    monkeypatch.setattr(
        geometry_q_group_manager.gui_controllers,
        "set_geometry_preview_exclude_mode",
        lambda state_value, enabled: (
            events.append(("mode", state_value, enabled)) or True
        ),
    )

    bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=preview_state,
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: events.append("invalidate"),
        update_geometry_preview_exclude_button_label=lambda: events.append("label"),
        live_geometry_preview_enabled=lambda: False,
        refresh_live_geometry_preview=lambda: events.append("refresh"),
        set_hkl_pick_mode=lambda enabled: events.append(("hkl", enabled)),
        set_status_text=lambda text: events.append(("status", text)),
    )

    changed = geometry_q_group_manager.set_runtime_geometry_preview_exclude_mode(
        bindings,
        True,
        message="armed",
    )

    assert changed is True
    assert events == [
        ("mode", preview_state, True),
        ("hkl", False),
        "label",
        ("status", "armed"),
    ]


def test_geometry_q_group_manager_runtime_snapshot_capture_refreshes_open_window(
    monkeypatch,
) -> None:
    key1 = ("q_group", "primary", 1, 0)
    stale_key = ("q_group", "stale", 9, 9)
    preview_state = state.GeometryPreviewState(excluded_q_groups={key1, stale_key})
    q_group_state = state.GeometryQGroupState()
    events = []

    monkeypatch.setattr(
        geometry_q_group_manager.gui_views,
        "geometry_q_group_window_open",
        lambda view_state: True,
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "refresh_runtime_geometry_q_group_window",
        lambda bindings: events.append("refresh") or True,
    )

    bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(window=object()),
        preview_state=preview_state,
        q_group_state=q_group_state,
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: events.append("invalidate"),
        update_geometry_preview_exclude_button_label=lambda: events.append("label"),
        live_geometry_preview_enabled=lambda: False,
        refresh_live_geometry_preview=lambda: events.append("live"),
        build_entries_snapshot=lambda: [
            _entry(key1, peak_count=2, total_intensity=10.0),
        ],
    )

    entries = geometry_q_group_manager.capture_runtime_geometry_q_group_entries_snapshot(
        bindings
    )

    assert [entry["key"] for entry in entries] == [key1]
    assert q_group_state.cached_entries == entries
    assert preview_state.excluded_q_groups == {key1}
    assert events == ["invalidate", "label", "refresh"]


def test_geometry_q_group_manager_preview_exclusion_open_reports_status(
    monkeypatch,
) -> None:
    events = []

    def _open_window(*, root, bindings_factory):
        events.append(("open", root, bindings_factory()))
        return False

    monkeypatch.setattr(
        geometry_q_group_manager,
        "open_runtime_geometry_q_group_window",
        _open_window,
    )

    bindings = geometry_q_group_manager.GeometryQGroupRuntimeBindings(
        view_state=state.GeometryQGroupViewState(),
        preview_state=state.GeometryPreviewState(),
        q_group_state=state.GeometryQGroupState(),
        fit_config=None,
        current_geometry_fit_var_names_factory=lambda: [],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: False,
        refresh_live_geometry_preview=lambda: None,
        set_status_text=lambda text: events.append(("status", text)),
    )

    opened = geometry_q_group_manager.open_runtime_geometry_q_group_preview_exclusion_window(
        root="root-window",
        bindings_factory=lambda: bindings,
    )

    assert opened is False
    assert events[0] == ("open", "root-window", bindings)
    assert events[1] == (
        "status",
        "Opened the Qr/Qz group selector. Unchecked rows are skipped during manual picking and geometry fitting.",
    )


def test_geometry_q_group_manager_live_preview_toggle_helper_covers_disabled_scheduled_and_refreshed_paths() -> None:
    disabled_events = []

    refreshed = geometry_q_group_manager.toggle_live_geometry_preview_with_side_effects(
        enabled=False,
        disable_preview_exclude_mode=lambda: disabled_events.append("disable"),
        clear_geometry_preview_artists=lambda: disabled_events.append("clear"),
        open_geometry_q_group_window=lambda: disabled_events.append("open"),
        update_running=False,
        has_cached_hit_tables=True,
        schedule_update=lambda: disabled_events.append("schedule"),
        refresh_live_geometry_preview=lambda: disabled_events.append("refresh") or True,
        set_status_text=lambda text: disabled_events.append(text),
    )

    assert refreshed is False
    assert disabled_events == [
        "disable",
        "clear",
        "Live auto-match preview disabled.",
    ]

    scheduled_events = []
    refreshed = geometry_q_group_manager.toggle_live_geometry_preview_with_side_effects(
        enabled=True,
        disable_preview_exclude_mode=lambda: scheduled_events.append("disable"),
        clear_geometry_preview_artists=lambda: scheduled_events.append("clear"),
        open_geometry_q_group_window=lambda: scheduled_events.append("open"),
        update_running=True,
        has_cached_hit_tables=False,
        schedule_update=lambda: scheduled_events.append("schedule"),
        refresh_live_geometry_preview=lambda: scheduled_events.append("refresh") or True,
        set_status_text=lambda text: scheduled_events.append(text),
    )

    assert refreshed is True
    assert scheduled_events == ["open", "schedule"]

    failed_refresh_events = []
    refreshed = geometry_q_group_manager.toggle_live_geometry_preview_with_side_effects(
        enabled=True,
        disable_preview_exclude_mode=lambda: failed_refresh_events.append("disable"),
        clear_geometry_preview_artists=lambda: failed_refresh_events.append("clear"),
        open_geometry_q_group_window=lambda: failed_refresh_events.append("open"),
        update_running=False,
        has_cached_hit_tables=True,
        schedule_update=lambda: failed_refresh_events.append("schedule"),
        refresh_live_geometry_preview=lambda: failed_refresh_events.append("refresh")
        or False,
        set_status_text=lambda text: failed_refresh_events.append(text),
    )

    assert refreshed is False
    assert failed_refresh_events == ["open", "refresh", "schedule"]

    success_events = []
    refreshed = geometry_q_group_manager.toggle_live_geometry_preview_with_side_effects(
        enabled=True,
        disable_preview_exclude_mode=lambda: success_events.append("disable"),
        clear_geometry_preview_artists=lambda: success_events.append("clear"),
        open_geometry_q_group_window=lambda: success_events.append("open"),
        update_running=False,
        has_cached_hit_tables=True,
        schedule_update=lambda: success_events.append("schedule"),
        refresh_live_geometry_preview=lambda: success_events.append("refresh") or True,
        set_status_text=lambda text: success_events.append(text),
    )

    assert refreshed is True
    assert success_events == ["open", "refresh"]


def test_geometry_q_group_manager_save_dialog_workflow_writes_payload_and_reports_status() -> None:
    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "secondary", 2, 1)
    preview_state = state.GeometryPreviewState(excluded_q_groups={key2})
    q_group_state = state.GeometryQGroupState(
        cached_entries=[
            _entry(key1, peak_count=2, total_intensity=10.0),
            _entry(key2, peak_count=3, total_intensity=20.0, source="secondary"),
        ]
    )
    captured = {}
    messages = []

    def _asksaveasfilename(**kwargs):
        captured["dialog"] = kwargs
        return "C:/tmp/groups.json"

    saved = geometry_q_group_manager.save_geometry_q_group_selection_with_dialog(
        preview_state=preview_state,
        q_group_state=q_group_state,
        file_dialog_dir="C:/dialogs",
        asksaveasfilename=_asksaveasfilename,
        set_status_text=lambda text: messages.append(text),
        save_payload=lambda path, payload: captured.update(
            {"path": path, "payload": payload}
        ),
        now=lambda: datetime(2026, 3, 26, 12, 34, 56),
    )

    assert saved is True
    assert captured["dialog"]["initialdir"] == "C:/dialogs"
    assert captured["dialog"]["initialfile"] == "geometry_q_groups_20260326_123456.json"
    assert captured["path"] == "C:/tmp/groups.json"
    assert captured["payload"]["saved_at"] == "2026-03-26T12:34:56"
    assert captured["payload"]["included_count"] == 1
    assert messages == ["Saved 2 Qr/Qz groups to C:/tmp/groups.json"]


def test_geometry_q_group_manager_load_dialog_workflow_applies_state_and_refreshes() -> None:
    key1 = ("q_group", "primary", 1, 0)
    key2 = ("q_group", "secondary", 2, 1)
    key3 = ("q_group", "primary", 3, 2)
    preview_state = state.GeometryPreviewState()
    q_group_state = state.GeometryQGroupState(
        cached_entries=[
            _entry(key1, peak_count=2, total_intensity=10.0),
            _entry(key2, peak_count=3, total_intensity=20.0, source="secondary"),
            _entry(key3, peak_count=1, total_intensity=5.0),
        ]
    )
    events = []
    payload = geometry_q_group_manager.build_geometry_q_group_save_payload(
        [
            {"key": ["q_group", "primary", 1, 0], "included": True},
            {"key": ["q_group", "secondary", 2, 1], "included": False},
        ],
        saved_at="2026-03-26T12:00:00",
    )

    def _askopenfilename(**kwargs):
        events.append(("dialog", kwargs))
        return "C:/tmp/selector.json"

    loaded = geometry_q_group_manager.load_geometry_q_group_selection_with_dialog(
        preview_state=preview_state,
        q_group_state=q_group_state,
        file_dialog_dir="C:/dialogs",
        askopenfilename=_askopenfilename,
        update_geometry_preview_exclude_button_label=lambda: events.append("label"),
        refresh_geometry_q_group_window=lambda: events.append("refresh"),
        live_geometry_preview_enabled=lambda: True,
        refresh_live_geometry_preview=lambda: events.append("live"),
        set_status_text=lambda text: events.append(text),
        load_payload=lambda path: events.append(("load", path)) or payload,
    )

    assert loaded is True
    assert preview_state.excluded_q_groups == {key2, key3}
    assert events[0][0] == "dialog"
    assert events[0][1]["initialdir"] == "C:/dialogs"
    assert events[1] == ("load", "C:/tmp/selector.json")
    assert events[2:5] == ["label", "refresh", "live"]
    assert (
        events[5]
        == "Loaded Qr/Qz peak list from selector.json: matched 2, enabled 1, current-only excluded 1, saved-only missing 0."
    )


def test_geometry_q_group_runtime_binding_factory_builds_live_bindings(
    monkeypatch,
) -> None:
    calls = []
    counters = {"status": 0, "schedule": 0, "dir": 0}

    monkeypatch.setattr(
        geometry_q_group_manager,
        "GeometryQGroupRuntimeBindings",
        lambda **kwargs: calls.append(kwargs) or kwargs,
    )

    def build_status():
        counters["status"] += 1
        idx = counters["status"]
        return lambda text: f"status-{idx}:{text}"

    def build_schedule():
        counters["schedule"] += 1
        idx = counters["schedule"]
        return lambda: f"schedule-{idx}"

    def build_dialog_dir():
        counters["dir"] += 1
        return f"C:/dialogs/{counters['dir']}"

    factory = geometry_q_group_manager.make_runtime_geometry_q_group_bindings_factory(
        view_state="view-state",
        preview_state="preview-state",
        q_group_state="q-group-state",
        fit_config={"geometry": {}},
        current_geometry_fit_var_names_factory=lambda: ["gamma"],
        build_entries_snapshot=lambda: [{"key": ("q_group", "primary", 1, 0)}],
        invalidate_geometry_manual_pick_cache=lambda: None,
        update_geometry_preview_exclude_button_label=lambda: None,
        live_geometry_preview_enabled=lambda: False,
        refresh_live_geometry_preview=lambda: None,
        refresh_live_geometry_preview_quiet=lambda: None,
        clear_last_simulation_signature=lambda: None,
        schedule_update_factory=build_schedule,
        set_status_text_factory=build_status,
        file_dialog_dir_factory=build_dialog_dir,
        asksaveasfilename=lambda **kwargs: "save.json",
        askopenfilename=lambda **kwargs: "load.json",
    )

    assert factory()["view_state"] == "view-state"
    assert factory()["view_state"] == "view-state"
    assert calls[0]["preview_state"] == "preview-state"
    assert calls[0]["q_group_state"] == "q-group-state"
    assert calls[0]["fit_config"] == {"geometry": {}}
    assert callable(calls[0]["set_status_text"])
    assert callable(calls[0]["schedule_update"])
    assert callable(calls[0]["build_entries_snapshot"])
    assert calls[0]["build_entries_snapshot"]() == [{"key": ("q_group", "primary", 1, 0)}]
    assert calls[0]["file_dialog_dir"] == "C:/dialogs/1"
    assert calls[1]["file_dialog_dir"] == "C:/dialogs/2"
    assert calls[0]["set_status_text"] is not calls[1]["set_status_text"]
    assert calls[0]["schedule_update"] is not calls[1]["schedule_update"]


def test_geometry_q_group_runtime_callback_bundle_delegates_live_bindings(
    monkeypatch,
) -> None:
    calls = []
    versions = {"count": 0}

    monkeypatch.setattr(
        geometry_q_group_manager,
        "update_runtime_geometry_q_group_window_status",
        lambda bindings, entries=None: calls.append(("status", bindings, entries)),
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "refresh_runtime_geometry_q_group_window",
        lambda bindings: calls.append(("refresh", bindings)) or True,
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "on_runtime_geometry_q_group_checkbox_changed",
        lambda bindings, group_key, row_var: (
            calls.append(("toggle", bindings, group_key, row_var)),
            True,
        )[-1],
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "set_all_geometry_q_groups_enabled_runtime",
        lambda bindings, *, enabled: (
            calls.append(("bulk", bindings, enabled)),
            enabled,
        )[-1],
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "request_runtime_geometry_q_group_window_update",
        lambda bindings: calls.append(("update", bindings)),
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "save_geometry_q_group_selection_runtime",
        lambda bindings: calls.append(("save", bindings)) or True,
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "load_geometry_q_group_selection_runtime",
        lambda bindings: calls.append(("load", bindings)) or False,
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "close_runtime_geometry_q_group_window",
        lambda bindings: calls.append(("close", bindings)),
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "open_runtime_geometry_q_group_window",
        lambda *, root, bindings_factory: (
            calls.append(("open", root, bindings_factory())),
            True,
        )[-1],
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "open_runtime_geometry_q_group_preview_exclusion_window",
        lambda *, root, bindings_factory: (
            calls.append(("open-preview", root, bindings_factory())),
            False,
        )[-1],
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "set_runtime_geometry_preview_exclude_mode",
        lambda bindings, enabled, *, message=None: (
            calls.append(("mode", bindings, enabled, message)),
            True,
        )[-1],
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "clear_runtime_live_geometry_preview_exclusions",
        lambda bindings: calls.append(("clear-preview", bindings)),
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "toggle_runtime_live_geometry_preview_exclusion_at",
        lambda bindings, col, row: (
            calls.append(("toggle-preview", bindings, col, row)),
            False,
        )[-1],
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "toggle_runtime_live_geometry_preview",
        lambda bindings, *, root, bindings_factory: (
            calls.append(("live-toggle", bindings, root)),
            True,
        )[-1],
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "runtime_live_geometry_preview_enabled",
        lambda bindings: calls.append(("preview-enabled", bindings)) or False,
    )
    monkeypatch.setattr(
        geometry_q_group_manager,
        "render_runtime_live_geometry_preview_state",
        lambda bindings, *, update_status=True: (
            calls.append(("render-preview", bindings, update_status)),
            True,
        )[-1],
    )

    def build_bindings():
        versions["count"] += 1
        return f"bindings-{versions['count']}"

    callbacks = geometry_q_group_manager.make_runtime_geometry_q_group_callbacks(
        root="root-window",
        bindings_factory=build_bindings,
    )

    entries = [{"key": ("q_group", "primary", 1, 0)}]
    row_var = _FakeVar(True)
    group_key = ("q_group", "primary", 1, 0)

    callbacks.update_window_status(entries)
    assert callbacks.refresh_window() is True
    assert callbacks.on_toggle(group_key, row_var) is True
    assert callbacks.include_all() is True
    assert callbacks.exclude_all() is False
    callbacks.update_listed_peaks()
    assert callbacks.save_selection() is True
    assert callbacks.load_selection() is False
    callbacks.close_window()
    assert callbacks.open_window() is True
    assert callbacks.open_preview_exclusion_window() is False
    assert callbacks.set_preview_exclude_mode(True, message="armed") is True
    callbacks.clear_preview_exclusions()
    assert callbacks.toggle_preview_exclusion_at(1.5, 2.5) is False
    assert callbacks.toggle_live_preview() is True
    assert callbacks.live_preview_enabled() is False
    assert callbacks.render_live_preview_state(update_status=False) is True

    assert calls == [
        ("status", "bindings-1", entries),
        ("refresh", "bindings-2"),
        ("toggle", "bindings-3", group_key, row_var),
        ("bulk", "bindings-4", True),
        ("bulk", "bindings-5", False),
        ("update", "bindings-6"),
        ("save", "bindings-7"),
        ("load", "bindings-8"),
        ("close", "bindings-9"),
        ("open", "root-window", "bindings-10"),
        ("open-preview", "root-window", "bindings-11"),
        ("mode", "bindings-12", True, "armed"),
        ("clear-preview", "bindings-13"),
        ("toggle-preview", "bindings-14", 1.5, 2.5),
        ("live-toggle", "bindings-15", "root-window"),
        ("preview-enabled", "bindings-16"),
        ("render-preview", "bindings-17", False),
    ]
