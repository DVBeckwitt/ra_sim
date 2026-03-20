import ast
from pathlib import Path

import numpy as np


class _DummyVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


def _load_main_functions(*names: str) -> dict[str, object]:
    source = Path("main.py").read_text(encoding="utf-8")
    module = ast.parse(source, filename="main.py")
    available = {
        node.name
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    }
    missing = sorted(set(names) - available)
    if missing:
        raise AssertionError(f"Failed to extract functions from main.py: {missing}")

    extracted: list[str] = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            fn_source = ast.get_source_segment(source, node)
            if fn_source:
                extracted.append(fn_source)

    namespace: dict[str, object] = {}
    exec(
        "import numpy as np\n"
        "from typing import Sequence\n\n"
        + "\n\n".join(extracted),
        namespace,
    )
    return namespace


def test_manual_pair_store_keeps_backgrounds_separate() -> None:
    namespace = _load_main_functions(
        "_normalize_hkl_key",
        "_geometry_manual_position_error_px",
        "_geometry_manual_position_sigma_px",
        "_normalize_geometry_manual_pair_entry",
        "_geometry_manual_pairs_for_index",
        "_set_geometry_manual_pairs_for_index",
        "_geometry_manual_pair_group_count",
    )
    namespace["geometry_manual_pairs_by_background"] = {}
    namespace["GEOMETRY_MANUAL_POSITION_SIGMA_FLOOR_PX"] = 0.75

    set_pairs = namespace["_set_geometry_manual_pairs_for_index"]
    get_pairs = namespace["_geometry_manual_pairs_for_index"]
    group_count = namespace["_geometry_manual_pair_group_count"]

    bg0_pairs = set_pairs(
        0,
        [
            {
                "label": "1,0,2",
                "x": "10.5",
                "y": 12,
                "q_group_key": ["q_group", "primary", 1, 2],
                "source_table_index": "4",
                "source_row_index": "7",
                "raw_x": "9.5",
                "raw_y": 11.5,
            }
        ],
    )
    bg1_pairs = set_pairs(
        1,
        [
            {
                "hkl": (2, 0, 0),
                "x": 5,
                "y": 6,
                "q_group_key": ("q_group", "primary", 2, 0),
            }
        ],
    )

    assert bg0_pairs[0]["hkl"] == (1, 0, 2)
    assert bg0_pairs[0]["source_table_index"] == 4
    assert bg0_pairs[0]["source_row_index"] == 7
    assert bg0_pairs[0]["q_group_key"] == ("q_group", "primary", 1, 2)
    assert bg0_pairs[0]["raw_x"] == 9.5
    assert bg0_pairs[0]["raw_y"] == 11.5
    assert bg0_pairs[0]["placement_error_px"] > 0.0
    assert bg0_pairs[0]["sigma_px"] > bg0_pairs[0]["placement_error_px"]

    assert len(get_pairs(0)) == 1
    assert len(get_pairs(1)) == 1
    assert group_count(0) == 1
    assert group_count(1) == 1
    assert get_pairs(0)[0]["hkl"] != get_pairs(1)[0]["hkl"]


def test_peak_maximum_near_in_image_returns_local_brightest_pixel() -> None:
    namespace = _load_main_functions("_peak_maximum_near_in_image")
    peak_maximum = namespace["_peak_maximum_near_in_image"]

    import numpy as np

    image = np.zeros((9, 9), dtype=float)
    image[4, 4] = 2.0
    image[6, 5] = 9.5
    image[2, 2] = 7.0

    assert peak_maximum(image, 4.2, 4.1, search_radius=1) == (4.0, 4.0)
    assert peak_maximum(image, 4.9, 5.8, search_radius=2) == (5.0, 6.0)


def test_caked_axis_index_helpers_round_trip() -> None:
    namespace = _load_main_functions(
        "_caked_axis_to_image_index",
        "_caked_image_index_to_axis",
    )
    axis_to_index = namespace["_caked_axis_to_image_index"]
    index_to_axis = namespace["_caked_image_index_to_axis"]

    axis = np.linspace(-30.0, 30.0, 121)
    idx = axis_to_index(7.5, axis)
    restored = index_to_axis(idx, axis)

    assert np.isfinite(idx)
    assert abs(restored - 7.5) < 1e-9


def test_refine_caked_peak_center_finds_ridge_crest() -> None:
    namespace = _load_main_functions(
        "_caked_axis_to_image_index",
        "_caked_image_index_to_axis",
        "_refine_profile_peak_index",
        "_refine_caked_peak_center",
    )
    namespace["GEOMETRY_MANUAL_CAKED_SEARCH_TTH_DEG"] = 1.5
    namespace["GEOMETRY_MANUAL_CAKED_SEARCH_PHI_DEG"] = 10.0
    refine = namespace["_refine_caked_peak_center"]

    radial = np.linspace(10.0, 20.0, 201)
    azimuth = np.linspace(-30.0, 30.0, 301)
    radial_grid, azimuth_grid = np.meshgrid(radial, azimuth)
    image = (
        2.0
        + 6.0 * np.exp(-0.5 * ((radial_grid - 15.2) / 0.22) ** 2)
        * np.exp(-0.5 * ((azimuth_grid - 7.5) / 4.2) ** 2)
    )

    refined_tth, refined_phi = refine(
        image,
        radial,
        azimuth,
        14.7,
        11.0,
    )

    assert abs(refined_tth - 15.2) < 0.08
    assert abs(refined_phi - 7.5) < 0.35


def test_geometry_manual_candidate_source_key_prefers_source_indices() -> None:
    namespace = _load_main_functions(
        "_normalize_hkl_key",
        "_geometry_manual_candidate_source_key",
    )
    source_key = namespace["_geometry_manual_candidate_source_key"]

    assert source_key({"source_table_index": "3", "source_row_index": 9}) == ("source", 3, 9)
    assert source_key({"hkl": (1, 2, 3)}) == ("hkl", 1, 2, 3)
    assert source_key({"label": "1,2,3"}) == ("hkl", 1, 2, 3)
    assert source_key({"label": "left peak"}) == ("label", "left peak")


def test_current_geometry_manual_match_config_reuses_auto_match_defaults() -> None:
    namespace = _load_main_functions("_current_geometry_manual_match_config")
    namespace["fit_config"] = {
        "geometry": {
            "auto_match": {
                "search_radius_px": 17.5,
                "min_match_prominence_sigma": 3.25,
            }
        }
    }
    config_fn = namespace["_current_geometry_manual_match_config"]
    cfg = config_fn()

    assert cfg["search_radius_px"] == 17.5
    assert cfg["min_match_prominence_sigma"] == 3.25
    assert cfg["console_progress"] is False
    assert cfg["relax_on_low_matches"] is False
    assert cfg["require_candidate_ownership"] is True


def test_geometry_manual_choose_group_at_picks_nearest_seed() -> None:
    namespace = _load_main_functions("_geometry_manual_choose_group_at")
    choose_group = namespace["_geometry_manual_choose_group_at"]

    grouped_candidates = {
        ("q_group", "primary", 1, 0): [
            {"label": "1,0,0", "sim_col": 20.0, "sim_row": 24.0},
            {"label": "-1,0,0", "sim_col": 42.0, "sim_row": 24.0},
        ],
        ("q_group", "primary", 3, 0): [
            {"label": "2,1,0", "sim_col": 75.0, "sim_row": 24.0},
        ],
    }

    group_key, entries, best_dist = choose_group(
        grouped_candidates,
        19.5,
        23.5,
        window_size_px=50.0,
    )

    assert group_key == ("q_group", "primary", 1, 0)
    assert len(entries) == 2
    assert best_dist < 1.0


def test_geometry_manual_choose_group_at_ignores_peaks_outside_50px_window() -> None:
    namespace = _load_main_functions("_geometry_manual_choose_group_at")
    choose_group = namespace["_geometry_manual_choose_group_at"]

    grouped_candidates = {
        ("q_group", "primary", 1, 0): [
            {"label": "1,0,0", "sim_col": 80.0, "sim_row": 80.0},
        ],
    }

    group_key, entries, best_dist = choose_group(
        grouped_candidates,
        20.0,
        20.0,
        window_size_px=50.0,
    )

    assert group_key is None
    assert entries == []
    assert best_dist != best_dist


def test_geometry_manual_zoom_bounds_returns_clamped_100px_window() -> None:
    namespace = _load_main_functions("_geometry_manual_zoom_bounds")
    zoom_bounds = namespace["_geometry_manual_zoom_bounds"]

    x_min, x_max, y_min, y_max = zoom_bounds(
        150.0,
        80.0,
        (200, 300),
        window_size_px=100.0,
    )
    assert (x_min, x_max, y_min, y_max) == (100.0, 200.0, 30.0, 130.0)

    edge_bounds = zoom_bounds(
        12.0,
        15.0,
        (200, 300),
        window_size_px=100.0,
    )
    assert edge_bounds == (0.0, 100.0, 0.0, 100.0)


def test_geometry_manual_group_target_count_uses_single_bg_peak_for_00l() -> None:
    namespace = _load_main_functions(
        "_normalize_hkl_key",
        "_geometry_manual_group_target_count",
    )
    target_count = namespace["_geometry_manual_group_target_count"]

    assert (
        target_count(
            ("q_group", "primary", 0, 3),
            [
                {"hkl": (0, 0, 3), "label": "0,0,3"},
                {"hkl": (0, 0, 3), "label": "0,0,3"},
            ],
        )
        == 1
    )
    assert (
        target_count(
            ("q_group", "primary", 1, 2),
            [
                {"hkl": (1, 0, 2), "label": "1,0,2"},
                {"hkl": (-1, 0, 2), "label": "-1,0,2"},
            ],
        )
        == 2
    )


def test_geometry_manual_nearest_candidate_to_point_selects_closest_simulated_peak() -> None:
    namespace = _load_main_functions("_geometry_manual_nearest_candidate_to_point")
    nearest_candidate = namespace["_geometry_manual_nearest_candidate_to_point"]

    candidate, dist = nearest_candidate(
        28.0,
        15.5,
        [
            {"label": "left", "sim_col": 12.0, "sim_row": 15.0},
            {"label": "right", "sim_col": 30.0, "sim_row": 16.0},
        ],
    )

    assert isinstance(candidate, dict)
    assert candidate["label"] == "right"
    assert dist < 3.0


def test_geometry_manual_pair_entry_from_candidate_preserves_caked_coords() -> None:
    namespace = _load_main_functions(
        "_normalize_hkl_key",
        "_geometry_manual_pair_entry_from_candidate",
    )
    pair_entry_from_candidate = namespace["_geometry_manual_pair_entry_from_candidate"]

    entry = pair_entry_from_candidate(
        {
            "label": "1,0,2",
            "hkl": (1, 0, 2),
            "source_table_index": 3,
            "source_row_index": 8,
        },
        120.0,
        240.0,
        group_key=("q_group", "primary", 1, 2),
        raw_col=118.5,
        raw_row=239.0,
        caked_col=13.2,
        caked_row=-7.4,
        raw_caked_col=13.0,
        raw_caked_row=-7.0,
        placement_error_px=1.7,
        sigma_px=1.9,
    )

    assert entry is not None
    assert entry["x"] == 120.0
    assert entry["y"] == 240.0
    assert entry["caked_x"] == 13.2
    assert entry["caked_y"] == -7.4
    assert entry["raw_caked_x"] == 13.0
    assert entry["raw_caked_y"] == -7.0


def test_ensure_geometry_fit_caked_view_switches_and_refreshes_immediately() -> None:
    namespace = _load_main_functions("_ensure_geometry_fit_caked_view")

    calls: list[str] = []

    class _DummyRoot:
        def __init__(self) -> None:
            self.canceled: list[object] = []

        def after_cancel(self, token) -> None:
            self.canceled.append(token)

    namespace["show_caked_2d_var"] = _DummyVar(False)
    namespace["_geometry_manual_pick_uses_caked_space"] = lambda: False
    namespace["toggle_caked_2d"] = lambda: calls.append("toggle")
    namespace["do_update"] = lambda: calls.append("update")
    namespace["schedule_update"] = lambda: calls.append("schedule")
    namespace["root"] = _DummyRoot()
    namespace["update_pending"] = "update-token"
    namespace["integration_update_pending"] = "range-token"
    namespace["update_running"] = False

    namespace["_ensure_geometry_fit_caked_view"]()

    assert namespace["show_caked_2d_var"].get() is True
    assert calls == ["toggle", "update"]
    assert namespace["root"].canceled == ["range-token", "update-token"]
    assert namespace["update_pending"] is None
    assert namespace["integration_update_pending"] is None


def test_native_detector_coords_to_caked_display_coords_prefers_angular_maps() -> None:
    namespace = _load_main_functions(
        "_wrap_phi_range",
        "_native_detector_coords_to_caked_display_coords",
    )
    namespace["_ai_cache"] = {"ai": object()}
    namespace["_get_detector_angular_maps"] = lambda _ai: (
        np.array([[11.0, 12.5], [13.0, 14.0]], dtype=float),
        np.array([[181.0, -190.0], [35.0, 40.0]], dtype=float),
    )
    namespace["_detector_pixel_to_scattering_angles"] = lambda *_args, **_kwargs: (
        (_ for _ in ()).throw(AssertionError("fallback should not be used"))
    )
    namespace["center_x_var"] = _DummyVar(0.0)
    namespace["center_y_var"] = _DummyVar(0.0)
    namespace["corto_detector_var"] = _DummyVar(1.0)
    namespace["pixel_size_m"] = 1.0

    result = namespace["_native_detector_coords_to_caked_display_coords"](0.9, 0.1)

    assert result is not None
    assert result[0] == 12.5
    assert result[1] == 170.0


def test_native_detector_coords_to_caked_display_coords_falls_back_when_map_lookup_raises() -> None:
    namespace = _load_main_functions(
        "_wrap_phi_range",
        "_native_detector_coords_to_caked_display_coords",
    )
    namespace["_ai_cache"] = {"ai": object()}
    namespace["_get_detector_angular_maps"] = lambda _ai: (
        (_ for _ in ()).throw(RuntimeError("map lookup failed"))
    )
    namespace["_detector_pixel_to_scattering_angles"] = (
        lambda *_args, **_kwargs: (22.0, 190.0)
    )
    namespace["center_x_var"] = _DummyVar(0.0)
    namespace["center_y_var"] = _DummyVar(0.0)
    namespace["corto_detector_var"] = _DummyVar(1.0)
    namespace["pixel_size_m"] = 1.0

    result = namespace["_native_detector_coords_to_caked_display_coords"](5.0, 7.0)

    assert result is not None
    assert result == (22.0, -170.0)


def test_caked_angles_to_background_display_coords_returns_none_without_native_background() -> None:
    namespace = _load_main_functions(
        "_scattering_angles_to_detector_pixel",
        "_caked_angles_to_background_display_coords",
    )
    namespace["_ai_cache"] = {"ai": object()}
    namespace["_get_detector_angular_maps"] = lambda _ai: (None, None)
    namespace["_get_current_background_native"] = lambda: None
    namespace["center_x_var"] = _DummyVar(0.0)
    namespace["center_y_var"] = _DummyVar(0.0)
    namespace["corto_detector_var"] = _DummyVar(1.0)
    namespace["pixel_size_m"] = 1.0

    result = namespace["_caked_angles_to_background_display_coords"](12.0, 30.0)

    assert result == (None, None)


def test_geometry_manual_preview_due_throttles_small_motion() -> None:
    namespace = _load_main_functions(
        "_geometry_manual_pick_session_active",
        "_geometry_manual_preview_due",
    )
    preview_due = namespace["_geometry_manual_preview_due"]
    namespace["geometry_manual_pick_session"] = {
        "group_key": ("q_group", "primary", 1, 0),
        "group_entries": [{"label": "1,0,0"}],
        "background_index": 0,
        "preview_last_t": 0.0,
        "preview_last_xy": None,
    }
    namespace["current_background_index"] = 0
    namespace["GEOMETRY_MANUAL_PREVIEW_MIN_INTERVAL_S"] = 0.03
    namespace["GEOMETRY_MANUAL_PREVIEW_MIN_MOVE_PX"] = 0.8

    times = iter([1.0, 1.01, 1.05])
    namespace["perf_counter"] = lambda: next(times)

    assert preview_due(10.0, 20.0) is True
    assert preview_due(10.2, 20.1) is False
    assert preview_due(10.2, 20.1) is True


def test_geometry_manual_pair_json_round_trip_preserves_hkl_and_group_key() -> None:
    namespace = _load_main_functions(
        "_normalize_hkl_key",
        "_normalize_bragg_qr_source_label",
        "_q_group_key_component",
        "_integer_gz_index",
        "_geometry_manual_position_error_px",
        "_geometry_manual_position_sigma_px",
        "_geometry_q_group_key_to_jsonable",
        "_geometry_q_group_key_from_jsonable",
        "_normalize_geometry_manual_pair_entry",
        "_geometry_manual_pair_entry_to_jsonable",
        "_geometry_manual_pair_entry_from_jsonable",
    )
    namespace["GEOMETRY_MANUAL_POSITION_SIGMA_FLOOR_PX"] = 0.75

    to_json = namespace["_geometry_manual_pair_entry_to_jsonable"]
    from_json = namespace["_geometry_manual_pair_entry_from_jsonable"]

    serialized = to_json(
        {
            "label": "1,0,2",
            "hkl": (1, 0, 2),
            "x": 10.5,
            "y": 12.25,
            "q_group_key": ("q_group", "primary", 1.0, 2),
            "source_table_index": 4,
            "source_row_index": 7,
            "raw_x": 9.0,
            "raw_y": 11.0,
            "placement_error_px": 1.25,
            "sigma_px": 1.46,
        }
    )

    assert serialized["hkl"] == [1, 0, 2]
    assert serialized["q_group_key"] == ["q_group", "primary", 1.0, 2]
    assert serialized["placement_error_px"] == 1.25
    assert serialized["sigma_px"] == 1.46

    restored = from_json(serialized)

    assert restored["hkl"] == (1, 0, 2)
    assert restored["q_group_key"] == ("q_group", "primary", 1.0, 2)
    assert restored["source_table_index"] == 4
    assert restored["source_row_index"] == 7
    assert restored["raw_x"] == 9.0
    assert restored["raw_y"] == 11.0
    assert restored["placement_error_px"] == 1.25
    assert restored["sigma_px"] == 1.46
