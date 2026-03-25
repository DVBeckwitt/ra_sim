import numpy as np

from ra_sim.gui import manual_geometry as mg


class _DummyVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


def _pairs_for_index(pairs_by_background: dict[int, list[dict[str, object]]], index: int):
    return mg.geometry_manual_pairs_for_index(
        index,
        pairs_by_background=pairs_by_background,
        sigma_floor_px=0.75,
    )


def _set_pairs(
    pairs_by_background: dict[int, list[dict[str, object]]],
    index: int,
    entries,
):
    return mg.set_geometry_manual_pairs_for_index(
        index,
        entries,
        pairs_by_background=pairs_by_background,
        sigma_floor_px=0.75,
    )


def _group_count(
    pairs_by_background: dict[int, list[dict[str, object]]],
    index: int,
) -> int:
    return mg.geometry_manual_pair_group_count(
        index,
        pairs_by_background=pairs_by_background,
        sigma_floor_px=0.75,
    )


def _wrap_phi_range(phi_values):
    return ((np.asarray(phi_values) + 180.0) % 360.0) - 180.0


def test_manual_pair_store_keeps_backgrounds_separate() -> None:
    pairs_by_background: dict[int, list[dict[str, object]]] = {}

    bg0_pairs = _set_pairs(
        pairs_by_background,
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
    bg1_pairs = _set_pairs(
        pairs_by_background,
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

    assert len(_pairs_for_index(pairs_by_background, 0)) == 1
    assert len(_pairs_for_index(pairs_by_background, 1)) == 1
    assert _group_count(pairs_by_background, 0) == 1
    assert _group_count(pairs_by_background, 1) == 1
    assert _pairs_for_index(pairs_by_background, 0)[0]["hkl"] != _pairs_for_index(
        pairs_by_background,
        1,
    )[0]["hkl"]


def test_peak_maximum_near_in_image_returns_local_brightest_pixel() -> None:
    image = np.zeros((9, 9), dtype=float)
    image[4, 4] = 2.0
    image[6, 5] = 9.5
    image[2, 2] = 7.0

    assert mg.peak_maximum_near_in_image(image, 4.2, 4.1, search_radius=1) == (4.0, 4.0)
    assert mg.peak_maximum_near_in_image(image, 4.9, 5.8, search_radius=2) == (5.0, 6.0)


def test_caked_axis_index_helpers_round_trip() -> None:
    axis = np.linspace(-30.0, 30.0, 121)
    idx = mg.caked_axis_to_image_index(7.5, axis)
    restored = mg.caked_image_index_to_axis(idx, axis)

    assert np.isfinite(idx)
    assert abs(restored - 7.5) < 1e-9


def test_refine_caked_peak_center_finds_ridge_crest() -> None:
    radial = np.linspace(10.0, 20.0, 201)
    azimuth = np.linspace(-30.0, 30.0, 301)
    radial_grid, azimuth_grid = np.meshgrid(radial, azimuth)
    image = (
        2.0
        + 6.0 * np.exp(-0.5 * ((radial_grid - 15.2) / 0.22) ** 2)
        * np.exp(-0.5 * ((azimuth_grid - 7.5) / 4.2) ** 2)
    )

    refined_tth, refined_phi = mg.refine_caked_peak_center(image, radial, azimuth, 14.7, 11.0)

    assert abs(refined_tth - 15.2) < 0.08
    assert abs(refined_phi - 7.5) < 0.35


def test_geometry_manual_candidate_source_key_prefers_source_indices() -> None:
    assert mg.geometry_manual_candidate_source_key({"source_table_index": "3", "source_row_index": 9}) == (
        "source",
        3,
        9,
    )
    assert mg.geometry_manual_candidate_source_key({"hkl": (1, 2, 3)}) == ("hkl", 1, 2, 3)
    assert mg.geometry_manual_candidate_source_key({"label": "1,2,3"}) == ("hkl", 1, 2, 3)
    assert mg.geometry_manual_candidate_source_key({"label": "left peak"}) == ("label", "left peak")


def test_current_geometry_manual_match_config_reuses_auto_match_defaults() -> None:
    cfg = mg.current_geometry_manual_match_config(
        {
            "geometry": {
                "auto_match": {
                    "search_radius_px": 17.5,
                    "min_match_prominence_sigma": 3.25,
                }
            }
        }
    )

    assert cfg["search_radius_px"] == 17.5
    assert cfg["min_match_prominence_sigma"] == 3.25
    assert cfg["console_progress"] is False
    assert cfg["relax_on_low_matches"] is False
    assert cfg["require_candidate_ownership"] is True


def test_geometry_manual_choose_group_at_picks_nearest_seed() -> None:
    grouped_candidates = {
        ("q_group", "primary", 1, 0): [
            {"label": "1,0,0", "sim_col": 20.0, "sim_row": 24.0},
            {"label": "-1,0,0", "sim_col": 42.0, "sim_row": 24.0},
        ],
        ("q_group", "primary", 3, 0): [{"label": "2,1,0", "sim_col": 75.0, "sim_row": 24.0}],
    }

    group_key, entries, best_dist = mg.geometry_manual_choose_group_at(
        grouped_candidates,
        19.5,
        23.5,
        window_size_px=50.0,
    )

    assert group_key == ("q_group", "primary", 1, 0)
    assert len(entries) == 2
    assert best_dist < 1.0


def test_geometry_manual_choose_group_at_ignores_peaks_outside_50px_window() -> None:
    group_key, entries, best_dist = mg.geometry_manual_choose_group_at(
        {("q_group", "primary", 1, 0): [{"label": "1,0,0", "sim_col": 80.0, "sim_row": 80.0}]},
        20.0,
        20.0,
        window_size_px=50.0,
    )

    assert group_key is None
    assert entries == []
    assert np.isnan(best_dist)


def test_geometry_manual_zoom_bounds_returns_clamped_100px_window() -> None:
    assert mg.geometry_manual_zoom_bounds(150.0, 80.0, (200, 300), window_size_px=100.0) == (
        100.0,
        200.0,
        30.0,
        130.0,
    )
    assert mg.geometry_manual_zoom_bounds(12.0, 15.0, (200, 300), window_size_px=100.0) == (
        0.0,
        100.0,
        0.0,
        100.0,
    )


def test_geometry_manual_anchor_axis_limits_preserves_click_fraction() -> None:
    assert mg.geometry_manual_anchor_axis_limits(150.0, 100.0, 0.25, 0.0, 300.0) == (
        125.0,
        225.0,
    )
    assert mg.geometry_manual_anchor_axis_limits(80.0, -100.0, 0.75, 0.0, 200.0) == (
        155.0,
        55.0,
    )
    assert mg.geometry_manual_anchor_axis_limits(12.0, 100.0, 0.2, 0.0, 300.0) == (
        0.0,
        100.0,
    )


def test_geometry_manual_group_target_count_uses_single_bg_peak_for_00l() -> None:
    assert mg.geometry_manual_group_target_count(
        ("q_group", "primary", 0, 3),
        [{"hkl": (0, 0, 3), "label": "0,0,3"}, {"hkl": (0, 0, 3), "label": "0,0,3"}],
    ) == 1
    assert mg.geometry_manual_group_target_count(
        ("q_group", "primary", 1, 2),
        [{"hkl": (1, 0, 2), "label": "1,0,2"}, {"hkl": (-1, 0, 2), "label": "-1,0,2"}],
    ) == 2


def test_geometry_manual_nearest_candidate_to_point_selects_closest_simulated_peak() -> None:
    candidate, dist = mg.geometry_manual_nearest_candidate_to_point(
        28.0,
        15.5,
        [{"label": "left", "sim_col": 12.0, "sim_row": 15.0}, {"label": "right", "sim_col": 30.0, "sim_row": 16.0}],
    )

    assert isinstance(candidate, dict)
    assert candidate["label"] == "right"
    assert dist < 3.0


def test_geometry_manual_pair_entry_from_candidate_preserves_caked_coords() -> None:
    entry = mg.geometry_manual_pair_entry_from_candidate(
        {"label": "1,0,2", "hkl": (1, 0, 2), "source_table_index": 3, "source_row_index": 8},
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
    calls: list[str] = []

    class _DummyRoot:
        def __init__(self) -> None:
            self.canceled: list[object] = []

        def after_cancel(self, token) -> None:
            self.canceled.append(token)

    update_pending, integration_update_pending = mg.ensure_geometry_fit_caked_view(
        show_caked_2d_var=_DummyVar(False),
        pick_uses_caked_space=lambda: False,
        toggle_caked_2d=lambda: calls.append("toggle"),
        do_update=lambda: calls.append("update"),
        schedule_update=lambda: calls.append("schedule"),
        root=_DummyRoot(),
        update_pending="update-token",
        integration_update_pending="range-token",
        update_running=False,
    )

    assert calls == ["toggle", "update"]
    assert update_pending is None
    assert integration_update_pending is None


def test_native_detector_coords_to_caked_display_coords_prefers_angular_maps() -> None:
    result = mg.native_detector_coords_to_caked_display_coords(
        0.9,
        0.1,
        ai=object(),
        get_detector_angular_maps=lambda _ai: (
            np.array([[11.0, 12.5], [13.0, 14.0]], dtype=float),
            np.array([[181.0, -190.0], [35.0, 40.0]], dtype=float),
        ),
        detector_pixel_to_scattering_angles=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fallback should not be used")
        ),
        center=[0.0, 0.0],
        detector_distance=1.0,
        pixel_size=1.0,
        wrap_phi_range=_wrap_phi_range,
    )

    assert result == (12.5, 170.0)


def test_native_detector_coords_to_caked_display_coords_falls_back_when_map_lookup_raises() -> None:
    result = mg.native_detector_coords_to_caked_display_coords(
        5.0,
        7.0,
        ai=object(),
        get_detector_angular_maps=lambda _ai: (_ for _ in ()).throw(RuntimeError("map lookup failed")),
        detector_pixel_to_scattering_angles=lambda *_args, **_kwargs: (22.0, 190.0),
        center=[0.0, 0.0],
        detector_distance=1.0,
        pixel_size=1.0,
        wrap_phi_range=_wrap_phi_range,
    )

    assert result == (22.0, -170.0)


def test_caked_angles_to_background_display_coords_returns_none_without_native_background() -> None:
    result = mg.caked_angles_to_background_display_coords(
        12.0,
        30.0,
        ai=object(),
        native_background=None,
        get_detector_angular_maps=lambda _ai: (None, None),
        scattering_angles_to_detector_pixel=lambda *_args, **_kwargs: (10.0, 20.0),
        center=[0.0, 0.0],
        detector_distance=1.0,
        pixel_size=1.0,
    )

    assert result == (None, None)


def test_geometry_manual_preview_due_throttles_small_motion() -> None:
    session = {
        "group_key": ("q_group", "primary", 1, 0),
        "group_entries": [{"label": "1,0,0"}],
        "background_index": 0,
        "preview_last_t": 0.0,
        "preview_last_xy": None,
    }
    times = iter([1.0, 1.01, 1.05])

    assert mg.geometry_manual_preview_due(
        10.0,
        20.0,
        pick_session=session,
        current_background_index=0,
        min_interval_s=0.03,
        min_move_px=0.8,
        perf_counter_fn=lambda: next(times),
    )
    assert not mg.geometry_manual_preview_due(
        10.2,
        20.1,
        pick_session=session,
        current_background_index=0,
        min_interval_s=0.03,
        min_move_px=0.8,
        perf_counter_fn=lambda: next(times),
    )
    assert mg.geometry_manual_preview_due(
        10.2,
        20.1,
        pick_session=session,
        current_background_index=0,
        min_interval_s=0.03,
        min_move_px=0.8,
        perf_counter_fn=lambda: next(times),
    )


def test_should_collect_hit_tables_when_manual_geometry_overlay_is_visible() -> None:
    assert mg.should_collect_hit_tables_for_update(
        background_visible=True,
        current_background_index=2,
        hkl_pick_armed=False,
        selected_hkl_target=None,
        selected_peak_record=None,
        geometry_q_group_refresh_requested=False,
        live_geometry_preview_enabled=lambda: False,
        current_manual_pick_background_image=lambda: object(),
        geometry_manual_pairs_for_index=lambda idx: [{"hkl": (1, 0, 2)}] if int(idx) == 2 else [],
        geometry_manual_pick_session_active=lambda: False,
    )


def test_should_not_collect_hit_tables_for_hidden_manual_geometry_overlay() -> None:
    assert not mg.should_collect_hit_tables_for_update(
        background_visible=False,
        current_background_index=0,
        hkl_pick_armed=False,
        selected_hkl_target=None,
        selected_peak_record=None,
        geometry_q_group_refresh_requested=False,
        live_geometry_preview_enabled=lambda: False,
        current_manual_pick_background_image=lambda: object(),
        geometry_manual_pairs_for_index=lambda _idx: [{"hkl": (1, 0, 2)}],
        geometry_manual_pick_session_active=lambda: False,
    )


def test_geometry_manual_pair_json_round_trip_preserves_hkl_and_group_key() -> None:
    serialized = mg.geometry_manual_pair_entry_to_jsonable(
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
        },
        sigma_floor_px=0.75,
    )

    assert serialized["hkl"] == [1, 0, 2]
    assert serialized["q_group_key"] == ["q_group", "primary", 1.0, 2]
    assert serialized["placement_error_px"] == 1.25
    assert serialized["sigma_px"] == 1.46

    restored = mg.geometry_manual_pair_entry_from_jsonable(serialized, sigma_floor_px=0.75)

    assert restored["hkl"] == (1, 0, 2)
    assert restored["q_group_key"] == ("q_group", "primary", 1.0, 2)
    assert restored["source_table_index"] == 4
    assert restored["source_row_index"] == 7
    assert restored["raw_x"] == 9.0
    assert restored["raw_y"] == 11.0
    assert restored["placement_error_px"] == 1.25
    assert restored["sigma_px"] == 1.46
