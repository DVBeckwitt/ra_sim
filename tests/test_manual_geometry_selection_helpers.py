import numpy as np

from ra_sim.fitting.background_peak_matching import build_background_peak_context
from ra_sim.gui import manual_geometry as mg


class _DummyVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class _DummyAxis:
    def __init__(self, xlim=(0.0, 1.0), ylim=(0.0, 1.0)):
        self._xlim = tuple(float(v) for v in xlim)
        self._ylim = tuple(float(v) for v in ylim)

    def get_xlim(self):
        return self._xlim

    def get_ylim(self):
        return self._ylim

    def set_xlim(self, left, right):
        self._xlim = (float(left), float(right))

    def set_ylim(self, bottom, top):
        self._ylim = (float(bottom), float(top))


class _DummyCanvas:
    def __init__(self) -> None:
        self.draws = 0

    def draw_idle(self):
        self.draws += 1


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
    assert mg.geometry_manual_candidate_source_key(
        {
            "source_table_index": "3",
            "source_row_index": 9,
            "source_peak_index": 5,
        }
    ) == ("source_peak", 3, 5)
    assert mg.geometry_manual_candidate_source_key({"source_table_index": "3", "source_row_index": 9}) == (
        "source",
        3,
        9,
    )
    assert mg.geometry_manual_candidate_source_key({"hkl": (1, 2, 3)}) == ("hkl", 1, 2, 3)
    assert mg.geometry_manual_candidate_source_key({"label": "1,2,3"}) == ("hkl", 1, 2, 3)
    assert mg.geometry_manual_candidate_source_key({"label": "left peak"}) == ("label", "left peak")


def test_geometry_manual_tagged_candidate_from_session_returns_matching_entry() -> None:
    candidate_entries = [
        {"label": "left", "source_table_index": 1, "source_row_index": 2},
        {"label": "right", "source_table_index": 1, "source_row_index": 3},
    ]

    tagged = mg.geometry_manual_tagged_candidate_from_session(
        {"tagged_candidate_key": ("source", 1, 3)},
        candidate_entries,
    )

    assert tagged is not None
    assert tagged["label"] == "right"
    assert mg.geometry_manual_tagged_candidate_from_session(
        {"tagged_candidate_key": ("source", 9, 9)},
        candidate_entries,
    ) is None


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
        {
            "label": "1,0,2",
            "hkl": (1, 0, 2),
            "source_table_index": 3,
            "source_row_index": 8,
            "source_peak_index": 2,
        },
        120.0,
        240.0,
        group_key=("q_group", "primary", 1, 2),
        detector_col=119.5,
        detector_row=239.5,
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
    assert entry["detector_x"] == 119.5
    assert entry["detector_y"] == 239.5
    assert entry["caked_x"] == 13.2
    assert entry["caked_y"] == -7.4
    assert entry["raw_caked_x"] == 13.0
    assert entry["raw_caked_y"] == -7.0
    assert entry["source_peak_index"] == 2


def test_make_runtime_geometry_manual_callbacks_render_current_pairs_uses_live_state() -> None:
    events: list[tuple[object, ...]] = []
    status_texts: list[str] = []

    callbacks = mg.make_runtime_geometry_manual_callbacks(
        background_visible=lambda: True,
        current_background_index=lambda: 2,
        current_background_image=lambda: np.ones((4, 4), dtype=float),
        pick_session=lambda: None,
        build_initial_pairs_display=lambda index, *, prefer_cache: (
            [{"measured": int(index)}],
            [{"saved": int(index)}],
        ),
        session_initial_pairs_display=lambda: [{"pending": True}],
        clear_geometry_pick_artists=lambda *args, **kwargs: events.append(
            ("clear", args, kwargs)
        ),
        draw_initial_geometry_pairs_overlay=lambda pairs, *, max_display_markers: (
            events.append(("draw", list(pairs), int(max_display_markers)))
        ),
        update_button_label=lambda: events.append(("button",)),
        set_background_file_status_text=lambda: events.append(("background-status",)),
        pair_group_count=lambda index: 1,
        set_status_text=lambda text: status_texts.append(str(text)),
        get_cache_data=lambda **kwargs: {},
        set_pairs_for_index=lambda index, entries: list(entries or []),
        pairs_for_index=lambda index: [{"pair": int(index)}],
        set_pick_session=lambda session: events.append(("set-session", dict(session))),
        restore_view=lambda **kwargs: events.append(("restore", kwargs)),
        clear_preview_artists=lambda **kwargs: events.append(("clear-preview", kwargs)),
        use_caked_space=False,
        pick_search_window_px=50.0,
        refine_preview_point=lambda *args, **kwargs: (0.0, 0.0),
        remaining_candidates=lambda: [],
        preview_due=lambda col, row: True,
        show_preview=lambda *args, **kwargs: events.append(("show-preview", args, kwargs)),
    )

    assert callbacks.render_current_pairs(update_status=True) is True
    assert events == [
        ("draw", [{"saved": 2}, {"pending": True}], 2),
        ("button",),
        ("background-status",),
    ]
    assert status_texts == [
        "Current background has 1 saved manual points across 1 Qr/Qz groups."
    ]


def test_make_runtime_geometry_manual_callbacks_delegate_toggle_preview_and_cancel(
    monkeypatch,
) -> None:
    events: list[tuple[object, ...]] = []
    status_texts: list[str] = []
    pick_session_state: dict[str, object] = {"value": {"mode": "start"}}

    def _set_pick_session(session: dict[str, object]) -> None:
        pick_session_state["value"] = dict(session)
        events.append(("set-session", dict(session)))

    def _fake_toggle(col: float, row: float, **kwargs):
        events.append(
            (
                "toggle",
                float(col),
                float(row),
                kwargs["current_background_index"],
                kwargs["use_caked_space"],
                dict(kwargs["pick_session"]),
            )
        )
        kwargs["set_pick_session_fn"]({"mode": "toggle"})
        return True, {"ignored": True}, True

    def _fake_place(col: float, row: float, **kwargs):
        events.append(
            (
                "place",
                float(col),
                float(row),
                kwargs["current_background_index"],
                kwargs["use_caked_space"],
                dict(kwargs["pick_session"]),
            )
        )
        kwargs["set_pick_session_fn"]({"mode": "place"})
        return True, {"ignored": True}

    def _fake_preview_state(col: float, row: float, **kwargs):
        events.append(
            (
                "preview-state",
                float(col),
                float(row),
                kwargs["current_background_index"],
                kwargs["force"],
                list(kwargs["remaining_candidates"]),
                kwargs["use_caked_space"],
            )
        )
        return {
            "raw_col": 5.0,
            "raw_row": 6.0,
            "refined_col": 7.5,
            "refined_row": 8.5,
            "delta": 1.25,
            "sigma_px": 1.46,
            "preview_color": "#2ecc71",
            "message": "preview ready",
        }

    def _fake_cancel(pick_session, **kwargs):
        events.append(
            (
                "cancel",
                dict(pick_session),
                kwargs["current_background_index"],
                kwargs["restore_view"],
                kwargs["redraw"],
                kwargs["message"],
            )
        )
        return {"mode": "cancel"}

    monkeypatch.setattr(mg, "geometry_manual_toggle_selection_at", _fake_toggle)
    monkeypatch.setattr(mg, "geometry_manual_place_selection_at", _fake_place)
    monkeypatch.setattr(mg, "geometry_manual_pick_preview_state", _fake_preview_state)
    monkeypatch.setattr(mg, "cancel_geometry_manual_pick_session", _fake_cancel)

    callbacks = mg.make_runtime_geometry_manual_callbacks(
        background_visible=lambda: True,
        current_background_index=lambda: 2,
        current_background_image=lambda: "bg-image",
        pick_session=lambda: pick_session_state["value"],
        build_initial_pairs_display=lambda index, *, prefer_cache: ([], []),
        session_initial_pairs_display=lambda: [],
        clear_geometry_pick_artists=lambda *args, **kwargs: None,
        draw_initial_geometry_pairs_overlay=lambda pairs, *, max_display_markers: None,
        update_button_label=lambda: events.append(("button",)),
        set_background_file_status_text=lambda: events.append(("background-status",)),
        pair_group_count=lambda index: 0,
        set_status_text=lambda text: status_texts.append(str(text)),
        get_cache_data=lambda **kwargs: {"cache": True},
        set_pairs_for_index=lambda index, entries: list(entries or []),
        pairs_for_index=lambda index: [],
        set_pick_session=_set_pick_session,
        restore_view=lambda **kwargs: events.append(("restore-view", kwargs)),
        clear_preview_artists=lambda **kwargs: events.append(("clear-preview", kwargs)),
        push_undo_state=lambda: events.append(("push-undo",)),
        listed_q_group_entries=lambda: [{"key": ("q", 1)}],
        format_q_group_line=lambda entry: "Q1",
        use_caked_space=lambda: True,
        pick_search_window_px=25.0,
        set_suppress_drag_press_once=lambda enabled: events.append(("suppress", enabled)),
        sync_peak_selection_state=lambda: events.append(("sync",)),
        refine_preview_point=lambda *args, **kwargs: (11.0, 12.0),
        remaining_candidates=lambda: [{"label": "cand"}],
        preview_due=lambda col, row: True,
        nearest_candidate_to_point=lambda col, row, candidates: (
            {"label": "cand"},
            1.5,
        ),
        caked_angles_to_background_display_coords=lambda col, row: (col + 100.0, row + 200.0),
        show_preview=lambda *args, **kwargs: events.append(("show-preview", args, kwargs)),
    )

    assert callbacks.toggle_selection_at(10.0, 20.0) is True
    assert callbacks.place_selection_at(30.0, 40.0) is True
    callbacks.update_pick_preview(5.0, 6.0, force=True)
    callbacks.cancel_pick_session(restore_view=False, redraw=False, message="bye")

    assert pick_session_state["value"] == {"mode": "cancel"}
    assert status_texts == ["preview ready"]
    assert events == [
        ("toggle", 10.0, 20.0, 2, True, {"mode": "start"}),
        ("set-session", {"mode": "toggle"}),
        ("suppress", True),
        ("sync",),
        ("place", 30.0, 40.0, 2, True, {"mode": "toggle"}),
        ("set-session", {"mode": "place"}),
        ("preview-state", 5.0, 6.0, 2, True, [{"label": "cand"}], True),
        (
            "show-preview",
            (5.0, 6.0, 7.5, 8.5),
            {
                "delta_px": 1.25,
                "sigma_px": 1.46,
                "preview_color": "#2ecc71",
            },
        ),
        ("cancel", {"mode": "place"}, 2, False, False, "bye"),
        ("set-session", {"mode": "cancel"}),
    ]


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
        skip_preview_once=False,
        manual_pick_armed=False,
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
        skip_preview_once=False,
        manual_pick_armed=False,
        hkl_pick_armed=False,
        selected_hkl_target=None,
        selected_peak_record=None,
        geometry_q_group_refresh_requested=False,
        live_geometry_preview_enabled=lambda: False,
        current_manual_pick_background_image=lambda: object(),
        geometry_manual_pairs_for_index=lambda _idx: [{"hkl": (1, 0, 2)}],
        geometry_manual_pick_session_active=lambda: False,
    )


def test_should_not_collect_hit_tables_when_preview_skip_once_is_requested() -> None:
    assert not mg.should_collect_hit_tables_for_update(
        background_visible=True,
        current_background_index=2,
        skip_preview_once=True,
        manual_pick_armed=False,
        hkl_pick_armed=False,
        selected_hkl_target=None,
        selected_peak_record=None,
        geometry_q_group_refresh_requested=False,
        live_geometry_preview_enabled=lambda: True,
        current_manual_pick_background_image=lambda: object(),
        geometry_manual_pairs_for_index=lambda idx: [{"hkl": (1, 0, 2)}] if int(idx) == 2 else [],
        geometry_manual_pick_session_active=lambda: True,
    )


def test_should_collect_hit_tables_when_manual_pick_is_armed() -> None:
    assert mg.should_collect_hit_tables_for_update(
        background_visible=True,
        current_background_index=1,
        skip_preview_once=False,
        manual_pick_armed=True,
        hkl_pick_armed=False,
        selected_hkl_target=None,
        selected_peak_record=None,
        geometry_q_group_refresh_requested=False,
        live_geometry_preview_enabled=lambda: False,
        current_manual_pick_background_image=lambda: object(),
        geometry_manual_pairs_for_index=lambda _idx: [],
        geometry_manual_pick_session_active=lambda: False,
    )


def test_geometry_manual_pair_json_round_trip_preserves_hkl_and_group_key() -> None:
    serialized = mg.geometry_manual_pair_entry_to_jsonable(
        {
            "label": "1,0,2",
            "hkl": (1, 0, 2),
            "x": 10.5,
            "y": 12.25,
            "detector_x": 10.0,
            "detector_y": 12.0,
            "q_group_key": ("q_group", "primary", 1.0, 2),
            "source_table_index": 4,
            "source_row_index": 7,
            "source_peak_index": 3,
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
    assert restored["detector_x"] == 10.0
    assert restored["detector_y"] == 12.0
    assert restored["source_table_index"] == 4
    assert restored["source_row_index"] == 7
    assert restored["source_peak_index"] == 3
    assert restored["raw_x"] == 9.0
    assert restored["raw_y"] == 11.0
    assert restored["placement_error_px"] == 1.25
    assert restored["sigma_px"] == 1.46


def test_geometry_manual_pairs_export_rows_include_background_metadata() -> None:
    rows = mg.geometry_manual_pairs_export_rows(
        pairs_by_background={1: [{"label": "1,0,0", "x": 2.0, "y": 3.0}]},
        osc_files=["bg_0.osc", "bg_1.osc"],
        pairs_for_index=lambda idx: (
            [{"label": "1,0,0", "x": 2.0, "y": 3.0}]
            if int(idx) == 1
            else []
        ),
    )

    assert rows == [
        {
            "background_index": 1,
            "background_path": "bg_1.osc",
            "background_name": "bg_1.osc",
            "entries": [{"x": 2.0, "y": 3.0, "label": "1,0,0", "hkl": [1, 0, 0]}],
        }
    ]


def test_collect_geometry_manual_pairs_snapshot_records_loaded_backgrounds() -> None:
    snapshot = mg.collect_geometry_manual_pairs_snapshot(
        osc_files=["bg_0.osc", "bg_1.osc"],
        current_background_index=1,
        manual_pair_rows=[{"background_index": 1, "entries": []}],
    )

    assert snapshot == {
        "background_files": ["bg_0.osc", "bg_1.osc"],
        "current_background_index": 1,
        "manual_pairs": [{"background_index": 1, "entries": []}],
    }


def test_apply_geometry_manual_pairs_rows_replaces_state_and_refreshes_callbacks() -> None:
    calls: list[tuple[str, object]] = []
    replaced: dict[int, list[dict[str, object]]] = {}

    imported_backgrounds, imported_pairs, warnings = mg.apply_geometry_manual_pairs_rows(
        [
            {
                "background_path": "bg_1.osc",
                "background_name": "bg_1.osc",
                "entries": [{"label": "1,0,0", "x": 2.0, "y": 3.0}],
            }
        ],
        osc_files=["bg_0.osc", "bg_1.osc"],
        pairs_for_index=lambda idx: (
            [{"label": "keep", "x": 9.0, "y": 10.0}]
            if int(idx) == 0
            else []
        ),
        replace_pairs_by_background=lambda mapping: replaced.update(mapping),
        clear_preview_artists=lambda **kwargs: calls.append(("clear", kwargs)),
        cancel_pick_session=lambda **kwargs: calls.append(("cancel", kwargs)),
        invalidate_pick_cache=lambda: calls.append(("invalidate", None)),
        clear_manual_undo_stack=lambda: calls.append(("clear_manual", None)),
        clear_geometry_fit_undo_stack=lambda: calls.append(("clear_fit", None)),
        render_current_pairs=lambda **kwargs: calls.append(("render", kwargs)),
        update_button_label=lambda: calls.append(("button", None)),
        refresh_status=lambda: calls.append(("refresh", None)),
    )

    assert (imported_backgrounds, imported_pairs, warnings) == (1, 1, [])
    assert replaced == {1: [{"label": "1,0,0", "hkl": (1, 0, 0), "x": 2.0, "y": 3.0}]}
    assert ("clear", {"redraw": False}) in calls
    assert ("cancel", {"restore_view": True, "redraw": False}) in calls
    assert ("invalidate", None) in calls
    assert ("clear_manual", None) in calls
    assert ("clear_fit", None) in calls
    assert ("render", {"update_status": False}) in calls
    assert ("button", None) in calls
    assert ("refresh", None) in calls


def test_apply_geometry_manual_pairs_snapshot_reloads_backgrounds_before_apply(
    tmp_path,
) -> None:
    saved_backgrounds = [tmp_path / "bg_0.osc", tmp_path / "bg_1.osc"]
    for path in saved_backgrounds:
        path.write_text("", encoding="utf-8")

    calls: list[tuple[str, object]] = []
    message = mg.apply_geometry_manual_pairs_snapshot(
        {
            "background_files": [str(path) for path in saved_backgrounds],
            "current_background_index": 1,
            "manual_pairs": [{"background_index": 1, "entries": []}],
        },
        osc_files=["current_0.osc", "current_1.osc"],
        load_background_files=(
            lambda paths, index: calls.append(("load", (list(paths), index)))
        ),
        apply_pairs_rows=(
            lambda rows, replace_existing=True: (
                calls.append(("apply", (list(rows), replace_existing))) or (1, 2, [])
            )
        ),
        schedule_update=lambda: calls.append(("schedule", None)),
    )

    assert calls[0] == (
        "load",
        ([str(path) for path in saved_backgrounds], 1),
    )
    assert calls[1] == (
        "apply",
        ([{"background_index": 1, "entries": []}], True),
    )
    assert calls[2] == ("schedule", None)
    assert message == "Imported 2 manual placement(s) across 1 background(s)."


def test_export_geometry_manual_pairs_runs_dialog_and_save_callback(tmp_path) -> None:
    statuses: list[str] = []
    calls: list[tuple[str, object]] = []
    save_path = tmp_path / "placements.json"

    result = mg.export_geometry_manual_pairs(
        osc_files=["bg_0.osc"],
        pairs_for_index=lambda idx: (
            [{"label": "1,0,0", "x": 1.0, "y": 2.0}]
            if int(idx) == 0
            else []
        ),
        collect_snapshot=lambda: {"manual_pairs": [{"background_index": 0}]},
        initial_dir=tmp_path,
        asksaveasfilename=(
            lambda **kwargs: calls.append(("dialog", kwargs)) or str(save_path)
        ),
        save_file=(
            lambda path, payload, metadata=None: calls.append(
                ("save", (path, payload, metadata))
            )
        ),
        set_status_text=statuses.append,
        stamp_factory=lambda: "20260328_140000",
    )

    assert result == str(save_path)
    assert calls[0][0] == "dialog"
    assert calls[0][1]["initialfile"] == "ra_sim_geometry_placements_20260328_140000.json"
    assert calls[1] == (
        "save",
        (
            str(save_path),
            {"manual_pairs": [{"background_index": 0}]},
            {"entrypoint": "python -m ra_sim gui"},
        ),
    )
    assert statuses[-1] == f"Saved manual geometry placements to {save_path}"


def test_import_geometry_manual_pairs_loads_snapshot_and_reports_caked_view_warning(
    tmp_path,
) -> None:
    statuses: list[str] = []
    apply_calls: list[tuple[dict[str, object], bool]] = []
    open_path = tmp_path / "placements.json"

    result = mg.import_geometry_manual_pairs(
        initial_dir=tmp_path,
        askopenfilename=lambda **_kwargs: str(open_path),
        load_file=lambda _path: {"state": {"manual_pairs": [{"background_index": 0}]}},
        apply_snapshot=(
            lambda snapshot, allow_background_reload=True: (
                apply_calls.append((dict(snapshot), bool(allow_background_reload)))
                or "Imported placements."
            )
        ),
        ensure_geometry_fit_caked_view=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        set_status_text=statuses.append,
    )

    assert apply_calls == [({"manual_pairs": [{"background_index": 0}]}, True)]
    assert result == (
        "Imported placements. Warning: imported placements but could not switch "
        "to 2D caked view (boom)."
    )
    assert statuses[-1] == result


def test_geometry_manual_unassigned_candidates_and_current_pending_candidate() -> None:
    session = {
        "group_key": ("q_group", "primary", 1, 0),
        "group_entries": [
            {"label": "1,0,0", "source_table_index": 1, "source_row_index": 2},
            {"label": "-1,0,0", "source_table_index": 1, "source_row_index": 3},
        ],
        "pending_entries": [
            {"label": "1,0,0", "source_table_index": 1, "source_row_index": 2, "x": 8.0, "y": 9.0}
        ],
        "background_index": 0,
    }

    remaining = mg.geometry_manual_unassigned_group_candidates(
        session,
        current_background_index=0,
    )
    pending = mg.geometry_manual_current_pending_candidate(
        session,
        current_background_index=0,
    )

    assert len(remaining) == 1
    assert remaining[0]["label"] == "-1,0,0"
    assert pending["label"] == "-1,0,0"


def test_geometry_manual_refine_preview_point_falls_back_to_local_peak_maximum() -> None:
    image = np.zeros((9, 9), dtype=float)
    image[6, 5] = 9.5

    refined = mg.geometry_manual_refine_preview_point(
        None,
        4.9,
        5.8,
        display_background=image,
        cache_data={"match_config": {}, "background_context": None},
        use_caked_space=False,
    )

    assert refined == (5.0, 6.0)


def test_restore_geometry_manual_pick_view_resets_zoom_state() -> None:
    session = {
        "group_key": ("q_group", "primary", 1, 0),
        "group_entries": [],
        "zoom_active": True,
        "zoom_center": (5.0, 6.0),
        "saved_xlim": (10.0, 20.0),
        "saved_ylim": (30.0, 40.0),
    }
    axis = _DummyAxis()
    canvas = _DummyCanvas()

    restored = mg.restore_geometry_manual_pick_view(session, axis=axis, canvas=canvas)

    assert restored is True
    assert axis.get_xlim() == (10.0, 20.0)
    assert axis.get_ylim() == (30.0, 40.0)
    assert session["zoom_active"] is False
    assert session["zoom_center"] is None
    assert session["saved_xlim"] is None
    assert session["saved_ylim"] is None
    assert canvas.draws == 1


def test_apply_geometry_manual_pick_zoom_updates_axis_and_session() -> None:
    session = {
        "group_key": ("q_group", "primary", 1, 0),
        "group_entries": [],
        "background_index": 0,
        "zoom_active": False,
    }
    axis = _DummyAxis((0.0, 300.0), (0.0, 200.0))
    canvas = _DummyCanvas()

    updated = mg.apply_geometry_manual_pick_zoom(
        session,
        150.0,
        80.0,
        display_background=np.zeros((200, 300), dtype=float),
        axis=axis,
        canvas=canvas,
        use_caked_space=False,
        caked_zoom_tth_deg=4.0,
        caked_zoom_phi_deg=24.0,
        pick_zoom_window_px=100.0,
    )

    assert updated is True
    assert axis.get_xlim() == (100.0, 200.0)
    assert axis.get_ylim() == (30.0, 130.0)
    assert session["zoom_active"] is True
    assert session["zoom_center"] == (150.0, 80.0)
    assert session["saved_xlim"] == (0.0, 300.0)
    assert session["saved_ylim"] == (0.0, 200.0)
    assert canvas.draws == 1


def test_geometry_manual_pick_preview_state_builds_status_message() -> None:
    session = {
        "group_key": ("q_group", "primary", 1, 0),
        "q_label": "test group",
        "group_entries": [{"label": "1,0,0"}],
        "background_index": 0,
    }

    preview = mg.geometry_manual_pick_preview_state(
        10.0,
        20.0,
        pick_session=session,
        current_background_index=0,
        force=True,
        remaining_candidates=[{"label": "right", "sim_col": 12.5, "sim_row": 14.5}],
        display_background=np.zeros((8, 8), dtype=float),
        refine_preview_point=lambda *_args, **_kwargs: (12.0, 14.0),
        preview_due=lambda *_args, **_kwargs: True,
        use_caked_space=False,
    )

    assert preview is not None
    assert preview["refined_col"] == 12.0
    assert preview["refined_row"] == 14.0
    assert preview["delta"] > 0.0
    assert preview["sigma_px"] > preview["delta"]
    assert preview["preview_color"] == mg.geometry_manual_preview_color(
        preview["sigma_px"]
    )
    assert preview["quality_label"] == mg.geometry_manual_preview_quality_label(
        preview["sigma_px"]
    )
    assert "test group" in preview["message"]
    assert "nearest sim [right]" in preview["message"]
    assert f"quality={preview['quality_label']}" in preview["message"]


def test_geometry_manual_pick_preview_state_prefers_tagged_candidate() -> None:
    session = {
        "group_key": ("q_group", "primary", 1, 2),
        "q_label": "test group",
        "group_entries": [{"label": "left"}, {"label": "right"}],
        "background_index": 0,
        "tagged_candidate_key": ("source", 1, 2),
    }

    preview = mg.geometry_manual_pick_preview_state(
        10.0,
        20.0,
        pick_session=session,
        current_background_index=0,
        force=True,
        remaining_candidates=[
            {
                "label": "left",
                "sim_col": 35.0,
                "sim_row": 30.0,
                "source_table_index": 1,
                "source_row_index": 2,
            },
            {
                "label": "right",
                "sim_col": 12.5,
                "sim_row": 14.5,
                "source_table_index": 1,
                "source_row_index": 3,
            },
        ],
        display_background=np.zeros((8, 8), dtype=float),
        refine_preview_point=lambda *_args, **_kwargs: (12.0, 14.0),
        preview_due=lambda *_args, **_kwargs: True,
        use_caked_space=False,
    )

    assert preview is not None
    assert preview["candidate"]["label"] == "left"
    assert "tagged sim [left]" in preview["message"]


def test_geometry_manual_pick_preview_state_colors_from_match_confidence() -> None:
    background = np.zeros((33, 33), dtype=float)
    background[10, 10] = 1000.0
    background[10, 11] = 600.0
    background[11, 10] = 600.0
    state = {
        "match_config": {},
        "background_context": build_background_peak_context(background),
    }

    preview = mg.geometry_manual_pick_preview_state(
        24.0,
        24.0,
        pick_session={
            "group_key": ("q_group", "primary", 1, 0),
            "q_label": "test group",
            "group_entries": [{"label": "1,0,0"}],
            "background_index": 0,
        },
        current_background_index=0,
        force=True,
        remaining_candidates=[
            {"label": "1,0,0", "sim_col": 10.0, "sim_row": 10.0},
        ],
        display_background=background,
        cache_data=state,
        refine_preview_point=lambda *_args, **_kwargs: (10.0, 10.0),
        preview_due=lambda *_args, **_kwargs: True,
        use_caked_space=False,
    )

    assert preview is not None
    assert np.isfinite(preview["match_confidence"])
    assert preview["preview_color"] == mg.geometry_manual_preview_confidence_color(
        preview["match_confidence"]
    )
    assert preview["preview_color"] != mg.geometry_manual_preview_color(
        preview["sigma_px"]
    )
    assert "confidence=" in preview["message"]


def test_geometry_manual_preview_color_transitions_from_green_to_red() -> None:
    assert mg.geometry_manual_preview_color(0.75) == "#2ecc71"
    assert mg.geometry_manual_preview_color(12.0) == "#e74c3c"
    assert mg.geometry_manual_preview_color(6.0) not in {"#2ecc71", "#e74c3c"}


def test_geometry_manual_preview_confidence_color_transitions_from_red_to_green() -> None:
    assert mg.geometry_manual_preview_confidence_color(0.1) == "#e74c3c"
    assert mg.geometry_manual_preview_confidence_color(1.0) == "#2ecc71"
    assert mg.geometry_manual_preview_confidence_color(0.5) not in {
        "#2ecc71",
        "#e74c3c",
    }


def test_geometry_manual_preview_quality_label_tracks_sigma() -> None:
    assert mg.geometry_manual_preview_quality_label(0.75) == "good"
    assert mg.geometry_manual_preview_quality_label(4.0) == "warning"
    assert mg.geometry_manual_preview_quality_label(12.0) == "bad"


def test_geometry_manual_preview_confidence_quality_label_tracks_confidence() -> None:
    assert mg.geometry_manual_preview_confidence_quality_label(1.1) == "good"
    assert mg.geometry_manual_preview_confidence_quality_label(0.5) == "warning"
    assert mg.geometry_manual_preview_confidence_quality_label(0.1) == "bad"


def test_geometry_manual_session_initial_pairs_display_includes_pending_bg_points() -> None:
    session = {
        "group_key": ("q_group", "primary", 1, 0),
        "group_entries": [
            {
                "label": "1,0,0",
                "source_table_index": 1,
                "source_row_index": 2,
                "sim_col": 5.0,
                "sim_row": 6.0,
            }
        ],
        "pending_entries": [
            {
                "label": "1,0,0",
                "source_table_index": 1,
                "source_row_index": 2,
                "x": 9.0,
                "y": 10.0,
            }
        ],
        "background_index": 0,
    }

    entries = mg.geometry_manual_session_initial_pairs_display(
        session,
        current_background_index=0,
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert len(entries) == 1
    assert entries[0]["sim_display"] == (5.0, 6.0)
    assert entries[0]["bg_display"] == (9.0, 10.0)


def test_cancel_geometry_manual_pick_session_clears_session_and_triggers_callbacks() -> None:
    session = {
        "group_key": ("q_group", "primary", 1, 0),
        "group_entries": [],
        "background_index": 0,
    }
    calls: list[tuple[str, object]] = []

    cleared = mg.cancel_geometry_manual_pick_session(
        session,
        current_background_index=0,
        restore_view_fn=lambda **kwargs: calls.append(("restore", kwargs.get("redraw"))),
        clear_preview_artists_fn=lambda **kwargs: calls.append(("clear", kwargs.get("redraw"))),
        render_current_pairs_fn=lambda **kwargs: calls.append(("render", kwargs.get("update_status"))),
        update_button_label_fn=lambda: calls.append(("button", None)),
        set_status_text=lambda text: calls.append(("status", text)),
        message="done",
    )

    assert cleared == {}
    assert ("restore", False) in calls
    assert ("clear", False) in calls
    assert ("render", False) in calls
    assert ("button", None) in calls
    assert ("status", "done") in calls


def test_match_geometry_manual_group_to_background_builds_source_lookup() -> None:
    matches = mg.match_geometry_manual_group_to_background(
        [{"label": "1,0,0", "source_table_index": 3, "source_row_index": 8}],
        background_image=np.zeros((8, 8), dtype=float),
        cache_data={"match_config": {"search_radius_px": 12.0}, "background_context": {"img_valid": True}},
        match_simulated_peaks_to_peak_context=lambda entries, _context, _cfg: (
            [
                {
                    "source_table_index": entries[0]["source_table_index"],
                    "source_row_index": entries[0]["source_row_index"],
                    "x": 1.5,
                    "y": 2.5,
                }
            ],
            {},
        ),
    )

    assert matches == {("source", 3, 8): (1.5, 2.5)}


def test_geometry_manual_pick_cache_signature_tracks_background_state() -> None:
    signature = mg.geometry_manual_pick_cache_signature(
        last_simulation_signature=("sim", 7),
        background_index=2,
        background_image=np.zeros((6, 5), dtype=float),
        use_caked_space=True,
        geometry_preview_excluded_q_groups=[("q_group", "primary", 1, 0)],
        geometry_q_group_cached_entries=[{"key": ("q_group", "primary", 1, 0)}],
        stored_max_positions_local=[{"x": 1.0}],
        stored_peak_table_lattice=[{"hkl": (1, 0, 0)}, {"hkl": (0, 0, 2)}],
    )

    assert signature[0] == ("sim", 7)
    assert signature[1] == 2
    assert signature[2] is True
    assert signature[4] == 1
    assert signature[5] == 2
    assert signature[6] == 1
    assert signature[7] == ("('q_group', 'primary', 1, 0)",)


def test_build_geometry_manual_pick_cache_reuses_existing_current_background_state() -> None:
    existing_cache = {"signature": ("cached",), "value": 9}

    cache_data, next_sig, next_state = mg.build_geometry_manual_pick_cache(
        background_index=0,
        current_background_index=0,
        background_image=np.zeros((3, 3), dtype=float),
        existing_cache_signature=("cached",),
        existing_cache_data=existing_cache,
        cache_signature_fn=lambda **_kwargs: ("cached",),
        simulated_peaks_for_params=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cache miss")
        ),
        build_grouped_candidates=lambda _entries: {},
        build_simulated_lookup=lambda _entries: {},
        current_match_config=lambda: {},
    )

    assert cache_data is existing_cache
    assert next_sig == ("cached",)
    assert next_state is existing_cache


def test_build_geometry_manual_initial_pairs_display_uses_cache_lookup() -> None:
    measured_display, initial_pairs_display = mg.build_geometry_manual_initial_pairs_display(
        0,
        current_background_index=0,
        prefer_cache=True,
        pairs_for_index=lambda _idx: [
            {
                "label": "1,0,2",
                "hkl": (1, 0, 2),
                "x": 9.0,
                "y": 11.0,
                "source_table_index": 4,
                "source_row_index": 7,
            }
        ],
        current_geometry_fit_params=lambda: {"a": 1.0},
        get_cache_data=lambda **_kwargs: {
            "simulated_lookup": {
                (4, 7): {"sim_col": 13.5, "sim_row": 15.5},
            }
        },
        simulated_peaks_for_params=lambda *_args, **_kwargs: [],
        build_simulated_lookup=lambda _entries: {},
        entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
    )

    assert measured_display[0]["overlay_match_index"] == 0
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (1, 0, 2),
            "bg_display": (9.0, 11.0),
            "sim_display": (13.5, 15.5),
        }
    ]


def test_make_runtime_geometry_manual_cache_callbacks_store_cache_state_and_build_pairs() -> None:
    cache_state = {"signature": None, "data": {}}
    simulated_param_sets: list[dict[str, object]] = []

    def _replace_cache_state(signature, data) -> None:
        cache_state["signature"] = signature
        cache_state["data"] = dict(data)

    callbacks = mg.make_runtime_geometry_manual_cache_callbacks(
        fit_config={"geometry": {"auto_match": {"search_radius_px": 18.0}}},
        last_simulation_signature=lambda: ("sim", 3),
        current_background_index=lambda: 0,
        current_background_image=lambda: np.zeros((4, 4), dtype=float),
        use_caked_space=lambda: False,
        geometry_preview_excluded_q_groups=lambda: [("q_group", "primary", 1, 0)],
        geometry_q_group_cached_entries=lambda: [{"key": ("q_group", "primary", 1, 0)}],
        stored_max_positions_local=lambda: [{"x": 1.0}],
        stored_peak_table_lattice=lambda: [{"hkl": (1, 0, 0)}],
        current_cache_signature=lambda: cache_state["signature"],
        current_cache_data=lambda: cache_state["data"],
        replace_cache_state=_replace_cache_state,
        current_geometry_fit_params=lambda: {"gamma": 1.25},
        pairs_for_index=lambda idx: (
            [
                {
                    "label": "1,0,2",
                    "hkl": (1, 0, 2),
                    "x": 9.0,
                    "y": 11.0,
                    "source_table_index": 4,
                    "source_row_index": 7,
                }
            ]
            if int(idx) == 1
            else []
        ),
        simulated_peaks_for_params=lambda params, prefer_cache=True: (
            simulated_param_sets.append(dict(params or {}))
            or [
                {
                    "source_table_index": 4,
                    "source_row_index": 7,
                    "sim_col": 13.5,
                    "sim_row": 15.5,
                }
            ]
        ),
        build_grouped_candidates=lambda entries: {
            ("q_group", "primary", 1, 0): [dict(entry) for entry in entries or ()]
        },
        build_simulated_lookup=lambda entries: {
            (
                int(entry.get("source_table_index")),
                int(entry.get("source_row_index")),
            ): dict(entry)
            for entry in entries or ()
        },
        entry_display_coords=lambda entry: (
            float(entry["x"]),
            float(entry["y"]),
        )
        if isinstance(entry, dict)
        else None,
        auto_match_background_context=lambda image, cfg: (
            {**dict(cfg), "search_radius_px": 22.0},
            {"image_shape": np.asarray(image).shape},
        ),
    )

    cache_data = callbacks.get_pick_cache(param_set={"a": 2.0}, prefer_cache=False)
    measured_display, initial_pairs_display = callbacks.build_initial_pairs_display(
        1,
        prefer_cache=False,
    )

    assert callbacks.current_match_config()["search_radius_px"] == 18.0
    assert cache_data["match_config"]["search_radius_px"] == 22.0
    assert cache_state["signature"] == cache_data["signature"]
    assert cache_state["data"] == cache_data
    assert simulated_param_sets == [{"a": 2.0}, {"gamma": 1.25}]
    assert measured_display[0]["overlay_match_index"] == 0
    assert initial_pairs_display == [
        {
            "overlay_match_index": 0,
            "hkl": (1, 0, 2),
            "bg_display": (9.0, 11.0),
            "sim_display": (13.5, 15.5),
        }
    ]


def test_make_runtime_geometry_manual_projection_callbacks_project_caked_view() -> None:
    caked_image = np.zeros((6, 6), dtype=float)
    native_background = np.ones((6, 6), dtype=float)
    radial_axis = np.linspace(10.0, 15.0, 6)
    azimuth_axis = np.linspace(-2.0, 3.0, 6)
    two_theta_map = np.tile(radial_axis, (6, 1))
    phi_map = np.tile(azimuth_axis.reshape(-1, 1), (1, 6))
    simulated_param_sets: list[dict[str, object]] = []

    def _simulate_preview_style_peaks_for_fit(
        _miller: np.ndarray,
        _intensities: np.ndarray,
        _image_size: int,
        params: dict[str, object],
    ) -> list[dict[str, object]]:
        simulated_param_sets.append(dict(params))
        return [
            {
                "label": "1,0,0",
                "q_group_key": ("q_group", "primary", 1, 0),
                "source_table_index": 1,
                "source_row_index": 2,
                "sim_col": 3.0,
                "sim_row": 4.0,
            }
        ]

    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: caked_image,
        last_caked_radial_values=lambda: radial_axis,
        last_caked_azimuth_values=lambda: azimuth_axis,
        current_background_display=lambda: np.full((6, 6), 9.0, dtype=float),
        current_background_native=lambda: native_background,
        ai=lambda: object(),
        center=lambda: (0.0, 0.0),
        detector_distance=lambda: 1.0,
        pixel_size=lambda: 1.0,
        wrap_phi_range=lambda value: value,
        current_geometry_fit_params=lambda: {"gamma": 1.5},
        simulate_preview_style_peaks_for_fit=_simulate_preview_style_peaks_for_fit,
        miller=lambda: np.array([[1.0, 0.0, 0.0]], dtype=float),
        intensities=lambda: np.array([2.0], dtype=float),
        image_size=lambda: 6,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
        get_detector_angular_maps=lambda _ai: (two_theta_map, phi_map),
        detector_pixel_to_scattering_angles=lambda *_args: (None, None),
    )

    assert callbacks.pick_uses_caked_space() is True
    assert callbacks.current_background_image() is caked_image
    assert callbacks.entry_display_coords({"x": 2.0, "y": 3.0}) == (12.0, 1.0)
    assert callbacks.caked_angles_to_background_display_coords(13.0, 2.0) == (3.0, 4.0)

    projected = callbacks.simulated_peaks_for_params()

    assert simulated_param_sets == [{"gamma": 1.5}]
    assert projected[0]["caked_x"] == 13.0
    assert projected[0]["caked_y"] == 2.0
    assert projected[0]["sim_col"] == 13.0
    assert projected[0]["sim_row"] == 2.0
    assert projected[0]["sim_col_local"] == 3.0
    assert projected[0]["sim_row_local"] == 4.0

    grouped = callbacks.pick_candidates(projected)
    assert list(grouped) == [("q_group", "primary", 1, 0)]
    assert grouped[("q_group", "primary", 1, 0)][0]["sim_col"] == 13.0

    lookup = callbacks.simulated_lookup(projected)
    assert lookup[(1, 2)]["sim_row"] == 2.0


def test_make_runtime_geometry_manual_projection_callbacks_back_projects_caked_with_inverse_fallback() -> None:
    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.zeros((6, 6), dtype=float),
        last_caked_radial_values=lambda: np.linspace(10.0, 15.0, 6),
        last_caked_azimuth_values=lambda: np.linspace(-2.0, 3.0, 6),
        current_background_display=lambda: np.zeros((6, 6), dtype=float),
        current_background_native=lambda: np.ones((6, 6), dtype=float),
        center=lambda: (20.0, 30.0),
        detector_distance=lambda: 100.0,
        pixel_size=lambda: 0.25,
        image_size=lambda: 6,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
        get_detector_angular_maps=lambda _ai: (None, None),
        detector_pixel_to_scattering_angles=lambda *_args: (_ for _ in ()).throw(
            AssertionError("forward detector->angle conversion should not be used")
        ),
        scattering_angles_to_detector_pixel=lambda two_theta, phi, *_args: (
            phi + 100.0,
            two_theta + 200.0,
        ),
    )

    assert callbacks.caked_angles_to_background_display_coords(13.0, 2.0) == (102.0, 213.0)


def test_make_runtime_geometry_manual_projection_callbacks_back_projects_caked_through_backend_inverse() -> None:
    inverse_calls: list[tuple[float, float]] = []
    two_theta_map = np.full((4, 3), 999.0, dtype=float)
    phi_map = np.full((4, 3), 999.0, dtype=float)
    two_theta_map[1, 0] = 13.0
    phi_map[1, 0] = 2.0

    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.zeros((6, 6), dtype=float),
        last_caked_radial_values=lambda: np.linspace(10.0, 15.0, 6),
        last_caked_azimuth_values=lambda: np.linspace(-2.0, 3.0, 6),
        current_background_display=lambda: np.zeros((3, 4), dtype=float),
        current_background_native=lambda: np.ones((3, 4), dtype=float),
        image_size=lambda: 6,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
        get_detector_angular_maps=lambda _ai: (two_theta_map, phi_map),
        detector_pixel_to_scattering_angles=lambda *_args: (_ for _ in ()).throw(
            AssertionError("forward detector->angle conversion should not be used")
        ),
        backend_detector_coords_to_native_detector_coords=lambda col, row: (
            inverse_calls.append((float(col), float(row))) or (1.5, 2.5)
        ),
        scattering_angles_to_detector_pixel=lambda *_args: (_ for _ in ()).throw(
            AssertionError("analytic inverse fallback should not be used")
        ),
    )

    assert callbacks.caked_angles_to_background_display_coords(13.0, 2.0) == (1.5, 2.5)
    assert inverse_calls == [(0.0, 1.0)]


def test_make_runtime_geometry_manual_projection_callbacks_prefer_cache_uses_live_preview_peaks() -> None:
    simulation_calls: list[dict[str, object]] = []

    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: None,
        last_caked_radial_values=lambda: np.array([], dtype=float),
        last_caked_azimuth_values=lambda: np.array([], dtype=float),
        current_background_display=lambda: np.zeros((6, 6), dtype=float),
        current_background_native=lambda: np.zeros((6, 6), dtype=float),
        current_geometry_fit_params=lambda: {"gamma": 1.5},
        simulate_preview_style_peaks_for_fit=lambda *_args, **_kwargs: (
            simulation_calls.append({"called": True}) or []
        ),
        build_live_preview_simulated_peaks_from_cache=lambda: [
            {
                "label": "1,0,0",
                "q_group_key": ("q_group", "primary", 1, 0),
                "source_table_index": 1,
                "source_row_index": 2,
                "sim_col": 3.0,
                "sim_row": 4.0,
            }
        ],
        miller=lambda: np.array([[1.0, 0.0, 0.0]], dtype=float),
        intensities=lambda: np.array([2.0], dtype=float),
        image_size=lambda: 6,
        display_to_native_sim_coords=lambda col, row, _shape: (float(col), float(row)),
        detector_pixel_to_scattering_angles=lambda *_args: (None, None),
    )

    projected = callbacks.simulated_peaks_for_params(prefer_cache=True)

    assert simulation_calls == []
    assert projected == [
        {
            "label": "1,0,0",
            "q_group_key": ("q_group", "primary", 1, 0),
            "source_table_index": 1,
            "source_row_index": 2,
            "sim_col": 3.0,
            "sim_row": 4.0,
            "sim_col_raw": 3.0,
            "sim_row_raw": 4.0,
        }
    ]


def test_render_current_geometry_manual_pairs_updates_active_session_status() -> None:
    calls: list[tuple[str, object]] = []
    status_messages: list[str] = []

    rendered = mg.render_current_geometry_manual_pairs(
        background_visible=True,
        current_background_index=2,
        current_background_image=np.zeros((5, 5), dtype=float),
        pick_session={
            "group_key": ("q_group", "primary", 1, 0),
            "group_entries": [{"label": "1,0,0"}, {"label": "-1,0,0"}, {"label": "0,0,2"}],
            "pending_entries": [{"label": "1,0,0", "x": 9.0, "y": 10.0}],
            "target_count": 3,
            "q_label": "test group",
            "background_index": 2,
        },
        build_initial_pairs_display=lambda *_args, **_kwargs: (
            [{"overlay_match_index": 0}],
            [{"overlay_match_index": 0, "bg_display": (1.0, 2.0)}],
        ),
        session_initial_pairs_display=lambda: [],
        clear_geometry_pick_artists=lambda **kwargs: calls.append(("clear", kwargs)),
        draw_initial_geometry_pairs_overlay=lambda entries, **kwargs: calls.append(
            ("draw", (list(entries), kwargs.get("max_display_markers")))
        ),
        update_button_label_fn=lambda: calls.append(("button", None)),
        set_background_file_status_text_fn=lambda: calls.append(("background", None)),
        pair_group_count=lambda _idx: 4,
        set_status_text=status_messages.append,
        update_status=True,
    )

    assert rendered is True
    assert ("button", None) in calls
    assert ("background", None) in calls
    assert calls[0][0] == "draw"
    assert "Click background peak 2 of 3 for test group" in status_messages[-1]


def test_geometry_manual_pick_button_label_includes_progress() -> None:
    label = mg.geometry_manual_pick_button_label(
        armed=True,
        current_background_index=0,
        pick_session={
            "group_key": ("q_group", "primary", 1, 0),
            "group_entries": [{"label": "1,0,0"}, {"label": "-1,0,0"}],
            "pending_entries": [{"label": "1,0,0", "x": 9.0, "y": 10.0}],
            "target_count": 2,
            "background_index": 0,
        },
        pairs_for_index=lambda _idx: [{"x": 1.0, "y": 2.0}, {"x": 3.0, "y": 4.0}],
        pair_group_count=lambda _idx: 1,
    )

    assert label == "Pick Qr Sets on Image (Armed) [1 groups/2 pts] <placing 1/2>"


def test_geometry_manual_toggle_selection_at_starts_session() -> None:
    set_sessions: list[dict[str, object]] = []
    status_messages: list[str] = []
    calls: list[tuple[str, object]] = []
    group_key = ("q_group", "primary", 1, 0)

    handled, next_session, suppress_drag = mg.geometry_manual_toggle_selection_at(
        10.0,
        20.0,
        pick_session={},
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {
            "signature": ("cache",),
            "grouped_candidates": {
                group_key: [
                    {
                        "label": "1,0,0",
                        "hkl": (1, 0, 0),
                        "sim_col": 10.0,
                        "sim_row": 20.0,
                        "source_table_index": 1,
                        "source_row_index": 2,
                    }
                ]
            },
        },
        pairs_for_index=lambda _idx: [],
        set_pairs_for_index_fn=lambda _idx, entries: list(entries or []),
        set_pick_session_fn=lambda session: set_sessions.append(dict(session)),
        restore_view_fn=lambda **kwargs: calls.append(("restore", kwargs.get("redraw"))),
        clear_preview_artists_fn=lambda **kwargs: calls.append(("clear", kwargs.get("redraw"))),
        render_current_pairs_fn=lambda **kwargs: calls.append(("render", kwargs.get("update_status"))),
        update_button_label_fn=lambda: calls.append(("button", None)),
        set_status_text=status_messages.append,
        listed_q_group_entries=lambda: [{"key": group_key}],
        format_q_group_line=lambda _entry: "selected group",
        use_caked_space=False,
        pick_search_window_px=50.0,
    )

    assert handled is True
    assert suppress_drag is True
    assert next_session["group_key"] == group_key
    assert next_session["target_count"] == 1
    assert set_sessions[-1]["group_key"] == group_key
    assert ("render", False) in calls
    assert ("button", None) in calls
    assert "Selected selected group" in status_messages[-1]


def test_geometry_manual_toggle_selection_at_starts_two_peak_session_for_non_00l_group() -> None:
    set_sessions: list[dict[str, object]] = []
    status_messages: list[str] = []
    calls: list[tuple[str, object]] = []
    group_key = ("q_group", "primary", 1, 2)

    handled, next_session, suppress_drag = mg.geometry_manual_toggle_selection_at(
        10.0,
        20.0,
        pick_session={},
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {
            "signature": ("cache",),
            "grouped_candidates": {
                group_key: [
                    {
                        "label": "1,0,2",
                        "hkl": (1, 0, 2),
                        "sim_col": 10.0,
                        "sim_row": 20.0,
                        "source_table_index": 1,
                        "source_row_index": 2,
                    },
                    {
                        "label": "-1,0,2",
                        "hkl": (-1, 0, 2),
                        "sim_col": 30.0,
                        "sim_row": 40.0,
                        "source_table_index": 1,
                        "source_row_index": 3,
                    },
                ]
            },
        },
        pairs_for_index=lambda _idx: [],
        set_pairs_for_index_fn=lambda _idx, entries: list(entries or []),
        set_pick_session_fn=lambda session: set_sessions.append(dict(session)),
        restore_view_fn=lambda **kwargs: calls.append(("restore", kwargs.get("redraw"))),
        clear_preview_artists_fn=lambda **kwargs: calls.append(("clear", kwargs.get("redraw"))),
        render_current_pairs_fn=lambda **kwargs: calls.append(("render", kwargs.get("update_status"))),
        update_button_label_fn=lambda: calls.append(("button", None)),
        set_status_text=status_messages.append,
        listed_q_group_entries=lambda: [{"key": group_key}],
        format_q_group_line=lambda _entry: "selected group",
        use_caked_space=False,
        pick_search_window_px=50.0,
    )

    assert handled is True
    assert suppress_drag is True
    assert next_session["group_key"] == group_key
    assert next_session["target_count"] == 2
    assert set_sessions[-1]["target_count"] == 2
    assert ("render", False) in calls
    assert ("button", None) in calls
    assert "Click background peak 1 of 2" in status_messages[-1]


def test_geometry_manual_toggle_selection_at_tags_clicked_seed_within_group() -> None:
    set_sessions: list[dict[str, object]] = []
    status_messages: list[str] = []
    group_key = ("q_group", "primary", 1, 2)

    handled, next_session, suppress_drag = mg.geometry_manual_toggle_selection_at(
        29.5,
        39.5,
        pick_session={},
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {
            "signature": ("cache",),
            "grouped_candidates": {
                group_key: [
                    {
                        "label": "1,0,2",
                        "hkl": (1, 0, 2),
                        "sim_col": 10.0,
                        "sim_row": 20.0,
                        "source_table_index": 1,
                        "source_row_index": 2,
                    },
                    {
                        "label": "-1,0,2",
                        "hkl": (-1, 0, 2),
                        "sim_col": 30.0,
                        "sim_row": 40.0,
                        "source_table_index": 1,
                        "source_row_index": 3,
                    },
                ]
            },
        },
        pairs_for_index=lambda _idx: [],
        set_pairs_for_index_fn=lambda _idx, entries: list(entries or []),
        set_pick_session_fn=lambda session: set_sessions.append(dict(session)),
        restore_view_fn=lambda **_kwargs: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=status_messages.append,
        listed_q_group_entries=lambda: [{"key": group_key}],
        format_q_group_line=lambda _entry: "selected group",
        use_caked_space=False,
        pick_search_window_px=50.0,
    )

    assert handled is True
    assert suppress_drag is True
    assert next_session["tagged_candidate_key"] == ("source", 1, 3)
    assert next_session["group_entries"][0]["source_row_index"] == 3
    assert next_session["tagged_candidate"]["label"] == "-1,0,2"
    assert "Tagged central-beam seed [-1,0,2]" in status_messages[-1]
    assert set_sessions[-1]["tagged_candidate_key"] == ("source", 1, 3)


def test_geometry_manual_place_selection_at_saves_completed_group() -> None:
    set_sessions: list[dict[str, object]] = []
    saved_entry_sets: list[list[dict[str, object]]] = []
    status_messages: list[str] = []
    calls: list[tuple[str, object]] = []

    handled, next_session = mg.geometry_manual_place_selection_at(
        4.8,
        5.9,
        pick_session={
            "group_key": ("q_group", "primary", 1, 0),
            "group_entries": [
                {
                    "label": "1,0,0",
                    "hkl": (1, 0, 0),
                    "sim_col": 5.0,
                    "sim_row": 6.0,
                    "source_table_index": 1,
                    "source_row_index": 2,
                }
            ],
            "pending_entries": [],
            "target_count": 1,
            "base_entries": [{"label": "kept", "x": 1.0, "y": 2.0}],
            "q_label": "selected group",
            "background_index": 0,
            "zoom_active": True,
            "zoom_center": (5.0, 6.0),
            "saved_xlim": (0.0, 10.0),
            "saved_ylim": (0.0, 10.0),
        },
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {},
        refine_preview_point=lambda *_args, **_kwargs: (5.0, 6.0),
        set_pairs_for_index_fn=lambda _idx, entries: saved_entry_sets.append(list(entries or [])) or list(entries or []),
        set_pick_session_fn=lambda session: set_sessions.append(dict(session)),
        clear_preview_artists_fn=lambda **kwargs: calls.append(("clear", kwargs.get("redraw"))),
        restore_view_fn=lambda **kwargs: calls.append(("restore", kwargs.get("redraw"))),
        render_current_pairs_fn=lambda **kwargs: calls.append(("render", kwargs.get("update_status"))),
        update_button_label_fn=lambda: calls.append(("button", None)),
        set_status_text=status_messages.append,
        push_undo_state_fn=lambda: calls.append(("undo", None)),
        use_caked_space=False,
    )

    assert handled is True
    assert next_session == {}
    assert set_sessions[-1] == {}
    assert ("undo", None) in calls
    assert ("clear", False) in calls
    assert ("restore", False) in calls
    assert ("render", False) in calls
    assert saved_entry_sets
    assert saved_entry_sets[-1][0]["label"] == "kept"
    assert saved_entry_sets[-1][1]["label"] == "1,0,0"
    assert saved_entry_sets[-1][1]["q_group_key"] == ("q_group", "primary", 1, 0)
    assert saved_entry_sets[-1][1]["placement_error_px"] > 0.0
    assert "Saved 1 manual background points for selected group" in status_messages[-1]


def test_geometry_manual_place_selection_at_uses_tagged_candidate_first() -> None:
    set_sessions: list[dict[str, object]] = []
    status_messages: list[str] = []

    handled, next_session = mg.geometry_manual_place_selection_at(
        11.8,
        14.2,
        pick_session={
            "group_key": ("q_group", "primary", 1, 2),
            "group_entries": [
                {
                    "label": "left",
                    "hkl": (1, 0, 2),
                    "sim_col": 35.0,
                    "sim_row": 30.0,
                    "source_table_index": 1,
                    "source_row_index": 2,
                },
                {
                    "label": "right",
                    "hkl": (-1, 0, 2),
                    "sim_col": 12.0,
                    "sim_row": 14.0,
                    "source_table_index": 1,
                    "source_row_index": 3,
                },
            ],
            "pending_entries": [],
            "target_count": 2,
            "base_entries": [],
            "q_label": "selected group",
            "background_index": 0,
            "tagged_candidate_key": ("source", 1, 2),
            "zoom_active": True,
            "zoom_center": (12.0, 14.0),
            "saved_xlim": (0.0, 40.0),
            "saved_ylim": (0.0, 40.0),
        },
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {},
        refine_preview_point=lambda *_args, **_kwargs: (12.0, 14.0),
        set_pairs_for_index_fn=lambda _idx, entries: list(entries or []),
        set_pick_session_fn=lambda session: set_sessions.append(dict(session)),
        clear_preview_artists_fn=lambda **_kwargs: None,
        restore_view_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=status_messages.append,
        push_undo_state_fn=lambda: None,
        use_caked_space=False,
    )

    assert handled is True
    assert next_session["pending_entries"][0]["label"] == "left"
    assert next_session["pending_entries"][0]["source_row_index"] == 2
    assert "Assigned to left" in status_messages[-1]
    assert set_sessions[-1]["pending_entries"][0]["source_row_index"] == 2


def test_geometry_manual_place_selection_at_back_projects_caked_pick_to_detector_space() -> None:
    set_sessions: list[dict[str, object]] = []
    saved_entry_sets: list[list[dict[str, object]]] = []
    status_messages: list[str] = []
    calls: list[tuple[str, object]] = []

    handled, next_session = mg.geometry_manual_place_selection_at(
        13.0,
        2.0,
        pick_session={
            "group_key": ("q_group", "primary", 1, 0),
            "group_entries": [
                {
                    "label": "1,0,0",
                    "hkl": (1, 0, 0),
                    "sim_col": 13.2,
                    "sim_row": 2.5,
                    "source_table_index": 1,
                    "source_row_index": 2,
                }
            ],
            "pending_entries": [],
            "target_count": 1,
            "base_entries": [],
            "q_label": "selected group",
            "background_index": 0,
            "zoom_active": True,
            "zoom_center": (13.0, 2.0),
            "saved_xlim": (10.0, 16.0),
            "saved_ylim": (-4.0, 4.0),
        },
        current_background_index=0,
        display_background=np.zeros((8, 8), dtype=float),
        get_cache_data=lambda **_kwargs: {},
        refine_preview_point=lambda *_args, **_kwargs: (13.2, 2.5),
        set_pairs_for_index_fn=lambda _idx, entries: saved_entry_sets.append(list(entries or [])) or list(entries or []),
        set_pick_session_fn=lambda session: set_sessions.append(dict(session)),
        clear_preview_artists_fn=lambda **kwargs: calls.append(("clear", kwargs.get("redraw"))),
        restore_view_fn=lambda **kwargs: calls.append(("restore", kwargs.get("redraw"))),
        render_current_pairs_fn=lambda **kwargs: calls.append(("render", kwargs.get("update_status"))),
        update_button_label_fn=lambda: calls.append(("button", None)),
        set_status_text=status_messages.append,
        push_undo_state_fn=lambda: calls.append(("undo", None)),
        use_caked_space=True,
        caked_angles_to_background_display_coords=lambda two_theta, phi: (
            phi + 100.0,
            two_theta + 200.0,
        ),
        background_display_to_native_detector_coords=lambda col, row: (
            float(col) - 1.0,
            float(row) - 2.0,
        ),
    )

    assert handled is True
    assert next_session == {}
    assert set_sessions[-1] == {}
    assert ("undo", None) in calls
    assert saved_entry_sets
    assert saved_entry_sets[-1][0]["x"] == 102.5
    assert saved_entry_sets[-1][0]["y"] == 213.2
    assert saved_entry_sets[-1][0]["detector_x"] == 101.5
    assert saved_entry_sets[-1][0]["detector_y"] == 211.2
    assert saved_entry_sets[-1][0]["caked_x"] == 13.2
    assert saved_entry_sets[-1][0]["caked_y"] == 2.5
    assert saved_entry_sets[-1][0]["raw_caked_x"] == 13.0
    assert saved_entry_sets[-1][0]["raw_caked_y"] == 2.0
    assert saved_entry_sets[-1][0]["placement_error_px"] > 0.0
    assert "Saved 1 manual background points for selected group" in status_messages[-1]
