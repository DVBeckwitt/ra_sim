from ra_sim.gui import controllers, state


def test_app_state_has_isolated_manual_geometry_state() -> None:
    app_state = state.AppState()
    other_state = state.AppState()

    assert isinstance(app_state.manual_geometry, state.ManualGeometryState)
    assert isinstance(app_state.geometry_fit_history, state.GeometryFitHistoryState)
    assert isinstance(app_state.geometry_preview, state.GeometryPreviewState)
    assert isinstance(
        app_state.geometry_preview.overlay,
        state.GeometryPreviewOverlayState,
    )
    assert isinstance(app_state.geometry_q_groups, state.GeometryQGroupState)
    assert isinstance(app_state.geometry_q_group_view, state.GeometryQGroupViewState)
    assert app_state.manual_geometry is not other_state.manual_geometry
    assert app_state.geometry_fit_history is not other_state.geometry_fit_history
    assert app_state.geometry_preview is not other_state.geometry_preview
    assert app_state.geometry_preview.overlay is not other_state.geometry_preview.overlay
    assert app_state.geometry_q_groups is not other_state.geometry_q_groups
    assert app_state.geometry_q_group_view is not other_state.geometry_q_group_view

    app_state.manual_geometry.pick_session["group_key"] = ("q_group", "primary", 1, 0)
    app_state.geometry_preview.skip_once = True
    app_state.geometry_preview.overlay.pairs.append({"x": 1.0})
    app_state.geometry_q_groups.refresh_requested = True
    assert other_state.manual_geometry.pick_session == {}
    assert other_state.geometry_preview.skip_once is False
    assert other_state.geometry_preview.overlay.pairs == []
    assert other_state.geometry_q_groups.refresh_requested is False


def test_replace_manual_geometry_state_updates_in_place() -> None:
    manual_state = state.ManualGeometryState()
    pairs_alias = manual_state.pairs_by_background
    session_alias = manual_state.pick_session

    source_pairs = {
        "1": [
            {"label": "1,0,0", "x": 5.0, "y": 6.0},
        ]
    }
    controllers.replace_manual_geometry_pairs_by_background(
        manual_state,
        source_pairs,
    )
    controllers.replace_manual_geometry_pick_session(
        manual_state,
        {"background_index": 1, "group_key": ("q_group", "primary", 1, 0)},
    )

    assert manual_state.pairs_by_background is pairs_alias
    assert manual_state.pick_session is session_alias
    assert pairs_alias == {1: [{"label": "1,0,0", "x": 5.0, "y": 6.0}]}
    assert session_alias["background_index"] == 1

    source_pairs["1"][0]["x"] = 99.0
    assert manual_state.pairs_by_background[1][0]["x"] == 5.0


def test_manual_geometry_undo_controller_restores_latest_snapshot_and_limit() -> None:
    manual_state = state.ManualGeometryState()

    controllers.replace_manual_geometry_pairs_by_background(
        manual_state,
        {0: [{"label": "a", "x": 1.0, "y": 2.0}]},
    )
    controllers.replace_manual_geometry_pick_session(
        manual_state,
        {"group_key": "a"},
    )
    controllers.push_manual_geometry_undo_state(manual_state, limit=2)

    controllers.replace_manual_geometry_pairs_by_background(
        manual_state,
        {1: [{"label": "b", "x": 3.0, "y": 4.0}]},
    )
    controllers.replace_manual_geometry_pick_session(
        manual_state,
        {"group_key": "b"},
    )
    controllers.push_manual_geometry_undo_state(manual_state, limit=2)

    controllers.replace_manual_geometry_pairs_by_background(
        manual_state,
        {2: [{"label": "c", "x": 5.0, "y": 6.0}]},
    )
    controllers.replace_manual_geometry_pick_session(
        manual_state,
        {"group_key": "c"},
    )
    controllers.push_manual_geometry_undo_state(manual_state, limit=2)

    assert len(manual_state.undo_stack) == 2

    controllers.replace_manual_geometry_pairs_by_background(
        manual_state,
        {9: [{"label": "live", "x": 9.0, "y": 10.0}]},
    )
    controllers.replace_manual_geometry_pick_session(
        manual_state,
        {"group_key": "live"},
    )

    restored = controllers.restore_last_manual_geometry_undo_state(manual_state)
    assert restored is not None
    assert manual_state.pick_session["group_key"] == "c"
    assert manual_state.pairs_by_background == {2: [{"label": "c", "x": 5.0, "y": 6.0}]}

    restored = controllers.restore_last_manual_geometry_undo_state(manual_state)
    assert restored is not None
    assert manual_state.pick_session["group_key"] == "b"
    assert manual_state.pairs_by_background == {1: [{"label": "b", "x": 3.0, "y": 4.0}]}

    assert controllers.restore_last_manual_geometry_undo_state(manual_state) is None


def test_clear_manual_geometry_undo_stack_discards_history() -> None:
    manual_state = state.ManualGeometryState()
    controllers.push_manual_geometry_undo_state(manual_state, limit=2)

    assert len(manual_state.undo_stack) == 1
    controllers.clear_manual_geometry_undo_stack(manual_state)
    assert manual_state.undo_stack == []


def test_geometry_fit_history_controller_tracks_overlay_and_undo_redo() -> None:
    fit_state = state.GeometryFitHistoryState()

    controllers.replace_geometry_fit_last_overlay_state(
        fit_state,
        {"overlay_records": [{"x": 1.0}], "max_display_markers": 4},
    )
    assert fit_state.last_overlay_state == {
        "overlay_records": [{"x": 1.0}],
        "max_display_markers": 4,
    }

    controllers.push_geometry_fit_undo_state(
        fit_state,
        {"ui_params": {"gamma": 1.0}},
        copy_state_value=lambda value: dict(value),
        limit=2,
    )
    controllers.push_geometry_fit_undo_state(
        fit_state,
        {"ui_params": {"gamma": 2.0}},
        copy_state_value=lambda value: dict(value),
        limit=2,
    )
    controllers.push_geometry_fit_undo_state(
        fit_state,
        {"ui_params": {"gamma": 3.0}},
        copy_state_value=lambda value: dict(value),
        limit=2,
    )

    assert len(fit_state.undo_stack) == 2
    assert controllers.peek_last_geometry_fit_undo_state(
        fit_state,
        copy_state_value=lambda value: dict(value),
    ) == {"ui_params": {"gamma": 3.0}}

    controllers.commit_geometry_fit_undo(
        fit_state,
        {"ui_params": {"gamma": 9.0}},
        copy_state_value=lambda value: dict(value),
        limit=2,
    )
    assert len(fit_state.undo_stack) == 1
    assert fit_state.redo_stack == [{"ui_params": {"gamma": 9.0}}]

    assert controllers.peek_last_geometry_fit_redo_state(
        fit_state,
        copy_state_value=lambda value: dict(value),
    ) == {"ui_params": {"gamma": 9.0}}

    controllers.commit_geometry_fit_redo(
        fit_state,
        {"ui_params": {"gamma": 5.0}},
        copy_state_value=lambda value: dict(value),
        limit=2,
    )
    assert fit_state.redo_stack == []
    assert fit_state.undo_stack[-1] == {"ui_params": {"gamma": 5.0}}

    controllers.clear_geometry_fit_history(fit_state)
    assert fit_state.undo_stack == []
    assert fit_state.redo_stack == []
    assert fit_state.last_overlay_state is None


def test_geometry_preview_controller_tracks_exclusions_skip_flag_and_cache() -> None:
    preview_state = state.GeometryPreviewState()
    excluded_keys_alias = preview_state.excluded_keys
    excluded_alias = preview_state.excluded_q_groups

    controllers.replace_geometry_preview_excluded_q_groups(
        preview_state,
        [
            ("q_group", "primary", 1, 0),
            ["q_group", "secondary", 2, 1],
        ],
    )
    assert preview_state.excluded_q_groups is excluded_alias
    assert preview_state.excluded_q_groups == {
        ("q_group", "primary", 1, 0),
        ("q_group", "secondary", 2, 1),
    }

    controllers.retain_geometry_preview_excluded_q_groups(
        preview_state,
        {("q_group", "secondary", 2, 1)},
    )
    assert preview_state.excluded_q_groups == {("q_group", "secondary", 2, 1)}
    assert controllers.count_geometry_preview_excluded_q_groups(preview_state) == 1
    assert (
        controllers.count_geometry_preview_excluded_q_groups(
            preview_state,
            [
                ("q_group", "secondary", 2, 1),
                ("q_group", "primary", 1, 0),
            ],
        )
        == 1
    )

    controllers.set_geometry_preview_q_group_included(
        preview_state,
        ("q_group", "secondary", 2, 1),
        included=True,
    )
    controllers.set_geometry_preview_q_group_included(
        preview_state,
        ("q_group", "primary", 1, 0),
        included=False,
    )
    assert preview_state.excluded_q_groups == {("q_group", "primary", 1, 0)}
    assert preview_state.excluded_keys is excluded_keys_alias

    controllers.set_geometry_preview_match_included(
        preview_state,
        ("hkl", 1, 0, 0),
        included=False,
    )
    assert preview_state.excluded_keys == {("hkl", 1, 0, 0)}
    controllers.set_geometry_preview_match_included(
        preview_state,
        ("hkl", 1, 0, 0),
        included=True,
    )
    assert preview_state.excluded_keys == set()
    controllers.set_geometry_preview_exclude_mode(preview_state, True)
    assert preview_state.exclude_armed is True
    controllers.set_geometry_preview_exclude_mode(preview_state, False)
    assert preview_state.exclude_armed is False

    overlay_alias = preview_state.overlay
    pairs_alias = preview_state.overlay.pairs
    attempts_alias = preview_state.overlay.auto_match_attempts
    controllers.replace_geometry_preview_overlay_state(
        preview_state,
        {
            "signature": ("sig", 1),
            "pairs": [{"x": 1.0, "y": 2.0}],
            "simulated_count": 8,
            "min_matches": 6,
            "best_radius": 12.5,
            "mean_dist": 3.0,
            "p90_dist": 4.5,
            "quality_fail": True,
            "max_display_markers": 5,
            "auto_match_attempts": [{"name": "attempt-a"}],
            "q_group_total": 10,
            "q_group_excluded": 2,
            "excluded_q_peaks": 4,
            "collapsed_degenerate_peaks": 1,
        },
    )
    assert preview_state.overlay is overlay_alias
    assert preview_state.overlay.pairs is pairs_alias
    assert preview_state.overlay.auto_match_attempts is attempts_alias
    assert preview_state.overlay.signature == ("sig", 1)
    assert preview_state.overlay.pairs == [{"x": 1.0, "y": 2.0}]
    assert preview_state.overlay.simulated_count == 8
    assert preview_state.overlay.min_matches == 6
    assert preview_state.overlay.best_radius == 12.5
    assert preview_state.overlay.quality_fail is True
    assert preview_state.overlay.max_display_markers == 5
    assert preview_state.overlay.auto_match_attempts == [{"name": "attempt-a"}]
    assert preview_state.overlay.q_group_total == 10
    assert preview_state.overlay.q_group_excluded == 2
    assert preview_state.overlay.excluded_q_peaks == 4
    assert preview_state.overlay.collapsed_degenerate_peaks == 1

    controllers.request_geometry_preview_skip_once(preview_state)
    assert preview_state.skip_once is True
    assert controllers.consume_geometry_preview_skip_once(preview_state) is True
    assert preview_state.skip_once is False
    assert controllers.consume_geometry_preview_skip_once(preview_state) is False
    controllers.request_geometry_preview_skip_once(preview_state)
    controllers.clear_geometry_preview_skip_once(preview_state)
    assert preview_state.skip_once is False

    cache_key = ("bg", 1, 2, 3)
    cache_data = {"summit_records": [1, 2, 3]}
    controllers.replace_geometry_auto_match_background_cache(
        preview_state,
        cache_key,
        cache_data,
    )
    assert (
        controllers.get_geometry_auto_match_background_cache(preview_state, cache_key)
        is cache_data
    )
    assert (
        controllers.get_geometry_auto_match_background_cache(
            preview_state,
            ("bg", 9),
        )
        is None
    )
    controllers.clear_geometry_auto_match_background_cache(preview_state)
    assert preview_state.auto_match_background_cache_key is None
    assert preview_state.auto_match_background_cache_data is None
    controllers.clear_geometry_preview_excluded_keys(preview_state)
    assert preview_state.excluded_keys == set()
    controllers.clear_geometry_preview_excluded_q_groups(preview_state)
    assert preview_state.excluded_q_groups == set()


def test_geometry_q_group_controller_replaces_entries_row_vars_and_refresh_flag() -> None:
    q_state = state.GeometryQGroupState()
    row_vars_alias = q_state.row_vars
    cached_alias = q_state.cached_entries

    listed = controllers.replace_geometry_q_group_cached_entries(
        q_state,
        [
            {
                "key": ("q_group", "primary", 1, 0),
                "hkl_preview": [(1, 0, 0)],
                "peak_count": 2,
            }
        ],
    )

    assert q_state.row_vars is row_vars_alias
    assert q_state.cached_entries is cached_alias
    assert listed == [
        {
            "key": ("q_group", "primary", 1, 0),
            "hkl_preview": [(1, 0, 0)],
            "peak_count": 2,
        }
    ]
    assert controllers.listed_geometry_q_group_keys(q_state) == {
        ("q_group", "primary", 1, 0)
    }

    controllers.set_geometry_q_group_row_var(
        q_state,
        ("q_group", "primary", 1, 0),
        "var-a",
    )
    assert q_state.row_vars == {("q_group", "primary", 1, 0): "var-a"}
    controllers.clear_geometry_q_group_row_vars(q_state)
    assert q_state.row_vars == {}

    controllers.request_geometry_q_group_refresh(q_state)
    assert q_state.refresh_requested is True
    assert controllers.consume_geometry_q_group_refresh_request(q_state) is True
    assert q_state.refresh_requested is False
    assert controllers.consume_geometry_q_group_refresh_request(q_state) is False
