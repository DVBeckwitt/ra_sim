from ra_sim.gui import controllers, state


def test_app_state_has_isolated_manual_geometry_state() -> None:
    app_state = state.AppState()
    other_state = state.AppState()

    assert isinstance(app_state.manual_geometry, state.ManualGeometryState)
    assert app_state.manual_geometry is not other_state.manual_geometry

    app_state.manual_geometry.pick_session["group_key"] = ("q_group", "primary", 1, 0)
    assert other_state.manual_geometry.pick_session == {}


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
