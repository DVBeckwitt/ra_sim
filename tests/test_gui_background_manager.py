from pathlib import Path

import numpy as np

from ra_sim.gui import background_manager, state


def test_background_manager_apply_update_preserves_list_aliases() -> None:
    background_state = state.BackgroundRuntimeState()
    osc_alias = background_state.osc_files
    native_alias = background_state.background_images_native

    background_manager.apply_background_state_update(
        background_state,
        {
            "osc_files": ["a.osc", "b.osc"],
            "background_images": [np.zeros((2, 2)), np.ones((2, 2))],
            "background_images_native": [np.zeros((2, 2)), np.ones((2, 2))],
            "background_images_display": [np.zeros((2, 2)), np.ones((2, 2))],
            "current_background_index": 1,
            "current_background_image": np.ones((2, 2)),
            "current_background_display": np.ones((2, 2)),
        },
    )

    assert background_state.osc_files is osc_alias
    assert background_state.background_images_native is native_alias
    assert background_state.osc_files == ["a.osc", "b.osc"]
    assert background_state.current_background_index == 1


def test_background_manager_load_and_switch_update_state_in_place(tmp_path) -> None:
    path_a = tmp_path / "a.osc"
    path_b = tmp_path / "b.osc"
    path_a.write_text("", encoding="utf-8")
    path_b.write_text("", encoding="utf-8")
    background_state = state.BackgroundRuntimeState()

    def _fake_read_osc(path: str):
        if path == str(path_a):
            return np.zeros((2, 2))
        return np.ones((2, 2))

    background_manager.load_background_files(
        background_state,
        [str(path_a), str(path_b)],
        image_size=2,
        display_rotate_k=-1,
        read_osc=_fake_read_osc,
        select_index=0,
    )
    switched = background_manager.switch_background(
        background_state,
        display_rotate_k=-1,
        read_osc=_fake_read_osc,
    )

    assert background_state.osc_files == [str(path_a), str(path_b)]
    assert background_state.current_background_index == 1
    assert np.array_equal(background_state.current_background_image, np.ones((2, 2)))
    assert np.array_equal(switched["current_background_display"], np.rot90(np.ones((2, 2)), -1))


def test_background_manager_builds_background_status_text() -> None:
    text = background_manager.build_background_file_status_text(
        osc_files=["C:/data/a.osc", "C:/data/b.osc"],
        current_background_index=1,
        theta_base=12.34567,
        theta_effective=12.84567,
        use_shared_theta_offset=True,
        pair_count=3,
        group_count=2,
        sigma_values=[1.0, float("nan"), 3.0],
        fit_indices=[0, 1],
    )

    assert text.startswith("Background 2/2: b.osc")
    assert "theta_i=12.3457° | theta=12.8457°" in text
    assert "manual=2 groups/3 pts" in text
    assert "sigma~2.00px" in text
    assert "fit=2 backgrounds" in text

    assert (
        background_manager.build_background_file_status_text(
            osc_files=[],
            current_background_index=0,
        )
        == "Background: no files loaded"
    )


def test_background_manager_status_refresh_reads_runtime_state(monkeypatch) -> None:
    captured = {}
    view_state = state.WorkspacePanelsViewState(background_file_status_var=object())
    background_state = state.BackgroundRuntimeState(
        osc_files=["C:/data/a.osc", "C:/data/b.osc"],
        current_background_index=1,
    )

    monkeypatch.setattr(
        background_manager.gui_views,
        "set_background_file_status_text",
        lambda view_state_arg, text: captured.update(
            {"view_state": view_state_arg, "text": text}
        ),
    )

    text = background_manager.set_background_file_status_from_state(
        view_state=view_state,
        background_state=background_state,
        current_background_theta_values=lambda: [12.0, 13.5],
        background_theta_for_index=lambda idx: 14.25 + idx * 0.0,
        geometry_fit_uses_shared_theta_offset=lambda: True,
        geometry_manual_pairs_for_index=lambda idx: [
            {"sigma_px": 1.0},
            {"sigma_px": 3.0},
            {"sigma_px": "bad"},
        ],
        geometry_manual_pair_group_count=lambda idx: 2,
        current_geometry_fit_background_indices=lambda: [1],
    )

    assert captured == {
        "view_state": view_state,
        "text": text,
    }
    assert text.startswith("Background 2/2: b.osc")
    assert "theta_i=13.5000° | theta=14.2500°" in text
    assert "manual=2 groups/3 pts" in text
    assert "sigma~2.00px" in text
    assert "fit=bg 2" in text


def test_background_manager_load_workflow_runs_follow_on_side_effects(
    monkeypatch,
) -> None:
    background_state = state.BackgroundRuntimeState()
    events = []

    def _fake_load_background_files(
        background_state_arg,
        file_paths,
        *,
        image_size,
        display_rotate_k,
        read_osc,
        select_index,
    ):
        events.append(
            (
                "load",
                tuple(file_paths),
                image_size,
                display_rotate_k,
                select_index,
            )
        )
        background_state_arg.current_background_display = "loaded-display"
        return {"current_background_display": "loaded-display"}

    monkeypatch.setattr(
        background_manager,
        "load_background_files",
        _fake_load_background_files,
    )

    updated = background_manager.load_background_files_with_side_effects(
        background_state,
        ["a.osc", "b.osc"],
        image_size=2048,
        display_rotate_k=-1,
        read_osc=lambda path: path,
        sync_background_runtime_state=lambda: events.append(("sync",)),
        replace_geometry_manual_pairs_by_background=lambda payload: events.append(
            ("replace_pairs", payload)
        ),
        invalidate_geometry_manual_pick_cache=lambda: events.append(("invalidate",)),
        clear_geometry_manual_undo_stack=lambda: events.append(("clear_manual_undo",)),
        clear_geometry_fit_undo_stack=lambda: events.append(("clear_fit_undo",)),
        set_geometry_manual_pick_mode=lambda enabled: events.append(
            ("pick_mode", enabled)
        ),
        set_background_display_data=lambda image: events.append(("display", image)),
        update_background_slider_defaults=lambda image: events.append(
            ("slider_defaults", image)
        ),
        sync_background_theta_controls=lambda: events.append(("sync_theta",)),
        sync_geometry_fit_background_selection=lambda: events.append(
            ("sync_fit_selection",)
        ),
        clear_geometry_pick_artists=lambda: events.append(("clear_pick_artists",)),
        refresh_background_file_status=lambda: events.append(("refresh_status",)),
        schedule_update=lambda: events.append(("schedule_update",)),
        select_index=1,
    )

    assert updated == {"current_background_display": "loaded-display"}
    assert events == [
        ("load", ("a.osc", "b.osc"), 2048, -1, 1),
        ("sync",),
        ("replace_pairs", {}),
        ("invalidate",),
        ("clear_manual_undo",),
        ("clear_fit_undo",),
        ("pick_mode", False),
        ("display", "loaded-display"),
        ("slider_defaults", "loaded-display"),
        ("sync_theta",),
        ("sync_fit_selection",),
        ("clear_pick_artists",),
        ("refresh_status",),
        ("schedule_update",),
    ]


def test_background_manager_switch_workflow_runs_follow_on_side_effects(
    monkeypatch,
) -> None:
    background_state = state.BackgroundRuntimeState(current_background_index=1)
    events = []

    def _fake_switch_background(
        background_state_arg,
        *,
        display_rotate_k,
        read_osc,
    ):
        events.append(("switch", display_rotate_k))
        background_state_arg.current_background_display = "switched-display"
        return {"current_background_display": "switched-display"}

    monkeypatch.setattr(
        background_manager,
        "switch_background",
        _fake_switch_background,
    )

    updated = background_manager.switch_background_with_side_effects(
        background_state,
        display_rotate_k=-1,
        read_osc=lambda path: path,
        sync_background_runtime_state=lambda: events.append(("sync",)),
        invalidate_geometry_manual_pick_cache=lambda: events.append(("invalidate",)),
        clear_geometry_manual_undo_stack=lambda: events.append(("clear_manual_undo",)),
        clear_geometry_fit_undo_stack=lambda: events.append(("clear_fit_undo",)),
        sync_theta_initial_to_background=lambda idx: events.append(
            ("sync_theta_initial", idx)
        ),
        set_background_display_data=lambda image: events.append(("display", image)),
        update_background_slider_defaults=lambda image: events.append(
            ("slider_defaults", image)
        ),
        refresh_background_file_status=lambda: events.append(("refresh_status",)),
        render_current_geometry_manual_pairs=lambda: events.append(
            ("render_manual_pairs",)
        ),
        schedule_update=lambda: events.append(("schedule_update",)),
    )

    assert updated == {"current_background_display": "switched-display"}
    assert events == [
        ("switch", -1),
        ("sync",),
        ("invalidate",),
        ("clear_manual_undo",),
        ("clear_fit_undo",),
        ("sync_theta_initial", 1),
        ("display", "switched-display"),
        ("slider_defaults", "switched-display"),
        ("refresh_status",),
        ("render_manual_pairs",),
        ("schedule_update",),
    ]


def test_background_manager_chooses_background_dialog_initial_dir(tmp_path) -> None:
    current_path = tmp_path / "nested" / "a.osc"
    current_path.parent.mkdir()
    current_path.write_text("", encoding="utf-8")

    initial_dir = background_manager.background_file_dialog_initial_dir(
        [str(current_path)],
        0,
        tmp_path / "fallback",
    )

    assert initial_dir == str(current_path.parent)
    assert background_manager.background_file_dialog_initial_dir(
        ["bad-path"],
        99,
        Path("fallback"),
    ) == "fallback"


def test_background_manager_runtime_binding_factory_builds_live_bindings(
    monkeypatch,
) -> None:
    calls = []
    counters = {"status": 0, "schedule": 0, "dir": 0}

    monkeypatch.setattr(
        background_manager,
        "BackgroundRuntimeBindings",
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

    factory = background_manager.make_runtime_background_bindings_factory(
        view_state="view-state",
        background_state="background-state",
        image_size=2048,
        display_rotate_k=-1,
        read_osc=lambda path: path,
        current_background_theta_values=lambda: [1.0, 2.0],
        background_theta_for_index=lambda idx: idx,
        geometry_fit_uses_shared_theta_offset=lambda: False,
        geometry_manual_pairs_for_index=lambda idx: [],
        geometry_manual_pair_group_count=lambda idx: 0,
        current_geometry_fit_background_indices=lambda: [0],
        sync_background_runtime_state=lambda: None,
        replace_geometry_manual_pairs_by_background=lambda payload: None,
        invalidate_geometry_manual_pick_cache=lambda: None,
        clear_geometry_manual_undo_stack=lambda: None,
        clear_geometry_fit_undo_stack=lambda: None,
        set_geometry_manual_pick_mode=lambda enabled: None,
        set_background_display_data=lambda image: None,
        update_background_slider_defaults=lambda image: None,
        sync_background_theta_controls=lambda: None,
        sync_geometry_fit_background_selection=lambda: None,
        clear_geometry_pick_artists=lambda: None,
        sync_theta_initial_to_background=lambda idx: None,
        render_current_geometry_manual_pairs=lambda: None,
        schedule_update_factory=build_schedule,
        set_status_text_factory=build_status,
        file_dialog_dir_factory=build_dialog_dir,
        askopenfilenames=lambda **kwargs: ("a.osc",),
    )

    assert factory()["view_state"] == "view-state"
    assert factory()["view_state"] == "view-state"
    assert calls[0]["background_state"] == "background-state"
    assert calls[0]["image_size"] == 2048
    assert calls[0]["display_rotate_k"] == -1
    assert calls[0]["file_dialog_dir"] == "C:/dialogs/1"
    assert calls[1]["file_dialog_dir"] == "C:/dialogs/2"
    assert callable(calls[0]["set_status_text"])
    assert callable(calls[0]["schedule_update"])
    assert calls[0]["set_status_text"] is not calls[1]["set_status_text"]
    assert calls[0]["schedule_update"] is not calls[1]["schedule_update"]


def test_background_manager_runtime_helpers_and_callback_bundle_delegate_live_bindings(
    monkeypatch,
) -> None:
    calls = []
    background_state = state.BackgroundRuntimeState(
        osc_files=["C:/data/a.osc"],
        current_background_index=0,
    )

    monkeypatch.setattr(
        background_manager,
        "set_background_file_status_from_state",
        lambda **kwargs: calls.append(("status", kwargs)) or "status-text",
    )
    monkeypatch.setattr(
        background_manager,
        "load_background_files_with_side_effects",
        lambda background_state_arg, file_paths, **kwargs: (
            calls.append(("load", background_state_arg, tuple(file_paths), kwargs)),
            {"loaded": True},
        )[-1],
    )
    monkeypatch.setattr(
        background_manager,
        "switch_background_with_side_effects",
        lambda background_state_arg, **kwargs: (
            calls.append(("switch", background_state_arg, kwargs)),
            {"switched": True},
        )[-1],
    )

    messages = []
    bindings = background_manager.BackgroundRuntimeBindings(
        view_state="view-state",
        background_state=background_state,
        image_size=2048,
        display_rotate_k=-1,
        read_osc=lambda path: path,
        current_background_theta_values=lambda: [1.0],
        background_theta_for_index=lambda idx: 2.0,
        geometry_fit_uses_shared_theta_offset=lambda: False,
        geometry_manual_pairs_for_index=lambda idx: [],
        geometry_manual_pair_group_count=lambda idx: 0,
        current_geometry_fit_background_indices=lambda: [0],
        sync_background_runtime_state=lambda: None,
        replace_geometry_manual_pairs_by_background=lambda payload: None,
        invalidate_geometry_manual_pick_cache=lambda: None,
        clear_geometry_manual_undo_stack=lambda: None,
        clear_geometry_fit_undo_stack=lambda: None,
        set_geometry_manual_pick_mode=lambda enabled: None,
        set_background_display_data=lambda image: None,
        update_background_slider_defaults=lambda image: None,
        sync_background_theta_controls=lambda: None,
        sync_geometry_fit_background_selection=lambda: None,
        clear_geometry_pick_artists=lambda: None,
        sync_theta_initial_to_background=lambda idx: None,
        render_current_geometry_manual_pairs=lambda: None,
        schedule_update=lambda: None,
        set_status_text=lambda text: messages.append(text),
        file_dialog_dir="C:/dialogs",
        askopenfilenames=lambda **kwargs: calls.append(("dialog", kwargs))
        or ("a.osc", "b.osc"),
    )

    assert (
        background_manager.refresh_runtime_background_file_status(bindings)
        == "status-text"
    )
    assert (
        background_manager.load_runtime_background_files(
            bindings,
            ["a.osc", "b.osc"],
            select_index=1,
        )
        == {"loaded": True}
    )
    assert background_manager.browse_runtime_background_files(bindings) is True
    assert background_manager.switch_runtime_background(bindings) is True
    assert messages == ["Loaded 1 background file(s)."]

    versions = {"count": 0}

    def build_bindings():
        versions["count"] += 1
        return f"bindings-{versions['count']}"

    monkeypatch.setattr(
        background_manager,
        "refresh_runtime_background_file_status",
        lambda bindings_arg: calls.append(("refresh_cb", bindings_arg)) or "refreshed",
    )
    monkeypatch.setattr(
        background_manager,
        "load_runtime_background_files",
        lambda bindings_arg, file_paths, *, select_index=0: (
            calls.append(("load_cb", bindings_arg, tuple(file_paths), select_index)),
            {"loaded_cb": True},
        )[-1],
    )
    monkeypatch.setattr(
        background_manager,
        "browse_runtime_background_files",
        lambda bindings_arg: calls.append(("browse_cb", bindings_arg)) or False,
    )
    monkeypatch.setattr(
        background_manager,
        "switch_runtime_background",
        lambda bindings_arg: calls.append(("switch_cb", bindings_arg)) or True,
    )

    callbacks = background_manager.make_runtime_background_callbacks(build_bindings)

    assert callbacks.refresh_status() == "refreshed"
    assert callbacks.load_files(["x.osc"], 3) == {"loaded_cb": True}
    assert callbacks.browse_files() is False
    assert callbacks.switch_background() is True
    assert calls[-4:] == [
        ("refresh_cb", "bindings-1"),
        ("load_cb", "bindings-2", ("x.osc",), 3),
        ("browse_cb", "bindings-3"),
        ("switch_cb", "bindings-4"),
    ]
