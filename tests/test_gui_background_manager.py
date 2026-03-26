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
