from pathlib import Path

import numpy as np

from ra_sim.gui import background_manager, state


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
            "from": float(from_value),
            "to": float(to_value),
        }

    def cget(self, key):
        return self._values[key]

    def configure(self, *, from_, to):
        self._values["from"] = float(from_)
        self._values["to"] = float(to)


class _DummyDisplay:
    def __init__(self):
        self.alpha = None
        self.clim = None

    def set_alpha(self, value):
        self.alpha = value

    def set_clim(self, min_value, max_value):
        self.clim = (min_value, max_value)


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


def test_background_manager_normalize_runtime_state_coerces_values_and_updates_paths() -> None:
    background_state = state.BackgroundRuntimeState(
        osc_files=["a.osc"],
        background_images=(np.zeros((1, 1)),),
        background_images_native=(np.ones((1, 1)),),
        background_images_display=(np.full((1, 1), 2.0),),
        current_background_index="3",
        visible=0,
        backend_rotation_k="5",
        backend_flip_x=1,
        backend_flip_y="",
    )
    osc_alias = background_state.osc_files
    file_paths_state = {}

    updated = background_manager.normalize_background_runtime_state(
        background_state,
        file_paths_state=file_paths_state,
    )

    assert updated is background_state
    assert background_state.osc_files is osc_alias
    assert isinstance(background_state.background_images, list)
    assert isinstance(background_state.background_images_native, list)
    assert isinstance(background_state.background_images_display, list)
    assert background_state.current_background_index == 3
    assert background_state.visible is False
    assert background_state.backend_rotation_k == 5
    assert background_state.backend_flip_x is True
    assert background_state.backend_flip_y is False
    assert file_paths_state == {
        "simulation_background_osc_files": ["a.osc"],
    }


def test_background_manager_initialize_runtime_state_boots_cache_and_defaults(
    monkeypatch,
) -> None:
    background_state = state.BackgroundRuntimeState(osc_files=["a.osc", "b.osc"])
    file_paths_state = {}
    calls = []
    payload = {
        "background_images": [np.zeros((2, 2))],
        "background_images_native": [np.ones((2, 2))],
        "background_images_display": [np.full((2, 2), 2.0)],
        "current_background_index": 0,
        "current_background_image": "native-image",
        "current_background_display": "display-image",
    }

    monkeypatch.setattr(
        background_manager.gui_background,
        "initialize_background_cache",
        lambda first_path, **kwargs: calls.append((first_path, kwargs)) or payload,
    )

    updated = background_manager.initialize_background_runtime_state(
        background_state,
        "a.osc",
        total_count=2,
        display_rotate_k=-1,
        read_osc=lambda path: path,
        file_paths_state=file_paths_state,
        visible=False,
        backend_rotation_k=7,
        backend_flip_x=True,
        backend_flip_y=True,
    )

    assert updated is payload
    assert background_state.current_background_image == "native-image"
    assert background_state.current_background_display == "display-image"
    assert background_state.visible is False
    assert background_state.backend_rotation_k == 7
    assert background_state.backend_flip_x is True
    assert background_state.backend_flip_y is True
    assert file_paths_state == {"simulation_background_osc_files": ["a.osc", "b.osc"]}
    assert len(calls) == 1
    assert calls[0][0] == "a.osc"
    assert calls[0][1]["total_count"] == 2
    assert calls[0][1]["display_rotate_k"] == -1
    assert callable(calls[0][1]["read_osc"])


def test_background_manager_resolves_background_display_defaults_from_image() -> None:
    defaults = background_manager.resolve_background_display_defaults(
        np.array([-5.0, 10.0], dtype=float)
    )

    assert defaults.min_candidate < 0.0
    assert defaults.vmin_default == 0.0
    assert defaults.vmax_default > 0.0
    assert defaults.slider_min == defaults.min_candidate
    assert defaults.slider_max > defaults.vmax_default
    assert defaults.slider_step >= 0.01


def test_background_manager_slider_max_uses_true_background_peak() -> None:
    image = np.concatenate(
        [
            np.zeros(100, dtype=float),
            np.array([1000.0], dtype=float),
        ]
    )

    defaults = background_manager.resolve_background_display_defaults(image)

    assert defaults.vmax_default == 1.0
    assert defaults.slider_max == 1000.0


def test_background_manager_applies_background_limits_and_transparency() -> None:
    display_state = state.DisplayControlsState()
    view_state = state.DisplayControlsViewState(
        background_min_var=_DummyVar(1.0),
        background_max_var=_DummyVar(4.0),
        background_transparency_var=_DummyVar(0.25),
    )
    background_display = _DummyDisplay()
    draw_calls = []

    applied = background_manager.apply_background_limits(
        display_state,
        view_state,
        background_display=background_display,
        draw_idle=lambda: draw_calls.append("draw"),
    )

    assert applied is True
    assert display_state.background_limits_user_override is True
    assert background_display.clim == (1.0, 4.0)
    assert background_display.alpha == 0.75
    assert draw_calls == ["draw"]


def test_background_manager_invalid_background_limits_adjust_min_in_place() -> None:
    display_state = state.DisplayControlsState()
    view_state = state.DisplayControlsViewState(
        background_min_var=_DummyVar(5.0),
        background_max_var=_DummyVar(5.0),
        background_transparency_var=_DummyVar(0.0),
    )
    background_display = _DummyDisplay()

    applied = background_manager.apply_background_limits(
        display_state,
        view_state,
        background_display=background_display,
    )

    assert applied is False
    assert display_state.suppress_background_limit_callback is False
    assert view_state.background_min_var.get() < 5.0
    assert background_display.clim is None


def test_background_manager_updates_background_slider_defaults() -> None:
    display_state = state.DisplayControlsState(background_limits_user_override=True)
    view_state = state.DisplayControlsViewState(
        background_min_var=_DummyVar(2.0),
        background_max_var=_DummyVar(8.0),
        background_min_slider=_DummySlider(0.0, 10.0),
        background_max_slider=_DummySlider(0.0, 10.0),
    )
    background_display = _DummyDisplay()
    image = np.array([0.0, 2.0, 4.0, 8.0, 10.0], dtype=float)

    kept = background_manager.update_background_slider_defaults(
        display_state,
        view_state,
        background_display=background_display,
        image=image,
        reset_override=False,
    )
    reset = background_manager.update_background_slider_defaults(
        display_state,
        view_state,
        background_display=background_display,
        image=image,
        reset_override=True,
    )

    assert kept == (2.0, 8.0)
    assert reset[0] == 0.0
    assert reset[1] >= 8.0
    assert display_state.background_limits_user_override is False
    assert display_state.suppress_background_limit_callback is False
    assert background_display.clim == reset
    assert view_state.background_min_slider.cget("to") >= reset[1]
    assert view_state.background_max_slider.cget("from") <= 0.0


def test_background_manager_updates_slider_ceiling_to_true_background_peak() -> None:
    display_state = state.DisplayControlsState(background_limits_user_override=False)
    view_state = state.DisplayControlsViewState(
        background_min_var=_DummyVar(0.0),
        background_max_var=_DummyVar(1.0),
        background_min_slider=_DummySlider(0.0, 5.0),
        background_max_slider=_DummySlider(0.0, 5.0),
    )
    background_display = _DummyDisplay()
    image = np.concatenate(
        [
            np.zeros(100, dtype=float),
            np.array([1000.0], dtype=float),
        ]
    )

    background_manager.update_background_slider_defaults(
        display_state,
        view_state,
        background_display=background_display,
        image=image,
        reset_override=False,
    )

    assert view_state.background_min_slider.cget("to") == 1000.0
    assert view_state.background_max_slider.cget("to") == 1000.0
    assert view_state.background_max_var.get() == 1.0


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


def test_background_manager_load_background_image_by_index_updates_lazy_cache_in_place(
    monkeypatch,
) -> None:
    background_state = state.BackgroundRuntimeState(
        osc_files=["a.osc", "b.osc"],
        background_images=[np.zeros((1, 1)), None],
        background_images_native=[np.zeros((1, 1)), None],
        background_images_display=[np.zeros((1, 1)), None],
        current_background_image="existing-native",
        current_background_display="existing-display",
    )
    image_alias = background_state.background_images
    calls = []
    payload = {
        "background_images": [np.zeros((1, 1)), np.ones((1, 1))],
        "background_images_native": [np.zeros((1, 1)), np.ones((1, 1))],
        "background_images_display": [np.zeros((1, 1)), np.full((1, 1), 2.0)],
        "background_image": np.ones((1, 1)),
        "background_display": np.full((1, 1), 2.0),
    }

    monkeypatch.setattr(
        background_manager.gui_background,
        "load_background_image_by_index",
        lambda index, **kwargs: calls.append((index, kwargs)) or payload,
    )

    updated = background_manager.load_background_image_by_index(
        background_state,
        1,
        display_rotate_k=-1,
        read_osc=lambda path: path,
    )

    assert updated is payload
    assert background_state.background_images is image_alias
    assert np.array_equal(background_state.background_images[1], np.ones((1, 1)))
    assert np.array_equal(
        background_state.background_images_display[1],
        np.full((1, 1), 2.0),
    )
    assert background_state.current_background_image == "existing-native"
    assert background_state.current_background_display == "existing-display"
    assert calls[0][0] == 1
    assert calls[0][1]["osc_files"] == ["a.osc", "b.osc"]


def test_background_manager_current_background_helpers_update_lazy_cache_in_place(
    monkeypatch,
) -> None:
    background_state = state.BackgroundRuntimeState(
        osc_files=["a.osc"],
        background_images=[np.zeros((1, 1))],
        background_images_native=[np.zeros((1, 1))],
        background_images_display=[np.zeros((1, 1))],
        current_background_index=0,
        current_background_image="existing-native",
        current_background_display="existing-display",
    )
    calls = []
    native_payload = {
        "background_images": [np.full((1, 1), 3.0)],
        "background_images_native": [np.full((1, 1), 3.0)],
        "background_images_display": [np.full((1, 1), 4.0)],
        "background_image": np.full((1, 1), 3.0),
        "background_display": None,
    }
    display_payload = {
        "background_images": [np.full((1, 1), 5.0)],
        "background_images_native": [np.full((1, 1), 5.0)],
        "background_images_display": [np.full((1, 1), 6.0)],
        "background_image": None,
        "background_display": np.full((1, 1), 6.0),
    }

    monkeypatch.setattr(
        background_manager.gui_background,
        "get_current_background_native",
        lambda **kwargs: calls.append(("native", kwargs)) or native_payload,
    )
    monkeypatch.setattr(
        background_manager.gui_background,
        "get_current_background_display",
        lambda **kwargs: calls.append(("display", kwargs)) or display_payload,
    )

    updated_native = background_manager.get_current_background_native(
        background_state,
        display_rotate_k=-1,
        read_osc=lambda path: path,
    )
    updated_display = background_manager.get_current_background_display(
        background_state,
        display_rotate_k=-1,
        read_osc=lambda path: path,
    )

    assert updated_native is native_payload
    assert updated_display is display_payload
    assert np.array_equal(background_state.background_images_native[0], np.full((1, 1), 5.0))
    assert np.array_equal(background_state.background_images_display[0], np.full((1, 1), 6.0))
    assert background_state.current_background_image == "existing-native"
    assert background_state.current_background_display == "existing-display"
    assert calls[0][0] == "native"
    assert calls[0][1]["current_background_image"] == "existing-native"
    assert calls[1][0] == "display"
    assert calls[1][1]["current_background_display"] == "existing-display"


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


def test_background_manager_backend_status_refresh_reads_runtime_state(
    monkeypatch,
) -> None:
    captured = {}
    view_state = state.BackgroundBackendDebugViewState()
    background_state = state.BackgroundRuntimeState(
        backend_rotation_k=5,
        backend_flip_x=True,
        backend_flip_y=False,
    )

    monkeypatch.setattr(
        background_manager.gui_views,
        "set_background_backend_status_text",
        lambda view_state_arg, text: captured.update(
            {"view_state": view_state_arg, "text": text}
        ),
    )

    text = background_manager.set_background_backend_status_from_state(
        view_state=view_state,
        background_state=background_state,
    )

    assert text == "k=1 flip_x=True flip_y=False"
    assert background_manager.background_backend_status_text(background_state) == text
    assert captured == {
        "view_state": view_state,
        "text": text,
    }


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
        sync_background_theta_controls=lambda: events.append(("sync_theta_controls",)),
        sync_geometry_fit_background_selection=lambda: events.append(
            ("sync_fit_selection",)
        ),
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
        preempt_simulation_update=lambda: events.append(("preempt",)),
    )

    assert updated == {"current_background_display": "switched-display"}
    assert events == [
        ("preempt",),
        ("switch", -1),
        ("sync",),
        ("invalidate",),
        ("clear_manual_undo",),
        ("clear_fit_undo",),
        ("display", "switched-display"),
        ("slider_defaults", "switched-display"),
        ("sync_theta_controls",),
        ("sync_fit_selection",),
        ("render_manual_pairs",),
        ("refresh_status",),
        ("schedule_update",),
    ]


def test_background_manager_switch_workflow_defers_canvas_refresh_in_caked_view(
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
        sync_background_theta_controls=lambda: events.append(("sync_theta_controls",)),
        sync_geometry_fit_background_selection=lambda: events.append(
            ("sync_fit_selection",)
        ),
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
        preempt_simulation_update=lambda: events.append(("preempt",)),
        caked_view_active=lambda: True,
    )

    assert updated == {"current_background_display": "switched-display"}
    assert events == [
        ("preempt",),
        ("switch", -1),
        ("sync",),
        ("invalidate",),
        ("clear_manual_undo",),
        ("clear_fit_undo",),
        ("sync_theta_controls",),
        ("sync_fit_selection",),
        ("refresh_status",),
        ("schedule_update",),
    ]


def test_background_manager_backend_orientation_side_effects_update_status_and_redraw() -> None:
    background_state = state.BackgroundRuntimeState(
        backend_rotation_k=3,
        backend_flip_x=False,
        backend_flip_y=False,
    )
    events = []

    rotated = background_manager.rotate_background_backend_with_side_effects(
        background_state,
        delta_k=-1,
        sync_background_runtime_state=lambda: events.append(("sync",)),
        refresh_background_backend_status=lambda: events.append(("refresh",))
        or background_manager.background_backend_status_text(background_state),
        mark_chi_square_dirty=lambda: events.append(("dirty",)),
        refresh_chi_square_display=lambda: events.append(("redraw",)),
        schedule_update=lambda: events.append(("schedule",)),
    )

    assert rotated == "k=2 flip_x=False flip_y=False"
    assert background_state.backend_rotation_k == 2
    assert events == [
        ("sync",),
        ("refresh",),
        ("dirty",),
        ("redraw",),
        ("schedule",),
    ]

    events.clear()
    flipped = background_manager.toggle_background_backend_flip_with_side_effects(
        background_state,
        axis="x",
        sync_background_runtime_state=lambda: events.append(("sync",)),
        refresh_background_backend_status=lambda: events.append(("refresh",))
        or background_manager.background_backend_status_text(background_state),
        mark_chi_square_dirty=lambda: events.append(("dirty",)),
        refresh_chi_square_display=lambda: events.append(("redraw",)),
        schedule_update=lambda: events.append(("schedule",)),
    )

    assert flipped == "k=2 flip_x=True flip_y=False"
    assert background_state.backend_flip_x is True
    assert events == [
        ("sync",),
        ("refresh",),
        ("dirty",),
        ("redraw",),
        ("schedule",),
    ]

    events.clear()
    reset = background_manager.reset_background_backend_orientation_with_side_effects(
        background_state,
        sync_background_runtime_state=lambda: events.append(("sync",)),
        refresh_background_backend_status=lambda: events.append(("refresh",))
        or background_manager.background_backend_status_text(background_state),
        mark_chi_square_dirty=lambda: events.append(("dirty",)),
        refresh_chi_square_display=lambda: events.append(("redraw",)),
        schedule_update=lambda: events.append(("schedule",)),
    )

    assert reset == "k=0 flip_x=False flip_y=False"
    assert background_state.backend_rotation_k == 0
    assert background_state.backend_flip_x is False
    assert background_state.backend_flip_y is False
    assert events == [
        ("sync",),
        ("refresh",),
        ("dirty",),
        ("redraw",),
        ("schedule",),
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


def test_background_manager_toggle_visibility_updates_alpha_and_schedules() -> None:
    background_state = state.BackgroundRuntimeState(visible=True)
    events = []

    visible = background_manager.toggle_background_visibility_with_side_effects(
        background_state,
        sync_background_runtime_state=lambda: events.append("sync"),
        set_background_alpha=lambda alpha: events.append(("alpha", alpha)),
        schedule_update=lambda: events.append("schedule"),
    )

    assert visible is False
    assert background_state.visible is False
    assert background_manager.background_alpha_for_visibility(True) == 0.5
    assert background_manager.background_alpha_for_visibility(False) == 1.0
    assert events == ["sync", ("alpha", 1.0), "schedule"]


def test_background_manager_runtime_binding_factory_builds_live_bindings(
    monkeypatch,
) -> None:
    calls = []
    counters = {"status": 0, "schedule": 0, "preempt": 0, "dir": 0}

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

    def build_preempt():
        counters["preempt"] += 1
        idx = counters["preempt"]
        return lambda: f"preempt-{idx}"

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
        set_background_alpha=lambda alpha: None,
        update_background_slider_defaults=lambda image: None,
        sync_background_theta_controls=lambda: None,
        sync_geometry_fit_background_selection=lambda: None,
        clear_geometry_pick_artists=lambda: None,
        sync_theta_initial_to_background=lambda idx: None,
        render_current_geometry_manual_pairs=lambda: None,
        background_backend_debug_view_state="backend-view-state",
        mark_chi_square_dirty=lambda: None,
        refresh_chi_square_display=lambda: None,
        schedule_update_factory=build_schedule,
        preempt_simulation_update_factory=build_preempt,
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
    assert calls[0]["background_backend_debug_view_state"] == "backend-view-state"
    assert callable(calls[0]["mark_chi_square_dirty"])
    assert callable(calls[0]["refresh_chi_square_display"])
    assert callable(calls[0]["set_background_alpha"])
    assert callable(calls[0]["set_status_text"])
    assert callable(calls[0]["schedule_update"])
    assert callable(calls[0]["preempt_simulation_update"])
    assert calls[0]["set_status_text"] is not calls[1]["set_status_text"]
    assert calls[0]["schedule_update"] is not calls[1]["schedule_update"]
    assert calls[0]["preempt_simulation_update"] is not calls[1]["preempt_simulation_update"]


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
        "set_background_backend_status_from_state",
        lambda **kwargs: calls.append(("backend_status", kwargs)) or "backend-status",
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
        set_background_alpha=lambda alpha: calls.append(("alpha_set", alpha)),
        update_background_slider_defaults=lambda image: None,
        sync_background_theta_controls=lambda: None,
        sync_geometry_fit_background_selection=lambda: None,
        clear_geometry_pick_artists=lambda: None,
        sync_theta_initial_to_background=lambda idx: None,
        render_current_geometry_manual_pairs=lambda: None,
        background_backend_debug_view_state="backend-view",
        mark_chi_square_dirty=lambda: None,
        refresh_chi_square_display=lambda: None,
        schedule_update=lambda: None,
        preempt_simulation_update=lambda: None,
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
        background_manager.refresh_runtime_background_backend_status(bindings)
        == "backend-status"
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
    switch_call = calls[-1]
    assert switch_call[0] == "switch"
    assert switch_call[2]["preempt_simulation_update"] is bindings.preempt_simulation_update
    assert (
        switch_call[2]["sync_background_theta_controls"]
        is bindings.sync_background_theta_controls
    )
    assert (
        switch_call[2]["sync_geometry_fit_background_selection"]
        is bindings.sync_geometry_fit_background_selection
    )
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
        "toggle_runtime_background_visibility",
        lambda bindings_arg: calls.append(("toggle_visibility_cb", bindings_arg))
        or True,
    )
    monkeypatch.setattr(
        background_manager,
        "refresh_runtime_background_backend_status",
        lambda bindings_arg: calls.append(("backend_refresh_cb", bindings_arg))
        or "backend-refreshed",
    )
    monkeypatch.setattr(
        background_manager,
        "rotate_runtime_background_backend",
        lambda bindings_arg, *, delta_k: (
            calls.append(("rotate_cb", bindings_arg, delta_k)),
            f"rotated-{delta_k}",
        )[-1],
    )
    monkeypatch.setattr(
        background_manager,
        "toggle_runtime_background_backend_flip",
        lambda bindings_arg, *, axis: (
            calls.append(("flip_cb", bindings_arg, axis)),
            f"flip-{axis}",
        )[-1],
    )
    monkeypatch.setattr(
        background_manager,
        "reset_runtime_background_backend_orientation",
        lambda bindings_arg: calls.append(("reset_cb", bindings_arg)) or "reset",
    )
    monkeypatch.setattr(
        background_manager,
        "switch_runtime_background",
        lambda bindings_arg: calls.append(("switch_cb", bindings_arg)) or True,
    )

    callbacks = background_manager.make_runtime_background_callbacks(build_bindings)

    assert callbacks.refresh_status() == "refreshed"
    assert callbacks.toggle_visibility() is True
    assert callbacks.load_files(["x.osc"], 3) == {"loaded_cb": True}
    assert callbacks.browse_files() is False
    assert callbacks.refresh_backend_status() == "backend-refreshed"
    assert callbacks.rotate_backend_minus_90() == "rotated--1"
    assert callbacks.rotate_backend_plus_90() == "rotated-1"
    assert callbacks.flip_backend_x() == "flip-x"
    assert callbacks.flip_backend_y() == "flip-y"
    assert callbacks.reset_backend_orientation() == "reset"
    assert callbacks.switch_background() is True
    assert calls[-11:] == [
        ("refresh_cb", "bindings-1"),
        ("toggle_visibility_cb", "bindings-2"),
        ("load_cb", "bindings-3", ("x.osc",), 3),
        ("browse_cb", "bindings-4"),
        ("backend_refresh_cb", "bindings-5"),
        ("rotate_cb", "bindings-6", -1),
        ("rotate_cb", "bindings-7", 1),
        ("flip_cb", "bindings-8", "x"),
        ("flip_cb", "bindings-9", "y"),
        ("reset_cb", "bindings-10"),
        ("switch_cb", "bindings-11"),
    ]
