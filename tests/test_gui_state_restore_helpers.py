import numpy as np

from ra_sim.gui import state_io


class _DisplayRecorder:
    def __init__(self) -> None:
        self.last_data = None

    def set_data(self, value) -> None:
        self.last_data = value


class _Var:
    def __init__(self, value) -> None:
        self._value = value

    def get(self):
        return self._value

    def set(self, value) -> None:
        self._value = value


def test_load_background_files_for_state_reuses_identical_files_without_reread(
    tmp_path,
) -> None:
    path_a = tmp_path / "a.osc"
    path_b = tmp_path / "b.osc"
    path_a.write_text("", encoding="utf-8")
    path_b.write_text("", encoding="utf-8")

    native_a = np.arange(4, dtype=float).reshape(2, 2)
    native_b = np.arange(4, 8, dtype=float).reshape(2, 2)
    display_a = np.rot90(native_a, -1)
    display_b = np.rot90(native_b, -1)
    display = _DisplayRecorder()

    def _read_osc_should_not_run(_path: str):
        raise AssertionError("read_osc should not be called for identical backgrounds")

    updated = state_io.load_background_files_for_state(
        [str(path_a), str(path_b)],
        osc_files=[str(path_a), str(path_b)],
        background_images=[native_a.copy(), native_b.copy()],
        background_images_native=[native_a, native_b],
        background_images_display=[display_a, display_b],
        select_index=1,
        display_rotate_k=-1,
        read_osc=_read_osc_should_not_run,
        set_background_display=display.set_data,
    )

    assert updated is not None
    assert updated["osc_files"] == [str(path_a), str(path_b)]
    assert updated["current_background_index"] == 1
    assert updated["current_background_image"] is native_b
    assert updated["current_background_display"] is display_b
    assert display.last_data is display_b


def test_load_background_files_for_state_rereads_when_files_change(tmp_path) -> None:
    path_a = tmp_path / "a.osc"
    path_b = tmp_path / "b.osc"
    path_c = tmp_path / "c.osc"
    for path in (path_a, path_b, path_c):
        path.write_text("", encoding="utf-8")

    calls: list[str] = []

    def _fake_read_osc(path: str):
        calls.append(path)
        if path.endswith("b.osc"):
            return np.full((2, 2), 2.0)
        return np.full((2, 2), 3.0)

    display = _DisplayRecorder()
    updated = state_io.load_background_files_for_state(
        [str(path_b), str(path_c)],
        osc_files=[str(path_a), str(path_b)],
        background_images=[np.zeros((2, 2)), np.ones((2, 2))],
        background_images_native=[np.zeros((2, 2)), np.ones((2, 2))],
        background_images_display=[np.zeros((2, 2)), np.ones((2, 2))],
        select_index=1,
        display_rotate_k=-1,
        read_osc=_fake_read_osc,
        set_background_display=display.set_data,
    )

    assert updated is not None
    assert calls == [str(path_b), str(path_c)]
    assert updated["osc_files"] == [str(path_b), str(path_c)]
    assert updated["current_background_index"] == 1
    assert np.array_equal(
        updated["current_background_image"],
        np.full((2, 2), 3.0),
    )
    assert display.last_data is updated["current_background_display"]


def test_collect_full_gui_state_snapshot_filters_runtime_only_vars(tmp_path) -> None:
    snapshot = state_io.collect_full_gui_state_snapshot(
        global_items={
            "gamma_var": _Var(1.25),
            "_internal_var": _Var(99),
            "background_file_status_var": _Var("skip"),
            "custom_entry_var": _Var("skip"),
            "not_a_var": object(),
        },
        tk_variable_type=_Var,
        occ_vars=[_Var(0.4), _Var("bad")],
        atom_site_fract_vars=[
            {"x": _Var(0.1), "y": _Var(0.2), "z": _Var(0.3)},
        ],
        geometry_q_group_rows=[{"key": ["q", 1], "included": True}],
        geometry_manual_pairs=[{"background_index": 0, "entries": []}],
        selected_hkl_target=(1, 2, 3),
        primary_cif_path=tmp_path / "primary.cif",
        secondary_cif_path=None,
        osc_files=[tmp_path / "a.osc"],
        current_background_index=0,
        background_visible=True,
        background_backend_rotation_k=3,
        background_backend_flip_x=False,
        background_backend_flip_y=True,
        background_limits_user_override=True,
        simulation_limits_user_override=False,
        scale_factor_user_override=True,
    )

    assert snapshot["variables"] == {"gamma_var": 1.25}
    assert snapshot["dynamic_lists"]["occupancy_values"] == [0.4, None]
    assert snapshot["dynamic_lists"]["atom_site_fractional_values"] == [
        {"x": 0.1, "y": 0.2, "z": 0.3}
    ]
    assert snapshot["geometry"]["selected_hkl_target"] == [1, 2, 3]
    assert snapshot["files"]["background_files"] == [str(tmp_path / "a.osc")]


def test_apply_gui_state_flags_toggles_visibility_and_updates_values() -> None:
    toggled = {"count": 0}

    def _toggle_background() -> None:
        toggled["count"] += 1

    updated = state_io.apply_gui_state_flags(
        {
            "background_visible": False,
            "background_backend_rotation_k": 5,
            "background_backend_flip_x": True,
            "background_backend_flip_y": True,
            "background_limits_user_override": False,
            "simulation_limits_user_override": True,
            "scale_factor_user_override": False,
        },
        current_flags={
            "background_visible": True,
            "background_backend_rotation_k": 1,
            "background_backend_flip_x": False,
            "background_backend_flip_y": False,
            "background_limits_user_override": True,
            "simulation_limits_user_override": False,
            "scale_factor_user_override": True,
        },
        toggle_background=_toggle_background,
    )

    assert toggled["count"] == 1
    assert updated == {
        "background_visible": False,
        "background_backend_rotation_k": 1,
        "background_backend_flip_x": True,
        "background_backend_flip_y": True,
        "background_limits_user_override": False,
        "simulation_limits_user_override": True,
        "scale_factor_user_override": False,
    }
