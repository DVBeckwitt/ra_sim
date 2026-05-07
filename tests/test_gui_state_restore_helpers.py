import numpy as np
import pytest

from ra_sim.gui import controllers, state, state_io


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


def test_load_background_files_for_state_fills_missing_selected_cache_for_identical_files(
    tmp_path,
) -> None:
    path_a = tmp_path / "a.osc"
    path_b = tmp_path / "b.osc"
    path_a.write_text("", encoding="utf-8")
    path_b.write_text("", encoding="utf-8")

    native_a = np.arange(4, dtype=float).reshape(2, 2)
    display_a = np.rot90(native_a, -1)
    calls: list[str] = []
    display = _DisplayRecorder()

    def _fake_read_osc(path: str):
        calls.append(path)
        return np.full((2, 2), 9.0)

    updated = state_io.load_background_files_for_state(
        [str(path_a), str(path_b)],
        osc_files=[str(path_a), str(path_b)],
        background_images=[native_a.copy(), None],
        background_images_native=[native_a, None],
        background_images_display=[display_a, None],
        select_index=1,
        display_rotate_k=-1,
        read_osc=_fake_read_osc,
        set_background_display=display.set_data,
    )

    assert updated is not None
    assert calls == [str(path_b)]
    assert np.array_equal(updated["current_background_image"], np.full((2, 2), 9.0))
    assert updated["background_images"][1] is not None
    assert updated["background_images_native"][1] is not None
    assert updated["background_images_display"][1] is not None
    assert display.last_data is updated["current_background_display"]


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


def test_load_background_files_for_state_loads_only_first_and_selected_image(
    tmp_path,
) -> None:
    path_a = tmp_path / "a.osc"
    path_b = tmp_path / "b.osc"
    path_c = tmp_path / "c.osc"
    for path in (path_a, path_b, path_c):
        path.write_text("", encoding="utf-8")

    calls: list[str] = []

    def _fake_read_osc(path: str):
        calls.append(path)
        if path.endswith("a.osc"):
            return np.zeros((2, 2))
        if path.endswith("c.osc"):
            return np.full((2, 2), 3.0)
        return np.full((2, 2), 2.0)

    display = _DisplayRecorder()
    updated = state_io.load_background_files_for_state(
        [str(path_a), str(path_b), str(path_c)],
        osc_files=[],
        background_images=[],
        background_images_native=[],
        background_images_display=[],
        select_index=2,
        display_rotate_k=-1,
        read_osc=_fake_read_osc,
        set_background_display=display.set_data,
    )

    assert updated is not None
    assert calls == [str(path_a), str(path_c)]
    assert updated["background_images"][0] is not None
    assert updated["background_images"][1] is None
    assert updated["background_images"][2] is not None
    assert updated["current_background_index"] == 2
    assert display.last_data is updated["current_background_display"]


def test_load_background_files_for_state_validates_expected_shape(tmp_path) -> None:
    path_a = tmp_path / "a.osc"
    path_a.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="expected"):
        state_io.load_background_files_for_state(
            [str(path_a)],
            osc_files=[],
            background_images=[],
            background_images_native=[],
            background_images_display=[],
            select_index=0,
            display_rotate_k=-1,
            read_osc=lambda _path: np.ones((3, 3)),
            expected_shape=(4, 4),
        )


def test_apply_gui_state_files_calls_primary_and_secondary_callbacks(tmp_path) -> None:
    primary_path = tmp_path / "primary.cif"
    secondary_path = tmp_path / "secondary.cif"
    primary_path.write_text("data_primary", encoding="utf-8")
    secondary_path.write_text("data_secondary", encoding="utf-8")
    applied_primary: list[str] = []
    applied_secondary: list[str | None] = []

    warnings = state_io.apply_gui_state_files(
        {
            "primary_cif_path": str(primary_path),
            "secondary_cif_path": str(secondary_path),
        },
        apply_primary_cif_path=applied_primary.append,
        apply_secondary_cif_path=applied_secondary.append,
        load_background_files=lambda *_args: None,
    )

    assert warnings == []
    assert applied_primary == [str(primary_path)]
    assert applied_secondary == [str(secondary_path)]


def test_apply_gui_state_files_warns_when_secondary_cif_is_missing(tmp_path) -> None:
    missing_secondary_path = tmp_path / "missing-secondary.cif"
    applied_secondary: list[str | None] = []

    warnings = state_io.apply_gui_state_files(
        {
            "secondary_cif_path": str(missing_secondary_path),
        },
        apply_primary_cif_path=lambda _path: None,
        apply_secondary_cif_path=applied_secondary.append,
        load_background_files=lambda *_args: None,
    )

    assert warnings == [f"secondary CIF missing: {missing_secondary_path}"]
    assert applied_secondary == []


def test_apply_gui_state_files_allows_explicit_secondary_clear() -> None:
    applied_secondary: list[str | None] = []

    warnings = state_io.apply_gui_state_files(
        {
            "secondary_cif_path": None,
        },
        apply_primary_cif_path=lambda _path: None,
        apply_secondary_cif_path=applied_secondary.append,
        load_background_files=lambda *_args: None,
    )

    assert warnings == []
    assert applied_secondary == [None]


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
        geometry_disabled_qr_sets=[("primary", 1)],
        geometry_disabled_qz_sections=[("primary", 1, 2)],
        geometry_manual_pairs=[{"background_index": 0, "entries": []}],
        geometry_peak_records=[
            {
                "display_col": 10.0,
                "display_row": 20.0,
                "hkl": (1, 0, 2),
                "q_group_key": ("q_group", "primary", 1.0, 2),
            }
        ],
        selected_hkl_target=(1, 2, 3),
        primary_cif_path=tmp_path / "primary.cif",
        secondary_cif_path=None,
        osc_files=[tmp_path / "a.osc"],
        current_background_index=0,
        background_visible=True,
        simulation_overlay_visible=False,
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
    assert snapshot["geometry"]["disabled_qr_sets"] == [["primary", 1]]
    assert snapshot["geometry"]["disabled_qz_sections"] == [["primary", 1, 2]]
    assert snapshot["geometry"]["peak_records"] == [
        {
            "display_col": 10.0,
            "display_row": 20.0,
            "hkl": (1, 0, 2),
            "q_group_key": ("q_group", "primary", 1.0, 2),
        }
    ]
    assert snapshot["files"]["background_files"] == [str(tmp_path / "a.osc")]
    assert snapshot["flags"]["simulation_overlay_visible"] is False


def test_apply_gui_state_geometry_restores_peak_cache_when_present() -> None:
    restored_peak_records: list[object] = []
    q_group_state = state.GeometryQGroupState()

    updated = state_io.apply_gui_state_geometry(
        {
            "peak_records": [
                {
                    "display_col": 10.0,
                    "display_row": 20.0,
                    "hkl": [1, 0, 2],
                }
            ],
            "manual_pairs": [],
        },
        q_group_state=q_group_state,
        geometry_q_group_key_from_jsonable=lambda value: tuple(value) if isinstance(value, list) else None,
        invalidate_geometry_manual_pick_cache=lambda: None,
        apply_geometry_manual_pairs_snapshot=lambda *_args, **_kwargs: None,
        replace_runtime_peak_cache=lambda rows: restored_peak_records.extend(
            list(rows or ())
        ),
        current_background_index=0,
        selected_hkl_target=None,
    )

    assert updated["warnings"] == []
    assert restored_peak_records == [
        {
            "display_col": 10.0,
            "display_row": 20.0,
            "hkl": [1, 0, 2],
        }
    ]


def test_apply_gui_state_geometry_restores_saved_q_group_rows_without_peak_records() -> None:
    q_group_state = state.GeometryQGroupState(
        cached_entries=[{"key": ("q_group", "stale", 99, 1)}]
    )
    invalidations = []

    updated = state_io.apply_gui_state_geometry(
        {
            "q_group_rows": [
                {
                    "key": ["q_group", "primary", 1, 8],
                    "included": True,
                    "display_label": "primary Qr=1.0 Gz=8",
                    "peak_count": 2,
                    "hkl_preview": [[1, 0, 8]],
                },
                {
                    "key": ["bad"],
                    "included": True,
                },
            ],
            "peak_records": [],
            "manual_pairs": [],
        },
        q_group_state=q_group_state,
        geometry_q_group_key_from_jsonable=lambda value: tuple(value)
        if isinstance(value, list) and len(value) == 4
        else None,
        invalidate_geometry_manual_pick_cache=lambda: invalidations.append(True),
        apply_geometry_manual_pairs_snapshot=lambda *_args, **_kwargs: None,
        replace_runtime_peak_cache=lambda _rows: None,
        current_background_index=0,
        selected_hkl_target=None,
    )

    assert updated["warnings"] == []
    assert invalidations
    assert q_group_state.cached_entries == [
        {
            "key": ("q_group", "primary", 1, 8),
            "included": True,
            "display_label": "primary Qr=1.0 Gz=8",
            "peak_count": 2,
            "hkl_preview": [[1, 0, 8]],
        }
    ]
    assert q_group_state.restored_q_group_rows_pending_live_refresh is True
    assert q_group_state.refresh_requested is True


def test_apply_gui_state_geometry_clears_restored_q_group_pending_flag_without_rows() -> None:
    q_group_state = state.GeometryQGroupState(
        cached_entries=[{"key": ("q_group", "stale", 99, 1)}],
        restored_q_group_rows_pending_live_refresh=True,
    )

    state_io.apply_gui_state_geometry(
        {
            "q_group_rows": [],
            "peak_records": [],
            "manual_pairs": [],
        },
        q_group_state=q_group_state,
        geometry_q_group_key_from_jsonable=lambda _value: None,
        invalidate_geometry_manual_pick_cache=lambda: None,
        apply_geometry_manual_pairs_snapshot=lambda *_args, **_kwargs: None,
        replace_runtime_peak_cache=lambda _rows: None,
        current_background_index=0,
        selected_hkl_target=None,
    )

    assert q_group_state.restored_q_group_rows_pending_live_refresh is False


def test_replace_geometry_q_group_masks_normalizes_parent_keys_from_rows() -> None:
    q_group_state = state.GeometryQGroupState()

    changed = controllers.replace_geometry_q_group_masks(
        q_group_state,
        disabled_qr_sets=[
            {"q_group_key": ("q_group", "PRIMARY", 1.0, 8)},
            ("secondary", 2.0),
            ("q_group", "primary", 3.0, 9),
            ("primary", 4.5),
        ],
    )

    assert changed is True
    assert q_group_state.disabled_qr_sets == {
        ("primary", 1),
        ("secondary", 2),
        ("primary", 3),
    }


def test_apply_gui_state_geometry_filters_restored_peak_cache_after_masks() -> None:
    restored_peak_records: list[object] = []
    q_group_state = state.GeometryQGroupState(
        cached_entries=[
            {"key": ("q_group", "primary", 1, 8)},
            {"key": ("q_group", "primary", 2, 8)},
        ]
    )

    updated = state_io.apply_gui_state_geometry(
        {
            "disabled_qr_sets": [["primary", 1]],
            "peak_records": [
                {
                    "display_col": 10.0,
                    "display_row": 20.0,
                    "hkl": [1, 0, 8],
                    "q_group_key": ("q_group", "primary", 1, 8),
                },
                {
                    "display_col": 30.0,
                    "display_row": 40.0,
                    "hkl": [2, 0, 8],
                    "q_group_key": ("q_group", "primary", 2, 8),
                },
                {
                    "display_col": 50.0,
                    "display_row": 60.0,
                    "hkl": [9, 9, 9],
                },
            ],
            "manual_pairs": [],
        },
        q_group_state=q_group_state,
        geometry_q_group_key_from_jsonable=lambda value: tuple(value) if isinstance(value, list) else None,
        invalidate_geometry_manual_pick_cache=lambda: None,
        apply_geometry_manual_pairs_snapshot=lambda *_args, **_kwargs: None,
        replace_runtime_peak_cache=lambda rows: restored_peak_records.extend(
            list(rows or ())
        ),
        current_background_index=0,
        selected_hkl_target=None,
    )

    assert updated["warnings"] == []
    assert q_group_state.disabled_qr_sets == {("primary", 1)}
    assert restored_peak_records == [
        {
            "display_col": 30.0,
            "display_row": 40.0,
            "hkl": [2, 0, 8],
            "q_group_key": ("q_group", "primary", 2, 8),
        }
    ]


def test_apply_gui_state_geometry_preserves_explicit_masks_until_structural_refresh() -> None:
    q_group_state = state.GeometryQGroupState(
        cached_entries=[
            {"key": ("q_group", "stale", 99, 1)},
        ]
    )

    updated = state_io.apply_gui_state_geometry(
        {
            "disabled_qr_sets": [["primary", 1]],
            "disabled_qz_sections": [["primary", 2, 8]],
            "manual_pairs": [],
        },
        q_group_state=q_group_state,
        geometry_q_group_key_from_jsonable=lambda value: tuple(value) if isinstance(value, list) else None,
        invalidate_geometry_manual_pick_cache=lambda: None,
        apply_geometry_manual_pairs_snapshot=lambda *_args, **_kwargs: None,
        replace_runtime_peak_cache=lambda _rows: None,
        current_background_index=0,
        selected_hkl_target=None,
    )

    assert updated["warnings"] == []
    assert q_group_state.disabled_qr_sets == {("primary", 1)}
    assert q_group_state.disabled_qz_sections == {("primary", 2, 8)}
    assert q_group_state.refresh_requested is True


def test_apply_gui_state_geometry_defers_legacy_mask_resolution_until_structural_refresh() -> None:
    q_group_state = state.GeometryQGroupState(
        cached_entries=[
            {"key": ("q_group", "stale", 99, 1)},
        ]
    )

    updated = state_io.apply_gui_state_geometry(
        {
            "q_group_rows": [
                {
                    "key": ["q_group", "primary", 1, 8],
                    "included": False,
                },
                {
                    "key": ["q_group", "primary", 1, 9],
                    "included": True,
                },
            ],
            "manual_pairs": [],
        },
        q_group_state=q_group_state,
        geometry_q_group_key_from_jsonable=lambda value: tuple(value) if isinstance(value, list) else None,
        invalidate_geometry_manual_pick_cache=lambda: None,
        apply_geometry_manual_pairs_snapshot=lambda *_args, **_kwargs: None,
        replace_runtime_peak_cache=lambda _rows: None,
        current_background_index=0,
        selected_hkl_target=None,
    )

    assert updated["warnings"] == []
    assert q_group_state.disabled_qr_sets == set()
    assert q_group_state.disabled_qz_sections == set()
    assert q_group_state.pending_legacy_disabled_qz_sections == {
        ("primary", 1, 8)
    }
    assert q_group_state.refresh_requested is True


def test_apply_gui_state_flags_toggles_visibility_and_updates_values() -> None:
    toggled = {"background": 0, "simulation": 0}

    def _toggle_background() -> None:
        toggled["background"] += 1

    def _toggle_simulation_overlay() -> None:
        toggled["simulation"] += 1

    updated = state_io.apply_gui_state_flags(
        {
            "background_visible": False,
            "simulation_overlay_visible": False,
            "background_backend_rotation_k": 5,
            "background_backend_flip_x": True,
            "background_backend_flip_y": True,
            "background_limits_user_override": False,
            "simulation_limits_user_override": True,
            "scale_factor_user_override": False,
        },
        current_flags={
            "background_visible": True,
            "simulation_overlay_visible": True,
            "background_backend_rotation_k": 1,
            "background_backend_flip_x": False,
            "background_backend_flip_y": False,
            "background_limits_user_override": True,
            "simulation_limits_user_override": False,
            "scale_factor_user_override": True,
        },
        toggle_background=_toggle_background,
        toggle_simulation_overlay=_toggle_simulation_overlay,
    )

    assert toggled == {"background": 1, "simulation": 1}
    assert updated == {
        "background_visible": False,
        "simulation_overlay_visible": False,
        "background_backend_rotation_k": 1,
        "background_backend_flip_x": True,
        "background_backend_flip_y": True,
        "background_limits_user_override": False,
        "simulation_limits_user_override": True,
        "scale_factor_user_override": False,
    }


def test_apply_gui_state_flags_tolerates_legacy_string_values() -> None:
    toggled = {"background": 0, "simulation": 0}

    def _toggle_background() -> None:
        toggled["background"] += 1

    def _toggle_simulation_overlay() -> None:
        toggled["simulation"] += 1

    updated = state_io.apply_gui_state_flags(
        {
            "background_visible": "false",
            "simulation_overlay_visible": "off",
            "background_backend_rotation_k": "bad",
            "background_backend_flip_x": "yes",
            "background_backend_flip_y": "0",
            "background_limits_user_override": "off",
            "simulation_limits_user_override": "1",
            "scale_factor_user_override": None,
        },
        current_flags={
            "background_visible": True,
            "simulation_overlay_visible": True,
            "background_backend_rotation_k": 3,
            "background_backend_flip_x": False,
            "background_backend_flip_y": True,
            "background_limits_user_override": True,
            "simulation_limits_user_override": False,
            "scale_factor_user_override": True,
        },
        toggle_background=_toggle_background,
        toggle_simulation_overlay=_toggle_simulation_overlay,
    )

    assert toggled == {"background": 1, "simulation": 1}
    assert updated == {
        "background_visible": False,
        "simulation_overlay_visible": False,
        "background_backend_rotation_k": 3,
        "background_backend_flip_x": True,
        "background_backend_flip_y": False,
        "background_limits_user_override": False,
        "simulation_limits_user_override": True,
        "scale_factor_user_override": True,
    }


def test_geometry_state_requires_visible_background_when_q_group_is_included() -> None:
    def _key_from_jsonable(value):
        if value == ["q_group", "primary", 1.0, 2]:
            return ("q_group", "primary", 1.0, 2)
        return None

    assert state_io.geometry_state_requires_visible_background(
        {
            "q_group_rows": [
                {"key": ["q_group", "primary", 1.0, 2], "included": True},
                {"key": ["q_group", "primary", 1.0, 3], "included": False},
            ]
        },
        geometry_q_group_key_from_jsonable=_key_from_jsonable,
    )


def test_geometry_state_requires_visible_background_when_manual_pairs_exist() -> None:
    assert state_io.geometry_state_requires_visible_background(
        {
            "manual_pairs": [
                {
                    "background_index": 0,
                    "entries": [{"q_group_key": ["q_group", "primary", 1.0, 2]}],
                }
            ]
        },
        geometry_q_group_key_from_jsonable=lambda _value: None,
    )


def test_geometry_state_requires_visible_background_ignores_excluded_or_invalid_rows() -> None:
    def _key_from_jsonable(_value):
        return None

    assert (
        state_io.geometry_state_requires_visible_background(
            {
                "q_group_rows": [
                    {"key": ["bad"], "included": True},
                    {"key": ["also-bad"], "included": False},
                ]
            },
            geometry_q_group_key_from_jsonable=_key_from_jsonable,
        )
        is False
    )


def test_apply_geometry_state_background_view_compatibility_forces_caked_view() -> None:
    toggled = {"count": 0}
    show_1d_var = _Var(False)
    show_caked_2d_var = _Var(True)

    def _key_from_jsonable(value):
        if value == ["q_group", "primary", 1.0, 2]:
            return ("q_group", "primary", 1.0, 2)
        return None

    updated = state_io.apply_geometry_state_background_view_compatibility(
        {
            "q_group_rows": [
                {"key": ["q_group", "primary", 1.0, 2], "included": True},
            ]
        },
        geometry_q_group_key_from_jsonable=_key_from_jsonable,
        show_caked_2d_var=show_caked_2d_var,
        show_1d_var=show_1d_var,
        background_visible=False,
        toggle_background=lambda: toggled.__setitem__("count", toggled["count"] + 1),
    )

    assert updated is True
    assert show_caked_2d_var.get() is True
    assert show_1d_var.get() is True
    assert toggled["count"] == 1


def test_apply_geometry_state_background_view_compatibility_skips_when_no_q_groups() -> None:
    toggled = {"count": 0}
    show_caked_2d_var = _Var(True)

    updated = state_io.apply_geometry_state_background_view_compatibility(
        {"q_group_rows": [{"key": ["bad"], "included": True}]},
        geometry_q_group_key_from_jsonable=lambda _value: None,
        show_caked_2d_var=show_caked_2d_var,
        background_visible=False,
        toggle_background=lambda: toggled.__setitem__("count", toggled["count"] + 1),
    )

    assert updated is False
    assert show_caked_2d_var.get() is True
    assert toggled["count"] == 0


def test_apply_geometry_state_background_view_compatibility_uses_manual_pairs() -> None:
    toggled = {"count": 0}
    show_1d_var = _Var(False)
    show_caked_2d_var = _Var(False)

    updated = state_io.apply_geometry_state_background_view_compatibility(
        {
            "manual_pairs": [
                {
                    "background_index": 0,
                    "entries": [{"q_group_key": ["q_group", "primary", 1.0, 2]}],
                }
            ]
        },
        geometry_q_group_key_from_jsonable=lambda _value: None,
        show_caked_2d_var=show_caked_2d_var,
        show_1d_var=show_1d_var,
        background_visible=False,
        toggle_background=lambda: toggled.__setitem__("count", toggled["count"] + 1),
    )

    assert updated is True
    assert show_caked_2d_var.get() is True
    assert show_1d_var.get() is True
    assert toggled["count"] == 1
