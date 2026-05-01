from __future__ import annotations

from pathlib import Path

from ra_sim.gui import geometry_q_group_manager
from ra_sim.gui import state
from ra_sim.gui import state_io
from ra_sim.utils.pbi2_ht_shift_cif import DISORDERED_PHASE_SOURCE_LABEL


class _Var:
    def __init__(self, value) -> None:
        self._value = value

    def get(self):
        return self._value


def test_saved_disabled_qr_sets_preserve_disordered_source_label(tmp_path: Path):
    snapshot = state_io.collect_full_gui_state_snapshot(
        global_items={},
        tk_variable_type=_Var,
        occ_vars=[],
        atom_site_fract_vars=[],
        geometry_q_group_rows=[],
        geometry_disabled_qr_sets=[
            ("primary", 1),
            (DISORDERED_PHASE_SOURCE_LABEL, 1),
        ],
        geometry_disabled_qz_sections=[
            ("primary", 1, 0),
            (DISORDERED_PHASE_SOURCE_LABEL, 1, 0),
        ],
        geometry_manual_pairs=[],
        geometry_peak_records=[],
        selected_hkl_target=None,
        primary_cif_path=tmp_path / "primary.cif",
        secondary_cif_path=None,
        osc_files=[],
        current_background_index=0,
        background_visible=True,
        background_backend_rotation_k=0,
        background_backend_flip_x=False,
        background_backend_flip_y=False,
        background_limits_user_override=False,
        simulation_limits_user_override=False,
        scale_factor_user_override=False,
    )

    assert [DISORDERED_PHASE_SOURCE_LABEL, 1] in snapshot["geometry"]["disabled_qr_sets"]
    assert [DISORDERED_PHASE_SOURCE_LABEL, 1, 0] in snapshot["geometry"]["disabled_qz_sections"]


def test_loaded_disabled_qr_sets_preserve_disordered_source_label():
    q_group_state = state.GeometryQGroupState()

    updated = state_io.apply_gui_state_geometry(
        {
            "disabled_qr_sets": [[DISORDERED_PHASE_SOURCE_LABEL, 1]],
            "disabled_qz_sections": [[DISORDERED_PHASE_SOURCE_LABEL, 1, 0]],
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
    assert q_group_state.disabled_qr_sets == {(DISORDERED_PHASE_SOURCE_LABEL, 1)}
    assert q_group_state.disabled_qz_sections == {(DISORDERED_PHASE_SOURCE_LABEL, 1, 0)}


def test_loaded_disordered_source_label_is_not_normalized_to_primary():
    q_group_state = state.GeometryQGroupState(
        disabled_qr_sets={(DISORDERED_PHASE_SOURCE_LABEL, 2)},
        disabled_qz_sections={(DISORDERED_PHASE_SOURCE_LABEL, 2, 3)},
    )

    payload = geometry_q_group_manager.build_geometry_q_group_save_payload(
        [],
        q_group_state=q_group_state,
        saved_at="2026-05-01T00:00:00",
    )
    saved_state, error = geometry_q_group_manager.load_geometry_q_group_saved_state(payload)

    assert error is None
    assert payload["disabled_qr_sets"] == [[DISORDERED_PHASE_SOURCE_LABEL, 2]]
    assert payload["disabled_qz_sections"] == [[DISORDERED_PHASE_SOURCE_LABEL, 2, 3]]
    assert saved_state["disabled_qr_sets"] == [(DISORDERED_PHASE_SOURCE_LABEL, 2)]
    assert saved_state["disabled_qz_sections"] == [(DISORDERED_PHASE_SOURCE_LABEL, 2, 3)]
