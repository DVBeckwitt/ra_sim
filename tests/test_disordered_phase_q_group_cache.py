from __future__ import annotations

import importlib

import numpy as np

from ra_sim.gui import controllers
from ra_sim.gui import geometry_q_group_manager
from ra_sim.gui import state
from ra_sim.utils.pbi2_ht_shift_cif import (
    DISORDERED_PHASE_DISPLAY_LABEL,
    DISORDERED_PHASE_SOURCE_LABEL,
)
from tests.helpers.gui_fakes import FloatRuntimeVar as _Var


def _hit_table(h: int = 2, k: int = 0, l_val: int = 1) -> np.ndarray:
    return np.array([[10.0, 100.0, 101.0, 0.0, float(h), float(k), float(l_val)]])


def _disordered_entry() -> dict[str, object]:
    return {
        "key": ("q_group", DISORDERED_PHASE_SOURCE_LABEL, 4, 1),
        "source_label": DISORDERED_PHASE_SOURCE_LABEL,
        "phase_label": DISORDERED_PHASE_DISPLAY_LABEL,
        "structure_role": "disordered",
        "overlap_identity": (123, 456),
        "qr": 2.0,
        "qz": 0.3,
        "gz_index": 1,
        "total_intensity": 10.0,
        "peak_count": 1,
        "hkl_preview": [(2, 0, 1)],
    }


def _build_disordered_entries():
    return geometry_q_group_manager.build_geometry_q_group_entries(
        [_hit_table()],
        peak_table_lattice=[
            (
                4.557,
                20.937,
                DISORDERED_PHASE_SOURCE_LABEL,
                DISORDERED_PHASE_DISPLAY_LABEL,
                "disordered",
            )
        ],
        primary_a=4.557,
        primary_c=6.979,
        allow_nominal_hkl_indices=True,
    )


def test_disordered_phase_entries_are_added_to_q_group_cache(monkeypatch):
    runtime_session = importlib.import_module("ra_sim.gui._runtime.runtime_session")
    runtime_state = state.SimulationRuntimeState()
    runtime_state.stored_primary_sim_image = np.zeros((4, 4), dtype=np.float64)
    runtime_state.stored_disordered_phase_max_positions = [_hit_table()]
    runtime_state.stored_disordered_phase_peak_table_lattice = [
        (
            4.557,
            20.937,
            DISORDERED_PHASE_SOURCE_LABEL,
            DISORDERED_PHASE_DISPLAY_LABEL,
            "disordered",
        )
    ]
    monkeypatch.setattr(runtime_session, "simulation_runtime_state", runtime_state, raising=False)
    monkeypatch.setattr(runtime_session, "weight1_var", _Var(1.0), raising=False)
    monkeypatch.setattr(runtime_session, "weight2_var", _Var(0.0), raising=False)
    monkeypatch.setattr(runtime_session, "_geometry_disordered_phase_qr_enabled", lambda: True)

    runtime_session._publish_combined_simulation_state(
        image_size_value=4,
        primary_a_value=4.557,
        primary_c_value=6.979,
        secondary_a_value=float("nan"),
        secondary_c_value=float("nan"),
        active_peak_row_sides=("primary",),
    )
    runtime_state.geometry_q_group_entries_cache = (
        geometry_q_group_manager.build_geometry_q_group_entries(
            runtime_state.stored_max_positions_local,
            peak_table_lattice=runtime_state.stored_peak_table_lattice,
            primary_a=4.557,
            primary_c=6.979,
            allow_nominal_hkl_indices=True,
        )
    )

    assert any(
        entry.get("source_label") == DISORDERED_PHASE_SOURCE_LABEL
        for entry in runtime_state.geometry_q_group_entries_cache
    )


def test_disordered_phase_entries_include_phase_label():
    entries = _build_disordered_entries()

    assert entries[0]["source_label"] == DISORDERED_PHASE_SOURCE_LABEL
    assert entries[0]["phase_label"] == DISORDERED_PHASE_DISPLAY_LABEL
    assert entries[0]["overlap_identity"] is not None


def test_disordered_phase_entries_include_structure_role():
    entries = _build_disordered_entries()

    assert entries[0]["structure_role"] == "disordered"


def test_clone_geometry_q_group_entries_preserves_disordered_metadata():
    cloned = controllers.clone_geometry_q_group_entries([_disordered_entry()])

    assert cloned == [_disordered_entry()]
    cloned[0]["hkl_preview"].append((9, 9, 9))
    assert _disordered_entry()["hkl_preview"] == [(2, 0, 1)]


def test_export_rows_preserve_disordered_phase_label():
    q_group_state = state.GeometryQGroupState(cached_entries=[_disordered_entry()])

    rows = geometry_q_group_manager.build_geometry_q_group_export_rows(
        preview_state=state.GeometryPreviewState(),
        q_group_state=q_group_state,
    )

    assert rows[0]["source_label"] == DISORDERED_PHASE_SOURCE_LABEL
    assert rows[0]["phase_label"] == DISORDERED_PHASE_DISPLAY_LABEL
    assert rows[0]["structure_role"] == "disordered"
    assert rows[0]["overlap_identity"] == [123, 456]
