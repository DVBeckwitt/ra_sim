from __future__ import annotations

import numpy as np

from ra_sim.gui import manual_geometry as mg
from ra_sim.utils.pbi2_ht_shift_cif import DISORDERED_PHASE_SOURCE_LABEL


DISORDERED_KEY = ("q_group", DISORDERED_PHASE_SOURCE_LABEL, 1, 4)
PRIMARY_KEY = ("q_group", "primary", 1, 4)


def _candidate(group_key, source_label, x=100.0, y=100.0):
    return {
        "q_group_key": group_key,
        "source_label": source_label,
        "hkl": (1, 1, 4),
        "label": "1,1,4",
        "source_table_index": 0,
        "source_row_index": 0,
        "source_branch_index": 1,
        "display_col": float(x),
        "display_row": float(y),
        "sim_col": float(x),
        "sim_row": float(y),
    }


def test_manual_select_log_includes_source_consistency_fields_for_disordered_group():
    text = mg.geometry_manual_cmd_provenance_text(
        run_id="r",
        emitter="unit",
        event="manual_geometry_select_group",
        **mg._geometry_manual_source_consistency_fields(
            selected_group_key=DISORDERED_KEY,
            candidate=_candidate(DISORDERED_KEY, DISORDERED_PHASE_SOURCE_LABEL),
        ),
    )

    assert "selected_group_source=disordered_phase" in text
    assert "candidate_source=disordered_phase" in text
    assert "detector_position_source=disordered_phase_hit_table" in text


def test_manual_place_log_includes_source_consistency_fields_for_disordered_group():
    preview = mg.geometry_manual_pick_preview_state(
        100.0,
        100.0,
        pick_session={
            "manual_geometry_run_id": "r",
            "background_index": 0,
            "group_key": DISORDERED_KEY,
            "group_entries": [_candidate(DISORDERED_KEY, DISORDERED_PHASE_SOURCE_LABEL)],
            "pending_entries": [],
        },
        current_background_index=0,
        force=True,
        remaining_candidates=[_candidate(DISORDERED_KEY, DISORDERED_PHASE_SOURCE_LABEL)],
        display_background=np.zeros((200, 200), dtype=float),
        refine_preview_point=lambda _candidate, col, row, **_kwargs: (col, row),
        use_caked_space=False,
    )

    assert preview is not None
    assert "sim_peak_source=disordered_phase" in preview["message"]
    assert "assigned_hkl_source=disordered_phase" in preview["message"]


def test_primary_manual_logs_still_include_primary_source_fields():
    text = mg.geometry_manual_cmd_provenance_text(
        run_id="r",
        emitter="unit",
        event="manual_geometry_select_group",
        **mg._geometry_manual_source_consistency_fields(
            selected_group_key=PRIMARY_KEY,
            candidate=_candidate(PRIMARY_KEY, "primary"),
        ),
    )

    assert "selected_group_source=primary" in text
    assert "candidate_source=primary" in text
    assert "detector_position_source=primary_hit_table" in text
