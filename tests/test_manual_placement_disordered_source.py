from __future__ import annotations

import numpy as np

from ra_sim.gui import manual_geometry as mg
from ra_sim.utils.pbi2_ht_shift_cif import (
    DISORDERED_PHASE_DISPLAY_LABEL,
    DISORDERED_PHASE_SOURCE_LABEL,
)


PRIMARY_KEY = ("q_group", "primary", 1, 4)
DISORDERED_KEY = ("q_group", DISORDERED_PHASE_SOURCE_LABEL, 1, 4)


def _candidate(group_key, source_label, x, y):
    return {
        "q_group_key": group_key,
        "source_label": source_label,
        "phase_label": DISORDERED_PHASE_DISPLAY_LABEL if source_label == DISORDERED_PHASE_SOURCE_LABEL else "Primary phase",
        "structure_role": "disordered" if source_label == DISORDERED_PHASE_SOURCE_LABEL else "primary",
        "hkl": (-1, 1, 4),
        "label": "-1,1,4",
        "source_table_index": 0,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "display_col": float(x),
        "display_row": float(y),
        "sim_col": float(x),
        "sim_row": float(y),
        "native_col": float(x),
        "native_row": float(y),
    }


def _session(group_entries):
    return {
        "manual_geometry_run_id": "run",
        "background_index": 0,
        "group_key": DISORDERED_KEY,
        "group_entries": [dict(row) for row in group_entries],
        "pending_entries": [],
        "base_entries": [],
        "target_count": 1,
        "q_label": "disordered",
    }


def test_disordered_preview_filters_remaining_peaks_by_source():
    primary = _candidate(PRIMARY_KEY, "primary", 100.0, 100.0)
    disordered = _candidate(DISORDERED_KEY, DISORDERED_PHASE_SOURCE_LABEL, 500.0, 500.0)

    preview = mg.geometry_manual_pick_preview_state(
        501.0,
        500.0,
        pick_session=_session([primary, disordered]),
        current_background_index=0,
        force=True,
        remaining_candidates=[primary, disordered],
        display_background=np.zeros((700, 700), dtype=float),
        refine_preview_point=lambda _candidate, col, row, **_kwargs: (col, row),
        use_caked_space=False,
    )

    assert preview is not None
    assert preview["candidate"]["source_label"] == DISORDERED_PHASE_SOURCE_LABEL
    assert preview["sim_dist"] < 2.0
    assert "candidate_source=disordered_phase" in preview["message"]


def test_disordered_placement_distance_uses_disordered_sim_position():
    saved: list[dict[str, object]] = []
    statuses: list[str] = []
    primary = _candidate(PRIMARY_KEY, "primary", 100.0, 100.0)
    disordered = _candidate(DISORDERED_KEY, DISORDERED_PHASE_SOURCE_LABEL, 500.0, 500.0)

    handled, _next = mg.geometry_manual_place_selection_at(
        501.0,
        500.0,
        pick_session=_session([primary, disordered]),
        current_background_index=0,
        display_background=np.zeros((700, 700), dtype=float),
        get_cache_data=lambda **_kwargs: {},
        refine_preview_point=lambda _candidate, col, row, **_kwargs: (col, row),
        set_pairs_for_index_fn=lambda _idx, entries: saved.extend(dict(e) for e in entries) or entries,
        set_pick_session_fn=lambda _session: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        restore_view_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=statuses.append,
        use_caked_space=False,
        background_display_to_native_detector_coords=lambda col, row: (col, row),
        resolve_background_pick_fn=None,
    )

    assert handled is True
    assert saved[0]["source_label"] == DISORDERED_PHASE_SOURCE_LABEL
    assert saved[0]["assignment_distance_to_sim"] < 2.0
    assert "candidate_source=disordered_phase" in statuses[-1]


def test_disordered_placement_does_not_fallback_to_primary_when_source_missing():
    statuses: list[str] = []
    primary = _candidate(PRIMARY_KEY, "primary", 100.0, 100.0)

    handled, _next = mg.geometry_manual_place_selection_at(
        101.0,
        100.0,
        pick_session=_session([primary]),
        current_background_index=0,
        display_background=np.zeros((200, 200), dtype=float),
        get_cache_data=lambda **_kwargs: {},
        refine_preview_point=lambda _candidate, col, row, **_kwargs: (col, row),
        set_pairs_for_index_fn=lambda _idx, entries: entries,
        set_pick_session_fn=lambda _session: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        restore_view_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=statuses.append,
        use_caked_space=False,
        background_display_to_native_detector_coords=lambda col, row: (col, row),
        resolve_background_pick_fn=None,
    )

    assert handled is False
    assert any("source_mismatch_error=true" in msg for msg in statuses)


def test_disordered_saved_pair_keeps_source_label():
    disordered = _candidate(DISORDERED_KEY, DISORDERED_PHASE_SOURCE_LABEL, 500.0, 500.0)

    entry = mg.geometry_manual_pair_entry_from_candidate(
        disordered,
        501.0,
        500.0,
        group_key=DISORDERED_KEY,
        detector_col=501.0,
        detector_row=500.0,
    )

    assert entry["source_label"] == DISORDERED_PHASE_SOURCE_LABEL
    assert entry["q_group_key"] == DISORDERED_KEY
    assert entry["phase_label"] == DISORDERED_PHASE_DISPLAY_LABEL
    assert entry["structure_role"] == "disordered"
