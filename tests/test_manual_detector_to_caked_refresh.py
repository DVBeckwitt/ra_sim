from __future__ import annotations

import numpy as np

from ra_sim.gui import manual_geometry as mg
from ra_sim.utils.pbi2_ht_shift_cif import DISORDERED_PHASE_SOURCE_LABEL


PRIMARY_KEY = ("q_group", "primary", 1, 4)
DISORDERED_KEY = ("q_group", DISORDERED_PHASE_SOURCE_LABEL, 1, 4)


def _candidate(group_key=PRIMARY_KEY, source_label="primary"):
    return {
        "q_group_key": group_key,
        "source_label": source_label,
        "hkl": (1, 1, 4),
        "label": "1,1,4",
        "source_table_index": 0,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "display_col": 210.0,
        "display_row": 220.0,
        "sim_col": 210.0,
        "sim_row": 220.0,
        "native_col": 210.0,
        "native_row": 220.0,
        "caked_x": 9.0,
        "caked_y": 8.0,
    }


def _session(candidate):
    return {
        "manual_geometry_run_id": "run",
        "background_index": 0,
        "group_key": candidate["q_group_key"],
        "group_entries": [dict(candidate)],
        "pending_entries": [],
        "base_entries": [],
        "target_count": 1,
    }


def _place(candidate, *, native_to_caked=True):
    saved: list[dict[str, object]] = []
    statuses: list[str] = []

    handled, _next = mg.geometry_manual_place_selection_at(
        200.0,
        200.0,
        pick_session=_session(candidate),
        current_background_index=0,
        display_background=np.zeros((400, 400), dtype=float),
        get_cache_data=lambda **_kwargs: {},
        refine_preview_point=lambda _candidate, _col, _row, **_kwargs: (210.0, 220.0),
        set_pairs_for_index_fn=lambda _idx, entries: saved.extend(dict(e) for e in entries) or entries,
        set_pick_session_fn=lambda _session: None,
        clear_preview_artists_fn=lambda **_kwargs: None,
        restore_view_fn=lambda **_kwargs: None,
        render_current_pairs_fn=lambda **_kwargs: None,
        update_button_label_fn=lambda: None,
        set_status_text=statuses.append,
        use_caked_space=False,
        background_display_to_native_detector_coords=lambda col, row: (float(col), float(row)),
        native_detector_coords_to_caked_display_coords=(
            (lambda x, y: (float(x) / 10.0, float(y) / 10.0))
            if native_to_caked
            else None
        ),
        resolve_background_pick_fn=None,
    )
    assert handled is True
    return saved[0], statuses


def test_detector_space_placement_backfills_caked_fields_when_callback_available():
    entry, statuses = _place(_candidate())

    assert entry["detector_display_x"] == 210.0
    assert entry["detector_display_y"] == 220.0
    assert entry["detector_native_x"] == 210.0
    assert entry["detector_native_y"] == 220.0
    assert entry["background_two_theta_deg"] == 21.0
    assert entry["background_phi_deg"] == 22.0
    assert entry["caked_x"] == 21.0
    assert entry["caked_y"] == 22.0
    assert entry["raw_caked_x"] == 20.0
    assert entry["raw_caked_y"] == 20.0
    assert entry["manual_background_input_origin"] == "detector"
    assert entry["background_detector_frame_provenance"] == "detector_to_caked_refresh"
    assert "status_observed_refined_caked_deg=(21.000000,22.000000)" in statuses[-1]
    assert "status_sim_visual_caked_deg=(9.000000,8.000000)" in statuses[-1]


def test_detector_space_placement_logs_unavailable_when_caked_callback_missing():
    entry, statuses = _place(_candidate(), native_to_caked=False)

    assert entry["detector_to_caked_unavailable"] is True
    assert "detector_to_caked_unavailable=true" in statuses[-1]
    assert "background_two_theta_deg" not in entry


def test_disordered_detector_space_placement_backfills_caked_fields():
    entry, statuses = _place(
        _candidate(DISORDERED_KEY, DISORDERED_PHASE_SOURCE_LABEL),
    )

    assert entry["source_label"] == DISORDERED_PHASE_SOURCE_LABEL
    assert entry["background_two_theta_deg"] == 21.0
    assert entry["background_phi_deg"] == 22.0
    assert entry["background_detector_frame_provenance"] == "detector_to_caked_refresh"
    assert "candidate_source=disordered_phase" in statuses[-1]
    assert "status_observed_refined_caked_deg=(21.000000,22.000000)" in statuses[-1]
