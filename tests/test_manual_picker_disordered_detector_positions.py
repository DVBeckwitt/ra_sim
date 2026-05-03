from __future__ import annotations

import numpy as np

from ra_sim.gui import manual_geometry as mg
from ra_sim.utils.pbi2_ht_shift_cif import DISORDERED_PHASE_SOURCE_LABEL


PRIMARY_KEY = ("q_group", "primary", 1, 1)
DISORDERED_KEY = ("q_group", DISORDERED_PHASE_SOURCE_LABEL, 1, 1)


def _row(group_key, source_label, x, y, *, hkl=(1, 1, 4)):
    return {
        "q_group_key": group_key,
        "source_label": source_label,
        "phase_label": "Disordered phase" if source_label == DISORDERED_PHASE_SOURCE_LABEL else "Primary phase",
        "structure_role": "disordered" if source_label == DISORDERED_PHASE_SOURCE_LABEL else "primary",
        "hkl": hkl,
        "label": ",".join(str(v) for v in hkl),
        "source_table_index": 0,
        "source_row_index": 0,
        "source_branch_index": 1,
        "source_peak_index": 1,
        "display_col": float(x),
        "display_row": float(y),
        "native_col": float(x),
        "native_row": float(y),
        "qr": 1.0,
        "qz": 1.0,
    }


def _grouped(rows):
    return mg.geometry_manual_detector_picker_grouped_candidates_from_cache(
        {"detector_picker_rows": rows},
        display_background=np.zeros((700, 700), dtype=float),
        native_background=np.zeros((700, 700), dtype=float),
        profile_cache={},
    )


def test_disordered_picker_candidate_uses_disordered_hit_table_detector_position():
    grouped = _grouped(
        [
            _row(PRIMARY_KEY, "primary", 100.0, 100.0),
            _row(DISORDERED_KEY, DISORDERED_PHASE_SOURCE_LABEL, 500.0, 500.0),
        ]
    )

    candidate = grouped[DISORDERED_KEY][0]
    assert candidate["source_label"] == DISORDERED_PHASE_SOURCE_LABEL
    assert candidate["q_group_key"] == DISORDERED_KEY
    assert candidate["detector_display_px"] == (500.0, 500.0)


def test_disordered_candidate_does_not_use_primary_position_with_same_hkl():
    grouped = _grouped(
        [
            _row(PRIMARY_KEY, "primary", 100.0, 100.0, hkl=(1, 1, 4)),
            _row(DISORDERED_KEY, DISORDERED_PHASE_SOURCE_LABEL, 500.0, 500.0, hkl=(1, 1, 4)),
        ]
    )

    assert grouped[DISORDERED_KEY][0]["display_col"] == 500.0
    assert grouped[DISORDERED_KEY][0]["display_row"] == 500.0


def test_primary_and_disordered_candidates_same_pixel_remain_separate():
    grouped = _grouped(
        [
            _row(PRIMARY_KEY, "primary", 100.0, 100.0),
            _row(DISORDERED_KEY, DISORDERED_PHASE_SOURCE_LABEL, 100.0, 100.0),
        ]
    )

    assert set(grouped) == {PRIMARY_KEY, DISORDERED_KEY}
    assert grouped[PRIMARY_KEY][0]["source_label"] == "primary"
    assert grouped[DISORDERED_KEY][0]["source_label"] == DISORDERED_PHASE_SOURCE_LABEL


def test_disordered_detector_display_frame_wins_when_values_match_caked():
    group_key = ("q_group", DISORDERED_PHASE_SOURCE_LABEL, 3, 4)
    visible_seed = {
        "label": "-3,0,4",
        "q_group_key": group_key,
        "source_label": DISORDERED_PHASE_SOURCE_LABEL,
        "branch_id": "-x",
        "source_branch_index": 1,
        "display_col": 13.0,
        "display_row": 24.0,
        "display_frame": "detector_display",
        "sim_col_raw": 103.0,
        "sim_row_raw": 204.0,
        "caked_x": 13.0,
        "caked_y": 24.0,
    }

    found_key, entries, best_dist = mg.geometry_manual_choose_group_at(
        {group_key: [visible_seed]},
        13.0,
        24.0,
        window_size_px=10.0,
        use_caked_display=False,
    )

    assert found_key == group_key
    assert best_dist < 1.0
    assert entries[0]["source_label"] == DISORDERED_PHASE_SOURCE_LABEL
