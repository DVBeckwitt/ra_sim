from __future__ import annotations

import numpy as np

from ra_sim.gui import manual_geometry as mg
from ra_sim.utils.pbi2_ht_shift_cif import (
    DISORDERED_PHASE_DISPLAY_LABEL,
    DISORDERED_PHASE_SOURCE_LABEL,
)


PRIMARY_KEY = ("q_group", "primary", 1, 0)
DISORDERED_KEY = ("q_group", DISORDERED_PHASE_SOURCE_LABEL, 1, 0)


def _picker_row(
    group_key: tuple[object, ...],
    source_label: str,
    *,
    phase_label: str | None = None,
) -> dict[str, object]:
    row = {
        "q_group_key": group_key,
        "source_label": source_label,
        "hkl": (1, 0, 0),
        "label": "1,0,0",
        "branch_id": "+x",
        "source_branch_index": 0,
        "source_peak_index": 0,
        "source_table_index": 0,
        "source_row_index": 0,
        "display_col": 100.0,
        "display_row": 120.0,
        "native_col": 100.0,
        "native_row": 120.0,
        "qr": 1.592,
        "qz": 0.0,
        "distance_to_click": 0.0,
    }
    if phase_label is not None:
        row["phase_label"] = phase_label
    return row


def _grouped(rows: list[dict[str, object]]):
    return mg.geometry_manual_detector_picker_grouped_candidates_from_cache(
        {"detector_picker_rows": rows},
        display_background=np.zeros((240, 240), dtype=float),
        native_background=np.zeros((240, 240), dtype=float),
        profile_cache={},
    )


def test_picker_keeps_primary_and_disordered_candidates_at_same_pixel():
    grouped = _grouped(
        [
            _picker_row(PRIMARY_KEY, "primary", phase_label="Ordered phase"),
            _picker_row(DISORDERED_KEY, DISORDERED_PHASE_SOURCE_LABEL),
        ]
    )

    assert set(grouped) == {PRIMARY_KEY, DISORDERED_KEY}
    assert len(grouped[PRIMARY_KEY]) == 1
    assert len(grouped[DISORDERED_KEY]) == 1
    assert grouped[PRIMARY_KEY][0]["detector_display_px"] == (100.0, 120.0)
    assert grouped[DISORDERED_KEY][0]["detector_display_px"] == (100.0, 120.0)


def test_picker_keys_include_source_label():
    grouped = _grouped(
        [
            _picker_row(PRIMARY_KEY, "primary", phase_label="Ordered phase"),
            _picker_row(DISORDERED_KEY, DISORDERED_PHASE_SOURCE_LABEL),
        ]
    )

    assert list(grouped) == [PRIMARY_KEY, DISORDERED_KEY]
    assert mg._geometry_manual_real_q_group_key({"key": DISORDERED_KEY}) == DISORDERED_KEY


def test_picker_displays_disordered_phase_label():
    grouped = _grouped([_picker_row(DISORDERED_KEY, DISORDERED_PHASE_SOURCE_LABEL)])

    assert grouped[DISORDERED_KEY][0]["phase_label"] == DISORDERED_PHASE_DISPLAY_LABEL
    assert grouped[DISORDERED_KEY][0]["structure_role"] == "disordered"


def test_picker_sorts_overlapping_primary_before_disordered():
    grouped = _grouped(
        [
            _picker_row(DISORDERED_KEY, DISORDERED_PHASE_SOURCE_LABEL),
            _picker_row(PRIMARY_KEY, "primary", phase_label="Ordered phase"),
        ]
    )
    flattened = mg._geometry_manual_flatten_grouped_candidates(grouped)
    best_key, best_entries, best_dist = mg.geometry_manual_choose_group_at(
        grouped,
        100.0,
        120.0,
        window_size_px=20.0,
    )

    assert [entry["q_group_key"] for entry in flattened] == [PRIMARY_KEY, DISORDERED_KEY]
    assert best_key == PRIMARY_KEY
    assert best_entries[0]["source_label"] == "primary"
    assert best_dist == 0.0
