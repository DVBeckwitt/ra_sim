from __future__ import annotations

from ra_sim.gui import manual_geometry as mg
from ra_sim.utils.pbi2_ht_shift_cif import (
    DISORDERED_PHASE_DISPLAY_LABEL,
    DISORDERED_PHASE_SOURCE_LABEL,
)


PRIMARY_KEY = ("q_group", "primary", 1, 4)
DISORDERED_KEY = ("q_group", DISORDERED_PHASE_SOURCE_LABEL, 1, 4)


def _pair(group_key, source_label):
    return {
        "label": "1,1,4",
        "hkl": (1, 1, 4),
        "x": 10.0,
        "y": 11.0,
        "q_group_key": group_key,
        "source_label": source_label,
        "phase_label": DISORDERED_PHASE_DISPLAY_LABEL if source_label == DISORDERED_PHASE_SOURCE_LABEL else "Primary phase",
        "structure_role": "disordered" if source_label == DISORDERED_PHASE_SOURCE_LABEL else "primary",
    }


def test_save_load_disordered_manual_pair_preserves_source_label():
    serialized = mg.geometry_manual_pair_entry_to_jsonable(
        _pair(DISORDERED_KEY, DISORDERED_PHASE_SOURCE_LABEL)
    )
    restored = mg.geometry_manual_pair_entry_from_jsonable(serialized)

    assert restored["source_label"] == DISORDERED_PHASE_SOURCE_LABEL
    assert restored["phase_label"] == DISORDERED_PHASE_DISPLAY_LABEL
    assert restored["structure_role"] == "disordered"
    assert restored["q_group_key"] == DISORDERED_KEY


def test_load_legacy_manual_pair_defaults_to_primary():
    legacy = _pair(PRIMARY_KEY, "primary")
    legacy.pop("source_label")
    restored = mg.geometry_manual_pair_entry_from_jsonable(
        mg.geometry_manual_pair_entry_to_jsonable(legacy)
    )

    assert restored["source_label"] == "primary"


def test_save_load_mixed_primary_disordered_pairs_remain_distinct():
    rows = [
        mg.geometry_manual_pair_entry_to_jsonable(_pair(PRIMARY_KEY, "primary")),
        mg.geometry_manual_pair_entry_to_jsonable(
            _pair(DISORDERED_KEY, DISORDERED_PHASE_SOURCE_LABEL)
        ),
    ]
    restored = [mg.geometry_manual_pair_entry_from_jsonable(row) for row in rows]

    assert [entry["source_label"] for entry in restored] == [
        "primary",
        DISORDERED_PHASE_SOURCE_LABEL,
    ]
    assert [entry["q_group_key"] for entry in restored] == [PRIMARY_KEY, DISORDERED_KEY]
