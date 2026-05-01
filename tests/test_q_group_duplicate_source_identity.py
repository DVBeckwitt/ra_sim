from __future__ import annotations

import numpy as np

from ra_sim.gui import geometry_q_group_manager
from ra_sim.utils.pbi2_ht_shift_cif import (
    DISORDERED_PHASE_DISPLAY_LABEL,
    DISORDERED_PHASE_SOURCE_LABEL,
)


def _hit_table() -> np.ndarray:
    return np.array([[10.0, 100.0, 101.0, 0.0, 1.0, 0.0, 0.0]])


def _overlapping_entries() -> list[dict[str, object]]:
    return geometry_q_group_manager.build_geometry_q_group_entries(
        [_hit_table(), _hit_table()],
        peak_table_lattice=[
            (4.557, 20.937, "primary", "Ordered phase", "ordered"),
            (
                4.557,
                20.937,
                DISORDERED_PHASE_SOURCE_LABEL,
                DISORDERED_PHASE_DISPLAY_LABEL,
                "disordered",
            ),
        ],
        primary_a=4.557,
        primary_c=20.937,
        allow_nominal_hkl_indices=True,
    )


def _canonicalizer_rows() -> list[dict[str, object]]:
    return [
        {
            "q_group_key": ("q_group", "primary", 1, 0),
            "source_label": "primary",
            "qr": 1.592,
            "qz": 0.0,
        },
        {
            "q_group_key": ("q_group", DISORDERED_PHASE_SOURCE_LABEL, 1, 0),
            "source_label": DISORDERED_PHASE_SOURCE_LABEL,
            "qr": 1.592,
            "qz": 0.0,
        },
    ]


def test_primary_and_disordered_same_qr_qz_are_not_merged():
    entries = _overlapping_entries()

    assert len(entries) == 2
    assert {entry["source_label"] for entry in entries} == {
        "primary",
        DISORDERED_PHASE_SOURCE_LABEL,
    }


def test_primary_and_disordered_same_qr_qz_sort_adjacent():
    entries = _overlapping_entries()

    assert [entry["source_label"] for entry in entries] == [
        "primary",
        DISORDERED_PHASE_SOURCE_LABEL,
    ]
    np.testing.assert_allclose([entry["qr"] for entry in entries], [1.592, 1.592], rtol=1e-3)
    np.testing.assert_allclose([entry["qz"] for entry in entries], [0.0, 0.0], atol=1e-12)
    assert [entry.get("phase_label") for entry in entries] == [
        "Ordered phase",
        DISORDERED_PHASE_DISPLAY_LABEL,
    ]


def test_duplicate_canonicalizer_preserves_source_identity_when_enabled():
    rows = geometry_q_group_manager.canonicalize_qr_qz_duplicate_source_rows(
        _canonicalizer_rows(),
        preserve_source_identity=True,
    )

    assert rows[0]["q_group_key"] == ("q_group", "primary", 1, 0)
    assert rows[1]["q_group_key"] == (
        "q_group",
        DISORDERED_PHASE_SOURCE_LABEL,
        1,
        0,
    )
    assert "source_q_group_key" not in rows[1]


def test_duplicate_canonicalizer_old_behavior_still_available_when_disabled():
    rows = geometry_q_group_manager.canonicalize_qr_qz_duplicate_source_rows(
        _canonicalizer_rows(),
        preserve_source_identity=False,
    )

    assert rows[0]["q_group_key"] == ("q_group", "primary", 1, 0)
    assert rows[1]["q_group_key"] == ("q_group", "primary", 1, 0)
    assert rows[1]["source_q_group_key"] == (
        "q_group",
        DISORDERED_PHASE_SOURCE_LABEL,
        1,
        0,
    )
