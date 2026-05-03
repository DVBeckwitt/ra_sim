from __future__ import annotations

import numpy as np

from ra_sim.gui import geometry_fit
from ra_sim.utils.pbi2_ht_shift_cif import DISORDERED_PHASE_SOURCE_LABEL


PRIMARY_KEY = ("q_group", "primary", 1, 4)
DISORDERED_KEY = ("q_group", DISORDERED_PHASE_SOURCE_LABEL, 1, 4)


def _row(group_key, source_label, x, y, *, table=0, row=0):
    return {
        "hkl": (1, 1, 4),
        "label": "1,1,4",
        "q_group_key": group_key,
        "source_label": source_label,
        "source_table_index": int(table),
        "source_row_index": int(row),
        "source_branch_index": 1,
        "source_peak_index": 1,
        "sim_col": float(x),
        "sim_row": float(y),
        "display_col": float(x),
        "display_row": float(y),
        "native_col": float(x),
        "native_row": float(y),
    }


def _pair(group_key, source_label, x=501.0, y=500.0, *, table=0, row=0):
    return {
        "hkl": (1, 1, 4),
        "label": "1,1,4",
        "q_group_key": group_key,
        "source_label": source_label,
        "source_table_index": int(table),
        "source_row_index": int(row),
        "source_branch_index": 1,
        "source_peak_index": 1,
        "x": float(x),
        "y": float(y),
    }


def _bindings(pairs, sim_rows):
    native_background = np.zeros((700, 700), dtype=float)
    display_background = np.zeros((700, 700), dtype=float)

    return geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["bg0.osc"],
        current_background_index=0,
        image_size=700,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda _idx: [dict(pair) for pair in pairs],
        load_background_by_index=lambda _idx: (native_background, display_background),
        apply_background_backend_orientation=lambda image: image,
        geometry_manual_simulated_peaks_for_params=lambda *_args, **_kwargs: [dict(row) for row in sim_rows],
        geometry_manual_simulated_lookup=lambda _rows: {(0, 0): dict(sim_rows[0])} if sim_rows else {},
        geometry_manual_entry_display_coords=lambda entry: (float(entry["x"]), float(entry["y"])),
        unrotate_display_peaks=lambda entries, *_args, **_kwargs: [dict(entry) for entry in entries],
        display_to_native_sim_coords=lambda col, row, *_args, **_kwargs: (float(col), float(row)),
        select_fit_orientation=lambda sim, meas, shape, *, cfg: (
            {
                "indexing_mode": "yx",
                "k": 0,
                "flip_x": False,
                "flip_y": False,
                "flip_order": "xy",
            },
            {"mode": "unit"},
        ),
        apply_orientation_to_entries=lambda entries, shape, **_kwargs: [dict(entry) for entry in entries],
        orient_image_for_fit=lambda image, **_kwargs: image,
    )


def _dataset(pairs, sim_rows):
    return geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=0.0,
        base_fit_params={},
        manual_dataset_bindings=_bindings(pairs, sim_rows),
        orientation_cfg={},
    )


def test_geometry_fit_input_preserves_disordered_source_label():
    pair = _pair(DISORDERED_KEY, DISORDERED_PHASE_SOURCE_LABEL)
    dataset = _dataset(
        [pair],
        [
            _row(PRIMARY_KEY, "primary", 100.0, 100.0),
            _row(DISORDERED_KEY, DISORDERED_PHASE_SOURCE_LABEL, 500.0, 500.0),
        ],
    )

    assert dataset["initial_pairs_display"][0]["source_label"] == DISORDERED_PHASE_SOURCE_LABEL


def test_geometry_fit_prediction_uses_disordered_sim_position_not_primary():
    dataset = _dataset(
        [_pair(DISORDERED_KEY, DISORDERED_PHASE_SOURCE_LABEL, 501.0, 500.0)],
        [
            _row(PRIMARY_KEY, "primary", 100.0, 100.0),
            _row(DISORDERED_KEY, DISORDERED_PHASE_SOURCE_LABEL, 500.0, 500.0),
        ],
    )

    assert dataset["initial_pairs_display"][0]["sim_display"] == (500.0, 500.0)


def test_geometry_fit_legacy_pairs_default_to_primary():
    legacy_pair = _pair(PRIMARY_KEY, "primary", 101.0, 100.0)
    legacy_pair.pop("source_label")
    dataset = _dataset(
        [legacy_pair],
        [
            _row(PRIMARY_KEY, "primary", 100.0, 100.0),
            _row(DISORDERED_KEY, DISORDERED_PHASE_SOURCE_LABEL, 500.0, 500.0),
        ],
    )

    assert dataset["initial_pairs_display"][0]["source_label"] == "primary"
    assert dataset["initial_pairs_display"][0]["sim_display"] == (100.0, 100.0)


def test_geometry_fit_can_mix_primary_and_disordered_pairs():
    primary_pair = _pair(PRIMARY_KEY, "primary", 101.0, 100.0, table=0, row=0)
    disordered_pair = _pair(
        DISORDERED_KEY,
        DISORDERED_PHASE_SOURCE_LABEL,
        501.0,
        500.0,
        table=1,
        row=0,
    )
    dataset = _dataset(
        [primary_pair, disordered_pair],
        [
            _row(PRIMARY_KEY, "primary", 100.0, 100.0, table=0, row=0),
            _row(DISORDERED_KEY, DISORDERED_PHASE_SOURCE_LABEL, 500.0, 500.0, table=1, row=0),
        ],
    )

    by_source = {entry["source_label"]: entry for entry in dataset["initial_pairs_display"]}
    assert by_source["primary"]["sim_display"] == (100.0, 100.0)
    assert by_source[DISORDERED_PHASE_SOURCE_LABEL]["sim_display"] == (500.0, 500.0)
