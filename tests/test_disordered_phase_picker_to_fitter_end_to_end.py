from __future__ import annotations

import numpy as np

from ra_sim.gui import geometry_fit
from ra_sim.gui import manual_geometry as mg
from ra_sim.utils.pbi2_ht_shift_cif import DISORDERED_PHASE_SOURCE_LABEL


PRIMARY_KEY = ("q_group", "primary", 1, 4)
DISORDERED_KEY = ("q_group", DISORDERED_PHASE_SOURCE_LABEL, 1, 4)


def _candidate(group_key, source_label, x, y):
    return {
        "q_group_key": group_key,
        "source_label": source_label,
        "hkl": (1, 1, 4),
        "label": "1,1,4",
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


def _fit_bindings(saved_pair, rows):
    image = np.zeros((700, 700), dtype=float)
    return geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=["bg0.osc"],
        current_background_index=0,
        image_size=700,
        display_rotate_k=0,
        geometry_manual_pairs_for_index=lambda _idx: [dict(saved_pair)],
        load_background_by_index=lambda _idx: (image, image),
        apply_background_backend_orientation=lambda img: img,
        geometry_manual_simulated_peaks_for_params=lambda *_args, **_kwargs: [dict(row) for row in rows],
        geometry_manual_simulated_lookup=lambda _rows: {(0, 0): dict(rows[0])},
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
            {},
        ),
        apply_orientation_to_entries=lambda entries, shape, **_kwargs: [dict(entry) for entry in entries],
        orient_image_for_fit=lambda img, **_kwargs: img,
    )


def test_disordered_picker_to_fitter_uses_disordered_expected_point():
    primary = _candidate(PRIMARY_KEY, "primary", 100.0, 100.0)
    disordered = _candidate(DISORDERED_KEY, DISORDERED_PHASE_SOURCE_LABEL, 500.0, 500.0)
    saved: list[dict[str, object]] = []

    handled, _next = mg.geometry_manual_place_selection_at(
        501.0,
        500.0,
        pick_session={
            "manual_geometry_run_id": "run",
            "background_index": 0,
            "group_key": DISORDERED_KEY,
            "group_entries": [primary, disordered],
            "pending_entries": [],
            "base_entries": [],
            "target_count": 1,
        },
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
        use_caked_space=False,
        background_display_to_native_detector_coords=lambda col, row: (col, row),
        resolve_background_pick_fn=None,
    )
    assert handled is True
    assert saved[0]["q_group_key"] == DISORDERED_KEY
    assert saved[0]["source_label"] == DISORDERED_PHASE_SOURCE_LABEL

    dataset = geometry_fit.build_geometry_manual_fit_dataset(
        0,
        theta_base=0.0,
        base_fit_params={},
        manual_dataset_bindings=_fit_bindings(saved[0], [primary, disordered]),
        orientation_cfg={},
    )
    fitted_entry = dataset["initial_pairs_display"][0]

    assert fitted_entry["source_label"] == DISORDERED_PHASE_SOURCE_LABEL
    assert fitted_entry["sim_display"] == (500.0, 500.0)
