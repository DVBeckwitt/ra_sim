import numpy as np

from ra_sim.gui import manual_geometry as mg


def _group_candidates(entries):
    grouped = {}
    for raw_entry in entries or ():
        if not isinstance(raw_entry, dict):
            continue
        group_key = raw_entry.get("q_group_key")
        if not isinstance(group_key, tuple):
            continue
        grouped.setdefault(group_key, []).append(dict(raw_entry))
    return grouped


def _build_lookup(entries):
    lookup = {}
    for raw_entry in entries or ():
        if not isinstance(raw_entry, dict):
            continue
        try:
            key = (
                int(raw_entry.get("source_table_index")),
                int(raw_entry.get("source_row_index")),
            )
        except Exception:
            continue
        lookup[key] = dict(raw_entry)
    return lookup


def _fail_projection_legacy_path(message: str):
    def _fail(*_args, **_kwargs):
        raise AssertionError(message)

    return _fail


def test_build_geometry_manual_pick_cache_falls_back_to_live_peak_records() -> None:
    peak_record = {
        "display_col": 13.5,
        "display_row": 15.5,
        "hkl": (1, 0, 2),
        "q_group_key": ("q_group", "primary", 1, 2),
        "source_table_index": 4,
        "source_row_index": 7,
        "qr": 1.2345678901,
        "qz": -0.4567890123,
        "intensity": 9.0,
    }

    cache_data, next_sig, next_state = mg.build_geometry_manual_pick_cache(
        param_set={"a": 2.0},
        prefer_cache=True,
        background_index=0,
        current_background_index=0,
        background_image=np.zeros((4, 4), dtype=float),
        existing_cache_signature=None,
        existing_cache_data=None,
        cache_signature_fn=lambda **_kwargs: ("sig",),
        source_rows_for_background=lambda *_args, **_kwargs: [],
        build_grouped_candidates=_group_candidates,
        build_simulated_lookup=_build_lookup,
        current_match_config=lambda: {"search_radius_px": 18.0},
        peak_records=[peak_record],
    )

    group_entry = cache_data["grouped_candidates"][("q_group", "primary", 1, 2)][0]

    assert cache_data["cache_metadata"]["cache_source"] == "peak_records"
    assert cache_data["cache_metadata"]["stale_reason"] is None
    assert group_entry["label"] == "1,0,2"
    assert group_entry["sim_col"] == 13.5
    assert np.isclose(float(group_entry["qr"]), 1.2345678901)
    assert np.isclose(float(next_state["simulated_lookup"][(4, 7)]["qz"]), -0.4567890123)
    assert next_sig == ("sig",)


def test_make_runtime_geometry_manual_cache_callbacks_uses_live_peak_records() -> None:
    cache_state = {"signature": None, "data": {}}
    peak_record = {
        "display_col": 21.0,
        "display_row": 34.0,
        "hkl_raw": (1.0, 0.0, 2.0),
        "q_group_key": ("q_group", "primary", 1, 2),
        "source_table_index": 5,
        "source_row_index": 8,
        "qr": 1.5,
        "qz": -0.5,
        "intensity": 3.0,
    }

    def _replace_cache_state(signature, data) -> None:
        cache_state["signature"] = signature
        cache_state["data"] = dict(data)

    callbacks = mg.make_runtime_geometry_manual_cache_callbacks(
        fit_config={"geometry": {"auto_match": {"search_radius_px": 18.0}}},
        last_simulation_signature=lambda: ("sim", 3),
        current_background_index=lambda: 0,
        current_background_image=lambda: np.zeros((4, 4), dtype=float),
        use_caked_space=lambda: False,
        replace_cache_state=_replace_cache_state,
        current_geometry_fit_params=lambda: {"gamma": 1.25},
        pairs_for_index=lambda _idx: [],
        source_rows_for_background=lambda *_args, **_kwargs: [],
        build_grouped_candidates=_group_candidates,
        build_simulated_lookup=_build_lookup,
        entry_display_coords=lambda _entry: None,
        peak_records=lambda: [peak_record],
    )

    cache_data = callbacks.get_pick_cache(param_set={"a": 2.0}, prefer_cache=True)

    assert cache_data["cache_metadata"]["cache_source"] == "peak_records"
    assert cache_state["signature"] == cache_data["signature"]
    assert cache_state["data"]["grouped_candidates"][("q_group", "primary", 1, 2)][0][
        "source_row_index"
    ] == 8


def test_make_runtime_geometry_manual_cache_callbacks_prefers_shared_live_preview_cache() -> None:
    cache_state = {"signature": None, "data": {}}
    forwarded_prefer_cache: list[bool] = []
    peak_record = {
        "display_col": 21.0,
        "display_row": 34.0,
        "hkl_raw": (1.0, 0.0, 2.0),
        "q_group_key": ("q_group", "primary", 1, 2),
        "source_table_index": 5,
        "source_row_index": 8,
        "qr": 1.5,
        "qz": -0.5,
        "intensity": 3.0,
    }

    def _replace_cache_state(signature, data) -> None:
        cache_state["signature"] = signature
        cache_state["data"] = dict(data)

    callbacks = mg.make_runtime_geometry_manual_cache_callbacks(
        fit_config={"geometry": {"auto_match": {"search_radius_px": 18.0}}},
        last_simulation_signature=lambda: ("sim", 3),
        current_background_index=lambda: 0,
        current_background_image=lambda: np.zeros((4, 4), dtype=float),
        use_caked_space=lambda: False,
        replace_cache_state=_replace_cache_state,
        current_geometry_fit_params=lambda: {"gamma": 1.25},
        pairs_for_index=lambda _idx: [],
        source_rows_for_background=lambda *_args, **_kwargs: [],
        simulated_peaks_for_params=lambda _params, *, prefer_cache: (
            forwarded_prefer_cache.append(bool(prefer_cache))
            or (
                [
                    {
                        "label": "cached-preview",
                        "hkl": (3, 0, 1),
                        "q_group_key": ("q_group", "primary", 3, 1),
                        "source_table_index": 7,
                        "source_row_index": 9,
                        "sim_col": 44.0,
                        "sim_row": 55.0,
                        "weight": 1.0,
                    }
                ]
                if prefer_cache
                else []
            )
        ),
        build_grouped_candidates=_group_candidates,
        build_simulated_lookup=_build_lookup,
        entry_display_coords=lambda _entry: None,
        peak_records=lambda: [peak_record],
    )

    cache_data = callbacks.get_pick_cache(param_set={"a": 2.0}, prefer_cache=True)

    assert forwarded_prefer_cache == [True]
    assert (
        cache_data["cache_metadata"]["cache_source"]
        == "geometry_manual_simulated_peaks_for_params(prefer_cache=True)"
    )
    assert ("q_group", "primary", 3, 1) in cache_data["grouped_candidates"]
    assert ("q_group", "primary", 1, 2) not in cache_data["grouped_candidates"]
    assert cache_state["signature"] == cache_data["signature"]
    assert cache_state["data"]["simulated_lookup"][(7, 9)]["sim_col"] == 44.0


def test_build_geometry_manual_pick_cache_reprojects_existing_rows_without_analytic_forward_path(
    monkeypatch,
) -> None:
    bundle = mg.CakeTransformBundle(
        detector_shape=(8, 8),
        radial_deg=np.linspace(10.0, 17.0, 8, dtype=float),
        raw_azimuth_deg=np.linspace(-4.0, 3.0, 8, dtype=float),
        gui_azimuth_deg=np.linspace(-4.0, 3.0, 8, dtype=float),
        lut=object(),
    )
    forwarded_prefer_cache: list[bool] = []
    cached_entry = {
        "label": "1,0,0",
        "q_group_key": ("q_group", "primary", 1, 0),
        "source_table_index": 1,
        "source_row_index": 2,
        "sim_col": 90.0,
        "sim_row": 80.0,
        "sim_col_raw": 3.0,
        "sim_row_raw": 4.0,
        "caked_x": 66.0,
        "caked_y": 55.0,
        "raw_caked_x": 65.0,
        "raw_caked_y": 54.0,
    }

    monkeypatch.setattr(
        mg,
        "_detector_pixel_to_caked_bin",
        lambda live_bundle, col, row: (
            (13.0, 2.0)
            if live_bundle is bundle and (float(col), float(row)) == (3.0, 4.0)
            else (None, None)
        ),
    )

    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: np.zeros((8, 8), dtype=float),
        last_caked_radial_values=lambda: np.linspace(10.0, 17.0, 8, dtype=float),
        last_caked_azimuth_values=lambda: np.linspace(-4.0, 3.0, 8, dtype=float),
        current_background_display=lambda: np.zeros((8, 8), dtype=float),
        current_background_native=lambda: np.ones((8, 8), dtype=float),
        ai=lambda: object(),
        caked_transform_bundle=lambda: bundle,
        image_size=lambda: 8,
        display_to_native_sim_coords=lambda col, row, _shape: (
            float(col),
            float(row),
        ),
        get_detector_angular_maps=_fail_projection_legacy_path(
            "detector angular maps should not be used"
        ),
        detector_pixel_to_scattering_angles=_fail_projection_legacy_path(
            "analytic forward fallback should not be used"
        ),
    )

    cache_data, next_sig, next_state = mg.build_geometry_manual_pick_cache(
        param_set={"gamma": 1.5},
        prefer_cache=True,
        background_index=0,
        current_background_index=0,
        background_image=np.ones((3, 3), dtype=float),
        existing_cache_signature=(
            ("sim", 7),
            0,
            True,
            ("old-bg",),
            1,
            ("('q_group', 'primary', 1, 0)",),
        ),
        existing_cache_data={
            "signature": (
                ("sim", 7),
                0,
                True,
                ("old-bg",),
                1,
                ("('q_group', 'primary', 1, 0)",),
            ),
            "simulated_peaks": [dict(cached_entry)],
            "simulated_lookup": {(1, 2): dict(cached_entry)},
            "grouped_candidates": {
                ("q_group", "primary", 1, 0): [dict(cached_entry)]
            },
        },
        cache_signature_fn=lambda **_kwargs: (
            ("sim", 7),
            0,
            True,
            ("new-bg",),
            1,
            ("('q_group', 'primary', 1, 0)",),
        ),
        simulated_peaks_for_params=lambda _params, *, prefer_cache: (
            forwarded_prefer_cache.append(bool(prefer_cache)) or []
        ),
        build_grouped_candidates=_group_candidates,
        build_simulated_lookup=_build_lookup,
        project_peaks_to_current_view=callbacks.project_peaks_to_current_view,
        current_match_config=lambda: {"search_radius_px": 24.0},
    )

    assert forwarded_prefer_cache == [True, False]
    assert cache_data["cache_metadata"]["cache_source"] == (
        "existing_cache_data.simulated_peaks(reprojected)"
    )
    assert cache_data["simulated_lookup"][(1, 2)]["sim_col"] == 13.0
    assert cache_data["simulated_lookup"][(1, 2)]["sim_row"] == 2.0
    assert cache_data["simulated_lookup"][(1, 2)]["sim_col_raw"] == 3.0
    assert cache_data["simulated_lookup"][(1, 2)]["sim_row_raw"] == 4.0
    assert cache_data["simulated_lookup"][(1, 2)]["caked_x"] == 13.0
    assert cache_data["simulated_lookup"][(1, 2)]["caked_y"] == 2.0
    assert next_sig[3] == ("new-bg",)
    assert next_state["grouped_candidates"][("q_group", "primary", 1, 0)][0][
        "caked_x"
    ] == 13.0


def test_geometry_manual_live_peak_candidates_normalize_branch_and_full_provenance() -> None:
    candidates = mg.geometry_manual_live_peak_candidates_from_records(
        [
            {
                "display_col": 21.0,
                "display_row": 34.0,
                "hkl": (1, 0, 2),
                "q_group_key": ("q_group", "primary", 1, 2),
                "source_table_index": 0,
                "source_row_index": 8,
                "source_peak_index": 13,
                "phi": 15.0,
            }
        ],
        source_reflection_indices_local=[7],
        source_row_hkl_lookup={(0, 8): (1, 0, 2)},
        active_signature_matches=True,
    )

    assert len(candidates) == 1
    assert candidates[0]["source_branch_index"] == 1
    assert candidates[0]["source_peak_index"] == 1
    assert candidates[0]["source_reflection_index"] == 7
    assert candidates[0]["source_reflection_namespace"] == "full_reflection"
    assert candidates[0]["source_reflection_is_full"] is True


def test_geometry_manual_live_peak_candidates_fail_closed_when_provenance_does_not_match() -> None:
    candidates = mg.geometry_manual_live_peak_candidates_from_records(
        [
            {
                "display_col": 21.0,
                "display_row": 34.0,
                "hkl": (1, 0, 2),
                "q_group_key": ("q_group", "primary", 1, 2),
                "source_table_index": 0,
                "source_row_index": 8,
                "source_peak_index": 19,
                "phi": -15.0,
            }
        ],
        source_reflection_indices_local=[7],
        source_row_hkl_lookup={(0, 8): (2, 0, 0)},
        active_signature_matches=True,
    )

    assert len(candidates) == 1
    assert candidates[0]["source_branch_index"] == 0
    assert candidates[0]["source_peak_index"] == 0
    assert "source_reflection_index" not in candidates[0]
    assert "source_reflection_namespace" not in candidates[0]
    assert "source_reflection_is_full" not in candidates[0]


def test_geometry_manual_live_peak_candidates_restore_trust_on_revision_match() -> None:
    candidates = mg.geometry_manual_live_peak_candidates_from_records(
        [
            {
                "display_col": 21.0,
                "display_row": 34.0,
                "hkl": (1, 0, 2),
                "q_group_key": ("q_group", "primary", 1, 2),
                "source_table_index": 0,
                "source_row_index": 8,
                "source_peak_index": 13,
                "phi": 15.0,
            }
        ],
        source_reflection_indices_local=[7],
        source_row_hkl_lookup={(0, 8): (1, 0, 2)},
        provenance_signature_matches=False,
        provenance_revision_matches=True,
        expected_table_count=1,
    )

    assert len(candidates) == 1
    assert candidates[0]["source_branch_index"] == 1
    assert candidates[0]["source_peak_index"] == 1
    assert candidates[0]["source_reflection_index"] == 7
    assert candidates[0]["source_reflection_namespace"] == "full_reflection"
    assert candidates[0]["source_reflection_is_full"] is True


def test_geometry_manual_live_peak_candidates_drop_trust_when_map_length_mismatches() -> None:
    candidates = mg.geometry_manual_live_peak_candidates_from_records(
        [
            {
                "display_col": 21.0,
                "display_row": 34.0,
                "hkl": (1, 0, 2),
                "q_group_key": ("q_group", "primary", 1, 2),
                "source_table_index": 0,
                "source_row_index": 8,
                "source_peak_index": 13,
                "phi": 15.0,
            }
        ],
        source_reflection_indices_local=[7],
        source_row_hkl_lookup={(0, 8): (1, 0, 2)},
        provenance_signature_matches=True,
        expected_table_count=2,
    )

    assert len(candidates) == 1
    assert candidates[0]["source_branch_index"] == 1
    assert candidates[0]["source_peak_index"] == 1
    assert "source_reflection_index" not in candidates[0]
    assert "source_reflection_namespace" not in candidates[0]
    assert "source_reflection_is_full" not in candidates[0]


def test_geometry_manual_live_peak_candidates_are_permutation_invariant() -> None:
    records = [
        {
            "display_col": 21.0,
            "display_row": 34.0,
            "hkl": (1, 0, 2),
            "q_group_key": ("q_group", "primary", 1, 2),
            "source_table_index": 0,
            "source_row_index": 8,
            "source_peak_index": 13,
            "phi": 15.0,
        },
        {
            "display_col": 25.0,
            "display_row": 31.0,
            "hkl": (1, 0, 1),
            "q_group_key": ("q_group", "primary", 1, 1),
            "source_table_index": 1,
            "source_row_index": 4,
            "source_peak_index": 18,
            "phi": -15.0,
        },
    ]

    def _digest(entries):
        return sorted(
            [
                {
                    "hkl": tuple(entry["hkl"]),
                    "source_table_index": entry.get("source_table_index"),
                    "source_row_index": entry.get("source_row_index"),
                    "source_branch_index": entry.get("source_branch_index"),
                    "source_peak_index": entry.get("source_peak_index"),
                    "source_reflection_index": entry.get("source_reflection_index"),
                    "source_reflection_namespace": entry.get("source_reflection_namespace"),
                    "source_reflection_is_full": entry.get("source_reflection_is_full"),
                }
                for entry in entries
            ],
            key=lambda item: (int(item["source_table_index"]), int(item["source_row_index"])),
        )

    forward = mg.geometry_manual_live_peak_candidates_from_records(
        records,
        source_reflection_indices_local=[7, 8],
        source_row_hkl_lookup={(0, 8): (1, 0, 2), (1, 4): (1, 0, 1)},
        provenance_signature_matches=True,
        expected_table_count=2,
    )
    reversed_order = mg.geometry_manual_live_peak_candidates_from_records(
        list(reversed(records)),
        source_reflection_indices_local=[7, 8],
        source_row_hkl_lookup={(0, 8): (1, 0, 2), (1, 4): (1, 0, 1)},
        provenance_signature_matches=True,
        expected_table_count=2,
    )

    assert _digest(forward) == _digest(reversed_order)


def test_geometry_manual_canonicalize_live_source_entry_only_repairs_trust_from_explicit_proof() -> None:
    entry = {
        "display_col": 21.0,
        "display_row": 34.0,
        "hkl": (1, 0, 2),
        "q_group_key": ("q_group", "primary", 1, 2),
        "source_table_index": 0,
        "source_row_index": 8,
        "source_peak_index": 13,
        "phi": 15.0,
    }

    repaired = mg.geometry_manual_canonicalize_live_source_entry(
        entry,
        trusted_reflection_index=7,
    )
    unrepaired = mg.geometry_manual_canonicalize_live_source_entry(entry)

    assert repaired is not None
    assert repaired["source_reflection_index"] == 7
    assert repaired["source_reflection_namespace"] == "full_reflection"
    assert repaired["source_reflection_is_full"] is True
    assert unrepaired is not None
    assert "source_reflection_index" not in unrepaired
    assert "source_reflection_namespace" not in unrepaired
    assert "source_reflection_is_full" not in unrepaired


def test_refresh_geometry_manual_pair_entry_keeps_saved_caked_angles_as_truth() -> None:
    refreshed = mg.refresh_geometry_manual_pair_entry(
        {
            "label": "0,0,3",
            "hkl": (0, 0, 3),
            "x": 30.0,
            "y": 40.0,
            "detector_x": 30.0,
            "detector_y": 40.0,
            "caked_x": 150.0,
            "caked_y": 160.0,
            "raw_caked_x": 151.0,
            "raw_caked_y": 161.0,
        },
        background_display_shape=(200, 200),
        background_display_to_native_detector_coords=lambda col, row: (
            float(col),
            float(row),
        ),
        caked_angles_to_background_display_coords=lambda two_theta, phi: (
            float(two_theta) - 10.0,
            float(phi) - 20.0,
        ),
        native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col) + 10.0,
            float(row) + 20.0,
        ),
        rotate_point_for_display=lambda col, row, _shape, _k: (float(col), float(row)),
        display_rotate_k=0,
    )

    assert refreshed is not None
    assert refreshed["detector_x"] == 140.0
    assert refreshed["detector_y"] == 140.0
    assert refreshed["x"] == 140.0
    assert refreshed["y"] == 140.0
    assert refreshed["background_two_theta_deg"] == 150.0
    assert refreshed["background_phi_deg"] == 160.0
    assert refreshed["caked_x"] == 150.0
    assert refreshed["caked_y"] == 160.0
    assert refreshed["raw_caked_x"] == 151.0
    assert refreshed["raw_caked_y"] == 161.0
    assert "stale_caked_fields" not in refreshed

    display_point = (float(refreshed["caked_x"]) - 10.0, float(refreshed["caked_y"]) - 20.0)
    assert display_point == (140.0, 140.0)


def test_refresh_geometry_manual_pair_entry_prefers_canonical_background_angles() -> None:
    refreshed = mg.refresh_geometry_manual_pair_entry(
        {
            "label": "0,0,3",
            "hkl": (0, 0, 3),
            "x": 30.0,
            "y": 40.0,
            "background_two_theta_deg": 150.0,
            "background_phi_deg": 160.0,
            "caked_x": 40.0,
            "caked_y": 50.0,
            "raw_caked_x": 41.0,
            "raw_caked_y": 51.0,
        },
        background_display_shape=(200, 200),
        background_display_to_native_detector_coords=lambda col, row: (
            float(col),
            float(row),
        ),
        caked_angles_to_background_display_coords=lambda two_theta, phi: (
            float(two_theta) - 10.0,
            float(phi) - 20.0,
        ),
        native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col) + 10.0,
            float(row) + 20.0,
        ),
        rotate_point_for_display=lambda col, row, _shape, _k: (float(col), float(row)),
        display_rotate_k=0,
    )

    assert refreshed is not None
    assert refreshed["background_two_theta_deg"] == 150.0
    assert refreshed["background_phi_deg"] == 160.0
    assert refreshed["caked_x"] == 150.0
    assert refreshed["caked_y"] == 160.0
    assert refreshed["x"] == 140.0
    assert refreshed["y"] == 140.0


def test_refresh_geometry_manual_pair_entry_uses_inverse_lut_not_analytic_inverse(
    monkeypatch,
) -> None:
    bundle = mg.CakeTransformBundle(
        detector_shape=(200, 200),
        radial_deg=np.array([10.0, 12.0], dtype=float),
        raw_azimuth_deg=np.array([-5.0, 5.0], dtype=float),
        gui_azimuth_deg=np.array([-5.0, 5.0], dtype=float),
        lut=object(),
    )
    inverse_calls: list[tuple[float, float, object]] = []

    monkeypatch.setattr(
        mg,
        "_caked_point_to_detector_pixel",
        lambda _ai, detector_shape, radial_deg, azimuth_deg, two_theta, phi, **kwargs: (
            inverse_calls.append(
                (
                    float(two_theta),
                    float(phi),
                    kwargs.get("transform_bundle"),
                )
            )
            or (140.0, 141.0)
        ),
    )

    def _caked_angles_to_background_display(two_theta: float, phi: float):
        return mg.caked_angles_to_background_display_coords(
            two_theta,
            phi,
            ai=object(),
            native_background=np.ones((200, 200), dtype=float),
            caked_radial_values=bundle.radial_deg,
            caked_azimuth_values=bundle.gui_azimuth_deg,
            get_detector_angular_maps=_fail_projection_legacy_path(
                "detector angular maps should not be used"
            ),
            scattering_angles_to_detector_pixel=_fail_projection_legacy_path(
                "analytic inverse fallback should not be used"
            ),
            center=[0.0, 0.0],
            detector_distance=1.0,
            pixel_size=1.0,
            transform_bundle=bundle,
            rotate_point_for_display=lambda col, row, _shape, _k: (
                float(col),
                float(row),
            ),
            display_rotate_k=0,
        )

    refreshed = mg.refresh_geometry_manual_pair_entry(
        {
            "label": "0,0,3",
            "hkl": (0, 0, 3),
            "x": 30.0,
            "y": 40.0,
            "detector_x": 30.0,
            "detector_y": 40.0,
            "background_two_theta_deg": 150.0,
            "background_phi_deg": 160.0,
            "caked_x": 40.0,
            "caked_y": 50.0,
            "raw_caked_x": 41.0,
            "raw_caked_y": 51.0,
        },
        background_display_shape=(200, 200),
        background_display_to_native_detector_coords=lambda col, row: (
            float(col),
            float(row),
        ),
        caked_angles_to_background_display_coords=_caked_angles_to_background_display,
        native_detector_coords_to_caked_display_coords=_fail_projection_legacy_path(
            "detector-to-caked backfill should not run for canonical angles"
        ),
        rotate_point_for_display=lambda col, row, _shape, _k: (float(col), float(row)),
        display_rotate_k=0,
    )

    assert refreshed is not None
    assert refreshed["detector_x"] == 140.0
    assert refreshed["detector_y"] == 141.0
    assert refreshed["x"] == 140.0
    assert refreshed["y"] == 141.0
    assert refreshed["background_two_theta_deg"] == 150.0
    assert refreshed["background_phi_deg"] == 160.0
    assert refreshed["caked_x"] == 150.0
    assert refreshed["caked_y"] == 160.0
    assert refreshed["raw_caked_x"] == 41.0
    assert refreshed["raw_caked_y"] == 51.0
    assert inverse_calls == [(150.0, 160.0, bundle)]


def test_refresh_geometry_manual_pair_entry_migrates_legacy_peak_branch_once() -> None:
    refreshed = mg.refresh_geometry_manual_pair_entry(
        {
            "label": "1,0,3",
            "hkl": (1, 0, 3),
            "x": 30.0,
            "y": 40.0,
            "detector_x": 30.0,
            "detector_y": 40.0,
            "source_peak_index": 1,
        },
        background_display_shape=(200, 200),
        background_display_to_native_detector_coords=lambda col, row: (
            float(col),
            float(row),
        ),
        native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col) + 1.0,
            float(row) + 2.0,
        ),
        rotate_point_for_display=lambda col, row, _shape, _k: (float(col), float(row)),
        display_rotate_k=0,
    )

    assert refreshed is not None
    assert "source_branch_index" not in refreshed
    assert "source_peak_index" not in refreshed
