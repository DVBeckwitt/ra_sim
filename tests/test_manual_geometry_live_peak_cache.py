import numpy as np

from ra_sim.gui import manual_geometry as mg


def _source_key(entry):
    return mg.geometry_manual_candidate_source_key(dict(entry) if isinstance(entry, dict) else None)


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
        key = _source_key(raw_entry)
        if key is None:
            continue
        entry = dict(raw_entry)
        existing = lookup.get(key)
        if existing is None:
            lookup[key] = entry
        elif isinstance(existing, list):
            existing.append(entry)
        else:
            lookup[key] = [dict(existing), entry]
    return lookup


def _fail_projection_legacy_path(message: str):
    def _fail(*_args, **_kwargs):
        raise AssertionError(message)

    return _fail


def _caked_projection_bundle() -> mg.CakeTransformBundle:
    return mg.CakeTransformBundle(
        detector_shape=(10, 10),
        radial_deg=np.linspace(0.0, 9.0, 10, dtype=float),
        raw_azimuth_deg=np.linspace(-4.5, 4.5, 10, dtype=float),
        gui_azimuth_deg=np.linspace(-4.5, 4.5, 10, dtype=float),
        lut=object(),
    )


def _caked_projection_payload(
    bundle: mg.CakeTransformBundle,
    *,
    signature: object,
) -> dict[str, object]:
    return {
        "payload_kind": "projection",
        "signature": signature,
        "detector_shape": bundle.detector_shape,
        "radial_axis": bundle.radial_deg,
        "azimuth_axis": bundle.gui_azimuth_deg,
        "raw_azimuth_axis": bundle.raw_azimuth_deg,
        "raw_to_gui_row_permutation": np.arange(len(bundle.gui_azimuth_deg), dtype=np.int32),
        "transform_bundle": bundle,
    }


def test_build_geometry_manual_pick_cache_falls_back_to_live_peak_records() -> None:
    peak_record = {
        "display_col": 13.5,
        "display_row": 15.5,
        "native_col": 13.5,
        "native_row": 15.5,
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
    assert group_entry["display_col"] == 13.5
    assert group_entry["display_row"] == 15.5
    assert group_entry["native_col"] == 13.5
    assert group_entry["native_row"] == 15.5
    assert "sim_col" not in group_entry
    assert np.isclose(float(group_entry["qr"]), 1.2345678901)
    assert np.isclose(
        float(next_state["simulated_lookup"][_source_key(peak_record)]["qz"]),
        -0.4567890123,
    )
    assert next_sig == ("sig",)


def test_make_runtime_geometry_manual_cache_callbacks_uses_live_peak_records() -> None:
    cache_state = {"signature": None, "data": {}}
    peak_record = {
        "display_col": 21.0,
        "display_row": 34.0,
        "native_col": 21.0,
        "native_row": 34.0,
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
    assert (
        cache_state["data"]["grouped_candidates"][("q_group", "primary", 1, 2)][0][
            "source_row_index"
        ]
        == 8
    )


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
    assert (
        cache_state["data"]["simulated_lookup"][
            _source_key(
                {
                    "source_table_index": 7,
                    "source_row_index": 9,
                }
            )
        ]["sim_col"]
        == 44.0
    )


def test_refine_qr_sim_candidates_in_cache_skips_when_refinement_signature_matches(
    monkeypatch,
) -> None:
    calls: list[str] = []
    signature = ("refinement", 1)
    candidate = {
        "q_group_key": ("q_group", "primary", 1, 2),
        "hkl": (-1, 0, 2),
        "source_table_index": 4,
        "source_row_index": 5,
        "display_col": 5.0,
        "display_row": 6.0,
    }

    def _count_refine(candidate, *, view_mode, **_kwargs):
        calls.append(str(view_mode))
        return {**dict(candidate), "refined": True}

    monkeypatch.setattr(mg, "geometry_manual_refine_qr_sim_peak_for_view", _count_refine)
    warm_cache = {
        "simulated_peaks": [dict(candidate)],
        "qr_sim_refinement_signature": signature,
        "qr_sim_refinement_complete": True,
    }

    skipped = mg.geometry_manual_refine_qr_sim_candidates_in_cache(
        warm_cache,
        refinement_signature=signature,
    )
    direct = mg.geometry_manual_refine_qr_sim_candidates_in_cache(warm_cache)

    assert calls == ["detector"]
    assert "refined" not in skipped["simulated_peaks"][0]
    assert direct["simulated_peaks"][0]["refined"] is True


def test_refine_qr_sim_candidates_direct_no_signature_clears_stale_refinement_metadata(
    monkeypatch,
) -> None:
    calls: list[str] = []
    candidate = {
        "q_group_key": ("q_group", "primary", 1, 2),
        "hkl": (-1, 0, 2),
        "source_table_index": 4,
        "source_row_index": 5,
        "display_col": 5.0,
        "display_row": 6.0,
    }

    def _count_refine(candidate, *, view_mode, **_kwargs):
        calls.append(str(view_mode))
        return {**dict(candidate), "refined": True}

    monkeypatch.setattr(mg, "geometry_manual_refine_qr_sim_peak_for_view", _count_refine)

    result = mg.geometry_manual_refine_qr_sim_candidates_in_cache(
        {
            "active_simulated_peaks": [dict(candidate)],
            "qr_sim_refinement_signature": ("old",),
            "qr_sim_refinement_complete": True,
        }
    )

    assert calls == ["detector"]
    assert result["active_simulated_peaks"][0]["refined"] is True
    assert "qr_sim_refinement_signature" not in result
    assert "qr_sim_refinement_complete" not in result


def test_rebuild_refined_qr_cache_lookups_skips_when_refinement_signature_matches() -> None:
    calls: list[int] = []
    signature = ("refinement", 2)
    candidate = {
        "q_group_key": ("q_group", "primary", 1, 2),
        "hkl": (-1, 0, 2),
        "source_table_index": 4,
        "source_row_index": 5,
        "display_col": 5.0,
        "display_row": 6.0,
    }
    existing_lookup = _build_lookup([candidate])

    def _count_build_lookup(entries):
        calls.append(len(list(entries or ())))
        return _build_lookup(entries)

    skipped = mg.geometry_manual_rebuild_refined_qr_cache_lookups(
        {
            "active_simulated_peaks": [dict(candidate)],
            "simulated_lookup": existing_lookup,
            "qr_sim_refinement_lookup_signature": signature,
            "qr_sim_refinement_lookup_complete": True,
        },
        _count_build_lookup,
        refinement_signature=signature,
    )
    rebuilt_missing = mg.geometry_manual_rebuild_refined_qr_cache_lookups(
        {
            "active_simulated_peaks": [dict(candidate)],
            "qr_sim_refinement_lookup_signature": signature,
        },
        _count_build_lookup,
        refinement_signature=signature,
    )
    rebuilt_direct = mg.geometry_manual_rebuild_refined_qr_cache_lookups(
        {
            "active_simulated_peaks": [dict(candidate)],
            "simulated_lookup": existing_lookup,
            "qr_sim_refinement_lookup_signature": signature,
            "qr_sim_refinement_lookup_complete": True,
        },
        _count_build_lookup,
    )

    assert calls == [1, 1]
    assert skipped["simulated_lookup"] == existing_lookup
    assert _source_key(candidate) in rebuilt_missing["simulated_lookup"]
    assert rebuilt_missing["qr_sim_refinement_lookup_complete"] is True
    assert _source_key(candidate) in rebuilt_direct["simulated_lookup"]
    assert "qr_sim_refinement_lookup_signature" not in rebuilt_direct
    assert rebuilt_direct["qr_sim_refinement_lookup_complete"] is False


def test_rebuild_refined_qr_cache_lookups_direct_no_signature_clears_stale_lookup_metadata() -> (
    None
):
    candidate = {
        "q_group_key": ("q_group", "primary", 1, 2),
        "hkl": (-1, 0, 2),
        "source_table_index": 4,
        "source_row_index": 5,
        "display_col": 5.0,
        "display_row": 6.0,
    }
    existing_lookup = _build_lookup([candidate])
    calls: list[int] = []

    def _working_lookup(entries):
        calls.append(len(list(entries or ())))
        return _build_lookup(entries)

    result = mg.geometry_manual_rebuild_refined_qr_cache_lookups(
        {
            "active_simulated_peaks": [dict(candidate)],
            "simulated_lookup": existing_lookup,
            "qr_sim_refinement_lookup_signature": ("old",),
            "qr_sim_refinement_lookup_complete": True,
        },
        _working_lookup,
    )

    assert calls == [1]
    assert _source_key(candidate) in result["simulated_lookup"]
    assert "qr_sim_refinement_lookup_signature" not in result
    assert result["qr_sim_refinement_lookup_complete"] is False


def test_rebuild_refined_qr_cache_lookups_no_signature_failure_does_not_allow_later_skip() -> None:
    signature = ("old",)
    candidate = {
        "q_group_key": ("q_group", "primary", 1, 2),
        "hkl": (-1, 0, 2),
        "source_table_index": 4,
        "source_row_index": 5,
        "display_col": 5.0,
        "display_row": 6.0,
    }
    calls: list[str] = []

    def _raising_lookup(_entries):
        calls.append("raise")
        raise RuntimeError("lookup unavailable")

    def _working_lookup(entries):
        calls.append("work")
        return _build_lookup(entries)

    failed = mg.geometry_manual_rebuild_refined_qr_cache_lookups(
        {
            "active_simulated_peaks": [dict(candidate)],
            "simulated_lookup": _build_lookup([candidate]),
            "qr_sim_refinement_lookup_signature": signature,
            "qr_sim_refinement_lookup_complete": True,
        },
        _raising_lookup,
    )
    rebuilt = mg.geometry_manual_rebuild_refined_qr_cache_lookups(
        failed,
        _working_lookup,
        refinement_signature=signature,
    )

    assert calls == ["raise", "work"]
    assert failed["simulated_lookup"] == {}
    assert "qr_sim_refinement_lookup_signature" not in failed
    assert failed["qr_sim_refinement_lookup_complete"] is False
    assert _source_key(candidate) in rebuilt["simulated_lookup"]
    assert rebuilt["qr_sim_refinement_lookup_signature"] == signature
    assert rebuilt["qr_sim_refinement_lookup_complete"] is True


def test_rebuild_refined_qr_cache_lookups_retries_after_failed_signature_build() -> None:
    signature = ("refinement", 3)
    candidate = {
        "q_group_key": ("q_group", "primary", 1, 2),
        "hkl": (-1, 0, 2),
        "source_table_index": 4,
        "source_row_index": 5,
        "display_col": 5.0,
        "display_row": 6.0,
    }
    calls: list[str] = []

    def _raising_lookup(_entries):
        calls.append("raise")
        raise RuntimeError("lookup unavailable")

    def _working_lookup(entries):
        calls.append("work")
        return _build_lookup(entries)

    failed = mg.geometry_manual_rebuild_refined_qr_cache_lookups(
        {"active_simulated_peaks": [dict(candidate)]},
        _raising_lookup,
        refinement_signature=signature,
    )
    rebuilt = mg.geometry_manual_rebuild_refined_qr_cache_lookups(
        failed,
        _working_lookup,
        refinement_signature=signature,
    )

    assert calls == ["raise", "work"]
    assert failed["simulated_lookup"] == {}
    assert failed["qr_sim_refinement_lookup_complete"] is False
    assert failed.get("qr_sim_refinement_lookup_signature") != signature
    assert _source_key(candidate) in rebuilt["simulated_lookup"]
    assert rebuilt["qr_sim_refinement_lookup_signature"] == signature
    assert rebuilt["qr_sim_refinement_lookup_complete"] is True


def test_get_pick_cache_refines_same_next_cache_object_only_once(monkeypatch) -> None:
    calls: list[str] = []
    cache_state = {"signature": None, "data": {}}
    candidate = {
        "q_group_key": ("q_group", "primary", 1, 2),
        "hkl": (-1, 0, 2),
        "source_table_index": 4,
        "source_row_index": 5,
        "display_col": 5.0,
        "display_row": 6.0,
    }
    raw_cache = {"signature": ("manual-cache", 1), "simulated_peaks": [dict(candidate)]}

    def _fake_build_pick_cache(**_kwargs):
        return raw_cache, raw_cache["signature"], raw_cache

    def _count_refine(candidate, *, view_mode, **_kwargs):
        calls.append(str(view_mode))
        return dict(candidate)

    monkeypatch.setattr(mg, "build_geometry_manual_pick_cache", _fake_build_pick_cache)
    monkeypatch.setattr(mg, "geometry_manual_refine_qr_sim_peak_for_view", _count_refine)

    callbacks = mg.make_runtime_geometry_manual_cache_callbacks(
        fit_config={},
        last_simulation_signature=lambda: ("sim", 1),
        current_background_index=lambda: 0,
        current_background_image=lambda: np.zeros((4, 4), dtype=float),
        use_caked_space=lambda: False,
        replace_cache_state=lambda signature, data: cache_state.update(
            {"signature": signature, "data": dict(data)}
        ),
        current_geometry_fit_params=lambda: {},
        pairs_for_index=lambda _idx: [],
        build_grouped_candidates=_group_candidates,
        build_simulated_lookup=_build_lookup,
        entry_display_coords=lambda _entry: None,
    )

    cache_data = callbacks.get_pick_cache(prefer_cache=True)

    assert calls == ["detector"]
    assert (
        cache_state["data"]["qr_sim_refinement_signature"]
        == cache_data["qr_sim_refinement_signature"]
    )


def test_get_pick_cache_skips_row_refinement_on_warm_cache(monkeypatch) -> None:
    calls: list[str] = []
    background = np.zeros((10, 10), dtype=float)
    cache_state: dict[str, object] = {"signature": None, "data": {}}
    candidate = {
        "q_group_key": ("q_group", "primary", 1, 2),
        "hkl": (-1, 0, 2),
        "source_table_index": 4,
        "source_row_index": 5,
        "source_reflection_index": 42,
        "source_branch_index": 0,
        "source_ray_id": "ray-0",
        "branch_id": "branch-0",
        "display_col": 5.0,
        "display_row": 6.0,
        "sim_col": 5.0,
        "sim_row": 6.0,
        "sim_col_raw": 5.0,
        "sim_row_raw": 6.0,
        "native_col": 5.0,
        "native_row": 6.0,
        "qr": 1.0,
        "qz": 2.0,
        "weight": 1.0,
    }

    def _replace_cache_state(signature, data) -> None:
        cache_state["signature"] = signature
        cache_state["data"] = dict(data)

    def _count_refine(candidate, *, view_mode, **_kwargs):
        calls.append(str(view_mode))
        return dict(candidate)

    monkeypatch.setattr(mg, "geometry_manual_refine_qr_sim_peak_for_view", _count_refine)

    callbacks = mg.make_runtime_geometry_manual_cache_callbacks(
        fit_config={"geometry": {"auto_match": {"search_radius_px": 18.0}}},
        last_simulation_signature=lambda: ("sim", 1),
        current_background_index=lambda: 0,
        current_background_image=lambda: background,
        use_caked_space=lambda: False,
        replace_cache_state=_replace_cache_state,
        current_geometry_fit_params=lambda: {"a": 2.0},
        pairs_for_index=lambda _idx: [],
        simulated_peaks_for_params=lambda _params, *, prefer_cache: [dict(candidate)],
        build_grouped_candidates=_group_candidates,
        build_simulated_lookup=_build_lookup,
        entry_display_coords=lambda _entry: None,
        current_cache_signature=lambda: cache_state["signature"],
        current_cache_data=lambda: cache_state["data"],
    )

    first_cache = callbacks.get_pick_cache(param_set={"a": 2.0}, prefer_cache=True)
    first_call_count = len(calls)
    second_cache = callbacks.get_pick_cache(param_set={"a": 2.0}, prefer_cache=True)

    assert first_call_count > 0
    assert len(calls) == first_call_count
    assert (
        cache_state["data"]["qr_sim_refinement_signature"]
        == first_cache["qr_sim_refinement_signature"]
    )
    assert (
        cache_state["data"]["qr_sim_refinement_lookup_signature"]
        == first_cache["qr_sim_refinement_signature"]
    )
    assert second_cache["qr_sim_refinement_signature"] == first_cache["qr_sim_refinement_signature"]


def test_get_pick_cache_skips_caked_row_refinement_on_warm_cache(monkeypatch) -> None:
    calls: list[str] = []
    background = np.zeros((10, 10), dtype=float)
    radial = np.linspace(0.0, 9.0, 10)
    azimuth = np.linspace(0.0, 9.0, 10)
    caked_image = np.zeros((10, 10), dtype=float)
    caked_image[4, 7] = 100.0
    cache_state: dict[str, object] = {"signature": None, "data": {}}
    candidate = {
        "q_group_key": ("q_group", "primary", 1, 2),
        "hkl": (-1, 0, 2),
        "source_table_index": 4,
        "source_row_index": 5,
        "source_reflection_index": 42,
        "source_branch_index": 0,
        "source_ray_id": "ray-0",
        "branch_id": "branch-0",
        "display_col": 5.0,
        "display_row": 6.0,
        "sim_col": 5.0,
        "sim_row": 6.0,
        "sim_col_raw": 5.0,
        "sim_row_raw": 6.0,
        "native_col": 5.0,
        "native_row": 6.0,
        "qr": 1.0,
        "qz": 2.0,
        "weight": 1.0,
    }

    def _replace_cache_state(signature, data) -> None:
        cache_state["signature"] = signature
        cache_state["data"] = dict(data)

    def _project_to_caked(rows):
        return [
            {
                **dict(entry),
                "display_col": 7.0,
                "display_row": 4.0,
                "caked_x": 7.0,
                "caked_y": 4.0,
                "raw_caked_x": 7.0,
                "raw_caked_y": 4.0,
                "two_theta_deg": 7.0,
                "phi_deg": 4.0,
            }
            for entry in rows or ()
            if isinstance(entry, dict)
        ]

    def _count_refine(candidate, *, view_mode, **_kwargs):
        calls.append(str(view_mode))
        return dict(candidate)

    monkeypatch.setattr(mg, "geometry_manual_refine_qr_sim_peak_for_view", _count_refine)

    callbacks = mg.make_runtime_geometry_manual_cache_callbacks(
        fit_config={"geometry": {"auto_match": {"search_radius_px": 18.0}}},
        last_simulation_signature=lambda: ("sim", 1),
        current_background_index=lambda: 0,
        current_background_image=lambda: background,
        use_caked_space=lambda: True,
        replace_cache_state=_replace_cache_state,
        current_geometry_fit_params=lambda: {"a": 2.0},
        pairs_for_index=lambda _idx: [],
        simulated_peaks_for_params=lambda _params, *, prefer_cache: [dict(candidate)],
        build_grouped_candidates=_group_candidates,
        build_simulated_lookup=_build_lookup,
        entry_display_coords=lambda _entry: None,
        current_cache_signature=lambda: cache_state["signature"],
        current_cache_data=lambda: cache_state["data"],
        caked_projection_signature=lambda: ("caked-projection", 1),
        caked_simulation_image=lambda: caked_image,
        radial_axis=lambda: radial,
        azimuth_axis=lambda: azimuth,
    )

    first_cache = callbacks.get_pick_cache(
        param_set={"a": 2.0},
        prefer_cache=True,
        build_caked_projection_sidecar=True,
        project_peaks_to_caked_view=_project_to_caked,
    )
    first_caked_call_count = calls.count("caked")
    second_cache = callbacks.get_pick_cache(
        param_set={"a": 2.0},
        prefer_cache=True,
        build_caked_projection_sidecar=True,
        project_peaks_to_caked_view=_project_to_caked,
    )

    assert first_caked_call_count > 0
    assert calls.count("caked") == first_caked_call_count
    assert first_cache["caked_qr_projection_entries"]
    assert first_cache["caked_qr_projection_grouped_candidates"]
    assert first_cache["caked_qr_projection_lookup"]
    assert second_cache["caked_qr_projection_lookup"] == first_cache["caked_qr_projection_lookup"]
    assert (
        second_cache["qr_sim_refinement_lookup_signature"]
        == first_cache["qr_sim_refinement_signature"]
    )
    assert second_cache["qr_sim_refinement_lookup_complete"] is True


def test_caked_warm_pick_cache_skips_refinement_with_projection_only_payload(
    monkeypatch,
) -> None:
    calls: list[str] = []
    bundle = _caked_projection_bundle()
    payload = _caked_projection_payload(bundle, signature=("projection", "A"))
    projection_callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: None,
        last_caked_radial_values=lambda: payload["radial_axis"],
        last_caked_azimuth_values=lambda: payload["azimuth_axis"],
        current_background_display=lambda: np.zeros((10, 10), dtype=float),
        current_background_native=lambda: np.ones((10, 10), dtype=float),
        current_background_index=lambda: 0,
        caked_projection_payload=lambda: payload,
        caked_transform_bundle=lambda: bundle,
        image_size=lambda: 10,
    )
    projection_signature = projection_callbacks.caked_projection_signature()
    cache_state: dict[str, object] = {"signature": None, "data": {}}
    candidate = {
        "q_group_key": ("q_group", "primary", 1, 2),
        "hkl": (-1, 0, 2),
        "source_table_index": 4,
        "source_row_index": 5,
        "source_reflection_index": 42,
        "source_branch_index": 0,
        "source_ray_id": "ray-0",
        "branch_id": "branch-0",
        "display_col": 5.0,
        "display_row": 6.0,
        "sim_col": 5.0,
        "sim_row": 6.0,
        "sim_col_raw": 5.0,
        "sim_row_raw": 6.0,
        "native_col": 5.0,
        "native_row": 6.0,
        "qr": 1.0,
        "qz": 2.0,
        "weight": 1.0,
    }

    def _project_to_caked(rows):
        return [
            {
                **dict(entry),
                "display_col": 7.0,
                "display_row": 4.0,
                "caked_x": 7.0,
                "caked_y": 4.0,
                "raw_caked_x": 7.0,
                "raw_caked_y": 4.0,
                "two_theta_deg": 7.0,
                "phi_deg": 4.0,
            }
            for entry in rows or ()
            if isinstance(entry, dict)
        ]

    def _count_refine(candidate, *, view_mode, **_kwargs):
        calls.append(str(view_mode))
        return dict(candidate)

    monkeypatch.setattr(mg, "geometry_manual_refine_qr_sim_peak_for_view", _count_refine)
    callbacks = mg.make_runtime_geometry_manual_cache_callbacks(
        fit_config={"geometry": {"auto_match": {"search_radius_px": 18.0}}},
        last_simulation_signature=lambda: ("sim", 1),
        current_background_index=lambda: 0,
        current_background_image=lambda: None,
        use_caked_space=lambda: True,
        replace_cache_state=lambda signature, data: cache_state.update(
            {"signature": signature, "data": dict(data)}
        ),
        current_geometry_fit_params=lambda: {"a": 2.0},
        pairs_for_index=lambda _idx: [],
        simulated_peaks_for_params=lambda _params, *, prefer_cache: [dict(candidate)],
        build_grouped_candidates=_group_candidates,
        build_simulated_lookup=_build_lookup,
        entry_display_coords=lambda _entry: None,
        current_cache_signature=lambda: cache_state["signature"],
        current_cache_data=lambda: cache_state["data"],
        caked_projection_signature=projection_callbacks.caked_projection_signature,
        caked_simulation_image=lambda: None,
        radial_axis=lambda: payload["radial_axis"],
        azimuth_axis=lambda: payload["azimuth_axis"],
    )

    first_cache = callbacks.get_pick_cache(
        param_set={"a": 2.0},
        prefer_cache=True,
        build_caked_projection_sidecar=True,
        project_peaks_to_caked_view=_project_to_caked,
    )
    first_caked_call_count = calls.count("caked")
    second_cache = callbacks.get_pick_cache(
        param_set={"a": 2.0},
        prefer_cache=True,
        build_caked_projection_sidecar=True,
        project_peaks_to_caked_view=_project_to_caked,
    )

    assert first_caked_call_count > 0
    assert calls.count("caked") == first_caked_call_count
    assert first_cache["caked_qr_projection_lookup"]
    assert second_cache["caked_qr_projection_lookup"] == first_cache["caked_qr_projection_lookup"]
    assert projection_signature in first_cache["qr_sim_refinement_signature"]
    assert second_cache["qr_sim_refinement_lookup_complete"] is True


def test_caked_pick_cache_invalidates_when_projection_payload_signature_changes(
    monkeypatch,
) -> None:
    calls: list[str] = []
    projection_signature_state: dict[str, object] = {"value": ("projection", "A")}
    cache_state: dict[str, object] = {"signature": None, "data": {}}
    candidate = {
        "q_group_key": ("q_group", "primary", 1, 2),
        "hkl": (-1, 0, 2),
        "source_table_index": 4,
        "source_row_index": 5,
        "source_reflection_index": 42,
        "source_branch_index": 0,
        "source_ray_id": "ray-0",
        "branch_id": "branch-0",
        "display_col": 5.0,
        "display_row": 6.0,
        "sim_col": 5.0,
        "sim_row": 6.0,
        "sim_col_raw": 5.0,
        "sim_row_raw": 6.0,
        "native_col": 5.0,
        "native_row": 6.0,
        "qr": 1.0,
        "qz": 2.0,
        "weight": 1.0,
    }

    def _project_to_caked(rows):
        return [
            {
                **dict(entry),
                "display_col": 7.0,
                "display_row": 4.0,
                "caked_x": 7.0,
                "caked_y": 4.0,
            }
            for entry in rows or ()
            if isinstance(entry, dict)
        ]

    def _count_refine(candidate, *, view_mode, **_kwargs):
        calls.append(str(view_mode))
        return dict(candidate)

    monkeypatch.setattr(mg, "geometry_manual_refine_qr_sim_peak_for_view", _count_refine)
    callbacks = mg.make_runtime_geometry_manual_cache_callbacks(
        fit_config={"geometry": {"auto_match": {"search_radius_px": 18.0}}},
        last_simulation_signature=lambda: ("sim", 1),
        current_background_index=lambda: 0,
        current_background_image=lambda: None,
        use_caked_space=lambda: True,
        replace_cache_state=lambda signature, data: cache_state.update(
            {"signature": signature, "data": dict(data)}
        ),
        current_geometry_fit_params=lambda: {"a": 2.0},
        pairs_for_index=lambda _idx: [],
        simulated_peaks_for_params=lambda _params, *, prefer_cache: [dict(candidate)],
        build_grouped_candidates=_group_candidates,
        build_simulated_lookup=_build_lookup,
        entry_display_coords=lambda _entry: None,
        current_cache_signature=lambda: cache_state["signature"],
        current_cache_data=lambda: cache_state["data"],
        caked_projection_signature=lambda: projection_signature_state["value"],
    )

    first_cache = callbacks.get_pick_cache(
        param_set={"a": 2.0},
        prefer_cache=True,
        build_caked_projection_sidecar=True,
        project_peaks_to_caked_view=_project_to_caked,
    )
    first_caked_call_count = calls.count("caked")
    callbacks.get_pick_cache(
        param_set={"a": 2.0},
        prefer_cache=True,
        build_caked_projection_sidecar=True,
        project_peaks_to_caked_view=_project_to_caked,
    )
    projection_signature_state["value"] = ("projection", "B")
    third_cache = callbacks.get_pick_cache(
        param_set={"a": 2.0},
        prefer_cache=True,
        build_caked_projection_sidecar=True,
        project_peaks_to_caked_view=_project_to_caked,
    )

    assert first_caked_call_count > 0
    assert calls.count("caked") == first_caked_call_count * 2
    assert ("projection", "A") in first_cache["qr_sim_refinement_signature"]
    assert ("projection", "B") in third_cache["qr_sim_refinement_signature"]


def test_display_density_zero_support_nan_sanitization_does_not_churn_projection_cache() -> None:
    bundle = _caked_projection_bundle()
    payload = _caked_projection_payload(bundle, signature=("projection", "stable"))
    display_state = {"image": np.full((10, 10), np.nan, dtype=float)}
    native_state = {"image": np.ones((10, 10), dtype=float)}
    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: display_state["image"],
        last_caked_radial_values=lambda: payload["radial_axis"],
        last_caked_azimuth_values=lambda: payload["azimuth_axis"],
        current_background_display=lambda: display_state["image"],
        current_background_native=lambda: native_state["image"],
        current_background_index=lambda: 0,
        caked_projection_payload=lambda: payload,
        caked_transform_bundle=lambda: bundle,
        image_size=lambda: 10,
    )

    first_signature = callbacks.caked_projection_signature()
    display_state["image"] = np.nan_to_num(display_state["image"], nan=0.0)
    native_state["image"] = np.zeros((10, 10), dtype=float)
    second_signature = callbacks.caked_projection_signature()

    assert first_signature == second_signature


def test_projection_payload_background_index_is_not_cross_reused() -> None:
    bundle = _caked_projection_bundle()
    payload = _caked_projection_payload(bundle, signature=("projection", "stable"))
    background_state = {"index": 0}
    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: True,
        last_caked_background_image_unscaled=lambda: None,
        last_caked_radial_values=lambda: payload["radial_axis"],
        last_caked_azimuth_values=lambda: payload["azimuth_axis"],
        current_background_display=lambda: np.zeros((10, 10), dtype=float),
        current_background_native=lambda: np.ones((10, 10), dtype=float),
        current_background_index=lambda: background_state["index"],
        caked_projection_payload=lambda: payload,
        caked_transform_bundle=lambda: bundle,
        image_size=lambda: 10,
    )

    signature_zero = callbacks.caked_projection_signature()
    background_state["index"] = 1
    signature_one = callbacks.caked_projection_signature()

    assert signature_zero != signature_one
    assert signature_zero[1] == 0
    assert signature_one[1] == 1


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
            "simulated_lookup": {_source_key(cached_entry): dict(cached_entry)},
            "grouped_candidates": {("q_group", "primary", 1, 0): [dict(cached_entry)]},
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
    cache_entry = cache_data["simulated_lookup"][_source_key(cached_entry)]
    assert cache_entry["sim_col"] == 3.0
    assert cache_entry["sim_row"] == 4.0
    assert cache_entry["display_col"] == 13.0
    assert cache_entry["display_row"] == 2.0
    assert cache_entry["sim_col_raw"] == 3.0
    assert cache_entry["sim_row_raw"] == 4.0


def test_build_geometry_manual_pick_cache_peak_record_fallback_uses_peak_lookup_projection_when_available() -> (
    None
):
    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: None,
        last_caked_radial_values=lambda: np.array([], dtype=float),
        last_caked_azimuth_values=lambda: np.array([], dtype=float),
        current_background_display=lambda: np.zeros((8, 8), dtype=float),
        current_background_native=lambda: np.zeros((8, 8), dtype=float),
        image_size=lambda: 8,
        display_to_native_sim_coords=lambda *_args: (_ for _ in ()).throw(
            AssertionError("native detector coords should still drive display reprojection")
        ),
        native_sim_to_display_coords=lambda col, row, _shape: (
            float(col) + 100.0,
            float(row) + 200.0,
        ),
        rotate_point_for_display=lambda col, row, _shape, _k: (
            float(col),
            float(row),
        ),
        display_rotate_k=0,
        get_detector_angular_maps=_fail_projection_legacy_path(
            "detector angular maps should not be used"
        ),
        detector_pixel_to_scattering_angles=_fail_projection_legacy_path(
            "analytic forward fallback should not be used"
        ),
    )

    peak_record = {
        "display_col": 30.25,
        "display_row": -57.5,
        "native_col": 6.0,
        "native_row": 7.0,
        "hkl": (-1, 0, 5),
        "q_group_key": ("q_group", "primary", 1, 5),
        "source_table_index": 9,
        "source_row_index": 0,
        "source_branch_index": 1,
        "intensity": 3.0,
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
        project_peaks_to_current_view=callbacks.project_peaks_to_current_view,
    )

    cache_entry = cache_data["simulated_lookup"][_source_key(peak_record)]
    assert cache_entry["native_col"] == 6.0
    assert cache_entry["native_row"] == 7.0
    assert cache_entry["sim_col"] == 106.0
    assert cache_entry["sim_row"] == 207.0
    assert cache_entry["sim_col_raw"] == 106.0
    assert cache_entry["sim_row_raw"] == 207.0
    assert cache_entry["display_col"] == 106.0
    assert cache_entry["display_row"] == 207.0
    assert next_sig == ("sig",)
    grouped_entry = next_state["grouped_candidates"][peak_record["q_group_key"]][0]
    assert grouped_entry["sim_col"] == 106.0
    assert grouped_entry["sim_row"] == 207.0
    assert grouped_entry["display_col"] == 106.0
    assert grouped_entry["display_row"] == 207.0


def test_build_geometry_manual_pick_cache_peak_record_fallback_reprojects_detector_from_native_fields() -> (
    None
):
    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: None,
        last_caked_radial_values=lambda: np.array([], dtype=float),
        last_caked_azimuth_values=lambda: np.array([], dtype=float),
        current_background_display=lambda: np.zeros((8, 8), dtype=float),
        current_background_native=lambda: np.zeros((8, 8), dtype=float),
        image_size=lambda: 8,
        display_to_native_sim_coords=lambda *_args: (_ for _ in ()).throw(
            AssertionError("native detector coords should drive detector reprojection")
        ),
        get_detector_angular_maps=_fail_projection_legacy_path(
            "detector angular maps should not be used"
        ),
        detector_pixel_to_scattering_angles=_fail_projection_legacy_path(
            "analytic forward fallback should not be used"
        ),
    )

    peak_record = {
        "display_col": 30.25,
        "display_row": -57.5,
        "native_col": 6.0,
        "native_row": 7.0,
        "two_theta_deg": 30.25,
        "phi_deg": -57.5,
        "hkl": (-1, 0, 5),
        "q_group_key": ("q_group", "primary", 1, 5),
        "source_table_index": 9,
        "source_row_index": 0,
        "source_branch_index": 1,
        "intensity": 3.0,
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
        project_peaks_to_current_view=callbacks.project_peaks_to_current_view,
    )

    cache_entry = cache_data["simulated_lookup"][_source_key(peak_record)]
    assert cache_entry["sim_col"] == 6.0
    assert cache_entry["sim_row"] == 7.0
    assert cache_entry["sim_col_raw"] == 6.0
    assert cache_entry["sim_row_raw"] == 7.0
    assert cache_entry["native_col"] == 6.0
    assert cache_entry["native_row"] == 7.0
    assert cache_entry["display_col"] == 6.0
    assert cache_entry["display_row"] == 7.0
    assert "caked_x" not in cache_entry
    assert "caked_y" not in cache_entry
    assert next_sig == ("sig",)
    grouped_entry = next_state["grouped_candidates"][peak_record["q_group_key"]][0]
    assert grouped_entry["sim_col"] == 6.0
    assert grouped_entry["sim_row"] == 7.0
    assert grouped_entry["display_col"] == 6.0
    assert grouped_entry["display_row"] == 7.0
    assert "caked_x" not in grouped_entry


def test_geometry_manual_live_peak_candidates_from_records_does_not_promote_caked_display_into_detector_truth() -> (
    None
):
    peak_record = {
        "display_col": 30.25,
        "display_row": -57.5,
        "caked_x": 30.25,
        "caked_y": -57.5,
        "hkl": (-1, 0, 5),
        "q_group_key": ("q_group", "primary", 1, 5),
        "source_table_index": 9,
        "source_row_index": 0,
        "source_branch_index": 1,
        "intensity": 3.0,
    }

    candidates = mg.geometry_manual_live_peak_candidates_from_records([peak_record])

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["display_col"] == 30.25
    assert candidate["display_row"] == -57.5
    assert candidate["caked_x"] == 30.25
    assert candidate["caked_y"] == -57.5
    assert "sim_col" not in candidate
    assert "sim_row" not in candidate
    assert "sim_col_raw" not in candidate
    assert "sim_row_raw" not in candidate
    assert "native_col" not in candidate
    assert "native_row" not in candidate


def test_geometry_manual_live_peak_candidates_from_records_preserves_native_and_display_without_detector_truth() -> (
    None
):
    peak_record = {
        "display_col": 13.0,
        "display_row": 2.0,
        "native_col": 6.0,
        "native_row": 7.0,
        "caked_x": 13.0,
        "caked_y": 2.0,
        "hkl": (-1, 0, 5),
        "q_group_key": ("q_group", "primary", 1, 5),
        "source_table_index": 9,
        "source_row_index": 0,
        "source_branch_index": 1,
        "intensity": 3.0,
    }

    candidates = mg.geometry_manual_live_peak_candidates_from_records([peak_record])

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["display_col"] == 13.0
    assert candidate["display_row"] == 2.0
    assert candidate["native_col"] == 6.0
    assert candidate["native_row"] == 7.0
    assert candidate["sim_native_x"] == 6.0
    assert candidate["sim_native_y"] == 7.0
    assert candidate["caked_x"] == 13.0
    assert candidate["caked_y"] == 2.0
    assert "sim_col" not in candidate
    assert "sim_row" not in candidate
    assert "sim_col_raw" not in candidate
    assert "sim_row_raw" not in candidate


def test_geometry_manual_live_peak_candidates_from_records_skips_display_only_records() -> None:
    peak_record = {
        "display_col": 30.25,
        "display_row": -57.5,
        "hkl": (-1, 0, 5),
        "q_group_key": ("q_group", "primary", 1, 5),
        "source_table_index": 9,
        "source_row_index": 0,
        "source_branch_index": 1,
        "intensity": 3.0,
    }

    assert mg.geometry_manual_live_peak_candidates_from_records([peak_record]) == []


def test_project_peaks_to_current_view_does_not_promote_transient_display_into_detector_truth() -> (
    None
):
    callbacks = mg.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: False,
        last_caked_background_image_unscaled=lambda: None,
        last_caked_radial_values=lambda: np.array([], dtype=float),
        last_caked_azimuth_values=lambda: np.array([], dtype=float),
        current_background_display=lambda: np.zeros((8, 8), dtype=float),
        current_background_native=lambda: np.zeros((8, 8), dtype=float),
        image_size=lambda: 8,
        display_to_native_sim_coords=lambda *_args: (_ for _ in ()).throw(
            AssertionError("transient display coords must not seed detector reprojection")
        ),
        get_detector_angular_maps=_fail_projection_legacy_path(
            "detector angular maps should not be used"
        ),
        detector_pixel_to_scattering_angles=_fail_projection_legacy_path(
            "analytic forward fallback should not be used"
        ),
    )

    projected = callbacks.project_peaks_to_current_view(
        [
            {
                "display_col": 30.25,
                "display_row": -57.5,
                "sim_col": 30.25,
                "sim_row": -57.5,
                "hkl": (-1, 0, 5),
                "q_group_key": ("q_group", "primary", 1, 5),
                "source_table_index": 9,
                "source_row_index": 0,
                "source_branch_index": 1,
                "intensity": 3.0,
            }
        ]
    )

    assert projected == []


def test_project_peaks_to_current_view_derives_detector_truth_from_native_not_caked_display(
    monkeypatch,
) -> None:
    bundle = mg.CakeTransformBundle(
        detector_shape=(8, 8),
        radial_deg=np.linspace(10.0, 17.0, 8, dtype=float),
        raw_azimuth_deg=np.linspace(-4.0, 3.0, 8, dtype=float),
        gui_azimuth_deg=np.linspace(-4.0, 3.0, 8, dtype=float),
        lut=object(),
    )

    monkeypatch.setattr(
        mg,
        "_detector_pixel_to_caked_bin",
        lambda live_bundle, col, row: (
            (13.0, 2.0)
            if live_bundle is bundle and (float(col), float(row)) == (6.0, 7.0)
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
        display_to_native_sim_coords=lambda *_args: (_ for _ in ()).throw(
            AssertionError("native detector coords should drive reprojection")
        ),
        get_detector_angular_maps=_fail_projection_legacy_path(
            "detector angular maps should not be used"
        ),
        detector_pixel_to_scattering_angles=_fail_projection_legacy_path(
            "analytic forward fallback should not be used"
        ),
        rotate_point_for_display=lambda col, row, _shape, _k: (float(col), float(row)),
        display_rotate_k=0,
    )

    projected = callbacks.project_peaks_to_current_view(
        [
            {
                "display_col": 13.0,
                "display_row": 2.0,
                "native_col": 6.0,
                "native_row": 7.0,
                "caked_x": 13.0,
                "caked_y": 2.0,
                "hkl": (-1, 0, 5),
                "q_group_key": ("q_group", "primary", 1, 5),
                "source_table_index": 9,
                "source_row_index": 0,
                "source_branch_index": 1,
                "intensity": 3.0,
            }
        ]
    )

    assert len(projected) == 1
    entry = projected[0]
    assert entry["native_col"] == 6.0
    assert entry["native_row"] == 7.0
    assert entry["sim_native_x"] == 6.0
    assert entry["sim_native_y"] == 7.0
    assert entry["sim_col"] == 6.0
    assert entry["sim_row"] == 7.0
    assert entry["sim_col_raw"] == 6.0
    assert entry["sim_row_raw"] == 7.0
    assert entry["display_col"] == 13.0
    assert entry["display_row"] == 2.0
    assert entry["caked_x"] == 13.0
    assert entry["caked_y"] == 2.0
    assert entry["raw_caked_x"] == 13.0
    assert entry["raw_caked_y"] == 2.0


def test_build_geometry_manual_pick_cache_preserves_mirrored_branch_lookup_entries() -> None:
    mirrored_rows = [
        {
            "label": "left",
            "hkl": (-1, 0, 5),
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "sim_col": 181.0,
            "sim_row": 95.0,
            "weight": 1.0,
        },
        {
            "label": "right",
            "hkl": (-1, 0, 5),
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "sim_col": 190.0,
            "sim_row": 96.0,
            "weight": 1.0,
        },
    ]

    cache_data, _, next_state = mg.build_geometry_manual_pick_cache(
        param_set={"a": 2.0},
        prefer_cache=False,
        background_index=0,
        current_background_index=0,
        background_image=np.zeros((4, 4), dtype=float),
        existing_cache_signature=None,
        existing_cache_data=None,
        cache_signature_fn=lambda **_kwargs: ("sig",),
        source_rows_for_background=lambda *_args, **_kwargs: [],
        simulated_peaks_for_params=lambda _params, *, prefer_cache: [
            dict(entry) for entry in mirrored_rows
        ],
        build_grouped_candidates=_group_candidates,
        build_simulated_lookup=_build_lookup,
        current_match_config=lambda: {"search_radius_px": 18.0},
        peak_records=[],
    )

    left_key = _source_key(mirrored_rows[0])
    right_key = _source_key(mirrored_rows[1])

    assert left_key != right_key
    assert set(cache_data["simulated_lookup"]) == {left_key, right_key}
    assert cache_data["simulated_lookup"][left_key]["sim_col"] == 181.0
    assert cache_data["simulated_lookup"][right_key]["sim_col"] == 190.0
    assert set(next_state["simulated_lookup"]) == {left_key, right_key}


def test_build_geometry_manual_pick_cache_preserves_colliding_legacy_branch_lookup_entries() -> (
    None
):
    legacy_rows = [
        {
            "label": "other-right",
            "hkl": (-2, 0, 5),
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 9,
            "source_row_index": 2,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "sim_col": 181.0,
            "sim_row": 95.0,
            "weight": 1.0,
        },
        {
            "label": "target-right",
            "hkl": (-1, 0, 5),
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 9,
            "source_row_index": 3,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "sim_col": 190.0,
            "sim_row": 96.0,
            "weight": 1.0,
        },
    ]

    cache_data, _, next_state = mg.build_geometry_manual_pick_cache(
        param_set={"a": 2.0},
        prefer_cache=False,
        background_index=0,
        current_background_index=0,
        background_image=np.zeros((4, 4), dtype=float),
        existing_cache_signature=None,
        existing_cache_data=None,
        cache_signature_fn=lambda **_kwargs: ("sig",),
        source_rows_for_background=lambda *_args, **_kwargs: [],
        simulated_peaks_for_params=lambda _params, *, prefer_cache: [
            dict(entry) for entry in legacy_rows
        ],
        build_grouped_candidates=_group_candidates,
        build_simulated_lookup=_build_lookup,
        current_match_config=lambda: {"search_radius_px": 18.0},
        peak_records=[],
    )

    legacy_key = _source_key(legacy_rows[0])

    assert legacy_key == _source_key(legacy_rows[1])
    assert isinstance(cache_data["simulated_lookup"][legacy_key], list)
    assert len(cache_data["simulated_lookup"][legacy_key]) == 2
    assert isinstance(next_state["simulated_lookup"][legacy_key], list)
    resolved = mg.geometry_manual_lookup_source_entry(
        cache_data["simulated_lookup"],
        {
            "label": "target-right",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 3,
            "source_branch_index": 1,
            "source_peak_index": 1,
        },
    )

    assert resolved is not None
    assert resolved["label"] == "target-right"
    assert resolved["source_row_index"] == 3


def test_build_geometry_manual_pick_cache_resolves_legacy_peak_only_branch_entry() -> None:
    mirrored_rows = [
        {
            "label": "left",
            "hkl": (-1, 0, 5),
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": 0,
            "source_peak_index": 0,
            "sim_col": 181.0,
            "sim_row": 95.0,
            "weight": 1.0,
        },
        {
            "label": "right",
            "hkl": (-1, 0, 5),
            "q_group_key": ("q_group", "primary", 1, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "source_reflection_index": 203,
            "source_reflection_namespace": "full_reflection",
            "source_reflection_is_full": True,
            "source_branch_index": 1,
            "source_peak_index": 1,
            "sim_col": 190.0,
            "sim_row": 96.0,
            "weight": 1.0,
        },
    ]

    cache_data, _, _ = mg.build_geometry_manual_pick_cache(
        param_set={"a": 2.0},
        prefer_cache=False,
        background_index=0,
        current_background_index=0,
        background_image=np.zeros((4, 4), dtype=float),
        existing_cache_signature=None,
        existing_cache_data=None,
        cache_signature_fn=lambda **_kwargs: ("sig",),
        source_rows_for_background=lambda *_args, **_kwargs: [],
        simulated_peaks_for_params=lambda _params, *, prefer_cache: [
            dict(entry) for entry in mirrored_rows
        ],
        build_grouped_candidates=_group_candidates,
        build_simulated_lookup=_build_lookup,
        current_match_config=lambda: {"search_radius_px": 18.0},
        peak_records=[],
    )

    resolved = mg.geometry_manual_lookup_source_entry(
        cache_data["simulated_lookup"],
        {
            "label": "right",
            "hkl": (-1, 0, 5),
            "source_table_index": 9,
            "source_row_index": 0,
            "source_peak_index": 1,
        },
    )

    assert resolved is not None
    assert resolved["source_branch_index"] == 1
    assert resolved["source_reflection_index"] == 203


def test_geometry_manual_live_peak_candidates_normalize_branch_and_full_provenance() -> None:
    candidates = mg.geometry_manual_live_peak_candidates_from_records(
        [
            {
                "display_col": 21.0,
                "display_row": 34.0,
                "native_col": 21.0,
                "native_row": 34.0,
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
                "native_col": 21.0,
                "native_row": 34.0,
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


def test_mixed_background_manual_projection_groups_by_background() -> None:
    projection_calls: list[tuple[int, list[str]]] = []
    cached_entries = [
        {
            "label": "bg1",
            "background_index": 1,
            "q_group_key": ("q_group", "primary", 1, 0),
            "source_table_index": 1,
            "source_row_index": 2,
            "sim_col_raw": 3.0,
            "sim_row_raw": 4.0,
            "weight": 1.0,
        },
        {
            "label": "bg0",
            "background_index": 0,
            "q_group_key": ("q_group", "primary", 0, 0),
            "source_table_index": 5,
            "source_row_index": 6,
            "sim_col_raw": 7.0,
            "sim_row_raw": 8.0,
            "weight": 1.0,
        },
    ]

    cache_data, _, next_state = mg.build_geometry_manual_pick_cache(
        param_set={"gamma": 1.5},
        prefer_cache=True,
        background_index=0,
        current_background_index=0,
        background_image=np.ones((3, 3), dtype=float),
        existing_cache_signature=(
            ("sim", 7),
            0,
            False,
            ("old-bg",),
            2,
            (
                "('q_group', 'primary', 1, 0)",
                "('q_group', 'primary', 0, 0)",
            ),
        ),
        existing_cache_data={
            "signature": (
                ("sim", 7),
                0,
                False,
                ("old-bg",),
                2,
                (
                    "('q_group', 'primary', 1, 0)",
                    "('q_group', 'primary', 0, 0)",
                ),
            ),
            "simulated_peaks": [dict(entry) for entry in cached_entries],
            "simulated_lookup": _build_lookup(cached_entries),
            "grouped_candidates": _group_candidates(cached_entries),
        },
        cache_signature_fn=lambda **_kwargs: (
            ("sim", 7),
            0,
            False,
            ("new-bg",),
            2,
            (
                "('q_group', 'primary', 1, 0)",
                "('q_group', 'primary', 0, 0)",
            ),
        ),
        simulated_peaks_for_params=lambda _params, *, prefer_cache: [],
        build_grouped_candidates=_group_candidates,
        build_simulated_lookup=_build_lookup,
        project_peaks_for_background_view=lambda background_index, rows: (
            projection_calls.append(
                (
                    int(background_index),
                    [
                        str(dict(entry).get("label"))
                        for entry in rows or ()
                        if isinstance(entry, dict)
                    ],
                )
            )
            or [
                dict(
                    entry,
                    sim_col=float(background_index),
                    sim_row=float(background_index) + 0.5,
                )
                for entry in rows or ()
                if isinstance(entry, dict)
            ]
        ),
        current_match_config=lambda: {"search_radius_px": 24.0},
    )

    assert projection_calls == [(1, ["bg1"]), (0, ["bg0"])]
    assert [str(entry.get("label")) for entry in cache_data["simulated_peaks"]] == ["bg1", "bg0"]
    assert cache_data["simulated_peaks"][0]["sim_col"] == 1.0
    assert cache_data["simulated_peaks"][1]["sim_col"] == 0.0
    assert next_state["simulated_lookup"][_source_key(cached_entries[0])]["sim_col"] == 1.0


def test_manual_projection_defaults_missing_background_to_current_view() -> None:
    projection_calls: list[tuple[int, list[str]]] = []
    cached_entries = [
        {
            "label": "legacy",
            "q_group_key": ("q_group", "primary", 0, 0),
            "source_table_index": 1,
            "source_row_index": 2,
            "sim_col_raw": 3.0,
            "sim_row_raw": 4.0,
            "weight": 1.0,
        },
    ]
    old_signature = (("sim", 7), 0, False, ("old-bg",), 1, ("('q_group', 'primary', 0, 0)",))
    new_signature = (("sim", 7), 0, False, ("new-bg",), 1, ("('q_group', 'primary', 0, 0)",))

    cache_data, _, next_state = mg.build_geometry_manual_pick_cache(
        param_set={"gamma": 1.5},
        prefer_cache=True,
        background_index=0,
        current_background_index=0,
        background_image=np.ones((3, 3), dtype=float),
        existing_cache_signature=old_signature,
        existing_cache_data={
            "signature": old_signature,
            "simulated_peaks": [dict(entry) for entry in cached_entries],
            "simulated_lookup": _build_lookup(cached_entries),
            "grouped_candidates": _group_candidates(cached_entries),
        },
        cache_signature_fn=lambda **_kwargs: new_signature,
        simulated_peaks_for_params=lambda _params, *, prefer_cache: [],
        build_grouped_candidates=_group_candidates,
        build_simulated_lookup=_build_lookup,
        project_peaks_for_background_view=lambda background_index, rows: (
            projection_calls.append(
                (
                    int(background_index),
                    [
                        str(dict(entry).get("label"))
                        for entry in rows or ()
                        if isinstance(entry, dict)
                    ],
                )
            )
            or [
                dict(
                    entry,
                    sim_col=float(background_index),
                    sim_row=float(background_index) + 0.5,
                )
                for entry in rows or ()
                if isinstance(entry, dict)
            ]
        ),
        current_match_config=lambda: {"search_radius_px": 24.0},
    )

    assert projection_calls == [(0, ["legacy"])]
    assert [str(entry.get("label")) for entry in cache_data["simulated_peaks"]] == ["legacy"]
    assert cache_data["simulated_peaks"][0]["background_index"] == 0
    assert cache_data["simulated_peaks"][0]["sim_col"] == 0.0
    assert next_state["grouped_candidates"]
    assert next_state["simulated_lookup"][_source_key(cached_entries[0])]["sim_col"] == 0.0


def test_mixed_background_manual_projection_drops_missing_payload_groups() -> None:
    projection_calls: list[tuple[int, list[str]]] = []
    cached_entries = [
        {
            "label": "bg1",
            "background_index": 1,
            "q_group_key": ("q_group", "primary", 1, 0),
            "source_table_index": 1,
            "source_row_index": 2,
            "sim_col_raw": 3.0,
            "sim_row_raw": 4.0,
            "weight": 1.0,
        },
        {
            "label": "bg0",
            "background_index": 0,
            "q_group_key": ("q_group", "primary", 0, 0),
            "source_table_index": 5,
            "source_row_index": 6,
            "sim_col_raw": 7.0,
            "sim_row_raw": 8.0,
            "weight": 1.0,
        },
    ]

    cache_data, _, next_state = mg.build_geometry_manual_pick_cache(
        param_set={"gamma": 1.5},
        prefer_cache=True,
        background_index=0,
        current_background_index=0,
        background_image=np.ones((3, 3), dtype=float),
        existing_cache_signature=(
            ("sim", 7),
            0,
            False,
            ("old-bg",),
            2,
            (
                "('q_group', 'primary', 1, 0)",
                "('q_group', 'primary', 0, 0)",
            ),
        ),
        existing_cache_data={
            "signature": (
                ("sim", 7),
                0,
                False,
                ("old-bg",),
                2,
                (
                    "('q_group', 'primary', 1, 0)",
                    "('q_group', 'primary', 0, 0)",
                ),
            ),
            "simulated_peaks": [dict(entry) for entry in cached_entries],
            "simulated_lookup": _build_lookup(cached_entries),
            "grouped_candidates": _group_candidates(cached_entries),
        },
        cache_signature_fn=lambda **_kwargs: (
            ("sim", 7),
            0,
            False,
            ("new-bg",),
            2,
            (
                "('q_group', 'primary', 1, 0)",
                "('q_group', 'primary', 0, 0)",
            ),
        ),
        simulated_peaks_for_params=lambda _params, *, prefer_cache: [],
        build_grouped_candidates=_group_candidates,
        build_simulated_lookup=_build_lookup,
        project_peaks_for_background_view=lambda background_index, rows: (
            projection_calls.append(
                (
                    int(background_index),
                    [
                        str(dict(entry).get("label"))
                        for entry in rows or ()
                        if isinstance(entry, dict)
                    ],
                )
            )
            or (
                []
                if int(background_index) == 1
                else [
                    dict(
                        entry,
                        sim_col=float(background_index),
                        sim_row=float(background_index) + 0.5,
                    )
                    for entry in rows or ()
                    if isinstance(entry, dict)
                ]
            )
        ),
        current_match_config=lambda: {"search_radius_px": 24.0},
    )

    assert projection_calls == [(1, ["bg1"]), (0, ["bg0"])]
    assert [str(entry.get("label")) for entry in cache_data["simulated_peaks"]] == ["bg0"]
    assert _source_key(cached_entries[0]) not in next_state["simulated_lookup"]
    assert next_state["simulated_lookup"][_source_key(cached_entries[1])]["sim_col"] == 0.0


def test_geometry_manual_live_peak_candidates_restore_trust_on_revision_match() -> None:
    candidates = mg.geometry_manual_live_peak_candidates_from_records(
        [
            {
                "display_col": 21.0,
                "display_row": 34.0,
                "native_col": 21.0,
                "native_row": 34.0,
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
                "native_col": 21.0,
                "native_row": 34.0,
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


def test_geometry_manual_canonicalize_live_source_entry_only_repairs_trust_from_explicit_proof() -> (
    None
):
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


def test_refresh_geometry_manual_pair_entry_keeps_detector_truth_and_refreshes_caked_fields() -> (
    None
):
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
    assert refreshed["detector_x"] == 30.0
    assert refreshed["detector_y"] == 40.0
    assert refreshed["x"] == 30.0
    assert refreshed["y"] == 40.0
    assert refreshed["background_two_theta_deg"] == 40.0
    assert refreshed["background_phi_deg"] == 60.0
    assert refreshed["caked_x"] == 40.0
    assert refreshed["caked_y"] == 60.0
    assert refreshed["raw_caked_x"] == 40.0
    assert refreshed["raw_caked_y"] == 60.0
    assert "stale_caked_fields" not in refreshed

    display_point = (
        float(refreshed["caked_x"]) - 10.0,
        float(refreshed["caked_y"]) - 20.0,
    )
    assert display_point == (30.0, 40.0)


def test_refresh_geometry_manual_pair_entry_uses_canonical_background_angles_as_truth() -> None:
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
    assert refreshed["detector_x"] == 140.0
    assert refreshed["detector_y"] == 140.0
    assert refreshed["background_two_theta_deg"] == 150.0
    assert refreshed["background_phi_deg"] == 160.0
    assert refreshed["caked_x"] == 150.0
    assert refreshed["caked_y"] == 160.0
    assert refreshed["x"] == 140.0
    assert refreshed["y"] == 140.0


def test_refresh_geometry_manual_pair_entry_uses_inverse_lut_for_caked_only_entries(
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
            "background_two_theta_deg": 150.0,
            "background_phi_deg": 160.0,
            "caked_x": 40.0,
            "caked_y": 50.0,
            "raw_caked_x": 41.0,
            "raw_caked_y": 51.0,
        },
        background_display_shape=(200, 200),
        background_display_to_native_detector_coords=lambda col, row: (
            (None, None)
            if abs(float(col) - 30.0) <= 1.0e-9 and abs(float(row) - 40.0) <= 1.0e-9
            else (float(col), float(row))
        ),
        caked_angles_to_background_display_coords=_caked_angles_to_background_display,
        native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col) + 10.0,
            float(row) + 19.0,
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
    assert refreshed["raw_caked_x"] == 150.0
    assert refreshed["raw_caked_y"] == 160.0
    assert inverse_calls == [
        (150.0, 160.0, bundle),
        (150.0, 160.0, bundle),
    ]


def test_refresh_geometry_manual_pair_entry_marks_stale_caked_fields_when_roundtrip_breaks() -> (
    None
):
    refreshed = mg.refresh_geometry_manual_pair_entry(
        {
            "label": "0,0,3",
            "hkl": (0, 0, 3),
            "x": 30.0,
            "y": 40.0,
            "background_two_theta_deg": 150.0,
            "background_phi_deg": 160.0,
        },
        background_display_shape=(200, 200),
        background_display_to_native_detector_coords=lambda col, row: (
            (None, None)
            if abs(float(col) - 30.0) <= 1.0e-9 and abs(float(row) - 40.0) <= 1.0e-9
            else (float(col), float(row))
        ),
        caked_angles_to_background_display_coords=lambda two_theta, phi: (
            float(two_theta),
            float(phi),
        ),
        native_detector_coords_to_caked_display_coords=lambda col, row: (
            float(col) + 100.0,
            float(row) + 100.0,
        ),
        rotate_point_for_display=lambda col, row, _shape, _k: (float(col), float(row)),
        display_rotate_k=0,
        stale_caked_tolerance_px=0.5,
    )

    assert refreshed is not None
    assert refreshed["stale_caked_fields"] is True
    assert "detector_x" not in refreshed
    assert "detector_y" not in refreshed
    assert "x" not in refreshed
    assert "y" not in refreshed


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
