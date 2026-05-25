from __future__ import annotations

from ra_sim.gui import geometry_fit


def _detector_origin_pair() -> dict[str, object]:
    return {
        "pair_id": "detector-origin",
        "manual_background_input_origin": "detector",
        "background_detector_frame_provenance": "detector_to_caked_refresh",
        "q_group_key": ("q_group", "primary", 1, 10),
        "hkl": (-1, 0, 10),
        "source_row_index": 42,
        "detector_display_x": 300.0,
        "detector_display_y": 400.0,
        "detector_native_x": 300.0,
        "detector_native_y": 400.0,
        "background_two_theta_deg": 21.0,
        "background_phi_deg": 22.0,
        "caked_x": 21.0,
        "caked_y": 22.0,
    }


def _detector_origin_pair_without_caked_anchor() -> dict[str, object]:
    pair = dict(_detector_origin_pair())
    for key in (
        "background_two_theta_deg",
        "background_phi_deg",
        "caked_x",
        "caked_y",
    ):
        pair.pop(key, None)
    return pair


def _locked_detector_origin_pair_without_caked_anchor(branch: int) -> dict[str, object]:
    pair = _detector_origin_pair_without_caked_anchor()
    branch_i = int(branch)
    pair.update(
        {
            "pair_id": f"locked-detector-origin-{branch_i}",
            "source_table_index": 99,
            "source_reflection_index": 910 + branch_i,
            "source_reflection_namespace": "subset",
            "source_reflection_is_full": False,
            "source_row_index": 42 + branch_i,
            "source_branch_index": branch_i,
            "source_peak_index": branch_i,
            "background_detector_x": 300.0 + branch_i,
            "background_detector_y": 400.0 + branch_i,
            "background_detector_input_frame": "native_detector",
            "optimizer_request_source": "provider_pair",
            "optimizer_request_has_fixed_source": True,
            "fit_source_resolution_kind": "provider_fixed_source_local",
        }
    )
    return pair


def _locked_detector_origin_pair(branch: int) -> dict[str, object]:
    pair = _locked_detector_origin_pair_without_caked_anchor(branch)
    branch_i = int(branch)
    pair.update(
        {
            "background_two_theta_deg": 30.0 + branch_i,
            "background_phi_deg": 40.0 + branch_i,
            "caked_x": 30.0 + branch_i,
            "caked_y": 40.0 + branch_i,
        }
    )
    return pair


def _caked_origin_pair() -> dict[str, object]:
    return {
        "pair_id": "caked-origin",
        "manual_background_input_origin": "caked",
        "q_group_key": ("q_group", "primary", 1, 10),
        "hkl": (-1, 0, 10),
        "source_row_index": 42,
        "background_two_theta_deg": 21.0,
        "background_phi_deg": 22.0,
        "caked_x": 21.0,
        "caked_y": 22.0,
    }


def _prepare_with_pair(
    pair: dict[str, object] | list[dict[str, object]],
    *,
    ensure_geometry_fit_caked_view,
    projector_kind: str | None = None,
    projector=None,
    var_names: list[str] | None = None,
    manual_fit_requires_caked_space: bool = False,
):
    pairs = [dict(item) for item in pair] if isinstance(pair, list) else [dict(pair)]

    def _build_dataset(background_index, **_kwargs):
        return {
            "dataset_index": int(background_index),
            "label": "bg0.osc",
            "pair_count": len(pairs),
            "group_count": 1,
            "resolved_source_pair_count": len(pairs),
            "summary_line": "bg[0]",
            "measured_for_fit": [dict(item) for item in pairs],
            "manual_point_pairs": [dict(item) for item in pairs],
            "spec": {
                "dataset_index": int(background_index),
                "label": "bg0.osc",
                "measured_peaks": [dict(item) for item in pairs],
                "fit_space_projector": projector,
                "fit_space_projector_kind": projector_kind,
                "fit_space_projector_unavailable_reason": "missing_exact_caked_bundle",
            },
            "initial_pairs_display": [],
            "native_background": None,
        }

    return geometry_fit.prepare_geometry_fit_run(
        params={"theta_initial": 0.0},
        var_names=list(var_names or ["gamma"]),
        fit_config={},
        osc_files=["bg0.osc"],
        current_background_index=0,
        theta_initial=0.0,
        preserve_live_theta=False,
        apply_geometry_fit_background_selection=lambda **_kwargs: True,
        current_geometry_fit_background_indices=lambda **_kwargs: [0],
        geometry_fit_uses_shared_theta_offset=lambda *_args, **_kwargs: False,
        apply_background_theta_metadata=lambda **_kwargs: True,
        current_background_theta_values=lambda **_kwargs: [0.0],
        current_geometry_theta_offset=lambda **_kwargs: 0.0,
        geometry_manual_pairs_for_index=lambda _idx: [dict(item) for item in pairs],
        ensure_geometry_fit_caked_view=ensure_geometry_fit_caked_view,
        build_dataset=_build_dataset,
        build_runtime_config=lambda _params: {"solver": {"dynamic_point_geometry_fit": True}},
        manual_fit_requires_caked_space=bool(manual_fit_requires_caked_space),
    )


def test_detector_origin_pair_with_backfilled_caked_fields_uses_detector_fit_space() -> None:
    assert (
        geometry_fit.geometry_manual_pairs_fit_space_kind([_detector_origin_pair()]) == "detector"
    )


def test_empty_manual_pair_set_is_missing_fit_space() -> None:
    assert geometry_fit.geometry_manual_pairs_fit_space_kind([]) == "missing"


def test_missing_pairs_are_reported_before_mixed_fit_spaces() -> None:
    fit_spaces = geometry_fit.geometry_manual_fit_space_by_background(
        [0, 1, 2],
        {
            0: [_caked_origin_pair()],
            1: [],
            2: [],
        },
        pick_uses_caked_space=False,
        current_background_index=0,
    )

    error = geometry_fit.manual_geometry_fit_space_preflight_error(
        fit_spaces,
        osc_files=[
            "Bi2Se3_5m_5d.osc",
            "Bi2Se3_10d_5m.osc",
            "Bi2Se3_15d_5m.osc",
        ],
    )

    assert fit_spaces == {0: "caked", 1: "missing", 2: "missing"}
    assert error is not None
    assert "save manual Qr/Qz pairs first" in error
    assert "Bi2Se3_10d_5m.osc" in error
    assert "Bi2Se3_15d_5m.osc" in error
    assert "mix detector-pixel and caked fit-space" not in error


def test_detector_origin_pair_with_backfilled_caked_fields_stays_detector_for_two_tilts() -> None:
    spaces = geometry_fit.geometry_manual_fit_space_by_background(
        [0],
        {0: [_detector_origin_pair()]},
        pick_uses_caked_space=False,
        current_background_index=0,
        active_var_names=["gamma", "Gamma"],
    )

    assert spaces == {0: "detector"}


def test_caked_requirement_is_separate_from_detector_origin_provenance() -> None:
    spaces = geometry_fit.geometry_manual_fit_space_by_background(
        [0],
        {0: [_detector_origin_pair()]},
        pick_uses_caked_space=False,
        current_background_index=0,
        active_var_names=["gamma", "Gamma"],
    )

    required = geometry_fit.geometry_manual_caked_fit_space_required_by_background(
        [0],
        objective_space="caked_deg",
        requested_projection_view_mode="detector",
        explicit_fit_space=None,
    )

    assert spaces == {0: "detector"}
    assert required == {0: True}


def test_caked_requirement_overrides_mixed_pick_provenance_for_preflight() -> None:
    pairs_by_background = {
        0: [_detector_origin_pair()],
        1: [_caked_origin_pair()],
    }
    build_calls: list[tuple[int, bool]] = []

    def _build_dataset(background_index, **kwargs):
        requires_caked = bool(kwargs.get("manual_fit_requires_caked_space", False))
        build_calls.append((int(background_index), requires_caked))
        pair = dict(pairs_by_background[int(background_index)][0])
        return {
            "dataset_index": int(background_index),
            "label": f"bg{int(background_index)}.osc",
            "pair_count": 1,
            "group_count": 1,
            "resolved_source_pair_count": 1,
            "summary_line": f"bg[{int(background_index)}]",
            "measured_for_fit": [pair],
            "manual_point_pairs": [pair],
            "spec": {
                "dataset_index": int(background_index),
                "label": f"bg{int(background_index)}.osc",
                "measured_peaks": [pair],
                "fit_space_projector": lambda cols, rows, **_kwargs: {
                    "two_theta_deg": cols,
                    "phi_deg": rows,
                    "valid": True,
                    "fit_space_projector_kind": "exact_caked_bundle",
                },
                "fit_space_projector_kind": "exact_caked_bundle",
            },
            "initial_pairs_display": [],
            "native_background": None,
        }

    result = geometry_fit.prepare_geometry_fit_run(
        params={"theta_initial": 0.0},
        var_names=["gamma", "Gamma"],
        fit_config={},
        osc_files=["bg0.osc", "bg1.osc"],
        current_background_index=0,
        theta_initial=0.0,
        preserve_live_theta=False,
        apply_geometry_fit_background_selection=lambda **_kwargs: True,
        current_geometry_fit_background_indices=lambda **_kwargs: [0, 1],
        geometry_fit_uses_shared_theta_offset=lambda *_args, **_kwargs: False,
        apply_background_theta_metadata=lambda **_kwargs: True,
        current_background_theta_values=lambda **_kwargs: [0.0, 1.0],
        current_geometry_theta_offset=lambda **_kwargs: 0.0,
        geometry_manual_pairs_for_index=lambda idx: [dict(pairs_by_background[int(idx)][0])],
        ensure_geometry_fit_caked_view=lambda: None,
        build_dataset=_build_dataset,
        build_runtime_config=lambda _params: {"solver": {"dynamic_point_geometry_fit": True}},
        include_all_selected_backgrounds=True,
        manual_fit_requires_caked_space=True,
    )

    assert result.error_text is None
    assert result.prepared_run is not None
    assert build_calls == [(0, True), (1, True)]
    assert all(
        spec["_manual_caked_fit_space_required"] is True
        for spec in result.prepared_run.dataset_specs
    )


def test_detector_origin_pair_does_not_call_ensure_caked_view() -> None:
    result = _prepare_with_pair(
        _detector_origin_pair(),
        ensure_geometry_fit_caked_view=lambda: (_ for _ in ()).throw(
            AssertionError("detector-origin pair must not ensure caked view")
        ),
    )

    assert result.error_text is None
    assert result.prepared_run is not None
    assert result.prepared_run.geometry_runtime_cfg.get("projection_view_mode") != "caked"


def test_locked_detector_origin_qr_pairs_stay_detector_runtime() -> None:
    calls: list[str] = []
    build_calls: list[tuple[int, bool]] = []
    pairs = [
        _locked_detector_origin_pair_without_caked_anchor(0),
        _locked_detector_origin_pair_without_caked_anchor(1),
    ]

    def _projector(cols, rows, **_kwargs):
        return {
            "two_theta_deg": [float(value) * 0.1 for value in cols],
            "phi_deg": [float(value) * 0.1 for value in rows],
            "valid": True,
            "fit_space_projector_kind": "exact_caked_bundle",
        }

    def _build_dataset(background_index, **kwargs):
        requires_caked = bool(kwargs.get("manual_fit_requires_caked_space", False))
        build_calls.append((int(background_index), requires_caked))
        rows = [dict(pair) for pair in pairs]
        if requires_caked:
            for row in rows:
                projected = _projector(
                    [float(row["background_detector_x"])],
                    [float(row["background_detector_y"])],
                    input_frame="native_detector",
                    anchor_kind="measured",
                )
                row["background_two_theta_deg"] = projected["two_theta_deg"][0]
                row["background_phi_deg"] = projected["phi_deg"][0]
                row["caked_x"] = row["background_two_theta_deg"]
                row["caked_y"] = row["background_phi_deg"]
        return {
            "dataset_index": int(background_index),
            "label": "bg0.osc",
            "pair_count": len(rows),
            "group_count": 1,
            "resolved_source_pair_count": len(rows),
            "summary_line": "bg[0]",
            "measured_for_fit": [dict(row) for row in rows],
            "manual_point_pairs": [dict(row) for row in rows],
            "provider_pairs": [dict(row) for row in rows],
            "spec": {
                "dataset_index": int(background_index),
                "label": "bg0.osc",
                "measured_peaks": [dict(row) for row in rows],
                "fit_space_projector": _projector,
                "fit_space_projector_kind": "exact_caked_bundle",
            },
            "initial_pairs_display": [],
            "native_background": None,
        }

    result = geometry_fit.prepare_geometry_fit_run(
        params={"theta_initial": 0.0},
        var_names=["gamma", "Gamma"],
        fit_config={},
        osc_files=["bg0.osc"],
        current_background_index=0,
        theta_initial=0.0,
        preserve_live_theta=False,
        apply_geometry_fit_background_selection=lambda **_kwargs: True,
        current_geometry_fit_background_indices=lambda **_kwargs: [0],
        geometry_fit_uses_shared_theta_offset=lambda *_args, **_kwargs: False,
        apply_background_theta_metadata=lambda **_kwargs: True,
        current_background_theta_values=lambda **_kwargs: [0.0],
        current_geometry_theta_offset=lambda **_kwargs: 0.0,
        geometry_manual_pairs_for_index=lambda _idx: [dict(pair) for pair in pairs],
        ensure_geometry_fit_caked_view=lambda: calls.append("ensure"),
        build_dataset=_build_dataset,
        build_runtime_config=lambda _params: {"solver": {"dynamic_point_geometry_fit": False}},
    )

    assert calls == []
    assert build_calls == [(0, False)]
    assert result.error_text is None
    assert result.prepared_run is not None
    cfg = result.prepared_run.geometry_runtime_cfg
    solver = cfg["solver"]
    assert result.prepared_run.dataset_specs[0].get("_manual_caked_fit_space_required") is not True
    assert cfg.get("projection_view_mode") != "caked"
    assert solver.get("dynamic_point_geometry_fit") is not True
    assert solver["manual_point_fit_mode"] is True
    assert "background_two_theta_deg" not in result.prepared_run.current_dataset[
        "measured_for_fit"
    ][0]


def test_locked_detector_origin_qr_pair_nested_identity_stays_detector_runtime() -> None:
    pair = _detector_origin_pair()
    pair.update(
        {
            "pair_id": "locked-detector-origin-nested",
            "source_table_index": 99,
            "source_reflection_index": 910,
            "source_reflection_namespace": "subset",
            "source_reflection_is_full": False,
            "optimizer_request_source": "provider_pair",
            "optimizer_request_has_fixed_source": True,
            "fit_source_resolution_kind": "provider_fixed_source_local",
            "provider_selected_source_identity_canonical": {
                "q_group_key": ["q_group", "primary", 1, 10],
                "normalized_hkl": [-1, 0, 10],
                "source_row_index": 42,
                "source_branch_index": 1,
                "source_peak_index": 1,
            },
        }
    )
    for key in (
        "q_group_key",
        "hkl",
        "source_row_index",
        "source_branch_index",
        "source_peak_index",
    ):
        pair.pop(key, None)

    calls: list[str] = []

    result = _prepare_with_pair(
        pair,
        ensure_geometry_fit_caked_view=lambda: calls.append("ensure"),
        projector_kind="exact_caked_bundle",
        projector=lambda cols, rows, **_kwargs: {
            "two_theta_deg": cols,
            "phi_deg": rows,
            "valid": True,
        },
    )

    assert calls == []
    assert result.error_text is None
    assert result.prepared_run is not None
    cfg = result.prepared_run.geometry_runtime_cfg
    assert result.prepared_run.dataset_specs[0].get("_manual_caked_fit_space_required") is not True
    assert cfg.get("projection_view_mode") != "caked"
    assert cfg["solver"].get("dynamic_point_geometry_fit") is not True


def test_detector_origin_two_tilt_fit_uses_detector_manual_point_runtime() -> None:
    calls: list[str] = []

    def _projector(cols, rows, **_kwargs):
        return {
            "two_theta_deg": cols,
            "phi_deg": rows,
            "valid": True,
            "fit_space_projector_kind": "exact_caked_bundle",
        }

    result = _prepare_with_pair(
        _detector_origin_pair(),
        ensure_geometry_fit_caked_view=lambda: calls.append("ensure"),
        projector_kind="exact_caked_bundle",
        projector=_projector,
        var_names=["gamma", "Gamma"],
    )

    assert calls == []
    assert result.error_text is None
    assert result.prepared_run is not None
    cfg = result.prepared_run.geometry_runtime_cfg
    solver = cfg["solver"]
    seed_search = cfg["seed_search"]
    assert cfg.get("projection_view_mode") != "caked"
    assert solver["manual_point_fit_mode"] is True
    assert solver.get("dynamic_point_geometry_fit") is not True
    assert solver["seed_multistart"] is False
    assert solver["seed_multistart_enabled"] is False
    assert seed_search["enabled"] is False
    assert "_qr_fit_point_only_projection" not in solver
    assert "_headless_accept_caked_angular_metric_without_pixel_threshold" not in solver
    assert "bounds" not in cfg


def test_detector_origin_pair_requested_caked_objective_fails_without_observed_anchor() -> None:
    calls: list[str] = []

    result = _prepare_with_pair(
        _detector_origin_pair_without_caked_anchor(),
        ensure_geometry_fit_caked_view=lambda: calls.append("ensure"),
        projector_kind=None,
        projector=None,
        var_names=["gamma", "Gamma"],
        manual_fit_requires_caked_space=True,
    )

    assert calls == ["ensure"]
    assert result.prepared_run is None
    assert result.error_text is not None
    assert "manual_caked_fit_space_missing" in result.error_text
    assert "observed caked" in result.error_text


def test_detector_origin_pair_requested_caked_objective_uses_dynamic_angular_runtime() -> None:
    calls: list[str] = []

    def _projector(cols, rows, **_kwargs):
        return {
            "two_theta_deg": cols,
            "phi_deg": rows,
            "valid": True,
            "fit_space_projector_kind": "exact_caked_bundle",
        }

    result = _prepare_with_pair(
        _detector_origin_pair(),
        ensure_geometry_fit_caked_view=lambda: calls.append("ensure"),
        projector_kind="exact_caked_bundle",
        projector=_projector,
        var_names=["gamma", "Gamma"],
        manual_fit_requires_caked_space=True,
    )

    assert calls == ["ensure"]
    assert result.error_text is None
    assert result.prepared_run is not None
    cfg = result.prepared_run.geometry_runtime_cfg
    solver = cfg["solver"]
    assert cfg["projection_view_mode"] == "caked"
    assert solver["manual_point_fit_mode"] is True
    assert solver["dynamic_point_geometry_fit"] is True
    assert solver["_qr_fit_point_only_projection"] is True
    assert result.prepared_run.dataset_specs[0]["_manual_caked_fit_space_required"] is True


def test_locked_qr_two_branch_caked_runtime_enables_line_constraints() -> None:
    calls: list[str] = []

    def _projector(cols, rows, **_kwargs):
        return {
            "two_theta_deg": cols,
            "phi_deg": rows,
            "valid": True,
            "fit_space_projector_kind": "exact_caked_bundle",
        }

    result = _prepare_with_pair(
        [
            _locked_detector_origin_pair(0),
            _locked_detector_origin_pair(1),
        ],
        ensure_geometry_fit_caked_view=lambda: calls.append("ensure"),
        projector_kind="exact_caked_bundle",
        projector=_projector,
        var_names=["gamma", "Gamma"],
        manual_fit_requires_caked_space=True,
    )

    assert calls == ["ensure"]
    assert result.error_text is None
    assert result.prepared_run is not None
    solver = result.prepared_run.geometry_runtime_cfg["solver"]
    assert solver["dynamic_point_geometry_fit"] is True
    assert solver["q_group_line_constraints"] is True
    assert solver["q_group_line_constraints_enabled"] is True
    assert solver["q_group_line_requires_two_branches"] is True
    assert solver["q_group_line_angle_weight"] == 0.5


def test_caked_objective_missing_observed_anchor_fails_preflight() -> None:
    error = geometry_fit.manual_caked_geometry_fit_observed_anchor_preflight_error(
        [
            {
                "label": "bg0.osc",
                "measured_for_fit": [
                    {
                        "manual_background_input_origin": "detector",
                        "q_group_key": ("q_group", "primary", 1, 10),
                        "hkl": (-1, 0, 10),
                        "source_row_index": 42,
                    }
                ],
            }
        ]
    )

    assert error is not None
    assert "blocked before optimization" in error
    assert "could not be projected into caked fit-space" in error
    assert "bg0.osc (1 missing)" in error


def test_caked_origin_pair_requires_exact_caked_projector() -> None:
    calls: list[str] = []
    result = _prepare_with_pair(
        _caked_origin_pair(),
        ensure_geometry_fit_caked_view=lambda: calls.append("ensure"),
        projector_kind=None,
        projector=None,
    )

    assert calls == ["ensure"]
    assert result.prepared_run is None
    assert result.error_text is not None
    assert "exact caked projector unavailable for caked-origin background pair" in result.error_text


def test_caked_projector_unavailable_error_only_for_caked_pairs() -> None:
    detector_result = _prepare_with_pair(
        _detector_origin_pair(),
        ensure_geometry_fit_caked_view=lambda: (_ for _ in ()).throw(
            AssertionError("detector-origin pair must not ensure caked view")
        ),
        projector_kind=None,
        projector=None,
    )
    caked_result = _prepare_with_pair(
        _caked_origin_pair(),
        ensure_geometry_fit_caked_view=lambda: None,
        projector_kind=None,
        projector=None,
    )

    assert detector_result.error_text is None
    assert caked_result.error_text is not None
    assert "exact caked projector unavailable for caked-origin background pair" in (
        caked_result.error_text
    )
