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
    pair: dict[str, object],
    *,
    ensure_geometry_fit_caked_view,
    projector_kind: str | None = None,
    projector=None,
    var_names: list[str] | None = None,
):
    def _build_dataset(background_index, **_kwargs):
        return {
            "dataset_index": int(background_index),
            "label": "bg0.osc",
            "pair_count": 1,
            "group_count": 1,
            "resolved_source_pair_count": 1,
            "summary_line": "bg[0]",
            "measured_for_fit": [dict(pair)],
            "manual_point_pairs": [dict(pair)],
            "spec": {
                "dataset_index": int(background_index),
                "label": "bg0.osc",
                "measured_peaks": [dict(pair)],
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
        geometry_manual_pairs_for_index=lambda _idx: [dict(pair)],
        ensure_geometry_fit_caked_view=ensure_geometry_fit_caked_view,
        build_dataset=_build_dataset,
        build_runtime_config=lambda _params: {"solver": {"dynamic_point_geometry_fit": True}},
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
    assert "needs observed caked coordinates" in error
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
