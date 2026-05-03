from __future__ import annotations

from ra_sim.gui import geometry_fit


def _detector_origin_pair() -> dict[str, object]:
    return {
        "pair_id": "detector-origin",
        "manual_background_input_origin": "detector",
        "background_detector_frame_provenance": "detector_to_caked_refresh",
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
        var_names=["gamma"],
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
