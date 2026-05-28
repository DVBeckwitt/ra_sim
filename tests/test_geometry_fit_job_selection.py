from __future__ import annotations

from ra_sim.gui._runtime.geometry_fit_job import resolve_geometry_fit_selection


def test_resolve_geometry_fit_selection_keeps_detector_current_background_primary() -> None:
    snapshot = resolve_geometry_fit_selection(
        params={"theta_initial": 2.0},
        var_names=[],
        current_background_index=0,
        theta_initial_value=1.0,
        total_background_count=2,
        geometry_manual_pairs_for_index=lambda _idx: [{"pair": 1}],
        apply_geometry_fit_background_selection=lambda **_kwargs: True,
        current_geometry_fit_background_indices=lambda **_kwargs: [0],
        geometry_fit_uses_shared_theta_offset=lambda _indices: False,
        apply_background_theta_metadata=lambda **_kwargs: True,
        current_background_theta_values=lambda **_kwargs: [],
        current_geometry_theta_offset=lambda **_kwargs: 0.0,
        effective_geometry_fit_background_indices=lambda requested, **_kwargs: list(requested),
    )

    assert snapshot.preserve_live_theta is True
    assert snapshot.selected_background_indices == [0]
    assert snapshot.required_indices == [0]
    assert snapshot.primary_index == 0
    assert snapshot.fit_params_snapshot["theta_initial"] == 2.0
    assert snapshot.fit_params_snapshot["theta_offset"] == 0.0
    assert snapshot.background_theta_values == [2.0]


def test_resolve_geometry_fit_selection_applies_shared_theta_metadata() -> None:
    snapshot = resolve_geometry_fit_selection(
        params={"theta_initial": 2.0},
        var_names=["gamma"],
        current_background_index=1,
        theta_initial_value=1.0,
        total_background_count=2,
        geometry_manual_pairs_for_index=lambda _idx: [{"pair": 1}],
        apply_geometry_fit_background_selection=lambda **_kwargs: True,
        current_geometry_fit_background_indices=lambda **_kwargs: [0, 1],
        geometry_fit_uses_shared_theta_offset=lambda _indices: True,
        apply_background_theta_metadata=lambda **_kwargs: True,
        current_background_theta_values=lambda **_kwargs: [10.0, 20.0],
        current_geometry_theta_offset=lambda **_kwargs: 3.0,
        effective_geometry_fit_background_indices=lambda requested, **_kwargs: list(requested),
    )

    assert snapshot.uses_shared_theta is True
    assert snapshot.joint_background_mode is True
    assert snapshot.build_all_selected_backgrounds is True
    assert snapshot.required_indices == [0, 1]
    assert snapshot.primary_index == 1
    assert snapshot.theta_offset == 3.0
    assert snapshot.fit_params_snapshot["theta_initial"] == 20.0
    assert snapshot.fit_params_snapshot["theta_offset"] == 3.0


def test_resolve_geometry_fit_selection_fails_closed_on_selection_error() -> None:
    def _raise_selection(**_kwargs):
        raise ValueError("bad selection")

    snapshot = resolve_geometry_fit_selection(
        params={"theta_initial": 2.0},
        var_names=[],
        current_background_index=0,
        theta_initial_value=1.0,
        total_background_count=2,
        geometry_manual_pairs_for_index=lambda _idx: [{"pair": 1}],
        apply_geometry_fit_background_selection=lambda **_kwargs: True,
        current_geometry_fit_background_indices=_raise_selection,
        geometry_fit_uses_shared_theta_offset=lambda _indices: True,
        apply_background_theta_metadata=lambda **_kwargs: True,
        current_background_theta_values=lambda **_kwargs: [10.0, 20.0],
        current_geometry_theta_offset=lambda **_kwargs: 3.0,
        effective_geometry_fit_background_indices=lambda requested, **_kwargs: list(requested),
    )

    assert snapshot.selection_error == "bad selection"
    assert snapshot.required_indices == []
    assert snapshot.uses_shared_theta is False
