"""Pure geometry-fit async job snapshot helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class GeometryFitSelectionSnapshot:
    params: dict[str, object]
    var_names: list[str]
    preserve_live_theta: bool
    selected_background_indices: list[int]
    required_indices: list[int]
    skipped_empty_indices: set[int]
    selection_applied: bool
    selection_error: str | None
    uses_shared_theta: bool
    theta_metadata_applied: bool
    background_theta_values: list[float]
    background_theta_error: str | None
    theta_offset: float
    fit_params_snapshot: dict[str, object]
    primary_index: int
    joint_background_mode: bool
    build_all_selected_backgrounds: bool


def resolve_geometry_fit_selection(
    *,
    params: Mapping[str, object] | None,
    var_names: Sequence[object] | None,
    current_background_index: int,
    theta_initial_value: float,
    total_background_count: int,
    geometry_manual_pairs_for_index: Callable[[int], Sequence[Mapping[str, object]]],
    apply_geometry_fit_background_selection: Callable[..., bool],
    current_geometry_fit_background_indices: Callable[..., Sequence[int]],
    geometry_fit_uses_shared_theta_offset: Callable[..., bool],
    apply_background_theta_metadata: Callable[..., bool],
    current_background_theta_values: Callable[..., Sequence[float]],
    current_geometry_theta_offset: Callable[..., float],
    effective_geometry_fit_background_indices: Callable[..., list[int]],
) -> GeometryFitSelectionSnapshot:
    params_snapshot = dict(params or {})
    var_names_snapshot = [str(name) for name in (var_names or ())]
    preserve_live_theta = (
        "theta_initial" not in var_names_snapshot and "theta_offset" not in var_names_snapshot
    )

    selection_applied = bool(
        apply_geometry_fit_background_selection(
            trigger_update=False,
            sync_live_theta=not preserve_live_theta,
        )
    )
    selected_background_indices: list[int] = []
    requested_indices: list[int] = []
    skipped_empty_indices: set[int] = set()
    selection_error: str | None = None
    if selection_applied:
        try:
            requested_indices = [
                int(idx) for idx in current_geometry_fit_background_indices(strict=True)
            ]
            selected_background_indices = effective_geometry_fit_background_indices(
                requested_indices,
                total_count=int(total_background_count),
                geometry_manual_pairs_for_index=geometry_manual_pairs_for_index,
            )
            active_set = {int(idx) for idx in selected_background_indices}
            skipped_empty_indices = {int(idx) for idx in requested_indices if idx not in active_set}
        except Exception as exc:
            selection_error = str(exc)

    uses_shared_theta = False
    if selection_error is None:
        try:
            uses_shared_theta = bool(
                geometry_fit_uses_shared_theta_offset(list(selected_background_indices))
            )
        except Exception:
            uses_shared_theta = False

    theta_metadata_applied = True
    background_theta_values: list[float] = []
    background_theta_error: str | None = None
    theta_offset_value = 0.0
    if uses_shared_theta:
        theta_metadata_applied = bool(
            apply_background_theta_metadata(
                trigger_update=False,
                sync_live_theta=not preserve_live_theta,
            )
        )
        if theta_metadata_applied:
            try:
                background_theta_values = [
                    float(value) for value in current_background_theta_values(strict_count=True)
                ]
                theta_offset_value = float(current_geometry_theta_offset(strict=True))
            except Exception as exc:
                background_theta_error = str(exc)

    fit_params_snapshot = dict(params_snapshot)
    active_in_selection = int(current_background_index) in set(selected_background_indices)
    primary_index = (
        int(current_background_index)
        if active_in_selection
        else (
            int(selected_background_indices[0])
            if selected_background_indices
            else int(current_background_index)
        )
    )
    joint_background_mode = bool(uses_shared_theta and len(selected_background_indices) > 1)
    build_all_selected_backgrounds = bool(joint_background_mode)
    if uses_shared_theta:
        fit_params_snapshot["theta_offset"] = float(theta_offset_value)
        if 0 <= int(primary_index) < len(background_theta_values):
            fit_params_snapshot["theta_initial"] = float(background_theta_values[int(primary_index)])
    else:
        fit_params_snapshot["theta_offset"] = 0.0
        theta_default = float(fit_params_snapshot.get("theta_initial", theta_initial_value))
        fit_params_snapshot["theta_initial"] = float(theta_default)
        background_theta_values = [float(theta_default)]

    if not selection_applied or selection_error is not None or not active_in_selection:
        required_indices: list[int] = []
    elif build_all_selected_backgrounds:
        required_indices = [int(idx) for idx in selected_background_indices]
    else:
        required_indices = [int(primary_index)]

    return GeometryFitSelectionSnapshot(
        params=params_snapshot,
        var_names=var_names_snapshot,
        preserve_live_theta=bool(preserve_live_theta),
        selected_background_indices=[int(idx) for idx in selected_background_indices],
        required_indices=required_indices,
        skipped_empty_indices=skipped_empty_indices,
        selection_applied=bool(selection_applied),
        selection_error=selection_error,
        uses_shared_theta=bool(uses_shared_theta),
        theta_metadata_applied=bool(theta_metadata_applied),
        background_theta_values=[float(value) for value in background_theta_values],
        background_theta_error=background_theta_error,
        theta_offset=float(theta_offset_value),
        fit_params_snapshot=fit_params_snapshot,
        primary_index=int(primary_index),
        joint_background_mode=bool(joint_background_mode),
        build_all_selected_backgrounds=bool(build_all_selected_backgrounds),
    )
