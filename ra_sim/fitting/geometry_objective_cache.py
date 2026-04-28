"""Geometry objective cache signatures and reuse decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class GeometryObjectiveSignature:
    physics_sig: tuple
    detector_center_sig: tuple
    dataset_sig: tuple
    point_provider_sig: tuple
    qr_branch_identity_sig: tuple
    source_row_identity_sig: tuple
    manual_selection_sig: tuple
    refined_peak_sig: tuple
    objective_mode_sig: tuple
    active_fit_parameter_sig: tuple


@dataclass(frozen=True)
class GeometryObjectiveCacheDecision:
    can_reuse: bool
    mode: str
    reject_reason: str | None
    changed_fields: tuple[str, ...] = ()


_REJECT_REASONS_BY_FIELD: tuple[tuple[str, str], ...] = (
    ("physics_sig", "physics_changed"),
    ("dataset_sig", "dataset_changed"),
    ("point_provider_sig", "point_provider_changed"),
    ("qr_branch_identity_sig", "qr_branch_identity_changed"),
    ("source_row_identity_sig", "source_row_identity_changed"),
    ("manual_selection_sig", "manual_selection_changed"),
    ("refined_peak_sig", "refined_peak_changed"),
    ("objective_mode_sig", "objective_mode_changed"),
    ("active_fit_parameter_sig", "active_fit_parameter_changed"),
)


def geometry_objective_signature_changed_fields(
    previous: GeometryObjectiveSignature,
    current: GeometryObjectiveSignature,
) -> tuple[str, ...]:
    """Return signature field names that changed, preserving policy order."""

    return tuple(
        field
        for field in GeometryObjectiveSignature.__dataclass_fields__
        if getattr(previous, field) != getattr(current, field)
    )


def geometry_objective_signature_cache_key(
    signature: GeometryObjectiveSignature,
) -> tuple[tuple[str, tuple], ...]:
    """Return a stable tuple key for exact objective signature cache entries."""

    return tuple(
        (field, getattr(signature, field))
        for field in GeometryObjectiveSignature.__dataclass_fields__
    )


def geometry_objective_signature_reuse_key(
    signature: GeometryObjectiveSignature,
) -> tuple[tuple[str, tuple], ...]:
    """Return the identity key for center-only remap reuse candidates."""

    return tuple(
        (field, getattr(signature, field))
        for field in GeometryObjectiveSignature.__dataclass_fields__
        if field != "detector_center_sig"
    )


def geometry_objective_cache_decision(
    previous: GeometryObjectiveSignature | None,
    current: GeometryObjectiveSignature,
    *,
    exact_center_remap_cache_available: bool,
    cache_enabled: bool = True,
) -> GeometryObjectiveCacheDecision:
    """Decide whether objective inputs may reuse a center-remapped cache entry."""

    if not cache_enabled:
        return GeometryObjectiveCacheDecision(
            can_reuse=False,
            mode="disabled",
            reject_reason="cache_disabled",
            changed_fields=(),
        )
    if previous is None:
        return GeometryObjectiveCacheDecision(
            can_reuse=False,
            mode="full_simulation",
            reject_reason="initial_evaluation",
            changed_fields=(),
        )

    changed_fields = geometry_objective_signature_changed_fields(previous, current)
    changed_set = set(changed_fields)
    for field, reason in _REJECT_REASONS_BY_FIELD:
        if field in changed_set:
            return GeometryObjectiveCacheDecision(
                can_reuse=False,
                mode="full_simulation",
                reject_reason=reason,
                changed_fields=changed_fields,
            )

    if previous.detector_center_sig == current.detector_center_sig:
        return GeometryObjectiveCacheDecision(
            can_reuse=False,
            mode="full_simulation",
            reject_reason="detector_change_not_center_only",
            changed_fields=changed_fields,
        )
    if changed_set != {"detector_center_sig"}:
        return GeometryObjectiveCacheDecision(
            can_reuse=False,
            mode="full_simulation",
            reject_reason="detector_change_not_center_only",
            changed_fields=changed_fields,
        )
    if not exact_center_remap_cache_available:
        return GeometryObjectiveCacheDecision(
            can_reuse=False,
            mode="full_simulation",
            reject_reason="exact_center_remap_cache_missing",
            changed_fields=changed_fields,
        )
    return GeometryObjectiveCacheDecision(
        can_reuse=True,
        mode="center_remap",
        reject_reason=None,
        changed_fields=changed_fields,
    )


qr_geometry_objective_cache_decision = geometry_objective_cache_decision


def geometry_objective_cache_trace_payload(
    decision: GeometryObjectiveCacheDecision,
    *,
    residual_component_count: int | None = None,
) -> dict[str, object]:
    """Build trace fields common to objective-cache tests and diagnostics."""

    payload: dict[str, object] = {
        "objective_cache_mode": decision.mode,
        "objective_cache_hit": bool(decision.can_reuse),
        "objective_cache_reject_reason": decision.reject_reason,
        "objective_process_peaks_called": bool(
            not decision.can_reuse and decision.mode == "full_simulation"
        ),
        "objective_signature_changed_fields": list(decision.changed_fields),
    }
    if residual_component_count is not None:
        payload["objective_residual_component_count"] = int(residual_component_count)
    return payload


def residual_order_signature(
    residual_labels: Sequence[object],
) -> tuple[str, ...]:
    return tuple(str(label) for label in residual_labels)


def center_remap_residual_shape_and_order_match(
    previous_residual_labels: Sequence[object],
    current_residual_labels: Sequence[object],
) -> bool:
    return residual_order_signature(previous_residual_labels) == residual_order_signature(
        current_residual_labels
    )


def signature_from_parts(parts: Mapping[str, object]) -> GeometryObjectiveSignature:
    """Build a signature from already-normalized field values."""

    def _part(name: str) -> tuple:
        value = parts.get(name, ())
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        return (value,)

    return GeometryObjectiveSignature(
        physics_sig=_part("physics_sig"),
        detector_center_sig=_part("detector_center_sig"),
        dataset_sig=_part("dataset_sig"),
        point_provider_sig=_part("point_provider_sig"),
        qr_branch_identity_sig=_part("qr_branch_identity_sig"),
        source_row_identity_sig=_part("source_row_identity_sig"),
        manual_selection_sig=_part("manual_selection_sig"),
        refined_peak_sig=_part("refined_peak_sig"),
        objective_mode_sig=_part("objective_mode_sig"),
        active_fit_parameter_sig=_part("active_fit_parameter_sig"),
    )
