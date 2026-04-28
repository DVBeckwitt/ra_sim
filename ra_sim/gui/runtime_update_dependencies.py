"""Pure GUI update dependency classification helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class UpdateAction(Enum):
    DISPLAY_ONLY = "display_only"
    COMBINE_ONLY = "combine_only"
    DETECTOR_CENTER_REMAP = "detector_center_remap"
    PRIMARY_PRUNE_REUSE = "primary_prune_reuse"
    PRIMARY_PRUNE_FILL = "primary_prune_fill"
    ANALYSIS_ONLY = "analysis_only"
    HIT_TABLE_REFRESH = "hit_table_refresh"
    FULL_SIMULATION = "full_simulation"


@dataclass(frozen=True)
class SimulationDependencySignatures:
    """Hashable dependency slices for one GUI update.

    ``physics_sig`` must include every input that changes ray paths, reciprocal
    geometry, detector projection scale, source sampling, or optics mode.
    """

    source_sig: tuple[object, ...]
    physics_sig: tuple[object, ...]
    detector_projection_sig: tuple[object, ...]
    detector_center_sig: tuple[object, ...]
    primary_filter_sig: tuple[object, ...]
    combine_sig: tuple[object, ...]
    analysis_geometry_sig: tuple[object, ...]
    display_sig: tuple[object, ...]
    hit_table_sig: tuple[object, ...]
    full_image_sig: tuple[object, ...]


@dataclass(frozen=True)
class UpdateDecision:
    action: UpdateAction
    reason: str
    requires_worker: bool
    requires_analysis: bool = False
    missing_contribution_keys: frozenset[object] = field(default_factory=frozenset)


@dataclass(frozen=True)
class RuntimeCacheState:
    can_remap_detector_center: bool = False
    prune_cache_mode: str | None = None
    missing_contribution_keys: frozenset[object] = field(default_factory=frozenset)


_SEMANTIC_SIGNATURE_FIELDS = (
    "source_sig",
    "physics_sig",
    "detector_projection_sig",
    "detector_center_sig",
    "primary_filter_sig",
    "combine_sig",
    "analysis_geometry_sig",
    "display_sig",
    "hit_table_sig",
)

_PHYSICS_INVALIDATION_FIELDS = frozenset(
    {
        "source_sig",
        "physics_sig",
        "detector_projection_sig",
    }
)


def _full_simulation(reason: str) -> UpdateDecision:
    return UpdateDecision(
        action=UpdateAction.FULL_SIMULATION,
        reason=reason,
        requires_worker=True,
    )


def _changed_fields(
    previous: SimulationDependencySignatures,
    current: SimulationDependencySignatures,
) -> frozenset[str]:
    return frozenset(
        field_name
        for field_name in _SEMANTIC_SIGNATURE_FIELDS
        if getattr(previous, field_name) != getattr(current, field_name)
    )


def _classify_primary_prune(cache_state: RuntimeCacheState) -> UpdateDecision:
    mode = str(cache_state.prune_cache_mode or "").strip().lower()
    missing_keys = frozenset(cache_state.missing_contribution_keys or frozenset())
    if mode == "reuse" and not missing_keys:
        return UpdateDecision(
            action=UpdateAction.PRIMARY_PRUNE_REUSE,
            reason="primary_filter_changed_cache_reuse",
            requires_worker=False,
            missing_contribution_keys=missing_keys,
        )
    if mode == "fill" or (missing_keys and mode != "full"):
        return UpdateDecision(
            action=UpdateAction.PRIMARY_PRUNE_FILL,
            reason="primary_filter_changed_cache_fill",
            requires_worker=True,
            missing_contribution_keys=missing_keys,
        )
    return _full_simulation("primary_filter_changed_no_compatible_cache")


def classify_update(
    previous: SimulationDependencySignatures | None,
    current: SimulationDependencySignatures,
    cache_state: RuntimeCacheState,
) -> UpdateDecision:
    """Return a conservative pure classification for one GUI update."""

    if previous is None:
        return _full_simulation("initial_update")

    changed = _changed_fields(previous, current)
    full_image_changed = previous.full_image_sig != current.full_image_sig

    if changed & _PHYSICS_INVALIDATION_FIELDS:
        return _full_simulation("physics_dependency_changed")

    if changed == frozenset({"detector_center_sig", "hit_table_sig"}):
        if bool(cache_state.can_remap_detector_center):
            return UpdateDecision(
                action=UpdateAction.DETECTOR_CENTER_REMAP,
                reason="detector_center_changed_exact_cache_available",
                requires_worker=False,
            )
        return _full_simulation("detector_center_changed_without_exact_cache")

    if not changed:
        if full_image_changed:
            return _full_simulation("full_image_signature_changed")
        return UpdateDecision(
            action=UpdateAction.DISPLAY_ONLY,
            reason="no_dependency_change",
            requires_worker=False,
        )

    if len(changed) != 1:
        return _full_simulation("mixed_dependency_change")

    changed_field = next(iter(changed))
    if changed_field == "detector_center_sig":
        if bool(cache_state.can_remap_detector_center):
            return UpdateDecision(
                action=UpdateAction.DETECTOR_CENTER_REMAP,
                reason="detector_center_changed_exact_cache_available",
                requires_worker=False,
            )
        return _full_simulation("detector_center_changed_without_exact_cache")

    if changed_field == "primary_filter_sig":
        return _classify_primary_prune(cache_state)

    if changed_field == "combine_sig":
        return UpdateDecision(
            action=UpdateAction.COMBINE_ONLY,
            reason="combine_dependency_changed",
            requires_worker=False,
        )

    if changed_field == "analysis_geometry_sig":
        return UpdateDecision(
            action=UpdateAction.ANALYSIS_ONLY,
            reason="analysis_geometry_changed",
            requires_worker=False,
            requires_analysis=True,
        )

    if changed_field == "display_sig":
        return UpdateDecision(
            action=UpdateAction.DISPLAY_ONLY,
            reason="display_dependency_changed",
            requires_worker=False,
        )

    if changed_field == "hit_table_sig":
        return UpdateDecision(
            action=UpdateAction.HIT_TABLE_REFRESH,
            reason="hit_table_dependency_changed",
            requires_worker=True,
        )

    return _full_simulation("unknown_dependency_change")
