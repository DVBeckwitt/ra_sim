"""Helpers for persistent GUI runtime update trace logging."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np

from ra_sim.debug_controls import runtime_update_trace_logging_enabled


def initial_update_decision_trace() -> dict[str, object]:
    """Return the default per-update classifier trace payload."""

    return {
        "update_action": None,
        "update_reason": None,
        "requires_worker": None,
        "missing_contribution_count": None,
        "center_remap_used": None,
        "primary_prune_cache_mode": None,
        "qr_selector_entries_retained": None,
        "qr_selector_entries_refreshed": None,
        "qr_selector_refresh_deferred": None,
        "source_row_snapshots_retained": None,
        "q_group_content_signature_changed": None,
        "geometry_fitter_handoff_valid": None,
        "qr_selector_branch_identity_retained": None,
        "detector_projection_cache_refreshed": None,
        "caked_projection_cache_invalidated": None,
        "center_remap_fallback_reason": None,
        "classifier_update_action": None,
        "classifier_update_reason": None,
        "classifier_requires_worker": None,
        "classifier_requires_analysis": None,
        "classifier_missing_contribution_count": None,
        "effective_update_action": None,
    }


def set_update_decision_trace(
    trace: dict[str, object],
    *,
    update_action: str | None = None,
    update_reason: object = None,
    requires_worker: bool | None = None,
    missing_contribution_count: int | None = None,
    center_remap_used: bool | None = None,
    primary_prune_cache_mode: object = None,
) -> None:
    """Update the common runtime update decision fields."""

    if update_action is not None:
        trace["update_action"] = str(update_action)
    if update_reason is not None:
        trace["update_reason"] = str(update_reason)
    if requires_worker is not None:
        trace["requires_worker"] = bool(requires_worker)
    if missing_contribution_count is not None:
        trace["missing_contribution_count"] = int(missing_contribution_count)
    if center_remap_used is not None:
        trace["center_remap_used"] = bool(center_remap_used)
    if primary_prune_cache_mode is not None:
        trace["primary_prune_cache_mode"] = str(primary_prune_cache_mode)


def set_qr_selector_trace(
    trace: dict[str, object],
    *,
    policy: object = None,
    qr_selector_entries_retained: bool | None = None,
    qr_selector_entries_refreshed: bool | None = None,
    qr_selector_refresh_deferred: bool | None = None,
    source_row_snapshots_retained: bool | None = None,
    q_group_content_signature_changed: bool | None = None,
    geometry_fitter_handoff_valid: bool | None = None,
) -> None:
    """Update Qr selector retention fields in the runtime update trace."""

    if policy is not None:
        if qr_selector_entries_retained is None:
            qr_selector_entries_retained = bool(
                getattr(policy, "retain_geometry_q_group_entries", False)
            )
        if qr_selector_refresh_deferred is None:
            qr_selector_refresh_deferred = bool(
                getattr(policy, "defer_q_group_refresh_until_rows_available", False)
            )
        if source_row_snapshots_retained is None:
            source_row_snapshots_retained = bool(
                getattr(policy, "retain_source_row_snapshots", False)
            )
        if geometry_fitter_handoff_valid is None:
            geometry_fitter_handoff_valid = bool(
                getattr(policy, "retain_geometry_q_group_entries", False)
                and getattr(policy, "retain_source_row_snapshots", False)
                and getattr(policy, "retain_intersection_caches", False)
                and getattr(policy, "retain_manual_pick_cache", False)
                and not getattr(policy, "require_q_group_refresh_after_apply", False)
            )
    if qr_selector_entries_retained is not None:
        trace["qr_selector_entries_retained"] = bool(qr_selector_entries_retained)
    if qr_selector_entries_refreshed is not None:
        trace["qr_selector_entries_refreshed"] = bool(qr_selector_entries_refreshed)
    if qr_selector_refresh_deferred is not None:
        trace["qr_selector_refresh_deferred"] = bool(qr_selector_refresh_deferred)
    if source_row_snapshots_retained is not None:
        trace["source_row_snapshots_retained"] = bool(source_row_snapshots_retained)
    if q_group_content_signature_changed is not None:
        trace["q_group_content_signature_changed"] = bool(q_group_content_signature_changed)
    if geometry_fitter_handoff_valid is not None:
        trace["geometry_fitter_handoff_valid"] = bool(geometry_fitter_handoff_valid)


def set_detector_center_remap_trace(
    trace: dict[str, object],
    *,
    qr_selector_branch_identity_retained: bool | None = None,
    detector_projection_cache_refreshed: bool | None = None,
    caked_projection_cache_invalidated: bool | None = None,
    geometry_fitter_handoff_valid: bool | None = None,
    center_remap_fallback_reason: object = None,
) -> None:
    """Update detector-center remap fields in the runtime update trace."""

    if qr_selector_branch_identity_retained is not None:
        trace["qr_selector_branch_identity_retained"] = bool(
            qr_selector_branch_identity_retained
        )
    if detector_projection_cache_refreshed is not None:
        trace["detector_projection_cache_refreshed"] = bool(detector_projection_cache_refreshed)
    if caked_projection_cache_invalidated is not None:
        trace["caked_projection_cache_invalidated"] = bool(caked_projection_cache_invalidated)
    if geometry_fitter_handoff_valid is not None:
        trace["geometry_fitter_handoff_valid"] = bool(geometry_fitter_handoff_valid)
    if center_remap_fallback_reason is not None:
        trace["center_remap_fallback_reason"] = str(center_remap_fallback_reason)


def _update_decision_action(trace: Mapping[str, object]) -> str:
    """Return the active update action name from a decision trace."""

    return str(trace.get("update_action") or "")


def ensure_update_decision_defaults(trace: dict[str, object], reason: str) -> None:
    """Fill display-only decision defaults when no update action is recorded."""

    if _update_decision_action(trace):
        return
    set_update_decision_trace(
        trace,
        update_action="display_only",
        update_reason=reason,
        requires_worker=False,
        missing_contribution_count=0,
        center_remap_used=False,
        primary_prune_cache_mode="none",
    )


def set_classifier_decision_trace(
    trace: dict[str, object],
    decision: object,
    *,
    effective_action: object,
) -> None:
    """Record classifier and effective-action fields in a decision trace."""

    trace["classifier_update_action"] = getattr(decision.action, "value")
    trace["classifier_update_reason"] = str(decision.reason)
    trace["classifier_requires_worker"] = bool(decision.requires_worker)
    trace["classifier_requires_analysis"] = bool(decision.requires_analysis)
    trace["classifier_missing_contribution_count"] = int(
        len(decision.missing_contribution_keys)
    )
    trace["effective_update_action"] = getattr(effective_action, "value")


def resolve_runtime_update_trace_path(
    downloads_dir: Path | str | None,
    *,
    current_time: datetime | None = None,
    fallback_dir: Path | str | None = None,
) -> Path:
    """Return the daily GUI runtime trace log path."""

    if downloads_dir is not None:
        base_dir = Path(downloads_dir)
    elif fallback_dir is not None:
        base_dir = Path(fallback_dir)
    else:
        base_dir = Path.home() / "Downloads"
    stamp = (current_time or datetime.now()).strftime("%Y%m%d")
    return base_dir / f"runtime_update_trace_{stamp}.log"


def _trace_value_text(value: object) -> str:
    """Format one runtime trace field value."""

    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isnan(numeric):
            return "nan"
        if np.isposinf(numeric):
            return "inf"
        if np.isneginf(numeric):
            return "-inf"
        return f"{numeric:.6f}"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        parts = [
            f"{key}:{_trace_value_text(item)}"
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        ]
        return "{" + ",".join(parts) + "}"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        preview = [_trace_value_text(item) for item in items[:8]]
        if len(items) > 8:
            preview.append("...")
        return "[" + ",".join(preview) + "]"
    return str(value).replace("\r", "\\r").replace("\n", "\\n")


def format_runtime_update_trace_line(
    event: str,
    *,
    timestamp: datetime | None = None,
    pid: int | None = None,
    fields: Mapping[str, object] | None = None,
) -> str:
    """Format one append-only GUI runtime trace line."""

    now = timestamp or datetime.now()
    parts = [
        now.isoformat(timespec="milliseconds"),
        f"pid={int(os.getpid() if pid is None else pid)}",
        f"event={str(event)}",
    ]
    for key, value in sorted((fields or {}).items(), key=lambda item: str(item[0])):
        if value is None:
            continue
        parts.append(f"{key}={_trace_value_text(value)}")
    return " ".join(parts)


def append_runtime_update_trace_line(
    path: Path | str,
    event: str,
    **fields: object,
) -> None:
    """Append one line to the GUI runtime trace log."""

    if not runtime_update_trace_logging_enabled():
        return
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(format_runtime_update_trace_line(event, fields=fields) + "\n")
        handle.flush()
