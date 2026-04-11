"""Helpers for summarizing and resetting GUI runtime live-cache state."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def live_cache_short_text(value: object, *, max_chars: int = 80) -> str:
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, int(max_chars) - 3)] + "..."


def live_cache_shape(value: object) -> list[int]:
    if value is None:
        return []
    try:
        return [int(v) for v in np.asarray(value).shape]
    except Exception:
        return []


def live_cache_count(value: object) -> int:
    if value is None:
        return 0
    try:
        return int(len(value))  # type: ignore[arg-type]
    except Exception:
        try:
            return int(np.asarray(value).size)
        except Exception:
            return 0


def live_cache_signature_summary(
    signature: object,
    *,
    max_items: int = 4,
) -> str | None:
    if signature is None:
        return None
    if isinstance(signature, np.ndarray):
        return "ndarray(shape={shape}, dtype={dtype})".format(
            shape=tuple(int(v) for v in signature.shape),
            dtype=str(signature.dtype),
        )
    if isinstance(signature, Mapping):
        keys = list(signature.keys())
        preview = ", ".join(live_cache_short_text(key, max_chars=16) for key in keys[:max_items])
        if len(keys) > max_items:
            preview += ", ..."
        return f"mapping(len={len(keys)}, keys=[{preview}])"
    if isinstance(signature, Sequence) and not isinstance(
        signature,
        (str, bytes, bytearray),
    ):
        items = list(signature)
        preview = ", ".join(live_cache_short_text(item, max_chars=16) for item in items[:max_items])
        if len(items) > max_items:
            preview += ", ..."
        return f"{type(signature).__name__}(len={len(items)}, items=[{preview}])"
    return live_cache_short_text(signature, max_chars=96)


def empty_peak_overlay_cache() -> dict[str, object]:
    return {
        "sig": None,
        "positions": [],
        "millers": [],
        "intensities": [],
        "records": [],
        "click_spatial_index": None,
        "restored_from_gui_state": False,
    }


def empty_qr_cylinder_overlay_cache() -> dict[str, object]:
    return {
        "signature": None,
        "paths": [],
    }


def live_cache_inventory_snapshot(simulation_runtime_state: object) -> dict[str, object]:
    source_snapshots_raw = getattr(
        simulation_runtime_state,
        "source_row_snapshots",
        {},
    )
    source_snapshots: list[dict[str, object]] = []
    if isinstance(source_snapshots_raw, Mapping):
        for raw_background_index, raw_snapshot in sorted(
            source_snapshots_raw.items(),
            key=lambda item: int(item[0]),
        ):
            if not isinstance(raw_snapshot, Mapping):
                continue
            row_count = raw_snapshot.get("row_count")
            if row_count is None:
                row_count = live_cache_count(raw_snapshot.get("rows", ()))
            source_snapshots.append(
                {
                    "background_index": int(raw_background_index),
                    "row_count": int(row_count),
                    "created_from": raw_snapshot.get("created_from"),
                    "signature_summary": live_cache_signature_summary(
                        raw_snapshot.get("simulation_signature")
                    ),
                }
            )
    return {
        "preview_active": bool(getattr(simulation_runtime_state, "preview_active", False)),
        "preview_sample_count": getattr(
            simulation_runtime_state,
            "preview_sample_count",
            None,
        ),
        "stored_hit_table_signature_present": bool(
            getattr(simulation_runtime_state, "stored_hit_table_signature", None) is not None
        ),
        "stored_hit_table_signature_summary": live_cache_signature_summary(
            getattr(simulation_runtime_state, "stored_hit_table_signature", None)
        ),
        "last_simulation_signature_summary": live_cache_signature_summary(
            getattr(simulation_runtime_state, "last_simulation_signature", None)
        ),
        "primary_contribution_cache_signature_summary": live_cache_signature_summary(
            getattr(simulation_runtime_state, "primary_contribution_cache_signature", None)
        ),
        "primary_source_mode": getattr(simulation_runtime_state, "primary_source_mode", None),
        "primary_active_contribution_key_count": live_cache_count(
            getattr(simulation_runtime_state, "primary_active_contribution_keys", ())
        ),
        "primary_hit_table_cache_entry_count": live_cache_count(
            getattr(simulation_runtime_state, "primary_hit_table_cache", {})
        ),
        "source_snapshots": source_snapshots,
    }


__all__ = [
    "empty_peak_overlay_cache",
    "empty_qr_cylinder_overlay_cache",
    "live_cache_count",
    "live_cache_inventory_snapshot",
    "live_cache_shape",
    "live_cache_short_text",
    "live_cache_signature_summary",
]
