"""Shared mosaic-top representative selection for GUI placement paths."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


ZERO_QR_BRANCH_ID = "00l"
KNOWN_BRANCH_IDS = {"+x", "-x", ZERO_QR_BRANCH_ID}
SELECTION_REASON = "mosaic_top_per_branch"
Q_GROUP_SELECTION_REASON = "mosaic_top_per_q_group"


def _finite_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except Exception:
        return None
    return float(numeric) if np.isfinite(numeric) else None


_RANK_MISSING_VALUE = 1.0e300


def _plain(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    if isinstance(value, list):
        return tuple(_plain(item) for item in value)
    return value


def _json_safe_rank_value(value: object) -> object:
    value = _plain(value)
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else float(_RANK_MISSING_VALUE)
    if isinstance(value, tuple):
        return tuple(_json_safe_rank_value(item) for item in value)
    return value


def _stable_text(value: object) -> str:
    value = _plain(value)
    if isinstance(value, tuple):
        return "(" + ",".join(_stable_text(item) for item in value) + ")"
    return str(value)


def normalize_q_group_key(value: object) -> tuple[object, ...] | None:
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    if isinstance(value, list):
        return tuple(_plain(item) for item in value)
    return None


def _is_q_group_target(value: object) -> bool:
    key = normalize_q_group_key(value)
    return bool(key and key[0] == "q_group")


def normalize_hkl_key(value: object) -> tuple[int, int, int] | None:
    if isinstance(value, Mapping):
        return normalize_hkl_key(value.get("hkl", value.get("hkl_raw")))
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) < 3:
        return None
    vals: list[int] = []
    for item in value[:3]:
        numeric = _finite_float(item)
        if numeric is None:
            return None
        vals.append(int(round(float(numeric))))
    return (int(vals[0]), int(vals[1]), int(vals[2]))


def _q_group_key_is_00l(value: object) -> bool:
    key = normalize_q_group_key(value)
    if not key or len(key) < 4 or key[0] != "q_group":
        return False
    h_value = _finite_float(key[2])
    if h_value is None:
        return False
    return int(round(float(h_value))) == 0


def _entry_is_00l(
    entry: Mapping[str, object] | None,
    target_key: object = None,
) -> bool:
    hkl_key = normalize_hkl_key(entry) if isinstance(entry, Mapping) else None
    if hkl_key is not None:
        return int(hkl_key[0]) == 0 and int(hkl_key[1]) == 0
    return _q_group_key_is_00l(target_key)


def target_key_from_entry(
    entry: Mapping[str, object] | None,
    *,
    target_kind: str | None = None,
) -> tuple[object, ...] | None:
    if not isinstance(entry, Mapping):
        return None
    if target_kind == "hkl":
        hkl_key = normalize_hkl_key(entry)
        return ("hkl",) + hkl_key if hkl_key is not None else None
    if target_kind == "q_group":
        q_key = normalize_q_group_key(entry.get("q_group_key"))
        return q_key
    q_key = normalize_q_group_key(entry.get("q_group_key"))
    if q_key is not None:
        return q_key
    hkl_key = normalize_hkl_key(entry)
    return ("hkl",) + hkl_key if hkl_key is not None else None


def _unknown_branch_id(
    entry: Mapping[str, object] | None,
    target_key: object = None,
) -> str:
    source_group = None
    if isinstance(entry, Mapping):
        source_branch = entry.get("source_branch_index")
        if source_branch is not None:
            source_group = (target_key, "source_branch", _plain(source_branch))
        if source_group is None:
            hkl_key = normalize_hkl_key(entry)
            if hkl_key is not None:
                source_group = (target_key, "hkl", hkl_key)
        if source_group is None:
            source_group = entry.get("source_label")
        if source_group is None:
            source_group = entry.get("source_table_index")
        if source_group is None:
            source_group = entry.get("source_reflection_index")
    if source_group is None:
        source_group = target_key if target_key is not None else "global"
    return "unknown:" + _stable_text(source_group)


def normalize_branch_id(
    entry: Mapping[str, object] | None,
    *,
    target_key: object = None,
    profile_cache: Mapping[str, object] | None = None,
) -> tuple[str, str]:
    """Return normalized branch identity without mapping legacy branch indices.

    Known generated branches are the physical signed-x branches ``+x`` and ``-x``;
    HK=0 Qr/Qz rows use one canonical ``00l`` branch. If explicit signed-x
    branch metadata is absent, the candidate is kept in a stable unknown branch
    and is never merged into either known branch.
    """

    if _entry_is_00l(entry, target_key):
        return ZERO_QR_BRANCH_ID, "generated"

    if isinstance(entry, Mapping):
        raw_branch = entry.get("branch_id")
        if raw_branch is not None:
            branch_text = str(raw_branch)
            if branch_text in KNOWN_BRANCH_IDS or branch_text.startswith("unknown:"):
                source = str(entry.get("branch_source", "") or "")
                if source not in {"generated", "inferred", "unknown"}:
                    source = "generated" if branch_text in KNOWN_BRANCH_IDS else "unknown"
                return branch_text, source
        for key in ("x_branch_id", "signed_x_branch", "branch_x_sign"):
            raw_signed = entry.get(key)
            if raw_signed in {"+x", "-x"}:
                return str(raw_signed), "generated"
            signed = _finite_float(raw_signed)
            if signed is not None and signed != 0.0:
                return ("+x" if signed > 0.0 else "-x"), "generated"
        signed = _array_value(profile_cache, "beam_x_array", entry.get("best_sample_index"))
        if signed is not None and signed != 0.0:
            return ("+x" if signed > 0.0 else "-x"), "generated"
    return _unknown_branch_id(entry, target_key), "unknown"


def annotate_branch_metadata(
    entry: Mapping[str, object],
    *,
    target_key: object = None,
    profile_cache: Mapping[str, object] | None = None,
) -> dict[str, object]:
    annotated = dict(entry)
    branch_id, branch_source = normalize_branch_id(
        annotated,
        target_key=target_key,
        profile_cache=profile_cache,
    )
    annotated["branch_id"] = str(branch_id)
    annotated["branch_source"] = str(branch_source)
    return annotated


def annotate_selection_metadata(
    entry: Mapping[str, object],
    *,
    target_key: object = None,
    profile_cache: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Attach generated branch and true mosaic-weight metadata to a GUI ray copy."""

    annotated = annotate_branch_metadata(
        entry,
        target_key=target_key,
        profile_cache=profile_cache,
    )
    mosaic = true_mosaic_weight(annotated, profile_cache=profile_cache)
    if mosaic is not None:
        annotated["mosaic_weight"] = float(mosaic)
    for field_key, profile_key in (
        ("theta_offset", "theta_array"),
        ("phi_offset", "phi_array"),
        ("beam_x_offset", "beam_x_array"),
        ("beam_y_offset", "beam_y_array"),
        ("wavelength_offset", "wavelength_array"),
    ):
        if _finite_float(annotated.get(field_key)) is not None:
            continue
        value = _array_value(profile_cache, profile_key, annotated.get("best_sample_index"))
        if value is not None:
            annotated[field_key] = float(value)
    return annotated


def _array_value(
    profile_cache: Mapping[str, object] | None, key: str, index: object
) -> float | None:
    idx_float = _finite_float(index)
    if idx_float is None:
        return None
    idx = int(idx_float)
    if idx < 0 or profile_cache is None:
        return None
    try:
        values = profile_cache.get(key)
    except Exception:
        return None
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except Exception:
        return None
    if idx >= int(array.size):
        return None
    return _finite_float(array[idx])


def true_mosaic_weight(
    entry: Mapping[str, object] | None,
    *,
    profile_cache: Mapping[str, object] | None = None,
) -> float | None:
    if not isinstance(entry, Mapping):
        return None
    weight = _finite_float(entry.get("mosaic_weight"))
    if weight is not None:
        return float(weight)
    for key in ("sample_weights", "mosaic_weight_array", "mosaic_weights"):
        weight = _array_value(profile_cache, key, entry.get("best_sample_index"))
        if weight is not None:
            return float(weight)
    return None


def _cached_row_value(entry: Mapping[str, object], index: int) -> float | None:
    for key in ("cache_row", "source_cache_row", "intersection_cache_row", "row"):
        row = entry.get(key)
        try:
            seq = np.asarray(row, dtype=float).reshape(-1)
        except Exception:
            continue
        if 0 <= index < int(seq.size):
            value = _finite_float(seq[index])
            if value is not None:
                return value
    return None


def _offset_value(
    entry: Mapping[str, object],
    *keys: str,
    profile_cache: Mapping[str, object] | None = None,
    profile_key: str | None = None,
    cache_column: int | None = None,
) -> float | None:
    for key in keys:
        value = _finite_float(entry.get(key))
        if value is not None:
            return value
    if profile_key is not None:
        value = _array_value(profile_cache, profile_key, entry.get("best_sample_index"))
        if value is not None:
            return value
    if cache_column is not None:
        return _cached_row_value(entry, int(cache_column))
    return None


def angular_mosaic_metric(
    entry: Mapping[str, object] | None,
    *,
    profile_cache: Mapping[str, object] | None = None,
) -> float | None:
    """Squared angular offset from mosaic top: theta_offset^2 + phi_offset^2."""

    if not isinstance(entry, Mapping):
        return None
    theta = _offset_value(
        entry,
        "theta_offset",
        "theta_offset_deg",
        profile_cache=profile_cache,
        profile_key="theta_array",
        cache_column=11,
    )
    phi = _offset_value(
        entry,
        "phi_offset",
        "phi_offset_deg",
        profile_cache=profile_cache,
        profile_key="phi_array",
        cache_column=12,
    )
    if theta is None or phi is None:
        return None
    return float(theta * theta + phi * phi)


def beam_offset_metric(
    entry: Mapping[str, object] | None,
    *,
    profile_cache: Mapping[str, object] | None = None,
) -> float | None:
    if not isinstance(entry, Mapping):
        return None
    x_val = _offset_value(
        entry,
        "beam_x_offset",
        profile_cache=profile_cache,
        profile_key="beam_x_array",
        cache_column=9,
    )
    y_val = _offset_value(
        entry,
        "beam_y_offset",
        profile_cache=profile_cache,
        profile_key="beam_y_array",
        cache_column=10,
    )
    if x_val is None or y_val is None:
        return None
    return float(x_val * x_val + y_val * y_val)


def wavelength_offset_metric(
    entry: Mapping[str, object] | None,
    *,
    profile_cache: Mapping[str, object] | None = None,
) -> float | None:
    if not isinstance(entry, Mapping):
        return None
    value = _offset_value(
        entry,
        "wavelength_offset",
        profile_cache=profile_cache,
        profile_key="wavelength_array",
        cache_column=13,
    )
    if value is None:
        return None
    return float(abs(value))


def intensity_tie_breaker(entry: Mapping[str, object] | None) -> float | None:
    if not isinstance(entry, Mapping):
        return None
    return _finite_float(entry.get("intensity"))


def mosaic_top_rank_key(
    entry: Mapping[str, object] | None,
    *,
    branch_id: str | None = None,
    source_order: int = 0,
    profile_cache: Mapping[str, object] | None = None,
) -> tuple[object, ...]:
    """Stable serializable key for selecting the mosaic-top representative.

    Mosaic top means the ray with highest true ``mosaic_weight``. When true
    mosaic weight is absent or non-finite, angular mosaic distance
    ``theta_offset**2 + phi_offset**2`` is the fallback metric.
    """

    if not isinstance(entry, Mapping):
        return (
            1,
            1,
            _RANK_MISSING_VALUE,
            _RANK_MISSING_VALUE,
            _RANK_MISSING_VALUE,
            _RANK_MISSING_VALUE,
            _RANK_MISSING_VALUE,
            int(source_order),
        )
    normalized = annotate_selection_metadata(entry, profile_cache=profile_cache)
    entry_branch_id = str(normalized.get("branch_id", ""))
    branch_match = 0 if branch_id is None or entry_branch_id == str(branch_id) else 1
    mosaic = true_mosaic_weight(normalized, profile_cache=profile_cache)
    angular = angular_mosaic_metric(normalized, profile_cache=profile_cache)
    beam = beam_offset_metric(normalized, profile_cache=profile_cache)
    wavelength = wavelength_offset_metric(normalized, profile_cache=profile_cache)
    intensity = intensity_tie_breaker(normalized)

    if mosaic is not None:
        metric_mode = 0
        primary = -float(mosaic)
        angular_key = _RANK_MISSING_VALUE
    else:
        metric_mode = 1
        primary = _RANK_MISSING_VALUE
        angular_key = float(angular) if angular is not None else _RANK_MISSING_VALUE
    return (
        int(branch_match),
        int(metric_mode),
        float(primary),
        float(angular_key),
        float(beam) if beam is not None else _RANK_MISSING_VALUE,
        float(wavelength) if wavelength is not None else _RANK_MISSING_VALUE,
        -float(intensity) if intensity is not None else _RANK_MISSING_VALUE,
        int(source_order),
    )


def provenance_tuple(entry: Mapping[str, object] | None) -> tuple[object, ...]:
    if not isinstance(entry, Mapping):
        return (None, None, None, None, None, None, None)
    branch_id, _branch_source = normalize_branch_id(
        entry,
        target_key=entry.get("q_group_key"),
    )
    return (
        branch_id,
        normalize_hkl_key(entry),
        normalize_q_group_key(entry.get("q_group_key")),
        _plain(entry.get("source_table_index")),
        _plain(entry.get("source_row_index")),
        _plain(entry.get("best_sample_index")),
        _plain(entry.get("mosaic_weight")),
    )


def select_mosaic_top_representative(
    entries: Sequence[Mapping[str, object]] | None,
    *,
    branch_id: str | None = None,
    target_key: object = None,
    profile_cache: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    candidates: list[tuple[tuple[object, ...], dict[str, object]]] = []
    for order, raw_entry in enumerate(entries or ()):
        if not isinstance(raw_entry, Mapping):
            continue
        annotated = annotate_selection_metadata(
            raw_entry,
            target_key=target_key,
            profile_cache=profile_cache,
        )
        if branch_id is not None and annotated.get("branch_id") != str(branch_id):
            continue
        rank_key = mosaic_top_rank_key(
            annotated,
            branch_id=branch_id,
            source_order=int(order),
            profile_cache=profile_cache,
        )
        candidates.append((rank_key, annotated))
    if not candidates:
        return None
    rank_key, selected = min(candidates, key=lambda item: item[0])
    representative = dict(selected)
    if branch_id is None and _is_q_group_target(target_key):
        representative["selection_reason"] = Q_GROUP_SELECTION_REASON
        representative["selection_scope"] = "q_group"
        if target_key is not None:
            representative["selected_q_group_key"] = _plain(target_key)
    else:
        representative["selection_reason"] = SELECTION_REASON
    representative["mosaic_top_rank_key"] = tuple(_json_safe_rank_value(item) for item in rank_key)
    return representative


def build_selection_cache(
    entries: Sequence[Mapping[str, object]] | None,
    *,
    target_key_fn: Callable[[Mapping[str, object]], object | None] | None = None,
    profile_cache: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    buckets: dict[tuple[object, str], list[Mapping[str, object]]] = {}
    order: list[tuple[object, str]] = []
    for raw_entry in entries or ():
        if not isinstance(raw_entry, Mapping):
            continue
        target_key = (
            target_key_fn(raw_entry)
            if callable(target_key_fn)
            else target_key_from_entry(raw_entry)
        )
        if target_key is None:
            target_key = (
                "source",
                _plain(raw_entry.get("source_table_index")),
                _plain(raw_entry.get("source_row_index")),
            )
        annotated = annotate_selection_metadata(
            raw_entry,
            target_key=target_key,
            profile_cache=profile_cache,
        )
        key = (_plain(target_key), str(annotated.get("branch_id")))
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(raw_entry)

    representatives: list[dict[str, object]] = []
    for target_key, branch_id in order:
        representative = select_mosaic_top_representative(
            buckets[(target_key, branch_id)],
            branch_id=str(branch_id),
            target_key=target_key,
            profile_cache=profile_cache,
        )
        if representative is not None:
            representatives.append(representative)
    return representatives
