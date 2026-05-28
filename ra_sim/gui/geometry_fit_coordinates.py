"""Pure coordinate helpers for geometry-fit dataset construction."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def finite_pair(
    entry: Mapping[str, object] | None,
    x_key: str,
    y_key: str,
) -> tuple[float, float] | None:
    if not isinstance(entry, Mapping):
        return None
    x_value = finite_float(entry.get(x_key))
    y_value = finite_float(entry.get(y_key))
    if x_value is None or y_value is None:
        return None
    return float(x_value), float(y_value)


def entry_frame(entry: Mapping[str, object] | None, key: str) -> str:
    if not isinstance(entry, Mapping):
        return ""
    return str(entry.get(key) or "").strip().lower()


def background_detector_pair_for_frame(
    entry: Mapping[str, object] | None,
    expected_frame: str,
) -> tuple[float, float] | None:
    if entry_frame(entry, "background_detector_input_frame") != expected_frame:
        return None
    return finite_pair(entry, "background_detector_x", "background_detector_y")


def native_detector_anchor(
    entry: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    anchor = background_detector_pair_for_frame(entry, "native_detector")
    if anchor is not None:
        return anchor
    for x_key, y_key in (
        ("detector_native_x", "detector_native_y"),
        ("native_col", "native_row"),
    ):
        anchor = finite_pair(entry, x_key, y_key)
        if anchor is not None:
            return anchor
    return None


def native_detector_anchor_with_provenance(
    entry: Mapping[str, object] | None,
) -> tuple[tuple[float, float], str] | None:
    anchor = background_detector_pair_for_frame(entry, "native_detector")
    if anchor is not None and isinstance(entry, Mapping):
        return (
            anchor,
            str(
                entry.get(
                    "background_detector_frame_provenance",
                    "saved_background_detector_native",
                )
            ),
        )
    for x_key, y_key, provenance in (
        ("detector_native_x", "detector_native_y", "saved_detector_native_xy"),
        ("native_col", "native_row", "saved_native_col_row"),
    ):
        anchor = finite_pair(entry, x_key, y_key)
        if anchor is not None:
            return anchor, provenance
    return None


def caked_angle_pair(
    entry: Mapping[str, object] | None,
    *,
    x_keys: Sequence[str],
    y_keys: Sequence[str],
) -> tuple[float, float] | None:
    if not isinstance(entry, Mapping):
        return None
    two_theta_value: float | None = None
    phi_value: float | None = None
    for key in x_keys:
        two_theta_value = finite_float(entry.get(key))
        if two_theta_value is not None:
            break
    for key in y_keys:
        phi_value = finite_float(entry.get(key))
        if phi_value is not None:
            break
    if two_theta_value is None or phi_value is None:
        return None
    return float(two_theta_value), float(phi_value)
