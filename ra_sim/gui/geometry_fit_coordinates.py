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


def _anchor_payload(
    point: tuple[float, float],
    *,
    frame: str,
    authority: str,
    provenance: str,
) -> dict[str, object]:
    return {
        "point": (float(point[0]), float(point[1])),
        "space": "detector_px",
        "frame": str(frame),
        "authority": str(authority),
        "provenance": str(provenance),
        "fresh": True,
        "input_frame": str(frame),
    }


def _first_pair_value(value: object) -> tuple[float, float] | None:
    if isinstance(value, str | bytes | bytearray):
        return None
    try:
        first = value[0]  # type: ignore[index]
        second = value[1]  # type: ignore[index]
    except Exception:
        return None
    first_value = finite_float(first)
    second_value = finite_float(second)
    if first_value is None or second_value is None:
        return None
    return float(first_value), float(second_value)


def observed_detector_anchor_for_caked_projection(
    entry: Mapping[str, object] | None,
) -> dict[str, object] | None:
    native_anchor = native_detector_anchor_with_provenance(entry)
    if native_anchor is not None:
        anchor, provenance = native_anchor
        return _anchor_payload(
            anchor,
            frame="native_detector",
            authority="saved_native",
            provenance=str(provenance),
        )
    fit_anchor = finite_pair(entry, "fit_detector_x", "fit_detector_y")
    if fit_anchor is not None:
        return _anchor_payload(
            fit_anchor,
            frame="fit_detector",
            authority="display_cache",
            provenance="fit_detector_coords",
        )
    detector_anchor = finite_pair(entry, "detector_x", "detector_y")
    if (
        detector_anchor is not None
        and entry_frame(entry, "detector_input_frame") == "fit_detector"
    ):
        return _anchor_payload(
            detector_anchor,
            frame="fit_detector",
            authority="display_cache",
            provenance="detector_fit_frame",
        )
    return None


def simulated_detector_anchor_for_caked_projection(
    entry: Mapping[str, object] | None,
    source_row: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    for candidate in (entry, source_row):
        if not isinstance(candidate, Mapping):
            continue
        native_point = _first_pair_value(candidate.get("sim_native"))
        if native_point is not None:
            return _anchor_payload(
                native_point,
                frame="native_detector",
                authority="saved_native",
                provenance=str(candidate.get("sim_native_source") or "sim_native"),
            )
        for x_key, y_key, provenance in (
            ("refined_sim_native_x", "refined_sim_native_y", "refined_sim_native_px"),
            ("sim_native_x", "sim_native_y", "sim_native_xy"),
            ("sim_col_raw", "sim_row_raw", "sim_col_raw_row_raw"),
        ):
            anchor = finite_pair(candidate, x_key, y_key)
            if anchor is not None:
                return _anchor_payload(
                    anchor,
                    frame="native_detector",
                    authority="saved_native",
                    provenance=provenance,
                )
    return None


def _projection_failure(reason: str) -> dict[str, object]:
    return {
        "point": None,
        "space": "caked_deg",
        "frame": "caked_deg",
        "authority": "exact_projector",
        "provenance": "",
        "fresh": False,
        "source": None,
        "unavailable_reason": str(reason),
    }


_MISSING_PROJECTED_VALUE = object()


def _first_projected_raw(value: object) -> object:
    if isinstance(value, str | bytes | bytearray):
        return _MISSING_PROJECTED_VALUE
    try:
        return value[0]  # type: ignore[index]
    except TypeError:
        return value
    except Exception:
        return _MISSING_PROJECTED_VALUE


def project_detector_anchor_to_caked_fit_space(
    anchor: tuple[float, float],
    projector: object,
    *,
    local_params: Mapping[str, object] | None = None,
    anchor_kind: str = "measured",
    input_frame: str = "native_detector",
    source_fallback: str = "fit_space_projector_native_detector",
) -> dict[str, object]:
    detector_anchor = _first_pair_value(anchor)
    if detector_anchor is None:
        return _projection_failure("nonfinite_detector_coords")
    detector_col, detector_row = detector_anchor
    if not callable(projector):
        return _projection_failure("fit_space_projector_unavailable")
    try:
        projected = projector(
            [float(detector_col)],
            [float(detector_row)],
            local_params=dict(local_params or {}),
            anchor_kind=str(anchor_kind),
            input_frame=str(input_frame),
        )
    except Exception as exc:
        return _projection_failure(f"fit_space_projector_exception:{type(exc).__name__}")
    if not isinstance(projected, Mapping):
        return _projection_failure("fit_space_projector_returned_non_mapping")
    two_theta_raw = _first_projected_raw(projected.get("two_theta_deg", []))
    phi_raw = _first_projected_raw(projected.get("phi_deg", []))
    if two_theta_raw is _MISSING_PROJECTED_VALUE or phi_raw is _MISSING_PROJECTED_VALUE:
        return _projection_failure("fit_space_projector_returned_no_caked_point")
    two_theta = finite_float(two_theta_raw)
    phi = finite_float(phi_raw)
    if two_theta is None or phi is None:
        return _projection_failure(f"nonfinite_{str(anchor_kind)}_caked_projection")
    if projected.get("valid") is False:
        return _projection_failure(
            str(
                projected.get("invalid_reason")
                or f"nonfinite_{str(anchor_kind)}_caked_projection"
            )
        )
    source = str(projected.get("caked_projection_source") or source_fallback)
    return {
        "point": (float(two_theta), float(phi)),
        "space": "caked_deg",
        "frame": "caked_deg",
        "authority": "exact_projector",
        "provenance": source,
        "fresh": True,
        "source": source,
        "unavailable_reason": None,
    }


def resolve_fit_space_anchor(
    entry: Mapping[str, object] | None,
    source_row: Mapping[str, object] | None,
    objective_space: str,
    projector: object,
) -> dict[str, object] | None:
    del source_row
    anchor_payload = observed_detector_anchor_for_caked_projection(entry)
    if str(objective_space or "").strip().lower() != "caked_deg":
        return anchor_payload
    if anchor_payload is None:
        return _projection_failure("missing_observed_detector_anchor")
    return project_detector_anchor_to_caked_fit_space(
        anchor_payload["point"],  # type: ignore[arg-type]
        projector,
        input_frame=str(anchor_payload.get("input_frame") or "native_detector"),
        anchor_kind="measured",
        source_fallback="fit_space_projector_native_detector",
    )
