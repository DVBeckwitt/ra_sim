"""Helpers for per-background theta metadata and fit-background selection."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BackgroundThetaRuntimeBindings:
    """Live GUI/runtime inputs used by background-theta helpers."""

    osc_files: Sequence[object]
    current_background_index: int
    theta_initial_var: object | None
    defaults: Mapping[str, object] | None
    theta_initial: object
    background_theta_list_var: object | None
    geometry_theta_offset_var: object | None
    geometry_fit_background_selection_var: object | None
    fit_theta_checkbutton: object | None = None
    theta_controls: object = None
    set_background_file_status_text: Callable[[], None] | None = None
    schedule_update: Callable[[], None] | None = None
    progress_label: object | None = None
    progress_label_geometry: object | None = None


@dataclass(frozen=True)
class BackgroundThetaRuntimeCallbacks:
    """Bound helpers for live background-theta runtime workflows."""

    current_geometry_fit_background_indices: Callable[..., list[int]]
    geometry_fit_uses_shared_theta_offset: Callable[..., bool]
    current_geometry_theta_offset: Callable[..., float]
    current_background_theta_values: Callable[..., list[float]]
    background_theta_for_index: Callable[..., float]
    sync_background_theta_controls: Callable[..., None]
    apply_background_theta_metadata: Callable[..., bool]
    apply_geometry_fit_background_selection: Callable[..., bool]
    sync_geometry_fit_background_selection: Callable[..., None]


def _read_var_value(value: object) -> object:
    if value is None:
        return None
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter()
    return value


def _write_var_value(value: object, new_value: object) -> None:
    setter = getattr(value, "set", None)
    if callable(setter):
        setter(new_value)


def _set_widget_text(widget: object, text: str) -> None:
    config = getattr(widget, "config", None)
    if callable(config):
        config(text=text)


def _resolve_runtime_value(value_or_callable: object) -> object:
    if callable(value_or_callable):
        try:
            return value_or_callable()
        except Exception:
            return None
    return value_or_callable


def background_theta_default_value(
    *,
    theta_initial_var: object | None,
    defaults: Mapping[str, object] | None,
    theta_initial: object,
) -> float:
    """Return the fallback theta value used when no per-background list exists."""

    try:
        if theta_initial_var is not None:
            return float(_read_var_value(theta_initial_var))
    except Exception:
        pass
    try:
        if defaults is not None:
            return float(defaults.get("theta_initial", theta_initial))
    except Exception:
        pass
    return float(theta_initial)


def format_background_theta_values(values: Sequence[object]) -> str:
    """Format per-background theta values for the GUI entry."""

    formatted: list[str] = []
    for value in values:
        try:
            theta_val = float(value)
        except Exception:
            continue
        if not np.isfinite(theta_val):
            continue
        formatted.append(f"{theta_val:.6g}")
    return ", ".join(formatted)


def parse_background_theta_values(
    raw_text: object,
    *,
    expected_count: int | None = None,
) -> list[float]:
    """Parse comma/space/semicolon separated background theta values."""

    text = str(raw_text or "").strip()
    if not text:
        if expected_count is not None:
            raise ValueError(
                f"Expected {expected_count} background theta values, got 0."
            )
        return []

    tokens = [token for token in re.split(r"[\s,;]+", text) if token]
    values: list[float] = []
    for token in tokens:
        try:
            theta_val = float(token)
        except Exception as exc:
            raise ValueError(f"Invalid theta value '{token}'.") from exc
        if not np.isfinite(theta_val):
            raise ValueError(f"Non-finite theta value '{token}'.")
        values.append(float(theta_val))

    if expected_count is not None and len(values) != int(expected_count):
        raise ValueError(
            f"Expected {int(expected_count)} background theta values, got {len(values)}."
        )
    return values


def default_geometry_fit_background_selection(
    *,
    osc_files: Sequence[object],
) -> str:
    """Return the default geometry-fit background selector text."""

    try:
        return "all" if len(osc_files) > 1 else "current"
    except Exception:
        return "current"


def format_geometry_fit_background_indices(indices: Sequence[object]) -> str:
    """Format a background-index list using 1-based labels for the GUI entry."""

    labels: list[str] = []
    for raw_value in indices:
        try:
            idx = int(raw_value)
        except Exception:
            continue
        if idx < 0:
            continue
        labels.append(str(idx + 1))
    return ", ".join(labels)


def parse_geometry_fit_background_indices(
    raw_text: object,
    *,
    total_count: int,
    current_index: int = 0,
) -> list[int]:
    """Parse geometry-fit background selection text into 0-based indices."""

    count = max(0, int(total_count))
    if count <= 0:
        return []

    current_idx = max(0, min(int(current_index), count - 1))
    text = str(raw_text or "").strip().lower()
    if not text:
        return list(range(count)) if count > 1 else [current_idx]

    indices: list[int] = []
    saw_all = False
    tokens = [token for token in re.split(r"[\s,;]+", text) if token]
    for token in tokens:
        if token in {"all", "*"}:
            saw_all = True
            indices = list(range(count))
            continue
        if token in {"current", "cur", "active"}:
            indices.append(int(current_idx))
            continue

        range_match = re.fullmatch(r"(\d+)-(\d+)", token)
        if range_match:
            start_idx = int(range_match.group(1))
            stop_idx = int(range_match.group(2))
            if start_idx < 1 or stop_idx < 1:
                raise ValueError(f"Background ranges are 1-based; got '{token}'.")
            lo = max(1, min(start_idx, stop_idx))
            hi = min(count, max(start_idx, stop_idx))
            if lo > count:
                raise ValueError(
                    f"Background range '{token}' is outside the loaded background list."
                )
            indices.extend(idx - 1 for idx in range(lo, hi + 1))
            continue

        try:
            selected_idx = int(token)
        except Exception as exc:
            raise ValueError(
                "Geometry fit backgrounds must be 'current', 'all', "
                "a 1-based index, or a range like '1-3'."
            ) from exc
        if selected_idx < 1 or selected_idx > count:
            raise ValueError(
                f"Background index '{token}' is outside the loaded background list 1-{count}."
            )
        indices.append(selected_idx - 1)

    if saw_all:
        return list(range(count))

    unique_indices: list[int] = []
    seen: set[int] = set()
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        unique_indices.append(int(idx))
    if not unique_indices:
        raise ValueError("Select at least one background for geometry fitting.")
    return unique_indices


def current_geometry_fit_background_indices(
    *,
    osc_files: Sequence[object],
    current_background_index: int,
    geometry_fit_background_selection_var: object | None,
    strict: bool = False,
) -> list[int]:
    """Return the background indices currently selected for geometry fitting."""

    try:
        total_count = int(len(osc_files))
    except Exception:
        total_count = 0
    if total_count <= 0:
        return []

    default_indices = (
        list(range(total_count))
        if total_count > 1
        else [max(0, min(int(current_background_index), total_count - 1))]
    )
    if geometry_fit_background_selection_var is None:
        return default_indices

    raw_value = _read_var_value(geometry_fit_background_selection_var)
    try:
        indices = parse_geometry_fit_background_indices(
            raw_value,
            total_count=total_count,
            current_index=int(current_background_index),
        )
    except Exception as exc:
        if strict:
            raise ValueError(f"Invalid fit background selection '{raw_value}'.") from exc
        return default_indices
    return indices or default_indices


def geometry_fit_uses_shared_theta_offset(
    selected_indices: Sequence[int] | None = None,
    *,
    osc_files: Sequence[object] = (),
    current_background_index: int = 0,
    geometry_fit_background_selection_var: object | None = None,
) -> bool:
    """Return whether geometry fitting should use a shared theta offset."""

    if selected_indices is None:
        try:
            selected_indices = current_geometry_fit_background_indices(
                osc_files=osc_files,
                current_background_index=current_background_index,
                geometry_fit_background_selection_var=geometry_fit_background_selection_var,
                strict=False,
            )
        except Exception:
            selected_indices = []
    return len([int(idx) for idx in selected_indices]) > 1


def current_geometry_theta_offset(
    *,
    geometry_theta_offset_var: object | None,
    strict: bool = False,
) -> float:
    """Return the shared theta offset used by multi-background fitting."""

    if geometry_theta_offset_var is None:
        return 0.0
    raw_value = _read_var_value(geometry_theta_offset_var)
    try:
        theta_offset = float(raw_value)
    except Exception as exc:
        if strict:
            raise ValueError(f"Invalid shared theta offset '{raw_value}'.") from exc
        return 0.0
    if not np.isfinite(theta_offset):
        if strict:
            raise ValueError("Shared theta offset must be finite.")
        return 0.0
    return float(theta_offset)


def current_background_theta_values(
    *,
    osc_files: Sequence[object],
    theta_initial_var: object | None,
    defaults: Mapping[str, object] | None,
    theta_initial: object,
    background_theta_list_var: object | None,
    strict_count: bool = False,
) -> list[float]:
    """Return one configured theta value per loaded background image."""

    expected_count = int(len(osc_files))
    default_theta = background_theta_default_value(
        theta_initial_var=theta_initial_var,
        defaults=defaults,
        theta_initial=theta_initial,
    )
    raw_values: list[float] = []
    if background_theta_list_var is not None:
        try:
            raw_values = parse_background_theta_values(
                _read_var_value(background_theta_list_var),
                expected_count=expected_count if strict_count else None,
            )
        except Exception:
            if strict_count:
                raise
            raw_values = []

    if strict_count:
        if expected_count != len(raw_values):
            raise ValueError(
                f"Expected {expected_count} background theta values, got {len(raw_values)}."
            )
        return raw_values

    if expected_count <= 0:
        return []
    if not raw_values:
        return [float(default_theta)] * expected_count

    values = list(raw_values[:expected_count])
    if len(values) < expected_count:
        values.extend([float(default_theta)] * (expected_count - len(values)))
    return [float(v) for v in values]


def sync_live_theta_to_background_theta_list(
    *,
    osc_files: Sequence[object],
    current_background_index: int,
    theta_initial_var: object | None,
    defaults: Mapping[str, object] | None,
    theta_initial: object,
    background_theta_list_var: object | None,
) -> bool:
    """Mirror the live theta slider value into the current background entry."""

    if theta_initial_var is None or background_theta_list_var is None:
        return False

    expected_count = int(len(osc_files))
    if expected_count <= 0:
        return False

    try:
        live_theta = float(_read_var_value(theta_initial_var))
    except Exception:
        return False
    if not np.isfinite(live_theta):
        return False

    theta_values = current_background_theta_values(
        osc_files=osc_files,
        theta_initial_var=theta_initial_var,
        defaults=defaults,
        theta_initial=theta_initial,
        background_theta_list_var=background_theta_list_var,
        strict_count=False,
    )
    if not theta_values:
        return False

    idx = max(0, min(int(current_background_index), len(theta_values) - 1))
    if abs(float(theta_values[idx]) - float(live_theta)) <= 1.0e-12:
        return False

    theta_values[idx] = float(live_theta)
    _write_var_value(
        background_theta_list_var,
        format_background_theta_values(theta_values),
    )
    return True


def background_theta_base_for_index(
    index: int,
    *,
    osc_files: Sequence[object],
    theta_initial_var: object | None,
    defaults: Mapping[str, object] | None,
    theta_initial: object,
    background_theta_list_var: object | None,
    strict_count: bool = False,
) -> float:
    """Return the stored per-background theta_i value for one background index."""

    theta_values = current_background_theta_values(
        osc_files=osc_files,
        theta_initial_var=theta_initial_var,
        defaults=defaults,
        theta_initial=theta_initial,
        background_theta_list_var=background_theta_list_var,
        strict_count=strict_count,
    )
    if not theta_values:
        return background_theta_default_value(
            theta_initial_var=theta_initial_var,
            defaults=defaults,
            theta_initial=theta_initial,
        )
    idx = max(0, min(int(index), len(theta_values) - 1))
    return float(theta_values[idx])


def background_theta_for_index(
    index: int,
    *,
    osc_files: Sequence[object],
    theta_initial_var: object | None,
    defaults: Mapping[str, object] | None,
    theta_initial: object,
    background_theta_list_var: object | None,
    geometry_theta_offset_var: object | None,
    geometry_fit_background_selection_var: object | None,
    current_background_index: int,
    strict_count: bool = False,
) -> float:
    """Return the effective theta used for one background index."""

    theta_values = current_background_theta_values(
        osc_files=osc_files,
        theta_initial_var=theta_initial_var,
        defaults=defaults,
        theta_initial=theta_initial,
        background_theta_list_var=background_theta_list_var,
        strict_count=strict_count,
    )
    if not theta_values:
        return background_theta_default_value(
            theta_initial_var=theta_initial_var,
            defaults=defaults,
            theta_initial=theta_initial,
        )
    idx = max(0, min(int(index), len(theta_values) - 1))
    theta_val = float(theta_values[idx])
    if geometry_fit_uses_shared_theta_offset(
        osc_files=osc_files,
        current_background_index=current_background_index,
        geometry_fit_background_selection_var=geometry_fit_background_selection_var,
    ):
        theta_val += current_geometry_theta_offset(
            geometry_theta_offset_var=geometry_theta_offset_var,
            strict=strict_count,
        )
    return float(theta_val)


def refresh_geometry_fit_theta_checkbox_label(
    *,
    fit_theta_checkbutton: object | None,
    theta_controls: object,
    shared_theta: bool,
) -> None:
    """Update the theta fit toggle label for single vs multi-background mode."""

    if fit_theta_checkbutton is not None:
        config = getattr(fit_theta_checkbutton, "config", None)
        if callable(config):
            config(text="θ shared offset" if shared_theta else "θ sample tilt")
    if not isinstance(theta_controls, dict):
        theta_controls = {}
    theta_control = theta_controls.get("theta_initial", {})
    if isinstance(theta_control, dict):
        row = theta_control.get("row")
        if row is not None:
            configure = getattr(row, "configure", None)
            if callable(configure):
                configure(
                    text="Theta Shared Offset" if shared_theta else "Theta Sample Tilt"
                )


def sync_background_theta_controls(
    *,
    osc_files: Sequence[object],
    current_background_index: int,
    theta_initial_var: object | None,
    defaults: Mapping[str, object] | None,
    theta_initial: object,
    background_theta_list_var: object | None,
    geometry_theta_offset_var: object | None,
    geometry_fit_background_selection_var: object | None,
    fit_theta_checkbutton: object | None,
    theta_controls: object,
    set_background_file_status_text: Callable[[], None] | None,
    schedule_update: Callable[[], None] | None,
    preserve_existing: bool = True,
    trigger_update: bool = False,
) -> None:
    """Keep the theta list entry aligned with the currently loaded backgrounds."""

    expected_count = int(len(osc_files))
    default_theta = background_theta_default_value(
        theta_initial_var=theta_initial_var,
        defaults=defaults,
        theta_initial=theta_initial,
    )
    current_values: list[float] = []
    if preserve_existing:
        current_values = current_background_theta_values(
            osc_files=osc_files,
            theta_initial_var=theta_initial_var,
            defaults=defaults,
            theta_initial=theta_initial,
            background_theta_list_var=background_theta_list_var,
            strict_count=False,
        )
    if expected_count > 0:
        current_values = list(current_values[:expected_count])
        if len(current_values) < expected_count:
            current_values.extend(
                [float(default_theta)] * (expected_count - len(current_values))
            )
    else:
        current_values = []

    if background_theta_list_var is not None:
        _write_var_value(
            background_theta_list_var,
            format_background_theta_values(current_values),
        )

    refresh_geometry_fit_theta_checkbox_label(
        fit_theta_checkbutton=fit_theta_checkbutton,
        theta_controls=theta_controls,
        shared_theta=geometry_fit_uses_shared_theta_offset(
            osc_files=osc_files,
            current_background_index=current_background_index,
            geometry_fit_background_selection_var=geometry_fit_background_selection_var,
        ),
    )
    if set_background_file_status_text is not None:
        set_background_file_status_text()

    if theta_initial_var is not None and expected_count > 0:
        try:
            _write_var_value(
                theta_initial_var,
                background_theta_base_for_index(
                    current_background_index,
                    osc_files=osc_files,
                    theta_initial_var=theta_initial_var,
                    defaults=defaults,
                    theta_initial=theta_initial,
                    background_theta_list_var=background_theta_list_var,
                    strict_count=False,
                ),
            )
        except Exception:
            _write_var_value(theta_initial_var, float(default_theta))

    if trigger_update and schedule_update is not None:
        schedule_update()


def apply_background_theta_metadata(
    *,
    osc_files: Sequence[object],
    current_background_index: int,
    theta_initial_var: object | None,
    defaults: Mapping[str, object] | None,
    theta_initial: object,
    background_theta_list_var: object | None,
    geometry_theta_offset_var: object | None,
    geometry_fit_background_selection_var: object | None,
    fit_theta_checkbutton: object | None,
    theta_controls: object,
    set_background_file_status_text: Callable[[], None] | None,
    schedule_update: Callable[[], None] | None,
    progress_label: object | None = None,
    trigger_update: bool = True,
    sync_live_theta: bool = True,
) -> bool:
    """Validate the theta list/offset entries and optionally refresh the display."""

    try:
        theta_values = current_background_theta_values(
            osc_files=osc_files,
            theta_initial_var=theta_initial_var,
            defaults=defaults,
            theta_initial=theta_initial,
            background_theta_list_var=background_theta_list_var,
            strict_count=True,
        )
        _ = current_geometry_theta_offset(
            geometry_theta_offset_var=geometry_theta_offset_var,
            strict=True,
        )
    except Exception as exc:
        _set_widget_text(progress_label, f"Invalid background theta settings: {exc}")
        return False

    if background_theta_list_var is not None:
        _write_var_value(
            background_theta_list_var,
            format_background_theta_values(theta_values),
        )

    refresh_geometry_fit_theta_checkbox_label(
        fit_theta_checkbutton=fit_theta_checkbutton,
        theta_controls=theta_controls,
        shared_theta=geometry_fit_uses_shared_theta_offset(
            osc_files=osc_files,
            current_background_index=current_background_index,
            geometry_fit_background_selection_var=geometry_fit_background_selection_var,
        ),
    )
    if set_background_file_status_text is not None:
        set_background_file_status_text()
    if sync_live_theta and theta_initial_var is not None and theta_values:
        _write_var_value(
            theta_initial_var,
            background_theta_base_for_index(
                current_background_index,
                osc_files=osc_files,
                theta_initial_var=theta_initial_var,
                defaults=defaults,
                theta_initial=theta_initial,
                background_theta_list_var=background_theta_list_var,
                strict_count=True,
            ),
        )
    if trigger_update and schedule_update is not None:
        schedule_update()
    return True


def apply_geometry_fit_background_selection(
    *,
    osc_files: Sequence[object],
    current_background_index: int,
    theta_initial_var: object | None,
    defaults: Mapping[str, object] | None,
    theta_initial: object,
    background_theta_list_var: object | None,
    geometry_theta_offset_var: object | None,
    geometry_fit_background_selection_var: object | None,
    fit_theta_checkbutton: object | None,
    theta_controls: object,
    set_background_file_status_text: Callable[[], None] | None,
    schedule_update: Callable[[], None] | None,
    progress_label_geometry: object | None = None,
    trigger_update: bool = False,
    sync_live_theta: bool = True,
) -> bool:
    """Validate the geometry-fit background selection entry."""

    if geometry_fit_background_selection_var is None:
        return True

    live_theta_before = None
    effective_theta_before = None
    if sync_live_theta and theta_initial_var is not None:
        try:
            live_theta_before = float(_read_var_value(theta_initial_var))
        except Exception:
            live_theta_before = None
        try:
            effective_theta_before = float(
                background_theta_for_index(
                    current_background_index,
                    osc_files=osc_files,
                    theta_initial_var=theta_initial_var,
                    defaults=defaults,
                    theta_initial=theta_initial,
                    background_theta_list_var=background_theta_list_var,
                    geometry_theta_offset_var=geometry_theta_offset_var,
                    geometry_fit_background_selection_var=geometry_fit_background_selection_var,
                    current_background_index=current_background_index,
                    strict_count=False,
                )
            )
        except Exception:
            effective_theta_before = live_theta_before

    try:
        selected_indices = current_geometry_fit_background_indices(
            osc_files=osc_files,
            current_background_index=current_background_index,
            geometry_fit_background_selection_var=geometry_fit_background_selection_var,
            strict=True,
        )
    except Exception as exc:
        _set_widget_text(
            progress_label_geometry,
            f"Invalid geometry fit background selection: {exc}",
        )
        return False

    previous_shared_theta = None
    if isinstance(theta_controls, dict):
        previous_shared_theta = theta_controls.get("_shared_theta_active")

    total_count = len(osc_files)
    if total_count > 1 and len(selected_indices) == total_count:
        _write_var_value(geometry_fit_background_selection_var, "all")
    elif (
        len(selected_indices) == 1
        and int(selected_indices[0]) == int(current_background_index)
    ):
        _write_var_value(geometry_fit_background_selection_var, "current")
    else:
        _write_var_value(
            geometry_fit_background_selection_var,
            format_geometry_fit_background_indices(selected_indices),
        )

    shared_theta = geometry_fit_uses_shared_theta_offset(selected_indices)
    refresh_geometry_fit_theta_checkbox_label(
        fit_theta_checkbutton=fit_theta_checkbutton,
        theta_controls=theta_controls,
        shared_theta=shared_theta,
    )
    if isinstance(theta_controls, dict):
        theta_controls["_shared_theta_active"] = bool(shared_theta)
    if set_background_file_status_text is not None:
        set_background_file_status_text()

    theta_changed = False
    if sync_live_theta and theta_initial_var is not None:
        try:
            live_theta_after = float(
                background_theta_base_for_index(
                    current_background_index,
                    osc_files=osc_files,
                    theta_initial_var=theta_initial_var,
                    defaults=defaults,
                    theta_initial=theta_initial,
                    background_theta_list_var=background_theta_list_var,
                    strict_count=False,
                )
            )
            effective_theta_after = float(
                background_theta_for_index(
                    current_background_index,
                    osc_files=osc_files,
                    theta_initial_var=theta_initial_var,
                    defaults=defaults,
                    theta_initial=theta_initial,
                    background_theta_list_var=background_theta_list_var,
                    geometry_theta_offset_var=geometry_theta_offset_var,
                    geometry_fit_background_selection_var=geometry_fit_background_selection_var,
                    current_background_index=current_background_index,
                    strict_count=False,
                )
            )
            _write_var_value(theta_initial_var, live_theta_after)
            if (
                live_theta_before is None
                or not np.isfinite(live_theta_before)
                or abs(float(live_theta_after) - float(live_theta_before)) > 1.0e-12
            ):
                theta_changed = True
            elif (
                previous_shared_theta is not None
                and bool(previous_shared_theta) != bool(shared_theta)
            ):
                theta_changed = True
            elif (
                effective_theta_before is None
                or not np.isfinite(effective_theta_before)
                or abs(
                    float(effective_theta_after) - float(effective_theta_before)
                )
                > 1.0e-12
            ):
                theta_changed = True
        except Exception:
            theta_changed = bool(trigger_update)

    if trigger_update and theta_changed and schedule_update is not None:
        schedule_update()
    return True


def sync_geometry_fit_background_selection(
    *,
    osc_files: Sequence[object],
    current_background_index: int,
    theta_initial_var: object | None,
    defaults: Mapping[str, object] | None,
    theta_initial: object,
    background_theta_list_var: object | None,
    geometry_theta_offset_var: object | None,
    geometry_fit_background_selection_var: object | None,
    fit_theta_checkbutton: object | None,
    theta_controls: object,
    set_background_file_status_text: Callable[[], None] | None,
    schedule_update: Callable[[], None] | None,
    progress_label_geometry: object | None = None,
    preserve_existing: bool = True,
) -> None:
    """Keep the fit-background selector valid when the background list changes."""

    if geometry_fit_background_selection_var is None:
        return

    if preserve_existing:
        if apply_geometry_fit_background_selection(
            osc_files=osc_files,
            current_background_index=current_background_index,
            theta_initial_var=theta_initial_var,
            defaults=defaults,
            theta_initial=theta_initial,
            background_theta_list_var=background_theta_list_var,
            geometry_theta_offset_var=geometry_theta_offset_var,
            geometry_fit_background_selection_var=geometry_fit_background_selection_var,
            fit_theta_checkbutton=fit_theta_checkbutton,
            theta_controls=theta_controls,
            set_background_file_status_text=set_background_file_status_text,
            schedule_update=schedule_update,
            progress_label_geometry=progress_label_geometry,
            trigger_update=False,
        ):
            return
    _write_var_value(
        geometry_fit_background_selection_var,
        default_geometry_fit_background_selection(osc_files=osc_files),
    )
    apply_geometry_fit_background_selection(
        osc_files=osc_files,
        current_background_index=current_background_index,
        theta_initial_var=theta_initial_var,
        defaults=defaults,
        theta_initial=theta_initial,
        background_theta_list_var=background_theta_list_var,
        geometry_theta_offset_var=geometry_theta_offset_var,
        geometry_fit_background_selection_var=geometry_fit_background_selection_var,
        fit_theta_checkbutton=fit_theta_checkbutton,
        theta_controls=theta_controls,
        set_background_file_status_text=set_background_file_status_text,
        schedule_update=schedule_update,
        progress_label_geometry=progress_label_geometry,
        trigger_update=False,
    )


def make_runtime_background_theta_bindings_factory(
    *,
    osc_files_factory: object,
    current_background_index_factory: object,
    theta_initial_var_factory: object,
    defaults: Mapping[str, object] | None,
    theta_initial: object,
    background_theta_list_var_factory: object,
    geometry_theta_offset_var_factory: object,
    geometry_fit_background_selection_var_factory: object,
    fit_theta_checkbutton_factory: object = None,
    theta_controls_factory: object = None,
    set_background_file_status_text_factory: object = None,
    schedule_update_factory: object = None,
    progress_label_factory: object = None,
    progress_label_geometry_factory: object = None,
) -> Callable[[], BackgroundThetaRuntimeBindings]:
    """Return a zero-arg factory for live background-theta bindings."""

    def _build() -> BackgroundThetaRuntimeBindings:
        osc_files_value = _resolve_runtime_value(osc_files_factory)
        try:
            osc_files = tuple(osc_files_value or ())
        except Exception:
            osc_files = ()
        current_background_index_value = _resolve_runtime_value(
            current_background_index_factory
        )
        try:
            current_background_index = int(current_background_index_value)
        except Exception:
            current_background_index = 0
        return BackgroundThetaRuntimeBindings(
            osc_files=osc_files,
            current_background_index=current_background_index,
            theta_initial_var=_resolve_runtime_value(theta_initial_var_factory),
            defaults=defaults,
            theta_initial=theta_initial,
            background_theta_list_var=_resolve_runtime_value(
                background_theta_list_var_factory
            ),
            geometry_theta_offset_var=_resolve_runtime_value(
                geometry_theta_offset_var_factory
            ),
            geometry_fit_background_selection_var=_resolve_runtime_value(
                geometry_fit_background_selection_var_factory
            ),
            fit_theta_checkbutton=_resolve_runtime_value(fit_theta_checkbutton_factory),
            theta_controls=_resolve_runtime_value(theta_controls_factory),
            set_background_file_status_text=_resolve_runtime_value(
                set_background_file_status_text_factory
            ),
            schedule_update=_resolve_runtime_value(schedule_update_factory),
            progress_label=_resolve_runtime_value(progress_label_factory),
            progress_label_geometry=_resolve_runtime_value(
                progress_label_geometry_factory
            ),
        )

    return _build


def runtime_current_geometry_fit_background_indices(
    bindings: BackgroundThetaRuntimeBindings,
    *,
    strict: bool = False,
) -> list[int]:
    """Return the current fit-background selection from live runtime bindings."""

    return current_geometry_fit_background_indices(
        osc_files=bindings.osc_files,
        current_background_index=bindings.current_background_index,
        geometry_fit_background_selection_var=bindings.geometry_fit_background_selection_var,
        strict=bool(strict),
    )


def runtime_geometry_fit_uses_shared_theta_offset(
    bindings: BackgroundThetaRuntimeBindings,
    selected_indices: Sequence[int] | None = None,
) -> bool:
    """Return whether one live fit-background selection uses a shared offset."""

    return geometry_fit_uses_shared_theta_offset(
        selected_indices,
        osc_files=bindings.osc_files,
        current_background_index=bindings.current_background_index,
        geometry_fit_background_selection_var=bindings.geometry_fit_background_selection_var,
    )


def runtime_current_geometry_theta_offset(
    bindings: BackgroundThetaRuntimeBindings,
    *,
    strict: bool = False,
) -> float:
    """Return the live shared theta offset from runtime bindings."""

    return current_geometry_theta_offset(
        geometry_theta_offset_var=bindings.geometry_theta_offset_var,
        strict=bool(strict),
    )


def runtime_current_background_theta_values(
    bindings: BackgroundThetaRuntimeBindings,
    *,
    strict_count: bool = False,
) -> list[float]:
    """Return the live per-background theta list from runtime bindings."""

    return current_background_theta_values(
        osc_files=bindings.osc_files,
        theta_initial_var=bindings.theta_initial_var,
        defaults=bindings.defaults,
        theta_initial=bindings.theta_initial,
        background_theta_list_var=bindings.background_theta_list_var,
        strict_count=bool(strict_count),
    )


def runtime_background_theta_for_index(
    bindings: BackgroundThetaRuntimeBindings,
    index: int,
    *,
    strict_count: bool = False,
) -> float:
    """Return the live effective theta for one background index."""

    return background_theta_for_index(
        int(index),
        osc_files=bindings.osc_files,
        theta_initial_var=bindings.theta_initial_var,
        defaults=bindings.defaults,
        theta_initial=bindings.theta_initial,
        background_theta_list_var=bindings.background_theta_list_var,
        geometry_theta_offset_var=bindings.geometry_theta_offset_var,
        geometry_fit_background_selection_var=bindings.geometry_fit_background_selection_var,
        current_background_index=bindings.current_background_index,
        strict_count=bool(strict_count),
    )


def runtime_sync_background_theta_controls(
    bindings: BackgroundThetaRuntimeBindings,
    *,
    preserve_existing: bool = True,
    trigger_update: bool = False,
) -> None:
    """Sync the live background-theta controls from runtime bindings."""

    sync_background_theta_controls(
        osc_files=bindings.osc_files,
        current_background_index=bindings.current_background_index,
        theta_initial_var=bindings.theta_initial_var,
        defaults=bindings.defaults,
        theta_initial=bindings.theta_initial,
        background_theta_list_var=bindings.background_theta_list_var,
        geometry_theta_offset_var=bindings.geometry_theta_offset_var,
        geometry_fit_background_selection_var=bindings.geometry_fit_background_selection_var,
        fit_theta_checkbutton=bindings.fit_theta_checkbutton,
        theta_controls=bindings.theta_controls,
        set_background_file_status_text=bindings.set_background_file_status_text,
        schedule_update=bindings.schedule_update,
        preserve_existing=bool(preserve_existing),
        trigger_update=bool(trigger_update),
    )


def runtime_apply_background_theta_metadata(
    bindings: BackgroundThetaRuntimeBindings,
    *,
    trigger_update: bool = True,
    sync_live_theta: bool = True,
) -> bool:
    """Apply live background-theta metadata from runtime bindings."""

    return apply_background_theta_metadata(
        osc_files=bindings.osc_files,
        current_background_index=bindings.current_background_index,
        theta_initial_var=bindings.theta_initial_var,
        defaults=bindings.defaults,
        theta_initial=bindings.theta_initial,
        background_theta_list_var=bindings.background_theta_list_var,
        geometry_theta_offset_var=bindings.geometry_theta_offset_var,
        geometry_fit_background_selection_var=bindings.geometry_fit_background_selection_var,
        fit_theta_checkbutton=bindings.fit_theta_checkbutton,
        theta_controls=bindings.theta_controls,
        set_background_file_status_text=bindings.set_background_file_status_text,
        schedule_update=bindings.schedule_update,
        progress_label=bindings.progress_label,
        trigger_update=bool(trigger_update),
        sync_live_theta=bool(sync_live_theta),
    )


def runtime_apply_geometry_fit_background_selection(
    bindings: BackgroundThetaRuntimeBindings,
    *,
    trigger_update: bool = False,
    sync_live_theta: bool = True,
) -> bool:
    """Apply the live geometry-fit background selection from runtime bindings."""

    return apply_geometry_fit_background_selection(
        osc_files=bindings.osc_files,
        current_background_index=bindings.current_background_index,
        theta_initial_var=bindings.theta_initial_var,
        defaults=bindings.defaults,
        theta_initial=bindings.theta_initial,
        background_theta_list_var=bindings.background_theta_list_var,
        geometry_theta_offset_var=bindings.geometry_theta_offset_var,
        geometry_fit_background_selection_var=bindings.geometry_fit_background_selection_var,
        fit_theta_checkbutton=bindings.fit_theta_checkbutton,
        theta_controls=bindings.theta_controls,
        set_background_file_status_text=bindings.set_background_file_status_text,
        schedule_update=bindings.schedule_update,
        progress_label_geometry=bindings.progress_label_geometry,
        trigger_update=bool(trigger_update),
        sync_live_theta=bool(sync_live_theta),
    )


def runtime_sync_geometry_fit_background_selection(
    bindings: BackgroundThetaRuntimeBindings,
    *,
    preserve_existing: bool = True,
) -> None:
    """Sync the live fit-background selector from runtime bindings."""

    sync_geometry_fit_background_selection(
        osc_files=bindings.osc_files,
        current_background_index=bindings.current_background_index,
        theta_initial_var=bindings.theta_initial_var,
        defaults=bindings.defaults,
        theta_initial=bindings.theta_initial,
        background_theta_list_var=bindings.background_theta_list_var,
        geometry_theta_offset_var=bindings.geometry_theta_offset_var,
        geometry_fit_background_selection_var=bindings.geometry_fit_background_selection_var,
        fit_theta_checkbutton=bindings.fit_theta_checkbutton,
        theta_controls=bindings.theta_controls,
        set_background_file_status_text=bindings.set_background_file_status_text,
        schedule_update=bindings.schedule_update,
        progress_label_geometry=bindings.progress_label_geometry,
        preserve_existing=bool(preserve_existing),
    )


def make_runtime_background_theta_callbacks(
    bindings_factory: Callable[[], BackgroundThetaRuntimeBindings],
) -> BackgroundThetaRuntimeCallbacks:
    """Return bound callbacks for live background-theta runtime workflows."""

    return BackgroundThetaRuntimeCallbacks(
        current_geometry_fit_background_indices=(
            lambda *, strict=False: runtime_current_geometry_fit_background_indices(
                bindings_factory(),
                strict=bool(strict),
            )
        ),
        geometry_fit_uses_shared_theta_offset=(
            lambda selected_indices=None: runtime_geometry_fit_uses_shared_theta_offset(
                bindings_factory(),
                selected_indices=selected_indices,
            )
        ),
        current_geometry_theta_offset=(
            lambda *, strict=False: runtime_current_geometry_theta_offset(
                bindings_factory(),
                strict=bool(strict),
            )
        ),
        current_background_theta_values=(
            lambda *, strict_count=False: runtime_current_background_theta_values(
                bindings_factory(),
                strict_count=bool(strict_count),
            )
        ),
        background_theta_for_index=(
            lambda index, *, strict_count=False: runtime_background_theta_for_index(
                bindings_factory(),
                index,
                strict_count=bool(strict_count),
            )
        ),
        sync_background_theta_controls=(
            lambda *, preserve_existing=True, trigger_update=False: (
                runtime_sync_background_theta_controls(
                    bindings_factory(),
                    preserve_existing=bool(preserve_existing),
                    trigger_update=bool(trigger_update),
                )
            )
        ),
        apply_background_theta_metadata=(
            lambda *, trigger_update=True, sync_live_theta=True: (
                runtime_apply_background_theta_metadata(
                    bindings_factory(),
                    trigger_update=bool(trigger_update),
                    sync_live_theta=bool(sync_live_theta),
                )
            )
        ),
        apply_geometry_fit_background_selection=(
            lambda *, trigger_update=False, sync_live_theta=True: (
                runtime_apply_geometry_fit_background_selection(
                    bindings_factory(),
                    trigger_update=bool(trigger_update),
                    sync_live_theta=bool(sync_live_theta),
                )
            )
        ),
        sync_geometry_fit_background_selection=(
            lambda *, preserve_existing=True: (
                runtime_sync_geometry_fit_background_selection(
                    bindings_factory(),
                    preserve_existing=bool(preserve_existing),
                )
            )
        ),
    )
