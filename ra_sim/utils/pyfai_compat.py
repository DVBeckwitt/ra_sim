"""Compatibility helpers that smooth over pyFAI API differences."""

from __future__ import annotations

import inspect
import math
from typing import Any, Optional, Tuple

import numpy as np


def _function_accepts_parameter(func, parameter: str) -> bool:
    """Return True if *func* accepts a keyword named *parameter*."""

    try:
        return parameter in inspect.signature(func).parameters
    except (TypeError, ValueError):  # pragma: no cover - Cython functions
        # Built-in or Cython functions may not expose signatures; assume support.
        return True


def integrate2d_with_compat(
    ai,
    data,
    *,
    npt_rad: int,
    npt_azim: int,
    correct_solid_angle: bool = True,
    method: str = "lut",
    unit: Optional[str] = "2th_deg",
):
    """Call :meth:`AzimuthalIntegrator.integrate2d` with graceful fallback."""

    kwargs: dict[str, Any] = {
        "npt_rad": npt_rad,
        "npt_azim": npt_azim,
        "correctSolidAngle": correct_solid_angle,
        "method": method,
    }
    if unit and _function_accepts_parameter(ai.integrate2d, "unit"):
        kwargs["unit"] = unit
    try:
        return ai.integrate2d(data, **kwargs)
    except TypeError as exc:
        if "unit" in kwargs and "unit" in str(exc):
            kwargs.pop("unit", None)
            return ai.integrate2d(data, **kwargs)
        raise


def _to_unit_string(unit: Any) -> Optional[str]:
    if unit is None:
        return None
    if isinstance(unit, dict):
        return None
    if isinstance(unit, (list, tuple)) and unit:
        unit = unit[0]
    for attr in ("value", "name"):
        if hasattr(unit, attr):
            try:
                unit = getattr(unit, attr)
                break
            except Exception:  # pragma: no cover - defensive
                pass
    try:
        return str(unit).lower()
    except Exception:  # pragma: no cover - defensive
        return None


def extract_axis_unit(result, axis: str) -> Any:
    axis = axis.lower()
    attr_candidates = [
        f"{axis}_unit",
        f"{axis}Unit",
        f"unit_{axis}",
        f"{axis}UnitName",
    ]
    for attr in attr_candidates:
        if hasattr(result, attr):
            unit = getattr(result, attr)
            if unit is not None:
                return unit
    unit_attr = getattr(result, "unit", None)
    if isinstance(unit_attr, dict):
        return unit_attr.get(axis)
    if hasattr(unit_attr, axis):
        return getattr(unit_attr, axis)
    return unit_attr


def convert_axis_to_degrees(
    values: Any,
    unit: Any,
    wavelength_m: Optional[float] = None,
) -> np.ndarray:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    unit_str = _to_unit_string(unit)
    if not unit_str:
        return arr
    if "deg" in unit_str:
        return arr
    if "rad" in unit_str:
        return np.degrees(arr)
    if unit_str.startswith("q") and wavelength_m:
        wavelength_angstrom = wavelength_m * 1e10
        sin_theta = arr * wavelength_angstrom / (4.0 * math.pi)
        sin_theta = np.clip(sin_theta, -1.0, 1.0)
        return np.degrees(2.0 * np.arcsin(sin_theta))
    return arr


def ensure_axes_in_degrees(
    result,
    wavelength_m: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    radial_unit = extract_axis_unit(result, "radial")
    az_unit = extract_axis_unit(result, "azimuthal")
    radial = convert_axis_to_degrees(result.radial, radial_unit, wavelength_m)
    azimuthal = convert_axis_to_degrees(result.azimuthal, az_unit)
    return radial, azimuthal
