"""Run geometry fitting from a saved GUI state without launching Tk."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import importlib.util
import json
import math
import os
import time
import warnings
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from ra_sim.config import get_dir, get_instrument_config, get_path
from ra_sim.io.file_parsing import parse_poni_file
from ra_sim.io.osc_reader import read_osc
from ra_sim.simulation.intersection_cache_schema import extract_hit_row_provenance

if TYPE_CHECKING:
    from ra_sim.gui.state import (
        AtomSiteOverrideState,
        BackgroundRuntimeState,
        SimulationRuntimeState,
    )


DISPLAY_ROTATE_K = -1
SIM_DISPLAY_ROTATE_K = 0
HEADLESS_GEOMETRY_CAKED_RADIAL_BINS = 1000
HEADLESS_GEOMETRY_CAKED_AZIMUTH_BINS = 720
HEADLESS_GEOMETRY_SUPPORTED_ACTIVE_VAR_NAMES = (
    "zb",
    "zs",
    "theta_initial",
    "theta_offset",
    "psi_z",
    "chi",
    "cor_angle",
    "gamma",
    "Gamma",
    "corto_detector",
    "a",
    "c",
    "center_x",
    "center_y",
)
HEADLESS_GEOMETRY_SUPPORTED_ACTIVE_VAR_NAME_SET = set(
    HEADLESS_GEOMETRY_SUPPORTED_ACTIVE_VAR_NAMES
)
GEOMETRY_FIT_RECOVERY_SINGLE_STEP_JSON = "01_single_step_qr_coordinate_audit.json"
GEOMETRY_FIT_RECOVERY_SINGLE_STEP_PNG = "01_single_step_qr_coordinate_audit.png"
GEOMETRY_FIT_RECOVERY_SINGLE_STEP_CSV = "01_single_step_qr_coordinate_audit.csv"
GEOMETRY_FIT_RECOVERY_FULL_OVERLAY_JSON = "02_full_fit_initial_vs_final_qr_overlay.json"
GEOMETRY_FIT_RECOVERY_FULL_OVERLAY_PNG = "02_full_fit_initial_vs_final_qr_overlay.png"
GEOMETRY_FIT_RECOVERY_WORST_ROWS_JSON = "03_worst_residual_rows.json"
GEOMETRY_FIT_RECOVERY_WORST_ROWS_PNG = "03_worst_residual_rows.png"
_HEADLESS_GEOMETRY_FIT_SAVED_MANUAL_CAKED_DEFAULT_ACTIVE_VAR_NAMES = (
    "a",
    "theta_offset",
    "psi_z",
)


@dataclass(frozen=True)
class HeadlessGeometryFitResult:
    """Result metadata for one saved-state geometry fit."""

    state: dict[str, object]
    log_path: Path
    accepted: bool
    rejection_reason: str | None = None
    rms_px: float | None = None


@dataclass(frozen=True)
class _RuntimeDefaults:
    primary_cif_path: str
    secondary_cif_path: str | None
    osc_files: list[str]
    current_background_index: int
    image_size: int
    pixel_size_m: float
    lambda_angstrom: float
    psi_deg: float
    defaults: dict[str, object]
    fit_config: dict[str, object]
    intensity_threshold: float
    include_rods_flag: bool
    two_theta_range: tuple[float, float]
    mx: int
    background_flags: dict[str, object]


@dataclass(frozen=True)
class _LegacyBackgroundSubtractionConfig:
    enabled: bool = False
    mode: str = "off"
    apply_to_fit: bool = False
    diagnostics: bool = False
    scale: float = 1.0


class _HeadlessVar:
    """Minimal Tk-var shim used by the geometry-fit runtime helpers."""

    def __init__(self, value: object) -> None:
        self._value = value

    def get(self) -> object:
        return self._value

    def set(self, value: object) -> None:
        self._value = value


def normalize_headless_geometry_fit_active_var_names(
    active_var_names: Sequence[object] | str | None,
) -> list[str] | None:
    """Normalize one optional ordered active-variable override for headless fits."""

    if active_var_names is None:
        return None
    if isinstance(active_var_names, str):
        if not active_var_names.strip():
            raise ValueError("Geometry fit active-vars override cannot be empty.")
        raw_names = active_var_names.split(",")
    else:
        raw_names = list(active_var_names)
        if not raw_names:
            raise ValueError("Geometry fit active-vars override cannot be empty.")

    normalized_names: list[str] = []
    seen_names: set[str] = set()
    for raw_name in raw_names:
        name = str(raw_name).strip()
        if not name:
            raise ValueError("Geometry fit active-vars override contains an empty name.")
        if name not in HEADLESS_GEOMETRY_SUPPORTED_ACTIVE_VAR_NAME_SET:
            supported = ", ".join(HEADLESS_GEOMETRY_SUPPORTED_ACTIVE_VAR_NAMES)
            raise ValueError(
                f"Unknown geometry fit active var '{name}'. Supported names: {supported}."
            )
        if name in seen_names:
            raise ValueError(f"Duplicate geometry fit active var '{name}'.")
        seen_names.add(name)
        normalized_names.append(name)

    if {"theta_initial", "theta_offset"}.issubset(seen_names):
        raise ValueError(
            "Geometry fit active-vars override cannot include both 'theta_initial' and "
            "'theta_offset'."
        )
    return normalized_names


def _canonicalize_headless_geometry_fit_active_var_names(
    active_var_names: Sequence[str] | None,
    *,
    use_shared_theta_offset: bool,
) -> list[str] | None:
    """Map optional headless override names onto the runtime active-variable contract."""

    if active_var_names is None:
        return None
    canonical_names: list[str] = []
    seen_names: set[str] = set()
    for raw_name in active_var_names:
        name = gui_geometry_fit.geometry_fit_constraint_parameter_name(
            str(raw_name),
            use_shared_theta_offset=use_shared_theta_offset,
        )
        if name in seen_names:
            raise ValueError(
                f"Geometry fit active-vars override resolves to duplicate runtime var '{name}'."
            )
        seen_names.add(name)
        canonical_names.append(name)
    return canonical_names


def _headless_geometry_fit_bounds_section(
    fit_config: Mapping[str, object] | None,
) -> Mapping[str, object]:
    """Return one normalized headless geometry-fit bounds mapping."""

    if not isinstance(fit_config, Mapping):
        return {}
    fit_geometry_cfg = fit_config.get("geometry", {})
    if "geometry" in fit_config and isinstance(fit_geometry_cfg, Mapping):
        container_cfg = fit_geometry_cfg
    else:
        container_cfg = fit_config
    bounds_cfg = container_cfg.get("bounds", {}) or {}
    if not isinstance(bounds_cfg, Mapping):
        return {}
    return bounds_cfg


def _headless_geometry_fit_domain_from_bounds_entry(
    entry: object,
    *,
    current_value: object,
) -> tuple[float, float] | None:
    """Convert one geometry-fit bounds entry into a finite absolute domain."""

    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        try:
            lo = float(entry[0])
            hi = float(entry[1])
        except Exception:
            return None
    elif isinstance(entry, Mapping):
        mode = str(entry.get("mode", "absolute")).strip().lower()
        try:
            current_value_float = float(current_value)
        except Exception:
            current_value_float = float("nan")
        min_raw = entry.get("min")
        max_raw = entry.get("max")
        if mode in {"relative", "rel"}:
            if not np.isfinite(current_value_float):
                return None
            lo = (
                current_value_float + float(min_raw)
                if min_raw is not None
                else float("-inf")
            )
            hi = (
                current_value_float + float(max_raw)
                if max_raw is not None
                else float("inf")
            )
        elif mode in {"relative_min0", "rel_min0"}:
            if not np.isfinite(current_value_float):
                return None
            lo = (
                current_value_float + float(min_raw)
                if min_raw is not None
                else float("-inf")
            )
            if np.isfinite(lo):
                lo = max(0.0, lo)
            hi = (
                current_value_float + float(max_raw)
                if max_raw is not None
                else float("inf")
            )
        else:
            lo = float(min_raw) if min_raw is not None else float("-inf")
            hi = float(max_raw) if max_raw is not None else float("inf")
    else:
        return None

    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo > hi:
        lo, hi = hi, lo
    return float(lo), float(hi)


def _headless_runtime_geometry_fit_parameter_domains(
    *,
    fit_config: Mapping[str, object] | None,
    current_params: Mapping[str, object],
    image_size: object,
    names: Sequence[str],
    use_shared_theta_offset: bool,
) -> dict[str, tuple[float, float]]:
    """Build headless geometry-fit parameter domains for runtime-config shaping."""

    bounds_cfg = _headless_geometry_fit_bounds_section(fit_config)
    try:
        image_size_value = float(image_size)
    except Exception:
        image_size_value = 0.0

    domains: dict[str, tuple[float, float]] = {}
    for raw_name in names:
        name = str(raw_name)
        parameter_name = gui_geometry_fit.geometry_fit_constraint_parameter_name(
            name,
            use_shared_theta_offset=use_shared_theta_offset,
        )
        if parameter_name == "center_x" or parameter_name == "center_y":
            domains[name] = (0.0, max(image_size_value - 1.0, 0.0))
            continue
        source_name = gui_geometry_fit.geometry_fit_constraint_source_name(parameter_name)
        bound_entry = None
        for candidate_name in (parameter_name, source_name, name):
            if candidate_name in bounds_cfg:
                bound_entry = bounds_cfg.get(candidate_name)
                break
        if bound_entry is None:
            continue
        current_value = current_params.get(
            parameter_name,
            current_params.get(source_name, current_params.get(name)),
        )
        domain = _headless_geometry_fit_domain_from_bounds_entry(
            bound_entry,
            current_value=current_value,
        )
        if domain is None:
            continue
        if parameter_name == "theta_offset":
            span = max(abs(float(domain[0])), abs(float(domain[1])), 1.0)
            domain = (-float(span), float(span))
        domains[name] = (float(domain[0]), float(domain[1]))
    return domains


def _read_first_cif_block(path: str) -> tuple[object, object]:
    """Load one CIF and return the container plus its first block."""

    import CifFile

    cf = CifFile.ReadCif(path)
    keys = list(cf.keys())
    if not keys:
        raise ValueError(f"No CIF data blocks found in {path}")
    return cf, cf[keys[0]]


@lru_cache(maxsize=1)
def _load_gui_background_module():
    from ra_sim.gui import background

    return background


@lru_cache(maxsize=1)
def _load_gui_background_theta_module():
    from ra_sim.gui import background_theta

    return background_theta


@lru_cache(maxsize=1)
def _load_gui_controllers_module():
    from ra_sim.gui import controllers

    return controllers


@lru_cache(maxsize=1)
def _load_gui_geometry_fit_module():
    from ra_sim.gui import geometry_fit

    return geometry_fit


@lru_cache(maxsize=1)
def _load_gui_geometry_overlay_module():
    from ra_sim.gui import geometry_overlay

    return geometry_overlay


@lru_cache(maxsize=1)
def _load_gui_geometry_q_group_manager_module():
    from ra_sim.gui import geometry_q_group_manager

    return geometry_q_group_manager


@lru_cache(maxsize=1)
def _load_gui_manual_geometry_module():
    from ra_sim.gui import manual_geometry

    return manual_geometry


@lru_cache(maxsize=1)
def _load_gui_structure_model_module():
    from ra_sim.gui import structure_model

    return structure_model


@lru_cache(maxsize=1)
def _load_gui_modules() -> SimpleNamespace:
    return SimpleNamespace(
        gui_background=_LazyModuleProxy(_load_gui_background_module),
        gui_background_theta=_LazyModuleProxy(_load_gui_background_theta_module),
        gui_controllers=_LazyModuleProxy(_load_gui_controllers_module),
        gui_geometry_fit=_LazyModuleProxy(_load_gui_geometry_fit_module),
        gui_geometry_overlay=_LazyModuleProxy(_load_gui_geometry_overlay_module),
        gui_geometry_q_group_manager=_LazyModuleProxy(_load_gui_geometry_q_group_manager_module),
        gui_manual_geometry=_LazyModuleProxy(_load_gui_manual_geometry_module),
        gui_structure_model=_LazyModuleProxy(_load_gui_structure_model_module),
    )


@lru_cache(maxsize=1)
def _load_gui_state_types() -> SimpleNamespace:
    from ra_sim.gui.state import (
        AtomSiteOverrideState,
        BackgroundRuntimeState,
        SimulationRuntimeState,
    )

    return SimpleNamespace(
        AtomSiteOverrideState=AtomSiteOverrideState,
        BackgroundRuntimeState=BackgroundRuntimeState,
        SimulationRuntimeState=SimulationRuntimeState,
    )


@lru_cache(maxsize=1)
def _load_fitting_runtime():
    from ra_sim.fitting import optimization

    return optimization


@lru_cache(maxsize=1)
def _load_exact_cake_portable_module():
    from ra_sim.simulation import exact_cake_portable

    return exact_cake_portable


@lru_cache(maxsize=1)
def _load_simulation_diffraction():
    from ra_sim.simulation import diffraction

    return diffraction


@lru_cache(maxsize=1)
def _load_intersection_cache_schema():
    from ra_sim.simulation import intersection_cache_schema

    return intersection_cache_schema


@lru_cache(maxsize=1)
def _load_stacking_fault_runtime():
    from ra_sim.utils import stacking_fault

    return stacking_fault


@lru_cache(maxsize=1)
def _load_diffraction_tools():
    from ra_sim.utils import diffraction_tools

    return diffraction_tools


@lru_cache(maxsize=1)
def _load_calculation_runtime():
    from ra_sim.utils import calculations

    return calculations


@lru_cache(maxsize=1)
def _load_tools_runtime():
    from ra_sim.utils import tools

    return tools


class _LazyModuleProxy:
    """Resolve heavy module only when code touches it."""

    def __init__(self, loader) -> None:
        object.__setattr__(self, "_loader", loader)

    def __getattr__(self, name: str) -> object:
        return getattr(object.__getattribute__(self, "_loader")(), name)

    def __setattr__(self, name: str, value: object) -> None:
        setattr(object.__getattribute__(self, "_loader")(), name, value)


gui_background = _LazyModuleProxy(lambda: _load_gui_modules().gui_background)
gui_background_theta = _LazyModuleProxy(lambda: _load_gui_modules().gui_background_theta)
gui_controllers = _LazyModuleProxy(lambda: _load_gui_modules().gui_controllers)
gui_geometry_fit = _LazyModuleProxy(lambda: _load_gui_modules().gui_geometry_fit)
gui_geometry_overlay = _LazyModuleProxy(lambda: _load_gui_modules().gui_geometry_overlay)
gui_geometry_q_group_manager = _LazyModuleProxy(
    lambda: _load_gui_modules().gui_geometry_q_group_manager
)
gui_manual_geometry = _LazyModuleProxy(lambda: _load_gui_modules().gui_manual_geometry)
gui_structure_model = _LazyModuleProxy(lambda: _load_gui_modules().gui_structure_model)


def _headless_native_detector_coords_to_detector_display_coords_for_background(
    load_background_by_index,
    background_index: int,
    *,
    display_rotate_k: int = DISPLAY_ROTATE_K,
):
    try:
        bg_idx = int(background_index)
    except Exception:
        return None
    try:
        native_background, _display_background = load_background_by_index(bg_idx)
        native_shape = tuple(int(v) for v in np.asarray(native_background).shape[:2])
    except Exception:
        return None
    if len(native_shape) < 2 or min(native_shape) <= 0:
        return None

    def _to_display(col: float, row: float):
        return gui_geometry_overlay.rotate_point_for_display(
            float(col),
            float(row),
            native_shape,
            int(display_rotate_k),
        )

    _to_display.__name__ = (
        f"_headless_native_detector_coords_to_detector_display_coords_bg_{bg_idx}"
    )
    return _to_display


def _headless_background_display_to_native_detector_coords_for_background(
    load_background_by_index,
    background_index: int,
    *,
    display_rotate_k: int = DISPLAY_ROTATE_K,
):
    try:
        bg_idx = int(background_index)
    except Exception:
        return None
    try:
        native_background, _display_background = load_background_by_index(bg_idx)
        native_shape = tuple(int(v) for v in np.asarray(native_background).shape[:2])
    except Exception:
        return None
    if len(native_shape) < 2 or min(native_shape) <= 0:
        return None

    def _to_native(col: float, row: float):
        return gui_geometry_overlay.rotate_point_for_display(
            float(col),
            float(row),
            native_shape,
            -int(display_rotate_k),
        )

    _to_native.__name__ = (
        f"_headless_background_display_to_native_detector_coords_bg_{bg_idx}"
    )
    return _to_native


def _coerce_float(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    if not np.isfinite(parsed):
        return float(default)
    return float(parsed)


def _coerce_int(value: object, default: int, minimum: int | None = None) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = int(default)
    if minimum is not None:
        parsed = max(int(minimum), parsed)
    return int(parsed)


def _coerce_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


_legacy_background_subtraction_warning_emitted = False


def _legacy_background_subtraction_requested(
    saved_state: Mapping[str, object],
    defaults: _RuntimeDefaults,
    *,
    mode_override: object | None,
) -> bool:
    override_text = (
        ""
        if mode_override is None
        else str(mode_override).strip().lower().replace("_", "-")
    )
    if override_text and override_text not in {"off", "saved"}:
        return True
    if override_text == "off":
        return False

    saved_variables = (
        saved_state.get("variables", {})
        if isinstance(saved_state.get("variables"), Mapping)
        else {}
    )
    if _coerce_bool(saved_variables.get("background_subtraction_enabled_var"), False):
        return True
    saved_mode = str(
        saved_variables.get("background_subtraction_mode_var") or ""
    ).strip().lower().replace("_", "-")
    if saved_mode and saved_mode != "off":
        return True

    geometry_cfg = (
        defaults.fit_config.get("geometry", {})
        if isinstance(defaults.fit_config, Mapping)
        else {}
    )
    config_cfg = (
        geometry_cfg.get("background_subtraction", {})
        if isinstance(geometry_cfg, Mapping)
        else {}
    )
    if isinstance(config_cfg, Mapping):
        if _coerce_bool(config_cfg.get("enabled"), False):
            return True
        config_mode = str(config_cfg.get("mode") or "").strip().lower().replace("_", "-")
        if config_mode and config_mode != "off":
            return True
    return False


def _warn_legacy_background_subtraction_noop() -> None:
    global _legacy_background_subtraction_warning_emitted
    if _legacy_background_subtraction_warning_emitted:
        return
    _legacy_background_subtraction_warning_emitted = True
    warnings.warn(
        "Legacy global background subtraction is ignored. "
        "Use Analyze peak fitting's local linear background subtraction instead.",
        RuntimeWarning,
        stacklevel=3,
    )


def _headless_background_subtraction_config(
    saved_state: Mapping[str, object],
    defaults: _RuntimeDefaults,
    *,
    mode_override: object | None,
    scale_override: object | None,
    diagnostics_override: bool | None,
    phi_block_overrides: Mapping[str, object] | None = None,
) -> _LegacyBackgroundSubtractionConfig:
    _ = scale_override, diagnostics_override, phi_block_overrides
    if _legacy_background_subtraction_requested(
        saved_state,
        defaults,
        mode_override=mode_override,
    ):
        _warn_legacy_background_subtraction_noop()
    return _LegacyBackgroundSubtractionConfig()


def _headless_geometry_fit_center(params: Mapping[str, object]) -> tuple[float, float] | None:
    center_value = params.get("center")
    if (
        isinstance(center_value, Sequence)
        and not isinstance(center_value, (str, bytes))
        and len(center_value) >= 2
    ):
        try:
            center_row = float(center_value[0])
            center_col = float(center_value[1])
        except Exception:
            center_row = center_col = float("nan")
        if np.isfinite(center_row) and np.isfinite(center_col):
            return float(center_row), float(center_col)
    try:
        center_row = float(params.get("center_x", np.nan))
        center_col = float(params.get("center_y", np.nan))
    except Exception:
        return None
    if not (np.isfinite(center_row) and np.isfinite(center_col)):
        return None
    return float(center_row), float(center_col)


def _build_headless_geometry_fit_caked_view_payload(
    detector_image: object,
    *,
    params: Mapping[str, object],
    pixel_size_m: float,
    npt_rad: int = HEADLESS_GEOMETRY_CAKED_RADIAL_BINS,
    npt_azim: int = HEADLESS_GEOMETRY_CAKED_AZIMUTH_BINS,
) -> dict[str, object] | None:
    """Build the exact caked view/projector payload used by manual caked fits."""

    image = np.asarray(detector_image, dtype=np.float64)
    if image.ndim != 2:
        return None
    detector_shape = tuple(int(v) for v in image.shape[:2])
    if detector_shape[0] <= 0 or detector_shape[1] <= 0:
        return None
    center = _headless_geometry_fit_center(params)
    if center is None:
        return None
    try:
        distance_m = float(params.get("corto_detector", np.nan))
        pixel_size = float(pixel_size_m)
        wavelength_m = float(params.get("lambda", np.nan)) * 1.0e-10
    except Exception:
        return None
    if not (
        np.isfinite(distance_m)
        and distance_m > 0.0
        and np.isfinite(pixel_size)
        and pixel_size > 0.0
    ):
        return None

    exact_cake = _load_exact_cake_portable_module()
    try:
        ai = exact_cake.FastAzimuthalIntegrator(
            dist=float(distance_m),
            poni1=float(center[0]) * float(pixel_size),
            poni2=float(center[1]) * float(pixel_size),
            rot1=0.0,
            rot2=0.0,
            rot3=0.0,
            wavelength=float(wavelength_m) if np.isfinite(wavelength_m) else None,
            pixel1=float(pixel_size),
            pixel2=float(pixel_size),
        )
    except Exception:
        return None
    payload = gui_geometry_fit.build_geometry_fit_exact_caked_view_payload(
        image,
        ai=ai,
        detector_shape=detector_shape,
        npt_rad=int(max(2, npt_rad)),
        npt_azim=int(max(2, npt_azim)),
    )
    if not isinstance(payload, dict):
        return None
    payload["ai"] = ai
    payload["background_image"] = np.asarray(payload.get("background"), dtype=np.float64)
    return payload


def _build_headless_geometry_fit_caked_projection_payload(
    detector_shape: object,
    *,
    params: Mapping[str, object],
    pixel_size_m: float,
    background_index: int | None = None,
    npt_rad: int = HEADLESS_GEOMETRY_CAKED_RADIAL_BINS,
    npt_azim: int = HEADLESS_GEOMETRY_CAKED_AZIMUTH_BINS,
) -> dict[str, object] | None:
    """Build exact caked projector payload without integrating a display image."""

    del background_index
    try:
        normalized_shape = tuple(int(v) for v in tuple(detector_shape)[:2])
    except Exception:
        return None
    if len(normalized_shape) < 2 or min(normalized_shape) <= 0:
        return None
    center = _headless_geometry_fit_center(params)
    if center is None:
        return None
    try:
        distance_m = float(params.get("corto_detector", np.nan))
        pixel_size = float(pixel_size_m)
        wavelength_m = float(params.get("lambda", np.nan)) * 1.0e-10
    except Exception:
        return None
    if not (
        np.isfinite(distance_m)
        and distance_m > 0.0
        and np.isfinite(pixel_size)
        and pixel_size > 0.0
    ):
        return None

    exact_cake = _load_exact_cake_portable_module()
    try:
        ai = exact_cake.FastAzimuthalIntegrator(
            dist=float(distance_m),
            poni1=float(center[0]) * float(pixel_size),
            poni2=float(center[1]) * float(pixel_size),
            rot1=0.0,
            rot2=0.0,
            rot3=0.0,
            wavelength=float(wavelength_m) if np.isfinite(wavelength_m) else None,
            pixel1=float(pixel_size),
            pixel2=float(pixel_size),
        )
    except Exception:
        return None
    payload = gui_geometry_fit.build_geometry_fit_exact_caked_projection_view(
        detector_shape=normalized_shape,
        ai=ai,
        npt_rad=int(max(2, npt_rad)),
        npt_azim=int(max(2, npt_azim)),
    )
    projection = gui_geometry_fit.geometry_fit_caked_projection_payload(payload)
    return dict(projection) if isinstance(projection, Mapping) else None


def _ensure_triplet(raw_value: object, fallback: list[float]) -> list[float]:
    values: list[float] = []
    if isinstance(raw_value, (list, tuple, np.ndarray)):
        for item in raw_value:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                continue
    if len(values) < 3:
        values.extend(float(item) for item in fallback[len(values) : 3])
    return [float(values[idx]) for idx in range(3)]


def _resolve_solve_q_mode(mode_raw: object) -> int:
    diffraction = _load_simulation_diffraction()
    if isinstance(mode_raw, (int, np.integer, float, np.floating)):
        return 0 if int(round(float(mode_raw))) == 0 else 1

    mode_txt = str(mode_raw).strip().lower()
    if mode_txt in {"uniform", "fast", "0"}:
        return 0
    if mode_txt in {"adaptive", "robust", "1"}:
        return 1
    return int(diffraction.DEFAULT_SOLVE_Q_MODE)


def _normalize_optics_mode_label(value: object) -> str:
    diffraction = _load_simulation_diffraction()
    if value is None:
        return "fast"
    if isinstance(value, (int, np.integer, float, np.floating)):
        return "exact" if int(round(float(value))) == diffraction.OPTICS_MODE_EXACT else "fast"

    text = " ".join(str(value).strip().lower().split())
    if text in {
        "1",
        "true",
        "yes",
        "on",
        "exact",
        "precise",
        "slow",
        "complex_k_dwba_slab",
        "complex-k dwba slab optics",
        "phase-matched complex-k multilayer dwba",
    }:
        return "exact"
    if text in {
        "0",
        "false",
        "no",
        "off",
        "fast",
        "approx",
        "fresnel_ctr_damping",
        "fresnel-weighted kinematic ctr absorption correction",
        "uncoupled fresnel + ctr damping (ufd)",
        "fast dwba-lite (fresnel + depth-sum attenuation)",
        "ufd",
        "dwba-lite",
    }:
        return "fast"
    if "complex-k dwba" in text or "complex_k_dwba" in text:
        return "exact"
    return "fast"


def _resolve_optics_mode_flag(value: object) -> int:
    diffraction = _load_simulation_diffraction()
    return (
        diffraction.OPTICS_MODE_EXACT
        if _normalize_optics_mode_label(value) == "exact"
        else diffraction.OPTICS_MODE_FAST
    )


def _default_fit_toggle_values() -> dict[str, bool]:
    return {
        "fit_zb_var": True,
        "fit_zs_var": True,
        "fit_theta_var": True,
        "fit_psi_z_var": True,
        "fit_chi_var": True,
        "fit_cor_var": True,
        "fit_gamma_var": True,
        "fit_Gamma_var": True,
        "fit_dist_var": True,
        "fit_a_var": False,
        "fit_c_var": False,
        "fit_center_x_var": False,
        "fit_center_y_var": False,
    }


def _build_runtime_defaults(saved_state: dict[str, object]) -> _RuntimeDefaults:
    diffraction = _load_simulation_diffraction()
    pixel_tools = _load_diffraction_tools()
    gui_geometry_overlay = _load_gui_geometry_overlay_module()
    stack = _load_stacking_fault_runtime()
    instrument = get_instrument_config().get("instrument", {})
    detector_cfg = instrument.get("detector", {})
    geometry_cfg = instrument.get("geometry_defaults", {})
    beam_cfg = instrument.get("beam", {})
    sample_cfg = instrument.get("sample_orientation", {})
    debye_cfg = instrument.get("debye_waller", {})
    ht_cfg = instrument.get("hendricks_teller", {})
    fit_config = instrument.get("fit", {})
    files_state = saved_state.get("files", {}) if isinstance(saved_state.get("files"), dict) else {}
    flags_state = saved_state.get("flags", {}) if isinstance(saved_state.get("flags"), dict) else {}

    primary_cif_path = str(files_state.get("primary_cif_path") or get_path("cif_file"))
    secondary_cif_raw = files_state.get("secondary_cif_path")
    if secondary_cif_raw:
        secondary_cif_path = str(secondary_cif_raw)
    else:
        try:
            secondary_cif_path = str(get_path("cif_file2"))
        except KeyError:
            secondary_cif_path = None

    osc_files_raw = files_state.get("background_files", [])
    osc_files = [str(Path(str(path)).expanduser()) for path in osc_files_raw if str(path).strip()]
    if not osc_files:
        raise ValueError("Saved GUI state does not include any background files.")

    current_background_index = _coerce_int(
        files_state.get("current_background_index", 0),
        0,
        minimum=0,
    )
    current_background_index = min(current_background_index, len(osc_files) - 1)

    image_size = int(detector_cfg.get("image_size", 3000))
    pixel_size_m = float(detector_cfg.get("pixel_size_m", pixel_tools.DEFAULT_PIXEL_SIZE_M))

    poni = parse_poni_file(get_path("geometry_poni"))
    distance_m = float(poni.get("Dist", geometry_cfg.get("distance_m", 0.075)))
    gamma_initial = float(poni.get("Rot2", geometry_cfg.get("rot2", 0.0)))
    Gamma_initial = float(poni.get("Rot1", geometry_cfg.get("rot1", 0.0)))
    poni1 = float(poni.get("Poni1", geometry_cfg.get("poni1_m", 0.0)))
    poni2 = float(poni.get("Poni2", geometry_cfg.get("poni2_m", 0.0)))
    wave_m = float(poni.get("Wavelength", geometry_cfg.get("wavelength_m", 1.0e-10)))
    lambda_from_poni = wave_m * 1.0e10
    lambda_override = beam_cfg.get("wavelength_angstrom")
    lambda_angstrom = float(lambda_override if lambda_override is not None else lambda_from_poni)

    center_default = list(
        gui_geometry_overlay.beam_center_row_col_from_poni(
            float(poni1),
            float(poni2),
            float(pixel_size_m),
        )
    )
    two_theta_max = pixel_tools.detector_two_theta_max(
        image_size,
        center_default,
        distance_m,
        pixel_size=pixel_size_m,
    )

    cf, blk = _read_first_cif_block(primary_cif_path)
    av = gui_structure_model.parse_cif_num(blk.get("_cell_length_a"))
    bv = gui_structure_model.parse_cif_num(blk.get("_cell_length_b"))
    cv = gui_structure_model.parse_cif_num(blk.get("_cell_length_c"))
    if secondary_cif_path:
        cf2, blk2 = _read_first_cif_block(secondary_cif_path)
        av2 = gui_structure_model.parse_cif_num(blk2.get("_cell_length_a") or av)
        cv2 = gui_structure_model.parse_cif_num(blk2.get("_cell_length_c") or cv)
    else:
        av2 = None
        cv2 = None

    p_defaults = _ensure_triplet(ht_cfg.get("default_p"), [0.01, 0.99, 0.5])
    w_defaults = _ensure_triplet(ht_cfg.get("default_w"), [100.0, 0.0, 0.0])
    try:
        iodine_z_default = float(stack._infer_iodine_z_like_diffuse(primary_cif_path))
    except Exception:
        iodine_z_default = 0.0
    if not np.isfinite(iodine_z_default):
        iodine_z_default = 0.0
    iodine_z_default = float(np.clip(iodine_z_default, 0.0, 1.0))

    phase_delta_default = stack.normalize_phase_delta_expression(
        ht_cfg.get("phase_delta_expression", stack.DEFAULT_PHASE_DELTA_EXPRESSION),
        fallback=stack.DEFAULT_PHASE_DELTA_EXPRESSION,
    )
    try:
        phase_delta_default = stack.validate_phase_delta_expression(phase_delta_default)
    except ValueError:
        phase_delta_default = stack.DEFAULT_PHASE_DELTA_EXPRESSION

    defaults = {
        "theta_initial": float(sample_cfg.get("theta_initial_deg", 6.0)),
        "cor_angle": float(sample_cfg.get("cor_deg", 0.0)),
        "gamma": float(Gamma_initial),
        "Gamma": float(gamma_initial),
        "chi": float(sample_cfg.get("chi_deg", 0.0)),
        "psi_z": float(sample_cfg.get("psi_z_deg", 0.0)),
        "zs": float(sample_cfg.get("zs", 0.0)),
        "zb": float(sample_cfg.get("zb", 0.0)),
        "sample_width_m": float(sample_cfg.get("width_m", 0.0)),
        "sample_length_m": float(sample_cfg.get("length_m", 0.0)),
        "sample_depth_m": float(sample_cfg.get("depth_m", 0.0)),
        "debye_x": float(debye_cfg.get("x", 0.0)),
        "debye_y": float(debye_cfg.get("y", 0.0)),
        "corto_detector": float(distance_m),
        "sigma_mosaic_deg": float(beam_cfg.get("sigma_mosaic_fwhm_deg", 0.8)),
        "gamma_mosaic_deg": float(beam_cfg.get("gamma_mosaic_fwhm_deg", 0.7)),
        "eta": float(beam_cfg.get("eta", 0.0)),
        "a": float(av),
        "b": float(bv),
        "c": float(cv),
        "a2": float(av2) if av2 is not None else None,
        "c2": float(cv2) if cv2 is not None else None,
        "p0": float(p_defaults[0]),
        "p1": float(p_defaults[1]),
        "p2": float(p_defaults[2]),
        "w0": float(w_defaults[0]),
        "w1": float(w_defaults[1]),
        "w2": float(w_defaults[2]),
        "iodine_z": float(iodine_z_default),
        "phase_delta_expression": str(phase_delta_default),
        "phi_l_divisor": float(
            stack.normalize_phi_l_divisor(
                ht_cfg.get("phi_l_divisor", stack.DEFAULT_PHI_L_DIVISOR),
                fallback=stack.DEFAULT_PHI_L_DIVISOR,
            )
        ),
        "center_x": float(center_default[0]),
        "center_y": float(center_default[1]),
        "bandwidth_percent": float(
            np.clip(float(beam_cfg.get("bandwidth_percent", 0.7)), 0.0, 10.0)
        ),
        "solve_q_steps": int(beam_cfg.get("solve_q_steps", diffraction.DEFAULT_SOLVE_Q_STEPS)),
        "solve_q_rel_tol": float(
            beam_cfg.get("solve_q_rel_tol", diffraction.DEFAULT_SOLVE_Q_REL_TOL)
        ),
        "solve_q_mode": _resolve_solve_q_mode(
            beam_cfg.get("solve_q_mode", diffraction.DEFAULT_SOLVE_Q_MODE)
        ),
        "finite_stack": bool(ht_cfg.get("finite_stack", True)),
        "stack_layers": int(max(1, float(ht_cfg.get("stack_layers", 50)))),
        "optics_mode": "fast",
        "weight1": 0.5 if secondary_cif_path else 1.0,
        "weight2": 0.5 if secondary_cif_path else 0.0,
    }

    return _RuntimeDefaults(
        primary_cif_path=primary_cif_path,
        secondary_cif_path=secondary_cif_path,
        osc_files=osc_files,
        current_background_index=current_background_index,
        image_size=image_size,
        pixel_size_m=pixel_size_m,
        lambda_angstrom=lambda_angstrom,
        psi_deg=float(sample_cfg.get("psi_deg", 0.0)),
        defaults=defaults,
        fit_config=dict(fit_config) if isinstance(fit_config, dict) else {},
        intensity_threshold=float(detector_cfg.get("intensity_threshold", 1.0)),
        include_rods_flag=bool(ht_cfg.get("include_rods", False)),
        two_theta_range=(0.0, float(two_theta_max)),
        mx=int(ht_cfg.get("max_miller_index", 19)),
        background_flags={
            "backend_rotation_k": _coerce_int(
                flags_state.get("background_backend_rotation_k", 3),
                3,
            ),
            "backend_flip_x": _coerce_bool(
                flags_state.get("background_backend_flip_x", False),
                False,
            ),
            "backend_flip_y": _coerce_bool(
                flags_state.get("background_backend_flip_y", False),
                False,
            ),
        },
    )


def _build_var_store(
    saved_state: dict[str, object],
    defaults: _RuntimeDefaults,
) -> dict[str, _HeadlessVar]:
    saved_variables = (
        saved_state.get("variables", {}) if isinstance(saved_state.get("variables"), dict) else {}
    )
    geometry_fit_selection_default = gui_background_theta.default_geometry_fit_background_selection(
        osc_files=defaults.osc_files,
    )
    background_theta_default = gui_background_theta.format_background_theta_values(
        [defaults.defaults["theta_initial"]] * len(defaults.osc_files)
    )

    var_defaults: dict[str, object] = {
        **_default_fit_toggle_values(),
        "zb_var": defaults.defaults["zb"],
        "zs_var": defaults.defaults["zs"],
        "theta_initial_var": defaults.defaults["theta_initial"],
        "psi_z_var": defaults.defaults["psi_z"],
        "chi_var": defaults.defaults["chi"],
        "cor_angle_var": defaults.defaults["cor_angle"],
        "sample_width_var": defaults.defaults["sample_width_m"],
        "sample_length_var": defaults.defaults["sample_length_m"],
        "sample_depth_var": defaults.defaults["sample_depth_m"],
        "gamma_var": defaults.defaults["gamma"],
        "Gamma_var": defaults.defaults["Gamma"],
        "corto_detector_var": defaults.defaults["corto_detector"],
        "a_var": defaults.defaults["a"],
        "c_var": defaults.defaults["c"],
        "center_x_var": defaults.defaults["center_x"],
        "center_y_var": defaults.defaults["center_y"],
        "debye_x_var": defaults.defaults["debye_x"],
        "debye_y_var": defaults.defaults["debye_y"],
        "geometry_theta_offset_var": "0.0",
        "background_theta_list_var": background_theta_default,
        "geometry_fit_background_selection_var": geometry_fit_selection_default,
        "sigma_mosaic_var": defaults.defaults["sigma_mosaic_deg"],
        "gamma_mosaic_var": defaults.defaults["gamma_mosaic_deg"],
        "eta_var": defaults.defaults["eta"],
        "bandwidth_percent_var": defaults.defaults["bandwidth_percent"],
        "solve_q_steps_var": defaults.defaults["solve_q_steps"],
        "solve_q_rel_tol_var": defaults.defaults["solve_q_rel_tol"],
        "solve_q_mode_var": defaults.defaults["solve_q_mode"],
        "optics_mode_var": defaults.defaults["optics_mode"],
        "p0_var": defaults.defaults["p0"],
        "p1_var": defaults.defaults["p1"],
        "p2_var": defaults.defaults["p2"],
        "w0_var": defaults.defaults["w0"],
        "w1_var": defaults.defaults["w1"],
        "w2_var": defaults.defaults["w2"],
        "finite_stack_var": defaults.defaults["finite_stack"],
        "stack_layers_var": defaults.defaults["stack_layers"],
        "phase_delta_expr_var": defaults.defaults["phase_delta_expression"],
        "phi_l_divisor_var": defaults.defaults["phi_l_divisor"],
        "weight1_var": defaults.defaults["weight1"],
        "weight2_var": defaults.defaults["weight2"],
    }

    return {
        name: _HeadlessVar(saved_variables.get(name, default))
        for name, default in var_defaults.items()
    }


def _restore_manual_pairs(
    osc_files: list[str],
    saved_rows: list[object] | None,
) -> dict[int, list[dict[str, object]]]:
    pairs_by_background: dict[int, list[dict[str, object]]] = {}

    def _pairs_for_index(index: int) -> list[dict[str, object]]:
        return gui_manual_geometry.geometry_manual_pairs_for_index(
            int(index),
            pairs_by_background=pairs_by_background,
        )

    def _replace(payload: dict[int, list[dict[str, object]]]) -> None:
        pairs_by_background.clear()
        pairs_by_background.update(
            {
                int(idx): [dict(entry) for entry in entries]
                for idx, entries in payload.items()
                if entries
            }
        )

    gui_manual_geometry.apply_geometry_manual_pairs_rows(
        saved_rows,
        osc_files=osc_files,
        pairs_for_index=_pairs_for_index,
        replace_pairs_by_background=_replace,
        clear_preview_artists=lambda **_kwargs: None,
        cancel_pick_session=lambda **_kwargs: None,
        invalidate_pick_cache=lambda: None,
        clear_manual_undo_stack=lambda: None,
        clear_geometry_fit_undo_stack=lambda: None,
        render_current_pairs=lambda **_kwargs: None,
        update_button_label=lambda: None,
        refresh_status=lambda: None,
    )
    return pairs_by_background


def _load_structure_model(
    defaults: _RuntimeDefaults,
    saved_state: dict[str, object],
    var_store: dict[str, _HeadlessVar],
    simulation_runtime_state: SimulationRuntimeState,
) -> tuple[object, AtomSiteOverrideState, str, complex]:
    calc_runtime = _load_calculation_runtime()
    stack = _load_stacking_fault_runtime()
    state_types = _load_gui_state_types()
    tools_runtime = _load_tools_runtime()
    dynamic_lists = (
        saved_state.get("dynamic_lists", {})
        if isinstance(saved_state.get("dynamic_lists"), dict)
        else {}
    )
    cf, blk = _read_first_cif_block(defaults.primary_cif_path)
    occupancy_site_labels, occupancy_site_expanded_map = (
        gui_structure_model.extract_occupancy_site_metadata(
            blk,
            defaults.primary_cif_path,
        )
    )
    occ_source = dynamic_lists.get("occupancy_values")
    if not isinstance(occ_source, (list, tuple, np.ndarray)):
        occ_source = (
            get_instrument_config()
            .get("instrument", {})
            .get("occupancies", {})
            .get("default", [1.0])
        )
    occupancy_count = len(occupancy_site_labels) or max(1, len(list(occ_source)))
    occ_values = gui_controllers.clamp_site_occupancy_values(
        list(occ_source)[:occupancy_count]
        + [1.0] * max(0, occupancy_count - len(list(occ_source)[:occupancy_count]))
    )

    atom_site_fractional_metadata = gui_structure_model.extract_atom_site_fractional_metadata(blk)
    saved_atom_sites = dynamic_lists.get("atom_site_fractional_values")
    if isinstance(saved_atom_sites, list) and len(saved_atom_sites) == len(
        atom_site_fractional_metadata
    ):
        atom_site_values: list[tuple[float, float, float]] = []
        for idx, row in enumerate(saved_atom_sites):
            row_default = atom_site_fractional_metadata[idx]
            if isinstance(row, dict):
                atom_site_values.append(
                    (
                        _coerce_float(row.get("x"), float(row_default["x"])),
                        _coerce_float(row.get("y"), float(row_default["y"])),
                        _coerce_float(row.get("z"), float(row_default["z"])),
                    )
                )
            else:
                atom_site_values.append(
                    (
                        float(row_default["x"]),
                        float(row_default["y"]),
                        float(row_default["z"]),
                    )
                )
    else:
        atom_site_values = [
            (
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
            )
            for row in atom_site_fractional_metadata
        ]

    defaults_map = copy.deepcopy(defaults.defaults)
    for target_name, var_name in (
        ("a", "a_var"),
        ("c", "c_var"),
        ("p0", "p0_var"),
        ("p1", "p1_var"),
        ("p2", "p2_var"),
        ("w0", "w0_var"),
        ("w1", "w1_var"),
        ("w2", "w2_var"),
    ):
        defaults_map[target_name] = _coerce_float(
            var_store[var_name].get(),
            float(defaults_map[target_name]),
        )
    defaults_map["finite_stack"] = _coerce_bool(
        var_store["finite_stack_var"].get(),
        defaults.defaults["finite_stack"],
    )
    defaults_map["stack_layers"] = _coerce_int(
        var_store["stack_layers_var"].get(),
        defaults.defaults["stack_layers"],
        minimum=1,
    )
    defaults_map["phase_delta_expression"] = stack.validate_phase_delta_expression(
        stack.normalize_phase_delta_expression(
            var_store["phase_delta_expr_var"].get(),
            fallback=str(defaults.defaults["phase_delta_expression"]),
        )
    )
    defaults_map["phi_l_divisor"] = stack.normalize_phi_l_divisor(
        var_store["phi_l_divisor_var"].get(),
        fallback=float(defaults.defaults["phi_l_divisor"]),
    )

    structure_state = gui_structure_model.build_initial_structure_model_state(
        cif_file=defaults.primary_cif_path,
        cf=cf,
        blk=blk,
        cif_file2=defaults.secondary_cif_path,
        occupancy_site_labels=occupancy_site_labels,
        occupancy_site_expanded_map=occupancy_site_expanded_map,
        occ=occ_values,
        atom_site_fractional_metadata=atom_site_fractional_metadata,
        av=float(defaults.defaults["a"]),
        bv=float(defaults.defaults["b"]),
        cv=float(defaults.defaults["c"]),
        av2=defaults.defaults.get("a2"),
        cv2=defaults.defaults.get("c2"),
        defaults=defaults_map,
        mx=defaults.mx,
        lambda_angstrom=defaults.lambda_angstrom,
        intensity_threshold=defaults.intensity_threshold,
        two_theta_range=defaults.two_theta_range,
        include_rods_flag=defaults.include_rods_flag,
        combine_weighted_intensities=gui_controllers.combine_cif_weighted_intensities,
        miller_generator=tools_runtime.miller_generator,
        inject_fractional_reflections=tools_runtime.inject_fractional_reflections,
    )
    atom_site_override_state = state_types.AtomSiteOverrideState()
    gui_structure_model.rebuild_diffraction_inputs(
        structure_state,
        new_occ=occ_values,
        p_vals=[
            _coerce_float(var_store["p0_var"].get(), defaults.defaults["p0"]),
            _coerce_float(var_store["p1_var"].get(), defaults.defaults["p1"]),
            _coerce_float(var_store["p2_var"].get(), defaults.defaults["p2"]),
        ],
        weights=gui_controllers.normalize_stacking_weight_values(
            [
                var_store["w0_var"].get(),
                var_store["w1_var"].get(),
                var_store["w2_var"].get(),
            ]
        ),
        a_axis=_coerce_float(var_store["a_var"].get(), defaults.defaults["a"]),
        c_axis=_coerce_float(var_store["c_var"].get(), defaults.defaults["c"]),
        finite_stack_flag=_coerce_bool(
            var_store["finite_stack_var"].get(),
            defaults.defaults["finite_stack"],
        ),
        layers=_coerce_int(
            var_store["stack_layers_var"].get(),
            defaults.defaults["stack_layers"],
            minimum=1,
        ),
        phase_delta_expression_current=stack.validate_phase_delta_expression(
            stack.normalize_phase_delta_expression(
                var_store["phase_delta_expr_var"].get(),
                fallback=str(defaults.defaults["phase_delta_expression"]),
            )
        ),
        phi_l_divisor_current=stack.normalize_phi_l_divisor(
            var_store["phi_l_divisor_var"].get(),
            fallback=float(defaults.defaults["phi_l_divisor"]),
        ),
        atom_site_values=atom_site_values,
        iodine_z_current=gui_structure_model.current_iodine_z(
            structure_state,
            atom_site_override_state,
            atom_site_values=atom_site_values,
        ),
        atom_site_override_state=atom_site_override_state,
        simulation_runtime_state=simulation_runtime_state,
        combine_weighted_intensities=gui_controllers.combine_cif_weighted_intensities,
        build_intensity_dataframes=tools_runtime.build_intensity_dataframes,
        apply_bragg_qr_filters=lambda **_kwargs: None,
        schedule_update=lambda: None,
        weight1=_coerce_float(var_store["weight1_var"].get(), defaults.defaults["weight1"]),
        weight2=_coerce_float(var_store["weight2_var"].get(), defaults.defaults["weight2"]),
        force=True,
        trigger_update=False,
    )
    active_cif_path = gui_structure_model.active_primary_cif_path(
        structure_state,
        atom_site_override_state,
        atom_site_values=atom_site_values,
    )
    nominal_n2 = calc_runtime.resolve_index_of_refraction(
        defaults.lambda_angstrom * 1.0e-10,
        cif_path=active_cif_path,
    )
    return structure_state, atom_site_override_state, str(active_cif_path), nominal_n2


def _sync_background_theta_state(
    saved_state: dict[str, object],
    defaults: _RuntimeDefaults,
    var_store: dict[str, _HeadlessVar],
) -> None:
    del saved_state
    selection_var = var_store["geometry_fit_background_selection_var"]
    theta_var = var_store["theta_initial_var"]
    theta_list_var = var_store["background_theta_list_var"]
    if gui_background_theta.geometry_fit_uses_shared_theta_offset(
        osc_files=defaults.osc_files,
        current_background_index=defaults.current_background_index,
        geometry_fit_background_selection_var=selection_var,
    ):
        return
    try:
        theta_values = gui_background_theta.current_background_theta_values(
            osc_files=defaults.osc_files,
            theta_initial_var=theta_var,
            defaults={"theta_initial": defaults.defaults["theta_initial"]},
            theta_initial=defaults.defaults["theta_initial"],
            background_theta_list_var=theta_list_var,
            strict_count=False,
        )
    except Exception:
        theta_values = []
    if not theta_values:
        theta_values = [float(defaults.defaults["theta_initial"])] * len(defaults.osc_files)
    idx = min(max(defaults.current_background_index, 0), len(theta_values) - 1)
    theta_values[idx] = _coerce_float(theta_var.get(), theta_values[idx])
    theta_list_var.set(gui_background_theta.format_background_theta_values(theta_values))


def _updated_state_snapshot(
    saved_state: dict[str, object],
    defaults: _RuntimeDefaults,
    var_store: dict[str, _HeadlessVar],
) -> dict[str, object]:
    _sync_background_theta_state(saved_state, defaults, var_store)
    updated_state = copy.deepcopy(saved_state)
    variables = updated_state.get("variables")
    if not isinstance(variables, dict):
        variables = {}
        updated_state["variables"] = variables
    for name, var in var_store.items():
        variables[name] = var.get()
    return updated_state


def _empty_peak_overlay_cache() -> dict[str, object]:
    return {
        "sig": None,
        "positions": [],
        "millers": [],
        "intensities": [],
        "records": [],
        "click_spatial_index": None,
        "restored_from_gui_state": False,
    }


def _copy_hit_tables(hit_tables: Sequence[object] | None) -> list[np.ndarray]:
    schema = _load_intersection_cache_schema()
    copied: list[np.ndarray] = []
    if not isinstance(hit_tables, Sequence) or isinstance(hit_tables, (str, bytes)):
        return copied
    for table in hit_tables:
        try:
            copied.append(np.asarray(table, dtype=np.float64).copy())
        except Exception:
            copied.append(schema.empty_hit_table())
    return copied


def _copy_intersection_cache_tables(
    cache: Sequence[object] | None,
) -> list[np.ndarray]:
    schema = _load_intersection_cache_schema()
    copied: list[np.ndarray] = []
    if not isinstance(cache, Sequence) or isinstance(cache, (str, bytes)):
        return copied
    for table in cache:
        copied.append(schema.coerce_intersection_cache_table(table))
    return copied


def _restore_gui_state_peak_record(
    raw_record: object,
) -> dict[str, object] | None:
    if not isinstance(raw_record, Mapping):
        return None

    record = dict(raw_record)

    hkl_value = record.get("hkl")
    if isinstance(hkl_value, list) and len(hkl_value) >= 3:
        try:
            record["hkl"] = (
                int(hkl_value[0]),
                int(hkl_value[1]),
                int(hkl_value[2]),
            )
        except Exception:
            pass

    hkl_raw_value = record.get("hkl_raw")
    if isinstance(hkl_raw_value, list) and len(hkl_raw_value) >= 3:
        try:
            record["hkl_raw"] = (
                float(hkl_raw_value[0]),
                float(hkl_raw_value[1]),
                float(hkl_raw_value[2]),
            )
        except Exception:
            pass

    q_group_key = record.get("q_group_key")
    if isinstance(q_group_key, list):
        record["q_group_key"] = tuple(q_group_key)

    degenerate_hkls = record.get("degenerate_hkls")
    if isinstance(degenerate_hkls, list):
        normalized_deg_hkls: list[tuple[int, int, int]] = []
        for entry in degenerate_hkls:
            if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                continue
            try:
                normalized_deg_hkls.append((int(entry[0]), int(entry[1]), int(entry[2])))
            except Exception:
                continue
        record["degenerate_hkls"] = normalized_deg_hkls

    return record


def _replace_gui_state_peak_cache(
    simulation_runtime_state: SimulationRuntimeState,
    peak_records: Sequence[object] | None,
) -> None:
    restored_records: list[dict[str, object]] = []
    restored_positions: list[tuple[float, float]] = []
    restored_millers: list[tuple[int, int, int]] = []
    restored_intensities: list[float] = []

    for raw_record in peak_records or ():
        record = _restore_gui_state_peak_record(raw_record)
        if record is None:
            continue

        try:
            display_col = float(record.get("display_col", np.nan))
            display_row = float(record.get("display_row", np.nan))
        except Exception:
            display_col = float("nan")
            display_row = float("nan")

        hkl_value = record.get("hkl")
        if not isinstance(hkl_value, tuple) or len(hkl_value) < 3:
            continue
        try:
            hkl_triplet = (
                int(hkl_value[0]),
                int(hkl_value[1]),
                int(hkl_value[2]),
            )
        except Exception:
            continue

        try:
            intensity = float(record.get("intensity", record.get("weight", 0.0)))
        except Exception:
            intensity = 0.0
        if not np.isfinite(intensity):
            intensity = 0.0

        restored_records.append(record)
        if np.isfinite(display_col) and np.isfinite(display_row):
            restored_positions.append((float(display_col), float(display_row)))
        else:
            restored_positions.append((float("nan"), float("nan")))
        restored_millers.append(hkl_triplet)
        restored_intensities.append(float(intensity))

    simulation_runtime_state.peak_records = restored_records
    simulation_runtime_state.peak_positions = restored_positions
    simulation_runtime_state.peak_millers = restored_millers
    simulation_runtime_state.peak_intensities = restored_intensities
    simulation_runtime_state.selected_peak_record = None
    simulation_runtime_state.peak_overlay_cache = (
        {
            "sig": None,
            "positions": list(restored_positions),
            "millers": list(restored_millers),
            "intensities": list(restored_intensities),
            "records": [dict(record) for record in restored_records],
            "click_spatial_index": None,
            "restored_from_gui_state": bool(restored_records),
        }
        if restored_records
        else _empty_peak_overlay_cache()
    )


def _set_runtime_peak_cache_from_source_rows(
    simulation_runtime_state: SimulationRuntimeState,
    source_rows: Sequence[object] | None,
) -> None:
    restored_records: list[dict[str, object]] = []
    restored_positions: list[tuple[float, float]] = []
    restored_millers: list[tuple[int, int, int]] = []
    restored_intensities: list[float] = []

    for raw_entry in source_rows or ():
        if not isinstance(raw_entry, Mapping):
            continue
        peak_record = gui_manual_geometry.geometry_manual_canonicalize_live_source_entry(
            raw_entry,
            normalize_hkl_key=gui_geometry_overlay.normalize_hkl_key,
            allow_legacy_peak_fallback=False,
            preserve_existing_trusted_identity=True,
        )
        if peak_record is None:
            continue
        try:
            display_col = float(peak_record.get("sim_col", np.nan))
            display_row = float(peak_record.get("sim_row", np.nan))
        except Exception:
            continue
        if not (np.isfinite(display_col) and np.isfinite(display_row)):
            continue

        hkl_value = peak_record.get("hkl")
        if not isinstance(hkl_value, tuple) or len(hkl_value) < 3:
            continue
        try:
            hkl_triplet = (
                int(hkl_value[0]),
                int(hkl_value[1]),
                int(hkl_value[2]),
            )
        except Exception:
            continue

        try:
            intensity = float(peak_record.get("weight", peak_record.get("intensity", 0.0)))
        except Exception:
            intensity = 0.0
        if not np.isfinite(intensity):
            intensity = 0.0

        peak_record["display_col"] = float(display_col)
        peak_record["display_row"] = float(display_row)
        peak_record["intensity"] = float(intensity)
        peak_record["weight"] = float(intensity)

        restored_records.append(peak_record)
        restored_positions.append((float(display_col), float(display_row)))
        restored_millers.append(hkl_triplet)
        restored_intensities.append(float(intensity))

    simulation_runtime_state.peak_records = restored_records
    simulation_runtime_state.peak_positions = restored_positions
    simulation_runtime_state.peak_millers = restored_millers
    simulation_runtime_state.peak_intensities = restored_intensities
    simulation_runtime_state.selected_peak_record = None
    simulation_runtime_state.peak_overlay_cache = (
        {
            "sig": None,
            "positions": list(restored_positions),
            "millers": list(restored_millers),
            "intensities": list(restored_intensities),
            "records": [dict(record) for record in restored_records],
            "click_spatial_index": None,
            "restored_from_gui_state": False,
        }
        if restored_records
        else _empty_peak_overlay_cache()
    )


def _build_source_rows_from_hit_tables(
    hit_tables: Sequence[object] | None,
    *,
    image_size_value: int,
    params_local: Mapping[str, object],
    native_sim_to_display_coords,
    allow_nominal_hkl_indices: bool,
) -> tuple[
    list[dict[str, object]],
    list[tuple[float, float, str]],
    list[np.ndarray],
    list[int],
]:
    copied_hit_tables = _copy_hit_tables(hit_tables)
    if not copied_hit_tables:
        return [], [], [], []

    try:
        primary_a = float(params_local.get("a", np.nan))
    except Exception:
        primary_a = float("nan")
    try:
        primary_c = float(params_local.get("c", np.nan))
    except Exception:
        primary_c = float("nan")

    raw_rows, peak_table_lattice, source_reflection_indices = (
        gui_geometry_q_group_manager.build_geometry_fit_full_order_source_rows(
            copied_hit_tables,
            image_shape=(int(image_size_value), int(image_size_value)),
            native_sim_to_display_coords=native_sim_to_display_coords,
            primary_a=primary_a,
            primary_c=primary_c,
            default_source_label="primary",
            round_pixel_centers=False,
            allow_nominal_hkl_indices=bool(allow_nominal_hkl_indices),
            owner="headless_geometry_fit._build_source_rows_from_hit_tables",
        )
    )
    raw_rows = [dict(entry) for entry in (raw_rows or ()) if isinstance(entry, Mapping)]
    return raw_rows, peak_table_lattice, copied_hit_tables, source_reflection_indices


def _logged_cache_matches_params(
    metadata: Mapping[str, object] | None,
    params_local: Mapping[str, object],
) -> bool:
    if not isinstance(metadata, Mapping):
        return False

    comparisons = [
        (metadata.get("av"), params_local.get("a"), 1.0e-6),
        (metadata.get("cv"), params_local.get("c"), 1.0e-6),
        (metadata.get("wavelength_center"), params_local.get("lambda"), 1.0e-6),
        (metadata.get("theta_center"), params_local.get("theta_initial"), 1.0e-6),
    ]

    matched_checks = 0
    for logged_value, requested_value, tol in comparisons:
        try:
            logged_float = float(logged_value)
            requested_float = float(requested_value)
        except Exception:
            continue
        if not (np.isfinite(logged_float) and np.isfinite(requested_float)):
            continue
        matched_checks += 1
        if abs(float(logged_float) - float(requested_float)) > float(tol):
            return False
    return matched_checks > 0


HEADLESS_GEOMETRY_FIT_SEED_POLICY_DIRECT = "direct"
HEADLESS_GEOMETRY_FIT_SEED_POLICY_LADDER_MULTISTART = "ladder-multistart"
_HEADLESS_GEOMETRY_FIT_SEED_POLICIES = frozenset(
    {
        HEADLESS_GEOMETRY_FIT_SEED_POLICY_DIRECT,
        HEADLESS_GEOMETRY_FIT_SEED_POLICY_LADDER_MULTISTART,
    }
)
_HEADLESS_GEOMETRY_FIT_LADDER_SEED_SEARCH = {
    "prescore_top_k": 4,
    "n_global": 4,
    "n_jitter": 2,
    "min_seed_separation_u": 0.5,
}
_HEADLESS_GEOMETRY_FIT_SAVED_MANUAL_CAKED_SEED_SEARCH = {
    "prescore_top_k": 1,
    "n_global": 4,
    "n_jitter": 2,
    "min_seed_separation_u": 0.5,
    "_reuse_generation_for_prescore": True,
}
# Point-only direct solves converge within this cap for saved Bi caked states; higher
# finite-difference probes can enter unstable generated-row territory on Windows.
_HEADLESS_GEOMETRY_FIT_SAVED_MANUAL_CAKED_DIRECT_MAX_NFEV = 29
_HEADLESS_GEOMETRY_FIT_SAVED_MANUAL_CAKED_LADDER_MAX_NFEV = 60
_HEADLESS_GEOMETRY_FIT_POINT_ONLY_FLAG = "_qr_fit_point_only_projection"
_HEADLESS_GEOMETRY_FIT_PROGRESS_PHASES = frozenset(
    {
        "preflight",
        "runtime_config_ready",
        "solve_start",
        "seed_prescore",
        "selected_solve",
        "final_validation",
        "output_state_write",
    }
)
def _headless_geometry_fit_state_provenance(state_path: str | Path) -> dict[str, object]:
    state_file = Path(state_path).expanduser().resolve()
    try:
        with state_file.open("rb") as stream:
            state_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        state_hash = None
    return {
        "input_state_path": state_file,
        "input_state_sha256": state_hash,
    }


def _headless_progress_jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _headless_progress_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _headless_progress_jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _headless_progress_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_headless_progress_jsonable(item) for item in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except Exception:
            return str(value)
        if math.isfinite(number):
            return value
        return str(value)
    return str(value)


def _headless_progress_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _headless_progress_float(value: object) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return float(number)


def _headless_progress_pair(value: object) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    first = _headless_progress_float(value[0])
    second = _headless_progress_float(value[1])
    if first is None or second is None:
        return None
    return [float(first), float(second)]


def _headless_live_record_caked_display_alias(
    record: Mapping[str, object],
) -> list[float] | None:
    alias = _headless_progress_pair(record.get("fit_prediction_detector_display_px"))
    if alias is None:
        return None
    source = str(record.get("fit_prediction_detector_display_px_source") or "").strip()
    if source and source in _load_fitting_runtime().QR_FIT_CAKED_DEG_DETECTOR_DISPLAY_SOURCES:
        return alias
    if _headless_progress_pair(record.get("fit_prediction_detector_native_px")) is not None:
        return None
    simulated_detector_missing = (
        _headless_progress_float(record.get("simulated_detector_x")) is None
        or _headless_progress_float(record.get("simulated_detector_y")) is None
    )
    simulated_caked_present = _headless_progress_pair(
        record.get("sim_nominal_caked_deg")
    ) is not None or (
        _headless_progress_float(record.get("simulated_two_theta_deg")) is not None
        and _headless_progress_float(record.get("simulated_phi_deg")) is not None
    )
    if simulated_detector_missing and simulated_caked_present:
        return alias
    return None


def _headless_sanitize_live_cache_record(record: object) -> object:
    if not isinstance(record, Mapping):
        return record
    sanitized = dict(record)
    alias = _headless_live_record_caked_display_alias(sanitized)
    if alias is None:
        return sanitized
    sanitized["fit_prediction_detector_display_px"] = None
    sanitized["fit_prediction_detector_display_px_unavailable_reason"] = (
        "caked_degrees_not_detector_display_px"
    )
    if "fit_prediction_caked_deg" not in sanitized:
        sanitized["fit_prediction_caked_deg"] = alias
        sanitized["fit_prediction_caked_deg_source"] = "fit_prediction_detector_display_px"
    return sanitized


def _headless_progress_live_payload(value: object) -> object:
    payload = _headless_progress_jsonable(value)
    if not isinstance(payload, Mapping):
        return payload
    records = payload.get("live_cache_records")
    if not isinstance(records, list):
        return payload
    updated = dict(payload)
    updated["live_cache_records"] = [
        _headless_sanitize_live_cache_record(record) for record in records
    ]
    return updated


class _HeadlessGeometryFitProgressWriter:
    """Private JSON sidecar for long headless geometry fits."""

    def __init__(
        self,
        path: str | Path | None,
        *,
        active_vars: Sequence[object] | None = None,
        seed_policy: object | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve() if path is not None else None
        self.started_at = time.monotonic()
        self.data: dict[str, object] = {
            "phase": "preflight",
            "elapsed_s": 0.0,
            "pid": int(os.getpid()),
            "active_vars": [str(name) for name in (active_vars or ())],
            "seed_policy": str(seed_policy) if seed_policy is not None else None,
            "seed_count": 0,
            "current_seed": None,
            "selected_seed": None,
            "request_build_s": None,
            "initial_objective_s": None,
            "least_squares_s": None,
            "residual_eval_count": 0,
            "mean_residual_eval_s": None,
            "max_residual_eval_s": None,
            "optimizer_nfev": None,
            "optimizer_njev": None,
            "manual_pick_cache_rebuild_count": 0,
            "caked_projection_rebuild_count": 0,
            "dynamic_source_coordinate_recompute_count": 0,
            "fixed_source_resolved_count": 0,
            "matched_pair_count": 0,
            "missing_pair_count": 0,
            "fallback_entry_count": 0,
            "fallback_row_count": 0,
            "fallback_pair_count": 0,
            "point_only_projection_enabled": False,
            "reuse_generation_for_prescore_enabled": False,
        }

    def update_static(
        self,
        *,
        active_vars: Sequence[object] | None = None,
        seed_policy: object | None = None,
        runtime_cfg: Mapping[str, object] | None = None,
        manual_caked_fixed_row_count: int | None = None,
        bounded_budget_enabled: bool | None = None,
    ) -> None:
        updates: dict[str, object] = {}
        if active_vars is not None:
            updates["active_vars"] = [str(name) for name in active_vars]
        if seed_policy is not None:
            updates["seed_policy"] = str(seed_policy)
        if manual_caked_fixed_row_count is not None:
            updates["manual_caked_fixed_row_count"] = int(manual_caked_fixed_row_count)
        if bounded_budget_enabled is not None:
            updates["bounded_budget_enabled"] = bool(bounded_budget_enabled)
        if isinstance(runtime_cfg, Mapping):
            solver = runtime_cfg.get("solver")
            optimizer = runtime_cfg.get("optimizer")
            seed_search = runtime_cfg.get("seed_search")
            solver_map = solver if isinstance(solver, Mapping) else {}
            optimizer_map = optimizer if isinstance(optimizer, Mapping) else {}
            seed_search_map = seed_search if isinstance(seed_search, Mapping) else {}
            updates.update(
                {
                    "point_only_projection_enabled": bool(
                        solver_map.get(_HEADLESS_GEOMETRY_FIT_POINT_ONLY_FLAG, False)
                        or optimizer_map.get(_HEADLESS_GEOMETRY_FIT_POINT_ONLY_FLAG, False)
                    ),
                    "reuse_generation_for_prescore_enabled": bool(
                        seed_search_map.get("_reuse_generation_for_prescore", False)
                    ),
                    "seed_prescore_top_k": seed_search_map.get("prescore_top_k"),
                    "solver_max_nfev": solver_map.get("max_nfev"),
                    "optimizer_max_nfev": optimizer_map.get("max_nfev"),
                }
            )
        if updates:
            self.write(str(self.data.get("phase", "preflight")), **updates)

    def _merge_point_match_summary(self, summary: Mapping[str, object]) -> dict[str, object]:
        updates: dict[str, object] = {}
        for key in (
            "fixed_source_resolved_count",
            "matched_pair_count",
            "missing_pair_count",
            "fallback_entry_count",
            "fallback_row_count",
            "fallback_pair_count",
            "dynamic_source_coordinate_recompute_count",
        ):
            if key in summary:
                updates[key] = _headless_progress_int(summary.get(key), 0)
        for key in (
            "manual_pick_cache_rebuild_count",
            "caked_projection_rebuild_count",
        ):
            if key in summary:
                updates[key] = _headless_progress_int(summary.get(key), 0)
        seed_trace = summary.get("seed_multistart_trace")
        if isinstance(seed_trace, Mapping):
            if "seed_count" in seed_trace:
                updates["seed_count"] = _headless_progress_int(seed_trace.get("seed_count"), 0)
            if "selected_seed_index" in seed_trace:
                updates["selected_seed"] = seed_trace.get("selected_seed_index")
            if "optimizer_nfev" in seed_trace:
                updates["optimizer_nfev"] = seed_trace.get("optimizer_nfev")
            if "optimizer_njev" in seed_trace:
                updates["optimizer_njev"] = seed_trace.get("optimizer_njev")
        return updates

    def status(self, text: object) -> None:
        try:
            message = str(text).strip()
        except Exception:
            message = ""
        if not message:
            return
        phase = str(self.data.get("phase", "preflight"))
        if " prescore " in f" {message} " or "prescore total_seeds=" in message:
            phase = "seed_prescore"
        elif " eval=" in message or " seed " in message:
            phase = "selected_solve"
        self.write(phase, status_text=message)

    def live_update(self, payload: Mapping[str, object]) -> None:
        if not isinstance(payload, Mapping):
            return
        updates: dict[str, object] = {}
        if "evaluation_count" in payload:
            updates["residual_eval_count"] = _headless_progress_int(
                payload.get("evaluation_count"), 0
            )
            updates["optimizer_nfev"] = updates["residual_eval_count"]
        for source_key, dest_key in (
            ("mean_residual_eval_s", "mean_residual_eval_s"),
            ("max_residual_eval_s", "max_residual_eval_s"),
            ("last_residual_eval_s", "last_residual_eval_s"),
        ):
            if source_key in payload:
                updates[dest_key] = _headless_progress_float(payload.get(source_key))
        summary = payload.get("point_match_summary")
        if isinstance(summary, Mapping):
            updates.update(self._merge_point_match_summary(summary))
        updates["last_live_update"] = _headless_progress_live_payload(payload)
        self.write("selected_solve", **updates)

    def write(self, phase: str, **updates: object) -> None:
        if self.path is None:
            return
        phase_name = str(phase or self.data.get("phase") or "preflight")
        if phase_name not in _HEADLESS_GEOMETRY_FIT_PROGRESS_PHASES:
            phase_name = str(self.data.get("phase") or "preflight")
        self.data["phase"] = phase_name
        self.data["elapsed_s"] = float(max(0.0, time.monotonic() - self.started_at))
        self.data["pid"] = int(os.getpid())
        self.data.update({str(key): value for key, value in updates.items()})
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_name(f"{self.path.name}.tmp")
            tmp_path.write_text(
                json.dumps(_headless_progress_jsonable(self.data), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            tmp_path.replace(self.path)
        except Exception:
            pass


def write_headless_geometry_fit_progress(
    progress_path: str | Path | None,
    phase: str,
    **updates: object,
) -> None:
    writer = _HeadlessGeometryFitProgressWriter(progress_path)
    existing: dict[str, object] = {}
    if writer.path is not None and writer.path.exists():
        try:
            loaded = json.loads(writer.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            existing = {}
    writer.data.update(existing)
    writer.write(phase, **updates)


def _geometry_fit_recovery_artifacts_required(
    *,
    state_path: str | Path,
    active_var_names: Sequence[object] | None,
) -> bool:
    active_names = {str(name) for name in (active_var_names or ())}
    return {"gamma", "Gamma"}.issubset(active_names) and Path(state_path).expanduser().exists()


def _geometry_fit_recovery_float(value: object) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return float(number)


def _geometry_fit_recovery_pair(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
        return None
    first = _geometry_fit_recovery_float(value[0])
    second = _geometry_fit_recovery_float(value[1])
    if first is None or second is None:
        return None
    return first, second


def _geometry_fit_recovery_hkl(value: object) -> list[object]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_headless_progress_jsonable(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _geometry_fit_recovery_rows(
    final_summary: Mapping[str, object],
    progress_data: Mapping[str, object],
) -> list[dict[str, object]]:
    worst_rows = final_summary.get("worst_angular_residual_rows")
    if isinstance(worst_rows, Sequence) and not isinstance(worst_rows, (str, bytes)):
        rows = [dict(row) for row in worst_rows if isinstance(row, Mapping)]
        if rows:
            return rows

    live_update = progress_data.get("last_live_update")
    live_records = (
        live_update.get("live_cache_records") if isinstance(live_update, Mapping) else None
    )
    if not isinstance(live_records, Sequence) or isinstance(live_records, (str, bytes)):
        return []
    rows = []
    for record in live_records:
        if not isinstance(record, Mapping):
            continue
        predicted = _geometry_fit_recovery_pair(record.get("sim_refined_caked_deg"))
        if predicted is None:
            predicted = _geometry_fit_recovery_pair(record.get("sim_nominal_caked_deg"))
        rows.append(
            {
                "dataset_index": record.get("dataset_index"),
                "dataset_label": record.get("dataset_label"),
                "pair_id": record.get("pair_id"),
                "q_group_key": record.get("q_group_key"),
                "hkl": record.get("hkl"),
                "source_branch_index": record.get("source_branch_index"),
                "source_table_index": record.get("source_table_index"),
                "source_row_index": record.get("source_row_index"),
                "predicted_caked_deg": predicted,
            }
        )
    return rows


def _geometry_fit_recovery_initial_caked_by_pair(
    progress_data: Mapping[str, object],
) -> dict[str, tuple[float, float]]:
    live_update = progress_data.get("last_live_update")
    live_records = (
        live_update.get("live_cache_records") if isinstance(live_update, Mapping) else None
    )
    if not isinstance(live_records, Sequence) or isinstance(live_records, (str, bytes)):
        return {}
    caked_by_pair: dict[str, tuple[float, float]] = {}
    for record in live_records:
        if not isinstance(record, Mapping):
            continue
        pair_id = record.get("pair_id")
        caked = _geometry_fit_recovery_pair(record.get("sim_nominal_caked_deg"))
        if pair_id is not None and caked is not None:
            caked_by_pair[str(pair_id)] = caked
    return caked_by_pair


def _geometry_fit_recovery_row_identity(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "pair_id": row.get("pair_id") or row.get("manual_pair_id"),
        "q_group_key": _headless_progress_jsonable(row.get("q_group_key")),
        "hkl": _geometry_fit_recovery_hkl(row.get("hkl")),
        "branch": row.get("source_branch_index"),
        "source_table_index": row.get("source_table_index"),
        "source_row_index": row.get("source_row_index"),
    }


def _geometry_fit_recovery_label(row: Mapping[str, object]) -> str:
    identity = _geometry_fit_recovery_row_identity(row)
    pair_id = str(identity.get("pair_id") or "?")
    hkl = identity.get("hkl")
    branch = identity.get("branch")
    return f"{pair_id} hkl={hkl} b={branch}"


def _geometry_fit_recovery_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_headless_progress_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _geometry_fit_recovery_axis_limits(
    points: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    x_pad = max(1.0, 0.08 * x_span)
    y_pad = max(1.0, 0.08 * y_span)
    return (min(xs) - x_pad, max(xs) + x_pad), (min(ys) - y_pad, max(ys) + y_pad)


def _geometry_fit_recovery_report_label(
    *,
    state_path: Path,
    rows: Sequence[Mapping[str, object]],
    progress_data: Mapping[str, object],
) -> str:
    candidates = [state_path.stem]
    for row in rows:
        candidates.extend(
            str(row.get(key) or "")
            for key in ("dataset_label", "dataset/background", "background_path", "cif_path")
        )
    last_live_update = progress_data.get("last_live_update")
    if isinstance(last_live_update, Mapping):
        for record in last_live_update.get("live_cache_records", ()) or ():
            if isinstance(record, Mapping):
                candidates.extend(
                    str(record.get(key) or "")
                    for key in ("dataset_label", "background_path", "cif_path")
                )
    for candidate in candidates:
        if "bi2se3" in candidate.lower().replace("_", "").replace("-", ""):
            return "Bi2Se3"
    if state_path.stem.strip().lower() == "new4":
        return "Bi2Se3"
    return state_path.stem.strip() or "geometry fit"


def _plot_geometry_fit_recovery_overlay(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    initial_caked_by_pair: Mapping[str, tuple[float, float]],
    accepted: bool,
    report_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    plotted_points: list[tuple[float, float]] = []
    for row in rows:
        pair_id = str(row.get("pair_id") or row.get("manual_pair_id") or "")
        observed = _geometry_fit_recovery_pair(row.get("observed_caked_deg"))
        initial = initial_caked_by_pair.get(pair_id)
        final = _geometry_fit_recovery_pair(row.get("predicted_caked_deg"))
        label = _geometry_fit_recovery_label(row)
        if observed is not None:
            plotted_points.append(observed)
            ax.scatter(
                observed[0],
                observed[1],
                marker="o",
                c="#1f77b4",
                s=56,
                label="manual/background QR",
            )
        if initial is not None:
            plotted_points.append(initial)
            ax.scatter(
                initial[0],
                initial[1],
                marker="s",
                c="#4c4c4c",
                s=50,
                label="initial objective simulation QR",
            )
        if final is not None:
            plotted_points.append(final)
            ax.scatter(
                final[0],
                final[1],
                marker="^",
                c="#2ca02c" if accepted else "#d62728",
                s=62,
                label=(
                    "final accepted objective simulation QR"
                    if accepted
                    else "final rejected objective simulation QR"
                ),
            )
            ax.annotate(label, xy=final, xytext=(4, 4), textcoords="offset points", fontsize=7)
        if initial is not None and final is not None:
            ax.annotate(
                "",
                xy=final,
                xytext=initial,
                arrowprops={"arrowstyle": "->", "color": "#7f7f7f", "lw": 1.0},
            )
    limits = _geometry_fit_recovery_axis_limits(plotted_points)
    if limits is not None:
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
    else:
        ax.text(0.5, 0.5, "No caked QR rows available", ha="center", va="center")
    ax.set_xlabel("two theta (deg)")
    ax.set_ylabel("phi (deg)")
    ax.set_title(f"{report_label} gamma/Gamma QR fit: initial vs final")
    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels, strict=False):
        unique.setdefault(label, handle)
    if unique:
        ax.legend(unique.values(), unique.keys(), loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _geometry_fit_recovery_worst_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    failure_classification: object,
) -> list[dict[str, object]]:
    worst_rows: list[dict[str, object]] = []
    for rank, row in enumerate(rows, start=1):
        dataset = row.get("dataset_label")
        if dataset is None and row.get("dataset_index") is not None:
            dataset = f"bg{row.get('dataset_index')}"
        worst_rows.append(
            {
                "rank": rank,
                "dataset/background": dataset,
                "pair_id": row.get("pair_id") or row.get("manual_pair_id"),
                "q_group_key": _headless_progress_jsonable(row.get("q_group_key")),
                "hkl": _geometry_fit_recovery_hkl(row.get("hkl")),
                "branch": row.get("source_branch_index"),
                "observed_caked_deg": _headless_progress_jsonable(row.get("observed_caked_deg")),
                "predicted_caked_deg": _headless_progress_jsonable(row.get("predicted_caked_deg")),
                "delta_two_theta_deg": row.get("delta_two_theta_deg"),
                "wrapped_delta_phi_deg": row.get("wrapped_delta_phi_deg"),
                "angular_residual_norm_deg": row.get("angular_residual_norm_deg"),
                "same_q_group_hkl_candidate_count": row.get("same_q_group_hkl_candidate_count"),
                "nearest_same_q_group_hkl_candidate_residual_norm_deg": row.get(
                    "nearest_same_q_group_hkl_candidate_residual_norm_deg"
                ),
                "branch_swap_would_help": bool(row.get("branch_swap_would_help", False)),
                "failure_classification": failure_classification,
            }
        )
    return worst_rows


def _plot_geometry_fit_recovery_worst_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    failure_classification: object,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    plotted_points: list[tuple[float, float]] = []
    for rank, row in enumerate(rows, start=1):
        observed = _geometry_fit_recovery_pair(row.get("observed_caked_deg"))
        predicted = _geometry_fit_recovery_pair(row.get("predicted_caked_deg"))
        if observed is not None:
            plotted_points.append(observed)
            ax.scatter(observed[0], observed[1], marker="o", c="#1f77b4", s=58)
            ax.annotate(
                f"{rank}",
                xy=observed,
                xytext=(-8, -8),
                textcoords="offset points",
                fontsize=8,
            )
        if predicted is not None:
            plotted_points.append(predicted)
            ax.scatter(predicted[0], predicted[1], marker="^", c="#d62728", s=68)
            ax.annotate(
                _geometry_fit_recovery_label(row),
                xy=predicted,
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )
        if observed is not None and predicted is not None:
            ax.annotate(
                "",
                xy=predicted,
                xytext=observed,
                arrowprops={"arrowstyle": "->", "color": "#7f7f7f", "lw": 1.0},
            )
    limits = _geometry_fit_recovery_axis_limits(plotted_points)
    if limits is not None:
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
    else:
        ax.text(0.5, 0.5, "No worst residual rows available", ha="center", va="center")
    ax.set_xlabel("two theta (deg)")
    ax.set_ylabel("phi (deg)")
    ax.set_title(f"Worst QR residual rows ({failure_classification or 'unclassified'})")
    ax.scatter([], [], marker="o", c="#1f77b4", s=58, label="observed/manual QR")
    ax.scatter([], [], marker="^", c="#d62728", s=68, label="predicted QR")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_headless_geometry_fit_single_step_artifacts(
    *,
    state_path: Path,
    background_index: int,
    output_root: Path,
    report_label: str,
) -> Mapping[str, object]:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "debug"
        / "visualize_new4_qr_fit_coordinates.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_ra_sim_qr_fit_coordinates",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load QR coordinate audit script: {module_path}")
    coordinate_audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(coordinate_audit)

    return coordinate_audit.run_coordinate_audit(
        state_path=state_path,
        background_index=int(background_index),
        output_root=output_root,
        params_mode="base",
        active_vars=["gamma", "Gamma"],
        single_step_detector_angle_audit=True,
        max_angle_step_deg=5.0,
        fd_step_deg=0.05,
        report_label=report_label,
    )


def _headless_geometry_fit_recovery_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "single_step_json": output_dir / GEOMETRY_FIT_RECOVERY_SINGLE_STEP_JSON,
        "single_step_png": output_dir / GEOMETRY_FIT_RECOVERY_SINGLE_STEP_PNG,
        "single_step_csv": output_dir / GEOMETRY_FIT_RECOVERY_SINGLE_STEP_CSV,
        "full_fit_json": output_dir / GEOMETRY_FIT_RECOVERY_FULL_OVERLAY_JSON,
        "full_fit_png": output_dir / GEOMETRY_FIT_RECOVERY_FULL_OVERLAY_PNG,
        "worst_rows_json": output_dir / GEOMETRY_FIT_RECOVERY_WORST_ROWS_JSON,
        "worst_rows_png": output_dir / GEOMETRY_FIT_RECOVERY_WORST_ROWS_PNG,
    }


def _write_headless_geometry_fit_recovery_artifacts(
    *,
    state_path: str | Path,
    output_dir: str | Path,
    background_index: int,
    active_var_names: Sequence[object],
    accepted: bool,
    rejection_reason: object,
    final_summary: Mapping[str, object] | None,
    progress_data: Mapping[str, object],
    initial_params: Mapping[str, object],
    final_params: Mapping[str, object],
) -> dict[str, object]:
    if not _geometry_fit_recovery_artifacts_required(
        state_path=state_path,
        active_var_names=active_var_names,
    ):
        return {}

    state_file = Path(state_path).expanduser().resolve()
    state_provenance = _headless_geometry_fit_state_provenance(state_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = _headless_geometry_fit_recovery_paths(output_path)
    summary = dict(final_summary or {})
    rows = _geometry_fit_recovery_rows(summary, progress_data)
    failure_classification = (
        summary.get("dynamic_angular_failure_classification")
        or summary.get("failure_classification")
        or rejection_reason
    )
    initial_caked_by_pair = _geometry_fit_recovery_initial_caked_by_pair(progress_data)
    report_label = _geometry_fit_recovery_report_label(
        state_path=state_file,
        rows=rows,
        progress_data=progress_data,
    )

    single_step_report = _write_headless_geometry_fit_single_step_artifacts(
        state_path=state_file,
        background_index=int(background_index),
        output_root=output_path,
        report_label=report_label,
    )
    full_fit_report = {
        "json_authoritative": True,
        "png_diagnostic_only": True,
        "recovery_run_label": report_label,
        **state_provenance,
        "full_fit_success": bool(accepted),
        "geometry_updated": bool(accepted),
        "gamma_before": initial_params.get("gamma"),
        "Gamma_before": initial_params.get("Gamma"),
        "gamma_after": final_params.get("gamma"),
        "Gamma_after": final_params.get("Gamma"),
        "raw_angular_rms_deg": summary.get("raw_angular_rms_deg") or summary.get("final_rms_deg"),
        "raw_angular_max_deg": summary.get("raw_angular_max_deg") or summary.get("final_max_deg"),
        "qr_fit_resolved_count": summary.get("qr_fit_resolved_count")
        or summary.get("fixed_source_resolved_count"),
        "qr_fit_expected_count": summary.get("qr_fit_expected_count"),
        "qr_fit_missing_pairs": list(summary.get("qr_fit_missing_pairs", ()) or ()),
        "source_authority_mismatch_count": summary.get("source_authority_mismatch_count"),
        "visual_objective_surface_mismatch_count": summary.get(
            "visual_objective_surface_mismatch_count"
        ),
        "objective_param_sensitivity_status": summary.get("objective_param_sensitivity_status"),
        "failure_classification": failure_classification,
        "plotted_row_count": int(len(rows)),
        "plotted_row_identities": [_geometry_fit_recovery_row_identity(row) for row in rows],
        "rows": rows,
    }
    _geometry_fit_recovery_json(paths["full_fit_json"], full_fit_report)
    _plot_geometry_fit_recovery_overlay(
        paths["full_fit_png"],
        rows,
        initial_caked_by_pair=initial_caked_by_pair,
        accepted=bool(accepted),
        report_label=report_label,
    )

    required_pngs = [paths["single_step_png"], paths["full_fit_png"]]
    artifact_paths: dict[str, object] = {
        "single_step_json": paths["single_step_json"],
        "single_step_png": paths["single_step_png"],
        "single_step_csv": paths["single_step_csv"],
        "full_fit_json": paths["full_fit_json"],
        "full_fit_png": paths["full_fit_png"],
        "worst_rows_json": None,
        "worst_rows_png": None,
    }
    if not accepted:
        worst_rows = _geometry_fit_recovery_worst_rows(
            rows,
            failure_classification=failure_classification,
        )
        worst_report = {
            "json_authoritative": True,
            "png_diagnostic_only": True,
            **state_provenance,
            "failure_classification": failure_classification,
            "row_count": int(len(worst_rows)),
            "rows": worst_rows,
        }
        _geometry_fit_recovery_json(paths["worst_rows_json"], worst_report)
        _plot_geometry_fit_recovery_worst_rows(
            paths["worst_rows_png"],
            rows,
            failure_classification=failure_classification,
        )
        artifact_paths["worst_rows_json"] = paths["worst_rows_json"]
        artifact_paths["worst_rows_png"] = paths["worst_rows_png"]
        required_pngs.append(paths["worst_rows_png"])

    missing_required_pngs = [
        path for path in required_pngs if not path.exists() or path.stat().st_size <= 0
    ]
    if missing_required_pngs:
        missing_text = ", ".join(str(path) for path in missing_required_pngs)
        raise RuntimeError(
            f"Geometry fit recovery artifact generation missing required PNGs: {missing_text}"
        )

    artifact_payload = {
        "geometry_fit_recovery_artifact_status": "pass",
        "geometry_fit_recovery_run_label": report_label,
        **state_provenance,
        "geometry_fit_recovery_required_pngs": required_pngs,
        "geometry_fit_recovery_artifacts": artifact_paths,
        "single_step_status": single_step_report.get("status"),
    }
    artifact_payload.update(artifact_paths)
    return artifact_payload


def _headless_runtime_solver_mapping(runtime_cfg: Mapping[str, object]) -> dict[str, object]:
    solver_cfg = runtime_cfg.get("solver")
    return dict(solver_cfg) if isinstance(solver_cfg, Mapping) else {}


def normalize_headless_geometry_fit_seed_policy(seed_policy: object | None) -> str | None:
    """Normalize one optional headless seed-policy override."""

    try:
        return gui_geometry_fit.normalize_geometry_fit_seed_policy(seed_policy)
    except ValueError as exc:
        allowed = ", ".join(sorted(_HEADLESS_GEOMETRY_FIT_SEED_POLICIES))
        raise ValueError(
            f"Unsupported headless geometry-fit seed policy {str(seed_policy).strip()!r}; "
            f"expected one of: {allowed}."
        ) from exc


def _apply_headless_geometry_fit_seed_policy(
    runtime_cfg: dict[str, object],
    seed_policy: str | None,
) -> None:
    """Apply one opt-in headless seed policy without changing saved-state defaults."""

    gui_geometry_fit.apply_geometry_fit_seed_policy(runtime_cfg, seed_policy)


def _headless_geometry_runtime_is_saved_manual_caked_candidate(
    runtime_cfg: Mapping[str, object],
) -> bool:
    if str(runtime_cfg.get("projection_view_mode", "")).strip().lower() != "caked":
        return False
    solver = _headless_runtime_solver_mapping(runtime_cfg)
    return bool(
        solver.get("manual_point_fit_mode", False)
        and solver.get("dynamic_point_geometry_fit", False)
    )


def _headless_geometry_dataset_entries(prepared_run: object) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for dataset in (
        getattr(prepared_run, "current_dataset", None),
        *(getattr(prepared_run, "dataset_infos", None) or ()),
    ):
        if not isinstance(dataset, Mapping):
            continue
        for row_key in (
            "measured_for_fit",
            "initial_pairs_display",
            "manual_point_pairs",
            "provider_pairs",
        ):
            for entry in dataset.get(row_key, ()) or ():
                if isinstance(entry, Mapping):
                    entries.append(dict(entry))
    return entries


def _headless_geometry_entry_has_fixed_manual_caked_qr(entry: Mapping[str, object]) -> bool:
    return gui_geometry_fit.geometry_fit_entry_has_fixed_manual_caked_qr(entry)


def _infer_headless_saved_manual_caked_defaults(
    active_var_names: Sequence[str] | None,
    seed_policy: str | None,
    manual_pair_rows: Sequence[Mapping[str, object]] | None,
) -> tuple[list[str] | None, str | None, bool]:
    saved_manual_caked = any(
        isinstance(entry, Mapping) and _headless_geometry_entry_has_fixed_manual_caked_qr(entry)
        for entry in (manual_pair_rows or ())
    )
    if not saved_manual_caked:
        return (
            list(active_var_names) if active_var_names is not None else None,
            seed_policy,
            False,
        )
    resolved_active = (
        list(active_var_names)
        if active_var_names is not None
        else list(_HEADLESS_GEOMETRY_FIT_SAVED_MANUAL_CAKED_DEFAULT_ACTIVE_VAR_NAMES)
    )
    resolved_seed_policy = (
        seed_policy
        if seed_policy is not None
        else HEADLESS_GEOMETRY_FIT_SEED_POLICY_DIRECT
    )
    return resolved_active, resolved_seed_policy, True


def _headless_geometry_fixed_manual_caked_qr_row_count(prepared_run: object) -> int:
    return gui_geometry_fit.geometry_fit_fixed_manual_caked_qr_row_count(
        current_dataset=getattr(prepared_run, "current_dataset", None),
        dataset_infos=getattr(prepared_run, "dataset_infos", None),
    )


def _set_headless_caked_point_only_projection(runtime_cfg: dict[str, object]) -> None:
    solver = _headless_runtime_solver_mapping(runtime_cfg)
    solver[_HEADLESS_GEOMETRY_FIT_POINT_ONLY_FLAG] = True
    runtime_cfg["solver"] = solver
    optimizer_cfg = runtime_cfg.get("optimizer")
    optimizer = dict(optimizer_cfg) if isinstance(optimizer_cfg, Mapping) else dict(solver)
    optimizer[_HEADLESS_GEOMETRY_FIT_POINT_ONLY_FLAG] = True
    runtime_cfg["optimizer"] = optimizer


def _cap_headless_solver_nfev(runtime_cfg: dict[str, object], max_nfev: int) -> None:
    for section_name in ("solver", "optimizer"):
        section_cfg = runtime_cfg.get(section_name)
        section = dict(section_cfg) if isinstance(section_cfg, Mapping) else {}
        current_raw = section.get("max_nfev")
        try:
            current = int(current_raw)
        except Exception:
            current = int(max_nfev)
        if current <= 0:
            current = int(max_nfev)
        section["max_nfev"] = min(int(current), int(max_nfev))
        runtime_cfg[section_name] = section


def _apply_headless_saved_manual_caked_lean_runtime(
    runtime_cfg: dict[str, object],
    *,
    max_nfev: int,
    seed_multistart: bool,
) -> None:
    solver = _headless_runtime_solver_mapping(runtime_cfg)
    solver["manual_point_fit_mode"] = True
    solver["dynamic_point_geometry_fit"] = True
    solver["seed_multistart"] = bool(seed_multistart)
    solver["seed_multistart_enabled"] = bool(seed_multistart)
    solver[_HEADLESS_GEOMETRY_FIT_POINT_ONLY_FLAG] = True
    solver["max_nfev"] = int(max_nfev)
    solver["min_max_nfev"] = 1
    solver["loss"] = "linear"
    solver["f_scale_px"] = 1.0
    solver["weighted_matching"] = False
    solver["use_measurement_uncertainty"] = False
    solver["anisotropic_measurement_uncertainty"] = False
    solver["q_group_line_constraints"] = False
    solver["q_group_line_constraints_enabled"] = False
    solver["_headless_accept_caked_angular_metric_without_pixel_threshold"] = True
    solver["workers"] = solver.get("workers", "auto")
    solver["parallel_mode"] = "off"
    solver["worker_numba_threads"] = 0
    runtime_cfg["solver"] = solver
    runtime_cfg["optimizer"] = dict(solver)
    seed_search_cfg = runtime_cfg.get("seed_search")
    seed_search = dict(seed_search_cfg) if isinstance(seed_search_cfg, Mapping) else {}
    seed_search["enabled"] = bool(seed_multistart)
    runtime_cfg["seed_search"] = seed_search
    runtime_cfg["projection_view_mode"] = "caked"
    runtime_cfg["use_numba"] = bool(runtime_cfg.get("use_numba", False))
    runtime_cfg["allow_unsafe_runtime"] = False
    for feature_key in ("full_beam_polish", "ridge_refinement", "image_refinement"):
        runtime_cfg.pop(feature_key, None)
    discrete = (
        dict(runtime_cfg.get("discrete_modes", {}))
        if isinstance(runtime_cfg.get("discrete_modes"), Mapping)
        else {}
    )
    discrete["enabled"] = False
    runtime_cfg["discrete_modes"] = discrete
    ident = (
        dict(runtime_cfg.get("identifiability", {}))
        if isinstance(runtime_cfg.get("identifiability"), Mapping)
        else {}
    )
    ident["enabled"] = False
    ident.pop("auto_freeze", None)
    ident.pop("selective_thaw", None)
    ident.pop("adaptive_regularization", None)
    runtime_cfg["identifiability"] = ident


def _enforce_headless_gamma_bounds(
    runtime_cfg: dict[str, object],
    active_var_names: Sequence[object],
) -> None:
    active_names = {str(name) for name in active_var_names}
    gamma_names = [name for name in ("gamma", "Gamma") if name in active_names]
    if not gamma_names:
        return
    bounds_cfg = runtime_cfg.get("bounds")
    bounds = dict(bounds_cfg) if isinstance(bounds_cfg, Mapping) else {}
    for name in gamma_names:
        bounds[name] = [-90.0, 90.0]
    runtime_cfg["bounds"] = bounds


def _apply_headless_saved_manual_caked_budget(
    runtime_cfg: dict[str, object],
    *,
    seed_policy: str | None,
    prepared_run: object,
    active_var_names: Sequence[object],
    manual_pair_rows: Sequence[Mapping[str, object]] | None = None,
) -> tuple[bool, int]:
    return gui_geometry_fit.apply_saved_manual_caked_geometry_fit_budget(
        runtime_cfg,
        seed_policy=seed_policy,
        active_var_names=active_var_names,
        current_dataset=getattr(prepared_run, "current_dataset", None),
        dataset_infos=getattr(prepared_run, "dataset_infos", None),
        manual_pair_rows=manual_pair_rows,
    )


def run_headless_geometry_fit(
    saved_state: dict[str, object],
    *,
    state_path: str | Path,
    downloads_dir: str | Path | None = None,
    stamp: str | None = None,
    active_var_names: Sequence[object] | str | None = None,
    seed_policy: object | None = None,
    progress_path: str | Path | None = None,
    weighted_event_workers: int | None = None,
    background_subtraction_mode: object | None = None,
    background_subtraction_scale: object | None = None,
    background_subtraction_diagnostics: bool | None = None,
    background_subtraction_phi_block_overrides: Mapping[str, object] | None = None,
) -> HeadlessGeometryFitResult:
    """Run the geometry fit described by ``saved_state`` and return the updated state."""

    if not isinstance(saved_state, dict):
        raise ValueError("Saved GUI state must be a dictionary.")
    resolved_active_var_names = normalize_headless_geometry_fit_active_var_names(
        active_var_names
    )
    resolved_seed_policy = normalize_headless_geometry_fit_seed_policy(seed_policy)
    progress_writer = _HeadlessGeometryFitProgressWriter(
        progress_path,
        active_vars=resolved_active_var_names,
        seed_policy=resolved_seed_policy,
    )
    progress_writer.write(
        "preflight",
        state_path=Path(state_path),
        **_headless_geometry_fit_state_provenance(state_path),
        downloads_dir=downloads_dir,
    )
    weighted_event_worker_count = None
    if weighted_event_workers is not None:
        try:
            weighted_event_worker_count = int(weighted_event_workers)
        except (TypeError, ValueError) as exc:
            raise ValueError("Weighted-event workers must be a positive integer.") from exc
        if weighted_event_worker_count < 1:
            raise ValueError("Weighted-event workers must be a positive integer.")

    diffraction = _load_simulation_diffraction()
    fit_runtime = _load_fitting_runtime()
    state_types = _load_gui_state_types()
    defaults = _build_runtime_defaults(saved_state)
    var_store = _build_var_store(saved_state, defaults)
    _headless_background_subtraction_config(
        saved_state,
        defaults,
        mode_override=background_subtraction_mode,
        scale_override=background_subtraction_scale,
        diagnostics_override=background_subtraction_diagnostics,
        phi_block_overrides=background_subtraction_phi_block_overrides,
    )
    geometry_state = (
        saved_state.get("geometry", {}) if isinstance(saved_state.get("geometry"), dict) else {}
    )
    pairs_by_background = _restore_manual_pairs(
        defaults.osc_files,
        geometry_state.get("manual_pairs", []),
    )

    background_state = state_types.BackgroundRuntimeState(
        osc_files=list(defaults.osc_files),
        background_images=[None] * len(defaults.osc_files),
        background_images_native=[None] * len(defaults.osc_files),
        background_images_display=[None] * len(defaults.osc_files),
        current_background_index=int(defaults.current_background_index),
        visible=bool((saved_state.get("flags", {}) or {}).get("background_visible", True)),
        backend_rotation_k=int(defaults.background_flags["backend_rotation_k"]),
        backend_flip_x=bool(defaults.background_flags["backend_flip_x"]),
        backend_flip_y=bool(defaults.background_flags["backend_flip_y"]),
    )
    simulation_runtime_state = state_types.SimulationRuntimeState()
    saved_peak_records = geometry_state.get("peak_records")
    if isinstance(saved_peak_records, list):
        _replace_gui_state_peak_cache(
            simulation_runtime_state,
            saved_peak_records,
        )

    structure_state, _atom_site_override_state, active_cif_path, nominal_n2 = _load_structure_model(
        defaults,
        saved_state,
        var_store,
        simulation_runtime_state,
    )

    downloads_path = (
        Path(downloads_dir)
        if downloads_dir is not None
        else Path(state_path).expanduser().resolve().parent
    )
    downloads_path.mkdir(parents=True, exist_ok=True)
    fit_stamp = str(stamp or Path(state_path).stem)

    def _load_background_by_index(index: int) -> tuple[np.ndarray, np.ndarray]:
        loaded = gui_background.load_background_image_by_index(
            int(index),
            osc_files=background_state.osc_files,
            background_images=background_state.background_images,
            background_images_native=background_state.background_images_native,
            background_images_display=background_state.background_images_display,
            display_rotate_k=DISPLAY_ROTATE_K,
            read_osc=read_osc,
        )
        background_state.background_images = list(loaded["background_images"])
        background_state.background_images_native = list(loaded["background_images_native"])
        background_state.background_images_display = list(loaded["background_images_display"])
        background_state.current_background_index = int(index)
        background_state.current_background_image = np.asarray(loaded["background_image"])
        background_state.current_background_display = np.asarray(loaded["background_display"])
        return (
            background_state.current_background_image,
            background_state.current_background_display,
        )

    _load_background_by_index(background_state.current_background_index)

    def _current_background_native() -> np.ndarray:
        native, _display = _load_background_by_index(background_state.current_background_index)
        return native

    def _current_background_display() -> np.ndarray:
        _native, display = _load_background_by_index(background_state.current_background_index)
        return display

    def _pairs_for_index(index: int) -> list[dict[str, object]]:
        return gui_manual_geometry.geometry_manual_pairs_for_index(
            int(index),
            pairs_by_background=pairs_by_background,
        )

    theta_defaults = {"theta_initial": defaults.defaults["theta_initial"]}
    theta_controls: dict[str, object] = {}
    geometry_fit_selection_var = var_store["geometry_fit_background_selection_var"]
    background_theta_list_var = var_store["background_theta_list_var"]
    geometry_theta_offset_var = var_store["geometry_theta_offset_var"]
    theta_initial_var = var_store["theta_initial_var"]

    def _current_geometry_fit_background_indices(*, strict: bool = False) -> list[int]:
        return gui_background_theta.current_geometry_fit_background_indices(
            osc_files=defaults.osc_files,
            current_background_index=background_state.current_background_index,
            geometry_fit_background_selection_var=geometry_fit_selection_var,
            strict=strict,
        )

    def _geometry_fit_uses_shared_theta_offset(
        selected_indices: list[int] | None = None,
    ) -> bool:
        return gui_background_theta.geometry_fit_uses_shared_theta_offset(
            selected_indices,
            osc_files=defaults.osc_files,
            current_background_index=background_state.current_background_index,
            geometry_fit_background_selection_var=geometry_fit_selection_var,
        )

    def _current_geometry_theta_offset(*, strict: bool = False) -> float:
        return gui_background_theta.current_geometry_theta_offset(
            geometry_theta_offset_var=geometry_theta_offset_var,
            strict=strict,
        )

    def _current_background_theta_values(*, strict_count: bool = False) -> list[float]:
        return gui_background_theta.current_background_theta_values(
            osc_files=defaults.osc_files,
            theta_initial_var=theta_initial_var,
            defaults=theta_defaults,
            theta_initial=defaults.defaults["theta_initial"],
            background_theta_list_var=background_theta_list_var,
            strict_count=strict_count,
        )

    def _background_theta_for_index(index: int, *, strict_count: bool = False) -> float:
        return gui_background_theta.background_theta_for_index(
            int(index),
            osc_files=defaults.osc_files,
            theta_initial_var=theta_initial_var,
            defaults=theta_defaults,
            theta_initial=defaults.defaults["theta_initial"],
            background_theta_list_var=background_theta_list_var,
            geometry_theta_offset_var=geometry_theta_offset_var,
            geometry_fit_background_selection_var=geometry_fit_selection_var,
            current_background_index=background_state.current_background_index,
            strict_count=strict_count,
        )

    def _apply_background_theta_metadata(
        *,
        trigger_update: bool = False,
        sync_live_theta: bool = True,
    ) -> bool:
        return gui_background_theta.apply_background_theta_metadata(
            osc_files=defaults.osc_files,
            current_background_index=background_state.current_background_index,
            theta_initial_var=theta_initial_var,
            defaults=theta_defaults,
            theta_initial=defaults.defaults["theta_initial"],
            background_theta_list_var=background_theta_list_var,
            geometry_theta_offset_var=geometry_theta_offset_var,
            geometry_fit_background_selection_var=geometry_fit_selection_var,
            fit_theta_checkbutton=None,
            theta_controls=theta_controls,
            set_background_file_status_text=None,
            schedule_update=None,
            progress_label=None,
            trigger_update=trigger_update,
            sync_live_theta=sync_live_theta,
        )

    def _apply_geometry_fit_background_selection(
        *,
        trigger_update: bool = False,
        sync_live_theta: bool = True,
    ) -> bool:
        return gui_background_theta.apply_geometry_fit_background_selection(
            osc_files=defaults.osc_files,
            current_background_index=background_state.current_background_index,
            theta_initial_var=theta_initial_var,
            defaults=theta_defaults,
            theta_initial=defaults.defaults["theta_initial"],
            background_theta_list_var=background_theta_list_var,
            geometry_theta_offset_var=geometry_theta_offset_var,
            geometry_fit_background_selection_var=geometry_fit_selection_var,
            fit_theta_checkbutton=None,
            theta_controls=theta_controls,
            set_background_file_status_text=None,
            schedule_update=None,
            progress_label_geometry=None,
            trigger_update=trigger_update,
            sync_live_theta=sync_live_theta,
        )

    solve_q_steps = _coerce_int(
        var_store["solve_q_steps_var"].get(),
        defaults.defaults["solve_q_steps"],
        minimum=32,
    )
    solve_q_rel_tol = float(
        np.clip(
            _coerce_float(
                var_store["solve_q_rel_tol_var"].get(),
                defaults.defaults["solve_q_rel_tol"],
            ),
            1.0e-6,
            5.0e-2,
        )
    )
    nominal_lambda = float(defaults.lambda_angstrom)
    calc_runtime = _load_calculation_runtime()
    n2_source_meta = calc_runtime._normalize_n2_source_meta(("cif_path", active_cif_path))
    n2_wavelength_snapshot = calc_runtime._n2_wavelength_snapshot_from_angstrom(
        np.array([nominal_lambda], dtype=np.float64)
    )
    mosaic_params = {
        "beam_x_array": np.zeros(1, dtype=np.float64),
        "beam_y_array": np.zeros(1, dtype=np.float64),
        "theta_array": np.zeros(1, dtype=np.float64),
        "phi_array": np.zeros(1, dtype=np.float64),
        "wavelength_array": np.array([nominal_lambda], dtype=np.float64),
        "wavelength_i_array": np.array([nominal_lambda], dtype=np.float64),
        "n2_sample_array": np.array([nominal_n2], dtype=np.complex128),
        "_n2_sample_array_source": n2_source_meta,
        "_n2_sample_array_wavelength_snapshot": n2_wavelength_snapshot,
        "sigma_mosaic_deg": _coerce_float(
            var_store["sigma_mosaic_var"].get(),
            defaults.defaults["sigma_mosaic_deg"],
        ),
        "gamma_mosaic_deg": _coerce_float(
            var_store["gamma_mosaic_var"].get(),
            defaults.defaults["gamma_mosaic_deg"],
        ),
        "eta": _coerce_float(var_store["eta_var"].get(), defaults.defaults["eta"]),
        "solve_q_steps": solve_q_steps,
        "solve_q_rel_tol": solve_q_rel_tol,
        "solve_q_mode": _resolve_solve_q_mode(var_store["solve_q_mode_var"].get()),
    }

    value_callbacks = gui_geometry_fit.build_runtime_geometry_fit_value_callbacks(
        gui_geometry_fit.GeometryFitRuntimeValueBindings(
            fit_zb_var=var_store["fit_zb_var"],
            fit_zs_var=var_store["fit_zs_var"],
            fit_theta_var=var_store["fit_theta_var"],
            fit_psi_z_var=var_store["fit_psi_z_var"],
            fit_chi_var=var_store["fit_chi_var"],
            fit_cor_var=var_store["fit_cor_var"],
            fit_gamma_var=var_store["fit_gamma_var"],
            fit_Gamma_var=var_store["fit_Gamma_var"],
            fit_dist_var=var_store["fit_dist_var"],
            fit_a_var=var_store["fit_a_var"],
            fit_c_var=var_store["fit_c_var"],
            fit_center_x_var=var_store["fit_center_x_var"],
            fit_center_y_var=var_store["fit_center_y_var"],
            zb_var=var_store["zb_var"],
            zs_var=var_store["zs_var"],
            theta_initial_var=var_store["theta_initial_var"],
            psi_z_var=var_store["psi_z_var"],
            chi_var=var_store["chi_var"],
            cor_angle_var=var_store["cor_angle_var"],
            sample_width_var=var_store["sample_width_var"],
            sample_length_var=var_store["sample_length_var"],
            sample_depth_var=var_store["sample_depth_var"],
            gamma_var=var_store["gamma_var"],
            Gamma_var=var_store["Gamma_var"],
            corto_detector_var=var_store["corto_detector_var"],
            a_var=var_store["a_var"],
            c_var=var_store["c_var"],
            center_x_var=var_store["center_x_var"],
            center_y_var=var_store["center_y_var"],
            debye_x_var=var_store["debye_x_var"],
            debye_y_var=var_store["debye_y_var"],
            geometry_theta_offset_var=geometry_theta_offset_var,
            current_background_index=lambda: background_state.current_background_index,
            geometry_fit_uses_shared_theta_offset=_geometry_fit_uses_shared_theta_offset,
            current_geometry_theta_offset=_current_geometry_theta_offset,
            background_theta_for_index=_background_theta_for_index,
            build_mosaic_params=lambda: dict(mosaic_params),
            current_optics_mode_flag=lambda: _resolve_optics_mode_flag(
                var_store["optics_mode_var"].get()
            ),
            lambda_value=nominal_lambda,
            psi=float(defaults.psi_deg),
            n2=lambda: nominal_n2,
            pixel_size_value=float(defaults.pixel_size_m),
        )
    )

    def _process_peaks_parallel_for_headless(*args, **kwargs):
        call_kwargs = dict(kwargs)
        if (
            weighted_event_worker_count is not None
            and call_kwargs.get("numba_thread_count") is None
        ):
            call_kwargs["numba_thread_count"] = int(weighted_event_worker_count)
        return diffraction.process_peaks_parallel(*args, **call_kwargs)

    simulation_callbacks = (
        gui_geometry_q_group_manager.make_runtime_geometry_fit_simulation_callbacks(
            process_peaks_parallel=_process_peaks_parallel_for_headless,
            hit_tables_to_max_positions=diffraction.hit_tables_to_max_positions,
            native_sim_to_display_coords=lambda col, row, image_shape: (
                gui_geometry_overlay.native_sim_to_display_coords(
                    col,
                    row,
                    image_shape,
                    sim_display_rotate_k=SIM_DISPLAY_ROTATE_K,
                )
            ),
            primary_a_factory=lambda: _coerce_float(
                var_store["a_var"].get(), defaults.defaults["a"]
            ),
            primary_c_factory=lambda: _coerce_float(
                var_store["c_var"].get(), defaults.defaults["c"]
            ),
            default_source_label="primary",
            round_pixel_centers=True,
            default_solve_q_steps=solve_q_steps,
            default_solve_q_rel_tol=solve_q_rel_tol,
            default_solve_q_mode=_resolve_solve_q_mode(var_store["solve_q_mode_var"].get()),
            prefer_safe_python_runner=True,
        )
    )

    def _native_detector_coords_to_live_bundle_detector_coords(
        col: float,
        row: float,
    ) -> tuple[float | None, float | None]:
        shape = tuple(int(v) for v in np.asarray(_current_background_native()).shape[:2])
        if len(shape) < 2 or min(shape) <= 0:
            return None, None
        return gui_geometry_overlay.rotate_point_for_display(
            float(col),
            float(row),
            shape,
            DISPLAY_ROTATE_K,
        )

    def _live_bundle_detector_coords_to_background_display_coords(
        col: float,
        row: float,
    ) -> tuple[float | None, float | None]:
        try:
            col_val = float(col)
            row_val = float(row)
        except Exception:
            return None, None
        if not (np.isfinite(col_val) and np.isfinite(row_val)):
            return None, None
        return float(col_val), float(row_val)

    caked_views_by_background: dict[int, dict[str, object]] = {}
    caked_projection_payloads_by_background: dict[int, dict[str, object]] = {}

    def _manual_current_background_uses_caked_space() -> bool:
        try:
            background_idx = int(background_state.current_background_index)
        except Exception:
            return False
        return gui_geometry_fit.geometry_manual_pairs_use_caked_fit_space(
            _pairs_for_index(background_idx)
        )

    def _signature_numeric(value: object) -> object:
        try:
            parsed = float(value)
        except Exception:
            return None
        if not np.isfinite(parsed):
            return None
        return round(float(parsed), 9)

    def _signature_summary(signature: object) -> str | None:
        if signature is None:
            return None
        text = repr(signature)
        return text if len(text) <= 240 else text[:237] + "..."

    def _headless_geometry_fit_caked_payload_signature(
        background_idx: int,
        detector_shape: Sequence[object],
        params_local: Mapping[str, object],
    ) -> tuple[object, ...]:
        center = _headless_geometry_fit_center(params_local)
        center_signature = (
            _signature_numeric(center[0]) if center is not None else None,
            _signature_numeric(center[1]) if center is not None else None,
        )
        try:
            source_signature = _source_snapshot_signature_for_background(
                int(background_idx),
                dict(params_local),
            )
        except Exception:
            source_signature = None
        return (
            "headless_exact_caked_view",
            int(background_idx),
            tuple(int(v) for v in tuple(detector_shape)[:2]),
            int(HEADLESS_GEOMETRY_CAKED_RADIAL_BINS),
            int(HEADLESS_GEOMETRY_CAKED_AZIMUTH_BINS),
            "caked",
            _signature_numeric(params_local.get("corto_detector")),
            _signature_numeric(params_local.get("lambda")),
            center_signature,
            _signature_numeric(defaults.pixel_size_m),
            bool(background_state.backend_flip_x),
            bool(background_state.backend_flip_y),
            int(background_state.backend_rotation_k),
            source_signature,
        )

    def _headless_geometry_fit_projection_payload_token(
        payload: Mapping[str, object],
        *,
        signature: tuple[object, ...],
    ) -> str | None:
        radial_token = gui_manual_geometry.geometry_manual_stable_axis_value_token(
            payload.get("radial_axis")
        )
        azimuth_token = gui_manual_geometry.geometry_manual_stable_axis_value_token(
            payload.get("azimuth_axis")
        )
        raw_azimuth_token = gui_manual_geometry.geometry_manual_stable_axis_value_token(
            payload.get("raw_azimuth_axis")
        )
        permutation_token = gui_manual_geometry.geometry_manual_stable_permutation_value_token(
            payload.get("raw_to_gui_row_permutation")
        )
        try:
            detector_shape = tuple(int(v) for v in tuple(payload.get("detector_shape"))[:2])
        except Exception:
            detector_shape = ()
        if (
            len(detector_shape) < 2
            or min(detector_shape) <= 0
            or radial_token is None
            or azimuth_token is None
            or raw_azimuth_token is None
            or permutation_token is None
        ):
            return None
        token_payload = {
            "kind": "headless_exact_caked_projection_content_v3",
            "signature": signature,
            "detector_shape": detector_shape,
            "radial_axis": tuple(radial_token),
            "azimuth_axis": tuple(azimuth_token),
            "raw_azimuth_axis": tuple(raw_azimuth_token),
            "raw_to_gui_row_permutation": tuple(permutation_token),
            "transform_bundle_generation": payload.get("transform_bundle_generation"),
        }
        return gui_geometry_fit._geometry_fit_digest_payload(token_payload)

    def _headless_geometry_fit_hydrate_caked_payload(
        payload: object,
        *,
        signature: tuple[object, ...],
        detector_shape: Sequence[object],
        params_local: Mapping[str, object],
    ) -> dict[str, object] | None:
        if not isinstance(payload, Mapping):
            return None
        if payload.get("headless_caked_payload_signature") != signature:
            return None
        if str(payload.get("projection_view_mode") or "").strip().lower() != "caked":
            return None
        expected_shape = tuple(int(v) for v in tuple(detector_shape)[:2])
        try:
            payload_shape = tuple(int(v) for v in tuple(payload.get("detector_shape"))[:2])
        except Exception:
            return None
        if payload_shape != expected_shape:
            return None
        try:
            background = np.asarray(
                payload.get("background_image", payload.get("background")),
                dtype=np.float64,
            )
            radial_axis = np.asarray(payload.get("radial_axis"), dtype=np.float64).reshape(-1)
            azimuth_axis = np.asarray(payload.get("azimuth_axis"), dtype=np.float64).reshape(-1)
            raw_azimuth_axis = np.asarray(
                payload.get("raw_azimuth_axis"),
                dtype=np.float64,
            ).reshape(-1)
        except Exception:
            return None
        if (
            background.ndim != 2
            or radial_axis.size <= 0
            or azimuth_axis.size <= 0
            or raw_azimuth_axis.size <= 0
            or background.shape != (azimuth_axis.size, radial_axis.size)
            or not np.all(np.isfinite(radial_axis))
            or not np.all(np.isfinite(azimuth_axis))
            or not np.all(np.isfinite(raw_azimuth_axis))
        ):
            return None
        normalized_payload = gui_geometry_fit.normalize_geometry_fit_caked_view_payload(
            payload,
            detector_shape=expected_shape,
        )
        if not isinstance(normalized_payload, dict):
            return None
        hydrated_payload = gui_geometry_fit._geometry_fit_hydrate_exact_caked_payload(
            normalized_payload,
            detector_shape=expected_shape,
            params=params_local,
            require_background=True,
        )
        if not isinstance(hydrated_payload, dict):
            return None
        for key in (
            "background_index",
            "projection_view_mode",
            "headless_caked_payload_signature",
            "source_cache_signature",
            "geometry_projection_params_signature",
            "projection_parameter_signature",
            "transform_bundle_generation",
            "caked_axis_shape",
            "raw_to_gui_row_permutation",
        ):
            if key in payload:
                hydrated_payload[key] = payload[key]
        hydrated_payload["projection_view_mode"] = "caked"
        hydrated_payload["headless_caked_payload_signature"] = signature
        return hydrated_payload

    def _headless_geometry_fit_hydrate_caked_projection_payload(
        payload: object,
        *,
        signature: tuple[object, ...],
        detector_shape: Sequence[object],
        params_local: Mapping[str, object],
    ) -> dict[str, object] | None:
        if not isinstance(payload, Mapping):
            return None
        if payload.get("headless_caked_payload_signature") != signature:
            return None
        if str(payload.get("projection_view_mode") or "").strip().lower() != "caked":
            return None
        expected_shape = tuple(int(v) for v in tuple(detector_shape)[:2])
        try:
            payload_shape = tuple(int(v) for v in tuple(payload.get("detector_shape"))[:2])
        except Exception:
            return None
        if payload_shape != expected_shape:
            return None
        normalized_payload = gui_geometry_fit.normalize_geometry_fit_caked_view_payload(
            payload,
            detector_shape=expected_shape,
        )
        if not isinstance(normalized_payload, dict):
            return None
        hydrated_payload = gui_geometry_fit._geometry_fit_hydrate_exact_caked_payload(
            normalized_payload,
            detector_shape=expected_shape,
            params=params_local,
            require_background=False,
        )
        projection = gui_geometry_fit.geometry_fit_caked_projection_payload(hydrated_payload)
        if not isinstance(projection, Mapping):
            return None
        transform_bundle = projection.get("transform_bundle")
        exact_cake = _load_exact_cake_portable_module()
        if not isinstance(transform_bundle, exact_cake.CakeTransformBundle):
            return None
        stored = dict(projection)
        for key in (
            "background_index",
            "source_cache_signature",
            "geometry_projection_params_signature",
            "projection_parameter_signature",
            "transform_bundle_generation",
            "caked_axis_shape",
        ):
            if key in payload:
                stored[key] = payload[key]
        stored["payload_kind"] = "projection"
        stored["projection_view_mode"] = "caked"
        stored["headless_caked_payload_signature"] = signature
        projection_token = stored.get("projection_content_token_v3")
        if (
            str(stored.get("projection_token_schema") or "").strip()
            != "geometry_fit_projection_content_v3"
            or not isinstance(projection_token, str)
            or not projection_token
        ):
            projection_token = _headless_geometry_fit_projection_payload_token(
                stored,
                signature=signature,
            )
        if projection_token is None:
            return None
        stored["projection_token_schema"] = "geometry_fit_projection_content_v3"
        stored["projection_content_token_v3"] = str(projection_token)
        return stored

    def _headless_geometry_fit_caked_payload_is_fresh(
        payload: object,
        *,
        signature: tuple[object, ...],
        detector_shape: Sequence[object],
        params_local: Mapping[str, object],
    ) -> bool:
        return (
            _headless_geometry_fit_hydrate_caked_payload(
                payload,
                signature=signature,
                detector_shape=detector_shape,
                params_local=params_local,
            )
            is not None
        )

    def _geometry_fit_caked_view_for_index(index: int) -> dict[str, object] | None:
        background_idx = int(index)
        native_background, _display_background = _load_background_by_index(background_idx)
        backend_background = gui_background.apply_background_backend_orientation(
            np.asarray(native_background, dtype=np.float64),
            flip_x=background_state.backend_flip_x,
            flip_y=background_state.backend_flip_y,
            rotation_k=background_state.backend_rotation_k,
        )
        if backend_background is None:
            backend_background = native_background
        raw_backend_image = np.asarray(backend_background, dtype=np.float64)
        if raw_backend_image.ndim != 2:
            caked_views_by_background.pop(background_idx, None)
            return None
        detector_shape = tuple(int(v) for v in raw_backend_image.shape[:2])
        params_local = dict(value_callbacks.current_params())
        payload_signature = _headless_geometry_fit_caked_payload_signature(
            background_idx,
            detector_shape,
            params_local,
        )
        cached = caked_views_by_background.get(background_idx)
        hydrated_cached = _headless_geometry_fit_hydrate_caked_payload(
            cached,
            signature=payload_signature,
            detector_shape=detector_shape,
            params_local=params_local,
        )
        if isinstance(hydrated_cached, dict):
            caked_views_by_background[background_idx] = hydrated_cached
            return hydrated_cached
        payload = _build_headless_geometry_fit_caked_view_payload(
            raw_backend_image,
            params=params_local,
            pixel_size_m=float(defaults.pixel_size_m),
        )
        if not isinstance(payload, dict):
            caked_views_by_background.pop(background_idx, None)
            return None
        payload.update(
            {
                "background_index": int(background_idx),
                "projection_view_mode": "caked",
                "headless_caked_payload_signature": payload_signature,
                "source_cache_signature": payload_signature[-1],
                "geometry_projection_params_signature": payload_signature,
                "projection_parameter_signature": payload_signature,
                "transform_bundle_generation": _signature_summary(
                    (
                        detector_shape,
                        int(np.asarray(payload.get("radial_axis")).size),
                        int(np.asarray(payload.get("raw_azimuth_axis")).size),
                        payload_signature,
                    )
                ),
                "caked_axis_shape": (
                    int(np.asarray(payload.get("azimuth_axis")).size),
                    int(np.asarray(payload.get("radial_axis")).size),
                ),
            }
        )
        hydrated_payload = _headless_geometry_fit_hydrate_caked_payload(
            payload,
            signature=payload_signature,
            detector_shape=detector_shape,
            params_local=params_local,
        )
        if not isinstance(hydrated_payload, dict):
            caked_views_by_background.pop(background_idx, None)
            return None
        caked_views_by_background[background_idx] = hydrated_payload
        return hydrated_payload

    def _geometry_fit_caked_projection_for_index(index: int) -> dict[str, object] | None:
        background_idx = int(index)
        native_background, _display_background = _load_background_by_index(background_idx)
        backend_background = gui_background.apply_background_backend_orientation(
            np.asarray(native_background, dtype=np.float64),
            flip_x=background_state.backend_flip_x,
            flip_y=background_state.backend_flip_y,
            rotation_k=background_state.backend_rotation_k,
        )
        if backend_background is None:
            backend_background = native_background
        raw_backend_image = np.asarray(backend_background, dtype=np.float64)
        if raw_backend_image.ndim != 2:
            caked_projection_payloads_by_background.pop(background_idx, None)
            return None
        detector_shape = tuple(int(v) for v in raw_backend_image.shape[:2])
        params_local = dict(value_callbacks.current_params())
        payload_signature = _headless_geometry_fit_caked_payload_signature(
            background_idx,
            detector_shape,
            params_local,
        )
        cached = caked_projection_payloads_by_background.get(background_idx)
        hydrated_cached = _headless_geometry_fit_hydrate_caked_projection_payload(
            cached,
            signature=payload_signature,
            detector_shape=detector_shape,
            params_local=params_local,
        )
        if isinstance(hydrated_cached, dict):
            caked_projection_payloads_by_background[background_idx] = hydrated_cached
            return hydrated_cached
        payload = _build_headless_geometry_fit_caked_projection_payload(
            detector_shape,
            params=params_local,
            pixel_size_m=float(defaults.pixel_size_m),
            background_index=int(background_idx),
        )
        if not isinstance(payload, dict):
            caked_projection_payloads_by_background.pop(background_idx, None)
            return None
        payload.update(
            {
                "background_index": int(background_idx),
                "payload_kind": "projection",
                "projection_view_mode": "caked",
                "headless_caked_payload_signature": payload_signature,
                "source_cache_signature": payload_signature[-1],
                "geometry_projection_params_signature": payload_signature,
                "projection_parameter_signature": payload_signature,
                "transform_bundle_generation": _signature_summary(
                    (
                        detector_shape,
                        int(np.asarray(payload.get("radial_axis")).size),
                        int(np.asarray(payload.get("raw_azimuth_axis")).size),
                        payload_signature,
                    )
                ),
                "caked_axis_shape": (
                    int(np.asarray(payload.get("azimuth_axis")).size),
                    int(np.asarray(payload.get("radial_axis")).size),
                ),
            }
        )
        projection_token = _headless_geometry_fit_projection_payload_token(
            payload,
            signature=payload_signature,
        )
        if projection_token is not None:
            payload["projection_token_schema"] = "geometry_fit_projection_content_v3"
            payload["projection_content_token_v3"] = str(projection_token)
        hydrated_payload = _headless_geometry_fit_hydrate_caked_projection_payload(
            payload,
            signature=payload_signature,
            detector_shape=detector_shape,
            params_local=params_local,
        )
        if not isinstance(hydrated_payload, dict):
            caked_projection_payloads_by_background.pop(background_idx, None)
            return None
        caked_projection_payloads_by_background[background_idx] = hydrated_payload
        return hydrated_payload

    def _native_detector_coords_to_caked_coords_for_background(
        background_index: int,
    ):
        payload = _geometry_fit_caked_projection_for_index(int(background_index))
        if not isinstance(payload, Mapping):
            return None
        exact_cake = _load_exact_cake_portable_module()
        transform_bundle = payload.get("transform_bundle")
        if not isinstance(transform_bundle, exact_cake.CakeTransformBundle):
            return None

        def _to_caked(col: float, row: float) -> tuple[float, float] | None:
            try:
                two_theta_value, phi_value = exact_cake.detector_pixel_to_caked_bin(
                    transform_bundle,
                    float(col),
                    float(row),
                )
            except Exception:
                return None
            if two_theta_value is None or phi_value is None:
                return None
            try:
                two_theta_float = float(two_theta_value)
                phi_float = float(phi_value)
            except Exception:
                return None
            if not (np.isfinite(two_theta_float) and np.isfinite(phi_float)):
                return None
            return float(two_theta_float), float(phi_float)

        return _to_caked

    def _backfill_headless_manual_pair_caked_coordinates() -> tuple[int, list[int]]:
        changed_total = 0
        failed_indices: list[int] = []
        previous_background_idx = int(background_state.current_background_index)
        try:
            for raw_background_idx, raw_entries in list(pairs_by_background.items()):
                try:
                    background_idx = int(raw_background_idx)
                except Exception:
                    continue
                entries = [
                    dict(entry)
                    for entry in (raw_entries or ())
                    if isinstance(entry, Mapping)
                ]
                if not entries or not any(
                    gui_manual_geometry.geometry_manual_entry_needs_caked_coordinate_backfill(
                        entry
                    )
                    for entry in entries
                ):
                    continue
                display_to_native = (
                    _headless_background_display_to_native_detector_coords_for_background(
                        _load_background_by_index,
                        background_idx,
                        display_rotate_k=DISPLAY_ROTATE_K,
                    )
                )
                native_to_caked = _native_detector_coords_to_caked_coords_for_background(
                    background_idx
                )
                if native_to_caked is None:
                    failed_indices.append(int(background_idx))
                    continue
                backfilled, changed_count = (
                    gui_manual_geometry.geometry_manual_backfill_missing_caked_coordinates(
                        entries,
                        background_display_to_native_detector_coords=display_to_native,
                        native_detector_coords_to_caked_display_coords=native_to_caked,
                    )
                )
                if changed_count <= 0:
                    continue
                pairs_by_background[int(background_idx)] = [
                    dict(entry) for entry in backfilled
                ]
                changed_total += int(changed_count)
        finally:
            if int(background_state.current_background_index) != int(previous_background_idx):
                try:
                    _load_background_by_index(previous_background_idx)
                except Exception:
                    pass
        return int(changed_total), failed_indices

    (
        manual_caked_backfill_changed_count,
        manual_caked_backfill_failed_indices,
    ) = _backfill_headless_manual_pair_caked_coordinates()

    def _geometry_fit_required_background_indices() -> list[int]:
        selected = [int(idx) for idx in _current_geometry_fit_background_indices(strict=True)]
        if _geometry_fit_uses_shared_theta_offset(selected):
            return selected
        current_idx = int(background_state.current_background_index)
        if current_idx in set(selected):
            return [current_idx]
        return [int(selected[0])] if selected else [current_idx]

    def _headless_fixed_manual_caked_qr_pair_rows() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        seen_background_indices: set[int] = set()
        for background_idx in _geometry_fit_required_background_indices():
            background_key = int(background_idx)
            if background_key in seen_background_indices:
                continue
            seen_background_indices.add(background_key)
            for entry in _pairs_for_index(background_key):
                if not isinstance(entry, Mapping):
                    continue
                if _headless_geometry_entry_has_fixed_manual_caked_qr(entry):
                    rows.append(dict(entry))
        return rows

    def _ensure_geometry_fit_caked_view() -> None:
        previous_background_idx = int(background_state.current_background_index)
        try:
            for background_idx in _geometry_fit_required_background_indices():
                if not gui_geometry_fit.geometry_manual_pairs_use_caked_fit_space(
                    _pairs_for_index(int(background_idx))
                ):
                    continue
                if _geometry_fit_caked_projection_for_index(int(background_idx)) is None:
                    raise RuntimeError(
                        "exact caked projector unavailable for "
                        f"background {int(background_idx) + 1}"
                    )
        finally:
            if int(background_state.current_background_index) != int(previous_background_idx):
                try:
                    _load_background_by_index(previous_background_idx)
                except Exception:
                    pass

    def _headless_current_caked_projection_for_callbacks() -> dict[str, object] | None:
        try:
            background_idx = int(background_state.current_background_index)
        except Exception:
            return None
        if not gui_geometry_fit.geometry_manual_pairs_use_caked_fit_space(
            _pairs_for_index(background_idx)
        ):
            return None
        payload = _geometry_fit_caked_projection_for_index(background_idx)
        return payload if isinstance(payload, dict) else None

    def _headless_caked_payload_value(key: str) -> object:
        payload = _headless_current_caked_projection_for_callbacks()
        if not isinstance(payload, Mapping):
            return None
        return payload.get(key)

    def _headless_wrap_phi_range(value: object) -> object:
        return _load_exact_cake_portable_module().raw_phi_to_gui_phi(value)

    projection_callbacks = gui_manual_geometry.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: isinstance(
            _headless_current_caked_projection_for_callbacks(),
            Mapping,
        ),
        last_caked_background_image_unscaled=lambda: None,
        last_caked_radial_values=lambda: _headless_caked_payload_value("radial_axis"),
        last_caked_azimuth_values=lambda: _headless_caked_payload_value("azimuth_axis"),
        current_background_display=_current_background_display,
        current_background_native=_current_background_native,
        ai=lambda: None,
        center=lambda: [
            _coerce_float(var_store["center_x_var"].get(), defaults.defaults["center_x"]),
            _coerce_float(var_store["center_y_var"].get(), defaults.defaults["center_y"]),
        ],
        detector_distance=lambda: _coerce_float(
            var_store["corto_detector_var"].get(),
            defaults.defaults["corto_detector"],
        ),
        pixel_size=float(defaults.pixel_size_m),
        caked_transform_bundle=lambda: _headless_caked_payload_value("transform_bundle"),
        current_background_index=lambda: int(background_state.current_background_index),
        caked_projection_payload=_headless_current_caked_projection_for_callbacks,
        last_caked_raw_azimuth_values=lambda: _headless_caked_payload_value(
            "raw_azimuth_axis"
        ),
        last_caked_raw_to_gui_row_permutation=lambda: _headless_caked_payload_value(
            "raw_to_gui_row_permutation"
        ),
        wrap_phi_range=_headless_wrap_phi_range,
        rotate_point_for_display=gui_geometry_overlay.rotate_point_for_display,
        display_rotate_k=DISPLAY_ROTATE_K,
        current_geometry_fit_params=value_callbacks.current_params,
        miller=lambda: structure_state.miller,
        intensities=lambda: structure_state.intensities,
        image_size=int(defaults.image_size),
        display_to_native_sim_coords=lambda col, row, image_shape: (
            gui_geometry_overlay.display_to_native_sim_coords(
                col,
                row,
                image_shape,
                sim_display_rotate_k=SIM_DISPLAY_ROTATE_K,
            )
        ),
        native_detector_coords_to_bundle_detector_coords=(
            _native_detector_coords_to_live_bundle_detector_coords
        ),
        bundle_detector_coords_to_background_display_coords=(
            _live_bundle_detector_coords_to_background_display_coords
        ),
    )

    def _project_peaks_for_background_view(
        background_index: int,
        rows: Sequence[dict[str, object]] | None,
        *,
        mode_override: str | None = None,
        strict_caked_projection: bool = True,
    ) -> list[dict[str, object]]:
        normalized_rows = [dict(entry) for entry in (rows or ()) if isinstance(entry, Mapping)]
        if not normalized_rows:
            return []
        background_idx = int(background_index)
        if mode_override is None:
            use_caked_projection = gui_geometry_fit.geometry_manual_pairs_use_caked_fit_space(
                _pairs_for_index(background_idx)
            )
        else:
            use_caked_projection = str(mode_override).strip().lower() == "caked"
        if not use_caked_projection:
            return [
                dict(entry)
                for entry in (
                    projection_callbacks.project_peaks_to_current_view(normalized_rows) or ()
                )
                if isinstance(entry, Mapping)
            ]
        previous_background_idx = int(background_state.current_background_index)
        try:
            payload = _geometry_fit_caked_projection_for_index(background_idx)
            if not isinstance(payload, Mapping):
                if not bool(strict_caked_projection):
                    return []
                raise RuntimeError(
                    f"exact caked projector unavailable for background {int(background_idx) + 1}"
                )
            native_background, display_background = _load_background_by_index(background_idx)
            detector_shape = tuple(int(value) for value in np.asarray(native_background).shape[:2])

            def _native_detector_coords_to_bundle_detector_coords(
                col: float,
                row: float,
            ) -> tuple[float | None, float | None]:
                if len(detector_shape) < 2 or min(detector_shape) <= 0:
                    return None, None
                return gui_geometry_overlay.rotate_point_for_display(
                    float(col),
                    float(row),
                    detector_shape,
                    DISPLAY_ROTATE_K,
                )

            background_projection_callbacks = (
                gui_manual_geometry.make_runtime_geometry_manual_projection_callbacks(
                    caked_view_enabled=lambda: True,
                    last_caked_background_image_unscaled=lambda: None,
                    last_caked_radial_values=lambda: payload.get("radial_axis"),
                    last_caked_azimuth_values=lambda: payload.get("azimuth_axis"),
                    current_background_display=lambda: display_background,
                    current_background_native=lambda: native_background,
                    ai=lambda: None,
                    center=lambda: [
                        _coerce_float(
                            var_store["center_x_var"].get(),
                            defaults.defaults["center_x"],
                        ),
                        _coerce_float(
                            var_store["center_y_var"].get(),
                            defaults.defaults["center_y"],
                        ),
                    ],
                    detector_distance=lambda: _coerce_float(
                        var_store["corto_detector_var"].get(),
                        defaults.defaults["corto_detector"],
                    ),
                    pixel_size=float(defaults.pixel_size_m),
                    caked_transform_bundle=lambda: payload.get("transform_bundle"),
                    current_background_index=lambda: int(background_idx),
                    caked_projection_payload=lambda: payload,
                    last_caked_raw_azimuth_values=lambda: payload.get("raw_azimuth_axis"),
                    last_caked_raw_to_gui_row_permutation=lambda: payload.get(
                        "raw_to_gui_row_permutation"
                    ),
                    wrap_phi_range=_headless_wrap_phi_range,
                    rotate_point_for_display=gui_geometry_overlay.rotate_point_for_display,
                    display_rotate_k=DISPLAY_ROTATE_K,
                    current_geometry_fit_params=value_callbacks.current_params,
                    miller=lambda: structure_state.miller,
                    intensities=lambda: structure_state.intensities,
                    image_size=int(defaults.image_size),
                    display_to_native_sim_coords=lambda col, row, image_shape: (
                        gui_geometry_overlay.display_to_native_sim_coords(
                            col,
                            row,
                            image_shape,
                            sim_display_rotate_k=SIM_DISPLAY_ROTATE_K,
                        )
                    ),
                    native_detector_coords_to_bundle_detector_coords=(
                        _native_detector_coords_to_bundle_detector_coords
                    ),
                    bundle_detector_coords_to_background_display_coords=(
                        lambda col, row: (float(col), float(row))
                    ),
                )
            )
            projected_rows = background_projection_callbacks.project_peaks_to_current_view(
                normalized_rows
            )
            return [
                {**dict(entry), "background_index": int(background_idx)}
                for entry in (projected_rows or ())
                if isinstance(entry, Mapping)
            ]
        finally:
            if int(background_state.current_background_index) != int(previous_background_idx):
                try:
                    _load_background_by_index(previous_background_idx)
                except Exception:
                    pass

    source_snapshot_diagnostics_state: dict[str, object] = {}

    def _project_peaks_to_current_view_for_dataset(
        rows: Sequence[dict[str, object]] | None,
    ) -> list[dict[str, object]]:
        callback = getattr(projection_callbacks, "project_peaks_to_current_view", None)
        if callable(callback):
            try:
                projected = callback(rows)
            except Exception:
                projected = rows
            return [dict(entry) for entry in (projected or ()) if isinstance(entry, Mapping)]
        return [dict(entry) for entry in (rows or ()) if isinstance(entry, Mapping)]

    def _geometry_manual_simulated_peaks_for_params(
        param_set: dict[str, object] | None = None,
        *,
        prefer_cache: bool = True,
    ) -> list[dict[str, object]]:
        params_local = dict(value_callbacks.current_params())
        if isinstance(param_set, Mapping):
            params_local.update(dict(param_set))
        if bool(prefer_cache):
            cached = projection_callbacks.simulated_peaks_for_params(
                params_local,
                prefer_cache=True,
            )
            if cached:
                return [dict(entry) for entry in cached if isinstance(entry, Mapping)]
        required_pairs = [
            dict(entry)
            for entry in _pairs_for_index(int(background_state.current_background_index))
            if isinstance(entry, Mapping)
        ]
        required_targets = gui_geometry_fit.collect_geometry_fit_required_manual_fit_targets(
            required_pairs,
            background_index=int(background_state.current_background_index),
        )
        required_branch_keys = gui_geometry_fit._geometry_fit_required_branch_group_keys(
            required_targets
        )
        try:
            hit_tables = _simulate_hit_tables_for_fit(
                structure_state.miller,
                structure_state.intensities,
                int(defaults.image_size),
                params_local,
                required_branch_group_keys=required_branch_keys,
                required_manual_fit_targets=required_targets,
                preflight_mode="manual_geometry_targeted",
            )
            source_rows, _lattice, _hit_tables, _source_indices = _build_source_rows_from_hit_tables(
                hit_tables,
                image_size_value=int(defaults.image_size),
                params_local=params_local,
                native_sim_to_display_coords=lambda col, row, image_shape_local: (
                    gui_geometry_overlay.native_sim_to_display_coords(
                        col,
                        row,
                        image_shape_local,
                        sim_display_rotate_k=SIM_DISPLAY_ROTATE_K,
                    )
                ),
                allow_nominal_hkl_indices=True,
            )
            projected_rows = _project_peaks_to_current_view_for_dataset(source_rows)
            if projected_rows:
                return [
                    dict(entry)
                    for entry in projected_rows
                    if isinstance(entry, Mapping)
                ]
        except Exception:
            pass
        try:
            preview_rows = simulation_callbacks.simulate_preview_style_peaks(
                structure_state.miller,
                structure_state.intensities,
                int(defaults.image_size),
                params_local,
            )
        except Exception:
            return []
        return [
            dict(entry)
            for entry in _project_peaks_to_current_view_for_dataset(preview_rows)
            if isinstance(entry, Mapping)
        ]

    def _source_snapshot_signature_for_background(
        background_index: int,
        param_set: dict[str, object] | None = None,
    ) -> tuple[object, ...]:
        params_local = dict(value_callbacks.current_params())
        if isinstance(param_set, Mapping):
            params_local.update(dict(param_set))
        center_value = params_local.get("center", [np.nan, np.nan])
        if isinstance(center_value, (list, tuple, np.ndarray)) and len(center_value) >= 2:
            center_signature = (
                _signature_numeric(center_value[0]),
                _signature_numeric(center_value[1]),
            )
        else:
            center_signature = (None, None)
        return (
            int(background_index),
            int(defaults.image_size),
            tuple(np.asarray(structure_state.miller).shape),
            tuple(np.asarray(structure_state.intensities).shape),
            _signature_numeric(params_local.get("a")),
            _signature_numeric(params_local.get("c")),
            _signature_numeric(params_local.get("lambda")),
            _signature_numeric(params_local.get("theta_initial")),
            _signature_numeric(params_local.get("theta_offset")),
            _signature_numeric(params_local.get("corto_detector")),
            _signature_numeric(params_local.get("gamma")),
            _signature_numeric(params_local.get("Gamma")),
            _signature_numeric(params_local.get("chi")),
            _signature_numeric(params_local.get("cor_angle")),
            _signature_numeric(params_local.get("zb")),
            _signature_numeric(params_local.get("zs")),
            center_signature,
            int(background_state.backend_rotation_k),
            bool(background_state.backend_flip_x),
            bool(background_state.backend_flip_y),
        )

    def _live_cache_inventory_snapshot() -> dict[str, object]:
        source_snapshots: list[dict[str, object]] = []
        for raw_background_index, raw_snapshot in sorted(
            getattr(simulation_runtime_state, "source_row_snapshots", {}).items(),
            key=lambda item: int(item[0]),
        ):
            if not isinstance(raw_snapshot, Mapping):
                continue
            row_count = raw_snapshot.get("row_count")
            if row_count is None:
                row_count = len(raw_snapshot.get("rows", ()) or ())
            source_snapshots.append(
                {
                    "background_index": int(raw_background_index),
                    "row_count": int(row_count),
                    "created_from": raw_snapshot.get("created_from"),
                    "signature_summary": _signature_summary(
                        raw_snapshot.get("simulation_signature")
                    ),
                }
            )
        stored_cache = getattr(simulation_runtime_state, "stored_intersection_cache", None)
        return {
            "preview_active": False,
            "preview_sample_count": len(
                getattr(simulation_runtime_state, "peak_records", ()) or ()
            ),
            "stored_hit_table_signature_present": bool(
                isinstance(stored_cache, Sequence)
                and not isinstance(stored_cache, (str, bytes))
                and len(stored_cache) > 0
            ),
            "stored_hit_table_signature_summary": _signature_summary(
                getattr(simulation_runtime_state, "stored_hit_table_signature", None)
            ),
            "last_simulation_signature_summary": _signature_summary(
                getattr(simulation_runtime_state, "last_simulation_signature", None)
            ),
            "primary_contribution_cache_signature_summary": None,
            "primary_source_mode": None,
            "primary_active_contribution_key_count": 0,
            "primary_hit_table_cache_entry_count": 0,
            "source_snapshots": source_snapshots,
        }

    def _set_source_snapshot_diagnostics(**fields: object) -> None:
        source_snapshot_diagnostics_state.clear()
        source_snapshot_diagnostics_state.update(fields)

    def _build_live_preview_simulated_peaks_from_cache() -> (
        dict[str, object] | list[dict[str, object]]
    ):
        max_positions_local = simulation_runtime_state.stored_max_positions_local
        current_signature = getattr(
            simulation_runtime_state,
            "stored_hit_table_signature",
            None,
        )
        if current_signature is None:
            current_signature = getattr(
                simulation_runtime_state,
                "last_simulation_signature",
                None,
            )
        current_signature_summary = _signature_summary(current_signature)
        source_reflection_indices_local = (
            simulation_runtime_state.stored_source_reflection_indices_local
        )
        peak_record_count = int(len(simulation_runtime_state.peak_records or ()))
        max_positions_row_count = (
            int(len(max_positions_local))
            if isinstance(max_positions_local, Sequence)
            and not isinstance(max_positions_local, (str, bytes))
            else 0
        )
        provenance = gui_geometry_q_group_manager._resolve_live_peak_record_fallback_provenance(
            simulation_runtime_state,
            signature=current_signature,
            signature_summary=current_signature_summary,
            background_index=int(background_state.current_background_index),
            source_reflection_indices_local=source_reflection_indices_local,
        )
        live_rows = gui_manual_geometry.geometry_manual_live_peak_candidates_from_records(
            simulation_runtime_state.peak_records,
            source_reflection_indices_local=source_reflection_indices_local,
            source_row_hkl_lookup=provenance.get("source_row_hkl_lookup"),
            provenance_signature_matches=bool(provenance.get("active_signature_matches", False)),
            provenance_revision_matches=bool(provenance.get("active_revision_matches", False)),
            expected_table_count=provenance.get("expected_table_count"),
        )
        rows = [dict(entry) for entry in live_rows if isinstance(entry, Mapping)]
        if rows:
            return {
                "rows": rows,
                "cache_metadata": {
                    "cache_source": "peak_records",
                    "fallback_used": False,
                    "max_positions_row_count": int(max_positions_row_count),
                    "peak_record_count": int(peak_record_count),
                    "active_signature_matches": bool(
                        provenance.get("active_signature_matches", False)
                    ),
                    "source_snapshot_row_count": int(
                        provenance.get("source_snapshot_row_count", 0) or 0
                    ),
                    "source_snapshot_background_index": provenance.get(
                        "source_snapshot_background_index"
                    ),
                },
            }
        return {
            "rows": [],
            "cache_metadata": {
                "cache_source": "empty",
                "fallback_used": bool(peak_record_count > 0),
                "max_positions_row_count": int(max_positions_row_count),
                "peak_record_count": int(peak_record_count),
                "active_signature_matches": bool(provenance.get("active_signature_matches", False)),
                "source_snapshot_row_count": int(
                    provenance.get("source_snapshot_row_count", 0) or 0
                ),
                "source_snapshot_background_index": provenance.get(
                    "source_snapshot_background_index"
                ),
            },
        }

    def _filter_hit_tables_for_required_branch_groups(
        hit_tables: Sequence[object] | None,
        *,
        required_branch_group_keys: (
            Sequence[tuple[tuple[int, int, int], int | None, object | None]] | None
        ),
        required_manual_fit_targets: Sequence[Mapping[str, object]] | None = None,
    ) -> list[object]:
        table_list = _copy_hit_tables(hit_tables)
        required_keys = list(required_branch_group_keys or ())
        required_targets = [
            dict(entry)
            for entry in (required_manual_fit_targets or ())
            if isinstance(entry, Mapping)
        ]
        required_source_indices: set[int] = set()
        for target in required_targets:
            for index_key in ("source_reflection_index", "source_table_index"):
                try:
                    source_idx = int(target.get(index_key))
                except Exception:
                    continue
                if source_idx < 0:
                    continue
                required_source_indices.add(int(source_idx))
        if not table_list or not (required_keys or required_source_indices):
            return table_list

        required_hkls = {tuple(key[0]) for key in required_keys}
        filtered_tables: list[object] = []
        for table_idx, table in enumerate(table_list):
            arr = np.asarray(table, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] < 7:
                continue
            source_table_index, _source_row_index, _best_sample_index = (
                extract_hit_row_provenance(arr[0])
            )
            if source_table_index is None:
                source_table_index = int(table_idx)
            try:
                table_hkl = (
                    int(np.rint(float(arr[0, 4]))),
                    int(np.rint(float(arr[0, 5]))),
                    int(np.rint(float(arr[0, 6]))),
                )
            except Exception:
                continue
            source_index_required = (
                source_table_index is not None and int(source_table_index) in required_source_indices
            )
            if table_hkl not in required_hkls and not source_index_required:
                continue
            if required_hkls:
                row_mask = np.asarray(
                    [
                        (
                            int(np.rint(float(row[4]))),
                            int(np.rint(float(row[5]))),
                            int(np.rint(float(row[6]))),
                        )
                        in required_hkls
                        for row in arr
                    ],
                    dtype=bool,
                )
                if np.any(row_mask):
                    filtered_tables.append(np.asarray(arr[row_mask], dtype=np.float64).copy())
            else:
                filtered_tables.append(np.asarray(arr, dtype=np.float64).copy())
        return filtered_tables

    def _simulate_hit_tables_for_fit(
        miller_array: np.ndarray,
        intensity_array: np.ndarray,
        image_size_value: int,
        params_local: Mapping[str, object],
        *,
        required_branch_group_keys: (
            Sequence[tuple[tuple[int, int, int], int | None, object | None]] | None
        ) = None,
        required_manual_fit_targets: Sequence[Mapping[str, object]] | None = None,
        preflight_mode: str = "full",
    ) -> list[object]:
        return simulation_callbacks.simulate_hit_tables(
            np.asarray(miller_array, dtype=np.float64),
            np.asarray(intensity_array, dtype=np.float64),
            int(image_size_value),
            dict(params_local),
            required_branch_group_keys=required_branch_group_keys,
            required_manual_fit_targets=required_manual_fit_targets,
        )

    def _background_label_for_index(background_index: int) -> str:
        try:
            osc_path = background_state.osc_files[int(background_index)]
        except Exception:
            osc_path = None
        if osc_path is not None:
            try:
                label = Path(str(osc_path)).name
            except Exception:
                label = ""
            if str(label).strip():
                return str(label)
        return f"background {int(background_index) + 1}"

    def _commit_source_row_rebuild_result(
        rebuild_result: gui_geometry_fit.GeometryFitSourceRowRebuildResult,
        *,
        consumer: str | None = None,
    ) -> list[dict[str, object]]:
        if not isinstance(rebuild_result, gui_geometry_fit.GeometryFitSourceRowRebuildResult):
            return []

        background_idx = int(rebuild_result.background_index)
        stored_rows = [
            dict(entry)
            for entry in (rebuild_result.stored_rows or ())
            if isinstance(entry, Mapping)
        ]
        diagnostics = (
            dict(rebuild_result.diagnostics)
            if isinstance(rebuild_result.diagnostics, Mapping)
            else {}
        )
        if stored_rows:
            if rebuild_result.hit_tables is not None:
                try:
                    max_positions_local = diffraction.hit_tables_to_max_positions(
                        rebuild_result.hit_tables
                    )
                except Exception:
                    max_positions_local = np.empty((0, 6), dtype=np.float64)
                simulation_runtime_state.stored_max_positions_local = list(max_positions_local)
            if rebuild_result.peak_table_lattice is not None:
                simulation_runtime_state.stored_peak_table_lattice = list(
                    rebuild_result.peak_table_lattice
                )
            simulation_runtime_state.stored_source_reflection_indices_local = (
                list(rebuild_result.source_reflection_indices)
                if rebuild_result.source_reflection_indices is not None
                else None
            )
            simulation_runtime_state.stored_sim_image = np.zeros(
                (int(defaults.image_size), int(defaults.image_size)),
                dtype=np.float64,
            )
            if rebuild_result.intersection_cache is not None:
                simulation_runtime_state.stored_intersection_cache = (
                    _copy_intersection_cache_tables(rebuild_result.intersection_cache)
                )
                simulation_runtime_state.stored_hit_table_signature = (
                    rebuild_result.requested_signature
                )
            simulation_runtime_state.last_simulation_signature = rebuild_result.requested_signature
            _set_runtime_peak_cache_from_source_rows(
                simulation_runtime_state,
                stored_rows,
            )
            simulation_runtime_state.source_row_snapshots[int(background_idx)] = {
                "background_index": int(background_idx),
                "simulation_signature": rebuild_result.requested_signature,
                "simulation_signature_summary": _signature_summary(
                    rebuild_result.requested_signature
                ),
                "rows": [dict(entry) for entry in stored_rows],
                "row_count": int(len(stored_rows)),
                "created_from": str(rebuild_result.rebuild_source or "unknown"),
                "source_reflection_index_count": int(
                    len(rebuild_result.source_reflection_indices or ())
                ),
            }

        if diagnostics:
            _set_source_snapshot_diagnostics(**diagnostics)
        return [
            dict(entry)
            for entry in (
                rebuild_result.projected_rows if rebuild_result.projected_rows else stored_rows
            )
            if isinstance(entry, Mapping)
        ]

    def _geometry_manual_rebuild_source_rows_for_background(
        background_index: int,
        param_set: dict[str, object] | None = None,
        *,
        consumer: str | None = None,
        prior_diagnostics: Mapping[str, object] | None = None,
        required_pairs: Sequence[Mapping[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        background_idx = int(background_index)
        consumer_name = str(consumer or "unspecified")
        params_local = dict(value_callbacks.current_params())
        if isinstance(param_set, Mapping):
            params_local.update(dict(param_set))
        requested_signature = _source_snapshot_signature_for_background(
            background_idx,
            params_local,
        )
        requested_signature_summary = _signature_summary(requested_signature)

        def _build_source_rows_for_rebuild(
            source_tables: Sequence[object] | None,
            *,
            required_branch_group_keys: (
                Sequence[tuple[tuple[int, int, int], int | None, object | None]] | None
            ) = None,
            required_manual_fit_targets: Sequence[Mapping[str, object]] | None = None,
            preflight_mode: str = "full",
            consumer: str | None = None,
        ) -> tuple[list[dict[str, object]], list[tuple[float, float, str]], list[object]]:
            schema = _load_intersection_cache_schema()
            table_list = list(source_tables or ())
            if not table_list:
                return [], [], []
            if schema.is_intersection_cache_table(table_list[0]):
                hit_tables_local = diffraction.intersection_cache_to_hit_tables(table_list)
            else:
                hit_tables_local = _copy_hit_tables(table_list)
            if (
                str(preflight_mode or "full") == "manual_geometry_targeted"
                and str(consumer or consumer_name) != "geometry_fit_trial_source_rows"
            ):
                hit_tables_local = _filter_hit_tables_for_required_branch_groups(
                    hit_tables_local,
                    required_branch_group_keys=required_branch_group_keys,
                    required_manual_fit_targets=required_manual_fit_targets,
                )
            return _build_source_rows_from_hit_tables(
                hit_tables_local,
                image_size_value=int(defaults.image_size),
                params_local=params_local,
                native_sim_to_display_coords=lambda col, row, image_shape_local: (
                    gui_geometry_overlay.native_sim_to_display_coords(
                        col,
                        row,
                        image_shape_local,
                        sim_display_rotate_k=SIM_DISPLAY_ROTATE_K,
                    )
                ),
                allow_nominal_hkl_indices=True,
            )

        rebuild_result = gui_geometry_fit.rebuild_geometry_fit_source_rows(
            background_index=int(background_idx),
            background_label=_background_label_for_index(background_idx),
            params_local=params_local,
            consumer=consumer_name,
            prior_diagnostics=prior_diagnostics,
            requested_signature=requested_signature,
            requested_signature_summary=requested_signature_summary,
            can_use_live_runtime_cache=(
                background_idx == int(background_state.current_background_index)
            ),
            build_live_rows=_build_live_preview_simulated_peaks_from_cache,
            get_memory_intersection_cache=diffraction.get_last_intersection_cache,
            load_logged_intersection_cache=diffraction.load_most_recent_logged_intersection_cache,
            logged_cache_matches_params=_logged_cache_matches_params,
            build_source_rows_from_hit_tables=_build_source_rows_for_rebuild,
            simulate_hit_tables=(
                lambda normalized_params, **kwargs: _simulate_hit_tables_for_fit(
                    structure_state.miller,
                    structure_state.intensities,
                    int(defaults.image_size),
                    normalized_params,
                    **kwargs,
                )
            ),
            last_runtime_simulation_diagnostics=simulation_callbacks.last_simulation_diagnostics,
            required_pairs=required_pairs,
            live_cache_inventory=_live_cache_inventory_snapshot(),
        )
        return _commit_source_row_rebuild_result(
            rebuild_result,
            consumer=consumer_name,
        )

    def _geometry_manual_source_rows_for_background(
        background_index: int,
        param_set: dict[str, object] | None = None,
        *,
        consumer: str | None = None,
        required_pairs: Sequence[Mapping[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        background_idx = int(background_index)
        consumer_name = str(consumer or "unspecified")
        requested_signature = _source_snapshot_signature_for_background(
            background_idx,
            param_set,
        )
        requested_signature_summary = _signature_summary(requested_signature)
        snapshot = dict(simulation_runtime_state.source_row_snapshots.get(background_idx) or {})
        if not snapshot:
            _set_source_snapshot_diagnostics(
                source="source_snapshot",
                cache_family="source_snapshot",
                action="lookup",
                consumer=consumer_name,
                status="snapshot_missing_background",
                background_index=int(background_idx),
                requested_signature=requested_signature,
                requested_signature_summary=requested_signature_summary,
                raw_peak_count=0,
                projected_peak_count=0,
                signature_match=False,
                live_cache_inventory=_live_cache_inventory_snapshot(),
            )
            rebuilt_rows = _geometry_manual_rebuild_source_rows_for_background(
                background_idx,
                param_set,
                consumer=consumer_name,
                prior_diagnostics=_geometry_manual_last_source_snapshot_diagnostics(),
                required_pairs=required_pairs,
            )
            if rebuilt_rows:
                return rebuilt_rows
            return []

        snapshot_signature = snapshot.get("simulation_signature")
        stored_signature_summary = _signature_summary(snapshot_signature)
        created_from = snapshot.get("created_from")
        raw_rows = [
            dict(entry) for entry in (snapshot.get("rows", ()) or ()) if isinstance(entry, Mapping)
        ]
        if snapshot_signature != requested_signature:
            _set_source_snapshot_diagnostics(
                source="source_snapshot",
                cache_family="source_snapshot",
                action="lookup",
                consumer=consumer_name,
                status="snapshot_signature_mismatch",
                background_index=int(background_idx),
                requested_signature=requested_signature,
                requested_signature_summary=requested_signature_summary,
                snapshot_signature=snapshot_signature,
                stored_signature_summary=stored_signature_summary,
                raw_peak_count=int(len(raw_rows)),
                projected_peak_count=0,
                created_from=created_from,
                signature_match=False,
                live_cache_inventory=_live_cache_inventory_snapshot(),
            )
            rebuilt_rows = _geometry_manual_rebuild_source_rows_for_background(
                background_idx,
                param_set,
                consumer=consumer_name,
                prior_diagnostics=_geometry_manual_last_source_snapshot_diagnostics(),
                required_pairs=required_pairs,
            )
            if rebuilt_rows:
                return rebuilt_rows
            return []
        if not raw_rows:
            _set_source_snapshot_diagnostics(
                source="source_snapshot",
                cache_family="source_snapshot",
                action="lookup",
                consumer=consumer_name,
                status="snapshot_empty",
                background_index=int(background_idx),
                requested_signature=requested_signature,
                requested_signature_summary=requested_signature_summary,
                snapshot_signature=snapshot_signature,
                stored_signature_summary=stored_signature_summary,
                raw_peak_count=0,
                projected_peak_count=0,
                created_from=created_from,
                signature_match=True,
                live_cache_inventory=_live_cache_inventory_snapshot(),
            )
            rebuilt_rows = _geometry_manual_rebuild_source_rows_for_background(
                background_idx,
                param_set,
                consumer=consumer_name,
                prior_diagnostics=_geometry_manual_last_source_snapshot_diagnostics(),
                required_pairs=required_pairs,
            )
            if rebuilt_rows:
                return rebuilt_rows
            return []

        snapshot_validation = (
            gui_geometry_fit.validate_geometry_fit_live_source_rows(
                raw_rows,
                required_pairs=required_pairs,
            )
            if required_pairs
            else {}
        )
        if required_pairs and not bool(snapshot_validation.get("valid", False)):
            _set_source_snapshot_diagnostics(
                source="source_snapshot",
                cache_family="source_snapshot",
                action="lookup",
                consumer=consumer_name,
                status="snapshot_pair_validation_failed",
                background_index=int(background_idx),
                requested_signature=requested_signature,
                requested_signature_summary=requested_signature_summary,
                snapshot_signature=snapshot_signature,
                stored_signature_summary=stored_signature_summary,
                raw_peak_count=int(len(raw_rows)),
                projected_peak_count=0,
                created_from=created_from,
                signature_match=True,
                live_cache_inventory=_live_cache_inventory_snapshot(),
                live_runtime_cache_validation=snapshot_validation,
            )
            rebuilt_rows = _geometry_manual_rebuild_source_rows_for_background(
                background_idx,
                param_set,
                consumer=consumer_name,
                prior_diagnostics=_geometry_manual_last_source_snapshot_diagnostics(),
                required_pairs=required_pairs,
            )
            if rebuilt_rows:
                return rebuilt_rows
            return []

        _set_source_snapshot_diagnostics(
            source="source_snapshot",
            cache_family="source_snapshot",
            action="lookup",
            consumer=consumer_name,
            status="snapshot_hit",
            background_index=int(background_idx),
            requested_signature=requested_signature,
            requested_signature_summary=requested_signature_summary,
            snapshot_signature=snapshot_signature,
            stored_signature_summary=stored_signature_summary,
            raw_peak_count=int(len(raw_rows)),
            projected_peak_count=int(len(raw_rows)),
            created_from=created_from,
            signature_match=True,
            live_cache_inventory=_live_cache_inventory_snapshot(),
            live_runtime_cache_validation=snapshot_validation,
        )
        return raw_rows

    def _geometry_manual_last_source_snapshot_diagnostics() -> dict[str, object]:
        return dict(source_snapshot_diagnostics_state)

    manual_dataset_bindings = gui_geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=tuple(defaults.osc_files),
        current_background_index=int(background_state.current_background_index),
        image_size=int(defaults.image_size),
        display_rotate_k=int(DISPLAY_ROTATE_K),
        geometry_manual_pairs_for_index=_pairs_for_index,
        load_background_by_index=_load_background_by_index,
        apply_background_backend_orientation=lambda image: (
            gui_background.apply_background_backend_orientation(
                image,
                flip_x=background_state.backend_flip_x,
                flip_y=background_state.backend_flip_y,
                rotation_k=background_state.backend_rotation_k,
            )
        ),
        backend_detector_coords_to_native_detector_coords=lambda col, row, native_shape=None: (
            gui_background.background_backend_point_to_native_coords(
                float(col),
                float(row),
                native_shape=(
                    tuple(int(v) for v in tuple(native_shape)[:2])
                    if native_shape is not None
                    else np.asarray(
                        _load_background_by_index(int(background_state.current_background_index))[0]
                    ).shape[:2]
                ),
                flip_x=background_state.backend_flip_x,
                flip_y=background_state.backend_flip_y,
                rotation_k=background_state.backend_rotation_k,
            )
        ),
        native_detector_coords_to_detector_display_coords=(
            _headless_native_detector_coords_to_detector_display_coords_for_background(
                _load_background_by_index,
                int(background_state.current_background_index),
                display_rotate_k=DISPLAY_ROTATE_K,
            )
        ),
        native_detector_coords_to_detector_display_coords_for_background=(
            lambda background_index: (
                _headless_native_detector_coords_to_detector_display_coords_for_background(
                    _load_background_by_index,
                    int(background_index),
                    display_rotate_k=DISPLAY_ROTATE_K,
                )
            )
        ),
        geometry_manual_simulated_peaks_for_params=_geometry_manual_simulated_peaks_for_params,
        geometry_manual_simulated_lookup=projection_callbacks.simulated_lookup,
        geometry_manual_source_rows_for_background=_geometry_manual_source_rows_for_background,
        geometry_manual_rebuild_source_rows_for_background=(
            _geometry_manual_rebuild_source_rows_for_background
        ),
        geometry_manual_last_source_snapshot_diagnostics=_geometry_manual_last_source_snapshot_diagnostics,
        geometry_manual_last_simulation_diagnostics=simulation_callbacks.last_simulation_diagnostics,
        geometry_manual_match_config=lambda: (
            gui_manual_geometry.current_geometry_manual_match_config(headless_fit_config)
        ),
        pick_uses_caked_space=_manual_current_background_uses_caked_space,
        geometry_manual_caked_view_for_index=None,
        geometry_manual_caked_projection_for_index=_geometry_fit_caked_projection_for_index,
        geometry_manual_entry_display_coords=projection_callbacks.entry_display_coords,
        geometry_manual_project_peaks_to_current_view=(_project_peaks_to_current_view_for_dataset),
        geometry_manual_project_peaks_for_background_view=(_project_peaks_for_background_view),
        geometry_manual_refresh_pair_entry=projection_callbacks.refresh_entry_geometry,
        unrotate_display_peaks=lambda measured, rotated_shape, *, k=None: (
            gui_geometry_overlay.unrotate_display_peaks(
                measured,
                rotated_shape,
                k=k,
                default_display_rotate_k=DISPLAY_ROTATE_K,
            )
        ),
        display_to_native_sim_coords=lambda col, row, image_shape: (
            gui_geometry_overlay.display_to_native_sim_coords(
                col,
                row,
                image_shape,
                sim_display_rotate_k=SIM_DISPLAY_ROTATE_K,
            )
        ),
        select_fit_orientation=gui_geometry_overlay.select_fit_orientation,
        apply_orientation_to_entries=gui_geometry_overlay.apply_orientation_to_entries,
        orient_image_for_fit=gui_geometry_overlay.orient_image_for_fit,
    )

    (
        resolved_active_var_names,
        resolved_seed_policy,
        saved_manual_caked_defaults_enabled,
    ) = _infer_headless_saved_manual_caked_defaults(
        resolved_active_var_names,
        resolved_seed_policy,
        _headless_fixed_manual_caked_qr_pair_rows(),
    )
    if saved_manual_caked_defaults_enabled:
        progress_writer.update_static(
            active_vars=resolved_active_var_names,
            seed_policy=resolved_seed_policy,
        )

    params = value_callbacks.current_params()
    use_shared_theta_offset = bool(_geometry_fit_uses_shared_theta_offset())
    runtime_active_var_names = _canonicalize_headless_geometry_fit_active_var_names(
        resolved_active_var_names,
        use_shared_theta_offset=use_shared_theta_offset,
    )
    var_names = (
        list(runtime_active_var_names)
        if runtime_active_var_names is not None
        else list(value_callbacks.current_var_names())
    )
    if "c" in {str(name) for name in var_names}:
        var_names = [str(name) for name in var_names if str(name) != "c"]
        if not var_names:
            raise ValueError("Headless geometry fit has no active variables after excluding c.")
    progress_writer.update_static(active_vars=var_names)
    preserve_live_theta = "theta_initial" not in var_names and "theta_offset" not in var_names
    headless_fit_config = (
        copy.deepcopy(defaults.fit_config) if isinstance(defaults.fit_config, dict) else {}
    )
    if "a" in {str(name) for name in var_names}:
        geometry_cfg = headless_fit_config.get("geometry")
        if not isinstance(geometry_cfg, dict):
            geometry_cfg = {}
            headless_fit_config["geometry"] = geometry_cfg
        lattice_cfg = geometry_cfg.get("lattice_refinement")
        if not isinstance(lattice_cfg, dict):
            lattice_cfg = {}
            geometry_cfg["lattice_refinement"] = lattice_cfg
        lattice_cfg["enabled"] = True

    def _build_headless_runtime_config(_fit_params: Mapping[str, object]) -> dict[str, object]:
        base_runtime_cfg = copy.deepcopy(
            headless_fit_config.get("geometry", {})
            if isinstance(headless_fit_config, dict)
            else {}
        )
        if not isinstance(base_runtime_cfg, dict):
            base_runtime_cfg = {}
        if runtime_active_var_names is None:
            return base_runtime_cfg
        candidate_params = {
            str(name): _fit_params.get(str(name))
            for name in var_names
        }
        parameter_domains = _headless_runtime_geometry_fit_parameter_domains(
            fit_config=base_runtime_cfg,
            current_params=_fit_params,
            image_size=defaults.image_size,
            names=var_names,
            use_shared_theta_offset=use_shared_theta_offset,
        )
        return gui_geometry_fit.build_geometry_fit_runtime_config(
            base_runtime_cfg,
            candidate_params,
            {},
            parameter_domains,
            candidate_param_names=var_names,
        )

    preflight_started_at = time.monotonic()
    preparation = gui_geometry_fit.prepare_runtime_geometry_fit_run(
        params=params,
        var_names=var_names,
        preserve_live_theta=preserve_live_theta,
        bindings=gui_geometry_fit.GeometryFitRuntimePreparationBindings(
            fit_config=headless_fit_config,
            theta_initial=theta_initial_var.get(),
            apply_geometry_fit_background_selection=_apply_geometry_fit_background_selection,
            current_geometry_fit_background_indices=_current_geometry_fit_background_indices,
            geometry_fit_uses_shared_theta_offset=_geometry_fit_uses_shared_theta_offset,
            apply_background_theta_metadata=_apply_background_theta_metadata,
            current_background_theta_values=_current_background_theta_values,
            current_geometry_theta_offset=_current_geometry_theta_offset,
            ensure_geometry_fit_caked_view=_ensure_geometry_fit_caked_view,
            manual_dataset_bindings=manual_dataset_bindings,
            build_runtime_config=_build_headless_runtime_config,
        ),
    )
    if preparation.prepared_run is None:
        raise RuntimeError(str(preparation.error_text or "Geometry fit preparation failed."))
    prepared_run = preparation.prepared_run
    preflight_elapsed_s = float(max(0.0, time.monotonic() - preflight_started_at))
    headless_geometry_cfg = (
        copy.deepcopy(prepared_run.geometry_runtime_cfg)
        if isinstance(prepared_run.geometry_runtime_cfg, Mapping)
        else {}
    )
    headless_geometry_cfg[gui_geometry_fit.GEOMETRY_FIT_HEADLESS_RUNTIME_CONTEXT_FLAG] = True
    _apply_headless_geometry_fit_seed_policy(headless_geometry_cfg, resolved_seed_policy)
    budget_enabled, manual_caked_fixed_row_count = (
        _apply_headless_saved_manual_caked_budget(
            headless_geometry_cfg,
            seed_policy=resolved_seed_policy,
            prepared_run=prepared_run,
            active_var_names=var_names,
            manual_pair_rows=_headless_fixed_manual_caked_qr_pair_rows(),
        )
    )
    progress_writer.update_static(
        runtime_cfg=headless_geometry_cfg,
        manual_caked_fixed_row_count=manual_caked_fixed_row_count,
        bounded_budget_enabled=budget_enabled,
    )
    prepared_run = replace(
        prepared_run,
        start_log_sections=gui_geometry_fit.build_geometry_fit_start_log_sections(
            params=prepared_run.fit_params,
            var_names=var_names,
            dataset_infos=prepared_run.dataset_infos,
            current_dataset=prepared_run.current_dataset,
            selected_background_indices=prepared_run.selected_background_indices,
            joint_background_mode=prepared_run.joint_background_mode,
            geometry_runtime_cfg=headless_geometry_cfg,
        ),
        geometry_runtime_cfg=headless_geometry_cfg,
        stage_timing_s={
            **(
                dict(prepared_run.stage_timing_s)
                if isinstance(prepared_run.stage_timing_s, Mapping)
                else {}
            ),
            "preflight_rebind": float(preflight_elapsed_s),
        },
    )
    preparation = replace(preparation, prepared_run=prepared_run)
    progress_writer.write(
        "runtime_config_ready",
        request_build_s=preflight_elapsed_s,
        active_vars=var_names,
    )

    setup = gui_geometry_fit.build_runtime_geometry_fit_execution_setup(
        prepared_run=prepared_run,
        mosaic_params=mosaic_params,
        stamp=fit_stamp,
        downloads_dir=downloads_path,
        log_dir=get_dir("debug_log_dir"),
        simulation_runtime_state=simulation_runtime_state,
        background_runtime_state=background_state,
        theta_initial_var=theta_initial_var,
        geometry_theta_offset_var=geometry_theta_offset_var,
        current_ui_params=value_callbacks.current_ui_params,
        var_map=value_callbacks.var_map,
        background_theta_for_index=_background_theta_for_index,
        refresh_status=lambda: None,
        update_manual_pick_button_label=lambda: None,
        capture_undo_state=lambda: {},
        push_undo_state=lambda _state: None,
        replace_dataset_cache=lambda _payload: None,
        request_preview_skip_once=lambda: None,
        schedule_update=lambda: None,
        draw_overlay_records=lambda _records, _marker_limit: None,
        draw_initial_pairs_overlay=lambda _pairs, _marker_limit: None,
        set_last_overlay_state=lambda _state: None,
        set_progress_text=progress_writer.status,
        cmd_line=progress_writer.status,
        solver_inputs=gui_geometry_fit.GeometryFitRuntimeSolverInputs(
            miller=structure_state.miller,
            intensities=structure_state.intensities,
            image_size=int(defaults.image_size),
        ),
        sim_display_rotate_k=SIM_DISPLAY_ROTATE_K,
        background_display_rotate_k=DISPLAY_ROTATE_K,
        simulate_and_compare_hkl=fit_runtime.simulate_and_compare_hkl,
        aggregate_match_centers=gui_geometry_overlay.aggregate_match_centers,
        build_overlay_records=gui_geometry_overlay.build_geometry_fit_overlay_records,
        compute_frame_diagnostics=lambda records: (
            gui_geometry_overlay.compute_geometry_overlay_frame_diagnostics(
                records,
                show_caked_2d=False,
                native_detector_coords_to_caked_display_coords=None,
            )
        ),
        live_update_callback=progress_writer.live_update,
    )
    initial_fit_params = dict(value_callbacks.current_params())
    progress_writer.write("solve_start")
    execution = gui_geometry_fit.execute_runtime_geometry_fit(
        prepared_run=prepared_run,
        var_names=var_names,
        preserve_live_theta=preserve_live_theta,
        solve_fit=fit_runtime.fit_geometry_parameters,
        setup=setup,
    )
    if execution.error_text:
        raise RuntimeError(str(execution.error_text))
    if execution.apply_result is None:
        raise RuntimeError("Geometry fit finished without an apply result.")
    solver_result = getattr(execution, "solver_result", None)
    final_summary = getattr(solver_result, "point_match_summary", None)
    final_progress: dict[str, object] = {
        "accepted": bool(execution.apply_result.accepted),
        "rejection_reason": execution.apply_result.rejection_reason,
        "rms_px": execution.apply_result.rms,
    }
    if isinstance(final_summary, Mapping):
        final_progress.update(progress_writer._merge_point_match_summary(final_summary))
        final_progress["point_match_summary"] = copy.deepcopy(dict(final_summary))
    if solver_result is not None:
        final_progress["optimizer_nfev"] = getattr(solver_result, "nfev", None)
        final_progress["optimizer_njev"] = getattr(solver_result, "njev", None)
    progress_writer.write("final_validation", **final_progress)
    try:
        artifact_progress = _write_headless_geometry_fit_recovery_artifacts(
            state_path=state_path,
            output_dir=downloads_path,
            background_index=background_state.current_background_index,
            active_var_names=var_names,
            accepted=bool(execution.apply_result.accepted),
            rejection_reason=execution.apply_result.rejection_reason,
            final_summary=final_summary if isinstance(final_summary, Mapping) else None,
            progress_data=progress_writer.data,
            initial_params=initial_fit_params,
            final_params=dict(value_callbacks.current_params()),
        )
    except Exception as exc:
        progress_writer.write(
            "final_validation",
            geometry_fit_recovery_artifact_status="fail",
            geometry_fit_recovery_artifact_error=str(exc),
        )
        raise
    if artifact_progress:
        progress_writer.write("final_validation", **artifact_progress)

    updated_state = _updated_state_snapshot(saved_state, defaults, var_store)
    if manual_caked_backfill_changed_count > 0:
        geometry_out = updated_state.get("geometry")
        if not isinstance(geometry_out, dict):
            geometry_out = {}
            updated_state["geometry"] = geometry_out
        geometry_out["manual_pairs"] = gui_manual_geometry.geometry_manual_pairs_export_rows(
            pairs_by_background=pairs_by_background,
            osc_files=defaults.osc_files,
            pairs_for_index=_pairs_for_index,
        )
    if manual_caked_backfill_failed_indices:
        pass

    return HeadlessGeometryFitResult(
        state=updated_state,
        log_path=Path(execution.log_path),
        accepted=bool(execution.apply_result.accepted),
        rejection_reason=(
            str(execution.apply_result.rejection_reason)
            if execution.apply_result.rejection_reason
            else None
        ),
        rms_px=(
            float(execution.apply_result.rms) if execution.apply_result.rms is not None else None
        ),
    )
