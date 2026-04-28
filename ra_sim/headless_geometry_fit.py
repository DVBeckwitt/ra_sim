"""Run geometry fitting from a saved GUI state without launching Tk."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import json
import math
import time
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from ra_sim.config import get_dir, get_instrument_config, get_path
from ra_sim.fitting.diffuse_background import (
    DiffuseBackgroundConfig,
    diffuse_background_config_from_mapping,
    diffuse_background_config_to_mapping,
    fit_diffuse_background_native,
)
from ra_sim.io.file_parsing import parse_poni_file
from ra_sim.io.osc_reader import read_osc

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


def _headless_background_subtraction_config(
    saved_state: Mapping[str, object],
    defaults: _RuntimeDefaults,
    *,
    mode_override: object | None,
    scale_override: object | None,
    diagnostics_override: bool | None,
) -> DiffuseBackgroundConfig:
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
    config_mapping = diffuse_background_config_to_mapping(
        diffuse_background_config_from_mapping(
            config_cfg if isinstance(config_cfg, Mapping) else None
        )
    )

    saved_variables = (
        saved_state.get("variables", {})
        if isinstance(saved_state.get("variables"), Mapping)
        else {}
    )
    for key in tuple(config_mapping):
        saved_name = f"background_subtraction_{key}_var"
        if saved_name in saved_variables:
            config_mapping[key] = saved_variables[saved_name]

    override_text = (
        ""
        if mode_override is None
        else str(mode_override).strip().lower().replace("_", "-")
    )
    if override_text and override_text != "saved":
        if override_text == "off":
            config_mapping["enabled"] = False
            config_mapping["mode"] = "off"
        else:
            config_mapping["enabled"] = True
            config_mapping["mode"] = override_text

    if scale_override is not None:
        config_mapping["scale"] = scale_override
    if diagnostics_override is not None:
        config_mapping["diagnostics"] = bool(diagnostics_override)

    return diffuse_background_config_from_mapping(config_mapping)


def _headless_fit_background_subtraction_for_image(
    backend_image: np.ndarray,
    *,
    params: Mapping[str, object],
    pixel_size_m: float,
    config: DiffuseBackgroundConfig,
) -> dict[str, object] | None:
    if not (config.enabled and config.mode != "off"):
        return None
    image = np.asarray(backend_image, dtype=np.float64)
    if image.ndim != 2:
        return None
    detector_shape = tuple(int(v) for v in image.shape[:2])
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
        try:
            two_theta = ai.twoThetaArray(shape=detector_shape, unit="2th_deg")
        except TypeError:
            two_theta = np.rad2deg(ai.twoThetaArray(shape=detector_shape))
        try:
            raw_phi = ai.chiArray(shape=detector_shape, unit="deg")
        except TypeError:
            raw_phi = np.rad2deg(ai.chiArray(shape=detector_shape))
        phi = exact_cake.raw_phi_to_gui_phi(raw_phi)
    except Exception:
        return None

    finite_two_theta = np.asarray(two_theta, dtype=np.float64)
    finite_two_theta = finite_two_theta[np.isfinite(finite_two_theta)]
    radial_axis = None
    if finite_two_theta.size:
        radial_axis = np.linspace(
            float(np.nanmin(finite_two_theta)),
            float(np.nanmax(finite_two_theta)),
            500,
        )
    azimuth_axis = np.linspace(-180.0, 180.0, 361)
    try:
        return fit_diffuse_background_native(
            image,
            two_theta_deg=np.asarray(two_theta, dtype=np.float64),
            phi_deg=np.asarray(phi, dtype=np.float64),
            caked_radial_axis_deg=radial_axis,
            caked_azimuth_axis_deg=azimuth_axis,
            config=config,
            direct_beam_center_rc=(float(center[0]), float(center[1])),
        )
    except Exception:
        return None


def _headless_background_subtraction_json_safe(value: object) -> object:
    if isinstance(value, DiffuseBackgroundConfig):
        return diffuse_background_config_to_mapping(value)
    if isinstance(value, Mapping):
        return {
            str(key): _headless_background_subtraction_json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_headless_background_subtraction_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_headless_background_subtraction_diagnostics(
    result: Mapping[str, object],
    *,
    output_dir: Path,
    cache_signature_summary: object,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = result.get("config")
    diagnostics = dict(result.get("diagnostics", {}) or {})
    diagnostics["cache_signature_summary"] = cache_signature_summary
    (output_dir / "background_subtraction_config.json").write_text(
        json.dumps(
            _headless_background_subtraction_json_safe(config),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_dir / "background_subtraction_diagnostics.json").write_text(
        json.dumps(
            _headless_background_subtraction_json_safe(diagnostics),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    radial_centers = np.asarray(result.get("radial_bin_centers_deg", []), dtype=np.float64)
    radial_profile = np.asarray(result.get("radial_profile", []), dtype=np.float64)
    with (output_dir / "background_radial_profile.csv").open("w", encoding="utf-8") as handle:
        handle.write("two_theta_deg,background\n")
        for theta_value, profile_value in zip(radial_centers, radial_profile, strict=False):
            handle.write(f"{float(theta_value):.12g},{float(profile_value):.12g}\n")
    for key, filename in (
        ("raw", "background_raw_native.npy"),
        ("model", "background_model_native.npy"),
        ("corrected", "background_subtracted_native.npy"),
        ("valid_mask", "background_valid_mask.npy"),
        ("exclusion_mask", "background_exclusion_mask.npy"),
    ):
        value = result.get(key)
        if value is not None:
            np.save(output_dir / filename, np.asarray(value))


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

    center_default = [
        float(poni2 / pixel_size_m),
        float(image_size - (poni1 / pixel_size_m)),
    ]
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


HEADLESS_GEOMETRY_FIT_SEED_POLICY_LADDER_MULTISTART = "ladder-multistart"
_HEADLESS_GEOMETRY_FIT_SEED_POLICIES = frozenset(
    {HEADLESS_GEOMETRY_FIT_SEED_POLICY_LADDER_MULTISTART}
)
_HEADLESS_GEOMETRY_FIT_LADDER_SEED_SEARCH = {
    "prescore_top_k": 4,
    "n_global": 4,
    "n_jitter": 2,
    "min_seed_separation_u": 0.5,
}


def normalize_headless_geometry_fit_seed_policy(seed_policy: object | None) -> str | None:
    """Normalize one optional headless seed-policy override."""

    if seed_policy is None:
        return None
    seed_policy_text = str(seed_policy).strip()
    if not seed_policy_text:
        raise ValueError("Headless geometry-fit seed policy cannot be empty.")
    if seed_policy_text not in _HEADLESS_GEOMETRY_FIT_SEED_POLICIES:
        allowed = ", ".join(sorted(_HEADLESS_GEOMETRY_FIT_SEED_POLICIES))
        raise ValueError(
            f"Unsupported headless geometry-fit seed policy {seed_policy_text!r}; "
            f"expected one of: {allowed}."
        )
    return seed_policy_text


def _apply_headless_geometry_fit_seed_policy(
    runtime_cfg: dict[str, object],
    seed_policy: str | None,
) -> None:
    """Apply one opt-in headless seed policy without changing saved-state defaults."""

    if seed_policy is None:
        return
    if seed_policy != HEADLESS_GEOMETRY_FIT_SEED_POLICY_LADDER_MULTISTART:
        raise ValueError(f"Unsupported headless geometry-fit seed policy {seed_policy!r}.")
    seed_search_cfg = runtime_cfg.get("seed_search")
    if isinstance(seed_search_cfg, Mapping):
        seed_search = dict(seed_search_cfg)
    else:
        seed_search = {}
    seed_search.update(_HEADLESS_GEOMETRY_FIT_LADDER_SEED_SEARCH)
    runtime_cfg["seed_search"] = seed_search


def run_headless_geometry_fit(
    saved_state: dict[str, object],
    *,
    state_path: str | Path,
    downloads_dir: str | Path | None = None,
    stamp: str | None = None,
    active_var_names: Sequence[object] | str | None = None,
    seed_policy: object | None = None,
    weighted_event_workers: int | None = None,
    background_subtraction_mode: object | None = None,
    background_subtraction_scale: object | None = None,
    background_subtraction_diagnostics: bool | None = None,
) -> HeadlessGeometryFitResult:
    """Run the geometry fit described by ``saved_state`` and return the updated state."""

    if not isinstance(saved_state, dict):
        raise ValueError("Saved GUI state must be a dictionary.")
    resolved_active_var_names = normalize_headless_geometry_fit_active_var_names(
        active_var_names
    )
    resolved_seed_policy = normalize_headless_geometry_fit_seed_policy(seed_policy)
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
    background_subtraction_config = _headless_background_subtraction_config(
        saved_state,
        defaults,
        mode_override=background_subtraction_mode,
        scale_override=background_subtraction_scale,
        diagnostics_override=background_subtraction_diagnostics,
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
    background_subtraction_diagnostics_written: set[tuple[object, ...]] = set()

    def _manual_current_background_uses_caked_space() -> bool:
        try:
            background_idx = int(background_state.current_background_index)
        except Exception:
            return False
        return gui_geometry_fit.geometry_manual_pairs_use_caked_fit_space(
            _pairs_for_index(background_idx)
        )

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
            tuple(
                sorted(
                    diffuse_background_config_to_mapping(
                        background_subtraction_config
                    ).items()
                )
            ),
            source_signature,
        )

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
            "background_raw",
            "background_model",
            "background_subtracted",
            "background_subtraction_config",
            "background_subtraction_diagnostics",
        ):
            if key in payload:
                hydrated_payload[key] = payload[key]
        hydrated_payload["projection_view_mode"] = "caked"
        hydrated_payload["headless_caked_payload_signature"] = signature
        return hydrated_payload

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
        backend_image = raw_backend_image
        subtraction_result = None
        if (
            background_subtraction_config.enabled
            and background_subtraction_config.apply_to_fit
        ):
            subtraction_result = _headless_fit_background_subtraction_for_image(
                raw_backend_image,
                params=params_local,
                pixel_size_m=float(defaults.pixel_size_m),
                config=background_subtraction_config,
            )
            if isinstance(subtraction_result, Mapping):
                corrected = np.asarray(
                    subtraction_result.get("corrected"),
                    dtype=np.float64,
                )
                if corrected.shape == raw_backend_image.shape:
                    backend_image = corrected
                if background_subtraction_config.diagnostics:
                    diagnostic_signature = (int(background_idx), payload_signature)
                    if diagnostic_signature not in background_subtraction_diagnostics_written:
                        _write_headless_background_subtraction_diagnostics(
                            subtraction_result,
                            output_dir=downloads_path,
                            cache_signature_summary=_signature_summary(
                                payload_signature
                            ),
                        )
                        background_subtraction_diagnostics_written.add(
                            diagnostic_signature
                        )
        payload = _build_headless_geometry_fit_caked_view_payload(
            backend_image,
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
        if isinstance(subtraction_result, Mapping):
            payload.update(
                {
                    "background_raw": raw_backend_image,
                    "background_model": subtraction_result.get("model"),
                    "background_subtracted": subtraction_result.get("corrected"),
                    "background_subtraction_config": background_subtraction_config,
                    "background_subtraction_diagnostics": subtraction_result.get(
                        "diagnostics"
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

    def _geometry_fit_required_background_indices() -> list[int]:
        selected = [int(idx) for idx in _current_geometry_fit_background_indices(strict=True)]
        if _geometry_fit_uses_shared_theta_offset(selected):
            return selected
        current_idx = int(background_state.current_background_index)
        if current_idx in set(selected):
            return [current_idx]
        return [int(selected[0])] if selected else [current_idx]

    def _ensure_geometry_fit_caked_view() -> None:
        previous_background_idx = int(background_state.current_background_index)
        try:
            for background_idx in _geometry_fit_required_background_indices():
                if not gui_geometry_fit.geometry_manual_pairs_use_caked_fit_space(
                    _pairs_for_index(int(background_idx))
                ):
                    continue
                if _geometry_fit_caked_view_for_index(int(background_idx)) is None:
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

    def _headless_current_caked_view_for_callbacks() -> dict[str, object] | None:
        try:
            background_idx = int(background_state.current_background_index)
        except Exception:
            return None
        if not gui_geometry_fit.geometry_manual_pairs_use_caked_fit_space(
            _pairs_for_index(background_idx)
        ):
            return None
        payload = _geometry_fit_caked_view_for_index(background_idx)
        return payload if isinstance(payload, dict) else None

    def _headless_caked_payload_value(key: str) -> object:
        payload = _headless_current_caked_view_for_callbacks()
        if not isinstance(payload, Mapping):
            return None
        return payload.get(key)

    def _headless_wrap_phi_range(value: object) -> object:
        return _load_exact_cake_portable_module().raw_phi_to_gui_phi(value)

    projection_callbacks = gui_manual_geometry.make_runtime_geometry_manual_projection_callbacks(
        caked_view_enabled=lambda: isinstance(
            _headless_current_caked_view_for_callbacks(),
            Mapping,
        ),
        last_caked_background_image_unscaled=lambda: _headless_caked_payload_value("background"),
        last_caked_radial_values=lambda: _headless_caked_payload_value("radial_axis"),
        last_caked_azimuth_values=lambda: _headless_caked_payload_value("azimuth_axis"),
        current_background_display=_current_background_display,
        current_background_native=_current_background_native,
        ai=lambda: _headless_caked_payload_value("ai"),
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
    ) -> list[dict[str, object]]:
        normalized_rows = [dict(entry) for entry in (rows or ()) if isinstance(entry, Mapping)]
        if not normalized_rows:
            return []
        background_idx = int(background_index)
        if not gui_geometry_fit.geometry_manual_pairs_use_caked_fit_space(
            _pairs_for_index(background_idx)
        ):
            return [
                dict(entry)
                for entry in (
                    projection_callbacks.project_peaks_to_current_view(normalized_rows) or ()
                )
                if isinstance(entry, Mapping)
            ]
        previous_background_idx = int(background_state.current_background_index)
        try:
            payload = _geometry_fit_caked_view_for_index(background_idx)
            if not isinstance(payload, Mapping):
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
                    last_caked_background_image_unscaled=lambda: payload.get("background"),
                    last_caked_radial_values=lambda: payload.get("radial_axis"),
                    last_caked_azimuth_values=lambda: payload.get("azimuth_axis"),
                    current_background_display=lambda: display_background,
                    current_background_native=lambda: native_background,
                    ai=lambda: payload.get("ai"),
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

    def _simulate_hit_tables_for_fit(
        miller_array: np.ndarray,
        intensity_array: np.ndarray,
        image_size_value: int,
        params_local: Mapping[str, object],
    ) -> list[object]:
        return simulation_callbacks.simulate_hit_tables(
            np.asarray(miller_array, dtype=np.float64),
            np.asarray(intensity_array, dtype=np.float64),
            int(image_size_value),
            dict(params_local),
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
        ) -> tuple[list[dict[str, object]], list[tuple[float, float, str]], list[object]]:
            schema = _load_intersection_cache_schema()
            table_list = list(source_tables or ())
            if not table_list:
                return [], [], []
            if schema.is_intersection_cache_table(table_list[0]):
                hit_tables_local = diffraction.intersection_cache_to_hit_tables(table_list)
            else:
                hit_tables_local = _copy_hit_tables(table_list)
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
                lambda normalized_params: _simulate_hit_tables_for_fit(
                    structure_state.miller,
                    structure_state.intensities,
                    int(defaults.image_size),
                    normalized_params,
                )
            ),
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
        geometry_manual_simulated_peaks_for_params=projection_callbacks.simulated_peaks_for_params,
        geometry_manual_simulated_lookup=projection_callbacks.simulated_lookup,
        geometry_manual_source_rows_for_background=_geometry_manual_source_rows_for_background,
        geometry_manual_rebuild_source_rows_for_background=(
            _geometry_manual_rebuild_source_rows_for_background
        ),
        geometry_manual_last_source_snapshot_diagnostics=_geometry_manual_last_source_snapshot_diagnostics,
        geometry_manual_last_simulation_diagnostics=simulation_callbacks.last_simulation_diagnostics,
        geometry_manual_match_config=lambda: (
            gui_manual_geometry.current_geometry_manual_match_config(defaults.fit_config)
        ),
        pick_uses_caked_space=_manual_current_background_uses_caked_space,
        geometry_manual_caked_view_for_index=_geometry_fit_caked_view_for_index,
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
    preserve_live_theta = "theta_initial" not in var_names and "theta_offset" not in var_names

    def _build_headless_runtime_config(_fit_params: Mapping[str, object]) -> dict[str, object]:
        base_runtime_cfg = copy.deepcopy(
            defaults.fit_config.get("geometry", {})
            if isinstance(defaults.fit_config, dict)
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
            fit_config=defaults.fit_config,
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
    _apply_headless_geometry_fit_seed_policy(headless_geometry_cfg, resolved_seed_policy)
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
        set_progress_text=lambda _text: None,
        cmd_line=lambda _text: None,
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
    )
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

    return HeadlessGeometryFitResult(
        state=_updated_state_snapshot(saved_state, defaults, var_store),
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
