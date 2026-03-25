"""Pure helpers for GUI geometry-fit state and configuration."""

from __future__ import annotations

import copy
from typing import Mapping

import numpy as np


GEOMETRY_FIT_PARAM_ORDER = [
    "zb",
    "zs",
    "theta_initial",
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
]


def copy_geometry_fit_state_value(value):
    """Deep-copy simple geometry-fit GUI state."""

    if isinstance(value, np.ndarray):
        return np.asarray(value).copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            key: copy_geometry_fit_state_value(val)
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [copy_geometry_fit_state_value(val) for val in value]
    if isinstance(value, tuple):
        return tuple(copy_geometry_fit_state_value(val) for val in value)
    return value


def current_geometry_fit_ui_params(
    *,
    zb: float,
    zs: float,
    theta_initial: float,
    psi_z: float,
    chi: float,
    cor_angle: float,
    gamma: float,
    Gamma: float,
    corto_detector: float,
    a: float,
    c: float,
    center_x: float,
    center_y: float,
    theta_offset: float | None = None,
) -> dict[str, object]:
    """Capture the current geometry-fit UI parameter values."""

    params = {
        "zb": float(zb),
        "zs": float(zs),
        "theta_initial": float(theta_initial),
        "psi_z": float(psi_z),
        "chi": float(chi),
        "cor_angle": float(cor_angle),
        "gamma": float(gamma),
        "Gamma": float(Gamma),
        "corto_detector": float(corto_detector),
        "a": float(a),
        "c": float(c),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "center": [float(center_x), float(center_y)],
    }
    if theta_offset is not None:
        params["theta_offset"] = float(theta_offset)
    return params


def current_geometry_fit_var_names(
    *,
    fit_zb: bool,
    fit_zs: bool,
    fit_theta: bool,
    fit_psi_z: bool,
    fit_chi: bool,
    fit_cor: bool,
    fit_gamma: bool,
    fit_Gamma: bool,
    fit_dist: bool,
    fit_a: bool,
    fit_c: bool,
    fit_center_x: bool,
    fit_center_y: bool,
    use_shared_theta_offset: bool = False,
) -> list[str]:
    """Return the currently selected geometry variables for LSQ fitting."""

    var_names: list[str] = []
    if fit_zb:
        var_names.append("zb")
    if fit_zs:
        var_names.append("zs")
    if fit_theta:
        var_names.append("theta_offset" if use_shared_theta_offset else "theta_initial")
    if fit_psi_z:
        var_names.append("psi_z")
    if fit_chi:
        var_names.append("chi")
    if fit_cor:
        var_names.append("cor_angle")
    if fit_gamma:
        var_names.append("gamma")
    if fit_Gamma:
        var_names.append("Gamma")
    if fit_dist:
        var_names.append("corto_detector")
    if fit_a:
        var_names.append("a")
    if fit_c:
        var_names.append("c")
    if fit_center_x:
        var_names.append("center_x")
    if fit_center_y:
        var_names.append("center_y")
    return var_names


def build_geometry_fit_runtime_config(
    base_config,
    current_params,
    control_settings,
    parameter_domains,
):
    runtime_cfg = copy.deepcopy(base_config) if isinstance(base_config, dict) else {}
    if not isinstance(runtime_cfg, dict):
        runtime_cfg = {}

    bounds_cfg = runtime_cfg.get("bounds", {}) or {}
    if not isinstance(bounds_cfg, dict):
        bounds_cfg = {}
    runtime_cfg["bounds"] = bounds_cfg

    priors_cfg = runtime_cfg.get("priors", {}) or {}
    if not isinstance(priors_cfg, dict):
        priors_cfg = {}
    runtime_cfg["priors"] = priors_cfg

    for name, current in (current_params or {}).items():
        try:
            current_value = float(current)
        except Exception:
            continue
        if not np.isfinite(current_value):
            continue

        control = (control_settings or {}).get(name, {}) or {}
        try:
            window = float(control.get("window", 0.0))
        except Exception:
            window = 0.0
        if not np.isfinite(window):
            window = 0.0
        window = max(0.0, float(window))

        lo = float(current_value - window)
        hi = float(current_value + window)

        domain = (parameter_domains or {}).get(name)
        if isinstance(domain, (list, tuple)) and len(domain) >= 2:
            try:
                domain_lo = float(domain[0])
                domain_hi = float(domain[1])
            except Exception:
                domain_lo = float("nan")
                domain_hi = float("nan")
            if np.isfinite(domain_lo):
                lo = max(lo, float(domain_lo))
            if np.isfinite(domain_hi):
                hi = min(hi, float(domain_hi))

        if hi < lo:
            lo = hi = min(max(current_value, lo), hi)

        bounds_cfg[str(name)] = [float(lo), float(hi)]

        try:
            pull = float(control.get("pull", 0.0))
        except Exception:
            pull = 0.0
        if not np.isfinite(pull):
            pull = 0.0
        pull = min(max(float(pull), 0.0), 1.0)
        if pull > 0.0 and window > 0.0:
            sigma_scale = max(0.05, 1.0 - 0.95 * pull)
            priors_cfg[str(name)] = {
                "center": float(current_value),
                "sigma": float(max(window * sigma_scale, 1.0e-6)),
            }
        else:
            priors_cfg.pop(str(name), None)

    return runtime_cfg


def apply_geometry_fit_undo_state(
    state: dict[str, object],
    *,
    var_map: Mapping[str, object],
    geometry_theta_offset_var=None,
):
    """Apply saved UI values and return copied cache/overlay state."""

    if not isinstance(state, dict):
        return {
            "profile_cache": {},
            "overlay_state": None,
        }

    ui_params = state.get("ui_params", {}) or {}
    for name, var in var_map.items():
        try:
            value = float(ui_params.get(name))
        except Exception:
            continue
        if np.isfinite(value):
            var.set(value)

    if geometry_theta_offset_var is not None:
        try:
            theta_offset = float(ui_params.get("theta_offset", 0.0))
        except Exception:
            theta_offset = 0.0
        if np.isfinite(theta_offset):
            geometry_theta_offset_var.set(f"{theta_offset:.6g}")

    overlay_state = copy_geometry_fit_state_value(state.get("overlay_state"))
    if not overlay_state:
        overlay_state = None

    return {
        "profile_cache": copy_geometry_fit_state_value(state.get("profile_cache", {})) or {},
        "overlay_state": overlay_state,
    }
