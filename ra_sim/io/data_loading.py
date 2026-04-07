"""Functions for persisting and retrieving configuration data."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

SOLVE_Q_STEPS_MIN = 32
SOLVE_Q_STEPS_MAX = 8192
SOLVE_Q_REL_TOL_MIN = 1.0e-6
SOLVE_Q_REL_TOL_MAX = 5.0e-2
GUI_STATE_FILE_TYPE = "ra_sim.gui_state"
GUI_STATE_FILE_VERSION = 1
GEOMETRY_PLACEMENTS_FILE_TYPE = "ra_sim.geometry_placements"
GEOMETRY_PLACEMENTS_FILE_VERSION = 1
_GUI_STATE_SECTION_KEYS = frozenset(
    {"variables", "dynamic_lists", "files", "flags", "geometry"}
)


def _json_safe_gui_state_value(value: Any) -> Any:
    """Return *value* converted to JSON-safe builtin types."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            return None
        return numeric
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _json_safe_gui_state_value(item)
            for key, item in value.items()
        }
    if isinstance(value, set):
        try:
            ordered = sorted(value)
        except Exception:
            ordered = list(value)
        return [_json_safe_gui_state_value(item) for item in ordered]
    if isinstance(value, (list, tuple)):
        return [_json_safe_gui_state_value(item) for item in value]
    return str(value)


def build_gui_state_payload(
    state: dict[str, Any],
    *,
    saved_at: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a versioned GUI-state payload from *state*."""

    payload: dict[str, Any] = {
        "type": GUI_STATE_FILE_TYPE,
        "version": int(GUI_STATE_FILE_VERSION),
        "saved_at": saved_at or datetime.now().isoformat(timespec="seconds"),
        "state": _json_safe_gui_state_value(state),
    }
    if metadata:
        payload["metadata"] = _json_safe_gui_state_value(metadata)
    return payload


def build_geometry_placements_payload(
    state: dict[str, Any],
    *,
    saved_at: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a versioned geometry-placement payload from *state*."""

    payload: dict[str, Any] = {
        "type": GEOMETRY_PLACEMENTS_FILE_TYPE,
        "version": int(GEOMETRY_PLACEMENTS_FILE_VERSION),
        "saved_at": saved_at or datetime.now().isoformat(timespec="seconds"),
        "state": _json_safe_gui_state_value(state),
    }
    if metadata:
        payload["metadata"] = _json_safe_gui_state_value(metadata)
    return payload


def save_gui_state_file(
    path: str | os.PathLike[str],
    state: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a JSON GUI-state snapshot and return the payload."""

    payload = build_gui_state_payload(state, metadata=metadata)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload


def save_geometry_placements_file(
    path: str | os.PathLike[str],
    state: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a JSON geometry-placement snapshot and return the payload."""

    payload = build_geometry_placements_payload(state, metadata=metadata)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload


def _looks_like_gui_state_snapshot(value: object) -> bool:
    """Return whether *value* resembles a raw GUI-state snapshot dict."""

    if not isinstance(value, dict):
        return False
    return bool(_GUI_STATE_SECTION_KEYS.intersection(value.keys()))


def _legacy_gui_state_payload_from_object(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return a normalized payload for older GUI-state JSON shapes."""

    raw_state = None
    saved_at = payload.get("saved_at")
    metadata = payload.get("metadata")

    wrapped_state = payload.get("state")
    if _looks_like_gui_state_snapshot(wrapped_state):
        raw_state = wrapped_state
    elif _looks_like_gui_state_snapshot(payload):
        raw_state = {
            str(key): value
            for key, value in payload.items()
            if str(key) not in {"saved_at", "metadata"}
        }

    if not isinstance(raw_state, dict):
        return None

    normalized_payload: dict[str, Any] = {
        "type": GUI_STATE_FILE_TYPE,
        "version": 0,
        "state": raw_state,
    }
    if isinstance(saved_at, str) and saved_at.strip():
        normalized_payload["saved_at"] = saved_at
    if isinstance(metadata, dict):
        normalized_payload["metadata"] = metadata
    return normalized_payload


def load_gui_state_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and validate a JSON GUI-state snapshot."""

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("GUI state file must contain a JSON object.")
    payload_type = str(payload.get("type", "")).strip()
    if payload_type:
        if payload_type != GUI_STATE_FILE_TYPE:
            raise ValueError("Unsupported GUI state file type.")
        state = payload.get("state")
        if not isinstance(state, dict):
            raise ValueError("GUI state file is missing a valid 'state' object.")
        return payload

    normalized_payload = _legacy_gui_state_payload_from_object(payload)
    if normalized_payload is None:
        raise ValueError("Unsupported GUI state file type.")
    state = normalized_payload.get("state")
    if not isinstance(state, dict):
        raise ValueError("GUI state file is missing a valid 'state' object.")
    return normalized_payload


def load_geometry_placements_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and validate a JSON manual-placement snapshot."""

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Geometry placements file must contain a JSON object.")
    if str(payload.get("type", "")) != GEOMETRY_PLACEMENTS_FILE_TYPE:
        raise ValueError("Unsupported geometry placements file type.")
    state = payload.get("state")
    if not isinstance(state, dict):
        raise ValueError("Geometry placements file is missing a valid 'state' object.")
    return payload


def _normalize_optics_mode(value, fallback="fast"):
    """Return optics-mode label as ``'fast'`` or ``'exact'``."""

    if value is None:
        return fallback

    # Support legacy numeric storage and bool-like flags.
    if isinstance(value, (int, np.integer)):
        return "exact" if int(value) == 1 else "fast"
    if isinstance(value, (float, np.floating)):
        return "exact" if int(round(float(value))) == 1 else "fast"

    text = str(value).strip().lower()
    text = text.replace("–", "-").replace("—", "-")
    text = " ".join(text.split())

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
    if "fresnel" in text and "ctr" in text:
        return "fast"

    return fallback


def _normalize_solve_q_mode(value, fallback="uniform"):
    """Return solve-q mode label as ``'uniform'`` or ``'adaptive'``."""

    if value is None:
        return fallback

    if isinstance(value, (int, np.integer)):
        return "uniform" if int(value) == 0 else "adaptive"
    if isinstance(value, (float, np.floating)):
        return "uniform" if int(round(float(value))) == 0 else "adaptive"

    text = str(value).strip().lower()
    if text in {"uniform", "fast", "0"}:
        return "uniform"
    if text in {"adaptive", "robust", "1"}:
        return "adaptive"
    return fallback


def load_parameters(
    path,
    theta_initial_var,
    cor_angle_var,
    gamma_var,
    Gamma_var,
    chi_var,
    zs_var,
    zb_var,
    sample_width_var,
    sample_length_var,
    sample_depth_var,
    debye_x_var,
    debye_y_var,
    corto_detector_var,
    sigma_mosaic_var,
    gamma_mosaic_var,
    eta_var,
    a_var,
    c_var,
    center_x_var,      # <--- ADDED
    center_y_var,      # <--- ADDED
    resolution_var=None,
    custom_samples_var=None,
    rod_points_per_gz_var=None,
    bandwidth_percent_var=None,
    optics_mode_var=None,
    phase_delta_expr_var=None,
    iodine_z_var=None,
    phi_l_divisor_var=None,
    sf_prune_bias_var=None,
    solve_q_steps_var=None,
    solve_q_rel_tol_var=None,
    solve_q_mode_var=None,
):
    """
    Load slider parameters from a .npy file (dictionary). If the file does not exist,
    return a message. This now includes center_x and center_y.
    """
    # Backward compatibility for legacy positional callsites that passed
    # ``optics_mode_var`` in the old argument slot (now bandwidth).
    if optics_mode_var is None and bandwidth_percent_var is not None:
        try:
            legacy_mode = str(bandwidth_percent_var.get())
        except Exception:
            legacy_mode = ""
        if _normalize_optics_mode(legacy_mode, fallback="") in {"fast", "exact"}:
            optics_mode_var = bandwidth_percent_var
            bandwidth_percent_var = None

    if os.path.exists(path):
        params = np.load(path, allow_pickle=True).item()
        
        # Set all the old parameters
        theta_initial_var.set(params.get('theta_initial', theta_initial_var.get()))
        cor_angle_var.set(params.get('cor_angle', cor_angle_var.get()))
        gamma_var.set(params.get('gamma', gamma_var.get()))
        Gamma_var.set(params.get('Gamma', Gamma_var.get()))
        chi_var.set(params.get('chi', chi_var.get()))
        zs_var.set(params.get('zs', zs_var.get()))
        zb_var.set(params.get('zb', zb_var.get()))
        sample_width_var.set(params.get('sample_width_m', sample_width_var.get()))
        sample_length_var.set(params.get('sample_length_m', sample_length_var.get()))
        sample_depth_var.set(params.get('sample_depth_m', sample_depth_var.get()))
        debye_x_var.set(params.get('debye_x', debye_x_var.get()))
        debye_y_var.set(params.get('debye_y', debye_y_var.get()))
        corto_detector_var.set(params.get('corto_detector', corto_detector_var.get()))
        sigma_mosaic_var.set(params.get('sigma_mosaic', sigma_mosaic_var.get()))
        gamma_mosaic_var.set(params.get('gamma_mosaic', gamma_mosaic_var.get()))
        eta_var.set(params.get('eta', eta_var.get()))
        a_var.set(params.get('a', a_var.get()))
        c_var.set(params.get('c', c_var.get()))
        
        # Set the new beam center parameters
        center_x_var.set(params.get('center_x', center_x_var.get()))
        center_y_var.set(params.get('center_y', center_y_var.get()))
        if custom_samples_var is not None:
            stored_custom_count = params.get(
                'sampling_custom_count',
                params.get('sampling_count'),
            )
            if stored_custom_count is not None:
                try:
                    parsed_custom_count = int(round(float(stored_custom_count)))
                except (TypeError, ValueError):
                    parsed_custom_count = None
                if parsed_custom_count is not None and parsed_custom_count > 0:
                    custom_samples_var.set(str(parsed_custom_count))
        if rod_points_per_gz_var is not None:
            stored_rod_points = params.get('rod_points_per_gz')
            if stored_rod_points is not None:
                try:
                    parsed_rod_points = int(round(float(stored_rod_points)))
                except (TypeError, ValueError):
                    parsed_rod_points = None
                if parsed_rod_points is not None and parsed_rod_points > 0:
                    rod_points_per_gz_var.set(parsed_rod_points)
        if bandwidth_percent_var is not None:
            stored_bandwidth = params.get('bandwidth_percent')
            if stored_bandwidth is not None:
                try:
                    bandwidth_val = float(stored_bandwidth)
                except (TypeError, ValueError):
                    bandwidth_val = None
                if bandwidth_val is not None and np.isfinite(bandwidth_val):
                    bandwidth_percent_var.set(float(np.clip(bandwidth_val, 0.0, 10.0)))
        if resolution_var is not None:
            stored_resolution = params.get('sampling_resolution')
            if stored_resolution:
                resolution_var.set(stored_resolution)
        if optics_mode_var is not None:
            current_mode = _normalize_optics_mode(optics_mode_var.get(), fallback="fast")
            stored_mode = _normalize_optics_mode(
                params.get('optics_mode', current_mode),
                fallback=current_mode,
            )
            optics_mode_var.set(stored_mode)
        if iodine_z_var is not None:
            stored_iodine = params.get('iodine_z')
            if stored_iodine is not None:
                try:
                    iodine_val = float(stored_iodine)
                except (TypeError, ValueError):
                    iodine_val = None
                if iodine_val is not None and np.isfinite(iodine_val):
                    iodine_z_var.set(float(np.clip(iodine_val, 0.0, 1.0)))
        if phase_delta_expr_var is not None:
            stored_expr = params.get('phase_delta_expression')
            if stored_expr is not None:
                phase_delta_expr_var.set(str(stored_expr))
        if phi_l_divisor_var is not None:
            stored_divisor = params.get('phi_l_divisor')
            if stored_divisor is not None:
                try:
                    divisor_val = float(stored_divisor)
                except (TypeError, ValueError):
                    divisor_val = None
                if divisor_val is not None and np.isfinite(divisor_val) and divisor_val > 0.0:
                    phi_l_divisor_var.set(float(divisor_val))
        if sf_prune_bias_var is not None:
            stored_bias = params.get('sf_prune_bias')
            if stored_bias is not None:
                try:
                    bias_val = float(stored_bias)
                except (TypeError, ValueError):
                    bias_val = None
                if bias_val is not None and np.isfinite(bias_val):
                    sf_prune_bias_var.set(float(np.clip(bias_val, -2.0, 2.0)))
        if solve_q_steps_var is not None:
            stored_steps = params.get('solve_q_steps')
            if stored_steps is not None:
                try:
                    steps_val = int(round(float(stored_steps)))
                except (TypeError, ValueError):
                    steps_val = None
                if steps_val is not None and np.isfinite(steps_val):
                    steps_clipped = int(np.clip(steps_val, SOLVE_Q_STEPS_MIN, SOLVE_Q_STEPS_MAX))
                    solve_q_steps_var.set(float(steps_clipped))
        if solve_q_rel_tol_var is not None:
            stored_tol = params.get('solve_q_rel_tol')
            if stored_tol is not None:
                try:
                    tol_val = float(stored_tol)
                except (TypeError, ValueError):
                    tol_val = None
                if tol_val is not None and np.isfinite(tol_val):
                    solve_q_rel_tol_var.set(
                        float(np.clip(tol_val, SOLVE_Q_REL_TOL_MIN, SOLVE_Q_REL_TOL_MAX))
                    )
        if solve_q_mode_var is not None:
            current_mode = _normalize_solve_q_mode(solve_q_mode_var.get(), fallback="uniform")
            stored_mode = _normalize_solve_q_mode(
                params.get('solve_q_mode', current_mode),
                fallback=current_mode,
            )
            solve_q_mode_var.set(stored_mode)

        return "Parameters loaded from parameters.npy"
    else:
        return "No parameters.npy file found to load."

def save_all_parameters(
    filepath,
    theta_initial_var,
    cor_angle_var,
    gamma_var,
    Gamma_var,
    chi_var,
    zs_var,
    zb_var,
    sample_width_var,
    sample_length_var,
    sample_depth_var,
    debye_x_var,
    debye_y_var,
    corto_detector_var,
    sigma_mosaic_var,
    gamma_mosaic_var,
    eta_var,
    a_var,
    c_var,
    center_x_var,    # <--- ADDED
    center_y_var,    # <--- ADDED
    resolution_var=None,
    custom_samples_var=None,
    rod_points_per_gz_var=None,
    bandwidth_percent_var=None,
    optics_mode_var=None,
    phase_delta_expr_var=None,
    iodine_z_var=None,
    phi_l_divisor_var=None,
    sf_prune_bias_var=None,
    solve_q_steps_var=None,
    solve_q_rel_tol_var=None,
    solve_q_mode_var=None,
):
    """
    Save all slider parameters into a .npy file as a dictionary. This now
    includes beam center (center_x, center_y).
    """
    # Backward compatibility for legacy positional callsites that passed
    # ``optics_mode_var`` in the old argument slot (now bandwidth).
    if optics_mode_var is None and bandwidth_percent_var is not None:
        try:
            legacy_mode = str(bandwidth_percent_var.get())
        except Exception:
            legacy_mode = ""
        if _normalize_optics_mode(legacy_mode, fallback="") in {"fast", "exact"}:
            optics_mode_var = bandwidth_percent_var
            bandwidth_percent_var = None

    parameters = {
        'theta_initial':  theta_initial_var.get(),
        'cor_angle':      cor_angle_var.get(),
        'gamma':          gamma_var.get(),
        'Gamma':          Gamma_var.get(),
        'chi':            chi_var.get(),
        'zs':             zs_var.get(),
        'zb':             zb_var.get(),
        'sample_width_m': sample_width_var.get(),
        'sample_length_m': sample_length_var.get(),
        'sample_depth_m': sample_depth_var.get(),
        'debye_x':        debye_x_var.get(),
        'debye_y':        debye_y_var.get(),
        'corto_detector': corto_detector_var.get(),
        'sigma_mosaic':   sigma_mosaic_var.get(),
        'gamma_mosaic':   gamma_mosaic_var.get(),
        'eta':            eta_var.get(),
        'a':              a_var.get(),
        'c':              c_var.get(),
        # Beam center
        'center_x':       center_x_var.get(),
        'center_y':       center_y_var.get(),
    }
    if resolution_var is not None:
        resolution_value = resolution_var.get()
        parameters['sampling_resolution'] = resolution_value
    else:
        resolution_value = None

    if custom_samples_var is not None:
        try:
            custom_sample_count = int(round(float(custom_samples_var.get())))
        except (TypeError, ValueError):
            custom_sample_count = None
        if custom_sample_count is not None and custom_sample_count > 0:
            parameters['sampling_custom_count'] = custom_sample_count
            if resolution_value == "Custom":
                parameters['sampling_count'] = custom_sample_count
    if rod_points_per_gz_var is not None:
        try:
            rod_points_per_gz = int(round(float(rod_points_per_gz_var.get())))
        except (TypeError, ValueError):
            rod_points_per_gz = None
        if rod_points_per_gz is not None and rod_points_per_gz > 0:
            parameters['rod_points_per_gz'] = rod_points_per_gz
    if bandwidth_percent_var is not None:
        try:
            bandwidth_percent = float(bandwidth_percent_var.get())
        except (TypeError, ValueError):
            bandwidth_percent = None
        if bandwidth_percent is not None and np.isfinite(bandwidth_percent):
            parameters['bandwidth_percent'] = float(np.clip(bandwidth_percent, 0.0, 10.0))

    if optics_mode_var is not None:
        parameters['optics_mode'] = _normalize_optics_mode(
            optics_mode_var.get(),
            fallback="fast",
        )
    if iodine_z_var is not None:
        parameters['iodine_z'] = float(iodine_z_var.get())
    if phase_delta_expr_var is not None:
        parameters['phase_delta_expression'] = str(phase_delta_expr_var.get())
    if phi_l_divisor_var is not None:
        try:
            phi_divisor = float(phi_l_divisor_var.get())
        except (TypeError, ValueError):
            phi_divisor = None
        if phi_divisor is not None and np.isfinite(phi_divisor) and phi_divisor > 0.0:
            parameters['phi_l_divisor'] = float(phi_divisor)
    if sf_prune_bias_var is not None:
        try:
            sf_prune_bias = float(sf_prune_bias_var.get())
        except (TypeError, ValueError):
            sf_prune_bias = None
        if sf_prune_bias is not None and np.isfinite(sf_prune_bias):
            parameters['sf_prune_bias'] = float(np.clip(sf_prune_bias, -2.0, 2.0))
    if solve_q_steps_var is not None:
        try:
            solve_q_steps = int(round(float(solve_q_steps_var.get())))
        except (TypeError, ValueError):
            solve_q_steps = None
        if solve_q_steps is not None and np.isfinite(solve_q_steps):
            parameters['solve_q_steps'] = int(
                np.clip(solve_q_steps, SOLVE_Q_STEPS_MIN, SOLVE_Q_STEPS_MAX)
            )
    if solve_q_rel_tol_var is not None:
        try:
            solve_q_rel_tol = float(solve_q_rel_tol_var.get())
        except (TypeError, ValueError):
            solve_q_rel_tol = None
        if solve_q_rel_tol is not None and np.isfinite(solve_q_rel_tol):
            parameters['solve_q_rel_tol'] = float(
                np.clip(solve_q_rel_tol, SOLVE_Q_REL_TOL_MIN, SOLVE_Q_REL_TOL_MAX)
            )
    if solve_q_mode_var is not None:
        parameters['solve_q_mode'] = _normalize_solve_q_mode(
            solve_q_mode_var.get(),
            fallback="uniform",
        )
    np.save(filepath, parameters)
    print(f"Parameters saved successfully to {filepath}")


def load_background_image(file_path):
    """
    Example function to load an ASCII file (with 6 header lines, 3000x3000 pixel data).
    If needed, adapt for your real background file structure or use other formats.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        pixel_lines = lines[6:]
        pixels = [list(map(int, line.split())) for line in pixel_lines]
        flattened_pixels = np.array(pixels).flatten()
        image = flattened_pixels.reshape((3000, 3000))
    return image


def load_and_format_reference_profiles(input_filename):
    """
    Loads a .npy file produced by some preprocessing. Returns a dict:
      {
        region_name: {
          "Radial (2θ)": np.array(...),
          "Radial Intensity": np.array(...),
          "Azimuthal φ": np.array(...),
          "Azimuthal Intensity": np.array(...),
          "FittedParams": {...}
        },
        ...
      }
    """
    raw_data = np.load(input_filename, allow_pickle=True).item()
    reference_profiles = {}

    # "Regions" dictionary from raw_data
    regions_dict = raw_data.get("Regions", {})
    for region_name, region_info in regions_dict.items():
        radial_dict = region_info.get("Radial", {})
        azimuthal_dict = region_info.get("Azimuthal", {})
        fitted_params = region_info.get("FittedParams", {})

        radial_2theta = np.array(radial_dict.get("2θ", []), dtype=float)
        radial_intensity = np.array(radial_dict.get("Intensity", []), dtype=float)
        azimuthal_phi = np.array(azimuthal_dict.get("φ", []), dtype=float)
        azimuthal_intensity = np.array(azimuthal_dict.get("Intensity", []), dtype=float)

        reference_profiles[region_name] = {
            "Radial (2θ)": radial_2theta,
            "Radial Intensity": radial_intensity,
            "Azimuthal φ": azimuthal_phi,
            "Azimuthal Intensity": azimuthal_intensity,
            "FittedParams": fitted_params
        }

    return reference_profiles
