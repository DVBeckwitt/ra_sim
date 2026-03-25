"""Import-safe package entrypoint for the RA-SIM GUI.

The live single-window GUI lives in the packaged runtime module
``ra_sim.gui.runtime``. This module keeps package imports safe by exposing a
small compatibility surface and loading the runtime only when ``main()`` is
called.
"""

from __future__ import annotations

import copy
import importlib
import math
import os
import sys
from pathlib import Path

import numpy as np

write_excel = False

# Background and simulated overlays can use different display orientations.
DISPLAY_ROTATE_K = -1
SIM_DISPLAY_ROTATE_K = 0
# hBN fitter bundles are in native OSC orientation (no rot90).
HBN_FITTER_ROTATE_K = 0
# Simulation geometry runs in native simulation pixels; convert hBN bundle
# geometry so that the displayed simulation (after SIM_DISPLAY_ROTATE_K) lands
# in the same frame as the displayed background (after DISPLAY_ROTATE_K).
SIMULATION_GEOMETRY_ROTATE_K = DISPLAY_ROTATE_K - SIM_DISPLAY_ROTATE_K

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

_RUNTIME_MODULE = None
_RUNTIME_MODULE_NAME = "ra_sim.gui.runtime"


class _PlaceholderVar:
    def __init__(self, value):
        self._value = value
        self._callbacks = []

    def get(self):
        return self._value

    def set(self, value):
        self._value = value
        for callback in list(self._callbacks):
            callback()

    def trace_add(self, _mode, callback):
        self._callbacks.append(callback)
        return f"trace-{len(self._callbacks)}"


class _PlaceholderScale:
    def __init__(self, min_value, max_value):
        self._bounds = {
            "from": float(min_value),
            "to": float(max_value),
        }

    def cget(self, key):
        return self._bounds[key]


class _PlaceholderFrame:
    def __init__(self):
        self.frame = object()


class _PlaceholderDisplay:
    def __init__(self):
        self.last_data = None

    def set_data(self, value):
        self.last_data = value


class _PlaceholderRoot:
    def after_cancel(self, _token):
        return None


def make_slider(_label, min_value, max_value, default, _step, _parent, **_kwargs):
    return _PlaceholderVar(default), _PlaceholderScale(min_value, max_value)


def _geometry_fit_uses_shared_theta_offset() -> bool:
    return False


def _current_geometry_theta_offset(*, strict: bool = False) -> float:
    _ = strict
    return 0.0


def _draw_geometry_fit_overlay(*_args, **_kwargs) -> None:
    return None


def _draw_initial_geometry_pairs_overlay(*_args, **_kwargs) -> None:
    return None


def _set_background_file_status_text() -> None:
    return None


def _update_geometry_manual_pick_button_label() -> None:
    return None


def do_update() -> None:
    return None


def read_osc(_path):
    raise RuntimeError("GUI runtime is not loaded; call ra_sim.gui.app.main().")


defaults = {"psi_z": 0.0}
geo_frame = _PlaceholderFrame()
root = _PlaceholderRoot()
background_display = _PlaceholderDisplay()

osc_files = []
background_images = []
background_images_native = []
background_images_display = []
current_background_index = 0
current_background_image = None
current_background_display = None
profile_cache = {}
last_geometry_overlay_state = None
last_simulation_signature = None
update_pending = None
geometry_theta_offset_var = None

fit_zb_var = _PlaceholderVar(False)
fit_zs_var = _PlaceholderVar(False)
fit_theta_var = _PlaceholderVar(False)
fit_psi_z_var = _PlaceholderVar(False)
fit_chi_var = _PlaceholderVar(False)
fit_cor_var = _PlaceholderVar(False)
fit_gamma_var = _PlaceholderVar(False)
fit_Gamma_var = _PlaceholderVar(False)
fit_dist_var = _PlaceholderVar(False)
fit_a_var = _PlaceholderVar(False)
fit_c_var = _PlaceholderVar(False)
fit_center_x_var = _PlaceholderVar(False)
fit_center_y_var = _PlaceholderVar(False)

zb_var = _PlaceholderVar(0.0)
zs_var = _PlaceholderVar(0.0)
theta_initial_var = _PlaceholderVar(0.0)
chi_var = _PlaceholderVar(0.0)
cor_angle_var = _PlaceholderVar(0.0)
gamma_var = _PlaceholderVar(0.0)
Gamma_var = _PlaceholderVar(0.0)
corto_detector_var = _PlaceholderVar(0.0)
a_var = _PlaceholderVar(0.0)
c_var = _PlaceholderVar(0.0)
center_x_var = _PlaceholderVar(0.0)
center_y_var = _PlaceholderVar(0.0)

psi_z_var, psi_z_scale = make_slider(
    "Goniometer Axis Yaw (about z)",
    -5.0,
    5.0,
    defaults["psi_z"],
    0.01,
    geo_frame.frame,
)


def _clamp_psi_z_var(*_):
    try:
        value = float(psi_z_var.get())
        lo = float(psi_z_scale.cget("from"))
        hi = float(psi_z_scale.cget("to"))
    except Exception:
        return
    if hi < lo:
        lo, hi = hi, lo
    clipped = min(max(value, lo), hi)
    if not math.isclose(value, clipped, rel_tol=0.0, abs_tol=1.0e-12):
        psi_z_var.set(clipped)


psi_z_var.trace_add("write", _clamp_psi_z_var)
_clamp_psi_z_var()


def get_sim_signature():
    psi_z_updated = float(psi_z_var.get())
    return (round(psi_z_updated, 6),)


def _normalize_hkl_key(
    value: object,
) -> tuple[int, int, int] | None:
    """Return a rounded integer HKL tuple when *value* looks like one."""

    if isinstance(value, str):
        parts = (
            value.replace("(", "")
            .replace(")", "")
            .replace("[", "")
            .replace("]", "")
            .split(",")
        )
        if len(parts) < 3:
            return None
        try:
            return tuple(int(np.rint(float(parts[i].strip()))) for i in range(3))
        except Exception:
            return None

    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 3:
        try:
            return tuple(int(np.rint(float(value[i]))) for i in range(3))
        except Exception:
            return None

    return None


def _copy_geometry_fit_state_value(value):
    """Deep-copy simple geometry-fit GUI state."""

    if isinstance(value, np.ndarray):
        return np.asarray(value).copy()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            key: _copy_geometry_fit_state_value(val)
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_copy_geometry_fit_state_value(val) for val in value]
    if isinstance(value, tuple):
        return tuple(_copy_geometry_fit_state_value(val) for val in value)
    return value


def _current_geometry_fit_ui_params() -> dict[str, object]:
    """Capture the current geometry-fit UI parameter values."""

    params = {
        "zb": float(zb_var.get()),
        "zs": float(zs_var.get()),
        "theta_initial": float(theta_initial_var.get()),
        "psi_z": float(psi_z_var.get()),
        "chi": float(chi_var.get()),
        "cor_angle": float(cor_angle_var.get()),
        "gamma": float(gamma_var.get()),
        "Gamma": float(Gamma_var.get()),
        "corto_detector": float(corto_detector_var.get()),
        "a": float(a_var.get()),
        "c": float(c_var.get()),
        "center_x": float(center_x_var.get()),
        "center_y": float(center_y_var.get()),
        "center": [float(center_x_var.get()), float(center_y_var.get())],
    }
    if geometry_theta_offset_var is not None:
        params["theta_offset"] = float(_current_geometry_theta_offset(strict=False))
    return params


def _current_geometry_fit_var_names() -> list[str]:
    """Return the currently selected geometry variables for LSQ fitting."""

    var_names: list[str] = []
    if fit_zb_var.get():
        var_names.append("zb")
    if fit_zs_var.get():
        var_names.append("zs")
    if fit_theta_var.get():
        var_names.append(
            "theta_offset" if _geometry_fit_uses_shared_theta_offset() else "theta_initial"
        )
    if fit_psi_z_var.get():
        var_names.append("psi_z")
    if fit_chi_var.get():
        var_names.append("chi")
    if fit_cor_var.get():
        var_names.append("cor_angle")
    if fit_gamma_var.get():
        var_names.append("gamma")
    if fit_Gamma_var.get():
        var_names.append("Gamma")
    if fit_dist_var.get():
        var_names.append("corto_detector")
    if fit_a_var.get():
        var_names.append("a")
    if fit_c_var.get():
        var_names.append("c")
    if fit_center_x_var.get():
        var_names.append("center_x")
    if fit_center_y_var.get():
        var_names.append("center_y")
    return var_names


def _restore_geometry_fit_undo_state(state: dict[str, object]) -> None:
    """Restore a previously captured geometry-fit state."""

    global profile_cache, last_geometry_overlay_state, last_simulation_signature
    global update_pending

    if not isinstance(state, dict):
        return

    ui_params = state.get("ui_params", {}) or {}
    for name, var in (
        ("zb", zb_var),
        ("zs", zs_var),
        ("theta_initial", theta_initial_var),
        ("psi_z", psi_z_var),
        ("chi", chi_var),
        ("cor_angle", cor_angle_var),
        ("gamma", gamma_var),
        ("Gamma", Gamma_var),
        ("corto_detector", corto_detector_var),
        ("a", a_var),
        ("c", c_var),
        ("center_x", center_x_var),
        ("center_y", center_y_var),
    ):
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

    profile_cache = _copy_geometry_fit_state_value(state.get("profile_cache", {})) or {}
    last_geometry_overlay_state = _copy_geometry_fit_state_value(
        state.get("overlay_state")
    )
    last_simulation_signature = None

    if update_pending is not None:
        try:
            root.after_cancel(update_pending)
        except Exception:
            pass
        update_pending = None

    do_update()

    overlay_state = last_geometry_overlay_state or {}
    overlay_records = overlay_state.get("overlay_records", []) or []
    initial_pairs_display = overlay_state.get("initial_pairs_display", []) or []
    max_display_markers = int(overlay_state.get("max_display_markers", 120))
    max_display_markers = max(1, max_display_markers)
    if overlay_records:
        _draw_geometry_fit_overlay(
            overlay_records,
            max_display_markers=max_display_markers,
        )
    elif initial_pairs_display:
        _draw_initial_geometry_pairs_overlay(
            initial_pairs_display,
            max_display_markers=max_display_markers,
        )

    _set_background_file_status_text()
    _update_geometry_manual_pick_button_label()


def _build_geometry_fit_runtime_config(
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


def _canonicalize_gui_state_background_path(path: object) -> str:
    return os.path.normcase(str(Path(str(path)).expanduser().resolve(strict=False)))


def _background_files_match_loaded_state(file_paths: list[str]) -> bool:
    if not file_paths:
        return False
    if len(file_paths) != len(osc_files):
        return False
    if len(background_images_native) != len(file_paths):
        return False
    if len(background_images_display) != len(file_paths):
        return False
    requested_paths = [
        _canonicalize_gui_state_background_path(path) for path in file_paths
    ]
    current_paths = [
        _canonicalize_gui_state_background_path(path) for path in osc_files
    ]
    return requested_paths == current_paths


def _load_background_files_for_state(
    file_paths: list[str],
    *,
    select_index: int = 0,
) -> None:
    global osc_files, background_images, background_images_native, background_images_display
    global current_background_index, current_background_image, current_background_display

    normalized_paths = [
        str(Path(str(path)).expanduser())
        for path in file_paths
        if path is not None
    ]
    if not normalized_paths:
        return
    if _background_files_match_loaded_state(normalized_paths):
        osc_files = list(normalized_paths)
        index = max(0, min(int(select_index), len(background_images_native) - 1))
        current_background_index = index
        current_background_image = background_images_native[index]
        current_background_display = background_images_display[index]
        background_display.set_data(current_background_display)
        return

    loaded_native = [np.asarray(read_osc(path)) for path in normalized_paths]
    if not loaded_native:
        return
    osc_files = list(normalized_paths)
    background_images = [np.array(img) for img in loaded_native]
    background_images_native = [np.array(img) for img in loaded_native]
    background_images_display = [
        np.rot90(img, DISPLAY_ROTATE_K) for img in background_images_native
    ]
    index = max(0, min(int(select_index), len(background_images_native) - 1))
    current_background_index = index
    current_background_image = background_images_native[index]
    current_background_display = background_images_display[index]
    background_display.set_data(current_background_display)


def _convert_hbn_bundle_geometry_reference(
    *,
    tilt_x_deg: float = 0.0,
    tilt_y_deg: float = 0.0,
    center_xy=None,
    image_size=(0, 0),
):
    from ra_sim.hbn import convert_hbn_bundle_geometry_to_simulation

    return convert_hbn_bundle_geometry_to_simulation(
        tilt_x_deg=tilt_x_deg,
        tilt_y_deg=tilt_y_deg,
        center_xy=center_xy,
        source_rotate_k=HBN_FITTER_ROTATE_K,
        target_rotate_k=SIMULATION_GEOMETRY_ROTATE_K,
        image_size=image_size,
    )


def _load_runtime_module():
    global _RUNTIME_MODULE

    if _RUNTIME_MODULE is not None:
        return _RUNTIME_MODULE

    module = importlib.import_module(_RUNTIME_MODULE_NAME)
    _RUNTIME_MODULE = module
    return module


def main(
    write_excel_flag: bool | None = None,
    startup_mode: str = "prompt",
    calibrant_bundle: str | None = None,
) -> None:
    """Launch the full GUI runtime lazily."""

    global write_excel
    if write_excel_flag is not None:
        write_excel = bool(write_excel_flag)

    runtime = _load_runtime_module()
    runtime.write_excel = write_excel
    runtime.main(
        write_excel_flag=write_excel_flag,
        startup_mode=startup_mode,
        calibrant_bundle=calibrant_bundle,
    )


if __name__ == "__main__":
    from ra_sim.launcher import main as launcher_main

    launcher_main(["gui", *sys.argv[1:]])
