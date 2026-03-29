"""Import-safe compatibility wrapper for the packaged GUI runtime.

The canonical simulation runtime entrypoint lives in ``ra_sim.gui.runtime``.
This module keeps the historical ``ra_sim.gui.app.main(...)`` surface
available for callers that still import it directly.
"""

from __future__ import annotations

import copy
import math
import os
import sys
from typing import Mapping

import numpy as np
from ra_sim.gui import geometry_fit as gui_geometry_fit
from ra_sim.gui import lazy_runtime as gui_lazy_runtime
from ra_sim.gui import controllers as gui_controllers
from ra_sim.gui import state_io as gui_state_io

write_excel = False
__all__ = ["main", "write_excel"]

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
    *gui_geometry_fit.GEOMETRY_FIT_PARAM_ORDER,
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


def _background_theta_for_index(_index: int, *, strict_count: bool = False) -> float:
    _ = strict_count
    return float(theta_initial_var.get())


def _build_mosaic_params() -> dict[str, object]:
    return {}


def _current_optics_mode_flag():
    return None


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
    raise RuntimeError("GUI runtime is not loaded; call ra_sim.gui.runtime.main().")


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
_geometry_fit_runtime_value_callbacks = None

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
debye_x_var = _PlaceholderVar(0.0)
debye_y_var = _PlaceholderVar(0.0)
lambda_ = 0.0
psi = 0.0
n2 = 0.0

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

    return gui_geometry_fit.copy_geometry_fit_state_value(value)


def _geometry_fit_runtime_values() -> gui_geometry_fit.GeometryFitRuntimeValueCallbacks:
    """Return the bound live geometry-fit value readers for the app shim."""

    global _geometry_fit_runtime_value_callbacks

    callbacks = _geometry_fit_runtime_value_callbacks
    if callbacks is None:
        callbacks = gui_geometry_fit.build_runtime_geometry_fit_value_callbacks(
            gui_geometry_fit.GeometryFitRuntimeValueBindings(
                fit_zb_var=fit_zb_var,
                fit_zs_var=fit_zs_var,
                fit_theta_var=fit_theta_var,
                fit_psi_z_var=fit_psi_z_var,
                fit_chi_var=fit_chi_var,
                fit_cor_var=fit_cor_var,
                fit_gamma_var=fit_gamma_var,
                fit_Gamma_var=fit_Gamma_var,
                fit_dist_var=fit_dist_var,
                fit_a_var=fit_a_var,
                fit_c_var=fit_c_var,
                fit_center_x_var=fit_center_x_var,
                fit_center_y_var=fit_center_y_var,
                zb_var=zb_var,
                zs_var=zs_var,
                theta_initial_var=theta_initial_var,
                psi_z_var=psi_z_var,
                chi_var=chi_var,
                cor_angle_var=cor_angle_var,
                gamma_var=gamma_var,
                Gamma_var=Gamma_var,
                corto_detector_var=corto_detector_var,
                a_var=a_var,
                c_var=c_var,
                center_x_var=center_x_var,
                center_y_var=center_y_var,
                debye_x_var=debye_x_var,
                debye_y_var=debye_y_var,
                geometry_theta_offset_var=geometry_theta_offset_var,
                current_background_index=lambda: current_background_index,
                geometry_fit_uses_shared_theta_offset=_geometry_fit_uses_shared_theta_offset,
                current_geometry_theta_offset=_current_geometry_theta_offset,
                background_theta_for_index=_background_theta_for_index,
                build_mosaic_params=_build_mosaic_params,
                current_optics_mode_flag=_current_optics_mode_flag,
                lambda_value=lambda_,
                psi=psi,
                n2=n2,
            )
        )
        _geometry_fit_runtime_value_callbacks = callbacks
    return callbacks


def _current_geometry_fit_ui_params() -> dict[str, object]:
    """Capture the current geometry-fit UI parameter values."""

    return _geometry_fit_runtime_values().current_ui_params()


def _current_geometry_fit_var_names() -> list[str]:
    """Return the currently selected geometry variables for LSQ fitting."""

    return _geometry_fit_runtime_values().current_var_names()


def _geometry_fit_restore_var_map() -> dict[str, object]:
    """Return the current geometry-fit runtime Tk variable mapping."""

    return dict(_geometry_fit_runtime_values().var_map)


def _set_geometry_fit_profile_cache(cached_profile_state: Mapping[str, object]) -> None:
    """Replace the cached geometry-fit profile payload."""

    global profile_cache
    profile_cache = cached_profile_state


def _set_geometry_fit_last_overlay_state(
    overlay_state: Mapping[str, object] | None,
) -> None:
    """Replace the cached geometry-fit overlay payload."""

    global last_geometry_overlay_state
    last_geometry_overlay_state = overlay_state


def _mark_geometry_fit_last_simulation_dirty() -> None:
    """Clear the cached simulation signature after geometry-fit state restore."""

    global last_simulation_signature
    last_simulation_signature = None


def _cancel_geometry_fit_pending_update() -> None:
    """Cancel one queued geometry-fit UI update if it exists."""

    global update_pending
    if update_pending is not None:
        gui_controllers.clear_tk_after_token(root, update_pending)
        update_pending = None


def _draw_geometry_fit_overlay_records(
    overlay_records: list[dict[str, object]],
    marker_limit: int,
) -> None:
    """Redraw the fitted overlay records after restoring geometry-fit state."""

    _draw_geometry_fit_overlay(
        overlay_records,
        max_display_markers=marker_limit,
    )


def _draw_initial_geometry_fit_pairs_overlay_records(
    initial_pairs_display: list[dict[str, object]],
    marker_limit: int,
) -> None:
    """Redraw the initial-pair overlay after restoring geometry-fit state."""

    _draw_initial_geometry_pairs_overlay(
        initial_pairs_display,
        max_display_markers=marker_limit,
    )


_restore_geometry_fit_undo_state = (
    gui_geometry_fit.build_runtime_geometry_fit_undo_restore_callback(
        var_map_factory=_geometry_fit_restore_var_map,
        geometry_theta_offset_var_factory=lambda: geometry_theta_offset_var,
        replace_profile_cache=_set_geometry_fit_profile_cache,
        set_last_overlay_state=_set_geometry_fit_last_overlay_state,
        mark_last_simulation_dirty=_mark_geometry_fit_last_simulation_dirty,
        cancel_pending_update=_cancel_geometry_fit_pending_update,
        run_update=lambda: do_update(),
        draw_overlay_records=_draw_geometry_fit_overlay_records,
        draw_initial_pairs_overlay=_draw_initial_geometry_fit_pairs_overlay_records,
        refresh_status=lambda: _set_background_file_status_text(),
        update_manual_pick_button_label=lambda: _update_geometry_manual_pick_button_label(),
    )
)


def _build_geometry_fit_runtime_config(
    base_config,
    current_params,
    control_settings,
    parameter_domains,
):
    return gui_geometry_fit.build_geometry_fit_runtime_config(
        base_config,
        current_params,
        control_settings,
        parameter_domains,
    )


def _canonicalize_gui_state_background_path(path: object) -> str:
    return gui_state_io.canonicalize_gui_state_background_path(path)


def _background_files_match_loaded_state(file_paths: list[str]) -> bool:
    return gui_state_io.background_files_match_loaded_state(
        file_paths,
        osc_files=osc_files,
        background_images_native=background_images_native,
        background_images_display=background_images_display,
    )


def _load_background_files_for_state(
    file_paths: list[str],
    *,
    select_index: int = 0,
) -> None:
    global osc_files, background_images, background_images_native, background_images_display
    global current_background_index, current_background_image, current_background_display

    updated_state = gui_state_io.load_background_files_for_state(
        file_paths,
        osc_files=osc_files,
        background_images=background_images,
        background_images_native=background_images_native,
        background_images_display=background_images_display,
        select_index=select_index,
        display_rotate_k=DISPLAY_ROTATE_K,
        read_osc=read_osc,
        set_background_display=background_display.set_data,
    )
    if updated_state is None:
        return
    osc_files = list(updated_state["osc_files"])
    background_images = list(updated_state["background_images"])
    background_images_native = list(updated_state["background_images_native"])
    background_images_display = list(updated_state["background_images_display"])
    current_background_index = int(updated_state["current_background_index"])
    current_background_image = updated_state["current_background_image"]
    current_background_display = updated_state["current_background_display"]


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

    module = gui_lazy_runtime.load_cached_imported_module(
        _RUNTIME_MODULE,
        module_name=_RUNTIME_MODULE_NAME,
    )
    _RUNTIME_MODULE = module
    return module


def main(
    write_excel_flag: bool | None = None,
    startup_mode: str = "prompt",
    calibrant_bundle: str | None = None,
) -> None:
    """Compatibility wrapper that forwards to ``ra_sim.gui.runtime.main()``."""

    global write_excel
    runtime = _load_runtime_module()
    next_write_excel = bool(write_excel)
    if write_excel_flag is not None:
        next_write_excel = bool(write_excel_flag)
    runtime.write_excel = next_write_excel
    runtime.main(
        write_excel_flag=write_excel_flag,
        startup_mode=startup_mode,
        calibrant_bundle=calibrant_bundle,
    )
    write_excel = bool(getattr(runtime, "write_excel", next_write_excel))


def __getattr__(name: str):
    return gui_lazy_runtime.lazy_module_getattr(
        name=name,
        module_name=__name__,
        current_write_excel=write_excel,
        load_runtime_module=_load_runtime_module,
    )


def __dir__() -> list[str]:
    return gui_lazy_runtime.lazy_module_dir(
        module_globals=globals(),
        loaded_module=_RUNTIME_MODULE,
    )


if __name__ == "__main__":
    from ra_sim.launcher import main as launcher_main

    launcher_main(["gui", *sys.argv[1:]])
