#!/usr/bin/env python3

"""Main application entry point for running the Tk based GUI."""

import os
import sys

from ra_sim.gui import bootstrap as gui_bootstrap

write_excel = False


gui_bootstrap.early_main_bootstrap(__name__)

import math
import json
import copy
import concurrent.futures
import queue
import faulthandler
import re
import traceback
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Callable, Mapping, Sequence
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm, Normalize
from matplotlib.patches import Rectangle
import CifFile

from ra_sim.io.osc_reader import read_osc
from ra_sim.utils.stacking_fault import (
    DEFAULT_PHASE_DELTA_EXPRESSION,
    DEFAULT_PHI_L_DIVISOR,
    _infer_iodine_z_like_diffuse,
    ht_Iinf_dict,
    ht_dict_to_arrays,
    ht_dict_to_qr_dict,
    normalize_phi_l_divisor,
    normalize_phase_delta_expression,
    validate_phase_delta_expression,
)

from ra_sim.utils.calculations import (
    IndexofRefraction,
    resolve_index_of_refraction,
    resolve_index_of_refraction_array,
)
from ra_sim.utils.parallel import (
    default_reserved_cpu_worker_count,
    temporary_numba_thread_limit,
)
from ra_sim.io.file_parsing import parse_poni_file, Open_ASC
from ra_sim.utils.tools import (
    miller_generator,
    view_azimuthal_radial,
    detect_blobs,
    inject_fractional_reflections,
    build_intensity_dataframes,
    detector_two_theta_max,
    DEFAULT_PIXEL_SIZE_M,
)
from ra_sim.io.data_loading import (
    load_geometry_placements_file,
    load_and_format_reference_profiles,
    save_all_parameters,
    save_geometry_placements_file,
    load_parameters,
    load_gui_state_file,
    save_gui_state_file,
)
from ra_sim.fitting.optimization import (
    build_measured_dict,
    fit_geometry_parameters,
    fit_mosaic_shape_parameters,
    fit_mosaic_widths_separable,
    focus_mosaic_profile_dataset_specs,
    simulate_and_compare_hkl,
)
from ra_sim.fitting.background_peak_matching import (
    build_background_peak_context,
    match_simulated_peaks_to_peak_context,
)
from ra_sim.simulation.mosaic_profiles import (
    RANDOM_GAUSSIAN_SAMPLING,
    generate_random_profiles,
)
from ra_sim.simulation.diffraction import (
    DEFAULT_SOLVE_Q_REL_TOL,
    DEFAULT_SOLVE_Q_MODE,
    DEFAULT_SOLVE_Q_STEPS,
    MAX_SOLVE_Q_REL_TOL,
    MAX_SOLVE_Q_STEPS,
    MIN_SOLVE_Q_REL_TOL,
    MIN_SOLVE_Q_STEPS,
    SOLVE_Q_MODE_ADAPTIVE,
    SOLVE_Q_MODE_UNIFORM,
    get_last_intersection_cache,
    hit_tables_to_max_positions,
    intersection_cache_to_hit_tables,
    load_most_recent_logged_intersection_cache,
    OPTICS_MODE_EXACT,
    OPTICS_MODE_FAST,
    process_peaks_parallel,
    process_qr_rods_parallel,
)
from ra_sim.simulation.diffraction_debug import (
    process_peaks_parallel_debug,
    process_qr_rods_parallel_debug,
    dump_debug_log,
)
from ra_sim.simulation.engine import (
    simulate as simulate_request,
    simulate_qr_rods as simulate_qr_rods_request,
)
from ra_sim.simulation.exact_cake import start_exact_cake_numba_warmup_in_background
from ra_sim.simulation.exact_cake_portable import (
    FastAzimuthalIntegrator,
    start_exact_cake_geometry_warmup_in_background,
)
from ra_sim.simulation.simulation import simulate_diffraction
from ra_sim.simulation.types import (
    BeamSamples,
    DebyeWallerParams,
    DetectorGeometry,
    MosaicParams,
    SimulationRequest,
)
from ra_sim.gui import background as gui_background
from ra_sim.gui import background_manager as gui_background_manager
from ra_sim.gui import background_theta as gui_background_theta
from ra_sim.gui import analysis_figure_controls as gui_analysis_figure_controls
from ra_sim.gui import analysis_quick_controls as gui_analysis_quick_controls
from ra_sim.gui import analysis_visibility as gui_analysis_visibility
from ra_sim.gui import analysis_peak_tools as gui_analysis_peak_tools
from ra_sim.gui import bragg_qr_manager as gui_bragg_qr_manager
from ra_sim.gui import canvas_interactions as gui_canvas_interactions
from ra_sim.gui import geometry_q_group_manager as gui_geometry_q_group_manager
from ra_sim.gui import geometry_overlay as gui_geometry_overlay
from ra_sim.gui import integration_range_drag as gui_integration_range_drag
from ra_sim.gui import manual_geometry as gui_manual_geometry
from ra_sim.gui import runtime_background as gui_runtime_background
from ra_sim.gui import runtime_display_acceleration as gui_runtime_display_acceleration
from ra_sim.gui import runtime_fit_analysis as gui_runtime_fit_analysis
from ra_sim.gui import runtime_geometry_fit as gui_runtime_geometry_fit
from ra_sim.gui import runtime_geometry_interaction as gui_runtime_geometry_interaction
from ra_sim.gui import runtime_geometry_preview as gui_runtime_geometry_preview
from ra_sim.gui import runtime_primary_cache as gui_runtime_primary_cache
from ra_sim.gui import runtime_primary_viewport as gui_runtime_primary_viewport
from ra_sim.gui import runtime_position_preview as gui_runtime_position_preview
from ra_sim.gui import runtime_qr_cylinder_overlay as gui_runtime_qr_cylinder_overlay
from ra_sim.gui import runtime_startup as gui_runtime_startup
from ra_sim.gui import runtime_update_trace as gui_runtime_update_trace
from ra_sim.gui._runtime import primary_cache_helpers as _primary_cache_helpers
from ra_sim.gui._runtime.live_cache_helpers import (
    empty_peak_overlay_cache as _empty_peak_overlay_cache,
    empty_qr_cylinder_overlay_cache as _empty_qr_cylinder_overlay_cache,
    live_cache_count as _live_cache_count,
    live_cache_inventory_snapshot as _build_live_cache_inventory_snapshot,
    live_cache_shape as _live_cache_shape,
    live_cache_short_text as _live_cache_short_text,
    live_cache_signature_summary as _live_cache_signature_summary,
)
from ra_sim.gui import tk_primary_viewport as gui_tk_primary_viewport
from ra_sim.gui import fit2d_error_sound as gui_fit2d_error_sound
from ra_sim.gui import views as gui_views
from ra_sim.gui import ordered_structure_fit as gui_ordered_structure_fit
from ra_sim.gui import structure_model as gui_structure_model
from ra_sim.gui.geometry_overlay import (
    build_geometry_fit_overlay_records,
    compute_geometry_overlay_frame_diagnostics,
)
from ra_sim.gui import overlays as gui_overlays
from ra_sim.gui import peak_selection as gui_peak_selection
from ra_sim.gui import qr_cylinder_overlay as gui_qr_cylinder_overlay
from ra_sim.gui import geometry_fit as gui_geometry_fit
from ra_sim.gui import controllers as gui_controllers
from ra_sim.gui import display_projection as gui_display_projection
from ra_sim.gui import fast_plot_viewer as gui_fast_plot_viewer
from ra_sim.gui import main_matplotlib_interaction as gui_main_matplotlib_interaction
from ra_sim.gui import main_figure_chrome as gui_main_figure_chrome
from ra_sim.gui import state as gui_state
from ra_sim.gui import state_io as gui_state_io
from ra_sim.gui import structure_factor_pruning as gui_structure_factor_pruning
from ra_sim.gui.sliders import create_slider
from ra_sim.gui.diffuse_cif_toggle import (
    open_diffuse_cif_toggle_algebraic,
    export_algebraic_ht_txt,
)
from ra_sim.debug_controls import (
    mosaic_fit_log_files_enabled,
    retain_optional_cache,
    runtime_update_trace_logging_enabled as _runtime_update_trace_logging_enabled,
)
from ra_sim.debug_utils import debug_print, is_debug_enabled
from ra_sim.hbn_geometry import (
    build_hbn_geometry_debug_trace,
    convert_hbn_bundle_geometry_to_simulation,
    format_hbn_geometry_debug_trace,
    load_bundle_npz,
    load_tilt_hint,
)
from ra_sim.gui.collapsible import CollapsibleFrame


turbo = matplotlib.colormaps.get_cmap("turbo").resampled(256)
turbo_rgba = turbo(np.linspace(0, 1, 256))
turbo_rgba[0] = [1.0, 1.0, 1.0, 1.0]  # make the 0-bin white
turbo_white0 = ListedColormap(turbo_rgba, name="turbo_white0")
turbo_white0.set_bad("white")  # NaNs will also show white


# Force TkAgg backend to ensure GUI usage
matplotlib.use("TkAgg")
# Enable extra diagnostics when the RA_SIM_DEBUG environment variable is set.
DEBUG_ENABLED = is_debug_enabled()
if DEBUG_ENABLED:
    print("Debug mode active (RA_SIM_DEBUG=1)")
    from ra_sim.debug_utils import enable_numba_logging

    enable_numba_logging()

# Toggle creation of backend orientation controls (kept off while automated
# diagnostics try permutations internally).
BACKEND_ORIENTATION_UI_ENABLED = False
BACKGROUND_BACKEND_DEBUG_UI_ENABLED = False
HBN_GEOMETRY_DEBUG_ENABLED = False

_RUNTIME_SOURCE_COMPAT_SENTINELS = """
PREVIEW_CALCULATIONS_ENABLED
                and bool(live_geometry_preview_var.get())
if (
                one_d_analysis_requested
"""

_AZIMUTHAL_INTEGRATOR_CLS = None


def _get_azimuthal_integrator_cls():
    global _AZIMUTHAL_INTEGRATOR_CLS

    if _AZIMUTHAL_INTEGRATOR_CLS is None:
        _AZIMUTHAL_INTEGRATOR_CLS = FastAzimuthalIntegrator
    return _AZIMUTHAL_INTEGRATOR_CLS


###############################################################################
#                          DATA & PARAMETER SETUP
###############################################################################
from ra_sim.config import (
    get_path,
    get_path_first,
    get_dir,
)

HKL_PICK_MIN_SEPARATION_PX = 2.0
# Search a 100x100 box centered on the click (±50 px along each axis).
HKL_PICK_MAX_DISTANCE_PX = 50.0

_RUNTIME_UPDATE_TRACE_HANDLE = None
_RUNTIME_UPDATE_TRACE_HOOKS_INSTALLED = False


def _all_logging_disabled() -> bool:
    return not _runtime_update_trace_logging_enabled()


def _runtime_update_trace_path() -> Path:
    """Return the daily GUI runtime trace path."""

    try:
        downloads_dir = get_dir("downloads")
    except Exception:
        downloads_dir = Path.home() / "Downloads"
    return gui_runtime_update_trace.resolve_runtime_update_trace_path(downloads_dir)


def _append_runtime_update_trace(event: str, **fields: object) -> None:
    """Append one GUI runtime trace line, ignoring logging failures."""

    if _all_logging_disabled():
        return
    try:
        gui_runtime_update_trace.append_runtime_update_trace_line(
            _runtime_update_trace_path(),
            event,
            **fields,
        )
    except Exception:
        pass


def _retain_runtime_optional_cache(
    family: str,
    *,
    feature_needed: bool = True,
) -> bool:
    try:
        return retain_optional_cache(family, feature_needed=feature_needed)
    except Exception:
        return bool(feature_needed)


def _live_cache_inventory_snapshot() -> dict[str, object]:
    return _build_live_cache_inventory_snapshot(simulation_runtime_state)


def _trace_live_cache_event(
    family: str,
    action: str,
    *,
    background_index: object = None,
    current_background_index: object = None,
    **fields: object,
) -> None:
    event_name = f"live_cache_{str(family)}_{str(action)}"
    try:
        resolved_current_background_index = (
            getattr(background_runtime_state, "current_background_index", None)
            if current_background_index is None
            else current_background_index
        )
    except Exception:
        resolved_current_background_index = current_background_index
    resolved_background_index = (
        resolved_current_background_index if background_index is None else background_index
    )
    payload: dict[str, object] = {
        "cache_family": str(family),
        "action": str(action),
        "update_id": getattr(simulation_runtime_state, "current_update_trace_id", None),
        "stage": getattr(simulation_runtime_state, "current_update_trace_stage", None),
        "update_phase": getattr(simulation_runtime_state, "update_phase", None),
        "is_preview": bool(getattr(simulation_runtime_state, "preview_active", False)),
        "preview_sample_count": getattr(
            simulation_runtime_state,
            "preview_sample_count",
            None,
        ),
    }
    try:
        if resolved_background_index is not None:
            payload["background_index"] = int(resolved_background_index)
    except Exception:
        payload["background_index"] = resolved_background_index
    try:
        if resolved_current_background_index is not None:
            payload["current_background_index"] = int(resolved_current_background_index)
    except Exception:
        payload["current_background_index"] = resolved_current_background_index
    payload.update(fields)
    _append_runtime_update_trace(event_name, **payload)


def _append_runtime_update_exception_trace(
    event: str,
    exc_type: object,
    exc_value: object,
    exc_tb,
    **fields: object,
) -> None:
    """Append one exception event plus traceback text to the runtime trace."""

    if _all_logging_disabled():
        return
    header_fields = dict(fields)
    header_fields.update(
        {
            "exc_type": getattr(exc_type, "__name__", str(exc_type)),
            "error": str(exc_value),
        }
    )
    _append_runtime_update_trace(event, **header_fields)
    try:
        trace_path = _runtime_update_trace_path()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as handle:
            for line in traceback.format_exception(exc_type, exc_value, exc_tb):
                handle.write(line)
            handle.flush()
    except Exception:
        pass


def _ensure_runtime_update_trace_hooks() -> None:
    """Install persistent crash/exception hooks for GUI runtime tracing."""

    global _RUNTIME_UPDATE_TRACE_HANDLE, _RUNTIME_UPDATE_TRACE_HOOKS_INSTALLED
    if _all_logging_disabled():
        return
    if _RUNTIME_UPDATE_TRACE_HOOKS_INSTALLED:
        return

    try:
        trace_path = _runtime_update_trace_path()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        _RUNTIME_UPDATE_TRACE_HANDLE = trace_path.open(
            "a",
            encoding="utf-8",
            buffering=1,
        )
        try:
            faulthandler.enable(_RUNTIME_UPDATE_TRACE_HANDLE, all_threads=True)
        except Exception:
            pass

        previous_excepthook = sys.excepthook

        def _runtime_trace_excepthook(exc_type, exc_value, exc_tb):
            _append_runtime_update_exception_trace(
                "uncaught_exception",
                exc_type,
                exc_value,
                exc_tb,
                update_id=getattr(simulation_runtime_state, "current_update_trace_id", None),
                trace_stage=getattr(
                    simulation_runtime_state,
                    "current_update_trace_stage",
                    None,
                ),
                update_phase=getattr(simulation_runtime_state, "update_phase", None),
                update_running=getattr(simulation_runtime_state, "update_running", None),
            )
            if callable(previous_excepthook):
                previous_excepthook(exc_type, exc_value, exc_tb)

        sys.excepthook = _runtime_trace_excepthook
        _RUNTIME_UPDATE_TRACE_HOOKS_INSTALLED = True
        _append_runtime_update_trace(
            "runtime_update_trace_hooks_installed",
            trace_path=str(trace_path),
            faulthandler_enabled=bool(faulthandler.is_enabled()),
        )
    except Exception:
        pass


background_status_refreshers = gui_runtime_background.build_runtime_background_status_refreshers(
    background_controls_runtime_factory=lambda: globals().get("background_controls_runtime"),
    background_runtime_callbacks_factory=lambda: globals().get("background_runtime_callbacks"),
)
_refresh_background_status = background_status_refreshers.refresh_status
_refresh_background_backend_status = background_status_refreshers.refresh_backend_status


def _ensure_triplet(values, fallback):
    """Return a 3-element list combining *values* with *fallback*."""

    if not isinstance(values, (list, tuple)):
        return list(fallback)
    merged = list(fallback)
    for idx, val in enumerate(values[:3]):
        merged[idx] = val
    return merged


def _ensure_numeric_vector(values, fallback, length):
    """Return a float vector of ``length`` using ``values`` with fallback fill."""

    try:
        target_len = int(length)
    except (TypeError, ValueError):
        target_len = 1
    target_len = max(1, target_len)

    out = []
    if isinstance(values, (list, tuple, np.ndarray)):
        for raw in values:
            try:
                out.append(float(raw))
            except (TypeError, ValueError):
                continue

    if not out:
        if isinstance(fallback, (list, tuple, np.ndarray)):
            for raw in fallback:
                try:
                    out.append(float(raw))
                except (TypeError, ValueError):
                    continue
        else:
            try:
                out = [float(fallback)]
            except (TypeError, ValueError):
                out = [1.0]

    if not out:
        out = [1.0]

    if len(out) < target_len:
        out.extend([out[-1]] * (target_len - len(out)))
    elif len(out) > target_len:
        out = out[:target_len]
    return out


app_state = gui_controllers.build_initial_state()
background_runtime_state = app_state.background_runtime
peak_selection_state = app_state.peak_selection
integration_range_drag_state = app_state.integration_range_drag
atom_site_override_state = app_state.atom_site_override
geometry_runtime_state = app_state.geometry_runtime
simulation_runtime_state = app_state.simulation_runtime
bragg_qr_manager_state = app_state.bragg_qr_manager

instrument_config = app_state.instrument_config.get("instrument", {})
detector_config = instrument_config.get("detector", {})
geometry_config = instrument_config.get("geometry_defaults", {})
beam_config = instrument_config.get("beam", {})
sample_config = instrument_config.get("sample_orientation", {})
debye_config = instrument_config.get("debye_waller", {})
occupancy_config = instrument_config.get("occupancies", {})
hendricks_config = instrument_config.get("hendricks_teller", {})
output_config = instrument_config.get("output", {})
fit_config = instrument_config.get("fit", {})

background_runtime_state.osc_files = get_path_first("simulation_background_osc_files", "osc_files")
if isinstance(background_runtime_state.osc_files, str):
    background_runtime_state.osc_files = [background_runtime_state.osc_files]
if not background_runtime_state.osc_files:
    raise ValueError(
        "No oscillation images configured in simulation_background_osc_files/osc_files"
    )
app_state.file_paths["simulation_background_osc_files"] = list(background_runtime_state.osc_files)
background_runtime_state.osc_files = list(background_runtime_state.osc_files)

# Background and simulated overlays can use different display orientations.
# ``k`` is the np.rot90 factor; -1 is 90° clockwise, 0 keeps native orientation.
DISPLAY_ROTATE_K = -1
SIM_DISPLAY_ROTATE_K = 0
# hBN fitter bundles are in native OSC orientation (no rot90).
HBN_FITTER_ROTATE_K = 0
# Simulation geometry runs in native simulation pixels; convert hBN bundle
# geometry so that the displayed simulation (after SIM_DISPLAY_ROTATE_K) lands
# in the same frame as the displayed background (after DISPLAY_ROTATE_K).
SIMULATION_GEOMETRY_ROTATE_K = DISPLAY_ROTATE_K - SIM_DISPLAY_ROTATE_K

# Preserve native-orientation copies for fitting/analysis. Load only the first
# background at startup and lazy-load the rest on demand to improve first paint.
_initial_background_state = gui_background_manager.initialize_background_runtime_state(
    background_runtime_state,
    str(background_runtime_state.osc_files[0]),
    total_count=len(background_runtime_state.osc_files),
    display_rotate_k=DISPLAY_ROTATE_K,
    read_osc=read_osc,
    file_paths_state=app_state.file_paths,
    visible=True,
    backend_rotation_k=3,
    backend_flip_x=False,
    backend_flip_y=False,
)

# Parse geometry
poni_file_path = get_path("geometry_poni")
app_state.file_paths["geometry_poni"] = str(poni_file_path)
parameters = parse_poni_file(poni_file_path)

Distance_CoR_to_Detector = parameters.get("Dist", geometry_config.get("distance_m", 0.075))
Gamma_initial = parameters.get("Rot1", geometry_config.get("rot1", 0.0))
gamma_initial = parameters.get("Rot2", geometry_config.get("rot2", 0.0))
poni1 = parameters.get("Poni1", geometry_config.get("poni1_m", 0.0))
poni2 = parameters.get("Poni2", geometry_config.get("poni2_m", 0.0))
wave_m = parameters.get("Wavelength", geometry_config.get("wavelength_m", 1e-10))
lambda_from_poni = wave_m * 1e10  # Convert m -> Å

tilt_hint = load_tilt_hint()
hinted_center_row = None
hinted_center_col = None
if tilt_hint:
    hinted_gamma = tilt_hint.get("gamma_deg")
    hinted_Gamma = tilt_hint.get("Gamma_deg")
    hinted_center_row = tilt_hint.get("center_row")
    hinted_center_col = tilt_hint.get("center_col")
    try:
        hinted_gamma = float(hinted_gamma)
        hinted_Gamma = float(hinted_Gamma)
    except (TypeError, ValueError):
        hinted_gamma = None
        hinted_Gamma = None
    try:
        hinted_center_row = float(hinted_center_row)
        hinted_center_col = float(hinted_center_col)
    except (TypeError, ValueError):
        hinted_center_row = None
        hinted_center_col = None
    if hinted_center_row is not None and hinted_center_col is not None:
        if not (np.isfinite(hinted_center_row) and np.isfinite(hinted_center_col)):
            hinted_center_row = None
            hinted_center_col = None
    if hinted_gamma is not None and np.isfinite(hinted_gamma):
        # Historical naming here is swapped: defaults['gamma'] reads Gamma_initial.
        Gamma_initial = hinted_gamma
    if hinted_Gamma is not None and np.isfinite(hinted_Gamma):
        # Historical naming here is swapped: defaults['Gamma'] reads gamma_initial.
        gamma_initial = hinted_Gamma
    hinted_distance = tilt_hint.get("distance_m")
    try:
        hinted_distance = float(hinted_distance)
    except (TypeError, ValueError):
        hinted_distance = None
    if hinted_distance is not None and np.isfinite(hinted_distance):
        Distance_CoR_to_Detector = hinted_distance
    center_text = ""
    if hinted_center_row is not None and hinted_center_col is not None:
        center_text = f", center(row={hinted_center_row:.3f}, col={hinted_center_col:.3f})"
    print(
        "Initialized detector tilt from last hBN fit profile "
        f"(sim γ={Gamma_initial:.4f} deg, sim Γ={gamma_initial:.4f} deg{center_text})."
    )
    if hinted_distance is not None and np.isfinite(hinted_distance):
        print(
            "Initialized detector distance from last hBN fit profile "
            f"(Dist={hinted_distance:.4f} m)."
        )

image_size = int(app_state.image_size)
simulation_runtime_state.main_display_raster_limit = (
    gui_display_projection.default_display_raster_size(image_size)
)
pixel_size_m = float(detector_config.get("pixel_size_m", DEFAULT_PIXEL_SIZE_M))
legacy_resolution_sample_counts = {
    "Low": 25,
    "Medium": 250,
    "High": 500,
}
DEFAULT_RANDOM_SAMPLE_COUNT = 50
MIN_RANDOM_SAMPLE_COUNT = 1
MAX_RANDOM_SAMPLE_COUNT = 5000
MOSAIC_SHAPE_FIT_MIN_SAMPLE_COUNT = 50000
MOSAIC_SHAPE_FIT_MAX_IN_PLANE_GROUPS = 3
CUSTOM_SAMPLING_OPTION = "Custom"
simulation_runtime_state.num_samples = DEFAULT_RANDOM_SAMPLE_COUNT
write_excel = output_config.get("write_excel", write_excel)
intensity_threshold = detector_config.get("intensity_threshold", 1.0)
vmax_default = detector_config.get("vmax", 1000)
vmax_slider_max = detector_config.get("vmax_slider_max", 3000)

# Approximate beam center
center_default = [(poni2 / pixel_size_m), image_size - (poni1 / pixel_size_m)]
if hinted_center_row is not None and hinted_center_col is not None:
    center_default = [hinted_center_row, hinted_center_col]

two_theta_max = detector_two_theta_max(
    image_size,
    center_default,
    Distance_CoR_to_Detector,
    pixel_size=pixel_size_m,
)
two_theta_range = (0.0, two_theta_max)

mx = hendricks_config.get("max_miller_index", 19)

fwhm2sigma = 1 / (2 * math.sqrt(2 * math.log(2)))
divergence_fwhm = beam_config.get("divergence_fwhm_deg", 0.05)
divergence_sigma = math.radians(divergence_fwhm * fwhm2sigma)

sigma_mosaic = math.radians(beam_config.get("sigma_mosaic_fwhm_deg", 0.8) * fwhm2sigma)
gamma_mosaic = math.radians(beam_config.get("gamma_mosaic_fwhm_deg", 0.7) * fwhm2sigma)
eta = beam_config.get("eta", 0.0)

theta_initial = sample_config.get("theta_initial_deg", 6.0)
cor_angle = sample_config.get("cor_deg", 0.0)
chi = sample_config.get("chi_deg", 0.0)
psi = sample_config.get("psi_deg", 0.0)
psi_z = sample_config.get("psi_z_deg", 0.0)
zb = sample_config.get("zb", 0.0)
bw_sigma = beam_config.get("bandwidth_sigma_fraction", 0.05e-3) * fwhm2sigma
zs = sample_config.get("zs", 0.0)
sample_width_m = float(sample_config.get("width_m", 0.0))
sample_length_m = float(sample_config.get("length_m", 0.0))
sample_depth_m = float(sample_config.get("depth_m", 0.0))
debye_x = debye_config.get("x", 0.0)
debye_y = debye_config.get("y", 0.0)

# Print the computed complex index of refraction on startup and exit
# print("Computed complex index of refraction n2:", n2)
# sys.exit(0)

bandwidth_percent_default = float(beam_config.get("bandwidth_percent", 0.7))
bandwidth = bandwidth_percent_default / 100.0

# NOTE: Occupancy defaults are sized to the GUI's unique atom-site controls
# after loading the CIF below.
occupancy_default_values = occupancy_config.get("default", [1.0, 1.0, 1.0])

# When enabled, additional fractional reflections ("rods")
# are injected between integer L values.
include_rods_flag = hendricks_config.get("include_rods", False)
instrument_pruning_control_defaults = (
    gui_runtime_fit_analysis.resolve_runtime_pruning_control_defaults(
        structure_factor_pruning_module=gui_structure_factor_pruning,
        raw_prune_bias=hendricks_config.get("sf_prune_bias", 0.0),
        raw_solve_q_steps=beam_config.get("solve_q_steps", DEFAULT_SOLVE_Q_STEPS),
        raw_solve_q_rel_tol=beam_config.get(
            "solve_q_rel_tol",
            DEFAULT_SOLVE_Q_REL_TOL,
        ),
        raw_solve_q_mode=beam_config.get("solve_q_mode", "uniform"),
        prune_bias_fallback=0.0,
        prune_bias_minimum=gui_controllers.SF_PRUNE_BIAS_MIN,
        prune_bias_maximum=gui_controllers.SF_PRUNE_BIAS_MAX,
        steps_fallback=DEFAULT_SOLVE_Q_STEPS,
        steps_minimum=MIN_SOLVE_Q_STEPS,
        steps_maximum=MAX_SOLVE_Q_STEPS,
        rel_tol_fallback=DEFAULT_SOLVE_Q_REL_TOL,
        rel_tol_minimum=MIN_SOLVE_Q_REL_TOL,
        rel_tol_maximum=MAX_SOLVE_Q_REL_TOL,
        uniform_flag=SOLVE_Q_MODE_UNIFORM,
        adaptive_flag=SOLVE_Q_MODE_ADAPTIVE,
    )
)
sf_prune_bias_default = float(instrument_pruning_control_defaults.prune_bias)
solve_q_steps_default = int(instrument_pruning_control_defaults.solve_q.steps)
solve_q_rel_tol_default = float(instrument_pruning_control_defaults.solve_q.rel_tol)
solve_q_mode_default = int(instrument_pruning_control_defaults.solve_q.mode_flag)

lambda_override = beam_config.get("wavelength_angstrom")
lambda_ = lambda_override if lambda_override is not None else lambda_from_poni
n2 = IndexofRefraction(float(lambda_) * 1.0e-10)

# Parameters and file paths.
cif_file = get_path("cif_file")
try:
    cif_file2 = get_path("cif_file2")
except KeyError:
    cif_file2 = None

# read with PyCifRW
cf = CifFile.ReadCif(cif_file)
blk = cf[list(cf.keys())[0]]


def _normalize_occupancy_label(raw_label, fallback_idx):
    return gui_structure_model.normalize_occupancy_label(raw_label, fallback_idx)


def _extract_occupancy_site_metadata(cif_block, cif_path):
    """Return (unique labels, expanded-site -> unique-label index mapping)."""

    return gui_structure_model.extract_occupancy_site_metadata(cif_block, cif_path)


occupancy_site_labels, occupancy_site_expanded_map = _extract_occupancy_site_metadata(blk, cif_file)
if occupancy_site_labels:
    occupancy_site_count = len(occupancy_site_labels)
else:
    occupancy_site_count = max(
        1,
        len(occupancy_default_values)
        if isinstance(occupancy_default_values, (list, tuple, np.ndarray))
        else 1,
    )
occ = _ensure_numeric_vector(
    occupancy_default_values,
    [1.0],
    occupancy_site_count,
)
occ = [min(1.0, max(0.0, float(v))) for v in occ]


def _expand_occupancy_values_for_generated_sites(occ_values):
    """Map unique-site occupancies to generated per-site occupancies for HT."""

    return gui_structure_model.expand_occupancy_values_for_generated_sites(
        occ_values,
        occupancy_site_labels=occupancy_site_labels,
        occupancy_site_expanded_map=occupancy_site_expanded_map,
    )


def _extract_atom_site_fractional_metadata(cif_block):
    """Return atom-site fractional coordinates from raw CIF loop rows."""

    return gui_structure_model.extract_atom_site_fractional_metadata(cif_block)


atom_site_fractional_metadata = _extract_atom_site_fractional_metadata(blk)
atom_site_fract_vars = []


def _atom_site_fractional_default_values():
    return gui_structure_model.atom_site_fractional_default_values(structure_model_state)


def _current_atom_site_fractional_values():
    """Return current atom-site fractional coordinates from GUI state."""

    return gui_structure_model.current_atom_site_fractional_values(
        structure_model_state,
        tcl_error_types=(tk.TclError,),
    )


def _atom_site_fractional_signature(values):
    return gui_structure_model.atom_site_fractional_signature(values)


def _atom_site_fractional_defaults_signature():
    return _atom_site_fractional_signature(_atom_site_fractional_default_values())


def _atom_site_fractional_values_are_default(values):
    return gui_structure_model.atom_site_fractional_values_are_default(
        structure_model_state,
        values,
    )


def _active_primary_cif_path(atom_site_values=None):
    """Return primary CIF path with optional atom-site coordinate overrides."""

    return gui_structure_model.active_primary_cif_path(
        structure_model_state,
        atom_site_override_state,
        atom_site_values=atom_site_values,
        tcl_error_types=(tk.TclError,),
    )


def _resolve_optics_cif_path(atom_site_values=None):
    """Return the CIF path used for optical constants."""

    try:
        active_path = _active_primary_cif_path(atom_site_values=atom_site_values)
    except Exception:
        active_path = None
    if active_path:
        return str(active_path)
    return str(cif_file)


def _current_nominal_n2(active_cif_path=None):
    """Return the nominal complex index for the active beam wavelength."""

    optics_path = _resolve_optics_cif_path() if active_cif_path is None else str(active_cif_path)
    return resolve_index_of_refraction(
        float(lambda_) * 1.0e-10,
        cif_path=optics_path,
    )


def _current_sample_n2_array(wavelength_angstrom_array, active_cif_path=None):
    """Return wavelength-specific complex indices for the active beam samples."""

    optics_path = _resolve_optics_cif_path() if active_cif_path is None else str(active_cif_path)
    wavelength_m = np.asarray(wavelength_angstrom_array, dtype=np.float64) * 1.0e-10
    return resolve_index_of_refraction_array(
        wavelength_m,
        cif_path=optics_path,
    )


def _reset_atom_site_override_cache():
    gui_structure_model.reset_atom_site_override_cache(atom_site_override_state)


n2 = _current_nominal_n2()

# pull the raw text
a_text = blk.get("_cell_length_a")
b_text = blk.get("_cell_length_b")
c_text = blk.get("_cell_length_c")


# strip the '(uncertainty)' and cast
def parse_cif_num(txt):
    return gui_structure_model.parse_cif_num(txt)


if a_text is None or b_text is None or c_text is None:
    raise ValueError("CIF is missing one or more required cell-length fields (_a/_b/_c).")

av = parse_cif_num(a_text)
bv = parse_cif_num(b_text)
cv = parse_cif_num(c_text)

if cif_file2:
    cf2 = CifFile.ReadCif(cif_file2)
    blk2 = cf2[list(cf2.keys())[0]]
    a2_text = blk2.get("_cell_length_a")
    c2_text = blk2.get("_cell_length_c")
    av2 = parse_cif_num(a2_text) if a2_text is not None else av
    cv2 = parse_cif_num(c2_text) if c2_text is not None else cv
else:
    av2 = None
    cv2 = None

energy = 6.62607e-34 * 2.99792458e8 / (lambda_ * 1e-10) / (1.602176634e-19)  # keV

p_defaults = _ensure_triplet(hendricks_config.get("default_p"), [0.01, 0.99, 0.5])
w_defaults = _ensure_triplet(hendricks_config.get("default_w"), [100.0, 0.0, 0.0])
phase_delta_expression_default = normalize_phase_delta_expression(
    hendricks_config.get(
        "phase_delta_expression",
        DEFAULT_PHASE_DELTA_EXPRESSION,
    ),
    fallback=DEFAULT_PHASE_DELTA_EXPRESSION,
)
try:
    phase_delta_expression_default = validate_phase_delta_expression(phase_delta_expression_default)
except ValueError:
    phase_delta_expression_default = DEFAULT_PHASE_DELTA_EXPRESSION
phi_l_divisor_default = normalize_phi_l_divisor(
    hendricks_config.get(
        "phi_l_divisor",
        DEFAULT_PHI_L_DIVISOR,
    ),
    fallback=DEFAULT_PHI_L_DIVISOR,
)
finite_stack_default = bool(hendricks_config.get("finite_stack", True))
stack_layers_default = int(max(1, float(hendricks_config.get("stack_layers", 50))))
try:
    iodine_z_default = _infer_iodine_z_like_diffuse(str(cif_file))
except Exception:
    iodine_z_default = None
if iodine_z_default is None or not np.isfinite(float(iodine_z_default)):
    iodine_z_default = 0.0
iodine_z_default = float(np.clip(float(iodine_z_default), 0.0, 1.0))

# ---------------------------------------------------------------------------
# Default GUI/fit parameter values. These must be defined before any calls
# that reference them (e.g. ``ht_Iinf_dict`` below).
# ---------------------------------------------------------------------------
defaults = {
    "theta_initial": theta_initial,
    "cor_angle": cor_angle,
    "gamma": Gamma_initial,
    "Gamma": gamma_initial,
    "chi": chi,
    "psi_z": psi_z,
    "zs": zs,
    "zb": zb,
    "sample_width_m": sample_width_m,
    "sample_length_m": sample_length_m,
    "sample_depth_m": sample_depth_m,
    "debye_x": debye_x,
    "debye_y": debye_y,
    "corto_detector": Distance_CoR_to_Detector,
    "sigma_mosaic_deg": np.degrees(sigma_mosaic),
    "gamma_mosaic_deg": np.degrees(gamma_mosaic),
    "eta": eta,
    "a": av,
    "c": cv,
    "vmax": vmax_default,
    "p0": p_defaults[0],
    "p1": p_defaults[1],
    "p2": p_defaults[2],
    "w0": w_defaults[0],
    "w1": w_defaults[1],
    "w2": w_defaults[2],
    "iodine_z": iodine_z_default,
    "phase_delta_expression": phase_delta_expression_default,
    "phi_l_divisor": float(phi_l_divisor_default),
    "center_x": center_default[0],
    "center_y": center_default[1],
    "sampling_resolution": CUSTOM_SAMPLING_OPTION,
    "sampling_count": DEFAULT_RANDOM_SAMPLE_COUNT,
    "rod_points_per_gz": gui_controllers.default_rod_points_per_gz(cv),
    "bandwidth_percent": float(np.clip(bandwidth_percent_default, 0.0, 10.0)),
    "sf_prune_bias": sf_prune_bias_default,
    "solve_q_steps": solve_q_steps_default,
    "solve_q_rel_tol": solve_q_rel_tol_default,
    "solve_q_mode": solve_q_mode_default,
    "finite_stack": finite_stack_default,
    "stack_layers": stack_layers_default,
    "optics_mode": "fast",
    "ordered_structure_scale": 1.0,
}

# ---------------------------------------------------------------------------
# Replace the old miller_generator call with the new Hendricks–Teller helper.
# ---------------------------------------------------------------------------
structure_model_state = gui_structure_model.build_lightweight_structure_model_state(
    cif_file=cif_file,
    cf=cf,
    blk=blk,
    cif_file2=cif_file2,
    occupancy_site_labels=occupancy_site_labels,
    occupancy_site_expanded_map=occupancy_site_expanded_map,
    occ=occ,
    atom_site_fractional_metadata=atom_site_fractional_metadata,
    av=av,
    bv=bv,
    cv=cv,
    av2=av2,
    cv2=cv2,
    defaults=defaults,
    mx=mx,
    lambda_angstrom=lambda_,
    intensity_threshold=intensity_threshold,
    two_theta_range=two_theta_range,
    include_rods_flag=include_rods_flag,
    miller_generator=miller_generator,
    inject_fractional_reflections=inject_fractional_reflections,
)


def _sync_structure_model_aliases() -> None:
    global cf, blk, cf2, blk2
    global occupancy_site_labels, occupancy_site_count, occupancy_site_expanded_map
    global occ
    global av, cv, av2, cv2
    global defaults
    global ht_cache_multi, ht_curves_cache
    global miller, intensities, degeneracy, details
    global df_summary, df_details
    global intensities_cif1, intensities_cif2
    global _last_occ_for_ht, _last_p_triplet, _last_weights
    global _last_a_for_ht, _last_c_for_ht, _last_iodine_z_for_ht
    global _last_phi_l_divisor, _last_phase_delta_expression
    global _last_finite_stack, _last_stack_layers, _last_rod_points_per_gz
    global _last_atom_site_fractional_signature

    cf = structure_model_state.cf
    blk = structure_model_state.blk
    cf2 = structure_model_state.cf2
    blk2 = structure_model_state.blk2
    occupancy_site_labels = list(structure_model_state.occupancy_site_labels)
    occupancy_site_count = int(structure_model_state.occupancy_site_count)
    occupancy_site_expanded_map = list(structure_model_state.occupancy_site_expanded_map)
    occ = list(structure_model_state.occ)
    av = float(structure_model_state.av)
    cv = float(structure_model_state.cv)
    av2 = structure_model_state.av2
    cv2 = structure_model_state.cv2
    defaults = structure_model_state.defaults
    ht_cache_multi = structure_model_state.ht_cache_multi
    ht_curves_cache = structure_model_state.ht_curves_cache
    miller = structure_model_state.miller
    intensities = structure_model_state.intensities
    degeneracy = structure_model_state.degeneracy
    details = structure_model_state.details
    df_summary = structure_model_state.df_summary
    df_details = structure_model_state.df_details
    intensities_cif1 = structure_model_state.intensities_cif1
    intensities_cif2 = structure_model_state.intensities_cif2
    _last_occ_for_ht = list(structure_model_state.last_occ_for_ht)
    _last_p_triplet = list(structure_model_state.last_p_triplet)
    _last_weights = list(structure_model_state.last_weights)
    _last_a_for_ht = float(structure_model_state.last_a_for_ht)
    _last_c_for_ht = float(structure_model_state.last_c_for_ht)
    _last_iodine_z_for_ht = float(structure_model_state.last_iodine_z_for_ht)
    _last_phi_l_divisor = float(structure_model_state.last_phi_l_divisor)
    _last_phase_delta_expression = str(structure_model_state.last_phase_delta_expression)
    _last_finite_stack = bool(structure_model_state.last_finite_stack)
    _last_stack_layers = int(structure_model_state.last_stack_layers)
    _last_rod_points_per_gz = int(structure_model_state.last_rod_points_per_gz)
    _last_atom_site_fractional_signature = tuple(
        structure_model_state.last_atom_site_fractional_signature
    )


def _reset_structure_model_runtime_arrays() -> None:
    simulation_runtime_state.sim_miller1_all = structure_model_state.sim_miller1_all.copy()
    simulation_runtime_state.sim_intens1_all = structure_model_state.sim_intens1_all.copy()
    simulation_runtime_state.sim_miller2_all = structure_model_state.sim_miller2_all.copy()
    simulation_runtime_state.sim_intens2_all = structure_model_state.sim_intens2_all.copy()
    simulation_runtime_state.sim_primary_qr_all = dict(structure_model_state.sim_primary_qr_all)
    simulation_runtime_state.sim_miller1 = simulation_runtime_state.sim_miller1_all.copy()
    simulation_runtime_state.sim_intens1 = simulation_runtime_state.sim_intens1_all.copy()
    simulation_runtime_state.sim_miller2 = simulation_runtime_state.sim_miller2_all.copy()
    simulation_runtime_state.sim_intens2 = simulation_runtime_state.sim_intens2_all.copy()
    simulation_runtime_state.sim_primary_qr = {}
    simulation_runtime_state.sf_prune_stats = {
        "qr_total": 0,
        "qr_kept": 0,
        "hkl_primary_total": 0,
        "hkl_primary_kept": 0,
        "hkl_secondary_total": 0,
        "hkl_secondary_kept": 0,
    }


def _apply_structure_model_runtime_cache_state() -> None:
    global has_second_cif, weight1, weight2

    has_second_cif = bool(structure_model_state.has_second_cif)
    weight1 = 0.5 if has_second_cif else 1.0
    weight2 = 0.5 if has_second_cif else 0.0

    _reset_structure_model_runtime_arrays()
    if not bool(getattr(structure_model_state, "bootstrap_complete", False)):
        return

    primary_arrays = structure_model_state.ht_curves_cache.get("arrays")
    if not isinstance(primary_arrays, tuple) or len(primary_arrays) < 2:
        raise RuntimeError("Bootstrapped structure model did not populate HT curve arrays.")

    primary_miller, primary_intens, _, _ = primary_arrays
    if DEBUG_ENABLED:
        from ra_sim.debug_utils import check_ht_arrays

        check_ht_arrays(primary_miller, primary_intens)
        debug_print("miller1 dtype:", primary_miller.dtype, "shape:", primary_miller.shape)
        if primary_miller.size > 0:
            debug_print("L range:", primary_miller[:, 2].min(), primary_miller[:, 2].max())
        debug_print(
            "intens1 dtype:",
            primary_intens.dtype,
            "min:",
            primary_intens.min() if primary_intens.size > 0 else 0.0,
            "max:",
            primary_intens.max() if primary_intens.size > 0 else 0.0,
        )
        debug_print("miller1 contiguous:", primary_miller.flags["C_CONTIGUOUS"])
        debug_print("intens1 contiguous:", primary_intens.flags["C_CONTIGUOUS"])

    if has_second_cif:
        debug_print("combined miller count:", structure_model_state.miller.shape[0])
    else:
        debug_print("single CIF miller count:", structure_model_state.miller.shape[0])

    _sync_structure_model_aliases()


def _bootstrap_structure_model_state_for_startup() -> None:
    global structure_model_state

    if not bool(getattr(structure_model_state, "bootstrap_complete", False)):
        structure_model_state = gui_structure_model.bootstrap_structure_model_state(
            structure_model_state,
            combine_weighted_intensities=gui_controllers.combine_cif_weighted_intensities,
            debug_print=debug_print,
        )
        _sync_structure_model_aliases()
    _apply_structure_model_runtime_cache_state()


def _ensure_structure_model_dataframes() -> tuple[object, object]:
    if (
        structure_model_state.df_summary is None
        or structure_model_state.df_details is None
    ):
        df_summary, df_details = build_intensity_dataframes(
            structure_model_state.miller,
            structure_model_state.intensities,
            structure_model_state.degeneracy,
            structure_model_state.details,
        )
        structure_model_state.df_summary = df_summary
        structure_model_state.df_details = df_details
        _sync_structure_model_aliases()
    return structure_model_state.df_summary, structure_model_state.df_details


def _current_primary_cif_path() -> str:
    """Return the active primary CIF path from structure-model runtime state."""

    return str(structure_model_state.cif_file)


def _occupancy_control_vars() -> list[tk.DoubleVar]:
    """Return the live occupancy Tk variables from structure-model runtime state."""

    return structure_model_state.occ_vars


def _atom_site_fractional_rows() -> list[dict[str, object]]:
    """Return the active atom-site fractional coordinate metadata rows."""

    return structure_model_state.atom_site_fractional_metadata


def _atom_site_fractional_control_vars() -> list[dict[str, tk.DoubleVar]]:
    """Return the live atom-site fractional Tk variables from runtime state."""

    return structure_model_state.atom_site_fract_vars


_sync_structure_model_aliases()

BRAGG_QR_L_KEY_SCALE = gui_controllers.BRAGG_QR_L_KEY_SCALE
BRAGG_QR_L_INVALID_KEY = gui_controllers.BRAGG_QR_L_INVALID_KEY

_apply_structure_model_runtime_cache_state()


def export_initial_excel():
    """Write the initial intensity tables to Excel when enabled."""

    if not write_excel:
        return
    df_summary, df_details = _ensure_structure_model_dataframes()

    import pandas as pd

    download_dir = get_dir("downloads")
    excel_path = download_dir / "miller_intensities.xlsx"

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_details.to_excel(writer, sheet_name="Details", index=False)

        workbook = writer.book
        summary_sheet = writer.sheets["Summary"]
        details_sheet = writer.sheets["Details"]

        header_format = workbook.add_format(
            {
                "bold": True,
                "text_wrap": True,
                "valign": "vcenter",
                "align": "center",
                "fg_color": "#4F81BD",
                "font_color": "#FFFFFF",
                "border": 1,
            }
        )
        for col_num, col_name in enumerate(df_summary.columns):
            summary_sheet.write(0, col_num, col_name, header_format)
            summary_sheet.set_column(col_num, col_num, 18)

        header_format_details = workbook.add_format(
            {
                "bold": True,
                "text_wrap": True,
                "valign": "vcenter",
                "align": "center",
                "fg_color": "#4BACC6",
                "font_color": "#FFFFFF",
                "border": 1,
            }
        )
        for col_num, col_name in enumerate(df_details.columns):
            details_sheet.write(0, col_num, col_name, header_format_details)
            details_sheet.set_column(col_num, col_num, 18)

        last_row = len(df_summary) + 1
        summary_sheet.conditional_format(
            f"D2:D{last_row}",
            {
                "type": "3_color_scale",
                "min_color": "#FFFFFF",
                "mid_color": "#FFEB84",
                "max_color": "#FF0000",
            },
        )

        zebra_format = workbook.add_format({"bg_color": "#F2F2F2"})
        for row in range(1, len(df_details) + 1):
            if row % 2 == 1:
                details_sheet.set_row(row, cell_format=zebra_format)

    print(f"Excel file saved at {excel_path}")


app_shell_view_state = app_state.app_shell_view
status_panel_view_state = app_state.status_panel_view
background_theta_controls_view_state = app_state.background_theta_controls_view
workspace_panels_view_state = app_state.workspace_panels_view
background_backend_debug_view_state = app_state.background_backend_debug_view
background_theta_list_var = None
geometry_theta_offset_var = None
geometry_fit_background_selection_var = None
fit_theta_checkbutton = None
_geometry_fit_runtime_value_callbacks = None
_geometry_fit_var_map: dict[str, object] = {}


def _geometry_fit_runtime_values() -> gui_geometry_fit.GeometryFitRuntimeValueCallbacks:
    """Return the bound live geometry-fit value readers for the runtime."""

    callbacks = _geometry_fit_runtime_value_callbacks
    if callbacks is None:
        raise RuntimeError("Geometry-fit runtime values are not initialized.")
    return callbacks


def _sync_background_runtime_state() -> None:
    """Normalize shared background-runtime state after one update path mutates it."""

    gui_background_manager.normalize_background_runtime_state(
        background_runtime_state,
        file_paths_state=app_state.file_paths,
    )


_sync_background_runtime_state()


def _background_theta_default_value() -> float:
    """Return the fallback theta value used when no per-background list exists."""
    return gui_background_theta.background_theta_default_value(
        theta_initial_var=globals().get("theta_initial_var"),
        defaults=defaults,
        theta_initial=theta_initial,
    )


def _format_background_theta_values(values: Sequence[object]) -> str:
    """Format per-background theta values for the GUI entry."""
    return gui_background_theta.format_background_theta_values(values)


def _parse_background_theta_values(
    raw_text: object,
    *,
    expected_count: int | None = None,
) -> list[float]:
    """Parse comma/space/semicolon separated background theta values."""
    return gui_background_theta.parse_background_theta_values(
        raw_text,
        expected_count=expected_count,
    )


def _default_geometry_fit_background_selection() -> str:
    """Return the default geometry-fit background selector text."""
    return gui_background_theta.default_geometry_fit_background_selection(
        osc_files=background_runtime_state.osc_files,
    )


def _format_geometry_fit_background_indices(indices: Sequence[object]) -> str:
    """Format a background-index list using 1-based labels for the GUI entry."""
    return gui_background_theta.format_geometry_fit_background_indices(indices)


def _parse_geometry_fit_background_indices(
    raw_text: object,
    *,
    total_count: int,
    current_index: int = 0,
) -> list[int]:
    """Parse geometry-fit background selection text into 0-based indices."""
    return gui_background_theta.parse_geometry_fit_background_indices(
        raw_text,
        total_count=total_count,
        current_index=current_index,
    )


def _refresh_geometry_fit_theta_checkbox_label() -> None:
    """Update the theta fit toggle label for single vs multi-background mode."""
    gui_background_theta.refresh_geometry_fit_theta_checkbox_label(
        fit_theta_checkbutton=fit_theta_checkbutton,
        theta_controls=geometry_fit_constraints_view_state.controls,
        shared_theta=_geometry_fit_uses_shared_theta_offset(),
    )


def _load_background_image_by_index(index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return cached background arrays for *index*, loading from disk if needed."""

    updated = gui_background_manager.load_background_image_by_index(
        background_runtime_state,
        int(index),
        display_rotate_k=DISPLAY_ROTATE_K,
        read_osc=read_osc,
    )
    _sync_background_runtime_state()
    return (
        np.asarray(updated["background_image"]),
        np.asarray(updated["background_display"]),
    )


def _get_current_background_native() -> np.ndarray:
    """Return the unrotated background image corresponding to the current index."""

    updated = gui_background_manager.get_current_background_native(
        background_runtime_state,
        display_rotate_k=DISPLAY_ROTATE_K,
        read_osc=read_osc,
    )
    _sync_background_runtime_state()
    return np.asarray(updated["background_image"])


def _get_current_background_display() -> np.ndarray:
    """Return the rotated background image used for GUI display."""

    updated = gui_background_manager.get_current_background_display(
        background_runtime_state,
        display_rotate_k=DISPLAY_ROTATE_K,
        read_osc=read_osc,
    )
    _sync_background_runtime_state()
    return np.asarray(updated["background_display"])


def _apply_background_backend_orientation(image: np.ndarray | None) -> np.ndarray | None:
    """Apply debug-only backend rotations/flips to the background array."""

    return gui_background.apply_background_backend_orientation(
        image,
        flip_x=background_runtime_state.backend_flip_x,
        flip_y=background_runtime_state.backend_flip_y,
        rotation_k=background_runtime_state.backend_rotation_k,
    )


def _get_current_background_backend() -> np.ndarray | None:
    """Return the background array used for backend comparisons (debug)."""

    return _apply_background_backend_orientation(_get_current_background_native())


def _backend_background_to_native_detector_coords(
    col: float,
    row: float,
) -> tuple[float | None, float | None]:
    """Undo backend-only background orientation for one detector-space point."""

    native_background = _get_current_background_native()
    if native_background is None:
        return None, None
    return gui_background.background_backend_point_to_native_coords(
        float(col),
        float(row),
        native_shape=np.asarray(native_background).shape[:2],
        flip_x=background_runtime_state.backend_flip_x,
        flip_y=background_runtime_state.backend_flip_y,
        rotation_k=background_runtime_state.backend_rotation_k,
    )


def _rotate_point_for_display(col: float, row: float, shape: tuple[int, ...], k: int):
    """Rotate a single (col, row) pair by ``k`` using the same rule as ``np.rot90``.

    The transformation mirrors what ``np.rot90`` does to the underlying image so
    point overlays stay aligned with whichever orientation we render.
    """
    return gui_geometry_overlay.rotate_point_for_display(col, row, shape, k)


def _rotate_measured_peaks_for_display(measured, rotated_shape):
    """Rotate measured-peak coordinates to match the displayed background."""
    return gui_geometry_overlay.rotate_measured_peaks_for_display(
        measured,
        rotated_shape,
        display_rotate_k=DISPLAY_ROTATE_K,
    )


def _unrotate_display_peaks(measured, rotated_shape, *, k=None):
    """Undo a display rotation on peak coordinates.

    Pass ``k`` to match the rotation applied for display (e.g. ``DISPLAY_ROTATE_K``
    or ``SIM_DISPLAY_ROTATE_K``), so the returned points land back in that image's
    native orientation.
    """

    return gui_geometry_overlay.unrotate_display_peaks(
        measured,
        rotated_shape,
        k=k,
        default_display_rotate_k=DISPLAY_ROTATE_K,
    )


def _apply_indexing_mode_to_entries(
    measured,
    shape: tuple[int, int],
    *,
    indexing_mode: str = "xy",
):
    """Swap x/y coordinates when using alternate indexing modes."""
    return gui_geometry_overlay.apply_indexing_mode_to_entries(
        measured,
        shape,
        indexing_mode=indexing_mode,
    )


def _apply_orientation_to_entries(
    measured,
    rotated_shape,
    *,
    indexing_mode: str = "xy",
    k: int = 0,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_order: str = "yx",
):
    """Apply backend-only rotations/flips to measured peak entries."""
    return gui_geometry_overlay.apply_orientation_to_entries(
        measured,
        rotated_shape,
        indexing_mode=indexing_mode,
        k=k,
        flip_x=flip_x,
        flip_y=flip_y,
        flip_order=flip_order,
    )


def _orient_image_for_fit(
    image: np.ndarray | None,
    *,
    indexing_mode: str = "xy",
    k: int = 0,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_order: str = "yx",
):
    """Return a rotated/flipped copy of ``image`` for backend fitting only."""
    return gui_geometry_overlay.orient_image_for_fit(
        image,
        indexing_mode=indexing_mode,
        k=k,
        flip_x=flip_x,
        flip_y=flip_y,
        flip_order=flip_order,
    )


def _native_sim_to_display_coords(col: float, row: float, image_shape: tuple[int, ...]):
    """Rotate native simulation coordinates into the displayed frame."""
    return gui_geometry_overlay.native_sim_to_display_coords(
        col,
        row,
        image_shape,
        sim_display_rotate_k=SIM_DISPLAY_ROTATE_K,
    )


def _display_to_native_sim_coords(col: float, row: float, image_shape: tuple[int, ...]):
    """Map displayed simulation coordinates back into native simulation frame."""
    return gui_geometry_overlay.display_to_native_sim_coords(
        col,
        row,
        image_shape,
        sim_display_rotate_k=SIM_DISPLAY_ROTATE_K,
    )


def _transform_points_orientation(
    points: list[tuple[float, float]],
    shape: tuple[int, int],
    *,
    indexing_mode: str = "xy",
    k: int = 0,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_order: str = "yx",
) -> list[tuple[float, float]]:
    """Apply flips/rotations to a list of (col, row) points for diagnostics."""
    return gui_geometry_overlay.transform_points_orientation(
        points,
        shape,
        indexing_mode=indexing_mode,
        k=k,
        flip_x=flip_x,
        flip_y=flip_y,
        flip_order=flip_order,
    )


def _best_orientation_alignment(
    sim_coords: list[tuple[float, float]],
    meas_coords: list[tuple[float, float]],
    shape: tuple[int, int],
):
    """Search over 90° rotations and axis flips to minimize RMS distance."""
    return gui_geometry_overlay.best_orientation_alignment(
        sim_coords,
        meas_coords,
        shape,
    )


def _orientation_metrics(
    sim_coords: list[tuple[float, float]],
    meas_coords: list[tuple[float, float]],
    shape: tuple[int, int],
    *,
    indexing_mode: str,
    k: int,
    flip_x: bool,
    flip_y: bool,
    flip_order: str,
):
    """Return RMS/mean/max distance after transforming measured coordinates."""
    return gui_geometry_overlay.orientation_metrics(
        sim_coords,
        meas_coords,
        shape,
        indexing_mode=indexing_mode,
        k=k,
        flip_x=flip_x,
        flip_y=flip_y,
        flip_order=flip_order,
    )


def _select_fit_orientation(
    sim_coords: list[tuple[float, float]],
    meas_coords: list[tuple[float, float]],
    shape: tuple[int, int],
    *,
    cfg: dict[str, object] | None = None,
):
    """Choose a measured-peak orientation transform that best aligns to simulation."""
    return gui_geometry_overlay.select_fit_orientation(
        sim_coords,
        meas_coords,
        shape,
        cfg=cfg,
    )


def _aggregate_match_centers(
    sim_coords: list[tuple[float, float]],
    meas_coords: list[tuple[float, float]],
    sim_millers: list[tuple[int, int, int]],
    meas_millers: list[tuple[int, int, int]],
):
    """Collapse matched peaks by HKL and return centroid pairs."""
    return gui_geometry_overlay.aggregate_match_centers(
        sim_coords,
        meas_coords,
        sim_millers,
        meas_millers,
    )


def _normalize_hkl_key(
    value: object,
) -> tuple[int, int, int] | None:
    """Return a rounded integer HKL tuple when *value* looks like one."""
    from ra_sim.gui.geometry_overlay import normalize_hkl_key

    return normalize_hkl_key(value)


def _aggregate_initial_geometry_display_pairs(
    initial_pairs_display: Sequence[dict[str, object]] | None,
) -> dict[tuple[int, int, int], dict[str, tuple[float, float]]]:
    """Aggregate initial display-frame picks by HKL."""
    return gui_geometry_overlay.aggregate_initial_geometry_display_pairs(initial_pairs_display)


def _iter_orientation_transform_candidates():
    """Yield all discrete orientation transform candidates."""
    return gui_geometry_overlay.iter_orientation_transform_candidates()


def _inverse_orientation_transform(
    shape: tuple[int, int],
    orientation_choice: dict[str, object] | None,
) -> dict[str, object]:
    """Return the inverse transform for an orientation-choice dict."""
    return gui_geometry_overlay.inverse_orientation_transform(
        shape,
        orientation_choice,
    )


def _draw_geometry_fit_overlay(
    overlay_records: Sequence[dict[str, object]] | None,
    *,
    max_display_markers: int = 120,
) -> None:
    """Draw one fixed-background/fitted-simulation overlay record per match."""

    if not _geometry_overlays_enabled():
        _clear_all_geometry_overlay_artists(redraw=True)
        return

    gui_overlays.draw_geometry_fit_overlay(
        ax,
        overlay_records,
        geometry_pick_artists=geometry_runtime_state.pick_artists,
        clear_geometry_pick_artists=_clear_geometry_pick_artists,
        draw_idle=_request_overlay_canvas_redraw,
        max_display_markers=max_display_markers,
        show_caked_2d=bool(analysis_view_controls_view_state.show_caked_2d_var.get()),
        native_detector_coords_to_caked_display_coords=(
            _native_detector_coords_to_caked_display_coords
        ),
    )


def _draw_initial_geometry_pairs_overlay(
    initial_pairs_display: Sequence[dict[str, object]] | None,
    *,
    max_display_markers: int = 120,
) -> None:
    """Draw only the initially selected simulation/background peak pairs."""

    if not _geometry_overlays_enabled():
        _clear_all_geometry_overlay_artists(redraw=True)
        return

    gui_overlays.draw_initial_geometry_pairs_overlay(
        ax,
        initial_pairs_display,
        geometry_pick_artists=geometry_runtime_state.pick_artists,
        clear_geometry_pick_artists=_clear_geometry_pick_artists,
        draw_idle=_request_overlay_canvas_redraw,
        max_display_markers=max_display_markers,
        show_pair_connectors=bool(analysis_view_controls_view_state.show_caked_2d_var.get()),
    )


def _geometry_overlay_frame_diagnostics(
    overlay_records: Sequence[dict[str, object]] | None,
) -> tuple[dict[str, float], str]:
    """Compare per-match fitted overlay alignment in display space."""

    return compute_geometry_overlay_frame_diagnostics(
        overlay_records,
        show_caked_2d=bool(analysis_view_controls_view_state.show_caked_2d_var.get()),
        native_detector_coords_to_caked_display_coords=(
            _native_detector_coords_to_caked_display_coords
        ),
    )


# Measured peaks are collected interactively in the current GUI workflow.
# Keep this list for compatibility, but avoid loading a large file at startup.
measured_peaks = []

###############################################################################
#                                  TK SETUP
###############################################################################
root = gui_views.create_root_window("RA-SIM Simulation")
root.minsize(1200, 760)
fit2d_error_sound_var = tk.BooleanVar(value=False)
geometry_fit_caked_roi_enabled_var = tk.BooleanVar(value=True)
geometry_fit_caked_roi_preview_var = tk.BooleanVar(value=False)


def _runtime_report_callback_exception(exc_type, exc_value, exc_tb):
    """Persist Tk callback failures before surfacing them to the user."""

    _append_runtime_update_exception_trace(
        "tk_callback_exception",
        exc_type,
        exc_value,
        exc_tb,
        update_id=getattr(simulation_runtime_state, "current_update_trace_id", None),
        trace_stage=getattr(
            simulation_runtime_state,
            "current_update_trace_stage",
            None,
        ),
        update_phase=getattr(simulation_runtime_state, "update_phase", None),
        update_running=getattr(simulation_runtime_state, "update_running", None),
    )
    try:
        traceback.print_exception(exc_type, exc_value, exc_tb)
    except Exception:
        pass
    try:
        messagebox.showerror(
            "RA-SIM Error",
            "An unexpected GUI callback error occurred.\n"
            f"Trace log: {_runtime_update_trace_path()}",
        )
    except Exception:
        pass


root.report_callback_exception = _runtime_report_callback_exception
_ensure_runtime_update_trace_hooks()
gui_views.create_app_shell(
    root=root,
    view_state=app_shell_view_state,
    fit2d_error_sound_var=fit2d_error_sound_var,
    geometry_fit_caked_roi_enabled_var=geometry_fit_caked_roi_enabled_var,
    geometry_fit_caked_roi_preview_var=geometry_fit_caked_roi_preview_var,
)
gui_fit2d_error_sound.bind_fit2d_backspace_error_sound(
    root,
    enabled_var=fit2d_error_sound_var,
    bell_callback=getattr(root, "bell", None),
)
if (
    app_shell_view_state.workspace_body is None
    or app_shell_view_state.fit_body is None
    or app_shell_view_state.analysis_views_frame is None
    or app_shell_view_state.analysis_exports_frame is None
    or app_shell_view_state.analysis_popout_button is None
    or app_shell_view_state.status_frame is None
    or app_shell_view_state.fig_frame is None
    or app_shell_view_state.canvas_frame is None
    or app_shell_view_state.left_col is None
    or app_shell_view_state.right_col is None
    or app_shell_view_state.plot_frame_1d is None
):
    raise RuntimeError("Top-level GUI shell was not created.")

gui_views.create_workspace_panels(
    parent=app_shell_view_state.workspace_body,
    view_state=workspace_panels_view_state,
)
if (
    workspace_panels_view_state.workspace_actions_frame is None
    or workspace_panels_view_state.workspace_backgrounds_frame is None
    or workspace_panels_view_state.workspace_inputs_frame is None
    or workspace_panels_view_state.workspace_session_frame is None
    or workspace_panels_view_state.workspace_debug_frame is None
):
    raise RuntimeError("Workspace panels were not created.")


def _shutdown_gui():
    """Close all application windows and end the Tk event loop."""

    _append_runtime_update_trace(
        "shutdown_gui",
        update_phase=getattr(simulation_runtime_state, "update_phase", None),
        update_running=getattr(simulation_runtime_state, "update_running", None),
    )
    gui_controllers.clear_tk_after_token(
        root,
        simulation_runtime_state.worker_poll_token,
    )
    simulation_runtime_state.worker_poll_token = None
    gui_controllers.clear_tk_after_token(
        root,
        simulation_runtime_state.analysis_poll_token,
    )
    simulation_runtime_state.analysis_poll_token = None
    gui_controllers.clear_tk_after_token(
        root,
        simulation_runtime_state.interaction_settle_token,
    )
    simulation_runtime_state.interaction_settle_token = None

    executor = simulation_runtime_state.worker_executor
    simulation_runtime_state.worker_executor = None
    simulation_runtime_state.worker_future = None
    simulation_runtime_state.worker_active_job = None
    simulation_runtime_state.worker_queued_job = None
    simulation_runtime_state.worker_ready_result = None
    analysis_executor = simulation_runtime_state.analysis_executor
    simulation_runtime_state.analysis_executor = None
    simulation_runtime_state.analysis_future = None
    simulation_runtime_state.analysis_active_job = None
    simulation_runtime_state.analysis_queued_job = None
    simulation_runtime_state.analysis_ready_result = None
    if executor is not None:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=False)
        except Exception:
            pass
    if analysis_executor is not None:
        try:
            analysis_executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            analysis_executor.shutdown(wait=False)
        except Exception:
            pass
    fast_viewer_workflow = globals().get("fast_viewer_workflow")
    if fast_viewer_workflow is not None:
        try:
            fast_viewer_workflow.shutdown()
        except Exception:
            pass
    primary_viewport_backend = globals().get("primary_viewport_backend")
    if primary_viewport_backend is not None:
        try:
            primary_viewport_backend.shutdown()
        except Exception:
            pass
    try:
        gui_views.close_analysis_popout_window(analysis_popout_view_state)
    except Exception:
        pass

    try:
        if root.winfo_exists():
            root.destroy()
    except tk.TclError:
        # Window is already gone or cannot be destroyed cleanly; proceed to
        # shut down the rest of the application.
        pass

    try:
        root.quit()
    except tk.TclError:
        pass


root.protocol("WM_DELETE_WINDOW", _shutdown_gui)

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect("auto")
matplotlib_canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(
    fig,
    master=app_shell_view_state.canvas_frame,
)
matplotlib_canvas_widget = matplotlib_canvas.get_tk_widget()
gui_main_figure_chrome.configure_matplotlib_canvas_widget(matplotlib_canvas_widget)
PRIMARY_VIEWPORT_BACKEND = gui_tk_primary_viewport.parse_primary_viewport_backend(
    os.environ.get("RA_SIM_PRIMARY_VIEWPORT", "matplotlib")
)
canvas = matplotlib_canvas
primary_viewport_backend = None
FAST_VIEWER_DRAW_INTERVAL_S = 0.08
FAST_VIEWER_STARTUP_ENABLED = gui_runtime_display_acceleration.parse_fast_viewer_env_flag(
    os.environ.get("RA_SIM_FAST_VIEWER", "1")
)
FAST_VIEWER_EMBEDDED_SURFACE_ENABLED = False

global_image_buffer = np.zeros((image_size, image_size), dtype=np.float64)
simulation_runtime_state.unscaled_image = None

# ── replace the original imshow call ────────────────────────────
image_display = ax.imshow(
    gui_display_projection.downsample_raster_for_display(
        global_image_buffer,
        max_size=simulation_runtime_state.main_display_raster_limit,
    ),
    cmap=turbo_white0,
    alpha=0.5,
    zorder=1,
    origin="upper",
)


background_display = ax.imshow(
    gui_display_projection.downsample_raster_for_display(
        background_runtime_state.current_background_display,
        max_size=simulation_runtime_state.main_display_raster_limit,
    ),
    cmap="turbo",
    zorder=0,
    origin="upper",
)

highlight_cmap = gui_integration_range_drag.create_integration_region_highlight_cmap(
    listed_colormap_cls=ListedColormap,
)
integration_region_overlay = ax.imshow(
    gui_display_projection.downsample_raster_for_display(
        np.zeros_like(global_image_buffer),
        max_size=simulation_runtime_state.main_display_raster_limit,
    ),
    cmap=highlight_cmap,
    vmin=0,
    vmax=1,
    origin="upper",
    zorder=4,
    interpolation="nearest",
)
integration_region_overlay.set_visible(False)


_initial_simulation_loading_overlay_artists: list[object] = []


def _clear_initial_simulation_loading_overlay(*, redraw: bool = False) -> None:
    while _initial_simulation_loading_overlay_artists:
        artist = _initial_simulation_loading_overlay_artists.pop()
        try:
            artist.remove()
        except Exception:
            pass
    if bool(redraw):
        try:
            matplotlib_canvas.draw()
        except Exception:
            pass


def _show_initial_simulation_loading_overlay() -> None:
    _clear_initial_simulation_loading_overlay(redraw=False)
    text_artist = ax.text(
        0.5,
        0.56,
        "Loading first simulation may take longer",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="#102A43",
        zorder=10,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="#F7FAFC",
            edgecolor="#BCCCDC",
            alpha=0.96,
        ),
    )
    bar_x = 0.31
    bar_y = 0.485
    bar_w = 0.38
    bar_h = 0.026
    track_artist = Rectangle(
        (bar_x, bar_y),
        bar_w,
        bar_h,
        transform=ax.transAxes,
        facecolor="#E2E8F0",
        edgecolor="#BCCCDC",
        linewidth=1.0,
        alpha=0.96,
        zorder=10,
    )
    fill_artist = Rectangle(
        (bar_x + 0.01, bar_y + 0.004),
        bar_w * 0.58,
        max(bar_h - 0.008, 0.0),
        transform=ax.transAxes,
        facecolor="#4F81BD",
        edgecolor="none",
        alpha=0.96,
        zorder=11,
    )
    ax.add_patch(track_artist)
    ax.add_patch(fill_artist)
    _initial_simulation_loading_overlay_artists.extend(
        [text_artist, track_artist, fill_artist]
    )


def _apply_axes_image_origin(image_display, origin):
    """Set the origin for an AxesImage while tolerating older Matplotlib APIs."""

    setter = getattr(image_display, "set_origin", None)
    if callable(setter):
        try:
            setter(origin)
            return
        except AttributeError:
            pass
    try:
        image_display.origin = origin
    except AttributeError:
        return


def _resolve_primary_caked_extent() -> list[float]:
    raw_extent = getattr(simulation_runtime_state, "last_caked_extent", None)
    if isinstance(raw_extent, Sequence) and len(raw_extent) >= 4:
        try:
            extent = [float(raw_extent[index]) for index in range(4)]
        except Exception:
            extent = []
        if (
            len(extent) == 4
            and all(math.isfinite(value) for value in extent)
            and abs(extent[1] - extent[0]) > 1.0e-12
            and abs(extent[3] - extent[2]) > 1.0e-12
        ):
            return extent
    return [0.0, 90.0, -180.0, 180.0]


def _resolve_primary_q_space_extent() -> list[float]:
    raw_extent = getattr(simulation_runtime_state, "last_q_space_extent", None)
    if isinstance(raw_extent, Sequence) and len(raw_extent) >= 4:
        try:
            extent = [float(raw_extent[index]) for index in range(4)]
        except Exception:
            extent = []
        if (
            len(extent) == 4
            and all(math.isfinite(value) for value in extent)
            and abs(extent[1] - extent[0]) > 1.0e-12
            and abs(extent[3] - extent[2]) > 1.0e-12
        ):
            return extent
    return [0.0, 1.0, -1.0, 1.0]


def _sync_primary_raster_geometry(
    *,
    show_caked_image: bool | None = None,
    view_mode: str | None = None,
) -> None:
    detector_extent = [0.0, float(image_size), float(image_size), 0.0]
    caked_extent = _resolve_primary_caked_extent()
    q_space_extent = _resolve_primary_q_space_extent()
    normalized_mode = str(view_mode or "").strip().lower()
    if normalized_mode not in {"detector", "caked", "q_space"}:
        normalized_mode = "caked" if bool(show_caked_image) else "detector"
    if normalized_mode == "q_space":
        image_geometry = ("lower", q_space_extent)
        background_geometry = ("lower", q_space_extent)
        overlay_geometry = ("lower", q_space_extent)
    elif normalized_mode == "caked":
        image_geometry = ("lower", caked_extent)
        background_geometry = ("lower", caked_extent)
        overlay_geometry = ("lower", caked_extent)
    else:
        image_geometry = ("upper", detector_extent)
        background_geometry = ("upper", detector_extent)
        overlay_geometry = ("upper", detector_extent)
    for artist, geometry in (
        (image_display, image_geometry),
        (background_display, background_geometry),
        (integration_region_overlay, overlay_geometry),
    ):
        if artist is None:
            continue
        origin, extent = geometry
        store_geometry = globals().get("_store_primary_raster_geometry")
        apply_projection = globals().get("_apply_projected_primary_raster_to_artist")
        if callable(store_geometry):
            store_geometry(artist, origin=origin, extent=extent)
        if callable(apply_projection):
            apply_projection(artist)
            continue
        _apply_axes_image_origin(artist, origin)
        setter = getattr(artist, "set_extent", None)
        if callable(setter):
            setter(list(extent))


_sync_primary_raster_geometry(show_caked_image=False)


def _maybe_refresh_run_status_bar() -> None:
    refresh_fn = globals().get("_refresh_run_status_bar")
    if callable(refresh_fn):
        try:
            refresh_fn()
        except Exception:
            pass


def _set_runtime_canvas(target_canvas) -> None:
    global canvas

    canvas = target_canvas


def _fast_viewer_transient_artist_groups() -> tuple[object, ...]:
    runtime_state = globals().get("geometry_runtime_state")
    drag_bindings_factory = globals().get("integration_range_drag_runtime_bindings_factory")
    drag_artists: tuple[object, ...] = ()
    if callable(drag_bindings_factory):
        try:
            drag_bindings = drag_bindings_factory()
        except Exception:
            drag_bindings = None
        if drag_bindings is not None:
            drag_artists = (
                getattr(drag_bindings, "drag_select_rect", None),
                getattr(drag_bindings, "integration_region_rect", None),
            )
    return (
        drag_artists,
        getattr(runtime_state, "pick_artists", ()),
        getattr(runtime_state, "preview_artists", ()),
        getattr(runtime_state, "manual_preview_artists", ()),
        getattr(runtime_state, "qr_cylinder_overlay_artists", ()),
    )


def _fast_viewer_overlay_model():
    drag_state = globals().get("integration_range_drag_state")
    return gui_fast_plot_viewer.build_artist_overlay_model(
        transient_artists=_fast_viewer_transient_artist_groups(),
        transient_curve_specs=getattr(
            drag_state,
            "_fast_viewer_curve_specs",
            (),
        ),
        suppress_overlay_image=bool(
            getattr(drag_state, "_fast_viewer_suppress_overlay_image", False)
        ),
    )


def _fast_viewer_layer_versions() -> dict[str, object]:
    analysis_controls = globals().get("analysis_view_controls_view_state")
    show_caked_var = getattr(analysis_controls, "show_caked_2d_var", None)
    try:
        show_caked = bool(show_caked_var.get()) if show_caked_var is not None else False
    except Exception:
        show_caked = False

    return {
        "background": (
            bool(getattr(background_runtime_state, "visible", False)),
            int(getattr(background_runtime_state, "current_background_index", -1)),
            int(getattr(background_runtime_state, "backend_rotation_k", 0)) % 4,
            bool(getattr(background_runtime_state, "backend_flip_x", False)),
            bool(getattr(background_runtime_state, "backend_flip_y", False)),
            id(getattr(background_runtime_state, "current_background_display", None)),
        ),
        "simulation": (
            "caked" if show_caked else "detector",
            getattr(simulation_runtime_state, "last_unscaled_image_signature", None),
            getattr(simulation_runtime_state, "last_analysis_signature", None),
            getattr(simulation_runtime_state, "last_simulation_signature", None),
            tuple(getattr(simulation_runtime_state, "last_caked_extent", ()) or ()),
        ),
        "overlay": getattr(
            globals().get("integration_range_drag_state"),
            "_fast_viewer_overlay_version",
            None,
        ),
    }


fast_viewer_workflow = gui_runtime_display_acceleration.build_runtime_fast_viewer_workflow(
    fast_plot_viewer_module=gui_fast_plot_viewer,
    tk_module=tk,
    ttk_module=ttk,
    canvas_frame=app_shell_view_state.canvas_frame,
    matplotlib_canvas=matplotlib_canvas,
    ax=ax,
    image_artist=image_display,
    background_artist=background_display,
    overlay_artist=integration_region_overlay,
    marker_artist_factory=lambda: (
        globals().get("center_marker"),
        globals().get("selected_peak_marker"),
    ),
    overlay_model_factory=_fast_viewer_overlay_model,
    layer_versions_factory=_fast_viewer_layer_versions,
    display_controls_view_state_factory=lambda: globals().get("display_controls_view_state"),
    canvas_interaction_callbacks_factory=lambda: globals().get(
        "canvas_interaction_runtime_callbacks"
    ),
    bind_canvas_interactions=(
        gui_runtime_geometry_preview.initialize_runtime_canvas_interaction_bindings
    ),
    set_canvas=_set_runtime_canvas,
    set_progress_text=lambda text: progress_label.config(text=text),
    refresh_run_status_bar=_maybe_refresh_run_status_bar,
    manual_pick_armed_factory=lambda: getattr(
        globals().get("geometry_runtime_state"),
        "manual_pick_armed",
        False,
    ),
    hkl_pick_armed_factory=lambda: getattr(
        globals().get("peak_selection_state"),
        "hkl_pick_armed",
        False,
    ),
    manual_pick_session_active_factory=lambda: (
        globals().get("_geometry_manual_pick_session_active")(require_current_background=False)
        if callable(globals().get("_geometry_manual_pick_session_active"))
        else False
    ),
    geometry_preview_exclude_armed_factory=lambda: getattr(
        globals().get("geometry_preview_state"),
        "exclude_armed",
        False,
    ),
    live_geometry_preview_enabled_factory=lambda: (
        globals().get("_live_geometry_preview_enabled")()
        if callable(globals().get("_live_geometry_preview_enabled"))
        else False
    ),
    qr_overlay_var_factory=lambda: getattr(
        globals().get("geometry_overlay_actions_view_state"),
        "show_qr_cylinder_overlay_var",
        None,
    ),
    overlay_artist_groups_factory=lambda: (
        getattr(globals().get("geometry_runtime_state"), "pick_artists", ()),
        getattr(globals().get("geometry_runtime_state"), "preview_artists", ()),
        getattr(globals().get("geometry_runtime_state"), "manual_preview_artists", ()),
        getattr(
            globals().get("geometry_runtime_state"),
            "qr_cylinder_overlay_artists",
            (),
        ),
    ),
    defer_overlay_redraw_factory=lambda: (
        globals().get("_defer_nonessential_redraw")()
        if callable(globals().get("_defer_nonessential_redraw"))
        else False
    ),
    integration_drag_active_factory=lambda: getattr(
        globals().get("integration_range_drag_state"),
        "active",
        False,
    ),
    draw_interval_s=FAST_VIEWER_DRAW_INTERVAL_S,
    requested_enabled=bool(FAST_VIEWER_STARTUP_ENABLED and FAST_VIEWER_EMBEDDED_SURFACE_ENABLED),
    control_locked=True,
)
_fast_viewer_active = fast_viewer_workflow.active
_fast_viewer_requested_enabled = fast_viewer_workflow.requested_enabled
_fast_viewer_suspend_reason = fast_viewer_workflow.suspend_reason
_set_fast_viewer_requested_enabled = fast_viewer_workflow.set_requested_enabled
_refresh_fast_viewer_runtime_mode = fast_viewer_workflow.refresh_runtime_mode
_request_main_canvas_redraw = fast_viewer_workflow.request_main_canvas_redraw
_request_overlay_canvas_redraw = fast_viewer_workflow.request_overlay_canvas_redraw
_reset_fast_viewer_view = fast_viewer_workflow.reset_view
_toggle_fast_viewer = fast_viewer_workflow.toggle


# ---------------------------------------------------------------------------
#  helper – returns a fully populated, *consistent* mosaic_params dict
# ---------------------------------------------------------------------------
def build_mosaic_params(*, sample_count=None, solve_q_steps=None, rng_seed=None):
    update_mosaic_cache()
    solve_q = current_solve_q_values()
    beam_x_array = simulation_runtime_state.profile_cache["beam_x_array"]
    beam_y_array = simulation_runtime_state.profile_cache["beam_y_array"]
    theta_array = simulation_runtime_state.profile_cache["theta_array"]
    phi_array = simulation_runtime_state.profile_cache["phi_array"]
    wavelength_array = simulation_runtime_state.profile_cache["wavelength_array"]
    n2_sample_array = simulation_runtime_state.profile_cache.get("n2_sample_array")
    sample_weights = simulation_runtime_state.profile_cache.get("sample_weights")
    sampling_signature = tuple(
        simulation_runtime_state.profile_cache.get("_sampling_signature", ())
    )
    target_sample_count = None
    if sample_count is not None:
        try:
            target_sample_count = max(int(sample_count), 1)
        except (TypeError, ValueError):
            target_sample_count = None

    current_sample_count = int(np.size(beam_x_array))
    if (
        target_sample_count is not None
        and target_sample_count > 0
        and target_sample_count != current_sample_count
    ):
        resolved_rng_seed = 0
        if rng_seed is not None:
            try:
                resolved_rng_seed = int(rng_seed)
            except (TypeError, ValueError):
                resolved_rng_seed = 0
        (
            beam_x_array,
            beam_y_array,
            theta_array,
            phi_array,
            wavelength_array,
        ) = generate_random_profiles(
            num_samples=target_sample_count,
            divergence_sigma=divergence_sigma,
            bw_sigma=bw_sigma,
            lambda0=lambda_,
            bandwidth=_current_bandwidth_fraction(),
            rng=resolved_rng_seed,
        )
        sample_weights = None
        sampling_signature = (
            RANDOM_GAUSSIAN_SAMPLING,
            "override",
            int(target_sample_count),
            int(resolved_rng_seed),
            float(divergence_sigma),
            float(bw_sigma),
            float(lambda_),
            float(_current_bandwidth_fraction()),
        )
        n2_sample_array = _current_sample_n2_array(wavelength_array)

    resolved_solve_q_steps = solve_q.steps
    if solve_q_steps is not None:
        try:
            resolved_solve_q_steps = max(int(solve_q_steps), 1)
        except (TypeError, ValueError):
            resolved_solve_q_steps = solve_q.steps

    return {
        "beam_x_array": beam_x_array,
        "beam_y_array": beam_y_array,
        "theta_array": theta_array,
        "phi_array": phi_array,
        "wavelength_array": wavelength_array,  #  <<< name fixed
        "sample_weights": sample_weights,
        "n2_sample_array": n2_sample_array,
        "sigma_mosaic_deg": sigma_mosaic_var.get(),
        "gamma_mosaic_deg": gamma_mosaic_var.get(),
        "eta": eta_var.get(),
        "solve_q_steps": resolved_solve_q_steps,
        "solve_q_rel_tol": solve_q.rel_tol,
        "solve_q_mode": solve_q.mode_flag,
        "_sampling_signature": sampling_signature,
    }


def _normalize_optics_mode_label(value) -> str:
    """Normalize optics mode to UI labels: ``'fast'`` or ``'exact'``."""

    if value is None:
        return "fast"
    if isinstance(value, (int, np.integer)):
        return "exact" if int(value) == OPTICS_MODE_EXACT else "fast"
    if isinstance(value, (float, np.floating)):
        return "exact" if int(round(float(value))) == OPTICS_MODE_EXACT else "fast"

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
    return "fast"


def _current_optics_mode_flag() -> int:
    mode_var = globals().get("optics_mode_var")
    mode_label = _normalize_optics_mode_label(mode_var.get() if mode_var is not None else "fast")
    if mode_label == "exact":
        return OPTICS_MODE_EXACT
    return OPTICS_MODE_FAST


colorbar_main, caked_cbar_ax, caked_colorbar = gui_main_figure_chrome.configure_main_figure_layout(
    fig,
    ax,
    image_display,
)

(center_marker,) = ax.plot(center_default[1], center_default[0], "ro", markersize=5, zorder=2)
center_marker.set_visible(False)

ax.set_xlim(0, image_size)
ax.set_ylim(image_size, 0)
ax.set_xlabel("X (pixels)")
ax.set_ylabel("Y (pixels)")
gui_main_figure_chrome.apply_main_figure_axes_chrome(ax)
canvas.draw()

# -----------------------------------------------------------
# 1)  Highlight‑marker that we can move each click
# -----------------------------------------------------------
(selected_peak_marker,) = ax.plot(
    [],
    [],
    "ys",  # yellow square outline
    markersize=8,
    markerfacecolor="none",
    linewidth=1.5,
    zorder=6,
)
selected_peak_marker.set_visible(False)

_main_matplotlib_preview_controller = (
    gui_main_matplotlib_interaction.MainMatplotlibPreviewController(
        axis=ax,
        canvas=matplotlib_canvas,
        runtime_state=simulation_runtime_state,
        preview_artist_factory=lambda: (
            globals().get("background_display"),
            globals().get("image_display"),
            globals().get("center_marker"),
            globals().get("selected_peak_marker"),
        ),
        hidden_artist_factory=lambda: (globals().get("integration_region_overlay"),),
    )
)

# Geometry click markers (sim vs real)
geometry_runtime_state.pick_artists = []
geometry_runtime_state.preview_artists = []
geometry_runtime_state.manual_preview_artists = []
geometry_runtime_state.qr_cylinder_overlay_artists = []
geometry_runtime_state.qr_cylinder_overlay_cache = _empty_qr_cylinder_overlay_cache()
geometry_preview_state = app_state.geometry_preview
geometry_q_group_view_state = app_state.geometry_q_group_view
geometry_q_group_state = app_state.geometry_q_groups
geometry_manual_state = app_state.manual_geometry
geometry_runtime_state.manual_pick_armed = False
geometry_runtime_state.manual_pick_cache_signature = None
geometry_runtime_state.manual_pick_cache_data = {}
primary_viewport_selection = gui_runtime_primary_viewport.activate_runtime_primary_viewport(
    requested_backend=PRIMARY_VIEWPORT_BACKEND,
    tk_primary_viewport_module=gui_tk_primary_viewport,
    tk_module=tk,
    canvas_frame=app_shell_view_state.canvas_frame,
    matplotlib_canvas=matplotlib_canvas,
    ax=ax,
    image_artist=image_display,
    background_artist=background_display,
    overlay_artist=integration_region_overlay,
    marker_artist_factory=lambda: (
        globals().get("center_marker"),
        globals().get("selected_peak_marker"),
    ),
    overlay_model_factory=_fast_viewer_overlay_model,
    overlay_artist_groups_factory=lambda: (
        getattr(globals().get("geometry_runtime_state"), "pick_artists", ()),
        getattr(globals().get("geometry_runtime_state"), "preview_artists", ()),
        getattr(globals().get("geometry_runtime_state"), "manual_preview_artists", ()),
        getattr(
            globals().get("geometry_runtime_state"),
            "qr_cylinder_overlay_artists",
            (),
        ),
    ),
    layer_versions_factory=_fast_viewer_layer_versions,
    peak_cache_factory=lambda: getattr(
        globals().get("simulation_runtime_state"),
        "peak_positions",
        (),
    ),
    qgroup_cache_factory=lambda: getattr(
        globals().get("geometry_runtime_state"),
        "manual_pick_cache_data",
        {},
    ).get("grouped_candidates", {}),
    draw_interval_s=1.0 / 60.0,
    set_progress_text=lambda text: progress_label.config(text=text),
)
PRIMARY_VIEWPORT_BACKEND = primary_viewport_selection.active_backend
primary_viewport_backend = primary_viewport_selection.backend
_set_runtime_canvas(primary_viewport_selection.canvas_proxy)
primary_canvas_interaction_canvas = None
primary_canvas_interaction_cids: tuple[object, ...] = ()


def _legacy_main_matplotlib_interaction_active() -> bool:
    return bool(
        str(globals().get("PRIMARY_VIEWPORT_BACKEND", "matplotlib")) == "matplotlib"
        and not _fast_viewer_active()
    )


def _main_matplotlib_redraw_widget():
    try:
        return matplotlib_canvas.get_tk_widget()
    except Exception:
        return root


def _legacy_main_matplotlib_preview_controller():
    controller = globals().get("_main_matplotlib_preview_controller")
    if controller is None:
        return None
    return controller


def _draw_legacy_main_matplotlib_canvas_now() -> bool:
    if not _legacy_main_matplotlib_interaction_active():
        return False
    projection_refresh = globals().get("_apply_current_primary_raster_projection")
    if callable(projection_refresh):
        projection_refresh()
    active_canvas = globals().get("canvas", matplotlib_canvas)
    draw_fn = getattr(active_canvas, "draw", None)
    if not callable(draw_fn):
        draw_fn = getattr(matplotlib_canvas, "draw", None)
    if not callable(draw_fn):
        return False
    try:
        draw_fn()
    except Exception:
        return False
    return True


def _cancel_legacy_main_matplotlib_redraw() -> None:
    gui_main_matplotlib_interaction.cancel_pending_main_matplotlib_redraw(
        _main_matplotlib_redraw_widget(),
        simulation_runtime_state,
    )


def _request_legacy_main_matplotlib_redraw(*, force: bool = False) -> None:
    preview_controller = _legacy_main_matplotlib_preview_controller()
    if preview_controller is not None and bool(preview_controller.preview_active()):
        if not bool(force):
            return
        _clear_legacy_main_matplotlib_preview_view(redraw=False)
    gui_main_matplotlib_interaction.request_main_matplotlib_redraw(
        widget=_main_matplotlib_redraw_widget(),
        runtime_state=simulation_runtime_state,
        interval_s=float(MAIN_MATPLOTLIB_INTERACTION_REDRAW_INTERVAL_S),
        perf_counter_fn=perf_counter,
        draw_now=_draw_legacy_main_matplotlib_canvas_now,
        force=bool(force),
    )


def _suspend_legacy_main_matplotlib_overlays() -> None:
    if not _legacy_main_matplotlib_interaction_active():
        return
    gui_main_matplotlib_interaction.suspend_main_matplotlib_overlays(
        simulation_runtime_state,
        suspend_callback=(lambda: _clear_deferred_overlays(clear_qr_overlay=True)),
    )


def _restore_legacy_main_matplotlib_overlays() -> bool:
    if not _legacy_main_matplotlib_interaction_active():
        simulation_runtime_state.main_matplotlib_overlays_suspended = False
        return False
    return bool(
        gui_main_matplotlib_interaction.restore_main_matplotlib_overlays(
            simulation_runtime_state,
            restore_callback=lambda: (
                _refresh_settled_overlays(),
                _request_legacy_main_matplotlib_redraw(force=True),
            ),
        )
    )


def _preview_legacy_main_matplotlib_view_limits(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> bool:
    if not _legacy_main_matplotlib_interaction_active():
        return False
    preview_controller = _legacy_main_matplotlib_preview_controller()
    if preview_controller is None:
        return False
    return bool(preview_controller.preview_view_limits(xlim, ylim))


def _commit_legacy_main_matplotlib_preview_view() -> bool:
    preview_controller = _legacy_main_matplotlib_preview_controller()
    if preview_controller is None or not bool(preview_controller.preview_active()):
        return False
    if not bool(preview_controller.commit_preview_view()):
        return False
    gui_controllers.clear_tk_after_token(
        root,
        simulation_runtime_state.interaction_settle_token,
    )
    simulation_runtime_state.interaction_settle_token = None
    simulation_runtime_state.interaction_drag_active = False
    simulation_runtime_state.interaction_drag_requires_settled_update = False
    _refresh_run_status_bar()
    if not _restore_legacy_main_matplotlib_overlays():
        _request_legacy_main_matplotlib_redraw(force=True)
    return True


def _clear_legacy_main_matplotlib_preview_view(*, redraw: bool = True) -> bool:
    preview_controller = _legacy_main_matplotlib_preview_controller()
    if preview_controller is None:
        return False
    cleared = bool(preview_controller.clear_preview_view(redraw=bool(redraw)))
    if cleared:
        simulation_runtime_state.interaction_drag_active = False
        simulation_runtime_state.interaction_drag_requires_settled_update = False
        gui_controllers.clear_tk_after_token(
            root,
            simulation_runtime_state.interaction_settle_token,
        )
        simulation_runtime_state.interaction_settle_token = None
        simulation_runtime_state.main_matplotlib_overlays_suspended = False
        _refresh_run_status_bar()
    return cleared


def _configure_primary_viewport_redraw_helpers() -> None:
    global _request_main_canvas_redraw
    global _request_overlay_canvas_redraw
    global _reset_fast_viewer_view

    if str(globals().get("PRIMARY_VIEWPORT_BACKEND", "matplotlib")) == "tk_canvas":

        def _request_main_canvas_redraw(*, force_matplotlib: bool = False) -> None:
            active_canvas = globals().get("canvas", matplotlib_canvas)
            draw_name = "draw" if bool(force_matplotlib) else "draw_idle"
            projection_refresh = globals().get("_apply_current_primary_raster_projection")
            if callable(projection_refresh):
                projection_refresh()
            try:
                draw_fn = getattr(active_canvas, draw_name, None)
                if callable(draw_fn):
                    draw_fn()
                    return
            except Exception:
                pass
            fallback_draw = getattr(matplotlib_canvas, draw_name, None)
            if callable(fallback_draw):
                try:
                    fallback_draw()
                except Exception:
                    pass

        def _request_overlay_canvas_redraw(*, force: bool = False) -> None:
            if not force:
                defer_redraw = globals().get("_defer_nonessential_redraw")
                if callable(defer_redraw):
                    try:
                        if bool(defer_redraw()):
                            return
                    except Exception:
                        pass
            _request_main_canvas_redraw(force_matplotlib=bool(force))

        def _reset_fast_viewer_view() -> None:
            backend = globals().get("primary_viewport_backend")
            if backend is None:
                return
            try:
                backend.sync_from_matplotlib(force_view_range=True)
            except Exception:
                pass

        return

    def _request_main_canvas_redraw(*, force_matplotlib: bool = False) -> None:
        if _fast_viewer_active():
            _clear_legacy_main_matplotlib_preview_view(redraw=False)
            _cancel_legacy_main_matplotlib_redraw()
            fast_viewer_workflow.request_main_canvas_redraw(force_matplotlib=bool(force_matplotlib))
            return
        _request_legacy_main_matplotlib_redraw(force=bool(force_matplotlib))

    def _request_overlay_canvas_redraw(*, force: bool = False) -> None:
        if _fast_viewer_active():
            _clear_legacy_main_matplotlib_preview_view(redraw=False)
            _cancel_legacy_main_matplotlib_redraw()
            fast_viewer_workflow.request_overlay_canvas_redraw(force=bool(force))
            return
        if not force:
            defer_redraw = globals().get("_defer_nonessential_redraw")
            if callable(defer_redraw):
                try:
                    if bool(defer_redraw()):
                        return
                except Exception:
                    pass
        _request_legacy_main_matplotlib_redraw(force=bool(force))

    _reset_fast_viewer_view = fast_viewer_workflow.reset_view


def _bind_primary_canvas_interactions(target_canvas) -> None:
    global primary_canvas_interaction_canvas
    global primary_canvas_interaction_cids

    old_canvas = primary_canvas_interaction_canvas
    old_cids = tuple(primary_canvas_interaction_cids or ())
    disconnect = getattr(old_canvas, "mpl_disconnect", None)
    if callable(disconnect):
        for cid in old_cids:
            try:
                disconnect(cid)
            except Exception:
                pass

    callbacks = globals().get("canvas_interaction_runtime_callbacks")
    if callbacks is None or target_canvas is None:
        primary_canvas_interaction_canvas = target_canvas
        primary_canvas_interaction_cids = ()
        return

    try:
        primary_canvas_interaction_cids = tuple(
            gui_runtime_geometry_preview.initialize_runtime_canvas_interaction_bindings(
                canvas=target_canvas,
                callbacks=callbacks,
            )
            or ()
        )
    except Exception:
        primary_canvas_interaction_cids = ()
    primary_canvas_interaction_canvas = target_canvas


_configure_primary_viewport_redraw_helpers()

geometry_fit_history_state = app_state.geometry_fit_history
geometry_fit_dataset_cache_state = app_state.geometry_fit_dataset_cache
geometry_fit_parameter_controls_view_state = app_state.geometry_fit_parameter_controls_view
geometry_fit_constraints_view_state = app_state.geometry_fit_constraints_view
geometry_tool_actions_view_state = app_state.geometry_tool_actions_view
GEOMETRY_MANUAL_UNDO_LIMIT = 100
GEOMETRY_FIT_UNDO_LIMIT = 16
GEOMETRY_PREVIEW_TOGGLE_MAX_DISTANCE_PX = 14.0
GEOMETRY_MANUAL_PICK_SEARCH_WINDOW_PX = 50.0
GEOMETRY_MANUAL_PICK_ZOOM_WINDOW_PX = 100.0
GEOMETRY_MANUAL_CAKED_SEARCH_TTH_DEG = 1.5
GEOMETRY_MANUAL_CAKED_SEARCH_PHI_DEG = 10.0
GEOMETRY_MANUAL_CAKED_ZOOM_TTH_DEG = 4.0
GEOMETRY_MANUAL_CAKED_ZOOM_PHI_DEG = 24.0
GEOMETRY_MANUAL_PREVIEW_MIN_INTERVAL_S = 0.03
GEOMETRY_MANUAL_PREVIEW_MIN_MOVE_PX = 0.8
GEOMETRY_MANUAL_POSITION_SIGMA_FLOOR_PX = 0.75


def _geometry_manual_position_error_px(
    raw_col: float,
    raw_row: float,
    refined_col: float,
    refined_row: float,
) -> float:
    """Return the click-to-refined placement error in display pixels."""
    return gui_manual_geometry.geometry_manual_position_error_px(
        raw_col,
        raw_row,
        refined_col,
        refined_row,
    )


def _geometry_manual_position_sigma_px(
    placement_error_px: object,
    *,
    floor_px: float | None = None,
) -> float:
    """Convert a manual click-placement error into a fit sigma in pixels."""
    if floor_px is None:
        floor_px = float(globals().get("GEOMETRY_MANUAL_POSITION_SIGMA_FLOOR_PX", 0.75))
    return gui_manual_geometry.geometry_manual_position_sigma_px(
        placement_error_px,
        floor_px=float(floor_px),
    )


def _normalize_geometry_manual_pair_entry(
    entry: dict[str, object] | None,
) -> dict[str, object] | None:
    """Normalize one saved manual geometry-pair entry."""
    return gui_manual_geometry.normalize_geometry_manual_pair_entry(
        entry,
        normalize_hkl_key=_normalize_hkl_key,
        sigma_floor_px=float(GEOMETRY_MANUAL_POSITION_SIGMA_FLOOR_PX),
    )


def _geometry_manual_pairs_for_index(index: int) -> list[dict[str, object]]:
    """Return normalized saved manual geometry pairs for one background index."""
    return gui_manual_geometry.geometry_manual_pairs_for_index(
        index,
        pairs_by_background=geometry_manual_state.pairs_by_background,
        normalize_hkl_key=_normalize_hkl_key,
        sigma_floor_px=float(GEOMETRY_MANUAL_POSITION_SIGMA_FLOOR_PX),
    )


def _set_geometry_manual_pairs_for_index(
    index: int,
    entries: Sequence[dict[str, object]] | None,
) -> list[dict[str, object]]:
    """Replace one background's saved manual geometry-pair list."""
    return gui_manual_geometry.set_geometry_manual_pairs_for_index(
        index,
        entries,
        pairs_by_background=geometry_manual_state.pairs_by_background,
        normalize_hkl_key=_normalize_hkl_key,
        sigma_floor_px=float(GEOMETRY_MANUAL_POSITION_SIGMA_FLOOR_PX),
    )


def _geometry_manual_pair_group_count(index: int) -> int:
    """Return how many distinct Qr/Qz groups are saved for one background."""
    return gui_manual_geometry.geometry_manual_pair_group_count(
        index,
        pairs_by_background=geometry_manual_state.pairs_by_background,
        normalize_hkl_key=_normalize_hkl_key,
        sigma_floor_px=float(GEOMETRY_MANUAL_POSITION_SIGMA_FLOOR_PX),
    )


def _replace_geometry_manual_pairs_by_background(
    pairs_by_background: dict[int, list[dict[str, object]]] | None,
) -> dict[int, list[dict[str, object]]]:
    """Replace the stored manual-geometry pair map in the shared state container."""
    return gui_controllers.replace_manual_geometry_pairs_by_background(
        geometry_manual_state,
        pairs_by_background,
    )


def _set_geometry_manual_pick_session(
    pick_session: dict[str, object] | None,
) -> dict[str, object]:
    """Replace the active manual-geometry pick session in the shared state container."""
    result = gui_controllers.replace_manual_geometry_pick_session(
        geometry_manual_state,
        pick_session,
    )
    _refresh_fast_viewer_runtime_mode(announce=False)
    return result


def _clear_geometry_manual_undo_stack() -> None:
    """Discard the manual-placement undo history."""
    gui_controllers.clear_manual_geometry_undo_stack(geometry_manual_state)


def _geometry_manual_undo_snapshot() -> gui_state.ManualGeometryUndoSnapshot:
    """Return a deep copy of the manual-placement state for undo."""
    return gui_controllers.build_manual_geometry_undo_snapshot(geometry_manual_state)


def _push_geometry_manual_undo_state() -> None:
    """Push the current manual-placement state onto the undo stack."""
    gui_controllers.push_manual_geometry_undo_state(
        geometry_manual_state,
        limit=int(GEOMETRY_MANUAL_UNDO_LIMIT),
    )


def _undo_last_geometry_manual_placement() -> None:
    """Restore the most recent manual-placement state."""

    if not geometry_manual_state.undo_stack:
        progress_label_geometry.config(text="No manual placement changes are available to undo.")
        return

    gui_controllers.restore_last_manual_geometry_undo_state(geometry_manual_state)
    _clear_geometry_fit_dataset_cache()
    _clear_geometry_manual_preview_artists(redraw=False)
    _render_current_geometry_manual_pairs(update_status=True)
    _update_geometry_manual_pick_button_label()
    _refresh_background_status()
    progress_label_geometry.config(text="Undid the last manual placement change.")


def _refine_geometry_manual_pair_entry_from_cache(
    entry: dict[str, object] | None,
    *,
    source_entry: dict[str, object] | None = None,
) -> dict[str, object] | None:
    """Refine one saved manual-pair simulation point from the cached caked peak map."""

    if not isinstance(entry, dict):
        return None

    updated_entry = dict(entry)
    try:
        caked_sim_image = np.asarray(
            simulation_runtime_state.last_caked_image_unscaled,
            dtype=float,
        )
        radial_axis = np.asarray(
            simulation_runtime_state.last_caked_radial_values,
            dtype=float,
        )
        azimuth_axis = np.asarray(
            simulation_runtime_state.last_caked_azimuth_values,
            dtype=float,
        )
    except Exception:
        return updated_entry

    if (
        caked_sim_image.ndim != 2
        or caked_sim_image.size == 0
        or radial_axis.size <= 0
        or azimuth_axis.size <= 0
    ):
        return updated_entry

    stored_sim_image = getattr(simulation_runtime_state, "stored_sim_image", None)
    if stored_sim_image is not None:
        native_image_shape = tuple(int(v) for v in np.asarray(stored_sim_image).shape[:2])
    else:
        native_image_shape = (int(image_size), int(image_size))

    placement_refine_cache: dict[str, object] = {}
    try:
        placement_refine_cache = _get_geometry_manual_pick_cache(
            param_set=dict(_current_geometry_fit_params()),
            prefer_cache=True,
            background_image=_current_geometry_manual_pick_background_image(),
        )
        sim_match_cfg = dict(placement_refine_cache.get("match_config", {}))
    except Exception:
        sim_match_cfg = {}
    try:
        resolved_sim_match_cfg, sim_background_context = _auto_match_background_context(
            caked_sim_image,
            sim_match_cfg,
        )
    except Exception:
        resolved_sim_match_cfg = dict(sim_match_cfg)
        sim_background_context = None
    sim_refine_cache = {
        "match_config": dict(resolved_sim_match_cfg),
        "background_context": sim_background_context,
    }

    resolved_source_entry = dict(source_entry) if isinstance(source_entry, dict) else None
    source_key: tuple[int, int] | None
    try:
        source_key = (
            int(updated_entry.get("source_table_index")),
            int(updated_entry.get("source_row_index")),
        )
    except Exception:
        source_key = None
    if not isinstance(resolved_source_entry, dict):
        simulated_lookup = dict(placement_refine_cache.get("simulated_lookup", {}))
        resolved_source_entry = simulated_lookup.get(source_key) if source_key is not None else None
    if not isinstance(resolved_source_entry, dict):
        return updated_entry

    try:
        seed_tth = float(
            resolved_source_entry.get(
                "caked_x",
                resolved_source_entry.get("two_theta_deg", np.nan),
            )
        )
        seed_phi = float(
            resolved_source_entry.get(
                "caked_y",
                resolved_source_entry.get("phi_deg", np.nan),
            )
        )
    except Exception:
        seed_tth = float("nan")
        seed_phi = float("nan")
    if not (np.isfinite(seed_tth) and np.isfinite(seed_phi)):
        try:
            seed_tth = float(
                updated_entry.get(
                    "refined_sim_caked_x",
                    updated_entry.get("caked_x", updated_entry.get("raw_caked_x", np.nan)),
                )
            )
            seed_phi = float(
                updated_entry.get(
                    "refined_sim_caked_y",
                    updated_entry.get("caked_y", updated_entry.get("raw_caked_y", np.nan)),
                )
            )
        except Exception:
            seed_tth = float("nan")
            seed_phi = float("nan")
    if not (np.isfinite(seed_tth) and np.isfinite(seed_phi)):
        return updated_entry

    refined_tth, refined_phi = gui_manual_geometry.geometry_manual_refine_preview_point(
        dict(resolved_source_entry),
        float(seed_tth),
        float(seed_phi),
        display_background=caked_sim_image,
        cache_data=sim_refine_cache,
        use_caked_space=True,
        radial_axis=radial_axis,
        azimuth_axis=azimuth_axis,
        match_simulated_peaks_to_peak_context=match_simulated_peaks_to_peak_context,
        caked_axis_to_image_index_fn=_caked_axis_to_image_index,
        caked_image_index_to_axis_fn=_caked_image_index_to_axis,
        refine_caked_peak_center_fn=_refine_caked_peak_center,
    )
    if not (np.isfinite(refined_tth) and np.isfinite(refined_phi)):
        return updated_entry

    updated_entry["refined_sim_caked_x"] = float(refined_tth)
    updated_entry["refined_sim_caked_y"] = float(refined_phi)

    try:
        refined_background_display = _caked_angles_to_background_display_coords(
            float(refined_tth),
            float(refined_phi),
        )
    except Exception:
        refined_background_display = None
    try:
        native_point = (
            _background_display_to_native_detector_coords(
                float(refined_background_display[0]),
                float(refined_background_display[1]),
            )
            if (
                isinstance(refined_background_display, tuple)
                and len(refined_background_display) >= 2
                and refined_background_display[0] is not None
                and refined_background_display[1] is not None
                and np.isfinite(float(refined_background_display[0]))
                and np.isfinite(float(refined_background_display[1]))
            )
            else None
        )
    except Exception:
        native_point = None
    if (
        isinstance(native_point, tuple)
        and len(native_point) >= 2
        and np.isfinite(float(native_point[0]))
        and np.isfinite(float(native_point[1]))
    ):
        updated_entry["refined_sim_native_x"] = float(native_point[0])
        updated_entry["refined_sim_native_y"] = float(native_point[1])
        try:
            refined_display = _native_sim_to_display_coords(
                float(native_point[0]),
                float(native_point[1]),
                native_image_shape,
            )
        except Exception:
            refined_display = None
        if (
            isinstance(refined_display, tuple)
            and len(refined_display) >= 2
            and np.isfinite(float(refined_display[0]))
            and np.isfinite(float(refined_display[1]))
        ):
            updated_entry["refined_sim_x"] = float(refined_display[0])
            updated_entry["refined_sim_y"] = float(refined_display[1])

    gui_manual_geometry.update_geometry_manual_peak_record_cache(
        simulation_runtime_state.peak_records,
        source_key=source_key,
        refined_caked=(float(refined_tth), float(refined_phi)),
        refined_native=(
            (
                float(updated_entry["refined_sim_native_x"]),
                float(updated_entry["refined_sim_native_y"]),
            )
            if "refined_sim_native_x" in updated_entry and "refined_sim_native_y" in updated_entry
            else None
        ),
        refined_display=(
            (
                float(updated_entry["refined_sim_x"]),
                float(updated_entry["refined_sim_y"]),
            )
            if "refined_sim_x" in updated_entry and "refined_sim_y" in updated_entry
            else None
        ),
        peak_positions=simulation_runtime_state.peak_positions,
        peak_overlay_cache=simulation_runtime_state.peak_overlay_cache,
    )
    geometry_runtime_state.manual_pick_cache_signature = None
    geometry_runtime_state.manual_pick_cache_data = {}
    return updated_entry


def _refine_current_geometry_manual_pairs() -> None:
    """Refine current saved manual-pair simulation points onto local simulation maxima."""

    background_index = int(background_runtime_state.current_background_index)
    if gui_manual_geometry.geometry_manual_pick_session_active(
        geometry_manual_state.pick_session,
        current_background_index=background_index,
    ):
        progress_label_geometry.config(
            text="Finish the active Qr/Qz placement before refining the saved simulation peaks."
        )
        return

    current_entries = [dict(entry) for entry in _geometry_manual_pairs_for_index(background_index)]
    if not current_entries:
        progress_label_geometry.config(
            text="No saved Qr/Qz placements exist for the current background image."
        )
        return

    try:
        caked_sim_image = np.asarray(
            simulation_runtime_state.last_caked_image_unscaled,
            dtype=float,
        )
        radial_axis = np.asarray(simulation_runtime_state.last_caked_radial_values, dtype=float)
        azimuth_axis = np.asarray(simulation_runtime_state.last_caked_azimuth_values, dtype=float)
    except Exception:
        progress_label_geometry.config(
            text="No caked simulation image is available. Run Update Simulation first."
        )
        return

    if (
        caked_sim_image.ndim != 2
        or caked_sim_image.size == 0
        or radial_axis.size <= 0
        or azimuth_axis.size <= 0
    ):
        progress_label_geometry.config(
            text="The current caked simulation image cannot be used for Qr/Qz refinement."
        )
        return

    stored_sim_image = getattr(simulation_runtime_state, "stored_sim_image", None)
    if stored_sim_image is not None:
        native_image_shape = tuple(int(v) for v in np.asarray(stored_sim_image).shape[:2])
    else:
        native_image_shape = (int(image_size), int(image_size))

    placement_refine_cache: dict[str, object] = {}
    try:
        placement_refine_cache = _get_geometry_manual_pick_cache(
            param_set=dict(_current_geometry_fit_params()),
            prefer_cache=True,
            background_image=_current_geometry_manual_pick_background_image(),
        )
        sim_match_cfg = dict(placement_refine_cache.get("match_config", {}))
    except Exception:
        sim_match_cfg = {}
    try:
        resolved_sim_match_cfg, sim_background_context = _auto_match_background_context(
            caked_sim_image,
            sim_match_cfg,
        )
    except Exception:
        resolved_sim_match_cfg = dict(sim_match_cfg)
        sim_background_context = None
    sim_refine_cache = {
        "match_config": dict(resolved_sim_match_cfg),
        "background_context": sim_background_context,
    }

    simulated_lookup = dict(placement_refine_cache.get("simulated_lookup", {}))
    if not simulated_lookup:
        progress_label_geometry.config(
            text="The cached Qr/Qz peak map is unavailable. Run Update Simulation first."
        )
        return

    updated_entries: list[dict[str, object]] = []
    refined_count = 0
    moved_count = 0
    skipped_count = 0
    for raw_entry in current_entries:
        entry = dict(raw_entry)
        for key in (
            "refined_sim_x",
            "refined_sim_y",
            "refined_sim_native_x",
            "refined_sim_native_y",
            "refined_sim_caked_x",
            "refined_sim_caked_y",
        ):
            entry.pop(key, None)

        try:
            source_key = (
                int(entry.get("source_table_index")),
                int(entry.get("source_row_index")),
            )
        except Exception:
            source_key = None
        source_entry = simulated_lookup.get(source_key) if source_key is not None else None
        if not isinstance(source_entry, dict):
            updated_entries.append(entry)
            skipped_count += 1
            continue

        try:
            seed_tth = float(source_entry.get("caked_x", source_entry.get("two_theta_deg", np.nan)))
            seed_phi = float(source_entry.get("caked_y", source_entry.get("phi_deg", np.nan)))
        except Exception:
            seed_tth = float("nan")
            seed_phi = float("nan")
        if not (np.isfinite(seed_tth) and np.isfinite(seed_phi)):
            try:
                seed_tth = float(
                    entry.get(
                        "refined_sim_caked_x",
                        entry.get("caked_x", entry.get("raw_caked_x", np.nan)),
                    )
                )
                seed_phi = float(
                    entry.get(
                        "refined_sim_caked_y",
                        entry.get("caked_y", entry.get("raw_caked_y", np.nan)),
                    )
                )
            except Exception:
                seed_tth = float("nan")
                seed_phi = float("nan")
        if not (np.isfinite(seed_tth) and np.isfinite(seed_phi)):
            updated_entries.append(entry)
            skipped_count += 1
            continue

        refined_tth, refined_phi = gui_manual_geometry.geometry_manual_refine_preview_point(
            dict(source_entry),
            float(seed_tth),
            float(seed_phi),
            display_background=caked_sim_image,
            cache_data=sim_refine_cache,
            use_caked_space=True,
            radial_axis=radial_axis,
            azimuth_axis=azimuth_axis,
            match_simulated_peaks_to_peak_context=match_simulated_peaks_to_peak_context,
            caked_axis_to_image_index_fn=_caked_axis_to_image_index,
            caked_image_index_to_axis_fn=_caked_image_index_to_axis,
            refine_caked_peak_center_fn=_refine_caked_peak_center,
        )
        if not (np.isfinite(refined_tth) and np.isfinite(refined_phi)):
            updated_entries.append(entry)
            skipped_count += 1
            continue

        entry["refined_sim_caked_x"] = float(refined_tth)
        entry["refined_sim_caked_y"] = float(refined_phi)
        refined_count += 1
        if (
            abs(float(refined_tth) - float(seed_tth)) > 1.0e-9
            or abs(float(refined_phi) - float(seed_phi)) > 1.0e-9
        ):
            moved_count += 1

        try:
            refined_background_display = _caked_angles_to_background_display_coords(
                float(refined_tth),
                float(refined_phi),
            )
        except Exception:
            refined_background_display = None
        try:
            native_point = (
                _background_display_to_native_detector_coords(
                    float(refined_background_display[0]),
                    float(refined_background_display[1]),
                )
                if (
                    isinstance(refined_background_display, tuple)
                    and len(refined_background_display) >= 2
                    and refined_background_display[0] is not None
                    and refined_background_display[1] is not None
                    and np.isfinite(float(refined_background_display[0]))
                    and np.isfinite(float(refined_background_display[1]))
                )
                else None
            )
        except Exception:
            native_point = None
        if (
            isinstance(native_point, tuple)
            and len(native_point) >= 2
            and np.isfinite(float(native_point[0]))
            and np.isfinite(float(native_point[1]))
        ):
            entry["refined_sim_native_x"] = float(native_point[0])
            entry["refined_sim_native_y"] = float(native_point[1])
            try:
                refined_display = _native_sim_to_display_coords(
                    float(native_point[0]),
                    float(native_point[1]),
                    native_image_shape,
                )
            except Exception:
                refined_display = None
            if (
                isinstance(refined_display, tuple)
                and len(refined_display) >= 2
                and np.isfinite(float(refined_display[0]))
                and np.isfinite(float(refined_display[1]))
            ):
                entry["refined_sim_x"] = float(refined_display[0])
                entry["refined_sim_y"] = float(refined_display[1])

        gui_manual_geometry.update_geometry_manual_peak_record_cache(
            simulation_runtime_state.peak_records,
            source_key=source_key,
            refined_caked=(float(refined_tth), float(refined_phi)),
            refined_native=(
                (
                    float(entry["refined_sim_native_x"]),
                    float(entry["refined_sim_native_y"]),
                )
                if "refined_sim_native_x" in entry and "refined_sim_native_y" in entry
                else None
            ),
            refined_display=(
                (
                    float(entry["refined_sim_x"]),
                    float(entry["refined_sim_y"]),
                )
                if "refined_sim_x" in entry and "refined_sim_y" in entry
                else None
            ),
            peak_positions=simulation_runtime_state.peak_positions,
            peak_overlay_cache=simulation_runtime_state.peak_overlay_cache,
        )
        updated_entries.append(entry)

    if refined_count <= 0:
        progress_label_geometry.config(
            text="No saved Qr/Qz placements could be mapped onto the current simulation image."
        )
        return

    _push_geometry_manual_undo_state()
    geometry_runtime_state.manual_pick_cache_signature = None
    geometry_runtime_state.manual_pick_cache_data = {}
    _set_geometry_manual_pairs_for_index(background_index, updated_entries)
    _clear_geometry_fit_dataset_cache()
    _clear_geometry_manual_preview_artists(redraw=False)
    _render_current_geometry_manual_pairs(update_status=False)
    _update_geometry_manual_pick_button_label()
    _refresh_background_status()
    progress_label_geometry.config(
        text=(
            f"Refined {refined_count} Qr/Qz simulation points on background {background_index + 1} "
            f"({moved_count} moved, {skipped_count} skipped)."
        )
    )


def _copy_geometry_fit_state_value(value):
    """Deep-copy simple geometry-fit GUI state."""

    return gui_geometry_fit.copy_geometry_fit_state_value(value)


def _geometry_fit_last_overlay_state() -> dict[str, object] | None:
    """Return the remembered geometry-fit overlay state."""

    return geometry_fit_history_state.last_overlay_state


def _geometry_fit_dataset_cache_payload() -> dict[str, object] | None:
    """Return the cached successful geometry-fit dataset bundle."""

    return geometry_fit_dataset_cache_state.payload


def _set_geometry_fit_dataset_cache(
    payload: dict[str, object] | None,
) -> dict[str, object] | None:
    """Replace the cached successful geometry-fit dataset bundle."""

    if not _retain_runtime_optional_cache(
        "geometry_fit_dataset",
        feature_needed=bool(payload),
    ):
        gui_controllers.clear_geometry_fit_dataset_cache(geometry_fit_dataset_cache_state)
        return None
    return gui_controllers.replace_geometry_fit_dataset_cache(
        geometry_fit_dataset_cache_state,
        payload,
        copy_state_value=_copy_geometry_fit_state_value,
    )


def _clear_geometry_fit_dataset_cache() -> None:
    """Discard the cached successful geometry-fit dataset bundle."""

    gui_controllers.clear_geometry_fit_dataset_cache(geometry_fit_dataset_cache_state)


def _set_geometry_fit_last_overlay_state(
    overlay_state: dict[str, object] | None,
) -> dict[str, object] | None:
    """Replace the remembered geometry-fit overlay state."""

    return gui_controllers.replace_geometry_fit_last_overlay_state(
        geometry_fit_history_state,
        overlay_state,
        copy_state_value=_copy_geometry_fit_state_value,
    )


def _current_geometry_fit_ui_params() -> dict[str, object]:
    """Capture the current geometry-fit UI parameter values."""

    return _geometry_fit_runtime_values().current_ui_params()


def _clear_geometry_fit_undo_stack() -> None:
    """Discard the geometry-fit undo/redo history."""

    gui_controllers.clear_geometry_fit_history(geometry_fit_history_state)
    _clear_geometry_fit_dataset_cache()
    _update_geometry_fit_undo_button_state()


def _capture_geometry_fit_undo_state() -> dict[str, object]:
    """Capture the current geometry-fit state for undo."""

    return gui_geometry_fit.capture_runtime_geometry_fit_undo_state(
        current_ui_params=_current_geometry_fit_ui_params,
        current_profile_cache=lambda: simulation_runtime_state.profile_cache,
        copy_state_value=_copy_geometry_fit_state_value,
        last_overlay_state=_geometry_fit_last_overlay_state,
        build_initial_pairs_display=_build_geometry_manual_initial_pairs_display,
        current_background_index=lambda: int(background_runtime_state.current_background_index),
        current_fit_params=_current_geometry_fit_params,
        pending_pairs_display=_geometry_manual_session_initial_pairs_display,
    )


def _push_geometry_fit_undo_state(state: dict[str, object] | None) -> None:
    """Push one pre-fit state onto the geometry-fit undo stack."""
    gui_controllers.push_geometry_fit_undo_state(
        geometry_fit_history_state,
        state,
        copy_state_value=_copy_geometry_fit_state_value,
        limit=int(GEOMETRY_FIT_UNDO_LIMIT),
    )
    _update_geometry_fit_undo_button_state()


def _push_geometry_fit_redo_state(state: dict[str, object] | None) -> None:
    """Push one state onto the geometry-fit redo stack."""
    gui_controllers.push_geometry_fit_redo_state(
        geometry_fit_history_state,
        state,
        copy_state_value=_copy_geometry_fit_state_value,
        limit=int(GEOMETRY_FIT_UNDO_LIMIT),
    )
    _update_geometry_fit_undo_button_state()


def _replace_runtime_geometry_fit_profile_cache(
    profile_cache: dict[str, object],
) -> None:
    """Replace the cached runtime geometry-fit profile payload."""

    simulation_runtime_state.profile_cache = dict(profile_cache)


def _mark_runtime_geometry_fit_simulation_dirty() -> None:
    """Clear the cached runtime simulation signature after geometry-fit restore."""

    _invalidate_simulation_cache()


def _cancel_runtime_geometry_fit_pending_update() -> None:
    """Cancel one queued runtime update before replaying geometry-fit state."""

    gui_controllers.clear_tk_after_token(root, simulation_runtime_state.update_pending)
    simulation_runtime_state.update_pending = None


def _draw_runtime_geometry_fit_overlay_records(
    overlay_records: list[dict[str, object]],
    marker_limit: int,
) -> None:
    """Redraw the geometry-fit overlay records for one restored runtime state."""

    _draw_geometry_fit_overlay(
        overlay_records,
        max_display_markers=marker_limit,
    )


def _draw_runtime_geometry_fit_initial_pairs_overlay(
    initial_pairs_display: list[dict[str, object]],
    marker_limit: int,
) -> None:
    """Redraw the initial-pair overlay for one restored runtime state."""

    _draw_initial_geometry_pairs_overlay(
        initial_pairs_display,
        max_display_markers=marker_limit,
    )


_restore_geometry_fit_undo_state = (
    gui_geometry_fit.build_runtime_geometry_fit_undo_restore_callback(
        var_map_factory=lambda: _geometry_fit_var_map,
        geometry_theta_offset_var_factory=lambda: geometry_theta_offset_var,
        replace_profile_cache=_replace_runtime_geometry_fit_profile_cache,
        set_last_overlay_state=_set_geometry_fit_last_overlay_state,
        request_preview_skip_once=(
            lambda: gui_controllers.request_geometry_preview_skip_once(geometry_preview_state)
        ),
        mark_last_simulation_dirty=_mark_runtime_geometry_fit_simulation_dirty,
        cancel_pending_update=_cancel_runtime_geometry_fit_pending_update,
        run_update=lambda: do_update(),
        draw_overlay_records=_draw_runtime_geometry_fit_overlay_records,
        draw_initial_pairs_overlay=_draw_runtime_geometry_fit_initial_pairs_overlay,
        refresh_status=lambda: _refresh_background_status(),
        update_manual_pick_button_label=lambda: _update_geometry_manual_pick_button_label(),
    )
)

_geometry_fit_history_callbacks = gui_geometry_fit.build_runtime_geometry_fit_history_callbacks(
    history_state=geometry_fit_history_state,
    capture_current_state=_capture_geometry_fit_undo_state,
    restore_state=_restore_geometry_fit_undo_state,
    copy_state_value=_copy_geometry_fit_state_value,
    history_limit=lambda: GEOMETRY_FIT_UNDO_LIMIT,
    peek_last_undo_state=gui_controllers.peek_last_geometry_fit_undo_state,
    peek_last_redo_state=gui_controllers.peek_last_geometry_fit_redo_state,
    commit_undo=gui_controllers.commit_geometry_fit_undo,
    commit_redo=gui_controllers.commit_geometry_fit_redo,
    update_button_state=lambda: _update_geometry_fit_undo_button_state(),
    set_progress_text=lambda text: progress_label_geometry.config(text=text),
)
_undo_last_geometry_fit_base = _geometry_fit_history_callbacks.undo
_redo_last_geometry_fit_base = _geometry_fit_history_callbacks.redo


def _undo_last_geometry_fit_with_cache_invalidation() -> bool:
    """Undo one geometry fit and clear the reusable dataset cache on success."""

    restored = bool(_undo_last_geometry_fit_base())
    if restored:
        _clear_geometry_fit_dataset_cache()
    return restored


def _redo_last_geometry_fit_with_cache_invalidation() -> bool:
    """Redo one geometry fit and clear the reusable dataset cache on success."""

    restored = bool(_redo_last_geometry_fit_base())
    if restored:
        _clear_geometry_fit_dataset_cache()
    return restored


_undo_last_geometry_fit = _undo_last_geometry_fit_with_cache_invalidation
_redo_last_geometry_fit = _redo_last_geometry_fit_with_cache_invalidation


def _geometry_manual_pair_entry_to_jsonable(
    entry: dict[str, object] | None,
) -> dict[str, object] | None:
    """Convert one saved manual pair entry into a JSON-safe dictionary."""
    return gui_manual_geometry.geometry_manual_pair_entry_to_jsonable(
        entry,
        normalize_hkl_key=_normalize_hkl_key,
        sigma_floor_px=float(GEOMETRY_MANUAL_POSITION_SIGMA_FLOOR_PX),
    )


def _geometry_manual_pair_entry_from_jsonable(
    row: dict[str, object] | None,
) -> dict[str, object] | None:
    """Rebuild one saved manual pair entry from imported JSON data."""
    return gui_manual_geometry.geometry_manual_pair_entry_from_jsonable(
        row,
        normalize_hkl_key=_normalize_hkl_key,
        sigma_floor_px=float(GEOMETRY_MANUAL_POSITION_SIGMA_FLOOR_PX),
    )


def _normalized_background_path_for_compare(raw_path: object) -> str | None:
    """Return a normalized path string suitable for background matching."""

    return gui_manual_geometry.normalized_background_path_for_compare(raw_path)


def _geometry_manual_pairs_export_rows() -> list[dict[str, object]]:
    """Return the saved manual geometry pairs as JSON-safe background rows."""

    return gui_manual_geometry.geometry_manual_pairs_export_rows(
        pairs_by_background=geometry_manual_state.pairs_by_background,
        osc_files=background_runtime_state.osc_files,
        pairs_for_index=_geometry_manual_pairs_for_index,
        pair_entry_to_jsonable=_geometry_manual_pair_entry_to_jsonable,
    )


def _collect_geometry_manual_pairs_snapshot() -> dict[str, object]:
    """Return a portable snapshot of all saved manual geometry placements."""

    return gui_manual_geometry.collect_geometry_manual_pairs_snapshot(
        osc_files=background_runtime_state.osc_files,
        current_background_index=background_runtime_state.current_background_index,
        manual_pair_rows=_geometry_manual_pairs_export_rows(),
    )


def _apply_geometry_manual_pairs_rows(
    rows: Sequence[object] | None,
    *,
    replace_existing: bool = True,
) -> tuple[int, int, list[str]]:
    """Import saved manual geometry pairs onto the currently loaded backgrounds."""

    return gui_manual_geometry.apply_geometry_manual_pairs_rows(
        rows,
        osc_files=background_runtime_state.osc_files,
        pairs_for_index=_geometry_manual_pairs_for_index,
        pair_entry_from_jsonable=_geometry_manual_pair_entry_from_jsonable,
        replace_pairs_by_background=_replace_geometry_manual_pairs_by_background,
        clear_preview_artists=_clear_geometry_manual_preview_artists,
        cancel_pick_session=_cancel_geometry_manual_pick_session,
        invalidate_pick_cache=_invalidate_geometry_manual_pick_cache,
        clear_manual_undo_stack=_clear_geometry_manual_undo_stack,
        clear_geometry_fit_undo_stack=_clear_geometry_fit_undo_stack,
        render_current_pairs=_render_current_geometry_manual_pairs,
        update_button_label=_update_geometry_manual_pick_button_label,
        refresh_status=_refresh_background_status,
        replace_existing=replace_existing,
    )


def _apply_geometry_manual_pairs_snapshot(
    snapshot: dict[str, object],
    *,
    allow_background_reload: bool = True,
) -> str:
    """Restore saved manual geometry placements from a snapshot dictionary."""

    return gui_manual_geometry.apply_geometry_manual_pairs_snapshot(
        snapshot,
        allow_background_reload=allow_background_reload,
        osc_files=background_runtime_state.osc_files,
        load_background_files=background_runtime_callbacks.load_files,
        apply_pairs_rows=_apply_geometry_manual_pairs_rows,
        schedule_update=schedule_update,
    )


def _invalidate_geometry_manual_pick_cache() -> None:
    """Drop cached manual-pick simulation/background state."""

    geometry_runtime_state.manual_pick_cache_signature = None
    geometry_runtime_state.manual_pick_cache_data = {}


def _set_geometry_manual_pick_cache_state(
    signature: object,
    cache_data: dict[str, object] | None,
) -> None:
    if _retain_runtime_optional_cache("manual_pick", feature_needed=True):
        geometry_runtime_state.manual_pick_cache_signature = signature
        geometry_runtime_state.manual_pick_cache_data = (
            dict(cache_data) if isinstance(cache_data, dict) else {}
        )
        return
    geometry_runtime_state.manual_pick_cache_signature = None
    geometry_runtime_state.manual_pick_cache_data = {}


def _geometry_manual_candidate_source_key(
    entry: dict[str, object] | None,
) -> tuple[object, ...] | None:
    """Return a stable lookup key for one manual-pick candidate or match."""
    return gui_manual_geometry.geometry_manual_candidate_source_key(
        entry,
        normalize_hkl_key=_normalize_hkl_key,
    )


def _geometry_manual_choose_group_at(
    grouped_candidates: dict[tuple[object, ...], list[dict[str, object]]] | None,
    col: float,
    row: float,
    *,
    window_size_px: float,
) -> tuple[tuple[object, ...] | None, list[dict[str, object]], float]:
    """Return the nearest clickable Qr/Qz group inside a local click window."""
    return gui_manual_geometry.geometry_manual_choose_group_at(
        grouped_candidates,
        col,
        row,
        window_size_px=window_size_px,
    )


def _geometry_manual_zoom_bounds(
    col: float,
    row: float,
    image_shape: Sequence[int] | None,
    *,
    window_size_px: float = 100.0,
) -> tuple[float, float, float, float]:
    """Return clamped image-space bounds for a square manual-pick zoom window."""
    return gui_manual_geometry.geometry_manual_zoom_bounds(
        col,
        row,
        image_shape,
        window_size_px=window_size_px,
    )


def _geometry_manual_anchor_axis_limits(
    value: float,
    span: float,
    anchor_fraction: float,
    lower_bound: float,
    upper_bound: float,
) -> tuple[float, float]:
    """Return clamped axis limits that keep *value* at a fixed screen fraction."""
    return gui_manual_geometry.geometry_manual_anchor_axis_limits(
        value,
        span,
        anchor_fraction,
        lower_bound,
        upper_bound,
    )


def _geometry_manual_group_target_count(
    group_key: tuple[object, ...] | None,
    group_entries: Sequence[dict[str, object]] | None,
) -> int:
    """Return how many manual background peaks a selected group should collect."""
    return gui_manual_geometry.geometry_manual_group_target_count(
        group_key,
        group_entries,
        normalize_hkl_key=_normalize_hkl_key,
    )


def _geometry_manual_pick_session_active(*, require_current_background: bool = True) -> bool:
    """Return whether a manual background-placement session is in progress."""
    return gui_manual_geometry.geometry_manual_pick_session_active(
        geometry_manual_state.pick_session,
        current_background_index=background_runtime_state.current_background_index,
        require_current_background=require_current_background,
    )


def _geometry_manual_unassigned_group_candidates() -> list[dict[str, object]]:
    """Return manual-pick group candidates that do not yet have a BG assignment."""
    _refresh_geometry_manual_pick_session()
    return gui_manual_geometry.geometry_manual_unassigned_group_candidates(
        geometry_manual_state.pick_session,
        current_background_index=background_runtime_state.current_background_index,
        candidate_source_key=_geometry_manual_candidate_source_key,
    )


def _geometry_manual_current_pending_candidate() -> dict[str, object] | None:
    """Return one remaining simulated peak awaiting a manual background click."""
    _refresh_geometry_manual_pick_session()
    return gui_manual_geometry.geometry_manual_current_pending_candidate(
        geometry_manual_state.pick_session,
        current_background_index=background_runtime_state.current_background_index,
        candidate_source_key=_geometry_manual_candidate_source_key,
    )


def _geometry_manual_nearest_candidate_to_point(
    col: float,
    row: float,
    candidate_entries: Sequence[dict[str, object]] | None,
) -> tuple[dict[str, object] | None, float]:
    """Return the nearest simulated candidate to one display-space point."""
    return gui_manual_geometry.geometry_manual_nearest_candidate_to_point(
        col,
        row,
        candidate_entries,
    )


def _geometry_manual_pair_entry_from_candidate(
    candidate: dict[str, object] | None,
    peak_col: float,
    peak_row: float,
    *,
    group_key: tuple[object, ...] | None,
    raw_col: float | None = None,
    raw_row: float | None = None,
    caked_col: float | None = None,
    caked_row: float | None = None,
    raw_caked_col: float | None = None,
    raw_caked_row: float | None = None,
    placement_error_px: float | None = None,
    sigma_px: float | None = None,
) -> dict[str, object] | None:
    """Build one saved manual pair entry from a candidate + measured background point."""
    return gui_manual_geometry.geometry_manual_pair_entry_from_candidate(
        candidate,
        peak_col,
        peak_row,
        group_key=group_key,
        raw_col=raw_col,
        raw_row=raw_row,
        caked_col=caked_col,
        caked_row=caked_row,
        raw_caked_col=raw_caked_col,
        raw_caked_row=raw_caked_row,
        placement_error_px=placement_error_px,
        sigma_px=sigma_px,
        normalize_hkl_key=_normalize_hkl_key,
    )


def _clear_geometry_manual_preview_artists(*, redraw: bool = True) -> None:
    """Remove manual-placement preview markers from the plot."""

    for artist in geometry_runtime_state.manual_preview_artists:
        try:
            artist.remove()
        except ValueError:
            pass
    geometry_runtime_state.manual_preview_artists.clear()
    if redraw:
        _request_overlay_canvas_redraw()


def _show_geometry_manual_preview(
    raw_col: float,
    raw_row: float,
    refined_col: float | None = None,
    refined_row: float | None = None,
    *,
    delta_px: float | None = None,
    sigma_px: float | None = None,
    preview_color: str | None = None,
) -> None:
    """Draw raw and refined manual-placement preview markers."""

    _clear_geometry_manual_preview_artists(redraw=False)
    del delta_px, sigma_px
    preview_edge = str(preview_color).strip() if preview_color is not None else ""
    if not preview_edge:
        preview_edge = "cyan"
    (raw_artist,) = ax.plot(
        [float(raw_col)],
        [float(raw_row)],
        marker="D",
        markerfacecolor="none",
        markeredgecolor="yellow",
        markersize=8,
        markeredgewidth=1.6,
        linestyle="none",
        zorder=11,
    )
    geometry_runtime_state.manual_preview_artists.append(raw_artist)

    if (
        refined_col is not None
        and refined_row is not None
        and np.isfinite(float(refined_col))
        and np.isfinite(float(refined_row))
    ):
        (refined_artist,) = ax.plot(
            [float(refined_col)],
            [float(refined_row)],
            marker="o",
            markerfacecolor=preview_edge,
            markeredgecolor=preview_edge,
            markersize=9,
            markeredgewidth=1.8,
            linestyle="none",
            alpha=0.92,
            zorder=12,
        )
        (link_artist,) = ax.plot(
            [float(raw_col), float(refined_col)],
            [float(raw_row), float(refined_row)],
            "-",
            color=preview_edge,
            linewidth=1.1,
            alpha=0.75,
            zorder=11,
        )
        geometry_runtime_state.manual_preview_artists.extend([refined_artist, link_artist])

    _request_overlay_canvas_redraw()


def _geometry_manual_preview_due(col: float, row: float) -> bool:
    """Throttle manual-placement preview updates during mouse motion."""
    return gui_manual_geometry.geometry_manual_preview_due(
        col,
        row,
        pick_session=geometry_manual_state.pick_session,
        current_background_index=background_runtime_state.current_background_index,
        min_interval_s=float(GEOMETRY_MANUAL_PREVIEW_MIN_INTERVAL_S),
        min_move_px=float(GEOMETRY_MANUAL_PREVIEW_MIN_MOVE_PX),
        perf_counter_fn=perf_counter,
    )


def _geometry_manual_refine_preview_point(
    candidate: dict[str, object] | None,
    raw_col: float,
    raw_row: float,
    *,
    display_background: np.ndarray | None = None,
    cache_data: dict[str, object] | None = None,
) -> tuple[float, float]:
    """Refine one manual raw click/release position to the best background peak."""
    background_local = (
        _current_geometry_manual_pick_background_image()
        if display_background is None
        else display_background
    )
    use_caked_space = _geometry_manual_pick_uses_caked_space()
    if not use_caked_space:
        try:
            caked_background = np.asarray(
                simulation_runtime_state.last_caked_background_image_unscaled,
                dtype=float,
            )
            radial_axis = np.asarray(
                simulation_runtime_state.last_caked_radial_values,
                dtype=float,
            )
            azimuth_axis = np.asarray(
                simulation_runtime_state.last_caked_azimuth_values,
                dtype=float,
            )
        except Exception:
            caked_background = None
            radial_axis = np.array([], dtype=float)
            azimuth_axis = np.array([], dtype=float)
        if (
            isinstance(caked_background, np.ndarray)
            and caked_background.ndim == 2
            and caked_background.size > 0
            and radial_axis.size > 0
            and azimuth_axis.size > 0
        ):
            detector_to_caked = None
            try:
                native_point = _background_display_to_native_detector_coords(
                    float(raw_col),
                    float(raw_row),
                )
                if (
                    isinstance(native_point, tuple)
                    and len(native_point) >= 2
                    and np.isfinite(float(native_point[0]))
                    and np.isfinite(float(native_point[1]))
                ):
                    detector_to_caked = lambda col, row: (
                        _native_detector_coords_to_caked_display_coords(
                            *(
                                _background_display_to_native_detector_coords(
                                    float(col),
                                    float(row),
                                )
                                or (float("nan"), float("nan"))
                            )
                        )
                    )
            except Exception:
                detector_to_caked = None
            if callable(detector_to_caked):
                match_cfg = (
                    dict(cache_data.get("match_config", {}))
                    if isinstance(cache_data, dict)
                    else {}
                )
                try:
                    resolved_match_cfg, caked_background_context = (
                        _auto_match_background_context(
                            caked_background,
                            match_cfg,
                        )
                    )
                except Exception:
                    resolved_match_cfg = dict(match_cfg)
                    caked_background_context = None
                refined_detector_pick = (
                    gui_manual_geometry.refine_detector_pick_via_caked_background(
                        candidate,
                        float(raw_col),
                        float(raw_row),
                        detector_display_to_caked_coords=detector_to_caked,
                        caked_angles_to_detector_display_coords=(
                            _caked_angles_to_background_display_coords
                        ),
                        caked_background=caked_background,
                        radial_axis=radial_axis,
                        azimuth_axis=azimuth_axis,
                        cache_data={
                            "match_config": dict(resolved_match_cfg),
                            "background_context": caked_background_context,
                        },
                        match_simulated_peaks_to_peak_context=(
                            match_simulated_peaks_to_peak_context
                        ),
                        caked_axis_to_image_index_fn=_caked_axis_to_image_index,
                        caked_image_index_to_axis_fn=_caked_image_index_to_axis,
                        refine_caked_peak_center_fn=_refine_caked_peak_center,
                    )
                )
                if isinstance(refined_detector_pick, Mapping):
                    try:
                        refined_col = float(
                            refined_detector_pick["refined_display_col"]
                        )
                        refined_row = float(
                            refined_detector_pick["refined_display_row"]
                        )
                    except Exception:
                        refined_col = float("nan")
                        refined_row = float("nan")
                    if np.isfinite(refined_col) and np.isfinite(refined_row):
                        return float(refined_col), float(refined_row)
    return gui_manual_geometry.geometry_manual_refine_preview_point(
        candidate,
        raw_col,
        raw_row,
        display_background=background_local,
        cache_data=cache_data,
        build_cache_data=lambda: _get_geometry_manual_pick_cache(
            param_set=_current_geometry_fit_params(),
            prefer_cache=True,
            background_image=background_local,
        ),
        use_caked_space=use_caked_space,
        radial_axis=np.asarray(simulation_runtime_state.last_caked_radial_values, dtype=float),
        azimuth_axis=np.asarray(simulation_runtime_state.last_caked_azimuth_values, dtype=float),
        match_simulated_peaks_to_peak_context=match_simulated_peaks_to_peak_context,
        peak_maximum_near_in_image_fn=_peak_maximum_near_in_image,
        caked_axis_to_image_index_fn=_caked_axis_to_image_index,
        caked_image_index_to_axis_fn=_caked_image_index_to_axis,
        refine_caked_peak_center_fn=_refine_caked_peak_center,
    )


def _restore_geometry_manual_pick_view(*, redraw: bool = True) -> None:
    """Restore the pre-zoom axis view for manual background placement."""
    gui_manual_geometry.restore_geometry_manual_pick_view(
        geometry_manual_state.pick_session,
        axis=ax,
        canvas=canvas,
        redraw=redraw,
    )


def _apply_geometry_manual_pick_zoom(
    col: float,
    row: float,
    *,
    anchor_fraction_x: float = 0.5,
    anchor_fraction_y: float = 0.5,
) -> None:
    """Zoom to a fixed local window while the user is placing manual points."""
    gui_manual_geometry.apply_geometry_manual_pick_zoom(
        geometry_manual_state.pick_session,
        col,
        row,
        display_background=_current_geometry_manual_pick_background_image(),
        axis=ax,
        canvas=canvas,
        use_caked_space=_geometry_manual_pick_uses_caked_space(),
        last_caked_extent=simulation_runtime_state.last_caked_extent,
        caked_zoom_tth_deg=float(GEOMETRY_MANUAL_CAKED_ZOOM_TTH_DEG),
        caked_zoom_phi_deg=float(GEOMETRY_MANUAL_CAKED_ZOOM_PHI_DEG),
        pick_zoom_window_px=float(GEOMETRY_MANUAL_PICK_ZOOM_WINDOW_PX),
        anchor_fraction_x=anchor_fraction_x,
        anchor_fraction_y=anchor_fraction_y,
        anchor_axis_limits_fn=_geometry_manual_anchor_axis_limits,
    )


def _refresh_geometry_manual_pick_session() -> dict[str, object]:
    """Refresh the active manual Qr/Qz pick session against the latest simulation."""

    current_session = geometry_manual_state.pick_session
    if not gui_manual_geometry.geometry_manual_pick_session_active(
        current_session,
        current_background_index=background_runtime_state.current_background_index,
    ):
        return current_session

    background_image = _current_geometry_manual_pick_background_image()
    if background_image is None:
        return current_session

    try:
        cache_data = _get_geometry_manual_pick_cache(
            param_set=dict(_current_geometry_fit_params()),
            prefer_cache=True,
            background_index=int(background_runtime_state.current_background_index),
            background_image=background_image,
        )
    except Exception:
        return current_session
    if not isinstance(cache_data, dict):
        return current_session

    refreshed_session = gui_manual_geometry.refresh_geometry_manual_pick_session_candidates(
        current_session,
        grouped_candidates=cache_data.get("grouped_candidates"),
        cache_signature=cache_data.get("signature"),
        candidate_source_key=_geometry_manual_candidate_source_key,
    )
    if refreshed_session != current_session:
        return _set_geometry_manual_pick_session(refreshed_session)
    return current_session


def _geometry_manual_session_initial_pairs_display() -> list[dict[str, object]]:
    """Return overlay-ready display entries for the in-progress manual pick session."""
    _refresh_geometry_manual_pick_session()
    return gui_manual_geometry.geometry_manual_session_initial_pairs_display(
        geometry_manual_state.pick_session,
        current_background_index=background_runtime_state.current_background_index,
        candidate_source_key=_geometry_manual_candidate_source_key,
        entry_display_coords=_geometry_manual_entry_display_coords,
    )


def _match_geometry_manual_group_to_background(
    candidate_entries: Sequence[dict[str, object]] | None,
    *,
    background_image: np.ndarray | None = None,
    cache_data: dict[str, object] | None = None,
) -> dict[tuple[object, ...], tuple[float, float]]:
    """Return refined measured peak centers for one clicked symmetric Qr/Qz group."""
    background_local = (
        _get_current_background_display() if background_image is None else background_image
    )
    return gui_manual_geometry.match_geometry_manual_group_to_background(
        candidate_entries,
        background_image=background_local,
        cache_data=cache_data,
        build_cache_data=lambda: _get_geometry_manual_pick_cache(
            prefer_cache=True,
            background_image=background_local,
        ),
        auto_match_background_context=_auto_match_background_context,
        match_simulated_peaks_to_peak_context=match_simulated_peaks_to_peak_context,
        candidate_source_key=_geometry_manual_candidate_source_key,
    )


def _ensure_geometry_fit_caked_view(*, force_refresh: bool = False) -> None:
    """Switch geometry fitting/import into the 2D caked integration view now."""

    simulation_runtime_state.update_pending, integration_update_pending = (
        gui_manual_geometry.ensure_geometry_fit_caked_view(
            show_caked_2d_var=analysis_view_controls_view_state.show_caked_2d_var,
            pick_uses_caked_space=_geometry_manual_pick_uses_caked_space,
            toggle_caked_2d=toggle_caked_2d,
            do_update=do_update,
            schedule_update=schedule_update,
            root=root,
            update_pending=simulation_runtime_state.update_pending,
            integration_update_pending=simulation_runtime_state.integration_update_pending,
            update_running=bool(
                simulation_runtime_state.update_running
                or simulation_runtime_state.worker_active_job is not None
                or simulation_runtime_state.worker_queued_job is not None
            ),
            force_refresh=force_refresh,
        )
    )


def _geometry_fit_caked_view_for_index(
    index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return active caked background image plus axes for one background index."""

    if int(index) != int(background_runtime_state.current_background_index):
        return None
    try:
        caked_background = np.asarray(
            simulation_runtime_state.last_caked_background_image_unscaled,
            dtype=np.float64,
        )
        radial_axis = np.asarray(
            simulation_runtime_state.last_caked_radial_values,
            dtype=np.float64,
        )
        azimuth_axis = np.asarray(
            simulation_runtime_state.last_caked_azimuth_values,
            dtype=np.float64,
        )
    except Exception:
        return None
    if (
        caked_background.ndim != 2
        or caked_background.size <= 0
        or radial_axis.size <= 0
        or azimuth_axis.size <= 0
    ):
        return None
    return (
        np.asarray(caked_background, dtype=np.float64).copy(),
        np.asarray(radial_axis, dtype=np.float64).copy(),
        np.asarray(azimuth_axis, dtype=np.float64).copy(),
    )


def _current_geometry_fit_caked_roi_enabled() -> bool:
    """Return whether geometry-fit branch-restricted caking is enabled."""

    try:
        return bool(geometry_fit_caked_roi_enabled_var.get())
    except Exception:
        return True


def _current_geometry_fit_caked_roi_preview_enabled() -> bool:
    """Return whether the detector/caked ROI preview is enabled."""

    try:
        return bool(geometry_fit_caked_roi_preview_var.get())
    except Exception:
        return False


def _current_geometry_fit_caked_roi_selection(
    *,
    force_enabled: bool = False,
) -> Mapping[str, object] | None:
    """Return the current selected-branch detector ROI for geometry-fit preview/use."""

    background_index = int(background_runtime_state.current_background_index)
    required_pairs = [
        dict(entry)
        for entry in (_geometry_manual_pairs_for_index(int(background_index)) or ())
        if isinstance(entry, Mapping)
    ]
    if not required_pairs:
        return None

    native_background = _get_current_background_native()
    if native_background is None:
        return None
    native_shape = tuple(int(v) for v in np.asarray(native_background).shape[:2])
    if len(native_shape) < 2 or native_shape[0] <= 0 or native_shape[1] <= 0:
        return None

    snapshot = dict(simulation_runtime_state.source_row_snapshots.get(int(background_index)) or {})
    source_rows = [
        dict(entry)
        for entry in (snapshot.get("rows") or ())
        if isinstance(entry, Mapping)
    ]
    if not source_rows:
        source_rows = [
            dict(entry)
            for entry in (_build_live_preview_simulated_peaks_from_cache() or ())
            if isinstance(entry, Mapping)
        ]
    if not source_rows:
        return None

    selection = gui_geometry_fit.build_geometry_fit_caked_roi_selection(
        source_rows,
        required_pairs=required_pairs,
        image_shape=native_shape,
        fit_config=fit_config,
        enabled_override=(
            True if force_enabled else _current_geometry_fit_caked_roi_enabled()
        ),
        fit_space_to_detector_point=_geometry_fit_caked_roi_fit_space_to_detector_point(
            _current_geometry_fit_params()
        ),
    )
    return selection if isinstance(selection, Mapping) else None


def _rotate_native_detector_pixels_for_display(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    native_shape: tuple[int, int],
    k: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Rotate native detector pixel indices into display-space indices."""

    disp_rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    disp_cols = np.asarray(cols, dtype=np.int64).reshape(-1)
    height = int(native_shape[0])
    width = int(native_shape[1])
    for _ in range(int(k) % 4):
        disp_rows, disp_cols, height, width = (
            (int(width) - 1) - disp_cols,
            disp_rows,
            int(width),
            int(height),
        )
    return disp_rows, disp_cols, (int(height), int(width))


def _mask_detector_display_image(
    image: np.ndarray | None,
    *,
    selection: Mapping[str, object] | None,
    native_shape: tuple[int, int],
) -> np.ndarray | None:
    """Mask one detector-space display image down to the selected branch ROI."""

    if image is None or not isinstance(selection, Mapping):
        return image
    if not bool(selection.get("valid", False)):
        return image
    try:
        rows = np.asarray(selection.get("rows"), dtype=np.int64).reshape(-1)
        cols = np.asarray(selection.get("cols"), dtype=np.int64).reshape(-1)
    except Exception:
        return image
    if rows.size <= 0 or cols.size != rows.size:
        return image

    display_image = np.asarray(image)
    if display_image.ndim != 2 or display_image.size <= 0:
        return image

    disp_rows, disp_cols, display_shape = _rotate_native_detector_pixels_for_display(
        rows,
        cols,
        native_shape=native_shape,
        k=DISPLAY_ROTATE_K,
    )
    if tuple(int(v) for v in display_image.shape[:2]) != tuple(display_shape):
        return image

    valid = (
        (disp_rows >= 0)
        & (disp_rows < int(display_shape[0]))
        & (disp_cols >= 0)
        & (disp_cols < int(display_shape[1]))
    )
    if not np.any(valid):
        return image

    masked = np.zeros_like(display_image)
    masked[disp_rows[valid], disp_cols[valid]] = display_image[disp_rows[valid], disp_cols[valid]]
    return masked


def _current_geometry_fit_preview_integrator() -> FastAzimuthalIntegrator | None:
    """Build the live integrator used to project detector ROI pixels into caked space."""

    try:
        return _build_analysis_integrator(
            {
                "center": np.asarray(
                    [float(center_x_var.get()), float(center_y_var.get())],
                    dtype=np.float64,
                ),
                "distance_m": float(corto_detector_var.get()),
                "pixel_size_m": float(pixel_size_m),
                "wavelength_m": float(wave_m),
            }
        )
    except Exception:
        return None


def _nearest_sorted_axis_indices(
    axis_values: Sequence[float] | None,
    sample_values: np.ndarray,
) -> np.ndarray | None:
    """Return nearest-bin indices on one sorted axis for many sample values."""

    try:
        axis = np.asarray(axis_values, dtype=float).reshape(-1)
        samples = np.asarray(sample_values, dtype=float).reshape(-1)
    except Exception:
        return None
    if axis.size <= 0 or samples.size <= 0:
        return None
    if axis.size == 1:
        return np.zeros(samples.shape, dtype=np.int64)

    insertion = np.searchsorted(axis, samples, side="left")
    insertion = np.clip(insertion, 1, axis.size - 1)
    left = axis[insertion - 1]
    right = axis[insertion]
    choose_left = np.abs(samples - left) <= np.abs(right - samples)
    return np.where(choose_left, insertion - 1, insertion).astype(np.int64, copy=False)


def _geometry_fit_caked_roi_preview_mask(
    *,
    selection: Mapping[str, object] | None,
    native_shape: tuple[int, int],
    radial_axis: Sequence[float] | None,
    azimuth_axis: Sequence[float] | None,
) -> np.ndarray | None:
    """Return a caked-bin mask corresponding to the current detector ROI preview."""

    if not isinstance(selection, Mapping) or not bool(selection.get("valid", False)):
        return None
    ai = _current_geometry_fit_preview_integrator()
    two_theta_map, phi_map = _detector_angular_maps_for_shape(ai, native_shape)
    if two_theta_map is None or phi_map is None:
        return None

    try:
        rows = np.asarray(selection.get("rows"), dtype=np.int64).reshape(-1)
        cols = np.asarray(selection.get("cols"), dtype=np.int64).reshape(-1)
    except Exception:
        return None
    if rows.size <= 0 or cols.size != rows.size:
        return None

    radial = np.asarray(radial_axis, dtype=float).reshape(-1)
    azimuth = np.asarray(azimuth_axis, dtype=float).reshape(-1)
    if radial.size <= 0 or azimuth.size <= 0:
        return None

    valid = (
        (rows >= 0)
        & (rows < int(two_theta_map.shape[0]))
        & (cols >= 0)
        & (cols < int(two_theta_map.shape[1]))
    )
    if not np.any(valid):
        return None

    two_theta_values = np.asarray(two_theta_map[rows[valid], cols[valid]], dtype=float)
    phi_values = _wrap_phi_range(
        np.asarray(phi_map[rows[valid], cols[valid]], dtype=float)
    )
    finite = np.isfinite(two_theta_values) & np.isfinite(phi_values)
    if not np.any(finite):
        return None

    radial_idx = _nearest_sorted_axis_indices(radial, two_theta_values[finite])
    azimuth_idx = _nearest_sorted_axis_indices(azimuth, phi_values[finite])
    if radial_idx is None or azimuth_idx is None:
        return None

    mask = np.zeros((int(azimuth.size), int(radial.size)), dtype=bool)
    mask[azimuth_idx, radial_idx] = True
    return mask


def _mask_caked_display_image(
    image: np.ndarray | None,
    *,
    caked_mask: np.ndarray | None,
) -> np.ndarray | None:
    """Mask one caked image down to the previewed branch ROI."""

    if image is None or caked_mask is None:
        return image
    caked_image = np.asarray(image)
    if caked_image.ndim != 2 or tuple(caked_image.shape) != tuple(caked_mask.shape):
        return image
    masked = np.zeros_like(caked_image)
    masked[caked_mask] = caked_image[caked_mask]
    return masked


def _geometry_fit_caked_roi_preview_display_sources(
    *,
    show_caked_image: bool,
    simulation_image: np.ndarray | None,
    background_image: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Resolve the primary/secondary rasters used while ROI preview is enabled."""

    preview_sim_image, preview_background_image = _geometry_fit_caked_roi_preview_images(
        show_caked_image=bool(show_caked_image),
        simulation_image=simulation_image,
        background_image=background_image,
    )
    primary_source = (
        preview_sim_image if isinstance(preview_sim_image, np.ndarray) else simulation_image
    )
    secondary_source = (
        preview_background_image
        if isinstance(preview_background_image, np.ndarray)
        else background_image
    )
    if (
        _current_geometry_fit_caked_roi_preview_enabled()
        and isinstance(preview_background_image, np.ndarray)
    ):
        return np.asarray(preview_background_image), None
    return primary_source, secondary_source


def _hide_geometry_fit_caked_roi_preview_aux_artists() -> None:
    """Hide non-raster artists so ROI preview shows only the selected regions."""

    _clear_all_geometry_overlay_artists(redraw=False)
    try:
        gui_qr_cylinder_overlay.clear_runtime_qr_cylinder_overlay_artists(
            qr_cylinder_overlay_runtime_bindings_factory(),
            redraw=False,
        )
    except Exception:
        pass
    gui_controllers.clear_geometry_preview_skip_once(geometry_preview_state)
    for artist_name in ("center_marker", "selected_peak_marker", "integration_region_overlay"):
        artist = globals().get(artist_name)
        setter = getattr(artist, "set_visible", None)
        if callable(setter):
            try:
                setter(False)
            except Exception:
                continue


def _geometry_fit_caked_roi_preview_images(
    *,
    show_caked_image: bool,
    simulation_image: np.ndarray | None,
    background_image: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return detector/caked display images masked to the selected branch ROI."""

    if not _current_geometry_fit_caked_roi_preview_enabled():
        return simulation_image, background_image

    selection = _current_geometry_fit_caked_roi_selection(force_enabled=True)
    if not isinstance(selection, Mapping) or not bool(selection.get("valid", False)):
        return simulation_image, background_image

    native_background = _get_current_background_native()
    if native_background is None:
        return simulation_image, background_image
    native_shape = tuple(int(v) for v in np.asarray(native_background).shape[:2])
    if len(native_shape) < 2 or native_shape[0] <= 0 or native_shape[1] <= 0:
        return simulation_image, background_image

    if not show_caked_image:
        return (
            _mask_detector_display_image(
                simulation_image,
                selection=selection,
                native_shape=native_shape,
            ),
            _mask_detector_display_image(
                background_image,
                selection=selection,
                native_shape=native_shape,
            ),
        )

    caked_mask = _geometry_fit_caked_roi_preview_mask(
        selection=selection,
        native_shape=native_shape,
        radial_axis=simulation_runtime_state.last_caked_radial_values,
        azimuth_axis=simulation_runtime_state.last_caked_azimuth_values,
    )
    if caked_mask is None:
        return simulation_image, background_image
    return (
        _mask_caked_display_image(simulation_image, caked_mask=caked_mask),
        _mask_caked_display_image(background_image, caked_mask=caked_mask),
    )


def _peak_maximum_near_in_image(
    image: np.ndarray | None,
    col: float,
    row: float,
    *,
    search_radius: int = 5,
) -> tuple[float, float]:
    """Return the brightest local pixel near ``(col, row)`` in display coordinates."""
    return gui_manual_geometry.peak_maximum_near_in_image(
        image,
        col,
        row,
        search_radius=search_radius,
    )


def _detector_pixel_to_scattering_angles(
    col: float,
    row: float,
    center: Sequence[float] | None,
    detector_distance: float,
    pixel_size: float,
) -> tuple[float | None, float | None]:
    """Convert one detector pixel position to ``(2theta, phi)`` in degrees."""

    if center is None or len(center) < 2:
        return None, None
    if not (np.isfinite(detector_distance) and detector_distance > 0.0):
        return None, None
    if not (np.isfinite(pixel_size) and pixel_size > 0.0):
        return None, None
    try:
        center_row = float(center[0])
        center_col = float(center[1])
    except Exception:
        return None, None
    if not (
        np.isfinite(col)
        and np.isfinite(row)
        and np.isfinite(center_row)
        and np.isfinite(center_col)
    ):
        return None, None

    dx = (float(col) - center_col) * float(pixel_size)
    dy = (center_row - float(row)) * float(pixel_size)
    radius = float(np.hypot(dx, dy))
    two_theta = float(np.degrees(np.arctan2(radius, float(detector_distance))))
    phi = float(np.degrees(np.arctan2(dx, dy)))
    return two_theta, _wrap_phi_range(phi)


def _native_detector_coords_to_live_caked_coords(
    col: float,
    row: float,
) -> tuple[float, float] | None:
    return _native_detector_coords_to_caked_display_coords(
        col,
        row,
    )


def _scattering_angles_to_detector_pixel(
    two_theta_deg: float,
    phi_deg: float,
    center: Sequence[float] | None,
    detector_distance: float,
    pixel_size: float,
) -> tuple[float | None, float | None]:
    """Convert ``(2theta, phi)`` in degrees to one detector pixel position."""

    if center is None or len(center) < 2:
        return None, None
    if not (np.isfinite(detector_distance) and detector_distance > 0.0):
        return None, None
    if not (np.isfinite(pixel_size) and pixel_size > 0.0):
        return None, None
    try:
        center_row = float(center[0])
        center_col = float(center[1])
    except Exception:
        return None, None
    if not (
        np.isfinite(two_theta_deg)
        and np.isfinite(phi_deg)
        and np.isfinite(center_row)
        and np.isfinite(center_col)
    ):
        return None, None

    radius = float(detector_distance) * float(np.tan(np.deg2rad(float(two_theta_deg))))
    phi_rad = float(np.deg2rad(float(phi_deg)))
    dx = radius * float(np.sin(phi_rad))
    dy = radius * float(np.cos(phi_rad))
    col = center_col + dx / float(pixel_size)
    row = center_row - dy / float(pixel_size)
    return float(col), float(row)


def _geometry_fit_caked_roi_fit_space_to_detector_point(
    params_local: Mapping[str, object] | None,
) -> Callable[[float, float], tuple[float | None, float | None] | None] | None:
    """Build one fit-space `(2theta, phi)` to detector-point projector for ROI use."""

    if not isinstance(params_local, Mapping):
        return None
    center_value = params_local.get("center")
    try:
        detector_distance = float(params_local.get("corto_detector", np.nan))
    except Exception:
        detector_distance = float("nan")
    try:
        pixel_size = float(
            params_local.get(
                "pixel_size_m",
                params_local.get("pixel_size", np.nan),
            )
        )
    except Exception:
        pixel_size = float("nan")
    if center_value is None or not np.isfinite(detector_distance) or not np.isfinite(
        pixel_size
    ):
        return None

    def _project(
        two_theta_deg: float,
        phi_deg: float,
    ) -> tuple[float | None, float | None] | None:
        return _scattering_angles_to_detector_pixel(
            float(two_theta_deg),
            float(phi_deg),
            center=center_value,
            detector_distance=float(detector_distance),
            pixel_size=float(pixel_size),
        )

    return _project


def _caked_axis_to_image_index(
    value: float,
    axis_values: Sequence[float] | None,
) -> float:
    """Map one caked-axis coordinate in degrees to a floating image index."""
    return gui_manual_geometry.caked_axis_to_image_index(value, axis_values)


def _caked_image_index_to_axis(
    index_value: float,
    axis_values: Sequence[float] | None,
) -> float:
    """Map one floating caked image index back to axis-space degrees."""
    return gui_manual_geometry.caked_image_index_to_axis(index_value, axis_values)


def _refine_profile_peak_index(
    profile: Sequence[float] | None,
    seed_index: float,
) -> float:
    """Return one subpixel 1D peak center focused on the top of a local profile."""
    return gui_manual_geometry.refine_profile_peak_index(profile, seed_index)


def _refine_caked_peak_center(
    image: np.ndarray | None,
    radial_axis: Sequence[float] | None,
    azimuth_axis: Sequence[float] | None,
    two_theta_deg: float,
    phi_deg: float,
    *,
    tth_window_deg: float | None = None,
    phi_window_deg: float | None = None,
) -> tuple[float, float]:
    """Refine one caked click to the crest of the local 2theta/phi ridge."""
    return gui_manual_geometry.refine_caked_peak_center(
        image,
        radial_axis,
        azimuth_axis,
        two_theta_deg,
        phi_deg,
        tth_window_deg=tth_window_deg,
        phi_window_deg=phi_window_deg,
        default_tth_window_deg=float(GEOMETRY_MANUAL_CAKED_SEARCH_TTH_DEG),
        default_phi_window_deg=float(GEOMETRY_MANUAL_CAKED_SEARCH_PHI_DEG),
    )


geometry_manual_projection_workflow = (
    gui_runtime_geometry_interaction.build_runtime_geometry_manual_projection_workflow(
        bootstrap_module=gui_bootstrap,
        manual_geometry_module=gui_manual_geometry,
        caked_view_enabled=lambda: (
            bool(analysis_view_controls_view_state.show_caked_2d_var.get())
            if analysis_view_controls_view_state.show_caked_2d_var is not None
            else False
        ),
        last_caked_background_image_unscaled=(
            lambda: simulation_runtime_state.last_caked_background_image_unscaled
        ),
        last_caked_radial_values=(lambda: simulation_runtime_state.last_caked_radial_values),
        last_caked_azimuth_values=(lambda: simulation_runtime_state.last_caked_azimuth_values),
        current_background_display=_get_current_background_display,
        current_background_native=_get_current_background_native,
        ai=lambda: simulation_runtime_state.ai_cache.get("ai"),
        center=lambda: [float(center_x_var.get()), float(center_y_var.get())],
        detector_distance=lambda: float(corto_detector_var.get()),
        pixel_size=float(pixel_size_m),
        wrap_phi_range=lambda value: globals()["_wrap_phi_range"](value),
        rotate_point_for_display=_rotate_point_for_display,
        display_rotate_k=DISPLAY_ROTATE_K,
        current_geometry_fit_params=lambda: globals()["_current_geometry_fit_params"](),
        build_live_preview_simulated_peaks_from_cache=(
            lambda: globals()["_build_live_preview_simulated_peaks_from_cache"]()
        ),
        ensure_peak_overlay_data=(
            lambda force=False: (
                globals()["ensure_peak_overlay_data"](force=force)
                if callable(globals().get("ensure_peak_overlay_data"))
                else False
            )
        ),
        miller=lambda: miller,
        intensities=lambda: intensities,
        image_size=int(image_size),
        display_to_native_sim_coords=_display_to_native_sim_coords,
        get_detector_angular_maps=(
            lambda ai_value: globals()["_get_detector_angular_maps"](ai_value)
        ),
        detector_pixel_to_scattering_angles=_detector_pixel_to_scattering_angles,
        backend_detector_coords_to_native_detector_coords=(
            _backend_background_to_native_detector_coords
        ),
        scattering_angles_to_detector_pixel=_scattering_angles_to_detector_pixel,
        filter_simulated_peaks=(
            lambda *args, **kwargs: globals()["_filter_geometry_fit_simulated_peaks"](
                *args,
                **kwargs,
            )
        ),
        collapse_simulated_peaks=(
            lambda *args, **kwargs: globals()["_collapse_geometry_fit_simulated_peaks"](
                *args,
                **kwargs,
            )
        ),
    )
)
geometry_manual_projection_runtime = geometry_manual_projection_workflow.runtime
geometry_manual_projection_runtime_callbacks = geometry_manual_projection_workflow.callbacks
_geometry_manual_pick_uses_caked_space = geometry_manual_projection_workflow.pick_uses_caked_space
_current_geometry_manual_pick_background_image = (
    geometry_manual_projection_workflow.current_background_image
)
_geometry_manual_entry_display_coords = geometry_manual_projection_workflow.entry_display_coords
_caked_angles_to_background_display_coords = (
    geometry_manual_projection_workflow.caked_angles_to_background_display_coords
)
_background_display_to_native_detector_coords = (
    geometry_manual_projection_workflow.background_display_to_native_detector_coords
)
_native_detector_coords_to_caked_display_coords = (
    geometry_manual_projection_workflow.native_detector_coords_to_caked_display_coords
)
_project_geometry_manual_peaks_to_current_view = (
    geometry_manual_projection_workflow.project_peaks_to_current_view
)
_geometry_manual_simulated_peaks_for_params = (
    geometry_manual_projection_workflow.simulated_peaks_for_params
)
_geometry_manual_last_simulation_diagnostics = (
    geometry_manual_projection_workflow.last_simulation_diagnostics
)
_geometry_manual_pick_candidates = geometry_manual_projection_workflow.pick_candidates
_geometry_manual_simulated_lookup = geometry_manual_projection_workflow.simulated_lookup


geometry_manual_cache_workflow = (
    gui_runtime_geometry_interaction.build_runtime_geometry_manual_cache_workflow(
        bootstrap_module=gui_bootstrap,
        manual_geometry_module=gui_manual_geometry,
        fit_config=fit_config,
        last_simulation_signature=(lambda: simulation_runtime_state.last_simulation_signature),
        current_background_index=(lambda: int(background_runtime_state.current_background_index)),
        current_background_image=_current_geometry_manual_pick_background_image,
        use_caked_space=_geometry_manual_pick_uses_caked_space,
        geometry_preview_excluded_q_groups=(lambda: geometry_preview_state.excluded_q_groups),
        geometry_q_group_cached_entries=(lambda: geometry_q_group_state.cached_entries),
        stored_max_positions_local=(lambda: simulation_runtime_state.stored_max_positions_local),
        stored_peak_table_lattice=(lambda: simulation_runtime_state.stored_peak_table_lattice),
        peak_records=(lambda: simulation_runtime_state.peak_records),
        current_cache_signature=(lambda: geometry_runtime_state.manual_pick_cache_signature),
        current_cache_data=(lambda: geometry_runtime_state.manual_pick_cache_data),
        replace_cache_state=_set_geometry_manual_pick_cache_state,
        current_geometry_fit_params=lambda: globals()["_current_geometry_fit_params"](),
        pairs_for_index=_geometry_manual_pairs_for_index,
        source_rows_for_background=(
            lambda background_index, param_set=None, consumer=None: globals()[
                "_geometry_manual_source_rows_for_background"
            ](
                background_index,
                param_set,
                consumer=consumer,
            )
        ),
        simulated_peaks_for_params=_geometry_manual_simulated_peaks_for_params,
        source_snapshot_signature_for_background=(
            lambda background_index, param_set=None: globals()[
                "_geometry_source_snapshot_signature_for_background"
            ](
                background_index,
                param_set,
            )
        ),
        build_grouped_candidates=_geometry_manual_pick_candidates,
        build_simulated_lookup=_geometry_manual_simulated_lookup,
        entry_display_coords=_geometry_manual_entry_display_coords,
        auto_match_background_context=(
            lambda *args, **kwargs: globals()["_auto_match_background_context"](
                *args,
                **kwargs,
            )
        ),
    )
)
geometry_manual_cache_runtime = geometry_manual_cache_workflow.runtime
geometry_manual_cache_runtime_callbacks = geometry_manual_cache_workflow.callbacks
_current_geometry_manual_match_config = geometry_manual_cache_workflow.current_match_config
_geometry_manual_pick_cache_signature = geometry_manual_cache_workflow.pick_cache_signature
_get_geometry_manual_pick_cache = geometry_manual_cache_workflow.get_pick_cache
_build_geometry_manual_initial_pairs_display = (
    geometry_manual_cache_workflow.build_initial_pairs_display
)


def _clear_geometry_pick_artists(*, redraw: bool = True):
    """Remove geometry fit markers from the plot and reset the cache."""

    gui_overlays.clear_artists(
        geometry_runtime_state.pick_artists,
        draw_idle=_request_overlay_canvas_redraw,
        redraw=redraw,
    )


def _clear_geometry_preview_artists(*, redraw: bool = True):
    """Remove live geometry preview markers from the plot and reset the cache."""

    gui_overlays.clear_artists(
        geometry_runtime_state.preview_artists,
        draw_idle=_request_overlay_canvas_redraw,
        redraw=redraw,
    )


qr_cylinder_overlay_workflow = (
    gui_runtime_qr_cylinder_overlay.build_runtime_qr_cylinder_overlay_workflow(
        bragg_qr_manager_module=gui_bragg_qr_manager,
        qr_cylinder_overlay_module=gui_qr_cylinder_overlay,
        bootstrap_module=gui_bootstrap,
        active_entry_factory_kwargs={
            "simulation_runtime_state": simulation_runtime_state,
            "primary_candidate": (lambda: a_var.get()),
            "primary_fallback": float(av),
            "secondary_candidate": (lambda: av2),
            "primary_miller_all": (lambda: globals().get("SIM_MILLER1")),
            "secondary_miller_all": (lambda: globals().get("SIM_MILLER2")),
        },
        render_config_factory_kwargs={
            "render_in_caked_space_factory": (
                lambda: bool(analysis_view_controls_view_state.show_caked_2d_var.get())
            ),
            "image_size": int(image_size),
            "display_rotate_k": int(SIM_DISPLAY_ROTATE_K),
            "center_col_factory": (lambda: float(center_y_var.get())),
            "center_row_factory": (lambda: float(center_x_var.get())),
            "distance_cor_to_detector_factory": (lambda: float(corto_detector_var.get())),
            "gamma_deg_factory": (lambda: float(gamma_var.get())),
            "Gamma_deg_factory": (lambda: float(Gamma_var.get())),
            "chi_deg_factory": (lambda: float(chi_var.get())),
            "psi_deg_factory": (lambda: float(psi)),
            "psi_z_deg_factory": (lambda: float(psi_z_var.get())),
            "zs_factory": (lambda: float(zs_var.get())),
            "zb_factory": (lambda: float(zb_var.get())),
            "theta_initial_deg_factory": (
                lambda: float(_current_effective_theta_initial(strict_count=False))
            ),
            "cor_angle_deg_factory": (lambda: float(cor_angle_var.get())),
            "pixel_size_m": float(pixel_size_m),
            "wavelength": float(lambda_),
            "n2": n2,
        },
        overlay_bootstrap_kwargs={
            "ax": ax,
            "overlay_artists": geometry_runtime_state.qr_cylinder_overlay_artists,
            "overlay_cache": geometry_runtime_state.qr_cylinder_overlay_cache,
            "overlay_enabled_factory": (
                lambda: (
                    bool(geometry_overlay_actions_view_state.show_qr_cylinder_overlay_var.get())
                    if geometry_overlay_actions_view_state.show_qr_cylinder_overlay_var is not None
                    else False
                )
            ),
            "ai_factory": (lambda: simulation_runtime_state.ai_cache.get("ai")),
            "get_detector_angular_maps": (lambda ai: _get_detector_angular_maps(ai)),
            "native_sim_to_display_coords": _native_sim_to_display_coords,
            "draw_idle_factory": (lambda: canvas.draw_idle if "canvas" in globals() else None),
            "set_status_text_factory": (
                lambda: (
                    (lambda text: progress_label_positions.config(text=text))
                    if "progress_label_positions" in globals()
                    else None
                )
            ),
        },
    )
)
active_qr_cylinder_overlay_entries_factory = qr_cylinder_overlay_workflow.active_entries_factory
qr_cylinder_overlay_render_config_factory = qr_cylinder_overlay_workflow.render_config_factory
qr_cylinder_overlay_runtime = qr_cylinder_overlay_workflow.runtime
qr_cylinder_overlay_runtime_bindings_factory = qr_cylinder_overlay_workflow.bindings_factory
qr_cylinder_overlay_runtime_refresh = qr_cylinder_overlay_workflow.refresh
_qr_cylinder_overlay_runtime_toggle_impl = qr_cylinder_overlay_workflow.toggle


def qr_cylinder_overlay_runtime_toggle(*args, **kwargs):
    result = _qr_cylinder_overlay_runtime_toggle_impl(*args, **kwargs)
    _refresh_fast_viewer_runtime_mode(announce=False)
    return result


QR_CYLINDER_DISPLAY_MODE_OFF = "Off"
QR_CYLINDER_DISPLAY_MODE_ADD = "Add to Image"
QR_CYLINDER_DISPLAY_MODE_REPLACE = "Replace Simulation"
QR_CYLINDER_DISPLAY_MODE_OPTIONS = (
    QR_CYLINDER_DISPLAY_MODE_OFF,
    QR_CYLINDER_DISPLAY_MODE_ADD,
    QR_CYLINDER_DISPLAY_MODE_REPLACE,
)


def _qr_cylinder_display_mode(default=QR_CYLINDER_DISPLAY_MODE_OFF) -> str:
    mode_var = getattr(
        geometry_overlay_actions_view_state,
        "qr_cylinder_display_mode_var",
        None,
    )
    if mode_var is None:
        return str(default)
    try:
        mode = str(mode_var.get()).strip()
    except Exception:
        return str(default)
    if mode in QR_CYLINDER_DISPLAY_MODE_OPTIONS:
        return mode
    return str(default)


def _qr_cylinder_replace_simulation_enabled() -> bool:
    return _qr_cylinder_display_mode() == QR_CYLINDER_DISPLAY_MODE_REPLACE


def _sync_qr_cylinder_overlay_visibility_var() -> None:
    overlay_var = getattr(
        geometry_overlay_actions_view_state,
        "show_qr_cylinder_overlay_var",
        None,
    )
    if overlay_var is None:
        return
    try:
        overlay_var.set(_qr_cylinder_display_mode() != QR_CYLINDER_DISPLAY_MODE_OFF)
    except Exception:
        pass


def _geometry_overlays_enabled() -> bool:
    """Return whether geometry overlays should currently be visible."""

    overlay_var = geometry_overlay_actions_view_state.show_geometry_overlays_var
    if overlay_var is None:
        return True
    try:
        return bool(overlay_var.get())
    except Exception:
        return True


def _clear_all_geometry_overlay_artists(*, redraw: bool = True):
    """Clear fitted, live-preview, and manual-preview geometry overlays."""

    _clear_geometry_pick_artists(redraw=False)
    _clear_geometry_preview_artists(redraw=False)
    _clear_geometry_manual_preview_artists(redraw=False)
    if redraw:
        _request_overlay_canvas_redraw(force=True)


def _toggle_geometry_overlay_visibility() -> None:
    """Apply the persistent geometry-overlay visibility toggle."""

    if _geometry_overlays_enabled():
        _refresh_settled_overlays()
        status_text = "Geometry overlays shown."
    else:
        _clear_all_geometry_overlay_artists(redraw=True)
        gui_controllers.clear_geometry_preview_skip_once(geometry_preview_state)
        status_text = "Geometry overlays hidden. Turn on Show Geometry Overlays to redraw them."
    try:
        progress_label_geometry.config(text=status_text)
    except Exception:
        pass


def _live_geometry_preview_signature() -> tuple[object, ...]:
    """Return a lightweight signature for the current live preview context."""

    return (
        simulation_runtime_state.last_simulation_signature,
        int(background_runtime_state.current_background_index),
        id(background_runtime_state.current_background_display),
        bool(background_runtime_state.visible),
        bool(analysis_view_controls_view_state.show_caked_2d_var.get())
        if analysis_view_controls_view_state.show_caked_2d_var is not None
        else False,
    )


# -----------------------------------------------------------
# 2)  Mouse‑click handler
# -----------------------------------------------------------
hkl_lookup_view_state = app_state.hkl_lookup_view
bragg_qr_manager_view_state = app_state.bragg_qr_manager_view
hbn_geometry_debug_view_state = app_state.hbn_geometry_debug_view
geometry_overlay_actions_view_state = app_state.geometry_overlay_actions_view
analysis_view_controls_view_state = app_state.analysis_view_controls_view
analysis_export_controls_view_state = app_state.analysis_export_controls_view
analysis_peak_tools_view_state = app_state.analysis_peak_tools_view
analysis_popout_view_state = app_state.analysis_popout_view
integration_range_controls_view_state = app_state.integration_range_controls_view
analysis_peak_selection_state = app_state.analysis_peak_selection
display_controls_state = app_state.display_controls_state
display_controls_view_state = app_state.display_controls_view
primary_cif_controls_view_state = app_state.primary_cif_controls_view
cif_weight_controls_view_state = app_state.cif_weight_controls_view
structure_factor_pruning_controls_view_state = app_state.structure_factor_pruning_controls_view
beam_mosaic_parameter_sliders_view_state = app_state.beam_mosaic_parameter_sliders_view
sampling_optics_controls_view_state = app_state.sampling_optics_controls_view
finite_stack_controls_view_state = app_state.finite_stack_controls_view
ordered_structure_fit_view_state = app_state.ordered_structure_fit_view
stacking_parameter_controls_view_state = app_state.stacking_parameter_controls_view


def _sync_peak_selection_state() -> None:
    """Normalize peak/HKL interaction flags stored in shared app state."""

    peak_selection_state.hkl_pick_armed = bool(peak_selection_state.hkl_pick_armed)
    peak_selection_state.suppress_drag_press_once = bool(
        peak_selection_state.suppress_drag_press_once
    )


_sync_peak_selection_state()


def _update_geometry_preview_exclude_button_label():
    label = geometry_q_group_runtime_value_callbacks.build_preview_exclude_button_label()
    gui_views.set_geometry_tool_action_texts(
        geometry_tool_actions_view_state,
        preview_exclude_text=label,
    )
    refresh_mode_banner = globals().get("_refresh_interaction_mode_banner")
    if callable(refresh_mode_banner):
        refresh_mode_banner()


pruning_workflow = gui_runtime_fit_analysis.build_runtime_pruning_workflow(
    bootstrap_module=gui_bootstrap,
    views_module=gui_views,
    structure_factor_pruning_module=gui_structure_factor_pruning,
    view_state=structure_factor_pruning_controls_view_state,
    bragg_qr_bootstrap_kwargs={
        "bragg_qr_manager_module": gui_bragg_qr_manager,
        "root": root,
        "uniform_flag": SOLVE_Q_MODE_UNIFORM,
        "adaptive_flag": SOLVE_Q_MODE_ADAPTIVE,
        "structure_factor_pruning_view_state_factory": (
            lambda: globals().get("structure_factor_pruning_controls_view_state")
        ),
        "bragg_qr_view_state": bragg_qr_manager_view_state,
        "simulation_runtime_state": simulation_runtime_state,
        "bragg_qr_manager_state": bragg_qr_manager_state,
        "clip_prune_bias": (
            lambda value: gui_structure_factor_pruning.clip_runtime_sf_prune_bias(
                value,
                fallback=defaults.get("sf_prune_bias", 0.0),
                minimum=gui_controllers.SF_PRUNE_BIAS_MIN,
                maximum=gui_controllers.SF_PRUNE_BIAS_MAX,
            )
        ),
        "clip_solve_q_steps": (
            lambda value: gui_structure_factor_pruning.clip_runtime_solve_q_steps(
                value,
                fallback=defaults.get("solve_q_steps", DEFAULT_SOLVE_Q_STEPS),
                minimum=MIN_SOLVE_Q_STEPS,
                maximum=MAX_SOLVE_Q_STEPS,
            )
        ),
        "clip_solve_q_rel_tol": (
            lambda value: gui_structure_factor_pruning.clip_runtime_solve_q_rel_tol(
                value,
                fallback=defaults.get("solve_q_rel_tol", DEFAULT_SOLVE_Q_REL_TOL),
                minimum=MIN_SOLVE_Q_REL_TOL,
                maximum=MAX_SOLVE_Q_REL_TOL,
            )
        ),
        "normalize_solve_q_mode_label": (
            gui_structure_factor_pruning.normalize_runtime_solve_q_mode_label
        ),
        "schedule_update_factory": (
            lambda: (
                globals().get("schedule_update")
                if callable(globals().get("schedule_update"))
                else None
            )
        ),
        "primary_candidate": (lambda: a_var.get()),
        "primary_fallback": float(av),
        "secondary_candidate": (lambda: av2),
        "set_progress_text_factory": (
            lambda: (
                (lambda text: progress_label_positions.config(text=text))
                if "progress_label_positions" in globals()
                else None
            )
        ),
        "invalid_key": BRAGG_QR_L_INVALID_KEY,
        "tcl_error_types": (tk.TclError,),
    },
    pruning_controls_bootstrap_kwargs={
        "raw_prune_bias": defaults.get("sf_prune_bias", 0.0),
        "raw_solve_q_steps": defaults.get("solve_q_steps", DEFAULT_SOLVE_Q_STEPS),
        "raw_solve_q_rel_tol": defaults.get(
            "solve_q_rel_tol",
            DEFAULT_SOLVE_Q_REL_TOL,
        ),
        "raw_solve_q_mode": defaults.get("solve_q_mode", SOLVE_Q_MODE_UNIFORM),
        "prune_bias_fallback": defaults.get("sf_prune_bias", 0.0),
        "prune_bias_minimum": gui_controllers.SF_PRUNE_BIAS_MIN,
        "prune_bias_maximum": gui_controllers.SF_PRUNE_BIAS_MAX,
        "steps_fallback": defaults.get("solve_q_steps", DEFAULT_SOLVE_Q_STEPS),
        "steps_minimum": MIN_SOLVE_Q_STEPS,
        "steps_maximum": MAX_SOLVE_Q_STEPS,
        "rel_tol_fallback": defaults.get(
            "solve_q_rel_tol",
            DEFAULT_SOLVE_Q_REL_TOL,
        ),
        "rel_tol_minimum": MIN_SOLVE_Q_REL_TOL,
        "rel_tol_maximum": MAX_SOLVE_Q_REL_TOL,
        "uniform_flag": SOLVE_Q_MODE_UNIFORM,
        "adaptive_flag": SOLVE_Q_MODE_ADAPTIVE,
    },
    initialize_filters=True,
)
bragg_qr_workflow_runtime = pruning_workflow.runtime
current_sf_prune_bias = pruning_workflow.current_sf_prune_bias
current_solve_q_values = pruning_workflow.current_solve_q_values
update_sf_prune_status_label = pruning_workflow.update_status_label
apply_bragg_qr_filters = pruning_workflow.apply_filters
on_sf_prune_bias_change = pruning_workflow.on_sf_prune_bias_change
on_solve_q_steps_change = pruning_workflow.on_solve_q_steps_change
on_solve_q_rel_tol_change = pruning_workflow.on_solve_q_rel_tol_change
set_solve_q_control_states = pruning_workflow.set_solve_q_control_states
on_solve_q_mode_change = pruning_workflow.on_solve_q_mode_change
structure_factor_pruning_controls_runtime = pruning_workflow.controls_runtime


peak_selection_workflow = gui_runtime_geometry_interaction.build_runtime_peak_selection_workflow(
    bootstrap_module=gui_bootstrap,
    peak_selection_module=gui_peak_selection,
    views_module=gui_views,
    hkl_lookup_view_state=hkl_lookup_view_state,
    open_bragg_qr_groups=bragg_qr_workflow_runtime.open_window,
    simulation_runtime_state=simulation_runtime_state,
    peak_selection_state=peak_selection_state,
    image_size=int(image_size),
    hkl_lookup_view_state_factory=lambda: globals().get("hkl_lookup_view_state"),
    selected_peak_marker_factory=lambda: selected_peak_marker,
    current_primary_a_factory=lambda: float(av),
    caked_view_enabled_factory=lambda: _active_caked_primary_view(),
    primary_a_factory=lambda: float(av),
    primary_c_factory=lambda: float(cv),
    max_distance_px=float(HKL_PICK_MAX_DISTANCE_PX),
    min_separation_px=float(HKL_PICK_MIN_SEPARATION_PX),
    image_shape_factory=lambda: (
        tuple(int(v) for v in global_image_buffer.shape) if global_image_buffer.size else None
    ),
    center_col_factory=lambda: float(center_y_var.get()),
    center_row_factory=lambda: float(center_x_var.get()),
    distance_cor_to_detector_factory=lambda: float(corto_detector_var.get()),
    gamma_deg_factory=lambda: float(gamma_var.get()),
    Gamma_deg_factory=lambda: float(Gamma_var.get()),
    chi_deg_factory=lambda: float(chi_var.get()),
    psi_deg_factory=lambda: float(psi),
    psi_z_deg_factory=lambda: float(psi_z_var.get()),
    zs_factory=lambda: float(zs_var.get()),
    zb_factory=lambda: float(zb_var.get()),
    theta_initial_deg_factory=(lambda: float(_current_effective_theta_initial(strict_count=False))),
    cor_angle_deg_factory=lambda: float(cor_angle_var.get()),
    sigma_mosaic_deg_factory=lambda: float(sigma_mosaic_var.get()),
    gamma_mosaic_deg_factory=lambda: float(gamma_mosaic_var.get()),
    eta_factory=lambda: float(eta_var.get()),
    wavelength_factory=lambda: float(lambda_),
    sample_width_m_factory=lambda: float(sample_width_var.get()),
    sample_length_m_factory=lambda: float(sample_length_var.get()),
    pixel_size_m_factory=lambda: float(pixel_size_m),
    debye_x_factory=lambda: float(debye_x_var.get()),
    debye_y_factory=lambda: float(debye_y_var.get()),
    detector_center_factory=lambda: (
        float(center_x_var.get()),
        float(center_y_var.get()),
    ),
    optics_mode_factory=_current_optics_mode_flag,
    solve_q_values_factory=current_solve_q_values,
    overlay_primary_a_factory=lambda: float(a_var.get()) if "a_var" in globals() else float(av),
    overlay_primary_c_factory=lambda: float(c_var.get()) if "c_var" in globals() else float(cv),
    native_sim_to_display_coords=_native_sim_to_display_coords,
    native_detector_coords_to_caked_display_coords=(_native_detector_coords_to_live_caked_coords),
    reflection_q_group_metadata=(gui_geometry_q_group_manager.reflection_q_group_metadata),
    max_hits_per_reflection=lambda: HKL_PICK_MAX_HITS_PER_REFLECTION,
    sync_peak_selection_state=_sync_peak_selection_state,
    schedule_update_factory=lambda: (
        globals().get("schedule_update") if callable(globals().get("schedule_update")) else None
    ),
    set_status_text_factory=lambda: (
        (lambda text: progress_label_positions.config(text=text))
        if "progress_label_positions" in globals()
        else None
    ),
    draw_idle_factory=lambda: (
        _request_overlay_canvas_redraw
        if callable(globals().get("_request_overlay_canvas_redraw"))
        else None
    ),
    display_to_native_sim_coords=_display_to_native_sim_coords,
    deactivate_conflicting_modes_factory=lambda: (
        lambda: (
            _set_geometry_manual_pick_mode(False),
            _set_geometry_preview_exclude_mode(False),
        )
    ),
    on_hkl_pick_mode_changed_factory=lambda: _handle_hkl_pick_mode_changed,
    n2=n2,
    process_peaks_parallel=process_peaks_parallel,
    tcl_error_types=(tk.TclError,),
)
peak_selection_runtime = peak_selection_workflow.runtime
ensure_peak_overlay_data = peak_selection_workflow.ensure_peak_overlay_data
peak_selection_runtime_callbacks = peak_selection_workflow.callbacks
peak_selection_runtime_maintenance = peak_selection_workflow.maintenance_callbacks
hkl_lookup_controls_runtime = peak_selection_workflow.hkl_lookup_controls_runtime

_base_update_hkl_pick_button_label = peak_selection_runtime_callbacks.update_hkl_pick_button_label
_base_set_hkl_pick_mode = peak_selection_runtime_callbacks.set_hkl_pick_mode
_base_toggle_hkl_pick_mode = peak_selection_runtime_callbacks.toggle_hkl_pick_mode


def _handle_hkl_pick_mode_changed(_armed: bool) -> None:
    refresh_fast_viewer = globals().get("_refresh_fast_viewer_runtime_mode")
    if callable(refresh_fast_viewer):
        refresh_fast_viewer(announce=False)
    refresh_mode_banner = globals().get("_refresh_interaction_mode_banner")
    if callable(refresh_mode_banner):
        refresh_mode_banner()


def _update_hkl_pick_button_label_with_mode_banner() -> None:
    _base_update_hkl_pick_button_label()
    refresh_mode_banner = globals().get("_refresh_interaction_mode_banner")
    if callable(refresh_mode_banner):
        refresh_mode_banner()


def _set_hkl_pick_mode_with_mode_banner(
    enabled: bool,
    message: str | None = None,
) -> None:
    _base_set_hkl_pick_mode(bool(enabled), message=message)
    refresh_mode_banner = globals().get("_refresh_interaction_mode_banner")
    if callable(refresh_mode_banner):
        refresh_mode_banner()


def _toggle_hkl_pick_mode_with_mode_banner() -> None:
    _base_toggle_hkl_pick_mode()
    refresh_mode_banner = globals().get("_refresh_interaction_mode_banner")
    if callable(refresh_mode_banner):
        refresh_mode_banner()


peak_selection_runtime_callbacks = gui_peak_selection.SelectedPeakRuntimeCallbacks(
    update_hkl_pick_button_label=_update_hkl_pick_button_label_with_mode_banner,
    set_hkl_pick_mode=_set_hkl_pick_mode_with_mode_banner,
    toggle_hkl_pick_mode=_toggle_hkl_pick_mode_with_mode_banner,
    reselect_current_peak=peak_selection_runtime_callbacks.reselect_current_peak,
    select_peak_from_hkl_controls=(peak_selection_runtime_callbacks.select_peak_from_hkl_controls),
    clear_selected_peak=peak_selection_runtime_callbacks.clear_selected_peak,
    open_selected_peak_intersection_figure=(
        peak_selection_runtime_callbacks.open_selected_peak_intersection_figure
    ),
    select_peak_from_canvas_click=(peak_selection_runtime_callbacks.select_peak_from_canvas_click),
)
analysis_peak_runtime_callbacks = SimpleNamespace(
    set_pick_mode=(
        lambda enabled, message=None: globals()["_set_analysis_peak_pick_mode"](
            enabled,
            message=message,
        )
    ),
    select_peak_from_canvas_click=(
        lambda col, row: globals()["_select_analysis_peak_from_canvas_click"](
            col,
            row,
        )
    ),
)
hkl_lookup_controls_runtime = gui_bootstrap.build_runtime_hkl_lookup_controls_bootstrap(
    views_module=gui_views,
    view_state=hkl_lookup_view_state,
    peak_selection_callbacks=lambda: peak_selection_runtime_callbacks,
    open_bragg_qr_groups=bragg_qr_workflow_runtime.open_window,
)

geometry_manual_workflow = gui_runtime_geometry_interaction.build_runtime_geometry_manual_workflow(
    bootstrap_module=gui_bootstrap,
    manual_geometry_module=gui_manual_geometry,
    background_visible=lambda: bool(background_runtime_state.visible),
    current_background_index=(lambda: int(background_runtime_state.current_background_index)),
    current_background_image=_current_geometry_manual_pick_background_image,
    pick_session=lambda: geometry_manual_state.pick_session,
    build_initial_pairs_display=_build_geometry_manual_initial_pairs_display,
    session_initial_pairs_display=_geometry_manual_session_initial_pairs_display,
    clear_geometry_pick_artists=_clear_geometry_pick_artists,
    draw_initial_geometry_pairs_overlay=_draw_initial_geometry_pairs_overlay,
    update_button_label=(lambda: _update_geometry_manual_pick_button_label()),
    set_background_file_status_text=_refresh_background_status,
    pair_group_count=_geometry_manual_pair_group_count,
    set_status_text=lambda text: progress_label_geometry.config(text=text),
    get_cache_data=lambda **kwargs: _get_geometry_manual_pick_cache(
        param_set=_current_geometry_fit_params(),
        prefer_cache=True,
        **kwargs,
    ),
    set_pairs_for_index=_set_geometry_manual_pairs_for_index,
    pairs_for_index=_geometry_manual_pairs_for_index,
    set_pick_session=_set_geometry_manual_pick_session,
    restore_view=_restore_geometry_manual_pick_view,
    clear_preview_artists=_clear_geometry_manual_preview_artists,
    push_undo_state=_push_geometry_manual_undo_state,
    listed_q_group_entries=(lambda: globals()["_listed_geometry_q_group_entries"]()),
    format_q_group_line=(
        lambda *args, **kwargs: globals()["_format_geometry_q_group_line"](
            *args,
            **kwargs,
        )
    ),
    use_caked_space=_geometry_manual_pick_uses_caked_space,
    pick_search_window_px=float(GEOMETRY_MANUAL_PICK_SEARCH_WINDOW_PX),
    caked_search_tth_deg=float(GEOMETRY_MANUAL_CAKED_SEARCH_TTH_DEG),
    caked_search_phi_deg=float(GEOMETRY_MANUAL_CAKED_SEARCH_PHI_DEG),
    set_suppress_drag_press_once=(
        lambda enabled: setattr(
            peak_selection_state,
            "suppress_drag_press_once",
            bool(enabled),
        )
    ),
    sync_peak_selection_state=_sync_peak_selection_state,
    refine_preview_point=_geometry_manual_refine_preview_point,
    remaining_candidates=_geometry_manual_unassigned_group_candidates,
    preview_due=_geometry_manual_preview_due,
    nearest_candidate_to_point=_geometry_manual_nearest_candidate_to_point,
    position_error_px=_geometry_manual_position_error_px,
    position_sigma_px=_geometry_manual_position_sigma_px,
    caked_angles_to_background_display_coords=(_caked_angles_to_background_display_coords),
    caked_axis_to_image_index_fn=_caked_axis_to_image_index,
    last_caked_radial_values=(lambda: simulation_runtime_state.last_caked_radial_values),
    last_caked_azimuth_values=(lambda: simulation_runtime_state.last_caked_azimuth_values),
    background_display_to_native_detector_coords=(_background_display_to_native_detector_coords),
    refine_saved_pair_entry=(
        lambda entry, candidate=None: _refine_geometry_manual_pair_entry_from_cache(
            entry,
            source_entry=candidate,
        )
    ),
    show_preview=_show_geometry_manual_preview,
    refresh_pick_session=_refresh_geometry_manual_pick_session,
)
geometry_manual_runtime = geometry_manual_workflow.runtime
geometry_manual_runtime_callbacks = geometry_manual_workflow.callbacks
geometry_tool_action_workflow = (
    gui_runtime_geometry_interaction.build_runtime_geometry_tool_action_workflow(
        bootstrap_module=gui_bootstrap,
        geometry_fit_module=gui_geometry_fit,
        geometry_fit_history_state=geometry_fit_history_state,
        manual_pick_armed=lambda: bool(geometry_runtime_state.manual_pick_armed),
        set_manual_pick_armed=(
            lambda enabled: setattr(
                geometry_runtime_state,
                "manual_pick_armed",
                bool(enabled),
            )
        ),
        current_background_index=(lambda: int(background_runtime_state.current_background_index)),
        current_pick_session=lambda: geometry_manual_state.pick_session,
        manual_pick_session_active=_geometry_manual_pick_session_active,
        build_manual_pick_button_label=(gui_manual_geometry.geometry_manual_pick_button_label),
        pairs_for_index=_geometry_manual_pairs_for_index,
        pair_group_count=_geometry_manual_pair_group_count,
        set_manual_pick_text=(
            lambda text: gui_views.set_geometry_tool_action_texts(
                geometry_tool_actions_view_state,
                manual_pick_text=text,
            )
        ),
        set_history_button_state=(
            lambda can_undo, can_redo: gui_views.set_geometry_fit_history_button_state(
                geometry_tool_actions_view_state,
                can_undo=can_undo,
                can_redo=can_redo,
            )
        ),
        show_caked_2d_var=lambda: analysis_view_controls_view_state.show_caked_2d_var,
        toggle_caked_2d=lambda: toggle_caked_2d(),
        ensure_geometry_fit_caked_view=(
            lambda: _ensure_geometry_fit_caked_view(force_refresh=True)
        ),
        set_hkl_pick_mode=hkl_lookup_controls_runtime.set_hkl_pick_mode,
        set_geometry_preview_exclude_mode=(
            lambda enabled, message=None: _set_geometry_preview_exclude_mode(
                enabled,
                message=message,
            )
        ),
        cancel_manual_pick_session=geometry_manual_workflow.cancel_pick_session,
        canvas_widget=lambda: canvas.get_tk_widget(),
        push_manual_undo_state=_push_geometry_manual_undo_state,
        clear_pairs_for_current_background=(
            lambda index: _set_geometry_manual_pairs_for_index(index, [])
        ),
        clear_geometry_pick_artists=_clear_geometry_pick_artists,
        refresh_status=_refresh_background_status,
        set_progress_text=lambda text: progress_label_geometry.config(text=text),
    )
)
geometry_tool_action_runtime = geometry_tool_action_workflow.runtime
geometry_tool_action_runtime_callbacks = geometry_tool_action_workflow.callbacks
_update_geometry_fit_undo_button_state = (
    geometry_tool_action_workflow.update_fit_history_button_state
)
_update_geometry_manual_pick_button_label = (
    geometry_tool_action_workflow.update_manual_pick_button_label
)

_base_update_geometry_manual_pick_button_label = _update_geometry_manual_pick_button_label


def _update_geometry_manual_pick_button_label() -> None:
    _base_update_geometry_manual_pick_button_label()
    refresh_mode_banner = globals().get("_refresh_interaction_mode_banner")
    if callable(refresh_mode_banner):
        refresh_mode_banner()


_set_geometry_manual_pick_mode = geometry_tool_action_workflow.set_manual_pick_mode
_toggle_geometry_manual_pick_mode = geometry_tool_action_workflow.toggle_manual_pick_mode
_clear_current_geometry_manual_pairs = geometry_tool_action_workflow.clear_current_manual_pairs
_render_current_geometry_manual_pairs_base = geometry_manual_workflow.render_current_pairs


def _render_current_geometry_manual_pairs(*, update_status: bool = False) -> bool:
    """Render saved manual geometry pairs when overlays are visible."""

    if not _geometry_overlays_enabled():
        del update_status
        _clear_all_geometry_overlay_artists(redraw=True)
        return False
    return bool(_render_current_geometry_manual_pairs_base(update_status=update_status))


_toggle_geometry_manual_selection_at = geometry_manual_workflow.toggle_selection_at
_place_geometry_manual_selection_at = geometry_manual_workflow.place_selection_at
_update_geometry_manual_pick_preview_base = geometry_manual_workflow.update_pick_preview


def _update_geometry_manual_pick_preview(*args, **kwargs) -> None:
    """Suppress manual pick previews while overlays are hidden."""

    if not _geometry_overlays_enabled():
        _clear_geometry_manual_preview_artists(redraw=True)
        return
    _update_geometry_manual_pick_preview_base(*args, **kwargs)


_cancel_geometry_manual_pick_session = geometry_manual_workflow.cancel_pick_session

integration_range_workflow = gui_runtime_fit_analysis.build_runtime_integration_range_workflow(
    bootstrap_module=gui_bootstrap,
    views_module=gui_views,
    integration_range_drag_module=gui_integration_range_drag,
    integration_range_update_bootstrap_kwargs={
        "range_view_state": integration_range_controls_view_state,
        "analysis_view_state": analysis_view_controls_view_state,
        "root": root,
        "simulation_runtime_state": simulation_runtime_state,
        "analysis_view_state_factory": (lambda: analysis_view_controls_view_state),
        "range_view_state_factory": (lambda: integration_range_controls_view_state),
        "display_controls_state": display_controls_state,
        "hkl_lookup_controls_factory": (lambda: hkl_lookup_controls_runtime),
        "integration_range_drag_callbacks_factory": (
            lambda: globals().get("integration_range_drag_runtime_callbacks")
        ),
        "refresh_integration_from_cached_results_factory": (
            lambda: (
                globals().get("_refresh_integration_from_cached_results")
                if callable(globals().get("_refresh_integration_from_cached_results"))
                else None
            )
        ),
        "refresh_display_from_controls_factory": (
            lambda: (
                globals().get("_refresh_display_from_controls")
                if callable(globals().get("_refresh_display_from_controls"))
                else None
            )
        ),
        "toggle_caked_2d_factory": (
            lambda: (
                globals().get("toggle_caked_2d")
                if callable(globals().get("toggle_caked_2d"))
                else None
            )
        ),
        "schedule_update_factory": (
            lambda: (
                globals().get("schedule_update")
                if callable(globals().get("schedule_update"))
                else None
            )
        ),
        "range_update_debounce_ms_factory": (
            lambda: globals().get(
                "RANGE_UPDATE_DEBOUNCE_MS",
                120,
            )
        ),
    },
)
integration_range_update_runtime = integration_range_workflow.update_runtime
integration_range_update_runtime_callbacks = integration_range_workflow.callbacks
schedule_range_update = integration_range_workflow.schedule_range_update
_toggle_1d_plots_impl = integration_range_workflow.toggle_1d_plots
_toggle_caked_2d_impl = integration_range_workflow.toggle_caked_2d


def toggle_1d_plots() -> None:
    if bool(getattr(analysis_view_controls_view_state.show_1d_var, "get", lambda: False)()):
        ensure_analysis_surfaces_initialized()
    _toggle_1d_plots_impl()


def _invalidate_qr_cylinder_overlay_view_state(*, clear_artists: bool) -> None:
    bindings_factory = globals().get("qr_cylinder_overlay_runtime_bindings_factory")
    if not callable(bindings_factory):
        return
    try:
        bindings = bindings_factory()
    except Exception:
        return
    try:
        gui_qr_cylinder_overlay.invalidate_runtime_qr_cylinder_overlay_cache(
            bindings,
            clear_artists=bool(clear_artists),
            redraw=False,
        )
    except Exception:
        return


def _restore_combined_detector_intersection_cache() -> None:
    try:
        primary_weight = float(weight1_var.get())
    except Exception:
        primary_weight = 0.0
    try:
        secondary_weight = float(weight2_var.get())
    except Exception:
        secondary_weight = 0.0

    restored_tables = []
    if (
        simulation_runtime_state.stored_primary_sim_image is not None
        and abs(primary_weight) > 1.0e-12
        and simulation_runtime_state.stored_primary_intersection_cache is not None
    ):
        restored_tables.extend(
            _copy_intersection_cache_tables(
                simulation_runtime_state.stored_primary_intersection_cache
            )
        )
    if (
        simulation_runtime_state.stored_secondary_sim_image is not None
        and abs(secondary_weight) > 1.0e-12
        and simulation_runtime_state.stored_secondary_intersection_cache is not None
    ):
        restored_tables.extend(
            _copy_intersection_cache_tables(
                simulation_runtime_state.stored_secondary_intersection_cache
            )
        )
    simulation_runtime_state.stored_intersection_cache = restored_tables


def _apply_main_caked_view_toggle() -> None:
    show_caked_now = bool(
        getattr(
            analysis_view_controls_view_state.show_caked_2d_var,
            "get",
            lambda: False,
        )()
    )
    if show_caked_now or simulation_runtime_state.unscaled_image is None:
        schedule_update_fn = globals().get("schedule_update")
        if callable(schedule_update_fn):
            schedule_update_fn()
        return

    _restore_combined_detector_intersection_cache()
    previous_primary_view_mode = _current_primary_figure_mode()
    preserved_primary_limits = gui_canvas_interactions.capture_axis_limits(ax)
    apply_scale_factor_to_existing_results(
        update_limits=False,
        update_1d=False,
        update_canvas=False,
    )
    gui_canvas_interactions.restore_axis_view(
        ax,
        preserved_limits=preserved_primary_limits,
        default_xlim=(0.0, float(image_size)),
        default_ylim=(float(image_size), 0.0),
        preserve=(previous_primary_view_mode == "detector"),
    )
    ax.set_aspect("auto")
    gui_main_figure_chrome.set_main_figure_axes_axis_visibility(ax, visible=False)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title("")
    _sync_primary_raster_geometry(show_caked_image=False)
    gui_main_figure_chrome.apply_main_figure_axes_chrome(
        ax,
        axes_visible=False,
    )
    _request_main_canvas_redraw(force_matplotlib=False)
    _refresh_settled_overlays()
    _refresh_run_status_bar()


def toggle_caked_2d() -> None:
    _invalidate_qr_cylinder_overlay_view_state(clear_artists=True)
    _apply_main_caked_view_toggle()
    _sync_center_marker(redraw=False)
    if bool(analysis_peak_selection_state.pick_armed):
        show_caked_now = bool(
            getattr(
                analysis_view_controls_view_state.show_caked_2d_var,
                "get",
                lambda: False,
            )()
        )
        if not show_caked_now:
            _set_analysis_peak_pick_mode(
                False,
                message="Analysis peak picking requires the caked view.",
            )
            _sync_center_marker(redraw=True)
            return
    _render_analysis_peak_overlays(redraw=True)


def _refresh_display_from_controls() -> None:
    _refresh_current_intensity_display_scaling(redraw=True)


toggle_log_display = integration_range_workflow.toggle_log_display


def _set_persistent_view_mode(mode: str) -> None:
    normalized = str(mode or "detector").strip().lower()
    if normalized not in {"detector", "caked", "q_space"}:
        normalized = "detector"

    show_caked_2d_var = analysis_view_controls_view_state.show_caked_2d_var
    if show_caked_2d_var is None:
        return

    gui_views.set_app_shell_view_mode(app_shell_view_state, normalized)

    show_caked_now = bool(show_caked_2d_var.get())
    if normalized == "detector":
        if show_caked_now:
            show_caked_2d_var.set(False)
            toggle_caked_2d()
    else:
        if not show_caked_now:
            show_caked_2d_var.set(True)
            toggle_caked_2d()
        else:
            if normalized != "caked" and bool(analysis_peak_selection_state.pick_armed):
                _set_analysis_peak_pick_mode(
                    False,
                    message="Analysis peak picking requires the caked view.",
                )
            schedule_update_fn = globals().get("schedule_update")
            if callable(schedule_update_fn):
                schedule_update_fn()
    _refresh_run_status_bar()


integration_range_drag_runtime = gui_bootstrap.build_runtime_integration_range_workflow_bootstrap(
    integration_range_drag_module=gui_integration_range_drag,
    ax=ax,
    drag_state=integration_range_drag_state,
    peak_selection_state=peak_selection_state,
    range_view_state_factory=lambda: globals().get("integration_range_controls_view_state"),
    integration_region_overlay=integration_region_overlay,
    image_display=image_display,
    get_detector_angular_maps=lambda ai: _get_detector_angular_maps(ai),
    range_visible_factory=lambda: (
        bool(analysis_view_controls_view_state.show_1d_var.get())
        if analysis_view_controls_view_state.show_1d_var is not None
        else False
    ),
    caked_view_enabled_factory=lambda: _active_caked_primary_view(),
    unscaled_image_present_factory=lambda: simulation_runtime_state.unscaled_image is not None,
    ai_factory=lambda: simulation_runtime_state.ai_cache.get("ai"),
    show_1d_var_factory=lambda: analysis_view_controls_view_state.show_1d_var,
    sync_peak_selection_state=_sync_peak_selection_state,
    schedule_range_update_factory=lambda: (
        integration_range_update_runtime_callbacks.schedule_range_update
    ),
    last_sim_res2_factory=lambda: simulation_runtime_state.last_res2_sim,
    draw_idle_factory=lambda: (
        _request_main_canvas_redraw
        if callable(globals().get("_request_main_canvas_redraw"))
        else None
    ),
    set_integration_overlay_image_factory=lambda: _set_primary_integration_overlay_image,
    set_status_text_factory=lambda: (
        (lambda text: progress_label_positions.config(text=text))
        if "progress_label_positions" in globals()
        else None
    ),
)
integration_range_drag_runtime_bindings_factory = integration_range_drag_runtime.bindings_factory
integration_range_drag_runtime_callbacks = integration_range_drag_runtime.callbacks
refresh_integration_region_visuals = integration_range_drag_runtime.refresh_visuals
canvas_interaction_workflow = (
    gui_runtime_geometry_preview.build_runtime_canvas_interaction_workflow(
        bootstrap_module=gui_bootstrap,
        canvas_interactions_module=gui_canvas_interactions,
        geometry_q_group_manager_module=gui_geometry_q_group_manager,
        geometry_q_group_runtime_bindings_factory_resolver=(
            lambda: globals().get("geometry_q_group_runtime_bindings_factory")
        ),
        axis=ax,
        geometry_runtime_state=geometry_runtime_state,
        geometry_preview_state=geometry_preview_state,
        geometry_manual_state=geometry_manual_state,
        peak_selection_state=peak_selection_state,
        peak_selection_callbacks=peak_selection_runtime_callbacks,
        analysis_peak_state=analysis_peak_selection_state,
        analysis_peak_callbacks=analysis_peak_runtime_callbacks,
        integration_range_drag_callbacks=integration_range_drag_runtime_callbacks,
        manual_pick_session_active=_geometry_manual_pick_session_active,
        set_geometry_manual_pick_mode=_set_geometry_manual_pick_mode,
        toggle_geometry_manual_selection_at=_toggle_geometry_manual_selection_at,
        clamp_to_axis_view=gui_integration_range_drag.clamp_to_axis_view,
        apply_geometry_manual_pick_zoom=_apply_geometry_manual_pick_zoom,
        update_geometry_manual_pick_preview=_update_geometry_manual_pick_preview,
        place_geometry_manual_selection_at=_place_geometry_manual_selection_at,
        clear_geometry_manual_preview_artists=_clear_geometry_manual_preview_artists,
        restore_geometry_manual_pick_view=_restore_geometry_manual_pick_view,
        render_current_geometry_manual_pairs=_render_current_geometry_manual_pairs,
        caked_view_enabled_factory=lambda: _active_caked_primary_view(),
        set_geometry_status_text_factory=lambda: (
            (lambda text: progress_label_geometry.config(text=text))
            if "progress_label_geometry" in globals()
            else None
        ),
        draw_idle_factory=lambda: (
            _request_overlay_canvas_redraw
            if callable(globals().get("_request_overlay_canvas_redraw"))
            else None
        ),
        begin_live_interaction_factory=lambda: _begin_main_figure_live_interaction,
        touch_live_interaction_factory=lambda: _touch_main_figure_live_interaction,
        end_live_interaction_factory=lambda: _end_main_figure_live_interaction,
        preview_view_limits_factory=lambda: _preview_legacy_main_matplotlib_view_limits,
        commit_preview_view_factory=lambda: _commit_legacy_main_matplotlib_preview_view,
        clear_preview_view_factory=lambda: (
            lambda: _clear_legacy_main_matplotlib_preview_view(redraw=True)
        ),
    )
)
canvas_interaction_runtime = canvas_interaction_workflow.runtime
canvas_interaction_runtime_bindings_factory = canvas_interaction_workflow.bindings_factory
canvas_interaction_runtime_callbacks = canvas_interaction_workflow.callbacks

# -----------------------------------------------------------
# 3)  Bind the handler
# -----------------------------------------------------------
_bind_primary_canvas_interactions(canvas)


# ---------------------------------------------------------------------------
# Display controls for background and simulation intensity scaling
# ---------------------------------------------------------------------------


def _ensure_valid_range(min_val, max_val):
    return gui_controllers.ensure_display_intensity_range(min_val, max_val)


def _apply_background_transparency():
    return gui_background_manager.apply_background_transparency(
        display_controls_view_state,
        background_display=background_display,
    )


def _log_display_enabled() -> bool:
    log_display_var = getattr(analysis_view_controls_view_state, "log_display_var", None)
    getter = getattr(log_display_var, "get", None)
    if not callable(getter):
        return False
    try:
        return bool(getter())
    except Exception:
        return False


def _beam_center_spot_enabled() -> bool:
    beam_center_spot_var = getattr(
        analysis_view_controls_view_state,
        "show_beam_center_spot_var",
        None,
    )
    getter = getattr(beam_center_spot_var, "get", None)
    if not callable(getter):
        return False
    try:
        return bool(getter())
    except Exception:
        return False


def _sync_center_marker(*, redraw: bool) -> None:
    marker = globals().get("center_marker")
    if marker is None:
        return
    try:
        center_row = float(center_x_var.get())
        center_col = float(center_y_var.get())
    except Exception:
        return

    marker_x = float(center_col)
    marker_y = float(center_row)
    visible = _beam_center_spot_enabled()
    if visible and _current_geometry_fit_caked_roi_preview_enabled():
        visible = False
    show_caked_var = getattr(analysis_view_controls_view_state, "show_caked_2d_var", None)
    show_caked = bool(getattr(show_caked_var, "get", lambda: False)())
    if visible and show_caked:
        try:
            projected = _native_detector_coords_to_caked_display_coords(
                float(center_col),
                float(center_row),
            )
        except Exception:
            projected = None
        if projected is None:
            visible = False
        else:
            marker_x = float(projected[0])
            marker_y = float(projected[1])

    try:
        marker.set_xdata([marker_x])
        marker.set_ydata([marker_y])
        marker.set_visible(bool(visible))
    except Exception:
        return

    if not redraw:
        return
    request_overlay_redraw = globals().get("_request_overlay_canvas_redraw")
    if callable(request_overlay_redraw):
        try:
            request_overlay_redraw(force=True)
            return
        except Exception:
            pass
    request_main_redraw = globals().get("_request_main_canvas_redraw")
    if callable(request_main_redraw):
        try:
            request_main_redraw(force_matplotlib=True)
        except Exception:
            pass


def _toggle_beam_center_spot() -> None:
    _sync_center_marker(redraw=True)


def _minimum_positive_finite_value(image) -> float | None:
    if image is None:
        return None
    try:
        values = np.asarray(image, dtype=float)
    except Exception:
        return None
    if values.size == 0:
        return None
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return None
    return float(np.min(positive))


def _build_intensity_display_norm(image, min_val, max_val):
    min_val, max_val = _ensure_valid_range(min_val, max_val)
    if not _log_display_enabled():
        return Normalize(vmin=min_val, vmax=max_val)

    positive_floor = _minimum_positive_finite_value(image)
    if positive_floor is None:
        return Normalize(vmin=min_val, vmax=max_val)

    log_min = max(float(min_val), positive_floor)
    if not math.isfinite(log_min) or log_min <= 0.0:
        log_min = positive_floor

    log_max = float(max_val)
    if not math.isfinite(log_max) or log_max <= log_min:
        log_max = log_min + max(abs(log_min) * 1e-6, 1e-12)

    return LogNorm(vmin=log_min, vmax=log_max, clip=True)


def _apply_intensity_display_range(image_artist, image, min_val, max_val) -> None:
    image_artist.set_norm(_build_intensity_display_norm(image, min_val, max_val))


def _refresh_current_intensity_display_scaling(*, redraw: bool = True) -> None:
    image_source, _image_extent, _image_origin = _primary_raster_source_payload(image_display)
    if image_source is not None:
        image_min, image_max = image_display.get_clim()
        _apply_intensity_display_range(
            image_display,
            image_source,
            image_min,
            image_max,
        )

    background_source, _background_extent, _background_origin = _primary_raster_source_payload(
        background_display
    )
    if background_source is not None:
        background_min, background_max = background_display.get_clim()
        _apply_intensity_display_range(
            background_display,
            background_source,
            background_min,
            background_max,
        )

    colorbar_main.update_normal(image_display)
    caked_colorbar.update_normal(image_display)
    if redraw:
        _request_main_canvas_redraw(force_matplotlib=False)


def _apply_background_limits():
    background_min_var = display_controls_view_state.background_min_var
    background_max_var = display_controls_view_state.background_max_var
    if background_min_var is None or background_max_var is None:
        return False
    min_val = background_min_var.get()
    max_val = background_max_var.get()
    if min_val >= max_val:
        adjustment = max(abs(max_val) * 1.0e-6, 1.0e-6)
        display_controls_state.suppress_background_limit_callback = True
        try:
            background_min_var.set(max_val - adjustment)
        finally:
            display_controls_state.suppress_background_limit_callback = False
        return False
    display_controls_state.background_limits_user_override = True
    background_display.set_clim(min_val, max_val)
    _apply_background_transparency()
    _refresh_current_intensity_display_scaling(redraw=True)
    return True


def _apply_simulation_limits():
    simulation_min_var = display_controls_view_state.simulation_min_var
    simulation_max_var = display_controls_view_state.simulation_max_var
    if simulation_min_var is None or simulation_max_var is None:
        return
    min_val = simulation_min_var.get()
    max_val = simulation_max_var.get()
    if min_val >= max_val:
        adjustment = max(abs(max_val) * 1e-6, 1e-6)
        display_controls_state.suppress_simulation_limit_callback = True
        simulation_min_var.set(max_val - adjustment)
        display_controls_state.suppress_simulation_limit_callback = False
        return
    display_controls_state.simulation_limits_user_override = True
    apply_scale_factor_to_existing_results()


background_display_defaults = gui_background_manager.resolve_background_display_defaults(
    background_runtime_state.current_background_display
)
background_vmin_default = background_display_defaults.vmin_default
background_vmax_default = background_display_defaults.vmax_default
background_slider_min = background_display_defaults.slider_min
background_slider_max = background_display_defaults.slider_max
background_slider_step = background_display_defaults.slider_step


simulation_slider_min = 0.0
simulation_slider_max = max(background_slider_max, defaults["vmax"] * 5.0)
# Ensure fine-grained control so the intensity sliders support at least 1e-4 precision.
simulation_slider_step = min(
    max((simulation_slider_max - simulation_slider_min) / 500.0, 1e-6),
    1e-4,
)

scale_factor_slider_min = 0.0
scale_factor_slider_max = 2.0
scale_factor_step = 0.0001

gui_views.create_display_controls(
    parent=(
        app_shell_view_state.figure_controls_frame
        if app_shell_view_state.figure_controls_frame is not None
        else app_shell_view_state.fig_frame
    ),
    view_state=display_controls_view_state,
    background_range=(background_slider_min, background_slider_max),
    background_defaults=(background_vmin_default, background_vmax_default),
    background_step=background_slider_step,
    background_transparency=0.0,
    simulation_range=(simulation_slider_min, simulation_slider_max),
    simulation_defaults=(0.0, background_vmax_default),
    simulation_step=simulation_slider_step,
    scale_factor_range=(scale_factor_slider_min, scale_factor_slider_max),
    scale_factor_value=1.0,
    scale_factor_step=scale_factor_step,
    on_apply_background_limits=_apply_background_limits,
    on_apply_simulation_limits=_apply_simulation_limits,
)

_apply_intensity_display_range(
    background_display,
    background_runtime_state.current_background_display,
    background_vmin_default,
    background_vmax_default,
)
_apply_background_transparency()


def _get_scale_factor_value(default=1.0):
    simulation_scale_factor_var = display_controls_view_state.simulation_scale_factor_var
    if simulation_scale_factor_var is None:
        return default
    try:
        scale = float(simulation_scale_factor_var.get())
    except (tk.TclError, ValueError):
        return default
    if not np.isfinite(scale):
        return default
    return gui_controllers.normalize_display_scale_factor(scale, fallback=1.0)


MAIN_DISPLAY_RASTER_MIN_SIZE = gui_display_projection.MIN_DISPLAY_RASTER_SIZE
MAIN_DISPLAY_RASTER_MAX_SIZE = gui_display_projection.MAX_DISPLAY_RASTER_SIZE
_MAIN_RASTER_SOURCE_ARRAY_ATTR = "_ra_sim_source_image"
_MAIN_RASTER_SOURCE_EXTENT_ATTR = "_ra_sim_source_extent"
_MAIN_RASTER_SOURCE_ORIGIN_ATTR = "_ra_sim_source_origin"


def _default_main_display_raster_size_limit() -> int:
    return gui_display_projection.default_display_raster_size(image_size)


def _normalize_main_display_raster_size_limit(value: object | None = None) -> int:
    fallback = _default_main_display_raster_size_limit()
    candidate = value
    if candidate is None:
        control_var = globals().get("main_display_raster_size_var")
        getter = getattr(control_var, "get", None)
        if callable(getter):
            try:
                candidate = getter()
            except Exception:
                candidate = None
    normalized = gui_display_projection.normalize_display_raster_size_limit(
        candidate,
        fallback=getattr(
            simulation_runtime_state,
            "main_display_raster_limit",
            fallback,
        ),
        minimum=MAIN_DISPLAY_RASTER_MIN_SIZE,
        maximum=MAIN_DISPLAY_RASTER_MAX_SIZE,
    )
    simulation_runtime_state.main_display_raster_limit = int(normalized)
    control_var = globals().get("main_display_raster_size_var")
    setter = getattr(control_var, "set", None)
    getter = getattr(control_var, "get", None)
    if callable(setter) and callable(getter):
        try:
            current_value = float(getter())
        except Exception:
            current_value = math.nan
        if not math.isfinite(current_value) or abs(current_value - float(normalized)) > 1e-9:
            try:
                setter(float(normalized))
            except Exception:
                pass
    widgets = getattr(app_shell_view_state, "quick_control_widgets", None)
    if isinstance(widgets, dict):
        control = widgets.get("display_raster_size")
        if isinstance(control, dict):
            scale_widget = control.get("scale")
            configure = getattr(scale_widget, "configure", None)
            if callable(configure):
                try:
                    configure(
                        from_=float(MAIN_DISPLAY_RASTER_MIN_SIZE),
                        to=float(MAIN_DISPLAY_RASTER_MAX_SIZE),
                    )
                except Exception:
                    pass
    return int(normalized)


def _current_main_display_raster_size_limit() -> int:
    return int(_normalize_main_display_raster_size_limit())


def _store_primary_raster_source(
    artist: object | None,
    image: np.ndarray | None,
) -> np.ndarray | None:
    if artist is None:
        return None
    source = None if image is None else np.asarray(image)
    setattr(artist, _MAIN_RASTER_SOURCE_ARRAY_ATTR, source)
    return source


def _store_primary_raster_geometry(
    artist: object | None,
    *,
    origin: str,
    extent: tuple[float, float, float, float] | list[float],
) -> None:
    if artist is None:
        return
    try:
        normalized_extent = tuple(float(value) for value in extent)
    except Exception:
        return
    if len(normalized_extent) != 4:
        return
    setattr(artist, _MAIN_RASTER_SOURCE_EXTENT_ATTR, normalized_extent)
    setattr(artist, _MAIN_RASTER_SOURCE_ORIGIN_ATTR, str(origin))


def _primary_raster_source_payload(
    artist: object | None,
) -> tuple[np.ndarray | None, tuple[float, float, float, float] | None, str]:
    if artist is None:
        return (None, None, "upper")

    source = getattr(artist, _MAIN_RASTER_SOURCE_ARRAY_ATTR, None)
    if source is None:
        getter = getattr(artist, "get_array", None)
        if callable(getter):
            try:
                current = getter()
            except Exception:
                current = None
            if current is not None:
                source = np.asarray(current)

    extent = getattr(artist, _MAIN_RASTER_SOURCE_EXTENT_ATTR, None)
    if not isinstance(extent, tuple) or len(extent) != 4:
        get_extent = getattr(artist, "get_extent", None)
        if callable(get_extent):
            try:
                live_extent = tuple(float(value) for value in get_extent())
            except Exception:
                live_extent = None
            if isinstance(live_extent, tuple) and len(live_extent) == 4:
                extent = live_extent
    origin = str(getattr(artist, _MAIN_RASTER_SOURCE_ORIGIN_ATTR, "upper") or "upper")
    return (None if source is None else np.asarray(source), extent, origin)


def _current_primary_axis_limits() -> tuple[tuple[float, float], tuple[float, float]] | None:
    getter_x = getattr(ax, "get_xlim", None)
    getter_y = getattr(ax, "get_ylim", None)
    if not callable(getter_x) or not callable(getter_y):
        return None
    try:
        xlim = tuple(float(value) for value in getter_x())
        ylim = tuple(float(value) for value in getter_y())
    except Exception:
        return None
    if len(xlim) != 2 or len(ylim) != 2:
        return None
    if not all(np.isfinite(value) for value in (*xlim, *ylim)):
        return None
    return (xlim, ylim)


def _current_primary_axis_bbox_pixels() -> tuple[float | None, float | None]:
    bbox = getattr(ax, "bbox", None)
    width = getattr(bbox, "width", None)
    height = getattr(bbox, "height", None)
    try:
        width_value = float(width)
    except Exception:
        width_value = math.nan
    try:
        height_value = float(height)
    except Exception:
        height_value = math.nan
    return (
        width_value if math.isfinite(width_value) and width_value > 0.0 else None,
        height_value if math.isfinite(height_value) and height_value > 0.0 else None,
    )


def _apply_projected_primary_raster_to_artist(artist: object | None) -> bool:
    source, extent, origin = _primary_raster_source_payload(artist)
    if artist is None or source is None or extent is None:
        return False
    if _legacy_main_matplotlib_interaction_active():
        axis_limits = _current_primary_axis_limits()
        if axis_limits is None:
            axis_xlim = None
            axis_ylim = None
        else:
            axis_xlim, axis_ylim = axis_limits
        bbox_width, bbox_height = _current_primary_axis_bbox_pixels()
    else:
        axis_xlim = None
        axis_ylim = None
        bbox_width = None
        bbox_height = None
    projection = gui_display_projection.project_raster_to_view(
        source,
        extent=extent,
        axis_xlim=axis_xlim,
        axis_ylim=axis_ylim,
        max_size=_current_main_display_raster_size_limit(),
        bbox_width_px=bbox_width,
        bbox_height_px=bbox_height,
    )
    if projection is None:
        return False
    _apply_axes_image_origin(artist, origin)
    set_extent = getattr(artist, "set_extent", None)
    if callable(set_extent):
        try:
            set_extent(list(projection.extent))
        except Exception:
            pass
    set_data = getattr(artist, "set_data", None)
    if callable(set_data):
        try:
            set_data(projection.image)
        except Exception:
            return False
    return True


def _apply_current_primary_raster_projection() -> None:
    for artist in (
        globals().get("image_display"),
        globals().get("background_display"),
        globals().get("integration_region_overlay"),
    ):
        _apply_projected_primary_raster_to_artist(artist)


def _set_primary_integration_overlay_image(image: object) -> None:
    overlay_artist = globals().get("integration_region_overlay")
    if overlay_artist is None:
        return
    image_artist = globals().get("image_display")
    _store_primary_raster_source(
        overlay_artist,
        None if image is None else np.asarray(image, dtype=float),
    )
    _image_source, image_extent, image_origin = _primary_raster_source_payload(image_artist)
    if image_extent is not None:
        _store_primary_raster_geometry(
            overlay_artist,
            origin=image_origin,
            extent=image_extent,
        )
    _apply_projected_primary_raster_to_artist(overlay_artist)


def _refresh_main_display_raster_projection() -> None:
    _normalize_main_display_raster_size_limit()
    if _legacy_main_matplotlib_interaction_active():
        _clear_legacy_main_matplotlib_preview_view(redraw=False)
    _apply_current_primary_raster_projection()
    _request_main_canvas_redraw(force_matplotlib=False)


_store_primary_raster_source(image_display, global_image_buffer)
_store_primary_raster_source(
    background_display,
    background_runtime_state.current_background_display,
)
_store_primary_raster_source(
    integration_region_overlay,
    np.zeros_like(global_image_buffer),
)
_sync_primary_raster_geometry(show_caked_image=False)


def _on_scale_factor_change(*args):
    if display_controls_state.suppress_scale_factor_callback:
        return
    if _get_scale_factor_value(default=None) is None:
        return
    display_controls_state.scale_factor_user_override = True
    apply_scale_factor_to_existing_results()


if display_controls_view_state.simulation_scale_factor_var is not None:
    display_controls_view_state.simulation_scale_factor_var.trace_add(
        "write", _on_scale_factor_change
    )


def _update_background_slider_defaults(image, reset_override=False):
    return gui_background_manager.update_background_slider_defaults(
        display_controls_state,
        display_controls_view_state,
        background_display=background_display,
        image=image,
        reset_override=reset_override,
    )


def _update_simulation_sliders_from_image(image, reset_override=False):
    simulation_min_var = display_controls_view_state.simulation_min_var
    simulation_max_var = display_controls_view_state.simulation_max_var
    simulation_min_slider = display_controls_view_state.simulation_min_slider
    simulation_max_slider = display_controls_view_state.simulation_max_slider
    if image is None or image.size == 0:
        return
    finite_pixels = np.asarray(image, dtype=float)
    finite_pixels = finite_pixels[np.isfinite(finite_pixels)]
    if finite_pixels.size == 0:
        return
    lower_bound = 0.0
    upper_bound = gui_controllers.resolve_simulation_display_upper_bound(
        finite_pixels,
        percentile=99.0,
    )
    slider_to = upper_bound
    simulation_min_slider.configure(from_=lower_bound, to=slider_to)
    simulation_max_slider.configure(from_=lower_bound, to=slider_to)
    slider_from = lower_bound
    display_controls_state.suppress_simulation_limit_callback = True
    if reset_override or not display_controls_state.simulation_limits_user_override:
        min_value = 0.0
        max_value = upper_bound
    else:
        min_value = float(simulation_min_var.get())
        max_value = float(simulation_max_var.get())
        min_value = min(max(min_value, slider_from), slider_to)
        max_value = min(max(max_value, slider_from), slider_to)
    min_value, max_value = _ensure_valid_range(min_value, max_value)
    simulation_min_var.set(min_value)
    simulation_max_var.set(max_value)
    display_controls_state.suppress_simulation_limit_callback = False
    if reset_override:
        display_controls_state.simulation_limits_user_override = False


def _set_scale_factor_value(value, adjust_range=True, reset_override=False):
    simulation_scale_factor_var = display_controls_view_state.simulation_scale_factor_var
    scale_factor_slider = display_controls_view_state.scale_factor_slider
    if simulation_scale_factor_var is None or scale_factor_slider is None:
        return
    value = gui_controllers.normalize_display_scale_factor(value, fallback=1.0)

    slider_min = float(scale_factor_slider.cget("from"))
    slider_max = float(scale_factor_slider.cget("to"))
    if adjust_range:
        new_min = slider_min
        new_max = slider_max
        if value < slider_min:
            new_min = value
        if value > slider_max:
            new_max = value
        if new_min != slider_min or new_max != slider_max:
            scale_factor_slider.configure(from_=new_min, to=new_max)
    else:
        value = min(max(value, slider_min), slider_max)

    display_controls_state.suppress_scale_factor_callback = True
    simulation_scale_factor_var.set(value)
    display_controls_state.suppress_scale_factor_callback = False
    if reset_override:
        display_controls_state.scale_factor_user_override = False
    else:
        display_controls_state.scale_factor_user_override = True


def _install_scale_factor_entry_bindings():
    scale_entry = display_controls_view_state.scale_factor_entry
    if scale_entry is None:
        return

    def _apply_scale_entry(_event=None):
        raw = scale_entry.get().strip()
        if not raw:
            current = _get_scale_factor_value(default=1.0)
            scale_entry.delete(0, tk.END)
            scale_entry.insert(0, f"{current:.6g}")
            return
        try:
            value = float(raw)
        except ValueError:
            current = _get_scale_factor_value(default=1.0)
            scale_entry.delete(0, tk.END)
            scale_entry.insert(0, f"{current:.6g}")
            return
        _set_scale_factor_value(value, adjust_range=True, reset_override=False)
        apply_scale_factor_to_existing_results(update_limits=False)

    scale_entry.unbind("<FocusOut>")
    scale_entry.unbind("<Return>")
    scale_entry.bind("<FocusOut>", _apply_scale_entry)
    scale_entry.bind("<Return>", _apply_scale_entry)


_install_scale_factor_entry_bindings()
_refresh_fast_viewer_runtime_mode(announce=False)


def _suggest_scale_factor(sim_image, bg_image):
    sim_pixels = None if sim_image is None else np.asarray(sim_image, dtype=float)
    bg_pixels = None if bg_image is None else np.asarray(bg_image, dtype=float)
    if sim_pixels is None or bg_pixels is None:
        return 1.0

    # Percentiles over every pixel are expensive on full detector images.
    # Subsample finite values to keep this nearly constant-time.
    sim_pixels = sim_pixels[np.isfinite(sim_pixels)]
    bg_pixels = bg_pixels[np.isfinite(bg_pixels)]
    max_samples = 200_000
    if sim_pixels.size > max_samples:
        sim_step = max(1, sim_pixels.size // max_samples)
        sim_pixels = sim_pixels[::sim_step]
    if bg_pixels.size > max_samples:
        bg_step = max(1, bg_pixels.size // max_samples)
        bg_pixels = bg_pixels[::bg_step]

    if sim_pixels.size == 0 or bg_pixels.size == 0:
        return 1.0
    sim_reference_pixels = sim_pixels[sim_pixels > 0]
    if sim_reference_pixels.size == 0:
        sim_reference_pixels = np.abs(sim_pixels)
    bg_reference_pixels = bg_pixels[bg_pixels > 0]
    if bg_reference_pixels.size == 0:
        bg_reference_pixels = np.abs(bg_pixels)
    sim_ref = float(np.nanpercentile(sim_reference_pixels, 99))
    bg_ref = float(np.nanpercentile(bg_reference_pixels, 99))
    if not np.isfinite(sim_ref) or abs(sim_ref) < 1e-12:
        return 1.0
    if not np.isfinite(bg_ref) or abs(bg_ref) < 1e-12:
        return 1.0
    return bg_ref / sim_ref


def _mark_chi_square_dirty():
    simulation_runtime_state.chi_square_update_token = (
        int(globals().get("chi_square_update_token", 0)) + 1
    )


def _auto_match_scale_factor_to_radial_peak():
    sim_curve = simulation_runtime_state.last_1d_integration_data.get("intensities_2theta_sim")
    bg_curve = simulation_runtime_state.last_1d_integration_data.get("intensities_2theta_bg")

    if sim_curve is None or bg_curve is None:
        ai = simulation_runtime_state.ai_cache.get("ai")
        sim_img = simulation_runtime_state.last_1d_integration_data.get("simulated_2d_image")
        bg_img = _get_current_background_backend()
        if ai is not None and sim_img is not None and bg_img is not None:
            try:
                tth_min, tth_max = sorted((float(tth_min_var.get()), float(tth_max_var.get())))
                phi_min = float(phi_min_var.get())
                phi_max = float(phi_max_var.get())
                sim_res2 = caking(sim_img, ai)
                i2t_sim, _, _, _ = caked_up(
                    sim_res2,
                    tth_min,
                    tth_max,
                    phi_min,
                    phi_max,
                )
                bg_res2 = caking(bg_img, ai)
                i2t_bg, _, _, _ = caked_up(
                    bg_res2,
                    tth_min,
                    tth_max,
                    phi_min,
                    phi_max,
                )
                sim_curve = i2t_sim
                bg_curve = i2t_bg
                simulation_runtime_state.last_1d_integration_data["intensities_2theta_sim"] = (
                    i2t_sim
                )
                simulation_runtime_state.last_1d_integration_data["intensities_2theta_bg"] = i2t_bg
            except Exception:
                sim_curve = None
                bg_curve = None
        if sim_curve is None or bg_curve is None:
            progress_label_positions.config(
                text=(
                    "Auto-match requires background + simulation radial curves. "
                    "Enable background and run/update simulation once."
                )
            )
            return

    sim_vals = np.asarray(sim_curve, dtype=float)
    bg_vals = np.asarray(bg_curve, dtype=float)
    sim_vals = sim_vals[np.isfinite(sim_vals)]
    bg_vals = bg_vals[np.isfinite(bg_vals)]
    if sim_vals.size == 0 or bg_vals.size == 0:
        progress_label_positions.config(text="Auto-match failed: radial curves are empty.")
        return

    sim_peak = float(np.max(sim_vals))
    bg_peak = float(np.max(bg_vals))
    if not np.isfinite(sim_peak) or sim_peak <= 0.0:
        progress_label_positions.config(
            text="Auto-match failed: simulated radial peak is non-positive."
        )
        return
    if not np.isfinite(bg_peak) or bg_peak < 0.0:
        progress_label_positions.config(
            text="Auto-match failed: background radial peak is invalid."
        )
        return

    target_scale = bg_peak / sim_peak
    _set_scale_factor_value(target_scale, adjust_range=True, reset_override=False)
    apply_scale_factor_to_existing_results(update_limits=False)
    progress_label_positions.config(
        text=(
            f"Auto-matched scale factor to radial peak: {target_scale:.6g} "
            f"(sim max={sim_peak:.6g}, bg max={bg_peak:.6g})."
        )
    )


def _clear_analysis_integration_region() -> None:
    def _apply_cleared_integration() -> None:
        refresh_cached = globals().get("_refresh_integration_from_cached_results")
        if callable(refresh_cached):
            refresh_cached()
            return
        refresh_visuals = globals().get("refresh_integration_region_visuals")
        if callable(refresh_visuals):
            refresh_visuals()
        request_redraw = globals().get("_request_overlay_canvas_redraw")
        if callable(request_redraw):
            request_redraw()

    gui_analysis_quick_controls.clear_analysis_integration_region(
        show_1d_var=analysis_view_controls_view_state.show_1d_var,
        hide_drag_region=getattr(integration_range_drag_runtime_callbacks, "reset", None),
        disable_peak_pick=lambda: _set_analysis_peak_pick_mode(False),
        clear_peak_selection=_clear_selected_analysis_peaks,
        apply_cleared_integration=_apply_cleared_integration,
        set_status_text=lambda text: progress_label_positions.config(text=text),
    )


def _update_chi_square_display(force=False):
    state = globals().setdefault(
        "chi_square_state",
        {
            "last_ts": 0.0,
            "last_token": -1,
            "last_text": "Chi-Squared: N/A",
        },
    )
    now = perf_counter()
    # Throttle to avoid repeatedly scanning full detector arrays during slider
    # drags; keep the last displayed value between refreshes.
    if (not force) and ((now - float(state.get("last_ts", 0.0))) < CHI_SQUARE_UPDATE_INTERVAL_S):
        last_text = str(state.get("last_text", "Chi-Squared: N/A"))
        chi_square_label.config(text=last_text)
        return

    token = int(globals().get("chi_square_update_token", 0))
    if (not force) and token == int(state.get("last_token", -1)):
        last_text = str(state.get("last_text", "Chi-Squared: N/A"))
        chi_square_label.config(text=last_text)
        state["last_ts"] = now
        return

    try:
        native_background = _get_current_background_backend()
        text = "Chi-Squared: N/A"
        if (
            background_runtime_state.visible
            and native_background is not None
            and global_image_buffer.size
        ):
            sim_vals = np.asarray(global_image_buffer, dtype=float)
            bg_vals = np.asarray(native_background, dtype=float)
            if (
                sim_vals.size
                and bg_vals.size
                and np.max(sim_vals) > 0
                and np.max(bg_vals) > 0
                and sim_vals.shape == bg_vals.shape
            ):
                # Subsample large arrays for responsive UI updates.
                max_points = 250_000
                if sim_vals.ndim == 2 and sim_vals.size > max_points:
                    stride = int(max(1, np.sqrt(sim_vals.size / float(max_points))))
                    sim_vals = sim_vals[::stride, ::stride]
                    bg_vals = bg_vals[::stride, ::stride]
                norm_sim = sim_vals / np.max(sim_vals)
                norm_bg = bg_vals / np.max(bg_vals)
                chi_sq_val = float(np.mean(np.square(norm_bg - norm_sim)) * norm_sim.size)
                text = f"Chi-Squared: {chi_sq_val:.2e}"
    except Exception as exc:
        text = f"Chi-Squared Error: {exc}"

    state["last_ts"] = now
    state["last_token"] = token
    state["last_text"] = text
    chi_square_label.config(text=text)


def apply_scale_factor_to_existing_results(
    update_limits=False,
    *,
    update_1d=True,
    update_canvas=True,
    update_chi_square=True,
):
    chi_state = globals().setdefault(
        "chi_square_state",
        {
            "last_ts": 0.0,
            "last_token": -1,
            "last_text": "Chi-Squared: N/A",
        },
    )
    active_view_mode = _resolved_primary_analysis_display_mode()
    show_caked_image = active_view_mode == "caked"
    show_q_space_image = active_view_mode == "q_space"
    show_analysis_image = show_caked_image or show_q_space_image
    if simulation_runtime_state.unscaled_image is None:
        background_min_var = display_controls_view_state.background_min_var
        background_max_var = display_controls_view_state.background_max_var
        chi_square_sig = (
            None,
            bool(background_runtime_state.visible),
            int(background_runtime_state.current_background_index),
            int(background_runtime_state.backend_rotation_k) % 4,
            bool(background_runtime_state.backend_flip_x),
            bool(background_runtime_state.backend_flip_y),
        )
        if chi_state.get("buffer_sig") != chi_square_sig:
            chi_state["buffer_sig"] = chi_square_sig
            _mark_chi_square_dirty()
        if not show_analysis_image:
            _sync_primary_raster_geometry(view_mode="detector")
            if (
                background_runtime_state.visible
                and background_runtime_state.current_background_display is not None
            ):
                _preview_sim_unused, preview_background_display = (
                    _geometry_fit_caked_roi_preview_images(
                        show_caked_image=False,
                        simulation_image=None,
                        background_image=background_runtime_state.current_background_display,
                    )
                )
                background_preview_source = (
                    preview_background_display
                    if isinstance(preview_background_display, np.ndarray)
                    else background_runtime_state.current_background_display
                )
                _store_primary_raster_source(
                    background_display,
                    background_preview_source,
                )
                _apply_intensity_display_range(
                    background_display,
                    background_preview_source,
                    background_min_var.get(),
                    background_max_var.get(),
                )
                background_display.set_visible(True)
            else:
                background_display.set_visible(False)
        if update_canvas:
            _request_main_canvas_redraw(force_matplotlib=False)
        if update_chi_square:
            _update_chi_square_display()
        return

    scale = _get_scale_factor_value(default=1.0)
    if abs(float(scale) - 1.0) <= 1e-12:
        np.copyto(global_image_buffer, simulation_runtime_state.unscaled_image, casting="unsafe")
    else:
        np.multiply(simulation_runtime_state.unscaled_image, float(scale), out=global_image_buffer)
    scaled_image = global_image_buffer
    base_unscaled_sig = globals().get("last_unscaled_image_signature")
    chi_square_sig = (
        base_unscaled_sig,
        round(float(scale), 9),
        bool(background_runtime_state.visible),
        int(background_runtime_state.current_background_index),
        int(background_runtime_state.backend_rotation_k) % 4,
        bool(background_runtime_state.backend_flip_x),
        bool(background_runtime_state.backend_flip_y),
    )
    if chi_state.get("buffer_sig") != chi_square_sig:
        chi_state["buffer_sig"] = chi_square_sig
        _mark_chi_square_dirty()

    scaled_caked = None
    scaled_q_space = None
    display_image_for_limits = scaled_image
    if show_q_space_image:
        scaled_q_space = simulation_runtime_state.last_q_space_image_unscaled * scale
        display_image_for_limits = scaled_q_space
    elif show_caked_image:
        scaled_caked = simulation_runtime_state.last_caked_image_unscaled * scale
        display_image_for_limits = scaled_caked
    _update_simulation_sliders_from_image(
        display_image_for_limits,
        reset_override=update_limits,
    )

    preview_background_source = None
    if show_q_space_image:
        if (
            background_runtime_state.visible
            and simulation_runtime_state.last_q_space_background_image_unscaled is not None
        ):
            preview_background_source = (
                simulation_runtime_state.last_q_space_background_image_unscaled * scale
            )
    elif show_caked_image:
        if (
            background_runtime_state.visible
            and simulation_runtime_state.last_caked_background_image_unscaled is not None
        ):
            preview_background_source = (
                simulation_runtime_state.last_caked_background_image_unscaled * scale
            )
    elif (
        background_runtime_state.visible
        and background_runtime_state.current_background_display is not None
    ):
        preview_background_source = background_runtime_state.current_background_display

    simulation_source = (
        scaled_q_space
        if show_q_space_image
        else (scaled_caked if show_caked_image else global_image_buffer)
    )
    if show_q_space_image:
        simulation_preview_source = simulation_source
    else:
        simulation_preview_source, background_preview_source = (
            _geometry_fit_caked_roi_preview_display_sources(
                show_caked_image=bool(show_caked_image),
                simulation_image=simulation_source,
                background_image=preview_background_source,
            )
        )
        if not isinstance(simulation_preview_source, np.ndarray):
            simulation_preview_source = simulation_source

    if not show_analysis_image:
        _store_primary_raster_source(image_display, simulation_preview_source)
        _apply_intensity_display_range(
            image_display,
            simulation_preview_source,
            display_controls_view_state.simulation_min_var.get(),
            display_controls_view_state.simulation_max_var.get(),
        )
    else:
        _store_primary_raster_source(image_display, simulation_preview_source)
    _sync_primary_raster_geometry(view_mode=active_view_mode)

    if show_analysis_image:
        caked_min = float(display_controls_view_state.simulation_min_var.get())
        caked_max = float(display_controls_view_state.simulation_max_var.get())
        caked_min, caked_max = _ensure_valid_range(caked_min, caked_max)
        _apply_intensity_display_range(
            image_display,
            simulation_preview_source,
            caked_min,
            caked_max,
        )
        if not simulation_runtime_state.caked_limits_user_override:
            vmin_caked_var.set(caked_min)
            vmax_caked_var.set(caked_max)

    if not show_analysis_image:
        if isinstance(background_preview_source, np.ndarray):
            _store_primary_raster_source(
                background_display,
                background_preview_source,
            )
            _apply_intensity_display_range(
                background_display,
                background_preview_source,
                display_controls_view_state.background_min_var.get(),
                display_controls_view_state.background_max_var.get(),
            )
            background_display.set_visible(True)
        else:
            background_display.set_visible(False)
    elif isinstance(background_preview_source, np.ndarray):
        _store_primary_raster_source(background_display, background_preview_source)
        background_display.set_visible(True)
        _apply_intensity_display_range(
            background_display,
            background_preview_source,
            display_controls_view_state.background_min_var.get(),
            display_controls_view_state.background_max_var.get(),
        )
    else:
        background_display.set_visible(False)

    colorbar_main.update_normal(image_display)
    caked_colorbar.update_normal(image_display)

    if update_1d and (
        analysis_view_controls_view_state.show_1d_var.get()
        and _analysis_integration_outputs_visible()
        and "line_1d_rad" in globals()
        and "line_1d_az" in globals()
    ):
        if (
            simulation_runtime_state.last_1d_integration_data["radials_sim"] is not None
            and simulation_runtime_state.last_1d_integration_data["intensities_2theta_sim"]
            is not None
        ):
            line_1d_rad.set_data(
                simulation_runtime_state.last_1d_integration_data["radials_sim"],
                simulation_runtime_state.last_1d_integration_data["intensities_2theta_sim"] * scale,
            )
        if (
            simulation_runtime_state.last_1d_integration_data["azimuths_sim"] is not None
            and simulation_runtime_state.last_1d_integration_data["intensities_azimuth_sim"]
            is not None
        ):
            line_1d_az.set_data(
                simulation_runtime_state.last_1d_integration_data["azimuths_sim"],
                simulation_runtime_state.last_1d_integration_data["intensities_azimuth_sim"]
                * scale,
            )
        if "ax_1d_radial" in globals():
            ax_1d_radial.relim()
            ax_1d_radial.autoscale_view(scalex=False, scaley=True)
        if "ax_1d_azim" in globals():
            ax_1d_azim.relim()
            ax_1d_azim.autoscale_view(scalex=False, scaley=True)
        _clear_analysis_peak_fit_results(redraw=False, update_text=True)
        _render_analysis_peak_overlays(redraw=False)
        if "canvas_1d" in globals():
            canvas_1d.draw_idle()

    if update_canvas:
        _request_main_canvas_redraw(force_matplotlib=False)
    if update_chi_square:
        _update_chi_square_display()


_update_background_slider_defaults(
    background_runtime_state.current_background_display, reset_override=True
)


# Track caked intensity limits without exposing separate sliders in the UI.
simulation_runtime_state.caked_limits_user_override = False

vmin_caked_var = tk.DoubleVar(value=0.0)
vmax_caked_var = tk.DoubleVar(value=2000.0)

analysis_1d_interaction_bindings = None
analysis_surfaces_initialized = False


def _reset_analysis_figure_view() -> None:
    if ax_1d_radial is None or ax_1d_azim is None:
        return
    if not gui_analysis_figure_controls.reset_analysis_axes_view(
        ax_1d_radial,
        ax_1d_azim,
    ):
        return
    _render_analysis_peak_overlays(redraw=False)
    try:
        canvas_1d.draw_idle()
    except Exception:
        pass


def _mount_analysis_figure(parent) -> None:
    global canvas_1d
    global analysis_1d_interaction_bindings

    if canvas_1d is not None:
        try:
            canvas_1d.get_tk_widget().destroy()
        except Exception:
            pass
        canvas_1d = None

    canvas_1d = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(
        fig_1d,
        master=parent,
    )
    canvas_1d.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    analysis_1d_interaction_bindings = (
        gui_analysis_figure_controls.create_analysis_figure_interactions(
            canvas=canvas_1d,
            axes=(ax_1d_radial, ax_1d_azim),
            on_reset_view=_reset_analysis_figure_view,
        )
    )


fig_1d, (ax_1d_radial, ax_1d_azim) = plt.subplots(2, 1, figsize=(5, 8))
canvas_1d = None

(line_1d_rad,) = ax_1d_radial.plot([], [], "b-", label="Simulated (2θ)")
(line_1d_rad_bg,) = ax_1d_radial.plot([], [], "r--", label="Background (2θ)")
ax_1d_radial.legend()
ax_1d_radial.set_xlabel("2θ (degrees)")
ax_1d_radial.set_ylabel("Intensity")
ax_1d_radial.set_title("Radial Integration (2θ)")

(line_1d_az,) = ax_1d_azim.plot([], [], "b-", label="Simulated (φ)")
(line_1d_az_bg,) = ax_1d_azim.plot([], [], "r--", label="Background (φ)")
ax_1d_azim.legend()
ax_1d_azim.set_xlabel("Azimuth (degrees)")
ax_1d_azim.set_ylabel("Intensity")
ax_1d_azim.set_title("Azimuthal Integration (φ)")

tth_min_var = None
tth_max_var = None
phi_min_var = None
phi_max_var = None
tth_min_slider = None
tth_max_slider = None
phi_min_slider = None
phi_max_slider = None

PHI_ZERO_OFFSET_DEGREES = -90.0
DEFAULT_ANALYSIS_RADIAL_BINS = 1000
DEFAULT_ANALYSIS_AZIMUTH_BINS = 720


def _adjust_phi_zero(phi_values):
    """Center azimuths at ``PHI_ZERO_OFFSET_DEGREES`` and mirror about the x-axis."""

    return PHI_ZERO_OFFSET_DEGREES - np.asarray(phi_values)


def _wrap_phi_range(phi_values):
    """Wrap azimuthal values into the ``[-180, 180)`` interval."""

    wrapped = ((np.asarray(phi_values) + 180.0) % 360.0) - 180.0
    return wrapped


def caking(
    data,
    ai,
    *,
    npt_rad=DEFAULT_ANALYSIS_RADIAL_BINS,
    npt_azim=DEFAULT_ANALYSIS_AZIMUTH_BINS,
    rows=None,
    cols=None,
):
    integrate_kwargs = {
        "npt_rad": int(max(1, npt_rad)),
        "npt_azim": int(max(1, npt_azim)),
        "correctSolidAngle": True,
        "unit": "2th_deg",
    }
    if rows is None or cols is None:
        integrate_kwargs["method"] = "lut"
    else:
        integrate_kwargs["rows"] = rows
        integrate_kwargs["cols"] = cols
    return ai.integrate2d(
        data,
        **integrate_kwargs,
    )


def _copy_intersection_cache_tables(cache):
    return _primary_cache_helpers.copy_intersection_cache_tables(cache)


def _copy_hit_tables(hit_tables):
    return _primary_cache_helpers.copy_hit_tables(hit_tables)


def _resolved_peak_table_payload(intersection_cache, legacy_hit_tables):
    return _primary_cache_helpers.resolved_peak_table_payload(
        intersection_cache,
        legacy_hit_tables,
    )


def _clear_primary_contribution_cache() -> None:
    # Keep "primary_contribution" visible here; source-based tests assert that
    # this runtime still routes that cache family through local retention hooks.
    _primary_cache_helpers.clear_primary_contribution_cache(simulation_runtime_state)


def _reset_combined_simulation_artifacts() -> None:
    _primary_cache_helpers.reset_combined_simulation_artifacts(
        simulation_runtime_state,
        trace_live_cache_event=_trace_live_cache_event,
        live_cache_count=_live_cache_count,
    )


def _store_primary_cache_payload(
    *,
    cache_signature: object,
    source_mode: str,
    active_keys: Sequence[object],
    contribution_keys: Sequence[object],
    raw_hit_tables: Sequence[np.ndarray],
) -> None:
    _primary_cache_helpers.store_primary_cache_payload(
        simulation_runtime_state,
        cache_signature=cache_signature,
        source_mode=source_mode,
        active_keys=active_keys,
        contribution_keys=contribution_keys,
        raw_hit_tables=raw_hit_tables,
        retain_runtime_optional_cache=_retain_runtime_optional_cache,
        trace_live_cache_event=_trace_live_cache_event,
        live_cache_count=_live_cache_count,
        live_cache_signature_summary=_live_cache_signature_summary,
    )


def _rematerialize_primary_cache_artifacts(
    *,
    image_size: int,
    mosaic_params: dict[str, object],
    a_primary: float,
    c_primary: float,
) -> dict[str, object]:
    return _primary_cache_helpers.rematerialize_primary_cache_artifacts(
        simulation_runtime_state,
        image_size=image_size,
        mosaic_params=mosaic_params,
        a_primary=a_primary,
        c_primary=c_primary,
        trace_live_cache_event=_trace_live_cache_event,
        live_cache_signature_summary=_live_cache_signature_summary,
        live_cache_shape=_live_cache_shape,
        live_cache_count=_live_cache_count,
    )


def _apply_primary_cache_artifacts(payload: dict[str, object]) -> None:
    _primary_cache_helpers.apply_primary_cache_artifacts(
        simulation_runtime_state,
        payload,
        reset_combined_simulation_artifacts=_reset_combined_simulation_artifacts,
        trace_live_cache_event=_trace_live_cache_event,
        live_cache_signature_summary=_live_cache_signature_summary,
        live_cache_shape=_live_cache_shape,
        live_cache_count=_live_cache_count,
    )


def _auto_caked_limits(image):
    """Return sensible display limits for a caked image."""

    if image is None:
        return 0.0, 1.0

    finite_mask = np.isfinite(image)
    if not np.any(finite_mask):
        return 0.0, 1.0

    finite_vals = image[finite_mask]
    vmin = float(np.nanmin(finite_vals))
    vmax = float(np.nanmax(finite_vals))

    if not (math.isfinite(vmin) and math.isfinite(vmax)):
        return 0.0, 1.0

    if math.isclose(vmin, vmax):
        if vmin == 0.0:
            vmax = 1.0
        else:
            spread = abs(vmax) * 1e-3 or 1.0
            vmin -= spread
            vmax += spread

    return vmin, vmax


def _set_image_origin(image_display, origin):
    _apply_axes_image_origin(image_display, origin)


def caked_up(res2, tth_min, tth_max, phi_min, phi_max):
    intensity = res2.intensity
    radial_2theta = res2.radial
    azimuth_vals = _adjust_phi_zero(res2.azimuthal)

    tth_min, tth_max = sorted((float(tth_min), float(tth_max)))
    phi_min = float(phi_min)
    phi_max = float(phi_max)

    mask_rad = (radial_2theta >= tth_min) & (radial_2theta <= tth_max)
    radial_filtered = radial_2theta[mask_rad]

    mask_az = gui_integration_range_drag.detector_phi_mask(
        azimuth_vals,
        phi_min,
        phi_max,
    )
    azimuth_sub = azimuth_vals[mask_az]

    intensity_sub = intensity[np.ix_(mask_az, mask_rad)]
    if phi_max < phi_min and azimuth_sub.size:
        azimuth_sub = np.asarray(azimuth_sub, dtype=float)
        azimuth_sub = np.where(azimuth_sub < phi_min, azimuth_sub + 360.0, azimuth_sub)
        azimuth_order = np.argsort(azimuth_sub)
        azimuth_sub = azimuth_sub[azimuth_order]
        intensity_sub = intensity_sub[azimuth_order, :]
    intensity_vs_2theta = np.sum(intensity_sub, axis=0)
    intensity_vs_phi = np.sum(intensity_sub, axis=1)

    return intensity_vs_2theta, intensity_vs_phi, azimuth_sub, radial_filtered


def _detector_angular_maps_for_shape(ai, detector_shape):
    if ai is None:
        return None, None

    if detector_shape is None or len(detector_shape) < 2:
        return None, None
    if detector_shape[0] <= 0 or detector_shape[1] <= 0:
        return None, None

    detector_shape = tuple(int(v) for v in detector_shape[:2])
    use_cache = getattr(global_image_buffer, "shape", None) == detector_shape

    if use_cache and simulation_runtime_state.ai_cache.get("detector_shape") == detector_shape:
        two_theta = simulation_runtime_state.ai_cache.get("detector_two_theta")
        phi_vals = simulation_runtime_state.ai_cache.get("detector_phi")
    else:
        try:
            two_theta = ai.twoThetaArray(shape=detector_shape, unit="2th_deg")
        except TypeError:
            two_theta = np.rad2deg(ai.twoThetaArray(shape=detector_shape))

        try:
            phi_vals = ai.chiArray(shape=detector_shape, unit="deg")
        except TypeError:
            phi_vals = np.rad2deg(ai.chiArray(shape=detector_shape))

        if use_cache:
            simulation_runtime_state.ai_cache["detector_shape"] = detector_shape
            simulation_runtime_state.ai_cache["detector_two_theta"] = two_theta
            simulation_runtime_state.ai_cache["detector_phi"] = phi_vals

    if two_theta is None or phi_vals is None:
        return None, None

    return np.asarray(two_theta, dtype=float), _adjust_phi_zero(phi_vals)


def _get_detector_angular_maps(ai):
    return _detector_angular_maps_for_shape(
        ai,
        getattr(global_image_buffer, "shape", None),
    )


def _snap_caked_axis_value(axis_values, value):
    if axis_values is None or not np.isfinite(value):
        return float("nan")
    try:
        axis = np.asarray(axis_values, dtype=float).reshape(-1)
    except Exception:
        return float("nan")
    if axis.size <= 0:
        return float("nan")
    finite_axis = axis[np.isfinite(axis)]
    if finite_axis.size <= 0:
        return float("nan")
    best_idx = int(np.argmin(np.abs(finite_axis - float(value))))
    return float(finite_axis[best_idx])


def _prepare_caked_intersection_cache(
    intersection_cache,
    *,
    center,
    detector_distance,
    pixel_size,
):
    source_tables = _copy_intersection_cache_tables(intersection_cache)
    if not source_tables:
        return source_tables

    try:
        center_arr = np.asarray(center, dtype=float).reshape(-1)
    except Exception:
        center_arr = np.empty((0,), dtype=float)
    if center_arr.size < 2 or not np.isfinite(center_arr[0]) or not np.isfinite(center_arr[1]):
        return source_tables

    if not (np.isfinite(detector_distance) and float(detector_distance) > 0.0):
        return source_tables
    if not (np.isfinite(pixel_size) and float(pixel_size) > 0.0):
        return source_tables

    center_row = float(center_arr[0])
    center_col = float(center_arr[1])
    detector_distance = float(detector_distance)
    pixel_size = float(pixel_size)

    transformed = []
    for table in source_tables:
        try:
            arr = np.asarray(table, dtype=float)
        except Exception:
            arr = np.empty((0, 16), dtype=float)

        if arr.ndim != 2:
            transformed.append(arr.copy())
            continue

        # Keep the raw detector-space columns intact and refresh the trailing
        # 2theta / phi slots from detector geometry.
        out_cols = max(int(arr.shape[1]), 16)
        out = np.full((arr.shape[0], out_cols), np.nan, dtype=float)
        if arr.shape[0] > 0 and arr.shape[1] > 0:
            out[:, : arr.shape[1]] = arr
        if arr.shape[0] == 0 or arr.shape[1] < 4:
            transformed.append(out)
            continue

        cols = np.asarray(arr[:, 2], dtype=float)
        rows = np.asarray(arr[:, 3], dtype=float)
        valid = np.isfinite(cols) & np.isfinite(rows)
        if np.any(valid):
            dx = (cols[valid] - center_col) * pixel_size
            dy = (center_row - rows[valid]) * pixel_size
            radius = np.hypot(dx, dy)
            out[valid, 14] = np.degrees(np.arctan2(radius, detector_distance))
            out[valid, 15] = _wrap_phi_range(np.degrees(np.arctan2(dx, dy)))
        transformed.append(out)

    return transformed


simulation_runtime_state.profile_cache = {}
simulation_runtime_state.last_1d_integration_data = {
    "radials_sim": None,
    "intensities_2theta_sim": None,
    "azimuths_sim": None,
    "intensities_azimuth_sim": None,
    "radials_bg": None,
    "intensities_2theta_bg": None,
    "azimuths_bg": None,
    "intensities_azimuth_bg": None,
    "simulated_2d_image": None,
}

simulation_runtime_state.last_caked_image_unscaled = None
simulation_runtime_state.last_caked_extent = None
simulation_runtime_state.last_caked_background_image_unscaled = None
simulation_runtime_state.last_caked_radial_values = None
simulation_runtime_state.last_caked_azimuth_values = None
simulation_runtime_state.last_caked_intersection_cache = None
simulation_runtime_state.last_q_space_image_unscaled = None
simulation_runtime_state.last_q_space_extent = None
simulation_runtime_state.last_q_space_background_image_unscaled = None
simulation_runtime_state.last_q_space_qr_values = None
simulation_runtime_state.last_q_space_qz_values = None

simulation_runtime_state.last_res2_background = None
simulation_runtime_state.last_res2_sim = None
simulation_runtime_state.ai_cache = {}
simulation_runtime_state.geometry_fit_caking_ai_cache = {}


def _clear_1d_plot_cache_and_lines():
    line_1d_rad.set_data([], [])
    line_1d_az.set_data([], [])
    line_1d_rad_bg.set_data([], [])
    line_1d_az_bg.set_data([], [])
    _clear_analysis_peak_fit_results(redraw=False, update_text=True)
    _render_analysis_peak_overlays(redraw=False)
    if canvas_1d is not None:
        canvas_1d.draw_idle()
    simulation_runtime_state.last_1d_integration_data["radials_sim"] = None
    simulation_runtime_state.last_1d_integration_data["intensities_2theta_sim"] = None
    simulation_runtime_state.last_1d_integration_data["azimuths_sim"] = None
    simulation_runtime_state.last_1d_integration_data["intensities_azimuth_sim"] = None
    simulation_runtime_state.last_1d_integration_data["radials_bg"] = None
    simulation_runtime_state.last_1d_integration_data["intensities_2theta_bg"] = None
    simulation_runtime_state.last_1d_integration_data["azimuths_bg"] = None
    simulation_runtime_state.last_1d_integration_data["intensities_azimuth_bg"] = None


def _update_1d_plots_from_caked(sim_res2, bg_res2):
    _clear_analysis_peak_fit_results(redraw=False, update_text=True)
    i2t_sim, i_phi_sim, az_sim, rad_sim = caked_up(
        sim_res2,
        tth_min_var.get(),
        tth_max_var.get(),
        phi_min_var.get(),
        phi_max_var.get(),
    )
    simulation_runtime_state.last_1d_integration_data["radials_sim"] = rad_sim
    simulation_runtime_state.last_1d_integration_data["intensities_2theta_sim"] = i2t_sim
    simulation_runtime_state.last_1d_integration_data["azimuths_sim"] = az_sim
    simulation_runtime_state.last_1d_integration_data["intensities_azimuth_sim"] = i_phi_sim

    scale = _get_scale_factor_value(default=1.0)
    line_1d_rad.set_data(rad_sim, i2t_sim * scale)
    line_1d_az.set_data(az_sim, i_phi_sim * scale)

    if bg_res2 is not None:
        i2t_bg, i_phi_bg, az_bg, rad_bg = caked_up(
            bg_res2,
            tth_min_var.get(),
            tth_max_var.get(),
            phi_min_var.get(),
            phi_max_var.get(),
        )
        simulation_runtime_state.last_1d_integration_data["radials_bg"] = rad_bg
        simulation_runtime_state.last_1d_integration_data["intensities_2theta_bg"] = i2t_bg
        simulation_runtime_state.last_1d_integration_data["azimuths_bg"] = az_bg
        simulation_runtime_state.last_1d_integration_data["intensities_azimuth_bg"] = i_phi_bg
        line_1d_rad_bg.set_data(rad_bg, i2t_bg)
        line_1d_az_bg.set_data(az_bg, i_phi_bg)
    else:
        line_1d_rad_bg.set_data([], [])
        line_1d_az_bg.set_data([], [])
        simulation_runtime_state.last_1d_integration_data["radials_bg"] = None
        simulation_runtime_state.last_1d_integration_data["intensities_2theta_bg"] = None
        simulation_runtime_state.last_1d_integration_data["azimuths_bg"] = None
        simulation_runtime_state.last_1d_integration_data["intensities_azimuth_bg"] = None

    ax_1d_radial.set_yscale("linear")
    ax_1d_azim.set_yscale("linear")
    ax_1d_radial.relim()
    ax_1d_radial.autoscale_view()
    ax_1d_azim.relim()
    ax_1d_azim.autoscale_view()
    _render_analysis_peak_overlays(redraw=False)
    if canvas_1d is not None:
        canvas_1d.draw_idle()


def _refresh_integration_from_cached_results():

    ai = simulation_runtime_state.ai_cache.get("ai")
    if not analysis_view_controls_view_state.show_1d_var.get():
        _clear_1d_plot_cache_and_lines()
        refresh_integration_region_visuals()
        _request_overlay_canvas_redraw()
        return True
    if not _analysis_integration_outputs_visible():
        return True

    if simulation_runtime_state.unscaled_image is None:
        return False

    if ai is None:
        return False

    analysis_pending = (
        simulation_runtime_state.analysis_active_job is not None
        or simulation_runtime_state.analysis_queued_job is not None
    )
    if gui_integration_range_drag.range_refresh_requires_pending_analysis_result(
        active_job=simulation_runtime_state.analysis_active_job,
        queued_job=simulation_runtime_state.analysis_queued_job,
        cached_result=simulation_runtime_state.last_res2_sim,
    ):
        return False

    if simulation_runtime_state.last_res2_sim is None:
        simulation_runtime_state.last_res2_sim = caking(simulation_runtime_state.unscaled_image, ai)

    bg_res2 = None
    native_background = _get_current_background_backend()
    if background_runtime_state.visible and native_background is not None:
        if simulation_runtime_state.last_res2_background is None:
            if not analysis_pending:
                simulation_runtime_state.last_res2_background = caking(native_background, ai)
        bg_res2 = simulation_runtime_state.last_res2_background
    else:
        simulation_runtime_state.last_res2_background = None

    _update_1d_plots_from_caked(simulation_runtime_state.last_res2_sim, bg_res2)
    refresh_integration_region_visuals()
    _request_overlay_canvas_redraw()
    return True


def _clip_bandwidth_percent(value) -> float:
    fallback = float(defaults.get("bandwidth_percent", bandwidth * 100.0))
    try:
        bw_percent = float(value)
    except (TypeError, ValueError, tk.TclError):
        bw_percent = fallback
    if not np.isfinite(bw_percent):
        bw_percent = fallback
    return float(np.clip(bw_percent, 0.0, 10.0))


def _current_bandwidth_fraction() -> float:
    var = globals().get("bandwidth_percent_var")
    if var is None:
        return float(max(0.0, bandwidth))
    try:
        bw_percent = var.get()
    except Exception:
        bw_percent = defaults.get("bandwidth_percent", bandwidth * 100.0)
    return _clip_bandwidth_percent(bw_percent) / 100.0


def update_mosaic_cache():
    """
    Keep the current beam/mosaic samples unless sampling inputs changed.

    This preserves the same sampled beam positions/divergence across normal
    simulation updates so changing unrelated sliders does not rebuild the
    detector pattern unnecessarily.
    """
    active_bandwidth = _current_bandwidth_fraction()
    expected_sample_count = int(max(1, simulation_runtime_state.num_samples))
    sampling_signature = (
        RANDOM_GAUSSIAN_SAMPLING,
        expected_sample_count,
        float(divergence_sigma),
        float(bw_sigma),
        float(lambda_),
        float(active_bandwidth),
    )

    beam_x_cached = np.asarray(
        simulation_runtime_state.profile_cache.get("beam_x_array", []), dtype=np.float64
    ).ravel()
    beam_y_cached = np.asarray(
        simulation_runtime_state.profile_cache.get("beam_y_array", []), dtype=np.float64
    ).ravel()
    theta_cached = np.asarray(
        simulation_runtime_state.profile_cache.get("theta_array", []), dtype=np.float64
    ).ravel()
    phi_cached = np.asarray(
        simulation_runtime_state.profile_cache.get("phi_array", []), dtype=np.float64
    ).ravel()
    wavelength_cached = np.asarray(
        simulation_runtime_state.profile_cache.get("wavelength_array", []), dtype=np.float64
    ).ravel()

    has_cached_samples = (
        beam_x_cached.size > 0
        and beam_x_cached.size == int(expected_sample_count)
        and beam_y_cached.size == beam_x_cached.size
        and theta_cached.size == beam_x_cached.size
        and phi_cached.size == beam_x_cached.size
        and wavelength_cached.size == beam_x_cached.size
    )

    cached_signature = simulation_runtime_state.profile_cache.get("_sampling_signature")
    should_resample = not has_cached_samples
    if has_cached_samples:
        if cached_signature is None:
            # Existing externally-provided samples: keep them and adopt the
            # current signature so future explicit sampling-input changes can
            # trigger a re-sample.
            simulation_runtime_state.profile_cache["_sampling_signature"] = sampling_signature
        else:
            should_resample = tuple(cached_signature) != sampling_signature

    if should_resample:
        (
            beam_x_array,
            beam_y_array,
            theta_array,
            phi_array,
            wavelength_array,
        ) = generate_random_profiles(
            num_samples=expected_sample_count,
            divergence_sigma=divergence_sigma,
            bw_sigma=bw_sigma,
            lambda0=lambda_,
            bandwidth=active_bandwidth,
        )
        sample_weights_array = None
        simulation_runtime_state.profile_cache = {
            "beam_x_array": beam_x_array,
            "beam_y_array": beam_y_array,
            "theta_array": theta_array,
            "phi_array": phi_array,
            "wavelength_array": wavelength_array,
            "sample_weights": sample_weights_array,
            "_sampling_signature": sampling_signature,
        }
    else:
        sample_weights_array = simulation_runtime_state.profile_cache.get("sample_weights")

    active_optics_cif_path = _resolve_optics_cif_path()
    optics_signature = (
        tuple(
            simulation_runtime_state.profile_cache.get("_sampling_signature", sampling_signature)
        ),
        os.path.abspath(str(active_optics_cif_path)),
    )
    if simulation_runtime_state.profile_cache.get("_optics_signature") != optics_signature:
        simulation_runtime_state.profile_cache["n2_sample_array"] = _current_sample_n2_array(
            simulation_runtime_state.profile_cache.get("wavelength_array", []),
            active_cif_path=active_optics_cif_path,
        )
        simulation_runtime_state.profile_cache["_optics_signature"] = optics_signature

    simulation_runtime_state.profile_cache.update(
        {
            "sigma_mosaic_deg": sigma_mosaic_var.get(),
            "gamma_mosaic_deg": gamma_mosaic_var.get(),
            "eta": eta_var.get(),
            "solve_q_steps": current_solve_q_values().steps,
            "solve_q_rel_tol": current_solve_q_values().rel_tol,
            "solve_q_mode": current_solve_q_values().mode_flag,
            "bandwidth_percent": active_bandwidth * 100.0,
            "sample_weights": sample_weights_array,
        }
    )


def on_mosaic_slider_change(*args):
    update_mosaic_cache()
    schedule_update()


def _current_phase_delta_expression() -> str:
    """Return the active HT delta-phase expression from GUI state."""

    fallback = normalize_phase_delta_expression(
        defaults.get("phase_delta_expression", DEFAULT_PHASE_DELTA_EXPRESSION),
        fallback=DEFAULT_PHASE_DELTA_EXPRESSION,
    )
    var = finite_stack_controls_view_state.phase_delta_expr_var
    if var is None:
        return fallback

    try:
        value = str(var.get())
    except Exception:
        value = fallback

    candidate = normalize_phase_delta_expression(value, fallback=fallback)
    try:
        return validate_phase_delta_expression(candidate)
    except ValueError:
        return fallback


def _current_phi_l_divisor() -> float:
    """Return the active HT L-divisor used in ``phi = delta + 2*pi*L/div``."""

    fallback = normalize_phi_l_divisor(
        defaults.get("phi_l_divisor", DEFAULT_PHI_L_DIVISOR),
        fallback=DEFAULT_PHI_L_DIVISOR,
    )
    var = finite_stack_controls_view_state.phi_l_divisor_var
    if var is None:
        return fallback

    try:
        value = var.get()
    except Exception:
        value = fallback
    return normalize_phi_l_divisor(value, fallback=fallback)


def _current_iodine_z(active_cif_path=None) -> float:
    """Return iodine z inferred from the active CIF (clamped to [0, 1])."""

    return gui_structure_model.current_iodine_z(
        structure_model_state,
        atom_site_override_state,
        active_cif_path=active_cif_path,
        tcl_error_types=(tk.TclError,),
    )


def _current_ordered_structure_scale() -> float:
    """Return the current ordered-intensity scale as a finite non-negative float."""

    var = globals().get("ordered_structure_scale_var")
    fallback = defaults.get("ordered_structure_scale", 1.0)
    if var is None:
        return gui_ordered_structure_fit.normalize_ordered_structure_scale(
            fallback,
            fallback=1.0,
        )
    try:
        raw_value = var.get()
    except Exception:
        raw_value = fallback
    return gui_ordered_structure_fit.normalize_ordered_structure_scale(
        raw_value,
        fallback=float(fallback),
    )


def _scaled_bragg_qr_dict(qr_dict, scale: float):
    """Return a copied Bragg-Qr dictionary with its intensity arrays scaled."""

    copied = gui_controllers.copy_bragg_qr_dict(qr_dict)
    scale_value = float(scale)
    if not math.isfinite(scale_value):
        scale_value = 0.0
    for entry in copied.values():
        entry["I"] = np.asarray(entry.get("I", []), dtype=np.float64) * scale_value
    return copied


(line_rmin,) = ax.plot([], [], color="white", linestyle="-", linewidth=2, zorder=5)
(line_rmax,) = ax.plot([], [], color="white", linestyle="-", linewidth=2, zorder=5)
(line_amin,) = ax.plot([], [], color="cyan", linestyle="-", linewidth=2, zorder=5)
(line_amax,) = ax.plot([], [], color="cyan", linestyle="-", linewidth=2, zorder=5)

FULL_UPDATE_DEBOUNCE_MS = 90
PREVIEW_UPDATE_DEBOUNCE_MS = 24
UPDATE_DEBOUNCE_MS = FULL_UPDATE_DEBOUNCE_MS
RANGE_UPDATE_DEBOUNCE_MS = 120
CHI_SQUARE_UPDATE_INTERVAL_S = 0.5
SIMULATION_WORKER_POLL_MS = 40
PREVIEW_CALCULATIONS_ENABLED = True
LIVE_DRAG_PREVIEW_ENABLED = False
INITIAL_PREVIEW_MAX_SAMPLES = 24
LIVE_DRAG_PREVIEW_MAX_SAMPLES = 8
LIVE_DRAG_ANALYSIS_RADIAL_BINS = 240
LIVE_DRAG_ANALYSIS_AZIMUTH_BINS = 180
LIVE_DRAG_SETTLE_MS = 80
MAIN_MATPLOTLIB_INTERACTION_REDRAW_INTERVAL_S = 1.0 / 30.0
CAKING_CACHE_MAX_ENTRIES = 8

simulation_runtime_state.update_pending = None
simulation_runtime_state.integration_update_pending = None
simulation_runtime_state.update_running = False
simulation_runtime_state.update_trace_counter = 0


def _worker_job_key(payload: object) -> tuple[object, int, str]:
    if not isinstance(payload, dict):
        return (None, -1, "full")
    return (
        payload.get("signature"),
        int(payload.get("epoch", -1)),
        str(payload.get("job_kind", "full")),
    )


def _cached_hit_tables_reusable(
    requested_signature: object,
    *,
    run_primary: bool,
    run_secondary: bool,
) -> bool:
    if requested_signature != simulation_runtime_state.stored_hit_table_signature:
        return False
    if bool(run_primary) and simulation_runtime_state.stored_primary_max_positions is None:
        return False
    if bool(run_secondary) and simulation_runtime_state.stored_secondary_max_positions is None:
        return False
    return True


def _analysis_job_key(payload: object) -> tuple[object, int, bool]:
    if not isinstance(payload, dict):
        return (None, -1, False)
    return (
        payload.get("signature"),
        int(payload.get("epoch", -1)),
        bool(payload.get("is_preview", False)),
    )


def _job_trace_fields(
    payload: object,
    *,
    prefix: str = "job",
) -> dict[str, object]:
    """Return one compact set of trace fields describing a runtime job payload."""

    if not isinstance(payload, dict):
        return {}
    fields: dict[str, object] = {
        f"{prefix}_id": payload.get("job_id"),
        f"{prefix}_epoch": payload.get("epoch"),
        f"{prefix}_signature": payload.get("signature"),
    }
    if "is_preview" in payload:
        fields[f"{prefix}_preview"] = bool(payload.get("is_preview", False))
    return fields


def _append_job_queue_trace(
    event: str,
    *,
    lane: str,
    job: object = None,
    active_job: object = None,
    queued_job: object = None,
    replaced_job: object = None,
    reason: object = None,
    outcome: object = None,
    error: object = None,
) -> None:
    """Append one queue-state trace event for simulation or analysis jobs."""

    fields: dict[str, object] = {
        "lane": str(lane),
        "update_phase": getattr(simulation_runtime_state, "update_phase", None),
        "worker_active": bool(
            getattr(simulation_runtime_state, "worker_active_job", None) is not None
        ),
        "worker_queued": bool(
            getattr(simulation_runtime_state, "worker_queued_job", None) is not None
        ),
        "analysis_active": bool(
            getattr(simulation_runtime_state, "analysis_active_job", None) is not None
        ),
        "analysis_queued": bool(
            getattr(simulation_runtime_state, "analysis_queued_job", None) is not None
        ),
    }
    fields.update(_job_trace_fields(job, prefix="job"))
    fields.update(_job_trace_fields(active_job, prefix="active"))
    fields.update(_job_trace_fields(queued_job, prefix="queued"))
    fields.update(_job_trace_fields(replaced_job, prefix="replaced"))
    if reason is not None:
        fields["reason"] = str(reason)
    if outcome is not None:
        fields["outcome"] = str(outcome)
    if error is not None:
        fields["error"] = str(error)
    _append_runtime_update_trace(event, **fields)


def _truncate_run_status_piece(text: object, *, max_chars: int = 24) -> str:
    summary = " ".join(str(text).split())
    if len(summary) <= max_chars:
        return summary
    if max_chars <= 6:
        return summary[:max_chars]
    lead = max(2, (max_chars - 3) // 2)
    tail = max(1, max_chars - 3 - lead)
    return f"{summary[:lead]}...{summary[-tail:]}"


def _extract_fit_quality_text() -> str:
    """Return the best available user-facing fit quality summary."""

    ordered_view_state = globals().get("ordered_structure_fit_view_state")
    ordered_result_var = getattr(ordered_view_state, "result_var", None)
    if ordered_result_var is not None:
        try:
            ordered_text = " ".join(str(ordered_result_var.get() or "").splitlines()[0].split())
        except Exception:
            ordered_text = ""
        if ordered_text and ordered_text != "No ordered-structure fit run yet.":
            return ordered_text

    ordered_label = globals().get("progress_label_ordered_structure")
    if ordered_label is not None:
        try:
            ordered_status = " ".join(str(ordered_label.cget("text") or "").split())
        except Exception:
            ordered_status = ""
        if ordered_status and ordered_status != "Ordered structure fit: waiting.":
            return ordered_status

    mosaic_label = globals().get("progress_label_mosaic")
    if mosaic_label is not None:
        try:
            mosaic_text = " ".join(str(mosaic_label.cget("text") or "").split())
        except Exception:
            mosaic_text = ""
        if mosaic_text:
            return mosaic_text

    geometry_label = globals().get("progress_label_geometry")
    if geometry_label is not None:
        try:
            geometry_text = " ".join(str(geometry_label.cget("text") or "").split())
        except Exception:
            geometry_text = ""
        if geometry_text:
            return geometry_text

    chi_label = globals().get("chi_square_label")
    if chi_label is not None:
        try:
            chi_text = " ".join(str(chi_label.cget("text") or "").split())
        except Exception:
            chi_text = ""
        if chi_text and chi_text not in {"Chi-Squared:", "Chi-Squared: N/A"}:
            return chi_text

    return "Waiting for ordered-structure fit"


def _current_app_shell_view_mode() -> str:
    selected_mode = str(
        getattr(app_shell_view_state.view_mode_var, "get", lambda: "detector")() or "detector"
    ).strip().lower()
    if selected_mode not in {"detector", "caked", "q_space"}:
        selected_mode = "detector"
    show_caked = bool(
        getattr(analysis_view_controls_view_state.show_caked_2d_var, "get", lambda: False)()
    )
    if selected_mode == "q_space":
        return "q_space" if show_caked else "detector"
    if show_caked:
        return "caked"
    return "detector"


def _active_caked_primary_view() -> bool:
    return _current_app_shell_view_mode() == "caked"


def _resolved_primary_analysis_display_mode() -> str:
    requested_mode = _current_app_shell_view_mode()
    q_space_available = bool(
        simulation_runtime_state.last_q_space_image_unscaled is not None
        and simulation_runtime_state.last_q_space_extent is not None
    )
    caked_available = bool(
        simulation_runtime_state.last_caked_image_unscaled is not None
        and simulation_runtime_state.last_caked_extent is not None
    )
    if requested_mode == "q_space" and q_space_available:
        return "q_space"
    if requested_mode in {"q_space", "caked"} and caked_available:
        return "caked"
    return "detector"


def _analysis_integration_outputs_visible() -> bool:
    return gui_analysis_visibility.analysis_outputs_visible(
        control_tab_var=app_shell_view_state.control_tab_var,
        popout_open=gui_views.analysis_popout_window_open(analysis_popout_view_state),
        assume_visible_when_unknown=True,
    )


def _refresh_analysis_integration_if_visible() -> None:
    if not _analysis_integration_outputs_visible():
        return
    if not bool(getattr(analysis_view_controls_view_state.show_1d_var, "get", lambda: False)()):
        return
    ensure_analysis_surfaces_initialized()
    refresh_cached = globals().get("_refresh_integration_from_cached_results")
    if not callable(refresh_cached):
        return
    try:
        refresh_cached()
    except Exception:
        pass


def _current_app_shell_view_text() -> str:
    view_mode = _resolved_primary_analysis_display_mode()
    if view_mode == "q_space":
        base = "Q Space"
    elif view_mode == "caked":
        base = "Caked"
    else:
        base = "Detector"
    show_1d = bool(getattr(analysis_view_controls_view_state.show_1d_var, "get", lambda: False)())
    if show_1d:
        return f"{base} + 1D"
    return base


def _current_primary_figure_mode() -> str:
    current_x_label = " ".join(str(ax.get_xlabel() or "").split()).lower()
    current_y_label = " ".join(str(ax.get_ylabel() or "").split()).lower()
    if current_x_label == "qr (a^-1)" and current_y_label == "qz (a^-1)":
        return "q_space"
    if current_x_label == "2θ (degrees)".lower() and current_y_label == "φ (degrees)".lower():
        return "caked"
    return "detector"


def _default_primary_view_limits(
    view_mode: str | None = None,
) -> tuple[float, float, float, float]:
    normalized_mode = str(view_mode or _current_primary_figure_mode()).strip().lower()
    if normalized_mode == "q_space":
        qr_min, qr_max, qz_min, qz_max = (
            list(simulation_runtime_state.last_q_space_extent)
            if simulation_runtime_state.last_q_space_extent is not None
            else [0.0, 1.0, -1.0, 1.0]
        )
        if not (
            math.isfinite(qr_min) and math.isfinite(qr_max) and qr_max > qr_min
        ):
            qr_min, qr_max = 0.0, 1.0
        if not (
            math.isfinite(qz_min) and math.isfinite(qz_max) and qz_max > qz_min
        ):
            qz_min, qz_max = -1.0, 1.0
        return (
            float(qr_min),
            float(qr_max),
            float(qz_min),
            float(qz_max),
        )
    if normalized_mode == "caked":
        radial_min, radial_max, azimuth_min, azimuth_max = (
            list(simulation_runtime_state.last_caked_extent)
            if simulation_runtime_state.last_caked_extent is not None
            else [0.0, 90.0, -180.0, 180.0]
        )
        if not (
            math.isfinite(radial_min) and math.isfinite(radial_max) and radial_max > radial_min
        ):
            radial_min, radial_max = 0.0, 90.0
        if not (
            math.isfinite(azimuth_min) and math.isfinite(azimuth_max) and azimuth_max > azimuth_min
        ):
            azimuth_min, azimuth_max = -180.0, 180.0
        return (
            float(radial_min),
            float(radial_max),
            float(azimuth_min),
            float(azimuth_max),
        )
    return (0.0, float(image_size), float(image_size), 0.0)


def _reset_primary_figure_view() -> None:
    x0, x1, y0, y1 = _default_primary_view_limits()
    ax.set_xlim(float(x0), float(x1))
    ax.set_ylim(float(y0), float(y1))
    try:
        _reset_fast_viewer_view()
    except Exception:
        pass
    _request_main_canvas_redraw(force_matplotlib=not _fast_viewer_active())


def _current_geometry_preview_gate_summary() -> str:
    overlay = getattr(geometry_preview_state, "overlay", None)
    if overlay is None:
        return ""
    try:
        matched = len(getattr(overlay, "pairs", ()) or ())
    except Exception:
        matched = 0
    summary_parts: list[str] = []

    try:
        simulated_count = int(getattr(overlay, "simulated_count", 0) or 0)
    except Exception:
        simulated_count = 0
    if simulated_count > 0:
        summary_parts.append(f"Peaks {matched}/{simulated_count}")

    try:
        min_matches = int(getattr(overlay, "min_matches", 0) or 0)
    except Exception:
        min_matches = 0
    if min_matches > 0:
        summary_parts.append(f"Gate {matched}/{min_matches}")

    try:
        q_total = int(getattr(overlay, "q_group_total", 0) or 0)
    except Exception:
        q_total = 0
    if q_total > 0:
        try:
            q_excluded = int(getattr(overlay, "q_group_excluded", 0) or 0)
        except Exception:
            q_excluded = 0
        summary_parts.append(f"Qr/Qz {max(0, q_total - q_excluded)}/{q_total}")

    return " | ".join(summary_parts)


def _geometry_fit_cache_status(
    *,
    selected_indices: Sequence[int],
    current_background_index: int,
) -> tuple[str, str]:
    try:
        background_theta_values = list(_current_background_theta_values(strict_count=False))
    except Exception:
        background_theta_values = []
    stale_reason = gui_geometry_fit.geometry_fit_dataset_cache_stale_reason(
        geometry_fit_dataset_cache_state.payload,
        selected_background_indices=selected_indices,
        current_background_index=int(current_background_index),
        joint_background_mode=_geometry_fit_uses_shared_theta_offset(selected_indices),
        background_theta_values=background_theta_values,
    )
    if stale_reason is None:
        return ("Fresh", "")
    if str(stale_reason).strip() == "Run geometry fit first.":
        return ("Not run", str(stale_reason))
    return ("Stale", str(stale_reason))


def _build_current_dataset_acquisition_text(
    *,
    theta_base_value: float,
    theta_effective_value: float,
    shared_theta: bool,
) -> str:
    """Summarize the active acquisition angles for the selected background."""

    parts: list[str] = []
    if np.isfinite(theta_base_value):
        parts.append(f"theta_i={theta_base_value:.4f} deg")
    else:
        parts.append("theta_i=n/a")

    if np.isfinite(theta_effective_value):
        delta = theta_effective_value - theta_base_value
        if np.isfinite(theta_base_value) and abs(delta) > 1.0e-6:
            mode = "shared offset" if shared_theta else "fit-adjusted"
            parts.append(f"theta={theta_effective_value:.4f} deg ({delta:+.4f}, {mode})")
        else:
            parts.append(f"theta={theta_effective_value:.4f} deg")
    else:
        parts.append("theta=n/a")
    return " | ".join(parts)


def _build_current_dataset_geometry_text(
    *,
    background_index: int,
    selected_indices: Sequence[int],
    freshness_status: str,
) -> str:
    """Summarize how the selected background participates in geometry fitting."""

    try:
        selected_index_set = {int(idx) for idx in selected_indices}
    except Exception:
        selected_index_set = set()

    if not selected_index_set:
        inclusion_text = "geometry fit set not configured"
    elif int(background_index) in selected_index_set:
        inclusion_text = "included in geometry fit"
    else:
        inclusion_text = "excluded from geometry fit"

    try:
        pair_count = len(_geometry_manual_pairs_for_index(int(background_index)))
    except Exception:
        pair_count = 0
    try:
        group_count = int(_geometry_manual_pair_group_count(int(background_index)))
    except Exception:
        group_count = 0

    pair_text = f"{pair_count} manual pairs"
    if group_count > 0 and group_count != pair_count:
        pair_text += f" ({group_count} groups)"

    if str(freshness_status).strip().lower() == "not run":
        cache_text = "geometry fit not run"
    else:
        cache_text = f"cache {str(freshness_status).strip().lower()}"
    return " | ".join((inclusion_text, pair_text, cache_text))


def _build_current_dataset_simulation_text(*, view_text: str) -> str:
    """Summarize the live simulation controls that shape the current dataset."""

    optics_control = globals().get("optics_mode_var")

    try:
        optics_text = _normalize_optics_mode_label(optics_control.get())
    except Exception:
        optics_text = "Unknown"

    sample_count = int(max(1, getattr(simulation_runtime_state, "num_samples", 1)))
    try:
        sf_text = gui_controllers.format_structure_factor_pruning_status(
            simulation_runtime_state.sf_prune_stats,
            prune_bias=current_sf_prune_bias(),
        )
    except Exception:
        sf_text = "SF pruning unavailable"
    return f"{view_text} | optics {optics_text} | sampling {sample_count:,} samples | {sf_text}"


def _build_current_dataset_structure_text() -> str:
    """Summarize the active CIF model and phase weights for the current dataset."""

    try:
        primary_path = str(_current_primary_cif_path()).strip()
    except Exception:
        primary_path = ""
    if not primary_path:
        return "No CIF loaded"

    secondary_path = str(getattr(structure_model_state, "cif_file2", "") or "").strip()
    phase_parts = [Path(primary_path).name]
    if secondary_path:
        phase_parts.append(Path(secondary_path).name)

    weight1_control = globals().get("weight1_var")
    weight2_control = globals().get("weight2_var")
    try:
        weight1_value = float(weight1_control.get())
    except Exception:
        weight1_value = 1.0
    try:
        weight2_value = float(weight2_control.get())
    except Exception:
        weight2_value = 0.0

    if secondary_path:
        return f"{' + '.join(phase_parts)} | phase weights {weight1_value:.2f}/{weight2_value:.2f}"
    return f"{phase_parts[0]} | phase weight {weight1_value:.2f}"


def _refresh_session_summary_panel() -> None:
    """Refresh the always-visible session summary shown above the tabs."""

    osc_files = list(getattr(background_runtime_state, "osc_files", ()))
    background_total = len(osc_files)
    bg_idx = max(
        0,
        min(int(background_runtime_state.current_background_index), max(0, background_total - 1)),
    )
    if background_total > 0:
        background_text = f"{bg_idx + 1}/{background_total} - {Path(str(osc_files[bg_idx])).name}"
    else:
        background_text = "Not loaded"

    try:
        cif_path = str(_current_primary_cif_path()).strip()
    except Exception:
        cif_path = ""
    cif_text = Path(cif_path).name if cif_path else "Not loaded"

    fit_background_var = globals().get("geometry_fit_background_selection_var")
    try:
        fit_background_text = " ".join(str(fit_background_var.get()).split())
    except Exception:
        fit_background_text = "current"
    if not fit_background_text:
        fit_background_text = "current"

    view_text = _current_app_shell_view_text()
    fit_quality_text = _extract_fit_quality_text()
    try:
        selected_indices = list(_current_geometry_fit_background_indices(strict=False))
    except Exception:
        selected_indices = []
    fit_selection_ready = bool(selected_indices) if background_total > 0 else False

    try:
        theta_base_value = float(_background_theta_base_for_index(bg_idx, strict_count=False))
    except Exception:
        theta_base_value = float("nan")
    try:
        theta_effective_value = float(_background_theta_for_index(bg_idx, strict_count=False))
    except Exception:
        theta_effective_value = float("nan")
    shared_theta = _geometry_fit_uses_shared_theta_offset(selected_indices)
    acquisition_text = _build_current_dataset_acquisition_text(
        theta_base_value=theta_base_value,
        theta_effective_value=theta_effective_value,
        shared_theta=shared_theta,
    )

    freshness_status, freshness_detail = _geometry_fit_cache_status(
        selected_indices=selected_indices,
        current_background_index=bg_idx,
    )
    preview_summary_text = _current_geometry_preview_gate_summary()
    fit_health_secondary_parts = [
        text
        for text in (
            preview_summary_text,
            freshness_detail if freshness_status != "Fresh" else "",
        )
        if str(text).strip()
    ]
    fit_health_secondary_text = " | ".join(fit_health_secondary_parts)
    geometry_text = _build_current_dataset_geometry_text(
        background_index=bg_idx,
        selected_indices=selected_indices,
        freshness_status=freshness_status,
    )
    simulation_text = _build_current_dataset_simulation_text(view_text=view_text)
    structure_text = _build_current_dataset_structure_text()

    manual_pair_total = 0
    try:
        manual_pairs_state = globals().get("geometry_manual_state")
        manual_pairs_map = getattr(manual_pairs_state, "pairs_by_background", {}) or {}
        for pair_list in manual_pairs_map.values():
            manual_pair_total += len(pair_list or ())
    except Exception:
        manual_pair_total = 0
    workflow_statuses = {
        "backgrounds": "ready" if background_total > 0 else "missing",
        "cif": "ready" if cif_path else "missing",
        "fit_set": "ready" if fit_selection_ready else "missing",
        "manual_pairs": "ready" if manual_pair_total > 0 else "missing",
        "geometry_fit": freshness_status,
        "analysis": (
            "ready"
            if background_total > 0 and cif_path and freshness_status == "Fresh"
            else "stale"
            if background_total > 0 and cif_path and freshness_status == "Stale"
            else "missing"
        ),
    }
    gui_views.set_app_shell_session_summary_text(
        app_shell_view_state,
        "\n".join(
            [
                f"Background: {background_text}",
                f"CIF: {cif_text}",
                f"Fit backgrounds: {fit_background_text}",
                f"View: {view_text}",
                f"Fit quality: {fit_quality_text}",
            ]
        ),
    )
    gui_views.set_app_shell_workflow_statuses(
        app_shell_view_state,
        workflow_statuses,
    )
    gui_views.set_app_shell_dataset_values(
        app_shell_view_state,
        {
            "background": background_text,
            "theta_i": acquisition_text,
            "theta": geometry_text,
            "fit": simulation_text,
            "model": structure_text,
        },
    )
    gui_views.set_app_shell_fit_health_text(
        app_shell_view_state,
        status=freshness_status,
        primary=fit_quality_text,
        secondary=fit_health_secondary_text,
    )
    gui_views.set_app_shell_view_mode(
        app_shell_view_state,
        _current_app_shell_view_mode(),
    )

    gui_views.set_match_results_text(
        app_shell_view_state,
        (
            f"{fit_quality_text}. Review overlays on the detector image and the "
            "status panel below after each fit."
        ),
    )
    refresh_selector = globals().get("_refresh_geometry_fit_background_table")
    if callable(refresh_selector):
        refresh_selector(force_rebuild=False)


def _refresh_interaction_mode_banner() -> None:
    """Refresh the visible workflow mode banner from the current interaction state."""

    title = "Ready to start"
    detail = "Load backgrounds in Setup, then use Match to choose peaks and fit geometry."

    if bool(getattr(geometry_runtime_state, "manual_pick_armed", False)):
        title = "Manual geometry picking armed"
        detail = "Click a simulated Qr group on the image to start pairing it to measured peaks."

    if bool(_geometry_manual_pick_session_active(require_current_background=False)):
        title = "Manual peak placement active"
        detail = "Click the measured peak location for the selected simulated group."
    elif bool(getattr(analysis_peak_selection_state, "pick_armed", False)):
        title = "Analysis peak picking active"
        detail = "Click peaks inside the current caked integration region to mark and fit them."
    elif bool(getattr(peak_selection_state, "hkl_pick_armed", False)):
        title = "HKL image-pick active"
        detail = "Click a simulated feature on the image to select the nearest HKL."
    elif bool(getattr(geometry_preview_state, "exclude_armed", False)):
        title = "Preview exclusion active"
        detail = "Click peaks on the image to exclude or restore them from live geometry preview."
    elif _fast_viewer_active():
        title = "Fast viewer active"
        detail = (
            "Rendering is active in the separate fast-viewer window; the embedded canvas is paused."
        )
    else:
        suspend_reason = _fast_viewer_suspend_reason()
        if bool(_fast_viewer_requested_enabled()) and isinstance(suspend_reason, str):
            title = "Fast viewer paused"
            detail = f"Using the embedded canvas because {suspend_reason}."

    gui_views.set_app_shell_mode_banner_text(
        app_shell_view_state,
        title=title,
        detail=detail,
    )


def _refresh_run_status_bar() -> None:
    phase_raw = str(simulation_runtime_state.update_phase or "ready").strip().lower()
    phase_labels = {
        "startup": "Starting",
        "queued": "Queued",
        "computing": "Computing",
        "analyzing": "Analyzing",
        "applying": "Applying",
        "ready": "Ready",
        "error": "Error",
    }
    phase_text = phase_labels.get(phase_raw, "Ready")
    if simulation_runtime_state.worker_active_job is not None:
        phase_text = "Computing"
    elif simulation_runtime_state.worker_queued_job is not None and phase_text == "Ready":
        phase_text = "Queued"
    elif simulation_runtime_state.analysis_active_job is not None:
        phase_text = "Analyzing"
    elif simulation_runtime_state.analysis_queued_job is not None and phase_text == "Ready":
        phase_text = "Queued"

    background_total = max(1, len(background_runtime_state.osc_files))
    background_index = int(background_runtime_state.current_background_index) + 1

    try:
        cif_summary = Path(_current_primary_cif_path()).name
    except Exception:
        cif_summary = "n/a"
    cif_summary = _truncate_run_status_piece(cif_summary, max_chars=26)

    try:
        optics_summary = _normalize_optics_mode_label(optics_mode_var.get())
    except Exception:
        optics_summary = "fast"

    view_summary = (
        "Caked"
        if bool(
            getattr(analysis_view_controls_view_state.show_caked_2d_var, "get", lambda: False)()
        )
        else "Detector"
    )
    if bool(getattr(analysis_view_controls_view_state.show_1d_var, "get", lambda: False)()):
        view_summary = f"{view_summary}+1D"

    parts = [
        f"State: {phase_text}",
        f"BG {background_index}/{background_total}",
        f"CIF {cif_summary}",
        f"Samples {int(max(1, simulation_runtime_state.num_samples))}",
        f"Optics {optics_summary}",
        f"View {view_summary}",
    ]
    if simulation_runtime_state.preview_active:
        preview_count = simulation_runtime_state.preview_sample_count
        if isinstance(preview_count, int) and preview_count > 0:
            parts.append(f"Preview {preview_count}")
        else:
            parts.append("Preview")
    if simulation_runtime_state.analysis_preview_active:
        bins = simulation_runtime_state.analysis_preview_bins
        if (
            isinstance(bins, tuple)
            and len(bins) == 2
            and all(isinstance(value, int) and value > 0 for value in bins)
        ):
            parts.append(f"Caked {bins[0]}x{bins[1]}")
        else:
            parts.append("Caked Preview")
    if simulation_runtime_state.interaction_drag_active:
        parts.append("Dragging")
    if _fast_viewer_active():
        parts.append("Fast Viewer")
    elif bool(_fast_viewer_requested_enabled()) and isinstance(
        _fast_viewer_suspend_reason(),
        str,
    ):
        parts.append("Fast Viewer Paused")
    last_total_ms = simulation_runtime_state.last_total_update_ms
    if isinstance(last_total_ms, (int, float)) and np.isfinite(float(last_total_ms)):
        parts.append(f"Last {float(last_total_ms):.0f} ms")
    gui_views.set_app_shell_run_status_text(
        app_shell_view_state,
        " | ".join(parts),
    )
    _refresh_session_summary_panel()
    _refresh_interaction_mode_banner()


def _clear_cached_analysis_results(*, clear_1d_lines: bool = False) -> None:
    simulation_runtime_state.last_analysis_signature = None
    simulation_runtime_state.analysis_preview_active = False
    simulation_runtime_state.analysis_preview_bins = None
    simulation_runtime_state.last_res2_sim = None
    simulation_runtime_state.last_res2_background = None
    simulation_runtime_state.last_caked_image_unscaled = None
    simulation_runtime_state.last_caked_extent = None
    simulation_runtime_state.last_caked_background_image_unscaled = None
    simulation_runtime_state.last_caked_radial_values = None
    simulation_runtime_state.last_caked_azimuth_values = None
    simulation_runtime_state.last_caked_intersection_cache = None
    simulation_runtime_state.last_q_space_image_unscaled = None
    simulation_runtime_state.last_q_space_extent = None
    simulation_runtime_state.last_q_space_background_image_unscaled = None
    simulation_runtime_state.last_q_space_qr_values = None
    simulation_runtime_state.last_q_space_qz_values = None
    simulation_runtime_state.last_1d_integration_data.update(
        {
            "radials_sim": None,
            "intensities_2theta_sim": None,
            "azimuths_sim": None,
            "intensities_azimuth_sim": None,
            "radials_bg": None,
            "intensities_2theta_bg": None,
            "azimuths_bg": None,
            "intensities_azimuth_bg": None,
        }
    )
    if clear_1d_lines:
        _clear_1d_plot_cache_and_lines()


def _invalidate_analysis_cache(*, clear_visuals: bool = False) -> None:
    _trace_live_cache_event(
        "analysis",
        "clear",
        outcome="cleared",
        clear_visuals=bool(clear_visuals),
        last_analysis_signature_summary=_live_cache_signature_summary(
            simulation_runtime_state.last_analysis_signature
        ),
        had_sim_result=bool(simulation_runtime_state.last_res2_sim is not None),
        had_background_result=bool(simulation_runtime_state.last_res2_background is not None),
    )
    simulation_runtime_state.analysis_epoch = int(simulation_runtime_state.analysis_epoch) + 1

    ready_result = simulation_runtime_state.analysis_ready_result
    if isinstance(ready_result, dict) and int(ready_result.get("epoch", -1)) < int(
        simulation_runtime_state.analysis_epoch
    ):
        _append_job_queue_trace(
            "job_discarded_as_stale",
            lane="analysis",
            job=ready_result,
            active_job=simulation_runtime_state.analysis_active_job,
            queued_job=simulation_runtime_state.analysis_queued_job,
            reason="epoch_invalidated_ready_result",
        )
        simulation_runtime_state.analysis_ready_result = None

    queued_job = simulation_runtime_state.analysis_queued_job
    if isinstance(queued_job, dict) and int(queued_job.get("epoch", -1)) < int(
        simulation_runtime_state.analysis_epoch
    ):
        _append_job_queue_trace(
            "job_discarded_as_stale",
            lane="analysis",
            job=queued_job,
            active_job=simulation_runtime_state.analysis_active_job,
            queued_job=queued_job,
            reason="epoch_invalidated_queued_job",
        )
        simulation_runtime_state.analysis_queued_job = None

    if clear_visuals:
        _clear_cached_analysis_results(clear_1d_lines=True)


def _invalidate_simulation_cache() -> None:
    _trace_live_cache_event(
        "simulation",
        "clear",
        outcome="cleared",
        last_sim_signature_summary=_live_cache_signature_summary(
            simulation_runtime_state.last_sim_signature
        ),
        last_simulation_signature_summary=_live_cache_signature_summary(
            simulation_runtime_state.last_simulation_signature
        ),
        source_snapshot_count=_live_cache_count(simulation_runtime_state.source_row_snapshots),
        preview_active=bool(simulation_runtime_state.preview_active),
        preview_sample_count=simulation_runtime_state.preview_sample_count,
    )
    simulation_runtime_state.last_sim_signature = None
    simulation_runtime_state.last_simulation_signature = None
    simulation_runtime_state.source_row_snapshots = {}
    simulation_runtime_state.simulation_epoch = int(simulation_runtime_state.simulation_epoch) + 1
    simulation_runtime_state.preview_active = False
    simulation_runtime_state.preview_sample_count = None
    _invalidate_analysis_cache(clear_visuals=True)

    ready_result = simulation_runtime_state.worker_ready_result
    if isinstance(ready_result, dict) and int(ready_result.get("epoch", -1)) < int(
        simulation_runtime_state.simulation_epoch
    ):
        _append_job_queue_trace(
            "job_discarded_as_stale",
            lane="simulation",
            job=ready_result,
            active_job=simulation_runtime_state.worker_active_job,
            queued_job=simulation_runtime_state.worker_queued_job,
            reason="epoch_invalidated_ready_result",
        )
        simulation_runtime_state.worker_ready_result = None

    queued_job = simulation_runtime_state.worker_queued_job
    if isinstance(queued_job, dict) and int(queued_job.get("epoch", -1)) < int(
        simulation_runtime_state.simulation_epoch
    ):
        _append_job_queue_trace(
            "job_discarded_as_stale",
            lane="simulation",
            job=queued_job,
            active_job=simulation_runtime_state.worker_active_job,
            queued_job=queued_job,
            reason="epoch_invalidated_queued_job",
        )
        simulation_runtime_state.worker_queued_job = None

    if simulation_runtime_state.worker_active_job is None:
        simulation_runtime_state.update_phase = "queued"
    _refresh_run_status_bar()


def _invalidate_and_schedule_update() -> None:
    if bool(getattr(simulation_runtime_state, "startup_updates_suspended", False)):
        simulation_runtime_state.pending_startup_refresh = True
        return
    _invalidate_simulation_cache()
    schedule_update()


def _preempt_simulation_update_for_background_switch() -> None:
    """Drop pending simulation work so a background switch can requeue immediately."""

    gui_controllers.clear_tk_after_token(
        root,
        simulation_runtime_state.integration_update_pending,
    )
    simulation_runtime_state.integration_update_pending = None
    gui_controllers.clear_tk_after_token(
        root,
        simulation_runtime_state.update_pending,
    )
    simulation_runtime_state.update_pending = None
    gui_controllers.clear_tk_after_token(
        root,
        simulation_runtime_state.worker_poll_token,
    )
    simulation_runtime_state.worker_poll_token = None
    gui_controllers.clear_tk_after_token(
        root,
        simulation_runtime_state.analysis_poll_token,
    )
    simulation_runtime_state.analysis_poll_token = None

    worker_future = simulation_runtime_state.worker_future
    if worker_future is not None:
        try:
            worker_future.cancel()
        except Exception:
            pass
    analysis_future = simulation_runtime_state.analysis_future
    if analysis_future is not None:
        try:
            analysis_future.cancel()
        except Exception:
            pass

    worker_executor = simulation_runtime_state.worker_executor
    analysis_executor = simulation_runtime_state.analysis_executor
    simulation_runtime_state.worker_executor = None
    simulation_runtime_state.worker_future = None
    simulation_runtime_state.worker_active_job = None
    simulation_runtime_state.worker_queued_job = None
    simulation_runtime_state.worker_ready_result = None
    simulation_runtime_state.analysis_executor = None
    simulation_runtime_state.analysis_future = None
    simulation_runtime_state.analysis_active_job = None
    simulation_runtime_state.analysis_queued_job = None
    simulation_runtime_state.analysis_ready_result = None

    if worker_executor is not None:
        try:
            worker_executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            worker_executor.shutdown(wait=False)
        except Exception:
            pass
    if analysis_executor is not None:
        try:
            analysis_executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            analysis_executor.shutdown(wait=False)
        except Exception:
            pass

    _invalidate_simulation_cache()


def _ensure_simulation_worker_executor():
    executor = simulation_runtime_state.worker_executor
    if executor is None:
        executor = concurrent.futures.ThreadPoolExecutor(
            # Keep GUI simulation updates in latest-request-wins mode:
            # one active job plus at most one replacement queued in runtime state.
            # The executor can still expose the full reserved CPU budget because
            # runtime state only submits one active job at a time.
            max_workers=default_reserved_cpu_worker_count(),
            thread_name_prefix="ra-sim-sim",
        )
        simulation_runtime_state.worker_executor = executor
    return executor


def _ensure_analysis_worker_executor():
    executor = simulation_runtime_state.analysis_executor
    if executor is None:
        executor = concurrent.futures.ThreadPoolExecutor(
            # Keep GUI analysis updates in latest-request-wins mode:
            # one active job plus at most one replacement queued in runtime state.
            # The executor can still expose the full reserved CPU budget because
            # runtime state only submits one active job at a time.
            max_workers=default_reserved_cpu_worker_count(),
            thread_name_prefix="ra-sim-analysis",
        )
        simulation_runtime_state.analysis_executor = executor
    return executor


def _replace_queued_simulation_job(job: dict[str, object]) -> None:
    """Keep only the newest queued simulation replacement job."""

    previous_job = simulation_runtime_state.worker_queued_job
    if isinstance(previous_job, dict):
        _append_job_queue_trace(
            "job_replaced",
            lane="simulation",
            job=job,
            active_job=simulation_runtime_state.worker_active_job,
            queued_job=previous_job,
            replaced_job=previous_job,
            reason="latest_request_wins",
        )
    simulation_runtime_state.worker_queued_job = dict(job)


def _replace_queued_analysis_job(job: dict[str, object]) -> None:
    """Keep only the newest queued analysis replacement job."""

    previous_job = simulation_runtime_state.analysis_queued_job
    if isinstance(previous_job, dict):
        _append_job_queue_trace(
            "job_replaced",
            lane="analysis",
            job=job,
            active_job=simulation_runtime_state.analysis_active_job,
            queued_job=previous_job,
            replaced_job=previous_job,
            reason="latest_request_wins",
        )
    simulation_runtime_state.analysis_queued_job = dict(job)


def _promote_queued_simulation_job(*, reason: str) -> bool:
    """Promote the queued simulation replacement job into the active slot."""

    queued_job = simulation_runtime_state.worker_queued_job
    if not isinstance(queued_job, dict):
        return False
    simulation_runtime_state.worker_queued_job = None
    _append_job_queue_trace(
        "job_promoted_from_queued",
        lane="simulation",
        job=queued_job,
        active_job=simulation_runtime_state.worker_active_job,
        queued_job=None,
        reason=reason,
    )
    _append_job_queue_trace(
        "job_submitted",
        lane="simulation",
        job=queued_job,
        active_job=simulation_runtime_state.worker_active_job,
        queued_job=None,
        reason="promoted_from_queued",
    )
    _submit_async_simulation_job(queued_job)
    return True


def _promote_queued_analysis_job(*, reason: str) -> bool:
    """Promote the queued analysis replacement job into the active slot."""

    queued_job = simulation_runtime_state.analysis_queued_job
    if not isinstance(queued_job, dict):
        return False
    simulation_runtime_state.analysis_queued_job = None
    _append_job_queue_trace(
        "job_promoted_from_queued",
        lane="analysis",
        job=queued_job,
        active_job=simulation_runtime_state.analysis_active_job,
        queued_job=None,
        reason=reason,
    )
    _append_job_queue_trace(
        "job_submitted",
        lane="analysis",
        job=queued_job,
        active_job=simulation_runtime_state.analysis_active_job,
        queued_job=None,
        reason="promoted_from_queued",
    )
    _submit_async_analysis_job(queued_job)
    return True


def _empty_caking_cache() -> dict[str, object]:
    return {
        "sim_results": {},
        "bg_results": {},
    }


def _caking_cache_bucket(kind: str) -> dict[object, dict[str, object]]:
    cache = simulation_runtime_state.caking_cache
    if not isinstance(cache, dict):
        cache = _empty_caking_cache()
        simulation_runtime_state.caking_cache = cache
    bucket_name = f"{kind}_results"
    bucket = cache.get(bucket_name)
    if not isinstance(bucket, dict):
        bucket = {}
        cache[bucket_name] = bucket
    return bucket


def _get_cached_caking_entry(kind: str, signature: object) -> dict[str, object] | None:
    if signature is None:
        _trace_live_cache_event(
            f"caking_{str(kind)}",
            "lookup",
            outcome="skipped",
            reason="missing_signature",
        )
        return None
    if not _retain_runtime_optional_cache("caking", feature_needed=True):
        simulation_runtime_state.caking_cache = _empty_caking_cache()
        _trace_live_cache_event(
            f"caking_{str(kind)}",
            "lookup",
            outcome="miss",
            reason="retention_disabled",
            signature_summary=_live_cache_signature_summary(signature),
        )
        return None
    bucket = _caking_cache_bucket(kind)
    entry = bucket.get(signature)
    if not isinstance(entry, dict):
        _trace_live_cache_event(
            f"caking_{str(kind)}",
            "lookup",
            outcome="miss",
            signature_summary=_live_cache_signature_summary(signature),
            bucket_size=_live_cache_count(bucket),
        )
        return None
    bucket.pop(signature, None)
    bucket[signature] = entry
    _trace_live_cache_event(
        f"caking_{str(kind)}",
        "lookup",
        outcome="hit",
        signature_summary=_live_cache_signature_summary(signature),
        bucket_size=_live_cache_count(bucket),
        payload_available=bool(entry.get("payload") is not None),
    )
    return dict(entry)


def _store_cached_caking_entry(
    kind: str,
    signature: object,
    *,
    res2,
    payload: dict[str, object] | None,
) -> None:
    if signature is None or res2 is None:
        _trace_live_cache_event(
            f"caking_{str(kind)}",
            "store",
            outcome="skipped",
            reason=("missing_signature" if signature is None else "missing_result"),
        )
        return
    if not _retain_runtime_optional_cache("caking", feature_needed=True):
        simulation_runtime_state.caking_cache = _empty_caking_cache()
        _trace_live_cache_event(
            f"caking_{str(kind)}",
            "store",
            outcome="discarded",
            reason="retention_disabled",
            signature_summary=_live_cache_signature_summary(signature),
            payload_available=bool(payload is not None),
        )
        return
    bucket = _caking_cache_bucket(kind)
    bucket.pop(signature, None)
    bucket[signature] = {
        "res2": res2,
        "payload": None if payload is None else dict(payload),
    }
    evicted = False
    while len(bucket) > CAKING_CACHE_MAX_ENTRIES:
        oldest_key = next(iter(bucket))
        bucket.pop(oldest_key, None)
        evicted = True
    _trace_live_cache_event(
        f"caking_{str(kind)}",
        "store",
        outcome="stored",
        signature_summary=_live_cache_signature_summary(signature),
        bucket_size=_live_cache_count(bucket),
        evicted=bool(evicted),
        payload_available=bool(payload is not None),
    )


def _live_interaction_active() -> bool:
    return bool(simulation_runtime_state.interaction_drag_active)


def _current_preview_sample_limit() -> int:
    return (
        LIVE_DRAG_PREVIEW_MAX_SAMPLES if _live_interaction_active() else INITIAL_PREVIEW_MAX_SAMPLES
    )


def _current_update_debounce_ms() -> int:
    return PREVIEW_UPDATE_DEBOUNCE_MS if _live_interaction_active() else FULL_UPDATE_DEBOUNCE_MS


def _simulation_result_matches_signature(
    result: object,
    signature: object,
) -> bool:
    return bool(
        isinstance(result, dict)
        and result.get("signature") == signature
        and int(result.get("epoch", -1)) == int(simulation_runtime_state.simulation_epoch)
    )


def _simulation_result_superseded_by_queued_job(
    result: object,
    queued_job: object,
) -> bool:
    if not isinstance(result, dict) or not isinstance(queued_job, dict):
        return False
    if int(queued_job.get("job_id", -1)) <= int(result.get("job_id", -1)):
        return False
    queued_epoch = int(queued_job.get("epoch", -1))
    result_epoch = int(result.get("epoch", -1))
    if queued_epoch != result_epoch:
        return queued_epoch > result_epoch
    if queued_job.get("signature") != result.get("signature"):
        return True
    return str(queued_job.get("job_kind", "full")) == str(result.get("job_kind", "full"))


def _analysis_progress_text() -> str:
    active_job = simulation_runtime_state.analysis_active_job
    if not isinstance(active_job, dict):
        active_job = simulation_runtime_state.analysis_queued_job
    if isinstance(active_job, dict) and bool(active_job.get("is_preview", False)):
        return "Updating low-res caked preview..."
    return "Updating caked integration in background..."


def _defer_nonessential_redraw() -> bool:
    return bool(
        _live_interaction_active()
        or simulation_runtime_state.preview_active
        or simulation_runtime_state.analysis_preview_active
        or simulation_runtime_state.worker_active_job is not None
        or simulation_runtime_state.worker_queued_job is not None
        or simulation_runtime_state.analysis_active_job is not None
        or simulation_runtime_state.analysis_queued_job is not None
    )


def _refresh_settled_overlays() -> None:
    if _resolved_primary_analysis_display_mode() == "q_space":
        try:
            integration_region_overlay.set_visible(False)
        except Exception:
            pass
        _clear_deferred_overlays(clear_qr_overlay=True)
        _clear_all_geometry_overlay_artists(redraw=False)
        _clear_analysis_peak_overlay_artists(redraw=False)
        _request_overlay_canvas_redraw(force=True)
        return
    if _current_geometry_fit_caked_roi_preview_enabled():
        _hide_geometry_fit_caked_roi_preview_aux_artists()
        _request_main_canvas_redraw(force_matplotlib=False)
        _request_overlay_canvas_redraw(force=True)
        return
    if not _geometry_overlays_enabled():
        gui_controllers.clear_geometry_preview_skip_once(geometry_preview_state)
        _clear_all_geometry_overlay_artists(redraw=True)
        return
    refresh_integration_region_visuals()
    qr_cylinder_overlay_runtime_refresh(redraw=True, update_status=False)
    if _live_geometry_preview_enabled():
        if gui_controllers.consume_geometry_preview_skip_once(geometry_preview_state):
            _clear_geometry_preview_artists()
        else:
            _refresh_live_geometry_preview(update_status=True)
    else:
        gui_controllers.clear_geometry_preview_skip_once(geometry_preview_state)
        _clear_geometry_preview_artists()
    if not gui_geometry_fit.redraw_runtime_geometry_fit_overlay_state(
        _geometry_fit_last_overlay_state(),
        draw_overlay_records=_draw_runtime_geometry_fit_overlay_records,
        draw_initial_pairs_overlay=_draw_runtime_geometry_fit_initial_pairs_overlay,
    ):
        _render_current_geometry_manual_pairs(update_status=False)


def _clear_deferred_overlays(*, clear_qr_overlay: bool = True) -> None:
    _clear_geometry_pick_artists(redraw=False)
    if clear_qr_overlay:
        try:
            gui_qr_cylinder_overlay.clear_runtime_qr_cylinder_overlay_artists(
                qr_cylinder_overlay_runtime_bindings_factory(),
                redraw=False,
            )
        except Exception:
            pass
    _clear_geometry_preview_artists(redraw=False)
    gui_controllers.clear_geometry_preview_skip_once(geometry_preview_state)


def _finish_live_interaction() -> None:
    simulation_runtime_state.interaction_settle_token = None
    if _commit_legacy_main_matplotlib_preview_view():
        return
    if not simulation_runtime_state.interaction_drag_active:
        return
    simulation_runtime_state.interaction_drag_active = False
    should_schedule = bool(simulation_runtime_state.interaction_drag_requires_settled_update)
    simulation_runtime_state.interaction_drag_requires_settled_update = False
    _refresh_run_status_bar()
    if should_schedule:
        simulation_runtime_state.main_matplotlib_overlays_suspended = False
        schedule_update()
        return
    _restore_legacy_main_matplotlib_overlays()


def _begin_main_figure_live_interaction() -> None:
    if not _legacy_main_matplotlib_interaction_active():
        return
    _mark_live_interaction_start()
    _suspend_legacy_main_matplotlib_overlays()


def _touch_main_figure_live_interaction() -> None:
    if not _legacy_main_matplotlib_interaction_active():
        return
    _mark_live_interaction_start()
    _suspend_legacy_main_matplotlib_overlays()


def _end_main_figure_live_interaction() -> None:
    if not _legacy_main_matplotlib_interaction_active():
        return
    _mark_live_interaction_release()


def _mark_live_interaction_start(_event=None) -> None:
    gui_controllers.clear_tk_after_token(
        root,
        simulation_runtime_state.interaction_settle_token,
    )
    simulation_runtime_state.interaction_settle_token = None
    if simulation_runtime_state.interaction_drag_active:
        return
    simulation_runtime_state.interaction_drag_active = True
    simulation_runtime_state.interaction_drag_requires_settled_update = False
    _refresh_run_status_bar()


def _mark_live_interaction_release(_event=None) -> None:
    if not simulation_runtime_state.interaction_drag_active:
        return
    gui_controllers.clear_tk_after_token(
        root,
        simulation_runtime_state.interaction_settle_token,
    )
    simulation_runtime_state.interaction_settle_token = root.after(
        LIVE_DRAG_SETTLE_MS,
        _finish_live_interaction,
    )


root.bind_class("TScale", "<ButtonPress-1>", _mark_live_interaction_start, add="+")
root.bind_class("TScale", "<ButtonRelease-1>", _mark_live_interaction_release, add="+")
root.bind_class("Scale", "<ButtonPress-1>", _mark_live_interaction_start, add="+")
root.bind_class("Scale", "<ButtonRelease-1>", _mark_live_interaction_release, add="+")


def _preview_sample_indices(sample_count: int, *, max_samples: int) -> np.ndarray:
    total = max(0, int(sample_count))
    target = max(1, int(max_samples))
    if total <= target:
        return np.arange(total, dtype=np.int64)
    return np.unique(np.rint(np.linspace(0, total - 1, target)).astype(np.int64, copy=False))


def _build_preview_simulation_job(
    job: dict[str, object],
    *,
    max_samples: int = INITIAL_PREVIEW_MAX_SAMPLES,
) -> dict[str, object] | None:
    if not PREVIEW_CALCULATIONS_ENABLED:
        return None
    if not isinstance(job, dict):
        return None
    mosaic_params = job.get("mosaic_params")
    if not isinstance(mosaic_params, dict):
        return None

    beam_x = np.asarray(mosaic_params.get("beam_x_array", []), dtype=np.float64).reshape(-1)
    beam_y = np.asarray(mosaic_params.get("beam_y_array", []), dtype=np.float64).reshape(-1)
    theta = np.asarray(mosaic_params.get("theta_array", []), dtype=np.float64).reshape(-1)
    phi = np.asarray(mosaic_params.get("phi_array", []), dtype=np.float64).reshape(-1)
    wavelength = np.asarray(
        mosaic_params.get("wavelength_array", []),
        dtype=np.float64,
    ).reshape(-1)
    sample_count = min(
        beam_x.size,
        beam_y.size,
        theta.size,
        phi.size,
        wavelength.size,
    )
    if sample_count <= 0 or sample_count <= int(max_samples):
        return None

    preview_indices = _preview_sample_indices(sample_count, max_samples=int(max_samples))
    preview_job = dict(job)
    preview_job["collect_hit_tables"] = bool(
        job.get("collect_hit_tables", False) and not bool(job.get("accumulate_image", True))
    )
    preview_job["collect_primary_hit_tables"] = False
    preview_job["collect_secondary_hit_tables"] = False
    preview_job["capture_primary_hit_tables_raw"] = False
    preview_job["job_kind"] = "preview"
    preview_job["is_preview"] = True
    preview_job["preview_sample_count"] = int(preview_indices.size)
    preview_job["mosaic_params"] = {
        **mosaic_params,
        "beam_x_array": beam_x[preview_indices].copy(),
        "beam_y_array": beam_y[preview_indices].copy(),
        "theta_array": theta[preview_indices].copy(),
        "phi_array": phi[preview_indices].copy(),
        "wavelength_array": wavelength[preview_indices].copy(),
        "sample_weights": (
            None
            if mosaic_params.get("sample_weights") is None
            else np.asarray(
                mosaic_params["sample_weights"],
                dtype=np.float64,
            )
            .reshape(-1)[preview_indices]
            .copy()
        ),
        "n2_sample_array": (
            None
            if mosaic_params.get("n2_sample_array") is None
            else np.asarray(
                mosaic_params["n2_sample_array"],
                dtype=np.complex128,
            )
            .reshape(-1)[preview_indices]
            .copy()
        ),
    }
    return preview_job


def _run_simulation_generation_job(job: dict[str, object]) -> dict[str, object]:
    from ra_sim.simulation.projection_debug import finalize_projection_debug_session
    from ra_sim.simulation.projection_debug import start_projection_debug_session

    image_size = int(job["image_size"])
    pixel_size_m_job = float(job["pixel_size_m"])
    center = np.asarray(job["center"], dtype=np.float64)
    request_unit_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    request_n_detector = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    mosaic_params_job = dict(job["mosaic_params"])
    job_kind = str(job.get("job_kind", "full"))
    image_generation_start_time = perf_counter()

    if bool(job.get("qr_cylinder_replace_simulation", False)):
        blank = np.zeros((image_size, image_size), dtype=np.float64)
        return {
            "job_id": int(job["job_id"]),
            "job_kind": job_kind,
            "signature": job["signature"],
            "epoch": int(job["epoch"]),
            "is_preview": bool(job.get("is_preview", False)),
            "preview_sample_count": (
                int(job.get("preview_sample_count", 0))
                if job.get("preview_sample_count") is not None
                else None
            ),
            "primary_image": blank.copy(),
            "secondary_image": blank.copy(),
            "primary_max_positions": [],
            "secondary_max_positions": [],
            "primary_intersection_cache": [],
            "secondary_intersection_cache": [],
            "primary_peak_table_lattice": [],
            "secondary_peak_table_lattice": [],
            "primary_contribution_cache_signature": job.get("primary_contribution_cache_signature"),
            "primary_source_mode": job.get("primary_source_mode"),
            "primary_contribution_keys": list(job.get("primary_contribution_keys", [])),
            "active_primary_contribution_keys": list(
                job.get(
                    "active_primary_contribution_keys",
                    job.get("primary_contribution_keys", []),
                )
            ),
            "primary_hit_tables_raw": [],
            "projection_debug_log_path": None,
            "image_generation_elapsed_ms": (perf_counter() - image_generation_start_time) * 1e3,
        }

    projection_debug_session = start_projection_debug_session(
        {
            "job_id": int(job["job_id"]),
            "job_kind": job_kind,
            "signature": job["signature"],
            "is_preview": bool(job.get("is_preview", False)),
            "theta_i_deg": float(job["theta_initial_deg"]),
            "optics_mode": int(job["optics_mode"]),
            "exit_projection_mode": str(job.get("exit_projection_mode", "internal")),
        },
        source="simulation_generation_job",
    )

    def build_request(
        miller_arr,
        intens_vals,
        *,
        a_val,
        c_val,
        image_buffer,
        collect_hit_tables,
    ) -> SimulationRequest:
        return SimulationRequest(
            miller=np.asarray(miller_arr, dtype=np.float64),
            intensities=np.asarray(intens_vals, dtype=np.float64).reshape(-1),
            geometry=DetectorGeometry(
                image_size=image_size,
                av=float(a_val),
                cv=float(c_val),
                lambda_angstrom=float(job["lambda_value"]),
                distance_m=float(job["distance_m"]),
                gamma_deg=float(job["gamma_deg"]),
                Gamma_deg=float(job["Gamma_deg"]),
                chi_deg=float(job["chi_deg"]),
                psi_deg=float(job["psi_deg"]),
                psi_z_deg=float(job["psi_z_deg"]),
                zs=float(job["zs"]),
                zb=float(job["zb"]),
                center=center,
                theta_initial_deg=float(job["theta_initial_deg"]),
                cor_angle_deg=float(job["cor_angle_deg"]),
                unit_x=request_unit_x,
                n_detector=request_n_detector,
                pixel_size_m=pixel_size_m_job,
                sample_width_m=float(job["sample_width_m"]),
                sample_length_m=float(job["sample_length_m"]),
            ),
            beam=BeamSamples(
                beam_x_array=np.asarray(mosaic_params_job["beam_x_array"], dtype=np.float64),
                beam_y_array=np.asarray(mosaic_params_job["beam_y_array"], dtype=np.float64),
                theta_array=np.asarray(mosaic_params_job["theta_array"], dtype=np.float64),
                phi_array=np.asarray(mosaic_params_job["phi_array"], dtype=np.float64),
                wavelength_array=np.asarray(
                    mosaic_params_job["wavelength_array"],
                    dtype=np.float64,
                ),
                sample_weights=(
                    None
                    if mosaic_params_job.get("sample_weights") is None
                    else np.asarray(
                        mosaic_params_job["sample_weights"],
                        dtype=np.float64,
                    )
                ),
                n2_sample_array=(
                    None
                    if mosaic_params_job.get("n2_sample_array") is None
                    else np.asarray(
                        mosaic_params_job["n2_sample_array"],
                        dtype=np.complex128,
                    )
                ),
            ),
            mosaic=MosaicParams(
                sigma_mosaic_deg=float(mosaic_params_job["sigma_mosaic_deg"]),
                gamma_mosaic_deg=float(mosaic_params_job["gamma_mosaic_deg"]),
                eta=float(mosaic_params_job["eta"]),
                solve_q_steps=int(mosaic_params_job["solve_q_steps"]),
                solve_q_rel_tol=float(mosaic_params_job["solve_q_rel_tol"]),
                solve_q_mode=int(mosaic_params_job["solve_q_mode"]),
            ),
            debye_waller=DebyeWallerParams(
                x=float(job["debye_x"]),
                y=float(job["debye_y"]),
            ),
            n2=job["n2_value"],
            image_buffer=image_buffer,
            save_flag=0,
            thickness=float(job["sample_depth_m"]),
            optics_mode=int(job["optics_mode"]),
            collect_hit_tables=bool(collect_hit_tables),
            accumulate_image=bool(job.get("accumulate_image", True)),
            exit_projection_mode=str(job.get("exit_projection_mode", "internal")),
        )

    def run_one(
        data,
        intens_arr,
        a_val,
        c_val,
        *,
        collect_hit_tables: bool,
        capture_raw_hit_tables: bool,
    ):
        buf = np.zeros((image_size, image_size), dtype=np.float64)
        request_collect_hit_tables = bool(collect_hit_tables or capture_raw_hit_tables)
        if isinstance(data, dict):
            if len(data) == 0:
                return buf, [], []
            result = simulate_qr_rods_request(
                data,
                build_request(
                    np.empty((0, 3), dtype=np.float64),
                    np.empty(0, dtype=np.float64),
                    a_val=a_val,
                    c_val=c_val,
                    image_buffer=buf,
                    collect_hit_tables=request_collect_hit_tables,
                ),
            )
            return (
                result.image,
                _copy_hit_tables(result.hit_tables if request_collect_hit_tables else []),
                _copy_intersection_cache_tables(result.intersection_cache),
            )

        miller_arr = np.asarray(data, dtype=np.float64)
        intens_vals = np.asarray(intens_arr, dtype=np.float64).reshape(-1)
        if miller_arr.ndim != 2 or miller_arr.shape[1] < 3:
            return buf, [], []
        row_count = min(miller_arr.shape[0], intens_vals.shape[0])
        if row_count <= 0:
            return buf, [], []
        result = simulate_request(
            build_request(
                miller_arr[:row_count, :],
                intens_vals[:row_count],
                a_val=a_val,
                c_val=c_val,
                image_buffer=buf,
                collect_hit_tables=request_collect_hit_tables,
            )
        )
        return (
            result.image,
            _copy_hit_tables(result.hit_tables if request_collect_hit_tables else []),
            _copy_intersection_cache_tables(result.intersection_cache),
        )

    primary_data = job["primary_data"]
    primary_intensities = job["primary_intensities"]
    secondary_data = job["secondary_data"]
    secondary_intensities = job["secondary_intensities"]
    primary_collect_hit_tables = bool(
        job.get("collect_primary_hit_tables", job["collect_hit_tables"])
    )
    secondary_collect_hit_tables = bool(
        job.get("collect_secondary_hit_tables", job["collect_hit_tables"])
    )
    capture_primary_hit_tables_raw = bool(job.get("capture_primary_hit_tables_raw", False))

    img1 = np.zeros((image_size, image_size), dtype=np.float64)
    img2 = np.zeros((image_size, image_size), dtype=np.float64)
    raw_hit_tables1: list[object] = []
    raw_hit_tables2: list[object] = []
    cache1: list[object] = []
    cache2: list[object] = []

    try:
        if bool(job["run_primary"]):
            img1, raw_hit_tables1, cache1 = run_one(
                primary_data,
                primary_intensities,
                float(job["a_primary"]),
                float(job["c_primary"]),
                collect_hit_tables=primary_collect_hit_tables,
                capture_raw_hit_tables=capture_primary_hit_tables_raw,
            )

        if bool(job["run_secondary"]):
            img2, raw_hit_tables2, cache2 = run_one(
                secondary_data,
                secondary_intensities,
                float(job["a_secondary"]),
                float(job["c_secondary"]),
                collect_hit_tables=secondary_collect_hit_tables,
                capture_raw_hit_tables=False,
            )

        primary_peak_tables = _resolved_peak_table_payload(cache1, raw_hit_tables1)
        secondary_peak_tables = _resolved_peak_table_payload(cache2, raw_hit_tables2)
        primary_peak_table_count = len(primary_peak_tables)
        secondary_peak_table_count = len(secondary_peak_tables)

        if not bool(job.get("accumulate_image", True)):
            img1 = gui_runtime_position_preview.build_peak_position_preview_image(
                primary_peak_tables,
                image_size=image_size,
                hit_tables_to_max_positions=hit_tables_to_max_positions,
            )
            img2 = gui_runtime_position_preview.build_peak_position_preview_image(
                secondary_peak_tables,
                image_size=image_size,
                hit_tables_to_max_positions=hit_tables_to_max_positions,
            )

        projection_debug_log_path = finalize_projection_debug_session(projection_debug_session)
        return {
            "job_id": int(job["job_id"]),
            "job_kind": job_kind,
            "signature": job["signature"],
            "epoch": int(job["epoch"]),
            "is_preview": bool(job.get("is_preview", False)),
            "preview_sample_count": (
                int(job.get("preview_sample_count", 0))
                if job.get("preview_sample_count") is not None
                else None
            ),
            "primary_image": img1,
            "secondary_image": img2,
            "collected_hit_tables": bool(job["collect_hit_tables"]),
            "hit_table_signature": job.get("hit_table_signature"),
            "primary_contribution_cache_signature": job.get("primary_contribution_cache_signature"),
            "primary_source_mode": job.get("primary_source_mode"),
            "primary_contribution_keys": list(job.get("primary_contribution_keys", [])),
            "active_primary_contribution_keys": list(
                job.get(
                    "active_primary_contribution_keys",
                    job.get("primary_contribution_keys", []),
                )
            ),
            "primary_hit_tables_raw": _copy_hit_tables(
                raw_hit_tables1 if capture_primary_hit_tables_raw else []
            ),
            "primary_max_positions": list(primary_peak_tables),
            "secondary_max_positions": list(secondary_peak_tables),
            "primary_intersection_cache": _copy_intersection_cache_tables(cache1),
            "secondary_intersection_cache": _copy_intersection_cache_tables(cache2),
            "primary_peak_table_lattice": [
                (float(job["a_primary"]), float(job["c_primary"]), "primary")
                for _ in range(primary_peak_table_count)
            ],
            "secondary_peak_table_lattice": [
                (float(job["a_secondary"]), float(job["c_secondary"]), "secondary")
                for _ in range(secondary_peak_table_count)
            ],
            "projection_debug_log_path": projection_debug_log_path,
            "image_generation_elapsed_ms": (perf_counter() - image_generation_start_time) * 1e3,
        }
    except Exception:
        finalize_projection_debug_session(projection_debug_session)
        raise


def _submit_async_simulation_job(job: dict[str, object]) -> None:
    simulation_runtime_state.worker_active_job = job
    simulation_runtime_state.worker_future = _ensure_simulation_worker_executor().submit(
        _run_simulation_generation_job,
        dict(job),
    )
    simulation_runtime_state.worker_error_text = None
    simulation_runtime_state.update_phase = "computing"
    _append_job_queue_trace(
        "job_started",
        lane="simulation",
        job=job,
        active_job=job,
        queued_job=simulation_runtime_state.worker_queued_job,
    )
    _refresh_run_status_bar()
    if simulation_runtime_state.worker_poll_token is None:
        simulation_runtime_state.worker_poll_token = root.after(
            SIMULATION_WORKER_POLL_MS,
            _poll_async_simulation_job,
        )


def _poll_async_simulation_job() -> None:
    simulation_runtime_state.worker_poll_token = None
    future = simulation_runtime_state.worker_future
    active_job = simulation_runtime_state.worker_active_job
    if future is None or active_job is None:
        if _promote_queued_simulation_job(reason="active_slot_available"):
            return
        _refresh_run_status_bar()
        return

    if not future.done():
        simulation_runtime_state.worker_poll_token = root.after(
            SIMULATION_WORKER_POLL_MS,
            _poll_async_simulation_job,
        )
        return

    simulation_runtime_state.worker_future = None
    simulation_runtime_state.worker_active_job = None
    queued_job = simulation_runtime_state.worker_queued_job
    try:
        result = future.result()
    except Exception as exc:
        _append_job_queue_trace(
            "job_finished",
            lane="simulation",
            job=active_job,
            active_job=None,
            queued_job=queued_job,
            outcome="error",
            error=exc,
        )
        simulation_runtime_state.worker_error_text = str(exc)
        simulation_runtime_state.update_phase = "error"
        if "progress_label" in globals() and progress_label is not None:
            progress_label.config(text=f"Simulation update failed: {exc}")
        _refresh_run_status_bar()
        if _promote_queued_simulation_job(reason="previous_job_failed"):
            return
        return

    _append_job_queue_trace(
        "job_finished",
        lane="simulation",
        job=result,
        active_job=None,
        queued_job=queued_job,
        outcome="success",
    )
    latest_epoch = int(simulation_runtime_state.simulation_epoch)
    result_epoch = int(result.get("epoch", -1))
    superseded = _simulation_result_superseded_by_queued_job(result, queued_job)
    if result_epoch != latest_epoch or superseded:
        _append_job_queue_trace(
            "job_discarded_as_stale",
            lane="simulation",
            job=result,
            active_job=None,
            queued_job=queued_job,
            reason=("superseded_by_newer_queued_job" if superseded else "result_epoch_mismatch"),
        )
        if _promote_queued_simulation_job(reason="previous_job_stale"):
            return
        else:
            simulation_runtime_state.update_phase = "queued"
            _refresh_run_status_bar()
        return

    simulation_runtime_state.worker_ready_result = result
    simulation_runtime_state.last_image_generation_ms = float(
        result.get("image_generation_elapsed_ms", float("nan"))
    )
    simulation_runtime_state.update_phase = "applying"
    _refresh_run_status_bar()
    root.after_idle(do_update)


def _request_async_simulation_job(job: dict[str, object]) -> str:
    requested_key = _worker_job_key(job)
    _append_job_queue_trace(
        "job_requested",
        lane="simulation",
        job=job,
        active_job=simulation_runtime_state.worker_active_job,
        queued_job=simulation_runtime_state.worker_queued_job,
    )
    ready_result = simulation_runtime_state.worker_ready_result
    if _worker_job_key(ready_result) == requested_key:
        simulation_runtime_state.update_phase = "applying"
        _refresh_run_status_bar()
        return "ready"
    if isinstance(ready_result, dict) and not _simulation_result_matches_signature(
        ready_result,
        job.get("signature"),
    ):
        _append_job_queue_trace(
            "job_discarded_as_stale",
            lane="simulation",
            job=ready_result,
            active_job=simulation_runtime_state.worker_active_job,
            queued_job=simulation_runtime_state.worker_queued_job,
            reason="superseded_by_new_request",
        )
        simulation_runtime_state.worker_ready_result = None

    if _worker_job_key(simulation_runtime_state.worker_active_job) == requested_key:
        simulation_runtime_state.update_phase = "computing"
        _refresh_run_status_bar()
        return "running"

    if _worker_job_key(simulation_runtime_state.worker_queued_job) == requested_key:
        _append_job_queue_trace(
            "job_queued",
            lane="simulation",
            job=simulation_runtime_state.worker_queued_job,
            active_job=simulation_runtime_state.worker_active_job,
            queued_job=simulation_runtime_state.worker_queued_job,
            reason="already_queued",
        )
        simulation_runtime_state.update_phase = "queued"
        _refresh_run_status_bar()
        return "queued"

    simulation_runtime_state.worker_job_counter = (
        int(simulation_runtime_state.worker_job_counter) + 1
    )
    queued_job = dict(job)
    queued_job["job_id"] = int(simulation_runtime_state.worker_job_counter)
    simulation_runtime_state.worker_error_text = None

    if (
        simulation_runtime_state.worker_active_job is None
        and simulation_runtime_state.worker_future is None
    ):
        _append_job_queue_trace(
            "job_submitted",
            lane="simulation",
            job=queued_job,
            active_job=None,
            queued_job=simulation_runtime_state.worker_queued_job,
            reason="requested",
        )
        _submit_async_simulation_job(queued_job)
        return "submitted"

    # Latest-request-wins: keep one active job and one replacement queued job.
    _replace_queued_simulation_job(queued_job)
    _append_job_queue_trace(
        "job_queued",
        lane="simulation",
        job=queued_job,
        active_job=simulation_runtime_state.worker_active_job,
        queued_job=simulation_runtime_state.worker_queued_job,
        reason="awaiting_active_job",
    )
    simulation_runtime_state.update_phase = "queued"
    _refresh_run_status_bar()
    return "queued"


def _consume_ready_simulation_result(signature: object) -> dict[str, object] | None:
    ready_result = simulation_runtime_state.worker_ready_result
    if not _simulation_result_matches_signature(ready_result, signature):
        if isinstance(ready_result, dict) and int(ready_result.get("epoch", -1)) < int(
            simulation_runtime_state.simulation_epoch
        ):
            simulation_runtime_state.worker_ready_result = None
        return None

    consumed_result = dict(ready_result)
    simulation_runtime_state.worker_ready_result = None
    if (
        simulation_runtime_state.worker_future is None
        and simulation_runtime_state.worker_active_job is None
    ):
        _promote_queued_simulation_job(reason="previous_ready_result_consumed")
    return consumed_result


def _apply_ready_simulation_result(result: dict[str, object]) -> None:
    simulation_runtime_state.stored_primary_sim_image = np.asarray(
        result.get("primary_image"),
        dtype=np.float64,
    )
    simulation_runtime_state.stored_secondary_sim_image = np.asarray(
        result.get("secondary_image"),
        dtype=np.float64,
    )
    simulation_runtime_state.stored_primary_intersection_cache = _copy_intersection_cache_tables(
        result.get("primary_intersection_cache", [])
    )
    simulation_runtime_state.stored_secondary_intersection_cache = _copy_intersection_cache_tables(
        result.get("secondary_intersection_cache", [])
    )
    resolved_primary_peak_tables = _resolved_peak_table_payload(
        simulation_runtime_state.stored_primary_intersection_cache,
        result.get("primary_max_positions", []),
    )
    resolved_secondary_peak_tables = _resolved_peak_table_payload(
        simulation_runtime_state.stored_secondary_intersection_cache,
        result.get("secondary_max_positions", []),
    )
    primary_source_reflection_indices, secondary_source_reflection_indices = (
        gui_geometry_q_group_manager.audited_full_order_source_reflection_index_groups(
            (
                resolved_primary_peak_tables,
                resolved_secondary_peak_tables,
            ),
            owner="runtime_session.store_primary_secondary_peak_tables",
        )
    )
    simulation_runtime_state.stored_primary_peak_table_lattice = list(
        result.get("primary_peak_table_lattice", [])
    )
    simulation_runtime_state.stored_secondary_peak_table_lattice = list(
        result.get("secondary_peak_table_lattice", [])
    )
    simulation_runtime_state.stored_primary_source_reflection_indices = (
        primary_source_reflection_indices
    )
    simulation_runtime_state.stored_secondary_source_reflection_indices = (
        secondary_source_reflection_indices
    )
    if (
        bool(result.get("collected_hit_tables", False))
        or "primary_intersection_cache" in result
        or "secondary_intersection_cache" in result
    ):
        simulation_runtime_state.stored_primary_max_positions = list(resolved_primary_peak_tables)
        simulation_runtime_state.stored_secondary_max_positions = list(
            resolved_secondary_peak_tables
        )
        simulation_runtime_state.stored_hit_table_signature = result.get("hit_table_signature")
    if not bool(result.get("is_preview", False)):
        _store_primary_cache_payload(
            cache_signature=result.get("primary_contribution_cache_signature"),
            source_mode=str(
                result.get(
                    "primary_source_mode",
                    simulation_runtime_state.primary_source_mode,
                )
            ),
            active_keys=result.get(
                "active_primary_contribution_keys",
                simulation_runtime_state.primary_active_contribution_keys,
            ),
            contribution_keys=result.get("primary_contribution_keys", []),
            raw_hit_tables=result.get("primary_hit_tables_raw", []),
        )
    _reset_combined_simulation_artifacts()
    simulation_runtime_state.last_image_generation_ms = float(
        result.get("image_generation_elapsed_ms", float("nan"))
    )
    simulation_runtime_state.preview_active = bool(result.get("is_preview", False))
    preview_sample_count = result.get("preview_sample_count")
    simulation_runtime_state.preview_sample_count = (
        int(preview_sample_count) if preview_sample_count is not None else None
    )
    _trace_live_cache_event(
        "simulation_result",
        "apply",
        outcome="stored",
        is_preview=bool(result.get("is_preview", False)),
        job_kind=str(result.get("job_kind", "full")),
        image_generation_elapsed_ms=float(result.get("image_generation_elapsed_ms", float("nan"))),
        primary_image_shape=_live_cache_shape(result.get("primary_image")),
        secondary_image_shape=_live_cache_shape(result.get("secondary_image")),
        primary_intersection_cache_count=_live_cache_count(
            simulation_runtime_state.stored_primary_intersection_cache
        ),
        secondary_intersection_cache_count=_live_cache_count(
            simulation_runtime_state.stored_secondary_intersection_cache
        ),
        primary_peak_table_count=_live_cache_count(
            simulation_runtime_state.stored_primary_max_positions
        ),
        secondary_peak_table_count=_live_cache_count(
            simulation_runtime_state.stored_secondary_max_positions
        ),
        stored_hit_table_signature_summary=_live_cache_signature_summary(
            simulation_runtime_state.stored_hit_table_signature
        ),
    )


def _build_analysis_integrator(job: dict[str, object]):
    center = np.asarray(job["center"], dtype=np.float64)
    return _get_azimuthal_integrator_cls()(
        dist=float(job["distance_m"]),
        poni1=float(center[0]) * float(job["pixel_size_m"]),
        poni2=float(center[1]) * float(job["pixel_size_m"]),
        rot1=0.0,
        rot2=0.0,
        rot3=0.0,
        wavelength=float(job["wavelength_m"]),
        pixel1=float(job["pixel_size_m"]),
        pixel2=float(job["pixel_size_m"]),
    )


def _caked_geometry_cache_signature(
    distance_value: object,
    center_x_value: object,
    center_y_value: object,
    wavelength_value: object,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return the cache key for detector<->caked LUT state."""

    try:
        distance_float = float(distance_value)
    except Exception:
        distance_float = float("nan")
    try:
        center_x_float = float(center_x_value)
    except Exception:
        center_x_float = float("nan")
    try:
        center_y_float = float(center_y_value)
    except Exception:
        center_y_float = float("nan")
    try:
        wavelength_float = float(wavelength_value)
    except Exception:
        wavelength_float = float("nan")
    return (
        round(distance_float, 9) if np.isfinite(distance_float) else None,
        round(center_x_float, 6) if np.isfinite(center_x_float) else None,
        round(center_y_float, 6) if np.isfinite(center_y_float) else None,
        round(wavelength_float, 12) if np.isfinite(wavelength_float) else None,
    )


def _prepare_caked_display_payload(res2) -> dict[str, object] | None:
    if res2 is None:
        return None

    caked_img = np.asarray(res2.intensity, dtype=float)
    radial_vals = np.asarray(res2.radial, dtype=float)
    azimuth_vals = _wrap_phi_range(_adjust_phi_zero(res2.azimuthal))

    if azimuth_vals.size:
        azimuth_order = np.argsort(azimuth_vals)
        azimuth_vals = azimuth_vals[azimuth_order]
        caked_img = caked_img[azimuth_order, :]

    radial_mask = (radial_vals >= 0.0) & (radial_vals <= 90.0)
    if np.any(radial_mask):
        radial_vals = radial_vals[radial_mask]
        caked_img = caked_img[:, radial_mask]

    if radial_vals.size:
        radial_min = float(np.min(radial_vals))
        radial_max = float(np.max(radial_vals))
    else:
        radial_min, radial_max = 0.0, 90.0

    if azimuth_vals.size:
        azimuth_min = float(np.min(azimuth_vals))
        azimuth_max = float(np.max(azimuth_vals))
    else:
        azimuth_min, azimuth_max = -180.0, 180.0

    return {
        "image": np.asarray(caked_img, dtype=float),
        "radial": np.asarray(radial_vals, dtype=float),
        "azimuth": np.asarray(azimuth_vals, dtype=float),
        "extent": [
            radial_min,
            radial_max,
            azimuth_min,
            azimuth_max,
        ],
    }


def _prepare_q_space_display_payload(
    caked_payload: Mapping[str, object] | None,
    *,
    wavelength_m: object,
) -> dict[str, object] | None:
    if not isinstance(caked_payload, Mapping):
        return None
    return gui_controllers.caked_image_to_q_space_payload(
        caked_payload.get("image"),
        caked_payload.get("radial"),
        caked_payload.get("azimuth"),
        wavelength_m=wavelength_m,
    )


def _store_q_space_display_payload(
    *,
    sim_payload: Mapping[str, object] | None,
    bg_payload: Mapping[str, object] | None,
) -> None:
    if isinstance(sim_payload, Mapping):
        simulation_runtime_state.last_q_space_image_unscaled = np.asarray(
            sim_payload.get("image"),
            dtype=float,
        )
        simulation_runtime_state.last_q_space_qr_values = np.asarray(
            sim_payload.get("qr"),
            dtype=float,
        )
        simulation_runtime_state.last_q_space_qz_values = np.asarray(
            sim_payload.get("qz"),
            dtype=float,
        )
        simulation_runtime_state.last_q_space_extent = list(sim_payload.get("extent", []))
    else:
        simulation_runtime_state.last_q_space_image_unscaled = None
        simulation_runtime_state.last_q_space_qr_values = None
        simulation_runtime_state.last_q_space_qz_values = None
        simulation_runtime_state.last_q_space_extent = None

    if isinstance(bg_payload, Mapping):
        simulation_runtime_state.last_q_space_background_image_unscaled = np.asarray(
            bg_payload.get("image"),
            dtype=float,
        )
    else:
        simulation_runtime_state.last_q_space_background_image_unscaled = None


def _restore_caked_display_payload_from_cached_results(
    *,
    background_visible: bool,
) -> bool:
    """Rebuild caked display arrays from the current cached analysis results."""

    sim_caked = _prepare_caked_display_payload(simulation_runtime_state.last_res2_sim)
    if not isinstance(sim_caked, dict):
        return False

    simulation_runtime_state.last_caked_image_unscaled = np.asarray(
        sim_caked.get("image"),
        dtype=float,
    )
    simulation_runtime_state.last_caked_radial_values = np.asarray(
        sim_caked.get("radial"),
        dtype=float,
    )
    simulation_runtime_state.last_caked_azimuth_values = np.asarray(
        sim_caked.get("azimuth"),
        dtype=float,
    )
    simulation_runtime_state.last_caked_extent = list(sim_caked.get("extent", []))
    caked_intersection_cache = _prepare_caked_intersection_cache(
        getattr(simulation_runtime_state, "stored_intersection_cache", ()),
        center=(float(center_x_var.get()), float(center_y_var.get())),
        detector_distance=float(corto_detector_var.get()),
        pixel_size=float(pixel_size_m),
    )
    simulation_runtime_state.last_caked_intersection_cache = caked_intersection_cache

    bg_caked = None
    if background_visible and simulation_runtime_state.last_res2_background is not None:
        bg_caked = _prepare_caked_display_payload(simulation_runtime_state.last_res2_background)
    if isinstance(bg_caked, dict):
        simulation_runtime_state.last_caked_background_image_unscaled = np.asarray(
            bg_caked.get("image"),
            dtype=float,
        )
    else:
        simulation_runtime_state.last_caked_background_image_unscaled = None
    _store_q_space_display_payload(
        sim_payload=_prepare_q_space_display_payload(
            sim_caked,
            wavelength_m=lambda_ * 1.0e-10,
        ),
        bg_payload=_prepare_q_space_display_payload(
            bg_caked,
            wavelength_m=lambda_ * 1.0e-10,
        ),
    )
    return True


def _run_analysis_job(job: dict[str, object]) -> dict[str, object]:
    ai = _build_analysis_integrator(job)
    sim_image = np.asarray(job["image"], dtype=np.float64)
    bg_image = job.get("background_image")
    bg_array = None if bg_image is None else np.asarray(bg_image, dtype=np.float64)
    npt_rad = int(job.get("npt_rad", 1000))
    npt_azim = int(job.get("npt_azim", 720))
    is_preview = bool(job.get("is_preview", False))
    cached_bg_res2 = job.get("cached_bg_res2")
    cached_bg_caked = job.get("cached_bg_caked")
    intersection_cache = _copy_intersection_cache_tables(job.get("intersection_cache"))

    analysis_start_time = perf_counter()
    with temporary_numba_thread_limit(default_reserved_cpu_worker_count()):
        sim_res2 = caking(sim_image, ai, npt_rad=npt_rad, npt_azim=npt_azim)
        if cached_bg_res2 is not None:
            bg_res2 = cached_bg_res2
        elif bg_array is not None:
            bg_res2 = caking(bg_array, ai, npt_rad=npt_rad, npt_azim=npt_azim)
        else:
            bg_res2 = None
    sim_caked = _prepare_caked_display_payload(sim_res2)
    if isinstance(cached_bg_caked, dict):
        bg_caked = dict(cached_bg_caked)
    else:
        bg_caked = _prepare_caked_display_payload(bg_res2)
    sim_q_space = _prepare_q_space_display_payload(
        sim_caked,
        wavelength_m=job.get("wavelength_m"),
    )
    bg_q_space = _prepare_q_space_display_payload(
        bg_caked,
        wavelength_m=job.get("wavelength_m"),
    )
    sim_caked_intersection_cache = _prepare_caked_intersection_cache(
        intersection_cache,
        center=job.get("center"),
        detector_distance=float(job.get("distance_m", float("nan"))),
        pixel_size=float(job.get("pixel_size_m", float("nan"))),
    )

    return {
        "job_id": int(job["job_id"]),
        "signature": job["signature"],
        "epoch": int(job["epoch"]),
        "is_preview": is_preview,
        "analysis_bins": (npt_rad, npt_azim),
        "sim_cache_sig": job.get("sim_cache_sig"),
        "bg_cache_sig": job.get("bg_cache_sig"),
        "sim_caking_sig": job.get("sim_caking_sig"),
        "bg_caking_sig": job.get("bg_caking_sig"),
        "sim_res2": sim_res2,
        "bg_res2": bg_res2,
        "sim_caked": sim_caked,
        "sim_caked_intersection_cache": sim_caked_intersection_cache,
        "bg_caked": bg_caked,
        "sim_q_space": sim_q_space,
        "bg_q_space": bg_q_space,
        "analysis_elapsed_ms": (perf_counter() - analysis_start_time) * 1e3,
    }


def _submit_async_analysis_job(job: dict[str, object]) -> None:
    simulation_runtime_state.analysis_active_job = job
    simulation_runtime_state.analysis_future = _ensure_analysis_worker_executor().submit(
        _run_analysis_job,
        dict(job),
    )
    simulation_runtime_state.analysis_error_text = None
    if simulation_runtime_state.worker_active_job is None:
        simulation_runtime_state.update_phase = "analyzing"
    _append_job_queue_trace(
        "job_started",
        lane="analysis",
        job=job,
        active_job=job,
        queued_job=simulation_runtime_state.analysis_queued_job,
    )
    _refresh_run_status_bar()
    if simulation_runtime_state.analysis_poll_token is None:
        simulation_runtime_state.analysis_poll_token = root.after(
            SIMULATION_WORKER_POLL_MS,
            _poll_async_analysis_job,
        )


def _poll_async_analysis_job() -> None:
    simulation_runtime_state.analysis_poll_token = None
    future = simulation_runtime_state.analysis_future
    active_job = simulation_runtime_state.analysis_active_job
    if future is None or active_job is None:
        if _promote_queued_analysis_job(reason="active_slot_available"):
            return
        _refresh_run_status_bar()
        return

    if not future.done():
        simulation_runtime_state.analysis_poll_token = root.after(
            SIMULATION_WORKER_POLL_MS,
            _poll_async_analysis_job,
        )
        return

    simulation_runtime_state.analysis_future = None
    simulation_runtime_state.analysis_active_job = None
    queued_job = simulation_runtime_state.analysis_queued_job
    try:
        result = future.result()
    except Exception as exc:
        _append_job_queue_trace(
            "job_finished",
            lane="analysis",
            job=active_job,
            active_job=None,
            queued_job=queued_job,
            outcome="error",
            error=exc,
        )
        simulation_runtime_state.analysis_error_text = str(exc)
        if simulation_runtime_state.worker_active_job is None:
            simulation_runtime_state.update_phase = "error"
        if "progress_label" in globals() and progress_label is not None:
            progress_label.config(text=f"Analysis update failed: {exc}")
        _refresh_run_status_bar()
        if _promote_queued_analysis_job(reason="previous_job_failed"):
            return
        return

    _append_job_queue_trace(
        "job_finished",
        lane="analysis",
        job=result,
        active_job=None,
        queued_job=queued_job,
        outcome="success",
    )
    latest_epoch = int(simulation_runtime_state.analysis_epoch)
    result_epoch = int(result.get("epoch", -1))
    superseded = isinstance(queued_job, dict) and int(queued_job.get("job_id", -1)) > int(
        result.get("job_id", -1)
    )
    if result_epoch != latest_epoch or superseded:
        _append_job_queue_trace(
            "job_discarded_as_stale",
            lane="analysis",
            job=result,
            active_job=None,
            queued_job=queued_job,
            reason=("superseded_by_newer_queued_job" if superseded else "result_epoch_mismatch"),
        )
        if _promote_queued_analysis_job(reason="previous_job_stale"):
            return
        else:
            if simulation_runtime_state.worker_active_job is None:
                simulation_runtime_state.update_phase = "ready"
            _refresh_run_status_bar()
        return

    simulation_runtime_state.analysis_ready_result = result
    if simulation_runtime_state.worker_active_job is None:
        simulation_runtime_state.update_phase = "applying"
    _refresh_run_status_bar()
    root.after_idle(do_update)


def _request_async_analysis_job(job: dict[str, object]) -> str:
    requested_key = _analysis_job_key(job)
    _append_job_queue_trace(
        "job_requested",
        lane="analysis",
        job=job,
        active_job=simulation_runtime_state.analysis_active_job,
        queued_job=simulation_runtime_state.analysis_queued_job,
    )
    ready_result = simulation_runtime_state.analysis_ready_result
    if _analysis_job_key(ready_result) == requested_key:
        if simulation_runtime_state.worker_active_job is None:
            simulation_runtime_state.update_phase = "applying"
        _refresh_run_status_bar()
        return "ready"
    if isinstance(ready_result, dict):
        _append_job_queue_trace(
            "job_discarded_as_stale",
            lane="analysis",
            job=ready_result,
            active_job=simulation_runtime_state.analysis_active_job,
            queued_job=simulation_runtime_state.analysis_queued_job,
            reason="superseded_by_new_request",
        )
        simulation_runtime_state.analysis_ready_result = None

    if _analysis_job_key(simulation_runtime_state.analysis_active_job) == requested_key:
        if simulation_runtime_state.worker_active_job is None:
            simulation_runtime_state.update_phase = "analyzing"
        _refresh_run_status_bar()
        return "running"

    if _analysis_job_key(simulation_runtime_state.analysis_queued_job) == requested_key:
        _append_job_queue_trace(
            "job_queued",
            lane="analysis",
            job=simulation_runtime_state.analysis_queued_job,
            active_job=simulation_runtime_state.analysis_active_job,
            queued_job=simulation_runtime_state.analysis_queued_job,
            reason="already_queued",
        )
        if simulation_runtime_state.worker_active_job is None:
            simulation_runtime_state.update_phase = "queued"
        _refresh_run_status_bar()
        return "queued"

    simulation_runtime_state.analysis_job_counter = (
        int(simulation_runtime_state.analysis_job_counter) + 1
    )
    queued_job = dict(job)
    queued_job["job_id"] = int(simulation_runtime_state.analysis_job_counter)
    simulation_runtime_state.analysis_error_text = None

    if (
        simulation_runtime_state.analysis_active_job is None
        and simulation_runtime_state.analysis_future is None
    ):
        _append_job_queue_trace(
            "job_submitted",
            lane="analysis",
            job=queued_job,
            active_job=None,
            queued_job=simulation_runtime_state.analysis_queued_job,
            reason="requested",
        )
        _submit_async_analysis_job(queued_job)
        return "submitted"

    # Latest-request-wins: keep one active job and one replacement queued job.
    _replace_queued_analysis_job(queued_job)
    _append_job_queue_trace(
        "job_queued",
        lane="analysis",
        job=queued_job,
        active_job=simulation_runtime_state.analysis_active_job,
        queued_job=simulation_runtime_state.analysis_queued_job,
        reason="awaiting_active_job",
    )
    if simulation_runtime_state.worker_active_job is None:
        simulation_runtime_state.update_phase = "queued"
    _refresh_run_status_bar()
    return "queued"


def _consume_ready_analysis_result(
    signature: object,
    *,
    is_preview: bool,
) -> dict[str, object] | None:
    ready_result = simulation_runtime_state.analysis_ready_result
    if _analysis_job_key(ready_result) != (
        signature,
        int(simulation_runtime_state.analysis_epoch),
        bool(is_preview),
    ):
        if isinstance(ready_result, dict) and int(ready_result.get("epoch", -1)) < int(
            simulation_runtime_state.analysis_epoch
        ):
            simulation_runtime_state.analysis_ready_result = None
        return None

    simulation_runtime_state.analysis_ready_result = None
    return dict(ready_result)


def _apply_ready_analysis_result(result: dict[str, object]) -> None:
    simulation_runtime_state.last_analysis_signature = result.get("signature")
    simulation_runtime_state.analysis_preview_active = bool(result.get("is_preview", False))
    analysis_bins = result.get("analysis_bins")
    if (
        isinstance(analysis_bins, (tuple, list))
        and len(analysis_bins) == 2
        and all(isinstance(value, (int, np.integer)) for value in analysis_bins)
    ):
        simulation_runtime_state.analysis_preview_bins = (
            int(analysis_bins[0]),
            int(analysis_bins[1]),
        )
    else:
        simulation_runtime_state.analysis_preview_bins = None
    simulation_runtime_state.last_res2_sim = result.get("sim_res2")
    simulation_runtime_state.last_res2_background = result.get("bg_res2")

    sim_caked = result.get("sim_caked")
    if isinstance(sim_caked, dict):
        simulation_runtime_state.last_caked_image_unscaled = np.asarray(
            sim_caked.get("image"),
            dtype=float,
        )
        simulation_runtime_state.last_caked_radial_values = np.asarray(
            sim_caked.get("radial"),
            dtype=float,
        )
        simulation_runtime_state.last_caked_azimuth_values = np.asarray(
            sim_caked.get("azimuth"),
            dtype=float,
        )
        simulation_runtime_state.last_caked_extent = list(sim_caked.get("extent", []))
        caked_intersection_cache = _copy_intersection_cache_tables(
            result.get("sim_caked_intersection_cache", [])
        )
        simulation_runtime_state.last_caked_intersection_cache = caked_intersection_cache
    else:
        simulation_runtime_state.last_caked_image_unscaled = None
        simulation_runtime_state.last_caked_radial_values = None
        simulation_runtime_state.last_caked_azimuth_values = None
        simulation_runtime_state.last_caked_extent = None
        simulation_runtime_state.last_caked_intersection_cache = None

    bg_caked = result.get("bg_caked")
    if isinstance(bg_caked, dict):
        simulation_runtime_state.last_caked_background_image_unscaled = np.asarray(
            bg_caked.get("image"),
            dtype=float,
        )
    else:
        simulation_runtime_state.last_caked_background_image_unscaled = None
    _store_q_space_display_payload(
        sim_payload=result.get("sim_q_space"),
        bg_payload=result.get("bg_q_space"),
    )

    _store_cached_caking_entry(
        "sim",
        result.get("sim_cache_sig"),
        res2=simulation_runtime_state.last_res2_sim,
        payload=sim_caked if isinstance(sim_caked, dict) else None,
    )
    _store_cached_caking_entry(
        "bg",
        result.get("bg_cache_sig"),
        res2=simulation_runtime_state.last_res2_background,
        payload=bg_caked if isinstance(bg_caked, dict) else None,
    )


def schedule_update():
    """Queue a throttled simulation/redraw update."""

    if bool(getattr(simulation_runtime_state, "startup_updates_suspended", False)):
        simulation_runtime_state.pending_startup_refresh = True
        return

    _ensure_runtime_update_trace_hooks()
    previous_pending = simulation_runtime_state.update_pending
    gui_controllers.clear_tk_after_token(
        root,
        simulation_runtime_state.integration_update_pending,
    )
    simulation_runtime_state.integration_update_pending = None
    if simulation_runtime_state.interaction_drag_active:
        simulation_runtime_state.interaction_drag_requires_settled_update = True

    gui_controllers.clear_tk_after_token(
        root,
        simulation_runtime_state.update_pending,
    )
    simulation_runtime_state.update_pending = root.after(
        _current_update_debounce_ms(),
        do_update,
    )
    if simulation_runtime_state.worker_active_job is None:
        simulation_runtime_state.update_phase = "queued"
    _append_runtime_update_trace(
        "schedule_update",
        previous_pending=previous_pending,
        queued_token=simulation_runtime_state.update_pending,
        update_running=simulation_runtime_state.update_running,
        update_phase=getattr(simulation_runtime_state, "update_phase", None),
        worker_active=bool(simulation_runtime_state.worker_active_job is not None),
        worker_queued=bool(simulation_runtime_state.worker_queued_job is not None),
        analysis_active=bool(simulation_runtime_state.analysis_active_job is not None),
        analysis_queued=bool(simulation_runtime_state.analysis_queued_job is not None),
        background_index=int(background_runtime_state.current_background_index),
        manual_pick_armed=bool(geometry_runtime_state.manual_pick_armed),
    )
    _refresh_run_status_bar()


def _refresh_geometry_fit_caked_roi_preview(*_args) -> None:
    """Refresh the main raster display after ROI preview toggles change."""

    try:
        apply_scale_factor_to_existing_results(
            update_limits=False,
            update_1d=False,
            update_canvas=True,
            update_chi_square=True,
        )
        _refresh_settled_overlays()
    except Exception:
        schedule_update()


try:
    geometry_fit_caked_roi_preview_var.trace_add(
        "write",
        _refresh_geometry_fit_caked_roi_preview,
    )
except Exception:
    pass


def _should_collect_hit_tables_for_update() -> bool:
    """Return whether the next redraw needs per-hit detector tables."""
    if _live_interaction_active():
        return False
    return gui_manual_geometry.should_collect_hit_tables_for_update(
        background_visible=bool(background_runtime_state.visible),
        current_background_index=background_runtime_state.current_background_index,
        skip_preview_once=bool(getattr(geometry_preview_state, "skip_once", False)),
        manual_pick_armed=bool(geometry_runtime_state.manual_pick_armed),
        hkl_pick_armed=bool(peak_selection_state.hkl_pick_armed),
        selected_hkl_target=peak_selection_state.selected_hkl_target,
        selected_peak_record=simulation_runtime_state.selected_peak_record,
        geometry_q_group_refresh_requested=bool(geometry_q_group_state.refresh_requested),
        live_geometry_preview_enabled=_live_geometry_preview_enabled,
        current_manual_pick_background_image=_current_geometry_manual_pick_background_image,
        geometry_manual_pairs_for_index=_geometry_manual_pairs_for_index,
        geometry_manual_pick_session_active=_geometry_manual_pick_session_active,
    )


simulation_runtime_state.peak_positions = []
simulation_runtime_state.peak_millers = []
simulation_runtime_state.peak_intensities = []
simulation_runtime_state.peak_records = []
simulation_runtime_state.selected_peak_record = None
# 0 disables the cap and keeps all distinct per-reflection hit candidates.
HKL_PICK_MAX_HITS_PER_REFLECTION = 0

simulation_runtime_state.prev_background_visible = True
simulation_runtime_state.last_bg_signature = None
simulation_runtime_state.last_sim_signature = None
simulation_runtime_state.last_simulation_signature = None
simulation_runtime_state.source_row_snapshots = {}
simulation_runtime_state.stored_hit_table_signature = None
simulation_runtime_state.analysis_epoch = 0
simulation_runtime_state.last_analysis_signature = None
simulation_runtime_state.analysis_preview_active = False
simulation_runtime_state.analysis_preview_bins = None
simulation_runtime_state.stored_max_positions_local = None
simulation_runtime_state.stored_source_reflection_indices_local = None
simulation_runtime_state.stored_sim_image = None
simulation_runtime_state.stored_peak_table_lattice = None
simulation_runtime_state.stored_primary_sim_image = None
simulation_runtime_state.stored_secondary_sim_image = None
simulation_runtime_state.stored_primary_max_positions = None
simulation_runtime_state.stored_secondary_max_positions = None
simulation_runtime_state.stored_primary_source_reflection_indices = None
simulation_runtime_state.stored_secondary_source_reflection_indices = None
simulation_runtime_state.stored_primary_peak_table_lattice = None
simulation_runtime_state.stored_secondary_peak_table_lattice = None
simulation_runtime_state.stored_primary_intersection_cache = None
simulation_runtime_state.stored_secondary_intersection_cache = None
simulation_runtime_state.stored_intersection_cache = None
simulation_runtime_state.last_unscaled_image_signature = None
simulation_runtime_state.normalization_scale_cache = {"sig": None, "value": 1.0}
simulation_runtime_state.peak_overlay_cache = _empty_peak_overlay_cache()
simulation_runtime_state.caking_cache = _empty_caking_cache()
simulation_runtime_state.analysis_executor = None
simulation_runtime_state.analysis_future = None
simulation_runtime_state.analysis_poll_token = None
simulation_runtime_state.analysis_job_counter = 0
simulation_runtime_state.analysis_active_job = None
simulation_runtime_state.analysis_queued_job = None
simulation_runtime_state.analysis_ready_result = None
simulation_runtime_state.analysis_error_text = None
simulation_runtime_state.geometry_fit_executor = None
simulation_runtime_state.geometry_fit_future = None
simulation_runtime_state.geometry_fit_poll_token = None
simulation_runtime_state.geometry_fit_job_counter = 0
simulation_runtime_state.geometry_fit_active_job = None
simulation_runtime_state.geometry_fit_ready_result = None
simulation_runtime_state.geometry_fit_error_text = None
simulation_runtime_state.geometry_fit_event_queue = queue.SimpleQueue()
simulation_runtime_state.interaction_drag_active = False
simulation_runtime_state.interaction_settle_token = None
simulation_runtime_state.interaction_drag_requires_settled_update = False
simulation_runtime_state.chi_square_update_token = 0
simulation_runtime_state.chi_square_state = {
    "last_ts": 0.0,
    "last_token": -1,
    "last_text": "Chi-Squared: N/A",
}


###############################################################################
#                              MAIN UPDATE
###############################################################################
def do_update():
    global av2, cv2

    _ensure_runtime_update_trace_hooks()
    simulation_runtime_state.update_trace_counter = (
        int(getattr(simulation_runtime_state, "update_trace_counter", 0)) + 1
    )
    update_trace_id = int(simulation_runtime_state.update_trace_counter)
    update_trace_stage = "enter"
    simulation_runtime_state.current_update_trace_id = int(update_trace_id)
    simulation_runtime_state.current_update_trace_stage = str(update_trace_stage)

    def _set_update_trace_stage(stage: str) -> None:
        nonlocal update_trace_stage
        update_trace_stage = str(stage)
        simulation_runtime_state.current_update_trace_stage = str(update_trace_stage)

    def _trace_update(event: str, **fields: object) -> None:
        simulation_runtime_state.current_update_trace_id = int(update_trace_id)
        simulation_runtime_state.current_update_trace_stage = str(update_trace_stage)
        _append_runtime_update_trace(
            event,
            update_id=update_trace_id,
            stage=update_trace_stage,
            update_phase=getattr(simulation_runtime_state, "update_phase", None),
            update_running=getattr(simulation_runtime_state, "update_running", None),
            **fields,
        )

    _trace_update(
        "do_update_enter",
        pending_token=simulation_runtime_state.update_pending,
        worker_active=bool(simulation_runtime_state.worker_active_job is not None),
        worker_queued=bool(simulation_runtime_state.worker_queued_job is not None),
        analysis_active=bool(simulation_runtime_state.analysis_active_job is not None),
        analysis_queued=bool(simulation_runtime_state.analysis_queued_job is not None),
    )

    if simulation_runtime_state.update_running:
        # another update is in progress; try again shortly
        simulation_runtime_state.update_pending = root.after(
            _current_update_debounce_ms(),
            do_update,
        )
        _trace_update(
            "do_update_busy_rescheduled",
            queued_token=simulation_runtime_state.update_pending,
        )
        return

    simulation_runtime_state.update_pending = None
    simulation_runtime_state.update_running = True
    if simulation_runtime_state.worker_active_job is None:
        simulation_runtime_state.update_phase = "applying"
    _refresh_run_status_bar()
    _clear_initial_simulation_loading_overlay(redraw=False)
    update_start_time = perf_counter()
    image_generation_elapsed_ms = 0.0
    image_generation_cached = True
    _set_update_trace_stage("params")

    gamma_updated = float(gamma_var.get())
    Gamma_updated = float(Gamma_var.get())
    chi_updated = float(chi_var.get())
    psi_z_updated = float(psi_z_var.get())
    zs_updated = float(zs_var.get())
    zb_updated = float(zb_var.get())
    sample_width_updated = float(sample_width_var.get())
    sample_length_updated = float(sample_length_var.get())
    sample_depth_updated = float(sample_depth_var.get())
    cor_angle_updated = float(cor_angle_var.get())
    a_updated = float(a_var.get())
    c_updated = float(c_var.get())
    theta_init_up = float(_current_effective_theta_initial(strict_count=False))
    debye_x_updated = float(debye_x_var.get())
    debye_y_updated = float(debye_y_var.get())
    ordered_structure_scale = float(_current_ordered_structure_scale())
    corto_det_up = float(corto_detector_var.get())
    center_x_up = float(center_x_var.get())
    center_y_up = float(center_y_var.get())
    _trace_update(
        "do_update_params",
        background_index=int(background_runtime_state.current_background_index),
        theta_initial_deg=float(theta_init_up),
        center_x=float(center_x_up),
        center_y=float(center_y_up),
        show_caked_2d=bool(analysis_view_controls_view_state.show_caked_2d_var.get()),
        preview_active=bool(simulation_runtime_state.preview_active),
    )

    new_two_theta_max = detector_two_theta_max(
        image_size,
        [center_x_up, center_y_up],
        corto_det_up,
        pixel_size=pixel_size_m,
    )

    global two_theta_range, _last_a_for_ht, _last_c_for_ht, _last_iodine_z_for_ht
    global _last_phi_l_divisor, _last_phase_delta_expression
    global _last_atom_site_fractional_signature
    phase_delta_expression_current = _current_phase_delta_expression()
    phi_l_divisor_current = _current_phi_l_divisor()
    iodine_z_current = _current_iodine_z()
    atom_site_signature = _atom_site_fractional_signature(_current_atom_site_fractional_values())
    need_rebuild = False
    if not math.isclose(new_two_theta_max, two_theta_range[1], rel_tol=1e-6, abs_tol=1e-6):
        two_theta_range = (0.0, new_two_theta_max)
        need_rebuild = True
    if not math.isclose(c_updated, _last_c_for_ht, rel_tol=1e-9, abs_tol=1e-9):
        need_rebuild = True
    if not math.isclose(a_updated, _last_a_for_ht, rel_tol=1e-9, abs_tol=1e-9):
        need_rebuild = True
    if not math.isclose(iodine_z_current, _last_iodine_z_for_ht, rel_tol=1e-9, abs_tol=1e-9):
        need_rebuild = True
    if not math.isclose(
        float(phi_l_divisor_current),
        float(_last_phi_l_divisor),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        need_rebuild = True
    if phase_delta_expression_current != _last_phase_delta_expression:
        need_rebuild = True
    if atom_site_signature != _last_atom_site_fractional_signature:
        need_rebuild = True

    if need_rebuild:
        current_occ = [occ_var.get() for occ_var in _occupancy_control_vars()]
        current_p = [p0_var.get(), p1_var.get(), p2_var.get()]
        weight_values = [w0_var.get(), w1_var.get(), w2_var.get()]
        normalized_weights = gui_controllers.normalize_stacking_weight_values(weight_values)
        _rebuild_diffraction_inputs(
            current_occ,
            current_p,
            normalized_weights,
            a_updated,
            c_updated,
            force=True,
            trigger_update=False,
        )

    _sync_center_marker(redraw=False)

    mosaic_params = build_mosaic_params()
    optics_mode_flag = _current_optics_mode_flag()
    accumulate_image_requested = True
    qr_cylinder_replace_requested = _qr_cylinder_replace_simulation_enabled()
    collect_hit_tables_requested = bool(_should_collect_hit_tables_for_update())
    primary_source_signature = (
        (
            "qr",
            id(simulation_runtime_state.sim_primary_qr),
            len(simulation_runtime_state.sim_primary_qr),
        )
        if isinstance(simulation_runtime_state.sim_primary_qr, dict)
        and len(simulation_runtime_state.sim_primary_qr) > 0
        else (
            "miller",
            id(simulation_runtime_state.sim_miller1),
            tuple(np.asarray(simulation_runtime_state.sim_miller1).shape),
        )
    )
    secondary_source_signature = (
        "miller",
        id(simulation_runtime_state.sim_miller2),
        tuple(np.asarray(simulation_runtime_state.sim_miller2).shape),
    )
    secondary_a = float(av2) if av2 is not None else float(a_updated)
    secondary_c = float(cv2) if cv2 is not None else float(c_updated)

    primary_data = (
        gui_controllers.copy_bragg_qr_dict(simulation_runtime_state.sim_primary_qr)
        if isinstance(simulation_runtime_state.sim_primary_qr, dict)
        and len(simulation_runtime_state.sim_primary_qr) > 0
        else np.asarray(simulation_runtime_state.sim_miller1, dtype=np.float64).copy()
    )
    primary_intensities = np.asarray(
        simulation_runtime_state.sim_intens1,
        dtype=np.float64,
    ).copy()
    if isinstance(primary_data, dict):
        primary_data = _scaled_bragg_qr_dict(primary_data, ordered_structure_scale)
    else:
        primary_intensities *= float(ordered_structure_scale)
    secondary_data = np.asarray(
        simulation_runtime_state.sim_miller2,
        dtype=np.float64,
    ).copy()
    secondary_intensities = np.asarray(
        simulation_runtime_state.sim_intens2,
        dtype=np.float64,
    ).copy()
    primary_available = (
        len(primary_data) > 0
        if isinstance(primary_data, dict)
        else (
            np.asarray(primary_data).ndim == 2
            and np.asarray(primary_data).shape[0] > 0
            and primary_intensities.size > 0
        )
    )
    secondary_available = (
        secondary_data.ndim == 2 and secondary_data.shape[0] > 0 and secondary_intensities.size > 0
    )
    primary_requested_source_mode = str(
        getattr(
            simulation_runtime_state,
            "primary_requested_source_mode",
            "",
        )
        or ("qr" if isinstance(primary_data, dict) and len(primary_data) > 0 else "miller")
    )
    primary_requested_contribution_keys = tuple(
        getattr(
            simulation_runtime_state,
            "primary_requested_contribution_keys",
            (),
        )
        or ()
    )
    primary_requested_filter_signature = getattr(
        simulation_runtime_state,
        "primary_requested_filter_signature",
        None,
    )
    if primary_requested_filter_signature is None:
        primary_requested_filter_signature = (
            primary_requested_source_mode,
            primary_source_signature,
        )

    def _simulation_signature_base(
        *,
        optics_mode_component: int,
        include_mosaic_shape: bool = True,
    ):
        return _geometry_source_snapshot_signature_from_params(
            {
                "gamma": gamma_updated,
                "Gamma": Gamma_updated,
                "chi": chi_updated,
                "psi_z": psi_z_updated,
                "zs": zs_updated,
                "zb": zb_updated,
                "sample_width": sample_width_updated,
                "sample_length": sample_length_updated,
                "sample_depth": sample_depth_updated,
                "debye_x": debye_x_updated,
                "debye_y": debye_y_updated,
                "a": a_updated,
                "c": c_updated,
                "theta_initial": theta_init_up,
                "cor": cor_angle_updated,
                "center_x": center_x_up,
                "center_y": center_y_up,
                "sigma_mosaic_deg": mosaic_params.get("sigma_mosaic_deg", 0.0),
                "gamma_mosaic_deg": mosaic_params.get("gamma_mosaic_deg", 0.0),
                "eta": mosaic_params.get("eta", 0.0),
            },
            mosaic_params=mosaic_params,
            optics_mode_component=int(optics_mode_component),
            include_mosaic_shape=bool(include_mosaic_shape),
            sf_prune_bias=current_sf_prune_bias(),
            sf_prune_stats=simulation_runtime_state.sf_prune_stats,
            ordered_structure_scale=ordered_structure_scale,
            qr_cylinder_replace_requested=qr_cylinder_replace_requested,
            primary_source_signature=primary_source_signature,
            secondary_source_signature=secondary_source_signature,
            secondary_a=secondary_a,
            secondary_c=secondary_c,
        )

    def _primary_contribution_cache_signature_base():
        return _geometry_source_snapshot_signature_from_params(
            {
                "gamma": gamma_updated,
                "Gamma": Gamma_updated,
                "chi": chi_updated,
                "psi_z": psi_z_updated,
                "zs": zs_updated,
                "zb": zb_updated,
                "sample_width": sample_width_updated,
                "sample_length": sample_length_updated,
                "sample_depth": sample_depth_updated,
                "debye_x": debye_x_updated,
                "debye_y": debye_y_updated,
                "a": a_updated,
                "c": c_updated,
                "theta_initial": theta_init_up,
                "cor": cor_angle_updated,
                "center_x": center_x_up,
                "center_y": center_y_up,
                "sigma_mosaic_deg": mosaic_params.get("sigma_mosaic_deg", 0.0),
                "gamma_mosaic_deg": mosaic_params.get("gamma_mosaic_deg", 0.0),
                "eta": mosaic_params.get("eta", 0.0),
                "distance_m": corto_det_up,
            },
            mosaic_params=mosaic_params,
            optics_mode_component=int(optics_mode_flag),
            include_mosaic_shape=True,
            sf_prune_bias=0.0,
            sf_prune_stats={},
            ordered_structure_scale=ordered_structure_scale,
            qr_cylinder_replace_requested=qr_cylinder_replace_requested,
            primary_source_signature=(
                primary_requested_source_mode,
                primary_requested_filter_signature,
            ),
            secondary_source_signature=secondary_source_signature,
            secondary_a=secondary_a,
            secondary_c=secondary_c,
        ) + (round(float(corto_det_up), 9),)

    # Optics transport and mosaic-shape changes rescale intensities but do not
    # move detector hits, so a compatible cached hit-table bundle can be reused.
    requested_hit_table_sig = _simulation_signature_base(
        optics_mode_component=0,
        include_mosaic_shape=False,
    )
    primary_contribution_cache_signature = _primary_contribution_cache_signature_base()
    collect_hit_tables_for_job = bool(
        collect_hit_tables_requested
        and not _cached_hit_tables_reusable(
            requested_hit_table_sig,
            run_primary=primary_available,
            run_secondary=secondary_available,
        )
    )

    _set_update_trace_stage("simulation_signature")
    new_sim_image_sig = _simulation_signature_base(optics_mode_component=int(optics_mode_flag))
    new_sim_sig = new_sim_image_sig + (int(collect_hit_tables_for_job),)
    _trace_update(
        "do_update_signature",
        optics_mode=int(optics_mode_flag),
        collect_hit_tables=bool(collect_hit_tables_requested),
        collect_hit_tables_for_job=bool(collect_hit_tables_for_job),
        num_samples=int(simulation_runtime_state.num_samples),
        image_signature_changed=bool(
            new_sim_image_sig != simulation_runtime_state.last_sim_signature
        ),
        full_signature_changed=bool(
            new_sim_sig != simulation_runtime_state.last_simulation_signature
        ),
    )
    ready_simulation_result = _consume_ready_simulation_result(new_sim_sig)
    if ready_simulation_result is not None:
        _trace_update(
            "do_update_apply_ready_simulation",
            job_kind=str(ready_simulation_result.get("job_kind", "full")),
            image_generation_elapsed_ms=float(
                ready_simulation_result.get("image_generation_elapsed_ms", 0.0)
            ),
        )
        if str(ready_simulation_result.get("job_kind", "full")) == "primary_fill":
            _store_primary_cache_payload(
                cache_signature=ready_simulation_result.get("primary_contribution_cache_signature"),
                source_mode=str(
                    ready_simulation_result.get(
                        "primary_source_mode",
                        primary_requested_source_mode,
                    )
                ),
                active_keys=ready_simulation_result.get(
                    "active_primary_contribution_keys",
                    primary_requested_contribution_keys,
                ),
                contribution_keys=ready_simulation_result.get(
                    "primary_contribution_keys",
                    [],
                ),
                raw_hit_tables=ready_simulation_result.get(
                    "primary_hit_tables_raw",
                    [],
                ),
            )
            rematerialized_primary = _rematerialize_primary_cache_artifacts(
                image_size=int(image_size),
                mosaic_params=mosaic_params,
                a_primary=float(a_updated),
                c_primary=float(c_updated),
            )
            _apply_primary_cache_artifacts(rematerialized_primary)
            simulation_runtime_state.stored_hit_table_signature = requested_hit_table_sig
            simulation_runtime_state.primary_filter_signature = primary_requested_filter_signature
        else:
            _apply_ready_simulation_result(ready_simulation_result)
            simulation_runtime_state.primary_filter_signature = primary_requested_filter_signature
        simulation_runtime_state.last_sim_signature = new_sim_image_sig
        simulation_runtime_state.last_simulation_signature = new_sim_sig
        _capture_geometry_source_snapshot()
        image_generation_cached = False
        image_generation_elapsed_ms = float(
            ready_simulation_result.get("image_generation_elapsed_ms", 0.0)
        )
        simulation_runtime_state.update_phase = "applying"
        _refresh_run_status_bar()

    image_signature_changed = bool(new_sim_image_sig != simulation_runtime_state.last_sim_signature)
    need_hit_table_refresh = bool(
        collect_hit_tables_for_job
        and new_sim_sig != simulation_runtime_state.last_simulation_signature
    )

    def _build_simulation_job(
        *,
        collect_hit_tables_enabled: bool,
        job_kind: str = "full",
        primary_job_data: object | None = None,
        primary_job_intensities: np.ndarray | None = None,
        primary_job_keys: Sequence[object] | None = None,
        run_secondary_job: bool | None = None,
    ) -> dict[str, object]:
        selected_primary_data = primary_data if primary_job_data is None else primary_job_data
        if isinstance(selected_primary_data, dict):
            selected_primary_data = gui_controllers.copy_bragg_qr_dict(selected_primary_data)

        selected_primary_intensities = (
            np.asarray(primary_intensities, dtype=np.float64).copy()
            if primary_job_intensities is None
            else np.asarray(primary_job_intensities, dtype=np.float64).reshape(-1).copy()
        )
        selected_primary_keys = list(
            primary_requested_contribution_keys if primary_job_keys is None else primary_job_keys
        )
        run_primary_job = (
            len(selected_primary_data) > 0
            if isinstance(selected_primary_data, dict)
            else (
                np.asarray(selected_primary_data).ndim == 2
                and np.asarray(selected_primary_data).shape[0] > 0
                and selected_primary_intensities.size > 0
            )
        )
        run_secondary_enabled = (
            bool(secondary_available) if run_secondary_job is None else bool(run_secondary_job)
        )
        return {
            "job_kind": str(job_kind),
            "signature": new_sim_sig,
            "epoch": int(simulation_runtime_state.simulation_epoch),
            "image_size": int(image_size),
            "pixel_size_m": float(pixel_size_m),
            "center": np.asarray([center_x_up, center_y_up], dtype=np.float64).copy(),
            "mosaic_params": {
                "beam_x_array": np.asarray(mosaic_params["beam_x_array"], dtype=np.float64).copy(),
                "beam_y_array": np.asarray(mosaic_params["beam_y_array"], dtype=np.float64).copy(),
                "theta_array": np.asarray(mosaic_params["theta_array"], dtype=np.float64).copy(),
                "phi_array": np.asarray(mosaic_params["phi_array"], dtype=np.float64).copy(),
                "wavelength_array": np.asarray(
                    mosaic_params["wavelength_array"],
                    dtype=np.float64,
                ).copy(),
                "sample_weights": (
                    None
                    if mosaic_params.get("sample_weights") is None
                    else np.asarray(
                        mosaic_params["sample_weights"],
                        dtype=np.float64,
                    ).copy()
                ),
                "n2_sample_array": (
                    None
                    if mosaic_params.get("n2_sample_array") is None
                    else np.asarray(
                        mosaic_params["n2_sample_array"],
                        dtype=np.complex128,
                    ).copy()
                ),
                "sigma_mosaic_deg": float(mosaic_params["sigma_mosaic_deg"]),
                "gamma_mosaic_deg": float(mosaic_params["gamma_mosaic_deg"]),
                "eta": float(mosaic_params["eta"]),
                "solve_q_steps": int(mosaic_params["solve_q_steps"]),
                "solve_q_rel_tol": float(mosaic_params["solve_q_rel_tol"]),
                "solve_q_mode": int(mosaic_params["solve_q_mode"]),
                "_sampling_signature": tuple(mosaic_params.get("_sampling_signature", ())),
            },
            "lambda_value": float(lambda_),
            "distance_m": float(corto_det_up),
            "gamma_deg": float(gamma_updated),
            "Gamma_deg": float(Gamma_updated),
            "chi_deg": float(chi_updated),
            "psi_deg": float(psi),
            "psi_z_deg": float(psi_z_updated),
            "zs": float(zs_updated),
            "zb": float(zb_updated),
            "theta_initial_deg": float(theta_init_up),
            "cor_angle_deg": float(cor_angle_updated),
            "sample_width_m": float(sample_width_updated),
            "sample_length_m": float(sample_length_updated),
            "sample_depth_m": float(sample_depth_updated),
            "debye_x": float(debye_x_updated),
            "debye_y": float(debye_y_updated),
            "optics_mode": int(optics_mode_flag),
            "collect_hit_tables": bool(collect_hit_tables_enabled),
            "collect_primary_hit_tables": bool(collect_hit_tables_enabled and run_primary_job),
            "collect_secondary_hit_tables": bool(
                collect_hit_tables_enabled and run_secondary_enabled
            ),
            "capture_primary_hit_tables_raw": bool(run_primary_job),
            "hit_table_signature": requested_hit_table_sig,
            "primary_contribution_cache_signature": primary_contribution_cache_signature,
            "primary_source_mode": primary_requested_source_mode,
            "primary_contribution_keys": list(selected_primary_keys),
            "active_primary_contribution_keys": list(primary_requested_contribution_keys),
            "accumulate_image": bool(accumulate_image_requested),
            "qr_cylinder_replace_simulation": bool(qr_cylinder_replace_requested),
            "n2_value": n2,
            "primary_data": selected_primary_data,
            "primary_intensities": selected_primary_intensities,
            "secondary_data": secondary_data,
            "secondary_intensities": secondary_intensities,
            "run_primary": bool(run_primary_job),
            "run_secondary": bool(run_secondary_enabled),
            "a_primary": float(a_updated),
            "c_primary": float(c_updated),
            "a_secondary": float(secondary_a),
            "c_secondary": float(secondary_c),
        }

    def _build_scaled_primary_subset_payload(
        requested_keys: Sequence[object],
    ) -> dict[str, object]:
        subset_payload = gui_runtime_primary_cache.build_primary_subset_payload(
            source_mode=primary_requested_source_mode,
            all_primary_qr=simulation_runtime_state.sim_primary_qr_all,
            all_miller=simulation_runtime_state.sim_miller1_all,
            all_intensities=simulation_runtime_state.sim_intens1_all,
            requested_keys=requested_keys,
        )
        subset_primary_data = subset_payload.get("primary_data")
        subset_primary_intensities = np.asarray(
            subset_payload.get("primary_intensities", np.empty((0,), dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1)
        if isinstance(subset_primary_data, dict):
            subset_primary_data = _scaled_bragg_qr_dict(
                subset_primary_data,
                ordered_structure_scale,
            )
        else:
            subset_primary_intensities = subset_primary_intensities.copy() * float(
                ordered_structure_scale
            )
        return {
            "primary_data": subset_primary_data,
            "primary_intensities": subset_primary_intensities,
            "primary_contribution_keys": list(subset_payload.get("primary_contribution_keys", [])),
        }

    if ready_simulation_result is None and image_signature_changed:
        _set_update_trace_stage("simulation_generation")
        _trace_update(
            "do_update_regenerate_simulation",
            had_cached_primary=bool(simulation_runtime_state.stored_primary_sim_image is not None),
            had_cached_secondary=bool(
                simulation_runtime_state.stored_secondary_sim_image is not None
            ),
        )
        _invalidate_geometry_manual_pick_cache()
        simulation_runtime_state.peak_positions.clear()
        simulation_runtime_state.peak_millers.clear()
        simulation_runtime_state.peak_intensities.clear()
        simulation_runtime_state.peak_records.clear()
        simulation_runtime_state.selected_peak_record = None
        incremental_sf_prune_action = gui_runtime_primary_cache.resolve_incremental_sf_prune_action(
            cache_signature=primary_contribution_cache_signature,
            cached_signature=simulation_runtime_state.primary_contribution_cache_signature,
            source_mode=primary_requested_source_mode,
            cached_source_mode=simulation_runtime_state.primary_source_mode,
            active_keys=primary_requested_contribution_keys,
            previous_active_keys=simulation_runtime_state.primary_active_contribution_keys,
            primary_hit_table_cache=simulation_runtime_state.primary_hit_table_cache,
            active_job=simulation_runtime_state.worker_active_job,
            queued_job=simulation_runtime_state.worker_queued_job,
        )
        if incremental_sf_prune_action.mode == "reuse":
            rematerialize_start_time = perf_counter()
            simulation_runtime_state.primary_contribution_cache_signature = (
                primary_contribution_cache_signature
            )
            simulation_runtime_state.primary_source_mode = primary_requested_source_mode
            simulation_runtime_state.primary_filter_signature = primary_requested_filter_signature
            simulation_runtime_state.primary_active_contribution_keys = list(
                primary_requested_contribution_keys
            )
            rematerialized_primary = _rematerialize_primary_cache_artifacts(
                image_size=int(image_size),
                mosaic_params=mosaic_params,
                a_primary=float(a_updated),
                c_primary=float(c_updated),
            )
            _apply_primary_cache_artifacts(rematerialized_primary)
            simulation_runtime_state.stored_hit_table_signature = requested_hit_table_sig
            simulation_runtime_state.last_sim_signature = new_sim_image_sig
            simulation_runtime_state.last_simulation_signature = new_sim_sig
            _capture_geometry_source_snapshot()
            image_generation_cached = False
            image_generation_elapsed_ms = (perf_counter() - rematerialize_start_time) * 1e3
            simulation_runtime_state.last_image_generation_ms = float(image_generation_elapsed_ms)
            simulation_runtime_state.update_phase = "applying"
            _refresh_run_status_bar()
        elif incremental_sf_prune_action.mode == "fill":
            subset_payload = _build_scaled_primary_subset_payload(
                incremental_sf_prune_action.missing_keys
            )
            simulation_job = _build_simulation_job(
                collect_hit_tables_enabled=False,
                job_kind="primary_fill",
                primary_job_data=subset_payload["primary_data"],
                primary_job_intensities=subset_payload["primary_intensities"],
                primary_job_keys=subset_payload["primary_contribution_keys"],
                run_secondary_job=False,
            )
            if not bool(simulation_job.get("run_primary", False)):
                incremental_sf_prune_action = gui_runtime_primary_cache.IncrementalSfPruneAction(
                    mode="full",
                    added_keys=incremental_sf_prune_action.added_keys,
                    removed_keys=incremental_sf_prune_action.removed_keys,
                    missing_keys=incremental_sf_prune_action.missing_keys,
                    reason="invalid_fill_subset",
                )
            else:
                request_status = _request_async_simulation_job(simulation_job)
                if (
                    request_status in {"submitted", "queued", "running"}
                    and "progress_label" in globals()
                    and progress_label is not None
                ):
                    progress_label.config(text="Extending primary cache in background...")
                _trace_update(
                    "do_update_extend_primary_cache_in_background",
                    request_status=str(request_status),
                    missing_key_count=int(len(incremental_sf_prune_action.missing_keys)),
                    worker_active=bool(simulation_runtime_state.worker_active_job is not None),
                    worker_queued=bool(simulation_runtime_state.worker_queued_job is not None),
                )
                simulation_runtime_state.update_running = False
                return
        if incremental_sf_prune_action.mode == "full":
            simulation_job = _build_simulation_job(
                collect_hit_tables_enabled=collect_hit_tables_for_job
            )
            preview_job = (
                _build_preview_simulation_job(
                    simulation_job,
                    max_samples=_current_preview_sample_limit(),
                )
                if LIVE_DRAG_PREVIEW_ENABLED
                else None
            )
            has_cached_simulation = (
                simulation_runtime_state.stored_primary_sim_image is not None
                or simulation_runtime_state.stored_secondary_sim_image is not None
            )
            if _live_interaction_active() and LIVE_DRAG_PREVIEW_ENABLED:
                if preview_job is not None:
                    request_status = _request_async_simulation_job(preview_job)
                    if (
                        request_status in {"submitted", "queued", "running"}
                        and "progress_label" in globals()
                        and progress_label is not None
                    ):
                        progress_label.config(text="Computing preview simulation in background...")
                else:
                    request_status = "preview_unavailable"
                _trace_update(
                    "do_update_queue_live_preview",
                    request_status=str(request_status),
                    has_cached_simulation=bool(has_cached_simulation),
                    worker_active=bool(simulation_runtime_state.worker_active_job is not None),
                    worker_queued=bool(simulation_runtime_state.worker_queued_job is not None),
                )
                if not has_cached_simulation:
                    simulation_runtime_state.update_phase = "queued"
                    _refresh_run_status_bar()
                    simulation_runtime_state.update_running = False
                    return
            elif (
                not has_cached_simulation
                and simulation_runtime_state.worker_active_job is None
                and simulation_runtime_state.worker_queued_job is None
            ):
                if "progress_label" in globals() and progress_label is not None:
                    progress_label.config(text="Computing initial simulation...")
                sync_result = _run_simulation_generation_job(
                    {
                        **simulation_job,
                        "job_id": 0,
                    }
                )
                _apply_ready_simulation_result(sync_result)
                simulation_runtime_state.last_sim_signature = new_sim_image_sig
                simulation_runtime_state.last_simulation_signature = new_sim_sig
                simulation_runtime_state.primary_filter_signature = (
                    primary_requested_filter_signature
                )
                _capture_geometry_source_snapshot()
                image_generation_cached = False
                image_generation_elapsed_ms = float(
                    sync_result.get("image_generation_elapsed_ms", 0.0)
                )
                simulation_runtime_state.update_phase = "applying"
                _refresh_run_status_bar()
            else:
                request_status = _request_async_simulation_job(simulation_job)
                if (
                    request_status in {"submitted", "queued", "running"}
                    and "progress_label" in globals()
                    and progress_label is not None
                ):
                    progress_label.config(text="Computing simulation in background...")
                _trace_update(
                    "do_update_return_waiting_for_simulation",
                    request_status=str(request_status),
                    has_cached_simulation=bool(has_cached_simulation),
                    worker_active=bool(simulation_runtime_state.worker_active_job is not None),
                    worker_queued=bool(simulation_runtime_state.worker_queued_job is not None),
                )
                if request_status != "ready":
                    simulation_runtime_state.update_running = False
                    return
    elif ready_simulation_result is None and need_hit_table_refresh:
        _set_update_trace_stage("simulation_generation")
        request_status = _request_async_simulation_job(
            _build_simulation_job(collect_hit_tables_enabled=True)
        )
        if (
            request_status in {"submitted", "queued", "running"}
            and "progress_label" in globals()
            and progress_label is not None
        ):
            progress_label.config(text="Refreshing peak tables in background...")
        _trace_update(
            "do_update_refresh_hit_tables_in_background",
            request_status=str(request_status),
            worker_active=bool(simulation_runtime_state.worker_active_job is not None),
            worker_queued=bool(simulation_runtime_state.worker_queued_job is not None),
        )

    if (
        simulation_runtime_state.stored_primary_sim_image is None
        and simulation_runtime_state.stored_secondary_sim_image is None
    ):
        _trace_update("do_update_return_no_simulation_image")
        simulation_runtime_state.update_phase = "queued"
        _refresh_run_status_bar()
        simulation_runtime_state.update_running = False
        return

    w1 = float(weight1_var.get())
    w2 = float(weight2_var.get())
    run_primary = bool(
        simulation_runtime_state.stored_primary_sim_image is not None and abs(w1) > 1e-12
    )
    run_secondary = bool(
        simulation_runtime_state.stored_secondary_sim_image is not None and abs(w2) > 1e-12
    )

    img1 = (
        simulation_runtime_state.stored_primary_sim_image
        if simulation_runtime_state.stored_primary_sim_image is not None
        else np.zeros((image_size, image_size), dtype=np.float64)
    )
    img2 = (
        simulation_runtime_state.stored_secondary_sim_image
        if simulation_runtime_state.stored_secondary_sim_image is not None
        else np.zeros((image_size, image_size), dtype=np.float64)
    )

    updated_image = w1 * img1 + w2 * img2
    max_positions_local = []
    source_reflection_indices_local = []
    peak_table_lattice_local = []
    if run_primary and simulation_runtime_state.stored_primary_max_positions is not None:
        max_positions_local.extend(simulation_runtime_state.stored_primary_max_positions)
        if simulation_runtime_state.stored_primary_source_reflection_indices is not None:
            source_reflection_indices_local.extend(
                simulation_runtime_state.stored_primary_source_reflection_indices
            )
        if simulation_runtime_state.stored_primary_peak_table_lattice is not None:
            peak_table_lattice_local.extend(
                simulation_runtime_state.stored_primary_peak_table_lattice
            )
    if run_secondary and simulation_runtime_state.stored_secondary_max_positions is not None:
        max_positions_local.extend(simulation_runtime_state.stored_secondary_max_positions)
        if simulation_runtime_state.stored_secondary_source_reflection_indices is not None:
            source_reflection_indices_local.extend(
                simulation_runtime_state.stored_secondary_source_reflection_indices
            )
        if simulation_runtime_state.stored_secondary_peak_table_lattice is not None:
            peak_table_lattice_local.extend(
                simulation_runtime_state.stored_secondary_peak_table_lattice
            )

    simulation_runtime_state.stored_max_positions_local = list(max_positions_local)
    simulation_runtime_state.stored_source_reflection_indices_local = (
        list(source_reflection_indices_local)
        if len(source_reflection_indices_local) == len(max_positions_local)
        else None
    )
    simulation_runtime_state.stored_peak_table_lattice = list(peak_table_lattice_local)
    intersection_cache_local = []
    if run_primary and simulation_runtime_state.stored_primary_intersection_cache is not None:
        intersection_cache_local.extend(
            _copy_intersection_cache_tables(
                simulation_runtime_state.stored_primary_intersection_cache
            )
        )
    if run_secondary and simulation_runtime_state.stored_secondary_intersection_cache is not None:
        intersection_cache_local.extend(
            _copy_intersection_cache_tables(
                simulation_runtime_state.stored_secondary_intersection_cache
            )
        )
    simulation_runtime_state.stored_intersection_cache = intersection_cache_local
    simulation_runtime_state.stored_sim_image = updated_image

    if not peak_table_lattice_local or len(peak_table_lattice_local) != len(max_positions_local):
        peak_table_lattice_local = [
            (float(a_updated), float(c_updated), "primary") for _ in max_positions_local
        ]

    if not need_hit_table_refresh and gui_controllers.consume_geometry_q_group_refresh_request(
        geometry_q_group_state
    ):
        listed_entries = (
            gui_geometry_q_group_manager.capture_runtime_geometry_q_group_entries_snapshot(
                geometry_q_group_runtime_bindings_factory()
            )
        )
        progress_label_geometry.config(
            text=(
                f"Updated listed Qr/Qz peaks: {len(listed_entries)} groups, "
                f"{sum(int(entry.get('peak_count', 0)) for entry in listed_entries)} peaks."
            )
        )

    _set_update_trace_stage("redraw")
    _trace_update(
        "do_update_redraw_start",
        run_primary=bool(run_primary),
        run_secondary=bool(run_secondary),
        combined_peak_count=int(len(max_positions_local)),
    )
    redraw_update_start_time = perf_counter()
    display_image = np.rot90(updated_image, SIM_DISPLAY_ROTATE_K)
    gui_runtime_geometry_interaction.refresh_runtime_peak_selection_after_update(
        maintenance_callbacks=peak_selection_runtime_maintenance,
        live_geometry_preview_enabled=_live_geometry_preview_enabled(),
    )
    normalization_scale = 1.0
    normalization_sig = None
    normalization_cache_outcome = "disabled"
    native_background = _get_current_background_backend()
    if native_background is not None and display_image is not None:
        normalization_sig = (
            new_sim_image_sig,
            int(background_runtime_state.current_background_index),
            id(background_runtime_state.current_background_image),
            int(background_runtime_state.backend_rotation_k) % 4,
            bool(background_runtime_state.backend_flip_x),
            bool(background_runtime_state.backend_flip_y),
            tuple(display_image.shape),
            tuple(np.asarray(native_background).shape),
        )
        if simulation_runtime_state.normalization_scale_cache.get("sig") == normalization_sig:
            normalization_scale = float(
                simulation_runtime_state.normalization_scale_cache.get("value", 1.0)
            )
            normalization_cache_outcome = "reuse"
        else:
            normalization_scale = _suggest_scale_factor(display_image, native_background)
            simulation_runtime_state.normalization_scale_cache["sig"] = normalization_sig
            simulation_runtime_state.normalization_scale_cache["value"] = float(normalization_scale)
            normalization_cache_outcome = "update"
        if not np.isfinite(normalization_scale) or normalization_scale <= 0.0:
            normalization_scale = 1.0
    _trace_live_cache_event(
        "normalization_scale",
        normalization_cache_outcome,
        signature_summary=_live_cache_signature_summary(normalization_sig),
        scale=(float(normalization_scale) if np.isfinite(float(normalization_scale)) else None),
    )

    simulation_runtime_state.unscaled_image = None
    simulation_runtime_state.last_unscaled_image_signature = None
    if display_image is not None:
        if abs(float(normalization_scale) - 1.0) <= 1e-12:
            simulation_runtime_state.unscaled_image = display_image
        else:
            simulation_runtime_state.unscaled_image = display_image * normalization_scale
        simulation_runtime_state.last_unscaled_image_signature = (
            new_sim_image_sig,
            round(float(normalization_scale), 9),
            tuple(display_image.shape),
            int(background_runtime_state.current_background_index),
            id(background_runtime_state.current_background_image),
            int(background_runtime_state.backend_rotation_k) % 4,
            bool(background_runtime_state.backend_flip_x),
            bool(background_runtime_state.backend_flip_y),
        )
        if simulation_runtime_state.peak_intensities and normalization_scale != 1.0:
            simulation_runtime_state.peak_intensities[:] = [
                intensity * normalization_scale
                for intensity in simulation_runtime_state.peak_intensities
            ]
            for rec in simulation_runtime_state.peak_records:
                rec["intensity"] = float(rec.get("intensity", 0.0)) * normalization_scale

    simulation_runtime_state.last_1d_integration_data["simulated_2d_image"] = (
        simulation_runtime_state.unscaled_image
    )

    if simulation_runtime_state.unscaled_image is not None:
        if display_controls_state.scale_factor_user_override:
            _set_scale_factor_value(
                _get_scale_factor_value(default=1.0),
                adjust_range=False,
                reset_override=False,
            )
        else:
            _set_scale_factor_value(
                1.0,
                adjust_range=False,
                reset_override=True,
            )
    # ---------------------------------------------------------------
    # The exact-cake analysis integrator setup is relatively expensive. Cache
    # the runtime integrator instance and only recreate it when the detector
    # distance, detector center, or the fundamental wavelength changes. Tilt
    # updates intentionally do not flush the detector-map or LUT caches so the
    # live detector<->angle transform stays stable during geometry work.
    # ---------------------------------------------------------------
    sig = _caked_geometry_cache_signature(
        corto_det_up,
        center_x_up,
        center_y_up,
        wave_m,
    )
    ai_cache_action = "reuse"
    if simulation_runtime_state.ai_cache.get("sig") != sig:
        ai_cache_action = "rebuild"
        simulation_runtime_state.ai_cache = {
            "sig": sig,
            "ai": _get_azimuthal_integrator_cls()(
                dist=corto_det_up,
                # Keep the legacy row/col mapping aligned with simulation pixels.
                poni1=center_x_up * pixel_size_m,
                poni2=center_y_up * pixel_size_m,
                rot1=0.0,
                rot2=0.0,
                rot3=0.0,
                wavelength=wave_m,
                pixel1=pixel_size_m,
                pixel2=pixel_size_m,
            ),
        }
    _trace_live_cache_event(
        "ai_integrator",
        ai_cache_action,
        signature_summary=_live_cache_signature_summary(sig),
        image_signature_summary=_live_cache_signature_summary(new_sim_image_sig),
    )
    ai = simulation_runtime_state.ai_cache["ai"]
    if simulation_runtime_state.unscaled_image is not None:
        start_exact_cake_geometry_warmup_in_background(
            ai,
            np.asarray(simulation_runtime_state.unscaled_image).shape,
            npt_rad=DEFAULT_ANALYSIS_RADIAL_BINS,
            npt_azim=DEFAULT_ANALYSIS_AZIMUTH_BINS,
        )
    sim_caking_sig = (
        new_sim_image_sig,
        sig,
        round(float(normalization_scale), 9),
    )
    bg_caking_sig = None
    if native_background is not None:
        bg_caking_sig = (
            sig,
            int(background_runtime_state.current_background_index),
            id(background_runtime_state.current_background_image),
            int(background_runtime_state.backend_rotation_k) % 4,
            bool(background_runtime_state.backend_flip_x),
            bool(background_runtime_state.backend_flip_y),
            tuple(np.asarray(native_background).shape),
        )

    show_caked_2d = bool(analysis_view_controls_view_state.show_caked_2d_var.get())
    show_1d_requested = bool(analysis_view_controls_view_state.show_1d_var.get())
    one_d_analysis_requested = bool(
        (not qr_cylinder_replace_requested)
        and show_1d_requested
        and _analysis_integration_outputs_visible()
    )
    caked_analysis_requested = bool(
        show_caked_2d and simulation_runtime_state.unscaled_image is not None
    )
    analysis_requested = bool(
        simulation_runtime_state.unscaled_image is not None
        and (caked_analysis_requested or one_d_analysis_requested)
    )
    analysis_sig = (sim_caking_sig, bg_caking_sig) if analysis_requested else None
    desired_analysis_preview = bool(
        LIVE_DRAG_PREVIEW_ENABLED
        and PREVIEW_CALCULATIONS_ENABLED
        and analysis_requested
        and (simulation_runtime_state.preview_active or _live_interaction_active())
    )
    analysis_bins = (
        (LIVE_DRAG_ANALYSIS_RADIAL_BINS, LIVE_DRAG_ANALYSIS_AZIMUTH_BINS)
        if desired_analysis_preview
        else (DEFAULT_ANALYSIS_RADIAL_BINS, DEFAULT_ANALYSIS_AZIMUTH_BINS)
    )
    sim_cache_sig = (
        (sim_caking_sig, int(analysis_bins[0]), int(analysis_bins[1]))
        if analysis_requested
        else None
    )
    bg_cache_sig = (
        (bg_caking_sig, int(analysis_bins[0]), int(analysis_bins[1]))
        if bg_caking_sig is not None
        else None
    )
    ready_analysis_result = None
    if analysis_sig is not None:
        ready_analysis_result = _consume_ready_analysis_result(
            analysis_sig,
            is_preview=desired_analysis_preview,
        )
        if ready_analysis_result is not None:
            _apply_ready_analysis_result(ready_analysis_result)
            if one_d_analysis_requested and not bool(
                ready_analysis_result.get("is_preview", False)
            ):
                _refresh_integration_from_cached_results()
            simulation_runtime_state.update_phase = "applying"
            _refresh_run_status_bar()

    analysis_result_current = bool(
        analysis_sig is not None
        and simulation_runtime_state.last_analysis_signature == analysis_sig
        and simulation_runtime_state.last_res2_sim is not None
    )
    missing_caked_payload = bool(
        show_caked_2d
        and analysis_result_current
        and (
            simulation_runtime_state.last_caked_image_unscaled is None
            or simulation_runtime_state.last_caked_extent is None
            or (
                _current_app_shell_view_mode() == "q_space"
                and (
                    simulation_runtime_state.last_q_space_image_unscaled is None
                    or simulation_runtime_state.last_q_space_extent is None
                )
            )
        )
    )
    if missing_caked_payload:
        _trace_live_cache_event(
            "analysis",
            "restore_payload",
            outcome="requested",
            reason="missing_caked_payload",
            background_visible=bool(background_runtime_state.visible),
        )
        _restore_caked_display_payload_from_cached_results(
            background_visible=bool(background_runtime_state.visible)
        )
        _trace_live_cache_event(
            "analysis",
            "restore_payload",
            outcome=(
                "restored"
                if (
                    simulation_runtime_state.last_caked_image_unscaled is not None
                    and simulation_runtime_state.last_caked_extent is not None
                )
                else "missing"
            ),
            reason="missing_caked_payload",
            background_visible=bool(background_runtime_state.visible),
        )
    analysis_result_matches_target = bool(
        analysis_result_current
        and bool(simulation_runtime_state.analysis_preview_active) == desired_analysis_preview
        and (
            not show_caked_2d
            or (
                simulation_runtime_state.last_caked_image_unscaled is not None
                and simulation_runtime_state.last_caked_extent is not None
            )
        )
        and (
            _current_app_shell_view_mode() != "q_space"
            or (
                simulation_runtime_state.last_q_space_image_unscaled is not None
                and simulation_runtime_state.last_q_space_extent is not None
            )
        )
    )
    analysis_request_in_flight = bool(
        analysis_sig is not None
        and any(
            _analysis_job_key(payload)
            == (
                analysis_sig,
                int(simulation_runtime_state.analysis_epoch),
                desired_analysis_preview,
            )
            for payload in (
                simulation_runtime_state.analysis_ready_result,
                simulation_runtime_state.analysis_active_job,
                simulation_runtime_state.analysis_queued_job,
            )
        )
    )
    if (
        analysis_sig is not None
        and (not analysis_result_matches_target)
        and not analysis_request_in_flight
    ):
        cached_bg_entry = _get_cached_caking_entry("bg", bg_cache_sig)
        preserve_caked_visuals = bool(
            show_caked_2d
            and simulation_runtime_state.last_caked_image_unscaled is not None
            and simulation_runtime_state.last_caked_extent is not None
        )
        _invalidate_analysis_cache(clear_visuals=not preserve_caked_visuals)
        _request_async_analysis_job(
            {
                "signature": analysis_sig,
                "epoch": int(simulation_runtime_state.analysis_epoch),
                "is_preview": desired_analysis_preview,
                "npt_rad": int(analysis_bins[0]),
                "npt_azim": int(analysis_bins[1]),
                "sim_cache_sig": sim_cache_sig,
                "bg_cache_sig": bg_cache_sig,
                "image": np.asarray(
                    simulation_runtime_state.unscaled_image,
                    dtype=np.float64,
                ).copy(),
                "background_image": (
                    None
                    if not (background_runtime_state.visible and native_background is not None)
                    else np.asarray(native_background, dtype=np.float64).copy()
                ),
                "cached_bg_res2": (
                    None if not isinstance(cached_bg_entry, dict) else cached_bg_entry.get("res2")
                ),
                "cached_bg_caked": (
                    None
                    if not isinstance(cached_bg_entry, dict)
                    else cached_bg_entry.get("payload")
                ),
                "intersection_cache": _copy_intersection_cache_tables(
                    simulation_runtime_state.stored_intersection_cache
                ),
                "distance_m": float(corto_det_up),
                "center": np.asarray(
                    [center_x_up, center_y_up],
                    dtype=np.float64,
                ).copy(),
                "pixel_size_m": float(pixel_size_m),
                "wavelength_m": float(wave_m),
                "sim_caking_sig": sim_caking_sig,
                "bg_caking_sig": bg_caking_sig,
            }
        )
        if "progress_label" in globals() and progress_label is not None:
            if desired_analysis_preview:
                progress_label.config(text=_analysis_progress_text())
            else:
                progress_label.config(text=_analysis_progress_text())

    sim_res2 = simulation_runtime_state.last_res2_sim if analysis_result_current else None
    bg_res2 = simulation_runtime_state.last_res2_background if analysis_result_current else None
    previous_primary_view_mode = _current_primary_figure_mode()
    preserved_primary_limits = gui_canvas_interactions.capture_axis_limits(ax)
    defer_overlay_refresh = _defer_nonessential_redraw()
    analysis_space_display_mode = _resolved_primary_analysis_display_mode()
    analysis_space_display_available = analysis_space_display_mode in {"caked", "q_space"}

    if analysis_space_display_available:
        current_scale = _get_scale_factor_value(default=1.0)
        if analysis_space_display_mode == "q_space":
            analysis_image = np.asarray(
                simulation_runtime_state.last_q_space_image_unscaled,
                dtype=float,
            )
            analysis_background_source = None
            if (
                background_runtime_state.visible
                and simulation_runtime_state.last_q_space_background_image_unscaled is not None
            ):
                analysis_background_source = np.asarray(
                    simulation_runtime_state.last_q_space_background_image_unscaled,
                    dtype=float,
                )
            scaled_analysis_for_limits = analysis_image * current_scale
            display_primary_source = scaled_analysis_for_limits
            display_secondary_source = analysis_background_source
            axis_extent = (
                list(simulation_runtime_state.last_q_space_extent)
                if simulation_runtime_state.last_q_space_extent is not None
                else [0.0, 1.0, -1.0, 1.0]
            )
            x_label = "Qr (A^-1)"
            y_label = "Qz (A^-1)"
        else:
            analysis_image = np.asarray(
                simulation_runtime_state.last_caked_image_unscaled,
                dtype=float,
            )
            analysis_background_source = None
            if (
                background_runtime_state.visible
                and simulation_runtime_state.last_caked_background_image_unscaled is not None
            ):
                analysis_background_source = np.asarray(
                    simulation_runtime_state.last_caked_background_image_unscaled,
                    dtype=float,
                )
            scaled_analysis_for_limits = analysis_image * current_scale
            display_primary_source, display_secondary_source = (
                _geometry_fit_caked_roi_preview_display_sources(
                    show_caked_image=True,
                    simulation_image=scaled_analysis_for_limits,
                    background_image=analysis_background_source,
                )
            )
            if not isinstance(display_primary_source, np.ndarray):
                display_primary_source = scaled_analysis_for_limits
            axis_extent = (
                list(simulation_runtime_state.last_caked_extent)
                if simulation_runtime_state.last_caked_extent is not None
                else [0.0, 90.0, -180.0, 180.0]
            )
            x_label = "2θ (degrees)"
            y_label = "φ (degrees)"

        _store_primary_raster_source(image_display, display_primary_source)
        auto_vmin, auto_vmax = _auto_caked_limits(display_primary_source)

        if not display_controls_state.simulation_limits_user_override:
            _update_simulation_sliders_from_image(display_primary_source, reset_override=True)

        if not simulation_runtime_state.caked_limits_user_override:
            vmin_caked_var.set(auto_vmin)
            vmax_caked_var.set(auto_vmax)

        vmin_val = float(display_controls_view_state.simulation_min_var.get())
        vmax_val = float(display_controls_view_state.simulation_max_var.get())
        global_sim_max = vmax_val

        if not math.isfinite(vmin_val):
            vmin_val = auto_vmin
        if not math.isfinite(vmax_val):
            vmax_val = auto_vmax
        vmin_val, vmax_val = _ensure_valid_range(vmin_val, vmax_val)
        if not math.isfinite(global_sim_max) or global_sim_max <= vmin_val:
            global_sim_max = auto_vmax

        display_vmax = min(vmax_val, global_sim_max)
        if not math.isfinite(display_vmax):
            display_vmax = auto_vmax
        if display_vmax <= vmin_val:
            fallback_vmax = max(global_sim_max, auto_vmax, vmax_val)
            if math.isfinite(fallback_vmax) and fallback_vmax > vmin_val:
                display_vmax = fallback_vmax
            else:
                display_vmax = vmin_val + max(abs(vmin_val) * 1e-3, 1e-3)
        _apply_intensity_display_range(
            image_display,
            display_primary_source,
            vmin_val,
            display_vmax,
        )

        background_analysis_available = False
        if isinstance(display_secondary_source, np.ndarray):
            _store_primary_raster_source(background_display, display_secondary_source)
            bg_display_vmax = vmax_val
            if not math.isfinite(bg_display_vmax):
                bg_display_vmax = auto_vmax
            if not math.isfinite(bg_display_vmax):
                bg_display_vmax = display_vmax
            if bg_display_vmax <= vmin_val:
                fallback_vmax = None
                for candidate in (auto_vmax, display_vmax, vmax_val):
                    if math.isfinite(candidate) and candidate > vmin_val:
                        fallback_vmax = candidate
                        break
                if fallback_vmax is None:
                    fallback_vmax = vmin_val + max(abs(vmin_val) * 1e-3, 1e-3)
                bg_display_vmax = fallback_vmax
            background_display.set_clim(vmin_val, bg_display_vmax)
            background_display.set_visible(True)
            background_analysis_available = True
        else:
            background_display.set_visible(False)

        axis_min_x, axis_max_x, axis_min_y, axis_max_y = axis_extent
        if not background_analysis_available:
            background_display.set_visible(False)
        if not (
            math.isfinite(axis_min_x) and math.isfinite(axis_max_x) and axis_max_x > axis_min_x
        ):
            axis_min_x, axis_max_x = (0.0, 1.0) if analysis_space_display_mode == "q_space" else (0.0, 90.0)
        if not (
            math.isfinite(axis_min_y) and math.isfinite(axis_max_y) and axis_max_y > axis_min_y
        ):
            axis_min_y, axis_max_y = (
                (-1.0, 1.0) if analysis_space_display_mode == "q_space" else (-180.0, 180.0)
            )

        gui_canvas_interactions.restore_axis_view(
            ax,
            preserved_limits=preserved_primary_limits,
            default_xlim=(axis_min_x, axis_max_x),
            default_ylim=(axis_min_y, axis_max_y),
            preserve=(previous_primary_view_mode == analysis_space_display_mode),
        )
        ax.set_aspect("auto")
        gui_main_figure_chrome.set_main_figure_axes_axis_visibility(ax, visible=True)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title("")
        _sync_primary_raster_geometry(view_mode=analysis_space_display_mode)
    else:
        simulation_runtime_state.last_caked_image_unscaled = None
        simulation_runtime_state.last_caked_extent = None
        simulation_runtime_state.last_caked_background_image_unscaled = None
        simulation_runtime_state.last_caked_radial_values = None
        simulation_runtime_state.last_caked_azimuth_values = None
        simulation_runtime_state.last_q_space_image_unscaled = None
        simulation_runtime_state.last_q_space_extent = None
        simulation_runtime_state.last_q_space_background_image_unscaled = None
        simulation_runtime_state.last_q_space_qr_values = None
        simulation_runtime_state.last_q_space_qz_values = None
        detector_background_source = (
            background_runtime_state.current_background_display
            if (
                background_runtime_state.visible
                and background_runtime_state.current_background_display is not None
            )
            else None
        )
        display_primary_source, display_secondary_source = (
            _geometry_fit_caked_roi_preview_display_sources(
                show_caked_image=False,
                simulation_image=global_image_buffer,
                background_image=detector_background_source,
            )
        )
        if not isinstance(display_primary_source, np.ndarray):
            display_primary_source = global_image_buffer
        _store_primary_raster_source(image_display, display_primary_source)
        gui_canvas_interactions.restore_axis_view(
            ax,
            preserved_limits=preserved_primary_limits,
            default_xlim=(0.0, float(image_size)),
            default_ylim=(float(image_size), 0.0),
            preserve=(previous_primary_view_mode == "detector"),
        )
        ax.set_aspect("auto")
        gui_main_figure_chrome.set_main_figure_axes_axis_visibility(ax, visible=False)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.set_title("")
        if isinstance(display_secondary_source, np.ndarray):
            _store_primary_raster_source(
                background_display,
                display_secondary_source,
            )
            background_display.set_clim(
                display_controls_view_state.background_min_var.get(),
                display_controls_view_state.background_max_var.get(),
            )
            background_display.set_visible(True)
        else:
            background_display.set_visible(False)
        _sync_primary_raster_geometry(view_mode="detector")

    gui_main_figure_chrome.apply_main_figure_axes_chrome(
        ax,
        axes_visible=bool(analysis_space_display_available),
    )

    # 1D integration
    if one_d_analysis_requested and sim_res2 is not None and not defer_overlay_refresh:
        _update_1d_plots_from_caked(sim_res2, bg_res2)
    elif not defer_overlay_refresh:
        _clear_1d_plot_cache_and_lines()

    if analysis_result_current:
        simulation_runtime_state.last_res2_sim = sim_res2
        simulation_runtime_state.last_res2_background = bg_res2

    # Recompute the slider bounds from the current simulation on every refresh.
    # Manual overrides are preserved unless they exceed the refreshed bounds.
    apply_scale_factor_to_existing_results(
        update_limits=False,
        update_1d=False,
        update_chi_square=not defer_overlay_refresh,
    )

    if defer_overlay_refresh:
        if _live_interaction_active():
            _clear_deferred_overlays(clear_qr_overlay=True)
        else:
            qr_cylinder_overlay_runtime_refresh(redraw=True, update_status=False)
            _clear_deferred_overlays(clear_qr_overlay=False)
    else:
        _refresh_settled_overlays()

    redraw_update_elapsed_ms = (perf_counter() - redraw_update_start_time) * 1e3
    total_update_elapsed_ms = (perf_counter() - update_start_time) * 1e3
    image_generation_text = (
        "cached" if image_generation_cached else f"{image_generation_elapsed_ms:.1f} ms"
    )
    if "update_timing_label" in globals():
        update_timing_label.config(
            text=(
                "Timing | image generation: "
                f"{image_generation_text} | redraw/update: "
                f"{redraw_update_elapsed_ms:.1f} ms | total: "
                f"{total_update_elapsed_ms:.1f} ms"
            )
        )

    # mark update completion so future updates can run
    simulation_runtime_state.last_total_update_ms = float(total_update_elapsed_ms)
    if simulation_runtime_state.worker_active_job is not None:
        simulation_runtime_state.update_phase = "computing"
    elif simulation_runtime_state.worker_queued_job is not None:
        simulation_runtime_state.update_phase = "queued"
    elif simulation_runtime_state.analysis_active_job is not None:
        simulation_runtime_state.update_phase = "analyzing"
    elif simulation_runtime_state.analysis_queued_job is not None:
        simulation_runtime_state.update_phase = "queued"
    else:
        simulation_runtime_state.update_phase = "ready"
    _set_update_trace_stage("complete")
    _trace_update(
        "do_update_complete",
        image_generation_cached=bool(image_generation_cached),
        image_generation_elapsed_ms=float(image_generation_elapsed_ms),
        redraw_update_elapsed_ms=float(redraw_update_elapsed_ms),
        total_update_elapsed_ms=float(total_update_elapsed_ms),
        next_phase=str(simulation_runtime_state.update_phase),
        analysis_result_current=bool(analysis_result_current),
    )
    simulation_runtime_state.update_running = False
    if "progress_label" in globals() and progress_label is not None:
        try:
            current_progress_text = str(progress_label.cget("text"))
        except Exception:
            current_progress_text = ""
        if (
            simulation_runtime_state.preview_active
            and simulation_runtime_state.worker_active_job is not None
        ):
            progress_label.config(text="Preview ready, refining full simulation...")
        elif simulation_runtime_state.analysis_active_job is not None:
            progress_label.config(text=_analysis_progress_text())
        elif current_progress_text in {
            "Computing initial simulation...",
            "Computing preview simulation...",
            "Computing simulation in background...",
            "Simulation loading in background...",
            "Preview ready, refining full simulation...",
            "Updating low-res caked preview...",
            "Updating caked integration in background...",
        }:
            progress_label.config(text="Simulation ready.")
    _refresh_run_status_bar()


background_theta_workflow = gui_runtime_background.build_runtime_background_theta_workflow(
    bootstrap_module=gui_bootstrap,
    background_theta_module=gui_background_theta,
    osc_files_factory=lambda: tuple(background_runtime_state.osc_files),
    current_background_index_factory=(
        lambda: int(background_runtime_state.current_background_index)
    ),
    theta_initial_var_factory=lambda: globals().get("theta_initial_var"),
    defaults=defaults,
    theta_initial=theta_initial,
    background_theta_list_var_factory=lambda: background_theta_list_var,
    geometry_theta_offset_var_factory=lambda: geometry_theta_offset_var,
    geometry_fit_background_selection_var_factory=(lambda: geometry_fit_background_selection_var),
    fit_theta_checkbutton_factory=lambda: fit_theta_checkbutton,
    theta_controls_factory=lambda: geometry_fit_constraints_view_state.controls,
    set_background_file_status_text_factory=lambda: _refresh_background_status,
    schedule_update_factory=lambda: (
        globals().get("schedule_update") if callable(globals().get("schedule_update")) else None
    ),
    progress_label_factory=lambda: globals().get("progress_label"),
    progress_label_geometry_factory=lambda: globals().get("progress_label_geometry"),
)
background_theta_runtime = background_theta_workflow.runtime
background_theta_runtime_callbacks = background_theta_workflow.callbacks
_current_geometry_fit_background_indices = (
    background_theta_workflow.current_geometry_fit_background_indices
)
_geometry_fit_uses_shared_theta_offset = (
    background_theta_workflow.geometry_fit_uses_shared_theta_offset
)
_current_geometry_theta_offset = background_theta_workflow.current_geometry_theta_offset
_current_background_theta_values = background_theta_workflow.current_background_theta_values
_background_theta_for_index = background_theta_workflow.background_theta_for_index
_sync_background_theta_controls = background_theta_workflow.sync_background_theta_controls
_apply_background_theta_metadata_base = background_theta_workflow.apply_background_theta_metadata
_apply_geometry_fit_background_selection = (
    background_theta_workflow.apply_geometry_fit_background_selection
)
_sync_geometry_fit_background_selection = (
    background_theta_workflow.sync_geometry_fit_background_selection
)


def _apply_background_theta_metadata(*args, **kwargs):
    """Apply theta metadata and invalidate geometry-fit shape anchors on change."""

    result = _apply_background_theta_metadata_base(*args, **kwargs)
    if bool(result):
        _clear_geometry_fit_dataset_cache()
    refresh_selector = globals().get("_refresh_geometry_fit_background_table")
    if callable(refresh_selector):
        refresh_selector(force_rebuild=False)
    return result


_apply_geometry_fit_background_selection_base = _apply_geometry_fit_background_selection


def _apply_geometry_fit_background_selection(*args, **kwargs):
    """Apply multi-background geometry selection and invalidate cached anchors."""

    result = _apply_geometry_fit_background_selection_base(*args, **kwargs)
    if bool(result):
        _clear_geometry_fit_dataset_cache()
    refresh_selector = globals().get("_refresh_geometry_fit_background_table")
    if callable(refresh_selector):
        refresh_selector(force_rebuild=False)
    return result


_sync_geometry_fit_background_selection_base = _sync_geometry_fit_background_selection


def _sync_geometry_fit_background_selection(*args, **kwargs):
    """Keep the visual fit-background selector synced to the canonical text var."""

    _sync_geometry_fit_background_selection_base(*args, **kwargs)
    refresh_selector = globals().get("_refresh_geometry_fit_background_table")
    if callable(refresh_selector):
        refresh_selector(force_rebuild=False)


def _background_theta_base_for_index(
    index: int,
    *,
    strict_count: bool = False,
) -> float:
    """Return the stored per-background theta_i value for one loaded image."""

    try:
        theta_values = list(_current_background_theta_values(strict_count=strict_count))
    except Exception:
        theta_values = []
    if theta_values:
        idx = max(0, min(int(index), len(theta_values) - 1))
        return float(theta_values[idx])
    try:
        return float(theta_initial_var.get())
    except Exception:
        return float(theta_initial)


def _current_effective_theta_initial(*, strict_count: bool = False) -> float:
    """Return the current simulation theta including any shared fit offset."""

    background_index = int(background_runtime_state.current_background_index)
    try:
        return float(
            _background_theta_for_index(
                background_index,
                strict_count=strict_count,
            )
        )
    except Exception:
        return _background_theta_base_for_index(
            background_index,
            strict_count=strict_count,
        )


def _sync_theta_initial_to_background(index: int) -> None:
    """Best-effort bridge for background changes before slider vars exist."""

    theta_var = globals().get("theta_initial_var")
    if theta_var is None:
        return
    try:
        theta_var.set(
            _background_theta_base_for_index(
                index,
                strict_count=False,
            )
        )
    except Exception:
        return


background_workflow = gui_runtime_background.build_runtime_background_workflow(
    bootstrap_module=gui_bootstrap,
    background_manager_module=gui_background_manager,
    views_module=gui_views,
    workspace_view_state=workspace_panels_view_state,
    background_backend_debug_view_state=background_backend_debug_view_state,
    background_state=background_runtime_state,
    image_size=image_size,
    display_rotate_k=DISPLAY_ROTATE_K,
    read_osc=read_osc,
    current_background_theta_values=(lambda: _current_background_theta_values(strict_count=False)),
    background_theta_for_index=(lambda idx: _background_theta_for_index(idx, strict_count=False)),
    geometry_fit_uses_shared_theta_offset=_geometry_fit_uses_shared_theta_offset,
    geometry_manual_pairs_for_index=_geometry_manual_pairs_for_index,
    geometry_manual_pair_group_count=_geometry_manual_pair_group_count,
    current_geometry_fit_background_indices=(
        lambda: _current_geometry_fit_background_indices(strict=False)
    ),
    sync_background_runtime_state=_sync_background_runtime_state,
    replace_geometry_manual_pairs_by_background=_replace_geometry_manual_pairs_by_background,
    invalidate_geometry_manual_pick_cache=_invalidate_geometry_manual_pick_cache,
    clear_geometry_manual_undo_stack=_clear_geometry_manual_undo_stack,
    clear_geometry_fit_undo_stack=_clear_geometry_fit_undo_stack,
    set_geometry_manual_pick_mode=_set_geometry_manual_pick_mode,
    set_background_display_data=background_display.set_data,
    set_background_alpha=background_display.set_alpha,
    update_background_slider_defaults=(
        lambda image: _update_background_slider_defaults(
            image,
            reset_override=True,
        )
    ),
    sync_background_theta_controls=(
        lambda: _sync_background_theta_controls(
            preserve_existing=True,
            trigger_update=False,
        )
    ),
    sync_geometry_fit_background_selection=(
        lambda: _sync_geometry_fit_background_selection(
            preserve_existing=True,
        )
    ),
    clear_geometry_pick_artists=_clear_geometry_pick_artists,
    sync_theta_initial_to_background=_sync_theta_initial_to_background,
    render_current_geometry_manual_pairs=(
        lambda: _render_current_geometry_manual_pairs(update_status=False)
    ),
    caked_view_active=_active_caked_primary_view,
    mark_chi_square_dirty=_mark_chi_square_dirty,
    refresh_chi_square_display=lambda: _update_chi_square_display(force=True),
    schedule_update_factory=lambda: schedule_update,
    preempt_simulation_update_factory=(lambda: _preempt_simulation_update_for_background_switch),
    set_status_text_factory=lambda: (
        (lambda text: progress_label.config(text=text)) if "progress_label" in globals() else None
    ),
    file_dialog_dir_factory=lambda: get_dir("file_dialog_dir"),
    askopenfilenames=filedialog.askopenfilenames,
)
background_runtime = background_workflow.runtime
background_runtime_bindings_factory = background_workflow.bindings_factory
background_runtime_callbacks = background_workflow.callbacks
background_controls_runtime = background_workflow.controls_runtime
toggle_background = background_workflow.toggle_visibility
switch_background = background_workflow.switch_background


def reset_to_defaults():
    _clear_geometry_fit_undo_stack()
    theta_initial_var.set(defaults["theta_initial"])
    if geometry_theta_offset_var is not None:
        geometry_theta_offset_var.set("0.0")
    cor_angle_var.set(defaults["cor_angle"])
    gamma_var.set(defaults["gamma"])
    Gamma_var.set(defaults["Gamma"])
    chi_var.set(defaults["chi"])
    psi_z_var.set(defaults["psi_z"])
    zs_var.set(defaults["zs"])
    zb_var.set(defaults["zb"])
    sample_width_var.set(defaults["sample_width_m"])
    sample_length_var.set(defaults["sample_length_m"])
    sample_depth_var.set(defaults["sample_depth_m"])
    debye_x_var.set(defaults["debye_x"])
    debye_y_var.set(defaults["debye_y"])
    ordered_structure_scale_var.set(float(defaults.get("ordered_structure_scale", 1.0)))
    corto_detector_var.set(defaults["corto_detector"])
    sigma_mosaic_var.set(defaults["sigma_mosaic_deg"])
    gamma_mosaic_var.set(defaults["gamma_mosaic_deg"])
    eta_var.set(defaults["eta"])
    bandwidth_percent_var.set(
        _clip_bandwidth_percent(defaults.get("bandwidth_percent", bandwidth * 100.0))
    )
    a_var.set(defaults["a"])
    c_var.set(defaults["c"])
    default_sample_count = gui_controllers.normalize_sample_count(
        defaults.get("sampling_count", DEFAULT_RANDOM_SAMPLE_COUNT),
        DEFAULT_RANDOM_SAMPLE_COUNT,
        minimum=MIN_RANDOM_SAMPLE_COUNT,
        maximum=MAX_RANDOM_SAMPLE_COUNT,
    )
    resolution_var.set(CUSTOM_SAMPLING_OPTION)
    custom_samples_var.set(default_sample_count)
    pruning_defaults = gui_runtime_fit_analysis.resolve_runtime_pruning_control_defaults(
        structure_factor_pruning_module=gui_structure_factor_pruning,
        raw_prune_bias=defaults.get("sf_prune_bias", 0.0),
        raw_solve_q_steps=defaults.get("solve_q_steps", DEFAULT_SOLVE_Q_STEPS),
        raw_solve_q_rel_tol=defaults.get(
            "solve_q_rel_tol",
            DEFAULT_SOLVE_Q_REL_TOL,
        ),
        raw_solve_q_mode=defaults.get("solve_q_mode", SOLVE_Q_MODE_UNIFORM),
        prune_bias_fallback=defaults.get("sf_prune_bias", 0.0),
        prune_bias_minimum=gui_controllers.SF_PRUNE_BIAS_MIN,
        prune_bias_maximum=gui_controllers.SF_PRUNE_BIAS_MAX,
        steps_fallback=defaults.get("solve_q_steps", DEFAULT_SOLVE_Q_STEPS),
        steps_minimum=MIN_SOLVE_Q_STEPS,
        steps_maximum=MAX_SOLVE_Q_STEPS,
        rel_tol_fallback=defaults.get("solve_q_rel_tol", DEFAULT_SOLVE_Q_REL_TOL),
        rel_tol_minimum=MIN_SOLVE_Q_REL_TOL,
        rel_tol_maximum=MAX_SOLVE_Q_REL_TOL,
        uniform_flag=SOLVE_Q_MODE_UNIFORM,
        adaptive_flag=SOLVE_Q_MODE_ADAPTIVE,
    )
    optics_mode_var.set(_normalize_optics_mode_label(defaults.get("optics_mode", "fast")))
    gui_structure_factor_pruning.apply_runtime_structure_factor_pruning_defaults(
        structure_factor_pruning_controls_view_state,
        pruning_defaults,
    )
    center_x_var.set(defaults["center_x"])
    center_y_var.set(defaults["center_y"])
    _sync_background_theta_controls(preserve_existing=False, trigger_update=False)
    if geometry_fit_background_selection_var is not None:
        geometry_fit_background_selection_var.set(_default_geometry_fit_background_selection())
        _apply_geometry_fit_background_selection(trigger_update=False)
    tth_min_var.set(0.0)
    tth_max_var.set(80.0)
    phi_min_var.set(-15.0)
    phi_max_var.set(15.0)
    analysis_view_controls_view_state.show_1d_var.set(False)
    analysis_view_controls_view_state.show_caked_2d_var.set(False)
    vmin_caked_var.set(0.0)
    vmax_caked_var.set(2000.0)
    simulation_runtime_state.caked_limits_user_override = False

    display_controls_state.background_limits_user_override = False
    display_controls_state.simulation_limits_user_override = False
    display_controls_state.scale_factor_user_override = False

    _update_background_slider_defaults(
        background_runtime_state.current_background_display, reset_override=True
    )

    display_controls_state.suppress_simulation_limit_callback = True
    display_controls_view_state.simulation_min_var.set(0.0)
    display_controls_view_state.simulation_max_var.set(background_vmax_default)
    display_controls_state.suppress_simulation_limit_callback = False

    _set_scale_factor_value(1.0, adjust_range=False, reset_override=True)

    # ALSO reset occupancies to defaults for all configured unique atom sites.
    occupancy_vars = _occupancy_control_vars()
    for idx, occ_var in enumerate(occupancy_vars):
        default_occ = occ[idx] if idx < len(occ) else occ[-1]
        occ_var.set(default_occ)
    atom_site_vars = _atom_site_fractional_control_vars()
    atom_site_rows = _atom_site_fractional_rows()
    for idx, axis_vars in enumerate(atom_site_vars):
        if idx >= len(atom_site_rows):
            break
        row = atom_site_rows[idx]
        axis_vars["x"].set(float(row["x"]))
        axis_vars["y"].set(float(row["y"]))
        axis_vars["z"].set(float(row["z"]))
    _reset_atom_site_override_cache()
    p0_var.set(defaults["p0"])
    p1_var.set(defaults["p1"])
    p2_var.set(defaults["p2"])
    w0_var.set(defaults["w0"])
    w1_var.set(defaults["w1"])
    w2_var.set(defaults["w2"])
    finite_stack_var.set(defaults["finite_stack"])
    stack_layers_var.set(int(defaults["stack_layers"]))
    phase_delta_expr_var.set(str(defaults["phase_delta_expression"]))
    phi_l_divisor_var.set(float(defaults["phi_l_divisor"]))
    ordered_structure_fit_debye_x_var.set(True)
    ordered_structure_fit_debye_y_var.set(True)
    ordered_structure_coord_window_var.set(float(ordered_structure_coord_window_default))
    app_state.ordered_structure_fit_snapshot = None
    _set_ordered_structure_revert_enabled(False)
    _set_ordered_structure_result_text("No ordered-structure fit run yet.")
    progress_label_ordered_structure.config(text="Ordered structure fit: waiting.")
    if finite_stack_controls_view_state.phase_delta_entry_var is not None:
        gui_views.set_finite_stack_phase_delta_entry_text(
            finite_stack_controls_view_state,
            _current_phase_delta_expression(),
        )
    if finite_stack_controls_view_state.phi_l_divisor_entry_var is not None:
        gui_views.set_finite_stack_phi_l_divisor_entry_text(
            finite_stack_controls_view_state,
            gui_controllers.format_finite_stack_phi_l_divisor(_current_phi_l_divisor()),
        )
    _sync_finite_controls()

    update_mosaic_cache()
    _invalidate_simulation_cache()
    schedule_update()


background_controls_runtime.create_workspace_controls()

gui_views.create_background_theta_controls(
    parent=workspace_panels_view_state.workspace_backgrounds_frame,
    view_state=background_theta_controls_view_state,
    background_theta_values_text=_format_background_theta_values(
        [_background_theta_default_value()] * len(background_runtime_state.osc_files)
    ),
    geometry_theta_offset_text="0.0",
    on_apply=lambda: _apply_background_theta_metadata(trigger_update=True),
)
background_theta_list_var = background_theta_controls_view_state.background_theta_list_var
geometry_theta_offset_var = background_theta_controls_view_state.geometry_theta_offset_var
_sync_background_theta_controls(preserve_existing=True, trigger_update=False)

_theta_live_to_background_list_sync = {"active": False}
_theta_initial_background_theta_trace = {"attached": False}


def _sync_live_theta_into_background_theta_list(*_args) -> None:
    if _theta_live_to_background_list_sync["active"]:
        return
    _theta_live_to_background_list_sync["active"] = True
    try:
        gui_background_theta.sync_live_theta_to_background_theta_list(
            osc_files=background_runtime_state.osc_files,
            current_background_index=int(background_runtime_state.current_background_index),
            theta_initial_var=theta_initial_var,
            defaults=defaults,
            theta_initial=theta_initial,
            background_theta_list_var=background_theta_list_var,
        )
    finally:
        _theta_live_to_background_list_sync["active"] = False


def _attach_live_theta_background_theta_trace(theta_var: object | None = None) -> None:
    if _theta_initial_background_theta_trace["attached"]:
        return

    live_theta_var = globals().get("theta_initial_var") if theta_var is None else theta_var
    trace_add = getattr(live_theta_var, "trace_add", None)
    if not callable(trace_add):
        return

    trace_add("write", _sync_live_theta_into_background_theta_list)
    _theta_initial_background_theta_trace["attached"] = True


_geometry_fit_background_table_trace = {"attached": False}


def _geometry_fit_background_current_index() -> int:
    total_count = max(0, len(background_runtime_state.osc_files))
    if total_count <= 0:
        return 0
    return max(
        0,
        min(int(background_runtime_state.current_background_index), total_count - 1),
    )


def _build_geometry_fit_background_table_rows() -> list[dict[str, str]]:
    try:
        theta_values = list(_current_background_theta_values(strict_count=False))
    except Exception:
        theta_values = []
    manual_pairs_state = globals().get("geometry_manual_state")
    manual_pairs_map = getattr(manual_pairs_state, "pairs_by_background", {}) or {}
    rows: list[dict[str, str]] = []
    for idx, osc_path in enumerate(list(background_runtime_state.osc_files)):
        theta_text = "n/a"
        if idx < len(theta_values):
            try:
                theta_value = float(theta_values[idx])
            except Exception:
                theta_value = float("nan")
            if np.isfinite(theta_value):
                theta_text = f"{theta_value:.4f} deg"
        pair_count = 0
        try:
            pair_count = len(manual_pairs_map.get(int(idx), ()) or ())
        except Exception:
            pair_count = 0
        rows.append(
            {
                "background": Path(str(osc_path)).name,
                "theta_i": theta_text,
                "pairs": str(max(0, int(pair_count))),
            }
        )
    return rows


def _refresh_geometry_fit_background_table(*, force_rebuild: bool = False) -> None:
    rows = _build_geometry_fit_background_table_rows()
    if background_theta_controls_view_state.geometry_fit_background_rows_frame is None:
        return
    if force_rebuild or len(
        background_theta_controls_view_state.geometry_fit_background_row_frames
    ) != len(rows):
        gui_views.populate_geometry_fit_background_table(
            view_state=background_theta_controls_view_state,
            row_count=len(rows),
            on_toggle=_handle_geometry_fit_background_toggle,
        )
    try:
        selected_indices = list(_current_geometry_fit_background_indices(strict=False))
    except Exception:
        selected_indices = []
    gui_views.update_geometry_fit_background_table(
        view_state=background_theta_controls_view_state,
        rows=rows,
        selected_indices=selected_indices,
        current_index=_geometry_fit_background_current_index(),
    )


def _set_geometry_fit_background_selection_text(
    value: str,
    *,
    trigger_update: bool,
) -> None:
    if geometry_fit_background_selection_var is None:
        return
    geometry_fit_background_selection_var.set(str(value))
    _apply_geometry_fit_background_selection(trigger_update=bool(trigger_update))


def _handle_geometry_fit_background_toggle(index: int) -> None:
    if bool(background_theta_controls_view_state.geometry_fit_background_sync_active):
        return
    if geometry_fit_background_selection_var is None:
        return
    total_count = max(0, len(background_runtime_state.osc_files))
    if total_count <= 0:
        return
    selected_indices: list[int] = []
    for idx, include_var in enumerate(
        background_theta_controls_view_state.geometry_fit_background_include_vars
    ):
        getter = getattr(include_var, "get", None)
        if callable(getter) and bool(getter()):
            selected_indices.append(int(idx))
    if not selected_indices:
        selected_indices = [_geometry_fit_background_current_index()]
    geometry_fit_background_selection_var.set(
        gui_background_theta.serialize_geometry_fit_background_selection(
            selected_indices=selected_indices,
            total_count=total_count,
            current_index=_geometry_fit_background_current_index(),
        )
    )
    _apply_geometry_fit_background_selection(trigger_update=True)


def _attach_geometry_fit_background_selection_trace() -> None:
    if _geometry_fit_background_table_trace["attached"]:
        return
    trace_add = getattr(geometry_fit_background_selection_var, "trace_add", None)
    if not callable(trace_add):
        return
    trace_add(
        "write",
        lambda *_args: _refresh_geometry_fit_background_table(force_rebuild=False),
    )
    _geometry_fit_background_table_trace["attached"] = True


gui_views.create_geometry_fit_background_controls(
    parent=app_shell_view_state.match_backgrounds_frame,
    view_state=background_theta_controls_view_state,
    selection_text=_default_geometry_fit_background_selection(),
    on_apply=lambda: _apply_geometry_fit_background_selection(trigger_update=True),
    on_select_current=lambda: _set_geometry_fit_background_selection_text(
        "current",
        trigger_update=True,
    ),
    on_select_all=lambda: _set_geometry_fit_background_selection_text(
        ("all" if len(background_runtime_state.osc_files) > 1 else "current"),
        trigger_update=True,
    ),
    use_labelframe=False,
)
geometry_fit_background_selection_var = (
    background_theta_controls_view_state.geometry_fit_background_selection_var
)
_attach_geometry_fit_background_selection_trace()
_sync_geometry_fit_background_selection(preserve_existing=False)
_refresh_geometry_fit_background_table(force_rebuild=True)
_maybe_refresh_run_status_bar()

gui_views.populate_stacked_button_group(
    workspace_panels_view_state.workspace_actions_frame,
    [
        ("Reset to Defaults", reset_to_defaults),
    ],
)

gui_views.populate_stacked_button_group(
    (
        workspace_panels_view_state.workspace_debug_frame.frame
        if workspace_panels_view_state.workspace_debug_frame is not None
        else workspace_panels_view_state.workspace_actions_frame
    ),
    [
        (
            "Azim vs Radial Plot Demo",
            lambda: view_azimuthal_radial(
                simulate_diffraction(
                    theta_initial=_current_effective_theta_initial(strict_count=False),
                    cor_angle=cor_angle_var.get(),
                    gamma=gamma_var.get(),
                    Gamma=Gamma_var.get(),
                    chi=chi_var.get(),
                    psi_z=psi_z_var.get(),
                    zs=zs_var.get(),
                    zb=zb_var.get(),
                    debye_x_value=debye_x_var.get(),
                    debye_y_value=debye_y_var.get(),
                    corto_detector_value=corto_detector_var.get(),
                    miller=miller,
                    intensities=intensities,
                    image_size=image_size,
                    av=a_var.get(),
                    cv=c_var.get(),
                    lambda_=lambda_,
                    psi=psi,
                    n2=n2,
                    center=[center_x_var.get(), center_y_var.get()],
                    num_samples=simulation_runtime_state.num_samples,
                    divergence_sigma=divergence_sigma,
                    bw_sigma=bw_sigma,
                    sigma_mosaic_var=sigma_mosaic_var,
                    gamma_mosaic_var=gamma_mosaic_var,
                    eta_var=eta_var,
                    bandwidth=_current_bandwidth_fraction(),
                    optics_mode=_current_optics_mode_flag(),
                    solve_q_steps=current_solve_q_values().steps,
                    solve_q_rel_tol=current_solve_q_values().rel_tol,
                    solve_q_mode=current_solve_q_values().mode_flag,
                    profile_samples=simulation_runtime_state.profile_cache,
                    pixel_size_m=float(pixel_size_m),
                    sample_width_m=float(sample_width_var.get()),
                    sample_length_m=float(sample_length_var.get()),
                    thickness=float(sample_depth_var.get()),
                    n2_sample_array=simulation_runtime_state.profile_cache.get("n2_sample_array"),
                ),
                [center_x_var.get(), center_y_var.get()],
                {
                    "pixel_size": pixel_size_m,
                    "poni1": (center_x_var.get()) * pixel_size_m,
                    "poni2": (center_y_var.get()) * pixel_size_m,
                    "dist": corto_detector_var.get(),
                    "rot1": 0.0,
                    "rot2": 0.0,
                    "rot3": 0.0,
                    "wavelength": wave_m,
                },
            ),
        ),
    ],
)

gui_views.create_status_panel(
    parent=app_shell_view_state.status_frame,
    view_state=status_panel_view_state,
)
progress_label_positions = status_panel_view_state.progress_label_positions
progress_label_geometry = status_panel_view_state.progress_label_geometry
ordered_structure_progressbar = status_panel_view_state.ordered_structure_progressbar
progress_label_ordered_structure = status_panel_view_state.progress_label_ordered_structure
mosaic_progressbar = status_panel_view_state.mosaic_progressbar
progress_label_mosaic = status_panel_view_state.progress_label_mosaic
progress_label = status_panel_view_state.progress_label
update_timing_label = status_panel_view_state.update_timing_label
chi_square_label = status_panel_view_state.chi_square_label
if (
    progress_label_positions is None
    or progress_label_geometry is None
    or ordered_structure_progressbar is None
    or progress_label_ordered_structure is None
    or mosaic_progressbar is None
    or progress_label_mosaic is None
    or progress_label is None
    or update_timing_label is None
    or chi_square_label is None
):
    raise RuntimeError("Status panel was not created.")
progress_label_ordered_structure.config(text="Ordered structure fit: waiting.")

hbn_geometry_debug_view_state.report_text = (
    "No hBN geometry debug report yet.\nImport an hBN bundle to generate one."
)


def _close_hbn_geometry_debug_window() -> None:
    gui_views.close_hbn_geometry_debug_window(hbn_geometry_debug_view_state)


def show_last_hbn_geometry_debug():
    """Display the most recent hBN->simulation geometry debug report."""

    gui_views.open_hbn_geometry_debug_window(
        root=root,
        view_state=hbn_geometry_debug_view_state,
        text=str(hbn_geometry_debug_view_state.report_text),
        on_close=_close_hbn_geometry_debug_window,
    )


def import_hbn_tilt_from_bundle():
    """Load an hBN ellipse bundle NPZ and apply its tilt hint to the GUI sliders."""

    bundle_path = filedialog.askopenfilename(
        title="Select hBN bundle (.npz)",
        filetypes=[("hBN bundle", "*.npz"), ("All files", "*.*")],
    )
    if not bundle_path:
        return

    mean_dist = None
    tilt_x_deg = None
    tilt_y_deg = None
    source_rotate_k = int(HBN_FITTER_ROTATE_K)
    gamma_sign_from_tilt_x = 1
    gamma_sign_from_tilt_y = 1
    imported_center = None

    def _normalize_sign(value, default=1):
        try:
            iv = int(value)
        except Exception:
            iv = int(default)
        if iv < 0:
            return -1
        if iv > 0:
            return 1
        return -1 if int(default) < 0 else 1

    try:
        _, _, _, _, distance_info, tilt_correction, tilt_hint, _, imported_center = load_bundle_npz(
            bundle_path
        )
    except Exception as exc:  # pragma: no cover - GUI interaction
        progress_label.config(text=f"Failed to load bundle: {exc}")
        return

    if isinstance(tilt_correction, dict):
        try:
            tx = float(tilt_correction.get("tilt_x_deg"))
            ty = float(tilt_correction.get("tilt_y_deg"))
            if np.isfinite(tx) and np.isfinite(ty):
                tilt_x_deg = tx
                tilt_y_deg = ty
        except (TypeError, ValueError):
            pass
        source_rotate_k = int(tilt_correction.get("sim_background_rotate_k", source_rotate_k))
        gamma_sign_from_tilt_x = _normalize_sign(
            tilt_correction.get("simulation_gamma_sign_from_tilt_x", gamma_sign_from_tilt_x),
            gamma_sign_from_tilt_x,
        )
        gamma_sign_from_tilt_y = _normalize_sign(
            tilt_correction.get("simulation_Gamma_sign_from_tilt_y", gamma_sign_from_tilt_y),
            gamma_sign_from_tilt_y,
        )

    if (tilt_x_deg is None or tilt_y_deg is None) and isinstance(tilt_hint, dict):
        try:
            rx = float(tilt_hint.get("rot1_rad"))
            ry = float(tilt_hint.get("rot2_rad"))
            if np.isfinite(rx) and np.isfinite(ry):
                tilt_x_deg = float(np.degrees(rx))
                tilt_y_deg = float(np.degrees(ry))
        except (TypeError, ValueError):
            pass
        source_rotate_k = int(tilt_hint.get("sim_background_rotate_k", source_rotate_k))
        gamma_sign_from_tilt_x = _normalize_sign(
            tilt_hint.get("simulation_gamma_sign_from_tilt_x", gamma_sign_from_tilt_x),
            gamma_sign_from_tilt_x,
        )
        gamma_sign_from_tilt_y = _normalize_sign(
            tilt_hint.get("simulation_Gamma_sign_from_tilt_y", gamma_sign_from_tilt_y),
            gamma_sign_from_tilt_y,
        )

    if tilt_x_deg is None or tilt_y_deg is None:
        progress_label.config(text="Bundle loaded, but no tilt information was found.")
        return

    gamma_sign_from_tilt_x = -_normalize_sign(gamma_sign_from_tilt_x, 1)

    converted = convert_hbn_bundle_geometry_to_simulation(
        tilt_x_deg=float(tilt_x_deg),
        tilt_y_deg=float(tilt_y_deg),
        center_xy=imported_center,
        source_rotate_k=int(source_rotate_k),
        target_rotate_k=int(SIMULATION_GEOMETRY_ROTATE_K),
        image_size=(int(image_size), int(image_size)),
        simulation_gamma_sign_from_tilt_x=int(gamma_sign_from_tilt_x),
        simulation_Gamma_sign_from_tilt_y=int(gamma_sign_from_tilt_y),
    )

    def _ensure_slider_includes(slider_widget, value, pad=0.1):
        try:
            lo = float(slider_widget.cget("from"))
            hi = float(slider_widget.cget("to"))
        except Exception:
            return False
        if lo > hi:
            lo, hi = hi, lo
        val = float(value)
        new_lo = lo
        new_hi = hi
        if val < lo:
            new_lo = val - float(pad)
        if val > hi:
            new_hi = val + float(pad)
        if math.isclose(new_lo, lo, rel_tol=0.0, abs_tol=1e-12) and math.isclose(
            new_hi, hi, rel_tol=0.0, abs_tol=1e-12
        ):
            return False
        slider_widget.configure(from_=new_lo, to=new_hi)
        return True

    gamma_sim_deg = float(converted["gamma_deg"])
    Gamma_sim_deg = float(converted["Gamma_deg"])
    angle_range_expanded = False
    angle_range_expanded |= _ensure_slider_includes(gamma_scale, gamma_sim_deg, pad=0.1)
    angle_range_expanded |= _ensure_slider_includes(Gamma_scale, Gamma_sim_deg, pad=0.1)
    gamma_var.set(gamma_sim_deg)
    Gamma_var.set(Gamma_sim_deg)
    if distance_info and isinstance(distance_info, dict):
        mean_dist = distance_info.get("mean_m")
        if mean_dist is not None:
            try:
                _ensure_slider_includes(corto_detector_scale, float(mean_dist), pad=0.001)
                corto_detector_var.set(float(mean_dist))
            except Exception:
                pass
    center_text = ""
    center_row = converted.get("center_row")
    center_col = converted.get("center_col")
    if center_row is not None and center_col is not None:
        try:
            center_row = float(center_row)
            center_col = float(center_col)
            if np.isfinite(center_row) and np.isfinite(center_col):
                _ensure_slider_includes(center_x_scale, center_row, pad=5.0)
                _ensure_slider_includes(center_y_scale, center_col, pad=5.0)
                center_x_var.set(center_row)
                center_y_var.set(center_col)
                center_text = f" and center (row={center_row:.2f}, col={center_col:.2f}) px"
        except Exception:
            pass
    applied_center_row = None
    applied_center_col = None
    try:
        applied_center_row = float(center_x_var.get())
        applied_center_col = float(center_y_var.get())
    except Exception:
        applied_center_row = None
        applied_center_col = None

    debug_text = ""
    if HBN_GEOMETRY_DEBUG_ENABLED:
        debug_trace = build_hbn_geometry_debug_trace(
            npz_center_xy=imported_center,
            source_rotate_k=int(source_rotate_k),
            target_rotate_k=int(SIMULATION_GEOMETRY_ROTATE_K),
            image_size=(int(image_size), int(image_size)),
            tilt_x_deg=float(tilt_x_deg),
            tilt_y_deg=float(tilt_y_deg),
            simulation_gamma_sign_from_tilt_x=int(gamma_sign_from_tilt_x),
            simulation_Gamma_sign_from_tilt_y=int(gamma_sign_from_tilt_y),
            simulation_center_row=applied_center_row,
            simulation_center_col=applied_center_col,
        )
        hbn_geometry_debug_view_state.report_text = format_hbn_geometry_debug_trace(debug_trace)
        gui_views.set_hbn_geometry_debug_text(
            hbn_geometry_debug_view_state,
            str(hbn_geometry_debug_view_state.report_text),
        )
        print(hbn_geometry_debug_view_state.report_text)
        debug_text = " [debug report updated]"

    schedule_update()
    progress_label.config(
        text=(
            "Applied hBN bundle tilt hint "
            f"(γ={gamma_sim_deg:.3f}°, Γ={Gamma_sim_deg:.3f}°)"
            + (" [range expanded]" if angle_range_expanded else "")
            + (
                f" and distance {mean_dist:.4f} m"  # type: ignore[arg-type]
                if distance_info and mean_dist is not None
                else ""
            )
            + center_text
            + debug_text
            + f" from {bundle_path}"
        )
    )


def _gui_state_variable_items() -> dict[str, object]:
    """Return Tk variables that participate in full GUI-state snapshots."""

    items = dict(globals())
    items.update(
        {
            "background_min_var": display_controls_view_state.background_min_var,
            "background_max_var": display_controls_view_state.background_max_var,
            "background_transparency_var": (
                display_controls_view_state.background_transparency_var
            ),
            "simulation_min_var": display_controls_view_state.simulation_min_var,
            "simulation_max_var": display_controls_view_state.simulation_max_var,
            "simulation_scale_factor_var": (
                display_controls_view_state.simulation_scale_factor_var
            ),
            "qr_cylinder_display_mode_var": (
                geometry_overlay_actions_view_state.qr_cylinder_display_mode_var
            ),
        }
    )
    return items


def _restore_gui_state_peak_record(
    raw_record: object,
) -> dict[str, object] | None:
    """Normalize one imported GUI-state peak cache record for runtime use."""

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
    peak_records: Sequence[object] | None,
) -> None:
    """Replace the live peak-overlay cache from imported GUI-state records."""

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
    if _retain_runtime_optional_cache("peak_overlay", feature_needed=bool(restored_records)):
        simulation_runtime_state.peak_overlay_cache = {
            "sig": None,
            "positions": list(restored_positions),
            "millers": list(restored_millers),
            "intensities": list(restored_intensities),
            "records": [dict(record) for record in restored_records],
            "click_spatial_index": None,
            "restored_from_gui_state": bool(restored_records),
        }
    else:
        simulation_runtime_state.peak_overlay_cache = _empty_peak_overlay_cache()
    _invalidate_geometry_manual_pick_cache()


def _collect_full_gui_state_snapshot() -> dict[str, object]:
    return gui_state_io.collect_full_gui_state_snapshot(
        global_items=_gui_state_variable_items(),
        tk_variable_type=tk.Variable,
        occ_vars=_occupancy_control_vars(),
        atom_site_fract_vars=_atom_site_fractional_control_vars(),
        geometry_q_group_rows=_geometry_q_group_export_rows(),
        geometry_manual_pairs=_geometry_manual_pairs_export_rows(),
        geometry_peak_records=simulation_runtime_state.peak_records,
        selected_hkl_target=peak_selection_state.selected_hkl_target,
        primary_cif_path=_current_primary_cif_path(),
        secondary_cif_path=structure_model_state.cif_file2,
        osc_files=background_runtime_state.osc_files,
        current_background_index=background_runtime_state.current_background_index,
        background_visible=background_runtime_state.visible,
        background_backend_rotation_k=background_runtime_state.backend_rotation_k,
        background_backend_flip_x=background_runtime_state.backend_flip_x,
        background_backend_flip_y=background_runtime_state.backend_flip_y,
        background_limits_user_override=(display_controls_state.background_limits_user_override),
        simulation_limits_user_override=(display_controls_state.simulation_limits_user_override),
        scale_factor_user_override=display_controls_state.scale_factor_user_override,
    )


def _load_background_files_for_import_state(
    file_paths: list[str],
    select_index: int = 0,
) -> None:
    updated_state = gui_state_io.load_background_files_for_state(
        file_paths,
        osc_files=background_runtime_state.osc_files,
        background_images=background_runtime_state.background_images,
        background_images_native=background_runtime_state.background_images_native,
        background_images_display=background_runtime_state.background_images_display,
        select_index=select_index,
        display_rotate_k=DISPLAY_ROTATE_K,
        read_osc=read_osc,
        expected_shape=(int(image_size), int(image_size)),
    )
    if updated_state is None:
        return
    gui_background_manager.apply_background_payload_with_side_effects(
        background_runtime_state,
        updated_state,
        sync_background_runtime_state=_sync_background_runtime_state,
        replace_geometry_manual_pairs_by_background=(_replace_geometry_manual_pairs_by_background),
        invalidate_geometry_manual_pick_cache=_invalidate_geometry_manual_pick_cache,
        clear_geometry_manual_undo_stack=_clear_geometry_manual_undo_stack,
        clear_geometry_fit_undo_stack=_clear_geometry_fit_undo_stack,
        set_geometry_manual_pick_mode=_set_geometry_manual_pick_mode,
        set_background_display_data=background_display.set_data,
        update_background_slider_defaults=_update_background_slider_defaults,
        sync_background_theta_controls=(background_theta_workflow.sync_background_theta_controls),
        sync_geometry_fit_background_selection=(
            background_theta_workflow.sync_geometry_fit_background_selection
        ),
        clear_geometry_pick_artists=_clear_geometry_pick_artists,
        refresh_background_file_status=_refresh_background_status,
        schedule_update=None,
    )


def _apply_full_gui_state_snapshot(snapshot: dict[str, object]) -> str:

    if not isinstance(snapshot, dict):
        snapshot = {}

    warnings: list[str] = []

    warnings.extend(
        gui_state_io.apply_gui_state_files(
            snapshot.get("files", {}),
            apply_primary_cif_path=_apply_primary_cif_path,
            load_background_files=_load_background_files_for_import_state,
        )
    )

    variables = snapshot.get("variables", {})
    warnings.extend(
        gui_state_io.apply_gui_state_variables(
            variables,
            global_items=_gui_state_variable_items(),
            tk_variable_type=tk.Variable,
        )
    )

    gui_state_io.apply_gui_state_background_theta_compatibility(
        variables,
        osc_files=background_runtime_state.osc_files,
        theta_initial_var=theta_initial_var,
        background_theta_list_var=background_theta_list_var,
        geometry_theta_offset_var=geometry_theta_offset_var,
        geometry_fit_background_selection_var=geometry_fit_background_selection_var,
        format_background_theta_values=_format_background_theta_values,
        default_geometry_fit_background_selection=_default_geometry_fit_background_selection,
    )

    gui_state_io.apply_dynamic_gui_state_lists(
        snapshot.get("dynamic_lists", {}),
        occ_vars=_occupancy_control_vars(),
        atom_site_fract_vars=_atom_site_fractional_control_vars(),
    )

    flag_state = gui_state_io.apply_gui_state_flags(
        snapshot.get("flags", {}),
        current_flags={
            "background_visible": background_runtime_state.visible,
            "background_backend_rotation_k": background_runtime_state.backend_rotation_k,
            "background_backend_flip_x": background_runtime_state.backend_flip_x,
            "background_backend_flip_y": background_runtime_state.backend_flip_y,
            "background_limits_user_override": (
                display_controls_state.background_limits_user_override
            ),
            "simulation_limits_user_override": (
                display_controls_state.simulation_limits_user_override
            ),
            "scale_factor_user_override": (display_controls_state.scale_factor_user_override),
        },
        toggle_background=toggle_background,
    )
    background_runtime_state.visible = bool(flag_state["background_visible"])
    background_runtime_state.backend_rotation_k = int(flag_state["background_backend_rotation_k"])
    background_runtime_state.backend_flip_x = bool(flag_state["background_backend_flip_x"])
    background_runtime_state.backend_flip_y = bool(flag_state["background_backend_flip_y"])
    _sync_background_runtime_state()
    display_controls_state.background_limits_user_override = bool(
        flag_state["background_limits_user_override"]
    )
    display_controls_state.simulation_limits_user_override = bool(
        flag_state["simulation_limits_user_override"]
    )
    display_controls_state.scale_factor_user_override = bool(
        flag_state["scale_factor_user_override"]
    )

    geometry_state = gui_state_io.apply_gui_state_geometry(
        snapshot.get("geometry", {}),
        geometry_preview_excluded_q_groups=geometry_preview_state.excluded_q_groups,
        geometry_q_group_key_from_jsonable=_geometry_q_group_key_from_jsonable,
        invalidate_geometry_manual_pick_cache=_invalidate_geometry_manual_pick_cache,
        apply_geometry_manual_pairs_snapshot=_apply_geometry_manual_pairs_snapshot,
        replace_runtime_peak_cache=_replace_gui_state_peak_cache,
        current_background_index=background_runtime_state.current_background_index,
        selected_hkl_target=peak_selection_state.selected_hkl_target,
    )
    gui_runtime_geometry_interaction.apply_restored_runtime_selected_hkl_target(
        maintenance_callbacks=peak_selection_runtime_maintenance,
        selected_hkl_target=geometry_state["selected_hkl_target"],
    )
    warnings.extend(list(geometry_state["warnings"]))
    gui_state_io.apply_geometry_state_background_view_compatibility(
        snapshot.get("geometry", {}),
        geometry_q_group_key_from_jsonable=_geometry_q_group_key_from_jsonable,
        show_caked_2d_var=analysis_view_controls_view_state.show_caked_2d_var,
        show_1d_var=analysis_view_controls_view_state.show_1d_var,
        background_visible=bool(background_runtime_state.visible),
        toggle_background=toggle_background,
    )

    if finite_stack_controls_view_state.phase_delta_entry_var is not None:
        gui_views.set_finite_stack_phase_delta_entry_text(
            finite_stack_controls_view_state,
            _current_phase_delta_expression(),
        )
    if finite_stack_controls_view_state.phi_l_divisor_entry_var is not None:
        gui_views.set_finite_stack_phi_l_divisor_entry_text(
            finite_stack_controls_view_state,
            gui_controllers.format_finite_stack_phi_l_divisor(_current_phi_l_divisor()),
        )
    _sync_finite_controls()
    _apply_rod_points_per_gz(trigger_update=False)
    _apply_geometry_fit_background_selection(trigger_update=False)
    _refresh_background_status()
    _update_geometry_preview_exclude_button_label()
    geometry_q_group_runtime_callbacks.refresh_window()
    peak_selection_runtime_callbacks.reselect_current_peak()
    ensure_valid_resolution_choice()
    toggle_1d_plots()
    toggle_caked_2d()
    toggle_log_display()
    _sync_primary_raster_geometry(
        view_mode=_resolved_primary_analysis_display_mode()
    )
    _refresh_background_backend_status()
    _mark_chi_square_dirty()
    _update_chi_square_display(force=True)
    qr_cylinder_overlay_runtime_refresh(redraw=True, update_status=False)
    _refresh_live_geometry_preview(update_status=False)
    schedule_update()

    return gui_state_io.build_gui_state_import_summary(warnings)


def _export_full_gui_state() -> None:
    try:
        initial_dir = str(get_dir("file_dialog_dir"))
    except Exception:
        initial_dir = str(Path.cwd())
    file_path = filedialog.asksaveasfilename(
        title="Export Full GUI State",
        initialdir=initial_dir,
        defaultextension=".json",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        initialfile=f"ra_sim_gui_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    if not file_path:
        progress_label.config(text="GUI state export canceled.")
        return
    try:
        save_gui_state_file(
            file_path,
            _collect_full_gui_state_snapshot(),
            metadata={"entrypoint": gui_manual_geometry.DEFAULT_GUI_ENTRYPOINT},
        )
    except Exception as exc:
        progress_label.config(text=f"Failed to export GUI state: {exc}")
        return
    progress_label.config(text=f"Saved GUI state to {file_path}")


def _import_full_gui_state() -> None:
    try:
        initial_dir = str(get_dir("file_dialog_dir"))
    except Exception:
        initial_dir = str(Path.cwd())
    file_path = filedialog.askopenfilename(
        title="Import Full GUI State",
        initialdir=initial_dir,
        filetypes=[
            ("RA-SIM GUI state", "*.json"),
            ("Legacy parameter profiles", "*.npy"),
            ("All files", "*.*"),
        ],
    )
    if not file_path:
        progress_label.config(text="GUI state import canceled.")
        return

    suffix = Path(file_path).suffix.lower()
    if suffix == ".npy":
        message = load_parameters(
            file_path,
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
            center_x_var,
            center_y_var,
            resolution_var,
            custom_samples_var,
            rod_points_per_gz_var,
            bandwidth_percent_var=bandwidth_percent_var,
            optics_mode_var=optics_mode_var,
            phase_delta_expr_var=phase_delta_expr_var,
            phi_l_divisor_var=phi_l_divisor_var,
            sf_prune_bias_var=sf_prune_bias_var,
            solve_q_steps_var=solve_q_steps_var,
            solve_q_rel_tol_var=solve_q_rel_tol_var,
            solve_q_mode_var=solve_q_mode_var,
        )
        if finite_stack_controls_view_state.phase_delta_entry_var is not None:
            gui_views.set_finite_stack_phase_delta_entry_text(
                finite_stack_controls_view_state,
                _current_phase_delta_expression(),
            )
        if finite_stack_controls_view_state.phi_l_divisor_entry_var is not None:
            gui_views.set_finite_stack_phi_l_divisor_entry_text(
                finite_stack_controls_view_state,
                gui_controllers.format_finite_stack_phi_l_divisor(_current_phi_l_divisor()),
            )
        _sync_finite_controls()
        _apply_rod_points_per_gz(trigger_update=False)
        ensure_valid_resolution_choice()
        schedule_update()
        progress_label.config(text=message)
        return

    try:
        payload = load_gui_state_file(file_path)
        message = _apply_full_gui_state_snapshot(payload.get("state", {}))
    except Exception as exc:
        progress_label.config(text=f"Failed to import GUI state: {exc}")
        return
    progress_label.config(text=message)


def _export_geometry_manual_pairs() -> None:
    """Write the saved manual geometry placements to a JSON file."""

    try:
        initial_dir = str(get_dir("file_dialog_dir"))
    except Exception:
        initial_dir = str(Path.cwd())
    gui_manual_geometry.export_geometry_manual_pairs(
        osc_files=background_runtime_state.osc_files,
        pairs_for_index=_geometry_manual_pairs_for_index,
        collect_snapshot=_collect_geometry_manual_pairs_snapshot,
        initial_dir=initial_dir,
        asksaveasfilename=filedialog.asksaveasfilename,
        save_file=save_geometry_placements_file,
        set_status_text=lambda text: progress_label_geometry.config(text=text),
        stamp_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"),
        entrypoint=gui_manual_geometry.DEFAULT_GUI_ENTRYPOINT,
    )


def _import_geometry_manual_pairs() -> None:
    """Load manual geometry placements from a JSON file."""

    try:
        initial_dir = str(get_dir("file_dialog_dir"))
    except Exception:
        initial_dir = str(Path.cwd())
    gui_manual_geometry.import_geometry_manual_pairs(
        initial_dir=initial_dir,
        askopenfilename=filedialog.askopenfilename,
        load_file=load_geometry_placements_file,
        apply_snapshot=_apply_geometry_manual_pairs_snapshot,
        ensure_geometry_fit_caked_view=(
            lambda: _ensure_geometry_fit_caked_view(force_refresh=True)
        ),
        set_status_text=lambda text: progress_label_geometry.config(text=text),
    )


session_button_specs = [
    ("Export GUI State...", _export_full_gui_state),
    ("Import GUI State...", _import_full_gui_state),
    ("Import hBN Bundle Tilt", import_hbn_tilt_from_bundle),
]
if HBN_GEOMETRY_DEBUG_ENABLED:
    session_button_specs.append(("Show hBN Geometry Debug", show_last_hbn_geometry_debug))
gui_views.populate_stacked_button_group(
    workspace_panels_view_state.workspace_session_frame,
    session_button_specs,
)

# Frame for selecting which geometry params to fit
gui_views.create_geometry_fit_parameter_controls(
    parent=app_shell_view_state.match_parameter_frame,
    view_state=geometry_fit_parameter_controls_view_state,
    initial_values={
        "zb": True,
        "zs": True,
        "theta_initial": True,
        "psi_z": True,
        "chi": True,
        "cor_angle": True,
        "gamma": True,
        "Gamma": True,
        "corto_detector": True,
        "a": False,
        "c": False,
        "center_x": False,
        "center_y": False,
    },
    use_labelframe=False,
)
fit_frame = geometry_fit_parameter_controls_view_state.frame
fit_zb_var = geometry_fit_parameter_controls_view_state.fit_zb_var
fit_zs_var = geometry_fit_parameter_controls_view_state.fit_zs_var
fit_theta_var = geometry_fit_parameter_controls_view_state.fit_theta_var
fit_psi_z_var = geometry_fit_parameter_controls_view_state.fit_psi_z_var
fit_chi_var = geometry_fit_parameter_controls_view_state.fit_chi_var
fit_cor_var = geometry_fit_parameter_controls_view_state.fit_cor_var
fit_gamma_var = geometry_fit_parameter_controls_view_state.fit_gamma_var
fit_Gamma_var = geometry_fit_parameter_controls_view_state.fit_Gamma_var
fit_dist_var = geometry_fit_parameter_controls_view_state.fit_dist_var
fit_a_var = geometry_fit_parameter_controls_view_state.fit_a_var
fit_c_var = geometry_fit_parameter_controls_view_state.fit_c_var
fit_center_x_var = geometry_fit_parameter_controls_view_state.fit_center_x_var
fit_center_y_var = geometry_fit_parameter_controls_view_state.fit_center_y_var
fit_theta_checkbutton = geometry_fit_parameter_controls_view_state.fit_theta_checkbutton
if fit_frame is None:
    raise RuntimeError("Geometry-fit parameter controls did not create the frame.")
_refresh_geometry_fit_theta_checkbox_label()

GEOMETRY_FIT_PARAM_ORDER = [
    *gui_geometry_fit.GEOMETRY_FIT_PARAM_ORDER,
]
geometry_fit_toggle_vars = dict(geometry_fit_parameter_controls_view_state.toggle_vars)
geometry_fit_parameter_specs = {}


def _sync_geometry_fit_constraint_rows(*_args) -> None:
    controls = geometry_fit_constraints_view_state.controls
    for name in GEOMETRY_FIT_PARAM_ORDER:
        control = controls.get(name)
        if not isinstance(control, dict):
            continue
        row = control.get("row")
        if row is None:
            continue
        toggle_var = geometry_fit_toggle_vars.get(name)
        enabled = bool(toggle_var.get()) if toggle_var is not None else True
        mapped = bool(control.get("_mapped", False))
        if enabled and not mapped:
            row.pack(fill=tk.X, padx=4, pady=4)
            control["_mapped"] = True
        elif not enabled and mapped:
            row.pack_forget()
            control["_mapped"] = False


if BACKGROUND_BACKEND_DEBUG_UI_ENABLED:
    background_controls_runtime.create_backend_debug_controls(
        parent=workspace_panels_view_state.workspace_debug_frame.frame,
    )

if DEBUG_ENABLED and BACKEND_ORIENTATION_UI_ENABLED:
    gui_views.create_backend_orientation_debug_controls(
        parent=workspace_panels_view_state.workspace_debug_frame.frame,
        view_state=background_backend_debug_view_state,
    )


def _auto_match_console(cfg: dict[str, object] | None, text: str) -> None:
    """Emit one console progress line for geometry auto-match when enabled."""

    if not isinstance(cfg, dict) or not bool(cfg.get("console_progress", False)):
        return
    try:
        print(f"[geometry-fit] auto-match: {text}", flush=True)
    except Exception:
        pass


def _auto_match_refine_peak_center(
    peakness: np.ndarray,
    fine_image: np.ndarray,
    row_idx: int,
    col_idx: int,
) -> tuple[float, float]:
    """Refine one summit center to a local weighted centroid."""

    if peakness.ndim != 2:
        return float(col_idx), float(row_idx)
    height, width = peakness.shape
    r0 = max(0, int(row_idx) - 1)
    r1 = min(height, int(row_idx) + 2)
    c0 = max(0, int(col_idx) - 1)
    c1 = min(width, int(col_idx) + 2)
    peak_patch = peakness[r0:r1, c0:c1]
    weight_patch = np.clip(peak_patch, 0.0, None)
    if not np.any(weight_patch > 0.0) and fine_image.ndim == 2:
        fine_patch = fine_image[r0:r1, c0:c1]
        weight_patch = np.clip(fine_patch - np.min(fine_patch), 0.0, None)
    total_weight = float(np.sum(weight_patch))
    if total_weight <= 0.0 or not np.isfinite(total_weight):
        return float(col_idx), float(row_idx)
    rr, cc = np.mgrid[r0:r1, c0:c1]
    center_col = float(np.sum(weight_patch * cc) / total_weight)
    center_row = float(np.sum(weight_patch * rr) / total_weight)
    return center_col, center_row


def _auto_match_background_context(
    background_image: object,
    cfg: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Return cached background peakness data for seed-local auto-matching."""

    config = cfg if isinstance(cfg, dict) else {}
    local_max_size = int(config.get("local_max_size_px", 5))
    local_max_size = max(3, local_max_size)
    if local_max_size % 2 == 0:
        local_max_size += 1
    smooth_sigma = max(0.0, float(config.get("smooth_sigma_px", 3.0)))
    climb_sigma = max(
        0.0,
        float(
            config.get(
                "climb_sigma_px",
                min(1.0, 0.5 * smooth_sigma) if smooth_sigma > 0.0 else 0.8,
            )
        ),
    )
    min_prominence_sigma = float(config.get("min_prominence_sigma", 2.0))
    fallback_percentile = float(config.get("fallback_percentile", 99.5))
    fallback_percentile = min(100.0, max(50.0, fallback_percentile))

    raw_arr = np.asarray(background_image)
    bg_ptr = 0
    try:
        bg_ptr = int(raw_arr.__array_interface__["data"][0])
    except Exception:
        bg_ptr = int(id(raw_arr))
    bg_key = (
        bg_ptr,
        tuple(int(v) for v in raw_arr.shape),
        tuple(int(v) for v in raw_arr.strides),
        str(raw_arr.dtype),
    )
    cache_key = (
        bg_key,
        int(local_max_size),
        round(float(smooth_sigma), 6),
        round(float(climb_sigma), 6),
        round(float(min_prominence_sigma), 6),
        round(float(fallback_percentile), 6),
    )
    cached_context = gui_controllers.get_geometry_auto_match_background_cache(
        geometry_preview_state,
        cache_key,
    )
    if cached_context is not None:
        _auto_match_console(config, f"context cache hit shape={bg_key[1]}")
        return dict(config), cached_context

    cache_data = build_background_peak_context(
        background_image,
        config,
        logger=lambda text: _auto_match_console(config, text),
    )
    _auto_match_console(
        config,
        "context build done "
        f"summits={len(cache_data.get('summit_records', []))} "
        f"sigma_est={float(cache_data.get('sigma_est', np.nan)):.4f}",
    )
    gui_controllers.replace_geometry_auto_match_background_cache(
        geometry_preview_state,
        cache_key,
        cache_data,
    )
    return dict(config), cache_data


def _auto_match_background_peaks(
    simulated_peaks: list[dict[str, object]],
    background_image: np.ndarray,
    cfg: dict[str, object] | None = None,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Match simulated seeds to nearby background peaks using local assignment."""

    config = dict(cfg) if isinstance(cfg, dict) else {}
    _auto_match_console(config, f"seed-local match start seeds={len(simulated_peaks)}")
    context_offset_col = 0.0
    context_offset_row = 0.0
    context_simulated: list[dict[str, object]] = []
    raw_background = np.asarray(background_image)
    context_margin_px = float(config.get("context_margin_px", np.inf))
    if not np.isfinite(context_margin_px) or context_margin_px <= 0.0:
        context_margin_px = np.inf
    if raw_background.ndim == 2 and simulated_peaks and np.isfinite(context_margin_px):
        smooth_sigma = max(0.0, float(config.get("smooth_sigma_px", 3.0)))
        climb_sigma = max(
            0.0,
            float(
                config.get(
                    "climb_sigma_px",
                    min(1.0, 0.5 * smooth_sigma) if smooth_sigma > 0.0 else 0.8,
                )
            ),
        )
        broad_sigma = smooth_sigma if smooth_sigma > 0.0 else max(2.0, climb_sigma + 1.5)
        roi_pad = max(8.0, 3.0 * broad_sigma + 4.0)
        seed_cols = np.asarray(
            [float(entry.get("sim_col", np.nan)) for entry in simulated_peaks],
            dtype=float,
        )
        seed_rows = np.asarray(
            [float(entry.get("sim_row", np.nan)) for entry in simulated_peaks],
            dtype=float,
        )
        finite_seed = np.isfinite(seed_cols) & np.isfinite(seed_rows)
        if np.any(finite_seed):
            seed_cols = seed_cols[finite_seed]
            seed_rows = seed_rows[finite_seed]
            col0 = max(
                0,
                int(np.floor(float(np.min(seed_cols) - context_margin_px - roi_pad))),
            )
            col1 = min(
                int(raw_background.shape[1]),
                int(np.ceil(float(np.max(seed_cols) + context_margin_px + roi_pad))) + 1,
            )
            row0 = max(
                0,
                int(np.floor(float(np.min(seed_rows) - context_margin_px - roi_pad))),
            )
            row1 = min(
                int(raw_background.shape[0]),
                int(np.ceil(float(np.max(seed_rows) + context_margin_px + roi_pad))) + 1,
            )
            if row1 > row0 and col1 > col0:
                context_offset_col = float(col0)
                context_offset_row = float(row0)
                raw_background = raw_background[row0:row1, col0:col1]
    if np.asarray(raw_background).ndim == 2:
        _auto_match_console(
            config,
            "roi "
            f"shape={tuple(int(v) for v in np.asarray(raw_background).shape)} "
            f"offset=({int(round(context_offset_col))},{int(round(context_offset_row))})",
        )
    for entry in simulated_peaks:
        local_entry = dict(entry)
        sim_col_global = float(entry.get("sim_col", np.nan))
        sim_row_global = float(entry.get("sim_row", np.nan))
        local_entry["sim_col_global"] = sim_col_global
        local_entry["sim_row_global"] = sim_row_global
        local_entry["sim_col_local"] = sim_col_global - context_offset_col
        local_entry["sim_row_local"] = sim_row_global - context_offset_row
        context_simulated.append(local_entry)

    config, bg_ctx = _auto_match_background_context(raw_background, config)
    if not bool(bg_ctx.get("img_valid", False)):
        return [], {"simulated_count": float(len(simulated_peaks))}
    matches, stats = match_simulated_peaks_to_peak_context(
        context_simulated,
        bg_ctx,
        config,
        context_offset_col=context_offset_col,
        context_offset_row=context_offset_row,
        logger=lambda text: _auto_match_console(config, text),
    )
    _auto_match_console(
        config,
        f"done matches={len(matches)} pre_clip={int(stats.get('matched_pre_clip_count', 0.0))} "
        f"conflicts={int(stats.get('conflicted_match_count', 0.0))}",
    )
    return matches, stats


def _auto_match_background_peaks_with_relaxation(
    simulated_peaks: list[dict[str, object]],
    background_image: np.ndarray,
    cfg: dict[str, object] | None = None,
    *,
    min_matches: int = 1,
) -> tuple[
    list[dict[str, object]],
    dict[str, float],
    dict[str, object],
    list[dict[str, float | str]],
]:
    """Retry auto-match with progressively looser thresholds when needed."""

    base_cfg = dict(cfg) if isinstance(cfg, dict) else {}
    target_min_matches = max(1, min(int(min_matches), int(len(simulated_peaks))))
    attempts: list[dict[str, float | str]] = []

    def _run_attempt(
        attempt_cfg: dict[str, object], label: str
    ) -> tuple[list[dict[str, object]], dict[str, float]]:
        _auto_match_console(
            attempt_cfg,
            "attempt "
            f"{label} start min_prom="
            f"{float(attempt_cfg.get('min_match_prominence_sigma', attempt_cfg.get('min_prominence_sigma', np.nan))):.2f} "
            f"max_candidates={int(attempt_cfg.get('max_candidate_peaks', 0))}",
        )
        matches_i, stats_i = _auto_match_background_peaks(
            simulated_peaks,
            background_image,
            attempt_cfg,
        )
        attempts.append(
            {
                "label": label,
                "matches": float(len(matches_i)),
                "search_radius_px": float(attempt_cfg.get("search_radius_px", np.nan)),
                "min_match_prominence_sigma": float(
                    attempt_cfg.get(
                        "min_match_prominence_sigma",
                        attempt_cfg.get("min_prominence_sigma", np.nan),
                    )
                ),
                "max_candidate_peaks": float(attempt_cfg.get("max_candidate_peaks", np.nan)),
                "ambiguity_margin_px": float(attempt_cfg.get("ambiguity_margin_px", np.nan)),
                "ambiguity_ratio_min": float(attempt_cfg.get("ambiguity_ratio_min", np.nan)),
                "qualified_summit_count": float(stats_i.get("qualified_summit_count", np.nan)),
                "claimed_summit_count": float(stats_i.get("claimed_summit_count", np.nan)),
                "matched_pre_clip_count": float(stats_i.get("matched_pre_clip_count", np.nan)),
                "clipped_count": float(stats_i.get("clipped_count", np.nan)),
            }
        )
        _auto_match_console(
            attempt_cfg,
            "attempt "
            f"{label} result matches={len(matches_i)} "
            f"qualified={int(stats_i.get('qualified_summit_count', 0.0))} "
            f"claimed={int(stats_i.get('claimed_summit_count', 0.0))} "
            f"pre_clip={int(stats_i.get('matched_pre_clip_count', 0.0))} "
            f"clipped={int(stats_i.get('clipped_count', 0.0))}",
        )
        return matches_i, stats_i

    best_matches, best_stats = _run_attempt(base_cfg, "base")
    best_cfg = dict(base_cfg)

    if len(best_matches) >= target_min_matches:
        _auto_match_console(
            base_cfg,
            f"target reached on base attempt ({len(best_matches)}/{target_min_matches})",
        )
        return best_matches, best_stats, best_cfg, attempts

    if not bool(base_cfg.get("relax_on_low_matches", True)):
        _auto_match_console(
            base_cfg,
            f"relaxation disabled at {len(best_matches)}/{target_min_matches} matches",
        )
        return best_matches, best_stats, best_cfg, attempts

    base_radius = max(1.0, float(base_cfg.get("search_radius_px", 24.0)))
    base_match_prom = float(
        base_cfg.get(
            "min_match_prominence_sigma",
            base_cfg.get("min_prominence_sigma", 2.0),
        )
    )
    base_max_candidates = max(50, int(base_cfg.get("max_candidate_peaks", 1200)))
    base_ambiguity_margin = max(0.0, float(base_cfg.get("ambiguity_margin_px", 2.0)))
    base_ambiguity_ratio = max(1.0, float(base_cfg.get("ambiguity_ratio_min", 1.15)))
    base_distance_sigma_clip = max(0.0, float(base_cfg.get("distance_sigma_clip", 3.5)))

    relax_steps = max(1, min(6, int(base_cfg.get("relax_steps", 3))))
    radius_growth = max(0.1, float(base_cfg.get("relax_radius_growth", 0.65)))
    prominence_step = max(0.0, float(base_cfg.get("relax_prominence_step", 0.4)))
    candidate_growth = max(0.0, float(base_cfg.get("relax_candidate_growth", 0.35)))
    relax_ownership_guards = bool(base_cfg.get("relax_ownership_guards", False))

    for step in range(1, relax_steps + 1):
        relax_cfg = dict(base_cfg)
        relax_cfg["search_radius_px"] = float(
            min(120.0, max(base_radius, base_radius * (1.0 + radius_growth * step)))
        )
        relax_cfg["min_match_prominence_sigma"] = float(
            max(0.0, base_match_prom - prominence_step * step)
        )
        relax_cfg["max_candidate_peaks"] = int(
            min(
                5000,
                max(
                    base_max_candidates,
                    round(base_max_candidates * (1.0 + candidate_growth * step)),
                ),
            )
        )
        if relax_ownership_guards:
            relax_cfg["ambiguity_margin_px"] = float(max(0.0, base_ambiguity_margin - 0.5 * step))
            relax_cfg["ambiguity_ratio_min"] = float(max(1.0, base_ambiguity_ratio - 0.05 * step))
        else:
            relax_cfg["ambiguity_margin_px"] = float(base_ambiguity_margin)
            relax_cfg["ambiguity_ratio_min"] = float(base_ambiguity_ratio)
        relax_cfg["distance_sigma_clip"] = float(max(base_distance_sigma_clip, 3.5 + 0.5 * step))

        matches_i, stats_i = _run_attempt(relax_cfg, f"relax_{step}")
        best_is_improved = len(matches_i) > len(best_matches)
        if len(matches_i) == len(best_matches):
            best_conf = float(best_stats.get("median_match_confidence", -np.inf))
            cand_conf = float(stats_i.get("median_match_confidence", -np.inf))
            best_is_improved = cand_conf > best_conf
        if best_is_improved:
            best_matches = matches_i
            best_stats = stats_i
            best_cfg = relax_cfg

        if len(best_matches) >= target_min_matches:
            _auto_match_console(
                relax_cfg,
                f"target reached on relax_{step} ({len(best_matches)}/{target_min_matches})",
            )
            break

    _auto_match_console(
        best_cfg,
        "best attempt "
        f"matches={len(best_matches)}/{target_min_matches} "
        f"qualified={int(best_stats.get('qualified_summit_count', 0.0))} "
        f"claimed={int(best_stats.get('claimed_summit_count', 0.0))}",
    )
    return best_matches, best_stats, best_cfg, attempts


def _current_geometry_fit_var_names() -> list[str]:
    """Return the currently selected geometry variables for LSQ fitting."""

    return _geometry_fit_runtime_values().current_var_names()


def _current_geometry_fit_candidate_param_names() -> list[str]:
    """Return all currently fit-capable geometry parameters for diagnostics."""

    candidate_names = list(
        gui_geometry_fit.current_geometry_fit_var_names(
            fit_zb=True,
            fit_zs=True,
            fit_theta=True,
            fit_psi_z=True,
            fit_chi=True,
            fit_cor=True,
            fit_gamma=True,
            fit_Gamma=True,
            fit_dist=True,
            fit_a=True,
            fit_c=True,
            fit_center_x=True,
            fit_center_y=True,
            use_shared_theta_offset=_geometry_fit_uses_shared_theta_offset(),
        )
    )
    geometry_cfg = fit_config.get("geometry", {}) if isinstance(fit_config, Mapping) else {}
    if not isinstance(geometry_cfg, Mapping):
        geometry_cfg = {}
    lattice_cfg = geometry_cfg.get("lattice_refinement", {}) or {}
    if not isinstance(lattice_cfg, Mapping):
        lattice_cfg = {}
    if not bool(lattice_cfg.get("enabled", False)):
        candidate_names = [name for name in candidate_names if name not in {"a", "c"}]

    available_domains = _current_geometry_fit_parameter_domains(candidate_names)
    return [name for name in candidate_names if name in available_domains]


def _current_geometry_fit_params() -> dict[str, object]:
    """Assemble the current geometry-fit parameter dictionary."""

    return _geometry_fit_runtime_values().current_params()


def _current_geometry_fit_constraint_state(
    names: Sequence[str] | None = None,
) -> dict[str, dict[str, float]]:
    return {}


def _current_geometry_fit_parameter_domains(
    names: Sequence[str] | None = None,
) -> dict[str, tuple[float, float]]:
    return gui_geometry_fit.read_runtime_geometry_fit_parameter_domains(
        parameter_specs=geometry_fit_parameter_specs,
        image_size=image_size,
        fit_config=fit_config,
        names=names,
        use_shared_theta_offset=_geometry_fit_uses_shared_theta_offset(),
    )


def _build_geometry_fit_runtime_config(
    base_config,
    current_params,
    control_settings,
    parameter_domains,
    *,
    candidate_param_names=None,
):
    return gui_geometry_fit.build_geometry_fit_runtime_config(
        base_config,
        current_params,
        control_settings,
        parameter_domains,
        candidate_param_names=candidate_param_names,
    )


def _refresh_live_geometry_preview(*, update_status: bool = True) -> bool:
    """Recompute and redraw the live auto-match overlay from the current state."""

    if not _geometry_overlays_enabled():
        del update_status
        _clear_all_geometry_overlay_artists(redraw=True)
        return False

    display_background = (
        gui_geometry_q_group_manager.resolve_runtime_live_geometry_preview_background(
            geometry_q_group_runtime_bindings_factory(),
            update_status=update_status,
        )
    )
    if display_background is None:
        return False

    preview_auto_match_cfg = (
        gui_geometry_q_group_manager.build_live_geometry_preview_auto_match_config(fit_config)
    )

    var_names = _current_geometry_fit_var_names()
    min_matches = gui_geometry_q_group_manager.current_geometry_auto_match_min_matches(
        fit_config,
        var_names,
    )

    simulated_peaks = (
        gui_geometry_q_group_manager.resolve_runtime_live_geometry_preview_simulated_peaks(
            geometry_q_group_runtime_bindings_factory(),
            update_status=update_status,
        )
    )
    if not simulated_peaks:
        return False

    preview_signature = _live_geometry_preview_signature()
    preview_seed_state = (
        gui_geometry_q_group_manager.resolve_runtime_live_geometry_preview_seed_state(
            geometry_q_group_runtime_bindings_factory(),
            simulated_peaks,
            preview_auto_match_cfg=preview_auto_match_cfg,
            min_matches=int(min_matches),
            signature=preview_signature,
            update_status=update_status,
        )
    )
    if not preview_seed_state:
        return False
    simulated_peaks, excluded_q_peaks, q_group_total, collapsed_deg_preview = preview_seed_state

    matched_pairs, match_stats, _effective_auto_match_cfg, auto_match_attempts = (
        _auto_match_background_peaks_with_relaxation(
            simulated_peaks,
            display_background,
            preview_auto_match_cfg,
            min_matches=min_matches,
        )
    )
    return gui_geometry_q_group_manager.apply_runtime_live_geometry_preview_match_results(
        geometry_q_group_runtime_bindings_factory(),
        signature=preview_signature,
        matched_pairs=matched_pairs,
        match_stats=match_stats,
        preview_auto_match_cfg=preview_auto_match_cfg,
        auto_match_attempts=auto_match_attempts,
        min_matches=int(min_matches),
        q_group_total=int(q_group_total),
        excluded_q_peaks=int(excluded_q_peaks),
        collapsed_deg_preview=int(collapsed_deg_preview),
        update_status=update_status,
    )


geometry_q_group_runtime_value_callbacks = (
    gui_geometry_q_group_manager.make_runtime_geometry_q_group_value_callbacks(
        simulation_runtime_state=simulation_runtime_state,
        preview_state=geometry_preview_state,
        q_group_state=geometry_q_group_state,
        fit_config=fit_config,
        current_geometry_fit_var_names_factory=lambda: _current_geometry_fit_var_names(),
        primary_a_factory=lambda: float(a_var.get()) if "a_var" in globals() else float(av),
        primary_c_factory=lambda: float(c_var.get()) if "c_var" in globals() else float(cv),
        image_size_factory=lambda: int(image_size),
        native_sim_to_display_coords=_native_sim_to_display_coords,
    )
)
_build_live_preview_simulated_peaks_from_cache = (
    geometry_q_group_runtime_value_callbacks.build_live_preview_simulated_peaks_from_cache
)
_last_live_preview_cache_metadata = (
    geometry_q_group_runtime_value_callbacks.last_live_preview_cache_metadata
)
_filter_geometry_fit_simulated_peaks = (
    geometry_q_group_runtime_value_callbacks.filter_simulated_peaks
)
_collapse_geometry_fit_simulated_peaks = (
    geometry_q_group_runtime_value_callbacks.collapse_simulated_peaks
)
_build_geometry_q_group_entries = geometry_q_group_runtime_value_callbacks.build_entries_snapshot
_clone_geometry_q_group_entries = geometry_q_group_runtime_value_callbacks.clone_entries
_listed_geometry_q_group_entries = geometry_q_group_runtime_value_callbacks.listed_entries
_listed_geometry_q_group_keys = geometry_q_group_runtime_value_callbacks.listed_keys
_geometry_q_group_key_from_jsonable = geometry_q_group_runtime_value_callbacks.key_from_jsonable
_geometry_q_group_export_rows = geometry_q_group_runtime_value_callbacks.export_rows
_format_geometry_q_group_line = geometry_q_group_runtime_value_callbacks.format_line
_current_geometry_auto_match_min_matches = (
    geometry_q_group_runtime_value_callbacks.current_min_matches
)
_geometry_q_group_excluded_count = geometry_q_group_runtime_value_callbacks.excluded_count
_build_geometry_q_group_window_status_text = (
    geometry_q_group_runtime_value_callbacks.build_window_status
)
_live_preview_match_key = geometry_q_group_runtime_value_callbacks.live_preview_match_key
_live_preview_match_hkl = geometry_q_group_runtime_value_callbacks.live_preview_match_hkl
_live_preview_match_is_excluded = (
    geometry_q_group_runtime_value_callbacks.live_preview_match_is_excluded
)
_filter_live_preview_matches = geometry_q_group_runtime_value_callbacks.filter_live_preview_matches
_apply_live_preview_match_exclusions = (
    geometry_q_group_runtime_value_callbacks.apply_live_preview_match_exclusions
)


_geometry_manual_source_snapshot_diagnostics_state: dict[str, object] = {}


def _set_geometry_manual_source_snapshot_diagnostics(**kwargs) -> None:
    _geometry_manual_source_snapshot_diagnostics_state.clear()
    _geometry_manual_source_snapshot_diagnostics_state.update(kwargs)


def _geometry_manual_last_source_snapshot_diagnostics() -> dict[str, object]:
    return dict(_geometry_manual_source_snapshot_diagnostics_state)


def _geometry_source_signature_numeric(
    value: object,
    *,
    default: float = 0.0,
) -> float:
    try:
        numeric = float(value)
    except Exception:
        numeric = float(default)
    if not np.isfinite(numeric):
        numeric = float(default)
    return float(numeric)


def _geometry_source_snapshot_signature_from_params(
    param_set,
    *,
    mosaic_params,
    optics_mode_component: int,
    include_mosaic_shape: bool = True,
    sf_prune_bias: float,
    sf_prune_stats,
    ordered_structure_scale: float,
    qr_cylinder_replace_requested: bool,
    primary_source_signature,
    secondary_source_signature,
    secondary_a: float,
    secondary_c: float,
):
    params = dict(param_set or {})

    def _param_value(*keys: str, default: object = 0.0) -> object:
        for key in keys:
            if key in params:
                return params.get(key)
        return default

    signature = (
        round(_geometry_source_signature_numeric(_param_value("gamma")), 6),
        round(_geometry_source_signature_numeric(_param_value("Gamma")), 6),
        round(_geometry_source_signature_numeric(_param_value("chi")), 6),
        round(_geometry_source_signature_numeric(_param_value("psi_z", "psi")), 6),
        round(_geometry_source_signature_numeric(_param_value("zs")), 9),
        round(_geometry_source_signature_numeric(_param_value("zb")), 9),
        round(_geometry_source_signature_numeric(_param_value("sample_width")), 9),
        round(_geometry_source_signature_numeric(_param_value("sample_length")), 9),
        round(_geometry_source_signature_numeric(_param_value("sample_depth")), 12),
        round(_geometry_source_signature_numeric(_param_value("debye_x")), 6),
        round(_geometry_source_signature_numeric(_param_value("debye_y")), 6),
        round(_geometry_source_signature_numeric(_param_value("a")), 6),
        round(_geometry_source_signature_numeric(_param_value("c")), 6),
        round(
            _geometry_source_signature_numeric(_param_value("theta_initial", "theta_init")),
            6,
        ),
        round(_geometry_source_signature_numeric(_param_value("cor", "cor_angle")), 6),
        round(_geometry_source_signature_numeric(_param_value("center_x")), 3),
        round(_geometry_source_signature_numeric(_param_value("center_y")), 3),
        int(mosaic_params["solve_q_steps"]),
        round(float(mosaic_params["solve_q_rel_tol"]), 8),
        int(mosaic_params["solve_q_mode"]),
        tuple(mosaic_params.get("_sampling_signature", ())),
        round(_geometry_source_signature_numeric(sf_prune_bias), 3),
        int((sf_prune_stats or {}).get("qr_kept", 0)),
        int((sf_prune_stats or {}).get("hkl_primary_kept", 0)),
        round(_geometry_source_signature_numeric(ordered_structure_scale), 6),
        int(optics_mode_component),
        int(bool(qr_cylinder_replace_requested)),
        int(np.size(mosaic_params["beam_x_array"])),
        int(np.size(mosaic_params["theta_array"])),
        primary_source_signature,
        secondary_source_signature,
        round(_geometry_source_signature_numeric(secondary_a), 6),
        round(_geometry_source_signature_numeric(secondary_c), 6),
    )
    if include_mosaic_shape:
        signature += (
            round(
                _geometry_source_signature_numeric(
                    _param_value(
                        "sigma_mosaic_deg",
                        default=mosaic_params.get("sigma_mosaic_deg", 0.0),
                    )
                ),
                6,
            ),
            round(
                _geometry_source_signature_numeric(
                    _param_value(
                        "gamma_mosaic_deg",
                        default=mosaic_params.get("gamma_mosaic_deg", 0.0),
                    )
                ),
                6,
            ),
            round(
                _geometry_source_signature_numeric(
                    _param_value("eta", default=mosaic_params.get("eta", 0.0))
                ),
                6,
            ),
        )
    return signature


def _geometry_source_snapshot_signature_for_background(
    background_index: int,
    param_set: dict[str, object] | None = None,
):
    background_idx = int(background_index)
    params = dict(_current_geometry_fit_params())
    if isinstance(param_set, dict):
        params.update(dict(param_set))
    mosaic_params = build_mosaic_params()
    primary_source_signature = (
        (
            "qr",
            id(simulation_runtime_state.sim_primary_qr),
            len(simulation_runtime_state.sim_primary_qr),
        )
        if isinstance(simulation_runtime_state.sim_primary_qr, dict)
        and len(simulation_runtime_state.sim_primary_qr) > 0
        else (
            "miller",
            id(simulation_runtime_state.sim_miller1),
            tuple(np.asarray(simulation_runtime_state.sim_miller1).shape),
        )
    )
    secondary_source_signature = (
        "miller",
        id(simulation_runtime_state.sim_miller2),
        tuple(np.asarray(simulation_runtime_state.sim_miller2).shape),
    )
    primary_a = _geometry_source_signature_numeric(params.get("a"))
    primary_c = _geometry_source_signature_numeric(params.get("c"))
    secondary_a = float(av2) if av2 is not None else primary_a
    secondary_c = float(cv2) if cv2 is not None else primary_c
    requested_hit_table_sig = _geometry_source_snapshot_signature_from_params(
        params,
        mosaic_params=mosaic_params,
        optics_mode_component=0,
        include_mosaic_shape=False,
        sf_prune_bias=current_sf_prune_bias(),
        sf_prune_stats=simulation_runtime_state.sf_prune_stats,
        ordered_structure_scale=float(_current_ordered_structure_scale()),
        qr_cylinder_replace_requested=_qr_cylinder_replace_simulation_enabled(),
        primary_source_signature=primary_source_signature,
        secondary_source_signature=secondary_source_signature,
        secondary_a=secondary_a,
        secondary_c=secondary_c,
    )
    collect_hit_tables_requested = False
    try:
        collect_hit_tables_requested = bool(_should_collect_hit_tables_for_update())
    except Exception:
        collect_hit_tables_requested = False
    hit_tables_match_request = (
        requested_hit_table_sig == simulation_runtime_state.stored_hit_table_signature
    )
    collect_hit_tables_for_job = bool(collect_hit_tables_requested) or not bool(
        hit_tables_match_request
    )
    image_signature = _geometry_source_snapshot_signature_from_params(
        params,
        mosaic_params=mosaic_params,
        optics_mode_component=int(_current_optics_mode_flag()),
        include_mosaic_shape=True,
        sf_prune_bias=current_sf_prune_bias(),
        sf_prune_stats=simulation_runtime_state.sf_prune_stats,
        ordered_structure_scale=float(_current_ordered_structure_scale()),
        qr_cylinder_replace_requested=_qr_cylinder_replace_simulation_enabled(),
        primary_source_signature=primary_source_signature,
        secondary_source_signature=secondary_source_signature,
        secondary_a=secondary_a,
        secondary_c=secondary_c,
    )
    snapshot = simulation_runtime_state.source_row_snapshots.get(background_idx)
    if isinstance(snapshot, dict):
        snapshot_signature = snapshot.get("simulation_signature")
        if (
            isinstance(snapshot_signature, tuple)
            and len(snapshot_signature) == int(len(image_signature)) + 1
            and tuple(snapshot_signature[:-1]) == tuple(image_signature)
        ):
            return snapshot_signature
    return image_signature + (int(collect_hit_tables_for_job),)


def _capture_geometry_source_snapshot() -> None:
    background_index = int(background_runtime_state.current_background_index)
    previous_snapshot = dict(
        simulation_runtime_state.source_row_snapshots.get(int(background_index)) or {}
    )
    try:
        rows = [
            dict(entry)
            for entry in (_build_live_preview_simulated_peaks_from_cache() or ())
            if isinstance(entry, dict)
        ]
    except Exception:
        rows = []
    created_from = "none"
    for raw_record in simulation_runtime_state.peak_records or ():
        if not isinstance(raw_record, dict):
            continue
        try:
            display_col = float(raw_record.get("display_col", np.nan))
            display_row = float(raw_record.get("display_row", np.nan))
        except Exception:
            continue
        if np.isfinite(display_col) and np.isfinite(display_row):
            created_from = "peak_records"
            break
    if created_from == "none" and simulation_runtime_state.stored_max_positions_local is not None:
        created_from = "stored_max_positions_local"
    snapshot_payload = {
        "background_index": int(background_index),
        "simulation_signature": simulation_runtime_state.last_simulation_signature,
        "simulation_signature_summary": _live_cache_signature_summary(
            simulation_runtime_state.last_simulation_signature
        ),
        "rows": rows,
        "row_count": int(len(rows)),
        "created_from": created_from,
        "source_reflection_index_count": int(
            len(simulation_runtime_state.stored_source_reflection_indices_local or ())
        ),
    }
    retain_snapshot_cache = _retain_runtime_optional_cache(
        "source_snapshots",
        feature_needed=True,
    )
    if retain_snapshot_cache:
        simulation_runtime_state.source_row_snapshots[int(background_index)] = snapshot_payload
    else:
        simulation_runtime_state.source_row_snapshots.pop(int(background_index), None)
    _trace_live_cache_event(
        "source_snapshot",
        "store",
        background_index=int(background_index),
        outcome=("stored" if retain_snapshot_cache else "discarded"),
        overwrite=bool(previous_snapshot),
        previous_row_count=int(previous_snapshot.get("row_count", 0) or 0),
        row_count=int(len(rows)),
        created_from=created_from,
        previous_signature_summary=_live_cache_signature_summary(
            previous_snapshot.get("simulation_signature")
        ),
        signature_summary=_live_cache_signature_summary(
            simulation_runtime_state.last_simulation_signature
        ),
    )


def _geometry_manual_set_runtime_peak_cache_from_source_rows(
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
            normalize_hkl_key=_normalize_hkl_key,
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
    if _retain_runtime_optional_cache(
        "peak_overlay",
        feature_needed=bool(restored_records),
    ):
        simulation_runtime_state.peak_overlay_cache = {
            "sig": None,
            "positions": list(restored_positions),
            "millers": list(restored_millers),
            "intensities": list(restored_intensities),
            "records": [dict(record) for record in restored_records],
            "click_spatial_index": None,
            "restored_from_gui_state": False,
        }
    else:
        simulation_runtime_state.peak_overlay_cache = _empty_peak_overlay_cache()
    _invalidate_geometry_manual_pick_cache()


def _geometry_manual_build_source_rows_from_hit_tables(
    hit_tables: Sequence[object] | None,
    *,
    image_size_value: int,
    params_local: Mapping[str, object],
    allow_nominal_hkl_indices: bool,
) -> tuple[
    list[dict[str, object]],
    list[tuple[float, float, str]],
    list[object],
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
            native_sim_to_display_coords=_native_sim_to_display_coords,
            primary_a=primary_a,
            primary_c=primary_c,
            default_source_label="primary",
            round_pixel_centers=False,
            allow_nominal_hkl_indices=bool(allow_nominal_hkl_indices),
            owner="runtime_session._geometry_manual_build_source_rows_from_hit_tables",
        )
    )
    raw_rows = [dict(entry) for entry in (raw_rows or ()) if isinstance(entry, Mapping)]
    return raw_rows, peak_table_lattice, copied_hit_tables, source_reflection_indices


def _geometry_manual_logged_cache_matches_params(
    metadata: Mapping[str, object] | None,
    params_local: Mapping[str, object],
) -> bool:
    if not isinstance(metadata, Mapping):
        return False

    comparisons: list[tuple[object, float, float]] = []
    comparisons.append((metadata.get("av"), params_local.get("a"), 1.0e-6))
    comparisons.append((metadata.get("cv"), params_local.get("c"), 1.0e-6))
    comparisons.append((metadata.get("wavelength_center"), params_local.get("lambda"), 1.0e-6))
    comparisons.append((metadata.get("theta_center"), params_local.get("theta_initial"), 1.0e-6))

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


def _geometry_fit_background_label(background_index: int) -> str:
    """Return a short human-readable label for one background index."""

    try:
        osc_path = background_runtime_state.osc_files[int(background_index)]
    except Exception:
        osc_path = None
    if osc_path is not None:
        try:
            name = Path(str(osc_path)).name
        except Exception:
            name = ""
        if str(name).strip():
            return str(name)
    return f"background {int(background_index) + 1}"


def _commit_geometry_manual_source_row_rebuild_result(
    rebuild_result: gui_geometry_fit.GeometryFitSourceRowRebuildResult,
    *,
    consumer: str | None = None,
) -> list[dict[str, object]]:
    """Commit one pure source-row rebuild payload onto the live runtime state."""

    if not isinstance(rebuild_result, gui_geometry_fit.GeometryFitSourceRowRebuildResult):
        return []

    background_idx = int(rebuild_result.background_index)
    stored_rows = [
        dict(entry) for entry in (rebuild_result.stored_rows or ()) if isinstance(entry, Mapping)
    ]
    projected_rows = [
        dict(entry) for entry in (rebuild_result.projected_rows or ()) if isinstance(entry, Mapping)
    ]
    diagnostics = (
        dict(rebuild_result.diagnostics) if isinstance(rebuild_result.diagnostics, Mapping) else {}
    )
    lookup_context = str(diagnostics.get("consumer") or consumer or "unspecified")

    if stored_rows:
        hit_tables_local = rebuild_result.hit_tables
        if hit_tables_local is not None:
            try:
                max_positions_local = hit_tables_to_max_positions(hit_tables_local)
            except Exception:
                max_positions_local = np.empty((0, 6), dtype=np.float64)
            simulation_runtime_state.stored_max_positions_local = list(max_positions_local)
        simulation_runtime_state.stored_source_reflection_indices_local = (
            list(rebuild_result.source_reflection_indices)
            if isinstance(rebuild_result.source_reflection_indices, list)
            else None
        )
        if rebuild_result.peak_table_lattice is not None:
            simulation_runtime_state.stored_peak_table_lattice = list(
                rebuild_result.peak_table_lattice
            )
        simulation_runtime_state.stored_sim_image = np.zeros(
            (int(image_size), int(image_size)),
            dtype=np.float64,
        )
        if rebuild_result.intersection_cache is not None:
            simulation_runtime_state.stored_intersection_cache = _copy_intersection_cache_tables(
                rebuild_result.intersection_cache
            )
        simulation_runtime_state.last_simulation_signature = rebuild_result.requested_signature
        _geometry_manual_set_runtime_peak_cache_from_source_rows(stored_rows)

        snapshot_payload = {
            "background_index": int(background_idx),
            "simulation_signature": rebuild_result.requested_signature,
            "simulation_signature_summary": _live_cache_signature_summary(
                rebuild_result.requested_signature
            ),
            "rows": [dict(entry) for entry in stored_rows],
            "row_count": int(len(stored_rows)),
            "created_from": str(rebuild_result.rebuild_source or "unknown"),
            "source_reflection_index_count": int(
                len(rebuild_result.source_reflection_indices or ())
            ),
        }
        if _retain_runtime_optional_cache("source_snapshots", feature_needed=True):
            simulation_runtime_state.source_row_snapshots[int(background_idx)] = snapshot_payload

        if diagnostics:
            _set_geometry_manual_source_snapshot_diagnostics(**diagnostics)
        _trace_live_cache_event(
            "source_snapshot",
            "rebuild",
            background_index=int(background_idx),
            outcome="success",
            consumer=lookup_context,
            status=str(diagnostics.get("status", "snapshot_hit")),
            requested_signature_summary=diagnostics.get("requested_signature_summary"),
            raw_peak_count=int(len(stored_rows)),
            projected_peak_count=int(len(projected_rows)),
            rebuild_source=str(rebuild_result.rebuild_source or "unknown"),
        )
        return projected_rows

    if diagnostics:
        _set_geometry_manual_source_snapshot_diagnostics(**diagnostics)
    _trace_live_cache_event(
        "source_snapshot",
        "rebuild",
        background_index=int(background_idx),
        outcome="failed",
        consumer=lookup_context,
        status=str(diagnostics.get("status", "snapshot_rebuild_failed")),
        requested_signature_summary=diagnostics.get("requested_signature_summary"),
    )
    return []


def _geometry_manual_rebuild_source_rows_for_background(
    background_index: int,
    param_set: dict[str, object] | None = None,
    *,
    consumer: str | None = None,
    prior_diagnostics: Mapping[str, object] | None = None,
    required_pairs: Sequence[Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    background_idx = int(background_index)
    lookup_context = str(consumer or "unspecified")
    params_local = dict(_current_geometry_fit_params())
    if isinstance(param_set, Mapping):
        params_local.update(dict(param_set))

    requested_signature = _geometry_source_snapshot_signature_for_background(
        background_idx,
        params_local,
    )
    requested_signature_summary = _live_cache_signature_summary(requested_signature)
    can_use_live_runtime_cache = background_idx == int(
        background_runtime_state.current_background_index
    )
    live_cache_inventory = _live_cache_inventory_snapshot()
    background_label = _geometry_fit_background_label(background_idx)

    def _build_source_rows_for_rebuild(
        source_tables: Sequence[object] | None,
    ) -> tuple[list[dict[str, object]], list[tuple[float, float, str]], list[object]]:
        table_list = list(source_tables or ())
        if not table_list:
            return [], [], []
        first_table = np.asarray(table_list[0], dtype=np.float64)
        if first_table.ndim == 2 and first_table.shape[1] >= 14:
            hit_tables_local = intersection_cache_to_hit_tables(table_list)
        else:
            hit_tables_local = _copy_hit_tables(table_list)
        return _geometry_manual_build_source_rows_from_hit_tables(
            hit_tables_local,
            image_size_value=int(image_size),
            params_local=params_local,
            allow_nominal_hkl_indices=True,
        )

    def _build_geometry_fit_live_rows_payload_from_cache() -> (
        dict[str, object] | list[dict[str, object]]
    ):
        live_rows = [
            dict(entry)
            for entry in (_build_live_preview_simulated_peaks_from_cache() or ())
            if isinstance(entry, Mapping)
        ]
        cache_metadata = (
            dict(_last_live_preview_cache_metadata())
            if callable(_last_live_preview_cache_metadata)
            else {}
        )
        if cache_metadata:
            return {
                "rows": live_rows,
                "cache_metadata": cache_metadata,
            }
        return live_rows

    rebuild_result = gui_geometry_fit.rebuild_geometry_fit_source_rows(
        background_index=int(background_idx),
        background_label=background_label,
        params_local=params_local,
        consumer=lookup_context,
        prior_diagnostics=prior_diagnostics,
        requested_signature=requested_signature,
        requested_signature_summary=requested_signature_summary,
        can_use_live_runtime_cache=can_use_live_runtime_cache,
        build_live_rows=_build_geometry_fit_live_rows_payload_from_cache,
        get_memory_intersection_cache=get_last_intersection_cache,
        memory_cache_signature=simulation_runtime_state.last_simulation_signature,
        load_logged_intersection_cache=(lambda: load_most_recent_logged_intersection_cache()),
        logged_cache_matches_params=_geometry_manual_logged_cache_matches_params,
        build_source_rows_from_hit_tables=_build_source_rows_for_rebuild,
        simulate_hit_tables=(
            lambda normalized_params: _simulate_hit_tables_for_fit(
                miller,
                intensities,
                int(image_size),
                normalized_params,
            )
        ),
        last_runtime_simulation_diagnostics=(_geometry_manual_last_simulation_diagnostics),
        project_rows=(
            _project_geometry_manual_peaks_to_current_view
            if callable(_project_geometry_manual_peaks_to_current_view)
            else None
        ),
        required_pairs=required_pairs,
        live_cache_inventory=live_cache_inventory,
    )
    return _commit_geometry_manual_source_row_rebuild_result(
        rebuild_result,
        consumer=lookup_context,
    )


def _geometry_manual_source_rows_for_background(
    background_index: int,
    param_set: dict[str, object] | None = None,
    *,
    consumer: str | None = None,
    required_pairs: Sequence[Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    background_idx = int(background_index)
    lookup_context = str(consumer or "unspecified")
    # Import/manual-overlay refreshes should reuse restored peak records and
    # avoid forcing a rebuild until geometry-fit preflight explicitly needs one.
    allow_source_snapshot_rebuild = bool(lookup_context == "geometry_fit_dataset")
    requested_signature = _geometry_source_snapshot_signature_for_background(
        background_idx,
        param_set,
    )
    requested_signature_summary = _live_cache_signature_summary(requested_signature)
    snapshot = dict(simulation_runtime_state.source_row_snapshots.get(background_idx) or {})
    if not snapshot:
        live_cache_inventory = _live_cache_inventory_snapshot()
        _set_geometry_manual_source_snapshot_diagnostics(
            source="source_snapshot",
            cache_family="source_snapshot",
            action="lookup",
            consumer=lookup_context,
            status="snapshot_missing_background",
            background_index=int(background_idx),
            requested_signature=requested_signature,
            requested_signature_summary=requested_signature_summary,
            raw_peak_count=0,
            projected_peak_count=0,
            signature_match=False,
            live_cache_inventory=live_cache_inventory,
        )
        _trace_live_cache_event(
            "source_snapshot",
            "lookup",
            background_index=int(background_idx),
            outcome="miss",
            consumer=lookup_context,
            status="snapshot_missing_background",
            requested_signature_summary=requested_signature_summary,
            raw_peak_count=0,
            projected_peak_count=0,
            signature_match=False,
        )
        if allow_source_snapshot_rebuild:
            rebuilt_rows = _geometry_manual_rebuild_source_rows_for_background(
                background_idx,
                param_set,
                consumer=lookup_context,
                prior_diagnostics=_geometry_manual_last_source_snapshot_diagnostics(),
                required_pairs=required_pairs,
            )
            if rebuilt_rows:
                return rebuilt_rows
        return []
    snapshot_signature = snapshot.get("simulation_signature")
    stored_signature_summary = _live_cache_signature_summary(snapshot_signature)
    created_from = snapshot.get("created_from")
    raw_rows = [
        dict(entry) for entry in (snapshot.get("rows", ()) or ()) if isinstance(entry, dict)
    ]
    if snapshot_signature != requested_signature:
        live_cache_inventory = _live_cache_inventory_snapshot()
        _set_geometry_manual_source_snapshot_diagnostics(
            source="source_snapshot",
            cache_family="source_snapshot",
            action="lookup",
            consumer=lookup_context,
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
            live_cache_inventory=live_cache_inventory,
        )
        _trace_live_cache_event(
            "source_snapshot",
            "lookup",
            background_index=int(background_idx),
            outcome="mismatch",
            consumer=lookup_context,
            status="snapshot_signature_mismatch",
            requested_signature_summary=requested_signature_summary,
            stored_signature_summary=stored_signature_summary,
            created_from=created_from,
            raw_peak_count=int(len(raw_rows)),
            projected_peak_count=0,
            signature_match=False,
        )
        if allow_source_snapshot_rebuild:
            rebuilt_rows = _geometry_manual_rebuild_source_rows_for_background(
                background_idx,
                param_set,
                consumer=lookup_context,
                prior_diagnostics=_geometry_manual_last_source_snapshot_diagnostics(),
                required_pairs=required_pairs,
            )
            if rebuilt_rows:
                return rebuilt_rows
        return []
    if not raw_rows:
        live_cache_inventory = _live_cache_inventory_snapshot()
        _set_geometry_manual_source_snapshot_diagnostics(
            source="source_snapshot",
            cache_family="source_snapshot",
            action="lookup",
            consumer=lookup_context,
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
            live_cache_inventory=live_cache_inventory,
        )
        _trace_live_cache_event(
            "source_snapshot",
            "lookup",
            background_index=int(background_idx),
            outcome="empty",
            consumer=lookup_context,
            status="snapshot_empty",
            requested_signature_summary=requested_signature_summary,
            stored_signature_summary=stored_signature_summary,
            created_from=created_from,
            raw_peak_count=0,
            projected_peak_count=0,
            signature_match=True,
        )
        if allow_source_snapshot_rebuild:
            rebuilt_rows = _geometry_manual_rebuild_source_rows_for_background(
                background_idx,
                param_set,
                consumer=lookup_context,
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
        live_cache_inventory = _live_cache_inventory_snapshot()
        _set_geometry_manual_source_snapshot_diagnostics(
            source="source_snapshot",
            cache_family="source_snapshot",
            action="lookup",
            consumer=lookup_context,
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
            live_cache_inventory=live_cache_inventory,
            live_runtime_cache_validation=snapshot_validation,
        )
        _trace_live_cache_event(
            "source_snapshot",
            "lookup",
            background_index=int(background_idx),
            outcome="validation_failed",
            consumer=lookup_context,
            status="snapshot_pair_validation_failed",
            requested_signature_summary=requested_signature_summary,
            stored_signature_summary=stored_signature_summary,
            created_from=created_from,
            raw_peak_count=int(len(raw_rows)),
            projected_peak_count=0,
            signature_match=True,
        )
        if allow_source_snapshot_rebuild:
            rebuilt_rows = _geometry_manual_rebuild_source_rows_for_background(
                background_idx,
                param_set,
                consumer=lookup_context,
                prior_diagnostics=_geometry_manual_last_source_snapshot_diagnostics(),
                required_pairs=required_pairs,
            )
            if rebuilt_rows:
                return rebuilt_rows
        return []
    projected_rows = (
        _project_geometry_manual_peaks_to_current_view(raw_rows)
        if callable(_project_geometry_manual_peaks_to_current_view)
        else raw_rows
    )
    projected_rows = [dict(entry) for entry in (projected_rows or ()) if isinstance(entry, dict)]
    live_cache_inventory = _live_cache_inventory_snapshot()
    _set_geometry_manual_source_snapshot_diagnostics(
        source="source_snapshot",
        cache_family="source_snapshot",
        action="lookup",
        consumer=lookup_context,
        status="snapshot_hit",
        background_index=int(background_idx),
        requested_signature=requested_signature,
        requested_signature_summary=requested_signature_summary,
        snapshot_signature=snapshot_signature,
        stored_signature_summary=stored_signature_summary,
        raw_peak_count=int(len(raw_rows)),
        projected_peak_count=int(len(projected_rows)),
        created_from=created_from,
        signature_match=True,
        live_cache_inventory=live_cache_inventory,
        live_runtime_cache_validation=snapshot_validation,
    )
    _trace_live_cache_event(
        "source_snapshot",
        "lookup",
        background_index=int(background_idx),
        outcome="hit",
        consumer=lookup_context,
        status="snapshot_hit",
        requested_signature_summary=requested_signature_summary,
        stored_signature_summary=stored_signature_summary,
        created_from=created_from,
        raw_peak_count=int(len(raw_rows)),
        projected_peak_count=int(len(projected_rows)),
        signature_match=True,
    )
    return projected_rows


geometry_q_group_workflow = gui_runtime_geometry_preview.build_runtime_geometry_q_group_workflow(
    bootstrap_module=gui_bootstrap,
    geometry_q_group_manager_module=gui_geometry_q_group_manager,
    root=root,
    view_state=geometry_q_group_view_state,
    preview_state=geometry_preview_state,
    q_group_state=geometry_q_group_state,
    fit_config=fit_config,
    current_geometry_fit_var_names_factory=_current_geometry_fit_var_names,
    build_entries_snapshot=_build_geometry_q_group_entries,
    invalidate_geometry_manual_pick_cache=_invalidate_geometry_manual_pick_cache,
    update_geometry_preview_exclude_button_label=_update_geometry_preview_exclude_button_label,
    live_geometry_preview_enabled=lambda: (
        (PREVIEW_CALCULATIONS_ENABLED and bool(live_geometry_preview_var.get()))
        if "live_geometry_preview_var" in globals()
        else False
    ),
    refresh_live_geometry_preview=(lambda: _refresh_live_geometry_preview(update_status=True)),
    set_hkl_pick_mode=hkl_lookup_controls_runtime.set_hkl_pick_mode,
    live_preview_match_key=_live_preview_match_key,
    live_preview_match_hkl=_live_preview_match_hkl,
    render_live_geometry_preview_state=(
        lambda: gui_geometry_q_group_manager.render_runtime_live_geometry_preview_state(
            geometry_q_group_runtime_bindings_factory(),
            update_status=True,
        )
    ),
    clear_geometry_preview_artists=_clear_geometry_preview_artists,
    preview_toggle_max_distance_px=float(GEOMETRY_PREVIEW_TOGGLE_MAX_DISTANCE_PX),
    update_running_factory=lambda: bool(
        simulation_runtime_state.update_running
        or simulation_runtime_state.worker_active_job is not None
        or simulation_runtime_state.worker_queued_job is not None
    ),
    has_cached_hit_tables_factory=lambda: (
        simulation_runtime_state.stored_max_positions_local is not None
    ),
    build_live_preview_simulated_peaks_from_cache=(_build_live_preview_simulated_peaks_from_cache),
    miller_factory=lambda: miller,
    intensities_factory=lambda: intensities,
    image_size_value_factory=lambda: image_size,
    current_geometry_fit_params_factory=_current_geometry_fit_params,
    filter_simulated_peaks=_filter_geometry_fit_simulated_peaks,
    collapse_simulated_peaks=_collapse_geometry_fit_simulated_peaks,
    excluded_q_group_count=_geometry_q_group_excluded_count,
    caked_view_enabled=_active_caked_primary_view,
    background_visible_factory=lambda: bool(background_runtime_state.visible),
    current_background_display_factory=_get_current_background_display,
    axis=ax,
    geometry_preview_artists=geometry_runtime_state.preview_artists,
    draw_idle_factory=lambda: canvas.draw_idle,
    normalize_hkl_key=_normalize_hkl_key,
    live_preview_match_is_excluded=_live_preview_match_is_excluded,
    filter_live_preview_matches=_filter_live_preview_matches,
    refresh_live_geometry_preview_quiet=(
        lambda: _refresh_live_geometry_preview(update_status=False)
    ),
    clear_last_simulation_signature=(
        lambda: setattr(
            simulation_runtime_state,
            "last_simulation_signature",
            None,
        )
    ),
    schedule_update_resolver=(lambda: schedule_update),
    set_status_text_factory=lambda: (
        (lambda text: progress_label_geometry.config(text=text))
        if "progress_label_geometry" in globals()
        else None
    ),
    file_dialog_dir_factory=lambda: get_dir("file_dialog_dir"),
    asksaveasfilename=filedialog.asksaveasfilename,
    askopenfilename=filedialog.askopenfilename,
)
geometry_q_group_runtime = geometry_q_group_workflow.runtime
geometry_q_group_runtime_bindings_factory = geometry_q_group_workflow.bindings_factory
geometry_q_group_runtime_callbacks = geometry_q_group_workflow.callbacks
_live_geometry_preview_enabled = geometry_q_group_workflow.live_preview_enabled
_render_live_geometry_preview_state = geometry_q_group_workflow.render_live_preview_state
_set_geometry_preview_exclude_mode_impl = geometry_q_group_workflow.set_preview_exclude_mode
_clear_live_geometry_preview_exclusions_impl = geometry_q_group_workflow.clear_preview_exclusions
_toggle_live_geometry_preview_exclusion_at = geometry_q_group_workflow.toggle_preview_exclusion_at
_on_live_geometry_preview_toggle_impl = geometry_q_group_workflow.toggle_live_preview


def _set_geometry_preview_exclude_mode(enabled: bool, message: str | None = None):
    result = _set_geometry_preview_exclude_mode_impl(enabled, message=message)
    _refresh_fast_viewer_runtime_mode(announce=False)
    return result


def _clear_live_geometry_preview_exclusions(*args, **kwargs):
    result = _clear_live_geometry_preview_exclusions_impl(*args, **kwargs)
    _refresh_fast_viewer_runtime_mode(announce=False)
    return result


def _on_live_geometry_preview_toggle(*args, **kwargs):
    if not PREVIEW_CALCULATIONS_ENABLED:
        try:
            live_geometry_preview_var.set(False)
        except Exception:
            pass
        gui_controllers.clear_geometry_preview_skip_once(geometry_preview_state)
        _clear_geometry_preview_artists(redraw=True)
        _refresh_fast_viewer_runtime_mode(announce=False)
        return False
    result = _on_live_geometry_preview_toggle_impl(*args, **kwargs)
    _refresh_fast_viewer_runtime_mode(announce=False)
    return result


geometry_fit_simulation_runtime_callbacks = (
    gui_geometry_q_group_manager.make_runtime_geometry_fit_simulation_callbacks(
        process_peaks_parallel=process_peaks_parallel,
        hit_tables_to_max_positions=hit_tables_to_max_positions,
        native_sim_to_display_coords=_native_sim_to_display_coords,
        peak_table_lattice_factory=lambda: (
            simulation_runtime_state.stored_peak_table_lattice
            if isinstance(simulation_runtime_state.stored_peak_table_lattice, list)
            else None
        ),
        primary_a_factory=lambda: float(a_var.get()) if "a_var" in globals() else float(av),
        primary_c_factory=lambda: float(c_var.get()) if "c_var" in globals() else float(cv),
        default_source_label=None,
        round_pixel_centers=False,
        default_solve_q_steps=DEFAULT_SOLVE_Q_STEPS,
        default_solve_q_rel_tol=DEFAULT_SOLVE_Q_REL_TOL,
        default_solve_q_mode=DEFAULT_SOLVE_Q_MODE,
        prefer_safe_python_runner=True,
    )
)
_simulate_hit_tables_for_fit = geometry_fit_simulation_runtime_callbacks.simulate_hit_tables
_simulate_hkl_peak_centers_for_fit = geometry_fit_simulation_runtime_callbacks.simulate_peak_centers


def _legacy_auto_match_on_fit_geometry_click():
    _clear_geometry_pick_artists()
    _clear_geometry_preview_artists()

    def _cmd_line(text: str) -> None:
        try:
            print(f"[geometry-fit] {text}", flush=True)
        except Exception:
            pass

    params = _current_geometry_fit_params()
    try:
        mosaic_params = build_mosaic_params()
    except Exception:
        mosaic_params = {}
    var_names = _current_geometry_fit_var_names()
    if not var_names:
        _cmd_line("aborted: no geometry parameters selected")
        progress_label_geometry.config(text="No parameters selected!")
        return
    _cmd_line(f"start: vars={','.join(var_names)}")
    preserve_live_theta = "theta_initial" not in var_names and "theta_offset" not in var_names

    geometry_refine_cfg = fit_config.get("geometry", {}) if isinstance(fit_config, dict) else {}
    if not isinstance(geometry_refine_cfg, dict):
        geometry_refine_cfg = {}
    auto_match_cfg = geometry_refine_cfg.get("auto_match", {}) or {}
    if not isinstance(auto_match_cfg, dict):
        auto_match_cfg = {}
    orientation_cfg = geometry_refine_cfg.get("orientation", {}) or {}
    if not isinstance(orientation_cfg, dict):
        orientation_cfg = {}

    if not _apply_geometry_fit_background_selection(
        trigger_update=False,
        sync_live_theta=not preserve_live_theta,
    ):
        return

    try:
        selected_background_indices = _current_geometry_fit_background_indices(strict=True)
    except Exception as exc:
        _cmd_line(f"aborted: invalid fit background selection ({exc})")
        progress_label_geometry.config(
            text=f"Geometry fit unavailable: invalid fit background selection ({exc})."
        )
        return
    if int(background_runtime_state.current_background_index) not in {
        int(idx) for idx in selected_background_indices
    }:
        _cmd_line("aborted: active background missing from fit selection")
        progress_label_geometry.config(
            text=(
                "Geometry fit unavailable: the active background must be part of the fit "
                "selection so the overlay can be drawn on the current image."
            )
        )
        return

    joint_background_mode = False
    background_theta_values: list[float] = []
    shared_theta_offset_seed = 0.0
    if _geometry_fit_uses_shared_theta_offset(selected_background_indices):
        if not _apply_background_theta_metadata(
            trigger_update=False,
            sync_live_theta=not preserve_live_theta,
        ):
            _cmd_line("aborted: invalid background theta_i/shared offset settings")
            progress_label_geometry.config(
                text="Geometry fit unavailable: invalid background theta_i/shared offset settings."
            )
            return
        try:
            background_theta_values = _current_background_theta_values(strict_count=True)
            shared_theta_offset_seed = _current_geometry_theta_offset(strict=True)
        except Exception as exc:
            _cmd_line(f"aborted: failed to parse background theta settings ({exc})")
            progress_label_geometry.config(
                text=f"Geometry fit unavailable: failed to parse background theta settings ({exc})."
            )
            return
        joint_background_mode = len(selected_background_indices) > 1
        params["theta_offset"] = float(shared_theta_offset_seed)
        if background_theta_values:
            try:
                params["theta_initial"] = float(
                    background_theta_values[int(background_runtime_state.current_background_index)]
                )
            except Exception:
                pass
    else:
        params["theta_offset"] = 0.0

    native_background = _get_current_background_native()
    backend_background = _get_current_background_backend()
    display_background = background_runtime_state.current_background_display
    if display_background is None and native_background is not None:
        display_background = np.rot90(native_background, DISPLAY_ROTATE_K)
    if native_background is None or display_background is None:
        _cmd_line("aborted: no background image loaded")
        progress_label_geometry.config(
            text="Geometry fit unavailable: no background image is loaded."
        )
        return
    if backend_background is None:
        backend_background = native_background

    miller_array = np.asarray(miller, dtype=np.float64)
    intensity_array = np.asarray(intensities, dtype=np.float64)
    if miller_array.ndim != 2 or miller_array.shape[1] != 3 or miller_array.size == 0:
        _cmd_line("aborted: no simulated reflections available")
        progress_label_geometry.config(
            text="Geometry fit unavailable: no simulated reflections are available."
        )
        return
    if intensity_array.shape[0] != miller_array.shape[0]:
        _cmd_line("aborted: intensity array does not match HKLs")
        progress_label_geometry.config(
            text="Geometry fit unavailable: intensity array does not match HKLs."
        )
        return

    try:
        simulated_peaks = []
    except Exception as exc:
        _cmd_line(f"aborted: failed to simulate peak centers ({exc})")
        progress_label_geometry.config(
            text=f"Geometry fit unavailable: failed to simulate peak centers ({exc})."
        )
        return

    if not simulated_peaks:
        _cmd_line("aborted: no simulated Bragg peak centers were found")
        progress_label_geometry.config(
            text="Geometry fit unavailable: no simulated Bragg peak centers were found."
        )
        return

    simulated_peaks, excluded_q_peaks, q_group_total = _filter_geometry_fit_simulated_peaks(
        simulated_peaks
    )
    if not simulated_peaks:
        _cmd_line("aborted: no Qr/Qz groups selected for fitting")
        progress_label_geometry.config(
            text="Geometry fit unavailable: no Qr/Qz groups are selected for fitting."
        )
        return
    default_min_matches = max(6, len(var_names) + 2)
    min_matches = int(auto_match_cfg.get("min_matches", default_min_matches))
    min_matches = max(1, min_matches)
    fit_auto_match_cfg = dict(auto_match_cfg)
    fit_auto_match_cfg["console_progress"] = True
    fit_auto_match_cfg.setdefault("console_progress_every", 10)
    fit_auto_match_cfg.setdefault(
        "context_margin_px",
        max(256.0, 10.0 * float(auto_match_cfg.get("search_radius_px", 24.0))),
    )
    raw_simulated_peak_count = len(simulated_peaks)
    simulated_peaks, collapsed_deg_fit = _collapse_geometry_fit_simulated_peaks(
        simulated_peaks,
        merge_radius_px=float(
            fit_auto_match_cfg.get(
                "degenerate_merge_radius_px",
                min(
                    6.0,
                    0.33 * float(fit_auto_match_cfg.get("search_radius_px", 24.0)),
                ),
            )
        ),
    )
    if not simulated_peaks:
        _cmd_line("aborted: no geometry-fit seeds remain after Qr/Qz collapse")
        progress_label_geometry.config(
            text="Geometry fit unavailable: no geometry-fit seeds remain after collapsing degenerate Qr/Qz peaks."
        )
        return
    _cmd_line(
        "prepared: "
        f"simulated_peaks={len(simulated_peaks)} "
        f"(collapsed={collapsed_deg_fit} from {raw_simulated_peak_count}) "
        f"q_groups_on={max(0, q_group_total - _geometry_q_group_excluded_count())}/{q_group_total}"
    )

    _cmd_line(f"auto-match: local-peak match start (min_matches={min_matches})")

    matched_pairs, match_stats, effective_auto_match_cfg, auto_match_attempts = (
        _auto_match_background_peaks_with_relaxation(
            simulated_peaks,
            display_background,
            fit_auto_match_cfg,
            min_matches=min_matches,
        )
    )
    matched_pairs, match_stats, excluded_preview_count = _apply_live_preview_match_exclusions(
        matched_pairs,
        match_stats,
    )
    _cmd_line(
        "auto-match: "
        f"matched={len(matched_pairs)}/"
        f"{int(match_stats.get('simulated_count', len(simulated_peaks)))} "
        f"qualified_summits={int(match_stats.get('qualified_summit_count', 0))} "
        f"claimed_summits={int(match_stats.get('claimed_summit_count', 0))} "
        f"mean={float(match_stats.get('mean_match_distance_px', np.nan)):.2f}px "
        f"p90={float(match_stats.get('p90_match_distance_px', np.nan)):.2f}px "
        f"excluded_preview={excluded_preview_count}"
    )

    if len(matched_pairs) < min_matches:
        _cmd_line(f"cancelled: minimum-match gate failed ({len(matched_pairs)} < {min_matches})")
        simulated_count = int(match_stats.get("simulated_count", len(simulated_peaks)))
        progress_label_geometry.config(
            text=(
                "Geometry fit cancelled: local-peak auto-match found "
                f"{len(matched_pairs)}/{simulated_count} active peaks "
                f"(need at least {min_matches})."
                + (
                    f" Qr/Qz groups off={_geometry_q_group_excluded_count()}/{q_group_total}."
                    if q_group_total > 0
                    else ""
                )
                + (
                    f" Excluded in preview={excluded_preview_count}."
                    if excluded_preview_count > 0
                    else ""
                )
                + " Try Auto-Match Scale, then retry."
            )
        )
        return

    max_auto_p90 = float(auto_match_cfg.get("max_p90_distance_px", 35.0))
    max_auto_mean = float(auto_match_cfg.get("max_mean_distance_px", 22.0))
    p90_dist = float(match_stats.get("p90_match_distance_px", np.nan))
    mean_dist = float(match_stats.get("mean_match_distance_px", np.nan))
    quality_fail = (
        np.isfinite(max_auto_p90) and np.isfinite(p90_dist) and p90_dist > max_auto_p90
    ) or (np.isfinite(max_auto_mean) and np.isfinite(mean_dist) and mean_dist > max_auto_mean)
    if quality_fail:
        _cmd_line(
            "cancelled: quality gate failed "
            f"(mean={mean_dist:.2f}px, p90={p90_dist:.2f}px, "
            f"limits={max_auto_mean:.2f}/{max_auto_p90:.2f}px)"
        )
        progress_label_geometry.config(
            text=(
                "Geometry fit cancelled: auto-match quality gate failed "
                f"(mean={mean_dist:.1f}px, p90={p90_dist:.1f}px; "
                f"limits mean<={max_auto_mean:.1f}px, p90<={max_auto_p90:.1f}px). "
                + (
                    f"Qr/Qz groups off={_geometry_q_group_excluded_count()}/{q_group_total}. "
                    if q_group_total > 0
                    else ""
                )
                + (
                    f"Excluded in preview={excluded_preview_count}. "
                    if excluded_preview_count > 0
                    else ""
                )
                + "Tighten auto-match or use manual peak picks."
            )
        )
        return

    def _attach_overlay_match_indices(
        pairs: Sequence[dict[str, object]] | None,
    ) -> list[dict[str, object]]:
        normalized_pairs: list[dict[str, object]] = []
        for pair_idx, raw_entry in enumerate(pairs or []):
            if not isinstance(raw_entry, dict):
                continue
            entry = dict(raw_entry)
            try:
                overlay_idx = int(entry.get("overlay_match_index", pair_idx))
            except Exception:
                overlay_idx = int(pair_idx)
            if overlay_idx < 0:
                overlay_idx = int(pair_idx)
            entry["overlay_match_index"] = int(overlay_idx)
            normalized_pairs.append(entry)
        return normalized_pairs

    matched_pairs = _attach_overlay_match_indices(matched_pairs)

    initial_auto_pairs_display: list[dict[str, object]] = []
    for entry in matched_pairs:
        hkl_key = _normalize_hkl_key(entry.get("hkl", entry.get("label")))
        if hkl_key is None:
            continue
        try:
            sim_col = float(entry["sim_x"])
            sim_row = float(entry["sim_y"])
            bg_col = float(entry["x"])
            bg_row = float(entry["y"])
        except Exception:
            continue
        if not all(np.isfinite(v) for v in (sim_col, sim_row, bg_col, bg_row)):
            continue
        sim_display = _rotate_point_for_display(
            sim_col,
            sim_row,
            (image_size, image_size),
            SIM_DISPLAY_ROTATE_K,
        )
        initial_auto_pairs_display.append(
            {
                "overlay_match_index": int(
                    entry.get("overlay_match_index", len(initial_auto_pairs_display))
                ),
                "hkl": hkl_key,
                "sim_display": (float(sim_display[0]), float(sim_display[1])),
                "bg_display": (bg_col, bg_row),
            }
        )
    preview_marker_limit = int(auto_match_cfg.get("max_display_markers", 120))
    preview_marker_limit = max(1, preview_marker_limit)
    _draw_initial_geometry_pairs_overlay(
        initial_auto_pairs_display,
        max_display_markers=preview_marker_limit,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = gui_geometry_fit.build_geometry_fit_log_path(
        stamp=stamp,
        log_dir=get_dir("debug_log_dir"),
    )
    log_file = None
    if gui_geometry_fit.geometry_fit_log_files_enabled():
        log_file = log_path.open("w", encoding="utf-8")
        _cmd_line(f"log: {log_path}")

    def _log_line(text: str = ""):
        try:
            log_file.write(text + "\n")
            log_file.flush()
        except Exception:
            pass

    def _log_section(title: str, lines: list[str]):
        _log_line(title)
        for line in lines:
            _log_line(f"  {line}")
        _log_line()

    def _log_format_hkl(entry: object) -> str:
        if not isinstance(entry, dict):
            return "n/a"
        hkl_key = _normalize_hkl_key(entry.get("hkl", entry.get("label")))
        return str(hkl_key) if hkl_key is not None else str(entry.get("label", "n/a"))

    def _log_format_point(entry: object, x_key: str = "x", y_key: str = "y") -> str:
        if not isinstance(entry, dict):
            return "n/a"
        try:
            x_val = float(entry.get(x_key, np.nan))
            y_val = float(entry.get(y_key, np.nan))
        except Exception:
            return "n/a"
        if not (np.isfinite(x_val) and np.isfinite(y_val)):
            return "n/a"
        return f"({x_val:.3f},{y_val:.3f})"

    def _log_radius_to_center(
        entry: object,
        *,
        center_pair: tuple[float, float] | None = None,
    ) -> str:
        if center_pair is None or not isinstance(entry, dict):
            return "n/a"
        try:
            x_val = float(entry.get("x", np.nan))
            y_val = float(entry.get("y", np.nan))
            center_row = float(center_pair[0])
            center_col = float(center_pair[1])
        except Exception:
            return "n/a"
        if not (
            np.isfinite(x_val)
            and np.isfinite(y_val)
            and np.isfinite(center_row)
            and np.isfinite(center_col)
        ):
            return "n/a"
        return f"{math.hypot(x_val - center_col, y_val - center_row):.3f}px"

    def _log_source_descriptor(entry: object) -> str:
        if not isinstance(entry, dict):
            return "src=n/a"
        parts: list[str] = []
        if entry.get("source_label") is not None:
            parts.append(f"label={entry.get('source_label')}")
        if entry.get("source_table_index") is not None:
            parts.append(f"table={entry.get('source_table_index')}")
        if entry.get("source_row_index") is not None:
            parts.append(f"row={entry.get('source_row_index')}")
        if entry.get("source_peak_index") is not None:
            parts.append(f"peak={entry.get('source_peak_index')}")
        return "src=" + (",".join(parts) if parts else "n/a")

    def _log_fixed_peak_frames(
        display_entries: Sequence[dict[str, object]] | None,
        native_entries: Sequence[dict[str, object]] | None,
        fit_entries: Sequence[dict[str, object]] | None,
        *,
        center_pair: tuple[float, float] | None = None,
    ) -> list[str]:
        lines: list[str] = []
        count = max(
            len(display_entries or []),
            len(native_entries or []),
            len(fit_entries or []),
        )
        for idx in range(count):
            display_entry = (
                dict(display_entries[idx])
                if display_entries is not None and idx < len(display_entries)
                else None
            )
            native_entry = (
                dict(native_entries[idx])
                if native_entries is not None and idx < len(native_entries)
                else None
            )
            fit_entry = (
                dict(fit_entries[idx])
                if fit_entries is not None and idx < len(fit_entries)
                else None
            )
            base_entry = display_entry or native_entry or fit_entry or {}
            try:
                sim_x = float(base_entry.get("sim_x", np.nan))
                sim_y = float(base_entry.get("sim_y", np.nan))
                sim_text = (
                    f"({sim_x:.3f},{sim_y:.3f})"
                    if np.isfinite(sim_x) and np.isfinite(sim_y)
                    else "n/a"
                )
            except Exception:
                sim_text = "n/a"
            overlay_match_index = base_entry.get("overlay_match_index")
            lines.append(
                f"idx={idx} overlay={overlay_match_index if overlay_match_index is not None else 'n/a'} "
                f"HKL={_log_format_hkl(base_entry)} {_log_source_descriptor(base_entry)} "
                f"sim_display={sim_text} "
                f"meas_display={_log_format_point(display_entry)} "
                f"meas_native={_log_format_point(native_entry)} "
                f"meas_fit={_log_format_point(fit_entry)} "
                f"fit_r={_log_radius_to_center(fit_entry, center_pair=center_pair)}"
            )
        return lines or ["<none>"]

    def _log_point_match_diagnostics(entries: object) -> list[str]:
        if not isinstance(entries, list) or not entries:
            return ["<none>"]
        lines: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            hkl_key = entry.get("hkl", "n/a")
            src_text = (
                f"table={entry.get('source_table_index', 'n/a')},"
                f"row={entry.get('source_row_index', 'n/a')}"
            )
            try:
                meas_x = float(entry.get("measured_x", np.nan))
                meas_y = float(entry.get("measured_y", np.nan))
                sim_x = float(entry.get("simulated_x", np.nan))
                sim_y = float(entry.get("simulated_y", np.nan))
                dist_px = float(entry.get("distance_px", np.nan))
                meas_r = float(entry.get("measured_radius_px", np.nan))
                sim_r = float(entry.get("simulated_radius_px", np.nan))
            except Exception:
                meas_x = meas_y = sim_x = sim_y = dist_px = meas_r = sim_r = float("nan")
            lines.append(
                f"idx={entry.get('match_input_index', 'n/a')} "
                f"dataset={entry.get('dataset_index', 'n/a')}:{entry.get('dataset_label', 'n/a')} "
                f"overlay={entry.get('overlay_match_index', 'n/a')} HKL={hkl_key} "
                f"label={entry.get('label', 'n/a')} src={src_text} "
                f"resolution={entry.get('resolution_kind', 'n/a')}/"
                f"{entry.get('resolution_reason', 'n/a')} "
                f"status={entry.get('match_status', 'n/a')} "
                f"kind={entry.get('match_kind', 'n/a')} "
                f"meas=({meas_x:.3f},{meas_y:.3f}) "
                f"sim=({sim_x:.3f},{sim_y:.3f}) "
                f"d={dist_px:.3f}px meas_r={meas_r:.3f}px sim_r={sim_r:.3f}px"
            )
        return lines or ["<none>"]

    progress_label_geometry.config(text="Running geometry fit (auto-matched peaks)…")
    root.update_idletasks()
    _cmd_line(
        f"fit: running optimizer with {len(matched_pairs)} matched peaks and orientation auto-selection"
    )

    try:
        measured_from_display = []
        for entry in matched_pairs:
            try:
                x_val = float(entry["x"])
                y_val = float(entry["y"])
            except Exception:
                continue
            if not (np.isfinite(x_val) and np.isfinite(y_val)):
                continue
            normalized_hkl = _normalize_hkl_key(entry.get("hkl", entry.get("label")))
            measured_entry = dict(entry)
            measured_entry["label"] = str(entry["label"])
            measured_entry["x"] = x_val
            measured_entry["y"] = y_val
            measured_entry["overlay_match_index"] = int(
                entry.get("overlay_match_index", len(measured_from_display))
            )
            if normalized_hkl is not None:
                measured_entry["hkl"] = normalized_hkl
            measured_from_display.append(measured_entry)

        _log_line(f"Geometry fit started: {stamp}")
        _log_line()
        _log_section(
            "Auto-match requested configuration:",
            [f"{key}={value}" for key, value in sorted(auto_match_cfg.items())] or ["<defaults>"],
        )
        _log_section(
            "Auto-match effective configuration:",
            [f"{key}={value}" for key, value in sorted(effective_auto_match_cfg.items())]
            or ["<defaults>"],
        )
        _log_section(
            "Auto-match attempts:",
            [
                (
                    f"{entry.get('label', 'attempt')}: "
                    f"matches={int(float(entry.get('matches', 0.0)))} "
                    f"radius={float(entry.get('search_radius_px', np.nan)):.2f}px "
                    f"min_prom={float(entry.get('min_match_prominence_sigma', np.nan)):.2f}σ "
                    f"cands={int(float(entry.get('max_candidate_peaks', 0.0)))} "
                    f"amb_margin={float(entry.get('ambiguity_margin_px', np.nan)):.2f}px "
                    f"amb_ratio={float(entry.get('ambiguity_ratio_min', np.nan)):.2f} "
                    f"qualified={int(float(entry.get('qualified_summit_count', 0.0)))} "
                    f"claimed={int(float(entry.get('claimed_summit_count', 0.0)))} "
                    f"pre_clip={int(float(entry.get('matched_pre_clip_count', 0.0)))} "
                    f"clipped={int(float(entry.get('clipped_count', 0.0)))}"
                )
                for entry in auto_match_attempts
            ]
            or ["<none>"],
        )
        _log_section(
            "Auto-match summary:",
            [
                f"simulated_peaks={int(match_stats.get('simulated_count', len(simulated_peaks)))}",
                f"candidate_summits={int(match_stats.get('candidate_count', 0))}",
                f"qualified_summits={int(match_stats.get('qualified_summit_count', 0))}",
                f"within_radius_summits={int(match_stats.get('within_radius_count', 0))}",
                f"unambiguous_summits={int(match_stats.get('unambiguous_count', 0))}",
                f"ownership_filtered_summits={int(match_stats.get('ownership_filtered_count', 0))}",
                f"claimed_summits={int(match_stats.get('claimed_summit_count', 0))}",
                f"conflicted_matches={int(match_stats.get('conflicted_match_count', 0))}",
                f"matched_pre_clip={int(match_stats.get('matched_pre_clip_count', 0))}",
                f"clipped_matches={int(match_stats.get('clipped_count', 0))}",
                f"matched_peaks={len(matched_pairs)}",
                f"excluded_live_preview_peaks={int(match_stats.get('excluded_count', 0))}",
                f"excluded_q_group_peaks={int(excluded_q_peaks)}",
                f"collapsed_degenerate_peaks={int(collapsed_deg_fit)}",
                f"excluded_q_groups={_geometry_q_group_excluded_count()}",
                f"available_q_groups={int(q_group_total)}",
                f"prominence_sigma_est={float(match_stats.get('sigma_est', np.nan)):.6f}",
                f"mean_walk_steps={float(match_stats.get('mean_walk_steps', np.nan)):.3f}",
                f"mean_net_ascent_sigma={float(match_stats.get('mean_net_ascent_sigma', np.nan)):.3f}",
                f"distance_clip_limit_px={float(match_stats.get('distance_clip_limit_px', np.nan)):.3f}",
                f"mean_match_distance_px={float(match_stats.get('mean_match_distance_px', np.nan)):.3f}",
                f"p90_match_distance_px={float(match_stats.get('p90_match_distance_px', np.nan)):.3f}",
                f"median_match_confidence={float(match_stats.get('median_match_confidence', np.nan)):.3f}",
            ],
        )
        _log_section(
            "Auto-matched pairs (display frame):",
            [
                (
                    f"HKL=({entry['hkl'][0]},{entry['hkl'][1]},{entry['hkl'][2]}) "
                    f"sim=({entry['sim_x']:.3f},{entry['sim_y']:.3f}) "
                    f"meas=({entry['x']:.3f},{entry['y']:.3f}) "
                    f"Ibg={float(entry.get('background_intensity', np.nan)):.3f} "
                    f"d={entry['distance_px']:.3f}px "
                    f"prom={entry['prominence_sigma']:.2f}σ "
                    f"conf={entry['confidence']:.3f}"
                )
                for entry in matched_pairs
            ],
        )

        measured_native = _unrotate_display_peaks(
            measured_from_display,
            display_background.shape,
            k=DISPLAY_ROTATE_K,
        )

        sim_orientation_points: list[tuple[float, float]] = []
        meas_orientation_points: list[tuple[float, float]] = []
        for pair_entry, measured_entry in zip(matched_pairs, measured_native):
            if not isinstance(measured_entry, dict):
                continue
            try:
                sx = float(pair_entry["sim_x"])
                sy = float(pair_entry["sim_y"])
                mx = float(measured_entry["x"])
                my = float(measured_entry["y"])
            except Exception:
                continue
            if not (np.isfinite(sx) and np.isfinite(sy) and np.isfinite(mx) and np.isfinite(my)):
                continue
            sim_orientation_points.append((sx, sy))
            meas_orientation_points.append((mx, my))

        orientation_choice, orientation_diag = _select_fit_orientation(
            sim_orientation_points,
            meas_orientation_points,
            tuple(int(v) for v in native_background.shape[:2]),
            cfg=orientation_cfg,
        )
        _cmd_line(
            "orientation: "
            f"chosen={orientation_choice.get('label', 'identity')} "
            f"identity_rms={float(orientation_diag.get('identity_rms_px', np.nan)):.4f}px "
            f"best_rms={float(orientation_diag.get('best_rms_px', np.nan)):.4f}px "
            f"reason={orientation_diag.get('reason', 'n/a')}"
        )
        _log_section(
            "Orientation diagnostics:",
            [
                f"pairs={orientation_diag.get('pairs', 0)}",
                f"enabled={orientation_diag.get('enabled', True)}",
                f"display_unrotate_k={DISPLAY_ROTATE_K}",
                f"sim_display_k={SIM_DISPLAY_ROTATE_K}",
                f"identity_rms_px={float(orientation_diag.get('identity_rms_px', np.nan)):.4f}",
                f"best_label={orientation_diag.get('best_label', 'identity')}",
                f"best_rms_px={float(orientation_diag.get('best_rms_px', np.nan)):.4f}",
                f"improvement_px={float(orientation_diag.get('improvement_px', np.nan)):.4f}",
                f"chosen={orientation_choice.get('label', 'identity')}",
                f"reason={orientation_diag.get('reason', 'n/a')}",
            ],
        )

        measured_for_fit = _apply_orientation_to_entries(
            measured_native,
            native_background.shape,
            indexing_mode=orientation_choice["indexing_mode"],
            k=orientation_choice["k"],
            flip_x=orientation_choice["flip_x"],
            flip_y=orientation_choice["flip_y"],
            flip_order=orientation_choice["flip_order"],
        )
        experimental_image_for_fit = _orient_image_for_fit(
            backend_background,
            indexing_mode=orientation_choice["indexing_mode"],
            k=orientation_choice["k"],
            flip_x=orientation_choice["flip_x"],
            flip_y=orientation_choice["flip_y"],
            flip_order=orientation_choice["flip_order"],
        )
        _log_section(
            "Fixed peak frame transforms:",
            _log_fixed_peak_frames(
                measured_from_display,
                measured_native,
                measured_for_fit,
                center_pair=(
                    float(params.get("center_x", params["center"][0])),
                    float(params.get("center_y", params["center"][1])),
                ),
            ),
        )

        def _pairs_to_measured_display_entries(
            pairs: Sequence[dict[str, object]] | None,
        ) -> list[dict[str, object]]:
            measured_entries: list[dict[str, object]] = []
            for entry in pairs or []:
                try:
                    x_val = float(entry["x"])
                    y_val = float(entry["y"])
                except Exception:
                    continue
                if not (np.isfinite(x_val) and np.isfinite(y_val)):
                    continue
                normalized_hkl = _normalize_hkl_key(entry.get("hkl", entry.get("label")))
                measured_entry = dict(entry)
                measured_entry["label"] = str(entry["label"])
                measured_entry["x"] = x_val
                measured_entry["y"] = y_val
                measured_entry["overlay_match_index"] = int(
                    entry.get("overlay_match_index", len(measured_entries))
                )
                if normalized_hkl is not None:
                    measured_entry["hkl"] = normalized_hkl
                measured_entries.append(measured_entry)
            return measured_entries

        def _build_joint_background_dataset_specs(
            base_fit_params: dict[str, object],
        ) -> tuple[list[dict[str, object]], list[str]]:
            dataset_specs_local: list[dict[str, object]] = []
            dataset_summary_lines: list[str] = []
            if not joint_background_mode:
                return dataset_specs_local, dataset_summary_lines

            current_theta_base = float(
                background_theta_values[int(background_runtime_state.current_background_index)]
            )
            dataset_specs_local.append(
                {
                    "dataset_index": int(background_runtime_state.current_background_index),
                    "label": Path(
                        str(
                            background_runtime_state.osc_files[
                                background_runtime_state.current_background_index
                            ]
                        )
                    ).name,
                    "theta_initial": float(current_theta_base),
                    "measured_peaks": list(measured_for_fit),
                    "experimental_image": experimental_image_for_fit,
                }
            )
            dataset_summary_lines.append(
                "current[{idx}] {name}: theta_i={theta_base:.6f} theta={theta_eff:.6f} "
                "matches={matches} excluded={excluded}".format(
                    idx=int(background_runtime_state.current_background_index),
                    name=Path(
                        str(
                            background_runtime_state.osc_files[
                                background_runtime_state.current_background_index
                            ]
                        )
                    ).name,
                    theta_base=float(current_theta_base),
                    theta_eff=float(
                        current_theta_base + float(base_fit_params.get("theta_offset", 0.0))
                    ),
                    matches=len(matched_pairs),
                    excluded=int(match_stats.get("excluded_count", excluded_preview_count)),
                )
            )

            for bg_idx in selected_background_indices:
                if int(bg_idx) == int(background_runtime_state.current_background_index):
                    continue
                theta_base = float(background_theta_values[int(bg_idx)])

                native_bg_i, display_bg_i = _load_background_image_by_index(int(bg_idx))
                backend_bg_i = _apply_background_backend_orientation(native_bg_i)
                if backend_bg_i is None:
                    backend_bg_i = native_bg_i

                params_i = dict(base_fit_params)
                params_i["theta_initial"] = float(
                    theta_base + float(base_fit_params.get("theta_offset", 0.0))
                )

                simulated_i = []
                if not simulated_i:
                    raise RuntimeError(
                        f"background {bg_idx + 1} ({Path(str(background_runtime_state.osc_files[bg_idx])).name}) has no simulated peaks"
                    )
                simulated_i, excluded_q_i, q_group_total_i = _filter_geometry_fit_simulated_peaks(
                    simulated_i
                )
                if not simulated_i:
                    raise RuntimeError(
                        f"background {bg_idx + 1} ({Path(str(background_runtime_state.osc_files[bg_idx])).name}) has no selected Qr/Qz groups"
                    )
                simulated_i, collapsed_deg_i = _collapse_geometry_fit_simulated_peaks(
                    simulated_i,
                    merge_radius_px=float(
                        fit_auto_match_cfg.get(
                            "degenerate_merge_radius_px",
                            min(
                                6.0,
                                0.33 * float(fit_auto_match_cfg.get("search_radius_px", 24.0)),
                            ),
                        )
                    ),
                )
                if not simulated_i:
                    raise RuntimeError(
                        f"background {bg_idx + 1} ({Path(str(background_runtime_state.osc_files[bg_idx])).name}) has no seeds after Qr/Qz collapse"
                    )

                matched_i, stats_i, _, _ = _auto_match_background_peaks_with_relaxation(
                    simulated_i,
                    display_bg_i,
                    fit_auto_match_cfg,
                    min_matches=min_matches,
                )
                matched_i, stats_i, excluded_i = _apply_live_preview_match_exclusions(
                    matched_i,
                    stats_i,
                )
                matched_i = _attach_overlay_match_indices(matched_i)

                if len(matched_i) < min_matches:
                    raise RuntimeError(
                        f"background {bg_idx + 1} ({Path(str(background_runtime_state.osc_files[bg_idx])).name}) matched "
                        f"{len(matched_i)}/{int(stats_i.get('simulated_count', len(simulated_i)))} peaks "
                        f"(need {min_matches})"
                    )

                mean_i = float(stats_i.get("mean_match_distance_px", np.nan))
                p90_i = float(stats_i.get("p90_match_distance_px", np.nan))
                quality_fail_i = (
                    np.isfinite(max_auto_p90) and np.isfinite(p90_i) and p90_i > max_auto_p90
                ) or (np.isfinite(max_auto_mean) and np.isfinite(mean_i) and mean_i > max_auto_mean)
                if quality_fail_i:
                    raise RuntimeError(
                        f"background {bg_idx + 1} ({Path(str(background_runtime_state.osc_files[bg_idx])).name}) failed the quality gate "
                        f"(mean={mean_i:.2f}px, p90={p90_i:.2f}px)"
                    )

                measured_i_display = _pairs_to_measured_display_entries(matched_i)
                measured_i_native = _unrotate_display_peaks(
                    measured_i_display,
                    display_bg_i.shape,
                    k=DISPLAY_ROTATE_K,
                )
                measured_i_for_fit = _apply_orientation_to_entries(
                    measured_i_native,
                    native_bg_i.shape,
                    indexing_mode=orientation_choice["indexing_mode"],
                    k=orientation_choice["k"],
                    flip_x=orientation_choice["flip_x"],
                    flip_y=orientation_choice["flip_y"],
                    flip_order=orientation_choice["flip_order"],
                )
                experimental_i_for_fit = _orient_image_for_fit(
                    backend_bg_i,
                    indexing_mode=orientation_choice["indexing_mode"],
                    k=orientation_choice["k"],
                    flip_x=orientation_choice["flip_x"],
                    flip_y=orientation_choice["flip_y"],
                    flip_order=orientation_choice["flip_order"],
                )
                dataset_specs_local.append(
                    {
                        "dataset_index": int(bg_idx),
                        "label": Path(str(background_runtime_state.osc_files[bg_idx])).name,
                        "theta_initial": float(theta_base),
                        "measured_peaks": measured_i_for_fit,
                        "experimental_image": experimental_i_for_fit,
                    }
                )
                dataset_summary_lines.append(
                    "bg[{idx}] {name}: theta_i={theta_base:.6f} theta={theta_eff:.6f} "
                    "matches={matches} mean={mean:.3f}px p90={p90:.3f}px "
                    "excluded={excluded} q_groups_on={q_on}/{q_total} collapsed={collapsed} excluded_q={excluded_q}".format(
                        idx=int(bg_idx),
                        name=Path(str(background_runtime_state.osc_files[bg_idx])).name,
                        theta_base=float(theta_base),
                        theta_eff=float(
                            theta_base + float(base_fit_params.get("theta_offset", 0.0))
                        ),
                        matches=len(matched_i),
                        mean=float(mean_i),
                        p90=float(p90_i),
                        excluded=int(excluded_i),
                        q_on=max(0, int(q_group_total_i) - _geometry_q_group_excluded_count()),
                        q_total=int(q_group_total_i),
                        collapsed=int(collapsed_deg_i),
                        excluded_q=int(excluded_q_i),
                    )
                )

            return dataset_specs_local, dataset_summary_lines

        _log_section(
            "Fitting variables (start values):",
            [
                (
                    f"{name}: <missing>"
                    if params.get(name) is None
                    else f"{name}: {float(params.get(name)):.6f}"
                )
                for name in var_names
            ],
        )

        fit_iterations = int(auto_match_cfg.get("fit_iterations", 3))
        fit_iterations = max(1, min(8, fit_iterations))
        rematch_between_iterations = bool(auto_match_cfg.get("rematch_between_iterations", False))
        if joint_background_mode and rematch_between_iterations:
            _cmd_line(
                "auto-match: multi-background rematch is not yet supported; "
                "using the initial joint matches only"
            )
            rematch_between_iterations = False
        if not rematch_between_iterations and fit_iterations > 1:
            _cmd_line(
                "auto-match: background peaks fixed after initial selection; "
                "skipping iterative rematch"
            )
            fit_iterations = 1

        current_dataset_specs, joint_dataset_summary_lines = _build_joint_background_dataset_specs(
            params
        )
        if joint_dataset_summary_lines:
            _log_section(
                "Joint background datasets:",
                joint_dataset_summary_lines,
            )

        result = None
        iteration_logs: list[str] = []
        current_fit_params = dict(params)
        current_matched_pairs = list(matched_pairs)
        current_measured_for_fit = list(measured_for_fit)

        def _set_fit_param(target: dict[str, object], name: str, value: float) -> None:
            val = float(value)
            target[name] = val
            if name == "center_x" or name == "center_y":
                center_pair = list(target.get("center", [center_x_var.get(), center_y_var.get()]))
                if len(center_pair) < 2:
                    center_pair = [center_x_var.get(), center_y_var.get()]
                if name == "center_x":
                    center_pair[0] = val
                else:
                    center_pair[1] = val
                target["center"] = [float(center_pair[0]), float(center_pair[1])]
                target["center_x"] = float(center_pair[0])
                target["center_y"] = float(center_pair[1])

        for iter_idx in range(fit_iterations):
            _cmd_line(
                f"iter {iter_idx + 1}/{fit_iterations}: optimize start "
                f"(matches={len(current_matched_pairs)})"
            )
            progress_label_geometry.config(
                text=(
                    f"Geometry fit iteration {iter_idx + 1}/{fit_iterations} "
                    f"(matches={len(current_matched_pairs)})…"
                )
            )
            root.update_idletasks()
            candidate_param_names = _current_geometry_fit_candidate_param_names()
            geometry_runtime_cfg = _build_geometry_fit_runtime_config(
                geometry_refine_cfg,
                {name: current_fit_params.get(name) for name in candidate_param_names},
                _current_geometry_fit_constraint_state(candidate_param_names),
                _current_geometry_fit_parameter_domains(candidate_param_names),
                candidate_param_names=candidate_param_names,
            )

            result = fit_geometry_parameters(
                miller,
                intensities,
                image_size,
                current_fit_params,
                current_measured_for_fit,
                var_names,
                pixel_tol=float("inf"),
                experimental_image=experimental_image_for_fit,
                dataset_specs=current_dataset_specs if joint_background_mode else None,
                refinement_config=geometry_runtime_cfg,
                candidate_param_names=candidate_param_names,
            )

            if getattr(result, "x", None) is None or len(result.x) != len(var_names):
                _cmd_line(
                    f"iter {iter_idx + 1}/{fit_iterations}: optimizer returned no parameter vector"
                )
                iteration_logs.append(
                    f"iter={iter_idx + 1}: optimizer returned no parameter vector"
                )
                break

            for name, val in zip(var_names, result.x):
                _set_fit_param(current_fit_params, name, float(val))

            iter_rms = (
                float(getattr(result, "rms_px"))
                if np.isfinite(float(getattr(result, "rms_px", np.nan)))
                else (
                    float(np.sqrt(np.mean(result.fun**2)))
                    if getattr(result, "fun", None) is not None and result.fun.size
                    else float("nan")
                )
            )
            iteration_logs.append(
                (
                    f"iter={iter_idx + 1}: matches={len(current_matched_pairs)}, "
                    f"cost={float(getattr(result, 'cost', np.nan)):.6f}, "
                    f"RMS={iter_rms:.4f}px, "
                    f"robust_cost={float(getattr(result, 'robust_cost', np.nan)):.6f}, "
                    f"orientation={orientation_choice.get('label', 'identity')}"
                )
            )
            _cmd_line(
                f"iter {iter_idx + 1}/{fit_iterations}: "
                f"cost={float(getattr(result, 'cost', np.nan)):.6f} "
                f"rms={iter_rms:.4f}px "
                f"robust_cost={float(getattr(result, 'robust_cost', np.nan)):.6f}"
            )

            if iter_idx + 1 >= fit_iterations:
                break

            try:
                _cmd_line(f"iter {iter_idx + 1}/{fit_iterations}: rematch start")
                sim_iter = []
                if not sim_iter:
                    _cmd_line(
                        f"iter {iter_idx + 1}/{fit_iterations}: rematch skipped (no simulated peaks)"
                    )
                    iteration_logs.append(
                        f"iter={iter_idx + 1}: rematch skipped (no simulated peaks)"
                    )
                    break
                sim_iter, excluded_q_iter, q_group_total_iter = (
                    _filter_geometry_fit_simulated_peaks(sim_iter)
                )
                if not sim_iter:
                    _cmd_line(
                        f"iter {iter_idx + 1}/{fit_iterations}: rematch skipped (no Qr/Qz groups selected)"
                    )
                    iteration_logs.append(
                        f"iter={iter_idx + 1}: rematch skipped (no Qr/Qz groups selected)"
                    )
                    break
                sim_iter, collapsed_deg_iter = _collapse_geometry_fit_simulated_peaks(
                    sim_iter,
                    merge_radius_px=float(
                        fit_auto_match_cfg.get(
                            "degenerate_merge_radius_px",
                            min(
                                6.0,
                                0.33 * float(fit_auto_match_cfg.get("search_radius_px", 24.0)),
                            ),
                        )
                    ),
                )
                if not sim_iter:
                    _cmd_line(
                        f"iter {iter_idx + 1}/{fit_iterations}: rematch skipped (no seeds remain after Qr/Qz collapse)"
                    )
                    iteration_logs.append(
                        f"iter={iter_idx + 1}: rematch skipped (no seeds remain after Qr/Qz collapse)"
                    )
                    break
                iteration_logs.append(
                    f"iter={iter_idx + 1}: rematch seeds={len(sim_iter)} (collapsed={collapsed_deg_iter})"
                )

                matched_iter, stats_iter, _, rematch_attempts = (
                    _auto_match_background_peaks_with_relaxation(
                        sim_iter,
                        display_background,
                        fit_auto_match_cfg,
                        min_matches=min_matches,
                    )
                )
                matched_iter, stats_iter, excluded_iter_count = (
                    _apply_live_preview_match_exclusions(
                        matched_iter,
                        stats_iter,
                    )
                )
                matched_iter = _attach_overlay_match_indices(matched_iter)
                if len(matched_iter) < min_matches:
                    best_attempt = (
                        max(
                            rematch_attempts,
                            key=lambda entry: float(entry.get("matches", 0.0)),
                        )
                        if rematch_attempts
                        else None
                    )
                    best_attempt_note = ""
                    if best_attempt is not None:
                        best_attempt_note = (
                            "; best_attempt="
                            f"{best_attempt.get('label', 'n/a')}"
                            f"({int(float(best_attempt.get('matches', 0.0)))} matches, "
                            f"qualified={int(float(best_attempt.get('qualified_summit_count', 0.0)))}, "
                            f"claimed={int(float(best_attempt.get('claimed_summit_count', 0.0)))})"
                        )
                    iteration_logs.append(
                        (
                            f"iter={iter_idx + 1}: rematch skipped "
                            f"({len(matched_iter)} < min_matches={min_matches}"
                            + (
                                f", q_groups_off={_geometry_q_group_excluded_count()}/{q_group_total_iter}"
                                if q_group_total_iter > 0
                                else ""
                            )
                            + (
                                f", excluded={excluded_iter_count}"
                                if excluded_iter_count > 0
                                else ""
                            )
                            + f"{best_attempt_note})"
                        )
                    )
                    _cmd_line(
                        f"iter {iter_idx + 1}/{fit_iterations}: rematch skipped "
                        f"({len(matched_iter)} < {min_matches})"
                    )
                    break

                measured_iter_display = []
                for entry in matched_iter:
                    try:
                        x_val = float(entry["x"])
                        y_val = float(entry["y"])
                    except Exception:
                        continue
                    if not (np.isfinite(x_val) and np.isfinite(y_val)):
                        continue
                    normalized_hkl = _normalize_hkl_key(entry.get("hkl", entry.get("label")))
                    measured_entry = dict(entry)
                    measured_entry["label"] = str(entry["label"])
                    measured_entry["x"] = x_val
                    measured_entry["y"] = y_val
                    measured_entry["overlay_match_index"] = int(
                        entry.get("overlay_match_index", len(measured_iter_display))
                    )
                    if normalized_hkl is not None:
                        measured_entry["hkl"] = normalized_hkl
                    measured_iter_display.append(measured_entry)
                measured_iter_native = _unrotate_display_peaks(
                    measured_iter_display,
                    display_background.shape,
                    k=DISPLAY_ROTATE_K,
                )
                sim_iter_points: list[tuple[float, float]] = []
                meas_iter_points: list[tuple[float, float]] = []
                for pair_entry, measured_entry in zip(matched_iter, measured_iter_native):
                    if not isinstance(measured_entry, dict):
                        continue
                    try:
                        sx = float(pair_entry["sim_x"])
                        sy = float(pair_entry["sim_y"])
                        mx = float(measured_entry["x"])
                        my = float(measured_entry["y"])
                    except Exception:
                        continue
                    if not (
                        np.isfinite(sx) and np.isfinite(sy) and np.isfinite(mx) and np.isfinite(my)
                    ):
                        continue
                    sim_iter_points.append((sx, sy))
                    meas_iter_points.append((mx, my))

                orientation_choice, orientation_diag_iter = _select_fit_orientation(
                    sim_iter_points,
                    meas_iter_points,
                    tuple(int(v) for v in native_background.shape[:2]),
                    cfg=orientation_cfg,
                )
                _cmd_line(
                    f"iter {iter_idx + 1}/{fit_iterations}: rematch orientation="
                    f"{orientation_choice.get('label', 'identity')} "
                    f"best_rms={float(orientation_diag_iter.get('best_rms_px', np.nan)):.4f}px"
                )
                iteration_logs.append(
                    (
                        f"iter={iter_idx + 1}: rematch orientation={orientation_choice.get('label', 'identity')} "
                        f"(identity_rms={float(orientation_diag_iter.get('identity_rms_px', np.nan)):.4f}px, "
                        f"best_rms={float(orientation_diag_iter.get('best_rms_px', np.nan)):.4f}px, "
                        f"reason={orientation_diag_iter.get('reason', 'n/a')})"
                    )
                )
                measured_iter_for_fit = _apply_orientation_to_entries(
                    measured_iter_native,
                    native_background.shape,
                    indexing_mode=orientation_choice["indexing_mode"],
                    k=orientation_choice["k"],
                    flip_x=orientation_choice["flip_x"],
                    flip_y=orientation_choice["flip_y"],
                    flip_order=orientation_choice["flip_order"],
                )
                experimental_image_for_fit = _orient_image_for_fit(
                    backend_background,
                    indexing_mode=orientation_choice["indexing_mode"],
                    k=orientation_choice["k"],
                    flip_x=orientation_choice["flip_x"],
                    flip_y=orientation_choice["flip_y"],
                    flip_order=orientation_choice["flip_order"],
                )

                current_matched_pairs = matched_iter
                current_measured_for_fit = measured_iter_for_fit
                match_stats = stats_iter
                progress_label_geometry.config(
                    text=(
                        f"Geometry fit iteration {iter_idx + 1}/{fit_iterations} "
                        f"rematch complete (matches={len(current_matched_pairs)}"
                        + (
                            f", q_groups_off={_geometry_q_group_excluded_count()}/{q_group_total_iter}"
                            if q_group_total_iter > 0
                            else ""
                        )
                        + (f", excluded={excluded_iter_count}" if excluded_iter_count > 0 else "")
                        + ")."
                    )
                )
                root.update_idletasks()
                _cmd_line(
                    f"iter {iter_idx + 1}/{fit_iterations}: rematch complete "
                    f"(matches={len(current_matched_pairs)})"
                )
            except Exception as rematch_exc:
                _cmd_line(f"iter {iter_idx + 1}/{fit_iterations}: rematch failed ({rematch_exc})")
                iteration_logs.append(f"iter={iter_idx + 1}: rematch failed ({rematch_exc})")
                break

        if result is None:
            raise RuntimeError("Geometry optimizer did not run.")

        matched_pairs = current_matched_pairs
        measured_for_fit = current_measured_for_fit
        excluded_preview_count = int(match_stats.get("excluded_count", excluded_preview_count))

        _log_section(
            "Optimizer diagnostics:",
            [
                f"iterations={len(iteration_logs)}",
                *iteration_logs,
                f"final_orientation={orientation_choice.get('label', 'identity')}",
                f"success={getattr(result, 'success', False)}",
                f"status={getattr(result, 'status', '')}",
                f"message={(getattr(result, 'message', '') or '').strip()}",
                f"nfev={getattr(result, 'nfev', '<unknown>')}",
                f"cost={float(getattr(result, 'cost', np.nan)):.6f}",
                f"robust_cost={float(getattr(result, 'robust_cost', np.nan)):.6f}",
                f"solver_loss={getattr(result, 'solver_loss', '<unknown>')}",
                f"solver_f_scale={float(getattr(result, 'solver_f_scale', np.nan)):.6f}",
                f"optimality={float(getattr(result, 'optimality', np.nan)):.6f}",
                f"active_mask={list(getattr(result, 'active_mask', []))}",
                *[
                    "restart[{idx}] cost={cost:.6f} success={success} msg={msg}".format(
                        idx=int(entry.get("restart", -1)),
                        cost=float(entry.get("cost", np.nan)),
                        success=bool(entry.get("success", False)),
                        msg=str(entry.get("message", "")).strip(),
                    )
                    for entry in (getattr(result, "restart_history", []) or [])
                ],
            ],
        )

        for name in var_names:
            val = float(current_fit_params.get(name, params.get(name, 0.0)))
            if name == "zb":
                zb_var.set(val)
            elif name == "zs":
                zs_var.set(val)
            elif name == "theta_initial":
                theta_initial_var.set(val)
            elif name == "theta_offset":
                if geometry_theta_offset_var is not None:
                    geometry_theta_offset_var.set(f"{val:.6g}")
            elif name == "psi_z":
                psi_z_var.set(val)
            elif name == "chi":
                chi_var.set(val)
            elif name == "cor_angle":
                cor_angle_var.set(val)
            elif name == "gamma":
                gamma_var.set(val)
            elif name == "Gamma":
                Gamma_var.set(val)
            elif name == "corto_detector":
                corto_detector_var.set(val)
            elif name == "a":
                a_var.set(val)
            elif name == "c":
                c_var.set(val)
            elif name == "center_x":
                center_x_var.set(val)
            elif name == "center_y":
                center_y_var.set(val)

        if joint_background_mode and not preserve_live_theta:
            theta_initial_var.set(
                _background_theta_base_for_index(
                    background_runtime_state.current_background_index,
                    strict_count=False,
                )
            )
            _refresh_background_status()

        simulation_runtime_state.profile_cache = dict(simulation_runtime_state.profile_cache)
        simulation_runtime_state.profile_cache.update(mosaic_params)
        simulation_runtime_state.profile_cache.update(
            {
                "theta_initial": theta_initial_var.get(),
                "theta_offset": _current_geometry_theta_offset(strict=False),
                "cor_angle": cor_angle_var.get(),
                "chi": chi_var.get(),
                "zs": zs_var.get(),
                "zb": zb_var.get(),
                "gamma": gamma_var.get(),
                "Gamma": Gamma_var.get(),
                "corto_detector": corto_detector_var.get(),
                "a": a_var.get(),
                "c": c_var.get(),
                "center_x": center_x_var.get(),
                "center_y": center_y_var.get(),
            }
        )

        gui_controllers.request_geometry_preview_skip_once(geometry_preview_state)
        _invalidate_simulation_cache()
        schedule_update()

        rms = (
            float(getattr(result, "rms_px"))
            if np.isfinite(float(getattr(result, "rms_px", np.nan)))
            else (
                np.sqrt(np.mean(result.fun**2))
                if getattr(result, "fun", None) is not None and result.fun.size
                else 0.0
            )
        )
        _log_section(
            "Optimization result:",
            [f"{name} = {val:.6f}" for name, val in zip(var_names, result.x)]
            + [f"RMS residual = {rms:.6f} px"],
        )

        fitted_params = dict(params)
        fitted_params.update(
            {
                "zb": zb_var.get(),
                "zs": zs_var.get(),
                "theta_initial": theta_initial_var.get(),
                "theta_offset": _current_geometry_theta_offset(strict=False),
                "chi": chi_var.get(),
                "cor_angle": cor_angle_var.get(),
                "psi_z": psi_z_var.get(),
                "gamma": gamma_var.get(),
                "Gamma": Gamma_var.get(),
                "corto_detector": corto_detector_var.get(),
                "a": a_var.get(),
                "c": c_var.get(),
                "center": [center_x_var.get(), center_y_var.get()],
                "center_x": center_x_var.get(),
                "center_y": center_y_var.get(),
            }
        )
        point_match_summary = getattr(result, "point_match_summary", None)
        if isinstance(point_match_summary, dict):
            _log_section(
                "Final point-match summary:",
                [f"{key}={value}" for key, value in sorted(point_match_summary.items())]
                or ["<none>"],
            )
        _log_section(
            "Final point-match diagnostics:",
            _log_point_match_diagnostics(getattr(result, "point_match_diagnostics", None)),
        )
        _log_section(
            "Fixed peak frame transforms after fit:",
            _log_fixed_peak_frames(
                measured_from_display,
                measured_native,
                measured_for_fit,
                center_pair=(
                    float(fitted_params.get("center_x", fitted_params["center"][0])),
                    float(fitted_params.get("center_y", fitted_params["center"][1])),
                ),
            ),
        )

        (
            _,
            sim_coords,
            meas_coords,
            sim_millers,
            meas_millers,
        ) = simulate_and_compare_hkl(
            miller,
            intensities,
            image_size,
            fitted_params,
            measured_for_fit,
            pixel_tol=float("inf"),
            prefer_python_runner=True,
        )

        (
            agg_sim_coords,
            agg_meas_coords,
            agg_millers,
        ) = _aggregate_match_centers(
            sim_coords,
            meas_coords,
            sim_millers,
            meas_millers,
        )

        pixel_offsets: list[tuple[tuple[int, int, int], float, float, float]] = []
        max_display_markers = int(auto_match_cfg.get("max_display_markers", 120))
        max_display_markers = max(1, max_display_markers)

        for hkl_key, sim_center, meas_center in zip(agg_millers, agg_sim_coords, agg_meas_coords):
            dx = sim_center[0] - meas_center[0]
            dy = sim_center[1] - meas_center[1]
            dist = math.hypot(dx, dy)
            pixel_offsets.append((hkl_key, dx, dy, dist))

        native_overlay_shape = tuple(int(v) for v in native_background.shape[:2])
        overlay_point_match_diagnostics = getattr(result, "point_match_diagnostics", None)
        if joint_background_mode and isinstance(overlay_point_match_diagnostics, list):
            overlay_point_match_diagnostics = [
                dict(entry)
                for entry in overlay_point_match_diagnostics
                if isinstance(entry, dict)
                and int(entry.get("dataset_index", -1))
                == int(background_runtime_state.current_background_index)
            ]
        overlay_records = build_geometry_fit_overlay_records(
            initial_auto_pairs_display,
            overlay_point_match_diagnostics,
            native_shape=native_overlay_shape,
            orientation_choice=orientation_choice,
            sim_display_rotate_k=SIM_DISPLAY_ROTATE_K,
            background_display_rotate_k=DISPLAY_ROTATE_K,
        )
        frame_diag, frame_warning = _geometry_overlay_frame_diagnostics(
            overlay_records,
        )
        _log_section(
            "Overlay frame diagnostics:",
            [
                "transform_rule=sim:direct_native_to_display; bg:inverse_orientation_then_display_rotation",
                f"overlay_records={len(overlay_records)}",
                f"paired_records={int(frame_diag.get('paired_records', 0))}",
                f"sim_display_med_px={float(frame_diag.get('sim_display_med_px', np.nan)):.3f}",
                f"bg_display_med_px={float(frame_diag.get('bg_display_med_px', np.nan)):.3f}",
                f"sim_display_p90_px={float(frame_diag.get('sim_display_p90_px', np.nan)):.3f}",
                f"bg_display_p90_px={float(frame_diag.get('bg_display_p90_px', np.nan)):.3f}",
            ],
        )
        if frame_warning:
            _log_line(f"WARNING: {frame_warning}")
            _log_line()

        if overlay_records:
            _draw_geometry_fit_overlay(
                overlay_records,
                max_display_markers=max_display_markers,
            )
        else:
            _draw_initial_geometry_pairs_overlay(
                initial_auto_pairs_display,
                max_display_markers=max_display_markers,
            )

        _log_section(
            "Pixel offsets (native frame):",
            [
                f"HKL={hkl}: dx={dx:.4f}, dy={dy:.4f}, |Δ|={dist:.4f} px"
                for hkl, dx, dy, dist in pixel_offsets
            ]
            or ["No matched peaks"],
        )

        export_recs = []
        for hkl, (x, y), (_, _, _, dist) in zip(agg_millers, agg_sim_coords, pixel_offsets):
            export_recs.append(
                {
                    "source": "sim",
                    "hkl": tuple(int(v) for v in hkl),
                    "x": int(x),
                    "y": int(y),
                    "dist_px": float(dist),
                }
            )
        for hkl, (x, y), (_, _, _, dist) in zip(agg_millers, agg_meas_coords, pixel_offsets):
            export_recs.append(
                {
                    "source": "meas",
                    "hkl": tuple(int(v) for v in hkl),
                    "x": int(x),
                    "y": int(y),
                    "dist_px": float(dist),
                }
            )

        save_path = get_dir("downloads") / f"matched_peaks_{stamp}.npy"
        np.save(save_path, np.array(export_recs, dtype=object), allow_pickle=True)

        _log_section(
            "Fit summary:",
            [
                f"auto_matched_peaks={len(matched_pairs)}",
                f"overlay_records={len(overlay_records)}",
                f"excluded_live_preview_peaks={excluded_preview_count}",
                f"excluded_q_groups={_geometry_q_group_excluded_count()}",
                f"available_q_groups={int(q_group_total)}",
                f"excluded_q_group_peaks={int(excluded_q_peaks)}",
                f"auto_simulated_peaks={int(match_stats.get('simulated_count', len(simulated_peaks)))}",
                f"auto_distance_p90_px={float(match_stats.get('p90_match_distance_px', np.nan)):.3f}",
                f"auto_distance_clip_limit_px={float(match_stats.get('distance_clip_limit_px', np.nan)):.3f}",
                f"orientation={orientation_choice.get('label', 'identity')}",
                *[f"{name} = {val:.6f}" for name, val in zip(var_names, result.x)],
                f"RMS residual = {rms:.6f} px",
                f"Matched peaks saved to: {save_path}",
            ],
        )

        base_summary = (
            "Auto geometry fit complete:\n"
            + "\n".join(f"{name} = {val:.4f}" for name, val in zip(var_names, result.x))
            + f"\nRMS residual = {rms:.2f} px"
            + f"\nOrientation = {orientation_choice.get('label', 'identity')}"
        )
        overlay_hint = (
            "Overlay: blue squares=initial simulated picks, amber triangles=fixed background picks, "
            "green circles=fitted simulated peaks, dashed arrows=initial->fitted sim shifts."
        )
        frame_warning_line = f"{frame_warning}\n" if frame_warning else ""
        progress_label_geometry.config(
            text=(
                f"{base_summary}\n"
                f"Auto-matched peaks: {len(matched_pairs)}/"
                f"{int(match_stats.get('simulated_count', len(simulated_peaks)))}"
                + (
                    f" | Qr/Qz groups on {max(0, q_group_total - _geometry_q_group_excluded_count())}/{q_group_total}"
                    if q_group_total > 0
                    else ""
                )
                + (f" (excluded {excluded_preview_count})" if excluded_preview_count > 0 else "")
                + "\n"
                + f"{overlay_hint}\n"
                + frame_warning_line
                + f"Saved {len(export_recs)} peak records → {save_path}\n"
                + f"Fit log → {log_path}"
            )
        )
        _cmd_line(
            "done: "
            f"matched={len(matched_pairs)} "
            f"orientation={orientation_choice.get('label', 'identity')} "
            f"rms={float(rms):.4f}px "
            f"matched_peaks_file={save_path} "
            f"log_file={log_path}"
        )
        return
    except Exception as exc:
        _cmd_line(f"failed: {exc}")
        _log_line(f"Geometry fit failed: {exc}")
        progress_label_geometry.config(text=f"Geometry fit failed: {exc}")
        return
    finally:
        try:
            log_file.close()
        except Exception:
            pass

    if not ensure_peak_overlay_data(force=False) or not simulation_runtime_state.peak_positions:
        progress_label_geometry.config(text="No simulated peaks available to pick.")
        return

    def _nearest_simulated_peak(col: float, row: float):
        """Return (index, distance^2) of the nearest simulated peak, or (None, inf)."""

        best_idx, best_d2 = None, float("inf")
        for idx, (px, py) in enumerate(simulation_runtime_state.peak_positions):
            if px < 0 or py < 0:
                continue
            d2 = (px - col) ** 2 + (py - row) ** 2
            if d2 < best_d2:
                best_idx, best_d2 = idx, d2
        return best_idx, best_d2

    def _peak_maximum_near(col: float, row: float, search_radius: int = 5):
        """Return the (col,row) of the brightest pixel near ``(col,row)``."""

        display_extent = None
        for artist in (background_display, image_display):
            extent = getattr(artist, "_ra_sim_source_extent", None)
            if extent is None:
                getter = getattr(artist, "get_extent", None)
                if callable(getter):
                    try:
                        extent = getter()
                    except Exception:
                        extent = None
            if extent is not None:
                display_extent = extent
                break

        return gui_manual_geometry.peak_maximum_near_in_image(
            background_runtime_state.current_background_display,
            float(col),
            float(row),
            search_radius=search_radius,
            display_extent=display_extent,
        )

    def _mark_pick(col: float, row: float, label: str, color: str, marker: str):
        (point,) = ax.plot(
            [col],
            [row],
            marker,
            color=color,
            markersize=8,
            markerfacecolor="none",
            zorder=7,
            linestyle="None",
        )
        text = ax.text(
            col,
            row,
            label,
            color=color,
            fontsize=8,
            ha="left",
            va="bottom",
            zorder=8,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.0),
        )
        geometry_runtime_state.pick_artists.extend([point, text])
        _request_overlay_canvas_redraw()

    picked_pairs = []  # list[(h,k,l), (x_real, y_real)]
    initial_manual_pairs_display: list[dict[str, object]] = []
    selection_state = {
        "expecting": "sim",
        "pending_hkl": None,
        "pending_sim_display": None,
    }
    canvas_widget = canvas.get_tk_widget()

    progress_label_geometry.config(
        text="Click a simulated peak, then the matching real peak (right click to finish)."
    )
    canvas_widget.configure(cursor="crosshair")

    click_cid = None

    def _finish_pair_collection():
        nonlocal click_cid
        if click_cid is not None:
            canvas.mpl_disconnect(click_cid)
            click_cid = None
        canvas_widget.configure(cursor="")

    def _on_geometry_pick(event):
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return

        if event.button == 3 or getattr(event, "dblclick", False):
            _finish_pair_collection()
            if not picked_pairs:
                progress_label_geometry.config(text="No peak pairs selected; fit cancelled.")
                return

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = gui_geometry_fit.build_geometry_fit_log_path(
                stamp=stamp,
                log_dir=get_dir("debug_log_dir"),
            )
            log_file = None
            if gui_geometry_fit.geometry_fit_log_files_enabled():
                log_file = log_path.open("w", encoding="utf-8")

            def _log_line(text: str = ""):
                try:
                    log_file.write(text + "\n")
                    log_file.flush()
                except Exception:
                    pass

            def _log_section(title: str, lines: list[str]):
                _log_line(title)
                for line in lines:
                    _log_line(f"  {line}")
                _log_line()

            progress_label_geometry.config(text="Running geometry fit…")
            root.update_idletasks()

            measured_from_clicks = [
                {
                    "label": f"{h},{k},{l}",
                    "x": float(x),
                    "y": float(y),
                    "overlay_match_index": int(pair_idx),
                }
                for pair_idx, ((h, k, l), (x, y)) in enumerate(picked_pairs)
            ]
            _log_line(f"Geometry fit started: {stamp}")
            _log_line()
            _log_section(
                "Picked pairs (display frame):",
                [
                    f"HKL=({h},{k},{l}) display_px=({x:.3f}, {y:.3f})"
                    for (h, k, l), (x, y) in picked_pairs
                ],
            )
            native_background = _get_current_background_native()
            backend_background = _get_current_background_backend()
            if backend_background is None:
                backend_background = native_background
            measured_native = _unrotate_display_peaks(
                measured_from_clicks,
                background_runtime_state.current_background_display.shape,
                k=DISPLAY_ROTATE_K,
            )
            picked_frames = [
                {
                    "label": entry_disp.get("label"),
                    "display": (float(entry_disp.get("x")), float(entry_disp.get("y"))),
                    "native": (float(entry_nat.get("x")), float(entry_nat.get("y"))),
                }
                for entry_disp, entry_nat in zip(measured_from_clicks, measured_native)
            ]

            _log_section(
                "Unrotated measured peaks (fit frame):",
                [
                    (
                        "label="
                        f"{entry.get('label')} fit_px=({entry.get('x'):.3f}, {entry.get('y'):.3f})"
                    )
                    for entry in measured_native
                ],
            )

            sim_orientation_points: list[tuple[float, float]] = []
            meas_orientation_points: list[tuple[float, float]] = []
            try:
                simulated_for_orientation = _simulate_hkl_peak_centers_for_fit(
                    miller_array,
                    intensity_array,
                    image_size,
                    params,
                )
                simulated_lookup = {
                    str(entry.get("label")): (
                        float(entry.get("sim_col")),
                        float(entry.get("sim_row")),
                    )
                    for entry in simulated_for_orientation
                }
            except Exception:
                simulated_lookup = {}

            for entry in measured_native:
                if not isinstance(entry, dict):
                    continue
                label = str(entry.get("label", ""))
                sim_pt = simulated_lookup.get(label)
                if sim_pt is None:
                    continue
                try:
                    mx = float(entry.get("x"))
                    my = float(entry.get("y"))
                    sx = float(sim_pt[0])
                    sy = float(sim_pt[1])
                except Exception:
                    continue
                if not (
                    np.isfinite(mx) and np.isfinite(my) and np.isfinite(sx) and np.isfinite(sy)
                ):
                    continue
                sim_orientation_points.append((sx, sy))
                meas_orientation_points.append((mx, my))

            orientation_choice, orientation_diag = _select_fit_orientation(
                sim_orientation_points,
                meas_orientation_points,
                tuple(int(v) for v in native_background.shape[:2]),
                cfg=orientation_cfg,
            )
            _log_section(
                "Orientation diagnostics:",
                [
                    f"pairs={orientation_diag.get('pairs', 0)}",
                    f"enabled={orientation_diag.get('enabled', True)}",
                    f"display_unrotate_k={DISPLAY_ROTATE_K}",
                    f"sim_display_k={SIM_DISPLAY_ROTATE_K}",
                    f"identity_rms_px={float(orientation_diag.get('identity_rms_px', np.nan)):.4f}",
                    f"best_label={orientation_diag.get('best_label', 'identity')}",
                    f"best_rms_px={float(orientation_diag.get('best_rms_px', np.nan)):.4f}",
                    f"improvement_px={float(orientation_diag.get('improvement_px', np.nan)):.4f}",
                    f"chosen={orientation_choice.get('label', 'identity')}",
                    f"reason={orientation_diag.get('reason', 'n/a')}",
                ],
            )

            try:
                measured_for_fit = _apply_orientation_to_entries(
                    measured_native,
                    native_background.shape,
                    indexing_mode=orientation_choice["indexing_mode"],
                    k=orientation_choice["k"],
                    flip_x=orientation_choice["flip_x"],
                    flip_y=orientation_choice["flip_y"],
                    flip_order=orientation_choice["flip_order"],
                )
                for frame_entry, entry_fit in zip(picked_frames, measured_for_fit):
                    if isinstance(entry_fit, dict):
                        frame_entry["fit"] = (float(entry_fit.get("x")), float(entry_fit.get("y")))
                    elif isinstance(entry_fit, (list, tuple)) and len(entry_fit) >= 5:
                        frame_entry["fit"] = (float(entry_fit[3]), float(entry_fit[4]))
                _log_section(
                    "Measured peaks used for fitting (after orientation):",
                    [
                        (
                            "label="
                            f"{entry.get('label')} fit_px=({entry.get('x'):.3f}, {entry.get('y'):.3f})"
                        )
                        for entry in measured_for_fit
                    ],
                )
                experimental_image_for_fit = _orient_image_for_fit(
                    backend_background,
                    indexing_mode=orientation_choice["indexing_mode"],
                    k=orientation_choice["k"],
                    flip_x=orientation_choice["flip_x"],
                    flip_y=orientation_choice["flip_y"],
                    flip_order=orientation_choice["flip_order"],
                )

                def _log_assignment_snapshot(title: str, param_set: dict[str, float]):
                    try:
                        mosaic = param_set["mosaic_params"]
                        wavelength_array = mosaic.get("wavelength_array")
                        if wavelength_array is None:
                            wavelength_array = mosaic.get("wavelength_i_array")

                        sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)
                        _, hit_tables, *_ = process_peaks_parallel(
                            miller,
                            intensities,
                            image_size,
                            param_set["a"],
                            param_set["c"],
                            wavelength_array,
                            sim_buffer,
                            param_set["corto_detector"],
                            param_set["gamma"],
                            param_set["Gamma"],
                            param_set["chi"],
                            param_set.get("psi", 0.0),
                            param_set.get("psi_z", 0.0),
                            param_set["zs"],
                            param_set["zb"],
                            param_set["n2"],
                            mosaic["beam_x_array"],
                            mosaic["beam_y_array"],
                            mosaic["theta_array"],
                            mosaic["phi_array"],
                            mosaic["sigma_mosaic_deg"],
                            mosaic["gamma_mosaic_deg"],
                            mosaic["eta"],
                            wavelength_array,
                            param_set["debye_x"],
                            param_set["debye_y"],
                            param_set["center"],
                            param_set["theta_initial"],
                            param_set.get("cor_angle", 0.0),
                            np.array([1.0, 0.0, 0.0]),
                            np.array([0.0, 1.0, 0.0]),
                            save_flag=0,
                            optics_mode=_current_optics_mode_flag(),
                            solve_q_steps=int(mosaic.get("solve_q_steps", DEFAULT_SOLVE_Q_STEPS)),
                            solve_q_rel_tol=float(
                                mosaic.get("solve_q_rel_tol", DEFAULT_SOLVE_Q_REL_TOL)
                            ),
                            solve_q_mode=int(mosaic.get("solve_q_mode", DEFAULT_SOLVE_Q_MODE)),
                            thickness=float(sample_depth_var.get()),
                            pixel_size_m=float(pixel_size_m),
                            sample_width_m=float(sample_width_var.get()),
                            sample_length_m=float(sample_length_var.get()),
                            sample_weights=mosaic.get("sample_weights"),
                            n2_sample_array_override=mosaic.get("n2_sample_array"),
                        )

                        maxpos = hit_tables_to_max_positions(hit_tables)
                        measured_dict = build_measured_dict(measured_for_fit)

                        rows: list[str] = []
                        for idx, (H, K, L) in enumerate(miller):
                            key = (int(round(H)), int(round(K)), int(round(L)))
                            measured_list = measured_dict.get(key)
                            if not measured_list:
                                continue

                            I0, x0, y0, I1, x1, y1 = maxpos[idx]
                            sim_candidates = [
                                (float(x0), float(y0))
                                if np.isfinite(x0) and np.isfinite(y0)
                                else None,
                                (float(x1), float(y1))
                                if np.isfinite(x1) and np.isfinite(y1)
                                else None,
                            ]
                            sim_candidates = [p for p in sim_candidates if p is not None]
                            if not sim_candidates:
                                continue

                            for mx, my in measured_list:
                                best = None
                                for sx, sy in sim_candidates:
                                    dx = sx - float(mx)
                                    dy = sy - float(my)
                                    dist = math.hypot(dx, dy)
                                    if best is None or dist < best[0]:
                                        best = (dist, dx, dy, sx, sy)

                                if best is None:
                                    continue

                                dist, dx, dy, sx, sy = best

                                frame_entry = next(
                                    (
                                        fr
                                        for fr in picked_frames
                                        if math.isclose(
                                            fr.get("fit", (mx, my))[0], mx, abs_tol=1e-9
                                        )
                                        and math.isclose(
                                            fr.get("fit", (mx, my))[1], my, abs_tol=1e-9
                                        )
                                        and fr.get("label") == f"{key[0]},{key[1]},{key[2]}"
                                    ),
                                    None,
                                )
                                disp_part = (
                                    f"display=({frame_entry['display'][0]:.3f}, {frame_entry['display'][1]:.3f}), "
                                    f"native=({frame_entry['native'][0]:.3f}, {frame_entry['native'][1]:.3f}), "
                                    f"fit=({mx:.3f}, {my:.3f})"
                                    if frame_entry
                                    else f"fit=({mx:.3f}, {my:.3f})"
                                )

                                rows.append(
                                    "HKL=({},{},{}) {} -> sim=({:.3f}, {:.3f}) dx={:.3f} dy={:.3f} |Δ|={:.3f}".format(
                                        key[0], key[1], key[2], disp_part, sx, sy, dx, dy, dist
                                    )
                                )

                        _log_section(
                            title,
                            rows or ["No measured peaks matched"],
                        )
                    except Exception as exc:  # pragma: no cover - debug path
                        _log_section(title, [f"Failed to record assignments: {exc}"])

                def _log_pixel_match_snapshot(title: str, param_set: dict[str, float]):
                    mosaic = param_set["mosaic_params"]
                    wavelength_array = mosaic.get("wavelength_array")
                    if wavelength_array is None:
                        wavelength_array = mosaic.get("wavelength_i_array")

                    sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)
                    _, hit_tables, *_ = process_peaks_parallel(
                        miller,
                        intensities,
                        image_size,
                        param_set["a"],
                        param_set["c"],
                        wavelength_array,
                        sim_buffer,
                        param_set["corto_detector"],
                        param_set["gamma"],
                        param_set["Gamma"],
                        param_set["chi"],
                        param_set.get("psi", 0.0),
                        param_set.get("psi_z", 0.0),
                        param_set["zs"],
                        param_set["zb"],
                        param_set["n2"],
                        mosaic["beam_x_array"],
                        mosaic["beam_y_array"],
                        mosaic["theta_array"],
                        mosaic["phi_array"],
                        mosaic["sigma_mosaic_deg"],
                        mosaic["gamma_mosaic_deg"],
                        mosaic["eta"],
                        wavelength_array,
                        param_set["debye_x"],
                        param_set["debye_y"],
                        param_set["center"],
                        param_set["theta_initial"],
                        param_set.get("cor_angle", 0.0),
                        np.array([1.0, 0.0, 0.0]),
                        np.array([0.0, 1.0, 0.0]),
                        save_flag=0,
                        optics_mode=_current_optics_mode_flag(),
                        solve_q_steps=int(mosaic.get("solve_q_steps", DEFAULT_SOLVE_Q_STEPS)),
                        solve_q_rel_tol=float(
                            mosaic.get("solve_q_rel_tol", DEFAULT_SOLVE_Q_REL_TOL)
                        ),
                        solve_q_mode=int(mosaic.get("solve_q_mode", DEFAULT_SOLVE_Q_MODE)),
                        thickness=float(sample_depth_var.get()),
                        pixel_size_m=float(pixel_size_m),
                        sample_width_m=float(sample_width_var.get()),
                        sample_length_m=float(sample_length_var.get()),
                        sample_weights=mosaic.get("sample_weights"),
                        n2_sample_array_override=mosaic.get("n2_sample_array"),
                    )

                    maxpos = hit_tables_to_max_positions(hit_tables)
                    measured_dict = build_measured_dict(measured_for_fit)

                    rows: list[str] = []
                    per_residual: list[float] = []

                    def _center_from_maxpos(entry: Sequence[float]) -> tuple[float, float] | None:
                        _, x0, y0, _, x1, y1 = entry
                        candidates = [
                            (float(x0), float(y0)) if np.isfinite(x0) and np.isfinite(y0) else None,
                            (float(x1), float(y1)) if np.isfinite(x1) and np.isfinite(y1) else None,
                        ]
                        candidates = [p for p in candidates if p is not None]
                        if not candidates:
                            return None
                        cols, rows_local = zip(*candidates)
                        return float(np.mean(cols)), float(np.mean(rows_local))

                    simulated_by_hkl: dict[tuple[int, int, int], list[tuple[float, float]]] = {}
                    for idx, (H, K, L) in enumerate(miller):
                        key = (int(round(H)), int(round(K)), int(round(L)))
                        if key not in measured_dict:
                            continue
                        center = _center_from_maxpos(maxpos[idx])
                        if center is not None:
                            simulated_by_hkl.setdefault(key, []).append(center)

                    for hkl_key, measured_list in measured_dict.items():
                        sim_list = simulated_by_hkl.get(hkl_key)
                        if not sim_list:
                            continue

                        sim_arr = np.asarray(sim_list, dtype=float)
                        sim_center = (
                            float(sim_arr[:, 0].mean()),
                            float(sim_arr[:, 1].mean()),
                        )

                        meas_arr = np.asarray(measured_list, dtype=float)
                        meas_center = (
                            float(meas_arr[:, 0].mean()),
                            float(meas_arr[:, 1].mean()),
                        )

                        dx = sim_center[0] - meas_center[0]
                        dy = sim_center[1] - meas_center[1]
                        dist = math.hypot(dx, dy)
                        per_residual.append(dist)
                        rows.append(
                            "HKL=({},{},{}) sim=({:.3f}, {:.3f}) meas=({:.3f}, {:.3f}) "
                            "dx={:.3f} dy={:.3f} |Δ|={:.3f}".format(
                                hkl_key[0],
                                hkl_key[1],
                                hkl_key[2],
                                sim_center[0],
                                sim_center[1],
                                meas_center[0],
                                meas_center[1],
                                dx,
                                dy,
                                dist,
                            )
                        )

                    rms = (
                        math.sqrt(float(np.mean(np.square(per_residual)))) if per_residual else 0.0
                    )
                    max_dist = max(per_residual) if per_residual else 0.0
                    _log_section(
                        title,
                        [
                            f"matches={len(per_residual)}, RMS={rms:.3f} px, max={max_dist:.3f} px",
                            *rows,
                        ],
                    )

                def _log_matches_snapshot(title: str, param_set: dict[str, float]):
                    try:
                        (
                            _,
                            pre_sim_coords,
                            pre_meas_coords,
                            pre_sim_millers,
                            pre_meas_millers,
                        ) = simulate_and_compare_hkl(
                            miller,
                            intensities,
                            image_size,
                            param_set,
                            measured_for_fit,
                            pixel_tol=float("inf"),
                            prefer_python_runner=True,
                        )
                        (
                            pre_sim_centers,
                            pre_meas_centers,
                            pre_hkls,
                        ) = _aggregate_match_centers(
                            pre_sim_coords,
                            pre_meas_coords,
                            pre_sim_millers,
                            pre_meas_millers,
                        )
                    except Exception as exc:  # pragma: no cover - debug path
                        _log_section(title, [f"Failed to collect matches: {exc}"])
                        return None

                    if not pre_hkls:
                        _log_section(title, ["No matched peaks found; residuals would be empty."])
                        return []

                    rows: list[str] = []
                    distances: list[float] = []
                    for hkl_key, sim_ctr, meas_ctr in zip(
                        pre_hkls, pre_sim_centers, pre_meas_centers
                    ):
                        dx = sim_ctr[0] - meas_ctr[0]
                        dy = sim_ctr[1] - meas_ctr[1]
                        dist = math.hypot(dx, dy)
                        distances.append(dist)
                        rows.append(
                            "HKL="
                            f"{hkl_key}: sim=({sim_ctr[0]:.3f}, {sim_ctr[1]:.3f}), "
                            f"meas=({meas_ctr[0]:.3f}, {meas_ctr[1]:.3f}), "
                            f"dx={dx:.3f}, dy={dy:.3f}, |Δ|={dist:.3f}"
                        )

                    rms = math.sqrt(float(np.mean(np.square(distances)))) if distances else 0.0
                    max_dist = max(distances) if distances else 0.0
                    summary = [
                        f"Matches={len(rows)}, RMS={rms:.3f} px, max={max_dist:.3f} px",
                        *rows,
                    ]
                    _log_section(title, summary)
                    return distances

                _log_section(
                    "Fitting variables (start values):",
                    [
                        (
                            f"{name}: <missing>"
                            if params.get(name) is None
                            else f"{name}: {float(params.get(name)):.6f}"
                        )
                        for name in var_names
                    ],
                )
                _log_matches_snapshot("Matches before fit (native frame):", params)
                _log_pixel_match_snapshot("Pixel matches before fit (native frame):", params)
                _log_assignment_snapshot("Match assignments before fit (native frame):", params)
                candidate_param_names = _current_geometry_fit_candidate_param_names()
                geometry_runtime_cfg = _build_geometry_fit_runtime_config(
                    geometry_refine_cfg,
                    {name: params.get(name) for name in candidate_param_names},
                    _current_geometry_fit_constraint_state(candidate_param_names),
                    _current_geometry_fit_parameter_domains(candidate_param_names),
                    candidate_param_names=candidate_param_names,
                )

                result = fit_geometry_parameters(
                    miller,
                    intensities,
                    image_size,
                    params,
                    measured_for_fit,
                    var_names,
                    pixel_tol=float("inf"),
                    experimental_image=experimental_image_for_fit,
                    refinement_config=geometry_runtime_cfg,
                    candidate_param_names=candidate_param_names,
                )

                _log_section(
                    "Optimizer diagnostics:",
                    [
                        f"orientation={orientation_choice.get('label', 'identity')}",
                        f"success={getattr(result, 'success', False)}",
                        f"status={getattr(result, 'status', '')}",
                        f"message={(getattr(result, 'message', '') or '').strip()}",
                        f"nfev={getattr(result, 'nfev', '<unknown>')}",
                        f"cost={float(getattr(result, 'cost', np.nan)):.6f}",
                        f"robust_cost={float(getattr(result, 'robust_cost', np.nan)):.6f}",
                        f"solver_loss={getattr(result, 'solver_loss', '<unknown>')}",
                        f"solver_f_scale={float(getattr(result, 'solver_f_scale', np.nan)):.6f}",
                        f"optimality={float(getattr(result, 'optimality', np.nan)):.6f}",
                        f"active_mask={list(getattr(result, 'active_mask', []))}",
                        *[
                            "restart[{idx}] cost={cost:.6f} success={success} msg={msg}".format(
                                idx=int(entry.get("restart", -1)),
                                cost=float(entry.get("cost", np.nan)),
                                success=bool(entry.get("success", False)),
                                msg=str(entry.get("message", "")).strip(),
                            )
                            for entry in (getattr(result, "restart_history", []) or [])
                        ],
                    ],
                )

                for name, val in zip(var_names, result.x):
                    if name == "zb":
                        zb_var.set(val)
                    elif name == "zs":
                        zs_var.set(val)
                    elif name == "theta_initial":
                        theta_initial_var.set(val)
                    elif name == "theta_offset":
                        if geometry_theta_offset_var is not None:
                            geometry_theta_offset_var.set(f"{float(val):.6g}")
                    elif name == "psi_z":
                        psi_z_var.set(val)
                    elif name == "chi":
                        chi_var.set(val)
                    elif name == "cor_angle":
                        cor_angle_var.set(val)
                    elif name == "gamma":
                        gamma_var.set(val)
                    elif name == "Gamma":
                        Gamma_var.set(val)
                    elif name == "corto_detector":
                        corto_detector_var.set(val)
                    elif name == "a":
                        a_var.set(val)
                    elif name == "c":
                        c_var.set(val)
                    elif name == "center_x":
                        center_x_var.set(val)
                    elif name == "center_y":
                        center_y_var.set(val)

                if _geometry_fit_uses_shared_theta_offset() and not preserve_live_theta:
                    theta_initial_var.set(
                        _background_theta_base_for_index(
                            background_runtime_state.current_background_index,
                            strict_count=False,
                        )
                    )
                    _refresh_background_status()

                # Keep the cached profile in sync with the fitted geometry so the
                # next simulation uses the updated parameters even when diagnostics
                # are disabled.
                simulation_runtime_state.profile_cache = dict(
                    simulation_runtime_state.profile_cache
                )
                simulation_runtime_state.profile_cache.update(mosaic_params)
                simulation_runtime_state.profile_cache.update(
                    {
                        "theta_initial": theta_initial_var.get(),
                        "theta_offset": _current_geometry_theta_offset(strict=False),
                        "cor_angle": cor_angle_var.get(),
                        "chi": chi_var.get(),
                        "zs": zs_var.get(),
                        "zb": zb_var.get(),
                        "gamma": gamma_var.get(),
                        "Gamma": Gamma_var.get(),
                        "corto_detector": corto_detector_var.get(),
                        "a": a_var.get(),
                        "c": c_var.get(),
                        "center_x": center_x_var.get(),
                        "center_y": center_y_var.get(),
                    }
                )

                # Force a fresh simulation with the fitted values.
                gui_controllers.request_geometry_preview_skip_once(geometry_preview_state)
                _invalidate_simulation_cache()
                schedule_update()

                rms = (
                    float(getattr(result, "rms_px"))
                    if np.isfinite(float(getattr(result, "rms_px", np.nan)))
                    else (
                        np.sqrt(np.mean(result.fun**2))
                        if getattr(result, "fun", None) is not None and result.fun.size
                        else 0.0
                    )
                )
                _log_section(
                    "Optimization result:",
                    [f"{name} = {val:.6f}" for name, val in zip(var_names, result.x)]
                    + [f"RMS residual = {rms:.6f} px"],
                )
                base_summary = (
                    "Fit complete:\n"
                    + "\n".join(f"{name} = {val:.4f}" for name, val in zip(var_names, result.x))
                    + f"\nRMS residual = {rms:.2f} px"
                )

                fitted_params = dict(params)
                fitted_params.update(
                    {
                        "zb": zb_var.get(),
                        "zs": zs_var.get(),
                        "theta_initial": theta_initial_var.get(),
                        "theta_offset": _current_geometry_theta_offset(strict=False),
                        "chi": chi_var.get(),
                        "cor_angle": cor_angle_var.get(),
                        "psi_z": psi_z_var.get(),
                        "gamma": gamma_var.get(),
                        "Gamma": Gamma_var.get(),
                        "corto_detector": corto_detector_var.get(),
                        "a": a_var.get(),
                        "c": c_var.get(),
                        "center": [center_x_var.get(), center_y_var.get()],
                        "center_x": center_x_var.get(),
                        "center_y": center_y_var.get(),
                    }
                )

                _log_matches_snapshot("Matches after fit (native frame):", fitted_params)
                _log_pixel_match_snapshot("Pixel matches after fit (native frame):", fitted_params)
                _log_assignment_snapshot(
                    "Match assignments after fit (native frame):", fitted_params
                )

                (
                    _,
                    sim_coords,
                    meas_coords,
                    sim_millers,
                    meas_millers,
                ) = simulate_and_compare_hkl(
                    miller,
                    intensities,
                    image_size,
                    fitted_params,
                    measured_for_fit,
                    pixel_tol=float("inf"),
                    prefer_python_runner=True,
                )
            except Exception as exc:
                _log_line(f"Geometry fit failed: {exc}")
                try:
                    log_file.close()
                except Exception:
                    pass
                progress_label_geometry.config(text=f"Geometry fit failed: {exc}")
                return

            (
                agg_sim_coords,
                agg_meas_coords,
                agg_millers,
            ) = _aggregate_match_centers(sim_coords, meas_coords, sim_millers, meas_millers)

            pixel_offsets: list[tuple[tuple[int, int, int], float, float, float]] = []
            max_display_markers = int(auto_match_cfg.get("max_display_markers", 120))
            max_display_markers = max(1, max_display_markers)

            for hkl_key, sim_center, meas_center in zip(
                agg_millers, agg_sim_coords, agg_meas_coords
            ):
                dx = sim_center[0] - meas_center[0]
                dy = sim_center[1] - meas_center[1]
                dist = math.hypot(dx, dy)
                pixel_offsets.append((hkl_key, dx, dy, dist))

            native_overlay_shape = tuple(int(v) for v in native_background.shape[:2])
            overlay_records = build_geometry_fit_overlay_records(
                initial_manual_pairs_display,
                getattr(result, "point_match_diagnostics", None),
                native_shape=native_overlay_shape,
                orientation_choice=orientation_choice,
                sim_display_rotate_k=SIM_DISPLAY_ROTATE_K,
                background_display_rotate_k=DISPLAY_ROTATE_K,
            )
            frame_diag, frame_warning = _geometry_overlay_frame_diagnostics(
                overlay_records,
            )
            _log_section(
                "Overlay frame diagnostics:",
                [
                    "transform_rule=sim:direct_native_to_display; bg:inverse_orientation_then_display_rotation",
                    f"overlay_records={len(overlay_records)}",
                    f"paired_records={int(frame_diag.get('paired_records', 0))}",
                    f"sim_display_med_px={float(frame_diag.get('sim_display_med_px', np.nan)):.3f}",
                    f"bg_display_med_px={float(frame_diag.get('bg_display_med_px', np.nan)):.3f}",
                    f"sim_display_p90_px={float(frame_diag.get('sim_display_p90_px', np.nan)):.3f}",
                    f"bg_display_p90_px={float(frame_diag.get('bg_display_p90_px', np.nan)):.3f}",
                ],
            )
            if frame_warning:
                _log_line(f"WARNING: {frame_warning}")
                _log_line()

            if overlay_records:
                _draw_geometry_fit_overlay(
                    overlay_records,
                    max_display_markers=max_display_markers,
                )
            else:
                _draw_initial_geometry_pairs_overlay(
                    initial_manual_pairs_display,
                    max_display_markers=max_display_markers,
                )

            _log_section(
                "Pixel offsets (native frame):",
                [
                    f"HKL={hkl}: dx={dx:.4f}, dy={dy:.4f}, |Δ|={dist:.4f} px"
                    for hkl, dx, dy, dist in pixel_offsets
                ]
                or ["No matched peaks"],
            )

            try:
                log_file.close()
            except Exception:
                pass

            export_recs = []
            for hkl, (x, y), (_, _, _, dist) in zip(agg_millers, agg_sim_coords, pixel_offsets):
                export_recs.append(
                    {
                        "source": "sim",
                        "hkl": tuple(int(v) for v in hkl),
                        "x": int(x),
                        "y": int(y),
                        "dist_px": float(dist),
                    }
                )
            for hkl, (x, y), (_, _, _, dist) in zip(agg_millers, agg_meas_coords, pixel_offsets):
                export_recs.append(
                    {
                        "source": "meas",
                        "hkl": tuple(int(v) for v in hkl),
                        "x": int(x),
                        "y": int(y),
                        "dist_px": float(dist),
                    }
                )

            download_dir = get_dir("downloads")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = download_dir / f"matched_peaks_{stamp}.npy"
            np.save(save_path, np.array(export_recs, dtype=object), allow_pickle=True)

            _log_section(
                "Fit summary:",
                [
                    "Parameters:" if var_names else "Parameters: <none>",
                    *[f"{name} = {val:.6f}" for name, val in zip(var_names, result.x)],
                    f"overlay_records={len(overlay_records)}",
                    f"RMS residual = {rms:.6f} px",
                    f"Applied orientation: {orientation_choice.get('label', 'identity')}",
                    f"Matched peaks saved to: {save_path}",
                ],
            )

            if pixel_offsets:
                dist_lines = [
                    f"HKL={hkl}: |Δ|={dist:.2f}px (dx={dx:.2f}, dy={dy:.2f})"
                    for hkl, dx, dy, dist in pixel_offsets
                ]
                dist_report = "\n".join(dist_lines)
            else:
                dist_report = "No matched peaks to report distances."

            orientation_report = (
                f"Applied orientation: {orientation_choice.get('label', 'identity')}"
            )
            overlay_hint = (
                "Overlay: blue squares=initial simulated picks, amber triangles=fixed background picks, "
                "green circles=fitted simulated peaks, dashed arrows=initial->fitted sim shifts."
            )

            if DEBUG_ENABLED:
                final_text = (
                    f"{base_summary}\n{orientation_report}\n{overlay_hint}\n"
                    + (f"{frame_warning}\n" if frame_warning else "")
                    + f"Fit log → {log_path}"
                )
            else:
                final_text = (
                    f"{base_summary}\n\nSaved {len(export_recs)} peak records →\n{save_path}"
                    + f"\n\n{overlay_hint}\n"
                    + (f"{frame_warning}\n" if frame_warning else "")
                    + f"\nPixel offsets:\n{dist_report}\nFit log → {log_path}"
                )

            progress_label_geometry.config(text=final_text)
            return

        col, row = float(event.xdata), float(event.ydata)

        if selection_state["expecting"] == "sim":
            idx, _ = _nearest_simulated_peak(col, row)
            if idx is None:
                progress_label_geometry.config(text="No simulated peaks available to pick.")
                return
            selection_state["pending_hkl"] = simulation_runtime_state.peak_millers[idx]
            sim_col, sim_row = simulation_runtime_state.peak_positions[idx]
            selection_state["pending_sim_display"] = (float(sim_col), float(sim_row))
            _mark_pick(sim_col, sim_row, f"{selection_state['pending_hkl']} sim", "#00b894", "o")
            progress_label_geometry.config(
                text=(
                    f"Selected simulated peak HKL={selection_state['pending_hkl']} "
                    "→ click matching real peak (right click to finish)."
                )
            )
            selection_state["expecting"] = "real"
            return

        pending = selection_state.get("pending_hkl")
        if pending is None:
            selection_state["expecting"] = "sim"
            selection_state["pending_sim_display"] = None
            progress_label_geometry.config(text="Pick a simulated peak first.")
            return

        peak_col, peak_row = _peak_maximum_near(col, row, search_radius=6)
        _mark_pick(peak_col, peak_row, f"{pending} real", "#e17055", "x")

        pending_sim_display = selection_state.get("pending_sim_display")
        picked_pairs.append((pending, (peak_col, peak_row)))
        if pending_sim_display is not None:
            initial_manual_pairs_display.append(
                {
                    "overlay_match_index": len(initial_manual_pairs_display),
                    "hkl": tuple(int(v) for v in pending),
                    "sim_display": (
                        float(pending_sim_display[0]),
                        float(pending_sim_display[1]),
                    ),
                    "bg_display": (float(peak_col), float(peak_row)),
                }
            )
        progress_label_geometry.config(
            text=(
                f"Recorded pair for HKL={pending} at real px=({peak_col:.1f},{peak_row:.1f}). "
                "Select another simulated peak or right click to fit."
            )
        )
        selection_state["expecting"] = "sim"
        selection_state["pending_sim_display"] = None

    click_cid = canvas.mpl_connect("button_press_event", _on_geometry_pick)
    return


def on_fit_mosaic_click():
    """Run the detector-shape mosaic optimizer from cached or live fit datasets."""

    miller_array = np.asarray(structure_model_state.miller, dtype=np.float64)
    if miller_array.ndim != 2 or miller_array.shape[1] != 3 or miller_array.size == 0:
        progress_label_mosaic.config(
            text="Mosaic shape fit unavailable: no simulated reflections loaded."
        )
        return

    intensity_array = np.asarray(structure_model_state.intensities, dtype=np.float64)
    if intensity_array.shape[0] != miller_array.shape[0]:
        progress_label_mosaic.config(
            text="Mosaic shape fit unavailable: intensity array is not aligned with HKLs."
        )
        return

    mosaic_shape_cfg = fit_config.get("mosaic_shape", {}) if isinstance(fit_config, dict) else {}
    solver_cfg = mosaic_shape_cfg.get("solver", {}) if isinstance(mosaic_shape_cfg, dict) else {}
    roi_cfg = mosaic_shape_cfg.get("roi", {}) if isinstance(mosaic_shape_cfg, dict) else {}
    preprocessing_cfg = (
        mosaic_shape_cfg.get("preprocessing", {}) if isinstance(mosaic_shape_cfg, dict) else {}
    )
    sampling_cfg = (
        mosaic_shape_cfg.get("sampling", {}) if isinstance(mosaic_shape_cfg, dict) else {}
    )

    try:
        fit_sample_floor = int(
            sampling_cfg.get(
                "min_num_samples",
                sampling_cfg.get(
                    "target_num_samples",
                    MOSAIC_SHAPE_FIT_MIN_SAMPLE_COUNT,
                ),
            )
        )
    except (TypeError, ValueError):
        fit_sample_floor = MOSAIC_SHAPE_FIT_MIN_SAMPLE_COUNT
    fit_sample_floor = max(
        int(fit_sample_floor),
        int(MOSAIC_SHAPE_FIT_MIN_SAMPLE_COUNT),
    )
    fit_sample_count = max(
        int(simulation_runtime_state.num_samples),
        max(int(fit_sample_floor), 1),
    )
    fit_solve_q_steps = sampling_cfg.get("solve_q_steps")
    fit_sample_seed = sampling_cfg.get("seed", 0)

    mosaic_params = build_mosaic_params(
        sample_count=fit_sample_count,
        solve_q_steps=fit_solve_q_steps,
        rng_seed=fit_sample_seed,
    )
    required_keys = (
        "beam_x_array",
        "beam_y_array",
        "theta_array",
        "phi_array",
        "wavelength_array",
    )
    missing = [key for key in required_keys if not np.asarray(mosaic_params.get(key)).size]
    if missing:
        progress_label_mosaic.config(
            text="Mosaic shape fit unavailable: run a simulation to populate mosaic samples first."
        )
        return

    cache_payload = _geometry_fit_dataset_cache_payload()
    try:
        selected_background_indices = list(_current_geometry_fit_background_indices())
    except Exception:
        selected_background_indices = [int(background_runtime_state.current_background_index)]
    try:
        background_theta_values = list(_current_background_theta_values(strict_count=True))
    except Exception:
        background_theta_values = list(_current_background_theta_values(strict_count=False))
    try:
        shared_theta_mode = bool(
            _geometry_fit_uses_shared_theta_offset(selected_background_indices)
        )
    except Exception:
        shared_theta_mode = bool(_geometry_fit_uses_shared_theta_offset())

    params = {
        "a": a_var.get(),
        "c": c_var.get(),
        "lambda": lambda_,
        "psi": psi,
        "psi_z": psi_z_var.get(),
        "zs": zs_var.get(),
        "zb": zb_var.get(),
        "sample_width_m": sample_width_var.get(),
        "sample_length_m": sample_length_var.get(),
        "sample_depth_m": sample_depth_var.get(),
        "chi": chi_var.get(),
        "n2": n2,
        "mosaic_params": mosaic_params,
        "debye_x": debye_x_var.get(),
        "debye_y": debye_y_var.get(),
        "center": [center_x_var.get(), center_y_var.get()],
        "theta_initial": _current_effective_theta_initial(strict_count=False),
        "theta_offset": _current_geometry_theta_offset(),
        "uv1": np.array([1.0, 0.0, 0.0]),
        "uv2": np.array([0.0, 1.0, 0.0]),
        "corto_detector": corto_detector_var.get(),
        "gamma": gamma_var.get(),
        "Gamma": Gamma_var.get(),
        "optics_mode": _current_optics_mode_flag(),
        "pixel_size": pixel_size_m,
        "pixel_size_m": pixel_size_m,
    }

    def _format_mosaic_unavailable(reason: object, *, fallback: str) -> str:
        text = " ".join(str(reason).split())
        if not text:
            text = str(fallback)
        prefix = "Geometry fit unavailable:"
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
        return f"Mosaic shape fit unavailable: {text}"

    def _copy_dataset_specs(raw_specs: object) -> list[dict[str, object]]:
        copied_specs: list[dict[str, object]] = []
        for spec in raw_specs if isinstance(raw_specs, list) else []:
            if isinstance(spec, dict):
                copied_specs.append(gui_geometry_fit.copy_geometry_fit_state_value(dict(spec)))
        return copied_specs

    stale_reason = gui_geometry_fit.geometry_fit_dataset_cache_stale_reason(
        cache_payload,
        selected_background_indices=selected_background_indices,
        current_background_index=int(background_runtime_state.current_background_index),
        joint_background_mode=bool(shared_theta_mode),
        background_theta_values=background_theta_values,
    )
    dataset_specs = []
    if stale_reason is None:
        dataset_specs = _copy_dataset_specs(
            cache_payload.get("dataset_specs", []) if isinstance(cache_payload, dict) else []
        )
        cached_dataset_indices = []
        for spec in dataset_specs:
            try:
                cached_dataset_indices.append(int(spec.get("dataset_index")))
            except Exception:
                cached_dataset_indices.append(None)
        expected_dataset_indices = [int(idx) for idx in selected_background_indices]
        if cached_dataset_indices != expected_dataset_indices:
            stale_reason = "cached datasets do not match the current fit-background selection."
            dataset_specs = []
        elif any(
            not isinstance(spec.get("experimental_image"), np.ndarray) for spec in dataset_specs
        ):
            stale_reason = "cached experimental images are missing."
            dataset_specs = []

    prepared_dataset_run = None
    if not dataset_specs:
        manual_dataset_bindings = geometry_fit_manual_dataset_bindings_factory()
        prepared_result = gui_geometry_fit.prepare_geometry_fit_run(
            params=params,
            var_names=(),
            fit_config=fit_config,
            osc_files=manual_dataset_bindings.osc_files,
            current_background_index=int(manual_dataset_bindings.current_background_index),
            theta_initial=theta_initial_var.get(),
            preserve_live_theta=False,
            apply_geometry_fit_background_selection=(_apply_geometry_fit_background_selection),
            current_geometry_fit_background_indices=(_current_geometry_fit_background_indices),
            geometry_fit_uses_shared_theta_offset=(_geometry_fit_uses_shared_theta_offset),
            apply_background_theta_metadata=_apply_background_theta_metadata,
            current_background_theta_values=_current_background_theta_values,
            current_geometry_theta_offset=_current_geometry_theta_offset,
            geometry_manual_pairs_for_index=(
                manual_dataset_bindings.geometry_manual_pairs_for_index
            ),
            ensure_geometry_fit_caked_view=lambda: None,
            build_dataset=(
                lambda background_index, *, theta_base, base_fit_params, orientation_cfg: (
                    gui_geometry_fit.build_geometry_manual_fit_dataset(
                        background_index,
                        theta_base=theta_base,
                        base_fit_params=base_fit_params,
                        manual_dataset_bindings=manual_dataset_bindings,
                        orientation_cfg=orientation_cfg,
                    )
                )
            ),
            build_runtime_config=(
                lambda fit_params: geometry_fit_runtime_config_factory(
                    (),
                    fit_params,
                )
            ),
            require_selected_var_names=False,
            require_active_background_in_selection=False,
            include_all_selected_backgrounds=True,
        )
        prepared_dataset_run = prepared_result.prepared_run
        if prepared_dataset_run is None:
            progress_label_mosaic.config(
                text=_format_mosaic_unavailable(
                    prepared_result.error_text or stale_reason,
                    fallback="unable to prepare mosaic-fit datasets.",
                )
            )
            return
        dataset_specs = [
            gui_geometry_fit.copy_geometry_fit_state_value(dict(spec))
            for spec in prepared_dataset_run.dataset_specs
            if isinstance(spec, dict)
        ]
        selected_background_indices = list(prepared_dataset_run.selected_background_indices)
        background_theta_values = list(prepared_dataset_run.background_theta_values)
        shared_theta_mode = bool(prepared_dataset_run.joint_background_mode)
        params["theta_initial"] = prepared_dataset_run.fit_params.get(
            "theta_initial",
            params["theta_initial"],
        )
        params["theta_offset"] = prepared_dataset_run.fit_params.get(
            "theta_offset",
            params["theta_offset"],
        )

    if not dataset_specs:
        progress_label_mosaic.config(
            text="Mosaic shape fit unavailable: no prepared datasets are available."
        )
        return

    dataset_specs, focused_peak_selection = focus_mosaic_profile_dataset_specs(
        dataset_specs,
        source_miller=miller_array,
        source_intensities=intensity_array,
        reference_dataset_index=int(background_runtime_state.current_background_index),
        max_in_plane_groups=int(MOSAIC_SHAPE_FIT_MAX_IN_PLANE_GROUPS),
    )
    focused_dataset_peak_counts = [
        int(len(spec.get("measured_peaks", [])))
        for spec in dataset_specs
        if isinstance(spec, Mapping)
    ]
    configured_min_total_rois = int(roi_cfg.get("min_total_rois", 8))
    configured_min_per_dataset_rois = int(roi_cfg.get("min_per_dataset_rois", 3))
    min_total_rois = max(
        1,
        min(
            int(configured_min_total_rois),
            int(sum(focused_dataset_peak_counts)) if focused_dataset_peak_counts else 1,
        ),
    )
    min_per_dataset_rois = max(
        1,
        min(
            int(configured_min_per_dataset_rois),
            min(focused_dataset_peak_counts) if focused_dataset_peak_counts else 1,
        ),
    )

    theta_mode = "single"
    if len(dataset_specs) > 1:
        theta_mode = "shared_offset" if shared_theta_mode else "per_dataset"
    ridge_weight = float(solver_cfg.get("ridge_weight", 1.0))
    specular_relative_intensity_weight = float(
        solver_cfg.get(
            "specular_relative_intensity_weight",
            solver_cfg.get("point_match_weight", 1.0),
        )
    )
    fit_theta_i = bool(solver_cfg.get("fit_theta_i", solver_cfg.get("refine_theta", True)))
    configured_theta_i_mode = (
        str(solver_cfg.get("theta_i_mode", solver_cfg.get("theta_mode", "auto"))).strip().lower()
    )
    theta_i_mode = theta_mode if configured_theta_i_mode == "auto" else configured_theta_i_mode

    raw_theta_i_bounds = solver_cfg.get(
        "theta_i_bounds_deg",
        solver_cfg.get("theta_bounds", None),
    )

    theta_i_bounds_deg = None
    if isinstance(raw_theta_i_bounds, (list, tuple)) and len(raw_theta_i_bounds) == 2:
        try:
            theta_i_bounds_deg = (
                float(raw_theta_i_bounds[0]),
                float(raw_theta_i_bounds[1]),
            )
        except Exception:
            theta_i_bounds_deg = None

    def _read_mosaic_toggle(toggle_var, fallback: bool) -> bool:
        getter = getattr(toggle_var, "get", None)
        if callable(getter):
            try:
                return bool(getter())
            except Exception:
                return bool(fallback)
        return bool(fallback)

    fit_sigma_mosaic = _read_mosaic_toggle(
        geometry_overlay_actions_view_state.fit_sigma_mosaic_var,
        bool(solver_cfg.get("fit_sigma_mosaic", True)),
    )
    fit_gamma_mosaic = _read_mosaic_toggle(
        geometry_overlay_actions_view_state.fit_gamma_mosaic_var,
        bool(solver_cfg.get("fit_gamma_mosaic", True)),
    )
    fit_eta = _read_mosaic_toggle(
        geometry_overlay_actions_view_state.fit_eta_var,
        bool(solver_cfg.get("fit_eta", True)),
    )
    fit_theta_i = _read_mosaic_toggle(
        geometry_overlay_actions_view_state.fit_theta_i_var,
        bool(solver_cfg.get("fit_theta_i", solver_cfg.get("refine_theta", True))),
    )
    active_fit_parameters = []
    if fit_sigma_mosaic:
        active_fit_parameters.append("sigma")
    if fit_gamma_mosaic:
        active_fit_parameters.append("gamma")
    if fit_eta:
        active_fit_parameters.append("eta")
    if fit_theta_i:
        active_fit_parameters.append("theta_i")
    if not active_fit_parameters:
        progress_label_mosaic.config(
            text="Mosaic shape fit unavailable: enable at least one fit parameter."
        )
        return

    def _mosaic_log_json_safe(value: object) -> object:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, complex):
            return {
                "real": _mosaic_log_json_safe(value.real),
                "imag": _mosaic_log_json_safe(value.imag),
            }
        if isinstance(value, np.generic):
            return _mosaic_log_json_safe(value.item())
        if isinstance(value, np.ndarray):
            return [_mosaic_log_json_safe(item) for item in value.tolist()]
        if isinstance(value, Mapping):
            return {str(key): _mosaic_log_json_safe(val) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_mosaic_log_json_safe(item) for item in value]
        return str(value)

    def _mosaic_log_array_summary(values: object) -> dict[str, object]:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = arr[np.isfinite(arr)]
        summary: dict[str, object] = {
            "count": int(arr.size),
            "finite_count": int(finite.size),
        }
        if finite.size:
            summary.update(
                {
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "mean": float(np.mean(finite)),
                    "std": float(np.std(finite)),
                }
            )
        else:
            summary.update(
                {
                    "min": None,
                    "max": None,
                    "mean": None,
                    "std": None,
                }
            )
        return summary

    def _mosaic_log_image_summary(image: object) -> dict[str, object] | None:
        if image is None:
            return None
        arr = np.asarray(image, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        summary: dict[str, object] = {
            "shape": [int(dim) for dim in arr.shape],
            "finite_count": int(finite.size),
        }
        if finite.size:
            summary.update(
                {
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "mean": float(np.mean(finite)),
                    "sum": float(np.sum(finite)),
                }
            )
        else:
            summary.update(
                {
                    "min": None,
                    "max": None,
                    "mean": None,
                    "sum": None,
                }
            )
        return summary

    def _mosaic_log_measured_peaks(entries: object) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for entry in entries if isinstance(entries, (list, tuple)) else []:
            if not isinstance(entry, Mapping):
                continue
            row: dict[str, object] = {
                "hkl": _mosaic_log_json_safe(entry.get("hkl")),
                "x": _mosaic_log_json_safe(entry.get("x")),
                "y": _mosaic_log_json_safe(entry.get("y")),
            }
            if "source_table_index" in entry:
                row["source_table_index"] = _mosaic_log_json_safe(entry.get("source_table_index"))
            if "source_row_index" in entry:
                row["source_row_index"] = _mosaic_log_json_safe(entry.get("source_row_index"))
            rows.append(row)
        return rows

    input_dataset_summaries = []
    for spec in dataset_specs:
        if not isinstance(spec, Mapping):
            continue
        measured_peaks = spec.get("measured_peaks", [])
        input_dataset_summaries.append(
            {
                "dataset_index": _mosaic_log_json_safe(spec.get("dataset_index")),
                "dataset_label": _mosaic_log_json_safe(
                    spec.get("label", spec.get("dataset_index", "?"))
                ),
                "theta_initial_deg": _mosaic_log_json_safe(spec.get("theta_initial")),
                "measured_peak_count": int(
                    len(measured_peaks) if isinstance(measured_peaks, (list, tuple)) else 0
                ),
                "experimental_image": _mosaic_log_image_summary(spec.get("experimental_image")),
                "measured_peaks": _mosaic_log_measured_peaks(measured_peaks),
            }
        )

    launch_context = {
        "dataset_source": (
            "live_manual_dataset_preparation"
            if prepared_dataset_run is not None
            else "geometry_fit_dataset_cache"
        ),
        "cache_stale_reason": stale_reason,
        "used_prepared_dataset_run": bool(prepared_dataset_run is not None),
        "current_background_index": int(background_runtime_state.current_background_index),
        "selected_background_indices": [int(idx) for idx in selected_background_indices],
        "background_theta_values_deg": _mosaic_log_json_safe(background_theta_values),
        "shared_theta_mode": bool(shared_theta_mode),
        "fit_theta_i": bool(fit_theta_i),
        "theta_i_mode": str(theta_i_mode),
        "theta_i_bounds_deg": _mosaic_log_json_safe(theta_i_bounds_deg),
        "active_fit_parameters": list(active_fit_parameters),
        "fit_toggles": {
            "fit_sigma_mosaic": bool(fit_sigma_mosaic),
            "fit_gamma_mosaic": bool(fit_gamma_mosaic),
            "fit_eta": bool(fit_eta),
            "fit_theta_i": bool(fit_theta_i),
        },
        "solver_config": _mosaic_log_json_safe(solver_cfg),
        "roi_config": _mosaic_log_json_safe(roi_cfg),
        "preprocessing_config": _mosaic_log_json_safe(preprocessing_cfg),
        "sampling_config": _mosaic_log_json_safe(sampling_cfg),
        "fit_sample_count": int(fit_sample_count),
        "live_sample_count": int(simulation_runtime_state.num_samples),
        "fit_sample_seed": _mosaic_log_json_safe(fit_sample_seed),
        "fit_solve_q_steps_override": _mosaic_log_json_safe(fit_solve_q_steps),
        "reflection_source": "structure_model_state_unpruned",
        "focused_peak_selection": _mosaic_log_json_safe(focused_peak_selection),
        "resolved_roi_minimums": {
            "min_total_rois": int(min_total_rois),
            "min_per_dataset_rois": int(min_per_dataset_rois),
        },
        "geometry_parameters": {
            "a": _mosaic_log_json_safe(params.get("a")),
            "c": _mosaic_log_json_safe(params.get("c")),
            "lambda": _mosaic_log_json_safe(params.get("lambda")),
            "psi": _mosaic_log_json_safe(params.get("psi")),
            "psi_z": _mosaic_log_json_safe(params.get("psi_z")),
            "zs": _mosaic_log_json_safe(params.get("zs")),
            "zb": _mosaic_log_json_safe(params.get("zb")),
            "sample_width_m": _mosaic_log_json_safe(params.get("sample_width_m")),
            "sample_length_m": _mosaic_log_json_safe(params.get("sample_length_m")),
            "sample_depth_m": _mosaic_log_json_safe(params.get("sample_depth_m")),
            "chi": _mosaic_log_json_safe(params.get("chi")),
            "n2": _mosaic_log_json_safe(params.get("n2")),
            "center_xy_px": _mosaic_log_json_safe(params.get("center")),
            "theta_initial_deg": _mosaic_log_json_safe(params.get("theta_initial")),
            "theta_offset_deg": _mosaic_log_json_safe(params.get("theta_offset")),
            "corto_detector_m": _mosaic_log_json_safe(params.get("corto_detector")),
            "gamma_deg": _mosaic_log_json_safe(params.get("gamma")),
            "Gamma_deg": _mosaic_log_json_safe(params.get("Gamma")),
            "optics_mode": _mosaic_log_json_safe(params.get("optics_mode")),
            "pixel_size_m": _mosaic_log_json_safe(params.get("pixel_size_m")),
        },
        "mosaic_samples": {
            "beam_x": _mosaic_log_array_summary(mosaic_params.get("beam_x_array", [])),
            "beam_y": _mosaic_log_array_summary(mosaic_params.get("beam_y_array", [])),
            "theta": _mosaic_log_array_summary(mosaic_params.get("theta_array", [])),
            "phi": _mosaic_log_array_summary(mosaic_params.get("phi_array", [])),
            "wavelength": _mosaic_log_array_summary(mosaic_params.get("wavelength_array", [])),
            "n2_sample_array": _mosaic_log_array_summary(mosaic_params.get("n2_sample_array", [])),
            "sigma_mosaic_deg": _mosaic_log_json_safe(mosaic_params.get("sigma_mosaic_deg")),
            "gamma_mosaic_deg": _mosaic_log_json_safe(mosaic_params.get("gamma_mosaic_deg")),
            "eta": _mosaic_log_json_safe(mosaic_params.get("eta")),
            "solve_q_steps": _mosaic_log_json_safe(mosaic_params.get("solve_q_steps")),
            "solve_q_rel_tol": _mosaic_log_json_safe(mosaic_params.get("solve_q_rel_tol")),
            "solve_q_mode": _mosaic_log_json_safe(mosaic_params.get("solve_q_mode")),
        },
        "input_datasets": input_dataset_summaries,
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = None
    if mosaic_fit_log_files_enabled():
        try:
            log_dir = get_dir("debug_log_dir")
        except Exception:
            log_dir = Path.cwd() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir) / f"mosaic_shape_fit_log_{stamp}.txt"
    log_file = None

    def _mosaic_log_line(text: str = "") -> None:
        if log_file is None:
            return
        try:
            log_file.write(text + "\n")
            log_file.flush()
        except Exception:
            pass

    def _mosaic_log_section(title: str, lines: Sequence[str]) -> None:
        _mosaic_log_line(title)
        for line in lines:
            _mosaic_log_line(f"  {line}")
        _mosaic_log_line()

    def _mosaic_live_update(text: str) -> None:
        message = str(text).strip()
        if not message:
            return
        _mosaic_log_line(message)
        _mosaic_fit_cmd_line(message)
        try:
            progress_label_mosaic.config(text=message)
            root.update_idletasks()
        except Exception:
            pass

    progress_label_mosaic.config(
        text=(
            "Running mosaic shape optimization "
            f"({', '.join(active_fit_parameters)}; {fit_sample_count:,} samples)..."
        )
    )
    mosaic_progressbar.start(10)
    root.update_idletasks()

    result = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", encoding="utf-8")
            _mosaic_fit_cmd_line(f"log: {log_path}")
        _mosaic_log_line(f"Mosaic shape fit started: {stamp}")
        _mosaic_log_line()
        _mosaic_log_section(
            "Launch summary:",
            [
                f"dataset_source={launch_context['dataset_source']}",
                f"selected_background_indices={launch_context['selected_background_indices']}",
                f"background_theta_values_deg={launch_context['background_theta_values_deg']}",
                f"shared_theta_mode={launch_context['shared_theta_mode']}",
                f"fit_theta_i={launch_context['fit_theta_i']}",
                f"theta_i_mode={launch_context['theta_i_mode']}",
                f"theta_i_bounds_deg={launch_context['theta_i_bounds_deg']}",
                f"active_fit_parameters={launch_context['active_fit_parameters']}",
                f"focused_peak_selection={launch_context['focused_peak_selection']}",
                f"resolved_roi_minimums={launch_context['resolved_roi_minimums']}",
                f"cache_stale_reason={launch_context['cache_stale_reason']}",
                f"solver_loss={solver_cfg.get('loss', 'soft_l1')}",
                f"solver_max_nfev={solver_cfg.get('max_nfev', 80)}",
                f"solver_restarts={solver_cfg.get('restarts', 2)}",
                f"log_file={log_path if log_path is not None else 'disabled'}",
            ],
        )
        _mosaic_log_line("Launch context (JSON):")
        _mosaic_log_line(
            json.dumps(_mosaic_log_json_safe(launch_context), indent=2, sort_keys=True)
        )
        _mosaic_log_line()
        result = fit_mosaic_shape_parameters(
            miller_array,
            intensity_array,
            image_size,
            params,
            dataset_specs=dataset_specs,
            loss=str(solver_cfg.get("loss", "soft_l1")),
            f_scale=float(solver_cfg.get("f_scale_px", 1.0)),
            max_nfev=int(solver_cfg.get("max_nfev", 80)),
            max_restarts=int(solver_cfg.get("restarts", 2)),
            smooth_sigma_px=float(preprocessing_cfg.get("smooth_sigma_px", 1.0)),
            ridge_percentile=float(preprocessing_cfg.get("ridge_percentile", 85.0)),
            min_total_rois=int(min_total_rois),
            min_per_dataset_rois=int(min_per_dataset_rois),
            equal_dataset_weights=bool(roi_cfg.get("equal_dataset_weights", True)),
            workers=solver_cfg.get("workers", "auto"),
            parallel_mode=str(solver_cfg.get("parallel_mode", "auto")),
            worker_numba_threads=solver_cfg.get("worker_numba_threads", 0),
            restart_jitter=float(solver_cfg.get("restart_jitter", 0.15)),
            ridge_weight=float(ridge_weight),
            specular_relative_intensity_weight=float(specular_relative_intensity_weight),
            fit_theta_i=bool(fit_theta_i),
            theta_i_mode=str(theta_i_mode),
            theta_i_bounds_deg=theta_i_bounds_deg,
            fit_sigma_mosaic=bool(fit_sigma_mosaic),
            fit_gamma_mosaic=bool(fit_gamma_mosaic),
            fit_eta=bool(fit_eta),
            progress_callback=_mosaic_live_update,
        )
        result_summary_lines = [
            f"success={bool(getattr(result, 'success', False))}",
            f"acceptance_passed={bool(getattr(result, 'acceptance_passed', False))}",
            f"message={str(getattr(result, 'message', '') or '').strip()}",
            f"nfev={getattr(result, 'nfev', None)}",
            f"initial_cost={getattr(result, 'initial_cost', None)}",
            f"final_cost={getattr(result, 'final_cost', None)}",
            f"cost_reduction={getattr(result, 'cost_reduction', None)}",
            f"initial_residual_rms={getattr(result, 'initial_residual_rms', None)}",
            f"final_residual_rms={getattr(result, 'final_residual_rms', None)}",
            f"boundary_warning={getattr(result, 'boundary_warning', None)}",
            f"active_parameters={getattr(result, 'active_parameters', None)}",
            f"fixed_parameters={getattr(result, 'fixed_parameters', None)}",
        ]
        _mosaic_log_section("Optimizer summary:", result_summary_lines)
        _mosaic_log_line("Optimizer debug payload (JSON):")
        _mosaic_log_line(
            json.dumps(
                _mosaic_log_json_safe(getattr(result, "mosaic_fit_debug_summary", {}) or {}),
                indent=2,
                sort_keys=True,
            )
        )
        _mosaic_log_line()
    except Exception as exc:  # pragma: no cover - GUI feedback path
        _mosaic_fit_cmd_line(f"failed: {exc}")
        _mosaic_log_line(f"Mosaic shape fit failed: {exc}")
        _mosaic_log_line()
        _mosaic_log_line("Traceback:")
        for trace_line in traceback.format_exception(type(exc), exc, exc.__traceback__):
            for row in str(trace_line).rstrip().splitlines():
                _mosaic_log_line(row)
        progress_text = f"Mosaic shape fit failed: {exc}"
        if log_path is not None:
            progress_text += f"\nFit log → {log_path}"
        progress_label_mosaic.config(text=progress_text)
        return
    finally:
        if log_file is not None:
            try:
                log_file.close()
            except Exception:
                pass
        mosaic_progressbar.stop()
        mosaic_progressbar["value"] = 0
        root.update_idletasks()

    if result.x is None or not np.all(np.isfinite(result.x)):
        progress_text = "Mosaic shape fit failed: optimizer returned invalid parameters."
        if log_path is not None:
            progress_text += f"\nFit log → {log_path}"
        progress_label_mosaic.config(text=progress_text)
        return

    sigma_deg, gamma_deg, eta_val = map(float, result.x[:3])

    sigma_mosaic_var.set(sigma_deg)
    gamma_mosaic_var.set(gamma_deg)
    eta_var.set(eta_val)
    theta_status_text = ""

    best_params = getattr(result, "best_params", None)
    if best_params and "mosaic_params" in best_params:
        simulation_runtime_state.profile_cache = dict(best_params["mosaic_params"])
    else:
        simulation_runtime_state.profile_cache = dict(mosaic_params)
        simulation_runtime_state.profile_cache.update(
            {
                "sigma_mosaic_deg": sigma_deg,
                "gamma_mosaic_deg": gamma_deg,
                "eta": eta_val,
            }
        )

    _invalidate_simulation_cache()
    schedule_update()

    dataset_diagnostics = list(getattr(result, "dataset_diagnostics", []) or [])
    roi_count_by_dataset = dict(getattr(result, "roi_count_by_dataset", {}) or {})
    roi_count = int(getattr(result, "total_roi_count", sum(roi_count_by_dataset.values())))
    dataset_count = len(dataset_specs)
    status = (
        "accepted"
        if bool(getattr(result, "acceptance_passed", False))
        else ("converged" if bool(getattr(result, "success", False)) else "finished")
    )
    message = (getattr(result, "message", "") or "").strip()
    status_text = f"{status} ({message})" if message else status.capitalize()

    dataset_summary = ", ".join(
        f"{diag.get('dataset_label', diag.get('dataset_index'))}={int(diag.get('roi_count', 0))}"
        for diag in dataset_diagnostics
        if isinstance(diag, dict)
    )
    worst_hkls: list[str] = []
    for diag in dataset_diagnostics:
        if not isinstance(diag, dict):
            continue
        label = str(diag.get("dataset_label", diag.get("dataset_index", "?")))
        for hkl in list(diag.get("worst_hkls", []) or [])[:2]:
            try:
                worst_hkls.append(f"{label}:{tuple(int(v) for v in hkl)}")
            except Exception:
                continue
    if len(worst_hkls) > 6:
        worst_hkls = worst_hkls[:6]
        worst_hkls.append("...")

    cost_reduction = 100.0 * float(getattr(result, "cost_reduction", 0.0) or 0.0)
    boundary_warning = str(getattr(result, "boundary_warning", "") or "").strip()

    progress_label_mosaic.config(
        text=(
            f"Mosaic shape fit {status_text}\n"
            f"σ={sigma_deg:.3f}°, γ={gamma_deg:.3f}°, η={eta_val:.3f}{theta_status_text}\n"
            f"datasets={dataset_count}, ROIs={roi_count}, cost reduction={cost_reduction:.1f}%"
            + (f"\nPer dataset: {dataset_summary}" if dataset_summary else "")
            + (f"\nWorst HKLs: {', '.join(worst_hkls)}" if worst_hkls else "")
            + (f"\n{boundary_warning}" if boundary_warning else "")
            + f"\nFit log → {log_path}"
        )
    )


def _clone_structure_model_state_for_ordered_fit():
    """Return one detached copy of the live structure-model state."""

    cloned = copy.copy(structure_model_state)
    cloned.occupancy_site_labels = list(structure_model_state.occupancy_site_labels)
    cloned.occupancy_site_expanded_map = list(structure_model_state.occupancy_site_expanded_map)
    cloned.occ = list(structure_model_state.occ)
    cloned.occ_vars = []
    cloned.atom_site_fractional_metadata = [
        dict(row) for row in structure_model_state.atom_site_fractional_metadata
    ]
    cloned.atom_site_fract_vars = []
    cloned.defaults = dict(structure_model_state.defaults)
    cloned.ht_cache_multi = copy.deepcopy(structure_model_state.ht_cache_multi)
    cloned.ht_curves_cache = copy.deepcopy(structure_model_state.ht_curves_cache)
    cloned.sim_miller1_all = np.asarray(
        structure_model_state.sim_miller1_all,
        dtype=np.float64,
    ).copy()
    cloned.sim_intens1_all = np.asarray(
        structure_model_state.sim_intens1_all,
        dtype=np.float64,
    ).copy()
    cloned.sim_miller2_all = np.asarray(
        structure_model_state.sim_miller2_all,
        dtype=np.float64,
    ).copy()
    cloned.sim_intens2_all = np.asarray(
        structure_model_state.sim_intens2_all,
        dtype=np.float64,
    ).copy()
    cloned.sim_primary_qr_all = gui_controllers.copy_bragg_qr_dict(
        structure_model_state.sim_primary_qr_all
    )
    cloned.miller = np.asarray(structure_model_state.miller, dtype=np.float64).copy()
    cloned.intensities = np.asarray(structure_model_state.intensities, dtype=np.float64).copy()
    cloned.degeneracy = np.asarray(structure_model_state.degeneracy, dtype=np.int32).copy()
    cloned.details = copy.deepcopy(structure_model_state.details)
    cloned.intensities_cif1 = np.asarray(
        structure_model_state.intensities_cif1,
        dtype=np.float64,
    ).copy()
    cloned.intensities_cif2 = np.asarray(
        structure_model_state.intensities_cif2,
        dtype=np.float64,
    ).copy()
    cloned.last_occ_for_ht = list(structure_model_state.last_occ_for_ht)
    cloned.last_p_triplet = list(structure_model_state.last_p_triplet)
    cloned.last_weights = list(structure_model_state.last_weights)
    cloned.last_atom_site_fractional_signature = tuple(
        structure_model_state.last_atom_site_fractional_signature
    )
    return cloned


def _format_ordered_structure_result_text(result, *, mask_roi_count: int) -> str:
    """Return the read-only ordered-structure result summary for the GUI panel."""

    status = (
        "accepted"
        if bool(getattr(result, "acceptance_passed", False))
        else ("converged" if bool(getattr(result, "success", False)) else "finished")
    )
    message = str(getattr(result, "message", "") or "").strip()
    header = f"Ordered structure fit {status}"
    if message:
        header += f" ({message})"
    changed = list(getattr(result, "changed_parameter_names", []) or [])
    if len(changed) > 8:
        changed = changed[:8] + ["..."]
    return "\n".join(
        [
            header,
            (
                f"scale={float(getattr(result, 'scale', 0.0)):.4f}, "
                f"objective reduction={100.0 * float(getattr(result, 'objective_reduction', 0.0) or 0.0):.1f}%, "
                f"ROIs={int(mask_roi_count)}"
            ),
            (
                "changed parameters: " + ", ".join(changed)
                if changed
                else "changed parameters: none"
            ),
        ]
    )


def on_revert_last_ordered_fit():
    """Restore the last accepted ordered-structure fit snapshot."""

    snapshot = app_state.ordered_structure_fit_snapshot
    restored = gui_ordered_structure_fit.restore_ordered_structure_snapshot(
        snapshot,
        occupancy_vars=_occupancy_control_vars(),
        atom_site_vars=_atom_site_fractional_control_vars(),
        debye_x_var=debye_x_var,
        debye_y_var=debye_y_var,
        ordered_scale_var=ordered_structure_scale_var,
    )
    if not restored:
        progress_label_ordered_structure.config(text="Ordered structure fit revert unavailable.")
        return

    app_state.ordered_structure_fit_snapshot = None
    _set_ordered_structure_revert_enabled(False)
    _set_ordered_structure_result_text("Ordered structure fit reverted to the pre-fit snapshot.")
    progress_label_ordered_structure.config(text="Ordered structure fit reverted.")
    update_occupancies()


def on_fit_ordered_structure_click():
    """Run the ordered-structure detector-space intensity refinement."""

    measured_image = _get_current_background_backend()
    if measured_image is None:
        progress_label_ordered_structure.config(
            text="Ordered structure fit unavailable: no current background image."
        )
        return

    measured_arr = np.asarray(measured_image, dtype=np.float64)
    if measured_arr.ndim != 2 or measured_arr.size == 0:
        progress_label_ordered_structure.config(
            text="Ordered structure fit unavailable: background image is empty."
        )
        return

    if simulation_runtime_state.stored_primary_sim_image is None:
        progress_label_ordered_structure.config(
            text="Ordered structure fit unavailable: run a simulation first."
        )
        return

    primary_hit_tables = simulation_runtime_state.stored_primary_max_positions

    try:
        weight1_value = float(weight1_var.get())
        weight2_value = float(weight2_var.get())
    except Exception:
        progress_label_ordered_structure.config(
            text="Ordered structure fit unavailable: invalid CIF weights."
        )
        return

    if abs(weight1_value) <= 1.0e-12:
        progress_label_ordered_structure.config(
            text="Ordered structure fit unavailable: primary CIF weight is zero."
        )
        return

    occupancy_values = list(_current_occupancy_values())
    atom_site_values = list(_current_atom_site_fractional_values())
    parameter_specs: list[gui_ordered_structure_fit.OrderedStructureParameterSpec] = []
    occupancy_param_names: list[str] = []
    atom_param_names: list[dict[str, str]] = []

    coord_window = gui_ordered_structure_fit.normalize_coordinate_window(
        ordered_structure_coord_window_var.get(),
        fallback=ordered_structure_coord_window_default,
    )
    ordered_structure_coord_window_var.set(float(coord_window))

    for idx, toggle_var in enumerate(ordered_structure_fit_occ_toggle_vars):
        name = f"occ:{_occupancy_label_text(idx)}"
        occupancy_param_names.append(name)
        if idx >= len(occupancy_values) or not bool(toggle_var.get()):
            continue
        parameter_specs.append(
            gui_ordered_structure_fit.OrderedStructureParameterSpec(
                name=name,
                value=float(occupancy_values[idx]),
                lower=0.0,
                upper=1.0,
            )
        )

    for idx, axis_vars in enumerate(ordered_structure_fit_atom_toggle_vars):
        label = _atom_site_fractional_label_text(idx)
        name_map: dict[str, str] = {}
        base_values = atom_site_values[idx] if idx < len(atom_site_values) else (0.0, 0.0, 0.0)
        for axis_index, axis in enumerate(("x", "y", "z")):
            name = f"atom:{label}:{axis}"
            name_map[axis] = name
            if not bool(axis_vars.get(axis).get()):
                continue
            current_value = float(base_values[axis_index])
            parameter_specs.append(
                gui_ordered_structure_fit.OrderedStructureParameterSpec(
                    name=name,
                    value=current_value,
                    lower=current_value - float(coord_window),
                    upper=current_value + float(coord_window),
                )
            )
        atom_param_names.append(name_map)

    try:
        debye_x_current = float(debye_x_var.get())
        debye_y_current = float(debye_y_var.get())
    except Exception:
        progress_label_ordered_structure.config(
            text="Ordered structure fit unavailable: invalid Debye Q values."
        )
        return

    if bool(ordered_structure_fit_debye_x_var.get()):
        parameter_specs.append(
            gui_ordered_structure_fit.OrderedStructureParameterSpec(
                name="debye_x",
                value=debye_x_current,
                lower=0.0,
                upper=1.0,
            )
        )
    if bool(ordered_structure_fit_debye_y_var.get()):
        parameter_specs.append(
            gui_ordered_structure_fit.OrderedStructureParameterSpec(
                name="debye_y",
                value=debye_y_current,
                lower=0.0,
                upper=1.0,
            )
        )

    if not parameter_specs:
        progress_label_ordered_structure.config(
            text="Ordered structure fit unavailable: no active parameters selected."
        )
        return

    if not primary_hit_tables:
        try:
            mask_theta_initial = float(_current_effective_theta_initial(strict_count=False))
            mask_mosaic_params = build_mosaic_params()
            mask_optics_mode = int(_current_optics_mode_flag())
            mask_distance = float(corto_detector_var.get())
            mask_gamma = float(gamma_var.get())
            mask_Gamma = float(Gamma_var.get())
            mask_chi = float(chi_var.get())
            mask_psi_z = float(psi_z_var.get())
            mask_zs = float(zs_var.get())
            mask_zb = float(zb_var.get())
            mask_cor = float(cor_angle_var.get())
            mask_width = float(sample_width_var.get())
            mask_length = float(sample_length_var.get())
            mask_depth = float(sample_depth_var.get())
            mask_center = np.asarray(
                [float(center_x_var.get()), float(center_y_var.get())],
                dtype=np.float64,
            )
            mask_a = float(a_var.get())
            mask_c = float(c_var.get())
        except Exception as exc:
            progress_label_ordered_structure.config(
                text=f"Ordered structure fit failed: invalid mask inputs ({exc})."
            )
            return

        current_primary_data = (
            gui_controllers.copy_bragg_qr_dict(simulation_runtime_state.sim_primary_qr)
            if isinstance(simulation_runtime_state.sim_primary_qr, dict)
            and len(simulation_runtime_state.sim_primary_qr) > 0
            else np.asarray(simulation_runtime_state.sim_miller1, dtype=np.float64).copy()
        )
        current_primary_intensities = np.asarray(
            simulation_runtime_state.sim_intens1,
            dtype=np.float64,
        ).copy()
        current_primary_available = (
            len(current_primary_data) > 0
            if isinstance(current_primary_data, dict)
            else (
                np.asarray(current_primary_data).ndim == 2
                and np.asarray(current_primary_data).shape[0] > 0
                and current_primary_intensities.size > 0
            )
        )
        if not current_primary_available:
            progress_label_ordered_structure.config(
                text="Ordered structure fit unavailable: no primary reflections are active."
            )
            return
        try:
            mask_result = _run_simulation_generation_job(
                {
                    "job_id": 0,
                    "signature": ("ordered-structure-mask",),
                    "epoch": 0,
                    "image_size": int(image_size),
                    "pixel_size_m": float(pixel_size_m),
                    "center": mask_center.copy(),
                    "mosaic_params": {
                        "beam_x_array": np.asarray(
                            mask_mosaic_params["beam_x_array"], dtype=np.float64
                        ).copy(),
                        "beam_y_array": np.asarray(
                            mask_mosaic_params["beam_y_array"], dtype=np.float64
                        ).copy(),
                        "theta_array": np.asarray(
                            mask_mosaic_params["theta_array"], dtype=np.float64
                        ).copy(),
                        "phi_array": np.asarray(
                            mask_mosaic_params["phi_array"], dtype=np.float64
                        ).copy(),
                        "wavelength_array": np.asarray(
                            mask_mosaic_params["wavelength_array"],
                            dtype=np.float64,
                        ).copy(),
                        "sample_weights": (
                            None
                            if mask_mosaic_params.get("sample_weights") is None
                            else np.asarray(
                                mask_mosaic_params["sample_weights"],
                                dtype=np.float64,
                            ).copy()
                        ),
                        "n2_sample_array": (
                            None
                            if mask_mosaic_params.get("n2_sample_array") is None
                            else np.asarray(
                                mask_mosaic_params["n2_sample_array"],
                                dtype=np.complex128,
                            ).copy()
                        ),
                        "sigma_mosaic_deg": float(mask_mosaic_params["sigma_mosaic_deg"]),
                        "gamma_mosaic_deg": float(mask_mosaic_params["gamma_mosaic_deg"]),
                        "eta": float(mask_mosaic_params["eta"]),
                        "solve_q_steps": int(mask_mosaic_params["solve_q_steps"]),
                        "solve_q_rel_tol": float(mask_mosaic_params["solve_q_rel_tol"]),
                        "solve_q_mode": int(mask_mosaic_params["solve_q_mode"]),
                        "_sampling_signature": tuple(
                            mask_mosaic_params.get("_sampling_signature", ())
                        ),
                    },
                    "lambda_value": float(lambda_),
                    "distance_m": float(mask_distance),
                    "gamma_deg": float(mask_gamma),
                    "Gamma_deg": float(mask_Gamma),
                    "chi_deg": float(mask_chi),
                    "psi_deg": float(psi),
                    "psi_z_deg": float(mask_psi_z),
                    "zs": float(mask_zs),
                    "zb": float(mask_zb),
                    "theta_initial_deg": float(mask_theta_initial),
                    "cor_angle_deg": float(mask_cor),
                    "sample_width_m": float(mask_width),
                    "sample_length_m": float(mask_length),
                    "sample_depth_m": float(mask_depth),
                    "debye_x": float(debye_x_current),
                    "debye_y": float(debye_y_current),
                    "optics_mode": int(mask_optics_mode),
                    "collect_hit_tables": True,
                    "n2_value": n2,
                    "primary_data": current_primary_data,
                    "primary_intensities": current_primary_intensities,
                    "secondary_data": np.empty((0, 3), dtype=np.float64),
                    "secondary_intensities": np.empty((0,), dtype=np.float64),
                    "run_primary": bool(current_primary_available),
                    "run_secondary": False,
                    "a_primary": float(mask_a),
                    "c_primary": float(mask_c),
                    "a_secondary": float(mask_a),
                    "c_secondary": float(mask_c),
                }
            )
            primary_hit_tables = list(mask_result.get("primary_max_positions", []))
        except Exception as exc:
            progress_label_ordered_structure.config(
                text=f"Ordered structure fit failed while building the ROI mask: {exc}"
            )
            return
        if not primary_hit_tables:
            progress_label_ordered_structure.config(
                text="Ordered structure fit unavailable: no primary hit tables are visible."
            )
            return

    try:
        mask = gui_ordered_structure_fit.build_hybrid_ordered_structure_mask(
            image_shape=measured_arr.shape,
            primary_hit_tables=primary_hit_tables,
            max_reflections=int(ordered_structure_fit_mask_cfg.get("max_reflections", 24)),
            tube_width_scale=float(ordered_structure_fit_mask_cfg.get("tube_width_scale", 1.0)),
            specular_width_scale=float(
                ordered_structure_fit_mask_cfg.get("specular_width_scale", 2.5)
            ),
            equal_peak_weights=bool(ordered_structure_fit_mask_cfg.get("equal_peak_weights", True)),
        )
    except Exception as exc:
        progress_label_ordered_structure.config(text=f"Ordered structure fit failed: {exc}")
        return

    if int(mask.roi_count) <= 0 or int(np.count_nonzero(mask.pixel_mask)) <= 0:
        progress_label_ordered_structure.config(
            text="Ordered structure fit unavailable: mask generation produced no ROIs."
        )
        return

    snapshot = gui_ordered_structure_fit.capture_ordered_structure_snapshot(
        occupancy_values=occupancy_values,
        atom_site_values=atom_site_values,
        debye_x=debye_x_current,
        debye_y=debye_y_current,
        ordered_scale=_current_ordered_structure_scale(),
    )

    current_p_vals = [float(p0_var.get()), float(p1_var.get()), float(p2_var.get())]
    current_weights = gui_controllers.normalize_stacking_weight_values(
        [float(w0_var.get()), float(w1_var.get()), float(w2_var.get())]
    )
    fixed_secondary_image = (
        np.asarray(simulation_runtime_state.stored_secondary_sim_image, dtype=np.float64)
        * weight2_value
        if simulation_runtime_state.stored_secondary_sim_image is not None
        and abs(weight2_value) > 1.0e-12
        else np.zeros_like(measured_arr, dtype=np.float64)
    )
    local_structure_state = _clone_structure_model_state_for_ordered_fit()
    local_atom_site_override_state = gui_state.AtomSiteOverrideState()
    local_simulation_state = gui_state.SimulationRuntimeState()

    try:
        base_theta_initial = float(_current_effective_theta_initial(strict_count=False))
        base_mosaic_params = build_mosaic_params()
        base_optics_mode = int(_current_optics_mode_flag())
        base_distance = float(corto_detector_var.get())
        base_gamma = float(gamma_var.get())
        base_Gamma = float(Gamma_var.get())
        base_chi = float(chi_var.get())
        base_psi_z = float(psi_z_var.get())
        base_zs = float(zs_var.get())
        base_zb = float(zb_var.get())
        base_cor = float(cor_angle_var.get())
        base_width = float(sample_width_var.get())
        base_length = float(sample_length_var.get())
        base_depth = float(sample_depth_var.get())
        base_center = np.asarray(
            [float(center_x_var.get()), float(center_y_var.get())],
            dtype=np.float64,
        )
        base_a = float(a_var.get())
        base_c = float(c_var.get())
    except Exception as exc:
        progress_label_ordered_structure.config(
            text=f"Ordered structure fit failed: invalid frozen runtime values ({exc})."
        )
        return

    def _render_trial_components(parameter_values: dict[str, float]):
        local_occ = list(occupancy_values)
        for idx, name in enumerate(occupancy_param_names):
            if idx < len(local_occ) and name in parameter_values:
                local_occ[idx] = float(parameter_values[name])

        local_atom_values = [
            [float(x_val), float(y_val), float(z_val)] for (x_val, y_val, z_val) in atom_site_values
        ]
        for idx, name_map in enumerate(atom_param_names):
            if idx >= len(local_atom_values):
                break
            for axis_index, axis in enumerate(("x", "y", "z")):
                name = name_map.get(axis)
                if name in parameter_values:
                    local_atom_values[idx][axis_index] = float(parameter_values[name])

        trial_debye_x = float(parameter_values.get("debye_x", debye_x_current))
        trial_debye_y = float(parameter_values.get("debye_y", debye_y_current))
        local_structure_state.two_theta_range = two_theta_range
        local_iodine_z = gui_structure_model.current_iodine_z(
            local_structure_state,
            local_atom_site_override_state,
            atom_site_values=[tuple(values) for values in local_atom_values],
            tcl_error_types=(tk.TclError,),
        )
        gui_structure_model.rebuild_diffraction_inputs(
            local_structure_state,
            new_occ=local_occ,
            p_vals=current_p_vals,
            weights=current_weights,
            a_axis=base_a,
            c_axis=base_c,
            finite_stack_flag=bool(finite_stack_var.get()),
            layers=int(max(1, stack_layers_var.get())),
            phase_delta_expression_current=_current_phase_delta_expression(),
            phi_l_divisor_current=_current_phi_l_divisor(),
            atom_site_values=[tuple(values) for values in local_atom_values],
            iodine_z_current=local_iodine_z,
            rod_points_per_gz=_current_rod_points_per_gz(),
            atom_site_override_state=local_atom_site_override_state,
            simulation_runtime_state=local_simulation_state,
            combine_weighted_intensities=gui_controllers.combine_cif_weighted_intensities,
            build_intensity_dataframes=build_intensity_dataframes,
            apply_bragg_qr_filters=(
                lambda **_kwargs: gui_controllers.apply_bragg_qr_filters(
                    local_simulation_state,
                    bragg_qr_manager_state,
                    prune_bias=current_sf_prune_bias(),
                )
            ),
            schedule_update=lambda: None,
            weight1=weight1_var.get(),
            weight2=weight2_var.get(),
            tcl_error_types=(tk.TclError,),
            force=True,
            trigger_update=False,
        )

        primary_data = (
            gui_controllers.copy_bragg_qr_dict(local_simulation_state.sim_primary_qr)
            if isinstance(local_simulation_state.sim_primary_qr, dict)
            and len(local_simulation_state.sim_primary_qr) > 0
            else np.asarray(local_simulation_state.sim_miller1, dtype=np.float64).copy()
        )
        primary_intensities = np.asarray(
            local_simulation_state.sim_intens1,
            dtype=np.float64,
        ).copy()
        if isinstance(primary_data, dict):
            primary_data = _scaled_bragg_qr_dict(primary_data, weight1_value)
        else:
            primary_intensities *= weight1_value

        primary_available = (
            len(primary_data) > 0
            if isinstance(primary_data, dict)
            else (
                np.asarray(primary_data).ndim == 2
                and np.asarray(primary_data).shape[0] > 0
                and primary_intensities.size > 0
            )
        )
        if not primary_available:
            return np.zeros_like(measured_arr, dtype=np.float64), fixed_secondary_image

        result = _run_simulation_generation_job(
            {
                "job_id": 0,
                "signature": ("ordered-structure",),
                "epoch": 0,
                "image_size": int(image_size),
                "pixel_size_m": float(pixel_size_m),
                "center": base_center.copy(),
                "mosaic_params": {
                    "beam_x_array": np.asarray(
                        base_mosaic_params["beam_x_array"], dtype=np.float64
                    ).copy(),
                    "beam_y_array": np.asarray(
                        base_mosaic_params["beam_y_array"], dtype=np.float64
                    ).copy(),
                    "theta_array": np.asarray(
                        base_mosaic_params["theta_array"], dtype=np.float64
                    ).copy(),
                    "phi_array": np.asarray(
                        base_mosaic_params["phi_array"], dtype=np.float64
                    ).copy(),
                    "wavelength_array": np.asarray(
                        base_mosaic_params["wavelength_array"],
                        dtype=np.float64,
                    ).copy(),
                    "sample_weights": (
                        None
                        if base_mosaic_params.get("sample_weights") is None
                        else np.asarray(
                            base_mosaic_params["sample_weights"],
                            dtype=np.float64,
                        ).copy()
                    ),
                    "n2_sample_array": (
                        None
                        if base_mosaic_params.get("n2_sample_array") is None
                        else np.asarray(
                            base_mosaic_params["n2_sample_array"],
                            dtype=np.complex128,
                        ).copy()
                    ),
                    "sigma_mosaic_deg": float(base_mosaic_params["sigma_mosaic_deg"]),
                    "gamma_mosaic_deg": float(base_mosaic_params["gamma_mosaic_deg"]),
                    "eta": float(base_mosaic_params["eta"]),
                    "solve_q_steps": int(base_mosaic_params["solve_q_steps"]),
                    "solve_q_rel_tol": float(base_mosaic_params["solve_q_rel_tol"]),
                    "solve_q_mode": int(base_mosaic_params["solve_q_mode"]),
                    "_sampling_signature": tuple(base_mosaic_params.get("_sampling_signature", ())),
                },
                "lambda_value": float(lambda_),
                "distance_m": float(base_distance),
                "gamma_deg": float(base_gamma),
                "Gamma_deg": float(base_Gamma),
                "chi_deg": float(base_chi),
                "psi_deg": float(psi),
                "psi_z_deg": float(base_psi_z),
                "zs": float(base_zs),
                "zb": float(base_zb),
                "theta_initial_deg": float(base_theta_initial),
                "cor_angle_deg": float(base_cor),
                "sample_width_m": float(base_width),
                "sample_length_m": float(base_length),
                "sample_depth_m": float(base_depth),
                "debye_x": float(trial_debye_x),
                "debye_y": float(trial_debye_y),
                "optics_mode": int(base_optics_mode),
                "collect_hit_tables": False,
                "n2_value": n2,
                "primary_data": primary_data,
                "primary_intensities": primary_intensities,
                "secondary_data": np.empty((0, 3), dtype=np.float64),
                "secondary_intensities": np.empty((0,), dtype=np.float64),
                "run_primary": bool(primary_available),
                "run_secondary": False,
                "a_primary": float(base_a),
                "c_primary": float(base_c),
                "a_secondary": float(base_a),
                "c_secondary": float(base_c),
            }
        )
        return (
            np.asarray(result.get("primary_image"), dtype=np.float64),
            fixed_secondary_image,
        )

    progress_label_ordered_structure.config(text="Running ordered-structure fit...")
    ordered_structure_progressbar.start(10)
    root.update_idletasks()
    try:
        result = gui_ordered_structure_fit.fit_ordered_structure_parameters(
            measured_image=measured_arr,
            mask=mask,
            parameter_specs=parameter_specs,
            simulate_components=_render_trial_components,
            coarse_downsample_factor=int(
                ordered_structure_fit_solver_cfg.get("coarse_downsample_factor", 2)
            ),
            loss=str(ordered_structure_fit_solver_cfg.get("loss", "soft_l1")),
            f_scale=float(ordered_structure_fit_solver_cfg.get("f_scale", 2.0)),
            coarse_max_nfev=int(ordered_structure_fit_solver_cfg.get("coarse_max_nfev", 15)),
            polish_max_nfev=int(ordered_structure_fit_solver_cfg.get("polish_max_nfev", 10)),
            restarts=int(ordered_structure_fit_solver_cfg.get("restarts", 2)),
        )
    except Exception as exc:
        progress_label_ordered_structure.config(text=f"Ordered structure fit failed: {exc}")
        return
    finally:
        ordered_structure_progressbar.stop()
        try:
            ordered_structure_progressbar["value"] = 0
        except Exception:
            pass
        root.update_idletasks()

    _set_ordered_structure_result_text(
        _format_ordered_structure_result_text(result, mask_roi_count=int(mask.roi_count))
    )
    if not bool(getattr(result, "acceptance_passed", False)):
        progress_label_ordered_structure.config(
            text=(
                "Ordered structure fit finished without acceptance: "
                f"{getattr(result, 'message', 'objective did not improve')}"
            )
        )
        return

    if not np.isfinite(float(getattr(result, "scale", float("nan")))):
        progress_label_ordered_structure.config(
            text="Ordered structure fit failed: optimizer returned an invalid scale."
        )
        return

    app_state.ordered_structure_fit_snapshot = snapshot
    gui_ordered_structure_fit.apply_ordered_structure_values(
        result.parameter_values,
        occupancy_vars=_occupancy_control_vars(),
        occupancy_param_names=occupancy_param_names,
        atom_site_vars=_atom_site_fractional_control_vars(),
        atom_param_names=atom_param_names,
        debye_x_var=debye_x_var,
        debye_y_var=debye_y_var,
        ordered_scale_var=ordered_structure_scale_var,
        scale_value=float(result.scale),
    )
    _set_ordered_structure_revert_enabled(True)
    progress_label_ordered_structure.config(
        text=(
            f"Ordered structure fit accepted: scale={float(result.scale):.4f}, "
            f"objective reduction={100.0 * float(result.objective_reduction):.1f}%, "
            f"ROIs={int(mask.roi_count)}"
        )
    )
    update_occupancies()


def _geometry_fit_cmd_line(text: str) -> None:
    """Write one geometry-fit status line to the console when available."""

    try:
        print(f"[geometry-fit] {text}", flush=True)
    except Exception:
        pass


def _mosaic_fit_cmd_line(text: str) -> None:
    """Write one mosaic-fit status line to the console when available."""

    try:
        print(f"[mosaic-fit] {text}", flush=True)
    except Exception:
        pass


def _show_geometry_fit_action_notice(action_result) -> None:
    """Show a modal notice when a geometry fit fails or is rejected."""

    notice = gui_geometry_fit.build_geometry_fit_action_notice(action_result)
    if notice is None:
        return
    try:
        if str(notice.level).lower() == "error":
            messagebox.showerror(notice.title, notice.message, parent=root)
        else:
            messagebox.showwarning(notice.title, notice.message, parent=root)
    except Exception:
        pass


geometry_fit_action_bindings_factory = None
on_fit_geometry_click = lambda: None


fit_button_geometry = ttk.Button(
    app_shell_view_state.match_run_frame,
    text="Fit Positions & Geometry",
    command=on_fit_geometry_click,
)
fit_button_geometry.pack(side=tk.TOP, padx=5, pady=2)
fit_button_geometry.config(text="Fit Geometry (LSQ)", command=on_fit_geometry_click)
gui_views.create_geometry_fit_history_controls(
    parent=app_shell_view_state.match_run_frame,
    view_state=geometry_tool_actions_view_state,
    on_undo_fit=_undo_last_geometry_fit,
    on_redo_fit=_redo_last_geometry_fit,
)

live_geometry_preview_var = tk.BooleanVar(value=False)
geometry_tool_actions_runtime = (
    gui_runtime_geometry_interaction.build_runtime_geometry_tool_action_controls_runtime(
        bootstrap_module=gui_bootstrap,
        views_module=gui_views,
        view_state=geometry_tool_actions_view_state,
        on_toggle_manual_pick=_toggle_geometry_manual_pick_mode,
        on_refine_manual_pairs=_refine_current_geometry_manual_pairs,
        on_undo_manual_placement=_undo_last_geometry_manual_placement,
        on_export_manual_pairs=_export_geometry_manual_pairs,
        on_import_manual_pairs=_import_geometry_manual_pairs,
        on_toggle_preview_exclude=(
            geometry_q_group_runtime_callbacks.open_preview_exclusion_window
        ),
        on_clear_manual_pairs=_clear_current_geometry_manual_pairs,
    )
)
gui_runtime_geometry_interaction.initialize_runtime_geometry_interaction_controls(
    fit_actions_parent=app_shell_view_state.fit_actions_frame,
    geometry_tool_actions_runtime=geometry_tool_actions_runtime,
    hkl_lookup_controls_runtime=hkl_lookup_controls_runtime,
    update_fit_history_button_state=_update_geometry_fit_undo_button_state,
    update_manual_pick_button_label=_update_geometry_manual_pick_button_label,
    update_preview_exclude_button_label=_update_geometry_preview_exclude_button_label,
)

qr_cylinder_display_mode_var = tk.StringVar(value=QR_CYLINDER_DISPLAY_MODE_OFF)
geometry_overlay_actions_view_state.qr_cylinder_display_mode_var = qr_cylinder_display_mode_var
if geometry_overlay_actions_view_state.show_qr_cylinder_overlay_var is None:
    geometry_overlay_actions_view_state.show_qr_cylinder_overlay_var = tk.BooleanVar(value=False)
_sync_qr_cylinder_overlay_visibility_var()


def _on_qr_cylinder_display_mode_change(*_args) -> None:
    _sync_qr_cylinder_overlay_visibility_var()
    qr_cylinder_overlay_runtime_toggle()
    _invalidate_and_schedule_update()


qr_cylinder_display_mode_var.trace_add("write", _on_qr_cylinder_display_mode_change)

mosaic_shape_cfg = fit_config.get("mosaic_shape", {}) if isinstance(fit_config, dict) else {}
mosaic_shape_solver_cfg = (
    mosaic_shape_cfg.get("solver", {}) if isinstance(mosaic_shape_cfg, dict) else {}
)
mosaic_fit_initial_values = {
    "fit_sigma_mosaic": bool(mosaic_shape_solver_cfg.get("fit_sigma_mosaic", True)),
    "fit_gamma_mosaic": bool(mosaic_shape_solver_cfg.get("fit_gamma_mosaic", True)),
    "fit_eta": bool(mosaic_shape_solver_cfg.get("fit_eta", True)),
    "fit_theta_i": bool(
        mosaic_shape_solver_cfg.get(
            "fit_theta_i",
            mosaic_shape_solver_cfg.get("refine_theta", True),
        )
    ),
}
if geometry_overlay_actions_view_state.show_geometry_overlays_var is None:
    geometry_overlay_actions_view_state.show_geometry_overlays_var = tk.BooleanVar(value=True)
geometry_overlay_actions_view_state.show_geometry_overlays_checkbutton = None

gui_views.create_geometry_overlay_action_controls(
    parent=app_shell_view_state.match_peak_tools_frame,
    view_state=geometry_overlay_actions_view_state,
    on_toggle_geometry_overlays=_toggle_geometry_overlay_visibility,
    on_fit_mosaic=on_fit_mosaic_click,
    mosaic_fit_initial_values=mosaic_fit_initial_values,
    include_geometry_toggle=False,
    include_fit_button=False,
)
gui_views.create_geometry_overlay_action_controls(
    parent=app_shell_view_state.match_results_frame,
    view_state=geometry_overlay_actions_view_state,
    on_toggle_geometry_overlays=_toggle_geometry_overlay_visibility,
    on_fit_mosaic=on_fit_mosaic_click,
    mosaic_fit_initial_values=mosaic_fit_initial_values,
    include_geometry_toggle=False,
)
integration_range_update_runtime.create_analysis_controls(
    parent=None,
)
gui_views.populate_app_shell_view_switcher(
    view_state=app_shell_view_state,
    on_select=_set_persistent_view_mode,
)
gui_views.bind_app_shell_view_mode_sync(
    view_state=app_shell_view_state,
    show_1d_var=analysis_view_controls_view_state.show_1d_var,
    show_caked_2d_var=analysis_view_controls_view_state.show_caked_2d_var,
    resolve_mode=_current_app_shell_view_mode,
)

# Option to add fractional rods between integer L values. This can be enabled via
# configuration; the GUI control has been removed to reduce interface clutter.


def save_1d_snapshot():
    """
    Save only the final 2D simulated image as a .npy file.
    """
    file_path = filedialog.asksaveasfilename(
        initialdir=get_dir("file_dialog_dir"),
        defaultextension=".npy",
        filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
    )
    if not file_path:
        progress_label.config(text="No file path selected.")
        return

    if not file_path.lower().endswith(".npy"):
        file_path += ".npy"

    # Grab the currently displayed simulated image. ``global_image_buffer`` holds
    # the scaled image that is shown in the GUI so copying it ensures we save
    # exactly what the user sees.  ``simulation_runtime_state.last_1d_integration_data`` may be empty if
    # the simulation hasn't run yet so rely directly on the buffer instead of
    # that cache.
    sim_img = np.asarray(global_image_buffer, dtype=np.float64).copy()
    if sim_img.size == 0:
        progress_label.config(text="No simulated image available to save!")
        return
    try:
        np.save(file_path, sim_img, allow_pickle=False)
        progress_label.config(text=f"Saved simulated image to {file_path}")
    except Exception as e:
        progress_label.config(text=f"Error saving simulated image: {e}")


def save_q_space_representation():
    file_path = filedialog.asksaveasfilename(
        defaultextension=".npy",
        filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
        title="Save Q-Space Snapshot",
    )
    if not file_path:
        return

    param_dict = {
        "theta_initial": _current_effective_theta_initial(strict_count=False),
        "cor_angle": cor_angle_var.get(),
        "gamma": gamma_var.get(),
        "Gamma": Gamma_var.get(),
        "chi": chi_var.get(),
        "psi_z": psi_z_var.get(),
        "zs": zs_var.get(),
        "zb": zb_var.get(),
        "sample_width_m": sample_width_var.get(),
        "sample_length_m": sample_length_var.get(),
        "sample_depth_m": sample_depth_var.get(),
        "debye_x": debye_x_var.get(),
        "debye_y": debye_y_var.get(),
        "corto_detector": corto_detector_var.get(),
        "sigma_mosaic_deg": sigma_mosaic_var.get(),
        "gamma_mosaic_deg": gamma_mosaic_var.get(),
        "eta": eta_var.get(),
        "solve_q_steps": current_solve_q_values().steps,
        "solve_q_rel_tol": current_solve_q_values().rel_tol,
        "solve_q_mode": current_solve_q_values().mode_flag,
        "a": a_var.get(),
        "c": c_var.get(),
        "center_x": center_x_var.get(),
        "center_y": center_y_var.get(),
    }

    sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)

    mosaic_params = {
        "beam_x_array": simulation_runtime_state.profile_cache.get("beam_x_array", []),
        "beam_y_array": simulation_runtime_state.profile_cache.get("beam_y_array", []),
        "theta_array": simulation_runtime_state.profile_cache.get("theta_array", []),
        "phi_array": simulation_runtime_state.profile_cache.get("phi_array", []),
        "wavelength_array": simulation_runtime_state.profile_cache.get("wavelength_array", []),
        "sample_weights": simulation_runtime_state.profile_cache.get("sample_weights"),
        "n2_sample_array": simulation_runtime_state.profile_cache.get("n2_sample_array"),
        "sigma_mosaic_deg": simulation_runtime_state.profile_cache.get("sigma_mosaic_deg", 0.0),
        "gamma_mosaic_deg": simulation_runtime_state.profile_cache.get("gamma_mosaic_deg", 0.0),
        "eta": simulation_runtime_state.profile_cache.get("eta", 0.0),
        "solve_q_steps": simulation_runtime_state.profile_cache.get(
            "solve_q_steps",
            current_solve_q_values().steps,
        ),
        "solve_q_rel_tol": simulation_runtime_state.profile_cache.get(
            "solve_q_rel_tol",
            current_solve_q_values().rel_tol,
        ),
        "solve_q_mode": simulation_runtime_state.profile_cache.get(
            "solve_q_mode",
            current_solve_q_values().mode_flag,
        ),
    }

    image_result, hit_tables, q_data, q_count, _, _ = process_peaks_parallel(
        miller,
        intensities,
        image_size,
        a_var.get(),
        c_var.get(),
        lambda_,
        sim_buffer,
        corto_detector_var.get(),
        gamma_var.get(),
        Gamma_var.get(),
        chi_var.get(),
        psi,
        psi_z_var.get(),
        zs_var.get(),
        zb_var.get(),
        n2,
        mosaic_params["beam_x_array"],
        mosaic_params["beam_y_array"],
        mosaic_params["theta_array"],
        mosaic_params["phi_array"],
        mosaic_params["sigma_mosaic_deg"],
        mosaic_params["gamma_mosaic_deg"],
        mosaic_params["eta"],
        mosaic_params["wavelength_array"],
        debye_x_var.get(),
        debye_y_var.get(),
        [center_x_var.get(), center_y_var.get()],
        _current_effective_theta_initial(strict_count=False),
        cor_angle_var.get(),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        save_flag=1,
        thickness=float(sample_depth_var.get()),
        optics_mode=_current_optics_mode_flag(),
        solve_q_steps=int(mosaic_params["solve_q_steps"]),
        solve_q_rel_tol=float(mosaic_params["solve_q_rel_tol"]),
        solve_q_mode=int(mosaic_params["solve_q_mode"]),
        pixel_size_m=float(pixel_size_m),
        sample_width_m=float(sample_width_var.get()),
        sample_length_m=float(sample_length_var.get()),
        sample_weights=mosaic_params.get("sample_weights"),
        n2_sample_array_override=mosaic_params.get("n2_sample_array"),
    )

    max_positions_local = hit_tables_to_max_positions(hit_tables)

    current_2d_display = global_image_buffer.copy()

    data_dict = {
        "parameters": param_dict,
        "q_data": q_data,
        "q_count": q_count,
        "image_2d": current_2d_display,
    }
    np.save(file_path, data_dict, allow_pickle=True)
    progress_label.config(text=f"Saved Q-Space representation to {file_path}")


def save_1d_permutations():
    pass


def _clear_widget_children(parent) -> None:
    children_getter = getattr(parent, "winfo_children", None)
    if not callable(children_getter):
        return
    try:
        children = list(children_getter())
    except Exception:
        return
    for child in children:
        try:
            child.destroy()
        except Exception:
            pass


def _read_analysis_range_value(var, fallback: float) -> float:
    getter = getattr(var, "get", None)
    if not callable(getter):
        return float(fallback)
    try:
        return float(getter())
    except Exception:
        return float(fallback)


def _read_analysis_cached_range_value(attr_name: str, fallback: float) -> float:
    view_state = globals().get("integration_range_controls_view_state")
    if view_state is None:
        return float(fallback)
    try:
        return float(getattr(view_state, attr_name))
    except Exception:
        return float(fallback)


def _current_analysis_range_values() -> dict[str, float]:
    return {
        "tth_min": _read_analysis_range_value(
            globals().get("tth_min_var"),
            _read_analysis_cached_range_value("tth_min_value", 0.0),
        ),
        "tth_max": _read_analysis_range_value(
            globals().get("tth_max_var"),
            _read_analysis_cached_range_value("tth_max_value", 80.0),
        ),
        "phi_min": _read_analysis_range_value(
            globals().get("phi_min_var"),
            _read_analysis_cached_range_value("phi_min_value", -15.0),
        ),
        "phi_max": _read_analysis_range_value(
            globals().get("phi_max_var"),
            _read_analysis_cached_range_value("phi_max_value", 15.0),
        ),
    }


_ANALYSIS_PEAK_EMPTY_RESULTS_TEXT = "Fit results will appear here."
_ANALYSIS_PEAK_MODEL_COLORS = {
    gui_analysis_peak_tools.PROFILE_GAUSSIAN: "#2a9d8f",
    gui_analysis_peak_tools.PROFILE_LORENTZIAN: "#f4a261",
    gui_analysis_peak_tools.PROFILE_PSEUDO_VOIGT: "#d62828",
}


def _set_analysis_peak_selection_status_text(text: object) -> None:
    status_var = analysis_peak_tools_view_state.selection_status_var
    if status_var is None:
        return
    setter = getattr(status_var, "set", None)
    if callable(setter):
        setter(str(text))


def _set_analysis_peak_fit_results_text(text: object) -> None:
    results_var = analysis_peak_tools_view_state.fit_results_var
    if results_var is None:
        return
    setter = getattr(results_var, "set", None)
    if callable(setter):
        setter(str(text))


def _analysis_peak_selection_status_text() -> str:
    count = len(analysis_peak_selection_state.selected_peaks)
    prefix = (
        "Peak picking active."
        if bool(analysis_peak_selection_state.pick_armed)
        else "Peak picking idle."
    )
    if count <= 0:
        detail = "No peaks selected."
    else:
        detail = f"{count} peak(s) selected."
    if bool(analysis_peak_selection_state.pick_armed):
        detail += " Click peaks in the caked integration region. Right-click to stop."
    return f"{prefix} {detail}"


def _update_analysis_peak_pick_button_label() -> None:
    button = analysis_peak_tools_view_state.pick_button
    if button is None:
        return
    try:
        button.configure(
            text=(
                "Stop Picking Peaks"
                if bool(analysis_peak_selection_state.pick_armed)
                else "Pick Peaks in Region"
            ),
            command=_toggle_analysis_peak_pick_mode,
        )
    except Exception:
        return


def _analysis_peak_fit_results_text() -> str:
    lines = [
        line
        for line in (
            gui_analysis_peak_tools.format_peak_fit_axis_summary(
                "Radial",
                analysis_peak_selection_state.radial_fit_results,
            ),
            gui_analysis_peak_tools.format_peak_fit_axis_summary(
                "Azimuth",
                analysis_peak_selection_state.azimuth_fit_results,
            ),
        )
        if line
    ]
    if not lines:
        return _ANALYSIS_PEAK_EMPTY_RESULTS_TEXT
    return "\n".join(lines)


def _remove_artist_refs(artists: list[object]) -> None:
    while artists:
        artist = artists.pop()
        remover = getattr(artist, "remove", None)
        if callable(remover):
            try:
                remover()
            except Exception:
                pass


def _clear_analysis_peak_overlay_artists(*, redraw: bool) -> None:
    _remove_artist_refs(analysis_peak_selection_state.caked_peak_artists)
    _remove_artist_refs(analysis_peak_selection_state.radial_peak_artists)
    _remove_artist_refs(analysis_peak_selection_state.azimuth_peak_artists)
    _remove_artist_refs(analysis_peak_selection_state.radial_fit_artists)
    _remove_artist_refs(analysis_peak_selection_state.azimuth_fit_artists)
    if redraw:
        try:
            canvas_1d.draw_idle()
        except Exception:
            pass
        if callable(globals().get("_request_overlay_canvas_redraw")):
            _request_overlay_canvas_redraw(force=True)


def _clear_analysis_peak_fit_results(*, redraw: bool, update_text: bool) -> None:
    analysis_peak_selection_state.radial_fit_results.clear()
    analysis_peak_selection_state.azimuth_fit_results.clear()
    _remove_artist_refs(analysis_peak_selection_state.radial_fit_artists)
    _remove_artist_refs(analysis_peak_selection_state.azimuth_fit_artists)
    if update_text:
        _set_analysis_peak_fit_results_text(_ANALYSIS_PEAK_EMPTY_RESULTS_TEXT)
    if redraw:
        try:
            canvas_1d.draw_idle()
        except Exception:
            pass


def _current_analysis_caked_peak_source() -> dict[str, object] | None:
    radial_axis = simulation_runtime_state.last_caked_radial_values
    azimuth_axis = simulation_runtime_state.last_caked_azimuth_values
    if radial_axis is None or azimuth_axis is None:
        return None
    if (
        bool(background_runtime_state.visible)
        and simulation_runtime_state.last_caked_background_image_unscaled is not None
    ):
        return {
            "source": "background",
            "image": np.asarray(
                simulation_runtime_state.last_caked_background_image_unscaled,
                dtype=float,
            ),
            "radial_axis": np.asarray(radial_axis, dtype=float),
            "azimuth_axis": np.asarray(azimuth_axis, dtype=float),
        }
    if simulation_runtime_state.last_caked_image_unscaled is None:
        return None
    return {
        "source": "simulated",
        "image": np.asarray(simulation_runtime_state.last_caked_image_unscaled, dtype=float),
        "radial_axis": np.asarray(radial_axis, dtype=float),
        "azimuth_axis": np.asarray(azimuth_axis, dtype=float),
    }


def _analysis_curve_data(
    axis_kind: str,
    source_preference: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    data = simulation_runtime_state.last_1d_integration_data
    scale = _get_scale_factor_value(default=1.0)

    def _extract(source_name: str) -> tuple[np.ndarray, np.ndarray]:
        source_key = "bg" if str(source_name) == "background" else "sim"
        if str(axis_kind) == "radial":
            x_values = data.get(f"radials_{source_key}")
            y_values = data.get(f"intensities_2theta_{source_key}")
        else:
            x_values = data.get(f"azimuths_{source_key}")
            y_values = data.get(f"intensities_azimuth_{source_key}")
        if x_values is None or y_values is None:
            return np.empty((0,), dtype=float), np.empty((0,), dtype=float)
        x_arr = np.asarray(x_values, dtype=float)
        y_arr = np.asarray(y_values, dtype=float)
        if str(source_name) != "background":
            y_arr = y_arr * float(scale)
        return x_arr, y_arr

    preferred = str(source_preference or "simulated").strip().lower()
    resolved = "background" if preferred == "background" else "simulated"
    x_arr, y_arr = _extract(resolved)
    if x_arr.size > 0 and y_arr.size > 0:
        return x_arr, y_arr, resolved
    fallback = "simulated" if resolved == "background" else "background"
    x_arr, y_arr = _extract(fallback)
    if x_arr.size > 0 and y_arr.size > 0:
        return x_arr, y_arr, fallback
    return np.empty((0,), dtype=float), np.empty((0,), dtype=float), resolved


def _analysis_peak_duplicate_tolerances() -> tuple[float, float]:
    radial_axis = np.asarray(simulation_runtime_state.last_caked_radial_values, dtype=float)
    azimuth_axis = np.asarray(simulation_runtime_state.last_caked_azimuth_values, dtype=float)
    if radial_axis.size >= 2:
        radial_step = float(np.nanmedian(np.abs(np.diff(np.sort(radial_axis)))))
    else:
        radial_step = 0.0
    if azimuth_axis.size >= 2:
        azimuth_step = float(np.nanmedian(np.abs(np.diff(np.sort(azimuth_axis)))))
    else:
        azimuth_step = 0.0
    return max(2.0 * radial_step, 0.05), max(2.0 * azimuth_step, 0.5)


def _analysis_peak_axis_value(
    peak_entry: Mapping[str, object],
    *,
    axis_kind: str,
    axis_values: Sequence[float] | None,
) -> float:
    if str(axis_kind) == "radial":
        return float(peak_entry.get("two_theta_deg", np.nan))
    return gui_analysis_peak_tools.align_angle_to_axis(
        float(peak_entry.get("phi_deg", np.nan)),
        axis_values,
    )


def _analysis_cache_overlay_tables(show_caked: bool) -> list[object]:
    cache_tables = (
        simulation_runtime_state.last_caked_intersection_cache
        if bool(show_caked)
        else simulation_runtime_state.stored_intersection_cache
    )
    if not cache_tables:
        cache_tables = (
            simulation_runtime_state.stored_intersection_cache
            if bool(show_caked)
            else simulation_runtime_state.last_caked_intersection_cache
        )
    return list(cache_tables or [])


def _analysis_cache_overlay_coords(
    table: object,
    *,
    show_caked: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        arr = np.asarray(table, dtype=float)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 4:
        return None

    if bool(show_caked):
        x_vals = np.full(arr.shape[0], np.nan, dtype=float)
        y_vals = np.full(arr.shape[0], np.nan, dtype=float)
        if arr.shape[1] >= 16:
            x_vals[:] = arr[:, 14]
            y_vals[:] = arr[:, 15]
        invalid = ~(np.isfinite(x_vals) & np.isfinite(y_vals))
        if np.any(invalid):
            for idx in np.flatnonzero(invalid):
                col = float(arr[idx, 2])
                row = float(arr[idx, 3])
                converted = _native_detector_coords_to_live_caked_coords(col, row)
                if converted is None:
                    continue
                try:
                    x_vals[idx] = float(converted[0])
                    y_vals[idx] = float(converted[1])
                except Exception:
                    continue
        valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    else:
        x_vals = np.asarray(arr[:, 2], dtype=float)
        y_vals = np.asarray(arr[:, 3], dtype=float)
        valid = np.isfinite(x_vals) & np.isfinite(y_vals)

    if not np.any(valid):
        return None
    return x_vals[valid], y_vals[valid]


def _render_analysis_peak_overlays(*, redraw: bool) -> None:
    _clear_analysis_peak_overlay_artists(redraw=False)

    if _resolved_primary_analysis_display_mode() != "caked":
        if redraw:
            try:
                canvas_1d.draw_idle()
            except Exception:
                pass
            if callable(globals().get("_request_overlay_canvas_redraw")):
                _request_overlay_canvas_redraw(force=True)
        else:
            if callable(globals().get("_request_overlay_canvas_redraw")):
                _request_overlay_canvas_redraw()
        return

    selected_peaks = list(analysis_peak_selection_state.selected_peaks)
    show_caked = True
    cache_tables = _analysis_cache_overlay_tables(show_caked)
    if not selected_peaks and not cache_tables:
        if redraw:
            try:
                canvas_1d.draw_idle()
            except Exception:
                pass
            if callable(globals().get("_request_overlay_canvas_redraw")):
                _request_overlay_canvas_redraw(force=True)
        else:
            if callable(globals().get("_request_overlay_canvas_redraw")):
                _request_overlay_canvas_redraw()
        return

    # Disabled: temporary debug overlay that X-marked cached intersection points.

    if show_caked:
        for idx, peak_entry in enumerate(selected_peaks, start=1):
            try:
                peak_tth = float(peak_entry.get("two_theta_deg"))
                peak_phi = float(peak_entry.get("phi_deg"))
            except Exception:
                continue
            try:
                marker_artist = ax.plot(
                    [peak_tth],
                    [peak_phi],
                    linestyle="none",
                    marker="o",
                    markersize=7.0,
                    markerfacecolor="none",
                    markeredgecolor="#00c2ff",
                    markeredgewidth=1.6,
                    zorder=11,
                )[0]
                label_artist = ax.text(
                    peak_tth,
                    peak_phi,
                    f"P{idx}",
                    color="#00c2ff",
                    fontsize=8,
                    ha="left",
                    va="bottom",
                    zorder=12,
                )
                analysis_peak_selection_state.caked_peak_artists.extend(
                    [marker_artist, label_artist]
                )
            except Exception:
                continue

    axis_specs = (
        (
            "radial",
            ax_1d_radial,
            analysis_peak_selection_state.radial_peak_artists,
            analysis_peak_selection_state.radial_fit_artists,
            analysis_peak_selection_state.radial_fit_results,
        ),
        (
            "azimuth",
            ax_1d_azim,
            analysis_peak_selection_state.azimuth_peak_artists,
            analysis_peak_selection_state.azimuth_fit_artists,
            analysis_peak_selection_state.azimuth_fit_results,
        ),
    )
    for axis_kind, axis_obj, marker_store, fit_store, fit_results in axis_specs:
        for idx, peak_entry in enumerate(selected_peaks, start=1):
            x_curve, y_curve, _resolved_source = _analysis_curve_data(
                axis_kind,
                str(peak_entry.get("source", "simulated")),
            )
            if x_curve.size <= 0 or y_curve.size <= 0:
                continue
            axis_value = _analysis_peak_axis_value(
                peak_entry,
                axis_kind=axis_kind,
                axis_values=x_curve,
            )
            if not np.isfinite(axis_value):
                continue
            y_value = gui_analysis_peak_tools.sample_curve_value(x_curve, y_curve, axis_value)
            try:
                vline_artist = axis_obj.axvline(
                    axis_value,
                    color="#7b2cbf",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.65,
                )
                marker_store.append(vline_artist)
                if np.isfinite(y_value):
                    marker_artist = axis_obj.plot(
                        [axis_value],
                        [y_value],
                        linestyle="none",
                        marker="o",
                        markersize=5.5,
                        color="#7b2cbf",
                        zorder=6,
                    )[0]
                    marker_store.append(marker_artist)
                    label_artist = axis_obj.text(
                        axis_value,
                        y_value,
                        f"P{idx}",
                        color="#7b2cbf",
                        fontsize=8,
                        ha="left",
                        va="bottom",
                    )
                    marker_store.append(label_artist)
            except Exception:
                continue

        for fit_entry in fit_results:
            if not bool(fit_entry.get("success", False)):
                continue
            x_window = np.asarray(fit_entry.get("x_window"), dtype=float)
            y_fit = np.asarray(fit_entry.get("y_fit"), dtype=float)
            if x_window.size <= 0 or y_fit.size <= 0 or x_window.size != y_fit.size:
                continue
            try:
                fit_artist = axis_obj.plot(
                    x_window,
                    y_fit,
                    color=_ANALYSIS_PEAK_MODEL_COLORS.get(
                        str(fit_entry.get("model")),
                        "#6d597a",
                    ),
                    linewidth=1.5,
                    alpha=0.95,
                    zorder=5,
                )[0]
                fit_store.append(fit_artist)
            except Exception:
                continue

    if redraw:
        try:
            canvas_1d.draw_idle()
        except Exception:
            pass
        if callable(globals().get("_request_overlay_canvas_redraw")):
            _request_overlay_canvas_redraw(force=True)
    elif callable(globals().get("_request_overlay_canvas_redraw")):
        _request_overlay_canvas_redraw()


def _render_analysis_peak_tools_controls(parent) -> None:
    _clear_widget_children(parent)
    gui_views.create_analysis_peak_tools_controls(
        parent=parent,
        view_state=analysis_peak_tools_view_state,
        on_toggle_pick_mode=_toggle_analysis_peak_pick_mode,
        on_clear_selection=_clear_selected_analysis_peaks,
        on_fit_selected_peaks=_fit_selected_analysis_peaks,
        pick_enabled=bool(analysis_peak_selection_state.pick_armed),
        fit_gaussian=bool(
            getattr(
                analysis_peak_tools_view_state.fit_gaussian_var,
                "get",
                lambda: False,
            )()
        ),
        fit_lorentzian=bool(
            getattr(
                analysis_peak_tools_view_state.fit_lorentzian_var,
                "get",
                lambda: False,
            )()
        ),
        fit_pseudo_voigt=bool(
            getattr(
                analysis_peak_tools_view_state.fit_pseudo_voigt_var,
                "get",
                lambda: True,
            )()
        ),
        fit_radial=bool(
            getattr(
                analysis_peak_tools_view_state.fit_radial_var,
                "get",
                lambda: True,
            )()
        ),
        fit_azimuth=bool(
            getattr(
                analysis_peak_tools_view_state.fit_azimuth_var,
                "get",
                lambda: True,
            )()
        ),
        selection_status_text=_analysis_peak_selection_status_text(),
        fit_results_text=_analysis_peak_fit_results_text(),
    )


def _render_analysis_export_controls(parent) -> None:
    _clear_widget_children(parent)
    gui_views.create_analysis_export_controls(
        parent=parent,
        view_state=analysis_export_controls_view_state,
        on_save_snapshot=save_1d_snapshot,
        on_save_q_space=save_q_space_representation,
        on_save_1d_grid=save_1d_permutations,
        save_1d_grid_available=False,
    )


def _render_analysis_plot_controls(
    *,
    parent,
    range_values: Mapping[str, float] | None = None,
) -> None:
    global canvas_1d
    global analysis_1d_interaction_bindings
    global tth_min_var, tth_max_var, phi_min_var, phi_max_var
    global tth_min_slider, tth_max_slider, phi_min_slider, phi_max_slider

    values = dict(range_values or _current_analysis_range_values())
    _clear_widget_children(parent)

    analysis_1d_interaction_bindings = None
    plot_host = ttk.Frame(parent)
    plot_host.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    range_host = ttk.Frame(parent, padding=(0, 10, 0, 0))
    range_host.pack(side=tk.TOP, fill=tk.X)

    _mount_analysis_figure(plot_host)

    gui_integration_range_drag.create_runtime_integration_range_controls(
        parent=range_host,
        views_module=gui_views,
        view_state=integration_range_controls_view_state,
        show_1d_var=analysis_view_controls_view_state.show_1d_var,
        tth_min=float(values.get("tth_min", 0.0)),
        tth_max=float(values.get("tth_max", 80.0)),
        phi_min=float(values.get("phi_min", -15.0)),
        phi_max=float(values.get("phi_max", 15.0)),
        schedule_range_update=integration_range_update_runtime_callbacks.schedule_range_update,
    )

    tth_min_var = integration_range_controls_view_state.tth_min_var
    tth_max_var = integration_range_controls_view_state.tth_max_var
    phi_min_var = integration_range_controls_view_state.phi_min_var
    phi_max_var = integration_range_controls_view_state.phi_max_var
    tth_min_slider = integration_range_controls_view_state.tth_min_slider
    tth_max_slider = integration_range_controls_view_state.tth_max_slider
    phi_min_slider = integration_range_controls_view_state.phi_min_slider
    phi_max_slider = integration_range_controls_view_state.phi_max_slider

    if any(
        ref is None
        for ref in (
            tth_min_var,
            tth_max_var,
            phi_min_var,
            phi_max_var,
            tth_min_slider,
            tth_max_slider,
            phi_min_slider,
            phi_max_slider,
        )
    ):
        raise RuntimeError("Integration-range controls did not create the expected widgets.")

    _render_analysis_peak_overlays(redraw=False)
    canvas_1d.draw_idle()


def _restore_analysis_peak_axis_view(*, redraw: bool) -> None:
    preserved_limits = analysis_peak_selection_state.saved_axis_limits
    analysis_peak_selection_state.saved_axis_limits = None
    if preserved_limits is None:
        return
    try:
        gui_canvas_interactions.restore_axis_view(
            ax,
            preserved_limits=preserved_limits,
            default_xlim=ax.get_xlim(),
            default_ylim=ax.get_ylim(),
            preserve=True,
        )
    except Exception:
        return
    if redraw:
        _request_overlay_canvas_redraw(force=True)


def _zoom_to_current_analysis_region(*, redraw: bool) -> None:
    range_values = _current_analysis_range_values()
    tth_min = float(range_values["tth_min"])
    tth_max = float(range_values["tth_max"])
    phi_min = float(range_values["phi_min"])
    phi_max = float(range_values["phi_max"])
    tth_pad = max(0.15, 0.05 * abs(tth_max - tth_min))
    x_lo = min(tth_min, tth_max) - tth_pad
    x_hi = max(tth_min, tth_max) + tth_pad

    azimuth_axis = simulation_runtime_state.last_caked_azimuth_values
    phi_lo = gui_analysis_peak_tools.align_angle_to_axis(phi_min, azimuth_axis)
    phi_hi = gui_analysis_peak_tools.align_angle_to_axis(phi_max, azimuth_axis)
    if phi_hi < phi_lo:
        phi_lo, phi_hi = phi_hi, phi_lo
    phi_pad = max(0.6, 0.05 * abs(phi_hi - phi_lo))
    y_lo = phi_lo - phi_pad
    y_hi = phi_hi + phi_pad
    try:
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
    except Exception:
        return
    if redraw:
        _request_overlay_canvas_redraw(force=True)


def _set_analysis_peak_pick_mode(
    enabled: bool,
    message: str | None = None,
) -> None:
    enabled_flag = bool(enabled)
    current_flag = bool(analysis_peak_selection_state.pick_armed)
    if current_flag == enabled_flag and message is None:
        return

    if enabled_flag:
        _set_geometry_manual_pick_mode(False, message="Manual geometry picking paused.")
        _set_hkl_pick_mode_with_mode_banner(False, message="HKL image-pick paused.")
        _set_geometry_preview_exclude_mode(
            False,
            message="Preview exclusion paused.",
        )

        show_1d_var = analysis_view_controls_view_state.show_1d_var
        if show_1d_var is not None and not bool(show_1d_var.get()):
            show_1d_var.set(True)
            toggle_1d_plots()

        show_caked_var = analysis_view_controls_view_state.show_caked_2d_var
        if show_caked_var is not None and not bool(show_caked_var.get()):
            show_caked_var.set(True)
            toggle_caked_2d()
        if not _active_caked_primary_view():
            enabled_flag = False
            status_text = "Analysis peak picking requires the caked view."
            analysis_peak_selection_state.pick_armed = False
            _update_analysis_peak_pick_button_label()
            _set_analysis_peak_selection_status_text(_analysis_peak_selection_status_text())
            _set_analysis_peak_fit_results_text(_analysis_peak_fit_results_text())
            try:
                progress_label_positions.config(text=status_text)
            except Exception:
                pass
            _render_analysis_peak_overlays(redraw=True)
            return

        if analysis_peak_selection_state.saved_axis_limits is None:
            analysis_peak_selection_state.saved_axis_limits = (
                gui_canvas_interactions.capture_axis_limits(ax)
            )
        analysis_peak_selection_state.pick_armed = True
        _zoom_to_current_analysis_region(redraw=False)
        if callable(globals().get("_refresh_integration_from_cached_results")):
            try:
                _refresh_integration_from_cached_results()
            except Exception:
                pass
        status_text = (
            message or "Analysis peak picking armed. Click peaks in the caked integration region."
        )
    else:
        analysis_peak_selection_state.pick_armed = False
        _restore_analysis_peak_axis_view(redraw=False)
        status_text = message or "Analysis peak picking stopped."

    _update_analysis_peak_pick_button_label()
    _set_analysis_peak_selection_status_text(_analysis_peak_selection_status_text())
    _set_analysis_peak_fit_results_text(_analysis_peak_fit_results_text())
    refresh_mode_banner = globals().get("_refresh_interaction_mode_banner")
    if callable(refresh_mode_banner):
        refresh_mode_banner()
    try:
        progress_label_positions.config(text=status_text)
    except Exception:
        pass
    _render_analysis_peak_overlays(redraw=True)


def _toggle_analysis_peak_pick_mode() -> None:
    _set_analysis_peak_pick_mode(
        not bool(analysis_peak_selection_state.pick_armed),
    )


def _clear_selected_analysis_peaks() -> None:
    analysis_peak_selection_state.selected_peaks.clear()
    _clear_analysis_peak_fit_results(redraw=False, update_text=True)
    _set_analysis_peak_selection_status_text(_analysis_peak_selection_status_text())
    try:
        progress_label_positions.config(text="Cleared analysis peak selections.")
    except Exception:
        pass
    _render_analysis_peak_overlays(redraw=True)


def _select_analysis_peak_from_canvas_click(
    two_theta_deg: float,
    phi_deg: float,
) -> bool:
    range_values = _current_analysis_range_values()
    if not gui_analysis_peak_tools.integration_region_contains(
        two_theta_deg,
        phi_deg,
        tth_min=float(range_values["tth_min"]),
        tth_max=float(range_values["tth_max"]),
        phi_min=float(range_values["phi_min"]),
        phi_max=float(range_values["phi_max"]),
    ):
        try:
            progress_label_positions.config(
                text="Click inside the current integration region to pick a peak."
            )
        except Exception:
            pass
        return True

    source_payload = _current_analysis_caked_peak_source()
    if not isinstance(source_payload, dict):
        try:
            progress_label_positions.config(
                text="No caked image is available for analysis peak picking."
            )
        except Exception:
            pass
        return True

    source_image = np.asarray(source_payload.get("image"), dtype=float)
    radial_axis = np.asarray(source_payload.get("radial_axis"), dtype=float)
    azimuth_axis = np.asarray(source_payload.get("azimuth_axis"), dtype=float)
    if source_image.ndim != 2 or radial_axis.size <= 0 or azimuth_axis.size <= 0:
        try:
            progress_label_positions.config(
                text="The current caked image cannot be used for analysis peak picking."
            )
        except Exception:
            pass
        return True

    match_cfg = {
        "search_radius_px": 18.0,
        "local_max_size_px": 5,
        "smooth_sigma_px": 1.4,
        "climb_sigma_px": 0.8,
        "min_prominence_sigma": 1.5,
        "min_match_prominence_sigma": 1.5,
        "max_candidate_peaks": 10,
    }
    refined_tth = float(two_theta_deg)
    refined_phi = float(phi_deg)
    try:
        match_cfg, background_context = _auto_match_background_context(
            source_image,
            match_cfg,
        )
        refined_tth, refined_phi = gui_manual_geometry.geometry_manual_refine_preview_point(
            None,
            float(two_theta_deg),
            float(phi_deg),
            display_background=source_image,
            cache_data={
                "match_config": dict(match_cfg),
                "background_context": background_context,
            },
            use_caked_space=True,
            radial_axis=radial_axis,
            azimuth_axis=azimuth_axis,
            match_simulated_peaks_to_peak_context=match_simulated_peaks_to_peak_context,
            caked_axis_to_image_index_fn=_caked_axis_to_image_index,
            caked_image_index_to_axis_fn=_caked_image_index_to_axis,
            refine_caked_peak_center_fn=_refine_caked_peak_center,
        )
    except Exception:
        refined_tth = float(two_theta_deg)
        refined_phi = float(phi_deg)

    if not (np.isfinite(refined_tth) and np.isfinite(refined_phi)):
        try:
            progress_label_positions.config(text="Peak picking failed for that click.")
        except Exception:
            pass
        return True

    radial_tol, azimuth_tol = _analysis_peak_duplicate_tolerances()
    match_index = gui_analysis_peak_tools.match_selected_peak_index(
        analysis_peak_selection_state.selected_peaks,
        two_theta_deg=float(refined_tth),
        phi_deg=float(refined_phi),
        radial_tolerance_deg=float(radial_tol),
        azimuth_tolerance_deg=float(azimuth_tol),
    )

    if match_index is not None:
        analysis_peak_selection_state.selected_peaks.pop(int(match_index))
        status_text = (
            f"Removed peak near 2theta={float(refined_tth):.4f} deg, "
            f"phi={float(gui_analysis_peak_tools.wrap_angle_degrees(refined_phi)):.4f} deg."
        )
    else:
        analysis_peak_selection_state.selected_peaks.append(
            {
                "two_theta_deg": float(refined_tth),
                "phi_deg": float(gui_analysis_peak_tools.wrap_angle_degrees(refined_phi)),
                "source": str(source_payload.get("source", "simulated")),
                "raw_two_theta_deg": float(two_theta_deg),
                "raw_phi_deg": float(phi_deg),
            }
        )
        status_text = (
            f"Selected peak {len(analysis_peak_selection_state.selected_peaks)} at "
            f"2theta={float(refined_tth):.4f} deg, "
            f"phi={float(gui_analysis_peak_tools.wrap_angle_degrees(refined_phi)):.4f} deg."
        )

    _clear_analysis_peak_fit_results(redraw=False, update_text=True)
    _set_analysis_peak_selection_status_text(_analysis_peak_selection_status_text())
    try:
        progress_label_positions.config(text=status_text)
    except Exception:
        pass
    _render_analysis_peak_overlays(redraw=True)
    return True


def _fit_selected_analysis_peaks() -> None:
    selected_peaks = list(analysis_peak_selection_state.selected_peaks)
    if not selected_peaks:
        try:
            progress_label_positions.config(text="Select one or more peaks before fitting.")
        except Exception:
            pass
        return

    model_specs = (
        (
            getattr(analysis_peak_tools_view_state.fit_gaussian_var, "get", lambda: False)(),
            gui_analysis_peak_tools.PROFILE_GAUSSIAN,
        ),
        (
            getattr(analysis_peak_tools_view_state.fit_lorentzian_var, "get", lambda: False)(),
            gui_analysis_peak_tools.PROFILE_LORENTZIAN,
        ),
        (
            getattr(analysis_peak_tools_view_state.fit_pseudo_voigt_var, "get", lambda: True)(),
            gui_analysis_peak_tools.PROFILE_PSEUDO_VOIGT,
        ),
    )
    models = [model for enabled, model in model_specs if bool(enabled)]
    if not models:
        try:
            progress_label_positions.config(text="Choose at least one peak-profile model to fit.")
        except Exception:
            pass
        return

    axes_to_fit = []
    if bool(getattr(analysis_peak_tools_view_state.fit_radial_var, "get", lambda: True)()):
        axes_to_fit.append("radial")
    if bool(getattr(analysis_peak_tools_view_state.fit_azimuth_var, "get", lambda: True)()):
        axes_to_fit.append("azimuth")
    if not axes_to_fit:
        try:
            progress_label_positions.config(
                text="Choose the radial plot, azimuth plot, or both before fitting."
            )
        except Exception:
            pass
        return

    _clear_analysis_peak_fit_results(redraw=False, update_text=False)

    total_success = 0
    total_attempts = 0
    for axis_kind in axes_to_fit:
        axis_results: list[dict[str, object]] = []
        for idx, peak_entry in enumerate(selected_peaks):
            x_curve, y_curve, resolved_source = _analysis_curve_data(
                axis_kind,
                str(peak_entry.get("source", "simulated")),
            )
            if x_curve.size < 7 or y_curve.size < 7:
                continue
            axis_centers = np.asarray(
                [
                    _analysis_peak_axis_value(
                        entry,
                        axis_kind=axis_kind,
                        axis_values=x_curve,
                    )
                    for entry in selected_peaks
                ],
                dtype=float,
            )
            center_guess = float(axis_centers[idx])
            window_half_width = gui_analysis_peak_tools.recommended_peak_window_half_width(
                axis_centers,
                idx,
                axis_values=x_curve,
                axis_kind=axis_kind,
                region_bounds=(float(x_curve[0]), float(x_curve[-1])),
            )
            for model in models:
                fit_result = gui_analysis_peak_tools.fit_peak_profile(
                    x_curve,
                    y_curve,
                    center_guess=center_guess,
                    model=model,
                    window_half_width=float(window_half_width),
                )
                fit_entry = dict(fit_result)
                fit_entry.update(
                    {
                        "peak_index": int(idx) + 1,
                        "axis_kind": axis_kind,
                        "selected_axis_value": float(center_guess),
                        "two_theta_deg": float(peak_entry.get("two_theta_deg", np.nan)),
                        "phi_deg": float(peak_entry.get("phi_deg", np.nan)),
                        "curve_source": resolved_source,
                    }
                )
                axis_results.append(fit_entry)
                total_attempts += 1
                if bool(fit_entry.get("success", False)):
                    total_success += 1

        if axis_kind == "radial":
            analysis_peak_selection_state.radial_fit_results = axis_results
        else:
            analysis_peak_selection_state.azimuth_fit_results = axis_results

    _set_analysis_peak_fit_results_text(_analysis_peak_fit_results_text())
    if total_attempts <= 0:
        try:
            progress_label_positions.config(
                text="No valid 1D data were available for the selected peak fits."
            )
        except Exception:
            pass
    else:
        try:
            progress_label_positions.config(
                text=f"Finished {total_success}/{total_attempts} analysis peak fits."
            )
        except Exception:
            pass
    _render_analysis_peak_overlays(redraw=True)


def _show_analysis_lazy_placeholder(parent, text: str, *, wraplength: int = 360) -> None:
    _clear_widget_children(parent)
    ttk.Label(
        parent,
        text=text,
        justify=tk.LEFT,
        wraplength=wraplength,
    ).pack(fill=tk.BOTH, expand=True, padx=8, pady=8)


def _show_analysis_tab_lazy_placeholders() -> None:
    _show_analysis_lazy_placeholder(
        app_shell_view_state.analysis_exports_frame,
        "Analysis export controls load when the Analyze tab is first opened.",
    )
    _show_analysis_lazy_placeholder(
        app_shell_view_state.analysis_peak_tools_frame,
        "Analysis peak tools load when the Analyze tab is first opened.",
    )
    _show_analysis_lazy_placeholder(
        app_shell_view_state.plot_frame_1d,
        "Analysis plots load on first use.",
        wraplength=420,
    )


def _show_analysis_tab_detached_placeholders() -> None:
    _clear_widget_children(app_shell_view_state.analysis_exports_frame)
    ttk.Label(
        app_shell_view_state.analysis_exports_frame,
        text="Exports moved to the detached Analyze window.",
        justify=tk.LEFT,
        wraplength=360,
    ).pack(fill=tk.X, padx=6, pady=(4, 6))

    _clear_widget_children(app_shell_view_state.analysis_peak_tools_frame)
    ttk.Label(
        app_shell_view_state.analysis_peak_tools_frame,
        text="Peak picking and fitting controls moved to the detached Analyze window.",
        justify=tk.LEFT,
        wraplength=360,
    ).pack(fill=tk.X, padx=6, pady=(4, 6))

    _clear_widget_children(app_shell_view_state.plot_frame_1d)
    ttk.Label(
        app_shell_view_state.plot_frame_1d,
        text=(
            "Integration figures are open in a detached Analyze window.\n"
            "Use the button above or close that window to dock them back here."
        ),
        justify=tk.CENTER,
        anchor=tk.CENTER,
        wraplength=420,
    ).pack(fill=tk.BOTH, expand=True, padx=12, pady=12)


def _set_analysis_popout_button_state(*, detached: bool) -> None:
    button = app_shell_view_state.analysis_popout_button
    if button is None:
        return
    try:
        button.configure(
            text=("Dock Analyze Window" if detached else "Pop Out Analyze Window"),
            command=_toggle_analysis_popout,
        )
    except Exception:
        pass


def ensure_analysis_surfaces_initialized() -> None:
    global analysis_surfaces_initialized

    if analysis_surfaces_initialized:
        return
    _render_analysis_export_controls(app_shell_view_state.analysis_exports_frame)
    _render_analysis_peak_tools_controls(app_shell_view_state.analysis_peak_tools_frame)
    _render_analysis_plot_controls(parent=app_shell_view_state.plot_frame_1d)
    analysis_surfaces_initialized = True
    _set_analysis_popout_button_state(detached=False)


def _dock_analysis_window(*, restore_tab: bool = True) -> None:
    global analysis_surfaces_initialized

    range_values = _current_analysis_range_values()
    gui_views.close_analysis_popout_window(analysis_popout_view_state)
    if restore_tab:
        _render_analysis_export_controls(app_shell_view_state.analysis_exports_frame)
        _render_analysis_peak_tools_controls(app_shell_view_state.analysis_peak_tools_frame)
        _render_analysis_plot_controls(
            parent=app_shell_view_state.plot_frame_1d,
            range_values=range_values,
        )
        _refresh_analysis_integration_if_visible()
        analysis_surfaces_initialized = True
    _set_analysis_popout_button_state(detached=False)


def _pop_out_analysis_window() -> None:
    global analysis_surfaces_initialized

    range_values = _current_analysis_range_values()
    opened = gui_views.open_analysis_popout_window(
        root=root,
        view_state=analysis_popout_view_state,
        on_close=_dock_analysis_window,
    )
    if not opened and gui_views.analysis_popout_window_open(analysis_popout_view_state):
        return
    try:
        _show_analysis_tab_detached_placeholders()
        _render_analysis_export_controls(analysis_popout_view_state.exports_frame)
        _render_analysis_peak_tools_controls(analysis_popout_view_state.peak_tools_frame)
        _render_analysis_plot_controls(
            parent=analysis_popout_view_state.plot_frame,
            range_values=range_values,
        )
        _refresh_analysis_integration_if_visible()
        analysis_surfaces_initialized = True
    except Exception:
        gui_views.close_analysis_popout_window(analysis_popout_view_state)
        _render_analysis_export_controls(app_shell_view_state.analysis_exports_frame)
        _render_analysis_peak_tools_controls(app_shell_view_state.analysis_peak_tools_frame)
        _render_analysis_plot_controls(
            parent=app_shell_view_state.plot_frame_1d,
            range_values=range_values,
        )
        analysis_surfaces_initialized = True
        _set_analysis_popout_button_state(detached=False)
        raise
    _set_analysis_popout_button_state(detached=True)


def _toggle_analysis_popout() -> None:
    if gui_views.analysis_popout_window_open(analysis_popout_view_state):
        _dock_analysis_window()
        return
    _pop_out_analysis_window()


def _handle_analysis_integration_visibility_change(*_args) -> None:
    if _analysis_integration_outputs_visible():
        ensure_analysis_surfaces_initialized()
    _refresh_analysis_integration_if_visible()


_analysis_tab_trace_add = getattr(app_shell_view_state.control_tab_var, "trace_add", None)
if callable(_analysis_tab_trace_add):
    _analysis_tab_trace_add("write", _handle_analysis_integration_visibility_change)


_show_analysis_tab_lazy_placeholders()
_set_analysis_popout_button_state(detached=False)


def run_debug_simulation():

    gamma_val = float(gamma_var.get())
    Gamma_val = float(Gamma_var.get())
    chi_val = float(chi_var.get())
    zs_val = float(zs_var.get())
    zb_val = float(zb_var.get())
    a_val = float(a_var.get())
    c_val = float(c_var.get())
    theta_val = float(_current_effective_theta_initial(strict_count=False))
    dx_val = float(debye_x_var.get())
    dy_val = float(debye_y_var.get())
    corto_val = float(corto_detector_var.get())
    cx_val = float(center_x_var.get())
    cy_val = float(center_y_var.get())

    mosaic_params = {
        "beam_x_array": simulation_runtime_state.profile_cache.get("beam_x_array", []),
        "beam_y_array": simulation_runtime_state.profile_cache.get("beam_y_array", []),
        "theta_array": simulation_runtime_state.profile_cache.get("theta_array", []),
        "phi_array": simulation_runtime_state.profile_cache.get("phi_array", []),
        "wavelength_array": simulation_runtime_state.profile_cache.get("wavelength_array", []),
        "sigma_mosaic_deg": simulation_runtime_state.profile_cache.get("sigma_mosaic_deg", 0.0),
        "gamma_mosaic_deg": simulation_runtime_state.profile_cache.get("gamma_mosaic_deg", 0.0),
        "eta": simulation_runtime_state.profile_cache.get("eta", 0.0),
        "solve_q_steps": simulation_runtime_state.profile_cache.get(
            "solve_q_steps",
            current_solve_q_values().steps,
        ),
        "solve_q_rel_tol": simulation_runtime_state.profile_cache.get(
            "solve_q_rel_tol",
            current_solve_q_values().rel_tol,
        ),
        "solve_q_mode": simulation_runtime_state.profile_cache.get(
            "solve_q_mode",
            current_solve_q_values().mode_flag,
        ),
    }

    sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)
    image_out, maxpos, qdata, qcount = process_peaks_parallel_debug(
        miller,
        intensities,
        image_size,
        a_val,
        c_val,
        lambda_,
        sim_buffer,
        corto_val,
        gamma_val,
        Gamma_val,
        chi_val,
        psi,
        psi_z_var.get(),
        zs_val,
        zb_val,
        n2,
        mosaic_params["beam_x_array"],
        mosaic_params["beam_y_array"],
        mosaic_params["theta_array"],
        mosaic_params["phi_array"],
        mosaic_params["sigma_mosaic_deg"],
        mosaic_params["gamma_mosaic_deg"],
        mosaic_params["eta"],
        mosaic_params["wavelength_array"],
        dx_val,
        dy_val,
        [cx_val, cy_val],
        theta_val,
        cor_angle_var.get(),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        save_flag=1,
    )

    dump_debug_log()
    progress_label.config(text="Debug simulation complete. Log saved.")


gui_views.populate_stacked_button_group(
    (
        workspace_panels_view_state.workspace_debug_frame.frame
        if workspace_panels_view_state.workspace_debug_frame is not None
        else workspace_panels_view_state.workspace_actions_frame
    ),
    [
        ("Run Debug Simulation", run_debug_simulation),
        ("Force Update", lambda: update_occupancies()),
    ],
)

# Group related sliders in collapsible sections so the interface remains
# manageable as more controls are added.
geo_frame = CollapsibleFrame(
    app_shell_view_state.left_col,
    text="Geometry",
    expanded=True,
)
geo_frame.pack(fill=tk.X, padx=5, pady=5)

debye_frame = CollapsibleFrame(app_shell_view_state.right_col, text="Debye Parameters")
debye_frame.pack(fill=tk.X, padx=5, pady=5)

detector_frame = CollapsibleFrame(app_shell_view_state.left_col, text="Detector")
detector_frame.pack(fill=tk.X, padx=5, pady=5)

lattice_frame = CollapsibleFrame(
    app_shell_view_state.left_col,
    text="Lattice Parameters",
    expanded=True,
)
lattice_frame.pack(fill=tk.X, padx=5, pady=5)

mosaic_frame = CollapsibleFrame(
    app_shell_view_state.left_col,
    text="Mosaic Broadening",
)
mosaic_frame.pack(fill=tk.X, padx=5, pady=5)

sampling_pruning_frame = CollapsibleFrame(
    app_shell_view_state.right_col,
    text="Sampling, Optics, Arc Integration && Pruning",
)
sampling_pruning_frame.pack(fill=tk.X, padx=5, pady=5)

legacy_resolution_options = [*legacy_resolution_sample_counts.keys(), CUSTOM_SAMPLING_OPTION]
initial_resolution = str(defaults.get("sampling_resolution", CUSTOM_SAMPLING_OPTION))
if initial_resolution not in legacy_resolution_options:
    initial_resolution = CUSTOM_SAMPLING_OPTION


def _parse_sample_count(raw_value, fallback):
    return gui_controllers.normalize_sample_count(
        raw_value,
        fallback,
        minimum=MIN_RANDOM_SAMPLE_COUNT,
        maximum=MAX_RANDOM_SAMPLE_COUNT,
    )


def _default_random_sample_count() -> int:
    configured_count = defaults.get("sampling_count")
    if configured_count is not None:
        return _parse_sample_count(configured_count, DEFAULT_RANDOM_SAMPLE_COUNT)

    legacy_choice = str(defaults.get("sampling_resolution", CUSTOM_SAMPLING_OPTION))
    legacy_count = legacy_resolution_sample_counts.get(
        legacy_choice,
        DEFAULT_RANDOM_SAMPLE_COUNT,
    )
    return _parse_sample_count(legacy_count, DEFAULT_RANDOM_SAMPLE_COUNT)


def _parse_rod_points_per_gz(raw_value, fallback):
    return gui_controllers.normalize_rod_points_per_gz(raw_value, fallback)


def _current_rod_points_per_gz(default=None):
    fallback = gui_controllers.default_rod_points_per_gz(
        defaults.get("c", cv),
    )
    if default is not None:
        fallback = _parse_rod_points_per_gz(default, fallback)
    rod_var = sampling_optics_controls_view_state.rod_points_per_gz_var
    raw_value = rod_var.get() if rod_var is not None else fallback
    return _parse_rod_points_per_gz(raw_value, fallback)


def _current_random_sample_count(default=None):
    fallback = _parse_sample_count(
        default if default is not None else _default_random_sample_count(),
        DEFAULT_RANDOM_SAMPLE_COUNT,
    )
    sample_count_var = sampling_optics_controls_view_state.sample_count_var
    raw_value = sample_count_var.get() if sample_count_var is not None else fallback
    return _parse_sample_count(raw_value, fallback)


def _current_active_sample_count() -> int:
    return _current_random_sample_count(default=max(1, simulation_runtime_state.num_samples))


def _sync_active_sample_count() -> int:
    simulation_runtime_state.num_samples = int(max(1, _current_active_sample_count()))
    return int(simulation_runtime_state.num_samples)


def _set_random_sample_controls_state():
    gui_views.set_sampling_sample_count_controls_enabled(
        sampling_optics_controls_view_state,
        enabled=True,
    )


def _set_sampling_method_controls_state():
    _set_random_sample_controls_state()


def _refresh_resolution_display():
    random_summary_text = gui_controllers.format_sampling_count_summary(
        _current_random_sample_count()
    )
    gui_views.set_sampling_sample_count_text(
        sampling_optics_controls_view_state,
        random_summary_text,
    )
    rod_points_per_gz = _current_rod_points_per_gz(
        default=defaults.get("rod_points_per_gz"),
    )
    gui_views.set_sampling_rod_points_per_gz_text(
        sampling_optics_controls_view_state,
        gui_controllers.format_rod_points_per_gz(rod_points_per_gz),
    )
    gui_views.set_sampling_rod_point_total_text(
        sampling_optics_controls_view_state,
        gui_controllers.format_longest_rod_point_summary(
            rod_points_per_gz,
            two_theta_max=two_theta_range[1],
            lambda_angstrom=lambda_,
        ),
    )
    try:
        optics_summary = _normalize_optics_mode_label(optics_mode_var.get())
    except Exception:
        optics_summary = _normalize_optics_mode_label(defaults.get("optics_mode", "fast"))
    solve_q_mode_summary = ""
    solve_q_mode_var = getattr(
        structure_factor_pruning_controls_view_state,
        "solve_q_mode_var",
        None,
    )
    if solve_q_mode_var is not None:
        try:
            solve_q_mode_summary = (
                gui_structure_factor_pruning.normalize_runtime_solve_q_mode_label(
                    solve_q_mode_var.get()
                )
            )
        except Exception:
            solve_q_mode_summary = ""
    summary_parts = [
        random_summary_text,
        f"rods {rod_points_per_gz:,}/Gz",
        f"optics {optics_summary}",
    ]
    if solve_q_mode_summary:
        summary_parts.append(f"arc {solve_q_mode_summary}")
    gui_views.set_collapsible_header_summary(
        sampling_pruning_frame,
        " | ".join(summary_parts),
    )


def _normalize_random_sample_count_control(default=None) -> int:
    sample_count_var = sampling_optics_controls_view_state.sample_count_var
    normalized = _current_random_sample_count(default=default)
    if sample_count_var is not None:
        try:
            current_value = int(round(float(sample_count_var.get())))
        except Exception:
            current_value = normalized
        if current_value != normalized:
            sample_count_var.set(normalized)
    defaults["sampling_count"] = int(normalized)
    if resolution_var.get() != CUSTOM_SAMPLING_OPTION:
        resolution_var.set(CUSTOM_SAMPLING_OPTION)
    return int(normalized)


def _preview_random_sample_count(_value=None):
    _normalize_random_sample_count_control(default=_default_random_sample_count())
    _refresh_resolution_display()


def _apply_random_sample_count(*, trigger_update=True):
    previous_num_samples = int(simulation_runtime_state.num_samples)
    _normalize_random_sample_count_control(default=max(1, simulation_runtime_state.num_samples))
    _sync_active_sample_count()
    _refresh_resolution_display()
    if trigger_update and simulation_runtime_state.num_samples != previous_num_samples:
        update_mosaic_cache()
        schedule_update()


def _current_structure_model_rebuild_inputs():
    try:
        new_occ = [float(var.get()) for var in _occupancy_control_vars()]
    except (tk.TclError, ValueError):
        new_occ = list(occ)
    if not all(np.isfinite(v) for v in new_occ):
        new_occ = list(occ)
    new_occ = gui_controllers.clamp_site_occupancy_values(
        new_occ,
        fallback_values=occ,
    )

    try:
        p_vals = [float(p0_var.get()), float(p1_var.get()), float(p2_var.get())]
    except (tk.TclError, ValueError):
        p_vals = list(
            structure_model_state.last_p_triplet or [defaults["p0"], defaults["p1"], defaults["p2"]]
        )

    try:
        w_raw = [float(w0_var.get()), float(w1_var.get()), float(w2_var.get())]
    except (tk.TclError, ValueError):
        w_raw = list(structure_model_state.last_weights or [1.0, 0.0, 0.0])
    weights = gui_controllers.normalize_stacking_weight_values(w_raw)
    return new_occ, p_vals, weights


def _preview_rod_points_per_gz(_value=None):
    rod_var = sampling_optics_controls_view_state.rod_points_per_gz_var
    normalized = _current_rod_points_per_gz(
        default=defaults.get("rod_points_per_gz"),
    )
    if rod_var is not None:
        try:
            current_value = int(round(float(rod_var.get())))
        except Exception:
            current_value = normalized
        if current_value != normalized:
            rod_var.set(normalized)
    _refresh_resolution_display()


def _apply_rod_points_per_gz(*, trigger_update=True):
    rod_var = sampling_optics_controls_view_state.rod_points_per_gz_var
    normalized = _current_rod_points_per_gz(
        default=defaults.get("rod_points_per_gz"),
    )
    if rod_var is not None:
        rod_var.set(normalized)
    defaults["rod_points_per_gz"] = int(normalized)
    _refresh_resolution_display()

    if int(normalized) == int(getattr(structure_model_state, "last_rod_points_per_gz", normalized)):
        return
    if (
        globals().get("a_var") is None
        or globals().get("c_var") is None
        or globals().get("p0_var") is None
        or globals().get("p1_var") is None
        or globals().get("p2_var") is None
    ):
        return

    new_occ, p_vals, weights = _current_structure_model_rebuild_inputs()
    _rebuild_diffraction_inputs(
        new_occ,
        p_vals,
        weights,
        a_var.get(),
        c_var.get(),
        force=False,
        trigger_update=trigger_update,
    )


def ensure_valid_resolution_choice():
    normalized = gui_controllers.normalize_sampling_resolution_choice(
        resolution_var.get(),
        allowed_options=legacy_resolution_options,
        fallback=CUSTOM_SAMPLING_OPTION,
    )
    if normalized != CUSTOM_SAMPLING_OPTION:
        mapped_count = legacy_resolution_sample_counts.get(
            normalized,
            _default_random_sample_count(),
        )
        sample_count_var = sampling_optics_controls_view_state.sample_count_var
        if sample_count_var is not None:
            sample_count_var.set(_parse_sample_count(mapped_count, _default_random_sample_count()))
        normalized = CUSTOM_SAMPLING_OPTION
    if resolution_var.get() != normalized:
        resolution_var.set(normalized)
    _normalize_random_sample_count_control(default=_default_random_sample_count())
    _set_sampling_method_controls_state()
    _sync_active_sample_count()
    _refresh_resolution_display()


if initial_resolution != CUSTOM_SAMPLING_OPTION:
    initial_sample_count = _parse_sample_count(
        legacy_resolution_sample_counts.get(
            initial_resolution,
            _default_random_sample_count(),
        ),
        _default_random_sample_count(),
    )
else:
    initial_sample_count = _default_random_sample_count()

initial_rod_points_per_gz = _current_rod_points_per_gz(
    default=defaults.get("rod_points_per_gz"),
)

gui_views.create_sampling_optics_controls(
    parent=sampling_pruning_frame.frame,
    view_state=sampling_optics_controls_view_state,
    sample_count_value=initial_sample_count,
    sample_count_min=MIN_RANDOM_SAMPLE_COUNT,
    sample_count_max=MAX_RANDOM_SAMPLE_COUNT,
    sample_count_text=gui_controllers.format_sampling_count_summary(initial_sample_count),
    rod_points_per_gz_value=initial_rod_points_per_gz,
    rod_points_per_gz_min=gui_controllers.ROD_POINTS_PER_GZ_MIN,
    rod_points_per_gz_max=gui_controllers.ROD_POINTS_PER_GZ_MAX,
    rod_points_per_gz_text=gui_controllers.format_rod_points_per_gz(initial_rod_points_per_gz),
    rod_point_total_text=gui_controllers.format_longest_rod_point_summary(
        initial_rod_points_per_gz,
        two_theta_max=two_theta_range[1],
        lambda_angstrom=lambda_,
    ),
    optics_mode_text=_normalize_optics_mode_label(defaults.get("optics_mode", "fast")),
    on_sample_count_slide=_preview_random_sample_count,
    on_commit_sample_count=lambda _event: _apply_random_sample_count(trigger_update=True),
    on_rod_points_per_gz_slide=_preview_rod_points_per_gz,
    on_commit_rod_points_per_gz=lambda _event: _apply_rod_points_per_gz(trigger_update=True),
)
custom_samples_var = sampling_optics_controls_view_state.sample_count_var
resolution_var = tk.StringVar(value=CUSTOM_SAMPLING_OPTION)
sample_count_var = sampling_optics_controls_view_state.sample_count_var
sample_count_scale = sampling_optics_controls_view_state.sample_count_scale
rod_points_per_gz_var = sampling_optics_controls_view_state.rod_points_per_gz_var
optics_mode_var = sampling_optics_controls_view_state.optics_mode_var


def on_resolution_option_change(*_):
    ensure_valid_resolution_choice()


_set_sampling_method_controls_state()
_sync_active_sample_count()
_refresh_resolution_display()
_apply_rod_points_per_gz(trigger_update=False)
resolution_var.trace_add("write", on_resolution_option_change)
sample_count_trace_add = getattr(sample_count_var, "trace_add", None)
if callable(sample_count_trace_add):
    sample_count_trace_add("write", lambda *_args: _refresh_resolution_display())


def on_optics_mode_change(*_):
    _refresh_resolution_display()
    _invalidate_simulation_cache()
    schedule_update()


optics_mode_var.trace_add("write", on_optics_mode_change)

structure_factor_pruning_controls_runtime.create_controls(parent=sampling_pruning_frame.frame)
sf_prune_bias_var = structure_factor_pruning_controls_view_state.sf_prune_bias_var
sf_prune_bias_scale = structure_factor_pruning_controls_view_state.sf_prune_bias_scale
sf_prune_status_var = structure_factor_pruning_controls_view_state.sf_prune_status_var
solve_q_mode_var = structure_factor_pruning_controls_view_state.solve_q_mode_var
solve_q_steps_var = structure_factor_pruning_controls_view_state.solve_q_steps_var
solve_q_steps_scale = structure_factor_pruning_controls_view_state.solve_q_steps_scale
solve_q_rel_tol_var = structure_factor_pruning_controls_view_state.solve_q_rel_tol_var
solve_q_rel_tol_scale = structure_factor_pruning_controls_view_state.solve_q_rel_tol_scale
trace_add = getattr(solve_q_mode_var, "trace_add", None)
if callable(trace_add):
    trace_add("write", lambda *_args: _refresh_resolution_display())
_refresh_resolution_display()

center_frame = CollapsibleFrame(app_shell_view_state.left_col, text="Beam Controls")
center_frame.pack(fill=tk.X, padx=5, pady=5)

gui_views.create_beam_mosaic_parameter_sliders(
    geometry_parent=geo_frame.frame,
    debye_parent=debye_frame.frame,
    detector_parent=detector_frame.frame,
    lattice_parent=lattice_frame.frame,
    mosaic_parent=mosaic_frame.frame,
    beam_parent=center_frame.frame,
    view_state=beam_mosaic_parameter_sliders_view_state,
    image_size=float(image_size),
    values={
        "theta_initial": defaults["theta_initial"],
        "cor_angle": defaults["cor_angle"],
        "gamma": defaults["gamma"],
        "Gamma": defaults["Gamma"],
        "chi": defaults["chi"],
        "psi_z": defaults["psi_z"],
        "zs": defaults["zs"],
        "zb": defaults["zb"],
        "sample_width_m": defaults["sample_width_m"],
        "sample_length_m": defaults["sample_length_m"],
        "sample_depth_m": defaults["sample_depth_m"],
        "debye_x": defaults["debye_x"],
        "debye_y": defaults["debye_y"],
        "corto_detector": defaults["corto_detector"],
        "a": defaults["a"],
        "c": defaults["c"],
        "sigma_mosaic_deg": defaults["sigma_mosaic_deg"],
        "gamma_mosaic_deg": defaults["gamma_mosaic_deg"],
        "eta": defaults["eta"],
        "center_x": defaults["center_x"],
        "center_y": defaults["center_y"],
        "bandwidth_percent": _clip_bandwidth_percent(
            defaults.get("bandwidth_percent", bandwidth * 100.0)
        ),
    },
    on_standard_update=schedule_update,
    on_mosaic_update=on_mosaic_slider_change,
)
theta_initial_var = beam_mosaic_parameter_sliders_view_state.theta_initial_var
_attach_live_theta_background_theta_trace(theta_initial_var)
theta_initial_scale = beam_mosaic_parameter_sliders_view_state.theta_initial_scale
cor_angle_var = beam_mosaic_parameter_sliders_view_state.cor_angle_var
cor_angle_scale = beam_mosaic_parameter_sliders_view_state.cor_angle_scale
gamma_var = beam_mosaic_parameter_sliders_view_state.gamma_var
gamma_scale = beam_mosaic_parameter_sliders_view_state.gamma_scale
Gamma_var = beam_mosaic_parameter_sliders_view_state.Gamma_var
Gamma_scale = beam_mosaic_parameter_sliders_view_state.Gamma_scale
chi_var = beam_mosaic_parameter_sliders_view_state.chi_var
chi_scale = beam_mosaic_parameter_sliders_view_state.chi_scale
psi_z_var = beam_mosaic_parameter_sliders_view_state.psi_z_var
psi_z_scale = beam_mosaic_parameter_sliders_view_state.psi_z_scale
zs_var = beam_mosaic_parameter_sliders_view_state.zs_var
zs_scale = beam_mosaic_parameter_sliders_view_state.zs_scale
zb_var = beam_mosaic_parameter_sliders_view_state.zb_var
zb_scale = beam_mosaic_parameter_sliders_view_state.zb_scale
sample_width_var = beam_mosaic_parameter_sliders_view_state.sample_width_var
sample_width_scale = beam_mosaic_parameter_sliders_view_state.sample_width_scale
sample_length_var = beam_mosaic_parameter_sliders_view_state.sample_length_var
sample_length_scale = beam_mosaic_parameter_sliders_view_state.sample_length_scale
sample_depth_var = beam_mosaic_parameter_sliders_view_state.sample_depth_var
sample_depth_scale = beam_mosaic_parameter_sliders_view_state.sample_depth_scale
debye_x_var = beam_mosaic_parameter_sliders_view_state.debye_x_var
debye_x_scale = beam_mosaic_parameter_sliders_view_state.debye_x_scale
debye_y_var = beam_mosaic_parameter_sliders_view_state.debye_y_var
debye_y_scale = beam_mosaic_parameter_sliders_view_state.debye_y_scale
corto_detector_var = beam_mosaic_parameter_sliders_view_state.corto_detector_var
corto_detector_scale = beam_mosaic_parameter_sliders_view_state.corto_detector_scale
a_var = beam_mosaic_parameter_sliders_view_state.a_var
a_scale = beam_mosaic_parameter_sliders_view_state.a_scale
c_var = beam_mosaic_parameter_sliders_view_state.c_var
c_scale = beam_mosaic_parameter_sliders_view_state.c_scale
sigma_mosaic_var = beam_mosaic_parameter_sliders_view_state.sigma_mosaic_var
sigma_mosaic_scale = beam_mosaic_parameter_sliders_view_state.sigma_mosaic_scale
gamma_mosaic_var = beam_mosaic_parameter_sliders_view_state.gamma_mosaic_var
gamma_mosaic_scale = beam_mosaic_parameter_sliders_view_state.gamma_mosaic_scale
eta_var = beam_mosaic_parameter_sliders_view_state.eta_var
eta_scale = beam_mosaic_parameter_sliders_view_state.eta_scale
center_x_var = beam_mosaic_parameter_sliders_view_state.center_x_var
center_x_scale = beam_mosaic_parameter_sliders_view_state.center_x_scale
bandwidth_percent_var = beam_mosaic_parameter_sliders_view_state.bandwidth_percent_var
bandwidth_percent_scale = beam_mosaic_parameter_sliders_view_state.bandwidth_percent_scale
center_y_var = beam_mosaic_parameter_sliders_view_state.center_y_var
center_y_scale = beam_mosaic_parameter_sliders_view_state.center_y_scale
main_display_raster_size_var = tk.DoubleVar(value=float(_default_main_display_raster_size_limit()))
_main_display_raster_size_scale_bounds = SimpleNamespace(
    cget=lambda key: (
        float(MAIN_DISPLAY_RASTER_MIN_SIZE)
        if str(key) == "from"
        else float(MAIN_DISPLAY_RASTER_MAX_SIZE)
    )
)
_normalize_main_display_raster_size_limit()

gui_views.populate_app_shell_quick_controls(
    view_state=app_shell_view_state,
    controls=[
        {
            "key": "theta_initial",
            "label": "theta",
            "variable": theta_initial_var,
            "scale": theta_initial_scale,
            "command": schedule_update,
            "step": 0.01,
        },
        {
            "key": "corto_detector",
            "label": "distance",
            "variable": corto_detector_var,
            "scale": corto_detector_scale,
            "command": schedule_update,
            "step": 0.0001,
        },
        {
            "key": "sampling_count",
            "label": "samples",
            "variable": sample_count_var,
            "scale": sample_count_scale,
            "command": lambda: _apply_random_sample_count(trigger_update=True),
            "step": 1,
        },
        {
            "key": "display_raster_size",
            "label": "display px",
            "variable": main_display_raster_size_var,
            "scale": _main_display_raster_size_scale_bounds,
            "command": _refresh_main_display_raster_projection,
            "step": 1,
        },
        {
            "key": "qr_cylinder_mode",
            "label": "Qr cylinder lines",
            "control_type": "choice",
            "variable": qr_cylinder_display_mode_var,
            "options": QR_CYLINDER_DISPLAY_MODE_OPTIONS,
        },
        {
            "key": "show_geometry_overlays",
            "label": "Show Geometry Overlays",
            "control_type": "check",
            "variable": geometry_overlay_actions_view_state.show_geometry_overlays_var,
            "command": _toggle_geometry_overlay_visibility,
        },
        {
            "key": "toggle_background",
            "label": "Toggle Background",
            "control_type": "button",
            "command": toggle_background,
        },
        {
            "key": "switch_background",
            "label": "Switch Background",
            "control_type": "button",
            "command": switch_background,
        },
        {
            "key": "reset_view",
            "label": "Reset view",
            "control_type": "button",
            "command": _reset_primary_figure_view,
        },
        {
            "key": "clear_integration_region",
            "label": "Clear integration region",
            "control_type": "button",
            "command": _clear_analysis_integration_region,
        },
        {
            "key": "show_beam_center_spot",
            "label": "Beam center spot",
            "control_type": "check",
            "variable": analysis_view_controls_view_state.show_beam_center_spot_var,
            "command": _toggle_beam_center_spot,
        },
        {
            "key": "log_display",
            "label": "Log display",
            "control_type": "check",
            "variable": analysis_view_controls_view_state.log_display_var,
            "command": toggle_log_display,
        },
        {
            "key": "auto_match_scale",
            "label": "Auto-Match Scale (Radial Peak)",
            "control_type": "button",
            "command": _auto_match_scale_factor_to_radial_peak,
        },
        {
            "key": "sf_prune_bias",
            "label": "SF prune bias",
            "variable": sf_prune_bias_var,
            "scale": sf_prune_bias_scale,
            "step": 0.01,
        },
    ],
)
analysis_view_controls_view_state.check_beam_center_spot = (
    app_shell_view_state.quick_control_widgets.get("show_beam_center_spot", {}).get("checkbutton")
)
geometry_overlay_actions_view_state.show_geometry_overlays_checkbutton = (
    app_shell_view_state.quick_control_widgets.get("show_geometry_overlays", {}).get("checkbutton")
)
try:
    fast_viewer_workflow.refresh_status_text()
except Exception:
    pass


def _refresh_refine_section_summaries(*_args) -> None:
    """Refresh short summaries shown in collapsed refinement headers."""

    try:
        geo_summary = (
            f"theta {float(theta_initial_var.get()):.2f} deg | chi {float(chi_var.get()):.3f} deg"
        )
    except Exception:
        geo_summary = ""
    gui_views.set_collapsible_header_summary(geo_frame, geo_summary)

    try:
        detector_summary = (
            f"dist {float(corto_detector_var.get()) * 1e3:.2f} mm | "
            f"gamma {float(gamma_var.get()):.3f} | Gamma {float(Gamma_var.get()):.3f}"
        )
    except Exception:
        detector_summary = ""
    gui_views.set_collapsible_header_summary(detector_frame, detector_summary)

    try:
        lattice_summary = f"a {float(a_var.get()):.3f} A | c {float(c_var.get()):.3f} A"
    except Exception:
        lattice_summary = ""
    gui_views.set_collapsible_header_summary(lattice_frame, lattice_summary)

    try:
        beam_summary = f"row {float(center_x_var.get()):.0f} | col {float(center_y_var.get()):.0f}"
    except Exception:
        beam_summary = ""
    gui_views.set_collapsible_header_summary(center_frame, beam_summary)

    try:
        mosaic_summary = (
            f"sigma {float(sigma_mosaic_var.get()):.2f} deg | "
            f"gamma {float(gamma_mosaic_var.get()):.2f} deg | "
            f"eta {float(eta_var.get()):.3f}"
        )
    except Exception:
        mosaic_summary = ""
    gui_views.set_collapsible_header_summary(mosaic_frame, mosaic_summary)

    try:
        debye_summary = f"Qz {float(debye_x_var.get()):.3f} | Qr {float(debye_y_var.get()):.3f}"
    except Exception:
        debye_summary = ""
    gui_views.set_collapsible_header_summary(debye_frame, debye_summary)


for _summary_var in (
    theta_initial_var,
    chi_var,
    gamma_var,
    Gamma_var,
    corto_detector_var,
    a_var,
    c_var,
    center_x_var,
    center_y_var,
    sigma_mosaic_var,
    gamma_mosaic_var,
    eta_var,
    debye_x_var,
    debye_y_var,
):
    trace_add = getattr(_summary_var, "trace_add", None)
    if callable(trace_add):
        trace_add("write", _refresh_refine_section_summaries)

_refresh_refine_section_summaries()


def _geometry_fit_live_update_manual_peak_cache(
    update_payload: Mapping[str, object] | None,
) -> None:
    """Push one geometry-fit trial's simulated Qr/Qz positions into the live cache."""

    if not isinstance(update_payload, Mapping):
        return
    raw_records = update_payload.get("live_cache_records", ())
    if not isinstance(raw_records, (list, tuple)):
        return

    stored_sim_image = getattr(simulation_runtime_state, "stored_sim_image", None)
    if stored_sim_image is not None:
        native_image_shape = tuple(int(v) for v in np.asarray(stored_sim_image).shape[:2])
    else:
        native_image_shape = (int(image_size), int(image_size))

    def _source_key(record: Mapping[str, object]) -> tuple[int, int] | None:
        try:
            return (
                int(record.get("source_table_index")),
                int(record.get("source_row_index")),
            )
        except Exception:
            return None

    def _dataset_index(record: Mapping[str, object]) -> int:
        try:
            return int(
                record.get(
                    "dataset_index",
                    getattr(background_runtime_state, "current_background_index", 0),
                )
            )
        except Exception:
            return int(getattr(background_runtime_state, "current_background_index", 0))

    updated_any = False
    dataset_entries: dict[int, list[dict[str, object]]] = {}
    dirty_datasets: set[int] = set()

    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            continue
        source_key = _source_key(raw_record)
        if source_key is None:
            continue

        try:
            native_col = float(
                raw_record.get(
                    "simulated_detector_x",
                    raw_record.get("simulated_x", np.nan),
                )
            )
            native_row = float(
                raw_record.get(
                    "simulated_detector_y",
                    raw_record.get("simulated_y", np.nan),
                )
            )
        except Exception:
            native_col = float("nan")
            native_row = float("nan")
        native_point = (
            (float(native_col), float(native_row))
            if np.isfinite(native_col) and np.isfinite(native_row)
            else None
        )

        try:
            refined_two_theta = float(raw_record.get("simulated_two_theta_deg", np.nan))
            refined_phi = float(raw_record.get("simulated_phi_deg", np.nan))
        except Exception:
            refined_two_theta = float("nan")
            refined_phi = float("nan")
        refined_caked = (
            (float(refined_two_theta), float(refined_phi))
            if np.isfinite(refined_two_theta) and np.isfinite(refined_phi)
            else None
        )

        refined_display = None
        if native_point is not None:
            try:
                display_point = _native_sim_to_display_coords(
                    float(native_point[0]),
                    float(native_point[1]),
                    native_image_shape,
                )
            except Exception:
                display_point = None
            if (
                isinstance(display_point, tuple)
                and len(display_point) >= 2
                and np.isfinite(float(display_point[0]))
                and np.isfinite(float(display_point[1]))
            ):
                refined_display = (
                    float(display_point[0]),
                    float(display_point[1]),
                )

        if gui_manual_geometry.update_geometry_manual_peak_record_cache(
            simulation_runtime_state.peak_records,
            source_key=source_key,
            refined_caked=refined_caked,
            refined_native=native_point,
            refined_display=refined_display,
            peak_positions=simulation_runtime_state.peak_positions,
            peak_overlay_cache=simulation_runtime_state.peak_overlay_cache,
        ):
            updated_any = True

        dataset_index = _dataset_index(raw_record)
        entries = dataset_entries.get(dataset_index)
        if entries is None:
            entries = [
                dict(entry)
                for entry in (_geometry_manual_pairs_for_index(dataset_index) or ())
                if isinstance(entry, Mapping)
            ]
            dataset_entries[dataset_index] = entries

        for entry in entries:
            try:
                entry_key = (
                    int(entry.get("source_table_index")),
                    int(entry.get("source_row_index")),
                )
            except Exception:
                entry_key = None
            if entry_key != source_key:
                continue
            if refined_display is not None:
                entry["refined_sim_x"] = float(refined_display[0])
                entry["refined_sim_y"] = float(refined_display[1])
            if native_point is not None:
                entry["refined_sim_native_x"] = float(native_point[0])
                entry["refined_sim_native_y"] = float(native_point[1])
            if refined_caked is not None:
                entry["refined_sim_caked_x"] = float(refined_caked[0])
                entry["refined_sim_caked_y"] = float(refined_caked[1])
            dirty_datasets.add(int(dataset_index))
            updated_any = True
            break

    for dataset_index in sorted(dirty_datasets):
        _set_geometry_manual_pairs_for_index(
            int(dataset_index),
            dataset_entries.get(int(dataset_index), []),
        )

    if updated_any:
        _invalidate_geometry_manual_pick_cache()
        _render_current_geometry_manual_pairs(update_status=False)
        schedule_update()


def _geometry_fit_before_run() -> None:
    """Apply the existing geometry-fit pre-run UI cleanup on the main thread."""

    if bool(getattr(geometry_runtime_state, "manual_pick_armed", False)):
        _set_geometry_manual_pick_mode(False)
    _clear_geometry_preview_artists()
    _clear_geometry_pick_artists()


def _set_geometry_fit_button_running(running: bool) -> None:
    """Disable or enable the geometry-fit button for single-job execution."""

    try:
        fit_button_geometry.config(state=(tk.DISABLED if running else tk.NORMAL))
    except Exception:
        pass


def _ensure_geometry_fit_worker_executor():
    executor = simulation_runtime_state.geometry_fit_executor
    if executor is None:
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ra-sim-geometry-fit",
        )
        simulation_runtime_state.geometry_fit_executor = executor
    return executor


def _clear_geometry_fit_worker_events() -> None:
    event_queue = getattr(simulation_runtime_state, "geometry_fit_event_queue", None)
    if event_queue is None:
        return
    while True:
        try:
            event_queue.get_nowait()
        except Exception:
            break


def _geometry_fit_worker_action_result(
    job: Mapping[str, object],
    *,
    prepare_result: gui_geometry_fit.GeometryFitPreparationResult | None = None,
    execution_result: gui_geometry_fit.GeometryFitRuntimeExecutionResult | None = None,
    error_text: str | None = None,
) -> gui_geometry_fit.GeometryFitRuntimeActionResult:
    """Build one geometry-fit action result from async runtime job metadata."""

    resolved_error_text = (
        str(error_text)
        if error_text
        else (
            str(getattr(execution_result, "error_text", "") or "")
            if execution_result is not None
            else None
        )
    )
    return gui_geometry_fit.GeometryFitRuntimeActionResult(
        params=dict(job.get("params", {}) or {}),
        var_names=[str(name) for name in (job.get("var_names", ()) or ())],
        preserve_live_theta=bool(job.get("preserve_live_theta", False)),
        prepare_result=prepare_result,
        execution_result=execution_result,
        error_text=resolved_error_text or None,
    )


def _persist_geometry_fit_preflight_failure_log(
    *,
    stamp: str,
    log_path: Path | str,
    error_text: str,
    failure_log_sections: Sequence[tuple[str, Sequence[str]]] | None,
) -> Path | None:
    """Persist one geometry-fit preflight failure log when logging is enabled."""

    if gui_geometry_fit.geometry_fit_all_logging_disabled():
        return None
    try:
        return gui_geometry_fit.write_geometry_fit_preflight_failure_log(
            stamp=str(stamp),
            error_text=str(error_text),
            log_path=Path(log_path),
            log_sections=failure_log_sections,
        )
    except Exception:
        return None


def _build_geometry_fit_async_job(
    bindings: gui_geometry_fit.GeometryFitRuntimeActionBindings,
) -> dict[str, object]:
    """Capture the live geometry-fit state needed by the worker thread."""

    params = dict(bindings.value_callbacks.current_params() or {})
    mosaic_params = dict(params.get("mosaic_params", {}) or {})
    var_names = [str(name) for name in (bindings.value_callbacks.current_var_names() or ())]
    preserve_live_theta = "theta_initial" not in var_names and "theta_offset" not in var_names
    prepare_bindings = bindings.prepare_bindings_factory(var_names)
    manual_dataset_bindings = prepare_bindings.manual_dataset_bindings
    execution_bindings = bindings.execution_bindings
    current_background_index = int(manual_dataset_bindings.current_background_index)
    fit_config_snapshot = (
        copy.deepcopy(dict(prepare_bindings.fit_config))
        if isinstance(prepare_bindings.fit_config, Mapping)
        else {}
    )
    theta_initial_value = float(prepare_bindings.theta_initial)
    stamp = str(bindings.stamp_factory())
    log_path = gui_geometry_fit.build_geometry_fit_log_path(
        stamp=stamp,
        log_dir=execution_bindings.log_dir,
        downloads_dir=execution_bindings.downloads_dir,
    )
    if not gui_geometry_fit.geometry_fit_all_logging_disabled():
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.touch(exist_ok=True)
        except Exception:
            pass

    if callable(prepare_bindings.ensure_geometry_fit_caked_view):
        prepare_bindings.ensure_geometry_fit_caked_view()

    selection_applied = bool(
        prepare_bindings.apply_geometry_fit_background_selection(
            trigger_update=False,
            sync_live_theta=not preserve_live_theta,
        )
    )
    selected_background_indices: list[int] = []
    selection_error: str | None = None
    if selection_applied:
        try:
            selected_background_indices = [
                int(idx)
                for idx in prepare_bindings.current_geometry_fit_background_indices(strict=True)
            ]
        except Exception as exc:
            selection_error = str(exc)

    uses_shared_theta = False
    if selection_error is None:
        try:
            uses_shared_theta = bool(
                prepare_bindings.geometry_fit_uses_shared_theta_offset(
                    list(selected_background_indices)
                )
            )
        except Exception:
            uses_shared_theta = False

    theta_metadata_applied = True
    background_theta_values: list[float] = []
    background_theta_error: str | None = None
    theta_offset_value = 0.0
    if uses_shared_theta:
        theta_metadata_applied = bool(
            prepare_bindings.apply_background_theta_metadata(
                trigger_update=False,
                sync_live_theta=not preserve_live_theta,
            )
        )
        if theta_metadata_applied:
            try:
                background_theta_values = [
                    float(value)
                    for value in prepare_bindings.current_background_theta_values(strict_count=True)
                ]
                theta_offset_value = float(
                    prepare_bindings.current_geometry_theta_offset(strict=True)
                )
            except Exception as exc:
                background_theta_error = str(exc)

    fit_params_snapshot = dict(params)
    active_in_selection = current_background_index in set(selected_background_indices)
    primary_index = (
        int(current_background_index)
        if active_in_selection
        else (
            int(selected_background_indices[0])
            if selected_background_indices
            else int(current_background_index)
        )
    )
    joint_background_mode = bool(uses_shared_theta and len(selected_background_indices) > 1)
    build_all_selected_backgrounds = bool(joint_background_mode)
    if uses_shared_theta:
        fit_params_snapshot["theta_offset"] = float(theta_offset_value)
        if 0 <= int(primary_index) < len(background_theta_values):
            fit_params_snapshot["theta_initial"] = float(
                background_theta_values[int(primary_index)]
            )
    else:
        fit_params_snapshot["theta_offset"] = 0.0
        theta_default = float(fit_params_snapshot.get("theta_initial", theta_initial_value))
        fit_params_snapshot["theta_initial"] = float(theta_default)
        background_theta_values = [float(theta_default)]

    if not selection_applied or selection_error is not None or not active_in_selection:
        required_indices: list[int] = []
    elif build_all_selected_backgrounds:
        required_indices = [int(idx) for idx in selected_background_indices]
    else:
        required_indices = [int(primary_index)]

    background_images: dict[int, dict[str, np.ndarray]] = {}
    caked_views_by_background: dict[int, dict[str, np.ndarray]] = {}
    manual_pairs_by_background: dict[int, list[dict[str, object]]] = {}
    source_snapshots: dict[int, dict[str, object]] = {}
    requested_signatures: dict[int, object] = {}
    requested_signature_summaries: dict[int, object] = {}
    background_labels: dict[int, str] = {}
    theta_base_by_background: dict[int, float] = {}
    theta_initial_by_background: dict[int, float] = {}

    def _theta_base_for_index(dataset_index: int) -> float:
        if build_all_selected_backgrounds or joint_background_mode:
            if 0 <= int(dataset_index) < len(background_theta_values):
                return float(background_theta_values[int(dataset_index)])
        return float(fit_params_snapshot.get("theta_initial", theta_initial_value))

    for idx in required_indices:
        native_background, display_background = manual_dataset_bindings.load_background_by_index(
            int(idx)
        )
        background_images[int(idx)] = {
            "native": np.asarray(native_background, dtype=np.float64).copy(),
            "display": np.asarray(display_background, dtype=np.float64).copy(),
        }
        manual_pairs_by_background[int(idx)] = [
            dict(entry)
            for entry in (manual_dataset_bindings.geometry_manual_pairs_for_index(int(idx)) or ())
            if isinstance(entry, Mapping)
        ]
        source_snapshots[int(idx)] = copy.deepcopy(
            simulation_runtime_state.source_row_snapshots.get(int(idx)) or {}
        )
        background_labels[int(idx)] = _geometry_fit_background_label(int(idx))
        theta_base_value = float(_theta_base_for_index(int(idx)))
        theta_initial_value_i = float(
            theta_base_value + float(fit_params_snapshot.get("theta_offset", 0.0))
        )
        theta_base_by_background[int(idx)] = float(theta_base_value)
        theta_initial_by_background[int(idx)] = float(theta_initial_value_i)
        params_i = dict(fit_params_snapshot)
        params_i["theta_initial"] = float(theta_initial_value_i)
        requested_signature = _geometry_source_snapshot_signature_for_background(
            int(idx),
            params_i,
        )
        requested_signatures[int(idx)] = requested_signature
        requested_signature_summaries[int(idx)] = _live_cache_signature_summary(requested_signature)

    geometry_runtime_cfg = copy.deepcopy(
        prepare_bindings.build_runtime_config(dict(fit_params_snapshot))
    )
    solver_runtime_cfg = gui_geometry_fit.apply_manual_point_geometry_fit_runtime_overrides(
        gui_geometry_fit.apply_joint_geometry_fit_runtime_safety_overrides(
            geometry_runtime_cfg,
            joint_background_mode=joint_background_mode,
        ),
        joint_background_mode=joint_background_mode,
    )
    fit_solver_mosaic_params, fit_sample_count = (
        gui_geometry_fit.build_geometry_fit_solver_mosaic_params(
            params=fit_params_snapshot,
            geometry_runtime_cfg=solver_runtime_cfg,
            build_mosaic_params=build_mosaic_params,
        )
    )
    try:
        manual_match_config = (
            copy.deepcopy(manual_dataset_bindings.geometry_manual_match_config() or {})
            if callable(manual_dataset_bindings.geometry_manual_match_config)
            else {}
        )
    except Exception:
        manual_match_config = {}
    try:
        pick_uses_caked_space = (
            bool(manual_dataset_bindings.pick_uses_caked_space())
            if callable(manual_dataset_bindings.pick_uses_caked_space)
            else False
        )
    except Exception:
        pick_uses_caked_space = False

    live_rows_by_background: dict[int, list[dict[str, object]]] = {}
    live_rows_cache_metadata_by_background: dict[int, dict[str, object]] = {}
    if int(current_background_index) in set(required_indices):
        live_rows_by_background[int(current_background_index)] = [
            dict(entry)
            for entry in (_build_live_preview_simulated_peaks_from_cache() or ())
            if isinstance(entry, Mapping)
        ]
        live_rows_cache_metadata_by_background[int(current_background_index)] = (
            dict(_last_live_preview_cache_metadata())
            if callable(_last_live_preview_cache_metadata)
            else {}
        )
        caked_view_payload = _geometry_fit_caked_view_for_index(int(current_background_index))
        if isinstance(caked_view_payload, tuple) and len(caked_view_payload) >= 3:
            try:
                caked_views_by_background[int(current_background_index)] = {
                    "background": np.asarray(
                        caked_view_payload[0],
                        dtype=np.float64,
                    ).copy(),
                    "radial_axis": np.asarray(
                        caked_view_payload[1],
                        dtype=np.float64,
                    ).copy(),
                    "azimuth_axis": np.asarray(
                        caked_view_payload[2],
                        dtype=np.float64,
                    ).copy(),
                }
            except Exception:
                pass

    simulation_runtime_state.geometry_fit_job_counter = (
        int(simulation_runtime_state.geometry_fit_job_counter) + 1
    )
    job_id = int(simulation_runtime_state.geometry_fit_job_counter)
    return {
        "job_id": int(job_id),
        "stamp": stamp,
        "log_path": Path(log_path),
        "params": dict(params),
        "var_names": list(var_names),
        "preserve_live_theta": bool(preserve_live_theta),
        "mosaic_params": dict(mosaic_params),
        "fit_solver_mosaic_params": (
            dict(fit_solver_mosaic_params) if fit_sample_count is not None else None
        ),
        "fit_sample_count": fit_sample_count,
        "fit_config": fit_config_snapshot,
        "geometry_runtime_cfg": geometry_runtime_cfg,
        "theta_initial": float(theta_initial_value),
        "current_background_index": int(current_background_index),
        "selection_applied": bool(selection_applied),
        "selected_background_indices": [int(idx) for idx in selected_background_indices],
        "selection_error": selection_error,
        "uses_shared_theta": bool(uses_shared_theta),
        "theta_metadata_applied": bool(theta_metadata_applied),
        "background_theta_values": [float(value) for value in background_theta_values],
        "background_theta_error": background_theta_error,
        "theta_offset": float(theta_offset_value),
        "joint_background_mode": bool(joint_background_mode),
        "required_indices": [int(idx) for idx in required_indices],
        "primary_index": int(primary_index),
        "osc_files": [str(path) for path in (manual_dataset_bindings.osc_files or ())],
        "image_size": int(manual_dataset_bindings.image_size),
        "display_rotate_k": int(manual_dataset_bindings.display_rotate_k),
        "background_images": background_images,
        "caked_views_by_background": caked_views_by_background,
        "manual_pairs_by_background": manual_pairs_by_background,
        "source_snapshots": source_snapshots,
        "background_cache_by_index": {},
        "requested_signatures": requested_signatures,
        "requested_signature_summaries": requested_signature_summaries,
        "background_labels": background_labels,
        "theta_base_by_background": theta_base_by_background,
        "theta_initial_by_background": theta_initial_by_background,
        "source_snapshot_diagnostics": _geometry_manual_last_source_snapshot_diagnostics(),
        "simulation_diagnostics": _geometry_manual_last_simulation_diagnostics(),
        "live_cache_inventory": _live_cache_inventory_snapshot(),
        "memory_intersection_cache": _copy_intersection_cache_tables(get_last_intersection_cache()),
        "memory_intersection_cache_signature": (simulation_runtime_state.last_simulation_signature),
        "live_rows_by_background": live_rows_by_background,
        "live_rows_cache_metadata_by_background": live_rows_cache_metadata_by_background,
        "live_rows_signature": simulation_runtime_state.last_simulation_signature,
        "manual_match_config": manual_match_config,
        "pick_uses_caked_space": bool(pick_uses_caked_space),
        "geometry_manual_simulated_lookup": (
            manual_dataset_bindings.geometry_manual_simulated_lookup
        ),
        "geometry_manual_entry_display_coords": (
            manual_dataset_bindings.geometry_manual_entry_display_coords
        ),
        "geometry_manual_refresh_pair_entry": (
            manual_dataset_bindings.geometry_manual_refresh_pair_entry
        ),
        "unrotate_display_peaks": manual_dataset_bindings.unrotate_display_peaks,
        "display_to_native_sim_coords": (manual_dataset_bindings.display_to_native_sim_coords),
        "select_fit_orientation": manual_dataset_bindings.select_fit_orientation,
        "apply_orientation_to_entries": (manual_dataset_bindings.apply_orientation_to_entries),
        "orient_image_for_fit": manual_dataset_bindings.orient_image_for_fit,
        "apply_background_backend_orientation": (
            manual_dataset_bindings.apply_background_backend_orientation
        ),
        "project_rows": (
            _project_geometry_manual_peaks_to_current_view
            if callable(_project_geometry_manual_peaks_to_current_view)
            else None
        ),
        "solver_inputs": execution_bindings.solver_inputs,
        "solve_fit": bindings.solve_fit,
        "execution_bindings": execution_bindings,
        "event_queue": simulation_runtime_state.geometry_fit_event_queue,
        "enable_live_update_events": False,
    }


def _handle_geometry_fit_worker_event(kind: str, payload: object) -> None:
    """Apply one geometry-fit worker event on the Tk/main thread."""

    event_kind = str(kind or "")
    payload_mapping = payload if isinstance(payload, Mapping) else {}
    if event_kind == "cmd_line":
        text = str(payload_mapping.get("text", "") or "").strip()
        if text:
            _geometry_fit_cmd_line(text)
        return
    if event_kind == "progress_text":
        text = str(payload_mapping.get("text", "") or "")
        if text:
            progress_label_geometry.config(text=text)
        return
    if event_kind == "live_cache_update":
        _geometry_fit_live_update_manual_peak_cache(payload_mapping)
        return
    if event_kind == "source_snapshot_commit":
        _commit_geometry_manual_source_row_rebuild_result(payload)
        return

    message_text = str(payload_mapping.get("message", "") or "")
    if message_text:
        progress_label_geometry.config(text=message_text)
        if event_kind in {
            "source_cache_build_start",
            "source_cache_build_ready",
            "source_cache_build_failed",
        }:
            _geometry_fit_cmd_line(message_text)
    if event_kind == "preflight_failure":
        error_text = str(
            payload_mapping.get("error_text") or payload_mapping.get("message") or ""
        ).strip()
        if error_text:
            _geometry_fit_cmd_line(f"failed: {error_text}")
    elif event_kind == "solver_finished" and str(payload_mapping.get("status", "")) == "error":
        error_text = str(payload_mapping.get("error_text", "") or "").strip()
        if error_text:
            progress_label_geometry.config(text=error_text)


def _drain_geometry_fit_worker_events(*, job_id: int | None = None) -> None:
    """Drain queued geometry-fit worker events for the active job."""

    event_queue = getattr(simulation_runtime_state, "geometry_fit_event_queue", None)
    if event_queue is None:
        return
    while True:
        try:
            event = event_queue.get_nowait()
        except Exception:
            break
        if not isinstance(event, Mapping):
            continue
        try:
            event_job_id = int(event.get("job_id", -1))
        except Exception:
            event_job_id = -1
        if job_id is not None and int(job_id) != int(event_job_id):
            continue
        _handle_geometry_fit_worker_event(
            str(event.get("kind", "") or ""),
            event.get("payload"),
        )


def _run_async_geometry_fit_worker_job(
    job: Mapping[str, object],
) -> gui_geometry_fit.GeometryFitRuntimeActionResult:
    """Run geometry-fit preflight and solver work off the Tk thread."""

    job_data = dict(job or {})
    job_id = int(job_data.get("job_id", -1))
    event_queue = job_data.get("event_queue")

    def _emit_worker_event(kind: str, payload: object = None) -> None:
        if event_queue is None:
            return
        try:
            event_queue.put(
                {
                    "job_id": int(job_id),
                    "kind": str(kind),
                    "payload": copy.deepcopy(payload),
                }
            )
        except Exception:
            return

    worker_source_row_snapshots = {
        int(idx): copy.deepcopy(snapshot)
        for idx, snapshot in dict(job_data.get("source_snapshots", {}) or {}).items()
    }
    worker_source_snapshot_diagnostics = copy.deepcopy(
        job_data.get("source_snapshot_diagnostics") or {}
    )
    worker_simulation_diagnostics = copy.deepcopy(job_data.get("simulation_diagnostics") or {})
    worker_background_cache_by_index: dict[
        int,
        gui_geometry_fit.GeometryFitBackgroundCacheBundle,
    ] = {}
    worker_geometry_fit_caking_ai: FastAzimuthalIntegrator | None = None
    worker_geometry_fit_caking_sig: (
        tuple[float | None, float | None, float | None, float | None] | None
    ) = None

    def _set_worker_source_snapshot_diagnostics(**kwargs) -> None:
        worker_source_snapshot_diagnostics.clear()
        worker_source_snapshot_diagnostics.update(kwargs)

    def _last_worker_source_snapshot_diagnostics() -> dict[str, object]:
        return dict(worker_source_snapshot_diagnostics)

    def _last_worker_simulation_diagnostics() -> dict[str, object]:
        return dict(worker_simulation_diagnostics)

    def _load_background_by_index_snapshot(index: int) -> tuple[np.ndarray, np.ndarray]:
        background_payload = dict(
            dict(job_data.get("background_images", {}) or {}).get(int(index)) or {}
        )
        return (
            np.asarray(background_payload.get("native"), dtype=np.float64).copy(),
            np.asarray(background_payload.get("display"), dtype=np.float64).copy(),
        )

    def _load_caked_view_by_index_snapshot(
        index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        caked_payload = dict(
            dict(job_data.get("caked_views_by_background", {}) or {}).get(int(index)) or {}
        )
        if not caked_payload:
            return None
        try:
            caked_background = np.asarray(
                caked_payload.get("background"),
                dtype=np.float64,
            ).copy()
            radial_axis = np.asarray(
                caked_payload.get("radial_axis"),
                dtype=np.float64,
            ).copy()
            azimuth_axis = np.asarray(
                caked_payload.get("azimuth_axis"),
                dtype=np.float64,
            ).copy()
        except Exception:
            return None
        if (
            caked_background.ndim != 2
            or caked_background.size <= 0
            or radial_axis.size <= 0
            or azimuth_axis.size <= 0
        ):
            return None
        return caked_background, radial_axis, azimuth_axis

    def _project_source_rows(raw_rows: Sequence[object] | None) -> Sequence[object]:
        project_rows = job_data.get("project_rows")
        if callable(project_rows):
            return project_rows(raw_rows)
        return raw_rows or ()

    def _copy_source_rows(raw_rows: Sequence[object] | None) -> list[dict[str, object]]:
        return [dict(entry) for entry in (raw_rows or ()) if isinstance(entry, Mapping)]

    def _copy_optional_values(
        raw_values: Sequence[object] | None,
    ) -> list[object] | None:
        if raw_values is None:
            return None
        return [copy.deepcopy(entry) for entry in raw_values]

    def _theta_base_for_background_worker(background_index: int) -> float:
        theta_base_mapping = dict(job_data.get("theta_base_by_background", {}) or {})
        if int(background_index) in theta_base_mapping:
            return float(theta_base_mapping[int(background_index)])
        return float(job_data.get("theta_initial", 0.0))

    def _theta_initial_for_background_worker(background_index: int) -> float:
        theta_initial_mapping = dict(job_data.get("theta_initial_by_background", {}) or {})
        if int(background_index) in theta_initial_mapping:
            return float(theta_initial_mapping[int(background_index)])
        return float(job_data.get("theta_initial", 0.0))

    def _bundle_rows(
        bundle: gui_geometry_fit.GeometryFitBackgroundCacheBundle | None,
    ) -> list[dict[str, object]]:
        if not isinstance(bundle, gui_geometry_fit.GeometryFitBackgroundCacheBundle):
            return []
        rows = _copy_source_rows(bundle.projected_rows)
        if rows:
            return rows
        return _copy_source_rows(bundle.stored_rows)

    def _store_worker_background_cache_bundle(
        bundle: gui_geometry_fit.GeometryFitBackgroundCacheBundle,
    ) -> None:
        worker_background_cache_by_index[int(bundle.background_index)] = bundle
        worker_source_row_snapshots[int(bundle.background_index)] = {
            "background_index": int(bundle.background_index),
            "simulation_signature": bundle.requested_signature,
            "rows": _copy_source_rows(bundle.stored_rows),
            "row_count": int(len(bundle.stored_rows or ())),
            "created_from": str(bundle.cache_source or "geometry_fit_background_cache"),
        }

    def _build_geometry_fit_background_cache_bundle(
        *,
        background_index: int,
        background_label: str,
        requested_signature: object,
        requested_signature_summary: object,
        theta_base: float,
        theta_initial: float,
        stored_rows: Sequence[object] | None,
        cache_source: str,
        diagnostics: Mapping[str, object] | None = None,
        peak_table_lattice: Sequence[object] | None = None,
        hit_tables: Sequence[object] | None = None,
        intersection_cache: Sequence[object] | None = None,
        cache_metadata: Mapping[str, object] | None = None,
    ) -> gui_geometry_fit.GeometryFitBackgroundCacheBundle:
        copied_stored_rows = _copy_source_rows(stored_rows)
        copied_projected_rows = _copy_source_rows(_project_source_rows(copied_stored_rows))
        resolved_diagnostics = dict(diagnostics) if isinstance(diagnostics, Mapping) else {}
        resolved_diagnostics.setdefault("source", "geometry_fit_background_cache")
        resolved_diagnostics.setdefault(
            "cache_family",
            "geometry_fit_background_cache",
        )
        resolved_diagnostics.setdefault("action", "prepare")
        resolved_diagnostics.setdefault("status", "background_cache_ready")
        resolved_diagnostics.setdefault("background_index", int(background_index))
        resolved_diagnostics.setdefault("background_label", str(background_label))
        resolved_diagnostics.setdefault("requested_signature", requested_signature)
        resolved_diagnostics.setdefault(
            "requested_signature_summary",
            requested_signature_summary,
        )
        resolved_diagnostics.setdefault("snapshot_signature", requested_signature)
        resolved_diagnostics.setdefault(
            "stored_signature_summary",
            requested_signature_summary,
        )
        resolved_diagnostics.setdefault("theta_base", float(theta_base))
        resolved_diagnostics.setdefault("theta_initial", float(theta_initial))
        resolved_diagnostics.setdefault(
            "raw_peak_count",
            int(len(copied_stored_rows)),
        )
        resolved_diagnostics.setdefault(
            "projected_peak_count",
            int(len(copied_projected_rows)),
        )
        resolved_diagnostics.setdefault("cache_source", str(cache_source))
        resolved_diagnostics.setdefault("signature_match", True)
        resolved_diagnostics.setdefault(
            "live_cache_inventory",
            copy.deepcopy(job_data.get("live_cache_inventory", {})),
        )

        return gui_geometry_fit.GeometryFitBackgroundCacheBundle(
            background_index=int(background_index),
            requested_signature=requested_signature,
            requested_signature_summary=requested_signature_summary,
            background_label=str(background_label),
            theta_base=float(theta_base),
            theta_initial=float(theta_initial),
            projected_rows=copied_projected_rows,
            stored_rows=copied_stored_rows,
            cache_source=str(cache_source),
            diagnostics=resolved_diagnostics,
            peak_table_lattice=_copy_optional_values(peak_table_lattice),
            hit_tables=_copy_optional_values(hit_tables),
            intersection_cache=_copy_optional_values(intersection_cache),
            cache_metadata=(dict(cache_metadata) if isinstance(cache_metadata, Mapping) else None),
        )

    def _worker_geometry_fit_caking_integrator() -> FastAzimuthalIntegrator | None:
        nonlocal worker_geometry_fit_caking_ai, worker_geometry_fit_caking_sig
        params_local = dict(job_data.get("params", {}) or {})
        center_value = np.asarray(
            params_local.get("center", [np.nan, np.nan]),
            dtype=np.float64,
        ).reshape(-1)
        requested_sig = _caked_geometry_cache_signature(
            params_local.get("corto_detector", np.nan),
            center_value[0] if center_value.size > 0 else np.nan,
            center_value[1] if center_value.size > 1 else np.nan,
            params_local.get("lambda", np.nan),
        )
        persistent_cache = getattr(
            simulation_runtime_state,
            "geometry_fit_caking_ai_cache",
            {},
        )
        if (
            worker_geometry_fit_caking_ai is not None
            and worker_geometry_fit_caking_sig == requested_sig
        ):
            return worker_geometry_fit_caking_ai
        if (
            isinstance(persistent_cache, Mapping)
            and persistent_cache.get("ai") is not None
            and persistent_cache.get("sig") == requested_sig
        ):
            worker_geometry_fit_caking_ai = persistent_cache.get("ai")
            worker_geometry_fit_caking_sig = requested_sig
            return worker_geometry_fit_caking_ai
        pixel_size_value = params_local.get(
            "pixel_size_m",
            params_local.get("pixel_size"),
        )
        try:
            worker_geometry_fit_caking_ai = _build_analysis_integrator(
                {
                    "center": center_value,
                    "distance_m": float(params_local.get("corto_detector", np.nan)),
                    "pixel_size_m": float(pixel_size_value),
                    "wavelength_m": float(params_local.get("lambda", np.nan)),
                }
            )
            worker_geometry_fit_caking_sig = requested_sig
            simulation_runtime_state.geometry_fit_caking_ai_cache = {
                "sig": requested_sig,
                "ai": worker_geometry_fit_caking_ai,
            }
        except Exception:
            worker_geometry_fit_caking_ai = None
            worker_geometry_fit_caking_sig = None
        return worker_geometry_fit_caking_ai

    def _store_worker_caked_view_for_background(
        bundle: gui_geometry_fit.GeometryFitBackgroundCacheBundle,
    ) -> dict[str, object]:
        caked_views_by_background = job_data.setdefault("caked_views_by_background", {})
        background_images = dict(job_data.get("background_images", {}) or {})
        background_payload = dict(background_images.get(int(bundle.background_index)) or {})
        native_background = background_payload.get("native")
        if native_background is None:
            return {
                "roi_enabled": False,
                "roi_used_restricted_cake": False,
                "roi_pixel_count": 0,
                "roi_fraction": 0.0,
                "roi_fallback_reason": "missing_native_background",
                "roi_half_width_px": 0.0,
            }

        backend_background = np.asarray(native_background, dtype=np.float64)
        apply_backend_orientation = job_data.get("apply_background_backend_orientation")
        if callable(apply_backend_orientation):
            try:
                oriented_background = apply_backend_orientation(backend_background)
            except Exception:
                oriented_background = backend_background
            if oriented_background is not None:
                backend_background = np.asarray(oriented_background, dtype=np.float64)

        roi_selection = gui_geometry_fit.build_geometry_fit_caked_roi_selection(
            bundle.stored_rows,
            required_pairs=list(
                dict(job_data.get("manual_pairs_by_background", {}) or {}).get(
                    int(bundle.background_index),
                    (),
                )
                or ()
            ),
            image_shape=backend_background.shape[:2],
            fit_config=dict(job_data.get("geometry_runtime_cfg", {}) or {}),
            fit_space_to_detector_point=_geometry_fit_caked_roi_fit_space_to_detector_point(
                job_data.get("params", {})
            ),
        )
        roi_enabled = bool(roi_selection.get("enabled", False))
        roi_pixel_count = int(roi_selection.get("pixel_count", 0) or 0)
        roi_fraction = float(roi_selection.get("fraction", 0.0) or 0.0)
        roi_half_width_px = float(roi_selection.get("half_width_px", 0.0) or 0.0)
        roi_fallback_reason = roi_selection.get("fallback_reason")
        roi_used_restricted_cake = bool(roi_selection.get("valid", False))

        rows = cols = None
        if roi_used_restricted_cake:
            try:
                rows = np.asarray(roi_selection.get("rows"), dtype=np.int32)
                cols = np.asarray(roi_selection.get("cols"), dtype=np.int32)
            except Exception:
                rows = cols = None
                roi_used_restricted_cake = False
                roi_fallback_reason = "invalid_roi_pixels"

        ai = _worker_geometry_fit_caking_integrator()
        if ai is None:
            return {
                "roi_enabled": bool(roi_enabled),
                "roi_used_restricted_cake": False,
                "roi_pixel_count": int(roi_pixel_count),
                "roi_fraction": float(roi_fraction),
                "roi_fallback_reason": "missing_integrator",
                "roi_half_width_px": float(roi_half_width_px),
            }

        try:
            with temporary_numba_thread_limit(default_reserved_cpu_worker_count()):
                res2 = caking(
                    backend_background,
                    ai,
                    rows=rows,
                    cols=cols,
                )
        except Exception as exc:
            if roi_used_restricted_cake:
                roi_used_restricted_cake = False
                roi_fallback_reason = (
                    f"restricted_cake_exception:{type(exc).__name__}"
                )
                with temporary_numba_thread_limit(default_reserved_cpu_worker_count()):
                    res2 = caking(backend_background, ai)
            else:
                return {
                    "roi_enabled": bool(roi_enabled),
                    "roi_used_restricted_cake": False,
                    "roi_pixel_count": int(roi_pixel_count),
                    "roi_fraction": float(roi_fraction),
                    "roi_fallback_reason": f"full_cake_exception:{type(exc).__name__}",
                    "roi_half_width_px": float(roi_half_width_px),
                }

        caked_payload = _prepare_caked_display_payload(res2)
        if not isinstance(caked_payload, Mapping):
            return {
                "roi_enabled": bool(roi_enabled),
                "roi_used_restricted_cake": bool(roi_used_restricted_cake),
                "roi_pixel_count": int(roi_pixel_count),
                "roi_fraction": float(roi_fraction),
                "roi_fallback_reason": "invalid_caked_payload",
                "roi_half_width_px": float(roi_half_width_px),
            }

        try:
            caked_views_by_background[int(bundle.background_index)] = {
                "background": np.asarray(caked_payload.get("image"), dtype=np.float64).copy(),
                "radial_axis": np.asarray(caked_payload.get("radial"), dtype=np.float64).copy(),
                "azimuth_axis": np.asarray(caked_payload.get("azimuth"), dtype=np.float64).copy(),
                "roi_enabled": bool(roi_enabled),
                "roi_used_restricted_cake": bool(roi_used_restricted_cake),
                "roi_pixel_count": int(roi_pixel_count),
                "roi_fraction": float(roi_fraction),
                "roi_fallback_reason": (
                    str(roi_fallback_reason) if roi_fallback_reason is not None else None
                ),
                "roi_half_width_px": float(roi_half_width_px),
            }
        except Exception:
            return {
                "roi_enabled": bool(roi_enabled),
                "roi_used_restricted_cake": bool(roi_used_restricted_cake),
                "roi_pixel_count": int(roi_pixel_count),
                "roi_fraction": float(roi_fraction),
                "roi_fallback_reason": "store_caked_payload_failed",
                "roi_half_width_px": float(roi_half_width_px),
            }

        return {
            "roi_enabled": bool(roi_enabled),
            "roi_used_restricted_cake": bool(roi_used_restricted_cake),
            "roi_pixel_count": int(roi_pixel_count),
            "roi_fraction": float(roi_fraction),
            "roi_fallback_reason": (
                str(roi_fallback_reason) if roi_fallback_reason is not None else None
            ),
            "roi_half_width_px": float(roi_half_width_px),
        }

    def _build_source_rows_for_rebuild(
        source_tables: Sequence[object] | None,
        *,
        params_local: Mapping[str, object],
    ) -> tuple[list[dict[str, object]], list[tuple[float, float, str]], list[object]]:
        table_list = list(source_tables or ())
        if not table_list:
            return [], [], []
        first_table = np.asarray(table_list[0], dtype=np.float64)
        if first_table.ndim == 2 and first_table.shape[1] >= 14:
            hit_tables_local = intersection_cache_to_hit_tables(table_list)
        else:
            hit_tables_local = _copy_hit_tables(table_list)
        return _geometry_manual_build_source_rows_from_hit_tables(
            hit_tables_local,
            image_size_value=int(job_data.get("image_size", image_size)),
            params_local=params_local,
            allow_nominal_hkl_indices=True,
        )

    def _prebuild_background_cache_bundle_worker(
        background_index: int,
        *,
        theta_base: float,
        param_set: Mapping[str, object] | None = None,
        consumer: str = "geometry_fit_preflight_cache",
        prior_diagnostics: Mapping[str, object] | None = None,
        required_pairs: Sequence[Mapping[str, object]] | None = None,
    ) -> gui_geometry_fit.GeometryFitBackgroundCacheBundle | None:
        background_idx = int(background_index)
        params_local = dict(job_data.get("params", {}) or {})
        params_local["theta_initial"] = float(
            _theta_initial_for_background_worker(int(background_idx))
        )
        if isinstance(param_set, Mapping):
            params_local.update(dict(param_set))

        requested_signature = dict(job_data.get("requested_signatures", {}) or {}).get(
            int(background_idx)
        )
        requested_signature_summary = dict(
            job_data.get("requested_signature_summaries", {}) or {}
        ).get(int(background_idx))
        if requested_signature_summary is None:
            requested_signature_summary = _live_cache_signature_summary(requested_signature)
        background_label = dict(job_data.get("background_labels", {}) or {}).get(
            int(background_idx),
            f"background {int(background_idx) + 1}",
        )
        live_rows_signature = job_data.get("live_rows_signature")
        live_rows_cache_metadata = dict(
            copy.deepcopy(
                dict(job_data.get("live_rows_cache_metadata_by_background", {}) or {}).get(
                    int(background_idx),
                    {},
                )
            )
            or {}
        )
        live_rows = _copy_source_rows(
            dict(job_data.get("live_rows_by_background", {}) or {}).get(
                int(background_idx),
                (),
            )
        )

        snapshot = dict(worker_source_row_snapshots.get(int(background_idx)) or {})
        snapshot_signature = snapshot.get("simulation_signature")
        snapshot_rows = _copy_source_rows(snapshot.get("rows"))
        snapshot_validation = (
            gui_geometry_fit.validate_geometry_fit_live_source_rows(
                snapshot_rows,
                required_pairs=required_pairs,
            )
            if required_pairs
            else {}
        )
        if (
            snapshot_signature == requested_signature
            and snapshot_rows
            and (not required_pairs or bool(snapshot_validation.get("valid", False)))
        ):
            bundle = _build_geometry_fit_background_cache_bundle(
                background_index=int(background_idx),
                background_label=str(background_label),
                requested_signature=requested_signature,
                requested_signature_summary=requested_signature_summary,
                theta_base=float(theta_base),
                theta_initial=float(params_local.get("theta_initial", 0.0)),
                stored_rows=snapshot_rows,
                cache_source=str(snapshot.get("created_from") or "source_snapshot"),
                diagnostics={
                    "created_from": snapshot.get("created_from"),
                    "cache_source": str(snapshot.get("created_from") or "source_snapshot"),
                    "live_runtime_cache_validation": snapshot_validation,
                },
            )
            _store_worker_background_cache_bundle(bundle)
            _set_worker_source_snapshot_diagnostics(**dict(bundle.diagnostics))
            worker_simulation_diagnostics.clear()
            worker_simulation_diagnostics.update(dict(bundle.diagnostics))
            return bundle
        if required_pairs and snapshot_signature == requested_signature and snapshot_rows:
            _set_worker_source_snapshot_diagnostics(
                source="geometry_fit_background_cache",
                cache_family="geometry_fit_background_cache",
                action="lookup",
                consumer=str(consumer or "geometry_fit_preflight_cache"),
                status="background_cache_pair_validation_failed",
                background_index=int(background_idx),
                background_label=str(background_label),
                requested_signature=requested_signature,
                requested_signature_summary=requested_signature_summary,
                snapshot_signature=snapshot_signature,
                stored_signature_summary=_live_cache_signature_summary(snapshot_signature),
                raw_peak_count=int(len(snapshot_rows)),
                projected_peak_count=0,
                created_from=snapshot.get("created_from"),
                signature_match=True,
                live_cache_inventory=copy.deepcopy(job_data.get("live_cache_inventory", {})),
                live_runtime_cache_validation=snapshot_validation,
            )

        def _worker_live_rows_payload() -> dict[str, object] | list[dict[str, object]]:
            if requested_signature != live_rows_signature:
                return []
            if live_rows_cache_metadata:
                return {
                    "rows": [
                        dict(entry) for entry in (live_rows or ()) if isinstance(entry, Mapping)
                    ],
                    "cache_metadata": dict(live_rows_cache_metadata),
                }
            return [dict(entry) for entry in (live_rows or ()) if isinstance(entry, Mapping)]

        rebuild_result = gui_geometry_fit.rebuild_geometry_fit_source_rows(
            background_index=int(background_idx),
            background_label=str(background_label),
            params_local=params_local,
            consumer=str(consumer or "geometry_fit_preflight_cache"),
            prior_diagnostics=(
                dict(prior_diagnostics)
                if isinstance(prior_diagnostics, Mapping)
                else _last_worker_source_snapshot_diagnostics()
            ),
            requested_signature=requested_signature,
            requested_signature_summary=requested_signature_summary,
            can_use_live_runtime_cache=(
                int(background_idx) == int(job_data.get("current_background_index", -1))
            ),
            build_live_rows=_worker_live_rows_payload,
            get_memory_intersection_cache=(
                lambda: _copy_intersection_cache_tables(
                    job_data.get("memory_intersection_cache", [])
                )
            ),
            memory_cache_signature=job_data.get("memory_intersection_cache_signature"),
            load_logged_intersection_cache=load_most_recent_logged_intersection_cache,
            logged_cache_matches_params=_geometry_manual_logged_cache_matches_params,
            build_source_rows_from_hit_tables=(
                lambda source_tables: _build_source_rows_for_rebuild(
                    source_tables,
                    params_local=params_local,
                )
            ),
            simulate_hit_tables=(
                lambda normalized_params: _simulate_hit_tables_for_fit(
                    job_data["solver_inputs"].miller,
                    job_data["solver_inputs"].intensities,
                    int(job_data["solver_inputs"].image_size),
                    normalized_params,
                )
            ),
            last_runtime_simulation_diagnostics=_last_worker_simulation_diagnostics,
            project_rows=_project_source_rows,
            required_pairs=required_pairs,
            live_cache_inventory=job_data.get("live_cache_inventory", {}),
        )
        _set_worker_source_snapshot_diagnostics(**dict(rebuild_result.diagnostics or {}))
        worker_simulation_diagnostics.clear()
        worker_simulation_diagnostics.update(dict(rebuild_result.diagnostics or {}))
        if rebuild_result.stored_rows:
            bundle = _build_geometry_fit_background_cache_bundle(
                background_index=int(background_idx),
                background_label=str(background_label),
                requested_signature=rebuild_result.requested_signature,
                requested_signature_summary=(rebuild_result.requested_signature_summary),
                theta_base=float(theta_base),
                theta_initial=float(params_local.get("theta_initial", 0.0)),
                stored_rows=rebuild_result.stored_rows,
                cache_source=str(rebuild_result.rebuild_source or "unknown"),
                diagnostics=dict(rebuild_result.diagnostics or {}),
                peak_table_lattice=rebuild_result.peak_table_lattice,
                hit_tables=rebuild_result.hit_tables,
                intersection_cache=rebuild_result.intersection_cache,
                cache_metadata=rebuild_result.metadata,
            )
            _store_worker_background_cache_bundle(bundle)
            return bundle
        return None

    def _rebuild_source_rows_for_background_worker(
        background_index: int,
        param_set: dict[str, object] | None = None,
        *,
        consumer: str | None = None,
        prior_diagnostics: Mapping[str, object] | None = None,
        required_pairs: Sequence[Mapping[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        background_idx = int(background_index)
        bundle = _prebuild_background_cache_bundle_worker(
            background_idx,
            theta_base=float(_theta_base_for_background_worker(int(background_idx))),
            param_set=param_set,
            consumer=str(consumer or "geometry_fit_dataset"),
            prior_diagnostics=prior_diagnostics,
            required_pairs=required_pairs,
        )
        if not isinstance(bundle, gui_geometry_fit.GeometryFitBackgroundCacheBundle):
            return []
        return _bundle_rows(bundle)

    def _source_rows_for_background_worker(
        background_index: int,
        param_set: dict[str, object] | None = None,
        *,
        consumer: str | None = None,
        required_pairs: Sequence[Mapping[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        background_idx = int(background_index)
        lookup_context = str(consumer or "unspecified")
        requested_signature = dict(job_data.get("requested_signatures", {}) or {}).get(
            int(background_idx)
        )
        requested_signature_summary = dict(
            job_data.get("requested_signature_summaries", {}) or {}
        ).get(int(background_idx))
        if requested_signature_summary is None:
            requested_signature_summary = _live_cache_signature_summary(requested_signature)
        background_label = dict(job_data.get("background_labels", {}) or {}).get(
            int(background_idx),
            f"background {int(background_idx) + 1}",
        )
        live_cache_inventory = copy.deepcopy(job_data.get("live_cache_inventory", {}))
        bundle = worker_background_cache_by_index.get(int(background_idx))
        if not isinstance(bundle, gui_geometry_fit.GeometryFitBackgroundCacheBundle):
            _set_worker_source_snapshot_diagnostics(
                source="geometry_fit_background_cache",
                cache_family="geometry_fit_background_cache",
                action="lookup",
                consumer=lookup_context,
                status="background_cache_missing",
                background_index=int(background_idx),
                background_label=str(background_label),
                requested_signature=requested_signature,
                requested_signature_summary=requested_signature_summary,
                raw_peak_count=0,
                projected_peak_count=0,
                signature_match=False,
                live_cache_inventory=live_cache_inventory,
            )
            return []

        stored_signature_summary = _live_cache_signature_summary(bundle.requested_signature)
        if bundle.requested_signature != requested_signature:
            _set_worker_source_snapshot_diagnostics(
                source="geometry_fit_background_cache",
                cache_family="geometry_fit_background_cache",
                action="lookup",
                consumer=lookup_context,
                status="background_cache_signature_mismatch",
                background_index=int(background_idx),
                background_label=str(background_label),
                requested_signature=requested_signature,
                requested_signature_summary=requested_signature_summary,
                snapshot_signature=bundle.requested_signature,
                stored_signature_summary=stored_signature_summary,
                raw_peak_count=int(len(bundle.stored_rows or ())),
                projected_peak_count=0,
                created_from=bundle.cache_source,
                signature_match=False,
                live_cache_inventory=live_cache_inventory,
            )
            return []

        projected_rows = _bundle_rows(bundle)
        if not projected_rows:
            _set_worker_source_snapshot_diagnostics(
                source="geometry_fit_background_cache",
                cache_family="geometry_fit_background_cache",
                action="lookup",
                consumer=lookup_context,
                status="background_cache_empty",
                background_index=int(background_idx),
                background_label=str(background_label),
                requested_signature=requested_signature,
                requested_signature_summary=requested_signature_summary,
                snapshot_signature=bundle.requested_signature,
                stored_signature_summary=stored_signature_summary,
                raw_peak_count=int(len(bundle.stored_rows or ())),
                projected_peak_count=0,
                created_from=bundle.cache_source,
                signature_match=True,
                live_cache_inventory=live_cache_inventory,
            )
            return []
        bundle_validation = (
            gui_geometry_fit.validate_geometry_fit_live_source_rows(
                bundle.stored_rows,
                required_pairs=required_pairs,
            )
            if required_pairs
            else {}
        )
        if required_pairs and not bool(bundle_validation.get("valid", False)):
            _set_worker_source_snapshot_diagnostics(
                source="geometry_fit_background_cache",
                cache_family="geometry_fit_background_cache",
                action="lookup",
                consumer=lookup_context,
                status="background_cache_pair_validation_failed",
                background_index=int(background_idx),
                background_label=str(background_label),
                requested_signature=requested_signature,
                requested_signature_summary=requested_signature_summary,
                snapshot_signature=bundle.requested_signature,
                stored_signature_summary=stored_signature_summary,
                raw_peak_count=int(len(bundle.stored_rows or ())),
                projected_peak_count=0,
                created_from=bundle.cache_source,
                cache_source=bundle.cache_source,
                signature_match=True,
                theta_base=float(bundle.theta_base),
                theta_initial=float(bundle.theta_initial),
                live_cache_inventory=live_cache_inventory,
                live_runtime_cache_validation=bundle_validation,
            )
            rebuilt_rows = _rebuild_source_rows_for_background_worker(
                background_idx,
                param_set,
                consumer=lookup_context,
                prior_diagnostics=_last_worker_source_snapshot_diagnostics(),
                required_pairs=required_pairs,
            )
            if rebuilt_rows:
                return rebuilt_rows
            return []

        _set_worker_source_snapshot_diagnostics(
            source="geometry_fit_background_cache",
            cache_family="geometry_fit_background_cache",
            action="lookup",
            consumer=lookup_context,
            status="background_cache_hit",
            background_index=int(background_idx),
            background_label=str(background_label),
            requested_signature=requested_signature,
            requested_signature_summary=requested_signature_summary,
            snapshot_signature=bundle.requested_signature,
            stored_signature_summary=stored_signature_summary,
            raw_peak_count=int(len(bundle.stored_rows or ())),
            projected_peak_count=int(len(projected_rows)),
            created_from=bundle.cache_source,
            cache_source=bundle.cache_source,
            signature_match=True,
            theta_base=float(bundle.theta_base),
            theta_initial=float(bundle.theta_initial),
            live_cache_inventory=live_cache_inventory,
            live_runtime_cache_validation=bundle_validation,
        )
        return projected_rows

    def _prebuild_required_background_caches() -> None:
        required_indices = [int(idx) for idx in (job_data.get("required_indices", ()) or ())]
        for background_idx in required_indices:
            background_label = dict(job_data.get("background_labels", {}) or {}).get(
                int(background_idx),
                f"background {int(background_idx) + 1}",
            )
            _emit_worker_event(
                "source_cache_build_start",
                {
                    "background_index": int(background_idx),
                    "background_label": str(background_label),
                    "message": (
                        f"preflight: building source cache for background {int(background_idx) + 1}"
                    ),
                },
            )
            bundle = _prebuild_background_cache_bundle_worker(
                int(background_idx),
                theta_base=float(_theta_base_for_background_worker(int(background_idx))),
                required_pairs=list(
                    dict(job_data.get("manual_pairs_by_background", {}) or {}).get(
                        int(background_idx),
                        (),
                    )
                    or ()
                ),
            )
            if not isinstance(bundle, gui_geometry_fit.GeometryFitBackgroundCacheBundle):
                failure_status = str(
                    _last_worker_source_snapshot_diagnostics().get(
                        "status",
                        "background_cache_build_failed",
                    )
                )
                _emit_worker_event(
                    "source_cache_build_failed",
                    {
                        "background_index": int(background_idx),
                        "background_label": str(background_label),
                        "status": failure_status,
                        "message": (
                            "preflight: source cache failed for "
                            f"background {int(background_idx) + 1} "
                            f"(status={failure_status})"
                        ),
                    },
                )
                continue
            _emit_worker_event(
                "source_cache_build_ready",
                {
                    "background_index": int(background_idx),
                    "background_label": str(background_label),
                    "cache_source": str(bundle.cache_source or "unknown"),
                    "row_count": int(len(bundle.projected_rows or ())),
                    **dict(_store_worker_caked_view_for_background(bundle) or {}),
                    "message": (
                        "preflight: source cache ready for "
                        f"background {int(background_idx) + 1} "
                        f"({int(len(bundle.projected_rows or ()))} rows via "
                        f"{str(bundle.cache_source or 'unknown')})"
                    ),
                },
            )

    worker_manual_dataset_bindings = gui_geometry_fit.GeometryFitRuntimeManualDatasetBindings(
        osc_files=list(job_data.get("osc_files", ()) or ()),
        current_background_index=int(job_data.get("current_background_index", 0)),
        image_size=int(job_data.get("image_size", image_size)),
        display_rotate_k=int(job_data.get("display_rotate_k", DISPLAY_ROTATE_K)),
        geometry_manual_pairs_for_index=(
            lambda idx: [
                dict(entry)
                for entry in (
                    dict(job_data.get("manual_pairs_by_background", {}) or {}).get(
                        int(idx),
                        (),
                    )
                    or ()
                )
                if isinstance(entry, Mapping)
            ]
        ),
        load_background_by_index=_load_background_by_index_snapshot,
        apply_background_backend_orientation=job_data.get("apply_background_backend_orientation"),
        geometry_manual_simulated_peaks_for_params=(lambda *_args, **_kwargs: []),
        geometry_manual_simulated_lookup=job_data.get("geometry_manual_simulated_lookup"),
        geometry_manual_source_rows_for_background=_source_rows_for_background_worker,
        geometry_manual_rebuild_source_rows_for_background=(
            _rebuild_source_rows_for_background_worker
        ),
        geometry_manual_last_source_snapshot_diagnostics=(_last_worker_source_snapshot_diagnostics),
        geometry_manual_last_simulation_diagnostics=_last_worker_simulation_diagnostics,
        geometry_manual_match_config=(
            lambda: copy.deepcopy(job_data.get("manual_match_config", {}))
        ),
        geometry_manual_entry_display_coords=job_data.get("geometry_manual_entry_display_coords"),
        geometry_manual_refresh_pair_entry=job_data.get("geometry_manual_refresh_pair_entry"),
        geometry_manual_caked_view_for_index=_load_caked_view_by_index_snapshot,
        unrotate_display_peaks=job_data.get("unrotate_display_peaks"),
        display_to_native_sim_coords=job_data.get("display_to_native_sim_coords"),
        select_fit_orientation=job_data.get("select_fit_orientation"),
        apply_orientation_to_entries=job_data.get("apply_orientation_to_entries"),
        orient_image_for_fit=job_data.get("orient_image_for_fit"),
        pick_uses_caked_space=(lambda: bool(job_data.get("pick_uses_caked_space", False))),
    )

    def _stage_callback(stage: str, payload: Mapping[str, object]) -> None:
        _emit_worker_event(str(stage), dict(payload or {}))

    try:
        _prebuild_required_background_caches()
        prepare_result = gui_geometry_fit.prepare_geometry_fit_run(
            params=dict(job_data.get("params", {}) or {}),
            var_names=list(job_data.get("var_names", ()) or ()),
            fit_config=dict(job_data.get("fit_config", {}) or {}),
            osc_files=list(job_data.get("osc_files", ()) or ()),
            current_background_index=int(job_data.get("current_background_index", 0)),
            theta_initial=float(job_data.get("theta_initial", 0.0)),
            preserve_live_theta=bool(job_data.get("preserve_live_theta", False)),
            apply_geometry_fit_background_selection=(
                lambda **_kwargs: bool(job_data.get("selection_applied", True))
            ),
            current_geometry_fit_background_indices=(
                lambda **_kwargs: (
                    (_ for _ in ()).throw(RuntimeError(str(job_data.get("selection_error"))))
                    if job_data.get("selection_error")
                    else list(job_data.get("selected_background_indices", ()) or ())
                )
            ),
            geometry_fit_uses_shared_theta_offset=(
                lambda _indices: bool(job_data.get("uses_shared_theta", False))
            ),
            apply_background_theta_metadata=(
                lambda **_kwargs: bool(job_data.get("theta_metadata_applied", True))
            ),
            current_background_theta_values=(
                lambda **_kwargs: (
                    (_ for _ in ()).throw(RuntimeError(str(job_data.get("background_theta_error"))))
                    if job_data.get("background_theta_error")
                    else list(job_data.get("background_theta_values", ()) or ())
                )
            ),
            current_geometry_theta_offset=(
                lambda **_kwargs: float(job_data.get("theta_offset", 0.0))
            ),
            geometry_manual_pairs_for_index=(
                worker_manual_dataset_bindings.geometry_manual_pairs_for_index
            ),
            ensure_geometry_fit_caked_view=lambda: None,
            build_dataset=(
                lambda background_index, *, theta_base, base_fit_params, orientation_cfg, stage_callback=None: (
                    gui_geometry_fit.build_geometry_manual_fit_dataset(
                        background_index,
                        theta_base=theta_base,
                        base_fit_params=base_fit_params,
                        manual_dataset_bindings=worker_manual_dataset_bindings,
                        orientation_cfg=orientation_cfg,
                        stage_callback=stage_callback,
                    )
                )
            ),
            build_runtime_config=(
                lambda _fit_params: copy.deepcopy(
                    dict(job_data.get("geometry_runtime_cfg", {}) or {})
                )
            ),
            stage_callback=_stage_callback,
        )
    except Exception as exc:
        error_text = f"Geometry fit failed: {exc}"
        failure_log_sections = gui_geometry_fit.build_geometry_fit_preflight_log_sections(
            error_text=error_text,
            params=dict(job_data.get("params", {}) or {}),
            var_names=list(job_data.get("var_names", ()) or ()),
            dataset_infos=None,
            current_dataset=None,
            selected_background_indices=list(job_data.get("selected_background_indices", ()) or ()),
            joint_background_mode=bool(job_data.get("joint_background_mode", False)),
            geometry_runtime_cfg=dict(job_data.get("fit_config", {}) or {}),
        )
        logged_path = _persist_geometry_fit_preflight_failure_log(
            stamp=str(job_data.get("stamp", "")),
            log_path=job_data.get("log_path"),
            error_text=error_text,
            failure_log_sections=failure_log_sections,
        )
        prepare_result = gui_geometry_fit.GeometryFitPreparationResult(
            error_text=error_text,
            failure_log_sections=failure_log_sections,
            log_path=logged_path,
        )
        _emit_worker_event(
            "preflight_failure",
            {"error_text": error_text, "message": error_text},
        )
        worker_background_cache_by_index.clear()
        return _geometry_fit_worker_action_result(
            job_data,
            prepare_result=prepare_result,
            error_text=error_text,
        )

    if prepare_result.prepared_run is None:
        should_force_preflight_log = bool(prepare_result.error_text) or bool(
            gui_geometry_fit.geometry_fit_debug_logging_enabled(job_data.get("fit_config"))
        )
        if should_force_preflight_log:
            preflight_error_text = str(
                prepare_result.error_text or "Geometry fit aborted before solver start."
            )
            failure_log_sections = list(
                prepare_result.failure_log_sections or ()
            ) or gui_geometry_fit.build_geometry_fit_preflight_log_sections(
                error_text=preflight_error_text,
                params=dict(job_data.get("params", {}) or {}),
                var_names=list(job_data.get("var_names", ()) or ()),
                dataset_infos=None,
                current_dataset=None,
                selected_background_indices=list(
                    job_data.get("selected_background_indices", ()) or ()
                ),
                joint_background_mode=bool(job_data.get("joint_background_mode", False)),
                geometry_runtime_cfg=dict(job_data.get("fit_config", {}) or {}),
            )
            logged_path = _persist_geometry_fit_preflight_failure_log(
                stamp=str(job_data.get("stamp", "")),
                log_path=job_data.get("log_path"),
                error_text=preflight_error_text,
                failure_log_sections=failure_log_sections,
            )
            prepare_result = gui_geometry_fit.GeometryFitPreparationResult(
                prepared_run=prepare_result.prepared_run,
                error_text=prepare_result.error_text,
                failure_log_sections=[
                    (str(title), [str(line) for line in (lines or ())])
                    for title, lines in failure_log_sections
                ],
                log_path=logged_path,
            )
        worker_background_cache_by_index.clear()
        return _geometry_fit_worker_action_result(
            job_data,
            prepare_result=prepare_result,
            error_text=prepare_result.error_text,
        )

    fit_solver_mosaic_params = job_data.get("fit_solver_mosaic_params")
    if isinstance(fit_solver_mosaic_params, Mapping):
        prepare_result.prepared_run.fit_params["mosaic_params"] = copy.deepcopy(
            dict(fit_solver_mosaic_params)
        )

    execution_result = gui_geometry_fit.execute_runtime_geometry_fit_solver_phase(
        prepared_run=prepare_result.prepared_run,
        var_names=list(job_data.get("var_names", ()) or ()),
        solve_fit=job_data.get("solve_fit"),
        solver_inputs=job_data.get("solver_inputs"),
        stamp=str(job_data.get("stamp", "")),
        log_path=job_data.get("log_path"),
        event_callback=(lambda kind, payload: _emit_worker_event(str(kind), dict(payload or {}))),
        live_update_callback=(
            lambda payload: _emit_worker_event(
                "live_cache_update",
                dict(payload or {}),
            )
        )
        if bool(job_data.get("enable_live_update_events", True))
        else None,
    )
    worker_background_cache_by_index.clear()
    return _geometry_fit_worker_action_result(
        job_data,
        prepare_result=prepare_result,
        execution_result=execution_result,
        error_text=execution_result.error_text,
    )


def _submit_async_geometry_fit_job(job: dict[str, object]) -> None:
    simulation_runtime_state.geometry_fit_active_job = dict(job)
    simulation_runtime_state.geometry_fit_ready_result = None
    simulation_runtime_state.geometry_fit_future = _ensure_geometry_fit_worker_executor().submit(
        _run_async_geometry_fit_worker_job,
        dict(job),
    )
    simulation_runtime_state.geometry_fit_error_text = None
    if simulation_runtime_state.geometry_fit_poll_token is None:
        simulation_runtime_state.geometry_fit_poll_token = root.after(
            SIMULATION_WORKER_POLL_MS,
            _poll_async_geometry_fit_job,
        )


def _finish_geometry_fit_async_job(
    job: Mapping[str, object] | None,
    action_result: gui_geometry_fit.GeometryFitRuntimeActionResult | None,
) -> None:
    simulation_runtime_state.geometry_fit_active_job = None
    simulation_runtime_state.geometry_fit_future = None
    simulation_runtime_state.geometry_fit_poll_token = None
    simulation_runtime_state.geometry_fit_ready_result = None
    simulation_runtime_state.geometry_fit_error_text = (
        str(action_result.error_text)
        if action_result is not None and action_result.error_text
        else None
    )
    _set_geometry_fit_button_running(False)
    if action_result is not None:
        _show_geometry_fit_action_notice(action_result)


def _consume_ready_geometry_fit_result() -> None:
    ready_payload = simulation_runtime_state.geometry_fit_ready_result
    simulation_runtime_state.geometry_fit_ready_result = None
    if not isinstance(ready_payload, Mapping):
        return
    job = dict(ready_payload.get("job", {}) or {})
    if job:
        _drain_geometry_fit_worker_events(job_id=int(job.get("job_id", -1)))
    action_result = ready_payload.get("action_result")
    if not isinstance(action_result, gui_geometry_fit.GeometryFitRuntimeActionResult):
        _finish_geometry_fit_async_job(job, None)
        return

    prepare_result = action_result.prepare_result
    execution_result = action_result.execution_result
    finalized_action_result = action_result
    if (
        isinstance(prepare_result, gui_geometry_fit.GeometryFitPreparationResult)
        and prepare_result.prepared_run is not None
        and isinstance(execution_result, gui_geometry_fit.GeometryFitRuntimeExecutionResult)
        and execution_result.error_text is None
        and execution_result.solver_result is not None
    ):
        execution_setup = gui_geometry_fit.build_runtime_geometry_fit_execution_setup_from_bindings(
            prepared_run=prepare_result.prepared_run,
            mosaic_params=dict(job.get("mosaic_params", {}) or {}),
            stamp=str(job.get("stamp", "")),
            bindings=job.get("execution_bindings"),
        )
        finalized_execution = gui_geometry_fit.finalize_runtime_geometry_fit_execution(
            execution_result=execution_result,
            prepared_run=prepare_result.prepared_run,
            var_names=list(job.get("var_names", ()) or ()),
            preserve_live_theta=bool(job.get("preserve_live_theta", False)),
            setup=execution_setup,
        )
        finalized_action_result = _geometry_fit_worker_action_result(
            job,
            prepare_result=prepare_result,
            execution_result=finalized_execution,
            error_text=finalized_execution.error_text,
        )

    _finish_geometry_fit_async_job(job, finalized_action_result)


def _poll_async_geometry_fit_job() -> None:
    simulation_runtime_state.geometry_fit_poll_token = None
    active_job = simulation_runtime_state.geometry_fit_active_job
    future = simulation_runtime_state.geometry_fit_future
    active_job_id = int(active_job.get("job_id", -1)) if isinstance(active_job, Mapping) else None
    _drain_geometry_fit_worker_events(job_id=active_job_id)

    if future is None or not isinstance(active_job, Mapping):
        return

    if not future.done():
        simulation_runtime_state.geometry_fit_poll_token = root.after(
            SIMULATION_WORKER_POLL_MS,
            _poll_async_geometry_fit_job,
        )
        return

    simulation_runtime_state.geometry_fit_future = None
    simulation_runtime_state.geometry_fit_active_job = None
    _drain_geometry_fit_worker_events(job_id=active_job_id)
    try:
        action_result = future.result()
    except Exception as exc:
        error_text = f"Geometry fit failed: {exc}"
        progress_label_geometry.config(text=error_text)
        _geometry_fit_cmd_line(f"failed: {exc}")
        action_result = _geometry_fit_worker_action_result(
            active_job,
            prepare_result=gui_geometry_fit.GeometryFitPreparationResult(
                error_text=error_text,
                log_path=(
                    Path(active_job.get("log_path"))
                    if active_job.get("log_path") is not None
                    else None
                ),
            ),
            error_text=error_text,
        )
    simulation_runtime_state.geometry_fit_ready_result = {
        "job": dict(active_job),
        "action_result": action_result,
    }
    root.after_idle(_consume_ready_geometry_fit_result)


def _on_fit_geometry_click_async():
    """Run geometry fit through the dedicated background worker lane."""

    if (
        simulation_runtime_state.geometry_fit_active_job is not None
        or simulation_runtime_state.geometry_fit_future is not None
    ):
        message = "Geometry fit already running."
        _geometry_fit_cmd_line("geometry fit already running")
        progress_label_geometry.config(text=message)
        return None

    if not callable(geometry_fit_action_bindings_factory):
        return None

    _geometry_fit_before_run()
    bindings = geometry_fit_action_bindings_factory()
    execution_bindings = bindings.execution_bindings
    _set_geometry_fit_button_running(True)
    execution_bindings.cmd_line("queued: geometry-fit preflight")
    execution_bindings.set_progress_text("preflight: collecting geometry-fit datasets")
    if callable(bindings.flush_ui):
        try:
            bindings.flush_ui()
        except Exception:
            pass

    try:
        job = _build_geometry_fit_async_job(bindings)
    except Exception as exc:
        error_text = f"Geometry fit failed: {exc}"
        _set_geometry_fit_button_running(False)
        execution_bindings.cmd_line(f"failed: {exc}")
        execution_bindings.set_progress_text(error_text)
        current_var_names = list(bindings.value_callbacks.current_var_names() or ())
        action_result = _geometry_fit_worker_action_result(
            {
                "params": dict(bindings.value_callbacks.current_params() or {}),
                "var_names": current_var_names,
                "preserve_live_theta": (
                    "theta_initial" not in current_var_names
                    and "theta_offset" not in current_var_names
                ),
            },
            prepare_result=gui_geometry_fit.GeometryFitPreparationResult(error_text=error_text),
            error_text=error_text,
        )
        _show_geometry_fit_action_notice(action_result)
        return None

    _clear_geometry_fit_worker_events()
    _submit_async_geometry_fit_job(job)
    return None


geometry_fit_runtime_workflow = gui_runtime_geometry_fit.build_runtime_geometry_fit_workflow(
    geometry_fit_module=gui_geometry_fit,
    runtime_fit_analysis_module=gui_runtime_fit_analysis,
    bootstrap_module=gui_bootstrap,
    value_bindings=gui_geometry_fit.GeometryFitRuntimeValueBindings(
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
        sample_width_var=sample_width_var,
        sample_length_var=sample_length_var,
        sample_depth_var=sample_depth_var,
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
        current_background_index=(lambda: background_runtime_state.current_background_index),
        geometry_fit_uses_shared_theta_offset=_geometry_fit_uses_shared_theta_offset,
        current_geometry_theta_offset=_current_geometry_theta_offset,
        background_theta_for_index=_background_theta_for_index,
        build_mosaic_params=build_mosaic_params,
        current_optics_mode_flag=_current_optics_mode_flag,
        lambda_value=lambda_,
        psi=psi,
        n2=lambda: n2,
        pixel_size_value=float(pixel_size_m),
    ),
    manual_dataset_bindings_factory_kwargs={
        "osc_files_factory": (lambda: tuple(background_runtime_state.osc_files)),
        "current_background_index_factory": (
            lambda: int(background_runtime_state.current_background_index)
        ),
        "image_size": image_size,
        "display_rotate_k": DISPLAY_ROTATE_K,
        "geometry_manual_pairs_for_index": _geometry_manual_pairs_for_index,
        "load_background_by_index": _load_background_image_by_index,
        "apply_background_backend_orientation": (_apply_background_backend_orientation),
        "geometry_manual_simulated_peaks_for_params": (_geometry_manual_simulated_peaks_for_params),
        "geometry_manual_simulated_lookup": _geometry_manual_simulated_lookup,
        "geometry_manual_source_rows_for_background": (_geometry_manual_source_rows_for_background),
        "geometry_manual_rebuild_source_rows_for_background": (
            _geometry_manual_rebuild_source_rows_for_background
        ),
        "geometry_manual_last_source_snapshot_diagnostics": (
            _geometry_manual_last_source_snapshot_diagnostics
        ),
        "geometry_manual_last_simulation_diagnostics": (
            _geometry_manual_last_simulation_diagnostics
        ),
        "geometry_manual_match_config": _current_geometry_manual_match_config,
        "geometry_manual_entry_display_coords": (_geometry_manual_entry_display_coords),
        "geometry_manual_refresh_pair_entry": (
            geometry_manual_projection_workflow.refresh_entry_geometry
        ),
        "pick_uses_caked_space": _geometry_manual_pick_uses_caked_space,
        "geometry_manual_caked_view_for_index": (_geometry_fit_caked_view_for_index),
        "unrotate_display_peaks": _unrotate_display_peaks,
        "display_to_native_sim_coords": _display_to_native_sim_coords,
        "select_fit_orientation": _select_fit_orientation,
        "apply_orientation_to_entries": _apply_orientation_to_entries,
        "orient_image_for_fit": _orient_image_for_fit,
    },
    runtime_config_factory_kwargs={
        "base_config": (fit_config.get("geometry", {}) if isinstance(fit_config, dict) else {}),
        "current_constraint_state": _current_geometry_fit_constraint_state,
        "current_parameter_domains": _current_geometry_fit_parameter_domains,
        "current_candidate_param_names": _current_geometry_fit_candidate_param_names,
        "current_caked_roi_enabled": (
            lambda: bool(geometry_fit_caked_roi_enabled_var.get())
        ),
    },
    action_bootstrap_kwargs={
        "value_callbacks_factory": _geometry_fit_runtime_values,
        "fit_config": fit_config,
        "theta_initial_factory": (lambda: theta_initial_var.get()),
        "apply_geometry_fit_background_selection": (_apply_geometry_fit_background_selection),
        "current_geometry_fit_background_indices": (_current_geometry_fit_background_indices),
        "geometry_fit_uses_shared_theta_offset": (_geometry_fit_uses_shared_theta_offset),
        "apply_background_theta_metadata": _apply_background_theta_metadata,
        "current_background_theta_values": _current_background_theta_values,
        "current_geometry_theta_offset": _current_geometry_theta_offset,
        "ensure_geometry_fit_caked_view": _ensure_geometry_fit_caked_view,
        "downloads_dir": get_dir("downloads"),
        "log_dir": get_dir("debug_log_dir"),
        "simulation_runtime_state": simulation_runtime_state,
        "background_runtime_state": background_runtime_state,
        "theta_initial_var": theta_initial_var,
        "geometry_theta_offset_var": geometry_theta_offset_var,
        "current_ui_params": _current_geometry_fit_ui_params,
        "background_theta_for_index": _background_theta_for_index,
        "refresh_status": _refresh_background_status,
        "update_manual_pick_button_label": (_update_geometry_manual_pick_button_label),
        "capture_undo_state": _capture_geometry_fit_undo_state,
        "push_undo_state": _push_geometry_fit_undo_state,
        "replace_dataset_cache": _set_geometry_fit_dataset_cache,
        "request_preview_skip_once": (
            lambda: gui_controllers.request_geometry_preview_skip_once(geometry_preview_state)
        ),
        "schedule_update": schedule_update,
        "draw_overlay_records": (
            lambda records, marker_limit: _draw_geometry_fit_overlay(
                records,
                max_display_markers=marker_limit,
            )
        ),
        "draw_initial_pairs_overlay": (
            lambda pairs, marker_limit: _draw_initial_geometry_pairs_overlay(
                pairs,
                max_display_markers=marker_limit,
            )
        ),
        "set_last_overlay_state": _set_geometry_fit_last_overlay_state,
        "set_progress_text": (lambda text: progress_label_geometry.config(text=text)),
        "cmd_line": _geometry_fit_cmd_line,
        "solver_inputs_factory": (
            lambda: gui_geometry_fit.GeometryFitRuntimeSolverInputs(
                miller=miller,
                intensities=intensities,
                image_size=image_size,
            )
        ),
        "sim_display_rotate_k": SIM_DISPLAY_ROTATE_K,
        "background_display_rotate_k": DISPLAY_ROTATE_K,
        "simulate_and_compare_hkl": simulate_and_compare_hkl,
        "aggregate_match_centers": _aggregate_match_centers,
        "build_overlay_records": build_geometry_fit_overlay_records,
        "compute_frame_diagnostics": _geometry_overlay_frame_diagnostics,
        "solve_fit": fit_geometry_parameters,
        "live_update_callback": _geometry_fit_live_update_manual_peak_cache,
        "stamp_factory": (lambda: datetime.now().strftime("%Y%m%d_%H%M%S")),
        "flush_ui": root.update_idletasks,
        "before_run": (
            lambda: (
                _set_geometry_manual_pick_mode(False)
                if bool(getattr(geometry_runtime_state, "manual_pick_armed", False))
                else None,
                _clear_geometry_preview_artists(),
                _clear_geometry_pick_artists(),
            )
        ),
        "after_run": _show_geometry_fit_action_notice,
    },
)
_geometry_fit_runtime_value_callbacks = geometry_fit_runtime_workflow.value_callbacks
_geometry_fit_var_map = geometry_fit_runtime_workflow.var_map
geometry_fit_manual_dataset_bindings_factory = (
    geometry_fit_runtime_workflow.manual_dataset_bindings_factory
)
geometry_fit_runtime_config_factory = geometry_fit_runtime_workflow.runtime_config_factory
geometry_fit_action_workflow = geometry_fit_runtime_workflow.action_workflow
geometry_fit_action_runtime = geometry_fit_runtime_workflow.action_runtime
geometry_fit_action_bindings_factory = geometry_fit_runtime_workflow.action_bindings_factory
on_fit_geometry_click = _on_fit_geometry_click_async
fit_button_geometry.config(command=on_fit_geometry_click)


def _clamp_psi_z_var(*_):
    try:
        value = float(psi_z_var.get())
        lo = float(psi_z_scale.cget("from"))
        hi = float(psi_z_scale.cget("to"))
    except Exception:
        return
    clipped = gui_controllers.clamp_slider_value_to_bounds(
        value,
        lower_bound=lo,
        upper_bound=hi,
        fallback=defaults.get("psi_z", 0.0),
    )
    if not math.isclose(value, clipped, rel_tol=0.0, abs_tol=1e-12):
        psi_z_var.set(clipped)


psi_z_var.trace_add("write", _clamp_psi_z_var)
_clamp_psi_z_var()
geometry_fit_parameter_specs = {
    "zb": {
        "label": "Beam Offset",
        "value_var": zb_var,
        "value_slider": zb_scale,
        "step": 0.0001,
    },
    "zs": {
        "label": "Sample Offset",
        "value_var": zs_var,
        "value_slider": zs_scale,
        "step": 0.0001,
    },
    "theta_initial": {
        "label": "Theta Sample Tilt",
        "value_var": theta_initial_var,
        "value_slider": theta_initial_scale,
        "step": 0.01,
    },
    "psi_z": {
        "label": "Goniometer Axis Yaw (about z)",
        "value_var": psi_z_var,
        "value_slider": psi_z_scale,
        "step": 0.01,
    },
    "chi": {
        "label": "Sample Pitch",
        "value_var": chi_var,
        "value_slider": chi_scale,
        "step": 0.001,
    },
    "cor_angle": {
        "label": "Goniometer Axis Pitch (about y)",
        "value_var": cor_angle_var,
        "value_slider": cor_angle_scale,
        "step": 0.01,
    },
    "gamma": {
        "label": "Detector Pitch",
        "value_var": gamma_var,
        "value_slider": gamma_scale,
        "step": 0.001,
    },
    "Gamma": {
        "label": "Detector Yaw",
        "value_var": Gamma_var,
        "value_slider": Gamma_scale,
        "step": 0.001,
    },
    "corto_detector": {
        "label": "Detector Distance",
        "value_var": corto_detector_var,
        "value_slider": corto_detector_scale,
        "step": 0.0001,
    },
    "a": {
        "label": "a Lattice Parameter",
        "value_var": a_var,
        "value_slider": a_scale,
        "step": 0.01,
    },
    "c": {
        "label": "c Lattice Parameter",
        "value_var": c_var,
        "value_slider": c_scale,
        "step": 0.01,
    },
    "center_x": {
        "label": "Beam Center Row",
        "value_var": center_x_var,
        "value_slider": center_x_scale,
        "step": 1.0,
    },
    "center_y": {
        "label": "Beam Center Col",
        "value_var": center_y_var,
        "value_slider": center_y_scale,
        "step": 1.0,
    },
}

# Slider controlling contribution of the first CIF file, only if a second CIF
# was provided.
gui_views.create_cif_weight_controls(
    parent=app_shell_view_state.right_col,
    view_state=cif_weight_controls_view_state,
    has_second_cif=bool(has_second_cif),
    weight1=float(weight1),
    weight2=float(weight2),
)
weight1_var = cif_weight_controls_view_state.weight1_var
weight2_var = cif_weight_controls_view_state.weight2_var
if weight1_var is None or weight2_var is None:
    raise RuntimeError("CIF weight controls did not create slider variables.")


def update_weights(*args):
    """Recompute intensities using the current CIF weights."""
    gui_structure_model.update_weighted_intensities(
        structure_model_state,
        weight1=weight1_var.get(),
        weight2=weight2_var.get(),
        combine_weighted_intensities=gui_controllers.combine_cif_weighted_intensities,
        schedule_update=_invalidate_and_schedule_update,
    )
    _sync_structure_model_aliases()


if has_second_cif:
    weight1_var.trace_add("write", update_weights)
    weight2_var.trace_add("write", update_weights)
# ---------------------------------------------------------------------------
#  OCCUPANCY CONTROLS: one control per structure site in the loaded CIF.
# ---------------------------------------------------------------------------
structure_model_state.occ_vars = [tk.DoubleVar(value=float(val)) for val in occ]
structure_model_state.atom_site_fract_vars = [
    {
        "x": tk.DoubleVar(value=float(row["x"])),
        "y": tk.DoubleVar(value=float(row["y"])),
        "z": tk.DoubleVar(value=float(row["z"])),
    }
    for row in _atom_site_fractional_rows()
]


def _occupancy_label_text(site_idx: int, *, input_label: bool = False) -> str:
    """Build a user-facing occupancy label for a unique atom site."""

    idx = int(site_idx)
    atom_name = occupancy_site_labels[idx] if idx < len(occupancy_site_labels) else "not in CIF"
    prefix = "Input Occupancy" if input_label else "Occupancy"
    return f"{prefix} Site {idx + 1} ({atom_name})"


def _rebuild_diffraction_inputs(
    new_occ,
    p_vals,
    weights,
    a_axis,
    c_axis,
    *,
    force=False,
    trigger_update=True,
):
    """Refresh cached HT curves and peak lists for the current settings."""
    structure_model_state.two_theta_range = two_theta_range
    gui_structure_model.rebuild_diffraction_inputs(
        structure_model_state,
        new_occ=new_occ,
        p_vals=p_vals,
        weights=weights,
        a_axis=a_axis,
        c_axis=c_axis,
        finite_stack_flag=bool(finite_stack_var.get()),
        layers=int(max(1, stack_layers_var.get())),
        phase_delta_expression_current=_current_phase_delta_expression(),
        phi_l_divisor_current=_current_phi_l_divisor(),
        atom_site_values=_current_atom_site_fractional_values(),
        iodine_z_current=_current_iodine_z(),
        rod_points_per_gz=_current_rod_points_per_gz(),
        atom_site_override_state=atom_site_override_state,
        simulation_runtime_state=simulation_runtime_state,
        combine_weighted_intensities=gui_controllers.combine_cif_weighted_intensities,
        build_intensity_dataframes=build_intensity_dataframes,
        apply_bragg_qr_filters=apply_bragg_qr_filters,
        schedule_update=_invalidate_and_schedule_update,
        weight1=weight1_var.get(),
        weight2=weight2_var.get(),
        tcl_error_types=(tk.TclError,),
        force=force,
        trigger_update=trigger_update,
    )
    _sync_structure_model_aliases()


def update_occupancies(*args):
    """Recompute Hendricks–Teller curves when occupancies or p-values change."""

    try:
        new_occ = [float(var.get()) for var in _occupancy_control_vars()]
    except (tk.TclError, ValueError):
        return

    if not all(np.isfinite(v) for v in new_occ):
        return

    # Keep occupancies physically meaningful and reflect clamped values in the UI.
    clamped_occ = gui_controllers.clamp_site_occupancy_values(
        new_occ,
        fallback_values=occ,
    )
    for var, val in zip(_occupancy_control_vars(), clamped_occ):
        try:
            current = float(var.get())
        except (tk.TclError, ValueError):
            current = val
        if not math.isclose(current, val, rel_tol=1e-12, abs_tol=1e-12):
            var.set(val)
    new_occ = clamped_occ

    try:
        p_vals = [float(p0_var.get()), float(p1_var.get()), float(p2_var.get())]
        w_raw = [float(w0_var.get()), float(w1_var.get()), float(w2_var.get())]
    except (tk.TclError, ValueError):
        return
    weights = gui_controllers.normalize_stacking_weight_values(w_raw)

    _rebuild_diffraction_inputs(new_occ, p_vals, weights, a_var.get(), c_var.get())


def _sync_finite_controls():
    enabled = bool(
        finite_stack_controls_view_state.finite_stack_var is not None
        and finite_stack_controls_view_state.finite_stack_var.get()
    )
    gui_views.set_finite_stack_layer_controls_enabled(
        finite_stack_controls_view_state,
        enabled=enabled,
    )


def _normalize_layer_value(raw_value):
    fallback = (
        stack_layers_var.get()
        if finite_stack_controls_view_state.stack_layers_var is not None
        else 1
    )
    return gui_controllers.normalize_finite_stack_layer_count(raw_value, fallback)


def _normalize_phase_delta_expression_value(raw_value):
    fallback = _current_phase_delta_expression()
    return gui_controllers.normalize_finite_stack_phase_delta_expression(
        raw_value,
        fallback=fallback,
    )


def _normalize_phi_l_divisor_value(raw_value):
    return gui_controllers.normalize_finite_stack_phi_l_divisor(
        raw_value,
        fallback=_current_phi_l_divisor(),
    )


def _sync_layer_entry_from_var(*_):
    if finite_stack_controls_view_state.layers_entry_var is None:
        return
    normalized = gui_controllers.format_finite_stack_layer_count(stack_layers_var.get())
    if finite_stack_controls_view_state.layers_entry_var.get().strip() != normalized:
        gui_views.set_finite_stack_layer_entry_text(
            finite_stack_controls_view_state,
            normalized,
        )


def _commit_layer_entry(_event=None):
    if finite_stack_controls_view_state.layers_entry_var is None:
        return
    value = _normalize_layer_value(finite_stack_controls_view_state.layers_entry_var.get())
    gui_views.ensure_finite_stack_layer_scale_max(
        finite_stack_controls_view_state,
        value,
    )
    changed = stack_layers_var.get() != value
    if changed:
        stack_layers_var.set(value)
    _sync_layer_entry_from_var()
    if changed and finite_stack_var.get():
        update_occupancies()


def _commit_phase_delta_expression_entry(_event=None):
    if finite_stack_controls_view_state.phase_delta_entry_var is None:
        return

    try:
        value = _normalize_phase_delta_expression_value(
            finite_stack_controls_view_state.phase_delta_entry_var.get()
        )
    except ValueError as exc:
        gui_views.set_finite_stack_phase_delta_entry_text(
            finite_stack_controls_view_state,
            _current_phase_delta_expression(),
        )
        progress_label.config(text=f"Invalid phase delta equation: {exc}")
        return

    current = _current_phase_delta_expression()
    if current != value:
        phase_delta_expr_var.set(value)
        gui_views.set_finite_stack_phase_delta_entry_text(
            finite_stack_controls_view_state,
            value,
        )
        update_occupancies()
        progress_label.config(text="Updated phase delta equation.")
        return

    gui_views.set_finite_stack_phase_delta_entry_text(
        finite_stack_controls_view_state,
        value,
    )


def _sync_phi_l_divisor_entry_from_var(*_):
    if finite_stack_controls_view_state.phi_l_divisor_entry_var is None:
        return
    normalized = gui_controllers.format_finite_stack_phi_l_divisor(_current_phi_l_divisor())
    if finite_stack_controls_view_state.phi_l_divisor_entry_var.get().strip() != normalized:
        gui_views.set_finite_stack_phi_l_divisor_entry_text(
            finite_stack_controls_view_state,
            normalized,
        )


def _commit_phi_l_divisor_entry(_event=None):
    if finite_stack_controls_view_state.phi_l_divisor_entry_var is None:
        return

    value = _normalize_phi_l_divisor_value(
        finite_stack_controls_view_state.phi_l_divisor_entry_var.get()
    )
    current = _current_phi_l_divisor()
    if not math.isclose(current, value, rel_tol=1e-12, abs_tol=1e-12):
        phi_l_divisor_var.set(float(value))
        _sync_phi_l_divisor_entry_from_var()
        update_occupancies()
        progress_label.config(text="Updated HT phi L divisor.")
        return

    _sync_phi_l_divisor_entry_from_var()


def _on_finite_toggle():
    _sync_finite_controls()
    update_occupancies()


def _on_layer_slider(val):
    value = _normalize_layer_value(val)
    if stack_layers_var.get() != value:
        stack_layers_var.set(value)
    _sync_layer_entry_from_var()
    if finite_stack_var.get():
        update_occupancies()


ordered_structure_fit_cfg = (
    fit_config.get("ordered_structure", {}) if isinstance(fit_config, dict) else {}
)
ordered_structure_fit_solver_cfg = (
    ordered_structure_fit_cfg.get("solver", {})
    if isinstance(ordered_structure_fit_cfg, dict)
    else {}
)
ordered_structure_fit_mask_cfg = (
    ordered_structure_fit_cfg.get("mask", {}) if isinstance(ordered_structure_fit_cfg, dict) else {}
)
ordered_structure_fit_defaults_cfg = (
    ordered_structure_fit_cfg.get("defaults", {})
    if isinstance(ordered_structure_fit_cfg, dict)
    else {}
)
ordered_structure_coord_window_default = gui_ordered_structure_fit.normalize_coordinate_window(
    ordered_structure_fit_defaults_cfg.get("coord_window", 0.02),
    fallback=0.02,
)
ordered_structure_scale_var = tk.DoubleVar(
    value=float(defaults.get("ordered_structure_scale", 1.0))
)
ordered_structure_coord_window_var = tk.DoubleVar(
    value=float(ordered_structure_coord_window_default)
)
ordered_structure_fit_debye_x_var = tk.BooleanVar(value=True)
ordered_structure_fit_debye_y_var = tk.BooleanVar(value=True)
ordered_structure_fit_result_var = tk.StringVar(value="No ordered-structure fit run yet.")
ordered_structure_fit_occ_toggle_vars: list[tk.BooleanVar] = []
ordered_structure_fit_atom_toggle_vars: list[dict[str, tk.BooleanVar]] = []


def _set_ordered_structure_result_text(text: object) -> None:
    ordered_structure_fit_result_var.set(str(text))


def _set_ordered_structure_revert_enabled(enabled: bool) -> None:
    button = ordered_structure_fit_view_state.revert_button
    if button is not None:
        button.configure(state=(tk.NORMAL if enabled else tk.DISABLED))


def _commit_ordered_structure_scale_entry(_event=None) -> None:
    try:
        raw_value = ordered_structure_scale_var.get()
    except Exception:
        raw_value = _current_ordered_structure_scale()
    normalized = gui_ordered_structure_fit.normalize_ordered_structure_scale(
        raw_value,
        fallback=_current_ordered_structure_scale(),
    )
    if not math.isclose(
        normalized,
        _current_ordered_structure_scale(),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        ordered_structure_scale_var.set(float(normalized))
        schedule_update()
    else:
        ordered_structure_scale_var.set(float(normalized))


def _commit_ordered_structure_coord_window_entry(_event=None) -> None:
    try:
        raw_value = ordered_structure_coord_window_var.get()
    except Exception:
        raw_value = ordered_structure_coord_window_default
    ordered_structure_coord_window_var.set(
        float(
            gui_ordered_structure_fit.normalize_coordinate_window(
                raw_value,
                fallback=ordered_structure_coord_window_default,
            )
        )
    )


def _rebuild_ordered_structure_fit_selection_controls() -> None:
    """Rebuild the ordered-structure fit parameter-selection checkboxes."""

    global ordered_structure_fit_occ_toggle_vars, ordered_structure_fit_atom_toggle_vars

    previous_occ = {
        _occupancy_label_text(idx): bool(var.get())
        for idx, var in enumerate(ordered_structure_fit_occ_toggle_vars)
    }
    ordered_structure_fit_occ_toggle_vars = [
        tk.BooleanVar(value=previous_occ.get(_occupancy_label_text(idx), True))
        for idx in range(len(_occupancy_control_vars()))
    ]
    ordered_structure_fit_view_state.occupancy_toggle_vars = list(
        ordered_structure_fit_occ_toggle_vars
    )
    gui_views.rebuild_ordered_structure_fit_occupancy_controls(
        view_state=ordered_structure_fit_view_state,
        occupancy_vars=ordered_structure_fit_occ_toggle_vars,
        occupancy_label_text=lambda idx: _occupancy_label_text(idx),
    )

    previous_atom = {}
    for idx, axis_vars in enumerate(ordered_structure_fit_atom_toggle_vars):
        label = _atom_site_fractional_label_text(idx)
        for axis in ("x", "y", "z"):
            previous_atom[(label, axis)] = bool(axis_vars.get(axis).get())

    ordered_structure_fit_atom_toggle_vars = []
    for idx, _axis_vars in enumerate(_atom_site_fractional_control_vars()):
        label = _atom_site_fractional_label_text(idx)
        ordered_structure_fit_atom_toggle_vars.append(
            {
                "x": tk.BooleanVar(value=previous_atom.get((label, "x"), False)),
                "y": tk.BooleanVar(value=previous_atom.get((label, "y"), False)),
                "z": tk.BooleanVar(value=previous_atom.get((label, "z"), True)),
            }
        )
    ordered_structure_fit_view_state.atom_toggle_vars = list(ordered_structure_fit_atom_toggle_vars)
    gui_views.rebuild_ordered_structure_fit_atom_coordinate_controls(
        view_state=ordered_structure_fit_view_state,
        atom_toggle_vars=ordered_structure_fit_atom_toggle_vars,
        atom_site_label_text=_atom_site_fractional_label_text,
    )


gui_views.create_ordered_structure_fit_panel(
    parent=app_shell_view_state.right_col,
    view_state=ordered_structure_fit_view_state,
    ordered_scale_var=ordered_structure_scale_var,
    coord_window_var=ordered_structure_coord_window_var,
    fit_debye_x_var=ordered_structure_fit_debye_x_var,
    fit_debye_y_var=ordered_structure_fit_debye_y_var,
    result_var=ordered_structure_fit_result_var,
    on_fit=lambda: on_fit_ordered_structure_click(),
    on_revert=lambda: on_revert_last_ordered_fit(),
    on_commit_ordered_scale=_commit_ordered_structure_scale_entry,
    on_commit_coord_window=_commit_ordered_structure_coord_window_entry,
)
_set_ordered_structure_revert_enabled(False)


# Sliders for three disorder probabilities and weights inside a collapsible frame
gui_views.create_stacking_parameter_panels(
    parent=app_shell_view_state.right_col,
    view_state=stacking_parameter_controls_view_state,
)
gui_views.create_finite_stack_controls(
    parent=stacking_parameter_controls_view_state.stack_frame.frame,
    view_state=finite_stack_controls_view_state,
    finite_stack=bool(defaults["finite_stack"]),
    stack_layers=int(defaults["stack_layers"]),
    phi_l_divisor=float(defaults["phi_l_divisor"]),
    phase_delta_expression=str(defaults["phase_delta_expression"]),
    on_toggle_finite_stack=_on_finite_toggle,
    on_layer_slider=_on_layer_slider,
    on_commit_layer_entry=_commit_layer_entry,
    on_commit_phi_l_divisor_entry=_commit_phi_l_divisor_entry,
    on_commit_phase_delta_expression_entry=_commit_phase_delta_expression_entry,
)
finite_stack_var = finite_stack_controls_view_state.finite_stack_var
stack_layers_var = finite_stack_controls_view_state.stack_layers_var
phase_delta_expr_var = finite_stack_controls_view_state.phase_delta_expr_var
phi_l_divisor_var = finite_stack_controls_view_state.phi_l_divisor_var
stack_layers_var.trace_add("write", _sync_layer_entry_from_var)
_sync_layer_entry_from_var()
_sync_finite_controls()
phi_l_divisor_var.trace_add("write", _sync_phi_l_divisor_entry_from_var)
_sync_phi_l_divisor_entry_from_var()
gui_views.create_stacking_probability_sliders(
    parent=stacking_parameter_controls_view_state.stack_frame.frame,
    view_state=stacking_parameter_controls_view_state,
    values={
        "p0": float(defaults["p0"]),
        "w0": float(defaults["w0"]),
        "p1": float(defaults["p1"]),
        "w1": float(defaults["w1"]),
        "p2": float(defaults["p2"]),
        "w2": float(defaults["w2"]),
    },
    on_update=update_occupancies,
)
p0_var = stacking_parameter_controls_view_state.p0_var
w0_var = stacking_parameter_controls_view_state.w0_var
p1_var = stacking_parameter_controls_view_state.p1_var
w1_var = stacking_parameter_controls_view_state.w1_var
p2_var = stacking_parameter_controls_view_state.p2_var
w2_var = stacking_parameter_controls_view_state.w2_var


def _rebuild_occupancy_controls():
    """Recreate occupancy sliders/entries for the current structure-model vars."""

    gui_views.rebuild_occupancy_controls(
        view_state=stacking_parameter_controls_view_state,
        occ_vars=_occupancy_control_vars(),
        occupancy_label_text=lambda idx: _occupancy_label_text(idx),
        occupancy_input_label_text=lambda idx: _occupancy_label_text(
            idx,
            input_label=True,
        ),
        on_update=update_occupancies,
    )


def _current_occupancy_values():
    """Return the occupancy values currently shown in the controls."""

    return gui_structure_model.current_occupancy_values(
        structure_model_state,
        tcl_error_types=(tk.TclError,),
    )


def _atom_site_fractional_label_text(site_idx: int) -> str:
    idx = int(site_idx)
    atom_site_rows = _atom_site_fractional_rows()
    if idx < len(atom_site_rows):
        return str(atom_site_rows[idx].get("label", f"site_{idx + 1}"))
    return f"site_{idx + 1}"


def _rebuild_atom_site_fractional_controls():
    """Recreate x/y/z entry controls for atom-site fractional coordinates."""

    gui_views.rebuild_atom_site_fractional_controls(
        view_state=stacking_parameter_controls_view_state,
        atom_site_fract_vars=_atom_site_fractional_control_vars(),
        atom_site_label_text=_atom_site_fractional_label_text,
        on_update=update_occupancies,
        empty_text="No _atom_site_fract_x/_y/_z loop found in the active CIF.",
    )


structure_model_controls_built = False


def _rebuild_structure_model_controls() -> None:
    """Populate the deferred occupancy/atom-site structure controls."""

    global structure_model_controls_built

    _rebuild_ordered_structure_fit_selection_controls()
    _rebuild_occupancy_controls()
    _rebuild_atom_site_fractional_controls()
    structure_model_controls_built = True


def _set_widget_tree_enabled(widget, *, enabled: bool) -> None:
    if widget is None:
        return

    state_flags = ["!disabled"] if enabled else ["disabled"]
    normal_state = tk.NORMAL if enabled else tk.DISABLED

    try:
        widget.state(state_flags)
    except Exception:
        try:
            widget.configure(state=normal_state)
        except Exception:
            pass

    try:
        children = list(widget.winfo_children())
    except Exception:
        children = []
    for child in children:
        _set_widget_tree_enabled(child, enabled=enabled)


def _set_structure_bootstrap_controls_enabled(enabled: bool) -> None:
    control_roots = (
        workspace_panels_view_state.workspace_inputs_frame,
        app_shell_view_state.right_col,
        app_shell_view_state.fit_body,
    )
    for widget in control_roots:
        _set_widget_tree_enabled(widget, enabled=enabled)


def _reset_structure_model_control_vars(
    occupancy_values,
    atom_site_values,
):
    """Replace occupancy and atom-site Tk variables for the active structure model."""

    if len(structure_model_state.occ_vars) != len(occupancy_values):
        structure_model_state.occ_vars = [
            tk.DoubleVar(value=float(value)) for value in occupancy_values
        ]
    else:
        for occ_var, value in zip(structure_model_state.occ_vars, occupancy_values):
            occ_var.set(float(value))

    structure_model_state.atom_site_fract_vars = [
        {
            "x": tk.DoubleVar(value=float(x_val)),
            "y": tk.DoubleVar(value=float(y_val)),
            "z": tk.DoubleVar(value=float(z_val)),
        }
        for (x_val, y_val, z_val) in atom_site_values
    ]
    _rebuild_structure_model_controls()


def _apply_primary_cif_path(raw_path):
    """Load a new primary CIF file and rebuild diffraction inputs."""

    global n2

    text_path = str(raw_path).strip().strip('"').strip("'")
    if not text_path:
        progress_label.config(text="No CIF file path provided.")
        return

    candidate = Path(text_path).expanduser()
    if not candidate.is_file():
        progress_label.config(text=f"CIF file not found: {candidate}")
        return

    snapshot = gui_structure_model.capture_primary_cif_reload_snapshot(
        structure_model_state,
        current_occ_values=_current_occupancy_values(),
        current_atom_site_values=_current_atom_site_fractional_values(),
    )
    try:
        old_slider_a = float(a_var.get())
    except Exception:
        old_slider_a = snapshot.default_a
    try:
        old_slider_c = float(c_var.get())
    except Exception:
        old_slider_c = snapshot.default_c
    try:
        reload_plan = gui_structure_model.prepare_primary_cif_reload_plan(
            structure_model_state,
            str(candidate),
            current_occ_values=snapshot.current_occ_values,
            clamp_site_occupancy_values=gui_controllers.clamp_site_occupancy_values,
        )

        try:
            p_vals = [float(p0_var.get()), float(p1_var.get()), float(p2_var.get())]
        except (tk.TclError, ValueError):
            p_vals = [float(defaults["p0"]), float(defaults["p1"]), float(defaults["p2"])]
        try:
            w_raw = [float(w0_var.get()), float(w1_var.get()), float(w2_var.get())]
        except (tk.TclError, ValueError):
            w_raw = [float(defaults["w0"]), float(defaults["w1"]), float(defaults["w2"])]
        weights = gui_controllers.normalize_stacking_weight_values(w_raw)

        _reset_structure_model_control_vars(
            reload_plan.occ,
            reload_plan.atom_site_values,
        )
        gui_structure_model.apply_primary_cif_reload_plan(
            structure_model_state,
            reload_plan,
            occ_vars=_occupancy_control_vars(),
            atom_site_fract_vars=_atom_site_fractional_control_vars(),
            has_second_cif=has_second_cif,
        )
        _sync_structure_model_aliases()

        cif_file_var.set(_current_primary_cif_path())
        _reset_atom_site_override_cache()
        _rebuild_diffraction_inputs(
            reload_plan.occ,
            p_vals,
            weights,
            reload_plan.av,
            reload_plan.cv,
            force=True,
            trigger_update=True,
        )
        a_var.set(av)
        c_var.set(cv)
        n2 = _current_nominal_n2(_current_primary_cif_path())
        simulation_runtime_state.profile_cache.pop("_optics_signature", None)
        _invalidate_simulation_cache()
        progress_label.config(text=f"Loaded CIF: {Path(_current_primary_cif_path()).name}")
        _refresh_session_summary_panel()
    except Exception as exc:
        _reset_structure_model_control_vars(
            snapshot.current_occ_values,
            snapshot.current_atom_site_values,
        )
        gui_structure_model.restore_primary_cif_reload_snapshot(
            structure_model_state,
            snapshot,
            occ_vars=_occupancy_control_vars(),
            atom_site_fract_vars=_atom_site_fractional_control_vars(),
        )
        _reset_atom_site_override_cache()
        _sync_structure_model_aliases()
        a_var.set(float(old_slider_a))
        c_var.set(float(old_slider_c))

        cif_file_var.set(snapshot.cif_file)
        progress_label.config(text=f"Failed to load CIF: {exc}")
        _refresh_session_summary_panel()


def _browse_primary_cif():
    """Open a file picker and apply the selected primary CIF path."""
    gui_structure_model.browse_primary_cif_with_dialog(
        current_cif_path=_current_primary_cif_path(),
        file_dialog_dir=get_dir("file_dialog_dir"),
        askopenfilename=filedialog.askopenfilename,
        set_cif_path_text=cif_file_var.set,
        apply_primary_cif_path=_apply_primary_cif_path,
    )


def _apply_primary_cif_from_entry(_event=None):
    """Apply the CIF path currently entered in the CIF control."""

    _apply_primary_cif_path(cif_file_var.get())


def _build_diffuse_ht_request():
    """Collect the active algebraic HT request from GUI state."""

    return gui_structure_model.build_diffuse_ht_request(
        structure_model_state,
        atom_site_override_state,
        p_values=[float(p0_var.get()), float(p1_var.get()), float(p2_var.get())],
        w_values=[float(w0_var.get()), float(w1_var.get()), float(w2_var.get())],
        a_lattice=float(a_var.get()),
        c_lattice=float(c_var.get()),
        lambda_angstrom=float(lambda_),
        mx=int(mx),
        two_theta_range=two_theta_range,
        finite_stack=bool(finite_stack_var.get()),
        stack_layers=int(max(1, stack_layers_var.get())),
        phase_delta_expression=_current_phase_delta_expression(),
        phi_l_divisor=_current_phi_l_divisor(),
        rod_points_per_gz=_current_rod_points_per_gz(),
        tcl_error_types=(tk.TclError,),
    )


def _open_diffuse_cif_toggle():
    """Open algebraic HT diffuse viewer using the active simulation inputs."""
    gui_structure_model.open_diffuse_ht_view_with_status(
        build_request=_build_diffuse_ht_request,
        open_view=(
            lambda request: open_diffuse_cif_toggle_algebraic(
                cif_path=request.active_cif,
                occ=request.occ,
                p_values=request.p_values,
                w_values=request.w_values,
                a_lattice=request.a_lattice,
                c_lattice=request.c_lattice,
                lambda_angstrom=request.lambda_angstrom,
                mx=request.mx,
                two_theta_max=request.two_theta_max,
                finite_stack=request.finite_stack,
                stack_layers=request.stack_layers,
                iodine_z=request.iodine_z,
                phase_delta_expression=request.phase_delta_expression,
                phi_l_divisor=request.phi_l_divisor,
                rod_points_per_gz=request.rod_points_per_gz,
            )
        ),
        set_status_text=lambda text: progress_label.config(text=text),
        tcl_error_types=(tk.TclError,),
    )


def _export_diffuse_ht_txt():
    """Export algebraic HT values to a fixed-width text table."""
    gui_structure_model.export_diffuse_ht_txt_with_dialog(
        build_request=_build_diffuse_ht_request,
        get_download_dir=lambda: get_dir("downloads"),
        asksaveasfilename=filedialog.asksaveasfilename,
        export_table=(
            lambda save_path, request: export_algebraic_ht_txt(
                output_path=save_path,
                cif_path=request.active_cif,
                occ=request.occ,
                p_values=request.p_values,
                w_values=request.w_values,
                a_lattice=request.a_lattice,
                c_lattice=request.c_lattice,
                lambda_angstrom=request.lambda_angstrom,
                mx=request.mx,
                two_theta_max=request.two_theta_max,
                finite_stack=request.finite_stack,
                stack_layers=request.stack_layers,
                iodine_z=request.iodine_z,
                phase_delta_expression=request.phase_delta_expression,
                phi_l_divisor=request.phi_l_divisor,
                rod_points_per_gz=request.rod_points_per_gz,
            )
        ),
        set_status_text=lambda text: progress_label.config(text=text),
        tcl_error_types=(tk.TclError,),
    )


gui_views.create_primary_cif_controls(
    parent=workspace_panels_view_state.workspace_inputs_frame,
    view_state=primary_cif_controls_view_state,
    cif_path_text=_current_primary_cif_path(),
    on_apply_from_entry=_apply_primary_cif_from_entry,
    on_browse_primary_cif=_browse_primary_cif,
    on_open_diffuse_ht=_open_diffuse_cif_toggle,
    on_export_diffuse_ht=_export_diffuse_ht_txt,
)
cif_file_var = primary_cif_controls_view_state.cif_file_var
if cif_file_var is None:
    raise RuntimeError("Primary CIF controls did not create the path variable.")
_maybe_refresh_run_status_bar()


@dataclass
class RuntimeContext:
    state: object
    simulation_runtime_state: object
    background_runtime_state: object
    app_shell_view_state: object | None = None
    workspace_panels_view_state: object | None = None
    root: object | None = None
    primary_figure: object | None = None
    primary_canvas: object | None = None
    analysis_figure: object | None = None
    analysis_canvas: object | None = None


def build_runtime_state_context() -> RuntimeContext:
    context = RuntimeContext(
        state=app_state,
        simulation_runtime_state=simulation_runtime_state,
        background_runtime_state=background_runtime_state,
    )
    globals()["app_state"] = context.state
    globals()["simulation_runtime_state"] = context.simulation_runtime_state
    globals()["background_runtime_state"] = context.background_runtime_state
    return context


def build_runtime_window_context(context: RuntimeContext) -> RuntimeContext:
    context.root = root
    context.app_shell_view_state = app_shell_view_state
    context.workspace_panels_view_state = workspace_panels_view_state
    globals()["root"] = context.root
    globals()["app_shell_view_state"] = context.app_shell_view_state
    globals()["workspace_panels_view_state"] = context.workspace_panels_view_state
    return context


def build_runtime_plot_context(context: RuntimeContext) -> RuntimeContext:
    context.primary_figure = fig
    context.primary_canvas = canvas
    context.analysis_figure = fig_1d
    context.analysis_canvas = canvas_1d
    globals()["fig"] = context.primary_figure
    globals()["canvas"] = context.primary_canvas
    globals()["fig_1d"] = context.analysis_figure
    globals()["canvas_1d"] = context.analysis_canvas
    return context


def build_runtime_controls_context(context: RuntimeContext) -> RuntimeContext:
    globals()["app_shell_view_state"] = context.app_shell_view_state
    globals()["workspace_panels_view_state"] = context.workspace_panels_view_state
    return context


_STARTUP_BENCHMARK_ENABLED = str(
    os.environ.get("RA_SIM_STARTUP_BENCHMARK", "")
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_STARTUP_BENCHMARK_AUTO_EXIT = str(
    os.environ.get("RA_SIM_STARTUP_BENCHMARK_AUTO_EXIT", "")
).strip().lower() in {"1", "true", "yes", "on"}


def _emit_startup_benchmark_event(event: str, **payload: object) -> None:
    if not _STARTUP_BENCHMARK_ENABLED:
        return
    message = {"event": str(event), **payload}
    print(f"RA_SIM_STARTUP_EVENT {json.dumps(message, sort_keys=True)}", flush=True)


def main(write_excel_flag=None, startup_mode="prompt", calibrant_bundle=None):
    """Entry point for running the GUI application.

    Parameters
    ----------
    write_excel_flag : bool or None, optional
        When ``True`` the initial intensities are written to an Excel
        file in the configured downloads directory.  When ``None`` the
        value from the instrument configuration file is used.
    startup_mode : {"prompt", "simulation", "calibrant", "mosaic"}, optional
        Startup behavior. ``prompt`` shows a launcher GUI asking which mode
        to run, ``simulation`` starts this GUI directly, and ``calibrant``
        launches the hBN calibrant fitter. ``mosaic`` launches the installed
        mosaic_sim visualizer.
    calibrant_bundle : str or None, optional
        Optional NPZ bundle path to preload when launching calibrant mode.
    """

    global write_excel
    if write_excel_flag is not None:
        write_excel = write_excel_flag

    if startup_mode not in {"prompt", "simulation", "calibrant", "mosaic"}:
        raise ValueError("startup_mode must be one of: prompt, simulation, calibrant, mosaic")

    resolved_mode = startup_mode
    if resolved_mode == "prompt":
        resolved_mode = gui_bootstrap.choose_startup_mode_dialog(root)

    if resolved_mode is None:
        _shutdown_gui()
        return

    if resolved_mode == "calibrant":
        _shutdown_gui()
        gui_bootstrap.launch_calibrant_gui(bundle=calibrant_bundle)
        return

    if resolved_mode == "mosaic":
        _shutdown_gui()
        from ra_sim import launcher as package_launcher

        package_launcher.launch_mosaic_visualizer()
        return

    runtime_context = build_runtime_state_context()
    runtime_context = build_runtime_window_context(runtime_context)
    runtime_context = build_runtime_plot_context(runtime_context)
    runtime_context = build_runtime_controls_context(runtime_context)
    _ = runtime_context

    gui_views.apply_launch_window_context(root)
    try:
        root.deiconify()
    except tk.TclError:
        pass
    _emit_startup_benchmark_event("window_deiconified")
    _set_structure_bootstrap_controls_enabled(
        bool(getattr(structure_model_state, "bootstrap_complete", False))
    )

    params_file_path = get_path("parameters_file")
    profile_loaded = False
    if os.path.exists(params_file_path):
        simulation_runtime_state.startup_updates_suspended = True
        simulation_runtime_state.pending_startup_refresh = False
        try:
            load_parameters(
                params_file_path,
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
                center_x_var,
                center_y_var,
                resolution_var,
                custom_samples_var,
                rod_points_per_gz_var,
                bandwidth_percent_var=bandwidth_percent_var,
                optics_mode_var=optics_mode_var,
                phase_delta_expr_var=phase_delta_expr_var,
                phi_l_divisor_var=phi_l_divisor_var,
                sf_prune_bias_var=sf_prune_bias_var,
                solve_q_steps_var=solve_q_steps_var,
                solve_q_rel_tol_var=solve_q_rel_tol_var,
                solve_q_mode_var=solve_q_mode_var,
            )
        finally:
            simulation_runtime_state.startup_updates_suspended = False
        if finite_stack_controls_view_state.phase_delta_entry_var is not None:
            gui_views.set_finite_stack_phase_delta_entry_text(
                finite_stack_controls_view_state,
                _current_phase_delta_expression(),
            )
        if finite_stack_controls_view_state.phi_l_divisor_entry_var is not None:
            gui_views.set_finite_stack_phi_l_divisor_entry_text(
                finite_stack_controls_view_state,
                gui_controllers.format_finite_stack_phi_l_divisor(_current_phi_l_divisor()),
            )
        _apply_rod_points_per_gz(trigger_update=False)
        ensure_valid_resolution_choice()
        profile_loaded = True
    else:
        ensure_valid_resolution_choice()

    sample_count = int(max(1, simulation_runtime_state.num_samples))
    cif_summary = Path(_current_primary_cif_path()).name
    if structure_model_state.cif_file2:
        cif_summary = f"{cif_summary}, {Path(str(structure_model_state.cif_file2)).name}"
    print(
        "Startup ready: "
        f"profile={'loaded' if profile_loaded else 'defaults'}; "
        f"sampling={sample_count} samples; "
        f"sf_prune={current_sf_prune_bias():+.2f}; "
        f"q_mode={gui_structure_factor_pruning.normalize_runtime_solve_q_mode_label(solve_q_mode_var.get())}; "
        f"q_steps={current_solve_q_values().steps}; "
        f"q_tol={current_solve_q_values().rel_tol:.2e}; "
        f"optics={_normalize_optics_mode_label(optics_mode_var.get())}; "
        f"cif={cif_summary}"
    )

    gui_controllers.clear_tk_after_token(root, simulation_runtime_state.update_pending)
    simulation_runtime_state.update_pending = None
    gui_controllers.clear_tk_after_token(
        root,
        simulation_runtime_state.integration_update_pending,
    )
    simulation_runtime_state.integration_update_pending = None

    post_startup_tasks: list[gui_runtime_startup.StartupTask] = []
    if not structure_model_controls_built:
        post_startup_tasks.append(
            gui_runtime_startup.StartupTask(
                "structure controls",
                _rebuild_structure_model_controls,
            )
        )
    if write_excel:
        post_startup_tasks.append(
            gui_runtime_startup.StartupTask(
                "initial Excel export",
                export_initial_excel,
            )
        )

    def _handle_post_startup_error(task_name: str, exc: Exception) -> None:
        progress_label.config(text=f"Startup post-processing failed during {task_name}: {exc}")
        try:
            import traceback

            traceback.print_exc()
        except Exception:
            pass

    post_startup_task_runner = gui_runtime_startup.build_runtime_startup_task_runner(
        root=root,
        tasks=post_startup_tasks,
        on_error=_handle_post_startup_error,
        initial_delay_ms=200,
        inter_task_delay_ms=75,
    )

    def _run_initial_startup_work():
        try:
            _emit_startup_benchmark_event("after_idle_startup_task")
            progress_label.config(text="Initializing structure model...")
            _bootstrap_structure_model_state_for_startup()
            _set_structure_bootstrap_controls_enabled(True)
            progress_label.config(text="Initializing simulation...")
            _show_initial_simulation_loading_overlay()
            matplotlib_canvas.draw()
            root.update_idletasks()
            do_update()
        except Exception as exc:
            progress_label.config(text=f"Startup initialization failed: {exc}")
            try:
                import traceback

                traceback.print_exc()
            except Exception:
                pass
        else:
            if (
                simulation_runtime_state.worker_active_job is not None
                and simulation_runtime_state.stored_sim_image is None
            ):
                progress_label.config(text="Simulation loading in background...")
            elif (
                simulation_runtime_state.preview_active
                and simulation_runtime_state.worker_active_job is not None
            ):
                progress_label.config(text="Preview ready, refining full simulation...")
            elif simulation_runtime_state.analysis_active_job is not None:
                progress_label.config(text=_analysis_progress_text())
            else:
                progress_label.config(text="Simulation ready.")
            _emit_startup_benchmark_event(
                "ready_status",
                text=str(progress_label.cget("text")),
            )
            if post_startup_task_runner.has_tasks():
                post_startup_task_runner.schedule()
            if _STARTUP_BENCHMARK_AUTO_EXIT:
                root.after(750 if post_startup_task_runner.has_tasks() else 150, _shutdown_gui)

    # Start the exact-cake Numba warmup in the background so the first caked
    # conversion is less likely to pay the JIT compile cost on the UI path.
    start_exact_cake_numba_warmup_in_background()

    # Let Tk paint the windows first, then run the expensive initial update.
    root.after_idle(_run_initial_startup_work)
    root.mainloop()


if __name__ == "__main__":
    cli_argv = list(sys.argv[1:])
    early_mode = os.environ.get("RA_SIM_EARLY_STARTUP_MODE")
    args = gui_bootstrap.parse_launch_args(cli_argv)

    try:
        override_flag = False if args.no_excel else None
        mode = gui_bootstrap.resolve_startup_mode(
            args.command,
            early_mode=early_mode,
        )
        main(
            write_excel_flag=override_flag,
            startup_mode=mode,
            calibrant_bundle=args.bundle,
        )
    except Exception as exc:
        print("Unhandled exception during startup:", exc)
        import traceback

        traceback.print_exc()
