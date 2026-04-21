"""Workflow helpers for HKL lookup and selected-peak state."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ra_sim.simulation.intersection_cache_schema import (
    extract_cache_row_provenance,
    extract_cached_caked_angles,
)
from ra_sim.simulation.diffraction import get_process_peaks_runtime_kwargs
from ra_sim.debug_controls import retain_optional_cache
from ra_sim.utils.calculations import (
    _n2_wavelength_snapshot_from_angstrom,
    _normalize_n2_source_meta,
    resolve_index_of_refraction_array,
    source_branch_index_from_phi_deg,
)
from . import views as gui_views
from . import manual_geometry as gui_manual_geometry
from . import mosaic_top_selection as gui_mosaic_top


@dataclass(frozen=True)
class SelectedPeakIntersectionConfig:
    """Scalar GUI inputs needed to launch one selected peak in the external viewer."""

    image_size: int
    center_col: float
    center_row: float
    distance_cor_to_detector: float
    gamma_deg: float
    Gamma_deg: float
    chi_deg: float
    psi_deg: float
    psi_z_deg: float
    zs: float
    zb: float
    theta_initial_deg: float
    cor_angle_deg: float
    sigma_mosaic_deg: float
    gamma_mosaic_deg: float
    eta: float
    sample_width_m: float
    sample_length_m: float
    wavelength_angstrom: float
    solve_q_steps: int
    solve_q_rel_tol: float
    solve_q_mode: int
    pixel_size_m: float = 100e-6


_SPECULAR_VIEW_DEFAULT_SAMPLE_WIDTH_M = 0.02
_SPECULAR_VIEW_DEFAULT_SAMPLE_HEIGHT_M = 0.08
_SPECULAR_VIEW_DEFAULT_DETECTOR_DISTANCE_M = 0.075
_SPECULAR_VIEW_DEFAULT_PIXEL_SIZE_M = 100e-6
_PEAK_CLICK_INDEX_CELL_SIZE_PX = 50.0
_REFINED_SIMULATION_SIGNATURE_KEY = "_refined_simulation_signature"


def _retain_peak_overlay_cache() -> bool:
    return retain_optional_cache("peak_overlay", feature_needed=True)


def _empty_peak_overlay_cache() -> dict[str, object]:
    return {
        "sig": None,
        "positions": [],
        "millers": [],
        "intensities": [],
        "records": [],
        "click_spatial_index": None,
        "peak_positions_filtered": False,
        "restored_from_gui_state": False,
    }


@dataclass(frozen=True)
class SelectedPeakCanvasPickConfig:
    """Inputs needed to resolve one HKL image-pick click."""

    image_size: int
    primary_a: float
    primary_c: float
    max_distance_px: float
    min_separation_px: float
    image_shape: tuple[int, ...] | None = None


@dataclass(frozen=True)
class SelectedPeakIdealCenterProbeConfig:
    """Inputs needed to simulate one idealized selected-peak center."""

    image_size: int
    lattice_a: float
    lattice_c: float
    wavelength: float
    distance_cor_to_detector: float
    gamma_deg: float
    Gamma_deg: float
    chi_deg: float
    psi_deg: float
    psi_z_deg: float
    zs: float
    zb: float
    debye_x: float
    debye_y: float
    detector_center: tuple[float, float]
    theta_initial_deg: float
    cor_angle_deg: float
    optics_mode: int
    solve_q_steps: int
    solve_q_rel_tol: float
    solve_q_mode: int
    unit_x: tuple[float, float, float] = (1.0, 0.0, 0.0)
    n_detector: tuple[float, float, float] = (0.0, 1.0, 0.0)


@dataclass
class SelectedPeakRuntimeBindings:
    """Runtime callbacks and shared state used by selected-peak workflows."""

    simulation_runtime_state: Any
    peak_selection_state: Any
    hkl_lookup_view_state: Any
    selected_peak_marker: Any
    current_primary_a_factory: object
    caked_view_enabled_factory: object
    current_canvas_pick_config_factory: object
    current_intersection_config_factory: object
    ensure_peak_overlay_data: Callable[..., object]
    sync_peak_selection_state: Callable[[], None] | None = None
    schedule_update: Callable[[], None] | None = None
    set_status_text: Callable[[str], None] | None = None
    draw_idle: Callable[[], None] | None = None
    display_to_native_sim_coords: Callable[..., tuple[float, float]] | None = None
    detector_display_to_native_detector_coords: (
        Callable[[float, float], tuple[float, float] | None] | None
    ) = None
    native_sim_to_display_coords: Callable[..., tuple[float, float]] | None = None
    native_detector_coords_to_detector_display_coords: (
        Callable[[float, float], tuple[float, float] | None] | None
    ) = None
    caked_angles_to_detector_display_coords: (
        Callable[[float, float], tuple[float, float] | None] | None
    ) = None
    hkl_pick_simulation_points_factory: Callable[[], object] | None = None
    simulate_ideal_hkl_native_center: Callable[..., tuple[float, float] | None] | None = None
    deactivate_conflicting_modes: Callable[[], None] | None = None
    on_hkl_pick_mode_changed: Callable[[bool], None] | None = None
    n2: Any = None
    tcl_error_types: tuple[type[BaseException], ...] = ()


@dataclass(frozen=True)
class SelectedPeakRuntimeCallbacks:
    """Bound callbacks for the runtime selected-peak workflow."""

    update_hkl_pick_button_label: Callable[[], None]
    set_hkl_pick_mode: Callable[[bool, str | None], None]
    toggle_hkl_pick_mode: Callable[[], None]
    reselect_current_peak: Callable[[], bool]
    select_peak_from_hkl_controls: Callable[[], bool]
    clear_selected_peak: Callable[[], None]
    open_selected_peak_intersection_figure: Callable[[], bool]
    select_peak_from_canvas_click: Callable[[float, float], bool]
    find_peak_record_for_canvas_click: Callable[
        [float, float, float],
        tuple[int, dict[str, object] | None, float, bool],
    ]


@dataclass(frozen=True)
class SelectedPeakRuntimeMaintenanceCallbacks:
    """Bound maintenance callbacks around runtime selected-peak refresh flows."""

    refresh_after_simulation_update: Callable[[bool], bool]
    apply_restored_selected_hkl_target: Callable[[object], tuple[int, int, int] | None]


@dataclass(frozen=True)
class SelectedPeakRuntimeConfigFactories:
    """Live config factories used by the selected-peak runtime workflow."""

    canvas_pick: Callable[[], SelectedPeakCanvasPickConfig]
    intersection: Callable[[], SelectedPeakIntersectionConfig]
    ideal_center: Callable[[float, float, float, float, float], tuple[float, float] | None]


def _resolve_runtime_value(value_or_callable: object) -> object:
    if callable(value_or_callable):
        try:
            return value_or_callable()
        except Exception:
            return None
    return value_or_callable


def _set_status_text(setter: Callable[[str], None] | None, text: str) -> None:
    if callable(setter):
        setter(str(text))


def _sync_runtime_peak_selection_state(bindings: SelectedPeakRuntimeBindings) -> None:
    if callable(bindings.sync_peak_selection_state):
        bindings.sync_peak_selection_state()


def _runtime_draw_idle(bindings: SelectedPeakRuntimeBindings) -> None:
    if callable(bindings.draw_idle):
        bindings.draw_idle()


def _hide_runtime_selected_peak_marker(bindings: SelectedPeakRuntimeBindings) -> None:
    marker = bindings.selected_peak_marker
    set_visible = getattr(marker, "set_visible", None)
    if callable(set_visible):
        try:
            set_visible(False)
        except Exception:
            return


def _runtime_primary_a(bindings: SelectedPeakRuntimeBindings) -> float:
    try:
        return float(_resolve_runtime_value(bindings.current_primary_a_factory))
    except Exception:
        return float("nan")


def _runtime_canvas_pick_config(
    bindings: SelectedPeakRuntimeBindings,
) -> SelectedPeakCanvasPickConfig | None:
    config = _resolve_runtime_value(bindings.current_canvas_pick_config_factory)
    if isinstance(config, SelectedPeakCanvasPickConfig):
        return config
    return None


def _runtime_intersection_config(
    bindings: SelectedPeakRuntimeBindings,
) -> SelectedPeakIntersectionConfig | None:
    config = _resolve_runtime_value(bindings.current_intersection_config_factory)
    if isinstance(config, SelectedPeakIntersectionConfig):
        return config
    return None


def build_selected_peak_canvas_pick_config(
    *,
    image_size: int,
    primary_a: float,
    primary_c: float,
    max_distance_px: float,
    min_separation_px: float,
    image_shape: tuple[int, ...] | None = None,
) -> SelectedPeakCanvasPickConfig:
    """Build one validated selected-peak canvas-pick config."""

    normalized_shape = tuple(int(v) for v in image_shape) if image_shape is not None else None
    return SelectedPeakCanvasPickConfig(
        image_size=int(image_size),
        primary_a=float(primary_a),
        primary_c=float(primary_c),
        max_distance_px=float(max_distance_px),
        min_separation_px=float(min_separation_px),
        image_shape=normalized_shape,
    )


def build_selected_peak_intersection_config(
    *,
    image_size: int,
    center_col: float,
    center_row: float,
    distance_cor_to_detector: float,
    gamma_deg: float,
    Gamma_deg: float,
    chi_deg: float,
    psi_deg: float,
    psi_z_deg: float,
    zs: float,
    zb: float,
    theta_initial_deg: float,
    cor_angle_deg: float,
    sigma_mosaic_deg: float,
    gamma_mosaic_deg: float,
    eta: float,
    sample_width_m: float,
    sample_length_m: float,
    wavelength_angstrom: float,
    solve_q_steps: int,
    solve_q_rel_tol: float,
    solve_q_mode: int,
    pixel_size_m: float = 100e-6,
) -> SelectedPeakIntersectionConfig:
    """Build one validated selected-peak intersection config."""

    return SelectedPeakIntersectionConfig(
        image_size=int(image_size),
        center_col=float(center_col),
        center_row=float(center_row),
        distance_cor_to_detector=float(distance_cor_to_detector),
        gamma_deg=float(gamma_deg),
        Gamma_deg=float(Gamma_deg),
        chi_deg=float(chi_deg),
        psi_deg=float(psi_deg),
        psi_z_deg=float(psi_z_deg),
        zs=float(zs),
        zb=float(zb),
        theta_initial_deg=float(theta_initial_deg),
        cor_angle_deg=float(cor_angle_deg),
        sigma_mosaic_deg=float(sigma_mosaic_deg),
        gamma_mosaic_deg=float(gamma_mosaic_deg),
        eta=float(eta),
        sample_width_m=float(sample_width_m),
        sample_length_m=float(sample_length_m),
        wavelength_angstrom=float(wavelength_angstrom),
        solve_q_steps=int(solve_q_steps),
        solve_q_rel_tol=float(solve_q_rel_tol),
        solve_q_mode=int(solve_q_mode),
        pixel_size_m=float(pixel_size_m),
    )


def build_selected_peak_ideal_center_probe_config(
    *,
    image_size: int,
    lattice_a: float,
    lattice_c: float,
    wavelength: float,
    distance_cor_to_detector: float,
    gamma_deg: float,
    Gamma_deg: float,
    chi_deg: float,
    psi_deg: float,
    psi_z_deg: float,
    zs: float,
    zb: float,
    debye_x: float,
    debye_y: float,
    detector_center: tuple[float, float],
    theta_initial_deg: float,
    cor_angle_deg: float,
    optics_mode: int,
    solve_q_steps: int,
    solve_q_rel_tol: float,
    solve_q_mode: int,
    unit_x: tuple[float, float, float] = (1.0, 0.0, 0.0),
    n_detector: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> SelectedPeakIdealCenterProbeConfig:
    """Build one validated ideal-center probe config."""

    return SelectedPeakIdealCenterProbeConfig(
        image_size=int(image_size),
        lattice_a=float(lattice_a),
        lattice_c=float(lattice_c),
        wavelength=float(wavelength),
        distance_cor_to_detector=float(distance_cor_to_detector),
        gamma_deg=float(gamma_deg),
        Gamma_deg=float(Gamma_deg),
        chi_deg=float(chi_deg),
        psi_deg=float(psi_deg),
        psi_z_deg=float(psi_z_deg),
        zs=float(zs),
        zb=float(zb),
        debye_x=float(debye_x),
        debye_y=float(debye_y),
        detector_center=(
            float(detector_center[0]),
            float(detector_center[1]),
        ),
        theta_initial_deg=float(theta_initial_deg),
        cor_angle_deg=float(cor_angle_deg),
        optics_mode=int(optics_mode),
        solve_q_steps=int(solve_q_steps),
        solve_q_rel_tol=float(solve_q_rel_tol),
        solve_q_mode=int(solve_q_mode),
        unit_x=tuple(float(v) for v in unit_x),
        n_detector=tuple(float(v) for v in n_detector),
    )


def _runtime_float(value_or_callable: object, default: float = 0.0) -> float:
    try:
        return float(_resolve_runtime_value(value_or_callable))
    except Exception:
        return float(default)


def _runtime_int(value_or_callable: object, default: int = 0) -> int:
    try:
        return int(_resolve_runtime_value(value_or_callable))
    except Exception:
        return int(default)


def _runtime_bool(value_or_callable: object, default: bool = False) -> bool:
    try:
        return bool(_resolve_runtime_value(value_or_callable))
    except Exception:
        return bool(default)


def _runtime_sequence(value_or_callable: object) -> Sequence[object] | None:
    raw_value = _resolve_runtime_value(value_or_callable)
    if raw_value is None:
        return None
    if hasattr(raw_value, "shape") and not isinstance(raw_value, (str, bytes)):
        try:
            raw_value = tuple(int(v) for v in raw_value.shape)
        except Exception:
            raw_value = getattr(raw_value, "shape", raw_value)
    if isinstance(raw_value, np.ndarray):
        return tuple(raw_value.tolist())
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
        return raw_value
    return None


def _runtime_float_pair(
    value_or_callable: object,
    *,
    default: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float]:
    raw_value = _runtime_sequence(value_or_callable)
    if raw_value is None or len(raw_value) < 2:
        return (float(default[0]), float(default[1]))
    try:
        return (float(raw_value[0]), float(raw_value[1]))
    except Exception:
        return (float(default[0]), float(default[1]))


def _runtime_selected_peak_solve_q_fields(
    value_or_callable: object,
) -> tuple[int, float, int]:
    raw_value = _resolve_runtime_value(value_or_callable)
    if isinstance(raw_value, Mapping):
        return (
            _runtime_int(raw_value.get("steps"), 0),
            _runtime_float(raw_value.get("rel_tol"), 0.0),
            _runtime_int(
                raw_value.get("mode_flag", raw_value.get("mode", 0)),
                0,
            ),
        )
    return (
        _runtime_int(getattr(raw_value, "steps", 0), 0),
        _runtime_float(getattr(raw_value, "rel_tol", 0.0), 0.0),
        _runtime_int(getattr(raw_value, "mode_flag", 0), 0),
    )


def build_runtime_selected_peak_canvas_pick_config(
    *,
    image_size: int,
    primary_a: object,
    primary_c: object,
    max_distance_px: object,
    min_separation_px: object,
    image_shape: object = None,
) -> SelectedPeakCanvasPickConfig:
    """Build one live canvas-pick config from runtime scalar sources."""

    normalized_shape = _runtime_sequence(image_shape)
    if normalized_shape is None or len(normalized_shape) == 0:
        normalized_shape = (int(image_size), int(image_size))
    return build_selected_peak_canvas_pick_config(
        image_size=int(image_size),
        primary_a=_runtime_float(primary_a, float("nan")),
        primary_c=_runtime_float(primary_c, float("nan")),
        max_distance_px=_runtime_float(max_distance_px, 0.0),
        min_separation_px=_runtime_float(min_separation_px, 0.0),
        image_shape=tuple(int(v) for v in normalized_shape),
    )


def build_runtime_selected_peak_intersection_config(
    *,
    image_size: int,
    center_col: object,
    center_row: object,
    distance_cor_to_detector: object,
    gamma_deg: object,
    Gamma_deg: object,
    chi_deg: object,
    psi_deg: object,
    psi_z_deg: object,
    zs: object,
    zb: object,
    theta_initial_deg: object,
    cor_angle_deg: object,
    sigma_mosaic_deg: object,
    gamma_mosaic_deg: object,
    eta: object,
    sample_width_m: object,
    sample_length_m: object,
    wavelength_angstrom: object,
    solve_q_values: object,
    pixel_size_m: object = _SPECULAR_VIEW_DEFAULT_PIXEL_SIZE_M,
) -> SelectedPeakIntersectionConfig:
    """Build one live selected-peak intersection config from runtime sources."""

    solve_q_steps, solve_q_rel_tol, solve_q_mode = _runtime_selected_peak_solve_q_fields(
        solve_q_values
    )
    return build_selected_peak_intersection_config(
        image_size=int(image_size),
        center_col=_runtime_float(center_col, float("nan")),
        center_row=_runtime_float(center_row, float("nan")),
        distance_cor_to_detector=_runtime_float(
            distance_cor_to_detector,
            float("nan"),
        ),
        gamma_deg=_runtime_float(gamma_deg, float("nan")),
        Gamma_deg=_runtime_float(Gamma_deg, float("nan")),
        chi_deg=_runtime_float(chi_deg, float("nan")),
        psi_deg=_runtime_float(psi_deg, float("nan")),
        psi_z_deg=_runtime_float(psi_z_deg, float("nan")),
        zs=_runtime_float(zs, float("nan")),
        zb=_runtime_float(zb, float("nan")),
        theta_initial_deg=_runtime_float(theta_initial_deg, float("nan")),
        cor_angle_deg=_runtime_float(cor_angle_deg, float("nan")),
        sigma_mosaic_deg=_runtime_float(sigma_mosaic_deg, float("nan")),
        gamma_mosaic_deg=_runtime_float(gamma_mosaic_deg, float("nan")),
        eta=_runtime_float(eta, float("nan")),
        sample_width_m=_runtime_float(sample_width_m, float("nan")),
        sample_length_m=_runtime_float(sample_length_m, float("nan")),
        wavelength_angstrom=_runtime_float(wavelength_angstrom, float("nan")),
        solve_q_steps=solve_q_steps,
        solve_q_rel_tol=solve_q_rel_tol,
        solve_q_mode=solve_q_mode,
        pixel_size_m=_runtime_float(
            pixel_size_m,
            _SPECULAR_VIEW_DEFAULT_PIXEL_SIZE_M,
        ),
    )


def build_runtime_selected_peak_ideal_center_probe_config(
    *,
    image_size: int,
    lattice_a: object,
    lattice_c: object,
    wavelength: object,
    distance_cor_to_detector: object,
    gamma_deg: object,
    Gamma_deg: object,
    chi_deg: object,
    psi_deg: object,
    psi_z_deg: object,
    zs: object,
    zb: object,
    debye_x: object,
    debye_y: object,
    detector_center: object,
    theta_initial_deg: object,
    cor_angle_deg: object,
    optics_mode: object,
    solve_q_values: object,
) -> SelectedPeakIdealCenterProbeConfig:
    """Build one live ideal-center probe config from runtime scalar sources."""

    solve_q_steps, solve_q_rel_tol, solve_q_mode = _runtime_selected_peak_solve_q_fields(
        solve_q_values
    )
    return build_selected_peak_ideal_center_probe_config(
        image_size=int(image_size),
        lattice_a=_runtime_float(lattice_a, float("nan")),
        lattice_c=_runtime_float(lattice_c, float("nan")),
        wavelength=_runtime_float(wavelength, float("nan")),
        distance_cor_to_detector=_runtime_float(
            distance_cor_to_detector,
            float("nan"),
        ),
        gamma_deg=_runtime_float(gamma_deg, float("nan")),
        Gamma_deg=_runtime_float(Gamma_deg, float("nan")),
        chi_deg=_runtime_float(chi_deg, float("nan")),
        psi_deg=_runtime_float(psi_deg, float("nan")),
        psi_z_deg=_runtime_float(psi_z_deg, float("nan")),
        zs=_runtime_float(zs, float("nan")),
        zb=_runtime_float(zb, float("nan")),
        debye_x=_runtime_float(debye_x, float("nan")),
        debye_y=_runtime_float(debye_y, float("nan")),
        detector_center=_runtime_float_pair(detector_center),
        theta_initial_deg=_runtime_float(theta_initial_deg, float("nan")),
        cor_angle_deg=_runtime_float(cor_angle_deg, float("nan")),
        optics_mode=_runtime_int(optics_mode, 0),
        solve_q_steps=solve_q_steps,
        solve_q_rel_tol=solve_q_rel_tol,
        solve_q_mode=solve_q_mode,
    )


def make_runtime_selected_peak_canvas_pick_config_factory(
    *,
    image_size: int,
    primary_a_factory: object,
    primary_c_factory: object,
    max_distance_px: object,
    min_separation_px: object,
    image_shape_factory: object = None,
) -> Callable[[], SelectedPeakCanvasPickConfig]:
    """Return a zero-arg factory for the live canvas-pick config."""

    return lambda: build_runtime_selected_peak_canvas_pick_config(
        image_size=int(image_size),
        primary_a=primary_a_factory,
        primary_c=primary_c_factory,
        max_distance_px=max_distance_px,
        min_separation_px=min_separation_px,
        image_shape=image_shape_factory,
    )


def make_runtime_selected_peak_intersection_config_factory(
    *,
    image_size: int,
    center_col_factory: object,
    center_row_factory: object,
    distance_cor_to_detector_factory: object,
    gamma_deg_factory: object,
    Gamma_deg_factory: object,
    chi_deg_factory: object,
    psi_deg_factory: object,
    psi_z_deg_factory: object,
    zs_factory: object,
    zb_factory: object,
    theta_initial_deg_factory: object,
    cor_angle_deg_factory: object,
    sigma_mosaic_deg_factory: object,
    gamma_mosaic_deg_factory: object,
    eta_factory: object,
    sample_width_m_factory: object,
    sample_length_m_factory: object,
    wavelength_angstrom_factory: object,
    solve_q_values_factory: object,
    pixel_size_m_factory: object = _SPECULAR_VIEW_DEFAULT_PIXEL_SIZE_M,
) -> Callable[[], SelectedPeakIntersectionConfig]:
    """Return a zero-arg factory for the live intersection-analysis config."""

    return lambda: build_runtime_selected_peak_intersection_config(
        image_size=int(image_size),
        center_col=center_col_factory,
        center_row=center_row_factory,
        distance_cor_to_detector=distance_cor_to_detector_factory,
        gamma_deg=gamma_deg_factory,
        Gamma_deg=Gamma_deg_factory,
        chi_deg=chi_deg_factory,
        psi_deg=psi_deg_factory,
        psi_z_deg=psi_z_deg_factory,
        zs=zs_factory,
        zb=zb_factory,
        theta_initial_deg=theta_initial_deg_factory,
        cor_angle_deg=cor_angle_deg_factory,
        sigma_mosaic_deg=sigma_mosaic_deg_factory,
        gamma_mosaic_deg=gamma_mosaic_deg_factory,
        eta=eta_factory,
        sample_width_m=sample_width_m_factory,
        sample_length_m=sample_length_m_factory,
        wavelength_angstrom=wavelength_angstrom_factory,
        solve_q_values=solve_q_values_factory,
        pixel_size_m=pixel_size_m_factory,
    )


def make_runtime_selected_peak_ideal_center_factory(
    *,
    simulation_runtime_state,
    image_size: int,
    wavelength_factory: object,
    distance_cor_to_detector_factory: object,
    gamma_deg_factory: object,
    Gamma_deg_factory: object,
    chi_deg_factory: object,
    psi_deg_factory: object,
    psi_z_deg_factory: object,
    zs_factory: object,
    zb_factory: object,
    debye_x_factory: object,
    debye_y_factory: object,
    detector_center_factory: object,
    theta_initial_deg_factory: object,
    cor_angle_deg_factory: object,
    optics_mode_factory: object,
    solve_q_values_factory: object,
    n2: Any,
    process_peaks_parallel: Callable[..., object],
) -> Callable[[float, float, float, float, float], tuple[float, float] | None]:
    """Return the live ideal-center probe callback used by canvas HKL picking."""

    def _simulate(
        h: float,
        k: float,
        l: float,
        lattice_a: float,
        lattice_c: float,
    ) -> tuple[float, float] | None:
        return simulate_ideal_hkl_native_center(
            simulation_runtime_state,
            h,
            k,
            l,
            config=build_runtime_selected_peak_ideal_center_probe_config(
                image_size=int(image_size),
                lattice_a=lattice_a,
                lattice_c=lattice_c,
                wavelength=wavelength_factory,
                distance_cor_to_detector=distance_cor_to_detector_factory,
                gamma_deg=gamma_deg_factory,
                Gamma_deg=Gamma_deg_factory,
                chi_deg=chi_deg_factory,
                psi_deg=psi_deg_factory,
                psi_z_deg=psi_z_deg_factory,
                zs=zs_factory,
                zb=zb_factory,
                debye_x=debye_x_factory,
                debye_y=debye_y_factory,
                detector_center=detector_center_factory,
                theta_initial_deg=theta_initial_deg_factory,
                cor_angle_deg=cor_angle_deg_factory,
                optics_mode=optics_mode_factory,
                solve_q_values=solve_q_values_factory,
            ),
            n2=n2,
            process_peaks_parallel=process_peaks_parallel,
        )

    return _simulate


def make_runtime_selected_peak_config_factories(
    *,
    simulation_runtime_state,
    image_size: int,
    primary_a_factory: object,
    primary_c_factory: object,
    max_distance_px: object,
    min_separation_px: object,
    image_shape_factory: object,
    center_col_factory: object,
    center_row_factory: object,
    distance_cor_to_detector_factory: object,
    gamma_deg_factory: object,
    Gamma_deg_factory: object,
    chi_deg_factory: object,
    psi_deg_factory: object,
    psi_z_deg_factory: object,
    zs_factory: object,
    zb_factory: object,
    theta_initial_deg_factory: object,
    cor_angle_deg_factory: object,
    sigma_mosaic_deg_factory: object,
    gamma_mosaic_deg_factory: object,
    eta_factory: object,
    wavelength_factory: object,
    sample_width_m_factory: object,
    sample_length_m_factory: object,
    pixel_size_m_factory: object,
    debye_x_factory: object,
    debye_y_factory: object,
    detector_center_factory: object,
    optics_mode_factory: object,
    solve_q_values_factory: object,
    n2: Any,
    process_peaks_parallel: Callable[..., object],
) -> SelectedPeakRuntimeConfigFactories:
    """Return the shared selected-peak runtime config factory bundle."""

    return SelectedPeakRuntimeConfigFactories(
        canvas_pick=make_runtime_selected_peak_canvas_pick_config_factory(
            image_size=int(image_size),
            primary_a_factory=primary_a_factory,
            primary_c_factory=primary_c_factory,
            max_distance_px=max_distance_px,
            min_separation_px=min_separation_px,
            image_shape_factory=image_shape_factory,
        ),
        intersection=make_runtime_selected_peak_intersection_config_factory(
            image_size=int(image_size),
            center_col_factory=center_col_factory,
            center_row_factory=center_row_factory,
            distance_cor_to_detector_factory=distance_cor_to_detector_factory,
            gamma_deg_factory=gamma_deg_factory,
            Gamma_deg_factory=Gamma_deg_factory,
            chi_deg_factory=chi_deg_factory,
            psi_deg_factory=psi_deg_factory,
            psi_z_deg_factory=psi_z_deg_factory,
            zs_factory=zs_factory,
            zb_factory=zb_factory,
            theta_initial_deg_factory=theta_initial_deg_factory,
            cor_angle_deg_factory=cor_angle_deg_factory,
            sigma_mosaic_deg_factory=sigma_mosaic_deg_factory,
            gamma_mosaic_deg_factory=gamma_mosaic_deg_factory,
            eta_factory=eta_factory,
            sample_width_m_factory=sample_width_m_factory,
            sample_length_m_factory=sample_length_m_factory,
            wavelength_angstrom_factory=wavelength_factory,
            solve_q_values_factory=solve_q_values_factory,
            pixel_size_m_factory=pixel_size_m_factory,
        ),
        ideal_center=make_runtime_selected_peak_ideal_center_factory(
            simulation_runtime_state=simulation_runtime_state,
            image_size=int(image_size),
            wavelength_factory=wavelength_factory,
            distance_cor_to_detector_factory=distance_cor_to_detector_factory,
            gamma_deg_factory=gamma_deg_factory,
            Gamma_deg_factory=Gamma_deg_factory,
            chi_deg_factory=chi_deg_factory,
            psi_deg_factory=psi_deg_factory,
            psi_z_deg_factory=psi_z_deg_factory,
            zs_factory=zs_factory,
            zb_factory=zb_factory,
            debye_x_factory=debye_x_factory,
            debye_y_factory=debye_y_factory,
            detector_center_factory=detector_center_factory,
            theta_initial_deg_factory=theta_initial_deg_factory,
            cor_angle_deg_factory=cor_angle_deg_factory,
            optics_mode_factory=optics_mode_factory,
            solve_q_values_factory=solve_q_values_factory,
            n2=n2,
            process_peaks_parallel=process_peaks_parallel,
        ),
    )


def hkl_pick_button_text(armed: bool) -> str:
    """Return the current HKL image-pick button label."""

    return "Pick HKL on Image (Armed)" if bool(armed) else "Pick HKL on Image"


def nearest_integer_hkl(h: float, k: float, l: float) -> tuple[int, int, int]:
    """Round one HKL triplet to the nearest integer indices."""

    return (
        int(np.rint(float(h))),
        int(np.rint(float(k))),
        int(np.rint(float(l))),
    )


def format_hkl_triplet(h: int, k: int, l: int) -> str:
    """Return a compact string for one integer HKL triplet."""

    return f"({int(h)} {int(k)} {int(l)})"


def source_miller_for_label(simulation_runtime_state, source_label: str | None) -> np.ndarray:
    """Return the active reflection table for one source label."""

    label = str(source_label or "primary").lower()
    if (
        label == "secondary"
        and isinstance(simulation_runtime_state.sim_miller2, np.ndarray)
        and simulation_runtime_state.sim_miller2.size
    ):
        return np.asarray(simulation_runtime_state.sim_miller2, dtype=float)
    if (
        isinstance(simulation_runtime_state.sim_miller1, np.ndarray)
        and simulation_runtime_state.sim_miller1.size
    ):
        return np.asarray(simulation_runtime_state.sim_miller1, dtype=float)
    return np.empty((0, 3), dtype=float)


def degenerate_hkls_for_qr(
    simulation_runtime_state,
    h: int,
    k: int,
    l: int,
    *,
    source_label: str | None,
) -> list[tuple[int, int, int]]:
    """Return all integer HKLs that share the same Qr rod and nearest L."""

    source_miller = source_miller_for_label(simulation_runtime_state, source_label)
    if source_miller.ndim != 2 or source_miller.shape[1] < 3 or source_miller.shape[0] == 0:
        return []

    finite = (
        np.isfinite(source_miller[:, 0])
        & np.isfinite(source_miller[:, 1])
        & np.isfinite(source_miller[:, 2])
    )
    if not np.any(finite):
        return []
    source_int = np.rint(source_miller[finite, :3]).astype(np.int64, copy=False)

    h_vals = source_int[:, 0]
    k_vals = source_int[:, 1]
    l_vals = source_int[:, 2]
    m_vals = h_vals * h_vals + h_vals * k_vals + k_vals * k_vals
    m_target = int(h * h + h * k + k * k)

    mask = (l_vals == int(l)) & (m_vals == m_target)
    if not np.any(mask):
        rod_mask = m_vals == m_target
        if not np.any(rod_mask):
            return []
        nearest_l = int(l_vals[rod_mask][np.argmin(np.abs(l_vals[rod_mask] - int(l)))])
        mask = rod_mask & (l_vals == nearest_l)

    out: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for hh, kk, ll in source_int[mask]:
        key = (int(hh), int(kk), int(ll))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)

    out.sort(key=lambda vals: (vals[0], vals[1], vals[2]))
    return out


def selected_peak_qr_and_degenerates(
    simulation_runtime_state,
    H: float,
    K: float,
    L: float,
    selected_peak: Mapping[str, object] | None,
    *,
    primary_a: float,
) -> tuple[float, list[tuple[int, int, int]]]:
    """Return the selected peak's Qr value and degenerate HKL set."""

    h_raw, k_raw, l_raw = float(H), float(K), float(L)
    source_label = "primary"
    a_used = float(primary_a)
    if isinstance(selected_peak, Mapping):
        raw_hkl = selected_peak.get("hkl_raw")
        if isinstance(raw_hkl, (list, tuple, np.ndarray)) and len(raw_hkl) >= 3:
            h_raw = float(raw_hkl[0])
            k_raw = float(raw_hkl[1])
            l_raw = float(raw_hkl[2])
        source_label = str(selected_peak.get("source_label", "primary"))
        try:
            a_used = float(selected_peak.get("av", primary_a))
        except (TypeError, ValueError):
            a_used = float(primary_a)

    h_int, k_int, l_int = nearest_integer_hkl(h_raw, k_raw, l_raw)
    m_val = float(h_int * h_int + h_int * k_int + k_int * k_int)
    if a_used > 0.0 and np.isfinite(a_used) and m_val >= 0.0:
        qr_val = (2.0 * np.pi / a_used) * np.sqrt((4.0 / 3.0) * m_val)
    else:
        qr_val = float("nan")

    deg_hkls = degenerate_hkls_for_qr(
        simulation_runtime_state,
        h_int,
        k_int,
        l_int,
        source_label=source_label,
    )
    if not deg_hkls:
        deg_hkls = [(h_int, k_int, l_int)]
    return float(qr_val), deg_hkls


def build_selected_peak_status_text(
    *,
    prefix: str,
    h: int,
    k: int,
    l: int,
    display_col: float,
    display_row: float,
    intensity: float,
    qr_val: float,
    deg_hkls: list[tuple[int, int, int]],
) -> str:
    """Build the GUI status text for one selected Bragg peak."""

    shown_deg = deg_hkls[:12]
    deg_text = ", ".join(format_hkl_triplet(hv, kv, lv) for hv, kv, lv in shown_deg)
    if len(deg_hkls) > len(shown_deg):
        deg_text += f", ... (+{len(deg_hkls) - len(shown_deg)} more)"
    qr_text = f"  Qr={qr_val:.4f} A^-1" if np.isfinite(qr_val) else ""
    return (
        f"{prefix}: HKL=({int(h)} {int(k)} {int(l)})  "
        f"pixel=({float(display_col):.1f},{float(display_row):.1f})  "
        f"I={float(intensity):.2g}{qr_text}  HKLs@same Qr,L: {deg_text}"
    )


def brightest_hit_native_from_table(
    hit_table: object,
) -> tuple[float, float] | None:
    """Return the native detector center for the strongest valid hit row."""

    try:
        arr = np.asarray(hit_table, dtype=float)
    except Exception:
        return None

    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 3:
        return None

    finite_mask = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & np.isfinite(arr[:, 2])
    if not np.any(finite_mask):
        return None

    valid = arr[finite_mask]
    best_idx = int(np.argmax(valid[:, 0]))
    return float(valid[best_idx, 1]), float(valid[best_idx, 2])


def simulate_ideal_hkl_native_center(
    simulation_runtime_state,
    h: float,
    k: float,
    l: float,
    *,
    config: SelectedPeakIdealCenterProbeConfig,
    n2: Any,
    process_peaks_parallel: Any,
) -> tuple[float, float] | None:
    """Simulate one idealized HKL and return its native detector center."""

    def _resolve_probe_n2_sample_array_override(
        profile_cache: Mapping[str, object],
        wavelength_arr: np.ndarray,
    ) -> np.ndarray | None:
        wavelength_arr = _n2_wavelength_snapshot_from_angstrom(wavelength_arr)
        if wavelength_arr.size == 0:
            return None

        cached_probe_n2_fallback = None
        cached_n2 = profile_cache.get("n2_sample_array")
        if cached_n2 is not None:
            try:
                normalized_cached_n2 = np.ascontiguousarray(
                    np.asarray(cached_n2, dtype=np.complex128).reshape(-1),
                    dtype=np.complex128,
                )
            except Exception:
                normalized_cached_n2 = None
            if normalized_cached_n2 is None:
                pass
            elif normalized_cached_n2.size == wavelength_arr.size:
                cached_wavelength_snapshot = profile_cache.get(
                    "_n2_sample_array_wavelength_snapshot"
                )
                if cached_wavelength_snapshot is None:
                    return normalized_cached_n2
                try:
                    normalized_cached_wavelength_snapshot = _n2_wavelength_snapshot_from_angstrom(
                        cached_wavelength_snapshot
                    )
                except Exception:
                    normalized_cached_wavelength_snapshot = None
                if normalized_cached_wavelength_snapshot is not None and (
                    normalized_cached_wavelength_snapshot.size == wavelength_arr.size
                    and np.array_equal(
                        normalized_cached_wavelength_snapshot,
                        wavelength_arr,
                        equal_nan=True,
                    )
                ):
                    return normalized_cached_n2
                if normalized_cached_wavelength_snapshot is None:
                    # Keep a valid cached array when metadata is malformed; only replace it
                    # if we can successfully rebuild from the authoritative source.
                    cached_probe_n2_fallback = normalized_cached_n2

        source_meta = _normalize_n2_source_meta(profile_cache.get("_n2_sample_array_source"))
        if source_meta is None or source_meta[0] != "cif_path":
            return cached_probe_n2_fallback

        try:
            return np.ascontiguousarray(
                np.asarray(
                    resolve_index_of_refraction_array(
                        wavelength_arr * 1.0e-10,
                        cif_path=str(source_meta[1]),
                    ),
                    dtype=np.complex128,
                ).reshape(-1),
                dtype=np.complex128,
            )
        except Exception:
            return cached_probe_n2_fallback

    profile_cache = getattr(simulation_runtime_state, "profile_cache", {})
    if not isinstance(profile_cache, Mapping):
        profile_cache = {}

    def _run_single(
        beam_x: np.ndarray,
        beam_y: np.ndarray,
        theta_arr: np.ndarray,
        phi_arr: np.ndarray,
        wavelength_arr: np.ndarray,
        *,
        sample_weights_arr: np.ndarray | None = None,
        n2_sample_array_override: np.ndarray | None = None,
    ) -> tuple[float, float] | None:
        image_buf = np.zeros(
            (int(config.image_size), int(config.image_size)),
            dtype=np.float64,
        )
        miller_arr = np.array([[float(h), float(k), float(l)]], dtype=np.float64)
        intens_arr = np.array([1.0], dtype=np.float64)
        try:
            _image, hit_tables, *_ = process_peaks_parallel(
                miller_arr,
                intens_arr,
                int(config.image_size),
                float(config.lattice_a),
                float(config.lattice_c),
                float(config.wavelength),
                image_buf,
                float(config.distance_cor_to_detector),
                float(config.gamma_deg),
                float(config.Gamma_deg),
                float(config.chi_deg),
                float(config.psi_deg),
                float(config.psi_z_deg),
                float(config.zs),
                float(config.zb),
                n2,
                beam_x,
                beam_y,
                theta_arr,
                phi_arr,
                1e-6,
                1e-6,
                0.0,
                wavelength_arr,
                float(config.debye_x),
                float(config.debye_y),
                [
                    float(config.detector_center[0]),
                    float(config.detector_center[1]),
                ],
                float(config.theta_initial_deg),
                float(config.cor_angle_deg),
                np.asarray(config.unit_x, dtype=np.float64),
                np.asarray(config.n_detector, dtype=np.float64),
                save_flag=0,
                record_status=False,
                optics_mode=int(config.optics_mode),
                solve_q_steps=int(config.solve_q_steps),
                solve_q_rel_tol=float(config.solve_q_rel_tol),
                solve_q_mode=int(config.solve_q_mode),
                sample_weights=sample_weights_arr,
                n2_sample_array_override=n2_sample_array_override,
                **get_process_peaks_runtime_kwargs(),
            )
        except Exception:
            return None

        if hit_tables is None or len(hit_tables) == 0:
            return None
        return brightest_hit_native_from_table(hit_tables[0])

    strict_center = _run_single(
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        np.array([float(config.wavelength)], dtype=np.float64),
        n2_sample_array_override=_resolve_probe_n2_sample_array_override(
            profile_cache,
            np.array([float(config.wavelength)], dtype=np.float64),
        ),
    )
    if strict_center is not None:
        return strict_center

    beam_x = np.asarray(
        profile_cache.get("beam_x_array", []),
        dtype=np.float64,
    ).ravel()
    beam_y = np.asarray(
        profile_cache.get("beam_y_array", []),
        dtype=np.float64,
    ).ravel()
    theta_arr = np.asarray(
        profile_cache.get("theta_array", []),
        dtype=np.float64,
    ).ravel()
    phi_arr = np.asarray(
        profile_cache.get("phi_array", []),
        dtype=np.float64,
    ).ravel()
    wavelength_arr = np.asarray(
        profile_cache.get("wavelength_array", []),
        dtype=np.float64,
    ).ravel()
    sample_weights_arr = profile_cache.get("sample_weights")
    if sample_weights_arr is not None:
        sample_weights_arr = np.asarray(sample_weights_arr, dtype=np.float64).ravel()

    n_samp = beam_x.size
    if (
        n_samp == 0
        or beam_y.size != n_samp
        or theta_arr.size != n_samp
        or phi_arr.size != n_samp
        or wavelength_arr.size != n_samp
    ):
        return None
    if sample_weights_arr is not None and sample_weights_arr.size != n_samp:
        sample_weights_arr = None

    return _run_single(
        beam_x,
        beam_y,
        theta_arr,
        phi_arr,
        wavelength_arr,
        sample_weights_arr=sample_weights_arr,
        n2_sample_array_override=_resolve_probe_n2_sample_array_override(
            profile_cache,
            wavelength_arr,
        ),
    )


def _clear_peak_overlay_lists(simulation_runtime_state) -> None:
    simulation_runtime_state.peak_positions.clear()
    simulation_runtime_state.peak_positions_filtered = False
    simulation_runtime_state.peak_millers.clear()
    simulation_runtime_state.peak_intensities.clear()
    simulation_runtime_state.peak_records.clear()


def _peak_overlay_live_lists_match_cache(
    simulation_runtime_state,
    *,
    sig: object,
    cache_sig_key: str = "sig",
) -> bool:
    """Return whether live overlay lists already match one cached signature."""

    cache = getattr(simulation_runtime_state, "peak_overlay_cache", None)
    if not isinstance(cache, Mapping) or cache.get(cache_sig_key) != sig:
        return False

    cached_positions = cache.get("positions")
    cached_millers = cache.get("millers")
    cached_intensities = cache.get("intensities")
    cached_records = cache.get("records")
    if not all(
        isinstance(values, (list, tuple))
        for values in (
            cached_positions,
            cached_millers,
            cached_intensities,
            cached_records,
        )
    ):
        return False

    return (
        len(simulation_runtime_state.peak_positions) == len(cached_positions)
        and len(simulation_runtime_state.peak_millers) == len(cached_millers)
        and len(simulation_runtime_state.peak_intensities) == len(cached_intensities)
        and len(simulation_runtime_state.peak_records) == len(cached_records)
    )


def _peak_overlay_restored_view_signature(
    simulation_runtime_state,
    *,
    show_caked: bool,
) -> tuple[object, ...]:
    """Return one lightweight view signature for restored GUI-state overlays."""

    return (
        bool(show_caked),
        id(getattr(simulation_runtime_state, "last_caked_transform_bundle", None)),
        id(getattr(simulation_runtime_state, "last_caked_radial_values", None)),
        id(getattr(simulation_runtime_state, "last_caked_azimuth_values", None)),
        getattr(simulation_runtime_state, "last_analysis_signature", None),
    )


def _peak_overlay_has_restored_gui_state_cache(simulation_runtime_state) -> bool:
    """Return whether the overlay cache came from an imported GUI-state snapshot."""

    cache = getattr(simulation_runtime_state, "peak_overlay_cache", None)
    return isinstance(cache, Mapping) and bool(cache.get("restored_from_gui_state"))


def _peak_overlay_has_intersection_cache(simulation_runtime_state) -> bool:
    """Return whether any detector/caked intersection cache tables are available."""

    for attr_name in (
        "stored_primary_intersection_cache",
        "stored_secondary_intersection_cache",
        "stored_intersection_cache",
        "last_caked_intersection_cache",
    ):
        cache_tables = getattr(simulation_runtime_state, attr_name, None)
        if isinstance(cache_tables, (list, tuple)) and len(cache_tables) > 0:
            return True
    return False


def _peak_overlay_cache_signature(cache_tables: object) -> tuple[int, int]:
    """Return one lightweight cache identity tuple for overlay invalidation."""

    if not isinstance(cache_tables, (list, tuple)):
        return (0, 0)
    return (id(cache_tables), len(cache_tables))


def _peak_overlay_source_label(value: object, *, default: str = "primary") -> str:
    """Normalize one serialized peak-source label."""

    label = str(value or default).strip().lower()
    return "secondary" if label == "secondary" else "primary"


def _peak_overlay_source_defaults(
    lattice_entries: object,
    *,
    fallback_a: float,
    fallback_c: float,
    default_label: str,
) -> tuple[float, float, str]:
    """Resolve one source's lattice defaults from saved peak-table metadata."""

    av_used = float(fallback_a)
    cv_used = float(fallback_c)
    source_label = _peak_overlay_source_label(default_label, default=default_label)
    if not isinstance(lattice_entries, (list, tuple)) or len(lattice_entries) <= 0:
        return av_used, cv_used, source_label

    entry = lattice_entries[0]
    if isinstance(entry, (list, tuple)) and len(entry) >= 1:
        try:
            av_used = float(entry[0])
        except Exception:
            av_used = float(fallback_a)
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        try:
            cv_used = float(entry[1])
        except Exception:
            cv_used = float(fallback_c)
    if isinstance(entry, (list, tuple)) and len(entry) >= 3:
        source_label = _peak_overlay_source_label(entry[2], default=default_label)
    return av_used, cv_used, source_label


def _peak_overlay_table_defaults(
    lattice_entries: object,
    *,
    table_index: int,
    fallback_a: float,
    fallback_c: float,
    default_label: str,
) -> tuple[float, float, str]:
    """Resolve one table's lattice defaults from aligned per-table metadata."""

    if not isinstance(lattice_entries, (list, tuple)):
        return _peak_overlay_source_defaults(
            None,
            fallback_a=fallback_a,
            fallback_c=fallback_c,
            default_label=default_label,
        )
    if int(table_index) < 0 or int(table_index) >= len(lattice_entries):
        return _peak_overlay_source_defaults(
            None,
            fallback_a=fallback_a,
            fallback_c=fallback_c,
            default_label=default_label,
        )
    return _peak_overlay_source_defaults(
        [lattice_entries[int(table_index)]],
        fallback_a=fallback_a,
        fallback_c=fallback_c,
        default_label=default_label,
    )


def _peak_overlay_live_caked_cache_is_current(simulation_runtime_state) -> bool:
    """Return whether cached caked rows match live bundle and detector cache."""

    live_bundle = getattr(simulation_runtime_state, "last_caked_transform_bundle", None)
    cache_bundle = getattr(
        simulation_runtime_state,
        "last_caked_intersection_cache_transform_bundle",
        None,
    )
    if live_bundle is None or cache_bundle is not live_bundle:
        return False
    cached_source_sig = getattr(
        simulation_runtime_state,
        "last_caked_intersection_cache_source_signature",
        None,
    )
    current_source_sig = _peak_overlay_cache_signature(
        getattr(simulation_runtime_state, "stored_intersection_cache", None)
    )
    return cached_source_sig == current_source_sig


def _peak_overlay_intersection_entries(
    simulation_runtime_state,
    *,
    fallback_a: float,
    fallback_c: float,
    show_caked: bool = False,
) -> list[tuple[np.ndarray, float, float, str, str]]:
    """Return intersection-cache tables with resolved source metadata."""

    entries: list[tuple[np.ndarray, float, float, str, str]] = []
    live_caked_cache_current = _peak_overlay_live_caked_cache_is_current(simulation_runtime_state)

    if bool(show_caked):
        if live_caked_cache_current:
            combined_lattice_entries = getattr(
                simulation_runtime_state,
                "stored_peak_table_lattice",
                None,
            )
            caked_tables = getattr(
                simulation_runtime_state,
                "last_caked_intersection_cache",
                None,
            )
            if isinstance(caked_tables, (list, tuple)):
                for table_index, table in enumerate(caked_tables):
                    try:
                        table_arr = np.asarray(table, dtype=float)
                    except Exception:
                        continue
                    if table_arr.ndim != 2 or table_arr.shape[0] <= 0:
                        continue
                    av_used, cv_used, source_label = _peak_overlay_table_defaults(
                        combined_lattice_entries,
                        table_index=int(table_index),
                        fallback_a=fallback_a,
                        fallback_c=fallback_c,
                        default_label="primary",
                    )
                    entries.append(
                        (
                            table_arr,
                            float(av_used),
                            float(cv_used),
                            str(source_label),
                            "last_caked_intersection_cache",
                        )
                    )
        return entries

    primary_defaults = _peak_overlay_source_defaults(
        getattr(simulation_runtime_state, "stored_primary_peak_table_lattice", None),
        fallback_a=fallback_a,
        fallback_c=fallback_c,
        default_label="primary",
    )
    secondary_defaults = _peak_overlay_source_defaults(
        getattr(simulation_runtime_state, "stored_secondary_peak_table_lattice", None),
        fallback_a=fallback_a,
        fallback_c=fallback_c,
        default_label="secondary",
    )
    cache_sources = (
        (
            getattr(simulation_runtime_state, "stored_primary_intersection_cache", None),
            primary_defaults,
            "stored_primary_intersection_cache",
        ),
        (
            getattr(simulation_runtime_state, "stored_secondary_intersection_cache", None),
            secondary_defaults,
            "stored_secondary_intersection_cache",
        ),
    )
    for cache_tables, defaults, cache_attr_name in cache_sources:
        if not isinstance(cache_tables, (list, tuple)):
            continue
        av_used, cv_used, source_label = defaults
        for table in cache_tables:
            try:
                table_arr = np.asarray(table, dtype=float)
            except Exception:
                continue
            if table_arr.ndim != 2 or table_arr.shape[0] <= 0:
                continue
            entries.append(
                (
                    table_arr,
                    float(av_used),
                    float(cv_used),
                    str(source_label),
                    str(cache_attr_name),
                )
            )
    if entries:
        return entries

    combined_lattice_entries = getattr(
        simulation_runtime_state,
        "stored_peak_table_lattice",
        None,
    )
    cache_attr_names = ["stored_intersection_cache"]
    if live_caked_cache_current and not bool(show_caked):
        cache_attr_names.append("last_caked_intersection_cache")
    for attr_name in cache_attr_names:
        cache_tables = getattr(simulation_runtime_state, attr_name, None)
        if not isinstance(cache_tables, (list, tuple)):
            continue
        for table_index, table in enumerate(cache_tables):
            try:
                table_arr = np.asarray(table, dtype=float)
            except Exception:
                continue
            if table_arr.ndim != 2 or table_arr.shape[0] <= 0:
                continue
            av_used, cv_used, source_label = _peak_overlay_table_defaults(
                combined_lattice_entries,
                table_index=int(table_index),
                fallback_a=fallback_a,
                fallback_c=fallback_c,
                default_label="primary",
            )
            entries.append(
                (
                    table_arr,
                    float(av_used),
                    float(cv_used),
                    str(source_label),
                    str(attr_name),
                )
            )
        if entries:
            break
    return entries


def _peak_overlay_cache_row_caked_coords(row: np.ndarray) -> tuple[float, float] | None:
    """Return cached ``(2theta, phi)`` display coordinates for one caked cache row."""

    cached_two_theta, cached_phi = extract_cached_caked_angles(row)
    if np.isfinite(cached_two_theta) and np.isfinite(cached_phi):
        return float(cached_two_theta), float(cached_phi)
    return None


def ensure_runtime_peak_overlay_data(
    simulation_runtime_state,
    *,
    primary_a: object,
    primary_c: object,
    native_sim_to_display_coords: Callable[..., tuple[float, float]],
    reflection_q_group_metadata: Callable[..., tuple[object, object, object]],
    caked_view_enabled_factory: object = False,
    native_detector_coords_to_caked_display_coords: Callable[
        [float, float], tuple[float, float] | None
    ]
    | None = None,
    native_detector_coords_to_detector_display_coords: Callable[
        [float, float], tuple[float, float] | None
    ]
    | None = None,
    max_hits_per_reflection: object = 0,
    min_separation_px: object = 0.0,
    force: bool = False,
) -> bool:
    """Ensure the live simulated-peak overlay cache is populated for one frame."""

    if (
        simulation_runtime_state.stored_sim_image is None
        or not _peak_overlay_has_intersection_cache(simulation_runtime_state)
    ):
        show_caked = _runtime_bool(caked_view_enabled_factory, False)
        restored_view_sig = _peak_overlay_restored_view_signature(
            simulation_runtime_state,
            show_caked=bool(show_caked),
        )
        if _peak_overlay_has_restored_gui_state_cache(
            simulation_runtime_state
        ) and _peak_overlay_live_lists_match_cache(
            simulation_runtime_state,
            sig=restored_view_sig,
            cache_sig_key="restored_view_sig",
        ):
            return True
        if _peak_overlay_has_restored_gui_state_cache(
            simulation_runtime_state
        ) and _restore_peak_overlay_lists_from_cached_records(
            simulation_runtime_state,
            show_caked=bool(show_caked),
            image_shape=(0, 0),
            native_sim_to_display_coords=native_sim_to_display_coords,
            native_detector_coords_to_caked_display_coords=(
                native_detector_coords_to_caked_display_coords
            ),
            native_detector_coords_to_detector_display_coords=(
                native_detector_coords_to_detector_display_coords
            ),
            view_sig=restored_view_sig,
        ):
            return True
        _clear_peak_overlay_lists(simulation_runtime_state)
        return False

    updated_image = simulation_runtime_state.stored_sim_image

    fallback_a = _runtime_float(primary_a, float("nan"))
    fallback_c = _runtime_float(primary_c, float("nan"))

    max_hits_raw = _runtime_int(max_hits_per_reflection, 0)
    min_separation_value = _runtime_float(min_separation_px, 0.0)
    max_hits_limit = max_hits_raw if max_hits_raw > 0 else None
    min_sep_sq = float(min_separation_value) ** 2
    show_caked = _runtime_bool(caked_view_enabled_factory, False)
    peak_sig = (
        simulation_runtime_state.last_simulation_signature,
        tuple(updated_image.shape),
        int(max_hits_raw),
        float(min_separation_value),
        bool(show_caked),
        _peak_overlay_cache_signature(
            getattr(simulation_runtime_state, "stored_primary_intersection_cache", None)
        ),
        _peak_overlay_cache_signature(
            getattr(simulation_runtime_state, "stored_secondary_intersection_cache", None)
        ),
        _peak_overlay_cache_signature(
            getattr(simulation_runtime_state, "stored_intersection_cache", None)
        ),
        _peak_overlay_cache_signature(
            getattr(simulation_runtime_state, "last_caked_intersection_cache", None)
        ),
        id(getattr(simulation_runtime_state, "last_caked_radial_values", None)),
        id(getattr(simulation_runtime_state, "last_caked_azimuth_values", None)),
        id(getattr(simulation_runtime_state, "last_caked_transform_bundle", None)),
        id(
            getattr(
                simulation_runtime_state,
                "last_caked_intersection_cache_transform_bundle",
                None,
            )
        ),
        getattr(
            simulation_runtime_state,
            "last_caked_intersection_cache_source_signature",
            None,
        ),
        getattr(simulation_runtime_state, "last_analysis_signature", None),
    )
    retain_cache = _retain_peak_overlay_cache()
    peak_cached = (
        not force
        and retain_cache
        and simulation_runtime_state.peak_overlay_cache.get("sig") == peak_sig
    )
    if peak_cached and _peak_overlay_live_lists_match_cache(
        simulation_runtime_state,
        sig=peak_sig,
    ):
        return True

    _clear_peak_overlay_lists(simulation_runtime_state)
    source_reflection_indices_local = (
        list(getattr(simulation_runtime_state, "stored_source_reflection_indices_local", ()) or ())
        if isinstance(
            getattr(simulation_runtime_state, "stored_source_reflection_indices_local", None),
            (list, tuple),
        )
        else []
    )
    source_reflection_indices_by_label = {
        "primary": list(
            getattr(
                simulation_runtime_state,
                "stored_primary_source_reflection_indices",
                (),
            )
            or ()
        ),
        "secondary": list(
            getattr(
                simulation_runtime_state,
                "stored_secondary_source_reflection_indices",
                (),
            )
            or ()
        ),
    }

    if peak_cached:
        simulation_runtime_state.peak_positions.extend(
            list(simulation_runtime_state.peak_overlay_cache.get("positions", ()))
        )
        simulation_runtime_state.peak_positions_filtered = False
        simulation_runtime_state.peak_millers.extend(
            list(simulation_runtime_state.peak_overlay_cache.get("millers", ()))
        )
        simulation_runtime_state.peak_intensities.extend(
            list(simulation_runtime_state.peak_overlay_cache.get("intensities", ()))
        )
        simulation_runtime_state.peak_records.extend(
            dict(rec) for rec in simulation_runtime_state.peak_overlay_cache.get("records", ())
        )
        return True

    image_shape = tuple(int(v) for v in updated_image.shape)
    intersection_entries = _peak_overlay_intersection_entries(
        simulation_runtime_state,
        fallback_a=fallback_a,
        fallback_c=fallback_c,
        show_caked=bool(show_caked),
    )
    if intersection_entries:
        for cache_table_idx, (
            tbl_arr,
            av_used,
            cv_used,
            source_label,
            cache_attr_name,
        ) in enumerate(intersection_entries):
            if tbl_arr.ndim != 2 or tbl_arr.shape[0] == 0 or tbl_arr.shape[1] < 9:
                continue

            chosen_rows: list[
                tuple[int, np.ndarray, float, float, float, float, tuple[float, float] | None]
            ] = []
            row_order: Sequence[int]
            if max_hits_limit is None and min_sep_sq <= 0.0:
                row_order = list(range(int(tbl_arr.shape[0])))
            else:
                row_order = list(np.argsort(np.asarray(tbl_arr[:, 4], dtype=float))[::-1])

            for row_idx in row_order:
                row = np.asarray(tbl_arr[int(row_idx)], dtype=float).reshape(-1)
                try:
                    qr_hint = float(row[0])
                    qz_hint = float(row[1])
                    cx = float(row[2])
                    cy = float(row[3])
                    intensity = float(row[4])
                    phi_val = float(row[5])
                    hkl_raw = (float(row[6]), float(row[7]), float(row[8]))
                except Exception:
                    continue
                if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(intensity)):
                    continue

                caked_coords = None
                if bool(show_caked):
                    caked_coords = _peak_overlay_cache_row_caked_coords(
                        np.asarray(row, dtype=float).reshape(-1),
                    )
                    if caked_coords is None:
                        continue
                    disp_cx = float(caked_coords[0])
                    disp_cy = float(caked_coords[1])
                elif callable(native_detector_coords_to_detector_display_coords):
                    detector_display = native_detector_coords_to_detector_display_coords(
                        float(cx),
                        float(cy),
                    )
                    if (
                        isinstance(detector_display, tuple)
                        and len(detector_display) >= 2
                        and np.isfinite(float(detector_display[0]))
                        and np.isfinite(float(detector_display[1]))
                    ):
                        disp_cx = float(detector_display[0])
                        disp_cy = float(detector_display[1])
                    else:
                        disp_cx, disp_cy = native_sim_to_display_coords(
                            float(cx),
                            float(cy),
                            image_shape,
                        )
                else:
                    disp_cx, disp_cy = native_sim_to_display_coords(
                        float(cx),
                        float(cy),
                        image_shape,
                    )
                too_close = False
                if min_sep_sq > 0.0:
                    for _, _, _, _, prev_col, prev_row, _ in chosen_rows:
                        d2 = (float(disp_cx) - float(prev_col)) ** 2 + (
                            float(disp_cy) - float(prev_row)
                        ) ** 2
                        if d2 < min_sep_sq:
                            too_close = True
                            break
                if too_close:
                    continue
                chosen_rows.append(
                    (
                        int(row_idx),
                        row,
                        float(cx),
                        float(cy),
                        float(disp_cx),
                        float(disp_cy),
                        caked_coords,
                    )
                )
                if max_hits_limit is not None and len(chosen_rows) >= max_hits_limit:
                    break

            for row_idx, row, cx, cy, disp_cx, disp_cy, caked_coords in chosen_rows:
                try:
                    qr_hint = float(row[0])
                    qz_hint = float(row[1])
                    intensity = float(row[4])
                    phi_val = float(row[5])
                    hkl_raw = (float(row[6]), float(row[7]), float(row[8]))
                except Exception:
                    continue
                hkl = tuple(int(np.rint(val)) for val in hkl_raw)
                m_val = float(
                    hkl_raw[0] * hkl_raw[0] + hkl_raw[0] * hkl_raw[1] + hkl_raw[1] * hkl_raw[1]
                )
                qr_val = float(qr_hint)
                if not np.isfinite(qr_val):
                    qr_val = (
                        (2.0 * np.pi / float(av_used)) * np.sqrt((4.0 / 3.0) * m_val)
                        if float(av_used) > 0.0 and np.isfinite(float(av_used)) and m_val >= 0.0
                        else float("nan")
                    )
                q_group_key, _, qz_meta = reflection_q_group_metadata(
                    hkl_raw,
                    source_label=source_label,
                    a_value=av_used,
                    c_value=cv_used,
                    qr_value=qr_val,
                    allow_nominal_hkl_indices=True,
                )
                qz_val = float(qz_hint) if np.isfinite(qz_hint) else float(qz_meta)
                source_table_index, source_row_index, best_sample_index = (
                    extract_cache_row_provenance(row)
                )
                if source_table_index is None:
                    source_table_index = int(cache_table_idx)
                if source_row_index is None:
                    source_row_index = int(row_idx)
                source_branch_index = source_branch_index_from_phi_deg(phi_val)
                source_reflection_index = None
                source_reflection_indices = source_reflection_indices_by_label.get(
                    str(source_label),
                    source_reflection_indices_local,
                )
                if source_table_index is not None and 0 <= int(source_table_index) < len(
                    source_reflection_indices
                ):
                    try:
                        source_reflection_index = int(
                            source_reflection_indices[int(source_table_index)]
                        )
                    except Exception:
                        source_reflection_index = None

                simulation_runtime_state.peak_positions.append((float(disp_cx), float(disp_cy)))
                simulation_runtime_state.peak_intensities.append(float(intensity))
                simulation_runtime_state.peak_millers.append(hkl)

                record = {
                    "display_col": float(disp_cx),
                    "display_row": float(disp_cy),
                    "native_col": float(cx),
                    "native_row": float(cy),
                    "hkl": hkl,
                    "hkl_raw": hkl_raw,
                    "intensity": float(intensity),
                    "qr": float(qr_val),
                    "qz": float(qz_val),
                    "q_group_key": q_group_key,
                    "phi": float(phi_val),
                    "two_theta_deg": (
                        float(caked_coords[0]) if caked_coords is not None else float("nan")
                    ),
                    "phi_deg": (
                        float(caked_coords[1]) if caked_coords is not None else float("nan")
                    ),
                    "caked_x": (
                        float(caked_coords[0]) if caked_coords is not None else float("nan")
                    ),
                    "caked_y": (
                        float(caked_coords[1]) if caked_coords is not None else float("nan")
                    ),
                    "source_label": str(source_label),
                    "av": float(av_used),
                    "cv": float(cv_used),
                    "q_group_nominal_hkl": True,
                }
                if source_table_index is not None:
                    record["source_table_index"] = int(source_table_index)
                if source_row_index is not None:
                    record["source_row_index"] = int(source_row_index)
                if best_sample_index is not None:
                    record["best_sample_index"] = int(best_sample_index)
                if source_branch_index in {0, 1}:
                    record["source_branch_index"] = int(source_branch_index)
                    record["source_peak_index"] = int(source_branch_index)
                record = gui_manual_geometry.geometry_manual_canonicalize_live_source_entry(
                    record,
                    allow_legacy_peak_fallback=False,
                    preserve_existing_trusted_identity=False,
                    trusted_reflection_index=source_reflection_index,
                )
                if record is None:
                    continue
                record = gui_mosaic_top.annotate_selection_metadata(
                    record,
                    target_key=q_group_key,
                    profile_cache=_runtime_profile_cache(simulation_runtime_state),
                )
                simulation_runtime_state.peak_records.append(record)

        if retain_cache:
            simulation_runtime_state.peak_overlay_cache.update(
                {
                    "sig": peak_sig,
                    "positions": list(simulation_runtime_state.peak_positions),
                    "millers": list(simulation_runtime_state.peak_millers),
                    "intensities": list(simulation_runtime_state.peak_intensities),
                    "records": [dict(rec) for rec in simulation_runtime_state.peak_records],
                    "click_spatial_index": _build_peak_click_spatial_index(
                        simulation_runtime_state.peak_positions
                    ),
                    "peak_positions_filtered": False,
                    "restored_from_gui_state": False,
                }
            )
        else:
            simulation_runtime_state.peak_overlay_cache = _empty_peak_overlay_cache()
        return True
    _clear_peak_overlay_lists(simulation_runtime_state)
    return False


def make_runtime_peak_overlay_data_callback(
    *,
    simulation_runtime_state,
    primary_a_factory: object,
    primary_c_factory: object,
    native_sim_to_display_coords: Callable[..., tuple[float, float]],
    reflection_q_group_metadata: Callable[..., tuple[object, object, object]],
    caked_view_enabled_factory: object = False,
    native_detector_coords_to_caked_display_coords: Callable[
        [float, float], tuple[float, float] | None
    ]
    | None = None,
    native_detector_coords_to_detector_display_coords: Callable[
        [float, float], tuple[float, float] | None
    ]
    | None = None,
    max_hits_per_reflection: object = 0,
    min_separation_px: object = 0.0,
) -> Callable[..., bool]:
    """Return the live simulated-peak overlay cache callback for runtime use."""

    def _ensure(*, force: bool = False) -> bool:
        return ensure_runtime_peak_overlay_data(
            simulation_runtime_state,
            primary_a=primary_a_factory,
            primary_c=primary_c_factory,
            native_sim_to_display_coords=native_sim_to_display_coords,
            reflection_q_group_metadata=reflection_q_group_metadata,
            caked_view_enabled_factory=caked_view_enabled_factory,
            native_detector_coords_to_caked_display_coords=(
                native_detector_coords_to_caked_display_coords
            ),
            native_detector_coords_to_detector_display_coords=(
                native_detector_coords_to_detector_display_coords
            ),
            max_hits_per_reflection=max_hits_per_reflection,
            min_separation_px=min_separation_px,
            force=force,
        )

    return _ensure


def _copy_selected_peak_record(
    simulation_runtime_state,
    idx: int,
) -> dict[str, object] | None:
    if idx >= len(simulation_runtime_state.peak_records):
        return None
    raw_record = simulation_runtime_state.peak_records[idx]
    if not isinstance(raw_record, Mapping):
        return None
    return dict(raw_record)


def _finite_record_pair(
    record: Mapping[str, object] | None,
    x_key: str,
    y_key: str,
) -> tuple[float, float] | None:
    """Return one finite coordinate pair from a cached peak record."""

    if not isinstance(record, Mapping):
        return None
    try:
        col = float(record.get(x_key, np.nan))
        row = float(record.get(y_key, np.nan))
    except Exception:
        return None
    if not (np.isfinite(col) and np.isfinite(row)):
        return None
    return float(col), float(row)


def _record_caked_pairs(record: Mapping[str, object] | None) -> list[tuple[float, float]]:
    """Return finite caked-coordinate pairs carried by one peak record."""

    if not isinstance(record, Mapping):
        return []
    pairs: list[tuple[float, float]] = []
    for x_key, y_key in (
        ("caked_x", "caked_y"),
        ("raw_caked_x", "raw_caked_y"),
        ("two_theta_deg", "phi_deg"),
        ("refined_sim_caked_x", "refined_sim_caked_y"),
        ("background_two_theta_deg", "background_phi_deg"),
        ("simulated_two_theta_deg", "simulated_phi_deg"),
    ):
        point = _finite_record_pair(record, x_key, y_key)
        if point is not None:
            pairs.append(point)
    return pairs


def _point_matches_any(
    point: tuple[float, float] | None,
    candidates: Sequence[tuple[float, float]],
    *,
    tol: float = 1.0e-9,
) -> bool:
    if point is None:
        return False
    return any(
        abs(float(point[0]) - float(candidate[0])) <= float(tol)
        and abs(float(point[1]) - float(candidate[1])) <= float(tol)
        for candidate in candidates
    )


def _record_detector_display_pair(
    record: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    """Return detector-display coordinates without falling back to caked axes."""

    if not isinstance(record, Mapping):
        return None
    for x_key, y_key in (
        ("refined_detector_display_col", "refined_detector_display_row"),
        ("detector_display_col", "detector_display_row"),
    ):
        point = _finite_record_pair(record, x_key, y_key)
        if point is not None:
            return point

    caked_pairs = _record_caked_pairs(record)
    for x_key, y_key in (
        ("display_col", "display_row"),
        ("sim_col_raw", "sim_row_raw"),
        ("sim_col", "sim_row"),
        ("x", "y"),
        ("simulated_x", "simulated_y"),
    ):
        point = _finite_record_pair(record, x_key, y_key)
        if point is None or _point_matches_any(point, caked_pairs):
            continue
        return point
    return None


def _record_display_pair(
    record: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    """Return the active-view display point carried by a cached peak record."""

    return _record_detector_display_pair(record)


def _record_matches_hkl_target(
    record: Mapping[str, object] | None,
    target: Sequence[int],
) -> bool:
    hkl = _record_hkl_triplet(record)
    if hkl is None or len(target) < 3:
        return False
    return tuple(int(v) for v in hkl) == tuple(int(target[idx]) for idx in range(3))


def _record_has_caked_refinement(record: Mapping[str, object] | None) -> bool:
    if not isinstance(record, Mapping):
        return False
    if str(record.get("refined_by", "")) == "caked_peak_center":
        return True
    return bool(
        _finite_record_pair(record, "refined_sim_caked_x", "refined_sim_caked_y") is not None
        and _record_detector_display_pair(record) is not None
    )


def _simulation_signature_matches(left: object, right: object) -> bool:
    if left is right:
        return True
    try:
        result = left == right
    except Exception:
        return False
    if isinstance(result, np.ndarray):
        try:
            return bool(np.all(result))
        except Exception:
            return False
    try:
        return bool(result)
    except Exception:
        return False


def _record_refinement_signature_matches(
    record: Mapping[str, object] | None,
    simulation_runtime_state,
) -> bool:
    current_signature = getattr(
        simulation_runtime_state,
        "last_simulation_signature",
        None,
    )
    if not isinstance(record, Mapping):
        return False
    if current_signature is None:
        return (
            _REFINED_SIMULATION_SIGNATURE_KEY not in record
            or record.get(_REFINED_SIMULATION_SIGNATURE_KEY) is None
        )
    if _REFINED_SIMULATION_SIGNATURE_KEY not in record:
        return False
    return _simulation_signature_matches(
        record.get(_REFINED_SIMULATION_SIGNATURE_KEY),
        current_signature,
    )


def _record_native_pair(
    record: Mapping[str, object] | None,
) -> tuple[float, float] | None:
    """Return the native detector point carried by a cached peak record."""

    return _finite_record_pair(record, "native_col", "native_row") or _finite_record_pair(
        record,
        "sim_native_x",
        "sim_native_y",
    )


def _normalize_record_hkl(value: object) -> tuple[int, int, int] | None:
    """Return one rounded HKL triplet from a record/list value."""

    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 3:
        return None
    try:
        hkl = tuple(int(np.rint(float(value[idx]))) for idx in range(3))
    except Exception:
        return None
    return hkl  # type: ignore[return-value]


def _record_hkl_triplet(
    record: Mapping[str, object] | None,
) -> tuple[int, int, int] | None:
    """Return the HKL encoded in a cached peak record."""

    if not isinstance(record, Mapping):
        return None
    return _normalize_record_hkl(record.get("hkl")) or _normalize_record_hkl(record.get("hkl_raw"))


def _peak_record_for_index(
    simulation_runtime_state,
    idx: int,
    record_override: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    """Return the authoritative peak record for a selected index."""

    if isinstance(record_override, Mapping):
        return dict(record_override)
    return _copy_selected_peak_record(simulation_runtime_state, int(idx))


def _peak_hkl_for_index(
    simulation_runtime_state,
    idx: int,
    record_override: Mapping[str, object] | None = None,
) -> tuple[int, int, int] | None:
    """Return the HKL for one peak, preferring the cached record over parallel lists."""

    record = _peak_record_for_index(simulation_runtime_state, int(idx), record_override)
    hkl = _record_hkl_triplet(record)
    if hkl is not None:
        return hkl
    try:
        return _normalize_record_hkl(simulation_runtime_state.peak_millers[int(idx)])
    except Exception:
        return None


def _peak_intensity_for_index(
    simulation_runtime_state,
    idx: int,
    record_override: Mapping[str, object] | None = None,
) -> float:
    """Return one peak intensity, preferring the cached record value."""

    idx_value = int(idx)
    record = _peak_record_for_index(simulation_runtime_state, idx_value, record_override)
    if isinstance(record, Mapping):
        for key in ("intensity", "weight"):
            try:
                val = float(record.get(key, np.nan))
            except Exception:
                val = float("nan")
            if np.isfinite(val):
                return float(val)
    if isinstance(record_override, Mapping) and idx_value < 0:
        return float("nan")
    try:
        val = float(simulation_runtime_state.peak_intensities[idx_value])
    except Exception:
        return float("nan")
    return float(val) if np.isfinite(val) else float("nan")


def _peak_display_pair_for_index(
    simulation_runtime_state,
    idx: int,
    record_override: Mapping[str, object] | None = None,
    *,
    use_caked_display: bool = False,
) -> tuple[float, float] | None:
    """Return the active-view display point for one peak index."""

    record = _peak_record_for_index(simulation_runtime_state, int(idx), record_override)
    point = (
        _simulation_point_active_view_pair(record, use_caked_display=True)
        if bool(use_caked_display)
        else _record_display_pair(record)
    )
    if point is not None:
        return point
    try:
        px, py = simulation_runtime_state.peak_positions[int(idx)]
        px = float(px)
        py = float(py)
    except Exception:
        return None
    if not (np.isfinite(px) and np.isfinite(py)):
        return None
    return float(px), float(py)


def _simulation_point_active_view_pair(
    record: Mapping[str, object] | None,
    *,
    use_caked_display: bool,
) -> tuple[float, float] | None:
    """Return the same active-view point used by manual Qr picking."""

    if not isinstance(record, Mapping):
        return None
    if not bool(use_caked_display):
        return _record_detector_display_pair(record)
    try:
        point = gui_manual_geometry._geometry_manual_entry_active_view_point(
            record,
            use_caked_display=bool(use_caked_display),
        )
    except Exception:
        point = None
    if point is not None:
        try:
            col = float(point[0])
            row = float(point[1])
        except Exception:
            return None
        if np.isfinite(col) and np.isfinite(row):
            return float(col), float(row)
    try:
        if gui_manual_geometry._geometry_manual_entry_has_stale_caked_fields(record):
            return None
    except Exception:
        pass
    return (
        _finite_record_pair(record, "caked_x", "caked_y")
        or _finite_record_pair(record, "raw_caked_x", "raw_caked_y")
        or _finite_record_pair(record, "two_theta_deg", "phi_deg")
    )


def _runtime_profile_cache(simulation_runtime_state) -> Mapping[str, object] | None:
    profile_cache = getattr(simulation_runtime_state, "profile_cache", None)
    return profile_cache if isinstance(profile_cache, Mapping) else None


def _peak_candidate_for_index(
    simulation_runtime_state,
    idx: int,
    *,
    target_key: object = None,
) -> dict[str, object] | None:
    record = _peak_record_for_index(simulation_runtime_state, int(idx))
    if not isinstance(record, dict):
        record = {}
    hkl = _peak_hkl_for_index(simulation_runtime_state, int(idx), record)
    if hkl is not None:
        record.setdefault("hkl", hkl)
    display = _peak_display_pair_for_index(simulation_runtime_state, int(idx), record)
    if display is not None:
        record.setdefault("display_col", float(display[0]))
        record.setdefault("display_row", float(display[1]))
    try:
        intensity = float(simulation_runtime_state.peak_intensities[int(idx)])
    except Exception:
        intensity = float("nan")
    if np.isfinite(intensity):
        record.setdefault("intensity", float(intensity))
    record["_peak_index"] = int(idx)
    record.pop("mosaic_top_rank_key", None)
    return gui_mosaic_top.annotate_selection_metadata(
        record,
        target_key=target_key,
        profile_cache=_runtime_profile_cache(simulation_runtime_state),
    )


def _selected_peak_index_from_representative(
    simulation_runtime_state,
    representative: Mapping[str, object],
) -> int:
    try:
        idx = int(representative.get("_peak_index"))
    except Exception:
        idx = -1
    if idx >= 0:
        return int(idx)
    resolved = _peak_record_index_for_simulation_point(
        simulation_runtime_state,
        representative,
    )
    return -1 if resolved is None else int(resolved)


def _select_mosaic_top_peak_record(
    simulation_runtime_state,
    indices: Sequence[int],
    *,
    branch_id: str | None = None,
    target_key: object = None,
) -> tuple[int, dict[str, object] | None]:
    candidates: list[dict[str, object]] = []
    for idx in indices:
        candidate = _peak_candidate_for_index(
            simulation_runtime_state,
            int(idx),
            target_key=target_key,
        )
        if candidate is not None:
            candidates.append(candidate)
    representative = gui_mosaic_top.select_mosaic_top_representative(
        candidates,
        branch_id=branch_id,
        target_key=target_key,
        profile_cache=_runtime_profile_cache(simulation_runtime_state),
    )
    if representative is None:
        return -1, None
    return _selected_peak_index_from_representative(
        simulation_runtime_state,
        representative,
    ), representative


def _refine_selected_hkl_caked_record(
    simulation_runtime_state,
    record: Mapping[str, object] | None,
    *,
    caked_angles_to_detector_display_coords: Callable[[float, float], tuple[float, float] | None]
    | None = None,
    detector_display_to_native_detector_coords: Callable[[float, float], tuple[float, float] | None]
    | None = None,
) -> tuple[dict[str, object] | None, tuple[float, float] | None]:
    if not isinstance(record, Mapping):
        return None, None
    selected = dict(record)
    seed = _simulation_point_active_view_pair(selected, use_caked_display=True)
    if seed is None:
        return selected, None
    raw_two_theta = float(seed[0])
    raw_phi = float(seed[1])
    selected.setdefault("raw_caked_x", raw_two_theta)
    selected.setdefault("raw_caked_y", raw_phi)
    selected["pre_refine_two_theta_deg"] = raw_two_theta
    selected["pre_refine_phi_deg"] = raw_phi
    image = getattr(simulation_runtime_state, "last_caked_image_unscaled", None)
    radial = getattr(simulation_runtime_state, "last_caked_radial_values", None)
    azimuth = getattr(simulation_runtime_state, "last_caked_azimuth_values", None)
    refined_two_theta, refined_phi = gui_manual_geometry.refine_caked_peak_center(
        image,
        radial,
        azimuth,
        raw_two_theta,
        raw_phi,
    )
    selected["caked_x"] = float(refined_two_theta)
    selected["caked_y"] = float(refined_phi)
    selected["two_theta_deg"] = float(refined_two_theta)
    selected["phi_deg"] = float(refined_phi)
    selected["selected_display_col"] = float(refined_two_theta)
    selected["selected_display_row"] = float(refined_phi)
    selected["refined_by"] = "caked_peak_center"
    selected[_REFINED_SIMULATION_SIGNATURE_KEY] = getattr(
        simulation_runtime_state,
        "last_simulation_signature",
        None,
    )

    if callable(caked_angles_to_detector_display_coords):
        try:
            detector_display = caked_angles_to_detector_display_coords(
                float(refined_two_theta),
                float(refined_phi),
            )
        except Exception:
            detector_display = None
        if (
            isinstance(detector_display, tuple)
            and len(detector_display) >= 2
            and np.isfinite(float(detector_display[0]))
            and np.isfinite(float(detector_display[1]))
        ):
            detector_col = float(detector_display[0])
            detector_row = float(detector_display[1])
            selected["display_col"] = detector_col
            selected["display_row"] = detector_row
            selected["sim_col"] = detector_col
            selected["sim_row"] = detector_row
            selected["sim_col_raw"] = detector_col
            selected["sim_row_raw"] = detector_row
            selected["refined_detector_display_col"] = detector_col
            selected["refined_detector_display_row"] = detector_row
            selected["refined_detector_projection_source"] = "caked_angles"

            if callable(detector_display_to_native_detector_coords):
                try:
                    native_point = detector_display_to_native_detector_coords(
                        detector_col,
                        detector_row,
                    )
                except Exception:
                    native_point = None
                if (
                    isinstance(native_point, tuple)
                    and len(native_point) >= 2
                    and np.isfinite(float(native_point[0]))
                    and np.isfinite(float(native_point[1]))
                ):
                    native_col = float(native_point[0])
                    native_row = float(native_point[1])
                    selected["native_col"] = native_col
                    selected["native_row"] = native_row
                    selected["sim_native_x"] = native_col
                    selected["sim_native_y"] = native_row
                    selected["detector_x"] = native_col
                    selected["detector_y"] = native_row
                    selected["background_detector_x"] = native_col
                    selected["background_detector_y"] = native_row
    return selected, (float(refined_two_theta), float(refined_phi))


def _build_simulation_point_click_index(
    candidate_records: Sequence[Mapping[str, object]] | None,
    *,
    use_caked_display: bool,
    cell_size_px: float = _PEAK_CLICK_INDEX_CELL_SIZE_PX,
) -> dict[str, object]:
    """Bucket Qr/manual simulation-point rows for fast HKL click lookup."""

    cell_size = max(1.0, float(cell_size_px))
    cells: dict[tuple[int, int], list[int]] = {}
    points: list[tuple[float, float] | None] = []
    count = 0
    for idx, raw_record in enumerate(candidate_records or ()):
        count += 1
        point = _simulation_point_active_view_pair(
            raw_record,
            use_caked_display=bool(use_caked_display),
        )
        if point is None:
            points.append(None)
            continue
        col = float(point[0])
        row = float(point[1])
        if not (np.isfinite(col) and np.isfinite(row)):
            points.append(None)
            continue
        points.append((float(col), float(row)))
        cell_key = (
            int(np.floor(col / cell_size)),
            int(np.floor(row / cell_size)),
        )
        cells.setdefault(cell_key, []).append(int(idx))
    return {
        "cell_size_px": float(cell_size),
        "candidate_count": int(count),
        "use_caked_display": bool(use_caked_display),
        "cells": cells,
        "points": points,
    }


def build_hkl_pick_simulation_point_payload(
    candidate_records: Sequence[Mapping[str, object]] | None,
    *,
    cell_size_px: float = _PEAK_CLICK_INDEX_CELL_SIZE_PX,
) -> dict[str, object]:
    """Build reusable HKL click payload from the exact Qr/manual candidate rows."""

    candidates = tuple(
        dict(entry) for entry in (candidate_records or ()) if isinstance(entry, Mapping)
    )
    return {
        "candidates": candidates,
        "candidate_count": int(len(candidates)),
        "detector_index": _build_simulation_point_click_index(
            candidates,
            use_caked_display=False,
            cell_size_px=float(cell_size_px),
        ),
        "caked_index": _build_simulation_point_click_index(
            candidates,
            use_caked_display=True,
            cell_size_px=float(cell_size_px),
        ),
    }


def _simulation_point_payload_from_factory(
    factory: Callable[[], object] | None,
) -> dict[str, object] | None:
    """Return current Qr-picker simulation point payload, if a provider is wired."""

    if not callable(factory):
        return None
    try:
        raw_payload = factory()
    except TypeError:
        try:
            raw_payload = factory(None, prefer_cache=True)  # type: ignore[misc]
        except Exception:
            return None
    except Exception:
        return None

    if isinstance(raw_payload, Mapping):
        candidates = raw_payload.get("candidates")
        if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
            if isinstance(raw_payload.get("detector_index"), Mapping) or isinstance(
                raw_payload.get("caked_index"),
                Mapping,
            ):
                return dict(raw_payload)
            payload = build_hkl_pick_simulation_point_payload(candidates)
            for key, value in raw_payload.items():
                if key not in payload:
                    payload[key] = value
            return payload
        grouped = raw_payload.get("grouped_candidates")
        if isinstance(grouped, Mapping):
            rows: list[Mapping[str, object]] = []
            for entries in grouped.values():
                if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
                    rows.extend(entry for entry in entries if isinstance(entry, Mapping))
            if rows:
                payload = build_hkl_pick_simulation_point_payload(rows)
                for key, value in raw_payload.items():
                    if key not in payload:
                        payload[key] = value
                return payload
        return None

    if not isinstance(raw_payload, Sequence) or isinstance(raw_payload, (str, bytes)):
        return None
    return build_hkl_pick_simulation_point_payload(
        [entry for entry in raw_payload if isinstance(entry, Mapping)]
    )


def _simulation_point_candidates_from_factory(
    factory: Callable[[], object] | None,
) -> list[dict[str, object]]:
    """Return current Qr-picker simulation point rows, if a provider is wired."""

    payload = _simulation_point_payload_from_factory(factory)
    candidates = payload.get("candidates") if isinstance(payload, Mapping) else None
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return []
    return [dict(entry) for entry in candidates if isinstance(entry, Mapping)]


def _runtime_hkl_pick_simulation_point_payload(bindings: SelectedPeakRuntimeBindings) -> object:
    """Return the exact Qr/manual picker payload for HKL image picking."""

    payload = _simulation_point_payload_from_factory(
        getattr(bindings, "hkl_pick_simulation_points_factory", None)
    )
    if not isinstance(payload, Mapping):
        return None
    candidates = payload.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return None
    if not candidates:
        return None
    return payload


def _simulation_point_candidates_from_payload(
    payload_or_candidates: object,
) -> Sequence[Mapping[str, object]]:
    """Normalize indexed payloads or legacy candidate lists to one candidate sequence."""

    if isinstance(payload_or_candidates, Mapping):
        candidates = payload_or_candidates.get("candidates")
        if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
            return candidates
        return ()
    if isinstance(payload_or_candidates, Sequence) and not isinstance(
        payload_or_candidates,
        (str, bytes),
    ):
        return tuple(entry for entry in payload_or_candidates if isinstance(entry, Mapping))
    return ()


def _simulation_point_click_index_from_payload(
    payload_or_candidates: object,
    *,
    use_caked_display: bool,
) -> Mapping[str, object] | None:
    """Return the prebuilt click index for the active view, if present."""

    if not isinstance(payload_or_candidates, Mapping):
        return None
    key = "caked_index" if bool(use_caked_display) else "detector_index"
    value = payload_or_candidates.get(key)
    return value if isinstance(value, Mapping) else None


def _simulation_point_candidate_indices_for_click(
    spatial_index: Mapping[str, object] | None,
    click_col: float,
    click_row: float,
    *,
    max_axis_distance_px: float | None,
) -> list[int]:
    """Return indexed Qr/manual candidates whose cells intersect the click window."""

    if not isinstance(spatial_index, Mapping):
        return []
    cells = spatial_index.get("cells")
    if not isinstance(cells, Mapping):
        return []
    try:
        cell_size = max(
            1.0,
            float(
                spatial_index.get(
                    "cell_size_px",
                    _PEAK_CLICK_INDEX_CELL_SIZE_PX,
                )
            ),
        )
    except Exception:
        cell_size = float(_PEAK_CLICK_INDEX_CELL_SIZE_PX)
    half_window = (
        max(0.0, float(max_axis_distance_px))
        if max_axis_distance_px is not None
        else float(cell_size)
    )
    min_cell_x = int(np.floor((float(click_col) - half_window) / cell_size))
    max_cell_x = int(np.floor((float(click_col) + half_window) / cell_size))
    min_cell_y = int(np.floor((float(click_row) - half_window) / cell_size))
    max_cell_y = int(np.floor((float(click_row) + half_window) / cell_size))

    candidate_indices: list[int] = []
    for cell_x in range(min_cell_x, max_cell_x + 1):
        for cell_y in range(min_cell_y, max_cell_y + 1):
            cell_entries = cells.get((cell_x, cell_y))
            if not isinstance(cell_entries, Sequence) or isinstance(
                cell_entries,
                (str, bytes),
            ):
                continue
            candidate_indices.extend(int(idx) for idx in cell_entries)
    return candidate_indices


def _nearest_indexed_simulation_point(
    candidates: Sequence[Mapping[str, object]],
    candidate_indices: Sequence[int] | None,
    *,
    points: Sequence[tuple[float, float] | None] | None,
    click_col: float,
    click_row: float,
    use_caked_display: bool,
    apply_window: bool,
    max_axis_distance_px: float | None,
) -> tuple[int | None, float]:
    """Return nearest candidate index using precomputed points when possible."""

    best_idx: int | None = None
    best_d2 = float("inf")
    half_window = (
        max(0.0, float(max_axis_distance_px)) if max_axis_distance_px is not None else None
    )
    for raw_idx in candidate_indices or ():
        try:
            idx = int(raw_idx)
        except Exception:
            continue
        if idx < 0 or idx >= len(candidates):
            continue
        point = None
        if points is not None and idx < len(points):
            point = points[idx]
        if point is None:
            point = _simulation_point_active_view_pair(
                candidates[idx],
                use_caked_display=bool(use_caked_display),
            )
        if point is None:
            continue
        px = float(point[0])
        py = float(point[1])
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        dx = px - float(click_col)
        dy = py - float(click_row)
        if (
            apply_window
            and half_window is not None
            and (abs(dx) > half_window or abs(dy) > half_window)
        ):
            continue
        d2 = dx * dx + dy * dy
        if d2 < best_d2 or (d2 == best_d2 and (best_idx is None or int(idx) < int(best_idx))):
            best_idx = int(idx)
            best_d2 = float(d2)
    return best_idx, best_d2


def _peak_record_index_for_simulation_point(
    simulation_runtime_state,
    candidate: Mapping[str, object],
    *,
    allow_slow_scan: bool = True,
) -> int | None:
    """Resolve a Qr-picker simulation point back to a live peak-record index."""

    peak_records = getattr(simulation_runtime_state, "peak_records", ())
    if not isinstance(peak_records, Sequence) or isinstance(peak_records, (str, bytes)):
        return None

    candidate_dict = dict(candidate)
    if not bool(allow_slow_scan):
        return None
    try:
        candidate_key = gui_manual_geometry.geometry_manual_candidate_source_key(candidate_dict)
    except Exception:
        candidate_key = None
    if (
        isinstance(candidate_key, tuple)
        and candidate_key
        and candidate_key[0] in {"source", "source_branch"}
    ):
        for idx, raw_record in enumerate(peak_records):
            if not isinstance(raw_record, Mapping):
                continue
            record_dict = dict(raw_record)
            if str(candidate_key[0]) == "source":
                try:
                    record_key = gui_manual_geometry.geometry_manual_candidate_source_key(
                        record_dict
                    )
                except Exception:
                    record_key = None
                if record_key != candidate_key:
                    continue
                return int(idx)
            else:
                try:
                    same_identity = (
                        gui_manual_geometry.geometry_manual_source_entries_share_identity(
                            candidate_dict,
                            record_dict,
                        )
                    )
                except Exception:
                    same_identity = False
                if not same_identity:
                    continue
                return int(idx)

    candidate_hkl = _record_hkl_triplet(candidate_dict)
    candidate_native = _record_native_pair(candidate_dict)
    if candidate_hkl is None and candidate_native is None:
        return None
    best_idx: int | None = None
    best_d2 = float("inf")
    for idx, raw_record in enumerate(peak_records):
        if not isinstance(raw_record, Mapping):
            continue
        record_hkl = _record_hkl_triplet(raw_record)
        if candidate_hkl is not None and record_hkl != candidate_hkl:
            continue
        record_native = _record_native_pair(raw_record)
        if candidate_native is not None:
            if record_native is None:
                continue
            d2 = (float(record_native[0]) - float(candidate_native[0])) ** 2 + (
                float(record_native[1]) - float(candidate_native[1])
            ) ** 2
        else:
            d2 = 0.0
        if d2 < best_d2:
            best_idx = int(idx)
            best_d2 = float(d2)
    return best_idx


def _merged_simulation_point_peak_record(
    simulation_runtime_state,
    candidate: Mapping[str, object],
    idx: int | None,
) -> dict[str, object]:
    """Merge the base live peak record with the Qr-picker simulation point row."""

    merged: dict[str, object] = {}
    if idx is not None and idx >= 0:
        base = _copy_selected_peak_record(simulation_runtime_state, int(idx))
        if isinstance(base, dict):
            merged.update(base)
    merged.update(dict(candidate))
    return merged


def _nearest_simulation_point_for_click(
    simulation_runtime_state,
    click_col: float,
    click_row: float,
    *,
    candidate_records: object,
    max_axis_distance_px: float | None,
    use_caked_display: bool,
) -> tuple[int, dict[str, object] | None, float, bool]:
    """Return the nearest Qr-picker simulation point for an HKL click."""

    candidates = _simulation_point_candidates_from_payload(candidate_records)
    if not candidates:
        return -1, None, float("nan"), False

    spatial_index = _simulation_point_click_index_from_payload(
        candidate_records,
        use_caked_display=bool(use_caked_display),
    )
    points: Sequence[tuple[float, float] | None] | None = None
    if isinstance(spatial_index, Mapping):
        raw_points = spatial_index.get("points")
        if isinstance(raw_points, Sequence) and not isinstance(raw_points, (str, bytes)):
            points = raw_points  # type: ignore[assignment]

    candidate_indices = _simulation_point_candidate_indices_for_click(
        spatial_index,
        float(click_col),
        float(click_row),
        max_axis_distance_px=max_axis_distance_px,
    )
    if candidate_indices:
        best_candidate_idx, best_d2 = _nearest_indexed_simulation_point(
            candidates,
            candidate_indices,
            points=points,
            click_col=float(click_col),
            click_row=float(click_row),
            use_caked_display=bool(use_caked_display),
            apply_window=True,
            max_axis_distance_px=max_axis_distance_px,
        )
    else:
        best_candidate_idx, best_d2 = None, float("inf")

    if best_candidate_idx is None:
        full_range = range(len(candidates))
        best_candidate_idx, best_d2 = _nearest_indexed_simulation_point(
            candidates,
            full_range,
            points=points,
            click_col=float(click_col),
            click_row=float(click_row),
            use_caked_display=bool(use_caked_display),
            apply_window=False,
            max_axis_distance_px=max_axis_distance_px,
        )
    if best_candidate_idx is None or not np.isfinite(best_d2):
        return -1, None, float("nan"), False

    nearest_candidate = dict(candidates[int(best_candidate_idx)])
    active_point = None
    if points is not None and int(best_candidate_idx) < len(points):
        active_point = points[int(best_candidate_idx)]
    if active_point is None:
        active_point = _simulation_point_active_view_pair(
            nearest_candidate,
            use_caked_display=bool(use_caked_display),
        )
    if active_point is None:
        return -1, None, float("nan"), False
    dx = float(active_point[0]) - float(click_col)
    dy = float(active_point[1]) - float(click_row)
    dist = float(np.sqrt(best_d2))
    within_window = True
    if max_axis_distance_px is not None:
        half_window = max(0.0, float(max_axis_distance_px))
        within_window = bool(abs(dx) <= half_window and abs(dy) <= half_window)

    idx = _peak_record_index_for_simulation_point(
        simulation_runtime_state,
        nearest_candidate,
        allow_slow_scan=_record_hkl_triplet(nearest_candidate) is None,
    )
    merged_record = _merged_simulation_point_peak_record(
        simulation_runtime_state,
        nearest_candidate,
        idx,
    )
    return (-1 if idx is None else int(idx)), merged_record, float(dist), bool(within_window)


def _apply_selected_peak_record_coordinates(
    record: dict[str, object] | None,
    *,
    clicked_display: tuple[float, float] | None = None,
    clicked_native: tuple[float, float] | None = None,
    selected_display: tuple[float, float] | None = None,
    selected_native: tuple[float, float] | None = None,
) -> dict[str, object] | None:
    if record is None:
        return None
    if clicked_display is not None:
        record["clicked_display_col"] = float(clicked_display[0])
        record["clicked_display_row"] = float(clicked_display[1])
    if clicked_native is not None:
        record["clicked_native_col"] = float(clicked_native[0])
        record["clicked_native_row"] = float(clicked_native[1])
    if selected_display is not None:
        record["selected_display_col"] = float(selected_display[0])
        record["selected_display_row"] = float(selected_display[1])
    if selected_native is not None:
        record["selected_native_col"] = float(selected_native[0])
        record["selected_native_row"] = float(selected_native[1])
    elif clicked_native is not None:
        record["selected_native_col"] = float(clicked_native[0])
        record["selected_native_row"] = float(clicked_native[1])
    else:
        native_point = _record_native_pair(record)
        if native_point is not None:
            record["selected_native_col"] = float(native_point[0])
            record["selected_native_row"] = float(native_point[1])
    return record


def _build_peak_click_spatial_index(
    peak_positions: Sequence[tuple[float, float]] | None,
    *,
    cell_size_px: float = _PEAK_CLICK_INDEX_CELL_SIZE_PX,
) -> dict[str, object]:
    """Bucket peak positions into a coarse spatial grid for fast click lookup."""

    cell_size = max(1.0, float(cell_size_px))
    cells: dict[tuple[int, int], list[int]] = {}
    position_count = 0
    for idx, coords in enumerate(peak_positions or ()):
        position_count += 1
        try:
            px = float(coords[0])
            py = float(coords[1])
        except Exception:
            continue
        if not (np.isfinite(px) and np.isfinite(py)) or px < 0.0:
            continue
        cell_key = (
            int(np.floor(px / cell_size)),
            int(np.floor(py / cell_size)),
        )
        cells.setdefault(cell_key, []).append(int(idx))
    return {
        "cell_size_px": float(cell_size),
        "position_count": int(position_count),
        "cells": cells,
    }


def _resolve_peak_record_display_coords(
    raw_record: Mapping[str, object],
    *,
    show_caked: bool,
    image_shape: tuple[int, int],
    native_sim_to_display_coords: Callable[..., tuple[float, float]] | None,
    allow_frozen_display_fallback: bool = False,
    native_detector_coords_to_caked_display_coords: Callable[
        [float, float], tuple[float, float] | None
    ]
    | None = None,
    native_detector_coords_to_detector_display_coords: Callable[
        [float, float], tuple[float, float] | None
    ]
    | None = None,
) -> tuple[float, float] | None:
    """Return active-view display coords for one cached peak record."""

    def _point(x_key: str, y_key: str) -> tuple[float, float] | None:
        try:
            col = float(raw_record.get(x_key, np.nan))
            row = float(raw_record.get(y_key, np.nan))
        except Exception:
            return None
        if not (np.isfinite(col) and np.isfinite(row)):
            return None
        return float(col), float(row)

    def _projected_caked_point(
        projected: object,
    ) -> tuple[float, float] | None:
        try:
            col = float(projected[0])  # type: ignore[index]
            row = float(projected[1])  # type: ignore[index]
        except Exception:
            return None
        if not (np.isfinite(col) and np.isfinite(row)):
            return None
        return float(col), float(row)

    native_point = _point("native_col", "native_row") or _point(
        "sim_native_x",
        "sim_native_y",
    )
    raw_detector_display = _point("sim_col_raw", "sim_row_raw")
    explicit_display_point = _point("display_col", "display_row")
    caked_point = (
        _point("caked_x", "caked_y")
        or _point("raw_caked_x", "raw_caked_y")
        or _point("two_theta_deg", "phi_deg")
    )

    if bool(show_caked):
        if caked_point is not None:
            return caked_point
        if native_point is not None and callable(native_detector_coords_to_caked_display_coords):
            try:
                projected = native_detector_coords_to_caked_display_coords(
                    float(native_point[0]),
                    float(native_point[1]),
                )
            except Exception:
                projected = None
            projected_point = _projected_caked_point(projected)
            if projected_point is not None:
                return projected_point
        if bool(allow_frozen_display_fallback) and explicit_display_point is not None:
            return explicit_display_point
        return None

    if native_point is not None and callable(native_detector_coords_to_detector_display_coords):
        try:
            projected = native_detector_coords_to_detector_display_coords(
                float(native_point[0]),
                float(native_point[1]),
            )
        except Exception:
            projected = None
        if (
            isinstance(projected, tuple)
            and len(projected) >= 2
            and np.isfinite(float(projected[0]))
            and np.isfinite(float(projected[1]))
        ):
            return (float(projected[0]), float(projected[1]))

    if native_point is not None and callable(native_sim_to_display_coords):
        try:
            projected = native_sim_to_display_coords(
                float(native_point[0]),
                float(native_point[1]),
                image_shape,
            )
        except Exception:
            projected = None
        if (
            isinstance(projected, tuple)
            and len(projected) >= 2
            and np.isfinite(float(projected[0]))
            and np.isfinite(float(projected[1]))
        ):
            return (float(projected[0]), float(projected[1]))
    if raw_detector_display is not None:
        return raw_detector_display
    if (
        bool(allow_frozen_display_fallback)
        and explicit_display_point is not None
        and caked_point is None
    ):
        return explicit_display_point
    return None


def _restore_peak_overlay_lists_from_cached_records(
    simulation_runtime_state,
    *,
    show_caked: bool,
    image_shape: tuple[int, int],
    native_sim_to_display_coords: Callable[..., tuple[float, float]] | None,
    native_detector_coords_to_caked_display_coords: Callable[
        [float, float], tuple[float, float] | None
    ]
    | None = None,
    native_detector_coords_to_detector_display_coords: Callable[
        [float, float], tuple[float, float] | None
    ]
    | None = None,
    view_sig: object = None,
) -> bool:
    """Rebuild live overlay lists from imported cached peak records."""

    cache = getattr(simulation_runtime_state, "peak_overlay_cache", None)
    cache_restored_from_gui_state = (
        bool(cache.get("restored_from_gui_state", False)) if isinstance(cache, Mapping) else False
    )
    restored_view_sig = cache.get("restored_view_sig") if isinstance(cache, Mapping) else None
    frozen_display_fallback_allowed = bool(
        cache_restored_from_gui_state
        and isinstance(cache, Mapping)
        and restored_view_sig is not None
        and restored_view_sig == view_sig
    )
    cache_filtered = (
        bool(
            cache.get(
                "peak_positions_filtered",
                False,
            )
        )
        if isinstance(cache, Mapping)
        else False
    )
    raw_records = cache.get("records") if isinstance(cache, Mapping) else None
    if not isinstance(raw_records, (list, tuple)):
        raw_records = getattr(simulation_runtime_state, "peak_records", ())

    restored_records: list[dict[str, object]] = []
    restored_positions: list[tuple[float, float]] = []
    restored_millers: list[tuple[int, int, int]] = []
    restored_intensities: list[float] = []

    for raw_record in raw_records or ():
        if not isinstance(raw_record, Mapping):
            continue
        display_point = _resolve_peak_record_display_coords(
            raw_record,
            show_caked=bool(show_caked),
            image_shape=image_shape,
            native_sim_to_display_coords=native_sim_to_display_coords,
            allow_frozen_display_fallback=frozen_display_fallback_allowed,
            native_detector_coords_to_caked_display_coords=(
                native_detector_coords_to_caked_display_coords
            ),
            native_detector_coords_to_detector_display_coords=(
                native_detector_coords_to_detector_display_coords
            ),
        )
        if display_point is None:
            continue

        if not bool(show_caked) and not cache_restored_from_gui_state:
            detector_display_point = display_point
        else:
            detector_display_point = _resolve_peak_record_display_coords(
                raw_record,
                show_caked=False,
                image_shape=image_shape,
                native_sim_to_display_coords=native_sim_to_display_coords,
                native_detector_coords_to_caked_display_coords=(
                    native_detector_coords_to_caked_display_coords
                ),
                native_detector_coords_to_detector_display_coords=(
                    native_detector_coords_to_detector_display_coords
                ),
            )
            explicit_detector_point = gui_manual_geometry._geometry_manual_finite_point(
                raw_record,
                (
                    ("x", "y"),
                    ("sim_col_raw", "sim_row_raw"),
                    ("simulated_x", "simulated_y"),
                ),
            )
            native_detector_point = gui_manual_geometry._geometry_manual_entry_native_point(
                raw_record
            )
            if detector_display_point is None and (
                not cache_restored_from_gui_state
                or explicit_detector_point is not None
                or native_detector_point is not None
            ):
                legacy_detector_point = gui_manual_geometry._geometry_manual_finite_point(
                    raw_record,
                    (("sim_col", "sim_row"),),
                )
                if legacy_detector_point is not None:
                    active_view_point = (
                        gui_manual_geometry._geometry_manual_entry_matching_current_view_point(
                            raw_record
                        )
                        if bool(show_caked)
                        else gui_manual_geometry._geometry_manual_entry_explicit_current_view_display_point(
                            raw_record
                        )
                    )
                    if active_view_point is None or (
                        abs(float(legacy_detector_point[0]) - float(active_view_point[0])) > 1.0e-9
                        or abs(float(legacy_detector_point[1]) - float(active_view_point[1]))
                        > 1.0e-9
                    ):
                        detector_display_point = (
                            float(legacy_detector_point[0]),
                            float(legacy_detector_point[1]),
                        )

        hkl_value = raw_record.get("hkl")
        if not isinstance(hkl_value, (list, tuple)) or len(hkl_value) < 3:
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
            intensity = float(raw_record.get("intensity", raw_record.get("weight", 0.0)))
        except Exception:
            intensity = 0.0
        if not np.isfinite(intensity):
            intensity = 0.0

        record = dict(raw_record)
        if isinstance(record.get("hkl"), list):
            record["hkl"] = hkl_triplet
        if isinstance(record.get("hkl_raw"), list) and len(record["hkl_raw"]) >= 3:
            try:
                record["hkl_raw"] = (
                    float(record["hkl_raw"][0]),
                    float(record["hkl_raw"][1]),
                    float(record["hkl_raw"][2]),
                )
            except Exception:
                pass
        if isinstance(record.get("q_group_key"), list):
            record["q_group_key"] = tuple(record["q_group_key"])
        if detector_display_point is not None:
            record["sim_col"] = float(detector_display_point[0])
            record["sim_row"] = float(detector_display_point[1])
            record["sim_col_raw"] = float(detector_display_point[0])
            record["sim_row_raw"] = float(detector_display_point[1])
            record["display_col"] = float(detector_display_point[0])
            record["display_row"] = float(detector_display_point[1])
        else:
            record.pop("sim_col", None)
            record.pop("sim_row", None)
            record.pop("sim_col_raw", None)
            record.pop("sim_row_raw", None)
        if bool(show_caked):
            record["display_col"] = float(display_point[0])
            record["display_row"] = float(display_point[1])
            record["caked_x"] = float(display_point[0])
            record["caked_y"] = float(display_point[1])
            record.setdefault("raw_caked_x", float(display_point[0]))
            record.setdefault("raw_caked_y", float(display_point[1]))
            record.setdefault("two_theta_deg", float(display_point[0]))
            record.setdefault("phi_deg", float(display_point[1]))
        else:
            record["display_col"] = float(display_point[0])
            record["display_row"] = float(display_point[1])

        restored_records.append(record)
        restored_positions.append((float(display_point[0]), float(display_point[1])))
        restored_millers.append(hkl_triplet)
        restored_intensities.append(float(intensity))

    _clear_peak_overlay_lists(simulation_runtime_state)
    simulation_runtime_state.peak_positions.extend(restored_positions)
    simulation_runtime_state.peak_positions_filtered = cache_filtered
    simulation_runtime_state.peak_millers.extend(restored_millers)
    simulation_runtime_state.peak_intensities.extend(restored_intensities)
    simulation_runtime_state.peak_records.extend(restored_records)

    if isinstance(cache, dict):
        cache.update(
            {
                "positions": list(restored_positions),
                "millers": list(restored_millers),
                "intensities": list(restored_intensities),
                "records": [dict(record) for record in restored_records],
                "restored_view_sig": view_sig,
                "click_spatial_index": _build_peak_click_spatial_index(restored_positions),
                "peak_positions_filtered": cache_filtered,
                "restored_from_gui_state": cache_restored_from_gui_state,
            }
        )

    return bool(restored_records)


def _peak_click_spatial_index(
    simulation_runtime_state,
    *,
    cell_size_px: float = _PEAK_CLICK_INDEX_CELL_SIZE_PX,
) -> dict[str, object]:
    """Return the cached click spatial index, rebuilding it when needed."""

    cache = getattr(simulation_runtime_state, "peak_overlay_cache", None)
    position_count = len(getattr(simulation_runtime_state, "peak_positions", ()))
    if isinstance(cache, dict):
        payload = cache.get("click_spatial_index")
        if (
            isinstance(payload, Mapping)
            and int(payload.get("position_count", -1)) == int(position_count)
            and abs(float(payload.get("cell_size_px", 0.0)) - float(cell_size_px)) <= 1.0e-9
            and isinstance(payload.get("cells"), Mapping)
        ):
            return dict(payload)

    payload = _build_peak_click_spatial_index(
        getattr(simulation_runtime_state, "peak_positions", ()),
        cell_size_px=float(cell_size_px),
    )
    if isinstance(cache, dict):
        cache["click_spatial_index"] = payload
    return payload


def _peak_click_candidate_indices(
    simulation_runtime_state,
    click_col: float,
    click_row: float,
    *,
    max_axis_distance_px: float,
) -> list[int]:
    """Return candidate peak indices whose grid cells intersect the click window."""

    half_window = max(0.0, float(max_axis_distance_px))
    if half_window <= 0.0:
        return []

    payload = _peak_click_spatial_index(simulation_runtime_state)
    try:
        cell_size = max(1.0, float(payload.get("cell_size_px", _PEAK_CLICK_INDEX_CELL_SIZE_PX)))
    except Exception:
        cell_size = float(_PEAK_CLICK_INDEX_CELL_SIZE_PX)
    cells = payload.get("cells")
    if not isinstance(cells, Mapping):
        return []

    min_cell_x = int(np.floor((float(click_col) - half_window) / cell_size))
    max_cell_x = int(np.floor((float(click_col) + half_window) / cell_size))
    min_cell_y = int(np.floor((float(click_row) - half_window) / cell_size))
    max_cell_y = int(np.floor((float(click_row) + half_window) / cell_size))

    candidate_indices: list[int] = []
    for cell_x in range(min_cell_x, max_cell_x + 1):
        for cell_y in range(min_cell_y, max_cell_y + 1):
            cell_entries = cells.get((cell_x, cell_y))
            if isinstance(cell_entries, Sequence):
                candidate_indices.extend(int(idx) for idx in cell_entries)
    return candidate_indices


def select_peak_by_index(
    simulation_runtime_state,
    peak_selection_state,
    hkl_lookup_view_state,
    selected_peak_marker: object,
    idx: int,
    *,
    primary_a: float,
    sync_peak_selection_state: Any,
    set_status_text: Any,
    draw_idle: Any,
    prefix: str = "Selected peak",
    sync_hkl_vars: bool = True,
    clicked_display: tuple[float, float] | None = None,
    clicked_native: tuple[float, float] | None = None,
    selected_display: tuple[float, float] | None = None,
    selected_native: tuple[float, float] | None = None,
    record_override: Mapping[str, object] | None = None,
) -> bool:
    """Select one simulated peak by cached index and update GUI state."""

    selected_record_override = (
        dict(record_override) if isinstance(record_override, Mapping) else None
    )
    if idx < 0 or idx >= len(simulation_runtime_state.peak_positions):
        if selected_record_override is None:
            return False

    hkl = _peak_hkl_for_index(
        simulation_runtime_state,
        int(idx),
        record_override=selected_record_override,
    )
    if hkl is None:
        return False
    H, K, L = hkl

    intensity = _peak_intensity_for_index(
        simulation_runtime_state,
        int(idx),
        record_override=selected_record_override,
    )
    display_point = _peak_display_pair_for_index(
        simulation_runtime_state,
        int(idx),
        record_override=selected_record_override,
    )
    if selected_display is not None:
        disp_col, disp_row = (float(selected_display[0]), float(selected_display[1]))
    elif display_point is not None:
        disp_col, disp_row = (float(display_point[0]), float(display_point[1]))
    else:
        return False

    selected_peak_marker.set_data([disp_col], [disp_row])
    selected_peak_marker.set_visible(True)

    peak_selection_state.selected_hkl_target = (int(H), int(K), int(L))
    sync_peak_selection_state()

    selected_record = _peak_record_for_index(
        simulation_runtime_state,
        int(idx),
        record_override=selected_record_override,
    )
    selected_record = _apply_selected_peak_record_coordinates(
        selected_record,
        clicked_display=clicked_display,
        clicked_native=clicked_native,
        selected_display=(float(disp_col), float(disp_row)),
        selected_native=selected_native,
    )
    simulation_runtime_state.selected_peak_record = selected_record

    if sync_hkl_vars:
        gui_views.set_hkl_lookup_values(
            hkl_lookup_view_state,
            h_text=str(int(H)),
            k_text=str(int(K)),
            l_text=str(int(L)),
        )

    qr_val, deg_hkls = selected_peak_qr_and_degenerates(
        simulation_runtime_state,
        H,
        K,
        L,
        simulation_runtime_state.selected_peak_record,
        primary_a=primary_a,
    )
    if simulation_runtime_state.selected_peak_record is not None:
        simulation_runtime_state.selected_peak_record["qr"] = float(qr_val)
        simulation_runtime_state.selected_peak_record["degenerate_hkls"] = [
            (int(hv), int(kv), int(lv)) for hv, kv, lv in deg_hkls
        ]

    set_status_text(
        build_selected_peak_status_text(
            prefix=prefix,
            h=int(H),
            k=int(K),
            l=int(L),
            display_col=disp_col,
            display_row=disp_row,
            intensity=float(intensity),
            qr_val=float(qr_val),
            deg_hkls=deg_hkls,
        )
    )
    draw_idle()
    return True


def _m_index(hkl_triplet: tuple[int, int, int]) -> int:
    h0, k0, _l0 = hkl_triplet
    return int(h0 * h0 + h0 * k0 + k0 * k0)


def select_peak_by_hkl(
    simulation_runtime_state,
    peak_selection_state,
    hkl_lookup_view_state,
    selected_peak_marker: object,
    h: int,
    k: int,
    l: int,
    *,
    primary_a: float,
    ensure_peak_overlay_data: Any,
    schedule_update: Any,
    sync_peak_selection_state: Any,
    set_status_text: Any,
    draw_idle: Any,
    caked_view_enabled: bool = False,
    sync_hkl_vars: bool = True,
    silent_if_missing: bool = False,
    caked_angles_to_detector_display_coords: Callable[[float, float], tuple[float, float] | None]
    | None = None,
    detector_display_to_native_detector_coords: Callable[[float, float], tuple[float, float] | None]
    | None = None,
) -> bool:
    """Select the simulated peak matching one requested integer HKL."""

    ensure_peak_overlay_data(force=False)
    target = (int(h), int(k), int(l))

    if not simulation_runtime_state.peak_positions:
        if not silent_if_missing and simulation_runtime_state.unscaled_image is not None:
            schedule_update()
        if not silent_if_missing:
            set_status_text("Preparing simulated peak map... try again after update.")
        return False

    def _visible_peak_index(idx_value: int) -> bool:
        point = _peak_display_pair_for_index(
            simulation_runtime_state,
            int(idx_value),
            use_caked_display=bool(caked_view_enabled),
        )
        return bool(point is not None and float(point[0]) >= 0.0)

    matches = [
        idx
        for idx in range(len(simulation_runtime_state.peak_positions))
        if _visible_peak_index(idx) and _peak_hkl_for_index(simulation_runtime_state, idx) == target
    ]

    if not matches:
        m_target = _m_index(target)
        l_target = int(target[2])
        matches = []
        for idx in range(len(simulation_runtime_state.peak_positions)):
            if not _visible_peak_index(idx):
                continue
            hkl_triplet = _peak_hkl_for_index(simulation_runtime_state, idx)
            if hkl_triplet is None:
                continue
            if int(hkl_triplet[2]) != l_target:
                continue
            if _m_index(hkl_triplet) == m_target:
                matches.append(idx)

    if not matches:
        if not silent_if_missing:
            set_status_text(
                f"HKL ({target[0]} {target[1]} {target[2]}) not found in current simulation."
            )
        peak_selection_state.selected_hkl_target = target
        sync_peak_selection_state()
        simulation_runtime_state.selected_peak_record = None
        return False

    best_idx, selected_record = _select_mosaic_top_peak_record(
        simulation_runtime_state,
        matches,
        target_key=("hkl",) + target,
    )
    if selected_record is None:
        return False
    selected_display = _peak_display_pair_for_index(
        simulation_runtime_state,
        int(best_idx),
        record_override=selected_record,
        use_caked_display=bool(caked_view_enabled),
    )
    record_override = selected_record
    if bool(caked_view_enabled):
        record_override, refined_display = _refine_selected_hkl_caked_record(
            simulation_runtime_state,
            selected_record,
            caked_angles_to_detector_display_coords=(caked_angles_to_detector_display_coords),
            detector_display_to_native_detector_coords=(detector_display_to_native_detector_coords),
        )
        if refined_display is not None:
            selected_display = refined_display
    return select_peak_by_index(
        simulation_runtime_state,
        peak_selection_state,
        hkl_lookup_view_state,
        selected_peak_marker,
        best_idx,
        primary_a=primary_a,
        sync_peak_selection_state=sync_peak_selection_state,
        set_status_text=set_status_text,
        draw_idle=draw_idle,
        prefix="Selected peak",
        selected_display=selected_display,
        record_override=record_override,
        sync_hkl_vars=sync_hkl_vars,
    )


def selected_hkl_from_lookup_controls(
    hkl_lookup_view_state,
    *,
    tcl_error_types: tuple[type[BaseException], ...] = (),
) -> tuple[int, int, int] | None:
    """Read one integer HKL target from the lookup entry vars."""

    try:
        h = int(round(float(hkl_lookup_view_state.selected_h_var.get().strip())))
        k = int(round(float(hkl_lookup_view_state.selected_k_var.get().strip())))
        l = int(round(float(hkl_lookup_view_state.selected_l_var.get().strip())))
    except Exception as exc:
        if tcl_error_types and isinstance(exc, tcl_error_types):
            return None
        if isinstance(exc, (ValueError, AttributeError)):
            return None
        raise
    return (h, k, l)


def select_peak_from_hkl_controls(
    simulation_runtime_state,
    peak_selection_state,
    hkl_lookup_view_state,
    selected_peak_marker: object,
    *,
    primary_a: float,
    ensure_peak_overlay_data: Any,
    schedule_update: Any,
    sync_peak_selection_state: Any,
    set_status_text: Any,
    draw_idle: Any,
    caked_view_enabled: bool = False,
    tcl_error_types: tuple[type[BaseException], ...] = (),
    caked_angles_to_detector_display_coords: Callable[[float, float], tuple[float, float] | None]
    | None = None,
    detector_display_to_native_detector_coords: Callable[[float, float], tuple[float, float] | None]
    | None = None,
) -> bool:
    """Select one peak from the HKL lookup entry controls."""

    target = selected_hkl_from_lookup_controls(
        hkl_lookup_view_state,
        tcl_error_types=tcl_error_types,
    )
    if target is None:
        set_status_text("Enter numeric H, K, L values.")
        return False

    peak_selection_state.selected_hkl_target = target
    sync_peak_selection_state()
    return select_peak_by_hkl(
        simulation_runtime_state,
        peak_selection_state,
        hkl_lookup_view_state,
        selected_peak_marker,
        int(target[0]),
        int(target[1]),
        int(target[2]),
        primary_a=primary_a,
        ensure_peak_overlay_data=ensure_peak_overlay_data,
        schedule_update=schedule_update,
        sync_peak_selection_state=sync_peak_selection_state,
        set_status_text=set_status_text,
        draw_idle=draw_idle,
        caked_view_enabled=bool(caked_view_enabled),
        sync_hkl_vars=True,
        silent_if_missing=False,
        caked_angles_to_detector_display_coords=caked_angles_to_detector_display_coords,
        detector_display_to_native_detector_coords=detector_display_to_native_detector_coords,
    )


def clear_selected_peak(
    simulation_runtime_state,
    peak_selection_state,
    selected_peak_marker: object,
    *,
    sync_peak_selection_state: Any,
    set_status_text: Any,
    draw_idle: Any,
) -> None:
    """Clear the current selected-peak state and marker."""

    peak_selection_state.selected_hkl_target = None
    sync_peak_selection_state()
    simulation_runtime_state.selected_peak_record = None
    selected_peak_marker.set_visible(False)
    set_status_text("Peak selection cleared.")
    draw_idle()


def toggle_hkl_pick_mode(
    simulation_runtime_state,
    peak_selection_state,
    *,
    caked_view_enabled: bool,
    ensure_peak_overlay_data: Any,
    schedule_update: Any,
    set_pick_mode: Any,
    set_status_text: Any,
) -> None:
    """Arm or disarm HKL image-pick mode based on current GUI state."""

    if peak_selection_state.hkl_pick_armed:
        set_pick_mode(False, message="HKL image-pick canceled.")
        return

    if simulation_runtime_state.unscaled_image is None:
        set_status_text("Run a simulation first.")
        return

    if not ensure_peak_overlay_data(force=False) or not simulation_runtime_state.peak_positions:
        set_pick_mode(
            True,
            message=("Preparing simulated peak map for HKL picking... wait for the next update."),
        )
        schedule_update()
        return

    set_pick_mode(
        True,
        message=(
            "HKL image-pick armed: click near a Bragg peak in the 2D caked view."
            if bool(caked_view_enabled)
            else "HKL image-pick armed: click near a Bragg peak in raw camera view."
        ),
    )


def _nearest_peak_index_for_click(
    simulation_runtime_state,
    click_col: float,
    click_row: float,
    *,
    max_axis_distance_px: float | None = None,
) -> tuple[int, float, bool]:
    peak_positions = getattr(simulation_runtime_state, "peak_positions", ())
    peak_intensities = getattr(simulation_runtime_state, "peak_intensities", ())

    def _best_peak(
        candidate_indices: Sequence[int] | None,
        *,
        apply_window: bool,
    ) -> tuple[int, float]:
        best_i = -1
        best_d2 = float("inf")
        best_i_val = float("-inf")
        for raw_idx in candidate_indices or ():
            i = int(raw_idx)
            if i < 0 or i >= len(peak_positions):
                continue
            px, py = peak_positions[i]
            try:
                px = float(px)
                py = float(py)
            except Exception:
                continue
            if not (np.isfinite(px) and np.isfinite(py)) or px < 0.0:
                continue
            if (
                apply_window
                and max_axis_distance_px is not None
                and (
                    abs(px - float(click_col)) > float(max_axis_distance_px)
                    or abs(py - float(click_row)) > float(max_axis_distance_px)
                )
            ):
                continue
            d2 = (px - float(click_col)) ** 2 + (py - float(click_row)) ** 2
            val = _peak_intensity_for_index(simulation_runtime_state, int(i))
            score_val = float(val) if np.isfinite(val) else float("-inf")
            if d2 < best_d2 - 1e-9 or (abs(d2 - best_d2) <= 1e-9 and score_val > best_i_val):
                best_i = int(i)
                best_d2 = float(d2)
                best_i_val = float(score_val)
        return best_i, best_d2

    candidate_indices = _peak_click_candidate_indices(
        simulation_runtime_state,
        float(click_col),
        float(click_row),
        max_axis_distance_px=float(max_axis_distance_px)
        if max_axis_distance_px is not None
        else 0.0,
    )
    best_i, best_d2 = _best_peak(candidate_indices, apply_window=True)
    if best_i != -1:
        return best_i, best_d2, True

    best_i, best_d2 = _best_peak(range(len(peak_positions)), apply_window=False)
    return best_i, best_d2, False


def find_peak_record_for_canvas_click(
    simulation_runtime_state,
    click_col: float,
    click_row: float,
    *,
    ensure_peak_overlay_data: Any,
    max_axis_distance_px: float,
    simulation_point_candidates: object = None,
    use_caked_display: bool = False,
) -> tuple[int, dict[str, object] | None, float, bool]:
    """Return nearest visible simulation point from one click without mutating selection."""

    ensure_peak_overlay_data(force=False)
    candidate_result = _nearest_simulation_point_for_click(
        simulation_runtime_state,
        float(click_col),
        float(click_row),
        candidate_records=simulation_point_candidates,
        max_axis_distance_px=float(max_axis_distance_px),
        use_caked_display=bool(use_caked_display),
    )
    if candidate_result[1] is not None and bool(candidate_result[3]):
        return candidate_result

    fallback_result: tuple[int, dict[str, object] | None, float, bool] = (
        -1,
        None,
        float("nan"),
        False,
    )
    fallback_valid = False
    if (
        bool(getattr(simulation_runtime_state, "peak_positions_filtered", False))
        and simulation_runtime_state.peak_positions
    ):
        best_i, best_d2, within_window = _nearest_peak_index_for_click(
            simulation_runtime_state,
            float(click_col),
            float(click_row),
            max_axis_distance_px=float(max_axis_distance_px),
        )
        if best_i != -1 and np.isfinite(best_d2):
            peak_record = None
            if best_i < len(simulation_runtime_state.peak_records):
                raw_record = simulation_runtime_state.peak_records[best_i]
                if isinstance(raw_record, Mapping):
                    peak_record = dict(raw_record)

            fallback_result = (
                int(best_i),
                peak_record,
                float(np.sqrt(best_d2)),
                bool(within_window),
            )
            fallback_valid = True
            if bool(within_window):
                return fallback_result

    if candidate_result[1] is not None and np.isfinite(candidate_result[2]):
        if not fallback_valid or not np.isfinite(fallback_result[2]):
            return candidate_result
        if float(candidate_result[2]) <= float(fallback_result[2]):
            return candidate_result

    if fallback_valid:
        return fallback_result
    return -1, None, float("nan"), False


def _resolve_selected_peak_click_coordinates(
    simulation_runtime_state,
    idx: int,
    *,
    click_col: float,
    click_row: float,
    clicked_native_col: float,
    clicked_native_row: float,
    config: SelectedPeakCanvasPickConfig,
    native_sim_to_display_coords: Any,
    simulate_ideal_hkl_native_center: Any,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    selected_display = None
    selected_native = None

    if idx >= len(simulation_runtime_state.peak_records):
        return selected_display, selected_native

    peak_record = simulation_runtime_state.peak_records[idx]
    if not isinstance(peak_record, Mapping):
        return selected_display, selected_native

    image_shape = (
        tuple(int(v) for v in config.image_shape)
        if config.image_shape is not None
        else (int(config.image_size), int(config.image_size))
    )
    try:
        raw_hkl = peak_record.get("hkl_raw")
        if isinstance(raw_hkl, (list, tuple, np.ndarray)) and len(raw_hkl) >= 3:
            rec_h = float(raw_hkl[0])
            rec_k = float(raw_hkl[1])
            rec_l = float(raw_hkl[2])
        else:
            rec_h, rec_k, rec_l = tuple(
                float(v) for v in simulation_runtime_state.peak_millers[idx]
            )
        rec_av = float(peak_record.get("av", float(config.primary_a)))
        rec_cv = float(peak_record.get("cv", float(config.primary_c)))
        ideal_native = simulate_ideal_hkl_native_center(
            rec_h,
            rec_k,
            rec_l,
            rec_av,
            rec_cv,
        )
        if ideal_native is not None:
            ideal_display = native_sim_to_display_coords(
                ideal_native[0],
                ideal_native[1],
                image_shape,
            )
            base_display = simulation_runtime_state.peak_positions[idx]
            snap_delta = float(
                np.hypot(
                    float(ideal_display[0]) - float(base_display[0]),
                    float(ideal_display[1]) - float(base_display[1]),
                )
            )
            snap_limit = max(4.0, float(config.min_separation_px) * 2.0)
            if snap_delta <= snap_limit:
                selected_native = (
                    float(ideal_native[0]),
                    float(ideal_native[1]),
                )
                selected_display = (
                    float(ideal_display[0]),
                    float(ideal_display[1]),
                )
    except Exception:
        selected_display = None
        selected_native = None

    if selected_native is None:
        selected_native = (
            float(peak_record.get("native_col", clicked_native_col)),
            float(peak_record.get("native_row", clicked_native_row)),
        )

    return selected_display, selected_native


def select_peak_from_canvas_click(
    simulation_runtime_state,
    peak_selection_state,
    click_col: float,
    click_row: float,
    *,
    config: SelectedPeakCanvasPickConfig,
    ensure_peak_overlay_data: Any,
    schedule_update: Any,
    display_to_native_sim_coords: Any,
    native_sim_to_display_coords: Any,
    simulate_ideal_hkl_native_center: Any,
    select_peak_by_index: Any,
    set_pick_mode: Any,
    sync_peak_selection_state: Any,
    set_status_text: Any,
    caked_view_enabled: bool = False,
    detector_display_to_native_detector_coords: Any = None,
    caked_angles_to_detector_display_coords: Callable[[float, float], tuple[float, float] | None]
    | None = None,
    simulation_point_candidates: object = None,
) -> bool:
    """Select the nearest visible peak from one detector or caked-view click."""

    best_i, peak_record, best_dist, within_window = find_peak_record_for_canvas_click(
        simulation_runtime_state,
        float(click_col),
        float(click_row),
        ensure_peak_overlay_data=ensure_peak_overlay_data,
        max_axis_distance_px=max(0.0, float(config.max_distance_px)),
        simulation_point_candidates=simulation_point_candidates,
        use_caked_display=bool(caked_view_enabled),
    )
    if not simulation_runtime_state.peak_positions and peak_record is None:
        schedule_update()
        set_status_text("Preparing simulated peak map... click again after update.")
        return False

    if best_i == -1 and peak_record is None:
        set_status_text("No peaks on screen.")
        return False
    if not within_window:
        window_size_px = max(1.0, 2.0 * max(0.0, float(config.max_distance_px)))
        set_status_text(
            f"No simulated peak within the {window_size_px:.0f}x{window_size_px:.0f}px "
            f"search window (nearest is {best_dist:.1f}px away)."
        )
        return False

    target_hkl = _record_hkl_triplet(peak_record)
    if target_hkl is not None:
        branch_id, _branch_source = gui_mosaic_top.normalize_branch_id(
            peak_record,
            target_key=("hkl",) + target_hkl,
            profile_cache=_runtime_profile_cache(simulation_runtime_state),
        )
        payload_candidates = _simulation_point_candidates_from_payload(simulation_point_candidates)
        if payload_candidates:
            candidate_records: list[dict[str, object]] = []
            for raw_candidate in payload_candidates:
                if not isinstance(raw_candidate, Mapping):
                    continue
                if _record_hkl_triplet(raw_candidate) != target_hkl:
                    continue
                candidate = dict(raw_candidate)
                resolved_idx = _peak_record_index_for_simulation_point(
                    simulation_runtime_state,
                    candidate,
                )
                if resolved_idx is not None:
                    candidate["_peak_index"] = int(resolved_idx)
                candidate_records.append(
                    gui_mosaic_top.annotate_selection_metadata(
                        candidate,
                        target_key=("hkl",) + target_hkl,
                        profile_cache=_runtime_profile_cache(simulation_runtime_state),
                    )
                )
            selected_candidate = gui_mosaic_top.select_mosaic_top_representative(
                candidate_records,
                branch_id=branch_id,
                target_key=("hkl",) + target_hkl,
                profile_cache=_runtime_profile_cache(simulation_runtime_state),
            )
            if isinstance(selected_candidate, dict):
                peak_record = selected_candidate
                best_i = _selected_peak_index_from_representative(
                    simulation_runtime_state,
                    selected_candidate,
                )
        else:
            branch_matches = [
                idx
                for idx in range(len(simulation_runtime_state.peak_positions))
                if _peak_hkl_for_index(simulation_runtime_state, idx) == target_hkl
            ]
            if len(branch_matches) > 1:
                selected_idx, selected_record = _select_mosaic_top_peak_record(
                    simulation_runtime_state,
                    branch_matches,
                    branch_id=branch_id,
                    target_key=("hkl",) + target_hkl,
                )
                if isinstance(selected_record, dict):
                    peak_record = selected_record
                    best_i = int(selected_idx)

    selected_native = _record_native_pair(peak_record)
    selected_display = (
        _simulation_point_active_view_pair(
            peak_record,
            use_caked_display=True,
        )
        if bool(caked_view_enabled)
        else _record_display_pair(peak_record)
    )
    if bool(caked_view_enabled):
        peak_record, refined_display = _refine_selected_hkl_caked_record(
            simulation_runtime_state,
            peak_record,
            caked_angles_to_detector_display_coords=(caked_angles_to_detector_display_coords),
            detector_display_to_native_detector_coords=(detector_display_to_native_detector_coords),
        )
        if refined_display is not None:
            selected_display = refined_display
    if bool(caked_view_enabled):
        clicked_native_for_record = (
            (float(selected_native[0]), float(selected_native[1]))
            if selected_native is not None
            else None
        )
    else:
        cx = int(round(float(click_col)))
        cy = int(round(float(click_row)))
        image_shape = (
            tuple(int(v) for v in config.image_shape)
            if config.image_shape is not None
            else (int(config.image_size), int(config.image_size))
        )
        if callable(detector_display_to_native_detector_coords):
            native_coords = detector_display_to_native_detector_coords(
                float(cx),
                float(cy),
            )
            if (
                isinstance(native_coords, tuple)
                and len(native_coords) >= 2
                and np.isfinite(float(native_coords[0]))
                and np.isfinite(float(native_coords[1]))
            ):
                clicked_native_col = float(native_coords[0])
                clicked_native_row = float(native_coords[1])
            elif callable(display_to_native_sim_coords):
                clicked_native_col, clicked_native_row = display_to_native_sim_coords(
                    cx,
                    cy,
                    image_shape,
                )
            else:
                return False
        else:
            if not callable(display_to_native_sim_coords):
                return False
            clicked_native_col, clicked_native_row = display_to_native_sim_coords(
                cx,
                cy,
                image_shape,
            )
        clicked_native_for_record = (float(clicked_native_col), float(clicked_native_row))

    picked = bool(
        select_peak_by_index(
            best_i,
            prefix=f"Nearest peak (Δ={best_dist:.1f}px)",
            sync_hkl_vars=True,
            clicked_display=(float(click_col), float(click_row)),
            clicked_native=clicked_native_for_record,
            selected_display=selected_display,
            selected_native=selected_native,
            record_override=peak_record,
        )
    )
    if picked:
        set_pick_mode(False)
        peak_selection_state.suppress_drag_press_once = True
        sync_peak_selection_state()
    return picked


def _meters_to_millimeters(value: float) -> float:
    return float(value) * 1000.0


def _angstrom_to_meters(value: float) -> float:
    return float(value) * 1.0e-10


def _finite_float(value: object, default: float) -> float:
    try:
        numeric = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return numeric


def _positive_finite_float(value: object, default: float) -> float:
    numeric = _finite_float(value, default)
    if numeric <= 0.0:
        return float(default)
    return numeric


def _profile_values(profile_cache: Mapping[object, object], key: str) -> np.ndarray:
    try:
        values = profile_cache.get(key)
    except Exception:
        return np.empty((0,), dtype=np.float64)
    if values is None:
        return np.empty((0,), dtype=np.float64)
    try:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
    except Exception:
        return np.empty((0,), dtype=np.float64)
    if array.size == 0:
        return array
    return array[np.isfinite(array)]


def _profile_std(profile_cache: Mapping[object, object], key: str) -> float:
    values = _profile_values(profile_cache, key)
    if values.size == 0:
        return 0.0
    std = float(np.std(values))
    if not np.isfinite(std):
        return 0.0
    return std


def _rotation_x_matrix(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )


def _nearest_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return rotation


def _ra_sample_rotation_matrix(config: SelectedPeakIntersectionConfig) -> np.ndarray:
    chi_rad = np.deg2rad(float(config.chi_deg))
    psi_rad = np.deg2rad(float(config.psi_deg))
    psi_z_rad = np.deg2rad(float(config.psi_z_deg))
    theta_initial_rad = np.deg2rad(float(config.theta_initial_deg))
    cor_angle_rad = np.deg2rad(float(config.cor_angle_deg))

    c_chi = float(np.cos(chi_rad))
    s_chi = float(np.sin(chi_rad))
    r_y = np.array(
        [
            [c_chi, 0.0, s_chi],
            [0.0, 1.0, 0.0],
            [-s_chi, 0.0, c_chi],
        ],
        dtype=np.float64,
    )

    c_psi = float(np.cos(psi_rad))
    s_psi = float(np.sin(psi_rad))
    r_z = np.array(
        [
            [c_psi, s_psi, 0.0],
            [-s_psi, c_psi, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    r_z_r_y = r_z @ r_y

    ct = float(np.cos(theta_initial_rad))
    st = float(np.sin(theta_initial_rad))
    ax = float(np.cos(cor_angle_rad))
    ay = 0.0
    az = float(np.sin(cor_angle_rad))
    c_psi_z = float(np.cos(psi_z_rad))
    s_psi_z = float(np.sin(psi_z_rad))
    ax_yawed = c_psi_z * ax + s_psi_z * ay
    ay_yawed = -s_psi_z * ax + c_psi_z * ay
    axis = np.array([ax_yawed, ay_yawed, az], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    ax = float(axis[0])
    ay = float(axis[1])
    az = float(axis[2])
    one_ct = 1.0 - ct
    r_cor = np.array(
        [
            [ct + ax * ax * one_ct, ax * ay * one_ct - az * st, ax * az * one_ct + ay * st],
            [ay * ax * one_ct + az * st, ct + ay * ay * one_ct, ay * az * one_ct - ax * st],
            [az * ax * one_ct - ay * st, az * ay * one_ct + ax * st, ct + az * az * one_ct],
        ],
        dtype=np.float64,
    )
    return _nearest_rotation_matrix(r_cor @ r_z_r_y)


def _decompose_specular_sample_pose(
    config: SelectedPeakIntersectionConfig,
) -> tuple[float, float, float, float]:
    theta_i_deg = float(np.clip(_finite_float(config.theta_initial_deg, 0.0), 0.0, 90.0))
    residual = _nearest_rotation_matrix(
        _ra_sample_rotation_matrix(config) @ _rotation_x_matrix(-np.deg2rad(theta_i_deg))
    )
    alpha_rad = float(np.arcsin(np.clip(residual[1, 0], -1.0, 1.0)))
    cos_alpha = float(np.cos(alpha_rad))
    if abs(cos_alpha) > 1.0e-8:
        psi_rad = float(np.arctan2(-residual[2, 0], residual[0, 0]))
        delta_rad = float(np.arctan2(-residual[1, 2], residual[1, 1]))
    else:
        psi_rad = 0.0
        delta_rad = float(np.arctan2(residual[0, 2], residual[2, 2]))
    return (
        theta_i_deg,
        float(np.rad2deg(delta_rad)),
        float(np.rad2deg(alpha_rad)),
        float(np.rad2deg(psi_rad)),
    )


def _specular_ray_count(simulation_runtime_state) -> int:
    profile_cache = getattr(simulation_runtime_state, "profile_cache", {})
    if not isinstance(profile_cache, Mapping):
        profile_cache = {}
    for key in (
        "wavelength_array",
        "beam_x_array",
        "beam_y_array",
        "theta_array",
        "phi_array",
    ):
        values = _profile_values(profile_cache, key)
        if values.size:
            return int(values.size)
    return max(int(getattr(simulation_runtime_state, "num_samples", 0) or 0), 1)


def _build_selected_peak_specular_initial_state(
    simulation_runtime_state,
    *,
    config: SelectedPeakIntersectionConfig,
    selected_peak: Mapping[object, object],
) -> dict[str, dict[str, float | int]]:
    profile_cache = getattr(simulation_runtime_state, "profile_cache", {})
    if not isinstance(profile_cache, Mapping):
        profile_cache = {}

    ray_count = _specular_ray_count(simulation_runtime_state)
    theta_i_deg, delta_deg, alpha_deg, psi_deg = _decompose_specular_sample_pose(config)
    pixel_size_mm = _meters_to_millimeters(
        _positive_finite_float(
            config.pixel_size_m,
            _SPECULAR_VIEW_DEFAULT_PIXEL_SIZE_M,
        )
    )
    detector_span_mm = max(float(config.image_size), 1.0) * pixel_size_mm
    wavelength_angstrom = _finite_float(config.wavelength_angstrom, 1.5406)
    lattice_a_angstrom = _finite_float(selected_peak.get("av"), 4.143)
    lattice_c_angstrom = _finite_float(selected_peak.get("cv"), 28.636)
    h, k, l = tuple(int(v) for v in selected_peak["hkl"])

    return {
        "specular-view": {
            "rays": int(ray_count),
            "seed": 7,
            "display_rays": int(max(1, min(ray_count, 80))),
            "source_y": -20.0,
            "beam_width_x": _meters_to_millimeters(_profile_std(profile_cache, "beam_x_array")),
            "beam_width_z": _meters_to_millimeters(_profile_std(profile_cache, "beam_y_array")),
            "divergence_x": float(np.rad2deg(_profile_std(profile_cache, "phi_array"))),
            "divergence_z": float(np.rad2deg(_profile_std(profile_cache, "theta_array"))),
            "z_beam": _meters_to_millimeters(-_finite_float(config.zb, 0.0)),
            "sample_width": _meters_to_millimeters(
                _positive_finite_float(
                    config.sample_width_m,
                    _SPECULAR_VIEW_DEFAULT_SAMPLE_WIDTH_M,
                )
            ),
            "sample_height": _meters_to_millimeters(
                _positive_finite_float(
                    config.sample_length_m,
                    _SPECULAR_VIEW_DEFAULT_SAMPLE_HEIGHT_M,
                )
            ),
            "theta_i": theta_i_deg,
            "delta": delta_deg,
            "alpha": alpha_deg,
            "psi": psi_deg,
            "z_sample": _meters_to_millimeters(-_finite_float(config.zs, 0.0)),
            "distance": _meters_to_millimeters(
                _positive_finite_float(
                    config.distance_cor_to_detector,
                    _SPECULAR_VIEW_DEFAULT_DETECTOR_DISTANCE_M,
                )
            ),
            "detector_width": detector_span_mm,
            "detector_height": detector_span_mm,
            "beta": -_finite_float(config.gamma_deg, 0.0),
            "gamma": -_finite_float(config.Gamma_deg, 0.0),
            "chi": 0.0,
            "pixel_u": pixel_size_mm,
            "pixel_v": pixel_size_mm,
            "i0": _finite_float(config.center_col, 0.0),
            "j0": _finite_float(config.center_row, 0.0),
            "H": int(h),
            "K": int(k),
            "L": int(l),
            "sigma_deg": _finite_float(config.sigma_mosaic_deg, 0.8),
            "mosaic_gamma_deg": _finite_float(config.gamma_mosaic_deg, 5.0),
            "eta": _finite_float(config.eta, 0.5),
            "wavelength_m": _angstrom_to_meters(wavelength_angstrom),
            "lattice_a_m": _angstrom_to_meters(lattice_a_angstrom),
            "lattice_c_m": _angstrom_to_meters(lattice_c_angstrom),
        }
    }


def open_selected_peak_intersection_figure(
    simulation_runtime_state,
    *,
    config: SelectedPeakIntersectionConfig,
    n2: Any,
    set_status_text: Any,
    launch_specular_visualizer: Any = None,
) -> bool:
    """Open the external 2D Mosaic specular view for the selected peak."""

    del n2
    selected_peak = simulation_runtime_state.selected_peak_record
    if not isinstance(selected_peak, Mapping):
        set_status_text("Select a Bragg peak first (arm Pick HKL on Image or use HKL controls).")
        return False

    if launch_specular_visualizer is None:
        from ra_sim import launcher as app_launcher

        launch_specular_visualizer = app_launcher.launch_mosaic_specular_visualizer

    try:
        h, k, l = tuple(int(v) for v in selected_peak["hkl"])
        initial_state = _build_selected_peak_specular_initial_state(
            simulation_runtime_state,
            config=config,
            selected_peak=selected_peak,
        )
        launch_specular_visualizer(initial_state)
        set_status_text(
            f"Opened 2D Mosaic specular view for HKL=({h} {k} {l}) "
            f"from source={selected_peak.get('source_label', 'unknown')}."
        )
        return True
    except Exception as exc:
        set_status_text(f"Specular visualizer launch failed for selected peak: {exc}")
        return False


def update_runtime_hkl_pick_button_label(
    bindings: SelectedPeakRuntimeBindings,
) -> None:
    """Refresh the runtime HKL image-pick button label."""

    if bindings.hkl_lookup_view_state is None:
        return
    gui_views.set_hkl_pick_button_text(
        bindings.hkl_lookup_view_state,
        hkl_pick_button_text(bool(bindings.peak_selection_state.hkl_pick_armed)),
    )


def runtime_selected_peak_overlay_needed(
    peak_selection_state,
    simulation_runtime_state,
    *,
    live_geometry_preview_enabled: bool,
) -> bool:
    """Return whether the current runtime state still needs peak overlay data."""

    return bool(
        getattr(peak_selection_state, "hkl_pick_armed", False)
        or getattr(peak_selection_state, "selected_hkl_target", None) is not None
        or getattr(simulation_runtime_state, "selected_peak_record", None) is not None
        or bool(live_geometry_preview_enabled)
    )


def refresh_runtime_selected_peak_after_simulation_update(
    bindings: SelectedPeakRuntimeBindings,
    *,
    live_geometry_preview_enabled: bool,
) -> bool:
    """Refresh runtime peak-overlay/selection state after one simulation update."""

    need_peak_overlay = runtime_selected_peak_overlay_needed(
        bindings.peak_selection_state,
        bindings.simulation_runtime_state,
        live_geometry_preview_enabled=bool(live_geometry_preview_enabled),
    )
    if need_peak_overlay:
        overlay_ready = bool(bindings.ensure_peak_overlay_data(force=False))
    else:
        _clear_peak_overlay_lists(bindings.simulation_runtime_state)
        overlay_ready = False

    if getattr(bindings.peak_selection_state, "selected_hkl_target", None) is not None:
        reselection_ok = reselect_runtime_selected_peak(bindings)
        if not reselection_ok:
            _hide_runtime_selected_peak_marker(bindings)
            _runtime_draw_idle(bindings)
    return overlay_ready


def normalize_runtime_selected_hkl_target(
    selected_hkl_target: object,
) -> tuple[int, int, int] | None:
    """Normalize one restored selected-HKL target from saved/runtime state."""

    if not isinstance(selected_hkl_target, (list, tuple)) or len(selected_hkl_target) < 3:
        return None
    try:
        return (
            int(selected_hkl_target[0]),
            int(selected_hkl_target[1]),
            int(selected_hkl_target[2]),
        )
    except Exception:
        return None


def apply_runtime_restored_selected_hkl_target(
    bindings: SelectedPeakRuntimeBindings,
    selected_hkl_target: object,
) -> tuple[int, int, int] | None:
    """Apply one restored selected-HKL target back onto the live runtime state."""

    normalized_target = normalize_runtime_selected_hkl_target(selected_hkl_target)
    bindings.peak_selection_state.selected_hkl_target = normalized_target
    _sync_runtime_peak_selection_state(bindings)
    update_runtime_hkl_pick_button_label(bindings)
    return normalized_target


def set_runtime_hkl_pick_mode(
    bindings: SelectedPeakRuntimeBindings,
    enabled: bool,
    *,
    message: str | None = None,
) -> None:
    """Arm or disarm runtime HKL image-pick mode."""

    bindings.peak_selection_state.hkl_pick_armed = bool(enabled)
    _sync_runtime_peak_selection_state(bindings)
    if bindings.peak_selection_state.hkl_pick_armed and callable(
        bindings.deactivate_conflicting_modes
    ):
        bindings.deactivate_conflicting_modes()
    update_runtime_hkl_pick_button_label(bindings)
    if callable(bindings.on_hkl_pick_mode_changed):
        try:
            bindings.on_hkl_pick_mode_changed(bindings.peak_selection_state.hkl_pick_armed)
        except Exception:
            pass
    if message:
        _set_status_text(bindings.set_status_text, message)


def toggle_runtime_hkl_pick_mode(bindings: SelectedPeakRuntimeBindings) -> None:
    """Toggle runtime HKL image-pick mode using the current GUI state."""

    toggle_hkl_pick_mode(
        bindings.simulation_runtime_state,
        bindings.peak_selection_state,
        caked_view_enabled=bool(_resolve_runtime_value(bindings.caked_view_enabled_factory)),
        ensure_peak_overlay_data=bindings.ensure_peak_overlay_data,
        schedule_update=(
            bindings.schedule_update if callable(bindings.schedule_update) else lambda: None
        ),
        set_pick_mode=lambda enabled, message=None: set_runtime_hkl_pick_mode(
            bindings,
            enabled,
            message=message,
        ),
        set_status_text=lambda text: _set_status_text(bindings.set_status_text, text),
    )


def select_peak_by_hkl_runtime(
    bindings: SelectedPeakRuntimeBindings,
    h: int,
    k: int,
    l: int,
    *,
    sync_hkl_vars: bool = True,
    silent_if_missing: bool = False,
) -> bool:
    """Select one peak by HKL using live runtime bindings."""

    if bindings.selected_peak_marker is None:
        return False
    caked_view_enabled = _runtime_bool(bindings.caked_view_enabled_factory, False)
    return select_peak_by_hkl(
        bindings.simulation_runtime_state,
        bindings.peak_selection_state,
        bindings.hkl_lookup_view_state,
        bindings.selected_peak_marker,
        h,
        k,
        l,
        primary_a=_runtime_primary_a(bindings),
        ensure_peak_overlay_data=bindings.ensure_peak_overlay_data,
        schedule_update=(
            bindings.schedule_update if callable(bindings.schedule_update) else lambda: None
        ),
        sync_peak_selection_state=lambda: _sync_runtime_peak_selection_state(bindings),
        set_status_text=lambda text: _set_status_text(bindings.set_status_text, text),
        draw_idle=lambda: _runtime_draw_idle(bindings),
        caked_view_enabled=bool(caked_view_enabled),
        sync_hkl_vars=sync_hkl_vars,
        silent_if_missing=silent_if_missing,
        caked_angles_to_detector_display_coords=(bindings.caked_angles_to_detector_display_coords),
        detector_display_to_native_detector_coords=(
            bindings.detector_display_to_native_detector_coords
        ),
    )


def _reselect_runtime_refined_detector_record(
    bindings: SelectedPeakRuntimeBindings,
    target: Sequence[int],
) -> bool:
    """Keep a caked-refined simulation record selected when returning to detector view."""

    if _runtime_bool(bindings.caked_view_enabled_factory, False):
        return False
    if len(target) < 3:
        return False
    record = getattr(bindings.simulation_runtime_state, "selected_peak_record", None)
    if not (
        isinstance(record, Mapping)
        and _record_matches_hkl_target(record, target)
        and _record_has_caked_refinement(record)
        and _record_refinement_signature_matches(record, bindings.simulation_runtime_state)
    ):
        return False
    detector_point = _record_detector_display_pair(record)
    if detector_point is None:
        return False
    detector_col, detector_row = float(detector_point[0]), float(detector_point[1])
    if not (np.isfinite(detector_col) and np.isfinite(detector_row)):
        return False

    marker = bindings.selected_peak_marker
    if marker is None:
        return False
    marker.set_data([detector_col], [detector_row])
    marker.set_visible(True)

    selected_record = dict(record)
    selected_record["display_col"] = detector_col
    selected_record["display_row"] = detector_row
    selected_record["sim_col"] = detector_col
    selected_record["sim_row"] = detector_row
    selected_record["sim_col_raw"] = detector_col
    selected_record["sim_row_raw"] = detector_row
    bindings.simulation_runtime_state.selected_peak_record = selected_record
    bindings.peak_selection_state.selected_hkl_target = (
        int(target[0]),
        int(target[1]),
        int(target[2]),
    )
    _sync_runtime_peak_selection_state(bindings)
    _runtime_draw_idle(bindings)
    return True


def reselect_runtime_selected_peak(bindings: SelectedPeakRuntimeBindings) -> bool:
    """Refresh the currently selected HKL target after a simulation update."""

    target = getattr(bindings.peak_selection_state, "selected_hkl_target", None)
    if target is None or len(target) < 3:
        return False
    if _reselect_runtime_refined_detector_record(bindings, target):
        return True
    return select_peak_by_hkl_runtime(
        bindings,
        int(target[0]),
        int(target[1]),
        int(target[2]),
        sync_hkl_vars=False,
        silent_if_missing=True,
    )


def select_peak_from_runtime_hkl_controls(
    bindings: SelectedPeakRuntimeBindings,
) -> bool:
    """Select one peak from the live HKL lookup controls."""

    if bindings.selected_peak_marker is None:
        return False
    return select_peak_from_hkl_controls(
        bindings.simulation_runtime_state,
        bindings.peak_selection_state,
        bindings.hkl_lookup_view_state,
        bindings.selected_peak_marker,
        primary_a=_runtime_primary_a(bindings),
        ensure_peak_overlay_data=bindings.ensure_peak_overlay_data,
        schedule_update=(
            bindings.schedule_update if callable(bindings.schedule_update) else lambda: None
        ),
        sync_peak_selection_state=lambda: _sync_runtime_peak_selection_state(bindings),
        set_status_text=lambda text: _set_status_text(bindings.set_status_text, text),
        draw_idle=lambda: _runtime_draw_idle(bindings),
        caked_view_enabled=_runtime_bool(bindings.caked_view_enabled_factory, False),
        tcl_error_types=tuple(bindings.tcl_error_types or ()),
        caked_angles_to_detector_display_coords=(bindings.caked_angles_to_detector_display_coords),
        detector_display_to_native_detector_coords=(
            bindings.detector_display_to_native_detector_coords
        ),
    )


def clear_runtime_selected_peak(bindings: SelectedPeakRuntimeBindings) -> None:
    """Clear the selected runtime peak state."""

    if bindings.selected_peak_marker is None:
        return
    clear_selected_peak(
        bindings.simulation_runtime_state,
        bindings.peak_selection_state,
        bindings.selected_peak_marker,
        sync_peak_selection_state=lambda: _sync_runtime_peak_selection_state(bindings),
        set_status_text=lambda text: _set_status_text(bindings.set_status_text, text),
        draw_idle=lambda: _runtime_draw_idle(bindings),
    )


def open_runtime_selected_peak_intersection_figure(
    bindings: SelectedPeakRuntimeBindings,
) -> bool:
    """Open the selected-peak intersection plot using live runtime controls."""

    config = _runtime_intersection_config(bindings)
    if config is None:
        _set_status_text(bindings.set_status_text, "Selected-peak analysis is unavailable.")
        return False
    return open_selected_peak_intersection_figure(
        bindings.simulation_runtime_state,
        config=config,
        n2=bindings.n2,
        set_status_text=lambda text: _set_status_text(bindings.set_status_text, text),
    )


def select_peak_from_runtime_canvas_click(
    bindings: SelectedPeakRuntimeBindings,
    click_col: float,
    click_row: float,
) -> bool:
    """Resolve one raw-image click through the live runtime bindings."""

    if bindings.selected_peak_marker is None:
        return False
    config = _runtime_canvas_pick_config(bindings)
    if config is None:
        _set_status_text(bindings.set_status_text, "HKL image-pick is unavailable.")
        return False
    caked_view_enabled = _runtime_bool(bindings.caked_view_enabled_factory, False)
    if not bool(caked_view_enabled) and not (
        callable(bindings.display_to_native_sim_coords)
        or callable(bindings.detector_display_to_native_detector_coords)
    ):
        return False
    simulation_point_candidates = _runtime_hkl_pick_simulation_point_payload(bindings)
    return select_peak_from_canvas_click(
        bindings.simulation_runtime_state,
        bindings.peak_selection_state,
        float(click_col),
        float(click_row),
        config=config,
        ensure_peak_overlay_data=bindings.ensure_peak_overlay_data,
        schedule_update=(
            bindings.schedule_update if callable(bindings.schedule_update) else lambda: None
        ),
        display_to_native_sim_coords=bindings.display_to_native_sim_coords,
        native_sim_to_display_coords=bindings.native_sim_to_display_coords,
        simulate_ideal_hkl_native_center=bindings.simulate_ideal_hkl_native_center,
        select_peak_by_index=lambda idx, **kwargs: select_peak_by_index(
            bindings.simulation_runtime_state,
            bindings.peak_selection_state,
            bindings.hkl_lookup_view_state,
            bindings.selected_peak_marker,
            idx,
            primary_a=_runtime_primary_a(bindings),
            sync_peak_selection_state=lambda: _sync_runtime_peak_selection_state(bindings),
            set_status_text=lambda text: _set_status_text(bindings.set_status_text, text),
            draw_idle=lambda: _runtime_draw_idle(bindings),
            **kwargs,
        ),
        set_pick_mode=lambda enabled, message=None: set_runtime_hkl_pick_mode(
            bindings,
            enabled,
            message=message,
        ),
        sync_peak_selection_state=lambda: _sync_runtime_peak_selection_state(bindings),
        set_status_text=lambda text: _set_status_text(bindings.set_status_text, text),
        caked_view_enabled=bool(caked_view_enabled),
        detector_display_to_native_detector_coords=(
            bindings.detector_display_to_native_detector_coords
        ),
        caked_angles_to_detector_display_coords=(bindings.caked_angles_to_detector_display_coords),
        simulation_point_candidates=simulation_point_candidates,
    )


def find_peak_record_from_runtime_canvas_click(
    bindings: SelectedPeakRuntimeBindings,
    click_col: float,
    click_row: float,
    *,
    max_axis_distance_px: float,
) -> tuple[int, dict[str, object] | None, float, bool]:
    """Return nearest visible live peak for one click under runtime bindings."""

    use_caked_display = _runtime_bool(bindings.caked_view_enabled_factory, False)
    simulation_point_candidates = _runtime_hkl_pick_simulation_point_payload(bindings)
    return find_peak_record_for_canvas_click(
        bindings.simulation_runtime_state,
        float(click_col),
        float(click_row),
        ensure_peak_overlay_data=bindings.ensure_peak_overlay_data,
        max_axis_distance_px=float(max_axis_distance_px),
        simulation_point_candidates=simulation_point_candidates,
        use_caked_display=use_caked_display,
    )


def make_runtime_peak_selection_bindings_factory(
    *,
    simulation_runtime_state,
    peak_selection_state,
    hkl_lookup_view_state_factory: object,
    selected_peak_marker_factory: object,
    current_primary_a_factory: object,
    caked_view_enabled_factory: object,
    current_canvas_pick_config_factory: object,
    current_intersection_config_factory: object,
    ensure_peak_overlay_data: Callable[..., object],
    sync_peak_selection_state: Callable[[], None] | None,
    schedule_update_factory: object = None,
    set_status_text_factory: object = None,
    draw_idle_factory: object = None,
    display_to_native_sim_coords: Callable[..., tuple[float, float]] | None = None,
    detector_display_to_native_detector_coords: Callable[[float, float], tuple[float, float] | None]
    | None = None,
    native_sim_to_display_coords: Callable[..., tuple[float, float]] | None = None,
    native_detector_coords_to_detector_display_coords: Callable[
        [float, float], tuple[float, float] | None
    ]
    | None = None,
    caked_angles_to_detector_display_coords: Callable[[float, float], tuple[float, float] | None]
    | None = None,
    hkl_pick_simulation_points_factory: Callable[[], object] | None = None,
    simulate_ideal_hkl_native_center: Callable[..., tuple[float, float] | None] | None = None,
    deactivate_conflicting_modes_factory: object = None,
    on_hkl_pick_mode_changed_factory: object = None,
    n2: Any = None,
    tcl_error_types: tuple[type[BaseException], ...] = (),
):
    """Build a factory that resolves the live runtime peak-selection bindings."""

    def _build() -> SelectedPeakRuntimeBindings:
        return SelectedPeakRuntimeBindings(
            simulation_runtime_state=simulation_runtime_state,
            peak_selection_state=peak_selection_state,
            hkl_lookup_view_state=_resolve_runtime_value(hkl_lookup_view_state_factory),
            selected_peak_marker=_resolve_runtime_value(selected_peak_marker_factory),
            current_primary_a_factory=current_primary_a_factory,
            caked_view_enabled_factory=caked_view_enabled_factory,
            current_canvas_pick_config_factory=current_canvas_pick_config_factory,
            current_intersection_config_factory=current_intersection_config_factory,
            ensure_peak_overlay_data=ensure_peak_overlay_data,
            sync_peak_selection_state=sync_peak_selection_state,
            schedule_update=_resolve_runtime_value(schedule_update_factory),
            set_status_text=_resolve_runtime_value(set_status_text_factory),
            draw_idle=_resolve_runtime_value(draw_idle_factory),
            display_to_native_sim_coords=display_to_native_sim_coords,
            detector_display_to_native_detector_coords=(detector_display_to_native_detector_coords),
            native_sim_to_display_coords=native_sim_to_display_coords,
            native_detector_coords_to_detector_display_coords=(
                native_detector_coords_to_detector_display_coords
            ),
            caked_angles_to_detector_display_coords=(caked_angles_to_detector_display_coords),
            hkl_pick_simulation_points_factory=hkl_pick_simulation_points_factory,
            simulate_ideal_hkl_native_center=simulate_ideal_hkl_native_center,
            deactivate_conflicting_modes=_resolve_runtime_value(
                deactivate_conflicting_modes_factory
            ),
            on_hkl_pick_mode_changed=_resolve_runtime_value(on_hkl_pick_mode_changed_factory),
            n2=n2,
            tcl_error_types=tuple(tcl_error_types or ()),
        )

    return _build


def make_runtime_peak_selection_callbacks(
    bindings_factory: Callable[[], SelectedPeakRuntimeBindings],
) -> SelectedPeakRuntimeCallbacks:
    """Build bound callbacks for the runtime selected-peak workflow."""

    return SelectedPeakRuntimeCallbacks(
        update_hkl_pick_button_label=lambda: update_runtime_hkl_pick_button_label(
            bindings_factory()
        ),
        set_hkl_pick_mode=lambda enabled, message=None: set_runtime_hkl_pick_mode(
            bindings_factory(),
            enabled,
            message=message,
        ),
        toggle_hkl_pick_mode=lambda: toggle_runtime_hkl_pick_mode(bindings_factory()),
        reselect_current_peak=lambda: reselect_runtime_selected_peak(bindings_factory()),
        select_peak_from_hkl_controls=lambda: select_peak_from_runtime_hkl_controls(
            bindings_factory()
        ),
        clear_selected_peak=lambda: clear_runtime_selected_peak(bindings_factory()),
        open_selected_peak_intersection_figure=lambda: (
            open_runtime_selected_peak_intersection_figure(bindings_factory())
        ),
        select_peak_from_canvas_click=lambda click_col, click_row: (
            select_peak_from_runtime_canvas_click(
                bindings_factory(),
                float(click_col),
                float(click_row),
            )
        ),
        find_peak_record_for_canvas_click=lambda click_col, click_row, max_axis_distance_px: (
            find_peak_record_from_runtime_canvas_click(
                bindings_factory(),
                float(click_col),
                float(click_row),
                max_axis_distance_px=float(max_axis_distance_px),
            )
        ),
    )


def make_runtime_peak_selection_maintenance_callbacks(
    bindings_factory: Callable[[], SelectedPeakRuntimeBindings],
) -> SelectedPeakRuntimeMaintenanceCallbacks:
    """Build bound refresh/restore callbacks for runtime selected-peak upkeep."""

    return SelectedPeakRuntimeMaintenanceCallbacks(
        refresh_after_simulation_update=lambda live_geometry_preview_enabled: (
            refresh_runtime_selected_peak_after_simulation_update(
                bindings_factory(),
                live_geometry_preview_enabled=bool(live_geometry_preview_enabled),
            )
        ),
        apply_restored_selected_hkl_target=lambda selected_hkl_target: (
            apply_runtime_restored_selected_hkl_target(
                bindings_factory(),
                selected_hkl_target,
            )
        ),
    )
