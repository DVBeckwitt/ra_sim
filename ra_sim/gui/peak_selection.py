"""Workflow helpers for HKL lookup and selected-peak state."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ra_sim.simulation.intersection_analysis import (
    BeamSamples as IntersectionBeamSamples,
    IntersectionGeometry,
    MosaicParams as IntersectionMosaicParams,
    analyze_reflection_intersection,
    plot_intersection_analysis,
)

from . import views as gui_views


@dataclass(frozen=True)
class SelectedPeakIntersectionConfig:
    """Scalar GUI inputs needed to inspect one selected peak intersection."""

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
    solve_q_steps: int
    solve_q_rel_tol: float
    solve_q_mode: int
    pixel_size_m: float = 100e-6


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
    native_sim_to_display_coords: Callable[..., tuple[float, float]] | None = None
    simulate_ideal_hkl_native_center: Callable[..., tuple[float, float] | None] | None = None
    deactivate_conflicting_modes: Callable[[], None] | None = None
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

    normalized_shape = (
        tuple(int(v) for v in image_shape)
        if image_shape is not None
        else None
    )
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
    solve_q_values: object,
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
        solve_q_steps=solve_q_steps,
        solve_q_rel_tol=solve_q_rel_tol,
        solve_q_mode=solve_q_mode,
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
    solve_q_values_factory: object,
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
        solve_q_values=solve_q_values_factory,
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
            solve_q_values_factory=solve_q_values_factory,
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
    if (
        source_miller.ndim != 2
        or source_miller.shape[1] < 3
        or source_miller.shape[0] == 0
    ):
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
        nearest_l = int(
            l_vals[rod_mask][np.argmin(np.abs(l_vals[rod_mask] - int(l)))]
        )
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
    deg_text = ", ".join(
        format_hkl_triplet(hv, kv, lv) for hv, kv, lv in shown_deg
    )
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

    finite_mask = (
        np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & np.isfinite(arr[:, 2])
    )
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

    def _run_single(
        beam_x: np.ndarray,
        beam_y: np.ndarray,
        theta_arr: np.ndarray,
        phi_arr: np.ndarray,
        wavelength_arr: np.ndarray,
        *,
        forced_sample_indices: np.ndarray | None = None,
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
                single_sample_indices=forced_sample_indices,
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
    )
    if strict_center is not None:
        return strict_center

    beam_x = np.asarray(
        simulation_runtime_state.profile_cache.get("beam_x_array", []),
        dtype=np.float64,
    ).ravel()
    beam_y = np.asarray(
        simulation_runtime_state.profile_cache.get("beam_y_array", []),
        dtype=np.float64,
    ).ravel()
    theta_arr = np.asarray(
        simulation_runtime_state.profile_cache.get("theta_array", []),
        dtype=np.float64,
    ).ravel()
    phi_arr = np.asarray(
        simulation_runtime_state.profile_cache.get("phi_array", []),
        dtype=np.float64,
    ).ravel()
    wavelength_arr = np.asarray(
        simulation_runtime_state.profile_cache.get("wavelength_array", []),
        dtype=np.float64,
    ).ravel()

    n_samp = beam_x.size
    if (
        n_samp == 0
        or beam_y.size != n_samp
        or theta_arr.size != n_samp
        or phi_arr.size != n_samp
        or wavelength_arr.size != n_samp
    ):
        return None

    metric = theta_arr * theta_arr + phi_arr * phi_arr
    valid_metric = np.where(np.isfinite(metric), metric, np.inf)
    best_idx = int(np.argmin(valid_metric))
    if not np.isfinite(valid_metric[best_idx]):
        return None

    forced = np.array([best_idx], dtype=np.int64)
    return _run_single(
        beam_x,
        beam_y,
        theta_arr,
        phi_arr,
        wavelength_arr,
        forced_sample_indices=forced,
    )


def _clear_peak_overlay_lists(simulation_runtime_state) -> None:
    simulation_runtime_state.peak_positions.clear()
    simulation_runtime_state.peak_millers.clear()
    simulation_runtime_state.peak_intensities.clear()
    simulation_runtime_state.peak_records.clear()


def ensure_runtime_peak_overlay_data(
    simulation_runtime_state,
    *,
    primary_a: object,
    primary_c: object,
    native_sim_to_display_coords: Callable[..., tuple[float, float]],
    reflection_q_group_metadata: Callable[..., tuple[object, object, object]],
    max_hits_per_reflection: object = 0,
    min_separation_px: object = 0.0,
    force: bool = False,
) -> bool:
    """Ensure the live simulated-peak overlay cache is populated for one frame."""

    if (
        simulation_runtime_state.stored_max_positions_local is None
        or simulation_runtime_state.stored_sim_image is None
    ):
        _clear_peak_overlay_lists(simulation_runtime_state)
        return False

    max_positions_local = simulation_runtime_state.stored_max_positions_local
    updated_image = simulation_runtime_state.stored_sim_image
    peak_table_lattice_local = simulation_runtime_state.stored_peak_table_lattice

    fallback_a = _runtime_float(primary_a, float("nan"))
    fallback_c = _runtime_float(primary_c, float("nan"))
    if (
        not peak_table_lattice_local
        or len(peak_table_lattice_local) != len(max_positions_local)
    ):
        peak_table_lattice_local = [
            (fallback_a, fallback_c, "primary")
            for _ in max_positions_local
        ]

    max_hits_raw = _runtime_int(max_hits_per_reflection, 0)
    min_separation_value = _runtime_float(min_separation_px, 0.0)
    peak_sig = (
        simulation_runtime_state.last_simulation_signature,
        id(max_positions_local),
        len(max_positions_local),
        tuple(updated_image.shape),
        int(max_hits_raw),
        float(min_separation_value),
    )
    peak_cached = (
        not force
        and simulation_runtime_state.peak_overlay_cache.get("sig") == peak_sig
    )

    _clear_peak_overlay_lists(simulation_runtime_state)

    if peak_cached:
        simulation_runtime_state.peak_positions.extend(
            list(simulation_runtime_state.peak_overlay_cache.get("positions", ()))
        )
        simulation_runtime_state.peak_millers.extend(
            list(simulation_runtime_state.peak_overlay_cache.get("millers", ()))
        )
        simulation_runtime_state.peak_intensities.extend(
            list(simulation_runtime_state.peak_overlay_cache.get("intensities", ()))
        )
        simulation_runtime_state.peak_records.extend(
            dict(rec)
            for rec in simulation_runtime_state.peak_overlay_cache.get("records", ())
        )
        return True

    max_hits_limit = max_hits_raw if max_hits_raw > 0 else None
    min_sep_sq = float(min_separation_value) ** 2
    image_shape = tuple(int(v) for v in updated_image.shape)

    for table_idx, tbl in enumerate(max_positions_local):
        tbl_arr = np.asarray(tbl, dtype=float)
        if tbl_arr.ndim != 2 or tbl_arr.shape[0] == 0 or tbl_arr.shape[1] < 7:
            continue

        lattice_entry = peak_table_lattice_local[table_idx]
        try:
            av_used = float(lattice_entry[0])
        except Exception:
            av_used = float(fallback_a)
        try:
            cv_used = float(lattice_entry[1])
        except Exception:
            cv_used = float(fallback_c)
        try:
            source_label = str(lattice_entry[2])
        except Exception:
            source_label = "primary"

        row_order = np.argsort(tbl_arr[:, 0])[::-1]
        chosen_rows: list[tuple[int, np.ndarray, float, float, float, float]] = []

        for row_idx in row_order:
            row = tbl_arr[row_idx]
            I, xpix, ypix, phi_val, H, K, L = row[:7]
            if not (np.isfinite(I) and np.isfinite(xpix) and np.isfinite(ypix)):
                continue
            cx = float(xpix)
            cy = float(ypix)
            disp_cx, disp_cy = native_sim_to_display_coords(cx, cy, image_shape)
            too_close = False
            for _, _, _, _, prev_col, prev_row in chosen_rows:
                d2 = (disp_cx - prev_col) ** 2 + (disp_cy - prev_row) ** 2
                if d2 < min_sep_sq:
                    too_close = True
                    break
            if too_close:
                continue
            chosen_rows.append((int(row_idx), row, cx, cy, disp_cx, disp_cy))
            if max_hits_limit is not None and len(chosen_rows) >= max_hits_limit:
                break

        for row_idx, row, cx, cy, disp_cx, disp_cy in chosen_rows:
            I, _xpix, _ypix, phi_val, H, K, L = row[:7]
            simulation_runtime_state.peak_positions.append((disp_cx, disp_cy))
            simulation_runtime_state.peak_intensities.append(float(I))
            hkl = tuple(int(np.rint(val)) for val in (H, K, L))
            hkl_raw = (float(H), float(K), float(L))
            m_val = float(H * H + H * K + K * K)
            qr_val = (
                (2.0 * np.pi / float(av_used)) * np.sqrt((4.0 / 3.0) * m_val)
                if float(av_used) > 0.0
                and np.isfinite(float(av_used))
                and m_val >= 0.0
                else float("nan")
            )
            q_group_key, _, qz_val = reflection_q_group_metadata(
                hkl_raw,
                source_label=source_label,
                a_value=av_used,
                c_value=cv_used,
                qr_value=qr_val,
            )
            simulation_runtime_state.peak_millers.append(hkl)
            simulation_runtime_state.peak_records.append(
                {
                    "display_col": float(disp_cx),
                    "display_row": float(disp_cy),
                    "native_col": float(cx),
                    "native_row": float(cy),
                    "hkl": hkl,
                    "hkl_raw": hkl_raw,
                    "intensity": float(I),
                    "qr": float(qr_val),
                    "qz": float(qz_val),
                    "q_group_key": q_group_key,
                    "phi": float(phi_val),
                    "source_table_index": int(table_idx),
                    "source_row_index": int(row_idx),
                    "source_label": str(source_label),
                    "av": float(av_used),
                    "cv": float(cv_used),
                }
            )

    simulation_runtime_state.peak_overlay_cache.update(
        {
            "sig": peak_sig,
            "positions": list(simulation_runtime_state.peak_positions),
            "millers": list(simulation_runtime_state.peak_millers),
            "intensities": list(simulation_runtime_state.peak_intensities),
            "records": [dict(rec) for rec in simulation_runtime_state.peak_records],
        }
    )
    return True


def make_runtime_peak_overlay_data_callback(
    *,
    simulation_runtime_state,
    primary_a_factory: object,
    primary_c_factory: object,
    native_sim_to_display_coords: Callable[..., tuple[float, float]],
    reflection_q_group_metadata: Callable[..., tuple[object, object, object]],
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
        record["selected_native_col"] = float(record["native_col"])
        record["selected_native_row"] = float(record["native_row"])
    return record


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
) -> bool:
    """Select one simulated peak by cached index and update GUI state."""

    if idx < 0 or idx >= len(simulation_runtime_state.peak_positions):
        return False

    px, py = simulation_runtime_state.peak_positions[idx]
    H, K, L = simulation_runtime_state.peak_millers[idx]
    intensity = simulation_runtime_state.peak_intensities[idx]
    disp_col, disp_row = (
        (float(selected_display[0]), float(selected_display[1]))
        if selected_display is not None
        else (float(px), float(py))
    )

    selected_peak_marker.set_data([disp_col], [disp_row])
    selected_peak_marker.set_visible(True)

    peak_selection_state.selected_hkl_target = (int(H), int(K), int(L))
    sync_peak_selection_state()

    selected_record = _copy_selected_peak_record(simulation_runtime_state, idx)
    selected_record = _apply_selected_peak_record_coordinates(
        selected_record,
        clicked_display=clicked_display,
        clicked_native=clicked_native,
        selected_display=selected_display,
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
    sync_hkl_vars: bool = True,
    silent_if_missing: bool = False,
) -> bool:
    """Select the simulated peak matching one requested integer HKL."""

    ensure_peak_overlay_data(force=False)
    target = (int(h), int(k), int(l))

    if not simulation_runtime_state.peak_positions:
        if (
            not silent_if_missing
            and simulation_runtime_state.unscaled_image is not None
        ):
            schedule_update()
        if not silent_if_missing:
            set_status_text("Preparing simulated peak map... try again after update.")
        return False

    matches = [
        idx
        for idx, hkl_triplet in enumerate(simulation_runtime_state.peak_millers)
        if (
            tuple(int(np.rint(v)) for v in hkl_triplet) == target
            and simulation_runtime_state.peak_positions[idx][0] >= 0
        )
    ]

    if not matches:
        m_target = _m_index(target)
        l_target = int(target[2])
        matches = [
            idx
            for idx, hkl_triplet in enumerate(simulation_runtime_state.peak_millers)
            if (
                simulation_runtime_state.peak_positions[idx][0] >= 0
                and int(np.rint(hkl_triplet[2])) == l_target
                and _m_index(
                    (
                        int(np.rint(hkl_triplet[0])),
                        int(np.rint(hkl_triplet[1])),
                        int(np.rint(hkl_triplet[2])),
                    )
                )
                == m_target
            )
        ]

    if not matches:
        if not silent_if_missing:
            set_status_text(
                f"HKL ({target[0]} {target[1]} {target[2]}) not found in current simulation."
            )
        peak_selection_state.selected_hkl_target = target
        sync_peak_selection_state()
        simulation_runtime_state.selected_peak_record = None
        return False

    def _score(idx_value: int) -> float:
        val = simulation_runtime_state.peak_intensities[idx_value]
        return float(val) if np.isfinite(val) else float("-inf")

    best_idx = max(matches, key=_score)
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
    tcl_error_types: tuple[type[BaseException], ...] = (),
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
        sync_hkl_vars=True,
        silent_if_missing=False,
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

    if bool(caked_view_enabled):
        set_status_text("Switch off 2D caked view before picking HKL in the image.")
        return

    if simulation_runtime_state.unscaled_image is None:
        set_status_text("Run a simulation first.")
        return

    if (
        not ensure_peak_overlay_data(force=False)
        or not simulation_runtime_state.peak_positions
    ):
        set_pick_mode(
            True,
            message=(
                "Preparing simulated peak map for HKL picking... "
                "wait for the next update."
            ),
        )
        schedule_update()
        return

    set_pick_mode(
        True,
        message="HKL image-pick armed: click near a Bragg peak in raw camera view.",
    )


def _nearest_peak_index_for_click(
    simulation_runtime_state,
    click_col: float,
    click_row: float,
) -> tuple[int, float]:
    best_i = -1
    best_d2 = float("inf")
    best_i_val = float("-inf")
    for i, (px, py) in enumerate(simulation_runtime_state.peak_positions):
        if float(px) < 0.0:
            continue
        d2 = (float(px) - float(click_col)) ** 2 + (float(py) - float(click_row)) ** 2
        val = simulation_runtime_state.peak_intensities[i]
        score_val = float(val) if np.isfinite(val) else float("-inf")
        if d2 < best_d2 - 1e-9 or (
            abs(d2 - best_d2) <= 1e-9 and score_val > best_i_val
        ):
            best_i = int(i)
            best_d2 = float(d2)
            best_i_val = float(score_val)
    return best_i, best_d2


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
) -> bool:
    """Select the nearest visible peak from one raw-image click."""

    ensure_peak_overlay_data(force=False)
    if not simulation_runtime_state.peak_positions:
        schedule_update()
        set_status_text("Preparing simulated peak map... click again after update.")
        return False

    best_i, best_d2 = _nearest_peak_index_for_click(
        simulation_runtime_state,
        float(click_col),
        float(click_row),
    )
    if best_i == -1:
        set_status_text("No peaks on screen.")
        return False
    if best_d2 > float(config.max_distance_px) ** 2:
        set_status_text(
            f"No simulated peak within {float(config.max_distance_px):.0f}px "
            f"(nearest is {best_d2**0.5:.1f}px away)."
        )
        return False

    cx = int(round(float(click_col)))
    cy = int(round(float(click_row)))
    image_shape = (
        tuple(int(v) for v in config.image_shape)
        if config.image_shape is not None
        else (int(config.image_size), int(config.image_size))
    )
    clicked_native_col, clicked_native_row = display_to_native_sim_coords(
        cx,
        cy,
        image_shape,
    )
    selected_display, selected_native = _resolve_selected_peak_click_coordinates(
        simulation_runtime_state,
        best_i,
        click_col=float(click_col),
        click_row=float(click_row),
        clicked_native_col=float(clicked_native_col),
        clicked_native_row=float(clicked_native_row),
        config=config,
        native_sim_to_display_coords=native_sim_to_display_coords,
        simulate_ideal_hkl_native_center=simulate_ideal_hkl_native_center,
    )

    prefix = f"Nearest peak (Δ={best_d2**0.5:.1f}px)"
    if selected_display is not None:
        snapped_dist = float(
            np.hypot(
                float(selected_display[0]) - float(click_col),
                float(selected_display[1]) - float(click_row),
            )
        )
        prefix = f"Ideal HKL center (click Δ={snapped_dist:.1f}px)"

    picked = bool(
        select_peak_by_index(
            best_i,
            prefix=prefix,
            sync_hkl_vars=True,
            clicked_display=(float(click_col), float(click_row)),
            clicked_native=(float(clicked_native_col), float(clicked_native_row)),
            selected_display=selected_display,
            selected_native=selected_native,
        )
    )
    if picked:
        set_pick_mode(False)
        peak_selection_state.suppress_drag_press_once = True
        sync_peak_selection_state()
    return picked


def open_selected_peak_intersection_figure(
    simulation_runtime_state,
    *,
    config: SelectedPeakIntersectionConfig,
    n2: Any,
    set_status_text: Any,
    geometry_factory: Any = IntersectionGeometry,
    beam_factory: Any = IntersectionBeamSamples,
    mosaic_factory: Any = IntersectionMosaicParams,
    analyze_intersection: Any = analyze_reflection_intersection,
    plot_intersection: Any = plot_intersection_analysis,
) -> bool:
    """Open a Bragg/Ewald intersection analysis plot for the selected peak."""

    selected_peak = simulation_runtime_state.selected_peak_record
    if not isinstance(selected_peak, Mapping):
        set_status_text(
            "Select a Bragg peak first (arm Pick HKL on Image or use HKL controls)."
        )
        return False

    try:
        h, k, l = tuple(int(v) for v in selected_peak["hkl"])
        native_col = float(
            selected_peak.get(
                "selected_native_col",
                selected_peak.get("native_col"),
            )
        )
        native_row = float(
            selected_peak.get(
                "selected_native_row",
                selected_peak.get("native_row"),
            )
        )
        lattice_a = float(selected_peak["av"])
        lattice_c = float(selected_peak["cv"])

        geometry = geometry_factory(
            image_size=int(config.image_size),
            center_col=float(config.center_col),
            center_row=float(config.center_row),
            distance_cor_to_detector=float(config.distance_cor_to_detector),
            gamma_deg=float(config.gamma_deg),
            Gamma_deg=float(config.Gamma_deg),
            chi_deg=float(config.chi_deg),
            psi_deg=float(config.psi_deg),
            psi_z_deg=float(config.psi_z_deg),
            zs=float(config.zs),
            zb=float(config.zb),
            theta_initial_deg=float(config.theta_initial_deg),
            cor_angle_deg=float(config.cor_angle_deg),
            n_detector=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            unit_x=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            pixel_size_m=float(config.pixel_size_m),
        )

        profile_cache = simulation_runtime_state.profile_cache
        beam = beam_factory(
            beam_x_array=np.asarray(profile_cache["beam_x_array"], dtype=np.float64),
            beam_y_array=np.asarray(profile_cache["beam_y_array"], dtype=np.float64),
            theta_array=np.asarray(profile_cache["theta_array"], dtype=np.float64),
            phi_array=np.asarray(profile_cache["phi_array"], dtype=np.float64),
            wavelength_array=np.asarray(
                profile_cache["wavelength_array"],
                dtype=np.float64,
            ),
        )
        mosaic = mosaic_factory(
            sigma_mosaic_deg=float(config.sigma_mosaic_deg),
            gamma_mosaic_deg=float(config.gamma_mosaic_deg),
            eta=float(config.eta),
            solve_q_steps=int(config.solve_q_steps),
            solve_q_rel_tol=float(config.solve_q_rel_tol),
            solve_q_mode=int(config.solve_q_mode),
        )

        analysis = analyze_intersection(
            h=h,
            k=k,
            l=l,
            lattice_a=lattice_a,
            lattice_c=lattice_c,
            selected_native_col=native_col,
            selected_native_row=native_row,
            geometry=geometry,
            beam=beam,
            mosaic=mosaic,
            n2=n2,
        )
        fig_analysis = plot_intersection(analysis)
        manager = getattr(getattr(fig_analysis, "canvas", None), "manager", None)
        if manager is not None:
            manager.set_window_title(f"Bragg/Ewald HKL=({h},{k},{l})")
            manager.show()
        else:
            fig_analysis.show()

        set_status_text(
            f"Opened Bragg/Ewald analysis for HKL=({h} {k} {l}) "
            f"from source={selected_peak.get('source_label', 'unknown')}."
        )
        return True
    except Exception as exc:
        set_status_text(f"Intersection analysis failed for selected peak: {exc}")
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
    if message:
        _set_status_text(bindings.set_status_text, message)


def toggle_runtime_hkl_pick_mode(bindings: SelectedPeakRuntimeBindings) -> None:
    """Toggle runtime HKL image-pick mode using the current GUI state."""

    toggle_hkl_pick_mode(
        bindings.simulation_runtime_state,
        bindings.peak_selection_state,
        caked_view_enabled=bool(
            _resolve_runtime_value(bindings.caked_view_enabled_factory)
        ),
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
        sync_hkl_vars=sync_hkl_vars,
        silent_if_missing=silent_if_missing,
    )


def reselect_runtime_selected_peak(bindings: SelectedPeakRuntimeBindings) -> bool:
    """Refresh the currently selected HKL target after a simulation update."""

    target = getattr(bindings.peak_selection_state, "selected_hkl_target", None)
    if target is None or len(target) < 3:
        return False
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
        tcl_error_types=tuple(bindings.tcl_error_types or ()),
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
    if not (
        callable(bindings.display_to_native_sim_coords)
        and callable(bindings.native_sim_to_display_coords)
        and callable(bindings.simulate_ideal_hkl_native_center)
    ):
        return False
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
    native_sim_to_display_coords: Callable[..., tuple[float, float]] | None = None,
    simulate_ideal_hkl_native_center: Callable[..., tuple[float, float] | None] | None = None,
    deactivate_conflicting_modes_factory: object = None,
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
            native_sim_to_display_coords=native_sim_to_display_coords,
            simulate_ideal_hkl_native_center=simulate_ideal_hkl_native_center,
            deactivate_conflicting_modes=_resolve_runtime_value(
                deactivate_conflicting_modes_factory
            ),
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
    )
