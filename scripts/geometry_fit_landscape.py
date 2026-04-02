"""Sweep geometry-fit parameters around one saved GUI baseline."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any
import warnings

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy.stats import spearmanr

from ra_sim.fitting.optimization import (
    build_geometry_fit_central_mosaic_params,
    process_peaks_parallel,
)
from ra_sim.gui import background_theta as gui_background_theta
from ra_sim.gui import geometry_fit as gui_geometry_fit
from ra_sim.gui import geometry_q_group_manager as gui_geometry_q_group_manager
from ra_sim.gui import structure_model as gui_structure_model
from ra_sim.gui.state import SimulationRuntimeState
from ra_sim.headless_geometry_fit import (
    _build_runtime_defaults,
    _build_var_store,
    _coerce_float,
    _coerce_int,
    _load_structure_model,
    _resolve_optics_mode_flag,
    _resolve_solve_q_mode,
)
from ra_sim.simulation.diffraction import hit_tables_to_max_positions


DEFAULT_STATE_PATH = Path.home() / ".local" / "share" / "ra_sim" / "init.json"
SUMMARY_METRIC_NAMES = [
    "visible_peak_count",
    "total_peak_weight",
    "centroid_x_px",
    "centroid_y_px",
    "radius_gyration_px",
    "x_span_px",
    "y_span_px",
    "anisotropy_ratio",
    "runtime_s",
]
PANEL_METRIC_NAMES = [
    "visible_peak_count",
    "centroid_shift_px",
    "radius_gyration_px",
    "anisotropy_ratio",
]
PANEL_METRIC_COLORS = {
    "visible_peak_count": "#0b6e4f",
    "centroid_shift_px": "#bc4b51",
    "radius_gyration_px": "#1f5aa6",
    "anisotropy_ratio": "#9c6644",
}
DEFAULT_WORKER_FRACTION = 0.9


@dataclass(frozen=True)
class SweepSpec:
    """Resolved one-at-a-time sweep for one geometry-fit parameter."""

    name: str
    baseline: float
    min_value: float
    max_value: float
    values: np.ndarray
    source: str


@dataclass(frozen=True)
class LandscapeContext:
    """Resolved saved-state runtime needed for the geometry landscape."""

    state_path: Path
    saved_state: dict[str, object]
    defaults: Any
    structure_state: Any
    baseline_params: dict[str, object]
    candidate_param_names: list[str]
    fit_geometry_config: dict[str, object]
    solve_q_steps: int
    solve_q_rel_tol: float
    solve_q_mode: int
    theta_base_current: float
    use_shared_theta_offset: bool
    theta_param_name: str
    active_cif_path: str


@dataclass(frozen=True)
class SweepTask:
    """One concrete sweep-point simulation request."""

    parameter: str
    parameter_value: float
    baseline_value: float
    sweep_min: float
    sweep_max: float
    sweep_source: str
    sweep_index: int


_WORKER_CONTEXT: LandscapeContext | None = None


def _positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _default_worker_count() -> int:
    """Return floor(90% of available CPU threads), with a minimum of one worker."""

    cpu_count = os.cpu_count() or 1
    return max(int(math.floor(float(cpu_count) * DEFAULT_WORKER_FRACTION)), 1)


def _load_saved_state(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        isinstance(payload, dict)
        and payload.get("type") == "ra_sim.gui_state"
        and isinstance(payload.get("state"), dict)
    ):
        return dict(payload["state"])
    if isinstance(payload, dict):
        return dict(payload)
    raise ValueError(f"Saved GUI state at {path} is not a dictionary payload.")


def resolve_candidate_parameter_names(
    *,
    use_shared_theta_offset: bool,
    lattice_refinement_enabled: bool,
) -> list[str]:
    """Return the full fit-capable geometry parameter set for the current mode."""

    return gui_geometry_fit.current_geometry_fit_var_names(
        fit_zb=True,
        fit_zs=True,
        fit_theta=True,
        fit_psi_z=True,
        fit_chi=True,
        fit_cor=True,
        fit_gamma=True,
        fit_Gamma=True,
        fit_dist=True,
        fit_a=bool(lattice_refinement_enabled),
        fit_c=bool(lattice_refinement_enabled),
        fit_center_x=True,
        fit_center_y=True,
        use_shared_theta_offset=bool(use_shared_theta_offset),
    )


def resolve_geometry_sweep_range(
    name: str,
    baseline_value: float,
    *,
    bounds_config: dict[str, object] | None,
    priors_config: dict[str, object] | None,
) -> tuple[float, float, str]:
    """Resolve the sweep limits for one geometry-fit parameter."""

    bounds_cfg = bounds_config if isinstance(bounds_config, dict) else {}
    priors_cfg = priors_config if isinstance(priors_config, dict) else {}

    if name in {"center_x", "center_y"}:
        prior_entry = priors_cfg.get(name, {})
        sigma = None
        if isinstance(prior_entry, dict):
            try:
                sigma_value = float(prior_entry.get("sigma"))
            except (TypeError, ValueError):
                sigma_value = float("nan")
            if np.isfinite(sigma_value) and sigma_value > 0.0:
                sigma = float(sigma_value)
        if sigma is None:
            return (
                float(baseline_value - 60.0),
                float(baseline_value + 60.0),
                "center_fallback_pm60",
            )
        span = float(3.0 * sigma)
        return (
            float(baseline_value - span),
            float(baseline_value + span),
            "center_prior_pm3sigma",
        )

    bound_entry = bounds_cfg.get(name, {})
    if not isinstance(bound_entry, dict):
        raise KeyError(f"No geometry sweep bounds configured for '{name}'.")

    mode = str(bound_entry.get("mode", "absolute")).strip().lower()
    try:
        min_value = float(bound_entry["min"])
        max_value = float(bound_entry["max"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid bounds for '{name}'.") from exc

    if mode == "absolute":
        lo = float(min_value)
        hi = float(max_value)
    elif mode == "relative":
        lo = float(baseline_value + min_value)
        hi = float(baseline_value + max_value)
    elif mode == "relative_min0":
        lo = float(max(0.0, baseline_value + min_value))
        hi = float(baseline_value + max_value)
    else:
        raise ValueError(f"Unsupported bounds mode '{mode}' for '{name}'.")

    if lo > hi:
        lo, hi = hi, lo
    return float(lo), float(hi), f"bounds:{mode}"


def build_sweep_specs(
    candidate_param_names: list[str],
    baseline_params: dict[str, object],
    *,
    fit_geometry_config: dict[str, object] | None,
    points: int,
) -> list[SweepSpec]:
    """Resolve one sweep specification per candidate geometry parameter."""

    geometry_cfg = fit_geometry_config if isinstance(fit_geometry_config, dict) else {}
    bounds_cfg = geometry_cfg.get("bounds", {})
    priors_cfg = geometry_cfg.get("priors", {})

    specs: list[SweepSpec] = []
    for name in candidate_param_names:
        baseline = float(baseline_params[name])
        lo, hi, source = resolve_geometry_sweep_range(
            name,
            baseline,
            bounds_config=bounds_cfg if isinstance(bounds_cfg, dict) else {},
            priors_config=priors_cfg if isinstance(priors_cfg, dict) else {},
        )
        if math.isclose(lo, hi, abs_tol=1.0e-12):
            values = np.full(points, lo, dtype=float)
        else:
            values = np.linspace(lo, hi, points, dtype=float)
        specs.append(
            SweepSpec(
                name=str(name),
                baseline=float(baseline),
                min_value=float(lo),
                max_value=float(hi),
                values=values,
                source=str(source),
            )
        )
    return specs


def _build_sweep_tasks(sweep_specs: list[SweepSpec]) -> list[SweepTask]:
    """Flatten one sweep task per parameter point in stable order."""

    tasks: list[SweepTask] = []
    for spec in sweep_specs:
        for sweep_index, value in enumerate(spec.values):
            tasks.append(
                SweepTask(
                    parameter=str(spec.name),
                    parameter_value=float(value),
                    baseline_value=float(spec.baseline),
                    sweep_min=float(spec.min_value),
                    sweep_max=float(spec.max_value),
                    sweep_source=str(spec.source),
                    sweep_index=int(sweep_index),
                )
            )
    return tasks


def _resolve_worker_count(requested_workers: int | None, task_count: int) -> int:
    """Clamp the requested process count against the concrete task count."""

    if task_count <= 0:
        return 1
    if requested_workers is None:
        requested = _default_worker_count()
    else:
        requested = int(requested_workers)
    return max(1, min(int(requested), int(task_count)))


def _numba_threads_per_worker(worker_count: int) -> int:
    """Return a conservative per-process Numba thread budget."""

    cpu_count = os.cpu_count() or 1
    if worker_count <= 1:
        return max(int(cpu_count), 1)
    return max(int(cpu_count // worker_count), 1)


def compute_peak_metrics(
    peaks: list[dict[str, object]] | tuple[dict[str, object], ...] | None,
) -> dict[str, float]:
    """Return simulation-summary metrics for one geometry-fit peak set."""

    entries = list(peaks or [])
    if not entries:
        return {
            "visible_peak_count": 0.0,
            "total_peak_weight": 0.0,
            "centroid_x_px": 0.0,
            "centroid_y_px": 0.0,
            "radius_gyration_px": 0.0,
            "x_span_px": 0.0,
            "y_span_px": 0.0,
            "anisotropy_ratio": 1.0,
        }

    cols: list[float] = []
    rows: list[float] = []
    weights: list[float] = []
    for entry in entries:
        try:
            col = float(entry.get("sim_col", np.nan))
            row = float(entry.get("sim_row", np.nan))
            weight = float(entry.get("weight", 0.0))
        except Exception:
            continue
        if not (np.isfinite(col) and np.isfinite(row)):
            continue
        cols.append(col)
        rows.append(row)
        weights.append(max(0.0, weight))

    if not cols:
        return {
            "visible_peak_count": 0.0,
            "total_peak_weight": 0.0,
            "centroid_x_px": 0.0,
            "centroid_y_px": 0.0,
            "radius_gyration_px": 0.0,
            "x_span_px": 0.0,
            "y_span_px": 0.0,
            "anisotropy_ratio": 1.0,
        }

    col_arr = np.asarray(cols, dtype=float)
    row_arr = np.asarray(rows, dtype=float)
    weight_arr = np.asarray(weights, dtype=float)
    total_peak_weight = float(weight_arr.sum())
    metric_weights = (
        weight_arr
        if np.isfinite(total_peak_weight) and total_peak_weight > 0.0
        else np.ones_like(weight_arr, dtype=float)
    )
    metric_weight_sum = float(metric_weights.sum())
    if metric_weight_sum <= 0.0:
        metric_weights = np.ones_like(weight_arr, dtype=float)
        metric_weight_sum = float(metric_weights.sum())

    centroid_x = float(np.dot(col_arr, metric_weights) / metric_weight_sum)
    centroid_y = float(np.dot(row_arr, metric_weights) / metric_weight_sum)
    delta_x = col_arr - centroid_x
    delta_y = row_arr - centroid_y
    radius_gyration = float(
        math.sqrt(
            np.dot(metric_weights, delta_x * delta_x + delta_y * delta_y)
            / metric_weight_sum
        )
    )
    x_span = float(col_arr.max() - col_arr.min()) if col_arr.size else 0.0
    y_span = float(row_arr.max() - row_arr.min()) if row_arr.size else 0.0

    if col_arr.size <= 1:
        anisotropy_ratio = 1.0
    else:
        cov_xx = float(np.dot(metric_weights, delta_x * delta_x) / metric_weight_sum)
        cov_yy = float(np.dot(metric_weights, delta_y * delta_y) / metric_weight_sum)
        cov_xy = float(np.dot(metric_weights, delta_x * delta_y) / metric_weight_sum)
        eigvals = np.linalg.eigvalsh(
            np.asarray([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)
        )
        lambda_min = max(float(eigvals[0]), 0.0)
        lambda_max = max(float(eigvals[1]), 0.0)
        if lambda_max <= 0.0:
            anisotropy_ratio = 1.0
        else:
            anisotropy_ratio = float(
                math.sqrt(lambda_max / max(lambda_min, 1.0e-12))
            )

    return {
        "visible_peak_count": float(col_arr.size),
        "total_peak_weight": float(total_peak_weight if np.isfinite(total_peak_weight) else 0.0),
        "centroid_x_px": float(centroid_x),
        "centroid_y_px": float(centroid_y),
        "radius_gyration_px": float(radius_gyration),
        "x_span_px": float(x_span),
        "y_span_px": float(y_span),
        "anisotropy_ratio": float(anisotropy_ratio),
    }


def _safe_spearman(values_x: np.ndarray, values_y: np.ndarray) -> float:
    """Return one finite signed Spearman correlation coefficient."""

    if values_x.size < 2 or values_y.size < 2:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr = spearmanr(values_x, values_y)
    rho = getattr(corr, "statistic", corr[0] if isinstance(corr, tuple) else np.nan)
    try:
        rho_value = float(rho)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(rho_value):
        return 0.0
    return float(np.clip(rho_value, -1.0, 1.0))


def build_correlation_matrix(
    rows: list[dict[str, object]],
    sweep_specs: list[SweepSpec],
    *,
    metric_names: list[str],
) -> np.ndarray:
    """Return one parameter-by-metric signed Spearman heatmap matrix."""

    matrix = np.zeros((len(sweep_specs), len(metric_names)), dtype=float)
    for row_index, spec in enumerate(sweep_specs):
        spec_rows = [row for row in rows if row.get("parameter") == spec.name]
        x_values = np.asarray(
            [float(row["parameter_value"]) for row in spec_rows],
            dtype=float,
        )
        for col_index, metric_name in enumerate(metric_names):
            y_values = np.asarray(
                [float(row[metric_name]) for row in spec_rows],
                dtype=float,
            )
            matrix[row_index, col_index] = _safe_spearman(x_values, y_values)
    return matrix


def _zscore(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if not np.isfinite(std) or std <= 0.0:
        return np.zeros_like(arr, dtype=float)
    return (arr - mean) / std


def _update_param_set_for_sweep(
    baseline_params: dict[str, object],
    *,
    param_name: str,
    value: float,
    theta_base_current: float,
) -> dict[str, object]:
    updated = copy.deepcopy(baseline_params)
    value_float = float(value)

    if param_name == "center_x":
        center_seed = updated.get("center", [updated.get("center_x", 0.0), updated.get("center_y", 0.0)])
        center_y = float(updated.get("center_y", center_seed[1]))
        updated["center_x"] = value_float
        updated["center"] = [value_float, center_y]
        return updated
    if param_name == "center_y":
        center_seed = updated.get("center", [updated.get("center_x", 0.0), updated.get("center_y", 0.0)])
        center_x = float(updated.get("center_x", center_seed[0]))
        updated["center_y"] = value_float
        updated["center"] = [center_x, value_float]
        return updated
    if param_name == "theta_offset":
        updated["theta_offset"] = value_float
        updated["theta_initial"] = float(theta_base_current + value_float)
        return updated

    updated[param_name] = value_float
    return updated


def _normalize_atom_site_fractional_rows(values: object) -> list[tuple[float, float, float]]:
    normalized: list[tuple[float, float, float]] = []
    if not isinstance(values, (list, tuple, np.ndarray)):
        return normalized
    for row in values:
        if isinstance(row, dict):
            raw_triplet = (row.get("x"), row.get("y"), row.get("z"))
        elif isinstance(row, (list, tuple, np.ndarray)) and len(row) >= 3:
            raw_triplet = (row[0], row[1], row[2])
        else:
            continue
        try:
            normalized.append(
                (
                    float(raw_triplet[0]),
                    float(raw_triplet[1]),
                    float(raw_triplet[2]),
                )
            )
        except (TypeError, ValueError):
            continue
    return normalized


@contextmanager
def _patched_atom_site_fractional_helpers():
    """Normalize saved atom-site rows for legacy headless structure rebuilds."""

    original_signature = gui_structure_model.atom_site_fractional_signature
    original_values_are_default = gui_structure_model.atom_site_fractional_values_are_default
    original_active_primary_cif_path = gui_structure_model.active_primary_cif_path

    def _signature(values):
        return original_signature(_normalize_atom_site_fractional_rows(values))

    def _values_are_default(state, values):
        return original_values_are_default(
            state,
            _normalize_atom_site_fractional_rows(values),
        )

    def _active_primary_cif_path(state, atom_site_override_state, *, atom_site_values=None, **kwargs):
        normalized_values = (
            _normalize_atom_site_fractional_rows(atom_site_values)
            if atom_site_values is not None
            else None
        )
        return original_active_primary_cif_path(
            state,
            atom_site_override_state,
            atom_site_values=normalized_values,
            **kwargs,
        )

    gui_structure_model.atom_site_fractional_signature = _signature
    gui_structure_model.atom_site_fractional_values_are_default = _values_are_default
    gui_structure_model.active_primary_cif_path = _active_primary_cif_path
    try:
        yield
    finally:
        gui_structure_model.atom_site_fractional_signature = original_signature
        gui_structure_model.atom_site_fractional_values_are_default = original_values_are_default
        gui_structure_model.active_primary_cif_path = original_active_primary_cif_path


def _build_geometry_fit_value_state(
    saved_state: dict[str, object],
    state_path: Path,
) -> LandscapeContext:
    defaults = _build_runtime_defaults(saved_state)
    var_store = _build_var_store(saved_state, defaults)
    simulation_runtime_state = SimulationRuntimeState()
    with _patched_atom_site_fractional_helpers():
        structure_state, _atom_state, active_cif_path, nominal_n2 = _load_structure_model(
            defaults,
            saved_state,
            var_store,
            simulation_runtime_state,
        )

    theta_defaults = {"theta_initial": defaults.defaults["theta_initial"]}
    selection_var = var_store["geometry_fit_background_selection_var"]
    theta_initial_var = var_store["theta_initial_var"]
    background_theta_list_var = var_store["background_theta_list_var"]
    geometry_theta_offset_var = var_store["geometry_theta_offset_var"]

    def _use_shared_theta_offset(selected_indices: list[int] | None = None) -> bool:
        return gui_background_theta.geometry_fit_uses_shared_theta_offset(
            selected_indices,
            osc_files=defaults.osc_files,
            current_background_index=defaults.current_background_index,
            geometry_fit_background_selection_var=selection_var,
        )

    def _current_geometry_theta_offset(*, strict: bool = False) -> float:
        return gui_background_theta.current_geometry_theta_offset(
            geometry_theta_offset_var=geometry_theta_offset_var,
            strict=strict,
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
            geometry_fit_background_selection_var=selection_var,
            current_background_index=defaults.current_background_index,
            strict_count=strict_count,
        )

    theta_base_current = gui_background_theta.background_theta_base_for_index(
        defaults.current_background_index,
        osc_files=defaults.osc_files,
        theta_initial_var=theta_initial_var,
        defaults=theta_defaults,
        theta_initial=defaults.defaults["theta_initial"],
        background_theta_list_var=background_theta_list_var,
        strict_count=False,
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
    solve_q_mode = int(_resolve_solve_q_mode(var_store["solve_q_mode_var"].get()))
    nominal_lambda = float(defaults.lambda_angstrom)
    mosaic_params = {
        "beam_x_array": np.zeros(1, dtype=np.float64),
        "beam_y_array": np.zeros(1, dtype=np.float64),
        "theta_array": np.zeros(1, dtype=np.float64),
        "phi_array": np.zeros(1, dtype=np.float64),
        "wavelength_array": np.array([nominal_lambda], dtype=np.float64),
        "wavelength_i_array": np.array([nominal_lambda], dtype=np.float64),
        "n2_sample_array": np.array([nominal_n2], dtype=np.complex128),
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
        "solve_q_mode": solve_q_mode,
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
            current_background_index=lambda: defaults.current_background_index,
            geometry_fit_uses_shared_theta_offset=_use_shared_theta_offset,
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
    baseline_params = copy.deepcopy(value_callbacks.current_params())
    fit_geometry_config = (
        dict(defaults.fit_config.get("geometry", {}))
        if isinstance(defaults.fit_config, dict)
        and isinstance(defaults.fit_config.get("geometry", {}), dict)
        else {}
    )
    lattice_refinement_cfg = fit_geometry_config.get("lattice_refinement", {})
    lattice_refinement_enabled = bool(
        lattice_refinement_cfg.get("enabled", False)
        if isinstance(lattice_refinement_cfg, dict)
        else False
    )
    use_shared_theta_offset = bool(_use_shared_theta_offset())
    candidate_param_names = resolve_candidate_parameter_names(
        use_shared_theta_offset=use_shared_theta_offset,
        lattice_refinement_enabled=lattice_refinement_enabled,
    )
    return LandscapeContext(
        state_path=state_path,
        saved_state=saved_state,
        defaults=defaults,
        structure_state=structure_state,
        baseline_params=baseline_params,
        candidate_param_names=candidate_param_names,
        fit_geometry_config=fit_geometry_config,
        solve_q_steps=solve_q_steps,
        solve_q_rel_tol=solve_q_rel_tol,
        solve_q_mode=solve_q_mode,
        theta_base_current=float(theta_base_current),
        use_shared_theta_offset=use_shared_theta_offset,
        theta_param_name="theta_offset" if use_shared_theta_offset else "theta_initial",
        active_cif_path=str(active_cif_path),
    )


def simulate_geometry_fit_metrics(
    context: LandscapeContext,
    param_set: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Run one deterministic geometry-fit preview simulation and score it."""

    peaks = gui_geometry_q_group_manager.simulate_geometry_fit_peak_centers(
        np.asarray(context.structure_state.miller, dtype=float),
        np.asarray(context.structure_state.intensities, dtype=float),
        int(context.defaults.image_size),
        param_set,
        build_geometry_fit_central_mosaic_params=build_geometry_fit_central_mosaic_params,
        process_peaks_parallel=process_peaks_parallel,
        hit_tables_to_max_positions=hit_tables_to_max_positions,
        default_solve_q_steps=int(context.solve_q_steps),
        default_solve_q_rel_tol=float(context.solve_q_rel_tol),
        default_solve_q_mode=int(context.solve_q_mode),
    )
    return peaks, compute_peak_metrics(peaks)


def _build_sweep_row(
    task: SweepTask,
    metrics: dict[str, float],
    *,
    baseline_metrics: dict[str, float],
    runtime_s: float,
) -> dict[str, object]:
    centroid_shift_px = float(
        math.hypot(
            float(metrics["centroid_x_px"]) - float(baseline_metrics["centroid_x_px"]),
            float(metrics["centroid_y_px"]) - float(baseline_metrics["centroid_y_px"]),
        )
    )
    row = {
        "parameter": str(task.parameter),
        "parameter_value": float(task.parameter_value),
        "baseline_value": float(task.baseline_value),
        "sweep_min": float(task.sweep_min),
        "sweep_max": float(task.sweep_max),
        "sweep_source": str(task.sweep_source),
        "sweep_index": int(task.sweep_index),
        "runtime_s": float(runtime_s),
        "centroid_shift_px": float(centroid_shift_px),
    }
    row.update(metrics)
    return row


def _init_landscape_worker(
    saved_state: dict[str, object],
    state_path: str,
    numba_threads: int,
) -> None:
    """Build one reusable geometry-fit context inside a worker process."""

    global _WORKER_CONTEXT

    try:
        from numba import set_num_threads as _set_num_threads
    except Exception:
        _set_num_threads = None
    if callable(_set_num_threads):
        try:
            _set_num_threads(max(int(numba_threads), 1))
        except Exception:
            pass

    _WORKER_CONTEXT = _build_geometry_fit_value_state(
        copy.deepcopy(saved_state),
        Path(state_path),
    )


def _run_landscape_task(
    task: SweepTask,
    baseline_metrics: dict[str, float],
) -> dict[str, object]:
    """Execute one sweep task against the process-local geometry context."""

    if _WORKER_CONTEXT is None:
        raise RuntimeError("Landscape worker context is not initialized.")
    param_set = _update_param_set_for_sweep(
        _WORKER_CONTEXT.baseline_params,
        param_name=task.parameter,
        value=float(task.parameter_value),
        theta_base_current=float(_WORKER_CONTEXT.theta_base_current),
    )
    started = time.perf_counter()
    try:
        _peaks, metrics = simulate_geometry_fit_metrics(_WORKER_CONTEXT, param_set)
    except Exception as exc:
        raise RuntimeError(
            f"Geometry sweep failed for '{task.parameter}' at value {float(task.parameter_value):.8g}."
        ) from exc
    runtime_s = float(time.perf_counter() - started)
    return _build_sweep_row(
        task,
        metrics,
        baseline_metrics=baseline_metrics,
        runtime_s=runtime_s,
    )


def run_landscape_sweeps(
    context: LandscapeContext,
    sweep_specs: list[SweepSpec],
    *,
    workers: int | None = None,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Run all one-at-a-time geometry sweeps and return recorded rows."""

    _, baseline_metrics = simulate_geometry_fit_metrics(context, context.baseline_params)
    tasks = _build_sweep_tasks(sweep_specs)
    if not tasks:
        return [], baseline_metrics

    worker_count = _resolve_worker_count(int(workers), len(tasks))
    if worker_count <= 1:
        global _WORKER_CONTEXT
        previous_context = _WORKER_CONTEXT
        _WORKER_CONTEXT = context
        try:
            rows = [_run_landscape_task(task, baseline_metrics) for task in tasks]
        finally:
            _WORKER_CONTEXT = previous_context
        return rows, baseline_metrics

    numba_threads = _numba_threads_per_worker(worker_count)
    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_landscape_worker,
        initargs=(
            copy.deepcopy(context.saved_state),
            str(context.state_path),
            int(numba_threads),
        ),
    ) as executor:
        rows = list(
            executor.map(
                _run_landscape_task,
                tasks,
                [baseline_metrics] * len(tasks),
            )
        )
    return rows, baseline_metrics


def write_landscape_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write one CSV row per recorded sweep simulation."""

    if not rows:
        raise ValueError("No sweep rows were recorded.")
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value: object) -> object:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    return str(value)


def write_baseline_metadata(
    context: LandscapeContext,
    sweep_specs: list[SweepSpec],
    baseline_metrics: dict[str, float],
    output_path: Path,
) -> None:
    """Write resolved baseline + sweep metadata as JSON."""

    payload = {
        "state_path": str(context.state_path),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "active_cif_path": str(context.active_cif_path),
        "image_size": int(context.defaults.image_size),
        "theta_mode": str(context.theta_param_name),
        "use_shared_theta_offset": bool(context.use_shared_theta_offset),
        "candidate_param_names": list(context.candidate_param_names),
        "baseline_params": {
            name: float(context.baseline_params[name]) for name in context.candidate_param_names
        },
        "baseline_metrics": {
            name: float(value) for name, value in baseline_metrics.items()
        },
        "sweeps": [
            {
                "name": spec.name,
                "baseline": float(spec.baseline),
                "min_value": float(spec.min_value),
                "max_value": float(spec.max_value),
                "points": int(spec.values.size),
                "source": spec.source,
                "values": spec.values.tolist(),
            }
            for spec in sweep_specs
        ],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2)


def render_landscape_figure(
    rows: list[dict[str, object]],
    sweep_specs: list[SweepSpec],
    *,
    state_path: Path,
    output_path: Path,
    baseline_metrics: dict[str, float],
) -> None:
    """Render one omnibus correlation + sweep landscape figure."""

    small_panel_count = len(sweep_specs) + 1
    small_rows = int(math.ceil(small_panel_count / 3.0))
    fig = plt.figure(figsize=(18, 4.0 + 4.5 * small_rows))
    grid = gridspec.GridSpec(
        small_rows + 1,
        3,
        height_ratios=[1.1] + [1.0] * small_rows,
        hspace=0.55,
        wspace=0.30,
    )

    heatmap_ax = fig.add_subplot(grid[0, :])
    correlation_matrix = build_correlation_matrix(
        rows,
        sweep_specs,
        metric_names=SUMMARY_METRIC_NAMES,
    )
    image = heatmap_ax.imshow(
        correlation_matrix,
        cmap="coolwarm",
        aspect="auto",
        vmin=-1.0,
        vmax=1.0,
    )
    heatmap_ax.set_title("Geometry-Fit Sweep Correlations")
    heatmap_ax.set_xticks(np.arange(len(SUMMARY_METRIC_NAMES)))
    heatmap_ax.set_xticklabels(
        SUMMARY_METRIC_NAMES,
        rotation=30,
        ha="right",
        fontsize=9,
    )
    heatmap_ax.set_yticks(np.arange(len(sweep_specs)))
    heatmap_ax.set_yticklabels([spec.name for spec in sweep_specs], fontsize=9)
    colorbar = fig.colorbar(image, ax=heatmap_ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Spearman rho", rotation=90)

    panel_axes: list[plt.Axes] = []
    for row_index in range(1, small_rows + 1):
        for col_index in range(3):
            panel_axes.append(fig.add_subplot(grid[row_index, col_index]))

    for axis, spec in zip(panel_axes, sweep_specs):
        spec_rows = [row for row in rows if row.get("parameter") == spec.name]
        spec_rows.sort(key=lambda row: float(row["parameter_value"]))
        x_values = [float(row["parameter_value"]) for row in spec_rows]
        for metric_name in PANEL_METRIC_NAMES:
            y_values = [float(row[metric_name]) for row in spec_rows]
            axis.plot(
                x_values,
                _zscore(y_values),
                marker="o",
                linewidth=1.5,
                markersize=3.5,
                color=PANEL_METRIC_COLORS[metric_name],
                label=metric_name,
            )
        axis.axvline(float(spec.baseline), color="#333333", linestyle="--", linewidth=1.0)
        axis.set_title(spec.name, fontsize=10)
        axis.set_xlabel(spec.name, fontsize=9)
        axis.set_ylabel("z-score", fontsize=9)
        axis.grid(alpha=0.25, linewidth=0.5)
        axis.tick_params(labelsize=8)

    if sweep_specs and panel_axes:
        panel_axes[0].legend(loc="best", fontsize=8, frameon=False)

    summary_axis = panel_axes[len(sweep_specs)]
    summary_axis.axis("off")
    summary_lines = [
        "Summary",
        f"State: {state_path}",
        f"Points per sweep: {int(sweep_specs[0].values.size) if sweep_specs else 0}",
        f"Recorded runs: {len(rows)}",
        f"Parameters: {', '.join(spec.name for spec in sweep_specs)}",
        (
            "Baseline centroid: "
            f"({baseline_metrics['centroid_x_px']:.2f}, "
            f"{baseline_metrics['centroid_y_px']:.2f}) px"
        ),
        f"Baseline visible peaks: {baseline_metrics['visible_peak_count']:.0f}",
        f"Timestamp: {datetime.now().astimezone().isoformat(timespec='seconds')}",
    ]
    summary_axis.text(
        0.0,
        1.0,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    for axis in panel_axes[len(sweep_specs) + 1 :]:
        axis.axis("off")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _default_output_dir() -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return Path("artifacts") / f"geometry_fit_landscape_{stamp}"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic one-at-a-time sweeps over the geometry-fit "
            "parameter surface and render one omnibus landscape figure."
        )
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help=f"Saved GUI baseline state (default: {DEFAULT_STATE_PATH})",
    )
    parser.add_argument(
        "--points",
        type=_positive_int,
        default=7,
        help="Evenly spaced sweep points per parameter (default: 7).",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=_default_worker_count(),
        help=(
            "CPU worker processes for independent sweep points "
            f"(default: floor(0.9 * available cores) = {_default_worker_count()})."
        ),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for CSV, PNG, and metadata JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    state_path = Path(args.state).expanduser().resolve()
    if not state_path.exists():
        raise FileNotFoundError(f"Saved GUI state not found: {state_path}")

    outdir = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir is not None
        else _default_output_dir().resolve()
    )
    outdir.mkdir(parents=True, exist_ok=True)

    saved_state = _load_saved_state(state_path)
    context = _build_geometry_fit_value_state(saved_state, state_path)
    sweep_specs = build_sweep_specs(
        context.candidate_param_names,
        context.baseline_params,
        fit_geometry_config=context.fit_geometry_config,
        points=int(args.points),
    )
    rows, baseline_metrics = run_landscape_sweeps(
        context,
        sweep_specs,
        workers=int(args.workers),
    )

    csv_path = outdir / "landscape_runs.csv"
    figure_path = outdir / "landscape_figure.png"
    metadata_path = outdir / "baseline_metadata.json"
    write_landscape_csv(rows, csv_path)
    render_landscape_figure(
        rows,
        sweep_specs,
        state_path=state_path,
        output_path=figure_path,
        baseline_metrics=baseline_metrics,
    )
    write_baseline_metadata(context, sweep_specs, baseline_metrics, metadata_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {figure_path}")
    print(f"Wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
