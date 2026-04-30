#!/usr/bin/env python
"""Audit New4 manual caked Qr fit coordinates as JSON plus scatter PNG."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ra_sim.fitting import optimization as opt  # noqa: E402
from scripts.debug import run_new4_caked_point_reprojection_check as reprojection  # noqa: E402
from scripts.debug import run_new4_geometry_fit_ladder as ladder  # noqa: E402
from scripts.debug import validate_geometry_preflight_rebind as preflight  # noqa: E402


DEFAULT_STATE_PATH = REPO_ROOT / "artifacts" / "geometry_fit_gui_states" / "new4.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "geometry_fit_ladder" / "new4_coordinate_audit"
REPORT_NAME = "new4_qr_fit_coordinates.json"
PLOT_NAME = "new4_qr_fit_coordinates.png"
EXACT_TOL_DEG = 1.0e-6


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_pair(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) < 2:
        return None
    try:
        x = float(value[0])
        y = float(value[1])
    except Exception:
        return None
    if math.isfinite(x) and math.isfinite(y):
        return float(x), float(y)
    return None


def _cached_target(entry: Mapping[str, object]) -> tuple[float, float] | None:
    for x_key, y_key in (
        ("background_two_theta_deg", "background_phi_deg"),
        ("caked_x", "caked_y"),
        ("raw_caked_x", "raw_caked_y"),
    ):
        try:
            point = (float(entry.get(x_key, np.nan)), float(entry.get(y_key, np.nan)))
        except Exception:
            continue
        if math.isfinite(point[0]) and math.isfinite(point[1]):
            return point
    return None


def _delta(source: tuple[float, float] | None, target: tuple[float, float] | None) -> dict[str, object]:
    if source is None or target is None:
        return {
            "delta_two_theta": None,
            "delta_phi_wrapped": None,
            "norm": None,
            "within_exact_tolerance": False,
        }
    dt = float(source[0] - target[0])
    dp = float(opt._angular_difference_deg(float(source[1]), float(target[1])))
    return {
        "delta_two_theta": dt,
        "delta_phi_wrapped": dp,
        "norm": float(math.hypot(dt, dp)),
        "within_exact_tolerance": bool(abs(dt) <= EXACT_TOL_DEG and abs(dp) <= EXACT_TOL_DEG),
    }


def _center_from_params(params: Mapping[str, object]) -> list[float]:
    center = params.get("center")
    if isinstance(center, Sequence) and len(center) >= 2:
        try:
            return [float(center[0]), float(center[1])]
        except Exception:
            pass
    return [
        float(params.get("center_x", np.nan)),
        float(params.get("center_y", np.nan)),
    ]


def _apply_perturb(params: Mapping[str, object], perturb: str | None) -> tuple[dict[str, object], dict[str, object]]:
    out = dict(params)
    if not perturb:
        return out, {"applied": False}
    if "=" not in str(perturb):
        raise ValueError("--perturb must be NAME=DELTA")
    name, raw_delta = str(perturb).split("=", 1)
    name = name.strip()
    delta = float(raw_delta)
    base = float(out.get(name, 0.0) or 0.0)
    out[name] = float(base + delta)
    if name in {"center_x", "center_y"}:
        center = _center_from_params(out)
        if name == "center_x":
            center[0] = float(out[name])
        else:
            center[1] = float(out[name])
        out["center"] = center
    return out, {"applied": True, "name": name, "delta": float(delta), "base": base, "value": float(out[name])}


def _build_dataset_context(request, params: Mapping[str, object]):
    contexts = opt._build_geometry_fit_dataset_contexts(
        np.asarray(request.miller, dtype=np.float64),
        np.asarray(request.intensities, dtype=np.float64),
        dict(params),
        request.measured_peaks,
        None,
        request.dataset_specs,
    )
    if not contexts:
        raise RuntimeError("geometry fit dataset context unavailable")
    return contexts[0]


def _evaluate_pairs(request, params: Mapping[str, object]) -> list[dict[str, object]]:
    dataset_ctx = _build_dataset_context(request, params)
    pixel_size = float(opt._fit_space_pixel_size_provenance(params).get("value", np.nan))
    detector_distance = float(params.get("corto_detector", np.nan))
    gamma_deg = float(params.get("gamma", 0.0) or 0.0)
    Gamma_deg = float(params.get("Gamma", 0.0) or 0.0)
    image_size = int(request.image_size)
    sim_buffer = np.zeros((image_size, image_size), dtype=np.float64)
    rows: list[dict[str, object]] = []
    for index, entry in enumerate(dataset_ctx.subset.measured_entries):
        if not isinstance(entry, Mapping):
            continue
        cached = _cached_target(entry)
        measured_anchor, measured_reason, measured_meta = opt._measured_fit_space_anchor(
            entry,
            center=_center_from_params(params),
            detector_distance=detector_distance,
            pixel_size=pixel_size,
            gamma_deg=gamma_deg,
            Gamma_deg=Gamma_deg,
            a_lattice=float(params.get("a", np.nan)),
            c_lattice=float(params.get("c", np.nan)),
            wavelength=float(params.get("lambda", np.nan)),
            dataset_ctx=dataset_ctx,
            local_params=params,
        )
        prediction = opt._resolve_qr_fit_prediction_from_trial_params(
            entry,
            params,
            {
                "dataset_ctx": dataset_ctx,
                "hit_tables": (),
                "sim_buffer": sim_buffer,
                "image_size": image_size,
                "fit_center": _center_from_params(params),
                "detector_distance": detector_distance,
                "pixel_size": pixel_size,
                "gamma_deg": gamma_deg,
                "Gamma_deg": Gamma_deg,
                "prediction_source_rows_cache": {},
                "_qr_fit_point_only_projection": True,
            },
            entry,
        )
        dynamic_source = _finite_pair(prediction.get("dynamic_baseline_anchor_caked_deg"))
        optimizer_source = _finite_pair(prediction.get("sim_refined_caked_deg"))
        target_delta = _delta(measured_anchor, cached)
        source_delta = _delta(optimizer_source, dynamic_source)
        residual_delta = _delta(optimizer_source, cached)
        rows.append(
            {
                "pair_index": int(index),
                "pair_id": reprojection._pair_id(entry, index),
                "q_group_key": entry.get("q_group_key"),
                "HKL": entry.get("hkl", entry.get("normalized_hkl")),
                "physical_branch_slot": entry.get("physical_branch_slot", entry.get("source_branch_index")),
                "fit_qr_branch_key": entry.get("fit_qr_branch_key"),
                "manual_visual_point": [float(cached[0]), float(cached[1])] if cached else None,
                "cached_click_candidate_two_theta_phi": [float(cached[0]), float(cached[1])] if cached else None,
                "optimizer_measured_anchor_two_theta_phi": (
                    [float(measured_anchor[0]), float(measured_anchor[1])] if measured_anchor else None
                ),
                "dynamic_sim_visual_caked_deg_two_theta_phi": (
                    [float(dynamic_source[0]), float(dynamic_source[1])] if dynamic_source else None
                ),
                "optimizer_simulated_source_two_theta_phi": (
                    [float(optimizer_source[0]), float(optimizer_source[1])] if optimizer_source else None
                ),
                "target_minus_cached_delta": target_delta,
                "source_minus_dynamic_delta": source_delta,
                "residual_delta_two_theta": residual_delta["delta_two_theta"],
                "residual_delta_phi_wrapped": residual_delta["delta_phi_wrapped"],
                "measured_anchor_source": measured_reason,
                "measured_fit_space_source": measured_meta.get("fit_space_source"),
                "prediction_available": bool(prediction.get("available", False)),
                "prediction_unavailable_reason": prediction.get("unavailable_reason"),
                "sim_refinement_status": prediction.get("sim_refinement_status"),
            }
        )
    return rows


def _pair_checks(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    measured_ok = all(
        isinstance(row.get("target_minus_cached_delta"), Mapping)
        and bool(row["target_minus_cached_delta"].get("within_exact_tolerance"))
        and str(row.get("measured_fit_space_source")) == "cached_fit_space_anchor"
        for row in rows
    )
    source_ok = all(
        isinstance(row.get("source_minus_dynamic_delta"), Mapping)
        and bool(row["source_minus_dynamic_delta"].get("within_exact_tolerance"))
        for row in rows
    )
    residual_ok = all(
        row.get("residual_delta_two_theta") is not None
        and row.get("residual_delta_phi_wrapped") is not None
        for row in rows
    )
    return {
        "optimizer_measured_target_equals_cached_target": bool(measured_ok),
        "optimizer_source_equals_dynamic_source": bool(source_ok),
        "residual_vector_machine_checkable": bool(residual_ok),
    }


def _cross_checks(base_rows: Sequence[Mapping[str, object]], rows: Sequence[Mapping[str, object]], perturb_applied: bool) -> dict[str, object]:
    target_fixed: list[bool] = []
    source_shift_norms: list[float] = []
    for base, row in zip(base_rows, rows):
        base_target = _finite_pair(base.get("optimizer_measured_anchor_two_theta_phi"))
        target = _finite_pair(row.get("optimizer_measured_anchor_two_theta_phi"))
        base_source = _finite_pair(base.get("optimizer_simulated_source_two_theta_phi"))
        source = _finite_pair(row.get("optimizer_simulated_source_two_theta_phi"))
        target_fixed.append(bool(_delta(target, base_target)["within_exact_tolerance"]))
        source_shift = _delta(source, base_source)
        norm = source_shift.get("norm")
        if isinstance(norm, (int, float)) and math.isfinite(float(norm)):
            source_shift_norms.append(float(norm))
    return {
        "target_unchanged_under_perturbation": bool(target_fixed and all(target_fixed)),
        "source_shift_norms": source_shift_norms,
        "source_moves_under_perturbation": (
            bool(any(value > EXACT_TOL_DEG for value in source_shift_norms))
            if perturb_applied
            else None
        ),
    }


def _plot_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    series = (
        ("cached target", "cached_click_candidate_two_theta_phi", "o"),
        ("optimizer measured", "optimizer_measured_anchor_two_theta_phi", "x"),
        ("dynamic source", "dynamic_sim_visual_caked_deg_two_theta_phi", "^"),
        ("optimizer source", "optimizer_simulated_source_two_theta_phi", "+"),
    )
    for label, key, marker in series:
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            point = _finite_pair(row.get(key))
            if point is not None:
                xs.append(point[0])
                ys.append(point[1])
        if xs:
            ax.scatter(xs, ys, label=label, marker=marker, s=55)
    for row in rows:
        target = _finite_pair(row.get("cached_click_candidate_two_theta_phi"))
        source = _finite_pair(row.get("optimizer_simulated_source_two_theta_phi"))
        if target is not None and source is not None:
            ax.annotate(
                "",
                xy=target,
                xytext=source,
                arrowprops={"arrowstyle": "->", "linewidth": 0.8, "alpha": 0.45},
            )
            ax.text(source[0], source[1], str(row.get("pair_index")), fontsize=8)
    ax.set_xlabel("two_theta_deg")
    ax.set_ylabel("phi_deg")
    ax.set_title("New4 Qr Fit Coordinate Audit")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_coordinate_audit(
    *,
    state_path: Path,
    background_index: int,
    output_root: Path,
    params_mode: str = "base",
    perturb: str | None = None,
) -> dict[str, object]:
    if str(params_mode).strip().lower() != "base":
        raise ValueError("only --params base is supported")
    state_path = state_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    provider_report = preflight._run_point_provider_report_only(state_path, int(background_index))
    context = ladder._capture_solver_context(state_path, int(background_index))
    request = ladder.build_solver_request(context, ["center_x"], max_nfev=1)
    prepared_run = context.get("prepared_run")
    prepared_params = (
        getattr(prepared_run, "fit_params", None) if prepared_run is not None else None
    )
    base_params = dict(prepared_params or request.params)
    current_params, perturb_info = _apply_perturb(base_params, perturb)
    base_rows = _evaluate_pairs(request, base_params)
    rows = _evaluate_pairs(request, current_params)
    checks = _pair_checks(rows)
    checks.update(_cross_checks(base_rows, rows, bool(perturb_info.get("applied", False))))
    status = "pass" if all(
        bool(value)
        for key, value in checks.items()
        if key != "source_moves_under_perturbation" or bool(perturb_info.get("applied", False))
    ) else "fail"
    if not bool(perturb_info.get("applied", False)) and checks.get("source_moves_under_perturbation") is None:
        status = "pass" if all(
            bool(value)
            for key, value in checks.items()
            if key != "source_moves_under_perturbation"
        ) else "fail"
    report_path = output_root / REPORT_NAME
    plot_path = output_root / PLOT_NAME
    report = {
        "status": status,
        "checks": checks,
        "state_path": str(state_path),
        "background_index": int(background_index),
        "params_mode": "base",
        "perturb": perturb_info,
        "provider_pair_count": reprojection._provider_pair_count(provider_report),
        "pair_count": int(len(rows)),
        "exact_tolerance_deg": EXACT_TOL_DEG,
        "base_pairs": base_rows,
        "pairs": rows,
        "json_path": str(report_path),
        "png_path": str(plot_path),
        "created_at_unix": time.time(),
    }
    _write_json(report_path, report)
    _plot_rows(plot_path, rows)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize New4 Qr fit coordinate contract.")
    parser.add_argument("--state", "--state-path", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--background-index", type=int, default=0)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--params", default="base")
    parser.add_argument("--perturb", default=None)
    args = parser.parse_args(argv)
    report = run_coordinate_audit(
        state_path=Path(args.state),
        background_index=int(args.background_index),
        output_root=Path(args.output_root),
        params_mode=str(args.params),
        perturb=args.perturb,
    )
    print(json.dumps(_jsonable({"status": report["status"], "json_path": report["json_path"], "png_path": report["png_path"]}), sort_keys=True))
    return 0 if str(report.get("status")) == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
