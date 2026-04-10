"""Lightweight hBN bundle geometry helpers.

This module intentionally avoids importing the full calibrant fitting stack so
simulation startup paths can reuse bundle geometry metadata cheaply.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

# Simulator detector geometry is rotated 90 degrees clockwise from hBN-fitter
# native coordinates.
SIM_BACKGROUND_ROTATE_K = -1

_CANONICAL_TILT_CORRECTION_KIND = "to_flat"
_CANONICAL_TILT_MODEL = "RzRx"
_CANONICAL_TILT_FRAME = "simulation_background_display"
_CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_X = 1
_CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_Y = 1

__all__ = [
    "SIM_BACKGROUND_ROTATE_K",
    "build_hbn_geometry_debug_trace",
    "convert_hbn_bundle_geometry_to_simulation",
    "format_hbn_geometry_debug_trace",
    "load_bundle_npz",
    "load_tilt_hint",
    "resolve_hbn_paths",
]


def _rotate_point_by_k(col, row, image_size, k):
    """Rotate a point with the same semantics as ``np.rot90(..., k)``."""

    if isinstance(image_size, (tuple, list, np.ndarray)):
        if len(image_size) < 2:
            raise ValueError("image_size must provide at least (height, width).")
        height = int(image_size[0])
        width = int(image_size[1])
    else:
        height = int(image_size)
        width = int(image_size)
    if height <= 0 or width <= 0:
        raise ValueError("image_size must be positive.")

    col_new = float(col)
    row_new = float(row)
    for _ in range(int(k) % 4):
        row_new, col_new, height, width = (
            width - 1 - col_new,
            row_new,
            width,
            height,
        )
    return float(col_new), float(row_new)


def _parse_bundle_tilt_degrees(tilt_correction, tilt_hint):
    """Extract ``(tilt_x_deg, tilt_y_deg)`` from bundle payload structures."""

    if isinstance(tilt_correction, dict):
        tx = tilt_correction.get("tilt_x_deg")
        ty = tilt_correction.get("tilt_y_deg")
        try:
            tx = float(tx)
            ty = float(ty)
            if np.isfinite(tx) and np.isfinite(ty):
                return tx, ty
        except (TypeError, ValueError):
            pass

    if isinstance(tilt_hint, dict):
        try:
            rx = float(tilt_hint.get("rot1_rad"))
            ry = float(tilt_hint.get("rot2_rad"))
            if np.isfinite(rx) and np.isfinite(ry):
                return float(np.degrees(rx)), float(np.degrees(ry))
        except (TypeError, ValueError):
            pass

    return None, None


def _normalize_sign(value, default):
    """Return +/-1 sign metadata, falling back to ``default`` when invalid."""

    try:
        iv = int(value)
    except Exception:
        iv = int(default)
    if iv < 0:
        return -1
    if iv > 0:
        return 1
    return -1 if int(default) < 0 else 1


def convert_hbn_bundle_geometry_to_simulation(
    *,
    tilt_x_deg,
    tilt_y_deg,
    center_xy,
    source_rotate_k,
    target_rotate_k,
    image_size,
    simulation_gamma_sign_from_tilt_x=_CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_X,
    simulation_Gamma_sign_from_tilt_y=_CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_Y,
):
    """Convert hBN correction geometry into simulation detector geometry."""

    tx = float(tilt_x_deg)
    ty = float(tilt_y_deg)

    gamma_sign = _normalize_sign(
        simulation_gamma_sign_from_tilt_x,
        _CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_X,
    )
    Gamma_sign = _normalize_sign(
        simulation_Gamma_sign_from_tilt_y,
        _CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_Y,
    )

    gamma_src = float(gamma_sign * tx)
    Gamma_src = float(Gamma_sign * ty)

    k_delta = int(target_rotate_k) - int(source_rotate_k)
    alpha_deg = 90.0 * float(k_delta)
    alpha_rad = float(np.deg2rad(alpha_deg))
    c = float(np.cos(alpha_rad))
    s = float(np.sin(alpha_rad))
    gamma_deg = float(c * gamma_src + s * Gamma_src)
    Gamma_deg = float(-s * gamma_src + c * Gamma_src)

    center_row = None
    center_col = None
    if center_xy is not None:
        try:
            center_col_src = float(center_xy[0])
            center_row_src = float(center_xy[1])
            if np.isfinite(center_col_src) and np.isfinite(center_row_src):
                center_col_tgt, center_row_tgt = _rotate_point_by_k(
                    center_col_src,
                    center_row_src,
                    image_size,
                    k_delta,
                )
                center_row = float(center_row_tgt)
                center_col = float(center_col_tgt)
        except Exception:
            center_row = None
            center_col = None

    return {
        "gamma_deg": gamma_deg,
        "Gamma_deg": Gamma_deg,
        "center_row": center_row,
        "center_col": center_col,
        "k_delta": int(k_delta),
        "conversion_notes": {
            "tilt_correction_kind": _CANONICAL_TILT_CORRECTION_KIND,
            "tilt_model": _CANONICAL_TILT_MODEL,
            "source_rotate_k": int(source_rotate_k),
            "target_rotate_k": int(target_rotate_k),
            "frame_rotation_deg": float(alpha_deg),
            "component_rotation_applied": True,
            "simulation_gamma_sign_from_tilt_x": int(gamma_sign),
            "simulation_Gamma_sign_from_tilt_y": int(Gamma_sign),
        },
    }


def build_hbn_geometry_debug_trace(
    *,
    npz_center_xy=None,
    source_rotate_k=0,
    target_rotate_k=0,
    image_size=(1, 1),
    tilt_x_deg=None,
    tilt_y_deg=None,
    simulation_gamma_sign_from_tilt_x=_CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_X,
    simulation_Gamma_sign_from_tilt_y=_CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_Y,
    simulation_center_row=None,
    simulation_center_col=None,
):
    """Return a structured center/rotation trace for hBN-to-simulation mapping."""

    if isinstance(image_size, (tuple, list, np.ndarray)) and len(image_size) >= 2:
        image_h = int(image_size[0])
        image_w = int(image_size[1])
    else:
        image_h = int(image_size)
        image_w = int(image_size)
    image_h = max(1, image_h)
    image_w = max(1, image_w)

    source_k = int(source_rotate_k)
    target_k = int(target_rotate_k)
    k_delta = int(target_k - source_k)
    frame_rotation_deg = float(90.0 * k_delta)

    gamma_sign = _normalize_sign(
        simulation_gamma_sign_from_tilt_x,
        _CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_X,
    )
    Gamma_sign = _normalize_sign(
        simulation_Gamma_sign_from_tilt_y,
        _CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_Y,
    )

    trace = {
        "image_shape": (image_h, image_w),
        "source_rotate_k": source_k,
        "target_rotate_k": target_k,
        "k_delta": k_delta,
        "frame_rotation_deg": frame_rotation_deg,
        "simulation_gamma_sign_from_tilt_x": int(gamma_sign),
        "simulation_Gamma_sign_from_tilt_y": int(Gamma_sign),
        "npz_center": None,
        "expected_sim_center": None,
        "expected_center_roundtrip_to_npz": None,
        "expected_roundtrip_error": None,
        "applied_sim_center": None,
        "applied_center_back_to_npz": None,
        "applied_back_to_npz_error": None,
        "applied_center_rotated_inverse_k_delta": None,
        "applied_inverse_k_delta_error": None,
        "tilt_component_source": None,
        "tilt_component_target": None,
    }

    try:
        tx = float(tilt_x_deg)
        ty = float(tilt_y_deg)
        if np.isfinite(tx) and np.isfinite(ty):
            gamma_src = float(gamma_sign * tx)
            Gamma_src = float(Gamma_sign * ty)
            alpha_rad = float(np.deg2rad(frame_rotation_deg))
            c = float(np.cos(alpha_rad))
            s = float(np.sin(alpha_rad))
            gamma_tgt = float(c * gamma_src + s * Gamma_src)
            Gamma_tgt = float(-s * gamma_src + c * Gamma_src)
            trace["tilt_component_source"] = {
                "tilt_x_deg": tx,
                "tilt_y_deg": ty,
                "gamma_src_deg": gamma_src,
                "Gamma_src_deg": Gamma_src,
            }
            trace["tilt_component_target"] = {
                "gamma_target_deg": gamma_tgt,
                "Gamma_target_deg": Gamma_tgt,
            }
    except Exception:
        pass

    src_col = None
    src_row = None
    if npz_center_xy is not None:
        try:
            src_col = float(npz_center_xy[0])
            src_row = float(npz_center_xy[1])
            if np.isfinite(src_col) and np.isfinite(src_row):
                trace["npz_center"] = {"col": src_col, "row": src_row}
                exp_col, exp_row = _rotate_point_by_k(
                    src_col,
                    src_row,
                    (image_h, image_w),
                    k_delta,
                )
                trace["expected_sim_center"] = {"row": float(exp_row), "col": float(exp_col)}
                rt_col, rt_row = _rotate_point_by_k(
                    exp_col,
                    exp_row,
                    (image_h, image_w),
                    -k_delta,
                )
                trace["expected_center_roundtrip_to_npz"] = {
                    "col": float(rt_col),
                    "row": float(rt_row),
                }
                trace["expected_roundtrip_error"] = {
                    "d_col": float(rt_col - src_col),
                    "d_row": float(rt_row - src_row),
                }
        except Exception:
            pass

    try:
        sim_row = float(simulation_center_row)
        sim_col = float(simulation_center_col)
        if np.isfinite(sim_row) and np.isfinite(sim_col):
            trace["applied_sim_center"] = {"row": sim_row, "col": sim_col}
            back_col, back_row = _rotate_point_by_k(
                sim_col,
                sim_row,
                (image_h, image_w),
                -k_delta,
            )
            trace["applied_center_back_to_npz"] = {
                "col": float(back_col),
                "row": float(back_row),
            }
            if src_col is not None and src_row is not None:
                trace["applied_back_to_npz_error"] = {
                    "d_col": float(back_col - src_col),
                    "d_row": float(back_row - src_row),
                }
            inv_col, inv_row = _rotate_point_by_k(
                sim_col,
                sim_row,
                (image_h, image_w),
                -k_delta,
            )
            trace["applied_center_rotated_inverse_k_delta"] = {
                "col": float(inv_col),
                "row": float(inv_row),
                "inverse_k_delta": int(-k_delta),
            }
            if src_col is not None and src_row is not None:
                trace["applied_inverse_k_delta_error"] = {
                    "d_col": float(inv_col - src_col),
                    "d_row": float(inv_row - src_row),
                }
    except Exception:
        pass

    return trace


def format_hbn_geometry_debug_trace(trace):
    """Format :func:`build_hbn_geometry_debug_trace` output for display/logging."""

    image_h, image_w = trace.get("image_shape", (None, None))
    source_k = trace.get("source_rotate_k")
    target_k = trace.get("target_rotate_k")
    k_delta = trace.get("k_delta")
    frame_rotation_deg = trace.get("frame_rotation_deg")
    gamma_sign = trace.get("simulation_gamma_sign_from_tilt_x")
    Gamma_sign = trace.get("simulation_Gamma_sign_from_tilt_y")

    lines = []
    lines.append("hBN Bundle Geometry Debug")
    lines.append(f"  image_shape: ({image_h}, {image_w})")
    lines.append(
        "  rotation_k: "
        f"source={source_k}, target={target_k}, k_delta={k_delta}, "
        f"frame_rotation_deg={frame_rotation_deg:.1f}"
    )
    lines.append(
        "  sign_map: "
        f"gamma<-tilt_x ({gamma_sign:+d}), Gamma<-tilt_y ({Gamma_sign:+d})"
    )

    tilt_src = trace.get("tilt_component_source")
    tilt_tgt = trace.get("tilt_component_target")
    if isinstance(tilt_src, dict) and isinstance(tilt_tgt, dict):
        lines.append(
            "  tilt_source_deg: "
            f"tilt_x={tilt_src['tilt_x_deg']:.6f}, tilt_y={tilt_src['tilt_y_deg']:.6f}"
        )
        lines.append(
            "  component_source_deg: "
            f"gamma_src={tilt_src['gamma_src_deg']:.6f}, "
            f"Gamma_src={tilt_src['Gamma_src_deg']:.6f}"
        )
        lines.append(
            "  component_target_deg: "
            f"gamma_target={tilt_tgt['gamma_target_deg']:.6f}, "
            f"Gamma_target={tilt_tgt['Gamma_target_deg']:.6f}"
        )

    npz_center = trace.get("npz_center")
    expected_sim = trace.get("expected_sim_center")
    roundtrip = trace.get("expected_center_roundtrip_to_npz")
    roundtrip_err = trace.get("expected_roundtrip_error")
    if isinstance(npz_center, dict):
        lines.append(
            "  npz_center(col,row): "
            f"({npz_center['col']:.3f}, {npz_center['row']:.3f})"
        )
    if isinstance(expected_sim, dict):
        lines.append(
            "  expected_sim_center(row,col): "
            f"({expected_sim['row']:.3f}, {expected_sim['col']:.3f})"
        )
    if isinstance(roundtrip, dict) and isinstance(roundtrip_err, dict):
        lines.append(
            "  expected_roundtrip_to_npz(col,row): "
            f"({roundtrip['col']:.3f}, {roundtrip['row']:.3f}) "
            f"error(d_col={roundtrip_err['d_col']:+.6f}, d_row={roundtrip_err['d_row']:+.6f})"
        )

    applied_sim = trace.get("applied_sim_center")
    applied_back = trace.get("applied_center_back_to_npz")
    applied_err = trace.get("applied_back_to_npz_error")
    if isinstance(applied_sim, dict):
        lines.append(
            "  applied_sim_center(row,col): "
            f"({applied_sim['row']:.3f}, {applied_sim['col']:.3f})"
        )
    if isinstance(applied_back, dict):
        line = (
            "  applied_back_to_npz(col,row): "
            f"({applied_back['col']:.3f}, {applied_back['row']:.3f})"
        )
        if isinstance(applied_err, dict):
            line += (
                " "
                f"error(d_col={applied_err['d_col']:+.6f}, d_row={applied_err['d_row']:+.6f})"
            )
        lines.append(line)

    if int(k_delta or 0) != 0:
        direction = "CCW" if int(k_delta) < 0 else "CW"
        lines.append(
            "  note: rotating simulation by "
            f"{abs(int(k_delta)) * 90:+d} deg {direction} should recover hBN frame"
        )

    return "\n".join(lines)


def _load_paths_from_file(paths_file):
    with open(paths_file, encoding="utf-8") as fh:
        text = fh.read()
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            data = json.loads(text)
    return data or {}


def resolve_hbn_paths(
    osc_path=None,
    dark_path=None,
    bundle_path=None,
    click_profile_path=None,
    fit_profile_path=None,
    paths_file=None,
):
    """Return resolved paths using CLI args or a YAML/JSON file."""

    def _pick(keys, data):
        for key in keys:
            value = data.get(key)
            if value:
                return os.path.expanduser(value)
        return None

    resolved = dict(
        osc=os.path.expanduser(osc_path) if osc_path else None,
        dark=os.path.expanduser(dark_path) if dark_path else None,
        bundle=os.path.expanduser(bundle_path) if bundle_path else None,
        click_profile=os.path.expanduser(click_profile_path) if click_profile_path else None,
        fit_profile=os.path.expanduser(fit_profile_path) if fit_profile_path else None,
        beam_center=None,
    )

    search_file = paths_file
    if search_file is None:
        config_dir = Path(__file__).resolve().parents[1] / "config"
        for candidate_name in ("hbn_paths.yaml", "hbn_paths.example.yaml"):
            candidate_path = config_dir / candidate_name
            if candidate_path.exists():
                search_file = str(candidate_path)
                break

    file_data = None
    if search_file:
        file_data = _load_paths_from_file(search_file)
        if resolved["osc"] is None:
            resolved["osc"] = _pick(["calibrant", "osc", "calibrant_path", "calibrant_file"], file_data)
        if resolved["dark"] is None:
            resolved["dark"] = _pick(["dark", "dark_file", "dark_path"], file_data)
        if resolved["bundle"] is None:
            resolved["bundle"] = _pick(["bundle", "npz", "bundle_path"], file_data)
        if resolved["click_profile"] is None:
            resolved["click_profile"] = _pick(["click_profile", "profile", "click_profile_path"], file_data)
        if resolved["fit_profile"] is None:
            resolved["fit_profile"] = _pick(["fit_profile", "fit", "fit_profile_path"], file_data)
        if resolved["beam_center"] is None:
            beam_x = _pick(["beam_center_x", "beam_x", "center_x", "xc"], file_data)
            beam_y = _pick(["beam_center_y", "beam_y", "center_y", "yc"], file_data)
            beam_center_from_list = file_data.get("beam_center") if isinstance(file_data, dict) else None
            if beam_center_from_list and isinstance(beam_center_from_list, (list, tuple)):
                if len(beam_center_from_list) == 2:
                    beam_x, beam_y = beam_center_from_list
            if beam_x is not None and beam_y is not None:
                try:
                    resolved["beam_center"] = (float(beam_x), float(beam_y))
                except (TypeError, ValueError):
                    resolved["beam_center"] = None
    resolved["paths_file"] = search_file if search_file else None
    return resolved


def load_bundle_npz(path, *, verbose=True):
    data = np.load(path, allow_pickle=True)
    img_bgsub = data["img_bgsub"]
    img_log = data["img_log"]
    ell_points_arr = data["ell_points_ds"]
    ellipse_params = data["ellipse_params"]
    distance_info = data.get("distance_estimate_m")
    tilt_correction = data.get("tilt_correction")
    tilt_hint = data.get("tilt_hint")
    expected_peaks = data.get("expected_peaks")
    center = data.get("center")

    required_metadata_keys = (
        "sim_background_rotate_k",
        "tilt_correction_kind",
        "tilt_model",
        "tilt_frame",
        "simulation_gamma_sign_from_tilt_x",
        "simulation_Gamma_sign_from_tilt_y",
    )
    missing_metadata = [key for key in required_metadata_keys if key not in data]
    if missing_metadata:
        missing_text = ", ".join(missing_metadata)
        raise KeyError(
            "Bundle is missing required canonical metadata keys: "
            f"{missing_text}"
        )

    if tilt_correction is None and "tilt_x_deg" in data and "tilt_y_deg" in data:
        try:
            tx = float(np.asarray(data["tilt_x_deg"]).reshape(-1)[0])
            ty = float(np.asarray(data["tilt_y_deg"]).reshape(-1)[0])
            if np.isfinite(tx) and np.isfinite(ty):
                tilt_correction = {
                    "tilt_x_deg": tx,
                    "tilt_y_deg": ty,
                    "source": "hbn_fitter",
                }
        except Exception:
            tilt_correction = None

    try:
        source_rotate_k = int(np.asarray(data["sim_background_rotate_k"]).reshape(-1)[0])
    except Exception as exc:
        raise ValueError("Invalid canonical key 'sim_background_rotate_k'.") from exc
    try:
        tilt_correction_kind = str(np.asarray(data["tilt_correction_kind"]).reshape(-1)[0])
    except Exception as exc:
        raise ValueError("Invalid canonical key 'tilt_correction_kind'.") from exc
    try:
        tilt_model = str(np.asarray(data["tilt_model"]).reshape(-1)[0])
    except Exception as exc:
        raise ValueError("Invalid canonical key 'tilt_model'.") from exc
    try:
        tilt_frame = str(np.asarray(data["tilt_frame"]).reshape(-1)[0])
    except Exception as exc:
        raise ValueError("Invalid canonical key 'tilt_frame'.") from exc
    try:
        gamma_sign_from_tilt_x = _normalize_sign(
            np.asarray(data["simulation_gamma_sign_from_tilt_x"]).reshape(-1)[0],
            _CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_X,
        )
    except Exception as exc:
        raise ValueError(
            "Invalid canonical key 'simulation_gamma_sign_from_tilt_x'."
        ) from exc
    try:
        gamma_sign_from_tilt_y = _normalize_sign(
            np.asarray(data["simulation_Gamma_sign_from_tilt_y"]).reshape(-1)[0],
            _CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_Y,
        )
    except Exception as exc:
        raise ValueError(
            "Invalid canonical key 'simulation_Gamma_sign_from_tilt_y'."
        ) from exc

    if center is None:
        center = data.get("detector_center")
    if center is None:
        center = data.get("center_fixed")

    ell_points_ds = []
    for arr in ell_points_arr:
        arr = np.asarray(arr)
        ell_points_ds.append([(float(x), float(y)) for x, y in arr])

    ellipses = []
    for row in ellipse_params:
        xc, yc, a, b, theta = [float(v) for v in row]
        ellipses.append(
            dict(
                xc=xc,
                yc=yc,
                a=a,
                b=b,
                theta=theta,
            )
        )

    if distance_info is not None:
        try:
            distance_info = distance_info.item()
        except Exception:
            pass
    if tilt_correction is not None:
        try:
            tilt_correction = tilt_correction.item()
        except Exception:
            pass
    if isinstance(tilt_correction, dict):
        if "tilt_x_deg" not in tilt_correction and "tilt_x_deg" in data:
            try:
                tilt_correction["tilt_x_deg"] = float(
                    np.asarray(data["tilt_x_deg"]).reshape(-1)[0]
                )
            except Exception:
                pass
        if "tilt_y_deg" not in tilt_correction and "tilt_y_deg" in data:
            try:
                tilt_correction["tilt_y_deg"] = float(
                    np.asarray(data["tilt_y_deg"]).reshape(-1)[0]
                )
            except Exception:
                pass
    elif "tilt_x_deg" in data and "tilt_y_deg" in data:
        try:
            tx = float(np.asarray(data["tilt_x_deg"]).reshape(-1)[0])
            ty = float(np.asarray(data["tilt_y_deg"]).reshape(-1)[0])
            if np.isfinite(tx) and np.isfinite(ty):
                tilt_correction = {
                    "tilt_x_deg": tx,
                    "tilt_y_deg": ty,
                    "source": "hbn_bundle",
                }
        except Exception:
            tilt_correction = None

    if isinstance(tilt_correction, dict):
        tilt_correction.setdefault("tilt_correction_kind", str(tilt_correction_kind))
        tilt_correction.setdefault("tilt_model", str(tilt_model))
        tilt_correction.setdefault("tilt_frame", str(tilt_frame))
        tilt_correction.setdefault("sim_background_rotate_k", int(source_rotate_k))
        tilt_correction.setdefault(
            "simulation_gamma_sign_from_tilt_x", int(gamma_sign_from_tilt_x)
        )
        tilt_correction.setdefault(
            "simulation_Gamma_sign_from_tilt_y", int(gamma_sign_from_tilt_y)
        )
    if tilt_hint is not None:
        try:
            tilt_hint = tilt_hint.item()
        except Exception:
            pass
    if isinstance(tilt_hint, dict):
        tilt_hint.setdefault("sim_background_rotate_k", int(source_rotate_k))
        tilt_hint.setdefault("tilt_correction_kind", str(tilt_correction_kind))
        tilt_hint.setdefault("tilt_model", str(tilt_model))
        tilt_hint.setdefault("tilt_frame", str(tilt_frame))
        tilt_hint.setdefault(
            "simulation_gamma_sign_from_tilt_x", int(gamma_sign_from_tilt_x)
        )
        tilt_hint.setdefault(
            "simulation_Gamma_sign_from_tilt_y", int(gamma_sign_from_tilt_y)
        )
    if expected_peaks is not None:
        try:
            expected_peaks = expected_peaks.item()
        except Exception:
            pass
    if center is not None:
        try:
            center = tuple(center.tolist()) if hasattr(center, "tolist") else tuple(center)
        except Exception:
            pass

    if verbose:
        print(f"Loaded bundle from:\n  {path}")
        print(f"  image shape: {img_bgsub.shape}")
        print(f"  number of ellipses: {len(ellipses)}")
    if verbose and distance_info:
        print(
            "  distance estimate: "
            f"mean={distance_info.get('mean_m', float('nan')):.4f} m"
        )
    if verbose and center is not None:
        try:
            cx, cy = center
            print(f"  detector center: ({float(cx):.3f}, {float(cy):.3f}) px")
        except Exception:
            pass
    return (
        img_bgsub,
        img_log,
        ell_points_ds,
        ellipses,
        distance_info,
        tilt_correction,
        tilt_hint,
        expected_peaks,
        center,
    )


def load_tilt_hint(
    paths_file=None,
    *,
    resolve_hbn_paths_fn: Callable[..., dict[str, Any]] | None = None,
    load_bundle_npz_fn: Callable[..., tuple[Any, ...]] | None = None,
    convert_bundle_geometry_fn: Callable[..., dict[str, Any]] | None = None,
):
    """Load the latest converted detector geometry hint from an hBN bundle."""

    resolve_paths = resolve_hbn_paths if resolve_hbn_paths_fn is None else resolve_hbn_paths_fn
    load_bundle = load_bundle_npz if load_bundle_npz_fn is None else load_bundle_npz_fn
    convert_geometry = (
        convert_hbn_bundle_geometry_to_simulation
        if convert_bundle_geometry_fn is None
        else convert_bundle_geometry_fn
    )

    resolved = resolve_paths(paths_file=paths_file)
    bundle_path = resolved.get("bundle")
    if not bundle_path or not os.path.exists(bundle_path):
        return None

    try:
        img_bgsub, _, _, _, distance_info, tilt_correction, tilt_hint, _, center = load_bundle(
            bundle_path,
            verbose=False,
        )
    except Exception:
        return None

    tilt_x_deg, tilt_y_deg = _parse_bundle_tilt_degrees(tilt_correction, tilt_hint)
    if tilt_x_deg is None or tilt_y_deg is None:
        return None

    source_rotate_k = 0
    gamma_sign_from_tilt_x = _CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_X
    gamma_sign_from_tilt_y = _CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_Y
    if isinstance(tilt_correction, dict):
        source_rotate_k = int(tilt_correction.get("sim_background_rotate_k", 0))
        gamma_sign_from_tilt_x = _normalize_sign(
            tilt_correction.get(
                "simulation_gamma_sign_from_tilt_x",
                gamma_sign_from_tilt_x,
            ),
            gamma_sign_from_tilt_x,
        )
        gamma_sign_from_tilt_y = _normalize_sign(
            tilt_correction.get(
                "simulation_Gamma_sign_from_tilt_y",
                gamma_sign_from_tilt_y,
            ),
            gamma_sign_from_tilt_y,
        )
    elif isinstance(tilt_hint, dict):
        source_rotate_k = int(tilt_hint.get("sim_background_rotate_k", 0))
        gamma_sign_from_tilt_x = _normalize_sign(
            tilt_hint.get(
                "simulation_gamma_sign_from_tilt_x",
                gamma_sign_from_tilt_x,
            ),
            gamma_sign_from_tilt_x,
        )
        gamma_sign_from_tilt_y = _normalize_sign(
            tilt_hint.get(
                "simulation_Gamma_sign_from_tilt_y",
                gamma_sign_from_tilt_y,
            ),
            gamma_sign_from_tilt_y,
        )

    gamma_sign_from_tilt_x = -_normalize_sign(
        gamma_sign_from_tilt_x, _CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_X
    )

    image_shape = (1, 1)
    try:
        if img_bgsub is not None and np.asarray(img_bgsub).ndim >= 2:
            image_shape = tuple(np.asarray(img_bgsub).shape[:2])
    except Exception:
        image_shape = (1, 1)

    converted = convert_geometry(
        tilt_x_deg=float(tilt_x_deg),
        tilt_y_deg=float(tilt_y_deg),
        center_xy=center,
        source_rotate_k=int(source_rotate_k),
        target_rotate_k=int(SIM_BACKGROUND_ROTATE_K),
        image_size=image_shape,
        simulation_gamma_sign_from_tilt_x=int(gamma_sign_from_tilt_x),
        simulation_Gamma_sign_from_tilt_y=int(gamma_sign_from_tilt_y),
    )
    hint = dict(converted)
    if distance_info:
        try:
            mean_m = float(distance_info.get("mean_m", float("nan")))
        except Exception:
            mean_m = float("nan")
        if np.isfinite(mean_m):
            hint["distance_m"] = mean_m
    return hint
