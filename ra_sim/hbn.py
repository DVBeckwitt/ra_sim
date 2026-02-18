"""
hbn_bgsub_fit_ellipses_profiles_bundle.py

Workflow
1) Default mode (no --load-bundle):
   - Load hBN OSC image and dark frame, subtract background.
   - Apply log to the image for visualization and intensity use.
   - Optionally reuse a saved "click profile" with --reuse-profile when a JSON exists,
     otherwise interactively collect points:
       * Left click: point on current ring
       * Right drag: zoom box
       * 'r': reset zoom
     and save click profile JSON.
   - For each ellipse:
       * Fit initial ellipse from clicked points.
       * Refine ellipse by following intensity maxima
         on the full resolution log image.
   - Save:
       * hbn_bgsub.tiff                (background subtracted image)
       * hbn_bgsub_ellipses.png        (overlay with ellipses)
       * hbn_ellipse_profile.json      (clicked points)
       * hbn_ellipse_fit_profile.json  (ellipse parameters)
       * hbn_ellipse_bundle.npz        (image + log + clicks + params)

2) Bundle mode:
   - python hbn_bgsub_fit_ellipses_profiles_bundle.py --load-bundle PATH
   - Loads the NPZ created by this script.
   - Recreates JSONs and overlay from the bundle.

3) High resolution refine from bundle:
   - python hbn_bgsub_fit_ellipses_profiles_bundle.py --load-bundle PATH --highres-refine
   - Loads the NPZ created by this script.
   - Recomputes a full resolution background subtracted image from the OSC files
     (for example 3000x3000), uses the previous ellipse parameters as initial guesses,
     refines them on this full resolution image, and saves updated JSONs, bundle, and overlay.
"""

import math
import os
import json
import argparse
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml
from scipy.optimize import minimize

from matplotlib.patches import Rectangle
from skimage.measure import EllipseModel

import OSC_Reader
from OSC_Reader import OSC_Reader
from ra_sim.path_config import get_instrument_config
from ra_sim.utils import calculations


# ------------------------------------------------------------
# Configuration parameters
# ------------------------------------------------------------

# Number of rings / ellipses
N_ELLIPSES = 5

# Points clicked per ellipse (>=5 recommended)
# An unconstrained ellipse has five free parameters (xc, yc, a, b, theta), so
# five non-collinear points are the minimum needed to uniquely determine it
# without imposing extra assumptions.
POINTS_PER_ELLIPSE = 5

# Log intensity for display and fitting
USE_LOG_INTENSITY = True
LOG_EPS = 1e-3

# Try to reuse existing click profile JSON instead of clicking again
LOAD_PROFILE = False

# Intensity based refinement settings
REFINE_N_ANGLES = 360    # angular sampling along ellipse
REFINE_DR = 10.0         # half width of radial search window [pixels]
REFINE_STEP = 1.0        # radial step [pixels]


# hBN reference parameters for expected peak calculation (Cu Kα, λ=1.5406 Å)
HBN_LATTICE_A_ANG = 2.504
HBN_LATTICE_C_ANG = 6.661
CU_K_ALPHA_WAVELENGTH_ANG = 1.5406
# Five low-angle peaks that typically appear in the hBN calibrant image
HBN_HKLS = [
    (0, 0, 2),
    (1, 0, 0),
    (1, 0, 1),
    (1, 0, 2),
    (0, 0, 4),
]

# Simulator detector geometry is rotated 90° clockwise from hBN-fitter native.
SIM_BACKGROUND_ROTATE_K = -1

_CANONICAL_TILT_CORRECTION_KIND = "to_flat"
_CANONICAL_TILT_MODEL = "RzRx"
_CANONICAL_TILT_FRAME = "simulation_background_display"
_CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_X = 1
_CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_Y = 1


def _rotation_matrix_x_deg(angle_deg):
    """Rotation matrix matching diffraction.py detector X-rotation convention."""

    ang = float(np.deg2rad(angle_deg))
    c = float(np.cos(ang))
    s = float(np.sin(ang))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, s],
            [0.0, -s, c],
        ],
        dtype=np.float64,
    )


def _rotation_matrix_z_deg(angle_deg):
    """Rotation matrix matching diffraction.py detector Z-rotation convention."""

    ang = float(np.deg2rad(angle_deg))
    c = float(np.cos(ang))
    s = float(np.sin(ang))
    return np.array(
        [
            [c, s, 0.0],
            [-s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


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
    """Extract (tilt_x_deg, tilt_y_deg) from bundle payload structures."""

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
    """Convert hBN correction geometry into simulation detector geometry.

    Parameters
    ----------
    tilt_x_deg, tilt_y_deg
        hBN fitter detector correction angles (degrees) in source frame.
    center_xy
        Detector center tuple in source frame as ``(col, row)``.
    source_rotate_k, target_rotate_k
        Source and target orientation factors (``np.rot90`` convention).
    image_size
        Detector image size (square integer or ``(height, width)`` tuple) used
        for center rotation.
    simulation_gamma_sign_from_tilt_x, simulation_Gamma_sign_from_tilt_y
        Sign convention metadata linking hBN tilt components to simulator
        detector rotations. ``+1`` means same sign; ``-1`` means opposite sign.
    """

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

    # Signed source-frame components before frame rotation.
    gamma_src = float(gamma_sign * tx)
    Gamma_src = float(Gamma_sign * ty)

    # Rotate components into target frame using the same k-delta convention as
    # np.rot90: positive k = 90 deg CCW.
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
    """Return a structured center/rotation trace for hBN->simulation mapping."""

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

    inv_item = trace.get("applied_center_rotated_inverse_k_delta")
    inv_err = trace.get("applied_inverse_k_delta_error")
    if isinstance(inv_item, dict):
        line = (
            f"  sim_rotated_by_inverse_k={inv_item['inverse_k_delta']:+d}(col,row): "
            f"({inv_item['col']:.3f}, {inv_item['row']:.3f})"
        )
        if isinstance(inv_err, dict):
            line += (
                " "
                f"error(d_col={inv_err['d_col']:+.6f}, d_row={inv_err['d_row']:+.6f})"
            )
        lines.append(line)

    if int(k_delta) == -1:
        lines.append(
            "  note: k_delta=-1 means simulation is 90 deg CW from hBN; "
            "rotating simulation by +90 deg CCW should recover hBN frame."
        )

    return "\n".join(lines)


# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fit diffraction rings with ellipses and save bundle."
    )
    parser.add_argument(
        "--osc",
        type=str,
        default=None,
        help=(
            "Path to the hBN OSC image. Required unless --load-bundle is used without "
            "--highres-refine."
        ),
    )
    parser.add_argument(
        "--dark",
        type=str,
        default=None,
        help=(
            "Path to the dark frame OSC image. Required unless --load-bundle is used "
            "without --highres-refine."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write outputs (defaults to ~/Downloads).",
    )
    parser.add_argument(
        "--reuse-profile",
        action="store_true",
        help="Reuse an existing click profile JSON in the output directory if present.",
    )
    parser.add_argument(
        "--paths-file",
        type=str,
        default=None,
        help=(
            "Optional YAML/JSON file containing calibrant, dark, and artifact paths. "
            "Accepted keys: calibrant/osc, dark/dark_file, bundle/npz, click_profile/profile, "
            "fit_profile/fit. If omitted, the workflow will look for "
            "config/hbn_paths.yaml."
        ),
    )
    parser.add_argument(
        "--load-bundle",
        nargs="?",
        const="",
        default=None,
        help=(
            "Path to NPZ bundle created by this script. If given, load it instead of "
            "recomputing everything. You may omit the path to let the script pull the "
            "bundle from a paths file (defaults to config/hbn_paths.yaml)."
        ),
    )
    parser.add_argument(
        "--highres-refine",
        action="store_true",
        help=(
            "When used together with --load-bundle, recompute a full resolution "
            "background subtracted image from the OSC files and refine ellipses on it, "
            "starting from the bundle fit."
        ),
    )
    parser.add_argument(
        "--reclick",
        action="store_true",
        help=(
            "Force a new interactive click session even when loading a bundle. "
            "Requires --osc and --dark so the background image can be rebuilt before "
            "collecting 5 points per ring."
        ),
    )
    parser.add_argument(
        "--prompt-save-bundle",
        action="store_true",
        help=(
            "After a successful fit, open a file-save dialog to choose where to write "
            "an hBN NPZ bundle."
        ),
    )
    parser.add_argument(
        "--load-clicks",
        type=str,
        default=None,
        help=(
            "Optional JSON click profile to load instead of interactively collecting points "
            "(keys: image_shape, points)."
        ),
    )
    parser.add_argument(
        "--save-clicks",
        nargs="?",
        const="",
        default=None,
        help=(
            "Write the clicked points to a JSON profile after selection (defaults to "
            "hbn_click_profile.json in the output directory when no path is provided)."
        ),
    )
    parser.add_argument(
        "--clicks-only",
        action="store_true",
        help=(
            "Stop after collecting points (and saving them if requested) without fitting ellipses or "
            "writing the full bundle."
        ),
    )
    parser.add_argument(
        "--beam-center-x",
        type=float,
        default=None,
        help=(
            "Beam center x-position in pixels (origin at image top-left). When provided "
            "with --beam-center-y, guides will radiate from this point during clicking."
        ),
    )
    parser.add_argument(
        "--beam-center-y",
        type=float,
        default=None,
        help=(
            "Beam center y-position in pixels (origin at image top-left). When provided "
            "with --beam-center-x, guides will radiate from this point during clicking."
        ),
    )
    return parser.parse_args(argv)

# ------------------------------------------------------------
# Background subtraction
# ------------------------------------------------------------
def load_and_bgsub(file_path_hbn, dark_file):
    print("Loading dark image...")
    dark = OSC_Reader.read_osc(dark_file).astype(np.float32)

    print("Loading hBN calibrant image...")
    raw = OSC_Reader.read_osc(file_path_hbn).astype(np.float32)

    k = int(SIM_BACKGROUND_ROTATE_K) % 4
    if k:
        dark = np.rot90(dark, k)
        raw = np.rot90(raw, k)

    print("Subtracting dark frame...")
    main = np.clip(raw - dark, 0, None)

    print("Estimating smooth background with Gaussian blur...")
    bg = cv2.GaussianBlur(main, ksize=(0, 0), sigmaX=25, sigmaY=25)

    print("Subtracting smooth background...")
    img_bgsub = np.clip(main - bg, 0, None).astype(np.float32)

    return img_bgsub


# ------------------------------------------------------------
# Image helpers
# ------------------------------------------------------------
def make_log_image(img_bgsub):
    img = np.clip(img_bgsub, 0, None).astype(np.float32)
    if USE_LOG_INTENSITY:
        print("Applying log transform...")
        img = np.log(img + LOG_EPS)
    return img


def make_click_image(img_bgsub):
    img_log = make_log_image(img_bgsub)

    vmin, vmax = np.percentile(img_log, (0.5, 99.5))
    norm = (img_log - vmin) / (vmax - vmin + 1e-9)
    norm = np.clip(norm, 0, 1)

    return norm


def make_display_image(img_bgsub):
    img_log = make_log_image(img_bgsub)
    p_low, p_high = np.percentile(img_log, (0.1, 99.9))
    disp = (img_log - p_low) / (p_high - p_low + 1e-9)
    disp = np.clip(disp, 0, 1)
    return disp


def _get_pixel_size_m(default=1e-4):
    """Return detector pixel size in meters (defaults to 100 µm)."""

    try:
        inst_cfg = get_instrument_config()
        return float(inst_cfg["instrument"]["detector"]["pixel_size_m"])
    except Exception:
        return default


def hbn_expected_peaks():
    """Return expected d-spacing and 2θ values for hBN with Cu Kα."""

    peaks = []
    for h, k, l in HBN_HKLS:
        d_ang = calculations.d_spacing(h, k, l, HBN_LATTICE_A_ANG, HBN_LATTICE_C_ANG)
        tth = calculations.two_theta(d_ang, CU_K_ALPHA_WAVELENGTH_ANG)
        if tth is None:
            continue
        peaks.append(
            {
                "hkl": [h, k, l],
                "d_spacing_ang": float(d_ang),
                "two_theta_deg": float(tth),
            }
        )
    return peaks


# ------------------------------------------------------------
# Path helpers
# ------------------------------------------------------------
def _load_paths_from_file(paths_file):
    with open(paths_file, "r", encoding="utf-8") as fh:
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
        default_path = Path(__file__).resolve().parents[1] / "config" / "hbn_paths.yaml"
        if default_path.exists():
            search_file = str(default_path)

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


# ------------------------------------------------------------
# Profile save / load
# ------------------------------------------------------------
def save_click_profile(path, ell_points_ds, img_shape):
    profile = {
        "image_shape": list(img_shape),
        "points": [
            [[float(x), float(y)] for (x, y) in ellipse_pts]
            for ellipse_pts in ell_points_ds
        ],
    }
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"Saved click profile to:\n  {path}")


def load_click_profile(path):
    with open(path, "r") as f:
        profile = json.load(f)
    points_raw = profile.get("points", [])
    ell_points_ds = []
    for ell in points_raw:
        ell_points_ds.append([(float(x), float(y)) for (x, y) in ell])
    print(f"Loaded click profile from:\n  {path}")
    return ell_points_ds


def save_fit_profile(
    path,
    ellipses,
    img_shape,
    tilt_hint=None,
    expected_peaks=None,
    distance_info=None,
    tilt_correction=None,
):
    profile = {
        "image_shape": list(img_shape),
        "ellipses": [
            {
                "index": i + 1,
                "xc": float(e["xc"]),
                "yc": float(e["yc"]),
                "a": float(e["a"]),
                "b": float(e["b"]),
                "theta_rad": float(e["theta"]),
                "theta_deg": float(np.degrees(e["theta"])),
            }
            for i, e in enumerate(ellipses)
        ],
    }
    if tilt_hint:
        profile["tilt_hint"] = tilt_hint
    if expected_peaks:
        profile["expected_peaks"] = expected_peaks
    if distance_info:
        profile["distance_estimate_m"] = distance_info
    if tilt_correction:
        profile["tilt_correction"] = tilt_correction
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"Saved fit profile to:\n  {path}")


def save_bundle(
    path,
    img_bgsub,
    img_log,
    ell_points_ds,
    ellipses,
    *,
    distance_info=None,
    tilt_correction=None,
    tilt_hint=None,
    expected_peaks=None,
    center=None,
):
    ell_points_arr = np.array(
        [np.array(pts, dtype=np.float32) for pts in ell_points_ds],
        dtype=object,
    )

    ellipse_params = np.zeros((len(ellipses), 5), dtype=np.float32)
    for i, e in enumerate(ellipses):
        ellipse_params[i, :] = [
            float(e["xc"]),
            float(e["yc"]),
            float(e["a"]),
            float(e["b"]),
            float(e["theta"]),
        ]

    np.savez(
        path,
        sim_background_rotate_k=np.array(int(SIM_BACKGROUND_ROTATE_K), dtype=np.int32),
        tilt_correction_kind=np.array(_CANONICAL_TILT_CORRECTION_KIND),
        tilt_model=np.array(_CANONICAL_TILT_MODEL),
        tilt_frame=np.array(_CANONICAL_TILT_FRAME),
        simulation_gamma_sign_from_tilt_x=np.array(
            int(_CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_X), dtype=np.int32
        ),
        simulation_Gamma_sign_from_tilt_y=np.array(
            int(_CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_Y), dtype=np.int32
        ),
        img_bgsub=img_bgsub.astype(np.float32),
        img_log=img_log.astype(np.float32),
        ell_points_ds=ell_points_arr,
        ellipse_params=ellipse_params,
        distance_estimate_m=distance_info,
        tilt_correction=tilt_correction,
        tilt_hint=tilt_hint,
        expected_peaks=expected_peaks,
        center=center,
    )
    print(f"Saved bundle NPZ to:\n  {path}")
    print("Note: load with allow_pickle=True for ell_points_ds.")


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

    # hbn_fitter bundles may store tilt as scalar keys instead of a dict.
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

    # hbn_fitter bundles store center under detector_center/center_fixed.
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


def load_tilt_hint(paths_file=None):
    """Load the latest converted detector geometry hint from an hBN bundle."""

    resolved = resolve_hbn_paths(paths_file=paths_file)
    bundle_path = resolved.get("bundle")
    if not bundle_path or not os.path.exists(bundle_path):
        return None

    try:
        img_bgsub, _, _, _, distance_info, tilt_correction, tilt_hint, _, center = load_bundle_npz(
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

    # Simulator convention: imported detector pitch uses the opposite sign
    # relative to bundle tilt_x mapping.
    gamma_sign_from_tilt_x = -_normalize_sign(
        gamma_sign_from_tilt_x, _CANONICAL_SIM_GAMMA_SIGN_FROM_TILT_X
    )

    image_shape = (1, 1)
    try:
        if img_bgsub is not None and np.asarray(img_bgsub).ndim >= 2:
            image_shape = tuple(np.asarray(img_bgsub).shape[:2])
    except Exception:
        image_shape = (1, 1)

    converted = convert_hbn_bundle_geometry_to_simulation(
        tilt_x_deg=float(tilt_x_deg),
        tilt_y_deg=float(tilt_y_deg),
        center_xy=center,
        source_rotate_k=int(source_rotate_k),
        target_rotate_k=int(SIM_BACKGROUND_ROTATE_K),
        image_size=image_shape,
        simulation_gamma_sign_from_tilt_x=int(gamma_sign_from_tilt_x),
        simulation_Gamma_sign_from_tilt_y=int(gamma_sign_from_tilt_y),
    )

    hint = {
        "gamma_deg": float(converted["gamma_deg"]),
        "Gamma_deg": float(converted["Gamma_deg"]),
        "center_row": converted.get("center_row"),
        "center_col": converted.get("center_col"),
        "k_delta": int(converted["k_delta"]),
        "conversion_notes": converted.get("conversion_notes", {}),
        # Backward-compatible aliases.
        "rot1_rad": float(np.deg2rad(converted["gamma_deg"])),
        "rot2_rad": float(np.deg2rad(converted["Gamma_deg"])),
    }
    hint["tilt_rad"] = float(math.hypot(hint["rot1_rad"], hint["rot2_rad"]))

    if distance_info and isinstance(distance_info, dict):
        hint["distance_m"] = distance_info.get("mean_m")

    return hint


# ------------------------------------------------------------
# Interactive clicking with zoom
# ------------------------------------------------------------
def get_points_per_ellipse_with_zoom(
    small, n_ellipses, pts_per_ellipse, beam_center=None, n_slices=5
):
    if pts_per_ellipse < 5:
        raise ValueError("Need at least 5 points per ellipse for a stable fit.")

    ell_points = [[] for _ in range(n_ellipses)]
    current_ellipse = 0
    current_point = 0
    all_points = []

    fig, ax = plt.subplots(figsize=(6, 6))

    h, w = small.shape
    ax.imshow(small, cmap="gray", origin="upper")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    title_text = ax.set_title(
        f"Ellipse 1/{n_ellipses}  point 0/{pts_per_ellipse}  "
        f"(Left: point, Right drag: zoom, 'r': reset)"
    )

    full_xlim = (0, w)
    full_ylim = (h, 0)

    zoom_rect = Rectangle(
        (0, 0), 1, 1,
        edgecolor="yellow",
        facecolor="none",
        linewidth=1.0,
        visible=False,
    )
    ax.add_patch(zoom_rect)
    zooming = {"active": False, "x0": None, "y0": None}

    seed_scatter = ax.scatter([], [], s=20, c="red", marker="x")

    center_marker = None
    guide_lines = []
    beam_center = (
        tuple(beam_center)
        if beam_center is not None and len(beam_center) == 2
        else (w / 2.0, h / 2.0)
    )

    def draw_guides():
        nonlocal center_marker, guide_lines

        for ln in guide_lines:
            ln.remove()
        guide_lines = []

        if center_marker is not None:
            center_marker.remove()
            center_marker = None

        if beam_center is None or n_slices <= 0:
            return

        xc, yc = beam_center
        radius = math.hypot(w, h)
        angles = np.linspace(0.0, 2.0 * np.pi, n_slices, endpoint=False)

        for ang in angles:
            x1 = xc + radius * math.cos(ang)
            y1 = yc + radius * math.sin(ang)
            ln = ax.plot([xc, x1], [yc, y1], linestyle=":", color="cyan", alpha=0.6)[
                0
            ]
            guide_lines.append(ln)

        center_marker = ax.plot(
            xc, yc, marker="+", markersize=6, markeredgewidth=1.4, color="cyan"
        )[0]

    draw_guides()

    def update_title():
        title_text.set_text(
            f"Ellipse {current_ellipse + 1}/{n_ellipses}  "
            f"point {current_point}/{pts_per_ellipse}  "
            f"(Left: point, Right drag: zoom, 'r': reset)"
        )

    def on_button_press(event):
        nonlocal current_ellipse, current_point, all_points

        if event.inaxes is not ax:
            return

        if event.button == 1:
            if event.xdata is None or event.ydata is None:
                return
            if current_ellipse >= n_ellipses:
                return

            x, y = float(event.xdata), float(event.ydata)
            ell_points[current_ellipse].append((x, y))
            all_points.append((x, y))
            xs, ys = zip(*all_points)
            seed_scatter.set_offsets(np.column_stack([xs, ys]))

            current_point += 1
            print(
                f"Ellipse {current_ellipse + 1}, point {current_point}: "
                f"x={x:.1f}, y={y:.1f}"
            )

            if current_point >= pts_per_ellipse:
                current_ellipse += 1
                current_point = 0
                if current_ellipse >= n_ellipses:
                    update_title()
                    fig.canvas.draw_idle()
                    plt.close(fig)
                    return

            update_title()
            fig.canvas.draw_idle()

        elif event.button == 3:
            if event.xdata is None or event.ydata is None:
                return
            zooming["active"] = True
            zooming["x0"] = event.xdata
            zooming["y0"] = event.ydata
            zoom_rect.set_visible(True)
            zoom_rect.set_xy((event.xdata, event.ydata))
            zoom_rect.set_width(0)
            zoom_rect.set_height(0)
            fig.canvas.draw_idle()

    def on_motion(event):
        if not zooming["active"]:
            return
        if event.inaxes is not ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x0 = zooming["x0"]
        y0 = zooming["y0"]
        x1 = event.xdata
        y1 = event.ydata

        x_min = min(x0, x1)
        x_max = max(x0, x1)
        y_min = min(y0, y1)
        y_max = max(y0, y1)

        zoom_rect.set_xy((x_min, y_min))
        zoom_rect.set_width(x_max - x_min)
        zoom_rect.set_height(y_max - y_min)
        fig.canvas.draw_idle()

    def on_button_release(event):
        if not zooming["active"]:
            return
        if event.button != 3:
            return

        zooming["active"] = False
        zoom_rect.set_visible(False)

        if event.inaxes is not ax:
            fig.canvas.draw_idle()
            return
        if event.xdata is None or event.ydata is None:
            fig.canvas.draw_idle()
            return

        x0 = zooming["x0"]
        y0 = zooming["y0"]
        x1 = event.xdata
        y1 = event.ydata

        if abs(x1 - x0) < 2 or abs(y1 - y0) < 2:
            fig.canvas.draw_idle()
            return

        x_min = min(x0, x1)
        x_max = max(x0, x1)
        y_min = min(y0, y1)
        y_max = max(y0, y1)

        y_lim0, y_lim1 = ax.get_ylim()
        if y_lim0 > y_lim1:
            ax.set_ylim(max(y_min, y_max), min(y_min, y_max))
        else:
            ax.set_ylim(min(y_min, y_max), max(y_min, y_max))
        ax.set_xlim(min(x_min, x_max), max(x_min, x_max))
        fig.canvas.draw_idle()

    def on_key_press(event):
        if event.key == "r":
            ax.set_xlim(*full_xlim)
            ax.set_ylim(*full_ylim)
            fig.canvas.draw_idle()

    cid_press = fig.canvas.mpl_connect("button_press_event", on_button_press)
    cid_release = fig.canvas.mpl_connect("button_release_event", on_button_release)
    cid_motion = fig.canvas.mpl_connect("motion_notify_event", on_motion)
    cid_key = fig.canvas.mpl_connect("key_press_event", on_key_press)

    plt.tight_layout()
    plt.show()

    fig.canvas.mpl_disconnect(cid_press)
    fig.canvas.mpl_disconnect(cid_release)
    fig.canvas.mpl_disconnect(cid_motion)
    fig.canvas.mpl_disconnect(cid_key)

    for i, pts in enumerate(ell_points, 1):
        print(f"Ellipse {i} collected {len(pts)} points.")

    return ell_points


# ------------------------------------------------------------
# Ellipse fitting and refinement
# ------------------------------------------------------------
def fit_initial_ellipse_from_points(pts_full):
    model = EllipseModel()
    ok = model.estimate(pts_full)
    if not ok:
        x = pts_full[:, 0]
        y = pts_full[:, 1]
        xc = float(x.mean())
        yc = float(y.mean())
        a = max(float((x.max() - x.min()) / 2.0), 1.0)
        b = max(float((y.max() - y.min()) / 2.0), 1.0)
        theta = 0.0
        print("EllipseModel.estimate failed, using crude bounding box estimate.")
        return (xc, yc, a, b, theta), False
    return model.params, True


def bilinear_sample(img, x, y):
    h, w = img.shape
    if x < 0 or x > w - 1 or y < 0 or y > h - 1:
        return 0.0

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)

    dx = x - x0
    dy = y - y0

    v00 = img[y0, x0]
    v10 = img[y0, x1]
    v01 = img[y1, x0]
    v11 = img[y1, x1]

    return (
        v00 * (1 - dx) * (1 - dy)
        + v10 * dx * (1 - dy)
        + v01 * (1 - dx) * dy
        + v11 * dx * dy
    )


def refine_ellipse_with_intensity(img_log, params):
    xc, yc, a, b, theta = params
    h, w = img_log.shape

    t_vals = np.linspace(0, 2 * np.pi, REFINE_N_ANGLES, endpoint=False)
    maxima_points = []

    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    for t in t_vals:
        ct = np.cos(t)
        st = np.sin(t)

        x_e = xc + a * ct * cos_th - b * st * sin_th
        y_e = yc + a * ct * sin_th + b * st * cos_th

        dx = x_e - xc
        dy = y_e - yc
        r0 = np.hypot(dx, dy)
        if r0 < 1e-6:
            continue
        dx /= r0
        dy /= r0

        best_val = -np.inf
        best_x = x_e
        best_y = y_e

        s_vals = np.arange(-REFINE_DR, REFINE_DR + REFINE_STEP, REFINE_STEP)
        for s in s_vals:
            r = r0 + s
            x = xc + r * dx
            y = yc + r * dy

            if x < 1 or x > w - 2 or y < 1 or y > h - 2:
                continue

            val = bilinear_sample(img_log, x, y)
            if val > best_val:
                best_val = val
                best_x = x
                best_y = y

        maxima_points.append((best_x, best_y))

    maxima_points = np.array(maxima_points, dtype=float)
    if maxima_points.shape[0] < 5:
        print("Refinement: not enough maxima points, keeping initial ellipse.")
        return params

    model = EllipseModel()
    ok = model.estimate(maxima_points)
    if not ok:
        print("Refinement: EllipseModel.estimate failed, keeping initial ellipse.")
        return params

    xc_new, yc_new, a_new, b_new, theta_new = model.params
    return float(xc_new), float(yc_new), float(a_new), float(b_new), float(theta_new)


def fit_ellipses_from_points(ell_points_ds, img_bgsub):
    img_log = make_log_image(img_bgsub)
    ellipses = []

    for i, pts in enumerate(ell_points_ds, 1):
        pts = np.array(pts, dtype=float)
        if pts.shape[0] < 5:
            print(f"Ellipse {i}: not enough points to fit (have {pts.shape[0]}). Skipping.")
            continue

        params0, _ = fit_initial_ellipse_from_points(pts)
        print(
            f"Ellipse {i} initial fit: "
            f"xc={params0[0]:.2f}, yc={params0[1]:.2f}, "
            f"a={params0[2]:.2f}, b={params0[3]:.2f}, "
            f"theta={np.degrees(params0[4]):.2f} deg"
        )

        params_refined = refine_ellipse_with_intensity(img_log, params0)
        xc, yc, a, b, theta = params_refined

        ellipses.append(
            dict(
                xc=xc,
                yc=yc,
                a=a,
                b=b,
                theta=theta,
            )
        )

        print(
            f"Ellipse {i} refined fit: "
            f"xc={xc:.2f}, yc={yc:.2f}, "
            f"a={a:.2f}, b={b:.2f}, "
            f"theta={np.degrees(theta):.2f} deg"
        )

    return ellipses


def refine_ellipses_from_existing(ellipses_prev, img_bgsub_prev, img_bgsub_hr, img_log_hr):
    h_prev, w_prev = img_bgsub_prev.shape
    h_hr, w_hr = img_bgsub_hr.shape
    sx = w_hr / w_prev
    sy = h_hr / h_prev

    ellipses_hr = []
    for i, e in enumerate(ellipses_prev, 1):
        xc0 = e["xc"] * sx
        yc0 = e["yc"] * sy
        a0 = e["a"] * sx
        b0 = e["b"] * sy
        theta0 = e["theta"]
        params0 = (xc0, yc0, a0, b0, theta0)

        print(
            f"High res refine ellipse {i}: "
            f"start xc={xc0:.2f}, yc={yc0:.2f}, a={a0:.2f}, b={b0:.2f}, "
            f"theta={np.degrees(theta0):.2f} deg"
        )

        xc, yc, a, b, theta = refine_ellipse_with_intensity(img_log_hr, params0)

        ellipses_hr.append(
            dict(
                xc=xc,
                yc=yc,
                a=a,
                b=b,
                theta=theta,
            )
        )

        print(
            f"High res ellipse {i} refined: "
            f"xc={xc:.2f}, yc={yc:.2f}, "
            f"a={a:.2f}, b={b:.2f}, "
            f"theta={np.degrees(theta):.2f} deg"
        )

    return ellipses_hr


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def ellipse_curve(xc, yc, a, b, theta, num=400):
    t = np.linspace(0, 2 * np.pi, num, endpoint=False)
    ct = np.cos(t)
    st = np.sin(t)
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    x = xc + a * ct * cos_th - b * st * sin_th
    y = yc + a * ct * sin_th + b * st * cos_th
    return x, y


def compute_common_center(ellipses):
    xs = np.array([e["xc"] for e in ellipses], dtype=float)
    ys = np.array([e["yc"] for e in ellipses], dtype=float)
    return float(xs.mean()), float(ys.mean())


def _effective_radius(a, b):
    """Return an effective circle radius from ellipse semiaxes.

    Using the arithmetic mean keeps the radius aligned with the measured axes while
    staying robust to minor eccentricity from noise or tilt.
    """

    return 0.5 * (abs(a) + abs(b))


def _estimate_tilt_components(a, b, theta):
    """Estimate detector tilt components (rot1, rot2) from an ellipse."""

    a = float(abs(a))
    b = float(abs(b))
    if a == 0 or b == 0:
        return None
    if b > a:
        a, b = b, a

    ratio = max(0.0, min(1.0, b / a))
    tilt_mag = float(np.arccos(ratio))
    rot1 = tilt_mag * float(np.cos(theta))
    rot2 = tilt_mag * float(np.sin(theta))
    return {
        "rot1_rad": rot1,
        "rot2_rad": rot2,
        "tilt_rad": tilt_mag,
        "tilt_deg": float(np.degrees(tilt_mag)),
        "theta_rad": float(theta),
        "theta_deg": float(np.degrees(theta)),
    }


def estimate_detector_tilt(ellipses):
    """Return an estimated detector tilt hint (rot1/rot2) from fitted ellipses."""

    if not ellipses:
        return None
    target = max(ellipses, key=lambda e: _effective_radius(e["a"], e["b"]))
    return _estimate_tilt_components(target["a"], target["b"], target["theta"])


def estimate_sample_detector_distance(
    ellipses, peaks, pixel_size_m=None, basis="ellipses"
):
    """Estimate the sample–detector distance using fitted rings and hBN peaks.

    Parameters
    ----------
    ellipses : list of dict
        Ellipse parameters (xc, yc, a, b, theta). When *basis* is "circles",
        the entries should already represent circularized radii (a == b).
    peaks : list of dict
        Expected hBN peak metadata (from :func:`hbn_expected_peaks`).
    pixel_size_m : float, optional
        Detector pixel size in meters. If omitted, defaults to the instrument
        configuration.
    basis : {"ellipses", "circles"}
        Records how the radii were produced (raw ellipse fits vs. circularized).
    """

    if not ellipses or not peaks:
        return None

    pixel_size_m = pixel_size_m or _get_pixel_size_m()
    distances = []
    for e, peak in zip(ellipses, peaks):
        radius_pix = _effective_radius(e["a"], e["b"])
        tth_rad = math.radians(peak["two_theta_deg"])
        if not math.isfinite(radius_pix) or radius_pix <= 0 or math.isclose(math.tan(tth_rad), 0.0):
            continue
        radius_m = radius_pix * pixel_size_m
        distances.append(radius_m / math.tan(tth_rad))

    if not distances:
        return None

    return {
        "per_ring_m": [float(d) for d in distances],
        "mean_m": float(np.mean(distances)),
        "pixel_size_m": float(pixel_size_m),
        "basis": str(basis),
    }


# ------------------------------------------------------------
# Tilt circularization helpers (adapted from hbn.py helper script)
# ------------------------------------------------------------

def _apply_tilt_xy(pts, center, tilt_x_deg, tilt_y_deg):
    """Apply inverse tilt correction assuming rotations about x/y through the center."""

    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.size == 0:
        return pts

    xc, yc = center

    max_tilt_deg = 45.0
    tilt_x_deg = float(np.clip(tilt_x_deg, -max_tilt_deg, max_tilt_deg))
    tilt_y_deg = float(np.clip(tilt_y_deg, -max_tilt_deg, max_tilt_deg))

    tx = np.deg2rad(tilt_x_deg)
    ty = np.deg2rad(tilt_y_deg)

    cx = np.cos(tx)  # affects y
    cy = np.cos(ty)  # affects x

    scale_y = 1e3 if abs(cx) < 1e-3 else 1.0 / cx
    scale_x = 1e3 if abs(cy) < 1e-3 else 1.0 / cy

    dx = (pts[:, 0] - xc) * scale_x
    dy = (pts[:, 1] - yc) * scale_y
    x_corr = xc + dx
    y_corr = yc + dy

    return np.column_stack([x_corr, y_corr])


def _circularity_cost(params_deg, ell_points_ds, center):
    tilt_x_deg, tilt_y_deg = params_deg
    xc, yc = center
    total_cost = 0.0

    for pts in ell_points_ds:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.size == 0:
            continue

        pts_corr = _apply_tilt_xy(pts, center, tilt_x_deg, tilt_y_deg)

        dx = pts_corr[:, 0] - xc
        dy = pts_corr[:, 1] - yc
        r = np.sqrt(dx * dx + dy * dy)

        r_mean = r.mean()
        if r_mean < 1e-9:
            continue

        metric = np.std(r) / r_mean
        total_cost += metric * metric

    return total_cost


def _circularity_metrics(ell_points_ds, center, tilt_x_deg, tilt_y_deg):
    xc, yc = center
    metrics = []

    for pts in ell_points_ds:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.size == 0:
            metrics.append(np.nan)
            continue

        pts_corr = _apply_tilt_xy(pts, center, tilt_x_deg, tilt_y_deg)

        dx = pts_corr[:, 0] - xc
        dy = pts_corr[:, 1] - yc
        r = np.sqrt(dx * dx + dy * dy)
        r_mean = r.mean()
        if r_mean < 1e-9:
            metrics.append(np.nan)
            continue

        metrics.append(float(np.std(r) / r_mean))

    return metrics


def _apply_tilt_correction_all(ell_points_ds, center, tilt_x_deg, tilt_y_deg):
    corrected = []
    for pts in ell_points_ds:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.size == 0:
            corrected.append(np.empty((0, 2)))
            continue
        corrected.append(_apply_tilt_xy(pts, center, tilt_x_deg, tilt_y_deg))
    return corrected


def _fit_ring_radii(ell_points, center):
    xc, yc = center
    radii = []
    for pts in ell_points:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.size == 0:
            radii.append(np.nan)
            continue
        dx = pts[:, 0] - xc
        dy = pts[:, 1] - yc
        r = np.sqrt(dx * dx + dy * dy)
        radii.append(float(r.mean()))
    return radii


def _find_tilt_correction(ell_points_ds, center):
    if not ell_points_ds:
        return None

    cost_zero = _circularity_cost((0.0, 0.0), ell_points_ds, center)

    tx_grid = np.linspace(-10.0, 10.0, 41)
    ty_grid = np.linspace(-10.0, 10.0, 41)
    best_cost = np.inf
    best_tx = 0.0
    best_ty = 0.0

    for tx in tx_grid:
        for ty in ty_grid:
            c = _circularity_cost((tx, ty), ell_points_ds, center)
            if c < best_cost:
                best_cost = c
                best_tx = tx
                best_ty = ty

    result = minimize(
        _circularity_cost,
        np.array([best_tx, best_ty], dtype=float),
        args=(ell_points_ds, center),
        method="Nelder-Mead",
        options=dict(maxiter=1000, xatol=1e-4, fatol=1e-8, disp=False),
    )

    tilt_x_opt, tilt_y_opt = result.x

    circ_before = _circularity_metrics(ell_points_ds, center, 0.0, 0.0)
    circ_after = _circularity_metrics(ell_points_ds, center, tilt_x_opt, tilt_y_opt)

    corrected_points = _apply_tilt_correction_all(
        ell_points_ds, center, tilt_x_opt, tilt_y_opt
    )

    radii_before = _fit_ring_radii(ell_points_ds, center)
    radii_after = _fit_ring_radii(corrected_points, center)

    return {
        "tilt_x_deg": float(tilt_x_opt),
        "tilt_y_deg": float(tilt_y_opt),
        "cost_zero": float(cost_zero),
        "cost_final": float(result.fun),
        "circ_before": circ_before,
        "circ_after": circ_after,
        "radii_before": radii_before,
        "radii_after": radii_after,
        "corrected_points": corrected_points,
    }


def _serialize_tilt_correction(tilt_correction, center=None):
    if not tilt_correction:
        return None

    serialized = {
        "tilt_x_deg": float(tilt_correction.get("tilt_x_deg", float("nan"))),
        "tilt_y_deg": float(tilt_correction.get("tilt_y_deg", float("nan"))),
        "cost_zero": float(tilt_correction.get("cost_zero", float("nan"))),
        "cost_final": float(tilt_correction.get("cost_final", float("nan"))),
        "circ_before": [
            float(x) if np.isfinite(x) else float("nan")
            for x in tilt_correction.get("circ_before", [])
        ],
        "circ_after": [
            float(x) if np.isfinite(x) else float("nan")
            for x in tilt_correction.get("circ_after", [])
        ],
        "radii_before": [
            float(x) if np.isfinite(x) else float("nan")
            for x in tilt_correction.get("radii_before", [])
        ],
        "radii_after": [
            float(x) if np.isfinite(x) else float("nan")
            for x in tilt_correction.get("radii_after", [])
        ],
        "radii_after_fit": [
            float(x) if np.isfinite(x) else float("nan")
            for x in tilt_correction.get("radii_after_fit", [])
        ],
    }

    if center is not None:
        serialized["center"] = [float(center[0]), float(center[1])]

    corrected_points = tilt_correction.get("corrected_points")
    if corrected_points is not None:
        serialized["corrected_points"] = [
            np.asarray(pts, dtype=float).tolist() for pts in corrected_points
        ]

    return serialized


def _circularize_fitted_ellipses(ellipses, center, tilt_x_deg, tilt_y_deg):
    """Return circularized radii from the fitted ellipses using tilt correction."""

    if not ellipses or center is None:
        return None

    xc, yc = center
    theta = np.linspace(0.0, 2.0 * np.pi, 720)
    radii = []

    for e in ellipses:
        x_curve = e["xc"] + e["a"] * np.cos(theta) * np.cos(e["theta"]) - e["b"] * np.sin(theta) * np.sin(e["theta"])
        y_curve = e["yc"] + e["a"] * np.cos(theta) * np.sin(e["theta"]) + e["b"] * np.sin(theta) * np.cos(e["theta"])

        pts = np.column_stack([x_curve, y_curve])
        pts_corr = _apply_tilt_xy(pts, center, tilt_x_deg, tilt_y_deg)

        dx = pts_corr[:, 0] - xc
        dy = pts_corr[:, 1] - yc
        r = np.sqrt(dx * dx + dy * dy)
        radii.append(float(np.mean(r)))

    return radii


def _format_ellipse_lines(ellipses):
    lines = []
    for i, e in enumerate(ellipses):
        lines.append(
            "  "
            f"{i}: xc={e['xc']:.2f}, yc={e['yc']:.2f}, "
            f"a={e['a']:.2f}, b={e['b']:.2f}, "
            f"theta={np.degrees(e['theta']):.2f} deg"
        )
    return "\n".join(lines)


def plot_ellipses(img_bgsub, ellipses, save_path=None):
    disp = make_display_image(img_bgsub)
    h, w = disp.shape

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(disp, cmap="gray", origin="upper")

    for e in ellipses:
        x_curve, y_curve = ellipse_curve(
            e["xc"], e["yc"], e["a"], e["b"], e["theta"]
        )
        ax.plot(x_curve, y_curve, "r-", linewidth=1.5)
        ax.scatter(e["xc"], e["yc"], s=20, c="yellow", marker="+")
    if ellipses:
        xc0, yc0 = compute_common_center(ellipses)
        ax.scatter(xc0, yc0, s=40, c="cyan", marker="x")
        print(f"Common center estimate: xc={xc0:.2f}, yc={yc0:.2f}")

        # Show the fitted parameters alongside the real image overlay.
        lines = _format_ellipse_lines(ellipses)
        if lines:
            ax.text(
                0.01,
                0.99,
                "Ellipses (xc, yc, a, b, theta):\n" + lines,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="white",
                bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=6),
            )

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_title(f"{len(ellipses)} ellipses (clicked points, intensity refined)")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        print(f"Saved overlay image with ellipses to:\n  {save_path}")
    plt.show()


def plot_tilt_correction_overlay(
    img_bgsub,
    ellipses,
    ell_points_ds,
    corrected_points,
    center,
    radii_before,
    radii_after,
    tilt_x_deg,
    tilt_y_deg,
    distance_info=None,
    save_path=None,
):
    xc, yc = center
    theta = np.linspace(0.0, 2.0 * np.pi, 720)

    fig, ax = plt.subplots(figsize=(6, 6))

    if img_bgsub is not None:
        disp = make_display_image(img_bgsub)
        ax.imshow(disp, cmap="gray", origin="upper")

    all_pts = []
    for pts in ell_points_ds:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 2 and pts.shape[1] == 2 and pts.size > 0:
            all_pts.append(pts)
    for pts in corrected_points:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 2 and pts.shape[1] == 2 and pts.size > 0:
            all_pts.append(pts)

    if all_pts:
        all_xy = np.vstack(all_pts)
        xmin, xmax = np.min(all_xy[:, 0]), np.max(all_xy[:, 0])
        ymin, ymax = np.min(all_xy[:, 1]), np.max(all_xy[:, 1])
    else:
        xmin = xmax = ymin = ymax = 0.0

    for e in ellipses:
        x_curve, y_curve = ellipse_curve(e["xc"], e["yc"], e["a"], e["b"], e["theta"])
        ax.plot(x_curve, y_curve, "r-", linewidth=1.0, label="_orig_fit")

    for pts in ell_points_ds:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 2 and pts.shape[1] == 2:
            ax.plot(pts[:, 0], pts[:, 1], ".", markersize=1, alpha=0.35, label="_orig_pts")

    for pts in corrected_points:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 2 and pts.shape[1] == 2:
            ax.plot(pts[:, 0], pts[:, 1], ".", markersize=1, alpha=0.7, label="_corr_pts")

    for r in radii_before:
        if not np.isfinite(r) or r <= 0:
            continue
        x_circ = xc + r * np.cos(theta)
        y_circ = yc + r * np.sin(theta)
        ax.plot(x_circ, y_circ, linestyle="--", linewidth=1, color="orange", label="_orig_circle")

    for r in radii_after:
        if not np.isfinite(r) or r <= 0:
            continue
        x_circ = xc + r * np.cos(theta)
        y_circ = yc + r * np.sin(theta)
        ax.plot(x_circ, y_circ, linewidth=1.2, color="cyan", label="_corr_circle")

    ax.axhline(y=yc, linestyle="-.", linewidth=1.0, color="white", label="x tilt axis")
    ax.axvline(x=xc, linestyle="-.", linewidth=1.0, color="white", label="y tilt axis")

    ax.plot(xc, yc, "x", markersize=6, color="yellow", label="center")

    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)

    title = (
        "Tilt-corrected rings on image\n"
        f"tilt_x={tilt_x_deg:.2f} deg, tilt_y={tilt_y_deg:.2f} deg"
    )
    if distance_info:
        mean_m = distance_info.get("mean_m")
        basis = distance_info.get("basis")
        if mean_m is not None:
            basis_label = f" ({basis})" if basis else ""
            title += f"\nshared distance L={mean_m:.4f} m{basis_label}"
    ax.set_title(title)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.legend(loc="best")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        print(f"Saved tilt-corrected overlay to:\n  {save_path}")
    plt.show()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def _resolve_output_dir(output_dir, load_bundle):
    if output_dir:
        resolved = output_dir
    elif load_bundle:
        resolved = os.path.dirname(os.path.abspath(load_bundle))
    else:
        resolved = os.path.join(os.path.expanduser("~"), "Downloads")
    os.makedirs(resolved, exist_ok=True)
    return resolved


def _prompt_save_bundle_path(default_path):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    selected = filedialog.asksaveasfilename(
        title="Save hBN bundle (.npz)",
        defaultextension=".npz",
        filetypes=[("hBN bundle", "*.npz"), ("All files", "*.*")],
        initialdir=os.path.dirname(default_path),
        initialfile=os.path.basename(default_path),
    )
    root.destroy()

    if not selected:
        return None
    if not selected.lower().endswith(".npz"):
        selected = f"{selected}.npz"
    return selected


def run_hbn_fit(
    osc_path,
    dark_path,
    output_dir=None,
    load_bundle=None,
    load_bundle_requested=False,
    highres_refine=False,
    reclick=False,
    reuse_profile=False,
    click_profile_path=None,
    fit_profile_path=None,
    bundle_path=None,
    paths_file=None,
    prompt_save_bundle=False,
    load_clicks=None,
    save_clicks=None,
    clicks_only=False,
    beam_center=None,
    ):
    resolved = resolve_hbn_paths(
        osc_path=osc_path,
        dark_path=dark_path,
        bundle_path=bundle_path,
        click_profile_path=click_profile_path,
        fit_profile_path=fit_profile_path,
        paths_file=paths_file,
    )

    if resolved.get("paths_file"):
        print(f"Loaded hBN paths from file:\n  {resolved['paths_file']}")

    osc_path = resolved["osc"]
    dark_path = resolved["dark"]
    if beam_center is None and resolved.get("beam_center") is not None:
        beam_center = resolved["beam_center"]
    bundle_from_file = resolved["bundle"] if load_bundle_requested else None
    bundle_path_in = None
    if load_bundle_requested:
        bundle_path_in = load_bundle if load_bundle not in (None, "") else None
        if bundle_path_in is None and bundle_from_file and os.path.exists(bundle_from_file):
            bundle_path_in = bundle_from_file
        if bundle_path_in is None:
            raise ValueError(
                "--load-bundle was specified but no bundle path was provided and no "
                "bundle/npz entry was found in the paths file."
            )

    output_dir = _resolve_output_dir(output_dir, bundle_path_in)
    out_tiff_path = os.path.join(output_dir, "hbn_bgsub.tiff")
    out_overlay_path = os.path.join(output_dir, "hbn_bgsub_ellipses.png")
    out_tilt_overlay_path = os.path.join(
        output_dir, "hbn_bgsub_ellipses_tilt_corrected.png"
    )

    click_profile_in = load_clicks or resolved.get("click_profile")
    click_profile_out = None
    if save_clicks is not None or clicks_only:
        click_profile_out = save_clicks
        if click_profile_out in (None, ""):
            click_profile_out = os.path.join(output_dir, "hbn_click_profile.json")

    bundle_path_save = resolved["bundle"] or bundle_path_in
    if bundle_path_save is None:
        bundle_path_save = os.path.join(output_dir, "hbn_ellipse_bundle.npz")

    for p in [bundle_path_save, click_profile_out]:
        if not p:
            continue
        parent = os.path.dirname(os.path.abspath(p))
        os.makedirs(parent, exist_ok=True)

    outputs = {
        "output_dir": output_dir,
        "background_subtracted": out_tiff_path,
        "overlay": out_overlay_path,
        "tilt_overlay": out_tilt_overlay_path,
        "bundle": bundle_path_save,
        "click_profile": click_profile_out,
        "ellipses": [],
        "tilt_hint": None,
        "expected_peaks": None,
        "distance_estimate_m": None,
        "tilt_correction": None,
        "aborted": False,
        "abort_reason": None,
    }

    click_profile_saved = False

    bundle_loaded = None
    center_from_bundle = None
    center_common = None
    if bundle_path_in is not None:
        bundle_loaded = load_bundle_npz(bundle_path_in)
        try:
            center_from_bundle = bundle_loaded[8]
        except Exception:
            center_from_bundle = None
        if center_common is None and center_from_bundle is not None:
            center_common = center_from_bundle

    if beam_center is not None and len(beam_center) == 2:
        center_common = tuple(beam_center)

    # Shared output containers
    img_bgsub_out = None
    img_log_out = None
    ell_points_ds = None
    ellipses_out = None
    distance_info = None
    tilt_correction = None
    tilt_correction_serialized = None
    radii_after_fit = None
    tilt_hint = None
    expected_peaks = None

    abort_reason = None
    completed = False

    try:
        if bundle_loaded is not None and not reclick:
            (
                img_bgsub_b,
                img_log_b,
                ell_points_ds,
                ellipses_b,
                distance_info,
                tilt_correction,
                tilt_hint,
                expected_peaks,
                center_common,
            ) = bundle_loaded
            tilt_correction_serialized = tilt_correction

            if beam_center is not None and len(beam_center) == 2:
                center_common = tuple(beam_center)

            if highres_refine:
                if osc_path is None or dark_path is None:
                    raise ValueError(
                        "Refitting a bundle requires both --osc and --dark so the background "
                        "image can be recomputed."
                    )

                print("Refitting bundle ellipses at full resolution...")
                img_bgsub_out = load_and_bgsub(osc_path, dark_path)
                img_log_out = make_log_image(img_bgsub_out)
            else:
                img_bgsub_out = img_bgsub_b
                img_log_out = img_log_b

            ellipses_out = refine_ellipses_from_existing(
                ellipses_b, img_bgsub_b, img_bgsub_out, img_log_out
            )
        else:
            if reclick and bundle_loaded is not None:
                print(
                    "Reclick requested: ignoring stored clicks in bundle and collecting new points."
                )

            if osc_path is None or dark_path is None:
                raise ValueError(
                    "Both --osc and --dark are required unless --load-bundle is used "
                    "without --highres-refine."
                )

            img_bgsub_out = load_and_bgsub(osc_path, dark_path)

            if reuse_profile and not click_profile_in:
                print(
                    "Reuse profile requested, but no click profile was provided; collecting new clicks."
                )

            if click_profile_in and not reclick:
                if not os.path.exists(click_profile_in):
                    print(
                        "Click profile not found; collecting new clicks instead:\n  "
                        f"{click_profile_in}"
                    )
                    click_profile_in = None
                else:
                    ell_points_ds = load_click_profile(click_profile_in)
                    if outputs["click_profile"] is None:
                        outputs["click_profile"] = click_profile_in

            if ell_points_ds is None:
                print(
                    f"Interactive picking: collect {POINTS_PER_ELLIPSE} points on each of "
                    f"{N_ELLIPSES} rings (left click = point, right drag = zoom, 'r' = reset)."
                )
                small = make_click_image(img_bgsub_out)
                guide_center = center_common or center_from_bundle
                if guide_center is not None:
                    print(
                        "Beam center guide enabled at "
                        f"x={guide_center[0]:.2f}, y={guide_center[1]:.2f}; "
                        f"drawing {N_ELLIPSES} radial guides."
                    )
                ell_points_ds = get_points_per_ellipse_with_zoom(
                    small,
                    n_ellipses=N_ELLIPSES,
                    pts_per_ellipse=POINTS_PER_ELLIPSE,
                    beam_center=guide_center,
                    n_slices=N_ELLIPSES,
                )

            if click_profile_out and ell_points_ds:
                save_click_profile(click_profile_out, ell_points_ds, img_bgsub_out.shape)
                outputs["click_profile"] = click_profile_out
                click_profile_saved = True

            expected_points = N_ELLIPSES * POINTS_PER_ELLIPSE
            collected_points = 0 if ell_points_ds is None else sum(len(pts) for pts in ell_points_ds)

            if clicks_only:
                if collected_points < expected_points:
                    abort_reason = (
                        "Ellipse picking did not finish; collected "
                        f"{collected_points}/{expected_points} points. Skipping save."
                    )
                else:
                    print(
                        "Click-only mode requested; returning after saving collected points."
                    )
                    outputs["clicks_only"] = True
                    outputs["ellipses"] = []
                    outputs["aborted"] = True
                    outputs["abort_reason"] = "Click-only save requested; ellipse fitting skipped."
                    return outputs

            if collected_points < expected_points:
                abort_reason = (
                    "Ellipse picking did not finish; collected "
                    f"{collected_points}/{expected_points} points. Skipping save."
                )
            else:
                ellipses_out = fit_ellipses_from_points(
                    ell_points_ds,
                    img_bgsub=img_bgsub_out,
                )
                print(
                    f"Fitted {len(ellipses_out)} ellipses from clicked points and intensity refinement."
                )

            img_log_out = make_log_image(img_bgsub_out)

        if click_profile_out and ell_points_ds and not clicks_only and not click_profile_saved:
            save_click_profile(click_profile_out, ell_points_ds, img_bgsub_out.shape)
            outputs["click_profile"] = click_profile_out
            click_profile_saved = True

        if abort_reason:
            outputs["aborted"] = True
            outputs["abort_reason"] = abort_reason
            print(abort_reason)
            return outputs

        if ellipses_out and center_common is None:
            center_common = compute_common_center(ellipses_out)
        tilt_correction = (
            _find_tilt_correction(ell_points_ds, center_common)
            if center_common is not None
            else None
        )
        tilt_correction_serialized = _serialize_tilt_correction(
            tilt_correction, center_common
        )
        if expected_peaks is None:
            expected_peaks = hbn_expected_peaks()
        distance_basis = "ellipses"
        ellipses_for_distance = ellipses_out

        if tilt_correction:
            print(
                "Tilt circularization completed: "
                f"tilt_x={tilt_correction['tilt_x_deg']:.4f} deg, "
                f"tilt_y={tilt_correction['tilt_y_deg']:.4f} deg, "
                f"cost={tilt_correction['cost_final']:.6e} (zero={tilt_correction['cost_zero']:.6e})"
            )

        if tilt_correction and center_common is not None:
            corrected_radii_clicks = tilt_correction.get("radii_after", [])
            radii_after_fit = _circularize_fitted_ellipses(
                ellipses_out,
                center_common,
                tilt_correction.get("tilt_x_deg", 0.0),
                tilt_correction.get("tilt_y_deg", 0.0),
            )
            if radii_after_fit is not None:
                tilt_correction["radii_after_fit"] = radii_after_fit

            radii_for_distance = (
                radii_after_fit
                if radii_after_fit is not None and any(np.isfinite(r) for r in radii_after_fit)
                else corrected_radii_clicks
            )

            ellipses_for_distance = [
                dict(xc=center_common[0], yc=center_common[1], a=r, b=r, theta=0.0)
                for r in radii_for_distance
                if np.isfinite(r) and r > 0
            ]
            distance_basis = "circles"

        distance_info = estimate_sample_detector_distance(
            ellipses_for_distance, expected_peaks, basis=distance_basis
        )

        if tilt_correction:
            tilt_correction_serialized = _serialize_tilt_correction(
                tilt_correction, center_common
            )

        tilt_hint_source = "ellipse fit"
        if tilt_hint is None:
            tilt_hint = estimate_detector_tilt(ellipses_out)
        if tilt_correction:
            tilt_hint_source = "circularization"
            tilt_hint = dict(
                rot1_rad=float(np.deg2rad(tilt_correction.get("tilt_x_deg", 0.0))),
                rot2_rad=float(np.deg2rad(tilt_correction.get("tilt_y_deg", 0.0))),
            )
            tilt_hint["tilt_rad"] = float(
                math.hypot(tilt_hint["rot1_rad"], tilt_hint["rot2_rad"])
            )

        if tilt_hint:
            print(
                "Estimated detector tilt from hBN fit (using largest ring): "
                f"Rot1={tilt_hint['rot1_rad']:.4f} rad, Rot2={tilt_hint['rot2_rad']:.4f} rad "
                f"(source={tilt_hint_source})"
            )
        if expected_peaks:
            print("Expected hBN peaks for Cu Kα:")
            for i, peak in enumerate(expected_peaks, 1):
                h, k, l = peak["hkl"]
                print(
                    f"  Ring {i}: hkl=({h}{k}{l}) d={peak['d_spacing_ang']:.4f} Å "
                    f"2θ={peak['two_theta_deg']:.2f}°"
                )
        if distance_info:
            basis_label = distance_info.get("basis", "ellipses")
            print(
                "Estimated sample-detector distance (using matched rings): "
                f"mean={distance_info['mean_m']:.4f} m "
                f"(basis={basis_label})"
            )
            for i, dist in enumerate(distance_info["per_ring_m"], 1):
                print(f"  Ring {i}: {dist:.4f} m")
        completed = True
    except KeyboardInterrupt:
        abort_reason = "hBN fitting interrupted by user; skipping save."

    if not completed:
        outputs["aborted"] = True
        outputs["abort_reason"] = abort_reason
        if abort_reason:
            print(abort_reason)
        return outputs

    print(f"Saving background-subtracted image to:\n  {out_tiff_path}")
    cv2.imwrite(out_tiff_path, img_bgsub_out)
    save_bundle(
        bundle_path_save,
        img_bgsub_out,
        img_log_out,
        ell_points_ds,
        ellipses_out,
        distance_info=distance_info,
        tilt_correction=tilt_correction_serialized,
        tilt_hint=tilt_hint,
        expected_peaks=expected_peaks,
        center=center_common,
    )

    if prompt_save_bundle:
        print("prompt_save_bundle is deprecated; only the primary bundle is written.")

    plot_ellipses(img_bgsub_out, ellipses_out, save_path=out_overlay_path)

    if tilt_correction:
        corrected_points = tilt_correction.get("corrected_points", [])
        radii_before = tilt_correction.get("radii_before", [])
        radii_after = tilt_correction.get("radii_after_fit") or tilt_correction.get(
            "radii_after", []
        )
        plot_tilt_correction_overlay(
            img_bgsub_out,
            ellipses_out,
            ell_points_ds,
            corrected_points,
            center_common,
            radii_before,
            radii_after,
            tilt_correction.get("tilt_x_deg", 0.0),
            tilt_correction.get("tilt_y_deg", 0.0),
            distance_info=distance_info,
            save_path=out_tilt_overlay_path,
        )

    if ellipses_out:
        print("Fitted ellipse parameters:")
        print(_format_ellipse_lines(ellipses_out))

    outputs["ellipses"] = ellipses_out
    outputs["tilt_hint"] = tilt_hint
    outputs["expected_peaks"] = expected_peaks
    outputs["distance_estimate_m"] = distance_info
    outputs["tilt_correction"] = tilt_correction_serialized
    return outputs


def main(argv=None):
    args = parse_args(argv)

    results = run_hbn_fit(
        osc_path=args.osc,
        dark_path=args.dark,
        output_dir=args.output_dir,
        load_bundle=args.load_bundle,
        load_bundle_requested=args.load_bundle is not None,
        highres_refine=args.highres_refine,
        reclick=args.reclick,
        reuse_profile=args.reuse_profile,
        paths_file=args.paths_file,
        prompt_save_bundle=args.prompt_save_bundle,
        load_clicks=args.load_clicks,
        save_clicks=args.save_clicks,
        clicks_only=args.clicks_only,
        beam_center=(args.beam_center_x, args.beam_center_y)
        if args.beam_center_x is not None and args.beam_center_y is not None
        else None,
    )

    if results.get("aborted"):
        reason = results.get("abort_reason") or "early termination"
        print(f"hBN ellipse fitting did not complete: {reason}")
        return

    print("Completed hBN ellipse fitting. Outputs written to:")
    for key in [
        "background_subtracted",
        "overlay",
        "click_profile",
        "bundle",
    ]:
        value = results.get(key, "n/a")
        print(f"  {key.replace('_', ' ').title()}: {value}")


if __name__ == "__main__":
    main()
