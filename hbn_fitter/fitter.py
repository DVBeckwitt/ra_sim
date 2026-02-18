#!/usr/bin/env python3
"""Unified GUI for hBN ring fitting + detector tilt optimization."""

import argparse
import datetime as dt
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import MouseButton
from scipy.optimize import minimize, curve_fit
from skimage.measure import EllipseModel, ransac

from OSC_Reader import OSC_Reader


DEFAULT_NUM_RINGS = 5
DEFAULT_POINTS_PER_RING = 5
DEFAULT_DOWNSAMPLE = 4
DEFAULT_REFINE_N_ANGLES = 360
DEFAULT_REFINE_DR = 10.0
DEFAULT_REFINE_STEP = 1.0
DEFAULT_REFINE_ITERS = 5
DEFAULT_DENSE_POINTS = 720
DEFAULT_LOG_EPS = 1e-3
DEFAULT_CLICK_SEARCH_ALONG = 12.0
DEFAULT_CLICK_SEARCH_ACROSS = 3.0
DEFAULT_CLICK_SEARCH_STEP = 0.5
DEFAULT_SNAP_MIN_PEAK_SNR = 0.35
DEFAULT_SNAP_MIN_POSTERIOR = -2.5
DEFAULT_SNAP_MIN_MARGIN = 0.08
DEFAULT_SNAP_W_PEAK = 1.0
DEFAULT_SNAP_W_RESID = 1.0
DEFAULT_SNAP_W_ACROSS = 1.0
DEFAULT_SNAP_W_CLICK_DIST = 0.65
DEFAULT_SNAP_UNCERT_RADII_PX = (0.0, 1.0, 2.0, 3.0)
DEFAULT_SNAP_UNCERT_N_ANGLES = 8
DEFAULT_SNAP_UNCERT_MIN_PX = 0.20
DEFAULT_SNAP_UNCERT_MAX_PX = 12.0
DEFAULT_PV_MIN_SAMPLES = 7
DEFAULT_PV_MAXFEV = 400
DEFAULT_SUBPIXEL_PATCH_HALF = 1
DEFAULT_PRECISION_PICK_SIZE_DS = 40.0
DEFAULT_EDIT_SELECT_RADIUS_DS = 6.0
DEFAULT_PREVIEW_MIN_INTERVAL_S = 0.03
DEFAULT_PREVIEW_MIN_MOVE_PX = 0.8
DEFAULT_CENTER_DRIFT_LIMIT_PX = 35.0
DEFAULT_CENTER_PRIOR_SIGMA_PX = 12.0
FINAL_REFINE_DOWNSAMPLE = 1
FINAL_REFINE_N_ANGLES_FALLBACK = 1080
FINAL_REFINE_DR_MAX = 8.0
FINAL_REFINE_STEP_MAX = 0.25
FINAL_REFINE_STEP_MIN = 0.12
FINAL_REFINE_ITERS = 8
FINAL_VERIFY_ANGLE_MULT = 1.33
FINAL_VERIFY_STEP_MULT = 0.75
FINAL_VERIFY_MAX_ANGLES = 1800
FAST_CALIB_MAX_ANGLES = 240
FAST_CALIB_MIN_STEP = 0.75
FAST_CALIB_MAX_DR = 8.0
FAST_CALIB_ITERS_COLD = 3
FAST_CALIB_ITERS_WARM = 2
OPT_GRID_SIZE_COLD = 31
OPT_GRID_SPAN_COLD = 10.0
OPT_GRID_SIZE_WARM = 17
OPT_GRID_SPAN_WARM = 4.0
OPT_NELDER_MAXITER = 500
OPT_POWELL_MAXITER = 250
PROJECTIVE_TILT_MAX_DEG = 25.0
PROJECTIVE_DIST_MIN_PX = 120.0
PROJECTIVE_DIST_MAX_MULT = 20.0
PROJECTIVE_DIST_PRIOR_SIGMA_LOG = 0.9

# Keep hBN fitter inputs in native OSC orientation (no np.rot90).
SIM_BACKGROUND_ROTATE_K = 0
# Area-detector CCW-positive convention for exported simulation rotations:
# - left/right tilt -> +Gamma when CCW viewed from image top
# - up/down tilt   -> +gamma when CCW viewed from image left->right
SIM_GAMMA_SIGN_FROM_TILT_X = 1
SIM_GAMMA_SIGN_FROM_TILT_Y = 1


def safe_int(text, default, minimum=1):
    try:
        value = int(str(text).strip())
    except Exception:
        value = default
    if value < minimum:
        return int(default)
    return int(value)


def safe_float(text, default):
    try:
        return float(str(text).strip())
    except Exception:
        return float(default)


def npz_scalar(data, key, default=np.nan):
    if key not in data:
        return default
    arr = np.asarray(data[key]).reshape(-1)
    if arr.size == 0:
        return default
    try:
        return float(arr[0])
    except Exception:
        return default


def npz_string(data, key, default=""):
    if key not in data:
        return default
    arr = np.asarray(data[key]).reshape(-1)
    if arr.size == 0:
        return default
    return str(arr[0])


def pts_to_obj(points):
    return np.array([np.asarray(p, dtype=np.float32).reshape(-1, 2) for p in points], dtype=object)


def scalars_to_obj(values):
    return np.array([np.asarray(v, dtype=np.float32).reshape(-1) for v in values], dtype=object)


def obj_to_pts_list(obj):
    out = []
    for p in obj:
        arr = np.asarray(p, dtype=float).reshape(-1, 2)
        out.append([(float(x), float(y)) for x, y in arr])
    return out


def obj_to_scalar_lists(obj):
    out = []
    for v in obj:
        arr = np.asarray(v, dtype=float).reshape(-1)
        out.append([float(x) for x in arr])
    return out


def obj_to_ndarrays(obj):
    out = []
    for p in obj:
        arr = np.asarray(p, dtype=float).reshape(-1, 2)
        if arr.size:
            out.append(arr)
    return out


def ellipses_to_array(ellipses):
    arr = np.zeros((len(ellipses), 5), dtype=np.float32)
    for i, e in enumerate(ellipses):
        arr[i] = [e["xc"], e["yc"], e["a"], e["b"], e["theta"]]
    return arr


def array_to_ellipses(arr):
    arr = np.asarray(arr, dtype=float).reshape(-1, 5)
    return [
        {"xc": float(r[0]), "yc": float(r[1]), "a": float(r[2]), "b": float(r[3]), "theta": float(r[4])}
        for r in arr
    ]


def ellipse_ring_indices(ellipses):
    out = []
    for i, e in enumerate(ellipses, start=1):
        out.append(int(e.get("ring_index", i)))
    return np.asarray(out, dtype=np.int32)


def apply_ellipse_ring_indices(ellipses, ring_indices):
    if ring_indices is None:
        for i, e in enumerate(ellipses, start=1):
            e["ring_index"] = int(i)
        return ellipses

    rid = np.asarray(ring_indices, dtype=int).reshape(-1)
    for i, e in enumerate(ellipses, start=1):
        if i - 1 < rid.size:
            e["ring_index"] = int(rid[i - 1])
        else:
            e["ring_index"] = int(i)
    return ellipses


def load_and_bgsub(hbn_path, dark_path):
    dark = OSC_Reader.read_osc(dark_path).astype(np.float32)
    raw = OSC_Reader.read_osc(hbn_path).astype(np.float32)
    main = np.clip(raw - dark, 0, None)
    bg = cv2.GaussianBlur(main, ksize=(0, 0), sigmaX=25, sigmaY=25)
    return np.clip(main - bg, 0, None).astype(np.float32)


def make_log_image(img, eps=DEFAULT_LOG_EPS):
    return np.log(np.clip(img, 0, None).astype(np.float32) + eps)


def build_display(img_log, downsample):
    lo, hi = np.percentile(img_log, (0.1, 99.9))
    disp = np.clip((img_log - lo) / (hi - lo + 1e-9), 0, 1)
    if downsample <= 1:
        return disp
    h, w = disp.shape
    return cv2.resize(disp, (max(1, w // downsample), max(1, h // downsample)), interpolation=cv2.INTER_AREA)


def ellipse_residuals_px(points, ellipse):
    arr = np.asarray(points, dtype=float).reshape(-1, 2)
    if arr.shape[0] == 0:
        return np.array([], dtype=float)

    if isinstance(ellipse, dict):
        xc = float(ellipse["xc"])
        yc = float(ellipse["yc"])
        a = max(abs(float(ellipse["a"])), 1e-6)
        b = max(abs(float(ellipse["b"])), 1e-6)
        th = float(ellipse["theta"])
    else:
        xc, yc, a, b, th = [float(v) for v in ellipse]
        a = max(abs(a), 1e-6)
        b = max(abs(b), 1e-6)

    c, s = np.cos(th), np.sin(th)
    dx = arr[:, 0] - xc
    dy = arr[:, 1] - yc
    u = dx * c + dy * s
    v = -dx * s + dy * c
    q = np.sqrt((u * u) / (a * a) + (v * v) / (b * b) + 1e-12)
    resid_unit = np.abs(q - 1.0)
    return resid_unit * (0.5 * (a + b))


def robust_fit_ellipse(points, residual_threshold_px=2.5, max_trials=300):
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if pts.shape[0] < 5:
        return None, np.zeros(pts.shape[0], dtype=bool)

    params = None
    inliers = np.ones(pts.shape[0], dtype=bool)

    try:
        model, inlier_mask = ransac(
            pts,
            EllipseModel,
            min_samples=5,
            residual_threshold=float(residual_threshold_px),
            max_trials=int(max_trials),
        )
        if model is not None and inlier_mask is not None and np.sum(inlier_mask) >= 5:
            params = tuple(float(v) for v in model.params)
            inliers = np.asarray(inlier_mask, dtype=bool)
    except Exception:
        params = None

    if params is None:
        model = EllipseModel()
        if not model.estimate(pts):
            return None, np.zeros(pts.shape[0], dtype=bool)
        params = tuple(float(v) for v in model.params)
        inliers = np.ones(pts.shape[0], dtype=bool)

    # Two quick sigma-clipping passes after RANSAC to suppress residual outliers.
    for _ in range(2):
        resid = ellipse_residuals_px(pts, params)
        if resid.size == 0:
            break
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        sigma = max(1.4826 * mad, 1e-6)
        thr = max(float(residual_threshold_px), 3.0 * sigma)
        keep = resid <= thr
        if np.sum(keep) < 5:
            break
        model = EllipseModel()
        if not model.estimate(pts[keep]):
            break
        params = tuple(float(v) for v in model.params)
        inliers = keep

    return params, inliers


def _wrap_angle_pi(theta):
    return float((float(theta) + np.pi) % (2.0 * np.pi) - np.pi)


def _sanitize_point_sigma(point_sigma_px, n, default_sigma=1.0):
    n = int(max(int(n), 0))
    if n == 0:
        return np.array([], dtype=float)
    arr = np.asarray(point_sigma_px, dtype=float).reshape(-1) if point_sigma_px is not None else np.array([], dtype=float)
    out = np.full(n, np.nan, dtype=float)
    m = min(n, arr.size)
    if m > 0:
        out[:m] = arr[:m]
    valid = np.isfinite(out) & (out > 0.0)
    if np.any(valid):
        ref = float(np.median(out[valid]))
    else:
        ref = float(default_sigma)
    ref = float(np.clip(ref, 0.20, 10.0))
    out[~valid] = ref
    out = np.clip(out, 0.15, 20.0)
    return out


def weighted_refine_ellipse(points, params0, point_sigma_px=None, maxiter=120):
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if pts.shape[0] < 5:
        return tuple(float(v) for v in params0)

    x0, y0, a0, b0, th0 = [float(v) for v in params0]
    a0 = max(abs(a0), 1e-3)
    b0 = max(abs(b0), 1e-3)
    th0 = _wrap_angle_pi(th0)

    sig = _sanitize_point_sigma(point_sigma_px, pts.shape[0], default_sigma=1.0)
    w = 1.0 / np.clip(sig, 0.15, 20.0) ** 2
    w = w / max(float(np.median(w[np.isfinite(w)])) if np.any(np.isfinite(w)) else 1.0, 1e-6)
    delta = float(np.clip(np.median(sig), 0.6, 4.0))
    span = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 4.0))

    def obj(z):
        xc, yc, loga, logb, th = [float(v) for v in z]
        a = max(np.exp(loga), 1e-4)
        b = max(np.exp(logb), 1e-4)
        th = _wrap_angle_pi(th)
        if not (np.isfinite(xc) and np.isfinite(yc) and np.isfinite(a) and np.isfinite(b) and np.isfinite(th)):
            return np.inf

        resid = ellipse_residuals_px(pts, (xc, yc, a, b, th))
        if resid.size != pts.shape[0]:
            return np.inf
        u = resid / max(delta, 1e-6)
        rho = (np.sqrt(1.0 + u * u) - 1.0) * (delta * delta)
        loss = float(np.mean(w * rho))

        # Keep the local weighted refinement near the robust seed.
        reg_center = 2e-4 * (((xc - x0) / max(span, 1.0)) ** 2 + ((yc - y0) / max(span, 1.0)) ** 2)
        reg_axes = 1e-4 * (((np.log(a / a0)) ** 2) + ((np.log(b / b0)) ** 2))
        return float(loss + reg_center + reg_axes)

    z0 = np.array([x0, y0, np.log(a0), np.log(b0), th0], dtype=float)
    try:
        res = minimize(
            obj,
            z0,
            method="Powell",
            options={"maxiter": int(maxiter), "xtol": 1e-3, "ftol": 1e-8, "disp": False},
        )
        if res.success and np.all(np.isfinite(res.x)):
            xc, yc, loga, logb, th = [float(v) for v in res.x]
            a = max(np.exp(loga), 1e-4)
            b = max(np.exp(logb), 1e-4)
            if np.isfinite(xc) and np.isfinite(yc) and np.isfinite(a) and np.isfinite(b):
                return (float(xc), float(yc), float(a), float(b), _wrap_angle_pi(th))
    except Exception:
        pass
    return (float(x0), float(y0), float(a0), float(b0), float(th0))


def fit_initial_ellipse(pts, point_sigma_px=None):
    params, inliers = robust_fit_ellipse(pts, residual_threshold_px=3.0, max_trials=250)
    if params is None:
        x = pts[:, 0]
        y = pts[:, 1]
        params = (
            float(x.mean()),
            float(y.mean()),
            max(float((x.max() - x.min()) / 2), 1.0),
            max(float((y.max() - y.min()) / 2), 1.0),
            0.0,
        )
    return weighted_refine_ellipse(pts, params, point_sigma_px=point_sigma_px, maxiter=80)


def bilinear_sample(img, x, y):
    h, w = img.shape
    if x < 0 or x > w - 1 or y < 0 or y > h - 1:
        return 0.0
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0
    v00, v10 = img[y0, x0], img[y0, x1]
    v01, v11 = img[y1, x0], img[y1, x1]
    return v00 * (1 - dx) * (1 - dy) + v10 * dx * (1 - dy) + v01 * (1 - dx) * dy + v11 * dx * dy


def pseudo_voigt_1d(x, amp, x0, sigma, eta, baseline):
    sigma = np.maximum(np.abs(sigma), 1e-6)
    u = (x - x0) / sigma
    gauss = np.exp(-0.5 * u * u)
    lorentz = 1.0 / (1.0 + u * u)
    return baseline + amp * (eta * lorentz + (1.0 - eta) * gauss)


def subpixel_quadratic_peak_1d(x_vals, y_vals, x_hint=None):
    x = np.asarray(x_vals, dtype=float).reshape(-1)
    y = np.asarray(y_vals, dtype=float).reshape(-1)
    if x.size < 3 or y.size < 3 or x.size != y.size:
        return np.nan, np.nan, False

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return np.nan, np.nan, False

    if x_hint is None or not np.isfinite(float(x_hint)):
        i = int(np.argmax(y))
    else:
        i = int(np.argmin(np.abs(x - float(x_hint))))
    i = int(np.clip(i, 1, x.size - 2))

    xx = np.asarray(x[i - 1:i + 2], dtype=float)
    yy = np.asarray(y[i - 1:i + 2], dtype=float)
    if xx.size != 3 or yy.size != 3:
        return np.nan, np.nan, False

    try:
        p = np.polyfit(xx, yy, 2)
    except Exception:
        return np.nan, np.nan, False

    a, b, c = [float(v) for v in p]
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)) or abs(a) < 1e-12 or a >= 0:
        return float(xx[1]), float(yy[1]), False

    x_peak = float(-0.5 * b / a)
    x_lo = float(np.min(xx))
    x_hi = float(np.max(xx))
    x_peak = float(np.clip(x_peak, x_lo, x_hi))
    y_peak = float(np.polyval(p, x_peak))
    if not (np.isfinite(x_peak) and np.isfinite(y_peak)):
        return np.nan, np.nan, False
    return x_peak, y_peak, True


def subpixel_centroid_2d(img, x, y, half_window=DEFAULT_SUBPIXEL_PATCH_HALF):
    arr = np.asarray(img, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return float(x), float(y), False

    hw = int(max(int(half_window), 1))
    h, w = arr.shape
    ix = int(np.round(float(x)))
    iy = int(np.round(float(y)))
    if ix - hw < 0 or ix + hw >= w or iy - hw < 0 or iy + hw >= h:
        return float(x), float(y), False

    patch = arr[iy - hw:iy + hw + 1, ix - hw:ix + hw + 1]
    if patch.size == 0:
        return float(x), float(y), False

    base = float(np.min(patch))
    weights = np.clip(patch - base, 0.0, None)
    sw = float(np.sum(weights))
    if sw <= 1e-12 or not np.isfinite(sw):
        return float(x), float(y), False

    xs = np.arange(ix - hw, ix + hw + 1, dtype=float)
    ys = np.arange(iy - hw, iy + hw + 1, dtype=float)
    xx, yy = np.meshgrid(xs, ys)

    xr = float(np.sum(weights * xx) / sw)
    yr = float(np.sum(weights * yy) / sw)
    if not (np.isfinite(xr) and np.isfinite(yr)):
        return float(x), float(y), False
    return xr, yr, True


def find_profile_peak_pseudovoigt(s_vals, y_vals):
    s = np.asarray(s_vals, dtype=float).reshape(-1)
    y = np.asarray(y_vals, dtype=float).reshape(-1)
    if s.size == 0 or y.size == 0 or s.size != y.size:
        return np.nan, np.nan, False

    m = np.isfinite(s) & np.isfinite(y)
    s = s[m]
    y = y[m]
    if s.size < DEFAULT_PV_MIN_SAMPLES:
        if s.size == 0:
            return np.nan, np.nan, False
        i = int(np.argmax(y))
        sq, yq, ok_q = subpixel_quadratic_peak_1d(s, y, x_hint=float(s[i]))
        if ok_q:
            return float(sq), float(yq), False
        return float(s[i]), float(y[i]), False

    i_max = int(np.argmax(y))
    x0_guess = float(s[i_max])
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    amp_guess = max(y_max - y_min, 1e-6)
    dx = np.median(np.diff(s)) if s.size > 1 else 1.0
    span = max(float(s[-1] - s[0]), abs(dx))
    sigma_guess = max(abs(dx) * 2.0, span / 12.0)
    baseline_guess = float(np.percentile(y, 20))

    p0 = [amp_guess, x0_guess, sigma_guess, 0.5, baseline_guess]
    lower = [0.0, float(s[0]), max(abs(dx) * 0.25, 1e-6), 0.0, y_min - amp_guess]
    upper = [amp_guess * 4.0 + 1e-6, float(s[-1]), span * 0.8 + abs(dx), 1.0, y_max + amp_guess]

    try:
        popt, _ = curve_fit(
            pseudo_voigt_1d,
            s,
            y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=DEFAULT_PV_MAXFEV,
        )
        amp, x0, sigma, eta, baseline = [float(v) for v in popt]
        x0 = float(np.clip(x0, s[0], s[-1]))
        sq, yq, ok_q = subpixel_quadratic_peak_1d(s, y, x_hint=x0)
        if ok_q:
            step = np.median(np.diff(s)) if s.size > 1 else 1.0
            tol = max(2.0 * abs(float(step)), 1e-6)
            if abs(float(sq) - x0) <= tol:
                x0 = float(np.clip(sq, s[0], s[-1]))
        y0 = float(pseudo_voigt_1d(np.array([x0], dtype=float), amp, x0, sigma, eta, baseline)[0])
        if ok_q and np.isfinite(yq):
            y0 = float(max(y0, yq))
        return x0, y0, True
    except Exception:
        i = int(np.argmax(y))
        sq, yq, ok_q = subpixel_quadratic_peak_1d(s, y, x_hint=float(s[i]))
        if ok_q:
            return float(sq), float(yq), False
        return float(s[i]), float(y[i]), False


def find_profile_peak_fast(s_vals, y_vals):
    s = np.asarray(s_vals, dtype=float).reshape(-1)
    y = np.asarray(y_vals, dtype=float).reshape(-1)
    if s.size == 0 or y.size == 0 or s.size != y.size:
        return np.nan, np.nan, False

    m = np.isfinite(s) & np.isfinite(y)
    s = s[m]
    y = y[m]
    if s.size == 0:
        return np.nan, np.nan, False

    i = int(np.argmax(y))
    sq, yq, ok_q = subpixel_quadratic_peak_1d(s, y, x_hint=float(s[i]))
    if ok_q:
        return float(sq), float(yq), True
    return float(s[i]), float(y[i]), False


def snap_points_to_ring(
    img_log,
    points_full,
    params,
    search_along=DEFAULT_CLICK_SEARCH_ALONG,
    search_across=DEFAULT_CLICK_SEARCH_ACROSS,
    search_step=DEFAULT_CLICK_SEARCH_STEP,
    use_pseudovoigt=True,
    enforce_confidence=False,
    min_peak_snr=DEFAULT_SNAP_MIN_PEAK_SNR,
    min_posterior=DEFAULT_SNAP_MIN_POSTERIOR,
    min_margin=DEFAULT_SNAP_MIN_MARGIN,
    score_w_peak=DEFAULT_SNAP_W_PEAK,
    score_w_resid=DEFAULT_SNAP_W_RESID,
    score_w_across=DEFAULT_SNAP_W_ACROSS,
    score_w_click_dist=DEFAULT_SNAP_W_CLICK_DIST,
    return_meta=False,
):
    pts = np.asarray(points_full, dtype=float).reshape(-1, 2)
    if pts.shape[0] == 0:
        empty = np.empty((0, 2), dtype=float)
        return (empty, []) if return_meta else empty

    h, w = img_log.shape
    xc, yc, a, b, theta = [float(v) for v in params]
    scale = max(0.5 * (abs(a) + abs(b)), 1.0)
    sigma_resid = max(1.5, 0.03 * scale, 0.25 * max(abs(float(search_step)), 1e-6))
    sigma_v = max(0.75, 0.45 * max(float(search_across), 0.5))
    sigma_d = max(1.0, 0.55 * max(float(search_along), 1.0))

    snapped = []
    meta = []
    along = np.arange(-search_along, search_along + search_step, search_step)
    across = np.arange(-search_across, search_across + search_step, search_step)

    for px, py in pts:
        vx = px - xc
        vy = py - yc
        vr = np.hypot(vx, vy)
        if vr < 1e-6:
            snapped.append((float(px), float(py)))
            meta.append(
                {
                    "used_snap": False,
                    "reason": "near_center",
                    "best_posterior": np.nan,
                    "posterior_margin": np.nan,
                    "best_peak_snr": np.nan,
                    "best_resid_px": np.nan,
                    "best_tangent_offset_px": np.nan,
                    "best_click_distance_px": np.nan,
                    "candidate_count": 0,
                }
            )
            continue
        nx = vx / vr
        ny = vy / vr
        tx = -ny
        ty = nx

        candidates = []

        for v in across:
            u_valid = []
            y_valid = []
            for u in along:
                x = px + u * nx + v * tx
                y = py + u * ny + v * ty
                if x < 1 or x > w - 2 or y < 1 or y > h - 2:
                    continue
                u_valid.append(float(u))
                y_valid.append(float(bilinear_sample(img_log, x, y)))

            if len(u_valid) < DEFAULT_PV_MIN_SAMPLES:
                continue

            y_prof = np.asarray(y_valid, dtype=float)
            y_med = float(np.median(y_prof))
            y_mad = float(np.median(np.abs(y_prof - y_med)))
            y_sigma = max(1.4826 * y_mad, 1e-6)

            if use_pseudovoigt:
                u_peak, y_peak, _ = find_profile_peak_pseudovoigt(u_valid, y_valid)
            else:
                u_peak, y_peak, _ = find_profile_peak_fast(u_valid, y_valid)
            if not np.isfinite(u_peak):
                continue
            x_peak = px + u_peak * nx + v * tx
            y_peak_xy = py + u_peak * ny + v * ty
            x_peak, y_peak_xy, _ = subpixel_centroid_2d(img_log, x_peak, y_peak_xy)
            if x_peak < 1 or x_peak > w - 2 or y_peak_xy < 1 or y_peak_xy > h - 2:
                continue

            resid = ellipse_residuals_px(
                np.array([[x_peak, y_peak_xy]], dtype=float),
                (xc, yc, a, b, theta),
            )[0]
            click_dist = float(np.hypot(float(u_peak), float(v)))
            peak_snr = float((float(y_peak) - y_med) / y_sigma)
            log_post = (
                float(score_w_peak) * peak_snr
                - 0.5 * float(score_w_resid) * ((float(resid) / sigma_resid) ** 2)
                - 0.5 * float(score_w_across) * ((float(v) / sigma_v) ** 2)
                - 0.5 * float(score_w_click_dist) * ((click_dist / sigma_d) ** 2)
            )

            candidates.append(
                {
                    "x": float(x_peak),
                    "y": float(y_peak_xy),
                    "posterior": float(log_post),
                    "peak_snr": float(peak_snr),
                    "resid_px": float(resid),
                    "tangent_offset_px": float(v),
                    "click_distance_px": float(click_dist),
                }
            )

        if not candidates:
            snapped.append((float(px), float(py)))
            meta.append(
                {
                    "used_snap": False,
                    "reason": "no_candidates",
                    "best_posterior": np.nan,
                    "posterior_margin": np.nan,
                    "best_peak_snr": np.nan,
                    "best_resid_px": np.nan,
                    "best_tangent_offset_px": np.nan,
                    "best_click_distance_px": np.nan,
                    "candidate_count": 0,
                }
            )
            continue

        candidates.sort(key=lambda c: c["posterior"], reverse=True)
        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None
        posterior_margin = (
            float(best["posterior"] - second["posterior"]) if second is not None else np.inf
        )

        use_snap = True
        reason = "accepted"
        if enforce_confidence:
            resid_limit = max(2.5, 0.02 * scale)
            dist_limit = max(1.5, 0.90 * float(search_along))
            alt_sep_limit = max(1.25, 1.5 * abs(float(search_step)))
            if float(best["posterior"]) < float(min_posterior):
                use_snap = False
                reason = "low_posterior"
            elif np.isfinite(posterior_margin) and float(posterior_margin) < float(min_margin):
                # Margins can be small on broad/flat rings; only reject if
                # the alternatives are materially separated and geometry is weak.
                alt_sep = (
                    float(np.hypot(float(best["x"]) - float(second["x"]), float(best["y"]) - float(second["y"])))
                    if second is not None
                    else 0.0
                )
                is_geom_weak = (
                    float(best["resid_px"]) > float(resid_limit)
                    and float(best["click_distance_px"]) > float(dist_limit)
                )
                if np.isfinite(alt_sep) and float(alt_sep) > float(alt_sep_limit) and is_geom_weak:
                    use_snap = False
                    reason = "ambiguous_posterior"

            # Ring-constrained snapping should not fail just because signal contrast is modest.
            # Only reject weak-SNR candidates when they are also geometrically implausible.
            if use_snap and (
                float(best["peak_snr"]) < float(min_peak_snr)
                and float(best["resid_px"]) > float(resid_limit)
                and float(best["click_distance_px"]) > float(dist_limit)
            ):
                use_snap = False
                reason = "low_peak_snr"

        if use_snap:
            snapped.append((float(best["x"]), float(best["y"])))
        else:
            snapped.append((float(px), float(py)))

        meta.append(
            {
                "used_snap": bool(use_snap),
                "reason": str(reason),
                "best_posterior": float(best["posterior"]),
                "posterior_margin": float(posterior_margin),
                "best_peak_snr": float(best["peak_snr"]),
                "best_resid_px": float(best["resid_px"]),
                "best_tangent_offset_px": float(best["tangent_offset_px"]),
                "best_click_distance_px": float(best["click_distance_px"]),
                "candidate_count": int(len(candidates)),
            }
        )

    snapped_arr = np.asarray(snapped, dtype=float).reshape(-1, 2)
    if return_meta:
        return snapped_arr, meta
    return snapped_arr


def refine_ellipse(
    img_log,
    params,
    n_angles,
    dr,
    step,
    n_iter=DEFAULT_REFINE_ITERS,
    seed_points=None,
    seed_sigma_px=None,
    use_pseudovoigt=True,
):
    current = tuple(float(v) for v in params)
    h, w = img_log.shape

    for _ in range(max(int(n_iter), 1)):
        xc, yc, a, b, theta = current
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        peaks = []

        for t in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
            ct, st = np.cos(t), np.sin(t)
            xe = xc + a * ct * cos_t - b * st * sin_t
            ye = yc + a * ct * sin_t + b * st * cos_t
            dx, dy = xe - xc, ye - yc
            r0 = np.hypot(dx, dy)
            if r0 < 1e-6:
                continue
            dx, dy = dx / r0, dy / r0

            s_vals = np.arange(-dr, dr + step, step)
            s_valid = []
            y_valid = []
            for s in s_vals:
                r = r0 + s
                x = xc + r * dx
                y = yc + r * dy
                if x < 1 or x > w - 2 or y < 1 or y > h - 2:
                    continue
                s_valid.append(float(s))
                y_valid.append(float(bilinear_sample(img_log, x, y)))

            if len(s_valid) >= DEFAULT_PV_MIN_SAMPLES:
                if use_pseudovoigt:
                    s_peak, _, _ = find_profile_peak_pseudovoigt(s_valid, y_valid)
                else:
                    s_peak, _, _ = find_profile_peak_fast(s_valid, y_valid)
                if np.isfinite(s_peak):
                    r_best = r0 + s_peak
                else:
                    i = int(np.argmax(np.asarray(y_valid, dtype=float)))
                    r_best = r0 + s_valid[i]
            elif len(s_valid) > 0:
                i = int(np.argmax(np.asarray(y_valid, dtype=float)))
                r_best = r0 + s_valid[i]
            else:
                r_best = r0

            best_x = xc + r_best * dx
            best_y = yc + r_best * dy
            best_x, best_y, _ = subpixel_centroid_2d(img_log, best_x, best_y)
            peaks.append((best_x, best_y))

        peaks = np.asarray(peaks, dtype=float)
        if peaks.shape[0] < 5:
            break

        fit_points = peaks
        fit_sigma = None
        if seed_points is not None:
            anchors = snap_points_to_ring(
                img_log,
                seed_points,
                current,
                search_along=DEFAULT_CLICK_SEARCH_ALONG,
                search_across=DEFAULT_CLICK_SEARCH_ACROSS,
                search_step=DEFAULT_CLICK_SEARCH_STEP,
                use_pseudovoigt=use_pseudovoigt,
            )
            if anchors.shape[0] >= 5:
                if seed_sigma_px is not None:
                    # Use pointwise snap uncertainty in the ellipse update.
                    s_anchor = _sanitize_point_sigma(seed_sigma_px, anchors.shape[0], default_sigma=1.0)
                    s_peak = np.full(peaks.shape[0], float(np.median(s_anchor)), dtype=float)
                    fit_points = np.vstack([peaks, anchors])
                    fit_sigma = np.r_[s_peak, s_anchor]
                else:
                    # Backward-compatible behavior when no uncertainty is available.
                    fit_points = np.vstack([peaks, anchors, anchors, anchors])
                    fit_sigma = None

        # Residual threshold scales mildly with search window, then robustly clips.
        ransac_thr = max(2.0, 0.35 * float(dr))
        new_params, inliers = robust_fit_ellipse(fit_points, residual_threshold_px=ransac_thr, max_trials=300)
        if new_params is None:
            break
        new_params = weighted_refine_ellipse(
            fit_points,
            new_params,
            point_sigma_px=fit_sigma,
            maxiter=90,
        )

        prev = np.asarray(current, dtype=float)
        nxt = np.asarray(new_params, dtype=float)
        current = tuple(float(v) for v in nxt)

        # Early stop when ellipse update is small.
        if np.linalg.norm(nxt[:2] - prev[:2]) < 0.05 and np.linalg.norm(nxt[2:4] - prev[2:4]) < 0.05:
            break

    return current


def fit_ellipses(
    points_ds,
    downsample,
    img_bgsub,
    n_angles,
    dr,
    step,
    initial_ellipses=None,
    img_log=None,
    n_iter=DEFAULT_REFINE_ITERS,
    use_pseudovoigt=True,
    point_sigma_ds=None,
):
    if img_log is None:
        img_log = make_log_image(img_bgsub)
    out = []
    prior_by_ring = {}
    for i, e in enumerate(initial_ellipses or [], start=1):
        try:
            rid = int(e.get("ring_index", i))
            params = (
                float(e["xc"]),
                float(e["yc"]),
                float(e["a"]),
                float(e["b"]),
                float(e["theta"]),
            )
        except Exception:
            continue
        if not (np.isfinite(params[0]) and np.isfinite(params[1]) and np.isfinite(params[2]) and np.isfinite(params[3]) and np.isfinite(params[4])):
            continue
        if params[2] <= 0.0 or params[3] <= 0.0:
            continue
        prior_by_ring[rid] = params

    for ring_idx, pts in enumerate(points_ds, start=1):
        arr = np.asarray(pts, dtype=float)
        if arr.shape[0] < 5:
            continue
        arr_full = arr * float(downsample)
        ring_sigma = None
        if point_sigma_ds is not None and ring_idx - 1 < len(point_sigma_ds):
            s_arr = np.asarray(point_sigma_ds[ring_idx - 1], dtype=float).reshape(-1)
            if s_arr.size > 0:
                # Incoming sigma is in fit-space px when called from GUI fit path.
                ring_sigma = _sanitize_point_sigma(
                    s_arr * float(downsample),
                    arr_full.shape[0],
                    default_sigma=1.0,
                )
        p0 = prior_by_ring.get(int(ring_idx))
        if p0 is not None:
            resid0 = ellipse_residuals_px(arr_full, p0)
            med_resid0 = float(np.median(resid0)) if resid0.size else np.inf
            ring_scale = max(0.5 * (abs(float(p0[2])) + abs(float(p0[3]))), 1.0)
            if (not np.isfinite(med_resid0)) or med_resid0 > max(30.0, 0.45 * ring_scale):
                p0 = None
        if p0 is None:
            p0 = fit_initial_ellipse(arr_full, point_sigma_px=ring_sigma)

        snapped0 = snap_points_to_ring(
            img_log,
            arr_full,
            p0,
            search_along=DEFAULT_CLICK_SEARCH_ALONG,
            search_across=DEFAULT_CLICK_SEARCH_ACROSS,
            search_step=DEFAULT_CLICK_SEARCH_STEP,
            use_pseudovoigt=use_pseudovoigt,
        )
        if snapped0.shape[0] >= 5:
            p1 = fit_initial_ellipse(snapped0, point_sigma_px=ring_sigma)
        else:
            p1 = p0

        xc, yc, a, b, theta = refine_ellipse(
            img_log,
            p1,
            n_angles,
            dr,
            step,
            n_iter=n_iter,
            seed_points=arr_full,
            seed_sigma_px=ring_sigma,
            use_pseudovoigt=use_pseudovoigt,
        )
        out.append({"xc": xc, "yc": yc, "a": a, "b": b, "theta": theta, "ring_index": int(ring_idx)})
    return out


def ellipse_curve(xc, yc, a, b, theta, num=400):
    t = np.linspace(0, 2 * np.pi, num, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    c, s = np.cos(theta), np.sin(theta)
    x = xc + a * ct * c - b * st * s
    y = yc + a * ct * s + b * st * c
    return x, y


def ellipse_center(ellipses, fit_quality=None):
    if not ellipses:
        return np.nan, np.nan

    xs = np.asarray([float(e["xc"]) for e in ellipses], dtype=float)
    ys = np.asarray([float(e["yc"]) for e in ellipses], dtype=float)
    ring_ids = np.asarray([int(e.get("ring_index", i + 1)) for i, e in enumerate(ellipses)], dtype=int)

    weights = np.ones(xs.shape[0], dtype=float)
    if fit_quality:
        conf = np.asarray(fit_quality.get("per_ring_confidence", np.array([], dtype=float)), dtype=float).reshape(-1)
        conf_rings = np.asarray(fit_quality.get("ring_indices", np.array([], dtype=int)), dtype=int).reshape(-1)
        conf_map = {int(r): float(c) for r, c in zip(conf_rings, conf)}
        for i, rid in enumerate(ring_ids):
            c = conf_map.get(int(rid), np.nan)
            if np.isfinite(c):
                weights[i] = np.clip(c / 100.0, 0.2, 1.0)

    cx0 = np.median(xs)
    cy0 = np.median(ys)
    d = np.hypot(xs - cx0, ys - cy0)
    med = np.median(d)
    mad = np.median(np.abs(d - med))
    sigma = max(1.4826 * mad, 1e-6)
    keep = d <= (med + 2.5 * sigma)
    if not np.any(keep):
        keep = np.ones_like(d, dtype=bool)

    w = np.maximum(weights[keep], 1e-6)
    cx = float(np.average(xs[keep], weights=w))
    cy = float(np.average(ys[keep], weights=w))
    return cx, cy


def sample_ellipses(ellipses, n=DEFAULT_DENSE_POINTS):
    out = []
    for e in ellipses:
        x, y = ellipse_curve(e["xc"], e["yc"], e["a"], e["b"], e["theta"], num=n)
        out.append(np.column_stack([x, y]))
    return out


def ellipse_residual_px(points, ellipse):
    resid_px = ellipse_residuals_px(points, ellipse)
    if resid_px.size == 0:
        return np.nan

    med = np.median(resid_px)
    mad = np.median(np.abs(resid_px - med))
    robust_sigma = 1.4826 * mad
    rms = np.sqrt(np.mean(resid_px * resid_px))
    if np.isfinite(robust_sigma) and robust_sigma > 1e-6:
        return float(robust_sigma)
    return float(rms)


def ring_signal_snr(img_log, ellipse, n_angles=180, radial_offset=4.0):
    xc = float(ellipse["xc"])
    yc = float(ellipse["yc"])
    a = float(ellipse["a"])
    b = float(ellipse["b"])
    th = float(ellipse["theta"])
    c, s = np.cos(th), np.sin(th)
    h, w = img_log.shape

    on_vals = []
    in_vals = []
    out_vals = []

    for t in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
        ct, st = np.cos(t), np.sin(t)
        x_on = xc + a * ct * c - b * st * s
        y_on = yc + a * ct * s + b * st * c

        rx = x_on - xc
        ry = y_on - yc
        rr = np.hypot(rx, ry)
        if rr < 1e-6:
            continue
        ux, uy = rx / rr, ry / rr

        x_in = x_on - radial_offset * ux
        y_in = y_on - radial_offset * uy
        x_out = x_on + radial_offset * ux
        y_out = y_on + radial_offset * uy

        if (
            x_on < 1
            or x_on > w - 2
            or y_on < 1
            or y_on > h - 2
            or x_in < 1
            or x_in > w - 2
            or y_in < 1
            or y_in > h - 2
            or x_out < 1
            or x_out > w - 2
            or y_out < 1
            or y_out > h - 2
        ):
            continue

        on_vals.append(bilinear_sample(img_log, x_on, y_on))
        in_vals.append(bilinear_sample(img_log, x_in, y_in))
        out_vals.append(bilinear_sample(img_log, x_out, y_out))

    if len(on_vals) < 20:
        return np.nan

    on_vals = np.asarray(on_vals, dtype=float)
    in_vals = np.asarray(in_vals, dtype=float)
    out_vals = np.asarray(out_vals, dtype=float)

    background = 0.5 * (in_vals + out_vals)
    signal = on_vals - background
    med_signal = np.median(signal)

    bg_med = np.median(background)
    bg_mad = np.median(np.abs(background - bg_med))
    noise = max(1.4826 * bg_mad, 1e-6)
    return float(med_signal / noise)


def click_angular_coverage(points, ellipse):
    arr = np.asarray(points, dtype=float).reshape(-1, 2)
    if arr.shape[0] < 2:
        return 0.0

    xc = float(ellipse["xc"])
    yc = float(ellipse["yc"])
    a = max(abs(float(ellipse["a"])), 1e-6)
    b = max(abs(float(ellipse["b"])), 1e-6)
    th = float(ellipse["theta"])
    c, s = np.cos(th), np.sin(th)

    dx = arr[:, 0] - xc
    dy = arr[:, 1] - yc
    u = dx * c + dy * s
    v = -dx * s + dy * c

    ang = np.mod(np.arctan2(v / b, u / a), 2 * np.pi)
    ang.sort()
    diffs = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
    largest_gap = np.max(diffs)
    coverage = 1.0 - largest_gap / (2 * np.pi)
    return float(np.clip(coverage, 0.0, 1.0))


def compute_fit_confidence(img_log, points_ds, ellipses, downsample):
    ds = max(float(downsample), 1.0)
    # Clicks are made on downsampled pixels; mapped full-res uncertainty is ~ds/sqrt(12).
    click_sigma_px = ds / np.sqrt(12.0)
    downsample_score = 1.0 / (1.0 + (click_sigma_px / 1.5) ** 2)

    ring_indices = []
    conf = []
    residual_px = []
    signal_snr = []
    coverage = []
    n_points = []

    by_ring = {}
    for i, e in enumerate(ellipses):
        rid = int(e.get("ring_index", i + 1))
        by_ring[rid] = e

    for ring_i, pts in enumerate(points_ds, start=1):
        arr_ds = np.asarray(pts, dtype=float)
        if arr_ds.shape[0] < 5:
            continue
        if ring_i not in by_ring:
            continue

        e = by_ring[ring_i]

        arr_full = arr_ds * float(downsample)
        rp = ellipse_residual_px(arr_full, e)
        snr = ring_signal_snr(img_log, e, n_angles=180, radial_offset=4.0)
        cov = click_angular_coverage(arr_full, e)
        npt = int(arr_full.shape[0])

        # Residual score includes click quantization floor from downsampling.
        rp_eff = np.sqrt(max(float(rp), 0.0) ** 2 + click_sigma_px ** 2) if np.isfinite(rp) else np.inf
        residual_score = 1.0 / (1.0 + (rp_eff / 2.0) ** 2) if np.isfinite(rp_eff) else 0.0
        snr_pos = max(float(snr), 0.0) if np.isfinite(snr) else 0.0
        snr_score = 1.0 - np.exp(-snr_pos / 4.0)
        coverage_score = float(np.clip(cov, 0.0, 1.0))
        points_score = 1.0 - np.exp(-npt / 6.0)

        score_core = 0.45 * residual_score + 0.35 * snr_score + 0.15 * coverage_score + 0.05 * points_score
        score = score_core * downsample_score
        conf_pct = 100.0 * float(np.clip(score, 0.0, 1.0))

        ring_indices.append(int(ring_i))
        conf.append(conf_pct)
        residual_px.append(float(rp))
        signal_snr.append(float(snr))
        coverage.append(float(cov))
        n_points.append(npt)

    conf = np.asarray(conf, dtype=float)
    n_points_arr = np.asarray(n_points, dtype=float)
    if conf.size > 0:
        if np.any(n_points_arr > 0):
            overall = float(np.average(conf, weights=np.maximum(n_points_arr, 1.0)))
        else:
            overall = float(np.mean(conf))
    else:
        overall = np.nan

    return {
        "ring_indices": np.asarray(ring_indices, dtype=np.int32),
        "per_ring_confidence": conf,
        "overall_confidence": overall,
        "residual_px": np.asarray(residual_px, dtype=float),
        "signal_snr": np.asarray(signal_snr, dtype=float),
        "angular_coverage": np.asarray(coverage, dtype=float),
        "n_points": np.asarray(n_points, dtype=np.int32),
        "downsample_factor": float(ds),
        "click_sigma_px": float(click_sigma_px),
        "downsample_score": float(downsample_score),
    }


def apply_tilt_xy(pts, center, tilt_x_deg, tilt_y_deg):
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.size == 0:
        return pts
    xc, yc = center
    tx = np.deg2rad(np.clip(float(tilt_x_deg), -45.0, 45.0))
    ty = np.deg2rad(np.clip(float(tilt_y_deg), -45.0, 45.0))
    sx = 1.0 / max(abs(float(np.cos(ty))), 1e-3)
    sy = 1.0 / max(abs(float(np.cos(tx))), 1e-3)
    dx = (pts[:, 0] - xc) * sx
    dy = (pts[:, 1] - yc) * sy
    return np.column_stack([xc + dx, yc + dy])


def _rotation_from_tilts_xy(tilt_x_deg, tilt_y_deg):
    tx = np.deg2rad(np.clip(float(tilt_x_deg), -45.0, 45.0))
    ty = np.deg2rad(np.clip(float(tilt_y_deg), -45.0, 45.0))
    cx, sx = np.cos(tx), np.sin(tx)
    cy, sy = np.cos(ty), np.sin(ty)
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx],
        ],
        dtype=float,
    )
    ry = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=float,
    )
    return ry @ rx


def _robust_sigma_1d(values):
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = 1.4826 * mad
    if np.isfinite(sigma) and sigma > 1e-9:
        return float(sigma)
    return float(np.std(arr))


def apply_tilt_projective(pts, center, tilt_x_deg, tilt_y_deg, distance_px):
    arr = np.asarray(pts, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.size == 0:
        return arr

    xc, yc = float(center[0]), float(center[1])
    d = max(float(distance_px), 1e-3)
    rot = _rotation_from_tilts_xy(tilt_x_deg, tilt_y_deg)
    ex = rot[:, 0]
    ey = rot[:, 1]
    center_vec = np.array([0.0, 0.0, d], dtype=float)

    dx = arr[:, 0] - xc
    dy = arr[:, 1] - yc
    p3 = center_vec[None, :] + dx[:, None] * ex[None, :] + dy[:, None] * ey[None, :]

    z = p3[:, 2]
    out = arr.copy()
    good = np.isfinite(z) & (z > 1e-6)
    if np.any(good):
        s = d / z[good]
        out[good, 0] = xc + p3[good, 0] * s
        out[good, 1] = yc + p3[good, 1] * s
    return out


def circularity_cost_projective(params, point_sets, center, weights=None):
    tx, ty, distance_px = [float(v) for v in params]
    if not np.isfinite(distance_px) or distance_px <= 1e-3:
        return np.inf

    xc, yc = float(center[0]), float(center[1])
    w = _prepare_ring_weights(weights, len(point_sets))
    cost = 0.0
    wsum = 0.0
    for i, pts in enumerate(point_sets):
        arr = np.asarray(pts, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.size == 0:
            continue
        corr = apply_tilt_projective(arr, center, tx, ty, distance_px)
        r = np.hypot(corr[:, 0] - xc, corr[:, 1] - yc)
        r = r[np.isfinite(r)]
        if r.size < 3:
            continue
        m = float(np.median(r))
        sig = _robust_sigma_1d(r)
        if m > 1e-9 and np.isfinite(sig):
            c = sig / m
            wi = float(w[i]) if i < w.size else 1.0
            cost += wi * c * c
            wsum += wi
    if wsum <= 1e-12:
        return np.inf
    return float(cost / wsum)


def circularity_metrics_projective(point_sets, center, tx, ty, distance_px):
    xc, yc = float(center[0]), float(center[1])
    vals = []
    for pts in point_sets:
        arr = np.asarray(pts, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.size == 0:
            vals.append(np.nan)
            continue
        corr = apply_tilt_projective(arr, center, tx, ty, distance_px)
        r = np.hypot(corr[:, 0] - xc, corr[:, 1] - yc)
        r = r[np.isfinite(r)]
        if r.size < 3:
            vals.append(np.nan)
            continue
        m = float(np.median(r))
        sig = _robust_sigma_1d(r)
        vals.append(float(sig / m) if m > 1e-9 and np.isfinite(sig) else np.nan)
    return np.asarray(vals, dtype=float)


def optimize_tilts_projective(
    point_sets,
    center,
    weights=None,
    optimize_center=True,
    center_prior=None,
    center_prior_sigma_px=DEFAULT_CENTER_PRIOR_SIGMA_PX,
    center_drift_limit_px=DEFAULT_CENTER_DRIFT_LIMIT_PX,
    initial_tilts=None,
):
    if not point_sets:
        raise ValueError("No point sets available for optimization.")
    weights_arr = _prepare_ring_weights(weights, len(point_sets))

    valid = []
    for p in point_sets:
        arr = np.asarray(p, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.size > 0:
            valid.append(arr)
    if not valid:
        raise ValueError("No valid points available for projective optimization.")

    all_pts = np.vstack(valid)
    xmin, xmax = float(np.min(all_pts[:, 0])), float(np.max(all_pts[:, 0]))
    ymin, ymax = float(np.min(all_pts[:, 1])), float(np.max(all_pts[:, 1]))
    sx = max(xmax - xmin, 1.0)
    sy = max(ymax - ymin, 1.0)
    span_max = max(sx, sy, 1.0)

    cx0, cy0 = float(center[0]), float(center[1])
    ring_r = []
    for arr in valid:
        rr = np.hypot(arr[:, 0] - cx0, arr[:, 1] - cy0)
        rr = rr[np.isfinite(rr)]
        if rr.size > 0:
            ring_r.append(float(np.median(rr)))
    if ring_r:
        r_med = float(np.median(np.asarray(ring_r, dtype=float)))
    else:
        r_med = 0.25 * span_max
    if not np.isfinite(r_med) or r_med <= 0.0:
        r_med = 0.25 * span_max

    dist0 = float(np.clip(4.0 * r_med, PROJECTIVE_DIST_MIN_PX, PROJECTIVE_DIST_MAX_MULT * span_max))
    dist_lo = float(max(PROJECTIVE_DIST_MIN_PX, 0.5 * r_med, 0.15 * span_max))
    dist_hi = float(max(dist_lo + 1.0, PROJECTIVE_DIST_MAX_MULT * span_max))
    logd_lo = float(np.log(max(dist_lo, 1e-3)))
    logd_hi = float(np.log(max(dist_hi, dist_lo + 1e-3)))
    dist_prior_sigma = float(max(PROJECTIVE_DIST_PRIOR_SIGMA_LOG, 0.1))

    center_opt = (cx0, cy0)
    prior_center = None
    if center_prior is not None:
        try:
            prior_center = (float(center_prior[0]), float(center_prior[1]))
            if not (np.isfinite(prior_center[0]) and np.isfinite(prior_center[1])):
                prior_center = None
        except Exception:
            prior_center = None

    prior_sigma = float(center_prior_sigma_px) if center_prior_sigma_px is not None else np.nan
    if not np.isfinite(prior_sigma) or prior_sigma <= 0:
        prior_sigma = DEFAULT_CENTER_PRIOR_SIGMA_PX
    drift_limit = float(center_drift_limit_px) if center_drift_limit_px is not None else np.nan
    if not np.isfinite(drift_limit) or drift_limit <= 0:
        drift_limit = DEFAULT_CENTER_DRIFT_LIMIT_PX

    def _cost_with_distance_prior(cx, cy, tx, ty, dist):
        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(tx) and np.isfinite(ty) and np.isfinite(dist)):
            return np.inf
        if dist <= 1e-3:
            return np.inf
        base = circularity_cost_projective((tx, ty, dist), point_sets, (cx, cy), weights=weights_arr)
        if not np.isfinite(base):
            return np.inf
        lnd = np.log(dist / dist0)
        return float(base + 1e-3 * (lnd / dist_prior_sigma) ** 2)

    # Stage 1: coarse tilt grid with fixed center and nominal distance.
    grid_n = int(OPT_GRID_SIZE_COLD)
    grid_span = float(OPT_GRID_SPAN_COLD)
    best = (0.0, 0.0, dist0, _cost_with_distance_prior(center_opt[0], center_opt[1], 0.0, 0.0, dist0))
    if initial_tilts is not None:
        try:
            tx0 = float(initial_tilts[0])
            ty0 = float(initial_tilts[1])
            if np.isfinite(tx0) and np.isfinite(ty0):
                tx0 = float(np.clip(tx0, -PROJECTIVE_TILT_MAX_DEG, PROJECTIVE_TILT_MAX_DEG))
                ty0 = float(np.clip(ty0, -PROJECTIVE_TILT_MAX_DEG, PROJECTIVE_TILT_MAX_DEG))
                c0 = _cost_with_distance_prior(center_opt[0], center_opt[1], tx0, ty0, dist0)
                if np.isfinite(c0):
                    best = (tx0, ty0, dist0, float(c0))
                    grid_n = int(OPT_GRID_SIZE_WARM)
                    grid_span = float(OPT_GRID_SPAN_WARM)
        except Exception:
            pass
    tx_center = best[0] if np.isfinite(best[0]) else 0.0
    ty_center = best[1] if np.isfinite(best[1]) else 0.0
    tx_grid = np.linspace(
        max(-PROJECTIVE_TILT_MAX_DEG, tx_center - grid_span),
        min(PROJECTIVE_TILT_MAX_DEG, tx_center + grid_span),
        max(grid_n, 7),
    )
    ty_grid = np.linspace(
        max(-PROJECTIVE_TILT_MAX_DEG, ty_center - grid_span),
        min(PROJECTIVE_TILT_MAX_DEG, ty_center + grid_span),
        max(grid_n, 7),
    )
    for tx in tx_grid:
        for ty in ty_grid:
            c = _cost_with_distance_prior(center_opt[0], center_opt[1], tx, ty, dist0)
            if c < best[3]:
                best = (float(tx), float(ty), float(dist0), float(c))

    # Stage 2: local tilt+distance refinement with fixed center.
    def cost3(p3):
        tx, ty, logd = [float(v) for v in p3]
        tx = float(np.clip(tx, -PROJECTIVE_TILT_MAX_DEG, PROJECTIVE_TILT_MAX_DEG))
        ty = float(np.clip(ty, -PROJECTIVE_TILT_MAX_DEG, PROJECTIVE_TILT_MAX_DEG))
        logd = float(np.clip(logd, logd_lo, logd_hi))
        dist = float(np.exp(logd))
        return _cost_with_distance_prior(center_opt[0], center_opt[1], tx, ty, dist)

    res_tilt = minimize(
        cost3,
        np.array([best[0], best[1], np.log(best[2])], dtype=float),
        method="Powell",
        bounds=[
            (-PROJECTIVE_TILT_MAX_DEG, PROJECTIVE_TILT_MAX_DEG),
            (-PROJECTIVE_TILT_MAX_DEG, PROJECTIVE_TILT_MAX_DEG),
            (logd_lo, logd_hi),
        ],
        options={"maxiter": int(OPT_POWELL_MAXITER), "xtol": 1e-3, "ftol": 1e-8, "disp": False},
    )

    tx = float(best[0])
    ty = float(best[1])
    dist_opt = float(best[2])
    if res_tilt.success and np.all(np.isfinite(res_tilt.x)):
        tx = float(np.clip(res_tilt.x[0], -PROJECTIVE_TILT_MAX_DEG, PROJECTIVE_TILT_MAX_DEG))
        ty = float(np.clip(res_tilt.x[1], -PROJECTIVE_TILT_MAX_DEG, PROJECTIVE_TILT_MAX_DEG))
        dist_opt = float(np.exp(np.clip(res_tilt.x[2], logd_lo, logd_hi)))

    # Stage 3: jointly refine center + tilts + distance.
    res_center = None
    if optimize_center:
        mx = max(20.0, 0.2 * sx)
        my = max(20.0, 0.2 * sy)
        bounds_base = [
            (xmin - mx, xmax + mx),
            (ymin - my, ymax + my),
            (-PROJECTIVE_TILT_MAX_DEG, PROJECTIVE_TILT_MAX_DEG),
            (-PROJECTIVE_TILT_MAX_DEG, PROJECTIVE_TILT_MAX_DEG),
            (logd_lo, logd_hi),
        ]

        if prior_center is not None:
            cxp, cyp = prior_center
            cx_lo = max(bounds_base[0][0], cxp - drift_limit)
            cx_hi = min(bounds_base[0][1], cxp + drift_limit)
            cy_lo = max(bounds_base[1][0], cyp - drift_limit)
            cy_hi = min(bounds_base[1][1], cyp + drift_limit)
            if cx_hi - cx_lo > 1e-6 and cy_hi - cy_lo > 1e-6:
                bounds = [
                    (cx_lo, cx_hi),
                    (cy_lo, cy_hi),
                    bounds_base[2],
                    bounds_base[3],
                    bounds_base[4],
                ]
            else:
                bounds = bounds_base
        else:
            bounds = bounds_base

        prior_scale = max(_cost_with_distance_prior(center_opt[0], center_opt[1], tx, ty, dist_opt), 1e-6)

        def cost5(p5):
            cx, cy, txx, tyy, logd = [float(v) for v in p5]
            txx = float(np.clip(txx, -PROJECTIVE_TILT_MAX_DEG, PROJECTIVE_TILT_MAX_DEG))
            tyy = float(np.clip(tyy, -PROJECTIVE_TILT_MAX_DEG, PROJECTIVE_TILT_MAX_DEG))
            logd = float(np.clip(logd, logd_lo, logd_hi))
            dist = float(np.exp(logd))
            base = _cost_with_distance_prior(cx, cy, txx, tyy, dist)
            if prior_center is not None:
                dxn = (cx - prior_center[0]) / prior_sigma
                dyn = (cy - prior_center[1]) / prior_sigma
                base += prior_scale * (dxn * dxn + dyn * dyn)
            return base

        x0_center = prior_center if prior_center is not None else center_opt
        x0 = np.array([x0_center[0], x0_center[1], tx, ty, np.log(dist_opt)], dtype=float)
        res_center = minimize(
            cost5,
            x0,
            method="Powell",
            bounds=bounds,
            options={"maxiter": int(OPT_POWELL_MAXITER), "xtol": 1e-3, "ftol": 1e-8, "disp": False},
        )
        if res_center.success and np.all(np.isfinite(res_center.x)):
            center_opt = (float(res_center.x[0]), float(res_center.x[1]))
            tx = float(np.clip(res_center.x[2], -PROJECTIVE_TILT_MAX_DEG, PROJECTIVE_TILT_MAX_DEG))
            ty = float(np.clip(res_center.x[3], -PROJECTIVE_TILT_MAX_DEG, PROJECTIVE_TILT_MAX_DEG))
            dist_opt = float(np.exp(np.clip(res_center.x[4], logd_lo, logd_hi)))

    corrected = [apply_tilt_projective(p, center_opt, tx, ty, dist_opt) for p in point_sets]
    return {
        "center_initial": (float(center[0]), float(center[1])),
        "center": center_opt,
        "tilt_x_deg": tx,
        "tilt_y_deg": ty,
        "distance_px": float(dist_opt),
        "cost_zero": circularity_cost_projective((0.0, 0.0, dist_opt), point_sets, center_opt, weights=weights_arr),
        "cost_final": circularity_cost_projective((tx, ty, dist_opt), point_sets, center_opt, weights=weights_arr),
        "circ_before": circularity_metrics_projective(point_sets, center_opt, 0.0, 0.0, dist_opt),
        "circ_after": circularity_metrics_projective(point_sets, center_opt, tx, ty, dist_opt),
        "radii_before": ring_radii(point_sets, center_opt),
        "radii_after": ring_radii(corrected, center_opt),
        "corrected_points": corrected,
        "ring_weights": weights_arr.astype(float),
        "center_prior": (np.nan, np.nan) if prior_center is None else prior_center,
        "center_prior_sigma_px": float(prior_sigma),
        "center_drift_limit_px": float(drift_limit),
        "optimizer_success": bool(res_tilt.success if res_center is None else res_center.success and res_tilt.success),
        "optimizer_message": str(
            res_tilt.message if res_center is None else f"{res_tilt.message} | {res_center.message}"
        ),
        "optimizer_kind": "projective_v1",
    }


def _prepare_ring_weights(weights, n):
    if n <= 0:
        return np.array([], dtype=float)
    if weights is None:
        return np.ones(n, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size < n:
        pad = np.ones(n - w.size, dtype=float)
        w = np.r_[w, pad]
    w = w[:n]
    w = np.where(np.isfinite(w), w, 1.0)
    w = np.clip(w, 0.05, 5.0)
    return w


def circularity_cost(params, point_sets, center, weights=None):
    tx, ty = params
    xc, yc = center
    w = _prepare_ring_weights(weights, len(point_sets))
    cost = 0.0
    wsum = 0.0
    for i, pts in enumerate(point_sets):
        arr = np.asarray(pts, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.size == 0:
            continue
        corr = apply_tilt_xy(arr, center, tx, ty)
        r = np.hypot(corr[:, 0] - xc, corr[:, 1] - yc)
        m = r.mean()
        if m > 1e-9:
            c = np.std(r) / m
            wi = float(w[i]) if i < w.size else 1.0
            cost += wi * c * c
            wsum += wi
    if wsum <= 1e-12:
        return np.inf
    return float(cost / wsum)


def circularity_metrics(point_sets, center, tx, ty):
    xc, yc = center
    vals = []
    for pts in point_sets:
        arr = np.asarray(pts, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.size == 0:
            vals.append(np.nan)
            continue
        corr = apply_tilt_xy(arr, center, tx, ty)
        r = np.hypot(corr[:, 0] - xc, corr[:, 1] - yc)
        m = r.mean()
        vals.append(float(np.std(r) / m) if m > 1e-9 else np.nan)
    return np.asarray(vals, dtype=float)


def ring_radii(point_sets, center):
    xc, yc = center
    vals = []
    for pts in point_sets:
        arr = np.asarray(pts, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.size == 0:
            vals.append(np.nan)
            continue
        r = np.hypot(arr[:, 0] - xc, arr[:, 1] - yc)
        vals.append(float(r.mean()))
    return np.asarray(vals, dtype=float)


def optimize_tilts(
    point_sets,
    center,
    weights=None,
    optimize_center=True,
    center_prior=None,
    center_prior_sigma_px=DEFAULT_CENTER_PRIOR_SIGMA_PX,
    center_drift_limit_px=DEFAULT_CENTER_DRIFT_LIMIT_PX,
    initial_tilts=None,
):
    if not point_sets:
        raise ValueError("No point sets available for optimization.")
    weights_arr = _prepare_ring_weights(weights, len(point_sets))

    # Stage 1: coarse tilt grid with fixed center.
    grid_n = int(OPT_GRID_SIZE_COLD)
    grid_span = float(OPT_GRID_SPAN_COLD)
    best = (0.0, 0.0, np.inf)
    if initial_tilts is not None:
        try:
            tx0 = float(initial_tilts[0])
            ty0 = float(initial_tilts[1])
            if np.isfinite(tx0) and np.isfinite(ty0):
                tx0 = float(np.clip(tx0, -20.0, 20.0))
                ty0 = float(np.clip(ty0, -20.0, 20.0))
                c0 = circularity_cost((tx0, ty0), point_sets, center, weights=weights_arr)
                if np.isfinite(c0):
                    best = (tx0, ty0, float(c0))
                    grid_n = int(OPT_GRID_SIZE_WARM)
                    grid_span = float(OPT_GRID_SPAN_WARM)
        except Exception:
            pass
    tx_center = best[0] if np.isfinite(best[0]) else 0.0
    ty_center = best[1] if np.isfinite(best[1]) else 0.0
    tx_grid = np.linspace(
        max(-20.0, tx_center - grid_span),
        min(20.0, tx_center + grid_span),
        max(grid_n, 7),
    )
    ty_grid = np.linspace(
        max(-20.0, ty_center - grid_span),
        min(20.0, ty_center + grid_span),
        max(grid_n, 7),
    )
    for tx in tx_grid:
        for ty in ty_grid:
            c = circularity_cost((tx, ty), point_sets, center, weights=weights_arr)
            if c < best[2]:
                best = (float(tx), float(ty), float(c))

    # Stage 2: local tilt refinement with fixed center.
    res_tilt = minimize(
        circularity_cost,
        np.array([best[0], best[1]], dtype=float),
        args=(point_sets, center, weights_arr),
        method="Nelder-Mead",
        options={"maxiter": int(OPT_NELDER_MAXITER), "xatol": 1e-4, "fatol": 1e-8, "disp": False},
    )
    tx, ty = float(res_tilt.x[0]), float(res_tilt.x[1])
    center_opt = (float(center[0]), float(center[1]))
    res_center = None
    prior_center = None
    if center_prior is not None:
        try:
            prior_center = (float(center_prior[0]), float(center_prior[1]))
            if not (np.isfinite(prior_center[0]) and np.isfinite(prior_center[1])):
                prior_center = None
        except Exception:
            prior_center = None
    prior_sigma = float(center_prior_sigma_px) if center_prior_sigma_px is not None else np.nan
    if not np.isfinite(prior_sigma) or prior_sigma <= 0:
        prior_sigma = DEFAULT_CENTER_PRIOR_SIGMA_PX
    drift_limit = float(center_drift_limit_px) if center_drift_limit_px is not None else np.nan
    if not np.isfinite(drift_limit) or drift_limit <= 0:
        drift_limit = DEFAULT_CENTER_DRIFT_LIMIT_PX

    # Stage 3: jointly refine center + tilts.
    if optimize_center:
        valid = []
        for p in point_sets:
            arr = np.asarray(p, dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 2 and arr.size > 0:
                valid.append(arr)

        if valid:
            all_pts = np.vstack(valid)
            xmin, xmax = float(np.min(all_pts[:, 0])), float(np.max(all_pts[:, 0]))
            ymin, ymax = float(np.min(all_pts[:, 1])), float(np.max(all_pts[:, 1]))
            sx = max(xmax - xmin, 1.0)
            sy = max(ymax - ymin, 1.0)
            mx = max(20.0, 0.2 * sx)
            my = max(20.0, 0.2 * sy)
            bounds_base = [
                (xmin - mx, xmax + mx),
                (ymin - my, ymax + my),
                (-20.0, 20.0),
                (-20.0, 20.0),
            ]

            if prior_center is not None:
                cx0, cy0 = prior_center
                cx_lo = max(bounds_base[0][0], cx0 - drift_limit)
                cx_hi = min(bounds_base[0][1], cx0 + drift_limit)
                cy_lo = max(bounds_base[1][0], cy0 - drift_limit)
                cy_hi = min(bounds_base[1][1], cy0 + drift_limit)
                if cx_hi - cx_lo > 1e-6 and cy_hi - cy_lo > 1e-6:
                    bounds = [
                        (cx_lo, cx_hi),
                        (cy_lo, cy_hi),
                        bounds_base[2],
                        bounds_base[3],
                    ]
                else:
                    bounds = bounds_base
            else:
                bounds = bounds_base

            prior_scale = max(float(best[2]), 1e-6)

            def cost4(p4):
                cx, cy, txx, tyy = [float(v) for v in p4]
                base = circularity_cost((txx, tyy), point_sets, (cx, cy), weights=weights_arr)
                if prior_center is not None:
                    dxn = (cx - prior_center[0]) / prior_sigma
                    dyn = (cy - prior_center[1]) / prior_sigma
                    base += prior_scale * (dxn * dxn + dyn * dyn)
                return base

            x0_center = prior_center if prior_center is not None else center_opt
            x0 = np.array([x0_center[0], x0_center[1], tx, ty], dtype=float)
            res_center = minimize(
                cost4,
                x0,
                method="Powell",
                bounds=bounds,
                options={"maxiter": int(OPT_POWELL_MAXITER), "xtol": 1e-3, "ftol": 1e-8, "disp": False},
            )
            if res_center.success and np.all(np.isfinite(res_center.x)):
                center_opt = (float(res_center.x[0]), float(res_center.x[1]))
                tx = float(res_center.x[2])
                ty = float(res_center.x[3])

    corrected = [apply_tilt_xy(p, center_opt, tx, ty) for p in point_sets]
    return {
        "center_initial": (float(center[0]), float(center[1])),
        "center": center_opt,
        "tilt_x_deg": tx,
        "tilt_y_deg": ty,
        "cost_zero": circularity_cost((0.0, 0.0), point_sets, center_opt, weights=weights_arr),
        "cost_final": circularity_cost((tx, ty), point_sets, center_opt, weights=weights_arr),
        "circ_before": circularity_metrics(point_sets, center_opt, 0.0, 0.0),
        "circ_after": circularity_metrics(point_sets, center_opt, tx, ty),
        "radii_before": ring_radii(point_sets, center_opt),
        "radii_after": ring_radii(corrected, center_opt),
        "corrected_points": corrected,
        "ring_weights": weights_arr.astype(float),
        "center_prior": (np.nan, np.nan) if prior_center is None else prior_center,
        "center_prior_sigma_px": float(prior_sigma),
        "center_drift_limit_px": float(drift_limit),
        "optimizer_success": bool(res_tilt.success if res_center is None else res_center.success and res_tilt.success),
        "optimizer_message": str(res_tilt.message if res_center is None else f"{res_tilt.message} | {res_center.message}"),
        "optimizer_kind": "legacy_v1",
        "distance_px": np.nan,
    }


class HBNFitterGUI:
    def __init__(self, root, startup_bundle=None):
        self.root = root
        self.root.title("hBN Ring Fitter + Circularization")

        self.hbn_path = tk.StringVar()
        self.dark_path = tk.StringVar()
        self.num_rings = tk.StringVar(value=str(DEFAULT_NUM_RINGS))
        self.points_per_ring = tk.StringVar(value=str(DEFAULT_POINTS_PER_RING))
        self.downsample = tk.StringVar(value=str(DEFAULT_DOWNSAMPLE))
        self.refine_angles = tk.StringVar(value=str(DEFAULT_REFINE_N_ANGLES))
        self.refine_dr = tk.StringVar(value=str(DEFAULT_REFINE_DR))
        self.refine_step = tk.StringVar(value=str(DEFAULT_REFINE_STEP))
        self.dense_points = tk.StringVar(value=str(DEFAULT_DENSE_POINTS))
        self.center_x = tk.StringVar(value="")
        self.center_y = tk.StringVar(value="")
        self.status = tk.StringVar(value="Set hBN and dark OSC files; image loads automatically.")
        self.progress = tk.StringVar(value="")
        self.fit_text = tk.StringVar(value="fit_conf=--")
        self.opt_text = tk.StringVar(value="tilt_x=--, tilt_y=--, cost=--")

        self.img_bgsub = None
        self.img_log = None
        self.img_disp = None
        self.down = DEFAULT_DOWNSAMPLE
        self.points_ds = []
        self.points_raw_ds = []
        self.points_sigma_ds = []
        self.ellipses = []
        self.fit_quality = None
        self.optim = None
        self.corrected = []
        self.pick_mode = False
        self.edit_mode = False
        self.center_pick_mode = False
        self.edit_btn = None
        self.active_ring = 0
        self.center_pick_btn = None
        self.refine_more_btn = None
        self._center_user_defined = False
        self._auto_load_job = None
        self._auto_load_suspended = False
        self._last_loaded_pair = None
        self._pan_active = False
        self._pan_anchor = None
        self._precision_active = False
        self._precision_ring_idx = None
        self._precision_restore_limits = None
        self._precision_preview_xy = None
        self._precision_preview_snap_xy = None
        self._precision_preview_artist = None
        self._precision_preview_snap_artist = None
        self._precision_preview_link_artist = None
        self._precision_preview_last_t = 0.0
        self._precision_preview_last_xy = None
        self._drag_edit_active = False
        self._drag_edit_ring_idx = None
        self._drag_edit_point_idx = None
        self._drag_edit_restore_limits = None
        self._drag_edit_preview_xy = None
        self._drag_edit_preview_snap_xy = None
        self._drag_edit_preview_artist = None
        self._drag_edit_preview_snap_artist = None
        self._drag_edit_preview_link_artist = None
        self._drag_edit_preview_last_t = 0.0
        self._drag_edit_preview_last_xy = None
        self._show_suggested_regions = False
        self.suggested_regions = []

        self._build_ui()
        self.hbn_path.trace_add("write", self._on_file_path_changed)
        self.dark_path.trace_add("write", self._on_file_path_changed)
        self.points_per_ring.trace_add("write", self._on_points_per_ring_changed)
        self._init_points(clear=True)
        self._update_progress()
        self._update_fit_text()
        self._update_refine_more_btn_state()
        self.refresh_plot()

        if startup_bundle:
            self.load_bundle(Path(startup_bundle))

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)
        left = ttk.Frame(main)
        left.pack(side="left", fill="y")
        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True)

        f_files = ttk.LabelFrame(left, text="Files", padding=6)
        f_files.pack(fill="x", pady=(0, 6))
        ttk.Label(f_files, text="hBN OSC").grid(row=0, column=0, sticky="w")
        ttk.Entry(f_files, textvariable=self.hbn_path, width=34).grid(row=0, column=1, sticky="ew")
        ttk.Button(f_files, text="Browse", command=self.browse_hbn).grid(row=0, column=2)
        ttk.Label(f_files, text="Dark OSC").grid(row=1, column=0, sticky="w")
        ttk.Entry(f_files, textvariable=self.dark_path, width=34).grid(row=1, column=1, sticky="ew")
        ttk.Button(f_files, text="Browse", command=self.browse_dark).grid(row=1, column=2)
        f_files.columnconfigure(1, weight=1)

        f_set = ttk.LabelFrame(left, text="Settings", padding=6)
        f_set.pack(fill="x", pady=(0, 6))
        labels = [
            ("Rings (↑prec)", self.num_rings),
            ("Points/Ring (↑prec)", self.points_per_ring),
            ("Downsample (↓prec)", self.downsample),
            ("Refine Angles (↑prec)", self.refine_angles),
            ("Refine dR (↓prec)", self.refine_dr),
            ("Refine Step (↓prec)", self.refine_step),
            ("Dense Points (↑prec)", self.dense_points),
            ("Center X", self.center_x),
            ("Center Y", self.center_y),
        ]
        for i, (name, var) in enumerate(labels):
            ttk.Label(f_set, text=name).grid(row=i, column=0, sticky="w")
            ttk.Entry(f_set, textvariable=var, width=12).grid(row=i, column=1, sticky="ew")
        f_set.columnconfigure(1, weight=1)

        f_btn = ttk.LabelFrame(left, text="Actions", padding=6)
        f_btn.pack(fill="x", pady=(0, 6))
        ttk.Button(
            f_btn,
            text="Run Full Calibration",
            command=self.solve_all_action,
        ).pack(fill="x", pady=2)
        self.center_pick_btn = ttk.Button(f_btn, text="Pick Beam Center", command=self.toggle_center_pick)
        self.center_pick_btn.pack(fill="x", pady=2)
        self.pick_btn = ttk.Button(f_btn, text="Enable Pick Mode", command=self.toggle_pick)
        self.pick_btn.pack(fill="x", pady=2)
        self.edit_btn = ttk.Button(f_btn, text="Enable Edit Mode", command=self.toggle_edit_mode)
        self.edit_btn.pack(fill="x", pady=2)
        ttk.Button(f_btn, text="Add +1 Point/Ring", command=self.add_more_points_action).pack(fill="x", pady=2)
        ttk.Button(f_btn, text="Undo Last Point", command=self.undo_last).pack(fill="x", pady=2)
        ttk.Button(f_btn, text="Reset Points", command=self.reset_points).pack(fill="x", pady=2)
        ttk.Button(f_btn, text="Clear Fits (Keep Points)", command=self.clear_fits_keep_points).pack(fill="x", pady=2)
        self.refine_more_btn = ttk.Button(
            f_btn,
            text="Further Refine (High-Res)",
            command=self.further_refine_action,
            state="disabled",
        )
        self.refine_more_btn.pack(fill="x", pady=2)
        ttk.Button(f_btn, text="Load Bundle NPZ", command=self.load_bundle_dialog).pack(fill="x", pady=2)
        ttk.Button(f_btn, text="Save Bundle NPZ", command=self.save_bundle_dialog).pack(fill="x", pady=2)
        ttk.Button(f_btn, text="Save Overlay PNG", command=self.save_overlay_dialog).pack(fill="x", pady=2)

        f_status = ttk.LabelFrame(left, text="Status", padding=6)
        f_status.pack(fill="x")
        ttk.Label(f_status, textvariable=self.progress, wraplength=300, justify="left").pack(fill="x")
        ttk.Label(f_status, textvariable=self.fit_text, wraplength=300, justify="left").pack(fill="x")
        ttk.Label(f_status, textvariable=self.opt_text, wraplength=300, justify="left").pack(fill="x")
        ttk.Label(f_status, textvariable=self.status, wraplength=300, justify="left").pack(fill="x")

        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(fill="x")
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

    def _init_points(self, clear=False):
        n = safe_int(self.num_rings.get(), DEFAULT_NUM_RINGS, minimum=1)
        if clear or not self.points_ds:
            self.points_ds = [[] for _ in range(n)]
            self.points_raw_ds = [[] for _ in range(n)]
            self.points_sigma_ds = [[] for _ in range(n)]
        else:
            if len(self.points_ds) < n:
                self.points_ds.extend([[] for _ in range(n - len(self.points_ds))])
            elif len(self.points_ds) > n:
                self.points_ds = self.points_ds[:n]
            if not self.points_raw_ds:
                self.points_raw_ds = [list(r) for r in self.points_ds]
            elif len(self.points_raw_ds) < n:
                self.points_raw_ds.extend([[] for _ in range(n - len(self.points_raw_ds))])
            elif len(self.points_raw_ds) > n:
                self.points_raw_ds = self.points_raw_ds[:n]
            if not self.points_sigma_ds:
                self.points_sigma_ds = [[np.nan] * len(r) for r in self.points_ds]
            elif len(self.points_sigma_ds) < n:
                self.points_sigma_ds.extend([[] for _ in range(n - len(self.points_sigma_ds))])
            elif len(self.points_sigma_ds) > n:
                self.points_sigma_ds = self.points_sigma_ds[:n]
        self._sync_point_sigma_shape()
        nxt = self._next_ring(0)
        self.active_ring = 0 if nxt is None else nxt

    def _sync_point_sigma_shape(self):
        n = len(self.points_ds)
        if not self.points_sigma_ds:
            self.points_sigma_ds = [[] for _ in range(n)]
        if len(self.points_sigma_ds) < n:
            self.points_sigma_ds.extend([[] for _ in range(n - len(self.points_sigma_ds))])
        elif len(self.points_sigma_ds) > n:
            self.points_sigma_ds = self.points_sigma_ds[:n]

        for i in range(n):
            target = len(self.points_ds[i])
            ring = list(self.points_sigma_ds[i]) if self.points_sigma_ds[i] is not None else []
            if len(ring) < target:
                ring.extend([np.nan] * (target - len(ring)))
            elif len(ring) > target:
                ring = ring[:target]
            clean = []
            for v in ring:
                try:
                    fv = float(v)
                except Exception:
                    fv = np.nan
                clean.append(fv if np.isfinite(fv) and fv > 0.0 else np.nan)
            self.points_sigma_ds[i] = clean

    def _next_ring(self, start):
        if not self.points_ds:
            return None
        cap = safe_int(self.points_per_ring.get(), DEFAULT_POINTS_PER_RING, minimum=1)
        for off in range(len(self.points_ds)):
            i = (start + off) % len(self.points_ds)
            if len(self.points_ds[i]) < cap:
                return i
        return None

    def _update_progress(self):
        cap = safe_int(self.points_per_ring.get(), DEFAULT_POINTS_PER_RING, minimum=1)
        self.progress.set(" | ".join([f"R{i+1}:{len(p)}/{cap}" for i, p in enumerate(self.points_ds)]))

    def _update_fit_text(self):
        if not self.fit_quality:
            self.fit_text.set("fit_conf=--")
            return

        per = np.asarray(self.fit_quality.get("per_ring_confidence", np.array([], dtype=float)), dtype=float).reshape(-1)
        rings = np.asarray(self.fit_quality.get("ring_indices", np.array([], dtype=np.int32)), dtype=int).reshape(-1)
        overall = float(self.fit_quality.get("overall_confidence", np.nan))
        ds = float(self.fit_quality.get("downsample_factor", np.nan))
        ds_score = float(self.fit_quality.get("downsample_score", np.nan))

        if per.size == 0:
            if np.isfinite(overall):
                if np.isfinite(ds) and np.isfinite(ds_score):
                    self.fit_text.set(f"fit_conf overall={overall:.1f}% (ds={ds:.0f}, ds_score={100.0*ds_score:.1f}%)")
                else:
                    self.fit_text.set(f"fit_conf overall={overall:.1f}%")
            else:
                self.fit_text.set("fit_conf=--")
            return

        parts = []
        n_show = min(4, per.size)
        for i in range(n_show):
            rid = int(rings[i]) if i < rings.size else int(i + 1)
            if np.isfinite(per[i]):
                parts.append(f"R{rid}:{per[i]:.1f}%")
        suffix = " ..." if per.size > n_show else ""
        tail = ", ".join(parts) + suffix
        if np.isfinite(overall):
            if tail:
                if np.isfinite(ds) and np.isfinite(ds_score):
                    self.fit_text.set(
                        f"fit_conf overall={overall:.1f}% (ds={ds:.0f}, ds_score={100.0*ds_score:.1f}%) | {tail}"
                    )
                else:
                    self.fit_text.set(f"fit_conf overall={overall:.1f}% | {tail}")
            else:
                if np.isfinite(ds) and np.isfinite(ds_score):
                    self.fit_text.set(f"fit_conf overall={overall:.1f}% (ds={ds:.0f}, ds_score={100.0*ds_score:.1f}%)")
                else:
                    self.fit_text.set(f"fit_conf overall={overall:.1f}%")
        else:
            self.fit_text.set(f"fit_conf | {tail}" if tail else "fit_conf=--")

    def _update_refine_more_btn_state(self):
        if self.refine_more_btn is None:
            return
        enabled = bool(self.ellipses)
        self.refine_more_btn.configure(state="normal" if enabled else "disabled")

    def _clear_suggested_regions(self):
        self._show_suggested_regions = False
        self.suggested_regions = []

    def _largest_angle_gap_mid(self, ang):
        a = np.asarray(ang, dtype=float).reshape(-1)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return 0.0
        if a.size == 1:
            return float(np.mod(a[0] + np.pi, 2.0 * np.pi))

        a = np.mod(a, 2.0 * np.pi)
        a.sort()
        d = np.diff(np.r_[a, a[0] + 2.0 * np.pi])
        k = int(np.argmax(d))
        return float(np.mod(a[k] + 0.5 * d[k], 2.0 * np.pi))

    def _best_snap_from_seeds(
        self,
        img_log_full,
        params,
        seeds_xy,
        search_along,
        search_across,
        search_step,
    ):
        if img_log_full is None or params is None:
            return None

        try:
            scale = max(0.5 * (abs(float(params[2])) + abs(float(params[3]))), 1.0)
        except Exception:
            scale = 1.0

        def _score_from_meta(meta):
            m = meta or {}
            post = float(m.get("best_posterior", np.nan))
            margin = float(m.get("posterior_margin", np.nan))
            snr = float(m.get("best_peak_snr", np.nan))
            resid = float(m.get("best_resid_px", np.nan))
            cdist = float(m.get("best_click_distance_px", np.nan))
            if not np.isfinite(post):
                post = -1e6
            if not np.isfinite(margin):
                margin = 0.0
            if not np.isfinite(snr):
                snr = 0.0
            if not np.isfinite(resid):
                resid = 1e3
            if not np.isfinite(cdist):
                cdist = 1e3
            return (
                post
                + 0.22 * np.clip(snr, -2.0, 14.0)
                + 0.10 * np.clip(margin, 0.0, 10.0)
                - 0.07 * (resid / max(scale, 1.0))
                - 0.025 * cdist
            )

        def _snap_once(x_seed, y_seed, along, across, step):
            snapped, meta = snap_points_to_ring(
                img_log_full,
                np.array([[float(x_seed), float(y_seed)]], dtype=float),
                params,
                search_along=float(along),
                search_across=float(across),
                search_step=float(step),
                use_pseudovoigt=True,
                enforce_confidence=False,
                return_meta=True,
            )
            if snapped.shape[0] < 1 or not np.all(np.isfinite(snapped[0])):
                return None
            m = meta[0] if meta else {}
            score = float(_score_from_meta(m))
            return (float(snapped[0, 0]), float(snapped[0, 1]), score, m)

        best = None
        for x0, y0 in seeds_xy:
            primary = _snap_once(x0, y0, search_along, search_across, search_step)
            if primary is None:
                continue

            fine_step = float(np.clip(0.55 * float(search_step), 0.20, 0.35))
            fine = _snap_once(
                primary[0],
                primary[1],
                max(0.60 * float(search_along), 8.0),
                max(0.60 * float(search_across), 2.5),
                fine_step,
            )
            cand = primary if fine is None or primary[2] >= fine[2] else fine
            if best is None or cand[2] > best[2]:
                best = cand
        return best

    def _suggest_region_for_ring(self, ring_idx0):
        img_log_full = self._get_fullres_log_image()
        if img_log_full is None:
            return None

        pts_full = np.asarray(self.points_ds[ring_idx0], dtype=float).reshape(-1, 2)
        if pts_full.size == 0:
            pts_full = np.empty((0, 2), dtype=float)

        params = self._ring_ellipse_params(ring_idx0)
        if params is None and pts_full.shape[0] >= 5:
            params = fit_initial_ellipse(pts_full)

        if params is not None:
            xc, yc, a, b, theta = [float(v) for v in params]
            c, s = np.cos(theta), np.sin(theta)
            mean_r = max(0.5 * (abs(a) + abs(b)), 10.0)
            if pts_full.shape[0] >= 1:
                dx = pts_full[:, 0] - xc
                dy = pts_full[:, 1] - yc
                u = dx * c + dy * s
                v = -dx * s + dy * c
                phi = np.mod(np.arctan2(v / max(abs(b), 1e-6), u / max(abs(a), 1e-6)), 2.0 * np.pi)
                t0 = self._largest_angle_gap_mid(phi)
            else:
                t0 = 0.0

            dth = float(np.clip(10.0 / max(mean_r, 1.0), 0.020, 0.18))
            offsets = np.array([0.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0, -4.0, 4.0], dtype=float) * dth
            t_candidates = [t0 + float(off) for off in offsets]
            seeds_xy = []
            for tt in t_candidates:
                ct, st = np.cos(tt), np.sin(tt)
                xs = xc + a * ct * c - b * st * s
                ys = yc + a * ct * s + b * st * c
                seeds_xy.append((float(xs), float(ys)))

            best = self._best_snap_from_seeds(
                img_log_full,
                params,
                seeds_xy,
                search_along=max(DEFAULT_CLICK_SEARCH_ALONG, 18.0),
                search_across=max(DEFAULT_CLICK_SEARCH_ACROSS, 5.0),
                search_step=min(DEFAULT_CLICK_SEARCH_STEP, 0.35),
            )
            if best is not None:
                x_full, y_full = float(best[0]), float(best[1])
            else:
                x_full, y_full = float(seeds_xy[0][0]), float(seeds_xy[0][1])

            r_full = float(np.clip(0.035 * mean_r, 10.0, 26.0))
        else:
            cx, cy = self._default_center_full()
            if not (np.isfinite(cx) and np.isfinite(cy)):
                return None
            if pts_full.shape[0] >= 1:
                ang = np.mod(np.arctan2(pts_full[:, 1] - cy, pts_full[:, 0] - cx), 2.0 * np.pi)
                t0 = self._largest_angle_gap_mid(ang)
                rr = np.median(np.hypot(pts_full[:, 0] - cx, pts_full[:, 1] - cy))
                rr = float(max(rr, 20.0))
            else:
                t0 = 0.0
                rr = 0.25 * float(min(img_log_full.shape[0], img_log_full.shape[1]))

            dth = float(np.clip(10.0 / max(rr, 1.0), 0.025, 0.20))
            offsets = np.array([0.0, -1.0, 1.0, -2.0, 2.0, -3.0, 3.0, -4.0, 4.0], dtype=float) * dth
            t_candidates = [t0 + float(off) for off in offsets]
            seeds_xy = [(float(cx + rr * np.cos(tt)), float(cy + rr * np.sin(tt))) for tt in t_candidates]
            params_c = (float(cx), float(cy), float(rr), float(rr), 0.0)
            best = self._best_snap_from_seeds(
                img_log_full,
                params_c,
                seeds_xy,
                search_along=max(DEFAULT_CLICK_SEARCH_ALONG, 18.0),
                search_across=max(DEFAULT_CLICK_SEARCH_ACROSS, 5.0),
                search_step=min(DEFAULT_CLICK_SEARCH_STEP, 0.35),
            )
            if best is not None:
                x_full, y_full = float(best[0]), float(best[1])
            else:
                x_full, y_full = seeds_xy[0]
            r_full = float(np.clip(0.055 * rr, 10.0, 28.0))

        return {
            "ring_index": int(ring_idx0 + 1),
            "x_ds": float(x_full),
            "y_ds": float(y_full),
            "r_ds": float(max(r_full, 2.0)),
        }

    def _update_suggested_regions(self):
        if not self._show_suggested_regions:
            return
        self.suggested_regions = []
        if self.img_disp is None:
            self._clear_suggested_regions()
            return

        cap = safe_int(self.points_per_ring.get(), DEFAULT_POINTS_PER_RING, minimum=1)
        for ring_idx0 in range(len(self.points_ds)):
            if len(self.points_ds[ring_idx0]) >= cap:
                continue
            s = self._suggest_region_for_ring(ring_idx0)
            if s is not None:
                self.suggested_regions.append(s)

        if not self.suggested_regions:
            self._clear_suggested_regions()

    def _highres_refine_settings(self):
        # Estimate radial localization sigma from current fit residuals.
        sigma_r = np.nan
        if self.fit_quality:
            rp = np.asarray(self.fit_quality.get("residual_px", np.array([], dtype=float)), dtype=float).reshape(-1)
            rp = rp[np.isfinite(rp)]
            if rp.size > 0:
                sigma_r = float(np.median(rp))
        if not np.isfinite(sigma_r):
            sigma_r = 0.35
        sigma_r = float(np.clip(sigma_r, 0.12, 2.5))

        # Use largest ring to set angular density.
        r_max = np.nan
        if self.ellipses:
            rr = []
            for e in self.ellipses:
                try:
                    rr.append(0.5 * (abs(float(e["a"])) + abs(float(e["b"]))))
                except Exception:
                    continue
            if rr:
                r_max = float(np.nanmax(np.asarray(rr, dtype=float)))
        if not np.isfinite(r_max):
            r_max = 1200.0
        r_max = float(max(r_max, 100.0))

        # Target tangential spacing tied to radial uncertainty.
        # Spacing tighter than ~8*sigma_r has diminishing gains for ellipse parameters.
        arc_target = float(np.clip(8.0 * sigma_r, 3.0, 8.0))
        n_tan = float((2.0 * np.pi * r_max) / arc_target)

        # Also enforce enough independent points for center precision target.
        # Center stderr ~ sigma_r / sqrt(N): ask for <= 0.02 px when feasible.
        center_target = 0.02
        n_center = float((sigma_r / center_target) ** 2)

        n_angles = int(np.ceil(max(n_tan, n_center) / 36.0) * 36.0)
        n_angles = int(np.clip(n_angles, 720, FINAL_REFINE_N_ANGLES_FALLBACK))

        step = float(np.clip(sigma_r / 2.5, FINAL_REFINE_STEP_MIN, FINAL_REFINE_STEP_MAX))
        dr = float(np.clip(4.0 * sigma_r + 2.0, 4.0, FINAL_REFINE_DR_MAX))
        return {
            "sigma_r": sigma_r,
            "r_max": r_max,
            "angles": int(n_angles),
            "dr": float(dr),
            "step": float(step),
            "iters": int(FINAL_REFINE_ITERS),
        }

    def _fit_conf_overall(self):
        if not self.fit_quality:
            return np.nan
        return float(self.fit_quality.get("overall_confidence", np.nan))

    def _ring_map(self, ellipses):
        out = {}
        for i, e in enumerate(ellipses, start=1):
            rid = int(e.get("ring_index", i))
            out[rid] = e
        return out

    def _ellipse_solution_delta(self, e1, e2):
        m1 = self._ring_map(e1)
        m2 = self._ring_map(e2)
        common = sorted(set(m1.keys()) & set(m2.keys()))
        if not common:
            return {
                "center_shift_px": np.nan,
                "axis_shift_px": np.nan,
                "angle_shift_deg": np.nan,
            }

        dc = []
        da = []
        dt = []
        for rid in common:
            a = m1[rid]
            b = m2[rid]
            dxy = np.hypot(float(a["xc"]) - float(b["xc"]), float(a["yc"]) - float(b["yc"]))
            dax = 0.5 * (abs(float(a["a"]) - float(b["a"])) + abs(float(a["b"]) - float(b["b"])))
            dth = abs(float(a["theta"]) - float(b["theta"]))
            dth = min(dth, 2.0 * np.pi - dth)
            dc.append(float(dxy))
            da.append(float(dax))
            dt.append(float(np.rad2deg(dth)))

        return {
            "center_shift_px": float(np.median(np.asarray(dc, dtype=float))),
            "axis_shift_px": float(np.median(np.asarray(da, dtype=float))),
            "angle_shift_deg": float(np.median(np.asarray(dt, dtype=float))),
        }

    def _scale_point_lists(self, scale):
        if not np.isfinite(scale) or abs(float(scale) - 1.0) < 1e-12:
            return

        def _rescale(rings):
            out = []
            for ring in rings:
                arr = np.asarray(ring, dtype=float).reshape(-1, 2)
                if arr.size == 0:
                    out.append([])
                    continue
                arr *= float(scale)
                out.append([(float(x), float(y)) for x, y in arr])
            return out

        self.points_ds = _rescale(self.points_ds)
        if self.points_raw_ds:
            self.points_raw_ds = _rescale(self.points_raw_ds)
        if self.points_sigma_ds:
            scaled = []
            for ring in self.points_sigma_ds:
                arr = np.asarray(ring, dtype=float).reshape(-1)
                if arr.size == 0:
                    scaled.append([])
                    continue
                arr = arr * float(scale)
                arr[~np.isfinite(arr)] = np.nan
                scaled.append([float(v) if np.isfinite(v) and v > 0.0 else np.nan for v in arr])
            self.points_sigma_ds = scaled
        self._sync_point_sigma_shape()

    def _points_for_fit(self, points=None, downsample=None):
        ds = max(float(self.down if downsample is None else downsample), 1.0)
        src = self.points_ds if points is None else points
        out = []
        for ring in src:
            arr = np.asarray(ring, dtype=float).reshape(-1, 2)
            if arr.size == 0:
                out.append([])
                continue
            arr_fit = arr / ds
            out.append([(float(x), float(y)) for x, y in arr_fit])
        return out

    def _sigma_for_fit(self, sigmas=None, downsample=None):
        ds = max(float(self.down if downsample is None else downsample), 1.0)
        src = self.points_sigma_ds if sigmas is None else sigmas
        out = []
        for ring in src:
            arr = np.asarray(ring, dtype=float).reshape(-1)
            if arr.size == 0:
                out.append([])
                continue
            arr_fit = arr / ds
            arr_fit[~np.isfinite(arr_fit)] = np.nan
            out.append([float(v) if np.isfinite(v) and v > 0.0 else np.nan for v in arr_fit])
        return out

    def _apply_downsample_from_ui(self, reset_view=False):
        new_down = safe_int(self.downsample.get(), self.down, minimum=1)
        self.downsample.set(str(new_down))
        old_down = safe_int(self.down, DEFAULT_DOWNSAMPLE, minimum=1)
        if int(new_down) == int(old_down):
            return False

        self.down = int(new_down)
        if self.img_log is not None:
            # Viewer always stays full-resolution; downsample only affects fitting.
            self.img_disp = build_display(self.img_log, 1)

        self._cancel_precision_pick()
        self._cancel_drag_edit()
        self._pan_active = False
        self._pan_anchor = None

        if self.ellipses and self.points_ds and self.img_log is not None:
            fit_points_ds = self._points_for_fit()
            self.fit_quality = compute_fit_confidence(self.img_log, fit_points_ds, self.ellipses, self.down)
        elif not self.ellipses:
            self.fit_quality = None
        self._update_fit_text()
        self._update_progress()

        if self.img_disp is not None:
            self.refresh_plot(reset_view=bool(reset_view), view_scale=1.0)
        return True

    def _ring_snap_sigma_map(self, point_sigmas=None):
        src = self.points_sigma_ds if point_sigmas is None else point_sigmas
        out = {}
        for i, ring in enumerate(src, start=1):
            arr = np.asarray(ring, dtype=float).reshape(-1)
            arr = arr[np.isfinite(arr) & (arr > 0.0)]
            if arr.size > 0:
                out[int(i)] = float(np.median(arr))
        return out

    def _build_optimizer_sets(self, dense, ellipses=None, points_ds=None, downsample=None, fit_quality=None):
        dense = safe_int(dense, DEFAULT_DENSE_POINTS, minimum=60)

        ring_ids = []
        sets = []
        ell = self.ellipses if ellipses is None else ellipses
        if ell:
            for i, e in enumerate(ell, start=1):
                x, y = ellipse_curve(e["xc"], e["yc"], e["a"], e["b"], e["theta"], num=dense)
                sets.append(np.column_stack([x, y]))
                ring_ids.append(int(e.get("ring_index", i)))
        else:
            pts_src = self.points_ds if points_ds is None else points_ds
            for i, p in enumerate(pts_src, start=1):
                arr = np.asarray(p, dtype=float)
                if arr.shape[0] >= 5:
                    sets.append(arr)
                    ring_ids.append(int(i))

        if not sets:
            return [], [], None

        weights = None
        fq = self.fit_quality if fit_quality is None else fit_quality
        if fq:
            conf = np.asarray(fq.get("per_ring_confidence", np.array([], dtype=float)), dtype=float).reshape(-1)
            conf_rings = np.asarray(fq.get("ring_indices", np.array([], dtype=int)), dtype=int).reshape(-1)
            if conf.size > 0 and conf_rings.size == conf.size:
                conf_map = {int(r): float(c) for r, c in zip(conf_rings, conf)}
                filt_sets = []
                filt_ids = []
                w = []
                for pts, rid in zip(sets, ring_ids):
                    c = conf_map.get(int(rid), np.nan)
                    if np.isfinite(c) and c < 8.0:
                        continue
                    filt_sets.append(pts)
                    filt_ids.append(int(rid))
                    if np.isfinite(c):
                        w.append(np.clip(c / 100.0, 0.2, 1.0))
                    else:
                        w.append(1.0)
                if len(filt_sets) >= 2:
                    sets = filt_sets
                    ring_ids = filt_ids
                    weights = np.asarray(w, dtype=float)
                else:
                    w_full = []
                    for rid in ring_ids:
                        c = conf_map.get(int(rid), np.nan)
                        if np.isfinite(c):
                            w_full.append(np.clip(c / 100.0, 0.2, 1.0))
                        else:
                            w_full.append(1.0)
                    if w_full:
                        weights = np.asarray(w_full, dtype=float)
        sigma_map = self._ring_snap_sigma_map()
        if sigma_map:
            sig = np.asarray([sigma_map.get(int(rid), np.nan) for rid in ring_ids], dtype=float)
            valid = np.isfinite(sig) & (sig > 0.0)
            if np.any(valid):
                ref = float(np.median(sig[valid]))
                ref = float(np.clip(ref, 0.35, 4.0))
                w_sigma = np.ones(len(ring_ids), dtype=float)
                denom = np.clip(sig[valid], 0.20, DEFAULT_SNAP_UNCERT_MAX_PX)
                w_sigma[valid] = np.clip((ref / denom) ** 2, 0.25, 4.0)
                if weights is None:
                    weights = w_sigma
                else:
                    weights = np.asarray(weights, dtype=float).reshape(-1) * w_sigma
                    weights = np.clip(weights, 0.15, 4.0)

        return sets, ring_ids, weights

    def _apply_optimization_result(self, optim, ring_ids, source):
        self.optim = optim
        self.optim["center_source"] = source
        self.optim["ring_ids"] = np.asarray(ring_ids, dtype=np.int32)
        self.corrected = self.optim["corrected_points"]

        if "center" in self.optim:
            cxo, cyo = self.optim["center"]
            if np.isfinite(cxo) and np.isfinite(cyo):
                self.center_x.set(f"{cxo:.3f}")
                self.center_y.set(f"{cyo:.3f}")

        self.opt_text.set(
            f"tilt_x={self.optim['tilt_x_deg']:.4f} deg, tilt_y={self.optim['tilt_y_deg']:.4f} deg, cost={self.optim['cost_final']:.3e}"
        )

    def _get_center(self, strict=True):
        cx = self.center_x.get().strip()
        cy = self.center_y.get().strip()
        if cx and cy:
            try:
                x, y = float(cx), float(cy)
                if np.isfinite(x) and np.isfinite(y):
                    return (x, y), "ui_entry"
            except Exception:
                if strict:
                    raise ValueError("Center entries are invalid.")
        if self.ellipses:
            x, y = ellipse_center(self.ellipses, fit_quality=self.fit_quality)
            if np.isfinite(x) and np.isfinite(y):
                return (x, y), "ellipse_mean"
        if strict:
            raise ValueError("Center unavailable. Fit ellipses or enter center values.")
        return (np.nan, np.nan), "unknown"

    def _default_center_full(self):
        try:
            (cx, cy), _ = self._get_center(strict=False)
        except Exception:
            cx, cy = np.nan, np.nan
        if np.isfinite(cx) and np.isfinite(cy):
            return float(cx), float(cy)
        img_log_full = self._get_fullres_log_image()
        if img_log_full is not None:
            h, w = img_log_full.shape
            return 0.5 * float(w - 1), 0.5 * float(h - 1)
        return np.nan, np.nan

    def _get_fullres_log_image(self):
        if self.img_bgsub is not None:
            bg = np.asarray(self.img_bgsub, dtype=np.float32)
            if bg.ndim == 2 and bg.size > 0:
                needs_rebuild = True
                if self.img_log is not None:
                    log = np.asarray(self.img_log, dtype=np.float32)
                    if log.ndim == 2 and log.shape == bg.shape:
                        self.img_log = log
                        needs_rebuild = False
                if needs_rebuild:
                    self.img_log = make_log_image(bg)
                return self.img_log

        if self.img_log is None:
            return None
        log = np.asarray(self.img_log, dtype=np.float32)
        if log.ndim != 2 or log.size == 0:
            return None
        self.img_log = log
        return self.img_log

    def _ring_ellipse_params(self, ring_idx0):
        rid = int(ring_idx0 + 1)
        for i, e in enumerate(self.ellipses, start=1):
            erid = int(e.get("ring_index", i))
            if erid == rid:
                return (float(e["xc"]), float(e["yc"]), float(e["a"]), float(e["b"]), float(e["theta"]))
        return None

    def _estimate_snap_uncertainty_px(
        self,
        img_log_full,
        params,
        x_seed,
        y_seed,
        x_snap,
        y_snap,
        search_along,
        search_across,
        search_step,
    ):
        base = {
            "snap_sigma_px": np.nan,
            "snap_seed_count": 0,
            "snap_seed_spread_px": np.nan,
            "snap_seed_bias_px": np.nan,
        }
        if img_log_full is None or params is None:
            return base

        seeds = []
        for r in DEFAULT_SNAP_UNCERT_RADII_PX:
            rr = float(max(r, 0.0))
            if rr <= 1e-12:
                seeds.append((float(x_seed), float(y_seed), 0.0))
                continue
            n_ang = int(max(DEFAULT_SNAP_UNCERT_N_ANGLES, 4))
            for k in range(n_ang):
                t = (2.0 * np.pi * float(k)) / float(n_ang)
                seeds.append(
                    (
                        float(x_seed + rr * np.cos(t)),
                        float(y_seed + rr * np.sin(t)),
                        rr,
                    )
                )

        along_u = float(min(float(search_along), 10.0))
        across_u = float(min(float(search_across), 3.0))
        step_u = float(max(float(search_step), 0.75))

        samples = []
        posts = []
        radii = []
        for xs, ys, rr in seeds:
            snapped, meta = snap_points_to_ring(
                img_log_full,
                np.array([[float(xs), float(ys)]], dtype=float),
                params,
                search_along=along_u,
                search_across=across_u,
                search_step=step_u,
                use_pseudovoigt=False,
                enforce_confidence=False,
                return_meta=True,
            )
            if snapped.shape[0] < 1 or not np.all(np.isfinite(snapped[0])):
                continue
            m0 = meta[0] if meta else {}
            samples.append((float(snapped[0, 0]), float(snapped[0, 1])))
            posts.append(float(m0.get("best_posterior", np.nan)))
            radii.append(float(rr))

        n = len(samples)
        base["snap_seed_count"] = int(n)
        if n < 2:
            return base

        arr = np.asarray(samples, dtype=float).reshape(-1, 2)
        p = np.asarray(posts, dtype=float).reshape(-1)
        r = np.asarray(radii, dtype=float).reshape(-1)
        p_finite = np.where(np.isfinite(p), p, np.nanmin(p[np.isfinite(p)]) if np.any(np.isfinite(p)) else 0.0)
        p_max = float(np.max(p_finite)) if p_finite.size > 0 else 0.0
        w_post = np.exp(np.clip(p_finite - p_max, -8.0, 0.0))
        w_rad = 1.0 / (1.0 + (r / 2.0) ** 2)
        w = w_post * w_rad
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 1e-12:
            w = np.ones(arr.shape[0], dtype=float)
            w_sum = float(np.sum(w))
        w = w / max(w_sum, 1e-12)

        mu_x = float(np.sum(w * arr[:, 0]))
        mu_y = float(np.sum(w * arr[:, 1]))
        d_mu = np.hypot(arr[:, 0] - mu_x, arr[:, 1] - mu_y)
        scatter = float(np.sqrt(max(np.sum(w * (d_mu ** 2)), 0.0)))
        bias = float(np.hypot(mu_x - float(x_snap), mu_y - float(y_snap)))
        spread_from_snap = float(np.sqrt(max(np.sum(w * ((np.hypot(arr[:, 0] - float(x_snap), arr[:, 1] - float(y_snap))) ** 2)), 0.0)))
        sigma = float(np.sqrt(scatter * scatter + bias * bias))
        sigma = float(np.clip(sigma, DEFAULT_SNAP_UNCERT_MIN_PX, DEFAULT_SNAP_UNCERT_MAX_PX))

        base["snap_sigma_px"] = sigma
        base["snap_seed_spread_px"] = spread_from_snap
        base["snap_seed_bias_px"] = bias
        return base

    def _snap_click_point_ds(
        self,
        ring_idx0,
        x_ds,
        y_ds,
        enforce_confidence=True,
        return_meta=False,
        preview_fast=False,
    ):
        base_meta = {
            "used_snap": False,
            "reason": "manual",
            "best_posterior": np.nan,
            "posterior_margin": np.nan,
            "best_peak_snr": np.nan,
            "best_resid_px": np.nan,
            "best_tangent_offset_px": np.nan,
            "best_click_distance_px": np.nan,
            "candidate_count": 0,
            "snap_sigma_px": np.nan,
            "snap_seed_count": 0,
            "snap_seed_spread_px": np.nan,
            "snap_seed_bias_px": np.nan,
        }

        def _return(xv, yv, meta_override=None):
            meta_out = dict(base_meta)
            if meta_override:
                meta_out.update(meta_override)
            if return_meta:
                return float(xv), float(yv), meta_out
            return float(xv), float(yv)

        img_log_full = self._get_fullres_log_image()
        if img_log_full is None:
            return _return(x_ds, y_ds, {"reason": "image_unavailable"})

        px = float(x_ds)
        py = float(y_ds)
        params = self._ring_ellipse_params(ring_idx0)

        if params is None:
            candidate = list(self.points_ds[ring_idx0]) + [(float(x_ds), float(y_ds))]
            if len(candidate) >= 5:
                arr_full = np.asarray(candidate, dtype=float)
                params = fit_initial_ellipse(arr_full)

        if params is None:
            cx, cy = self._default_center_full()
            if not (np.isfinite(cx) and np.isfinite(cy)):
                return _return(x_ds, y_ds, {"reason": "center_unavailable"})
            rr = max(np.hypot(px - cx, py - cy), 3.0)
            params = (float(cx), float(cy), float(rr), float(rr), 0.0)

        search_along = float(DEFAULT_CLICK_SEARCH_ALONG)
        search_across = float(DEFAULT_CLICK_SEARCH_ACROSS)
        search_step = float(DEFAULT_CLICK_SEARCH_STEP)
        use_pv = True
        conf_gate = bool(enforce_confidence)
        if preview_fast:
            # Lightweight live preview during mouse motion; final commit still
            # runs full-accuracy snapping.
            search_along = float(min(DEFAULT_CLICK_SEARCH_ALONG, 10.0))
            search_across = float(min(DEFAULT_CLICK_SEARCH_ACROSS, 2.5))
            search_step = float(max(DEFAULT_CLICK_SEARCH_STEP, 1.0))
            use_pv = False
            conf_gate = False

        snapped, snap_meta = snap_points_to_ring(
            img_log_full,
            np.array([[px, py]], dtype=float),
            params,
            search_along=search_along,
            search_across=search_across,
            search_step=search_step,
            use_pseudovoigt=use_pv,
            enforce_confidence=conf_gate,
            return_meta=True,
        )
        if snapped.shape[0] >= 1 and np.all(np.isfinite(snapped[0])):
            meta0 = snap_meta[0] if snap_meta else {"reason": "missing_meta"}
            if not preview_fast:
                extra = self._estimate_snap_uncertainty_px(
                    img_log_full,
                    params,
                    float(px),
                    float(py),
                    float(snapped[0, 0]),
                    float(snapped[0, 1]),
                    search_along,
                    search_across,
                    search_step,
                )
                meta0 = dict(meta0)
                meta0.update(extra)
                if not np.isfinite(float(meta0.get("snap_sigma_px", np.nan))):
                    resid = float(meta0.get("best_resid_px", np.nan))
                    cdist = float(meta0.get("best_click_distance_px", np.nan))
                    fallback = np.nan
                    if np.isfinite(resid) or np.isfinite(cdist):
                        rr = max(float(resid), 0.0) if np.isfinite(resid) else 0.0
                        dd = max(float(cdist), 0.0) if np.isfinite(cdist) else 0.0
                        fallback = np.sqrt((0.40 * rr) ** 2 + (0.12 * dd) ** 2)
                    if np.isfinite(fallback):
                        meta0["snap_sigma_px"] = float(
                            np.clip(fallback, DEFAULT_SNAP_UNCERT_MIN_PX, DEFAULT_SNAP_UNCERT_MAX_PX)
                        )
            return _return(snapped[0, 0], snapped[0, 1], meta0)
        return _return(x_ds, y_ds, {"reason": "invalid_snap"})

    def _nearest_existing_point_ds(self, x_ds, y_ds, max_dist_ds=DEFAULT_EDIT_SELECT_RADIUS_DS):
        if not self.points_ds:
            return None

        max_dist = float(max(max_dist_ds, 0.5))
        try:
            x0, x1 = self.ax.get_xlim()
            y0, y1 = self.ax.get_ylim()
            view_span = max(abs(float(x1) - float(x0)), abs(float(y1) - float(y0)))
            max_dist = max(max_dist, 0.0125 * view_span)
        except Exception:
            pass

        best = None
        best_d2 = max_dist * max_dist
        x = float(x_ds)
        y = float(y_ds)
        for ring_idx, ring_pts in enumerate(self.points_ds):
            arr = np.asarray(ring_pts, dtype=float).reshape(-1, 2)
            if arr.size == 0:
                continue
            d2 = (arr[:, 0] - x) ** 2 + (arr[:, 1] - y) ** 2
            j = int(np.argmin(d2))
            d2j = float(d2[j])
            if np.isfinite(d2j) and d2j <= best_d2:
                best_d2 = d2j
                best = (int(ring_idx), int(j), float(np.sqrt(d2j)))
        return best

    def _show_drag_edit_preview(self, x_ds, y_ds, x_snap=None, y_snap=None):
        if self._drag_edit_preview_artist is None or self._drag_edit_preview_artist.axes is not self.ax:
            (artist,) = self.ax.plot(
                [x_ds],
                [y_ds],
                marker="D",
                markerfacecolor="none",
                markeredgecolor="yellow",
                markersize=8,
                markeredgewidth=1.6,
                linestyle="none",
                zorder=11,
            )
            self._drag_edit_preview_artist = artist
        else:
            self._drag_edit_preview_artist.set_data([x_ds], [y_ds])
            self._drag_edit_preview_artist.set_visible(True)

        if (
            x_snap is None
            or y_snap is None
            or not (np.isfinite(float(x_snap)) and np.isfinite(float(y_snap)))
        ):
            if self._drag_edit_preview_snap_artist is not None:
                self._drag_edit_preview_snap_artist.set_visible(False)
            if self._drag_edit_preview_link_artist is not None:
                self._drag_edit_preview_link_artist.set_visible(False)
            return

        if self._drag_edit_preview_snap_artist is None or self._drag_edit_preview_snap_artist.axes is not self.ax:
            (s_artist,) = self.ax.plot(
                [x_snap],
                [y_snap],
                marker="o",
                markerfacecolor="none",
                markeredgecolor="cyan",
                markersize=9,
                markeredgewidth=1.8,
                linestyle="none",
                zorder=12,
            )
            self._drag_edit_preview_snap_artist = s_artist
        else:
            self._drag_edit_preview_snap_artist.set_data([x_snap], [y_snap])
            self._drag_edit_preview_snap_artist.set_visible(True)

        if self._drag_edit_preview_link_artist is None or self._drag_edit_preview_link_artist.axes is not self.ax:
            (l_artist,) = self.ax.plot(
                [x_ds, x_snap],
                [y_ds, y_snap],
                "-",
                color="cyan",
                linewidth=1.1,
                alpha=0.75,
                zorder=11,
            )
            self._drag_edit_preview_link_artist = l_artist
        else:
            self._drag_edit_preview_link_artist.set_data([x_ds, x_snap], [y_ds, y_snap])
            self._drag_edit_preview_link_artist.set_visible(True)

    def _cancel_drag_edit(self):
        self._drag_edit_active = False
        self._drag_edit_ring_idx = None
        self._drag_edit_point_idx = None
        self._drag_edit_restore_limits = None
        self._drag_edit_preview_xy = None
        self._drag_edit_preview_snap_xy = None
        self._drag_edit_preview_last_t = 0.0
        self._drag_edit_preview_last_xy = None
        if self._drag_edit_preview_artist is not None:
            try:
                self._drag_edit_preview_artist.remove()
            except Exception:
                self._drag_edit_preview_artist.set_visible(False)
            self._drag_edit_preview_artist = None
        if self._drag_edit_preview_snap_artist is not None:
            try:
                self._drag_edit_preview_snap_artist.remove()
            except Exception:
                self._drag_edit_preview_snap_artist.set_visible(False)
            self._drag_edit_preview_snap_artist = None
        if self._drag_edit_preview_link_artist is not None:
            try:
                self._drag_edit_preview_link_artist.remove()
            except Exception:
                self._drag_edit_preview_link_artist.set_visible(False)
            self._drag_edit_preview_link_artist = None

    def _snap_status_detail(self, x_raw, y_raw, x_snap, y_snap, meta):
        info = meta if isinstance(meta, dict) else {}
        used = bool(info.get("used_snap", False))
        reason = str(info.get("reason", "manual"))
        delta = float(np.hypot(float(x_snap) - float(x_raw), float(y_snap) - float(y_raw)))
        snr = float(info.get("best_peak_snr", np.nan))
        margin = float(info.get("posterior_margin", np.nan))
        sigma = float(info.get("snap_sigma_px", np.nan))

        if not used:
            if np.isfinite(sigma):
                return f"snap=off ({reason}), d={delta:.2f}px, sigma={sigma:.2f}px"
            return f"snap=off ({reason}), d={delta:.2f}px"

        parts = [f"snap=on, d={delta:.2f}px"]
        if np.isfinite(snr):
            parts.append(f"snr={snr:.2f}")
        if np.isfinite(margin) and margin < 1e6:
            parts.append(f"margin={margin:.2f}")
        if np.isfinite(sigma):
            parts.append(f"sigma={sigma:.2f}px")
        return ", ".join(parts)

    def _start_drag_edit(self, x_ds, y_ds):
        hit = self._nearest_existing_point_ds(x_ds, y_ds)
        if hit is None:
            self.status.set("Edit mode: click closer to an existing point (zoom in for finer control).")
            return

        ring_idx, point_idx, dist_ds = hit
        if ring_idx >= len(self.points_ds) or point_idx >= len(self.points_ds[ring_idx]):
            self.status.set("Edit mode: selected point is unavailable.")
            return

        x0, y0 = self.points_ds[ring_idx][point_idx]
        self._drag_edit_active = True
        self._drag_edit_ring_idx = int(ring_idx)
        self._drag_edit_point_idx = int(point_idx)
        self._drag_edit_restore_limits = (tuple(self.ax.get_xlim()), tuple(self.ax.get_ylim()))
        self._drag_edit_preview_xy = (float(x0), float(y0))
        x0_snap, y0_snap = self._snap_click_point_ds(
            int(ring_idx),
            float(x0),
            float(y0),
            enforce_confidence=False,
            preview_fast=True,
        )
        self._drag_edit_preview_snap_xy = (float(x0_snap), float(y0_snap))
        self._drag_edit_preview_last_t = time.perf_counter()
        self._drag_edit_preview_last_xy = (float(x0), float(y0))
        self._set_precision_box(float(x0), float(y0), size_ds=DEFAULT_PRECISION_PICK_SIZE_DS)
        self._show_drag_edit_preview(float(x0), float(y0), float(x0_snap), float(y0_snap))
        ring_total = len(self.points_ds)
        ring_n = len(self.points_ds[ring_idx])
        self.status.set(
            f"Editing ring {ring_idx + 1}/{ring_total}, point {point_idx + 1}/{ring_n} "
            f"(picked at d={dist_ds:.2f}px). 40x40 zoom active; drag and release to place (auto-snap)."
        )
        self.canvas.draw_idle()

    def _move_existing_point(self, ring_idx, point_idx, x_raw, y_raw, restore_limits=None):
        if ring_idx < 0 or ring_idx >= len(self.points_ds):
            return
        if point_idx < 0 or point_idx >= len(self.points_ds[ring_idx]):
            return

        while len(self.points_raw_ds) <= ring_idx:
            self.points_raw_ds.append([])
        while len(self.points_sigma_ds) <= ring_idx:
            self.points_sigma_ds.append([])
        raw_ring = self.points_raw_ds[ring_idx]
        sigma_ring = self.points_sigma_ds[ring_idx]
        while len(raw_ring) < point_idx:
            raw_ring.append(tuple(self.points_ds[ring_idx][len(raw_ring)]))
        while len(sigma_ring) < point_idx:
            sigma_ring.append(np.nan)

        x_snap, y_snap, snap_meta = self._snap_click_point_ds(
            int(ring_idx),
            float(x_raw),
            float(y_raw),
            enforce_confidence=True,
            return_meta=True,
        )
        if point_idx < len(raw_ring):
            raw_ring[point_idx] = (float(x_raw), float(y_raw))
        else:
            raw_ring.append((float(x_raw), float(y_raw)))
        self.points_ds[ring_idx][point_idx] = (float(x_snap), float(y_snap))
        sigma_px = float(snap_meta.get("snap_sigma_px", np.nan)) if isinstance(snap_meta, dict) else np.nan
        if point_idx < len(sigma_ring):
            sigma_ring[point_idx] = sigma_px if np.isfinite(sigma_px) and sigma_px > 0.0 else np.nan
        else:
            sigma_ring.append(sigma_px if np.isfinite(sigma_px) and sigma_px > 0.0 else np.nan)
        self._sync_point_sigma_shape()

        if self._show_suggested_regions:
            self._update_suggested_regions()

        ring_total = len(self.points_ds)
        ring_n = len(self.points_ds[ring_idx])
        snap_detail = self._snap_status_detail(float(x_raw), float(y_raw), float(x_snap), float(y_snap), snap_meta)
        self.status.set(
            f"Moved ring {ring_idx + 1}/{ring_total}, point {point_idx + 1}/{ring_n}. "
            f"placed=({x_snap:.1f},{y_snap:.1f}) px, {snap_detail}."
        )
        self.refresh_plot()
        if restore_limits is not None:
            self._restore_view_limits(restore_limits)
            self.canvas.draw_idle()

    def _set_precision_box(self, x_ds, y_ds, size_ds=DEFAULT_PRECISION_PICK_SIZE_DS):
        size_ds = max(float(size_ds), 2.0)
        half = 0.5 * size_ds
        x0_prev, x1_prev = self.ax.get_xlim()
        y0_prev, y1_prev = self.ax.get_ylim()
        if x0_prev <= x1_prev:
            self.ax.set_xlim(x_ds - half, x_ds + half)
        else:
            self.ax.set_xlim(x_ds + half, x_ds - half)
        if y0_prev <= y1_prev:
            self.ax.set_ylim(y_ds - half, y_ds + half)
        else:
            self.ax.set_ylim(y_ds + half, y_ds - half)

    def _preview_motion_due(self, mode, x_ds, y_ds):
        now = float(time.perf_counter())
        if mode == "precision":
            last_t = float(self._precision_preview_last_t)
            last_xy = self._precision_preview_last_xy
        else:
            last_t = float(self._drag_edit_preview_last_t)
            last_xy = self._drag_edit_preview_last_xy

        due = False
        if last_xy is None:
            due = True
        else:
            dx = float(x_ds) - float(last_xy[0])
            dy = float(y_ds) - float(last_xy[1])
            if (dx * dx + dy * dy) >= float(DEFAULT_PREVIEW_MIN_MOVE_PX * DEFAULT_PREVIEW_MIN_MOVE_PX):
                due = True
        if not due and (now - last_t) >= float(DEFAULT_PREVIEW_MIN_INTERVAL_S):
            due = True
        if not due:
            return False

        if mode == "precision":
            self._precision_preview_last_t = now
            self._precision_preview_last_xy = (float(x_ds), float(y_ds))
        else:
            self._drag_edit_preview_last_t = now
            self._drag_edit_preview_last_xy = (float(x_ds), float(y_ds))
        return True

    def _show_precision_preview(self, x_ds, y_ds, x_snap=None, y_snap=None):
        if self._precision_preview_artist is None or self._precision_preview_artist.axes is not self.ax:
            (artist,) = self.ax.plot(
                [x_ds],
                [y_ds],
                marker="+",
                color="yellow",
                markersize=10,
                markeredgewidth=1.8,
                linestyle="none",
                zorder=10,
            )
            self._precision_preview_artist = artist
        else:
            self._precision_preview_artist.set_data([x_ds], [y_ds])
            self._precision_preview_artist.set_visible(True)

        if (
            x_snap is None
            or y_snap is None
            or not (np.isfinite(float(x_snap)) and np.isfinite(float(y_snap)))
        ):
            if self._precision_preview_snap_artist is not None:
                self._precision_preview_snap_artist.set_visible(False)
            if self._precision_preview_link_artist is not None:
                self._precision_preview_link_artist.set_visible(False)
            return

        if self._precision_preview_snap_artist is None or self._precision_preview_snap_artist.axes is not self.ax:
            (s_artist,) = self.ax.plot(
                [x_snap],
                [y_snap],
                marker="o",
                markerfacecolor="none",
                markeredgecolor="cyan",
                markersize=9,
                markeredgewidth=1.8,
                linestyle="none",
                zorder=12,
            )
            self._precision_preview_snap_artist = s_artist
        else:
            self._precision_preview_snap_artist.set_data([x_snap], [y_snap])
            self._precision_preview_snap_artist.set_visible(True)

        if self._precision_preview_link_artist is None or self._precision_preview_link_artist.axes is not self.ax:
            (l_artist,) = self.ax.plot(
                [x_ds, x_snap],
                [y_ds, y_snap],
                "-",
                color="cyan",
                linewidth=1.1,
                alpha=0.75,
                zorder=11,
            )
            self._precision_preview_link_artist = l_artist
        else:
            self._precision_preview_link_artist.set_data([x_ds, x_snap], [y_ds, y_snap])
            self._precision_preview_link_artist.set_visible(True)

    def _restore_view_limits(self, limits):
        if limits is None:
            return
        try:
            xlim, ylim = limits
            self.ax.set_xlim(xlim[0], xlim[1])
            self.ax.set_ylim(ylim[0], ylim[1])
        except Exception:
            return

    def _cancel_precision_pick(self):
        self._precision_active = False
        self._precision_ring_idx = None
        self._precision_restore_limits = None
        self._precision_preview_xy = None
        self._precision_preview_snap_xy = None
        self._precision_preview_last_t = 0.0
        self._precision_preview_last_xy = None
        if self._precision_preview_artist is not None:
            try:
                self._precision_preview_artist.remove()
            except Exception:
                self._precision_preview_artist.set_visible(False)
            self._precision_preview_artist = None
        if self._precision_preview_snap_artist is not None:
            try:
                self._precision_preview_snap_artist.remove()
            except Exception:
                self._precision_preview_snap_artist.set_visible(False)
            self._precision_preview_snap_artist = None
        if self._precision_preview_link_artist is not None:
            try:
                self._precision_preview_link_artist.remove()
            except Exception:
                self._precision_preview_link_artist.set_visible(False)
            self._precision_preview_link_artist = None

    def _commit_point(self, ring_idx, x_raw, y_raw, restore_limits=None):
        cap = safe_int(self.points_per_ring.get(), DEFAULT_POINTS_PER_RING, minimum=1)
        ring_total = len(self.points_ds)
        point_idx = len(self.points_ds[ring_idx]) + 1
        x_snap, y_snap, snap_meta = self._snap_click_point_ds(
            int(ring_idx),
            float(x_raw),
            float(y_raw),
            enforce_confidence=True,
            return_meta=True,
        )
        self.points_raw_ds[ring_idx].append((x_raw, y_raw))
        self.points_ds[ring_idx].append((x_snap, y_snap))
        sigma_px = float(snap_meta.get("snap_sigma_px", np.nan)) if isinstance(snap_meta, dict) else np.nan
        while len(self.points_sigma_ds) <= ring_idx:
            self.points_sigma_ds.append([])
        self.points_sigma_ds[ring_idx].append(sigma_px if np.isfinite(sigma_px) and sigma_px > 0.0 else np.nan)
        self._sync_point_sigma_shape()
        nxt = self._next_ring(ring_idx)
        self.active_ring = ring_idx if nxt is None else nxt
        self._update_progress()
        if self._show_suggested_regions:
            self._update_suggested_regions()
        snap_detail = self._snap_status_detail(float(x_raw), float(y_raw), float(x_snap), float(y_snap), snap_meta)
        self.status.set(
            f"Added point {point_idx}/{cap} to ring {ring_idx + 1}/{ring_total}. "
            f"placed=({x_snap:.1f},{y_snap:.1f}) px, {snap_detail}."
        )
        self.refresh_plot()
        if restore_limits is not None:
            self._restore_view_limits(restore_limits)
            self.canvas.draw_idle()

    def _on_file_path_changed(self, *_):
        self._schedule_auto_load(force=False)

    def _on_points_per_ring_changed(self, *_):
        self._init_points(clear=False)
        self._update_progress()
        if self._show_suggested_regions:
            self._update_suggested_regions()
            if self.img_disp is not None:
                self.refresh_plot()

    def add_more_points_action(self):
        if self.img_disp is None:
            messagebox.showerror("Add Points", "Load image data first.")
            return

        cap_old = safe_int(self.points_per_ring.get(), DEFAULT_POINTS_PER_RING, minimum=1)
        cap_new = int(cap_old + 1)
        self.points_per_ring.set(str(cap_new))
        self._init_points(clear=False)
        nxt = self._next_ring(self.active_ring)
        if nxt is not None:
            self.active_ring = int(nxt)

        if self.center_pick_mode:
            self.center_pick_mode = False
            self._update_center_pick_btn_text()
        if self.edit_mode:
            self.edit_mode = False
            self._cancel_drag_edit()
            self._update_edit_btn_text()
        if not self.pick_mode:
            self.pick_mode = True
            self.pick_btn.configure(text="Disable Pick Mode")

        self._show_suggested_regions = True
        self._update_suggested_regions()
        self._update_progress()
        ring_total = len(self.points_ds)
        cap = safe_int(self.points_per_ring.get(), DEFAULT_POINTS_PER_RING, minimum=1)
        point_idx = len(self.points_ds[self.active_ring]) + 1 if ring_total > 0 else 1
        if self.suggested_regions:
            self.status.set(
                f"Points/Ring increased to {cap}. Suggested regions shown (dashed circles). "
                f"Continue picking: ring {self.active_ring + 1}/{ring_total}, point {point_idx}/{cap}."
            )
        else:
            self.status.set(
                f"Points/Ring increased to {cap}. Continue picking: ring {self.active_ring + 1}/{ring_total}, "
                f"point {point_idx}/{cap}."
            )
        self.refresh_plot()

    def _schedule_auto_load(self, force=False):
        if self._auto_load_suspended:
            return
        if self._auto_load_job is not None:
            self.root.after_cancel(self._auto_load_job)
        if force:
            self._last_loaded_pair = None
        self._auto_load_job = self.root.after(150, self._auto_load_if_ready)

    def _auto_load_if_ready(self):
        self._auto_load_job = None
        if self._auto_load_suspended:
            return

        hbn_text = self.hbn_path.get().strip()
        dark_text = self.dark_path.get().strip()
        if not hbn_text or not dark_text:
            return

        hbn = Path(hbn_text)
        dark = Path(dark_text)
        if not hbn.exists() or not dark.exists():
            self.status.set("Waiting for valid hBN and dark OSC files...")
            return

        try:
            pair = (str(hbn.resolve()), str(dark.resolve()))
        except Exception:
            pair = (str(hbn), str(dark))

        if self._last_loaded_pair == pair:
            return

        self.load_from_osc(show_errors=False, source="auto")

    def browse_hbn(self):
        p = filedialog.askopenfilename(title="Select hBN OSC", filetypes=[("OSC", "*.osc"), ("All", "*.*")])
        if p:
            self.hbn_path.set(p)
            if not self.dark_path.get().strip():
                self.status.set("hBN OSC set. Select dark OSC to auto-load.")
            self._schedule_auto_load(force=True)

    def browse_dark(self):
        p = filedialog.askopenfilename(title="Select dark OSC", filetypes=[("OSC", "*.osc"), ("All", "*.*")])
        if p:
            self.dark_path.set(p)
            if not self.hbn_path.get().strip():
                self.status.set("Dark OSC set. Select hBN OSC to auto-load.")
            self._schedule_auto_load(force=True)

    def load_from_osc(self, show_errors=True, source="manual"):
        hbn_text = self.hbn_path.get().strip()
        dark_text = self.dark_path.get().strip()
        if not hbn_text or not dark_text:
            if show_errors:
                messagebox.showerror("Load Failed", "Set both hBN and dark OSC paths.")
            return False

        hbn = Path(hbn_text)
        dark = Path(dark_text)
        if not hbn.exists() or not dark.exists():
            if show_errors:
                messagebox.showerror("Load Failed", "hBN and dark OSC paths must exist.")
            else:
                self.status.set("Waiting for valid hBN and dark OSC files...")
            return False

        try:
            pair = (str(hbn.resolve()), str(dark.resolve()))
        except Exception:
            pair = (str(hbn), str(dark))

        self.status.set("Auto-loading OSC files..." if source == "auto" else "Loading OSC files...")
        self.root.update_idletasks()
        try:
            self.img_bgsub = load_and_bgsub(str(hbn), str(dark))
            self.img_log = make_log_image(self.img_bgsub)
        except Exception as exc:
            if show_errors:
                messagebox.showerror("Load Failed", str(exc))
            self.status.set(f"Failed to load OSC files: {exc}")
            return False

        self.down = safe_int(self.downsample.get(), DEFAULT_DOWNSAMPLE, minimum=1)
        self.downsample.set(str(self.down))
        # Keep viewer full-resolution; downsample is used only for fitting math.
        self.img_disp = build_display(self.img_log, 1)
        self._init_points(clear=True)
        self.ellipses = []
        self.fit_quality = None
        self.optim = None
        self.corrected = []
        self._clear_suggested_regions()
        self.pick_mode = False
        self.center_pick_mode = False
        self._pan_active = False
        self._pan_anchor = None
        self._cancel_precision_pick()
        self._cancel_drag_edit()
        self.edit_mode = False
        self.pick_btn.configure(text="Enable Pick Mode")
        self._update_center_pick_btn_text()
        self._update_edit_btn_text()
        self._last_loaded_pair = pair
        self._update_progress()
        self._update_fit_text()
        self._update_refine_more_btn_state()
        self.opt_text.set("tilt_x=--, tilt_y=--, cost=--")
        if source == "auto":
            self.status.set("Loaded data from file paths. Enable pick mode and click ring points.")
        else:
            self.status.set("Loaded data. Enable pick mode and click ring points.")
        self.refresh_plot(reset_view=True)
        return True

    def toggle_pick(self):
        changed_ds = self._apply_downsample_from_ui(reset_view=False)
        if self.img_disp is None:
            messagebox.showerror("Pick Mode", "Load image data first.")
            return
        if self.edit_mode:
            self.edit_mode = False
            self._cancel_drag_edit()
            self._update_edit_btn_text()
        if self.center_pick_mode:
            self.center_pick_mode = False
            self._update_center_pick_btn_text()
        self.pick_mode = not self.pick_mode
        self.pick_btn.configure(text="Disable Pick Mode" if self.pick_mode else "Enable Pick Mode")
        if self.pick_mode:
            nxt = self._next_ring(self.active_ring)
            if nxt is None:
                self.status.set("All rings already full. Reset or increase points/ring.")
            else:
                self.active_ring = nxt
                cap = safe_int(self.points_per_ring.get(), DEFAULT_POINTS_PER_RING, minimum=1)
                next_pt = len(self.points_ds[self.active_ring]) + 1
                ring_total = len(self.points_ds)
                self.status.set(
                    f"Pick mode ON. Ring {self.active_ring + 1}/{ring_total}, point {next_pt}/{cap}. "
                    "Hold left=precision pick (40x40), wheel=zoom, right-drag=pan."
                )
                if changed_ds:
                    self.status.set(
                        f"Downsample set to x{self.down}. Pick mode ON. Ring {self.active_ring + 1}/{ring_total}, "
                        f"point {next_pt}/{cap}. Hold left=precision pick (40x40), wheel=zoom, right-drag=pan."
                    )
        else:
            self._pan_active = False
            self._pan_anchor = None
            self._cancel_precision_pick()
            self.status.set("Pick mode OFF.")

    def _update_center_pick_btn_text(self):
        if self.center_pick_btn is None:
            return
        self.center_pick_btn.configure(
            text="Cancel Beam Center Pick" if self.center_pick_mode else "Pick Beam Center"
        )

    def _update_edit_btn_text(self):
        if self.edit_btn is None:
            return
        self.edit_btn.configure(text="Disable Edit Mode" if self.edit_mode else "Enable Edit Mode")

    def toggle_center_pick(self):
        self._apply_downsample_from_ui(reset_view=False)
        if self.img_disp is None:
            messagebox.showerror("Beam Center", "Load image data first.")
            return
        self.center_pick_mode = not self.center_pick_mode
        self._update_center_pick_btn_text()
        if self.center_pick_mode:
            if self.pick_mode:
                self.pick_mode = False
                self.pick_btn.configure(text="Enable Pick Mode")
                self._cancel_precision_pick()
            if self.edit_mode:
                self.edit_mode = False
                self._cancel_drag_edit()
                self._update_edit_btn_text()
            self.status.set("Beam center pick ON. Left click the beam center.")
        else:
            self.status.set("Beam center pick OFF.")

    def toggle_edit_mode(self):
        self._apply_downsample_from_ui(reset_view=False)
        if self.img_disp is None:
            messagebox.showerror("Edit Mode", "Load image data first.")
            return

        self.edit_mode = not self.edit_mode
        self._update_edit_btn_text()
        if self.edit_mode:
            if self.center_pick_mode:
                self.center_pick_mode = False
                self._update_center_pick_btn_text()
            if self.pick_mode:
                self.pick_mode = False
                self.pick_btn.configure(text="Enable Pick Mode")
                self._cancel_precision_pick()
            self.status.set("Edit mode ON. Left-drag an existing point to move it; release to re-snap.")
        else:
            self._cancel_drag_edit()
            self.status.set("Edit mode OFF.")
        self.refresh_plot()

    def on_click(self, event):
        if self.img_disp is None:
            return
        if not self.pick_mode and not self.center_pick_mode and not self.edit_mode:
            return
        if event.inaxes is None or event.inaxes != self.ax:
            return

        # Do not start interactions while toolbar pan/zoom is active.
        if self.toolbar is not None and str(getattr(self.toolbar, "mode", "")).strip():
            self.status.set("Disable toolbar pan/zoom to interact with points.")
            return

        if event.button in (3, "3", "right", MouseButton.RIGHT):
            if event.xdata is None or event.ydata is None:
                return
            if self._precision_active:
                return
            self._pan_active = True
            self._pan_anchor = (
                float(event.xdata),
                float(event.ydata),
                tuple(self.ax.get_xlim()),
                tuple(self.ax.get_ylim()),
            )
            return

        is_left = event.button in (1, "1", "left", MouseButton.LEFT)
        if not is_left:
            return
        if event.xdata is None or event.ydata is None:
            return

        if self.center_pick_mode:
            x_ds = float(event.xdata)
            y_ds = float(event.ydata)
            x_full = x_ds
            y_full = y_ds
            self.center_x.set(f"{x_full:.3f}")
            self.center_y.set(f"{y_full:.3f}")
            self._center_user_defined = True
            self.center_pick_mode = False
            self._update_center_pick_btn_text()
            self.status.set(f"Beam center set to ({x_full:.2f}, {y_full:.2f}) [full-res px].")
            self.refresh_plot()
            return

        if self.edit_mode:
            self._start_drag_edit(float(event.xdata), float(event.ydata))
            return

        if not self.pick_mode:
            return

        self._init_points(clear=False)
        idx = self._next_ring(self.active_ring)
        if idx is None:
            self.status.set("All rings complete. Fit ellipses.")
            return

        x = float(event.xdata)
        y = float(event.ydata)
        self._precision_active = True
        self._precision_ring_idx = int(idx)
        self._precision_restore_limits = (tuple(self.ax.get_xlim()), tuple(self.ax.get_ylim()))
        self._precision_preview_xy = (x, y)
        x_snap, y_snap = self._snap_click_point_ds(
            int(idx),
            float(x),
            float(y),
            enforce_confidence=False,
            preview_fast=True,
        )
        self._precision_preview_snap_xy = (float(x_snap), float(y_snap))
        self._precision_preview_last_t = time.perf_counter()
        self._precision_preview_last_xy = (float(x), float(y))
        self._set_precision_box(x, y, size_ds=DEFAULT_PRECISION_PICK_SIZE_DS)
        self._show_precision_preview(x, y, x_snap, y_snap)
        ring_total = len(self.points_ds)
        point_idx = len(self.points_ds[idx]) + 1
        cap = safe_int(self.points_per_ring.get(), DEFAULT_POINTS_PER_RING, minimum=1)
        self.status.set(
            f"Precision pick: ring {idx + 1}/{ring_total}, point {point_idx}/{cap}. "
            "Drag while holding left inside 40x40 zoom, release to place."
        )
        self.canvas.draw_idle()

    def on_release(self, event):
        if event.button in (1, "1", "left", MouseButton.LEFT) and self._precision_active:
            ring_idx = self._precision_ring_idx
            restore = self._precision_restore_limits
            x = np.nan
            y = np.nan
            if event.inaxes is not None and event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
                x = float(event.xdata)
                y = float(event.ydata)
            elif self._precision_preview_xy is not None:
                x, y = self._precision_preview_xy

            self._cancel_precision_pick()
            if ring_idx is None or not (np.isfinite(x) and np.isfinite(y)):
                self._restore_view_limits(restore)
                self.canvas.draw_idle()
                return

            self._commit_point(int(ring_idx), float(x), float(y), restore_limits=restore)
            return

        if event.button in (1, "1", "left", MouseButton.LEFT) and self._drag_edit_active:
            ring_idx = self._drag_edit_ring_idx
            point_idx = self._drag_edit_point_idx
            restore = self._drag_edit_restore_limits
            x = np.nan
            y = np.nan
            if event.inaxes is not None and event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
                x = float(event.xdata)
                y = float(event.ydata)
            elif self._drag_edit_preview_xy is not None:
                x, y = self._drag_edit_preview_xy

            self._cancel_drag_edit()
            if ring_idx is None or point_idx is None or not (np.isfinite(x) and np.isfinite(y)):
                self._restore_view_limits(restore)
                self.canvas.draw_idle()
                return

            self._move_existing_point(int(ring_idx), int(point_idx), float(x), float(y), restore_limits=restore)
            return

        if event.button in (3, "3", "right", MouseButton.RIGHT):
            self._pan_active = False
            self._pan_anchor = None

    def on_motion(self, event):
        if self._precision_active:
            if event.inaxes is None or event.inaxes != self.ax:
                return
            if event.xdata is None or event.ydata is None:
                return
            x = float(event.xdata)
            y = float(event.ydata)
            self._precision_preview_xy = (x, y)
            if not self._preview_motion_due("precision", x, y):
                return
            ring_idx = self._precision_ring_idx
            if ring_idx is not None:
                x_snap, y_snap = self._snap_click_point_ds(
                    int(ring_idx),
                    x,
                    y,
                    enforce_confidence=False,
                    preview_fast=True,
                )
                self._precision_preview_snap_xy = (float(x_snap), float(y_snap))
                self._show_precision_preview(x, y, x_snap, y_snap)
            else:
                self._show_precision_preview(x, y)
            self.canvas.draw_idle()
            return

        if self._drag_edit_active:
            if event.inaxes is None or event.inaxes != self.ax:
                return
            if event.xdata is None or event.ydata is None:
                return
            x = float(event.xdata)
            y = float(event.ydata)
            self._drag_edit_preview_xy = (x, y)
            if not self._preview_motion_due("drag", x, y):
                return
            ring_idx = self._drag_edit_ring_idx
            if ring_idx is not None:
                x_snap, y_snap = self._snap_click_point_ds(
                    int(ring_idx),
                    x,
                    y,
                    enforce_confidence=False,
                    preview_fast=True,
                )
                self._drag_edit_preview_snap_xy = (float(x_snap), float(y_snap))
                self._show_drag_edit_preview(x, y, x_snap, y_snap)
            else:
                self._show_drag_edit_preview(x, y)
            self.canvas.draw_idle()
            return

        if not self._pan_active or self._pan_anchor is None:
            return
        if event.inaxes is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x0, y0, xlim0, ylim0 = self._pan_anchor
        dx = float(event.xdata) - x0
        dy = float(event.ydata) - y0
        self.ax.set_xlim(xlim0[0] - dx, xlim0[1] - dx)
        self.ax.set_ylim(ylim0[0] - dy, ylim0[1] - dy)
        self.canvas.draw_idle()

    def on_scroll(self, event):
        if self._precision_active or self._drag_edit_active:
            return
        if self.img_disp is None:
            return
        if event.inaxes is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        # matplotlib scroll uses button='up'/'down' on many backends.
        if event.button in ("up", 1):
            scale = 1.0 / 1.2
        elif event.button in ("down", -1):
            scale = 1.2
        else:
            return

        x = float(event.xdata)
        y = float(event.ydata)
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        w = float(x1 - x0)
        h = float(y1 - y0)
        if abs(w) < 1e-9 or abs(h) < 1e-9:
            return

        rx = (x - x0) / w
        ry = (y - y0) / h
        new_w = w * scale
        new_h = h * scale

        min_span = 8.0
        if abs(new_w) < min_span:
            new_w = np.sign(new_w) * min_span
        if abs(new_h) < min_span:
            new_h = np.sign(new_h) * min_span

        nx0 = x - rx * new_w
        nx1 = x + (1.0 - rx) * new_w
        ny0 = y - ry * new_h
        ny1 = y + (1.0 - ry) * new_h

        self.ax.set_xlim(nx0, nx1)
        self.ax.set_ylim(ny0, ny1)
        self.canvas.draw_idle()

    def undo_last(self):
        for i in range(len(self.points_ds) - 1, -1, -1):
            if self.points_ds[i]:
                self.points_ds[i].pop()
                if i < len(self.points_raw_ds) and self.points_raw_ds[i]:
                    self.points_raw_ds[i].pop()
                if i < len(self.points_sigma_ds) and self.points_sigma_ds[i]:
                    self.points_sigma_ds[i].pop()
                self._sync_point_sigma_shape()
                self.active_ring = i
                self._update_progress()
                self.status.set(f"Removed last point from ring {i + 1}.")
                self.refresh_plot()
                return
        self.status.set("No points to undo.")

    def reset_points(self):
        self._cancel_precision_pick()
        self._cancel_drag_edit()
        self._init_points(clear=True)
        self.ellipses = []
        self.fit_quality = None
        self.optim = None
        self.corrected = []
        self._clear_suggested_regions()
        self._update_fit_text()
        self.opt_text.set("tilt_x=--, tilt_y=--, cost=--")
        self._update_progress()
        self._update_refine_more_btn_state()
        self.status.set("Cleared points and fit results.")
        self.refresh_plot()

    def clear_fits_keep_points(self):
        self._cancel_precision_pick()
        self._cancel_drag_edit()
        had_fit = bool(self.ellipses or self.fit_quality or self.optim or self.corrected)
        self.ellipses = []
        self.fit_quality = None
        self.optim = None
        self.corrected = []
        if self._show_suggested_regions:
            self._update_suggested_regions()
        self._update_fit_text()
        self.opt_text.set("tilt_x=--, tilt_y=--, cost=--")
        self._update_refine_more_btn_state()
        if had_fit:
            self.status.set("Cleared fit/circularization results. Kept all sample points.")
        else:
            self.status.set("No fit results to clear. Sample points unchanged.")
        self.refresh_plot()

    def fit_action(self, fast_mode=False):
        ds_changed = self._apply_downsample_from_ui(reset_view=False)
        if self.img_bgsub is None:
            messagebox.showerror("Fit Failed", "Load image data first.")
            self._update_refine_more_btn_state()
            return
        self._init_points(clear=False)
        if not any(len(p) >= 5 for p in self.points_ds):
            messagebox.showerror("Fit Failed", "Need at least one ring with >=5 points.")
            self._update_refine_more_btn_state()
            return
        na = safe_int(self.refine_angles.get(), DEFAULT_REFINE_N_ANGLES, minimum=16)
        dr = safe_float(self.refine_dr.get(), DEFAULT_REFINE_DR)
        st = safe_float(self.refine_step.get(), DEFAULT_REFINE_STEP)
        if dr <= 0:
            dr = DEFAULT_REFINE_DR
        if st <= 0:
            st = DEFAULT_REFINE_STEP
        n_iter = DEFAULT_REFINE_ITERS
        use_pv = True

        prior_ellipses = [dict(e) for e in self.ellipses] if self.ellipses else []
        if fast_mode:
            na = min(int(na), int(FAST_CALIB_MAX_ANGLES))
            dr = min(float(dr), float(FAST_CALIB_MAX_DR))
            st = max(float(st), float(FAST_CALIB_MIN_STEP))
            n_iter = int(FAST_CALIB_ITERS_WARM if prior_ellipses else FAST_CALIB_ITERS_COLD)
            use_pv = False

        self.status.set("Fitting ellipses...")
        self.root.update_idletasks()
        self.fit_quality = None
        self._update_fit_text()
        points_fit_ds = self._points_for_fit()
        sigma_fit_ds = self._sigma_for_fit()
        self.ellipses = fit_ellipses(
            points_fit_ds,
            self.down,
            self.img_bgsub,
            na,
            dr,
            st,
            initial_ellipses=prior_ellipses,
            img_log=self.img_log,
            n_iter=n_iter,
            use_pseudovoigt=use_pv,
            point_sigma_ds=sigma_fit_ds,
        )
        if not self.ellipses:
            messagebox.showerror("Fit Failed", "No ellipses fitted.")
            self._update_refine_more_btn_state()
            return
        self.fit_quality = compute_fit_confidence(
            self.img_log,
            points_fit_ds,
            self.ellipses,
            self.down,
        )
        self._update_fit_text()
        if not self._center_user_defined:
            cx, cy = ellipse_center(self.ellipses, fit_quality=self.fit_quality)
            self.center_x.set(f"{cx:.3f}")
            self.center_y.set(f"{cy:.3f}")
        self.optim = None
        self.corrected = []
        self.opt_text.set("tilt_x=--, tilt_y=--, cost=--")
        fit_overall = float(self.fit_quality.get("overall_confidence", np.nan)) if self.fit_quality else np.nan
        warm_used = bool(prior_ellipses)
        if np.isfinite(fit_overall):
            if self._center_user_defined:
                tail = " Kept user center."
            else:
                tail = ""
            msg = f"Fitted {len(self.ellipses)} ellipses. Fit confidence: {fit_overall:.1f}%."
            if warm_used:
                msg += " Warm-started from previous fit."
            if ds_changed:
                msg += f" Downsample x{self.down}."
            if fast_mode:
                msg += f" Fast mode (angles={na}, step={st:.2f}, iters={n_iter})."
            self.status.set(msg + tail)
        else:
            msg = f"Fitted {len(self.ellipses)} ellipses."
            if warm_used:
                msg += " Warm-started from previous fit."
            if ds_changed:
                msg += f" Downsample x{self.down}."
            if fast_mode:
                msg += f" Fast mode (angles={na}, step={st:.2f}, iters={n_iter})."
            self.status.set(msg)
        self._update_refine_more_btn_state()
        self.refresh_plot()

    def further_refine_action(self):
        if self.img_bgsub is None:
            messagebox.showerror("Further Refine", "Load image data first.")
            return
        if not self.ellipses:
            messagebox.showerror("Further Refine", "Run Fit Ellipses first.")
            self._update_refine_more_btn_state()
            return

        self.downsample.set(str(FINAL_REFINE_DOWNSAMPLE))
        self._apply_downsample_from_ui(reset_view=False)
        self._init_points(clear=False)
        if not any(len(p) >= 5 for p in self.points_ds):
            messagebox.showerror("Further Refine", "Need at least one ring with >=5 points.")
            return

        cfg = self._highres_refine_settings()
        na = int(cfg["angles"])
        dr = float(cfg["dr"])
        st = float(cfg["step"])
        iters = int(cfg["iters"])
        sigma_r = float(cfg["sigma_r"])
        r_max = float(cfg["r_max"])

        self.refine_angles.set(str(na))
        self.refine_dr.set(f"{dr:.3f}".rstrip("0").rstrip("."))
        self.refine_step.set(f"{st:.3f}".rstrip("0").rstrip("."))

        prior_ellipses = [dict(e) for e in self.ellipses]
        conf_before = self._fit_conf_overall()
        self.status.set(
            f"Further refining (math cfg) at ds=1: sigma~{sigma_r:.3f}px, r_max~{r_max:.1f}px, "
            f"step={st:.3f}px, dR={dr:.2f}px, angles={na}..."
        )
        self.root.update_idletasks()

        points_fit_ds = self._points_for_fit()
        sigma_fit_ds = self._sigma_for_fit()
        refined = fit_ellipses(
            points_fit_ds,
            self.down,
            self.img_bgsub,
            na,
            dr,
            st,
            initial_ellipses=prior_ellipses,
            img_log=self.img_log,
            n_iter=iters,
            point_sigma_ds=sigma_fit_ds,
        )
        if not refined:
            messagebox.showerror("Further Refine", "Refinement failed.")
            return

        fitq_refined = compute_fit_confidence(
            self.img_log,
            points_fit_ds,
            refined,
            self.down,
        )

        # Verify convergence against a stricter pass.
        na_v = int(np.clip(np.ceil((na * FINAL_VERIFY_ANGLE_MULT) / 36.0) * 36.0, na, FINAL_VERIFY_MAX_ANGLES))
        st_v = float(max(st * FINAL_VERIFY_STEP_MULT, 0.10))
        dr_v = float(max(4.0, min(dr, 6.0)))
        refined_v = fit_ellipses(
            points_fit_ds,
            self.down,
            self.img_bgsub,
            na_v,
            dr_v,
            st_v,
            initial_ellipses=[dict(e) for e in refined],
            img_log=self.img_log,
            n_iter=max(iters, FINAL_REFINE_ITERS),
            point_sigma_ds=sigma_fit_ds,
        )
        use_verified = bool(refined_v)
        if use_verified:
            fitq_verified = compute_fit_confidence(
                self.img_log,
                points_fit_ds,
                refined_v,
                self.down,
            )
            delta = self._ellipse_solution_delta(refined, refined_v)
            conf_ref = float(fitq_refined.get("overall_confidence", np.nan))
            conf_ver = float(fitq_verified.get("overall_confidence", np.nan))
            conf_gain = conf_ver - conf_ref if np.isfinite(conf_ref) and np.isfinite(conf_ver) else np.nan
            stable = (
                np.isfinite(delta["center_shift_px"])
                and np.isfinite(delta["axis_shift_px"])
                and delta["center_shift_px"] <= 0.03
                and delta["axis_shift_px"] <= 0.03
                and (not np.isfinite(conf_gain) or conf_gain <= 0.15)
            )
            if stable:
                final_ell = refined
                final_fitq = fitq_refined
                verify_msg = (
                    f"verified plateau (center shift {delta['center_shift_px']:.3f}px, "
                    f"axis shift {delta['axis_shift_px']:.3f}px)"
                )
            else:
                final_ell = refined_v
                final_fitq = fitq_verified
                verify_msg = (
                    f"accepted stricter pass (dConf={conf_gain:.2f}%, center shift {delta['center_shift_px']:.3f}px)"
                )
        else:
            final_ell = refined
            final_fitq = fitq_refined
            verify_msg = "verification pass unavailable; kept primary refined result"

        self.ellipses = final_ell
        self.fit_quality = final_fitq
        self._update_fit_text()
        if not self._center_user_defined:
            cx, cy = ellipse_center(self.ellipses, fit_quality=self.fit_quality)
            self.center_x.set(f"{cx:.3f}")
            self.center_y.set(f"{cy:.3f}")

        self.optim = None
        self.corrected = []
        self.opt_text.set("tilt_x=--, tilt_y=--, cost=--")

        fit_after = self._fit_conf_overall()
        conf_before_txt = f"{conf_before:.1f}%" if np.isfinite(conf_before) else "--"
        conf_after_txt = f"{fit_after:.1f}%" if np.isfinite(fit_after) else "--"
        self.status.set(
            f"Further refinement complete. ds=1, step={st:.3f}px, angles={na}, conf {conf_before_txt}->{conf_after_txt}, {verify_msg}."
        )
        self._update_refine_more_btn_state()
        self.refresh_plot()

    def solve_all_action(self):
        if self.img_bgsub is None:
            messagebox.showerror("Solve Failed", "Load image data first.")
            return
        had_prev_fit = bool(self.ellipses)
        seed_tilts = None
        if self.optim:
            tx0 = float(self.optim.get("tilt_x_deg", np.nan))
            ty0 = float(self.optim.get("tilt_y_deg", np.nan))
            if np.isfinite(tx0) and np.isfinite(ty0):
                seed_tilts = (tx0, ty0)
        if had_prev_fit:
            self.status.set("Refining previous calibration with updated settings...")
            self.root.update_idletasks()
        self.fit_action(fast_mode=True)
        if not self.ellipses:
            return
        self.optimize_action(initial_tilts=seed_tilts)
        if self.optim:
            tx = float(self.optim.get("tilt_x_deg", np.nan))
            ty = float(self.optim.get("tilt_y_deg", np.nan))
            cf = float(self.optim.get("cost_final", np.nan))
            if np.isfinite(tx) and np.isfinite(ty) and np.isfinite(cf):
                if had_prev_fit:
                    self.status.set(
                        f"Refine complete. tilt_x={tx:.4f} deg, tilt_y={ty:.4f} deg, cost={cf:.3e}."
                    )
                else:
                    self.status.set(
                        f"Solve complete. tilt_x={tx:.4f} deg, tilt_y={ty:.4f} deg, cost={cf:.3e}."
                    )

    def center_from_fit(self):
        if not self.ellipses:
            messagebox.showerror("Center", "No ellipses fitted.")
            return
        cx, cy = ellipse_center(self.ellipses, fit_quality=self.fit_quality)
        self.center_x.set(f"{cx:.3f}")
        self.center_y.set(f"{cy:.3f}")
        self._center_user_defined = False
        self.status.set("Center set from fitted ellipses.")
        self.refresh_plot()

    def optimize_action(self, initial_tilts=None):
        self._apply_downsample_from_ui(reset_view=False)
        if self.img_bgsub is None:
            messagebox.showerror("Optimize Failed", "Load image data first.")
            return
        dense = safe_int(self.dense_points.get(), DEFAULT_DENSE_POINTS, minimum=60)
        self.dense_points.set(str(dense))
        sets, ring_ids, weights = self._build_optimizer_sets(dense=dense)
        if not sets:
            messagebox.showerror("Optimize Failed", "No valid rings for optimization.")
            return
        sigma_map = self._ring_snap_sigma_map()
        ring_sigma = np.asarray([sigma_map.get(int(rid), np.nan) for rid in ring_ids], dtype=float)

        try:
            center, source = self._get_center(strict=True)
        except Exception as exc:
            messagebox.showerror("Optimize Failed", str(exc))
            return
        use_projective = bool(self.ellipses)
        method_label = "projective" if use_projective else "legacy"
        self.status.set(f"Running optimization ({method_label})...")
        self.root.update_idletasks()
        if source == "ui_entry":
            center_prior = center
            drift_limit = DEFAULT_CENTER_DRIFT_LIMIT_PX
            prior_sigma = DEFAULT_CENTER_PRIOR_SIGMA_PX
        else:
            center_prior = None
            drift_limit = DEFAULT_CENTER_DRIFT_LIMIT_PX
            prior_sigma = DEFAULT_CENTER_PRIOR_SIGMA_PX

        try:
            if use_projective:
                self.optim = optimize_tilts_projective(
                    sets,
                    center,
                    weights=weights,
                    optimize_center=True,
                    center_prior=center_prior,
                    center_prior_sigma_px=prior_sigma,
                    center_drift_limit_px=drift_limit,
                    initial_tilts=initial_tilts,
                )
            else:
                self.optim = optimize_tilts(
                    sets,
                    center,
                    weights=weights,
                    optimize_center=True,
                    center_prior=center_prior,
                    center_prior_sigma_px=prior_sigma,
                    center_drift_limit_px=drift_limit,
                    initial_tilts=initial_tilts,
                )
        except Exception as exc:
            if not use_projective:
                messagebox.showerror("Optimize Failed", str(exc))
                return
            self.status.set(f"Projective optimization failed ({exc}). Falling back to legacy.")
            self.root.update_idletasks()
            self.optim = optimize_tilts(
                sets,
                center,
                weights=weights,
                optimize_center=True,
                center_prior=center_prior,
                center_prior_sigma_px=prior_sigma,
                center_drift_limit_px=drift_limit,
                initial_tilts=initial_tilts,
            )
            self.optim["optimizer_kind"] = "legacy_fallback"
        self.optim["ring_snap_sigma_px"] = ring_sigma
        self._apply_optimization_result(self.optim, ring_ids, source)
        cxo, cyo = self.optim.get("center", (np.nan, np.nan))
        if np.isfinite(cxo) and np.isfinite(cyo):
            if source == "ui_entry":
                cpi = self.optim.get("center_initial", (np.nan, np.nan))
                dc = np.hypot(cxo - cpi[0], cyo - cpi[1]) if np.all(np.isfinite(cpi)) else np.nan
                if np.isfinite(dc):
                    self.status.set(
                        f"Optimization finished. Center=({cxo:.2f}, {cyo:.2f}), "
                        f"drift={dc:.2f}px from provided center."
                    )
                else:
                    self.status.set(f"Optimization finished. Center=({cxo:.2f}, {cyo:.2f}).")
            else:
                self.status.set(f"Optimization finished. Center=({cxo:.2f}, {cyo:.2f}).")
        else:
            self.status.set("Optimization finished.")
        self.refresh_plot()

    def load_bundle_dialog(self):
        p = filedialog.askopenfilename(title="Load NPZ", filetypes=[("NPZ", "*.npz"), ("All", "*.*")])
        if not p:
            return
        try:
            self.load_bundle(Path(p))
        except Exception as exc:
            messagebox.showerror("Load Failed", str(exc))

    def save_bundle_dialog(self):
        if self.img_bgsub is None:
            messagebox.showerror("Save Failed", "No data to save.")
            return
        p = filedialog.asksaveasfilename(title="Save NPZ", defaultextension=".npz", filetypes=[("NPZ", "*.npz"), ("All", "*.*")])
        if not p:
            return
        try:
            self.save_bundle(Path(p))
            self.status.set(f"Saved bundle: {p}")
        except Exception as exc:
            messagebox.showerror("Save Failed", str(exc))

    def save_overlay_dialog(self):
        if self.img_disp is None:
            messagebox.showerror("Save Failed", "No display data.")
            return
        p = filedialog.asksaveasfilename(title="Save Overlay", defaultextension=".png", filetypes=[("PNG", "*.png"), ("All", "*.*")])
        if not p:
            return
        self.fig.savefig(p, dpi=300)
        self.status.set(f"Saved overlay: {p}")

    def save_bundle(self, path):
        img_log_full = self._get_fullres_log_image()
        if img_log_full is None:
            raise RuntimeError("Full-resolution log image unavailable for save.")
        center, center_src = self._get_center(strict=False)
        opt = self.optim or {}
        fitq = self.fit_quality or {}
        points_ds_save = self._points_for_fit(self.points_ds, downsample=self.down)
        points_raw_save = self._points_for_fit(self.points_raw_ds, downsample=self.down)
        points_sigma_save = self._sigma_for_fit(self.points_sigma_ds, downsample=self.down)
        tilt_x_deg_internal = float(opt.get("tilt_x_deg", np.nan))
        tilt_y_deg_internal = float(opt.get("tilt_y_deg", np.nan))
        tilt_x_deg_npz = float(tilt_x_deg_internal)
        # NPZ exchange convention: detector pitch (tilt_y) is exported with opposite sign.
        tilt_y_deg_npz = -float(tilt_y_deg_internal) if np.isfinite(tilt_y_deg_internal) else np.nan
        tilt_correction = None
        tilt_hint = None
        if np.isfinite(tilt_x_deg_npz) and np.isfinite(tilt_y_deg_npz):
            tilt_correction = {
                "tilt_x_deg": tilt_x_deg_npz,
                "tilt_y_deg": tilt_y_deg_npz,
                "source": "hbn_fitter",
                "simulation_gamma_sign_from_tilt_x": int(SIM_GAMMA_SIGN_FROM_TILT_X),
                "simulation_Gamma_sign_from_tilt_y": int(SIM_GAMMA_SIGN_FROM_TILT_Y),
            }
            rot1_rad = float(np.deg2rad(tilt_x_deg_npz))
            rot2_rad = float(np.deg2rad(tilt_y_deg_npz))
            tilt_hint = {
                "rot1_rad": rot1_rad,
                "rot2_rad": rot2_rad,
                "tilt_rad": float(np.hypot(rot1_rad, rot2_rad)),
                "simulation_gamma_sign_from_tilt_x": int(SIM_GAMMA_SIGN_FROM_TILT_X),
                "simulation_Gamma_sign_from_tilt_y": int(SIM_GAMMA_SIGN_FROM_TILT_Y),
            }
        bundle = {
            "npz_format_version": np.array(2, dtype=np.int32),
            "created_utc": np.array(dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"),
            "sim_background_rotate_k": np.array(int(SIM_BACKGROUND_ROTATE_K), dtype=np.int32),
            "tilt_correction_kind": np.array("to_flat"),
            "tilt_model": np.array("RzRx"),
            "tilt_frame": np.array("simulation_background_display"),
            "simulation_gamma_sign_from_tilt_x": np.array(
                int(SIM_GAMMA_SIGN_FROM_TILT_X), dtype=np.int32
            ),
            "simulation_Gamma_sign_from_tilt_y": np.array(
                int(SIM_GAMMA_SIGN_FROM_TILT_Y), dtype=np.int32
            ),
            "img_bgsub": self.img_bgsub.astype(np.float32),
            "img_log": img_log_full.astype(np.float32),
            "downsample_factor": np.array(self.down, dtype=np.int32),
            "point_coord_frame": np.array("downsampled"),
            "point_sigma_coord_frame": np.array("downsampled"),
            "ell_points_ds": pts_to_obj(points_ds_save),
            "ell_points_raw_ds": pts_to_obj(points_raw_save),
            "ell_points_sigma_px": scalars_to_obj(points_sigma_save),
            "ellipse_params": ellipses_to_array(self.ellipses),
            "ellipse_ring_indices": ellipse_ring_indices(self.ellipses),
            "fit_confidence_overall": np.array(float(fitq.get("overall_confidence", np.nan)), dtype=np.float64),
            "fit_confidence_per_ring": np.asarray(fitq.get("per_ring_confidence", np.array([], dtype=float)), dtype=np.float64),
            "fit_confidence_ring_indices": np.asarray(fitq.get("ring_indices", np.array([], dtype=np.int32)), dtype=np.int32),
            "fit_residual_px": np.asarray(fitq.get("residual_px", np.array([], dtype=float)), dtype=np.float64),
            "fit_signal_snr": np.asarray(fitq.get("signal_snr", np.array([], dtype=float)), dtype=np.float64),
            "fit_angular_coverage": np.asarray(fitq.get("angular_coverage", np.array([], dtype=float)), dtype=np.float64),
            "fit_points_used": np.asarray(fitq.get("n_points", np.array([], dtype=np.int32)), dtype=np.int32),
            "fit_downsample_factor": np.array(float(fitq.get("downsample_factor", np.nan)), dtype=np.float64),
            "fit_click_sigma_px": np.array(float(fitq.get("click_sigma_px", np.nan)), dtype=np.float64),
            "fit_downsample_score": np.array(float(fitq.get("downsample_score", np.nan)), dtype=np.float64),
            "detector_center": np.asarray(center, dtype=np.float64),
            "center_fixed": np.asarray(center, dtype=np.float64),
            "center_source": np.array(center_src),
            "center_initial": np.asarray(opt.get("center_initial", (np.nan, np.nan)), dtype=np.float64),
            "center_prior": np.asarray(opt.get("center_prior", (np.nan, np.nan)), dtype=np.float64),
            "center_prior_sigma_px": np.array(float(opt.get("center_prior_sigma_px", np.nan)), dtype=np.float64),
            "center_drift_limit_px": np.array(float(opt.get("center_drift_limit_px", np.nan)), dtype=np.float64),
            "tilt_x_deg": np.array(tilt_x_deg_npz, dtype=np.float64),
            "tilt_y_deg": np.array(tilt_y_deg_npz, dtype=np.float64),
            "tilt_x_deg_internal": np.array(tilt_x_deg_internal, dtype=np.float64),
            "tilt_y_deg_internal": np.array(tilt_y_deg_internal, dtype=np.float64),
            "cost_zero": np.array(float(opt.get("cost_zero", np.nan)), dtype=np.float64),
            "cost_final": np.array(float(opt.get("cost_final", np.nan)), dtype=np.float64),
            "circ_before": np.asarray(opt.get("circ_before", np.array([], dtype=float)), dtype=np.float64),
            "circ_after": np.asarray(opt.get("circ_after", np.array([], dtype=float)), dtype=np.float64),
            "radii_before": np.asarray(opt.get("radii_before", np.array([], dtype=float)), dtype=np.float64),
            "radii_after": np.asarray(opt.get("radii_after", np.array([], dtype=float)), dtype=np.float64),
            "ring_weights": np.asarray(opt.get("ring_weights", np.array([], dtype=float)), dtype=np.float64),
            "ring_snap_sigma_px": np.asarray(opt.get("ring_snap_sigma_px", np.array([], dtype=float)), dtype=np.float64),
            "optimizer_ring_ids": np.asarray(opt.get("ring_ids", np.array([], dtype=np.int32)), dtype=np.int32),
            "optimizer_kind": np.array(str(opt.get("optimizer_kind", "legacy")), dtype="<U32"),
            "projective_distance_px": np.array(float(opt.get("distance_px", np.nan)), dtype=np.float64),
            "ell_points_corrected": pts_to_obj(opt.get("corrected_points", [])),
            "ell_points_corr": pts_to_obj(opt.get("corrected_points", [])),
            "input_hbn_path": np.array(self.hbn_path.get().strip()),
            "input_dark_path": np.array(self.dark_path.get().strip()),
            # Legacy RA-SIM bundle schema compatibility:
            # downstream loaders expect these keys from ra_sim.hbn.save_bundle.
            "center": np.asarray(center, dtype=np.float64),
            "tilt_correction": tilt_correction,
            "tilt_hint": tilt_hint,
            "distance_estimate_m": opt.get("distance_estimate_m"),
            "expected_peaks": opt.get("expected_peaks"),
        }
        np.savez(path, **bundle)

    def load_bundle(self, path):
        if not Path(path).exists():
            raise FileNotFoundError(f"Bundle not found: {path}")
        if self._auto_load_job is not None:
            self.root.after_cancel(self._auto_load_job)
            self._auto_load_job = None
        d = np.load(path, allow_pickle=True)
        if "img_bgsub" not in d:
            raise KeyError("Bundle missing img_bgsub")

        self.img_bgsub = np.asarray(d["img_bgsub"], dtype=np.float32)
        # Always rebuild log intensity for display/fitting consistency, even for legacy bundles.
        self.img_log = make_log_image(self.img_bgsub)
        self._get_fullres_log_image()
        self.down = safe_int(npz_scalar(d, "downsample_factor", self.downsample.get()), DEFAULT_DOWNSAMPLE, minimum=1)
        self.downsample.set(str(self.down))
        self.img_disp = build_display(self.img_log, 1)

        self.points_ds = obj_to_pts_list(d["ell_points_ds"]) if "ell_points_ds" in d else []
        if "ell_points_raw_ds" in d:
            self.points_raw_ds = obj_to_pts_list(d["ell_points_raw_ds"])
        else:
            self.points_raw_ds = [list(r) for r in self.points_ds]
        if "ell_points_sigma_px" in d:
            self.points_sigma_ds = obj_to_scalar_lists(d["ell_points_sigma_px"])
        else:
            self.points_sigma_ds = [[np.nan] * len(r) for r in self.points_ds]
        coord_frame = npz_string(d, "point_coord_frame", "downsampled").strip().lower()
        sigma_coord_frame = npz_string(d, "point_sigma_coord_frame", coord_frame).strip().lower()
        if coord_frame != "full":
            scl = float(max(self.down, 1))
            if sigma_coord_frame == "full":
                sigma_keep = [list(r) for r in self.points_sigma_ds]
                self.points_sigma_ds = [[np.nan] * len(r) for r in self.points_ds]
                self._scale_point_lists(scl)
                self.points_sigma_ds = sigma_keep
            else:
                self._scale_point_lists(scl)
        elif sigma_coord_frame != "full":
            # Legacy bundles may provide sigma in downsampled frame while points are full.
            # Scale only the sigma vectors to keep units consistent.
            if self.points_sigma_ds:
                s = float(max(self.down, 1))
                for i, ring in enumerate(self.points_sigma_ds):
                    arr = np.asarray(ring, dtype=float).reshape(-1)
                    if arr.size == 0:
                        self.points_sigma_ds[i] = []
                        continue
                    arr = arr * s
                    arr[~np.isfinite(arr)] = np.nan
                    self.points_sigma_ds[i] = [
                        float(v) if np.isfinite(v) and float(v) > 0.0 else np.nan for v in arr
                    ]
        self.ellipses = array_to_ellipses(d["ellipse_params"]) if "ellipse_params" in d else []
        if self.ellipses:
            ring_idx_arr = d["ellipse_ring_indices"] if "ellipse_ring_indices" in d else None
            self.ellipses = apply_ellipse_ring_indices(self.ellipses, ring_idx_arr)
        n = max(len(self.points_ds), len(self.ellipses), 1)
        self.num_rings.set(str(n))
        self._init_points(clear=False)
        self._sync_point_sigma_shape()
        self.fit_quality = None
        self._clear_suggested_regions()
        self.pick_mode = False
        self.center_pick_mode = False
        self.edit_mode = False
        self._pan_active = False
        self._pan_anchor = None
        self._cancel_precision_pick()
        self._cancel_drag_edit()
        self.pick_btn.configure(text="Enable Pick Mode")
        self._update_center_pick_btn_text()
        self._update_edit_btn_text()

        hbn_in = npz_string(d, "input_hbn_path", "")
        dark_in = npz_string(d, "input_dark_path", "")
        center_src_in = npz_string(d, "center_source", "")
        self._auto_load_suspended = True
        try:
            if hbn_in:
                self.hbn_path.set(hbn_in)
            if dark_in:
                self.dark_path.set(dark_in)
        finally:
            self._auto_load_suspended = False

        hbn_now = self.hbn_path.get().strip()
        dark_now = self.dark_path.get().strip()
        if hbn_now and dark_now and Path(hbn_now).exists() and Path(dark_now).exists():
            try:
                self._last_loaded_pair = (str(Path(hbn_now).resolve()), str(Path(dark_now).resolve()))
            except Exception:
                self._last_loaded_pair = (hbn_now, dark_now)
        else:
            self._last_loaded_pair = None

        center = None
        if "detector_center" in d:
            c = np.asarray(d["detector_center"], dtype=float).reshape(-1)
            if c.size >= 2:
                center = (float(c[0]), float(c[1]))
        elif "center_fixed" in d:
            c = np.asarray(d["center_fixed"], dtype=float).reshape(-1)
            if c.size >= 2:
                center = (float(c[0]), float(c[1]))
        elif self.ellipses:
            center = ellipse_center(self.ellipses, fit_quality=self.fit_quality)

        if center is not None and np.isfinite(center[0]) and np.isfinite(center[1]):
            self.center_x.set(f"{center[0]:.3f}")
            self.center_y.set(f"{center[1]:.3f}")
        self._center_user_defined = center_src_in == "ui_entry"

        tx = npz_scalar(d, "tilt_x_deg_internal", npz_scalar(d, "tilt_x_deg", np.nan))
        ty = npz_scalar(d, "tilt_y_deg_internal", npz_scalar(d, "tilt_y_deg", np.nan))
        corrected = obj_to_ndarrays(d["ell_points_corrected"]) if "ell_points_corrected" in d else (
            obj_to_ndarrays(d["ell_points_corr"]) if "ell_points_corr" in d else []
        )
        self.corrected = corrected

        if np.isfinite(tx) and np.isfinite(ty):
            cxy, src = self._get_center(strict=False)
            ring_weights = np.asarray(d["ring_weights"], dtype=float) if "ring_weights" in d else np.array([], dtype=float)
            ring_ids = np.asarray(d["optimizer_ring_ids"], dtype=np.int32) if "optimizer_ring_ids" in d else np.array([], dtype=np.int32)
            center_initial = np.asarray(d["center_initial"], dtype=float).reshape(-1) if "center_initial" in d else np.array([], dtype=float)
            if center_initial.size >= 2:
                center_init_val = (float(center_initial[0]), float(center_initial[1]))
            else:
                center_init_val = (np.nan, np.nan)
            center_prior = np.asarray(d["center_prior"], dtype=float).reshape(-1) if "center_prior" in d else np.array([], dtype=float)
            if center_prior.size >= 2:
                center_prior_val = (float(center_prior[0]), float(center_prior[1]))
            else:
                center_prior_val = (np.nan, np.nan)
            self.optim = {
                "center": cxy,
                "center_initial": center_init_val,
                "center_prior": center_prior_val,
                "center_prior_sigma_px": npz_scalar(d, "center_prior_sigma_px", np.nan),
                "center_drift_limit_px": npz_scalar(d, "center_drift_limit_px", np.nan),
                "center_source": src,
                "tilt_x_deg": float(tx),
                "tilt_y_deg": float(ty),
                "cost_zero": npz_scalar(d, "cost_zero", np.nan),
                "cost_final": npz_scalar(d, "cost_final", np.nan),
                "circ_before": np.asarray(d["circ_before"], dtype=float) if "circ_before" in d else np.array([], dtype=float),
                "circ_after": np.asarray(d["circ_after"], dtype=float) if "circ_after" in d else np.array([], dtype=float),
                "radii_before": np.asarray(d["radii_before"], dtype=float) if "radii_before" in d else np.array([], dtype=float),
                "radii_after": np.asarray(d["radii_after"], dtype=float) if "radii_after" in d else np.array([], dtype=float),
                "ring_weights": ring_weights,
                "ring_snap_sigma_px": np.asarray(d["ring_snap_sigma_px"], dtype=float)
                if "ring_snap_sigma_px" in d
                else np.array([], dtype=float),
                "ring_ids": ring_ids,
                "corrected_points": corrected,
                "optimizer_kind": npz_string(d, "optimizer_kind", "legacy"),
                "distance_px": npz_scalar(d, "projective_distance_px", np.nan),
            }
            self.opt_text.set(f"tilt_x={tx:.4f} deg, tilt_y={ty:.4f} deg, cost={self.optim['cost_final']:.3e}")
        else:
            self.optim = None
            self.opt_text.set("tilt_x=--, tilt_y=--, cost=--")

        if "fit_confidence_per_ring" in d or "fit_confidence_overall" in d:
            self.fit_quality = {
                "overall_confidence": npz_scalar(d, "fit_confidence_overall", np.nan),
                "per_ring_confidence": np.asarray(d["fit_confidence_per_ring"], dtype=float)
                if "fit_confidence_per_ring" in d
                else np.array([], dtype=float),
                "ring_indices": np.asarray(d["fit_confidence_ring_indices"], dtype=np.int32)
                if "fit_confidence_ring_indices" in d
                else np.array([], dtype=np.int32),
                "residual_px": np.asarray(d["fit_residual_px"], dtype=float)
                if "fit_residual_px" in d
                else np.array([], dtype=float),
                "signal_snr": np.asarray(d["fit_signal_snr"], dtype=float)
                if "fit_signal_snr" in d
                else np.array([], dtype=float),
                "angular_coverage": np.asarray(d["fit_angular_coverage"], dtype=float)
                if "fit_angular_coverage" in d
                else np.array([], dtype=float),
                "n_points": np.asarray(d["fit_points_used"], dtype=np.int32)
                if "fit_points_used" in d
                else np.array([], dtype=np.int32),
                "downsample_factor": npz_scalar(d, "fit_downsample_factor", np.nan),
                "click_sigma_px": npz_scalar(d, "fit_click_sigma_px", np.nan),
                "downsample_score": npz_scalar(d, "fit_downsample_score", np.nan),
            }
        elif self.ellipses and self.points_ds:
            points_fit_ds = self._points_for_fit()
            self.fit_quality = compute_fit_confidence(self.img_log, points_fit_ds, self.ellipses, self.down)

        self._update_progress()
        self._update_fit_text()
        self._update_refine_more_btn_state()
        self.status.set(f"Loaded bundle: {path}")
        self.refresh_plot(reset_view=True)

    def refresh_plot(self, reset_view=False, view_scale=1.0):
        prev_xlim = None
        prev_ylim = None
        if not reset_view and self.ax is not None:
            try:
                prev_xlim = tuple(self.ax.get_xlim())
                prev_ylim = tuple(self.ax.get_ylim())
            except Exception:
                prev_xlim = None
                prev_ylim = None

        self.ax.clear()
        if self.img_disp is None:
            self.ax.set_axis_off()
            self.ax.text(0.5, 0.5, "Load OSC files or NPZ bundle to begin", transform=self.ax.transAxes, ha="center", va="center")
            self.canvas.draw_idle()
            return

        h, w = self.img_disp.shape
        f = 1.0
        # Display with the same vertical orientation as OSC Viewer while
        # keeping the underlying array unchanged.
        self.ax.imshow(self.img_disp, cmap="gray", origin="lower", interpolation="nearest", resample=False)

        for i, pts in enumerate(self.points_ds):
            snap_arr = np.asarray(pts, dtype=float)
            raw_arr = (
                np.asarray(self.points_raw_ds[i], dtype=float)
                if i < len(self.points_raw_ds)
                else np.empty((0, 2), dtype=float)
            )
            color = cm.tab10(i % 10)

            if raw_arr.size > 0:
                self.ax.plot(raw_arr[:, 0], raw_arr[:, 1], "x", color=color, markersize=4, alpha=0.45)

            if snap_arr.size > 0:
                self.ax.plot(
                    snap_arr[:, 0],
                    snap_arr[:, 1],
                    "o",
                    markerfacecolor="none",
                    markeredgecolor=color,
                    markersize=6,
                    markeredgewidth=1.2,
                )

            if raw_arr.shape == snap_arr.shape and raw_arr.size > 0:
                for p_raw, p_snap in zip(raw_arr, snap_arr):
                    self.ax.plot(
                        [p_raw[0], p_snap[0]],
                        [p_raw[1], p_snap[1]],
                        "-",
                        color=color,
                        linewidth=0.8,
                        alpha=0.35,
                    )

        for i, e in enumerate(self.ellipses):
            x, y = ellipse_curve(e["xc"], e["yc"], e["a"], e["b"], e["theta"], num=600)
            c = cm.tab10(i % 10)
            self.ax.plot(x / f, y / f, "-", color=c, linewidth=1.3)

        if self._show_suggested_regions and self.suggested_regions:
            for s in self.suggested_regions:
                rid = int(s.get("ring_index", 1))
                x_ds = float(s.get("x_ds", np.nan))
                y_ds = float(s.get("y_ds", np.nan))
                r_ds = float(s.get("r_ds", np.nan))
                if not (np.isfinite(x_ds) and np.isfinite(y_ds) and np.isfinite(r_ds) and r_ds > 0):
                    continue
                c = cm.tab10((rid - 1) % 10)
                accent = "#00E5FF"
                glow = Circle(
                    (x_ds, y_ds),
                    1.75 * r_ds,
                    fill=True,
                    linewidth=0.0,
                    facecolor=accent,
                    alpha=0.16,
                    zorder=4,
                )
                self.ax.add_patch(glow)
                halo = Circle(
                    (x_ds, y_ds),
                    1.28 * r_ds,
                    fill=False,
                    linestyle="-",
                    linewidth=3.0,
                    edgecolor="black",
                    alpha=0.75,
                    zorder=5,
                )
                self.ax.add_patch(halo)
                patch_outer = Circle(
                    (x_ds, y_ds),
                    1.15 * r_ds,
                    fill=False,
                    linestyle="-",
                    linewidth=2.8,
                    edgecolor="white",
                    alpha=0.95,
                    zorder=6,
                )
                self.ax.add_patch(patch_outer)
                patch = Circle(
                    (x_ds, y_ds),
                    r_ds,
                    fill=False,
                    linestyle="--",
                    linewidth=3.0,
                    edgecolor=accent,
                    alpha=1.0,
                    zorder=7,
                )
                self.ax.add_patch(patch)
                for ch_color, ch_lw, ch_alpha, ch_z in (
                    ("black", 2.8, 0.72, 8),
                    ("white", 1.6, 0.95, 9),
                ):
                    self.ax.plot(
                        [x_ds - 0.95 * r_ds, x_ds + 0.95 * r_ds],
                        [y_ds, y_ds],
                        "-",
                        color=ch_color,
                        linewidth=ch_lw,
                        alpha=ch_alpha,
                        zorder=ch_z,
                    )
                    self.ax.plot(
                        [x_ds, x_ds],
                        [y_ds - 0.95 * r_ds, y_ds + 0.95 * r_ds],
                        "-",
                        color=ch_color,
                        linewidth=ch_lw,
                        alpha=ch_alpha,
                        zorder=ch_z,
                    )
                self.ax.plot(
                    [x_ds],
                    [y_ds],
                    marker="*",
                    color=accent,
                    markersize=18,
                    markeredgewidth=1.6,
                    markeredgecolor="black",
                    linestyle="none",
                    zorder=10,
                )
                self.ax.plot(
                    [x_ds],
                    [y_ds],
                    marker="o",
                    color=c,
                    markersize=4,
                    markeredgewidth=0.0,
                    linestyle="none",
                    zorder=11,
                )
                self.ax.text(
                    x_ds + 1.3 * r_ds,
                    y_ds,
                    f"R{rid} target",
                    color="black",
                    fontsize=9,
                    fontweight="bold",
                    ha="left",
                    va="center",
                    alpha=1.0,
                    bbox=dict(boxstyle="round,pad=0.20", fc=accent, ec="black", lw=1.0, alpha=0.97),
                    zorder=12,
                )

        try:
            (cx_cur, cy_cur), _ = self._get_center(strict=False)
        except Exception:
            cx_cur, cy_cur = np.nan, np.nan
        if np.isfinite(cx_cur) and np.isfinite(cy_cur):
            self.ax.plot(
                cx_cur / f,
                cy_cur / f,
                "+",
                color="yellow",
                markersize=12,
                markeredgewidth=2.0,
            )

        t = np.linspace(0, 2 * np.pi, 720)
        if self.optim:
            cx, cy = self.optim.get("center", (np.nan, np.nan))
            if np.isfinite(cx) and np.isfinite(cy):
                self.ax.plot(cx / f, cy / f, "x", color="cyan", markersize=8)
                for r in np.asarray(self.optim.get("radii_before", []), dtype=float):
                    if np.isfinite(r) and r > 0:
                        self.ax.plot((cx + r * np.cos(t)) / f, (cy + r * np.sin(t)) / f, "--", color="orange", linewidth=0.8)
                for r in np.asarray(self.optim.get("radii_after", []), dtype=float):
                    if np.isfinite(r) and r > 0:
                        self.ax.plot((cx + r * np.cos(t)) / f, (cy + r * np.sin(t)) / f, "-", color="lime", linewidth=0.9)
            for pts in self.corrected:
                arr = np.asarray(pts, dtype=float)
                if arr.size:
                    self.ax.plot(arr[:, 0] / f, arr[:, 1] / f, ".", color="lime", markersize=1, alpha=0.2)

        title = f"hBN rings | downsample x{self.down} | fitted={len(self.ellipses)}"
        if self.fit_quality:
            fit_overall = float(self.fit_quality.get("overall_confidence", np.nan))
            ds_score = float(self.fit_quality.get("downsample_score", np.nan))
            if np.isfinite(fit_overall):
                title += f" | fit_conf={fit_overall:.1f}%"
            if np.isfinite(ds_score):
                title += f" | ds_score={100.0*ds_score:.1f}%"
        if self.optim:
            tx = self.optim.get("tilt_x_deg", np.nan)
            ty = self.optim.get("tilt_y_deg", np.nan)
            if np.isfinite(tx) and np.isfinite(ty):
                title += f" | tilt_x={tx:.3f} deg, tilt_y={ty:.3f} deg"
        self.ax.set_title(title)

        use_prev = False
        if not reset_view and prev_xlim is not None and prev_ylim is not None:
            scale = float(view_scale) if np.isfinite(view_scale) and view_scale > 0 else 1.0
            x0, x1 = float(prev_xlim[0]) * scale, float(prev_xlim[1]) * scale
            y0, y1 = float(prev_ylim[0]) * scale, float(prev_ylim[1]) * scale
            x_min = max(0.0, min(x0, x1))
            x_max = min(float(max(w - 1, 1)), max(x0, x1))
            y_min = max(0.0, min(y0, y1))
            y_max = min(float(max(h - 1, 1)), max(y0, y1))
            if (x_max - x_min) > 1e-6 and (y_max - y_min) > 1e-6:
                if x0 <= x1:
                    self.ax.set_xlim(x_min, x_max)
                else:
                    self.ax.set_xlim(x_max, x_min)
                if y0 <= y1:
                    self.ax.set_ylim(y_min, y_max)
                else:
                    self.ax.set_ylim(y_max, y_min)
                use_prev = True

        if not use_prev:
            self.ax.set_xlim(0, max(w - 1, 1))
            self.ax.set_ylim(0, max(h - 1, 1))

        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("x (full-res px)")
        self.ax.set_ylabel("y (full-res px)")
        self.canvas.draw_idle()


def parse_args():
    parser = argparse.ArgumentParser(description="hBN ring fitter + tilt optimizer GUI")
    parser.add_argument("--bundle", type=str, default=None, help="Optional NPZ bundle to load at startup")
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    app = HBNFitterGUI(root, startup_bundle=args.bundle)
    _ = app
    root.mainloop()


if __name__ == "__main__":
    main()
