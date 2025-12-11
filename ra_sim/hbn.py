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

import os
import json
import argparse
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
import yaml

from matplotlib.patches import Rectangle
from skimage.measure import EllipseModel

import OSC_Reader
from OSC_Reader import OSC_Reader


# ------------------------------------------------------------
# Configuration parameters
# ------------------------------------------------------------

# Number of rings / ellipses
N_ELLIPSES = 5

# Points clicked per ellipse (>=5 recommended)
POINTS_PER_ELLIPSE = 5

# Downsample factor for interactive view only
DOWNSAMPLE_FACTOR = 4

# Log intensity for display and fitting
USE_LOG_INTENSITY = True
LOG_EPS = 1e-3

# Try to reuse existing click profile JSON instead of clicking again
LOAD_PROFILE = False

# Intensity based refinement settings
REFINE_N_ANGLES = 360    # angular sampling along ellipse
REFINE_DR = 10.0         # half width of radial search window [pixels]
REFINE_STEP = 1.0        # radial step [pixels]


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
        "--downsample-factor",
        type=int,
        default=None,
        help="Override the downsample factor used for interactive clicking.",
    )
    parser.add_argument(
        "--paths-file",
        type=str,
        default=None,
        help=(
            "Optional YAML/JSON file containing calibrant, dark, and artifact paths. "
            "Accepted keys: calibrant/osc, dark/dark_file, bundle/npz, click_profile/profile, "
            "fit_profile/fit, and fit_compression."
        ),
    )
    parser.add_argument(
        "--fit-compression",
        type=int,
        default=None,
        help=(
            "Downsample factor applied before fitting/refinement (1 keeps full resolution, "
            "4 downsamples to 1/4 size, etc.)."
        ),
    )
    parser.add_argument(
        "--load-bundle",
        type=str,
        default=None,
        help="Path to NPZ bundle created by this script. If given, load it instead of recomputing everything.",
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
    return parser.parse_args(argv)

# ------------------------------------------------------------
# Background subtraction
# ------------------------------------------------------------
def load_and_bgsub(file_path_hbn, dark_file):
    print("Loading dark image...")
    dark = OSC_Reader.read_osc(dark_file).astype(np.float32)

    print("Loading hBN calibrant image...")
    raw = OSC_Reader.read_osc(file_path_hbn).astype(np.float32)

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


def make_click_image(img_bgsub, down_factor):
    img_log = make_log_image(img_bgsub)

    vmin, vmax = np.percentile(img_log, (0.5, 99.5))
    norm = (img_log - vmin) / (vmax - vmin + 1e-9)
    norm = np.clip(norm, 0, 1)

    h, w = norm.shape
    if down_factor < 1:
        raise ValueError("DOWNSAMPLE_FACTOR must be >= 1")
    small_w = max(1, w // down_factor)
    small_h = max(1, h // down_factor)
    print(f"Downsampling from {w}x{h} to {small_w}x{small_h} (factor {down_factor}) for clicking...")
    small = cv2.resize(norm, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return small


def make_display_image(img_bgsub):
    img_log = make_log_image(img_bgsub)
    p_low, p_high = np.percentile(img_log, (0.1, 99.9))
    disp = (img_log - p_low) / (p_high - p_low + 1e-9)
    disp = np.clip(disp, 0, 1)
    return disp


def compress_image(img, factor):
    if factor is None:
        raise ValueError("Compression factor must not be None")
    factor = int(factor)
    if factor < 1:
        raise ValueError("fit_compression must be >= 1")
    if factor == 1:
        return img

    h, w = img.shape
    small_w = max(1, w // factor)
    small_h = max(1, h // factor)
    print(f"Compressing image from {w}x{h} to {small_w}x{small_h} (factor {factor}) for fitting...")
    return cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)


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
    fit_compression=None,
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
        fit_compression=fit_compression,
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
        if resolved["fit_compression"] is None and file_data:
            fc = file_data.get("fit_compression")
            if fc is not None:
                resolved["fit_compression"] = int(fc)

    if resolved["fit_compression"] is None:
        resolved["fit_compression"] = 1

    return resolved


# ------------------------------------------------------------
# Profile save / load
# ------------------------------------------------------------
def save_click_profile(path, ell_points_ds, img_shape, down_factor):
    profile = {
        "downsample_factor": down_factor,
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


def save_fit_profile(path, ellipses, img_shape, down_factor):
    profile = {
        "downsample_factor": down_factor,
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
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"Saved fit profile to:\n  {path}")


def save_bundle(path, img_bgsub, img_log, ell_points_ds, ellipses, down_factor, fit_compression):
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
        img_bgsub=img_bgsub.astype(np.float32),
        img_log=img_log.astype(np.float32),
        ell_points_ds=ell_points_arr,
        ellipse_params=ellipse_params,
        downsample_factor=np.array(down_factor, dtype=np.int32),
        fit_compression=np.array(fit_compression, dtype=np.int32),
    )
    print(f"Saved bundle NPZ to:\n  {path}")
    print("Note: load with allow_pickle=True for ell_points_ds.")


def load_bundle_npz(path):
    data = np.load(path, allow_pickle=True)
    img_bgsub = data["img_bgsub"]
    img_log = data["img_log"]
    ell_points_arr = data["ell_points_ds"]
    ellipse_params = data["ellipse_params"]
    down_factor = int(data["downsample_factor"])
    fit_compression = int(data["fit_compression"]) if "fit_compression" in data else 1

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

    print(f"Loaded bundle from:\n  {path}")
    print(f"  image shape: {img_bgsub.shape}")
    print(f"  number of ellipses: {len(ellipses)}")
    print(f"  downsample_factor in bundle: {down_factor}")
    print(f"  fit_compression in bundle: {fit_compression}")
    return img_bgsub, img_log, ell_points_ds, ellipses, down_factor, fit_compression


# ------------------------------------------------------------
# Interactive clicking with zoom
# ------------------------------------------------------------
def get_points_per_ellipse_with_zoom(small, n_ellipses, pts_per_ellipse):
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
                f"x={x:.1f}, y={y:.1f} (downsampled)"
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


def fit_ellipses_from_points(ell_points_ds, down_factor, img_bgsub, fit_compression=1):
    img_fit = compress_image(img_bgsub, fit_compression)
    img_log = make_log_image(img_fit)
    ellipses = []

    scale_to_fit = float(down_factor) / float(fit_compression)
    scale_back = float(fit_compression)

    for i, pts in enumerate(ell_points_ds, 1):
        pts = np.array(pts, dtype=float)
        if pts.shape[0] < 5:
            print(f"Ellipse {i}: not enough points to fit (have {pts.shape[0]}). Skipping.")
            continue

        pts_full = pts * scale_to_fit

        params0, _ = fit_initial_ellipse_from_points(pts_full)
        print(
            f"Ellipse {i} initial fit: "
            f"xc={params0[0]:.2f}, yc={params0[1]:.2f}, "
            f"a={params0[2]:.2f}, b={params0[3]:.2f}, "
            f"theta={np.degrees(params0[4]):.2f} deg"
        )

        params_refined = refine_ellipse_with_intensity(img_log, params0)
        xc, yc, a, b, theta = params_refined

        xc *= scale_back
        yc *= scale_back
        a *= scale_back
        b *= scale_back

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


def refine_ellipses_from_existing(
    ellipses_prev, img_bgsub_prev, img_bgsub_hr, img_log_hr, fit_compression=1
):
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
        params0 = (
            xc0 / float(fit_compression),
            yc0 / float(fit_compression),
            a0 / float(fit_compression),
            b0 / float(fit_compression),
            theta0,
        )

        print(
            f"High res refine ellipse {i}: "
            f"start xc={xc0:.2f}, yc={yc0:.2f}, a={a0:.2f}, b={b0:.2f}, "
            f"theta={np.degrees(theta0):.2f} deg"
        )

        xc, yc, a, b, theta = refine_ellipse_with_intensity(img_log_hr, params0)

        xc *= float(fit_compression)
        yc *= float(fit_compression)
        a *= float(fit_compression)
        b *= float(fit_compression)

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


def plot_ellipses(img_bgsub, ellipses, save_path=None, down_factor=DOWNSAMPLE_FACTOR):
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

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_title(
        f"{len(ellipses)} ellipses "
        f"(clicked points, intensity refined, downsample factor {down_factor})"
    )

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        print(f"Saved overlay image with ellipses to:\n  {save_path}")
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


def run_hbn_fit(
    osc_path,
    dark_path,
    output_dir=None,
    load_bundle=None,
    highres_refine=False,
    reuse_profile=False,
    downsample_factor=None,
    click_profile_path=None,
    fit_profile_path=None,
    bundle_path=None,
    fit_compression=None,
    paths_file=None,
):
    resolved = resolve_hbn_paths(
        osc_path=osc_path,
        dark_path=dark_path,
        bundle_path=bundle_path,
        click_profile_path=click_profile_path,
        fit_profile_path=fit_profile_path,
        fit_compression=fit_compression,
        paths_file=paths_file,
    )

    osc_path = resolved["osc"]
    dark_path = resolved["dark"]
    bundle_from_file = resolved["bundle"]
    bundle_path_in = load_bundle or (
        bundle_from_file if bundle_from_file and os.path.exists(bundle_from_file) else None
    )
    fit_compression = int(resolved["fit_compression"])

    output_dir = _resolve_output_dir(output_dir, bundle_path_in)
    out_tiff_path = os.path.join(output_dir, "hbn_bgsub.tiff")
    out_overlay_path = os.path.join(output_dir, "hbn_bgsub_ellipses.png")

    click_profile_path = resolved["click_profile"] or click_profile_path
    if click_profile_path is None:
        click_profile_path = os.path.join(output_dir, "hbn_ellipse_profile.json")
    fit_profile_path = resolved["fit_profile"] or fit_profile_path
    if fit_profile_path is None:
        fit_profile_path = os.path.join(output_dir, "hbn_ellipse_fit_profile.json")
    bundle_path_save = resolved["bundle"] or bundle_path_in
    if bundle_path_save is None:
        bundle_path_save = os.path.join(output_dir, "hbn_ellipse_bundle.npz")

    for p in [click_profile_path, fit_profile_path, bundle_path_save]:
        parent = os.path.dirname(os.path.abspath(p))
        os.makedirs(parent, exist_ok=True)

    outputs = {
        "output_dir": output_dir,
        "background_subtracted": out_tiff_path,
        "overlay": out_overlay_path,
        "click_profile": click_profile_path,
        "fit_profile": fit_profile_path,
        "bundle": bundle_path_save,
        "fit_compression": fit_compression,
        "ellipses": [],
    }

    # Bundle path mode
    if bundle_path_in is not None:
        (
            img_bgsub_b,
            img_log_b,
            ell_points_ds,
            ellipses_b,
            down_factor_b,
            fit_compression_b,
        ) = load_bundle_npz(bundle_path_in)

        down_factor = int(downsample_factor or down_factor_b or DOWNSAMPLE_FACTOR)
        fit_compression = int(fit_compression or fit_compression_b or 1)

        should_refit = highres_refine
        refit_reason = "--highres-refine requested" if highres_refine else ""
        if not should_refit and fit_compression != fit_compression_b:
            should_refit = True
            refit_reason = (
                "fit_compression changed from bundle "
                f"({fit_compression_b} -> {fit_compression})"
            )

        if should_refit:
            if osc_path is None or dark_path is None:
                raise ValueError(
                    "Refitting a bundle requires both --osc and --dark so the background "
                    "image can be recomputed."
                )

            print(f"Refitting bundle ellipses: {refit_reason}...")
            img_bgsub_hr = load_and_bgsub(osc_path, dark_path)
            img_fit_hr = compress_image(img_bgsub_hr, fit_compression)
            img_log_hr = make_log_image(img_fit_hr)
            ellipses_out = refine_ellipses_from_existing(
                ellipses_b, img_bgsub_b, img_bgsub_hr, img_log_hr, fit_compression=fit_compression
            )
            img_bgsub_out = img_bgsub_hr
            img_log_out = make_log_image(img_bgsub_hr)
        else:
            img_bgsub_out = img_bgsub_b
            img_log_out = img_log_b
            ellipses_out = ellipses_b

        cv2.imwrite(out_tiff_path, img_bgsub_out)
        save_click_profile(click_profile_path, ell_points_ds, img_bgsub_out.shape, down_factor)
        save_fit_profile(fit_profile_path, ellipses_out, img_bgsub_out.shape, down_factor)
        save_bundle(
            bundle_path_save,
            img_bgsub_out,
            img_log_out,
            ell_points_ds,
            ellipses_out,
            down_factor,
            fit_compression,
        )
        plot_ellipses(img_bgsub_out, ellipses_out, save_path=out_overlay_path, down_factor=down_factor)

        outputs["ellipses"] = ellipses_out
        return outputs

    # Normal pipeline using OSC files
    if osc_path is None or dark_path is None:
        raise ValueError(
            "Both --osc and --dark are required unless --load-bundle is used "
            "without --highres-refine."
        )

    down_factor = int(downsample_factor or DOWNSAMPLE_FACTOR)
    img_bgsub = load_and_bgsub(osc_path, dark_path)

    print(f"Saving background-subtracted image to:\n  {out_tiff_path}")
    cv2.imwrite(out_tiff_path, img_bgsub)
    print("Saved hbn_bgsub.tiff.")

    if reuse_profile and os.path.exists(click_profile_path):
        print(f"Loading existing click profile from:\n  {click_profile_path}")
        ell_points_ds = load_click_profile(click_profile_path)
    else:
        small = make_click_image(img_bgsub, down_factor)
        ell_points_ds = get_points_per_ellipse_with_zoom(
            small,
            n_ellipses=N_ELLIPSES,
            pts_per_ellipse=POINTS_PER_ELLIPSE,
        )
        save_click_profile(click_profile_path, ell_points_ds, img_bgsub.shape, down_factor)

    ellipses = fit_ellipses_from_points(
        ell_points_ds,
        down_factor=down_factor,
        img_bgsub=img_bgsub,
        fit_compression=fit_compression,
    )
    print(f"Fitted {len(ellipses)} ellipses from clicked points and intensity refinement.")

    save_fit_profile(fit_profile_path, ellipses, img_bgsub.shape, down_factor)

    img_log = make_log_image(img_bgsub)
    save_bundle(
        bundle_path_save,
        img_bgsub,
        img_log,
        ell_points_ds,
        ellipses,
        down_factor,
        fit_compression,
    )

    if ellipses:
        plot_ellipses(img_bgsub, ellipses, save_path=out_overlay_path, down_factor=down_factor)

    outputs["ellipses"] = ellipses
    return outputs


def main(argv=None):
    args = parse_args(argv)

    results = run_hbn_fit(
        osc_path=args.osc,
        dark_path=args.dark,
        output_dir=args.output_dir,
        load_bundle=args.load_bundle,
        highres_refine=args.highres_refine,
        reuse_profile=args.reuse_profile,
        downsample_factor=args.downsample_factor,
        fit_compression=args.fit_compression,
        paths_file=args.paths_file,
    )

    print("Completed hBN ellipse fitting. Outputs written to:")
    for key in [
        "background_subtracted",
        "overlay",
        "click_profile",
        "fit_profile",
        "bundle",
    ]:
        print(f"  {key.replace('_', ' ').title()}: {results[key]}")
    print(f"  Fit compression: {results['fit_compression']}")


if __name__ == "__main__":
    main()
