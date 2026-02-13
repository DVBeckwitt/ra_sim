#!/usr/bin/env python3
"""
Find detector tilts about x and y (around a fixed center) that make rings
as circular as possible, overlay original and corrected rings, and show the
rotation axes.

Fixed common center:
  xc = 1547, yc = 1598   (pixels, full-resolution detector coordinates)

Model:
  tilt_x_deg: detector tilt about x axis (horizontal) through the center
              causes foreshortening in y; we "untilt" by scaling y by 1/cos(tilt_x)
  tilt_y_deg: detector tilt about y axis (vertical) through the center
              causes foreshortening in x; we "untilt" by scaling x by 1/cos(tilt_y)

Input:
  C:/Users/Kenpo/Downloads/hbn_ellipse_bundle.npz
    - ell_points_ds: object array of [N_i x 2] arrays of points per ring

Output:
  - hbn_ellipses_circularized.png  (overlay: original + corrected + axes)
  - hbn_ellipse_bundle_circularized.npz
  - printed circularity metrics before and after, plus cost at zero tilt
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize


# ---------------------------------------------------------
# Geometry: fixed common center and tilt transform
# ---------------------------------------------------------

FIXED_CENTER = ( 1598.0,1547.0)  # (xc, yc) in pixels


def apply_tilt_xy(pts, center, tilt_x_deg, tilt_y_deg):
    """
    Apply inverse tilt correction assuming rotations about x and y axes
    through the center.

    tilt_x_deg: tilt about x axis (horizontal), affects y scaling
    tilt_y_deg: tilt about y axis (vertical), affects x scaling
    """
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.size == 0:
        return pts

    xc, yc = center

    # Clamp tilts to a realistic range
    max_tilt_deg = 45.0
    tilt_x_deg = float(np.clip(tilt_x_deg, -max_tilt_deg, max_tilt_deg))
    tilt_y_deg = float(np.clip(tilt_y_deg, -max_tilt_deg, max_tilt_deg))

    tx = np.deg2rad(tilt_x_deg)
    ty = np.deg2rad(tilt_y_deg)

    cx = np.cos(tx)  # affects y
    cy = np.cos(ty)  # affects x

    # Avoid division by near zero
    if abs(cx) < 1e-3:
        scale_y = 1e3
    else:
        scale_y = 1.0 / cx

    if abs(cy) < 1e-3:
        scale_x = 1e3
    else:
        scale_x = 1.0 / cy

    # Translate to center, scale, translate back
    dx = (pts[:, 0] - xc) * scale_x
    dy = (pts[:, 1] - yc) * scale_y
    x_corr = xc + dx
    y_corr = yc + dy

    return np.column_stack([x_corr, y_corr])


# ---------------------------------------------------------
# Objective and metrics
# ---------------------------------------------------------

def circularity_cost(params_deg, ell_points_ds, center):
    """
    Cost = sum over rings of (std(r) / mean(r))^2 after tilt_xy correction.

    params_deg = [tilt_x_deg, tilt_y_deg]
    """
    tilt_x_deg, tilt_y_deg = params_deg
    xc, yc = center
    total_cost = 0.0

    for pts in ell_points_ds:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.size == 0:
            continue

        pts_corr = apply_tilt_xy(pts, center, tilt_x_deg, tilt_y_deg)

        dx = pts_corr[:, 0] - xc
        dy = pts_corr[:, 1] - yc
        r = np.sqrt(dx * dx + dy * dy)

        r_mean = r.mean()
        if r_mean < 1e-9:
            continue

        metric = np.std(r) / r_mean
        total_cost += metric * metric

    return total_cost


def circularity_metrics(ell_points_ds, center, tilt_x_deg, tilt_y_deg):
    """Per-ring std(r)/mean(r) after applying tilt_xy with given tilts."""
    xc, yc = center
    metrics = []

    for pts in ell_points_ds:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.size == 0:
            metrics.append(np.nan)
            continue

        pts_corr = apply_tilt_xy(pts, center, tilt_x_deg, tilt_y_deg)

        dx = pts_corr[:, 0] - xc
        dy = pts_corr[:, 1] - yc
        r = np.sqrt(dx * dx + dy * dy)
        r_mean = r.mean()
        if r_mean < 1e-9:
            metrics.append(np.nan)
            continue

        metrics.append(float(np.std(r) / r_mean))

    return metrics


def apply_tilt_correction_all(ell_points_ds, center, tilt_x_deg, tilt_y_deg):
    """Apply tilt_xy correction to every ring."""
    corrected = []
    for pts in ell_points_ds:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.size == 0:
            continue
        corrected.append(apply_tilt_xy(pts, center, tilt_x_deg, tilt_y_deg))
    return corrected


def fit_ring_radii(ell_points, center):
    """For each ring, compute mean radius from common center."""
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


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    bundle_path = Path(r"C:/Users/Kenpo/Downloads/hbn_ellipse_bundle.npz")
    bundle = np.load(bundle_path, allow_pickle=True)

    ell_points_ds = bundle["ell_points_ds"]
    center = FIXED_CENTER
    xc, yc = center
    print(f"Using FIXED center: xc={xc:.2f}, yc={yc:.2f}")

    # Cost at zero tilt (for reference)
    cost_zero = circularity_cost((0.0, 0.0), ell_points_ds, center)
    print(f"Cost at zero tilt (tilt_x=0, tilt_y=0): {cost_zero:.6e}")

    # ----- coarse grid search to avoid local minima -----
    print("Performing coarse grid search for tilts...")
    tx_grid = np.linspace(-10.0, 10.0, 41)  # step 0.5 deg
    ty_grid = np.linspace(-10.0, 10.0, 41)
    best_cost = np.inf
    best_tx = 0.0
    best_ty = 0.0

    for tx in tx_grid:
        for ty in ty_grid:
            c = circularity_cost((tx, ty), ell_points_ds, center)
            if c < best_cost:
                best_cost = c
                best_tx = tx
                best_ty = ty

    print(f"Grid best: tilt_x={best_tx:.3f} deg, tilt_y={best_ty:.3f} deg, cost={best_cost:.6e}")

    # ----- refine with Nelder–Mead starting from grid best -----
    x0 = np.array([best_tx, best_ty], dtype=float)

    result = minimize(
        circularity_cost,
        x0,
        args=(ell_points_ds, center),
        method="Nelder-Mead",
        options=dict(maxiter=1000, xatol=1e-4, fatol=1e-8, disp=True),
    )

    tilt_x_opt, tilt_y_opt = result.x
    print("\nOptimization finished:")
    print(f"  tilt_x_deg (about x axis) = {tilt_x_opt:.4f}")
    print(f"  tilt_y_deg (about y axis) = {tilt_y_opt:.4f}")
    print(f"  final cost                = {result.fun:.6e}")

    # Circularity before (no tilt) and after (optimized tilt_xy)
    circ_before = circularity_metrics(ell_points_ds, center, 0.0, 0.0)
    circ_after = circularity_metrics(ell_points_ds, center, tilt_x_opt, tilt_y_opt)

    print("\nRing circularity (std(r)/mean(r)):")
    print("Ring   before[%]   after[%]")
    for i, (cb, ca) in enumerate(zip(circ_before, circ_after), start=1):
        cb_pct = 100.0 * cb if np.isfinite(cb) else np.nan
        ca_pct = 100.0 * ca if np.isfinite(ca) else np.nan
        print(f"{i:4d}   {cb_pct:8.3f}   {ca_pct:8.3f}")

    # Apply correction to points
    ell_points_corr = apply_tilt_correction_all(
        ell_points_ds, center, tilt_x_opt, tilt_y_opt
    )

    # Fit circle radius for each ring before and after
    radii_before = fit_ring_radii(ell_points_ds, center)
    radii_after = fit_ring_radii(ell_points_corr, center)

    # Build overlay plot: original + corrected + axes
    out_dir = bundle_path.parent
    fig, ax = plt.subplots(figsize=(6, 6))

    # Collect bounds from both original and corrected points
    all_pts = []
    for pts in ell_points_ds:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 2 and pts.shape[1] == 2 and pts.size > 0:
            all_pts.append(pts)
    for pts in ell_points_corr:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 2 and pts.shape[1] == 2 and pts.size > 0:
            all_pts.append(pts)
    all_xy = np.vstack(all_pts)
    xmin, xmax = np.min(all_xy[:, 0]), np.max(all_xy[:, 0])
    ymin, ymax = np.min(all_xy[:, 1]), np.max(all_xy[:, 1])

    theta = np.linspace(0.0, 2.0 * np.pi, 720)

    # Original ring points
    for pts in ell_points_ds:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 2 and pts.shape[1] == 2:
            ax.plot(pts[:, 0], pts[:, 1], ".", markersize=1, alpha=0.3, label="_orig_pts")

    # Corrected ring points
    for pts in ell_points_corr:
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 2 and pts.shape[1] == 2:
            ax.plot(pts[:, 0], pts[:, 1], ".", markersize=1, alpha=0.6, label="_corr_pts")

    # Original fitted circles (dashed)
    for r in radii_before:
        if not np.isfinite(r) or r <= 0:
            continue
        x_circ = xc + r * np.cos(theta)
        y_circ = yc + r * np.sin(theta)
        ax.plot(x_circ, y_circ, linestyle="--", linewidth=1, label="_orig_circle")

    # Corrected fitted circles (solid)
    for r in radii_after:
        if not np.isfinite(r) or r <= 0:
            continue
        x_circ = xc + r * np.cos(theta)
        y_circ = yc + r * np.sin(theta)
        ax.plot(x_circ, y_circ, linewidth=1, label="_corr_circle")

    # Rotation axes about which tilts are defined
    ax.axhline(y=yc, linestyle="-.", linewidth=1.5, label="x tilt axis")
    ax.axvline(x=xc, linestyle="-.", linewidth=1.5, label="y tilt axis")

    # Center marker
    ax.plot(xc, yc, "x", markersize=6, label="center")

    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)  # inverted y

    ax.set_title(
        "Original and tilt-corrected rings\n"
        f"center=({xc:.0f},{yc:.0f}), "
        f"tilt_x={tilt_x_opt:.2f} deg, tilt_y={tilt_y_opt:.2f} deg"
    )
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    ax.legend(loc="best")

    fig.tight_layout()
    out_png = out_dir / "hbn_ellipses_circularized.png"
    fig.savefig(out_png, dpi=300)
    print(f"\nSaved overlay (original + corrected + axes) to:")
    print(f"  {out_png}")

    plt.show()

    # Save corrected data and metrics
    out_npz = out_dir / "hbn_ellipse_bundle_circularized.npz"
    np.savez(
        out_npz,
        ell_points_corr=np.array(ell_points_corr, dtype=object),
        center_fixed=np.array(center, dtype=float),
        radii_before=np.array(radii_before, dtype=float),
        radii_after=np.array(radii_after, dtype=float),
        circ_before=np.array(circ_before, dtype=float),
        circ_after=np.array(circ_after, dtype=float),
        tilt_x_deg=float(tilt_x_opt),
        tilt_y_deg=float(tilt_y_opt),
        cost_zero=float(cost_zero),
        cost_final=float(result.fun),
    )
    print(f"Saved corrected bundle to:")
    print(f"  {out_npz}")


if __name__ == "__main__":
    main()
