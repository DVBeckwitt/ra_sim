"""Helpers to inspect Bragg-sphere and Ewald-sphere intersections."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, cos, pi, sin, sqrt

import matplotlib.pyplot as plt
import numpy as np

from .diffraction import (
    compute_intensity_array,
    intersect_infinite_line_plane,
    intersect_line_plane,
    ktz_components,
    solve_q,
    transmit_angle_grazing,
)


@dataclass(frozen=True)
class IntersectionGeometry:
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
    n_detector: np.ndarray
    unit_x: np.ndarray
    pixel_size_m: float = 100e-6


@dataclass(frozen=True)
class BeamSamples:
    beam_x_array: np.ndarray
    beam_y_array: np.ndarray
    theta_array: np.ndarray
    phi_array: np.ndarray
    wavelength_array: np.ndarray


@dataclass(frozen=True)
class MosaicParams:
    sigma_mosaic_deg: float
    gamma_mosaic_deg: float
    eta: float


@dataclass(frozen=True)
class ReflectionIntersectionAnalysis:
    hkl: tuple[int, int, int]
    g_vec: np.ndarray
    sphere_x: np.ndarray
    sphere_y: np.ndarray
    sphere_z: np.ndarray
    sphere_intensity: np.ndarray
    arc_q: np.ndarray
    arc_intensity: np.ndarray
    arc_angle_deg: np.ndarray
    arc_detector_col: np.ndarray
    arc_detector_row: np.ndarray
    arc_valid_mask: np.ndarray
    nearest_index: int
    selected_native_col: float
    selected_native_row: float


@dataclass(frozen=True)
class _BeamContext:
    i_plane: np.ndarray
    r_sample: np.ndarray
    detector_pos: np.ndarray
    n_det_rot: np.ndarray
    e1_det: np.ndarray
    e2_det: np.ndarray
    k_scat: float
    k_x_scat: float
    k_y_scat: float
    re_k_z: float
    k_in_crystal: np.ndarray
    all_q: np.ndarray


def compute_g_vec(h: int, k: int, l: int, a: float, c: float) -> np.ndarray:
    """Return the reciprocal-space G vector used by the diffraction kernels."""

    gz0 = 2.0 * pi * (float(l) / float(c))
    gr0 = 4.0 * pi / float(a) * sqrt((h * h + h * k + k * k) / 3.0)
    return np.array([0.0, gr0, gz0], dtype=np.float64)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-14:
        return np.asarray(v, dtype=np.float64)
    return np.asarray(v, dtype=np.float64) / n


def _build_detector_frame(geometry: IntersectionGeometry):
    gamma_rad = np.deg2rad(float(geometry.gamma_deg))
    Gamma_rad = np.deg2rad(float(geometry.Gamma_deg))

    cg = np.cos(gamma_rad)
    sg = np.sin(gamma_rad)
    cG = np.cos(Gamma_rad)
    sG = np.sin(Gamma_rad)

    r_x_det = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cg, sg],
            [0.0, -sg, cg],
        ],
        dtype=np.float64,
    )
    r_z_det = np.array(
        [
            [cG, sG, 0.0],
            [-sG, cG, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    n_det_rot = _unit(r_z_det @ (r_x_det @ np.asarray(geometry.n_detector, dtype=np.float64)))
    detector_pos = np.array([0.0, float(geometry.distance_cor_to_detector), 0.0], dtype=np.float64)

    unit_x = np.asarray(geometry.unit_x, dtype=np.float64)
    e1_det = unit_x - np.dot(unit_x, n_det_rot) * n_det_rot
    if float(np.linalg.norm(e1_det)) < 1e-14:
        e1_det = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        e1_det = _unit(e1_det)

    e2_det = np.array(
        [
            -(n_det_rot[1] * e1_det[2] - n_det_rot[2] * e1_det[1]),
            -(n_det_rot[2] * e1_det[0] - n_det_rot[0] * e1_det[2]),
            -(n_det_rot[0] * e1_det[1] - n_det_rot[1] * e1_det[0]),
        ],
        dtype=np.float64,
    )
    if float(np.linalg.norm(e2_det)) < 1e-14:
        e2_det = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        e2_det = _unit(e2_det)

    return detector_pos, n_det_rot, e1_det, e2_det


def _build_sample_frame(geometry: IntersectionGeometry):
    chi_rad = np.deg2rad(float(geometry.chi_deg))
    psi_rad = np.deg2rad(float(geometry.psi_deg))
    psi_z_rad = np.deg2rad(float(geometry.psi_z_deg))
    theta_initial_rad = np.deg2rad(float(geometry.theta_initial_deg))
    _ = geometry.cor_angle_deg

    c_chi = cos(chi_rad)
    s_chi = sin(chi_rad)
    r_y = np.array(
        [
            [c_chi, 0.0, s_chi],
            [0.0, 1.0, 0.0],
            [-s_chi, 0.0, c_chi],
        ],
        dtype=np.float64,
    )

    c_psi = cos(psi_rad)
    s_psi = sin(psi_rad)
    r_z = np.array(
        [
            [c_psi, s_psi, 0.0],
            [-s_psi, c_psi, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    c_psi_z = cos(psi_z_rad)
    s_psi_z = sin(psi_z_rad)
    r_z_gonio = np.array(
        [
            [c_psi_z, s_psi_z, 0.0],
            [-s_psi_z, c_psi_z, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    r_z_r_y = (r_z_gonio @ r_z) @ r_y

    ct = cos(theta_initial_rad)
    st = sin(theta_initial_rad)
    r_cor = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, ct, -st],
            [0.0, st,  ct],
        ],
        dtype=np.float64,
    )
    r_sample = r_cor @ r_z_r_y

    n1 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n_surf = _unit(r_cor @ (r_z_r_y @ n1))

    p0_rot = np.array([0.0, 0.0, -float(geometry.zs)], dtype=np.float64)

    return r_sample, n_surf, p0_rot


def _beam_order(beam: BeamSamples) -> np.ndarray:
    metric = np.square(np.asarray(beam.theta_array, dtype=np.float64)) + np.square(
        np.asarray(beam.phi_array, dtype=np.float64)
    )
    return np.argsort(metric)


def _solve_for_best_beam_sample(
    h: int,
    k: int,
    l: int,
    g_vec: np.ndarray,
    beam: BeamSamples,
    geometry: IntersectionGeometry,
    mosaic: MosaicParams,
    n2: complex,
    selected_native_col: float | None = None,
    selected_native_row: float | None = None,
) -> _BeamContext:
    detector_pos, n_det_rot, e1_det, e2_det = _build_detector_frame(geometry)
    r_sample, n_surf, p0_rot = _build_sample_frame(geometry)

    sigma_rad = float(np.deg2rad(mosaic.sigma_mosaic_deg))
    gamma_rad = float(np.deg2rad(mosaic.gamma_mosaic_deg))
    eta = float(mosaic.eta)

    best_ctx = None
    best_dist2 = np.inf
    fallback_ctx = None

    for idx in _beam_order(beam):
        bx = float(beam.beam_x_array[idx])
        by = float(beam.beam_y_array[idx])
        dtheta = float(beam.theta_array[idx])
        dphi = float(beam.phi_array[idx])
        wavelength = float(beam.wavelength_array[idx])
        if wavelength <= 0.0:
            continue

        beam_start = np.array([bx, -20e-3, -float(geometry.zb) + by], dtype=np.float64)
        k_in = np.array(
            [
                cos(dtheta) * sin(dphi),
                cos(dtheta) * cos(dphi),
                sin(dtheta),
            ],
            dtype=np.float64,
        )

        ix, iy, iz, valid_int = intersect_line_plane(beam_start, k_in, p0_rot, n_surf)
        if not valid_int:
            continue

        i_plane = np.array([ix, iy, iz], dtype=np.float64)
        kn_dot = float(np.dot(k_in, n_surf))
        kn_dot = float(np.clip(kn_dot, -1.0, 1.0))
        th_i_prime = (pi / 2.0) - acos(kn_dot)

        projected = k_in - kn_dot * n_surf
        pln = float(np.linalg.norm(projected))
        if pln > 1e-12:
            projected = projected / pln
        else:
            projected[:] = 0.0
        p1 = float(np.dot(projected, np.cross(n_surf, np.array([0.0, 0.0, -1.0]))))
        e1_temp = np.cross(n_surf, np.array([0.0, 0.0, -1.0]))
        if float(np.linalg.norm(e1_temp)) < 1e-12:
            e1_temp = np.cross(n_surf, np.array([1.0, 0.0, 0.0]))
        e1_temp = _unit(e1_temp)
        e2_temp = np.cross(n_surf, e1_temp)
        p1 = float(np.dot(projected, e1_temp))
        p2 = float(np.dot(projected, e2_temp))
        phi_i_prime = (pi / 2.0) - np.arctan2(p2, p1)

        th_t = float(transmit_angle_grazing(th_i_prime, n2))
        k0 = 2.0 * pi / wavelength
        k_scat = float(k0 * np.sqrt(max(np.real(n2 * n2), 0.0)))
        k_x_scat = float(k_scat * np.cos(th_t) * np.sin(phi_i_prime))
        k_y_scat = float(k_scat * np.cos(th_t) * np.cos(phi_i_prime))
        re_k_z, _ = ktz_components(k0, n2, th_t)
        re_k_z = float(-re_k_z)

        k_in_crystal = np.array([k_x_scat, k_y_scat, re_k_z], dtype=np.float64)
        all_q, status = solve_q(
            k_in_crystal,
            k_scat,
            g_vec,
            sigma_rad,
            gamma_rad,
            eta,
            1000,
        )
        if int(status) != 0 or all_q.shape[0] == 0:
            continue

        ctx = _BeamContext(
            i_plane=i_plane,
            r_sample=r_sample,
            detector_pos=detector_pos,
            n_det_rot=n_det_rot,
            e1_det=e1_det,
            e2_det=e2_det,
            k_scat=k_scat,
            k_x_scat=k_x_scat,
            k_y_scat=k_y_scat,
            re_k_z=re_k_z,
            k_in_crystal=k_in_crystal,
            all_q=np.asarray(all_q, dtype=np.float64),
        )
        if fallback_ctx is None:
            fallback_ctx = ctx

        if selected_native_col is None or selected_native_row is None:
            return ctx

        cols, rows, valid = _project_arc_to_detector(
            np.asarray(ctx.all_q[:, :3], dtype=np.float64),
            ctx,
            geometry,
            n2,
        )
        if not np.any(valid):
            continue

        d2 = np.square(cols[valid] - float(selected_native_col)) + np.square(
            rows[valid] - float(selected_native_row)
        )
        cur_best = float(np.min(d2))
        if cur_best < best_dist2:
            best_dist2 = cur_best
            best_ctx = ctx
            # Good-enough pixel-level match; stop scanning further beam samples.
            if best_dist2 <= 0.25:
                break

    if best_ctx is not None:
        return best_ctx
    if fallback_ctx is not None:
        return fallback_ctx
    raise ValueError("No valid beam sample produced an Ewald-Bragg intersection arc.")


def _circle_frame(k_in_crystal: np.ndarray, k_scat: float, g_vec: np.ndarray):
    g_sq = float(np.dot(g_vec, g_vec))
    if g_sq < 1e-14:
        return None

    ax = -float(k_in_crystal[0])
    ay = -float(k_in_crystal[1])
    az = -float(k_in_crystal[2])
    a_sq = ax * ax + ay * ay + az * az
    if a_sq < 1e-14:
        return None
    a_len = sqrt(a_sq)

    c = (g_sq + a_sq - k_scat * k_scat) / (2.0 * a_len)
    circle_r_sq = g_sq - c * c
    if circle_r_sq < 0.0:
        return None

    ah = np.array([ax, ay, az], dtype=np.float64) / a_len
    o = c * ah

    anchor = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(anchor, ah))) > 0.9999:
        anchor = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e1 = anchor - float(np.dot(anchor, ah)) * ah
    if float(np.linalg.norm(e1)) < 1e-14:
        return None
    e1 = _unit(e1)

    e2 = np.cross(ah, e1)
    if float(np.linalg.norm(e2)) < 1e-14:
        return None
    e2 = _unit(e2)
    return o, e1, e2


def _project_arc_to_detector(
    all_q: np.ndarray,
    beam_ctx: _BeamContext,
    geometry: IntersectionGeometry,
    n2: complex,
):
    cols = np.full(all_q.shape[0], np.nan, dtype=np.float64)
    rows = np.full(all_q.shape[0], np.nan, dtype=np.float64)
    valid = np.zeros(all_q.shape[0], dtype=bool)

    center_col = float(geometry.center_col)
    center_row = float(geometry.center_row)
    pixel_size = float(geometry.pixel_size_m)
    image_size = int(geometry.image_size)

    for idx, q in enumerate(all_q):
        qx, qy, qz = float(q[0]), float(q[1]), float(q[2])
        k_tx_prime = qx + beam_ctx.k_x_scat
        k_ty_prime = qy + beam_ctx.k_y_scat
        k_tz_prime = qz + beam_ctx.re_k_z

        kr = sqrt(k_tx_prime * k_tx_prime + k_ty_prime * k_ty_prime)
        if kr < 1e-12:
            twotheta_t_prime = 0.0
        else:
            twotheta_t_prime = np.arctan(k_tz_prime / kr)

        twotheta_t = (
            np.arccos(np.clip(np.cos(twotheta_t_prime) * np.real(n2), -1.0, 1.0))
            * np.sign(twotheta_t_prime)
        )
        phi_f = np.arctan2(k_tx_prime, k_ty_prime)
        k_tx_f = beam_ctx.k_scat * np.cos(twotheta_t) * np.sin(phi_f)
        k_ty_f = beam_ctx.k_scat * np.cos(twotheta_t) * np.cos(phi_f)
        k_tz_f = beam_ctx.k_scat * np.sin(twotheta_t)
        kf = np.array([k_tx_f, k_ty_f, k_tz_f], dtype=np.float64)
        kf_prime = beam_ctx.r_sample @ kf

        dx, dy, dz, valid_det = intersect_infinite_line_plane(
            beam_ctx.i_plane,
            kf_prime,
            beam_ctx.detector_pos,
            beam_ctx.n_det_rot,
        )
        if not valid_det:
            continue

        plane_to_det = np.array(
            [
                dx - beam_ctx.detector_pos[0],
                dy - beam_ctx.detector_pos[1],
                dz - beam_ctx.detector_pos[2],
            ],
            dtype=np.float64,
        )
        x_det = float(np.dot(plane_to_det, beam_ctx.e1_det))
        y_det = float(np.dot(plane_to_det, beam_ctx.e2_det))

        row = center_row - y_det / pixel_size
        col = center_col + x_det / pixel_size
        if not (0.0 <= row < image_size and 0.0 <= col < image_size):
            continue

        cols[idx] = col
        rows[idx] = row
        valid[idx] = True

    return cols, rows, valid


def analyze_reflection_intersection(
    *,
    h: int,
    k: int,
    l: int,
    lattice_a: float,
    lattice_c: float,
    selected_native_col: float,
    selected_native_row: float,
    geometry: IntersectionGeometry,
    beam: BeamSamples,
    mosaic: MosaicParams,
    n2: complex,
    sphere_res: int = 72,
) -> ReflectionIntersectionAnalysis:
    """Compute Bragg sphere field and Ewald intersection arc for a reflection."""

    g_vec = compute_g_vec(int(h), int(k), int(l), float(lattice_a), float(lattice_c))
    g_mag = float(np.linalg.norm(g_vec))
    if g_mag < 1e-12:
        raise ValueError("Reflection has near-zero |G|.")

    beam_ctx = _solve_for_best_beam_sample(
        int(h),
        int(k),
        int(l),
        g_vec,
        beam,
        geometry,
        mosaic,
        n2,
        selected_native_col=float(selected_native_col),
        selected_native_row=float(selected_native_row),
    )
    arc_q = np.asarray(beam_ctx.all_q[:, :3], dtype=np.float64)
    arc_intensity = np.asarray(beam_ctx.all_q[:, 3], dtype=np.float64)

    circle = _circle_frame(beam_ctx.k_in_crystal, beam_ctx.k_scat, g_vec)
    if circle is None:
        raise ValueError("Could not compute an arc-angle frame for the selected reflection.")
    o, e1, e2 = circle
    rel = arc_q - o[None, :]
    a1 = rel @ e1
    a2 = rel @ e2
    arc_angle = np.unwrap(np.arctan2(a2, a1))
    # Keep arc-angle convention aligned with GUI azimuth usage where 90° maps
    # to 0° on the displayed profile.
    arc_angle_deg = np.rad2deg(arc_angle) - 90.0

    arc_cols, arc_rows, arc_valid = _project_arc_to_detector(arc_q, beam_ctx, geometry, n2)
    if np.any(arc_valid):
        d2 = np.square(arc_cols[arc_valid] - float(selected_native_col)) + np.square(
            arc_rows[arc_valid] - float(selected_native_row)
        )
        nearest_valid_idx = int(np.argmin(d2))
        nearest_index = int(np.nonzero(arc_valid)[0][nearest_valid_idx])
    else:
        nearest_index = int(np.argmax(arc_intensity))

    u = np.linspace(0.0, 2.0 * np.pi, int(sphere_res))
    v = np.linspace(0.0, np.pi, int(sphere_res))
    uu, vv = np.meshgrid(u, v)
    sphere_x = g_mag * np.sin(vv) * np.cos(uu)
    sphere_y = g_mag * np.sin(vv) * np.sin(uu)
    sphere_z = g_mag * np.cos(vv)
    sigma_rad = float(np.deg2rad(mosaic.sigma_mosaic_deg))
    gamma_rad = float(np.deg2rad(mosaic.gamma_mosaic_deg))
    sphere_intensity = compute_intensity_array(
        sphere_x,
        sphere_y,
        sphere_z,
        g_vec,
        sigma_rad,
        gamma_rad,
        float(mosaic.eta),
    )
    sphere_intensity = np.nan_to_num(
        np.asarray(sphere_intensity, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return ReflectionIntersectionAnalysis(
        hkl=(int(h), int(k), int(l)),
        g_vec=g_vec,
        sphere_x=np.asarray(sphere_x, dtype=np.float64),
        sphere_y=np.asarray(sphere_y, dtype=np.float64),
        sphere_z=np.asarray(sphere_z, dtype=np.float64),
        sphere_intensity=sphere_intensity,
        arc_q=arc_q,
        arc_intensity=arc_intensity,
        arc_angle_deg=np.asarray(arc_angle_deg, dtype=np.float64),
        arc_detector_col=arc_cols,
        arc_detector_row=arc_rows,
        arc_valid_mask=arc_valid,
        nearest_index=nearest_index,
        selected_native_col=float(selected_native_col),
        selected_native_row=float(selected_native_row),
    )


def plot_intersection_analysis(result: ReflectionIntersectionAnalysis) -> plt.Figure:
    """Create the 3D Bragg-sphere + 2D arc-profile figure."""

    fig = plt.figure(figsize=(13.0, 5.3))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    sphere_vals = np.asarray(result.sphere_intensity, dtype=np.float64)
    sphere_for_color = np.log1p(np.maximum(sphere_vals, 0.0))
    flat = sphere_for_color.ravel()
    vmin = float(np.percentile(flat, 1.0))
    vmax = float(np.percentile(flat, 99.5))
    if vmax <= vmin:
        vmin = float(np.min(flat))
        vmax = float(np.max(flat))
    denom = max(vmax - vmin, 1e-30)
    norm = np.clip((sphere_for_color - vmin) / denom, 0.0, 1.0)
    colors = plt.cm.viridis(norm)
    # Keep the Bragg sphere highly translucent so the intersection band remains visible.
    colors[..., 3] = 0.16

    ax3d.plot_surface(
        result.sphere_x,
        result.sphere_y,
        result.sphere_z,
        rstride=1,
        cstride=1,
        facecolors=colors,
        linewidth=0,
        edgecolor="none",
        shade=False,
        antialiased=False,
    )
    radius = float(np.linalg.norm(result.g_vec))
    arc_q = np.asarray(result.arc_q, dtype=np.float64)
    arc_intensity = np.nan_to_num(
        np.asarray(result.arc_intensity, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    arc_norm = np.linalg.norm(arc_q, axis=1)
    safe_norm = np.where(arc_norm > 1e-20, arc_norm, 1.0)
    # Draw the arc slightly outside the Bragg sphere so it remains visible.
    arc_offset = max(radius * 2.0e-3, 1.0e-6)
    arc_q_vis = arc_q + (arc_q / safe_norm[:, None]) * arc_offset
    arc_strength = np.log1p(np.maximum(arc_intensity, 0.0))
    s_lo = float(np.percentile(arc_strength, 5.0))
    s_hi = float(np.percentile(arc_strength, 99.0))
    s_den = max(s_hi - s_lo, 1e-30)
    arc_strength = np.clip((arc_strength - s_lo) / s_den, 0.0, 1.0)
    hi_thr = float(np.percentile(arc_strength, 80.0))
    hi_mask = arc_strength >= hi_thr
    if int(np.count_nonzero(hi_mask)) < 8:
        top_n = max(8, min(48, arc_strength.size // 5))
        hi_mask = np.zeros_like(arc_strength, dtype=bool)
        hi_mask[np.argsort(arc_strength)[-top_n:]] = True

    ax3d.plot(
        arc_q_vis[:, 0],
        arc_q_vis[:, 1],
        arc_q_vis[:, 2],
        color="black",
        linewidth=4.2,
        alpha=0.8,
    )
    ax3d.plot(
        arc_q_vis[:, 0],
        arc_q_vis[:, 1],
        arc_q_vis[:, 2],
        color="#fff06a",
        linewidth=2.6,
        alpha=0.95,
        label="Ewald-Bragg intersection arc",
    )
    ax3d.scatter(
        arc_q_vis[hi_mask, 0],
        arc_q_vis[hi_mask, 1],
        arc_q_vis[hi_mask, 2],
        c=arc_strength[hi_mask],
        cmap="autumn",
        s=22.0 + 160.0 * arc_strength[hi_mask],
        alpha=0.95,
        depthshade=False,
        label="Highlighted intersection band",
    )
    nearest_q = arc_q_vis[result.nearest_index]
    ax3d.scatter(
        [nearest_q[0]],
        [nearest_q[1]],
        [nearest_q[2]],
        color="orangered",
        s=58,
        label="Selected peak match on arc",
    )
    g = result.g_vec
    ax3d.scatter([g[0]], [g[1]], [g[2]], color="crimson", s=36, label="G")
    ax3d.plot([0.0, g[0]], [0.0, g[1]], [0.0, g[2]], color="crimson", linewidth=1.3)

    lim = radius * 1.1
    ax3d.set_xlim(-lim, lim)
    ax3d.set_ylim(-lim, lim)
    ax3d.set_zlim(-lim, lim)
    try:
        ax3d.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass
    ax3d.set_xlabel("Qx")
    ax3d.set_ylabel("Qy")
    ax3d.set_zlabel("Qz")
    h, k, l = result.hkl
    ax3d.set_title(f"Bragg sphere + arc for HKL=({h},{k},{l})")
    ax3d.legend(loc="upper right")

    ax2d.plot(
        result.arc_angle_deg,
        arc_intensity,
        color="#2f6db2",
        linewidth=2.0,
        label="Arc intensity",
    )
    ax2d.fill_between(
        result.arc_angle_deg,
        0.0,
        arc_intensity,
        color="#7fb2e8",
        alpha=0.22,
    )
    ax2d.scatter(
        [result.arc_angle_deg[result.nearest_index]],
        [arc_intensity[result.nearest_index]],
        color="orangered",
        s=40,
        label="Selected peak match on arc",
        zorder=5,
    )
    ax2d.set_xlabel("Arc angle (deg)")
    ax2d.set_ylabel("Arc intensity (a.u.)")
    ax2d.set_title("Intersection arc intensity profile")
    ax2d.grid(True, alpha=0.25)
    ax2d.legend(loc="best")

    fig.tight_layout()
    return fig
