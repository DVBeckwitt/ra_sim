"""Plot mosaic omega distribution and Bragg sphere for a given reflection.

Usage:
  python tests/plot_mosaic_omega_and_bragg_sphere.py --H 1 --K 0 --L 0

Notes:
- sigma_deg and gamma_deg are the pseudo-Voigt widths in degrees.
- eta is the pseudo-Voigt mixing parameter (0=Gaussian, 1=Lorentzian).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox


def omega_components(dtheta_rad, sigma_rad, gamma_rad, eta):
    a_gauss = 1.0 / (sigma_rad * np.sqrt(2.0 * np.pi))
    gauss = a_gauss * np.exp(-0.5 * (dtheta_rad / sigma_rad) ** 2)

    a_lor = 1.0 / (np.pi * gamma_rad)
    lor = a_lor / (1.0 + (dtheta_rad / gamma_rad) ** 2)

    omega = (1.0 - eta) * gauss + eta * lor
    return gauss, lor, omega


def compute_g_vec(H, K, L, a, c):
    gz0 = 2.0 * np.pi * (L / c)
    gr0 = 4.0 * np.pi / a * np.sqrt((H * H + H * K + K * K) / 3.0)
    return np.array([0.0, gr0, gz0], dtype=np.float64)


def plot_omega(ax, sigma_deg, gamma_deg, eta, half_range_deg):
    sigma_rad = np.deg2rad(sigma_deg)
    gamma_rad = np.deg2rad(gamma_deg)

    dtheta_deg = np.linspace(-half_range_deg, half_range_deg, 1000)
    dtheta_rad = np.deg2rad(dtheta_deg)

    gauss, lor, omega = omega_components(dtheta_rad, sigma_rad, gamma_rad, eta)

    ax.plot(dtheta_deg, gauss, label="Gaussian", lw=1.8)
    ax.plot(dtheta_deg, lor, label="Lorentzian", lw=1.8)
    ax.plot(dtheta_deg, omega, label="Pseudo-Voigt", lw=2.2)

    ax.set_xlabel("Delta theta (deg)")
    ax.set_ylabel("omega(Delta theta)")
    ax.set_title("Mosaic omega distribution")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def wrap_to_pi(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def sigma_on_sphere(x, y, z, G, sigma_rad, gamma_rad, eta):
    r = np.sqrt(x * x + y * y)
    # Signed grazing angle to avoid mirrored distributions across the plane.
    theta = np.arctan2(z, r)

    gr = np.sqrt(G[0] ** 2 + G[1] ** 2)
    theta0 = np.arctan2(G[2], gr)
    dtheta = wrap_to_pi(theta - theta0)

    a_gauss = 1.0 / (sigma_rad * np.sqrt(2.0 * np.pi))
    gauss = a_gauss * np.exp(-0.5 * (dtheta / sigma_rad) ** 2)

    a_lor = 1.0 / (np.pi * gamma_rad)
    lor = a_lor / (1.0 + (dtheta / gamma_rad) ** 2)

    omega = (1.0 - eta) * gauss + eta * lor

    r_g = np.sqrt(G[0] ** 2 + G[1] ** 2 + G[2] ** 2)
    # Keep the same geometry-normalization used by the simulation kernel.
    # Dividing by cos(theta) creates pole amplification when eta>0 (Lorentzian
    # tails), which collapses the color scale and makes the distribution appear
    # to vanish.
    denom = 2.0 * np.pi * r_g * r_g
    return omega / denom


def plot_bragg_sphere(
    ax,
    H,
    K,
    L,
    a,
    c,
    sphere_res,
    sigma_deg,
    gamma_deg,
    eta,
    color_scale,
    vmin_pct,
    vmax_pct,
):
    G = compute_g_vec(H, K, L, a, c)
    R = np.sqrt(G[0] ** 2 + G[1] ** 2 + G[2] ** 2)

    u = np.linspace(0.0, 2.0 * np.pi, sphere_res)
    v = np.linspace(0.0, np.pi, sphere_res)
    uu, vv = np.meshgrid(u, v)

    x = R * np.sin(vv) * np.cos(uu)
    y = R * np.sin(vv) * np.sin(uu)
    z = R * np.cos(vv)

    sigma_rad = np.deg2rad(sigma_deg)
    gamma_rad = np.deg2rad(gamma_deg)
    sigma_vals = sigma_on_sphere(x, y, z, G, sigma_rad, gamma_rad, eta)
    sigma_vals = np.nan_to_num(sigma_vals, nan=0.0, posinf=0.0, neginf=0.0)

    if color_scale == "log":
        sigma_for_color = np.log1p(sigma_vals)
    else:
        sigma_for_color = sigma_vals

    flat = sigma_for_color.ravel()
    vmin = np.percentile(flat, vmin_pct)
    vmax = np.percentile(flat, vmax_pct)
    if vmax <= vmin:
        vmin = sigma_for_color.min()
        vmax = sigma_for_color.max()
    denom = vmax - vmin
    if denom < 1e-30:
        denom = 1e-30
    sigma_norm = (sigma_for_color - vmin) / denom
    sigma_norm = np.clip(sigma_norm, 0.0, 1.0)
    colors = plt.cm.viridis(sigma_norm)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=colors, linewidth=0, antialiased=False)
    ax.scatter([G[0]], [G[1]], [G[2]], color="crimson", s=40, label="G")
    ax.plot([0.0, G[0]], [0.0, G[1]], [0.0, G[2]], color="crimson", lw=1.5)

    ax.set_title(f"Bragg sphere |G| for H,K,L = ({H},{K},{L})")
    ax.set_xlabel("Qx")
    ax.set_ylabel("Qy")
    ax.set_zlabel("Qz")

    lim = R * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass
    ax.legend(loc="upper right")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot omega distribution and Bragg sphere.")
    parser.add_argument("--H", type=int, default=1, help="Miller index H")
    parser.add_argument("--K", type=int, default=0, help="Miller index K")
    parser.add_argument("--L", type=int, default=0, help="Miller index L")
    parser.add_argument("--a", type=float, default=4.0, help="Lattice a (A)")
    parser.add_argument("--c", type=float, default=7.0, help="Lattice c (A)")
    parser.add_argument("--sigma-deg", type=float, default=0.5, help="Gaussian width (deg)")
    parser.add_argument("--gamma-deg", type=float, default=0.2, help="Lorentzian HWHM (deg)")
    parser.add_argument("--eta", type=float, default=0.1, help="Pseudo-Voigt mixing (0..1)")
    parser.add_argument("--half-range-deg", type=float, default=5.0, help="Plot range +/- (deg)")
    parser.add_argument("--sphere-res", type=int, default=60, help="Sphere resolution")
    parser.add_argument(
        "--color-scale",
        choices=("log", "linear"),
        default="log",
        help="Color scale for sigma(theta) on the Bragg sphere.",
    )
    parser.add_argument(
        "--vmin-percentile",
        type=float,
        default=1.0,
        help="Lower percentile for color normalization.",
    )
    parser.add_argument(
        "--vmax-percentile",
        type=float,
        default=99.5,
        help="Upper percentile for color normalization.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(1, 2, left=0.06, right=0.98, top=0.95, bottom=0.33, wspace=0.22)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")

    state = {
        "H": int(args.H),
        "K": int(args.K),
        "L": int(args.L),
        "sigma_deg": float(args.sigma_deg),
        "gamma_deg": float(args.gamma_deg),
        "eta": float(args.eta),
    }

    def redraw():
        ax1.cla()
        ax2.cla()
        plot_omega(ax1, state["sigma_deg"], state["gamma_deg"], state["eta"], args.half_range_deg)
        plot_bragg_sphere(
            ax2,
            state["H"],
            state["K"],
            state["L"],
            args.a,
            args.c,
            args.sphere_res,
            state["sigma_deg"],
            state["gamma_deg"],
            state["eta"],
            args.color_scale,
            args.vmin_percentile,
            args.vmax_percentile,
        )
        fig.suptitle(
            (
                f"HKL=({state['H']},{state['K']},{state['L']})   "
                f"sigma={state['sigma_deg']:.3f} deg   "
                f"gamma={state['gamma_deg']:.3f} deg   "
                f"eta={state['eta']:.3f}"
            ),
            fontsize=11,
        )
        fig.canvas.draw_idle()

    redraw()

    ax_sigma = fig.add_axes([0.10, 0.22, 0.55, 0.03])
    ax_gamma = fig.add_axes([0.10, 0.17, 0.55, 0.03])
    ax_eta = fig.add_axes([0.10, 0.12, 0.55, 0.03])

    sigma_slider = Slider(
        ax_sigma, "sigma (deg)", 0.01, 5.0, valinit=state["sigma_deg"], valstep=0.01
    )
    gamma_slider = Slider(
        ax_gamma, "gamma (deg)", 0.01, 5.0, valinit=state["gamma_deg"], valstep=0.01
    )
    eta_slider = Slider(ax_eta, "eta", 0.0, 1.0, valinit=state["eta"], valstep=0.01)

    def _on_profile_change(_val):
        state["sigma_deg"] = float(sigma_slider.val)
        state["gamma_deg"] = float(gamma_slider.val)
        state["eta"] = float(eta_slider.val)
        redraw()

    sigma_slider.on_changed(_on_profile_change)
    gamma_slider.on_changed(_on_profile_change)
    eta_slider.on_changed(_on_profile_change)

    ax_h = fig.add_axes([0.72, 0.205, 0.07, 0.05])
    ax_k = fig.add_axes([0.81, 0.205, 0.07, 0.05])
    ax_l = fig.add_axes([0.90, 0.205, 0.07, 0.05])
    h_box = TextBox(ax_h, "H", initial=str(state["H"]))
    k_box = TextBox(ax_k, "K", initial=str(state["K"]))
    l_box = TextBox(ax_l, "L", initial=str(state["L"]))

    ax_apply = fig.add_axes([0.72, 0.12, 0.25, 0.06])
    apply_button = Button(ax_apply, "Apply H,K,L")

    def _apply_hkl(_event=None):
        try:
            state["H"] = int(round(float(h_box.text.strip())))
            state["K"] = int(round(float(k_box.text.strip())))
            state["L"] = int(round(float(l_box.text.strip())))
        except ValueError:
            print("Invalid H/K/L input. Enter numeric values.")
            return
        redraw()

    apply_button.on_clicked(_apply_hkl)
    h_box.on_submit(_apply_hkl)
    k_box.on_submit(_apply_hkl)
    l_box.on_submit(_apply_hkl)

    plt.show()


if __name__ == "__main__":
    main()
