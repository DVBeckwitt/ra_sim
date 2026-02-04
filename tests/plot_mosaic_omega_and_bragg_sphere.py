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
    denom = 2.0 * np.pi * r_g * r_g * np.maximum(np.cos(theta), 1e-12)
    return omega / denom


def plot_bragg_sphere(ax, H, K, L, a, c, sphere_res, sigma_deg, gamma_deg, eta):
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
    sigma_norm = (sigma_vals - sigma_vals.min()) / (sigma_vals.max() - sigma_vals.min() + 1e-30)
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
    return parser.parse_args()


def main():
    args = parse_args()

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    plot_omega(ax1, args.sigma_deg, args.gamma_deg, args.eta, args.half_range_deg)
    plot_bragg_sphere(
        ax2,
        args.H,
        args.K,
        args.L,
        args.a,
        args.c,
        args.sphere_res,
        args.sigma_deg,
        args.gamma_deg,
        args.eta,
    )

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
