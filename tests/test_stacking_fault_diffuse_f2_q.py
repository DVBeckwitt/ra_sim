from pathlib import Path

import numpy as np
import Dans_Diffraction as dif
from Dans_Diffraction.functions_crystallography import (
    xray_dispersion_corrections,
    xray_scattering_factor,
)

from ra_sim.utils.stacking_fault import (
    _F2,
    _HT_BASE_CACHE,
    _cell_c_from_cif,
    _energy_kev_from_lambda,
    _get_base_curves,
    _sites_from_cif_with_factors,
)


def test_diffuse_f2_uses_legacy_q_reference_axes():
    cif = Path("tests/Diffuse/PbI2_2H.cif")
    _HT_BASE_CACHE.clear()

    common = dict(
        cif_path=str(cif),
        hk_list=[(1, 0)],
        L_step=0.1,
        L_max=2.0,
        two_theta_max=None,
        lambda_=1.5406,
        occ_factors=1.0,
        phase_z_divisor=3.0,
        iodine_single_plane=True,
    )

    ref = _get_base_curves(
        **common,
        a_lattice=4.557,
        c_lattice=None,
    )
    shifted_axes = _get_base_curves(
        **common,
        a_lattice=9.0,
        c_lattice=80.0,
    )

    f2_ref = np.asarray(ref[(1, 0)]["F2"], dtype=float)
    f2_shifted = np.asarray(shifted_axes[(1, 0)]["F2"], dtype=float)

    assert np.allclose(f2_ref, f2_shifted, rtol=1e-12, atol=1e-10)


def test_f2_helper_matches_legacy_diffuse_shape_for_m1():
    cif = Path("tests/Diffuse/PbI2_2H.cif")
    l_vals = np.arange(0.0, 10.0 + 0.01 / 2.0, 0.01, dtype=float)
    hk_pairs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

    c_2h = _cell_c_from_cif(str(cif))
    sites = _sites_from_cif_with_factors(str(cif), occ_factors=1.0)
    energy_kev = _energy_kev_from_lambda(1.5406)

    f2_helper = np.zeros_like(l_vals)
    for h, k in hk_pairs:
        f2_helper += _F2(
            h,
            k,
            l_vals,
            c_2h,
            energy_kev,
            sites,
            a_axis=4.557,
            phase_z_divisor=3.0,
        )

    iodine_z = None
    with open(cif, "r", encoding="utf-8", errors="ignore") as fp:
        for ln in fp:
            if ln.strip().startswith("I1"):
                parts = ln.split()
                if len(parts) >= 5:
                    iodine_z = float(parts[4])
                    break
    assert iodine_z is not None

    xtl = dif.Crystal(str(cif))
    xtl.Symmetry.generate_matrices()
    xtl.generate_structure()
    structure = xtl.Structure
    legacy_sites = [
        (float(structure.u[i]), float(structure.v[i]), float(structure.w[i]), str(structure.type[i]))
        for i in range(len(structure.u))
    ]

    ion_labels = {"Pb": "Pb2+", "I": "I1-"}
    neutral_labels = {"Pb": "Pb", "I": "I"}
    energy_cuka = 12398.4193 / 1.5406

    def _legacy_ff(sym: str, q_vals: np.ndarray) -> np.ndarray:
        element = "Pb" if str(sym).lower().startswith("pb") else "I"
        q = np.asarray(q_vals, dtype=float).reshape(-1)
        f0 = xray_scattering_factor([ion_labels[element]], q)[:, 0]
        f1, f2 = xray_dispersion_corrections(
            [neutral_labels[element]],
            energy_kev=[energy_cuka / 1000.0],
        )
        return (f0 + float(f1[0, 0]) + 1j * float(f2[0, 0])).reshape(q_vals.shape)

    f2_legacy = np.zeros_like(l_vals)
    for h, k in hk_pairs:
        q_mag = 2.0 * np.pi * np.sqrt(
            (4.0 / 3.0) * (h * h + h * k + k * k) / (4.557**2)
            + (l_vals**2) / (c_2h**2)
        )
        fixed = np.zeros_like(l_vals, dtype=complex)
        iodine_factor = np.zeros_like(l_vals, dtype=complex)
        for x, y, z, sym in legacy_sites:
            ff = _legacy_ff(sym, q_mag)
            phase_xy = np.exp(2j * np.pi * (h * x + k * y))
            if str(sym).startswith("I"):
                iodine_factor += ff * phase_xy
            else:
                fixed += ff * phase_xy * np.exp(2j * np.pi * l_vals * (z / 3.0))
        f2_legacy += np.abs(
            fixed + iodine_factor * np.exp(2j * np.pi * l_vals * (iodine_z / 3.0))
        ) ** 2

    assert np.allclose(f2_helper, f2_legacy, rtol=1e-12, atol=1e-10)
