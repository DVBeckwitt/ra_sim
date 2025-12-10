# stacking_fault.py - Markov HT with diffuse-consistent F^2, C2H, and factors

import os
import numpy as np

# Global constants
A_HEX   = 4.557            # Å
P_CLAMP = 1e-6
N_P, A_C = 3, 17.98e-10    # number of sublayers, real-space area (m^2)
AREA    = (2*np.pi)**2 / A_C * N_P
_TWO_PI = 2.0 * np.pi

# Cache of base L grids and F2 values keyed by parameters that do not depend
# on occupancy or stacking probability. Each entry: {(h,k): {"L": array, "F2": array}}
_HT_BASE_CACHE: dict[tuple, dict] = {}

# ----------------------------- occupancy helper -----------------------------
def _temp_cif_with_occ(cif_in: str, occ):
    """
    Return path of a temporary CIF in which every `_atom_site_occupancy`
    is multiplied by `occ`.

    Accepts:
      - scalar      -> global factor
      - list/tuple  -> per-site factors; if length != n_sites, occ[0] is used.
    """
    import tempfile, CifFile

    cf = CifFile.ReadCif(cif_in)
    block_name = list(cf.keys())[0]
    block = cf[block_name]

    occ_field = block.get('_atom_site_occupancy')
    if occ_field is None:
        labels = block.get('_atom_site_label')
        occ_field = ['1.0'] * (len(labels) if isinstance(labels, list) else 1)
        block['_atom_site_occupancy'] = occ_field

    # harmonise occ -> list of factors
    if isinstance(occ, (list, tuple)):
        if len(occ) != len(occ_field):
            occ = [occ[0]] * len(occ_field)
    else:
        occ = [occ] * len(occ_field)

    for i, fac in enumerate(occ):
        occ_field[i] = str(float(occ_field[i]) * float(fac))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.cif')
    tmp.close()
    try:  # PyCifRW >= 5
        CifFile.WriteCif(cf, tmp.name)
    except AttributeError:
        with open(tmp.name, 'w') as f:
            f.write(cf.WriteOut())
    return tmp.name, os.unlink


# -------------------------- low-level physics helpers -----------------------
def _cell_c_from_cif(cif_path: str) -> float:
    """Return the c lattice parameter from cif_path."""
    import re
    pat = re.compile(r"_cell_length_c\s+([\d.]+)")
    with open(cif_path, "r", encoding="utf-8", errors="ignore") as fp:
        for ln in fp:
            m = pat.match(ln)
            if m:
                return float(m.group(1))
    raise ValueError(f"_cell_length_c not found in {cif_path}")


def _sites_from_cif(cif_path: str):
    """Return atomic sites from cif_path with symmetry applied."""
    import Dans_Diffraction as dif
    xtl = dif.Crystal(str(cif_path))
    xtl.Symmetry.generate_matrices()
    xtl.generate_structure()
    st = xtl.Structure
    return [
        (float(st.u[i]), float(st.v[i]), float(st.w[i]), str(st.type[i]))
        for i in range(len(st.u))
    ]


def _energy_kev_from_lambda(lambda_a: float) -> float:
    """Convert wavelength in Å to energy in keV."""
    return (12398.4193 / float(lambda_a)) / 1000.0


# Scattering-factor composition identical to diffuse_cif_toggle.py
# f0 from ionic species, dispersion corrections from neutral species at given energy.
ION   = {"Pb": "Pb2+", "I": "I1-"}
NEUTR = {"Pb": "Pb",   "I": "I"}

def _element_key(sym: str) -> str:
    """Map CIF site symbol to periodic table symbol used in the tables."""
    # Handle common cases like 'Pb1', 'I1', 'I', 'Pb'
    s = "".join(ch for ch in sym if ch.isalpha())
    if s in ION:
        return s
    # Fallback: try first letter uppercase + optional second letter lowercase
    s = sym.strip()
    if s and s[0].isalpha():
        cand = s[0].upper() + (s[1].lower() if len(s) > 1 and s[1].isalpha() else "")
        if cand in ION:
            return cand
    raise KeyError(f"Unknown element symbol in CIF type '{sym}'")

def f_comp(el_sym: str, Q: np.ndarray, energy_kev: float) -> np.ndarray:
    """
    Composite atomic form factor f = f0 + f' + i f''.

    Parameters
    ----------
    el_sym : str
        CIF site symbol (e.g. 'Pb', 'I', 'Pb1').
    Q : ndarray
        |Q| magnitude in Å^-1.
    energy_kev : float
        Photon energy in keV.
    """
    from Dans_Diffraction.functions_crystallography import (
        xray_scattering_factor,
        xray_dispersion_corrections,
    )

    key = _element_key(el_sym)
    q = np.asarray(Q, dtype=float).reshape(-1)
    # f0 from ionic form
    f0 = xray_scattering_factor([ION[key]], q)[:, 0]
    # dispersion from neutral form
    f1, f2 = xray_dispersion_corrections([NEUTR[key]], energy_kev=[float(energy_kev)])
    f1 = float(f1[0, 0])
    f2 = float(f2[0, 0])
    out = f0 + f1 + 1j * f2
    return out.reshape(Q.shape)


def _Qmag(h: int, k: int, L: np.ndarray, c_2h: float) -> np.ndarray:
    """
    |Q| for a hexagonal lattice using the diffuse_cif_toggle convention:
    Q = 2π * sqrt( (4/3)*(h^2+k^2+hk)/a^2 + (L^2)/c^2 )
    where L is the dimensionless ℓ coordinate and c is the 2H c parameter.
    """
    inv_d2 = (4.0/3.0)*(h*h + k*k + h*k)/A_HEX**2 + (L**2)/(c_2h**2)
    return 2.0*np.pi*np.sqrt(inv_d2)


def _F2(h: int, k: int, L: np.ndarray, c_2h: float, energy_kev: float, sites) -> np.ndarray:
    """
    Return |F|^2 using the same single-layer phase used in diffuse_cif_toggle:
    phase = exp(2πi[hx + ky + L*(z/3)])
    """
    Q = _Qmag(h, k, L, c_2h)
    F = np.zeros_like(Q, dtype=complex)
    for x, y, z, sym in sites:
        ff = f_comp(sym, Q, energy_kev)
        phase_xy = np.exp(1j * _TWO_PI * (h * x + k * y))
        phase_z  = np.exp(1j * _TWO_PI * (L * (float(z) / 3.0)))
        F += ff * phase_xy * phase_z
    return np.abs(F) ** 2


# -------------------------- Markov transfer-matrix HT ------------------------
def _rho_alpha_from_p(p: float):
    p_clamped = float(np.clip(p, 0.0, 1.0))
    return 1.0 - p_clamped, p_clamped  # rho, alpha


def _class_stationary(rho: float, alpha: float):
    denom = 2.0 - alpha - rho
    P_S = (1.0 - alpha) / denom
    P_E = (1.0 - rho) / (2.0 * denom)
    return P_S, P_E, P_E


def _slip_phase(h: int, k: int) -> float:
    return _TWO_PI * ((2*h + k) / 3.0)


def _finite_series_sum(t: np.ndarray, N: int) -> np.ndarray:
    """Return ∑_{n=1}^{N-1} (N-n) t^n for complex *t* (vectorised)."""

    t = np.asarray(t, dtype=complex)
    N = int(max(1, N))

    if N == 1:
        return np.zeros_like(t, dtype=complex)

    one = 1.0 + 0.0j
    mask = np.isclose(t, one)
    out = np.empty_like(t, dtype=complex)

    if np.any(~mask):
        t_nm = t[~mask]
        denom = one - t_nm
        S1 = t_nm * (1 - t_nm ** (N - 1)) / denom
        S2 = t_nm * (
            1 - N * t_nm ** (N - 1) + (N - 1) * t_nm ** N
        ) / (denom ** 2)
        out[~mask] = N * S1 - S2

    if np.any(mask):
        # For t → 1 the series approaches N(N-1)/2 exactly.
        out[mask] = (N * (N - 1) / 2.0) * one

    return out


def _R_from_transfer(
    phi: float,
    theta: np.ndarray,
    rho: float,
    alpha: float,
    *,
    finite_layers: int | None = None,
) -> np.ndarray:
    """Return R(theta) for the order 2 Markov chain."""
    eip = np.exp(1j * phi)
    T = np.array(
        [[rho,            0.5*(1.0-rho)*eip,      0.5*(1.0-rho)*np.conj(eip)],
         [1.0-alpha,      alpha*eip,              0.0                      ],
         [1.0-alpha,      0.0,                    alpha*np.conj(eip)       ]],
        dtype=complex,
    )
    lam, R = np.linalg.eig(T)
    R_inv = np.linalg.inv(R)

    P_S, P_Ep, P_Em = _class_stationary(rho, alpha)
    pi = np.array([P_S, P_Ep, P_Em], dtype=complex)
    ones = np.ones(3, dtype=complex)
    piR  = pi @ R
    Rin1 = R_inv @ ones
    w    = piR * Rin1

    z = (1.0 - P_CLAMP) * np.exp(1j * theta)

    if finite_layers is not None:
        N = int(max(1, finite_layers))
        t = lam[:, None] * z[None, :]
        series = _finite_series_sum(t, N)
        weighted = (w[:, None] * series).sum(axis=0)
        Rtheta = (N + 2.0 * np.real(weighted)) / N
    else:
        # Regularize near poles for the infinite-layer expression.
        frac = (lam[:, None] * z[None, :]) / (1.0 - lam[:, None] * z[None, :])
        S = (w[:, None] * frac).sum(axis=0)
        Rtheta = 1.0 + 2.0 * np.real(S)

    return np.maximum(Rtheta, 0.0)


def _I_inf_markov(
    L: np.ndarray,
    p: float,
    h: int,
    k: int,
    F2: np.ndarray,
    *,
    finite_layers: int | None = None,
) -> np.ndarray:
    rho, alpha = _rho_alpha_from_p(p)
    phi = _slip_phase(h, k)
    theta = _TWO_PI * (L / 3.0)
    Rtheta = _R_from_transfer(
        phi,
        theta,
        rho,
        alpha,
        finite_layers=finite_layers,
    )
    return AREA * F2 * Rtheta


# ----------------------------- base curve builder ---------------------------
def _get_base_curves(
    cif_path: str,
    hk_list=None,
    mx: int | None = None,
    L_step: float = 0.01,
    L_max: float = 10.0,
    two_theta_max: float | None = None,
    lambda_: float = 1.54,
    c_lattice: float | None = None,
):
    """
    Return cached {(h,k): {"L": ..., "F2": ...}} independent of occ and p.

    Conventions align with diffuse_cif_toggle:
      - c_2h is read directly from the CIF (no 3x).
      - Q uses 2π*sqrt(radial + (L/c)^2).
      - Phase uses z/3 in the vertical factor.
    """
    import itertools, math

    if hk_list is None:
        if mx is None:
            raise ValueError("Specify hk_list or mx")
        hk_list = [(h, k) for h, k in itertools.product(range(-mx + 1, mx), repeat=2)]

    key = (
        os.path.abspath(cif_path),
        tuple(hk_list),
        float(L_step),
        float(L_max),
        two_theta_max,
        float(lambda_),
        None if c_lattice is None else float(c_lattice),
    )
    cached = _HT_BASE_CACHE.get(key)
    if cached is not None:
        return cached

    c_2h = _cell_c_from_cif(cif_path)                # no *3
    sites = _sites_from_cif(cif_path)
    if L_step <= 0.0:
        raise ValueError("L_step must be > 0")
    if L_step < 1e-4:
        L_step = 1e-4

    energy_kev = _energy_kev_from_lambda(lambda_)
    out: dict[tuple, dict] = {}

    if two_theta_max is None:
        base_L = np.arange(0.0, L_max + L_step / 2.0, L_step)
        for h, k in hk_list:
            F2 = _F2(h, k, base_L, c_2h, energy_kev, sites)
            out[(h, k)] = {"L": base_L.copy(), "F2": F2}
    else:
        q_max = (4.0 * math.pi / lambda_) * math.sin(math.radians(two_theta_max / 2.0))
        if c_lattice is None or not math.isfinite(float(c_lattice)) or float(c_lattice) <= 0.0:
            c_effective = c_2h
        else:
            c_effective = float(c_lattice)
        for h, k in hk_list:
            const = (4.0 / 3.0) * (h*h + k*k + h*k) / (A_HEX**2)
            l_sq = (q_max / (2.0 * math.pi))**2 - const
            if l_sq <= 0:
                L_vals = np.array([], dtype=float)
                out[(h, k)] = {"L": L_vals, "F2": L_vals}
                continue
            L_max_local = c_effective * math.sqrt(l_sq)
            L_vals = np.arange(0.0, L_max_local + L_step / 2.0, L_step)
            F2 = _F2(h, k, L_vals, c_2h, energy_kev, sites)
            out[(h, k)] = {"L": L_vals.copy(), "F2": F2}

    _HT_BASE_CACHE[key] = out
    return out


# ------------------------------- public routine -----------------------------
def ht_Iinf_dict(
    cif_path: str,
    hk_list=None,                 # explicit list or None
    mx: int | None = None,        # generate -mx+1..mx-1 if hk_list is None
    occ=1.0,                      # occupancy scaling
    p: float = 0.1,
    L_step: float = 0.01,
    L_max: float = 10.0,
    two_theta_max: float | None = None,
    lambda_: float = 1.54,
    c_lattice: float | None = None,
    *,
    finite_stack: bool = True,
    stack_layers: int = 50,
):
    """
    Hendricks–Teller intensities using the Markov transfer model.

    Returns {(h,k): {'L':..., 'I':...}} with F² and C2H conventions identical
    to diffuse_cif_toggle.py. The 'occ' parameter applies a temporary CIF with
    scaled occupancies before computing F².  When ``c_lattice`` is provided the
    two-theta clipping window is expanded or contracted according to that
    effective c-axis length instead of the raw 2H value from the CIF.
    When ``finite_stack`` is ``True`` the per-layer finite-thickness factor for
    ``stack_layers`` layers is applied instead of the infinite-domain limit.
    """
    cif_to_use = cif_path
    cleanup = None
    try:
        if isinstance(occ, (list, tuple)) or float(occ) != 1.0:
            cif_to_use, cleanup = _temp_cif_with_occ(cif_path, occ)

        base = _get_base_curves(
            cif_path=cif_to_use,
            hk_list=hk_list,
            mx=mx,
            L_step=L_step,
            L_max=L_max,
            two_theta_max=two_theta_max,
            lambda_=lambda_,
            c_lattice=c_lattice,
        )

        out = {}
        finite_layers = int(max(1, stack_layers)) if finite_stack else None

        for (h, k), data in base.items():
            L_vals = data["L"]
            F2 = data["F2"]
            I = _I_inf_markov(
                L_vals,
                p,
                h,
                k,
                F2,
                finite_layers=finite_layers,
            )
            out[(h, k)] = {"L": L_vals.copy(), "I": I}
        return out
    finally:
        if cleanup is not None:
            try:
                cleanup(cif_to_use)
            except Exception:
                pass


# ------------------------- array and rod grouping helpers -------------------
def ht_dict_to_arrays(ht_curves):
    """
    Convert the dict output of ht_Iinf_dict to arrays compatible with
    miller_generator style consumers.
    """
    total = sum(len(c["L"]) for c in ht_curves.values())
    miller = np.empty((total, 3), dtype=np.float64)
    intens = np.empty(total, dtype=np.float64)
    degeneracy = np.ones(total, dtype=np.int32)
    details = []

    idx = 0
    for (h, k), curve in ht_curves.items():
        L_vals = curve["L"]
        I_vals = curve["I"]
        n = len(L_vals)

        miller[idx:idx+n, 0] = h
        miller[idx:idx+n, 1] = k
        miller[idx:idx+n, 2] = L_vals
        intens[idx:idx+n] = I_vals

        for L_val, inten in zip(L_vals, I_vals):
            details.append([((h, k, float(L_val)), float(inten))])

        idx += n

    return miller, intens, degeneracy, details


def ht_dict_to_qr_dict(ht_curves):
    """
    Combine HT curves with identical radial index m = h^2 + hk + k^2.
    """
    rods = {}
    for (h, k), curve in ht_curves.items():
        L_vals = np.asarray(curve["L"], dtype=float)
        I_vals = np.asarray(curve["I"], dtype=float)
        m = h*h + h*k + k*k
        if m not in rods:
            rods[m] = {"L": L_vals.copy(), "I": I_vals.copy(), "hk": (h, k), "deg": 1}
            continue

        entry = rods[m]
        if entry["L"].shape != L_vals.shape or not np.allclose(entry["L"], L_vals):
            union = np.union1d(entry["L"], L_vals)
            entry_I = np.interp(union, entry["L"], entry["I"], left=0.0, right=0.0)
            add_I   = np.interp(union, L_vals, I_vals,      left=0.0, right=0.0)
            entry["L"] = union
            entry["I"] = entry_I + add_I
        else:
            entry["I"] += I_vals
        entry["deg"] += 1

    return rods


def qr_dict_to_arrays(qr_dict):
    """Convert a qr_dict from ht_dict_to_qr_dict into arrays."""
    total = sum(len(v["L"]) for v in qr_dict.values())
    miller = np.empty((total, 3), dtype=np.float64)
    intens = np.empty(total, dtype=np.float64)
    degeneracy = np.empty(total, dtype=np.int32)
    details = []

    idx = 0
    for m, data in sorted(qr_dict.items()):
        h, k = data["hk"]
        L_vals = data["L"]
        I_vals = data["I"]
        deg = int(data.get("deg", 1))
        n = len(L_vals)

        miller[idx:idx+n, 0] = h
        miller[idx:idx+n, 1] = k
        miller[idx:idx+n, 2] = L_vals
        intens[idx:idx+n] = I_vals
        degeneracy[idx:idx+n] = deg

        for L_val, inten in zip(L_vals, I_vals):
            details.append([((h, k, float(L_val)), float(inten))])

        idx += n

    return miller, intens, degeneracy, details
