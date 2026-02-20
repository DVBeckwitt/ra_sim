# stacking_fault.py - analytical HT with diffuse-consistent F^2, C2H, and factors

import ast
import os
import re
import numpy as np
import io
from contextlib import redirect_stdout

# Global constants
A_HEX   = 4.557            # Å
P_CLAMP = 1e-6
N_P, A_C = 3, 17.98e-10    # number of sublayers, real-space area (m^2)
AREA    = (2*np.pi)**2 / A_C * N_P
_TWO_PI = 2.0 * np.pi

DEFAULT_PHASE_DELTA_EXPRESSION = "2*pi*((2*h + k)/3)"
DEFAULT_PHI_L_DIVISOR = 1.0

# Cache of base L grids and F2 values keyed by geometry and occupancy mapping.
# Each entry: {(h,k): {"L": array, "F2": array}}
_HT_BASE_CACHE: dict[tuple, dict] = {}
_HT_BASE_CACHE_MAX_ENTRIES = 24

_PHASE_DELTA_EXPR_CACHE: dict[str, object] = {}
_ALLOWED_PHASE_DELTA_FUNCS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "log": np.log,
    "log10": np.log10,
    "abs": np.abs,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "clip": np.clip,
    "where": np.where,
    "real": np.real,
    "imag": np.imag,
    "angle": np.angle,
}
_ALLOWED_PHASE_DELTA_NAMES = {"h", "k", "L", "p", "pi"} | set(
    _ALLOWED_PHASE_DELTA_FUNCS.keys()
)
_ALLOWED_PHASE_DELTA_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.FloorDiv,
    ast.USub,
    ast.UAdd,
)


def normalize_phase_delta_expression(
    expression: str | None,
    *,
    fallback: str = DEFAULT_PHASE_DELTA_EXPRESSION,
) -> str:
    """Return a normalized phase-delta expression string."""

    text = fallback if expression is None else str(expression)
    text = text.strip()
    if not text:
        text = str(fallback).strip()
    return text


def normalize_phi_l_divisor(
    value: float | str | None,
    *,
    fallback: float = DEFAULT_PHI_L_DIVISOR,
) -> float:
    """Return a finite positive divisor used in the HT out-of-plane phase."""

    try:
        fallback_val = float(fallback)
    except (TypeError, ValueError):
        fallback_val = float(DEFAULT_PHI_L_DIVISOR)
    if not np.isfinite(fallback_val) or fallback_val <= 0.0:
        fallback_val = float(DEFAULT_PHI_L_DIVISOR)

    try:
        out = float(fallback_val if value is None else value)
    except (TypeError, ValueError):
        out = fallback_val
    if not np.isfinite(out) or out <= 0.0:
        out = fallback_val
    return float(out)


class _PhaseDeltaExprValidator(ast.NodeVisitor):
    """Validate expression AST against a restricted whitelist."""

    def generic_visit(self, node):
        if not isinstance(node, _ALLOWED_PHASE_DELTA_NODES):
            raise ValueError(
                f"Unsupported expression construct: {type(node).__name__}"
            )
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id not in _ALLOWED_PHASE_DELTA_NAMES:
            raise ValueError(f"Unsupported name '{node.id}' in phase expression")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed")
        if node.func.id not in _ALLOWED_PHASE_DELTA_FUNCS:
            raise ValueError(
                f"Function '{node.func.id}' is not allowed in phase expression"
            )
        if node.keywords:
            raise ValueError("Keyword arguments are not allowed in phase expression")
        self.generic_visit(node)


def _compile_phase_delta_expression(expression: str | None):
    """Compile and cache a validated phase-delta expression."""

    normalized = normalize_phase_delta_expression(expression)
    cached = _PHASE_DELTA_EXPR_CACHE.get(normalized)
    if cached is not None:
        return cached, normalized

    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid phase expression syntax: {exc.msg}") from exc

    _PhaseDeltaExprValidator().visit(tree)
    compiled = compile(tree, "<phase_delta_expression>", "eval")
    _PHASE_DELTA_EXPR_CACHE[normalized] = compiled
    return compiled, normalized


def validate_phase_delta_expression(expression: str | None) -> str:
    """Validate expression and return normalized text."""

    compiled, normalized = _compile_phase_delta_expression(expression)
    L_test = np.asarray([0.0, 0.5], dtype=float)
    namespace = dict(_ALLOWED_PHASE_DELTA_FUNCS)
    namespace.update(
        {"h": 1.0, "k": 0.0, "L": L_test, "p": 0.25, "pi": np.pi}
    )
    try:
        raw = eval(compiled, {"__builtins__": {}}, namespace)
    except Exception as exc:
        raise ValueError(f"Phase expression evaluation failed: {exc}") from exc

    arr = np.asarray(raw, dtype=float)
    try:
        np.broadcast_to(arr, L_test.shape)
    except ValueError as exc:
        raise ValueError(
            "Phase expression must evaluate to a scalar or match L shape"
        ) from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError("Phase expression must produce finite values")
    return normalized


def evaluate_phase_delta_expression(
    expression: str | None,
    h: int,
    k: int,
    L: np.ndarray,
    p: float,
) -> np.ndarray:
    """Evaluate custom delta(h, k, L, p) expression in radians."""

    compiled, normalized = _compile_phase_delta_expression(expression)
    L_vals = np.asarray(L, dtype=float)
    namespace = dict(_ALLOWED_PHASE_DELTA_FUNCS)
    namespace.update(
        {
            "h": float(h),
            "k": float(k),
            "L": L_vals,
            "p": float(p),
            "pi": np.pi,
        }
    )
    try:
        raw = eval(compiled, {"__builtins__": {}}, namespace)
    except Exception as exc:
        raise ValueError(
            f"Failed to evaluate phase expression '{normalized}': {exc}"
        ) from exc

    delta = np.asarray(raw, dtype=float)
    try:
        delta = np.broadcast_to(delta, L_vals.shape).astype(float, copy=False)
    except ValueError as exc:
        raise ValueError(
            "Phase expression must evaluate to a scalar or match L shape"
        ) from exc
    if not np.all(np.isfinite(delta)):
        raise ValueError("Phase expression must produce finite values")
    return delta

# ----------------------------- occupancy helper -----------------------------
def _temp_cif_with_occ(cif_in: str, occ):
    """
    Return path of a temporary CIF in which every `_atom_site_occupancy`
    is multiplied by `occ`.

    Accepts:
      - scalar      -> global factor
      - list/tuple  -> per-site factors; truncated/extended as needed.
    """
    import tempfile, CifFile

    # PyCifRW emits noisy "All blocks output." lines while serializing CIF
    # objects. Suppress that chatter during startup cache generation.
    with redirect_stdout(io.StringIO()):
        cf = CifFile.ReadCif(cif_in)
    block_name = list(cf.keys())[0]
    block = cf[block_name]

    occ_field = block.get('_atom_site_occupancy')
    if occ_field is None:
        labels = block.get('_atom_site_label')
        occ_field = ['1.0'] * (len(labels) if isinstance(labels, list) else 1)
        block['_atom_site_occupancy'] = occ_field

    def _parse_cif_float(raw):
        txt = str(raw).strip()
        try:
            return float(txt)
        except (TypeError, ValueError):
            # CIF numeric fields can include uncertainty suffixes like 0.50(2).
            m = re.match(r"[-+0-9.eE]+", txt)
            if m is None:
                return 1.0
            return float(m.group(0))

    n_sites = len(occ_field)

    # Harmonize occupancy factors to per-site values.
    if isinstance(occ, (list, tuple)):
        factors = [float(v) for v in occ]
        if len(factors) < n_sites:
            fill = factors[-1] if factors else 1.0
            factors.extend([fill] * (n_sites - len(factors)))
        else:
            factors = factors[:n_sites]
    else:
        factors = [float(occ)] * n_sites

    for i, fac in enumerate(factors):
        occ_field[i] = str(_parse_cif_float(occ_field[i]) * fac)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.cif')
    tmp.close()
    try:  # PyCifRW >= 5
        with redirect_stdout(io.StringIO()):
            CifFile.WriteCif(cf, tmp.name)
    except AttributeError:
        with open(tmp.name, 'w') as f:
            with redirect_stdout(io.StringIO()):
                f.write(cf.WriteOut())
    return tmp.name, os.unlink


# -------------------------- low-level physics helpers -----------------------
def _cell_c_from_cif(cif_path: str) -> float:
    """Return the c lattice parameter from cif_path."""
    _a, c = _cell_a_c_from_cif(cif_path)
    return float(c)


def _parse_cif_num(raw) -> float:
    """Parse CIF numeric values, including uncertainty suffixes."""

    if isinstance(raw, (int, float, np.integer, np.floating)):
        return float(raw)
    txt = str(raw).strip()
    m = re.match(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", txt)
    if m is None:
        raise ValueError(f"Unable to parse CIF numeric value: {raw!r}")
    return float(m.group(0))


def _cell_a_c_from_cif(cif_path: str) -> tuple[float, float]:
    """Return (a, c) lattice parameters from a CIF using PyCifRW."""

    import CifFile

    cf = CifFile.ReadCif(cif_path)
    keys = list(cf.keys())
    if not keys:
        raise ValueError(f"No CIF data blocks found in {cif_path}")
    blk = cf[keys[0]]
    a_raw = blk.get("_cell_length_a")
    c_raw = blk.get("_cell_length_c")
    if a_raw is None or c_raw is None:
        raise ValueError(f"_cell_length_a/_cell_length_c not found in {cif_path}")
    return _parse_cif_num(a_raw), _parse_cif_num(c_raw)


def _cif_cache_signature(cif_path: str) -> tuple[str, int | None, int | None]:
    """Return a cache signature that changes when the CIF file content changes."""

    abs_path = os.path.abspath(str(cif_path))
    try:
        st = os.stat(abs_path)
        return abs_path, int(st.st_mtime_ns), int(st.st_size)
    except OSError:
        return abs_path, None, None


def _infer_iodine_z_like_diffuse(cif_path: str, sites=None) -> float | None:
    """Infer iodine z parameter in the same way diffuse_cif_toggle.py does.

    diffuse_cif_toggle.py tries to find a line whose stripped text starts with 'I1'
    and then reads the 5th whitespace-separated token as the z value.
    If that fails, fall back to the first iodine site z from the generated structure.
    Returns None if iodine is not present or no value can be inferred.
    """

    # 1) Text scan for a tokenized 'I1' row.
    try:
        with open(cif_path, "r", encoding="utf-8", errors="ignore") as fp:
            for ln in fp:
                s = ln.strip()
                if s.startswith("I1"):
                    parts = s.split()
                    if len(parts) >= 5:
                        try:
                            return float(parts[4])
                        except Exception:
                            break
    except Exception:
        pass

    # 1b) Robust fallback for CIF loop formatting variations:
    # try _atom_site_label/_atom_site_fract_z columns and match I1-like labels.
    try:
        import CifFile

        cf = CifFile.ReadCif(cif_path)
        keys = list(cf.keys())
        if keys:
            blk = cf[keys[0]]
            labels = blk.get("_atom_site_label")
            z_vals = blk.get("_atom_site_fract_z")
            if labels is not None and z_vals is not None:
                if not isinstance(labels, (list, tuple)):
                    labels = [labels]
                if not isinstance(z_vals, (list, tuple)):
                    z_vals = [z_vals]

                i1_candidate = None
                iodine_candidate = None
                for label_raw, z_raw in zip(labels, z_vals):
                    label = str(label_raw).strip().strip("'\"")
                    if not label:
                        continue
                    z_val = _parse_cif_num(z_raw)
                    if label.startswith("I1"):
                        i1_candidate = z_val
                        break
                    if iodine_candidate is None and label[:1].upper() == "I":
                        iodine_candidate = z_val

                if i1_candidate is not None:
                    return float(i1_candidate)
                if iodine_candidate is not None:
                    return float(iodine_candidate)
    except Exception:
        pass

    # 2) Fallback: use the first iodine z from the generated structure sites.
    try:
        if sites is None:
            sites = _sites_from_cif_with_factors(cif_path, occ_factors=1.0)
        for _x, _y, z, sym, _occ in sites:
            if _element_key(sym) == "I":
                return float(z)
    except Exception:
        pass

    return None


def _sites_from_cif(cif_path: str):
    """Return atomic sites with occupancy from cif_path with symmetry applied."""
    return _sites_from_cif_with_factors(cif_path, occ_factors=1.0)


def _normalize_occ_factors(occ_factors, n_sites: int) -> np.ndarray:
    """Return one occupancy scale factor per generated structure site."""

    n_sites = int(max(1, n_sites))
    if isinstance(occ_factors, (list, tuple, np.ndarray)):
        values = [float(v) for v in occ_factors]
        if not values:
            values = [1.0]
        if len(values) < n_sites:
            values.extend([values[-1]] * (n_sites - len(values)))
        else:
            values = values[:n_sites]
        return np.asarray(values, dtype=np.float64)

    return np.full(n_sites, float(occ_factors), dtype=np.float64)


def _sites_from_cif_with_factors(cif_path: str, occ_factors=1.0):
    """Return atomic sites with per-generated-site occupancy factors applied."""
    import Dans_Diffraction as dif
    xtl = dif.Crystal(str(cif_path))
    xtl.Symmetry.generate_matrices()
    xtl.generate_structure()
    st = xtl.Structure
    n_sites = len(st.u)
    occ_vals = np.ones(n_sites, dtype=np.float64)  # match diffuse_cif_toggle: ignore CIF occupancy
    site_factors = _normalize_occ_factors(occ_factors, n_sites)
    return [
        (
            float(st.u[i]),
            float(st.v[i]),
            float(st.w[i]),
            str(st.type[i]),
            float(occ_vals[i]) * float(site_factors[i]),
        )
        for i in range(n_sites)
    ]


def _energy_kev_from_lambda(lambda_a: float) -> float:
    """Convert wavelength in Å to energy in keV."""
    return (12398.4193 / float(lambda_a)) / 1000.0


# Prefer known ionic scattering-factor labels when available; fall back to
# neutral-element labels for all other species.
IONIC_F0_LABELS = {
    "Pb": "Pb2+",
    "I": "I1-",
}

def _element_key(sym: str) -> str:
    """Map CIF site symbol to a periodic-table element key."""

    text = str(sym).strip()
    m = re.match(r"([A-Za-z]{1,2})", text)
    if m:
        token = m.group(1)
    else:
        letters = "".join(ch for ch in text if ch.isalpha())
        if not letters:
            raise KeyError(f"Unknown element symbol in CIF type '{sym}'")
        token = letters[:2]

    token = token[0].upper() + (token[1].lower() if len(token) > 1 else "")
    return token

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

    element = _element_key(el_sym)
    q = np.asarray(Q, dtype=float).reshape(-1)

    # f0: prefer configured ionic label, otherwise neutral element symbol.
    f0_label = IONIC_F0_LABELS.get(element, element)
    try:
        f0 = xray_scattering_factor([f0_label], q)[:, 0]
    except Exception:
        f0 = xray_scattering_factor([element], q)[:, 0]

    # f' + i f'': use neutral element label when available.
    try:
        f1, f2 = xray_dispersion_corrections([element], energy_kev=[float(energy_kev)])
        f1 = float(f1[0, 0])
        f2 = float(f2[0, 0])
    except Exception:
        f1 = 0.0
        f2 = 0.0

    out = f0 + f1 + 1j * f2
    return out.reshape(Q.shape)


def _Qmag(
    h: int,
    k: int,
    L: np.ndarray,
    c_axis: float,
    a_axis: float = A_HEX,
) -> np.ndarray:
    """
    |Q| for a hexagonal lattice in the active L-axis convention:
    Q = 2π * sqrt( (4/3)*(h^2+k^2+hk)/a^2 + (L^2)/c^2 )
    where L is the dimensionless ℓ coordinate and c matches that coordinate
    system (e.g. c_2h for legacy 2H L, or c_lattice for 3c-scaled L).
    """
    inv_d2 = (4.0/3.0)*(h*h + k*k + h*k)/(a_axis**2) + (L**2)/(c_axis**2)
    return 2.0*np.pi*np.sqrt(inv_d2)


def _F_complex(
    h: int,
    k: int,
    L: np.ndarray,
    c_axis: float,
    energy_kev: float,
    sites,
    *,
    a_axis: float = A_HEX,
    phase_z_divisor: float = 1.0,
) -> np.ndarray:
    """Return complex F for the supplied reciprocal-space coordinates."""

    Q = _Qmag(h, k, L, c_axis, a_axis=a_axis)
    F = np.zeros_like(Q, dtype=complex)
    z_div = float(phase_z_divisor) if float(phase_z_divisor) != 0.0 else 1.0
    for x, y, z, sym, occ in sites:
        ff = f_comp(sym, Q, energy_kev)
        phase_xy = np.exp(1j * _TWO_PI * (h * x + k * y))
        phase_z = np.exp(1j * _TWO_PI * (L * (float(z) / z_div)))
        F += float(occ) * ff * phase_xy * phase_z
    return F


def _F2(
    h: int,
    k: int,
    L: np.ndarray,
    c_axis: float,
    energy_kev: float,
    sites,
    *,
    a_axis: float = A_HEX,
    phase_z_divisor: float = 1.0,
    iodine_single_plane: bool = True,
    iodine_z: float | None = None,
) -> np.ndarray:
    """Return |F|^2.

    By default this matches the diffuse_cif_toggle iodine treatment:
    iodine contributions are collapsed onto one shared z-plane (``iodine_z`` or
    the first iodine z found in ``sites``). Set ``iodine_single_plane=False`` to
    recover the generic per-site z-phase sum.
    """

    if not bool(iodine_single_plane):
        return np.abs(
            _F_complex(
                h,
                k,
                L,
                c_axis,
                energy_kev,
                sites,
                a_axis=a_axis,
                phase_z_divisor=phase_z_divisor,
            )
        ) ** 2

    L_vals = np.asarray(L, dtype=float)
    Q = _Qmag(h, k, L_vals, c_axis, a_axis=a_axis)
    fixed = np.zeros_like(Q, dtype=complex)
    iodine_factor = np.zeros_like(Q, dtype=complex)
    z_div = float(phase_z_divisor) if float(phase_z_divisor) != 0.0 else 1.0
    iodine_z_eff = None if iodine_z is None else float(iodine_z)

    for x, y, z, sym, occ in sites:
        ff = f_comp(sym, Q, energy_kev)
        phase_xy = np.exp(1j * _TWO_PI * (h * x + k * y))
        if _element_key(sym) == "I":
            iodine_factor += float(occ) * ff * phase_xy
            if iodine_z_eff is None:
                iodine_z_eff = float(z)
        else:
            phase_z = np.exp(1j * _TWO_PI * (L_vals * (float(z) / z_div)))
            fixed += float(occ) * ff * phase_xy * phase_z

    if iodine_z_eff is None:
        return np.abs(fixed) ** 2

    phase_z_I = np.exp(1j * _TWO_PI * (L_vals * (iodine_z_eff / z_div)))
    return np.abs(fixed + iodine_factor * phase_z_I) ** 2


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


def _finite_R_from_t(t: np.ndarray, n_layers: int) -> np.ndarray:
    """Finite-domain correction from the legacy analytical HT expression."""

    t = np.asarray(t, dtype=complex)
    n = int(max(1, n_layers))
    if n == 1:
        return np.ones_like(np.real(t), dtype=float)

    one = 1.0 + 0.0j
    mask = np.isclose(t, one)
    out = np.empty_like(np.real(t), dtype=float)

    if np.any(~mask):
        t_nm = t[~mask]
        denom = one - t_nm
        s1 = t_nm * (1 - t_nm ** (n - 1)) / denom
        s2 = t_nm * (1 - n * t_nm ** (n - 1) + (n - 1) * t_nm ** n) / (denom ** 2)
        series = n * s1 - s2
        out[~mask] = (n + 2.0 * np.real(series)) / n

    if np.any(mask):
        out[mask] = float(n)

    return np.maximum(out, 0.0)


def analytical_ht_intensity_for_pair(
    L_vals,
    F2_vals,
    h: int,
    k: int,
    p: float,
    *,
    phase_delta_expression: str | None = None,
    phi_l_divisor: float = DEFAULT_PHI_L_DIVISOR,
    finite_layers: int | None = None,
    f2_only: bool = False,
) -> np.ndarray:
    """Return analytical HT intensity for one (h, k) rod.

    Matches the algebraic HT implementation used in diffuse_cif_toggle.py:
      - p is flipped (p -> 1 - p) before forming z
      - R uses the same infinite- or finite-layer closed forms
      - no extra clipping/regularization beyond P_CLAMP
    """

    F2_vals = np.asarray(F2_vals, dtype=float)
    if f2_only:
        return F2_vals

    L_vals = np.asarray(L_vals, dtype=float)

    # Match diffuse_cif_toggle.py convention: flip p -> 1 - p
    p_flipped = 1.0 - float(p)

    # Allow custom delta(h,k,L,p) but pass the *flipped* p so delta uses the same p
    # that appears in z, consistent with the diffuse algebra.
    delta = evaluate_phase_delta_expression(
        phase_delta_expression,
        h,
        k,
        L_vals,
        p_flipped,
    )

    z = (1.0 - p_flipped) + p_flipped * np.exp(1j * delta)
    f_val = np.minimum(np.abs(z), 1.0 - float(P_CLAMP))
    psi = np.angle(z)
    phi_div = normalize_phi_l_divisor(phi_l_divisor)
    phi = delta + _TWO_PI * L_vals * (1.0 / phi_div)

    if finite_layers is None:
        R = (1.0 - f_val**2) / (1.0 + f_val**2 - 2.0 * f_val * np.cos(phi - psi))
    else:
        t = f_val * np.exp(1j * (phi - psi))
        R = _finite_R_from_t(t, int(max(1, finite_layers)))

    return float(AREA) * F2_vals * R


# Legacy Markov transfer helpers are kept for backwards compatibility.
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
    phi_l_divisor: float = DEFAULT_PHI_L_DIVISOR,
    finite_layers: int | None = None,
) -> np.ndarray:
    rho, alpha = _rho_alpha_from_p(p)
    phi = _slip_phase(h, k)
    theta = _TWO_PI * (L / normalize_phi_l_divisor(phi_l_divisor))
    Rtheta = _R_from_transfer(
        phi,
        theta,
        rho,
        alpha,
        finite_layers=finite_layers,
    )
    return AREA * F2 * Rtheta


# ----------------------------- base curve builder ---------------------------
def _hk_radial_index(h: int, k: int) -> int:
    """Hexagonal in-plane radial class index m = h^2 + hk + k^2."""

    return int(h * h + h * k + k * k)


def _normalize_complex_phase_vector(
    values: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Normalize complex coefficients by a stable anchor phase.

    This turns globally phase-shifted coefficient vectors into the same canonical
    representation, so we can safely reuse |F|^2 when a reflection differs only
    by an overall complex phase factor.
    """

    arr = np.asarray(values, dtype=np.complex128).reshape(-1)
    if arr.size == 0:
        return arr

    nz = np.flatnonzero(np.abs(arr) > float(eps))
    if nz.size == 0:
        return np.zeros_like(arr, dtype=np.complex128)

    anchor = arr[int(nz[0])]
    if abs(anchor) <= float(eps):
        return arr.copy()
    return arr / anchor


def _complex_phase_signature_key(values: np.ndarray, *, digits: int = 12) -> bytes:
    """Quantized byte-key for complex vectors used in reflection dedup lookup."""

    arr = np.asarray(values, dtype=np.complex128).reshape(-1)
    packed = np.empty((arr.size, 2), dtype=np.float64)
    packed[:, 0] = np.round(np.real(arr), int(digits))
    packed[:, 1] = np.round(np.imag(arr), int(digits))
    return packed.tobytes()


def _get_base_curves(
    cif_path: str,
    hk_list=None,
    mx: int | None = None,
    L_step: float = 0.01,
    L_max: float = 10.0,
    two_theta_max: float | None = None,
    lambda_: float = 1.5406,
    a_lattice: float | None = None,
    c_lattice: float | None = None,
    occ_factors=1.0,
    phase_z_divisor: float = DEFAULT_PHI_L_DIVISOR,
    iodine_z: float | None = None,
    iodine_single_plane: bool = True,
    include_f_components: bool = False,
):
    """
    Return cached {(h,k): {"L": ..., "F2": ...}} for a given occupancy mapping.

    Conventions align with diffuse_cif_toggle:
      - a/c used for |Q| (and optional two-theta clipping) are a_lattice/c_lattice when
        provided, otherwise the CIF a/c (no 3x scaling).
      - With iodine_single_plane=True (default), iodine contributions are collapsed onto one
        shared z-plane (iodine_z or the first iodine z found in the CIF), matching diffuse_cif_toggle.
      - Phase uses z/phase_z_divisor in the vertical factor.
    """
    import itertools
    import math

    if hk_list is None:
        if mx is None:
            raise ValueError("Specify hk_list or mx")
        hk_list = [(h, k) for h, k in itertools.product(range(-mx + 1, mx), repeat=2)]
    hk_list = [(int(h), int(k)) for (h, k) in hk_list]

    if isinstance(occ_factors, (list, tuple, np.ndarray)):
        occ_key = tuple(
            np.round(
                np.asarray(occ_factors, dtype=np.float64).reshape(-1),
                12,
            ).tolist()
        )
    else:
        occ_key = float(occ_factors)

    cif_signature = _cif_cache_signature(cif_path)
    key = (
        cif_signature,
        tuple(hk_list),
        float(L_step),
        float(L_max),
        two_theta_max,
        float(lambda_),
        None if a_lattice is None else float(a_lattice),
        None if c_lattice is None else float(c_lattice),
        occ_key,
        float(phase_z_divisor),
        None if iodine_z is None else float(iodine_z),
        bool(iodine_single_plane),
        bool(include_f_components),
    )
    cached = _HT_BASE_CACHE.get(key)
    if cached is not None:
        return cached

    _a_cif, c_cif = _cell_a_c_from_cif(cif_path)
    # Match diffuse_cif_toggle.py: ignore CIF occupancy and any external occupancy scaling in F².
    sites = _sites_from_cif_with_factors(cif_path, occ_factors=1.0)
    if L_step <= 0.0:
        raise ValueError("L_step must be > 0")
    if L_step < 1e-4:
        L_step = 1e-4

    if a_lattice is None or not math.isfinite(float(a_lattice)) or float(a_lattice) <= 0.0:
        # Match diffuse_cif_toggle.py: default to the module's A_HEX constant unless overridden.
        a_effective = float(A_HEX)
    else:
        a_effective = float(a_lattice)

    if c_lattice is None or not math.isfinite(float(c_lattice)) or float(c_lattice) <= 0.0:
        c_effective = c_cif
    else:
        c_effective = float(c_lattice)

    # Keep legacy diffuse behavior:
    # - two-theta clipping uses the "active" lattice values (possibly overridden)
    # - |Q| used inside F(L) stays on the legacy diffuse reference axes
    #   (A_HEX and CIF c), independent of the active windowing axes.
    a_window_factor = float(a_effective)
    c_window_factor = float(c_effective)
    a_form_factor = float(A_HEX)
    c_form_factor = float(c_cif)
    energy_kev = _energy_kev_from_lambda(lambda_)

    z_div = float(phase_z_divisor)
    if (not np.isfinite(z_div)) or abs(z_div) < 1e-14:
        z_div = 1.0

    site_count = len(sites)
    site_x = np.empty(site_count, dtype=np.float64)
    site_y = np.empty(site_count, dtype=np.float64)
    site_z = np.empty(site_count, dtype=np.float64)
    site_element: list[str] = []
    for idx, (x, y, z, sym, _occ) in enumerate(sites):
        site_x[idx] = float(x)
        site_y[idx] = float(y)
        site_z[idx] = float(z)
        site_element.append(_element_key(sym))
    site_is_iodine = np.asarray([el == "I" for el in site_element], dtype=bool)

    # Match diffuse_cif_toggle.py F² behaviour: treat iodine as a single plane at
    # z = iodine_z (default inferred from the CIF) by removing iodine z from the
    # per-site phase and applying one shared exp(2π i L * (iodine_z/phase_z_divisor)).
    iodine_z_eff = None
    iodine_active = False
    if bool(iodine_single_plane) and bool(np.any(site_is_iodine)):
        iodine_z_eff = iodine_z
        if iodine_z_eff is None:
            iodine_z_eff = _infer_iodine_z_like_diffuse(cif_path, sites=sites)
        if iodine_z_eff is not None:
            iodine_active = True

    # Preserve site-encounter order for deterministic behavior.
    unique_elements = list(dict.fromkeys(site_element))
    if iodine_active and "I" not in unique_elements:
        unique_elements.append("I")

    base_L = None
    q_max = None
    if two_theta_max is None:
        base_L = np.arange(0.0, L_max + L_step / 2.0, L_step, dtype=float)
    else:
        q_max = (4.0 * math.pi / lambda_) * math.sin(math.radians(two_theta_max / 2.0))

    L_grid_cache: dict[int, np.ndarray] = {}

    def _L_for_m(m: int) -> np.ndarray:
        cached_L = L_grid_cache.get(m)
        if cached_L is not None:
            return cached_L

        if base_L is not None:
            L_vals = base_L
        else:
            const = (4.0 / 3.0) * float(m) / (a_window_factor**2)
            l_sq = (float(q_max) / (2.0 * math.pi))**2 - const
            if l_sq <= 0.0:
                L_vals = np.array([], dtype=float)
            else:
                L_max_local = c_window_factor * math.sqrt(l_sq)
                L_vals = np.arange(0.0, L_max_local + L_step / 2.0, L_step, dtype=float)

        L_grid_cache[m] = L_vals
        return L_vals

    # Per-radial-class cache:
    # - form factors ff(Q) by element
    # - z-phase factors by site
    # - optional iodine z-phase
    # - reflection-level dedup cache by normalized in-plane phase signature
    m_state_cache: dict[int, dict] = {}

    def _state_for_m(m: int) -> dict:
        state = m_state_cache.get(m)
        if state is not None:
            return state

        L_vals = _L_for_m(m)
        state = {
            "L": L_vals,
            "ff_by_element": {},
            "phase_z_by_site": [],
            "phase_z_iodine": None,
            "signature_cache": {},
        }
        if L_vals.size > 0:
            q_term = (4.0 / 3.0) * float(m) / (a_form_factor**2)
            Q_vals = 2.0 * np.pi * np.sqrt(q_term + (L_vals * L_vals) / (c_form_factor**2))

            ff_by_element = {}
            for elem in unique_elements:
                ff_by_element[elem] = np.asarray(
                    f_comp(elem, Q_vals, energy_kev),
                    dtype=np.complex128,
                )
            state["ff_by_element"] = ff_by_element

            phase_z_by_site = []
            phase_z_cache: dict[float, np.ndarray] = {}
            for z_val in site_z:
                z_key = round(float(z_val), 12)
                phase_z = phase_z_cache.get(z_key)
                if phase_z is None:
                    phase_z = np.exp(1j * _TWO_PI * (L_vals * (float(z_val) / z_div)))
                    phase_z_cache[z_key] = phase_z
                phase_z_by_site.append(phase_z)
            state["phase_z_by_site"] = phase_z_by_site

            if iodine_active:
                state["phase_z_iodine"] = np.exp(
                    1j * _TWO_PI * (L_vals * (float(iodine_z_eff) / z_div))
                )

        m_state_cache[m] = state
        return state

    out: dict[tuple, dict] = {}
    for h, k in hk_list:
        m = _hk_radial_index(h, k)
        state = _state_for_m(m)
        L_vals = state["L"]

        if L_vals.size == 0:
            L_empty = np.array([], dtype=float)
            out[(h, k)] = {"L": L_empty, "F2": L_empty}
            continue

        phase_xy = np.exp(1j * _TWO_PI * (float(h) * site_x + float(k) * site_y))
        coeff_norm = _normalize_complex_phase_vector(phase_xy)
        signature_key = _complex_phase_signature_key(coeff_norm)

        reused_curve = None
        candidates = state["signature_cache"].get(signature_key)
        if candidates is not None:
            for prev_norm, prev_curve in candidates:
                if np.allclose(coeff_norm, prev_norm, rtol=0.0, atol=1e-11):
                    reused_curve = prev_curve
                    break

        if reused_curve is None:
            F = np.zeros(L_vals.shape, dtype=np.complex128)
            ff_by_element = state["ff_by_element"]
            phase_z_by_site = state["phase_z_by_site"]

            if iodine_active and state["phase_z_iodine"] is not None:
                iodine_coeff = 0.0 + 0.0j
                for idx in range(site_count):
                    if site_is_iodine[idx]:
                        iodine_coeff += phase_xy[idx]
                    else:
                        el = site_element[idx]
                        F += ff_by_element[el] * phase_xy[idx] * phase_z_by_site[idx]
                F += ff_by_element["I"] * iodine_coeff * state["phase_z_iodine"]
            else:
                for idx in range(site_count):
                    el = site_element[idx]
                    F += ff_by_element[el] * phase_xy[idx] * phase_z_by_site[idx]

            curve = {
                "F2": np.abs(F) ** 2,
            }
            if include_f_components:
                curve["F_real"] = np.real(F)
                curve["F_imag"] = np.imag(F)
                curve["F_abs"] = np.abs(F)

            entries = state["signature_cache"].setdefault(signature_key, [])
            entries.append((coeff_norm.copy(), curve))
            reused_curve = curve

        out_entry = {
            "L": np.asarray(L_vals, dtype=float).copy(),
            "F2": np.asarray(reused_curve["F2"], dtype=float).copy(),
        }
        if include_f_components:
            out_entry["F_real"] = np.asarray(reused_curve["F_real"], dtype=float).copy()
            out_entry["F_imag"] = np.asarray(reused_curve["F_imag"], dtype=float).copy()
            out_entry["F_abs"] = np.asarray(reused_curve["F_abs"], dtype=float).copy()
        out[(h, k)] = out_entry

    _HT_BASE_CACHE[key] = out
    # Bound cache growth because occupancy sliders can produce many unique keys.
    while len(_HT_BASE_CACHE) > _HT_BASE_CACHE_MAX_ENTRIES:
        try:
            _HT_BASE_CACHE.pop(next(iter(_HT_BASE_CACHE)))
        except StopIteration:
            break
    return out


# ------------------------------- public routine -----------------------------
def ht_Iinf_dict(
    cif_path: str,
    hk_list=None,                 # explicit list or None
    mx: int | None = None,        # generate -mx+1..mx-1 if hk_list is None
    occ=1.0,                      # occupancy scaling per generated structure site
    p: float = 0.1,
    L_step: float = 0.01,
    L_max: float = 10.0,
    two_theta_max: float | None = None,
    lambda_: float = 1.5406,
    a_lattice: float | None = None,
    c_lattice: float | None = None,
    phase_z_divisor: float | None = None,
    iodine_z: float | None = None,
    phase_delta_expression: str | None = None,
    phi_l_divisor: float = DEFAULT_PHI_L_DIVISOR,
    *,
    finite_stack: bool = False,
    stack_layers: int = 50,
):
    """
    Hendricks–Teller intensities using the analytical HT expression.

    Returns {(h,k): {'L':..., 'I':...}} with F² and C2H conventions identical
    to diffuse_cif_toggle.py. The 'occ' parameter is applied per generated
    structure site (symmetry-expanded), so Pb/I1/I2 can be controlled
    independently when the structure contains those three positions.
    When ``c_lattice`` is provided it defines the active L-axis convention:
    both the two-theta clipping window and the Qz scaling inside F² use that
    effective c-axis length instead of the raw 2H value from the CIF.
    ``a_lattice`` optionally overrides the active in-plane lattice constant
    used for |Q| and two-theta clipping.
    ``phase_z_divisor`` controls vertical phase scaling in F². When omitted it
    defaults to ``phi_l_divisor`` so the F² vertical phase and HT correlation
    phase use the same L-axis convention.
    ``iodine_z`` optionally pins the iodine z-plane used in F². When ``None``,
    the value is inferred from the CIF in the same way as diffuse_cif_toggle.
    ``phase_delta_expression`` defines delta(h, k, L, p) in radians for the
    analytical HT correlation term. The default expression is
    ``2*pi*((2*h + k)/3)``.
    ``phi_l_divisor`` sets the out-of-plane term in the HT phase as
    ``phi = delta + 2*pi*L/phi_l_divisor``.
    When ``finite_stack`` is ``True`` the per-layer finite-thickness factor for
    ``stack_layers`` layers is applied instead of the infinite-domain limit.
    """
    phase_expr = validate_phase_delta_expression(phase_delta_expression)
    phi_div = normalize_phi_l_divisor(phi_l_divisor)
    if phase_z_divisor is None:
        phase_z_div = phi_div
    else:
        phase_z_div = normalize_phi_l_divisor(phase_z_divisor, fallback=phi_div)

    base = _get_base_curves(
        cif_path=cif_path,
        hk_list=hk_list,
        mx=mx,
        L_step=L_step,
        L_max=L_max,
        two_theta_max=two_theta_max,
        lambda_=lambda_,
        a_lattice=a_lattice,
        c_lattice=c_lattice,
        occ_factors=occ,
        phase_z_divisor=phase_z_div,
        iodine_z=iodine_z,
    )

    out = {}
    finite_layers = int(max(1, stack_layers)) if finite_stack else None

    for (h, k), data in base.items():
        L_vals = data["L"]
        F2 = data["F2"]
        I = analytical_ht_intensity_for_pair(
            L_vals,
            F2,
            h,
            k,
            p,
            phase_delta_expression=phase_expr,
            phi_l_divisor=phi_div,
            finite_layers=finite_layers,
        )
        out[(h, k)] = {"L": L_vals.copy(), "I": I}
    return out


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
