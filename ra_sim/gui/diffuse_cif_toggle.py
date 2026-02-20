"""Algebraic Hendricks-Teller diffuse viewer for the simulation GUI."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from ra_sim.utils.stacking_fault import (
    DEFAULT_PHASE_DELTA_EXPRESSION,
    DEFAULT_PHI_L_DIVISOR,
    _cell_a_c_from_cif,
    _get_base_curves,
    _infer_iodine_z_like_diffuse,
    analytical_ht_intensity_for_pair,
    normalize_phi_l_divisor,
    normalize_phase_delta_expression,
    validate_phase_delta_expression,
)


def _as_triplet(values, fallback):
    """Return a float triplet, padding or truncating as needed."""

    try:
        seq = [float(v) for v in values]
    except Exception:
        seq = list(fallback)

    if len(seq) < 3:
        seq.extend([seq[-1] if seq else float(fallback[-1])] * (3 - len(seq)))
    return [float(seq[0]), float(seq[1]), float(seq[2])]


def _read_lattice_from_cif(cif_path):
    """Return (a, c) from the active CIF file."""

    return _cell_a_c_from_cif(str(cif_path))


def _normalize_weights(weights):
    """Normalize three weights, falling back to equal-weight sum protection."""

    w = np.asarray(_as_triplet(weights, [1.0, 0.0, 0.0]), dtype=float)
    denom = float(np.sum(w))
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return w / denom


def _normalize_phase_delta_expression(value, fallback=DEFAULT_PHASE_DELTA_EXPRESSION):
    """Return a validated phase-delta expression string."""

    normalized = normalize_phase_delta_expression(value, fallback=fallback)
    return validate_phase_delta_expression(normalized)


def _merge_curves(x0, y0, x1, y1):
    """Merge two curves by interpolation onto a shared x-union."""

    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(y0, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    y1 = np.asarray(y1, dtype=float)

    if x1.size == 0 or y1.size == 0:
        return x0, y0
    if x0.size == 0 or y0.size == 0:
        return x1.copy(), y1.copy()

    union = np.union1d(x0, x1)
    y0_u = np.interp(union, x0, y0, left=0.0, right=0.0)
    y1_u = np.interp(union, x1, y1, left=0.0, right=0.0)
    return union, y0_u + y1_u


def _weighted_total(components, weights):
    """Return the weighted sum curve from three component curves."""

    total_x = np.array([], dtype=float)
    total_y = np.array([], dtype=float)
    for weight, (x_vals, y_vals) in zip(weights, components):
        if float(weight) == 0.0:
            continue
        total_x, total_y = _merge_curves(total_x, total_y, x_vals, float(weight) * y_vals)
    return total_x, total_y


def _hex_d_spacing(h, k, l_val, a_axis, c_axis):
    """Return hexagonal d-spacing in Angstrom for one (h, k, l)."""

    inv_d2 = (4.0 / 3.0) * (h * h + h * k + k * k) / (a_axis * a_axis)
    inv_d2 += (l_val * l_val) / (c_axis * c_axis)
    if inv_d2 <= 0.0:
        return float("nan")
    return float(1.0 / np.sqrt(inv_d2))


def _two_theta_deg_from_d(d_angstrom, lambda_angstrom):
    """Return 2theta in degrees from d-spacing and wavelength."""

    if not np.isfinite(d_angstrom) or d_angstrom <= 0.0:
        return float("nan")
    arg = float(lambda_angstrom) / (2.0 * float(d_angstrom))
    if arg < -1.0 or arg > 1.0:
        return float("nan")
    return float(2.0 * np.degrees(np.arcsin(arg)))


def _hex_multiplicity(h, k):
    """Approximate hexagonal reflection multiplicity for exported rows."""

    h = int(h)
    k = int(k)
    if h == 0 and k == 0:
        return 2
    if h == 0 or k == 0 or h == -k:
        return 6
    return 12


def _format_legacy_value(value, width, precision, fmt="f"):
    """Format one numeric column using legacy nan text for invalid values."""

    if not np.isfinite(value):
        return f"{'-nan(ind)':>{width}}"
    return f"{float(value):{width}.{precision}{fmt}}"


def _build_algebraic_ht_export_rows(
    *,
    cif_path,
    occ,
    p_values,
    w_values,
    a_lattice=None,
    c_lattice=None,
    lambda_angstrom=None,
    mx=None,
    two_theta_max=None,
    finite_stack=False,
    stack_layers=50,
    iodine_z=None,
    phase_delta_expression=None,
):
    """Return row dictionaries for the algebraic HT text export."""

    cif_path = str(Path(str(cif_path)).expanduser())
    if not Path(cif_path).is_file():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")
    if lambda_angstrom is None or mx is None:
        raise ValueError("lambda_angstrom and mx are required.")

    p_triplet = _as_triplet(p_values, [0.01, 0.99, 0.5])
    w_triplet = _as_triplet(w_values, [50.0, 50.0, 0.0])
    weights = _normalize_weights(w_triplet)
    hk_limit = int(max(2, int(mx)))

    a_cif = None
    c_cif = None
    try:
        a_cif, c_cif = _read_lattice_from_cif(cif_path)
    except Exception:
        pass

    try:
        a_axis_for_recip = float(a_lattice)
        if not np.isfinite(a_axis_for_recip) or a_axis_for_recip <= 0.0:
            raise ValueError
    except Exception:
        a_axis_for_recip = float(a_cif) if a_cif is not None else 4.557

    try:
        c_axis_for_recip = float(c_lattice)
        if not np.isfinite(c_axis_for_recip) or c_axis_for_recip <= 0.0:
            raise ValueError
    except Exception:
        c_axis_for_recip = float(c_cif) if c_cif is not None else 1.0
    phase_expr = _normalize_phase_delta_expression(phase_delta_expression)
    phi_div = normalize_phi_l_divisor(phi_l_divisor, fallback=DEFAULT_PHI_L_DIVISOR)
    if iodine_z is None:
        iodine_z_value = _infer_iodine_z_like_diffuse(cif_path)
    else:
        try:
            iodine_z_value = float(iodine_z)
        except Exception:
            iodine_z_value = _infer_iodine_z_like_diffuse(cif_path)
    if iodine_z_value is None or not np.isfinite(float(iodine_z_value)):
        iodine_z_value = 0.0
    iodine_z_value = float(np.clip(float(iodine_z_value), 0.0, 1.0))

    base = _get_base_curves(
        cif_path=cif_path,
        mx=hk_limit,
        L_step=0.01,
        L_max=10.0,
        two_theta_max=two_theta_max,
        lambda_=float(lambda_angstrom),
        a_lattice=float(a_axis_for_recip),
        c_lattice=float(c_axis_for_recip),
        occ_factors=occ,
        phase_z_divisor=phi_div,
        iodine_z=iodine_z_value,
        include_f_components=True,
    )

    finite_layers = int(max(1, stack_layers)) if finite_stack else None
    rows = []

    for (h, k), data in sorted(base.items()):
        h = int(h)
        k = int(k)
        if h < 0 or k < 0:
            continue

        L_vals = np.asarray(data.get("L", []), dtype=float)
        F2_vals = np.asarray(data.get("F2", []), dtype=float)
        if L_vals.size == 0 or F2_vals.size == 0:
            continue

        f_real_vals = np.asarray(
            data.get("F_real", np.full_like(L_vals, np.nan)),
            dtype=float,
        )
        f_imag_vals = np.asarray(
            data.get("F_imag", np.full_like(L_vals, np.nan)),
            dtype=float,
        )
        f_abs_vals = np.asarray(
            data.get("F_abs", np.sqrt(np.maximum(F2_vals, 0.0))),
            dtype=float,
        )

        i0_vals = analytical_ht_intensity_for_pair(
            L_vals,
            F2_vals,
            h,
            k,
            p_triplet[0],
            phase_delta_expression=phase_expr,
            phi_l_divisor=phi_div,
            finite_layers=finite_layers,
            f2_only=False,
        )
        i1_vals = analytical_ht_intensity_for_pair(
            L_vals,
            F2_vals,
            h,
            k,
            p_triplet[1],
            phase_delta_expression=phase_expr,
            phi_l_divisor=phi_div,
            finite_layers=finite_layers,
            f2_only=False,
        )
        i2_vals = analytical_ht_intensity_for_pair(
            L_vals,
            F2_vals,
            h,
            k,
            p_triplet[2],
            phase_delta_expression=phase_expr,
            phi_l_divisor=phi_div,
            finite_layers=finite_layers,
            f2_only=False,
        )
        total_i = weights[0] * i0_vals + weights[1] * i1_vals + weights[2] * i2_vals
        multiplicity = _hex_multiplicity(h, k)

        l_int = np.rint(L_vals).astype(int)
        integer_mask = np.isclose(L_vals, l_int, atol=1e-10)
        valid_indices = np.where(integer_mask & (l_int >= 1))[0]

        for idx in valid_indices:
            l_val = float(l_int[idx])
            d_val = _hex_d_spacing(h, k, float(l_val), a_axis_for_recip, c_axis_for_recip)
            two_theta_val = _two_theta_deg_from_d(d_val, float(lambda_angstrom))
            rows.append(
                {
                    "h": h,
                    "k": k,
                    "l": float(l_val),
                    "d": float(d_val),
                    "f_real": float(f_real_vals[idx]),
                    "f_imag": float(f_imag_vals[idx]),
                    "f_abs": float(f_abs_vals[idx]),
                    "two_theta": float(two_theta_val),
                    "intensity": float(total_i[idx]),
                    "multiplicity": int(multiplicity),
                }
            )

    rows.sort(
        key=lambda row: (
            float(row["two_theta"]) if np.isfinite(row["two_theta"]) else float("inf"),
            int(row["h"]),
            int(row["k"]),
            float(row["l"]),
        )
    )
    return rows


def export_algebraic_ht_txt(
    *,
    output_path,
    cif_path,
    occ,
    p_values,
    w_values,
    a_lattice=None,
    c_lattice=None,
    lambda_angstrom=None,
    mx=None,
    two_theta_max=None,
    finite_stack=False,
    stack_layers=50,
    iodine_z=None,
    phase_delta_expression=None,
    phi_l_divisor=DEFAULT_PHI_L_DIVISOR,
):
    """Write algebraic HT values to a legacy fixed-width text table."""

    rows = _build_algebraic_ht_export_rows(
        cif_path=cif_path,
        occ=occ,
        p_values=p_values,
        w_values=w_values,
        a_lattice=a_lattice,
        c_lattice=c_lattice,
        lambda_angstrom=lambda_angstrom,
        mx=mx,
        two_theta_max=two_theta_max,
        finite_stack=finite_stack,
        stack_layers=stack_layers,
        iodine_z=iodine_z,
        phase_delta_expression=phase_delta_expression,
        phi_l_divisor=phi_l_divisor,
    )

    out_path = Path(str(output_path)).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = "   h    k    l      d (Å)      F(real)      F(imag)          |F|         2θ          I    M"
    lines = [header]
    for row in rows:
        line = (
            f"{int(row['h']):4d}"
            f"{int(row['k']):5d}"
            f"{int(round(float(row['l']))):5d}"
            f"{_format_legacy_value(row['d'], 12, 6)}"
            f"{_format_legacy_value(row['f_real'], 13, 6)}"
            f"{_format_legacy_value(row['f_imag'], 13, 6)}"
            f"{_format_legacy_value(row['f_abs'], 12, 4)}"
            f"{_format_legacy_value(row['two_theta'], 11, 5)}"
            f"{_format_legacy_value(row['intensity'], 13, 6, fmt='g')}"
            f"{int(row['multiplicity']):5d}"
        )
        lines.append(line)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return int(len(rows))


def open_diffuse_cif_toggle_algebraic(
    *,
    cif_path,
    occ,
    p_values,
    w_values,
    a_lattice=None,
    c_lattice=None,
    lambda_angstrom=None,
    mx=None,
    two_theta_max=None,
    finite_stack=False,
    stack_layers=50,
    iodine_z=None,
    phase_delta_expression=None,
    phi_l_divisor=DEFAULT_PHI_L_DIVISOR,
):
    """Open an interactive diffuse viewer using algebraic HT only."""

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, CheckButtons, RangeSlider, Slider

    cif_path = str(Path(str(cif_path)).expanduser())
    if not Path(cif_path).is_file():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")
    if lambda_angstrom is None or mx is None:
        raise ValueError("lambda_angstrom and mx are required.")

    p_triplet = _as_triplet(p_values, [0.01, 0.99, 0.5])
    w_triplet = _as_triplet(w_values, [50.0, 50.0, 0.0])
    hk_limit = int(max(2, int(mx)))

    a_cif = None
    c_cif = None
    try:
        a_cif, c_cif = _read_lattice_from_cif(cif_path)
    except Exception:
        pass

    try:
        a_axis_for_recip = float(a_lattice)
        if not np.isfinite(a_axis_for_recip) or a_axis_for_recip <= 0.0:
            raise ValueError
    except Exception:
        a_axis_for_recip = float(a_cif) if a_cif is not None else 4.557

    try:
        c_axis_for_recip = float(c_lattice)
        if not np.isfinite(c_axis_for_recip) or c_axis_for_recip <= 0.0:
            raise ValueError
    except Exception:
        c_axis_for_recip = float(c_cif) if c_cif is not None else 1.0
    phase_expr = _normalize_phase_delta_expression(phase_delta_expression)
    phi_div = normalize_phi_l_divisor(phi_l_divisor, fallback=DEFAULT_PHI_L_DIVISOR)
    if iodine_z is None:
        iodine_z_default = _infer_iodine_z_like_diffuse(cif_path)
    else:
        try:
            iodine_z_default = float(iodine_z)
        except Exception:
            iodine_z_default = _infer_iodine_z_like_diffuse(cif_path)
    if iodine_z_default is None or not np.isfinite(float(iodine_z_default)):
        iodine_z_default = 0.0
    iodine_z_default = float(np.clip(float(iodine_z_default), 0.0, 1.0))

    def _build_curve_data(iodine_z_current):
        base_local = _get_base_curves(
            cif_path=cif_path,
            mx=hk_limit,
            L_step=0.01,
            L_max=10.0,
            # Keep diffuse HT plots on the canonical full L domain [0, 10]
            # regardless of detector two-theta clipping used by simulation.
            two_theta_max=None,
            lambda_=float(lambda_angstrom),
            a_lattice=float(a_axis_for_recip),
            c_lattice=float(c_axis_for_recip),
            occ_factors=occ,
            phase_z_divisor=phi_div,
            iodine_z=iodine_z_current,
        )

        curve_map_local = {}
        for (h, k), data in base_local.items():
            L_vals = np.asarray(data.get("L", []), dtype=float)
            F2_vals = np.asarray(data.get("F2", []), dtype=float)
            if L_vals.size == 0 or F2_vals.size == 0:
                continue
            curve_map_local[(int(h), int(k))] = {"L": L_vals, "F2": F2_vals}

        if not curve_map_local:
            raise ValueError("No diffuse rods available for current CIF/geometry settings.")

        pairs_by_m_local = defaultdict(list)
        for h, k in sorted(curve_map_local):
            m_idx = int(h * h + h * k + k * k)
            pairs_by_m_local[m_idx].append((h, k))
        allowed_m_local = sorted(pairs_by_m_local.keys())

        computed_L_max_local = max(float(np.max(v["L"])) for v in curve_map_local.values())
        return curve_map_local, pairs_by_m_local, allowed_m_local, computed_L_max_local

    curve_map, pairs_by_m, allowed_m, computed_L_max = _build_curve_data(iodine_z_default)
    curve_store = {
        "curve_map": curve_map,
        "pairs_by_m": pairs_by_m,
        "allowed_m": allowed_m,
    }

    # Keep diffuse viewer controls on the canonical L domain [0, 10] even when
    # detector two-theta clipping truncates computed rod data.
    L_max = max(10.0, computed_L_max)
    first_m = int(allowed_m[0])
    first_pair = pairs_by_m[first_m][0]

    state = {
        "mode": "m",
        "m": first_m,
        "h": int(first_pair[0]),
        "k": int(first_pair[1]),
        "p0": float(p_triplet[0]),
        "p1": float(p_triplet[1]),
        "p2": float(p_triplet[2]),
        "w0": float(w_triplet[0]),
        "w1": float(w_triplet[1]),
        "w2": float(w_triplet[2]),
        "L_lo": 0.0,
        "L_hi": float(L_max),
        "qz_axis": False,
        "f2_only": False,
        "show_components": False,
        "finite_N": bool(finite_stack),
        "N_layers": int(max(1, stack_layers)),
    }

    fig, ax = plt.subplots(figsize=(8.2, 6.1))
    fig.subplots_adjust(left=0.25, bottom=0.42, top=0.88)
    ax.set_xlabel("l")
    ax.set_ylabel("I (a.u.)")
    ax.set_yscale("log")

    try:
        fig.canvas.manager.set_window_title(
            f"Diffuse HT (algebraic) - {Path(cif_path).name}"
        )
    except Exception:
        pass

    (line_total,) = ax.plot([], [], lw=2.2, label="Total (algebraic)")
    (line_p0,) = ax.plot([], [], ls="--", label="I_alg(p~0)", visible=False)
    (line_p1,) = ax.plot([], [], ls="--", label="I_alg(p~1)", visible=False)
    (line_p2,) = ax.plot([], [], ls="--", label="I_alg(p)", visible=False)
    title = ax.set_title("")

    cache = {"key": None, "components": None, "elapsed_ms": 0.0}

    def active_pairs():
        pairs_by_m_current = curve_store["pairs_by_m"]
        curve_map_current = curve_store["curve_map"]
        if state["mode"] == "m":
            return list(pairs_by_m_current.get(int(state["m"]), []))
        pair = (int(state["h"]), int(state["k"]))
        return [pair] if pair in curve_map_current else []

    def _compute_components():
        pairs = active_pairs()
        curve_map_current = curve_store["curve_map"]
        finite_layers = int(state["N_layers"]) if state["finite_N"] else None
        start = perf_counter()

        def _component_for_p(p_val):
            x_acc = np.array([], dtype=float)
            y_acc = np.array([], dtype=float)
            for h, k in pairs:
                item = curve_map_current[(h, k)]
                L_vals = item["L"]
                F2_vals = item["F2"]
                I_vals = analytical_ht_intensity_for_pair(
                    L_vals,
                    F2_vals,
                    h,
                    k,
                    p_val,
                    phase_delta_expression=phase_expr,
                    phi_l_divisor=phi_div,
                    finite_layers=finite_layers,
                    f2_only=bool(state["f2_only"]),
                )
                x_acc, y_acc = _merge_curves(x_acc, y_acc, L_vals, I_vals)
            return x_acc, y_acc

        comp0 = _component_for_p(state["p0"])
        comp1 = _component_for_p(state["p1"])
        comp2 = _component_for_p(state["p2"])
        cache["elapsed_ms"] = (perf_counter() - start) * 1e3
        return (comp0, comp1, comp2)

    def _curve_key():
        return (
            state["mode"],
            int(state["m"]),
            int(state["h"]),
            int(state["k"]),
            round(float(state["p0"]), 6),
            round(float(state["p1"]), 6),
            round(float(state["p2"]), 6),
            bool(state["f2_only"]),
            bool(state["finite_N"]),
            int(state["N_layers"]),
        )

    def _line_data(x_vals, y_vals):
        if x_vals.size == 0 or y_vals.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)

        lo = float(min(state["L_lo"], state["L_hi"]))
        hi = float(max(state["L_lo"], state["L_hi"]))
        mask = (x_vals >= lo) & (x_vals <= hi)
        if not np.any(mask):
            return np.array([], dtype=float), np.array([], dtype=float)

        x_sel = x_vals[mask]
        y_sel = np.asarray(y_vals[mask], dtype=float)
        if state["qz_axis"]:
            x_plot = (2.0 * np.pi / c_axis_for_recip) * x_sel
        else:
            x_plot = x_sel

        if ax.get_yscale() == "log":
            y_sel = np.maximum(y_sel, 1e-20)
        return x_plot, y_sel

    def refresh(_event=None):
        key = _curve_key()
        if key != cache["key"]:
            cache["components"] = _compute_components()
            cache["key"] = key

        components = cache["components"] or (
            (np.array([], dtype=float), np.array([], dtype=float)),
            (np.array([], dtype=float), np.array([], dtype=float)),
            (np.array([], dtype=float), np.array([], dtype=float)),
        )
        weights = _normalize_weights([state["w0"], state["w1"], state["w2"]])
        total = _weighted_total(components, weights)

        x_total, y_total = _line_data(*total)
        line_total.set_data(x_total, y_total)

        show_components = bool(state["show_components"])
        for line, component, weight in (
            (line_p0, components[0], weights[0]),
            (line_p1, components[1], weights[1]),
            (line_p2, components[2], weights[2]),
        ):
            visible = show_components and float(weight) > 0.0
            line.set_visible(visible)
            if visible:
                xx, yy = _line_data(*component)
                line.set_data(xx, yy)
            else:
                line.set_data([], [])

        lo = float(min(state["L_lo"], state["L_hi"]))
        hi = float(max(state["L_lo"], state["L_hi"]))
        l_tick_start = int(np.ceil(lo))
        l_tick_stop = int(np.floor(hi))
        if l_tick_stop >= l_tick_start:
            l_ticks = np.arange(l_tick_start, l_tick_stop + 1, dtype=float)
        else:
            l_ticks = np.array([], dtype=float)

        if state["qz_axis"]:
            ax.set_xlabel("qz (A^-1)")
            ax.set_xlim(
                (2.0 * np.pi / c_axis_for_recip) * lo,
                (2.0 * np.pi / c_axis_for_recip) * hi,
            )
            if l_ticks.size:
                ax.set_xticks((2.0 * np.pi / c_axis_for_recip) * l_ticks)
            else:
                ax.set_xticks([])
        else:
            ax.set_xlabel("l")
            ax.set_xlim(lo, hi)
            if l_ticks.size:
                ax.set_xticks(l_ticks)
            else:
                ax.set_xticks([])

        if state["mode"] == "m":
            m_val = int(state["m"])
            q_r = (
                2.0 * np.pi / a_axis_for_recip * np.sqrt(4.0 * m_val / 3.0)
                if m_val > 0
                else 0.0
            )
            caption = f"m={m_val}, Qr={q_r:.3f} A^-1"
        else:
            h_val = int(state["h"])
            k_val = int(state["k"])
            m_val = h_val * h_val + h_val * k_val + k_val * k_val
            q_r = 2.0 * np.pi / a_axis_for_recip * np.sqrt(4.0 * m_val / 3.0)
            caption = f"(h,k)=({h_val},{k_val}), m={m_val}, Qr={q_r:.3f} A^-1"

        title.set_text(
            f"{Path(cif_path).name} | {caption} | dt_alg={cache['elapsed_ms']:.1f} ms"
        )

        handles = [line_total]
        for line in (line_p0, line_p1, line_p2):
            if line.get_visible():
                handles.append(line)
        ax.legend(handles, [h.get_label() for h in handles], loc="upper right")

        ax.relim()
        ax.autoscale_view(scaley=True)
        fig.canvas.draw_idle()

    sliders = []

    def _make_slider(rect, label, vmin, vmax, valinit, valstep, on_change):
        axis = plt.axes(rect)
        slider = Slider(axis, "", vmin, vmax, valinit=valinit, valstep=valstep)
        axis.text(0.5, 1.2, label, transform=axis.transAxes, ha="center")
        slider.on_changed(on_change)
        sliders.append(slider)
        return slider

    range_ax = plt.axes([0.25, 0.05, 0.65, 0.03])
    l_range = RangeSlider(
        range_ax,
        "l range",
        0.0,
        float(L_max),
        valinit=(state["L_lo"], state["L_hi"]),
        valstep=0.01,
    )
    l_range.on_changed(lambda vals: (state.update(L_lo=float(vals[0]), L_hi=float(vals[1])), refresh()))
    sliders.append(l_range)

    rows = [0.34, 0.28, 0.22, 0.16]

    m_slider = _make_slider(
        [0.25, rows[0], 0.65, 0.03],
        "m index",
        int(min(allowed_m)),
        int(max(allowed_m)),
        int(state["m"]),
        allowed_m,
        lambda value: (state.update(m=int(round(value))), refresh()),
    )

    hk_sliders = []

    hk_slider_limit = int(max(1, hk_limit - 1))

    h_slider = _make_slider(
        [0.25, rows[0], 0.30, 0.03],
        "H",
        -hk_slider_limit,
        hk_slider_limit,
        int(state["h"]),
        1,
        lambda value: (state.update(h=int(round(value))), refresh()),
    )
    k_slider = _make_slider(
        [0.60, rows[0], 0.30, 0.03],
        "K",
        -hk_slider_limit,
        hk_slider_limit,
        int(state["k"]),
        1,
        lambda value: (state.update(k=int(round(value))), refresh()),
    )
    hk_sliders.extend([h_slider, k_slider])
    for slider in hk_sliders:
        slider.ax.set_visible(False)

    _make_slider(
        [0.25, rows[1], 0.45, 0.03],
        "p~0",
        0.0,
        0.2,
        state["p0"],
        1e-3,
        lambda value: (state.update(p0=float(value)), refresh()),
    )
    _make_slider(
        [0.72, rows[1], 0.20, 0.03],
        "w(p~0)%",
        0.0,
        100.0,
        state["w0"],
        0.1,
        lambda value: (state.update(w0=float(value)), refresh()),
    )
    _make_slider(
        [0.25, rows[2], 0.45, 0.03],
        "p~1",
        0.8,
        1.0,
        state["p1"],
        1e-3,
        lambda value: (state.update(p1=float(value)), refresh()),
    )
    _make_slider(
        [0.72, rows[2], 0.20, 0.03],
        "w(p~1)%",
        0.0,
        100.0,
        state["w1"],
        0.1,
        lambda value: (state.update(w1=float(value)), refresh()),
    )
    _make_slider(
        [0.25, rows[3], 0.45, 0.03],
        "p",
        0.0,
        1.0,
        state["p2"],
        1e-3,
        lambda value: (state.update(p2=float(value)), refresh()),
    )
    _make_slider(
        [0.72, rows[3], 0.20, 0.03],
        "w(p)%",
        0.0,
        100.0,
        state["w2"],
        0.1,
        lambda value: (state.update(w2=float(value)), refresh()),
    )

    finite_axis = plt.axes([0.92, 0.10, 0.06, 0.03])
    finite_slider = Slider(
        finite_axis,
        "N",
        1,
        1000,
        valinit=int(state["N_layers"]),
        valstep=1,
    )
    finite_axis.text(0.5, 1.2, "Layers", transform=finite_axis.transAxes, ha="center")
    finite_slider.on_changed(lambda value: (state.update(N_layers=int(round(value))), refresh()))

    def _sync_finite_visibility():
        visible = bool(state["finite_N"])
        finite_slider.ax.set_visible(visible)
        finite_slider.label.set_visible(visible)
        finite_slider.valtext.set_visible(visible)

    _sync_finite_visibility()

    scale_button = Button(plt.axes([0.25, 0.01, 0.16, 0.03]), "Toggle scale")
    scale_button.on_clicked(
        lambda _event: (
            ax.set_yscale("linear" if ax.get_yscale() == "log" else "log"),
            refresh(),
        )
    )

    def _toggle_mode(_event):
        state["mode"] = "hk" if state["mode"] == "m" else "m"
        m_slider.ax.set_visible(state["mode"] == "m")
        for slider in hk_sliders:
            slider.ax.set_visible(state["mode"] == "hk")
        refresh()

    mode_button = Button(plt.axes([0.43, 0.01, 0.16, 0.03]), "H/K panel")
    mode_button.on_clicked(_toggle_mode)

    checks_axis = plt.axes([0.05, 0.01, 0.18, 0.18])
    checks = CheckButtons(
        checks_axis,
        ["F^2 only", "qz axis", "Components", "Finite N"],
        [
            bool(state["f2_only"]),
            bool(state["qz_axis"]),
            bool(state["show_components"]),
            bool(state["finite_N"]),
        ],
    )

    def _on_check(label):
        if label == "F^2 only":
            state["f2_only"] = not state["f2_only"]
        elif label == "qz axis":
            state["qz_axis"] = not state["qz_axis"]
        elif label == "Components":
            state["show_components"] = not state["show_components"]
        elif label == "Finite N":
            state["finite_N"] = not state["finite_N"]
            _sync_finite_visibility()
        refresh()

    checks.on_clicked(_on_check)

    # Keep widget instances/callback closures alive for the life of the window.
    fig._ra_sim_diffuse_ui = {  # type: ignore[attr-defined]
        "state": state,
        "refresh": refresh,
        "sliders": sliders,
        "range_slider": l_range,
        "m_slider": m_slider,
        "hk_sliders": hk_sliders,
        "finite_slider": finite_slider,
        "scale_button": scale_button,
        "mode_button": mode_button,
        "checks": checks,
        "cache": cache,
        "curve_store": curve_store,
    }

    refresh()
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None:
        try:
            manager.show()
        except Exception:
            pass
    fig.canvas.draw_idle()
    return fig
