"""Algebraic Hendricks-Teller diffuse viewer for the simulation GUI."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter
import re

import numpy as np

from ra_sim.utils.stacking_fault import AREA, P_CLAMP, _get_base_curves


def _as_triplet(values, fallback):
    """Return a float triplet, padding or truncating as needed."""

    try:
        seq = [float(v) for v in values]
    except Exception:
        seq = list(fallback)

    if len(seq) < 3:
        seq.extend([seq[-1] if seq else float(fallback[-1])] * (3 - len(seq)))
    return [float(seq[0]), float(seq[1]), float(seq[2])]


def _parse_cif_num(raw_value):
    """Parse a CIF numeric field, handling uncertainty suffixes."""

    if isinstance(raw_value, (int, float, np.integer, np.floating)):
        return float(raw_value)
    text = str(raw_value).strip()
    match = re.match(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    if match is None:
        raise ValueError(f"Unable to parse CIF numeric value: {raw_value!r}")
    return float(match.group(0))


def _read_lattice_from_cif(cif_path):
    """Return (a, c) from the active CIF file."""

    import CifFile

    cf = CifFile.ReadCif(str(cif_path))
    keys = list(cf.keys())
    if not keys:
        raise ValueError(f"No CIF data blocks found in {cif_path}")
    block = cf[keys[0]]
    a_raw = block.get("_cell_length_a")
    c_raw = block.get("_cell_length_c")
    if a_raw is None or c_raw is None:
        raise ValueError("CIF is missing _cell_length_a/_cell_length_c")
    return _parse_cif_num(a_raw), _parse_cif_num(c_raw)


def _normalize_weights(weights):
    """Normalize three weights, falling back to equal-weight sum protection."""

    w = np.asarray(_as_triplet(weights, [1.0, 0.0, 0.0]), dtype=float)
    denom = float(np.sum(w))
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return w / denom


def _finite_R_from_t(t, n_layers):
    """Finite-domain correction from the legacy algebraic HT expression."""

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


def _algebraic_I_for_pair(
    L_vals,
    F2_vals,
    h,
    k,
    p,
    *,
    finite_layers=None,
    f2_only=False,
):
    """Return algebraic HT intensity for one (h,k) rod."""

    if f2_only:
        return np.asarray(F2_vals, dtype=float)

    L_vals = np.asarray(L_vals, dtype=float)
    F2_vals = np.asarray(F2_vals, dtype=float)

    # Match the legacy viewer's p inversion convention.
    p_flipped = 1.0 - float(np.clip(p, 0.0, 1.0))
    delta = 2.0 * np.pi * ((2.0 * float(h) + float(k)) / 3.0)
    z = (1.0 - p_flipped) + p_flipped * np.exp(1j * delta)
    f_val = min(float(np.abs(z)), 1.0 - float(P_CLAMP))
    psi = float(np.angle(z))
    phi = delta + 2.0 * np.pi * L_vals * (1.0 / 3.0)

    if finite_layers is None:
        denom = 1.0 + f_val * f_val - 2.0 * f_val * np.cos(phi - psi)
        denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
        R = (1.0 - f_val * f_val) / denom
    else:
        t = f_val * np.exp(1j * (phi - psi))
        R = _finite_R_from_t(t, int(max(1, finite_layers)))

    return np.maximum(0.0, float(AREA) * F2_vals * R)


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


def open_diffuse_cif_toggle_algebraic(
    *,
    cif_path,
    occ,
    p_values,
    w_values,
    c_lattice,
    lambda_angstrom,
    mx,
    two_theta_max=None,
    finite_stack=False,
    stack_layers=50,
):
    """Open an interactive diffuse viewer using algebraic HT only."""

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, CheckButtons, RangeSlider, Slider

    cif_path = str(Path(str(cif_path)).expanduser())
    if not Path(cif_path).is_file():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    p_triplet = _as_triplet(p_values, [0.01, 0.99, 0.5])
    w_triplet = _as_triplet(w_values, [50.0, 50.0, 0.0])
    hk_limit = int(max(2, int(mx)))

    try:
        a_axis_for_recip, c_axis_for_recip = _read_lattice_from_cif(cif_path)
    except Exception:
        a_axis_for_recip = 4.557
        c_axis_for_recip = float(c_lattice) if float(c_lattice) > 0.0 else 1.0

    base = _get_base_curves(
        cif_path=cif_path,
        mx=hk_limit,
        L_step=0.01,
        L_max=10.0,
        two_theta_max=two_theta_max,
        lambda_=float(lambda_angstrom),
        # Use reciprocal scaling from the active CIF so l<->qz conversion
        # is tied to whichever structure is currently loaded.
        c_lattice=float(c_axis_for_recip),
        occ_factors=occ,
    )

    curve_map = {}
    for (h, k), data in base.items():
        L_vals = np.asarray(data.get("L", []), dtype=float)
        F2_vals = np.asarray(data.get("F2", []), dtype=float)
        if L_vals.size == 0 or F2_vals.size == 0:
            continue
        curve_map[(int(h), int(k))] = {"L": L_vals, "F2": F2_vals}

    if not curve_map:
        raise ValueError("No diffuse rods available for current CIF/geometry settings.")

    pairs_by_m = defaultdict(list)
    for h, k in sorted(curve_map):
        m_idx = int(h * h + h * k + k * k)
        pairs_by_m[m_idx].append((h, k))
    allowed_m = sorted(pairs_by_m.keys())

    L_max = max(float(np.max(v["L"])) for v in curve_map.values())
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
        if state["mode"] == "m":
            return list(pairs_by_m.get(int(state["m"]), []))
        pair = (int(state["h"]), int(state["k"]))
        return [pair] if pair in curve_map else []

    def _compute_components():
        pairs = active_pairs()
        finite_layers = int(state["N_layers"]) if state["finite_N"] else None
        start = perf_counter()

        def _component_for_p(p_val):
            x_acc = np.array([], dtype=float)
            y_acc = np.array([], dtype=float)
            for h, k in pairs:
                item = curve_map[(h, k)]
                L_vals = item["L"]
                F2_vals = item["F2"]
                I_vals = _algebraic_I_for_pair(
                    L_vals,
                    F2_vals,
                    h,
                    k,
                    p_val,
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
