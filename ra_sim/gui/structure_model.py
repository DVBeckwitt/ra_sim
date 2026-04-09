"""Structure-model and diffuse-HT helpers for the GUI runtime."""

from __future__ import annotations

import io
import math
import os
import re
import tempfile
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import CifFile
import Dans_Diffraction as dif
import numpy as np

from ra_sim.gui import controllers as gui_controllers
from ra_sim.utils.stacking_fault import (
    _infer_iodine_z_like_diffuse,
    ht_Iinf_dict,
    ht_dict_to_arrays,
    ht_dict_to_qr_dict,
)


def _ensure_numeric_vector(values, fallback, length):
    try:
        target_len = int(length)
    except (TypeError, ValueError):
        target_len = 1
    target_len = max(1, target_len)

    out = []
    if isinstance(values, (list, tuple, np.ndarray)):
        for raw in values:
            try:
                out.append(float(raw))
            except (TypeError, ValueError):
                continue

    if not out:
        if isinstance(fallback, (list, tuple, np.ndarray)):
            for raw in fallback:
                try:
                    out.append(float(raw))
                except (TypeError, ValueError):
                    continue
        else:
            try:
                out = [float(fallback)]
            except (TypeError, ValueError):
                out = [1.0]

    if not out:
        out = [1.0]

    if len(out) < target_len:
        out.extend([out[-1]] * (target_len - len(out)))
    elif len(out) > target_len:
        out = out[:target_len]
    return out


def normalize_occupancy_label(raw_label, fallback_idx):
    text = str(raw_label).strip().strip("'\"")
    if text:
        return text
    return f"site_{int(fallback_idx) + 1}"


def extract_occupancy_site_metadata(cif_block, cif_path):
    """Return (unique labels, expanded-site -> unique-label index mapping)."""

    try:
        xtl = dif.Crystal(str(cif_path))
        xtl.Symmetry.generate_matrices()
        xtl.generate_structure()
        st = xtl.Structure
        n_sites = len(st.u)

        labels_src = getattr(st, "label", None)
        if labels_src is None or len(labels_src) != n_sites:
            labels_src = getattr(st, "type", None)
        if labels_src is None or len(labels_src) != n_sites:
            labels_src = [f"site_{idx + 1}" for idx in range(n_sites)]

        unique_labels = []
        label_to_idx = {}
        expanded_to_unique = []

        for idx in range(n_sites):
            label = normalize_occupancy_label(labels_src[idx], idx)
            mapped = label_to_idx.get(label)
            if mapped is None:
                mapped = len(unique_labels)
                unique_labels.append(label)
                label_to_idx[label] = mapped
            expanded_to_unique.append(mapped)

        if unique_labels:
            return unique_labels, expanded_to_unique
    except Exception:
        pass

    raw_labels = cif_block.get("_atom_site_label")
    if raw_labels is None:
        raw_labels = cif_block.get("_atom_site_type_symbol")
    if raw_labels is None:
        return [], []
    if isinstance(raw_labels, str):
        raw_labels = [raw_labels]

    unique_labels = []
    for idx, raw in enumerate(raw_labels):
        label = normalize_occupancy_label(raw, idx)
        if label not in unique_labels:
            unique_labels.append(label)
    return unique_labels, []


def expand_occupancy_values_for_generated_sites(
    occ_values,
    *,
    occupancy_site_labels,
    occupancy_site_expanded_map,
):
    target_len = len(occupancy_site_labels)
    if target_len <= 0:
        if isinstance(occ_values, (list, tuple, np.ndarray)):
            target_len = max(1, len(occ_values))
        else:
            target_len = 1
    normalized = _ensure_numeric_vector(occ_values, [1.0], target_len)
    normalized = [min(1.0, max(0.0, float(v))) for v in normalized]

    if not occupancy_site_expanded_map:
        return normalized

    fallback = normalized[-1] if normalized else 1.0
    expanded = []
    for raw_idx in occupancy_site_expanded_map:
        idx = int(raw_idx)
        if 0 <= idx < len(normalized):
            expanded.append(float(normalized[idx]))
        else:
            expanded.append(float(fallback))
    return expanded


def as_cif_list(raw_value):
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return [raw_value]
    if isinstance(raw_value, list):
        return raw_value
    if isinstance(raw_value, tuple):
        return list(raw_value)
    if isinstance(raw_value, np.ndarray):
        return raw_value.tolist()
    try:
        return list(raw_value)
    except TypeError:
        return [raw_value]


def parse_cif_float_or_default(raw_value, default=0.0):
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        text = str(raw_value).strip().strip("'\"")
        if not text:
            return float(default)

        frac_match = re.fullmatch(
            r"([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)",
            text,
        )
        if frac_match:
            try:
                numer = float(frac_match.group(1))
                denom = float(frac_match.group(2))
                if denom != 0.0:
                    return float(numer / denom)
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        match = re.match(r"[-+0-9\.Ee]+", text)
        if not match:
            return float(default)
        try:
            return float(match.group(0))
        except (TypeError, ValueError):
            return float(default)


def extract_atom_site_fractional_metadata(cif_block):
    x_vals = as_cif_list(cif_block.get("_atom_site_fract_x"))
    y_vals = as_cif_list(cif_block.get("_atom_site_fract_y"))
    z_vals = as_cif_list(cif_block.get("_atom_site_fract_z"))
    if not x_vals or not y_vals or not z_vals:
        return []

    labels = as_cif_list(cif_block.get("_atom_site_label"))
    if not labels:
        labels = as_cif_list(cif_block.get("_atom_site_type_symbol"))

    n_sites = max(len(x_vals), len(y_vals), len(z_vals))
    base_labels = []
    for idx in range(n_sites):
        if idx < len(labels):
            label = normalize_occupancy_label(labels[idx], idx)
        else:
            label = f"site_{idx + 1}"
        base_labels.append(label)

    totals = {}
    for label in base_labels:
        totals[label] = totals.get(label, 0) + 1

    seen = {}
    rows = []
    for idx, base_label in enumerate(base_labels):
        seen[base_label] = seen.get(base_label, 0) + 1
        display = (
            f"{base_label} #{seen[base_label]}"
            if totals[base_label] > 1
            else base_label
        )
        rows.append(
            {
                "row_index": int(idx),
                "label": str(display),
                "x": float(parse_cif_float_or_default(x_vals[idx] if idx < len(x_vals) else 0.0)),
                "y": float(parse_cif_float_or_default(y_vals[idx] if idx < len(y_vals) else 0.0)),
                "z": float(parse_cif_float_or_default(z_vals[idx] if idx < len(z_vals) else 0.0)),
            }
        )

    return rows


def parse_cif_num(txt):
    if txt is None:
        raise ValueError("Missing CIF numeric value")
    if isinstance(txt, (int, float)):
        return float(txt)
    match = re.match(r"[-+0-9\.Ee]+", str(txt).strip())
    if not match:
        raise ValueError(f"Can't parse '{txt}' as a number")
    return float(match.group(0))


@dataclass
class StructureModelState:
    cif_file: str
    cf: Any
    blk: Any
    cif_file2: str | None = None
    cf2: Any = None
    blk2: Any = None
    occupancy_site_labels: list[str] = field(default_factory=list)
    occupancy_site_count: int = 0
    occupancy_site_expanded_map: list[int] = field(default_factory=list)
    occ: list[float] = field(default_factory=list)
    occ_vars: list[Any] = field(default_factory=list)
    atom_site_fractional_metadata: list[dict[str, object]] = field(default_factory=list)
    atom_site_fract_vars: list[dict[str, Any]] = field(default_factory=list)
    av: float = 0.0
    bv: float = 0.0
    cv: float = 0.0
    av2: float | None = None
    cv2: float | None = None
    defaults: dict[str, object] = field(default_factory=dict)
    energy: float = 0.0
    mx: int = 0
    lambda_angstrom: float = 1.0
    intensity_threshold: float = 0.0
    two_theta_range: tuple[float, float] = (0.0, 0.0)
    include_rods_flag: bool = False
    has_second_cif: bool = False
    ht_cache_multi: dict[str, dict[str, object]] = field(default_factory=dict)
    ht_curves_cache: dict[str, object] = field(default_factory=dict)
    sim_miller1_all: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    sim_intens1_all: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float64))
    sim_miller2_all: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    sim_intens2_all: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float64))
    sim_primary_qr_all: dict[object, object] = field(default_factory=dict)
    miller: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    intensities: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float64))
    degeneracy: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int32))
    details: list[object] = field(default_factory=list)
    df_summary: Any = None
    df_details: Any = None
    intensities_cif1: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float64))
    intensities_cif2: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.float64))
    last_occ_for_ht: list[float] = field(default_factory=list)
    last_p_triplet: list[float] = field(default_factory=list)
    last_weights: list[float] = field(default_factory=list)
    last_a_for_ht: float = 0.0
    last_c_for_ht: float = 0.0
    last_iodine_z_for_ht: float = 0.0
    last_phi_l_divisor: float = 0.0
    last_phase_delta_expression: str = ""
    last_finite_stack: bool = False
    last_stack_layers: int = 1
    last_rod_points_per_gz: int = 0
    last_atom_site_fractional_signature: tuple[float, ...] = field(default_factory=tuple)
    miller_generator: Callable[..., tuple[object, object, object, object]] | None = None
    inject_fractional_reflections: Callable[..., tuple[object, object]] | None = None
    bootstrap_complete: bool = False


@dataclass
class PrimaryCifReloadSnapshot:
    cif_file: str
    cf: Any
    blk: Any
    cf2: Any = None
    blk2: Any = None
    occupancy_site_labels: list[str] = field(default_factory=list)
    occupancy_site_expanded_map: list[int] = field(default_factory=list)
    occ: list[float] = field(default_factory=list)
    current_occ_values: list[float] = field(default_factory=list)
    atom_site_fractional_metadata: list[dict[str, object]] = field(default_factory=list)
    current_atom_site_values: list[tuple[float, float, float]] = field(default_factory=list)
    av: float = 0.0
    cv: float = 0.0
    av2: float | None = None
    cv2: float | None = None
    default_a: float = 0.0
    default_c: float = 0.0
    default_iodine_z: float = 0.0
    ht_cache_multi: dict[str, dict[str, object]] = field(default_factory=dict)


@dataclass
class PrimaryCifReloadPlan:
    candidate_path: str
    cf: Any
    blk: Any
    occupancy_site_labels: list[str] = field(default_factory=list)
    occupancy_site_expanded_map: list[int] = field(default_factory=list)
    occupancy_site_count: int = 0
    occ: list[float] = field(default_factory=list)
    atom_site_fractional_metadata: list[dict[str, object]] = field(default_factory=list)
    atom_site_values: list[tuple[float, float, float]] = field(default_factory=list)
    av: float = 0.0
    cv: float = 0.0
    iodine_z: float = 0.0


@dataclass
class DiffuseHTRequest:
    source_cif: str
    active_cif: str
    occ: list[float] = field(default_factory=list)
    p_values: list[float] = field(default_factory=list)
    w_values: list[float] = field(default_factory=list)
    a_lattice: float = 0.0
    c_lattice: float = 0.0
    lambda_angstrom: float = 1.0
    mx: int = 0
    two_theta_max: float | None = None
    finite_stack: bool = False
    stack_layers: int = 1
    iodine_z: float = 0.0
    phase_delta_expression: str = ""
    phi_l_divisor: float = 1.0
    rod_points_per_gz: int = 0


def _coerce_iodine_z(value, fallback):
    try:
        fallback_value = float(fallback)
    except (TypeError, ValueError):
        fallback_value = 0.0
    if not np.isfinite(fallback_value):
        fallback_value = 0.0

    if value is None:
        value = fallback_value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = fallback_value
    if not np.isfinite(numeric):
        numeric = fallback_value
    return float(np.clip(numeric, 0.0, 1.0))


def _infer_iodine_z_or_default(cif_path, *, fallback):
    try:
        value = _infer_iodine_z_like_diffuse(str(cif_path))
    except Exception:
        value = None
    return _coerce_iodine_z(value, fallback)


def atom_site_fractional_default_values(state: StructureModelState):
    return [
        (float(row["x"]), float(row["y"]), float(row["z"]))
        for row in state.atom_site_fractional_metadata
    ]


def current_atom_site_fractional_values(
    state: StructureModelState,
    *,
    tcl_error_types: tuple[type[BaseException], ...] = (),
):
    error_types = tcl_error_types + (ValueError, TypeError, KeyError)
    defaults_xyz = atom_site_fractional_default_values(state)
    if not state.atom_site_fract_vars:
        return defaults_xyz

    values = []
    for idx, axis_vars in enumerate(state.atom_site_fract_vars):
        fallback = defaults_xyz[idx] if idx < len(defaults_xyz) else (0.0, 0.0, 0.0)
        try:
            x_val = float(axis_vars["x"].get())
        except error_types:
            x_val = float(fallback[0])
        if not np.isfinite(x_val):
            x_val = float(fallback[0])
        try:
            y_val = float(axis_vars["y"].get())
        except error_types:
            y_val = float(fallback[1])
        if not np.isfinite(y_val):
            y_val = float(fallback[1])
        try:
            z_val = float(axis_vars["z"].get())
        except error_types:
            z_val = float(fallback[2])
        if not np.isfinite(z_val):
            z_val = float(fallback[2])
        values.append((x_val, y_val, z_val))
    return values


def atom_site_fractional_signature(values):
    flat = []
    for x_val, y_val, z_val in values:
        flat.extend(
            [
                round(float(x_val), 12),
                round(float(y_val), 12),
                round(float(z_val), 12),
            ]
        )
    return tuple(flat)


def atom_site_fractional_values_are_default(
    state: StructureModelState,
    values,
):
    defaults_xyz = atom_site_fractional_default_values(state)
    if len(values) != len(defaults_xyz):
        return False
    for (x_cur, y_cur, z_cur), (x_def, y_def, z_def) in zip(values, defaults_xyz):
        if not math.isclose(float(x_cur), float(x_def), rel_tol=1e-9, abs_tol=1e-9):
            return False
        if not math.isclose(float(y_cur), float(y_def), rel_tol=1e-9, abs_tol=1e-9):
            return False
        if not math.isclose(float(z_cur), float(z_def), rel_tol=1e-9, abs_tol=1e-9):
            return False
    return True


def current_occupancy_values(
    state: StructureModelState,
    *,
    tcl_error_types: tuple[type[BaseException], ...] = (),
):
    error_types = tcl_error_types + (ValueError, TypeError)
    values = []
    for idx, occ_var in enumerate(state.occ_vars):
        try:
            value = float(occ_var.get())
        except error_types:
            fallback = state.occ[idx] if idx < len(state.occ) else 1.0
            value = float(fallback)
        values.append(min(1.0, max(0.0, value)))
    return values


def capture_primary_cif_reload_snapshot(
    state: StructureModelState,
    *,
    current_occ_values,
    current_atom_site_values,
):
    return PrimaryCifReloadSnapshot(
        cif_file=str(state.cif_file),
        cf=state.cf,
        blk=state.blk,
        cf2=state.cf2,
        blk2=state.blk2,
        occupancy_site_labels=list(state.occupancy_site_labels),
        occupancy_site_expanded_map=list(state.occupancy_site_expanded_map),
        occ=list(state.occ),
        current_occ_values=[float(value) for value in current_occ_values],
        atom_site_fractional_metadata=[
            dict(row) for row in state.atom_site_fractional_metadata
        ],
        current_atom_site_values=[
            (float(x_val), float(y_val), float(z_val))
            for (x_val, y_val, z_val) in current_atom_site_values
        ],
        av=float(state.av),
        cv=float(state.cv),
        av2=state.av2,
        cv2=state.cv2,
        default_a=float(state.defaults.get("a", state.av)),
        default_c=float(state.defaults.get("c", state.cv)),
        default_iodine_z=_coerce_iodine_z(
            state.defaults.get("iodine_z", 0.0),
            0.0,
        ),
        ht_cache_multi=dict(state.ht_cache_multi),
    )


def prepare_primary_cif_reload_plan(
    state: StructureModelState,
    raw_path,
    *,
    current_occ_values,
    clamp_site_occupancy_values: Callable[..., list[float]],
):
    text_path = str(raw_path).strip().strip("'\"")
    if not text_path:
        raise ValueError("No CIF file path provided.")

    candidate = Path(text_path).expanduser()
    if not candidate.is_file():
        raise FileNotFoundError(f"CIF file not found: {candidate}")

    new_cf = CifFile.ReadCif(str(candidate))
    keys = list(new_cf.keys())
    if not keys:
        raise ValueError("No CIF data blocks found.")
    new_blk = new_cf[keys[0]]

    new_a_text = new_blk.get("_cell_length_a")
    new_c_text = new_blk.get("_cell_length_c")
    if new_a_text is None or new_c_text is None:
        raise ValueError("CIF is missing _cell_length_a/_cell_length_c fields.")

    new_av = float(parse_cif_num(new_a_text))
    new_cv = float(parse_cif_num(new_c_text))
    new_iodine_z = _infer_iodine_z_or_default(
        str(candidate),
        fallback=state.defaults.get("iodine_z", 0.0),
    )

    new_labels, new_expanded_map = extract_occupancy_site_metadata(
        new_blk,
        str(candidate),
    )
    site_count = len(new_labels) if new_labels else max(1, len(current_occ_values))
    new_occ_values = clamp_site_occupancy_values(
        _ensure_numeric_vector(current_occ_values, [1.0], site_count),
    )

    new_atom_site_metadata = extract_atom_site_fractional_metadata(new_blk)
    new_atom_site_values = [
        (float(row["x"]), float(row["y"]), float(row["z"]))
        for row in new_atom_site_metadata
    ]

    return PrimaryCifReloadPlan(
        candidate_path=str(candidate),
        cf=new_cf,
        blk=new_blk,
        occupancy_site_labels=list(new_labels),
        occupancy_site_expanded_map=list(new_expanded_map),
        occupancy_site_count=int(site_count),
        occ=list(new_occ_values),
        atom_site_fractional_metadata=[dict(row) for row in new_atom_site_metadata],
        atom_site_values=list(new_atom_site_values),
        av=float(new_av),
        cv=float(new_cv),
        iodine_z=float(new_iodine_z),
    )


def apply_primary_cif_reload_plan(
    state: StructureModelState,
    plan: PrimaryCifReloadPlan,
    *,
    occ_vars=None,
    atom_site_fract_vars=None,
    has_second_cif=False,
):
    state.cif_file = str(plan.candidate_path)
    state.cf = plan.cf
    state.blk = plan.blk
    state.occupancy_site_labels = list(plan.occupancy_site_labels)
    state.occupancy_site_expanded_map = list(plan.occupancy_site_expanded_map)
    state.occupancy_site_count = int(plan.occupancy_site_count)
    state.occ = list(plan.occ)
    state.atom_site_fractional_metadata = [
        dict(row) for row in plan.atom_site_fractional_metadata
    ]
    if occ_vars is not None:
        state.occ_vars = list(occ_vars)
    if atom_site_fract_vars is not None:
        state.atom_site_fract_vars = list(atom_site_fract_vars)
    state.av = float(plan.av)
    state.cv = float(plan.cv)
    state.defaults["a"] = float(plan.av)
    state.defaults["c"] = float(plan.cv)
    state.defaults["iodine_z"] = float(plan.iodine_z)
    state.ht_cache_multi = {}
    if has_second_cif and state.blk2 is not None:
        if state.blk2.get("_cell_length_a") is None:
            state.av2 = float(plan.av)
        if state.blk2.get("_cell_length_c") is None:
            state.cv2 = float(plan.cv)


def restore_primary_cif_reload_snapshot(
    state: StructureModelState,
    snapshot: PrimaryCifReloadSnapshot,
    *,
    occ_vars=None,
    atom_site_fract_vars=None,
):
    state.cif_file = snapshot.cif_file
    state.cf = snapshot.cf
    state.blk = snapshot.blk
    state.cf2 = snapshot.cf2
    state.blk2 = snapshot.blk2
    state.occupancy_site_labels = list(snapshot.occupancy_site_labels)
    state.occupancy_site_expanded_map = list(snapshot.occupancy_site_expanded_map)
    state.occupancy_site_count = (
        len(snapshot.occupancy_site_labels)
        if snapshot.occupancy_site_labels
        else max(1, len(snapshot.current_occ_values))
    )
    state.occ = (
        list(snapshot.occ)
        if snapshot.occ
        else list(snapshot.current_occ_values)
    )
    state.atom_site_fractional_metadata = [
        dict(row) for row in snapshot.atom_site_fractional_metadata
    ]
    if occ_vars is not None:
        state.occ_vars = list(occ_vars)
    if atom_site_fract_vars is not None:
        state.atom_site_fract_vars = list(atom_site_fract_vars)
    state.av = float(snapshot.av)
    state.cv = float(snapshot.cv)
    state.av2 = snapshot.av2
    state.cv2 = snapshot.cv2
    state.defaults["a"] = float(snapshot.default_a)
    state.defaults["c"] = float(snapshot.default_c)
    state.defaults["iodine_z"] = float(snapshot.default_iodine_z)
    state.ht_cache_multi = dict(snapshot.ht_cache_multi)


def current_iodine_z(
    state: StructureModelState,
    atom_site_override_state,
    *,
    active_cif_path=None,
    atom_site_values=None,
    tcl_error_types: tuple[type[BaseException], ...] = (),
):
    fallback = _coerce_iodine_z(state.defaults.get("iodine_z", 0.0), 0.0)

    cif_path = str(active_cif_path).strip() if active_cif_path is not None else ""
    if not cif_path:
        try:
            cif_path = active_primary_cif_path(
                state,
                atom_site_override_state,
                atom_site_values=atom_site_values,
                tcl_error_types=tcl_error_types,
            )
        except Exception:
            cif_path = str(state.cif_file)

    return _infer_iodine_z_or_default(cif_path, fallback=fallback)


def active_primary_cif_path(
    state: StructureModelState,
    atom_site_override_state,
    *,
    atom_site_values=None,
    tcl_error_types: tuple[type[BaseException], ...] = (),
):
    source_path = str(state.cif_file)
    if not state.atom_site_fractional_metadata:
        return source_path

    current_values = (
        current_atom_site_fractional_values(state, tcl_error_types=tcl_error_types)
        if atom_site_values is None
        else list(atom_site_values)
    )
    if atom_site_fractional_values_are_default(state, current_values):
        return source_path

    signature = atom_site_fractional_signature(current_values)
    abs_source = os.path.abspath(source_path)
    if (
        atom_site_override_state.temp_path
        and atom_site_override_state.source_path == abs_source
        and atom_site_override_state.signature == signature
        and Path(atom_site_override_state.temp_path).is_file()
    ):
        return atom_site_override_state.temp_path

    with redirect_stdout(io.StringIO()):
        cf_local = CifFile.ReadCif(abs_source)
    keys = list(cf_local.keys())
    if not keys:
        return source_path
    block = cf_local[keys[0]]

    x_vals = as_cif_list(block.get("_atom_site_fract_x"))
    y_vals = as_cif_list(block.get("_atom_site_fract_y"))
    z_vals = as_cif_list(block.get("_atom_site_fract_z"))
    if not x_vals or not y_vals or not z_vals:
        return source_path

    n_rows = max(len(x_vals), len(y_vals), len(z_vals))
    if len(x_vals) < n_rows:
        fill = str(x_vals[-1]) if x_vals else "0.0"
        x_vals.extend([fill] * (n_rows - len(x_vals)))
    if len(y_vals) < n_rows:
        fill = str(y_vals[-1]) if y_vals else "0.0"
        y_vals.extend([fill] * (n_rows - len(y_vals)))
    if len(z_vals) < n_rows:
        fill = str(z_vals[-1]) if z_vals else "0.0"
        z_vals.extend([fill] * (n_rows - len(z_vals)))

    for row_data, (x_val, y_val, z_val) in zip(state.atom_site_fractional_metadata, current_values):
        row_idx = int(row_data.get("row_index", -1))
        if row_idx < 0 or row_idx >= n_rows:
            continue
        x_vals[row_idx] = f"{float(x_val):.12g}"
        y_vals[row_idx] = f"{float(y_val):.12g}"
        z_vals[row_idx] = f"{float(z_val):.12g}"

    block["_atom_site_fract_x"] = x_vals
    block["_atom_site_fract_y"] = y_vals
    block["_atom_site_fract_z"] = z_vals

    if (
        atom_site_override_state.temp_path is None
        or atom_site_override_state.source_path != abs_source
    ):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".cif")
        tmp.close()
        atom_site_override_state.temp_path = tmp.name

    try:
        with redirect_stdout(io.StringIO()):
            CifFile.WriteCif(cf_local, atom_site_override_state.temp_path)
    except AttributeError:
        with open(atom_site_override_state.temp_path, "w", encoding="utf-8") as fh:
            with redirect_stdout(io.StringIO()):
                fh.write(cf_local.WriteOut())

    atom_site_override_state.source_path = abs_source
    atom_site_override_state.signature = signature
    return atom_site_override_state.temp_path


def reset_atom_site_override_cache(atom_site_override_state):
    atom_site_override_state.source_path = None
    atom_site_override_state.signature = None


def build_diffuse_ht_request(
    state: StructureModelState,
    atom_site_override_state,
    *,
    p_values,
    w_values,
    a_lattice,
    c_lattice,
    lambda_angstrom,
    mx,
    two_theta_range,
    finite_stack,
    stack_layers,
    phase_delta_expression,
    phi_l_divisor,
    rod_points_per_gz=0,
    occupancy_values=None,
    atom_site_values=None,
    tcl_error_types: tuple[type[BaseException], ...] = (),
):
    active_cif = active_primary_cif_path(
        state,
        atom_site_override_state,
        atom_site_values=atom_site_values,
        tcl_error_types=tcl_error_types,
    )
    source_cif = str(state.cif_file)
    if not Path(active_cif).is_file():
        raise FileNotFoundError(f"CIF file not found: {source_cif}")

    current_occ = (
        current_occupancy_values(state, tcl_error_types=tcl_error_types)
        if occupancy_values is None
        else [float(value) for value in occupancy_values]
    )
    occ_vals = expand_occupancy_values_for_generated_sites(
        current_occ,
        occupancy_site_labels=state.occupancy_site_labels,
        occupancy_site_expanded_map=state.occupancy_site_expanded_map,
    )

    try:
        two_theta_max = float(two_theta_range[1])
    except Exception:
        two_theta_max = None

    return DiffuseHTRequest(
        source_cif=source_cif,
        active_cif=str(active_cif),
        occ=list(occ_vals),
        p_values=[float(value) for value in p_values],
        w_values=[float(value) for value in w_values],
        a_lattice=float(a_lattice),
        c_lattice=float(c_lattice),
        lambda_angstrom=float(lambda_angstrom),
        mx=int(mx),
        two_theta_max=two_theta_max,
        finite_stack=bool(finite_stack),
        stack_layers=int(max(1, stack_layers)),
        iodine_z=current_iodine_z(
            state,
            atom_site_override_state,
            active_cif_path=active_cif,
            atom_site_values=atom_site_values,
            tcl_error_types=tcl_error_types,
        ),
        phase_delta_expression=str(phase_delta_expression),
        phi_l_divisor=float(phi_l_divisor),
        rod_points_per_gz=gui_controllers.normalize_rod_points_per_gz(
            rod_points_per_gz,
            gui_controllers.default_rod_points_per_gz(c_lattice),
        ),
    )


def _set_status_text(
    set_status_text: Callable[[str], None] | None,
    text: str,
) -> None:
    if callable(set_status_text):
        set_status_text(str(text))


def primary_cif_dialog_initial_dir(current_cif_path: object, default_dir: object) -> str:
    """Return the preferred initial directory for the primary-CIF file dialog."""

    try:
        return str(Path(str(current_cif_path)).expanduser().parent)
    except Exception:
        return str(default_dir)


def browse_primary_cif_with_dialog(
    *,
    current_cif_path: object,
    file_dialog_dir: object,
    askopenfilename: Callable[..., object],
    set_cif_path_text: Callable[[str], None],
    apply_primary_cif_path: Callable[[str], None],
) -> bool:
    """Open the primary-CIF picker and apply the selected path."""

    selected = askopenfilename(
        title="Select Primary CIF",
        initialdir=primary_cif_dialog_initial_dir(current_cif_path, file_dialog_dir),
        filetypes=[("CIF files", "*.cif *.CIF"), ("All files", "*.*")],
    )
    if not selected:
        return False

    selected_path = str(selected)
    set_cif_path_text(selected_path)
    apply_primary_cif_path(selected_path)
    return True


def open_diffuse_ht_view_with_status(
    *,
    build_request: Callable[[], DiffuseHTRequest],
    open_view: Callable[[DiffuseHTRequest], None],
    set_status_text: Callable[[str], None] | None = None,
    tcl_error_types: tuple[type[BaseException], ...] = (),
) -> bool:
    """Open the diffuse-HT viewer and report GUI status text."""

    input_error_types = tuple(tcl_error_types) + (ValueError,)
    try:
        request = build_request()
    except FileNotFoundError as exc:
        _set_status_text(set_status_text, str(exc))
        return False
    except Exception as exc:
        if isinstance(exc, input_error_types):
            _set_status_text(
                set_status_text,
                f"Failed to read diffuse HT inputs: {exc}",
            )
            return False
        raise

    try:
        open_view(request)
    except Exception as exc:
        _set_status_text(
            set_status_text,
            f"Failed to open diffuse HT viewer: {exc}",
        )
        return False

    _set_status_text(
        set_status_text,
        f"Opened diffuse HT viewer: {Path(request.source_cif).name}",
    )
    return True


def export_diffuse_ht_txt_with_dialog(
    *,
    build_request: Callable[[], DiffuseHTRequest],
    get_download_dir: Callable[[], object] | None,
    asksaveasfilename: Callable[..., object],
    export_table: Callable[[str, DiffuseHTRequest], int],
    set_status_text: Callable[[str], None] | None = None,
    tcl_error_types: tuple[type[BaseException], ...] = (),
) -> bool:
    """Export one algebraic HT table through a save-file dialog."""

    input_error_types = tuple(tcl_error_types) + (ValueError,)
    try:
        request = build_request()
    except FileNotFoundError as exc:
        _set_status_text(set_status_text, str(exc))
        return False
    except Exception as exc:
        if isinstance(exc, input_error_types):
            _set_status_text(
                set_status_text,
                f"Failed to read algebraic HT export inputs: {exc}",
            )
            return False
        raise

    try:
        initial_dir = (
            str(get_download_dir()) if callable(get_download_dir) else None
        )
    except Exception:
        initial_dir = None
    if not initial_dir:
        initial_dir = str(Path(request.source_cif).expanduser().parent)

    save_path = asksaveasfilename(
        title="Export Algebraic HT Table",
        defaultextension=".txt",
        initialdir=initial_dir,
        initialfile=f"{Path(request.source_cif).stem}_algebraic_ht.txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
    )
    if not save_path:
        return False

    try:
        row_count = int(export_table(str(save_path), request))
    except Exception as exc:
        _set_status_text(
            set_status_text,
            f"Failed to export algebraic HT table: {exc}",
        )
        return False

    _set_status_text(
        set_status_text,
        f"Exported algebraic HT table ({row_count} rows): {Path(str(save_path)).name}",
    )
    return True


def build_ht_cache(
    state: StructureModelState,
    p_val,
    occ_vals,
    a_axis,
    c_axis,
    iodine_z,
    phi_l_divisor,
    finite_stack_flag,
    stack_layers_count,
    phase_delta_expression,
    rod_points_per_gz=0,
    *,
    cif_path_override=None,
):
    layers = int(max(1, stack_layers_count))
    rod_points_value = gui_controllers.normalize_rod_points_per_gz(
        rod_points_per_gz,
        gui_controllers.default_rod_points_per_gz(c_axis),
    )
    l_step = gui_controllers.rod_l_step_from_points_per_gz(
        rod_points_value,
        c_axis,
        fallback_points=rod_points_value,
        fallback_l_step=gui_controllers.DEFAULT_ROD_L_STEP,
    )
    occ_expanded = expand_occupancy_values_for_generated_sites(
        occ_vals,
        occupancy_site_labels=state.occupancy_site_labels,
        occupancy_site_expanded_map=state.occupancy_site_expanded_map,
    )
    active_cif_path = (
        str(cif_path_override)
        if cif_path_override is not None
        else str(state.cif_file)
    )
    curves_raw = ht_Iinf_dict(
        cif_path=active_cif_path,
        mx=int(state.mx),
        occ=occ_expanded,
        p=p_val,
        L_step=l_step,
        two_theta_max=state.two_theta_range[1],
        lambda_=state.lambda_angstrom,
        a_lattice=a_axis,
        c_lattice=c_axis,
        phase_z_divisor=phi_l_divisor,
        iodine_z=iodine_z,
        phase_delta_expression=phase_delta_expression,
        phi_l_divisor=phi_l_divisor,
        finite_stack=finite_stack_flag,
        stack_layers=layers,
    )
    curves = {}
    for hk, data in curves_raw.items():
        l_vals = np.asarray(data["L"], dtype=float)
        i_vals = np.asarray(data["I"], dtype=float)
        keep = np.isfinite(l_vals) & np.isfinite(i_vals) & (l_vals > 0.0)
        if not np.any(keep):
            continue
        curves[hk] = {"L": l_vals[keep], "I": i_vals[keep]}
    qr = ht_dict_to_qr_dict(curves)
    arrays = ht_dict_to_arrays(curves)
    return {
        "p": p_val,
        "occ": tuple(occ_vals),
        "ht": curves,
        "qr": qr,
        "arrays": arrays,
        "two_theta_max": state.two_theta_range[1],
        "a": float(a_axis),
        "c": float(c_axis),
        "iodine_z": float(iodine_z),
        "phi_l_divisor": float(phi_l_divisor),
        "phase_delta_expression": str(phase_delta_expression),
        "finite_stack": bool(finite_stack_flag),
        "stack_layers": layers,
        "rod_points_per_gz": int(rod_points_value),
        "L_step": float(l_step),
        "cif_path": active_cif_path,
    }


def combine_ht_dicts(caches, weights):
    out = {}
    for cache, weight in zip(caches, weights):
        ht_curves = cache["ht"]
        for hk, data in ht_curves.items():
            if hk not in out:
                out[hk] = {
                    "L": data["L"].copy(),
                    "I": float(weight) * data["I"].copy(),
                }
            else:
                entry = out[hk]
                if entry["L"].shape != data["L"].shape or not np.allclose(entry["L"], data["L"]):
                    union_l = np.union1d(entry["L"], data["L"])
                    entry_i = np.interp(
                        union_l,
                        entry["L"],
                        entry["I"],
                        left=0.0,
                        right=0.0,
                    )
                    add_i = float(weight) * np.interp(
                        union_l,
                        data["L"],
                        data["I"],
                        left=0.0,
                        right=0.0,
                    )
                    entry["L"] = union_l
                    entry["I"] = entry_i + add_i
                else:
                    entry["I"] += float(weight) * data["I"]
    return out


def build_lightweight_structure_model_state(
    *,
    cif_file: str,
    cf,
    blk,
    cif_file2: str | None,
    occupancy_site_labels,
    occupancy_site_expanded_map,
    occ,
    atom_site_fractional_metadata,
    av,
    bv,
    cv,
    av2,
    cv2,
    defaults,
    mx,
    lambda_angstrom,
    intensity_threshold,
    two_theta_range,
    include_rods_flag,
    miller_generator: Callable[..., tuple[object, object, object, object]] | None = None,
    inject_fractional_reflections: Callable[..., tuple[object, object]] | None = None,
):
    """Build a lightweight structure-model shell without HT cache materialization."""

    return StructureModelState(
        cif_file=str(cif_file),
        cf=cf,
        blk=blk,
        cif_file2=str(cif_file2) if cif_file2 else None,
        occupancy_site_labels=list(occupancy_site_labels),
        occupancy_site_count=int(len(occupancy_site_labels) or max(1, len(occ))),
        occupancy_site_expanded_map=list(occupancy_site_expanded_map),
        occ=[float(v) for v in occ],
        atom_site_fractional_metadata=[dict(row) for row in atom_site_fractional_metadata],
        av=float(av),
        bv=float(bv),
        cv=float(cv),
        av2=float(av2) if av2 is not None else None,
        cv2=float(cv2) if cv2 is not None else None,
        defaults=dict(defaults),
        energy=6.62607e-34 * 2.99792458e8 / (float(lambda_angstrom) * 1e-10) / (1.602176634e-19),
        mx=int(mx),
        lambda_angstrom=float(lambda_angstrom),
        intensity_threshold=float(intensity_threshold),
        two_theta_range=(float(two_theta_range[0]), float(two_theta_range[1])),
        include_rods_flag=bool(include_rods_flag),
        has_second_cif=bool(cif_file2),
        miller_generator=miller_generator,
        inject_fractional_reflections=inject_fractional_reflections,
        bootstrap_complete=False,
    )


def build_initial_structure_model_state(
    *,
    cif_file: str,
    cf,
    blk,
    cif_file2: str | None,
    occupancy_site_labels,
    occupancy_site_expanded_map,
    occ,
    atom_site_fractional_metadata,
    av,
    bv,
    cv,
    av2,
    cv2,
    defaults,
    mx,
    lambda_angstrom,
    intensity_threshold,
    two_theta_range,
    include_rods_flag,
    combine_weighted_intensities: Callable[..., np.ndarray],
    miller_generator: Callable[..., tuple[object, object, object, object]],
    inject_fractional_reflections: Callable[..., tuple[object, object]],
    debug_print: Callable[..., None] | None = None,
):
    """Build the initial structure-model cache and diffraction arrays."""

    state = build_lightweight_structure_model_state(
        cif_file=cif_file,
        cf=cf,
        blk=blk,
        cif_file2=cif_file2,
        occupancy_site_labels=occupancy_site_labels,
        occupancy_site_expanded_map=occupancy_site_expanded_map,
        occ=occ,
        atom_site_fractional_metadata=atom_site_fractional_metadata,
        av=av,
        bv=bv,
        cv=cv,
        av2=av2,
        cv2=cv2,
        defaults=defaults,
        mx=mx,
        lambda_angstrom=lambda_angstrom,
        intensity_threshold=intensity_threshold,
        two_theta_range=two_theta_range,
        include_rods_flag=include_rods_flag,
        miller_generator=miller_generator,
        inject_fractional_reflections=inject_fractional_reflections,
    )

    if state.has_second_cif:
        state.cf2 = CifFile.ReadCif(state.cif_file2)
        state.blk2 = state.cf2[list(state.cf2.keys())[0]]

    default_a_axis = float(state.defaults["a"])
    default_c_axis = float(state.defaults["c"])
    default_rod_points_per_gz = gui_controllers.normalize_rod_points_per_gz(
        state.defaults.get("rod_points_per_gz"),
        gui_controllers.default_rod_points_per_gz(default_c_axis),
    )
    state.ht_cache_multi = {
        "p0": build_ht_cache(
            state,
            state.defaults["p0"],
            state.occ,
            default_a_axis,
            default_c_axis,
            state.defaults["iodine_z"],
            state.defaults["phi_l_divisor"],
            state.defaults["finite_stack"],
            state.defaults["stack_layers"],
            state.defaults["phase_delta_expression"],
            default_rod_points_per_gz,
        ),
        "p1": build_ht_cache(
            state,
            state.defaults["p1"],
            state.occ,
            default_a_axis,
            default_c_axis,
            state.defaults["iodine_z"],
            state.defaults["phi_l_divisor"],
            state.defaults["finite_stack"],
            state.defaults["stack_layers"],
            state.defaults["phase_delta_expression"],
            default_rod_points_per_gz,
        ),
        "p2": build_ht_cache(
            state,
            state.defaults["p2"],
            state.occ,
            default_a_axis,
            default_c_axis,
            state.defaults["iodine_z"],
            state.defaults["phi_l_divisor"],
            state.defaults["finite_stack"],
            state.defaults["stack_layers"],
            state.defaults["phase_delta_expression"],
            default_rod_points_per_gz,
        ),
    }

    weights_init = np.array(
        [state.defaults["w0"], state.defaults["w1"], state.defaults["w2"]],
        dtype=float,
    )
    weights_init /= weights_init.sum() if weights_init.sum() else 1.0
    combined_ht = combine_ht_dicts(
        [
            state.ht_cache_multi["p0"],
            state.ht_cache_multi["p1"],
            state.ht_cache_multi["p2"],
        ],
        weights_init,
    )
    combined_qr = ht_dict_to_qr_dict(combined_ht)
    miller1, intens1, degeneracy1, details1 = ht_dict_to_arrays(combined_ht)
    state.ht_curves_cache = {
        "curves": combined_ht,
        "qr_curves": combined_qr,
        "arrays": (miller1, intens1, degeneracy1, details1),
        "a": default_a_axis,
        "c": default_c_axis,
        "iodine_z": float(state.defaults["iodine_z"]),
        "phi_l_divisor": float(state.defaults["phi_l_divisor"]),
        "phase_delta_expression": str(state.defaults["phase_delta_expression"]),
        "finite_stack": bool(state.defaults["finite_stack"]),
        "stack_layers": int(max(1, state.defaults["stack_layers"])),
        "rod_points_per_gz": int(default_rod_points_per_gz),
    }
    state.last_occ_for_ht = list(state.occ)
    state.last_p_triplet = [
        state.defaults["p0"],
        state.defaults["p1"],
        state.defaults["p2"],
    ]
    state.last_weights = list(weights_init)
    state.last_a_for_ht = default_a_axis
    state.last_c_for_ht = default_c_axis
    state.last_iodine_z_for_ht = float(state.defaults["iodine_z"])
    state.last_phi_l_divisor = float(state.defaults["phi_l_divisor"])
    state.last_phase_delta_expression = str(state.defaults["phase_delta_expression"])
    state.last_finite_stack = bool(state.defaults["finite_stack"])
    state.last_stack_layers = int(max(1, state.defaults["stack_layers"]))
    state.last_rod_points_per_gz = int(default_rod_points_per_gz)
    state.last_atom_site_fractional_signature = atom_site_fractional_signature(
        atom_site_fractional_default_values(state)
    )

    if debug_print is not None:
        debug_print("miller1 shape:", miller1.shape, "intens1 shape:", intens1.shape)
        debug_print("miller1 sample:", miller1[:5])

    if state.has_second_cif:
        miller2, intens2, degeneracy2, details2 = miller_generator(
            state.mx,
            state.cif_file2,
            state.occ,
            state.lambda_angstrom,
            state.energy,
            state.intensity_threshold,
            state.two_theta_range,
        )
        if state.include_rods_flag:
            miller2, intens2 = inject_fractional_reflections(
                miller2,
                intens2,
                state.mx,
            )
        union_set = {tuple(hkl) for hkl in miller1} | {tuple(hkl) for hkl in miller2}
        state.miller = np.array(sorted(union_set), dtype=float)
        int1_dict = {tuple(h): i for h, i in zip(miller1, intens1)}
        int2_dict = {tuple(h): i for h, i in zip(miller2, intens2)}
        deg_dict1 = {tuple(h): d for h, d in zip(miller1, degeneracy1)}
        deg_dict2 = {tuple(h): d for h, d in zip(miller2, degeneracy2)}
        details_dict1 = {tuple(miller1[i]): details1[i] for i in range(len(miller1))}
        details_dict2 = {tuple(miller2[i]): details2[i] for i in range(len(miller2))}
        state.intensities_cif1 = np.array(
            [int1_dict.get(tuple(h), 0.0) for h in state.miller]
        )
        state.intensities_cif2 = np.array(
            [int2_dict.get(tuple(h), 0.0) for h in state.miller]
        )
        state.degeneracy = np.array(
            [
                deg_dict1.get(tuple(h), 0) + deg_dict2.get(tuple(h), 0)
                for h in state.miller
            ],
            dtype=np.int32,
        )
        state.details = [
            details_dict1.get(tuple(h), []) + details_dict2.get(tuple(h), [])
            for h in state.miller
        ]
        state.intensities = combine_weighted_intensities(
            state.intensities_cif1,
            state.intensities_cif2,
            weight1=0.5,
            weight2=0.5,
        )
        state.sim_miller1_all = np.asarray(miller1, dtype=np.float64)
        state.sim_intens1_all = np.asarray(intens1, dtype=np.float64)
        state.sim_miller2_all = np.asarray(miller2, dtype=np.float64)
        state.sim_intens2_all = np.asarray(intens2, dtype=np.float64)
        state.sim_primary_qr_all = combined_qr
    else:
        state.miller = np.asarray(miller1, dtype=np.float64)
        state.intensities_cif1 = np.asarray(intens1, dtype=np.float64)
        state.intensities_cif2 = np.zeros_like(state.intensities_cif1)
        state.degeneracy = np.asarray(degeneracy1, dtype=np.int32)
        state.details = list(details1)
        state.intensities = state.intensities_cif1.copy()
        state.sim_miller1_all = np.asarray(miller1, dtype=np.float64)
        state.sim_intens1_all = np.asarray(intens1, dtype=np.float64)
        state.sim_miller2_all = np.empty((0, 3), dtype=np.float64)
        state.sim_intens2_all = np.empty((0,), dtype=np.float64)
        state.sim_primary_qr_all = combined_qr

    state.bootstrap_complete = True
    return state


def bootstrap_structure_model_state(
    state: StructureModelState,
    *,
    combine_weighted_intensities: Callable[..., np.ndarray],
    debug_print: Callable[..., None] | None = None,
) -> StructureModelState:
    """Materialize one lightweight structure-model shell into a full cache."""

    bootstrapped = build_initial_structure_model_state(
        cif_file=state.cif_file,
        cf=state.cf,
        blk=state.blk,
        cif_file2=state.cif_file2,
        occupancy_site_labels=state.occupancy_site_labels,
        occupancy_site_expanded_map=state.occupancy_site_expanded_map,
        occ=state.occ,
        atom_site_fractional_metadata=state.atom_site_fractional_metadata,
        av=state.av,
        bv=state.bv,
        cv=state.cv,
        av2=state.av2,
        cv2=state.cv2,
        defaults=state.defaults,
        mx=state.mx,
        lambda_angstrom=state.lambda_angstrom,
        intensity_threshold=state.intensity_threshold,
        two_theta_range=state.two_theta_range,
        include_rods_flag=state.include_rods_flag,
        combine_weighted_intensities=combine_weighted_intensities,
        miller_generator=state.miller_generator,
        inject_fractional_reflections=state.inject_fractional_reflections,
        debug_print=debug_print,
    )
    bootstrapped.occ_vars = list(state.occ_vars)
    bootstrapped.atom_site_fract_vars = list(state.atom_site_fract_vars)
    return bootstrapped


def update_weighted_intensities(
    state: StructureModelState,
    *,
    weight1,
    weight2,
    combine_weighted_intensities: Callable[..., np.ndarray],
    schedule_update: Callable[[], None],
):
    """Recompute intensities using the current CIF weights."""

    state.intensities = combine_weighted_intensities(
        state.intensities_cif1,
        state.intensities_cif2,
        weight1=weight1,
        weight2=weight2,
    )
    if state.df_summary is not None:
        state.df_summary["Intensity"] = state.intensities
    schedule_update()


def rebuild_diffraction_inputs(
    state: StructureModelState,
    *,
    new_occ,
    p_vals,
    weights,
    a_axis,
    c_axis,
    finite_stack_flag,
    layers,
    phase_delta_expression_current,
    phi_l_divisor_current,
    atom_site_values,
    iodine_z_current,
    rod_points_per_gz=0,
    atom_site_override_state,
    simulation_runtime_state,
    combine_weighted_intensities: Callable[..., np.ndarray],
    build_intensity_dataframes: Callable[..., tuple[object, object]],
    apply_bragg_qr_filters: Callable[..., None],
    schedule_update: Callable[[], None],
    weight1,
    weight2,
    tcl_error_types: tuple[type[BaseException], ...] = (),
    force=False,
    trigger_update=True,
):
    """Refresh cached HT curves and peak lists for the current settings."""

    rod_points_per_gz = gui_controllers.normalize_rod_points_per_gz(
        rod_points_per_gz,
        gui_controllers.default_rod_points_per_gz(c_axis),
    )
    atom_site_signature = atom_site_fractional_signature(atom_site_values)
    active_cif = active_primary_cif_path(
        state,
        atom_site_override_state,
        atom_site_values=atom_site_values,
        tcl_error_types=tcl_error_types,
    )

    if (
        not force
        and list(new_occ) == state.last_occ_for_ht
        and list(p_vals) == state.last_p_triplet
        and list(weights) == state.last_weights
        and math.isclose(float(a_axis), state.last_a_for_ht, rel_tol=1e-9, abs_tol=1e-9)
        and math.isclose(float(c_axis), state.last_c_for_ht, rel_tol=1e-9, abs_tol=1e-9)
        and math.isclose(iodine_z_current, state.last_iodine_z_for_ht, rel_tol=1e-9, abs_tol=1e-9)
        and math.isclose(float(phi_l_divisor_current), float(state.last_phi_l_divisor), rel_tol=1e-9, abs_tol=1e-9)
        and phase_delta_expression_current == state.last_phase_delta_expression
        and state.last_finite_stack == bool(finite_stack_flag)
        and (not finite_stack_flag or state.last_stack_layers == int(layers))
        and int(rod_points_per_gz) == int(state.last_rod_points_per_gz)
        and atom_site_signature == state.last_atom_site_fractional_signature
    ):
        simulation_runtime_state.last_sim_signature = None
        simulation_runtime_state.last_simulation_signature = None
        if trigger_update:
            schedule_update()
        return

    def get_cache(label, p_val):
        cache = state.ht_cache_multi.get(label)
        if (
            cache is None
            or cache["p"] != p_val
            or list(cache["occ"]) != list(new_occ)
            or cache.get("two_theta_max") != state.two_theta_range[1]
            or not math.isclose(cache.get("a", float("nan")), float(a_axis), rel_tol=1e-9, abs_tol=1e-9)
            or not math.isclose(cache.get("c", float("nan")), float(c_axis), rel_tol=1e-9, abs_tol=1e-9)
            or not math.isclose(cache.get("iodine_z", float("nan")), iodine_z_current, rel_tol=1e-9, abs_tol=1e-9)
            or not math.isclose(cache.get("phi_l_divisor", float("nan")), float(phi_l_divisor_current), rel_tol=1e-9, abs_tol=1e-9)
            or cache.get("phase_delta_expression", "") != phase_delta_expression_current
            or bool(cache.get("finite_stack")) != bool(finite_stack_flag)
            or (finite_stack_flag and cache.get("stack_layers") != int(layers))
            or int(cache.get("rod_points_per_gz", -1)) != int(rod_points_per_gz)
            or cache.get("cif_path", str(state.cif_file)) != active_cif
        ):
            cache = build_ht_cache(
                state,
                p_val,
                new_occ,
                a_axis,
                c_axis,
                iodine_z_current,
                phi_l_divisor_current,
                finite_stack_flag,
                layers,
                phase_delta_expression_current,
                rod_points_per_gz,
                cif_path_override=active_cif,
            )
            state.ht_cache_multi[label] = cache
        return cache

    caches = [
        get_cache("p0", p_vals[0]),
        get_cache("p1", p_vals[1]),
        get_cache("p2", p_vals[2]),
    ]

    combined_ht_local = combine_ht_dicts(caches, weights)
    combined_qr_local = ht_dict_to_qr_dict(combined_ht_local)
    arrays_local = ht_dict_to_arrays(combined_ht_local)
    state.ht_curves_cache = {
        "curves": combined_ht_local,
        "qr_curves": combined_qr_local,
        "arrays": arrays_local,
        "a": float(a_axis),
        "c": float(c_axis),
        "iodine_z": float(iodine_z_current),
        "phi_l_divisor": float(phi_l_divisor_current),
        "phase_delta_expression": str(phase_delta_expression_current),
        "finite_stack": bool(finite_stack_flag),
        "stack_layers": int(layers),
        "rod_points_per_gz": int(rod_points_per_gz),
    }
    state.last_occ_for_ht = list(new_occ)
    state.last_p_triplet = list(p_vals)
    state.last_weights = list(weights)
    state.last_a_for_ht = float(a_axis)
    state.last_c_for_ht = float(c_axis)
    state.last_iodine_z_for_ht = float(iodine_z_current)
    state.last_phi_l_divisor = float(phi_l_divisor_current)
    state.last_phase_delta_expression = str(phase_delta_expression_current)
    state.last_finite_stack = bool(finite_stack_flag)
    state.last_stack_layers = int(max(1, layers))
    state.last_rod_points_per_gz = int(rod_points_per_gz)
    state.last_atom_site_fractional_signature = atom_site_signature

    m1, i1, d1, det1 = arrays_local
    deg_dict1 = {tuple(m1[i]): int(d1[i]) for i in range(len(m1))}
    det_dict1 = {tuple(m1[i]): det1[i] for i in range(len(m1))}

    if state.has_second_cif:
        if state.miller_generator is None or state.inject_fractional_reflections is None:
            raise RuntimeError("Secondary CIF rebuild requested without generator helpers.")
        m2, i2, d2, det2 = state.miller_generator(
            state.mx,
            state.cif_file2,
            new_occ,
            state.lambda_angstrom,
            state.energy,
            state.intensity_threshold,
            state.two_theta_range,
        )
        if state.include_rods_flag:
            m2, i2 = state.inject_fractional_reflections(
                m2,
                i2,
                state.mx,
            )
        deg_dict2 = {tuple(m2[i]): int(d2[i]) for i in range(len(m2))}
        det_dict2 = {tuple(m2[i]): det2[i] for i in range(len(m2))}
        union = {tuple(h) for h in m1} | {tuple(h) for h in m2}
        state.miller = np.array(sorted(union), dtype=float)
        int1 = {tuple(h): v for h, v in zip(m1, i1)}
        int2 = {tuple(h): v for h, v in zip(m2, i2)}
        state.intensities_cif1 = np.array(
            [int1.get(tuple(h), 0.0) for h in state.miller]
        )
        state.intensities_cif2 = np.array(
            [int2.get(tuple(h), 0.0) for h in state.miller]
        )
        state.intensities = combine_weighted_intensities(
            state.intensities_cif1,
            state.intensities_cif2,
            weight1=weight1,
            weight2=weight2,
        )
        state.sim_miller1_all = np.asarray(m1, dtype=np.float64)
        state.sim_intens1_all = np.asarray(i1, dtype=np.float64)
        state.sim_miller2_all = np.asarray(m2, dtype=np.float64)
        state.sim_intens2_all = np.asarray(i2, dtype=np.float64)
        state.sim_primary_qr_all = combined_qr_local
        simulation_runtime_state.sim_miller1_all = np.asarray(m1, dtype=np.float64)
        simulation_runtime_state.sim_intens1_all = np.asarray(i1, dtype=np.float64)
        simulation_runtime_state.sim_primary_qr_all = combined_qr_local
        simulation_runtime_state.sim_miller2_all = np.asarray(m2, dtype=np.float64)
        simulation_runtime_state.sim_intens2_all = np.asarray(i2, dtype=np.float64)
        state.degeneracy = np.array(
            [
                deg_dict1.get(tuple(h), 0) + deg_dict2.get(tuple(h), 0)
                for h in state.miller
            ],
            dtype=np.int32,
        )
        state.details = [
            det_dict1.get(tuple(h), []) + det_dict2.get(tuple(h), [])
            for h in state.miller
        ]
    else:
        state.miller = np.asarray(m1, dtype=np.float64)
        state.intensities_cif1 = np.asarray(i1, dtype=np.float64)
        state.intensities_cif2 = np.zeros_like(state.intensities_cif1)
        state.intensities = state.intensities_cif1
        state.degeneracy = np.asarray(d1, dtype=np.int32)
        state.details = list(det1)
        state.sim_miller1_all = np.asarray(m1, dtype=np.float64)
        state.sim_intens1_all = np.asarray(i1, dtype=np.float64)
        state.sim_miller2_all = np.empty((0, 3), dtype=np.float64)
        state.sim_intens2_all = np.empty((0,), dtype=np.float64)
        state.sim_primary_qr_all = combined_qr_local
        simulation_runtime_state.sim_miller1_all = np.asarray(m1, dtype=np.float64)
        simulation_runtime_state.sim_intens1_all = np.asarray(i1, dtype=np.float64)
        simulation_runtime_state.sim_primary_qr_all = combined_qr_local
        simulation_runtime_state.sim_miller2_all = np.empty((0, 3), dtype=np.float64)
        simulation_runtime_state.sim_intens2_all = np.empty((0,), dtype=np.float64)

    apply_bragg_qr_filters(trigger_update=False)
    state.df_summary, state.df_details = build_intensity_dataframes(
        state.miller,
        state.intensities,
        state.degeneracy,
        state.details,
    )

    simulation_runtime_state.last_sim_signature = None
    simulation_runtime_state.last_simulation_signature = None
    if trigger_update:
        schedule_update()
