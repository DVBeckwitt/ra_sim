"""Generate PbI2 HT-shifted CIFs from an active ordered CIF."""

from __future__ import annotations

import hashlib
import math
import re
import shlex
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence


DISORDERED_PHASE_SOURCE_LABEL = "disordered_phase"
DISORDERED_PHASE_DISPLAY_LABEL = "Disordered phase"

_GENERATOR_SCHEMA_ID = "ra_sim.pbi2_ht_shift_cif.v1"
_TOL = 1.0e-5
_HT_BASAL_SHIFT = (2.0 / 3.0, 1.0 / 3.0)


@dataclass(frozen=True)
class GeneratedDisorderedCif:
    cif_path: Path
    source_cif_path: Path
    source_signature: tuple[object, ...]
    phase_label: str
    source_label: str
    a: float
    c: float


@dataclass(frozen=True)
class _Atom:
    label: str
    species: str
    x: float
    y: float
    z: float
    occ: float = 1.0


@dataclass(frozen=True)
class _Cell:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float


@dataclass(frozen=True)
class _ParsedCif:
    data_name: str
    cell: _Cell
    symops: list[str]
    atoms_asym: list[_Atom]


def _clean_token(value: str) -> str:
    return str(value).strip().strip("'\"")


def _parse_fraction(value: str) -> float:
    text = _clean_token(value)
    if "/" in text and not any(ch in text for ch in ".eE"):
        return float(Fraction(text))
    return float(text)


def _parse_number(value: str) -> float:
    text = _clean_token(value)
    if text in {"?", "."}:
        raise ValueError(f"missing numeric value: {value!r}")
    text = re.sub(r"\([^)]+\)$", "", text)
    return _parse_fraction(text)


def _parse_optional_number(value: str, default: float = 1.0) -> float:
    text = _clean_token(value)
    return default if text in {"?", "."} else _parse_number(text)


def _wrap01(value: float, tol: float = _TOL) -> float:
    wrapped = float(value) % 1.0
    if abs(wrapped) < tol or abs(wrapped - 1.0) < tol:
        return 0.0
    for target in (1.0 / 3.0, 2.0 / 3.0):
        if abs(wrapped - target) < tol:
            return target
    return wrapped


def _periodic_delta(a_value: float, b_value: float) -> float:
    return (float(a_value) - float(b_value) + 0.5) % 1.0 - 0.5


def _strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    out: list[str] = []
    for ch in line:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out).strip()


def _read_cif_simple(path: Path) -> _ParsedCif:
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = [_strip_inline_comment(line) for line in raw]
    lines = [line for line in lines if line]

    data_name = "data_generated"
    cell_values: dict[str, str] = {}
    loops: list[tuple[list[str], list[list[str]]]] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        lower = line.lower()

        if lower.startswith("data_"):
            data_name = line
            i += 1
            continue

        if line.startswith("_"):
            parts = shlex.split(line, posix=True)
            if len(parts) >= 2 and parts[0].startswith("_cell_"):
                cell_values[parts[0]] = parts[1]
            i += 1
            continue

        if lower == "loop_":
            i += 1
            headers: list[str] = []
            while i < len(lines) and lines[i].startswith("_"):
                headers.append(shlex.split(lines[i], posix=True)[0])
                i += 1

            rows: list[list[str]] = []
            while i < len(lines):
                nxt = lines[i]
                nxt_lower = nxt.lower()
                if nxt_lower == "loop_" or nxt_lower.startswith("data_") or nxt.startswith("_"):
                    break
                fields = shlex.split(nxt, posix=True)
                if fields:
                    rows.append(fields)
                i += 1

            loops.append((headers, rows))
            continue

        i += 1

    required = (
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma",
    )
    missing = [key for key in required if key not in cell_values]
    if missing:
        raise ValueError(f"missing required cell field(s): {missing}")

    cell = _Cell(
        a=_parse_number(cell_values["_cell_length_a"]),
        b=_parse_number(cell_values["_cell_length_b"]),
        c=_parse_number(cell_values["_cell_length_c"]),
        alpha=_parse_number(cell_values["_cell_angle_alpha"]),
        beta=_parse_number(cell_values["_cell_angle_beta"]),
        gamma=_parse_number(cell_values["_cell_angle_gamma"]),
    )

    symops: list[str] = []
    atoms_asym: list[_Atom] = []
    for headers, rows in loops:
        header_set = set(headers)
        if "_space_group_symop_operation_xyz" in header_set:
            idx = headers.index("_space_group_symop_operation_xyz")
            symops.extend(_clean_token(row[idx]) for row in rows if idx < len(row))
            continue
        if "_symmetry_equiv_pos_as_xyz" in header_set:
            idx = headers.index("_symmetry_equiv_pos_as_xyz")
            symops.extend(_clean_token(row[idx]) for row in rows if idx < len(row))
            continue

        if {"_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"}.issubset(
            header_set
        ):
            ix = headers.index("_atom_site_fract_x")
            iy = headers.index("_atom_site_fract_y")
            iz = headers.index("_atom_site_fract_z")
            ilabel = headers.index("_atom_site_label") if "_atom_site_label" in header_set else None
            iocc = (
                headers.index("_atom_site_occupancy")
                if "_atom_site_occupancy" in header_set
                else None
            )
            ispecies = (
                headers.index("_atom_site_type_symbol")
                if "_atom_site_type_symbol" in header_set
                else None
            )

            for n, row in enumerate(rows, start=1):
                label = (
                    _clean_token(row[ilabel])
                    if ilabel is not None and ilabel < len(row)
                    else f"X{n}"
                )
                if ispecies is not None and ispecies < len(row):
                    species = _clean_token(row[ispecies])
                else:
                    match = re.match(r"[A-Za-z]+", label)
                    species = match.group(0) if match else "X"
                occ = (
                    _parse_optional_number(row[iocc], 1.0)
                    if iocc is not None and iocc < len(row)
                    else 1.0
                )
                atoms_asym.append(
                    _Atom(
                        label=label,
                        species=species,
                        x=_parse_number(row[ix]),
                        y=_parse_number(row[iy]),
                        z=_parse_number(row[iz]),
                        occ=occ,
                    )
                )

    if not symops:
        symops = ["x, y, z"]
    if not atoms_asym:
        raise ValueError("no atom_site fractional-coordinate loop found")

    return _ParsedCif(data_name=data_name, cell=cell, symops=symops, atoms_asym=atoms_asym)


def _eval_sym_expr(expr: str, x_value: float, y_value: float, z_value: float) -> float:
    expr = _clean_token(expr).replace(" ", "")
    if not expr:
        raise ValueError("empty symmetry expression")
    if expr[0] not in "+-":
        expr = "+" + expr

    value = 0.0
    for sign, term in re.findall(r"([+-])([^+-]+)", expr):
        sgn = 1.0 if sign == "+" else -1.0
        if term in {"x", "y", "z"}:
            term_value = {"x": x_value, "y": y_value, "z": z_value}[term]
        else:
            match = re.fullmatch(r"(\d+(?:/\d+)?|\d+\.\d+)\*?([xyz])", term)
            if match:
                term_value = _parse_fraction(match.group(1)) * {
                    "x": x_value,
                    "y": y_value,
                    "z": z_value,
                }[match.group(2)]
            else:
                term_value = _parse_fraction(term)
        value += sgn * term_value
    return _wrap01(value)


def _apply_symop(symop: str, atom: _Atom) -> _Atom:
    parts = [part.strip() for part in _clean_token(symop).split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected three coordinate expressions in symop {symop!r}")
    return _Atom(
        label=atom.label,
        species=atom.species,
        x=_eval_sym_expr(parts[0], atom.x, atom.y, atom.z),
        y=_eval_sym_expr(parts[1], atom.x, atom.y, atom.z),
        z=_eval_sym_expr(parts[2], atom.x, atom.y, atom.z),
        occ=atom.occ,
    )


def _same_position(a_atom: _Atom, b_atom: _Atom, tol: float = _TOL) -> bool:
    return (
        a_atom.species == b_atom.species
        and abs(_periodic_delta(a_atom.x, b_atom.x)) < tol
        and abs(_periodic_delta(a_atom.y, b_atom.y)) < tol
        and abs(_periodic_delta(a_atom.z, b_atom.z)) < tol
    )


def _expand_to_p1(parsed: _ParsedCif, tol: float = _TOL) -> list[_Atom]:
    atoms: list[_Atom] = []
    for atom in parsed.atoms_asym:
        for symop in parsed.symops:
            candidate = _apply_symop(symop, atom)
            if not any(_same_position(candidate, existing, tol=tol) for existing in atoms):
                atoms.append(candidate)
    atoms.sort(key=lambda atom: (atom.z, atom.species, atom.y, atom.x))
    return atoms


def _find_layer_origin_z(atoms: Sequence[_Atom], anchor_species: str) -> float:
    anchors = [atom.z for atom in atoms if atom.species.lower() == anchor_species.lower()]
    if not anchors:
        species = sorted({atom.species for atom in atoms})
        raise ValueError(f"anchor species {anchor_species!r} not found. Available species: {species}")
    return sorted(anchors)[0]


def _infer_internal_z(atoms: Sequence[_Atom], origin_z: float, anchor_species: str) -> float:
    offsets = [
        abs(_periodic_delta(atom.z, origin_z))
        for atom in atoms
        if atom.species.lower() != anchor_species.lower()
    ]
    offsets = [value for value in offsets if value > _TOL]
    if not offsets:
        raise ValueError("could not infer a non-anchor internal z offset")
    return min(offsets)


def _parse_sequence(sequence: str) -> list[int]:
    normalized = str(sequence).replace(",", "").replace(" ", "").replace("_", "")
    if not normalized or not re.fullmatch(r"[01]+", normalized):
        raise ValueError("sequence must contain only 0 and 1")
    return [int(ch) for ch in normalized]


def _offsets_from_sequence(
    layers: int,
    sequence: Sequence[int],
    slip: tuple[float, float],
) -> list[tuple[float, float]]:
    if len(sequence) != int(layers):
        raise ValueError(f"sequence length must be {int(layers)} for layers={int(layers)}")
    if sum(sequence) % 3 != 0:
        raise ValueError("periodic PbI2 closure requires the number of slip steps to be divisible by 3")

    tx, ty = slip
    ox = 0.0
    oy = 0.0
    offsets: list[tuple[float, float]] = []
    for layer in range(int(layers)):
        offsets.append((_wrap01(ox), _wrap01(oy)))
        if layer < int(layers) - 1 and sequence[layer]:
            ox += tx
            oy += ty
    return offsets


def _transform_layer_dz(dz_parent: float, internal_z: float | None) -> float:
    dz = -float(dz_parent)
    if internal_z is not None and abs(dz) > 1.0e-8:
        dz = math.copysign(float(internal_z), dz)
    return dz


def _build_explicit_supercell(
    base_atoms: Sequence[_Atom],
    layers: int,
    offsets: Sequence[tuple[float, float]],
    origin_z: float,
    internal_z: float | None,
) -> list[_Atom]:
    if len(offsets) != int(layers):
        raise ValueError("one basal offset is required per layer")

    prepared = []
    for atom in base_atoms:
        dz_parent = _periodic_delta(atom.z, origin_z)
        prepared.append((atom, _transform_layer_dz(dz_parent, internal_z)))

    out: list[_Atom] = []
    counts: dict[str, int] = {}
    for layer_idx in range(int(layers)):
        ox, oy = offsets[layer_idx]
        for atom, dz_out in prepared:
            species = atom.species
            counts[species] = counts.get(species, 0) + 1
            out.append(
                _Atom(
                    label=f"{species}{counts[species]}",
                    species=species,
                    x=_wrap01(atom.x + ox),
                    y=_wrap01(atom.y + oy),
                    z=_wrap01((layer_idx + dz_out) / int(layers)),
                    occ=atom.occ,
                )
            )
    out.sort(key=lambda atom: (atom.z, atom.species, atom.y, atom.x))
    return out


def _cell_volume(cell: _Cell) -> float:
    alpha = math.radians(cell.alpha)
    beta = math.radians(cell.beta)
    gamma = math.radians(cell.gamma)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    cg = math.cos(gamma)
    factor = math.sqrt(max(0.0, 1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg))
    return cell.a * cell.b * cell.c * factor


def _fmt(value: float, ndigits: int = 6) -> str:
    value = 0.0 if abs(float(value)) < 5.0e-12 else float(value)
    return f"{value:.{ndigits}f}"


def _positive_float_or_default(value: float | None, default: float, name: str) -> float:
    if value is None:
        return float(default)
    coerced = float(value)
    if not math.isfinite(coerced) or coerced <= 0.0:
        raise ValueError(f"{name} must be a positive finite value")
    return coerced


def _optional_fractional_z(value: float | None) -> float | None:
    if value is None:
        return None
    coerced = float(value)
    if not math.isfinite(coerced) or coerced < 0.0 or coerced > 1.0:
        raise ValueError("internal_z must be a finite fractional value between 0 and 1")
    return coerced


def _output_cell(
    parent: _Cell,
    layers: int,
    *,
    cell_a: float | None,
    cell_b: float | None,
    cell_c_per_layer: float | None,
) -> _Cell:
    a_value = _positive_float_or_default(cell_a, parent.a, "cell_a")
    b_value = _positive_float_or_default(cell_b, parent.b, "cell_b")
    c_per_layer = _positive_float_or_default(cell_c_per_layer, parent.c, "cell_c_per_layer")
    return _Cell(
        a=a_value,
        b=b_value,
        c=c_per_layer * int(layers),
        alpha=parent.alpha,
        beta=parent.beta,
        gamma=parent.gamma,
    )


def _write_atom_loop(handle, atoms: Sequence[_Atom]) -> None:
    handle.write("loop_\n")
    handle.write("   _atom_site_label\n")
    handle.write("   _atom_site_occupancy\n")
    handle.write("   _atom_site_fract_x\n")
    handle.write("   _atom_site_fract_y\n")
    handle.write("   _atom_site_fract_z\n")
    handle.write("   _atom_site_adp_type\n")
    handle.write("   _atom_site_U_iso_or_equiv\n")
    handle.write("   _atom_site_type_symbol\n")
    for atom in atoms:
        handle.write(
            f"   {atom.label:<8s} "
            f"{atom.occ:6.3f} "
            f"{_fmt(atom.x, 10):>14s} "
            f"{_fmt(atom.y, 10):>14s} "
            f"{_fmt(atom.z, 10):>14s} "
            f"   Uiso  ? {atom.species}\n"
        )


def _r3m_symops() -> list[str]:
    base = [
        "x, y, z",
        "-x, -y, -z",
        "-y, x-y, z",
        "y, -x+y, -z",
        "-x+y, -x, z",
        "x-y, x, -z",
        "y, x, -z",
        "-y, -x, z",
        "x-y, -y, -z",
        "-x+y, y, z",
        "-x, -x+y, -z",
        "x, x-y, z",
    ]
    translations = [("", "", ""), ("+2/3", "+1/3", "+1/3"), ("+1/3", "+2/3", "+2/3")]
    out: list[str] = []
    for tx, ty, tz in translations:
        for symop in base:
            x_expr, y_expr, z_expr = [part.strip() for part in symop.split(",")]
            out.append(f"{x_expr}{tx}, {y_expr}{ty}, {z_expr}{tz}")
    return out


def _write_p1_cif(
    path: Path,
    new_cell: _Cell,
    atoms: Sequence[_Atom],
    *,
    layers: int,
    sequence_text: str,
    offsets: Sequence[tuple[float, float]],
    internal_z: float | None,
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Generated by ra_sim.utils.pbi2_ht_shift_cif\n")
        handle.write("# Explicit P1 coordinate model for a PbI2 HT stacking sequence.\n")
        handle.write("# delta(h,k)=2*pi*(2h+k)/3; basal slip=(2/3,1/3).\n")
        handle.write("# layer_z_mode=mirror\n")
        handle.write(f"# layers: {int(layers)}\n")
        handle.write(f"# sequence: {sequence_text}\n")
        handle.write("# slip_vector_fractional_ab: (0.6666666667, 0.3333333333)\n")
        if internal_z is not None:
            handle.write(f"# internal_z_parent_fraction_override: {internal_z:.10f}\n")
        handle.write(
            "# layer_offsets: "
            + " ".join(f"{i}:({ox:.6f},{oy:.6f})" for i, (ox, oy) in enumerate(offsets))
            + "\n"
        )
        handle.write("\n")
        handle.write("data_PbI2_HT_shifted_P1\n\n")
        handle.write("_chemical_name_common                  'PbI2 HT shifted explicit stack'\n")
        handle.write(f"_cell_length_a                         {_fmt(new_cell.a, 10)}\n")
        handle.write(f"_cell_length_b                         {_fmt(new_cell.b, 10)}\n")
        handle.write(f"_cell_length_c                         {_fmt(new_cell.c, 10)}\n")
        handle.write(f"_cell_angle_alpha                      {_fmt(new_cell.alpha, 10)}\n")
        handle.write(f"_cell_angle_beta                       {_fmt(new_cell.beta, 10)}\n")
        handle.write(f"_cell_angle_gamma                      {_fmt(new_cell.gamma, 10)}\n")
        handle.write(f"_cell_volume                           {_fmt(_cell_volume(new_cell), 10)}\n")
        handle.write("_space_group_name_H-M_alt              'P 1'\n")
        handle.write("_space_group_IT_number                 1\n\n")
        handle.write("loop_\n")
        handle.write("_space_group_symop_operation_xyz\n")
        handle.write("   'x, y, z'\n\n")
        _write_atom_loop(handle, atoms)


def _write_compact_6h_cif(
    path: Path,
    new_cell: _Cell,
    internal_z_parent: float,
    *,
    anchor_species: str = "Pb",
    iodine_species: str = "I",
) -> None:
    z_compact = (1.0 - float(internal_z_parent)) / 3.0
    atoms = [
        _Atom(label="Pb1", species=anchor_species, x=0.0, y=0.0, z=0.0),
        _Atom(label="I1", species=iodine_species, x=0.0, y=0.0, z=z_compact),
    ]

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Generated by ra_sim.utils.pbi2_ht_shift_cif\n")
        handle.write("# Compact 6H R-3m setting equivalent to sequence 111 with layer_z_mode=mirror.\n")
        handle.write(
            "# HT basal step: (2/3,1/3) and R translations: "
            "(0,0,0), (2/3,1/3,1/3), (1/3,2/3,2/3).\n"
        )
        handle.write(f"# internal_z_parent_fraction: {float(internal_z_parent):.10f}\n")
        handle.write(f"# compact_iodine_z: {z_compact:.10f}\n")
        handle.write("\n")
        handle.write("data_PbI2_6H_R-3m_from_HT_shift\n\n")
        handle.write("_chemical_name_common                  'I2 Pb1'\n")
        handle.write(f"_cell_length_a                         {_fmt(new_cell.a, 10)}\n")
        handle.write(f"_cell_length_b                         {_fmt(new_cell.b, 10)}\n")
        handle.write(f"_cell_length_c                         {_fmt(new_cell.c, 10)}\n")
        handle.write(f"_cell_angle_alpha                      {_fmt(new_cell.alpha, 10)}\n")
        handle.write(f"_cell_angle_beta                       {_fmt(new_cell.beta, 10)}\n")
        handle.write(f"_cell_angle_gamma                      {_fmt(new_cell.gamma, 10)}\n")
        handle.write(f"_cell_volume                           {_fmt(_cell_volume(new_cell), 10)}\n")
        handle.write("_space_group_name_H-M_alt              'R -3 m'\n")
        handle.write("_space_group_IT_number                 166\n\n")
        handle.write("loop_\n")
        handle.write("_space_group_symop_operation_xyz\n")
        for symop in _r3m_symops():
            handle.write(f"   '{symop}'\n")
        handle.write("\n")
        _write_atom_loop(handle, atoms)


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature_float(value: float) -> float | None:
    coerced = float(value)
    if not math.isfinite(coerced):
        return None
    return round(coerced, 12)


def _output_hash(source_signature: tuple[object, ...]) -> str:
    return hashlib.sha256(repr(source_signature).encode("utf-8")).hexdigest()[:12]


def generate_pbii_ht_shifted_cif(
    source_cif: str | Path,
    output_dir: str | Path,
    *,
    mode: str = "compact_6h",
    layers: int = 3,
    sequence: str = "111",
    internal_z: float | None = None,
    cell_a: float | None = None,
    cell_b: float | None = None,
    cell_c_per_layer: float | None = None,
) -> GeneratedDisorderedCif:
    """Generate a PbI2 HT-shifted CIF and return its stable metadata."""

    source_path = Path(source_cif).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"CIF file not found: {source_path}")

    output_dir_path = Path(output_dir).expanduser()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_dir_path = output_dir_path.resolve()

    mode_norm = str(mode).strip().lower().replace("-", "_")
    if mode_norm not in {"compact_6h", "explicit_p1"}:
        raise ValueError("mode must be 'compact_6h' or 'explicit_p1'")

    layer_count = int(layers)
    if layer_count < 1:
        raise ValueError("layers must be at least 1")
    sequence_values = _parse_sequence(sequence)
    sequence_text = "".join(str(value) for value in sequence_values)
    internal_z_override = _optional_fractional_z(internal_z)

    if mode_norm == "compact_6h":
        if layer_count != 3:
            raise ValueError("compact_6h requires layers=3")
        if sequence_values != [1, 1, 1]:
            raise ValueError("compact_6h requires sequence='111'")

    parent = _read_cif_simple(source_path)
    base_atoms = _expand_to_p1(parent)
    origin_z = _find_layer_origin_z(base_atoms, "Pb")
    internal_z_parent = (
        internal_z_override
        if internal_z_override is not None
        else _infer_internal_z(base_atoms, origin_z, "Pb")
    )
    new_cell = _output_cell(
        parent.cell,
        layer_count,
        cell_a=cell_a,
        cell_b=cell_b,
        cell_c_per_layer=cell_c_per_layer,
    )
    source_hash = _source_sha256(source_path)
    cell_c_per_layer_resolved = new_cell.c / layer_count
    source_signature = (
        _GENERATOR_SCHEMA_ID,
        ("source_sha256", source_hash),
        ("mode", mode_norm),
        ("layers", layer_count),
        ("sequence", sequence_text),
        ("internal_z", _signature_float(internal_z_parent)),
        ("cell_a", _signature_float(new_cell.a)),
        ("cell_b", _signature_float(new_cell.b)),
        ("cell_c_per_layer", _signature_float(cell_c_per_layer_resolved)),
        ("cell_c", _signature_float(new_cell.c)),
        ("source_label", DISORDERED_PHASE_SOURCE_LABEL),
        ("phase_label", DISORDERED_PHASE_DISPLAY_LABEL),
    )
    hash12 = _output_hash(source_signature)
    output_path = output_dir_path / f"{source_path.stem}.{DISORDERED_PHASE_SOURCE_LABEL}.{hash12}.cif"

    if mode_norm == "compact_6h":
        _write_compact_6h_cif(output_path, new_cell, internal_z_parent)
    else:
        offsets = _offsets_from_sequence(layer_count, sequence_values, _HT_BASAL_SHIFT)
        atoms = _build_explicit_supercell(
            base_atoms,
            layer_count,
            offsets,
            origin_z,
            internal_z_override,
        )
        _write_p1_cif(
            output_path,
            new_cell,
            atoms,
            layers=layer_count,
            sequence_text=sequence_text,
            offsets=offsets,
            internal_z=internal_z_override,
        )

    return GeneratedDisorderedCif(
        cif_path=output_path,
        source_cif_path=source_path,
        source_signature=source_signature,
        phase_label=DISORDERED_PHASE_DISPLAY_LABEL,
        source_label=DISORDERED_PHASE_SOURCE_LABEL,
        a=float(new_cell.a),
        c=float(new_cell.c),
    )


__all__ = [
    "DISORDERED_PHASE_DISPLAY_LABEL",
    "DISORDERED_PHASE_SOURCE_LABEL",
    "GeneratedDisorderedCif",
    "generate_pbii_ht_shifted_cif",
]
