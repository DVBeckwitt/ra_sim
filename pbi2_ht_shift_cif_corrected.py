#!/usr/bin/env python3
"""
Generate PbI2 HT layer-shift CIFs from a parent 2H PbI2 CIF.

The key correction is the layer handedness.  The HT basal registry step

    delta(h,k) = 2*pi*(2*h + k)/3

corresponds to the direct-space step

    s = (2/3, 1/3, 0)

because a fractional translation contributes phase

    2*pi*(h*dx + k*dy + l*dz).

For the supplied 2H PbI2 setting, the 6H/R-3m all-slip limit uses the mirror of
that input I-Pb-I local z handedness.  Therefore the default layer mode is
"mirror".  With layers=3 and sequence=111, the explicit P1 output expands to
the same topology as the supplied PbI2_6H CIF.  The compact mode writes the
conventional R-3m cell directly.

Examples:

  # Compact 6H/R-3m cell inferred from the input 2H geometry.
  python pbi2_ht_shift_cif_corrected.py PbI2_2H.cif PbI2_6H_from_2H.cif --compact-6h

  # Compact cell matching the uploaded 6H metrics more closely.
  # internal-z=0.265 gives compact I z=(1-0.265)/3=0.245.
  python pbi2_ht_shift_cif_corrected.py PbI2_2H.cif PbI2_6H_target_like.cif \
      --compact-6h --internal-z 0.265 --cell-a 4.557 --cell-b 4.557 --cell-c-per-layer 6.979

  # Explicit P1 all-slip 3-layer version of the same 6H topology.
  python pbi2_ht_shift_cif_corrected.py PbI2_2H.cif PbI2_6H_P1.cif --layers 3 --sequence 111

  # Random finite HT representative.  RA-SIM convention: p_flip=1-p.
  python pbi2_ht_shift_cif_corrected.py PbI2_2H.cif PbI2_HT_N60.cif --layers 60 --p 0.35 --seed 7
"""

from __future__ import annotations

import argparse
import math
import random
import re
import shlex
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

TOL = 1e-5


@dataclass
class Atom:
    label: str
    species: str
    x: float
    y: float
    z: float
    occ: float = 1.0


@dataclass
class Cell:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float


@dataclass
class ParsedCif:
    data_name: str
    cell: Cell
    symops: List[str]
    atoms_asym: List[Atom]


def clean_token(value: str) -> str:
    return value.strip().strip("'\"")


def parse_fraction(value: str) -> float:
    s = clean_token(str(value))
    if "/" in s and not any(ch in s for ch in ".eE"):
        return float(Fraction(s))
    return float(s)


def parse_number(value: str) -> float:
    s = clean_token(value)
    if s in {"?", "."}:
        raise ValueError(f"missing numeric value: {value!r}")
    s = re.sub(r"\([^)]+\)$", "", s)
    return parse_fraction(s)


def parse_optional_number(value: str, default: float = 1.0) -> float:
    s = clean_token(value)
    return default if s in {"?", "."} else parse_number(s)


def wrap01(value: float, tol: float = TOL) -> float:
    v = value % 1.0
    if abs(v) < tol or abs(v - 1.0) < tol:
        return 0.0
    for target in (1.0 / 3.0, 2.0 / 3.0):
        if abs(v - target) < tol:
            return target
    return v


def periodic_delta(a: float, b: float) -> float:
    return (a - b + 0.5) % 1.0 - 0.5


def strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    out: List[str] = []
    for ch in line:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            break
        out.append(ch)
    return "".join(out).strip()


def read_cif_simple(path: Path) -> ParsedCif:
    raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = [strip_inline_comment(line) for line in raw]
    lines = [line for line in lines if line]

    data_name = "data_generated"
    cell_values = {}
    loops: List[Tuple[List[str], List[List[str]]]] = []

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
            headers: List[str] = []
            while i < len(lines) and lines[i].startswith("_"):
                headers.append(shlex.split(lines[i], posix=True)[0])
                i += 1

            rows: List[List[str]] = []
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

    required = [
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma",
    ]
    missing = [key for key in required if key not in cell_values]
    if missing:
        raise ValueError(f"missing required cell field(s): {missing}")

    cell = Cell(
        a=parse_number(cell_values["_cell_length_a"]),
        b=parse_number(cell_values["_cell_length_b"]),
        c=parse_number(cell_values["_cell_length_c"]),
        alpha=parse_number(cell_values["_cell_angle_alpha"]),
        beta=parse_number(cell_values["_cell_angle_beta"]),
        gamma=parse_number(cell_values["_cell_angle_gamma"]),
    )

    symops: List[str] = []
    atoms_asym: List[Atom] = []

    for headers, rows in loops:
        header_set = set(headers)
        if "_space_group_symop_operation_xyz" in header_set:
            idx = headers.index("_space_group_symop_operation_xyz")
            symops.extend(clean_token(row[idx]) for row in rows if idx < len(row))
            continue
        if "_symmetry_equiv_pos_as_xyz" in header_set:
            idx = headers.index("_symmetry_equiv_pos_as_xyz")
            symops.extend(clean_token(row[idx]) for row in rows if idx < len(row))
            continue

        if {"_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"}.issubset(header_set):
            ix = headers.index("_atom_site_fract_x")
            iy = headers.index("_atom_site_fract_y")
            iz = headers.index("_atom_site_fract_z")
            ilabel = headers.index("_atom_site_label") if "_atom_site_label" in header_set else None
            iocc = headers.index("_atom_site_occupancy") if "_atom_site_occupancy" in header_set else None
            ispecies = headers.index("_atom_site_type_symbol") if "_atom_site_type_symbol" in header_set else None

            for n, row in enumerate(rows, start=1):
                label = clean_token(row[ilabel]) if ilabel is not None and ilabel < len(row) else f"X{n}"
                if ispecies is not None and ispecies < len(row):
                    species = clean_token(row[ispecies])
                else:
                    m = re.match(r"[A-Za-z]+", label)
                    species = m.group(0) if m else "X"
                occ = parse_optional_number(row[iocc], 1.0) if iocc is not None and iocc < len(row) else 1.0
                atoms_asym.append(
                    Atom(
                        label=label,
                        species=species,
                        x=parse_number(row[ix]),
                        y=parse_number(row[iy]),
                        z=parse_number(row[iz]),
                        occ=occ,
                    )
                )

    if not symops:
        symops = ["x, y, z"]
    if not atoms_asym:
        raise ValueError("no atom_site fractional-coordinate loop found")

    return ParsedCif(data_name=data_name, cell=cell, symops=symops, atoms_asym=atoms_asym)


def eval_sym_expr(expr: str, x: float, y: float, z: float) -> float:
    expr = clean_token(expr).replace(" ", "")
    if not expr:
        raise ValueError("empty symmetry expression")
    if expr[0] not in "+-":
        expr = "+" + expr

    value = 0.0
    for sign, term in re.findall(r"([+-])([^+-]+)", expr):
        sgn = 1.0 if sign == "+" else -1.0
        if term in {"x", "y", "z"}:
            term_value = {"x": x, "y": y, "z": z}[term]
        else:
            m = re.fullmatch(r"(\d+(?:/\d+)?|\d+\.\d+)\*?([xyz])", term)
            if m:
                term_value = parse_fraction(m.group(1)) * {"x": x, "y": y, "z": z}[m.group(2)]
            else:
                term_value = parse_fraction(term)
        value += sgn * term_value
    return wrap01(value)


def apply_symop(symop: str, atom: Atom) -> Atom:
    parts = [part.strip() for part in clean_token(symop).split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected three coordinate expressions in symop {symop!r}")
    return Atom(
        label=atom.label,
        species=atom.species,
        x=eval_sym_expr(parts[0], atom.x, atom.y, atom.z),
        y=eval_sym_expr(parts[1], atom.x, atom.y, atom.z),
        z=eval_sym_expr(parts[2], atom.x, atom.y, atom.z),
        occ=atom.occ,
    )


def same_position(a: Atom, b: Atom, tol: float = TOL) -> bool:
    return (
        a.species == b.species
        and abs(periodic_delta(a.x, b.x)) < tol
        and abs(periodic_delta(a.y, b.y)) < tol
        and abs(periodic_delta(a.z, b.z)) < tol
    )


def expand_to_p1(parsed: ParsedCif, tol: float = TOL) -> List[Atom]:
    atoms: List[Atom] = []
    for atom in parsed.atoms_asym:
        for op in parsed.symops:
            candidate = apply_symop(op, atom)
            if not any(same_position(candidate, existing, tol=tol) for existing in atoms):
                atoms.append(candidate)
    atoms.sort(key=lambda a: (a.z, a.species, a.y, a.x))
    return atoms


def find_layer_origin_z(atoms: Sequence[Atom], anchor_species: str) -> float:
    anchors = [atom.z for atom in atoms if atom.species.lower() == anchor_species.lower()]
    if not anchors:
        species = sorted({atom.species for atom in atoms})
        raise ValueError(f"anchor species {anchor_species!r} not found. Available species: {species}")
    return sorted(anchors)[0]


def infer_internal_z(atoms: Sequence[Atom], origin_z: float, anchor_species: str) -> float:
    offsets = [
        abs(periodic_delta(atom.z, origin_z))
        for atom in atoms
        if atom.species.lower() != anchor_species.lower()
    ]
    offsets = [v for v in offsets if v > TOL]
    if not offsets:
        raise ValueError("could not infer a non-anchor internal z offset")
    return min(offsets)


def parse_sequence(sequence: str) -> List[int]:
    s = sequence.replace(",", "").replace(" ", "").replace("_", "")
    if not s or not re.fullmatch(r"[01]+", s):
        raise ValueError("sequence must contain only 0 and 1")
    return [int(ch) for ch in s]


def draw_transition_sequence(
    layers: int,
    slip_probability: float,
    seed: Optional[int],
    periodic: bool,
    max_tries: int = 100_000,
) -> List[int]:
    if not (0.0 <= slip_probability <= 1.0):
        raise ValueError("slip probability must be between 0 and 1")
    length = layers if periodic else max(layers - 1, 0)
    rng = random.Random(seed)
    for _ in range(max_tries):
        seq = [1 if rng.random() < slip_probability else 0 for _ in range(length)]
        if not periodic or sum(seq) % 3 == 0:
            return seq
    raise RuntimeError("could not draw a periodically closed sequence. Try a different seed, layers, or sequence")


def offsets_from_sequence(
    layers: int,
    sequence: Sequence[int],
    slip: Tuple[float, float],
    periodic: bool,
) -> List[Tuple[float, float]]:
    required = layers if periodic else max(layers - 1, 0)
    if len(sequence) != required:
        raise ValueError(
            f"sequence length must be {required} for layers={layers} in "
            f"{'periodic' if periodic else 'nonperiodic'} mode"
        )
    if periodic and sum(sequence) % 3 != 0:
        raise ValueError("periodic PbI2 closure requires the number of slip steps to be divisible by 3")

    tx, ty = slip
    ox = 0.0
    oy = 0.0
    offsets: List[Tuple[float, float]] = []
    for layer in range(layers):
        offsets.append((wrap01(ox), wrap01(oy)))
        if layer < layers - 1:
            if sequence[layer] not in {0, 1}:
                raise ValueError("sequence entries must be 0 or 1")
            if sequence[layer]:
                ox += tx
                oy += ty
    return offsets


def transform_layer_dz(dz_parent: float, mode: str, internal_z: Optional[float]) -> float:
    if mode == "as-input":
        dz = dz_parent
    elif mode == "mirror":
        dz = -dz_parent
    else:
        raise ValueError("layer z mode must be 'mirror' or 'as-input'")
    if internal_z is not None and abs(dz) > 1e-8:
        dz = math.copysign(internal_z, dz)
    return dz


def build_explicit_supercell(
    base_atoms: Sequence[Atom],
    layers: int,
    offsets: Sequence[Tuple[float, float]],
    origin_z: float,
    layer_z_mode: str,
    internal_z: Optional[float],
) -> List[Atom]:
    if len(offsets) != layers:
        raise ValueError("one basal offset is required per layer")

    prepared = []
    for atom in base_atoms:
        dz_parent = periodic_delta(atom.z, origin_z)
        dz_out = transform_layer_dz(dz_parent, layer_z_mode, internal_z)
        prepared.append((atom, dz_out))

    out: List[Atom] = []
    counts = {}
    for layer_idx in range(layers):
        ox, oy = offsets[layer_idx]
        for atom, dz_out in prepared:
            species = atom.species
            counts[species] = counts.get(species, 0) + 1
            out.append(
                Atom(
                    label=f"{species}{counts[species]}",
                    species=species,
                    x=wrap01(atom.x + ox),
                    y=wrap01(atom.y + oy),
                    z=wrap01((layer_idx + dz_out) / layers),
                    occ=atom.occ,
                )
            )
    out.sort(key=lambda a: (a.z, a.species, a.y, a.x))
    return out


def cell_volume(cell: Cell) -> float:
    alpha = math.radians(cell.alpha)
    beta = math.radians(cell.beta)
    gamma = math.radians(cell.gamma)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    cg = math.cos(gamma)
    factor = math.sqrt(max(0.0, 1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg))
    return cell.a * cell.b * cell.c * factor


def fmt(x: float, ndigits: int = 6) -> str:
    x = 0.0 if abs(x) < 5e-12 else x
    return f"{x:.{ndigits}f}"


def output_cell(parent: Cell, layers: int, args: argparse.Namespace) -> Cell:
    a = args.cell_a if args.cell_a is not None else parent.a
    b = args.cell_b if args.cell_b is not None else parent.b
    c_per_layer = args.cell_c_per_layer if args.cell_c_per_layer is not None else parent.c
    return Cell(a=a, b=b, c=c_per_layer * layers, alpha=parent.alpha, beta=parent.beta, gamma=parent.gamma)


def write_atom_loop(f, atoms: Sequence[Atom]) -> None:
    f.write("loop_\n")
    f.write("   _atom_site_label\n")
    f.write("   _atom_site_occupancy\n")
    f.write("   _atom_site_fract_x\n")
    f.write("   _atom_site_fract_y\n")
    f.write("   _atom_site_fract_z\n")
    f.write("   _atom_site_adp_type\n")
    f.write("   _atom_site_U_iso_or_equiv\n")
    f.write("   _atom_site_type_symbol\n")
    for atom in atoms:
        f.write(
            f"   {atom.label:<8s} "
            f"{atom.occ:6.3f} "
            f"{fmt(atom.x, 10):>14s} "
            f"{fmt(atom.y, 10):>14s} "
            f"{fmt(atom.z, 10):>14s} "
            f"   Uiso  ? {atom.species}\n"
        )


def write_p1_cif(
    path: Path,
    parent: ParsedCif,
    atoms: Sequence[Atom],
    layers: int,
    sequence: Sequence[int],
    offsets: Sequence[Tuple[float, float]],
    slip: Tuple[float, float],
    layer_z_mode: str,
    internal_z: Optional[float],
    args: argparse.Namespace,
) -> None:
    new_cell = output_cell(parent.cell, layers, args)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Generated by pbi2_ht_shift_cif_corrected.py\n")
        f.write("# Explicit P1 coordinate model for a PbI2 HT stacking sequence.\n")
        f.write("# delta(h,k)=2*pi*(2h+k)/3; basal slip=(2/3,1/3).\n")
        f.write("# layer_z_mode=mirror is required to recover the uploaded 6H topology from the supplied 2H CIF.\n")
        f.write(f"# layers: {layers}\n")
        f.write(f"# sequence: {''.join(str(x) for x in sequence)}\n")
        f.write(f"# slip_vector_fractional_ab: ({slip[0]:.10f}, {slip[1]:.10f})\n")
        f.write(f"# layer_z_mode: {layer_z_mode}\n")
        if internal_z is not None:
            f.write(f"# internal_z_parent_fraction_override: {internal_z:.10f}\n")
        f.write("# layer_offsets: " + " ".join(f"{i}:({ox:.6f},{oy:.6f})" for i, (ox, oy) in enumerate(offsets)) + "\n")
        f.write("\n")
        f.write("data_PbI2_HT_shifted_P1\n\n")
        f.write("_chemical_name_common                  'PbI2 HT shifted explicit stack'\n")
        f.write(f"_cell_length_a                         {fmt(new_cell.a, 10)}\n")
        f.write(f"_cell_length_b                         {fmt(new_cell.b, 10)}\n")
        f.write(f"_cell_length_c                         {fmt(new_cell.c, 10)}\n")
        f.write(f"_cell_angle_alpha                      {fmt(new_cell.alpha, 10)}\n")
        f.write(f"_cell_angle_beta                       {fmt(new_cell.beta, 10)}\n")
        f.write(f"_cell_angle_gamma                      {fmt(new_cell.gamma, 10)}\n")
        f.write(f"_cell_volume                           {fmt(cell_volume(new_cell), 10)}\n")
        f.write("_space_group_name_H-M_alt              'P 1'\n")
        f.write("_space_group_IT_number                 1\n\n")
        f.write("loop_\n")
        f.write("_space_group_symop_operation_xyz\n")
        f.write("   'x, y, z'\n\n")
        write_atom_loop(f, atoms)


def r3m_symops() -> List[str]:
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
    out: List[str] = []
    for tx, ty, tz in translations:
        for op in base:
            x, y, z = [part.strip() for part in op.split(",")]
            out.append(f"{x}{tx}, {y}{ty}, {z}{tz}")
    return out


def write_compact_6h_cif(
    path: Path,
    parent: ParsedCif,
    internal_z_parent: float,
    args: argparse.Namespace,
) -> None:
    new_cell = output_cell(parent.cell, 3, args)
    z_compact = (1.0 - internal_z_parent) / 3.0
    atoms = [
        Atom(label="Pb1", species=args.anchor_species, x=0.0, y=0.0, z=0.0),
        Atom(label="I1", species=args.iodine_species, x=0.0, y=0.0, z=z_compact),
    ]

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Generated by pbi2_ht_shift_cif_corrected.py\n")
        f.write("# Compact 6H R-3m setting equivalent to sequence 111 with layer_z_mode=mirror.\n")
        f.write("# HT basal step: (2/3,1/3) and R translations: (0,0,0), (2/3,1/3,1/3), (1/3,2/3,2/3).\n")
        f.write(f"# internal_z_parent_fraction: {internal_z_parent:.10f}\n")
        f.write(f"# compact_iodine_z: {z_compact:.10f}\n")
        f.write("\n")
        f.write("data_PbI2_6H_R-3m_from_HT_shift\n\n")
        f.write("_chemical_name_common                  'I2 Pb1'\n")
        f.write(f"_cell_length_a                         {fmt(new_cell.a, 10)}\n")
        f.write(f"_cell_length_b                         {fmt(new_cell.b, 10)}\n")
        f.write(f"_cell_length_c                         {fmt(new_cell.c, 10)}\n")
        f.write(f"_cell_angle_alpha                      {fmt(new_cell.alpha, 10)}\n")
        f.write(f"_cell_angle_beta                       {fmt(new_cell.beta, 10)}\n")
        f.write(f"_cell_angle_gamma                      {fmt(new_cell.gamma, 10)}\n")
        f.write(f"_cell_volume                           {fmt(cell_volume(new_cell), 10)}\n")
        f.write("_space_group_name_H-M_alt              'R -3 m'\n")
        f.write("_space_group_IT_number                 166\n\n")
        f.write("loop_\n")
        f.write("_space_group_symop_operation_xyz\n")
        for op in r3m_symops():
            f.write(f"   '{op}'\n")
        f.write("\n")
        write_atom_loop(f, atoms)


def analytical_ht_factor(
    h: int,
    k: int,
    L: float,
    p_user: float,
    phi_l_div: float = 1.0,
    clamp: float = 1e-9,
    finite_N: Optional[int] = None,
) -> float:
    p_flip = 1.0 - p_user
    delta = 2.0 * math.pi * ((2.0 * h + k) / 3.0)
    z_real = (1.0 - p_flip) + p_flip * math.cos(delta)
    z_imag = p_flip * math.sin(delta)
    f = min(math.hypot(z_real, z_imag), 1.0 - clamp)
    psi = math.atan2(z_imag, z_real)
    phi = delta + 2.0 * math.pi * L / phi_l_div

    if finite_N is None:
        return (1.0 - f * f) / (1.0 + f * f - 2.0 * f * math.cos(phi - psi))

    total = 1.0
    theta = phi - psi
    for n in range(1, finite_N):
        total += (2.0 / finite_N) * (finite_N - n) * (f ** n) * math.cos(n * theta)
    return total


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate PbI2 HT-shifted explicit or compact 6H CIFs.")
    p.add_argument("input_cif", type=Path)
    p.add_argument("output_cif", type=Path)
    p.add_argument("--layers", type=int, default=None, help="number of PbI2 layers in the output c-axis repeat")
    p.add_argument("--sequence", type=str, default=None, help="0/1 transition sequence. Periodic length must equal layers")
    p.add_argument("--p", type=float, default=None, help="RA-SIM user-facing p. Internal slip probability is 1-p")
    p.add_argument("--slip-prob", type=float, default=None, help="direct slip probability. Overrides --p for random sequence generation")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-periodic", action="store_true", help="use layers-1 transitions and skip closure check")
    p.add_argument("--slip-vector", nargs=2, default=("2/3", "1/3"), metavar=("DX", "DY"))
    p.add_argument("--anchor-species", default="Pb")
    p.add_argument("--iodine-species", default="I")
    p.add_argument("--layer-origin-z", type=float, default=None)
    p.add_argument(
        "--layer-z-mode",
        choices=("mirror", "as-input"),
        default="mirror",
        help="mirror maps the supplied 2H setting to the supplied 6H topology. as-input reproduces the old rigid-copy behavior",
    )
    p.add_argument(
        "--internal-z",
        type=float,
        default=None,
        help="optional non-anchor z offset in parent-cell fractional units. Example: 0.265 gives compact I z=0.245",
    )
    p.add_argument("--compact-6h", action="store_true", help="write compact R-3m CIF for the all-slip 3-layer limit")
    p.add_argument("--cell-a", type=float, default=None, help="optional override for output a")
    p.add_argument("--cell-b", type=float, default=None, help="optional override for output b")
    p.add_argument("--cell-c-per-layer", type=float, default=None, help="optional override for per-layer c before multiplying by layers")
    p.add_argument("--print-ht-check", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    parent = read_cif_simple(args.input_cif)
    base_atoms = expand_to_p1(parent)
    origin_z = wrap01(args.layer_origin_z) if args.layer_origin_z is not None else find_layer_origin_z(base_atoms, args.anchor_species)
    internal_z_parent = args.internal_z if args.internal_z is not None else infer_internal_z(base_atoms, origin_z, args.anchor_species)

    if args.compact_6h:
        if args.no_periodic:
            raise ValueError("--compact-6h is a periodic all-slip 3-layer structure; do not use --no-periodic")
        if args.layers is not None and args.layers != 3:
            raise ValueError("--compact-6h requires --layers 3 if --layers is supplied")
        if args.sequence is not None and parse_sequence(args.sequence) != [1, 1, 1]:
            raise ValueError("--compact-6h requires sequence 111 if --sequence is supplied")
        if args.layer_z_mode != "mirror":
            raise ValueError("--compact-6h requires --layer-z-mode mirror")
        write_compact_6h_cif(args.output_cif, parent, internal_z_parent, args)
        print(f"Wrote compact 6H R-3m CIF: {args.output_cif}", file=sys.stderr)
        print(f"Compact I z = {(1.0 - internal_z_parent) / 3.0:.10f}", file=sys.stderr)
        return 0

    if args.layers is None:
        raise ValueError("provide --layers unless --compact-6h is used")
    if args.layers < 1:
        raise ValueError("--layers must be at least 1")

    periodic = not args.no_periodic
    slip = (parse_fraction(args.slip_vector[0]), parse_fraction(args.slip_vector[1]))

    if args.sequence is not None:
        sequence = parse_sequence(args.sequence)
        if args.slip_prob is not None:
            slip_probability = args.slip_prob
        elif args.p is not None:
            if not (0.0 <= args.p <= 1.0):
                raise ValueError("--p must be between 0 and 1")
            slip_probability = 1.0 - args.p
        else:
            slip_probability = sum(sequence) / len(sequence) if sequence else 0.0
    else:
        if args.slip_prob is not None:
            slip_probability = args.slip_prob
        elif args.p is not None:
            if not (0.0 <= args.p <= 1.0):
                raise ValueError("--p must be between 0 and 1")
            slip_probability = 1.0 - args.p
        else:
            raise ValueError("provide --sequence, --p, or --slip-prob")
        sequence = draw_transition_sequence(args.layers, slip_probability, args.seed, periodic)

    offsets = offsets_from_sequence(args.layers, sequence, slip, periodic)
    atoms = build_explicit_supercell(
        base_atoms=base_atoms,
        layers=args.layers,
        offsets=offsets,
        origin_z=origin_z,
        layer_z_mode=args.layer_z_mode,
        internal_z=args.internal_z,
    )
    write_p1_cif(
        args.output_cif,
        parent,
        atoms,
        args.layers,
        sequence,
        offsets,
        slip,
        args.layer_z_mode,
        args.internal_z,
        args,
    )
    print(f"Wrote explicit P1 CIF: {args.output_cif}", file=sys.stderr)
    print(f"Expanded parent cell: {len(base_atoms)} atoms. Output atoms: {len(atoms)}.", file=sys.stderr)
    print(f"Slip probability represented or requested: {slip_probability:.10f}", file=sys.stderr)
    print(f"Layer z mode: {args.layer_z_mode}. Internal z parent fraction: {internal_z_parent:.10f}", file=sys.stderr)
    if periodic:
        print(f"Periodic slip count: {sum(sequence)}. Closure modulo 3: {sum(sequence) % 3}.", file=sys.stderr)

    if args.print_ht_check and args.p is not None:
        for L in [0.0, 0.25, 0.5, 0.75, 1.0]:
            print(f"HT R_inf h=1 k=0 L={L:.2f}: {analytical_ht_factor(1, 0, L, args.p):.8g}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
