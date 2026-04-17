"""Utilities for expanding atomic coordinates via symmetry operations."""

from pathlib import Path

import numpy as np

_SITE_TOLERANCE = 1e-8


def _is_duplicate_site(candidate, existing_positions, tol):
    for existing in existing_positions:
        delta = np.abs(candidate - existing)
        if np.all(np.minimum(delta, 1.0 - delta) <= tol):
            return True
    return False


def get_atomic_coordinates(
    lattice,
    positions,
    numbers,
    space_group_operations,
    atomic_labels,
    cell_params,
):
    del lattice, numbers

    atoms = []
    unique_positions_by_label = {}
    rotations = space_group_operations["rotations"]
    translations = space_group_operations["translations"]

    # Apply space group symmetry operations, wrap into unit cell, drop same-label duplicates.
    for i, pos in enumerate(positions):
        label = atomic_labels[i]
        unique_positions = unique_positions_by_label.setdefault(label, [])
        for rotation, translation in zip(rotations, translations):
            new_pos = np.mod(np.dot(pos, np.transpose(rotation)) + translation, 1.0)
            if _is_duplicate_site(new_pos, unique_positions, _SITE_TOLERANCE):
                continue
            unique_positions.append(new_pos)
            atoms.append((label, new_pos[0], new_pos[1], new_pos[2]))

    # Use cell parameters provided from the CIF file
    a, b, c, alpha, beta, gamma = cell_params

    return (a, b, c, alpha, beta, gamma), atoms


def _resolve_xtl_title(filename):
    stem = Path(filename).stem if filename else ""
    return stem or "output"


def _derive_xtl_symmetry_metadata(lattice, positions, numbers):
    try:
        import spglib
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "automatic symmetry derivation requires spglib, or the caller must pass "
            "symmetry_number/symmetry_label explicitly."
        ) from exc

    dataset = spglib.get_symmetry_dataset(
        (
            np.asarray(lattice, dtype=float),
            np.asarray(positions, dtype=float),
            np.asarray(numbers, dtype=int),
        )
    )

    if dataset is None:
        raise RuntimeError("spglib could not derive symmetry metadata from the supplied structure.")

    symmetry_number = ""
    if int(dataset.number) != 1:
        symmetry_number = str(int(dataset.number))

    symmetry_label = str(dataset.international)
    if symmetry_label.replace(" ", "") == "P1":
        symmetry_label = ""

    return symmetry_number, symmetry_label


def write_xtl(
    lattice,
    positions,
    numbers,
    space_group_operations,
    atomic_labels,
    cell_params,
    filename="output.xtl",
    *,
    title=None,
    symmetry_number=None,
    symmetry_label=None,
):
    (a, b, c, alpha, beta, gamma), atoms = get_atomic_coordinates(
        lattice,
        positions,
        numbers,
        space_group_operations,
        atomic_labels,
        cell_params,
    )
    resolved_title = _resolve_xtl_title(filename) if title is None else str(title)
    derived_symmetry_number = ""
    derived_symmetry_label = ""
    if symmetry_number is None or symmetry_label is None:
        derived_symmetry_number, derived_symmetry_label = _derive_xtl_symmetry_metadata(
            lattice,
            positions,
            numbers,
        )
    resolved_symmetry_number = (
        derived_symmetry_number if symmetry_number is None else str(symmetry_number)
    )
    resolved_symmetry_label = (
        derived_symmetry_label if symmetry_label is None else str(symmetry_label)
    )

    # Write to the .xtl file
    with open(filename, "w") as f:
        f.write(f"TITLE {resolved_title}\n")
        f.write("CELL\n")
        f.write(f"  {a:.6f}   {b:.6f}  {c:.6f}  {alpha:.6f}  {beta:.6f}  {gamma:.6f}\n")
        f.write(f"SYMMETRY NUMBER {resolved_symmetry_number}\n")
        f.write(f"SYMMETRY LABEL  {resolved_symmetry_label}\n")
        f.write("ATOMS\n")
        f.write("NAME         X           Y           Z\n")
        for atom_type, x, y, z in atoms:
            f.write("{:<12s}{:11.6f}{:11.6f}{:11.6f}\n".format(atom_type, x, y, z))
        f.write("EOF\n")
