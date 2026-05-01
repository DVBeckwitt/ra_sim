from __future__ import annotations

import importlib
import sys
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_2H = ROOT / "tests" / "Diffuse" / "PbI2_2H.cif"
REFERENCE_6H = ROOT / "tests" / "Diffuse" / "PbI2_6H.cif"


def _module():
    return importlib.import_module("ra_sim.utils.pbi2_ht_shift_cif")


def _gui_modules_loaded() -> set[str]:
    return {name for name in sys.modules if name == "ra_sim.gui" or name.startswith("ra_sim.gui.")}


def _atom_counter(atoms) -> Counter[tuple[object, ...]]:
    return Counter(
        (
            atom.species,
            round(float(atom.x) % 1.0, 8),
            round(float(atom.y) % 1.0, 8),
            round(float(atom.z) % 1.0, 8),
            round(float(atom.occ), 8),
        )
        for atom in atoms
    )


def test_generate_pbii_ht_shifted_cif_writes_file(tmp_path):
    before_gui_modules = _gui_modules_loaded()
    mod = _module()
    after_gui_modules = _gui_modules_loaded()

    assert after_gui_modules == before_gui_modules

    generated = mod.generate_pbii_ht_shifted_cif(SOURCE_2H, tmp_path)

    assert generated.cif_path.is_file()
    assert generated.cif_path.parent == tmp_path.resolve()
    assert generated.cif_path.name.startswith("PbI2_2H.disordered_phase.")
    assert generated.cif_path.suffix == ".cif"
    assert generated.source_cif_path == SOURCE_2H.resolve()
    assert generated.source_signature[0] == "ra_sim.pbi2_ht_shift_cif.v1"

    regenerated = mod.generate_pbii_ht_shifted_cif(SOURCE_2H, tmp_path)
    assert regenerated.cif_path == generated.cif_path
    assert regenerated.source_signature == generated.source_signature


def test_generated_compact_6h_has_expected_cell_scaling(tmp_path):
    mod = _module()

    generated = mod.generate_pbii_ht_shifted_cif(SOURCE_2H, tmp_path)
    source = mod._read_cif_simple(SOURCE_2H)
    generated_cif = mod._read_cif_simple(generated.cif_path)

    assert generated_cif.cell.a == pytest.approx(source.cell.a)
    assert generated_cif.cell.b == pytest.approx(source.cell.b)
    assert generated_cif.cell.c == pytest.approx(3.0 * source.cell.c)
    assert generated.a == pytest.approx(source.cell.a)
    assert generated.c == pytest.approx(3.0 * source.cell.c)


def test_generated_compact_6h_has_expected_source_label(tmp_path):
    mod = _module()

    generated = mod.generate_pbii_ht_shifted_cif(SOURCE_2H, tmp_path)

    assert generated.source_label == mod.DISORDERED_PHASE_SOURCE_LABEL
    assert generated.phase_label == mod.DISORDERED_PHASE_DISPLAY_LABEL
    assert ("source_label", "disordered_phase") in generated.source_signature
    assert ("phase_label", "Disordered phase") in generated.source_signature


def test_generated_compact_6h_is_structurally_equivalent_to_reference(tmp_path):
    mod = _module()

    generated = mod.generate_pbii_ht_shifted_cif(
        SOURCE_2H,
        tmp_path,
        internal_z=0.265,
        cell_a=4.557,
        cell_b=4.557,
        cell_c_per_layer=6.979,
    )
    generated_cif = mod._read_cif_simple(generated.cif_path)
    reference_cif = mod._read_cif_simple(REFERENCE_6H)

    assert generated_cif.cell.a == pytest.approx(reference_cif.cell.a, abs=1.0e-8)
    assert generated_cif.cell.b == pytest.approx(reference_cif.cell.b, abs=1.0e-8)
    assert generated_cif.cell.c == pytest.approx(reference_cif.cell.c, abs=1.0e-8)
    assert generated_cif.cell.alpha == pytest.approx(reference_cif.cell.alpha, abs=1.0e-8)
    assert generated_cif.cell.beta == pytest.approx(reference_cif.cell.beta, abs=1.0e-8)
    assert generated_cif.cell.gamma == pytest.approx(reference_cif.cell.gamma, abs=1.0e-8)

    generated_atoms = mod._expand_to_p1(generated_cif)
    reference_atoms = mod._expand_to_p1(reference_cif)

    assert len(generated_atoms) == len(reference_atoms)
    assert _atom_counter(generated_atoms) == _atom_counter(reference_atoms)
