import builtins
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ra_sim.cli import _parse_cif_cell_a_c
from ra_sim.StructureFactor.AtomicCoordinates import get_atomic_coordinates, write_xtl


def _simple_cubic_structure():
    return (
        np.eye(3),
        np.array([[0.0, 0.0, 0.0]]),
        [11],
        {
            "rotations": [np.eye(3, dtype=int)],
            "translations": [np.zeros(3)],
        },
        ["Na"],
        (1.0, 1.0, 1.0, 90.0, 90.0, 90.0),
    )


def _low_symmetry_p1_structure():
    return (
        np.array(
            [
                [1.0, 0.1, 0.2],
                [0.0, 1.1, 0.3],
                [0.0, 0.0, 1.2],
            ]
        ),
        np.array(
            [
                [0.123, 0.234, 0.345],
                [0.111, 0.222, 0.333],
            ]
        ),
        [11, 12],
        {
            "rotations": [np.eye(3, dtype=int)],
            "translations": [np.zeros(3)],
        },
        ["Na", "Mg"],
        (1.0, 1.1, 1.2, 75.0, 80.0, 85.0),
    )


def _block_spglib_import(monkeypatch):
    original_import = builtins.__import__

    def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "spglib":
            raise ModuleNotFoundError("No module named 'spglib'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "spglib", raising=False)
    monkeypatch.setattr(builtins, "__import__", _blocked_import)


def _block_ciffile_import(monkeypatch):
    original_import = builtins.__import__

    def _blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "CifFile":
            raise ModuleNotFoundError("No module named 'CifFile'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "CifFile", raising=False)
    monkeypatch.setattr(builtins, "__import__", _blocked_import)


def test_parse_cif_cell_a_c_reads_raw_values():
    a_val, c_val = _parse_cif_cell_a_c(str(Path("tests/local_test.cif")))
    assert a_val == pytest.approx(4.0)
    assert c_val == pytest.approx(10.0)


def test_parse_cif_cell_a_c_handles_uncertainty_suffix(tmp_path):
    cif_path = tmp_path / "uncertainty.cif"
    cif_path.write_text(
        "\n".join(
            [
                "data_test",
                "_cell_length_a    4.123(4)",
                "_cell_length_c    30.456(7)",
            ]
        ),
        encoding="utf-8",
    )
    a_val, c_val = _parse_cif_cell_a_c(str(cif_path))
    assert a_val == pytest.approx(4.123)
    assert c_val == pytest.approx(30.456)


def test_parse_cif_cell_a_c_falls_back_without_pycifrw(tmp_path, monkeypatch):
    cif_path = tmp_path / "fallback.cif"
    cif_path.write_text(
        "\n".join(
            [
                "data_test",
                "_cell_length_a    4.123(4)",
                "_cell_length_c    30.456(7)",
            ]
        ),
        encoding="utf-8",
    )
    _block_ciffile_import(monkeypatch)

    a_val, c_val = _parse_cif_cell_a_c(str(cif_path))

    assert a_val == pytest.approx(4.123)
    assert c_val == pytest.approx(30.456)


def test_parse_cif_cell_a_c_fallback_accepts_quoted_scalars(tmp_path, monkeypatch):
    cif_path = tmp_path / "quoted.cif"
    cif_path.write_text(
        "\n".join(
            [
                "data_test",
                "_cell_length_a    '4.123(4)'",
                '_cell_length_c    "30.456(7)"',
            ]
        ),
        encoding="utf-8",
    )
    _block_ciffile_import(monkeypatch)

    a_val, c_val = _parse_cif_cell_a_c(str(cif_path))

    assert a_val == pytest.approx(4.123)
    assert c_val == pytest.approx(30.456)


def test_parse_cif_cell_a_c_fallback_accepts_quoted_scalars_with_comments(
    tmp_path,
    monkeypatch,
):
    cif_path = tmp_path / "quoted-comments.cif"
    cif_path.write_text(
        "\n".join(
            [
                "data_test",
                "_cell_length_a    '4.123(4)' # comment",
                '_cell_length_c    "30.456(7)" # another comment',
            ]
        ),
        encoding="utf-8",
    )
    _block_ciffile_import(monkeypatch)

    a_val, c_val = _parse_cif_cell_a_c(str(cif_path))

    assert a_val == pytest.approx(4.123)
    assert c_val == pytest.approx(30.456)


def test_parse_cif_cell_a_c_fallback_ignores_semicolon_text_fields(tmp_path, monkeypatch):
    cif_path = tmp_path / "semicolon-text.cif"
    cif_path.write_text(
        "\n".join(
            [
                "data_test",
                "_audit_note",
                ";",
                "_cell_length_a 999",
                "_cell_length_c 888",
                ";",
                "_cell_length_a    4.123(4)",
                "_cell_length_c    30.456(7)",
            ]
        ),
        encoding="utf-8",
    )
    _block_ciffile_import(monkeypatch)

    a_val, c_val = _parse_cif_cell_a_c(str(cif_path))

    assert a_val == pytest.approx(4.123)
    assert c_val == pytest.approx(30.456)


def test_parse_cif_cell_a_c_fallback_accepts_split_line_scalars(tmp_path, monkeypatch):
    cif_path = tmp_path / "split-line.cif"
    cif_path.write_text(
        "\n".join(
            [
                "data_test",
                "_cell_length_a",
                "  4.123(4)",
                "_cell_length_c",
                "  30.456(7)",
            ]
        ),
        encoding="utf-8",
    )
    _block_ciffile_import(monkeypatch)

    a_val, c_val = _parse_cif_cell_a_c(str(cif_path))

    assert a_val == pytest.approx(4.123)
    assert c_val == pytest.approx(30.456)


def test_parse_cif_cell_a_c_fallback_accepts_split_line_scalars_after_tag_comments(
    tmp_path,
    monkeypatch,
):
    cif_path = tmp_path / "split-line-tag-comments.cif"
    cif_path.write_text(
        "\n".join(
            [
                "data_test",
                "_cell_length_a # comment",
                "  4.123(4)",
                "_cell_length_c # another comment",
                "  30.456(7)",
            ]
        ),
        encoding="utf-8",
    )
    _block_ciffile_import(monkeypatch)

    a_val, c_val = _parse_cif_cell_a_c(str(cif_path))

    assert a_val == pytest.approx(4.123)
    assert c_val == pytest.approx(30.456)


def test_write_xtl_uses_filename_and_derived_symmetry_by_default(tmp_path):
    xtl_path = tmp_path / "sodium.xtl"

    write_xtl(*_simple_cubic_structure(), filename=xtl_path)

    content = xtl_path.read_text(encoding="utf-8")
    assert "TITLE sodium\n" in content
    assert "TITLE Bi2 Se3\n" not in content
    assert "SYMMETRY NUMBER 221\n" in content
    assert "SYMMETRY LABEL  Pm-3m\n" in content
    assert "SYMMETRY LABEL  P1\n" not in content


def test_write_xtl_allows_explicit_legacy_metadata_override(tmp_path):
    xtl_path = tmp_path / "override.xtl"

    write_xtl(
        *_simple_cubic_structure(),
        filename=xtl_path,
        title="Bi2 Se3",
        symmetry_number=1,
        symmetry_label="P1",
    )

    content = xtl_path.read_text(encoding="utf-8")
    assert "TITLE Bi2 Se3\n" in content
    assert "SYMMETRY NUMBER 1\n" in content
    assert "SYMMETRY LABEL  P1\n" in content


def test_write_xtl_explicit_override_skips_spglib_requirement(tmp_path, monkeypatch):
    xtl_path = tmp_path / "override-no-spglib.xtl"
    _block_spglib_import(monkeypatch)

    write_xtl(
        *_simple_cubic_structure(),
        filename=xtl_path,
        title="Bi2 Se3",
        symmetry_number=1,
        symmetry_label="P1",
    )

    content = xtl_path.read_text(encoding="utf-8")
    assert "TITLE Bi2 Se3\n" in content
    assert "SYMMETRY NUMBER 1\n" in content
    assert "SYMMETRY LABEL  P1\n" in content


def test_write_xtl_preserves_explicit_empty_metadata(tmp_path):
    xtl_path = tmp_path / "ignored.xtl"

    write_xtl(
        *_simple_cubic_structure(),
        filename=xtl_path,
        title="",
        symmetry_number="",
        symmetry_label="",
    )

    content = xtl_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    assert lines[0] == "TITLE "
    assert lines[3] == "SYMMETRY NUMBER "
    assert lines[4] == "SYMMETRY LABEL  "
    assert "TITLE ignored\n" not in content
    assert "SYMMETRY NUMBER 221\n" not in content
    assert "SYMMETRY LABEL  Pm-3m\n" not in content


def test_write_xtl_blanks_auto_derived_p1_metadata(tmp_path):
    xtl_path = tmp_path / "triclinic.xtl"

    write_xtl(*_low_symmetry_p1_structure(), filename=xtl_path)

    content = xtl_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    assert lines[0] == "TITLE triclinic"
    assert lines[3] == "SYMMETRY NUMBER "
    assert lines[4] == "SYMMETRY LABEL  "
    assert "SYMMETRY NUMBER 1\n" not in content
    assert "SYMMETRY LABEL  P1\n" not in content


def test_write_xtl_missing_spglib_raises_clear_error(tmp_path, monkeypatch):
    xtl_path = tmp_path / "missing-spglib.xtl"
    _block_spglib_import(monkeypatch)

    with pytest.raises(RuntimeError, match="automatic symmetry derivation requires spglib"):
        write_xtl(*_simple_cubic_structure(), filename=xtl_path)


@pytest.mark.parametrize(
    "metadata_override",
    [
        {"symmetry_number": "221"},
        {"symmetry_label": "Pm-3m"},
    ],
)
def test_write_xtl_partial_explicit_metadata_still_requires_spglib(
    tmp_path,
    monkeypatch,
    metadata_override,
):
    xtl_path = tmp_path / "partial-metadata.xtl"
    _block_spglib_import(monkeypatch)

    with pytest.raises(RuntimeError, match="automatic symmetry derivation requires spglib"):
        write_xtl(*_simple_cubic_structure(), filename=xtl_path, **metadata_override)


def test_write_xtl_spglib_dataset_errors_surface(tmp_path, monkeypatch):
    xtl_path = tmp_path / "spglib-failure.xtl"

    def _boom(_cell):
        raise ValueError("bad symmetry dataset")

    monkeypatch.setitem(
        sys.modules,
        "spglib",
        SimpleNamespace(get_symmetry_dataset=_boom),
    )

    with pytest.raises(ValueError, match="bad symmetry dataset"):
        write_xtl(*_simple_cubic_structure(), filename=xtl_path)


def test_write_xtl_spglib_dataset_none_raises_clear_error(tmp_path, monkeypatch):
    xtl_path = tmp_path / "spglib-none.xtl"

    monkeypatch.setitem(
        sys.modules,
        "spglib",
        SimpleNamespace(get_symmetry_dataset=lambda _cell: None),
    )

    with pytest.raises(
        RuntimeError,
        match="spglib could not derive symmetry metadata from the supplied structure",
    ):
        write_xtl(*_simple_cubic_structure(), filename=xtl_path)


def test_get_atomic_coordinates_wraps_and_deduplicates_same_label_expansion_rows():
    cell_params, atoms = get_atomic_coordinates(
        np.eye(3),
        np.array([[0.25, 0.5, 0.75]]),
        [11],
        {
            "rotations": [
                np.eye(3, dtype=int),
                np.eye(3, dtype=int),
                np.eye(3, dtype=int),
            ],
            "translations": [
                np.zeros(3),
                np.zeros(3),
                np.array([1.0, 0.0, 0.0]),
            ],
        },
        ["Na"],
        (1.0, 1.0, 1.0, 90.0, 90.0, 90.0),
    )

    assert cell_params == (1.0, 1.0, 1.0, 90.0, 90.0, 90.0)
    assert atoms == [("Na", 0.25, 0.5, 0.75)]
