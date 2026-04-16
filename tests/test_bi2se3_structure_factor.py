import math

import pytest

from ra_sim.StructureFactor import calculate_bi2se3_structure_factor
from ra_sim.StructureFactor.StructureFactor import (
    _nominal_layer_count_from_spacing_nm,
    attenuation,
)


def test_calculate_bi2se3_structure_factor_rejects_unsupported_labels_up_front():
    def _should_not_run(_q):
        raise AssertionError("scattering-factor callable should not run for bad labels")

    with pytest.raises(ValueError) as exc_info:
        calculate_bi2se3_structure_factor(
            1,
            0,
            0,
            [("Te", 0.0, 0.0, 0.0)],
            [_should_not_run, _should_not_run],
            [1.0, 1.0, 1.0],
        )
    message = str(exc_info.value)
    assert "Te" in message
    assert "Bi" in message
    assert "Se1" in message
    assert "Se2" in message


@pytest.mark.parametrize(
    ("label", "occ_index", "base_value", "offset"),
    [
        ("Bi", 0, 11.0, -4.23706 + 8.83640j),
        ("Se1", 1, 7.0, -0.787865 + 1.13462j),
        ("Se2", 2, 7.0, -0.787865 + 1.13462j),
    ],
)
def test_calculate_bi2se3_structure_factor_uses_explicit_label_mapping(
    label, occ_index, base_value, offset
):
    occ = [2.0, 3.0, 5.0]
    data = [
        lambda _q: 11.0,
        lambda _q: 7.0,
    ]

    intensity = calculate_bi2se3_structure_factor(
        0,
        0,
        0,
        [(label, 0.0, 0.0, 0.0)],
        data,
        occ,
    )

    expected_factor = (base_value + offset) * occ[occ_index]
    assert math.isclose(intensity, abs(expected_factor) ** 2, rel_tol=1e-12, abs_tol=1e-12)


def test_calculate_bi2se3_structure_factor_supports_iterator_atoms():
    atoms = [
        ("Bi", 0.0, 0.0, 0.0),
        ("Se1", 0.25, 0.0, 0.0),
        ("Se2", 0.0, 0.25, 0.5),
    ]
    data = [lambda _q: 11.0, lambda _q: 7.0]
    occ = [2.0, 3.0, 5.0]

    expected = calculate_bi2se3_structure_factor(1, 1, 1, atoms, data, occ)
    actual = calculate_bi2se3_structure_factor(1, 1, 1, iter(atoms), data, occ)

    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_attenuation_uses_50_nm_nominal_thickness_for_500_angstrom_film():
    c_nm = 0.625
    layer_count = _nominal_layer_count_from_spacing_nm(c_nm)

    assert layer_count == 80
    assert layer_count != 800
    assert math.isclose(
        attenuation(c_nm, 0.0, 0.0),
        float(layer_count**2),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_attenuation_clamps_large_spacing_to_one_layer():
    c_nm = 100.0
    layer_count = _nominal_layer_count_from_spacing_nm(c_nm)

    assert layer_count == 1
    assert math.isclose(
        attenuation(c_nm, 0.0, 0.0),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


@pytest.mark.parametrize("c_nm", (0.0, -0.625, float("nan"), float("inf")))
def test_attenuation_rejects_non_physical_layer_spacing(c_nm):
    with pytest.raises(ValueError, match="positive and finite"):
        _nominal_layer_count_from_spacing_nm(c_nm)

    with pytest.raises(ValueError, match="positive and finite"):
        attenuation(c_nm, 0.0, 0.0)


@pytest.mark.parametrize("occ", ([1.0, 1.0], [1.0, 1.0, 1.0, 1.0]))
def test_calculate_bi2se3_structure_factor_rejects_bad_occupancy_length(occ):
    with pytest.raises(ValueError, match=r"exactly 3 occupancies"):
        calculate_bi2se3_structure_factor(
            1,
            0,
            0,
            [("Bi", 0.0, 0.0, 0.0)],
            [lambda _q: 1.0, lambda _q: 1.0],
            occ,
        )
