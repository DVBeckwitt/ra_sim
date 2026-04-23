import math

import pytest

from ra_sim.structure_factors.vesta_like_atomic_factors import (
    atomic_factor_debug_table,
    f0,
    f_total,
    s_from_d,
)


def test_s_from_d_units():
    d = 2.0

    assert s_from_d(d) == 0.25
    assert s_from_d(d) == s_from_d(d)
    with pytest.raises(ValueError):
        s_from_d(0.0)


def test_f0_forward_limit_reasonable():
    values = f0(["Bi", "Se"], [0.0], table="waaskirf")[0]

    assert values[0] == pytest.approx(83, rel=0.05)
    assert values[1] == pytest.approx(34, rel=0.08)


def test_atomic_factor_debug_table_contains_bi_and_se():
    table = atomic_factor_debug_table(["Bi", "Se"], 0.1, 1.5405929254021151)

    assert {row["element"] for row in table} == {"Bi", "Se"}
    assert all(abs(row["f_total_real"]) > 0.0 for row in table)


def test_vesta_coefficients_can_be_compared():
    total = f_total(
        ["Bi", "Se"],
        [0.1],
        1.5405929254021151,
        table="waaskirf",
        anomalous_mode="vesta_cu_ka1",
    )[0]

    assert math.isclose(total[0].imag, 8.83640, rel_tol=0.0, abs_tol=1e-8)
    assert math.isclose(total[1].imag, 1.13462, rel_tol=0.0, abs_tol=1e-8)
