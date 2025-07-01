import pytest

from ra_sim.utils.stacking_fault import R_exact, R_near_perfect, R_quasi_random


def test_R_exact_symmetry():
    f = 0.75
    th = 0.3
    assert R_exact(f, th) == pytest.approx(R_exact(f, -th))


def test_near_perfect_matches_exact_nonzero_theta():
    eps = 1e-3
    th = 0.5
    exact = R_exact(1 - eps, th)
    approx = R_near_perfect(th, eps)
    assert approx == pytest.approx(exact, rel=1e-3)


def test_near_perfect_theta_zero():
    eps = 1e-4
    th = 0.0
    exact = R_exact(1 - eps, th)
    approx = R_near_perfect(th, eps)
    assert approx == pytest.approx(exact, rel=1e-3)


def test_quasi_random_matches_exact():
    eps = 1e-3
    th = 0.6
    exact = R_exact(eps, th)
    approx = R_quasi_random(th, eps)
    assert approx == pytest.approx(exact, rel=1e-3)
