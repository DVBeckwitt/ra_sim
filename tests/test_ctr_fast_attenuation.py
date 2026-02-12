import numpy as np

from ra_sim.simulation.diffraction import _ctr_attenuation_factor


def _effective_layers(v_sum, c_ang):
    vc = max(v_sum, 0.0) * c_ang
    if vc <= 1e-6:
        return 1_000_000
    return max(1, min(1_000_000, int(np.ceil(1.0 / vc))))


def test_ctr_finite_discrete_matches_direct_series_sum():
    qz = 0.37
    c_ang = 6.25
    n_layers = 64
    v_in = 3e-4
    v_out = 7e-4

    val = _ctr_attenuation_factor(qz, v_in, v_out, c_ang, n_layers)

    v_sum = v_in + v_out
    ell = np.arange(n_layers, dtype=np.float64)
    terms_q = np.exp((1j * qz - v_sum) * c_ang * ell)
    terms_0 = np.exp((1j * qz) * c_ang * ell)
    expected = (np.abs(np.sum(terms_q)) ** 2) / (np.abs(np.sum(terms_0)) ** 2)

    assert np.isclose(val, expected, rtol=1e-12, atol=1e-12)


def test_ctr_finite_discrete_is_unity_without_absorption():
    qz = 0.37
    c_ang = 5.0
    n_layers = 25
    val = _ctr_attenuation_factor(qz, 0.0, 0.0, c_ang, n_layers)
    assert np.isclose(val, 1.0, rtol=0.0, atol=1e-12)


def test_ctr_semi_infinite_uses_absorption_limited_effective_depth():
    qz = 0.28
    c_ang = 6.25
    v_in = 4e-4
    v_out = 5e-4

    val = _ctr_attenuation_factor(qz, v_in, v_out, c_ang, 0)

    v_sum = v_in + v_out
    n_eff = _effective_layers(v_sum, c_ang)
    ell = np.arange(n_eff, dtype=np.float64)
    terms_v = np.exp((1j * qz - v_sum) * c_ang * ell)
    terms_0 = np.exp((1j * qz) * c_ang * ell)
    expected = (np.abs(np.sum(terms_v)) ** 2) / (np.abs(np.sum(terms_0)) ** 2)

    assert np.isclose(val, expected, rtol=1e-12, atol=1e-12)


def test_ctr_semi_infinite_no_longer_collapses_to_zero_at_qz_zero():
    c_ang = 6.25
    val = _ctr_attenuation_factor(0.0, 4e-4, 5e-4, c_ang, 0)
    assert val > 1e-2


def test_ctr_finite_discrete_depends_on_layer_count():
    qz = 0.2
    c_ang = 6.25
    v_in = 3e-4
    v_out = 2e-4
    i_20 = _ctr_attenuation_factor(qz, v_in, v_out, c_ang, 20)
    i_80 = _ctr_attenuation_factor(qz, v_in, v_out, c_ang, 80)

    assert np.isfinite(i_20)
    assert np.isfinite(i_80)
    assert not np.isclose(i_20, i_80, rtol=1e-8, atol=1e-12)


def test_ctr_finite_discrete_can_exceed_one_near_interference_minima():
    qz = 0.2
    c_ang = 6.25
    n_layers = 20
    val = _ctr_attenuation_factor(qz, 3e-4, 2e-4, c_ang, n_layers)
    assert val > 1.0
