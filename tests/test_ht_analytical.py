import numpy as np
import pytest

from ra_sim.utils.stacking_fault import (
    AREA,
    P_CLAMP,
    DEFAULT_PHASE_DELTA_EXPRESSION,
    DEFAULT_PHI_L_DIVISOR,
    analytical_ht_intensity_for_pair,
    validate_phase_delta_expression,
)


def _legacy_finite_R_from_t(t, n_layers):
    t = np.asarray(t, dtype=complex)
    n = int(max(1, n_layers))
    if n == 1:
        return np.ones_like(np.real(t), dtype=float)

    one = 1.0 + 0.0j
    mask = np.isclose(t, one)
    out = np.empty_like(np.real(t), dtype=float)

    if np.any(~mask):
        t_nm = t[~mask]
        denom = one - t_nm
        s1 = t_nm * (1 - t_nm ** (n - 1)) / denom
        s2 = t_nm * (1 - n * t_nm ** (n - 1) + (n - 1) * t_nm ** n) / (denom ** 2)
        out[~mask] = (n + 2.0 * np.real(n * s1 - s2)) / n

    if np.any(mask):
        out[mask] = float(n)

    return np.maximum(out, 0.0)


def _legacy_algebraic_reference(L_vals, F2_vals, h, k, p, *, finite_layers=None):
    L_vals = np.asarray(L_vals, dtype=float)
    F2_vals = np.asarray(F2_vals, dtype=float)

    p_flipped = 1.0 - float(np.clip(p, 0.0, 1.0))
    delta = 2.0 * np.pi * ((2.0 * float(h) + float(k)) / 3.0)
    z = (1.0 - p_flipped) + p_flipped * np.exp(1j * delta)
    f_val = min(float(np.abs(z)), 1.0 - float(P_CLAMP))
    psi = float(np.angle(z))
    phi = delta + 2.0 * np.pi * L_vals * (1.0 / DEFAULT_PHI_L_DIVISOR)

    if finite_layers is None:
        denom = 1.0 + f_val * f_val - 2.0 * f_val * np.cos(phi - psi)
        denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
        R = (1.0 - f_val * f_val) / denom
    else:
        t = f_val * np.exp(1j * (phi - psi))
        R = _legacy_finite_R_from_t(t, int(max(1, finite_layers)))

    return np.maximum(0.0, float(AREA) * F2_vals * R)


def test_analytical_backend_matches_reference_expression():
    h, k, p = 2, -1, 0.37
    L_vals = np.linspace(0.0, 4.0, 321)
    F2_vals = 0.2 + 2.0 * np.exp(-0.3 * L_vals) + 0.1 * np.cos(2.0 * np.pi * L_vals)

    observed = analytical_ht_intensity_for_pair(
        L_vals,
        F2_vals,
        h,
        k,
        p,
        phase_delta_expression=DEFAULT_PHASE_DELTA_EXPRESSION,
        finite_layers=None,
    )
    expected = _legacy_algebraic_reference(L_vals, F2_vals, h, k, p, finite_layers=None)

    assert np.allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_custom_phase_expression_changes_intensity_profile():
    h, k, p = 1, 0, 0.42
    L_vals = np.linspace(0.0, 3.0, 200)
    F2_vals = 1.0 + 0.25 * np.sin(L_vals)

    default_i = analytical_ht_intensity_for_pair(
        L_vals,
        F2_vals,
        h,
        k,
        p,
        phase_delta_expression=DEFAULT_PHASE_DELTA_EXPRESSION,
    )
    custom_i = analytical_ht_intensity_for_pair(
        L_vals,
        F2_vals,
        h,
        k,
        p,
        phase_delta_expression="2*pi*((2*h + k)/3) + 0.3*sin(2*pi*L)",
    )

    assert not np.allclose(default_i, custom_i)


def test_custom_phi_l_divisor_changes_intensity_profile():
    h, k, p = 1, 0, 0.42
    L_vals = np.linspace(0.0, 3.0, 200)
    F2_vals = 1.0 + 0.25 * np.sin(L_vals)

    default_i = analytical_ht_intensity_for_pair(
        L_vals,
        F2_vals,
        h,
        k,
        p,
        phase_delta_expression=DEFAULT_PHASE_DELTA_EXPRESSION,
    )
    explicit_default_i = analytical_ht_intensity_for_pair(
        L_vals,
        F2_vals,
        h,
        k,
        p,
        phase_delta_expression=DEFAULT_PHASE_DELTA_EXPRESSION,
        phi_l_divisor=DEFAULT_PHI_L_DIVISOR,
    )
    custom_i = analytical_ht_intensity_for_pair(
        L_vals,
        F2_vals,
        h,
        k,
        p,
        phase_delta_expression=DEFAULT_PHASE_DELTA_EXPRESSION,
        phi_l_divisor=6.0,
    )

    assert np.allclose(default_i, explicit_default_i, rtol=1e-12, atol=1e-12)
    assert not np.allclose(default_i, custom_i)


def test_invalid_phase_expression_is_rejected():
    with pytest.raises(ValueError):
        validate_phase_delta_expression("__import__('os').system('echo bad')")

    with pytest.raises(ValueError):
        validate_phase_delta_expression("unknown_name + 1")


def test_finite_stack_converges_to_infinite_limit():
    h, k, p = 1, 0, 0.35
    L_vals = np.linspace(0.0, 2.0, 181)
    F2_vals = 0.5 + np.exp(-0.4 * L_vals)

    infinite_i = analytical_ht_intensity_for_pair(
        L_vals,
        F2_vals,
        h,
        k,
        p,
        phase_delta_expression=DEFAULT_PHASE_DELTA_EXPRESSION,
        finite_layers=None,
    )
    finite_i = analytical_ht_intensity_for_pair(
        L_vals,
        F2_vals,
        h,
        k,
        p,
        phase_delta_expression=DEFAULT_PHASE_DELTA_EXPRESSION,
        finite_layers=600,
    )

    assert np.allclose(finite_i, infinite_i, rtol=3e-3, atol=1e-6)
