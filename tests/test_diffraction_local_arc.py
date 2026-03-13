from __future__ import annotations

import math

import numpy as np

from ra_sim.simulation import diffraction


def _circle_params(k_in_crystal: np.ndarray, k_scat: float, g_vec: np.ndarray):
    g_sq = float(np.dot(g_vec, g_vec))
    a_vec = -np.asarray(k_in_crystal, dtype=np.float64)
    a_sq = float(np.dot(a_vec, a_vec))
    a_len = math.sqrt(a_sq)
    c_val = (g_sq + a_sq - float(k_scat) * float(k_scat)) / (2.0 * a_len)
    circle_r = math.sqrt(g_sq - c_val * c_val)
    a_hat = a_vec / a_len
    origin = c_val * a_hat

    anchor = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(anchor, a_hat))) > 0.9999:
        anchor = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e1 = anchor - float(np.dot(anchor, a_hat)) * a_hat
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a_hat, e1)
    e2 /= np.linalg.norm(e2)
    return origin, circle_r, e1, e2


def test_local_arc_windows_reduce_domain_for_narrow_profile() -> None:
    k_in = np.array([0.0, 0.8, -0.05], dtype=np.float64)
    g_vec = np.array([0.0, 0.7, 0.2], dtype=np.float64)
    origin, circle_r, e1, e2 = _circle_params(k_in, 1.0, g_vec)

    starts, ends, count, use_full = diffraction._build_local_arc_windows.py_func(
        origin[0],
        origin[1],
        origin[2],
        circle_r,
        e1[0],
        e1[1],
        e1[2],
        e2[0],
        e2[1],
        e2[2],
        g_vec,
        0.002,
        0.001,
        0.0,
        512,
    )

    assert not use_full
    assert count >= 1
    assert float(np.sum(ends[:count] - starts[:count])) < 0.25 * (2.0 * np.pi)


def test_local_arc_windows_fall_back_for_broad_profile() -> None:
    k_in = np.array([0.0, 0.8, -0.05], dtype=np.float64)
    g_vec = np.array([0.0, 0.7, 0.2], dtype=np.float64)
    origin, circle_r, e1, e2 = _circle_params(k_in, 1.0, g_vec)

    starts, ends, count, use_full = diffraction._build_local_arc_windows.py_func(
        origin[0],
        origin[1],
        origin[2],
        circle_r,
        e1[0],
        e1[1],
        e1[2],
        e2[0],
        e2[1],
        e2[2],
        g_vec,
        1.0,
        0.5,
        0.5,
        512,
    )

    assert use_full
    assert count == 1
    np.testing.assert_allclose(starts[0], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(ends[0], 2.0 * np.pi, atol=0.0, rtol=0.0)


def test_local_arc_uniform_keeps_peak_mass_near_full_circle() -> None:
    k_in = np.array([0.0, 0.8, -0.05], dtype=np.float64)
    g_vec = np.array([0.0, 0.7, 0.2], dtype=np.float64)
    origin, circle_r, e1, e2 = _circle_params(k_in, 1.0, g_vec)

    full = diffraction._solve_q_uniform_full_circle.py_func(
        origin[0],
        origin[1],
        origin[2],
        circle_r,
        e1[0],
        e1[1],
        e1[2],
        e2[0],
        e2[1],
        e2[2],
        g_vec,
        0.002,
        0.001,
        0.0,
        512,
    )
    local = diffraction._solve_q_uniform.py_func(
        origin[0],
        origin[1],
        origin[2],
        circle_r,
        e1[0],
        e1[1],
        e1[2],
        e2[0],
        e2[1],
        e2[2],
        g_vec,
        0.002,
        0.001,
        0.0,
        512,
    )

    assert full.shape[0] > 0
    assert local.shape[0] > 0

    full_mass = float(np.sum(full[:, 3]))
    local_mass = float(np.sum(local[:, 3]))
    assert 0.6 <= (local_mass / full_mass) <= 1.4

    full_peak_q = full[int(np.argmax(full[:, 3])), :3]
    local_peak_q = local[int(np.argmax(local[:, 3])), :3]
    assert float(np.linalg.norm(full_peak_q - local_peak_q)) < 1.0e-2


def test_nominal_reflection_visible_culls_far_off_projection() -> None:
    sample_terms = np.zeros((1, diffraction._SAMPLE_COLS), dtype=np.float64)
    sample_terms[0, diffraction._SAMPLE_COL_VALID] = 1.0
    sample_terms[0, diffraction._SAMPLE_COL_I_PLANE_X] = 0.0
    sample_terms[0, diffraction._SAMPLE_COL_I_PLANE_Y] = 0.0
    sample_terms[0, diffraction._SAMPLE_COL_I_PLANE_Z] = 0.0
    sample_terms[0, diffraction._SAMPLE_COL_KX_SCAT] = 40.0
    sample_terms[0, diffraction._SAMPLE_COL_KY_SCAT] = 1.0
    sample_terms[0, diffraction._SAMPLE_COL_RE_KZ] = 0.0
    sample_terms[0, diffraction._SAMPLE_COL_K_SCAT] = 1.0
    sample_terms[0, diffraction._SAMPLE_COL_K0] = 1.0
    sample_terms[0, diffraction._SAMPLE_COL_N2_REAL] = 1.0

    visible, nominal_idx, no_valid = diffraction._nominal_reflection_visible.py_func(
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        64,
        np.array([32.0, 32.0], dtype=np.float64),
        np.eye(3, dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.1, 0.0], dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        sample_terms,
        0,
        0.002,
        0.001,
        diffraction.OPTICS_MODE_FAST,
        -1,
    )

    assert not visible
    assert nominal_idx == 0
    assert not no_valid


def test_nominal_reflection_visible_keeps_central_projection() -> None:
    sample_terms = np.zeros((1, diffraction._SAMPLE_COLS), dtype=np.float64)
    sample_terms[0, diffraction._SAMPLE_COL_VALID] = 1.0
    sample_terms[0, diffraction._SAMPLE_COL_I_PLANE_X] = 0.0
    sample_terms[0, diffraction._SAMPLE_COL_I_PLANE_Y] = 0.0
    sample_terms[0, diffraction._SAMPLE_COL_I_PLANE_Z] = 0.0
    sample_terms[0, diffraction._SAMPLE_COL_KX_SCAT] = 0.0
    sample_terms[0, diffraction._SAMPLE_COL_KY_SCAT] = 1.0
    sample_terms[0, diffraction._SAMPLE_COL_RE_KZ] = 0.0
    sample_terms[0, diffraction._SAMPLE_COL_K_SCAT] = 1.0
    sample_terms[0, diffraction._SAMPLE_COL_K0] = 1.0
    sample_terms[0, diffraction._SAMPLE_COL_N2_REAL] = 1.0

    visible, nominal_idx, no_valid = diffraction._nominal_reflection_visible.py_func(
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        64,
        np.array([32.0, 32.0], dtype=np.float64),
        np.eye(3, dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.1, 0.0], dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        sample_terms,
        0,
        0.002,
        0.001,
        diffraction.OPTICS_MODE_FAST,
        -1,
    )

    assert visible
    assert nominal_idx == 0
    assert not no_valid
