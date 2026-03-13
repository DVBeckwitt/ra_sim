import numpy as np

from ra_sim.gui.qr_cylinder_overlay import interpolate_trace_to_caked_coords


def test_interpolate_trace_to_caked_coords_preserves_exact_pixel_samples() -> None:
    two_theta_map = np.asarray(
        [
            [10.0, 20.0],
            [30.0, 40.0],
        ],
        dtype=float,
    )
    phi_map = np.asarray(
        [
            [-45.0, -15.0],
            [15.0, 45.0],
        ],
        dtype=float,
    )

    two_theta, phi = interpolate_trace_to_caked_coords(
        detector_cols=np.asarray([0.0, 1.0], dtype=float),
        detector_rows=np.asarray([0.0, 1.0], dtype=float),
        valid_mask=np.asarray([True, True]),
        two_theta_map=two_theta_map,
        phi_map_deg=phi_map,
    )

    assert np.allclose(two_theta, [10.0, 40.0], atol=1e-8, equal_nan=True)
    assert np.allclose(phi, [-45.0, 45.0], atol=1e-8, equal_nan=True)


def test_interpolate_trace_to_caked_coords_interpolates_wrapped_phi_safely() -> None:
    two_theta_map = np.asarray([[12.0, 12.0]], dtype=float)
    phi_map = np.asarray([[179.0, -179.0]], dtype=float)

    _, phi = interpolate_trace_to_caked_coords(
        detector_cols=np.asarray([0.5], dtype=float),
        detector_rows=np.asarray([0.0], dtype=float),
        valid_mask=np.asarray([True]),
        two_theta_map=two_theta_map,
        phi_map_deg=phi_map,
    )

    assert np.isfinite(phi[0])
    assert abs(abs(float(phi[0])) - 180.0) < 1.0


def test_interpolate_trace_to_caked_coords_breaks_wrap_discontinuities() -> None:
    two_theta_map = np.asarray([[10.0, 20.0, 30.0]], dtype=float)
    phi_map = np.asarray([[179.0, -179.0, -178.0]], dtype=float)

    two_theta, phi = interpolate_trace_to_caked_coords(
        detector_cols=np.asarray([0.0, 1.0, 2.0], dtype=float),
        detector_rows=np.asarray([0.0, 0.0, 0.0], dtype=float),
        valid_mask=np.asarray([True, True, True]),
        two_theta_map=two_theta_map,
        phi_map_deg=phi_map,
    )

    assert np.isfinite(two_theta[0])
    assert np.isnan(two_theta[1])
    assert np.isnan(phi[1])
    assert np.isfinite(two_theta[2])
    assert np.isfinite(phi[2])
