from types import SimpleNamespace

import numpy as np

from ra_sim.gui import qr_cylinder_overlay


def _projection_context() -> dict[str, object]:
    bundle = qr_cylinder_overlay.CakeTransformBundle(
        detector_shape=(3, 3),
        radial_deg=np.array([10.0, 20.0], dtype=float),
        raw_azimuth_deg=np.array([-95.0, -85.0], dtype=float),
        gui_azimuth_deg=np.array([5.0, -5.0], dtype=float),
        lut=SimpleNamespace(),
    )
    return {
        "detector_shape": (3, 3),
        "radial_axis": np.array([10.0, 20.0], dtype=float),
        "azimuth_axis": np.array([-5.0, 5.0], dtype=float),
        "raw_azimuth_axis": np.array([-95.0, -85.0], dtype=float),
        "transform_bundle": bundle,
    }


def test_interpolate_trace_to_caked_coords_projects_each_valid_detector_sample_through_exact_bundle(
    monkeypatch,
) -> None:
    context = _projection_context()
    projection_calls: list[tuple[object, float, float]] = []

    def _record_projection(bundle, col, row):
        projection_calls.append((bundle, float(col), float(row)))
        if (float(col), float(row)) == (0.0, 0.0):
            return (10.0, -5.0)
        if (float(col), float(row)) == (1.0, 1.0):
            return (20.0, 5.0)
        return (None, None)

    monkeypatch.setattr(
        qr_cylinder_overlay,
        "detector_pixel_to_caked_bin",
        _record_projection,
    )

    two_theta, phi = qr_cylinder_overlay.interpolate_trace_to_caked_coords(
        detector_cols=np.asarray([0.0, 1.0], dtype=float),
        detector_rows=np.asarray([0.0, 1.0], dtype=float),
        valid_mask=np.asarray([True, True]),
        projection_context=context,
    )

    assert np.allclose(two_theta, [10.0, 20.0], atol=1e-8, equal_nan=True)
    assert np.allclose(phi, [-5.0, 5.0], atol=1e-8, equal_nan=True)
    assert projection_calls == [
        (context["transform_bundle"], 0.0, 0.0),
        (context["transform_bundle"], 1.0, 1.0),
    ]


def test_interpolate_trace_to_caked_coords_returns_nan_without_valid_projection_context() -> None:
    two_theta, phi = qr_cylinder_overlay.interpolate_trace_to_caked_coords(
        detector_cols=np.asarray([0.0, 1.0], dtype=float),
        detector_rows=np.asarray([0.0, 1.0], dtype=float),
        valid_mask=np.asarray([True, True]),
        projection_context=None,
    )

    assert np.all(np.isnan(two_theta))
    assert np.all(np.isnan(phi))


def test_interpolate_trace_to_caked_coords_rejects_mismatched_gui_axis_before_projection(
    monkeypatch,
) -> None:
    context = _projection_context()
    context["azimuth_axis"] = np.array([-6.0, 4.0], dtype=float)

    def _fail_projection(*_args, **_kwargs):
        raise AssertionError("mismatched projection context should fail before projection")

    monkeypatch.setattr(
        qr_cylinder_overlay,
        "detector_pixel_to_caked_bin",
        _fail_projection,
    )

    two_theta, phi = qr_cylinder_overlay.interpolate_trace_to_caked_coords(
        detector_cols=np.asarray([0.0, 1.0], dtype=float),
        detector_rows=np.asarray([0.0, 1.0], dtype=float),
        valid_mask=np.asarray([True, True]),
        projection_context=context,
    )

    assert np.all(np.isnan(two_theta))
    assert np.all(np.isnan(phi))


def test_interpolate_trace_to_caked_coords_breaks_wrap_discontinuities_after_exact_projection(
    monkeypatch,
) -> None:
    def _projection(_bundle, col, row):
        del row
        if float(col) == 0.0:
            return (10.0, 179.0)
        if float(col) == 1.0:
            return (20.0, -179.0)
        if float(col) == 2.0:
            return (30.0, -178.0)
        return (None, None)

    monkeypatch.setattr(
        qr_cylinder_overlay,
        "detector_pixel_to_caked_bin",
        _projection,
    )

    two_theta, phi = qr_cylinder_overlay.interpolate_trace_to_caked_coords(
        detector_cols=np.asarray([0.0, 1.0, 2.0], dtype=float),
        detector_rows=np.asarray([0.0, 0.0, 0.0], dtype=float),
        valid_mask=np.asarray([True, True, True]),
        projection_context=_projection_context(),
    )

    assert np.isfinite(two_theta[0])
    assert np.isnan(two_theta[1])
    assert np.isnan(phi[1])
    assert np.isfinite(two_theta[2])
    assert np.isfinite(phi[2])


def test_interpolate_trace_to_caked_coords_clips_two_theta_limits_after_exact_projection(
    monkeypatch,
) -> None:
    def _projection(_bundle, col, row):
        del row
        return (95.0 if float(col) > 0.0 else 10.0, 0.0)

    monkeypatch.setattr(
        qr_cylinder_overlay,
        "detector_pixel_to_caked_bin",
        _projection,
    )

    two_theta, phi = qr_cylinder_overlay.interpolate_trace_to_caked_coords(
        detector_cols=np.asarray([0.0, 1.0], dtype=float),
        detector_rows=np.asarray([0.0, 0.0], dtype=float),
        valid_mask=np.asarray([True, True]),
        projection_context=_projection_context(),
        two_theta_limits=(0.0, 90.0),
    )

    assert np.isfinite(two_theta[0])
    assert np.isfinite(phi[0])
    assert np.isnan(two_theta[1])
    assert np.isnan(phi[1])
