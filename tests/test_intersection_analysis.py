from dataclasses import replace

import numpy as np
import pytest

from ra_sim.simulation import intersection_analysis as ia
from ra_sim.simulation import exact_cake_portable
from ra_sim.simulation.intersection_analysis import (
    BeamSamples,
    IntersectionGeometry,
    MosaicParams,
    analyze_reflection_intersection,
    detector_points_to_sample_qr_qz,
    project_qr_cylinder_to_detector,
)
from ra_sim.utils.calculations import IndexofRefraction


def _qr_projection_geometry(*, image_size: int = 128) -> IntersectionGeometry:
    center = float(image_size) / 2.0
    return IntersectionGeometry(
        image_size=image_size,
        center_col=center,
        center_row=center,
        distance_cor_to_detector=0.075,
        gamma_deg=0.0,
        Gamma_deg=0.0,
        chi_deg=0.0,
        psi_deg=0.0,
        psi_z_deg=0.0,
        zs=0.0,
        zb=0.0,
        theta_initial_deg=6.0,
        cor_angle_deg=0.0,
        n_detector=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        unit_x=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        pixel_size_m=1.0e-4,
    )


@pytest.fixture(scope="module")
def reference_analysis():
    geometry = IntersectionGeometry(
        image_size=3000,
        center_col=1500.0,
        center_row=1500.0,
        distance_cor_to_detector=0.075,
        gamma_deg=0.0,
        Gamma_deg=0.0,
        chi_deg=0.0,
        psi_deg=0.0,
        psi_z_deg=0.0,
        zs=0.0,
        zb=0.0,
        theta_initial_deg=6.0,
        cor_angle_deg=0.0,
        n_detector=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        unit_x=np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )
    beam = BeamSamples(
        beam_x_array=np.array([0.0], dtype=np.float64),
        beam_y_array=np.array([0.0], dtype=np.float64),
        theta_array=np.array([0.0], dtype=np.float64),
        phi_array=np.array([0.0], dtype=np.float64),
        wavelength_array=np.array([1.0], dtype=np.float64),
    )
    mosaic = MosaicParams(
        sigma_mosaic_deg=0.8,
        gamma_mosaic_deg=0.3,
        eta=0.05,
    )
    return analyze_reflection_intersection(
        h=1,
        k=0,
        l=0,
        lattice_a=4.0,
        lattice_c=7.0,
        selected_native_col=1500.0,
        selected_native_row=1500.0,
        geometry=geometry,
        beam=beam,
        mosaic=mosaic,
        n2=IndexofRefraction(),
        sphere_res=36,
    )


def test_intersection_analysis_arc_non_empty(reference_analysis):
    assert reference_analysis.arc_q.shape[0] > 0
    assert reference_analysis.arc_intensity.shape[0] == reference_analysis.arc_q.shape[0]


def test_intersection_analysis_arc_values_finite_non_negative(reference_analysis):
    assert np.all(np.isfinite(reference_analysis.arc_intensity))
    assert np.all(reference_analysis.arc_intensity >= 0.0)


def test_intersection_analysis_nearest_index_is_deterministic(reference_analysis):
    geometry = IntersectionGeometry(
        image_size=3000,
        center_col=1500.0,
        center_row=1500.0,
        distance_cor_to_detector=0.075,
        gamma_deg=0.0,
        Gamma_deg=0.0,
        chi_deg=0.0,
        psi_deg=0.0,
        psi_z_deg=0.0,
        zs=0.0,
        zb=0.0,
        theta_initial_deg=6.0,
        cor_angle_deg=0.0,
        n_detector=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        unit_x=np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )
    beam = BeamSamples(
        beam_x_array=np.array([0.0], dtype=np.float64),
        beam_y_array=np.array([0.0], dtype=np.float64),
        theta_array=np.array([0.0], dtype=np.float64),
        phi_array=np.array([0.0], dtype=np.float64),
        wavelength_array=np.array([1.0], dtype=np.float64),
    )
    mosaic = MosaicParams(
        sigma_mosaic_deg=0.8,
        gamma_mosaic_deg=0.3,
        eta=0.05,
    )
    result_2 = analyze_reflection_intersection(
        h=1,
        k=0,
        l=0,
        lattice_a=4.0,
        lattice_c=7.0,
        selected_native_col=1500.0,
        selected_native_row=1500.0,
        geometry=geometry,
        beam=beam,
        mosaic=mosaic,
        n2=IndexofRefraction(),
        sphere_res=36,
    )
    assert result_2.nearest_index == reference_analysis.nearest_index


def test_intersection_analysis_sphere_shape_and_finiteness(reference_analysis):
    assert reference_analysis.sphere_x.shape == reference_analysis.sphere_intensity.shape
    assert reference_analysis.sphere_y.shape == reference_analysis.sphere_intensity.shape
    assert reference_analysis.sphere_z.shape == reference_analysis.sphere_intensity.shape
    assert np.all(np.isfinite(reference_analysis.sphere_intensity))


def test_detector_points_to_sample_qr_qz_round_trips_qr_cylinder_projection():
    geometry = _qr_projection_geometry()
    qr0 = 0.05
    traces = project_qr_cylinder_to_detector(
        qr_value=qr0,
        geometry=geometry,
        wavelength=1.0,
        n2=IndexofRefraction(),
        phi_samples=721,
    )
    finite_trace = next(trace for trace in traces if np.any(trace.valid_mask))
    sample_idx = np.flatnonzero(finite_trace.valid_mask)[::97][:8]
    recovered_qr, recovered_qz, valid = detector_points_to_sample_qr_qz(
        detector_col=finite_trace.detector_col[sample_idx],
        detector_row=finite_trace.detector_row[sample_idx],
        geometry=geometry,
        wavelength=1.0,
        n2=IndexofRefraction(),
    )

    assert np.all(valid)
    assert recovered_qr == pytest.approx(np.full(sample_idx.size, qr0), rel=2.0e-8)
    assert recovered_qz == pytest.approx(finite_trace.qz[sample_idx], rel=2.0e-8, abs=1.0e-10)


def test_detector_points_to_sample_qr_qz_round_trips_projected_qr_high_abs_phi():
    geometry = _qr_projection_geometry()
    qr0 = 0.05
    detector_shape = (int(geometry.image_size), int(geometry.image_size))
    radial_axis, raw_azimuth_axis = exact_cake_portable.build_angle_axes(
        npt_rad=64,
        npt_azim=180,
        tth_min_deg=0.0,
        tth_max_deg=90.0,
    )
    ai = exact_cake_portable.FastAzimuthalIntegrator(
        dist=float(geometry.distance_cor_to_detector),
        poni1=float(geometry.center_row) * float(geometry.pixel_size_m),
        poni2=float(geometry.center_col) * float(geometry.pixel_size_m),
        pixel1=float(geometry.pixel_size_m),
        pixel2=float(geometry.pixel_size_m),
    )
    bundle = exact_cake_portable.build_cake_transform_bundle(
        ai,
        detector_shape,
        radial_axis,
        raw_azimuth_axis,
        workers=1,
    )
    assert bundle is not None
    traces = project_qr_cylinder_to_detector(
        qr_value=qr0,
        geometry=geometry,
        wavelength=1.0,
        n2=IndexofRefraction(),
        phi_samples=1441,
    )

    selected_col = selected_row = selected_qz = None
    for trace in traces:
        for idx in np.flatnonzero(trace.valid_mask):
            two_theta, phi = exact_cake_portable.detector_pixel_to_caked_bin(
                bundle,
                float(trace.detector_col[idx]),
                float(trace.detector_row[idx]),
            )
            if (
                two_theta is not None
                and phi is not None
                and -85.0 <= float(phi) <= -72.5
            ):
                selected_col = float(trace.detector_col[idx])
                selected_row = float(trace.detector_row[idx])
                selected_qz = float(trace.qz[idx])
                break
        if selected_col is not None:
            break

    assert selected_col is not None
    recovered_qr, recovered_qz, valid = detector_points_to_sample_qr_qz(
        detector_col=np.asarray([selected_col], dtype=float),
        detector_row=np.asarray([selected_row], dtype=float),
        geometry=geometry,
        wavelength=1.0,
        n2=IndexofRefraction(),
    )

    assert bool(valid[0])
    assert float(recovered_qr[0]) == pytest.approx(qr0, rel=2.0e-8)
    assert float(recovered_qz[0]) == pytest.approx(selected_qz, rel=2.0e-8, abs=1.0e-10)


def test_build_sample_frame_psi_z_yaws_cor_axis():
    geometry = IntersectionGeometry(
        image_size=3000,
        center_col=1500.0,
        center_row=1500.0,
        distance_cor_to_detector=0.075,
        gamma_deg=0.0,
        Gamma_deg=0.0,
        chi_deg=0.0,
        psi_deg=0.0,
        psi_z_deg=0.0,
        zs=0.0,
        zb=0.0,
        theta_initial_deg=8.0,
        cor_angle_deg=15.0,
        n_detector=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        unit_x=np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )

    _, n_surf_0, _ = ia._build_sample_frame(geometry)
    _, n_surf_60, _ = ia._build_sample_frame(replace(geometry, psi_z_deg=60.0))

    assert np.isclose(float(n_surf_0[2]), float(n_surf_60[2]), atol=1e-12, rtol=0.0)
    assert abs(float(n_surf_60[0])) > abs(float(n_surf_0[0])) + 1e-6
    assert abs(float(n_surf_60[1])) < abs(float(n_surf_0[1])) - 1e-6
    assert not np.allclose(n_surf_0, n_surf_60)
