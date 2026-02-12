import numpy as np
import pytest

from ra_sim.simulation.intersection_analysis import (
    BeamSamples,
    IntersectionGeometry,
    MosaicParams,
    analyze_reflection_intersection,
)
from ra_sim.utils.calculations import IndexofRefraction


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
