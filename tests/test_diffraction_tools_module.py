from ra_sim.simulation import geometry as simulation_geometry
from ra_sim.utils import diffraction_tools, tools


def test_tools_reexport_diffraction_helpers() -> None:
    assert tools.detector_two_theta_max is diffraction_tools.detector_two_theta_max
    assert tools.miller_generator is diffraction_tools.miller_generator
    assert tools.intensities_for_hkls is diffraction_tools.intensities_for_hkls


def test_simulation_geometry_reexports_integrator_helper() -> None:
    assert (
        simulation_geometry.setup_azimuthal_integrator
        is diffraction_tools.setup_azimuthal_integrator
    )
