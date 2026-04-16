import importlib
import sys

from ra_sim.simulation import geometry as simulation_geometry
from ra_sim.utils import diffraction_tools, tools


def test_utils_factor_exports_are_lazy() -> None:
    module_names = (
        "ra_sim.utils",
        "ra_sim.utils.factors",
        "ra_sim.utils.diffraction_tools",
        "ra_sim.utils.tools",
    )
    previous_modules = {name: sys.modules.pop(name, None) for name in module_names}

    try:
        utils = importlib.import_module("ra_sim.utils")

        assert utils.__name__ == "ra_sim.utils"
        assert "ra_sim.utils.factors" not in sys.modules
        assert "ionic_atomic_form_factors" in dir(utils)
        assert "F_comp" in dir(utils)
        assert "ra_sim.utils.factors" not in sys.modules

        imported_modules: dict[str, object] = {}
        exec("from ra_sim.utils import diffraction_tools, tools", {}, imported_modules)

        assert "ra_sim.utils.factors" not in sys.modules
        assert imported_modules["diffraction_tools"].__name__ == "ra_sim.utils.diffraction_tools"
        assert imported_modules["tools"].__name__ == "ra_sim.utils.tools"

        imported_exports: dict[str, object] = {}
        exec(
            "from ra_sim.utils import ionic_atomic_form_factors, F_comp",
            {},
            imported_exports,
        )

        assert "ra_sim.utils.factors" in sys.modules
        assert callable(imported_exports["ionic_atomic_form_factors"])
        assert callable(imported_exports["F_comp"])
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        for name, module in previous_modules.items():
            if module is not None:
                sys.modules[name] = module


def test_tools_reexport_diffraction_helpers() -> None:
    assert tools.detector_two_theta_max is diffraction_tools.detector_two_theta_max
    assert tools.miller_generator is diffraction_tools.miller_generator
    assert tools.intensities_for_hkls is diffraction_tools.intensities_for_hkls


def test_simulation_geometry_reexports_integrator_helper() -> None:
    assert (
        simulation_geometry.setup_azimuthal_integrator
        is diffraction_tools.setup_azimuthal_integrator
    )
