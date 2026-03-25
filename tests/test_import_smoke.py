import importlib
import pkgutil

import ra_sim
import pytest


ALL_MODULES = sorted(
    module_info.name
    for module_info in pkgutil.walk_packages(ra_sim.__path__, prefix=f"{ra_sim.__name__}.")
)


@pytest.mark.parametrize("module_name", ALL_MODULES)
def test_import_module(module_name: str) -> None:
    if module_name == "ra_sim.gui.runtime":
        pytest.skip("GUI runtime module still performs heavy startup work on import.")
    importlib.import_module(module_name)
