import importlib
import sys


MODULE_NAME = "ra_sim.hbn_geometry"
HEAVY_MODULES = (
    "cv2",
    "matplotlib.pyplot",
    "scipy.optimize",
    "skimage.measure",
)


def test_hbn_geometry_import_avoids_calibrant_stack() -> None:
    previous = {name: sys.modules.pop(name, None) for name in (MODULE_NAME, *HEAVY_MODULES)}

    try:
        module = importlib.import_module(MODULE_NAME)

        assert module.__name__ == MODULE_NAME
        for heavy_name in HEAVY_MODULES:
            assert heavy_name not in sys.modules
    finally:
        for name in (MODULE_NAME, *HEAVY_MODULES):
            sys.modules.pop(name, None)
        for name, module in previous.items():
            if module is not None:
                sys.modules[name] = module
