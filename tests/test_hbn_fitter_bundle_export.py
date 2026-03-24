from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import numpy as np


def _load_fitter_save_bundle_symbols() -> dict[str, object]:
    source = Path("hbn_fitter/fitter.py").read_text(encoding="utf-8")
    module = ast.parse(source, filename="hbn_fitter/fitter.py")

    wanted_constants = {
        "SIM_BACKGROUND_ROTATE_K",
        "SIM_GAMMA_SIGN_FROM_TILT_X",
        "SIM_GAMMA_SIGN_FROM_TILT_Y",
    }
    wanted_functions = {
        "pts_to_obj",
        "scalars_to_obj",
        "ellipses_to_array",
        "ellipse_ring_indices",
    }

    extracted: list[str] = []
    discovered_constants: set[str] = set()
    discovered_functions: set[str] = set()
    save_bundle_source: str | None = None

    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted_constants:
                    segment = ast.get_source_segment(source, node)
                    if segment:
                        extracted.append(segment)
                        discovered_constants.add(target.id)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
            segment = ast.get_source_segment(source, node)
            if segment:
                extracted.append(segment)
                discovered_functions.add(node.name)
        elif isinstance(node, ast.ClassDef) and node.name == "HBNFitterGUI":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "save_bundle":
                    segment = ast.get_source_segment(source, child)
                    if segment:
                        save_bundle_source = textwrap.dedent(segment)

    missing_constants = sorted(wanted_constants - discovered_constants)
    missing_functions = sorted(wanted_functions - discovered_functions)
    if missing_constants or missing_functions or save_bundle_source is None:
        raise AssertionError(
            "Failed to extract hBN fitter bundle save symbols: "
            f"constants={missing_constants}, functions={missing_functions}, "
            f"save_bundle_found={save_bundle_source is not None}"
        )

    namespace: dict[str, object] = {}
    exec(
        "import datetime as dt\n"
        "from pathlib import Path\n"
        "import numpy as np\n\n"
        + "\n\n".join(extracted)
        + "\n\n"
        + save_bundle_source,
        namespace,
    )
    return namespace


class _DummyVar:
    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value


class _DummyFitter:
    def __init__(self) -> None:
        self.img_bgsub = np.arange(16, dtype=np.float32).reshape(4, 4)
        self.optim = {
            "tilt_x_deg": 2.75,
            "tilt_y_deg": -1.5,
            "cost_zero": 0.12,
            "cost_final": 0.03,
        }
        self.fit_quality = {}
        self.points_ds = [[(1.0, 2.0), (3.0, 4.0)]]
        self.points_raw_ds = [[(1.5, 2.5), (3.5, 4.5)]]
        self.points_sigma_ds = [[0.2, 0.3]]
        self.down = 1
        self.ellipses = [
            {
                "xc": 12.0,
                "yc": 10.0,
                "a": 8.0,
                "b": 7.5,
                "theta": 0.25,
                "ring_index": 0,
            }
        ]
        self.hbn_path = _DummyVar("hbn.osc")
        self.dark_path = _DummyVar("dark.osc")

    def _get_fullres_log_image(self):
        return np.ones((4, 4), dtype=np.float32)

    def _get_center(self, strict=False):
        return (12.0, 10.0), "ui_entry"

    def _points_for_fit(self, points, downsample=1):
        return points

    def _sigma_for_fit(self, values, downsample=1):
        return values


def test_hbn_fitter_save_bundle_exports_opposite_sign_detector_tilts(tmp_path: Path) -> None:
    namespace = _load_fitter_save_bundle_symbols()
    save_bundle = namespace["save_bundle"]

    fitter = _DummyFitter()
    out_path = tmp_path / "bundle.npz"
    save_bundle(fitter, out_path)

    data = np.load(out_path, allow_pickle=True)
    tilt_correction = data["tilt_correction"].item()
    tilt_hint = data["tilt_hint"].item()

    assert np.isclose(float(data["tilt_x_deg"]), -2.75)
    assert np.isclose(float(data["tilt_y_deg"]), 1.5)
    assert np.isclose(float(data["tilt_x_deg_internal"]), 2.75)
    assert np.isclose(float(data["tilt_y_deg_internal"]), -1.5)

    assert np.isclose(float(tilt_correction["tilt_x_deg"]), -2.75)
    assert np.isclose(float(tilt_correction["tilt_y_deg"]), 1.5)
    assert np.isclose(float(tilt_hint["rot1_rad"]), np.deg2rad(-2.75))
    assert np.isclose(float(tilt_hint["rot2_rad"]), np.deg2rad(1.5))
