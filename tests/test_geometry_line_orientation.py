import ast
from pathlib import Path

import numpy as np


def _load_main_functions(*names: str) -> dict[str, object]:
    source = Path("main.py").read_text(encoding="utf-8")
    module = ast.parse(source, filename="main.py")
    extracted: list[str] = []
    available = {
        node.name
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    }
    missing = sorted(set(names) - available)
    if missing:
        raise AssertionError(f"Failed to extract functions from main.py: {missing}")

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            fn_source = ast.get_source_segment(source, node)
            if fn_source:
                extracted.append(fn_source)

    namespace: dict[str, object] = {}
    exec(
        "import math\n"
        "import numpy as np\n\n"
        "DISPLAY_ROTATE_K = -1\n"
        "SIM_DISPLAY_ROTATE_K = 0\n\n"
        + "\n\n".join(extracted),
        namespace,
    )
    return namespace


def test_fixed_geometry_fit_orientation_choice_defaults_to_known_transform() -> None:
    namespace = _load_main_functions("_fixed_geometry_fit_orientation_choice")
    choice = namespace["_fixed_geometry_fit_orientation_choice"]()

    assert choice == {
        "k": 1,
        "flip_x": True,
        "flip_y": True,
        "flip_order": "yx",
        "indexing_mode": "xy",
        "label": "rot90° CCW + flip_x + flip_y + order=yx + indexing=xy",
    }


def test_background_display_and_native_sim_orientation_helpers_roundtrip() -> None:
    namespace = _load_main_functions(
        "_fixed_geometry_fit_orientation_choice",
        "_rotate_point_for_display",
        "_native_background_to_display_coords",
        "_display_to_native_background_coords",
        "_transform_points_orientation",
        "_iter_orientation_transform_candidates",
        "_inverse_orientation_transform",
        "_background_display_to_native_sim_coords",
        "_native_sim_to_background_display_coords",
    )

    orientation_choice = namespace["_fixed_geometry_fit_orientation_choice"]()
    to_native_sim = namespace["_background_display_to_native_sim_coords"]
    to_bg_display = namespace["_native_sim_to_background_display_coords"]
    shape = (1846, 3000)
    display_points = [
        (0.0, 0.0),
        (125.5, 331.25),
        (1499.0, 922.0),
        (2999.0, 1845.0),
    ]

    for disp_col, disp_row in display_points:
        sim_native = to_native_sim(
            disp_col,
            disp_row,
            shape,
            orientation_choice=orientation_choice,
        )
        roundtrip = to_bg_display(
            sim_native[0],
            sim_native[1],
            shape,
            orientation_choice=orientation_choice,
        )
        assert np.allclose(roundtrip, (disp_col, disp_row), atol=1e-6)
