import ast
from pathlib import Path


def _load_main_functions(*names: str) -> dict[str, object]:
    source = Path("main.py").read_text(encoding="utf-8")
    module = ast.parse(source, filename="main.py")
    available = {
        node.name
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    }
    missing = sorted(set(names) - available)
    if missing:
        raise AssertionError(f"Failed to extract functions from main.py: {missing}")

    extracted: list[str] = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            fn_source = ast.get_source_segment(source, node)
            if fn_source:
                extracted.append(fn_source)

    namespace: dict[str, object] = {}
    exec(
        "import numpy as np\n"
        "from typing import Sequence\n\n"
        + "\n\n".join(extracted),
        namespace,
    )
    return namespace


def test_manual_pair_store_keeps_backgrounds_separate() -> None:
    namespace = _load_main_functions(
        "_normalize_hkl_key",
        "_normalize_geometry_manual_pair_entry",
        "_geometry_manual_pairs_for_index",
        "_set_geometry_manual_pairs_for_index",
        "_geometry_manual_pair_group_count",
    )
    namespace["geometry_manual_pairs_by_background"] = {}

    set_pairs = namespace["_set_geometry_manual_pairs_for_index"]
    get_pairs = namespace["_geometry_manual_pairs_for_index"]
    group_count = namespace["_geometry_manual_pair_group_count"]

    bg0_pairs = set_pairs(
        0,
        [
            {
                "label": "1,0,2",
                "x": "10.5",
                "y": 12,
                "q_group_key": ["q_group", "primary", 1, 2],
                "source_table_index": "4",
                "source_row_index": "7",
            }
        ],
    )
    bg1_pairs = set_pairs(
        1,
        [
            {
                "hkl": (2, 0, 0),
                "x": 5,
                "y": 6,
                "q_group_key": ("q_group", "primary", 2, 0),
            }
        ],
    )

    assert bg0_pairs[0]["hkl"] == (1, 0, 2)
    assert bg0_pairs[0]["source_table_index"] == 4
    assert bg0_pairs[0]["source_row_index"] == 7
    assert bg0_pairs[0]["q_group_key"] == ("q_group", "primary", 1, 2)

    assert len(get_pairs(0)) == 1
    assert len(get_pairs(1)) == 1
    assert group_count(0) == 1
    assert group_count(1) == 1
    assert get_pairs(0)[0]["hkl"] != get_pairs(1)[0]["hkl"]


def test_peak_maximum_near_in_image_returns_local_brightest_pixel() -> None:
    namespace = _load_main_functions("_peak_maximum_near_in_image")
    peak_maximum = namespace["_peak_maximum_near_in_image"]

    import numpy as np

    image = np.zeros((9, 9), dtype=float)
    image[4, 4] = 2.0
    image[6, 5] = 9.5
    image[2, 2] = 7.0

    assert peak_maximum(image, 4.2, 4.1, search_radius=1) == (4.0, 4.0)
    assert peak_maximum(image, 4.9, 5.8, search_radius=2) == (5.0, 6.0)


def test_geometry_manual_candidate_source_key_prefers_source_indices() -> None:
    namespace = _load_main_functions(
        "_normalize_hkl_key",
        "_geometry_manual_candidate_source_key",
    )
    source_key = namespace["_geometry_manual_candidate_source_key"]

    assert source_key({"source_table_index": "3", "source_row_index": 9}) == ("source", 3, 9)
    assert source_key({"hkl": (1, 2, 3)}) == ("hkl", 1, 2, 3)
    assert source_key({"label": "1,2,3"}) == ("hkl", 1, 2, 3)
    assert source_key({"label": "left peak"}) == ("label", "left peak")


def test_current_geometry_manual_match_config_reuses_auto_match_defaults() -> None:
    namespace = _load_main_functions("_current_geometry_manual_match_config")
    namespace["fit_config"] = {
        "geometry": {
            "auto_match": {
                "search_radius_px": 17.5,
                "min_match_prominence_sigma": 3.25,
            }
        }
    }
    config_fn = namespace["_current_geometry_manual_match_config"]
    cfg = config_fn()

    assert cfg["search_radius_px"] == 17.5
    assert cfg["min_match_prominence_sigma"] == 3.25
    assert cfg["console_progress"] is False
    assert cfg["relax_on_low_matches"] is False
    assert cfg["require_candidate_ownership"] is True
