import ast
from pathlib import Path


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
        "import re\n"
        "import numpy as np\n"
        "from typing import Sequence\n\n"
        + "\n\n".join(extracted),
        namespace,
    )
    return namespace


def test_parse_background_theta_values_accepts_common_delimiters() -> None:
    namespace = _load_main_functions("_parse_background_theta_values")
    parse_values = namespace["_parse_background_theta_values"]

    assert parse_values("1.0, 2.5  3.25;4.0") == [1.0, 2.5, 3.25, 4.0]

    try:
        parse_values("1.0, bad", expected_count=2)
    except ValueError as exc:
        assert "Invalid theta value" in str(exc)
    else:
        raise AssertionError("Expected invalid theta token to raise ValueError")


def test_background_theta_for_index_adds_shared_offset_for_multiple_backgrounds() -> None:
    namespace = _load_main_functions(
        "_background_theta_default_value",
        "_parse_background_theta_values",
        "_parse_geometry_fit_background_indices",
        "_current_geometry_fit_background_indices",
        "_geometry_fit_uses_shared_theta_offset",
        "_current_geometry_theta_offset",
        "_current_background_theta_values",
        "_background_theta_for_index",
    )

    class _Var:
        def __init__(self, value: str):
            self._value = value

        def get(self) -> str:
            return self._value

    namespace["theta_initial"] = 6.0
    namespace["defaults"] = {"theta_initial": 6.0}
    namespace["background_theta_list_var"] = _Var("4.0, 7.5")
    namespace["geometry_fit_background_selection_var"] = _Var("all")
    namespace["geometry_theta_offset_var"] = _Var("0.25")
    namespace["osc_files"] = ["bg0.osc", "bg1.osc"]
    namespace["current_background_index"] = 0

    theta_for_index = namespace["_background_theta_for_index"]

    assert theta_for_index(0, strict_count=True) == 4.25
    assert theta_for_index(1, strict_count=True) == 7.75


def test_parse_geometry_fit_background_indices_supports_current_all_and_ranges() -> None:
    namespace = _load_main_functions("_parse_geometry_fit_background_indices")
    parse_selection = namespace["_parse_geometry_fit_background_indices"]

    assert parse_selection("current", total_count=4, current_index=2) == [2]
    assert parse_selection("all", total_count=3, current_index=1) == [0, 1, 2]
    assert parse_selection("1, 3-4", total_count=4, current_index=0) == [0, 2, 3]


def test_geometry_fit_shared_theta_offset_depends_on_selected_fit_backgrounds() -> None:
    namespace = _load_main_functions(
        "_parse_geometry_fit_background_indices",
        "_current_geometry_fit_background_indices",
        "_geometry_fit_uses_shared_theta_offset",
    )

    class _Var:
        def __init__(self, value: str):
            self._value = value

        def get(self) -> str:
            return self._value

    namespace["osc_files"] = ["bg0.osc", "bg1.osc", "bg2.osc"]
    namespace["current_background_index"] = 1
    namespace["geometry_fit_background_selection_var"] = _Var("current")

    assert namespace["_current_geometry_fit_background_indices"](strict=True) == [1]
    assert namespace["_geometry_fit_uses_shared_theta_offset"]() is False

    namespace["geometry_fit_background_selection_var"] = _Var("1,3")
    assert namespace["_current_geometry_fit_background_indices"](strict=True) == [0, 2]
    assert namespace["_geometry_fit_uses_shared_theta_offset"]() is True


def test_apply_geometry_fit_background_selection_skips_redraw_when_live_theta_is_unchanged() -> None:
    namespace = _load_main_functions(
        "_background_theta_default_value",
        "_format_geometry_fit_background_indices",
        "_parse_background_theta_values",
        "_parse_geometry_fit_background_indices",
        "_current_geometry_fit_background_indices",
        "_geometry_fit_uses_shared_theta_offset",
        "_current_geometry_theta_offset",
        "_current_background_theta_values",
        "_background_theta_for_index",
        "_apply_geometry_fit_background_selection",
    )

    class _Var:
        def __init__(self, value):
            self._value = value

        def get(self):
            return self._value

        def set(self, value) -> None:
            self._value = value

    scheduled = {"count": 0}

    def _schedule_update() -> None:
        scheduled["count"] += 1

    namespace["theta_initial"] = 6.0
    namespace["defaults"] = {"theta_initial": 6.0}
    namespace["osc_files"] = ["bg0.osc", "bg1.osc"]
    namespace["current_background_index"] = 0
    namespace["background_theta_list_var"] = _Var("4.0, 7.5")
    namespace["geometry_theta_offset_var"] = _Var("0.0")
    namespace["theta_initial_var"] = _Var(4.0)
    namespace["geometry_fit_background_selection_var"] = _Var("current")
    namespace["_refresh_geometry_fit_theta_checkbox_label"] = lambda: None
    namespace["_set_background_file_status_text"] = lambda: None
    namespace["schedule_update"] = _schedule_update

    assert namespace["_apply_geometry_fit_background_selection"](trigger_update=True) is True
    assert namespace["geometry_fit_background_selection_var"].get() == "current"
    assert namespace["theta_initial_var"].get() == 4.0
    assert scheduled["count"] == 0


def test_apply_geometry_fit_background_selection_redraws_when_live_theta_changes() -> None:
    namespace = _load_main_functions(
        "_background_theta_default_value",
        "_format_geometry_fit_background_indices",
        "_parse_background_theta_values",
        "_parse_geometry_fit_background_indices",
        "_current_geometry_fit_background_indices",
        "_geometry_fit_uses_shared_theta_offset",
        "_current_geometry_theta_offset",
        "_current_background_theta_values",
        "_background_theta_for_index",
        "_apply_geometry_fit_background_selection",
    )

    class _Var:
        def __init__(self, value):
            self._value = value

        def get(self):
            return self._value

        def set(self, value) -> None:
            self._value = value

    scheduled = {"count": 0}

    def _schedule_update() -> None:
        scheduled["count"] += 1

    namespace["theta_initial"] = 6.0
    namespace["defaults"] = {"theta_initial": 6.0}
    namespace["osc_files"] = ["bg0.osc", "bg1.osc"]
    namespace["current_background_index"] = 0
    namespace["background_theta_list_var"] = _Var("4.0, 7.5")
    namespace["geometry_theta_offset_var"] = _Var("0.25")
    namespace["theta_initial_var"] = _Var(4.25)
    namespace["geometry_fit_background_selection_var"] = _Var("current")
    namespace["_refresh_geometry_fit_theta_checkbox_label"] = lambda: None
    namespace["_set_background_file_status_text"] = lambda: None
    namespace["schedule_update"] = _schedule_update

    assert namespace["_apply_geometry_fit_background_selection"](trigger_update=True) is True
    assert namespace["theta_initial_var"].get() == 4.0
    assert scheduled["count"] == 1


def test_apply_geometry_fit_background_selection_preserves_live_theta_when_theta_list_is_blank() -> None:
    namespace = _load_main_functions(
        "_background_theta_default_value",
        "_format_geometry_fit_background_indices",
        "_parse_background_theta_values",
        "_parse_geometry_fit_background_indices",
        "_current_geometry_fit_background_indices",
        "_geometry_fit_uses_shared_theta_offset",
        "_current_geometry_theta_offset",
        "_current_background_theta_values",
        "_background_theta_for_index",
        "_apply_geometry_fit_background_selection",
    )

    class _Var:
        def __init__(self, value):
            self._value = value

        def get(self):
            return self._value

        def set(self, value) -> None:
            self._value = value

    scheduled = {"count": 0}

    def _schedule_update() -> None:
        scheduled["count"] += 1

    namespace["theta_initial"] = 6.0
    namespace["defaults"] = {"theta_initial": 6.0}
    namespace["osc_files"] = ["bg0.osc", "bg1.osc"]
    namespace["current_background_index"] = 0
    namespace["background_theta_list_var"] = _Var("")
    namespace["geometry_theta_offset_var"] = _Var("0.0")
    namespace["theta_initial_var"] = _Var(12.75)
    namespace["geometry_fit_background_selection_var"] = _Var("current")
    namespace["_refresh_geometry_fit_theta_checkbox_label"] = lambda: None
    namespace["_set_background_file_status_text"] = lambda: None
    namespace["schedule_update"] = _schedule_update

    assert namespace["_apply_geometry_fit_background_selection"](trigger_update=True) is True
    assert namespace["theta_initial_var"].get() == 12.75
    assert scheduled["count"] == 0
