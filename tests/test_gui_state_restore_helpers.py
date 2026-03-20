import ast
from pathlib import Path

import numpy as np


def _load_app_functions(*names: str) -> dict[str, object]:
    source = Path("ra_sim/gui/app.py").read_text(encoding="utf-8")
    module = ast.parse(source, filename="ra_sim/gui/app.py")
    available = {
        node.name
        for node in module.body
        if isinstance(node, ast.FunctionDef)
    }
    missing = sorted(set(names) - available)
    if missing:
        raise AssertionError(
            f"Failed to extract functions from ra_sim/gui/app.py: {missing}"
        )

    extracted: list[str] = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            fn_source = ast.get_source_segment(source, node)
            if fn_source:
                extracted.append(fn_source)

    namespace: dict[str, object] = {}
    exec(
        "import os\n"
        "from pathlib import Path\n"
        "import numpy as np\n\n"
        + "\n\n".join(extracted),
        namespace,
    )
    return namespace


class _DisplayRecorder:
    def __init__(self) -> None:
        self.last_data = None

    def set_data(self, value) -> None:
        self.last_data = value


def test_load_background_files_for_state_reuses_identical_files_without_reread(
    tmp_path,
) -> None:
    namespace = _load_app_functions(
        "_canonicalize_gui_state_background_path",
        "_background_files_match_loaded_state",
        "_load_background_files_for_state",
    )
    loader = namespace["_load_background_files_for_state"]

    path_a = tmp_path / "a.osc"
    path_b = tmp_path / "b.osc"
    path_a.write_text("", encoding="utf-8")
    path_b.write_text("", encoding="utf-8")

    native_a = np.arange(4, dtype=float).reshape(2, 2)
    native_b = np.arange(4, 8, dtype=float).reshape(2, 2)
    display_a = np.rot90(native_a, -1)
    display_b = np.rot90(native_b, -1)
    display = _DisplayRecorder()

    def _read_osc_should_not_run(_path: str):
        raise AssertionError("read_osc should not be called for identical backgrounds")

    namespace["osc_files"] = [str(path_a), str(path_b)]
    namespace["background_images"] = [native_a.copy(), native_b.copy()]
    namespace["background_images_native"] = [native_a, native_b]
    namespace["background_images_display"] = [display_a, display_b]
    namespace["current_background_index"] = 0
    namespace["current_background_image"] = native_a
    namespace["current_background_display"] = display_a
    namespace["background_display"] = display
    namespace["DISPLAY_ROTATE_K"] = -1
    namespace["read_osc"] = _read_osc_should_not_run

    loader([str(path_a), str(path_b)], select_index=1)

    assert namespace["osc_files"] == [str(path_a), str(path_b)]
    assert namespace["current_background_index"] == 1
    assert namespace["current_background_image"] is native_b
    assert namespace["current_background_display"] is display_b
    assert display.last_data is display_b


def test_load_background_files_for_state_rereads_when_files_change(tmp_path) -> None:
    namespace = _load_app_functions(
        "_canonicalize_gui_state_background_path",
        "_background_files_match_loaded_state",
        "_load_background_files_for_state",
    )
    loader = namespace["_load_background_files_for_state"]

    path_a = tmp_path / "a.osc"
    path_b = tmp_path / "b.osc"
    path_c = tmp_path / "c.osc"
    for path in (path_a, path_b, path_c):
        path.write_text("", encoding="utf-8")

    calls: list[str] = []

    def _fake_read_osc(path: str):
        calls.append(path)
        if path.endswith("b.osc"):
            return np.full((2, 2), 2.0)
        return np.full((2, 2), 3.0)

    display = _DisplayRecorder()
    namespace["osc_files"] = [str(path_a), str(path_b)]
    namespace["background_images"] = [np.zeros((2, 2)), np.ones((2, 2))]
    namespace["background_images_native"] = [np.zeros((2, 2)), np.ones((2, 2))]
    namespace["background_images_display"] = [
        np.zeros((2, 2)),
        np.ones((2, 2)),
    ]
    namespace["current_background_index"] = 0
    namespace["current_background_image"] = namespace["background_images_native"][0]
    namespace["current_background_display"] = namespace["background_images_display"][0]
    namespace["background_display"] = display
    namespace["DISPLAY_ROTATE_K"] = -1
    namespace["read_osc"] = _fake_read_osc

    loader([str(path_b), str(path_c)], select_index=1)

    assert calls == [str(path_b), str(path_c)]
    assert namespace["osc_files"] == [str(path_b), str(path_c)]
    assert namespace["current_background_index"] == 1
    assert np.array_equal(
        namespace["current_background_image"],
        np.full((2, 2), 3.0),
    )
    assert display.last_data is namespace["current_background_display"]
