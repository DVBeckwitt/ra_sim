import numpy as np

from ra_sim.gui import state_io


class _DisplayRecorder:
    def __init__(self) -> None:
        self.last_data = None

    def set_data(self, value) -> None:
        self.last_data = value


def test_load_background_files_for_state_reuses_identical_files_without_reread(
    tmp_path,
) -> None:
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

    updated = state_io.load_background_files_for_state(
        [str(path_a), str(path_b)],
        osc_files=[str(path_a), str(path_b)],
        background_images=[native_a.copy(), native_b.copy()],
        background_images_native=[native_a, native_b],
        background_images_display=[display_a, display_b],
        select_index=1,
        display_rotate_k=-1,
        read_osc=_read_osc_should_not_run,
        set_background_display=display.set_data,
    )

    assert updated is not None
    assert updated["osc_files"] == [str(path_a), str(path_b)]
    assert updated["current_background_index"] == 1
    assert updated["current_background_image"] is native_b
    assert updated["current_background_display"] is display_b
    assert display.last_data is display_b


def test_load_background_files_for_state_rereads_when_files_change(tmp_path) -> None:
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
    updated = state_io.load_background_files_for_state(
        [str(path_b), str(path_c)],
        osc_files=[str(path_a), str(path_b)],
        background_images=[np.zeros((2, 2)), np.ones((2, 2))],
        background_images_native=[np.zeros((2, 2)), np.ones((2, 2))],
        background_images_display=[np.zeros((2, 2)), np.ones((2, 2))],
        select_index=1,
        display_rotate_k=-1,
        read_osc=_fake_read_osc,
        set_background_display=display.set_data,
    )

    assert updated is not None
    assert calls == [str(path_b), str(path_c)]
    assert updated["osc_files"] == [str(path_b), str(path_c)]
    assert updated["current_background_index"] == 1
    assert np.array_equal(
        updated["current_background_image"],
        np.full((2, 2), 3.0),
    )
    assert display.last_data is updated["current_background_display"]
