from __future__ import annotations

import numpy as np
import pytest

from ra_sim.gui import background


def test_initialize_background_cache_loads_only_first_image(tmp_path) -> None:
    path_a = tmp_path / "a.osc"
    path_a.write_text("", encoding="utf-8")
    calls: list[str] = []

    def _fake_read_osc(path: str):
        calls.append(path)
        return np.arange(4, dtype=float).reshape(2, 2)

    state = background.initialize_background_cache(
        str(path_a),
        total_count=3,
        display_rotate_k=-1,
        read_osc=_fake_read_osc,
    )

    assert calls == [str(path_a)]
    assert len(state["background_images"]) == 3
    assert state["background_images"][1] is None
    assert state["background_images_native"][2] is None
    assert np.array_equal(
        state["current_background_display"],
        np.rot90(np.arange(4, dtype=float).reshape(2, 2), -1),
    )


def test_load_background_image_by_index_populates_lazy_cache(tmp_path) -> None:
    path_a = tmp_path / "a.osc"
    path_b = tmp_path / "b.osc"
    path_a.write_text("", encoding="utf-8")
    path_b.write_text("", encoding="utf-8")

    first = np.arange(4, dtype=float).reshape(2, 2)
    calls: list[str] = []

    def _fake_read_osc(path: str):
        calls.append(path)
        if path == str(path_a):
            return first
        return np.full((2, 2), 9.0)

    initial = background.initialize_background_cache(
        str(path_a),
        total_count=2,
        display_rotate_k=-1,
        read_osc=_fake_read_osc,
    )
    updated = background.load_background_image_by_index(
        1,
        osc_files=[str(path_a), str(path_b)],
        background_images=initial["background_images"],
        background_images_native=initial["background_images_native"],
        background_images_display=initial["background_images_display"],
        display_rotate_k=-1,
        read_osc=_fake_read_osc,
    )

    assert calls == [str(path_a), str(path_b)]
    assert np.array_equal(updated["background_image"], np.full((2, 2), 9.0))
    assert np.array_equal(
        updated["background_display"],
        np.rot90(np.full((2, 2), 9.0), -1),
    )


def test_load_background_files_validates_shape_and_selects_index(tmp_path) -> None:
    path_a = tmp_path / "a.osc"
    path_b = tmp_path / "b.osc"
    path_a.write_text("", encoding="utf-8")
    path_b.write_text("", encoding="utf-8")

    def _fake_read_osc(path: str):
        if path == str(path_a):
            return np.ones((3, 3))
        return np.full((3, 3), 2.0)

    state = background.load_background_files(
        [str(path_a), str(path_b)],
        image_size=3,
        display_rotate_k=-1,
        read_osc=_fake_read_osc,
        select_index=1,
    )

    assert state["osc_files"] == [str(path_a), str(path_b)]
    assert state["current_background_index"] == 1
    assert np.array_equal(state["current_background_image"], np.full((3, 3), 2.0))


def test_load_background_files_rejects_wrong_shape(tmp_path) -> None:
    path_a = tmp_path / "a.osc"
    path_a.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="expected"):
        background.load_background_files(
            [str(path_a)],
            image_size=4,
            display_rotate_k=-1,
            read_osc=lambda _path: np.ones((3, 3)),
        )


def test_switch_background_cycles_and_reuses_cached_state(tmp_path) -> None:
    path_a = tmp_path / "a.osc"
    path_b = tmp_path / "b.osc"
    path_a.write_text("", encoding="utf-8")
    path_b.write_text("", encoding="utf-8")

    calls: list[str] = []

    def _fake_read_osc(path: str):
        calls.append(path)
        if path == str(path_a):
            return np.zeros((2, 2))
        return np.ones((2, 2))

    initial = background.initialize_background_cache(
        str(path_a),
        total_count=2,
        display_rotate_k=-1,
        read_osc=_fake_read_osc,
    )
    switched = background.switch_background(
        osc_files=[str(path_a), str(path_b)],
        background_images=initial["background_images"],
        background_images_native=initial["background_images_native"],
        background_images_display=initial["background_images_display"],
        current_background_index=0,
        display_rotate_k=-1,
        read_osc=_fake_read_osc,
    )

    assert switched["current_background_index"] == 1
    assert np.array_equal(switched["current_background_image"], np.ones((2, 2)))
    assert calls == [str(path_a), str(path_b)]


def test_apply_background_backend_orientation_flips_then_rotates() -> None:
    image = np.array([[1, 2], [3, 4]])

    oriented = background.apply_background_backend_orientation(
        image,
        flip_x=True,
        flip_y=True,
        rotation_k=1,
    )

    expected = np.rot90(np.flip(np.flip(image, axis=0), axis=1), 1)
    assert np.array_equal(oriented, expected)
