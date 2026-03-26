from pathlib import Path

import numpy as np

from ra_sim.gui import background_manager, state


def test_background_manager_apply_update_preserves_list_aliases() -> None:
    background_state = state.BackgroundRuntimeState()
    osc_alias = background_state.osc_files
    native_alias = background_state.background_images_native

    background_manager.apply_background_state_update(
        background_state,
        {
            "osc_files": ["a.osc", "b.osc"],
            "background_images": [np.zeros((2, 2)), np.ones((2, 2))],
            "background_images_native": [np.zeros((2, 2)), np.ones((2, 2))],
            "background_images_display": [np.zeros((2, 2)), np.ones((2, 2))],
            "current_background_index": 1,
            "current_background_image": np.ones((2, 2)),
            "current_background_display": np.ones((2, 2)),
        },
    )

    assert background_state.osc_files is osc_alias
    assert background_state.background_images_native is native_alias
    assert background_state.osc_files == ["a.osc", "b.osc"]
    assert background_state.current_background_index == 1


def test_background_manager_load_and_switch_update_state_in_place(tmp_path) -> None:
    path_a = tmp_path / "a.osc"
    path_b = tmp_path / "b.osc"
    path_a.write_text("", encoding="utf-8")
    path_b.write_text("", encoding="utf-8")
    background_state = state.BackgroundRuntimeState()

    def _fake_read_osc(path: str):
        if path == str(path_a):
            return np.zeros((2, 2))
        return np.ones((2, 2))

    background_manager.load_background_files(
        background_state,
        [str(path_a), str(path_b)],
        image_size=2,
        display_rotate_k=-1,
        read_osc=_fake_read_osc,
        select_index=0,
    )
    switched = background_manager.switch_background(
        background_state,
        display_rotate_k=-1,
        read_osc=_fake_read_osc,
    )

    assert background_state.osc_files == [str(path_a), str(path_b)]
    assert background_state.current_background_index == 1
    assert np.array_equal(background_state.current_background_image, np.ones((2, 2)))
    assert np.array_equal(switched["current_background_display"], np.rot90(np.ones((2, 2)), -1))


def test_background_manager_builds_background_status_text() -> None:
    text = background_manager.build_background_file_status_text(
        osc_files=["C:/data/a.osc", "C:/data/b.osc"],
        current_background_index=1,
        theta_base=12.34567,
        theta_effective=12.84567,
        use_shared_theta_offset=True,
        pair_count=3,
        group_count=2,
        sigma_values=[1.0, float("nan"), 3.0],
        fit_indices=[0, 1],
    )

    assert text.startswith("Background 2/2: b.osc")
    assert "theta_i=12.3457° | theta=12.8457°" in text
    assert "manual=2 groups/3 pts" in text
    assert "sigma~2.00px" in text
    assert "fit=2 backgrounds" in text

    assert (
        background_manager.build_background_file_status_text(
            osc_files=[],
            current_background_index=0,
        )
        == "Background: no files loaded"
    )


def test_background_manager_chooses_background_dialog_initial_dir(tmp_path) -> None:
    current_path = tmp_path / "nested" / "a.osc"
    current_path.parent.mkdir()
    current_path.write_text("", encoding="utf-8")

    initial_dir = background_manager.background_file_dialog_initial_dir(
        [str(current_path)],
        0,
        tmp_path / "fallback",
    )

    assert initial_dir == str(current_path.parent)
    assert background_manager.background_file_dialog_initial_dir(
        ["bad-path"],
        99,
        Path("fallback"),
    ) == "fallback"
