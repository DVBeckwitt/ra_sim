from pathlib import Path

import numpy as np
import pytest

from ra_sim.io.data_loading import (
    GEOMETRY_PLACEMENTS_FILE_TYPE,
    GUI_STATE_FILE_TYPE,
    build_geometry_placements_payload,
    build_gui_state_payload,
    load_geometry_placements_file,
    load_gui_state_file,
    save_geometry_placements_file,
    save_gui_state_file,
)


def test_build_gui_state_payload_normalizes_numpy_and_path_values(tmp_path):
    payload = build_gui_state_payload(
        {
            "flag": np.bool_(True),
            "count": np.int64(7),
            "distance": np.float64(2.5),
            "missing": np.float64(np.nan),
            "path": tmp_path / "state.json",
            "nested": {
                "coords": (1, 2, 3),
                "items": {3, 1},
            },
        },
        metadata={"entrypoint": Path("main.py")},
    )

    assert payload["type"] == GUI_STATE_FILE_TYPE
    assert payload["state"]["flag"] is True
    assert payload["state"]["count"] == 7
    assert payload["state"]["distance"] == 2.5
    assert payload["state"]["missing"] is None
    assert payload["state"]["path"] == str(tmp_path / "state.json")
    assert payload["state"]["nested"]["coords"] == [1, 2, 3]
    assert payload["state"]["nested"]["items"] == [1, 3]
    assert payload["metadata"]["entrypoint"] == "main.py"


def test_save_and_load_gui_state_file_round_trip(tmp_path):
    file_path = tmp_path / "gui_state.json"
    save_gui_state_file(
        file_path,
        {
            "variables": {"gamma_var": 1.25, "show_1d_var": True},
            "files": {"background_files": ["a.osc", "b.osc"]},
        },
    )

    loaded = load_gui_state_file(file_path)

    assert loaded["type"] == GUI_STATE_FILE_TYPE
    assert loaded["state"]["variables"]["gamma_var"] == 1.25
    assert loaded["state"]["variables"]["show_1d_var"] is True
    assert loaded["state"]["files"]["background_files"] == ["a.osc", "b.osc"]


def test_load_gui_state_file_rejects_wrong_type(tmp_path):
    file_path = tmp_path / "bad_state.json"
    file_path.write_text('{"type":"wrong","state":{}}', encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported GUI state file type"):
        load_gui_state_file(file_path)


def test_build_geometry_placements_payload_normalizes_nested_values(tmp_path):
    payload = build_geometry_placements_payload(
        {
            "background_files": [tmp_path / "a.osc"],
            "manual_pairs": [
                {
                    "background_index": np.int64(0),
                    "entries": [
                        {
                            "hkl": (1, 0, 2),
                            "q_group_key": ("q_group", "primary", 1.0, 2),
                        }
                    ],
                }
            ],
        },
        metadata={"entrypoint": Path("main.py")},
    )

    assert payload["type"] == GEOMETRY_PLACEMENTS_FILE_TYPE
    assert payload["state"]["background_files"] == [str(tmp_path / "a.osc")]
    assert payload["state"]["manual_pairs"][0]["background_index"] == 0
    assert payload["state"]["manual_pairs"][0]["entries"][0]["hkl"] == [1, 0, 2]
    assert payload["state"]["manual_pairs"][0]["entries"][0]["q_group_key"] == [
        "q_group",
        "primary",
        1.0,
        2,
    ]
    assert payload["metadata"]["entrypoint"] == "main.py"


def test_save_and_load_geometry_placements_file_round_trip(tmp_path):
    file_path = tmp_path / "placements.json"
    save_geometry_placements_file(
        file_path,
        {
            "background_files": ["a.osc", "b.osc"],
            "manual_pairs": [
                {
                    "background_index": 1,
                    "background_name": "b.osc",
                    "entries": [{"label": "1,0,0", "x": 12.5, "y": 9.0}],
                }
            ],
        },
    )

    loaded = load_geometry_placements_file(file_path)

    assert loaded["type"] == GEOMETRY_PLACEMENTS_FILE_TYPE
    assert loaded["state"]["background_files"] == ["a.osc", "b.osc"]
    assert loaded["state"]["manual_pairs"][0]["background_name"] == "b.osc"
    assert loaded["state"]["manual_pairs"][0]["entries"][0]["x"] == 12.5
