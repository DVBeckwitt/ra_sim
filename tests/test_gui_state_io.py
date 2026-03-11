from pathlib import Path

import numpy as np
import pytest

from ra_sim.io.data_loading import (
    GUI_STATE_FILE_TYPE,
    build_gui_state_payload,
    load_gui_state_file,
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
