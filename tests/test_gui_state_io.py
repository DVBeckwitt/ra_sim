import json

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
        metadata={"entrypoint": "python -m ra_sim gui"},
    )

    assert payload["type"] == GUI_STATE_FILE_TYPE
    assert payload["state"]["flag"] is True
    assert payload["state"]["count"] == 7
    assert payload["state"]["distance"] == 2.5
    assert payload["state"]["missing"] is None
    assert payload["state"]["path"] == str(tmp_path / "state.json")
    assert payload["state"]["nested"]["coords"] == [1, 2, 3]
    assert payload["state"]["nested"]["items"] == [1, 3]
    assert payload["metadata"]["entrypoint"] == "python -m ra_sim gui"


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


def test_load_gui_state_file_accepts_legacy_raw_snapshot(tmp_path):
    file_path = tmp_path / "legacy_state.json"
    file_path.write_text(
        json.dumps(
            {
                "variables": {"gamma_var": 1.25},
                "files": {"background_files": ["a.osc"]},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_gui_state_file(file_path)

    assert loaded["type"] == GUI_STATE_FILE_TYPE
    assert loaded["version"] == 0
    assert loaded["state"]["variables"]["gamma_var"] == 1.25
    assert loaded["state"]["files"]["background_files"] == ["a.osc"]


def test_load_gui_state_file_accepts_legacy_wrapper_without_type(tmp_path):
    file_path = tmp_path / "legacy_wrapped_state.json"
    file_path.write_text(
        json.dumps(
            {
                "saved_at": "2025-01-01T00:00:00",
                "metadata": {"entrypoint": "main.py"},
                "state": {
                    "variables": {"show_1d_var": True},
                    "flags": {"background_visible": False},
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_gui_state_file(file_path)

    assert loaded["type"] == GUI_STATE_FILE_TYPE
    assert loaded["version"] == 0
    assert loaded["saved_at"] == "2025-01-01T00:00:00"
    assert loaded["metadata"]["entrypoint"] == "main.py"
    assert loaded["state"]["variables"]["show_1d_var"] is True
    assert loaded["state"]["flags"]["background_visible"] is False


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
        metadata={"entrypoint": "python -m ra_sim gui"},
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
    assert payload["metadata"]["entrypoint"] == "python -m ra_sim gui"


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


def test_save_and_load_geometry_placements_file_preserves_caked_angles(
    tmp_path,
):
    file_path = tmp_path / "placements_angles.json"
    save_geometry_placements_file(
        file_path,
        {
            "background_files": ["a.osc"],
            "manual_pairs": [
                {
                    "background_index": 0,
                    "background_name": "a.osc",
                    "entries": [
                        {
                            "label": "1,0,0",
                            "x": 140.0,
                            "y": 141.0,
                            "caked_x": 23.0,
                            "caked_y": -17.5,
                            "raw_caked_x": 22.75,
                            "raw_caked_y": -17.25,
                            "stale_caked_fields": True,
                        }
                    ],
                }
            ],
        },
    )

    loaded = load_geometry_placements_file(file_path)

    entry = loaded["state"]["manual_pairs"][0]["entries"][0]
    assert entry["caked_x"] == 23.0
    assert entry["caked_y"] == -17.5
    assert entry["raw_caked_x"] == 22.75
    assert entry["raw_caked_y"] == -17.25
    assert entry["stale_caked_fields"] is True


def test_save_and_load_geometry_placements_file_preserves_canonical_trusted_pair_fields(
    tmp_path,
):
    file_path = tmp_path / "placements_canonical.json"
    save_geometry_placements_file(
        file_path,
        {
            "background_files": ["bg0.osc"],
            "manual_pairs": [
                {
                    "background_index": 0,
                    "background_name": "bg0.osc",
                    "entries": [
                        {
                            "label": "-1,0,5",
                            "hkl": (-1, 0, 5),
                            "x": 1822.0,
                            "y": 1375.0,
                            "source_reflection_index": 9,
                            "source_reflection_namespace": "full_reflection",
                            "source_reflection_is_full": True,
                            "source_branch_index": 1,
                            "source_peak_index": 1,
                        }
                    ],
                }
            ],
        },
    )

    loaded = load_geometry_placements_file(file_path)

    entry = loaded["state"]["manual_pairs"][0]["entries"][0]
    assert entry["hkl"] == [-1, 0, 5]
    assert entry["source_reflection_index"] == 9
    assert entry["source_reflection_namespace"] == "full_reflection"
    assert entry["source_reflection_is_full"] is True
    assert entry["source_branch_index"] == 1
    assert entry["source_peak_index"] == 1


def test_save_and_load_gui_state_file_preserves_fresh_one_pair_current_selection(
    tmp_path,
) -> None:
    file_path = tmp_path / "fresh_one_pair_state.json"
    save_gui_state_file(
        file_path,
        {
            "files": {
                "background_files": ["C:/tmp/bg0.osc", "C:/tmp/bg1.osc"],
                "current_background_index": 0,
            },
            "variables": {
                "geometry_fit_background_selection_var": "current",
            },
            "geometry": {
                "manual_pairs": [
                    {
                        "background_index": 0,
                        "entries": [
                            {
                                "label": "-1,0,5",
                                "hkl": (-1, 0, 5),
                                "q_group_key": ("q_group", "primary", 1, 5),
                                "x": 182.0,
                                "y": 138.0,
                                "source_reflection_index": 203,
                                "source_reflection_namespace": "full_reflection",
                                "source_reflection_is_full": True,
                                "source_branch_index": 1,
                                "source_peak_index": 1,
                            }
                        ],
                    }
                ],
                "peak_records": [],
                "q_group_rows": [],
            },
        },
    )

    loaded = load_gui_state_file(file_path)["state"]

    assert loaded["variables"]["geometry_fit_background_selection_var"] == "current"
    entry = loaded["geometry"]["manual_pairs"][0]["entries"][0]
    assert entry["hkl"] == [-1, 0, 5]
    assert entry["q_group_key"] == ["q_group", "primary", 1, 5]
    assert entry["source_reflection_index"] == 203
    assert entry["source_reflection_namespace"] == "full_reflection"
    assert entry["source_reflection_is_full"] is True
    assert entry["source_branch_index"] == 1
    assert entry["source_peak_index"] == 1
