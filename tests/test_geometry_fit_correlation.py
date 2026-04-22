from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ra_sim.gui.geometry_fit_correlation import (
    GeometryFitCorrelationResult,
    available_geometry_fit_parameters,
    write_correlation_artifacts,
)


def test_available_geometry_fit_parameters_uses_active_theta_offset() -> None:
    context = {
        "prepared_run": SimpleNamespace(
            fit_params={
                "center_x": 1024.0,
                "center_y": 1031.0,
                "theta_initial": 4.0,
                "theta_offset": 0.0,
                "gamma": 1.0,
            }
        ),
        "saved_var_names": ["theta_offset", "gamma"],
    }

    parameters = available_geometry_fit_parameters(context)

    assert parameters == ["center_x", "center_y", "gamma", "theta_offset"]
    assert "theta_initial" not in parameters


def test_write_correlation_artifacts_exports_matrix_pairs_and_parameter_rows(
    tmp_path: Path,
) -> None:
    summary = {
        "status": "ok",
        "diagnostic_scope": "data_only_all_selectable",
        "includes_priors": False,
        "parameter_entries": [
            {
                "name": "a",
                "index": 0,
                "valid": True,
                "column_norm": 2.0,
                "relative_sensitivity": 1.0,
                "std_u": 0.1,
                "std_theta": 0.01,
            },
            {
                "name": "c",
                "index": 1,
                "valid": True,
                "column_norm": 1.0,
                "relative_sensitivity": 0.5,
                "std_u": 0.2,
                "std_theta": 0.02,
            },
            {
                "name": "gamma",
                "index": 2,
                "valid": True,
                "column_norm": 0.5,
                "relative_sensitivity": 0.25,
                "std_u": 0.3,
                "std_theta": 0.03,
            },
        ],
        "correlation_u": np.array(
            [
                [1.0, 0.95, -0.1],
                [0.95, 1.0, 0.4],
                [-0.1, 0.4, 1.0],
            ]
        ),
        "all_correlation_pairs": [
            {
                "parameter_i": "a",
                "index_i": 0,
                "parameter_j": "c",
                "index_j": 1,
                "correlation": 0.95,
                "abs_correlation": 0.95,
                "high_correlation": True,
            },
            {
                "parameter_i": "c",
                "index_i": 1,
                "parameter_j": "gamma",
                "index_j": 2,
                "correlation": 0.4,
                "abs_correlation": 0.4,
                "high_correlation": False,
            },
            {
                "parameter_i": "a",
                "index_i": 0,
                "parameter_j": "gamma",
                "index_j": 2,
                "correlation": -0.1,
                "abs_correlation": 0.1,
                "high_correlation": False,
            },
        ],
    }
    result = GeometryFitCorrelationResult(
        state_path=Path("state.json"),
        background_index=None,
        parameters=["a", "c", "gamma"],
        summary=summary,
        request_summary={"candidate_param_names": ["a", "c", "gamma"]},
        solver_probe_records=[],
        metadata={
            "status": "ok",
            "parameters": ["a", "c", "gamma"],
            "pair_count": 3,
            "high_correlation_pair_count": 1,
        },
    )

    paths = write_correlation_artifacts(result, tmp_path)

    assert set(paths) == {
        "summary_json",
        "correlation_matrix",
        "correlation_pairs",
        "parameter_sensitivity",
    }
    assert all(path.exists() for path in paths.values())

    with paths["correlation_matrix"].open(newline="", encoding="utf-8") as handle:
        matrix_rows = list(csv.reader(handle))
    assert matrix_rows[0] == ["parameter", "a", "c", "gamma"]
    assert matrix_rows[1][:3] == ["a", "1.0", "0.95"]

    with paths["correlation_pairs"].open(newline="", encoding="utf-8") as handle:
        pair_rows = list(csv.DictReader(handle))
    assert pair_rows[0]["parameter_i"] == "a"
    assert pair_rows[0]["parameter_j"] == "c"
    assert pair_rows[0]["high_correlation"] == "True"

    with paths["parameter_sensitivity"].open(newline="", encoding="utf-8") as handle:
        sensitivity_rows = list(csv.DictReader(handle))
    assert [row["name"] for row in sensitivity_rows] == ["a", "c", "gamma"]

    with paths["summary_json"].open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["metadata"]["high_correlation_pair_count"] == 1
    assert payload["summary"]["correlation_u"][0][1] == 0.95
