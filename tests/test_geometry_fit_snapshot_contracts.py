from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ra_sim.gui import geometry_fit, geometry_fit_contracts
from tests.helpers.geometry_fit_snapshots import normalize_geometry_fit_snapshot


def test_geometry_fit_snapshot_normalizes_unstable_strings_and_mapping_order() -> None:
    payload = {
        "z": "object at 0x7ffdeadc0de and geometry_fit_trace_20260528_123456.jsonl",
        "a": {2: "two", "one": Path("artifacts/current.log")},
    }

    assert normalize_geometry_fit_snapshot(payload) == {
        "a": {
            "int:2": "two",
            "one": "artifacts/current.log",
        },
        "z": "object at 0x<addr> and geometry_fit_trace_<timestamp>.jsonl",
    }


def test_geometry_fit_snapshot_preserves_non_string_mapping_key_identity() -> None:
    payload = {0: "integer background", "0": "string background"}

    assert normalize_geometry_fit_snapshot(payload) == {
        "0": "string background",
        "int:0": "integer background",
    }


def test_geometry_fit_snapshot_rejects_typed_key_alias_collision() -> None:
    with pytest.raises(ValueError, match="snapshot key collision"):
        normalize_geometry_fit_snapshot({0: "integer background", "int:0": "string alias"})


def test_geometry_fit_snapshot_rejects_normalized_string_key_collision() -> None:
    with pytest.raises(ValueError, match="snapshot key collision"):
        normalize_geometry_fit_snapshot(
            {
                "x20260528_123456": "first trace",
                "x20260529_123456": "second trace",
            }
        )


def test_geometry_fit_snapshot_normalizes_arrays_nonfinite_values_and_namespaces() -> None:
    payload = {
        "array": np.asarray([[1.1234567890123, np.nan], [np.inf, -np.inf]]),
        "namespace": SimpleNamespace(metric_unit="px", matched_pair_count=0),
        "values": (np.float64(2.5), math.nan, math.inf, -math.inf),
    }

    assert normalize_geometry_fit_snapshot(payload) == {
        "array": {
            "__ndarray__": {
                "dtype": "float64",
                "shape": [2, 2],
                "values": [[1.123456789012, "NaN"], ["Infinity", "-Infinity"]],
            },
        },
        "namespace": {"matched_pair_count": 0, "metric_unit": "px"},
        "values": [2.5, "NaN", "Infinity", "-Infinity"],
    }


def test_geometry_fit_reexports_contract_types_for_existing_call_sites() -> None:
    names = [
        "GeometryFitStageCallback",
        "GeometryFitPreparedRun",
        "GeometryFitRuntimeManualDatasetBindings",
        "GeometryFitRuntimePreparationBindings",
        "GeometryFitRuntimeValueBindings",
        "GeometryFitRuntimeSolverInputs",
        "GeometryFitSolverRequest",
        "GeometryFitPreparationResult",
        "GeometryFitSourceRowRebuildResult",
        "GeometryFitBackgroundCacheBundle",
        "GeometryFitPostprocessResult",
        "GeometryFitRuntimeResultBindings",
        "GeometryFitRuntimeUiBindings",
        "GeometryFitRuntimePostprocessConfig",
        "GeometryFitRuntimeExecutionSetup",
        "GeometryFitRuntimeValueCallbacks",
        "GeometryFitRuntimeActionExecutionBindings",
        "GeometryFitRuntimeActionBindings",
        "GeometryFitRuntimeActionResult",
        "GeometryFitActionNotice",
        "GeometryFitRuntimeApplyResult",
        "GeometryFitSweepApplyResult",
        "GeometryFitRuntimeExecutionResult",
        "GeometryToolActionRuntimeCallbacks",
        "GeometryFitRuntimeHistoryCallbacks",
    ]

    for name in names:
        assert getattr(geometry_fit, name) is getattr(geometry_fit_contracts, name)
