#!/usr/bin/env python3
"""Aggregate data-only geometry-fit Jacobian diagnostics across scenarios.

This utility is intentionally offline and scenario-driven. Feed it
representative nominal geometries plus representative point selections, then
use the aggregate report to choose default priors, nominal scales, and a staged
release order. Do not treat the aggregate output as a dataset-specific final
uncertainty statement.

Scenario file schema (YAML or JSON):

scenarios:
  - name: nominal-angle
    image_size: 512
    miller: [[1, 0, 0], [0, 1, 0]]
    intensities: [1.0, 0.8]
    params: {... complete geometry parameter dict ...}
    measured_peaks:
      - [1, 0, 0, 128.0, 196.0]
      - [0, 1, 0, 260.0, 188.0]
    var_names: ["gamma"]
    candidate_param_names: ["gamma", "Gamma", "corto_detector"]
    refinement_config: {}

  - name: nominal-point-match
    image_size: 512
    miller: [[1, 0, 0]]
    intensities: [1.0]
    params: {... complete geometry parameter dict ...}
    measured_peaks:
      - {label: "1,0,0", x: 240.0, y: 170.0}
    experimental_image: zeros
    var_names: []
    candidate_param_names: ["gamma", "Gamma"]
    refinement_config: {}
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from ra_sim.config import get_instrument_config
from ra_sim.fitting.optimization import fit_geometry_parameters


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario_file", type=Path, help="YAML/JSON scenario file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the aggregate report. Defaults to stdout.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "yaml"),
        default="yaml",
        help="Output format for the aggregate report.",
    )
    return parser.parse_args()


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[str(key)] = _deep_merge(
                dict(merged.get(key, {})),
                dict(value),
            )
        else:
            merged[str(key)] = copy.deepcopy(value)
    return merged


def _load_structured_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise ValueError(f"Scenario file must decode to a mapping: {path}")
    return dict(data)


def _maybe_load_image(raw_value: Any, *, image_size: int, measured_peaks: Sequence[Any]) -> np.ndarray | None:
    if raw_value is None:
        if measured_peaks and isinstance(measured_peaks[0], Mapping):
            return np.zeros((image_size, image_size), dtype=np.float64)
        return None
    if isinstance(raw_value, str) and raw_value.strip().lower() == "zeros":
        return np.zeros((image_size, image_size), dtype=np.float64)
    if isinstance(raw_value, Mapping):
        path_value = raw_value.get("path")
        if isinstance(path_value, str) and path_value:
            image_path = Path(path_value)
            if image_path.suffix.lower() == ".npy":
                return np.asarray(np.load(image_path), dtype=np.float64)
            try:
                from PIL import Image

                return np.asarray(Image.open(image_path), dtype=np.float64)
            except Exception as exc:  # pragma: no cover - best-effort path loading
                raise ValueError(f"Failed to load experimental image from {image_path}: {exc}") from exc
    return np.asarray(raw_value, dtype=np.float64)


def _scenario_report(result: Any, *, name: str) -> dict[str, Any]:
    data_summary = (
        dict(result.data_only_identifiability_summary)
        if isinstance(getattr(result, "data_only_identifiability_summary", None), Mapping)
        else {}
    )
    solver_summary = (
        dict(result.identifiability_summary)
        if isinstance(getattr(result, "identifiability_summary", None), Mapping)
        else {}
    )
    return {
        "name": str(name),
        "success": bool(getattr(result, "success", False)),
        "message": str(getattr(result, "message", "")),
        "cost": float(getattr(result, "cost", np.nan)),
        "solver_status": str(solver_summary.get("status", "")),
        "data_only_status": str(data_summary.get("status", "")),
        "weak_combinations": list(data_summary.get("weak_combinations", [])),
        "high_correlation_pairs": list(data_summary.get("high_correlation_pairs", [])),
        "next_stage_recommendations": list(getattr(result, "next_stage_recommendations", [])),
        "chosen_discrete_mode": copy.deepcopy(getattr(result, "chosen_discrete_mode", None)),
    }


def _canonicalize_combo(combo: Mapping[str, Any]) -> str:
    entries = [(str(name), float(weight)) for name, weight in combo.items()]
    if not entries:
        return ""
    pivot_name, pivot_weight = max(entries, key=lambda item: abs(item[1]))
    sign = -1.0 if pivot_weight < 0.0 else 1.0
    normalized = sorted((name, round(sign * weight, 3)) for name, weight in entries)
    return "|".join(f"{name}:{weight:+.3f}" for name, weight in normalized)


def _correlation_blocks(high_pairs: Sequence[Mapping[str, Any]]) -> list[tuple[str, ...]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for pair in high_pairs:
        left = str(pair.get("name_i", "")).strip()
        right = str(pair.get("name_j", "")).strip()
        if not left or not right:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)
    blocks: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        block: list[str] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            block.append(current)
            stack.extend(sorted(adjacency[current] - seen))
        if len(block) >= 2:
            blocks.append(tuple(sorted(block)))
    return blocks


def _aggregate_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    weak_combo_counts: Counter[str] = Counter()
    weak_combo_examples: dict[str, dict[str, Any]] = {}
    block_counts: Counter[tuple[str, ...]] = Counter()
    sensitivity_sum: dict[str, float] = defaultdict(float)
    sensitivity_count: dict[str, int] = defaultdict(int)
    thaw_order_counts: Counter[tuple[str, ...]] = Counter()

    for report in reports:
        for weak in report.get("weak_combinations", []):
            if not isinstance(weak, Mapping):
                continue
            combo = weak.get("combo", {})
            if not isinstance(combo, Mapping):
                continue
            key = _canonicalize_combo(combo)
            if not key:
                continue
            weak_combo_counts[key] += 1
            weak_combo_examples.setdefault(
                key,
                {
                    "combo": {str(name): float(weight) for name, weight in combo.items()},
                    "sv_rel": float(weak.get("sv_rel", np.nan)),
                },
            )

        for block in _correlation_blocks(report.get("high_correlation_pairs", [])):
            block_counts[block] += 1

        for recommendation in report.get("next_stage_recommendations", []):
            params = tuple(str(name) for name in recommendation.get("params", []))
            if params:
                thaw_order_counts[params] += 1

    instrument_fit_geometry = (
        get_instrument_config()
        .get("instrument", {})
        .get("fit", {})
        .get("geometry", {})
    )
    candidate_names = instrument_fit_geometry.get("candidate_param_names", [])
    if isinstance(candidate_names, Sequence) and not isinstance(candidate_names, (str, bytes)):
        for name in candidate_names:
            sensitivity_sum[str(name)] += 0.0
            sensitivity_count[str(name)] += 0

    for report in reports:
        data_summary = report.get("data_only_summary", {})
        if not isinstance(data_summary, Mapping):
            continue
        for entry in data_summary.get("parameter_entries", []):
            if not isinstance(entry, Mapping):
                continue
            name = str(entry.get("name", "")).strip()
            if not name:
                continue
            sensitivity_sum[name] += float(entry.get("relative_sensitivity", 0.0))
            sensitivity_count[name] += 1

    average_sensitivity = []
    for name in sorted(sensitivity_sum):
        count = max(int(sensitivity_count.get(name, 0)), 1)
        average_sensitivity.append(
            {
                "name": str(name),
                "mean_relative_sensitivity": float(sensitivity_sum[name] / count),
                "sample_count": int(sensitivity_count.get(name, 0)),
            }
        )
    average_sensitivity.sort(
        key=lambda item: (-float(item["mean_relative_sensitivity"]), str(item["name"]))
    )

    return {
        "frequent_weak_combinations": [
            {
                "count": int(count),
                "combo_key": key,
                "combo": copy.deepcopy(weak_combo_examples[key]["combo"]),
                "example_sv_rel": float(weak_combo_examples[key]["sv_rel"]),
            }
            for key, count in weak_combo_counts.most_common()
        ],
        "frequent_correlation_blocks": [
            {"count": int(count), "params": list(block)}
            for block, count in block_counts.most_common()
        ],
        "average_sensitivity_by_parameter": average_sensitivity,
        "typical_recommended_thaw_order": [
            {"count": int(count), "params": list(params)}
            for params, count in thaw_order_counts.most_common()
        ],
    }


def _run_scenario(
    scenario: Mapping[str, Any],
    *,
    base_refinement_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    name = str(scenario.get("name", "scenario"))
    image_size = int(scenario["image_size"])
    measured_peaks = list(scenario.get("measured_peaks", []))
    experimental_image = _maybe_load_image(
        scenario.get("experimental_image"),
        image_size=image_size,
        measured_peaks=measured_peaks,
    )
    refinement_config = _deep_merge(
        base_refinement_config,
        dict(scenario.get("refinement_config", {})),
    )
    result = fit_geometry_parameters(
        np.asarray(scenario["miller"], dtype=np.float64),
        np.asarray(scenario["intensities"], dtype=np.float64),
        image_size,
        copy.deepcopy(dict(scenario["params"])),
        measured_peaks=measured_peaks,
        var_names=list(scenario.get("var_names", [])),
        experimental_image=experimental_image,
        refinement_config=refinement_config,
        candidate_param_names=list(scenario.get("candidate_param_names", scenario.get("var_names", []))),
    )
    report = _scenario_report(result, name=name)
    report["data_only_summary"] = copy.deepcopy(
        getattr(result, "data_only_identifiability_summary", {})
    )
    return report, {
        "name": name,
        "success": bool(getattr(result, "success", False)),
        "cost": float(getattr(result, "cost", np.nan)),
    }


def main() -> int:
    args = _parse_args()
    payload = _load_structured_file(args.scenario_file)
    raw_scenarios = payload.get("scenarios", [])
    if not isinstance(raw_scenarios, Sequence) or isinstance(raw_scenarios, (str, bytes)):
        raise ValueError("Scenario file must contain a 'scenarios' list.")

    base_refinement_config = (
        get_instrument_config()
        .get("instrument", {})
        .get("fit", {})
        .get("geometry", {})
    )
    reports: list[dict[str, Any]] = []
    run_summary: list[dict[str, Any]] = []
    for raw_scenario in raw_scenarios:
        if not isinstance(raw_scenario, Mapping):
            continue
        report, summary = _run_scenario(
            dict(raw_scenario),
            base_refinement_config=base_refinement_config,
        )
        reports.append(report)
        run_summary.append(summary)

    output = {
        "note": (
            "Offline aggregate only. Use this to choose default priors, scales, and staged "
            "release order. Do not treat it as a dataset-specific identifiability claim."
        ),
        "scenario_count": int(len(reports)),
        "run_summary": run_summary,
        "aggregate": _aggregate_reports(reports),
        "scenarios": reports,
    }
    if args.format == "json":
        text = json.dumps(output, indent=2, sort_keys=True)
    else:
        text = yaml.safe_dump(output, sort_keys=False)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
