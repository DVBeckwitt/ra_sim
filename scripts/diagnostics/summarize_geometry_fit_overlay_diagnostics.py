from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from io import StringIO
from pathlib import Path


SUMMARY_COLUMNS = [
    "run_label",
    "json_path",
    "points_count",
    "initial_frame_statuses",
    "max_err_to_background_native_px",
    "max_initial_sim_display_raw_vs_rebuilt_delta_px",
    "chosen_initial_sim_display_sources",
    "caked_projection_input_sources",
    "arrow_semantics_statuses",
    "fit_prediction_sources",
    "fit_prediction_is_dynamic_values",
    "resolver_paths",
    "handoff_accepted_count",
    "trial_source_rows_available_count",
    "objective_space",
    "gamma_residual_delta_norm_max",
    "Gamma_residual_delta_norm_max",
    "gamma_prediction_delta_px_max",
    "Gamma_prediction_delta_px_max",
    "gamma_delta_caked_deg_max",
    "Gamma_delta_caked_deg_max",
    "projector_gamma_delta_caked_deg_max",
    "projector_Gamma_delta_caked_deg_max",
    "initial_rmse",
    "final_rmse",
    "rmse_delta",
    "stale_final_sim",
    "blue_square_moved_from_raw_display",
    "stale_arrow_drawn",
]


def _records(value: object) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _finite_float(value: object) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return float(out) if math.isfinite(out) else None


def _truthy(value: object) -> bool:
    if value is True:
        return True
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _compact_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _unique_join(values: Iterable[object]) -> str:
    unique = sorted({text for value in values if (text := _compact_text(value))})
    return ";".join(unique)


def _max_finite(values: Iterable[object]) -> float | None:
    parsed = [number for value in values if (number := _finite_float(value)) is not None]
    return max(parsed) if parsed else None


def _delta_magnitude(value: object) -> float | None:
    scalar = _finite_float(value)
    if scalar is not None:
        return abs(float(scalar))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None
    first = _finite_float(value[0])
    second = _finite_float(value[1])
    if first is None or second is None:
        return None
    return float(math.hypot(first, second))


def _max_delta_magnitude(values: Iterable[object]) -> float | None:
    parsed = [number for value in values if (number := _delta_magnitude(value)) is not None]
    return max(parsed) if parsed else None


def _frame_diag(summary: Mapping[str, object]) -> Mapping[str, object]:
    final_fit_summary = summary.get("final_fit_summary")
    if isinstance(final_fit_summary, Mapping):
        frame_diag = final_fit_summary.get("frame_diag")
        if isinstance(frame_diag, Mapping):
            return frame_diag
        return final_fit_summary
    return {}


def _objective_probe_rows(summary: Mapping[str, object]) -> list[dict[str, object]]:
    objective = summary.get("objective_sensitivity")
    if not isinstance(objective, Mapping):
        return []
    rows: list[dict[str, object]] = []
    for by_var in _records(objective.get("objective_param_sensitivity_by_var")):
        var_name = by_var.get("var_name", by_var.get("param"))
        probes = _records(by_var.get("probes"))
        if probes:
            for probe in probes:
                row = dict(probe)
                row.setdefault("param", var_name)
                rows.append(row)
        else:
            row = dict(by_var)
            row.setdefault("param", var_name)
            rows.append(row)
    return rows


def _objective_max(
    rows: Sequence[Mapping[str, object]],
    *,
    param_name: str,
    key: str,
) -> float | None:
    return _max_finite(
        row.get(key)
        for row in rows
        if str(row.get("param", row.get("var_name", ""))) == param_name
    )


def _objective_delta_max(
    rows: Sequence[Mapping[str, object]],
    *,
    param_name: str,
    key: str,
) -> float | None:
    return _max_delta_magnitude(
        row.get(key)
        for row in rows
        if str(row.get("param", row.get("var_name", ""))) == param_name
    )


def _projector_delta_max(
    rows: Sequence[Mapping[str, object]],
    *,
    param_name: str,
) -> float | None:
    return _max_delta_magnitude(
        row.get("delta_caked_deg")
        for row in rows
        if str(row.get("param", row.get("var_name", ""))) == param_name
    )


def _caked_projection_input_sources(draw_audit: Sequence[Mapping[str, object]]) -> str:
    sources: list[object] = []
    for record in draw_audit:
        audit = record.get("caked_projection_audit")
        if not isinstance(audit, Mapping):
            continue
        for value in audit.values():
            if isinstance(value, Mapping):
                sources.append(value.get("caked_projection_input_source"))
    return _unique_join(sources)


def _rmse_value(frame_diag: Mapping[str, object], *keys: str) -> float | None:
    for key in keys:
        value = _finite_float(frame_diag.get(key))
        if value is not None:
            return value
    return None


def summarize_diagnostic_bundle(
    path: Path | str,
    *,
    run_label: str | None = None,
) -> dict[str, object]:
    json_path = Path(path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        payload = {}

    initial_pairs = _records(payload.get("initial_pairs_audit"))
    overlay_records = _records(payload.get("overlay_record_audit"))
    draw_audit = _records(payload.get("draw_audit"))
    resolver_records = _records(payload.get("prediction_resolver_audit"))
    objective_rows = _objective_probe_rows(payload)
    projector_rows = _records(payload.get("projector_sensitivity"))
    frame_diag = _frame_diag(payload)

    initial_frame_statuses = _unique_join(
        [
            *(record.get("frame_status") for record in initial_pairs),
            *(
                record.get("initial_sim_native_frame_status")
                for record in overlay_records
            ),
            *(
                (
                    record.get("frame_audit", {}).get("frame_status")
                    if isinstance(record.get("frame_audit"), Mapping)
                    else None
                )
                for record in initial_pairs
            ),
        ]
    )
    max_err_to_background = _max_finite(
        [
            *(record.get("err_to_background_native_px") for record in initial_pairs),
            *(
                (
                    record.get("frame_audit", {}).get("err_to_background_native_px")
                    if isinstance(record.get("frame_audit"), Mapping)
                    else None
                )
                for record in initial_pairs
            ),
        ]
    )
    max_rebuilt_delta = _max_finite(
        record.get("initial_sim_display_raw_vs_rebuilt_delta_px")
        for record in overlay_records
    )

    stale_statuses = {"locked_saved_prediction", "stale_prediction"}
    stale_arrow_drawn = any(
        str(record.get("arrow_semantics_status", "")).strip() in stale_statuses
        and not (
            record.get("stale_arrow_drawn") is False
            or _truthy(record.get("suppressed_stale_arrow"))
            or _truthy(record.get("dashed_arrow_suppressed_by_diagnostic_flag"))
        )
        for record in draw_audit
    )
    chosen_sources = [
        record.get("chosen_initial_sim_display_source") for record in overlay_records
    ]
    blue_square_moved = bool(
        (max_rebuilt_delta is not None and float(max_rebuilt_delta) > 0.25)
        and any(str(source).startswith("recomputed_from") for source in chosen_sources)
    )

    initial_rmse = _rmse_value(
        frame_diag,
        "holistic_residual_initial_rmse",
        "initial_rmse",
    )
    final_rmse = _rmse_value(
        frame_diag,
        "holistic_residual_final_rmse",
        "final_rmse",
        "rms",
    )
    rmse_delta = _rmse_value(frame_diag, "holistic_residual_delta", "rmse_delta")
    if rmse_delta is None and initial_rmse is not None and final_rmse is not None:
        rmse_delta = float(final_rmse - initial_rmse)

    row: dict[str, object] = {
        "run_label": run_label or json_path.stem.replace("geometry_fit_overlay_diagnostic_", ""),
        "json_path": str(json_path),
        "points_count": int(len(overlay_records) or len(initial_pairs)),
        "initial_frame_statuses": initial_frame_statuses,
        "max_err_to_background_native_px": max_err_to_background,
        "max_initial_sim_display_raw_vs_rebuilt_delta_px": max_rebuilt_delta,
        "chosen_initial_sim_display_sources": _unique_join(chosen_sources),
        "caked_projection_input_sources": _caked_projection_input_sources(draw_audit),
        "arrow_semantics_statuses": _unique_join(
            record.get("arrow_semantics_status") for record in draw_audit
        ),
        "fit_prediction_sources": _unique_join(
            [
                *(record.get("fit_prediction_source") for record in draw_audit),
                *(record.get("fit_prediction_source") for record in overlay_records),
                *(record.get("prediction_source") for record in resolver_records),
            ]
        ),
        "fit_prediction_is_dynamic_values": _unique_join(
            [
                *(record.get("fit_prediction_is_dynamic") for record in draw_audit),
                *(record.get("fit_prediction_is_dynamic") for record in overlay_records),
                *(
                    record.get("fit_prediction_is_dynamic_candidate")
                    for record in resolver_records
                ),
            ]
        ),
        "resolver_paths": _unique_join(record.get("resolver_path") for record in resolver_records),
        "handoff_accepted_count": sum(
            1 for record in resolver_records if _truthy(record.get("handoff_accepted"))
        ),
        "trial_source_rows_available_count": sum(
            1
            for record in resolver_records
            if _truthy(record.get("trial_source_rows_available"))
        ),
        "objective_space": _unique_join(record.get("objective_space") for record in objective_rows),
        "gamma_residual_delta_norm_max": _objective_max(
            objective_rows,
            param_name="gamma",
            key="residual_delta_norm",
        ),
        "Gamma_residual_delta_norm_max": _objective_max(
            objective_rows,
            param_name="Gamma",
            key="residual_delta_norm",
        ),
        "gamma_prediction_delta_px_max": _objective_max(
            objective_rows,
            param_name="gamma",
            key="prediction_delta_px",
        ),
        "Gamma_prediction_delta_px_max": _objective_max(
            objective_rows,
            param_name="Gamma",
            key="prediction_delta_px",
        ),
        "gamma_delta_caked_deg_max": _objective_delta_max(
            objective_rows,
            param_name="gamma",
            key="prediction_delta_caked_deg",
        ),
        "Gamma_delta_caked_deg_max": _objective_delta_max(
            objective_rows,
            param_name="Gamma",
            key="prediction_delta_caked_deg",
        ),
        "projector_gamma_delta_caked_deg_max": _projector_delta_max(
            projector_rows,
            param_name="gamma",
        ),
        "projector_Gamma_delta_caked_deg_max": _projector_delta_max(
            projector_rows,
            param_name="Gamma",
        ),
        "initial_rmse": initial_rmse,
        "final_rmse": final_rmse,
        "rmse_delta": rmse_delta,
        "stale_final_sim": any(
            _truthy(record.get("stale_final_sim")) for record in overlay_records
        ),
        "blue_square_moved_from_raw_display": blue_square_moved,
        "stale_arrow_drawn": stale_arrow_drawn,
    }
    return row


def summarize_diagnostic_bundles(paths: Iterable[Path | str]) -> list[dict[str, object]]:
    return [summarize_diagnostic_bundle(path) for path in paths]


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_markdown_table(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "| " + " | ".join(SUMMARY_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in SUMMARY_COLUMNS) + " |",
    ]
    for row in rows:
        cells = [_format_cell(row.get(column)).replace("|", "\\|") for column in SUMMARY_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_csv(rows: Sequence[Mapping[str, object]]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: _format_cell(row.get(column)) for column in SUMMARY_COLUMNS})
    return buffer.getvalue()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize geometry-fit overlay diagnostic JSON bundles."
    )
    parser.add_argument(
        "json_paths",
        nargs="+",
        help="One or more geometry_fit_overlay_diagnostic_*.json files.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="Output table format.",
    )
    args = parser.parse_args(argv)

    rows = summarize_diagnostic_bundles(args.json_paths)
    if args.format == "csv":
        sys.stdout.write(format_csv(rows))
    else:
        sys.stdout.write(format_markdown_table(rows))
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
