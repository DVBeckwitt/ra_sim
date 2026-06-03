"""Internal helpers for geometry-fit manual dataset assembly."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GeometryManualDatasetInputSnapshot:
    """Normalized manual-pair inputs consumed by dataset assembly."""

    background_idx: int
    raw_selected_entries: list[object]
    selected_entry_inputs: list[dict[str, object]]
    selected_entries: list[dict[str, object]]
    manual_picker_truth_pairs: object
    manual_picker_truth_by_order: dict[object, dict[str, object]]
    use_caked_display: bool
    baseline_fit_params_i: dict[str, object]
    params_i: dict[str, object]
    theta_offset: float
    theta_initial: float
    reference_a: float | None
    reference_c: float | None
    reference_lambda: float | None

    def required_pairs_callback_payload(self) -> list[dict[str, object]]:
        callback_pairs: list[dict[str, object]] = []
        for entry in self.selected_entries:
            copied = dict(entry)
            copied.pop("source_label", None)
            callback_pairs.append(copied)
        return callback_pairs


def collect_geometry_manual_dataset_inputs(
    *,
    background_idx: int,
    theta_base: float,
    base_fit_params: Mapping[str, object] | None,
    manual_dataset_bindings: Any,
    manual_fit_requires_caked_space: bool,
    pair_enabled: Callable[[object], bool],
    background_detector_pair_for_frame: Callable[[Mapping[str, object], str], object | None],
    entry_source_label: Callable[[Mapping[str, object] | None], str],
    build_manual_picker_truth_pairs: Callable[..., object],
    truth_by_order_key: Callable[[object], dict[object, dict[str, object]]],
    finite_float: Callable[[object], float | None],
    raw_selected_entries: Sequence[object] | None = None,
    use_caked_display: bool | None = None,
) -> GeometryManualDatasetInputSnapshot:
    if use_caked_display is None:
        use_caked_display = bool(manual_fit_requires_caked_space)
        pick_uses_caked_space = getattr(manual_dataset_bindings, "pick_uses_caked_space", None)
        if callable(pick_uses_caked_space):
            try:
                use_caked_display = bool(use_caked_display or pick_uses_caked_space())
            except Exception:
                pass

    selected_raw_entries = (
        list(raw_selected_entries)
        if raw_selected_entries is not None
        else [
            entry
            for entry in (manual_dataset_bindings.geometry_manual_pairs_for_index(background_idx) or ())
            if pair_enabled(entry)
        ]
    )
    if not selected_raw_entries:
        raise RuntimeError(f"background {background_idx + 1} has no saved manual geometry pairs")

    selected_entry_inputs: list[dict[str, object]] = []
    refresh_pair_entry = getattr(
        manual_dataset_bindings,
        "geometry_manual_refresh_pair_entry",
        None,
    )
    if callable(refresh_pair_entry):
        for raw_entry in selected_raw_entries:
            raw_saved_entry = dict(raw_entry) if isinstance(raw_entry, Mapping) else None
            refreshed = refresh_pair_entry(raw_entry)
            normalized_entry = (
                dict(refreshed)
                if isinstance(refreshed, Mapping)
                else (dict(raw_entry) if isinstance(raw_entry, Mapping) else None)
            )
            if normalized_entry is None:
                continue
            if background_detector_pair_for_frame(normalized_entry, "native_detector") is not None:
                normalized_entry.setdefault(
                    "background_detector_frame_provenance",
                    "geometry_manual_refresh_pair_entry",
                )
            selected_entry_inputs.append(
                {
                    "raw_saved_entry": (
                        raw_saved_entry
                        if isinstance(raw_saved_entry, dict)
                        else dict(normalized_entry)
                    ),
                    "entry": dict(normalized_entry),
                }
            )
    else:
        for raw_entry in selected_raw_entries:
            if not isinstance(raw_entry, Mapping):
                continue
            selected_entry_inputs.append(
                {
                    "raw_saved_entry": dict(raw_entry),
                    "entry": dict(raw_entry),
                }
            )

    selected_entries = [
        {
            **dict(item["entry"]),
            "source_label": entry_source_label(
                item["entry"] if isinstance(item.get("entry"), Mapping) else None
            ),
        }
        for item in selected_entry_inputs
        if isinstance(item.get("entry"), Mapping)
    ]

    manual_picker_truth_pairs = build_manual_picker_truth_pairs(
        background_idx,
        [
            dict(item["raw_saved_entry"])
            for item in selected_entry_inputs
            if isinstance(item.get("raw_saved_entry"), Mapping)
        ],
        refresh_pair_entry=refresh_pair_entry,
    )
    manual_picker_truth_by_order = truth_by_order_key(manual_picker_truth_pairs)

    baseline_fit_params_i = dict(base_fit_params or {})
    params_i = dict(base_fit_params or {})
    theta_offset = float(params_i.get("theta_offset", 0.0))
    theta_initial = float(theta_base + theta_offset)
    params_i["theta_initial"] = float(theta_initial)

    return GeometryManualDatasetInputSnapshot(
        background_idx=int(background_idx),
        raw_selected_entries=list(selected_raw_entries),
        selected_entry_inputs=selected_entry_inputs,
        selected_entries=selected_entries,
        manual_picker_truth_pairs=manual_picker_truth_pairs,
        manual_picker_truth_by_order=manual_picker_truth_by_order,
        use_caked_display=bool(use_caked_display),
        baseline_fit_params_i=baseline_fit_params_i,
        params_i=params_i,
        theta_offset=float(theta_offset),
        theta_initial=float(theta_initial),
        reference_a=finite_float(params_i.get("a")),
        reference_c=finite_float(params_i.get("c")),
        reference_lambda=finite_float(params_i.get("lambda")),
    )
