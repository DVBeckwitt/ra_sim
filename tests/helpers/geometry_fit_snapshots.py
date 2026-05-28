from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np


_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]+")
_GEOMETRY_FIT_TIMESTAMP_RE = re.compile(r"(?<!\d)20\d{6}_\d{6}(?!\d)")


def _stable_string(value: str) -> str:
    text = _ADDRESS_RE.sub("0x<addr>", str(value))
    return _GEOMETRY_FIT_TIMESTAMP_RE.sub("<timestamp>", text)


def _normalize_float(value: float, *, float_digits: int) -> float | str:
    number = float(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "Infinity" if number > 0.0 else "-Infinity"
    rounded = round(number, int(float_digits))
    return 0.0 if rounded == 0.0 else rounded


def _json_sort_key(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _normalize_mapping_key(key: object, *, float_digits: int) -> str:
    if isinstance(key, str):
        return _stable_string(key)
    normalized = normalize_geometry_fit_snapshot(key, float_digits=float_digits)
    return f"{type(key).__name__}:{_json_sort_key(normalized)}"


def _normalize_array(
    value: np.ndarray,
    *,
    float_digits: int,
    max_array_values: int,
) -> dict[str, object]:
    array = np.asarray(value)
    payload: dict[str, object] = {
        "shape": [int(item) for item in array.shape],
        "dtype": str(array.dtype),
    }
    if array.size <= int(max_array_values):
        payload["values"] = normalize_geometry_fit_snapshot(
            array.tolist(),
            float_digits=float_digits,
            max_array_values=max_array_values,
        )
        return {"__ndarray__": payload}

    if not np.issubdtype(array.dtype, np.number):
        return {"__ndarray__": payload}

    numeric = np.asarray(array, dtype=float)
    finite_mask = np.isfinite(numeric)
    finite_values = numeric[finite_mask]
    payload.update(
        {
            "finite_count": int(finite_mask.sum()),
            "nan_count": int(np.isnan(numeric).sum()),
            "pos_inf_count": int(np.isposinf(numeric).sum()),
            "neg_inf_count": int(np.isneginf(numeric).sum()),
        }
    )
    if finite_values.size:
        payload["min"] = _normalize_float(float(finite_values.min()), float_digits=float_digits)
        payload["max"] = _normalize_float(float(finite_values.max()), float_digits=float_digits)
        payload["mean"] = _normalize_float(
            float(finite_values.mean()),
            float_digits=float_digits,
        )
    return {"__ndarray__": payload}


def normalize_geometry_fit_snapshot(
    value: object,
    *,
    float_digits: int = 12,
    max_array_values: int = 64,
) -> object:
    """Return a JSON-compatible geometry-fit payload for semantic comparisons."""

    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, str):
        return _stable_string(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return _normalize_array(
            value,
            float_digits=float_digits,
            max_array_values=max_array_values,
        )
    if isinstance(value, np.generic):
        return normalize_geometry_fit_snapshot(
            value.item(),
            float_digits=float_digits,
            max_array_values=max_array_values,
        )
    if isinstance(value, float):
        return _normalize_float(value, float_digits=float_digits)
    if is_dataclass(value) and not isinstance(value, type):
        return normalize_geometry_fit_snapshot(
            asdict(value),
            float_digits=float_digits,
            max_array_values=max_array_values,
        )
    if isinstance(value, SimpleNamespace):
        return normalize_geometry_fit_snapshot(
            vars(value),
            float_digits=float_digits,
            max_array_values=max_array_values,
        )
    if isinstance(value, Mapping):
        normalized_items: list[tuple[str, object]] = []
        for raw_key, raw_value in value.items():
            normalized_key = _normalize_mapping_key(raw_key, float_digits=float_digits)
            normalized_value = normalize_geometry_fit_snapshot(
                raw_value,
                float_digits=float_digits,
                max_array_values=max_array_values,
            )
            normalized_items.append((normalized_key, normalized_value))
        return {key: item for key, item in sorted(normalized_items, key=lambda pair: pair[0])}
    if isinstance(value, set | frozenset):
        normalized_values = [
            normalize_geometry_fit_snapshot(
                item,
                float_digits=float_digits,
                max_array_values=max_array_values,
            )
            for item in value
        ]
        return sorted(normalized_values, key=_json_sort_key)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [
            normalize_geometry_fit_snapshot(
                item,
                float_digits=float_digits,
                max_array_values=max_array_values,
            )
            for item in value
        ]
    if isinstance(value, Callable):
        module = getattr(value, "__module__", "")
        qualname = getattr(value, "__qualname__", repr(value))
        return _stable_string(f"<callable {module}.{qualname}>")
    return _stable_string(repr(value))
