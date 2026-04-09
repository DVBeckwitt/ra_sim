"""Central debug and logging control resolution for RA-SIM."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ra_sim.config import get_config_bundle, get_dir


_DEBUG_DEFAULTS: dict[str, Any] = {
    "global": {"disable_all": False},
    "console": {"enabled": False},
    "runtime_update_trace": {"enabled": True},
    "geometry_fit": {"log_files": True, "extra_sections": True},
    "mosaic_fit": {"log_files": True},
    "projection_debug": {"enabled": True},
    "diffraction_debug_csv": {"enabled": True},
    "intersection_cache": {"enabled": True, "log_dir": None},
}

_TRUTHY_VALUES = {"1", "true", "yes", "on"}
_FALSY_VALUES = {"", "0", "false", "no", "off"}


def env_flag_enabled(
    name: str,
    env: Mapping[str, object] | None = None,
) -> bool:
    """Return whether one environment-style flag is set to a truthy value."""

    source = os.environ if env is None else env
    value = str(source.get(name, "")).strip().lower()
    return bool(value) and value not in {"0", "false", "no", "off"}


def _coerce_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUTHY_VALUES:
            return True
        if normalized in _FALSY_VALUES:
            return False
        return default
    return bool(value)


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _env_source(env: Mapping[str, object] | None) -> Mapping[str, object]:
    return os.environ if env is None else env


def _bundle_debug_config() -> Mapping[str, Any]:
    try:
        return _as_mapping(get_config_bundle().debug)
    except Exception:
        return {}


def _bundle_instrument_config() -> Mapping[str, Any]:
    try:
        return _as_mapping(get_config_bundle().instrument)
    except Exception:
        return {}


def _debug_root(debug_config: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    raw = _bundle_debug_config() if debug_config is None else _as_mapping(debug_config)
    nested = raw.get("debug")
    return _as_mapping(nested) if isinstance(nested, Mapping) else raw


def _instrument_root(
    instrument_config: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    raw = (
        _bundle_instrument_config()
        if instrument_config is None
        else _as_mapping(instrument_config)
    )
    nested = raw.get("instrument")
    return _as_mapping(nested) if isinstance(nested, Mapping) else raw


def _lookup_mapping_value(
    mapping: Mapping[str, Any],
    path: tuple[str, ...],
) -> tuple[bool, object]:
    current: object = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return False, None
        current = current[key]
    return True, current


def _resolve_config_bool(
    path: tuple[str, ...],
    *,
    default: bool,
    debug_config: Mapping[str, Any] | None = None,
) -> bool:
    present, value = _lookup_mapping_value(_debug_root(debug_config), path)
    if not present:
        return default
    return _coerce_bool(value, default)


def _env_override_bool(
    name: str,
    env: Mapping[str, object] | None = None,
) -> bool | None:
    source = _env_source(env)
    if name not in source:
        return None
    return env_flag_enabled(name, source)


def is_logging_disabled(
    env: Mapping[str, object] | None = None,
    *,
    debug_config: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether the global debug/logging kill-switch is active."""

    if _resolve_config_bool(
        ("global", "disable_all"),
        default=_DEBUG_DEFAULTS["global"]["disable_all"],
        debug_config=debug_config,
    ):
        return True
    return env_flag_enabled("RA_SIM_DISABLE_ALL_LOGGING", env) or env_flag_enabled(
        "RA_SIM_DISABLE_LOGGING",
        env,
    )


def console_debug_enabled(
    env: Mapping[str, object] | None = None,
    *,
    debug_config: Mapping[str, Any] | None = None,
) -> bool:
    if is_logging_disabled(env, debug_config=debug_config):
        return False
    override = _env_override_bool("RA_SIM_DEBUG", env)
    if override is not None:
        return override
    return _resolve_config_bool(
        ("console", "enabled"),
        default=_DEBUG_DEFAULTS["console"]["enabled"],
        debug_config=debug_config,
    )


def runtime_update_trace_logging_enabled(
    env: Mapping[str, object] | None = None,
    *,
    debug_config: Mapping[str, Any] | None = None,
) -> bool:
    if is_logging_disabled(env, debug_config=debug_config):
        return False
    return _resolve_config_bool(
        ("runtime_update_trace", "enabled"),
        default=_DEBUG_DEFAULTS["runtime_update_trace"]["enabled"],
        debug_config=debug_config,
    )


def geometry_fit_log_files_enabled(
    env: Mapping[str, object] | None = None,
    *,
    debug_config: Mapping[str, Any] | None = None,
) -> bool:
    if is_logging_disabled(env, debug_config=debug_config):
        return False
    return _resolve_config_bool(
        ("geometry_fit", "log_files"),
        default=_DEBUG_DEFAULTS["geometry_fit"]["log_files"],
        debug_config=debug_config,
    )


def geometry_fit_extra_sections_enabled(
    geometry_runtime_cfg: Mapping[str, Any] | None = None,
    env: Mapping[str, object] | None = None,
    *,
    debug_config: Mapping[str, Any] | None = None,
    instrument_config: Mapping[str, Any] | None = None,
) -> bool:
    if not geometry_fit_log_files_enabled(env, debug_config=debug_config):
        return False

    present, value = _lookup_mapping_value(
        _debug_root(debug_config),
        ("geometry_fit", "extra_sections"),
    )
    if present:
        return _coerce_bool(value, _DEBUG_DEFAULTS["geometry_fit"]["extra_sections"])

    fit_cfg = (
        _as_mapping(geometry_runtime_cfg)
        if isinstance(geometry_runtime_cfg, Mapping)
        else _as_mapping(_instrument_root(instrument_config).get("fit"))
    )
    geometry_cfg = fit_cfg.get("geometry")
    if isinstance(geometry_cfg, Mapping):
        fit_cfg = geometry_cfg
    return _coerce_bool(
        fit_cfg.get("debug_logging", fit_cfg.get("debug_mode", False)),
        False,
    )


def mosaic_fit_log_files_enabled(
    env: Mapping[str, object] | None = None,
    *,
    debug_config: Mapping[str, Any] | None = None,
) -> bool:
    if is_logging_disabled(env, debug_config=debug_config):
        return False
    return _resolve_config_bool(
        ("mosaic_fit", "log_files"),
        default=_DEBUG_DEFAULTS["mosaic_fit"]["log_files"],
        debug_config=debug_config,
    )


def projection_debug_logging_enabled(
    env: Mapping[str, object] | None = None,
    *,
    debug_config: Mapping[str, Any] | None = None,
) -> bool:
    if is_logging_disabled(env, debug_config=debug_config):
        return False
    override = _env_override_bool("RA_SIM_DISABLE_PROJECTION_DEBUG", env)
    if override is not None:
        return not override
    return _resolve_config_bool(
        ("projection_debug", "enabled"),
        default=_DEBUG_DEFAULTS["projection_debug"]["enabled"],
        debug_config=debug_config,
    )


def diffraction_debug_csv_logging_enabled(
    env: Mapping[str, object] | None = None,
    *,
    debug_config: Mapping[str, Any] | None = None,
) -> bool:
    if is_logging_disabled(env, debug_config=debug_config):
        return False
    return _resolve_config_bool(
        ("diffraction_debug_csv", "enabled"),
        default=_DEBUG_DEFAULTS["diffraction_debug_csv"]["enabled"],
        debug_config=debug_config,
    )


def intersection_cache_logging_enabled(
    env: Mapping[str, object] | None = None,
    *,
    debug_config: Mapping[str, Any] | None = None,
) -> bool:
    if is_logging_disabled(env, debug_config=debug_config):
        return False
    override = _env_override_bool("RA_SIM_LOG_INTERSECTION_CACHE", env)
    if override is not None:
        return override
    return _resolve_config_bool(
        ("intersection_cache", "enabled"),
        default=_DEBUG_DEFAULTS["intersection_cache"]["enabled"],
        debug_config=debug_config,
    )


def resolve_intersection_cache_log_root(
    env: Mapping[str, object] | None = None,
    *,
    debug_config: Mapping[str, Any] | None = None,
) -> Path:
    source = _env_source(env)
    configured_root = str(source.get("RA_SIM_INTERSECTION_CACHE_LOG_DIR", "")).strip()
    if configured_root:
        return Path(os.path.expanduser(configured_root)).expanduser()

    present, value = _lookup_mapping_value(
        _debug_root(debug_config),
        ("intersection_cache", "log_dir"),
    )
    if present and value is not None and str(value).strip():
        return Path(os.path.expanduser(str(value).strip())).expanduser()

    try:
        return Path(get_dir("debug_log_dir"))
    except Exception:
        return Path.cwd() / "logs"


__all__ = [
    "console_debug_enabled",
    "diffraction_debug_csv_logging_enabled",
    "env_flag_enabled",
    "geometry_fit_extra_sections_enabled",
    "geometry_fit_log_files_enabled",
    "intersection_cache_logging_enabled",
    "is_logging_disabled",
    "mosaic_fit_log_files_enabled",
    "projection_debug_logging_enabled",
    "resolve_intersection_cache_log_root",
    "runtime_update_trace_logging_enabled",
]
