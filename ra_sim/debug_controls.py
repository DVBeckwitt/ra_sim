"""Central debug and logging control resolution for RA-SIM."""

from __future__ import annotations

import atexit
import json
import os
import threading
import zipfile
from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ra_sim.config import get_config_bundle, get_config_dir, get_dir


_DEBUG_DEFAULTS: dict[str, Any] = {
    "global": {"disable_all": False},
    "console": {"enabled": False},
    "runtime_update_trace": {"enabled": True},
    "geometry_fit": {"log_files": True, "extra_sections": True},
    "mosaic_fit": {"log_files": True},
    "projection_debug": {"enabled": True},
    "diffraction_debug_csv": {"enabled": True},
    "intersection_cache": {"enabled": True, "log_dir": None},
    "cache": {
        "default_retention": "auto",
        "families": {
            "primary_contribution": "auto",
            "source_snapshots": "auto",
            "caking": "auto",
            "peak_overlay": "auto",
            "background_history": "auto",
            "manual_pick": "auto",
            "geometry_fit_dataset": "auto",
            "qr_cylinder_overlay": "auto",
            "diffraction_safe": "auto",
            "diffraction_last_intersection": "never",
            "fit_simulation": "auto",
            "stacking_fault_base": "auto",
        },
    },
}

_TRUTHY_VALUES = {"1", "true", "yes", "on"}
_FALSY_VALUES = {"", "0", "false", "no", "off"}
_VALID_CACHE_RETENTIONS = {"never", "auto", "always"}
_STARTUP_DEBUG_LOG_PATH: Path | None = None
_STARTUP_DEBUG_OVERRIDE: DebugStartupOverride = "inherit"
_STARTUP_DEBUG_OVERRIDE_LOCK = threading.Lock()
_RUN_BUNDLE_EXCLUDED_INPUT_SUFFIXES = {".osc", ".cif"}
_RUN_BUNDLE_LOCK = threading.Lock()


@dataclass
class _RunBundleState:
    stamp: str
    started_at: datetime
    entrypoints: list[str] = field(default_factory=list)
    roots: dict[str, Path] = field(default_factory=dict)
    tracked_inputs: set[Path] = field(default_factory=set)
    tracked_outputs: set[Path] = field(default_factory=set)
    config_dir: Path | None = None
    final_zip_path: Path | None = None
    finalized: bool = False


_RUN_BUNDLE_STATE: _RunBundleState | None = None

CacheRetention = Literal["never", "auto", "always"]
DebugStartupOverride = Literal["inherit", "enable_all", "disable_all"]
CacheFamily = Literal[
    "primary_contribution",
    "source_snapshots",
    "caking",
    "peak_overlay",
    "background_history",
    "manual_pick",
    "geometry_fit_dataset",
    "qr_cylinder_overlay",
    "diffraction_safe",
    "diffraction_last_intersection",
    "fit_simulation",
    "stacking_fault_base",
]


def _normalize_startup_debug_override(mode: str | None) -> DebugStartupOverride:
    normalized = "inherit" if mode is None else str(mode).strip().lower()
    if normalized in {"", "default"}:
        normalized = "inherit"
    if normalized not in {"inherit", "enable_all", "disable_all"}:
        raise ValueError(
            "startup debug override must be one of: inherit, enable_all, disable_all"
        )
    return normalized  # type: ignore[return-value]


def _startup_debug_override_mode() -> DebugStartupOverride:
    return _STARTUP_DEBUG_OVERRIDE


@contextmanager
def temporary_startup_debug_override(
    mode: DebugStartupOverride | str | None,
):
    """Apply one startup debug override for the current process temporarily."""

    normalized = _normalize_startup_debug_override(mode)
    if normalized == "inherit":
        yield
        return

    global _STARTUP_DEBUG_OVERRIDE
    with _STARTUP_DEBUG_OVERRIDE_LOCK:
        previous = _STARTUP_DEBUG_OVERRIDE
        _STARTUP_DEBUG_OVERRIDE = normalized
    try:
        yield
    finally:
        with _STARTUP_DEBUG_OVERRIDE_LOCK:
            _STARTUP_DEBUG_OVERRIDE = previous


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


def _resolve_config_cache_retention(
    path: tuple[str, ...],
    *,
    default: CacheRetention,
    debug_config: Mapping[str, Any] | None = None,
) -> CacheRetention:
    present, value = _lookup_mapping_value(_debug_root(debug_config), path)
    if not present or value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in _VALID_CACHE_RETENTIONS:
        return normalized  # type: ignore[return-value]
    return default


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

    startup_override = _startup_debug_override_mode()
    if startup_override == "disable_all":
        return True
    if startup_override == "enable_all":
        return False

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
    if _startup_debug_override_mode() == "enable_all":
        return True
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
    if _startup_debug_override_mode() == "enable_all":
        return True
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
    if _startup_debug_override_mode() == "enable_all":
        return True
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
    if _startup_debug_override_mode() == "enable_all":
        return True

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
    if _startup_debug_override_mode() == "enable_all":
        return True
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
    if _startup_debug_override_mode() == "enable_all":
        return True
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
    if _startup_debug_override_mode() == "enable_all":
        return True
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
    if _startup_debug_override_mode() == "enable_all":
        return True
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


def resolve_startup_debug_log_path(
    *,
    stamp: str | None = None,
    log_dir: Path | str | None = None,
    downloads_dir: Path | str | None = None,
    env: Mapping[str, object] | None = None,
    debug_config: Mapping[str, Any] | None = None,
) -> Path:
    """Return the shared startup-scoped debug log path for this process."""

    global _STARTUP_DEBUG_LOG_PATH

    if _STARTUP_DEBUG_LOG_PATH is not None:
        return _STARTUP_DEBUG_LOG_PATH

    resolved_root: Path | None = None
    if log_dir is not None and str(log_dir).strip():
        resolved_root = Path(log_dir).expanduser()
    elif downloads_dir is not None and str(downloads_dir).strip():
        resolved_root = Path(downloads_dir).expanduser()
    else:
        resolved_root = resolve_intersection_cache_log_root(
            env,
            debug_config=debug_config,
        )

    session_stamp = str(stamp).strip() if stamp is not None else ""
    if not session_stamp:
        session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    _STARTUP_DEBUG_LOG_PATH = resolved_root / f"geometry_fit_log_{session_stamp}.txt"
    return _STARTUP_DEBUG_LOG_PATH


def reset_startup_debug_log_path() -> None:
    """Clear the cached startup-scoped debug log path.

    This is primarily useful in tests that simulate multiple app startups in one
    Python process.
    """

    global _STARTUP_DEBUG_LOG_PATH
    _STARTUP_DEBUG_LOG_PATH = None


def current_startup_debug_log_path() -> Path | None:
    """Return the cached startup-scoped debug log path if one already exists."""

    return _STARTUP_DEBUG_LOG_PATH


def _coerce_run_bundle_path(value: object) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        path = value.expanduser()
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        path = Path(text).expanduser()
    else:
        return None
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    return path


def _iter_run_bundle_paths(value: object) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        paths: list[Path] = []
        for item in value.values():
            paths.extend(_iter_run_bundle_paths(item))
        return paths
    if isinstance(value, (list, tuple, set, frozenset)):
        paths: list[Path] = []
        for item in value:
            paths.extend(_iter_run_bundle_paths(item))
        return paths
    path = _coerce_run_bundle_path(value)
    return [path] if path is not None else []


def _run_bundle_roots() -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for key in ("debug_log_dir",):
        try:
            roots[key] = Path(get_dir(key)).resolve()
        except Exception:
            continue
    try:
        roots["intersection_cache_log_root"] = Path(
            resolve_intersection_cache_log_root()
        ).expanduser().resolve()
    except Exception:
        pass
    return roots


def _register_run_bundle_config_files(state: _RunBundleState) -> None:
    try:
        config_dir = get_config_dir().resolve()
    except Exception:
        return
    state.config_dir = config_dir
    for pattern in ("*.yaml", "*.yml", "*.json"):
        for path in sorted(config_dir.glob(pattern)):
            if path.is_file():
                state.tracked_inputs.add(path.resolve())


def _runtime_trace_output_path(day: datetime) -> Path | None:
    try:
        downloads_dir = Path(get_dir("downloads")).resolve()
    except Exception:
        return None
    return downloads_dir / f"runtime_update_trace_{day.strftime('%Y%m%d')}.log"


def start_run_bundle(*, entrypoint: str | None = None) -> None:
    """Start one process-scoped artifact bundle session if not already active."""

    global _RUN_BUNDLE_STATE

    with _RUN_BUNDLE_LOCK:
        state = _RUN_BUNDLE_STATE
        if state is None:
            now = datetime.now()
            state = _RunBundleState(
                stamp=now.strftime("%Y%m%d_%H%M%S_%f"),
                started_at=now,
                roots=_run_bundle_roots(),
            )
            _register_run_bundle_config_files(state)
            _RUN_BUNDLE_STATE = state
        if entrypoint:
            label = str(entrypoint).strip()
            if label and label not in state.entrypoints:
                state.entrypoints.append(label)
        trace_path = _runtime_trace_output_path(state.started_at)
        if trace_path is not None:
            state.tracked_outputs.add(trace_path)


def register_run_input_paths(value: object) -> None:
    """Track one or more input paths for the active run bundle."""

    global _RUN_BUNDLE_STATE

    with _RUN_BUNDLE_LOCK:
        if _RUN_BUNDLE_STATE is None:
            return
        for path in _iter_run_bundle_paths(value):
            _RUN_BUNDLE_STATE.tracked_inputs.add(path)


def register_run_output_path(value: object) -> None:
    """Track one output path for the active run bundle."""

    global _RUN_BUNDLE_STATE

    with _RUN_BUNDLE_LOCK:
        if _RUN_BUNDLE_STATE is None:
            return
        path = _coerce_run_bundle_path(value)
        if path is not None:
            _RUN_BUNDLE_STATE.tracked_outputs.add(path)


def _run_bundle_path_changed_since_start(
    path: Path,
    *,
    started_at: datetime,
) -> bool:
    try:
        return path.stat().st_mtime >= started_at.timestamp()
    except OSError:
        return False


def _run_bundle_safe_zip_path(path: Path) -> str:
    parts = list(path.parts)
    if path.drive:
        parts = [path.drive.rstrip(":")] + parts[1:]
    elif path.is_absolute():
        parts = ["root"] + parts[1:]
    cleaned = [str(part).replace("\\", "/") for part in parts if str(part) not in {"", "\\", "/"}]
    return "/".join(cleaned)


def _run_bundle_collect_directory_files(
    root: Path,
    *,
    archive_prefix: str,
    started_at: datetime,
    collected: dict[Path, str],
) -> None:
    if not root.exists() or not root.is_dir():
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if not _run_bundle_path_changed_since_start(path, started_at=started_at):
            continue
        resolved = path.resolve()
        if resolved in collected:
            continue
        try:
            rel = resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = resolved.name
        collected[resolved] = f"{archive_prefix}/{rel}"


def _run_bundle_collect_registered_path(
    path: Path,
    *,
    archive_prefix: str,
    started_at: datetime,
    collected: dict[Path, str],
    require_change: bool,
) -> None:
    if not path.exists():
        return
    if path.is_dir():
        _run_bundle_collect_directory_files(
            path,
            archive_prefix=archive_prefix,
            started_at=started_at,
            collected=collected,
        )
        return
    if require_change and not _run_bundle_path_changed_since_start(
        path,
        started_at=started_at,
    ):
        return
    resolved = path.resolve()
    if resolved not in collected:
        collected[resolved] = f"{archive_prefix}/{_run_bundle_safe_zip_path(resolved)}"


def _run_bundle_manifest(
    state: _RunBundleState,
    *,
    zip_path: Path,
    bundled_files: dict[Path, str],
    omitted_inputs: list[str],
) -> dict[str, object]:
    return {
        "started_at": state.started_at.isoformat(),
        "finalized_at": datetime.now().isoformat(),
        "entrypoints": list(state.entrypoints),
        "config_dir": str(state.config_dir) if state.config_dir is not None else None,
        "zip_path": str(zip_path),
        "roots": {name: str(path) for name, path in sorted(state.roots.items())},
        "tracked_inputs": [str(path) for path in sorted(state.tracked_inputs, key=str)],
        "tracked_outputs": [str(path) for path in sorted(state.tracked_outputs, key=str)],
        "excluded_input_suffixes": sorted(_RUN_BUNDLE_EXCLUDED_INPUT_SUFFIXES),
        "omitted_inputs": sorted(omitted_inputs),
        "bundled_files": [
            {"path": str(path), "archive_path": arcname}
            for path, arcname in sorted(bundled_files.items(), key=lambda item: item[1])
        ],
    }


def finalize_run_bundle() -> Path | None:
    """Write one per-run zip bundle into ``debug_log_dir`` and return its path."""

    global _RUN_BUNDLE_STATE

    with _RUN_BUNDLE_LOCK:
        state = _RUN_BUNDLE_STATE
        if state is None:
            return None
        if state.finalized:
            return state.final_zip_path

        debug_log_dir = state.roots.get("debug_log_dir")
        if debug_log_dir is None:
            try:
                debug_log_dir = Path(get_dir("debug_log_dir")).resolve()
            except Exception:
                debug_log_dir = (Path.cwd() / "logs").resolve()
                debug_log_dir.mkdir(parents=True, exist_ok=True)

        bundled_files: dict[Path, str] = {}
        roots = sorted(
            state.roots.items(),
            key=lambda item: (-len(str(item[1])), item[0]),
        )
        for root_name, root in roots:
            _run_bundle_collect_directory_files(
                root,
                archive_prefix=f"roots/{root_name}",
                started_at=state.started_at,
                collected=bundled_files,
            )

        omitted_inputs: list[str] = []
        for path in sorted(state.tracked_inputs, key=str):
            if path.suffix.lower() in _RUN_BUNDLE_EXCLUDED_INPUT_SUFFIXES:
                omitted_inputs.append(str(path))
                continue
            _run_bundle_collect_registered_path(
                path,
                archive_prefix="inputs",
                started_at=state.started_at,
                collected=bundled_files,
                require_change=False,
            )

        trace_path = _runtime_trace_output_path(datetime.now())
        if trace_path is not None:
            state.tracked_outputs.add(trace_path)
        for path in sorted(state.tracked_outputs, key=str):
            _run_bundle_collect_registered_path(
                path,
                archive_prefix="outputs",
                started_at=state.started_at,
                collected=bundled_files,
                require_change=True,
            )

        zip_path = debug_log_dir / f"run_bundle_{state.stamp}.zip"
        manifest = _run_bundle_manifest(
            state,
            zip_path=zip_path,
            bundled_files=bundled_files,
            omitted_inputs=omitted_inputs,
        )

        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, indent=2, sort_keys=True),
            )
            for path, arcname in sorted(bundled_files.items(), key=lambda item: item[1]):
                if path.resolve() == zip_path.resolve():
                    continue
                if path.exists() and path.is_file():
                    archive.write(path, arcname)

        state.final_zip_path = zip_path
        state.finalized = True
        return zip_path


def reset_run_bundle_state() -> None:
    """Clear the process-scoped run bundle session."""

    global _RUN_BUNDLE_STATE
    with _RUN_BUNDLE_LOCK:
        _RUN_BUNDLE_STATE = None


def cache_retention_mode(
    family: CacheFamily | str,
    *,
    debug_config: Mapping[str, Any] | None = None,
) -> CacheRetention:
    """Return the configured retention mode for one optional cache family."""

    family_name = str(family).strip()
    families = _DEBUG_DEFAULTS["cache"]["families"]
    if family_name not in families:
        raise KeyError(f"Unknown cache family: {family_name}")
    default_mode = _resolve_config_cache_retention(
        ("cache", "default_retention"),
        default=_DEBUG_DEFAULTS["cache"]["default_retention"],
        debug_config=debug_config,
    )
    return _resolve_config_cache_retention(
        ("cache", "families", family_name),
        default=default_mode,
        debug_config=debug_config,
    )


def retain_optional_cache(
    family: CacheFamily | str,
    *,
    feature_needed: bool,
    debug_config: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether one optional cache should retain data after it is built."""

    mode = cache_retention_mode(family, debug_config=debug_config)
    if mode == "always":
        return True
    if mode == "never":
        return False
    return bool(feature_needed)


__all__ = [
    "CacheFamily",
    "CacheRetention",
    "cache_retention_mode",
    "console_debug_enabled",
    "current_startup_debug_log_path",
    "diffraction_debug_csv_logging_enabled",
    "env_flag_enabled",
    "finalize_run_bundle",
    "geometry_fit_extra_sections_enabled",
    "geometry_fit_log_files_enabled",
    "intersection_cache_logging_enabled",
    "is_logging_disabled",
    "mosaic_fit_log_files_enabled",
    "projection_debug_logging_enabled",
    "register_run_input_paths",
    "register_run_output_path",
    "reset_run_bundle_state",
    "reset_startup_debug_log_path",
    "retain_optional_cache",
    "resolve_intersection_cache_log_root",
    "resolve_startup_debug_log_path",
    "start_run_bundle",
    "temporary_startup_debug_override",
    "runtime_update_trace_logging_enabled",
]


atexit.register(finalize_run_bundle)
