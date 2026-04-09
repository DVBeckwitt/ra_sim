# Logging, Debug, and Cache Controls

RA-SIM now uses `config/debug.yaml` as the primary control surface for user-facing debug/logging output and optional cache retention.

The main kill switch is:

```yaml
debug:
  global:
    disable_all: true
```

When `debug.global.disable_all` is `true`, every debug/log output path documented here is disabled, regardless of the other entries in `debug.yaml`.

Legacy environment variables still work as compatibility overrides. They are no longer the primary interface.

## Primary config

The repo default is `config/debug.yaml`:

```yaml
debug:
  global:
    disable_all: false
  console:
    enabled: false
  runtime_update_trace:
    enabled: true
  geometry_fit:
    log_files: true
    extra_sections: true
  mosaic_fit:
    log_files: true
  projection_debug:
    enabled: true
  diffraction_debug_csv:
    enabled: true
  intersection_cache:
    enabled: true
    log_dir: null
  cache:
    default_retention: auto
    families:
      primary_contribution: auto
      source_snapshots: auto
      caking: auto
      peak_overlay: auto
      background_history: auto
      manual_pick: auto
      geometry_fit_dataset: auto
      qr_cylinder_overlay: auto
      diffraction_safe: auto
      diffraction_last_intersection: never
      fit_simulation: auto
      stacking_fault_base: auto
```

Meaning of each key:

1. `debug.global.disable_all`
   Global kill switch for all user-facing debug/log output covered by this document.

2. `debug.console.enabled`
   Enables console debug printing and Numba logging.

3. `debug.runtime_update_trace.enabled`
   Enables the GUI runtime trace log.

4. `debug.geometry_fit.log_files`
   Enables geometry-fit log file creation.

5. `debug.geometry_fit.extra_sections`
   Enables the more verbose geometry-fit diagnostic sections inside those logs.

6. `debug.mosaic_fit.log_files`
   Enables mosaic-shape fit log file creation in both GUI and CLI paths.

7. `debug.projection_debug.enabled`
   Enables projection-debug JSON logging.

8. `debug.diffraction_debug_csv.enabled`
   Enables the explicit diffraction debug CSV dump written by `dump_debug_log()`.

9. `debug.intersection_cache.enabled`
   Enables intersection-cache dump folders.

10. `debug.intersection_cache.log_dir`
    Optional root directory for intersection-cache dumps. `null` means use `debug_log_dir`.

11. `debug.cache.default_retention`
    Default policy for optional retained caches. Valid values are `never`, `auto`, and `always`.

12. `debug.cache.families.<name>`
    Per-cache-family override for optional retained caches. Valid values are `never`, `auto`, and `always`.

## Cache policy

The cache section controls only optional retained caches. It does not control active simulation state.

Three categories matter:

1. Mandatory current state
   Current simulation images, current peak tables, current active intersection caches, current integration payloads, the active background image, and the active beam/profile bundle. These are required for the current UI/runtime state and are not gated by the cache policy.

2. Optional retained caches
   Recomputable data kept only for reuse or debug convenience. These are controlled by `debug.cache`.

3. Per-call scratch buffers
   Temporary hot-loop work arrays. These are not retained caches and are not controlled here.

Current optional cache families:

1. `primary_contribution`
   Per-contribution primary hit-table cache used by incremental SF-prune reuse.

2. `source_snapshots`
   Stored source-row snapshots used by manual-geometry/source-row reuse flows.

3. `caking`
   Retained caked-analysis payloads reused across repeated analysis refreshes.

4. `peak_overlay`
   Reusable peak-overlay records and click-index payloads.

5. `background_history`
   Inactive background-image history. The currently selected background image remains mandatory.

6. `manual_pick`
   Geometry manual-pick candidate/match cache.

7. `geometry_fit_dataset`
   Cached geometry-fit dataset bundle for follow-on geometry-fit workflows.

8. `qr_cylinder_overlay`
   Cached analytic Qr-cylinder overlay paths.

9. `diffraction_safe`
   Retained diffraction safe-cache internals such as Q-vector reuse state.

10. `diffraction_last_intersection`
    Retained global last-intersection snapshot. Default is `never`.

11. `fit_simulation`
    Reusable fitting simulation/image caches.

12. `stacking_fault_base`
    Retained HT base-curve cache in stacking-fault generation.

Retention modes:

1. `never`
   Build on demand if needed for the current action, then discard.

2. `auto`
   Retain only when the active feature benefits from reuse. This is the default balanced mode.

3. `always`
   Retain whenever built.

Important detail:

- `debug.global.disable_all` disables logging/debug output only.
- `debug.global.disable_all` does not disable optional caches.
- Tiny infrastructure/compile caches such as config bundle loading, CIF parsing, and compiled expression helpers stay always-on and are not controlled by `debug.cache`.

## Resolution order

Debug control resolution follows this order:

1. Global disable is active if either `debug.global.disable_all` is `true` or `RA_SIM_DISABLE_ALL_LOGGING` / `RA_SIM_DISABLE_LOGGING` is truthy.
2. If global disable is active, all subsystem debug/log outputs are disabled.
3. Otherwise, existing environment variables override the matching `debug.yaml` entry.
4. Otherwise, `debug.yaml` provides the value.
5. For `debug.geometry_fit.extra_sections` only, if that key is absent, the code falls back to the legacy instrument config keys `instrument.fit.geometry.debug_logging` and then `instrument.fit.geometry.debug_mode`.

Important detail:

- `RA_SIM_DEBUG=1` does not bypass the global kill switch.
- Some debug keys are config-only because there was no legacy env var for them.
- Cache retention has no environment-variable overrides in v1.

## Compatibility environment variables

These env vars are still honored:

1. `RA_SIM_DISABLE_ALL_LOGGING=0/1`
   Compatibility override for the global kill switch.

2. `RA_SIM_DISABLE_LOGGING=0/1`
   Legacy alias for the same global kill switch.

3. `RA_SIM_DEBUG=0/1`
   Compatibility override for `debug.console.enabled`.

4. `RA_SIM_DISABLE_PROJECTION_DEBUG=0/1`
   Negative compatibility override for `debug.projection_debug.enabled`.
   `1` disables projection-debug logging.

5. `RA_SIM_LOG_INTERSECTION_CACHE=0/1`
   Compatibility override for `debug.intersection_cache.enabled`.

6. `RA_SIM_INTERSECTION_CACHE_LOG_DIR=/path/to/dir`
   Compatibility override for `debug.intersection_cache.log_dir`.

Legacy geometry-fit compatibility:

1. `instrument.fit.geometry.debug_logging`
   Fallback for `debug.geometry_fit.extra_sections` when the new key is absent.

2. `instrument.fit.geometry.debug_mode`
   Older fallback alias used only if `debug_logging` is absent.

## What is covered by the global kill switch

`debug.global.disable_all: true` disables all of these:

1. Console debug output and Numba logging.
2. GUI runtime update trace logging.
3. Geometry-fit log file creation.
4. Geometry-fit verbose diagnostic sections.
5. Mosaic-shape fit log file creation in GUI and CLI flows.
6. Projection-debug JSON output.
7. Diffraction debug CSV output from `dump_debug_log()`.
8. Intersection-cache dump folders.

This includes the older direct geometry-fit and mosaic-fit writers in the GUI runtime and CLI paths. They are now routed through the centralized resolver.

## Output files and directories

Default output locations still come from `config/dir_paths.yaml`.

Relevant directory keys:

1. `downloads`
2. `debug_log_dir`
3. `overlay_dir`
4. `temp_root`
5. `file_dialog_dir`

Default directory values:

1. `downloads`: `~/Downloads`
2. `debug_log_dir`: `~/.cache/ra_sim/logs`
3. `overlay_dir`: `~/.cache/ra_sim/overlays`
4. `temp_root`: `~/.cache/ra_sim`
5. `file_dialog_dir`: `~/.local/share/ra_sim`

Current debug/log outputs:

1. GUI runtime update trace
   File: `runtime_update_trace_<YYYYMMDD>.log`
   Location: `downloads`
   Controlled by: `debug.runtime_update_trace.enabled`

2. Geometry-fit logs
   File: `geometry_fit_log_<stamp>.txt`
   Location: `debug_log_dir`
   Controlled by: `debug.geometry_fit.log_files`
   Verbosity controlled by: `debug.geometry_fit.extra_sections`

3. Mosaic-shape fit logs
   File: `mosaic_shape_fit_log_<stamp>.txt`
   Location: `debug_log_dir`
   Controlled by: `debug.mosaic_fit.log_files`

4. Projection-debug JSON
   File: `projection_debug_<stamp>.json`
   Location: `debug_log_dir`
   Controlled by: `debug.projection_debug.enabled`

5. Diffraction debug CSV
   File: `mosaic_full_debug_log.csv`
   Location: `debug_log_dir`
   Controlled by: `debug.diffraction_debug_csv.enabled`

6. Intersection-cache dumps
   Directory pattern: `intersection_cache_<stamp>_<pid>`
   Root location: `debug.intersection_cache.log_dir` when set, otherwise `debug_log_dir`
   Controlled by: `debug.intersection_cache.enabled`

`get_dir(...)` still creates missing configured directories automatically. Changing a directory setting redirects output, but it does not enable or disable output by itself.

## Practical examples

Disable all debug/log output in config:

```yaml
debug:
  global:
    disable_all: true
```

Keep everything on except console spam:

```yaml
debug:
  global:
    disable_all: false
  console:
    enabled: false
```

Disable only projection-debug JSON and intersection-cache dumps:

```yaml
debug:
  projection_debug:
    enabled: false
  intersection_cache:
    enabled: false
```

Disable geometry-fit extra sections but keep the log files:

```yaml
debug:
  geometry_fit:
    log_files: true
    extra_sections: false
```

Redirect intersection-cache dumps:

```yaml
debug:
  intersection_cache:
    enabled: true
    log_dir: /tmp/ra-sim-cache-dumps
```

Temporary compatibility override from PowerShell:

```powershell
$env:RA_SIM_DISABLE_ALL_LOGGING = "1"
```

Temporary console-debug override from PowerShell:

```powershell
$env:RA_SIM_DEBUG = "1"
```

## Bottom line

Use `config/debug.yaml` for normal project configuration.

If you need a master OFF switch, set:

```yaml
debug:
  global:
    disable_all: true
```

If you need a temporary shell-level override, use:

```powershell
$env:RA_SIM_DISABLE_ALL_LOGGING = "1"
```

The config kill switch and the env kill switches both disable every debug/log output covered by this README.
