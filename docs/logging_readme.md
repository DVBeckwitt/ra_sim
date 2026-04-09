# Logging README

There is not currently a single master switch that disables all logging, debug file output, and cache-related disk artifacts.

This note records the current behavior in the codebase so it is clear which controls actually work, which outputs still happen even with those controls disabled, and which cache mechanisms are only in memory.

## Easiest partial disable

For a PowerShell session, the quickest available disable is:

```powershell
$env:RA_SIM_DEBUG = "0"
$env:RA_SIM_LOG_INTERSECTION_CACHE = "0"
```

And in `config/instrument.yaml`:

```yaml
instrument:
  fit:
    geometry:
      debug_logging: false
```

That is only a partial disable. Normal geometry-fit, mosaic-fit, and projection-debug files can still be written.

## Current switches that matter

1. `RA_SIM_DEBUG=0/1`
   Controls console debug output and Numba logging.

   Current implementation:
   - `ra_sim/debug_utils.py`
   - `ra_sim/gui/_runtime/runtime_impl.py`

   Behavior:
   - `0`, `false`, or `no` disables the debug prints.
   - `1`, `true`, or `yes` enables them.

2. `RA_SIM_LOG_INTERSECTION_CACHE=0/1`
   Controls the `intersection_cache_*` dump folders written by the diffraction intersection-cache logger.

   Current implementation:
   - `ra_sim/simulation/diffraction.py`

   Behavior:
   - This is effectively enabled by default because the code falls back to `"1"` when the variable is unset.
   - Set it to `0` to stop writing those cache-dump folders.

3. `RA_SIM_INTERSECTION_CACHE_LOG_DIR=/path/to/dir`
   Redirects the intersection-cache dump root to a different directory.

   Current implementation:
   - `ra_sim/simulation/diffraction.py`

   Behavior:
   - If unset, intersection-cache dumps go under `debug_log_dir/intersection_cache`.
   - This changes location only. It does not disable writing.

4. `instrument.fit.geometry.debug_logging: false/true`
   Controls whether extra geometry-fit diagnostic sections are added to the geometry-fit log.

   Current implementation:
   - configured in `config/instrument.yaml`
   - checked in `ra_sim/gui/geometry_fit.py`

   Important limitation:
   - Setting this to `false` does not prevent the geometry-fit log file itself from being created.
   - It only suppresses the extra debug sections inside that file.

## Disk outputs that still happen

These outputs currently do not have a single top-level off switch.

1. Geometry-fit text logs
   Current implementation:
   - `ra_sim/gui/_runtime/runtime_impl.py`
   - `ra_sim/gui/geometry_fit.py`
   - `ra_sim/headless_geometry_fit.py`

   Behavior:
   - Geometry-fit runs create `geometry_fit_log_<stamp>.txt` in `debug_log_dir`.
   - Preflight failures can also create a geometry-fit log file before the solver starts.
   - `debug_logging: false` reduces verbosity but does not stop file creation.

2. Mosaic-fit text logs
   Current implementation:
   - `ra_sim/gui/_runtime/runtime_impl.py`
   - `ra_sim/cli.py`

   Behavior:
   - Mosaic-shape fitting writes `mosaic_shape_fit_log_<stamp>.txt` in `debug_log_dir`.

3. Projection-debug JSON
   Current implementation:
   - `ra_sim/simulation/engine.py`
   - `ra_sim/simulation/projection_debug.py`

   Behavior:
   - The simulation engine allocates projection-debug buffers on the normal path and finalizes a `projection_debug_<stamp>.json` file under `debug_log_dir`.
   - There is no separate environment variable or config flag in the current code that disables this at the top level.

4. Debug CSV from the explicit debug simulation path
   Current implementation:
   - `ra_sim/simulation/diffraction_debug.py`
   - `ra_sim/gui/_runtime/runtime_impl.py`

   Behavior:
   - `dump_debug_log()` writes `mosaic_full_debug_log.csv` under `debug_log_dir`.
   - This is tied to the explicit debug simulation workflow rather than the normal run path.

5. Intersection-cache dump folders
   Current implementation:
   - `ra_sim/simulation/diffraction.py`

   Behavior:
   - When enabled, the code writes `intersection_cache/intersection_cache_<stamp>_<pid>/...` under the resolved log root.
   - This is the one cache-dump output that currently has a dedicated disable flag.

## Directory settings that affect where things go

The main directory locations come from `config/dir_paths.yaml`, with defaults provided in `ra_sim/config/loader.py`.

Relevant keys:
- `debug_log_dir`
- `overlay_dir`
- `temp_root`
- `downloads`
- `file_dialog_dir`

Important behavior from `ra_sim/config/loader.py`:
- `get_dir(...)` creates the directory automatically if it does not already exist.
- `get_temp_dir(...)` creates a dedicated temporary subdirectory under `temp_root` and caches that temp path per active config directory.

Practical meaning:
- Changing these settings can redirect output.
- They do not function as enable or disable switches.

## Cache behavior

There is not currently a global cache-off switch.

Current state:
1. Most cache behavior in the normal runtime is in memory.
2. The standard simulation-engine wrappers already force `enable_safe_cache=False` for the safe peak wrappers in:
   - `ra_sim/simulation/engine.py`
3. GUI caches such as background caches, preview caches, image buffers, and similar runtime caches are internal state caches rather than documented disk caches.
4. `temp_root` is a location for temporary working directories, not a user-facing cache toggle.

## Potentially confusing settings

Some config entries look like output controls but are not the main switches for the current runtime behavior.

1. `config/file_paths.yaml -> debug_log_csv`
   The current code paths found for debug logging write through `debug_log_dir` directly. The `debug_log_csv` path does not appear to be the active control for the current debug CSV writer.

2. `config/file_paths.yaml -> overlay_output`
   This is not the primary logging control surface for the runtime paths inspected here.

3. `overlay_dir`
   This is used by the debug optimization script in `scripts/debug/optimization.py`, not as a general off switch for runtime logging.

## Bottom line

The current easiest partial disable is:

```powershell
$env:RA_SIM_DEBUG = "0"
$env:RA_SIM_LOG_INTERSECTION_CACHE = "0"
```

plus:

```yaml
instrument:
  fit:
    geometry:
      debug_logging: false
```

That still leaves normal geometry-fit logs, mosaic-fit logs, and projection-debug JSON active. If a true all-logging or all-cache-output switch is needed, it will require a code change.
