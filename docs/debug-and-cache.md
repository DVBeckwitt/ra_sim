# Debug and cache guide

This page is the short operational guide for logging, debug output, and cache
retention. Use it when you need to understand where RA-SIM writes diagnostics
or why a debug path is noisy or silent.

See also:

- [docs index](index.md)
- [Troubleshooting guide](troubleshooting.md)
- [Canonical logging section](simulation_and_fitting.md#logging-debug-and-cache-controls)

## Main Control Surface

RA-SIM uses `config/debug.yaml` as the primary debug and logging configuration.

Important points:

- `debug.global.disable_all: true` is the master kill switch for user-facing debug/log output
- the simulation GUI launcher can temporarily force diagnostics off or on for one run without rewriting `config/debug.yaml`
- `debug.console.enabled` controls console debug printing
- geometry-fit and mosaic-fit log file creation have dedicated toggles
- optional retained caches are controlled separately from logging output

The simulation GUI Advanced/Debug panel also has runtime-only physics toggles
for diagnostic comparisons:

- `Disable HT: Integer Bragg` switches the primary structure-factor source from
  Hendricks-Teller rods to integer CIF Bragg peaks, placing all primary
  structure-factor weight at integer HKL rows until `Enable HT` is pressed.
- `Disable Refraction` forces the active simulation and fit payloads to use
  vacuum `n2=1+0j` and invalidates optics-dependent caches until `Enable
  Refraction` is pressed.

These buttons are not saved-state or config settings. They are intended for
one-session comparisons while debugging simulation physics.

## Output Locations

The default output directories come from `config/dir_paths.yaml`.

Common outputs:

- runtime trace logs
- geometry-fit logs
- mosaic-shape fit logs
- projection-debug JSON
- diffraction debug CSV
- intersection-cache dump folders

## Cache Policy

The cache policy only affects optional retained caches. It does not turn off the
active runtime state needed for the current simulation or GUI session.

Retention modes:

- `never`: build on demand, then discard
- `auto`: retain when the active feature benefits from reuse
- `always`: retain whenever built

## Geometry-Fit Visual Residual Diagnostics

GUI geometry fits now emit three visual-consistency diagnostics after the fitted
simulation redraw settles:

- overlay frame diagnostics report
  `fit_sim_render_caked_delta_med` and `fit_sim_render_caked_delta_max`, which
  compare each green `fit sim` marker against the caked point used by the
  rendered fitted simulation image
- draw-time visual probes report `visual_probe_artist_to_image_peak_med` and
  `visual_probe_artist_to_image_peak_max`, which compare the actual drawn green
  marker artist coordinates against the strongest visible simulation-image
  pixel in a small local search window
- geometry-fit logs report `holistic detector` and `holistic caked` residual
  lines, comparing the full background image against the pre-fit and post-fit
  simulation images with one least-squares intensity scale

These checks are diagnostic only. They do not change the solver objective, fit
parameters, saved-state schema, CLI flags, or cache retention policy. A positive
holistic residual delta or `suspicious=True` means the full-image agreement got
worse even if the point objective improved. A
`visual_probe_warning=fit_sim_marker_image_mismatch` line means the green marker
artist and the rendered simulation image peak disagree in the visible GUI frame,
so the fit log can catch the class of screenshot-only mismatch where cached
numbers look plausible but the overlay is not moving with the image.

The GUI acceptance gate remains detector-pixel strict even when caked angular
diagnostics are present. The caked angular acceptance shortcut is reserved for
true headless geometry-fit runs that set the private headless runtime flag.

### Manual QR/Qz Caked Acceptance Status

Status: fixed in the current working tree.

Locked manual Qr/Qz caked fits distinguish dynamic prediction candidates from
dynamic prediction proof. A callable trial source-row builder only means the
fit can probe for dynamic predictions; it is not enough to label the route as
`dynamic_angular_point_match` before a row proves:

- `prediction_role` is `objective_trial` or `final_dynamic_prediction`
- `fit_prediction_is_dynamic=yes`
- `fit_prediction_source` is not a locked saved manual QR handoff

If pair-audit caked residuals and final caked acceptance rows disagree for the
same pairs, the failure class is `caked_acceptance_metric_inconsistent` and the
recommended next fix is `inspect_acceptance_metric_sources`, not manual
repicking. When active `gamma` or `Gamma` fitting has only locked/non-dynamic
predictions available, the route fails closed with
`manual_qr_dynamic_prediction_unavailable` before optimizer solve.

The 2026-05-26 regression fixture pins the manual-run signature where pair
audit residuals are about `1.70 deg` and `0.75 deg`, but stale cached final
deltas report `104.39 deg` and `75.16 deg` (`90.96 deg` RMS). When
`observed_caked_deg` and `predicted_caked_deg` are finite, final acceptance
recomputes residuals from those endpoints and treats `delta_two_theta_deg` plus
`wrapped_delta_phi_deg` as cache fields that must agree. A mismatch is retained
in the trace as `supplied_delta_cache_mismatch`; it is not used as the final
caked acceptance metric.

For Bi2Se3 headless `gamma,Gamma` recovery runs, `fit-geometry` now writes visual
approval artifacts into the same output directory as the fitted state and
progress sidecar. A run such as:

```bash
python -m ra_sim fit-geometry artifacts/geometry_fit_gui_states/bi2se3.json \
  --active-vars gamma,Gamma \
  --seed-policy ladder-multistart \
  --out-state artifacts/geometry_fit_recovery/bi2se3_headless_gamma_gamma/bi2se3_gamma_gamma_fit.json
```

must write:

- `01_single_step_qr_coordinate_audit.json`
- `01_single_step_qr_coordinate_audit.csv`
- `01_single_step_qr_coordinate_audit.png`
- `02_full_fit_initial_vs_final_qr_overlay.json`
- `02_full_fit_initial_vs_final_qr_overlay.png`
- `03_worst_residual_rows.json` and `03_worst_residual_rows.png` when the fit rejects

The progress JSON records the generated paths under
`geometry_fit_recovery_artifacts` and mirrors the common paths at top level for
operator inspection. It also records `input_state_path` and
`input_state_sha256` so the artifact folder proves which saved GUI state was
used. Required PNGs are gate artifacts: the run fails if the single-step PNG or
full-fit overlay PNG is missing, and a rejected fit also fails if the worst-row
PNG is missing. This keeps a caked angular rejection such as
`branch_source_pairing_mismatch` paired with visual evidence in the same folder
as `bi2se3_gamma_gamma_fit.progress.json`.

The same headless `fit-geometry` command supports an opt-in dry-run parameter
sweep for controlled Bi2Se3 recovery investigations:

```bash
python -m ra_sim fit-geometry artifacts/geometry_fit_gui_states/bi2se3.json \
  --seed-policy ladder-multistart \
  --parameter-combo-sweep \
  --exclude-pair-id bg1:pair15 \
  --exclude-pair-id bg0:pair20 \
  --exclude-pair-id bg2:pair17 \
  --out-state artifacts/geometry_fit_recovery/bi2se3_headless_gamma_gamma/bi2se3_gamma_gamma_fit.json
```

`--exclude-pair-id` removes exact manual-pair IDs before objective assembly; it
does not match by HKL and does not mutate the saved GUI state. Sweep reports
record `excluded_pair_ids`, `excluded_pair_count`, `excluded_rows`,
`original_qr_fit_expected_count`, `qr_fit_expected_count`,
`qr_fit_resolved_count`, `qr_fit_missing_pairs`, and
`excluded_rows_do_not_count_as_missing`. For the Bi2Se3 recovery case above, the
contract is `original_qr_fit_expected_count=82`,
`qr_fit_expected_count=79`, `qr_fit_resolved_count=79`, and
`excluded_pair_count=3`.

`--parameter-combo-sweep` writes one subdirectory per combo under
`sweep/`, with each combo containing the single-step audit, full-fit overlay,
rejected-fit worst-row image/JSON when applicable, and `combo_result.json`.
The top-level `sweep_report.json`, `sweep_report.md`, and
`sweep_summary.png` list every combo, pass/fail state, RMS/max residuals,
dry-run geometry-update status, and artifact links. Accepted dry-run combos set
`would_update_geometry=true` and `geometry_updated=false`; geometry changes are
reserved for the explicit apply step, which verifies the saved-state hash,
approved exclusion list, accepted combo result, caked residual thresholds,
complete QR contract with strict integer counts, source/objective mismatch
counters, parameter sensitivity, and the accepted overlay PNG artifact before
applying variables. The apply is atomic for GUI geometry variables: missing or
non-settable targets are rejected before mutation, and setter failures, overlay
destination failures, overlay temp-file failures, and rebuild exceptions restore
the pre-apply values and leave no new applied overlay behind.
CLI-requested sweeps isolate each combo in a child Python process so a native
Windows `python313.dll` access violation or other non-Python process exit is
reported as a rejected `combo_result.json` with the child return code and
required placeholder artifacts instead of terminating the parent sweep. The
temporary child request/result/log files are deleted after the parent records
bounded stdout/stderr tails and writes durable `subprocess_*_tail.txt` files
for failed child exits, so the durable artifact folder keeps only documented
sweep outputs.

To apply the accepted Bi2Se3 `gamma,Gamma` sweep result, run an explicit apply
command with the exact user-approved exclusions:

```bash
python -m ra_sim.cli fit-geometry %LOCALAPPDATA%\ra_sim\Bi2Se3.json \
  --out-state artifacts/geometry_fit_recovery/bi2se3_headless_gamma_gamma/bi2se3_gamma_gamma_fit_applied.json \
  --apply-sweep-result artifacts/geometry_fit_recovery/bi2se3_headless_gamma_gamma/sweep/00_gamma_Gamma/combo_result.json \
  --approve-excluded-pair-id bg1:pair15 \
  --approve-excluded-pair-id bg0:pair20 \
  --approve-excluded-pair-id bg2:pair17
```

Successful apply writes
`04_applied_geometry_overlay.png` and `04_applied_geometry_overlay.json` beside
the output state. The JSON records the applied combo name, active variables,
approved excluded IDs, `gamma/Gamma` before and after, caked RMS/max residuals,
input/output state hashes, `geometry_updated=true`, and plotted row identities.
When the CLI writes `--out-state`, the overlay JSON's `output_state_sha256` is
finalized from the saved state file; lower-level apply helpers mark that hash as
pending until an output state exists.
Failed apply attempts report failure with geometry unchanged.
The saved manual pairs are preserved; the output state records
`geometry.geometry_fit_excluded_pair_ids` and
`geometry.geometry_fit_exclusion_reason=manual_outliers_or_physical_bad_fit`.

Current Bi2Se3 status as of the 2026-05-14 exact CLI sweep: with
`bg1:pair15`, `bg0:pair20`, and `bg2:pair17` excluded, `00_gamma_Gamma`
accepted at `0.8083400569700655 deg` RMS and `2.5981580333851113 deg` max, and
`03_gamma_Gamma_center_x_center_y` accepted at `1.140136774957117 deg` RMS and
`6.206891352600093 deg` max. Required combos
`01_gamma_Gamma_theta_initial` and `02_gamma_Gamma_corto_detector` failed
closed with native child return code `3221225477`, while
`04_gamma_Gamma_theta_initial_corto_detector` and
`05_gamma_Gamma_theta_initial_center_x_center_y` failed closed with
`qr_fit_objective_incomplete` on `bg2:pair4` (`[-1,0,5]`, branch 1). Therefore
`all_supported_required_combos_pass=false`; the report's best accepted dry-run
combo remains `00_gamma_Gamma`, and no real GUI geometry is updated by the dry
run.

2026-05-16 rerun note: a direct excluded `gamma,Gamma` headless run from the
current code wrote
`artifacts/geometry_fit_recovery/bi2se3_headless_gamma_gamma/direct_00_gamma_gamma_20260516/`.
The fit resolved 79/79 QR rows with `raw_angular_rms_deg=0.7688782306077002`
and `raw_angular_max_deg=2.5981580333851113`, but the strict single-step audit
reported `proof_status="fail"`, `proof_scope="caked_space_contract"`,
`detector_panel_status="not_plotted"`, and nonzero source-authority/surface
mismatches. Treat this as low-RMS fit evidence, not accepted proof that the
points fit cleanly.

## Background Peak-Fit Diagnostic Caches

The parallel background peak-fit diagnostic script
`scripts/diagnostics/all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py`
uses pre-editor and final Qr-rod profile caches in the diagnostic output
directory.

On Windows, direct `.py` execution with the default `process` backend now
relaunches through `scripts/diagnostics/run_all_background_peak_fits.py` with
`RA_SIM_ALL_BACKGROUND_PROCESS_GUARD=1` before expensive fitting begins. The
child run should report `process_guard=True`; uncached peak fitting should
report `backend=process_pool` with multiple PIDs. Use
`BACKGROUND_FIT_BACKEND=thread` or `BACKGROUND_FIT_BACKEND=serial` to opt out
of that relaunch for debugging.

Current status as of 2026-05-21:

- pre-editor cache filename:
  `<state-stem>_pre_qr_rod_marker_editor_cache.pkl`
- pre-editor cache format: pickle envelope with schema and full input
  signature validation
- pre-editor cache identity: GUI-state filename plus background filenames,
  incident-angle mapping, backend orientation, detector geometry, fit settings,
  and peak-job signatures
- sample/output name overrides are intentionally excluded from pre-editor cache
  identity, so changing only `RA_SIM_ALL_BACKGROUND_SAMPLE_NAME` can reuse the
  same fitted data
- pre-editor reset control: `RA_SIM_RESET_PRE_EDITOR_CACHE=1`
- pre-editor cached stages: global background peak fits, local line-profile
  fits, and Qr-rod profile/marker-table construction before the marker editor
  opens
- cache filename: `<state-stem>_qr_rod_profile_cache.pkl`
- cache format: pickle envelope with schema and state cache-key validation
- cache identity: GUI-state filename, not the absolute state path, so reruns of
  the same state filename can reuse the previous final Qr-rod fit
- reset control: `RA_SIM_RESET_QR_ROD_PROFILE_CACHE=1`
- optional imported peak-edit key: `RA_SIM_QR_ROD_PEAK_EDITS`
- peak-edit mode: `RA_SIM_QR_ROD_PEAK_EDIT_MODE=popup|skip|auto`;
  default is `auto`, which opens the popup on interactive Matplotlib backends
  and skips it for CI/headless runs
- cache-hit guard: final-fit payloads must include
  `final_rod_profile_table`, `final_marker_table`,
  `final_rod_component_table`, `final_peak_edit_cache_key`, and marker columns
  `m`, `branch`, `qz_marker`, `display_l`, and either `fit_l` or `l`;
  `marker_title` is included when a final-figure label was edited
- after the final cache lookup, the diagnostic rebuilds the exported `HK=0`
  marker table from the post-editor marker rows so manually moved specular
  markers drive the marker CSV, detector selected-region figure, and
  `hk0_l3_star.png`
- `all_background_peak_fits.ipynb` has been removed from the active diagnostic
  path; the maintained executable artifact is the parallel `.py` diagnostic and
  regression tests now load joint-Qz helpers from that script
- saved background image products default to raw detector/caked backgrounds
  through `BACKGROUND_IMAGE_SUBTRACTION_DISABLED_OVERRIDE=1`; fitted peak
  models remain written separately for diagnostics
- final-fit cache keys include
  `fit_signature=joint_qz_labeled_marker_fit_specular_roi_v18_caked_phi_m90_90_plane`,
  and pre-editor Qr-rod stages use
  `qr_rod_pre_marker_profiles_hk0_roi_v17_caked_phi_m90_90_plane`, so older
  cached joint fits that could drop weak labeled markers, overfill the m=0
  low-L full profile, truncate HK=0 below an obsolete specular L window, draw
  the removed broad HK=0 fallback support mask, predate the PbI2 sideband plot
  policy, or carry marker-only HK=0 state without positive-pixel real
  `(0, "qz")` profile rows are recomputed
- pre-editor and final-fit cache validation rejects HK=0 marker tables unless
  the matching rod profile table contains real `m=0`, `branch="qz"` rows with
  positive `pixel_count`
- split Qr-rod marker editing loads imported edit JSON before the nonzero/HK=0
  phase split, writes edit JSON only after the final HK=0/specular phase is
  accepted, records nonzero L/theta_i controls in the final-fit cache key, and
  records the HK=0/specular phi/2theta ROI and accepted HK=0/qz profile rows
  in the cache key so stale ROI/profile fits are not reused
- final nonlinear Pearson-VII Qz refinement minimizes the existing
  intensity-weighted residual plus a bounded log-intensity residual, matching
  the log-scaled Qr-rod plot without changing marker-table, CSV, CLI, or
  saved-state interfaces
- PbI2 nonzero Qr-rod profiles use same-Qz transverse Qr sideband subtraction
  when enabled: `background_density_raw` keeps the central uncorrected profile,
  `qr_sideband_background_density` stores the local off-rod estimate, and
  `background_density` is the sideband-corrected profile used for fitting and
  audit
- PbI2 nonzero profile plots show raw central `background_density_raw` as
  `Data`. Available dashed `Fit` overlays use `joint_fit_density` plus
  `qr_sideband_background_density`, so the curve is drawn on the same raw-data
  basis as the selected-region display. Marker/L mapping and Qz-baseline
  cancellation checks are recorded in the generated markdown `Plot model
  decisions` table but no longer suppress available m=3 or m=4 overlays.
- PbI2 Qr-rod profile plots use a logarithmic intensity axis only for `HK=0`.
  Nonzero HK panels use linear intensity and the shared `0.5 <= L <= 3.0`
  display window. The HK=0/specular editor, detector companion preview, and
  pre-editor/profile refresh paths are controlled by phi/2theta ROI bounds; the
  specular path no longer exposes or applies `L Min` / `L Max` bounds or
  L-derived Qz clipping.
- The nonzero HK Qr-rod marker editor has a numeric `theta_i` field. Editing
  it rebuilds the detector Q maps used for nonzero rod profiles and the
  detector companion preview, and the accepted value is part of the final
  Qr-rod cache policy. During nonzero profile refresh, both plus/minus branches
  for the same `m` rebuild their Qz bins from the same active L window and
  common finite detector support, so their displayed L starts stay aligned.
- Bug status fixed as of 2026-05-22: clicking or dragging a nonzero HK rod
  panel while an `L Min` / `L Max` text box is focused blurs the box without
  submitting its draft value, so panel interaction no longer changes the
  shared L window. The editor also freezes each panel's Qz-to-L mapping for
  the popup session, so dragging a marker in `m=1`, `branch="-"` updates only
  the selected marker and does not refit the panel coordinate transform from
  the mutated marker table. Qr-rod peak-edit import/export now round-trips
  nonzero `Delta Qr`, `L Min`, `L Max`, and `theta_i` values plus HK=0
  `Phi`/`2theta` ROI bounds, and HK=0 imports with their own markers replace
  stale required ROI rows instead of merging them into the imported set, even
  when the nonzero editor phase runs before the HK=0 phase. HK=0 imports now
  immediately refresh the specular editor profile with the imported marker
  table and ROI bounds, and loading an edit file path directly applies and
  preserves the saved HK=0 ROI parameters without dropping saved nonzero
  parameters. Accepted
  nonzero `L Min` / `L Max` bounds are reused when drawing final
  detector-region Qr-rod overlays, so exported detector images clip rod bands
  the same way the picker preview did. The final detector-region and final
  profile figure paths now rebuild their region overlays from the
  GUI-accepted `rod_profile_table`, `Delta Qr`, `L Min` / `L Max`, `theta_i`,
  marker table, and specular phi/2theta ROI instead of reusing the pre-editor
  `region_overlays`; nonzero final overlays are constrained to groups still
  present in the accepted marker table, and the accepted region specs are saved as
  `figure7_<sample>_qr_rod_region_specs.csv` and `.json` next to the profile
  CSV. Accepted HK=0/specular profile rows are also hashed into the final
  Qr-rod cache identity, so the final figure cannot reuse stale cached `m=0`
  data after the GUI selection changes. After final-fit cache resolution and
  final marker/specular normalization, the script rebuilds final region overlays
  from the accepted profile table before saving region specs or detector-region
  figures, and the GUI-vs-final audit compares accepted GUI rows against the
  post-cache final rows instead of comparing the final table to itself.
- PbI2 manuscript figures default to
  `C:\Users\Kenpo\OneDrive\Documents\GitHub\PhD Work\2D-Manuscript-Draft\figures\results_pbi2`.
  Other samples keep the `results_ordered` default, and
  `RA_SIM_ALL_BACKGROUND_FIGURE_OUT_DIR` remains the explicit override.
- configured-hidden Qr-rod rows such as `HK=7` are omitted from the editor,
  support diagnostics, and final profile subplots without changing exported
  profile CSV contents.
- For PbI2 debugging runs that must show no Qr-rod background subtraction, set
  `RA_SIM_PBI2_DISABLE_BACKGROUND_SUBTRACTION=1`. This forces PbI2 transverse
  sideband subtraction off, records the mode in the pre-editor and final-fit
  cache signatures, and plots raw `background_density` against full
  `joint_fit_density`.

```powershell
$env:RA_SIM_HEADLESS = "1"
$env:RA_SIM_PBI2_DISABLE_BACKGROUND_SUBTRACTION = "1"
python scripts/diagnostics/all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py
```

The diagnostic separates fitting coordinates from display labels. `fit_l` is
the fitted marker coordinate used for Qz-to-L mapping and joint profile fits.
`display_l` is the fallback visible L value used for final Qr-rod peak
annotations. `marker_title` is the optional user-edited final-figure label. If
`marker_title` is blank, the Qr-rod figure labels a peak as
`L=<rounded display_l>`.
Display-label-only edits must update `display_l` or `marker_title` and must not
move the fitted marker position. Qr-rod peak marker edits intentionally update
`qz_marker` and therefore invalidate the final joint-fit cache.

Qr-rod peak marker edits are applied before the final Qr-rod joint-fit cache is
checked. The default `auto` mode opens the editor only on interactive
Matplotlib desktop backends and skips it for CI/headless backends. Use `popup`
to force the editor and `skip` for unattended runs. The editor can load and save
JSON marker tables through `RA_SIM_QR_ROD_PEAK_EDITS`; accepted popup edits are
hashed into the final-fit cache key so stale fitted profiles are not reused.
For HK=0/specular edits, the accepted qz profile rows are also hashed because
the final figure is drawn from the accepted profile table, not only from marker
positions.
No-edit runs also require the current final-fit signature before a cached joint
fit is reused.
On a matching pre-editor cache hit, the diagnostic still prepares the current
background images and rewrites downstream artifacts, but it skips fitting and
profile-integration stages that were already completed for the same state and
input filenames. This gets repeated runs to the Qr-rod marker editor faster
without bypassing the editor or reusing a final joint Qz fit across changed
manual marker positions.
The editor input includes the dynamically projected `HK=0` / `00L` specular
markers before cache lookup and fitting, so the specular rod peaks can be
organized with the non-specular Qr rod peaks. Select a rod panel in the editor
and press `Snap` to move all markers in that panel to nearby local profile
peaks. The popup `Import` and `Export` buttons load or save the same JSON
marker-table format used by `RA_SIM_QR_ROD_PEAK_EDITS`, so manually adjusted
peak positions and labels can be reused across runs without editing environment
variables. The editor plots and accepts click/drag positions on the fitted
integer `L` axis for each rod panel, while the saved marker table still stores
`qz_marker`. Select a marker and edit the `Label` text box to set the exact
title used for that peak in the final Qr-rod figure; clicking another marker or
accepting the popup preserves the edited title. Final Qr-rod figure labels are
drawn above and to the right of the marked peak with a leader arrow pointing
back to the peak, and generated fallback L labels are rounded to integers.
The same popup carries Qr integration controls for nonzero HK rods: `Delta Qr
(+/- A^-1)`, `L Min`, `L Max`, and `theta_i`. Moving Delta Qr refreshes the
integrated profile table shown in every nonzero rod subplot by rerunning the
existing detector Qr/Qz profile accumulator with the new width. The accepted
Delta Qr, L window, and theta_i replace the nonzero profile rows sent to the
final joint Qz fit and are included in the final-fit cache identity. The HK=0
phase instead uses only `Phi Min`, `Phi Max`, `2theta Min`, and `2theta Max`.

The detector selected-region label-position helper remains available for the
`RA_SIM_DETECTOR_LABEL_SETTINGS` JSON schema, but the parallel diagnostic final
save path no longer opens that second popup after the Qr-rod marker editor. If
the unified editor result carries detector-label entries they are used;
otherwise the generated default label positions are drawn directly on the saved
detector selected-region figure.

On Windows, direct top-level execution of the generated `.py` diagnostic
relaunches `process` and `auto` fit backend requests through the guarded runner
before expensive fitting begins. This avoids `multiprocessing.spawn`
re-importing the notebook-style script as `__mp_main__` and rerunning the whole
diagnostic inside worker children while still using full CPU process
parallelism. The equivalent explicit guarded-runner command is:

```powershell
python scripts/diagnostics/run_all_background_peak_fits.py --notebook scripts/diagnostics/all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py --fit-backend process --fit-workers 28 --process-numba-threads 1 "$env:USERPROFILE\.local\share\ra_sim\Bi2Se3.json"
```

The runner accepts either a notebook or a `.py` diagnostic through the existing
`--notebook` compatibility flag and sets the internal process guard for the
duration of the run. Use `BACKGROUND_FIT_BACKEND=thread` or
`BACKGROUND_FIT_BACKEND=serial` to skip the relaunch for debugging. Status
2026-05-26: the internal process guard no longer suppresses the final figure
output folder chooser by itself. Interactive runs without explicit output
overrides can choose the final figure folder; final figures and the used-peaks
`.md`/`.csv` write there, profile artifacts write under `profiles/`, and
large arrays/caches remain under `OUT_DIR`. Use `RA_SIM_HEADLESS=1` or
explicit output overrides for unattended runs; explicit
`RA_SIM_ALL_BACKGROUND_SAVE_DIR_EDIT_MODE=popup` opens the chooser and prints
a waiting message first. A
2026-05-07 Bi2Se3 guarded run reported
`backend=process_pool`, `pids=28`, and `global peak fitting elapsed=22.83s`,
compared with the direct Windows thread-path report of `backend=thread_pool`,
`pids=1`, and `elapsed=220.07s`.

Direct `.py` runs can replace only the sample portion of generated labels and
figure/table filename stems by setting `SAMPLE_NAME_OVERRIDE = "Bi2Te3"` in the
parameter block, or by setting `RA_SIM_ALL_BACKGROUND_SAMPLE_NAME` before
launch. This leaves `RUN_NAME` and output-directory selection
unchanged, so the override changes stems such as `figure7_bi2te3_...` without
moving the run directory.

The diagnostic writes `hk0_l3_star.png` to the figure output directory. This is
a raw detector-intensity crop from the beam center through and above the
drawable `HK=0`, `L=3` / `00L` marker, saved with the same colored detector
colormap and log intensity normalization used by the detector-style diagnostic
figures. This log scaling applies only to the `hk0_l3_star.png` crop. If the
beam center, detector image, or drawable `HK=0`, `L=3` marker cannot be
resolved, the script prints a skipped reason and continues.

Focused validation status:

- `python -m py_compile scripts/diagnostics/all_background_peak_fits_peak_only_shared_linear_baseline_global_fit_parallel.py`
  passes
- `python -m pytest tests/test_background_peak_fits_notebook.py -k "qr_rod_peak_editor_is_wired_before_joint_fit_cache or qr_rod_peak_edit_runtime_mode_respects_headless or qr_sideband or pbi2_plot_policy or pbi2_debug or background_debug_policy or final_profile_plot_uses_model_decisions or pbi2_rod_profile_l_axis or pbi2_final_profile or shared_nonzero_rod_profile_y_axis_limits" -ra`
  passes with `16 passed`, including the PbI2 no-background debug flag and
  default-auto headless editor coverage
- `python -m pytest tests/test_background_peak_fits_notebook.py -k "hk0_l3_star or qr_rod_peak or qr_rod_marker or marker_title or sample_name_override or import_export_buttons or labeled_weak_hk0_marker or qr_rod_final_cache_requires_fit_signature or final_rod_labels_point_from_upper_right or qr_rod_editor_qz_l_axis_coefficients or qr_rod_peak_editor_uses_l_axis" -ra`
  passes, `21 passed`
- `python -m pytest tests/test_background_peak_fits_notebook.py -k "runner or backend or process" -ra`
  passes, `8 passed`
- `python -m pytest tests/test_background_peak_fits_notebook.py -k "detector_region_label" -ra`
  passes, `15 passed`, including the in-figure detector-label editor contract,
  settings round trip, runtime-mode handling, and final-save wiring
- `python -m pytest tests/test_background_peak_fits_notebook.py -k "unified_qr_rod_region_editor or unified_editor or detector_region_label_editor_wires_before_final_save or saved_figures_do_not_include_panel_letters or initial_placement_uses_default_geometry or axis_tick_labels_use_bottom_left_origin or qr_rod_peak_editor_is_wired_before_joint_fit_cache or pre_editor_cache_is_checked_before_expensive_stages or qr_rod_peak_editor_uses_l_axis" -ra`
  passes, `13 passed`, including same-popup Delta Qr/L controls, live profile
  refresh callback wiring, and accepted profile-table handoff to the final fit
- `RA_SIM_HEADLESS=1 RA_SIM_PBI2_DISABLE_BACKGROUND_SUBTRACTION=1` PbI2
  diagnostic script execution completed, skipped the marker and detector-label
  popups through default `auto` mode, recorded sideband subtraction disabled in
  generated markdown, kept `background_density == background_density_raw`,
  used log axes on every PbI2 panel, capped displayed L at 3, and skipped
  unsupported `m=7` in the final figure
- full `tests/test_background_peak_fits_notebook.py` was rerun on 2026-05-11
  and remains red only in four unrelated notebook/script source-token checks:
  `130 passed`, `2 skipped`, `4 failed`
- current `python -m ra_sim.dev check` is blocked by formatting drift in
  `ra_sim/fitting/optimization.py` and the unrelated dirty
  `ra_sim/gui/_runtime/runtime_session.py`
- Bi2Se3 guarded-process diagnostic validation completed with `fit_backend=process`,
  `backend=process_pool`, `pids=28`, `79/79` successful background peak fits,
  final Qr-rod cache write, marker CSV columns `fit_l`/`display_l`, and
  regenerated rod-profile figures under the benchmark artifact directory

## GUI Runtime Update Fast Paths

The GUI runtime records a pure update decision before it executes an update.
The trace fields are:

- `update_action`
- `update_reason`
- `requires_worker`
- `missing_contribution_count`
- `center_remap_used`
- `primary_prune_cache_mode`

Current action status:

- `full_simulation`: feature status implemented; still the conservative
  fallback for source, lattice, detector distance, detector rotation, pixel
  size, wavelength, beam sampling, mosaic sampling, solve-q, or mixed physics
  changes
- `primary_prune_reuse`: feature status implemented; rematerializes the
  primary image from cached contribution hit tables and does not request a full
  worker when all active keys are cached
- `primary_prune_fill`: feature status implemented; requests only missing
  primary contribution keys and preserves already cached keys
- `detector_center_remap`: feature status implemented; runs only with exact
  detector-relative/unclipped hit-table caches and falls back to full simulation
  for missing, clipped-only, stale, incompatible, or secondary-missing remap
  caches
- `display_only`: feature status implemented; redraws existing stored image
  state only when a valid stored image exists
- `combine_only`: feature status implemented; recombines cached primary and
  secondary images only when matching component images exist
- `analysis_only`: feature status implemented for current stored-image redraws;
  preserves cached simulation data for analysis/display updates such as
  background visibility toggles, and falls back to full simulation when the
  stored image signature is stale

Bug/error status:

- background visibility toggles no longer promote an `analysis_only` classifier
  decision into a full simulation when the stored image is current; the
  runtime syncs the existing background artist alpha and schedules only the
  cached-image update path
- stale full-worker results are discarded before newer display, prune-reuse, or
  detector-center-remap fast-path state can be overwritten
- `last_dependency_signatures` is updated only after the applied state reflects
  the current request, with display/combine paths allowed to advance dependency
  signatures when the stored numeric image is current and no worker is pending
- detector-center remap invalidates caking/q-space/detector-geometry analysis
  caches while preserving reusable source/physics contribution caches
- broad full-simulation invalidation remains the fallback for physics changes
  and signature-incompatible cache states

Validation status as of 2026-05-21:

- `python -m pytest tests/test_gui_runtime_import_safe.py -k "background_visibility_analysis_only or runtime_trace_records_classifier_display_decision" -ra`
  passed
- `python -m pytest tests/test_gui_runtime_background.py tests/test_gui_runtime_update_dependencies.py -ra`
  passed
- `python -m ra_sim.dev check` passed
- no saved-state, config, CLI, artifact-schema, or migration/deprecation work is
  required for this bug fix

Validation status as of 2026-04-28:

- final pre-merge requested suite
  `python -m pytest -ra tests/test_gui_runtime_update_dependencies.py tests/test_gui_runtime_detector_remap_cache.py tests/test_gui_runtime_primary_cache.py tests/test_gui_sim_signature.py tests/test_gui_runtime_update_trace.py tests/test_gui_runtime_update_actions.py tests/test_gui_runtime_invalidation.py tests/test_gui_runtime_optimization_scenarios.py tests/test_gui_runtime_import_safe.py`
  passed, `408 passed`
- manual GUI-runtime trace smoke passed with these action labels:
  `full_simulation` for initial update, detector-center without exact cache,
  detector distance change, and detector rotation/tilt change;
  `primary_prune_reuse` for prune with cached keys; `display_only` for a
  display-only change; `combine_only` for the combine/visibility change; and
  `detector_center_remap` for center shift with exact cache
- `python -m pytest tests/test_gui_runtime_update_dependencies.py tests/test_gui_runtime_detector_remap_cache.py tests/test_gui_runtime_primary_cache.py tests/test_gui_sim_signature.py tests/test_gui_runtime_update_trace.py tests/test_gui_runtime_update_actions.py tests/test_gui_runtime_invalidation.py tests/test_gui_runtime_optimization_scenarios.py -ra`
  passed, `92 passed`
- broad GUI runtime slice
  `python -m pytest tests/test_gui_runtime_*.py tests/test_gui_sim_signature.py -ra`
  passed via explicit PowerShell file expansion, `448 passed`
- `python -m compileall ra_sim/gui tests/test_gui_runtime_optimization_scenarios.py tests/test_gui_runtime_update_dependencies.py -q`
  passed

Repository hygiene status:

- fast-path implementation lives in `e756ba0`
  (`perf(gui): add update fast paths`)
- validation status lives in `32c6ac9`
  (`docs(gui): record fast-path validation`)
- final audit confirmed the fast-path commit contains only fast-path
  docs/code/tests; unrelated dirty files were left unstaged
- commit-time Git worktree cleanup warnings were permission warnings only and
  did not block commit creation
- `git diff --check` found no whitespace errors, only CRLF normalization
  warnings in the dirty worktree

## Geometry Qr/Qz and HKL Picker State

Manual Qr/Qz picking uses a structural group cache derived from the active
simulation hit tables and CIF/lattice state. It is not a detector-view or
caked-view cache. Detector/caked view switches must not invalidate or filter
the Qr/Qz group universe. CIF, unit-cell, or simulation-hit-table changes do
invalidate it.

Status as of 2026-05-22: `theta_i` / `theta_initial` is part of the live
simulation source-row content identity. Changing it is a full-simulation
update, refreshes Qr/Qz picker rows and geometry source-row snapshots, and
retains only user masks such as enabled/disabled Qr groups. This prevents
selected Qr/Qz sets from reusing stale source rows after sample-tilt changes.
Saved manual Qr/Qz placements from zero-intensity ghost representatives keep
their measured background point, but their simulated square marker is replayed
from current source rows after simulation changes, falling back to current
simulated-peak rows when the source-row snapshot is empty. The refresh is
required even when the saved placed row no longer carries ghost provenance and
only the warm manual-pick cache identifies the resolved simulated source as a
ghost. If the current ghost source identity is missing from both current
providers, the marker is reported unresolved instead of drawing the old saved
simulated pixel or falling back to saved refined simulated coordinates as
current truth. Full-simulation invalidation also clears remembered geometry-fit
overlay records so stale fit markers cannot preempt the saved-pair redraw path
after the new source rows apply. Settled overlay refreshes replay durable fitted
overlay records only; an `initial_pairs_display`-only remembered state is
treated as view-bound and rebuilt through the current manual-pair renderer.
Validation for this path should inspect the actual Matplotlib overlay artists:
the simulated Qr/Qz square must be drawn at the current ghost row coordinates,
not just returned in the intermediate display payload.
The selected Qr-set integration overlay is refreshed as an integration overlay
even when the separate geometry overlay visibility toggle is off.

Status as of 2026-05-25: fixed the visible selected-Qr blue-square bug where
changing `theta_initial` refreshed the live source row but the simulated marker
could still replay saved `refined_sim_x/y` detector pixels. Detector-view
initial-pair redraw now lets a same-source live row with changed theta move the
simulated square while preserving the measured background marker and saved
refined fallback for non-theta replay. Public GUI-state, CLI, config, and
artifact schemas are unchanged; no migration or deprecation is required.
Validation covers both the display-payload builder and the Matplotlib overlay
artist coordinates.

Generated disordered-phase Qr/Qz groups are structural rows, not live display
artifacts. When `Include generated disordered-phase Qr refs` is enabled and a
nonzero disordered stacking component is active, the runtime generates the
HT-shifted disordered CIF from the active PbI2 CIF, builds hit-table rows with
`accumulate_image=False`, tags them as `disordered_phase`, includes that source
signature in picker-cache validity checks, and publishes them into the active
Qr/Qz picker cache during current-simulation refreshes.

Optional PbI2 6H reference Qr/Qz groups are opt-in legacy structural rows, not
live display artifacts. When `Include packaged 6H Qr refs` is enabled and `w1` is nonzero,
the runtime loads the packaged `ra_sim.config/materials/PbI2_6H.cif`, builds
6H hit-table rows with the current wavelength/window/HKL limits, and tags them
as `pbii_6h_ref`. The q-group and manual-pick signatures include the toggle,
`w1`, HKL limit, wavelength, 2theta window, and intensity threshold so stale
picker rows are not reused across 6H-reference changes.

Duplicate Qr/Qz identities from primary, secondary, stacking-fault, or 6H
reference rows are merged numerically before listing and picking. The merge
tolerance is `atol=1e-6`, `rtol=1e-8`; one selector key remains, intensity and
candidate points are combined, and aux source aliases are kept for diagnostics.

Resolved picker behavior depends on keeping structural simulation truth
separate from current-view projection. Group membership comes from structural
state; detector or caked coordinates are derived later for the active view.

Detector-view manual Qr/Qz picking must not reuse a manual-pick cache that
matches the current signature but contains no detector source rows or detector
picker candidates. A matching empty detector cache is stale and must rebuild
from the current source snapshot, live peak records, or fresh simulation rows.
Only caked mode may accept `caked_qr_projection_grouped_candidates` as the
cache-reuse gate.

`manual_pick_cache` source-row lookup may rebuild a missing, stale, or empty
source snapshot only for the current background and only when stored simulation
artifacts exist (`stored_max_positions_local`, `stored_intersection_cache`, or
`peak_records`). Detector manual-pick rebuilds use detector projection mode
unless the manual picker is explicitly in caked space. This keeps detector
picking independent of caked integration while preserving the existing
geometry-fit dataset rebuild path for non-current backgrounds and targeted
preflight.

Detector picker diagnostics should distinguish these cases:

- matching empty manual-pick cache rebuilt instead of reused
- source snapshot missing, stale, empty, or rebuilt
- source rows present but missing `q_group_key`
- source rows present but missing detector display pixels
- detector candidates present by source row family

Manual Qr/Qz mouse clicks are no-build paths. Selection clicks call the
manual-pick cache with `reuse_only=True` and may use only a warm cache whose
picker signature, grouped candidates, refinement signature, and refined lookup
signature all match the current simulation/projection state. A reuse-only miss
returns a non-persisted `cache_ready=False` payload and must not replace or
erase the previous warm cache. Placement clicks use the selected session's
`remaining_candidates`; if the reuse-only cache is cold, local measured-peak
refinement receives a `manual_no_build_cache` sentinel so the shared refiner
cannot call the picker-cache builder indirectly.

Background Qr reference placement and redraw are background-only paths. `Place
Background Qr Set` saves a fit-disabled manual reference with measured
background coordinates and no HKL/Qr group identity. Placement uses a non-empty
no-build sentinel, not simulated picker inventory. Redraw renders
background-only references directly from saved coordinates and preserves mixed
saved-entry order with indexed placeholders, so simulation lookup runs only for
simulation-backed entries.

Manual-pick cache diagnostics now attach optional timing/count fields to
`cache_metadata` instead of printing by default. The main fields are
`build_pick_cache_wall_ms`, `get_pick_cache_total_wall_ms`,
`get_pick_cache_build_wall_ms`, `qr_sim_refinement_wall_ms`,
`qr_lookup_rebuild_wall_ms`, `source_rows_provider_calls`,
`fresh_simulation_prefer_cache_false_attempted`,
`qr_refinement_unique_candidate_count`,
`qr_refinement_reused_candidate_count`, and
`qr_refinement_total_row_count`. These fields are diagnostics only and are not
part of the persisted manual-pair schema.

Caked manual picking uses two different coordinate responsibilities:

- simulated Qr/Qz and HKL seed positions start from simulation-native detector
  branch pixels;
- caked click targets map those simulation-native pixels through the live caked
  simulation transform into `(2theta, phi)`;
- detector aliases such as `sim_col`, `sim_row`, `display_col`, and
  `display_row` remain detector/display coordinates;
- caked aliases such as `caked_x`, `caked_y`, `raw_caked_x`, `raw_caked_y`,
  `two_theta_deg`, and `phi_deg` hold current-view angular coordinates.

Manual background replay treats `manual_background_input_origin` as the
authoritative persisted origin. If a legacy or mixed saved row says
`manual_background_input_origin="detector"` but carries a stale caked
`manual_background_input_frame`, detector origin still wins and caked redraw
must reproject from detector truth instead of reading stale saved caked fields.
New manual background placements now persist both origin and frame
(`detector_display` or `caked_2theta_phi`) so future replay does not have to
infer the coordinate family from ambiguous legacy aliases.

For source-backed caked Qr/Qz selection, the detector-to-caked projection cache
is the authority for hit testing, active selected markers, and saved-pair
redraw. The cache is keyed by stable source/branch identity and stores the
native detector point, detector display point, caked visual point, and caked
`(2theta, phi)`. Saved or refined aliases can still describe the measured
background point, but they must not override the simulated Qr/Qz marker for a
source-backed saved pair.

Caked projection sidecars are view projections of the source cache, not source
truth. A detector-warmed manual-pick cache may satisfy a caked reuse-only click
only when the source/mask signature matches, the caked sidecar exists, and the
current caked projection token matches the cached sidecar token. If a caked
sidecar is requested and the current token is unavailable, the reuse-only path
must fail closed. Detector-only reuse does not require a caked token.

Async geometry-fit worker jobs must not call GUI/Tk manual-pair callbacks.
Worker dataset builds use job-local detector or caked coordinate fallbacks and
tag caked/q-space projected rows with worker-local cache markers. Those markers
are refreshed on every worker handoff so stale internal metadata cannot force a
second caked projection for rows that are already projected in the job payload.

Saved caked redraw must resolve simulated overlay points in this order:
source-matched projected caked row, source-matched resolved caked row, saved
simulation-only caked fields, then detector/native fallback only if no caked
simulated point exists. Measured/background caked fields such as
`background_two_theta_deg`, `background_phi_deg`, `caked_x`, or `caked_y`
belong to the background marker and must not contaminate the simulated marker.
Bare `caked_x/y` is intentionally ambiguous and may become simulated fit/cache
truth only for explicit simulated caked projection rows with source identity
and caked-projection provenance. Background/replay-shaped rows must use
explicit simulated fields such as `sim_refined_caked_deg`,
`refined_sim_caked_x/y`, `simulated_two_theta_deg/simulated_phi_deg`, or
`sim_caked_display`.

Geometry-fit overlay records must keep the visible fitted-simulation marker
source separate from stale legacy fit/cache aliases. Old caked point-only
records may carry the current caked `(2theta, phi)` prediction under
`fit_prediction_detector_display_px`; overlay replay may use that legacy alias
as `final_sim_caked_display` only when objective metadata proves the row is
caked, degree-based, or point-only. New live-cache progress records suppress
that detector-display value for caked-display rows, set
`fit_prediction_detector_display_px` to `null`, set
`fit_prediction_detector_display_px_unavailable_reason` to
`caked_degrees_not_detector_display_px`, and expose the coordinate as
`fit_prediction_caked_deg`. Live-cache records that carry a true detector
display prediction and detector/native projection proof keep
`fit_prediction_detector_display_px` unchanged. Detector display pixels must
not be promoted to caked angles without that metadata. Overlay diagnostics record
`final_sim_display_source`, `final_sim_native_source`, and
`final_sim_caked_display_source` so the drawn green `fit sim` marker source can
be compared with cached/calculated distance fields and the rendered caked
simulation image.

Live caked visual-source ledger rows are disabled by default. Set
`RA_SIM_LIVE_CAKED_TRACE=1` to print `[ra-sim] live_caked_visual_source` rows,
and add `RA_SIM_LIVE_CAKED_TRACE_ALL=1` to include unchanged duplicate rows.
`RA_SIM_SUPPRESS_LIVE_CAKED_TRACE=1` suppresses the ledger regardless of the
enable flags.

Warm-cache simulated-candidate refinement relies on cache, simulation, and
exact caked-projection signatures plus stable full-content value tokens. Projection-only
payloads are valid geometry inputs even when the display/background caked image
is absent. Caked pick-cache signatures intentionally ignore image-facing
background payload identity, so display density sanitization, including
zero-support `NaN -> 0.0` storage cleanup, must not churn the geometric
projection cache. Stable projection signatures use the verified
`projection_content_token_v3` recomputed during projection-payload
storage/hydration, plus full-content value tokens for axes/permutations
(`axis_content_v3` / `perm_content_v3`), not array object IDs, sampled axis
probes, legacy `signature` fields, full image hashes, or click-path LUT hashes.
DetectorCakeLUT-style projection tokens include `image_shape`, `n_rad`, `n_az`,
the detector-to-cake matrix content, and `count_flat` content. Projection
payload storage copies axes, permutations, and `CakeTransformBundle` LUT content
into private read-only objects before attaching a trusted
`projection_content_token_source`. Shared incoming bundles are not frozen.
Source-less or legacy projection tokens are not correctness keys for warm-cache
reuse. Do not mutate simulation or caked image arrays in place without also
bumping the corresponding simulation or projection signature.

Status as of 2026-05-25: detector-origin locked Qr/Qz source identity is no
longer a caked-objective trigger during Fit Geometry job build. Detector picks
with locked source rows stay in detector fit-space unless the requested
objective or selected manual fit-space is explicitly caked, so the GUI thread
does not call `ensure_geometry_fit_caked_view()` or load caked payloads before
submitting the async worker. The locked-Qr selected-row projection readiness
gate still runs in the worker. When a detector-mode worker caked payload is
available, readiness is refreshed from the stored selected detector rows
projected through that exact payload; this validation-only projection does not
promote the detector fit-space or caked-objective flags. The gate still fails
before dataset build when required projected rows are unavailable or nonfinite.
This separates provenance from fit-space selection while preserving fail-closed
locked-Qr validation.

Status as of 2026-05-28: review follow-up kept the manual Qr/Qz caked
fit-space behavior unchanged and only simplified the GUI preflight expression.
The dead `ensure_geometry_fit_caked_view` self-assignment was removed, and the
mixed-provenance preflight error remains scoped to the no-explicit-caked-
requirement path. Regression coverage now also asserts that an explicit caked
objective marks every prepared dataset spec with
`solver_requested_objective_space == "caked_deg"` even when detector-origin and
caked-origin pick provenance are mixed. Bug/error status: fixed and guarded in
commit `978ee3fe`. Migration/deprecation status: no saved-state schema, CLI
flag, config key, artifact field, dependency, or public workflow changed.
Shipping status: local quality gates passed before commit; no CI workflow,
feature flag, staged rollout, or release version bump is required, and rollback
is a normal git revert.

Status as of 2026-05-21: `objective_space=caked_deg` is now the fit-space
requirement source of truth for manual Qr/Qz geometry fits, even when the manual
pick provenance remains detector-origin. If every manual pair has finite
observed and predicted caked anchors, the optimizer routes to a caked degree
evaluator and logs `manual_caked_route_check ... observed_caked=2
predicted_caked=2 ... unit=deg` for the live two-pair path. If required caked
fit-space is missing, the fit fails before optimization with
`manual_caked_fit_space_missing`. A caked objective is not allowed to finalize as
`central_point_match`, `metric_unit=px`, or pixel `weighted_rms`; that invariant
fails closed as `manual_caked_route_invariant_violation`. The handoff checker
rejects the reported `f761e78f` text-log signature across whole-file state, and
the GUI rejection copy names the caked route block instead of reporting "No
matched peak pairs" for a caked manual-pair fit. Bug/error status: fixed in
source and automated regression tests for the live two-pair manual Tk failure
shape. Migration status: no saved-state schema, CLI flag, config key, artifact
field, dependency, or user workflow migration changed; the removed path is only
the internal caked-objective fall-through to detector-pixel point matching.
Shipping status: local focused tests, GUI safety tests, `python -m ra_sim.dev
check`, and `git diff --check` pass; rollback is a git revert. Remaining
acceptance proof before release: rerun the actual Tk Bi2Se3 workflow and check
the new live log.

Status as of 2026-05-20: detector-origin manual Qr/Qz geometry-fit rows are
kept in detector-pixel fit space, including `gamma,Gamma` two-tilt solves.
Saved caked aliases on those rows remain replay/display cache data and no
longer promote the row into the exact-caked angular objective. The manual
point-fit runtime also disables seed multistart for detector-origin rows, so a
simple picked-point fit reaches the direct LSQ solver instead of being rejected
when seed prescoring selects no candidates. Explicit caked-origin rows still
require the exact per-background projector and fail closed if it is unavailable.
Bug/error status: fixed for the reported GUI `Geometry Fit Rejected` paths where
a simple two-point detector-origin fit could be rejected by stale or
axis-mismatched caked residuals, or by `selected=0` in seed multistart before
the direct solver ran. Migration status: no saved-state schema, CLI flag,
config key, or artifact field changed; the removed path is internal
detector-origin auto-caked promotion plus unnecessary seed search in the GUI
manual-point profile. Shipping status: ready as a normal bug-fix slice after
local focused tests and `python -m ra_sim.dev check`; rollback is a git revert.

Status as of 2026-05-15: detector-origin manual Qr/Qz geometry-fit rows keep
their saved `manual_background_input_origin` and frame through dataset
orientation. The exact caked projector now replaces stale saved
`background_two_theta_deg` / `background_phi_deg` aliases before
`measured_for_fit`, the optimizer dataset spec, and QR handoff audit consume the
pair. Bug/error status: fixed for the reported `(-1,0,10)` two-branch handoff
where stale caked angles caused non-finite angular residual rejection.
Follow-up status: the regression fixture now shares one source-identity payload
between the simulated row and manual pair setup, reducing duplicate test setup
without changing the fitter contract or bug status.

Status as of 2026-05-07: detector/caked manual background replay now preserves
the saved origin/frame contract across detector -> caked -> detector view
replays. Detector-origin rows no longer fall back to stale caked fields when the
live caked projection is unavailable, and caked-origin rows preserve their
visual caked anchor while still projecting back to detector display when needed.
Manual pair creation also keeps ambiguous bare `caked_x/y` out of simulated
fit/cache fields unless the row is explicitly a simulated caked projection.

Status as of 2026-05-05: live caked trace output is opt-in, unchanged trace rows
are suppressed unless explicitly requested, warm caked pick-cache calls skip
row-level refinement when simulation/projection signatures are unchanged,
zero-support display sanitization no longer invalidates caked pick caches,
equivalent copied axes and projection payloads keep the same signature/digest,
token-only projection payloads are absent and can still generate fallback
payloads, explicit trusted projection signatures survive
normalize/hydrate/digest handoff, failed
lookup rebuilds retry, and no-signature direct refine/rebuild calls clear stale
skip metadata. The exact-caked cold-start path accepts projection-only payloads
without requiring a display image. The New4 ladder finalizer now repairs stale
exact-caked report/polish fields only when the selected exact-caked summary is
clean; real missing/lost manual-pair cases still fail the fixed-source gate.
Active saved-state validation now runs Bi2Se3 and Bi2Te3 from the RA-SIM user
data root with direct fixed-pair solves; both pass all fixed-pair matching and
residual-reduction gates.

Live GUI source-row fallback status as of 2026-05-05: after
`projection_payload_ready`, caked geometry-fit preflight now logs both
`background_index_internal` and a 1-based background label, records manual-pair
backgrounds and required/matched pair counts, and reports reject reasons for
live, targeted, memory, logged, and current-hit-table cache paths. Current
background live rows are validated before targeted-cache lookup. The
current-hit-table fallback accepts only `stored_max_positions_local` provenance
with matching table/requested base signatures, background, fit space, trusted
projection signature, source IDs, q-group keys, branch IDs, and fixed-pair
count. `last_intersection_cache` remains memory/logged-cache-only and must not
be rewrapped as current-hit metadata. If current live rows or signature-matched
current hit tables are usable, fresh simulation must not start. If the slow
fresh-simulation fallback does run, targeted and plain rebuilds both emit a
visible timeout plus late/still-running status before the eventual ready or
failed terminal event.

Status as of 2026-05-25: fixed the plain fresh-simulation timeout path so
geometry-fit preflight fails before optimizer preparation when a non-targeted
source-cache rebuild stalls. This matches targeted preflight timeout handling,
keeps timeout events in the existing source-cache stage stream, and does not add
new cache formats, migration steps, feature flags, or CI workflow changes. A
follow-up refactor only centralizes the derived fresh-simulation stage labels
and timeout status; emitted event names, diagnostics, and exception text remain
unchanged.

Validation status: runtime-level regressions cover the observed miss sequence
(`projection_payload_ready`, targeted projected miss, memory miss, logged miss)
and prove trusted current hit tables are consumed before `simulate_hit_tables`.
Forged `intersection_cache`/`last_intersection_cache` current-hit payloads are
rejected before source-row build. A background-index regression covers manual
pairs on UI background 2 building internal index 1 and not background 1. The
2026-05-25 timeout regression covers the plain fresh-simulation preflight block
and the existing targeted timeout workflow regression covers the targeted stage
names. The 2026-05-25 selected-Qr regression covers the data-builder and overlay
artist paths for theta-updated simulated markers. The
current Bi quality baseline ran the
same direct fixed-manual path as the smokes: Bi2Se3 accepted with 82/82 fixed
pairs matched, 0 missing, 0 branch mismatches, and direct RMS 34.5307 -> 31.078112;
Bi2Te3 accepted with 84/84 fixed pairs matched, 0 missing, 0 branch mismatches,
and direct RMS 36.8629 -> 34.394414. The live GUI manual acceptance smoke still
requires operator verification on a fresh session: import a Bi state, generate
once, run Fit Geometry without changing parameters, and confirm no first-fit
timeout, `invalid_exact_caked_payload`, or `exact caked projector unavailable`.

The HKL picker intentionally shares the corrected Qr/manual picker candidate
payload for hit testing and selected-marker placement. If either picker
regresses, first check whether the failing path bypassed that shared candidate
payload or treated detector/display aliases as caked coordinates.

## QR Selector Fast-Path Cache Policy

QR selector cache retention is centralized in
`ra_sim/gui/runtime_qr_selector_cache_policy.py`. Runtime invalidation calls the
policy before clearing selector entries, source-row snapshots, intersection
caches, or manual-pick projection payloads.

Status as of 2026-04-28:

- feature status: implemented for display-only, combine-only, analysis-only,
  primary-prune reuse, primary-prune fill, detector-center remap, and full
  simulation update actions; local Phase 3.5 validation also adds fast
  geometry-fitter handoff tests and optional New4 fixture skips; Phase 8 adds
  `scripts/debug/run_geometry_fitter_cache_regression_gate.py` as the repeatable
  local/strict cache regression gate; Phase 9 adds mixed-update and stale-worker
  sequence coverage to that gate
- bug status: fixed for overbroad fast-path invalidation that could clear QR
  selector entries or fitter handoff data before replacement rows were ready;
  fixed local New4 validation failures caused by absent optional artifacts
- error status: targeted cache-policy, runtime-invalidation, and fast handoff
  tests pass; slow/manual caked-refined geometry diagnostics are excluded from
  local mode and included in strict mode
- compatibility status: `disabled_qr_sets`, `disabled_qz_sections`, and
  `pending_legacy_disabled_qz_sections` remain explicit user/state selections
  and are not cleared by cache invalidation

Retention rules:

- display-only, combine-only, and analysis-only actions retain selector and
  fitter handoff caches
- primary-prune reuse keeps q-group entries when content signatures are
  unchanged and requests refresh when q-group content changes
- primary-prune fill keeps old q-group entries until replacement rows apply
- detector-center remap retains branch/source identity for exact remaps while
  refreshing projection geometry when detector geometry changes
- full simulation clears stale source rows and cache payloads when physics or
  hit-table signatures change, but retains QR masks

Validation:

- `python -m pytest tests/test_gui_runtime_geometry_fitter_handoff_fast.py -q`
  passed, `5 passed`
- local Phase 3.5 gate passed, `438 passed`
- New4 workflow slice passed locally with `26 passed, 2 skipped` when the
  optional `artifacts/geometry_fit_gui_states/new4.json` fixture was absent
- `python -m pytest tests/test_runtime_qr_selector_cache_policy.py tests/test_gui_runtime_invalidation.py -q`
  passed, `23 passed`
- `python -m pytest tests/test_gui_runtime_update_actions.py tests/test_gui_runtime_optimization_scenarios.py -q`
  passed, `22 passed`
- `python -m py_compile ra_sim/gui/runtime_invalidation.py ra_sim/gui/_runtime/runtime_session.py ra_sim/gui/runtime_qr_selector_cache_policy.py tests/test_gui_runtime_invalidation.py`
  passed
- `python scripts/debug/run_geometry_fitter_cache_regression_gate.py --mode local`
  passed; untracked local New4 artifacts are skipped by default unless
  `RA_SIM_ALLOW_UNTRACKED_NEW4=1` is set
- Phase 9 local gate passed with the mixed-update suite included: fast gate
  `497 passed`, manual identity `5 passed, 423 deselected`, workflow slice
  `26 passed, 2 skipped`

## Geometry-fit coordinate lineage diagnostics

GUI geometry-fit logs now include a bounded `Coordinate lineage:` section when
fit handoff audit rows exist. Each `coord_lineage` group uses the same
background, pair, q-group, HKL, branch, table, row, and peak identity fields
across:

- cached/refined simulation caked coordinates from the GUI handoff row
- fit-dataset prediction caked coordinates used to build the objective
- the first residual-evaluation snapshots from the optimizer
- final point-match diagnostics
- draw-time visual probe rows when available

Phi deltas are wrapped before norm calculations, and the summary reports
`first_divergence_stage` plus `max_delta_deg`. GUI solver requests opt into four
objective-trace snapshots on the effective optimizer/solver config by default;
true headless geometry-fit requests remain opt-in. Draw-time `visual_probe ...`
rows now also print the same identity fields, so a green `fit sim` marker can be
joined back to its cached and residual-eval coordinates.

For caked point-only Qr/Qz geometry fits, current trial hit tables are the
simulated detector-coordinate source of truth. The objective then projects those
detector coordinates through the exact detector-to-caked projector. Cached
`sim_visual_caked_deg` aliases are no longer used as the objective point when a
current hit-table row is available.

If a current hit table cannot resolve the locked Qr source row, or only recovers
a provider-local stale hit row, the objective now falls back to the current trial
source-row builder and the same exact caked projector. This fallback is still
dynamic trial data, not a stale visual/cached caked alias. If source rows also
cannot resolve the locked branch, the point fails closed through
`qr_fit_objective_incomplete`. Missing-pair diagnostics include hit-table
resolution reason, source-row availability/count/signature/source, source-row
candidate counts, `source_rows_rebuilt_or_reused`, and objective-cache status so
future failures distinguish "source rows unavailable" from "source rows were not
attempted."

## New4 single-step QR coordinate visual audit

Status as of 2026-05-13:

- feature status: debug-only proof mode
- bug/error status: fixed for caked-display QR source rows being reprojected as
  detector points. Valid caked-display rows now use the live
  `sim_visual_caked_deg` surface for the caked objective, and the regenerated
  New4 artifact reports `proof_status="pass"` with 7/7 source-authority matches.
- follow-up status: visual/objective surface divergence remains fail-closed.
  The artifact now records `source_authority_match_all_caked_display_rows`,
  `source_authority_mismatch_row_count`, and per-row
  `point_only_detector_projection_used` so future regressions cannot hide a
  `point_only_detector_projection` objective behind a caked visual row.
- migration status: no public API, config, saved-state, or artifact-schema
  migration is required; the new mode and optional diagnostic override are
  additive debug-script flags
- shipping status: local artifact generation is complete, generated PNG/JSON/CSV
  files are diagnostic proof artifacts and this is not a production release
- maintenance status: the single-step proof checks share small row-value and
  row-predicate helpers, reducing duplicate audit bookkeeping without changing
  JSON fields, PNG semantics, CLI flags, config, or saved-state behavior
- contract status: QR audit rows now include a nested
  `qr_fit_point_surface_contract` plus top-level contract counters. The
  regenerated New4 proof reports `qr_fit_contract_status="pass"`,
  `qr_fit_contract_failure_count=0`, `surface_mismatch_row_count=0`, and
  `source_authority_mismatch_row_count=0`.
- full-fit status: headless dynamic QR fits now reuse the same contract payloads
  in their point-match summary. The New4 `gamma,Gamma` reproduction reports
  `qr_fit_expected_count=7`, `qr_fit_resolved_count=7`,
  `qr_fit_missing_pairs=[]`, `qr_fit_contract_status="pass"`,
  `source_authority_mismatch_count=0`,
  `visual_objective_surface_mismatch_count=0`, and
  `objective_param_sensitivity_status="sensitive"`. The fit remains correctly
  rejected as `branch_source_pairing_mismatch`, with worst rows listed, rather
  than as a source-surface mismatch.
- full-fit maintenance status: the contract assembly now reuses the canonical
  manual/objective caked point payloads and resolves the GUI-drawn simulation
  source once before writing both source fields. This preserves the existing
  diagnostic schema and has no migration, deprecation, config, saved-state, or
  launch impact. A later long New4 headless recheck was interrupted before
  completion, so generated recovery artifacts from that attempt are diagnostic
  scratch output only and are not versioned.
- sensitivity status: dynamic caked QR fits now use a bounded angular
  sensitivity ladder for `gamma`, `Gamma`, and `theta_initial`-style variables:
  `0.1, 0.25, 0.5, 1, 2, 5` degrees. The diagnostic does not classify these
  variables from sub-0.1-degree probes alone and records
  `steps_deg`, `first_meaningful_step_deg`,
  `max_prediction_delta_caked_deg`, `max_residual_vector_delta_deg`, and probe
  metadata for trial params, source-row rebuild/cache state, and projection
  signatures. This is additive diagnostic output; it does not change fit
  thresholds or branch identity. Existing insensitive-objective rejection now
  has to be supported by the ladder through 5 degrees, and cache-reuse states
  such as `reused_for_same_params_signature` remain distinct from rebuilt source
  rows.
- fail-closed status: dataset-spec-backed dynamic caked fits now reject with
  `dynamic_objective_not_sensitive_to_fit_variables` when every active fit
  variable is insensitive across that ladder. The optimizer emits
  `Geometry fit: failed` for that case instead of a completed-fit status line.
- coordinate-frame status: QR handoff audit rows report caked observed targets
  only under `fit_observed_caked_deg`. `fit_observed_detector_display_px` is
  reconstructed from detector-native coordinates through the dataset
  native-to-display conversion; if no detector conversion is available, the
  detector-display field stays unavailable instead of reusing caked degrees.

Run:

```bash
python scripts/debug/visualize_new4_qr_fit_coordinates.py \
  --state artifacts/geometry_fit_gui_states/bi2se3.json \
  --background-index 0 \
  --single-step-detector-angle-audit \
  --active-vars gamma,Gamma \
  --report-label Bi2Se3 \
  --max-angle-step-deg 5 \
  --fd-step-deg 0.05 \
  --output-root artifacts/geometry_fit_recovery/latest
```

Outputs:

- `01_single_step_qr_coordinate_audit.json` is the authoritative row-level proof
- `01_single_step_qr_coordinate_audit.csv` mirrors the JSON row table for quick inspection
- `01_single_step_qr_coordinate_audit.png` is diagnostic visual evidence only

The JSON records every row identity, detector display/native coordinate,
caked coordinate, live-projector source, residual delta, validity flag, and
before/after identity check. `saved_sim_refined_caked_used` is always false in
this audit. Rows with inconsistent detector display/native conversion are
counted in `invalid_detector_display_row_count` and excluded from the detector
panel instead of being plotted on mixed coordinate frames.

Each row also records the GUI-drawn caked simulation source through
`gui_drawn_sim_caked_source`. For the current caked-display New4 proof this is
`sim_visual_caked_deg`, and it matches the optimizer source for all fitted QR
rows.

Proof mode is strict by default. `proof_status` is `pass` only when every row
has an evaluated QR fit surface contract, each contract passes, and every GUI
visual simulation QR point and optimizer objective QR point share the same
caked coordinate surface within `1e-6` degrees. Contract failure or a missing
contract status always reports `proof_status="fail"` with
`proof_failure_reason="qr_fit_point_surface_contract_failed"`. Surface-only
divergence without a contract failure reports
`proof_failure_reason="visual_simulation_surface_differs_from_objective_surface"`.
The proof scope is `caked_space_contract`: detector-panel plotting remains
diagnostic evidence, and the JSON reports `detector_panel_status` as
`complete`, `partial`, `not_plotted`, or `not_evaluated`. A caked-space
contract pass can therefore coexist with `plotted_row_count=0` only when the
report and PNG title explicitly mark the detector panel as diagnostic.
The caked panel plots objective base/trial simulation points as square/triangle
markers, and plots divergent GUI visual points separately as diagnostic
diamond/x markers. Use `--allow-visual-objective-surface-divergence` only for
diagnostic artifacts; it can mark the artifact `diagnostic_only` only when no
non-surface gate failed. Identity, row-count, branch/hkl, bounded-step, and
contract failures still leave the final status as `fail`.

Detector panel coordinates are display pixels only. Caked-degree sources such
as `fit_prediction_caked_deg`, `optimizer_simulated_source_two_theta_phi`, and
`dynamic_sim_visual_caked_deg_two_theta_phi` are rejected as detector display
points. Objective detector display points remain valid only when they come from
allowed detector-display fields such as `sim_nominal_detector_display_px` or
`resolved_detector_display_px`.

For `metric=dynamic_angular_point_match` and `objective_space=caked_deg`,
caked-display source rows prefer live caked coordinates in this order:
`sim_visual_caked_deg`, `sim_caked`, `sim_caked_display`, `two_theta_deg` plus
`phi_deg`, then `caked_x` plus `caked_y`. Detector projection remains allowed
only for true detector-frame source rows. If a caked-display row ever reports
`optimizer_source_source="point_only_detector_projection"`, the audit classifies
it as `caked_display_row_reprojected_as_detector_point` and recommends
`prefer_live_sim_visual_caked_for_caked_objective`.

The expected gate for this slice is:

```bash
python -m pytest tests/test_geometry_fitting.py -q
python -m pytest tests/test_gui_geometry_fit_workflow.py -q
python -m pytest tests/test_geometry_fit_manual_fit_space_classification.py -q
python -m ra_sim.dev check
```

The full manual-selection helper file is slow in local desktop runs; use the
targeted manual QR helper slice first when validating this debug mode, then run
the full file when time permits.

Generated files under `artifacts/geometry_fit_recovery/` are local diagnostic
output and are ignored by git unless a future task explicitly asks to version a
specific artifact.

## Weighted-event diffraction status

Current status:

- feature status: active normal runtime path
- optimization status: fixed for duplicate weighted-candidate projection and
  exact-preserving Q-set precompute
- bug status: slow Python raw-candidate enumeration fixed
- error status: covered invariants green in weighted-event regression tests

What changed:

- `process_peaks_parallel(...)` and `process_peaks_parallel_safe(...)` now run the
  weighted-event path through `_process_peaks_parallel_impl(...)`, which calls
  Numba pass-1/pass-2 helpers instead of enumerating raw candidates with Python
  lists and dicts in the hot loop
- off-detector, non-finite, or bilinear-unsupported candidates are rejected
  before they enter `V`, so they cannot affect event selection or image mass
- duplicate sampled ordinals still produce duplicate hit rows, duplicate sampled
  cache rows, and duplicate best-sample event counts; only image deposition may
  aggregate repeated ordinals internally
- branch representatives stay separate from sampled events; fitter-facing
  `get_last_intersection_cache()` returns deterministic zero-intensity ghost
  representatives at beam center, zero divergence, zero beam offset, and the
  default wavelength, while `get_last_intersection_cache_views()` exposes both
  sampled event rows and branch representative rows
- GUI primary-fill reuse stores representative intersection-cache entries per
  contribution key, drops stale entries when raw hit rows are replaced, and
  translates representative detector coordinates across detector-center remaps
- `get_last_process_peaks_weighted_event_stats()` exposes weighted-event debug
  counters and timers, including solve/project/select counts and pass-1/pass-2
  mass totals for test assertions
- the fast weighted-event path stores valid projected candidates during the
  first pass and emits selected events from those stored buffers during the
  second phase, so `_project_weighted_candidate_fast(...)` is not called twice
  for the same candidate in the default path
- memory-bounded fallback keeps the old `_weighted_event_pass2_for_qset(...)`
  projection path available for debugging and oversized samples
- candidate-reuse stats are reported as
  `n_stored_projected_candidates`, `candidate_buffer_capacity_max`, and
  `candidate_buffer_fallback_count`; `n_project_candidate_calls` now counts
  projection calls only, not stored-candidate emission
- the fast serial weighted-event path precomputes unique `(peak_idx, rep_idx)`
  Q sets into flat NumPy tables before pass 1, then both pass 1 and pass 2 use
  integer `qset_id` lookups instead of the previous Python dict cache
- Q-set precompute is intentionally exact-preserving: it does not group by
  `(Gr, Gz)`, does not change `solve_q(...)` inputs, and does not alter
  projection, event selection, image deposition, or hit-table semantics
- Q-set precompute stats are reported as `n_qsets_precomputed`,
  `n_qset_lookup_entries`, `n_qset_reuse_hits`, and `time_qset_index`

Still intentionally disabled for weighted events:

- source-template replay
- clustered beam replacement
- grouped event emission by `(Gr, Gz)`
- sampling from representative/cache rows
- sampled-row or old mosaic-top-rank representative selection for geometric-fit
  cache anchors

## Numba on-disk compilation cache

RA-SIM sets `NUMBA_CACHE_DIR` at package import time for stable startup behavior.
When `NUMBA_CACHE_DIR` is unset, RA-SIM defaults to `~/.cache/ra_sim/numba`.
If a value is already set, RA-SIM leaves it unchanged.

RA-SIM does not force a Numba CPU mode. `NUMBA_CPU_NAME=generic` is not
enabled by default.

To inspect cache activity:

1. `set NUMBA_DEBUG_CACHE=1` before launch (or `export NUMBA_DEBUG_CACHE=1` on
   macOS/Linux)
2. run one simulation (`python -m ra_sim simulate ...` or `ra-sim simulate ...`)
3. watch console output for cache write/hit events from the Numba cache loader

Known limitations:

- first import/first run of a new kernel still compiles once before reuse
- cache hit behavior can vary across hosts, Python versions, and Numba releases
- if the default path is not writable, RA-SIM keeps startup best-effort and may not use on-disk caching

Manual verification recipe:

1. remove `~/.cache/ra_sim/numba` (or platform equivalent configured in `NUMBA_CACHE_DIR`)
2. run one simulation with `NUMBA_DEBUG_CACHE=1`
3. confirm new cache artifacts appear in `NUMBA_CACHE_DIR`
4. rerun same simulation with `NUMBA_DEBUG_CACHE=1` and confirm cache read hits in output

## Developer Tool Caches

When you run RA-SIM development commands from the repository, tool caches stay
under the user cache root instead of the worktree.

- Python bytecode: `~/.cache/ra_sim/dev/pycache`
- `mypy`: `~/.cache/ra_sim/dev/mypy`
- `pytest`: `~/.cache/ra_sim/dev/pytest`
- `ruff`: `~/.cache/ra_sim/dev/ruff`

The tool-specific `mypy`/`pytest`/`ruff` cache dirs apply to both
`python -m ra_sim.dev ...` and direct `pytest`/`mypy`/`ruff` runs from the
repository root. Python bytecode redirection applies when the repo
`sitecustomize.py` is importable, which is guaranteed for `ra_sim.dev` and
`python -m ...` launches from the repository root.

Existing repo-local cache folders are not migrated or removed automatically.
It is safe to delete stale `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`, and
`__pycache__/` directories manually when they are no longer needed.

## Good Debug Hygiene

- Prefer config-based toggles over ad hoc path edits.
- Keep machine-local output paths in ignored local config, not versioned files.
- If you add a new debug artifact, document its toggle and output location.
- When sharing logs or screenshots, avoid leaking local absolute paths or private data.

For the exact key-by-key reference and current defaults, use the canonical doc:

- [Logging, debug, and cache controls](simulation_and_fitting.md#logging-debug-and-cache-controls)
