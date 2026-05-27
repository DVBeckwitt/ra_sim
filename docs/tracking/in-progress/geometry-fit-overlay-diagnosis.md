# Geometry Fit Overlay Diagnosis

Status: visual-semantics fix implemented; caked projection mismatch open
Type: bug investigation
Owner: -
Issue: #249
Priority: p1
Last updated: 2026-05-26

## Summary

This pass first kept the behavioral fix out of scope while gathering evidence.
The visual overlay symptoms have a narrow Fix E implementation, but the caked
projection mismatch remains open:

1. Selected blue squares keep the saved selected simulated display point when
   `sim_native` is explicitly classified as display-space data mislabeled as
   native.
2. Dashed fitted-motion arrows are drawn only for dynamic final predictions.
   Locked, saved, stale, or missing-dynamic endpoints keep their diagnostics but
   no longer render as fitted simulated motion.
3. Final simulation markers from locked, stale, or missing-dynamic endpoints are
   retained for diagnostics but labeled as diagnostic simulation markers rather
   than fitted simulated peaks.

The evidence is already strong for two visual bugs:

- The trace shape is a detector display/native frame leak: selected simulated
  display and native values are identical under a nonzero background display
  rotation, while the expected native point is about 800 px away.
- The fitted dashed arrow can be drawn to a locked saved handoff endpoint, not a
  dynamic post-fit simulation endpoint.

The live GUI A/B run is still not complete: there is no existing automated GUI
driver for the manual q-group selection/pick/fit sequence. The visual fix does
not change QR handoff precedence, caked `gamma,Gamma` route selection, or
optimizer behavior.

## Remaining Mismatch After Fix E

The remaining post-Fix-E failure is not the stale-arrow/blue-square visual
semantics bug. The same detector-native simulated points are projected to
different caked coordinates by the manual trace and geometry-fit paths:

| Branch | Native detector px | Manual trace caked deg | Geometry-fit caked deg | Delta |
| --- | ---: | ---: | ---: | ---: |
| 0 | `(1099, 1924)` | `(39.777, 37.250)` | `(32.744, 132.750)` | `d2theta=-7.033`, `dphi=95.500` |
| 1 | `(1082, 1132)` | `(41.351, -38.750)` | `(38.359, 38.750)` | `d2theta=-2.992`, `dphi=77.500` |

The mismatch is not explained by one simple phi transform across both branches.
The diagnostic regression checks identity, sign flip, `+/-90 deg`,
`+/-180 deg`, and keeps the 2theta deltas visible. No single transform brings
both branches into agreement.

## Projection Call-Site Map

| Path | File/function | Input frame claimed | Projector/callback | Phi convention | Output field | Consumer |
| --- | --- | --- | --- | --- | --- | --- |
| Manual live detector-to-caked trace | `ra_sim/gui/_runtime/runtime_session.py::_emit_detector_to_caked_manual_trace` | saved visual display converted to native detector for sim rows | `_detector_to_caked_manual_trace_display_to_native(...)` then `_detector_to_caked_manual_trace_native_to_caked(...)` | callback-defined GUI phi | `sim_caked_deg` text trace | operator trace and saved-run diagnosis |
| Manual live native-to-caked callback | `ra_sim/gui/_runtime/runtime_session.py::_native_detector_coords_to_live_caked_coords` | simulation/native detector pixels | `detector_pixel_to_caked_bin(_current_live_caked_transform_bundle(), col, row)` | exact-cake GUI phi from bundle bin weighting | callback return | manual trace, manual pick refresh |
| Manual entry caked enrichment | `ra_sim/gui/manual_geometry.py::_geometry_manual_apply_sim_visual_detector_fields` callers around saved/refined entries | detector display plus callback-derived native | `native_detector_coords_to_caked_display_coords(refined_native)` | callback-defined GUI phi | `sim_refined_caked_deg` | manual entry replay, fit handoff rows |
| Geometry-fit point-only projection | `ra_sim/fitting/optimization.py::_project_detector_points_to_fit_space` | `input_frame="native_detector"` when native prediction exists | dataset `fit_space_projector` if available, otherwise `_detector_pixels_to_fit_space(...)` analytic fallback | dataset projector metadata or analytic flat-detector phi | `fit_prediction_caked_deg`, `predicted_caked_deg`, `sim_refined_caked_deg` | optimizer residuals and overlay records |
| Overlay caked final marker | `ra_sim/gui/geometry_overlay.py::build_geometry_fit_overlay_records` | caked fields are already caked; native fields are detector native | chooses `final_prediction_caked_deg`, `dynamic_final_caked_deg`, `sim_visual_caked_deg`, `sim_refined_caked_deg`, `fit_prediction_caked_deg` | no conversion except source precedence | `final_sim_caked_display` | overlay draw and visual-distance summary |
| Visual-distance summary | `ra_sim/gui/geometry_overlay.py::summarize_geometry_fit_overlay_visual_distances` | current overlay display frame | `_resolve_overlay_display_point(...)` | caked when caked fields/native projector exist; detector display fallback otherwise | `initial_distance_median`, `final_distance_median` | progress text and fit log |

First divergence after the 2026-05-26 same-native run: for the same native
detector inputs, runtime live exact-cake, geometry-fit
`_project_detector_points_to_fit_space(...)`, the dataset projector, and the
exact-bundle alias agree. The remaining mismatch is therefore upstream of
`_project_detector_points_to_fit_space(...)`: saved/manual caked fields are
created or preserved with values that are not the live exact-cake projection of
their paired native detector coordinates.

## Saved/Manual Caked Field Provenance

Current source map for fields that can become saved/manual simulated caked
values:

| File/function | Field written or preserved | Input native/display evidence | Projector used | Consumer |
| --- | --- | --- | --- | --- |
| `ra_sim/gui/manual_geometry.py::_geometry_manual_apply_sim_visual_detector_fields` | `sim_visual_caked_deg`, `sim_caked`, `sim_visual_deg` | `sim_visual_detector_display_px` plus `sim_visual_detector_native_px` from display-to-native callback when available | `native_detector_coords_to_caked_display_coords(...)` | saved manual entries and later fit handoff |
| `ra_sim/gui/manual_geometry.py::_geometry_manual_caked_qr_projection_entry` | `sim_refined_caked_deg`, `sim_visual_caked_deg`, `sim_caked_display` | current-view caked row with optional native detector evidence | current-view projection cache row; no live native recomputation in this helper | caked QR projection cache |
| `ra_sim/gui/manual_geometry.py::geometry_manual_refine_qr_sim_peak_detector` | `sim_refined_caked_deg` | refined detector display converted to native when callback exists | `native_detector_coords_to_caked_display_coords(refined_native)` | selected simulated-point display/caked state |
| `ra_sim/gui/manual_geometry.py::geometry_manual_refine_qr_sim_peak_caked` | `sim_refined_caked_deg` | caked simulation image axes; detector native may be backfilled afterward | caked-image peak-center refinement axes | selected simulated-point caked state |
| `ra_sim/gui/geometry_fit.py::build_geometry_manual_fit_dataset` | `sim_refined_caked_deg`, `sim_visual_caked_deg`, `sim_caked_display` | copies selected initial entries and manual provider fields | preserves saved/manual caked values; now logs live recompute deltas under diagnostics | optimizer dataset, overlay records, diagnostic JSON |

Diagnostic-only addition:

- `build_geometry_manual_fit_dataset(...)` now writes
  `manual_caked_recompute_audit` only when
  `RA_SIM_GEOM_WRITE_DIAGNOSTICS=1`.
- Each row records `saved_manual_caked_deg`, `native_detector_px`,
  `recomputed_live_exact_caked_deg`, wrapped caked delta, projector callable
  metadata, background index, saved field source, native field source, and
  frame status.
- The saved value is not replaced. This only proves whether the saved/manual
  caked field agrees with the live exact-cake projection for its paired native
  detector point.
- `same_native_projection_comparison` can now include
  `runtime_live_exact_cake` rows from plain precomputed diagnostic data when the
  runtime callable itself is unavailable in the optimizer context. This closes
  the previous JSON gap without changing projection behavior.

Expected trace-shaped diagnostic rows:

| Branch | Native detector px | Saved/manual caked deg | Recomputed live exact-cake deg | Expected status |
| --- | ---: | ---: | ---: | --- |
| 0 | `(1099, 1924)` | `(39.777, 37.250)` | `(32.744, 132.750)` | `mismatch_saved_manual_caked_vs_live_exact` |
| 1 | `(1082, 1132)` | `(41.351, -38.750)` | `(38.359, 38.750)` | `mismatch_saved_manual_caked_vs_live_exact` |

First divergence function candidate:
`ra_sim/gui/geometry_fit.py::build_geometry_manual_fit_dataset` is the first
currently instrumented handoff where saved/manual caked fields and native
detector points are paired and preserved into the fit dataset. The next live run
should use `manual_caked_recompute_audit` to decide which earlier writer in
`manual_geometry.py` first created the stale value.

## Visual-Distance Source Table

The `initial=1763.93` style visual median is dimensionally suspect in caked
mode. The current resolver can still fall back to detector display fields when
caked display fields and native projection are missing:

| Role | Caked-mode preferred source | Fallback that can contaminate caked distance | Diagnostic status |
| --- | --- | --- | --- |
| initial simulated | `initial_sim_caked_display`, then projected `initial_sim_native` | `initial_sim_display` | `detector_display_used_in_caked_mode` |
| initial background | `initial_bg_caked_display`, then projected `initial_bg_native` | `initial_bg_display` | `detector_display_used_in_caked_mode` |
| final simulated | `final_sim_caked_display`, then projected `final_sim_native` | `final_sim_display` | `detector_display_used_in_caked_mode` |
| final background | `final_bg_caked_display`, then projected `final_bg_native` | `final_bg_display` | `detector_display_used_in_caked_mode` |

Added diagnostic-only coverage:

- `audit_geometry_fit_overlay_visual_distance_inputs(...)` reports the source
  field, point, claimed units, and space status for each visual-distance input.
- `compute_geometry_overlay_frame_diagnostics(...)` now includes
  `caked_visual_detector_display_input_count`. This is metadata only; the
  existing visual-distance calculation is intentionally unchanged in this
  investigation pass.

## Projection Comparison Diagnostics

Added focused diagnostic tests:

- `test_trace_caked_projection_mismatch_is_not_one_simple_phi_transform`
  documents that the branch 0/1 mismatch is not explained by identity, sign
  flip, `+/-90 deg`, or `+/-180 deg` phi transforms.
- `test_manual_and_geometry_fit_caked_projection_paths_are_comparable_for_same_native_point`
  is marked `xfail(strict=True)` with the same-native branch fixtures. It
  records the manual path as
  `_native_detector_coords_to_live_caked_coords:detector_pixel_to_caked_bin`
  and the geometry-fit path as `_project_detector_points_to_fit_space`.
- `test_caked_visual_distance_audit_flags_detector_pixel_fallback_inputs`
  proves the visual-distance resolver can include detector display inputs in
  caked mode and records that via
  `caked_visual_detector_display_input_count`.

Added a same-native projector comparison diagnostic:

- `optimization._compare_same_native_caked_projection_paths(...)` evaluates the
  fixed trace native detector points through the runtime live projector, the
  geometry-fit `_project_detector_points_to_fit_space(...)` path, the direct
  dataset projector, the exact-bundle alias when the dataset projector is an
  `exact_caked_bundle`, and the analytic detector fallback.
- The helper records projector callable identity, fit-space source,
  `cake_bundle_signature`, local-parameter signature, background index,
  `theta_initial`, `gamma`, `Gamma`, phi convention label, raw and wrapped caked
  output, fallback status, fallback reason, and deltas against both runtime live
  exact-cake and geometry-fit outputs.
- The runtime live projector is carried through the GUI dataset spec as
  `diagnostic_runtime_live_caked_projector`, bound to
  `runtime_session._native_detector_coords_to_live_caked_coords` when the live
  GUI path is available. The comparison only runs automatically under
  `RA_SIM_GEOM_WRITE_DIAGNOSTICS=1`.

Focused tests added:

- `test_same_native_projection_comparison_identifies_geometry_fit_projector_path`
  proves the diagnostic can show geometry fit matching the exact dataset
  projector while diverging from the runtime live exact-cake result.
- `test_same_native_projection_comparison_explains_analytic_geometry_fit_fallback`
  proves the diagnostic records `analytic_detector_fit_space` and
  `missing_exact_caked_bundle` when no exact dataset projector is available.

## Live Same-Native Projection Run, 2026-05-26

Live runtime reproduction was run from the current diagnostic `main` worktree
with `RA_SIM_GEOM_WRITE_DIAGNOSTICS=1`, restored from the local `new4` saved
state, filtered to `('q_group', 'primary', 1, 10)`, and with only
`gamma,Gamma` fit variables enabled.

Artifacts:

- Text log:
  `C:\Users\Kenpo\.cache\ra_sim\logs\geometry_fit_log_20260526_082330.txt`
- Diagnostic JSON:
  `C:\Users\Kenpo\.cache\ra_sim\logs\geometry_fit_overlay_diagnostic_20260526_082330.json`
- Runtime-live projector query:
  `temp/geometry_fit_runtime_live_projection_20260526_082549/runtime_live_projection_rows.json`

The JSON `same_native_projection_comparison` section contains the geometry-fit,
dataset projector, exact-bundle alias, and analytic fallback rows, but its
`runtime_live_exact_cake` rows are marked unavailable with
`fallback_reason=runtime_live_projector_unavailable`. A no-code live-runtime
query of `runtime_session._native_detector_coords_to_live_caked_coords(...)`
was therefore used to fill the runtime-live column for the same four native
detector inputs.

Same-native comparison:

| point_role | native_detector_px | runtime_live_exact_caked_deg | geometry_fit_caked_deg | dataset_projector_caked_deg | exact_bundle_alias_caked_deg | analytic_fallback_caked_deg | geometry_fit_matches_which_path | fallback_used | fallback_reason | projector_signature | caked_bundle_signature | caked_generation | background_index | theta_initial | gamma | Gamma | phi_convention | delta_geometry_fit_vs_runtime_live | delta_dataset_vs_runtime_live | delta_analytic_vs_runtime_live |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| branch_0_sim_native | `(1099.000, 1924.000)` | `(32.744, 132.750)` | `(32.744, 132.750)` | `(32.744, 132.750)` | `(32.744, 132.750)` | `(32.749, -132.770)` | runtime_live_exact, dataset_projector, exact_bundle_alias | False |  | `7fc57c0b7c0b8ac9de5cb354b99316c73578c115` | `95e395fd4e152a985d34f760a4baa85d9c71fe6e` |  | 0 | 5.000 | 0.000 | 45.000 | geometry_fit_projected_phi | `0.000 deg (d2t=0.000, dphi=0.000)` | `0.000 deg (d2t=0.000, dphi=0.000)` | `94.480 deg (d2t=0.005, dphi=94.480)` |
| branch_1_sim_native | `(1082.000, 1132.000)` | `(38.359, 38.750)` | `(38.359, 38.750)` | `(38.359, 38.750)` | `(38.359, 38.750)` | `(38.402, -38.628)` | runtime_live_exact, dataset_projector, exact_bundle_alias | False |  | `7fc57c0b7c0b8ac9de5cb354b99316c73578c115` | `95e395fd4e152a985d34f760a4baa85d9c71fe6e` |  | 0 | 5.000 | 0.000 | 45.000 | geometry_fit_projected_phi | `0.000 deg (d2t=0.000, dphi=0.000)` | `0.000 deg (d2t=0.000, dphi=0.000)` | `77.378 deg (d2t=0.043, dphi=-77.378)` |
| branch_0_observed_refined_native | `(1083.270, 1915.182)` | `(33.063, 130.754)` | `(33.063, 130.754)` | `(33.063, 130.754)` | `(33.063, 130.754)` | `(33.065, -130.757)` | runtime_live_exact, dataset_projector, exact_bundle_alias | False |  | `7fc57c0b7c0b8ac9de5cb354b99316c73578c115` | `95e395fd4e152a985d34f760a4baa85d9c71fe6e` |  | 0 | 5.000 | 0.000 | 45.000 | geometry_fit_projected_phi | `0.000 deg (d2t=0.000, dphi=0.000)` | `0.000 deg (d2t=0.000, dphi=0.000)` | `98.489 deg (d2t=0.002, dphi=98.489)` |
| branch_1_observed_refined_native | `(1083.734, 1152.380)` | `(37.566, 39.750)` | `(37.566, 39.750)` | `(37.566, 39.750)` | `(37.566, 39.750)` | `(37.601, -39.756)` | runtime_live_exact, dataset_projector, exact_bundle_alias | False |  | `7fc57c0b7c0b8ac9de5cb354b99316c73578c115` | `95e395fd4e152a985d34f760a4baa85d9c71fe6e` |  | 0 | 5.000 | 0.000 | 45.000 | geometry_fit_projected_phi | `0.000 deg (d2t=0.000, dphi=0.000)` | `0.000 deg (d2t=0.000, dphi=0.000)` | `79.506 deg (d2t=0.035, dphi=-79.506)` |

Decision from the same-native projector comparison:

- Geometry-fit does not match the analytic fallback.
- Geometry-fit matches the dataset projector, exact-bundle alias, and live
  runtime exact-cake projection for all four requested native detector points.
- The first divergence is therefore not
  `optimization._project_detector_points_to_fit_space(...)` for these native
  inputs. It is earlier in the value lineage: manual/saved caked fields are not
  the live exact-cake projection of their reported native detector points.
- In the same live run, the manual trace emitted
  `actual_source=saved_visual_coordinates`; for branch 0 it reported
  `sim_visual_detector_native_px=(1085.520,1921.262)` with
  `sim_caked_deg=(40.177,36.250)`, while the geometry-fit audit projected the
  same saved native point to `sim_refined_caked_deg=(33.186,131.631)`.
  Branch 1 repeats the pattern:
  `sim_visual_detector_native_px=(1085.325,1169.728)` with manual
  `sim_caked_deg=(40.230,-36.275)`, while geometry-fit projected it to
  `sim_refined_caked_deg=(36.874,40.750)`.

Visual-distance source table for the same run:

| pair | mode | initial_observed_point | initial_sim_point | fitted_observed_point | fitted_sim_point | source field for each point | units claimed | distance | included_in_median | reason_if_detector_pixels_entered_caked_mode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `<none>` | caked_2d | `<not recorded>` | `<not recorded>` | `<not recorded>` | `<not recorded>` | `overlay_record_audit` empty; `draw_audit` empty; `frame_diag.overlay_record_count=0` | deg | n/a | no | no visual-distance rows were emitted; `caked_visual_detector_display_input_count=0`, so this run does not prove detector pixels entered the visual-distance median |

The visual-distance diagnostic did not reproduce the earlier
`initial=1763.93` caked-mode median in this run because no overlay records were
produced for the visual-distance audit. The log did still report a separate
holistic caked check of `initial_rmse=777.996`, `final_rmse=777.996`,
`delta=0`; that is not the visual-distance input audit table and should be
tracked separately.

Recommended next fix PR scope:

1. Correct manual saved/sim caked lineage: when a manual trace or saved manual
   row reports both native detector coordinates and caked coordinates, validate
   or recompute the caked value from the live exact-cake native projector. If
   the saved caked value disagrees with live projection, mark it stale or reject
   it instead of comparing it as the same coordinate space.
2. Keep visual-distance caked input filtering as a separate small follow-up:
   the current run did not emit visual-distance rows, but the unit audit still
   proves detector display fallbacks can contaminate caked-mode visual medians
   when caked/native fields are missing.

## Caked Projection Decision Table

| Hypothesis | Evidence supporting | Evidence against | First divergence | Fix candidate | Confidence |
| --- | --- | --- | --- | --- | --- |
| H1: manual caked trace callback expects display pixels but receives native pixels. | Manual trace converts saved visual display to native before calling native-to-caked. | Callback name and docstring explicitly say native detector coords. | Not first divergence. | None yet. | Low |
| H2: manual caked trace callback expects native pixels but receives display pixels. | Earlier traces had display/native aliasing. | Current manual trace uses `_background_display_to_native_detector_coords` before native-to-caked. | Before callback only if display-to-native is wrong. | Verify display-to-native per background. | Medium-low |
| H3: manual trace and geometry-fit use different caked bundle generations or background indices. | Earlier source inspection showed the paths were different enough to require proof. | The 2026-05-26 same-native run showed runtime live exact-cake, geometry-fit, dataset projector, and exact-bundle alias agree for the requested native points. | Not the first divergence for those native inputs. | Keep signature comparison diagnostics; no bundle-unification fix until a live mismatch proves metadata divergence. | Low |
| H4: manual trace and geometry-fit use different phi conventions or wrapping. | Branch phi deltas are large. | Simple sign/offset transforms do not explain both branches; 2theta also differs. | Not a pure phi transform. | Only after projector/bundle parity is proven. | Medium |
| H5: visual-distance calculation mixes detector pixels into caked mode independent of projection mismatch. | Source audit proves caked mode can fall back to `*_display` detector pixels when caked/native fields are missing. | Final median may still use valid caked fields for some roles. | `_resolve_overlay_display_point(...)` fallback branch. | Filter or flag detector-display inputs in caked visual summaries. | High |
| H6: locked QR handoff is still stale, but projection paths are otherwise consistent. | Previous diagnostics showed locked saved endpoints. | Same-native projection paths are consistent for fixed native inputs, so stale handoff is not needed to explain saved/manual caked-field mismatch. | Separate from the caked-field handoff. | Separate QR handoff investigation. | Medium |
| H7: `gamma,Gamma` objective remains insensitive, but overlay projection mismatch is separate. | Prior sensitivity work incomplete. | The same native-point projection mismatch does not require optimizer motion. | Projection path. | Separate objective sensitivity investigation. | Medium |

Recommended next fix PR scope: split into two small fixes only after a live
ledger confirms metadata:

1. Fix visual-distance caked input filtering so detector display pixels are
   excluded or explicitly reported instead of included in caked-degree medians.
2. Validate or recompute saved/manual simulated caked fields from verified
   native detector coordinates before those values are used in geometry-fit
   overlay or caked visual-distance comparisons.

## Reproduction Steps

Target live scenario:

1. Load the saved GUI state that produced the uploaded trace.
2. Select group `('q_group', 'primary', 1, 10)`.
3. Place the two background points corresponding to branches 0 and 1.
4. Run geometry fit with `vars=gamma,Gamma`.
5. Save screenshot, text log, and
   `~/.cache/ra_sim/logs/geometry_fit_overlay_diagnostic_<timestamp>.json`.

The live GUI matrix was not completed in this inspection pass. The exact
uploaded trace file is not present in the checkout, and no existing automated
GUI driver performs that manual sequence end to end. I ran the non-GUI
diagnostic smoke, saved-state coordinate probes, and the shared headless
fit-geometry A/B surrogate listed below.

## Trace File Used

Primary reference: the uploaded trace described in the task text.

Repo-local supplemental artifacts:

- `artifacts/geometry_fit_gui_states/new4.json`
- local 15-pair New4 saved-state snapshot from the user data root; local
  variants are intentionally ignored rather than tracked.
- `temp/geometry_overlay_inspection/new4_visual_backend_baseline/coordinate_transform_diagnosis.json`
- `temp/geometry_overlay_inspection/new4_visual_backend_optimizer_request/coordinate_transform_diagnosis.json`
- `temp/geometry_overlay_inspection/new4_local15_visual_backend_optimizer_request/coordinate_transform_diagnosis.json`
- `temp/geometry_overlay_live_ab/baseline/run.log`
- `temp/geometry_overlay_live_ab/baseline_repo_new4/02_full_fit_initial_vs_final_qr_overlay.json`
- `temp/geometry_overlay_live_ab/disable_early_handoff_repo_new4/02_full_fit_initial_vs_final_qr_overlay.json`
- `C:\Users\Kenpo\.cache\ra_sim\logs\geometry_fit_overlay_diagnostic_codex_inspection_smoke_20260525.json`

The supplemental saved states are not exact reproductions of the uploaded trace:
the local 15-pair state contains the same `q_group` and branch identities but
different coordinates.

## Observed Failure

The uploaded trace records:

- Branch 0 selected simulated display/native alias:
  `(1079.897, 1098.761)`.
- Branch 0 refined native point: `(1098, 1922)`.
- Branch 1 selected simulated display/native alias:
  `(1862.136, 1077.417)`.
- Branch 1 refined native point: `(1077, 1142)`.
- `fit_prediction_source=locked_manual_qr:saved_detector_display_to_native:saved_display_px`.
- `fit_prediction_is_dynamic=no`.
- `initial_rmse=102.175`, `final_rmse=102.175`, `delta=0`.
- `stale_final_sim=True`.

Under `native_shape=(3000,3000)` and `background_display_rotate_k=-1`,
`display_point_to_native_for_rotation(1077, 1098, shape, -1)` returns
`(1098, 1922)`. A native value equal to display `(1079.897, 1098.761)` is
therefore not in background-native detector coordinates.

## Append-Site Coverage

| File | Line | Function | Entry variable | Used by manual Qr/Qz fit? | Includes `sim_display` | Includes `sim_native` | Includes `sim_native_source` | Frame audit fields | Provider overwrite/provenance |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| `ra_sim/gui/manual_geometry.py` | 13549 | `geometry_manual_session_initial_pairs_display` | `entry` | No, in-progress picker overlay only | Yes | No | No | No | No |
| `ra_sim/gui/manual_geometry.py` | 13687 | `build_geometry_manual_initial_pairs_display` | background-reference payload | No, saved overlay/background-reference replay | No | No | No | No | No |
| `ra_sim/gui/manual_geometry.py` | 14395 | `build_geometry_manual_initial_pairs_display` | background-reference payload | No, saved overlay/background-reference replay | No | No | No | No | No |
| `ra_sim/gui/manual_geometry.py` | 14542 | `build_geometry_manual_initial_pairs_display` | `initial_entry` | No, caked unresolved overlay replay | No, marks unresolved | No | No | No | No |
| `ra_sim/gui/manual_geometry.py` | 14564 | `build_geometry_manual_initial_pairs_display` | `initial_entry` | No, ghost unresolved overlay replay | No, marks unresolved | No | No | No | No |
| `ra_sim/gui/manual_geometry.py` | 14569 | `build_geometry_manual_initial_pairs_display` | `initial_entry` | No, detector replay unresolved overlay branch | No, marks unresolved | No | No | No | No |
| `ra_sim/gui/manual_geometry.py` | 14594 | `build_geometry_manual_initial_pairs_display` | `initial_entry` | No, detector replay unresolved overlay branch | No, marks unresolved | No | No | No | No |
| `ra_sim/gui/manual_geometry.py` | 14610 | `build_geometry_manual_initial_pairs_display` | `initial_entry` | No, saved overlay replay | Yes when resolved | No | No | No | No |
| `ra_sim/gui/geometry_fit.py` | 4942 | `build_geometry_fit_saved_state_point_provider_dataset` | `initial_entry` | Not the live manual button path; provider-only saved-state diagnostics | Yes for display-frame providers | Yes for native-frame providers | No | No | Basic provider frame/source fields |
| `ra_sim/gui/geometry_fit.py` | 16330 | `build_geometry_manual_fit_dataset` | `initial_entry` | Yes, active manual Qr/Qz geometry-fit dataset path | Yes | Yes | Yes | Yes, `initial_pair_construction_audit` | Yes |

Coverage conclusion:

- The active GUI/manual fit dataset append site is instrumented.
- The manual overlay replay builders are intentionally not optimizer input, but
  they remain a diagnostic gap if a future report says the blue square moves
  before the fit dataset exists.
- The provider-only saved-state dataset path can still write `sim_native`
  without frame audit. It is not the active live button path, but it is a
  diagnostic gap for saved-state-only replay tooling.

## Diagnostic Interfaces

- `RA_SIM_GEOM_WRITE_DIAGNOSTICS=1`: enables JSON bundle writing from the GUI
  postprocess path without changing route selection, handoff precedence,
  sim-native rebuild, fitted arrows, or fit parameters.
- `scripts/diagnostics/summarize_geometry_fit_overlay_diagnostics.py`: reads one
  or more `geometry_fit_overlay_diagnostic_*.json` bundles and prints the
  requested markdown or CSV table.

Smoke bundle:

```text
C:\Users\Kenpo\.cache\ra_sim\logs\geometry_fit_overlay_diagnostic_codex_inspection_smoke_20260525.json
```

Smoke summary:

```text
temp/geometry_overlay_inspection/diagnostic_bundle_summary.md
```

The smoke row confirms the summarizer flags a trace-shaped bundle as
`mismatch_display_labeled_native`, `recomputed_from_initial_sim_native`,
`locked_saved_prediction`, `blue_square_moved_from_raw_display=True`, and
`stale_arrow_drawn=True`.

## A/B Experiment Table

Live GUI rows remain pending because no automated driver exists for the manual
sequence. Strict audit and combined strict+handoff were not run because the
first live baseline could not be collected.

| Run | Environment | Artifact | Result |
| --- | --- | --- | --- |
| A baseline | `RA_SIM_GEOM_WRITE_DIAGNOSTICS=1` | Not produced from live GUI | Pending live GUI driver/manual run |
| C disable sim-native rebuild | `RA_SIM_GEOM_WRITE_DIAGNOSTICS=1`, `RA_SIM_GEOM_DISABLE_SIM_NATIVE_REBUILD=1` | Not produced from live GUI | Pending live GUI driver/manual run |
| E suppress stale fitted arrows | `RA_SIM_GEOM_WRITE_DIAGNOSTICS=1`, `RA_SIM_GEOM_DISABLE_FITTED_ARROW_FOR_STALE_ENDPOINT=1` | Not produced from live GUI | Pending live GUI driver/manual run |
| D disable early handoff | `RA_SIM_GEOM_WRITE_DIAGNOSTICS=1`, `RA_SIM_QR_DISABLE_EARLY_HANDOFF=1` | Not produced from live GUI | Pending live GUI driver/manual run |
| F disable auto caked `gamma,Gamma` route | `RA_SIM_GEOM_WRITE_DIAGNOSTICS=1`, `RA_SIM_DISABLE_AUTO_CAKED_ROUTE_FOR_GAMMA_GAMMA=1` | Not produced from live GUI | Pending live GUI driver/manual run |

Supplemental non-GUI runs:

| Run | Command | Artifact | Result |
| --- | --- | --- | --- |
| Saved-state visual/backend baseline | `python scripts/debug/diagnose_new4_visual_backend_coordinates.py --state artifacts/geometry_fit_gui_states/new4.json --background-index 0 --output-dir temp/geometry_overlay_inspection/new4_visual_backend_baseline` | `temp/geometry_overlay_inspection/new4_visual_backend_baseline/coordinate_transform_diagnosis.json` | `visual_backend_parity_ok`; visual surfaces agree before optimizer request capture. |
| Saved-state optimizer-request capture | Same state with `--include-optimizer-request` | `temp/geometry_overlay_inspection/new4_visual_backend_optimizer_request/coordinate_transform_diagnosis.json` | `frame_mismatch_detected`; first mismatch is `optimizer_request.measured_peaks`, where simulated endpoint frame changes from display to `caked_2theta_phi`. |
| Local 15-pair optimizer-request capture | Same diagnostic on an untracked local 15-pair New4 saved-state snapshot with `--include-optimizer-request` | `temp/geometry_overlay_inspection/new4_local15_visual_backend_optimizer_request/coordinate_transform_diagnosis.json` | Optimizer request capture blocked before solve by `locked_qr_caked_projection_frame_mismatch`, including `hkl=(-1,0,10)` branch 1. |
| Q-group sensitivity probe | `python scripts/debug/run_q_group_peak_sensitivity.py --state artifacts/geometry_fit_gui_states/new4.json --group-key q_group,primary,1,10 --params gamma,Gamma --outdir temp/geometry_overlay_inspection/q_group_primary_1_10_sensitivity` | No output artifacts | Aborted with `Baseline evaluation returned no observations`; useful as an abort reason, not sensitivity evidence. |

Shared headless `fit-geometry` A/B surrogate:

| Run | State | Environment | Artifact | Result |
| --- | --- | --- | --- | --- |
| User-local baseline | `C:\Users\Kenpo\.local\share\ra_sim\new4.json`, only `bg0:pair3,bg0:pair4` included | `RA_SIM_GEOM_WRITE_DIAGNOSTICS=1` | `temp/geometry_overlay_live_ab/baseline/run.log` | Blocked before optimization with `locked_qr_caked_projection_frame_mismatch`, branch 1 delta `1.308175 deg`; no GUI diagnostic bundle emitted. |
| User-local no early handoff | Same | `RA_SIM_GEOM_WRITE_DIAGNOSTICS=1`, `RA_SIM_QR_DISABLE_EARLY_HANDOFF=1` | `temp/geometry_overlay_live_ab/disable_early_handoff_user_local_new4/run.log` | Same pre-optimization block; early-handoff disable does not bypass this guard. |
| User-local no caked route | Same | `RA_SIM_GEOM_WRITE_DIAGNOSTICS=1`, `RA_SIM_DISABLE_AUTO_CAKED_ROUTE_FOR_GAMMA_GAMMA=1` | `temp/geometry_overlay_live_ab/disable_caked_gamma_gamma_route_user_local_new4/run.log` | Same pre-optimization block; route disable does not bypass this guard. |
| Repo baseline | `artifacts/geometry_fit_gui_states/new4.json`, only `bg0:pair3,bg0:pair4` included | `RA_SIM_GEOM_WRITE_DIAGNOSTICS=1` | `temp/geometry_overlay_live_ab/baseline_repo_new4/02_full_fit_initial_vs_final_qr_overlay.json`, `.png` | Rejected; `full_fit_success=False`, `gamma/Gamma` unchanged, `raw_angular_rms_deg=167.759554`, `visual_objective_surface_mismatch_count=2`, objective source is `handoff_native_anchor_projection`. |
| Repo disable sim-native rebuild | Same | `RA_SIM_GEOM_WRITE_DIAGNOSTICS=1`, `RA_SIM_GEOM_DISABLE_SIM_NATIVE_REBUILD=1` | `temp/geometry_overlay_live_ab/disable_sim_native_rebuild_repo_new4/02_full_fit_initial_vs_final_qr_overlay.json`, `.png` | Same as repo baseline; this flag is GUI overlay draw/rebuild only in the headless surrogate. |
| Repo suppress stale arrows | Same | `RA_SIM_GEOM_WRITE_DIAGNOSTICS=1`, `RA_SIM_GEOM_DISABLE_FITTED_ARROW_FOR_STALE_ENDPOINT=1` | `temp/geometry_overlay_live_ab/suppress_stale_arrows_repo_new4/02_full_fit_initial_vs_final_qr_overlay.json`, `.png` | Same as repo baseline; this flag is GUI draw-only in the headless surrogate. |
| Repo disable early handoff | Same | `RA_SIM_GEOM_WRITE_DIAGNOSTICS=1`, `RA_SIM_QR_DISABLE_EARLY_HANDOFF=1` | `temp/geometry_overlay_live_ab/disable_early_handoff_repo_new4/02_full_fit_initial_vs_final_qr_overlay.json`, `.png` | Accepted; `full_fit_success=True`, `geometry_updated=True`, `gamma/Gamma` still unchanged, `raw_angular_rms_deg=3.086964`, `source_authority_mismatch_count=0`, one row resolves through `hit_table_resolved` and one through `ghost_only`. |
| Repo disable caked gamma/Gamma route | Same | `RA_SIM_GEOM_WRITE_DIAGNOSTICS=1`, `RA_SIM_DISABLE_AUTO_CAKED_ROUTE_FOR_GAMMA_GAMMA=1` | `temp/geometry_overlay_live_ab/disable_caked_gamma_gamma_route_repo_new4/02_full_fit_initial_vs_final_qr_overlay.json`, `.png` | Same as repo baseline; the headless command still reports `acceptance_metric_space=caked_deg`. |

## Data Lineage

Uploaded trace branch 0:

| Stage | Value |
| --- | --- |
| Manual picked simulated display | `(1079.897, 1098.761)` |
| Saved/nominal simulated native | `(1079.897, 1098.761)` |
| Expected native from background display rotation | About `(1098.761, 1921.103)` |
| Refined simulated native | `(1098, 1922)` |
| Frame status | `mismatch_display_labeled_native` |
| Overlay consequence if rebuilt from bad native | Blue square can move about 823 px |

Uploaded trace branch 1:

| Stage | Value |
| --- | --- |
| Manual picked simulated display | `(1862.136, 1077.417)` |
| Saved/nominal simulated native | `(1862.136, 1077.417)` |
| Expected native from background display rotation | About `(1077.417, 1138.864)` |
| Refined simulated native | `(1077, 1142)` |
| Frame status | `mismatch_display_labeled_native` |
| Overlay consequence if rebuilt from bad native | Blue square can move about 788 px |

Supplemental local 15-pair `q_group=('q_group','primary',1,10)`:

| Branch | Saved display | Saved refined native | Visual surface status |
| --- | --- | --- | --- |
| 0 | `(1077.738, 1085.520)` | `(1085.520, 1921.262)` | Initial/manual/measured visual surfaces agree in display frame. |
| 1 | `(1829.272, 1085.325)` | `(1085.325, 1169.728)` | Initial/manual/measured visual surfaces agree in display frame, but optimizer capture is blocked by locked caked projection mismatch. |

Lineage conclusion:

- In the uploaded trace, bad `sim_native` exists before overlay drawing because
  the selected simulated display/native values are already identical.
- `build_geometry_fit_overlay_records(...)` is a proven jump source when it
  chooses `recomputed_from_initial_sim_native`; the test and smoke bundle both
  document the current failure mode.
- The local saved states also show that visual/manual surfaces can be coherent
  before optimizer-request construction, which keeps the optimizer handoff path
  under suspicion for stale or wrong-frame prediction data.

## QR Resolver Outcome

Live GUI resolver bundles are still pending. Current evidence:

- Uploaded trace: resolver source is
  `locked_manual_qr:saved_detector_display_to_native:saved_display_px` and
  `fit_prediction_is_dynamic=no`.
- The smoke bundle summarizer correctly reports `resolver_paths=handoff` and
  `handoff_accepted_count=1` when fed a locked saved endpoint.
- User-local headless baseline is blocked before optimization by
  `locked_qr_caked_projection_frame_mismatch`; disabling early handoff does not
  change that preflight block.
- Repo `new4.json` headless baseline resolves both q-group rows through
  `handoff_native_anchor_projection` with `objective_cache_mode=handoff_native_anchor`,
  rejects the fit, and reports `visual_objective_surface_mismatch_count=2`.
- Repo `new4.json` with `RA_SIM_QR_DISABLE_EARLY_HANDOFF=1` no longer uses the
  handoff-native objective source for both rows. One row resolves through
  `hit_table_resolved`, one through `ghost_only`; the fit is accepted and
  `source_authority_mismatch_count` drops from `2` to `0`.

Interpretation:

- Early handoff is not the only blocker for the user-local 15-pair state because
  the caked projection mismatch guard fires first.
- Early handoff is a real root-cause contributor in the repo `new4.json`
  surrogate: turning it off changes the objective source, shrinks angular RMS
  from `167.759554 deg` to `3.086964 deg`, and changes rejection to acceptance.
- The accepted no-handoff surrogate still leaves `gamma=0` and `Gamma=0`, with
  `optimizer_nfev=0`; it proves source precedence matters, but does not prove a
  physically meaningful gamma/Gamma optimization.

## Objective And Projector Sensitivity

The live `gamma,Gamma` objective/projector sensitivity bundle is still pending.

Current evidence:

- The uploaded trace has unchanged RMSE and `stale_final_sim=True`, which is
  consistent with a stale prediction path or an insensitive objective.
- The q-group sensitivity helper could not evaluate
  `q_group,primary,1,10` from the repo `new4.json` state and aborted with
  `Baseline evaluation returned no observations`.
- The shared headless repo baseline and no-handoff run both report
  `optimizer_nfev=0`, so they are source-resolution checks, not true solver
  sensitivity checks.
- `RA_SIM_DISABLE_AUTO_CAKED_ROUTE_FOR_GAMMA_GAMMA=1` did not change the repo
  headless surrogate; it still reports `acceptance_metric_space=caked_deg`.

Interpretation:

- Do not change the caked `gamma,Gamma` route based on the headless surrogate
  alone.
- The no-handoff run proves that stale/handoff source selection can dominate the
  result before any gamma/Gamma numerical sensitivity is exercised.
- The next fix PR should not claim to solve gamma/Gamma sensitivity unless a
  live bundle or a focused unit-level dynamic objective test reports nonzero
  `gamma_*` and `Gamma_*` deltas.

## UI Semantics

Current legend semantics:

```text
blue squares = selected simulated points
green circles = fitted simulated peaks
dashed arrows = initial->fitted sim shifts
```

Classification:

- If the blue square is rebuilt from `sim_native` and moves away from the saved
  selected display point, the drawn object violates the legend.
- If the dashed arrow endpoint is `locked_saved_prediction` or
  `stale_prediction`, the arrow violates fitted-motion semantics unless it is
  hidden or relabeled as non-dynamic diagnostic state.

## Hypothesis Decision Table

| Hypothesis | Evidence supporting | Evidence against | A/B run that tests it | Result | Confidence | Fix candidate |
| --- | --- | --- | --- | --- | --- | --- |
| H1: `sim_native` is display-space data mislabeled as native. | Uploaded trace has identical display/native values under nonzero rotation; frame-audit test and smoke bundle classify this as mismatch. | Supplemental local 15-pair saved state stores refined native fields correctly for its branch coordinates. | B strict frame audit | Pending live; old trace strongly supports | High for uploaded trace | Validate frame before storing or before trusting `sim_native`; keep display authoritative for selected blue square. |
| H2: overlay rebuild from `sim_native` moves blue squares. | Overlay-builder test and smoke bundle show `recomputed_from_initial_sim_native` with >800 px raw-vs-rebuilt delta. | Headless surrogate cannot draw GUI overlay records, so C does not test the visual square. | C disable sim-native rebuild | Pending live; mechanism already reproduced by unit/smoke bundle | High for mechanism, medium for live visual | Coordinate-frame storage/rebuild fix. |
| H3: dashed arrows use locked/stale prediction endpoints. | Uploaded trace has locked saved prediction source, non-dynamic flag, stale final sim, and no RMSE change; draw audit classifies this as locked saved. | Headless stale-arrow flag is draw-only and does not change recovery artifacts. | E suppress stale fitted arrows | Pending live screenshot; smoke bundle supports semantics | High | Stale fitted-arrow source correction or suppression. |
| H4: early QR handoff prevents dynamic prediction. | Uploaded trace accepted locked saved endpoint; repo surrogate baseline uses `handoff_native_anchor_projection` and fails, while no-handoff uses non-handoff sources and accepts. | User-local state still blocks before optimization even with no-handoff; no live GUI bundle yet. | D disable early handoff | Positive in repo headless surrogate; blocked in user-local state | High that precedence can be causal, medium for uploaded trace | Change resolver precedence only with focused regression around available dynamic rows. |
| H5: caked `gamma,Gamma` route is insensitive. | Uploaded trace RMSE unchanged; headless runs report `optimizer_nfev=0` and gamma/Gamma unchanged. | No true objective/projector sensitivity bundle yet; disabling caked route did not alter the repo headless surrogate. | F disable auto caked route and objective/projector probes | Inconclusive | Low to medium | Disable/rebuild caked `gamma,Gamma` route only with zero-delta evidence. |
| H6: solver is underconstrained but overlay is otherwise correct. | Two manual points for two angular variables can be weakly constrained; no-handoff accepted without moving gamma/Gamma. | Display/native alias, stale endpoint, and handoff source mismatch are sufficient to explain visual bug independent of conditioning. | Baseline plus objective sensitivity | Not proven; visual bug evidence points elsewhere | Low for blue-square movement, medium for fit quality | Separate conditioning follow-up only if objective is sensitive and overlay/handoff are fixed. |

## Exact Functions Involved

- `ra_sim.gui.geometry_fit.build_geometry_manual_fit_dataset`
- `ra_sim.gui.geometry_fit.write_geometry_fit_overlay_diagnostic_bundle`
- `ra_sim.gui.geometry_fit._geometry_fit_write_diagnostics_enabled`
- `ra_sim.gui.geometry_overlay.audit_detector_point_frames`
- `ra_sim.gui.geometry_overlay.build_geometry_fit_overlay_records`
- `ra_sim.gui.overlays.draw_geometry_fit_overlay`
- `ra_sim.fitting.optimization._resolve_qr_fit_prediction_from_trial_params`
- `ra_sim.fitting.optimization._dynamic_objective_param_sensitivity_summary`
- `ra_sim.fitting.optimization._probe_detector_projector_sensitivity`
- `ra_sim.gui._runtime.runtime_session._build_geometry_fit_async_job`
- `scripts.diagnostics.summarize_geometry_fit_overlay_diagnostics.summarize_diagnostic_bundle`

## Tests Added

Earlier diagnostic tests:

- Coordinate transform trace points.
- Frame-audit mismatch classification.
- Overlay-builder bad-`sim_native` rebuild delta and rebuild-disable flag.
- Draw-audit stale locked arrow classification and stale-arrow-disable flag.
- Draw-audit caked native projection provenance.
- QR resolver early-handoff A/B flag.
- Objective sensitivity diagnostic fields.
- Projector sensitivity probe for the two trace native detector points.
- Initial-pair construction provenance audit.

This inspection pass:

- `test_geometry_fit_overlay_diagnostic_summarizer_handles_missing_keys`
- `test_geometry_fit_write_diagnostics_env_flag`

Visual-semantics fix tests:

- `test_build_geometry_fit_overlay_records_audits_bad_initial_sim_native_jump`
  now asserts the selected blue square stays at the raw selected display point
  when frame audit reports `mismatch_display_labeled_native`.
- `test_build_geometry_fit_overlay_records_accepts_valid_initial_sim_native_rebuild`
  proves trusted background-native points can still rebuild the current overlay
  display point.
- `test_build_geometry_fit_overlay_records_rejects_source_agnostic_display_native_alias`
  proves display/native numeric aliasing is rejected for saved/overlay
  provenance even when the source label does not contain `display`.
- `test_draw_geometry_fit_overlay_suppresses_locked_saved_arrow_endpoint`
  proves locked saved endpoints do not draw fitted-motion arrows by default.
- `test_draw_geometry_fit_overlay_keeps_dynamic_fitted_arrow` proves dynamic
  final predictions still draw the dashed fitted-motion arrow.
- `test_draw_geometry_fit_overlay_renders_markers_labels_and_residual_arrow`
  now documents that endpoints without dynamic provenance are labeled
  `diagnostic sim`, not `fit sim`.
- `test_geometry_fit_overlay_diagnostic_summarizer_handles_missing_keys` now
  treats explicit `suppressed_stale_arrow=True` or `stale_arrow_drawn=False` as
  authoritative, while still supporting the older diagnostic flag field.

## Proposed Fix Options

Fix A: coordinate-frame storage/rebuild only.

- Files touched: `ra_sim/gui/geometry_fit.py`,
  `ra_sim/gui/geometry_overlay.py`, nearest overlay tests.
- Behavioral change: selected initial simulated display remains authoritative
  for blue squares; `sim_native` is only used when frame audit says it is valid
  for the current display/caked reconstruction.
- Tests required: trace-point regression where display/native alias no longer
  moves the blue square; strict audit still catches bad native.
- Risk: may expose saved states that depended on old native rebuild behavior.
- Rollback path: restore old rebuild branch or leave diagnostic flag to compare.

Fix E: combined minimal visual semantics fix.

- Files touched: Fix A files plus `ra_sim/gui/overlays.py` and possibly
  `ra_sim/fitting/optimization.py` if live no-handoff proves dynamic rows exist.
- Behavioral change: apply Fix A and prevent locked/non-dynamic saved endpoints
  from being drawn as fitted simulated motion.
- Tests required: stale endpoint draw test, live bundle with arrow semantics, and
  no-handoff resolver test if precedence changes.
- Risk: hiding a stale arrow may make a failed fit look less informative unless
  logs and summary text still expose the stale endpoint.
- Rollback path: re-enable arrow drawing with
  `RA_SIM_GEOM_DISABLE_FITTED_ARROW_FOR_STALE_ENDPOINT=0` while preserving frame
  audit evidence.

Implemented Fix E as a combined minimal visual-semantics fix:

1. Coordinate-frame storage/rebuild guard for selected blue squares.
2. Stale/non-dynamic fitted-arrow semantics guard.

The blue-square guard is intentionally narrow: it rejects native rebuild only
for explicit unsafe alias audit statuses. Existing saved overlay replay behavior
that intentionally prefers trusted native points over stale cached display
coordinates is preserved. The source-agnostic alias rule applies to saved or
overlay provenance labels such as `overlay_source_sim_native_xy`, so the known
numeric display/native alias is not dependent on the literal word `display`.

Keep QR handoff precedence as a separate follow-up unless a focused regression
can prove dynamic rows are available and early handoff alone selects the stale
source in the uploaded trace. Do not include the caked `gamma,Gamma` route in
the next fix PR unless a live projector/objective bundle shows near-zero deltas.

## Remaining Uncertainty

- The live A/B GUI matrix still needs screenshots, text logs, and real JSON
  bundles. The current run set used headless PNGs and recovery JSONs instead.
- `RA_SIM_QR_DISABLE_EARLY_HANDOFF=1` clearly changes the repo `new4.json`
  surrogate, but it has not yet proven whether dynamic rows are available in
  the uploaded trace.
- `RA_SIM_DISABLE_AUTO_CAKED_ROUTE_FOR_GAMMA_GAMMA=1` did not change the repo
  surrogate and has not proven whether the selected caked route is responsible
  for unchanged RMSE.
- The uploaded trace should be linked or copied into a local artifact path before
  the fix PR so the final regression can quote an exact file.
