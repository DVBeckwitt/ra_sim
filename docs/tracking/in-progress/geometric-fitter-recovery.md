# Geometric Fitter Recovery

Status: in-progress
Type: investigation
Owner:
Issue: [#249](https://github.com/DVBeckwitt/ra_sim/issues/249)
Priority: p1
Last updated: 2026-05-27

## Summary

2026-05-27 cleanup/status closeout: removed dead legacy fitter paths and
generated payloads after the locked-Qr readiness/dynamic-authority fixes. The
cleanup deleted the legacy GUI auto-match fit handler, the legacy mosaic-shape
fitter implementation, unused private helpers, stale fast/exact optics GUI
fields, the fast-only CTR attenuation helper/test, obsolete one-shot debug
scripts, tracked generated geometry ladder reports, and a package-local zip.
Generated geometry ladder output and package-local zip bundles are now ignored.
Bug/error status: no new fitter behavior change; this is maintenance cleanup
around already validated exact-only readiness and dynamic-authority diagnostics.
Feature/API/migration status: no new CLI flag, GUI control, config key,
saved-state schema, artifact schema, dependency, or public API. Deprecation
status: non-exact optics compatibility and the deleted debug scripts are not
supported paths. Validation status: `python -m compileall ra_sim tests scripts`,
`python -m pytest tests/test_gui_runtime_import_safe.py -ra`, focused
exact-optics, mosaic/cache, geometry authority/caked, background profile/rod,
testing-index suites, `python -m ra_sim.dev test-fast`,
`python -m ra_sim.dev check`, `python -m pytest --collect-only -q`, and
`git diff --check` passed. Known residual: the dynamic reanchor
`test_dynamic_reanchor_recovers_stale_peak_index_by_provider_branch` failure
also reproduces on clean `HEAD`, so it is tracked as pre-existing and not part
of this cleanup. Shipping handoff: cleanup committed as `023f31e2`; the branch
is ready for review after the documented validation gates, with rollback by
reverting that single cleanup commit if a removed internal path proves active.

2026-05-27 exact-only locked Qr/Qz readiness gate update: non-exact optics
modes are no longer supported for the fitter path, so the remaining operator
proof is exact-only. Run the exact-only GUI sequence. If exact readiness
passes, proceed to the dynamic authority mismatch. If exact readiness fails,
classify missing/nonfinite/storage using row-key diagnostics. The latest
exact-only manual run is the current reference for this failure path:
`expected_rows=2`, `projected_rows=2`, `finite_rows=2`,
`projection_ready=true`, `storage_required_for_fit=false`,
`storage_timeout_fatal=false`, and `optics=exact`; readiness is considered
passed for this path unless a future exact-only run fails readiness. Runtime
behavior, optics defaults, optimizer behavior, QR handoff precedence, caked
route selection, overlay drawing, acceptance metrics, solver logic, and
gamma/Gamma logic remain unchanged.

2026-05-27 locked Qr/Qz dynamic-authority integration closeout: final dynamic
acceptance now fails closed as `locked_qr_dynamic_authority_mismatch` when
explicit source or caked-coordinate authority drift is present, and the
GUI-consumed `fit_geometry_parameters()` result preserves `success=False`,
`status=-15`, and the mismatch message. Row-level diagnostics carry preflight,
pair-audit, and final dynamic caked-coordinate fields through the point-match
summary and rejection text, so hidden large caked RMS values no longer fall
back to `manual_outliers_or_physical_bad_fit` or manual repick guidance.
Bug/error status: fixed in source and covered by fit-level regression.
Feature, migration, deprecation, API, solver, optimizer, route-selection,
overlay, acceptance-metric, optics-default, gamma/Gamma, CI workflow, version,
and dependency status: unchanged. Shipping status: targeted geometry
authority/dynamic/caked tests, locked-Qr readiness tests, runtime import-safe
tests, `python -m ra_sim.dev check`, and `git diff --check` pass; the remaining
operator proof is one exact-only GUI sequence.

2026-05-27 locked Qr/Qz projection-readiness diagnostics fix: the worker now
passes stored source rows into the locked-Qr exact-caked readiness gate so
preflight can distinguish source-cache loss from projected-row loss. Missing
readiness failures include compact row keys (`pair_id`, hkl, branch, table, row,
peak, source, and first missing stage), event payloads distinguish source rows,
projected rows, exact projection payload, storage status, timeout fatality, and
optics mode, and caked view storage timeouts no longer appear as opaque
zero-projected-row failures when projected rows are already present. Follow-up
closeout: runtime failure text now chooses its primary row-key section from
`failure_reason`, so nonfinite failures show nonfinite row keys, missing
failures show missing row keys, and mixed failures preserve separately labeled
nonfinite and missing sections. Bug/error status: fixed for the remaining
locked-Qr readiness diagnostic mismatch. Feature, migration, deprecation, API,
solver, optimizer, route-selection, overlay, acceptance-metric, optics-default,
and gamma/Gamma status: unchanged.

2026-05-25 detector-mode locked Qr/Qz readiness projection fix: the live
`8f68fe5c` trace no longer froze, but the worker still failed after caked
storage with `expected_rows=2 projected_rows=0` because the locked-Qr readiness
gate was validating detector projected rows instead of the exact-caked
selected-row projection that had just become available. The worker now refreshes
locked-Qr readiness from stored selected detector rows projected through the
exact caked payload once that payload exists, without changing detector
fit-space, caked-objective flags, GUI-thread handoff behavior, public job keys,
or user controls. Bug/error status: fixed in source and automated regression
tests for the reported post-freeze preflight failure; the gate still fails
closed before dataset build when exact caked projection rows are unavailable or
nonfinite.

2026-05-25 detector-origin locked Qr/Qz GUI-thread handoff fix: the live
`7491a20c` trace showed Fit Geometry still froze before any worker
`source_cache_*` event, after job-build preflight had synchronously requested
caked-view preparation for detector-origin locked Qr/Qz picks. The runtime now
keeps locked source identity as projection-readiness provenance only: detector
manual fit-space stays detector during async job build, no GUI-thread
`ensure_geometry_fit_caked_view()` or caked payload load runs for detector
locked rows, and the async worker still owns the selected-row locked-Qr
projection readiness gate before dataset build. Bug/error status: fixed for
the reported unresponsive `preflight: collecting geometry-fit datasets` /
`Computing simulation in background...` handoff signature. Feature status: no
new GUI control, CLI flag, config key, saved-state schema, artifact schema,
dependency, public API, CI workflow, deprecation, or migration. Validation:
focused job-build regression, full runtime import-safe suite, manual fit-space
classification suite, `python -m compileall ra_sim tests`, `git diff --check`,
and `python -m ra_sim.dev check` pass.

2026-05-25 geometry-fit preflight timeout closeout: plain fresh-simulation
source-cache rebuilds now share the bounded timeout behavior used by targeted
manual-geometry preflight. When the non-targeted fallback stalls, the async
worker emits the existing source-cache timeout/late/failed stage sequence and
returns a preflight failure before optimizer preparation instead of leaving the
Fit Geometry worker running indefinitely after source-cache rebuild starts.
Follow-up cleanup centralized the derived fresh-simulation stage label and
timeout status without changing emitted event names, diagnostics, exception
text, or public interfaces. Bug/error status: fixed for downstream worker
source-cache rebuild stalls, but not sufficient for the later-discovered
GUI-thread caked-handoff freeze fixed above. Feature status: no new GUI control, CLI
flag, config key, saved-state schema, artifact schema, dependency, public API,
or CI workflow. Migration/deprecation status: none required. CI/shipping
status: focused plain/targeted timeout regressions, locked-Qr projection
regressions, `python -m compileall ra_sim tests`, and `python -m ra_sim.dev
check` pass. Remaining manual shipping smoke is the existing operator check:
restart the GUI, import a Bi saved state, generate once, click Fit Geometry, and
confirm no first-fit timeout or exact-caked projector error. Rollback is a
normal git revert of the timeout/refactor/tests/docs commits.

2026-05-22 settled Qr/Qz square redraw status: settled GUI overlay refresh now
replays durable fitted overlay records only and forces view-bound
`initial_pairs_display` Qr/Qz square markers through the current manual-pair
renderer. This closes the bug where changing simulation parameters moved the
central/ghost beam but the blue Qr-set square stayed at remembered display
coordinates. Bug/error status: fixed with a Matplotlib artist-level regression
that proves the drawn square uses the current simulated ghost coordinate rather
than the stale remembered pixel. Migration/deprecation status: none required;
no saved-state, CLI, config, public API, dependency, or artifact schema changed.
CI/shipping status: focused settled-overlay and ghost-refresh regressions plus
`python -m ra_sim.dev check` pass. Rollback is a normal git revert of the
runtime redraw filter, regression test, changelog/debug-doc entry, and this
tracking note.

2026-05-22 post-review shipping status: reviewed the Qr/Qz ghost refresh,
Advanced/Debug physics-toggle handoff, and Qr-rod editor payload
simplification for correctness, code size, security, performance,
migration/deprecation impact, and test coverage. Bug/error status remains fixed
for stale saved ghost-square refresh and duplicate Qr-rod edit-parameter
payload handling. Feature status: integer-Bragg primary-source and
refraction-disabled vacuum-`n2` toggles are documented runtime-only diagnostics;
no saved-state, config, CLI, artifact schema, dependency, or public API changed.
Migration and deprecation status: none required. CI/shipping status:
`python -m ra_sim.dev check`, the focused Qr-rod editor subset, the full
background peak-fit notebook regression file, and `git diff --check` pass.
Rollback is a normal git revert.

2026-05-22 locked Qr/Qz projection-contract diagnostics: locked-Qr handoff
audit rows now include `locked_qr_caked_projection_contract`, a compact
diagnostic payload that records the exact projector id/kind, caked bundle
generation, background index, `theta_initial`, `gamma`, `Gamma`, phi
convention, input frame, native detector point, and projected caked point used
for the fit handoff. This makes the existing projection-frame preflight block
auditable across manual/source trace, handoff/objective rows, final residual
summary, and overlay records instead of relying only on coordinate values.
Bug/error status: fixed for the missing diagnostic contract rung; mismatches
still fail at `locked_qr_caked_projection_frame_mismatch` and matching frames
now carry enough provenance to prove they are same projector/same frame.
Feature status: no new GUI control, CLI flag, config key, saved-state schema,
artifact schema, dependency, public API, CI workflow, deprecation, or migration.
Validation: focused projection-frame and overlay regressions, locked-Qr
geometry-fitting/handoff subsets, runtime projection-readiness subset, and
`python -m ra_sim.dev check` pass. Shipping status: ready as a normal bug-fix
diagnostic slice; rollback is a normal git revert of the audit-contract helper,
regression test, and this tracking update.

2026-05-22 locked Qr/Qz projection-frame contract slice: locked-Qr handoff
audit rows now compare source-trace simulated caked coordinates against the
exact fit-space projection for the same simulated native detector point. If the
same native point maps to different caked `(2theta, phi)` coordinates across
manual/source trace and fit handoff, preflight fails before optimization with
`locked_qr_caked_projection_frame_mismatch` instead of allowing the dynamic
objective to solve across mixed projection frames. Trace matching accepts
`source_reflection_index` and `source_table_index` identity shapes so projected
live rows are not skipped because of equivalent source-index naming. Bug/error
status: fixed for the projection-frame mismatch rung; accepted caked fits now
require coherent source-trace and handoff projection authority before solve.
CI status: focused locked-Qr GUI workflow, live-row handoff, geometry-fitting
tests, and `python -m ra_sim.dev check` pass. Migration/deprecation status: no
saved-state, CLI, config, dependency, public API, or artifact-schema migration
is required; this is a fail-closed validation and diagnostic change to the
existing locked-Qr route. Shipping status: normal git revert is the rollback
path, and logs should show the preflight block rather than a later
`locked_qr_dynamic_authority_mismatch` when frames disagree.

2026-05-22 locked Qr/Qz canonical-payload guard: detector-origin locked Qr/Qz
preflight now treats selected-row exact caked projections as the mandatory
artifact for fitting. Readiness validation uses `projected_rows` only, records
`projected_rows_len` and `stored_rows_len` for diagnostics, and fails before
dataset build when expected selected rows are not projected. Stored detector
rows are diagnostic only and no longer satisfy the projection gate. The dynamic
locked-Qr prediction resolver now tries the authoritative handoff-native
prediction anchor before hit-table/source-row rediscovery and fails closed if
that handoff projection cannot be made, so nominal caked fields cannot replace
the preflight payload. Bug/error status: fixed for the
`expected_rows=4 projected_rows=0` fail-open path and for stale hit-table
authority overriding clean handoff anchors. CI status: focused locked-Qr
runtime, GUI workflow, handoff, and geometry-fitting tests pass, and
`python -m ra_sim.dev check` passes. Migration/deprecation status: no
saved-state, CLI, config, public API, dependency, or artifact-schema migration
is required; rollback is a normal git revert.

2026-05-22 locked Qr/Qz projection-readiness gate: detector-origin explicit
fixed-source locked Qr/Qz fits now fail before dataset build when selected-row
exact caked projections are missing or nonfinite. Full caked image storage can
still time out or defer when row projections are complete, and caked-origin
locked-Qr baselines remain allowed to proceed from their saved caked anchors.
A detector-origin row-level projection miss no longer proceeds into
`manual_caked_route_check` or the dynamic angular optimizer. Projection
readiness diagnostics also flag collapsed selected-row projections as
`locked_qr_projection_degenerate` so repeated caked values from distinct
detector points are visible before fitting without being triggered by unrelated
projected rows. Feature status: implemented for the worker preflight route and
covered by runtime-session and GUI readiness regressions. Migration/deprecation
status: no saved-state, CLI, config, public API, dependency, or artifact-schema
migration is required.

2026-05-22 locked Qr/Qz caked-origin classifier slice: caked-view locked Qr/Qz
fits keep the existing accepted handoff/identity baseline behavior, and branch
source-pairing diagnostics now ignore `caked_simulation_image` refinements as
detector/native branch proof unless a same-frame detector authority is explicit.
The handoff audit marks those caked-image refinements as
`sim_refined_detector_authority=diagnostic_caked_image` and
`sim_refined_detector_same_frame=false`, so downstream classifiers do not turn a
valid caked-source baseline into `branch_source_pairing_mismatch`. Explicit
same-frame detector proof remains valid branch evidence, and handoff audit text
now prints the authority and same-frame fields for review. Feature status:
implemented for the caked-origin locked-Qr route and covered by focused
optimization and GUI handoff regressions for both diagnostic and explicit-proof
paths. Migration/deprecation status: no saved-state, CLI, config, public API,
dependency, or artifact-schema migration is required.

2026-05-22 locked Qr/Qz detector/caked mapping slice: visual probe records now
report the probed image source, image shape, image extent, axes limits, marker
pixel index, image peak index, and search window so detector-marker/image-raster
mismatches can be distinguished from artist drift. Geometry overlay records now
prefer explicit `final_prediction_detector_*` and `final_prediction_caked_deg`
fields over handoff prediction anchors. The locked-Qr dynamic resolver uses
the saved handoff-native anchor before hit-table/source-row rediscovery, and
successful dynamic detector projections emit
`final_prediction_*` payload fields for the overlay. Detector-origin locked-Qr
caked fits now declare when same-frame detector-pixel acceptance is required,
and GUI rejection text fails closed if that required metric is missing. Feature
status: implemented for the detector/caked drift diagnostic path and covered by
targeted overlay, dynamic prediction, visual-probe, and GUI acceptance tests.
Migration/deprecation status: no saved-state, CLI, config, or artifact schema
migration is required.

2026-05-22 shipping review cleanup: same-frame detector acceptance now reads
explicit final locked-Qr detector predictions before handoff prediction anchors,
so acceptance and overlay agree on the fitted marker source. Visual-probe logs
now distinguish simulation and background raster probes by source label, which
prevents marker/raster drift reports from being ambiguous. CI status: targeted
geometry tests and `python -m ra_sim.dev check` are required before commit.
Migration/deprecation status: no compatibility migration or deprecation is
needed; these are diagnostic and acceptance-source fixes for the existing route.

2026-05-21 locked Qr/Qz pipeline-validation slice: row-level caked projection
readiness now counts projected live rows whose locked identity is carried only
in nested canonical provenance, so `source_cache_project_rows_ready rows=4`
cannot later degrade to `locked_qr_projection_readiness projected_rows=0`.
The two-group dynamic fixture now uses the current `[-1,0,5]`/`[-1,0,10]`
live-shape values. The optimizer records a locked-Qr identity baseline before
least-squares and, when that baseline is already within acceptance, a failed or
worse solver candidate cannot replace it with a manual-outlier rejection. Stale
nominal caked fields remain diagnostics when exact/refined projection authority
is proven. Feature status: implemented for the current two-group locked-Qr
route and covered by targeted readiness, handoff, dynamic objective, and public
solver tests. Migration/deprecation status: no saved-state, CLI, config, or
artifact migration is required.

2026-05-21 locked Qr/Qz branch-line slice: saved manual caked Qr/Qz fits now
enable the existing q-group line residual only for locked groups that prove both
`source_branch_index=0` and `source_branch_index=1` for the same `q_group_key`.
The dynamic angular objective keeps the two endpoint residuals and appends the
branch-segment angle residual; line-offset weight stays disabled for this route.
GUI and headless saved-state routes share the same runtime-budget helper, and
completion/status output reports finite branch-line angle RMS when present.
Single-branch or branch-ambiguous groups do not create fake line constraints.
Dead headless-local budget helpers were removed after the shared helper replaced
them. Feature status: implemented and covered by targeted GUI/headless tests.
Migration/deprecation status: no saved-state, CLI, config, or artifact migration
is required; rollback is a normal git revert.

2026-05-21 locked Qr/Qz fit-speed slice: geometry-fit trial source rows now
prefer zero-intensity required-pair ghost rows in caked/Qr fit space. When
analytic ghost completion is enough, the trial builder skips source-row rebuild
and reports `objective_cache_mode=ghost_only` with
`objective_process_peaks_called=False`. When a rebuild is still needed, the
trial route reports `objective_cache_mode=targeted_required_pairs`, keeps the
required-branch filter, and no longer falls back to a broad fresh simulation if
targeted ghost filtering is unavailable. GUI and headless paths both keep
required-branch filtering for trial source-row rebuilds.

New4 Mode A dynamic/refined Qr fitting is now guarded at the actual optimizer
prediction path. For saved state `C:\Users\Kenpo\.local\share\ra_sim\new4.json`
and background index `0`, all 14 first-image paired Qr branches resolve to
locked dynamic identities, produce 28/28 caked residual components, and compute
residuals as `refined_sim_caked - observed_caked` in weighted caked degrees.
Baseline dynamic predictions now reproduce the saved refined sim centers before
solver start; a poisoned or drifting x0 anchor blocks optimization with
`optimizer_start_blocked_reason=dynamic_baseline_anchor_mismatch`. The refined
center diagnostic now proves both observed caked centers and simulated refined
caked centers are recomputed under changed trial geometry, but the simulated
peak refinement is integer caked-bin argmax only. The current Qr-only and full
dynamic/refined objectives are classified `refinement_bin_limited`: theta, phi,
and total norms are unchanged over `nfev=7`, no parameter step is accepted, and
full GUI/baseline convergence is still not claimed here.

Point-provider parity is fixed for the manual-geometry handoff layer, and the
next validation step is a bounded `new4` optimizer ladder. The manual Qr picker
saved/refined pair, the geometry-fit point provider pair, and the actual
dataset handoff row must agree before any optimizer entrypoint runs.

This closes the current point-provider bug/error scope: stale source locators
are diagnostic when the saved picker assignment still resolves semantically,
picker-owned saved/refined simulated points overwrite live/caked prefill, and
`new4` reports 7/7 provider pairs without launching the optimizer.

The Qr/Qz branch-seed bug/error scope is closed for the picker boundary:
raw-cache preview, manual toggle, manual refresh/view-change, and manual place
setup now retain one mosaic-top simulated seed per normalized branch for each
real Qr/Qz group. The lower-level collapse helper still keeps its legacy
per-branch default, and whole-group collapse remains explicit-only for fitting
callers that request it.

The caked-mode Qr/Qz detector-return bug/error scope is also closed: manual
picks made in caked `2theta,phi` space now convert back to detector view through
the same detector-display projection path used by simulation markers, including
stale-session refresh and refined simulated seed redraw. The adjacent
cross-view selection regression is closed too: detector-mode Qr/Qz hit-testing
uses the visible detector marker coordinates first, rejects stale caked
active-view display fields using explicit display-frame metadata before the
legacy numeric alias fallback, and then falls back to detector provenance, so
picking works after detector-to-caked and caked-to-detector view changes.

The detector-view Qr/Qz simulation-frame bug/error scope is now closed:
simulation hit-table rows with `native_col/native_row` are treated as
simulation-native detector pixels and projected with
`native_sim_to_display_coords`, while the background detector display adapter is
reserved for rows explicitly marked as background/native-detector. Guard tests
lock caked selection and caked-to-detector conversion behavior so this fix does
not reopen those known-good paths.

The manual caked geometry-fit drift bug/error scope is now closed through
Rung 7. Caked/Qr-Qz manual picks stay on the exact caked fit-space path,
require exact fit-space rows, reject fallback/analytic-detector rows, and the
import-safe preflight boundary now fails closed before dataset build.
Headless Rung 6 still seeds later combined candidates from accepted earlier
candidates plus accepted Rung 5 z/zb evidence, and the final Rung 7 feature
chain passed `dynamic_reanchor`, `discrete_modes`, `seed_multistart`,
`full_beam_polish`, and `identifiability_features` with exact-caked evidence
preserved in the finalizer.

2026-05-08 two-rotation Bi2Se3/Bi2Te3 headless slice: headless
`fit-geometry` now supplies the shared geometry-fit dataset builder with the
same per-background projector keyword contract used by GUI/runtime callbacks:
`mode_override` can force caked projection and `strict_caked_projection=False`
can return an empty projection instead of raising. This fixes the runtime error
`manual caked geometry fit requires a per-background projector that accepts
mode_override and strict_caked_projection` without adding a new public flag,
schema field, dependency, CI workflow, or migration path.

Bug/error status: fixed for the requested headless path. Constrained direct
fits were run for the user-root Bi2Se3 and Bi2Te3 saved states with
`--active-vars gamma,Gamma --seed-policy direct`, using only the two detector
rotation variables and no peak-position refinement. Bi2Se3 matched 82/82 fixed
pairs with zero missing pairs and zero branch mismatches; Bi2Te3 matched 84/84
fixed pairs with zero missing pairs and zero branch mismatches. Generated
output states were written under `C:\Users\Kenpo\.local\share\ra_sim\` as
`Bi2Se3_gamma_Gamma_fit.json` and `Bi2Te3_gamma_Gamma_fit.json`. Feature
status: this is an enabling bug fix for an existing CLI workflow, not a new GUI
feature. Shipping status: targeted CLI geometry-fit tests, compile, Ruff, and
diff hygiene pass; full `python -m ra_sim.dev check` remains blocked by
pre-existing formatting drift in `ra_sim/gui/_runtime/runtime_session.py`.
Rollback is a normal git revert; no saved-state cleanup or deprecation
migration is required.

2026-05-10 repeatable two-rotation baseline gate: the existing
`scripts/debug/run_geometry_fit_quality_baseline.py` runner now accepts
`--active-vars` and forwards it to headless `fit-geometry`, so the Bi saved
states can be validated with
`--active-vars gamma,Gamma --seed-policy direct` through the same report/gate
path used by other quality baselines. A real headless run wrote reports under
`C:\Users\Kenpo\.local\share\ra_sim\fit_quality_baseline\codex_gamma_Gamma_20260510_000000`.
Bi2Se3 matched 82/82 fixed pairs with zero missing pairs and zero branch
mismatches, reducing direct RMS from 34.5307 px to 15.701942 px. Bi2Te3
matched 84/84 fixed pairs with zero missing pairs and zero branch mismatches,
reducing direct RMS from 36.8629 px to 36.661839 px. Both runs preserved
exact/point-only caked fit-space provenance and passed the saved-state gate.
Feature status: complete for the existing debug runner, with default baseline
behavior unchanged when `--active-vars` is omitted. Bug/error status: fixed for
the repeatable two-detector-rotation quality gate. CI/automation status:
existing `python -m ra_sim.dev check` and integration tests remain the merge
gates; the real Bi material baseline stays an opt-in local validation because
it depends on user-root saved states and takes a long wall-clock run.
Deprecation/migration status: no deprecated entrypoint, saved-state schema, or
artifact schema changes; no migration path is required. Shipping status: ready
as an additive debug-script option. Rollback is a normal git revert.

2026-05-10 parameter-matrix rerun: the existing headless baseline runner
compared `gamma`, `Gamma`, `gamma,Gamma`, `gamma,Gamma,corto_detector`,
`gamma,Gamma,theta_initial`, and `corto_detector,theta_initial` on the user-root
Bi2Se3 and Bi2Te3 saved states. Multi-variable artifacts are under
`%USERPROFILE%\.local\share\ra_sim\fit_quality_baseline\param_matrix_20260510_002`.
Single-variable artifacts use non-colliding follow-up roots because Windows
case-insensitive paths made `gamma` and `Gamma` unsafe as sibling folder names:
`param_matrix_20260510_003_case_safe\single_gamma` for `gamma` and
`param_matrix_20260510_004_upper_gamma\single_upper_Gamma` for `Gamma`. Every
corrected combination preserved exact/point-only caked fit-space provenance,
passed the saved-state gate, kept zero missing fixed pairs and zero branch
mismatches, and reduced direct RMS for both materials.

`gamma,Gamma` remains the smallest two-rotation proof: Bi2Se3 34.5307 ->
15.701942 px and Bi2Te3 36.8629 -> 36.661839 px. The best tested residual
tradeoff was `gamma,Gamma,corto_detector`: Bi2Se3 34.5307 -> 10.922948 px and
Bi2Te3 36.8629 -> 35.581 px. `Gamma` alone helped Bi2Se3 more than `gamma`
alone but barely moved Bi2Te3; `corto_detector,theta_initial` helped Bi2Te3
but not Bi2Se3; `gamma,Gamma,theta_initial` regressed relative to
`gamma,Gamma`. `theta_initial` is normalized to `theta_offset` in reports.
Feature status: validated through the existing CLI/debug-runner interface only;
no GUI, saved state, config, or artifact schema change is required. Bug/error
status: the docs-index inventory guard is fixed by adding missing tracked
validation entrypoints to `docs/testing-and-validation.md`. CI/automation
status: targeted geometry/docs tests and `python -m ra_sim.dev check` pass.
Deprecation/migration status: no deprecated entrypoint, migration, saved-state
schema, config schema, or artifact schema change. Shipping status: docs-only
handoff is ready; rollback is a normal git revert.

2026-05-12 GUI gamma/Gamma overlay closeout: the saved-manual caked GUI path
for Bi2Se3 and Bi2Te3 now fits only detector tilts `gamma` and `Gamma` through
the same exact caked projector used by import/redraw, instead of a separate
detector-tilt caked projection shim. Final overlay diagnostics compare the
drawn fitted simulation markers against the saved background triangle picks in
the active display frame, so the reported distance matches what the operator
sees in the GUI. The follow-up review cleanup also shares the same caked/display
point resolver between drawing, visual-distance summaries, and frame diagnostics
instead of keeping a duplicate local parser in the diagnostic helper.

Bug/error status: fixed for the visually wrong fitted-simulation overlay points
in the Bi2Se3/Bi2Te3 saved GUI `gamma,Gamma` runs. Real GUI validation showed
improved drawn overlay medians for both saved states, and the screenshots were
kept as local validation artifacts only. Feature status: no new GUI control,
CLI flag, public API, config key, dependency, saved-state schema, or artifact
schema. CI/automation status: local quality gates are green for compile,
focused overlay/runtime tests, full import-safe GUI runtime tests, and
`python -m ra_sim.dev check`; no CI workflow changed. Deprecation/migration
status: the removed detector-tilt projection shim was internal dead code with
no public migration path; saved states and artifacts require no cleanup.
Shipping status: safe as a normal bug-fix/refactor slice. Rollback is a normal
git revert of the overlay/projector fix and follow-up resolver cleanup.

2026-05-12 fit-sim marker source closeout: the final rendered caked simulation
image was already moving with the refined `gamma,Gamma` parameters, but the
green `fit sim` circles could still draw from stale legacy caked aliases rather
than the current point-only fitted prediction. The overlay record builder now
resolves current detector-display, detector-native, and caked fitted prediction
sources before stale fallback aliases. It may use
`fit_prediction_detector_display_px` as a caked `(2theta, phi)` point only when
the objective metadata proves the row is caked, degree-based, or point-only; a
normal detector-display pixel is not accepted as a caked angle.

Bug/error status: fixed for the Bi2Se3 saved GUI caked point-only overlay where
the visual green markers disagreed with the actual simulated spots. GUI runtime
evidence was collected through real Tk/TkAgg drawing, not a headless plot:
`fit_sim_overlay_current_tree_20260511_204233.json` reported `status == "ok"`,
25/25 green markers sourced from `fit_prediction_detector_display_px`, maximum
drawn-marker-to-record delta `0.0`, and maximum selected-point-to-fit-prediction
delta `0.0`. Focused regressions cover current prediction precedence, the
caked point-only fallback, and the negative guard that prevents detector pixels
from being treated as caked angles. Feature status: no new GUI control, public
API, CLI flag, config key, saved-state schema, or artifact schema.
CI/automation status: local focused overlay suites and `python -m ra_sim.dev
check` are green; no CI workflow changed. Deprecation/migration status: no
user migration or compatibility shim is required. Shipping status: ready as a
normal bug fix with rollback by git revert.

2026-05-12 holistic visual-residual closeout: the GUI fit path now queues a
pre-redraw full-image residual baseline and flushes detector/caked holistic
residual diagnostics after the fitted simulation redraw settles. The caked
overlay builder also records the distance between each green `fit sim` marker
and the rendered fitted caked simulation point, so stale overlay markers are
visible in logs instead of only in screenshots. The follow-up review patch
precomputes the initial residual metrics before redraw and keeps only the
background image snapshot, avoiding retention of a second full detector
simulation array while waiting for the final redraw.

Bug/error status: fixed for the reported mismatch where the final simulation
image moved with refined `gamma,Gamma` but the green `fit sim` circles could
stay tied to stale caked sources. Feature status: internal GUI diagnostics and
overlay-source correction only; no new GUI control, CLI flag, config key,
saved-state schema, artifact schema, dependency, or public API. CI/automation
status: focused overlay/runtime regressions and `python -m ra_sim.dev check`
are the local quality gates; no CI workflow changed. Deprecation/migration
status: no user migration, compatibility shim, or deprecated interface.
Shipping status: ready as a normal bug-fix diagnostic slice; rollback is a git
revert of the overlay/residual diagnostic patch.

2026-05-12 visual-probe diagnostic update: the existing GUI overlay draw path
now records the actual Matplotlib green `fit sim` marker artist coordinates and
compares them with the strongest pixel in the currently visible simulation
image artist. This targets the still-observed failure mode where the fitted
simulation raster moves correctly but the green fit-sim circles remain tied to
stale initial/cached positions, which the earlier cached-distance and holistic
diagnostics could miss.

Bug/error status: diagnostic coverage improved, not claimed as a final fitter
fix. New log lines include `visual_probe_artist_to_image_peak_med`,
`visual_probe_artist_to_image_peak_max`, per-marker `visual_probe ...` rows, and
`visual_probe_warning=fit_sim_marker_image_mismatch` when the drawn marker is
more than two display units from the nearest visible local image peak. Feature
status: no GUI control, public API, CLI flag, config key, dependency,
saved-state schema, or artifact schema. CI/automation status: focused pure
geometry, overlay, runtime, and geometry-fit formatter tests cover the
diagnostic path; full real-GUI reproduction remains the next validation step.
Deprecation/migration status: no user migration or deprecated interface.
Shipping status: safe as a diagnostic-only slice; rollback is a normal git
revert of the visual-probe patch.

2026-05-12 visual-probe review closeout: the GUI geometry-fit acceptance gate
now fails closed by default. The saved-manual caked angular exception is only
available when an actual headless entrypoint sets the private
`_headless_geometry_fit_runtime` runtime flag; a config or saved-state
`runtime_context=headless` string no longer enables the detector-pixel
threshold bypass. Direct rejection-reason helpers also default to GUI-safe
threshold enforcement, so future callers must opt in explicitly for true
headless caked angular acceptance.

Bug/error status: fixed for the review finding where GUI/runtime config could
re-enable the headless caked angular acceptance path and hide absurd detector
pixel residuals. Feature status: no public API, GUI control, CLI flag, config
key, saved-state schema, artifact schema, dependency, or CI workflow change.
CI/automation status: targeted acceptance-gate regression tests and Python
compile checks pass locally; `python -m ra_sim.dev check` remains the release
gate. Deprecation/migration status: no migration or compatibility shim is
required because the removed `runtime_context` fallback was internal and
unsafe. Shipping status: ready as a normal diagnostic/hardening bug-fix slice;
rollback is a normal git revert.

2026-05-12 coordinate-lineage diagnostic update: rejected and accepted GUI
geometry-fit runs now log a bounded `Coordinate lineage:` section when handoff
audit rows exist. The section joins the cached/refined handoff point, the
fit-dataset prediction, the first optimizer residual-eval snapshots, and the
final point-match diagnostic by background/pair/q-group/HKL/branch/table/row/peak
identity, with wrapped-phi deltas and a `first_divergence_stage` summary. The
draw-time green `fit sim` visual-probe rows now carry the same identity fields,
so GUI screenshots, visible simulation-image probes, cached caked coordinates,
and actual optimizer residual inputs can be compared row-for-row.

Bug/error status: diagnostic coverage improved for the still-observed mismatch
where simulation rasters move but green `fit sim` markers or cached caked
positions may not. This slice does not claim the fitter is fixed; it is designed
to expose whether the divergence occurs in the handoff cache, residual evaluator,
final point matcher, or overlay draw. Feature status: internal GUI diagnostics
only; no public API, GUI control, CLI flag, config key, dependency, saved-state
schema, or artifact schema. CI/automation status: focused helper, runtime apply,
solver-request, and overlay probe tests pass locally. Deprecation/migration
status: no user migration or compatibility shim. Shipping status: safe as a
diagnostic-only slice; rollback is a normal git revert.

Visible GUI repro validation, 2026-05-12: restored the saved `Bi2Se3` GUI state,
enabled only `gamma` and `Gamma`, and ran the real `Fit Geometry (LSQ)` callback.
The accepted diagnostic finding is that the optimizer residual snapshots agree
with the fit-dataset prediction, while the rendered/refined simulation caked
point can be 77-97 degrees away for the same Qr/HKL/source identity. That means
the current failure is not just stale drawing; the objective path is using a
prediction coordinate surface that disagrees with the actual rendered simulation
peak surface.

2026-05-12 point-only source fix: caked point-only Qr/Qz objective evaluation now
builds current trial hit tables with image accumulation disabled, resolves locked
Qr detector points from those hit tables first, and projects the current detector
coordinates through the exact caked projector. The GUI source-row cache remains a
fallback after hit-table miss or provider-local stale hit recovery; the fallback
is rebuilt current trial source-row data and still projects through the exact
caked projector. If source rows also cannot resolve the locked row, the point
fails closed rather than silently reusing stale visual aliases. Bug/error status:
the stale-alias objective source was localized and fixed. Feature status:
internal fitter behavior and diagnostics only; no new public API, config,
dependency, saved-state schema, or artifact schema. CI/automation status:
focused geometry objective, overlay, solver-request, and worker-cache regression
tests pass locally. Shipping status: normal bug-fix slice with git revert as the
rollback path.

2026-05-12 point-only source-row fallback closeout: the GUI `gamma,Gamma` fit
failure `qr_fit_objective_incomplete=yes resolved_count=1 expected_count=36` is
fixed for the Bi2Se3 visible-GUI reproduction. The resolver still prefers exact
current hit-table points, but hit-table misses and stale provider-local hit rows
now fall back to current trial source rows before the incomplete-objective guard
runs. Missing-pair diagnostics record hit-table reason, source-row
availability/count/signature/source, source-row candidate counts,
`source_rows_rebuilt_or_reused`, and objective-cache status.

Bug/error status: fixed for the partial-objective abort. The visible GUI run
reached the solver with `qr_fit_expected_count` / `qr_fit_resolved_count` pairs
of `36/36`, `21/21`, and `25/25`, component counts `72/72`, `42/42`, and
`50/50`, empty missing-pair lists, and `matched=82`. The run still rejected the
geometry solution on fit quality (`RMS residual 1239.15 px`, largest offset
`2374.16 px`), which is a separate geometry/residual problem rather than source
resolution failure. Feature status: bug fix and diagnostics only; no GUI
control, CLI flag, public API, config key, dependency, saved-state schema, or
artifact schema changed. CI/automation status: local focused pytest suites,
Ruff, compile, visible-GUI smoke, and diff hygiene passed; no CI workflow was
changed. Deprecation/migration status: no deprecated entrypoint, compatibility
shim, or user migration. Shipping status: ready as a normal bug-fix slice;
rollback is a normal git revert.

Real full headless `fit-geometry` smoke is now run and still failing after a
clean exact-caked request. The first divergence from passing ladder evidence is
not routing or pair identity but seed/start state: real headless fit uses the
9-variable GUI/runtime contract
(`zb,zs,theta_initial,psi_z,chi,cor_angle,gamma,Gamma,corto_detector`) and
selected 13-seed identity multistart `axis:zb-1`, while the passing ladder
comparator uses the 6-variable New4 candidate bundle
(`corto_detector,theta_initial,cor_angle,chi,zs,zb`) and a different seed
family. `full_beam_polish` is disabled in the real headless path, so
candidate-selection is downstream of the first failure.

The adjacent startup/list-refresh bug is closed: normal simulation updates now
request the same hit-table/selection cache needed by Qr/Qz picking and refresh
the listed Qr/Qz peaks automatically when the grouped row content changes, so
operators no longer need to press Update Listed Peaks after simulation. Manual
Qr/Qz refresh requests now remain pending while a hit-table refresh is in
flight instead of being consumed before rows can be captured.

The `new4` rung 1 fixed-source handoff bug/error scope is closed for objective
dry-run: provider-local fixed-source rows keep provider provenance through
subset remap, the resolver accepts only branch-proven singleton stale-row
repairs for duplicate-HKL local rows, and all seven fixed rows resolve without
HKL fallback. The review hardening also keeps duplicate-HKL local rows without
branch provenance in fallback/fail state instead of silently accepting an
assigned singleton table. This is not a solve-rung feature; center fit, full
solve, and baseline runs remain blocked until a later solve-rung project starts.

The geometric optimizer hang/convergence problem is now handled by
`scripts/debug/run_new4_geometry_fit_ladder.py`, not by the old full baseline
as the first debug tool. Rung 1 objective dry-run is green with provider-fixed
source handoff, and solve rungs remain a separate next project.

Rung 2 sensitivity scan is now implemented as a residual-probe-only ladder stop
behind `--max-rung sensitivity`. It requires the rung 1 green counters first,
hashes `new4.json` before and after, reports fixed-source counters for each
plus/minus residual probe, distinguishes patched residual probing from real
`least_squares`, and writes no center/solve rung artifacts. The rung 2 review
hardening is closed: direct function calls now fail closed when rung 1 is not
green, and per-eval counters must come from live point-match summaries rather
than request-level fallback. The adjacent review bugs are also closed:
malformed rung 1 reports cannot make the aborted rung 2 report raise, `None`,
NaN, or non-numeric per-eval counters are dirty instead of zero, and
fixed-correspondence branch mismatch counts are measured from resolver payloads
instead of being hard-coded. Abort reports now also preserve strict boolean
semantics, so malformed truthy strings cannot be reported as provider identity
or point-match success.

Rung 3 one-parameter solves are now implemented behind `--max-rung one-param`.
The ladder runs fresh same-run rungs 0/1/2, reads current-run
`rung_02_sensitivity_scan.json` `active_params`, runs singleton solve requests
only, writes one JSON per attempted parameter plus a summary, and stops before
any center, paired, block, full, feature, or baseline rung. The 2026-04-21 real
run completed with partial success: eight active parameters passed, `a` timed
out cleanly, passing parameters preserved fixed-source counters, `new4.json`
was unchanged, and the provider-only guard remained green after the run. This
is not full geometric fitter validation. Rung 3 review hardening is also
closed: Rung 3 cannot start from dirty or malformed Rung 2 top-level or
per-active fixed-source counters, boolean counter payloads are rejected as
malformed counters, timeout reports emit the full one-param schema with partial
values or nulls, and clean one-param reports without heartbeat summaries fall
back to their top-level counters instead of being misclassified as pair loss.

Rung 3A `a` timeout diagnosis is complete under
`artifacts/geometry_fit_ladder/new4_a_diagnose/`. The filtered `a` runs used
fresh provider guard, Rung 1, and Rung 2 inputs, then attempted only singleton
`a`. Variants `a_nfev5_t120`, `a_nfev10_t120`, and `a_nfev20_t300` all
completed before timeout with `diagnosis_classification == "usable"`,
`last_nfev == 6`, finite residual/RMS/max-error metrics, clean fixed-source
counters, `dirty_timeout_abort == false`, and unchanged `new4.json`. No child
kill was needed because no timeout occurred. No center, paired, block, full,
feature, baseline, or non-`a` tuning artifacts were written. This justified
including `a` in the bounded Rung 4 candidate set.

Rung 4 paired solves are complete for the initial bounded pair set. Latest
result: `status == "ok"`, 5 attempted pairs, 5 passed pairs, 0 failed or
timed-out pairs, provider guard after green, and `new4.json` unchanged. Best
pair by both RMS and max error was `[corto_detector, theta_initial]`. The run
correctly stopped at Rung 4 and did not run full fit, feature rung, baseline,
GUI fit button, block solve, or any higher rung.

The repeated cold-start speed bug is fixed for ladder solve rungs. One solver
context is captured once per ladder session, then reused by one-param, pair,
block, and feature solve probes through the warm in-process worker path. Normal
solver probes also suppress debug/intersection-cache logging unless
`--diagnostic-logging` is requested. The previous cold subprocess path remains
available with `--use-subprocess` for isolation diagnostics. Measured Rung 4
pair solves improved from 315.74 seconds total across five pairs to 5.91
seconds total, about 53.4x faster. First residual time improved from a 62.07
second average to a 0.35 second average. End-to-end pair ladder runtime is
still not fully solved because one-time context capture and pre-solve setup
remain expensive.

## 2026-04-30 headless backfill and ladder telemetry

Legacy GUI states that stored detector/background pixels for manual Qr/Qz pairs
but missed caked `(2theta, phi)` anchors are now repaired before headless
geometry-fit preparation. The headless CLI and shared headless runner rebuild
each affected background's exact-cake transform, convert background-display
points back to native detector coordinates, project through the background's
own caked bundle, and write repaired `manual_pairs` into the returned state
snapshot. This keeps GUI import, CLI `fit-geometry`, and saved-state replay on
the same caked-coordinate contract.

Rung 3 dynamic source-row probes now support an axes-only caked payload. The
trial source-row builder can reuse caked axes and signatures under changed
trial parameters, recompute `sim_visual_caked_deg` from refined geometry, and
leave saved click targets unchanged.

New4 ladder workers now write `.partial.json` progress payloads during solves.
Timeout reports can recover the current phase, least-squares timing,
residual-evaluation timing, optimizer `nfev/njev`, dynamic row rebuild counts,
manual-pick cache rebuild counts, caked projection rebuild counts, and last
fixed/dynamic row counts even when a worker does not finish. Singleton solve
rungs also skip the redundant initial dry-run objective and use the first
solver evaluation as the baseline.

## 2026-04-26 Qr/Qz caked residual objective status

Target: `q_group_key=("q_group","primary",1,10)`, `hkl=(-1,0,10)`,
branches 0 and 1.

Status by work type:

- Bug/error fixed: detector-origin and caked-origin point-consistency rungs pass
  for the target Qr/Qz group. The same physical branch point is consistent
  across detector visual/native points, caked `2theta,phi`, manual/background
  observed values, visual simulation points, fit observed values, and fit
  predictions.
- Bug/error fixed: handoff/audit, optimizer dry-run, and solver callback now use
  one authoritative fixed-manual Qr prediction resolver,
  `_resolve_fixed_manual_qr_fit_prediction`. Branch 1 previously disagreed at
  x0 because the handoff path used `(40.427885, -36.750000)` while the optimizer
  path used `(41.312142, -113.750000)`; both paths now resolve the same locked
  source and caked position.
- Bug/error fixed: the geometric optimizer objective includes the selected
  Qr/Qz caked residuals. Dry-run evaluates the production objective without
  calling `least_squares`, so rung 1 proves residual-vector content,
  fixed-source counts, fallback counts, and Qr weights before any solve rung.
- Guard added: optimizer startup compares handoff prediction and solver
  callback prediction at x0 and blocks with
  `optimizer_start_blocked_reason=prediction_resolver_mismatch` if they differ.
- Feature added: focused fitter diagnostics print resolver details, candidate
  source rows, solver inputs, trial history, Qr-only before/after norm,
  full-fit objective decomposition, multi-group residuals, and theta/phi
  sensitivity.
- Remaining limitation: Qr phi residuals are present in the objective and
  weighted, but active parameters do not meaningfully move phi. Classification:
  C, active params cannot move phi enough; this is parameterization/coverage,
  not resolver mismatch, objective omission, or zero phi weight.

Validated current counters:

- `provider_pair_count == 7`
- `dataset_pair_count == 7`
- `optimizer_request_pair_count == 7`
- `fixed_source_pair_count == 7`
- `fallback_row_count == 0`
- `missing_fixed_source_count == 0`
- `fixed_source_resolution_fallback_count == 0`
- `matched_pair_count == 7`
- `missing_pair_count == 0`
- `branch_mismatch_count == 0`
- `qr_residual_block_absent == no`
- `qr_weights == [1.0]`
- `objective_eval_called == true`
- `objective_dry_run_residual_finite == true`
- `least_squares_called == false` in dry-run mode
- `optimizer_solve_called == false` in dry-run mode

Target baseline residuals:

| Branch | Observed caked deg | Predicted caked deg | Residual caked deg | Norm |
| ---: | --- | --- | --- | ---: |
| 0 | `(40.142509, 35.566836)` | `(40.230385, 36.649182)` | `(0.087876, 1.082346)` | `1.085907480` |
| 1 | `(40.853020, -37.565855)` | `(40.427885, -36.750000)` | `(-0.425135, 0.815855)` | `0.919977798` |

Solve evidence:

- Qr-only target fit starts at the correct residual scale and accepts a reducing
  step: `1.423219664 -> 1.419117984`, `success=True`.
- Full fit keeps the Qr block present and improves both total and Qr objective:
  full objective `6.847163064 -> 6.731263668`, Qr block
  `2.819315157 -> 2.644004804`, non-Qr point block
  `6.183053304 -> 6.168734874`, line block `0.839616529 -> 0.515615371`,
  prior block `0 -> 0`, `qr_weights=[1.0]`.
- Branch identity stayed fixed during optimizer evaluations. Target rows remain
  locked on `q_group_key=("q_group","primary",1,10)`, `hkl=(-1,0,10)`, tables
  `160/167`, branches `0/1`, with no nearest-row rematch, cache row switch, or
  visual-source mismatch.
- Multi-group Qr audit after the resolver fix:

| Group | Branch | Residual before | Residual after | Status |
| --- | ---: | --- | --- | --- |
| `(-1,0,5)` | 0 | `(0.013923, 0.670630)` | `(0.380536, 0.670630)` | stable source |
| `(-1,0,5)` | 1 | `(-0.569134, 0.823943)` | `(-0.207086, 0.823943)` | stable source |
| `(-1,0,10)` | 0 | `(0.087876, 1.082346)` | `(0.507454, 1.075339)` | stable source |
| `(-1,0,10)` | 1 | `(-0.425135, 0.815855)` | `(-0.002868, 0.815855)` | stable source |
| `(-1,0,16)` | 0 | `(-0.502506, 1.018640)` | `(-0.136533, 1.018640)` | stable source |
| `(-1,0,16)` | 1 | `(-0.993222, 1.481088)` | `(-0.587420, 1.481088)` | stable source |

- Theta/phi sensitivity diagnosis: only `corto_detector` affects phi above the
  tiny tolerance, and the movement is weak/sparse. `center_x` and `center_y`
  are fixed in this 9-parameter GUI/runtime contract, so the active set mostly
  improves Qr by 2theta.
- Latest focused verification: `py_compile` passed for optimizer/GUI/runtime
  modules; focused Qr phi diagnostics passed (`4 passed, 381 deselected`);
  `tests/test_gui_runtime_import_safe.py -k "toggle_caked_2d"` passed
  (`4 passed, 308 deselected`).

Current conclusion: the full geometric fitter now optimizes the same target
Qr/Qz caked residuals reported by the CMD/rung audit, and Qr residuals improve
rather than being sacrificed to other terms. Remaining Qr weakness is phi
parameter sensitivity, not point picking, transforms, source resolution, or
objective assembly.

## Validated Ladder State

The active `new4` recovery question has moved from "is the point handoff
correct?" to "which bounded solve rung should run next?" The point-provider,
fixed-source request handoff, sensitivity, singleton solve, `a` diagnosis,
caked-point reprojection, and initial paired-solve guards below are already
validated and should not be repeated unless they regress.

Speed status as of 2026-04-22:

- Bug/error: repeated per-solve cold setup is fixed for the ladder path.
- Bug/error: manual selected-point GUI fits no longer inherit the heavy
  `max_nfev: 400`, parallel orchestration, or identifiability defaults. The
  default interactive path is serial, capped at 30 evaluations, and diagnostics
  off; unsafe parallel runtime and dynamic point fitting remain explicit richer
  paths.
- Bug/error: ladder lean solve rungs no longer run finite-difference
  identifiability diagnostics by default. The identifiability feature run keeps
  those diagnostics opt-in.
- Bug/error: solve-rung heartbeat writes are throttled and reset before each
  rung, so running JSON keeps current progress without rewriting a stale or
  growing `residual_eval_trace`; final reports still keep the full trace.
- Feature: warm in-process solver reuse is implemented and covered by tests.
- Feature: fast manual selected-point and lean ladder runtime profiles are
  covered by focused tests, with dynamic point fitting guarded as unchanged.
- Bug/error: manual caked geometry fits now fail closed unless the exact caked
  fit-space projector is hydrated per background before dataset build. Worker,
  headless, and CLI projection callbacks use each row's own background payload
  and preserve row order; caked projector errors no longer return raw detector
  rows.
- Still open: initial context capture, provider guard, objective dry-run, and
  sensitivity setup still dominate whole-run wall time.

Point-provider parity is complete. The manual Qr picker saved/refined pairs,
geometry-fit `provider_pairs`, and actual fitter handoff rows match exactly for
`new4`. The provider-only report is green with
`classification == "point_provider_parity_ok"`, 7/7 manual/provider pairs, zero
dataset/provider mismatch, zero fallback, and no optimizer call. This proves
the fitter receives the correct manual-picked points. It does not prove the
optimizer converges.

Rung 1 fixed-source handoff is complete. Provider/dataset rows survive into
`GeometryFitSolverRequest` and the optimizer subset/objective as fixed-source
rows. Required green counters are:

- `provider_pair_count == 7`
- `dataset_pair_count == 7`
- `optimizer_request_pair_count == 7`
- `fixed_source_pair_count == 7`
- `fallback_row_count == 0`
- `fixed_source_resolution_fallback_count == 0`
- `missing_fixed_source_count == 0`
- `fixed_source_resolved_count == 7`
- `fallback_entry_count == 0`
- `provider_to_optimizer_identity_match == true`
- `provider_to_optimizer_point_match == true`
- `objective_eval_called == true`
- `objective_dry_run_residual_finite == true`
- `least_squares_called == false`
- `optimizer_solve_called == false`
- `matched_pair_count == 7`
- `missing_pair_count == 0`
- `branch_mismatch_count == 0`

Root cause fixed: local provider-fixed rows were previously falling back because
local source identity was not preserved correctly through optimizer
request/subset/remap. The duplicate-HKL provider-local remap/resolver path now
keeps all 7 rows fixed.

Rung 2 sensitivity scan is complete. The ladder can perturb each candidate
parameter without running a real solve and identify active, near-zero, and
unsafe parameters. Current validation result: `status == "ok"`, Rung 1 stayed
green, `provider_pair_count == 7`, `fixed_source_pair_count == 7`,
`fallback_entry_count == 0`, `residual_probe_called == true`,
`least_squares_called == false`, `optimizer_solve_called == false`, and
`state_hash_unchanged == true`. Current classification is 11 active, 2
near-zero, 0 non-finite, and 0 unsafe. Active parameters:

- `center_x`
- `center_y`
- `chi`
- `cor_angle`
- `theta_initial`
- `corto_detector`
- `zs`
- `zb`
- `a`
- `c`
- `psi_z`

Near-zero parameters:

- `gamma`
- `Gamma`

`center_x` and `center_y` moved from near-zero to active under the unchanged
rule `delta_norm > max(1e-7, 1e-7 * abs(residual_norm_base))` because
`residual_norm_base` dropped `17.32x`
(`2665.1915227297354 -> 153.89073085166189`), shrinking the classifier
threshold `17.32x`
(`2.6651915227297353e-4 -> 1.538907308516619e-5`). Their delta norms dropped
less: `center_x` `1.37x`, `center_y` `2.15x`. This is expected under the
current rule, not a fitting regression.

Rung 3 one-parameter solves are complete enough to proceed. Each active
parameter was attempted as a singleton real solve with all other parameters
frozen. Result: `status == "ok_with_failures"`. Attempted parameters were
`chi`, `cor_angle`, `theta_initial`, `corto_detector`, `zs`, `zb`, `a`, `c`,
and `psi_z`. Passing parameters were `chi`, `cor_angle`, `theta_initial`,
`corto_detector`, `zs`, `zb`, `c`, and `psi_z`; `a` timed out; failed
parameters were none. There was no pair loss, no branch mismatch, no
no-matched-pair rejection, passing params kept fixed-source counters clean,
`new4.json` was unchanged, and the provider guard after the run was green. This
does not prove full geometric fitter validation. It only proves singleton solve
viability for the listed passing params.

Rung 3A diagnosed `a`. The previous `a` timeout was isolated with
heartbeat/timeout instrumentation. Variants `a_nfev5_t120`,
`a_nfev10_t120`, and `a_nfev20_t300` completed with `last_nfev == 6`; all
completed before timeout, no child kill was needed,
`dirty_timeout_abort == false`, fixed-source counters stayed clean,
`diagnosis_classification == "usable"`, and `new4.json` hash stayed
`f5bf185ebcfbfa8b32f161cc4bd781e177175dad84b6fce4d563f23ca021ef36`. Key
metrics: `last_residual_norm == 2666.27241841688`,
`last_rms_px == 995.892462440596`,
`last_max_error_px == 1856.9210531158`, and
`last_parameter_value == 4.15299999995151`. This justified including `a` in
the bounded Rung 4 candidate set. Do not treat `a` as a solver pathology.

Rung 3B caked point reprojection is complete. Changing `theta_initial` or
`corto_detector` recomputes caked/fit-space coordinates for the selected
detector/background points through the live exact point projector, without
recaking the full background image. Result: `status == "pass"`,
`point_count == 7`, `exact_projector_available == true`, theta projector
signature changed, distance projector signature changed, theta-perturbed
points shifted, distance-perturbed points shifted,
`full_background_recake_call_count == 0`, provider guard before/after green,
and `new4_state_hash_unchanged == true`. Raw detector pixels do not move. The
geometry transform changes, and the selected detector/native points are
reprojected into caked/fit space. This proves the selected-point residual path
responds to `theta_initial` and `corto_detector` changes without recaking the
whole image.

Rung 4 paired solves are complete for the initial bounded pair list. The latest
run attempted 5 pairs and passed all 5 with no failures and no timeouts.
Provider guard after the run was green, `new4.json` was unchanged, and the best
pair by both RMS and max error was `[corto_detector, theta_initial]`. This is
still not full geometric fitter validation: full fit, feature rung, baseline,
GUI fit button, block solve, and higher rungs were not run and remain
unclaimed.

Rung 4 warm-path performance is validated for solve-rung overhead. The old
Rung 4 pair artifacts under `artifacts/geometry_fit_ladder/new4/20260421_235235`
showed individual pair solves at 56.95, 62.50, 59.47, 58.48, and 78.34 seconds.
The warm-path measurement under
`artifacts/geometry_fit_ladder/new4/20260422_004012` shows the same initial
pair set at 1.17, 1.17, 1.31, 1.16, and 1.11 seconds. All five warm pair solves
reported `solver_context_reused == true`, clean pass status, provider guard
after green, and unchanged `new4.json`.

Rung 5 small cumulative blocks are implemented behind
`--max-rung block|blocks`. The debug pair-backed caveat is resolved for New4:
fresh same-run run `20260422_115256` rebuilt same-run evidence and passed
Rungs 1-5. Rung 5 wrote `rung_05_block_summary.json` with `status == "ok"`,
four attempted blocks, four passed blocks, zero failed blocks, zero timed-out
blocks, provider guard after blocks green, fixed-source counters clean on
passing blocks, and unchanged `new4.json`
(`f5bf185ebcfbfa8b32f161cc4bd781e177175dad84b6fce4d563f23ca021ef36`).

Fresh Rung 6 combined validation is green from that block evidence. Run
`20260422_rung7_seedfix_combined` passed both combined candidates, including C2
`corto_detector/theta_initial/cor_angle/chi/zs/zb`, with clean 7/7
fixed-source counters, provider/caked/state guards green, and
`full_fitter_validated == false`.

Controlled full-sequence Rung 7 feature gating is now blocked at
`full_beam_polish`. Fresh chain `20260422_rung7_seedfix_provider_before`,
`20260422_rung7_seedfix_caked`, `20260422_rung7_seedfix_blocks`,
`20260422_rung7_seedfix_combined`, `20260422_rung7_seedfix_features`, and
`20260422_rung7_seedfix_provider_after` kept provider/caked/Rung 5/Rung 6
guards green and `new4.json` unchanged
(`f5bf185ebcfbfa8b32f161cc4bd781e177175dad84b6fce4d563f23ca021ef36`).
Rung 7 passed `dynamic_reanchor`, `discrete_modes`, and `seed_multistart` with
clean 7/7 fixed-source counters. `dynamic_reanchor` kept lost, rejected,
fallback, and rematched pair id lists empty. `seed_multistart` selected seed 11 cleanly,
preserved 7/7 fixed manual pairs, rejected one dirty seed before ranking, and
did not select a dirty seed. `full_beam_polish` failed with
`failure_reason == "fixed_source_or_pair_integrity_lost"` and
`identifiability_features` was skipped with `prior_feature_failed`. No full
fitter, baseline, GUI fit, unrestricted feature combination, or full fitter
validation is claimed, and `full_fitter_validated == false`.

New4 ladder timing observability is implemented for Rungs 0-5. Each current-run
rung report gets finite timing metadata. Each ladder run writes
`rung_timing_summary.json`; optional `--timing-report` writes the same summary
to a chosen path and prints the concise timing table. Timing thresholds are
diagnostic only and do not change rung status, summary status, process exit
code, or pass/fail semantics. Real opt-in timing run `20260422_123330` finished
the approved `--max-rung blocks` path with `status == "ok"`, total `26.612s`,
slowest rung `caked_point_reprojection` at `9.572s`, no missing expected rungs,
no Rung 6/7 timing records, and zero non-finite elapsed values.

Rung 5 closeout status by work type:

- Feature status: `--max-rung block|blocks` is implemented and green for fresh
  same-run New4 ladder validation with per-block JSON and
  `rung_05_block_summary.json`.
- Bug/error status: evidence handling is fixed. Fatal stale/malformed/provider
  mismatch/dirty-timeout evidence aborts before solve; local `a` usability
  failures only disallow `a`; missing pair dependencies skip only affected
  blocks; stale external pair evidence remains rejected.
- Timing feature status: current-run Rung 0-5 timing report and stdout table are
  available.
- Timing bug/error status: Rung 6/7 path mappings and expected timing IDs are
  excluded from timing collection, skipped Rung 5 reports are timed, and timing
  thresholds are non-gating.
- Validation status: run `20260422_115256` passed Rungs 1-5, Rung 5 passed 4/4
  attempted blocks, provider guard after blocks stayed green, and `new4.json`
  stayed unchanged.
- Rung 2 docs baseline update: current expected baseline is
  `active_param_count=11`, `near_zero_param_count=2`. `center_x` and
  `center_y` are active because lower `residual_norm_base` shrank the
  unchanged classifier threshold faster than their delta norms fell. Keep this
  isolated from the exact-caked path, then continue any remaining full fitter,
  baseline, GUI validation, unrestricted feature combinations, and freeze/thaw
  work in a later solve project.

## Do Not Redo

Do not redo these completed validations unless their guard output regresses:

- Point-provider parity for manual Qr picker pairs, provider pairs, and fitter
  handoff rows.
- Provider-only `new4` report with no optimizer call.
- Rung 1 fixed-source request/objective dry-run.
- Rung 2 sensitivity scan.
- Rung 3 singleton solves for active parameters.
- Rung 3A `a` timeout diagnosis.
- Rung 3B caked point reprojection guard.
- Rung 4 initial paired solves.
- Rung 5 fresh same-run cumulative blocks.
- Rung 6 C2 combined candidate validation from fresh Rung 5 evidence.
- Rung 7 `dynamic_reanchor`, `discrete_modes`, and `seed_multistart`
  controlled single-feature passes from the full-sequence gate.

## Next Rung

Rung 7 exact-caked work is complete. Current Rung 2 expected baseline is
`active_param_count=11`, `near_zero_param_count=2`; `center_x` and `center_y`
are active because the unchanged threshold rule now sees a much smaller
`residual_norm_base`, not because of fitting regression. Do not reopen the
exact-caked preflight, 3B harness, or full_beam_polish paths in this track
unless a guard regresses.

Allowed parameter set for Rung 4:

- `chi`
- `cor_angle`
- `theta_initial`
- `corto_detector`
- `zs`
- `zb`
- `a`
- `c`
- `psi_z`

Validated initial pair list:

- `[a, c]`
- `[chi, cor_angle]`
- `[theta_initial, cor_angle]`
- `[corto_detector, theta_initial]`
- `[zs, zb]`

Optional later pairs were not part of the green initial Rung 4 result. Treat
them as unvalidated candidates for a future explicit pair-expansion step, not
as repeat work:

- `[c, psi_z]`
- `[a, psi_z]`
- `[corto_detector, c]`

`[a, c, psi_z]` passed as part of the fresh Rung 5 block validation in
`20260422_115256`. Rung 6 C2 later passed in
`20260422_rung7_seedfix_combined`; the current blocker is the Rung 7
`full_beam_polish` feature.

Rung 4 pass requirements per pair:

- `least_squares_called == true`
- `optimizer_solve_called == true`
- `fixed_source_pair_count == 7`
- `fixed_source_resolved_count == 7`
- `fallback_row_count == 0`
- `fixed_source_resolution_fallback_count == 0`
- `missing_fixed_source_count == 0`
- `fallback_entry_count == 0`
- `matched_pair_count == 7`
- `missing_pair_count == 0`
- `branch_mismatch_count == 0`
- `provider_to_optimizer_identity_match == true`
- `provider_to_optimizer_point_match == true`
- finite residuals
- no "No matched peak pairs were available for the fitted solution." rejection
- `after_rms_px <= before_rms_px + 0.25`
- `after_max_error_px <= before_max_error_px + 1.0`
- `new4.json` unchanged
- if pair contains `theta_initial` or `corto_detector`,
  `caked_point_reprojection_guard_ok == true`

## Still Not Validated

Full geometric fitter validation is not yet claimed. Baseline completion is not
yet claimed. RMS/max global improvement is not yet claimed. Rung 6 C2 and the
controlled Rung 7 `dynamic_reanchor`, `discrete_modes`, `seed_multistart`,
`full_beam_polish`, and `identifiability_features` feature probes are claimed
as bounded ladder evidence. The GUI fit button is not the validation path.
`run_geometry_fit_quality_baseline.py` is not the immediate next step.

## Current State

- Provider selection preserves picker-owned saved/refined background and
  simulated coordinates as provider coordinates and dataset handoff rows.
- A saved source identity resolves when it has a picker-owned saved simulated
  point and the live row matches normalized HKL plus branch/group semantics.
  Exact locator mismatch is recorded as `stale_source_identity_diagnostic`,
  not fallback.
- Missing saved simulated point still forces explicit fallback with
  `fallback_reason == "missing_saved_simulated_point"`.
- Picker-owned provider simulated points always overwrite live overlay/caked
  prefill in `initial_pairs_display`; non-picker-owned live projection keeps
  the existing no-overwrite behavior.
- The caked overwrite regression uses only `refined_sim_caked_x/y` for the
  saved simulated point, so it exercises the caked-frame branch directly.
- The canonical `new4` parity test passes with seven manual pairs, seven
  provider pairs, no fallback, matching identities, matching saved/refined
  points, matching frames, and optimizer call count zero.
- The ladder runs the provider guard first, then blocks objective dry-run before
  `least_squares` when the request or objective resolver would use fallback
  fixed-source rows.
- Real `new4` rung 1 objective dry-run is green as of 2026-04-21: 7
  provider/dataset/request/fixed rows, 0 request fallback/missing rows, 7
  fixed-source objective resolutions, 0 objective fallback entries, exact
  provider-to-optimizer identity/point match, finite dry-run residual,
  `matched_pair_count == 7`, `missing_pair_count == 0`,
  `branch_mismatch_count == 0`, `least_squares_called == false`, and
  `optimizer_solve_called == false`.
- Coordinate parity is closed through the optimizer request as of 2026-04-22:
  visual truth, provider pairs, manual point pairs, initial display pairs,
  `measured_for_fit`, `spec["measured_peaks"]`, and
  `GeometryFitSolverRequest.measured_peaks` all match for seven `new4` pairs
  without `least_squares`, solver entrypoints, or `new4.json` mutation.
- Optimizer-request capture failure is now an incomplete diagnostic state, not
  a visual/backend frame mismatch. Failed capture leaves the optimizer request
  un-compared, keeps `optimizer_request.measured_peaks` out of
  `surfaces_compared`, records `solver_request_capture_failed`, recommends
  `optimizer_request_capture`, and returns `ok == false`. Runs without
  `--include-optimizer-request` report `not_requested` and continue to judge
  provider/dataset surfaces normally.
- The dataset-to-optimizer bridge copies provider canonical identity and
  measured-point fields into the optimizer request, and the optimizer
  subset/resolver preserves provider-local fixed-source rows through objective
  matching without HKL fallback.
- Duplicate-HKL provider-local subset remap now records whether the assignment
  came from branch-aware allocation. Singleton stale-row repair is limited to
  that branch-proven path and reports actual row diagnostics separately from
  requested branch/peak identity.
- Qr/Qz UI preview and manual seed paths keep both detector-side branch
  representatives for each real Qr/Qz group, selecting only the mosaic-top row
  within each branch and preserving branch/reflection/ray provenance on the
  kept rows.
- Raw cache rows that share a Qr/Qz group but differ by
  `branch_id`, `source_branch_index`, or `source_reflection_index` collapse to
  branch representatives before initial drawing, before manual session storage,
  and after refresh. Ungrouped rows with `q_group_key is None` remain separate.
- Non-00l Qr/Qz rows that lose explicit branch metadata are no longer forced
  into one group-wide unknown branch. Collapse now preserves detector-distinct
  unknown rows by branch/source-peak/source-row/reflection identity and finally
  detector-native clustering; 00l rows still collapse to one canonical branch.
- `collapse_geometry_fit_simulated_peaks(..., one_per_q_group=True)` remains
  available for explicit whole-Qr/Qz-group collapse, but default and Qr/Qz UI
  wrapper behavior remain branch-aware, and the Qr/Qz wrapper now forwards the
  explicit whole-group option.
- Caked-mode manual Qr/Qz placements store detector display/native/caked fields
  that round-trip through the same LUT/rotation path as the live simulation
  marker projection. Refresh now trusts authoritative caked `2theta,phi` fields
  over stale detector fields.
- Detector-mode Qr/Qz selection now clicks the same visible detector marker
  position that simulation draws, while rejecting stale caked display fields as
  detector click coordinates. Structural cross-view tests cover detector to
  caked, caked to detector, stale caked-cache candidates with valid detector
  provenance, and visible detector-display coordinates that differ from raw
  provenance while carrying caked metadata. Projected rows tag `display_frame`
  as detector or caked so equal numeric detector/caked values do not hide a
  valid detector marker. Peak-selection hit testing now shares the manual
  detector-coordinate resolver, so legacy `x/y` and `simulated_x/y` caked
  aliases cannot be matched as detector pixels.
- Detector-mode Qr/Qz recognition is isolated from caked/current-view
  projection. Detector picker source selection validates each cache source and
  falls through from caked-like or detector-incomplete rows to detector-stable
  rows; detector-mode cache reprojection keeps detector display/native fields;
  and caked projection cache availability is not required to start detector
  Q-set placement.
- Simulation-native Qr/Qz detector rows now keep the simulated detector image
  frame through manual projection, peak overlay restoration, and geometry
  fallback normalization. With divergent detector/simulation rotations on a
  non-square image, the sim projection wins and the background-detector rotated
  point is ignored unless the row is explicitly tagged as a background/native
  detector row.
- Listed Qr/Qz peak rows update automatically after simulation row content
  changes. The manual Update Listed Peaks action remains available, but it is
  no longer required before detector-mode Qr/Qz picking, and pending manual
  refresh is not consumed until the listing snapshot is actually captured.
- Rung 2 residual sensitivity scan runs with `--max-rung sensitivity` only
  after rung 1 reports 7 fixed rows, zero fallback/missing rows, a finite dry
  residual, and no `least_squares` or optimizer solve call. It reports active,
  near-zero, non-finite, and unsafe parameters without mutating `new4.json`.
  Direct `run_sensitivity_scan` calls now abort before probing if rung 1 is
  missing or not green. Each moved base/plus/minus residual eval must carry a
  live `point_match_summary`; missing or dirty counters make the parameter
  unsafe. Fixed-correspondence summaries now report real branch mismatch counts
  from the resolved branch. Historical pre-threshold-shrink Rung 2 baseline:
  the 2026-04-21 real `new4` scan at
  `artifacts/geometry_fit_ladder/new4/20260421_183827` reported rung 1 green,
  rung 2 `status == "ok"`, 9 active parameters, 4 near-zero parameters, 0
  non-finite parameters, 0 unsafe parameters, `state_hash_unchanged == true`,
  and no center/solve rung artifacts. Abort-report booleans are strict
  `is True` checks, so malformed string values remain failed in both
  `rung_1_failures` and the aborted report body.
- Historical 2026-04-21 Rung 3 one-parameter solve run
  `artifacts/geometry_fit_ladder/new4/20260421_193603` used that then-current
  Rung 2 active list only: `chi`, `cor_angle`, `theta_initial`,
  `corto_detector`, `zs`, `zb`, `a`, `c`, and `psi_z`. That historical Rung 2
  baseline was green with 9 active, 4 near-zero, 0 non-finite, and 0 unsafe
  parameters. Rung 3 summary status is `ok_with_failures`: passed params are
  `chi`, `cor_angle`, `theta_initial`, `corto_detector`, `zs`, `zb`, `c`, and
  `psi_z`; failed params are none; timed-out params are `a`; skipped params are
  none.
- Every passing one-param solve reported `least_squares_called == true`,
  `optimizer_solve_called == true`, 7 fixed-source pairs, 0 fallback rows, 0
  fixed-source resolution fallback, 0 missing fixed source, 7 resolved fixed
  sources, 0 fallback entries, 7 matched pairs, 0 missing pairs, 0 branch
  mismatches, provider identity/point match true, and
  `state_hash_unchanged == true`. `a` wrote a timeout partial JSON after
  120.09 seconds and the ladder continued cleanly.
- Rung 3 best single parameter by RMS and max error was `corto_detector`
  (`after_rms_px == 704.4849611916295`,
  `after_max_error_px == 1243.9093211467562`). Summary flags:
  `any_timeout == true`, `any_pair_loss == false`,
  `any_branch_mismatch == false`, `any_no_matched_peak_rejection == false`,
  `state_hash_unchanged == true`, and `provider_guard_after_ok == true`.
- Rung 3 review findings are closed in the ladder/test blast zone. The
  one-param entry gate now requires the full fixed-source/provider contract at
  both the Rung 2 top level and each active parameter entry, missing or boolean
  counter fields fail as `sensitivity_not_green`, timeout partial JSON keeps
  all required report fields present, clean top-level one-param reports no
  longer become false pair-loss failures when heartbeat/point-summary data is
  absent, and all-active metric failures keep
  `failure_reason == "no_one_param_solve_passed"`.
- Rung 3A `a` timeout diagnosis is complete:
  `artifacts/geometry_fit_ladder/new4_a_diagnose/variant_summary.json` reports
  `status == "ok"` and `diagnosis_classification == "usable"`. Attempted
  variants were `a_nfev5_t120`, `a_nfev10_t120`, and `a_nfev20_t300`; all three
  report `param_name == "a"`, `status == "ok"`, `last_nfev == 6`,
  `heartbeat_count == 6`, finite `last_residual_norm`, `last_rms_px`, and
  `last_max_error_px`, clean fixed-source counters at the last heartbeat, no
  fixed-source counter failures, `dirty_timeout_abort == false`, and
  `state_hash_unchanged == true`. The solve completed before timeout in all
  variants, so `child_process_killed_cleanly` is not applicable rather than
  dirty. Non-selected active params were recorded as `filtered_params`, not
  failures. The diagnose directory contains no rung 4/5/6, center, paired,
  block, full, feature, or baseline artifacts.
- Rung 3B caked point reprojection is complete. The live exact point projector
  recomputes fit-space coordinates for selected detector/native points when
  `theta_initial` or `corto_detector` changes, without recaking the full
  background image. The guard passed with 7 points, projector signatures and
  perturbed points changed, full background recake count 0, provider guards
  green before/after, and unchanged `new4.json`.
- Rung 4 paired solves are complete for the initial bounded pair list:
  `[a, c]`, `[chi, cor_angle]`, `[theta_initial, cor_angle]`,
  `[corto_detector, theta_initial]`, and `[zs, zb]`. Summary status was
  `ok`: attempted 5, passed 5, failed 0, timed out 0. Provider guard after was
  green, `new4.json` was unchanged, and best pair by both RMS and max error was
  `[corto_detector, theta_initial]`. The run intentionally did not create full
  fit, feature rung, baseline, GUI fit, block, or higher-rung validation.

## Next Actions

- Treat Rung 3 one-parameter solves, the Rung 3A `a` diagnosis, Rung 3B caked
  point reprojection, Rung 4 initial paired solves, Rung 5 blocks, Rung 6 C2,
  and Rung 7 `dynamic_reanchor`/`discrete_modes`/`seed_multistart`/
  `full_beam_polish`/`identifiability_features` as complete bounded ladder
  evidence unless a guard regresses.
- Keep the current Rung 2 docs baseline update isolated from the exact-caked
  path. Expected baseline is `active_param_count=11`,
  `near_zero_param_count=2` under the unchanged threshold rule. Do not reopen
  the exact-caked preflight, 3B harness, or full_beam_polish paths in this
  track unless a guard regresses.
- Treat warm solve-rung reuse as implemented. Do not reintroduce one Python
  subprocess or one fresh solver context per candidate unless explicitly
  running `--use-subprocess` for diagnostics.
- Profile the remaining one-time setup cost before more speed work. Current
  whole-run wall time is still dominated by context capture, provider guard,
  objective dry-run, and sensitivity setup rather than pair optimizer math.
- Keep provider logic closed unless the provider-only parity gate regresses.
- Keep Qr/Qz branch seed behavior closed unless raw-cache preview, manual
  toggle, refresh, or place setup regresses to either every raw ray or one
  whole-group-only ray.
- Keep Qr/Qz listing/selection behavior closed unless a simulation update stops
  refreshing listed Qr/Qz peaks automatically, or detector-mode clicks again
  report no Qr/Qz set when clicking a visible simulated Qr/Qz marker.
- Keep detector-view Qr/Qz simulation-frame behavior closed unless simulation
  hit-table rows again project through the background detector adapter by
  default, or stale caked display fields become detector click targets.
- Keep caked-to-detector Qr/Qz return behavior closed unless the same
  simulated `2theta,phi` seed no longer redraws at the same detector marker
  position after switching view or refreshing, or Qr/Qz seed selection stops
  recognizing the same visible marker after switching detector/caked views.
- Keep manual caked geometry-fit routing closed unless caked manual pairs can
  reach dataset build without exact per-background `CakeTransformBundle`
  hydration, mixed detector/caked manual selections reach launch, or any caked
  projector exception returns raw detector/source rows.
- Keep objective rung 1 as a guard: any request/objective fallback row must stop
  before `least_squares`.
- Do not run full, baseline, GUI fit button, unrestricted feature combinations,
  feature-combo solves, broad parameter tuning, higher rungs, or loosen
  fallback rules without an explicit next-rung gate.

## Point-Provider Stop Criteria

- `point_provider_parity_gate.ok == true`.
- Manual picker pair count equals provider pair count.
- Provider pairs match manual picker truth on selected source identity,
  normalized HKL, branch/group, background point, simulated point, and frame.
- Actual dataset handoff rows match provider pairs.
- `new4` has 7 manual pairs and 7 provider pairs.
- No missing pairs, branch mismatches, silent fallback, or optimizer call.
- Targeted branch-group counters remain zero for unrelated projected/scored
  rows.

## Targeted Preflight Gate

- Use `manual_geometry_targeted` mode when saved manual geometry picks exist.
- Collect required branch-group keys before source-cache rebuild.
- Pass required branch-group keys to the targeted source-generation path.
- Use targeted fresh simulation when supported.
- Diagnose full fresh fallback and do not let it pass
  `targeted_performance_gate`.
- Reuse targeted projected cache on unchanged repeated preflight.
- On unchanged repeated preflight, do not fresh-simulate, rebuild full source
  rows, or project the full table.
- Report `targeted_performance_gate.ok == true` for an accepted canonical run.

For accepted targeted performance, `targeted_performance_gate.ok` may be true
only when all of these are true:

- `preflight_mode == "manual_geometry_targeted"`.
- `targeted_preflight_enabled == true`.
- `unrelated_projected_row_count_for_rebinding == 0`.
- `unrelated_scored_row_count_for_rebinding == 0`.
- `full_source_rows_built_for_rebinding == false`.
- `full_source_rows_projected_for_rebinding == false`.
- `targeted_cache_hit == true` or `targeted_simulation_used == true`.

Targeted preflight remains a secondary performance gate. It should not obscure
the primary point-provider parity result.

## Red Flags

- Point-provider tests launch the geometric optimizer.
- Ladder starts before the provider-only parity report is green.
- Rung 1 reaches `least_squares` while `fallback_row_count > 0`.
- Acceptance depends on the old full baseline before the bounded ladder finds a
  stable parameter set.
- Provider chooses a different simulated source identity than the picker saved.
- Provider silently rebinds a stale source identity without fallback
  diagnostics.
- Provider replaces saved/refined picker coordinates with live source-row
  projection in normal saved-value parity.
- `targeted_simulation_fallback_reason == "simulator_filter_not_supported"`
  during uncached fresh preflight.
- Full `733181`-row source build/projection during unchanged targeted
  preflight.
- `source_rows_projected_for_rebinding` approximately equals
  `total_source_rows_available`.
- `candidate_rows_scored_for_background_distance` includes unrelated HKL or
  branch rows.
- `source_cache_build_ready` waits for caked-view work.
- Runtime/debug selected candidates disagree.
- `resolved_source_pair_count == 0`.
- Huge candidate distances.

## Validation

### Guard Commands

Provider parity:

```powershell
python -m pytest tests/test_gui_geometry_fit_workflow.py -k "point_provider or new4_saved_state_without_running_optimizer" -vv
```

Provider-only report:

```powershell
python scripts/debug/validate_geometry_preflight_rebind.py --state artifacts/geometry_fit_gui_states/new4.json --background-index 0 --point-provider-report-only --report-path artifacts/geometry_fit_gui_states/new4_point_provider_report.json
```

Rung 1 objective dry-run:

```powershell
python -m pytest tests/test_gui_geometry_fit_workflow.py::test_new4_rung1_direct_objective_dry_run_green_or_fail_before_solve -vv
```

Rung 2 sensitivity:

```powershell
python scripts/debug/run_new4_geometry_fit_ladder.py --state artifacts/geometry_fit_gui_states/new4.json --background-index 0 --output-root artifacts/geometry_fit_ladder/new4 --max-rung sensitivity
```

Rung 3 one-param:

```powershell
python scripts/debug/run_new4_geometry_fit_ladder.py --state artifacts/geometry_fit_gui_states/new4.json --background-index 0 --output-root artifacts/geometry_fit_ladder/new4 --max-rung one-param --max-nfev 20 --timeout-seconds 120
```

Rung 3A `a` diagnosis:

```powershell
python scripts/debug/run_new4_geometry_fit_ladder.py --state artifacts/geometry_fit_gui_states/new4.json --background-index 0 --output-root artifacts/geometry_fit_ladder/new4_a_diagnose --max-rung one-param --one-param-filter a --max-nfev 5 --timeout-seconds 120

python scripts/debug/run_new4_geometry_fit_ladder.py --state artifacts/geometry_fit_gui_states/new4.json --background-index 0 --output-root artifacts/geometry_fit_ladder/new4_a_diagnose --max-rung one-param --one-param-filter a --max-nfev 10 --timeout-seconds 120

python scripts/debug/run_new4_geometry_fit_ladder.py --state artifacts/geometry_fit_gui_states/new4.json --background-index 0 --output-root artifacts/geometry_fit_ladder/new4_a_diagnose --max-rung one-param --one-param-filter a --max-nfev 20 --timeout-seconds 300
```

Rung 3B caked point reprojection:

```powershell
python scripts/debug/run_new4_caked_point_reprojection_check.py --state artifacts/geometry_fit_gui_states/new4.json --background-index 0 --output-root artifacts/geometry_fit_ladder/new4
```

Focused point-provider gate:

```powershell
python -m py_compile ra_sim/gui/manual_geometry.py ra_sim/gui/geometry_fit.py scripts/debug/validate_geometry_preflight_rebind.py scripts/debug/run_new4_geometry_fit_ladder.py
pytest tests/test_gui_geometry_fit_workflow.py -k "point_provider or new4_saved_state_without_running_optimizer" -vv
```

Pinpoint regression gate:

```powershell
pytest tests/test_gui_geometry_fit_workflow.py::test_point_provider_stale_locator_is_diagnostic_when_saved_assignment_resolves tests/test_gui_geometry_fit_workflow.py::test_point_provider_marks_stale_saved_identity_as_fallback tests/test_gui_geometry_fit_workflow.py::test_point_provider_saved_refined_sim_point_overwrites_caked_prefill -q
```

Qr/Qz branch-seed regression gate:

```powershell
pytest tests/test_gui_geometry_q_group_manager.py tests/test_manual_geometry_selection_helpers.py -q
git diff --check
```

Caked-to-detector return and cross-view Qr/Qz selection regressions are covered
in the same gate by structural marker/candidate tests in
`tests/test_manual_geometry_selection_helpers.py`.
Automatic listed-Qr/Qz refresh is covered by runtime source guards in
`tests/test_gui_geometry_q_group_manager.py`.
Detector-view simulation-frame selection is covered by the divergent-rotation
regression in `tests/test_manual_geometry_selection_helpers.py` and the overlay
alignment contract in `tests/test_projection_alignment_contract.py`.

Provider-only validator check:

```powershell
python scripts/debug/validate_geometry_preflight_rebind.py `
  --state artifacts/geometry_fit_gui_states/new4.json `
  --background-index 0 `
  --mode full `
  --point-provider-report-only `
  --report-path artifacts/geometry_fit_gui_states/new4_preflight_report.json
```

Bounded optimizer ladder:

```powershell
python scripts/debug/run_new4_geometry_fit_ladder.py `
  --state artifacts/geometry_fit_gui_states/new4.json `
  --background-index 0 `
  --output-root artifacts/geometry_fit_ladder/new4 `
  --max-rung sensitivity `
  --max-nfev 20 `
  --timeout-seconds 120
```

Rung 2 review hardening validation, 2026-04-21:

- `py_compile`: passed for manual geometry, GUI fitting, optimization, preflight
  validator, and ladder script.
- Point-provider parity tests: 28 passed.
- Provider-only `new4` report before and after scan:
  `classification == "point_provider_parity_ok"`.
- Fixed-source/provider-local/resolver/live-update tests: 12 passed.
- Direct real rung 1 dry-run test: passed.
- Rung 2 sensitivity tests: 15 passed.
- Historical pre-threshold-shrink `new4 --max-rung sensitivity`: passed, wrote
  only rung 0/1/2 JSON, `status == "ok"`, `residual_probe_called == true`,
  `least_squares_called == false`, `optimizer_solve_called == false`,
  `state_hash_unchanged == true`, 9 active, 4 near-zero, 0 non-finite, 0
  unsafe, and every moved eval used `counter_source == "point_match_summary"`
  with clean fixed-source counters.

Rung 3 one-parameter validation, 2026-04-21:

- `py_compile`: passed for manual geometry, GUI fitting, optimization, preflight
  validator, and ladder script.
- Point-provider parity tests: 28 passed.
- Provider-only `new4` report before and after Rung 3:
  `classification == "point_provider_parity_ok"`.
- Fixed-source/provider-local/resolver tests: 11 passed.
- Direct real rung 1 dry-run test: passed.
- Historical pre-threshold-shrink `new4 --max-rung sensitivity`:
  `artifacts/geometry_fit_ladder/new4/20260421_193458`, status `pass`, current
  rung 2 `status == "ok"`, 9 active, 4 near-zero, 0 non-finite, 0 unsafe,
  `least_squares_called == false`, `optimizer_solve_called == false`.
- Real `new4 --max-rung one-param --max-nfev 20 --timeout-seconds 120`:
  `artifacts/geometry_fit_ladder/new4/20260421_193603`, status
  `ok_with_failures`, attempted only current-run active params, passed
  `chi`, `cor_angle`, `theta_initial`, `corto_detector`, `zs`, `zb`, `c`, and
  `psi_z`, timed out `a`, no failed params, no skipped params, no pair loss,
  no branch mismatch, no no-matched-pair rejection, `new4.json` unchanged, and
  provider guard after green.

Rung 3 review hardening validation, 2026-04-22:

- Focused one-param/review suite passed with 30 tests: stale sensitivity
  default avoidance, no-active abort, strict Rung 2 dirty/missing/bool counter
  gates, singleton `candidate_param_names`/`var_names`, same base state per
  param, schema-complete timeout partial JSON, clean timeout continuation,
  dirty timeout abort, fallback-row failure, all-fail summary reason,
  partial-success visibility, provider guard after one-param, and clean
  top-level report fallback when heartbeat/point-summary data is absent.

Rung 3A `a` timeout diagnosis validation, 2026-04-22:

- `py_compile`: passed for optimization, GUI fitting, preflight validator, and
  ladder script.
- Point-provider parity tests: 28 passed.
- Provider-only `new4` report before and after diagnosis:
  `classification == "point_provider_parity_ok"`.
- Fixed-source/Rung 1 guards: fixed-source tests passed, and direct real Rung 1
  dry-run tests passed.
- `--max-rung sensitivity` guard passed under
  `artifacts/geometry_fit_ladder/new4_a_diagnose/sensitivity` with
  `state_unchanged == true`.
- Focused `--one-param-filter`/timeout tests passed: 35 passed, covering parser
  acceptance, singleton `a` attempt filtering, `filtered_params`, inactive
  filter fail-before-solve, no higher-rung artifacts, timeout schema, slow/hang
  classification, dirty fixed-source heartbeat classification, dirty child kill
  abort, and unchanged `new4` hash.
- Real filtered variants attempted `a_nfev5_t120`, `a_nfev10_t120`, and
  `a_nfev20_t300`. All completed before timeout with
  `diagnosis_classification == "usable"`, `last_nfev == 6`,
  clean fixed-source counters, no dirty child kill, no dirty timeout abort,
  finite residual metrics, and unchanged `new4.json`.

Rung 4 paired-solve validation, 2026-04-22:

- Real `new4 --max-rung pairs` completed with `status == "ok"`.
- Attempted initial pairs were `[a, c]`, `[chi, cor_angle]`,
  `[theta_initial, cor_angle]`, `[corto_detector, theta_initial]`, and
  `[zs, zb]`.
- Results: 5 attempted, 5 passed, 0 failed, 0 timed out.
- Provider guard after was green, `new4.json` was unchanged, and best pair by
  both RMS and max error was `[corto_detector, theta_initial]`.
- The run intentionally did not run full fit, feature rung, baseline, GUI fit
  button, block solve, or any higher rung.

Coordinate parity closure, 2026-04-22:

- Focused coordinate diagnostic tests: 9 passed.
- `scripts/debug/diagnose_new4_visual_backend_coordinates.py
  --include-optimizer-request` reports `ok == true`,
  `classification == "visual_backend_parity_ok"`,
  `optimizer_request_compared == true`, `optimizer_request_pair_count == 7`,
  `optimizer_request_visual_parity_ok == true`, no first mismatching surface,
  `optimizer_called == false`, `least_squares_called == false`,
  `optimizer_entrypoints_called == []`, and `state_hash_unchanged == true`.
- Optimizer-request diagnostic failure semantics are closed. Focused
  `visual_backend or coordinate_diagnostic or new4_visual_backend` suite passed
  with 14 tests. Capture failure now returns
  `diagnostic_incomplete_optimizer_request_unavailable` instead of
  `frame_mismatch_detected`, and absent optimizer-request comparison reports
  `not_requested` while provider/dataset parity remains green.

Manual caked fit fail-closed validation, 2026-04-23:

- `python -m pytest tests/test_gui_runtime_import_safe.py -k "manual_caked or manual_fit_space or async_job or headless or cli or worker_caked or projection"`:
  43 passed.
- `python -m pytest tests/test_gui_geometry_fit_workflow.py -k "manual_caked or manual_fit_space or projector or new4_ladder_block or new4_ladder_combined"`:
  49 passed.
- `python -m pytest tests/test_cli_geometry_fit.py tests/test_manual_geometry_live_peak_cache.py`:
  36 passed.
- `ruff check ra_sim/gui/geometry_fit.py ra_sim/gui/_runtime/runtime_session.py ra_sim/headless_geometry_fit.py ra_sim/cli.py tests/test_gui_runtime_import_safe.py tests/test_gui_geometry_fit_workflow.py`:
  passed.
- `python -m py_compile ra_sim/gui/geometry_fit.py ra_sim/gui/_runtime/runtime_session.py ra_sim/headless_geometry_fit.py ra_sim/cli.py tests/test_gui_runtime_import_safe.py tests/test_gui_geometry_fit_workflow.py`:
  passed.
- `python -m ra_sim.dev check` stopped at format-check on unrelated existing
  format dirt in `ra_sim/fitting/optimization.py` and `tests/test_timing.py`.
  The full Rungs 0-6 ladder was not rerun in this final fail-closed pass.

Dynamic Qr/Qz acceptance metric repair, 2026-05-12:

- The dynamic Qr/Qz geometry-fit evaluator now chooses detector residuals by
  explicit same-frame detector coordinates. Native measured anchors compare
  only with native prediction points, display anchors compare only with display
  prediction points, and mixed display-vs-native detector distances are logged
  as diagnostics instead of driving acceptance.
- Dynamic angular fits now report their final metric as caked degrees:
  `acceptance_metric_space=caked_deg`, `acceptance_rms_input_units=deg`,
  `final_rms_deg=<finite>`, and `final_rms_px=<finite only when every matched
  row has a same-frame detector prediction>`.
- GUI rejection/result text is unit-aware. The old mixed-frame rejection line
  such as `RMS residual 1239.15 px exceeds ...` no longer applies to dynamic
  caked-degree fits. If a dynamic result is rejected for another reason, the
  GUI reports the caked angular residual and notes when detector-pixel
  acceptance was unavailable because same-frame detector predictions were
  incomplete.
- Bug/error status: fixed for the 2026-05-12 Bi2Se3 GUI fit failure where the
  objective path was coherent in caked degrees but the post-solve acceptance
  path could manufacture a large mixed display/native detector-pixel RMS.
- Feature status: no new GUI control, CLI flag, public API, config key,
  saved-state schema, artifact schema, dependency, CI workflow, deprecation, or
  migration.
- Validation: `python -m pytest tests/test_geometry_fitting.py -k "point_only_projection or dynamic_angular_point_match" -q`,
  `python -m pytest tests/test_manual_geometry_selection_helpers.py -k "no_partial_qr_objective_allowed" -q`,
  and `python -m ra_sim.dev check` pass. A visible Bi2Se3 GUI harness run
  completed the dynamic fit with
  `Geometry fit: complete (... rms=88.3075deg, metric=dynamic_angular_point_match, matched=82)`;
  the old mixed-frame `rms=1239.1529px` completion/rejection line is gone.
- Shipping status: ready as a normal bug-fix slice. Rollback is a normal git
  revert of the evaluator metric, GUI formatting, and focused tests.

Dynamic Qr/Qz acceptance guard review closure, 2026-05-12:

- Follow-up review found one correctness gap after the metric repair: dynamic
  caked-degree fits no longer used mixed detector pixels, but the GUI rejection
  path also lacked a caked-degree acceptance limit. That could let a finite but
  physically bad angular fit pass.
- The GUI now keeps acceptance fail-closed for dynamic caked fits. Caked angular
  RMS is rejected above `5.00 deg`, largest caked angular offset is rejected
  above `10.00 deg`, and complete same-frame detector-pixel summaries still
  enforce the existing `100.00 px` RMS and `150.00 px` max-offset limits.
- Optimizer diagnostics now label the dynamic solver objective as
  `weighted_objective_rms=<value> weighted_deg` instead of reusing the legacy
  `weighted_residual_rms_px` label.
- The mixed display/native detector-distance diagnostics are only computed when
  row diagnostics are requested; normal residual evaluation keeps the hot path
  limited to the same-frame metric needed for summaries.
- Bug/error status: fixed. The original mixed-frame false rejection remains
  fixed, and the review-discovered accidental acceptance weakening is closed.
- Feature status: no new GUI control, CLI flag, config key, saved-state schema,
  artifact schema, dependency, CI workflow, deprecation, or migration.
- Validation: `python -m pytest tests/test_gui_geometry_fit_workflow.py -k "caked_angular_metric or dynamic_caked_metric or optimizer_diagnostics" -q`,
  `python -m pytest tests/test_geometry_fitting.py -k "point_only_projection or dynamic_angular_point_match" -q`,
  and `python -m pytest tests/test_manual_geometry_selection_helpers.py -k "no_partial_qr_objective_allowed" -q`
  pass. `python -m ra_sim.dev check` remains the release gate for the final
  commit.
- Shipping status: ready after the final quality gate passes. Rollback is a
  normal git revert of the acceptance guard, diagnostic-label, and focused test
  changes.

Dynamic angular rejection localization, 2026-05-12:

- The dynamic caked-degree objective now records compact per-row angular
  residual records independent of verbose diagnostics. The final RMS/max
  summary is recomputed from those row records and checked against the older
  array audit, so a future summary mismatch fails closed instead of hiding
  behind a secondary aggregation path.
- Multi-background dynamic fits now merge those row records before computing
  global RMS/max and worst-row summaries. The full row list stays private; the
  public summary exposes only aggregate fields and the top worst rows.
- The final point-match summary now probes fitted-variable sensitivity by
  evaluating the dynamic objective at base, plus, and minus parameter steps.
  If all fitted variables are insensitive, the result is fail-closed with
  `dynamic_objective_not_sensitive_to_fit_variables` and recommends threading
  trial parameters to the projector/source-row path.
- Worst-row diagnostics now compare each worst locked row with current
  same-`q_group_key`/`hkl` source-row candidates. This classifies branch/source
  pairing evidence without remapping automatically, and also labels phi-wrap,
  dataset-dominance, objective-insensitivity, and summary-mismatch cases.
- GUI rejection text now includes the top worst caked residual rows, failure
  class, and recommended next fix while keeping the full details in the log.
  The log now prints row count, caked RMS/max, sensitivity status, failure
  classification, and the top 10 rows.
- Bug/error status: diagnostic localization added for the current coherent
  `rms=88.3075deg` Bi2Se3 rejection. This patch does not force acceptance and
  does not change thresholds; it identifies whether the remaining failure is
  parameter threading, branch/source pairing, phi wrapping, dataset outliers,
  or a physical/manual-pick mismatch.
- Feature status: internal diagnostics and GUI rejection text only; no new GUI
  control, CLI flag, public API, config key, saved-state schema, artifact
  schema, dependency, CI workflow, deprecation, or migration.
- Validation status: focused new regression tests pass locally. Full requested
  validation and the real Bi2Se3 GUI rerun remain the next gate before claiming
  the underlying fit passes.
- Shipping status: safe as a diagnostic bug-fix slice. Rollback is a normal git
  revert of the row-summary, sensitivity, classification, GUI text, and tests.

Branch-proven duplicate HKL resolver, 2026-05-21:

- Bug/error status: fixed for locked manual Qr/Qz provider pairs where two
  rows intentionally share one HKL but represent the two physical Qr branches.
  The failing signature dropped both pairs as
  `nested_full_identity_branch_ambiguous` or
  `provider_local_duplicate_hkl_unproven`, leaving `matched_pair_count == 0`.
- The geometry-fit acceptance resolver now treats duplicate HKL rows as safe
  only when explicit, non-conflicting branch identity and q-group/source
  provenance prove the selected simulated row. The effective identity for this
  path is HKL plus `q_group_key` plus branch, refined by resolved table/source
  row provenance when available.
- Saved detector pixels remain secondary evidence. They are not accepted as
  proof by themselves for duplicate HKL rows; branch/source row proof must
  already exist.
- Feature status: no new GUI control, CLI flag, public API, config key,
  saved-state schema, artifact schema, dependency, CI workflow, deprecation, or
  migration. Unproven or conflicting duplicate HKL rows still reject.
- Validation: the focused duplicate-HKL resolver test target passed, the
  broader locked/fixed/matched-pair geometry workflow target passed, full
  `tests/test_geometry_fitting.py` passed, and `python -m ra_sim.dev check`
  passed.
- Shipping status: ready as a normal bug-fix slice. Rollback is a normal git
  revert of the resolver and regression-test changes.

Locked Qr/Qz caked authority mismatch, 2026-05-21:

- Bug/error status: fixed for the follow-up dynamic-caked failure where
  preflight and final matching both kept the two locked pairs, but the predicted
  native detector point and refined simulated native detector point were equal
  while their caked coordinates disagreed by tens of degrees.
- Root cause: the handoff audit could keep stale saved
  `simulated_two_theta_deg/simulated_phi_deg` as `fit_prediction_caked_deg`
  even when an exact detector-native to caked projection was available for the
  same native point.
- The caked objective now prefers `exact_projector_from_native` for locked Qr
  predictions and preserves the stale handoff caked value only as diagnostic
  provenance. The optimizer records a
  `locked_qr_prediction_caked_authority_mismatch` diagnostic instead of letting
  this shape look like a physical/manual-pick outlier.
- The async geometry-fit job now merges job-local fallback source rows with
  existing live preview rows by q-group/HKL/branch/source identity, so fallback
  repair does not delete unrelated live rows such as disordered-phase rows.
- Feature status: no new GUI control, CLI flag, public API, config key,
  saved-state schema, artifact schema, dependency, CI workflow, deprecation, or
  migration. This is a routing/provenance bug fix inside existing interfaces.
- Validation: `python -m pytest -q tests/test_geometry_fitting.py -k "locked_qr or caked_authority or detector_frame or phi_wrap"`,
  `python -m pytest -q tests/test_geometry_fit_live_rows_signature_handoff.py tests/test_geometry_fit_manual_fit_space_classification.py tests/test_gui_geometry_fit_workflow.py -k "locked or caked or phi or handoff"`,
  and `python -m ra_sim.dev check` pass.
- Shipping status: ready as a normal bug-fix slice. Rollback is a normal git
  revert of the authority-precedence, live-row merge, focused tests, and this
  tracking update.

Locked Qr/Qz dynamic-objective authority mismatch, 2026-05-21:

- Bug/error status: fixed for the downstream dynamic-angle path where preflight
  showed `ready=true`, optimizer setup used `dynamic-angle`, and final matching
  kept two locked pairs, but the dynamic prediction resolver still allowed a
  stale caked-display/nominal source row to become `sim_refined_caked_deg`.
- The locked-Qr dynamic resolver now carries explicit source-row native
  prediction coordinates through the existing source-row payload and exact
  projects those native coordinates before caked-display values can become the
  objective prediction. Stale `sim_visual_caked_deg`,
  `simulated_two_theta_deg/simulated_phi_deg`, and `sim_nominal_caked_deg`
  remain diagnostic for that row.
- Regression tests cover the live-shape failure: a same-HKL locked branch pair
  with stale nominal caked values and refined native detector predictions now
  evaluates at identity against the refined/exact-projected caked anchors, not
  the stale phi values near `36.750` or `-38.250` degrees.
- Feature status: no new GUI control, CLI flag, public API, config key,
  saved-state schema, artifact schema, dependency, CI workflow, deprecation, or
  migration.
- Validation: `python -m pytest -q tests/test_geometry_fitting.py -k "locked_qr or dynamic_prediction or caked_authority or detector_frame or phi_wrap or point_only_projection"`,
  `python -m pytest -q tests/test_geometry_fit_live_rows_signature_handoff.py tests/test_geometry_fit_manual_fit_space_classification.py tests/test_gui_geometry_fit_workflow.py -k "locked or caked or handoff or dynamic"`,
  and `python -m ra_sim.dev check` pass.
- Shipping status: ready as a normal bug-fix slice. Rollback is a normal git
  revert of the dynamic resolver precedence change, regression tests, and this
  tracking update.

Locked Qr/Qz dynamic authority guard follow-up, 2026-05-21:

- Bug/error status: hardened. The dynamic angular objective now exposes the
  caked prediction authority used for each locked Qr row and records whether
  exact projection, refined caked payload, or nominal caked values were used.
  Final worst-row summaries are checked against the same objective payload
  fields, so stale-summary and stale-objective paths can be separated.
- Internal failure status: if a locked Qr dynamic path still reaches nominal or
  unknown caked authority, the failure class is
  `locked_qr_dynamic_authority_mismatch` with
  `repair_locked_qr_dynamic_authority`, not
  `manual_outliers_or_physical_bad_fit`.
- GUI status: rejection text now tells the operator this is an internal
  locked-Qr dynamic-angle projection-frame error, not a bad manual pick.
- Feature status: no new GUI control, CLI flag, public API, config key,
  saved-state schema, artifact schema, dependency, CI workflow, deprecation, or
  migration. Existing private row diagnostics were extended.
- Validation: `python -m pytest -q tests/test_geometry_fitting.py -k "locked_qr or dynamic_prediction or caked_authority or detector_frame or phi_wrap"`,
  `python -m pytest -q tests/test_geometry_fit_live_rows_signature_handoff.py tests/test_geometry_fit_manual_fit_space_classification.py tests/test_gui_geometry_fit_workflow.py -k "locked or caked or handoff or dynamic or authority_mismatch"`,
  and `python -m ra_sim.dev check` pass.
- Shipping status: safe as a diagnostic hardening bug-fix slice. Rollback is a
  normal git revert of the objective-row authority fields, classification,
  GUI message, regression tests, and this tracking update.

Locked Qr/Qz optimizer-request handoff preservation, 2026-05-21:

- Bug/error status: fixed for the live shape where preflight handoff rows had
  clean locked-Qr observed/predicted detector-native and caked anchors, but the
  optimizer request builder still let thinner initial/hit-table rows carry stale
  nominal prediction fields into the dynamic-angle objective.
- The GUI optimizer-request row builder now reuses the existing
  `fit_handoff_audit_rows` payload for locked Qr pairs, preserving finite
  observed/predicted native points, caked anchors, and authority fields before
  optimization starts. The dynamic resolver records projected prediction caked
  values into the objective payload fields used by residual summaries.
- Regression status: the live same-HKL two-branch fixture now keeps
  `matched_pair_count == 2`, identity RMS below 5 degrees, branch 0 predicted
  phi near `131.750`, and branch 1 predicted phi near `39.250`; stale nominal
  phi values near `22.750` and `-38.250` do not feed the identity objective.
- Feature status: no new GUI control, CLI flag, public API, config key,
  saved-state schema, artifact schema, dependency, CI workflow, deprecation, or
  migration. This only preserves existing private locked-Qr row fields across
  the GUI/runtime handoff.
- Validation: `python -m pytest -q tests/test_geometry_fitting.py -k "locked_qr or dynamic_prediction or caked_authority or detector_frame or phi_wrap"`,
  `python -m pytest -q tests/test_geometry_fit_live_rows_signature_handoff.py tests/test_geometry_fit_manual_fit_space_classification.py tests/test_gui_geometry_fit_workflow.py -k "locked or caked or handoff or dynamic or authority"`,
  and `python -m ra_sim.dev check` pass.
- Shipping status: ready as a normal bug-fix slice. The GUI/runtime process must
  be restarted so the startup log shows this commit or a later commit instead
  of `23cd6dc6`. Rollback is a normal git revert of the handoff-preservation
  patch, focused regressions, and this tracking update.

Detector-origin locked Qr/Qz projected-row generation, 2026-05-22:

- Follow-up status: fixed the route-order case where detector-origin locked
  Qr/Qz manual pairs were still classified as detector provenance, so the
  async job chose detector projection mode before the locked-Qr caked route
  could request selected-row exact caked projections. Detector provenance now
  remains visible in `manual_fit_space_by_background`, while
  `manual_caked_fit_space_required_by_background` and source-cache projection
  mode are forced to caked for branch-proven locked-Qr rows.
- Readiness normalization now also reads nested canonical source identity
  fields when matching projected rows back to selected locked-Qr pairs, keeping
  worker preflight aligned with handoff and optimizer-request identity.
- Job build now replaces unmatched stale live-preview source rows with the
  current saved manual-pair fallback rows, while preserving current live rows
  that already match saved source identity.
- Validation: focused GUI runtime job/projection tests, projection-readiness
  workflow tests, locked-Qr handoff tests, locked-Qr dynamic/caked-authority
  fitting tests, targeted `compileall`, and targeted Ruff check pass.
  The prior unrelated `F821 current_phase_peak_edit_parameters` diagnostics
  script blocker was cleared, and `python -m ra_sim.dev check` passes.
- Bug/error status: fixed the detector-mode locked-Qr preflight block where
  selected source rows existed and an exact caked projector was available, but
  strict caked worker bundles kept `projected_rows=[]`, causing
  `locked_qr_projection_readiness ... expected_rows=4 projected_rows=0` and a
  hard preflight failure before dataset build.
- The runtime cache bundle builder now generates caked projected rows from
  stored detector source rows when the fit path is in strict caked/q-space
  projection mode and no projected rows were returned by the source-row rebuild.
  Projection failures still leave `projected_rows=[]` and fail closed at the
  existing locked-Qr projection gate.
- The worker caked projection callback now resolves pixel size from the job's
  fit parameters with the existing module/default fallback, avoiding a worker
  failure when tests or headless jobs do not expose a module-level
  `pixel_size_m`.
- Feature status: no new GUI control, CLI flag, public API, config key,
  saved-state schema, artifact schema, dependency, CI workflow, deprecation, or
  migration.
- Validation: focused locked-Qr projection and caked-projection worker tests,
  neighboring projection-readiness/source-row rebuild tests, `git diff
  --check`, and `python -m ra_sim.dev check` pass.
- Shipping status: ready as a normal bug-fix slice. Rollback is a normal git
  revert of the runtime bundle projection fallback, focused regression, and
  this tracking update.

Locked Qr/Qz handoff simplification follow-up, 2026-05-21:

- Bug/error status: unchanged from the handoff-preservation fix. The follow-up
  only simplifies the private locked-Qr optimizer-request payload copy path by
  avoiding redundant mapping copies, repeated integer conversion, and repeated
  float casts while keeping the same source precedence and field payloads.
- Feature status: no new GUI control, CLI flag, public API, config key,
  saved-state schema, artifact schema, dependency, CI workflow, deprecation, or
  migration.
- Validation: focused locked-Qr handoff regressions pass, the broader
  GUI/runtime locked/caked subset passes, Ruff format/check passes for the
  changed GUI file, and `python -m ra_sim.dev check` is the shipping gate.
- Shipping status: safe as an internal refactor slice. Rollback is a normal git
  revert of the simplification and this tracking update.

Locked Qr/Qz caked-storage timeout race, 2026-05-21:

- Bug/error status: fixed for the preflight race where selected locked Qr/Qz
  rows had already reached `source_cache_project_rows_ready` with finite exact
  caked fit-space anchors, but full caked view storage timed out about 0.1 s
  before `source_cache_full_cake_ready` and blocked dataset build.
- The runtime now separates locked-Qr row projection readiness from full caked
  image storage readiness. For locked-Qr dynamic/caked fits, finite projected
  selected rows are sufficient for the solver; full caked storage may remain
  deferred/background-pending for display/cache persistence.
- True projection failures still fail closed with locked-Qr projection-specific
  reasons instead of the old combined projection/storage timeout message.
- Diagnostics now emit `locked_qr_projection_readiness` with expected,
  projected, finite row counts, storage status, and whether a storage timeout is
  fatal for the fit path.
- Feature status: no new GUI control, CLI flag, public API, config key,
  saved-state schema, artifact schema, dependency, CI workflow, deprecation, or
  migration.
- Validation: focused projection-readiness, storage-timeout, handoff/caked
  classification, locked-Qr dynamic/line-angle tests, and `python -m
  ra_sim.dev check` pass.
- Shipping status: ready as a normal bug-fix slice. Rollback is a normal git
  revert of the runtime preflight split, focused regressions, and this tracking
  update.

Locked Qr/Qz multi-group handoff, 2026-05-21:

- Bug/error status: fixed for the group-specific failure where the handoff audit
  builder only produced repaired locked-Qr payload rows for
  `q_group_key=("q_group","primary",1,10)` / `hkl=(-1,0,10)`, so a second
  selected group such as `(-1,0,5)` could fall back to stale caked authority and
  explode during dynamic-angle validation.
- The handoff audit path now emits rows for every branch-proven locked Qr/Qz
  fixed-source pair instead of filtering to one hardcoded group. The optimizer
  request builder matches those audit rows by q-group, HKL, source branch,
  source table/reflection, source row, and source label before using pair-index
  or row-order fallback.
- Nested canonical provider provenance is now treated as first-class identity
  for audit row construction, not only for the optimizer-request lookup. This
  covers reduced or merged rows where top-level q-group/HKL/branch fields are
  absent but `provider_selected_source_identity_canonical` still carries the
  locked-Qr source identity.
- Dynamic objective status: the existing dynamic-angle evaluator and branch-line
  residual builder are already multi-group safe when they receive clean
  handoff payloads. New regressions prove two selected Qr groups keep
  `matched_pair_count == 4`, identity residual RMS below 5 degrees, and
  separate line groups for `1,5` and `1,10`.
- Feature status: no new GUI control, CLI flag, public API, config key,
  saved-state schema, artifact schema, dependency, CI workflow, deprecation, or
  migration.
- Validation: focused multi-group handoff, optimizer-request, locked-Qr dynamic,
  branch-line grouping, and manual caked classification tests pass.
- Shipping status: ready as a normal bug-fix slice. Rollback is a normal git
  revert of the generic locked-Qr audit filter, identity-key optimizer payload
  matching, focused regressions, and this tracking update.

Do not use `run_geometry_fit_quality_baseline.py` as the first optimizer debug
tool. Run it only after the ladder identifies a stable parameter set.

## Links

- [New4 geometric fitter recovery handoff](new4-geometric-fitter-recovery-handoff.md)
- [Tracking hub](../index.md)
- [GUI workflow](../../gui-workflow.md)
- [Geometry fitting from picked spots](../../simulation_and_fitting.md#geometry-fitting-from-picked-spots)
