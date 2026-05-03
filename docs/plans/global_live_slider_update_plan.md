# Global Live Slider Update System

Status: implementation plan for later work

Scope: all GUI sliders in `ra_sim`, including display sliders, integration-region sliders, selected-Qr rod controls, plot sliders, and simulation/model sliders.

Primary goal: make slider interaction feel continuous. Slider state, labels, and preview-capable figures should update while the user drags. Expensive exact work should run once after interaction settles.

Core rule: this must be a throttled live-preview system plus delayed exact final refresh. It must not be only a debounce.

---

## 1. Problem

Many interface sliders still feel slow between frames. The math may be cheap, but the GUI can stall because each slider event may run a synchronous refresh path.

Typical current flow:

```text
slider event
-> update state
-> recompute masks/profiles/simulation
-> update plot data
-> autoscale/rebuild metadata
-> Matplotlib full draw
-> next slider event waits
```

This creates visible lag and can queue stale intermediate values.

The selected-Qr rod path already has several local optimizations, including a `delta_Qr` slider, selected-Qr visual band rendering, debounced selected-rod refresh, cached selected-Qr profile sums, and reusable plot artists. This plan generalizes the same idea across the whole interface.

---

## 2. Intended behavior

For every slider:

```text
slider moves
-> state updates immediately
-> paired entry/label updates immediately
-> one global preview frame is scheduled
-> preview frame reads latest state only
-> old intermediate slider values are dropped
-> visible figure updates at capped rate
-> exact final refresh runs once after idle
```

For preview-capable sliders, the figure should visibly update during dragging.

For expensive sliders that cannot honestly preview exact data, labels and lightweight visual hints should update immediately, and the exact result should run once after interaction settles.

---

## 3. Non-negotiable invariants

These invariants must remain true throughout implementation:

```text
Selected-Qr rod profiles use caked 2theta/phi data only.
Detector-native selected-Qr masks are overlay/drag only.
Union selected-Qr masks are overlay/drag only.
Per-rod selected-Qr profiles use per-rod caked masks.
Runtime selected-Qr paths never call detector-space profile integration.
Normal detector angular ROI still works when selected-Qr mode is off.
Normal radial/azimuth 1D plots recover after selected-Qr mode exits.
Preview must not change numerical semantics.
Final refresh must produce exact current output.
```

---

## 4. Architecture

Add one global scheduler for slider live updates.

Each slider registers a `SliderLiveUpdateSpec`.

Suggested modes:

```text
visual_only
    Display property only. Preview is final. No numerical recompute.

exact_fast
    Full exact update is cheap enough to run during preview frames.

preview_then_final
    Preview updates visible artists quickly. Exact refresh runs after idle.

final_only
    No truthful preview exists. State/label update immediately. Exact work is coalesced after idle.
```

The scheduler should coalesce events by domain.

Example domains:

```text
display
integration_roi
selected_qr_rod
simulation_model
plot_axes
fit_controls
```

A single global preview callback should service all dirty domains. Each domain gets its own delayed final refresh callback.

---

## 5. Proposed data structures

Create a new module:

```text
ra_sim/gui/live_update.py
```

Suggested spec:

```python
from dataclasses import dataclass, field
from typing import Callable, Any

@dataclass(frozen=True)
class SliderLiveUpdateSpec:
    slider_id: str
    domain: str
    live_mode: str  # visual_only, exact_fast, preview_then_final, final_only
    preview_frame_ms: int = 33
    final_idle_ms: int = 175
    coalesce_key: str = ""
    read_latest_value: Callable[[], Any] | None = None
    apply_state: Callable[[Any], None] | None = None
    apply_preview: Callable[[], bool] | None = None
    apply_final: Callable[[], bool] | None = None
    is_widget_alive: Callable[[], bool] | None = None
```

Suggested scheduler state:

```python
@dataclass
class SliderLiveUpdateState:
    preview_after_id: object | None = None
    final_after_ids_by_domain: dict[str, object] = field(default_factory=dict)
    dirty_slider_ids: set[str] = field(default_factory=set)
    dirty_domains: set[str] = field(default_factory=set)
    latest_values: dict[str, object] = field(default_factory=dict)
    sequence_by_domain: dict[str, int] = field(default_factory=dict)
    preview_running: bool = False
```

Suggested public API:

```python
def register_slider_live_update_spec(spec: SliderLiveUpdateSpec) -> None:
    ...


def request_slider_live_update(
    *,
    slider_id: str,
    value: object,
    reason: str,
) -> None:
    ...


def run_slider_preview_frame() -> None:
    ...


def run_slider_final_refresh(domain: str, sequence: int) -> None:
    ...
```

Scheduler rules:

```text
Only one preview callback may be scheduled globally.
Only one final callback may be scheduled per domain.
New input cancels/reschedules that domain final refresh.
Preview callback reads latest values when it runs.
Preview callback must not capture stale slider values.
Destroyed widgets skip safely.
Intermediate values are dropped.
Final callback ignores stale sequence IDs.
```

---

## 6. Slider inventory

Before implementation, inventory every slider-like control.

Likely files:

```text
ra_sim/gui/views.py
ra_sim/gui/bootstrap.py
ra_sim/gui/integration_range_drag.py
ra_sim/gui/_runtime/runtime_session.py
ra_sim/gui/state.py
```

Include these slider classes or equivalents:

```text
tk.Scale
ttk.Scale
custom sliders
slider-like spin/entry paired controls if they continuously update
```

For each slider, record:

```text
slider_id
domain
state variable
paired entry or label
current callback
current heavy refresh path
live mode
preview function
final function
```

Add a test-visible helper:

```python
def iter_runtime_slider_specs() -> list[SliderLiveUpdateSpec]:
    ...
```

---

## 7. Implementation phases

### Phase 0: Baseline inventory and guard tests

Goal: document every slider and lock down existing semantics.

Patch:

```text
Add slider inventory helper.
Add specs for every known slider.
Add tests proving no slider is missing from the registry.
Add selected-Qr detector-integration spy guard.
```

Checks:

```text
Every slider has a stable slider_id.
Every slider has a live-update spec.
No heavy selected-Qr detector integration path is called.
Existing selected-Qr caked-only profile guard passes.
```

Validation:

```bash
python -m pytest tests/test_gui_views.py tests/test_gui_runtime_import_safe.py -ra
python -m compileall ra_sim tests
git diff --check
```

Stop condition: do not add scheduler logic until the inventory is complete.

---

### Phase 1: Add global slider live-update scheduler

Goal: create one scheduler shared by all sliders.

Patch:

```text
Add ra_sim/gui/live_update.py.
Add SliderLiveUpdateSpec.
Add SliderLiveUpdateState.
Add request_slider_live_update().
Add preview frame callback.
Add domain final refresh callback.
Wire scheduler state into runtime state.
```

Checks:

```text
N rapid slider events schedule one active preview callback.
N rapid slider events schedule one final callback per domain.
Preview callback uses latest value.
Second burst after first frame works.
Destroyed widget skips preview.
Callback token clears after callback runs.
Stale after_cancel errors are swallowed.
```

Validation:

```bash
python -m pytest tests/test_gui_integration_range_drag.py tests/test_gui_runtime_import_safe.py -ra
python -m compileall ra_sim tests
git diff --check
```

---

### Phase 2: Route slider callbacks through scheduler

Goal: stop sliders from calling heavy refresh paths directly.

Patch:

Add a wrapper:

```python
def bind_live_slider(
    slider: object,
    *,
    spec: SliderLiveUpdateSpec,
    variable: object | None = None,
    entry_var: object | None = None,
    label_var: object | None = None,
) -> None:
    ...
```

Callback behavior:

```text
Slider move:
    parse value
    update state immediately
    sync entry/label immediately
    request live update

Entry commit:
    parse value
    update slider position
    update state
    sync label
    request live update
```

Do not call full refresh directly from slider callbacks.

Checks:

```text
Slider movement updates state immediately.
Slider movement updates paired entry/label immediately.
Entry commit updates slider position.
Slider movement does not call full refresh directly.
Every registered slider routes through request_slider_live_update().
Selected-Qr delta_Qr debounce path is not duplicated.
```

Validation:

```bash
python -m pytest tests/test_gui_views.py tests/test_gui_integration_range_drag.py -ra
python -m compileall ra_sim tests
git diff --check
```

---

### Phase 3: Display-only sliders

Goal: make visual sliders update continuously without numerical recompute.

Candidate sliders:

```text
image contrast
color limits
overlay opacity
band opacity
line width
marker size
visual scaling controls
```

Mode:

```text
visual_only
```

Preview behavior:

```text
set artist/image property
call canvas.draw_idle()
no simulation recompute
no profile recompute
no mask rebuild
no export metadata update
```

Checks:

```text
Display slider calls no math recompute.
Display slider updates artist property.
Display slider uses draw_idle.
Rapid display slider events produce bounded preview frames.
No final refresh is scheduled for visual_only sliders.
```

Validation:

```bash
python -m pytest tests/test_gui_views.py tests/test_gui_runtime_import_safe.py -ra
python -m compileall ra_sim tests
git diff --check
```

---

### Phase 4: Integration and ROI sliders

Goal: make ROI/profile sliders update visible overlays and profiles continuously while final bookkeeping runs after idle.

Candidate sliders:

```text
phi min/max
2theta min/max
qz min/max
delta_Qr
selected-Qr Qz bounds
integration range sliders
```

Mode:

```text
preview_then_final
```

Preview path:

```text
update overlay image/artist
update existing profile Line2D data
reuse axes
reuse cached geometry
reuse selected-Qr caked shared inputs
use draw_idle
skip export metadata
skip legends/layout rebuild
skip autoscale if y-limits are stable
```

Final path:

```text
run exact profile update
run exact overlay update
autoscale if enabled
update last_1d_integration_data
update export metadata
update status text
record profiling timing
```

Checks:

```text
Changing phi/qz/delta_Qr with same rod count preserves axes identity.
Changing phi/qz/delta_Qr with same rod count preserves Line2D identity.
Preview path does not call fig.clear.
Preview path does not call axis.clear.
Preview path does not update export metadata.
Final path updates last_1d_integration_data.
Final profile equals exact non-preview result.
Selected-Qr detector integration spy reports zero calls.
```

Validation:

```bash
python -m pytest \
  tests/test_gui_integration_range_drag.py \
  tests/test_gui_runtime_import_safe.py \
  tests/test_gui_qr_cylinder_overlay.py -ra
python -m compileall ra_sim tests
git diff --check
```

---

### Phase 5: Simulation and model sliders

Goal: prevent expensive model sliders from queuing stale simulations.

Candidate sliders:

```text
sample orientation
geometry parameters
lattice parameters
mosaic/spread controls
simulation intensity parameters
model/fitting sliders
```

Mode selection:

```text
exact_fast:
    use only if exact update reliably fits within preview budget

preview_then_final:
    use if preview can update lightweight visual hints

final_only:
    use if exact recompute is expensive and no truthful preview exists
```

Rules:

```text
State and labels update immediately.
Full simulation does not run for every intermediate value unless exact_fast is proven safe.
Final exact simulation runs once after idle.
Old delayed results are discarded using sequence IDs.
No rendered image updates with stale result.
```

Checks:

```text
N rapid simulation slider moves schedule one final simulation.
Old delayed result is discarded if newer sequence exists.
State/label updates immediately.
Exact final result matches direct non-scheduled compute.
```

Validation:

```bash
python -m pytest tests/test_gui_runtime_import_safe.py tests/test_gui_views.py -ra
python -m compileall ra_sim tests
git diff --check
```

---

### Phase 6: Plot and axis sliders

Goal: axis/view sliders update immediately without recomputing data.

Candidate sliders:

```text
plot x/y limits
profile scale controls
colorbar range sliders
smoothing controls
zoom-like controls
```

Rules:

```text
Axis-limit slider:
    set_xlim/set_ylim
    draw_idle
    no data recompute

Smoothing slider:
    if exported data changes, preview line only and final metadata update
    if display-only, visual_only
```

Checks:

```text
Axis slider calls no simulation/profile recompute.
Axis slider updates limits.
Axis slider uses draw_idle.
Final metadata only runs when slider affects exported data.
```

Validation:

```bash
python -m pytest tests/test_gui_views.py tests/test_gui_runtime_import_safe.py -ra
python -m compileall ra_sim tests
git diff --check
```

---

### Phase 7: Global Matplotlib render policy

Goal: reduce unnecessary full redraws.

Patch:

Add:

```python
def request_canvas_redraw(
    canvas: object,
    *,
    reason: str,
    prefer_blit: bool = False,
) -> None:
    ...
```

Default behavior:

```text
canvas.draw_idle()
```

Only force `canvas.draw()` in controlled setup or test cases.

Optional blitting:

```python
class LiveBlitManager:
    axes: list[object]
    animated_artists: list[object]
    background: object | None
    valid: bool

    def invalidate(self) -> None: ...
    def draw_preview(self) -> bool: ...
```

Use blitting only when:

```text
same axes
same artist objects
axis limits unchanged
figure size unchanged
legend/layout unchanged
```

Fallback to `draw_idle()` otherwise.

Checks:

```text
draw_idle used by default.
Blit path used only when background is valid.
Resize invalidates blit background.
Axis limit change invalidates blit background.
Numerical output unchanged with blit on/off.
```

Validation:

```bash
python -m pytest tests/test_gui_runtime_import_safe.py tests/test_gui_views.py -ra
python -m compileall ra_sim tests
git diff --check
```

---

### Phase 8: Global slider profiling

Goal: make slow sliders identifiable.

Patch:

Add env-gated profiling:

```bash
RA_SIM_PROFILE_SLIDERS=1
```

Use bounded records:

```python
_MAX_SLIDER_TIMINGS = 512
```

Record fields:

```python
{
    "slider_id": slider_id,
    "domain": domain,
    "stage": stage,
    "elapsed_ms": elapsed_ms,
    "sequence": sequence,
}
```

Stages:

```text
slider_callback
preview_schedule
preview_frame
domain_preview
domain_final
canvas_draw_idle
blit_draw
simulation_final
profile_final
overlay_final
```

Checks:

```text
No timing records when env unset.
Timing records exist when env enabled.
Timing sink capped at 512.
Profiling on/off produces same results.
```

Validation:

```bash
python -m pytest tests/test_gui_runtime_import_safe.py tests/test_gui_integration_range_drag.py -ra
python -m compileall ra_sim tests
git diff --check
```

---

### Phase 9: Full regression suite

Goal: verify all sliders are registered and correct.

Checks:

```text
Every slider has a SliderLiveUpdateSpec.
Every slider routes through request_slider_live_update.
No slider callback directly calls a known heavy full-refresh function.
Visual-only sliders do not schedule final recompute.
Preview-then-final sliders schedule one final refresh per rapid burst.
Final-only sliders still update labels immediately.
Destroyed widgets do not run preview/final callbacks.
No callback backlog remains after simulated event-loop drain.
Selected-Qr detector integration spy remains zero.
Selected-Qr profiles remain caked-only.
Union masks remain overlay/drag only.
Per-rod profiles remain per-rod caked masks.
Normal detector angular ROI still works when selected-Qr mode is off.
Normal radial/azimuth plotting still works after selected-Qr mode exit.
```

Validation:

```bash
python -m pytest \
  tests/test_gui_views.py \
  tests/test_gui_integration_range_drag.py \
  tests/test_gui_runtime_import_safe.py \
  tests/test_gui_qr_cylinder_overlay.py \
  tests/test_gui_state_io.py -ra
python -m compileall ra_sim tests
git diff --check
```

---

### Phase 10: Documentation and manual validation

Patch:

```text
Update CHANGELOG.md.
Update docs/gui-workflow.md.
Optionally add docs/tracking/in-progress/global-live-slider-update.md.
```

Manual workflow:

```text
Start app.
Move every slider slowly.
Move every slider quickly.
Confirm labels/entries update immediately.
Confirm preview-capable figures update during drag.
Confirm no visible backlog after release.
Confirm final resting frame is exact.
Confirm selected-Qr delta_Qr slider remains responsive.
Confirm selected-Qr profiles remain caked-data based.
Confirm detector overlay remains detector-native.
Confirm detector angular ROI works when selected-Qr mode is off.
Confirm normal radial/azimuth plot recovers after selected-Qr mode exit.
```

Profiling smoke:

```bash
RA_SIM_PROFILE_SLIDERS=1 python -m ra_sim
```

Inspect:

```text
slowest slider_id
slowest domain_preview
slowest domain_final
draw_idle/blit behavior
callback backlog
```

---

## 8. Global acceptance criteria

Implementation is acceptable only if all are true:

```text
Every slider is registered in the global live-update scheduler.
Slider state and labels update immediately.
Preview-capable sliders update visible figures continuously.
Intermediate slider values are dropped, not queued.
Each dirty domain gets one exact final refresh after idle.
Visual-only sliders do not trigger numerical recompute.
Selected-Qr profiles remain caked-path only.
Detector-native masks remain overlay/drag only.
Union masks remain overlay/drag only.
Per-rod profiles remain per-rod caked masks.
No runtime selected-Qr path calls detector integration.
Matplotlib draw calls are coalesced with draw_idle by default.
Blitting is optional and only used when safe.
Profiling is env-gated and bounded.
Manual GUI validation is run before merge.
```

---

## 9. Risks and mitigations

### Risk: preview and final outputs diverge

Mitigation:

```text
Preview may skip autoscale, legends, layout, export metadata.
Preview must not change numerical semantics.
Final refresh must always run after idle and produce exact current output.
```

### Risk: some sliders cannot truthfully preview exact data

Mitigation:

```text
Use final_only or preview_then_final with label/overlay-only preview.
Do not display fake exact science data.
```

### Risk: callback backlog remains

Mitigation:

```text
Only one preview callback globally.
Only one final callback per domain.
Drop intermediate states.
Use sequence IDs to discard stale delayed results.
```

### Risk: Matplotlib still dominates

Mitigation:

```text
Use draw_idle by default.
Reuse artists.
Avoid figure/axis clearing during preview.
Add blitting only after draw_idle path remains too slow.
```

### Risk: scheduler complexity obscures simple behavior

Mitigation:

```text
Every slider has one visible spec.
Every preview/final path is tested by slider_id and domain.
Profiling records identify slow paths.
Direct heavy callbacks are prohibited by tests.
```

---

## 10. Implementation notes for Codex

Use small phases and stop after each phase.

For each phase, report:

```text
files changed
tests added
test results
compileall result
git diff --check result
known unrelated failures
```

Do not run broad GUI manual validation until the scheduler and callback routing are patched. It would test the old path.

Preserve unrelated dirty worktree changes. Touch only files required for the active phase.

Do not commit until requested.

