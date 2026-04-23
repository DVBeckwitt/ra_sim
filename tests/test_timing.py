import json

from ra_sim.io.data_loading import load_gui_state_file
from ra_sim import launcher, timing
from scripts import measure_gui_timing


def _timing_test_event(name: str, ns: int, **fields: object) -> dict[str, object]:
    event: dict[str, object] = {
        "event": name,
        "perf_counter_ns": ns,
        "scenario_id": fields.pop("scenario_id", "defaults-restored"),
        "trial_id": fields.pop("trial_id", "1"),
        "phase": fields.pop("phase", "startup"),
    }
    event.update(fields)
    return event


def _saved_state_trial_events(*, uncorrelated: bool = False) -> list[dict[str, object]]:
    compute_update_id = 7
    visible_update_id = 8 if uncorrelated else compute_update_id
    return [
        _timing_test_event("process.launch.start", 0),
        _timing_test_event("python.entry", 10_000_000),
        _timing_test_event("gui.command.received", 20_000_000),
        _timing_test_event("gui.main.start", 25_000_000),
        _timing_test_event("gui.runtime.import.start", 30_000_000),
        _timing_test_event("gui.runtime.import.end", 40_000_000),
        _timing_test_event("gui.main.start", 50_000_000),
        _timing_test_event("tk.root.create.start", 60_000_000),
        _timing_test_event("tk.root.create.end", 70_000_000),
        _timing_test_event("widgets.build.start", 80_000_000),
        _timing_test_event("widgets.build.end", 90_000_000),
        _timing_test_event(
            "saved_state.start",
            100_000_000,
            state_path_basename="new4.json",
            state_file_size_bytes=1234,
        ),
        _timing_test_event(
            "saved_state.metadata",
            101_000_000,
            state_path_basename="new4.json",
            state_file_size_bytes=1234,
            restored_background_count=3,
            restored_manual_pair_count=1,
            restored_selection_count=2,
            restored_has_caked_state=True,
            restored_has_geometry_manual_pairs=True,
            metadata_status="measured",
        ),
        _timing_test_event("saved_state.file_read.start", 110_000_000),
        _timing_test_event("saved_state.file_read.end", 120_000_000),
        _timing_test_event("saved_state.json_parse.start", 130_000_000),
        _timing_test_event("saved_state.json_parse.end", 140_000_000),
        _timing_test_event("saved_state.snapshot_import.start", 141_000_000),
        _timing_test_event("saved_state.snapshot_import.end", 150_000_000),
        _timing_test_event("saved_state.variable_restore.start", 160_000_000),
        _timing_test_event("saved_state.variable_restore.end", 170_000_000),
        _timing_test_event("saved_state.background_load.start", 180_000_000),
        _timing_test_event("saved_state.background_load.end", 210_000_000),
        _timing_test_event("saved_state.cif_restore_or_read.start", 220_000_000),
        _timing_test_event("saved_state.cif_restore_or_read.end", 260_000_000),
        _timing_test_event("saved_state.geometry_manual_pairs_restore.start", 270_000_000),
        _timing_test_event("saved_state.geometry_manual_pairs_restore.end", 280_000_000),
        _timing_test_event("saved_state.manual_geometry_cache_restore.start", 281_000_000),
        _timing_test_event("saved_state.manual_geometry_cache_restore.end", 290_000_000),
        _timing_test_event("saved_state.caked_state_restore.start", 300_000_000),
        _timing_test_event("saved_state.caked_state_restore.end", 320_000_000),
        _timing_test_event("saved_state.overlay_restore.start", 321_000_000),
        _timing_test_event("saved_state.overlay_restore.end", 340_000_000),
        _timing_test_event("saved_state.end", 350_000_000),
        _timing_test_event("first_update.requested", 360_000_000),
        _timing_test_event(
            "first_update.requested",
            370_000_000,
            update_id=compute_update_id,
        ),
        _timing_test_event(
            "first_update.update_begin",
            371_000_000,
            update_id=compute_update_id,
        ),
        _timing_test_event(
            "first_simulation.compute.start",
            400_000_000,
            update_id=compute_update_id,
        ),
        _timing_test_event(
            "first_simulation.beam_sample_generation.start",
            410_000_000,
            update_id=compute_update_id,
        ),
        _timing_test_event(
            "first_simulation.beam_sample_generation.end",
            420_000_000,
            update_id=compute_update_id,
        ),
        _timing_test_event(
            "first_simulation.kernel_call.start",
            430_000_000,
            update_id=compute_update_id,
        ),
        _timing_test_event(
            "first_simulation.kernel_call.end",
            470_000_000,
            update_id=compute_update_id,
        ),
        _timing_test_event(
            "first_simulation.result_ready.start",
            480_000_000,
            update_id=compute_update_id,
        ),
        _timing_test_event(
            "first_simulation.result_ready.end",
            490_000_000,
            update_id=compute_update_id,
        ),
        _timing_test_event(
            "first_simulation.compute.end",
            500_000_000,
            update_id=compute_update_id,
        ),
        _timing_test_event(
            "first_result.apply.start",
            502_000_000,
            update_id=compute_update_id,
        ),
        _timing_test_event(
            "first_result.apply.end",
            506_000_000,
            update_id=compute_update_id,
        ),
        _timing_test_event(
            "first_gui_render_after_compute.start",
            510_000_000,
            update_id=compute_update_id,
        ),
        _timing_test_event(
            "first_gui_render_after_compute.end",
            600_000_000,
            update_id=compute_update_id,
        ),
        _timing_test_event(
            "first_canvas.draw_event",
            650_000_000,
            update_id=compute_update_id,
        ),
        _timing_test_event(
            "first_tk.after_idle.visible",
            700_000_000,
            update_id=visible_update_id,
        ),
    ]


def _saved_state_async_worker_events(*, queued: bool) -> list[dict[str, object]]:
    events = [
        event
        for event in _saved_state_trial_events()
        if not str(event["event"]).startswith("first_result.apply")
    ]
    async_events = []
    if queued:
        async_events.append(
            _timing_test_event(
                "first_worker.request.queued",
                380_000_000,
                update_id=7,
            )
        )
    async_events.extend(
        [
            _timing_test_event("first_worker.submit", 390_000_000, update_id=7),
            _timing_test_event(
                "simulation_worker.result.fetch.start",
                501_000_000,
                update_id=7,
            ),
            _timing_test_event(
                "simulation_worker.result.fetch.end",
                503_000_000,
                update_id=7,
            ),
            _timing_test_event(
                "simulation_worker.result.ready",
                504_000_000,
                update_id=7,
            ),
            _timing_test_event("first_result.apply.start", 506_000_000, update_id=7),
            _timing_test_event("first_result.apply.end", 509_000_000, update_id=7),
        ]
    )
    return sorted([*events, *async_events], key=lambda event: int(event["perf_counter_ns"]))


def _write_timing_summary(tmp_path, events: list[dict[str, object]], **metadata_fields):
    metadata = {
        "backend": "unit",
        "python": "unit",
        "scenario": "defaults-restored",
        "trials": 1,
        "state_file": "new4.json",
        **metadata_fields,
    }
    measure_gui_timing._write_summary(
        tmp_path,
        metadata,
        trial_events=[events],
        trial_results=[
            measure_gui_timing.TrialResult(
                path=tmp_path / "trial_001.jsonl",
                returncode=0,
                timed_out=False,
                termination="normal",
                child_pid=1,
                measured=True,
            )
        ],
    )
    return json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))


def test_timing_event_noops_without_env(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "timing.jsonl"
    monkeypatch.delenv("RA_SIM_TIMING", raising=False)
    monkeypatch.setenv("RA_SIM_TIMING_OUT", str(output_path))

    timing.timing_event("ignored")

    assert not output_path.exists()


def test_timing_event_writes_jsonl_with_required_fields(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "timing.jsonl"
    monkeypatch.setenv("RA_SIM_TIMING", "1")
    monkeypatch.setenv("RA_SIM_TIMING_OUT", str(output_path))
    monkeypatch.setenv("RA_SIM_TIMING_SCENARIO", "unit")
    monkeypatch.setenv("RA_SIM_TIMING_TRIAL", "3")

    timing.timing_event("example", phase="test", update_id=7, path_field="C:/tmp/input.cif")

    payload = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert payload["event"] == "example"
    assert isinstance(payload["perf_counter_ns"], int)
    assert payload["phase"] == "test"
    assert payload["update_id"] == 7
    assert payload["scenario_id"] == "unit"
    assert payload["trial_id"] == "3"
    assert payload["path_field"] == "input.cif"


def test_timing_span_and_new_update_id_emit_events(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "timing.jsonl"
    monkeypatch.setenv("RA_SIM_TIMING", "1")
    monkeypatch.setenv("RA_SIM_TIMING_OUT", str(output_path))

    update_id = timing.new_update_id("unit")
    with timing.timing_span("work", update_id=update_id):
        pass

    events = [
        json.loads(line)["event"] for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert "update_id.new" in events
    assert "work.start" in events
    assert "work.end" in events


def test_duration_pairs_ordered_events_not_first_unordered_match() -> None:
    events = [
        {"event": "end", "perf_counter_ns": 5},
        {"event": "start", "perf_counter_ns": 10},
        {"event": "end", "perf_counter_ns": 30},
    ]

    assert measure_gui_timing._duration_ms(events, "start", "end") == 0.00002


def test_duration_can_use_last_draw_before_visible() -> None:
    events = [
        {"event": "theta_change.canvas_draw.end", "perf_counter_ns": 100},
        {"event": "theta_change.canvas_draw.end", "perf_counter_ns": 500},
        {"event": "theta_change.after_idle.visible", "perf_counter_ns": 800},
    ]

    assert (
        measure_gui_timing._duration_ms(
            events,
            "theta_change.canvas_draw.end",
            "theta_change.after_idle.visible",
            pairing="last_before_end",
        )
        == 0.0003
    )


def test_timing_automation_dialog_bypass_requires_timing_enabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RA_SIM_TIMING_AUTOMATION", "1")
    monkeypatch.delenv("RA_SIM_TIMING", raising=False)
    monkeypatch.setenv("RA_SIM_TIMING_OUT", str(tmp_path / "timing.jsonl"))

    assert not launcher._should_disable_debug_dialog_for_timing_automation()

    monkeypatch.setenv("RA_SIM_TIMING", "1")

    assert launcher._should_disable_debug_dialog_for_timing_automation()


def test_harness_main_returns_failure_after_writing_summary(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_run_trial(*, output_dir, spec, **_kwargs):
        path = output_dir / f"{spec.artifact_stem}.jsonl"
        path.write_text("", encoding="utf-8")
        return measure_gui_timing.TrialResult(
            path=path,
            returncode=7,
            timed_out=False,
            termination="normal",
            child_pid=123,
            measured=True,
        )

    def fake_write_summary(output_dir, metadata, trial_events, trial_results, **_kwargs):
        calls.append((output_dir, metadata, trial_events, trial_results))

    monkeypatch.setattr(measure_gui_timing, "_now_stamp", lambda: "unit_nonzero")
    monkeypatch.setattr(measure_gui_timing, "_run_trial", fake_run_trial)
    monkeypatch.setattr(measure_gui_timing, "_write_summary", fake_write_summary)

    exit_code = measure_gui_timing.main(
        ["--scenario", "defaults", "--trials", "1", "--output-root", str(tmp_path)]
    )

    assert exit_code == 1
    assert calls


def test_harness_main_returns_failure_after_timeout_summary(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_run_trial(*, output_dir, spec, **_kwargs):
        path = output_dir / f"{spec.artifact_stem}.jsonl"
        path.write_text("", encoding="utf-8")
        return measure_gui_timing.TrialResult(
            path=path,
            returncode=0,
            timed_out=True,
            termination="terminate",
            child_pid=456,
            measured=True,
        )

    def fake_write_summary(output_dir, metadata, trial_events, trial_results, **_kwargs):
        calls.append((output_dir, metadata, trial_events, trial_results))

    monkeypatch.setattr(measure_gui_timing, "_now_stamp", lambda: "unit_timeout")
    monkeypatch.setattr(measure_gui_timing, "_run_trial", fake_run_trial)
    monkeypatch.setattr(measure_gui_timing, "_write_summary", fake_write_summary)

    exit_code = measure_gui_timing.main(
        ["--scenario", "defaults", "--trials", "1", "--output-root", str(tmp_path)]
    )

    assert exit_code == 1
    assert calls


def test_startup_grouping_requires_actual_warmup() -> None:
    five_specs = measure_gui_timing._build_run_specs("defaults", 5, 1)
    six_specs = measure_gui_timing._build_run_specs("defaults", 6, 1)

    assert all(spec.trial_group == "fresh_process" for spec in five_specs)
    assert not any(not spec.measured for spec in five_specs)
    assert [spec.trial_group for spec in six_specs] == [
        "fresh_process",
        "fresh_process",
        "fresh_process",
        "fresh_process",
        "fresh_process",
        "warmup_process",
        "warm_fresh_process",
    ]
    assert not six_specs[5].measured


def test_warmup_failure_stops_warm_measured_trials(tmp_path, monkeypatch) -> None:
    specs = measure_gui_timing._build_run_specs("defaults", 6, 1)
    calls = []

    def fake_run_trial(*, output_dir, spec, **_kwargs):
        calls.append(spec.artifact_stem)
        path = output_dir / f"{spec.artifact_stem}.jsonl"
        path.write_text("", encoding="utf-8")
        failed = spec.artifact_stem == "warmup_001"
        return measure_gui_timing.TrialResult(
            path=path,
            returncode=1 if failed else 0,
            timed_out=False,
            termination="normal",
            child_pid=123,
            measured=spec.measured,
        )

    monkeypatch.setattr(measure_gui_timing, "_run_trial", fake_run_trial)

    results = measure_gui_timing._run_specs(
        output_dir=tmp_path,
        scenario="defaults",
        run_specs=specs,
        state=None,
        timeout_s=1.0,
    )

    assert calls == [
        "trial_001",
        "trial_002",
        "trial_003",
        "trial_004",
        "trial_005",
        "warmup_001",
    ]
    assert not any(result.path.name == "trial_006.jsonl" for result in results)
    assert measure_gui_timing._harness_exit_code(results) == 1


def test_child_runtime_metadata_overrides_parent_backend() -> None:
    metadata = {
        "backend": "not detected",
        "parent_matplotlib_probe": {"backend": "qtagg", "version": "parent"},
    }
    merged = measure_gui_timing._merge_child_runtime_metadata(
        metadata,
        [
            [
                {
                    "event": "gui.runtime.metadata",
                    "matplotlib_backend": "TkAgg",
                    "matplotlib_version": "3.10.3",
                    "tk_version": "8.6",
                    "tcl_version": "8.6",
                    "root_geometry": "1200x760+0+0",
                }
            ]
        ],
    )

    assert merged["backend"] == "TkAgg"
    assert merged["matplotlib"] == {"backend": "TkAgg", "version": "3.10.3"}
    assert merged["tk"] == {"tk_version": "8.6", "tcl_version": "8.6"}
    assert merged["window_geometry"] == "1200x760+0+0"


def test_summary_can_merge_metadata_from_unmeasured_warmup(tmp_path) -> None:
    metadata = {"backend": "not detected"}
    warmup_event = {
        "event": "gui.runtime.metadata",
        "matplotlib_backend": "TkAgg",
        "matplotlib_version": "3.10.3",
        "tk_version": "8.6",
        "tcl_version": "8.6",
        "root_geometry": "1200x760+0+0",
    }

    measure_gui_timing._write_summary(
        tmp_path,
        metadata,
        trial_events=[[]],
        trial_results=[
            measure_gui_timing.TrialResult(
                path=tmp_path / "trial_001.jsonl",
                returncode=0,
                timed_out=False,
                termination="normal",
                child_pid=123,
                measured=True,
            )
        ],
        all_run_events=[[warmup_event], []],
    )

    merged = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert merged["backend"] == "TkAgg"
    assert merged["window_geometry"] == "1200x760+0+0"


def test_summary_writes_unmeasured_warmup_to_combined_csv(tmp_path) -> None:
    warmup_event = {
        "event": "process.launch.start",
        "trial_id": 0,
        "scenario_id": "defaults",
        "trial_group": "warmup_process",
        "perf_counter_ns": 10,
    }
    measured_event = {
        "event": "process.launch.start",
        "trial_id": 6,
        "scenario_id": "defaults",
        "trial_group": "warm_fresh_process",
        "perf_counter_ns": 20,
    }

    measure_gui_timing._write_summary(
        tmp_path,
        {"backend": "not detected", "python": "unit"},
        trial_events=[[measured_event]],
        trial_results=[
            measure_gui_timing.TrialResult(
                path=tmp_path / "warmup_001.jsonl",
                returncode=0,
                timed_out=False,
                termination="normal",
                child_pid=1,
                measured=False,
            ),
            measure_gui_timing.TrialResult(
                path=tmp_path / "trial_006.jsonl",
                returncode=0,
                timed_out=False,
                termination="normal",
                child_pid=2,
                measured=True,
            ),
        ],
        all_run_events=[[warmup_event], [measured_event]],
    )

    combined = (tmp_path / "combined_events.csv").read_text(encoding="utf-8")
    assert "warmup_process" in combined
    assert "warm_fresh_process" in combined


def test_markdown_table_includes_missing_count_and_raw_ms() -> None:
    table = measure_gui_timing._markdown_table(
        {
            "metric": {
                "trial_count": 2,
                "missing_count": 1,
                "median_ms": 2.0,
                "mean_ms": 2.0,
                "min_ms": 1.0,
                "max_ms": 3.0,
                "p95_ms": 3.0,
                "std_ms": 1.414,
                "raw_ms": [1.0, 3.0],
            }
        }
    )

    assert "missing_count" in table
    assert "raw_ms" in table
    assert "1.000, 3.000" in table


def test_saved_state_startup_summary_has_phase_breakdown(tmp_path) -> None:
    summary = _write_timing_summary(tmp_path, _saved_state_trial_events())
    saved_summary = summary["saved_state_startup"]["phase_breakdown"]
    saved_rows = summary["saved_state_startup"]["trial_rows"]

    assert set(saved_summary) == set(measure_gui_timing.SAVED_STATE_BREAKDOWN_ORDER)
    assert saved_summary["saved-state file read"]["phase_status"] == "measured"
    assert saved_summary["PONI parse/restore"]["phase_status"] == "not_applicable"
    assert saved_summary["first simulation compute"]["correlation_status"] == "correlated"
    assert saved_rows[0]["state_path_basename"] == "new4.json"
    assert saved_rows[0]["state_file_size_bytes"] == 1234
    assert saved_rows[0]["restored_background_count"] == 3
    assert saved_rows[0]["first_visible_total_ms"] == 700.0
    assert saved_rows[0]["saved_state_restore_total_ms"] == 250.0
    assert saved_rows[0]["saved_state_unexplained_gap_ms"] is not None


def test_saved_state_startup_timeline_has_non_overlapping_rows(tmp_path) -> None:
    summary = _write_timing_summary(tmp_path, _saved_state_trial_events())
    timeline = summary["saved_state_startup"]["startup_timeline"]["phase_breakdown"]

    assert set(timeline) == set(measure_gui_timing.STARTUP_TIMELINE_ORDER)
    assert all(row["phase_status"] == "measured" for row in timeline.values())
    assert all(row["missing_count"] == 0 for row in timeline.values())


def test_saved_state_startup_timeline_reports_unexplained_gaps(tmp_path) -> None:
    summary = _write_timing_summary(tmp_path, _saved_state_trial_events())
    row = summary["saved_state_startup"]["trial_rows"][0]
    timeline = summary["saved_state_startup"]["startup_timeline"]

    assert row["startup_timeline_measured_total_ms"] == 510.0
    assert row["startup_timeline_unexplained_gap_ms"] == 190.0
    assert row["longest_unexplained_interval_ms"] == 100.0
    assert row["longest_unexplained_interval_before_event"] == "first_simulation.compute.start"
    assert row["longest_unexplained_interval_after_event"] == "first_simulation.compute.end"
    assert timeline["total_unattributed_gap_ms"]["median_ms"] == 190.0


def test_saved_state_startup_timeline_does_not_guess_missing_events(tmp_path) -> None:
    events = [event for event in _saved_state_trial_events() if event["event"] != "python.entry"]

    summary = _write_timing_summary(tmp_path, events)
    timeline = summary["saved_state_startup"]["startup_timeline"]["phase_breakdown"]

    first_row = timeline["process.launch.start -> python.entry"]
    second_row = timeline["python.entry -> gui.command.received"]
    assert first_row["phase_status"] == "not_detected"
    assert first_row["median_ms"] is None
    assert first_row["missing_phase_reason"] == "missing_boundary:python.entry"
    assert second_row["phase_status"] == "not_detected"
    assert second_row["median_ms"] is None


def test_saved_state_startup_timeline_does_not_use_update_request_as_startup_boundary(
    tmp_path,
) -> None:
    events = [
        event
        for event in _saved_state_trial_events()
        if not (event["event"] == "first_update.requested" and event.get("update_id") is None)
    ]

    summary = _write_timing_summary(tmp_path, events)
    timeline = summary["saved_state_startup"]["startup_timeline"]["phase_breakdown"]
    after_restore = timeline["saved_state.end -> first_update.requested"]
    before_update = timeline["first_update.requested -> first_update.update_begin"]

    assert after_restore["phase_status"] == "not_detected"
    assert after_restore["median_ms"] is None
    assert after_restore["missing_phase_reason"] == (
        "boundary_requires_no_update_id:first_update.requested"
    )
    assert before_update["phase_status"] == "not_detected"
    assert before_update["median_ms"] is None
    assert before_update["missing_phase_reason"] == (
        "boundary_requires_no_update_id:first_update.requested"
    )


def test_saved_state_startup_timeline_redacts_paths(tmp_path) -> None:
    events = _saved_state_trial_events()
    for event in events:
        if event["event"] in {"saved_state.start", "saved_state.metadata"}:
            event["state_path_basename"] = "C:/Users/Kenpo/OneDrive/private/new4.json"

    summary = _write_timing_summary(tmp_path, events)
    summary_text = json.dumps(summary, sort_keys=True)
    readme_text = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert "Startup Timeline" in readme_text
    assert summary["saved_state_startup"]["trial_rows"][0]["state_path_basename"] == "new4.json"
    for forbidden in ("C:/", "Users", "Kenpo", "OneDrive", "private"):
        assert forbidden not in summary_text
        assert forbidden not in readme_text


def test_saved_state_startup_missing_phase_is_reported_not_guessed(tmp_path) -> None:
    events = [
        event
        for event in _saved_state_trial_events()
        if not str(event["event"]).startswith("saved_state.background_load")
    ]

    summary = _write_timing_summary(tmp_path, events)
    background = summary["saved_state_startup"]["phase_breakdown"]["background load"]

    assert background["phase_status"] == "not_detected"
    assert background["median_ms"] is None
    assert "missing_span:saved_state.background_load.start" in background["missing_phase_reason"]


def test_saved_state_startup_paths_are_redacted(tmp_path) -> None:
    events = _saved_state_trial_events()
    for event in events:
        if event["event"] in {"saved_state.start", "saved_state.metadata"}:
            event["state_path_basename"] = "C:/Users/Kenpo/OneDrive/private/new4.json"

    summary = _write_timing_summary(tmp_path, events)
    summary_text = json.dumps(summary, sort_keys=True)
    readme_text = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert summary["saved_state_startup"]["trial_rows"][0]["state_path_basename"] == "new4.json"
    for forbidden in ("C:/", "Users", "Kenpo", "OneDrive", "private"):
        assert forbidden not in summary_text
        assert forbidden not in readme_text


def test_saved_state_startup_does_not_change_status_semantics(tmp_path) -> None:
    metadata = {
        "backend": "unit",
        "python": "unit",
        "scenario": "defaults-restored",
        "trials": 1,
        "state_file": "new4.json",
    }
    measure_gui_timing._write_summary(
        tmp_path,
        metadata,
        trial_events=[_saved_state_trial_events()],
        trial_results=[
            measure_gui_timing.TrialResult(
                path=tmp_path / "trial_001.jsonl",
                returncode=7,
                timed_out=False,
                termination="normal",
                child_pid=1,
                measured=True,
            )
        ],
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["trial_failures"][0]["returncode"] == 7


def test_timing_disabled_does_not_emit_saved_state_events(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "timing.jsonl"
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"type": "ra_sim.gui_state", "version": 1, "state": {}}),
        encoding="utf-8",
    )
    monkeypatch.delenv("RA_SIM_TIMING", raising=False)
    monkeypatch.setenv("RA_SIM_TIMING_OUT", str(output_path))

    payload = load_gui_state_file(state_path)

    assert payload["state"] == {}
    assert not output_path.exists()


def test_timing_disabled_emits_no_startup_timeline_events(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "timing.jsonl"
    monkeypatch.delenv("RA_SIM_TIMING", raising=False)
    monkeypatch.setenv("RA_SIM_TIMING_OUT", str(output_path))

    timing.timing_event("first_update.update_begin", update_id=1)

    assert not output_path.exists()


def test_saved_state_startup_update_id_mismatch_is_uncorrelated(tmp_path) -> None:
    summary = _write_timing_summary(tmp_path, _saved_state_trial_events(uncorrelated=True))
    saved_summary = summary["saved_state_startup"]["phase_breakdown"]
    startup_timeline = summary["saved_state_startup"]["startup_timeline"]["phase_breakdown"]

    assert saved_summary["draw_event to after_idle visible"]["phase_status"] == "not_detected"
    assert saved_summary["draw_event to after_idle visible"]["correlation_status"] == (
        "uncorrelated"
    )
    assert (
        saved_summary["draw_event to after_idle visible"]["missing_phase_reason"]
        == "uncorrelated_update_id"
    )
    assert (
        startup_timeline["first_update.update_begin -> first_simulation.compute.start"][
            "phase_status"
        ]
        == "uncorrelated"
    )


def test_saved_state_startup_timeline_mixed_bad_trials_surface_status(tmp_path) -> None:
    measure_gui_timing._write_summary(
        tmp_path,
        {
            "backend": "unit",
            "python": "unit",
            "scenario": "defaults-restored",
            "trials": 2,
            "state_file": "new4.json",
        },
        trial_events=[
            _saved_state_trial_events(),
            _saved_state_trial_events(uncorrelated=True),
        ],
        trial_results=[
            measure_gui_timing.TrialResult(
                path=tmp_path / "trial_001.jsonl",
                returncode=0,
                timed_out=False,
                termination="normal",
                child_pid=1,
                measured=True,
            ),
            measure_gui_timing.TrialResult(
                path=tmp_path / "trial_002.jsonl",
                returncode=0,
                timed_out=False,
                termination="normal",
                child_pid=2,
                measured=True,
            ),
        ],
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    row = summary["saved_state_startup"]["startup_timeline"]["phase_breakdown"][
        "first_update.update_begin -> first_simulation.compute.start"
    ]

    assert row["phase_status"] == "uncorrelated"
    assert row["trial_count"] == 1
    assert row["missing_count"] == 1


def test_saved_state_startup_overlap_reports_diagnostics(tmp_path) -> None:
    events = _saved_state_trial_events()
    for event in events:
        if event["event"] == "saved_state.json_parse.start":
            event["perf_counter_ns"] = 115_000_000

    summary = _write_timing_summary(tmp_path, events)
    gap = summary["saved_state_startup"]["phase_breakdown"]["unexplained saved-state startup gap"]

    assert gap["phase_status"] == "not_detected"
    assert gap["missing_phase_reason"] == "overlapping_child_spans"


def test_saved_state_startup_partial_aggregate_span_is_not_measured(tmp_path) -> None:
    events = [
        event
        for event in _saved_state_trial_events()
        if not str(event["event"]).startswith("saved_state.manual_geometry_cache_restore")
    ]

    summary = _write_timing_summary(tmp_path, events)
    manual = summary["saved_state_startup"]["phase_breakdown"]["manual geometry restore"]

    assert manual["phase_status"] == "not_detected"
    assert manual["median_ms"] is None
    assert (
        "missing_span:saved_state.manual_geometry_cache_restore.start"
        in (manual["missing_phase_reason"])
    )


def test_saved_state_startup_duplicate_span_does_not_hide_missing_sibling(tmp_path) -> None:
    events = [
        event
        for event in _saved_state_trial_events()
        if not str(event["event"]).startswith("saved_state.manual_geometry_cache_restore")
    ]
    events.extend(
        [
            _timing_test_event("saved_state.geometry_manual_pairs_restore.start", 291_000_000),
            _timing_test_event("saved_state.geometry_manual_pairs_restore.end", 292_000_000),
        ]
    )

    summary = _write_timing_summary(tmp_path, events)
    manual = summary["saved_state_startup"]["phase_breakdown"]["manual geometry restore"]

    assert manual["phase_status"] == "not_detected"
    assert manual["median_ms"] is None
    assert (
        "missing_span:saved_state.manual_geometry_cache_restore.start"
        in (manual["missing_phase_reason"])
    )


def test_saved_state_startup_duplicate_span_extra_start_is_incomplete(tmp_path) -> None:
    events = [
        *_saved_state_trial_events(),
        _timing_test_event("saved_state.cif_restore_or_read.start", 261_000_000),
    ]

    summary = _write_timing_summary(tmp_path, events)
    cif = summary["saved_state_startup"]["phase_breakdown"]["CIF read/restore"]

    assert cif["phase_status"] == "not_detected"
    assert cif["median_ms"] is None
    assert cif["missing_phase_reason"] == (
        "incomplete_span:saved_state.cif_restore_or_read.start->saved_state.cif_restore_or_read.end"
    )


def test_saved_state_startup_duplicate_span_extra_end_is_incomplete(tmp_path) -> None:
    events = [
        *_saved_state_trial_events(),
        _timing_test_event("saved_state.cif_restore_or_read.end", 261_000_000),
    ]

    summary = _write_timing_summary(tmp_path, events)
    cif = summary["saved_state_startup"]["phase_breakdown"]["CIF read/restore"]

    assert cif["phase_status"] == "not_detected"
    assert cif["median_ms"] is None
    assert cif["missing_phase_reason"] == (
        "incomplete_span:saved_state.cif_restore_or_read.start->saved_state.cif_restore_or_read.end"
    )


def test_saved_state_startup_incomplete_not_applicable_span_is_not_detected(tmp_path) -> None:
    events = [
        *_saved_state_trial_events(),
        _timing_test_event("saved_state.poni_restore_or_parse.start", 345_000_000),
    ]

    summary = _write_timing_summary(tmp_path, events)
    poni = summary["saved_state_startup"]["phase_breakdown"]["PONI parse/restore"]

    assert poni["phase_status"] == "not_detected"
    assert (
        poni["missing_phase_reason"]
        == "incomplete_span:saved_state.poni_restore_or_parse.start->saved_state.poni_restore_or_parse.end"
    )


def test_saved_state_correlated_span_extra_start_is_incomplete(tmp_path) -> None:
    events = [
        *_saved_state_trial_events(),
        _timing_test_event("first_simulation.compute.start", 505_000_000, update_id=7),
    ]

    summary = _write_timing_summary(tmp_path, events)
    compute = summary["saved_state_startup"]["phase_breakdown"]["first simulation compute"]
    row = summary["saved_state_startup"]["trial_rows"][0]

    assert compute["phase_status"] == "not_detected"
    assert compute["median_ms"] is None
    assert compute["missing_phase_reason"] == (
        "incomplete_span:first_simulation.compute.start->first_simulation.compute.end"
    )
    assert row["first_simulation_compute_total_ms"] is None


def test_saved_state_correlated_span_extra_end_is_incomplete(tmp_path) -> None:
    events = [
        *_saved_state_trial_events(),
        _timing_test_event("first_gui_render_after_compute.end", 605_000_000, update_id=7),
    ]

    summary = _write_timing_summary(tmp_path, events)
    render = summary["saved_state_startup"]["phase_breakdown"]["post-compute render"]

    assert render["phase_status"] == "not_detected"
    assert render["median_ms"] is None
    assert render["missing_phase_reason"] == (
        "incomplete_span:first_gui_render_after_compute.start"
        "->first_gui_render_after_compute.end"
    )


def test_saved_state_draw_to_visible_uses_last_correlated_draw(tmp_path) -> None:
    events = [
        *_saved_state_trial_events(),
        _timing_test_event("first_canvas.draw_event", 675_000_000, update_id=7),
    ]

    summary = _write_timing_summary(tmp_path, events)
    draw = summary["saved_state_startup"]["phase_breakdown"]["draw_event to after_idle visible"]

    assert draw["phase_status"] == "measured"
    assert draw["median_ms"] == 25.0
    assert draw["correlation_status"] == "correlated"


def test_saved_state_draw_to_visible_rejects_later_uncorrelated_draw(tmp_path) -> None:
    events = sorted(
        [
            *_saved_state_trial_events(),
            _timing_test_event("first_canvas.draw_event", 675_000_000, update_id=8),
        ],
        key=lambda event: int(event["perf_counter_ns"]),
    )

    summary = _write_timing_summary(tmp_path, events)
    draw = summary["saved_state_startup"]["phase_breakdown"]["draw_event to after_idle visible"]

    assert draw["phase_status"] == "not_detected"
    assert draw["median_ms"] is None
    assert draw["missing_phase_reason"] == "uncorrelated_update_id"
    assert draw["correlation_status"] == "uncorrelated"


def test_first_simulation_compute_breakdown_has_required_rows(tmp_path) -> None:
    summary = _write_timing_summary(tmp_path, _saved_state_trial_events())
    breakdown = summary["saved_state_startup"]["first_simulation_compute_breakdown"]
    phases = breakdown["phase_breakdown"]
    row = summary["saved_state_startup"]["trial_rows"][0]
    readme_text = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert set(phases) == set(measure_gui_timing.FIRST_SIMULATION_COMPUTE_ORDER)
    assert phases["beam_sample_generation"]["status"] == "measured"
    assert phases["kernel_call"]["status"] == "measured"
    assert phases["q_or_rod_solving"]["status"] == "not_detected"
    assert phases["worker_wait_or_sync_compute"]["status"] == "not_applicable"
    assert phases["worker_wait_or_sync_compute"]["missing_phase_reason"] == (
        "synchronous_first_compute_has_no_exclusive_worker_wait_span"
    )
    assert phases["normalization"]["status"] == "not_applicable"
    assert row["first_simulation_compute_total_ms"] == 100.0
    assert row["first_simulation_compute_measured_subphase_total_ms"] == 60.0
    assert row["first_simulation_compute_unexplained_gap_ms"] == 40.0
    assert row["first_simulation_compute_slowest_subphase"] == "kernel_call"
    assert row["first_simulation_compute_slowest_subphase_ms"] == 40.0
    assert row["first_simulation_update_id"] == 7
    assert breakdown["total_unattributed_gap_ms"]["median_ms"] == 40.0
    assert "First Simulation Compute Breakdown" in readme_text


def test_first_simulation_worker_handoff_reports_sync_boundaries(tmp_path) -> None:
    summary = _write_timing_summary(tmp_path, _saved_state_trial_events())
    handoff = summary["saved_state_startup"]["first_simulation_worker_handoff"]
    phases = handoff["phase_breakdown"]
    row = summary["saved_state_startup"]["trial_rows"][0]
    readme_text = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert phases["worker_queue_wait_ms"]["status"] == "not_applicable"
    assert phases["worker_submit_to_start_ms"]["status"] == "not_applicable"
    assert phases["worker_result_transfer_ms"]["status"] == "not_applicable"
    assert phases["worker_start_to_kernel_start_ms"]["median_ms"] == 30.0
    assert phases["kernel_call_ms"]["median_ms"] == 40.0
    assert phases["kernel_end_to_result_ready_ms"]["median_ms"] == 10.0
    assert phases["result_ready_to_gui_thread_ms"]["median_ms"] == 12.0
    assert phases["gui_thread_result_apply_ms"]["median_ms"] == 4.0
    assert row["worker_start_to_kernel_start_ms"] == 30.0
    assert row["worker_submit_to_start_ms"] is None
    assert "First Simulation Worker Handoff" in readme_text
    assert "broad diagnostic span" in handoff["conclusion"]


def test_first_simulation_worker_handoff_reports_async_boundaries(tmp_path) -> None:
    summary = _write_timing_summary(tmp_path, _saved_state_async_worker_events(queued=True))
    phases = summary["saved_state_startup"]["first_simulation_worker_handoff"]["phase_breakdown"]
    row = summary["saved_state_startup"]["trial_rows"][0]

    assert phases["worker_queue_wait_ms"]["median_ms"] == 10.0
    assert phases["worker_submit_to_start_ms"]["median_ms"] == 10.0
    assert phases["worker_start_to_kernel_start_ms"]["median_ms"] == 30.0
    assert phases["kernel_call_ms"]["median_ms"] == 40.0
    assert phases["kernel_end_to_result_ready_ms"]["median_ms"] == 10.0
    assert phases["result_ready_to_gui_thread_ms"]["median_ms"] == 14.0
    assert phases["gui_thread_result_apply_ms"]["median_ms"] == 3.0
    assert phases["worker_result_transfer_ms"]["median_ms"] == 2.0
    assert row["worker_result_transfer_ms"] == 2.0


def test_first_simulation_worker_handoff_reports_immediate_async_submit(
    tmp_path,
) -> None:
    summary = _write_timing_summary(
        tmp_path,
        _saved_state_async_worker_events(queued=False),
    )
    phases = summary["saved_state_startup"]["first_simulation_worker_handoff"]["phase_breakdown"]
    row = summary["saved_state_startup"]["trial_rows"][0]

    assert phases["worker_queue_wait_ms"]["status"] == "not_applicable"
    assert phases["worker_queue_wait_ms"]["missing_phase_reason"] == (
        "worker_request_was_not_queued"
    )
    assert phases["worker_submit_to_start_ms"]["median_ms"] == 10.0
    assert phases["worker_result_transfer_ms"]["median_ms"] == 2.0
    assert row["worker_queue_wait_ms"] is None
    assert row["worker_submit_to_start_ms"] == 10.0


def test_first_simulation_worker_handoff_requires_update_id_correlation(tmp_path) -> None:
    events = [
        *_saved_state_trial_events(),
        _timing_test_event("first_worker.submit", 390_000_000, update_id=8),
    ]

    summary = _write_timing_summary(tmp_path, events)
    phases = summary["saved_state_startup"]["first_simulation_worker_handoff"]["phase_breakdown"]

    assert phases["worker_submit_to_start_ms"]["status"] == "uncorrelated"


def test_first_simulation_worker_handoff_reverse_events_are_excluded(tmp_path) -> None:
    events = [
        *_saved_state_trial_events(),
        _timing_test_event(
            "simulation_worker.result.fetch.start",
            503_000_000,
            update_id=7,
        ),
        _timing_test_event(
            "simulation_worker.result.fetch.end",
            501_000_000,
            update_id=7,
        ),
    ]
    events = sorted(events, key=lambda event: int(event["perf_counter_ns"]))

    summary = _write_timing_summary(tmp_path, events)
    phases = summary["saved_state_startup"]["first_simulation_worker_handoff"]["phase_breakdown"]
    row = summary["saved_state_startup"]["trial_rows"][0]

    assert phases["worker_result_transfer_ms"]["status"] == "overlap_error"
    assert row["worker_result_transfer_ms"] is None


def test_first_simulation_compute_breakdown_ignores_broad_worker_span(tmp_path) -> None:
    events = [
        *_saved_state_trial_events(),
        _timing_test_event(
            "first_simulation.worker_wait_or_sync_compute.start",
            400_000_000,
            update_id=7,
        ),
        _timing_test_event(
            "first_simulation.worker_wait_or_sync_compute.end",
            500_000_000,
            update_id=7,
        ),
    ]

    summary = _write_timing_summary(tmp_path, events)
    phases = summary["saved_state_startup"]["first_simulation_compute_breakdown"]["phase_breakdown"]
    row = summary["saved_state_startup"]["trial_rows"][0]

    assert phases["worker_wait_or_sync_compute"]["status"] == "not_applicable"
    assert row["first_simulation_compute_measured_subphase_total_ms"] == 60.0
    assert row["first_simulation_compute_unexplained_gap_ms"] == 40.0
    assert row["first_simulation_compute_slowest_subphase"] == "kernel_call"


def test_first_simulation_compute_breakdown_requires_update_id_correlation(tmp_path) -> None:
    summary = _write_timing_summary(tmp_path, _saved_state_trial_events(uncorrelated=True))
    phases = summary["saved_state_startup"]["first_simulation_compute_breakdown"]["phase_breakdown"]

    assert phases["beam_sample_generation"]["status"] == "uncorrelated"
    assert phases["kernel_call"]["status"] == "uncorrelated"
    assert phases["result_ready"]["status"] == "uncorrelated"


def test_first_simulation_compute_breakdown_reports_missing_not_detected(tmp_path) -> None:
    events = [
        event
        for event in _saved_state_trial_events()
        if not str(event["event"]).startswith("first_simulation.kernel_call")
    ]

    summary = _write_timing_summary(tmp_path, events)
    kernel = summary["saved_state_startup"]["first_simulation_compute_breakdown"][
        "phase_breakdown"
    ]["kernel_call"]

    assert kernel["status"] == "not_detected"
    assert kernel["median_ms"] is None
    assert "missing_span:first_simulation.kernel_call.start" in kernel["missing_phase_reason"]


def test_first_simulation_compute_breakdown_extra_start_is_incomplete(tmp_path) -> None:
    events = [
        *_saved_state_trial_events(),
        _timing_test_event(
            "first_simulation.kernel_call.start",
            472_000_000,
            update_id=7,
        ),
    ]

    summary = _write_timing_summary(tmp_path, events)
    kernel = summary["saved_state_startup"]["first_simulation_compute_breakdown"][
        "phase_breakdown"
    ]["kernel_call"]
    row = summary["saved_state_startup"]["trial_rows"][0]

    assert kernel["status"] == "not_detected"
    assert kernel["median_ms"] is None
    assert kernel["missing_phase_reason"] == (
        "incomplete_span:first_simulation.kernel_call.start->first_simulation.kernel_call.end"
    )
    assert row["first_simulation_compute_measured_subphase_total_ms"] == 20.0
    assert row["first_simulation_compute_unexplained_gap_ms"] == 80.0


def test_first_simulation_compute_breakdown_extra_end_is_incomplete(tmp_path) -> None:
    events = [
        *_saved_state_trial_events(),
        _timing_test_event(
            "first_simulation.kernel_call.end",
            472_000_000,
            update_id=7,
        ),
    ]

    summary = _write_timing_summary(tmp_path, events)
    kernel = summary["saved_state_startup"]["first_simulation_compute_breakdown"][
        "phase_breakdown"
    ]["kernel_call"]
    row = summary["saved_state_startup"]["trial_rows"][0]

    assert kernel["status"] == "not_detected"
    assert kernel["median_ms"] is None
    assert kernel["missing_phase_reason"] == (
        "incomplete_span:first_simulation.kernel_call.start->first_simulation.kernel_call.end"
    )
    assert row["first_simulation_compute_measured_subphase_total_ms"] == 20.0
    assert row["first_simulation_compute_unexplained_gap_ms"] == 80.0


def test_first_simulation_compute_breakdown_does_not_guess_parent_span(tmp_path) -> None:
    events = [
        event
        for event in _saved_state_trial_events()
        if not str(event["event"]).startswith("first_simulation.beam_sample_generation")
        and not str(event["event"]).startswith("first_simulation.kernel_call")
        and not str(event["event"]).startswith("first_simulation.result_ready")
    ]

    summary = _write_timing_summary(tmp_path, events)
    phases = summary["saved_state_startup"]["first_simulation_compute_breakdown"]["phase_breakdown"]
    row = summary["saved_state_startup"]["trial_rows"][0]

    assert phases["beam_sample_generation"]["status"] == "not_detected"
    assert phases["kernel_call"]["status"] == "not_detected"
    assert phases["result_ready"]["status"] == "not_detected"
    assert row["first_simulation_compute_total_ms"] == 100.0
    assert row["first_simulation_compute_measured_subphase_total_ms"] == 0.0
    assert row["first_simulation_compute_unexplained_gap_ms"] == 100.0
    assert row["first_simulation_compute_slowest_subphase"] is None


def test_first_simulation_compute_breakdown_overlap_reverse_excluded(tmp_path) -> None:
    events = _saved_state_trial_events()
    for event in events:
        if event["event"] == "first_simulation.kernel_call.end":
            event["perf_counter_ns"] = 425_000_000

    summary = _write_timing_summary(tmp_path, events)
    breakdown = summary["saved_state_startup"]["first_simulation_compute_breakdown"]
    kernel = breakdown["phase_breakdown"]["kernel_call"]
    row = summary["saved_state_startup"]["trial_rows"][0]

    assert kernel["status"] == "overlap_error"
    assert row["first_simulation_compute_measured_subphase_total_ms"] == 20.0
    assert row["first_simulation_compute_unexplained_gap_ms"] == 80.0


def test_first_simulation_compute_breakdown_redacts_paths(tmp_path) -> None:
    events = _saved_state_trial_events()
    for event in events:
        if event["event"] in {"saved_state.start", "saved_state.metadata"}:
            event["state_path_basename"] = "C:/Users/Kenpo/OneDrive/private/new4.json"

    summary = _write_timing_summary(tmp_path, events)
    summary_text = json.dumps(summary, sort_keys=True)
    readme_text = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert "First Simulation Compute Breakdown" in readme_text
    for forbidden in ("C:/", "Users", "Kenpo", "OneDrive", "private"):
        assert forbidden not in summary_text
        assert forbidden not in readme_text


def test_timing_disabled_emits_no_compute_breakdown_events(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "timing.jsonl"
    monkeypatch.delenv("RA_SIM_TIMING", raising=False)
    monkeypatch.setenv("RA_SIM_TIMING_OUT", str(output_path))

    timing.timing_event("simulation_worker.result.fetch.start", update_id=1)
    timing.timing_event("simulation_worker.result.fetch.end", update_id=1)
    with timing.timing_span("first_simulation.kernel_call", update_id=1):
        pass

    assert not output_path.exists()


def test_runtime_timing_fields_call_path_does_not_retry_type_errors() -> None:
    source = (measure_gui_timing.REPO_ROOT / "ra_sim/gui/_runtime/runtime_session.py").read_text(
        encoding="utf-8"
    )

    assert "_call_simulation_with_optional_timing_fields" not in source
    assert "_simulation_callable_accepts_timing_fields" not in source
