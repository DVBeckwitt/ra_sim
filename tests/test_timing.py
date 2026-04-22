import json

from ra_sim import launcher, timing
from scripts import measure_gui_timing


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
