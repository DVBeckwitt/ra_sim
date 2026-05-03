from __future__ import annotations

from collections.abc import Mapping, Sequence

from ra_sim.gui._runtime import runtime_session


def test_fresh_rebuild_does_not_pass_duplicate_consumer() -> None:
    calls: list[dict[str, object]] = []

    def _fake_rebuild(
        source_tables: Sequence[object] | None,
        *,
        params_local: Mapping[str, object],
        consumer: str | None = None,
        **kwargs: object,
    ) -> tuple[list[dict[str, object]], list[object], list[object], list[int]]:
        calls.append(
            {
                "source_tables": list(source_tables or ()),
                "params_local": dict(params_local),
                "consumer": consumer,
                "kwargs": dict(kwargs),
            }
        )
        return [], [], [], []

    result = runtime_session._geometry_fit_forward_source_rows_for_rebuild(
        _fake_rebuild,
        [object()],
        params_local={"a": 4.143},
        fallback_consumer="geometry_fit_preflight_cache",
        kwargs={
            "consumer": "geometry_fit_dataset",
            "preflight_mode": "manual_geometry_targeted",
        },
    )

    assert result == ([], [], [], [])
    assert len(calls) == 1
    assert calls[0]["consumer"] == "geometry_fit_dataset"
    assert calls[0]["kwargs"] == {"preflight_mode": "manual_geometry_targeted"}


def test_build_source_rows_wrapper_prefers_explicit_consumer_once() -> None:
    seen_consumers: list[str | None] = []

    def _fake_rebuild(
        _source_tables: Sequence[object] | None,
        *,
        params_local: Mapping[str, object],
        consumer: str | None = None,
        **_kwargs: object,
    ) -> tuple[list[dict[str, object]], list[object], list[object], list[int]]:
        assert params_local == {"a": 4.143}
        seen_consumers.append(consumer)
        return [], [], [], []

    runtime_session._geometry_fit_forward_source_rows_for_rebuild(
        _fake_rebuild,
        [],
        params_local={"a": 4.143},
        fallback_consumer="fallback_consumer",
        kwargs={"consumer": "explicit_consumer"},
    )
    runtime_session._geometry_fit_forward_source_rows_for_rebuild(
        _fake_rebuild,
        [],
        params_local={"a": 4.143},
        fallback_consumer="fallback_consumer",
        kwargs={},
    )

    assert seen_consumers == ["explicit_consumer", "fallback_consumer"]


def test_fresh_rebuild_wrapper_emits_deduped_marker(monkeypatch) -> None:
    traces: list[tuple[str, dict[str, object]]] = []

    def _fake_rebuild(
        _source_tables: Sequence[object] | None,
        *,
        params_local: Mapping[str, object],
        consumer: str | None = None,
        **_kwargs: object,
    ) -> tuple[list[dict[str, object]], list[object], list[object], list[int]]:
        assert consumer == "geometry_fit_dataset"
        assert params_local == {"a": 4.143}
        return [], [], [], []

    monkeypatch.setattr(
        runtime_session,
        "_append_runtime_update_trace",
        lambda event, **fields: traces.append((str(event), dict(fields))),
        raising=False,
    )

    runtime_session._geometry_fit_forward_source_rows_for_rebuild(
        _fake_rebuild,
        [],
        params_local={"a": 4.143},
        fallback_consumer="geometry_fit_preflight_cache",
        kwargs={"consumer": "geometry_fit_dataset"},
    )

    assert traces == [
        (
            "geometry_fit_fresh_rebuild_consumer_wrapper",
            {
                "fresh_rebuild_consumer_wrapper": "deduped",
                "consumer": "geometry_fit_dataset",
            },
        )
    ]
