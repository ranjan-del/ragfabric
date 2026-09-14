import time

from ragfabric_core.strategies.base import TraceSpan
from ragfabric_core.telemetry.tracing import configure_otel, otel_enabled, start_trace, trace


def test_spans_are_recorded_with_relative_offsets_and_attributes():
    with start_trace() as ctx:
        with trace("parse", fmt="pdf"):
            time.sleep(0.005)
            with trace("chunk", count=3):
                pass
    names = [s.name for s in ctx.spans]
    assert names == ["chunk", "parse"]  # inner span closes first
    parse = next(s for s in ctx.spans if s.name == "parse")
    assert (
        isinstance(parse, TraceSpan)
        and parse.attributes == {"fmt": "pdf"}
        and parse.duration_ms >= 5
    )
    assert all(s.started_ms >= 0 for s in ctx.spans)


def test_trace_outside_a_context_is_a_no_op():
    with trace("orphan"):
        pass


def test_nested_traces_do_not_leak_between_contexts():
    with start_trace() as a:
        with trace("x"):
            pass
    with start_trace() as b:
        pass
    assert [s.name for s in a.spans] == ["x"] and b.spans == []


def test_otel_is_off_by_default_and_on_with_an_endpoint():
    assert configure_otel(None) is False and otel_enabled() is False
    assert configure_otel("http://localhost:4318") is True and otel_enabled() is True
    with start_trace() as ctx:
        with trace("exported", k=1):
            pass
    assert ctx.spans[0].name == "exported"
    configure_otel(None)
