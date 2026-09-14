"""Lightweight tracing that stores spans with the retrieval run.

Every request opens a trace; every stage opens a span. Spans are collected in
a context variable and written to retrieval_runs.trace, which is what the
Trace page reads. When telemetry.otlp_endpoint is configured, the same spans
are also emitted through the OpenTelemetry SDK to an OTLP HTTP collector, so
adopters can see them in Grafana, Jaeger or Datadog without any extra code.
"""

from __future__ import annotations

import contextvars
import time
from collections.abc import Iterator
from contextlib import contextmanager

from ragfabric_core.strategies.base import TraceSpan

_current: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "ragfabric_trace", default=None
)
_otel_tracer = None


class TraceContext:
    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.spans: list[TraceSpan] = []

    def __enter__(self) -> TraceContext:
        self._token = _current.set(self)
        return self

    def __exit__(self, *exc) -> None:
        _current.reset(self._token)


def start_trace() -> TraceContext:
    return TraceContext()


def otel_enabled() -> bool:
    return _otel_tracer is not None


def configure_otel(endpoint: str | None) -> bool:
    global _otel_tracer
    if not endpoint:
        _otel_tracer = None
        return False
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "ragfabric"}))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces"))
    )
    _otel_tracer = provider.get_tracer("ragfabric")
    return True


@contextmanager
def trace(name: str, **attributes) -> Iterator[None]:
    ctx = _current.get()
    otel_span = (
        _otel_tracer.start_span(name, attributes=attributes) if _otel_tracer is not None else None
    )
    started = time.perf_counter()
    try:
        yield
    finally:
        if otel_span is not None:
            otel_span.end()
        if ctx is not None:
            ctx.spans.append(
                TraceSpan(
                    name=name,
                    started_ms=int((started - ctx.started) * 1000),
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    attributes={
                        k: v
                        for k, v in attributes.items()
                        if isinstance(v, (str, int, float, bool)) or v is None
                    },
                )
            )
