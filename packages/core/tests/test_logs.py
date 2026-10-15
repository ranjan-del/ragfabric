"""Structured logging: the JSON formatter, request ids and configure_logging."""

from __future__ import annotations

import io
import json
import logging

import pytest

from ragfabric_core.telemetry import logs
from ragfabric_core.telemetry.logs import (
    JsonFormatter,
    bind_request_id,
    configure_logging,
    current_request_id,
    new_request_id,
    valid_request_id,
)


@pytest.mark.parametrize("value", ["abc-123_x.y", "a", "A" * 128, "0123456789abcdef"])
def test_valid_request_ids_are_accepted(value):
    assert valid_request_id(value) is True


@pytest.mark.parametrize(
    "value",
    ["", "A" * 129, "abc\ndef", 'a"b', "a b", "a{b}", "é", None],
)
def test_unsafe_request_ids_are_refused(value):
    assert valid_request_id(value) is False


def test_new_request_ids_are_32_hex_characters_and_unique():
    first, second = new_request_id(), new_request_id()
    assert len(first) == 32 and int(first, 16) >= 0
    assert first != second


def _record(msg="hello", level=logging.INFO, exc_info=None, **extra):
    logger = logging.getLogger("ragfabric.test")
    record = logger.makeRecord(
        "ragfabric.test", level, __file__, 1, msg, (), exc_info, extra=extra or None
    )
    return record


def test_a_json_line_has_the_core_fields_and_no_request_id_outside_a_request():
    line = JsonFormatter().format(_record())
    data = json.loads(line)
    assert data["level"] == "INFO"
    assert data["logger"] == "ragfabric.test"
    assert data["msg"] == "hello"
    assert data["ts"].endswith("Z")
    assert "request_id" not in data
    assert "\n" not in line


def test_the_request_id_in_context_is_attached():
    with bind_request_id("req-1"):
        assert current_request_id() == "req-1"
        data = json.loads(JsonFormatter().format(_record()))
    assert data["request_id"] == "req-1"
    assert current_request_id() is None


def test_extra_fields_appear_and_unserialisable_values_are_stringified():
    class Odd:
        def __str__(self) -> str:
            return "odd-value"

    data = json.loads(JsonFormatter().format(_record(status=200, thing=Odd())))
    assert data["status"] == 200
    assert data["thing"] == "odd-value"


def test_an_exception_adds_its_traceback():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _record(level=logging.ERROR, exc_info=sys.exc_info())
    data = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in data["exc"]


def test_message_arguments_are_interpolated():
    logger = logging.getLogger("ragfabric.test")
    record = logger.makeRecord("ragfabric.test", logging.INFO, __file__, 1, "a %s c", ("b",), None)
    assert json.loads(JsonFormatter().format(record))["msg"] == "a b c"


@pytest.fixture()
def clean_root():
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    yield root
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
    logs._installed = None


def test_configure_logging_json_installs_one_handler_and_is_idempotent(clean_root, monkeypatch):
    monkeypatch.delenv("RAGFABRIC_LOG_FORMAT", raising=False)
    monkeypatch.delenv("RAGFABRIC_LOG_LEVEL", raising=False)
    stream = io.StringIO()
    configure_logging("json", "INFO", stream=stream)
    configure_logging("json", "INFO", stream=stream)
    ours = [h for h in clean_root.handlers if getattr(h, "_ragfabric", False)]
    assert len(ours) == 1
    logging.getLogger("ragfabric.x").info("one")
    assert json.loads(stream.getvalue().strip())["msg"] == "one"


def test_environment_overrides_the_configured_format_and_level(clean_root, monkeypatch):
    monkeypatch.setenv("RAGFABRIC_LOG_FORMAT", "json")
    monkeypatch.setenv("RAGFABRIC_LOG_LEVEL", "WARNING")
    stream = io.StringIO()
    fmt, level = configure_logging("text", "INFO", stream=stream)
    assert (fmt, level) == ("json", "WARNING")
    logging.getLogger("ragfabric.x").info("hidden")
    logging.getLogger("ragfabric.x").warning("shown")
    lines = stream.getvalue().strip().splitlines()
    assert [json.loads(line)["msg"] for line in lines] == ["shown"]


def test_an_unknown_format_in_the_environment_is_refused(clean_root, monkeypatch):
    monkeypatch.setenv("RAGFABRIC_LOG_FORMAT", "xml")
    with pytest.raises(ValueError, match="RAGFABRIC_LOG_FORMAT"):
        configure_logging("text", "INFO", stream=io.StringIO())


def test_text_format_includes_the_request_id_when_present(clean_root, monkeypatch):
    monkeypatch.delenv("RAGFABRIC_LOG_FORMAT", raising=False)
    monkeypatch.delenv("RAGFABRIC_LOG_LEVEL", raising=False)
    stream = io.StringIO()
    configure_logging("text", "INFO", stream=stream)
    with bind_request_id("abc"):
        logging.getLogger("ragfabric.x").info("inside")
    logging.getLogger("ragfabric.x").info("outside")
    out = stream.getvalue()
    assert "[abc] inside" in out
    assert "outside" in out and "[None]" not in out


def test_uvicorn_loggers_are_adopted_and_its_access_log_silenced(clean_root, monkeypatch):
    monkeypatch.delenv("RAGFABRIC_LOG_FORMAT", raising=False)
    monkeypatch.delenv("RAGFABRIC_LOG_LEVEL", raising=False)
    error = logging.getLogger("uvicorn.error")
    error.addHandler(logging.NullHandler())
    error.propagate = False
    stream = io.StringIO()
    configure_logging("json", "INFO", stream=stream)
    error.info("started")
    logging.getLogger("uvicorn.access").info("GET / 200")
    lines = [json.loads(line) for line in stream.getvalue().strip().splitlines()]
    assert [line["msg"] for line in lines] == ["started"]


def test_the_worker_binds_the_job_id_while_a_job_runs():
    from ragfabric_core.queue.base import Job
    from ragfabric_core.queue.memory_queue import MemoryJobQueue
    from ragfabric_core.workers.runner import Worker

    seen: list[str | None] = []

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def rollback(self):
            pass

    queue = MemoryJobQueue()
    queue.enqueue(Job(id="j-7", kind="probe"))
    worker = Worker(queue, _Session, {"probe": lambda db, job: seen.append(current_request_id())})
    assert worker.run_once(timeout_seconds=0) is True
    assert seen == ["job:j-7"]
    assert current_request_id() is None


def test_running_migrations_in_process_leaves_existing_loggers_and_handlers_alone(
    clean_root, tmp_path, monkeypatch
):
    """Alembic's env.py used to call fileConfig unconditionally, which disables
    every logger that already exists and swaps the root handlers for its own.
    In a process that migrates and then serves (quickstart, the tests), that
    silenced the access log and threw away the JSON formatter."""
    from ragfabric_core.db.migrate import upgrade

    monkeypatch.delenv("RAGFABRIC_LOG_FORMAT", raising=False)
    monkeypatch.delenv("RAGFABRIC_LOG_LEVEL", raising=False)
    existing = logging.getLogger("ragfabric.access")
    stream = io.StringIO()
    configure_logging("json", "INFO", stream=stream)
    ours = [h for h in clean_root.handlers if getattr(h, "_ragfabric", False)]
    upgrade(f"sqlite:///{tmp_path / 'm.db'}")
    assert existing.disabled is False
    assert [h for h in clean_root.handlers if getattr(h, "_ragfabric", False)] == ours
