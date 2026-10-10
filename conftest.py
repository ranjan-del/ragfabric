"""Repository-wide test isolation.

``configure_logging`` (called by the server's startup, ``ragfabric serve`` and
``ragfabric worker``) installs a handler on the root logger and changes the
root level. Left in place, one test's logging setup leaks into every later
test, so each test gets the root logger and uvicorn's loggers back as it found
them.
"""

from __future__ import annotations

import logging

import pytest

_ADOPTED = ("uvicorn", "uvicorn.error", "uvicorn.access")


@pytest.fixture(autouse=True)
def _restore_logging():
    root = logging.getLogger()
    saved_root = (list(root.handlers), root.level)
    saved = {
        name: (list(logging.getLogger(name).handlers), logging.getLogger(name).propagate)
        for name in _ADOPTED
    }
    yield
    root.handlers[:] = saved_root[0]
    root.setLevel(saved_root[1])
    for name, (handlers, propagate) in saved.items():
        logger = logging.getLogger(name)
        logger.handlers[:] = handlers
        logger.propagate = propagate
