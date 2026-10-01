import io

from ragfabric_cli.ui.console import is_rich, mask_url


class Tty(io.StringIO):
    def isatty(self):
        return True


def test_a_pipe_is_never_rich():
    assert is_rich(io.StringIO()) is False


def test_a_terminal_is_rich(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("RAGFABRIC_PLAIN", raising=False)
    assert is_rich(Tty()) is True


def test_no_color_turns_rich_off(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert is_rich(Tty()) is False


def test_ragfabric_plain_turns_rich_off(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("RAGFABRIC_PLAIN", "1")
    assert is_rich(Tty()) is False


def test_a_password_is_masked():
    assert (
        mask_url("postgresql+psycopg://rf:s3cret@db:5432/rf")
        == "postgresql+psycopg://rf:***@db:5432/rf"
    )


def test_a_url_without_a_password_is_unchanged():
    assert mask_url("sqlite:///./ragfabric.db") == "sqlite:///./ragfabric.db"
    assert mask_url("not a url") == "not a url"
