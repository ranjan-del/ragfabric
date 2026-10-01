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


def test_a_password_containing_an_at_sign_is_fully_masked():
    from ragfabric_cli.ui.console import mask_url

    assert mask_url("postgresql://u:p@ss@db/x") == "postgresql://u:***@db/x"


def test_every_url_in_a_message_is_masked():
    from ragfabric_cli.ui.console import mask_urls_in

    out = mask_urls_in("a postgresql://u:one@h/d and b redis://x:two@r:6379/0 end")
    assert "one" not in out and "two" not in out
    assert "postgresql://u:***@h/d" in out and "redis://x:***@r:6379/0" in out
