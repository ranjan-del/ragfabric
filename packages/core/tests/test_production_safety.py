"""Tests for the production safety rails in ragfabric_core.config.

Every default in config.py is chosen so the service runs offline with no setup.
Three of them are dangerous the moment the service is reachable: a signing
secret printed in the source, a bootstrap admin whose email and password are
printed beside it, and create_all standing in for migrations.

These rails only fire when ENVIRONMENT=production, so nothing else in the suite
ever executes them. Code that runs exactly once, during a deploy, is precisely
the code that turns out not to work.
"""

import pytest

from ragfabric_core.config import (
    DEFAULT_ADMIN_EMAIL,
    DEFAULT_ADMIN_PASSWORD,
    MIN_SECRET_LENGTH,
    Settings,
)

REAL_SECRET = "s" * MIN_SECRET_LENGTH


def production(**overrides) -> Settings:
    """Build Settings as if deploying, ignoring any ambient .env file."""
    values = {"environment": "production", "jwt_secret": REAL_SECRET}
    values.update(overrides)
    return Settings(_env_file=None, **values)


class TestSigningSecret:
    @pytest.mark.parametrize(
        "placeholder", ["change-me", "change-me-for-production", "secret", ""]
    )
    def test_refuses_to_boot_on_a_placeholder_secret(self, placeholder):
        with pytest.raises(RuntimeError, match="JWT_SECRET"):
            production(jwt_secret=placeholder)

    def test_refuses_a_secret_that_is_merely_short(self):
        with pytest.raises(RuntimeError, match=str(MIN_SECRET_LENGTH)):
            production(jwt_secret="short-but-not-a-placeholder")

    def test_accepts_a_real_secret(self):
        assert production().jwt_secret == REAL_SECRET

    def test_the_placeholder_is_fine_outside_production(self):
        """The default exists so a local run needs no setup at all."""
        settings = Settings(_env_file=None, environment="development", jwt_secret="change-me")
        assert settings.jwt_secret == "change-me"


class TestBootstrapAdmin:
    def test_the_shipped_admin_credentials_are_refused(self):
        """admin@example.com / adminpass123 are published in this repository."""
        settings = production(
            first_admin_email=DEFAULT_ADMIN_EMAIL,
            first_admin_password=DEFAULT_ADMIN_PASSWORD,
        )
        assert settings.first_admin_email is None
        assert settings.first_admin_password is None

    def test_the_default_password_is_refused_even_under_a_different_email(self):
        settings = production(
            first_admin_email="ops@company.example",
            first_admin_password=DEFAULT_ADMIN_PASSWORD,
        )
        assert settings.first_admin_email is None

    def test_a_genuinely_configured_admin_is_kept(self):
        settings = production(
            first_admin_email="ops@company.example",
            first_admin_password="a-real-and-private-password",
        )
        assert settings.first_admin_email == "ops@company.example"
        assert settings.first_admin_password == "a-real-and-private-password"

    def test_the_defaults_survive_outside_production(self):
        settings = Settings(
            _env_file=None,
            environment="development",
            first_admin_email=DEFAULT_ADMIN_EMAIL,
            first_admin_password=DEFAULT_ADMIN_PASSWORD,
        )
        assert settings.first_admin_email == DEFAULT_ADMIN_EMAIL


class TestSchemaOwnership:
    def test_create_all_is_disabled_so_alembic_owns_the_schema(self):
        assert production(auto_create_tables=True).auto_create_tables is False

    def test_create_all_stays_on_in_development(self):
        settings = Settings(_env_file=None, environment="development", auto_create_tables=True)
        assert settings.auto_create_tables is True


def test_environment_matching_is_forgiving_about_case_and_spacing():
    with pytest.raises(RuntimeError):
        Settings(_env_file=None, environment="  Production ", jwt_secret="change-me")
