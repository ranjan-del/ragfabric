"""Application configuration loaded from environment variables.

Uses pydantic-settings so values can come from the environment or a `.env`
file (see `.env.example`). Field names are matched case-insensitively, so the
environment variable ``DATABASE_URL`` populates the ``database_url`` field.

Design note: every default is chosen so the service runs offline with zero
setup. ``database_url`` defaults to a local SQLite file and the embedding /
answer stack is fully deterministic and in-process, so no API key, no Postgres,
and no vector-DB server are required to run or test the app.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from ragfabric_core import __version__

# Placeholder secrets that ship in source control. None of them may ever sign a
# real token: anyone who has read the repository could mint an admin token.
INSECURE_SECRETS = {"change-me", "change-me-for-production", "secret", ""}

# Minimum acceptable length for a production signing secret.
MIN_SECRET_LENGTH = 32

# The bootstrap admin credentials that ship as defaults. They are in the README
# and in this file, so seeding them on a reachable deployment hands an admin
# account to every reader of the repository.
DEFAULT_ADMIN_EMAIL = "admin@example.com"
DEFAULT_ADMIN_PASSWORD = "adminpass123"


class Settings(BaseSettings):
    """Runtime settings with offline-first defaults."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "ragfabric"
    version: str = __version__

    # "development" (default) or "production". Production turns on the safety
    # rails in model_post_init below.
    environment: str = "development"

    # Database. Local default = SQLite (no install needed). Docker/hosting can
    # override with a PostgreSQL URL via the DATABASE_URL env var.
    database_url: str = "sqlite:///./rag.db"

    # Create missing tables from the models at startup. Convenient locally and
    # in the tests, wrong anywhere real: create_all only ever ADDS tables, so a
    # changed column is skipped in silence and the app then queries a schema
    # the database does not have. Alembic owns the schema instead
    # (`alembic upgrade head`), and production forces this off below.
    auto_create_tables: bool = True

    # Embeddings (offline hashing embedder). Larger dim = finer buckets.
    embedding_dim: int = 512

    # Retrieval defaults.
    default_top_k: int = 5
    max_context_chars: int = 4000
    # Hybrid fusion weight: blend of semantic vs. lexical score (0..1).
    hybrid_alpha: float = 0.6

    # CORS: Angular dev-server origins allowed to call this API.
    cors_origins: str = "http://localhost:4200,http://localhost:80"

    # Auth (JWT). Override JWT_SECRET in production.
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    # Optional bootstrap admin, seeded on startup when both are set.
    first_admin_email: str | None = "admin@example.com"
    first_admin_password: str | None = "adminpass123"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.strip().lower() == "production"

    def model_post_init(self, __context) -> None:
        """Apply production safety rails as soon as settings load.

        Every default above is chosen so `git clone` and one command works.
        Three of them stop being conveniences the moment the service has a
        public URL: a signing secret printed in this file, an admin account
        whose password is printed next to it, and create_all standing in for
        migrations. The secret is a refusal to boot, because a forgeable token
        is a total auth bypass and silently correcting it would hide the
        misconfiguration. The other two are forced off.
        """
        if not self.is_production:
            return

        if self.jwt_secret.strip() in INSECURE_SECRETS:
            raise RuntimeError(
                "JWT_SECRET is still a placeholder value while ENVIRONMENT=production. "
                "Set a real secret, e.g. "
                'python -c "import secrets; print(secrets.token_urlsafe(64))"'
            )
        if len(self.jwt_secret) < MIN_SECRET_LENGTH:
            raise RuntimeError(
                f"JWT_SECRET must be at least {MIN_SECRET_LENGTH} characters in production "
                f"(got {len(self.jwt_secret)})."
            )

        # Refuse the shipped admin credentials rather than quietly seeding
        # them. A deployment that wants a bootstrap admin can still have one,
        # it just cannot have THIS one.
        if (
            self.first_admin_email == DEFAULT_ADMIN_EMAIL
            or self.first_admin_password == DEFAULT_ADMIN_PASSWORD
        ):
            self.first_admin_email = None
            self.first_admin_password = None

        self.auto_create_tables = False


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


settings = get_settings()
