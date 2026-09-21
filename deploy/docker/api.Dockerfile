# deploy/docker/api.Dockerfile
# RagFabric API image. Multi stage: resolve the uv workspace, then copy only the venv.
# python:3.13-slim as of 2026-09-21
FROM python@sha256:8d9d0b8bcf6506481eae4907c18f5e3e7902e629f5f6d684f9e7c32e85e3ddf0 AS builder
# ghcr.io/astral-sh/uv:0.12.13 as of 2026-09-21
COPY --from=ghcr.io/astral-sh/uv@sha256:b485bd65cc2cf1c9a93b3554012c9c3778cf7b1b5fd3d3096ce9e1226c97e1e6 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./
COPY packages/core/pyproject.toml packages/core/README.md packages/core/
COPY packages/server/pyproject.toml packages/server/README.md packages/server/
COPY packages/cli/pyproject.toml packages/cli/README.md packages/cli/
RUN uv sync --frozen --no-dev --no-install-workspace
COPY packages/ packages/
RUN uv sync --frozen --no-dev --all-packages

# python:3.13-slim as of 2026-09-21
FROM python@sha256:8d9d0b8bcf6506481eae4907c18f5e3e7902e629f5f6d684f9e7c32e85e3ddf0
RUN useradd --create-home --uid 10001 ragfabric
WORKDIR /app
COPY --from=builder --chown=ragfabric:ragfabric /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
RUN mkdir -p /app/data/uploads && chown -R ragfabric:ragfabric /app/data
USER ragfabric
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=12 CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"
CMD ["sh", "-c", "ragfabric db upgrade && ragfabric serve --host 0.0.0.0 --port 8000"]
