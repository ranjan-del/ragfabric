# deploy/docker/api.Dockerfile
# RagFabric API image. Multi stage: resolve the uv workspace, then copy only the venv.
FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./
COPY packages/core/pyproject.toml packages/core/README.md packages/core/
COPY packages/server/pyproject.toml packages/server/README.md packages/server/
COPY packages/cli/pyproject.toml packages/cli/README.md packages/cli/
RUN uv sync --frozen --no-dev --no-install-workspace
COPY packages/ packages/
RUN uv sync --frozen --no-dev --all-packages

FROM python:3.13-slim
RUN useradd --create-home --uid 10001 ragfabric
WORKDIR /app
COPY --from=builder --chown=ragfabric:ragfabric /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER ragfabric
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=12 CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"
CMD ["sh", "-c", "ragfabric db upgrade && ragfabric serve --host 0.0.0.0 --port 8000"]
