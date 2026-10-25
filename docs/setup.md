# Setup guide

> How to install RagFabric, connect it to a model, run it, host it for other people and use it. Each
> option carries one of the repository's three states: **Released** (in a tagged release and exercised
> end to end), **Implemented, validation pending** (the code is on `main`, but a live run or a
> deployment is not yet recorded) and **Planned** (not built yet).

## What you need

| Need | For | Notes |
|---|---|---|
| Python 3.13 and [uv](https://docs.astral.sh/uv/) | Everything | The only hard requirement |
| [Ollama](https://ollama.com/) | A free local model | Optional; without a model RagFabric runs in offline mode |
| An API key from OpenAI or Anthropic | A hosted model | Optional |
| Docker with Compose | PostgreSQL, or hosting the full stack | Optional; the quickstart uses SQLite |
| Node 24 | Running the web app from source | Optional |

## 1. Install

```bash
git clone https://github.com/ranjan-del/ragfabric.git
cd ragfabric
uv sync                         # the engine, server, CLI, SDK and the provider clients
uv run ragfabric quickstart     # writes config, sets up SQLite, ingests samples, asks a first question
uv run ragfabric doctor         # checks Python, config, database, migrations, model and embeddings
```

From a checkout, put `uv run` in front of every `ragfabric` command. `pip install ragfabric` installs the
CLI from PyPI, but PyPI currently holds 0.3.1, which predates `quickstart` and `doctor`; use the
checkout until 0.4.0 is published. The quickstart was verified from an empty folder on 25 October 2026.

## 2. Connect a model

The answering model (`llm`) and the embedding model (`embeddings`) are set in `ragfabric.yaml`; keys
come from the environment or `.env`, never from the file. After changing the embedding model, run
`ragfabric reindex --yes` so stored vectors match the new model.

| Option | State | Use it when |
|---|---|---|
| Offline mode | Released | Trying it with nothing installed; answers are extractive quotes |
| Ollama on the same machine | Released | Free, private, no key; the default and the one every benchmark used |
| Ollama on another machine or server | Implemented, validation pending | One model server shared by several installs; run with Ollama in a separate container on one workstation, not yet across machines |
| OpenAI with an API key | Implemented, validation pending | You want a hosted model; tested with fakes, no live run recorded yet |
| Anthropic with an API key | Implemented, validation pending | You want Claude models; tested with fakes, no live run recorded yet |
| Any OpenAI compatible endpoint (vLLM, LM Studio, a hosted gateway) | Planned | Not available yet: `llm.base_url` is read only for Ollama |

### A local model with Ollama

```bash
ollama pull llama3.2:3b         # answering model, about 2 GB
ollama pull nomic-embed-text    # embedding model, 768 dimensions
mv ragfabric.yaml ragfabric.yaml.bak && uv run ragfabric quickstart && uv run ragfabric reindex --yes
```

Quickstart picks Ollama when it is running. Or set it directly:

```yaml
llm:
  provider: ollama
  model: llama3.2:3b             # llama3.1:8b answers better if the machine has 16 GB or more
  base_url: http://localhost:11434/v1   # or http://<model-server>:11434/v1 for Ollama elsewhere
  num_ctx: 8192
embeddings:
  provider: ollama
  model: nomic-embed-text
  base_url: http://localhost:11434/v1
```

### A hosted model with an API key

Put the key in `.env` (or the environment), then point `llm` at the provider:

```bash
echo 'OPENAI_API_KEY=...' >> .env        # or ANTHROPIC_API_KEY=...
```

```yaml
llm:
  provider: openai               # or: anthropic
  model: gpt-5.4-mini            # anthropic default: claude-sonnet-5
embeddings:
  provider: openai               # Anthropic has no embedding model: use openai or ollama here
  model: text-embedding-3-small
  dim: 768                       # must stay 768, the dimension migration 0004 pins
```

`uv run ragfabric doctor` then makes one real call to each provider and names the fix if it fails. A
model with a different embedding dimension needs a migration first; see
[configuration.md](configuration.md#changing-the-embedding-model). The answer quality you can expect
from each kind of model is in [Benchmark results](benchmarks/comparison.md) and the
[model sweep](benchmarks/results/model-sweep.md).

## 3. Run it

| Part | Command | Where |
|---|---|---|
| API | `uv run ragfabric serve` | `http://127.0.0.1:8000`, OpenAPI at `/docs` |
| Web app (from source) | `cd apps/assistant && npm ci && npm start` | `http://localhost:4200`, with `/api` proxied to the API |
| PostgreSQL and Redis instead of SQLite | `docker compose --profile lite up -d postgres redis`, then `uv run ragfabric db upgrade` | `DATABASE_URL` and `REDIS_URL` in `.env` |
| Chroma for vectors | `docker compose --profile full up -d`, and `vector_store.kind: chroma` | |

Sign in with the first admin from `.env` (`FIRST_ADMIN_EMAIL`, `FIRST_ADMIN_PASSWORD`).

## 4. Use it

| Way | How |
|---|---|
| Web app | Upload on Documents, ask on Ask, compare strategies on Compare, open any run on Trace; admins manage users, groups, grants and keys on Console ([assistant.md](assistant.md)) |
| CLI | `ragfabric ingest ./my-docs --collection handbook`, `ragfabric ask "question"`, `ragfabric strategies`, `ragfabric eval run` |
| HTTP API | `POST /api/auth/login` for a token, or an API key in the `X-API-Key` header; `POST /api/ask` with `{"query": "..."}`; the full contract is at `/docs` |
| Python SDK | `from ragfabric_sdk import Client; Client("http://127.0.0.1:8000", api_key="...").ask("question")` |
| TypeScript SDK | `@ragfabric/sdk` in `packages/sdk-typescript` (not yet on npm) |
| Keep a folder or Drive in sync | `ragfabric connectors sync`; the folder connector is verified, Drive is tested with fake HTTP only ([connectors.md](connectors.md)) |

Create an API key for a script with `ragfabric keys create --name my-script --user you@example.com`; the
plaintext is shown once and only its hash is stored.

## 5. Host it for other people

| Route | State | What it is | Guide |
|---|---|---|---|
| One Linux server with Docker Compose and Caddy HTTPS, optionally with a local model (`ollama` profile) and folder sync (`connectors` profile) | Implemented, validation pending | Everything on one machine; run end to end on a workstation, not yet on a public server | [deploy/single-vm.md](deploy/single-vm.md) |
| Render, from the Blueprint in `render.yaml` | Implemented, validation pending | API, PostgreSQL, Key Value and web app; answers come from a hosted model, no local model | [deploy/render.md](deploy/render.md) |
| Railway | Implemented, validation pending | The same pieces assembled in Railway's dashboard | [deploy/railway.md](deploy/railway.md) |
| The web app on a static host | Implemented, validation pending | The web app from a CDN, talking to an API you run elsewhere | [deploy/static-ui.md](deploy/static-ui.md) |
| Kubernetes or a Helm chart | Planned | | |

Before anything is reachable from a network: `ENVIRONMENT=production`, a random `JWT_SECRET` of at
least 32 characters, your own first admin, HTTPS in front of the API, and only the proxy exposed. The
production setting refuses to start without the first three. See [deployment.md](deployment.md) and
[operations.md](operations.md) for logs, rate limits, backups and upgrades.

## When something fails

`uv run ragfabric doctor` names the failing part and the command that fixes it.
[troubleshooting.md](troubleshooting.md) lists the common failures, including a model that is not
pulled, Ollama not running and a missing provider package.
