# What the first run of the production stack looked like, learning notes

> Status: Phase 10. A record of **one** run of `deploy/compose/docker-compose.prod.yml` on a
> workstation, written down because the offline suite cannot tell you this. It includes what went
> wrong.

## This is a record, not a benchmark

Per ADR 0004, everything below is what happened once, on 2026-10-12, on one Apple M1 laptop with
16 GB of memory, under OrbStack 2.2.4 with Docker Compose 5.5.0. Models: `llama3.2:3b` for answers
and `nomic-embed-text` for embeddings, served by Ollama 0.34.0 running on the host. The corpus was
two short text files. Times are wall clock for single requests and say nothing about throughput or
about any other machine. Nothing here is a measurement of answer quality.

## Setup

The stack ran as compose project `rf-p10`, with Caddy on host ports 18080 and 18443 so it could not
collide with anything else on the machine, `DOMAIN=localhost` (Caddy's own local certificate
authority, so clients skipped verification), and images built from the branch with `--build`.
`.env` was generated with random secrets and a non-default first admin. The config was
`ragfabric.prod.yaml` with both `base_url` values pointed at `host.docker.internal`, plus an
override adding `extra_hosts` to the API and connectors services, as
[the single VM guide](../deploy/single-vm.md) describes.

## What worked

| Check | Result |
|---|---|
| Stack comes up | Postgres and Redis healthy, migrations 0001 to 0009 applied on the API's first start, API healthy, then Caddy started. Only Caddy published ports |
| First admin signs in over HTTPS | 200, through Caddy at `https://localhost:18443` |
| Upload | A text file: 201, `ready`, one chunk, about a second including the embedding call |
| Cited answer | `strategy: traditional`: "According to the passage [1], an employee is entitled to twenty five days of paid annual leave per calendar year.", citation [1] to the uploaded file, about 2 seconds. `vectorless` gave the same answer |
| Streamed answer through Caddy | 200; `retrieval`, 23 `token` events, `superseded`, `citations`, `done`. Caddy passed the stream through |
| A binary renamed `.pdf` | 415, "File is named .pdf but is not a PDF (no %PDF- header)." |
| A 52 MB upload with `Content-Length` | 413 from the size middleware before the route ran |
| A 60 MB chunked upload with no `Content-Length` | 413 in about 0.1 seconds: the body was cut off at the limit, not read to the end |
| Sign-in attempts | With `auth_attempts_per_minute: 10`, the successful sign-in plus nine wrong passwords used the window; the next attempt got 429 with `Retry-After: 27` and `X-RateLimit-Limit: 10` |
| Client address behind Caddy | Access lines showed the Docker gateway address (where the host's traffic enters the VM), not Caddy's container address, so `X-Forwarded-For` from Caddy was honoured. Health checks from inside the container showed `127.0.0.1` |
| JSON logs | Every line from the running API was one JSON object. The `X-Request-ID` of the cited answer found four lines, all carrying that id: the embeddings call and two chat calls to Ollama (logged by the HTTP client library) and the access line |
| No secret in the logs | `grep -i` for password and secret over the whole API log found nothing |
| Folder connector (`connectors` profile) | A text file and a fake PDF dropped into the inbox: `inbox: 1 added, ... 1 skipped`, the PDF refused with its reason. The document landed in the `inbox` collection and a question about it was answered from it. Deleting the file: the next pass reported `1 deleted`, and the document was gone from the API |
| Web app | `/` and a client route (`/login`) both served the app through Caddy, with `Strict-Transport-Security` set |
| Docs site | `mkdocs build --strict` clean |

## What went wrong

1. **The migration lines were not JSON.** The container runs `ragfabric db upgrade` before
   `ragfabric serve`. `db upgrade` never configured logging, so Alembic's own console format went
   into a stream that was otherwise JSON: eleven plain lines at the top of a fresh start, which
   a log shipper parsing JSON would reject or mangle. Fixed on this branch: when the format is JSON
   (from `ragfabric.yaml` or `RAGFABRIC_LOG_FORMAT`), `db upgrade` and `db downgrade` now set up
   JSON logging before migrating, with a test. Under the text format Alembic keeps its own output. After a rebuild the Alembic lines were JSON. One plain line remains, the command's own
   `database upgraded to head`, which is printed output rather than a log line.
2. **`auto` sent "How many days of annual leave does an employee get?" to the agentic strategy, and
   the 3B model then answered "I could not find an answer to that in the documents provided."**
   The router's signals treat "how many" as an aggregation marker, which routes to agentic, so the
   decision was by rule (`router.source: signals`), not by the model. The agent with `llama3.2:3b`
   failed on a question the traditional strategy answered correctly in 2 seconds; the agentic run
   took about 19 seconds. Whether "how many" should mean aggregation for a question about a single
   entitlement is a routing question for the Phase 8 evaluation, not something to tune from one
   run.
3. **The streamed answer was superseded.** For "How many unused leave days carry over?" the
   streamed text broke the citation contract and the server sent `superseded` with a repaired
   answer. That is the designed behaviour for a small model, and the client has to handle it.
4. **One answer contradicted itself.** Asked "Are receipts required for expense claims?" through
   `traditional`, the model wrote "Receipts are required for expense claims exceeding €50 [1].
   There is no mention of receipts being required for expense claims in the provided passages."
   The first sentence is cited and correct (the source says fifty euros); the second is
   unsupported and was not cited. The citation contract checks what is cited, not what a model
   adds uncited.
5. **The plain HTTP redirect assumes port 443.** `http://localhost:18080/` redirected to
   `https://localhost/`, without the port. On a real server Caddy listens on 80 and 443 and this
   does not arise; on non-default ports, use the HTTPS address directly.
6. **The Docker engine restarted once during the session**, while another workload on the same
   machine was pulling a large image, and stopped every container including this run's test
   database. Nothing in RagFabric caused it, but it is a reminder that `restart: unless-stopped`
   only covers container exits, not the engine going away; the production compose file relies on
   the host's Docker service starting again on boot.

## Not checked here

A public domain and a real certificate; the `ollama` profile (the host's Ollama was used
instead); the `workers` profile; a Google Drive connector against a real account (no credentials
were available; the Drive code is tested with fake HTTP only); backups and restores; an upgrade
from a published image.

## Torn down

The `rf-p10` stack was removed with its volumes after the run. OrbStack itself was left running.
