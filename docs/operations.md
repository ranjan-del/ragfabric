# Operations

> Status: Phase 10. Describes behaviour that is tested in the suite; the commands for backups and
> upgrades follow the single VM stack and were run in [the first run](learning/hardening-first-run.md)
> only where that page says so.

What an operator needs once RagFabric is running: reading logs, following one request, what the
rate limits and upload rules do, and how to back up and upgrade.

## Logs

`logging.format` in `ragfabric.yaml` (or `RAGFABRIC_LOG_FORMAT`, which wins) selects:

| Format | For | Looks like |
|---|---|---|
| `text` (default) | a person at a terminal | `2026-10-12 09:30:01,207 INFO ragfabric.access [3f9c...] POST /api/ask 200 812ms` |
| `json` | a log shipper; the production compose file and the Render Blueprint set it | one JSON object per line |

`logging.level` (or `RAGFABRIC_LOG_LEVEL`) sets the level: `DEBUG`, `INFO`, `WARNING`, `ERROR`,
`CRITICAL`.

A JSON line always has `ts` (UTC), `level`, `logger` and `msg`; `request_id` when the line belongs
to a request, a worker job (`job:<id>`) or a connector pass (`connector:<name>`); `trace_id` and
`span_id` when an OpenTelemetry span is active; `exc` with the traceback for an error; and any
structured fields the line carries. Every library RagFabric uses logs through the same handler, so
SQLAlchemy, httpx and the provider SDKs come out in the same format with the same id.

Each request writes one access line from the logger `ragfabric.access` with `method`, `path`
(never the query string, which can carry tokens), `status`, `duration_ms` and `client`. uvicorn's
own access log is off.

## Following one request

Every response carries `X-Request-ID`. A client or proxy may send its own (1 to 128 characters of
letters, digits, `.`, `_`, `-`); anything else is replaced by a fresh 32 character id. When a user
reports a problem, ask for that header (the web app's network tab shows it), then:

```bash
docker compose -f deploy/compose/docker-compose.prod.yml logs api --no-log-prefix \
  | grep '"request_id": "3f9c1b0e..."'
```

An unhandled error returns `{"detail": "Internal server error", "request_id": "..."}`; the
traceback is logged once under that id, and never sent to the client.

## Rate limits

Fixed one-minute windows, counted in the cache (`cache.kind`):

| Who | Limit | Key |
|---|---|---|
| An API key | the key's own `rate_limit_per_minute` (new keys take `limits.rate_limit_per_minute`, 60) | the key |
| A signed-in user | `limits.user_rate_limit_per_minute` (120) | the user |
| Sign-in and registration attempts | `limits.auth_attempts_per_minute` (10) | the client address |

A request is charged once however many times it identifies its caller. Over the limit the answer
is 429 with `{"detail": "Rate limit exceeded"}`, `Retry-After` (seconds to the next window),
`X-RateLimit-Limit` and `X-RateLimit-Remaining`.

Trade-offs to know:

- **With `cache.kind: memory`, each process counts on its own.** Two API processes double every
  limit. Use Redis for anything with more than one process.
- **If Redis is unreachable, requests are allowed**, and each one logs an ERROR with its request
  id. Rate limiting protects capacity, not data, and a Redis outage should not become an API
  outage. Password guessing stays slow regardless, because bcrypt is slow by design. Alert on those
  ERROR lines.
- **A fixed window allows up to twice the limit across a window edge** (the end of one minute and
  the start of the next). It protects capacity; it is not a precise quota.
- **The sign-in limit is only as good as the client address.** The address comes from the
  connection, or from `X-Forwarded-For` only when the peer is trusted through uvicorn's
  `FORWARDED_ALLOW_IPS`. Trust only your own proxy, and never publish the API's port while
  trusting `*`.

## Upload validation

The API, `ragfabric ingest` and every connector apply the same rules
(`ragfabric_core/ingest/validate.py`):

| Check | Refused with |
|---|---|
| Body larger than `limits.max_upload_mb` plus 1 MB of form overhead, counted on the wire before the file is read | 413 |
| Unknown extension, or one not in `limits.allowed_types` | 400 |
| Empty file | 400 |
| Larger than `limits.max_upload_mb` | 413 |
| Content not what the extension claims: a PDF without `%PDF-`, a DOCX or PPTX that is not a zip with the right part inside, a text file with NUL bytes or a binary signature | 415 |
| A DOCX or PPTX over 10,000 entries, over `limits.max_uncompressed_mb` uncompressed, or with an entry compressed more than 100 to 1 (read from the zip's directory, nothing decompressed) | 413 |

RagFabric does not scan uploads for malware. Where that matters, put a scanner in front (for
example ClamAV through a proxy that checks uploads before forwarding) or scan the uploads volume.

## Backups

Two things hold state: the database and the uploaded originals. Redis holds only counters and
caches and needs no backup.

```bash
# database (custom format, restorable with pg_restore)
docker compose -f deploy/compose/docker-compose.prod.yml exec -T postgres \
  pg_dump -U ragfabric -d ragfabric -Fc > ragfabric-$(date +%F).dump

# uploaded originals
docker run --rm -v ragfabric_uploads:/data -v "$PWD":/backup alpine \
  tar czf /backup/uploads-$(date +%F).tar.gz -C /data .
```

The volume name is `<project>_uploads`; the project is `ragfabric` unless you pass `-p`. Keep
backups off the machine, and test a restore:

```bash
docker compose -f deploy/compose/docker-compose.prod.yml exec -T postgres \
  pg_restore -U ragfabric -d ragfabric --clean --if-exists < ragfabric-2026-10-12.dump
```

## Upgrades

1. Read the CHANGELOG for the target release.
2. Back up (above).
3. Set `RAGFABRIC_VERSION` in `.env` to the new release, then:

   ```bash
   docker compose -f deploy/compose/docker-compose.prod.yml --env-file deploy/compose/.env pull
   docker compose -f deploy/compose/docker-compose.prod.yml --env-file deploy/compose/.env up -d
   ```

   From a source checkout: `git pull`, then `up -d --build`.

The API runs `ragfabric db upgrade` before it serves, so migrations apply on start. Downgrading a
release does not downgrade the database; restore the backup instead.

## Startup warnings

At startup the API logs a warning while development defaults are in use (the shipped admin
password, a placeholder `JWT_SECRET`), and in production when it refused the shipped admin and
seeded none. The password itself is never logged.
