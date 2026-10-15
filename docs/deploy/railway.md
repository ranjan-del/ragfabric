# Railway

> Status: written against Railway's documentation. The project has not deployed RagFabric to
> Railway, and there is no Railway template file. Treat the first deployment as a trial.

Railway builds the API from the repository's Dockerfile and provides PostgreSQL and Redis as
services in the same project, on a private network.

## Steps

1. Create a project and add **PostgreSQL**. RagFabric needs the pgvector extension: use Railway's
   pgvector PostgreSQL template, or check that `CREATE EXTENSION vector` works in the database's
   query tab before going further. Migration 0003 fails without it.
2. Add **Redis**.
3. Add a service from the GitHub repository. In its settings set the Dockerfile path to
   `deploy/docker/api.Dockerfile` (Railway reads `RAILWAY_DOCKERFILE_PATH` as a variable for this).
4. Variables on the API service:

   | Variable | Value |
   |---|---|
   | `ENVIRONMENT` | `production` |
   | `RAGFABRIC_LOG_FORMAT` | `json` |
   | `JWT_SECRET` | 48 or more random characters |
   | `DATABASE_URL` | a reference to the PostgreSQL service's `DATABASE_URL` (mapped to the psycopg driver at startup) |
   | `REDIS_URL` | a reference to the Redis service's `REDIS_URL` |
   | `FIRST_ADMIN_EMAIL`, `FIRST_ADMIN_PASSWORD` | your first admin, not the shipped defaults |
   | `OPENAI_API_KEY` | if the config selects OpenAI |
   | `FORWARDED_ALLOW_IPS` | the private range Railway's proxy connects from (see the note below), never `*` |
   | `RAGFABRIC_CONFIG` | `/app/data/ragfabric.yaml` (see step 5) |
   | `PORT` | `8000` |

5. Configuration: attach a volume to the API service at `/app/data`. It holds uploaded originals
   (`/app/data/uploads`) and the config file. Put `deploy/render/ragfabric.yaml` (it suits Railway
   too: a hosted provider, Redis, JSON logs) at `/app/data/ragfabric.yaml`, for example with
   `railway ssh` and a heredoc, then restart the service.
6. Generate a public domain for the API service and check `https://<domain>/health`.
7. For the web app, follow [Static web app](static-ui.md) with this API as the target.

## Notes

- **Client addresses.** Railway's proxy appends the client's address to any `X-Forwarded-For` the
  client sent, so `*` would let a client choose its own address and dodge the sign-in limit. Trust
  only the proxy's private range, so uvicorn takes the right-most address outside it. The project
  has not verified Railway's range: start with `10.0.0.0/8,100.64.0.0/10`, sign in, and check that
  the access log's `client` field shows your own public address; correct the range if not.
- **Volume ownership.** The image runs as a non-root user (uid 10001), and a Railway volume is
  mounted owned by root, so writes to `/app/data` can fail with "permission denied". Railway's
  volume documentation, at the time of writing, gives the service variable `RAILWAY_RUN_UID=0` for
  this, which runs the container as root (check the current page);
  the alternative is a start command that changes the owner of `/app/data` first. The same check
  applies to a Render disk: upload a document once and read the log.

- Railway's proxy terminates HTTPS. Only the API service should have a public domain; PostgreSQL
  and Redis stay on the private network.
- One volume ties the service to one replica, as on Render.
