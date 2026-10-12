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
   | `FORWARDED_ALLOW_IPS` | `*`, as the API is reached only through Railway's proxy |
   | `RAGFABRIC_CONFIG` | `/app/data/ragfabric.yaml` (see step 5) |
   | `PORT` | `8000` |

5. Configuration: attach a volume to the API service at `/app/data`. It holds uploaded originals
   (`/app/data/uploads`) and the config file. Put `deploy/render/ragfabric.yaml` (it suits Railway
   too: a hosted provider, Redis, JSON logs) at `/app/data/ragfabric.yaml`, for example with
   `railway ssh` and a heredoc, then restart the service.
6. Generate a public domain for the API service and check `https://<domain>/health`.
7. For the web app, follow [Static web app](static-ui.md) with this API as the target.

## Notes

- Railway's proxy terminates HTTPS. Only the API service should have a public domain; PostgreSQL
  and Redis stay on the private network.
- One volume ties the service to one replica, as on Render.
