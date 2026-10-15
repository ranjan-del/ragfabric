# Render

> Status: written against Render's Blueprint documentation. The project has not deployed RagFabric
> to Render, so treat the first deployment as a trial and check each step.

`render.yaml` at the repository root is a Render Blueprint. It describes the API, a PostgreSQL
database, a Key Value store (Redis compatible) and the web app, and Render creates them together.

| Blueprint entry | What it is |
|---|---|
| `ragfabric-api` | The API from `deploy/docker/api.Dockerfile`, with a 5 GB disk at `/app/data/uploads` for uploaded originals. Applies migrations on every start |
| `ragfabric-db` | PostgreSQL. pgvector is available on Render's PostgreSQL; migration 0003 enables it |
| `ragfabric-cache` | Key Value, so rate limits are shared across instances |
| `ragfabric-ui` | The web app as a static site, with `/api/*` rewritten to the API |

Render runs no model server, so `deploy/render/ragfabric.yaml` selects OpenAI for answers and
embeddings (`text-embedding-3-small`, asked for 768 dimensions to match migration 0004). Any
OpenAI compatible endpoint works through `base_url`.

## Steps

1. Fork or push the repository to an account Render can read.
2. In Render: Blueprints, New Blueprint Instance, pick the repository. Render reads `render.yaml`
   and lists what it will create.
3. Render asks for the values marked `sync: false`: `FIRST_ADMIN_EMAIL`, `FIRST_ADMIN_PASSWORD`
   (not `admin@example.com`; production refuses it) and `OPENAI_API_KEY`. `JWT_SECRET` is
   generated for you.
4. Before the first deploy finishes, open the `ragfabric-api` service, Environment, Secret Files,
   and add a file named `ragfabric.yaml` with the contents of `deploy/render/ragfabric.yaml`
   (edited as you like). The service reads it at `/etc/secrets/ragfabric.yaml`. Then redeploy.
5. When the API is live, check `https://<api>.onrender.com/health`.
6. If Render gave the API a different address from `ragfabric-api.onrender.com`, edit the `/api/*`
   rewrite on `ragfabric-ui` (in `render.yaml`, or in the static site's Redirects and Rewrites) to
   point at it.
7. Open the static site's address, sign in, upload, ask.

## Notes

- `DATABASE_URL` arrives as `postgresql://...`; RagFabric maps it to its psycopg driver at
  startup, so it can be used as given.
- **Client addresses.** The sign-in limit counts per client address. Render's proxy appends the
  client's address to any `X-Forwarded-For` the client sent, so trusting every hop (`*`) would let
  a client choose its own address and dodge the limit. The Blueprint sets
  `FORWARDED_ALLOW_IPS=10.0.0.0/8`: uvicorn then walks the header from the right, skips addresses
  in that range, and takes the first one outside it, which is the one Render's edge added. That
  range is an assumption about Render's internal network that the project has not verified. After
  deploying, sign in once and check that the access log's `client` field shows your own public
  address; if it shows a private address, widen or correct the range, and if it shows a different
  public address, remove the setting.
- Logs are JSON lines in the service's Logs tab. Each response's `X-Request-ID` finds its lines
  (see [Operations](../operations.md)).
- A disk ties the service to one instance. Scaling out needs the uploads moved off local disk,
  which RagFabric does not yet support. Alternatively set `ingestion.retain_originals: false` and
  drop the disk, at the cost of the download link.
- Plans and prices are Render's; check them before creating anything. The plan names in
  `render.yaml` are the smallest that support each feature at the time of writing.
