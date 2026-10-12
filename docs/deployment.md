# Deployment

> Status: Phase 10. The single VM stack has been started and exercised end to end on a workstation
> (see [the first run](learning/hardening-first-run.md)); the project has not run it on a public
> server. The Render and Railway guides are written against those providers' documentation and
> have not been deployed by the project. The project itself runs no public instance.

How to run RagFabric for other people: on one machine behind HTTPS, on a platform that runs
containers, or with the web app on a static host.

## Choosing a route

| Route | What you get | Guide |
|---|---|---|
| One Linux machine with Docker | Everything, including a local model with the `ollama` profile and a watched folder with the `connectors` profile. Caddy provides HTTPS | [Single VM](deploy/single-vm.md) |
| Render | The API, PostgreSQL, a Key Value store and the web app from one Blueprint file. No local model: answers and embeddings come from a hosted provider | [Render](deploy/render.md) |
| Railway | The same pieces assembled by hand in Railway's dashboard | [Railway](deploy/railway.md) |
| A static host for the web app | The web app served from a CDN, talking to an API you run elsewhere | [Static web app](deploy/static-ui.md) |

Whatever the route, the API needs PostgreSQL with the pgvector extension, and should have Redis so
rate limits are shared between processes. Without Redis (`cache.kind: memory`) each process counts
on its own.

## Before anything is reachable

These hold for every route, and the production setting enforces the first three:

- `ENVIRONMENT=production`. It refuses a placeholder or short `JWT_SECRET` (the process will not
  start), refuses the shipped `admin@example.com` bootstrap admin, and turns off table creation
  outside migrations.
- A `JWT_SECRET` of at least 32 random characters, kept out of the repository.
- Your own `FIRST_ADMIN_EMAIL` and `FIRST_ADMIN_PASSWORD`, or an admin created with
  `ragfabric users create --role admin`.
- HTTPS in front of the API. Tokens travel in headers; without TLS anyone on the path can read them.
- Only the proxy reachable from outside. The database, Redis and the API's own port stay on a
  private network.

## Operating it

Logs, request ids, rate limits, upload validation, backups and upgrades are in
[Operations](operations.md). Document sources that sync on their own are in
[Connectors](connectors.md).

## Not covered

Kubernetes manifests and a Helm chart: no user has asked, and an untested chart is worse than none.
Antivirus scanning of uploads: an operator option, described in [Operations](operations.md).
