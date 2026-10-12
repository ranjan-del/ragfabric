# The web app on a static host

> Status: written from the providers' documentation for each host below. The project has not hosted
> the web app on any of them.

The web app is a static bundle (HTML, CSS, JavaScript). It calls the API at `/api/...` **on its own
origin**: there is no setting for an API address. So a static host can serve it only if it can
**proxy** (rewrite) `/api/*` to the API. A host that can only serve files is not enough, and CORS
does not help, because the app never calls another origin.

Build it once:

```bash
cd apps/assistant
npm ci
npm run build          # output: apps/assistant/dist/assistant/browser
```

Every route the app handles in the browser must fall back to `index.html`.

| Host | Proxies `/api/*`? | How |
|---|---|---|
| Firebase Hosting | Only to Cloud Run or Cloud Functions in the same project | `rewrites` with `"run": {"serviceId": ...}` for `/api/**`, then `"**"` to `/index.html`. An API elsewhere cannot be the target |
| Netlify | Yes, to any URL | `_redirects`: `/api/*  https://api.example.com/api/:splat  200`, then `/*  /index.html  200` |
| Cloudflare Pages | Not with `_redirects` (same-site only) | A Pages Function under `functions/api/[[path]].js` that forwards the request |
| Render static site | Yes | A `rewrite` route, as `render.yaml` does |
| Vercel | Yes | `rewrites` in `vercel.json`: `/api/:path*` to the API, then `/(.*)` to `/index.html` |
| GitHub Pages, plain object storage | No | Use another host, or the single VM stack |

Whichever host: the API must be on HTTPS, and the proxy must pass `Authorization` and stream
`text/event-stream` responses (the streamed answer) without buffering them to the end. Test a
streamed question after setting it up.

The sign-in limit counts per client address. Behind a host's proxy the API sees the proxy's
address unless it trusts the proxy's `X-Forwarded-For` (`FORWARDED_ALLOW_IPS`). Most hosts append
to a header the client sent, so trust only the proxy's own address range, never `*`; see
[Operations](../operations.md).

A runtime setting for the API address, so any static host works without a proxy, is planned once
the new web app lands.
