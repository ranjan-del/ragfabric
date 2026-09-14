# Access control, concept notes

> Status: Phase 2 (shipped). This is a learning document: why the policy is shaped the way it is, not
> an API reference. For the code, see `packages/core/src/ragfabric_core/auth/`.

## The vocabulary

| Term | What it is |
|---|---|
| Principal | Whoever is making the request: a logged in user (JWT) or an API key, resolved once by `deps.get_principal` |
| Group | A named set of users (`auth/service.py` manages membership) |
| Grant | A group's read (or write) access to a collection (`collection_grants`) |
| Override | A read or deny rule on one document for one user or group, exceptions to what the collection grants already say |
| API key scope | An optional list of collection ids baked into a key, which narrows everything else, never widens it |
| Audit row | One `audit_log` entry per query or search recording who, what strategy, and what was filtered |

## The rules, in order

`auth/policy.compute_access_filter` builds an `AccessFilter` once per request by applying these rules,
in this order, because later rules can only take away access, never grant it back:

1. A principal presenting a missing or inactive API key sees nothing. Checked first: a revoked key
   should not fall through to any of the more permissive rules below it.
2. Admins are unrestricted (`AccessFilter.unrestricted()`).
3. A collection is readable when a grant to one of the principal's groups exists, when the principal
   owns it, **or when it carries no grants at all**. That last clause is the open by default rule.
4. A document is readable when the principal owns it, when a read override names them or one of their
   groups, or when the document belongs to no collection.
5. A deny override for the principal or one of their groups always wins, even over a grant that would
   otherwise allow it.
6. If the API key in use carries collection scopes, everything computed above is intersected with those
   scopes. A key can only narrow what its owner can already see, never extend it.

## Open by default, and how to close a collection

**A collection with no grants is open to every signed in user; the first grant restricts it to its
grantees.** This is deliberate, not an oversight: it preserves v1 behaviour (everyone sees everything)
until an admin actively decides a collection needs restricting, rather than silently locking every new
collection until someone remembers to grant it. To close a collection, create a group, add its members,
and add a `collection_grant` from that group to the collection; from that point on, only members of a
group with a grant (or the owner) can read it. A per-document `deny` override can still exclude one
document from an otherwise open or granted collection without touching the grant itself.

## Filter before rank, not after

The filter is applied **inside** the store query, not on the result list afterward. Given a document
column and a collection column, `stores/access_sql.access_clause` turns an `AccessFilter` into one SQL
predicate:

```python
allow = document_col.in_(allowed_document_ids) | collection_col.in_(allowed_collection_ids)
deny  = ~document_col.in_(denied_document_ids)
WHERE allow AND deny
```

Both `PgVectorStore` and `PostgresLexicalStore` add this predicate to the query itself, before the
database ranks or limits anything. The alternative, fetching the top-k candidates and then discarding
the ones the caller cannot see, is wrong twice: it leaks the *existence* of a chunk into ranking (a
sensitive chunk can push a permitted one out of the top k), and a caller who cannot see enough results
after filtering has no way to get more without exposing the store's raw ranking. Filtering inside the
query means a chunk the caller may not read is never ranked, never enters the model's context, and
never appears in a citation.

## Audit rows

Every call to `/api/search/query` writes a `RetrievalRun`, its `Source` rows, and an
`AuditLog(action="query", sources_returned=..., sources_filtered=...)`; `/semantic` and `/hybrid` write
`AuditLog(action="search")`. `sources_filtered` records how much the access filter actually removed
compared to what an unrestricted query would have returned, which is what makes the "open by default"
rule observable rather than assumed: a collection nobody has restricted should show zero filtering.
`GET /api/runs/{id}` returns a run to its owner or an admin, never to anyone else.

## What Phase 2 does not do yet

- **Row level security in the database.** The filter is enforced in application code (the query
  builder), not by a PostgreSQL `RLS` policy on the tables themselves. A bug in a future store
  implementation that forgets to call `access_clause` would not be caught by the database.
- **OIDC.** Only local users with JWT and hashed API keys exist. External identity providers are a
  later release.
- **Write grants.** `collection_grants.permission` can already be stored as `read` or `write`, but
  `compute_access_filter` does not yet distinguish them: any grant on a collection currently reads as
  read access. Enforcing a real read/write distinction is not done yet.
