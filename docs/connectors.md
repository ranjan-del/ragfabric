# Connectors

> Status: shipped in Phase 10. The watched folder connector is tested end to end, including a real
> run in the production compose stack. The Google Drive connector is tested against the official
> client library with fake HTTP; the project has not run it against a real Google account, so the
> setup steps below follow Google's documentation and are not yet confirmed by a real run.

A connector keeps a RagFabric collection in step with a place where documents already live. It
lists the files there, ingests what is new, re-ingests what changed (same document id, version
bumped, so existing citations stay valid), skips what did not change without reading it, and
deletes documents whose file went away.

Two kinds exist:

| Kind | Source | Extra needed |
|---|---|---|
| `folder` | A directory on the machine or a mounted volume (a dropped-file inbox, a network share) | none |
| `google_drive` | One Google Drive folder, on My Drive or a shared drive | `pip install 'ragfabric[drive]'` |

## Running them

Connectors are configured under `connectors:` in `ragfabric.yaml` and run from the command line,
in-process against the database, like `ragfabric worker`. There is no HTTP endpoint for them.

```bash
ragfabric connectors list                 # what is configured
ragfabric connectors sync                 # one pass over every connector, then exit
ragfabric connectors sync inbox           # one pass over one connector
ragfabric connectors sync --watch         # keep going, each connector every interval_seconds
```

One pass prints a line per connector:

```text
inbox: 1 added, 0 updated, 3 unchanged, 0 deleted, 1 skipped, 0 failed, 0 pending
  scans/receipt.pdf: File is named .pdf but is not a PDF (no %PDF- header).
```

| Count | Meaning |
|---|---|
| added | ingested for the first time |
| updated | the bytes changed; re-ingested in place |
| unchanged | size, modified time and source version all match; nothing was read |
| deleted | the file left the source and its document was deleted (`on_delete: delete`) |
| skipped | refused by the upload rules (wrong content for its extension, too large, a zip bomb). Not read again until the file changes |
| failed | could not be fetched, or ingestion failed (for example the embedding server was down); tried again next pass, re-ingesting the same document |
| pending | still being written (folder) and left for the next pass |
| ignored | passed over as not a document (a Drive image, form or shortcut); shown only when non-zero |

Without `--watch` the command exits 1 if any connector failed or any file was skipped, so a cron
job or CI step notices. With `--watch` it runs until Ctrl-C or SIGTERM, printing only passes that
changed something. In the production compose stack the `connectors` profile runs it as a service
(see [the single VM guide](deploy/single-vm.md)).

## Configuration

```yaml
connectors:
  - name: inbox                 # keys the sync state; keep it stable
    kind: folder
    path: /data/inbox
    collection: handbook        # created if missing; omit for no collection
    owner: admin@example.com    # optional; the user must exist
    recursive: true
    interval_seconds: 30
    settle_seconds: 5
    on_delete: delete           # delete | keep
  - name: shared-policies
    kind: google_drive
    folder_id: 1AbCdEfGhIjKlMnOp
    credentials_file: /run/secrets/drive-service-account.json
    collection: policies
    interval_seconds: 300       # at least 30
```

| Key | Kinds | Default | Meaning |
|---|---|---|---|
| `name` | both | required | Lower case letters, digits, `-` and `_`. Unique. It keys the connector's state, so **renaming a connector makes every file look new** and they are ingested again as new documents |
| `kind` | both | required | `folder` or `google_drive` |
| `collection` | both | none | Collection the documents go into, created if missing |
| `owner` | both | none | Email of an existing user who owns the documents |
| `recursive` | both | `true` | Include subfolders |
| `on_delete` | both | `delete` | `delete` removes the document when its file leaves the source; `keep` leaves it |
| `interval_seconds` | both | 30 (folder), 300 (Drive) | Time between passes under `--watch` |
| `path` | folder | required | The directory to watch |
| `settle_seconds` | folder | 5 | A file modified more recently than this is still being written and waits |
| `folder_id` | Drive | required | The folder's id, the last part of its URL |
| `credentials_file` | Drive | required | Path to a service account key file. The key itself never goes in `ragfabric.yaml` |

### What a connector will and will not delete

- Only documents **that connector** created. An uploaded document, or one from another connector,
  is never touched.
- Nothing at all when a pass lists no files while documents are recorded. An unmounted volume or an
  unshared Drive folder looks exactly like "every file was removed", so the pass deletes nothing
  and prints a warning instead. If the source really is empty, delete the documents by hand.
- Nothing when listing fails part way: the pass stops before any deletion.
- A file still being written (folder) is neither ingested nor deleted that pass.

### The upload rules apply

Every file goes through the same validation as an upload (`ingest/validate.py`): a known, allowed
extension, at most `limits.max_upload_mb`, content that matches the extension, and zip-based
formats within `limits.max_uncompressed_mb`. When the source states a size (every folder file,
every non-Google Drive file), a file over the limit is refused before it is downloaded.

## Watched folder

The folder is polled, not watched for events: filesystem events do not cross Docker bind mounts
from a macOS or Windows host, nor network shares, which is where a dropped-file inbox usually
lives. A scan is cheap.

- Supported formats only: PDF, DOCX, PPTX, TXT, MD, CSV. Other files are left alone silently.
- Hidden files and folders (`.name`), Office lock files (`~$name`) and download or editor
  temporaries (`.tmp`, `.part`, `.crdownload`, `.partial`, `.swp`) are ignored.
- A symlink is followed only if it resolves inside the folder. One pointing elsewhere (`/etc`, a
  home directory) is ignored, so the inbox cannot be used to ingest files the operator never put
  there.
- A file is ingested only once it has not been modified for `settle_seconds`, and, under
  `--watch`, its size and modified time are the same as at the previous pass, so a file that is
  still being copied is not ingested half-written, even by a copy tool that sets the source's old
  modified time before it finishes.
- The source id is the path relative to the folder. Moving a file within the folder therefore
  deletes the old document and adds a new one.

## Google Drive

The connector reads one folder with a Google Cloud **service account**, which runs unattended in
a container with no browser. You share the folder with the service account's address, so you
decide exactly what it can read. It asks only for the read-only scope
(`https://www.googleapis.com/auth/drive.readonly`), so a leaked key cannot change or delete
anything.

### Setting it up

1. Install the extra: `pip install 'ragfabric[drive]'` (or `uv sync` in a checkout, whose dev group
   includes it). Without it the connector stops with a message naming that command.
2. In the [Google Cloud console](https://console.cloud.google.com/), pick or create a project and
   enable the **Google Drive API** for it (APIs and services, Library).
3. Create a service account (IAM and admin, Service accounts). It needs no project roles.
4. On the service account, open Keys, add a key of type JSON and download it. This file is a
   secret: store it outside the repository with permissions `600`, or as a Docker secret, and
   never paste it into `ragfabric.yaml`, an issue or a chat.
5. In Google Drive, share the folder with the service account's email address
   (`name@project.iam.gserviceaccount.com`) as a **Viewer**. For a shared drive, add the service
   account as a member of the shared drive, or share the folder within it.
6. Copy the folder id from its URL: `https://drive.google.com/drive/folders/<folder id>`.
7. Add the connector to `ragfabric.yaml` with `folder_id` and `credentials_file`, then run
   `ragfabric connectors sync <name>` once and read the report.

Some Google Workspace organisations forbid sharing with addresses outside the domain or forbid
service account keys; an administrator then has to allow it.

### What is read, and how

| In the folder | Becomes | Notes |
|---|---|---|
| PDF, DOCX, PPTX, TXT, MD, CSV files | the file as it is | version is Drive's `md5Checksum` |
| Google Docs | `<name>.docx` | exported as DOCX |
| Google Slides | `<name>.pptx` | exported as PPTX |
| Google Sheets | `<name>.csv` | exported as CSV: **the first sheet only**, a Drive export limit |
| Subfolders | listed too, when `recursive` | each visited once, however many times it is reached |
| Shortcuts, forms, drawings, images, archives, anything else | ignored | counted as `ignored` in the report |
| Files in the trash | not listed | treated as removed |

A file whose name lacks an extension but whose type is supported (a plain text file named
`notes`) gets the extension added.

Limits to know:

- Drive refuses to export a Google Doc, Slides or Sheet whose export is larger than **10 MB**. Such
  a file is counted as failed with Drive's error, every pass, until it is smaller.
- A Google native file has no size or checksum, so its Drive version number stands in. A change
  that does not alter the exported bytes is detected by the SHA-256 and does not create a new
  document version.
- Each pass lists the whole folder tree (one request per 1,000 files per folder) and downloads only
  what changed. At the default 300 second interval this stays far inside Drive's default quota.
- Changes are found by polling. Drive push notifications need a public HTTPS endpoint and are not
  used.

### When it fails

| Message | Cause |
|---|---|
| `needs the drive extra: pip install 'ragfabric[drive]'` | the extra is not installed |
| `credentials file ... does not exist` | wrong `credentials_file` path, or the secret is not mounted |
| `is not a readable service account key` | the file is not JSON, is an OAuth client or user credential rather than a service account key, or is missing fields. The key's content is never printed |
| `listed nothing while N source(s) are recorded` | the folder is empty or no longer shared with the service account; nothing was deleted |
| `folder ... was not found or not shared with the service account (HTTP 404)` | wrong `folder_id`, or the folder is not shared with the service account. Checked before every listing, so a first sync does not quietly report an empty folder |
| `id ... is not a folder` | `folder_id` names a file |
| `HttpError 403 ... has not been used in project` | the Drive API is not enabled for the key's project |
